#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    patterns: tuple[str, ...]


RULES: tuple[Rule, ...] = (
    Rule(
        name="environmental_exposure",
        description=(
            "Environmental exposure or degradation not encoded by the dataset "
            "features, such as fire/high temperature, low temperature, freeze-thaw, "
            "corrosion, seawater, salt spray, or acid rain."
        ),
        patterns=(
            r"\bfire\b",
            r"post-fire",
            r"fire resistance",
            r"fire-resistant performance",
            r"high temperatures?",
            r"high-temperature",
            r"low-temperature",
            r"low temperatures",
            r"low temperature environment",
            r"freeze-thaw",
            r"freezing and thawing",
            r"salt spray",
            r"chloride corrosion",
            r"\bcorrosion\b",
            r"acid rain",
            r"seawater exposure",
            r"after exposure to high temperature",
            r"after high temperature",
            r"exposed to fire",
            r"arctic environment",
            r"offshore region",
            r"water spraying cooling",
        ),
    ),
    Rule(
        name="strengthening_or_repair",
        description=(
            "External strengthening, repair, jackets, or FRP/CFRP confinement "
            "not described by the retained input features."
        ),
        patterns=(
            r"\bcfrp\b",
            r"\bfrp\b",
            r"strengthened",
            r"strengthening",
            r"repaired",
            r"jacketed",
            r"wrapped",
            r"wraps",
            r"textile grid",
            r"wire mesh",
            r"uhpc jackets",
            r"steel jackets",
            r"cfrp-stirrups",
            r"cfrp sheet",
            r"steel strip",
            r"confined by cfrp",
            r"confined with cfrp",
        ),
    ),
    Rule(
        name="defects_or_damage",
        description=(
            "Defects or damage states such as debonding, notch, gap defects, "
            "void defects, imperfections, or local corrosion."
        ),
        patterns=(
            r"debond",
            r"notch",
            r"gap defects?",
            r"void defects?",
            r"initial imperfection",
            r"local corrosion",
            r"localized pitting corrosion",
            r"corrosion pits?",
            r"machining defects?",
            r"defect representing",
            r"with defect",
            r"with defects",
            r"imperfection",
        ),
    ),
    Rule(
        name="load_history_or_rate",
        description=(
            "Non-monotonic or time-history effects such as cyclic loading, "
            "repeated loading, impact, creep, preload, or sustained loading."
        ),
        patterns=(
            r"\bcyclic\b",
            r"repeated loading",
            r"axial impact loading",
            r"\bcreep\b",
            r"pre-load",
            r"preload",
            r"preloading",
            r"loading methods",
            r"sustained loading",
            r"long-term sustained loading",
        ),
    ),
    Rule(
        name="local_or_partial_loading",
        description=(
            "Local or partial loading configuration rather than the usual full-section "
            "axial/eccentric compression setup."
        ),
        patterns=(
            r"local compression",
            r"partial compression",
            r"axially local compression",
            r"local axial compression",
        ),
    ),
    Rule(
        name="special_member_or_joint",
        description=(
            "Clearly special members outside an ordinary CFST column dataset, such as "
            "bridge braces, piles under lateral load, or bolt-welded joints."
        ),
        patterns=(
            r"arch bridge",
            r"transverse brace",
            r"rock-socketed pile",
            r"bolt-welded joint",
            r"bolt-welded joints",
            r"under lateral load",
        ),
    ),
)


MANUAL_KEEP: dict[str, str] = {
    (
        "湖南大学学报(自然科学版) (Journal of Hunan University (Natural Sciences)) | "
        "高强耐火钢方钢管混凝土柱极限承载力研究 (Study on Ultimate Bearing Capacity of "
        "Concrete-filled High-strength Fire-resistant Square Steel Tubular Columns) | "
        "王彦博 (WANG Yanbo), 陈美玲 (CHEN Meiling), 叶泽华 (YE Zehua)... | 2025"
    ): (
        "The title uses fire-resistant steel as a material descriptor. It does not "
        "state fire exposure, post-fire testing, or another non-standard condition."
    ),
}

MANUAL_DROP: dict[str, str] = {}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def find_rule_hits(title: str) -> list[Rule]:
    normalized = normalize_text(title)
    hits: list[Rule] = []
    for rule in RULES:
        if any(re.search(pattern, normalized) for pattern in rule.patterns):
            hits.append(rule)
    return hits


def iter_rows(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            yield row


def build_title_counts(input_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    rows = iter_rows(input_path)
    next(rows)
    for row in rows:
        if row:
            counts[row[0].strip()] += 1
    return counts


def classify_title(title: str) -> dict[str, str]:
    hits = find_rule_hits(title)
    rule_names = "|".join(rule.name for rule in hits)
    rule_descriptions = " | ".join(rule.description for rule in hits)
    first_pass_action = "drop_candidate" if hits else "keep_candidate"

    if title in MANUAL_KEEP:
        return {
            "first_pass_action": first_pass_action,
            "matched_rule_names": rule_names,
            "matched_rule_descriptions": rule_descriptions,
            "manual_override": "keep",
            "final_action": "keep",
            "final_reason": MANUAL_KEEP[title],
        }

    if title in MANUAL_DROP:
        return {
            "first_pass_action": first_pass_action,
            "matched_rule_names": rule_names,
            "matched_rule_descriptions": rule_descriptions,
            "manual_override": "drop",
            "final_action": "drop",
            "final_reason": MANUAL_DROP[title],
        }

    if hits:
        return {
            "first_pass_action": first_pass_action,
            "matched_rule_names": rule_names,
            "matched_rule_descriptions": rule_descriptions,
            "manual_override": "",
            "final_action": "drop",
            "final_reason": rule_descriptions,
        }

    return {
        "first_pass_action": first_pass_action,
        "matched_rule_names": "",
        "matched_rule_descriptions": "",
        "manual_override": "",
        "final_action": "keep",
        "final_reason": "",
    }


def write_title_review(
    output_path: Path,
    title_counts: Counter[str],
    title_decisions: dict[str, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "title",
                "row_count",
                "first_pass_action",
                "matched_rule_names",
                "matched_rule_descriptions",
                "manual_override",
                "final_action",
                "final_reason",
            ]
        )
        for title, row_count in sorted(
            title_counts.items(),
            key=lambda item: (
                title_decisions[item[0]]["final_action"],
                -item[1],
                item[0],
            ),
        ):
            decision = title_decisions[title]
            writer.writerow(
                [
                    title,
                    row_count,
                    decision["first_pass_action"],
                    decision["matched_rule_names"],
                    decision["matched_rule_descriptions"],
                    decision["manual_override"],
                    decision["final_action"],
                    decision["final_reason"],
                ]
            )


def write_filtered_rows(
    input_path: Path,
    kept_output_path: Path,
    removed_output_path: Path,
    title_decisions: dict[str, dict[str, str]],
) -> tuple[int, int]:
    kept_output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_rows(input_path)
    header = next(rows)

    kept_count = 0
    removed_count = 0

    with kept_output_path.open("w", encoding="utf-8-sig", newline="") as kept_handle:
        with removed_output_path.open("w", encoding="utf-8-sig", newline="") as removed_handle:
            kept_writer = csv.writer(kept_handle)
            removed_writer = csv.writer(removed_handle)
            kept_writer.writerow(header)
            removed_writer.writerow(header)

            for row in rows:
                if not row:
                    continue
                title = row[0].strip()
                final_action = title_decisions[title]["final_action"]
                if final_action == "drop":
                    removed_writer.writerow(row)
                    removed_count += 1
                else:
                    kept_writer.writerow(row)
                    kept_count += 1

    return kept_count, removed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows whose paper titles clearly indicate non-standard conditions "
            "for the current CFST dataset."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/all_dedup.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/all_dedup_standard_condition.csv",
        help="Filtered output CSV path.",
    )
    parser.add_argument(
        "--removed-output",
        default="data/raw/all_dedup_removed_nonstandard.csv",
        help="CSV path to store removed rows for audit.",
    )
    parser.add_argument(
        "--review-output",
        default="data/raw/all_dedup_title_review.csv",
        help="CSV path to store title-level first-pass and final review decisions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    removed_output_path = Path(args.removed_output)
    review_output_path = Path(args.review_output)

    title_counts = build_title_counts(input_path)
    title_decisions = {title: classify_title(title) for title in title_counts}

    kept_count, removed_count = write_filtered_rows(
        input_path=input_path,
        kept_output_path=output_path,
        removed_output_path=removed_output_path,
        title_decisions=title_decisions,
    )
    write_title_review(
        output_path=review_output_path,
        title_counts=title_counts,
        title_decisions=title_decisions,
    )

    removed_titles = sum(
        1 for decision in title_decisions.values() if decision["final_action"] == "drop"
    )
    kept_titles = sum(
        1 for decision in title_decisions.values() if decision["final_action"] == "keep"
    )

    print(f"Input rows: {kept_count + removed_count}")
    print(f"Kept rows: {kept_count}")
    print(f"Removed rows: {removed_count}")
    print(f"Kept titles: {kept_titles}")
    print(f"Removed titles: {removed_titles}")
    print(f"Filtered CSV: {output_path}")
    print(f"Removed-row CSV: {removed_output_path}")
    print(f"Title review CSV: {review_output_path}")


if __name__ == "__main__":
    main()
