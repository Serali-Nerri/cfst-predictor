#!/usr/bin/env python3
"""
Run the three configured training experiments and summarize their outputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = [
    REPO_ROOT / "config/experiments/raw_original_metric.yaml",
    REPO_ROOT / "config/experiments/log_transformed_metric.yaml",
    REPO_ROOT / "config/experiments/log_original_metric.yaml",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def read_report(config_path: Path) -> Dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    report_path = REPO_ROOT / output_dir / "evaluation_report.json"
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["config_path"] = str(config_path.relative_to(REPO_ROOT))
    report["output_dir"] = str(output_dir)
    return report


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    test_metrics = report["test_metrics_original_space"]
    cv_results = report["cv_results"]
    optuna_info = report.get("optuna_run_info") or {}
    return {
        "config_path": report["config_path"],
        "output_dir": report["output_dir"],
        "split_strategy_effective": report.get("split_strategy_effective"),
        "optuna_metric_space": report.get("optuna_metric_space"),
        "cv_metric_space": report.get("cv_metric_space"),
        "target_transform": report.get("target_transform", {}),
        "best_score": optuna_info.get("best_score"),
        "best_params": optuna_info.get("best_params"),
        "test_rmse": test_metrics.get("rmse"),
        "test_mae": test_metrics.get("mae"),
        "test_r2": test_metrics.get("r2"),
        "test_cov": test_metrics.get("cov"),
        "cv_rmse": -float(cv_results["mean_cv_score"]),
        "cv_rmse_std": float(cv_results["std_cv_score"]),
    }


def main() -> int:
    summaries: List[Dict[str, Any]] = []
    for config_path in EXPERIMENT_CONFIGS:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "train.py"), "--config", str(config_path)],
            cwd=REPO_ROOT,
            check=True,
        )
        summaries.append(summarize_report(read_report(config_path)))

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / "experiment_suite_summary_20260309.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Saved experiment summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
