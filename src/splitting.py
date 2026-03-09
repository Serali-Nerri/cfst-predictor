"""
Utilities for regression-aware stratified splitting and regime binning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _coerce_auxiliary_specs(raw_specs: Any) -> List[Dict[str, Any]]:
    """Normalize auxiliary stratification feature specs from config."""
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise ValueError("split.auxiliary_features must be a list of mappings")

    normalized: List[Dict[str, Any]] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("Each split.auxiliary_features item must be a mapping")
        column = raw_spec.get("column")
        bins = raw_spec.get("bins", 3)
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Each auxiliary stratification feature requires a column")
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError(
                f"Invalid bins for auxiliary stratification feature '{column}': {bins}"
            )
        normalized.append({"column": column, "bins": bins})

    return normalized


def _quantile_codes(series: pd.Series, n_bins: int) -> Tuple[pd.Series, int]:
    """
    Create stable quantile bin codes for splitting.

    Uses raw-value qcut first. If duplicated quantile edges collapse bins, it falls back
    to rank-based qcut while still preserving the requested ordering.
    """
    series_no_na = cast(pd.Series, series.astype(float))
    unique_count = int(series_no_na.nunique(dropna=True))
    effective_bins = min(max(2, int(n_bins)), unique_count)
    if effective_bins < 2:
        return pd.Series(["bin0"] * len(series_no_na), index=series_no_na.index), 1

    try:
        raw_codes = pd.qcut(
            series_no_na,
            q=effective_bins,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        raw_codes = None

    if raw_codes is not None and not raw_codes.isna().any():
        int_codes = cast(pd.Series, raw_codes.astype(int))
        return int_codes.map(lambda value: f"bin{value}"), int(int_codes.nunique())

    ranked = series_no_na.rank(method="first")
    rank_codes = pd.qcut(
        ranked,
        q=effective_bins,
        labels=False,
        duplicates="drop",
    )
    int_codes = cast(pd.Series, rank_codes.astype(int))
    return int_codes.map(lambda value: f"bin{value}"), int(int_codes.nunique())


def build_regression_stratification_labels(
    features: pd.DataFrame,
    target_raw: pd.Series,
    split_config: Optional[Dict[str, Any]] = None,
    minimum_count: int = 2,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Build robust stratification labels for regression tasks.

    The target is always the primary stratification axis. Optional auxiliary features
    are appended in config order, and are automatically dropped again if the resulting
    strata become too sparse for stable splitting.
    """
    split_config = split_config or {}
    target_bins_requested = split_config.get("target_bins", 10)
    if (
        isinstance(target_bins_requested, bool)
        or not isinstance(target_bins_requested, int)
        or target_bins_requested < 2
    ):
        raise ValueError("split.target_bins must be an integer >= 2")

    auxiliary_specs = _coerce_auxiliary_specs(split_config.get("auxiliary_features"))
    available_auxiliary_specs: List[Dict[str, Any]] = []
    for spec in auxiliary_specs:
        if spec["column"] not in features.columns:
            logger.warning(
                "Auxiliary stratification column '%s' not found; skipping",
                spec["column"],
            )
            continue
        available_auxiliary_specs.append(spec)

    min_required = max(2, int(minimum_count))
    for target_bins in range(target_bins_requested, 1, -1):
        target_component, target_bins_used = _quantile_codes(target_raw, target_bins)
        base_parts = [target_component.map(lambda value: f"target:{value}")]

        for keep_count in range(len(available_auxiliary_specs), -1, -1):
            parts = base_parts.copy()
            used_auxiliary_specs = available_auxiliary_specs[:keep_count]
            auxiliary_bins_used: List[Dict[str, Any]] = []

            for spec in used_auxiliary_specs:
                aux_component, bins_used = _quantile_codes(
                    features[spec["column"]],
                    int(spec["bins"]),
                )
                parts.append(
                    aux_component.map(
                        lambda value, column=spec["column"]: f"{column}:{value}"
                    )
                )
                auxiliary_bins_used.append(
                    {
                        "column": spec["column"],
                        "requested_bins": int(spec["bins"]),
                        "used_bins": int(bins_used),
                    }
                )

            combined = pd.Series(
                [
                    "|".join(values)
                    for values in zip(*(part.astype(str).tolist() for part in parts))
                ],
                index=target_raw.index,
                dtype="object",
            )

            counts = combined.value_counts()
            if counts.empty:
                continue
            min_count_observed = int(counts.min())
            if combined.nunique() > 1 and min_count_observed >= min_required:
                metadata = {
                    "strategy": "regression_stratified",
                    "requested_target_bins": int(target_bins_requested),
                    "used_target_bins": int(target_bins_used),
                    "requested_auxiliary_features": auxiliary_specs,
                    "used_auxiliary_features": auxiliary_bins_used,
                    "minimum_count_required": min_required,
                    "minimum_count_observed": min_count_observed,
                    "n_strata": int(combined.nunique()),
                    "largest_stratum_size": int(counts.max()),
                }
                return combined, metadata

    fallback = pd.Series(["fallback"] * len(target_raw), index=target_raw.index, dtype="object")
    metadata = {
        "strategy": "random",
        "reason": "Unable to build stable regression strata; using fallback label",
        "minimum_count_required": min_required,
        "n_strata": 1,
    }
    logger.warning("Falling back to a single stratum label; random split will be used")
    return fallback, metadata


def required_stratum_count(
    test_size: float,
    validation_size: float,
    n_splits: int,
    configured_minimum: Optional[int] = None,
) -> int:
    """
    Estimate a safe minimum stratum size for train/test, inner validation, and CV.
    """
    required = [2, int(n_splits)]
    if test_size > 0:
        required.append(int(math.ceil(1.0 / test_size)))
    if validation_size > 0 and n_splits > 1:
        effective_train_fraction = 1.0 - (1.0 / n_splits)
        required.append(
            int(math.ceil(1.0 / max(validation_size * effective_train_fraction, 1e-9)))
        )
    if configured_minimum is not None:
        required.append(int(configured_minimum))
    return max(required)


def build_regime_labels(
    values: pd.Series,
    n_bins: int,
    prefix: str,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Build human-readable quantile regime labels for reporting.
    """
    series = cast(pd.Series, values.astype(float))
    unique_count = int(series.nunique(dropna=True))
    effective_bins = min(max(2, int(n_bins)), unique_count)
    if effective_bins < 2:
        label = f"{prefix}_all"
        return (
            pd.Series([label] * len(series), index=series.index, dtype="object"),
            [{"label": label, "lower": float(series.min()), "upper": float(series.max())}],
        )

    categories, _ = pd.qcut(
        series,
        q=effective_bins,
        duplicates="drop",
        retbins=True,
    )
    categories = cast(pd.Series, categories)
    labels: Dict[Any, str] = {}
    regime_ranges: List[Dict[str, Any]] = []
    category_index = categories.cat.categories
    for idx, interval in enumerate(category_index):
        lower = float(interval.left)
        upper = float(interval.right)
        label = f"{prefix}_q{idx + 1}"
        labels[interval] = label
        regime_ranges.append(
            {
                "label": label,
                "lower": lower,
                "upper": upper,
            }
        )

    mapped = categories.map(labels).astype("object")
    return cast(pd.Series, mapped), regime_ranges


def get_split_strategy(split_config: Optional[Dict[str, Any]]) -> str:
    """Read the configured split strategy."""
    strategy = str((split_config or {}).get("strategy", "random")).strip().lower()
    if strategy not in {"random", "regression_stratified"}:
        raise ValueError(
            "split.strategy must be either 'random' or 'regression_stratified'"
        )
    return strategy
