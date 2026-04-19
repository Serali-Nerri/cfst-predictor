"""
Shared CFST target-space helpers.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


REPORT_TARGET_COLUMN_DEFAULT = "Nexp (kN)"
TARGET_MODE_RAW = "raw"
TARGET_MODE_PSI_OVER_NPL = "psi_over_npl"
VALID_TARGET_MODES = {TARGET_MODE_RAW, TARGET_MODE_PSI_OVER_NPL}

NPL_COLUMN = "Npl (kN)"
PSI_COLUMN = "psi"


def normalize_target_mode(target_mode: Optional[str]) -> str:
    """Normalize the configured training target mode."""
    normalized = (target_mode or TARGET_MODE_RAW).strip().lower()
    if normalized not in VALID_TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode '{target_mode}'. "
            f"Expected one of {sorted(VALID_TARGET_MODES)}."
        )
    return normalized


def get_training_target_name(
    report_target_column: str,
    target_mode: Optional[str],
) -> str:
    """Return the modeled target name for the configured target mode."""
    normalized_mode = normalize_target_mode(target_mode)
    if normalized_mode == TARGET_MODE_PSI_OVER_NPL:
        return PSI_COLUMN
    return report_target_column


def apply_target_transform(
    values: Union[pd.Series, np.ndarray],
    target_transform_type: Optional[str],
) -> pd.Series:
    """Apply the configured target transform in model space."""
    series = pd.Series(np.asarray(values, dtype=float).reshape(-1))
    if target_transform_type == "log":
        if (~np.isfinite(series)).any() or (series <= 0).any():
            raise ValueError("log target transform requires all target values to be finite and > 0")
        return pd.Series(np.log(series), index=series.index)
    if target_transform_type == "sqrt":
        if (~np.isfinite(series)).any() or (series < 0).any():
            raise ValueError("sqrt target transform requires all target values to be finite and >= 0")
        return pd.Series(np.sqrt(series), index=series.index)
    return series


def inverse_target_transform(
    values: Union[pd.Series, np.ndarray],
    target_transform_type: Optional[str],
) -> np.ndarray:
    """Invert the configured target transform."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if target_transform_type == "log":
        return np.exp(array)
    if target_transform_type == "sqrt":
        return np.square(array)
    return array


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def compute_training_target(
    df: pd.DataFrame,
    *,
    report_target_column: str = REPORT_TARGET_COLUMN_DEFAULT,
    target_mode: Optional[str] = None,
) -> pd.Series:
    """Read the modeled target in its untransformed training space."""
    normalized_mode = normalize_target_mode(target_mode)
    if normalized_mode == TARGET_MODE_RAW:
        _require_columns(df, [report_target_column], "raw target derivation")
        return pd.Series(df[report_target_column].astype(float).copy())

    _require_columns(df, [PSI_COLUMN], "psi_over_npl target derivation")
    return pd.Series(df[PSI_COLUMN].astype(float).copy())


def restore_report_target(
    values: Union[pd.Series, np.ndarray],
    *,
    target_mode: Optional[str] = None,
    target_transform_type: Optional[str] = None,
    reference_features: Optional[pd.DataFrame] = None,
    reference_scale: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """Map modeled outputs back into the reported Nexp-space."""
    normalized_mode = normalize_target_mode(target_mode)
    modeled_values = inverse_target_transform(values, target_transform_type)

    if normalized_mode == TARGET_MODE_RAW:
        return modeled_values

    if reference_scale is not None:
        scale = np.asarray(reference_scale, dtype=float).reshape(-1)
    else:
        if reference_features is None:
            raise ValueError(
                "reference_features or reference_scale is required to restore psi_over_npl targets"
            )
        _require_columns(reference_features, [NPL_COLUMN], "psi_over_npl target restoration")
        scale = reference_features[NPL_COLUMN].to_numpy(dtype=float)

    return modeled_values * scale
