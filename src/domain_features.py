"""
Shared CFST target-space helpers.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


REPORT_TARGET_COLUMN_DEFAULT = "Nexp (kN)"
TARGET_MODE_RAW = "raw"
TARGET_MODE_ETA_U_OVER_NPL = "eta_u_over_npl"
TARGET_MODE_R_OVER_NPL = "r_over_npl"
TARGET_MODE_ALIASES = {
    "psi_over_npl": TARGET_MODE_ETA_U_OVER_NPL,
    "strength_ratio_over_npl": TARGET_MODE_ETA_U_OVER_NPL,
    "relative_residual_over_npl": TARGET_MODE_R_OVER_NPL,
}
VALID_TARGET_MODES = {
    TARGET_MODE_RAW,
    TARGET_MODE_ETA_U_OVER_NPL,
    TARGET_MODE_R_OVER_NPL,
    *TARGET_MODE_ALIASES.keys(),
}

NPL_COLUMN = "Npl (kN)"
ETA_U_COLUMN = "eta_u"
R_COLUMN = "r"
LEGACY_PSI_COLUMN = "psi"
LEGACY_STRENGTH_RATIO_COLUMN = "strength_ratio"
LEGACY_RELATIVE_RESIDUAL_COLUMN = "relative_residual"


def get_keml_config_payload(keml_config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Normalize KeML config for hashing and metadata serialization."""
    config = keml_config or {}
    linear_features = config.get("linear_features", [])
    if not isinstance(linear_features, list):
        linear_features = list(linear_features)
    return {
        "enabled": bool(config.get("enabled", False)),
        "linear_features": list(linear_features),
        "linear_ridge_alpha": float(config.get("linear_ridge_alpha", 1.0)),
    }


def normalize_target_mode(target_mode: Optional[str]) -> str:
    """Normalize the configured training target mode."""
    normalized = (target_mode or TARGET_MODE_RAW).strip().lower()
    normalized = TARGET_MODE_ALIASES.get(normalized, normalized)
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
    if normalized_mode == TARGET_MODE_ETA_U_OVER_NPL:
        return ETA_U_COLUMN
    if normalized_mode == TARGET_MODE_R_OVER_NPL:
        return R_COLUMN
    return report_target_column


def parse_boxcox_lambda(target_transform_type: str) -> Optional[float]:
    normalized = target_transform_type.strip().lower()
    if not normalized.startswith("boxcox_"):
        return None
    try:
        return float(normalized.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Unsupported target transform '{target_transform_type}'") from exc


def format_target_transform_label(
    target_name: str,
    target_transform_type: Optional[str],
) -> str:
    if target_transform_type is None:
        return target_name
    normalized = target_transform_type.strip().lower()
    if normalized == "log":
        return f"ln({target_name})"
    boxcox_lambda = parse_boxcox_lambda(normalized)
    if boxcox_lambda is not None:
        return f"BoxCox({target_name}; λ={boxcox_lambda:g})"
    raise ValueError(f"Unsupported target transform '{target_transform_type}'")


def apply_target_transform(
    values: Union[pd.Series, np.ndarray],
    target_transform_type: Optional[str],
) -> pd.Series:
    """Apply the configured target transform in model space."""
    series = pd.Series(np.asarray(values, dtype=float).reshape(-1))
    if target_transform_type is None:
        return series
    if (~np.isfinite(series)).any() or (series <= 0).any():
        raise ValueError(
            f"{target_transform_type} target transform requires all target values to be finite and > 0"
        )
    if target_transform_type == "log":
        return pd.Series(np.log(series), index=series.index)
    boxcox_lambda = parse_boxcox_lambda(target_transform_type)
    if boxcox_lambda is not None:
        if abs(boxcox_lambda) < 1e-12:
            return pd.Series(np.log(series), index=series.index)
        transformed = (np.power(series, boxcox_lambda) - 1.0) / boxcox_lambda
        return pd.Series(transformed, index=series.index)
    raise ValueError(f"Unsupported target transform '{target_transform_type}'")


def inverse_target_transform(
    values: Union[pd.Series, np.ndarray],
    target_transform_type: Optional[str],
) -> np.ndarray:
    """Invert the configured target transform."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if target_transform_type is None:
        return array
    if target_transform_type == "log":
        return np.exp(array)
    boxcox_lambda = parse_boxcox_lambda(target_transform_type)
    if boxcox_lambda is not None:
        if abs(boxcox_lambda) < 1e-12:
            return np.exp(array)
        base = boxcox_lambda * array + 1.0
        base = np.where(np.isfinite(base), base, np.nan)
        base = np.maximum(base, 1e-12)
        return np.power(base, 1.0 / boxcox_lambda)
    raise ValueError(f"Unsupported target transform '{target_transform_type}'")


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

    if normalized_mode == TARGET_MODE_ETA_U_OVER_NPL:
        if ETA_U_COLUMN in df.columns:
            return pd.Series(df[ETA_U_COLUMN].astype(float).copy())
        if LEGACY_STRENGTH_RATIO_COLUMN in df.columns:
            return pd.Series(df[LEGACY_STRENGTH_RATIO_COLUMN].astype(float).copy())
        _require_columns(df, [LEGACY_PSI_COLUMN], "eta_u_over_npl target derivation")
        return pd.Series(df[LEGACY_PSI_COLUMN].astype(float).copy())

    if R_COLUMN in df.columns:
        return pd.Series(df[R_COLUMN].astype(float).copy())

    if LEGACY_RELATIVE_RESIDUAL_COLUMN in df.columns:
        return pd.Series(df[LEGACY_RELATIVE_RESIDUAL_COLUMN].astype(float).copy())

    if ETA_U_COLUMN in df.columns:
        return pd.Series(df[ETA_U_COLUMN].astype(float).copy() - 1.0)

    if LEGACY_STRENGTH_RATIO_COLUMN in df.columns:
        return pd.Series(df[LEGACY_STRENGTH_RATIO_COLUMN].astype(float).copy() - 1.0)

    _require_columns(df, [LEGACY_PSI_COLUMN], "r_over_npl target derivation")
    return pd.Series(df[LEGACY_PSI_COLUMN].astype(float).copy() - 1.0)


def project_report_target_to_model_space(
    values: Union[pd.Series, np.ndarray],
    *,
    target_mode: Optional[str] = None,
    reference_features: Optional[pd.DataFrame] = None,
    reference_scale: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """Map reported Nexp-space values into the untransformed model space."""
    normalized_mode = normalize_target_mode(target_mode)
    reported_values = np.asarray(values, dtype=float).reshape(-1)

    if normalized_mode == TARGET_MODE_RAW:
        return reported_values

    if reference_scale is not None:
        scale = np.asarray(reference_scale, dtype=float).reshape(-1)
    else:
        if reference_features is None:
            raise ValueError(
                "reference_features or reference_scale is required to project Npl-based targets"
            )
        _require_columns(reference_features, [NPL_COLUMN], "Npl-based target projection")
        scale = reference_features[NPL_COLUMN].to_numpy(dtype=float)

    eta_u_values = reported_values / scale
    if normalized_mode == TARGET_MODE_R_OVER_NPL:
        return eta_u_values - 1.0
    return eta_u_values


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
                "reference_features or reference_scale is required to restore Npl-based targets"
            )
        _require_columns(reference_features, [NPL_COLUMN], "Npl-based target restoration")
        scale = reference_features[NPL_COLUMN].to_numpy(dtype=float)

    if normalized_mode == TARGET_MODE_R_OVER_NPL:
        return (1.0 + modeled_values) * scale

    return modeled_values * scale
