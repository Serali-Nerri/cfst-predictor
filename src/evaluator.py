"""
Evaluator Module for CFST XGBoost Pipeline

This module handles model evaluation and metrics calculation for regression tasks.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Protocol, cast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import json

from src.splitting import (
    _resolve_regime_source_values,
    apply_regime_schema,
    fit_regime_schema,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegressionModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


VALID_REGIME_SORT_METRICS = {
    "rmse",
    "mae",
    "r2",
    "cov",
    "mean_ratio",
    "std_ratio",
    "n_samples",
}


def _calculate_cov_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    ratios = y_pred / y_true
    valid_mask = np.isfinite(ratios) & (y_true != 0)
    valid_ratios = ratios[valid_mask]
    if len(valid_ratios) == 0:
        return {"cov": None, "mean_ratio": None, "std_ratio": None}

    mean_ratio = float(np.mean(valid_ratios))
    std_ratio = float(np.std(valid_ratios, ddof=1)) if len(valid_ratios) > 1 else 0.0
    cov = std_ratio / mean_ratio if mean_ratio != 0 else None
    return {
        "cov": float(cov) if cov is not None else None,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
    }


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> Dict[str, Optional[float]]:
    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)

    mse = float(mean_squared_error(y_true_array, y_pred_array))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    r2 = float(r2_score(y_true_array, y_pred_array))
    max_error = float(np.max(np.abs(y_true_array - y_pred_array)))
    mean_pred = float(np.mean(y_pred_array))
    mean_true = float(np.mean(y_true_array))

    non_zero_mask = y_true_array != 0
    mape = (
        float(np.mean(np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask])) * 100)
        if non_zero_mask.any()
        else None
    )
    cov_stats = _calculate_cov_statistics(y_true_array, y_pred_array)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "mse": mse,
        "max_error": max_error,
        "mean_prediction": mean_pred,
        "mean_actual": mean_true,
        "cov": cov_stats["cov"],
        "mean_ratio": cov_stats["mean_ratio"],
        "std_ratio": cov_stats["std_ratio"],
        "n_samples": len(y_true_array),
    }


def _normalize_regime_sort_metric(raw_metric: Optional[str]) -> str:
    metric = str(raw_metric or "rmse").strip().lower()
    if metric not in VALID_REGIME_SORT_METRICS:
        raise ValueError(
            "evaluation.regime_analysis.sort_metric must be one of "
            f"{sorted(VALID_REGIME_SORT_METRICS)}"
        )
    return metric


def _group_sort_value(group: Dict[str, Any], sort_metric: str) -> float:
    if sort_metric == "n_samples":
        return float(group.get("n_samples", 0))

    metrics = group.get("metrics", {})
    raw_value = metrics.get(sort_metric)
    if raw_value is None:
        return float("-inf")
    if sort_metric == "mean_ratio":
        return abs(float(raw_value) - 1.0)
    return float(raw_value)


def _sort_regime_groups(groups: List[Dict[str, Any]], sort_metric: str) -> List[Dict[str, Any]]:
    reverse = sort_metric != "r2"
    return sorted(
        groups,
        key=lambda group: _group_sort_value(group, sort_metric),
        reverse=reverse,
    )


class Evaluator:
    """
    Model evaluator for regression tasks.

    Calculates comprehensive metrics including RMSE, MAE, R², and MAPE.
    Generates evaluation reports and handles metric persistence.
    """

    def __init__(self):
        """Initialize Evaluator."""
        self.metrics_history = []
        logger.info("Evaluator initialized")

    def calculate_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            Dictionary containing all calculated metrics

        Raises:
            ValueError: If input arrays have different lengths
        """
        logger.info("Calculating evaluation metrics")

        if len(y_true) != len(y_pred):
            error_msg = f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
        if np.isnan(y_true_array).any() or np.isnan(y_pred_array).any():
            logger.warning("NaN values detected in predictions or targets")

        metrics = calculate_regression_metrics(y_true_array, y_pred_array)
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        if metrics["mape"] is None:
            logger.warning("Cannot calculate MAPE: all true values are zero")
        else:
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Max Error: {metrics['max_error']:.4f}")
        if metrics["cov"] is not None:
            logger.info(
                f"COV: {metrics['cov']:.4f} "
                f"(μ={metrics['mean_ratio']:.4f}, σ={metrics['std_ratio']:.4f})"
            )

        return metrics

    def calculate_regime_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        features: Optional[pd.DataFrame],
        regime_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate metrics for previously fitted regime schemas.
        """
        if not regime_schema or not regime_schema.get("enabled", False):
            return {}

        fitted_regimes = regime_schema.get("regimes", [])
        sort_metric = _normalize_regime_sort_metric(regime_schema.get("sort_metric", "rmse"))
        if not isinstance(fitted_regimes, list):
            raise ValueError("evaluation.regime_analysis.regimes must be a list")

        y_true_series = pd.Series(np.asarray(y_true).reshape(-1), index=y_true.index)
        y_pred_array = np.asarray(y_pred).reshape(-1)
        if len(y_true_series) != len(y_pred_array):
            raise ValueError("y_true and y_pred must have the same length for regime metrics")

        results: Dict[str, Any] = {}
        for fitted in fitted_regimes:
            if not isinstance(fitted, dict):
                raise ValueError("Each fitted regime schema must be a mapping")
            regime_name = str(fitted.get("name", "")).strip()
            if not regime_name:
                raise ValueError("Each fitted regime schema requires a non-empty name")

            try:
                regime_values = _resolve_regime_source_values(
                    features=features,
                    target=y_true_series,
                    regime_spec=fitted,
                )
            except ValueError as exc:
                logger.warning("Skipping regime '%s': %s", regime_name, exc)
                continue

            labels = apply_regime_schema(regime_values, fitted)
            grouped_results = []
            for label in sorted(labels.dropna().unique()):
                mask = labels == label
                sample_count = int(mask.sum())
                if sample_count < 2:
                    logger.warning(
                        "Skipping regime '%s' bucket '%s' because it has fewer than 2 samples",
                        regime_name,
                        label,
                    )
                    continue
                grouped_results.append(
                    {
                        "label": label,
                        "n_samples": sample_count,
                        "metrics": self.calculate_metrics(
                            y_true_series[mask],
                            y_pred_array[mask.to_numpy()],
                        ),
                    }
                )

            if grouped_results:
                grouped_results = _sort_regime_groups(grouped_results, sort_metric)
                worst_bucket = max(
                    grouped_results,
                    key=lambda item: item["metrics"]["rmse"],
                )
                worst_cov_bucket = max(
                    grouped_results,
                    key=lambda item: (
                        item["metrics"]["cov"]
                        if item["metrics"].get("cov") is not None
                        else float("-inf")
                    ),
                )
                worst_r2_bucket = min(
                    grouped_results,
                    key=lambda item: item["metrics"]["r2"],
                )
                best_bucket = min(
                    grouped_results,
                    key=lambda item: item["metrics"]["rmse"],
                )
            else:
                worst_bucket = None
                worst_cov_bucket = None
                worst_r2_bucket = None
                best_bucket = None

            results[regime_name] = {
                "mode": fitted.get("mode"),
                "source": fitted.get("source"),
                "column": fitted.get("column"),
                "schema": fitted,
                "ranges": fitted.get("ranges", []),
                "groups": grouped_results,
                "worst_rmse_group": worst_bucket,
                "worst_cov_group": worst_cov_bucket,
                "worst_r2_group": worst_r2_bucket,
                "best_rmse_group": best_bucket,
            }

        return results

    def fit_regime_schema(
        self,
        y_true: pd.Series,
        features: Optional[pd.DataFrame],
        regime_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fit comparable regime schemas on the reference split."""
        if not regime_config or not regime_config.get("enabled", False):
            return {}

        regimes = regime_config.get("regimes", [])
        if not isinstance(regimes, list):
            raise ValueError("evaluation.regime_analysis.regimes must be a list")
        reference_split = str(regime_config.get("reference_split", "train")).strip().lower()
        if reference_split != "train":
            raise ValueError("evaluation.regime_analysis.reference_split currently only supports 'train'")
        sort_metric = _normalize_regime_sort_metric(regime_config.get("sort_metric", "rmse"))

        y_true_series = pd.Series(np.asarray(y_true).reshape(-1), index=y_true.index)
        fitted_regimes = []
        for regime_spec in regimes:
            if not isinstance(regime_spec, dict):
                raise ValueError("Each regime definition must be a mapping")
            regime_values = _resolve_regime_source_values(
                features=features,
                target=y_true_series,
                regime_spec=regime_spec,
            )
            fitted_regimes.append(fit_regime_schema(regime_values, regime_spec))

        return {
            "enabled": True,
            "reference_split": reference_split,
            "sort_metric": sort_metric,
            "regimes": fitted_regimes,
        }

    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a trained model on given data.

        Args:
            model: Trained model with predict method
            X: Features DataFrame
            y: True target values
            model_name: Name of the model for logging

        Returns:
            Dictionary containing metrics and evaluation info
        """
        logger.info(f"Evaluating {model_name} on {len(X)} samples")

        try:
            # Make predictions
            start_time = pd.Timestamp.now()
            y_pred = model.predict(X)
            prediction_time = (pd.Timestamp.now() - start_time).total_seconds()

            # Calculate metrics
            metrics = self.calculate_metrics(y, y_pred)

            # Add evaluation metadata
            evaluation_result = {
                'model_name': model_name,
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics': metrics,
                'prediction_time': prediction_time,
                'prediction_time_per_sample': prediction_time / len(X) if len(X) > 0 else 0
            }

            # Store in history
            self.metrics_history.append(evaluation_result)

            logger.info(f"Model evaluation completed in {prediction_time:.4f} seconds")

            return evaluation_result

        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def cross_validate_evaluation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                 cv_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.

        Args:
            model: Model to evaluate (will be cloned for each fold)
            X: Features DataFrame
            y: Target Series
            cv_splits: Number of cross-validation folds
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {cv_splits}-fold cross-validation evaluation")

        from sklearn.model_selection import KFold
        from sklearn.base import clone

        try:
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.debug(f"Evaluating fold {fold + 1}/{cv_splits}")

                # Split data
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]

                # Clone and train model
                fold_model = cast(RegressionModelProtocol, clone(model))
                fold_model.fit(X_train_fold, y_train_fold)

                # Evaluate on validation set
                y_pred_fold = fold_model.predict(X_val_fold)
                metrics = self.calculate_metrics(y_val_fold, y_pred_fold)

                fold_metrics.append({
                    'fold': fold + 1,
                    'metrics': metrics,
                    'n_train_samples': len(train_idx),
                    'n_val_samples': len(val_idx)
                })

            # Calculate aggregate statistics
            all_rmse = [f['metrics']['rmse'] for f in fold_metrics]
            all_mae = [f['metrics']['mae'] for f in fold_metrics]
            all_r2 = [f['metrics']['r2'] for f in fold_metrics]

            cv_results = {
                'fold_metrics': fold_metrics,
                'aggregate': {
                    'mean_rmse': float(np.mean(all_rmse)),
                    'std_rmse': float(np.std(all_rmse)),
                    'mean_mae': float(np.mean(all_mae)),
                    'std_mae': float(np.std(all_mae)),
                    'mean_r2': float(np.mean(all_r2)),
                    'std_r2': float(np.std(all_r2)),
                    'n_folds': cv_splits
                }
            }

            # Log aggregate results
            agg = cv_results['aggregate']
            logger.info(f"CV RMSE: {agg['mean_rmse']:.4f} (+/- {agg['std_rmse']:.4f})")
            logger.info(f"CV MAE: {agg['mean_mae']:.4f} (+/- {agg['std_mae']:.4f})")
            logger.info(f"CV R²: {agg['mean_r2']:.4f} (+/- {agg['std_r2']:.4f})")

            return cv_results

        except Exception as e:
            error_msg = f"Cross-validation evaluation failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def save_evaluation_report(self, evaluation_result: Dict[str, Any],
                              output_path: str) -> None:
        """
        Save evaluation report to JSON file.

        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Path to save the report

        Raises:
            Exception: If saving fails
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(evaluation_result, f, indent=2)

            logger.info(f"Evaluation report saved to {output_path}")

        except Exception as e:
            error_msg = f"Failed to save evaluation report: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def load_evaluation_report(self, report_path: str) -> Dict[str, Any]:
        """
        Load evaluation report from JSON file.

        Args:
            report_path: Path to the evaluation report

        Returns:
            Evaluation result dictionary

        Raises:
            Exception: If loading fails
        """
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)

            logger.info(f"Evaluation report loaded from {report_path}")
            return report

        except Exception as e:
            error_msg = f"Failed to load evaluation report: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def compare_models(self, model_results: list) -> pd.DataFrame:
        """
        Compare multiple models' evaluation results.

        Args:
            model_results: List of evaluation result dictionaries

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for result in model_results:
            model_name = result.get('model_name', 'unknown')
            metrics = result.get('metrics', {})

            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics.get('rmse'),
                'MAE': metrics.get('mae'),
                'R²': metrics.get('r2'),
                'MAPE': metrics.get('mape'),
                'COV': metrics.get('cov'),
                'Samples': metrics.get('n_samples')
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')

        logger.info("Model comparison completed")
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        return comparison_df

    def get_best_model(self, model_results: list, metric: str = 'rmse') -> Dict[str, Any]:
        """
        Get the best model based on specified metric.

        Args:
            model_results: List of evaluation result dictionaries
            metric: Metric to use for comparison (rmse, mae, r2)

        Returns:
            Best model's evaluation result
        """
        if not model_results:
            logger.warning("No model results provided")
            return {}

        # Determine if higher or lower is better
        lower_is_better = metric in ['rmse', 'mae', 'mape']

        def metric_value(result: Dict[str, Any]) -> float:
            raw_value = result.get('metrics', {}).get(metric)
            if raw_value is None:
                return float('inf') if lower_is_better else float('-inf')
            return float(raw_value)

        # Find best model
        if lower_is_better:
            best_result = min(model_results, key=metric_value)
        else:
            best_result = max(model_results, key=metric_value)

        best_model_name = best_result.get('model_name', 'unknown')
        best_value = best_result.get('metrics', {}).get(metric)
        best_value_text = f"{best_value:.4f}" if isinstance(best_value, (int, float)) else "N/A"

        logger.info(f"Best model: {best_model_name} ({metric}: {best_value_text})")

        return best_result
