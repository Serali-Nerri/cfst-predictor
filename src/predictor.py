"""
Predictor Module for CFST XGBoost Pipeline

This module handles model prediction, including single and batch predictions,
input validation, and prediction export functionality.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Any, Optional, Dict, cast
from pathlib import Path

from src.domain_features import NPL_COLUMN, normalize_target_mode, restore_report_target
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Predictor for making predictions with trained XGBoost models.

    Supports both single record prediction and batch prediction with
    comprehensive input validation and error handling.
    """

    def __init__(self, model: Any, preprocessor: Optional[Any] = None,
                 feature_names: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Predictor.

        Args:
            model: Trained XGBoost model with predict method
            preprocessor: Fitted preprocessor for data transformation (optional)
            feature_names: List of expected feature names (optional)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names or []
        self.metadata = metadata or {}
        self._validate_model()
        logger.info("Predictor initialized")
        if self.feature_names:
            logger.info(f"Expected features: {len(self.feature_names)}")

    def _validate_model(self) -> None:
        """Validate that the model has a predict method."""
        if not hasattr(self.model, 'predict'):
            error_msg = "Model does not have a predict method"
            logger.error(error_msg)
            raise AttributeError(error_msg)
        logger.info("Model validation passed")

    def _validate_input_data(self, X: pd.DataFrame) -> None:
        """
        Validate input data matches expected format.

        Args:
            X: Input features DataFrame

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            error_msg = "Input must be a pandas DataFrame"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if X.empty:
            error_msg = "Input DataFrame is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for required features
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                error_msg = f"Missing required features: {missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Check for extra features
            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                logger.warning(f"Extra features will be ignored: {extra_features}")

        logger.debug(f"Input validation passed for {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features DataFrame

        Returns:
            Array of predictions

        Raises:
            Exception: If prediction fails
        """
        logger.info(f"Making predictions on {len(X)} samples")

        try:
            self._validate_input_data(X)

            X_processed = X[self.feature_names] if self.feature_names else X
            if self.feature_names:
                logger.debug(f"Filtered to {len(self.feature_names)} features")

            reference_features: Optional[pd.DataFrame] = None
            reference_scale: Optional[np.ndarray] = None
            if self._get_target_mode() != "raw":
                if NPL_COLUMN in X.columns:
                    reference_scale = X[NPL_COLUMN].to_numpy(dtype=float)
                else:
                    reference_features = X

            if self.preprocessor is not None:
                logger.debug("Applying preprocessor transformation")
                X_processed = self.preprocessor.transform(X_processed)

            predictions = self.model.predict(X_processed)
            predictions = restore_report_target(
                predictions,
                target_mode=self._get_target_mode(),
                target_transform_type=self._get_target_transform_type(),
                reference_features=reference_features,
                reference_scale=reference_scale,
            )

            logger.info(f"Predictions completed: {len(predictions)} samples")
            logger.debug(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

            return predictions

        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise

    def _get_target_mode(self) -> str:
        target_transform = self.metadata.get("target_transform", {})
        raw_mode = self.metadata.get("target_mode", target_transform.get("mode", "raw"))
        return normalize_target_mode(raw_mode)

    def _get_target_transform_type(self) -> Optional[str]:
        target_transform = self.metadata.get("target_transform", {})
        return target_transform.get("type") if target_transform.get("enabled") else None

    def predict_single(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> float:
        """
        Make prediction for a single record.

        Args:
            data: Single record as DataFrame (1 row) or dictionary

        Returns:
            Single prediction value

        Raises:
            Exception: If prediction fails
        """
        logger.debug("Making single prediction")

        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])

            if len(data) != 1:
                error_msg = f"Single prediction expects 1 row, got {len(data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            predictions = self.predict(data)
            prediction = float(predictions[0])
            logger.info(f"Single prediction: {prediction:.4f}")
            return prediction

        except Exception as exc:
            logger.error("Single prediction failed: %s", exc)
            raise

    def predict_batch(self, X: pd.DataFrame, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Make predictions in batches for large datasets.

        Args:
            X: Input features DataFrame
            batch_size: Number of samples per batch (optional, uses all if None)

        Returns:
            Array of all predictions

        Raises:
            Exception: If prediction fails
        """
        n_samples = len(X)
        logger.info(f"Making batch predictions on {n_samples} samples")

        if batch_size is None or batch_size >= n_samples:
            # Predict on all data at once
            logger.debug("Predicting on all data at once")
            return self.predict(X)

        # Predict in batches
        all_predictions = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        logger.info(f"Processing in {n_batches} batches (batch_size={batch_size})")

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = X.iloc[i:end_idx]

            logger.debug(f"Processing batch {(i // batch_size) + 1}/{n_batches}: samples {i}-{end_idx}")

            batch_predictions = self.predict(batch)
            all_predictions.extend(batch_predictions)

            # Log progress every 5 batches
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Completed {i // batch_size + 1}/{n_batches} batches")

        predictions_array = np.array(all_predictions)
        logger.info(f"Batch predictions completed: {len(predictions_array)} samples")

        return predictions_array

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions (for classification models).

        Note: This is not applicable for regression models like XGBRegressor.
        Included for interface completeness.

        Args:
            X: Input features DataFrame

        Returns:
            Array of probabilities (raises error for regression models)

        Raises:
            NotImplementedError: For regression models
        """
        logger.warning("predict_proba is not implemented for regression models")
        raise NotImplementedError("predict_proba is not available for regression models")

    def get_feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get SHAP values or feature contributions for predictions.

        Note: This is a placeholder for future SHAP integration.
        Currently returns basic feature values.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with feature contributions

        Raises:
            Exception: If calculation fails
        """
        logger.info("Getting feature contributions")

        try:
            self._validate_input_data(X)

            if self.feature_names:
                contributions = cast(pd.DataFrame, X[self.feature_names].copy())
            else:
                contributions = X.copy()

            logger.info(f"Feature contributions calculated for {len(X)} samples")
            return contributions

        except Exception as exc:
            logger.error("Failed to get feature contributions: %s", exc)
            raise


def export_predictions(X: pd.DataFrame, predictions: Union[np.ndarray, List[float]],
                      output_path: str, include_features: bool = True) -> None:
    """
    Export predictions to CSV file with optional feature inclusion.

    Args:
        X: Input features DataFrame
        predictions: Array of predictions
        output_path: Path to save predictions
        include_features: Whether to include input features in output

    Raises:
        Exception: If export fails
    """
    logger.info(f"Exporting predictions to {output_path}")

    try:
        prediction_array = np.asarray(predictions, dtype=float).reshape(-1)

        # Create output DataFrame
        if include_features:
            output_df = X.copy()
        else:
            output_df = pd.DataFrame()

        # Add prediction column
        output_df['prediction'] = prediction_array

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        output_df.to_csv(output_path, index=False)

        logger.info(f"Predictions exported successfully: {len(output_df)} rows, {len(output_df.columns)} columns")
        logger.debug(
            f"Prediction statistics - Min: {prediction_array.min():.4f}, "
            f"Max: {prediction_array.max():.4f}, Mean: {prediction_array.mean():.4f}"
        )

    except Exception as exc:
        logger.error("Failed to export predictions: %s", exc)
        raise


def load_predictions_and_features(prediction_path: str) -> pd.DataFrame:
    """
    Load predictions CSV file.

    Args:
        prediction_path: Path to predictions CSV file

    Returns:
        DataFrame with predictions and features

    Raises:
        Exception: If loading fails
    """
    logger.info(f"Loading predictions from {prediction_path}")

    try:
        df = pd.read_csv(prediction_path)
        logger.info(f"Predictions loaded: {len(df)} rows, {len(df.columns)} columns")
        return df

    except Exception as exc:
        logger.error("Failed to load predictions: %s", exc)
        raise


def compare_predictions(actual_path: str, prediction_path: str,
                       output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare actual vs predicted values.

    Args:
        actual_path: Path to actual values CSV
        prediction_path: Path to predictions CSV
        output_path: Optional path to save comparison

    Returns:
        DataFrame with comparison results

    Raises:
        Exception: If comparison fails
    """
    logger.info("Comparing actual vs predicted values")

    try:
        # Load data
        actual_df = pd.read_csv(actual_path)
        pred_df = pd.read_csv(prediction_path)

        if 'prediction' not in pred_df.columns:
            raise ValueError("Prediction file must contain a 'prediction' column")
        if len(actual_df) != len(pred_df):
            raise ValueError(
                "Actual and prediction files must contain the same number of rows "
                f"(got {len(actual_df)} and {len(pred_df)})"
            )

        # Merge dataframes
        comparison_df = pred_df.copy()
        comparison_df['actual'] = actual_df.iloc[:, -1].to_numpy()  # Assume actual is last column

        # Calculate errors
        comparison_df['error'] = comparison_df['actual'] - comparison_df['prediction']
        comparison_df['abs_error'] = np.abs(comparison_df['error'])
        comparison_df['rel_error'] = comparison_df['error'] / comparison_df['actual'] * 100

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(output_path, index=False)
            logger.info(f"Comparison saved to {output_path}")

        # Log statistics
        logger.info(f"Error statistics - Mean: {comparison_df['error'].mean():.4f}, Std: {comparison_df['error'].std():.4f}")
        logger.info(f"Absolute error - Mean: {comparison_df['abs_error'].mean():.4f}, Max: {comparison_df['abs_error'].max():.4f}")

        return comparison_df

    except Exception as exc:
        logger.error("Failed to compare predictions: %s", exc)
        raise
