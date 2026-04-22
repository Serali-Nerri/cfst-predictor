'''
Model utilities module for the CFST backbone pipeline

This module handles saving and loading trained models, preprocessors, and metadata.
'''

import joblib
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Any, Optional, Dict, cast
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
LEGACY_MODEL_NAME = "xgboost_model.pkl"
DEFAULT_PREPROCESSOR_NAME = "preprocessor.pkl"
DEFAULT_FEATURE_NAMES_NAME = "feature_names.json"
DEFAULT_METADATA_NAME = "training_metadata.json"


def _default_artifact_manifest() -> Dict[str, str]:
    """Return default artifact file names for current and legacy compatibility."""
    return {
        "model": LEGACY_MODEL_NAME,
        "preprocessor": DEFAULT_PREPROCESSOR_NAME,
        "feature_names": DEFAULT_FEATURE_NAMES_NAME,
        "metadata": DEFAULT_METADATA_NAME,
    }


def _load_artifact_manifest(model_dir_path: Path, manifest_name: str = DEFAULT_ARTIFACT_MANIFEST_NAME) -> Dict[str, str]:
    """Load artifact manifest from disk when present and valid."""
    manifest_path = model_dir_path / manifest_name
    if not manifest_path.exists():
        logger.info(
            "Artifact manifest not found at %s; using legacy defaults",
            manifest_path,
        )
        return _default_artifact_manifest()

    try:
        with open(manifest_path, 'r') as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to read artifact manifest at %s (%s); using legacy defaults",
            manifest_path,
            exc,
        )
        return _default_artifact_manifest()

    if not isinstance(payload, dict):
        logger.warning(
            "Invalid artifact manifest root at %s; using legacy defaults",
            manifest_path,
        )
        return _default_artifact_manifest()

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        logger.warning(
            "Invalid artifact manifest format at %s; using legacy defaults",
            manifest_path,
        )
        return _default_artifact_manifest()

    manifest = _default_artifact_manifest()
    for key in manifest:
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            manifest[key] = value.strip()

    return manifest


def _save_artifact_manifest(
    output_path: Path,
    model_name: str,
    preprocessor_name: str,
    feature_names_name: str,
    metadata_name: str,
    manifest_name: str = DEFAULT_ARTIFACT_MANIFEST_NAME,
) -> None:
    """Save lightweight artifact manifest describing artifact file names."""
    manifest_path = output_path / manifest_name
    payload = {
        "version": 1,
        "artifacts": {
            "model": model_name,
            "preprocessor": preprocessor_name,
            "feature_names": feature_names_name,
            "metadata": metadata_name,
        },
    }

    try:
        with open(manifest_path, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Artifact manifest saved to {manifest_path}")
    except Exception as exc:
        logger.error("Failed to save artifact manifest to %s: %s", manifest_path, exc)
        raise Exception(f"Failed to save artifact manifest: {exc}")


def _resolve_artifact_path(model_dir_path: Path, artifact_name: str) -> Path:
    candidate = (model_dir_path / artifact_name).resolve()
    try:
        candidate.relative_to(model_dir_path.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Artifact path '{artifact_name}' escapes model directory '{model_dir_path}'"
        ) from exc
    return candidate



def resolve_artifact_paths(
    model_dir: str,
    model_name: Optional[str] = None,
    preprocessor_name: Optional[str] = None,
    feature_names_name: Optional[str] = None,
    metadata_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve model artifact paths from directory with manifest + legacy fallback."""
    model_dir_path = Path(model_dir).resolve()

    if not model_dir_path.exists():
        error_msg = f"Model directory not found: {model_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    manifest = _load_artifact_manifest(model_dir_path)

    resolved_model_name = model_name or manifest["model"]
    resolved_preprocessor_name = preprocessor_name or manifest["preprocessor"]
    resolved_feature_names_name = feature_names_name or manifest["feature_names"]
    resolved_metadata_name = metadata_name or manifest["metadata"]

    model_path = _resolve_artifact_path(model_dir_path, resolved_model_name)
    preprocessor_path = _resolve_artifact_path(model_dir_path, resolved_preprocessor_name)
    feature_names_path = _resolve_artifact_path(model_dir_path, resolved_feature_names_name)
    metadata_path = _resolve_artifact_path(model_dir_path, resolved_metadata_name)

    if model_name is None and not model_path.exists() and manifest["model"] != LEGACY_MODEL_NAME:
        legacy_model_path = _resolve_artifact_path(model_dir_path, LEGACY_MODEL_NAME)
        if legacy_model_path.exists():
            model_path = legacy_model_path

    return {
        "model": str(model_path) if model_path.exists() else None,
        "preprocessor": str(preprocessor_path) if preprocessor_path.exists() else None,
        "feature_names": str(feature_names_path) if feature_names_path.exists() else None,
        "metadata": str(metadata_path) if metadata_path.exists() else None,
    }


def save_model(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    output_dir: str,
    model_name: str = LEGACY_MODEL_NAME,
    preprocessor_name: str = DEFAULT_PREPROCESSOR_NAME,
    feature_names_name: str = DEFAULT_FEATURE_NAMES_NAME,
    metadata: Optional[dict] = None,
    metadata_name: str = DEFAULT_METADATA_NAME,
) -> None:
    """
    Save trained model, preprocessor, and metadata.

    Args:
        model: Trained model object
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        output_dir: Output directory path
        model_name: Model file name (default: legacy xgboost_model.pkl)
        preprocessor_name: Preprocessor file name (default: preprocessor.pkl)
        feature_names_name: Feature names file name (default: feature_names.json)
        metadata: Additional metadata to save (optional)
        metadata_name: Metadata file name (default: training_metadata.json)

    Raises:
        Exception: If saving fails
    """
    logger.info(f"Saving model and artifacts to {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / model_name
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise Exception(f"Failed to save model: {str(e)}")

    # Save preprocessor
    if preprocessor is not None:
        preprocessor_path = output_path / preprocessor_name
        try:
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {str(e)}")
            raise Exception(f"Failed to save preprocessor: {str(e)}")

    # Save feature names
    feature_path = output_path / feature_names_name
    try:
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Feature names saved to {feature_path}")
    except Exception as e:
        logger.error(f"Failed to save feature names: {str(e)}")
        raise Exception(f"Failed to save feature names: {str(e)}")

    # Save metadata if provided
    if metadata is not None:
        metadata_path = output_path / metadata_name
        try:
            # Convert any non-serializable objects
            serializable_metadata = _make_serializable(metadata)

            with open(metadata_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            # Don't raise exception for metadata, just log the error
            logger.warning("Model and preprocessor saved, but metadata failed")

    _save_artifact_manifest(
        output_path=output_path,
        model_name=model_name,
        preprocessor_name=preprocessor_name,
        feature_names_name=feature_names_name,
        metadata_name=metadata_name,
    )

    logger.info("Model and artifacts saved successfully")


def load_model(
    model_path: str,
    preprocessor_path: Optional[str] = None,
    feature_names_path: Optional[str] = None
) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """
    Load trained model, preprocessor, and feature names.

    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file (optional)
        feature_names_path: Path to feature names file (optional)

    Returns:
        Tuple of (model, preprocessor, feature_names)

    Raises:
        Exception: If loading fails
    """
    logger.info(f"Loading model from {model_path}")

    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

    # Load preprocessor
    preprocessor = None
    if preprocessor_path is not None and Path(preprocessor_path).exists():
        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {str(e)}")
            logger.warning("Continuing without preprocessor")

    # Load feature names
    feature_names = None
    if feature_names_path is not None and Path(feature_names_path).exists():
        try:
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Feature names loaded from {feature_names_path}")
            logger.info(f"Number of features: {len(feature_names)}")
        except Exception as e:
            logger.error(f"Failed to load feature names: {str(e)}")
            logger.warning("Continuing without feature names")

    return model, preprocessor, feature_names


def load_model_from_directory(
    model_dir: str,
    model_name: Optional[str] = None,
    preprocessor_name: Optional[str] = None,
    feature_names_name: Optional[str] = None,
) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """
    Load model and artifacts from a directory.

    Args:
        model_dir: Directory containing model artifacts
        model_name: Model file name (default: legacy xgboost_model.pkl)
        preprocessor_name: Preprocessor file name (default: preprocessor.pkl)
        feature_names_name: Feature names file name (default: feature_names.json)

    Returns:
        Tuple of (model, preprocessor, feature_names)
    """
    resolved_paths = resolve_artifact_paths(
        model_dir=model_dir,
        model_name=model_name,
        preprocessor_name=preprocessor_name,
        feature_names_name=feature_names_name,
        metadata_name=DEFAULT_METADATA_NAME,
    )

    model_path = resolved_paths["model"]
    if model_path is None:
        raise FileNotFoundError(f"Required model artifact not found under {model_dir}")

    return load_model(
        cast(str, model_path),
        resolved_paths["preprocessor"],
        resolved_paths["feature_names"],
    )


def save_metadata(metadata: dict, output_path: str) -> None:
    """
    Save metadata to JSON file.

    Args:
        metadata: Metadata dictionary
        output_path: Output file path

    Raises:
        Exception: If saving fails
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Make metadata serializable
        serializable_metadata = _make_serializable(metadata)

        with open(output_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

        logger.info(f"Metadata saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")
        raise Exception(f"Failed to save metadata: {str(e)}")


def load_metadata(metadata_path: str) -> dict:
    """
    Load metadata from JSON file.

    Args:
        metadata_path: Path to metadata file

    Returns:
        Metadata dictionary

    Raises:
        Exception: If loading fails
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict):
            raise ValueError("Metadata JSON root must be an object")

        logger.info(f"Metadata loaded from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        raise Exception(f"Failed to load metadata: {str(e)}")


def _make_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to serializable format.

    Args:
        obj: Object to convert

    Returns:
        Serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    else:
        return str(obj)


def validate_model(model: Any, X_sample: pd.DataFrame) -> bool:
    """
    Validate that a model can make predictions.

    Args:
        model: Trained model
        X_sample: Sample features for prediction

    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Try to make a prediction
        prediction = model.predict(X_sample.iloc[:1])

        # Check if prediction is reasonable
        if prediction is None:
            logger.error("Model returned None prediction")
            return False

        if not np.isfinite(prediction).all():
            logger.error("Model returned non-finite prediction")
            return False

        logger.info("Model validation passed")
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False


def get_model_size(model_path: str) -> int:
    """
    Get the size of a model file in bytes.

    Args:
        model_path: Path to model file

    Returns:
        File size in bytes
    """
    try:
        return Path(model_path).stat().st_size
    except Exception as e:
        logger.error(f"Failed to get model size: {str(e)}")
        return 0


def list_model_files(model_dir: str) -> dict:
    """
    List all model-related files in a directory.

    Args:
        model_dir: Directory containing model files

    Returns:
        Dictionary with file information
    """
    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return {}

    files_info = {}
    for file_path in model_dir_path.iterdir():
        if file_path.is_file():
            files_info[file_path.name] = {
                'size_bytes': file_path.stat().st_size,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': file_path.stat().st_mtime
            }

    return files_info


def save_best_params(
    best_params: dict,
    best_score: float,
    trial_number: int,
    n_trials: int,
    output_path: str = 'logs/best_params.json',
    context_hash: Optional[str] = None,
    data_file: Optional[str] = None,
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
    score_label: str = 'best_rmse',
) -> None:
    """
    Save best Optuna parameters to JSON file.

    Args:
        best_params: Dictionary of best hyperparameters
        best_score: Best RMSE score achieved
        trial_number: Trial number that achieved the best score
        n_trials: Total number of trials run
        output_path: Output file path (default: logs/best_params.json)
        context_hash: Hash of data/config context for safe reuse
        data_file: Absolute data file path used for optimization
        study_name: Optuna study name
        storage_url: Optuna storage URL
        score_label: Field name used to persist the best score
    """
    from datetime import datetime

    data = {
        'trial_number': trial_number,
        score_label: best_score,
        'best_score': best_score,
        'parameters': best_params,
        'timestamp': datetime.now().isoformat(),
        'n_trials_total': n_trials,
        'context_hash': context_hash,
        'data_file': data_file,
        'study_name': study_name,
        'storage_url': storage_url,
    }

    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Best parameters saved to {output_path}")
    logger.info(f"  Best score ({score_label}): {best_score:.4f} (Trial {trial_number})")


def load_best_params(
    input_path: str = 'logs/best_params.json',
    expected_context_hash: Optional[str] = None,
) -> Optional[dict]:
    """
    Load best Optuna parameters from JSON file.

    Args:
        input_path: Path to best parameters file (default: logs/best_params.json)
        expected_context_hash: Expected context hash for strict compatibility check

    Returns:
        Dictionary of best parameters, or None if file doesn't exist or context mismatch

    Note:
        Returns None if file doesn't exist (first run without Optuna).
    """
    path = Path(input_path)
    if not path.exists():
        logger.warning(f"No saved parameters found at {input_path}")
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    if 'parameters' not in data or not isinstance(data['parameters'], dict):
        logger.warning(f"Invalid best parameters file format at {input_path}")
        return None

    saved_context_hash = data.get('context_hash')
    if expected_context_hash is not None:
        if saved_context_hash is None:
            logger.warning(
                "Saved best parameters do not include context_hash; ignoring to avoid cross-dataset reuse"
            )
            return None
        if saved_context_hash != expected_context_hash:
            logger.warning(
                "Saved best parameters context hash mismatch "
                f"(expected={expected_context_hash}, found={saved_context_hash}); ignoring"
            )
            return None

    logger.info(f"Loaded best parameters from {input_path}")
    if 'best_score' in data and 'trial_number' in data:
        logger.info(f"  Best score: {data['best_score']:.4f} (Trial {data['trial_number']})")
    elif 'best_rmse' in data and 'trial_number' in data:
        logger.info(f"  Best RMSE: {data['best_rmse']:.4f} (Trial {data['trial_number']})")
    if 'timestamp' in data:
        logger.info(f"  Saved on: {data['timestamp']}")
    if 'n_trials_total' in data:
        logger.info(f"  Total trials: {data['n_trials_total']}")
    if saved_context_hash:
        logger.info(f"  Context hash: {saved_context_hash}")
    if data.get('study_name'):
        logger.info(f"  Study name: {data['study_name']}")

    return data['parameters']
