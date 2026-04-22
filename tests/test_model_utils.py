import json

import pytest

from src.utils.model_utils import (
    DEFAULT_ARTIFACT_MANIFEST_NAME,
    LEGACY_MODEL_NAME,
    load_best_params,
    load_model_from_directory,
    resolve_artifact_paths,
    save_model,
)


def test_load_best_params_returns_none_when_file_missing(tmp_path):
    result = load_best_params(str(tmp_path / "missing.json"), expected_context_hash="abc123")
    assert result is None


def test_load_best_params_returns_none_for_invalid_format(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({"best_rmse": 1.23}), encoding="utf-8")

    result = load_best_params(str(path), expected_context_hash="abc123")
    assert result is None


def test_load_best_params_returns_none_for_context_mismatch(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(
        json.dumps(
            {
                "parameters": {"max_depth": 5},
                "context_hash": "wrong-context",
            }
        ),
        encoding="utf-8",
    )

    result = load_best_params(str(path), expected_context_hash="expected-context")
    assert result is None


def test_load_best_params_returns_parameters_on_context_match(tmp_path):
    path = tmp_path / "best_params.json"
    expected = {"max_depth": 5, "learning_rate": 0.05}
    path.write_text(
        json.dumps(
            {
                "parameters": expected,
                "context_hash": "expected-context",
            }
        ),
        encoding="utf-8",
    )

    result = load_best_params(str(path), expected_context_hash="expected-context")
    assert result == expected


class _StubModel:
    def predict(self, X):
        return [0.0 for _ in range(len(X))]


def test_save_model_writes_manifest_and_resolve_paths(tmp_path):
    output_dir = tmp_path / "artifacts"

    save_model(
        model=_StubModel(),
        preprocessor=None,
        feature_names=["a", "b"],
        output_dir=str(output_dir),
    )

    manifest_path = output_dir / DEFAULT_ARTIFACT_MANIFEST_NAME
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["model"] == LEGACY_MODEL_NAME
    assert payload["artifacts"]["feature_names"] == "feature_names.json"

    resolved = resolve_artifact_paths(str(output_dir))
    assert resolved["model"] == str(output_dir / LEGACY_MODEL_NAME)
    assert resolved["feature_names"] == str(output_dir / "feature_names.json")


def test_load_model_from_directory_supports_manifest_named_model(tmp_path):
    output_dir = tmp_path / "manifest_model"
    output_dir.mkdir()

    save_model(
        model=_StubModel(),
        preprocessor=None,
        feature_names=["x"],
        output_dir=str(output_dir),
        model_name="model.pkl",
    )

    (output_dir / LEGACY_MODEL_NAME).unlink(missing_ok=True)

    model, preprocessor, feature_names = load_model_from_directory(str(output_dir))

    assert hasattr(model, "predict")
    assert preprocessor is None
    assert feature_names == ["x"]


def test_load_model_from_directory_falls_back_without_manifest(tmp_path):
    output_dir = tmp_path / "legacy_model"
    output_dir.mkdir()

    save_model(
        model=_StubModel(),
        preprocessor=None,
        feature_names=["y"],
        output_dir=str(output_dir),
    )

    (output_dir / DEFAULT_ARTIFACT_MANIFEST_NAME).unlink()

    model, preprocessor, feature_names = load_model_from_directory(str(output_dir))

    assert hasattr(model, "predict")
    assert preprocessor is None
    assert feature_names == ["y"]



def test_resolve_artifact_paths_rejects_manifest_path_escape(tmp_path):
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / DEFAULT_ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": {
                    "model": "../outside.pkl",
                    "preprocessor": "preprocessor.pkl",
                    "feature_names": "feature_names.json",
                    "metadata": "training_metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes model directory"):
        resolve_artifact_paths(str(output_dir))



def test_load_model_from_directory_falls_back_when_manifest_model_missing(tmp_path):
    output_dir = tmp_path / "stale_manifest"
    output_dir.mkdir()

    save_model(
        model=_StubModel(),
        preprocessor=None,
        feature_names=["z"],
        output_dir=str(output_dir),
    )

    (output_dir / DEFAULT_ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": {
                    "model": "missing-model.pkl",
                    "preprocessor": "preprocessor.pkl",
                    "feature_names": "feature_names.json",
                    "metadata": "training_metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    model, preprocessor, feature_names = load_model_from_directory(str(output_dir))

    assert hasattr(model, "predict")
    assert preprocessor is None
    assert feature_names == ["z"]
