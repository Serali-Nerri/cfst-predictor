from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predictor import Predictor, compare_predictions, export_predictions


class StubModel:
    def predict(self, X):
        return X.sum(axis=1).to_numpy(dtype=float)


class ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class StubPreprocessor:
    def transform(self, X):
        transformed = X.copy()
        transformed["a"] = transformed["a"] * 10
        return transformed


def test_predict_ignores_extra_features_and_applies_preprocessor():
    predictor = Predictor(
        model=StubModel(),
        preprocessor=StubPreprocessor(),
        feature_names=["a", "b"],
    )
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "extra": [99.0, 100.0]})

    predictions = predictor.predict(X)

    assert np.allclose(predictions, np.array([13.0, 24.0]))


def test_predict_raises_when_required_feature_missing():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0]})

    with pytest.raises(ValueError, match="Missing required features"):
        predictor.predict(X)


def test_predict_single_raises_for_multiple_rows():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(ValueError, match="Single prediction expects 1 row"):
        predictor.predict_single(X)


def test_predict_batch_matches_direct_predict():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    direct = predictor.predict(X)
    batched = predictor.predict_batch(X, batch_size=2)

    assert np.allclose(direct, batched)


def test_predict_restores_nexp_for_psi_target_with_precomputed_npl_input():
    predictor = Predictor(
        model=ConstantModel(np.log(0.8)),
        feature_names=["As (mm^2)", "Ac (mm^2)", "fy (MPa)", "fc (MPa)", "Npl (kN)"],
        metadata={
            "target_mode": "psi_over_npl",
            "report_target_column": "Nexp (kN)",
            "target_transform": {
                "enabled": True,
                "type": "log",
                "mode": "psi_over_npl",
                "original_column": "Nexp (kN)",
            },
        },
    )
    X = pd.DataFrame(
        {
            "As (mm^2)": [1000.0, 1200.0],
            "Ac (mm^2)": [2000.0, 1800.0],
            "fy (MPa)": [300.0, 320.0],
            "fc (MPa)": [40.0, 50.0],
            "Npl (kN)": [380.0, 474.0],
        }
    )

    predictions = predictor.predict(X)

    assert np.allclose(predictions, np.array([304.0, 379.2]))


def test_export_predictions_accepts_python_list(tmp_path):
    X = pd.DataFrame({"a": [1.0]})
    output_path = tmp_path / "predictions.csv"

    export_predictions(X, [3.5], str(output_path))

    exported = pd.read_csv(output_path)
    assert exported["prediction"].tolist() == [3.5]


def test_compare_predictions_rejects_row_count_mismatch(tmp_path):
    actual_path = tmp_path / "actual.csv"
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame({"actual": [1.0, 2.0, 3.0]}).to_csv(actual_path, index=False)
    pd.DataFrame({"prediction": [0.9, 1.9]}).to_csv(pred_path, index=False)

    with pytest.raises(ValueError, match="same number of rows"):
        compare_predictions(str(actual_path), str(pred_path))
