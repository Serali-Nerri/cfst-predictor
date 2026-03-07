import numpy as np
import pandas as pd
import pytest

from src.predictor import Predictor


class StubModel:
    def predict(self, X):
        return X.sum(axis=1).to_numpy(dtype=float)


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

    with pytest.raises(Exception, match="Missing required features"):
        predictor.predict(X)


def test_predict_single_raises_for_multiple_rows():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(Exception, match="Single prediction expects 1 row"):
        predictor.predict_single(X)


def test_predict_batch_matches_direct_predict():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    direct = predictor.predict(X)
    batched = predictor.predict_batch(X, batch_size=2)

    assert np.allclose(direct, batched)
