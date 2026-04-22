import numpy as np
import pandas as pd

from src.backbones.random_forest_adapter import RandomForestBackboneAdapter


class RecordingRandomForestRegressor:
    last_init_kwargs = None
    last_fit_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = dict(kwargs)

    def fit(self, X, y, **kwargs):
        type(self).last_fit_kwargs = dict(kwargs)
        return self


def test_rf_adapter_passes_sample_weight(monkeypatch):
    adapter = RandomForestBackboneAdapter()
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    y = pd.Series([0.0, 0.1, 0.2])
    sample_weight = pd.Series([1.0, 2.0, 3.0])

    monkeypatch.setattr(
        "src.backbones.random_forest_adapter.RandomForestRegressor",
        RecordingRandomForestRegressor,
    )

    adapter.fit(adapter.get_default_params(), X, y, sample_weight=sample_weight)

    assert RecordingRandomForestRegressor.last_fit_kwargs is not None
    assert np.allclose(
        RecordingRandomForestRegressor.last_fit_kwargs["sample_weight"],
        np.array([1.0, 2.0, 3.0]),
    )
