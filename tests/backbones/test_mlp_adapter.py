import pandas as pd

from src.backbones.mlp_adapter import MLPBackboneAdapter


class DummyTrial:
    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


def test_mlp_adapter_builds_hidden_layers_from_trial():
    adapter = MLPBackboneAdapter()

    params = adapter.build_optuna_trial_params(DummyTrial(), adapter.get_default_params())

    assert "hidden_layer_sizes" in params
    assert isinstance(params["hidden_layer_sizes"], tuple)
    assert len(params["hidden_layer_sizes"]) == 2


def test_mlp_adapter_fit_ignores_injected_n_jobs(monkeypatch):
    adapter = MLPBackboneAdapter()
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    target = pd.Series([0.5, 1.5, 2.5])
    captured = {}

    class RecordingMLPRegressor:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def fit(self, X_fit, y_fit, **kwargs):
            captured["fit_kwargs"] = kwargs
            captured["shape"] = (X_fit.shape, y_fit.shape)
            return self

    monkeypatch.setattr("src.backbones.mlp_adapter.MLPRegressor", RecordingMLPRegressor)

    model = adapter.fit(
        {
            **adapter.get_default_params(),
            "n_jobs": 1,
            "max_iter": 10,
            "early_stopping": False,
        },
        X_train=X,
        y_train=target,
    )

    assert isinstance(model, RecordingMLPRegressor)
    assert "n_jobs" not in captured["init_kwargs"]
    assert captured["init_kwargs"]["max_iter"] == 10
    assert captured["fit_kwargs"] == {}


def test_mlp_adapter_normalizes_optuna_layer_params_for_fit():
    adapter = MLPBackboneAdapter()

    params = adapter._normalize_fit_params(
        {
            "first_layer": 96,
            "second_layer": 24,
            "n_jobs": 1,
            "alpha": 1e-4,
        }
    )

    assert params["hidden_layer_sizes"] == (96, 24)
    assert "first_layer" not in params
    assert "second_layer" not in params
    assert "n_jobs" not in params


def test_mlp_adapter_finalize_after_cv_is_noop():
    adapter = MLPBackboneAdapter()
    params = adapter.get_default_params()

    finalized, iterations = adapter.finalize_after_cv(params, {"fold_details": []})

    assert finalized == params
    assert iterations == []
