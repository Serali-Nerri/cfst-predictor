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


def test_mlp_adapter_finalize_after_cv_is_noop():
    adapter = MLPBackboneAdapter()
    params = adapter.get_default_params()

    finalized, iterations = adapter.finalize_after_cv(params, {"fold_details": []})

    assert finalized == params
    assert iterations == []
