import pytest
import pandas as pd

from src.backbones.catboost_adapter import CatBoostBackboneAdapter
from src.backbones.lightgbm_adapter import LightGBMBackboneAdapter


@pytest.mark.parametrize(
    ("adapter_cls", "missing_module", "expected_hint"),
    [
        (LightGBMBackboneAdapter, "lightgbm", "pip install lightgbm"),
        (CatBoostBackboneAdapter, "catboost", "pip install catboost"),
    ],
)
def test_optional_dependency_backbones_raise_clear_install_error(monkeypatch, adapter_cls, missing_module, expected_hint):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == missing_module:
            raise ImportError(f"No module named {missing_module}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    adapter = adapter_cls()
    X = pd.DataFrame({"feature": [0.0, 1.0]})
    y = pd.Series([0.0, 0.1])

    with pytest.raises(ImportError, match=expected_hint):
        adapter.fit(adapter.get_default_params(), X, y)
