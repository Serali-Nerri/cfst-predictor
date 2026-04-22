from src.backbones import list_backbone_adapters, resolve_backbone_adapter


def test_registry_lists_new_backbones():
    backbones = set(list_backbone_adapters())

    assert "xgboost" in backbones
    assert "xgb" in backbones
    assert "rf" in backbones
    assert "random_forest" in backbones
    assert "mlp" in backbones
    assert "lightgbm" in backbones
    assert "lgbm" in backbones
    assert "catboost" in backbones


def test_registry_resolves_rf_alias_to_adapter():
    adapter = resolve_backbone_adapter("random_forest")

    assert adapter.name == "rf"
