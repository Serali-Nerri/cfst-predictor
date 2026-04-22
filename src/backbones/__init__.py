"""Backbone adapters package."""

from src.backbones.registry import list_backbone_adapters, resolve_backbone_adapter
from src.backbones.xgboost_adapter import XGBoostBackboneAdapter

__all__ = [
    "XGBoostBackboneAdapter",
    "list_backbone_adapters",
    "resolve_backbone_adapter",
]
