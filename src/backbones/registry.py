"""Backbone adapter registry."""

from typing import Callable, Dict

from src.backbones.base import BackboneAdapter

_BACKBONE_FACTORIES: Dict[str, Callable[[], BackboneAdapter]] = {}


def register_backbone_adapter(name: str, factory: Callable[[], BackboneAdapter]) -> None:
    normalized = name.strip().lower()
    _BACKBONE_FACTORIES[normalized] = factory


def resolve_backbone_adapter(name: str) -> BackboneAdapter:
    normalized = name.strip().lower()
    if normalized not in _BACKBONE_FACTORIES:
        raise ValueError(
            f"Unsupported model backbone '{name}'. "
            f"Expected one of {sorted(_BACKBONE_FACTORIES.keys())}."
        )
    return _BACKBONE_FACTORIES[normalized]()


def list_backbone_adapters() -> list[str]:
    return sorted(_BACKBONE_FACTORIES.keys())
