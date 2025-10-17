from __future__ import annotations
from typing import Dict, Type

class ComponentFactory:
    _registry: Dict[str, type] = {}

    @classmethod
    def register_component(cls, key: str, component_cls: type) -> None:
        cls._registry[key] = component_cls

    @classmethod
    def create(cls, key: str, *args, **kwargs):
        if key not in cls._registry:
            raise KeyError(f"Component '{key}' not registered")
        return cls._registry[key](*args, **kwargs)