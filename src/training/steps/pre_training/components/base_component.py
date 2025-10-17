from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import logging

@dataclass
class ComponentConfig:
    name: str = "feature_generation_period_optimization_step"
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentResult:
    success: bool
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class BasePreTrainingComponent:
    """Very small base class with a uniform interface + helpers."""
    def __init__(self, config: Optional[ComponentConfig] = None):
        self.config = config or ComponentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> ComponentResult:
        raise NotImplementedError

    # Optional hooks many teams like
    def validate(self, training_input: Dict[str, Any]) -> None:
        pass

    def on_success(self, result: ComponentResult) -> None:
        pass

    def on_failure(self, result: ComponentResult) -> None:
        pass