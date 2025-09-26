"""Shared orchestration service interfaces for NAS/TAS pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import pandas as pd

from src.utils.nas_tas.model_manager import ModelManagerConfig
from src.utils.nas_tas.model_selector import (
    ModelSelectionConfig,
    ModelSelectionResult,
)
from src.utils.nas_tas.performance_tracker import PerformanceConfig
from src.utils.nas_tas.regime_aware_trainer import (
    RegimeAwareTrainingConfig,
    RegimeTrainingResult,
)


@runtime_checkable
class RegimeTrainerService(Protocol):
    """Protocol describing the interface required from regime trainers."""

    config: RegimeAwareTrainingConfig

    def train_models(
        self,
        market_data: pd.DataFrame,
        target_variable: str,
        feature_columns: Optional[list[str]] = None,
        timestamps: Optional[pd.Series] = None,
        unified_regime_output: Optional[Dict[str, Any]] = None,
    ) -> RegimeTrainingResult:
        """Train models for the supplied market data."""


@runtime_checkable
class ModelSelectorService(Protocol):
    """Protocol describing the interface required from model selectors."""

    config: ModelSelectionConfig

    def register_models(
        self,
        regime_models: Dict[int, Dict[str, Any]],
        ensemble_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register trained models for downstream selection."""

    def select_model(
        self,
        market_data: pd.DataFrame,
        current_regime: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelSelectionResult:
        """Select the best model for the provided market context."""


@runtime_checkable
class ModelManagerService(Protocol):
    """Protocol describing the interface required from model managers."""

    config: ModelManagerConfig

    def register_models(self, regime_models: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
        """Register trained models for lifecycle management."""

    def deploy_models(self) -> Dict[str, Any]:
        """Deploy managed models according to the configured strategy."""

    def setup_monitoring(self) -> Dict[str, Any]:
        """Prepare monitoring for deployed models."""


@runtime_checkable
class PerformanceTrackerService(Protocol):
    """Protocol describing the interface required from performance trackers."""

    config: PerformanceConfig

    def setup_model_tracking(self, model_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a model for ongoing performance tracking."""


@dataclass
class SharedOrchestrationServices:
    """Container bundling orchestration services for reuse across pipelines."""

    trainer: Optional[RegimeTrainerService] = None
    selector: Optional[ModelSelectorService] = None
    manager: Optional[ModelManagerService] = None
    performance_tracker: Optional[PerformanceTrackerService] = None

    def summary(self) -> Dict[str, bool]:
        """Return availability summary for the configured services."""

        return {
            "trainer": self.trainer is not None,
            "selector": self.selector is not None,
            "manager": self.manager is not None,
            "performance_tracker": self.performance_tracker is not None,
        }

    def with_updates(
        self,
        *,
        trainer: Optional[RegimeTrainerService] = None,
        selector: Optional[ModelSelectorService] = None,
        manager: Optional[ModelManagerService] = None,
        performance_tracker: Optional[PerformanceTrackerService] = None,
    ) -> "SharedOrchestrationServices":
        """Return a new dataclass instance with the provided overrides."""

        return SharedOrchestrationServices(
            trainer=trainer if trainer is not None else self.trainer,
            selector=selector if selector is not None else self.selector,
            manager=manager if manager is not None else self.manager,
            performance_tracker=(
                performance_tracker
                if performance_tracker is not None
                else self.performance_tracker
            ),
        )
