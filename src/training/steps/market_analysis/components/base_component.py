"""Core abstractions for market analysis pipeline components.

The previous implementation of this module attempted to wrap almost every
utility in the code base which resulted in thousands of lines of tightly
coupled code, import cycles and silent failure paths.  The refactored module
focuses on three responsibilities:

* Provide light-weight data containers for component configuration and
  execution results with sensible validation.
* Define a small asynchronous execution contract for pipeline components.
* Handle error reporting and optional artifact persistence without masking
  failures.

The new implementation removes hard dependencies on optional subsystems and
keeps the public API that other components rely on (`ComponentConfig`,
`ComponentResult` and `BaseMarketAnalysisComponent`).  It deliberately raises
exceptions when critical steps fail so that pipeline orchestration layers can
react immediately instead of continuing with corrupted state.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

try:  # Pandas is optional during tests but we avoid a hard dependency here.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is not required for unit tests.
    pd = None  # type: ignore

from .artifact_manager import ArtifactManager


class ComponentConfigurationError(ValueError):
    """Raised when an invalid configuration is supplied to a component."""


class ComponentExecutionError(RuntimeError):
    """Raised when `BaseMarketAnalysisComponent.run` fails irrecoverably."""


@dataclass
class ComponentConfig:
    """Configuration shared by the majority of market analysis components."""

    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    data_dir: str = "historical_data"
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    max_retry_attempts: int = 1
    retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ComponentConfigurationError("`symbol` must be a non-empty string")
        if not self.exchange:
            raise ComponentConfigurationError("`exchange` must be a non-empty string")
        if not self.timeframe:
            raise ComponentConfigurationError("`timeframe` must be a non-empty string")
        if self.max_retry_attempts < 0:
            raise ComponentConfigurationError("`max_retry_attempts` cannot be negative")
        if self.retry_delay_seconds < 0:
            raise ComponentConfigurationError("`retry_delay_seconds` cannot be negative")


@dataclass
class ComponentResult:
    """Container describing the outcome of a component execution."""

    success: bool
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_performance_metric(self, name: str, value: float) -> None:
        self.performance_metrics[name] = float(value)


class BaseMarketAnalysisComponent(ABC):
    """Base class for all market analysis pipeline components."""

    def __init__(
        self,
        config: Optional[ComponentConfig] = None,
        *,
        artifact_manager: Optional[ArtifactManager] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or ComponentConfig()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.artifact_manager = artifact_manager or ArtifactManager()
        self.performance_metrics: Dict[str, float] = {}

        # Stash basic execution timing state.
        self._execution_started_at: Optional[datetime] = None

    @property
    def component_name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Hooks that subclasses may override
    # ------------------------------------------------------------------
    def get_required_artifacts(self) -> Iterable[str]:
        """Return an iterable of artifact names that must be produced."""
        return ()

    def validate_input(self, data: Any) -> None:
        """Validate the incoming data before executing the component.

        The default implementation only checks basic Pandas DataFrame inputs
        when pandas is available.  Sub-classes are encouraged to override this
        method with domain specific validation.
        """

        if pd is None or data is None:
            return

        if not isinstance(data, pd.DataFrame):  # type: ignore[attr-defined]
            raise ComponentExecutionError(
                "Input data must be a pandas.DataFrame when pandas is available"
            )
        if data.empty:
            raise ComponentExecutionError("Input DataFrame is empty")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(
        self,
        data: Any,
        pipeline_state: Optional[Dict[str, Any]] = None,
    ) -> ComponentResult:
        """Execute the component with validation, timing and persistence.

        Any exception raised by :meth:`execute` or the validation steps will be
        wrapped in :class:`ComponentExecutionError`.  The caller is expected to
        handle this exception; we intentionally do not hide failures behind
        boolean return codes anymore.
        """

        pipeline_state = pipeline_state or {}
        self.validate_input(data)
        self._execution_started_at = datetime.utcnow()

        try:
            result = await self._execute_with_optional_async(data, pipeline_state)
        except Exception as exc:  # pragma: no cover - exercised in integration tests
            self.logger.exception("Component execution failed")
            raise ComponentExecutionError(str(exc)) from exc

        self._validate_result(result)
        await self._persist_artifacts_if_requested(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _execute_with_optional_async(
        self, data: Any, pipeline_state: Dict[str, Any]
    ) -> ComponentResult:
        """Execute the component and allow synchronous overrides for tests."""

        maybe_coro = self.execute(data, pipeline_state)
        if asyncio.iscoroutine(maybe_coro):
            return await maybe_coro  # type: ignore[return-value]
        # Fallback to synchronous execution for older components.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: maybe_coro)  # type: ignore[arg-type]

    async def _persist_artifacts_if_requested(self, result: ComponentResult) -> None:
        if not self.artifact_manager or not result.success or not result.artifacts:
            return

        loop = asyncio.get_running_loop()
        try:
            saved_paths = await loop.run_in_executor(
                None,
                lambda: self.artifact_manager.save_artifacts(
                    self.component_name, result.artifacts, result.metadata
                ),
            )
            result.metadata.setdefault("artifact_paths", saved_paths)
        except Exception as exc:  # pragma: no cover - depends on filesystem
            self.logger.exception("Failed to persist artifacts")
            raise ComponentExecutionError(f"Artifact persistence failed: {exc}") from exc

    def _validate_result(self, result: ComponentResult) -> None:
        if not isinstance(result, ComponentResult):
            raise ComponentExecutionError(
                "Component implementations must return a ComponentResult instance"
            )

        missing = [
            name for name in self.get_required_artifacts() if name not in result.artifacts
        ]
        if missing:
            raise ComponentExecutionError(
                f"Component did not produce required artifacts: {', '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abstractmethod
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Component specific execution logic."""


__all__ = [
    "ArtifactManager",
    "BaseMarketAnalysisComponent",
    "ComponentConfig",
    "ComponentExecutionError",
    "ComponentConfigurationError",
    "ComponentResult",
]
