"""Wrapper registration for the multi-horizon profit labeler component."""

from __future__ import annotations

from typing import Any, Optional

from src.utils.logger import system_logger

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from .component_factory import register_component, register_unavailable_component
from .contracts import MultiHorizonArtifacts, PipelineState
from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging

module_logger = system_logger.getChild('MultiHorizonComponent')
module_event_logger = PreTrainingEventLogger(configure_pre_training_logging())

try:
    from ..multi_horizon_profit_labeler import MultiHorizonProfitLabelerComponent

    MULTI_HORIZON_AVAILABLE = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    MULTI_HORIZON_AVAILABLE = False
    _IMPORT_ERROR = exc


if not MULTI_HORIZON_AVAILABLE:
    message = (
        "Multi-horizon profit labeler unavailable: "
        f"{_IMPORT_ERROR}" if _IMPORT_ERROR else "Unknown import error"
    )
    module_logger.warning(message)
    module_event_logger.warning(
        "Multi-horizon profit labeler unavailable",
        context={'error': str(_IMPORT_ERROR) if _IMPORT_ERROR else 'unknown', 'step': 'component.multi_horizon_import'},
    )
    register_unavailable_component(
        'multi_horizon_profit_labeler',
        error=str(_IMPORT_ERROR) if _IMPORT_ERROR else 'Unknown import error',
        source=__name__,
    )
else:

    @register_component('multi_horizon_profit_labeler')
    class MultiHorizonComponentWrapper(BasePreTrainingComponent):
        """Wrapper for Multi-Horizon Profit Labeler to work as a factory component."""

        def __init__(self, config: Optional[ComponentConfig] = None):
            super().__init__(config)
            self.component: Optional[MultiHorizonProfitLabelerComponent] = None
            self._log_info(
                "🧩 [MULTI_HORIZON_WRAPPER] Initialized wrapper for multi-horizon component",
                event='multi_horizon_wrapper_init',
            )

        def get_required_artifacts(self) -> list[str]:
            """Get list of required artifacts this component must produce."""

            self._log_info(
                "📦 [MULTI_HORIZON_WRAPPER] Retrieving required artifacts",
                event='multi_horizon_wrapper_required_artifacts',
            )
            return ['multi_horizon_labeling_result', 'labeling_report']

        async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
            """Execute multi-horizon labeling as a component."""

            pipeline_state = PipelineState.ensure(pipeline_state)

            if self.component is None:
                try:
                    self.component = MultiHorizonProfitLabelerComponent(self.config)
                    self._log_info(
                        "🛠️ [MULTI_HORIZON_WRAPPER] Instantiated multi-horizon component",
                        event='multi_horizon_wrapper_component_initialized',
                    )
                except Exception as exc:
                    self._log_warning(
                        f"⚠️ [MULTI_HORIZON_WRAPPER] Failed to initialize component: {exc}",
                        event='multi_horizon_wrapper_component_init_failed',
                        error=str(exc),
                    )
                    return ComponentResult(
                        success=False,
                        artifacts=MultiHorizonArtifacts(),
                        error_message=str(exc),
                        metadata={'component_type': 'multi_horizon_profit_labeler'},
                    )

            try:
                result = await self.component.execute(data, pipeline_state)
                self._log_success(
                    "✅ [MULTI_HORIZON_WRAPPER] Execution completed successfully",
                    event='multi_horizon_wrapper_execute_success',
                )
                return result
            except Exception as exc:  # pragma: no cover - runtime safety
                self._log_warning(
                    f"⚠️ [MULTI_HORIZON_WRAPPER] Execution failed: {exc}",
                    event='multi_horizon_wrapper_execute_failed',
                    error=str(exc),
                )
                return ComponentResult(
                    success=False,
                    artifacts=MultiHorizonArtifacts(),
                    error_message=str(exc),
                    metadata={'component_type': 'multi_horizon_profit_labeler'},
                )
