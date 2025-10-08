"""Wrapper registration for the multi-horizon profit labeler component."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.tprint import tprint, tprint_warning

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from .component_factory import register_component, register_unavailable_component

try:
    from ..multi_horizon_profit_labeler import MultiHorizonProfitLabelerComponent

    MULTI_HORIZON_AVAILABLE = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    MULTI_HORIZON_AVAILABLE = False
    _IMPORT_ERROR = exc


if not MULTI_HORIZON_AVAILABLE:
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
            tprint(
                "🧩 [MULTI_HORIZON_WRAPPER] Initialized wrapper for multi-horizon component",
                color="blue",
            )

        def get_required_artifacts(self) -> list[str]:
            """Get list of required artifacts this component must produce."""

            tprint(
                "📦 [MULTI_HORIZON_WRAPPER] Retrieving required artifacts",
                color="magenta",
            )
            return ['multi_horizon_labeling_result', 'labeling_report']

        async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
            """Execute multi-horizon labeling as a component."""

            if self.component is None:
                try:
                    self.component = MultiHorizonProfitLabelerComponent(self.config)
                    tprint(
                        "🛠️ [MULTI_HORIZON_WRAPPER] Instantiated multi-horizon component",
                        color="yellow",
                    )
                except Exception as exc:
                    tprint_warning(
                        f"⚠️ [MULTI_HORIZON_WRAPPER] Failed to initialize component: {exc}",
                        color="red",
                    )
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message=str(exc),
                        metadata={'component_type': 'multi_horizon_profit_labeler'},
                    )

            try:
                result = await self.component.execute(data, pipeline_state)
                tprint(
                    "✅ [MULTI_HORIZON_WRAPPER] Execution completed successfully",
                    color="green",
                )
                return result
            except Exception as exc:  # pragma: no cover - runtime safety
                tprint_warning(
                    f"⚠️ [MULTI_HORIZON_WRAPPER] Execution failed: {exc}",
                    color="red",
                )
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=str(exc),
                    metadata={'component_type': 'multi_horizon_profit_labeler'},
                )
