"""Wrapper registration for the multi-horizon profit labeler component."""

from __future__ import annotations

from typing import Any, Optional

from src.utils.logger import system_logger
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import validate_finite
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    safe_matrix_multiply,
    optimize_dataframe,
    matrix_correlation_analysis
)
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error

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

            # Initialize utility managers
            self.common_utils = CommonUtilities()
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.gpu_manager = M1GPUManager()
            self.data_manager = KlinesParquetManager()

            # Initialize matrix operations managers
            tprint("🔢 Initializing matrix operations managers for multi-horizon component...")
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            tprint_success("✅ Matrix operations managers initialized for multi-horizon component")

            self._log_info(
                "🧩 [MULTI_HORIZON_WRAPPER] Initialized wrapper for multi-horizon component with utility and matrix operation managers",
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

        # Utility methods for enhanced functionality

        def validate_finite_values(self, value, name: str = "value"):
            """Validate that values are finite using math validation utilities."""
            return validate_finite(value, name)

        def safe_dataframe_operation(self, df, operation, *args, **kwargs):
            """Safely perform DataFrame operations with error handling."""
            return safe_dataframe_operation(df, operation, *args, **kwargs)

        def serialize_results_json(self, results, filepath: str) -> bool:
            """Serialize results to JSON format."""
            tprint(f"💾 Serializing multi-horizon results to {filepath}")
            try:
                return self.json_serializer.save(results, filepath)
            except Exception as e:
                tprint_error(f"❌ Failed to serialize multi-horizon results: {e}")
                self.logger.error(f"Failed to serialize results: {e}")
                return False

        def deserialize_results_json(self, filepath):
            """Deserialize results from JSON format."""
            tprint(f"📖 Deserializing multi-horizon results from {filepath}")
            try:
                return self.json_serializer.load(filepath)
            except Exception as e:
                tprint_error(f"❌ Failed to deserialize multi-horizon results: {e}")
                self.logger.error(f"Failed to deserialize results: {e}")
                return None

        def is_hardware_accelerated(self) -> bool:
            """Check if hardware acceleration is available."""
            return self.gpu_manager.is_m1 if self.gpu_manager else False

        def load_klines_data(self, symbol: str, timeframe: str, start_date=None, end_date=None):
            """Load klines data using the data manager."""
            if self.data_manager:
                return self.data_manager.load_symbol_data(
                    symbol, timeframe, start_date, end_date
                )
            return None

        def safe_matrix_multiply(self, A, B):
            """Safely perform matrix multiplication with error handling."""
            tprint(f"🔢 Performing safe matrix multiplication in multi-horizon component ({A.shape} x {B.shape})")
            return safe_matrix_multiply(A, B)

        def optimize_dataframe_for_matrix_ops(self, df):
            """Optimize DataFrame for matrix operations."""
            tprint(f"⚡ Optimizing DataFrame for matrix operations in multi-horizon component (shape: {df.shape})")
            return optimize_dataframe(df)

        def compute_matrix_correlation_analysis(self, data):
            """Compute matrix correlation analysis."""
            tprint(f"📊 Computing matrix correlation analysis in multi-horizon component (shape: {data.shape})")
            return matrix_correlation_analysis(data)

        def perform_vectorized_matrix_ops(self, data, operations):
            """Perform vectorized matrix operations using the vectorized core."""
            tprint(f"🚀 Performing vectorized matrix operations in multi-horizon component (shape: {data.shape})")
            if self.vectorized_core:
                return self.vectorized_core.optimize_dataframe_for_processing(data)
            return data
