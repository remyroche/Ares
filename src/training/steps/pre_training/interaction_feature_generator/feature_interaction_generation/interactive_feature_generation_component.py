"""
Interactive Feature Generation Component

This component integrates the optimized interaction feature generation pipeline
with the pre-training sub_pipeline architecture. It provides a clean interface
that can be used by ares_launcher and maintains consistency with the existing
sub_pipeline structure.

Key Features:
- Consistent with sub_pipeline.py architecture
- Integrates with ares_launcher
- Uses optimized interaction orchestrator
- Maintains backward compatibility
- Extensive logging and error handling
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress
)
from src.training.steps.pre_training.validation.schemas import (
    SchemaValidationException,
    schema_metadata,
    validate_engineered_features,
)

# Import common operations and utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        optimize_memory_usage, parallel_processing_optimizer
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Common operations not available: {e}")
    COMMON_OPS_AVAILABLE = False

    def safe_divide(a, b):
        return a / b

    def safe_log(x):
        return np.log(x)

    def safe_sqrt(x):
        return np.sqrt(x)

    def safe_power(x, y):
        return np.power(x, y)

    def validate_finite(x):
        return np.isfinite(x).all()

    def get_m1_gpu_manager():
        return None

    def get_m1_memory_optimizer():
        return None

    def get_m1_cpu_optimizer():
        return None

    def optimize_memory_usage(*args, **kwargs):
        return None

    def parallel_processing_optimizer(*args, **kwargs):
        return None

# Import math validation
from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, validate_finite as math_validate_finite
)

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, safe_matrix_multiply,
        vectorized_rolling_features, parallel_feature_engineering,
        optimize_dataframe, get_hardware_performance_report
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.cross_validation import PurgedKFold
    from src.utils.ml_common.feature_selection import FeatureSelector
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# Import data utilities
try:
    from src.utils.data.data_loader import DataLoader
    from src.utils.data.data_validation import DataValidator
    from src.utils.kline_parquet import KlineParquetLoader
    from src.utils.serialization_utils import save_pickle, load_pickle
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Data utilities not available: {e}")
    DATA_UTILS_AVAILABLE = False

# Import the optimized orchestrator
from .optimized_interaction_orchestrator import (
    OptimizedInteractionOrchestrator, OptimizedInteractionConfig, OptimizedInteractionResult
)

# Import sub_pipeline components for compatibility
try:
    from ..components.base_component import BaseComponent, ComponentResult
    from ..components.component_factory import ComponentFactory
except ImportError:
    tprint_warning(
        "Component subsystem not available; using lightweight stubs for tests"
    )

    class ComponentResult:  # type: ignore
        pass

    class BaseComponent:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def run(self, *args, **kwargs):  # pragma: no cover - stub
            return None

    class ComponentFactory:  # type: ignore
        @staticmethod
        def create(*args, **kwargs):  # pragma: no cover - stub
            return BaseComponent(*args, **kwargs)

# Setup logging
logger = logging.getLogger(__name__)


class InteractiveFeatureGenerationStatus(Enum):
    """Status of interactive feature generation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class InteractiveFeatureGenerationConfig:
    """Configuration for interactive feature generation component."""
    # Basic configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = "historical_data"
    
    # Feature generation configuration
    feature_budget_pre: int = 120
    feature_budget_post: Tuple[int, int] = (30, 60)
    interactions_cap: int = 15
    transforms_per_parent: int = 1
    lookback_ceiling_minutes: int = 118
    latency_budget_ms: int = 50
    
    # Optimization configuration
    enable_matrix_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    
    # Validation configuration
    enable_validation: bool = True
    validation_threshold: float = 0.02
    
    # Logging configuration
    verbose_logging: bool = True
    log_performance: bool = True
    
    # Integration configuration
    integrate_with_ares_launcher: bool = True
    maintain_backward_compatibility: bool = True


@dataclass
class InteractiveFeatureGenerationResult:
    """Result of interactive feature generation."""
    # Core results
    features: pd.DataFrame
    feature_names: List[str]
    selected_features: List[str]
    interaction_features: pd.DataFrame
    cross_timeframe_features: pd.DataFrame
    
    # Pipeline metadata
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Artifacts for downstream components
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Backward compatibility
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InteractiveFeatureGenerationComponent(BaseComponent):
    """
    Interactive Feature Generation Component for Pre-Training Pipeline.
    
    This component integrates the optimized interaction feature generation
    pipeline with the pre-training sub_pipeline architecture.
    """
    
    def __init__(self, config: Optional[InteractiveFeatureGenerationConfig] = None):
        """Initialize the interactive feature generation component."""
        super().__init__()
        
        self.config = config or InteractiveFeatureGenerationConfig()
        self.logger = logger.getChild('InteractiveFeatureGenerationComponent')
        
        # Initialize the optimized orchestrator
        self._initialize_orchestrator()
        
        # Performance tracking
        self.performance_metrics = {}
        
        tprint_success("🚀 Interactive Feature Generation Component initialized")
        tprint_info(f"📊 Symbol: {self.config.symbol}, Exchange: {self.config.exchange}")
        tprint_info(f"⏰ Timeframe: {self.config.timeframe}")
        tprint_info(f"🔧 Matrix ops: {MATRIX_OPS_AVAILABLE}, ML common: {ML_COMMON_AVAILABLE}")
    
    def _initialize_orchestrator(self):
        """Initialize the optimized interaction orchestrator."""
        tprint_debug("🔧 Initializing optimized interaction orchestrator...")
        
        # Convert config to orchestrator config
        orchestrator_config = OptimizedInteractionConfig(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            data_dir=self.config.data_dir,
            feature_budget_pre=self.config.feature_budget_pre,
            feature_budget_post=self.config.feature_budget_post,
            interactions_cap=self.config.interactions_cap,
            transforms_per_parent=self.config.transforms_per_parent,
            lookback_ceiling_minutes=self.config.lookback_ceiling_minutes,
            latency_budget_ms=self.config.latency_budget_ms,
            enable_matrix_optimization=self.config.enable_matrix_optimization,
            enable_hardware_optimization=self.config.enable_hardware_optimization,
            enable_parallel_processing=self.config.enable_parallel_processing,
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            enable_validation=self.config.enable_validation,
            validation_threshold=self.config.validation_threshold,
            verbose_logging=self.config.verbose_logging,
            log_performance=self.config.log_performance
        )
        
        self.orchestrator = OptimizedInteractionOrchestrator(orchestrator_config)
        tprint_debug("✅ Optimized interaction orchestrator initialized")
    
    async def execute(self,
                     training_input: Dict[str, Any],
                     pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the interactive feature generation component.
        
        Args:
            training_input: Input data for feature generation
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with generated features
        """
        start_time = time.time()
        tprint_success("🚀 Starting interactive feature generation")
        validation_metadata: Dict[str, Dict[str, Optional[Dict[str, str]]]] = {
            'inputs': {},
            'outputs': {},
            'derived': {},
        }

        try:
            # Validate inputs
            self._validate_inputs(training_input, pipeline_state)

            # Extract data
            data = training_input.get('data')
            if data is None:
                raise ValueError("No data provided in training input")

            data = validate_engineered_features(
                data,
                context="interactive_feature_generation.input_features"
            )
            validation_metadata['inputs']['feature_matrix'] = schema_metadata('engineered_features').get('engineered_features')

            tprint_info(f"📊 Processing data: {data.shape[0]} rows, {data.shape[1]} columns")

            # Update orchestrator config with pipeline state
            self._update_orchestrator_config(pipeline_state)

            # Execute feature generation
            tprint_info("🔧 Executing optimized interaction feature generation...")
            result = await self.orchestrator.generate_features(training_input, pipeline_state)

            if not result.success:
                raise RuntimeError(f"Feature generation failed: {result.error_message}")

            if isinstance(result.features, pd.DataFrame) and not result.features.empty:
                result.features = validate_engineered_features(
                    result.features,
                    context="interactive_feature_generation.generated_features"
                )
                validation_metadata['outputs']['features'] = schema_metadata('engineered_features').get('engineered_features')

            if isinstance(result.interaction_features, pd.DataFrame) and not result.interaction_features.empty:
                result.interaction_features = validate_engineered_features(
                    result.interaction_features,
                    context="interactive_feature_generation.interaction_features"
                )
                validation_metadata['outputs']['interaction_features'] = schema_metadata('engineered_features').get('engineered_features')

            if isinstance(result.cross_timeframe_features, pd.DataFrame) and not result.cross_timeframe_features.empty:
                result.cross_timeframe_features = validate_engineered_features(
                    result.cross_timeframe_features,
                    context="interactive_feature_generation.cross_timeframe_features"
                )
                validation_metadata['outputs']['cross_timeframe_features'] = schema_metadata('engineered_features').get('engineered_features')

            # Convert result to component result format
            component_result = self._convert_to_component_result(result, start_time, validation_metadata)

            # Log success
            tprint_success("✅ Interactive feature generation completed successfully")
            tprint_info(f"📊 Generated {len(result.feature_names)} total features")
            tprint_info(f"🎯 Selected {len(result.selected_features)} features")
            tprint_info(f"🔗 Generated {len(result.interaction_features.columns)} interactions")
            tprint_info(f"⏰ Generated {len(result.cross_timeframe_features.columns)} cross-timeframe features")
            tprint_info(f"💾 Memory usage: {result.memory_usage_mb:.2f} MB")
            tprint_info(f"⏱️ Execution time: {result.execution_time:.3f}s")
            
            return component_result
            
        except SchemaValidationException as schema_error:
            execution_time = time.time() - start_time
            error_message = str(schema_error)
            tprint_error(f"❌ {error_message}")
            self.logger.error(f"Interactive feature generation schema error: {error_message}")
            return ComponentResult(
                success=False,
                error_message=error_message,
                artifacts={},
                execution_time=execution_time,
                metadata={
                    'schema_error': {
                        'schema_key': schema_error.schema_key,
                        'context': schema_error.context,
                        'schema_metadata': schema_metadata(schema_error.schema_key).get(schema_error.schema_key)
                    }
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Interactive feature generation failed: {str(e)}"

            tprint_error(f"❌ {error_message}")
            self.logger.error(f"Interactive feature generation failed: {error_message}", exc_info=True)

            return ComponentResult(
                success=False,
                error_message=error_message,
                artifacts={},
                execution_time=execution_time
            )
    
    def _validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Validate input data and pipeline state."""
        tprint_debug("🔍 Validating inputs...")
        
        if not training_input:
            raise ValueError("No training input provided")
        
        if not pipeline_state:
            raise ValueError("No pipeline state provided")
        
        # Check for required data
        data = training_input.get('data')
        if data is None:
            raise ValueError("No data provided in training input")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} < 100 rows")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        tprint_debug("✅ Input validation passed")
    
    def _update_orchestrator_config(self, pipeline_state: Dict[str, Any]) -> None:
        """Update orchestrator configuration with pipeline state."""
        tprint_debug("🔧 Updating orchestrator configuration...")
        
        # Update symbol and exchange if provided
        if 'symbol' in pipeline_state:
            self.config.symbol = pipeline_state['symbol']
            self.orchestrator.config.symbol = pipeline_state['symbol']
        
        if 'exchange' in pipeline_state:
            self.config.exchange = pipeline_state['exchange']
            self.orchestrator.config.exchange = pipeline_state['exchange']
        
        if 'timeframe' in pipeline_state:
            self.config.timeframe = pipeline_state['timeframe']
            self.orchestrator.config.timeframe = pipeline_state['timeframe']
        
        # Update data directory if provided
        if 'data_dir' in pipeline_state:
            self.config.data_dir = pipeline_state['data_dir']
            self.orchestrator.config.data_dir = pipeline_state['data_dir']
        
        tprint_debug("✅ Orchestrator configuration updated")
    
    def _convert_to_component_result(self,
                                   result: OptimizedInteractionResult,
                                   start_time: float,
                                   validation_metadata: Dict[str, Dict[str, Optional[Dict[str, str]]]]) -> ComponentResult:
        """Convert orchestrator result to component result format."""
        tprint_debug("🔄 Converting result to component format...")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create artifacts
        artifacts = {
            'interactive_feature_generation_result': {
                'features': result.features,
                'feature_names': result.feature_names,
                'selected_features': result.selected_features,
                'interaction_features': result.interaction_features,
                'cross_timeframe_features': result.cross_timeframe_features,
                'execution_time': result.execution_time,
                'memory_usage_mb': result.memory_usage_mb,
                'success': result.success,
                'error_message': result.error_message,
                'validated_schemas': validation_metadata
            },
            'stage_results': result.stage_results,
            'performance_metrics': result.performance_metrics,
            'artifacts': result.artifacts
        }
        artifacts.setdefault('validated_schemas', validation_metadata)

        # Create output files list (for backward compatibility)
        output_files = []
        if result.success:
            # Add feature files to output list
            output_files.append(f"features_{self.config.symbol}_{self.config.timeframe}.parquet")
            output_files.append(f"interactions_{self.config.symbol}_{self.config.timeframe}.parquet")
            output_files.append(f"cross_timeframe_{self.config.symbol}_{self.config.timeframe}.parquet")
        
        # Create metadata
        metadata = {
            'component_type': 'interactive_feature_generation',
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe,
            'total_features': len(result.feature_names),
            'selected_features': len(result.selected_features),
            'interaction_features': len(result.interaction_features.columns),
            'cross_timeframe_features': len(result.cross_timeframe_features.columns),
            'execution_time': result.execution_time,
            'memory_usage_mb': result.memory_usage_mb,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'ml_common_available': ML_COMMON_AVAILABLE,
            'data_utils_available': DATA_UTILS_AVAILABLE,
            'validated_schemas': validation_metadata
        }

        tprint_debug("✅ Result conversion completed")

        return ComponentResult(
            success=result.success,
            error_message=result.error_message,
            artifacts=artifacts,
            execution_time=execution_time,
            output_files=output_files,
            metadata=metadata
        )
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'name': 'interactive_feature_generation',
            'description': 'Optimized interaction feature generation with matrix operations and hardware acceleration',
            'version': '1.0.0',
            'dependencies': [
                'src.utils.tprint',
                'src.utils.common_operations',
                'src.utils.math_validation',
                'src.utils.matrix_operations',
                'src.utils.ml_common',
                'src.utils.data'
            ],
            'config': {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'feature_budget_pre': self.config.feature_budget_pre,
                'interactions_cap': self.config.interactions_cap,
                'enable_matrix_optimization': self.config.enable_matrix_optimization,
                'enable_hardware_optimization': self.config.enable_hardware_optimization
            },
            'capabilities': [
                'Parent feature generation',
                'Lookback optimization',
                'Transform application',
                'Interaction generation',
                'Cross-timeframe features',
                'Matrix operations optimization',
                'Hardware acceleration',
                'Feature selection'
            ]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics
    
    def cleanup(self):
        """Cleanup resources."""
        tprint_debug("🧹 Cleaning up interactive feature generation component...")
        
        # Cleanup orchestrator resources
        if hasattr(self.orchestrator, 'cleanup'):
            self.orchestrator.cleanup()
        
        # Clear performance metrics
        self.performance_metrics.clear()
        
        tprint_debug("✅ Cleanup completed")


# Factory function for component creation
def create_interactive_feature_generation_component(
    config: Optional[InteractiveFeatureGenerationConfig] = None
) -> InteractiveFeatureGenerationComponent:
    """
    Create an interactive feature generation component.
    
    Args:
        config: Configuration for the component
        
    Returns:
        InteractiveFeatureGenerationComponent instance
    """
    return InteractiveFeatureGenerationComponent(config)


# Integration with component factory
def register_interactive_feature_generation_component():
    """Register the interactive feature generation component with the factory."""
    try:
        from ..components.component_factory import ComponentFactory
        
        # Register the component
        ComponentFactory.register_component(
            'interactive_feature_generation',
            create_interactive_feature_generation_component
        )
        
        tprint_success("✅ Interactive feature generation component registered with factory")
        
    except ImportError as e:
        tprint_warning(f"⚠️ Could not register component with factory: {e}")


# Convenience function for direct execution
async def execute_interactive_feature_generation(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
    config: Optional[InteractiveFeatureGenerationConfig] = None
) -> ComponentResult:
    """
    Execute interactive feature generation with the given configuration.
    
    Args:
        training_input: Input data for feature generation
        pipeline_state: Current pipeline state
        config: Configuration for feature generation
        
    Returns:
        ComponentResult with generated features
    """
    component = create_interactive_feature_generation_component(config)
    return await component.execute(training_input, pipeline_state)


# Auto-register component on import
if __name__ != "__main__":
    register_interactive_feature_generation_component()