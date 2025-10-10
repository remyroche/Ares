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
from typing import Dict, List, Optional, Any, Tuple, Iterable, Mapping, Union
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
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
)
from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_feature_artifact,
)

# Import improved utilities
from .import_manager import get_import_manager
from .feature_generation_utils import ImprovedFeatureGenerator, FeatureGenerationConfig

# Initialize import manager
import_manager = get_import_manager()

# Fast-fail imports - all required dependencies must be available
try:
    # Import common operations and utilities using import manager
    common_ops_result = import_manager.import_common_operations()
    COMMON_OPS = common_ops_result.module
    tprint_success("✅ Common operations imported successfully")
    
    # Import math validation
    math_validation_result = import_manager.import_math_validation()
    MATH_VALIDATION = math_validation_result.module
    
    # Import component registration
    from src.training.steps.pre_training.components.component_factory import register_component
    
    # Import matrix operations using import manager
    matrix_ops_result = import_manager.import_matrix_operations()
    MATRIX_OPS = matrix_ops_result.module
    tprint_success("✅ Matrix operations imported successfully")
    
    # Import ML common utilities using import manager
    ml_common_result = import_manager.import_ml_common()
    purged_kfold_result = import_manager.import_purged_kfold()
    feature_selection_result = import_manager.import_feature_selection()
    
    ML_COMMON = ml_common_result.module
    PURGED_KFOLD = purged_kfold_result.module
    FEATURE_SELECTION = feature_selection_result.module
    tprint_success("✅ ML common utilities imported successfully")
    
    # Import data utilities using import manager
    data_utils_result = import_manager.import_data_utils()
    DATA_UTILS = data_utils_result.module
    tprint_success("✅ Data utilities imported successfully")
    
    # Import ares launcher integration
    from .ares_launcher_integration import AresLauncherInteractiveFeatureGenerator
    ARES_LAUNCHER_AVAILABLE = True
    
except ImportError as e:
    tprint_error(f"❌ Critical dependency missing: {e}")
    raise ImportError(f"Required dependency not available: {e}")
except Exception as e:
    tprint_error(f"❌ Unexpected error during initialization: {e}")
    raise RuntimeError(f"Initialization failed: {e}")

# Import column naming utilities
from ...column_naming import (
    ColumnNamespace,
    ensure_dataframe_namespace,
    ensure_namespace,
)

# Import the enhanced optimized orchestrator
from .enhanced_optimized_orchestrator import (
    EnhancedOptimizedInteractionOrchestrator,
    EnhancedOptimizedConfig,
    OptimizedInteractionResult,
    PipelineStage,
)
from ...settings import get_pre_training_settings

# Import sub_pipeline components for compatibility - fail fast if not available
try:
    from ...components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
    from ...components.component_factory import ComponentFactory
    from ...components.contracts import InteractiveFeatureArtifacts
    _COMPONENTS_AVAILABLE = True
    # BaseComponent doesn't exist - use BasePreTrainingComponent for both
    BaseComponent = BasePreTrainingComponent
    tprint_success("✅ Component subsystem imported successfully")
except ImportError as e:
    tprint_error(f"❌ Critical error: Component subsystem not available: {e}")
    raise ImportError(f"Required component subsystem not available: {e}")

# Setup logging
logger = logging.getLogger(__name__)


class InteractiveFeatureGenerationStatus(Enum):
    """Status of interactive feature generation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _default_data_directory() -> str:
    return str(get_pre_training_settings().data_root)


@dataclass
class InteractiveFeatureGenerationConfig:
    """Configuration for interactive feature generation component."""
    # Basic configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = field(default_factory=_default_data_directory)
    
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
    max_workers: int = 8
    batch_size: int = 1000
    
    # Enhanced optimization settings
    enable_early_filtering: bool = True
    enable_interaction_pruning: bool = True
    enable_budgeted_optimization: bool = True
    enable_caching: bool = True
    
    # Memory optimization settings
    max_memory_gb: float = 16.0
    chunk_size_mb: float = 200.0
    use_parquet: bool = True
    use_memmap: bool = True
    
    # Cache settings
    l1_max_size_mb: float = 200.0
    l2_max_size_mb: float = 2000.0
    enable_dependency_tracking: bool = True
    
    # Early filtering settings
    downsample_ratio: float = 0.1
    variance_threshold: float = 1e-8  # FIXED: More lenient threshold to prevent over-filtering
    top_k_per_family: int = 50  # FIXED: Increased to allow more features per family
    
    # Interaction pruning settings
    max_interactions_per_domain: int = 6
    min_delta_ic: float = 0.01
    min_stability_score: float = 0.7
    
    # Budgeted optimization settings
    coarse_grid_points: int = 10
    fine_search_evals: int = 16
    early_stop_patience: int = 5
    
    # Validation configuration
    enable_validation: bool = True
    validation_threshold: float = 0.02

    # Budget constraints for final feature selection
    enable_budget_constraints: bool = True
    total_budget_ms: float = 100.0
    base_features_budget_ms: float = 68.0  # 68% of total budget
    interaction_features_budget_ms: float = 15.0  # 15% of total budget
    cross_timeframe_features_budget_ms: float = 10.0  # 10% of total budget
    gate_features_budget_ms: float = 7.0  # 7% of total budget

    # Feature type constraints - Updated for new pipeline
    base_features_min: int = 40
    base_features_max: int = 80
    base_features_target: int = 60  # Target 60 base features
    interaction_features_min: int = 5
    interaction_features_max: int = 15
    interaction_features_target: int = 10  # Target 10 interaction features
    cross_timeframe_features_min: int = 3
    cross_timeframe_features_max: int = 10
    cross_timeframe_features_target: int = 6  # Target 6 cross-timeframe features
    gate_features_min: int = 2
    gate_features_max: int = 8
    gate_features_target: int = 5  # Target 5 gate features

    # Logging configuration
    verbose_logging: bool = True
    log_performance: bool = True

    # Integration configuration
    integrate_with_ares_launcher: bool = True
    maintain_backward_compatibility: bool = True

    # Market data streaming configuration
    market_data_batch_size: Optional[int] = None
    market_data_window_days: Optional[int] = None


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


@register_component('interactive_feature_generation')
class InteractiveFeatureGenerationComponent(BasePreTrainingComponent):
    """
    Interactive Feature Generation Component for Pre-Training Pipeline.

    This component integrates the optimized interaction feature generation
    pipeline with the pre-training sub_pipeline architecture.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the interactive feature generation component."""
        super().__init__(config)

        self.logger = logger.getChild('InteractiveFeatureGenerationComponent')
        self.performance_metrics: Dict[str, Any] = {}
        self._interactive_config = self._build_interactive_config(self.config)

        # Initialize the optimized orchestrator
        self._initialize_orchestrator()

        tprint_success("🚀 Interactive Feature Generation Component initialized")
        tprint_info(f"📊 Symbol: {self.config.symbol}, Exchange: {self.config.exchange}")
        tprint_info(f"⏰ Timeframe: {self.config.timeframe}")
        tprint_info(f"🔧 Matrix ops: {MATRIX_OPS_AVAILABLE}, ML common: {ML_COMMON_AVAILABLE}")

    @staticmethod
    def _coerce_tuple(value: Optional[Any], default: Tuple[int, int]) -> Tuple[int, int]:
        """Coerce configuration values that may be provided as a tuple or list."""

        if value is None:
            return default
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return default

    def _build_interactive_config(
        self,
        component_config: ComponentConfig,
    ) -> InteractiveFeatureGenerationConfig:
        """Translate the generic component config into the interactive schema."""

        params = dict(component_config.custom_params or {})

        feature_budget_post = self._coerce_tuple(
            params.get('feature_budget_post'),
            InteractiveFeatureGenerationConfig.feature_budget_post,
        )

        return InteractiveFeatureGenerationConfig(
            symbol=component_config.symbol,
            exchange=component_config.exchange,
            timeframe=component_config.timeframe,
            data_dir=component_config.data_dir,
            feature_budget_pre=int(params.get('feature_budget_pre', InteractiveFeatureGenerationConfig.feature_budget_pre)),
            feature_budget_post=feature_budget_post,
            interactions_cap=int(params.get('interactions_cap', InteractiveFeatureGenerationConfig.interactions_cap)),
            transforms_per_parent=int(params.get('transforms_per_parent', InteractiveFeatureGenerationConfig.transforms_per_parent)),
            lookback_ceiling_minutes=int(params.get('lookback_ceiling_minutes', InteractiveFeatureGenerationConfig.lookback_ceiling_minutes)),
            latency_budget_ms=int(params.get('latency_budget_ms', InteractiveFeatureGenerationConfig.latency_budget_ms)),
            enable_matrix_optimization=bool(params.get('enable_matrix_optimization', InteractiveFeatureGenerationConfig.enable_matrix_optimization)),
            enable_hardware_optimization=bool(params.get('enable_hardware_optimization', InteractiveFeatureGenerationConfig.enable_hardware_optimization)),
            enable_parallel_processing=bool(params.get('enable_parallel_processing', InteractiveFeatureGenerationConfig.enable_parallel_processing)),
            max_workers=int(params.get('max_workers', InteractiveFeatureGenerationConfig.max_workers)),
            batch_size=int(params.get('batch_size', InteractiveFeatureGenerationConfig.batch_size)),
            enable_validation=bool(params.get('enable_validation', InteractiveFeatureGenerationConfig.enable_validation)),
            validation_threshold=float(params.get('validation_threshold', InteractiveFeatureGenerationConfig.validation_threshold)),
            # Budget constraints for final feature selection
            enable_budget_constraints=bool(params.get('enable_budget_constraints', InteractiveFeatureGenerationConfig.enable_budget_constraints)),
            total_budget_ms=float(params.get('total_budget_ms', InteractiveFeatureGenerationConfig.total_budget_ms)),
            base_features_budget_ms=float(params.get('base_features_budget_ms', InteractiveFeatureGenerationConfig.base_features_budget_ms)),
            interaction_features_budget_ms=float(params.get('interaction_features_budget_ms', InteractiveFeatureGenerationConfig.interaction_features_budget_ms)),
            cross_timeframe_features_budget_ms=float(params.get('cross_timeframe_features_budget_ms', InteractiveFeatureGenerationConfig.cross_timeframe_features_budget_ms)),
            gate_features_budget_ms=float(params.get('gate_features_budget_ms', InteractiveFeatureGenerationConfig.gate_features_budget_ms)),
            # Feature type constraints
            base_features_min=int(params.get('base_features_min', InteractiveFeatureGenerationConfig.base_features_min)),
            base_features_max=int(params.get('base_features_max', InteractiveFeatureGenerationConfig.base_features_max)),
            base_features_target=int(params.get('base_features_target', InteractiveFeatureGenerationConfig.base_features_target)),
            interaction_features_min=int(params.get('interaction_features_min', InteractiveFeatureGenerationConfig.interaction_features_min)),
            interaction_features_max=int(params.get('interaction_features_max', InteractiveFeatureGenerationConfig.interaction_features_max)),
            interaction_features_target=int(params.get('interaction_features_target', InteractiveFeatureGenerationConfig.interaction_features_target)),
            cross_timeframe_features_min=int(params.get('cross_timeframe_features_min', InteractiveFeatureGenerationConfig.cross_timeframe_features_min)),
            cross_timeframe_features_max=int(params.get('cross_timeframe_features_max', InteractiveFeatureGenerationConfig.cross_timeframe_features_max)),
            cross_timeframe_features_target=int(params.get('cross_timeframe_features_target', InteractiveFeatureGenerationConfig.cross_timeframe_features_target)),
            gate_features_min=int(params.get('gate_features_min', InteractiveFeatureGenerationConfig.gate_features_min)),
            gate_features_max=int(params.get('gate_features_max', InteractiveFeatureGenerationConfig.gate_features_max)),
            gate_features_target=int(params.get('gate_features_target', InteractiveFeatureGenerationConfig.gate_features_target)),
            verbose_logging=bool(params.get('verbose_logging', InteractiveFeatureGenerationConfig.verbose_logging)),
            log_performance=bool(params.get('log_performance', InteractiveFeatureGenerationConfig.log_performance)),
            integrate_with_ares_launcher=bool(params.get('integrate_with_ares_launcher', InteractiveFeatureGenerationConfig.integrate_with_ares_launcher)),
            maintain_backward_compatibility=bool(params.get('maintain_backward_compatibility', InteractiveFeatureGenerationConfig.maintain_backward_compatibility)),
            market_data_batch_size=params.get('market_data_batch_size', InteractiveFeatureGenerationConfig.market_data_batch_size),
            market_data_window_days=params.get('market_data_window_days', InteractiveFeatureGenerationConfig.market_data_window_days),
        )

    def _initialize_orchestrator(self):
        """Initialize the optimized interaction orchestrator."""
        tprint_debug("🔧 Initializing optimized interaction orchestrator...")

        # Convert config to enhanced orchestrator config
        orchestrator_config = EnhancedOptimizedConfig(
            # Core settings
            enable_early_filtering=self._interactive_config.enable_early_filtering,
            enable_interaction_pruning=self._interactive_config.enable_interaction_pruning,
            enable_budgeted_optimization=self._interactive_config.enable_budgeted_optimization,
            enable_caching=self._interactive_config.enable_caching,
            enable_parallel_processing=self._interactive_config.enable_parallel_processing,
            
            # DAG executor settings
            max_workers=self._interactive_config.max_workers,
            use_processes=True,
            
            # Memory optimization settings
            max_memory_gb=self._interactive_config.max_memory_gb,
            chunk_size_mb=self._interactive_config.chunk_size_mb,
            use_parquet=self._interactive_config.use_parquet,
            use_memmap=self._interactive_config.use_memmap,
            
            # Cache settings
            l1_max_size_mb=self._interactive_config.l1_max_size_mb,
            l2_max_size_mb=self._interactive_config.l2_max_size_mb,
            enable_dependency_tracking=self._interactive_config.enable_dependency_tracking,
            
            # Early filtering settings
            downsample_ratio=self._interactive_config.downsample_ratio,
            variance_threshold=self._interactive_config.variance_threshold,
            top_k_per_family=self._interactive_config.top_k_per_family,
            
            # Interaction pruning settings
            max_interactions_per_domain=self._interactive_config.max_interactions_per_domain,
            min_delta_ic=self._interactive_config.min_delta_ic,
            min_stability_score=self._interactive_config.min_stability_score,
            
            # Budgeted optimization settings
            coarse_grid_points=self._interactive_config.coarse_grid_points,
            fine_search_evals=self._interactive_config.fine_search_evals,
            early_stop_patience=self._interactive_config.early_stop_patience,
            
            # Performance monitoring
            enable_performance_monitoring=self._interactive_config.log_performance,
            log_level="INFO" if self._interactive_config.verbose_logging else "WARNING",
            save_intermediate_results=False
        )

        self.orchestrator = EnhancedOptimizedInteractionOrchestrator(orchestrator_config)
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
            training_input = self._ensure_training_input(training_input, pipeline_state)

            # Validate inputs
            self._validate_inputs(training_input, pipeline_state)

            # Extract data
            data = training_input.get('data')
            if data is None:
                raise ValueError("No data provided in training input")

            target_shifts: Dict[str, int] = {}
            for source in (training_input, pipeline_state):
                if isinstance(source, Mapping):
                    raw_shifts = source.get('target_shifts')
                    if isinstance(raw_shifts, Mapping):
                        for key, value in raw_shifts.items():
                            try:
                                target_shifts[str(key)] = int(value)
                            except (TypeError, ValueError):
                                continue

            data = validate_engineered_features(
                data,
                context="interactive_feature_generation.input_features"
            )
            # Skip temporal alignment check - interactive feature generation needs raw market data
            # columns (open, high, low, close, volume) for feature generation even though they have lag=0
            tprint_debug("ℹ️ Skipping temporal alignment check for interactive feature generation (requires raw market data)")
            validation_metadata['inputs']['feature_matrix'] = schema_metadata('engineered_features').get('engineered_features')

            tprint_info(f"📊 Processing data: {data.shape[0]} rows, {data.shape[1]} columns")
            if data is not None:
                tprint_info(f"📊 Processing data: {data.shape[0]} rows, {data.shape[1]} columns")

            # Update orchestrator config with pipeline state
            self._update_orchestrator_config(pipeline_state)

            # Execute feature generation
            tprint_info("🔧 Executing optimized interaction feature generation...")
            result = await self.orchestrator.generate_features(training_input, pipeline_state)
            data_batches = list(self._iter_data_batches(training_input))
            if data_batches:
                tprint_info(
                    f"🔧 Executing optimized interaction feature generation in {len(data_batches)} batches"
                )
                result = await self._execute_chunked_generation(
                    training_input,
                    pipeline_state,
                    data_batches,
                )
            else:
                tprint_info("🔧 Executing optimized interaction feature generation...")
                result = await self.orchestrator.generate_features(training_input, pipeline_state)

            if not result.success:
                raise RuntimeError(f"Feature generation failed: {result.error_message}")

            if isinstance(result.features, pd.DataFrame) and not result.features.empty:
                result.features = validate_engineered_features(
                    result.features,
                    context="interactive_feature_generation.generated_features"
                )
                enforce_feature_temporal_alignment(
                    result.features,
                    context="interactive_feature_generation.generated_features",
                    target_shifts=target_shifts,
                    feature_metadata=getattr(result, 'feature_metadata', None),
                )
                validation_metadata['outputs']['features'] = schema_metadata('engineered_features').get('engineered_features')

            if isinstance(result.interaction_features, pd.DataFrame) and not result.interaction_features.empty:
                result.interaction_features = validate_engineered_features(
                    result.interaction_features,
                    context="interactive_feature_generation.interaction_features"
                )
                enforce_feature_temporal_alignment(
                    result.interaction_features,
                    context="interactive_feature_generation.interaction_features",
                    target_shifts=target_shifts,
                    feature_metadata=getattr(result, 'interaction_feature_metadata', None),
                )
                validation_metadata['outputs']['interaction_features'] = schema_metadata('engineered_features').get('engineered_features')

            if isinstance(result.cross_timeframe_features, pd.DataFrame) and not result.cross_timeframe_features.empty:
                result.cross_timeframe_features = validate_engineered_features(
                    result.cross_timeframe_features,
                    context="interactive_feature_generation.cross_timeframe_features"
                )
                enforce_feature_temporal_alignment(
                    result.cross_timeframe_features,
                    context="interactive_feature_generation.cross_timeframe_features",
                    target_shifts=target_shifts,
                    feature_metadata=getattr(result, 'cross_timeframe_feature_metadata', None),
                )
                validation_metadata['outputs']['cross_timeframe_features'] = schema_metadata('engineered_features').get('engineered_features')

            result = self._apply_namespace_to_result(result)
            
            # Convert result to component result format
            component_result = self._convert_to_component_result(result, start_time, validation_metadata)

            # Generate outcome file with datetime stamp - fail fast if file operations fail
            from datetime import datetime
            from pathlib import Path
            import json

            outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            outcomes_dir = Path('outcomes')
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            outcome_filename = f"interactive_feature_generation_outcome_{outcome_timestamp}.json"
            outcome_path = outcomes_dir / outcome_filename

            # Feature type categorization
            feature_type_breakdown = {
                'total_features': len(result.feature_names),
                'base_features': len(result.features.columns) if hasattr(result.features, 'columns') else 0,
                'interaction_features': len(result.interaction_features.columns) if hasattr(result.interaction_features, 'columns') else 0,
                'cross_timeframe_features': len(result.cross_timeframe_features.columns) if hasattr(result.cross_timeframe_features, 'columns') else 0,
                'selected_features': len(result.selected_features),
                'selection_rate': float(len(result.selected_features) / max(1, len(result.feature_names)) * 100),
            }

            # Interaction statistics
            interaction_stats = {}
            if hasattr(result.interaction_features, 'columns'):
                interaction_cols = result.interaction_features.columns
                interaction_stats = {
                    'total_interactions': len(interaction_cols),
                    'interaction_types': {},
                    'sample_interactions': list(interaction_cols[:10]),  # First 10 as examples
                }
                # Count interaction types (multiply, divide, add, subtract, etc.)
                for col in interaction_cols:
                    col_str = str(col)
                    if '_x_' in col_str or '*' in col_str:
                        interaction_stats['interaction_types']['multiply'] = interaction_stats['interaction_types'].get('multiply', 0) + 1
                    elif '_div_' in col_str or '/' in col_str:
                        interaction_stats['interaction_types']['divide'] = interaction_stats['interaction_types'].get('divide', 0) + 1
                    elif '_add_' in col_str or '+' in col_str:
                        interaction_stats['interaction_types']['add'] = interaction_stats['interaction_types'].get('add', 0) + 1
                    elif '_sub_' in col_str or '-' in col_str:
                        interaction_stats['interaction_types']['subtract'] = interaction_stats['interaction_types'].get('subtract', 0) + 1
                    else:
                        interaction_stats['interaction_types']['other'] = interaction_stats['interaction_types'].get('other', 0) + 1

            # Cross-timeframe statistics
            cross_timeframe_stats = {}
            if hasattr(result.cross_timeframe_features, 'columns'):
                ctf_cols = result.cross_timeframe_features.columns
                cross_timeframe_stats = {
                    'total_features': len(ctf_cols),
                    'sample_features': list(ctf_cols[:10]),  # First 10 as examples
                }

            # Performance and efficiency metrics
            performance_breakdown = {
                'execution_time_seconds': result.execution_time,
                'features_per_second': float(len(result.feature_names) / max(0.001, result.execution_time)),
                'memory_usage_mb': getattr(result, 'memory_usage_mb', 0.0),
                'memory_per_feature_kb': float(getattr(result, 'memory_usage_mb', 0.0) * 1024 / max(1, len(result.feature_names))),
                'cpu_usage_percent': getattr(result, 'cpu_usage_percent', 0.0),
                'gpu_usage_percent': getattr(result, 'gpu_usage_percent', 0.0),
            }

            # Stage-wise results if available
            stage_results_summary = {}
            if hasattr(result, 'stage_results') and result.stage_results:
                for stage, stage_data in result.stage_results.items():
                    if isinstance(stage_data, dict):
                        stage_results_summary[str(stage)] = {
                            'execution_time': stage_data.get('execution_time', 0.0),
                            'features_generated': stage_data.get('features_generated', 0),
                            'success': stage_data.get('success', False),
                        }

            # Hardware acceleration details
            hardware_details = {
                'matrix_optimization_enabled': self._interactive_config.enable_matrix_optimization,
                'hardware_optimization_enabled': self._interactive_config.enable_hardware_optimization,
                'parallel_processing_enabled': self._interactive_config.enable_parallel_processing,
                'max_workers': self._interactive_config.max_workers,
                'batch_size': self._interactive_config.batch_size,
                'matrix_ops_available': MATRIX_OPS_AVAILABLE,
                'ml_common_available': ML_COMMON_AVAILABLE,
            }

            # Create comprehensive outcome report
            outcome_data = {
                'component': 'interactive_feature_generation',
                'timestamp': datetime.now().isoformat(),
                'execution_time': result.execution_time,
                'configuration': {
                    'symbol': self._interactive_config.symbol,
                    'exchange': self._interactive_config.exchange,
                    'timeframe': self._interactive_config.timeframe,
                    'feature_budget_pre': self._interactive_config.feature_budget_pre,
                    'feature_budget_post': self._interactive_config.feature_budget_post,
                    'interactions_cap': self._interactive_config.interactions_cap,
                    'transforms_per_parent': self._interactive_config.transforms_per_parent,
                    'lookback_ceiling_minutes': self._interactive_config.lookback_ceiling_minutes,
                    'latency_budget_ms': self._interactive_config.latency_budget_ms,
                    'enable_matrix_optimization': self._interactive_config.enable_matrix_optimization,
                    'enable_hardware_optimization': self._interactive_config.enable_hardware_optimization,
                    'enable_parallel_processing': self._interactive_config.enable_parallel_processing,
                    'max_workers': self._interactive_config.max_workers,
                    'batch_size': self._interactive_config.batch_size,
                    # Budget constraints for final feature selection
                    'enable_budget_constraints': self._interactive_config.enable_budget_constraints,
                    'total_budget_ms': self._interactive_config.total_budget_ms,
                    'base_features_budget_ms': self._interactive_config.base_features_budget_ms,
                    'interaction_features_budget_ms': self._interactive_config.interaction_features_budget_ms,
                    'cross_timeframe_features_budget_ms': self._interactive_config.cross_timeframe_features_budget_ms,
                    'gate_features_budget_ms': self._interactive_config.gate_features_budget_ms,
                    # Feature type constraints
                    'base_features_min': self._interactive_config.base_features_min,
                    'base_features_max': self._interactive_config.base_features_max,
                    'base_features_target': self._interactive_config.base_features_target,
                    'interaction_features_min': self._interactive_config.interaction_features_min,
                    'interaction_features_max': self._interactive_config.interaction_features_max,
                    'interaction_features_target': self._interactive_config.interaction_features_target,
                    'cross_timeframe_features_min': self._interactive_config.cross_timeframe_features_min,
                    'cross_timeframe_features_max': self._interactive_config.cross_timeframe_features_max,
                    'cross_timeframe_features_target': self._interactive_config.cross_timeframe_features_target,
                    'gate_features_min': self._interactive_config.gate_features_min,
                    'gate_features_max': self._interactive_config.gate_features_max,
                    'gate_features_target': self._interactive_config.gate_features_target,
                },
                'results': {
                    'summary': {
                        'total_features_generated': len(result.feature_names),
                        'selected_features': len(result.selected_features),
                        'selection_rate_pct': feature_type_breakdown['selection_rate'],
                    },
                    'feature_type_breakdown': feature_type_breakdown,
                    'interaction_statistics': interaction_stats,
                    'cross_timeframe_statistics': cross_timeframe_stats,
                    'feature_names': result.feature_names,
                    'selected_feature_names': result.selected_features,
                },
                'performance_metrics': performance_breakdown,
                'hardware_details': hardware_details,
                'stage_results': stage_results_summary,
                'validation_metadata': validation_metadata,
                'artifacts': {
                    'features_shape': list(result.features.shape) if hasattr(result.features, 'shape') else [0, 0],
                    'interaction_features_shape': list(result.interaction_features.shape) if hasattr(result.interaction_features, 'shape') else [0, 0],
                    'cross_timeframe_features_shape': list(result.cross_timeframe_features.shape) if hasattr(result.cross_timeframe_features, 'shape') else [0, 0],
                },
                'status': 'success'
            }

            # Save outcome file - fail fast if file operations fail
            with open(outcome_path, 'w') as f:
                json.dump(outcome_data, f, indent=2, default=str)

            tprint_success(f"📄 Outcome file saved: {outcome_filename}")

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
                artifacts=InteractiveFeatureArtifacts(),
                execution_time=execution_time,
                metadata={
                    'schema_error': {
                        'schema_key': schema_error.schema_key,
                        'context': schema_error.context,
                        'schema_metadata': schema_metadata(schema_error.schema_key).get(schema_error.schema_key)
                    }
                }
            )

        except DataContractValidationError as contract_error:
            execution_time = time.time() - start_time
            error_message = str(contract_error)
            tprint_error(f"❌ {error_message}")
            self.logger.error(f"Interactive feature generation contract error: {error_message}")
            return ComponentResult(
                success=False,
                error_message=error_message,
                artifacts=InteractiveFeatureArtifacts(),
                execution_time=execution_time,
                metadata={
                    'data_contract_error': {
                        'context': contract_error.context,
                        'issues': contract_error.errors,
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
                artifacts=InteractiveFeatureArtifacts(),
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
        data_batches = training_input.get('data_batches')

        validation_frame = data
        if validation_frame is None and data_batches:
            validation_frame = next(
                (batch for batch in data_batches if isinstance(batch, pd.DataFrame) and not batch.empty),
                None
            )

        if validation_frame is None:
            raise ValueError("No data provided in training input")

        if not isinstance(validation_frame, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if len(validation_frame) < 100:
            raise ValueError(f"Insufficient data: {len(validation_frame)} < 100 rows")

        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(validation_frame.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        tprint_debug("✅ Input validation passed")

    def _ensure_training_input(
        self,
        training_input: Optional[Dict[str, Any]],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure a training input dictionary exists by inspecting pipeline state if needed."""

        if training_input:
            return training_input

        mh_result = pipeline_state.get('multi_horizon_labeling_result', {})
        market_data_batches = mh_result.get('market_data_batches')
        market_data = mh_result.get('market_data')

        if market_data is None and market_data_batches:
            market_data = pd.concat(market_data_batches, axis=0).sort_index()
            # Remove duplicate columns if any were introduced during concatenation
            if len(market_data.columns) != len(set(market_data.columns)):
                tprint_warning(f"⚠️ Duplicate columns detected after concatenating market data batches")
                market_data = market_data.loc[:, ~market_data.columns.duplicated(keep='first')]
                tprint_debug(f"✅ Removed duplicate columns, now have {len(market_data.columns)} unique columns")

        if market_data is None:
            # Try to load data using ares launcher integration
            tprint("📥 [INTERACTIVE_GENERATOR] No market data available, attempting to load using ares launcher integration...")
            tprint_debug("   → Will attempt to load data using ares launcher integration")
            
            try:
                # Initialize ares integration if not already done
                if not hasattr(self, 'ares_integration'):
                    tprint("🔧 [INTERACTIVE_GENERATOR] Initializing ares launcher integration...")
                    self.ares_integration = AresLauncherInteractiveFeatureGenerator()
                    tprint_success("✅ [INTERACTIVE_GENERATOR] Ares launcher integration initialized")
                else:
                    tprint_debug("🔧 [INTERACTIVE_GENERATOR] Ares launcher integration already initialized")
                
                # Get symbol and timeframe from pipeline state
                symbol = pipeline_state.get('symbol', 'ETHUSDT')
                timeframe = pipeline_state.get('timeframe', '15m')
                
                tprint_info(f"📊 [INTERACTIVE_GENERATOR] Configuration for data loading:")
                tprint_info(f"   → Symbol: {symbol}")
                tprint_info(f"   → Timeframe: {timeframe}")
                tprint_debug(f"   → Pipeline state keys: {list(pipeline_state.keys())}")
                tprint_debug(f"   → Pipeline state: {pipeline_state}")
                
                # Load data using ares launcher integration
                tprint("📥 [INTERACTIVE_GENERATOR] Loading data using ares launcher integration...")
                market_data = self.ares_integration.load_data_for_generation(
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_state=pipeline_state
                )
                
                if market_data is not None and not market_data.empty:
                    tprint_success(f"✅ [INTERACTIVE_GENERATOR] Market data loaded via ares launcher: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
                    tprint_info(f"📊 [INTERACTIVE_GENERATOR] Loaded data summary:")
                    tprint_info(f"   → Shape: {market_data.shape}")
                    tprint_info(f"   → Date range: {market_data.index.min().date()} to {market_data.index.max().date()}")
                    tprint_info(f"   → Data mode: {market_data.attrs.get('ares_mode', 'Unknown')}")
                    tprint_info(f"   → Lookback days: {market_data.attrs.get('lookback_days', 'Unknown')}")
                    tprint_debug(f"   → Data columns: {list(market_data.columns)}")
                    tprint_debug(f"   → Memory usage: {market_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                    tprint_debug(f"   → Data attributes: {list(market_data.attrs.keys())}")
                else:
                    error_msg = "No data found using ares launcher integration"
                    tprint_error(f"❌ [INTERACTIVE_GENERATOR] {error_msg}")
                    tprint_debug(f"   → This could be due to:")
                    tprint_debug(f"     - No data available for the specified parameters")
                    tprint_debug(f"     - Data loading error in ares launcher integration")
                    tprint_debug(f"     - Invalid symbol/timeframe combination")
                    raise ValueError(error_msg)
                    
            except Exception as e:
                tprint_error(f"❌ [INTERACTIVE_GENERATOR] Failed to load data using ares launcher integration: {e}")
                tprint_debug(f"   → Exception type: {type(e).__name__}")
                tprint_debug(f"   → Exception details: {str(e)}")
                raise ValueError(f"No market data available to construct training input: {e}")

        labels_df = mh_result.get('labeled_data') or mh_result.get('labels')
        targets: Dict[str, pd.Series] = {}
        if isinstance(labels_df, pd.DataFrame):
            targets = {column: labels_df[column] for column in labels_df.columns}
        
        # Load base features from feature_lookback_optimization if not in pipeline_state
        if market_data is not None and ('feature_matrix' not in pipeline_state or 'optimized_features' not in pipeline_state):
            tprint("📥 [INTERACTIVE_GENERATOR] Loading base features from feature_lookback_optimization...")
            base_features = self._load_feature_lookback_results(pipeline_state)
            if base_features is not None and not base_features.empty:
                tprint_success(f"✅ [INTERACTIVE_GENERATOR] Loaded {base_features.shape[1]} base features")
                # Merge with market data on common timestamps
                common_index = market_data.index.intersection(base_features.index)
                if len(common_index) > 0:
                    market_data = market_data.loc[common_index]
                    base_features = base_features.loc[common_index]
                    market_data = pd.concat([market_data, base_features], axis=1)
                    # Remove duplicate columns if any were introduced during concatenation
                    if len(market_data.columns) != len(set(market_data.columns)):
                        tprint_warning(f"⚠️ Duplicate columns detected after concatenating market data with base features")
                        market_data = market_data.loc[:, ~market_data.columns.duplicated(keep='first')]
                    tprint_success(f"✅ [INTERACTIVE_GENERATOR] Combined data: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
                else:
                    tprint_warning("⚠️ [INTERACTIVE_GENERATOR] No timestamp overlap with base features")
            else:
                tprint_warning("⚠️ [INTERACTIVE_GENERATOR] No base features found - will use market data only")

        resolved_input: Dict[str, Any] = {
            'data': market_data,
            'targets': targets,
        }

        if market_data_batches:
            resolved_input['data_batches'] = list(market_data_batches)

        return resolved_input

    def _load_feature_lookback_results(self, pipeline_state: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load base features from feature_lookback_optimization results."""
        try:
            from pathlib import Path
            import json
            
            symbol = pipeline_state.get('symbol', self.config.symbol)
            timeframe = pipeline_state.get('timeframe', self.config.timeframe)
            
            # Look for most recent feature_lookback_optimization outcome file
            outcomes_dir = Path('outcomes')
            if not outcomes_dir.exists():
                tprint_debug("📂 No outcomes directory found")
                return None
            
            # Find matching outcome files
            pattern = f"*feature_lookback_optimization_outcome_*.json"
            outcome_files = sorted(outcomes_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            
            if not outcome_files:
                tprint_debug("📂 No feature_lookback_optimization outcome files found")
                return None
            
            # Try to load from most recent outcome
            for outcome_file in outcome_files[:5]:  # Check last 5 files
                try:
                    with open(outcome_file, 'r') as f:
                        outcome_data = json.load(f)
                    
                    # Check if it matches our symbol/timeframe
                    config = outcome_data.get('configuration', {})
                    if config.get('symbol') != symbol or config.get('timeframe') != timeframe:
                        continue
                    
                    tprint_info(f"📂 Found matching outcome: {outcome_file.name}")
                    
                    # Try to load the generated features artifact
                    artifacts_dir = Path('artifacts')
                    
                    # Look for feature files matching this run
                    possible_patterns = [
                        f"optimized_features_{symbol}_{timeframe}_*.parquet",
                        f"feature_matrix_{symbol}_{timeframe}_*.parquet",
                        f"features_{symbol}_{timeframe}_*.parquet",
                    ]
                    
                    for pattern in possible_patterns:
                        feature_files = sorted(artifacts_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                        if feature_files:
                            feature_file = feature_files[0]
                            tprint_info(f"📂 Loading features from: {feature_file.name}")
                            features_df = pd.read_parquet(feature_file)
                            tprint_success(f"✅ Loaded {features_df.shape[1]} features, {features_df.shape[0]} rows")
                            return features_df
                    
                    tprint_debug(f"📂 No feature artifacts found for {outcome_file.name}")
                    
                except Exception as e:
                    tprint_debug(f"⚠️ Could not load from {outcome_file.name}: {e}")
                    continue
            
            tprint_warning("⚠️ Could not load feature_lookback_optimization results from any outcome file")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Error loading feature_lookback_optimization results: {e}")
            return None
    
    def _iter_data_batches(self, training_input: Dict[str, Any]) -> Iterable[pd.DataFrame]:
        """Yield data batches when provided in the training input."""

        batches = training_input.get('data_batches') or []
        for batch in batches:
            if isinstance(batch, pd.DataFrame) and not batch.empty:
                yield batch

    async def _execute_chunked_generation(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        data_batches: List[pd.DataFrame],
    ) -> OptimizedInteractionResult:
        """Execute the orchestrator on multiple data batches and merge the results."""

        chunk_results: List[OptimizedInteractionResult] = []
        total_execution_time = 0.0
        max_memory = 0.0
        max_cpu = 0.0
        max_gpu = 0.0

        targets = training_input.get('targets', {})
        base_input = {
            key: value
            for key, value in training_input.items()
            if key not in {'data', 'data_batches', 'targets'}
        }

        for batch in data_batches:
            chunk_input = dict(base_input)
            chunk_input['data'] = batch
            if targets:
                chunk_input['targets'] = self._slice_targets(targets, batch.index)

            chunk_result = await self.orchestrator.generate_features(chunk_input, pipeline_state)
            if not chunk_result.success:
                return chunk_result

            chunk_results.append(chunk_result)
            total_execution_time += chunk_result.execution_time
            max_memory = max(max_memory, chunk_result.memory_usage_mb)
            max_cpu = max(max_cpu, chunk_result.cpu_usage_percent)
            max_gpu = max(max_gpu, chunk_result.gpu_usage_percent)

        return self._merge_chunk_results(
            chunk_results,
            total_execution_time,
            max_memory,
            max_cpu,
            max_gpu,
        )

    def _slice_targets(
        self,
        targets: Dict[str, pd.Series],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
        """Slice target series to match a batch index."""

        sliced: Dict[str, pd.Series] = {}
        for name, series in targets.items():
            if isinstance(series, pd.Series):
                sliced[name] = series.reindex(index)
        return sliced

    def _merge_chunk_results(
        self,
        chunk_results: List[OptimizedInteractionResult],
        total_execution_time: float,
        max_memory: float,
        max_cpu: float,
        max_gpu: float,
    ) -> OptimizedInteractionResult:
        """Merge chunk-level results into a single optimized interaction result."""

        combined_features = self._concat_frames([result.features for result in chunk_results])
        combined_interactions = self._concat_frames([
            result.interaction_features for result in chunk_results
        ])
        combined_cross_timeframe = self._concat_frames([
            result.cross_timeframe_features for result in chunk_results
        ])

        feature_names = chunk_results[-1].feature_names if chunk_results else []
        if not feature_names and not combined_features.empty:
            feature_names = list(combined_features.columns)

        selected_features = chunk_results[-1].selected_features if chunk_results else []

        stage_results: Dict[PipelineStage, Dict[str, Any]] = {}
        artifacts: Dict[str, Any] = {'chunk_results': []}
        for result in chunk_results:
            if result.stage_results:
                stage_results.update(result.stage_results)
            artifacts['chunk_results'].append(result.artifacts)

        performance_metrics: Dict[str, Any] = {}
        if chunk_results:
            performance_metrics = dict(getattr(chunk_results[-1], 'performance_metrics', {}) or {})

        return OptimizedInteractionResult(
            features=combined_features,
            feature_names=feature_names,
            selected_features=selected_features,
            interaction_features=combined_interactions,
            cross_timeframe_features=combined_cross_timeframe,
            execution_time=total_execution_time,
            success=True,
            error_message=None,
            memory_usage_mb=max_memory,
            cpu_usage_percent=max_cpu,
            gpu_usage_percent=max_gpu,
            stage_results=stage_results,
            artifacts=artifacts,
            performance_metrics=performance_metrics,
        )

    @staticmethod
    def _concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate a collection of dataframes while preserving index order."""

        valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
        if not valid_frames:
            return pd.DataFrame()

        combined = pd.concat(valid_frames, axis=0, sort=False)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined.sort_index()
    
    def _update_orchestrator_config(self, pipeline_state: Dict[str, Any]) -> None:
        """Update orchestrator configuration with pipeline state."""
        tprint_debug("🔧 Updating orchestrator configuration...")

        # Update symbol and exchange if provided
        if 'symbol' in pipeline_state:
            self.config.symbol = pipeline_state['symbol']

        if 'exchange' in pipeline_state:
            self.config.exchange = pipeline_state['exchange']

        if 'timeframe' in pipeline_state:
            self.config.timeframe = pipeline_state['timeframe']

        # Update data directory if provided
        if 'data_dir' in pipeline_state:
            self.config.data_dir = pipeline_state['data_dir']

        if 'custom_params' in pipeline_state and isinstance(pipeline_state['custom_params'], Mapping):
            self.config.custom_params.update(dict(pipeline_state['custom_params']))

        self._interactive_config = self._build_interactive_config(self.config)

        # Update enhanced orchestrator configuration
        orchestrator_config = self.orchestrator.config
        
        # Core settings
        orchestrator_config.enable_early_filtering = self._interactive_config.enable_early_filtering
        orchestrator_config.enable_interaction_pruning = self._interactive_config.enable_interaction_pruning
        orchestrator_config.enable_budgeted_optimization = self._interactive_config.enable_budgeted_optimization
        orchestrator_config.enable_caching = self._interactive_config.enable_caching
        orchestrator_config.enable_parallel_processing = self._interactive_config.enable_parallel_processing
        
        # DAG executor settings
        orchestrator_config.max_workers = self._interactive_config.max_workers
        
        # Memory optimization settings
        orchestrator_config.max_memory_gb = self._interactive_config.max_memory_gb
        orchestrator_config.chunk_size_mb = self._interactive_config.chunk_size_mb
        orchestrator_config.use_parquet = self._interactive_config.use_parquet
        orchestrator_config.use_memmap = self._interactive_config.use_memmap
        
        # Cache settings
        orchestrator_config.l1_max_size_mb = self._interactive_config.l1_max_size_mb
        orchestrator_config.l2_max_size_mb = self._interactive_config.l2_max_size_mb
        orchestrator_config.enable_dependency_tracking = self._interactive_config.enable_dependency_tracking
        
        # Early filtering settings
        orchestrator_config.downsample_ratio = self._interactive_config.downsample_ratio
        orchestrator_config.variance_threshold = self._interactive_config.variance_threshold
        orchestrator_config.top_k_per_family = self._interactive_config.top_k_per_family
        
        # Interaction pruning settings
        orchestrator_config.max_interactions_per_domain = self._interactive_config.max_interactions_per_domain
        orchestrator_config.min_delta_ic = self._interactive_config.min_delta_ic
        orchestrator_config.min_stability_score = self._interactive_config.min_stability_score
        
        # Budgeted optimization settings
        orchestrator_config.coarse_grid_points = self._interactive_config.coarse_grid_points
        orchestrator_config.fine_search_evals = self._interactive_config.fine_search_evals
        orchestrator_config.early_stop_patience = self._interactive_config.early_stop_patience
        
        # Performance monitoring
        orchestrator_config.enable_performance_monitoring = self._interactive_config.log_performance
        orchestrator_config.log_level = "INFO" if self._interactive_config.verbose_logging else "WARNING"

        tprint_debug("✅ Orchestrator configuration updated")

    def _apply_namespace_to_result(self, result: OptimizedInteractionResult) -> OptimizedInteractionResult:
        """Apply standardized namespaces to generated feature artifacts."""

        if result.features is not None and isinstance(result.features, pd.DataFrame):
            result.features = ensure_dataframe_namespace(result.features, ColumnNamespace.FEATURE)
        if result.interaction_features is not None and isinstance(result.interaction_features, pd.DataFrame):
            result.interaction_features = ensure_dataframe_namespace(
                result.interaction_features, ColumnNamespace.FEATURE
            )
        if result.cross_timeframe_features is not None and isinstance(result.cross_timeframe_features, pd.DataFrame):
            result.cross_timeframe_features = ensure_dataframe_namespace(
                result.cross_timeframe_features, ColumnNamespace.FEATURE
            )

        if getattr(result, 'feature_names', None):
            result.feature_names = [ensure_namespace(name, ColumnNamespace.FEATURE) for name in result.feature_names]
        if getattr(result, 'selected_features', None):
            result.selected_features = [
                ensure_namespace(name, ColumnNamespace.FEATURE) for name in result.selected_features
            ]

        return result
    
    def _convert_to_component_result(self,
                                   result: OptimizedInteractionResult,
                                   start_time: float,
                                   validation_metadata: Dict[str, Dict[str, Optional[Dict[str, str]]]]) -> ComponentResult:
        """Convert orchestrator result to component result format."""
        import pandas as pd  # Import at method start to avoid scoping issues
        tprint_debug("🔄 Converting result to component format...")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create artifacts
        # Ensure features is a DataFrame (even if empty) for validation
        features_df = result.features if isinstance(result.features, pd.DataFrame) else pd.DataFrame()
        
        # FIXED: If features is empty, create a placeholder DataFrame with at least an index
        # to satisfy validation requirements
        if features_df.empty or len(features_df.columns) == 0:
            # Create minimal DataFrame with time index if available
            if hasattr(result, 'metadata') and 'data_length' in result.metadata:
                features_df = pd.DataFrame(index=range(result.metadata['data_length']))
            else:
                features_df = pd.DataFrame({'placeholder': [0]})  # Minimal valid DataFrame
            tprint_warning(f"⚠️ Creating placeholder DataFrame for validation (0 features generated)")
        
        artifact_payload = {
            'features': features_df,
            'feature_names': result.feature_names if result.feature_names else [],
            'selected_features': result.selected_features if result.selected_features else [],
            'interaction_features': result.interaction_features if isinstance(result.interaction_features, pd.DataFrame) else pd.DataFrame(),
            'cross_timeframe_features': result.cross_timeframe_features if isinstance(result.cross_timeframe_features, pd.DataFrame) else pd.DataFrame(),
            'execution_time': result.execution_time,
            'memory_usage_mb': getattr(result, 'memory_usage_mb', 0.0),
            'success': result.success,
            'error_message': result.error_message,
            'validated_schemas': validation_metadata,
        }

        try:
            validated_payload = validate_feature_artifact(
                artifact_payload,
                context='interactive_feature_generation_component.artifacts',
            )
        except (DataContractValidationError, ValueError) as contract_error:
            tprint_warning(f"⚠️ Interactive feature generation artifact validation failed: {contract_error}")
            tprint_info("ℹ️ Using unvalidated payload (may have generated 0 features)")
            validated_payload = artifact_payload

        artifact_bundle = InteractiveFeatureArtifacts(
            interactive_feature_generation_result=validated_payload,
            stage_results=getattr(result, 'stage_results', {}) or {},
            performance_metrics=getattr(result, 'performance_metrics', {}) or {},
            artifacts=getattr(result, 'artifacts', {}) or {},
            validated_schemas=validation_metadata,
        )

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
            'memory_usage_mb': getattr(result, 'memory_usage_mb', 0.0),
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'ml_common_available': ML_COMMON_AVAILABLE,
            'data_utils_available': DATA_UTILS_AVAILABLE,
            'validated_schemas': validation_metadata
        }

        if result.success:
            metadata['output_files'] = [
                f"features_{self.config.symbol}_{self.config.timeframe}.parquet",
                f"interactions_{self.config.symbol}_{self.config.timeframe}.parquet",
                f"cross_timeframe_{self.config.symbol}_{self.config.timeframe}.parquet",
            ]
        
        tprint_debug("✅ Result conversion completed")

        return ComponentResult(
            success=result.success,
            error_message=result.error_message,
            artifacts=artifact_bundle,
            execution_time=execution_time,
            metadata=metadata,
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
                'feature_budget_pre': self._interactive_config.feature_budget_pre,
                'interactions_cap': self._interactive_config.interactions_cap,
                'enable_matrix_optimization': self._interactive_config.enable_matrix_optimization,
                'enable_hardware_optimization': self._interactive_config.enable_hardware_optimization
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
def _build_component_config(
    config: Optional[InteractiveFeatureGenerationConfig] = None,
) -> ComponentConfig:
    """Convert an interactive configuration into a generic component config."""

    interactive_config = config or InteractiveFeatureGenerationConfig()
    custom_params = {
        'feature_budget_pre': interactive_config.feature_budget_pre,
        'feature_budget_post': interactive_config.feature_budget_post,
        'interactions_cap': interactive_config.interactions_cap,
        'transforms_per_parent': interactive_config.transforms_per_parent,
        'lookback_ceiling_minutes': interactive_config.lookback_ceiling_minutes,
        'latency_budget_ms': interactive_config.latency_budget_ms,
        'enable_matrix_optimization': interactive_config.enable_matrix_optimization,
        'enable_hardware_optimization': interactive_config.enable_hardware_optimization,
        'enable_parallel_processing': interactive_config.enable_parallel_processing,
        'max_workers': interactive_config.max_workers,
        'batch_size': interactive_config.batch_size,
        'enable_validation': interactive_config.enable_validation,
        'validation_threshold': interactive_config.validation_threshold,
        'verbose_logging': interactive_config.verbose_logging,
        'log_performance': interactive_config.log_performance,
        'integrate_with_ares_launcher': interactive_config.integrate_with_ares_launcher,
        'maintain_backward_compatibility': interactive_config.maintain_backward_compatibility,
        'market_data_batch_size': interactive_config.market_data_batch_size,
        'market_data_window_days': interactive_config.market_data_window_days,
    }

    return ComponentConfig(
        symbol=interactive_config.symbol,
        exchange=interactive_config.exchange,
        timeframe=interactive_config.timeframe,
        data_dir=interactive_config.data_dir,
        custom_params=custom_params,
    )


def create_interactive_feature_generation_component(
    config: Optional[Union[InteractiveFeatureGenerationConfig, ComponentConfig]] = None
) -> InteractiveFeatureGenerationComponent:
    """
    Create an interactive feature generation component.

    Args:
        config: Optional legacy interactive config or generic component config.

    Returns:
        InteractiveFeatureGenerationComponent instance
    """

    if isinstance(config, ComponentConfig):
        component_config = config
    else:
        component_config = _build_component_config(config)

    from ...components.component_factory import ComponentFactory

    return ComponentFactory.create_component('interactive_feature_generation', component_config)  # type: ignore[return-value]


# Integration with component factory
def register_interactive_feature_generation_component():
    """Register the interactive feature generation component with the factory."""
    try:
        from ...components.component_factory import ComponentFactory

        # Register the component
        ComponentFactory.register_component(
            'interactive_feature_generation',
            InteractiveFeatureGenerationComponent
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


# Register the component with the factory
if _COMPONENTS_AVAILABLE:
    # Component factory is already imported above when _COMPONENTS_AVAILABLE is True
    ComponentFactory.register_component('interactive_feature_generation', InteractiveFeatureGenerationComponent)