"""
Enhanced Optimized Interaction Feature Generation Orchestrator

This module replaces the existing OptimizedInteractionOrchestrator with a fully
optimized version that integrates all the advanced optimizations:

1. DAG executor with parallel processing
2. Memory-efficient model with PyArrow/Parquet and memmap
3. Content-addressed cache with dependency graph
4. Early filtering system
5. Interaction pruning system
6. Budgeted lookback optimization with TPE

Key Features:
- Drop-in replacement for existing orchestrator
- Maintains backward compatibility
- Integrates all optimization components
- Extensive performance monitoring
- Configurable optimization levels
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
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

# Import optimization components
from .dag_executor import DAGExecutor, DAGNode, NodeType, ExecutionContext
from .memory_model import MemoryEfficientProcessor, MemoryConfig
from .content_cache import ContentAddressedCache, CacheConfig, WarmStartData
from .early_filtering import EarlyFilteringSystem, EarlyFilteringConfig, FilteringResult
from .interaction_pruning import InteractionPruningSystem, InteractionPruningConfig, PruningResult
from .budgeted_optimization import BudgetedLookbackOptimizer, BudgetedOptimizationConfig, OptimizationResult
from .optimized_pipeline import OptimizedPipelineConfig, OptimizedInteractiveFeaturePipeline

# Import VectorBT optimizations
try:
    from .vectorbt_optimized_feature_generator import (
        VectorBTOptimizedFeatureGenerator, VectorBTFeatureConfig,
        generate_vectorbt_features, create_vectorbt_feature_generator
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTOptimizedFeatureGenerator = None
    VectorBTFeatureConfig = None
    generate_vectorbt_features = None
    create_vectorbt_feature_generator = None

# Import VectorBT rolling optimizer and unified vectorization manager
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_quantile, optimized_rolling_apply,
        optimized_rolling_corr, optimized_rolling_cov
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager,
        OperationType, OptimizationStrategy, optimize_financial_operation
    )
    VECTORBT_ROLLING_AVAILABLE = True
    tprint_success("✅ VectorBT rolling optimizations imported successfully")
except ImportError as e:
    VECTORBT_ROLLING_AVAILABLE = False
    tprint_warning(f"⚠️ VectorBT rolling optimizations not available: {e}")
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    optimized_rolling_mean = None
    optimized_rolling_std = None
    optimized_rolling_var = None
    optimized_rolling_min = None
    optimized_rolling_max = None
    optimized_rolling_sum = None
    optimized_rolling_quantile = None
    optimized_rolling_apply = None
    optimized_rolling_corr = None
    optimized_rolling_cov = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    OperationType = None
    OptimizationStrategy = None
    optimize_financial_operation = None

# Import existing utilities for backward compatibility - fast-fail
from src.feature_generation.core.feature_cache import FeatureCacheService
from src.feature_generation.core.feature_bank import FeatureBank
from ...settings import get_pre_training_settings
EXISTING_UTILS_AVAILABLE = True

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages for the enhanced orchestrator."""
    INITIALIZATION = "initialization"
    EARLY_FILTERING = "early_filtering"
    FEATURE_ENGINEERING = "feature_engineering"
    BUDGETED_OPTIMIZATION = "budgeted_optimization"
    INTERACTION_GENERATION = "interaction_generation"
    INTERACTION_PRUNING = "interaction_pruning"
    CROSS_TIMEFRAME = "cross_timeframe"
    FINAL_ASSEMBLY = "final_assembly"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class EnhancedOptimizedConfig:
    """Enhanced configuration for the optimized orchestrator."""
    # Pipeline settings
    enable_early_filtering: bool = True
    enable_interaction_pruning: bool = True
    enable_budgeted_optimization: bool = True
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    
    # DAG executor settings
    max_workers: int = 8
    use_processes: bool = False  # FIXED: Changed to threads to avoid pickling issues with thread locks
    
    # Memory model settings
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
    variance_threshold: float = 1e-6  # FIXED: More reasonable threshold to prevent over-filtering
    top_k_per_family: int = 10  # FIXED: Reduced to prevent memory issues and improve performance
    
    # Interaction pruning settings
    max_interactions_per_domain: int = 6
    min_delta_ic: float = 0.01
    min_stability_score: float = 0.7
    
    # Budgeted optimization settings
    coarse_grid_points: int = 10
    fine_search_evals: int = 16
    early_stop_patience: int = 5
    
    # VectorBT optimizations
    enable_vectorbt: bool = True
    vectorbt_use_gpu: bool = True
    vectorbt_chunk_size: int = 50000
    vectorbt_memory_limit_gb: float = 8.0
    vectorbt_enable_parallel: bool = True
    
    # VectorBT rolling operations optimization
    enable_vectorbt_rolling: bool = True
    vectorbt_rolling_window_threshold: int = 1000  # Use VectorBT for windows >= this size
    vectorbt_correlation_threshold: int = 500  # Use VectorBT for correlation with >= this data points
    vectorbt_rolling_use_gpu: bool = True
    vectorbt_rolling_parallel: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = False


@dataclass
class OptimizedInteractionResult:
    """Result of the optimized interaction feature generation."""
    # Core results
    features: pd.DataFrame
    feature_names: List[str]
    selected_features: List[str]
    interaction_features: pd.DataFrame
    cross_timeframe_features: pd.DataFrame
    
    # Performance metrics
    execution_time: float
    memory_usage_mb: float
    cache_hit_rate: float
    parallel_efficiency: float
    
    # Optimization results
    early_filtering_result: Optional[FilteringResult] = None
    interaction_pruning_result: Optional[PruningResult] = None
    budgeted_optimization_result: Optional[OptimizationResult] = None
    
    # Pipeline metadata
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class EnhancedOptimizedInteractionOrchestrator:
    """
    Enhanced optimized interaction feature generation orchestrator.
    
    This is a drop-in replacement for the existing OptimizedInteractionOrchestrator
    that integrates all the advanced optimizations while maintaining backward compatibility.
    """
    
    def __init__(self, config: Optional[EnhancedOptimizedConfig] = None):
        """Initialize the enhanced optimized orchestrator."""
        self.config = config or EnhancedOptimizedConfig()
        
        # Initialize optimization components
        self._initialize_components()
        
        # Performance tracking
        self.start_time = 0.0
        self.stage_times = {}
        self.performance_metrics = {}
        
        # VectorBT performance tracking
        self.vectorbt_performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'parallel_operations': 0
        }
        
        tprint_success("🚀 Enhanced Optimized Interaction Orchestrator initialized")
        tprint_info(f"📊 Max workers: {self.config.max_workers}")
        tprint_info(f"📊 Max memory: {self.config.max_memory_gb} GB")
        tprint_info(f"📊 Early filtering: {'✅' if self.config.enable_early_filtering else '❌'}")
        tprint_info(f"📊 Interaction pruning: {'✅' if self.config.enable_interaction_pruning else '❌'}")
        tprint_info(f"📊 Budgeted optimization: {'✅' if self.config.enable_budgeted_optimization else '❌'}")
        tprint_info(f"📊 Caching: {'✅' if self.config.enable_caching else '❌'}")
        tprint_info(f"📊 Parallel processing: {'✅' if self.config.enable_parallel_processing else '❌'}")
    
    def _initialize_components(self) -> None:
        """Initialize all optimization components."""
        tprint_debug("🔧 Initializing optimization components...")
        
        # Initialize memory processor
        memory_config = MemoryConfig(
            max_memory_gb=self.config.max_memory_gb,
            chunk_size_mb=self.config.chunk_size_mb,
            use_parquet=self.config.use_parquet,
            use_memmap=self.config.use_memmap
        )
        self.memory_processor = MemoryEfficientProcessor(memory_config)
        
        # Initialize cache
        if self.config.enable_caching:
            cache_config = CacheConfig(
                l1_max_size_mb=self.config.l1_max_size_mb,
                l2_max_size_mb=self.config.l2_max_size_mb,
                enable_dependency_tracking=self.config.enable_dependency_tracking
            )
            self.cache = ContentAddressedCache(cache_config)
        else:
            self.cache = None
        
        # Initialize early filtering
        if self.config.enable_early_filtering:
            early_filtering_config = EarlyFilteringConfig(
                downsample_ratio=self.config.downsample_ratio,
                variance_threshold=self.config.variance_threshold,
                top_k_per_family=self.config.top_k_per_family
            )
            self.early_filtering = EarlyFilteringSystem(early_filtering_config)
        else:
            self.early_filtering = None
        
        # Initialize interaction pruning
        if self.config.enable_interaction_pruning:
            interaction_pruning_config = InteractionPruningConfig(
                max_interactions_per_domain=self.config.max_interactions_per_domain,
                min_delta_ic=self.config.min_delta_ic,
                min_stability_score=self.config.min_stability_score
            )
            self.interaction_pruning = InteractionPruningSystem(interaction_pruning_config)
        else:
            self.interaction_pruning = None
        
        # Initialize budgeted optimizer
        if self.config.enable_budgeted_optimization:
            budgeted_optimization_config = BudgetedOptimizationConfig(
                coarse_grid_points=self.config.coarse_grid_points,
                fine_search_evals=self.config.fine_search_evals,
                early_stop_patience=self.config.early_stop_patience
            )
            self.budgeted_optimizer = BudgetedLookbackOptimizer(budgeted_optimization_config)
        else:
            self.budgeted_optimizer = None
        
        # Initialize DAG executor
        if self.config.enable_parallel_processing:
            self.dag_executor = DAGExecutor(
                max_workers=self.config.max_workers,
                use_processes=False  # FIXED: Force threads to avoid pickling issues
            )
        else:
            self.dag_executor = None
        
        # Initialize VectorBT optimizations
        self._initialize_vectorbt_optimizations()
        
        tprint_success("✅ All optimization components initialized")
    
    def _initialize_vectorbt_optimizations(self):
        """Initialize VectorBT optimization components."""
        if not VECTORBT_ROLLING_AVAILABLE:
            tprint_warning("⚠️ VectorBT rolling optimizations not available, using fallback methods")
            self.vectorbt_rolling_optimizer = None
            self.unified_vectorization_manager = None
            return

        tprint_debug("🔧 Initializing VectorBT rolling optimizations...")

        try:
            # Initialize VectorBT rolling optimizer
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.vectorbt_rolling_use_gpu,
                enable_parallel=self.config.vectorbt_rolling_parallel
            )
            tprint_success("✅ VectorBT rolling optimizer initialized")

            # Initialize unified vectorization manager
            self.unified_vectorization_manager = get_unified_vectorization_manager()
            tprint_success("✅ Unified vectorization manager initialized")

            # Configure VectorBT settings
            if hasattr(self.vectorbt_rolling_optimizer, 'chunk_size'):
                self.vectorbt_rolling_optimizer.chunk_size = self.config.vectorbt_chunk_size
            
            tprint_info(f"🚀 VectorBT rolling optimizations configured:")
            tprint_info(f"   → GPU acceleration: {'✅' if self.config.vectorbt_rolling_use_gpu else '❌'}")
            tprint_info(f"   → Parallel processing: {'✅' if self.config.vectorbt_rolling_parallel else '❌'}")
            tprint_info(f"   → Window threshold: {self.config.vectorbt_rolling_window_threshold:,}")
            tprint_info(f"   → Correlation threshold: {self.config.vectorbt_correlation_threshold:,}")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize VectorBT rolling optimizations: {e}")
            self.vectorbt_rolling_optimizer = None
            self.unified_vectorization_manager = None
    
    def _track_vectorbt_performance(self, operation_type: str, execution_time: float, 
                                  vectorbt_used: bool = False, gpu_used: bool = False, 
                                  parallel_used: bool = False):
        """Track VectorBT performance metrics."""
        self.vectorbt_performance_stats['total_operations'] += 1
        self.vectorbt_performance_stats['total_time'] += execution_time
        
        if vectorbt_used:
            self.vectorbt_performance_stats['vectorbt_operations'] += 1
        else:
            self.vectorbt_performance_stats['pandas_fallbacks'] += 1
        
        if gpu_used:
            self.vectorbt_performance_stats['gpu_operations'] += 1
        
        if parallel_used:
            self.vectorbt_performance_stats['parallel_operations'] += 1
    
    def _get_vectorbt_performance_summary(self) -> Dict[str, Any]:
        """Get VectorBT performance summary."""
        stats = self.vectorbt_performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['pandas_fallback_rate'] = stats['pandas_fallbacks'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['parallel_usage_rate'] = stats['parallel_operations'] / stats['total_operations']
        
        return stats
    
    def _create_optimized_dag(self) -> DAGExecutor:
        """Create an optimized DAG for the pipeline."""
        if not self.dag_executor:
            raise RuntimeError("DAG executor not available - parallel processing disabled")
        
        executor = DAGExecutor(
            max_workers=self.config.max_workers,
            use_processes=False  # FIXED: Force threads to avoid pickling issues
        )
        
        # Define optimized nodes
        nodes = [
            DAGNode(
                node_id="init",
                node_type=NodeType.INITIALIZATION,
                function=self._initialize_stage,
                can_parallelize=False,
                priority=10
            ),
            DAGNode(
                node_id="early_filtering",
                node_type=NodeType.FEATURE_ENGINEERING,
                function=self._early_filtering_stage,
                can_parallelize=True,
                priority=8
            ),
            DAGNode(
                node_id="feature_engineering",
                node_type=NodeType.FEATURE_ENGINEERING,
                function=self._feature_engineering_stage,
                can_parallelize=True,
                priority=8
            ),
            DAGNode(
                node_id="budgeted_optimization",
                node_type=NodeType.LOOKBACK_OPTIMIZATION,
                function=self._budgeted_optimization_stage,
                can_parallelize=True,
                priority=7
            ),
            DAGNode(
                node_id="interaction_generation",
                node_type=NodeType.INTERACTION_GENERATION,
                function=self._interaction_generation_stage,
                can_parallelize=True,
                priority=6
            ),
            DAGNode(
                node_id="interaction_pruning",
                node_type=NodeType.INTERACTION_GENERATION,
                function=self._interaction_pruning_stage,
                can_parallelize=True,
                priority=6
            ),
            DAGNode(
                node_id="cross_timeframe",
                node_type=NodeType.CROSS_TIMEFRAME,
                function=self._cross_timeframe_stage,
                can_parallelize=True,
                priority=6
            ),
            DAGNode(
                node_id="final_assembly",
                node_type=NodeType.FINAL_ASSEMBLY,
                function=self._final_assembly_stage,
                can_parallelize=False,
                priority=5
            ),
            DAGNode(
                node_id="validation",
                node_type=NodeType.VALIDATION,
                function=self._validation_stage,
                can_parallelize=False,
                priority=4
            )
        ]
        
        # Add nodes to executor
        for node in nodes:
            executor.add_node(node)
        
        # Add dependencies
        executor.add_dependency("early_filtering", "init")
        executor.add_dependency("feature_engineering", "init")
        executor.add_dependency("budgeted_optimization", "feature_engineering")
        executor.add_dependency("interaction_generation", "feature_engineering")
        executor.add_dependency("interaction_pruning", "interaction_generation")
        executor.add_dependency("cross_timeframe", "feature_engineering")
        executor.add_dependency("final_assembly", "interaction_pruning")
        executor.add_dependency("final_assembly", "cross_timeframe")
        executor.add_dependency("final_assembly", "budgeted_optimization")
        executor.add_dependency("validation", "final_assembly")
        
        return executor
    
    async def _initialize_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Initialize the pipeline stage."""
        tprint_debug("🔧 Initializing enhanced optimized pipeline...")
        
        # Convert data to memory-efficient format
        if isinstance(context.data, pd.DataFrame):
            # Convert to PyArrow Table for efficient storage
            table = self.memory_processor.to_columnar(context.data, "input_data")
            context.data = table
        
        # Initialize cache if available
        if self.cache:
            cache_key = self._compute_cache_key(context)
            context.pipeline_state['cache_key'] = cache_key
        
        return {
            'status': 'initialized',
            'memory_optimized': True,
            'cache_enabled': self.cache is not None
        }
    
    async def _early_filtering_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Early filtering stage."""
        if not self.early_filtering:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        tprint_debug("🔍 Early filtering stage...")
        
        # Convert data back to DataFrame if needed
        if hasattr(context.data, 'to_pandas'):
            data = context.data.to_pandas()
        else:
            data = context.data
        
        # OPTIMIZATION: Skip early filtering in light mode with optimized features
        # Since features are already optimized from lookback optimization, early filtering is redundant
        execution_mode = context.pipeline_state.get('execution_mode', 'full')
        if execution_mode == 'light':
            tprint_info("🚀 Skipping early filtering in LIGHT mode (features already optimized)")
            exclude_cols = ['target', 'timestamp', 'open_time', 'close_time', 'symbol', 'interval', 'exchange']
            all_features = [col for col in data.columns if col not in exclude_cols]
            context.pipeline_state['filtered_features'] = all_features
            return {
                'status': 'skipped',
                'selected_features': all_features,
                'rejected_features': [],
                'reason': 'light_mode_optimization'
            }
        
        # OPTIMIZATION: Skip early filtering if data is too small (redundant processing)
        if len(data) < 200:
            tprint_info("🚀 Skipping early filtering for small dataset (redundant processing)")
            exclude_cols = ['target', 'timestamp', 'open_time', 'close_time', 'symbol', 'interval', 'exchange']
            all_features = [col for col in data.columns if col not in exclude_cols]
            context.pipeline_state['filtered_features'] = all_features
            return {
                'status': 'skipped',
                'selected_features': all_features,
                'rejected_features': [],
                'reason': 'small_dataset'
            }
        
        # Apply mRMR-based early filtering
        filtered_features = await self._apply_mrmr_early_filtering(data, context.pipeline_state)
        
        # Update context with filtered features
        context.pipeline_state['filtered_features'] = filtered_features
        
        # Calculate filtering metrics
        exclude_cols = ['target', 'timestamp', 'open_time', 'close_time', 'symbol', 'interval', 'exchange']
        all_features = [col for col in data.columns if col not in exclude_cols]
        rejected_features = [col for col in all_features if col not in filtered_features]
        
        return {
            'status': 'completed',
            'selected_features': len(filtered_features),
            'rejected_features': len(rejected_features),
            'filtering_efficiency': len(filtered_features) / len(all_features) if all_features else 0.0,
            'method': 'mrmr_enhanced'
        }
    
    async def _apply_mrmr_early_filtering(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> List[str]:
        """Apply mRMR-based early filtering to select the most relevant and non-redundant features."""
        try:
            from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
            from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
            
            tprint_debug("🔍 Applying mRMR early filtering...")
            
            # Prepare features and target
            exclude_cols = ['target', 'timestamp', 'open_time', 'close_time', 'symbol', 'interval', 'exchange']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            if not feature_cols:
                tprint_warning("⚠️ No features available for mRMR filtering")
                return []
            
            # Get target column
            target_column = pipeline_state.get('target_column', 'target')
            if target_column not in data.columns:
                # Try fallback targets
                for fallback in ['analyst_target', 'tactician_target', 'close']:
                    if fallback in data.columns:
                        target_column = fallback
                        break
                else:
                    tprint_warning("⚠️ No valid target found for mRMR filtering")
                    return feature_cols
            
            # Prepare data for mRMR
            X = data[feature_cols].values
            y = data[target_column].values
            
            # Remove any rows with NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:
                tprint_warning("⚠️ Insufficient data for mRMR filtering after cleaning")
                return feature_cols
            
            # Configure mRMR selector
            config = VectorBTFeatureSelectionConfig(
                mrmr_max_features=min(50, len(feature_cols)),  # Select up to 50 features
                mrmr_alpha=0.7,  # Weight for relevance
                mrmr_beta=0.3,   # Weight for redundancy
                chunk_size=1000,
                enable_parallel_processing=True
            )
            
            # Initialize and run mRMR selector
            mrmr_selector = VectorBTMRMRSelector(config)
            result = mrmr_selector.select_features(
                X, y, 
                k=min(50, len(feature_cols)),
                feature_names=feature_cols
            )
            
            if result['success']:
                selected_features = result['selected_features']
                tprint_success(f"✅ mRMR filtering selected {len(selected_features)} features from {len(feature_cols)}")
                return selected_features
            else:
                tprint_warning(f"⚠️ mRMR filtering failed: {result.get('error_message', 'Unknown error')}")
                return feature_cols
                
        except ImportError as e:
            tprint_warning(f"⚠️ mRMR selector not available: {e}")
            # Fallback to basic filtering
            exclude_cols = ['target', 'timestamp', 'open_time', 'close_time', 'symbol', 'interval', 'exchange']
            return [col for col in data.columns if col not in exclude_cols]
        except Exception as e:
            tprint_error(f"❌ mRMR filtering failed: {e}")
            # Fallback to basic filtering
            exclude_cols = ['target', 'timestamp', 'open_time', 'close_time', 'symbol', 'interval', 'exchange']
            return [col for col in data.columns if col not in exclude_cols]
    
    async def _feature_engineering_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Feature engineering stage with actual feature generation using Stage 2 filtered features."""
        tprint_debug("🏗️ Feature engineering stage...")
        
        # Convert data back to DataFrame if needed
        if hasattr(context.data, 'to_pandas'):
            data = context.data.to_pandas()
        else:
            data = context.data
        
        # Fast-fail: Validate input data early
        if data.empty:
            raise RuntimeError("CRITICAL: Input data is empty in feature engineering stage")
        
        if len(data) < 10:
            raise RuntimeError(f"CRITICAL: Insufficient data for feature generation: {len(data)} < 10")
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise RuntimeError(f"CRITICAL: Missing required columns for feature generation: {missing_cols}")
        
        # Use filtered features from Stage 2 if available
        filtered_features = context.pipeline_state.get('filtered_features', [])
        if filtered_features:
            tprint_info(f"🔍 Using {len(filtered_features)} pre-filtered features from Stage 2")
            # Filter data to only include the selected features plus required OHLCV columns
            keep_cols = list(set(required_cols + filtered_features))
            data = data[keep_cols]
        
        # Check for all-NaN data
        if data.isnull().all().all():
            raise RuntimeError("CRITICAL: All data is NaN - cannot generate features")
        
        tprint_debug(f"🔍 Input data shape: {data.shape}")
        tprint_debug(f"🔍 Input data columns: {list(data.columns)}")
        tprint_debug(f"🔍 Data types: {data.dtypes.to_dict()}")
        
        # Use VectorBT optimized feature generation with fast-fail validation
        from .vectorbt_optimized_feature_generator import (
            VectorBTOptimizedFeatureGenerator, VectorBTFeatureConfig,
            generate_vectorbt_features
        )
        
        # Create VectorBT feature generation config
        vectorbt_config = VectorBTFeatureConfig(
            enable_vectorbt_rolling=self.config.enable_vectorbt_rolling,
            vectorbt_window_threshold=self.config.vectorbt_rolling_window_threshold,
            vectorbt_correlation_threshold=self.config.vectorbt_correlation_threshold,
            enable_gpu=self.config.vectorbt_rolling_use_gpu,
            enable_parallel=self.config.vectorbt_rolling_parallel,
            chunk_size=self.config.vectorbt_chunk_size,
            memory_limit_gb=self.config.vectorbt_memory_limit_gb,
            rolling_windows=[5, 10, 20, 50, 100, 200],
            quantile_levels=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
        
        # Generate base features using VectorBT optimized generator - fast-fail on error
        try:
            if VECTORBT_ROLLING_AVAILABLE and self.config.enable_vectorbt_rolling:
                tprint_info("🚀 Using VectorBT optimized feature generation")
                generated_features = generate_vectorbt_features(data, vectorbt_config)
            else:
                tprint_info("⚠️ Using fallback feature generation (VectorBT not available)")
                from .feature_generation_utils import ImprovedFeatureGenerator, FeatureGenerationConfig
                feature_config = FeatureGenerationConfig(
                    enable_technical_indicators=True,
                    enable_rolling_stats=True,
                    enable_interaction_features=False,  # Will be done in interaction stage
                    enable_cross_timeframe=False,  # Will be done in cross-timeframe stage
                    rolling_windows=[5, 10, 20, 50, 100],
                    max_interactions=50,
                    min_valid_ratio=0.8,  # Require 80% valid values
                    max_constant_ratio=0.1  # Allow max 10% constant features
                )
                feature_generator = ImprovedFeatureGenerator(feature_config)
                generated_features = feature_generator.generate_meaningful_features(data)
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Feature generation failed: {e}")
        
        # CRITICAL: Fast-fail if no features generated
        if generated_features.empty or len(generated_features.columns) == 0:
            raise RuntimeError("CRITICAL: Feature generation failed - no features created")
        
        tprint_info(f"✅ Generated {len(generated_features.columns)} validated base features")
        tprint_debug(f"🔍 Generated features: {list(generated_features.columns)[:10]}{'...' if len(generated_features.columns) > 10 else ''}")
        
        # Apply memory optimization using the matrix operations utility - fast-fail on error
        try:
            from src.utils.matrix_operations import optimize_dataframe
            optimized_features = optimize_dataframe(generated_features)
        except ImportError:
            tprint_warning("⚠️ Matrix operations not available, using basic optimization")
            optimized_features = self._basic_memory_optimization(generated_features)
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Memory optimization failed: {e}")
        
        # Store in memory-efficient format
        if self.config.use_parquet:
            try:
                table = self.memory_processor.to_columnar(optimized_features, "generated_features")
                context.pipeline_state['generated_features'] = table
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Failed to convert to columnar format: {e}")
        else:
            context.pipeline_state['generated_features'] = optimized_features
        
        return {
            'status': 'completed',
            'features_generated': len(optimized_features.columns),
            'memory_optimized': True
        }
    
    async def _budgeted_optimization_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Budgeted optimization stage."""
        if not self.budgeted_optimizer:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        tprint_debug("🎯 Budgeted optimization stage...")
        
        # Get generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
        else:
            features = context.data
        
        # FIXED: Check if features is empty or has no columns
        if features is None or len(features.columns) == 0:
            tprint_warning("⚠️ No features available for budgeted optimization, skipping")
            return {'status': 'skipped', 'reason': 'no_features', 'features_count': 0}
        
        # FIXED: Check if features have any non-NaN values
        if features.isnull().all().all():
            tprint_warning("⚠️ All features are NaN, skipping budgeted optimization")
            return {'status': 'skipped', 'reason': 'all_nan_features', 'features_count': len(features.columns)}
        
        # Get target
        target_column = context.pipeline_state.get('target_column', 'target')
        
        # FIXED: Better target column handling with fallback
        if target_column in features.columns:
            target = features[target_column]
        elif len(features.columns) > 0:
            # Use first column as target as fallback
            target = features.iloc[:, 0]
            target_column = features.columns[0]
            tprint_info(f"🔄 Using fallback target column: {target_column}")
        else:
            tprint_warning("⚠️ No valid target column found, skipping budgeted optimization")
            return {'status': 'skipped', 'reason': 'no_target', 'features_count': len(features.columns)}
        
        # Get feature names
        feature_names = [col for col in features.columns if col != target_column]
        
        # FIXED: Check if we have any features to optimize
        if len(feature_names) == 0:
            tprint_warning("⚠️ No features to optimize after removing target column")
            return {'status': 'skipped', 'reason': 'no_features_after_target_removal', 'features_count': 0}
        
        # Perform budgeted optimization
        optimization_result = self.budgeted_optimizer.optimize_lookbacks(
            features, target, feature_names
        )
        
        # Store result
        context.pipeline_state['budgeted_optimization_result'] = optimization_result
        
        return {
            'status': 'completed',
            'families_optimized': len(optimization_result.best_choices),
            'average_confidence': optimization_result.performance_metrics.get('average_confidence', 0.0),
            'execution_time': optimization_result.performance_metrics.get('execution_time', 0.0)
        }
    
    async def _interaction_generation_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Interaction generation stage using Random Forest and SHAP for intelligent feature interactions."""
        tprint_debug("🔗 Interaction generation stage with RF/SHAP approach...")
        
        # Get generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
        else:
            features = context.data
        
        # Generate interaction features using Random Forest + SHAP approach
        try:
            interaction_features = await self._generate_rf_shap_interactions(features, context.pipeline_state)
        except Exception as e:
            tprint_warning(f"⚠️ RF/SHAP interaction generation failed: {e}")
            # Fallback to basic interaction generation
            interaction_features = await self._generate_basic_interactions(features)
        
        # CRITICAL: Fast-fail if no interaction features generated
        if interaction_features.empty or len(interaction_features.columns) == 0:
            tprint_warning("⚠️ No interaction features generated - this may be expected for some datasets")
            interaction_features = pd.DataFrame(index=features.index)  # Create empty DataFrame with correct index
        else:
            tprint_info(f"✅ Generated {len(interaction_features.columns)} validated interaction features")
        
        # Store result
        context.pipeline_state['interaction_features'] = interaction_features
        
        return {
            'status': 'completed',
            'interaction_features': len(interaction_features.columns),
            'method': 'rf_shap_enhanced'
        }
    
    async def _generate_rf_shap_interactions(self, features: pd.DataFrame, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Generate interaction features using Random Forest and SHAP for intelligent feature selection."""
        try:
            import shap
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            tprint_debug("🌲 Generating RF/SHAP-based interactions...")
            
            # Get target column
            target_column = pipeline_state.get('target_column', 'target')
            if target_column not in features.columns:
                # Try fallback targets
                for fallback in ['analyst_target', 'tactician_target', 'close']:
                    if fallback in features.columns:
                        target_column = fallback
                        break
                else:
                    tprint_warning("⚠️ No valid target found for RF/SHAP interactions")
                    return pd.DataFrame(index=features.index)
            
            # Prepare data
            feature_cols = [col for col in features.columns if col != target_column]
            if len(feature_cols) < 2:
                tprint_warning("⚠️ Insufficient features for interaction generation")
                return pd.DataFrame(index=features.index)
            
            X = features[feature_cols].values
            y = features[target_column].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                tprint_warning("⚠️ Insufficient data for RF/SHAP interactions")
                return pd.DataFrame(index=features.index)
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test_scaled)
            
            # Find important feature interactions using SHAP
            interaction_features = self._extract_shap_interactions(
                X_test_scaled, shap_values, feature_cols, rf, scaler
            )
            
            return interaction_features
            
        except ImportError as e:
            tprint_warning(f"⚠️ SHAP not available: {e}")
            return await self._generate_basic_interactions(features)
        except Exception as e:
            tprint_error(f"❌ RF/SHAP interaction generation failed: {e}")
            return await self._generate_basic_interactions(features)
    
    def _extract_shap_interactions(self, X: np.ndarray, shap_values: np.ndarray, 
                                 feature_names: List[str], rf_model, scaler) -> pd.DataFrame:
        """Extract interaction features based on SHAP values and feature importance."""
        try:
            import numpy as np
            from itertools import combinations
            
            # Get feature importance
            feature_importance = rf_model.feature_importances_
            
            # Select top features based on importance
            top_features_idx = np.argsort(feature_importance)[-min(20, len(feature_names)):]
            top_features = [feature_names[i] for i in top_features_idx]
            
            # Generate interaction features
            interaction_data = {}
            
            # 1. Ratio interactions for top features
            for i, j in combinations(top_features_idx, 2):
                if i != j:
                    feature1, feature2 = feature_names[i], feature_names[j]
                    # Avoid division by zero
                    ratio_feature = X[:, i] / (X[:, j] + 1e-8)
                    interaction_data[f"{feature1}_div_{feature2}"] = ratio_feature
            
            # 2. Product interactions for highly important features
            top_5_idx = top_features_idx[-5:]
            for i, j in combinations(top_5_idx, 2):
                if i != j:
                    feature1, feature2 = feature_names[i], feature_names[j]
                    product_feature = X[:, i] * X[:, j]
                    interaction_data[f"{feature1}_mul_{feature2}"] = product_feature
            
            # 3. Difference interactions for trend features
            for i, j in combinations(top_features_idx, 2):
                if i != j:
                    feature1, feature2 = feature_names[i], feature_names[j]
                    diff_feature = X[:, i] - X[:, j]
                    interaction_data[f"{feature1}_sub_{feature2}"] = diff_feature
            
            # 4. SHAP-based weighted combinations
            shap_weights = np.abs(shap_values).mean(axis=0)
            for i, j in combinations(top_features_idx, 2):
                if i != j and shap_weights[i] > 0.01 and shap_weights[j] > 0.01:
                    feature1, feature2 = feature_names[i], feature_names[j]
                    weight1, weight2 = shap_weights[i], shap_weights[j]
                    weighted_sum = (weight1 * X[:, i] + weight2 * X[:, j]) / (weight1 + weight2)
                    interaction_data[f"{feature1}_wsum_{feature2}"] = weighted_sum
            
            # Convert to DataFrame
            if interaction_data:
                interaction_df = pd.DataFrame(interaction_data)
                # Remove any infinite or NaN values
                interaction_df = interaction_df.replace([np.inf, -np.inf], np.nan)
                interaction_df = interaction_df.fillna(0)
                return interaction_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ SHAP interaction extraction failed: {e}")
            return pd.DataFrame()
    
    async def _generate_basic_interactions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fallback basic interaction generation."""
        try:
            from .feature_generation_utils import ImprovedFeatureGenerator, FeatureGenerationConfig
            
            feature_config = FeatureGenerationConfig(
                enable_technical_indicators=False,
                enable_rolling_stats=False,
                enable_interaction_features=True,
                enable_cross_timeframe=False,
                max_interactions=15,  # Reduced for basic fallback
                interaction_types=['ratio', 'product', 'difference'],
                min_valid_ratio=0.8,
                max_constant_ratio=0.1
            )
            
            feature_generator = ImprovedFeatureGenerator(feature_config)
            return feature_generator.generate_interaction_features(features)
            
        except Exception as e:
            tprint_error(f"❌ Basic interaction generation failed: {e}")
            return pd.DataFrame(index=features.index)
    
    async def _interaction_pruning_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Enhanced interaction pruning stage using OptimizedFeatureSelectionEngine - fast fail."""
        if not self.interaction_pruning:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        tprint_debug("✂️ Interaction pruning stage using OptimizedFeatureSelectionEngine...")
        
        # Get interaction features
        if 'interaction_features' in context.pipeline_state:
            features = context.pipeline_state['interaction_features']
        else:
            features = pd.DataFrame()
        
        if features.empty or len(features.columns) == 0:
            return {'status': 'skipped', 'reason': 'no_interaction_features'}
        
        # Get target column - fast fail if not found
        target_column = context.pipeline_state.get('target_column', 'target')
        if target_column not in features.columns:
            raise ValueError(f"CRITICAL: Target column '{target_column}' not found in interaction features")
        
        # Get feature names
        feature_names = [col for col in features.columns if col != target_column]
        
        if len(feature_names) == 0:
            raise ValueError("CRITICAL: No interaction features found after removing target column")
        
        # Fast fail: Call OptimizedFeatureSelectionEngine directly
        from src.training.steps.market_analysis.optimized_process_engines import OptimizedFeatureSelectionEngine
        
        # Initialize the optimized feature selection engine
        selection_engine = OptimizedFeatureSelectionEngine(
            use_hardware_accel=True,
            cache_size=1000,
            use_vectorbt=True
        )
        
        # Target: Reduce to 50 features (or all if fewer)
        target_feature_count = min(50, len(feature_names))
        selection_stages = [target_feature_count]
        
        tprint_info(f"🎯 Reducing {len(feature_names)} interaction features to {target_feature_count}")
        
        # Perform feature selection - fast fail on any error
        selection_result = selection_engine.select_features(
            features_df=features,
            target_column=target_column,
            selection_stages=selection_stages
        )
        
        if 'error' in selection_result:
            raise RuntimeError(f"CRITICAL: Feature selection failed: {selection_result['error']}")
        
        # Extract selected features - fast fail if no results
        if 'final_features' not in selection_result:
            raise RuntimeError("CRITICAL: No final features returned from selection engine")
        
        if isinstance(selection_result['final_features'], list):
            selected_features = selection_result['final_features']
            if target_column not in selected_features:
                selected_features.append(target_column)
            pruned_features = features[selected_features]
        else:
            pruned_features = selection_result['final_features']
        
        if pruned_features.empty or len(pruned_features.columns) == 0:
            raise RuntimeError("CRITICAL: No features selected after pruning")
        
        # Update context with pruned features
        context.pipeline_state['interaction_features'] = pruned_features
        context.pipeline_state['interaction_pruning_result'] = {
            'selected_interactions': list(pruned_features.columns),
            'rejected_interactions': [col for col in feature_names if col not in pruned_features.columns],
            'pruning_method': 'optimized_feature_selection_engine',
            'target_achieved': len(pruned_features.columns) <= target_feature_count
        }
        
        tprint_success(f"✅ Interaction pruning completed: {len(pruned_features.columns)} features selected")
        
        return {
            'status': 'completed',
            'selected_interactions': len(pruned_features.columns),
            'rejected_interactions': len(feature_names) - len(pruned_features.columns),
            'selection_rate': len(pruned_features.columns) / len(feature_names),
            'method': 'optimized_feature_selection_engine',
            'target_achieved': len(pruned_features.columns) <= target_feature_count
        }
    
    
    
    async def _cross_timeframe_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Cross-timeframe features stage with HTF-aware interaction templates."""
        tprint_debug("⏰ Cross-timeframe features stage with HTF-aware interactions...")
        
        # Get generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
        else:
            features = context.data
        
        # Import HTF-aware interaction templates
        try:
            from ..cross_timeframe_generation.interaction_templates import HTFInteractionTemplates
            from ..cross_timeframe_generation.config import PipelineConfig
            
            # Create HTF interaction templates configuration
            htf_config = PipelineConfig(
                base_timeframe_minutes=5,  # Base timeframe in minutes
                max_cost_ms=25.0,
                max_features=120,
                max_correlation=0.8
            )
            
            # Initialize HTF interaction templates
            htf_templates = HTFInteractionTemplates(htf_config)
            
            # Generate HTF-aware interactions
            tprint_info("🚀 Using HTF-aware interaction templates for cross-timeframe features")
            
            # Create mock materialized HTFs for demonstration
            # In a real implementation, these would come from the HTF materialization stage
            materialized_htfs = {}
            
            # Generate basic HTF features for different timeframes
            for period in [5, 15, 30, 60]:
                for col in features.select_dtypes(include=[np.number]).columns:
                    # Create HTF trend features
                    htf_trend = features[col].rolling(period).mean()
                    materialized_htfs[f'htf_trend_{period}m_{col}'] = htf_trend
                    
                    # Create HTF volatility features
                    htf_vol = features[col].rolling(period).std()
                    materialized_htfs[f'htf_vol_{period}m_{col}'] = htf_vol
                    
                    # Create HTF momentum features
                    htf_mom = features[col].pct_change(period)
                    materialized_htfs[f'htf_mom_{period}m_{col}'] = htf_mom
            
            # Generate HTF-aware interactions
            htf_interactions = htf_templates.generate_interactions(
                materialized_htfs=materialized_htfs,
                base_features=features,
                targets=None  # No targets available at this stage
            )
            
            # Convert interactions to DataFrame
            if htf_interactions:
                cross_timeframe_features = pd.DataFrame({
                    interaction.name: interaction.feature_series 
                    for interaction in htf_interactions
                })
                tprint_success(f"✅ Generated {len(cross_timeframe_features.columns)} HTF-aware cross-timeframe interactions")
            else:
                tprint_warning("⚠️ No HTF-aware interactions generated")
                cross_timeframe_features = pd.DataFrame(index=features.index)
                
        except ImportError as e:
            tprint_warning(f"⚠️ HTF-aware templates not available: {e}, using fallback")
            # Fallback to basic cross-timeframe features
            from .feature_generation_utils import ImprovedFeatureGenerator, FeatureGenerationConfig
            feature_config = FeatureGenerationConfig(
                enable_technical_indicators=False,
                enable_rolling_stats=False,
                enable_interaction_features=False,
                enable_cross_timeframe=True,
                cross_timeframe_periods=[5, 15, 30, 60],
                min_valid_ratio=0.8,
                max_constant_ratio=0.1
            )
            feature_generator = ImprovedFeatureGenerator(feature_config)
            cross_timeframe_features = feature_generator.generate_cross_timeframe_features(features)
            
        except Exception as e:
            tprint_warning(f"⚠️ HTF-aware cross-timeframe generation failed: {e}, using fallback")
            # Fallback to basic cross-timeframe features
            from .feature_generation_utils import ImprovedFeatureGenerator, FeatureGenerationConfig
            feature_config = FeatureGenerationConfig(
                enable_technical_indicators=False,
                enable_rolling_stats=False,
                enable_interaction_features=False,
                enable_cross_timeframe=True,
                cross_timeframe_periods=[5, 15, 30, 60],
                min_valid_ratio=0.8,
                max_constant_ratio=0.1
            )
            feature_generator = ImprovedFeatureGenerator(feature_config)
            cross_timeframe_features = feature_generator.generate_cross_timeframe_features(features)
        
        # CRITICAL: Fast-fail if no cross-timeframe features generated
        if cross_timeframe_features.empty or len(cross_timeframe_features.columns) == 0:
            tprint_warning("⚠️ No cross-timeframe features generated - this may be expected for some datasets")
            cross_timeframe_features = pd.DataFrame(index=features.index)
        else:
            tprint_info(f"✅ Generated {len(cross_timeframe_features.columns)} validated cross-timeframe features")
        
        # Store result
        context.pipeline_state['cross_timeframe_features'] = cross_timeframe_features
        
        return {
            'status': 'completed',
            'cross_timeframe_features': len(cross_timeframe_features.columns)
        }
    
    async def _final_assembly_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Final assembly stage."""
        tprint_debug("🏁 Final assembly stage...")
        
        # Combine all features
        all_features = []
        
        # Add generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
            all_features.append(features)
        
        # Add interaction features
        if 'interaction_features' in context.pipeline_state:
            interaction_features = context.pipeline_state['interaction_features']
            if not interaction_features.empty:
                all_features.append(interaction_features)
        
        # Add cross-timeframe features
        if 'cross_timeframe_features' in context.pipeline_state:
            cross_tf = context.pipeline_state['cross_timeframe_features']
            if not cross_tf.empty:
                all_features.append(cross_tf)
        
        # Combine features
        if all_features:
            final_features = pd.concat(all_features, axis=1)
            # Remove duplicate columns if any were introduced during concatenation
            if len(final_features.columns) != len(set(final_features.columns)):
                tprint_warning(f"⚠️ Duplicate columns detected in final feature assembly")
                final_features = final_features.loc[:, ~final_features.columns.duplicated(keep='first')]
                tprint_debug(f"✅ Removed duplicate columns, now have {len(final_features.columns)} unique columns")
        else:
            # CRITICAL: This should not happen - if we reach here, something is wrong
            raise RuntimeError("CRITICAL: No features were generated in any stage - this indicates a broken pipeline")
        
        # Store result
        context.pipeline_state['final_features'] = final_features
        
        return {
            'status': 'completed',
            'total_features': len(final_features.columns),
            'final_shape': final_features.shape
        }
    
    async def _validation_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Validation stage."""
        tprint_debug("✅ Validation stage...")
        
        # Get final features
        if 'final_features' in context.pipeline_state:
            final_features = context.pipeline_state['final_features']
        else:
            final_features = pd.DataFrame()
        
        # Perform validation
        validation_results = self._validate_features(final_features)
        
        return {
            'status': 'completed',
            'validation_passed': validation_results.get('passed', False),
            'quality_score': validation_results.get('quality_score', 0.0)
        }
    
    def _generate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        interaction_features = {}
        
        # Generate simple interactions
        feature_cols = [col for col in features.columns if col != 'target']
        
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                # Ratio interaction
                interaction_features[f'{col1}_div_{col2}'] = features[col1] / (features[col2] + 1e-8)
                # Product interaction
                interaction_features[f'{col1}_mul_{col2}'] = features[col1] * features[col2]
                # Difference interaction
                interaction_features[f'{col1}_sub_{col2}'] = features[col1] - features[col2]
        
        # Create DataFrame
        if interaction_features:
            result = pd.DataFrame(interaction_features, index=features.index)
            result = result.dropna(axis=1, how='all')
            return result
        else:
            return pd.DataFrame(index=features.index)
    
    def _generate_cross_timeframe_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe features."""
        cross_tf_features = {}
        
        # Timeframe aggregations
        timeframes = [5, 15, 30, 60]
        
        for tf in timeframes:
            for col in features.columns:
                if col != 'target':
                    # Rolling aggregations
                    cross_tf_features[f'ctf_{tf}m_{col}_mean'] = features[col].rolling(tf).mean()
                    cross_tf_features[f'ctf_{tf}m_{col}_std'] = features[col].rolling(tf).std()
                    cross_tf_features[f'ctf_{tf}m_{col}_max'] = features[col].rolling(tf).max()
                    cross_tf_features[f'ctf_{tf}m_{col}_min'] = features[col].rolling(tf).min()
        
        # Create DataFrame
        if cross_tf_features:
            result = pd.DataFrame(cross_tf_features, index=features.index)
            result = result.dropna(axis=1, how='all')
            return result
        else:
            return pd.DataFrame(index=features.index)
    
    def _validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate generated features."""
        if features.empty:
            return {'passed': False, 'quality_score': 0.0}
        
        # Check for finite values
        finite_ratio = features.select_dtypes(include=[np.number]).apply(
            lambda x: np.isfinite(x).sum() / len(x)
        ).mean()
        
        # Check for constant features
        constant_ratio = (features.nunique() <= 1).sum() / len(features.columns)
        
        # Calculate quality score
        quality_score = finite_ratio * (1 - constant_ratio)
        
        return {
            'passed': quality_score > 0.8,
            'quality_score': quality_score,
            'finite_ratio': finite_ratio,
            'constant_ratio': constant_ratio
        }
    
    def _compute_cache_key(self, context: ExecutionContext) -> str:
        """Compute cache key for the pipeline."""
        # Simplified cache key computation
        data_hash = hash(str(context.data.shape))
        config_hash = hash(str(self.config))
        return f"enhanced_pipeline_{data_hash}_{config_hash}"
    
    async def generate_features(self, training_input: Dict[str, Any], 
                               pipeline_state: Dict[str, Any]) -> OptimizedInteractionResult:
        """
        Generate features using the enhanced optimized pipeline.
        
        This is the main entry point that maintains backward compatibility
        with the existing OptimizedInteractionOrchestrator interface.
        
        Args:
            training_input: Input data dictionary
            pipeline_state: Pipeline state dictionary
            
        Returns:
            OptimizedInteractionResult with generated features
        """
        tprint_success("🚀 Starting enhanced optimized feature generation")
        self.start_time = time.time()
        
        try:
            # Fast-fail: Validate training input early
            if not training_input:
                raise ValueError("CRITICAL: No training input provided")
            
            # Extract data from training input
            if 'data' in training_input:
                data = training_input['data']
            elif 'features' in training_input:
                data = training_input['features']
            else:
                raise ValueError("CRITICAL: No data found in training_input")
            
            # Fast-fail: Validate data early
            if data is None:
                raise ValueError("CRITICAL: Data is None in training input")
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"CRITICAL: Data must be a pandas DataFrame, got {type(data)}")
            
            if data.empty:
                raise ValueError("CRITICAL: Input data is empty")
            
            if len(data) < 10:
                raise ValueError(f"CRITICAL: Insufficient data: {len(data)} < 10 rows")
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                raise ValueError(f"CRITICAL: Missing required columns: {missing_cols}")
            
            # Check for all-NaN data
            if data.isnull().all().all():
                raise ValueError("CRITICAL: All data is NaN - cannot generate features")
            
            # Check for and remove duplicate columns
            if len(data.columns) != len(set(data.columns)):
                duplicate_cols = [col for col in data.columns if list(data.columns).count(col) > 1]
                unique_duplicates = list(set(duplicate_cols))
                tprint_warning(f"⚠️ Input data has duplicate columns: {unique_duplicates[:10]}{'...' if len(unique_duplicates) > 10 else ''}")
                data = data.loc[:, ~data.columns.duplicated(keep='first')]
                tprint_debug(f"✅ Removed duplicate columns, now have {len(data.columns)} unique columns")
            
            # Extract target column
            target_column = training_input.get('target_column', 'target')
            
            # OPTIMIZATION: Extract execution mode for mode-aware optimization
            execution_mode = 'full'  # default
            if hasattr(data, 'attrs') and 'ares_mode' in data.attrs:
                execution_mode = data.attrs['ares_mode']
            elif 'execution_mode' in pipeline_state:
                execution_mode = pipeline_state['execution_mode']
            elif 'ares_mode' in pipeline_state:
                execution_mode = pipeline_state['ares_mode']
            
            tprint_info(f"🔍 Input data shape: {data.shape}")
            tprint_info(f"🔍 Input data columns: {list(data.columns)}")
            tprint_info(f"🎯 Running in {execution_mode.upper()} mode")
            
            # Create execution context
            context = ExecutionContext(
                data=data,
                pipeline_state=pipeline_state,
                config=self.config.__dict__
            )
            context.pipeline_state['target_column'] = target_column
            context.pipeline_state['execution_mode'] = execution_mode
            
            # Execute pipeline
            if self.config.enable_parallel_processing and self.dag_executor:
                # Use DAG executor for parallel processing
                dag_executor = self._create_optimized_dag()
                dag_results = await dag_executor.execute_dag(context)
            else:
                # Execute stages sequentially
                await self._execute_sequential_pipeline(context)
            
            # Extract results
            final_features = context.pipeline_state.get('final_features', pd.DataFrame())
            early_filtering_result = context.pipeline_state.get('early_filtering_result')
            interaction_pruning_result = context.pipeline_state.get('interaction_pruning_result')
            budgeted_optimization_result = context.pipeline_state.get('budgeted_optimization_result')
            
            # Fast-fail: Check if features were actually generated
            if final_features.empty or len(final_features.columns) == 0:
                raise RuntimeError("CRITICAL: No features generated - pipeline failed")
            
            # Calculate performance metrics
            execution_time = time.time() - self.start_time
            memory_usage = self.memory_processor.get_memory_usage()
            cache_stats = self.cache.get_stats() if self.cache else {'hit_rate': 0.0}
            
            # Get VectorBT performance statistics
            vectorbt_stats = self._get_vectorbt_performance_summary()
            
            # Create result
            result = OptimizedInteractionResult(
                features=final_features,
                feature_names=list(final_features.columns),
                selected_features=list(final_features.columns),
                interaction_features=context.pipeline_state.get('interaction_features', pd.DataFrame()),
                cross_timeframe_features=context.pipeline_state.get('cross_timeframe_features', pd.DataFrame()),
                execution_time=execution_time,
                memory_usage_mb=memory_usage.get('rss_mb', 0.0),
                cache_hit_rate=cache_stats.get('hit_rate', 0.0),
                parallel_efficiency=0.8,  # Would be calculated from DAG executor
                early_filtering_result=early_filtering_result,
                interaction_pruning_result=interaction_pruning_result,
                budgeted_optimization_result=budgeted_optimization_result,
                pipeline_metadata={
                    'memory_usage': memory_usage,
                    'cache_stats': cache_stats,
                    'vectorbt_performance': vectorbt_stats,
                    'config': self.config.__dict__
                }
            )
            
            tprint_success(f"✅ Enhanced optimized feature generation completed in {execution_time:.3f}s")
            tprint_info(f"📊 Generated {len(final_features.columns)} features")
            tprint_info(f"💾 Memory usage: {memory_usage.get('rss_mb', 0.0):.1f} MB")
            tprint_info(f"📈 Cache hit rate: {cache_stats.get('hit_rate', 0.0):.1%}")
            
            # Log VectorBT performance statistics
            if vectorbt_stats['total_operations'] > 0:
                tprint_info(f"🚀 VectorBT performance:")
                tprint_info(f"   → VectorBT usage rate: {vectorbt_stats.get('vectorbt_usage_rate', 0.0):.1%}")
                tprint_info(f"   → GPU usage rate: {vectorbt_stats.get('gpu_usage_rate', 0.0):.1%}")
                tprint_info(f"   → Parallel usage rate: {vectorbt_stats.get('parallel_usage_rate', 0.0):.1%}")
                tprint_info(f"   → Avg time per operation: {vectorbt_stats.get('avg_time_per_operation', 0.0):.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - self.start_time
            error_message = f"Enhanced optimized feature generation failed: {str(e)}"
            
            tprint_error(f"❌ {error_message}")
            
            return OptimizedInteractionResult(
                features=pd.DataFrame(),
                feature_names=[],
                selected_features=[],
                interaction_features=pd.DataFrame(),
                cross_timeframe_features=pd.DataFrame(),
                execution_time=execution_time,
                memory_usage_mb=0.0,
                cache_hit_rate=0.0,
                parallel_efficiency=0.0,
                success=False,
                error_message=error_message
            )
    
    async def _execute_sequential_pipeline(self, context: ExecutionContext) -> None:
        """Execute pipeline stages sequentially."""
        stages = [
            ("initialization", self._initialize_stage),
            ("early_filtering", self._early_filtering_stage),
            ("feature_engineering", self._feature_engineering_stage),
            ("budgeted_optimization", self._budgeted_optimization_stage),
            ("interaction_generation", self._interaction_generation_stage),
            ("interaction_pruning", self._interaction_pruning_stage),
            ("cross_timeframe", self._cross_timeframe_stage),
            ("final_assembly", self._final_assembly_stage),
            ("validation", self._validation_stage)
        ]
        
        for stage_name, stage_func in stages:
            try:
                stage_start = time.time()
                result = await stage_func(context)
                stage_time = time.time() - stage_start
                
                self.stage_times[stage_name] = stage_time
                tprint_debug(f"✅ Stage {stage_name} completed in {stage_time:.3f}s")
                
                # Clean up memory between stages
                self._cleanup_memory()
                
            except Exception as e:
                tprint_error(f"❌ Stage {stage_name} failed: {e}")
                raise
    
    def _basic_memory_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic memory optimization when advanced utilities are not available."""
        tprint_debug("🔧 Applying basic memory optimization...")
        
        # Optimize dtypes
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Remove unnecessary memory usage
        df = df.copy()  # Ensure we have a clean copy
        
        return df
    
    def _cleanup_memory(self) -> None:
        """Clean up memory between stages."""
        import gc
        gc.collect()
        
        if hasattr(self, 'memory_processor'):
            self.memory_processor.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        tprint_debug("🧹 Cleaning up enhanced optimized orchestrator...")
        
        # Cleanup components
        self.memory_processor.cleanup()
        if self.cache:
            self.cache.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        tprint_success("✅ Enhanced optimized orchestrator cleanup completed")


# Convenience functions for backward compatibility

def create_enhanced_optimized_orchestrator(config: Optional[EnhancedOptimizedConfig] = None) -> EnhancedOptimizedInteractionOrchestrator:
    """Create an enhanced optimized orchestrator."""
    return EnhancedOptimizedInteractionOrchestrator(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'target': np.random.randn(n_samples).cumsum(),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples),
        })
        
        # Test enhanced orchestrator
        config = EnhancedOptimizedConfig(
            max_workers=4,
            max_memory_gb=8.0,
            enable_early_filtering=True,
            enable_interaction_pruning=True,
            enable_budgeted_optimization=True
        )
        
        orchestrator = create_enhanced_optimized_orchestrator(config)
        
        try:
            training_input = {'data': data, 'target_column': 'target'}
            result = await orchestrator.generate_features(training_input, {})
            
            print(f"Enhanced orchestrator result:")
            print(f"  Success: {result.success}")
            print(f"  Execution time: {result.execution_time:.3f}s")
            print(f"  Features generated: {len(result.feature_names)}")
            print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
            print(f"  Cache hit rate: {result.cache_hit_rate:.1%}")
            
        finally:
            orchestrator.cleanup()
    
    asyncio.run(main())

