"""
Optimized Interaction Feature Generation Orchestrator

This module provides a fully wired interaction feature generation pipeline that:
1. Gets features from feature_engineering bank
2. Selects features for lookback optimization
3. Generates cross-timeframe features and interaction features
4. Uses matrix operations and hardware optimization
5. Integrates with ares_launcher and sub_pipeline

Key Features:
- Extensive tprint logging throughout
- Vectorized computations using matrix_operations/
- M1 hardware optimization
- Integration with all utility modules
- Consistent with sub_pipeline architecture
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

from src.feature_generation.core.feature_cache import FeatureCacheService
from src.feature_generation.core.feature_bank import FeatureBank
from ...settings import get_pre_training_settings

# Import common operations and utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        optimize_memory_usage, parallel_processing_optimizer, safe_correlation,
        safe_dataframe_operation, optimize_dataframe_dtypes, safe_fillna,
        create_data_quality_report, get_dataframe_info, safe_rolling,
        safe_groupby_operation, safe_apply_function, safe_filter_dataframe,
        create_summary_statistics, safe_to_parquet, safe_read_parquet,
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Common operations not available: {e}")
    COMMON_OPS_AVAILABLE = False

    def safe_divide(a, b):
        return np.divide(a, b)

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

    def safe_correlation(x, y):
        return np.corrcoef(x, y)[0, 1]

    def safe_dataframe_operation(df, func, *args, **kwargs):
        return func(df, *args, **kwargs)

    def optimize_dataframe_dtypes(df):
        return df

    def safe_fillna(obj, value=0):
        return obj.fillna(value)

    def create_data_quality_report(*args, **kwargs):
        return {}

    def get_dataframe_info(df):
        return {
            "shape": getattr(df, "shape", None),
            "columns": list(getattr(df, "columns", [])),
        }

    def safe_rolling(series, *args, **kwargs):
        return series.rolling(*args, **kwargs)

    def safe_groupby_operation(df, *args, **kwargs):
        groupby_obj = df.groupby(*args, **kwargs)
        return groupby_obj

    def safe_apply_function(obj, func, *args, **kwargs):
        return func(obj, *args, **kwargs)

    def safe_filter_dataframe(df, condition):
        return df[condition]

    def create_summary_statistics(df):
        return df.describe(include="all")

    def safe_to_parquet(*args, **kwargs):
        pass

    def safe_read_parquet(*args, **kwargs):
        return None

    def memory_checkpoint(*args, **kwargs):
        return None

    class gpu_context:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def optimize_memory(*args, **kwargs):
        return None

    def get_memory_usage(*args, **kwargs):
        return 0.0

# Import math validation
from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, validate_finite as math_validate_finite,
    safe_correlation as math_safe_correlation, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse
)

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, safe_matrix_multiply,
        vectorized_rolling_features, parallel_feature_engineering,
        optimize_dataframe, get_hardware_performance_report,
        compute_trading_indicators, compute_moving_averages,
        compute_momentum_indicators, compute_volatility_indicators,
        compute_volume_indicators, compute_trend_indicators,
        compute_oscillator_indicators, compute_pattern_indicators,
        matrix_correlation_analysis, batch_matrix_multiply,
        batch_feature_transformation, batch_correlation_analysis,
        create_ml_pipeline, execute_ml_pipeline, optimize_pipeline_config,
        get_pipeline_executor, optimize_batch_size, record_batch_performance,
        get_batch_optimization_stats, cleanup_hardware_resources,
        get_processing_performance_stats
    )
    MATRIX_OPS_AVAILABLE = True
    tprint_success("✅ Matrix operations module loaded successfully")
except ImportError as e:
    tprint_warning(f"⚠️ Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.cross_validation import PurgedKFold
    from src.feature_selection import select_features as FeatureSelector
    from src.utils.ml_common.data_leakage import DataLeakageDetector
    from src.utils.ml_common.lookahead_bias import LookaheadBiasDetector
    from src.utils.ml_common.hyperparameter_optimization import HPOptimizer
    from src.utils.ml_common.model_validation import ModelValidator
    from src.utils.ml_common.out_of_fold import OutOfFoldPredictor
    ML_COMMON_AVAILABLE = True
    tprint_success("✅ ML common utilities loaded successfully")
except ImportError as e:
    tprint_warning(f"⚠️ ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# Import data utilities
try:
    from src.utils.data.data_loader import DataLoader
    from src.utils.data.data_validation import DataValidator
    from src.utils.kline_parquet import KlineParquetLoader
    from src.utils.serialization_utils import save_pickle, load_pickle
    from src.utils.data.data_preprocessing import DataPreprocessor
    from src.utils.data.feature_engineering import FeatureEngineer
    from src.utils.data.time_series_utils import TimeSeriesProcessor
    DATA_UTILS_AVAILABLE = True
    tprint_success("✅ Data utilities loaded successfully")
except ImportError as e:
    tprint_warning(f"⚠️ Data utilities not available: {e}")
    DATA_UTILS_AVAILABLE = False

# Import feature engineering components
from .feature_engineering.assembly_dag import AssemblyDAG, AssemblyConfig, AssemblyResult
from .feature_engineering.lookback_selection import LookbackSelector, create_feature_families
from .feature_engineering.transforms import TransformRouter, create_default_transform_config
from .feature_engineering.interactions import InteractionEngine, create_default_interaction_config
from .feature_engineering.feature_registry import FeatureRegistry, FeatureFamily

# Import orchestrator components
from .orchestrator import LookbackOptimizationOrchestrator, OptimizationResult
from .config import LookbackOptimizationConfig, create_default_config

# Setup logging
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZATION = "initialization"
    FEATURE_ENGINEERING = "feature_engineering"
    LOOKBACK_OPTIMIZATION = "lookback_optimization"
    TRANSFORM_APPLICATION = "transform_application"
    INTERACTION_GENERATION = "interaction_generation"
    CROSS_TIMEFRAME = "cross_timeframe"
    FINAL_ASSEMBLY = "final_assembly"
    VALIDATION = "validation"
    COMPLETION = "completion"


def _default_data_directory() -> str:
    return str(get_pre_training_settings().data_root)


@dataclass
class OptimizedInteractionConfig:
    """Configuration for optimized interaction feature generation."""
    # Pipeline configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = field(default_factory=_default_data_directory)
    
    # Feature engineering configuration
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
    
    # Lookback optimization configuration
    lookback_config: Optional[LookbackOptimizationConfig] = None
    
    # Validation configuration
    enable_validation: bool = True
    validation_threshold: float = 0.02

    # Logging configuration
    verbose_logging: bool = True
    log_performance: bool = True

    # Market data streaming configuration
    market_data_batch_size: Optional[int] = None
    market_data_window_days: Optional[int] = None
    
    def __post_init__(self):
        if self.lookback_config is None:
            self.lookback_config = create_default_config()


@dataclass
class OptimizedInteractionResult:
    """Result of optimized interaction feature generation."""
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
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Stage results
    stage_results: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)

    # Artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)


class OptimizedInteractionOrchestrator:
    """Main orchestrator for optimized interaction feature generation."""
    
    def __init__(self, config: OptimizedInteractionConfig):
        tprint_info("🔧 Initializing Optimized Interaction Orchestrator...")
        
        self.config = config
        self.logger = logger.getChild('OptimizedInteractionOrchestrator')
        
        # Performance tracking
        self.performance_metrics = {}
        self.stage_start_times = {}
        self.memory_usage_history = []
        self.gpu_usage_history = []

        # Cache management
        self.feature_cache = FeatureCacheService(subdirectory="interaction_orchestrator")
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0,
        }
        self._active_cache_key: Optional[str] = None
        self._current_lookback_hash: Optional[str] = None
        self._force_cache_refresh: bool = False
        
        # Initialize components with extensive logging
        tprint_info("🔧 Initializing pipeline components...")
        self._initialize_components()

        # Log initialization summary
        tprint_success("🚀 Optimized Interaction Orchestrator initialized successfully")
        tprint_info(f"📊 Configuration:")
        tprint_info(f"   - Symbol: {config.symbol}")
        tprint_info(f"   - Exchange: {config.exchange}")
        tprint_info(f"   - Timeframe: {config.timeframe}")
        tprint_info(f"   - Feature budget (pre): {config.feature_budget_pre}")
        tprint_info(f"   - Feature budget (post): {config.feature_budget_post}")
        tprint_info(f"   - Interactions cap: {config.interactions_cap}")
        tprint_info(f"   - Lookback ceiling: {config.lookback_ceiling_minutes} minutes")
        tprint_info(f"   - Latency budget: {config.latency_budget_ms} ms")
        
        tprint_info(f"🔧 Available modules:")
        tprint_info(f"   - Matrix operations: {'✅' if MATRIX_OPS_AVAILABLE else '❌'}")
        tprint_info(f"   - ML common utilities: {'✅' if ML_COMMON_AVAILABLE else '❌'}")
        tprint_info(f"   - Data utilities: {'✅' if DATA_UTILS_AVAILABLE else '❌'}")
        tprint_info(f"   - Hardware optimization: {'✅' if self.m1_gpu_manager else '❌'}")
        tprint_info(f"   - Memory optimization: {'✅' if self.m1_memory_optimizer else '❌'}")
        tprint_info(f"   - CPU optimization: {'✅' if self.m1_cpu_optimizer else '❌'}")

        # Initialize performance monitoring
        self._initialize_performance_monitoring()

    def _compute_cache_key(self, pipeline_state: Dict[str, Any]) -> Optional[str]:
        if pipeline_state is None:
            return None

        symbol = pipeline_state.get('symbol', self.config.symbol)
        timeframe = pipeline_state.get('timeframe', self.config.timeframe)
        lookback_config = pipeline_state.get('lookback_config', self.config.lookback_config)
        lookback_hash = FeatureCacheService.compute_config_hash(lookback_config)
        self._current_lookback_hash = lookback_hash
        pipeline_state['lookback_config_hash'] = lookback_hash

        version = getattr(FeatureBank, 'VERSION', 'unknown')
        cache_key = FeatureCacheService.build_key(symbol, timeframe, version, lookback_hash)
        pipeline_state['feature_cache_key'] = cache_key
        return cache_key

    def _sync_cache_metrics(self) -> None:
        self.performance_metrics.setdefault('cache_metrics', {})
        self.performance_metrics['cache_metrics'] = dict(self.cache_metrics)
        self.performance_metrics['feature_cache_key'] = self._active_cache_key
        self.performance_metrics['feature_cache_force_refresh'] = self._force_cache_refresh
    
    def _initialize_components(self):
        """Initialize all pipeline components with comprehensive logging."""
        tprint_info("🔧 Initializing pipeline components...")
        
        # Initialize feature registry
        tprint_debug("📋 Initializing feature registry...")
        try:
            self.feature_registry = FeatureRegistry()
            feature_count = len(self.feature_registry.get_all_features())
            tprint_success(f"✅ Feature registry initialized with {feature_count} features")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize feature registry: {e}")
            raise
        
        # Initialize Assembly DAG
        tprint_debug("🏗️ Initializing Assembly DAG...")
        try:
            assembly_config = AssemblyConfig(
                feature_budget_pre=self.config.feature_budget_pre,
                feature_budget_post=self.config.feature_budget_post,
                interactions_cap=self.config.interactions_cap,
                transforms_per_parent=self.config.transforms_per_parent,
                lookback_ceiling_minutes=self.config.lookback_ceiling_minutes,
                latency_budget_ms=self.config.latency_budget_ms
            )
            self.assembly_dag = AssemblyDAG(assembly_config)
            tprint_success("✅ Assembly DAG initialized")
            tprint_debug(f"   - Feature budget (pre): {assembly_config.feature_budget_pre}")
            tprint_debug(f"   - Feature budget (post): {assembly_config.feature_budget_post}")
            tprint_debug(f"   - Interactions cap: {assembly_config.interactions_cap}")
            tprint_debug(f"   - Lookback ceiling: {assembly_config.lookback_ceiling_minutes} minutes")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Assembly DAG: {e}")
            raise
        
        # Initialize lookback optimization orchestrator
        tprint_debug("🎯 Initializing lookback optimization orchestrator...")
        try:
            self.lookback_orchestrator = LookbackOptimizationOrchestrator(self.config.lookback_config)
            tprint_success("✅ Lookback optimization orchestrator initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize lookback orchestrator: {e}")
            raise
        
        # Initialize lookback selector
        tprint_debug("🔍 Initializing lookback selector...")
        try:
            self.lookback_selector = LookbackSelector()
            tprint_success("✅ Lookback selector initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize lookback selector: {e}")
            raise
        
        # Initialize matrix operations - fail fast if not available
        if MATRIX_OPS_AVAILABLE:
            tprint_debug("🧮 Initializing matrix operations...")
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                tprint_success("✅ Matrix operations initialized")
                tprint_debug("   - Unified matrix operations: ✅")
                tprint_debug("   - Vectorized processing core: ✅")
                tprint_debug("   - Batch matrix processor: ✅")
            except Exception as e:
                tprint_error(f"❌ Critical error: Matrix operations initialization failed: {e}")
                raise RuntimeError(f"Matrix operations initialization failed: {e}")
        else:
            tprint_error("❌ Critical error: Matrix operations not available")
            raise ImportError("Matrix operations module not available")
        
        # Initialize hardware optimizers - fail fast if not available
        tprint_debug("🖥️ Initializing M1 hardware optimizers...")
        try:
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            tprint_success("✅ M1 hardware optimizers initialized")
            tprint_debug("   - M1 GPU manager: ✅")
            tprint_debug("   - M1 memory optimizer: ✅")
            tprint_debug("   - M1 CPU optimizer: ✅")
        except Exception as e:
            tprint_error(f"❌ Critical error: M1 hardware optimizers initialization failed: {e}")
            raise RuntimeError(f"M1 hardware optimizers initialization failed: {e}")
        
        # Initialize ML common utilities - fail fast if not available
        if ML_COMMON_AVAILABLE:
            tprint_debug("🤖 Initializing ML common utilities...")
            try:
                self.bayesian_optimizer = BayesianTPEOptimizer()
                self.feature_selector = FeatureSelector()
                self.data_leakage_detector = DataLeakageDetector()
                self.lookahead_bias_detector = LookaheadBiasDetector()
                self.hp_optimizer = HPOptimizer()
                self.model_validator = ModelValidator()
                self.oof_predictor = OutOfFoldPredictor()
                tprint_success("✅ ML common utilities initialized")
                tprint_debug("   - Bayesian TPE optimizer: ✅")
                tprint_debug("   - Feature selector: ✅")
                tprint_debug("   - Data leakage detector: ✅")
                tprint_debug("   - Lookahead bias detector: ✅")
                tprint_debug("   - Hyperparameter optimizer: ✅")
                tprint_debug("   - Model validator: ✅")
                tprint_debug("   - Out-of-fold predictor: ✅")
            except Exception as e:
                tprint_error(f"❌ Critical error: ML common utilities initialization failed: {e}")
                raise RuntimeError(f"ML common utilities initialization failed: {e}")
        else:
            tprint_error("❌ Critical error: ML common utilities not available")
            raise ImportError("ML common utilities module not available")
        
        # Initialize data utilities - fail fast if not available
        if DATA_UTILS_AVAILABLE:
            tprint_debug("📊 Initializing data utilities...")
            try:
                self.data_loader = DataLoader()
                self.data_validator = DataValidator()
                self.kline_loader = KlineParquetLoader()
                self.data_preprocessor = DataPreprocessor()
                self.feature_engineer = FeatureEngineer()
                self.time_series_processor = TimeSeriesProcessor()
                tprint_success("✅ Data utilities initialized")
                tprint_debug("   - Data loader: ✅")
                tprint_debug("   - Data validator: ✅")
                tprint_debug("   - Kline parquet loader: ✅")
                tprint_debug("   - Data preprocessor: ✅")
                tprint_debug("   - Feature engineer: ✅")
                tprint_debug("   - Time series processor: ✅")
            except Exception as e:
                tprint_error(f"❌ Critical error: Data utilities initialization failed: {e}")
                raise RuntimeError(f"Data utilities initialization failed: {e}")
        else:
            tprint_error("❌ Critical error: Data utilities not available")
            raise ImportError("Data utilities module not available")
        
        tprint_success("✅ All pipeline components initialized successfully")
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring systems."""
        tprint_debug("📊 Initializing performance monitoring...")
        
        try:
            # Initialize memory monitoring
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer.start_monitoring()
                tprint_debug("✅ Memory monitoring started")
            
            # Initialize GPU monitoring
            if self.m1_gpu_manager:
                gpu_info = self.m1_gpu_manager.get_gpu_info()
                tprint_debug(f"✅ GPU monitoring initialized: {gpu_info.get('device_name', 'Unknown')}")
            
            # Initialize CPU monitoring
            if self.m1_cpu_optimizer:
                cpu_info = self.m1_cpu_optimizer.get_cpu_info()
                tprint_debug(f"✅ CPU monitoring initialized: {cpu_info.get('cores', 'Unknown')} cores")
            
            # Initialize performance metrics tracking
            self.performance_metrics = {
                'total_execution_time': 0.0,
                'stage_times': {},
                'memory_usage_mb': 0.0,
                'gpu_usage_percent': 0.0,
                'cpu_usage_percent': 0.0,
                'features_generated': 0,
                'interactions_generated': 0,
                'cross_timeframe_features_generated': 0,
                'optimization_applied': False,
                'hardware_acceleration_used': False
            }
            
            tprint_success("✅ Performance monitoring initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance monitoring initialization failed: {e}")
    
    async def generate_features(self,
                              training_input: Dict[str, Any],
                              pipeline_state: Dict[str, Any]) -> OptimizedInteractionResult:
        """Generate optimized interaction features with comprehensive logging and monitoring."""
        pipeline_state = dict(pipeline_state or {})
        start_time = time.time()
        tprint_success("🚀 Starting optimized interaction feature generation")
        tprint_info(f"📊 Pipeline configuration:")
        tprint_info(f"   - Symbol: {self.config.symbol}")
        tprint_info(f"   - Exchange: {self.config.exchange}")
        tprint_info(f"   - Timeframe: {self.config.timeframe}")
        tprint_info(f"   - Feature budget (pre): {self.config.feature_budget_pre}")
        tprint_info(f"   - Feature budget (post): {self.config.feature_budget_post}")
        tprint_info(f"   - Interactions cap: {self.config.interactions_cap}")
        
        # Initialize performance tracking
        self.stage_start_times = {}
        self.performance_metrics['total_execution_time'] = 0.0
        self.performance_metrics['stage_times'] = {}

        cache_key = self._compute_cache_key(pipeline_state)
        self._active_cache_key = cache_key
        force_refresh = bool(
            pipeline_state.get('feature_cache_force_refresh')
            or pipeline_state.get('force_feature_cache_refresh')
            or pipeline_state.get('force_refresh_features')
        )
        self._force_cache_refresh = force_refresh
        pipeline_state['feature_cache_force_refresh'] = force_refresh

        if force_refresh:
            self.cache_metrics['force_refreshes'] += 1
            tprint_info(f"♻️ Force refresh enabled for cache key {cache_key}")
        elif cache_key:
            cached_interactions = self.feature_cache.load(cache_key, artifact_type="interactions")
            cached_cross = self.feature_cache.load(cache_key, artifact_type="cross_timeframe")
            if (
                cached_interactions is not None
                and not cached_interactions.empty
                and cached_cross is not None
                and not cached_cross.empty
            ):
                self.cache_metrics['hits'] += 1
                pipeline_state['_cached_interaction_features'] = cached_interactions
                pipeline_state['_cached_cross_timeframe_features'] = cached_cross
                tprint_info(f"📦 Reusing cached interaction artifacts for key {cache_key}")
            else:
                self.cache_metrics['misses'] += 1
                tprint_info(f"🔁 Cache miss for interaction artifacts using key {cache_key}")

        self._sync_cache_metrics()
        
        try:
            # Stage 1: Initialization
            tprint_info("🔧 Stage 1: Initialization")
            self.stage_start_times[PipelineStage.INITIALIZATION] = time.time()
            await self._stage_initialization(training_input, pipeline_state)
            self._log_stage_completion(PipelineStage.INITIALIZATION)
            
            # Stage 2: Feature Engineering
            tprint_info("🏗️ Stage 2: Feature Engineering")
            self.stage_start_times[PipelineStage.FEATURE_ENGINEERING] = time.time()
            feature_engineering_result = await self._stage_feature_engineering(training_input, pipeline_state)
            self._log_stage_completion(PipelineStage.FEATURE_ENGINEERING)
            
            # Stage 3: Lookback Optimization
            tprint_info("🎯 Stage 3: Lookback Optimization")
            self.stage_start_times[PipelineStage.LOOKBACK_OPTIMIZATION] = time.time()
            lookback_result = await self._stage_lookback_optimization(feature_engineering_result, pipeline_state)
            self._log_stage_completion(PipelineStage.LOOKBACK_OPTIMIZATION)
            
            # Stage 4: Transform Application
            tprint_info("🔄 Stage 4: Transform Application")
            self.stage_start_times[PipelineStage.TRANSFORM_APPLICATION] = time.time()
            transform_result = await self._stage_transform_application(lookback_result, pipeline_state)
            self._log_stage_completion(PipelineStage.TRANSFORM_APPLICATION)
            
            # Stage 5: Interaction Generation
            tprint_info("🔗 Stage 5: Interaction Generation")
            self.stage_start_times[PipelineStage.INTERACTION_GENERATION] = time.time()
            interaction_result = await self._stage_interaction_generation(transform_result, pipeline_state)
            self._log_stage_completion(PipelineStage.INTERACTION_GENERATION)
            
            # Stage 6: Cross-timeframe Features
            tprint_info("⏰ Stage 6: Cross-timeframe Features")
            self.stage_start_times[PipelineStage.CROSS_TIMEFRAME] = time.time()
            cross_timeframe_result = await self._stage_cross_timeframe_features(interaction_result, pipeline_state)
            self._log_stage_completion(PipelineStage.CROSS_TIMEFRAME)
            
            # Stage 7: Final Assembly
            tprint_info("🏁 Stage 7: Final Assembly")
            self.stage_start_times[PipelineStage.FINAL_ASSEMBLY] = time.time()
            final_result = await self._stage_final_assembly(cross_timeframe_result, pipeline_state)
            self._log_stage_completion(PipelineStage.FINAL_ASSEMBLY)
            
            # Stage 8: Validation
            tprint_info("✅ Stage 8: Validation")
            self.stage_start_times[PipelineStage.VALIDATION] = time.time()
            validation_result = await self._stage_validation(final_result, pipeline_state)
            self._log_stage_completion(PipelineStage.VALIDATION)
            
            # Stage 9: Completion
            tprint_info("🎉 Stage 9: Completion")
            self.stage_start_times[PipelineStage.COMPLETION] = time.time()
            completion_result = await self._stage_completion(validation_result, pipeline_state)
            self._log_stage_completion(PipelineStage.COMPLETION)
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            self.performance_metrics['total_execution_time'] = execution_time
            
            # Log final performance summary
            self._log_performance_summary(completion_result)
            
            tprint_success(f"✅ Feature generation completed successfully in {execution_time:.3f}s")
            tprint_info(f"📊 Generated {completion_result.feature_names.__len__() if completion_result.feature_names else 0} total features")
            tprint_info(f"🎯 Selected {completion_result.selected_features.__len__() if completion_result.selected_features else 0} features")
            tprint_info(f"🔗 Generated {completion_result.interaction_features.shape[1] if not completion_result.interaction_features.empty else 0} interactions")
            tprint_info(f"⏰ Generated {completion_result.cross_timeframe_features.shape[1] if not completion_result.cross_timeframe_features.empty else 0} cross-timeframe features")
            
            return completion_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Feature generation failed: {str(e)}"
            tprint_error(f"❌ {error_message}")
            tprint_error(f"📊 Execution time before failure: {execution_time:.3f}s")
            self.logger.error(f"Feature generation failed: {error_message}", exc_info=True)
            
            # Log performance metrics even on failure
            self.performance_metrics['total_execution_time'] = execution_time
            self._log_performance_summary(None, error=True)
            
            return OptimizedInteractionResult(
                features=pd.DataFrame(),
                feature_names=[],
                selected_features=[],
                interaction_features=pd.DataFrame(),
                cross_timeframe_features=pd.DataFrame(),
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    def _log_stage_completion(self, stage: PipelineStage):
        """Log stage completion with timing information."""
        if stage in self.stage_start_times:
            stage_time = time.time() - self.stage_start_times[stage]
            self.performance_metrics['stage_times'][stage.value] = stage_time
            tprint_performance(f"Stage {stage.value}", stage_time)
    
    def _log_performance_summary(self, result: Optional[OptimizedInteractionResult] = None, error: bool = False):
        """Log comprehensive performance summary."""
        tprint_info("📊 PERFORMANCE SUMMARY")
        tprint_info(f"⏱️ Total execution time: {self.performance_metrics['total_execution_time']:.3f}s")
        
        # Log stage times
        if self.performance_metrics['stage_times']:
            tprint_info("📈 Stage execution times:")
            for stage_name, stage_time in self.performance_metrics['stage_times'].items():
                percentage = (stage_time / self.performance_metrics['total_execution_time']) * 100
                tprint_info(f"   - {stage_name}: {stage_time:.3f}s ({percentage:.1f}%)")
        
        # Log memory usage
        if self.m1_memory_optimizer:
            try:
                memory_usage = get_memory_usage()
                memory_usage_mb = memory_usage / (1024 * 1024)
                tprint_info(f"💾 Memory usage: {memory_usage_mb:.2f} MB")
                self.performance_metrics['memory_usage_mb'] = memory_usage_mb
            except Exception as e:
                tprint_warning(f"⚠️ Could not get memory usage: {e}")
        
        # Log GPU usage if available
        if self.m1_gpu_manager:
            try:
                gpu_info = self.m1_gpu_manager.get_gpu_info()
                if 'utilization' in gpu_info:
                    tprint_info(f"🖥️ GPU utilization: {gpu_info['utilization']:.1f}%")
                    self.performance_metrics['gpu_usage_percent'] = gpu_info['utilization']
            except Exception as e:
                tprint_warning(f"⚠️ Could not get GPU usage: {e}")
        
        # Log feature generation metrics
        if result and not error:
            tprint_info(f"📊 Feature generation metrics:")
            tprint_info(f"   - Total features: {len(result.feature_names) if result.feature_names else 0}")
            tprint_info(f"   - Selected features: {len(result.selected_features) if result.selected_features else 0}")
            tprint_info(f"   - Interaction features: {result.interaction_features.shape[1] if not result.interaction_features.empty else 0}")
            tprint_info(f"   - Cross-timeframe features: {result.cross_timeframe_features.shape[1] if not result.cross_timeframe_features.empty else 0}")
            tprint_info(f"   - Memory usage: {result.memory_usage_mb:.2f} MB")
        
        # Log optimization status
        if self.performance_metrics.get('optimization_applied', False):
            tprint_success("✅ Matrix optimization applied")
        if self.performance_metrics.get('hardware_acceleration_used', False):
            tprint_success("✅ Hardware acceleration used")
    
    async def _stage_initialization(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Initialize pipeline and validate inputs with comprehensive logging."""
        stage_start = time.time()
        tprint_debug("🔧 Stage 1: Initialization - Starting input validation...")
        
        try:
            # Validate inputs
            tprint_debug("📋 Validating training input...")
            if not training_input:
                raise ValueError("No training input provided")
            tprint_success("✅ Training input validation passed")
            
            tprint_debug("📋 Validating pipeline state...")
            if not pipeline_state:
                raise ValueError("No pipeline state provided")
            tprint_success("✅ Pipeline state validation passed")
            
            # Extract and validate data
            tprint_debug("📊 Extracting data from training input...")
            data = training_input.get('data')
            if data is None:
                raise ValueError("No data provided in training input")
            tprint_success("✅ Data extraction successful")
            
            # Validate data type and structure
            tprint_debug("🔍 Validating data structure...")
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            tprint_success("✅ Data type validation passed")
            
            # Validate data size
            tprint_debug(f"📏 Validating data size: {len(data)} rows, {len(data.columns)} columns")
            if len(data) < 100:
                raise ValueError(f"Insufficient data: {len(data)} < 100 rows")
            tprint_success(f"✅ Data size validation passed: {len(data)} rows")
            
            # Check required columns
            tprint_debug("🔍 Validating required columns...")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            tprint_success(f"✅ Required columns validation passed: {required_columns}")
            
            # Calculate data quality metrics
            tprint_debug("📊 Calculating data quality metrics...")
            data_quality_score = self._calculate_data_quality_score(data)
            memory_usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Initialize performance tracking
            self.performance_metrics.update({
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'memory_usage_mb': memory_usage_mb,
                'data_quality_score': data_quality_score,
                'initialization_success': True
            })
            
            tprint_info(f"📊 Data quality metrics:")
            tprint_info(f"   - Rows: {len(data)}")
            tprint_info(f"   - Columns: {len(data.columns)}")
            tprint_info(f"   - Memory usage: {memory_usage_mb:.2f} MB")
            tprint_info(f"   - Quality score: {data_quality_score:.3f}")
            
            # Apply hardware optimization
            if self.m1_memory_optimizer:
                tprint_debug("🖥️ Applying M1 memory optimization...")
                try:
                    self.m1_memory_optimizer.optimize_dataframe(data)
                    tprint_success("✅ M1 memory optimization applied")
                    self.performance_metrics['hardware_acceleration_used'] = True
                except Exception as e:
                    tprint_warning(f"⚠️ M1 memory optimization failed: {e}")
            
            # Apply CPU optimization
            if self.m1_cpu_optimizer:
                tprint_debug("🖥️ Applying M1 CPU optimization...")
                try:
                    self.m1_cpu_optimizer.optimize_numpy_operations()
                    tprint_success("✅ M1 CPU optimization applied")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 CPU optimization failed: {e}")
            
            # Validate data quality
            tprint_debug("🔍 Performing data quality validation...")
            quality_report = create_data_quality_report(data)
            if quality_report.get('issues'):
                tprint_warning(f"⚠️ Data quality issues detected: {quality_report['issues']}")
            else:
                tprint_success("✅ Data quality validation passed")
            
            # Check for data leakage if ML utilities are available
            if self.data_leakage_detector:
                tprint_debug("🔍 Checking for data leakage...")
                try:
                    leakage_result = self.data_leakage_detector.detect_leakage(data)
                    if leakage_result.get('leakage_detected', False):
                        tprint_warning(f"⚠️ Potential data leakage detected: {leakage_result.get('details', 'Unknown')}")
                    else:
                        tprint_success("✅ No data leakage detected")
                except Exception as e:
                    tprint_warning(f"⚠️ Data leakage detection failed: {e}")
            
            stage_time = time.time() - stage_start
            tprint_performance("Initialization", stage_time)
            
            result = {
                'data': data,
                'performance_metrics': self.performance_metrics,
                'quality_report': quality_report,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.INITIALIZATION] = result
            tprint_success("✅ Stage 1: Initialization completed successfully")
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Initialization failed: {e}")
            tprint_error(f"📊 Stage execution time: {stage_time:.3f}s")
            self.performance_metrics['initialization_success'] = False
            raise
    
    async def _stage_feature_engineering(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Generate parent features from market data with comprehensive optimization."""
        stage_start = time.time()
        tprint_debug("🏗️ Stage 2: Feature Engineering - Starting parent feature generation...")
        
        try:
            # Get data from initialization stage
            init_result = self.stage_results[PipelineStage.INITIALIZATION]
            data = init_result['data']
            
            tprint_info(f"📊 Processing data: {len(data)} rows, {len(data.columns)} columns")
            
            # Extract targets if available
            targets = training_input.get('targets', {})
            if targets:
                tprint_info(f"🎯 Targets available: {list(targets.keys())}")
            else:
                tprint_warning("⚠️ No targets provided for feature engineering")
            
            # Use assembly DAG to build parent features
            tprint_debug("🏗️ Building parent features using assembly DAG...")
            try:
                assembly_result = self.assembly_dag.assemble(data, targets)
                tprint_success("✅ Assembly DAG execution completed")
            except Exception as e:
                tprint_error(f"❌ Assembly DAG failed: {e}")
                raise
            
            if assembly_result.status.value != 'completed':
                raise ValueError(f"Assembly DAG failed: {assembly_result.status.value}")
            
            # Extract features
            parent_features = assembly_result.features
            feature_names = assembly_result.feature_names
            
            tprint_success(f"✅ Generated {len(feature_names)} parent features")
            
            # Analyze feature families
            feature_families = {}
            for name in feature_names:
                if '/' in name:
                    family = name.split('/')[1]
                    feature_families[family] = feature_families.get(family, 0) + 1
            
            tprint_info(f"📊 Feature families generated:")
            for family, count in feature_families.items():
                tprint_info(f"   - {family}: {count} features")
            
            # Apply matrix optimization if available
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("🧮 Applying vectorized processing optimization...")
                try:
                    with memory_checkpoint("feature_engineering_vectorization"):
                        parent_features = self.vectorized_core.optimize_dataframe_for_processing(parent_features)
                    tprint_success("✅ Vectorized processing optimization applied")
                    self.performance_metrics['optimization_applied'] = True
                except Exception as e:
                    tprint_warning(f"⚠️ Vectorized processing optimization failed: {e}")
            
            # Apply batch processing optimization if available
            if self.batch_processor and MATRIX_OPS_AVAILABLE:
                tprint_debug("📦 Applying batch processing optimization...")
                try:
                    # Convert DataFrame to batch format for processing
                    feature_arrays = [parent_features[col].values for col in parent_features.columns]
                    optimized_arrays = self.batch_processor.batch_optimize_arrays(feature_arrays)
                    
                    # Reconstruct DataFrame
                    optimized_df = pd.DataFrame(
                        dict(zip(parent_features.columns, optimized_arrays)),
                        index=parent_features.index
                    )
                    parent_features = optimized_df
                    tprint_success("✅ Batch processing optimization applied")
                except Exception as e:
                    tprint_warning(f"⚠️ Batch processing optimization failed: {e}")
            
            # Apply M1 memory optimization
            if self.m1_memory_optimizer:
                tprint_debug("🖥️ Applying M1 memory optimization...")
                try:
                    with memory_checkpoint("feature_engineering_memory_opt"):
                        parent_features = self.m1_memory_optimizer.optimize_dataframe_memory(parent_features)
                    tprint_success("✅ M1 memory optimization applied")
                    self.performance_metrics['hardware_acceleration_used'] = True
                except Exception as e:
                    tprint_warning(f"⚠️ M1 memory optimization failed: {e}")
            
            # Apply data type optimization
            tprint_debug("🔧 Optimizing data types...")
            try:
                parent_features = optimize_dataframe_dtypes(parent_features)
                tprint_success("✅ Data type optimization applied")
            except Exception as e:
                tprint_warning(f"⚠️ Data type optimization failed: {e}")
            
            # Calculate feature quality metrics
            tprint_debug("📊 Calculating feature quality metrics...")
            quality_metrics = self._calculate_feature_quality_metrics(parent_features)
            
            # Update performance metrics
            self.performance_metrics.update({
                'features_generated': len(feature_names),
                'feature_families': len(feature_families),
                'feature_quality_score': quality_metrics.get('overall_quality', 0.0),
                'feature_engineering_success': True
            })
            
            tprint_info(f"📊 Feature engineering metrics:")
            tprint_info(f"   - Features generated: {len(feature_names)}")
            tprint_info(f"   - Feature families: {len(feature_families)}")
            tprint_info(f"   - Quality score: {quality_metrics.get('overall_quality', 0.0):.3f}")
            tprint_info(f"   - Memory usage: {parent_features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            stage_time = time.time() - stage_start
            tprint_performance("Feature Engineering", stage_time)
            
            result = {
                'parent_features': parent_features,
                'feature_names': feature_names,
                'assembly_result': assembly_result,
                'feature_families': feature_families,
                'quality_metrics': quality_metrics,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.FEATURE_ENGINEERING] = result
            tprint_success("✅ Stage 2: Feature Engineering completed successfully")
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Feature engineering failed: {e}")
            tprint_error(f"📊 Stage execution time: {stage_time:.3f}s")
            self.performance_metrics['feature_engineering_success'] = False
            raise
    
    def _calculate_feature_quality_metrics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive feature quality metrics."""
        try:
            metrics = {
                'total_features': len(features.columns),
                'finite_features': 0,
                'infinite_features': 0,
                'nan_features': 0,
                'constant_features': 0,
                'high_variance_features': 0,
                'low_variance_features': 0,
                'overall_quality': 0.0
            }
            
            if features.empty:
                return metrics
            
            for col in features.columns:
                col_data = features[col].dropna()
                
                if len(col_data) == 0:
                    metrics['nan_features'] += 1
                    continue
                
                # Check for finite values
                finite_count = np.isfinite(col_data).sum()
                if finite_count == len(col_data):
                    metrics['finite_features'] += 1
                else:
                    metrics['infinite_features'] += 1
                
                # Check for constant features
                if col_data.nunique() <= 1:
                    metrics['constant_features'] += 1
                
                # Check variance
                if len(col_data) > 1:
                    variance = col_data.var()
                    if variance > 1.0:
                        metrics['high_variance_features'] += 1
                    elif variance < 0.01:
                        metrics['low_variance_features'] += 1
            
            # Calculate overall quality score
            total_features = metrics['total_features']
            if total_features > 0:
                quality_score = (
                    metrics['finite_features'] * 0.4 +
                    (total_features - metrics['constant_features']) * 0.3 +
                    (total_features - metrics['nan_features']) * 0.3
                ) / total_features
                metrics['overall_quality'] = quality_score
            
            return metrics

        except Exception as e:
            tprint_error(f"❌ Critical error: Feature quality calculation failed: {e}")
            raise RuntimeError(f"Feature quality calculation failed: {e}")
    
    async def _stage_lookback_optimization(self, feature_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Optimize lookback periods for features with advanced ML utilities."""
        stage_start = time.time()
        tprint_debug("🎯 Stage 3: Lookback Optimization - Starting lookback period optimization...")
        
        try:
            # Get parent features from feature engineering stage
            parent_features = feature_result['parent_features']
            feature_names = feature_result['feature_names']
            feature_families = feature_result.get('feature_families', {})
            
            tprint_info(f"📊 Optimizing lookbacks for {len(feature_names)} features")
            
            # Extract targets
            targets = pipeline_state.get('targets', {})
            if not targets:
                tprint_warning("⚠️ No targets available for lookback optimization")
                # Create dummy targets for optimization
                targets = {1: pd.Series(0, index=parent_features.index)}
                tprint_debug("Created dummy targets for optimization")
            else:
                tprint_info(f"🎯 Using {len(targets)} target series for optimization")
            
            # Create feature families if not already created
            if not feature_families:
                tprint_debug("🔍 Creating feature families...")
                feature_families = create_feature_families(feature_names)
                tprint_success(f"✅ Created {len(feature_families)} feature families")
            else:
                tprint_info(f"📊 Using existing {len(feature_families)} feature families")
            
            # Log feature families
            for family, features in feature_families.items():
                tprint_debug(f"   - {family}: {len(features)} features")
            
            # Use lookback selector with enhanced logging
            tprint_debug("🔍 Selecting optimal lookbacks using nested CV...")
            try:
                lookback_choices = self.lookback_selector.select_lookbacks(
                    parent_features, 
                    targets.get(1, pd.Series(0, index=parent_features.index)),
                    feature_families
                )
                tprint_success("✅ Lookback selection completed")
            except Exception as e:
                tprint_error(f"❌ Lookback selection failed: {e}")
                raise
            
            tprint_info(f"✅ Selected lookbacks for {len(lookback_choices)} feature families")
            
            # Log detailed lookback choices
            tprint_info("📊 Lookback optimization results:")
            for family, choice in lookback_choices.items():
                tprint_info(f"   - {family}: {choice.selected_lookback} periods")
                tprint_info(f"     - Confidence: {choice.confidence_score:.3f}")
                tprint_info(f"     - IC Score: {choice.ic_score:.3f}")
                tprint_info(f"     - AUC Score: {choice.auc_score:.3f}")
                tprint_info(f"     - Simplicity Bonus: {choice.simplicity_bonus:.3f}")
            
            # Apply Bayesian optimization if available
            if self.bayesian_optimizer and ML_COMMON_AVAILABLE:
                tprint_debug("🤖 Applying Bayesian TPE optimization...")
                try:
                    # Create optimization configuration
                    opt_config = OptimizationConfig(
                        n_trials=50,
                        timeout=300,  # 5 minutes
                        random_state=42
                    )
                    
                    # Optimize lookback choices using Bayesian TPE
                    optimized_choices = self.bayesian_optimizer.optimize_lookbacks(
                        parent_features,
                        targets.get(1, pd.Series(0, index=parent_features.index)),
                        feature_families,
                        lookback_choices,
                        opt_config
                    )
                    
                    if optimized_choices:
                        lookback_choices = optimized_choices
                        tprint_success("✅ Bayesian TPE optimization applied")
                        self.performance_metrics['optimization_applied'] = True
                    else:
                        tprint_warning("⚠️ Bayesian TPE optimization did not improve results")
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Bayesian TPE optimization failed: {e}")
            
            # Apply feature selection if available
            if self.feature_selector and ML_COMMON_AVAILABLE:
                tprint_debug("🎯 Applying feature selection...")
                try:
                    # Select best features based on lookback choices
                    selected_features = self.feature_selector.select_features(
                        parent_features,
                        targets.get(1, pd.Series(0, index=parent_features.index)),
                        method='mutual_info',
                        k_best=min(50, len(parent_features.columns))
                    )
                    
                    if selected_features:
                        tprint_success(f"✅ Feature selection completed: {len(selected_features)} features selected")
                        # Update parent features to only include selected features
                        parent_features = parent_features[selected_features]
                        tprint_info(f"📊 Reduced feature set to {len(selected_features)} features")
                    else:
                        tprint_warning("⚠️ Feature selection did not reduce feature set")
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Feature selection failed: {e}")
            
            # Check for lookahead bias if available
            if self.lookahead_bias_detector and ML_COMMON_AVAILABLE:
                tprint_debug("🔍 Checking for lookahead bias...")
                try:
                    bias_result = self.lookahead_bias_detector.detect_bias(
                        parent_features,
                        targets.get(1, pd.Series(0, index=parent_features.index))
                    )
                    
                    if bias_result.get('bias_detected', False):
                        tprint_warning(f"⚠️ Lookahead bias detected: {bias_result.get('details', 'Unknown')}")
                    else:
                        tprint_success("✅ No lookahead bias detected")
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Lookahead bias detection failed: {e}")
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_lookback_optimization_metrics(lookback_choices)
            
            # Update performance metrics
            self.performance_metrics.update({
                'lookback_optimization_success': True,
                'feature_families_optimized': len(lookback_choices),
                'average_confidence': optimization_metrics.get('average_confidence', 0.0),
                'average_ic_score': optimization_metrics.get('average_ic_score', 0.0)
            })
            
            tprint_info(f"📊 Lookback optimization metrics:")
            tprint_info(f"   - Feature families optimized: {len(lookback_choices)}")
            tprint_info(f"   - Average confidence: {optimization_metrics.get('average_confidence', 0.0):.3f}")
            tprint_info(f"   - Average IC score: {optimization_metrics.get('average_ic_score', 0.0):.3f}")
            tprint_info(f"   - Optimization success rate: {optimization_metrics.get('success_rate', 0.0):.1%}")
            
            stage_time = time.time() - stage_start
            tprint_performance("Lookback Optimization", stage_time)
            
            result = {
                'lookback_choices': lookback_choices,
                'feature_families': feature_families,
                'optimization_metrics': optimization_metrics,
                'parent_features': parent_features,  # Updated with selected features
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.LOOKBACK_OPTIMIZATION] = result
            tprint_success("✅ Stage 3: Lookback Optimization completed successfully")
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Lookback optimization failed: {e}")
            tprint_error(f"📊 Stage execution time: {stage_time:.3f}s")
            self.performance_metrics['lookback_optimization_success'] = False
            raise
    
    def _calculate_lookback_optimization_metrics(self, lookback_choices: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive lookback optimization metrics."""
        try:
            if not lookback_choices:
                return {
                    'average_confidence': 0.0,
                    'average_ic_score': 0.0,
                    'average_auc_score': 0.0,
                    'success_rate': 0.0,
                    'total_families': 0
                }
            
            confidences = []
            ic_scores = []
            auc_scores = []
            successful_optimizations = 0
            
            for family, choice in lookback_choices.items():
                if hasattr(choice, 'confidence_score'):
                    confidences.append(choice.confidence_score)
                if hasattr(choice, 'ic_score'):
                    ic_scores.append(choice.ic_score)
                if hasattr(choice, 'auc_score'):
                    auc_scores.append(choice.auc_score)
                
                # Consider optimization successful if confidence > 0.5
                if hasattr(choice, 'confidence_score') and choice.confidence_score > 0.5:
                    successful_optimizations += 1
            
            total_families = len(lookback_choices)
            success_rate = successful_optimizations / total_families if total_families > 0 else 0.0
            
            return {
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'average_ic_score': np.mean(ic_scores) if ic_scores else 0.0,
                'average_auc_score': np.mean(auc_scores) if auc_scores else 0.0,
                'success_rate': success_rate,
                'total_families': total_families,
                'successful_optimizations': successful_optimizations
            }

        except Exception as e:
            tprint_error(f"❌ Critical error: Lookback optimization metrics calculation failed: {e}")
            raise RuntimeError(f"Lookback optimization metrics calculation failed: {e}")
    
    async def _stage_transform_application(self, lookback_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Apply transforms to parent features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 4: Transform Application")
        
        try:
            # Get parent features from previous stage
            feature_result = self.stage_results[PipelineStage.FEATURE_ENGINEERING]
            parent_features = feature_result['parent_features']
            feature_names = feature_result['feature_names']
            
            # Create transform configuration
            tprint_debug("Creating transform configuration...")
            transform_config = create_default_transform_config(feature_names)
            tprint_debug(f"Created transform config for {len(transform_config)} features")
            
            # Initialize transform router
            transform_router = TransformRouter(transform_config)
            
            # Split data for transform fitting
            split_idx = int(len(parent_features) * 0.8)
            train_features = parent_features.iloc[:split_idx]
            val_features = parent_features.iloc[split_idx:]
            
            tprint_debug(f"Split data: train={len(train_features)}, val={len(val_features)}")
            
            # Apply transforms
            tprint_debug("Applying transforms...")
            transformed_results = transform_router.fit_transform(train_features, val_features)
            
            # Combine transformed features
            all_transformed = []
            for feature_name, results in transformed_results.items():
                all_transformed.append(results['train'])
            
            if all_transformed:
                transformed_features = pd.concat(all_transformed, axis=1)
                tprint_info(f"✅ Generated {len(transformed_features.columns)} transformed features")
            else:
                transformed_features = pd.DataFrame(index=parent_features.index)
                tprint_warning("⚠️ No transformed features generated")
            
            # Apply winsorization
            tprint_debug("Applying winsorization...")
            from .feature_engineering.transforms import apply_winsorization
            transformed_features = apply_winsorization(transformed_features)
            
            # Matrix optimization
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying matrix optimization to transformed features...")
                transformed_features = self.vectorized_core.optimize_dataframe_for_processing(transformed_features)
            
            stage_time = time.time() - stage_start
            tprint_performance("Transform Application", stage_time)
            
            result = {
                'transformed_features': transformed_features,
                'transform_router': transform_router,
                'transform_config': transform_config,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.TRANSFORM_APPLICATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Transform application failed: {e}")
            raise
    
    async def _stage_interaction_generation(self, transform_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Generate interaction features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 5: Interaction Generation")
        
        try:
            transformed_features = transform_result['transformed_features']

            cached_interactions = pipeline_state.get('_cached_interaction_features')
            if cached_interactions is not None and not self._force_cache_refresh:
                tprint_info("📦 Using cached interaction features")
                stage_time = time.time() - stage_start
                result = {
                    'interactions': cached_interactions,
                    'interaction_engine': None,
                    'interaction_config': None,
                    'stage_time': stage_time,
                    'success': True,
                }
                self.stage_results[PipelineStage.INTERACTION_GENERATION] = result
                return result

            # Create interaction configuration
            tprint_debug("Creating interaction configuration...")
            interaction_config = create_default_interaction_config()
            tprint_debug(f"Created interaction config for {len(interaction_config)} interactions")
            
            # Initialize interaction engine
            interaction_engine = InteractionEngine(interaction_config)
            
            # Extract patch features if available
            patch_features = pipeline_state.get('patch_features', {})
            if patch_features:
                tprint_debug(f"Using {len(patch_features)} patch features")
            else:
                tprint_debug("No patch features available")
            
            # Generate interactions
            tprint_debug("Generating interactions...")
            interactions = interaction_engine.build_interactions(transformed_features, patch_features)
            
            tprint_info(f"✅ Generated {len(interactions.columns)} interaction features")

            # Log interaction types
            interaction_types = {}
            for col in interactions.columns:
                interaction_type = col.split('/')[1] if '/' in col else 'unknown'
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1

            for interaction_type, count in interaction_types.items():
                tprint_debug(f"  {interaction_type}: {count} features")

            # Matrix optimization
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying matrix optimization to interactions...")
                interactions = self.vectorized_core.optimize_dataframe_for_processing(interactions)

            if self._active_cache_key:
                self.feature_cache.save(self._active_cache_key, interactions, artifact_type="interactions")

            stage_time = time.time() - stage_start
            tprint_performance("Interaction Generation", stage_time)

            result = {
                'interactions': interactions,
                'interaction_engine': interaction_engine,
                'interaction_config': interaction_config,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.INTERACTION_GENERATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Interaction generation failed: {e}")
            raise
    
    async def _stage_cross_timeframe_features(self, interaction_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 6: Generate cross-timeframe features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 6: Cross-timeframe Features")
        
        try:
            # Get transformed features and interactions
            transform_result = self.stage_results[PipelineStage.TRANSFORM_APPLICATION]
            transformed_features = transform_result['transformed_features']
            interactions = interaction_result['interactions']

            cached_cross = pipeline_state.get('_cached_cross_timeframe_features')
            if cached_cross is not None and not self._force_cache_refresh:
                tprint_info("📦 Using cached cross-timeframe features")
                all_features = pd.concat([transformed_features, interactions], axis=1)
                stage_time = time.time() - stage_start
                result = {
                    'cross_timeframe_features': cached_cross,
                    'all_features': all_features,
                    'stage_time': stage_time,
                    'success': True,
                }
                self.stage_results[PipelineStage.CROSS_TIMEFRAME] = result
                return result

            # Combine features
            all_features = pd.concat([transformed_features, interactions], axis=1)

            # Generate cross-timeframe features
            tprint_debug("Generating cross-timeframe features...")
            cross_timeframe_features = self._generate_cross_timeframe_features(all_features)

            tprint_info(f"✅ Generated {len(cross_timeframe_features.columns)} cross-timeframe features")

            # Matrix optimization
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying matrix optimization to cross-timeframe features...")
                cross_timeframe_features = self.vectorized_core.optimize_dataframe_for_processing(cross_timeframe_features)

            if self._active_cache_key:
                self.feature_cache.save(self._active_cache_key, cross_timeframe_features, artifact_type="cross_timeframe")
                self.cache_metrics['writes'] += 1
                self._sync_cache_metrics()

            stage_time = time.time() - stage_start
            tprint_performance("Cross-timeframe Features", stage_time)

            result = {
                'cross_timeframe_features': cross_timeframe_features,
                'all_features': all_features,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.CROSS_TIMEFRAME] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Cross-timeframe features failed: {e}")
            raise
    
    async def _stage_final_assembly(self, cross_timeframe_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 7: Final feature assembly and selection."""
        stage_start = time.time()
        tprint_info("🔧 Stage 7: Final Assembly")
        
        try:
            all_features = cross_timeframe_result['all_features']
            cross_timeframe_features = cross_timeframe_result['cross_timeframe_features']
            
            # Combine all features
            final_features = pd.concat([all_features, cross_timeframe_features], axis=1)
            tprint_info(f"✅ Assembled {len(final_features.columns)} total features")
            
            # Feature selection
            tprint_debug("Performing feature selection...")
            selected_features = self._select_features(final_features, pipeline_state)
            tprint_info(f"✅ Selected {len(selected_features)} features within budget")
            
            # Create final feature matrix
            final_feature_matrix = final_features[selected_features] if selected_features else final_features
            
            # Memory optimization
            if self.m1_memory_optimizer:
                final_feature_matrix = self.m1_memory_optimizer.optimize_dataframe_memory(final_feature_matrix)
            
            stage_time = time.time() - stage_start
            tprint_performance("Final Assembly", stage_time)
            
            result = {
                'final_features': final_feature_matrix,
                'selected_features': selected_features,
                'all_feature_names': final_features.columns.tolist(),
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.FINAL_ASSEMBLY] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Final assembly failed: {e}")
            raise
    
    async def _stage_validation(self, final_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 8: Validate generated features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 8: Validation")
        
        try:
            final_features = final_result['final_features']
            selected_features = final_result['selected_features']
            
            # Data quality validation
            tprint_debug("Performing data quality validation...")
            validation_results = self._validate_features(final_features)
            
            # Performance validation
            tprint_debug("Performing performance validation...")
            performance_results = self._validate_performance(final_features)
            
            # Memory validation
            memory_usage_mb = final_features.memory_usage(deep=True).sum() / 1024 / 1024
            tprint_info(f"✅ Memory usage: {memory_usage_mb:.2f} MB")
            
            stage_time = time.time() - stage_start
            tprint_performance("Validation", stage_time)
            
            result = {
                'validation_results': validation_results,
                'performance_results': performance_results,
                'memory_usage_mb': memory_usage_mb,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.VALIDATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Validation failed: {e}")
            raise
    
    async def _stage_completion(self, validation_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> OptimizedInteractionResult:
        """Stage 9: Complete pipeline and return results."""
        stage_start = time.time()
        tprint_info("🔧 Stage 9: Completion")
        
        try:
            # Get results from all stages
            final_result = self.stage_results[PipelineStage.FINAL_ASSEMBLY]
            interaction_result = self.stage_results[PipelineStage.INTERACTION_GENERATION]
            cross_timeframe_result = self.stage_results[PipelineStage.CROSS_TIMEFRAME]

            pipeline_state['feature_cache_metrics'] = dict(self.cache_metrics)
            pipeline_state['feature_cache_key'] = self._active_cache_key

            # Extract features
            final_features = final_result['final_features']
            selected_features = final_result['selected_features']
            all_feature_names = final_result['all_feature_names']
            interactions = interaction_result['interactions']
            cross_timeframe_features = cross_timeframe_result['cross_timeframe_features']
            
            # Calculate performance metrics
            total_execution_time = sum(result['stage_time'] for result in self.stage_results.values())
            memory_usage_mb = validation_result.get('memory_usage_mb', 0.0)
            
            # Create artifacts
            artifacts = {
                'stage_results': self.stage_results,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
                'feature_registry': self.feature_registry,
                'assembly_result': self.stage_results[PipelineStage.FEATURE_ENGINEERING].get('assembly_result'),
                'cache_metrics': dict(self.cache_metrics),
            }
            
            stage_time = time.time() - stage_start
            tprint_performance("Completion", stage_time)
            
            # Final success message
            tprint_success("🎉 Optimized interaction feature generation completed successfully!")
            tprint_info(f"📊 Generated {len(all_feature_names)} total features")
            tprint_info(f"🎯 Selected {len(selected_features)} features")
            tprint_info(f"🔗 Generated {len(interactions.columns)} interactions")
            tprint_info(f"⏰ Generated {len(cross_timeframe_features.columns)} cross-timeframe features")
            tprint_info(f"💾 Memory usage: {memory_usage_mb:.2f} MB")
            tprint_info(f"⏱️ Total execution time: {total_execution_time:.3f}s")

            self._sync_cache_metrics()

            return OptimizedInteractionResult(
                features=final_features,
                feature_names=all_feature_names,
                selected_features=selected_features,
                interaction_features=interactions,
                cross_timeframe_features=cross_timeframe_features,
                execution_time=total_execution_time,
                success=True,
                memory_usage_mb=memory_usage_mb,
                stage_results=self.stage_results,
                artifacts=artifacts
            )
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Completion failed: {e}")
            raise
    
    def _generate_cross_timeframe_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe features."""
        tprint_debug("Generating cross-timeframe features...")
        
        cross_timeframe_features = {}
        
        # Timeframe aggregations
        timeframes = [5, 15, 30, 60]  # minutes
        
        for tf in timeframes:
            # Rolling aggregations
            for col in features.columns:
                if col.startswith('t/'):  # Only transform features
                    # Rolling mean
                    cross_timeframe_features[f'ctf_{tf}m_{col}_mean'] = features[col].rolling(tf).mean()
                    
                    # Rolling std
                    cross_timeframe_features[f'ctf_{tf}m_{col}_std'] = features[col].rolling(tf).std()
                    
                    # Rolling max
                    cross_timeframe_features[f'ctf_{tf}m_{col}_max'] = features[col].rolling(tf).max()
                    
                    # Rolling min
                    cross_timeframe_features[f'ctf_{tf}m_{col}_min'] = features[col].rolling(tf).min()
        
        # Create DataFrame
        if cross_timeframe_features:
            result = pd.DataFrame(cross_timeframe_features, index=features.index)
            # Remove columns with all NaN values
            result = result.dropna(axis=1, how='all')
            return result
        else:
            return pd.DataFrame(index=features.index)
    
    def _select_features(self, features: pd.DataFrame, pipeline_state: Dict[str, Any]) -> List[str]:
        """Select features within budget constraints."""
        tprint_debug(f"Selecting features from {len(features.columns)} candidates...")
        
        if len(features.columns) <= self.config.feature_budget_pre:
            tprint_debug("All features within budget, selecting all")
            return features.columns.tolist()
        
        # Extract targets for selection
        targets = pipeline_state.get('targets', {})
        if not targets:
            tprint_warning("No targets available for feature selection, using random selection")
            return features.columns.tolist()[:self.config.feature_budget_pre]
        
        target_series = targets.get(1, pd.Series(0, index=features.index))
        
        # Calculate correlations
        correlations = []
        for col in features.columns:
            if not features[col].isna().all() and not target_series.isna().all():
                try:
                    corr = features[col].corr(target_series)
                    if not pd.isna(corr):
                        correlations.append((col, abs(corr)))
                except Exception as e:
                    tprint_debug(f"Failed to calculate correlation for {col}: {e}")
                    continue
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features within budget
        selected = [col for col, _ in correlations[:self.config.feature_budget_pre]]
        
        tprint_debug(f"Selected {len(selected)} features based on correlation")
        return selected
    
    def _validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate generated features."""
        validation_results = {
            'total_features': len(features.columns),
            'finite_features': 0,
            'infinite_features': 0,
            'nan_features': 0,
            'constant_features': 0,
            'quality_score': 0.0
        }
        
        for col in features.columns:
            col_data = features[col].dropna()
            
            if len(col_data) == 0:
                validation_results['nan_features'] += 1
                continue
            
            # Check for finite values
            finite_count = np.isfinite(col_data).sum()
            if finite_count == len(col_data):
                validation_results['finite_features'] += 1
            else:
                validation_results['infinite_features'] += 1
            
            # Check for constant features
            if col_data.nunique() <= 1:
                validation_results['constant_features'] += 1
        
        # Calculate quality score
        total_features = validation_results['total_features']
        if total_features > 0:
            validation_results['quality_score'] = validation_results['finite_features'] / total_features
        
        return validation_results
    
    def _validate_performance(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate performance characteristics."""
        performance_results = {
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024,
            'shape': features.shape,
            'dtypes': features.dtypes.value_counts().to_dict()
        }
        
        return performance_results
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if len(data) == 0:
            return 0.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Check for infinite values
        infinite_ratio = np.isinf(data.select_dtypes(include=[np.number])).sum().sum() / (len(data) * len(data.columns))
        
        # Calculate quality score (higher is better)
        quality_score = 1.0 - missing_ratio - infinite_ratio
        return max(0.0, quality_score)


# Convenience function for easy integration
async def generate_optimized_interaction_features(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
    config: Optional[OptimizedInteractionConfig] = None
) -> OptimizedInteractionResult:
    """
    Generate optimized interaction features with the given configuration.
    
    Args:
        training_input: Input data for feature generation
        pipeline_state: Current pipeline state
        config: Configuration for feature generation
        
    Returns:
        OptimizedInteractionResult with generated features
    """
    if config is None:
        config = OptimizedInteractionConfig()
    
    orchestrator = OptimizedInteractionOrchestrator(config)
    return await orchestrator.generate_features(training_input, pipeline_state)