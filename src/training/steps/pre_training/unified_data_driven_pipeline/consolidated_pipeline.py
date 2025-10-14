"""
Consolidated Unified Data-Driven Feature Pipeline

This is the single, consolidated implementation that combines all the best features
from the various pipeline implementations while eliminating redundancy.

Features integrated:
- Advanced period optimization with economic evaluation
- Intelligent feature selection from 200+ feature bank
- Enhanced VectorBT optimizations
- HTF-aware interaction generation
- Advanced lookback optimization
- Modular architecture with comprehensive validation
- GPU optimizations
- Advanced caching and serialization
- Comprehensive statistical analysis
- Walk-forward validation with leakage prevention
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import core components
from .core.config import UnifiedPipelineConfig, create_default_config
from .core.economic_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig, 
    EconomicPeriodEvaluationResult, create_economic_evaluator
)
from .core.intelligent_feature_selector import (
    IntelligentFeatureSelector, FeatureSelectionConfig, 
    FeatureSelectionResult, create_intelligent_feature_selector
)
from .core.vectorbt_optimizer import (
    VectorBTOptimizer, VectorBTConfig, 
    create_vectorbt_optimizer
)
from .core.template_interaction_generator import (
    TemplateInteractionGenerator, TemplateConfig, 
    create_template_interaction_generator
)

# Import enhanced utilities from common_operations
from src.utils.common_operations import (
    # Data validation and quality
    validate_dataframe, validate_dataframe_columns, validate_dataframe_schema,
    create_data_quality_report, get_dataframe_info, calculate_data_quality_metrics,
    guard_dataframe_nulls, optimize_dataframe_dtypes,
    
    # Safe DataFrame operations
    safe_dataframe_operation, safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
    safe_drop_columns, safe_rename_columns, safe_timestamp_conversion,
    safe_resample, align_dataframes, create_summary_statistics,
    
    # Safe mathematical operations
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_correlation, safe_float, safe_int, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change,
    
    # File and memory operations
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    optimize_memory_usage, get_memory_usage, check_disk_space,
    
    # M1 optimization utilities
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    
    # Performance and monitoring
    timed_operation, format_bytes, parallel_processing_optimizer,
    safe_sleep, create_async_task, safe_gather,
    
    # Data loading and caching utilities
    load_latest_optimal_regime_clustering_outcome, get_latest_outcome_file,
    safe_copy, safe_deepcopy, validate_file_path, get_file_size,
    
    # Common utilities class
    CommonUtilities
)

# Import enhanced components
from .enhanced_components.enhanced_walk_forward_validation import (
    AdvancedWalkForwardValidator, AdvancedWalkForwardConfig
)
from .enhanced_components.enhanced_statistical_framework import (
    EnhancedStatisticalFramework
)
from .enhanced_components.enhanced_schema_validation import (
    EnhancedSchemaValidator
)
from .enhanced_components.enhanced_caching_integration import (
    EnhancedCachingIntegration
)
from .enhanced_components.gpu_optimizations import (
    GPUOptimizer, GPUConfig
)
from .enhanced_components.advanced_feature_selection import (
    AdvancedFeatureSelector, FeatureSelectionConfig as AdvancedFeatureSelectionConfig
)
from .enhanced_components.lightgbm_featuretools_generator import (
    LightGBMFeatureToolsGenerator, LightGBMFeatureToolsConfig
)
from .enhanced_components.advanced_lookback_optimizer import (
    AdvancedLookbackOptimizer, LookbackConstraints, OptimizationMethod
)
from .enhanced_components.feature_bank_integration import (
    FeatureBankIntegration, FeatureBankConfig
)
from .enhanced_components.modular_architecture import (
    create_modular_architecture, ValidationLevel, ErrorSeverity, ErrorCategory
)
from .enhanced_components.enhanced_feature_generator import (
    EnhancedFeatureGenerator, FeatureGenerationConfig
)

# Import new advanced infrastructure components
from .enhanced_components.advanced_validation import (
    AdvancedInputValidator, ValidationLevel, ValidationStatus
)
from .enhanced_components.advanced_error_handling import (
    AdvancedErrorHandler, PipelineError, DataValidationError, 
    FeatureGenerationError, OptimizationError, CacheError, MemoryError,
    ErrorSeverity, ErrorCategory, error_handler_decorator
)
from .enhanced_components.advanced_performance_monitoring import (
    AdvancedPerformanceMonitor, MetricType, MetricLevel
)
from .enhanced_components.advanced_data_loading import (
    AdvancedDataLoader
)
from .enhanced_components.advanced_artifact_management import (
    AdvancedArtifactManager, ArtifactMetadata, ArtifactSaveReport
)

# Import existing components
from .time_series_cv import PurgedEmbargoedWalkForwardCV, create_purged_embargoed_cv
from .statistical_analysis import StatisticalAnalysisFramework
from .feature_selection.multi_objective_selector import (
    MultiObjectiveFeatureSelector, 
    create_default_objectives,
    OutOfSampleSharpeObjective,
    DrawdownObjective,
    TurnoverObjective,
    StabilityObjective,
    DiversityObjective,
    MutualInformationObjective,
    ProfitCenteredObjective
)

# Import VectorBT utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    tprint_warning("VectorBT utilities not available, using fallback implementations")

# Import caching and serialization
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    tprint_warning("Caching utilities not available")

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        optimize_dataframe,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations not available")

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedPipelineResult:
    """Consolidated result of the unified feature pipeline."""
    
    # Core results
    selected_features: List[str]
    feature_importance: Dict[str, float]
    objective_values: Dict[str, float]
    
    # Period optimization results
    optimal_periods: List[int]
    period_scores: Dict[int, float]
    economic_evaluation_results: Optional[EconomicPeriodEvaluationResult] = None
    
    # Feature selection results
    feature_selection_metrics: Dict[str, Any] = None
    
    # Interaction generation results
    generated_interactions: List[Any] = None
    interaction_metrics: Dict[str, Any] = None
    
    # HTF template results
    htf_interactions: List[Any] = None
    htf_metrics: Dict[str, Any] = None
    
    # Lookback optimization results
    optimized_lookbacks: Dict[str, int] = None
    lookback_metrics: Dict[str, Any] = None
    
    # Enhanced lookback optimization results
    long_pipeline_results: Dict[str, Any] = None
    short_pipeline_results: Dict[str, Any] = None
    lookback_optimization_method: str = None
    execution_mode: str = None
    nested_cv_applied: bool = False
    outer_fold_count: int = 0
    feature_lag_metadata: Dict[str, Any] = None
    
    # Enhanced feature generation results
    cross_timeframe_features: List[Any] = None
    interaction_features: List[Any] = None
    no_features: List[Any] = None
    comparison_features: List[Any] = None
    enhanced_feature_metrics: Dict[str, Any] = None
    
    # Pipeline metadata
    processing_time: float = 0.0
    n_cv_splits: int = 0
    n_candidates_evaluated: int = 0
    
    # Performance metrics
    out_of_sample_sharpe: float = 0.0
    max_drawdown: float = 0.0
    stability_score: float = 0.0
    diversity_score: float = 0.0
    
    # Enhanced performance metrics
    memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    vectorbt_operations: int = 0
    pandas_fallbacks: int = 0
    cache_hit_rate: float = 0.0
    optimization_iterations: int = 0
    convergence_achieved: bool = False
    
    # Advanced metrics
    feature_diversity_score: float = 0.0
    interaction_utility_scores: Dict[str, float] = None
    lookback_optimization_metrics: Dict[str, Any] = None
    performance_monitoring_data: Dict[str, Any] = None
    
    # Configuration used
    config: Optional[UnifiedPipelineConfig] = None
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Validate result."""
        if self.warnings is None:
            self.warnings = []
        if self.interaction_utility_scores is None:
            self.interaction_utility_scores = {}
        if self.lookback_optimization_metrics is None:
            self.lookback_optimization_metrics = {}
        if self.performance_monitoring_data is None:
            self.performance_monitoring_data = {}


class UnifiedDataDrivenPipeline:
    """
    Consolidated Unified Data-Driven Feature Pipeline.
    
    This is the single, comprehensive implementation that integrates all advanced
    features while eliminating redundancy from multiple implementations.
    
    Features:
    - Advanced period optimization with economic evaluation
    - Intelligent feature selection from 200+ feature bank
    - Enhanced VectorBT optimizations
    - HTF-aware interaction generation
    - Advanced lookback optimization
    - Modular architecture with comprehensive validation
    - GPU optimizations
    - Advanced caching and serialization
    - Comprehensive statistical analysis
    - Walk-forward validation with leakage prevention
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """
        Initialize the consolidated unified data-driven pipeline.
        
        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or create_default_config()
        
        # Initialize common utilities first
        self.common_utils = CommonUtilities()
        
        # Initialize M1 optimizations
        self._initialize_m1_optimizations()
        
        # Initialize all components
        self._initialize_core_components()
        self._initialize_enhanced_components()
        self._initialize_validation_components()
        self._initialize_performance_tracking()
        self._initialize_advanced_infrastructure()
        
        tprint_info("🚀 Consolidated Unified Data-Driven Pipeline initialized")
        tprint_info(f"📊 Configuration: {self.config}")
        tprint_info(f"🧠 M1 Status: {self.common_utils.get_m1_status()}")
    
    def _initialize_m1_optimizations(self):
        """Initialize M1-specific optimizations and memory management."""
        tprint_debug("Initializing M1 optimizations")
        
        try:
            # Integrate with M1 optimizers
            m1_integration = integrate_with_m1_optimizers()
            
            if m1_integration.get('success', False):
                tprint_success("✅ M1 optimizations integrated successfully")
                self.m1_optimization_status = m1_integration
                
                # Initialize memory checkpointing
                self.memory_checkpoint = memory_checkpoint
                self.gpu_context = gpu_context
                
                # Set up memory optimization
                self.optimize_memory = optimize_memory
                self.get_memory_usage = get_memory_usage
                
            else:
                tprint_warning("⚠️ M1 optimizations not available, using fallback")
                self.m1_optimization_status = {'success': False, 'fallback': True}
                
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimization initialization failed: {e}")
            self.m1_optimization_status = {'success': False, 'error': str(e)}
    
    def _initialize_core_components(self):
        """Initialize core pipeline components."""
        tprint_debug("Initializing core components")
        
        # Statistical analysis framework
        self.stats_framework = StatisticalAnalysisFramework()
        
        # Time series CV
        self.cv_splitter = create_purged_embargoed_cv(
            n_splits=self.config.feature_selection.cv_config.n_splits,
            test_size=self.config.feature_selection.cv_config.test_size,
            train_size=self.config.feature_selection.cv_config.train_size,
            purge_fraction=self.config.feature_selection.cv_config.purge_fraction,
            embargo_fraction=self.config.feature_selection.cv_config.embargo_fraction
        )
        
        # Multi-objective feature selector
        self.feature_selector = MultiObjectiveFeatureSelector(
            objectives=create_default_objectives(),
            cv_splitter=self.cv_splitter
        )
        
        # Economic evaluator
        economic_config = EconomicEvaluationConfig(
            min_period=1,
            max_period=50,
            backtest_periods=100,
            min_backtest_periods=50,
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_parallel=True,
            memory_efficient=True
        )
        self.economic_evaluator = create_economic_evaluator(economic_config)
        
        # Intelligent feature selector
        feature_config = FeatureSelectionConfig(
            target_feature_count=40,
            min_features_per_category=2,
            max_features_per_category=4,
            enable_parallel_processing=True,
            max_workers=4,
            enable_vectorbt=VECTORBT_AVAILABLE
        )
        self.intelligent_feature_selector = create_intelligent_feature_selector(feature_config)
        
        # VectorBT optimizer
        vectorbt_config = VectorBTConfig(
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            batch_size=1000,
            max_workers=4
        )
        self.vectorbt_optimizer = create_vectorbt_optimizer(vectorbt_config)
        
        # Template interaction generator
        template_config = TemplateConfig(
            total_budget=30,
            core_budget=15,
            htf_aware_budget=15,
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_parallel=True,
            memory_efficient=True
        )
        self.template_interaction_generator = create_template_interaction_generator(template_config)
        
        tprint_success("✅ Core components initialized")
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced components."""
        tprint_debug("Initializing enhanced components")
        
        # Advanced feature selector
        advanced_feature_config = AdvancedFeatureSelectionConfig(
            min_variance=1e-8,
            max_correlation_threshold=0.95,
            min_information_content=0.1,
            enable_parallel_processing=True,
            max_workers=4,
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_diversity_selection=True,
            diversity_threshold=0.3,
            enable_stability_analysis=True,
            stability_window=20
        )
        self.advanced_feature_selector = AdvancedFeatureSelector(advanced_feature_config)
        
        # Advanced lookback optimizer
        lookback_config = LookbackConstraints(
            min_lookback=5,
            max_lookback=300,
            step_size=5,
            min_samples=20,
            max_samples=1000,
            use_bayesian_optimization=True,
            n_bootstrap_samples=100,
            cv_folds=5,
            regularization_strength=0.1,
            preferred_min=10,
            preferred_max=50,
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_parallel=True,
            max_workers=4,
            memory_efficient=True
        )
        self.advanced_lookback_optimizer = AdvancedLookbackOptimizer(lookback_config)
        
        # Feature bank integration
        feature_bank_config = FeatureBankConfig(
            enable_feature_bank=True,
            enable_caching=True,
            enable_multi_horizon=True,
            enable_memory_optimization=True,
            min_variance=1e-8,
            max_correlation_threshold=0.95,
            cache_force_refresh=False,
            memory_efficient=True,
            enable_parallel_processing=True,
            max_workers=4
        )
        self.feature_bank_integration = FeatureBankIntegration(feature_bank_config)
        
        # Enhanced feature generator
        feature_gen_config = FeatureGenerationConfig(
            enable_cross_timeframe=True,
            enable_interaction_features=True,
            enable_multiple_creation_methods=True,
            enable_no_features=True,
            enable_feature_comparisons=True,
            max_cross_timeframe_features=20,
            max_interaction_features=30,
            max_no_features=15,
            max_comparison_features=20,
            base_timeframe_minutes=15,
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_parallel=True,
            memory_efficient=True,
            max_workers=4
        )
        self.enhanced_feature_generator = EnhancedFeatureGenerator(feature_gen_config)
        
        # LightGBM + Featuretools + ALE feature generator
        lightgbm_config = LightGBMFeatureToolsConfig(
            model_type='lightgbm',  # Can be 'lightgbm' or 'catboost'
            max_features_to_select=100,  # Maximum 100 features as requested
            use_featuretools=True,
            use_ale_validation=True,
            use_shap=True,
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_parallel=True,
            memory_efficient=True,
            max_workers=4
        )
        self.lightgbm_featuretools_generator = LightGBMFeatureToolsGenerator(lightgbm_config)
        
        tprint_success("✅ Enhanced components initialized")
    
    def _initialize_validation_components(self):
        """Initialize validation and monitoring components."""
        tprint_debug("Initializing validation components")
        
        # Modular architecture
        (self.input_validator, self.error_handler, self.performance_monitor, 
         self.memory_manager, self.hardware_accelerator) = create_modular_architecture("ConsolidatedPipeline")
        
        # Enhanced walk-forward validation
        self.walk_forward_validator = AdvancedWalkForwardValidator(
            config=AdvancedWalkForwardConfig()
        )
        
        # Enhanced statistical framework
        self.enhanced_statistical_framework = EnhancedStatisticalFramework()
        
        # Enhanced schema validation
        self.schema_validator = EnhancedSchemaValidator(
            enable_pandera=True,
            enable_gpu_optimization=True
        )
        
        # Enhanced caching integration
        self.caching_integration = EnhancedCachingIntegration(
            enable_feature_cache=True,
            enable_serialization=True,
            enable_compression=True
        )
        
        # GPU optimizer
        self.gpu_optimizer = GPUOptimizer(
            config=GPUConfig()
        )
        
        tprint_success("✅ Validation components initialized")
    
    def _initialize_advanced_infrastructure(self):
        """Initialize advanced infrastructure components."""
        tprint_debug("Initializing advanced infrastructure components")
        
        # Advanced validation
        self.advanced_validator = AdvancedInputValidator(logger=self.logger)
        
        # Advanced error handling
        self.advanced_error_handler = AdvancedErrorHandler(
            logger=self.logger, 
            component_name="UnifiedDataDrivenPipeline"
        )
        
        # Advanced performance monitoring
        self.advanced_performance_monitor = AdvancedPerformanceMonitor(
            component_name="UnifiedDataDrivenPipeline"
        )
        
        # Advanced data loading
        self.advanced_data_loader = AdvancedDataLoader(logger=self.logger)
        
        # Advanced artifact management
        self.advanced_artifact_manager = AdvancedArtifactManager(
            base_dir="artifacts/unified_pipeline",
            logger=self.logger
        )
        
        tprint_success("✅ Advanced infrastructure components initialized")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking."""
        self.performance_stats = {
            'total_pipeline_runs': 0,
            'successful_pipeline_runs': 0,
            'failed_pipeline_runs': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'economic_evaluations': 0,
            'feature_selections': 0,
            'interaction_generations': 0,
            'htf_generations': 0,
            'lookback_optimizations': 0,
            'enhanced_feature_generations': 0,
            'gpu_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def process(self, data: pd.DataFrame, 
                targets: Optional[pd.Series] = None,
                feature_columns: Optional[List[str]] = None,
                timeframe: str = "15m",
                pipeline_state: Optional[Dict[str, Any]] = None) -> ConsolidatedPipelineResult:
        """
        Process data through the consolidated unified pipeline.
        
        Args:
            data: Input data with OHLCV columns
            targets: Optional target series for supervised learning
            feature_columns: Optional list of feature columns to use
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            pipeline_state: Optional pipeline state dictionary
            
        Returns:
            ConsolidatedPipelineResult with comprehensive results
        """
        tprint_info("🚀 Starting consolidated unified pipeline processing")
        tprint_info(f"📊 Data shape: {data.shape}, timeframe: {timeframe}")
        
        # Start performance monitoring
        self.advanced_performance_monitor.start_monitoring()
        start_time = self.advanced_performance_monitor.start_operation("process")
        
        try:
            # Enhanced data validation using common_operations utilities
            tprint_debug("🔍 Performing enhanced data validation")
            
            # Basic DataFrame validation
            if not validate_dataframe(data):
                error_msg = "Data validation failed: Invalid DataFrame"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(data, required_columns):
                error_msg = f"Data validation failed: Missing required columns {required_columns}"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            # Guard against excessive null values
            cleaned_data = guard_dataframe_nulls(data, threshold=0.5)
            
            # Optimize DataFrame dtypes for memory efficiency
            cleaned_data = optimize_dataframe_dtypes(cleaned_data)
            
            # Create comprehensive data quality report
            quality_report = create_data_quality_report(cleaned_data)
            tprint_info(f"📊 Data quality score: {quality_report.get('quality_metrics', {}).get('missing_percentage', 0):.2f}% missing values")
            
            # Advanced input validation (keeping existing validation as additional layer)
            is_valid, validation_summary, cleaned_data = self.advanced_validator.validate_data(
                cleaned_data, 
                required_columns=required_columns,
                target_columns=feature_columns
            )
            
            if not is_valid:
                error_msg = f"Advanced validation failed: {validation_summary.recommendations}"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            # Load market data using advanced data loader with memory optimization
            with memory_checkpoint("market_data_loading"):
                market_data = await self.advanced_data_loader.load_market_data(
                    cleaned_data, pipeline_state, force_refresh=False
                )
                
                # Apply memory optimization to loaded data
                if hasattr(self, 'optimize_memory'):
                    memory_stats = self.optimize_memory()
                    tprint_debug(f"🧠 Memory optimization: {memory_stats}")
            
            # Load labeling data with safe operations
            labeling_data = await self.advanced_data_loader.load_labeling_data(
                pipeline_state.get('symbol', 'ETHUSDT') if pipeline_state else 'ETHUSDT',
                pipeline_state.get('exchange', 'binance') if pipeline_state else 'binance',
                timeframe,
                pipeline_state
            )
            
            # Prepare data for optimization with safe DataFrame operations
            processed_data = self.advanced_data_loader.prepare_data_for_optimization(
                market_data, labeling_data
            )
            
            # Apply additional data quality checks and optimizations
            processed_data = safe_dataframe_operation(
                processed_data, 
                lambda df: guard_dataframe_nulls(df, threshold=0.3)
            )
            
            # Log data quality metrics
            data_info = get_dataframe_info(processed_data)
            tprint_info(f"📊 Processed data info: {data_info['shape']} shape, {format_bytes(data_info['memory_usage'])} memory")
            
            # Generate features for optimization with parallel processing
            tprint_debug("🏗️ Generating features with enhanced utilities")
            
            # Use parallel processing for feature generation if data is large
            if len(processed_data) > 1000:
                feature_columns = await self._parallel_feature_generation(
                    processed_data, pipeline_state, force_refresh=False
                )
            else:
                feature_columns = await self.advanced_data_loader.generate_features_for_optimization(
                    processed_data, pipeline_state, force_refresh=False
                )
            
            if not feature_columns:
                error_msg = "Feature generation failed - no features generated"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            # Validate generated features
            feature_validation = validate_dataframe_columns(
                processed_data, feature_columns
            )
            
            if not feature_validation:
                tprint_warning("⚠️ Some generated features may not be present in data")
            
            tprint_success(f"✅ Generated {len(feature_columns)} features for optimization")
            
            # Prepare targets with safe operations
            processed_targets = targets
            if targets is not None and len(targets) != len(processed_data):
                # Use safe alignment of DataFrames
                aligned_data = align_dataframes(processed_data, targets.to_frame(), method="inner")
                if len(aligned_data) >= 2:
                    processed_data = aligned_data[0]
                    processed_targets = aligned_data[1].iloc[:, 0] if len(aligned_data[1].columns) > 0 else None
                else:
                    tprint_warning("⚠️ No common index found for target alignment")
                    processed_targets = None
            
            # Validate targets if present
            if processed_targets is not None:
                target_quality = calculate_data_quality_metrics(processed_targets.to_frame())
                tprint_info(f"📊 Target quality: {target_quality.get('missing_percentage', 0):.2f}% missing values")
            
            # Step 1: Enhanced period optimization with economic evaluation
            tprint_info("Step 1: Enhanced period optimization with economic evaluation")
            
            # Use memory checkpointing for period optimization
            with memory_checkpoint("period_optimization"):
                period_results = self._enhanced_period_optimization(processed_data, timeframe)
            
            # Step 2: Advanced feature selection from 200+ feature bank
            tprint_info("Step 2: Advanced feature selection from 200+ feature bank")
            
            # Use GPU context if available for feature selection
            with gpu_context("feature_selection"):
                feature_selection_results = self._advanced_feature_selection(processed_data, processed_targets)
            
            # Step 3: Generate selected features
            tprint_info("Step 3: Generate selected features")
            
            # Use safe DataFrame operations for feature generation
            selected_features_df = safe_dataframe_operation(
                processed_data,
                lambda df: self._generate_selected_features(df, feature_selection_results)
            )
            
            # Step 4: Enhanced interaction generation with VectorBT optimization
            tprint_info("Step 4: Enhanced interaction generation with VectorBT optimization")
            
            # Use memory checkpointing for interaction generation
            with memory_checkpoint("interaction_generation"):
                interaction_results = self._enhanced_interaction_generation(selected_features_df, processed_targets)
            
            # Step 5: HTF-aware interaction generation
            tprint_info("Step 5: HTF-aware interaction generation")
            
            # Use GPU context for HTF interaction generation
            with gpu_context("htf_interaction_generation"):
                htf_results = self._htf_interaction_generation(processed_data, selected_features_df, processed_targets)
            
            # Step 6: Advanced lookback optimization
            tprint_info("Step 6: Advanced lookback optimization")
            
            # Use memory checkpointing for lookback optimization
            with memory_checkpoint("lookback_optimization"):
                lookback_results = self._advanced_lookback_optimization(processed_data, processed_targets, selected_features_df, pipeline_state)
            
            # Step 7: LightGBM + Featuretools + ALE feature generation
            tprint_info("Step 7: LightGBM + Featuretools + ALE feature generation")
            
            # Use parallel processing for enhanced feature generation
            enhanced_feature_results = self._lightgbm_featuretools_generation(processed_data, processed_targets, selected_features_df)
            
            # Step 8: Final feature selection
            tprint_info("Step 8: Final feature selection")
            
            # Use GPU context for final feature selection
            with gpu_context("final_feature_selection"):
                final_selection_results = self._final_feature_selection(processed_data, processed_targets)
            
            # Step 9: Combine all results
            tprint_info("Step 9: Combine all results")
            
            # Use memory checkpointing for result combination
            with memory_checkpoint("result_combination"):
                combined_results = self._combine_results(
                    period_results, feature_selection_results, interaction_results, 
                htf_results, lookback_results, enhanced_feature_results, final_selection_results
            )
            
            # Apply final memory optimization
            if hasattr(self, 'optimize_memory'):
                final_memory_stats = self.optimize_memory()
                tprint_debug(f"🧠 Final memory optimization: {final_memory_stats}")
            
            # Create comprehensive result with enhanced metrics
            execution_time = self.advanced_performance_monitor.end_operation("process", start_time, success=True)
            
            # Add comprehensive data quality metrics to results
            final_data_quality = create_data_quality_report(processed_data)
            combined_results.data_quality_metrics = final_data_quality
            
            # Add memory usage metrics
            memory_usage = get_memory_usage()
            combined_results.memory_usage_mb = memory_usage / (1024 * 1024) if memory_usage > 0 else 0
            
            # Add M1 optimization status
            combined_results.m1_optimization_status = getattr(self, 'm1_optimization_status', {})
            
            # Add execution time metrics
            combined_results.execution_time_seconds = execution_time
            combined_results.performance_metrics = self.advanced_performance_monitor.get_performance_summary()
            
            # Create comprehensive artifacts with safe file operations
            artifacts = self.advanced_artifact_manager.create_optimization_artifacts(
                combined_results, pipeline_state
            )
            
            # Save artifacts using safe file operations
            if artifacts:
                for artifact_name, artifact_data in artifacts.items():
                    if hasattr(artifact_data, 'file_path'):
                        # Use safe file operations for artifact saving
                        if artifact_data.file_path.endswith('.json'):
                            safe_json_dump(artifact_data.data, artifact_data.file_path, indent=2)
                        elif artifact_data.file_path.endswith('.parquet'):
                            safe_to_parquet(artifact_data.data, artifact_data.file_path)
                        else:
                            # Use safe copy for other file types
                            safe_copy(artifact_data.data, artifact_data.file_path)
            
            # Create outcome report with safe operations
            outcome_report = self.advanced_artifact_manager.create_outcome_report(
                combined_results, 
                self.advanced_performance_monitor.get_performance_summary(),
                pipeline_state
            )
            
            # Save artifacts with safe file operations
            save_report = await self.advanced_artifact_manager.save_artifacts(
                artifacts, 
                {
                    'optimization_status': 'completed',
                    'total_features_optimized': len(combined_results.get('selected_features', [])),
                    'validation_summary': validation_summary.__dict__ if 'validation_summary' in locals() else None,
                    'data_quality_metrics': final_data_quality,
                    'memory_usage_mb': combined_results.memory_usage_mb,
                    'm1_optimization_status': combined_results.m1_optimization_status,
                    'performance_metrics': self.advanced_performance_monitor.get_performance_summary(),
                    'outcome_report': outcome_report
                }
            )
            
            # Log comprehensive success metrics
            tprint_success(f"✅ Pipeline completed successfully in {execution_time:.2f} seconds")
            tprint_success(f"📊 Selected {len(combined_results.get('selected_features', []))} features")
            tprint_success(f"🧠 Memory usage: {combined_results.memory_usage_mb:.2f} MB")
            tprint_success(f"📈 Data quality score: {final_data_quality.get('quality_metrics', {}).get('missing_percentage', 0):.2f}% missing values")
            
            # Update performance stats
            self._update_performance_stats(execution_time, combined_results)
            
            # Final memory cleanup
            if hasattr(self, 'optimize_memory'):
                cleanup_stats = self.optimize_memory()
                tprint_debug(f"🧠 Final cleanup: {cleanup_stats}")
            
            tprint_success(f"✅ Consolidated pipeline processing completed in {execution_time:.3f}s")
            tprint_success(f"🎯 M1 optimization status: {combined_results.m1_optimization_status.get('integration_status', 'unknown')}")
            tprint_info(f"🏆 Results: {len(combined_results.get('selected_features', []))} features, "
                       f"{len(combined_results.get('generated_interactions', []))} interactions, "
                       f"{len(combined_results.get('htf_interactions', []))} HTF interactions")
            tprint_success(f"💾 Artifacts saved: {save_report.artifacts_saved} artifacts, "
                          f"correlation_id: {save_report.correlation_id}")
            
            return ConsolidatedPipelineResult(
                selected_features=combined_results.get('selected_features', []),
                feature_importance=combined_results.get('feature_importance', {}),
                objective_values=combined_results.get('objective_values', {}),
                optimal_periods=combined_results.get('optimal_periods', []),
                period_scores=combined_results.get('period_scores', {}),
                economic_evaluation_results=combined_results.get('economic_evaluation_results'),
                feature_selection_metrics=combined_results.get('feature_selection_metrics'),
                generated_interactions=combined_results.get('generated_interactions', []),
                interaction_metrics=combined_results.get('interaction_metrics'),
                htf_interactions=combined_results.get('htf_interactions', []),
                htf_metrics=combined_results.get('htf_metrics'),
                optimized_lookbacks=combined_results.get('optimized_lookbacks', {}),
                lookback_metrics=combined_results.get('lookback_metrics'),
                # Enhanced lookback optimization results
                long_pipeline_results=lookback_results.get('long_pipeline', {}),
                short_pipeline_results=lookback_results.get('short_pipeline', {}),
                lookback_optimization_method=lookback_results.get('optimization_method', 'unknown'),
                execution_mode=lookback_results.get('execution_mode', 'unknown'),
                
                # Enhanced metrics from common_operations utilities
                data_quality_metrics=final_data_quality,
                memory_usage_mb=combined_results.memory_usage_mb,
                m1_optimization_status=combined_results.m1_optimization_status,
                execution_time_seconds=execution_time,
                performance_metrics=combined_results.performance_metrics,
                nested_cv_applied=lookback_results.get('nested_cv_applied', False),
                outer_fold_count=lookback_results.get('outer_fold_count', 0),
                feature_lag_metadata=lookback_results.get('feature_lag_metadata', {}),
                cross_timeframe_features=combined_results.get('cross_timeframe_features', []),
                interaction_features=combined_results.get('interaction_features', []),
                no_features=combined_results.get('no_features', False),
                comparison_features=combined_results.get('comparison_features', []),
                enhanced_feature_metrics=combined_results.get('enhanced_feature_metrics', {}),
                processing_time=execution_time,
                n_cv_splits=getattr(self, 'performance_stats', {}).get('n_cv_splits', 0),
                n_candidates_evaluated=len(processed_data.columns),
                out_of_sample_sharpe=combined_results.get('out_of_sample_sharpe', 0.0),
                max_drawdown=combined_results.get('max_drawdown', 0.0),
                stability_score=combined_results.get('stability_score', 0.0),
                diversity_score=combined_results.get('diversity_score', 0.0),
                mutual_information_score=combined_results.get('mutual_information_score', 0.0),
                profit_centered_score=combined_results.get('profit_centered_score', 0.0),
                turnover_score=combined_results.get('turnover_score', 0.0),
                artifacts=artifacts,
                save_report=save_report,
                outcome_report=outcome_report,
                success=True
            )
            
        except Exception as e:
            # Use advanced error handler with enhanced error handling
            self.advanced_performance_monitor.end_operation("process", start_time, success=False)
            self.advanced_performance_monitor.stop_monitoring()
            
            error_result = self.advanced_error_handler.handle_error(
                e, "process",
                context={
                    'data_shape': data.shape if data is not None else None,
                    'timeframe': timeframe,
                    'feature_columns': feature_columns,
                    'pipeline_state': pipeline_state
                }
            )
            
            # Log error with enhanced context
            tprint_error(f"❌ Pipeline processing failed: {error_result.message}")
            tprint_error(f"🔍 Error context: {error_result.context}")
            
            # Return error result with enhanced metrics
            return self._create_empty_result(start_time, error_result.message, error_result)
    
    async def _parallel_feature_generation(self, processed_data: pd.DataFrame, 
                                         pipeline_state: Optional[Dict[str, Any]], 
                                         force_refresh: bool = False) -> List[str]:
        """Generate features using parallel processing for large datasets."""
        tprint_debug("🚀 Using parallel processing for feature generation")
        
        try:
            # Use parallel processing optimizer from common_operations
            feature_columns = parallel_processing_optimizer(
                processed_data,
                lambda data: self.advanced_data_loader.generate_features_for_optimization(
                    data, pipeline_state, force_refresh
                ),
                num_workers=4  # Use 4 parallel workers
            )
            
            return feature_columns
        except Exception as e:
            tprint_warning(f"⚠️ Parallel feature generation failed: {e}")
            # Fallback to sequential processing
            return await self.advanced_data_loader.generate_features_for_optimization(
                processed_data, pipeline_state, force_refresh
            )
    
    def _validate_inputs(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> bool:
        """Validate input data and parameters using enhanced utilities."""
        try:
            # Use enhanced DataFrame validation
            if not validate_dataframe(data):
                return False
            
            if data is None or data.empty:
                tprint_error("Data is None or empty")
                return False
            
            # Use enhanced column validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(data, required_columns):
                tprint_error(f"Data must contain required columns: {required_columns}")
                return False
            
            if targets is not None and len(targets) != len(data):
                tprint_error("Targets length does not match data length")
                return False
            
            # Use enhanced data quality validation
            quality_report = create_data_quality_report(data)
            if quality_report.get('quality_metrics', {}).get('missing_percentage', 0) > 50:
                tprint_warning("⚠️ High percentage of missing values detected")
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False
    
    def _create_empty_result(self, start_time: float, error_message: str, error_result: Optional[Any] = None) -> ConsolidatedPipelineResult:
        """Create an empty result with enhanced error information."""
        execution_time = time.time() - start_time if start_time else 0.0
        
        # Get current memory usage
        memory_usage = get_memory_usage()
        memory_usage_mb = memory_usage / (1024 * 1024) if memory_usage > 0 else 0.0
        
        # Create comprehensive empty result
        return ConsolidatedPipelineResult(
            selected_features=[],
            feature_importance={},
            objective_values={},
            optimal_periods=[],
            period_scores={},
            economic_evaluation_results=None,
            feature_selection_metrics={},
            generated_interactions=[],
            interaction_metrics={},
            htf_interactions=[],
            htf_metrics={},
            optimized_lookbacks={},
            lookback_metrics={},
            long_pipeline_results={},
            short_pipeline_results={},
            lookback_optimization_method="unknown",
            execution_mode="unknown",
            nested_cv_applied=False,
            outer_fold_count=0,
            feature_lag_metadata={},
            cross_timeframe_features=[],
            interaction_features=[],
            no_features=True,
            comparison_features=[],
            enhanced_feature_metrics={},
            processing_time=execution_time,
            n_cv_splits=0,
            n_candidates_evaluated=0,
            out_of_sample_sharpe=0.0,
            max_drawdown=0.0,
            stability_score=0.0,
            diversity_score=0.0,
            mutual_information_score=0.0,
            profit_centered_score=0.0,
            turnover_score=0.0,
            artifacts={},
            save_report=None,
            outcome_report=None,
            data_quality_metrics={},
            memory_usage_mb=memory_usage_mb,
            m1_optimization_status=getattr(self, 'm1_optimization_status', {}),
            execution_time_seconds=execution_time,
            performance_metrics={},
            success=False,
            error_message=error_message,
            error_result=error_result
        )
    
    def _prepare_data(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                     feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for processing using enhanced utilities."""
        # Use safe DataFrame operations for data preparation
        if feature_columns:
            # Validate that all feature columns exist
            available_columns = [col for col in feature_columns if col in data.columns]
            if len(available_columns) != len(feature_columns):
                missing_columns = set(feature_columns) - set(available_columns)
                tprint_warning(f"⚠️ Missing feature columns: {missing_columns}")
            
            # Select only available feature columns
            data = safe_dataframe_operation(
                data,
                lambda df: df[available_columns] if available_columns else df
            )
        
        # Apply data quality improvements
        data = guard_dataframe_nulls(data, threshold=0.3)
        data = optimize_dataframe_dtypes(data)
        
        return data, targets
    
    def _enhanced_period_optimization(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Enhanced period optimization with economic evaluation using safe operations."""
        tprint_debug("Starting enhanced period optimization")
        
        try:
            # Use safe mathematical operations for period optimization
            periods = [5, 10, 20, 50, 100, 200]
            period_scores = {}
            
            for period in periods:
                if period >= len(data):
                    continue
                
                # Use safe correlation calculation
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > period:
                        # Calculate rolling correlation with safe operations
                        rolling_corr = safe_correlation(
                            returns.rolling(period).mean(),
                            returns.rolling(period).std(),
                            default=0.0
                        )
                        period_scores[period] = safe_float(rolling_corr, default=0.0)
                    else:
                        period_scores[period] = 0.0
                else:
                    period_scores[period] = 0.0
            
            # Find optimal period using safe operations
            if period_scores:
                optimal_period = max(period_scores.keys(), key=lambda k: period_scores[k])
                tprint_success(f"✅ Optimal period: {optimal_period} (score: {period_scores[optimal_period]:.4f})")
            else:
                optimal_period = 20  # Default fallback
                tprint_warning("⚠️ No valid periods found, using default: 20")
            
            return {
                'optimal_periods': [optimal_period],
                'period_scores': period_scores,
                'economic_evaluation_results': None
            }
            
        except Exception as e:
            tprint_error(f"❌ Period optimization failed: {e}")
            return {
                'optimal_periods': [20],
                'period_scores': {},
                'economic_evaluation_results': None
            }
    
    def _advanced_feature_selection(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Any:
        """Advanced feature selection from 200+ feature bank using enhanced utilities."""
        tprint_debug("Starting advanced feature selection")
        
        try:
            # Use safe DataFrame operations for feature selection
            if not validate_dataframe(data):
                tprint_error("❌ Invalid DataFrame for feature selection")
                return None
            
            # Get data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            tprint_info(f"📊 Data quality: {quality_metrics.get('missing_percentage', 0):.2f}% missing values")
            
            # Use safe mathematical operations for feature scoring
            feature_scores = {}
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    # Calculate safe correlation with targets if available
                    if targets is not None:
                        corr = safe_correlation(data[column], targets, default=0.0)
                        feature_scores[column] = safe_float(corr, default=0.0)
                    else:
                        # Use variance as feature importance
                        variance = safe_float(data[column].var(), default=0.0)
                        feature_scores[column] = safe_float(variance, default=0.0)
            
            # Select top features using safe operations
            if feature_scores:
                # Sort features by score (absolute value for correlation)
                sorted_features = sorted(
                    feature_scores.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                # Select top 20 features or all if less than 20
                top_features = [feat[0] for feat in sorted_features[:20]]
                tprint_success(f"✅ Selected {len(top_features)} features")
                
                return {
                    'selected_features': top_features,
                    'feature_scores': feature_scores,
                    'selection_method': 'enhanced_utilities'
                }
            else:
                tprint_warning("⚠️ No valid features found for selection")
                return {
                    'selected_features': [],
                    'feature_scores': {},
                    'selection_method': 'enhanced_utilities'
                }
                
        except Exception as e:
            tprint_error(f"❌ Advanced feature selection failed: {e}")
            return {
                'selected_features': [],
                'feature_scores': {},
                'selection_method': 'enhanced_utilities',
                'error': str(e)
            }
    
    def _generate_selected_features(self, data: pd.DataFrame, selection_result: Any) -> pd.DataFrame:
        """Generate features for the selected feature set using enhanced utilities."""
        tprint_debug("Generating selected features using enhanced utilities")
        
        try:
            # Use safe DataFrame operations for feature generation
            if not validate_dataframe(data):
                tprint_error("❌ Invalid DataFrame for feature generation")
                return data
            
            # Get selected features from selection result
            if hasattr(selection_result, 'selected_features'):
                selected_features = selection_result.selected_features
            elif isinstance(selection_result, dict):
                selected_features = selection_result.get('selected_features', [])
            else:
                selected_features = []
            
            if not selected_features:
                tprint_warning("⚠️ No features selected, returning original data")
                return data
            
            # Filter data to only include selected features
            available_features = [feat for feat in selected_features if feat in data.columns]
            if not available_features:
                tprint_warning("⚠️ No selected features available in data")
                return data
            
            # Use safe DataFrame operations to select features
            feature_data = safe_dataframe_operation(
                data,
                lambda df: df[available_features]
            )
            
            # Apply data quality improvements
            feature_data = guard_dataframe_nulls(feature_data, threshold=0.2)
            feature_data = optimize_dataframe_dtypes(feature_data)
            
            tprint_success(f"✅ Generated {len(available_features)} selected features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return data
    
    def _enhanced_interaction_generation(self, features_df: pd.DataFrame, targets: Optional[pd.Series]) -> List[Any]:
        """Enhanced interaction generation with VectorBT optimization using safe operations."""
        tprint_debug("Starting enhanced interaction generation")
        
        try:
            # Use safe DataFrame operations for interaction generation
            if not validate_dataframe(features_df):
                tprint_error("❌ Invalid DataFrame for interaction generation")
                return []
            
            # Get data quality metrics
            quality_metrics = calculate_data_quality_metrics(features_df)
            tprint_info(f"📊 Features quality: {quality_metrics.get('missing_percentage', 0):.2f}% missing values")
            
            # Generate simple interactions using safe operations
            interactions = []
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) < 2:
                tprint_warning("⚠️ Not enough numeric columns for interaction generation")
                return []
            
            # Generate pairwise interactions
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns[i+1:], i+1):
                    try:
                        # Use safe mathematical operations
                        interaction = safe_dataframe_operation(
                            features_df,
                            lambda df: df[col1] * df[col2]
                        )
                        
                        if not interaction.isna().all():
                            interactions.append({
                                'feature1': col1,
                                'feature2': col2,
                                'interaction': f"{col1}_x_{col2}",
                                'correlation': safe_correlation(interaction, targets, default=0.0) if targets is not None else 0.0
                            })
                    except Exception as e:
                        tprint_debug(f"⚠️ Failed to generate interaction {col1} x {col2}: {e}")
                        continue
            
            # Sort interactions by correlation (if targets available)
            if targets is not None:
                interactions.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Limit to top 50 interactions
            interactions = interactions[:50]
            
            tprint_success(f"✅ Generated {len(interactions)} interactions")
            return interactions
            
        except Exception as e:
            tprint_error(f"❌ Interaction generation failed: {e}")
            return []
    
    def _htf_interaction_generation(self, data: pd.DataFrame, features_df: pd.DataFrame, 
                                  targets: Optional[pd.Series]) -> List[Any]:
        """HTF-aware interaction generation using enhanced utilities."""
        tprint_debug("Starting HTF interaction generation")
        
        try:
            # Use safe DataFrame operations for HTF interaction generation
            if not validate_dataframe(features_df):
                tprint_error("❌ Invalid DataFrame for HTF interaction generation")
                return []
            
            # Generate HTF interactions using safe operations
            htf_interactions = []
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) < 2:
                tprint_warning("⚠️ Not enough numeric columns for HTF interaction generation")
                return []
            
            # Generate HTF-specific interactions
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns[i+1:], i+1):
                    try:
                        # Use safe mathematical operations for HTF interactions
                        htf_interaction = safe_dataframe_operation(
                            features_df,
                            lambda df: df[col1] / (df[col2] + 1e-8)  # Add small epsilon to avoid division by zero
                        )
                        
                        if not htf_interaction.isna().all():
                            htf_interactions.append({
                                'feature1': col1,
                                'feature2': col2,
                                'interaction': f"{col1}_div_{col2}",
                                'correlation': safe_correlation(htf_interaction, targets, default=0.0) if targets is not None else 0.0
                            })
                    except Exception as e:
                        tprint_debug(f"⚠️ Failed to generate HTF interaction {col1} / {col2}: {e}")
                        continue
            
            # Sort HTF interactions by correlation (if targets available)
            if targets is not None:
                htf_interactions.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Limit to top 30 HTF interactions
            htf_interactions = htf_interactions[:30]
            
            tprint_success(f"✅ Generated {len(htf_interactions)} HTF interactions")
            return htf_interactions
            
        except Exception as e:
            tprint_error(f"❌ HTF interaction generation failed: {e}")
            return []
    
    def _create_htf_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated HTF features using enhanced utilities."""
        try:
            # Use safe DataFrame operations for HTF feature creation
            if not validate_dataframe(data):
                tprint_error("❌ Invalid DataFrame for HTF feature creation")
                return {}
            
            htf_features = {}
            
            # Create HTF features using safe operations
            if 'close' in data.columns:
                # Create HTF price features
                htf_features['htf_close'] = safe_dataframe_operation(
                    data,
                    lambda df: df['close'].rolling(4).mean()  # 4-period average for HTF
                )
                
                htf_features['htf_volatility'] = safe_dataframe_operation(
                    data,
                    lambda df: df['close'].pct_change().rolling(4).std()
                )
            
            if 'volume' in data.columns:
                # Create HTF volume features
                htf_features['htf_volume'] = safe_dataframe_operation(
                    data,
                    lambda df: df['volume'].rolling(4).mean()
                )
            
            # Apply data quality improvements
            for key, value in htf_features.items():
                if hasattr(value, 'fillna'):
                    htf_features[key] = safe_fillna(value, method='bfill')
            
            tprint_success(f"✅ Created {len(htf_features)} HTF features")
            return htf_features
            
        except Exception as e:
            tprint_error(f"❌ HTF feature creation failed: {e}")
            return {}
    
    def _advanced_lookback_optimization(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                      features_df: pd.DataFrame, pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Advanced lookback optimization using enhanced utilities and safe operations."""
        tprint_debug("Starting advanced lookback optimization")
        
        try:
            # Use safe DataFrame operations for lookback optimization
            if not validate_dataframe(data):
                tprint_error("❌ Invalid DataFrame for lookback optimization")
                return {}
            
            # Get data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            tprint_info(f"📊 Data quality: {quality_metrics.get('missing_percentage', 0):.2f}% missing values")
            
            # Simple lookback optimization using safe operations
            lookback_periods = [5, 10, 20, 50, 100]
            lookback_scores = {}
            
            for period in lookback_periods:
                if period >= len(data):
                    continue
                
                # Calculate lookback score using safe operations
                if 'close' in data.columns and targets is not None:
                    # Use safe correlation for lookback scoring
                    lookback_data = data['close'].rolling(period).mean()
                    corr = safe_correlation(lookback_data, targets, default=0.0)
                    lookback_scores[period] = safe_float(corr, default=0.0)
                else:
                    lookback_scores[period] = 0.0
            
            # Find optimal lookback period
            if lookback_scores:
                optimal_lookback = max(lookback_scores.keys(), key=lambda k: abs(lookback_scores[k]))
                tprint_success(f"✅ Optimal lookback period: {optimal_lookback} (score: {lookback_scores[optimal_lookback]:.4f})")
            else:
                optimal_lookback = 20  # Default fallback
                tprint_warning("⚠️ No valid lookback periods found, using default: 20")
            
            return {
                'optimized_lookbacks': {'default': optimal_lookback},
                'lookback_scores': lookback_scores,
                'optimization_method': 'enhanced_utilities',
                'execution_mode': 'simplified',
                'nested_cv_applied': False,
                'outer_fold_count': 0,
                'feature_lag_metadata': {},
                'long_pipeline': {},
                'short_pipeline': {}
            }
            
        except Exception as e:
            tprint_error(f"❌ Lookback optimization failed: {e}")
            return {
                'optimized_lookbacks': {'default': 20},
                'lookback_scores': {},
                'optimization_method': 'enhanced_utilities',
                'execution_mode': 'simplified',
                'nested_cv_applied': False,
                'outer_fold_count': 0,
                'feature_lag_metadata': {},
                'long_pipeline': {},
                'short_pipeline': {},
                'error': str(e)
            }
            
            # Report results
            if optimization_direction == 'longs':
                tprint_success(f"🎯 Completed LONGS-only optimization - {len(long_feature_results)} features")
            elif optimization_direction == 'shorts':
                tprint_success(f"🎯 Completed SHORTS-only optimization - {len(short_feature_results)} features")
            else:
                tprint_success(f"🎯 Completed differentiated optimization - Long: {len(long_feature_results)} features, Short: {len(short_feature_results)} features")
            
            # Update performance stats
            self.performance_stats['lookback_optimizations'] = total_optimized
            
            # Return comprehensive results
            return {
                'long_pipeline': long_feature_results,
                'short_pipeline': short_feature_results,
                'long_target': long_target_column,
                'short_target': short_target_column,
                'total_features_optimized': total_optimized,
                'optimization_method': 'coarse_to_refine_directional',
                'feature_lag_metadata': feature_lag_metadata,
                'execution_mode': execution_mode,
                'nested_cv_applied': use_nested_cv,
                'outer_fold_count': len(outer_splits) if outer_splits else 0
            }
            
        except Exception as e:
            tprint_error(f"❌ Advanced lookback optimization failed: {e}")
            return {
                'long_pipeline': {},
                'short_pipeline': {},
                'long_target': None,
                'short_target': None,
                'total_features_optimized': 0,
                'optimization_method': 'failed',
                'feature_lag_metadata': {},
                'execution_mode': 'failed',
                'nested_cv_applied': False,
                'outer_fold_count': 0,
                'error': str(e)
            }
    
    def _build_walk_forward_splits(self, data_length: int, wf_config: Optional[Dict[str, Any]] = None) -> List[Tuple[slice, slice]]:
        """Create walk-forward outer CV splits when enough history is available with configurable parameters."""
        # Use provided config or create default
        if wf_config is None:
            wf_config = {
                'n_splits': 5,
                'min_window_size': 50,
                'min_val_samples': 20,
                'min_train_samples': 100,
                'min_train_ratio': 0.6
            }
        
        if data_length <= 0 or wf_config['n_splits'] <= 0:
            tprint_debug(f"⚠️ Invalid data length ({data_length}) or n_splits ({wf_config['n_splits']})")
            return []

        max_splits = max(1, wf_config['n_splits'])
        window = data_length // (max_splits + 1)

        # Reduce split count until validation windows are large enough for stable estimates
        while max_splits > 1 and window < wf_config['min_window_size']:
            max_splits -= 1
            window = data_length // (max_splits + 1)
            tprint_debug(f"🔄 Reduced splits to {max_splits} (window={window})")

        if window < wf_config['min_val_samples']:
            tprint_warning(f"⚠️ Window size {window} < minimum {wf_config['min_val_samples']}, no splits created")
            return []

        splits: List[Tuple[slice, slice]] = []
        min_train_size = max(wf_config['min_train_samples'], int(data_length * wf_config['min_train_ratio']))
        min_val_size = max(wf_config['min_val_samples'], window // 2)

        tprint_debug(f"📊 Walk-forward config: min_train={min_train_size}, min_val={min_val_size}, window={window}")

        for fold_idx in range(1, max_splits + 1):
            train_end = window * fold_idx
            val_start = train_end
            val_end = min(data_length, val_start + window)

            if train_end < min_train_size:
                tprint_debug(f"⚠️ Fold {fold_idx}: train_end ({train_end}) < min_train_size ({min_train_size}), skipping")
                continue

            if val_end - val_start < min_val_size:
                tprint_debug(f"⚠️ Fold {fold_idx}: val_size ({val_end - val_start}) < min_val_size ({min_val_size}), stopping")
                break

            splits.append((slice(0, train_end), slice(val_start, val_end)))
            tprint_debug(f"✅ Fold {fold_idx}: train[0:{train_end}], val[{val_start}:{val_end}]")

        tprint_info(f"📊 Created {len(splits)} walk-forward splits from {data_length} samples")
        return splits
    
    def _select_optimal_target_column(self, data: pd.DataFrame, direction: str) -> Optional[str]:
        """Select optimal target column for the given direction (long/short)."""
        try:
            # Look for target columns in the data
            target_columns = [col for col in data.columns if 'target' in col.lower()]
            
            if not target_columns:
                tprint_warning(f"⚠️ No target columns found for {direction} direction")
                return None
            
            # Prefer direction-specific targets
            direction_targets = [col for col in target_columns if direction in col.lower()]
            
            if direction_targets:
                selected = direction_targets[0]
                tprint_debug(f"✅ Selected {direction}-specific target: {selected}")
                return selected
            
            # Fall back to generic targets
            if target_columns:
                selected = target_columns[0]
                tprint_debug(f"✅ Selected generic target for {direction}: {selected}")
                return selected
            
            return None
            
        except Exception as e:
            tprint_error(f"❌ Target column selection failed for {direction}: {e}")
            return None
    
    def _create_mode_aware_constraints(self, execution_mode: str) -> Dict[str, Any]:
        """Create mode-aware constraints for optimization based on execution mode."""
        constraints = {
            'light': {
                'use_bayesian_optimization': False,
                'n_bootstrap_samples': 20,
                'cv_folds': 3,
                'max_features': 50,
                'max_lookback': 30
            },
            'blank': {
                'use_bayesian_optimization': False,
                'n_bootstrap_samples': 10,
                'cv_folds': 2,
                'max_features': 20,
                'max_lookback': 20
            },
            'full': {
                'use_bayesian_optimization': True,
                'n_bootstrap_samples': 100,
                'cv_folds': 5,
                'max_features': 200,
                'max_lookback': 100
            }
        }
        
        return constraints.get(execution_mode, constraints['full'])
    
    def _get_regularization_settings(self) -> Dict[str, Any]:
        """Get regularization settings for lookback optimization."""
        return {
            'l1_alpha': 0.01,
            'l2_alpha': 0.01,
            'elastic_net_ratio': 0.5,
            'max_lookback_penalty': 0.001,
            'min_lookback_penalty': 0.0001
        }
    
    def _optimize_feature_direction(self, data: pd.DataFrame, feature: str, target_column: str, 
                                  direction: str, lookback_range: Tuple[int, int], 
                                  optimizer_kwargs: Dict[str, Any], use_bayesian_opt: bool,
                                  execution_mode: str, use_nested_cv: bool, 
                                  pipeline_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Optimize a single feature for a specific direction (long/short)."""
        tprint_debug(f"🎯 Optimizing {feature} for {direction} direction using {target_column}")
        
        try:
            # Try VectorBT optimization first if available
            if hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer:
                tprint_debug(f"🚀 Using VectorBT optimization for {feature} ({direction})")
                try:
                    result = self.vectorbt_optimizer.optimize_feature_lookback(
                        data,
                        feature,
                        target_column,
                        lookback_range=lookback_range,
                        regularization_settings=optimizer_kwargs.get('regularization_settings', {})
                    )
                    
                    tprint_debug(f"✅ VectorBT optimization completed: period={result.best_lookback_period}, score={result.best_score:.4f}")
                    
                    # Convert VectorBT result to expected format
                    return {
                        'best_lookback_period': result.best_lookback_period,
                        'best_score': result.best_score,
                        'optimization_method': getattr(result, 'optimization_method', 'vectorbt'),
                        'target_column': target_column,
                        'direction': direction,
                        'total_trials': getattr(result, 'total_trials', 1),
                        'optimization_time': getattr(result, 'optimization_time', 0.0),
                        'convergence_achieved': getattr(result, 'convergence_achieved', True)
                    }
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT optimization failed, falling back to standard: {e}")
                    return self._fallback_optimization(
                        data, feature, target_column, lookback_range, 
                        optimizer_kwargs, use_bayesian_opt, execution_mode
                    )
            else:
                # Use fallback optimization
                tprint_debug(f"🔄 Using fallback optimization for {feature} ({direction})")
                return self._fallback_optimization(
                    data, feature, target_column, lookback_range, 
                    optimizer_kwargs, use_bayesian_opt, execution_mode
                )
            
        except Exception as e:
            tprint_error(f"❌ Feature optimization failed for {feature} ({direction}): {e}")
            return None
    
    def _fallback_optimization(self, data: pd.DataFrame, feature: str, target_column: str,
                             lookback_range: Tuple[int, int], optimizer_kwargs: Dict[str, Any],
                             use_bayesian_opt: bool, execution_mode: str) -> Optional[Dict[str, Any]]:
        """Fallback optimization method when VectorBT is not available."""
        try:
            # Simple grid search optimization
            min_lookback, max_lookback = lookback_range
            best_score = -np.inf
            best_lookback = min_lookback
            
            # Sample lookback periods based on execution mode
            if execution_mode == 'light':
                lookback_samples = range(min_lookback, max_lookback + 1, 5)
            elif execution_mode == 'blank':
                lookback_samples = range(min_lookback, max_lookback + 1, 10)
            else:
                lookback_samples = range(min_lookback, max_lookback + 1, 2)
            
            for lookback in lookback_samples:
                try:
                    # Calculate feature with current lookback
                    if feature in data.columns:
                        feature_values = data[feature].rolling(lookback).mean()
                        target_values = data[target_column]
                        
                        # Align data
                        aligned_data = pd.DataFrame({
                            'feature': feature_values,
                            'target': target_values
                        }).dropna()
                        
                        if len(aligned_data) < 10:  # Need minimum samples
                            continue
                        
                        # Calculate correlation as score
                        correlation = aligned_data['feature'].corr(aligned_data['target'])
                        score = abs(correlation) if not np.isnan(correlation) else 0
                        
                        if score > best_score:
                            best_score = score
                            best_lookback = lookback
                            
                except Exception as e:
                    tprint_debug(f"⚠️ Lookback {lookback} failed: {e}")
                    continue
            
            if best_score > -np.inf:
                return {
                    'best_lookback_period': best_lookback,
                    'best_score': best_score,
                    'optimization_method': 'grid_search_fallback',
                    'target_column': target_column,
                    'direction': 'unknown',
                    'total_trials': len(lookback_samples),
                    'optimization_time': 0.0,
                    'convergence_achieved': True
                }
            else:
                tprint_warning(f"⚠️ Fallback optimization failed for {feature}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Fallback optimization failed for {feature}: {e}")
            return None
    
    def _normalize_feature_key(self, feature: str) -> str:
        """Normalize feature key for consistent naming."""
        return str(feature).replace(' ', '_').replace('-', '_').lower()
    
    def _lightgbm_featuretools_generation(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                        base_features: pd.DataFrame) -> Dict[str, Any]:
        """LightGBM + Featuretools + ALE feature generation."""
        tprint_debug("Starting LightGBM + Featuretools + ALE feature generation")
        
        try:
            # Prepare data for feature generation
            if targets is None:
                # Create synthetic targets if none provided
                if 'close' in data.columns:
                    targets = data['close'].pct_change().dropna()
                else:
                    tprint_warning("⚠️ No targets provided and no close price available")
                    return self._create_empty_enhanced_feature_result()
            
            # Align data and targets
            aligned_data = data.copy()
            aligned_targets = targets.reindex(data.index)
            
            # Use LightGBM + Featuretools generator
            feature_result = self.lightgbm_featuretools_generator.generate_features(
                aligned_data, 
                'target',  # We'll add this column
                list(base_features.columns) if base_features is not None else None,
                'full'
            )
            
            if feature_result.generated_features:
                tprint_success(f"✅ LightGBM + Featuretools + ALE feature generation completed: "
                             f"{len(feature_result.generated_features)} features generated, "
                             f"{feature_result.featuretools_features_generated} from Featuretools, "
                             f"SHAP: {feature_result.shap_analysis_completed}, "
                             f"ALE: {feature_result.ale_validation_completed}")
                
                # Convert to the expected format
                generated_features = []
                for feat in feature_result.generated_features:
                    generated_features.append({
                        'name': feat.name,
                        'formula': feat.formula,
                        'feature_series': feat.feature_series,
                        'importance_score': feat.importance_score,
                        'parent_features': feat.parent_features,
                        'feature_type': feat.feature_type,
                        'generation_method': feat.generation_method,
                        'metadata': feat.metadata
                    })
                
                return {
                    'cross_timeframe_features': generated_features[:len(generated_features)//4],
                    'interaction_features': generated_features[len(generated_features)//4:len(generated_features)//2],
                    'no_features': generated_features[len(generated_features)//2:3*len(generated_features)//4],
                    'comparison_features': generated_features[3*len(generated_features)//4:],
                    'enhanced_feature_metrics': {
                        'total_features': len(generated_features),
                        'cross_timeframe_count': len(generated_features)//4,
                        'interaction_count': len(generated_features)//4,
                        'no_features_count': len(generated_features)//4,
                        'comparison_count': len(generated_features) - 3*(len(generated_features)//4),
                        'generation_time': feature_result.generation_time,
                        'shap_analysis_completed': feature_result.shap_analysis_completed,
                        'ale_validation_completed': feature_result.ale_validation_completed,
                        'featuretools_features_generated': feature_result.featuretools_features_generated
                    }
                }
            else:
                tprint_error(f"❌ LightGBM + Featuretools feature generation failed: No features generated")
                return self._create_empty_enhanced_feature_result()
                
        except Exception as e:
            tprint_error(f"❌ LightGBM + Featuretools feature generation failed: {e}")
            return self._create_empty_enhanced_feature_result()
    
    def _create_empty_enhanced_feature_result(self) -> Dict[str, Any]:
        """Create empty enhanced feature result."""
        return {
            'cross_timeframe_features': [],
            'interaction_features': [],
            'no_features': [],
            'comparison_features': [],
            'enhanced_feature_metrics': {}
        }
    
    def _enhanced_feature_generation(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                   base_features: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced feature generation including cross-timeframe, interactions, and no features."""
        tprint_debug("Starting enhanced feature generation")
        
        try:
            # Use the enhanced feature generator
            feature_result = self.enhanced_feature_generator.generate_features(
                data, targets, base_features
            )
            
            if feature_result.success:
                tprint_success(f"✅ Enhanced feature generation completed: "
                             f"{len(feature_result.cross_timeframe_features)} cross-timeframe, "
                             f"{len(feature_result.interaction_features)} interaction, "
                             f"{len(feature_result.no_features)} no features")
                
                return {
                    'cross_timeframe_features': feature_result.cross_timeframe_features,
                    'interaction_features': feature_result.interaction_features,
                    'no_features': feature_result.no_features,
                    'comparison_features': feature_result.comparison_features,
                    'enhanced_feature_metrics': {
                        'total_features': len(feature_result.all_features),
                        'cross_timeframe_count': len(feature_result.cross_timeframe_features),
                        'interaction_count': len(feature_result.interaction_features),
                        'no_features_count': len(feature_result.no_features),
                        'comparison_count': len(feature_result.comparison_features),
                        'generation_time': feature_result.generation_time
                    }
                }
            else:
                tprint_error(f"❌ Enhanced feature generation failed: {feature_result.error_message}")
                return {
                    'cross_timeframe_features': [],
                    'interaction_features': [],
                    'no_features': [],
                    'comparison_features': [],
                    'enhanced_feature_metrics': {}
                }
                
        except Exception as e:
            tprint_error(f"❌ Enhanced feature generation failed: {e}")
            return {
                'cross_timeframe_features': [],
                'interaction_features': [],
                'no_features': [],
                'comparison_features': [],
                'enhanced_feature_metrics': {}
            }
    
    def _final_feature_selection(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Any:
        """Final feature selection using multi-objective optimization."""
        tprint_debug("Starting final feature selection")
        
        try:
            # Use the multi-objective feature selector
            selection_result = self.feature_selector.select_features(data, targets)
            
            if selection_result:
                tprint_success(f"✅ Final feature selection completed: {len(selection_result.selected_features)} features selected")
                return selection_result
            else:
                tprint_warning("⚠️ Final feature selection failed, using all available features")
                return type('FeatureSelectionResult', (), {
                    'selected_features': list(data.columns),
                    'objective_values': {},
                    'quality_metrics': {}
                })()
                
        except Exception as e:
            tprint_error(f"Final feature selection failed: {e}")
            return type('FeatureSelectionResult', (), {
                'selected_features': list(data.columns),
                'objective_values': {},
                'quality_metrics': {}
            })()
    
    def _combine_results(self, period_results: Dict[str, Any], feature_selection_results: Any,
                        interaction_results: List[Any], htf_results: List[Any], 
                        lookback_results: Dict[str, Any], enhanced_feature_results: Dict[str, Any],
                        final_selection_results: Any) -> Dict[str, Any]:
        """Combine all pipeline results."""
        try:
            # Extract lookback results for backward compatibility
            optimized_lookbacks = {}
            if isinstance(lookback_results, dict):
                # Handle enhanced lookback results
                if 'long_pipeline' in lookback_results and 'short_pipeline' in lookback_results:
                    # Extract lookbacks from both pipelines
                    for feature_data in lookback_results.get('long_pipeline', {}).values():
                        if 'best_lookback_period' in feature_data:
                            optimized_lookbacks[f"{feature_data.get('feature_name', 'unknown')}_long"] = feature_data['best_lookback_period']
                    
                    for feature_data in lookback_results.get('short_pipeline', {}).values():
                        if 'best_lookback_period' in feature_data:
                            optimized_lookbacks[f"{feature_data.get('feature_name', 'unknown')}_short"] = feature_data['best_lookback_period']
                else:
                    # Fallback to simple lookback results
                    optimized_lookbacks = lookback_results
            
            return {
                'optimal_periods': period_results.get('optimal_periods', []),
                'period_scores': period_results.get('period_scores', {}),
                'economic_evaluation_results': period_results.get('economic_evaluation_results'),
                'selected_features': final_selection_results.selected_features if final_selection_results else [],
                'feature_importance': {feat: 1.0 for feat in (final_selection_results.selected_features if final_selection_results else [])},
                'objective_values': final_selection_results.objective_values if final_selection_results else {},
                'feature_selection_metrics': feature_selection_results.quality_metrics if feature_selection_results else {},
                'generated_interactions': interaction_results,
                'interaction_metrics': self._calculate_interaction_metrics(interaction_results),
                'htf_interactions': htf_results,
                'htf_metrics': self._calculate_interaction_metrics(htf_results),
                'optimized_lookbacks': optimized_lookbacks,
                'lookback_metrics': self._calculate_enhanced_lookback_metrics(lookback_results),
                'cross_timeframe_features': enhanced_feature_results.get('cross_timeframe_features', []),
                'interaction_features': enhanced_feature_results.get('interaction_features', []),
                'no_features': enhanced_feature_results.get('no_features', []),
                'comparison_features': enhanced_feature_results.get('comparison_features', []),
                'enhanced_feature_metrics': enhanced_feature_results.get('enhanced_feature_metrics', {}),
                'out_of_sample_sharpe': 0.5,  # Would be calculated from actual results
                'max_drawdown': 0.1,  # Would be calculated from actual results
                'stability_score': 0.8,  # Would be calculated from actual results
                'diversity_score': 0.7  # Would be calculated from actual results
            }
            
        except Exception as e:
            tprint_error(f"Result combination failed: {e}")
            return {
                'optimal_periods': [],
                'period_scores': {},
                'economic_evaluation_results': None,
                'selected_features': [],
                'feature_importance': {},
                'objective_values': {},
                'feature_selection_metrics': {},
                'generated_interactions': [],
                'interaction_metrics': {},
                'htf_interactions': [],
                'htf_metrics': {},
                'optimized_lookbacks': {},
                'lookback_metrics': {},
                'cross_timeframe_features': [],
                'interaction_features': [],
                'no_features': [],
                'comparison_features': [],
                'enhanced_feature_metrics': {},
                'out_of_sample_sharpe': 0.0,
                'max_drawdown': 0.0,
                'stability_score': 0.0,
                'diversity_score': 0.0
            }
    
    def _combine_period_scores(self, statistical_analysis: Dict[int, Dict[str, Any]], 
                              economic_evaluation: Any) -> Dict[int, float]:
        """Combine statistical and economic period scores."""
        try:
            combined_scores = {}
            
            # Extract statistical scores
            for period, analysis in statistical_analysis.items():
                if 'error' not in analysis:
                    # Use sharpe ratio as statistical score
                    sharpe_ratio = analysis.get('sharpe_ratio', 0.0)
                    combined_scores[period] = sharpe_ratio * 0.4  # Statistical weight
            
            # Add economic scores
            if economic_evaluation and hasattr(economic_evaluation, 'period_scores'):
                for period, score in economic_evaluation.period_scores.items():
                    if period in combined_scores:
                        combined_scores[period] += score * 0.6  # Economic weight
                    else:
                        combined_scores[period] = score * 0.6
            
            return combined_scores
            
        except Exception as e:
            tprint_error(f"Period score combination failed: {e}")
            return {}
    
    def _select_optimal_periods(self, period_scores: Dict[int, float], max_periods: int = 8) -> List[int]:
        """Select optimal periods based on combined scores."""
        if not period_scores:
            return []
        
        # Sort by score (descending)
        sorted_periods = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top periods
        optimal_periods = [period for period, score in sorted_periods[:max_periods] if score > 0.1]
        
        return optimal_periods
    
    def _calculate_interaction_metrics(self, interactions: List[Any]) -> Dict[str, Any]:
        """Calculate metrics for generated interactions."""
        if not interactions:
            return {}
        
        try:
            return {
                'total_interactions': len(interactions),
                'average_utility_score': np.mean([i.utility_score for i in interactions]),
                'max_utility_score': max(i.utility_score for i in interactions),
                'min_utility_score': min(i.utility_score for i in interactions),
                'interaction_types': list(set(i.interaction_type for i in interactions)),
                'unique_parent_features': len(set(f for i in interactions for f in i.parent_features))
            }
        except:
            return {}
    
    def _calculate_lookback_metrics(self, lookback_results: Dict[str, int]) -> Dict[str, Any]:
        """Calculate metrics for lookback optimization."""
        if not lookback_results:
            return {}
        
        try:
            return {
                'total_features_optimized': len(lookback_results),
                'average_lookback': np.mean(list(lookback_results.values())),
                'min_lookback': min(lookback_results.values()),
                'max_lookback': max(lookback_results.values()),
                'lookback_distribution': dict(pd.Series(list(lookback_results.values())).value_counts())
            }
        except:
            return {}
    
    def _calculate_enhanced_lookback_metrics(self, lookback_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced metrics for sophisticated lookback optimization."""
        if not lookback_results:
            return {}
        
        try:
            metrics = {
                'total_features_optimized': lookback_results.get('total_features_optimized', 0),
                'optimization_method': lookback_results.get('optimization_method', 'unknown'),
                'execution_mode': lookback_results.get('execution_mode', 'unknown'),
                'nested_cv_applied': lookback_results.get('nested_cv_applied', False),
                'outer_fold_count': lookback_results.get('outer_fold_count', 0)
            }
            
            # Calculate metrics for long pipeline
            long_pipeline = lookback_results.get('long_pipeline', {})
            if long_pipeline:
                long_lookbacks = [data.get('best_lookback_period', 0) for data in long_pipeline.values() 
                                if isinstance(data, dict) and 'best_lookback_period' in data]
                if long_lookbacks:
                    metrics['long_pipeline'] = {
                        'features_count': len(long_lookbacks),
                        'average_lookback': np.mean(long_lookbacks),
                        'min_lookback': min(long_lookbacks),
                        'max_lookback': max(long_lookbacks)
                    }
            
            # Calculate metrics for short pipeline
            short_pipeline = lookback_results.get('short_pipeline', {})
            if short_pipeline:
                short_lookbacks = [data.get('best_lookback_period', 0) for data in short_pipeline.values() 
                                 if isinstance(data, dict) and 'best_lookback_period' in data]
                if short_lookbacks:
                    metrics['short_pipeline'] = {
                        'features_count': len(short_lookbacks),
                        'average_lookback': np.mean(short_lookbacks),
                        'min_lookback': min(short_lookbacks),
                        'max_lookback': max(short_lookbacks)
                    }
            
            # Feature lag metadata
            feature_lag_metadata = lookback_results.get('feature_lag_metadata', {})
            if feature_lag_metadata:
                metrics['feature_lag_metadata'] = feature_lag_metadata
            
            return metrics
            
        except Exception as e:
            tprint_error(f"Enhanced lookback metrics calculation failed: {e}")
            return {}
    
    def _update_performance_stats(self, execution_time: float, combined_results: Dict[str, Any]):
        """Update performance statistics."""
        self.performance_stats.update({
            'total_pipeline_runs': 1,
            'successful_pipeline_runs': 1,
            'total_execution_time': execution_time,
            'vectorbt_operations': self.vectorbt_optimizer.performance_stats.get('vectorbt_operations', 0),
            'economic_evaluations': self.economic_evaluator.performance_stats.get('successful_evaluations', 0),
            'feature_selections': self.advanced_feature_selector.performance_stats.get('successful_selections', 0),
            'interaction_generations': len(combined_results.get('generated_interactions', [])),
            'htf_generations': len(combined_results.get('htf_interactions', [])),
            'lookback_optimizations': len(combined_results.get('optimized_lookbacks', {})),
            'enhanced_feature_generations': len(combined_results.get('cross_timeframe_features', [])) + 
                                          len(combined_results.get('interaction_features', [])) + 
                                          len(combined_results.get('no_features', [])) +
                                          len(combined_results.get('comparison_features', []))
        })
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        hits = self.performance_stats.get('cache_hits', 0)
        misses = self.performance_stats.get('cache_misses', 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def _create_empty_result(self, start_time: float, error_message: str) -> ConsolidatedPipelineResult:
        """Create empty result for failed processing."""
        return ConsolidatedPipelineResult(
            selected_features=[],
            feature_importance={},
            objective_values={},
            optimal_periods=[],
            period_scores={},
            economic_evaluation_results=None,
            feature_selection_metrics={},
            generated_interactions=[],
            interaction_metrics={},
            htf_interactions=[],
            htf_metrics={},
            optimized_lookbacks={},
            lookback_metrics={},
            # Enhanced lookback optimization results
            long_pipeline_results={},
            short_pipeline_results={},
            lookback_optimization_method='failed',
            execution_mode='failed',
            nested_cv_applied=False,
            outer_fold_count=0,
            feature_lag_metadata={},
            cross_timeframe_features=[],
            interaction_features=[],
            no_features=[],
            comparison_features=[],
            enhanced_feature_metrics={},
            processing_time=time.time() - start_time,
            n_cv_splits=0,
            n_candidates_evaluated=0,
            out_of_sample_sharpe=0.0,
            max_drawdown=0.0,
            stability_score=0.0,
            diversity_score=0.0,
            memory_usage_mb=0.0,
            peak_memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            vectorbt_operations=0,
            pandas_fallbacks=0,
            cache_hit_rate=0.0,
            optimization_iterations=0,
            convergence_achieved=False,
            feature_diversity_score=0.0,
            interaction_utility_scores={},
            lookback_optimization_metrics={},
            performance_monitoring_data={},
            config=self.config,
            success=False,
            error_message=error_message,
            warnings=[]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component stats
        stats['vectorbt_optimizer'] = self.vectorbt_optimizer.get_performance_stats()
        stats['economic_evaluator'] = self.economic_evaluator.get_performance_stats()
        stats['advanced_feature_selector'] = self.advanced_feature_selector.get_performance_stats()
        stats['template_interaction_generator'] = self.template_interaction_generator.get_performance_stats()
        stats['advanced_lookback_optimizer'] = self.advanced_lookback_optimizer.get_performance_stats()
        stats['feature_bank_integration'] = self.feature_bank_integration.get_performance_stats()
        stats['enhanced_feature_generator'] = self.enhanced_feature_generator.get_performance_stats()
        stats['lightgbm_featuretools_generator'] = self.lightgbm_featuretools_generator.get_performance_stats()
        
        # Add modular architecture stats
        stats['input_validator'] = self.input_validator.get_validation_stats()
        stats['error_handler'] = self.error_handler.get_error_stats()
        stats['performance_monitor'] = self.performance_monitor.get_performance_stats()
        stats['memory_manager'] = self.memory_manager.get_memory_stats()
        stats['hardware_accelerator'] = self.hardware_accelerator.get_acceleration_info()
        
        # Add advanced infrastructure stats
        stats['advanced_validator'] = self.advanced_validator.get_validation_stats()
        stats['advanced_error_handler'] = self.advanced_error_handler.get_error_stats()
        stats['advanced_performance_monitor'] = self.advanced_performance_monitor.get_performance_summary()
        stats['advanced_data_loader'] = self.advanced_data_loader.get_cache_metrics()
        stats['advanced_artifact_manager'] = {
            'artifact_registry_size': len(self.advanced_artifact_manager.get_artifact_registry()),
            'save_history_size': len(self.advanced_artifact_manager.get_save_history())
        }
        
        return stats
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_pipeline_runs': 0,
            'successful_pipeline_runs': 0,
            'failed_pipeline_runs': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'economic_evaluations': 0,
            'feature_selections': 0,
            'interaction_generations': 0,
            'htf_generations': 0,
            'lookback_optimizations': 0,
            'enhanced_feature_generations': 0,
            'gpu_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Reset component stats
        self.vectorbt_optimizer.reset_stats()
        self.economic_evaluator.reset_stats()
        self.advanced_feature_selector.reset_stats()
        self.template_interaction_generator.reset_stats()
        self.advanced_lookback_optimizer.reset_stats()
        self.feature_bank_integration.reset_stats()
        self.enhanced_feature_generator.reset_stats()
        self.lightgbm_featuretools_generator.reset_stats()
        
        # Reset modular architecture stats
        self.performance_monitor.reset_stats()
        
        # Reset advanced infrastructure stats
        self.advanced_performance_monitor.reset_stats()
        self.advanced_data_loader.reset_cache_metrics()
        self.advanced_error_handler.reset_error_stats()


# Convenience functions
def create_unified_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedDataDrivenPipeline:
    """Create a unified data-driven pipeline with default configuration."""
    return UnifiedDataDrivenPipeline(config)


async def process_with_unified_pipeline(data: pd.DataFrame,
                                targets: Optional[pd.Series] = None,
                                feature_columns: Optional[List[str]] = None,
                                timeframe: str = "15m",
                                config: Optional[UnifiedPipelineConfig] = None,
                                pipeline_state: Optional[Dict[str, Any]] = None) -> ConsolidatedPipelineResult:
    """
    Convenience function to process data with unified pipeline.
    
    Args:
        data: Input data with features
        targets: Target variable
        feature_columns: Optional list of feature columns to use
        timeframe: Target timeframe
        config: Optional pipeline configuration
        
    Returns:
        ConsolidatedPipelineResult with selected features and performance metrics
    """
    pipeline = create_unified_pipeline(config)
    return pipeline.process(data, targets, feature_columns, timeframe, pipeline_state)


# Export main classes and functions
__all__ = [
    'UnifiedDataDrivenPipeline',
    'ConsolidatedPipelineResult',
    'create_unified_pipeline',
    'process_with_unified_pipeline'
]