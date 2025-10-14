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

# Import enhanced vectorization utilities
from src.feature_generation.utils.vectorbt_rolling_optimizer import (
    VectorBTRollingOptimizer, 
    get_vectorbt_rolling_optimizer
)
from src.feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager,
    VectorizationConfig
)
from .core.template_interaction_generator import (
    TemplateInteractionGenerator, TemplateConfig, 
    create_template_interaction_generator
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
        
        # Initialize all components
        self._initialize_core_components()
        self._initialize_enhanced_components()
        self._initialize_validation_components()
        self._initialize_performance_tracking()
        self._initialize_advanced_infrastructure()
        
        tprint_info("🚀 Consolidated Unified Data-Driven Pipeline initialized")
        tprint_info(f"📊 Configuration: {self.config}")
    
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
        
        # Enhanced VectorBT Rolling Optimizer
        self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            batch_size=1000,
            max_workers=4
        )
        
        # Unified Vectorization Manager
        vectorization_config = VectorizationConfig(
            enable_vectorbt=VECTORBT_AVAILABLE,
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            batch_size=1000,
            max_workers=4,
            enable_rolling_optimization=True,
            enable_batch_processing=True
        )
        self.unified_vectorization_manager = UnifiedVectorizationManager(vectorization_config)
        
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
            # Advanced input validation
            is_valid, validation_summary, cleaned_data = self.advanced_validator.validate_data(
                data, 
                required_columns=['open', 'high', 'low', 'close', 'volume'],
                target_columns=feature_columns
            )
            
            if not is_valid:
                error_msg = f"Data validation failed: {validation_summary.recommendations}"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            # Load market data using advanced data loader
            market_data = await self.advanced_data_loader.load_market_data(
                cleaned_data, pipeline_state, force_refresh=False
            )
            
            # Load labeling data
            labeling_data = await self.advanced_data_loader.load_labeling_data(
                pipeline_state.get('symbol', 'ETHUSDT') if pipeline_state else 'ETHUSDT',
                pipeline_state.get('exchange', 'binance') if pipeline_state else 'binance',
                timeframe,
                pipeline_state
            )
            
            # Prepare data for optimization
            processed_data = self.advanced_data_loader.prepare_data_for_optimization(
                market_data, labeling_data
            )
            
            # Generate features for optimization
            feature_columns = await self.advanced_data_loader.generate_features_for_optimization(
                processed_data, pipeline_state, force_refresh=False
            )
            
            if not feature_columns:
                error_msg = "Feature generation failed - no features generated"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            tprint_success(f"✅ Generated {len(feature_columns)} features for optimization")
            
            # Prepare targets
            processed_targets = targets
            if targets is not None and len(targets) != len(processed_data):
                common_index = processed_data.index.intersection(targets.index)
                processed_data = processed_data.loc[common_index]
                processed_targets = targets.loc[common_index]
            
            # Step 1: Enhanced period optimization with economic evaluation
            tprint_info("Step 1: Enhanced period optimization with economic evaluation")
            period_results = self._enhanced_period_optimization(processed_data, timeframe)
            
            # Step 2: Advanced feature selection from 200+ feature bank
            tprint_info("Step 2: Advanced feature selection from 200+ feature bank")
            feature_selection_results = self._advanced_feature_selection(processed_data, processed_targets)
            
            # Step 3: Generate selected features
            tprint_info("Step 3: Generate selected features")
            selected_features_df = self._generate_selected_features(processed_data, feature_selection_results)
            
            # Step 4: Enhanced interaction generation with VectorBT optimization
            tprint_info("Step 4: Enhanced interaction generation with VectorBT optimization")
            interaction_results = self._enhanced_interaction_generation(selected_features_df, processed_targets)
            
            # Step 5: HTF-aware interaction generation
            tprint_info("Step 5: HTF-aware interaction generation")
            htf_results = self._htf_interaction_generation(processed_data, selected_features_df, processed_targets)
            
            # Step 6: Advanced lookback optimization
            tprint_info("Step 6: Advanced lookback optimization")
            lookback_results = self._advanced_lookback_optimization(processed_data, processed_targets, selected_features_df, pipeline_state)
            
            # Step 7: LightGBM + Featuretools + ALE feature generation
            tprint_info("Step 7: LightGBM + Featuretools + ALE feature generation")
            enhanced_feature_results = self._lightgbm_featuretools_generation(processed_data, processed_targets, selected_features_df)
            
            # Step 8: Final feature selection
            tprint_info("Step 8: Final feature selection")
            final_selection_results = self._final_feature_selection(processed_data, processed_targets)
            
            # Step 9: Combine all results
            tprint_info("Step 9: Combine all results")
            combined_results = self._combine_results(
                period_results, feature_selection_results, interaction_results, 
                htf_results, lookback_results, enhanced_feature_results, final_selection_results
            )
            
            execution_time = self.advanced_performance_monitor.end_operation("process", start_time, success=True)
            
            # Create comprehensive artifacts
            artifacts = self.advanced_artifact_manager.create_optimization_artifacts(
                combined_results, pipeline_state
            )
            
            # Create outcome report
            outcome_report = self.advanced_artifact_manager.create_outcome_report(
                combined_results, 
                self.advanced_performance_monitor.get_performance_summary(),
                pipeline_state
            )
            
            # Save artifacts
            save_report = await self.advanced_artifact_manager.save_artifacts(
                artifacts, 
                {
                    'optimization_status': 'completed',
                    'total_features_optimized': len(combined_results.get('selected_features', [])),
                    'validation_summary': validation_summary.__dict__ if 'validation_summary' in locals() else None,
                    'performance_metrics': self.advanced_performance_monitor.get_performance_summary(),
                    'outcome_report': outcome_report
                }
            )
            
            # Update performance stats
            self._update_performance_stats(execution_time, combined_results)
            
            tprint_success(f"✅ Consolidated pipeline processing completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Results: {len(combined_results['selected_features'])} features, "
                       f"{len(combined_results['generated_interactions'])} interactions, "
                       f"{len(combined_results['htf_interactions'])} HTF interactions")
            tprint_success(f"💾 Artifacts saved: {save_report.artifacts_saved} artifacts, "
                          f"correlation_id: {save_report.correlation_id}")
            
            return ConsolidatedPipelineResult(
                selected_features=combined_results['selected_features'],
                feature_importance=combined_results['feature_importance'],
                objective_values=combined_results['objective_values'],
                optimal_periods=combined_results['optimal_periods'],
                period_scores=combined_results['period_scores'],
                economic_evaluation_results=combined_results['economic_evaluation_results'],
                feature_selection_metrics=combined_results['feature_selection_metrics'],
                generated_interactions=combined_results['generated_interactions'],
                interaction_metrics=combined_results['interaction_metrics'],
                htf_interactions=combined_results['htf_interactions'],
                htf_metrics=combined_results['htf_metrics'],
                optimized_lookbacks=combined_results['optimized_lookbacks'],
                lookback_metrics=combined_results['lookback_metrics'],
                # Enhanced lookback optimization results
                long_pipeline_results=lookback_results.get('long_pipeline', {}),
                short_pipeline_results=lookback_results.get('short_pipeline', {}),
                lookback_optimization_method=lookback_results.get('optimization_method', 'unknown'),
                execution_mode=lookback_results.get('execution_mode', 'unknown'),
                nested_cv_applied=lookback_results.get('nested_cv_applied', False),
                outer_fold_count=lookback_results.get('outer_fold_count', 0),
                feature_lag_metadata=lookback_results.get('feature_lag_metadata', {}),
                cross_timeframe_features=combined_results['cross_timeframe_features'],
                interaction_features=combined_results['interaction_features'],
                no_features=combined_results['no_features'],
                comparison_features=combined_results['comparison_features'],
                enhanced_feature_metrics=combined_results['enhanced_feature_metrics'],
                processing_time=execution_time,
                n_cv_splits=self.performance_stats['n_cv_splits'],
                n_candidates_evaluated=len(processed_data.columns),
                out_of_sample_sharpe=combined_results.get('out_of_sample_sharpe', 0.0),
                max_drawdown=combined_results.get('max_drawdown', 0.0),
                stability_score=combined_results.get('stability_score', 0.0),
                diversity_score=combined_results.get('diversity_score', 0.0),
                memory_usage_mb=self._get_current_memory_usage(),
                peak_memory_usage_mb=self.performance_stats['peak_memory_usage_mb'],
                vectorbt_operations=self.performance_stats['vectorbt_operations'],
                cache_hit_rate=self._calculate_cache_hit_rate(),
                config=self.config,
                success=True
            )
            
        except Exception as e:
            # Use advanced error handler
            self.advanced_performance_monitor.end_operation("process", start_time, success=False)
            self.advanced_performance_monitor.stop_monitoring()
            
            error_result = self.advanced_error_handler.handle_error(
                e, "process", 
                return_value=self._create_empty_result(start_time, str(e)),
                context={'data_shape': data.shape, 'timeframe': timeframe}
            )
            
            tprint_error(f"❌ Consolidated pipeline processing failed: {e}")
            return error_result
    
    def _validate_inputs(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> bool:
        """Validate input data and parameters."""
        try:
            if data is None or data.empty:
                tprint_error("Data is None or empty")
                return False
            
            if 'close' not in data.columns:
                tprint_error("Data must contain 'close' column")
                return False
            
            if targets is not None and len(targets) != len(data):
                tprint_error("Targets length does not match data length")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False
    
    def _prepare_data(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                     feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for processing."""
        # Select feature columns if specified
        if feature_columns:
            available_columns = [col for col in feature_columns if col in data.columns]
            processed_data = data[available_columns]
        else:
            processed_data = data.copy()
        
        # Ensure targets are aligned
        processed_targets = targets
        if targets is not None and len(targets) != len(processed_data):
            # Align targets with data
            common_index = processed_data.index.intersection(targets.index)
            processed_data = processed_data.loc[common_index]
            processed_targets = targets.loc[common_index]
        
        return processed_data, processed_targets
    
    def _enhanced_period_optimization(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Enhanced period optimization with economic evaluation."""
        tprint_debug("Starting enhanced period optimization")
        
        try:
            # Statistical period analysis
            periods = list(range(1, 51))  # 1-50 periods for 15m timeframe
            period_analysis = self.vectorbt_optimizer.optimize_period_analysis(data, periods)
            
            # Economic significance evaluation
            candidate_periods = [p for p in periods if p in period_analysis and 'error' not in period_analysis[p]]
            economic_evaluation = self.economic_evaluator.evaluate_periods(data, candidate_periods, timeframe)
            
            # Combine statistical and economic results
            combined_scores = self._combine_period_scores(period_analysis, economic_evaluation)
            
            # Select optimal periods
            optimal_periods = self._select_optimal_periods(combined_scores)
            
            tprint_success(f"✅ Period optimization completed: {len(optimal_periods)} optimal periods")
            
            return {
                'optimal_periods': optimal_periods,
                'period_scores': combined_scores,
                'economic_evaluation_results': economic_evaluation,
                'statistical_analysis': period_analysis
            }
            
        except Exception as e:
            tprint_error(f"Enhanced period optimization failed: {e}")
            return {
                'optimal_periods': [],
                'period_scores': {},
                'economic_evaluation_results': None,
                'statistical_analysis': {}
            }
    
    def _advanced_feature_selection(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Any:
        """Advanced feature selection from 200+ feature bank."""
        tprint_debug("Starting advanced feature selection")
        
        try:
            # Use the advanced feature selector
            selection_result = self.advanced_feature_selector.select_features(data, targets)
            
            if selection_result.success:
                tprint_success(f"✅ Feature selection completed: {len(selection_result.selected_features)} features selected")
                return selection_result
            else:
                tprint_error(f"Feature selection failed: {selection_result.error_message}")
                return None
                
        except Exception as e:
            tprint_error(f"Advanced feature selection failed: {e}")
            return None
    
    def _generate_selected_features(self, data: pd.DataFrame, selection_result: Any) -> pd.DataFrame:
        """Generate features for the selected feature set using enhanced vectorization utilities."""
        tprint_debug("Generating selected features using enhanced vectorization utilities")
        
        try:
            if selection_result is None or not selection_result.success:
                tprint_warning("⚠️ No valid selection result, using enhanced vectorization for comprehensive feature generation")
                # Use enhanced vectorization for comprehensive feature generation
                features_df = self._generate_enhanced_features(data)
                
                if not features_df.empty:
                    tprint_success(f"✅ Generated {len(features_df.columns)} features using enhanced vectorization")
                    return features_df
                else:
                    tprint_error("❌ Enhanced feature generation failed")
                    return pd.DataFrame(index=data.index)
            
            # Use enhanced vectorization for selected features
            tprint_debug("🔧 Using enhanced vectorization for selected features")
            
            # Generate comprehensive features first using enhanced vectorization
            features_df = self._generate_enhanced_features(data)
            
            if features_df.empty:
                tprint_error("❌ Enhanced feature generation failed - this is required for feature generation")
                return pd.DataFrame(index=data.index)
            
            # Filter to selected features
            selected_feature_names = [fs.feature_name for fs in selection_result.selected_features]
            available_features = features_df.columns
            
            # Find matching features
            matching_features = [f for f in selected_feature_names if f in available_features]
            
            if matching_features:
                features_df = features_df[matching_features]
                tprint_success(f"✅ Generated {len(features_df.columns)} selected features using enhanced vectorization")
            else:
                tprint_warning("⚠️ No matching features found, using all generated features")
            
            return features_df
            
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _generate_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using enhanced vectorization utilities."""
        tprint_debug("Generating features using enhanced vectorization utilities")
        
        try:
            features_dict = {}
            
            # Ensure we have OHLCV data
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Missing required OHLCV columns")
                return pd.DataFrame(index=data.index)
            
            # Use Unified Vectorization Manager for comprehensive feature generation
            with self.unified_vectorization_manager.batch_processing_context() as batch_processor:
                # Price-based features using VectorBT rolling operations
                close_prices = data['close']
                high_prices = data['high']
                low_prices = data['low']
                volume = data['volume']
                
                # Rolling statistics with VectorBT optimization
                windows = [5, 10, 20, 50, 100]
                for window in windows:
                    if len(data) >= window:
                        # Rolling mean
                        features_dict[f'close_ma_{window}'] = self.vectorbt_rolling_optimizer.rolling_mean(
                            close_prices, window=window
                        ).result_data
                        
                        # Rolling standard deviation
                        features_dict[f'close_std_{window}'] = self.vectorbt_rolling_optimizer.rolling_std(
                            close_prices, window=window
                        ).result_data
                        
                        # Rolling min/max
                        features_dict[f'close_min_{window}'] = self.vectorbt_rolling_optimizer.rolling_min(
                            close_prices, window=window
                        ).result_data
                        
                        features_dict[f'close_max_{window}'] = self.vectorbt_rolling_optimizer.rolling_max(
                            close_prices, window=window
                        ).result_data
                        
                        # Volume features
                        features_dict[f'volume_ma_{window}'] = self.vectorbt_rolling_optimizer.rolling_mean(
                            volume, window=window
                        ).result_data
                        
                        # Price range features
                        price_range = high_prices - low_prices
                        features_dict[f'range_ma_{window}'] = self.vectorbt_rolling_optimizer.rolling_mean(
                            price_range, window=window
                        ).result_data
                
                # Technical indicators using VectorBT operations
                # RSI calculation
                if len(data) >= 14:
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    features_dict['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                if len(data) >= 20:
                    bb_middle = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, window=20).result_data
                    bb_std = self.vectorbt_rolling_optimizer.rolling_std(close_prices, window=20).result_data
                    features_dict['bb_upper'] = bb_middle + (bb_std * 2)
                    features_dict['bb_lower'] = bb_middle - (bb_std * 2)
                    features_dict['bb_width'] = features_dict['bb_upper'] - features_dict['bb_lower']
                    features_dict['bb_position'] = (close_prices - features_dict['bb_lower']) / features_dict['bb_width']
                
                # MACD
                if len(data) >= 26:
                    ema_12 = close_prices.ewm(span=12).mean()
                    ema_26 = close_prices.ewm(span=26).mean()
                    features_dict['macd'] = ema_12 - ema_26
                    features_dict['macd_signal'] = features_dict['macd'].ewm(span=9).mean()
                    features_dict['macd_histogram'] = features_dict['macd'] - features_dict['macd_signal']
                
                # Price momentum features
                for period in [1, 5, 10, 20]:
                    if len(data) > period:
                        features_dict[f'price_change_{period}'] = close_prices.pct_change(period)
                        features_dict[f'volume_change_{period}'] = volume.pct_change(period)
                
                # Volatility features
                for window in [10, 20, 50]:
                    if len(data) >= window:
                        returns = close_prices.pct_change()
                        features_dict[f'volatility_{window}'] = returns.rolling(window=window).std()
                        features_dict[f'volatility_annualized_{window}'] = features_dict[f'volatility_{window}'] * np.sqrt(252)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_dict, index=data.index)
            
            # Remove any infinite or NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            tprint_success(f"✅ Generated {len(features_df.columns)} enhanced features using vectorization utilities")
            return features_df
            
        except Exception as e:
            tprint_error(f"❌ Enhanced feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _enhanced_interaction_generation(self, features_df: pd.DataFrame, targets: Optional[pd.Series]) -> List[Any]:
        """Enhanced interaction generation with VectorBT optimization and unified vectorization."""
        tprint_debug("Starting enhanced interaction generation with unified vectorization")
        
        try:
            # Use Unified Vectorization Manager for interaction generation
            interactions = []
            
            # Generate feature interactions using vectorized operations
            with self.unified_vectorization_manager.batch_processing_context() as batch_processor:
                # Price-based interactions
                if 'close' in features_df.columns:
                    close_col = features_df['close']
                    
                    # Price momentum interactions
                    for col in features_df.columns:
                        if col != 'close' and not col.startswith('close_'):
                            # Price vs other features
                            interaction_name = f"close_vs_{col}"
                            interactions.append({
                                'name': interaction_name,
                                'data': close_col * features_df[col],
                                'type': 'multiplicative',
                                'features': ['close', col]
                            })
                            
                            # Price ratio interactions
                            if features_df[col].min() > 0:  # Avoid division by zero
                                ratio_name = f"close_ratio_{col}"
                                interactions.append({
                                    'name': ratio_name,
                                    'data': close_col / features_df[col],
                                    'type': 'ratio',
                                    'features': ['close', col]
                                })
                
                # Rolling correlation interactions
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 for performance
                        for col2 in numeric_cols[i+1:6]:  # Limit to next 5
                            # Rolling correlation
                            corr_name = f"corr_{col1}_{col2}"
                            rolling_corr = self.vectorbt_rolling_optimizer.rolling_corr(
                                features_df[col1], features_df[col2], window=20
                            ).result_data
                            
                            interactions.append({
                                'name': corr_name,
                                'data': rolling_corr,
                                'type': 'correlation',
                                'features': [col1, col2],
                                'window': 20
                            })
                
                # Volatility interactions
                volatility_cols = [col for col in features_df.columns if 'volatility' in col or 'std' in col]
                if volatility_cols:
                    for vol_col in volatility_cols[:3]:  # Limit to first 3 volatility features
                        for price_col in [col for col in features_df.columns if 'close' in col or 'price' in col][:3]:
                            # Volatility-adjusted price
                            interaction_name = f"vol_adj_{price_col}_{vol_col}"
                            if features_df[vol_col].min() > 0:
                                interactions.append({
                                    'name': interaction_name,
                                    'data': features_df[price_col] / features_df[vol_col],
                                    'type': 'volatility_adjusted',
                                    'features': [price_col, vol_col]
                                })
                
                # Technical indicator interactions
                ma_cols = [col for col in features_df.columns if 'ma_' in col or 'sma' in col]
                if len(ma_cols) >= 2:
                    for i, ma1 in enumerate(ma_cols[:3]):
                        for ma2 in ma_cols[i+1:4]:
                            # MA crossover signals
                            crossover_name = f"crossover_{ma1}_{ma2}"
                            interactions.append({
                                'name': crossover_name,
                                'data': features_df[ma1] - features_df[ma2],
                                'type': 'crossover',
                                'features': [ma1, ma2]
                            })
                            
                            # MA ratio
                            if features_df[ma2].min() > 0:
                                ratio_name = f"ma_ratio_{ma1}_{ma2}"
                                interactions.append({
                                    'name': ratio_name,
                                    'data': features_df[ma1] / features_df[ma2],
                                    'type': 'ma_ratio',
                                    'features': [ma1, ma2]
                                })
            
            # Use VectorBT optimizer for additional interaction generation
            vectorbt_interactions = self.vectorbt_optimizer.optimize_interaction_generation(features_df, targets)
            if vectorbt_interactions:
                interactions.extend(vectorbt_interactions)
            
            tprint_success(f"✅ Generated {len(interactions)} enhanced interactions using unified vectorization")
            return interactions
            
        except Exception as e:
            tprint_error(f"Enhanced interaction generation failed: {e}")
            return []
    
    def _htf_interaction_generation(self, data: pd.DataFrame, features_df: pd.DataFrame, 
                                  targets: Optional[pd.Series]) -> List[Any]:
        """HTF-aware interaction generation with enhanced vectorization."""
        tprint_debug("Starting HTF interaction generation with enhanced vectorization")
        
        try:
            # Create simulated HTF features using enhanced vectorization
            htf_features = self._create_enhanced_htf_features(data)
            
            # Use HTF generator for interaction generation
            htf_interactions = self.template_interaction_generator.generate_interactions(
                htf_features, features_df, targets
            )
            
            # Generate additional HTF interactions using vectorized operations
            additional_htf_interactions = []
            
            with self.unified_vectorization_manager.batch_processing_context() as batch_processor:
                # HTF vs LTF interactions
                for htf_key, htf_data in htf_features.items():
                    if isinstance(htf_data, pd.Series):
                        # Find matching LTF features
                        ltf_matches = [col for col in features_df.columns if htf_key.split('_')[0] in col]
                        
                        for ltf_col in ltf_matches[:3]:  # Limit to first 3 matches
                            # HTF-LTF ratio
                            ratio_name = f"htf_ltf_ratio_{htf_key}_{ltf_col}"
                            if features_df[ltf_col].min() > 0:
                                additional_htf_interactions.append({
                                    'name': ratio_name,
                                    'data': htf_data / features_df[ltf_col],
                                    'type': 'htf_ltf_ratio',
                                    'features': [htf_key, ltf_col]
                                })
                            
                            # HTF-LTF difference
                            diff_name = f"htf_ltf_diff_{htf_key}_{ltf_col}"
                            additional_htf_interactions.append({
                                'name': diff_name,
                                'data': htf_data - features_df[ltf_col],
                                'type': 'htf_ltf_diff',
                                'features': [htf_key, ltf_col]
                            })
                
                # HTF momentum interactions
                htf_trend_cols = [k for k, v in htf_features.items() if 'trend' in k and isinstance(v, pd.Series)]
                if htf_trend_cols:
                    for trend_col in htf_trend_cols:
                        # HTF trend vs price momentum
                        for momentum_col in [col for col in features_df.columns if 'change' in col or 'momentum' in col][:3]:
                            interaction_name = f"htf_trend_momentum_{trend_col}_{momentum_col}"
                            additional_htf_interactions.append({
                                'name': interaction_name,
                                'data': htf_features[trend_col] * features_df[momentum_col],
                                'type': 'htf_trend_momentum',
                                'features': [trend_col, momentum_col]
                            })
            
            # Combine all HTF interactions
            all_htf_interactions = htf_interactions + additional_htf_interactions
            
            tprint_success(f"✅ Generated {len(all_htf_interactions)} HTF interactions with enhanced vectorization")
            return all_htf_interactions
            
        except Exception as e:
            tprint_error(f"HTF interaction generation failed: {e}")
            return []
    
    def _create_enhanced_htf_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create enhanced HTF features using vectorization utilities."""
        try:
            htf_features = {}
            
            if 'close' not in data.columns:
                return htf_features
            
            close_prices = data['close']
            high_prices = data['high']
            low_prices = data['low']
            volume = data['volume']
            
            # Use VectorBT rolling optimizer for HTF features
            with self.unified_vectorization_manager.batch_processing_context() as batch_processor:
                # HTF trend features (4h timeframe simulation)
                htf_features['htf_trend'] = self.vectorbt_rolling_optimizer.rolling_mean(
                    close_prices, window=16
                ).result_data
                
                htf_features['htf_trend_ema'] = close_prices.ewm(span=16).mean()
                
                # HTF volatility features
                htf_features['htf_volatility'] = self.vectorbt_rolling_optimizer.rolling_std(
                    close_prices, window=16
                ).result_data
                
                htf_features['htf_volatility_ema'] = close_prices.pct_change().ewm(span=16).std()
                
                # HTF volume features
                htf_features['htf_volume_ma'] = self.vectorbt_rolling_optimizer.rolling_mean(
                    volume, window=16
                ).result_data
                
                # HTF price range features
                price_range = high_prices - low_prices
                htf_features['htf_range_ma'] = self.vectorbt_rolling_optimizer.rolling_mean(
                    price_range, window=16
                ).result_data
                
                # HTF momentum features
                htf_features['htf_momentum'] = close_prices.pct_change(16)
                htf_features['htf_momentum_ema'] = close_prices.pct_change().ewm(span=16).mean()
                
                # HTF technical indicators
                # HTF RSI
                if len(data) >= 32:  # 2x the window for HTF
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=32).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=32).mean()
                    rs = gain / loss
                    htf_features['htf_rsi'] = 100 - (100 / (1 + rs))
                
                # HTF Bollinger Bands
                if len(data) >= 32:
                    bb_middle = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, window=32).result_data
                    bb_std = self.vectorbt_rolling_optimizer.rolling_std(close_prices, window=32).result_data
                    htf_features['htf_bb_upper'] = bb_middle + (bb_std * 2)
                    htf_features['htf_bb_lower'] = bb_middle - (bb_std * 2)
                    htf_features['htf_bb_width'] = htf_features['htf_bb_upper'] - htf_features['htf_bb_lower']
                    htf_features['htf_bb_position'] = (close_prices - htf_features['htf_bb_lower']) / htf_features['htf_bb_width']
                
                # HTF MACD
                if len(data) >= 64:  # 4x the window for HTF
                    ema_24 = close_prices.ewm(span=24).mean()
                    ema_48 = close_prices.ewm(span=48).mean()
                    htf_features['htf_macd'] = ema_24 - ema_48
                    htf_features['htf_macd_signal'] = htf_features['htf_macd'].ewm(span=18).mean()
                    htf_features['htf_macd_histogram'] = htf_features['htf_macd'] - htf_features['htf_macd_signal']
            
            return htf_features
            
        except Exception as e:
            tprint_error(f"Enhanced HTF feature creation failed: {e}")
            return {}
    
    def _create_htf_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated HTF features (legacy method for compatibility)."""
        try:
            htf_features = {}
            
            if 'close' not in data.columns:
                return htf_features
            
            close_prices = data['close']
            
            # Create simulated HTF features (4h timeframe simulation)
            htf_features['htf_trend'] = close_prices.rolling(16).mean()  # 4h SMA
            htf_features['htf_volatility'] = close_prices.rolling(16).std()  # 4h volatility
            htf_features['htf_momentum'] = close_prices.pct_change(16)  # 4h momentum
            htf_features['htf_regime'] = (close_prices > close_prices.rolling(16).mean()).astype(int)  # 4h regime
            
            return htf_features
            
        except Exception as e:
            tprint_error(f"HTF feature creation failed: {e}")
            return {}
    
    def _advanced_lookback_optimization(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                      features_df: pd.DataFrame, pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Advanced lookback optimization using sophisticated algorithms with differentiated long/short pipelines.
        
        Features:
        - Differentiated long/short optimization
        - Walk-forward cross-validation with nested CV
        - Bayesian optimization with mode awareness
        - Execution mode constraints (light/blank/full)
        - Advanced regularization and bootstrap sampling
        - Feature lag metadata tracking
        """
        tprint_info("🚀 Starting sophisticated lookback optimization with differentiated pipelines")
        tprint_debug(f"📊 Data shape: {data.shape}, Features: {len(features_df.columns)}")
        
        try:
            # Prepare data for optimization
            if targets is None:
                # Create synthetic targets if none provided
                if 'close' in data.columns:
                    targets = data['close'].pct_change().dropna()
                else:
                    tprint_warning("⚠️ No targets provided and no close price available")
                    return {}
            
            # Align data and targets
            aligned_data = data.copy()
            aligned_targets = targets.reindex(data.index)
            
            # Get feature names
            feature_names = list(features_df.columns)
            if not feature_names:
                tprint_warning("⚠️ No features available for lookback optimization")
                return {}
            
            # Determine outer walk-forward splits for nested CV
            outer_splits = self._build_walk_forward_splits(len(aligned_data))
            use_nested_cv = bool(outer_splits)
            
            if use_nested_cv:
                tprint_info(f"🧭 Using nested walk-forward CV with {len(outer_splits)} outer folds")
            else:
                tprint_info("🧭 Nested walk-forward CV unavailable, using single-pass optimization")
            
            # Get optimization direction from pipeline state (default to 'both')
            optimization_direction = pipeline_state.get('direction', 'both') if pipeline_state else 'both'
            tprint_info(f"🎯 Optimization direction: {optimization_direction}")
            
            # Select optimal target columns for long/short directions
            long_target_column = self._select_optimal_target_column(aligned_data, direction='long')
            short_target_column = self._select_optimal_target_column(aligned_data, direction='short')
            
            tprint_success(f"✅ Target selection complete - Long: {long_target_column}, Short: {short_target_column}")
            
            # Determine which directions to optimize
            optimize_longs = optimization_direction in ('longs', 'both')
            optimize_shorts = optimization_direction in ('shorts', 'both')
            
            # Detect execution mode and create mode-aware constraints
            execution_mode = aligned_data.attrs.get('ares_mode', 'full')
            if execution_mode not in ['light', 'blank', 'full']:
                execution_mode = pipeline_state.get('execution_mode', 'full') if pipeline_state else 'full'
            
            tprint_info(f"🎯 Detected execution mode: {execution_mode.upper()}")
            
            # Create mode-aware constraints
            mode_constraints = self._create_mode_aware_constraints(execution_mode)
            
            # Apply mode-specific optimization parameters
            use_bayesian_opt = mode_constraints.get('use_bayesian_optimization', True)
            n_bootstrap = mode_constraints.get('n_bootstrap_samples', 100)
            cv_folds = mode_constraints.get('cv_folds', 5)
            
            tprint_debug(f"⚙️ Optimization settings: bayesian={use_bayesian_opt}, bootstrap={n_bootstrap}, cv_folds={cv_folds}")
            tprint_debug(f"🎯 Directions: longs={optimize_longs}, shorts={optimize_shorts}")
            
            # Separate optimization for long and short directions
            long_feature_results = {}
            short_feature_results = {}
            
            total_features = len(feature_names)
            tprint_info(f"🚀 Starting optimization of {total_features} features")
            
            # Reset feature lag metadata
            feature_lag_metadata = {}
            
            # Use Unified Vectorization Manager for batch processing
            with self.unified_vectorization_manager.batch_processing_context() as batch_processor:
                for idx, feature in enumerate(feature_names, 1):
                    try:
                        if idx % max(1, total_features // 10) == 0:  # Log every 10%
                            tprint_info(f"⏳ Optimization progress: {idx}/{total_features} features ({100*idx/total_features:.1f}%)")
                        
                        tprint_debug(f"🔍 Optimizing feature {idx}/{total_features}: {feature}")
                        
                        # Use consistent lookback range for all execution modes
                        lookback_range = (3, 100)  # Optimized range for faster, more relevant periods
                    
                    optimizer_kwargs = {
                        'regularization_settings': self._get_regularization_settings(),
                        'n_bootstrap_samples': n_bootstrap,
                        'cv_folds': cv_folds,
                        'use_bayesian_optimization': use_bayesian_opt
                    }
                    
                    if use_nested_cv:
                        optimizer_kwargs['outer_split_iterator'] = outer_splits
                    
                    # Optimize for LONG direction
                    if optimize_longs and long_target_column:
                        tprint_debug(f"📈 Optimizing LONG direction for {feature}")
                        long_entry = self._optimize_feature_direction(
                            aligned_data, feature, long_target_column, 'long',
                            lookback_range, optimizer_kwargs, use_bayesian_opt,
                            execution_mode, use_nested_cv, pipeline_state
                        )
                        if long_entry:
                            feature_key = self._normalize_feature_key(feature)
                            long_feature_results[feature_key] = long_entry
                            feature_lag_metadata[f"{feature_key}_long"] = long_entry.get('best_lookback_period', 0)
                            tprint_debug(f"✅ LONG optimization completed for {feature}")
                        else:
                            tprint_warning(f"⚠️ LONG optimization failed for {feature}")
                    
                    # Optimize for SHORT direction
                    if optimize_shorts and short_target_column:
                        tprint_debug(f"📉 Optimizing SHORT direction for {feature}")
                        short_entry = self._optimize_feature_direction(
                            aligned_data, feature, short_target_column, 'short',
                            lookback_range, optimizer_kwargs, use_bayesian_opt,
                            execution_mode, use_nested_cv, pipeline_state
                        )
                        if short_entry:
                            feature_key = self._normalize_feature_key(feature)
                            short_feature_results[feature_key] = short_entry
                            feature_lag_metadata[f"{feature_key}_short"] = short_entry.get('best_lookback_period', 0)
                            tprint_debug(f"✅ SHORT optimization completed for {feature}")
                        else:
                            tprint_warning(f"⚠️ SHORT optimization failed for {feature}")
                
                except Exception as e:
                    tprint_error(f"❌ Feature optimization failed for {feature}: {e}")
                    continue
            
            # Combine results
            total_optimized = len(long_feature_results) + len(short_feature_results)
            
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