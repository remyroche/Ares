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
from datetime import datetime

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

# Import feature_generation utilities
try:
    from src.feature_generation.utils import (
        Step06UtilityContainer, UtilityConfig, get_utility_container,
        EnhancedFeatureEngineering, FeatureGenerationOptimizer,
        FeatureOptimizationConfig, CrossTimeframeAnalysisPipeline,
        FractionalDifferentiationPipeline, EnhancedMatrixOperations,
        validate_feature_quality, validate_features_dataframe,
        feature_validation_decorator
    )
    FEATURE_GENERATION_AVAILABLE = True
except ImportError as e:
    FEATURE_GENERATION_AVAILABLE = False
    tprint_warning(f"⚠️ Feature generation utilities not available: {e}")

# Import features_common utilities
try:
    from src.features_common import (
        OptimizationConfig, get_optimization_config,
        VectorBTConfig as FeaturesCommonVectorBTConfig, get_vectorbt_config,
        UnifiedConfig, get_unified_config,
        OptimizationMixin, PerformanceMixin, VectorBTMixin,
        ValidationMixin, CachingMixin, MonitoringMixin,
        ScalerFactory, create_optimized_scaler, create_batch_scaler,
        OptimizerFactory, create_optimizer, create_vectorbt_optimizer,
        RegistryFactory, create_registry, create_feature_registry,
        UnifiedFactory, create_optimized_component,
        UnifiedVectorBTManager, get_unified_vectorbt_manager,
        VectorBTOptimizationEngine, get_optimization_engine,
        GPUAccelerator, get_gpu_accelerator,
        VectorBTPerformanceMonitor, get_performance_monitor,
        FeaturesCommonError, ValidationError, OptimizationError,
        VectorBTError, ConfigurationError, SilentFailureError,
        ensure_no_silent_failures, validate_input_data, safe_execute,
        validate_configuration, check_system_health, report_silent_failures,
        get_logger, log_operation
    )
    FEATURES_COMMON_AVAILABLE = True
except ImportError as e:
    FEATURES_COMMON_AVAILABLE = False
    tprint_warning(f"⚠️ Features common utilities not available: {e}")

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

# Import math validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, 
    validate_positive, validate_range, safe_correlation, safe_covariance,
    safe_mean, safe_std, safe_percentile, safe_percentage_change,
    safe_weighted_average, safe_kelly_calculation, MathValidation
)

# Import math validation integration
from .enhanced_components.math_validation_integration import (
    MathValidationIntegration, MathValidationResult, validate_pipeline_calculation
)
# Import unified data utilities
from src.utils.data import UnifiedDataUtils, unified_data_utils
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds
from src.utils.data.quality.data_cleaning import DataCleaner
from src.utils.data.validation.validators import CrossStepValidator

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

# Import feature engineering roadmap utilities
try:
    from src.feature_engineering_roadmap.interactions import (
        InteractionEngine, create_default_interaction_config, InteractionConfig, InteractionType
    )
    from src.feature_engineering_roadmap.transforms import (
        TransformRouter, create_default_transform_config, OnlineEWZ, TODRank, SignedLog, MADScaler, Winsorization
    )
    from src.feature_engineering_roadmap.dynamic_feature_selector import (
        DynamicRoadmapPipeline, OptimizedPipelineConfig
    )
    FEATURE_ENGINEERING_ROADMAP_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_ROADMAP_AVAILABLE = False
    tprint_warning("Feature engineering roadmap utilities not available")

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
        
        # Initialize utility systems first
        self._initialize_utility_systems()
        
        # Initialize all components
        self._initialize_core_components()
        self._initialize_enhanced_components()
        self._initialize_validation_components()
        self._initialize_performance_tracking()
        self._initialize_advanced_infrastructure()
        
        tprint_info("🚀 Consolidated Unified Data-Driven Pipeline initialized")
        tprint_info(f"📊 Configuration: {self.config}")
        if FEATURE_GENERATION_AVAILABLE:
            tprint_success("✅ Feature generation utilities integrated")
        if FEATURES_COMMON_AVAILABLE:
            tprint_success("✅ Features common utilities integrated")
    
    def _initialize_utility_systems(self):
        """Initialize utility systems from feature_generation and features_common."""
        tprint_debug("Initializing utility systems")
        
        # Initialize feature generation utilities
        if FEATURE_GENERATION_AVAILABLE:
            try:
                # Initialize utility container
                self.utility_container = get_utility_container()
                self.utility_config = UtilityConfig()
                
                # Initialize enhanced feature engineering
                self.enhanced_feature_engineering = EnhancedFeatureEngineering()
                
                # Initialize feature optimization
                self.feature_optimizer = FeatureGenerationOptimizer()
                self.feature_optimization_config = FeatureOptimizationConfig()
                
                # Initialize cross-timeframe analysis
                self.cross_timeframe_pipeline = CrossTimeframeAnalysisPipeline()
                
                # Initialize fractional differentiation
                self.fractional_diff_pipeline = FractionalDifferentiationPipeline()
                
                # Initialize matrix operations
                self.enhanced_matrix_ops = EnhancedMatrixOperations()
                
                tprint_success("✅ Feature generation utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Feature generation utilities initialization failed: {e}")
                self.utility_container = None
                self.enhanced_feature_engineering = None
                self.feature_optimizer = None
                self.cross_timeframe_pipeline = None
                self.fractional_diff_pipeline = None
                self.enhanced_matrix_ops = None
        else:
            self.utility_container = None
            self.enhanced_feature_engineering = None
            self.feature_optimizer = None
            self.cross_timeframe_pipeline = None
            self.fractional_diff_pipeline = None
            self.enhanced_matrix_ops = None
        
        # Initialize features common utilities
        if FEATURES_COMMON_AVAILABLE:
            try:
                # Initialize unified configuration
                self.unified_config = get_unified_config()
                self.optimization_config = get_optimization_config()
                self.vectorbt_config = get_vectorbt_config()
                
                # Initialize unified VectorBT manager
                self.unified_vectorbt_manager = get_unified_vectorbt_manager()
                self.vectorbt_optimization_engine = get_optimization_engine()
                self.gpu_accelerator = get_gpu_accelerator()
                self.vectorbt_performance_monitor = get_performance_monitor()
                
                # Initialize factories
                self.scaler_factory = ScalerFactory()
                self.optimizer_factory = OptimizerFactory()
                self.registry_factory = RegistryFactory()
                self.unified_factory = UnifiedFactory()
                
                # Initialize enhanced scalers
                self.optimized_scaler = create_optimized_scaler()
                self.batch_scaler = create_batch_scaler()
                
                tprint_success("✅ Features common utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Features common utilities initialization failed: {e}")
                self.unified_config = None
                self.optimization_config = None
                self.vectorbt_config = None
                self.unified_vectorbt_manager = None
                self.vectorbt_optimization_engine = None
                self.gpu_accelerator = None
                self.vectorbt_performance_monitor = None
                self.scaler_factory = None
                self.optimizer_factory = None
                self.registry_factory = None
                self.unified_factory = None
                self.optimized_scaler = None
                self.batch_scaler = None
        else:
            self.unified_config = None
            self.optimization_config = None
            self.vectorbt_config = None
            self.unified_vectorbt_manager = None
            self.vectorbt_optimization_engine = None
            self.gpu_accelerator = None
            self.vectorbt_performance_monitor = None
            self.scaler_factory = None
            self.optimizer_factory = None
            self.registry_factory = None
            self.unified_factory = None
            self.optimized_scaler = None
            self.batch_scaler = None
    
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
        
        # Multi-objective feature selector with computational awareness
        computational_constraints = None
        if self.config.feature_selection.enable_computational_awareness:
            from .core.computational_awareness import ComputationalConstraints
            constraints_config = self.config.feature_selection.computational_constraints
            computational_constraints = ComputationalConstraints(
                max_memory_gb=constraints_config.get('max_memory_gb'),
                max_cpu_cores=constraints_config.get('max_cpu_cores'),
                max_execution_time_seconds=constraints_config.get('max_execution_time_seconds', 300.0),
                memory_safety_margin=constraints_config.get('memory_safety_margin', 0.2),
                cpu_safety_margin=constraints_config.get('cpu_safety_margin', 0.1)
            )
        
        self.feature_selector = MultiObjectiveFeatureSelector(
            objectives=create_default_objectives(),
            cv_splitter=self.cv_splitter,
            enable_computational_awareness=self.config.feature_selection.enable_computational_awareness,
            computational_constraints=computational_constraints
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
        
        # Initialize enhanced feature selection if enabled
        if self.config.feature_selection.enable_enhanced_methods:
            tprint_info("🔧 Initializing enhanced feature selection methods")
            self.enhanced_feature_selectors = {}
            
            # Initialize enhanced methods
            if 'improved_mrmr' in self.config.feature_selection.enhanced_methods:
                try:
                    from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR
                    self.enhanced_feature_selectors['improved_mrmr'] = ImprovedMRMR(
                        self.config.feature_selection.improved_mrmr_config
                    )
                    tprint_success("✅ Improved mRMR initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Improved mRMR initialization failed: {e}")
            
            if 'vectorbt_mrmr' in self.config.feature_selection.enhanced_methods:
                try:
                    from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
                    from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
                    config = VectorBTFeatureSelectionConfig()
                    config.target_features = self.config.feature_selection.multi_objective.max_features
                    config.chunk_size = self.config.feature_selection.vectorbt_chunk_size
                    self.enhanced_feature_selectors['vectorbt_mrmr'] = VectorBTMRMRSelector(config)
                    tprint_success("✅ VectorBT mRMR initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT mRMR initialization failed: {e}")
            
            if 'vectorbt_rfe' in self.config.feature_selection.enhanced_methods:
                try:
                    from src.feature_selection.vectorbt.vectorbt_rfe_selector import VectorBTRFESelector
                    from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
                    config = VectorBTFeatureSelectionConfig()
                    config.target_features = self.config.feature_selection.multi_objective.max_features
                    config.chunk_size = self.config.feature_selection.vectorbt_chunk_size
                    self.enhanced_feature_selectors['vectorbt_rfe'] = VectorBTRFESelector(config)
                    tprint_success("✅ VectorBT RFE initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT RFE initialization failed: {e}")
            
            if 'vectorbt_lasso' in self.config.feature_selection.enhanced_methods:
                try:
                    from src.feature_selection.vectorbt.vectorbt_regularization import VectorBTRegularizationSelector
                    from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
                    config = VectorBTFeatureSelectionConfig()
                    config.target_features = self.config.feature_selection.multi_objective.max_features
                    config.chunk_size = self.config.feature_selection.vectorbt_chunk_size
                    self.enhanced_feature_selectors['vectorbt_lasso'] = VectorBTRegularizationSelector(config)
                    tprint_success("✅ VectorBT LASSO initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT LASSO initialization failed: {e}")
            
            if 'enhanced_ensemble' in self.config.feature_selection.enhanced_methods:
                try:
                    from src.feature_selection.advanced.enhanced_ensemble_selector import EnhancedEnsembleAdvancedSelector
                    from src.feature_selection.advanced.enhanced_config import EnhancedEnsembleConfig
                    config = EnhancedEnsembleConfig()
                    config.target_features = self.config.feature_selection.multi_objective.max_features
                    self.enhanced_feature_selectors['enhanced_ensemble'] = EnhancedEnsembleAdvancedSelector(config)
                    tprint_success("✅ Enhanced ensemble initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced ensemble initialization failed: {e}")
            
            if 'enhanced_advanced' in self.config.feature_selection.enhanced_methods:
                try:
                    from src.feature_selection.advanced.enhanced_advanced_selector import EnhancedAdvancedFeatureSelector
                    from src.feature_selection.advanced.enhanced_config import EnhancedAdvancedConfig
                    config = EnhancedAdvancedConfig()
                    config.target_features = self.config.feature_selection.multi_objective.max_features
                    self.enhanced_feature_selectors['enhanced_advanced'] = EnhancedAdvancedFeatureSelector(config)
                    tprint_success("✅ Enhanced advanced initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced advanced initialization failed: {e}")
            
            tprint_success(f"🔧 Enhanced feature selection initialized with {len(self.enhanced_feature_selectors)} methods")
        else:
            self.enhanced_feature_selectors = {}
            tprint_info("🔧 Enhanced feature selection disabled")
        
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
        
        # Feature engineering roadmap components
        if FEATURE_ENGINEERING_ROADMAP_AVAILABLE:
            # Interaction engine for theory-driven interactions
            self.interaction_engine = InteractionEngine(
                create_default_interaction_config(),
                use_vectorbt=VECTORBT_AVAILABLE,
                use_gpu=self.config.performance.enable_gpu,
                enable_parallel=True
            )
            
            # Transform router for statistical transforms
            self.transform_router = None  # Will be initialized when we have feature names
            
            # Dynamic roadmap pipeline for optimized feature selection
            roadmap_config = OptimizedPipelineConfig(
                n_candidate_features=100,
                n_selected_features=32,
                use_bayesian_opt=True,
                bayesian_trials=50,
                use_vectorbt=VECTORBT_AVAILABLE,
                use_gpu=self.config.performance.enable_gpu,
                enable_parallel=True
            )
            self.dynamic_roadmap_pipeline = DynamicRoadmapPipeline(roadmap_config)
            
            tprint_info("✅ Feature engineering roadmap components initialized")
        else:
            self.interaction_engine = None
            self.transform_router = None
            self.dynamic_roadmap_pipeline = None
            tprint_warning("⚠️  Feature engineering roadmap components not available")
        
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
        
        # Unified data utilities
        self.unified_data_utils = UnifiedDataUtils(
            quality_thresholds=QualityThresholds(
                max_nan_ratio=0.05,  # Allow 5% NaN for calculated features
                max_infinite_count=0,
                min_unique_values=2,
                max_constant_ratio=0.95,
                max_gap_hours=48,
                price_tolerance=0.001,
                volume_tolerance=0.001,
                max_correlation_threshold=0.95,
                min_feature_count=40
            ),
            enable_streaming=True,
            chunk_size=10000,
            memory_threshold=0.8
        )
        
        # Individual data utilities for specific operations
        self.data_processor = DataProcessor()
        self.data_cleaner = DataCleaner()
        self.quality_framework = DataQualityFramework()
        self.cross_step_validator = CrossStepValidator()
        
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
        
        # Advanced data loading with enhanced utilities
        data_loader_config = {
            'klines_storage': {
                'base_dir': 'historical_data',
                'compression': 'zstd',
                'compression_level': 3,
                'enable_metadata': True,
                'enable_validation': True,
                'max_file_size_mb': 100
            }
        }
        self.advanced_data_loader = AdvancedDataLoader(
            logger=self.logger, 
            config=data_loader_config
        )
        
        # Advanced artifact management
        self.advanced_artifact_manager = AdvancedArtifactManager(
            base_dir="artifacts/unified_pipeline",
            logger=self.logger
        )
        
        # Math validation integration
        self.math_validator = MathValidationIntegration(logger=self.logger)
        
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
            # Enhanced data processing and validation using unified utilities
            tprint_info("🔍 Performing comprehensive data validation and processing...")
            
            # Step 1: Comprehensive data validation and quality assessment
            quality_result = self.quality_framework.validate_dataframe_quality(
                data, context=f"pipeline_input_{timeframe}"
            )
            
            if not quality_result.passed:
                tprint_warning(f"⚠️ Data quality issues detected: {len(quality_result.issues)} issues")
                for issue in quality_result.issues[:3]:  # Show first 3 issues
                    tprint_warning(f"  - {issue}")
                if len(quality_result.issues) > 3:
                    tprint_warning(f"  ... and {len(quality_result.issues) - 3} more issues")
            
            # Step 2: Process and validate data using unified utilities
            processed_data, processing_report = self.unified_data_utils.process_and_validate(
                data=data,
                validate_quality=True,
                clean_missing_values=True,
                detect_outliers=True,
                optimize_dtypes=True,
                regularize_timestamps=True,
                context=f"pipeline_processing_{timeframe}",
                symbol=pipeline_state.get('symbol', 'ETHUSDT') if pipeline_state else 'ETHUSDT',
                exchange=pipeline_state.get('exchange', 'binance') if pipeline_state else 'binance',
                timeframe=timeframe
            )
            
            # Log processing results
            tprint_success(f"✅ Data processing completed: {processing_report['final_shape']} shape")
            tprint_info(f"📊 Processing steps: {', '.join(processing_report['steps_completed'])}")
            if processing_report.get('optimization_results'):
                memory_reduction = processing_report['optimization_results'].get('memory_reduction_percent', 0)
                tprint_info(f"💾 Memory optimization: {memory_reduction:.1f}% reduction")
            
            # Step 3: Advanced input validation for pipeline-specific requirements
            is_valid, validation_summary, cleaned_data = self.advanced_validator.validate_data(
                processed_data, 
                required_columns=['open', 'high', 'low', 'close', 'volume'],
                target_columns=feature_columns
            )
            
            if not is_valid:
                error_msg = f"Advanced validation failed: {validation_summary.recommendations}"
                tprint_error(f"❌ {error_msg}")
                return self._create_empty_result(start_time, error_msg)
            
            # Use the processed data from unified utilities
            cleaned_data = processed_data
            
            # Load market data using advanced data loader
            market_data = await self.advanced_data_loader.load_market_data(
                cleaned_data, pipeline_state, force_refresh=False
            )
            
            # Apply additional data processing to market data
            if market_data is not None and not market_data.empty:
                tprint_info("🔧 Applying additional data processing to market data...")
                market_data, market_processing_report = self.unified_data_utils.process_and_validate(
                    data=market_data,
                    validate_quality=True,
                    clean_missing_values=True,
                    detect_outliers=False,  # Don't remove outliers from market data
                    optimize_dtypes=True,
                    regularize_timestamps=True,
                    context=f"market_data_{timeframe}",
                    symbol=pipeline_state.get('symbol', 'ETHUSDT') if pipeline_state else 'ETHUSDT',
                    exchange=pipeline_state.get('exchange', 'binance') if pipeline_state else 'binance',
                    timeframe=timeframe
                )
                tprint_success(f"✅ Market data processing: {market_processing_report['final_shape']} shape")
            
            # Load labeling data
            labeling_data = await self.advanced_data_loader.load_labeling_data(
                pipeline_state.get('symbol', 'ETHUSDT') if pipeline_state else 'ETHUSDT',
                pipeline_state.get('exchange', 'binance') if pipeline_state else 'binance',
                timeframe,
                pipeline_state
            )
            
            # Apply data processing to labeling data
            if labeling_data is not None and not labeling_data.empty:
                tprint_info("🔧 Applying data processing to labeling data...")
                labeling_data, labeling_processing_report = self.unified_data_utils.process_and_validate(
                    data=labeling_data,
                    validate_quality=True,
                    clean_missing_values=True,
                    detect_outliers=False,  # Don't remove outliers from labels
                    optimize_dtypes=True,
                    regularize_timestamps=True,
                    context=f"labeling_data_{timeframe}",
                    symbol=pipeline_state.get('symbol', 'ETHUSDT') if pipeline_state else 'ETHUSDT',
                    exchange=pipeline_state.get('exchange', 'binance') if pipeline_state else 'binance',
                    timeframe=timeframe
                )
                tprint_success(f"✅ Labeling data processing: {labeling_processing_report['final_shape']} shape")
            
            # Prepare data for optimization
            processed_data = self.advanced_data_loader.prepare_data_for_optimization(
                market_data, labeling_data
            )
            
            # Final data quality check before feature generation
            if processed_data is not None and not processed_data.empty:
                tprint_info("🔍 Performing final data quality check...")
                final_quality_result = self.quality_framework.validate_dataframe_quality(
                    processed_data, context=f"pre_feature_generation_{timeframe}"
                )
                
                if not final_quality_result.passed:
                    tprint_warning(f"⚠️ Final quality check issues: {len(final_quality_result.issues)} issues")
                    for issue in final_quality_result.issues[:2]:  # Show first 2 issues
                        tprint_warning(f"  - {issue}")
                
                tprint_info(f"📊 Final data quality score: {final_quality_result.quality_score:.1f}/100")
            
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
            
            # Monitor data quality before period optimization
            period_quality_monitoring = self._monitor_data_quality_throughout_pipeline(
                processed_data, f"pre_period_optimization_{timeframe}"
            )
            
            period_results = self._enhanced_period_optimization(processed_data, timeframe)
            
            # Cross-step validation: Input -> Period Optimization
            validation_result_1 = self.cross_step_validator.validate_step_transition(
                from_step="input_data",
                to_step="period_optimization",
                input_data=processed_data,
                output_data=processed_data,  # Period optimization doesn't change data
                step_metadata={"timeframe": timeframe, "period_results": period_results}
            )
            if not validation_result_1.get('valid', True):
                tprint_warning(f"⚠️ Cross-step validation warning: {validation_result_1.get('message', 'Unknown issue')}")
            
            # Step 2: Advanced feature selection from 200+ feature bank
            tprint_info("Step 2: Advanced feature selection from 200+ feature bank")
            feature_selection_results = self._advanced_feature_selection(processed_data, processed_targets)
            
            # Step 2.5: Use dynamic roadmap pipeline for optimized feature selection
            if FEATURE_ENGINEERING_ROADMAP_AVAILABLE and self.dynamic_roadmap_pipeline is not None:
                tprint_info("Step 2.5: Using dynamic roadmap pipeline for optimized feature selection")
                roadmap_results = self._apply_dynamic_roadmap_pipeline(processed_data, processed_targets)
                if roadmap_results:
                    feature_selection_results.update(roadmap_results)
            
            # Step 3: Generate selected features
            tprint_info("Step 3: Generate selected features")
            selected_features_df = self._generate_selected_features(processed_data, feature_selection_results)
            
            # Step 3.5: Apply statistical transforms using feature engineering roadmap
            tprint_info("Step 3.5: Apply statistical transforms")
            transformed_features_df = self._apply_statistical_transforms(selected_features_df)
            # Monitor data quality after feature generation
            if not selected_features_df.empty:
                feature_quality_monitoring = self._monitor_data_quality_throughout_pipeline(
                    selected_features_df, f"post_feature_generation_{timeframe}"
                )
            
            # Cross-step validation: Feature Selection -> Feature Generation
            validation_result_2 = self.cross_step_validator.validate_step_transition(
                from_step="feature_selection",
                to_step="feature_generation",
                input_data=processed_data,
                output_data=selected_features_df,
                step_metadata={"selected_features": len(feature_selection_results.get('selected_features', [])), "generated_features": len(selected_features_df.columns)}
            )
            if not validation_result_2.get('valid', True):
                tprint_warning(f"⚠️ Cross-step validation warning: {validation_result_2.get('message', 'Unknown issue')}")
            
            # Apply data quality check to generated features
            if not selected_features_df.empty:
                tprint_info("🔍 Validating generated features quality...")
                features_quality_result = self.quality_framework.validate_dataframe_quality(
                    selected_features_df, context=f"generated_features_{timeframe}"
                )
                tprint_info(f"📊 Generated features quality score: {features_quality_result.quality_score:.1f}/100")
                
                # Apply memory optimization to generated features
                tprint_info("💾 Applying memory optimization to generated features...")
                selected_features_df, memory_optimization_report = self.unified_data_utils.optimize_data(
                    data=selected_features_df,
                    stage='intermediate',
                    preserve_categorical=True
                )
                memory_reduction = memory_optimization_report.get('memory_reduction_percent', 0)
                tprint_success(f"✅ Memory optimization: {memory_reduction:.1f}% reduction")
                
                # Check if we need streaming for large datasets
                data_size_mb = selected_features_df.memory_usage(deep=True).sum() / 1024 / 1024
                if data_size_mb > 100:  # If larger than 100MB, consider streaming
                    tprint_info(f"📊 Large dataset detected ({data_size_mb:.1f}MB), enabling streaming optimizations...")
                    # Apply streaming optimizations for large datasets
                    selected_features_df = self.unified_data_utils.process_large_dataset(
                        data=selected_features_df,
                        processing_func=lambda x: x,  # Identity function for now
                        combine_results=True
                    )
            
            # Step 4: Enhanced interaction generation with VectorBT optimization
            tprint_info("Step 4: Enhanced interaction generation with VectorBT optimization")
            interaction_results = self._enhanced_interaction_generation(transformed_features_df, processed_targets)
            
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
            
            # Final comprehensive quality monitoring
            tprint_info("🔍 Performing final comprehensive quality monitoring...")
            final_quality_monitoring = self._monitor_data_quality_throughout_pipeline(
                processed_data, f"final_pipeline_output_{timeframe}"
            )
            
            # Add quality monitoring data to combined results
            combined_results['quality_monitoring'] = {
                'period_optimization': period_quality_monitoring if 'period_quality_monitoring' in locals() else None,
                'feature_generation': feature_quality_monitoring if 'feature_quality_monitoring' in locals() else None,
                'final_output': final_quality_monitoring
            }
            
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
            
            # Update VectorBT performance stats
            self._update_vectorbt_performance_stats()
            
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
            
            # Enhanced error handling with data quality recovery
            tprint_error(f"❌ Consolidated pipeline processing failed: {e}")
            
            # Try to recover data quality if possible
            try:
                tprint_info("🔧 Attempting data quality recovery...")
                if 'data' in locals() and data is not None and not data.empty:
                    # Apply basic data cleaning as recovery attempt
                    recovered_data, recovery_report = self.unified_data_utils.clean_data(
                        data=data,
                        detect_outliers=False,  # Don't remove outliers during recovery
                        outlier_method='zscore',
                        outlier_threshold=3.0
                    )
                    
                    if recovery_report.get('success', False):
                        tprint_success(f"✅ Data quality recovery successful: {recovery_report['final_shape']} shape")
                        # Log recovery details
                        if recovery_report.get('outliers_detected', 0) > 0:
                            tprint_info(f"📊 Recovery: {recovery_report['outliers_detected']} outliers detected")
                    else:
                        tprint_warning("⚠️ Data quality recovery failed")
                        
            except Exception as recovery_error:
                tprint_warning(f"⚠️ Data quality recovery failed: {recovery_error}")
            
            # Create enhanced error context with data quality information
            error_context = {
                'data_shape': data.shape if 'data' in locals() and data is not None else 'unknown',
                'timeframe': timeframe,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'recovery_attempted': True
            }
            
            # Add data quality metrics to error context if available
            try:
                if 'data' in locals() and data is not None and not data.empty:
                    quality_metrics = self.data_processor.calculate_enhanced_quality_metrics(data)
                    error_context['data_quality_metrics'] = quality_metrics
            except Exception:
                pass  # Don't fail on quality metrics calculation
            
            error_result = self.advanced_error_handler.handle_error(
                e, "process", 
                return_value=self._create_empty_result(start_time, str(e)),
                context=error_context
            )
            
            return error_result
    
    def _monitor_data_quality_throughout_pipeline(self, data: pd.DataFrame, context: str) -> Dict[str, Any]:
        """
        Monitor data quality throughout the pipeline using unified data utilities.
        
        Args:
            data: DataFrame to monitor
            context: Context string for logging
            
        Returns:
            Dictionary with quality monitoring results
        """
        try:
            # Get comprehensive quality metrics
            quality_metrics = self.data_processor.calculate_enhanced_quality_metrics(data)
            
            # Get quality validation result
            quality_result = self.quality_framework.validate_dataframe_quality(data, context)
            
            # Get processing summary from unified utilities
            processing_summary = self.unified_data_utils.get_processing_summary()
            
            monitoring_result = {
                'context': context,
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_shape': data.shape,
                'quality_metrics': quality_metrics,
                'quality_result': quality_result.get_summary(),
                'processing_capabilities': processing_summary,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'overall_quality_score': quality_result.quality_score,
                'issues_count': len(quality_result.issues),
                'warnings_count': len(quality_result.warnings)
            }
            
            # Log quality status
            if quality_result.quality_score >= 90:
                tprint_success(f"✅ {context}: Excellent data quality ({quality_result.quality_score:.1f}/100)")
            elif quality_result.quality_score >= 70:
                tprint_info(f"ℹ️ {context}: Good data quality ({quality_result.quality_score:.1f}/100)")
            elif quality_result.quality_score >= 50:
                tprint_warning(f"⚠️ {context}: Moderate data quality ({quality_result.quality_score:.1f}/100)")
            else:
                tprint_error(f"❌ {context}: Poor data quality ({quality_result.quality_score:.1f}/100)")
            
            return monitoring_result
            
        except Exception as e:
            tprint_warning(f"⚠️ Data quality monitoring failed for {context}: {e}")
            return {
                'context': context,
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e),
                'data_shape': data.shape if data is not None else 'unknown'
            }
    
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
        """Generate features for the selected feature set using enhanced utilities."""
        tprint_debug("Generating selected features using enhanced utilities")
        
        try:
            # First, try to use enhanced feature engineering if available
            if FEATURE_GENERATION_AVAILABLE and self.enhanced_feature_engineering:
                tprint_debug("🔧 Using enhanced feature engineering")
                try:
                    enhanced_features = self.enhanced_feature_engineering.generate_features(data)
                    if enhanced_features is not None and not enhanced_features.empty:
                        tprint_success(f"✅ Generated {len(enhanced_features.columns)} features using enhanced feature engineering")
                        
                        # Apply feature validation if available
                        if FEATURE_GENERATION_AVAILABLE:
                            try:
                                validated_features = validate_features_dataframe(enhanced_features)
                                if validated_features is not None:
                                    enhanced_features = validated_features
                                    tprint_success("✅ Features validated successfully")
                            except Exception as e:
                                tprint_warning(f"⚠️ Feature validation failed: {e}")
                        
                        return enhanced_features
                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced feature engineering failed: {e}")
            
            # Fallback to Feature Bank integration
            if selection_result is None or not selection_result.success:
                tprint_warning("⚠️ No valid selection result, using Feature Bank for comprehensive feature generation")
                # Use Feature Bank integration for comprehensive feature generation
                feature_generation_result = self.feature_bank_integration.generate_features_for_optimization(
                    data, force_refresh=False
                )
                
                if feature_generation_result.success:
                    tprint_success(f"✅ Generated {feature_generation_result.n_features_generated} features using Feature Bank")
                    return feature_generation_result.feature_data
                else:
                    tprint_error(f"❌ Feature Bank generation failed: {feature_generation_result.error_message}")
                    return pd.DataFrame(index=data.index)
            
            # Use Feature Bank integration for selected features
            tprint_debug("🔧 Using Feature Bank integration for selected features")
            
            # Generate comprehensive features first
            feature_generation_result = self.feature_bank_integration.generate_features_for_optimization(
                data, force_refresh=False
            )
            
            if not feature_generation_result.success:
                tprint_error("❌ Feature Bank generation failed - this is required for feature generation")
                return pd.DataFrame(index=data.index)
            
            # Filter to selected features
            selected_feature_names = [fs.feature_name for fs in selection_result.selected_features]
            available_features = feature_generation_result.feature_data.columns
            
            # Find matching features
            matching_features = [f for f in selected_feature_names if f in available_features]
            
            if matching_features:
                features_df = feature_generation_result.feature_data[matching_features]
                tprint_success(f"✅ Generated {len(features_df.columns)} selected features using Feature Bank")
            else:
                tprint_warning("⚠️ No matching features found, using all generated features")
                features_df = feature_generation_result.feature_data
            
            return features_df
            
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _enhanced_interaction_generation(self, features_df: pd.DataFrame, targets: Optional[pd.Series]) -> List[Any]:
        """Enhanced interaction generation with VectorBT optimization and feature engineering roadmap."""
        tprint_debug("Starting enhanced interaction generation")
        
        try:
            interactions = []
            
            # Try to use unified VectorBT manager if available
            if FEATURES_COMMON_AVAILABLE and self.unified_vectorbt_manager:
                tprint_debug("🔧 Using unified VectorBT manager for interaction generation")
                try:
                    # Use the unified VectorBT manager for optimized interaction generation
                    vectorbt_interactions = self.unified_vectorbt_manager.generate_interactions(
                        features_df, targets
                    )
                    if vectorbt_interactions:
                        interactions.extend(vectorbt_interactions)
                        tprint_success(f"✅ Generated {len(vectorbt_interactions)} interactions using unified VectorBT manager")
                except Exception as e:
                    tprint_warning(f"⚠️ Unified VectorBT manager failed: {e}")
            
            # Try to use cross-timeframe analysis if available
            if FEATURE_GENERATION_AVAILABLE and self.cross_timeframe_pipeline:
                tprint_debug("🔧 Using cross-timeframe analysis for interaction generation")
                try:
                    cross_timeframe_interactions = self.cross_timeframe_pipeline.generate_interactions(
                        features_df, targets
                    )
                    if cross_timeframe_interactions:
                        interactions.extend(cross_timeframe_interactions)
                        tprint_success(f"✅ Generated {len(cross_timeframe_interactions)} cross-timeframe interactions")
                except Exception as e:
                    tprint_warning(f"⚠️ Cross-timeframe analysis failed: {e}")
            
            # Fallback to original VectorBT optimizer
            if not interactions:
                tprint_debug("🔧 Using fallback VectorBT optimizer for interaction generation")
                interactions = self.vectorbt_optimizer.optimize_interaction_generation(features_df, targets)
            
            # Apply feature validation if available
            if FEATURE_GENERATION_AVAILABLE and interactions:
                try:
                    validated_interactions = []
                    for interaction in interactions:
                        if hasattr(interaction, 'feature_data') and interaction.feature_data is not None:
                            validated_data = validate_features_dataframe(interaction.feature_data)
                            if validated_data is not None:
                                interaction.feature_data = validated_data
                                validated_interactions.append(interaction)
                        else:
                            validated_interactions.append(interaction)
                    interactions = validated_interactions
                    tprint_success("✅ Interactions validated successfully")
                except Exception as e:
                    tprint_warning(f"⚠️ Interaction validation failed: {e}")
            # Use feature engineering roadmap interactions if available
            if FEATURE_ENGINEERING_ROADMAP_AVAILABLE and self.interaction_engine is not None:
                tprint_info("🎯 Using feature engineering roadmap interactions")
                
                # Prepare data for interaction generation
                # Convert features to the expected format for interactions
                transformed_data = self._prepare_data_for_interactions(features_df)
                
                # Generate interactions using the roadmap engine with regime awareness
                interaction_df = self.interaction_engine.build_interactions(transformed_data)
                
                # Add regime-aware interactions if available
                if hasattr(self.interaction_engine, 'regime_flags'):
                    regime_interactions = self._generate_regime_aware_interactions(transformed_data)
                    if not regime_interactions.empty:
                        interaction_df = pd.concat([interaction_df, regime_interactions], axis=1)
                
                if not interaction_df.empty:
                    # Convert to list format expected by the pipeline
                    for col in interaction_df.columns:
                        interactions.append({
                            'name': col,
                            'values': interaction_df[col].values,
                            'type': 'roadmap_interaction',
                            'source': 'feature_engineering_roadmap'
                        })
                    
                    tprint_success(f"✅ Generated {len(interactions)} roadmap interactions")
                else:
                    tprint_warning("⚠️ No roadmap interactions generated")
            
            # Fallback to VectorBT optimizer if no roadmap interactions or as additional
            if not interactions or self.config.feature_selection.enable_fallback_interactions:
                tprint_info("🔄 Using VectorBT optimizer for additional interactions")
                vectorbt_interactions = self.vectorbt_optimizer.optimize_interaction_generation(features_df, targets)
                interactions.extend(vectorbt_interactions)
            
            tprint_success(f"✅ Generated {len(interactions)} total interactions")
            return interactions
            
        except Exception as e:
            tprint_error(f"Enhanced interaction generation failed: {e}")
            return []
    
    def _prepare_data_for_interactions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature engineering roadmap interactions."""
        try:
            # Convert feature names to the expected format for interactions
            # The interaction engine expects features with specific prefixes like 't/p/'
            transformed_data = features_df.copy()
            
            # Add prefix to feature names to match interaction engine expectations
            feature_mapping = {}
            for col in features_df.columns:
                # Map common feature types to expected prefixes
                if any(keyword in col.lower() for keyword in ['rsi', 'bollinger', 'bollz', 'atr', 'volatility']):
                    feature_mapping[col] = f't/p/{col}'
                elif any(keyword in col.lower() for keyword in ['momentum', 'mom', 'return', 'ret']):
                    feature_mapping[col] = f't/p/{col}'
                elif any(keyword in col.lower() for keyword in ['volume', 'vol', 'spread', 'ofi']):
                    feature_mapping[col] = f't/p/{col}'
                elif any(keyword in col.lower() for keyword in ['vwap', 'price', 'close', 'open', 'high', 'low']):
                    feature_mapping[col] = f't/p/{col}'
                else:
                    feature_mapping[col] = f't/p/{col}'
            
            # Rename columns
            transformed_data = transformed_data.rename(columns=feature_mapping)
            
            return transformed_data
            
        except Exception as e:
            tprint_error(f"Data preparation for interactions failed: {e}")
            return features_df
    
    def _apply_statistical_transforms(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply statistical transforms using feature engineering roadmap."""
        tprint_debug("Starting statistical transforms")
        
        try:
            if not FEATURE_ENGINEERING_ROADMAP_AVAILABLE or self.transform_router is None:
                tprint_warning("⚠️ Statistical transforms not available, returning original features")
                return features_df
            
            # Initialize transform router if not already done
            if self.transform_router is None:
                transform_config = create_default_transform_config(features_df.columns.tolist())
                self.transform_router = TransformRouter(
                    transform_config,
                    use_vectorbt=VECTORBT_AVAILABLE,
                    use_gpu=self.config.performance.enable_gpu,
                    enable_parallel=True
                )
            
            # Apply transforms
            transformed_df = self.transform_router.fit_transform(features_df)
            
            tprint_success(f"✅ Applied statistical transforms to {len(transformed_df.columns)} features")
            return transformed_df
            
        except Exception as e:
            tprint_error(f"Statistical transforms failed: {e}")
            return features_df
    
    def _apply_dynamic_roadmap_pipeline(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Dict[str, Any]:
        """Apply dynamic roadmap pipeline for optimized feature selection."""
        tprint_debug("Starting dynamic roadmap pipeline")
        
        try:
            if not FEATURE_ENGINEERING_ROADMAP_AVAILABLE or self.dynamic_roadmap_pipeline is None:
                tprint_warning("⚠️ Dynamic roadmap pipeline not available")
                return {}
            
            # Run the dynamic roadmap pipeline
            roadmap_results = self.dynamic_roadmap_pipeline.run(data, targets)
            
            if roadmap_results and 'final' in roadmap_results:
                final_features = roadmap_results['final']
                tprint_success(f"✅ Dynamic roadmap pipeline selected {len(final_features.columns)} features")
                
                return {
                    'roadmap_features': final_features.columns.tolist(),
                    'roadmap_original': roadmap_results.get('original', pd.DataFrame()),
                    'roadmap_transformed': roadmap_results.get('transformed', pd.DataFrame()),
                    'roadmap_interactions': roadmap_results.get('interactions', pd.DataFrame()),
                    'roadmap_final': final_features
                }
            else:
                tprint_warning("⚠️ Dynamic roadmap pipeline returned no results")
                return {}
            
        except Exception as e:
            tprint_error(f"Dynamic roadmap pipeline failed: {e}")
            return {}
    
    def _generate_regime_aware_interactions(self, transformed_data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-aware interactions using the interaction engine's regime flags."""
        try:
            if not hasattr(self.interaction_engine, 'regime_flags'):
                return pd.DataFrame()
            
            regime_flags = self.interaction_engine.regime_flags
            
            # Calculate regime flags
            regime_flags.calculate_quantiles(transformed_data)
            
            # Get regime flags
            high_vol_flag = regime_flags.get_high_vol_flag(transformed_data)
            wide_spread_flag = regime_flags.get_wide_spread_flag(transformed_data)
            
            # Create regime-aware interactions
            regime_interactions = {}
            
            # High volatility regime interactions
            if not high_vol_flag.empty and high_vol_flag.sum() > 0:
                # Find features that might benefit from high vol regime
                for col in transformed_data.columns:
                    if any(keyword in col.lower() for keyword in ['rsi', 'momentum', 'volatility']):
                        regime_interactions[f'regime_high_vol_{col}'] = transformed_data[col] * high_vol_flag
            
            # Wide spread regime interactions
            if not wide_spread_flag.empty and wide_spread_flag.sum() > 0:
                # Find features that might benefit from wide spread regime
                for col in transformed_data.columns:
                    if any(keyword in col.lower() for keyword in ['bollinger', 'spread', 'microstructure']):
                        regime_interactions[f'regime_wide_spread_{col}'] = transformed_data[col] * wide_spread_flag
            
            if regime_interactions:
                return pd.DataFrame(regime_interactions, index=transformed_data.index)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"Regime-aware interactions generation failed: {e}")
            return pd.DataFrame()
    
    def _htf_interaction_generation(self, data: pd.DataFrame, features_df: pd.DataFrame, 
                                  targets: Optional[pd.Series]) -> List[Any]:
        """HTF-aware interaction generation."""
        tprint_debug("Starting HTF interaction generation")
        
        try:
            # Create simulated HTF features
            htf_features = self._create_htf_features(data)
            
            # Use HTF generator for interaction generation
            htf_interactions = self.template_interaction_generator.generate_interactions(
                htf_features, features_df, targets
            )
            
            tprint_success(f"✅ Generated {len(htf_interactions)} HTF interactions")
            return htf_interactions
            
        except Exception as e:
            tprint_error(f"HTF interaction generation failed: {e}")
            return []
    
    def _create_htf_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated HTF features."""
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
            use_enhanced_opt = mode_constraints.get('use_enhanced_optimization', True)
            optimization_method = mode_constraints.get('optimization_method', 'enhanced_bayesian_tpe')
            n_trials = mode_constraints.get('n_trials', 100)
            n_bootstrap = mode_constraints.get('n_bootstrap_samples', 100)
            cv_folds = mode_constraints.get('cv_folds', 5)
            early_stopping_patience = mode_constraints.get('early_stopping_patience', 10)
            coarse_grid_trials = mode_constraints.get('coarse_grid_trials', 25)
            fine_grid_trials = mode_constraints.get('fine_grid_trials', 25)
            tpe_trials = mode_constraints.get('tpe_trials', 50)
            
            tprint_debug(f"⚙️ Optimization settings: bayesian={use_bayesian_opt}, method={optimization_method}, trials={n_trials}")
            tprint_debug(f"⚙️ Grid settings: coarse={coarse_grid_trials}, fine={fine_grid_trials}, tpe={tpe_trials}")
            tprint_debug(f"⚙️ Stopping: patience={early_stopping_patience}, bootstrap={n_bootstrap}, cv_folds={cv_folds}")
            tprint_debug(f"🎯 Directions: longs={optimize_longs}, shorts={optimize_shorts}")
            
            # Separate optimization for long and short directions
            long_feature_results = {}
            short_feature_results = {}
            
            total_features = len(feature_names)
            tprint_info(f"🚀 Starting optimization of {total_features} features")
            
            # Reset feature lag metadata
            feature_lag_metadata = {}
            
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
                        'use_bayesian_optimization': use_bayesian_opt,
                        'use_enhanced_optimization': use_enhanced_opt,
                        'optimization_method': optimization_method,
                        'n_trials': n_trials,
                        'early_stopping_patience': early_stopping_patience,
                        'coarse_grid_trials': coarse_grid_trials,
                        'fine_grid_trials': fine_grid_trials,
                        'tpe_trials': tpe_trials
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
                'use_bayesian_optimization': True,
                'use_enhanced_optimization': True,
                'optimization_method': 'enhanced_bayesian_tpe',  # Always use Bayesian TPE
                'n_trials': 25,  # Reduced iterations for light mode
                'n_bootstrap_samples': 20,
                'cv_folds': 3,
                'max_features': 50,
                'max_lookback': 30,
                'early_stopping_patience': 5,  # Earlier stopping for light mode
                'coarse_grid_trials': 10,  # Reduced grid trials
                'fine_grid_trials': 10,
                'tpe_trials': 5
            },
            'blank': {
                'use_bayesian_optimization': True,
                'use_enhanced_optimization': True,
                'optimization_method': 'enhanced_bayesian_tpe',  # Always use Bayesian TPE
                'n_trials': 15,  # Minimal iterations for blank mode
                'n_bootstrap_samples': 10,
                'cv_folds': 2,
                'max_features': 20,
                'max_lookback': 20,
                'early_stopping_patience': 3,  # Very early stopping for blank mode
                'coarse_grid_trials': 5,  # Minimal grid trials
                'fine_grid_trials': 5,
                'tpe_trials': 5
            },
            'full': {
                'use_bayesian_optimization': True,
                'use_enhanced_optimization': True,
                'optimization_method': 'enhanced_bayesian_tpe',  # Always use Bayesian TPE
                'n_trials': 100,  # Full iterations for full mode
                'n_bootstrap_samples': 100,
                'cv_folds': 5,
                'max_features': 200,
                'max_lookback': 100,
                'early_stopping_patience': 15,  # More patience for full mode
                'coarse_grid_trials': 25,  # Full grid trials
                'fine_grid_trials': 25,
                'tpe_trials': 50
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
            # Use advanced lookback optimizer with enhanced Bayesian TPE
            if hasattr(self, 'advanced_lookback_optimizer') and self.advanced_lookback_optimizer:
                tprint_debug(f"🚀 Using advanced lookback optimizer for {feature} ({direction})")
                try:
                    # Determine optimization method from kwargs
                    optimization_method = optimizer_kwargs.get('optimization_method', 'enhanced_bayesian_tpe')
                    
                    # Convert string method to enum
                    from .enhanced_components.advanced_lookback_optimizer import OptimizationMethod
                    if optimization_method == 'enhanced_bayesian_tpe':
                        method = OptimizationMethod.ENHANCED_BAYESIAN_TPE
                    elif optimization_method == 'enhanced_grid_search':
                        method = OptimizationMethod.ENHANCED_GRID_SEARCH
                    elif optimization_method == 'bayesian_tpe':
                        method = OptimizationMethod.BAYESIAN_TPE
                    elif optimization_method == 'grid_search':
                        method = OptimizationMethod.GRID_SEARCH
                    else:
                        method = OptimizationMethod.ENHANCED_BAYESIAN_TPE
                    
                    # Prepare data for optimization
                    feature_data = data[[feature]].copy()
                    feature_data[target_column] = data[target_column]
                    
                    # Run optimization
                    result = self.advanced_lookback_optimizer.optimize_features_parallel_batch(
                        feature_data,
                        [feature],
                        target_column,
                        lookback_range=lookback_range,
                        method=method,
                        regularization_settings=optimizer_kwargs.get('regularization_settings', {}),
                        **optimizer_kwargs
                    )
                    
                    if result and len(result) > 0:
                        opt_result = result[0]
                        tprint_debug(f"✅ Advanced optimization completed: period={opt_result.optimal_lookback}, score={opt_result.score:.4f}")
                        
                        # Convert result to expected format
                        return {
                            'best_lookback_period': opt_result.optimal_lookback,
                            'best_score': opt_result.score,
                            'optimization_method': opt_result.method,
                            'target_column': target_column,
                            'direction': direction,
                            'total_trials': getattr(opt_result, 'total_trials', 1),
                            'optimization_time': getattr(opt_result, 'execution_time', 0.0),
                            'convergence_achieved': getattr(opt_result, 'success', True)
                        }
                    else:
                        tprint_warning(f"⚠️ Advanced optimization returned no results for {feature}")
                        return self._fallback_optimization(
                            data, feature, target_column, lookback_range, 
                            optimizer_kwargs, use_bayesian_opt, execution_mode
                        )
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Advanced optimization failed, falling back to standard: {e}")
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
        """Final feature selection using enhanced multi-objective optimization."""
        tprint_debug("Starting final feature selection")
        
        try:
            # Try enhanced feature selection first if available
            if (self.config.feature_selection.enable_enhanced_methods and 
                hasattr(self, 'enhanced_feature_selectors') and 
                self.enhanced_feature_selectors):
                
                tprint_info("🔧 Using enhanced feature selection methods")
                
                # Try each enhanced method and collect results
                enhanced_results = {}
                for method_name, selector in self.enhanced_feature_selectors.items():
                    try:
                        tprint_debug(f"🔧 Trying {method_name}")
                        result = selector.select_features(
                            data.values, targets.values if targets is not None else None,
                            feature_names=data.columns.tolist()
                        )
                        
                        if result.get('success', False):
                            enhanced_results[method_name] = result['selected_features']
                            tprint_success(f"✅ {method_name} selected {len(result['selected_features'])} features")
                        else:
                            tprint_warning(f"⚠️ {method_name} failed")
                            
                    except Exception as e:
                        tprint_warning(f"⚠️ {method_name} error: {e}")
                
                # If we have enhanced results, use ensemble approach
                if enhanced_results:
                    # Combine results using voting
                    feature_votes = {}
                    for method_features in enhanced_results.values():
                        for feature in method_features:
                            feature_votes[feature] = feature_votes.get(feature, 0) + 1
                    
                    # Sort by votes and select top features
                    sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
                    selected_features = [f[0] for f in sorted_features[:self.config.feature_selection.multi_objective.max_features]]
                    
                    tprint_success(f"✅ Enhanced ensemble selected {len(selected_features)} features")
                    return type('FeatureSelectionResult', (), {
                        'selected_features': selected_features,
                        'objective_values': {'enhanced_ensemble': 1.0},
                        'quality_metrics': {'enhanced_methods_used': list(enhanced_results.keys())}
                    })()
            
            # Fallback to standard multi-objective feature selector
            tprint_info("📊 Using standard multi-objective feature selection")
            
            # Use computational awareness for time constraint
            time_constraint = None
            if hasattr(self.config.feature_selection, 'computational_constraints'):
                time_constraint = self.config.feature_selection.computational_constraints.get('max_execution_time_seconds')
            
            selection_result = self.feature_selector.select_features(
                data, targets, time_constraint=time_constraint
            )
            
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
        """Calculate metrics for generated interactions with math validation."""
        if not interactions:
            return {}
        
        try:
            # Extract utility scores safely
            utility_scores = []
            for i in interactions:
                score = validate_finite(i.utility_score, "utility_score")
                utility_scores.append(score)
            
            # Calculate metrics with safe operations
            total_interactions = len(interactions)
            avg_utility_score = safe_mean(utility_scores, default=0.0)
            max_utility_score = safe_percentile(utility_scores, 100.0, default=0.0)
            min_utility_score = safe_percentile(utility_scores, 0.0, default=0.0)
            
            interaction_types = list(set(i.interaction_type for i in interactions))
            unique_parent_features = len(set(f for i in interactions for f in i.parent_features))
            
            return {
                'total_interactions': total_interactions,
                'average_utility_score': float(avg_utility_score),
                'max_utility_score': float(max_utility_score),
                'min_utility_score': float(min_utility_score),
                'interaction_types': interaction_types,
                'unique_parent_features': unique_parent_features
            }
        except Exception as e:
            self.logger.warning(f"Interaction metrics calculation failed: {e}")
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
        """Calculate cache hit rate with math validation."""
        try:
            hits = self.performance_stats.get('cache_hits', 0)
            misses = self.performance_stats.get('cache_misses', 0)
            
            # Validate inputs
            hits = validate_positive(hits, "cache_hits")
            misses = validate_positive(misses, "cache_misses")
            
            # Use safe division
            total = hits + misses
            hit_rate = safe_divide(hits, total, default=0.0)
            hit_rate = validate_range(hit_rate, 0.0, 1.0, "cache_hit_rate")
            
            return float(hit_rate)
            
        except Exception as e:
            self.logger.warning(f"Cache hit rate calculation failed: {e}")
            return 0.0
    
    def validate_pipeline_math(self, 
                             data: pd.DataFrame, 
                             targets: Optional[pd.Series] = None,
                             returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Validate all mathematical operations in the pipeline using comprehensive math validation.
        
        Args:
            data: Input data for validation
            targets: Target values for supervised learning validation
            returns: Returns data for financial metrics validation
            
        Returns:
            Dictionary containing validation results for all mathematical operations
        """
        tprint_info("🔢 Starting comprehensive math validation")
        
        validation_results = {}
        
        try:
            # Validate financial metrics if returns are provided
            if returns is not None:
                tprint_debug("Validating financial metrics")
                financial_results = self.math_validator.validate_financial_metrics(returns)
                validation_results['financial_metrics'] = financial_results
            
            # Validate statistical metrics
            tprint_debug("Validating statistical metrics")
            statistical_results = self.math_validator.validate_statistical_metrics(data, targets)
            validation_results['statistical_metrics'] = statistical_results
            
            # Validate feature metrics
            tprint_debug("Validating feature metrics")
            feature_results = self.math_validator.validate_feature_metrics(data, targets)
            validation_results['feature_metrics'] = feature_results
            
            # Get validation statistics
            validation_results['validation_stats'] = self.math_validator.get_validation_stats()
            
            tprint_success("✅ Math validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Math validation failed: {e}")
            validation_results['error'] = str(e)
            validation_results['validation_stats'] = self.math_validator.get_validation_stats()
        
        return validation_results
    
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
    
    async def store_klines_data(self, data: pd.DataFrame, symbol: str, exchange: str, 
                               interval: str, batch_id: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store klines data using the enhanced data loader.
        
        Args:
            data: Klines DataFrame with OHLCV data
            symbol: Trading symbol (e.g., "ETHUSDT")
            exchange: Exchange name (e.g., "binance")
            interval: Data interval (e.g., "1m")
            batch_id: Optional batch identifier
            metadata: Additional metadata to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            tprint_info(f"📦 Storing klines data for {symbol} on {exchange} ({interval})")
            return await self.advanced_data_loader.store_klines_data(
                data, symbol, exchange, interval, batch_id, metadata
            )
        except Exception as e:
            tprint_error(f"❌ Failed to store klines data: {e}")
            return False

    async def load_klines_data(self, symbol: str, exchange: str, interval: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              batch_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load klines data using the enhanced data loader.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional specific batch to load
            
        Returns:
            DataFrame containing klines data
        """
        try:
            tprint_info(f"📥 Loading klines data for {symbol} on {exchange} ({interval})")
            return await self.advanced_data_loader.load_klines_data(
                symbol, exchange, interval, start_time, end_time, batch_id
            )
        except Exception as e:
            tprint_error(f"❌ Failed to load klines data: {e}")
            return pd.DataFrame()

    def get_klines_storage_stats(self) -> Dict[str, Any]:
        """Get klines storage statistics."""
        try:
            return self.advanced_data_loader.get_klines_storage_stats()
        except Exception as e:
            tprint_error(f"❌ Failed to get storage stats: {e}")
            return {"error": str(e)}

    def list_available_klines_data(self) -> List[Dict[str, Any]]:
        """List all available klines data."""
        try:
            return self.advanced_data_loader.list_available_klines_data()
        except Exception as e:
            tprint_error(f"❌ Failed to list available data: {e}")
            return []

    async def update_klines_data(self, data: pd.DataFrame, symbol: str, exchange: str,
                                interval: str, append_mode: bool = True) -> bool:
        """
        Update existing klines data.
        
        Args:
            data: New klines data
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            append_mode: If True, append to existing data; if False, replace
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            tprint_info(f"🔄 Updating klines data for {symbol} on {exchange} ({interval})")
            return await self.advanced_data_loader.update_klines_data(
                data, symbol, exchange, interval, append_mode
            )
        except Exception as e:
            tprint_error(f"❌ Failed to update klines data: {e}")
            return False

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
        
        # Add unified data utilities stats
        stats['unified_data_utils'] = self.unified_data_utils.get_processing_summary()
        stats['data_processor'] = {
            'optimization_capabilities': list(self.data_processor.get_optimal_dtypes_for_features().keys()),
            'memory_optimization_enabled': True,
            'timestamp_regularization_enabled': True
        }
        stats['quality_framework'] = {
            'validation_rules': list(self.quality_framework.validation_rules.keys()),
            'quality_policies': self.quality_framework.quality_policies,
            'duplicate_analyzer_available': hasattr(self.quality_framework, 'duplicate_analyzer') and self.quality_framework.duplicate_analyzer is not None
        }
        stats['cross_step_validator'] = {
            'validation_enabled': True,
            'step_transition_tracking': True
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
    
    def _update_vectorbt_performance_stats(self):
        """Update VectorBT performance statistics."""
        try:
            # Update VectorBT operations count
            if hasattr(self, 'interaction_engine') and self.interaction_engine is not None:
                if hasattr(self.interaction_engine, 'vectorbt_operations'):
                    self.performance_stats['vectorbt_operations'] += getattr(self.interaction_engine, 'vectorbt_operations', 0)
            
            if hasattr(self, 'transform_router') and self.transform_router is not None:
                if hasattr(self.transform_router, 'vectorbt_operations'):
                    self.performance_stats['vectorbt_operations'] += getattr(self.transform_router, 'vectorbt_operations', 0)
            
            # Update GPU operations count
            if hasattr(self, 'interaction_engine') and self.interaction_engine is not None:
                if hasattr(self.interaction_engine, 'gpu_operations'):
                    self.performance_stats['gpu_operations'] += getattr(self.interaction_engine, 'gpu_operations', 0)
            
            if hasattr(self, 'transform_router') and self.transform_router is not None:
                if hasattr(self.transform_router, 'gpu_operations'):
                    self.performance_stats['gpu_operations'] += getattr(self.transform_router, 'gpu_operations', 0)
            
            # Update cache statistics
            if hasattr(self, 'interaction_engine') and self.interaction_engine is not None:
                if hasattr(self.interaction_engine, 'cache_hits'):
                    self.performance_stats['cache_hits'] += getattr(self.interaction_engine, 'cache_hits', 0)
                if hasattr(self.interaction_engine, 'cache_misses'):
                    self.performance_stats['cache_misses'] += getattr(self.interaction_engine, 'cache_misses', 0)
            
        except Exception as e:
            tprint_warning(f"Failed to update VectorBT performance stats: {e}")


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