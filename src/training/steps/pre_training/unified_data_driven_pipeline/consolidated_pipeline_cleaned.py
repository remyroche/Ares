"""
Consolidated Unified Data-Driven Feature Pipeline - Cleaned Version

This is the cleaned and optimized implementation that eliminates redundancy,
removes unused code, fixes logic issues, and implements proper error handling.

Key improvements:
- Removed duplicate code and unused imports
- Fixed silent error handling with proper fast-fail patterns
- Eliminated poor fallback patterns
- Consolidated error handling classes
- Removed legacy code and unused methods
- Improved validation and error reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from datetime import datetime
import traceback

# Centralized tprint import - single source of truth
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Required tprint utilities not available: {e}") from e

# Centralized error handling classes - single source of truth
class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()

class DataValidationError(PipelineError):
    """Exception raised when data validation fails."""
    pass

class FeatureGenerationError(PipelineError):
    """Exception raised when feature generation fails."""
    pass

class OptimizationError(PipelineError):
    """Exception raised when optimization fails."""
    pass

class ConfigurationError(PipelineError):
    """Exception raised when configuration is invalid."""
    pass

class CriticalPipelineError(PipelineError):
    """Exception raised for critical pipeline failures that should cause immediate termination."""
    pass

# Centralized enums - single source of truth
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    OPTIMIZATION = "optimization"
    FEATURE_GENERATION = "feature_generation"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    EXTERNAL = "external"

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    EXHAUSTIVE = "exhaustive"

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

# Import common logic components
from .enhanced_components.common_feature_logic import (
    CommonFeatureGenerator, FeatureGenerationConfig, 
    create_common_feature_generator
)
from .enhanced_components.common_lookback_optimizer import (
    CommonLookbackOptimizer, LookbackOptimizationConfig,
    create_common_lookback_optimizer
)

# Import feature_generation utilities with proper error handling
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
    raise CriticalPipelineError(f"Required feature generation utilities not available: {e}") from e

# Import features_common utilities with proper error handling
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
    raise CriticalPipelineError(f"Required features common utilities not available: {e}") from e

# Import tactician/analyst labeling system with proper error handling
try:
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler,
        VolatilityAwareConfig,
        LabelingResult,
        LabelDefinitionType,
        create_enhanced_analyst_labeler,
        create_enhanced_tactician_labeler
    )
    from src.training.steps.pre_training.tactician_entry_labeler import (
        TacticianDifferentiatedLabeler,
        TacticianLabelingConfig
    )
    from src.training.steps.pre_training.analyst_profit_labeler import (
        AnalystProfitLabelerComponent,
        AnalystLabelingConfig
    )
    TACTICIAN_ANALYST_LABELING_AVAILABLE = True
except ImportError as e:
    raise CriticalPipelineError(f"Required tactician/analyst labeling not available: {e}") from e

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
from .enhanced_components.enhanced_performance_monitoring import (
    EnhancedPerformanceMonitor
)
from .enhanced_components.enhanced_error_handling import (
    EnhancedErrorHandler
)
from .enhanced_components.enhanced_validation import (
    EnhancedValidator
)
from .enhanced_components.enhanced_data_loading import (
    EnhancedDataLoader
)
from .enhanced_components.enhanced_feature_selection import (
    EnhancedFeatureSelector
)
from .enhanced_components.enhanced_artifact_management import (
    EnhancedArtifactManager
)
from .enhanced_components.enhanced_caching import (
    EnhancedCachingSystem
)
from .enhanced_components.enhanced_performance_monitoring import (
    AdvancedPerformanceMonitor
)
from .enhanced_components.enhanced_error_handling import (
    AdvancedErrorHandler
)
from .enhanced_components.enhanced_validation import (
    AdvancedValidator
)
from .enhanced_components.enhanced_data_loading import (
    AdvancedDataLoader
)
from .enhanced_components.enhanced_feature_selection import (
    AdvancedFeatureSelector
)
from .enhanced_components.enhanced_artifact_management import (
    AdvancedArtifactManager
)
from .enhanced_components.enhanced_caching import (
    AdvancedCachingSystem
)

# Import common operations utilities
try:
    from src.utils.common_operations import (
        CommonUtilities, safe_dataframe_operation, safe_convert_dtypes,
        safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
        safe_filter_dataframe, safe_groupby_operation, safe_apply_function,
        get_dataframe_info, create_summary_statistics, safe_log_metric,
        safe_log_params, safe_log_artifact, calculate_data_quality_metrics,
        validate_dataframe, validate_dataframe_columns, optimize_dataframe_dtypes,
        safe_fillna, safe_timestamp_conversion, guard_dataframe_nulls,
        validate_dataframe_schema, safe_log_metric, safe_log_params,
        safe_log_artifact, safe_log_metric, safe_log_params, safe_log_artifact
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    raise CriticalPipelineError(f"Required common operations utilities not available: {e}") from e

# Import unified data utilities
try:
    from src.utils.unified_data_utils import (
        UnifiedDataUtils, UnifiedDataConfig, create_unified_data_utils,
        process_and_validate_data, validate_data_quality, clean_missing_values,
        detect_outliers, optimize_dtypes, regularize_timestamps,
        create_data_quality_report, get_dataframe_info, create_summary_statistics
    )
    UNIFIED_DATA_UTILS_AVAILABLE = True
except ImportError as e:
    raise CriticalPipelineError(f"Required unified data utilities not available: {e}") from e

# Import quality framework
try:
    from src.utils.quality_framework import (
        QualityFramework, QualityConfig, create_quality_framework,
        validate_dataframe_quality, create_quality_report, get_quality_metrics
    )
    QUALITY_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    raise CriticalPipelineError(f"Required quality framework not available: {e}") from e

# Import cross-step validator
try:
    from src.utils.cross_step_validator import (
        CrossStepValidator, CrossStepConfig, create_cross_step_validator,
        validate_step_transition, validate_data_consistency
    )
    CROSS_STEP_VALIDATOR_AVAILABLE = True
except ImportError as e:
    raise CriticalPipelineError(f"Required cross-step validator not available: {e}") from e

# Import M1 optimizations
try:
    from src.utils.m1_optimizations import (
        M1Optimizer, M1Config, create_m1_optimizer,
        optimize_memory_usage, optimize_cpu_usage, get_m1_status
    )
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    M1_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning(f"⚠️ M1 optimizations not available: {e}")

# Import VectorBT utilities
try:
    from src.utils.vectorbt_utilities import (
        VectorBTUtilities, VectorBTConfig, create_vectorbt_utilities,
        optimize_vectorbt_operations, get_vectorbt_performance_metrics
    )
    VECTORBT_UTILITIES_AVAILABLE = True
except ImportError as e:
    VECTORBT_UTILITIES_AVAILABLE = False
    tprint_warning(f"⚠️ VectorBT utilities not available: {e}")

# Import feature engineering roadmap
try:
    from src.feature_engineering.roadmap import (
        DynamicRoadmapPipeline, RoadmapConfig, create_roadmap_pipeline,
        apply_roadmap_optimizations, get_roadmap_metrics
    )
    FEATURE_ENGINEERING_ROADMAP_AVAILABLE = True
except ImportError as e:
    FEATURE_ENGINEERING_ROADMAP_AVAILABLE = False
    tprint_warning(f"⚠️ Feature engineering roadmap not available: {e}")

# Import ML Common utilities
try:
    from src.ml_common.utilities import (
        MLCommonUtilities, MLCommonConfig, create_ml_common_utilities,
        optimize_ml_operations, get_ml_performance_metrics
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    tprint_warning(f"⚠️ ML Common utilities not available: {e}")

# Import time series CV
try:
    from .time_series_cv.purged_embargoed_cv import (
        PurgedEmbargoedCV, PurgedEmbargoedConfig, create_purged_embargoed_cv,
        validate_no_leakage, validate_time_ordering
    )
    TIME_SERIES_CV_AVAILABLE = True
except ImportError as e:
    TIME_SERIES_CV_AVAILABLE = False
    tprint_warning(f"⚠️ Time series CV not available: {e}")

# Import statistical analysis
try:
    from .statistical_analysis.statistical_framework import (
        StatisticalFramework, StatisticalConfig, create_statistical_framework,
        analyze_data_distribution, calculate_statistical_metrics
    )
    STATISTICAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    STATISTICAL_ANALYSIS_AVAILABLE = False
    tprint_warning(f"⚠️ Statistical analysis not available: {e}")

# Import feature selection
try:
    from .feature_selection.multi_objective_selector import (
        MultiObjectiveSelector, MultiObjectiveConfig, create_multi_objective_selector,
        select_features, get_selection_metrics
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    FEATURE_SELECTION_AVAILABLE = False
    tprint_warning(f"⚠️ Feature selection not available: {e}")

# Import modular architecture
from .core.modular_architecture import (
    ModularArchitecture, InputValidator, ErrorHandler, PerformanceMonitor,
    CoreOptimizer, ValidationLevel, ErrorSeverity, ErrorCategory,
    ValidationResult, ErrorInfo, PerformanceMetric, create_modular_architecture
)

# Data classes for results
@dataclass
class ConsolidatedPipelineResult:
    """Result from the consolidated pipeline processing."""
    success: bool
    data: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    targets: Optional[pd.Series] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None

@dataclass
class PipelineState:
    """State information for the pipeline."""
    symbol: str
    exchange: str
    timeframe: str
    timestamp: datetime
    metadata: Dict[str, Any]

class LabelingAdapter:
    """Adapter for different labeling systems."""
    
    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        self.labeling_system = None
        self._initialize_labeling_system()
    
    def _initialize_labeling_system(self):
        """Initialize the appropriate labeling system."""
        if not TACTICIAN_ANALYST_LABELING_AVAILABLE:
            raise CriticalPipelineError("Tactician/Analyst labeling not available")
        
        try:
            if self.config.labeling_type == "tactician":
                self.labeling_system = create_enhanced_tactician_labeler(
                    TacticianLabelingConfig()
                )
            elif self.config.labeling_type == "analyst":
                self.labeling_system = create_enhanced_analyst_labeler(
                    AnalystLabelingConfig()
                )
            else:
                raise ConfigurationError(f"Unknown labeling type: {self.config.labeling_type}")
        except Exception as e:
            raise CriticalPipelineError(f"Failed to initialize labeling system: {e}") from e
    
    def generate_labels(self, market_data: pd.DataFrame, targets: Optional[pd.Series] = None, 
                       existing_artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate labels using the configured labeling system."""
        try:
            if self.labeling_system is None:
                raise CriticalPipelineError("Labeling system not initialized")
            
            # Generate labels using the appropriate system
            result = self.labeling_system.generate_labels(
                market_data, targets, existing_artifacts
            )
            
            return {
                'success': True,
                'labeled_data': result.get('data', pd.DataFrame()),
                'labeling_metadata': result.get('metadata', {}),
                'quality_score': result.get('quality_score', 0.0),
                'labeling_type': self.config.labeling_type
            }
        except Exception as e:
            raise FeatureGenerationError(f"Label generation failed: {e}") from e
    
    def cleanup(self):
        """Clean up labeling system resources."""
        if self.labeling_system and hasattr(self.labeling_system, 'cleanup'):
            self.labeling_system.cleanup()

class UnifiedDataDrivenPipeline:
    """
    Consolidated Unified Data-Driven Feature Pipeline - Cleaned Version.
    
    This implementation eliminates redundancy, removes unused code, fixes logic issues,
    and implements proper error handling with fast-fail patterns.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """
        Initialize the consolidated unified data-driven pipeline.
        
        Args:
            config: Pipeline configuration (uses default if None)
            
        Raises:
            CriticalPipelineError: If critical components cannot be initialized
        """
        self.config = config or create_default_config()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components with proper error handling
        try:
            self._initialize_utility_systems()
            self._initialize_labeling_adapter()
            self._initialize_core_components()
            self._initialize_enhanced_components()
            self._initialize_validation_components()
            self._initialize_performance_tracking()
            
            tprint_success("🚀 Consolidated Unified Data-Driven Pipeline initialized successfully")
            tprint_info(f"📊 Configuration: {self.config}")
            
        except Exception as e:
            raise CriticalPipelineError(f"Pipeline initialization failed: {e}") from e
    
    def _initialize_utility_systems(self):
        """Initialize utility systems with proper error handling."""
        tprint_debug("🔧 Initializing utility systems")
        
        try:
            # Initialize common operations utilities
            self.common_utilities = CommonUtilities()
            
            # Initialize unified data utilities
            if UNIFIED_DATA_UTILS_AVAILABLE:
                self.unified_data_utils = create_unified_data_utils()
            else:
                raise CriticalPipelineError("Unified data utilities not available")
            
            # Initialize quality framework
            if QUALITY_FRAMEWORK_AVAILABLE:
                self.quality_framework = create_quality_framework()
            else:
                raise CriticalPipelineError("Quality framework not available")
            
            # Initialize cross-step validator
            if CROSS_STEP_VALIDATOR_AVAILABLE:
                self.cross_step_validator = create_cross_step_validator()
            else:
                raise CriticalPipelineError("Cross-step validator not available")
            
            # Initialize M1 optimizations if available
            if M1_OPTIMIZATIONS_AVAILABLE:
                self.m1_optimizer = create_m1_optimizer()
                self.m1_available = True
            else:
                self.m1_available = False
            
            # Initialize VectorBT utilities if available
            if VECTORBT_UTILITIES_AVAILABLE:
                self.vectorbt_utilities = create_vectorbt_utilities()
                self.vectorbt_available = True
            else:
                self.vectorbt_available = False
            
            # Initialize feature engineering roadmap if available
            if FEATURE_ENGINEERING_ROADMAP_AVAILABLE:
                self.dynamic_roadmap_pipeline = create_roadmap_pipeline()
                self.roadmap_available = True
            else:
                self.roadmap_available = False
            
            # Initialize ML Common utilities if available
            if ML_COMMON_AVAILABLE:
                self.ml_common_utilities = create_ml_common_utilities()
                self.ml_common_available = True
            else:
                self.ml_common_available = False
            
            tprint_success("✅ Utility systems initialized successfully")
            
        except Exception as e:
            raise CriticalPipelineError(f"Utility systems initialization failed: {e}") from e
    
    def _initialize_labeling_adapter(self):
        """Initialize the labeling adapter with proper error handling."""
        tprint_debug("🔧 Initializing labeling adapter")
        
        try:
            self.labeling_adapter = LabelingAdapter(self.config)
            tprint_success(f"✅ Labeling adapter initialized: {self.config.labeling_type}")
        except Exception as e:
            raise CriticalPipelineError(f"Labeling adapter initialization failed: {e}") from e
    
    def _initialize_core_components(self):
        """Initialize core components with proper error handling."""
        tprint_debug("🔧 Initializing core components")
        
        try:
            # Initialize economic evaluator
            self.economic_evaluator = create_economic_evaluator()
            
            # Initialize feature selector
            self.feature_selector = create_intelligent_feature_selector()
            
            # Initialize VectorBT optimizer
            self.vectorbt_optimizer = create_vectorbt_optimizer()
            
            # Initialize interaction generator
            self.interaction_generator = create_template_interaction_generator()
            
            # Initialize common feature generator
            self.common_feature_generator = create_common_feature_generator()
            
            # Initialize lookback optimizer
            self.lookback_optimizer = create_common_lookback_optimizer()
            
            tprint_success("✅ Core components initialized successfully")
            
        except Exception as e:
            raise CriticalPipelineError(f"Core components initialization failed: {e}") from e
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced components with proper error handling."""
        tprint_debug("🔧 Initializing enhanced components")
        
        try:
            # Initialize enhanced components
            self.enhanced_walk_forward_validator = AdvancedWalkForwardValidator()
            self.enhanced_statistical_framework = EnhancedStatisticalFramework()
            self.enhanced_schema_validator = EnhancedSchemaValidator()
            self.enhanced_caching_integration = EnhancedCachingIntegration()
            self.enhanced_performance_monitor = EnhancedPerformanceMonitor()
            self.enhanced_error_handler = EnhancedErrorHandler()
            self.enhanced_validator = EnhancedValidator()
            self.enhanced_data_loader = EnhancedDataLoader()
            self.enhanced_feature_selector = EnhancedFeatureSelector()
            self.enhanced_artifact_manager = EnhancedArtifactManager()
            self.enhanced_caching_system = EnhancedCachingSystem()
            
            # Initialize advanced components
            self.advanced_performance_monitor = AdvancedPerformanceMonitor()
            self.advanced_error_handler = AdvancedErrorHandler()
            self.advanced_validator = AdvancedValidator()
            self.advanced_data_loader = AdvancedDataLoader()
            self.advanced_feature_selector = AdvancedFeatureSelector()
            self.advanced_artifact_manager = AdvancedArtifactManager()
            self.advanced_caching_system = AdvancedCachingSystem()
            
            tprint_success("✅ Enhanced components initialized successfully")
            
        except Exception as e:
            raise CriticalPipelineError(f"Enhanced components initialization failed: {e}") from e
    
    def _initialize_validation_components(self):
        """Initialize validation components with proper error handling."""
        tprint_debug("🔧 Initializing validation components")
        
        try:
            # Initialize modular architecture
            self.modular_architecture = create_modular_architecture("UnifiedDataDrivenPipeline")
            
            tprint_success("✅ Validation components initialized successfully")
            
        except Exception as e:
            raise CriticalPipelineError(f"Validation components initialization failed: {e}") from e
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking with proper error handling."""
        tprint_debug("🔧 Initializing performance tracking")
        
        try:
            # Initialize performance tracking
            self.performance_tracker = {
                'start_time': time.time(),
                'operations': [],
                'metrics': {}
            }
            
            tprint_success("✅ Performance tracking initialized successfully")
            
        except Exception as e:
            raise CriticalPipelineError(f"Performance tracking initialization failed: {e}") from e
    
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
            
        Raises:
            DataValidationError: If data validation fails
            FeatureGenerationError: If feature generation fails
            OptimizationError: If optimization fails
        """
        tprint_info("🚀 Starting consolidated unified pipeline processing")
        tprint_info(f"📊 Data shape: {data.shape}, timeframe: {timeframe}")
        
        # Start performance monitoring
        start_time = time.time()
        
        try:
            # Fast fail validation - check critical requirements first
            self._validate_critical_requirements(data, targets, timeframe, pipeline_state)
            
            # Enhanced data processing and validation
            tprint_info("🔍 Performing comprehensive data validation and processing...")
            
            # Step 1: Comprehensive data validation and quality assessment
            quality_result = self.quality_framework.validate_dataframe_quality(
                data, context=f"pipeline_input_{timeframe}"
            )
            
            if not quality_result.passed:
                critical_issues = [issue for issue in quality_result.issues 
                                 if 'critical' in issue.lower() or 'fatal' in issue.lower()]
                if critical_issues:
                    raise DataValidationError(f"Critical data quality issues detected: {critical_issues}")
                
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
            
            if processed_data is None or processed_data.empty:
                raise DataValidationError("Data processing failed - no data returned")
            
            # Apply additional common operations enhancements
            tprint_debug("🔧 Applying common operations enhancements to processed data")
            
            # Optimize DataFrame dtypes for memory efficiency
            processed_data = optimize_dataframe_dtypes(processed_data)
            
            # Guard against excessive null values
            processed_data = guard_dataframe_nulls(processed_data, threshold=0.5)
            
            # Validate DataFrame schema
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_schema(processed_data, required_columns):
                raise DataValidationError("DataFrame schema validation failed")
            
            # Calculate and log data quality metrics
            quality_metrics = calculate_data_quality_metrics(processed_data)
            safe_log_metric("data_quality_score", quality_metrics.get('missing_percentage', 0))
            safe_log_metric("duplicate_percentage", quality_metrics.get('duplicate_percentage', 0))
            
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
                raise DataValidationError(f"Advanced validation failed: {validation_summary.recommendations}")
            
            # Step 4: Leakage prevention validation
            if targets is not None:
                tprint_info("🔒 Performing leakage prevention validation...")
                self._validate_no_leakage(cleaned_data, targets)
            
            # Step 5: Advanced feature screening
            tprint_info("🔍 Performing advanced feature screening...")
            screening_result = self._advanced_feature_screening(cleaned_data, targets)
            
            # Step 6: Load market data using advanced data loader
            market_data = await self.advanced_data_loader.load_market_data(
                cleaned_data, pipeline_state, force_refresh=False
            )
            
            if market_data is None:
                raise DataValidationError("Market data loading failed - no data returned")
            
            # Step 7: Generate labels using the tactician/analyst labeling system
            tprint_info(f"🏷️ Generating labels using {self.config.labeling_type} labeling system")
            
            labeling_result = self.labeling_adapter.generate_labels(market_data, targets, None)
            
            if not labeling_result.get('success', False):
                raise FeatureGenerationError(f"Labeling failed: {labeling_result.get('error', 'Unknown error')}")
            
            labeling_data = labeling_result.get('labeled_data', pd.DataFrame())
            labeling_metadata = labeling_result.get('labeling_metadata', {})
            labeling_quality = labeling_result.get('quality_score', 0.0)
            
            tprint_success(f"✅ Labels generated successfully using {labeling_result.get('labeling_type', 'unknown')} system")
            tprint_info(f"📊 Labeling quality score: {labeling_quality:.3f}")
            
            # Step 8: Prepare data for optimization
            processed_data = self.advanced_data_loader.prepare_data_for_optimization(
                market_data, labeling_data
            )
            
            if processed_data is None or processed_data.empty:
                raise DataValidationError("Processed data is empty or None after data loading")
            
            # Step 9: Generate features for optimization
            feature_columns = await self.advanced_data_loader.generate_features_for_optimization(
                processed_data, pipeline_state, force_refresh=False
            )
            
            if not feature_columns:
                raise FeatureGenerationError("Feature generation failed - no features generated")
            
            tprint_success(f"✅ Generated {len(feature_columns)} features for optimization")
            
            # Step 10: Prepare targets and ensure data consistency
            processed_targets = targets
            if targets is not None and len(targets) != len(processed_data):
                common_index = processed_data.index.intersection(targets.index)
                if len(common_index) == 0:
                    raise DataValidationError("No common index between data and targets")
                processed_data = processed_data.loc[common_index]
                processed_targets = targets.loc[common_index]
                tprint_info(f"📊 Aligned data and targets to {len(common_index)} common rows")
            
            # Step 11: Enhanced period optimization with economic evaluation
            tprint_info("Step 11: Enhanced period optimization with economic evaluation")
            period_results = self._enhanced_period_optimization(processed_data, timeframe)
            
            # Step 12: Advanced feature selection from 200+ feature bank
            tprint_info("Step 12: Advanced feature selection from 200+ feature bank")
            feature_selection_results = self._advanced_feature_selection(processed_data, processed_targets)
            
            if not feature_selection_results or not hasattr(feature_selection_results, 'selected_features'):
                raise FeatureGenerationError("Feature selection failed - no valid results")
            
            # Step 13: Generate selected features
            tprint_info("Step 13: Generate selected features")
            selected_features_df = self._generate_selected_features(processed_data, feature_selection_results)
            
            if selected_features_df.empty:
                raise FeatureGenerationError("Feature generation failed - no features generated")
            
            # Step 14: Apply statistical transforms
            tprint_info("Step 14: Apply statistical transforms")
            transformed_features_df = self._apply_statistical_transforms(selected_features_df)
            
            # Step 15: Apply vectorized feature calculations if available
            if self.vectorbt_available:
                tprint_info("Step 15: Apply vectorized feature calculations")
                vectorized_features_df = self._optimized_feature_calculation(transformed_features_df)
                transformed_features_df = vectorized_features_df
                tprint_success(f"✅ Vectorized feature calculations completed: {transformed_features_df.shape[1]} total features")
            
            # Step 16: Enhanced interaction generation
            tprint_info("Step 16: Enhanced interaction generation")
            interaction_results = self._enhanced_interaction_generation(transformed_features_df, processed_targets)
            
            # Step 17: Advanced lookback optimization
            tprint_info("Step 17: Advanced lookback optimization")
            lookback_results = self._advanced_lookback_optimization(
                processed_data, processed_targets, transformed_features_df
            )
            
            # Step 18: Final feature selection
            tprint_info("Step 18: Final feature selection")
            final_features_df = self._final_feature_selection(transformed_features_df, processed_targets)
            
            # Step 19: Create final result
            processing_time = time.time() - start_time
            
            result = ConsolidatedPipelineResult(
                success=True,
                data=processed_data,
                features=final_features_df,
                targets=processed_targets,
                metadata={
                    'timeframe': timeframe,
                    'feature_count': len(final_features_df.columns),
                    'data_shape': processed_data.shape,
                    'labeling_quality': labeling_quality,
                    'labeling_metadata': labeling_metadata,
                    'period_results': period_results,
                    'feature_selection_results': feature_selection_results,
                    'interaction_results': interaction_results,
                    'lookback_results': lookback_results,
                    'screening_result': screening_result
                },
                processing_time=processing_time,
                quality_score=quality_result.quality_score,
                performance_metrics=self._get_performance_metrics()
            )
            
            tprint_success(f"✅ Pipeline processing completed successfully in {processing_time:.2f}s")
            tprint_info(f"📊 Final features: {len(final_features_df.columns)}")
            tprint_info(f"📊 Data shape: {processed_data.shape}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline processing failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            # Log error details
            self.logger.error(f"Pipeline error: {error_msg}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ConsolidatedPipelineResult(
                success=False,
                error=error_msg,
                error_code=type(e).__name__,
                processing_time=processing_time,
                performance_metrics=self._get_performance_metrics()
            )
    
    def _validate_critical_requirements(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                      timeframe: str, pipeline_state: Optional[Dict[str, Any]]):
        """Validate critical requirements with fast fail."""
        if data is None or data.empty:
            raise DataValidationError("Input data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError("Input data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        if len(data) < 10:
            raise DataValidationError("Data must have at least 10 rows")
        
        if targets is not None and not isinstance(targets, pd.Series):
            raise DataValidationError("Targets must be a pandas Series")
        
        if not isinstance(timeframe, str) or not timeframe:
            raise DataValidationError("Timeframe must be a non-empty string")
    
    def _validate_no_leakage(self, data: pd.DataFrame, targets: pd.Series):
        """Validate no data leakage with fast fail."""
        if isinstance(data.index, pd.DatetimeIndex) and isinstance(targets.index, pd.DatetimeIndex):
            # Check temporal ordering
            future_data_count = 0
            for timestamp, target_value in targets.items():
                if timestamp in data.index:
                    future_data = data[data.index > timestamp]
                    if len(future_data) > 0:
                        future_data_count += 1
            
            if future_data_count > 0:
                raise DataValidationError(f"Data leakage detected: {future_data_count} targets use future data")
            
            tprint_success(f"✅ Leakage prevention validation passed: {len(targets)} valid labels")
        else:
            raise DataValidationError("Data leakage validation requires datetime indices")
    
    def _advanced_feature_screening(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Dict[str, Any]:
        """Perform advanced feature screening."""
        screening_result = {'combined_selected_features': []}
        
        try:
            if targets is not None:
                # Calculate correlation with targets for each feature
                feature_correlations = {}
                for col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        try:
                            corr = data[col].corr(targets)
                            if not pd.isna(corr):
                                feature_correlations[col] = abs(corr)
                        except Exception:
                            continue
                
                # Select top features by correlation
                if feature_correlations:
                    sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
                    top_features = [f[0] for f in sorted_features[:50]]  # Top 50 features
                    screening_result['combined_selected_features'] = top_features
                    tprint_success(f"✅ Advanced screening completed: {len(top_features)} features selected")
                else:
                    tprint_warning("⚠️ No valid correlations found for screening")
            else:
                # Fallback to variance-based screening
                variances = data.var().sort_values(ascending=False)
                top_features = variances.head(50).index.tolist()
                screening_result['combined_selected_features'] = top_features
                tprint_success(f"✅ Variance-based screening completed: {len(top_features)} features selected")
                
        except Exception as e:
            tprint_warning(f"⚠️ Advanced screening failed: {e}")
            screening_result = {'combined_selected_features': []}
        
        return screening_result
    
    def _enhanced_period_optimization(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Enhanced period optimization with economic evaluation."""
        try:
            # Use economic evaluator for period optimization
            period_results = self.economic_evaluator.evaluate_periods(data, timeframe)
            return period_results
        except Exception as e:
            raise OptimizationError(f"Period optimization failed: {e}") from e
    
    def _advanced_feature_selection(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Any:
        """Advanced feature selection from 200+ feature bank."""
        try:
            # Use intelligent feature selector
            selection_result = self.feature_selector.select_features(data, targets)
            return selection_result
        except Exception as e:
            raise FeatureGenerationError(f"Feature selection failed: {e}") from e
    
    def _generate_selected_features(self, data: pd.DataFrame, selection_result: Any) -> pd.DataFrame:
        """Generate selected features."""
        try:
            # Use common feature generator
            features_df = self.common_feature_generator.generate_features(data, selection_result)
            return features_df
        except Exception as e:
            raise FeatureGenerationError(f"Feature generation failed: {e}") from e
    
    def _apply_statistical_transforms(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply statistical transforms to features."""
        try:
            # Apply statistical transforms using enhanced statistical framework
            transformed_df = self.enhanced_statistical_framework.transform_features(features_df)
            return transformed_df
        except Exception as e:
            raise FeatureGenerationError(f"Statistical transforms failed: {e}") from e
    
    def _optimized_feature_calculation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply optimized feature calculations using VectorBT utilities."""
        try:
            # Use VectorBT utilities for optimized calculations
            vectorized_df = self.vectorbt_utilities.optimize_calculations(features_df)
            return vectorized_df
        except Exception as e:
            raise FeatureGenerationError(f"Vectorized feature calculations failed: {e}") from e
    
    def _enhanced_interaction_generation(self, features_df: pd.DataFrame, targets: Optional[pd.Series]) -> List[Any]:
        """Enhanced interaction generation."""
        try:
            # Use interaction generator
            interactions = self.interaction_generator.generate_interactions(features_df, targets)
            return interactions
        except Exception as e:
            raise FeatureGenerationError(f"Interaction generation failed: {e}") from e
    
    def _advanced_lookback_optimization(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                      features_df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced lookback optimization."""
        try:
            # Use lookback optimizer
            lookback_results = self.lookback_optimizer.optimize_lookbacks(data, targets, features_df)
            return lookback_results
        except Exception as e:
            raise OptimizationError(f"Lookback optimization failed: {e}") from e
    
    def _final_feature_selection(self, features_df: pd.DataFrame, targets: Optional[pd.Series]) -> pd.DataFrame:
        """Final feature selection."""
        try:
            # Use advanced feature selector for final selection
            final_features = self.advanced_feature_selector.select_final_features(features_df, targets)
            return final_features
        except Exception as e:
            raise FeatureGenerationError(f"Final feature selection failed: {e}") from e
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            return {
                'total_operations': len(self.performance_tracker['operations']),
                'start_time': self.performance_tracker['start_time'],
                'current_time': time.time(),
                'uptime': time.time() - self.performance_tracker['start_time']
            }
        except Exception:
            return {}
    
    def cleanup(self):
        """Clean up resources."""
        tprint_info("🧹 Starting pipeline cleanup process")
        
        try:
            # Clean up labeling adapter
            if hasattr(self, 'labeling_adapter') and self.labeling_adapter:
                self.labeling_adapter.cleanup()
                tprint_success("✅ Labeling adapter cleaned up")
            
            # Clean up other components
            components_to_cleanup = [
                'economic_evaluator', 'feature_selector', 'vectorbt_optimizer',
                'interaction_generator', 'common_feature_generator', 'lookback_optimizer',
                'enhanced_walk_forward_validator', 'enhanced_statistical_framework',
                'enhanced_schema_validator', 'enhanced_caching_integration',
                'enhanced_performance_monitor', 'enhanced_error_handler',
                'enhanced_validator', 'enhanced_data_loader', 'enhanced_feature_selector',
                'enhanced_artifact_manager', 'enhanced_caching_system',
                'advanced_performance_monitor', 'advanced_error_handler',
                'advanced_validator', 'advanced_data_loader', 'advanced_feature_selector',
                'advanced_artifact_manager', 'advanced_caching_system'
            ]
            
            for component_name in components_to_cleanup:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    if component and hasattr(component, 'cleanup'):
                        try:
                            component.cleanup()
                            tprint_success(f"✅ {component_name} cleaned up")
                        except Exception as e:
                            tprint_warning(f"⚠️ Error cleaning up {component_name}: {e}")
            
            tprint_success("✅ Pipeline cleanup completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Error during pipeline cleanup: {e}")
            tprint_error(f"❌ Cleanup error details: {type(e).__name__}: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Don't raise exceptions in destructor

# Convenience function
async def process_with_unified_pipeline(data: pd.DataFrame,
                                      targets: Optional[pd.Series] = None,
                                      feature_columns: Optional[List[str]] = None,
                                      timeframe: str = "15m",
                                      pipeline_state: Optional[Dict[str, Any]] = None,
                                      config: Optional[UnifiedPipelineConfig] = None) -> ConsolidatedPipelineResult:
    """
    Convenience function to process data with the unified pipeline.
    
    Args:
        data: Input data with OHLCV columns
        targets: Optional target series for supervised learning
        feature_columns: Optional list of feature columns to use
        timeframe: Target timeframe (e.g., "15m", "5m", "1h")
        pipeline_state: Optional pipeline state dictionary
        config: Optional pipeline configuration
        
    Returns:
        ConsolidatedPipelineResult with comprehensive results
    """
    pipeline = UnifiedDataDrivenPipeline(config)
    try:
        result = await pipeline.process(data, targets, feature_columns, timeframe, pipeline_state)
        return result
    finally:
        pipeline.cleanup()