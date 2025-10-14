"""
Core Unified Data-Driven Pipeline

This module contains the main UnifiedDataDrivenPipeline class with a clean,
focused implementation that eliminates redundancy and implements fast fail patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from datetime import datetime

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Required tprint utilities not available: {e}") from e

# Import core components
from .config import UnifiedPipelineConfig, create_default_config
from .economic_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig, 
    EconomicPeriodEvaluationResult, create_economic_evaluator
)
from .intelligent_feature_selector import (
    IntelligentFeatureSelector, FeatureSelectionConfig, 
    FeatureSelectionResult, create_intelligent_feature_selector
)
from .vectorbt_optimizer import (
    VectorBTOptimizer, VectorBTConfig, 
    create_vectorbt_optimizer
)
from .template_interaction_generator import (
    TemplateInteractionGenerator, TemplateConfig, 
    create_template_interaction_generator
)

# Import required utilities - fast fail if not available
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
    raise ImportError(f"Required feature generation utilities not available: {e}") from e

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
    raise ImportError(f"Required features common utilities not available: {e}") from e

try:
    from src.tactician_analyst_labeling import (
        create_enhanced_analyst_labeler, create_enhanced_tactician_labeler,
        LabelDefinitionType
    )
    TACTICIAN_ANALYST_LABELING_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Required tactician/analyst labeling not available: {e}") from e

try:
    from src.utils.ml_common.integrated_analysis_pipeline import (
        IntegratedAnalysisPipeline, IntegratedAnalysisConfig
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"ML Common utilities are required but not available: {e}") from e

try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig
    VECTORBT_UTILITIES_AVAILABLE = True
    VECTORBT_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"VectorBT utilities are required but not available: {e}") from e

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
except ImportError as e:
    raise ImportError(f"Feature engineering roadmap utilities are required but not available: {e}") from e

try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Caching utilities are required but not available: {e}") from e

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        safe_matrix_multiply,
        optimize_dataframe,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Matrix operations are required but not available: {e}") from e

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedPipelineResult:
    """Result from the consolidated unified pipeline."""
    
    # Core results
    selected_features: List[str]
    feature_importance: Dict[str, float]
    objective_values: Dict[str, float]
    optimal_periods: List[int]
    period_scores: Dict[int, float]
    
    # Economic evaluation
    economic_evaluation_results: Optional[Dict[str, Any]] = None
    
    # Feature selection metrics
    feature_selection_metrics: Dict[str, Any] = None
    
    # Generated features
    generated_interactions: List[str] = None
    interaction_metrics: Dict[str, Any] = None
    htf_interactions: List[str] = None
    htf_metrics: Dict[str, Any] = None
    
    # Lookback optimization
    optimized_lookbacks: Dict[str, int] = None
    lookback_metrics: Dict[str, Any] = None
    
    # Enhanced results
    long_pipeline_results: Dict[str, Any] = None
    short_pipeline_results: Dict[str, Any] = None
    lookback_optimization_method: str = 'unknown'
    execution_mode: str = 'unknown'
    nested_cv_applied: bool = False
    outer_fold_count: int = 0
    
    # Feature metadata
    feature_lag_metadata: Dict[str, Any] = None
    cross_timeframe_features: List[str] = None
    interaction_features: List[str] = None
    no_features: List[str] = None
    comparison_features: List[str] = None
    enhanced_feature_metrics: Dict[str, Any] = None
    
    # Performance metrics
    processing_time: float = 0.0
    n_cv_splits: int = 0
    n_candidates_evaluated: int = 0
    out_of_sample_sharpe: float = 0.0
    max_drawdown: float = 0.0
    stability_score: float = 0.0
    diversity_score: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    vectorbt_operations: int = 0
    pandas_fallbacks: int = 0
    cache_hit_rate: float = 0.0
    optimization_iterations: int = 0
    convergence_achieved: bool = False
    feature_diversity_score: float = 0.0
    interaction_utility_scores: Dict[str, float] = None
    lookback_optimization_metrics: Dict[str, Any] = None
    performance_monitoring_data: Dict[str, Any] = None
    
    # Configuration and status
    config: Optional[UnifiedPipelineConfig] = None
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.warnings is None:
            self.warnings = []
        
        # Validate critical fields
        if not self.selected_features:
            self.warnings.append("No features were selected")
        
        if self.processing_time <= 0:
            self.warnings.append("Invalid processing time")
        
        if not self.success and not self.error_message:
            self.error_message = "Unknown error occurred"


class LabelingAdapter:
    """Adapter for switching between different labeling systems."""
    
    def __init__(self, config: 'UnifiedPipelineConfig'):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_labeling_components()
    
    def _initialize_labeling_components(self):
        """Initialize the appropriate labeling components based on configuration."""
        if not TACTICIAN_ANALYST_LABELING_AVAILABLE:
            raise ImportError("Tactician/Analyst labeling is required but not available. Please install required dependencies.")
        
        self.labeling_system = "tactician_analyst"
        
        if self.config.labeling_type == "analyst":
            tprint_info("🏷️ Initializing Analyst labeling system")
            self.labeler = create_enhanced_analyst_labeler()
            self.labeling_type = LabelDefinitionType.ANALYST
        elif self.config.labeling_type == "tactician":
            tprint_info("🏷️ Initializing Tactician labeling system")
            self.labeler = create_enhanced_tactician_labeler()
            self.labeling_type = LabelDefinitionType.TACTICIAN
        else:
            raise ValueError(f"Invalid labeling type: {self.config.labeling_type}. Must be 'analyst' or 'tactician'")
    
    def generate_labels(self, market_data: pd.DataFrame, targets: Optional[pd.Series] = None, 
                       existing_artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate labels using the configured labeling system."""
        return self._generate_tactician_analyst_labels(market_data, targets, existing_artifacts)
    
    def _generate_tactician_analyst_labels(self, market_data: pd.DataFrame, targets: Optional[pd.Series] = None, 
                                          existing_artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate labels using tactician/analyst labeling system."""
        try:
            tprint_info(f"🏷️ Generating {self.config.labeling_type} labels using volatility-aware labeler")
            
            # Check for existing artifacts first
            if existing_artifacts and self._is_artifact_compatible(existing_artifacts):
                tprint_info("📦 Using existing labeling artifacts")
                return self._process_existing_artifacts(existing_artifacts)
            
            # Generate new labels using the volatility-aware labeler
            labeling_result = self.labeler.generate_labels(market_data)
            
            if labeling_result.success:
                tprint_success(f"✅ {self.config.labeling_type.title()} labeling completed successfully")
                
                # Convert to the expected format
                result = {
                    'success': True,
                    'labeled_data': labeling_result.labeled_data,
                    'confidence_scores': labeling_result.confidence_scores,
                    'labeling_metadata': labeling_result.metadata,
                    'labeling_type': self.config.labeling_type,
                    'labeling_system': 'tactician_analyst',
                    'quality_score': labeling_result.quality_score,
                    'feature_importance': labeling_result.feature_importance,
                    'artifacts': {
                        'labeling_result': labeling_result,
                        'generated_at': pd.Timestamp.now(),
                        'labeling_type': self.config.labeling_type,
                        'market_data_shape': market_data.shape
                    }
                }
                
                return result
            else:
                error_msg = f"{self.config.labeling_type.title()} labeling failed: {labeling_result.error_message}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"{self.config.labeling_type.title()} labeling error: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _is_artifact_compatible(self, artifacts: Dict[str, Any]) -> bool:
        """Check if existing artifacts are compatible with current configuration."""
        try:
            # Check if artifacts contain the expected labeling type
            artifact_type = artifacts.get('labeling_type', '')
            if artifact_type != self.config.labeling_type:
                tprint_warning(f"⚠️ Artifact labeling type ({artifact_type}) doesn't match current ({self.config.labeling_type})")
                return False
            
            # Check if artifacts are recent enough (within 24 hours)
            generated_at = artifacts.get('generated_at')
            if generated_at:
                if isinstance(generated_at, str):
                    generated_at = pd.Timestamp(generated_at)
                age_hours = (pd.Timestamp.now() - generated_at).total_seconds() / 3600
                if age_hours > 24:
                    tprint_warning(f"⚠️ Artifacts are {age_hours:.1f} hours old, regenerating")
                    return False
            
            tprint_info("✅ Existing artifacts are compatible")
            return True
            
        except Exception as e:
            tprint_warning(f"⚠️ Error checking artifact compatibility: {e}")
            return False
    
    def _process_existing_artifacts(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Process existing artifacts and return in expected format."""
        try:
            # Extract labeling result from artifacts
            if 'labeling_result' in artifacts:
                labeling_result = artifacts['labeling_result']
            else:
                # Reconstruct from individual components
                labeling_result = type('LabelingResult', (), {
                    'success': True,
                    'labeled_data': artifacts.get('labeled_data', pd.DataFrame()),
                    'confidence_scores': artifacts.get('confidence_scores', {}),
                    'metadata': artifacts.get('labeling_metadata', {}),
                    'quality_score': artifacts.get('quality_score', 0.5),
                    'feature_importance': artifacts.get('feature_importance', {}),
                    'error_message': None
                })()
            
            return {
                'success': True,
                'labeled_data': labeling_result.labeled_data,
                'confidence_scores': labeling_result.confidence_scores,
                'labeling_metadata': labeling_result.metadata,
                'labeling_type': self.config.labeling_type,
                'labeling_system': 'tactician_analyst',
                'quality_score': labeling_result.quality_score,
                'feature_importance': labeling_result.feature_importance,
                'artifacts': artifacts
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error processing existing artifacts: {e}")
            # Fall back to generating new labels
            return self._generate_tactician_analyst_labels(
                artifacts.get('market_data', pd.DataFrame())
            )


class UnifiedDataDrivenPipeline:
    """
    Clean, focused Unified Data-Driven Feature Pipeline.
    
    This implementation eliminates redundancy, implements fast fail patterns,
    and provides a clean, maintainable architecture.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize the unified data-driven pipeline."""
        self.config = config or create_default_config()
        
        # Initialize components
        self._initialize_labeling_adapter()
        self._initialize_core_components()
        self._initialize_utility_systems()
        
        tprint_success("🚀 Unified Data-Driven Pipeline initialized")
        tprint_info(f"📊 Configuration: {self.config}")
    
    def _initialize_labeling_adapter(self):
        """Initialize the labeling adapter for tactician/analyst labeling."""
        tprint_debug("🔧 Initializing labeling adapter")
        
        try:
            self.labeling_adapter = LabelingAdapter(self.config)
            tprint_success(f"✅ Labeling adapter initialized: {self.config.labeling_system}/{self.config.labeling_type}")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize labeling adapter: {e}")
            raise RuntimeError(f"Labeling adapter initialization failed: {e}") from e
    
    def _initialize_core_components(self):
        """Initialize core pipeline components."""
        tprint_debug("Initializing core components")
        
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
    
    def _initialize_utility_systems(self):
        """Initialize utility systems."""
        tprint_debug("🔧 Initializing utility systems")
        
        try:
            # Initialize common operations utilities
            utility_config = UtilityConfig(
                enable_parallel_processing=True,
                max_workers=4,
                memory_efficient=True,
                enable_caching=True
            )
            self.unified_data_utils = get_utility_container(utility_config)
            tprint_success("✅ Common operations utilities initialized")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize utility systems: {e}")
            raise RuntimeError(f"Utility systems initialization failed: {e}") from e
    
    def process(self, data: pd.DataFrame, 
                targets: Optional[pd.Series] = None,
                feature_columns: Optional[List[str]] = None,
                timeframe: str = "15m",
                pipeline_state: Optional[Dict[str, Any]] = None) -> ConsolidatedPipelineResult:
        """
        Process data through the unified pipeline.
        
        Args:
            data: Input data with OHLCV columns
            targets: Optional target series for supervised learning
            feature_columns: Optional list of feature columns to use
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            pipeline_state: Optional pipeline state dictionary
            
        Returns:
            ConsolidatedPipelineResult with comprehensive results
        """
        tprint_info("🚀 Starting unified pipeline processing")
        tprint_info(f"📊 Data shape: {data.shape}, timeframe: {timeframe}")
        
        start_time = time.time()
        
        try:
            # Fast fail validation
            self._validate_inputs(data, targets, timeframe)
            
            # Process data
            processed_data = self._process_data(data, timeframe)
            
            # Generate features
            features = self._generate_features(processed_data, feature_columns)
            
            # Select features
            selected_features = self._select_features(features, targets)
            
            # Create result
            result = ConsolidatedPipelineResult(
                selected_features=selected_features,
                feature_importance={},
                objective_values={},
                optimal_periods=[],
                period_scores={},
                processing_time=time.time() - start_time,
                success=True,
                config=self.config
            )
            
            tprint_success("✅ Pipeline processing completed successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Pipeline processing failed: {e}")
            return ConsolidatedPipelineResult(
                selected_features=[],
                feature_importance={},
                objective_values={},
                optimal_periods=[],
                period_scores={},
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                config=self.config
            )
    
    def _validate_inputs(self, data: pd.DataFrame, targets: Optional[pd.Series], timeframe: str):
        """Validate input parameters with fast fail."""
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if targets is not None and len(targets) != len(data):
            raise ValueError("Targets length must match data length")
        
        if not isinstance(timeframe, str):
            raise TypeError("Timeframe must be a string")
        
        tprint_success("✅ Input validation passed")
    
    def _process_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Process and validate data."""
        tprint_info("🔍 Processing data")
        
        try:
            # Use unified data utilities for processing
            processed_data, _ = self.unified_data_utils.process_and_validate(
                data=data,
                validate_quality=True,
                clean_missing_values=True,
                detect_outliers=True,
                optimize_dtypes=True,
                regularize_timestamps=True,
                context=f"pipeline_processing_{timeframe}",
                symbol='ETHUSDT',  # Default symbol
                exchange='binance',  # Default exchange
                timeframe=timeframe
            )
            
            tprint_success("✅ Data processing completed")
            return processed_data
            
        except Exception as e:
            tprint_error(f"❌ Data processing failed: {e}")
            raise RuntimeError(f"Data processing failed: {e}") from e
    
    def _generate_features(self, data: pd.DataFrame, feature_columns: Optional[List[str]]) -> pd.DataFrame:
        """Generate features from processed data."""
        tprint_info("🔧 Generating features")
        
        try:
            # Use template interaction generator for feature generation
            feature_result = self.template_interaction_generator.generate_features(
                data=data,
                feature_columns=feature_columns
            )
            
            if feature_result.success:
                tprint_success("✅ Feature generation completed")
                return feature_result.features
            else:
                raise RuntimeError(f"Feature generation failed: {feature_result.error_message}")
                
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            raise RuntimeError(f"Feature generation failed: {e}") from e
    
    def _select_features(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[str]:
        """Select optimal features."""
        tprint_info("🎯 Selecting features")
        
        try:
            if targets is not None:
                # Use intelligent feature selector
                selection_result = self.intelligent_feature_selector.select_features(
                    features=features,
                    targets=targets
                )
                
                if selection_result.success:
                    tprint_success("✅ Feature selection completed")
                    return selection_result.selected_features
                else:
                    raise RuntimeError(f"Feature selection failed: {selection_result.error_message}")
            else:
                # Return all features if no targets
                tprint_info("ℹ️ No targets provided, returning all features")
                return list(features.columns)
                
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            raise RuntimeError(f"Feature selection failed: {e}") from e
    
    def cleanup(self):
        """Clean up resources."""
        tprint_info("🧹 Starting pipeline cleanup")
        
        try:
            # Clean up components
            cleanup_components = [
                ('economic_evaluator', 'Economic evaluator'),
                ('intelligent_feature_selector', 'Intelligent feature selector'),
                ('vectorbt_optimizer', 'VectorBT optimizer'),
                ('template_interaction_generator', 'Template interaction generator'),
                ('unified_data_utils', 'Unified data utils')
            ]
            
            for attr_name, display_name in cleanup_components:
                if hasattr(self, attr_name):
                    component = getattr(self, attr_name)
                    if component and hasattr(component, 'cleanup'):
                        component.cleanup()
                        tprint_success(f"✅ {display_name} cleaned up")
            
            tprint_success("✅ Pipeline cleanup completed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Don't raise during destruction


def create_unified_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedDataDrivenPipeline:
    """Create a new unified pipeline instance."""
    return UnifiedDataDrivenPipeline(config)


async def process_with_unified_pipeline(data: pd.DataFrame,
                                      targets: Optional[pd.Series] = None,
                                      feature_columns: Optional[List[str]] = None,
                                      timeframe: str = "15m",
                                      config: Optional[UnifiedPipelineConfig] = None,
                                      pipeline_state: Optional[Dict[str, Any]] = None) -> ConsolidatedPipelineResult:
    """Process data with unified pipeline (async wrapper)."""
    pipeline = create_unified_pipeline(config)
    try:
        return pipeline.process(data, targets, feature_columns, timeframe, pipeline_state)
    finally:
        pipeline.cleanup()