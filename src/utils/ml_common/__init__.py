"""
ML Common Utilities

This module provides common machine learning utilities and components
for the trading system, organized into logical sub-modules.
"""

from typing import TYPE_CHECKING, Any
from src.utils.logger import system_logger
from src.utils.lazy_module_loader import make_lazy_getattr, make_lazy_dir

# Configure logging
_LOGGER = system_logger.getChild('MLCommon')

# Import the proper tprint from utils (Eager load for logging)
from src.utils.tprint import tprint

# Legacy function for backward compatibility (deprecated)
def legacy_tprint(message: str, level: str = "INFO") -> None:
    """Legacy print message with timestamp and level (deprecated - use tprint from src.utils.tprint)."""
    if level.upper() == "ERROR":
        _LOGGER.error(message)
    elif level.upper() == "WARNING":
        _LOGGER.warning(message)
    elif level.upper() == "DEBUG":
        _LOGGER.debug(message)
    else:
        _LOGGER.info(message)

# HMM regime detection module has been deprecated; keep flag for compatibility probes
HMM_REGIME_DETECTION_AVAILABLE = False

# Define lazy loading map
_EXPORT_MAP = {
    # Models
    'EnhancedModelFactory': '.models',
    'ModelType': '.models',
    'ModelConfig': '.models',
    'create_model_factory': '.models',
    'MultiOutputConfig': '.models',
    'MultiOutputModel': '.models',
    'MultiOutputStackingModel': '.models',
    'MultiOutputResult': '.models',
    'prepare_multi_output_targets': '.models',
    'create_analyst_outputs': '.models',
    'create_tactician_outputs': '.models',
    'create_multi_output_stacking_model': '.models',
    'train_model_with_confidence_metrics': '.models',
    'ModelEvaluator': '.models',
    'ModelRegistry': '.models',
    'get_enhanced_model_trainer': '.models',

    # Ensembles
    'EnsembleManager': '.ensembles',
    'EnsembleType': '.ensembles',
    'EnsembleConfig': '.ensembles',
    'StackingEnsembleManager': '.ensembles',
    'StackingEnsembleConfig': '.ensembles',
    'StackingEnsembleResult': '.ensembles',
    'create_analyst_ensemble': '.ensembles',
    'create_tactician_ensemble': '.ensembles',
    'StackingConfidenceCalibrator': '.ensembles',
    'StackingCalibrationConfig': '.ensembles',
    'StackingCalibrationResult': '.ensembles',
    'create_analyst_calibrator': '.ensembles',
    'create_tactician_calibrator': '.ensembles',

    # Explainability
    'ModelExplainer': '.explainability',
    'ModelInterpretabilityEngine': '.explainability',
    'ExplanationResult': '.explainability',

    # Optimization
    'ParetoOptimizer': '.optimization',
    'ParetoFront': '.optimization',
    'ParetoFrontAnalyzer': '.optimization',

    # Data Processing
    'EnhancedDataLabelerGetter': '.data_processing',  # Aliased in map
    'LabelingConfigGetter': '.data_processing',      # Aliased in map

    # Validation
    'ConfigurationValidator': '.validation',
    'CrossValidationUtilities': '.validation',
    'PurgedKFold': '.validation',
    'TemporalCrossValidator': '.validation',
    'TimeSeriesSplitValidator': '.validation',
    'OOFGenerator': '.validation',
    'DataLeakageDetector': '.validation',
    'StabilityAnalyzer': '.validation',
    'UnifiedCrossValidator': '.validation',
    'UnifiedCVResult': '.validation',
    'perform_cross_validation': '.validation',
    'temporal_cross_validation': '.validation',
    'nested_cross_validation': '.validation',

    # Lookahead bias detection
    'LookaheadBiasDetector': '..lookahead_bias_detector',
    'LookaheadBiasError': '..lookahead_bias_detector',

    # Thresholding
    'optimize_threshold': '.validation.thresholding',
    'calibrate_probabilities': '.validation.thresholding',

    # Utils
    'setup_logger': '.utils',
    'get_logger': '.utils',
    'MemoryOptimizer': '.utils',
    'MemoryIntegrator': '.utils',
    'ParallelProcessor': '.utils',
    'UnifiedCache': '.utils',
    'get_unified_cache': '.utils',
    'cached': '.utils',
    'limit_blas_threads': '.utils',
    'get_thread_info': '.utils',
    'validate_thread_environment': '.utils',
    'LookaheadProtection': '.utils',
    'MLTrainingSafeguards': '.utils',
    'RobustErrorHandler': '.utils',

    # Legacy imports
    'FeatureSelector': '.feature_selection_backwards_compat',
    'FeatureSelectionConfig': '.feature_selection_backwards_compat',
    'LegacyFeatureSelector': '.feature_selection_backwards_compat', # Aliased? No, need to handle aliasing manually or use property
    'calculate_confidence_metrics': '.confidence_metrics',
    'calculate_calibration_metrics': '.confidence_metrics',
    
    # Matrix operations
    'M1EnhancedMatrixOperations': '..matrix_operations',
    'get_enhanced_matrix_operations': '..matrix_operations',

    # VectorBT-optimized utilities
    'MatrixCrossValidator': '.matrix_cross_validation',
    'matrix_cross_validate': '.matrix_cross_validation',
    'UnifiedVectorizationManager': '.unified_vectorization_manager',
    'OperationType': '.unified_vectorization_manager',
    'OptimizationStrategy': '.unified_vectorization_manager',
    'OperationConfig': '.unified_vectorization_manager',
    'OptimizationResult': '.unified_vectorization_manager',
    'optimize_cross_validation': '.unified_vectorization_manager',
    'optimize_backtesting': '.unified_vectorization_manager', 
    'optimize_financial_operation': '.unified_vectorization_manager',
    'optimize_vectorbt_backtesting': '.unified_vectorization_manager',
    'optimize_vectorbt_metrics': '.unified_vectorization_manager',
    'optimize_vectorbt_portfolio': '.unified_vectorization_manager',
    'get_unified_vectorization_manager': '.unified_vectorization_manager',

    # Hardware and component singletons
    'HardwareCapabilitiesManager': '.hardware_singleton',
    'HardwareCapabilities': '.hardware_singleton',
    'get_hardware_capabilities_manager': '.hardware_singleton',
    'get_hardware_capabilities': '.hardware_singleton',
    'get_hardware_capabilities_dict': '.hardware_singleton',
    'ComponentPool': '.component_pool',
    'get_component_pool': '.component_pool',
    'get_or_create_vectorbt_optimizer': '.component_pool',
    'get_or_create_performance_monitor': '.component_pool',
    'get_or_create_unified_vectorization_manager': '.component_pool',
    'PipelineOrchestrator': '.pipeline_orchestrator',

    # Feature selection analysis (external package)
    'FeatureImportanceAnalyzer': 'src.feature_selection.analysis.feature_importance_analyzer',
    'FeatureImportanceConfig': 'src.feature_selection.analysis.feature_importance_analyzer',
    'FeatureImportanceResult': 'src.feature_selection.analysis.feature_importance_analyzer',
    'ImportanceMethod': 'src.feature_selection.analysis.feature_importance_analyzer',
    'analyze_feature_importance': 'src.feature_selection.analysis.feature_importance_analyzer',
    'get_important_features': 'src.feature_selection.analysis.feature_importance_analyzer',

    # Data Drift Detector
    'DataDriftDetector': '.data_drift_detector',
    'DriftDetectionConfig': '.data_drift_detector',
    'DriftReport': '.data_drift_detector',
    'DriftResult': '.data_drift_detector',
    'DriftType': '.data_drift_detector',
    'DriftMethod': '.data_drift_detector',
    'DriftSeverity': '.data_drift_detector',
    'detect_data_drift': '.data_drift_detector',
    'get_drifted_features': '.data_drift_detector'
}

# Special handling for aliased imports
# We can't map 'LegacyFeatureSelector' directly to '.feature_selection_backwards_compat.FeatureSelector' via simple map
# We'll use a property for these specific aliases if generalized map doesn't support "import X as Y" naturally.
# Actually, make_lazy_getattr doesn't support 'module.Attribute'. It expects 'Attribute' in 'module'.
# So for 'LegacyFeatureSelector', we need a manual property or a wrapper.
# Or we just add 'LegacyFeatureSelector' to the submodule itself if possible. 
# For now, let's keep it handled manually in a customized getattr if needed, but checking the original file:
# from .feature_selection_backwards_compat import FeatureSelector as LegacyFeatureSelector
# We can't do this with simple mapping. 
# We will handle these overrides manually BEFORE calling the lazy getattr.

def lookahead_bias_detector():
    """Get lookahead bias detector instance."""
    from ..lookahead_bias_detector import LookaheadBiasDetector
    return LookaheadBiasDetector()

def hyperparameter_optimization():
    """Get hyperparameter optimization instance."""
    from .utils.hpo_utils import HyperparameterOptimization
    return HyperparameterOptimization()

__all__ = list(_EXPORT_MAP.keys()) + [
    'tprint', 'legacy_tprint', 'HMM_REGIME_DETECTION_AVAILABLE',
    'lookahead_bias_detector', 'hyperparameter_optimization'
]

if TYPE_CHECKING:
    # (Omitted full type checking block for brevity, ideally would be here)
    pass

# Custom getattr to handle aliases and fall back to generalized helper
_lazy_getattr_impl = make_lazy_getattr(_EXPORT_MAP, __package__)

def __getattr__(name: str) -> Any:
    # Handle aliases
    if name == 'EnhancedDataLabelerGetter':
        from .data_processing import get_enhanced_data_labeler
        return get_enhanced_data_labeler
    if name == 'LabelingConfigGetter':
        from .data_processing import get_labeling_config
        return get_labeling_config
    if name == 'LegacyFeatureSelector':
        from .feature_selection_backwards_compat import FeatureSelector
        return FeatureSelector
    if name == 'PipelineOrchestrator':
        from .pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator
        
    return _lazy_getattr_impl(name)

__dir__ = make_lazy_dir(_EXPORT_MAP, globals())
