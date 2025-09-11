"""Data Qualification Package for Trading Pipeline.

This package contains all the components for data qualification with enhanced utilities:
- Support/Resistance (SR) detection and optimization
- HMM regime discovery and clustering
- Regime data splitting
- Data labeling (triple barrier method)
- Unified import management with fallback handling
- Standardized configuration system
- Comprehensive error handling and recovery
- Type-safe interfaces with proper documentation

Enhanced Features:
- Centralized utility management with automatic fallbacks
- Unified configuration system with validation
- Comprehensive error handling with recovery strategies
- Performance monitoring and metrics collection
- ML Commons integration with graceful degradation
- Type-safe interfaces and comprehensive documentation
"""

# Import legacy steps for backward compatibility
from ..market_analysis.sub_pipeline import MarketAnalysisSubPipeline
from .step03_hmm_regime_discovery import Step03HMMRegimeDiscovery
from .step04_regime_data_splitting import RegimeDataSplittingStep
from .step05_labeling import LabelingStep
from .step05_labeling_updated import EnhancedLabelingStep

# Import enhanced utilities
from src.utils.data_quality.data_qualification_base import (
    DataQualificationStep,
    DataQualificationPipeline,
    DataQualificationResult,
    StepMetrics
)
from src.utils.data_quality.data_qualification_config import (
    DataQualificationConfig,
    PerformanceConfig,
    SROptimizationConfig,
    HMMRegimeConfig,
    TripleBarrierConfig,
    RegimeProcessingConfig,
    MLCommonsConfig
)
from src.utils.data_quality.data_qualification_error_handler import (
    DataQualificationErrorHandler,
    handle_utility_failure,
    register_fallback
)
from src.utils.data_quality.data_qualification_imports import (
    DataQualificationImportManager,
    get_utility_suite,
    get_ml_commons_utilities,
    get_m1_optimization_utilities
)

# Import example enhanced step
from .step_example_enhanced import EnhancedSROptimizationStep

__all__ = [
    # Legacy steps (backward compatibility)
    'SROptimizationStep',
    'Step03HMMRegimeDiscovery', 
    'RegimeDataSplittingStep',
    'LabelingStep',
    'EnhancedLabelingStep',
    
    # Enhanced base classes and interfaces
    'DataQualificationStep',
    'DataQualificationPipeline',
    'DataQualificationResult',
    'StepMetrics',
    
    # Configuration system
    'DataQualificationConfig',
    'PerformanceConfig',
    'SROptimizationConfig',
    'HMMRegimeConfig',
    'TripleBarrierConfig',
    'RegimeProcessingConfig',
    'MLCommonsConfig',
    
    # Error handling
    'DataQualificationErrorHandler',
    'handle_utility_failure',
    'register_fallback',
    
    # Import management
    'DataQualificationImportManager',
    'get_utility_suite',
    'get_ml_commons_utilities',
    'get_m1_optimization_utilities',
    
    # Example enhanced step
    'EnhancedSROptimizationStep'
]