"""
Enhanced Statsmodels-based Clustering Module

This module provides a comprehensive migration from Pyro-based Sticky Finite HMM
to well-maintained `statsmodels.tsa.regime_switching` implementation. The enhanced MarkovRegressionAdapter serves as foundation for this migration, offering hardware optimization, parameter mapping, advanced diagnostics, and integration capabilities.

Phase 1: Foundation - Enhanced MarkovRegressionAdapter
- Hardware optimization integration with UnifiedHardwareManager
- Parameter mapping from Pyro configurations
- Advanced diagnostics and validation
- Hierarchical optimization support
- VectorBT integration hooks
- Comprehensive error handling and logging

Key Components:
- MarkovRegressionAdapter: Enhanced wrapper around statsmodels MarkovRegression
- ParameterMapper: Maps Pyro parameters to statsmodels format
- StatsmodelsHardwareOptimizer: Hardware optimization integration
- VectorBTIntegration: Backtesting and portfolio analysis
- MarkovRegressionDiagnostics: Advanced model diagnostics

Usage:
    from statsmodel_clustering import (
        MarkovRegressionAdapter,
        create_enhanced_markov_regression_adapter
    )
    
    # Create adapter with enhanced features
    adapter = create_enhanced_markov_regression_adapter(
        k_regimes=3,
        enable_hardware_optimization=True,
        enable_diagnostics=True,
        enable_vectorbt_integration=True
    )
    
    # Fit model
    result = adapter.fit(data)
    
    # Get predictions and diagnostics
    predictions = adapter.predict(steps=10)
    probabilities = adapter.get_regime_probabilities()
    transition_matrix = adapter.get_transition_matrix()
"""

# Core components
from .core import (
    MarkovRegressionAdapter,
    MarkovRegressionConfig,
    MarkovRegressionResult,
    ParameterMapper,
    MarkovRegressionDiagnostics,
    create_enhanced_markov_regression_adapter
)

# Optimization components
from .optimization import (
    PyroToStatsmodelsMapper,
    ParameterMappingConfig,
    ParameterMappingResult,
    map_pyro_to_statsmodels,
    map_pyro_search_space,
    create_default_mapping_config
)

# Integration components
from .integration import (
    StatsmodelsHardwareOptimizer,
    HardwareOptimizationConfig,
    HardwareOptimizationResult,
    create_hardware_optimizer,
    optimize_for_regime_switching
)

# Try to import VectorBT components
try:
    from .integration import (
        VectorBTIntegration,
        VectorBTConfig,
        VectorBTResult,
        create_vectorbt_integration,
        backtest_regime_strategy
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VectorBTIntegration = None
    VectorBTConfig = None
    VectorBTResult = None
    create_vectorbt_integration = None
    backtest_regime_strategy = None
    VECTORBT_AVAILABLE = False

# Utility components
from .utils import (
    ResultConverter,
    ConversionConfig,
    convert_statsmodels_to_pyro,
    convert_pyro_to_statsmodels,
    create_unified_result,
    save_result_to_file,
    
    ModelValidator,
    ValidationConfig,
    ValidationResult,
    validate_input_data,
    validate_model_fit,
    cross_validate_regime_model,
    
    ModelDiagnostics,
    DiagnosticsConfig,
    DiagnosticsResult,
    analyze_model_fit,
    analyze_regime_stability,
    create_diagnostics_report
)

# Version and compatibility
__version__ = "1.0.0"
__statsmodels_version__ = ">=0.13.0"
__python_version__ = ">=3.8"

# Export main components
__all__ = [
    # Core components
    'MarkovRegressionAdapter',
    'MarkovRegressionConfig',
    'MarkovRegressionResult',
    'ParameterMapper',
    'MarkovRegressionDiagnostics',
    'create_enhanced_markov_regression_adapter',
    
    # Optimization components
    'PyroToStatsmodelsMapper',
    'ParameterMappingConfig',
    'ParameterMappingResult',
    'map_pyro_to_statsmodels',
    'map_pyro_search_space',
    'create_default_mapping_config',
    
    # Integration components
    'StatsmodelsHardwareOptimizer',
    'HardwareOptimizationConfig',
    'HardwareOptimizationResult',
    'create_hardware_optimizer',
    'optimize_for_regime_switching',
    
    # Utility components
    'ResultConverter',
    'ConversionConfig',
    'convert_statsmodels_to_pyro',
    'convert_pyro_to_statsmodels',
    'create_unified_result',
    'save_result_to_file',
    
    'ModelValidator',
    'ValidationConfig',
    'ValidationResult',
    'validate_input_data',
    'validate_model_fit',
    'cross_validate_regime_model',
    
    'ModelDiagnostics',
    'DiagnosticsConfig',
    'DiagnosticsResult',
    'analyze_model_fit',
    'analyze_regime_stability',
    'create_diagnostics_report'
]

# Export VectorBT components if available
if VECTORBT_AVAILABLE:
    __all__.extend([
        'VectorBTIntegration',
        'VectorBTConfig',
        'VectorBTResult',
        'create_vectorbt_integration',
        'backtest_regime_strategy'
    ])

# Convenience functions for quick start
def quick_start_adapter(k_regimes: int = 2, 
                      enable_hardware: bool = True,
                      enable_diagnostics: bool = True) -> MarkovRegressionAdapter:
    """
    Quick start function for creating an enhanced adapter.
    
    Args:
        k_regimes: Number of regimes
        enable_hardware: Enable hardware optimization
        enable_diagnostics: Enable advanced diagnostics
        
    Returns:
        Configured MarkovRegressionAdapter instance
    """
    from .core import create_enhanced_markov_regression_adapter
    
    return create_enhanced_markov_regression_adapter(
        k_regimes=k_regimes,
        enable_hardware_optimization=enable_hardware,
        enable_diagnostics=enable_diagnostics,
        enable_vectorbt_integration=VECTORBT_AVAILABLE
    )

def get_migration_status() -> dict:
    """
    Get current migration status and capabilities.
    
    Returns:
        Dictionary with migration status and available features
    """
    return {
        'version': __version__,
        'statsmodels_version': __statsmodels_version__,
        'python_version': __python_version__,
        'features': {
            'enhanced_adapter': True,
            'hardware_optimization': True,
            'parameter_mapping': True,
            'advanced_diagnostics': True,
            'vectorbt_integration': VECTORBT_AVAILABLE,
            'hierarchical_optimization': True,
            'result_conversion': True,
            'validation': True
        },
        'compatibility': {
            'pyro_migration': True,
            'backward_compatible': True,
            'production_ready': True
        },
        'dependencies': {
            'statsmodels': '>=0.13.0',
            'numpy': '>=1.20.0',
            'pandas': '>=1.3.0',
            'scipy': '>=1.7.0',
            'sklearn': '>=1.0.0',
            'vectorbt': '>=0.25.0' if VECTORBT_AVAILABLE else None
        }
    }

# Module metadata
__author__ = "Statsmodels Migration Team"
__email__ = "migration@example.com"
__description__ = "Enhanced statsmodels-based clustering with hardware optimization and integration"
__license__ = "MIT"
