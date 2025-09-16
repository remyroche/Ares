#!/usr/bin/env python3
"""
Enhanced HMM Clustering Package for Market Analysis

This package provides comprehensive HMM clustering capabilities for market regime detection,
leveraging all common utilities for optimal performance and reliability.

Main Components:
- enhanced_hmm_clustering: Core HMM clustering implementation
- config: Configuration system with presets and validation
- example_usage: Usage examples and demonstrations
- integration_example: Integration with existing pipeline
- test_implementation: Comprehensive test suite

Key Features:
- M1 hardware optimization (GPU, CPU, Memory)
- Matrix operations integration
- ML common utilities (CV, HPO, feature selection)
- Data processing utilities (klines, parquet)
- Math validation and error handling
- Comprehensive logging and monitoring
"""

# Import main classes and functions
from .enhanced_hmm_clustering import (
    EnhancedHMMClustering,
    HMMClusteringConfig,
    HMMClusteringResult,
    RegimeType,
    run_hmm_clustering_analysis
)

# Import configuration system
from .config import (
    HMMClusteringConfigFactory,
    ConfigValidator,
    ConfigPresets,
    get_config_by_name,
    create_custom_config,
    MarketType,
    TimeframeType
)

# Version information
__version__ = "1.0.0"
__author__ = "Market Analysis Team"
__description__ = "Enhanced HMM Clustering for Market Regime Detection"

# Package metadata
__all__ = [
    # Main classes
    'EnhancedHMMClustering',
    'HMMClusteringConfig', 
    'HMMClusteringResult',
    'RegimeType',
    
    # Main functions
    'run_hmm_clustering_analysis',
    
    # Configuration system
    'HMMClusteringConfigFactory',
    'ConfigValidator',
    'ConfigPresets',
    'get_config_by_name',
    'create_custom_config',
    'MarketType',
    'TimeframeType',
    
    # Version info
    '__version__',
    '__author__',
    '__description__'
]

# Package initialization
def get_package_info():
    """Get package information."""
    return {
        'name': 'Enhanced HMM Clustering',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'main_module': 'enhanced_hmm_clustering',
        'config_module': 'config',
        'example_module': 'example_usage',
        'integration_module': 'integration_example',
        'test_module': 'test_implementation'
    }

def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scikit-learn': 'sklearn',
        'hmmlearn': 'hmmlearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing = []
    available = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            available.append(name)
        except ImportError:
            missing.append(name)
    
    return {
        'available': available,
        'missing': missing,
        'all_available': len(missing) == 0
    }

def get_quick_start_example():
    """Get a quick start example."""
    return '''
# Quick Start Example
from market_analysis.hmm_clustering import run_hmm_clustering_analysis, get_config_by_name

# Use a preset configuration
config = get_config_by_name("crypto_btc_1h")

# Run HMM clustering analysis
result = run_hmm_clustering_analysis(
    symbol="BTCUSDT",
    interval="1h", 
    config=config,
    save_results=True
)

# Access results
print(f"Regime labels: {result.regime_labels}")
print(f"Regime characteristics: {result.regime_characteristics}")
print(f"Performance metrics: {result.performance_metrics}")
'''

# Package initialization message
if __name__ != "__main__":
    # Check dependencies on import
    deps = check_dependencies()
    if not deps['all_available']:
        import warnings
        warnings.warn(
            f"Some dependencies are missing: {deps['missing']}. "
            "Some features may not work correctly.",
            UserWarning
        )