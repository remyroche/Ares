"""
Advanced Clustering Research - Complete Implementation

This package provides a comprehensive, production-ready implementation of
advanced clustering methods for regime detection with multi-horizon analysis
and walk-forward validation.

Key Components:
1. Data-driven clustering framework
2. Similarity matrix clustering
3. Empirical threshold discovery
4. Production feature engineering
5. Comprehensive validation and metrics
6. Production deployment artifacts

Quick Start:
    from src.research.clusters import DataDrivenClusteringFramework
    
    framework = DataDrivenClusteringFramework()
    results = framework.run_complete_analysis(market_data_1h)
    
    print(f"Clusters detected: {len(results.cluster_characteristics)}")

Advanced Usage:
    from src.research.clusters import (
        DataDrivenClusteringFramework,
        SimilarityMatrixClusterer,
        EmpiricalThresholdDiscovery
    )
    
    # Custom configuration
    config = DataDrivenClusteringConfig(
        n_clusters=5,
        enable_validation=True,
        enable_feature_importance=True
    )
    
    framework = DataDrivenClusteringFramework(config)
    results = framework.run_complete_analysis(data)
"""

# Import data-driven clustering components
try:
    from .data_driven_clustering_framework import (
        DataDrivenClusteringFramework,
        DataDrivenClusteringConfig,
        DataDrivenClusteringResult,
        data_driven_regime_discovery,
        quick_regime_discovery
    )
    from .similarity_matrix_clustering import (
        SimilarityMatrixClusterer,
        SimilarityClusteringConfig,
        SimilarityMethod,
        similarity_matrix_clustering
    )
    from .empirical_threshold_discovery import (
        EmpiricalThresholdDiscovery,
        EmpiricalDiscoveryConfig,
        discover_optimal_clustering_thresholds
    )
    DATA_DRIVEN_CLUSTERING_AVAILABLE = True
except ImportError:
    DATA_DRIVEN_CLUSTERING_AVAILABLE = False

# Import legacy clustering and validation (if available)
try:
    from .regime_clusterer import (
        RegimeClusterer,
        ClusteringConfig,
        ClusteringMethod,
        ClusteringResult
    )
    from .validation_metrics import (
        RegimeValidationMetrics,
        ValidationConfig
    )
    from .integration_layer import (
        IntegrationLayer,
        IntegrationConfig,
        IntegrationMethod
    )
    LEGACY_CLUSTERING_AVAILABLE = True
except ImportError:
    LEGACY_CLUSTERING_AVAILABLE = False

# Export main classes
__all__ = [
    # Data-driven clustering framework
    'DataDrivenClusteringFramework',
    'DataDrivenClusteringConfig',
    'DataDrivenClusteringResult',
    'data_driven_regime_discovery',
    'quick_regime_discovery',
    
    # Similarity matrix clustering
    'SimilarityMatrixClusterer',
    'SimilarityClusteringConfig',
    'SimilarityMethod',
    'similarity_matrix_clustering',
    
    # Empirical threshold discovery
    'EmpiricalThresholdDiscovery',
    'EmpiricalDiscoveryConfig',
    'discover_optimal_clustering_thresholds'
]

# Add legacy clustering components if available
if LEGACY_CLUSTERING_AVAILABLE:
    __all__.extend([
        'RegimeClusterer',
        'ClusteringConfig', 
        'ClusteringMethod',
        'ClusteringResult',
        'RegimeValidationMetrics',
        'ValidationConfig',
        'IntegrationLayer',
        'IntegrationConfig',
        'IntegrationMethod'
    ])

# Version info
__version__ = '1.0.0'
__author__ = 'Advanced Clustering Research Team'
__description__ = 'Production-ready advanced clustering methods for regime detection'

# Package metadata
PACKAGE_INFO = {
    'name': 'advanced_clustering_research',
    'version': __version__,
    'description': __description__,
    'features': {
        'multi_horizon_analysis': [1, 2, 4],  # Hours
        'advanced_clustering': ['KMeans', 'GMM', 'HDBSCAN', 'Spectral'],
        'production_ready': True,
        'walk_forward_validation': True,
        'leakage_safe_features': True,
        'clustering_enhancement': True,
        'structural_break_detection': True,
        'duration_modeling': True,
        'regime_transition_analysis': True
    },
    'requirements': {
        'python': '>=3.8',
        'numpy': '>=1.20.0',
        'pandas': '>=1.3.0',
        'scikit-learn': '>=1.0.0',
        'scipy': '>=1.7.0'
    },
    'optional_requirements': {
        'hdbscan': 'Advanced clustering methods',
        'talib': 'Technical indicators'
    }
}


def get_package_info():
    """Get package information and capabilities."""
    return PACKAGE_INFO


def check_dependencies():
    """Check for optional dependencies and return availability status."""
    dependencies = {}
    
    # Check hdbscan for advanced clustering
    try:
        import hdbscan
        dependencies['hdbscan'] = {'available': True, 'version': getattr(hdbscan, '__version__', 'unknown')}
    except ImportError:
        dependencies['hdbscan'] = {'available': False, 'impact': 'Limited clustering methods'}
    
    # Check talib for technical indicators
    try:
        import talib
        dependencies['talib'] = {'available': True, 'version': getattr(talib, '__version__', 'unknown')}
    except ImportError:
        dependencies['talib'] = {'available': False, 'impact': 'Limited technical indicators'}
    
    return dependencies


def print_package_status():
    """Print package status and capabilities."""
    print("🚀 Advanced Clustering Research Package")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print()
    
    print("✅ Core Features:")
    features = PACKAGE_INFO['features']
    print(f"  • Multi-horizon analysis: {', '.join(map(str, features['multi_horizon_analysis']))}h windows")
    print(f"  • Advanced clustering: {', '.join(features['advanced_clustering'])}")
    print(f"  • Production ready: {'✅' if features['production_ready'] else '❌'}")
    print(f"  • Walk-forward validation: {'✅' if features['walk_forward_validation'] else '❌'}")
    print(f"  • Leakage-safe features: {'✅' if features['leakage_safe_features'] else '❌'}")
    print(f"  • Clustering enhancement: {'✅' if features['clustering_enhancement'] else '❌'}")
    print()
    
    print("📦 Dependency Status:")
    dependencies = check_dependencies()
    for dep_name, dep_info in dependencies.items():
        status = "✅" if dep_info['available'] else "❌"
        if dep_info['available']:
            print(f"  {status} {dep_name}: {dep_info.get('version', 'installed')}")
        else:
            print(f"  {status} {dep_name}: {dep_info.get('impact', 'not available')}")
    print()
    
    print("🎯 Quick Start:")
    print("  from src.research.clusters import DataDrivenClusteringFramework")
    print("  framework = DataDrivenClusteringFramework()")
    print("  results = framework.run_complete_analysis(market_data_1h)")
    print()
    
    print("📚 Main Components:")
    print("  • DataDrivenClusteringFramework: Complete clustering analysis")
    print("  • SimilarityMatrixClusterer: Similarity-based clustering")
    print("  • EmpiricalThresholdDiscovery: Data-driven threshold discovery")
    print("  • RegimeClusterer: Traditional clustering methods")


# Quick access functions
def create_default_framework(n_clusters: int = 5) -> 'DataDrivenClusteringFramework':
    """
    Create a default clustering framework with recommended settings.
    
    Args:
        n_clusters: Number of clusters to detect (default: 5)
        
    Returns:
        Configured DataDrivenClusteringFramework instance
    """
    config = DataDrivenClusteringConfig(
        n_clusters=n_clusters,
        enable_validation=True,
        enable_feature_importance=True,
        enable_empirical_thresholds=True
    )
    
    return DataDrivenClusteringFramework(config)


def create_fast_framework(n_clusters: int = 5) -> 'DataDrivenClusteringFramework':
    """
    Create a fast framework for quick testing with reduced validation.
    
    Args:
        n_clusters: Number of clusters to detect (default: 5)
        
    Returns:
        Configured DataDrivenClusteringFramework instance for fast execution
    """
    config = DataDrivenClusteringConfig(
        n_clusters=n_clusters,
        enable_validation=False,  # Skip for speed
        enable_feature_importance=False,  # Skip for speed
        enable_empirical_thresholds=False,  # Skip for speed
        max_iterations=10  # Reduced iterations
    )
    
    return DataDrivenClusteringFramework(config)


def create_production_framework(n_clusters: int = 5) -> 'DataDrivenClusteringFramework':
    """
    Create a production-ready framework with comprehensive validation.
    
    Args:
        n_clusters: Number of clusters to detect (default: 5)
        
    Returns:
        Configured DataDrivenClusteringFramework instance for production use
    """
    config = DataDrivenClusteringConfig(
        n_clusters=n_clusters,
        enable_validation=True,
        enable_feature_importance=True,
        enable_empirical_thresholds=True,
        enable_cross_validation=True,
        max_iterations=100,
        stability_tests=5
    )
    
    return DataDrivenClusteringFramework(config)