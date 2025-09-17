"""
Advanced Markov Models Research - Complete Implementation

This package provides a comprehensive, production-ready implementation of
advanced Markov models for regime detection with multi-horizon (1h, 2h, 4h)
feature engineering and walk-forward validation.

Key Components:
1. Data-driven Markov-Switching Models (MSM)
2. Hidden Semi-Markov Models (HSMM) with flexible durations
3. Production feature engineering with 1h, 2h, 4h horizons
4. Walk-forward validation framework
5. Advanced model integration and selection
6. Comprehensive clustering enhancement
7. Production deployment artifacts

Quick Start:
    from src.research.clusters import AdvancedMarkovPipeline
    
    pipeline = AdvancedMarkovPipeline()
    results = await pipeline.run_complete_analysis(market_data_1h)
    
    print(f"Best model: {results.best_model_type}")
    print(f"Regimes detected: {len(results.regime_characteristics)}")

Advanced Usage:
    from src.research.clusters import (
        AdvancedMarkovPipelineConfig,
        DataDrivenMarkovSwitchingModel,
        DataDrivenHiddenSemiMarkovModel,
        ProductionLeakageSafeFeatures
    )
    
    # Custom configuration
    config = AdvancedMarkovPipelineConfig(
        horizons=[1, 2, 4],  # 1h, 2h, 4h windows
        enable_structural_break_features=True,
        enable_duration_features=True
    )
    
    pipeline = AdvancedMarkovPipeline(config)
    results = await pipeline.run_complete_analysis(data)
"""

# Core pipeline components
from .complete_advanced_markov_pipeline import (
    AdvancedMarkovPipeline,
    AdvancedMarkovPipelineConfig,
    PipelineResults,
    PipelineStage
)

# Data-driven advanced models
from .data_driven_markov_models import (
    DataDrivenMarkovSwitchingModel,
    DataDrivenHiddenSemiMarkovModel,
    DataDrivenAdvancedMarkovIntegration,
    DataDrivenMSMConfig,
    DataDrivenHSMMConfig,
    DurationLearningMethod,
    DataDrivenRegimeType
)

# Production feature engineering
from .production_feature_integration import (
    ProductionLeakageSafeFeatures,
    ProductionFeatureConfig,
    LeakageSafeRollingStats
)

# Advanced model integration
from .advanced_model_integration import (
    AdvancedModelSelector,
    WalkForwardConfig,
    ValidationMetric,
    ModelType,
    ValidationResult
)

# Import existing clustering and validation (if available)
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
        HMMIntegrationLayer,
        IntegrationConfig,
        IntegrationMethod
    )
    CLUSTERING_COMPONENTS_AVAILABLE = True
except ImportError:
    CLUSTERING_COMPONENTS_AVAILABLE = False

# Export main classes
__all__ = [
    # Main pipeline
    'AdvancedMarkovPipeline',
    'AdvancedMarkovPipelineConfig',
    'PipelineResults',
    'PipelineStage',
    
    # Advanced models
    'DataDrivenMarkovSwitchingModel',
    'DataDrivenHiddenSemiMarkovModel',
    'DataDrivenAdvancedMarkovIntegration',
    'DataDrivenMSMConfig',
    'DataDrivenHSMMConfig',
    'DurationLearningMethod',
    'DataDrivenRegimeType',
    
    # Feature engineering
    'ProductionLeakageSafeFeatures',
    'ProductionFeatureConfig',
    'LeakageSafeRollingStats',
    
    # Model integration
    'AdvancedModelSelector',
    'WalkForwardConfig',
    'ValidationMetric',
    'ModelType',
    'ValidationResult',
]

# Add clustering components if available
if CLUSTERING_COMPONENTS_AVAILABLE:
    __all__.extend([
        'RegimeClusterer',
        'ClusteringConfig', 
        'ClusteringMethod',
        'ClusteringResult',
        'RegimeValidationMetrics',
        'ValidationConfig',
        'HMMIntegrationLayer',
        'IntegrationConfig',
        'IntegrationMethod'
    ])

# Version info
__version__ = '1.0.0'
__author__ = 'Advanced Markov Research Team'
__description__ = 'Production-ready advanced Markov models for regime detection'

# Package metadata
PACKAGE_INFO = {
    'name': 'advanced_markov_research',
    'version': __version__,
    'description': __description__,
    'features': {
        'multi_horizon_analysis': [1, 2, 4],  # Hours
        'advanced_models': ['MSM', 'HSMM', 'Hybrid'],
        'production_ready': True,
        'walk_forward_validation': True,
        'leakage_safe_features': True,
        'clustering_enhancement': CLUSTERING_COMPONENTS_AVAILABLE,
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
        'ruptures': 'Enhanced structural break detection',
        'hmmlearn': 'Traditional HMM baseline',
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
    
    # Check ruptures for structural break detection
    try:
        import ruptures
        dependencies['ruptures'] = {'available': True, 'version': getattr(ruptures, '__version__', 'unknown')}
    except ImportError:
        dependencies['ruptures'] = {'available': False, 'impact': 'Limited structural break detection'}
    
    # Check hmmlearn for traditional HMM baseline
    try:
        import hmmlearn
        dependencies['hmmlearn'] = {'available': True, 'version': getattr(hmmlearn, '__version__', 'unknown')}
    except ImportError:
        dependencies['hmmlearn'] = {'available': False, 'impact': 'No traditional HMM baseline'}
    
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
    print("🚀 Advanced Markov Models Research Package")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print()
    
    print("✅ Core Features:")
    features = PACKAGE_INFO['features']
    print(f"  • Multi-horizon analysis: {', '.join(map(str, features['multi_horizon_analysis']))}h windows")
    print(f"  • Advanced models: {', '.join(features['advanced_models'])}")
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
    print("  from src.research.clusters import AdvancedMarkovPipeline")
    print("  pipeline = AdvancedMarkovPipeline()")
    print("  results = await pipeline.run_complete_analysis(market_data_1h)")
    print()
    
    print("📚 Main Components:")
    print("  • AdvancedMarkovPipeline: Complete analysis pipeline")
    print("  • DataDrivenMarkovSwitchingModel: MSM with structural breaks")
    print("  • DataDrivenHiddenSemiMarkovModel: HSMM with flexible durations")
    print("  • ProductionLeakageSafeFeatures: Multi-horizon feature engineering")
    print("  • AdvancedModelSelector: Walk-forward model selection")


# Quick access functions
def create_default_pipeline(horizons: list = None) -> 'AdvancedMarkovPipeline':
    """
    Create a default advanced Markov pipeline with recommended settings.
    
    Args:
        horizons: Multi-horizon windows (default: [1, 2, 4] hours)
        
    Returns:
        Configured AdvancedMarkovPipeline instance
    """
    if horizons is None:
        horizons = [1, 2, 4]
    
    config = AdvancedMarkovPipelineConfig(
        primary_timeframe="1h",
        horizons=horizons,
        enable_structural_break_features=True,
        enable_duration_features=True,
        enable_regime_transition_features=True,
        enable_markov_switching=True,
        enable_hidden_semi_markov=True,
        enable_hybrid_model=True,
        enable_clustering_enhancement=True,
        save_artifacts=True,
        enable_monitoring=True
    )
    
    return AdvancedMarkovPipeline(config)


def create_fast_pipeline(horizons: list = None) -> 'AdvancedMarkovPipeline':
    """
    Create a fast pipeline for quick testing with reduced validation.
    
    Args:
        horizons: Multi-horizon windows (default: [1, 2, 4] hours)
        
    Returns:
        Configured AdvancedMarkovPipeline instance for fast execution
    """
    if horizons is None:
        horizons = [1, 2, 4]
    
    config = AdvancedMarkovPipelineConfig(
        primary_timeframe="1h",
        horizons=horizons,
        enable_structural_break_features=True,
        enable_duration_features=True,
        enable_regime_transition_features=True,
        
        # Reduced validation for speed
        train_months=3,
        validation_months=1,
        n_folds=3,
        stability_test_iterations=2,
        cross_validation_folds=3,
        
        # Enable key models only
        enable_markov_switching=True,
        enable_hidden_semi_markov=True,
        enable_hybrid_model=False,  # Skip for speed
        
        enable_clustering_enhancement=False,  # Skip for speed
        save_artifacts=False,
        enable_monitoring=False
    )
    
    return AdvancedMarkovPipeline(config)


def create_production_pipeline(horizons: list = None) -> 'AdvancedMarkovPipeline':
    """
    Create a production-ready pipeline with comprehensive validation.
    
    Args:
        horizons: Multi-horizon windows (default: [1, 2, 4] hours)
        
    Returns:
        Configured AdvancedMarkovPipeline instance for production use
    """
    if horizons is None:
        horizons = [1, 2, 4]
    
    config = AdvancedMarkovPipelineConfig(
        primary_timeframe="1h",
        horizons=horizons,
        
        # Enable all advanced features
        enable_structural_break_features=True,
        enable_duration_features=True,
        enable_regime_transition_features=True,
        enable_existing_features=True,
        
        # Comprehensive validation
        train_months=12,
        validation_months=1,
        n_folds=12,
        stability_test_iterations=5,
        cross_validation_folds=5,
        
        # Enable all models
        enable_traditional_hmm=True,
        enable_markov_switching=True,
        enable_hidden_semi_markov=True,
        enable_hybrid_model=True,
        
        # Full clustering and monitoring
        enable_clustering_enhancement=True,
        save_artifacts=True,
        enable_monitoring=True,
        
        # Production thresholds
        min_regime_stability=0.4,
        min_model_agreement=0.5,
        max_transition_rate=0.15
    )
    
    return AdvancedMarkovPipeline(config)