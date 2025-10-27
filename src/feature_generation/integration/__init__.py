"""
Feature Generation Integration Module

This module provides comprehensive integration between feature bank features
and regime-specific features for different ML tasks.

Modules:
- feature_bank_integration: Core feature bank integration
- feature_task_integration: Task-specific feature integration
- enhanced_hdbscan_clustering_integration: HDBSCAN clustering integration
- enhanced_regime_clustering_integration: Regime clustering integration
- enhanced_models_training_integration: Models training integration
- enhanced_ensemble_training_integration: Ensemble training integration
"""

from .feature_bank_integration import (
    FeatureBankIntegrator,
    FeatureBankConfig,
    FeatureBankCategory,
    get_comprehensive_hdbscan_features,
    get_comprehensive_regime_clustering_features,
    get_comprehensive_models_training_features,
    get_comprehensive_ensemble_training_features,
    get_feature_breakdown_for_task
)

from .feature_task_integration import (
    FeatureTaskIntegrator,
    MLTask,
    FeatureTaskConfig
)

from .enhanced_hdbscan_clustering_integration import (
    EnhancedHDBSCANClusteringIntegration,
    get_enhanced_hdbscan_features,
    perform_enhanced_hdbscan_clustering
)

from .enhanced_regime_clustering_integration import (
    EnhancedRegimeClusteringIntegration,
    get_enhanced_regime_clustering_features,
    perform_enhanced_regime_clustering
)

from .enhanced_models_training_integration import (
    EnhancedModelsTrainingIntegration,
    get_enhanced_training_features,
    train_enhanced_models
)

from .enhanced_ensemble_training_integration import (
    EnhancedEnsembleTrainingIntegration,
    get_enhanced_ensemble_features,
    train_enhanced_ensemble
)

__all__ = [
    # Core integration
    'FeatureBankIntegrator',
    'FeatureBankConfig',
    'FeatureBankCategory',
    
    # Task integration
    'FeatureTaskIntegrator',
    'MLTask',
    'FeatureTaskConfig',
    
    # Convenience functions
    'get_comprehensive_hdbscan_features',
    'get_comprehensive_regime_clustering_features',
    'get_comprehensive_models_training_features',
    'get_comprehensive_ensemble_training_features',
    'get_feature_breakdown_for_task',
    
    # Enhanced integrations
    'EnhancedHDBSCANClusteringIntegration',
    'EnhancedRegimeClusteringIntegration',
    'EnhancedModelsTrainingIntegration',
    'EnhancedEnsembleTrainingIntegration',
    
    # Enhanced convenience functions
    'get_enhanced_hdbscan_features',
    'perform_enhanced_hdbscan_clustering',
    'get_enhanced_regime_clustering_features',
    'perform_enhanced_regime_clustering',
    'get_enhanced_training_features',
    'train_enhanced_models',
    'get_enhanced_ensemble_features',
    'train_enhanced_ensemble'
]