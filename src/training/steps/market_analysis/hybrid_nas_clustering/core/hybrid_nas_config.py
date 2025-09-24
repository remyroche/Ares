"""
Hybrid NAS Clustering Configuration

Configuration for hybrid NAS clustering that combines tree-based and neural
approaches to complement the existing neural NAS system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class HybridNASClusteringConfig:
    """Configuration for hybrid NAS clustering system."""
    
    # Neural NAS configuration (existing)
    neural_nas_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_regimes': 12,
        'enable_micro_regime_detection': True,
        'economic_significance_threshold': 0.7,
        'trading_viability_threshold': 0.6,
        'timeframe': '15m',
        'micro_timeframe': '5m'
    })
    
    # Tree-based NAS configuration (complementary)
    tree_nas_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_types': ['xgboost', 'lightgbm', 'catboost'],
        'n_trials': 50,
        'objectives': ['accuracy', 'efficiency', 'interpretability'],
        'enable_feature_selection': True,
        'max_features': 50,
        'enable_regime_awareness': True
    })
    
    # Hybrid strategy configuration
    hybrid_strategy: str = 'complementary'  # 'complementary', 'ensemble', 'routing', 'sequential'
    
    # Data routing rules
    routing_rules: Dict[str, Any] = field(default_factory=lambda: {
        'use_tree_for_tabular': True,
        'use_neural_for_sequential': True,
        'use_tree_for_feature_selection': True,
        'use_neural_for_complex_patterns': True,
        'tabular_threshold': 0.7,
        'sequential_threshold': 0.5,
        'complexity_threshold': 0.8
    })
    
    # Ensemble configuration
    ensemble_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_ensemble': True,
        'ensemble_methods': ['voting', 'stacking', 'blending'],
        'tree_weight': 0.6,
        'neural_weight': 0.4,
        'max_ensemble_models': 5
    })
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_accuracy': 0.7,
        'min_efficiency': 0.5,
        'max_training_time': 3600,
        'min_interpretability': 0.3
    })
    
    # Integration settings
    integration_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_feature_transfer': True,
        'enable_architecture_transfer': True,
        'enable_performance_transfer': True,
        'complementary_mode': True,
        'fallback_to_neural': True
    })
    
    # Clustering specific settings
    clustering_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_regimes': 12,
        'enable_micro_regime_detection': True,
        'micro_regime_sensitivity': 0.7,
        'economic_significance_threshold': 0.7,
        'trading_viability_threshold': 0.6,
        'regime_transition_cost': 0.05
    })
    
    # Optimization settings
    optimization_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_trials': 100,
        'timeout_seconds': 7200,
        'early_stopping_patience': 10,
        'cv_folds': 5,
        'test_size': 0.2
    })
    
    # Logging and monitoring
    logging_config: Dict[str, Any] = field(default_factory=lambda: {
        'log_level': 'INFO',
        'enable_performance_monitoring': True,
        'enable_feature_importance_logging': True,
        'enable_regime_transition_logging': True
    })
    
    @classmethod
    def create_complementary_config(cls) -> 'HybridNASClusteringConfig':
        """Create configuration optimized for complementary approach."""
        config = cls()
        config.hybrid_strategy = 'complementary'
        config.tree_nas_config.update({
            'objectives': ['accuracy', 'efficiency', 'interpretability'],
            'enable_feature_selection': True,
            'max_features': 50
        })
        config.neural_nas_config.update({
            'objectives': ['accuracy', 'efficiency', 'robustness'],
            'enable_complex_patterns': True
        })
        return config
    
    @classmethod
    def create_ensemble_config(cls) -> 'HybridNASClusteringConfig':
        """Create configuration optimized for ensemble approach."""
        config = cls()
        config.hybrid_strategy = 'ensemble'
        config.ensemble_config.update({
            'enable_ensemble': True,
            'ensemble_methods': ['voting', 'stacking'],
            'tree_weight': 0.6,
            'neural_weight': 0.4
        })
        return config
    
    @classmethod
    def create_routing_config(cls) -> 'HybridNASClusteringConfig':
        """Create configuration optimized for routing approach."""
        config = cls()
        config.hybrid_strategy = 'routing'
        config.routing_rules.update({
            'use_tree_for_tabular': True,
            'use_neural_for_sequential': True,
            'tabular_threshold': 0.7,
            'sequential_threshold': 0.5
        })
        return config
    
    @classmethod
    def create_sequential_config(cls) -> 'HybridNASClusteringConfig':
        """Create configuration optimized for sequential approach."""
        config = cls()
        config.hybrid_strategy = 'sequential'
        config.tree_nas_config.update({
            'objectives': ['accuracy', 'efficiency', 'interpretability'],
            'enable_feature_selection': True
        })
        config.neural_nas_config.update({
            'objectives': ['accuracy', 'efficiency', 'robustness'],
            'enable_complex_patterns': True
        })
        return config
    
    def get_tree_config(self) -> Dict[str, Any]:
        """Get tree-based NAS configuration."""
        return self.tree_nas_config.copy()
    
    def get_neural_config(self) -> Dict[str, Any]:
        """Get neural NAS configuration."""
        return self.neural_nas_config.copy()
    
    def get_hybrid_config(self) -> Dict[str, Any]:
        """Get hybrid configuration."""
        return {
            'hybrid_strategy': self.hybrid_strategy,
            'routing_rules': self.routing_rules,
            'ensemble_config': self.ensemble_config,
            'performance_thresholds': self.performance_thresholds,
            'integration_config': self.integration_config
        }
    
    def get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering-specific configuration."""
        return self.clustering_config.copy()
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.optimization_config.copy()
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate hybrid strategy
        if self.hybrid_strategy not in ['complementary', 'ensemble', 'routing', 'sequential']:
                logger.warning(f"Invalid hybrid strategy: {self.hybrid_strategy}")
                return False
            
            # Validate tree NAS config
            if not self.tree_nas_config.get('model_types'):
                logger.warning("Tree NAS config missing model_types")
                return False
            
            # Validate neural NAS config
            if not self.neural_nas_config.get('n_regimes'):
                logger.warning("Neural NAS config missing n_regimes")
                return False
            
            # Validate ensemble config
            if self.hybrid_strategy == 'ensemble':
                if not self.ensemble_config.get('enable_ensemble'):
                    logger.warning("Ensemble strategy requires enable_ensemble=True")
                    return False
            
            # Validate routing config
            if self.hybrid_strategy == 'routing':
                if not self.routing_rules.get('use_tree_for_tabular'):
                    logger.warning("Routing strategy requires use_tree_for_tabular=True")
                    return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        try:
            for key, value in updates.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'neural_nas_config': self.neural_nas_config,
            'tree_nas_config': self.tree_nas_config,
            'hybrid_strategy': self.hybrid_strategy,
            'routing_rules': self.routing_rules,
            'ensemble_config': self.ensemble_config,
            'performance_thresholds': self.performance_thresholds,
            'integration_config': self.integration_config,
            'clustering_config': self.clustering_config,
            'optimization_config': self.optimization_config,
            'logging_config': self.logging_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridNASClusteringConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config