"""
Regime Training Integration Example

This script demonstrates how to use the new regime base training and meta-model training
configurations with enhanced meta-features for improved regime detection and prediction.
"""

import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import the new configuration classes
from src.utils.ml_common.config.base_training_config import (
    PerRegimeTrainingConfig, 
    RegimeMetaModelTrainingConfig,
    EnsembleTrainingConfig
)

logger = logging.getLogger(__name__)


class RegimeTrainingIntegration:
    """
    Integration class for regime base training and meta-model training
    with enhanced meta-features.
    """
    
    def __init__(self, 
                 base_config_path: str = "src/config/regime_base_training_config.yaml",
                 meta_config_path: str = "src/config/regime_metamodel_training_config.yaml"):
        """Initialize the regime training integration."""
        self.base_config_path = base_config_path
        self.meta_config_path = meta_config_path
        
        # Load configurations
        self.base_config = self._load_base_config()
        self.meta_config = self._load_meta_config()
        
        # Initialize training configurations
        self.per_regime_config = PerRegimeTrainingConfig()
        self.meta_model_config = RegimeMetaModelTrainingConfig()
        self.ensemble_config = EnsembleTrainingConfig()
        
        logger.info("✅ Regime Training Integration initialized")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base training configuration from YAML."""
        try:
            with open(self.base_config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Base config file not found: {self.base_config_path}")
            return {}
    
    def _load_meta_config(self) -> Dict[str, Any]:
        """Load meta-model training configuration from YAML."""
        try:
            with open(self.meta_config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Meta config file not found: {self.meta_config_path}")
            return {}
    
    def get_catboost_config(self) -> Dict[str, Any]:
        """Get CatBoost configuration with enhanced parameters."""
        catboost_config = self.base_config.get('catboost', {})
        
        # Apply stable sweet spot if available
        stable_spot = catboost_config.get('stable_sweet_spot', {})
        if stable_spot:
            logger.info("Using CatBoost stable sweet spot configuration")
            return {
                'task_type': catboost_config.get('task_type', 'CPU'),
                'loss_function': catboost_config.get('loss_function', 'MultiClass'),
                'depth': stable_spot.get('depth', 5),
                'learning_rate': stable_spot.get('learning_rate', 0.05),
                'l2_leaf_reg': stable_spot.get('l2_leaf_reg', 8),
                'iterations': stable_spot.get('iterations', 800),
                'subsample': catboost_config.get('subsample', 0.7),
                'colsample_bylevel': catboost_config.get('colsample_bylevel', 0.7),
                'grow_policy': catboost_config.get('grow_policy', 'SymmetricTree'),
                'bootstrap_type': catboost_config.get('bootstrap_type', 'Bayesian'),
                'eval_metric': catboost_config.get('eval_metric', 'MultiClass')
            }
        
        return catboost_config
    
    def get_extratrees_config(self) -> Dict[str, Any]:
        """Get ExtraTrees configuration with enhanced parameters."""
        extratrees_config = self.base_config.get('extratrees', {})
        
        # Apply stable sweet spot if available
        stable_spot = extratrees_config.get('stable_sweet_spot', {})
        if stable_spot:
            logger.info("Using ExtraTrees stable sweet spot configuration")
            return {
                'n_estimators': stable_spot.get('n_estimators', 500),
                'max_depth': stable_spot.get('max_depth', None),
                'min_samples_leaf': stable_spot.get('min_samples_leaf', 5),
                'max_features': stable_spot.get('max_features', 'sqrt'),
                'bootstrap': extratrees_config.get('bootstrap', False),
                'criterion': extratrees_config.get('criterion', 'gini'),
                'random_state': extratrees_config.get('random_state', 42)
            }
        
        return extratrees_config
    
    def get_bayesian_rules_config(self) -> Dict[str, Any]:
        """Get Bayesian Rule Lists configuration."""
        return self.base_config.get('bayesian_rule_lists', {})
    
    def get_lightgbm_meta_config(self) -> Dict[str, Any]:
        """Get LightGBM meta-model configuration with shallow parameters."""
        lightgbm_config = self.meta_config.get('lightgbm_meta', {})
        
        # Apply stable sweet spot if available
        stable_spot = lightgbm_config.get('stable_sweet_spot', {})
        if stable_spot:
            logger.info("Using LightGBM meta-model stable sweet spot configuration")
            return {
                'objective': lightgbm_config.get('objective', 'multiclass'),
                'num_leaves': stable_spot.get('num_leaves', 23),
                'max_depth': stable_spot.get('max_depth', 4),
                'learning_rate': stable_spot.get('learning_rate', 0.04),
                'min_data_in_leaf': stable_spot.get('min_data_in_leaf', 100),
                'n_estimators': stable_spot.get('n_estimators', 400),
                'feature_fraction': lightgbm_config.get('feature_fraction', [0.6, 0.75, 0.9]),
                'bagging_fraction': lightgbm_config.get('bagging_fraction', 0.8),
                'bagging_freq': lightgbm_config.get('bagging_freq', 1),
                'lambda_l1': lightgbm_config.get('lambda_l1', [0, 1e-2, 1e-1]),
                'lambda_l2': lightgbm_config.get('lambda_l2', [0, 1e-2, 1e-1]),
                'boosting': lightgbm_config.get('boosting', 'gbdt'),
                'metric': lightgbm_config.get('metric', 'multi_logloss')
            }
        
        return lightgbm_config
    
    def get_meta_features_config(self) -> Dict[str, Any]:
        """Get meta-features configuration for enhanced prediction."""
        meta_features_config = self.meta_config.get('meta_features', {})
        
        # Combine base and meta configs
        base_meta_features = self.base_config.get('meta_features', {})
        
        return {
            'disagreement_uncertainty': {
                **base_meta_features.get('disagreement_uncertainty', {}),
                **meta_features_config.get('disagreement_uncertainty', {})
            },
            'temporal_dynamics': {
                **base_meta_features.get('temporal_dynamics', {}),
                **meta_features_config.get('temporal_dynamics', {})
            },
            'calibration_reliability': {
                **base_meta_features.get('calibration_reliability', {}),
                **meta_features_config.get('calibration_reliability', {})
            },
            'diversity_specialist': {
                **base_meta_features.get('diversity_specialist', {}),
                **meta_features_config.get('diversity_specialist', {})
            }
        }
    
    def create_meta_features(self, 
                           base_model_predictions: np.ndarray,
                           base_model_probabilities: np.ndarray,
                           timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create enhanced meta-features from base model predictions.
        
        Args:
            base_model_predictions: Predictions from base models (n_samples, n_models)
            base_model_probabilities: Probabilities from base models (n_samples, n_models, n_classes)
            timestamps: Optional timestamps for temporal features
            
        Returns:
            Meta-features array (n_samples, n_meta_features)
        """
        logger.info("Creating enhanced meta-features...")
        
        meta_features = []
        n_samples, n_models = base_model_predictions.shape
        
        # Calculate mean predictions across models
        mean_predictions = np.mean(base_model_predictions, axis=1)
        mean_probabilities = np.mean(base_model_probabilities, axis=1)
        
        # Disagreement & uncertainty features
        meta_features_config = self.get_meta_features_config()
        
        if meta_features_config['disagreement_uncertainty'].get('margin', False):
            # Margin: max_k p̄_k - max_{j≠k*} p̄_j
            max_probs = np.max(mean_probabilities, axis=1)
            second_max_probs = np.partition(mean_probabilities, -2, axis=1)[:, -2]
            margin = max_probs - second_max_probs
            meta_features.append(margin.reshape(-1, 1))
        
        if meta_features_config['disagreement_uncertainty'].get('entropy', False):
            # Entropy: H(p̄) = -∑_k p̄_k log p̄_k
            entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + 1e-8), axis=1)
            meta_features.append(entropy.reshape(-1, 1))
        
        if meta_features_config['disagreement_uncertainty'].get('gini_impurity', False):
            # Gini impurity: 1 - ∑_k p̄_k²
            gini = 1 - np.sum(mean_probabilities**2, axis=1)
            meta_features.append(gini.reshape(-1, 1))
        
        if meta_features_config['disagreement_uncertainty'].get('disagreement_rate', False):
            # Disagreement rate: fraction of base models not equal to ensemble argmax
            ensemble_argmax = np.argmax(mean_probabilities, axis=1)
            disagreement = np.mean(base_model_predictions != ensemble_argmax[:, np.newaxis], axis=1)
            meta_features.append(disagreement.reshape(-1, 1))
        
        # Temporal dynamics features
        if timestamps is not None and meta_features_config['temporal_dynamics'].get('probability_slope', False):
            # Probability slope: Δp̄_{k*} = p̄_{k*}(t) - p̄_{k*}(t-1)
            if len(mean_probabilities) > 1:
                top_class_probs = np.max(mean_probabilities, axis=1)
                prob_slope = np.diff(top_class_probs, prepend=top_class_probs[0])
                meta_features.append(prob_slope.reshape(-1, 1))
        
        # Combine all meta-features
        if meta_features:
            meta_features_array = np.hstack(meta_features)
            logger.info(f"Created {meta_features_array.shape[1]} meta-features")
            return meta_features_array
        else:
            logger.warning("No meta-features created")
            return np.array([]).reshape(n_samples, 0)
    
    def get_training_configuration(self, 
                                 training_type: str = "base") -> Dict[str, Any]:
        """
        Get complete training configuration for base or meta-model training.
        
        Args:
            training_type: "base" or "meta"
            
        Returns:
            Complete training configuration
        """
        if training_type == "base":
            return {
                'catboost': self.get_catboost_config(),
                'extratrees': self.get_extratrees_config(),
                'bayesian_rules': self.get_bayesian_rules_config(),
                'meta_features': self.get_meta_features_config(),
                'training': self.base_config.get('training', {}),
                'performance': self.base_config.get('performance', {})
            }
        elif training_type == "meta":
            return {
                'lightgbm_meta': self.get_lightgbm_meta_config(),
                'meta_features': self.get_meta_features_config(),
                'meta_training': self.meta_config.get('meta_training', {}),
                'performance': self.meta_config.get('performance', {}),
                'advanced_features': self.meta_config.get('advanced_features', {})
            }
        else:
            raise ValueError(f"Unknown training type: {training_type}")
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate the training configuration."""
        try:
            # Check required sections
            required_sections = ['training', 'performance']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # Check model configurations
            model_configs = ['catboost', 'extratrees', 'bayesian_rules']
            for model in model_configs:
                if model in config and not isinstance(config[model], dict):
                    logger.error(f"Invalid configuration for {model}")
                    return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def save_configuration(self, 
                          config: Dict[str, Any], 
                          output_path: str):
        """Save configuration to file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise


def main():
    """Main function demonstrating the regime training integration."""
    logger.info("🚀 Starting Regime Training Integration Example")
    
    # Initialize integration
    integration = RegimeTrainingIntegration()
    
    # Get base training configuration
    base_config = integration.get_training_configuration("base")
    logger.info("📊 Base training configuration loaded")
    
    # Get meta-model training configuration
    meta_config = integration.get_training_configuration("meta")
    logger.info("📊 Meta-model training configuration loaded")
    
    # Validate configurations
    if integration.validate_configuration(base_config):
        logger.info("✅ Base configuration is valid")
    else:
        logger.error("❌ Base configuration is invalid")
        return
    
    if integration.validate_configuration(meta_config):
        logger.info("✅ Meta configuration is valid")
    else:
        logger.error("❌ Meta configuration is invalid")
        return
    
    # Demonstrate meta-features creation
    logger.info("🔧 Demonstrating meta-features creation...")
    
    # Create sample data
    n_samples, n_models, n_classes = 1000, 5, 3
    base_predictions = np.random.randint(0, n_classes, (n_samples, n_models))
    base_probabilities = np.random.dirichlet(np.ones(n_classes), (n_samples, n_models))
    timestamps = np.arange(n_samples)
    
    # Create meta-features
    meta_features = integration.create_meta_features(
        base_predictions, base_probabilities, timestamps
    )
    
    logger.info(f"📈 Created meta-features with shape: {meta_features.shape}")
    
    # Save configurations
    integration.save_configuration(
        base_config, 
        "output/regime_base_training_config.yaml"
    )
    
    integration.save_configuration(
        meta_config, 
        "output/regime_metamodel_training_config.yaml"
    )
    
    logger.info("🎉 Regime Training Integration Example completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()