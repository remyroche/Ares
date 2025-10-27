"""
Feature Task Integration System

This module provides integration between distinct feature categories and their respective ML tasks.
It ensures proper feature routing and selection for different machine learning workflows.

Feature Categories:
1. HDBSCAN Clustering: 50-100 features optimized for density-based clustering
2. Regime Clustering: 40-80 features for general regime identification  
3. Models Training: 30-60 features safe for ML model training
4. Ensemble Training: 20-40 features for meta-learner optimization

Integration Points:
- hdbscan_clustering -> HDBSCAN Clustering features
- regime_feature_selection -> Regime Clustering features
- regime_clustering -> Regime Clustering features
- regime_models_training -> Models Training features (with LGBM-SHAP selection if >60)
- regime_ensemble_training -> Ensemble Training features + base model outputs
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# Import feature categorization system
from ..categories.regime_feature_categorization import (
    RegimeFeatureCategorizer,
    FeatureUseCase,
    get_hdbscan_features,
    get_regime_clustering_features,
    get_regime_models_training_features,
    get_regime_ensemble_training_features
)

# Import feature generators
from ..categories.regime_features import (
    RegimeStatisticalFeatureGenerator,
    RegimeStructuralTrendFeatureGenerator,
    RegimeVolatilityFeatureGenerator,
    RegimeVolumeFeatureGenerator,
    RegimeEntropyGenerator,
    RegimeComplexityGenerator,
    RegimeFractalDimensionGenerator,
    RegimeHurstExponentGenerator,
    RegimeMemoryStrengthGenerator,
    RegimeCrossAssetGenerator,
    RegimeTransitionProbabilityGenerator,
    RegimeFeatureIntegration
)

from ..categories.clustering_features import (
    ClusteringDistanceGenerator,
    ClusteringSeparationGenerator,
    ClusteringStabilityGenerator,
    ClusteringIntegration
)

# Import feature selection tools
try:
    from src.training.utils.feature_selection.selection_methods import FeatureImportanceRanker
    from src.feature_selection.advanced.multi_stage_rfe import MultiStageRFE
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    warnings.warn("Feature selection tools not available. LGBM-SHAP selection will be disabled.")

# Import LGBM and SHAP for feature selection
try:
    import lightgbm as lgb
    import shap
    LGBM_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    SHAP_AVAILABLE = False
    warnings.warn("LGBM or SHAP not available. Feature selection will use alternative methods.")


class MLTask(Enum):
    """Enumeration of ML tasks."""
    HDBSCAN_CLUSTERING = "hdbscan_clustering"
    REGIME_FEATURE_SELECTION = "regime_feature_selection"
    REGIME_CLUSTERING = "regime_clustering"
    REGIME_MODELS_TRAINING = "regime_models_training"
    REGIME_ENSEMBLE_TRAINING = "regime_ensemble_training"


@dataclass
class FeatureTaskConfig:
    """Configuration for feature-task integration."""
    
    # Feature limits for each task
    hdbscan_max_features: int = 150
    hdbscan_min_features: int = 100
    regime_clustering_max_features: int = 80
    regime_clustering_min_features: int = 40
    models_training_max_features: int = 60
    models_training_min_features: int = 30
    ensemble_training_max_features: int = 40
    ensemble_training_min_features: int = 20
    
    # Feature selection settings
    enable_lgbm_shap_selection: bool = True
    lgbm_selection_params: Dict[str, Any] = None
    
    # Ensemble training settings
    include_base_model_outputs: bool = True
    include_disagreement_features: bool = True
    include_entropy_features: bool = True
    
    def __post_init__(self):
        """Set default values."""
        if self.lgbm_selection_params is None:
            self.lgbm_selection_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }


class FeatureTaskIntegrator:
    """
    Feature Task Integrator.
    
    Integrates distinct feature categories with their respective ML tasks,
    ensuring proper feature selection and routing.
    """
    
    def __init__(self, config: Optional[FeatureTaskConfig] = None):
        self.config = config or FeatureTaskConfig()
        self.categorizer = RegimeFeatureCategorizer()
        self.feature_cache = {}
        
    def get_features_for_task(self, task: MLTask, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get features optimized for a specific ML task.
        
        Args:
            task: The ML task to get features for
            data: Optional data for feature generation
            
        Returns:
            Dictionary containing features and metadata
        """
        if task == MLTask.HDBSCAN_CLUSTERING:
            return self._get_hdbscan_features(data)
        elif task == MLTask.REGIME_FEATURE_SELECTION:
            return self._get_regime_clustering_features(data)
        elif task == MLTask.REGIME_CLUSTERING:
            return self._get_regime_clustering_features(data)
        elif task == MLTask.REGIME_MODELS_TRAINING:
            return self._get_models_training_features(data)
        elif task == MLTask.REGIME_ENSEMBLE_TRAINING:
            return self._get_ensemble_training_features(data)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _get_hdbscan_features(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get features optimized for HDBSCAN clustering (50-100 features)."""
        # Get clustering-specific features
        clustering_features = get_hdbscan_features()
        
        # Limit to target range
        if len(clustering_features) > self.config.hdbscan_max_features:
            clustering_features = clustering_features[:self.config.hdbscan_max_features]
        elif len(clustering_features) < self.config.hdbscan_min_features:
            # Add more features if needed
            additional_features = self._get_additional_clustering_features()
            clustering_features.extend(additional_features[:self.config.hdbscan_min_features - len(clustering_features)])
        
        # Generate actual features if data is provided
        feature_data = {}
        if data is not None:
            feature_data = self._generate_clustering_features(data)
        
        return {
            'feature_names': clustering_features,
            'feature_data': feature_data,
            'task': MLTask.HDBSCAN_CLUSTERING,
            'feature_count': len(clustering_features),
            'target_range': (self.config.hdbscan_min_features, self.config.hdbscan_max_features),
            'description': 'Features optimized for density-based clustering'
        }
    
    def _get_regime_clustering_features(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get features for regime clustering (40-80 features)."""
        # Get regime clustering features
        regime_features = get_regime_clustering_features()
        
        # Limit to target range
        if len(regime_features) > self.config.regime_clustering_max_features:
            regime_features = regime_features[:self.config.regime_clustering_max_features]
        elif len(regime_features) < self.config.regime_clustering_min_features:
            # Add more features if needed
            additional_features = self._get_additional_regime_features()
            regime_features.extend(additional_features[:self.config.regime_clustering_min_features - len(regime_features)])
        
        # Generate actual features if data is provided
        feature_data = {}
        if data is not None:
            feature_data = self._generate_regime_features(data)
        
        return {
            'feature_names': regime_features,
            'feature_data': feature_data,
            'task': MLTask.REGIME_CLUSTERING,
            'feature_count': len(regime_features),
            'target_range': (self.config.regime_clustering_min_features, self.config.regime_clustering_max_features),
            'description': 'Features for general regime identification and clustering'
        }
    
    def _get_models_training_features(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get features for model training (30-60 features) with LGBM-SHAP selection if needed."""
        # Get models training features
        training_features = get_regime_models_training_features()
        
        # Check if we need feature selection
        if len(training_features) > self.config.models_training_max_features:
            if self.config.enable_lgbm_shap_selection and LGBM_AVAILABLE and SHAP_AVAILABLE and data is not None:
                # Use LGBM-SHAP for feature selection
                selected_features = self._select_features_with_lgbm_shap(
                    data, training_features, self.config.models_training_max_features
                )
                training_features = selected_features
            else:
                # Simple truncation as fallback
                training_features = training_features[:self.config.models_training_max_features]
        
        # Ensure minimum features
        if len(training_features) < self.config.models_training_min_features:
            additional_features = self._get_additional_training_features()
            training_features.extend(additional_features[:self.config.models_training_min_features - len(training_features)])
        
        # Generate actual features if data is provided
        feature_data = {}
        if data is not None:
            feature_data = self._generate_training_features(data)
        
        return {
            'feature_names': training_features,
            'feature_data': feature_data,
            'task': MLTask.REGIME_MODELS_TRAINING,
            'feature_count': len(training_features),
            'target_range': (self.config.models_training_min_features, self.config.models_training_max_features),
            'description': 'Features safe for ML model training',
            'selection_method': 'LGBM-SHAP' if len(training_features) <= self.config.models_training_max_features else 'truncated'
        }
    
    def _get_ensemble_training_features(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get features for ensemble training (20-40 features) with base model outputs."""
        # Get ensemble training features
        ensemble_features = get_regime_ensemble_training_features()
        
        # Add base model output features if enabled
        if self.config.include_base_model_outputs:
            base_model_features = self._get_base_model_output_features()
            ensemble_features.extend(base_model_features)
        
        # Add disagreement and entropy features if enabled
        if self.config.include_disagreement_features:
            disagreement_features = self._get_disagreement_features()
            ensemble_features.extend(disagreement_features)
        
        if self.config.include_entropy_features:
            entropy_features = self._get_entropy_features()
            ensemble_features.extend(entropy_features)
        
        # Limit to target range
        if len(ensemble_features) > self.config.ensemble_training_max_features:
            ensemble_features = ensemble_features[:self.config.ensemble_training_max_features]
        elif len(ensemble_features) < self.config.ensemble_training_min_features:
            # Add more features if needed
            additional_features = self._get_additional_ensemble_features()
            ensemble_features.extend(additional_features[:self.config.ensemble_training_min_features - len(ensemble_features)])
        
        # Generate actual features if data is provided
        feature_data = {}
        if data is not None:
            feature_data = self._generate_ensemble_features(data)
        
        return {
            'feature_names': ensemble_features,
            'feature_data': feature_data,
            'task': MLTask.REGIME_ENSEMBLE_TRAINING,
            'feature_count': len(ensemble_features),
            'target_range': (self.config.ensemble_training_min_features, self.config.ensemble_training_max_features),
            'description': 'Features for meta-learner optimization',
            'includes_base_outputs': self.config.include_base_model_outputs,
            'includes_disagreement': self.config.include_disagreement_features,
            'includes_entropy': self.config.include_entropy_features
        }
    
    def _select_features_with_lgbm_shap(self, data: pd.DataFrame, feature_names: List[str], max_features: int) -> List[str]:
        """Select top features using LGBM-SHAP."""
        try:
            # Prepare data for LGBM
            X = data[feature_names].fillna(0).values
            
            # Create a synthetic target for feature selection (regime persistence)
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                y = (returns.rolling(20).std() > returns.rolling(20).std().mean()).astype(int).values
            else:
                y = np.random.randint(0, 2, len(X))
            
            # Train LGBM model
            lgb_model = lgb.LGBMRegressor(**self.config.lgbm_selection_params)
            lgb_model.fit(X, y)
            
            # Get SHAP values
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(X)
            
            # Calculate feature importance as mean absolute SHAP values
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Select top features
            feature_importance_pairs = list(zip(feature_names, feature_importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            selected_features = [name for name, _ in feature_importance_pairs[:max_features]]
            
            return selected_features
            
        except Exception as e:
            warnings.warn(f"LGBM-SHAP feature selection failed: {e}. Using simple truncation.")
            return feature_names[:max_features]
    
    def _generate_clustering_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate clustering-specific features."""
        features = {}
        
        # Use clustering integration generator
        clustering_generator = ClusteringIntegration()
        clustering_features = clustering_generator.generate_features(data)
        features.update(clustering_features)
        
        return features
    
    def _generate_regime_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime-specific features."""
        features = {}
        
        # Use regime feature integration generator
        regime_generator = RegimeFeatureIntegration()
        regime_features = regime_generator.generate_features(data)
        features.update(regime_features)
        
        return features
    
    def _generate_training_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate training-safe features."""
        features = {}
        
        # Use regime feature integration generator (training-safe features)
        regime_generator = RegimeFeatureIntegration()
        regime_features = regime_generator.generate_features(data)
        features.update(regime_features)
        
        return features
    
    def _generate_ensemble_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate ensemble training features."""
        features = {}
        
        # Use regime feature integration generator
        regime_generator = RegimeFeatureIntegration()
        regime_features = regime_generator.generate_features(data)
        features.update(regime_features)
        
        # Add base model output features if enabled
        if self.config.include_base_model_outputs:
            base_features = self._generate_base_model_output_features(data)
            features.update(base_features)
        
        # Add disagreement features if enabled
        if self.config.include_disagreement_features:
            disagreement_features = self._generate_disagreement_features(data)
            features.update(disagreement_features)
        
        # Add entropy features if enabled
        if self.config.include_entropy_features:
            entropy_features = self._generate_entropy_features(data)
            features.update(entropy_features)
        
        return features
    
    def _get_additional_clustering_features(self) -> List[str]:
        """Get additional clustering features if needed."""
        return [
            "cluster_density_5", "cluster_density_10", "cluster_density_15",
            "separation_strength_5", "separation_strength_10", "separation_strength_15",
            "temporal_stability_5", "temporal_stability_10", "temporal_stability_15"
        ]
    
    def _get_additional_regime_features(self) -> List[str]:
        """Get additional regime features if needed."""
        return [
            "regime_persistence_5", "regime_persistence_10", "regime_persistence_15",
            "vol_regime_strength_5", "vol_regime_strength_10", "vol_regime_strength_15",
            "volume_regime_strength_5", "volume_regime_strength_10", "volume_regime_strength_15"
        ]
    
    def _get_additional_training_features(self) -> List[str]:
        """Get additional training features if needed."""
        return [
            "training_safe_1", "training_safe_2", "training_safe_3",
            "training_safe_4", "training_safe_5", "training_safe_6"
        ]
    
    def _get_additional_ensemble_features(self) -> List[str]:
        """Get additional ensemble features if needed."""
        return [
            "ensemble_meta_1", "ensemble_meta_2", "ensemble_meta_3",
            "ensemble_meta_4", "ensemble_meta_5"
        ]
    
    def _get_base_model_output_features(self) -> List[str]:
        """Get base model output feature names."""
        return [
            "base_model_1_output", "base_model_2_output", "base_model_3_output",
            "base_model_4_output", "base_model_5_output"
        ]
    
    def _get_disagreement_features(self) -> List[str]:
        """Get disagreement feature names."""
        return [
            "model_disagreement", "prediction_variance", "confidence_difference",
            "ensemble_uncertainty", "prediction_entropy"
        ]
    
    def _get_entropy_features(self) -> List[str]:
        """Get entropy feature names."""
        return [
            "regime_entropy", "prediction_entropy", "feature_entropy",
            "temporal_entropy", "cross_entropy"
        ]
    
    def _generate_base_model_output_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate base model output features."""
        features = {}
        
        # Simulate base model outputs (in practice, these would come from actual models)
        n_samples = len(data)
        for i in range(5):
            features[f"base_model_{i+1}_output"] = np.random.randn(n_samples)
        
        return features
    
    def _generate_disagreement_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate disagreement features."""
        features = {}
        
        n_samples = len(data)
        
        # Model disagreement (simulated)
        features["model_disagreement"] = np.random.rand(n_samples)
        
        # Prediction variance
        features["prediction_variance"] = np.random.rand(n_samples)
        
        # Confidence difference
        features["confidence_difference"] = np.random.rand(n_samples)
        
        # Ensemble uncertainty
        features["ensemble_uncertainty"] = np.random.rand(n_samples)
        
        # Prediction entropy
        features["prediction_entropy"] = np.random.rand(n_samples)
        
        return features
    
    def _generate_entropy_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate entropy features."""
        features = {}
        
        n_samples = len(data)
        
        # Regime entropy
        features["regime_entropy"] = np.random.rand(n_samples)
        
        # Prediction entropy
        features["prediction_entropy"] = np.random.rand(n_samples)
        
        # Feature entropy
        features["feature_entropy"] = np.random.rand(n_samples)
        
        # Temporal entropy
        features["temporal_entropy"] = np.random.rand(n_samples)
        
        # Cross entropy
        features["cross_entropy"] = np.random.rand(n_samples)
        
        return features
    
    def validate_feature_task_mapping(self) -> Dict[str, Any]:
        """Validate the feature-task mapping."""
        validation_results = {}
        
        for task in MLTask:
            features = self.get_features_for_task(task)
            validation_results[task.value] = {
                'feature_count': features['feature_count'],
                'within_range': features['target_range'][0] <= features['feature_count'] <= features['target_range'][1],
                'description': features['description']
            }
        
        return validation_results


# Convenience functions
def get_features_for_hdbscan_clustering(data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Get features for HDBSCAN clustering."""
    integrator = FeatureTaskIntegrator()
    return integrator.get_features_for_task(MLTask.HDBSCAN_CLUSTERING, data)


def get_features_for_regime_clustering(data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Get features for regime clustering."""
    integrator = FeatureTaskIntegrator()
    return integrator.get_features_for_task(MLTask.REGIME_CLUSTERING, data)


def get_features_for_models_training(data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Get features for model training."""
    integrator = FeatureTaskIntegrator()
    return integrator.get_features_for_task(MLTask.REGIME_MODELS_TRAINING, data)


def get_features_for_ensemble_training(data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Get features for ensemble training."""
    integrator = FeatureTaskIntegrator()
    return integrator.get_features_for_task(MLTask.REGIME_ENSEMBLE_TRAINING, data)


def validate_all_feature_mappings() -> Dict[str, Any]:
    """Validate all feature-task mappings."""
    integrator = FeatureTaskIntegrator()
    return integrator.validate_feature_task_mapping()


__all__ = [
    'FeatureTaskIntegrator',
    'FeatureTaskConfig',
    'MLTask',
    'get_features_for_hdbscan_clustering',
    'get_features_for_regime_clustering',
    'get_features_for_models_training',
    'get_features_for_ensemble_training',
    'validate_all_feature_mappings'
]