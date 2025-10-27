"""
Models Training Integration

This module provides integration between models training features and the training task.
It ensures 30-60 features are properly selected for ML model training, using LGBM-SHAP
for feature selection if more than 60 features are available.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

# Import feature categorization
from .regime_feature_categorization import FeatureUseCase, get_regime_models_training_features
from .regime_features import RegimeFeatureIntegration
from .feature_task_integration import FeatureTaskIntegrator, MLTask

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

# Import feature selection tools
try:
    from src.training.utils.feature_selection.selection_methods import FeatureImportanceRanker
    from src.feature_selection.advanced.multi_stage_rfe import MultiStageRFE
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    warnings.warn("Feature selection tools not available. Using basic selection methods.")


class ModelsTrainingIntegration:
    """
    Models Training Integration.
    
    Provides 30-60 features optimized for ML model training with LGBM-SHAP selection.
    """
    
    def __init__(self, 
                 min_features: int = 30,
                 max_features: int = 60,
                 enable_lgbm_shap: bool = True,
                 lgbm_params: Optional[Dict[str, Any]] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.enable_lgbm_shap = enable_lgbm_shap and LGBM_AVAILABLE and SHAP_AVAILABLE
        self.lgbm_params = lgbm_params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        # Initialize feature integrator
        self.feature_integrator = FeatureTaskIntegrator()
        
        # Initialize regime feature generator
        self.regime_generator = RegimeFeatureIntegration()
    
    def get_training_features(self, data: pd.DataFrame, 
                            target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get features optimized for model training.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            target: Optional target variable for feature selection
            
        Returns:
            Dictionary containing features and metadata
        """
        # Get features from the task integrator
        feature_result = self.feature_integrator.get_features_for_task(
            MLTask.REGIME_MODELS_TRAINING, data
        )
        
        # Generate actual training features
        training_features = self.regime_generator.generate_features(data)
        
        # Check if we need feature selection
        feature_names = list(training_features.keys())
        selection_method = 'none'
        
        if len(feature_names) > self.max_features:
            if self.enable_lgbm_shap and target is not None:
                # Use LGBM-SHAP for feature selection
                selected_features = self._select_features_with_lgbm_shap(
                    data, training_features, target, self.max_features
                )
                training_features = {name: training_features[name] for name in selected_features}
                feature_names = selected_features
                selection_method = 'lgbm_shap'
            elif FEATURE_SELECTION_AVAILABLE:
                # Use alternative feature selection
                selected_features = self._select_features_alternative(
                    data, training_features, target, self.max_features
                )
                training_features = {name: training_features[name] for name in selected_features}
                feature_names = selected_features
                selection_method = 'alternative'
            else:
                # Simple truncation as fallback
                feature_names = feature_names[:self.max_features]
                training_features = {name: training_features[name] for name in feature_names}
                selection_method = 'truncated'
        
        # Ensure minimum features
        if len(feature_names) < self.min_features:
            warnings.warn(f"Only {len(feature_names)} features available, minimum is {self.min_features}")
        
        return {
            'features': training_features,
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'target_range': (self.min_features, self.max_features),
            'training_optimized': True,
            'selection_method': selection_method,
            'description': 'Features optimized for ML model training'
        }
    
    def _select_features_with_lgbm_shap(self, data: pd.DataFrame, 
                                      features: Dict[str, np.ndarray],
                                      target: np.ndarray,
                                      max_features: int) -> List[str]:
        """Select features using LGBM-SHAP."""
        try:
            # Prepare feature matrix
            feature_names = list(features.keys())
            feature_matrix = np.column_stack([features[name] for name in feature_names])
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Train LGBM model
            lgb_model = lgb.LGBMRegressor(**self.lgbm_params)
            lgb_model.fit(feature_matrix, target)
            
            # Get SHAP values
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(feature_matrix)
            
            # Calculate feature importance as mean absolute SHAP values
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Select top features
            feature_importance_pairs = list(zip(feature_names, feature_importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            selected_features = [name for name, _ in feature_importance_pairs[:max_features]]
            
            return selected_features
            
        except Exception as e:
            warnings.warn(f"LGBM-SHAP feature selection failed: {e}. Using alternative method.")
            return self._select_features_alternative(data, features, target, max_features)
    
    def _select_features_alternative(self, data: pd.DataFrame,
                                   features: Dict[str, np.ndarray],
                                   target: Optional[np.ndarray],
                                   max_features: int) -> List[str]:
        """Select features using alternative methods."""
        feature_names = list(features.keys())
        
        if target is not None:
            # Use correlation with target for selection
            feature_matrix = np.column_stack([features[name] for name in feature_names])
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            correlations = []
            for i, name in enumerate(feature_names):
                corr = np.corrcoef(feature_matrix[:, i], target)[0, 1]
                correlations.append((name, abs(corr) if not np.isnan(corr) else 0))
            
            # Sort by correlation and select top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_features = [name for name, _ in correlations[:max_features]]
        else:
            # Use variance for selection
            variances = []
            for name in feature_names:
                var = np.var(features[name])
                variances.append((name, var))
            
            # Sort by variance and select top features
            variances.sort(key=lambda x: x[1], reverse=True)
            selected_features = [name for name, _ in variances[:max_features]]
        
        return selected_features
    
    def prepare_data_for_training(self, data: pd.DataFrame, 
                                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
        """
        Prepare data for model training.
        
        Args:
            data: Market data DataFrame
            target: Optional target variable
            
        Returns:
            Tuple of (feature_matrix, feature_names, target)
        """
        # Get training features
        feature_result = self.get_training_features(data, target)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        # Convert to numpy array
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
        
        return feature_matrix, feature_names, target
    
    def create_synthetic_target(self, data: pd.DataFrame, target_type: str = 'regime') -> np.ndarray:
        """
        Create synthetic target for feature selection if not provided.
        
        Args:
            data: Market data DataFrame
            target_type: Type of target to create ('regime', 'volatility', 'trend')
            
        Returns:
            Synthetic target array
        """
        if target_type == 'regime':
            # Create regime-based target using volatility clustering
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                volatility = returns.rolling(20).std()
                target = (volatility > volatility.rolling(50).mean()).astype(int).values
            else:
                target = np.random.randint(0, 2, len(data))
        
        elif target_type == 'volatility':
            # Create volatility-based target
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                volatility = returns.rolling(20).std()
                target = volatility.values
            else:
                target = np.random.rand(len(data))
        
        elif target_type == 'trend':
            # Create trend-based target
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                trend = returns.rolling(20).mean()
                target = (trend > 0).astype(int).values
            else:
                target = np.random.randint(0, 2, len(data))
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return target
    
    def train_model_with_features(self, data: pd.DataFrame, 
                                target: Optional[np.ndarray] = None,
                                model_type: str = 'lgbm') -> Dict[str, Any]:
        """
        Train a model using the selected features.
        
        Args:
            data: Market data DataFrame
            target: Target variable
            model_type: Type of model to train ('lgbm', 'rf', 'xgb')
            
        Returns:
            Dictionary containing training results
        """
        # Create synthetic target if not provided
        if target is None:
            target = self.create_synthetic_target(data)
        
        # Prepare data
        feature_matrix, feature_names, target = self.prepare_data_for_training(data, target)
        
        # Train model based on type
        if model_type == 'lgbm' and LGBM_AVAILABLE:
            model = lgb.LGBMRegressor(**self.lgbm_params)
            model.fit(feature_matrix, target)
            
            # Get feature importance
            feature_importance = model.feature_importances_
            importance_dict = dict(zip(feature_names, feature_importance))
            
        else:
            # Fallback to simple linear model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(feature_matrix, target)
            
            # Get feature importance (coefficients)
            feature_importance = np.abs(model.coef_)
            importance_dict = dict(zip(feature_names, feature_importance))
        
        # Calculate performance metrics
        predictions = model.predict(feature_matrix)
        mse = np.mean((predictions - target) ** 2)
        r2 = 1 - (mse / np.var(target))
        
        return {
            'model': model,
            'feature_names': feature_names,
            'feature_importance': importance_dict,
            'predictions': predictions,
            'target': target,
            'mse': mse,
            'r2': r2,
            'model_type': model_type,
            'n_features': len(feature_names)
        }
    
    def analyze_feature_importance(self, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feature importance from training results.
        
        Args:
            training_result: Result from train_model_with_features
            
        Returns:
            Dictionary containing feature analysis
        """
        feature_importance = training_result['feature_importance']
        feature_names = training_result['feature_names']
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate statistics
        importance_values = list(feature_importance.values())
        
        analysis = {
            'top_features': sorted_features[:10],
            'bottom_features': sorted_features[-10:],
            'importance_stats': {
                'mean': np.mean(importance_values),
                'std': np.std(importance_values),
                'min': np.min(importance_values),
                'max': np.max(importance_values),
                'median': np.median(importance_values)
            },
            'feature_categories': self._categorize_features_by_importance(feature_importance),
            'total_features': len(feature_names)
        }
        
        return analysis
    
    def _categorize_features_by_importance(self, feature_importance: Dict[str, float]) -> Dict[str, List[str]]:
        """Categorize features by importance level."""
        importance_values = list(feature_importance.values())
        mean_importance = np.mean(importance_values)
        std_importance = np.std(importance_values)
        
        high_threshold = mean_importance + std_importance
        low_threshold = mean_importance - std_importance
        
        categories = {
            'high_importance': [],
            'medium_importance': [],
            'low_importance': []
        }
        
        for feature, importance in feature_importance.items():
            if importance >= high_threshold:
                categories['high_importance'].append(feature)
            elif importance <= low_threshold:
                categories['low_importance'].append(feature)
            else:
                categories['medium_importance'].append(feature)
        
        return categories


# Convenience functions
def get_models_training_features(data: pd.DataFrame, target: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Get features for model training."""
    integrator = ModelsTrainingIntegration()
    return integrator.get_training_features(data, target)


def train_model_with_training_features(data: pd.DataFrame, 
                                     target: Optional[np.ndarray] = None,
                                     model_type: str = 'lgbm') -> Dict[str, Any]:
    """Train a model using training features."""
    integrator = ModelsTrainingIntegration()
    return integrator.train_model_with_features(data, target, model_type)


__all__ = [
    'ModelsTrainingIntegration',
    'get_models_training_features',
    'train_model_with_training_features'
]