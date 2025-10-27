"""
Ensemble Training Integration

This module provides integration between ensemble training features and the ensemble training task.
It ensures 20-40 features are properly selected for meta-learner optimization, including
base model outputs, disagreement features, and entropy features.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

# Import feature categorization
from .regime_feature_categorization import FeatureUseCase, get_regime_ensemble_training_features
from .regime_features import RegimeFeatureIntegration
from .feature_task_integration import FeatureTaskIntegrator, MLTask

# Import ensemble methods
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")


class EnsembleTrainingIntegration:
    """
    Ensemble Training Integration.
    
    Provides 20-40 features optimized for meta-learner training, including
    base model outputs, disagreement features, and entropy features.
    """
    
    def __init__(self, 
                 min_features: int = 20,
                 max_features: int = 40,
                 include_base_outputs: bool = True,
                 include_disagreement: bool = True,
                 include_entropy: bool = True,
                 n_base_models: int = 5):
        self.min_features = min_features
        self.max_features = max_features
        self.include_base_outputs = include_base_outputs
        self.include_disagreement = include_disagreement
        self.include_entropy = include_entropy
        self.n_base_models = n_base_models
        
        # Initialize feature integrator
        self.feature_integrator = FeatureTaskIntegrator()
        
        # Initialize regime feature generator
        self.regime_generator = RegimeFeatureIntegration()
    
    def get_ensemble_features(self, data: pd.DataFrame, 
                            base_model_outputs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Get features optimized for ensemble training.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            base_model_outputs: Optional base model outputs
            
        Returns:
            Dictionary containing features and metadata
        """
        # Get base ensemble features
        feature_result = self.feature_integrator.get_features_for_task(
            MLTask.REGIME_ENSEMBLE_TRAINING, data
        )
        
        # Generate actual ensemble features
        ensemble_features = self.regime_generator.generate_features(data)
        
        # Add base model outputs if enabled
        if self.include_base_outputs:
            if base_model_outputs is not None:
                ensemble_features.update(base_model_outputs)
            else:
                # Generate synthetic base model outputs
                base_outputs = self._generate_synthetic_base_outputs(data)
                ensemble_features.update(base_outputs)
        
        # Add disagreement features if enabled
        if self.include_disagreement:
            disagreement_features = self._generate_disagreement_features(data, ensemble_features)
            ensemble_features.update(disagreement_features)
        
        # Add entropy features if enabled
        if self.include_entropy:
            entropy_features = self._generate_entropy_features(data, ensemble_features)
            ensemble_features.update(entropy_features)
        
        # Ensure we have the right number of features
        feature_names = list(ensemble_features.keys())
        if len(feature_names) > self.max_features:
            # Select top features by ensemble relevance
            feature_scores = self._score_features_for_ensemble_relevance(ensemble_features)
            
            # Sort by relevance score and select top features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [name for name, _ in sorted_features[:self.max_features]]
            
            # Filter ensemble features
            filtered_features = {name: ensemble_features[name] for name in selected_features}
            ensemble_features = filtered_features
            feature_names = selected_features
        
        # Ensure minimum features
        if len(feature_names) < self.min_features:
            warnings.warn(f"Only {len(feature_names)} features available, minimum is {self.min_features}")
        
        return {
            'features': ensemble_features,
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'target_range': (self.min_features, self.max_features),
            'ensemble_optimized': True,
            'includes_base_outputs': self.include_base_outputs,
            'includes_disagreement': self.include_disagreement,
            'includes_entropy': self.include_entropy,
            'description': 'Features optimized for meta-learner training'
        }
    
    def _generate_synthetic_base_outputs(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate synthetic base model outputs for demonstration."""
        n_samples = len(data)
        base_outputs = {}
        
        # Generate synthetic outputs for different model types
        model_types = ['lgbm', 'rf', 'xgb', 'linear', 'neural']
        
        for i, model_type in enumerate(model_types[:self.n_base_models]):
            # Create realistic-looking synthetic outputs
            if 'close' in data.columns:
                # Base on price movements
                returns = data['close'].pct_change().fillna(0)
                base_output = returns.rolling(5).mean() + np.random.normal(0, 0.001, n_samples)
            else:
                # Random outputs
                base_output = np.random.normal(0, 1, n_samples)
            
            base_outputs[f'base_model_{model_type}_output'] = base_output
        
        return base_outputs
    
    def _generate_disagreement_features(self, data: pd.DataFrame, 
                                      base_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate disagreement features between base models."""
        disagreement_features = {}
        
        # Find base model outputs
        base_outputs = {name: values for name, values in base_features.items() 
                       if 'base_model' in name and 'output' in name}
        
        if len(base_outputs) >= 2:
            base_values = list(base_outputs.values())
            
            # Model disagreement (variance across base models)
            disagreement_features['model_disagreement'] = np.var(base_values, axis=0)
            
            # Prediction variance
            disagreement_features['prediction_variance'] = np.var(base_values, axis=0)
            
            # Confidence difference (max - min)
            disagreement_features['confidence_difference'] = np.max(base_values, axis=0) - np.min(base_values, axis=0)
            
            # Ensemble uncertainty (standard deviation)
            disagreement_features['ensemble_uncertainty'] = np.std(base_values, axis=0)
            
            # Prediction entropy (based on distribution)
            disagreement_features['prediction_entropy'] = self._calculate_prediction_entropy(base_values)
        
        return disagreement_features
    
    def _calculate_prediction_entropy(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate prediction entropy across base models."""
        n_samples = len(predictions[0])
        entropy = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample_predictions = [pred[i] for pred in predictions]
            
            # Create histogram
            hist, _ = np.histogram(sample_predictions, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            
            if len(hist) > 0:
                # Calculate entropy
                entropy[i] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _generate_entropy_features(self, data: pd.DataFrame, 
                                 base_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate entropy-based features."""
        entropy_features = {}
        
        # Regime entropy (based on price volatility)
        if 'close' in data.columns:
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std()
            entropy_features['regime_entropy'] = self._calculate_rolling_entropy(volatility)
        
        # Feature entropy (across all features)
        if base_features:
            feature_matrix = np.column_stack(list(base_features.values()))
            entropy_features['feature_entropy'] = self._calculate_feature_entropy(feature_matrix)
        
        # Temporal entropy (based on time series)
        if 'close' in data.columns:
            prices = data['close']
            entropy_features['temporal_entropy'] = self._calculate_temporal_entropy(prices)
        
        # Cross entropy (between different feature types)
        if len(base_features) >= 2:
            feature_names = list(base_features.keys())
            entropy_features['cross_entropy'] = self._calculate_cross_entropy(
                [base_features[name] for name in feature_names[:2]]
            )
        
        return entropy_features
    
    def _calculate_rolling_entropy(self, series: pd.Series, window: int = 20) -> np.ndarray:
        """Calculate rolling entropy of a time series."""
        entropy = np.zeros(len(series))
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i].dropna()
            if len(window_data) > 1:
                hist, _ = np.histogram(window_data, bins=10, density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    entropy[i] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _calculate_feature_entropy(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Calculate entropy across features for each sample."""
        n_samples = feature_matrix.shape[0]
        entropy = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample_features = feature_matrix[i, :]
            hist, _ = np.histogram(sample_features, bins=10, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy[i] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _calculate_temporal_entropy(self, prices: pd.Series) -> np.ndarray:
        """Calculate temporal entropy of price series."""
        returns = prices.pct_change().fillna(0)
        return self._calculate_rolling_entropy(returns, window=20)
    
    def _calculate_cross_entropy(self, feature_arrays: List[np.ndarray]) -> np.ndarray:
        """Calculate cross entropy between two feature arrays."""
        if len(feature_arrays) != 2:
            return np.zeros(len(feature_arrays[0]))
        
        arr1, arr2 = feature_arrays
        n_samples = len(arr1)
        cross_entropy = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Calculate KL divergence between the two features
            p = np.array([arr1[i], 1 - arr1[i]])
            q = np.array([arr2[i], 1 - arr2[i]])
            
            # Avoid division by zero
            p = np.maximum(p, 1e-10)
            q = np.maximum(q, 1e-10)
            
            cross_entropy[i] = np.sum(p * np.log(p / q))
        
        return cross_entropy
    
    def _score_features_for_ensemble_relevance(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Score features for ensemble relevance."""
        scores = {}
        
        for name, values in features.items():
            # Calculate ensemble relevance based on:
            # 1. Variance (higher variance = more information)
            # 2. Non-linearity (ensemble features should capture non-linear relationships)
            # 3. Stability (ensemble features should be stable)
            
            variance_score = np.var(values)
            
            # Non-linearity (based on deviation from linear trend)
            if len(values) > 2:
                x = np.arange(len(values))
                linear_fit = np.polyfit(x, values, 1)
                linear_pred = np.polyval(linear_fit, x)
                non_linearity = np.mean((values - linear_pred) ** 2)
            else:
                non_linearity = 0
            
            # Stability (based on autocorrelation)
            if len(values) > 1:
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                stability_score = abs(autocorr) if not np.isnan(autocorr) else 0
            else:
                stability_score = 0
            
            # Combined score
            scores[name] = variance_score + non_linearity + stability_score
        
        return scores
    
    def prepare_data_for_ensemble_training(self, data: pd.DataFrame, 
                                         target: Optional[np.ndarray] = None,
                                         base_model_outputs: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
        """
        Prepare data for ensemble training.
        
        Args:
            data: Market data DataFrame
            target: Optional target variable
            base_model_outputs: Optional base model outputs
            
        Returns:
            Tuple of (feature_matrix, feature_names, target)
        """
        # Get ensemble features
        feature_result = self.get_ensemble_features(data, base_model_outputs)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        # Convert to numpy array
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
        
        return feature_matrix, feature_names, target
    
    def train_ensemble_meta_learner(self, data: pd.DataFrame, 
                                   target: Optional[np.ndarray] = None,
                                   base_model_outputs: Optional[Dict[str, np.ndarray]] = None,
                                   meta_learner_type: str = 'linear') -> Dict[str, Any]:
        """
        Train ensemble meta-learner.
        
        Args:
            data: Market data DataFrame
            target: Target variable
            base_model_outputs: Base model outputs
            meta_learner_type: Type of meta-learner ('linear', 'rf', 'gb')
            
        Returns:
            Dictionary containing training results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Create synthetic target if not provided
        if target is None:
            target = self._create_synthetic_ensemble_target(data)
        
        # Prepare data
        feature_matrix, feature_names, target = self.prepare_data_for_ensemble_training(
            data, target, base_model_outputs
        )
        
        # Train meta-learner based on type
        if meta_learner_type == 'linear':
            meta_learner = LinearRegression()
        elif meta_learner_type == 'ridge':
            meta_learner = Ridge(alpha=1.0)
        elif meta_learner_type == 'lasso':
            meta_learner = Lasso(alpha=0.1)
        elif meta_learner_type == 'rf':
            meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
        elif meta_learner_type == 'gb':
            meta_learner = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")
        
        meta_learner.fit(feature_matrix, target)
        
        # Get predictions
        predictions = meta_learner.predict(feature_matrix)
        
        # Calculate performance metrics
        mse = mean_squared_error(target, predictions)
        r2 = r2_score(target, predictions)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(meta_learner, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, meta_learner.feature_importances_))
        elif hasattr(meta_learner, 'coef_'):
            feature_importance = dict(zip(feature_names, np.abs(meta_learner.coef_)))
        
        return {
            'meta_learner': meta_learner,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'predictions': predictions,
            'target': target,
            'mse': mse,
            'r2': r2,
            'meta_learner_type': meta_learner_type,
            'n_features': len(feature_names)
        }
    
    def _create_synthetic_ensemble_target(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic target for ensemble training."""
        if 'close' in data.columns:
            # Create target based on future returns
            returns = data['close'].pct_change().fillna(0)
            future_returns = returns.shift(-5).fillna(0)  # 5-period ahead returns
            target = future_returns.values
        else:
            target = np.random.randn(len(data))
        
        return target
    
    def analyze_ensemble_performance(self, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze ensemble performance.
        
        Args:
            training_result: Result from train_ensemble_meta_learner
            
        Returns:
            Dictionary containing performance analysis
        """
        predictions = training_result['predictions']
        target = training_result['target']
        feature_importance = training_result['feature_importance']
        
        # Calculate additional metrics
        mae = np.mean(np.abs(predictions - target))
        rmse = np.sqrt(np.mean((predictions - target) ** 2))
        
        # Analyze feature importance
        importance_analysis = {}
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            importance_analysis = {
                'top_features': sorted_features[:10],
                'base_model_features': [f for f, _ in sorted_features if 'base_model' in f],
                'disagreement_features': [f for f, _ in sorted_features if any(x in f for x in ['disagreement', 'variance', 'uncertainty'])],
                'entropy_features': [f for f, _ in sorted_features if 'entropy' in f]
            }
        
        return {
            'mse': training_result['mse'],
            'r2': training_result['r2'],
            'mae': mae,
            'rmse': rmse,
            'feature_importance_analysis': importance_analysis,
            'meta_learner_type': training_result['meta_learner_type'],
            'n_features': training_result['n_features']
        }


# Convenience functions
def get_ensemble_training_features(data: pd.DataFrame, 
                                 base_model_outputs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Get features for ensemble training."""
    integrator = EnsembleTrainingIntegration()
    return integrator.get_ensemble_features(data, base_model_outputs)


def train_ensemble_meta_learner(data: pd.DataFrame, 
                               target: Optional[np.ndarray] = None,
                               base_model_outputs: Optional[Dict[str, np.ndarray]] = None,
                               meta_learner_type: str = 'linear') -> Dict[str, Any]:
    """Train ensemble meta-learner."""
    integrator = EnsembleTrainingIntegration()
    return integrator.train_ensemble_meta_learner(data, target, base_model_outputs, meta_learner_type)


__all__ = [
    'EnsembleTrainingIntegration',
    'get_ensemble_training_features',
    'train_ensemble_meta_learner'
]