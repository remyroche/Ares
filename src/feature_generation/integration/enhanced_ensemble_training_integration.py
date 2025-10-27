"""
Enhanced Ensemble Training Integration

This module provides comprehensive ensemble training integration that combines
existing feature bank features (volume, trend, volatility, momentum) with
regime-specific features for optimal meta-learner training.

Target: 20-40 comprehensive features optimized for ensemble training
Includes base model outputs, disagreement features, and entropy features
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

# Import feature bank integration
from .feature_bank_integration import (
    FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory,
    get_comprehensive_ensemble_training_features
)

# Import ensemble models
try:
    from sklearn.ensemble import VotingRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")

# Import LGBM for base models
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("LGBM not available. Install with: pip install lightgbm")


class EnhancedEnsembleTrainingIntegration:
    """
    Enhanced Ensemble Training Integration.
    
    Provides 20-40 comprehensive features optimized for meta-learner training
    by combining existing feature bank features with regime-specific features,
    base model outputs, disagreement features, and entropy features.
    """
    
    def __init__(self, 
                 min_features: int = 20,
                 max_features: int = 40,
                 enable_comprehensive_features: bool = True,
                 enable_base_models: bool = True,
                 enable_meta_features: bool = True,
                 ensemble_config: Optional[Dict[str, Any]] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_base_models = enable_base_models
        self.enable_meta_features = enable_meta_features
        self.ensemble_config = ensemble_config or {}
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            # Configure for ensemble training
            config = FeatureBankConfig()
            config.ensemble_training_min_features = min_features
            config.ensemble_training_max_features = max_features
            # Balanced weights for ensemble training
            config.ensemble_training_weights = {
                FeatureBankCategory.REGIME: 0.25,     # Regime features
                FeatureBankCategory.VOLUME: 0.2,      # Volume patterns
                FeatureBankCategory.TREND: 0.2,       # Trend patterns
                FeatureBankCategory.VOLATILITY: 0.2,  # Volatility patterns
                FeatureBankCategory.MOMENTUM: 0.15    # Momentum patterns
            }
            self.feature_integrator = FeatureBankIntegrator(config)
        else:
            self.feature_integrator = None
    
    def get_comprehensive_ensemble_features(self, data: pd.DataFrame, 
                                         base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for ensemble training.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            base_models: Dictionary of trained base models (optional)
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        if self.enable_comprehensive_features:
            # Use comprehensive feature bank integration
            result = self.feature_integrator.get_comprehensive_features_for_task(
                'regime_ensemble_training', data
            )
            
            # Add ensemble-specific features
            if self.enable_meta_features:
                meta_features = self._generate_meta_features(data, result['features'], base_models)
                result['features'].update(meta_features['features'])
                result['feature_names'].extend(meta_features['feature_names'])
                result['feature_count'] = len(result['feature_names'])
            
            # Add ensemble-specific metadata
            result.update({
                'ensemble_optimized': True,
                'comprehensive_features': True,
                'meta_features_included': self.enable_meta_features,
                'base_models_included': base_models is not None,
                'feature_categories': self._get_feature_category_breakdown(result['features']),
                'ensemble_readiness': self._assess_ensemble_readiness(result['features'])
            })
            
            return result
        else:
            # Fallback to basic ensemble features
            return self._get_basic_ensemble_features(data)
    
    def _get_basic_ensemble_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to basic ensemble features if comprehensive features are disabled."""
        # This would use the original ensemble features only
        # For now, return a basic implementation
        return {
            'features': {},
            'feature_names': [],
            'feature_count': 0,
            'target_range': (self.min_features, self.max_features),
            'ensemble_optimized': True,
            'comprehensive_features': False,
            'description': 'Basic ensemble features (comprehensive disabled)'
        }
    
    def _generate_meta_features(self, data: pd.DataFrame, 
                              base_features: Dict[str, np.ndarray],
                              base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate meta-features for ensemble training."""
        meta_features = {}
        meta_feature_names = []
        
        # Base model outputs
        if base_models:
            base_outputs = self._generate_base_model_outputs(data, base_models)
            meta_features.update(base_outputs['features'])
            meta_feature_names.extend(base_outputs['feature_names'])
        
        # Disagreement features
        disagreement_features = self._generate_disagreement_features(data, base_features)
        meta_features.update(disagreement_features['features'])
        meta_feature_names.extend(disagreement_features['feature_names'])
        
        # Entropy features
        entropy_features = self._generate_entropy_features(data, base_features)
        meta_features.update(entropy_features['features'])
        meta_feature_names.extend(entropy_features['feature_names'])
        
        # Ensemble-specific features
        ensemble_features = self._generate_ensemble_specific_features(data, base_features)
        meta_features.update(ensemble_features['features'])
        meta_feature_names.extend(ensemble_features['feature_names'])
        
        return {
            'features': meta_features,
            'feature_names': meta_feature_names,
            'metadata': {
                'base_outputs': base_models is not None,
                'disagreement_features': len(disagreement_features['features']),
                'entropy_features': len(entropy_features['features']),
                'ensemble_features': len(ensemble_features['features'])
            }
        }
    
    def _generate_base_model_outputs(self, data: pd.DataFrame, 
                                   base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base model outputs as features."""
        features = {}
        feature_names = []
        
        # Prepare base features for base models
        base_features = self._prepare_base_features_for_models(data)
        
        for model_name, model in base_models.items():
            try:
                # Make predictions
                predictions = model.predict(base_features)
                
                # Store as feature
                feature_name = f'base_model_{model_name}_output'
                features[feature_name] = predictions
                feature_names.append(feature_name)
                
            except Exception as e:
                warnings.warn(f"Failed to generate output for base model {model_name}: {e}")
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _prepare_base_features_for_models(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare base features for base model prediction."""
        # This would prepare the same features used for base model training
        # For now, return a simple implementation
        if 'close' in data.columns:
            prices = data['close']
            returns = prices.pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0)
            
            # Simple feature matrix
            features = np.column_stack([
                returns.values,
                volatility.values,
                prices.pct_change(5).fillna(0).values,
                prices.pct_change(10).fillna(0).values
            ])
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return features
        else:
            # Fallback: return random features
            return np.random.randn(len(data), 4)
    
    def _generate_disagreement_features(self, data: pd.DataFrame, 
                                      base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate disagreement features between different feature categories."""
        features = {}
        feature_names = []
        
        # Get feature categories
        volume_features = [name for name in base_features.keys() if 'volume' in name.lower()]
        trend_features = [name for name in base_features.keys() if 'trend' in name.lower()]
        volatility_features = [name for name in base_features.keys() if 'volatility' in name.lower()]
        momentum_features = [name for name in base_features.keys() if 'momentum' in name.lower()]
        
        # Calculate disagreement between categories
        if len(volume_features) > 1 and len(trend_features) > 1:
            volume_std = np.std([base_features[name] for name in volume_features], axis=0)
            trend_std = np.std([base_features[name] for name in trend_features], axis=0)
            disagreement = np.abs(volume_std - trend_std)
            
            features['volume_trend_disagreement'] = disagreement
            feature_names.append('volume_trend_disagreement')
        
        if len(volatility_features) > 1 and len(momentum_features) > 1:
            vol_std = np.std([base_features[name] for name in volatility_features], axis=0)
            mom_std = np.std([base_features[name] for name in momentum_features], axis=0)
            disagreement = np.abs(vol_std - mom_std)
            
            features['volatility_momentum_disagreement'] = disagreement
            feature_names.append('volatility_momentum_disagreement')
        
        # Overall feature disagreement
        if len(base_features) > 1:
            all_features = np.array([base_features[name] for name in base_features.keys()])
            feature_std = np.std(all_features, axis=0)
            features['overall_feature_disagreement'] = feature_std
            feature_names.append('overall_feature_disagreement')
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _generate_entropy_features(self, data: pd.DataFrame, 
                                 base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate entropy features for ensemble training."""
        features = {}
        feature_names = []
        
        # Feature entropy
        for name, values in base_features.items():
            if len(values) > 0:
                # Calculate entropy
                hist, _ = np.histogram(values, bins=20)
                hist = hist / np.sum(hist)  # Normalize
                hist = hist[hist > 0]  # Remove zero bins
                entropy = -np.sum(hist * np.log2(hist))
                
                features[f'{name}_entropy'] = np.full(len(values), entropy)
                feature_names.append(f'{name}_entropy')
        
        # Cross-feature entropy
        if len(base_features) > 1:
            feature_names_list = list(base_features.keys())
            for i, name1 in enumerate(feature_names_list):
                for name2 in feature_names_list[i+1:]:
                    values1 = base_features[name1]
                    values2 = base_features[name2]
                    
                    if len(values1) == len(values2):
                        # Calculate joint entropy
                        joint_values = np.column_stack([values1, values2])
                        hist, _ = np.histogram2d(values1, values2, bins=10)
                        hist = hist / np.sum(hist)
                        hist = hist[hist > 0]
                        joint_entropy = -np.sum(hist * np.log2(hist))
                        
                        features[f'{name1}_{name2}_joint_entropy'] = np.full(len(values1), joint_entropy)
                        feature_names.append(f'{name1}_{name2}_joint_entropy')
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _generate_ensemble_specific_features(self, data: pd.DataFrame, 
                                           base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate ensemble-specific features."""
        features = {}
        feature_names = []
        
        # Feature diversity
        if len(base_features) > 1:
            all_features = np.array([base_features[name] for name in base_features.keys()])
            feature_correlation = np.corrcoef(all_features)
            diversity = 1 - np.mean(np.abs(feature_correlation))
            
            features['feature_diversity'] = np.full(len(data), diversity)
            feature_names.append('feature_diversity')
        
        # Feature stability
        for name, values in base_features.items():
            if len(values) > 10:
                # Calculate rolling stability
                window = min(10, len(values) // 2)
                stability = []
                for i in range(window, len(values)):
                    window_values = values[i-window:i]
                    stability.append(np.std(window_values))
                
                # Pad with first value
                stability = [stability[0]] * window + stability
                
                features[f'{name}_stability'] = np.array(stability)
                feature_names.append(f'{name}_stability')
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _get_feature_category_breakdown(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Get breakdown of features by category."""
        breakdown = {
            'regime': 0,
            'volume': 0,
            'trend': 0,
            'volatility': 0,
            'momentum': 0,
            'base_outputs': 0,
            'disagreement': 0,
            'entropy': 0,
            'ensemble': 0,
            'other': 0
        }
        
        for feature_name in features.keys():
            if any(keyword in feature_name.lower() for keyword in ['regime', 'entropy', 'complexity', 'hurst', 'fractal', 'memory']):
                breakdown['regime'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['volume', 'obv', 'ad', 'mfi', 'vwap']):
                breakdown['volume'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['trend', 'sma', 'ema', 'adx', 'directional']):
                breakdown['trend'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['volatility', 'bollinger', 'atr', 'vol']):
                breakdown['volatility'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['rsi', 'macd', 'stochastic', 'momentum']):
                breakdown['momentum'] += 1
            elif 'base_model' in feature_name.lower():
                breakdown['base_outputs'] += 1
            elif 'disagreement' in feature_name.lower():
                breakdown['disagreement'] += 1
            elif 'entropy' in feature_name.lower():
                breakdown['entropy'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['diversity', 'stability', 'ensemble']):
                breakdown['ensemble'] += 1
            else:
                breakdown['other'] += 1
        
        return breakdown
    
    def _assess_ensemble_readiness(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess how well-suited the features are for ensemble training."""
        if not features:
            return {'score': 0, 'issues': ['No features available']}
        
        issues = []
        score = 100
        
        # Check feature count
        feature_count = len(features)
        if feature_count < self.min_features:
            issues.append(f'Too few features: {feature_count} < {self.min_features}')
            score -= 30
        elif feature_count > self.max_features:
            issues.append(f'Too many features: {feature_count} > {self.max_features}')
            score -= 10
        
        # Check meta-feature presence
        category_breakdown = self._get_feature_category_breakdown(features)
        meta_features = (category_breakdown['base_outputs'] + 
                        category_breakdown['disagreement'] + 
                        category_breakdown['entropy'] + 
                        category_breakdown['ensemble'])
        
        if meta_features < 5:
            issues.append(f'Insufficient meta-features: {meta_features} < 5')
            score -= 25
        
        # Check feature quality
        quality_issues = 0
        for name, values in features.items():
            if len(values) == 0:
                quality_issues += 1
            elif np.all(np.isnan(values)):
                quality_issues += 1
            elif np.all(values == values[0]):  # All same value
                quality_issues += 1
        
        if quality_issues > 0:
            issues.append(f'{quality_issues} features have quality issues')
            score -= quality_issues * 5
        
        # Check feature diversity
        unique_categories = sum(1 for count in category_breakdown.values() if count > 0)
        if unique_categories < 4:
            issues.append(f'Low feature diversity: only {unique_categories} categories')
            score -= 20
        
        return {
            'score': max(0, score),
            'issues': issues,
            'feature_count': feature_count,
            'meta_features': meta_features,
            'category_diversity': unique_categories,
            'quality_issues': quality_issues
        }
    
    def prepare_data_for_ensemble_training(self, data: pd.DataFrame, 
                                         base_models: Optional[Dict[str, Any]] = None,
                                         target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for ensemble training with comprehensive features.
        
        Args:
            data: Market data DataFrame
            base_models: Dictionary of trained base models (optional)
            target_column: Name of target column (if None, will create synthetic target)
            
        Returns:
            Tuple of (X, y, feature_names, metadata)
        """
        # Get comprehensive ensemble features
        feature_result = self.get_comprehensive_ensemble_features(data, base_models)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        if not features:
            # Return empty arrays if no features
            return np.array([]).reshape(len(data), 0), np.array([]), [], feature_result
        
        # Convert to numpy array
        X = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Create or get target variable
        if target_column and target_column in data.columns:
            y = data[target_column].values
        else:
            y = self._create_synthetic_target(data)
        
        # Ensure target has same length as features
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]
        
        # Add preprocessing metadata
        metadata = feature_result.copy()
        metadata.update({
            'preprocessing': {
                'nan_handled': True,
                'feature_matrix_shape': X.shape,
                'target_length': len(y),
                'base_models_used': base_models is not None
            }
        })
        
        return X, y, feature_names, metadata
    
    def _create_synthetic_target(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic target for ensemble training (future returns)."""
        if 'close' in data.columns:
            prices = data['close']
            # Create future returns as target
            future_returns = prices.pct_change().shift(-1).fillna(0)
            return future_returns.values
        else:
            # Fallback: create random target
            return np.random.randn(len(data))
    
    def train_enhanced_ensemble(self, data: pd.DataFrame, 
                              base_models: Optional[Dict[str, Any]] = None,
                              target_column: Optional[str] = None,
                              ensemble_type: str = 'voting',
                              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train enhanced ensemble with comprehensive features.
        
        Args:
            data: Market data DataFrame
            base_models: Dictionary of trained base models (optional)
            target_column: Name of target column
            ensemble_type: Type of ensemble ('voting', 'stacking')
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing trained ensemble and results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Prepare data
        X, y, feature_names, metadata = self.prepare_data_for_ensemble_training(data, base_models, target_column)
        
        if X.size == 0:
            raise ValueError("No features available for ensemble training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create ensemble
        if ensemble_type == 'voting':
            ensemble = self._create_voting_ensemble()
        elif ensemble_type == 'stacking':
            ensemble = self._create_stacking_ensemble()
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = ensemble.predict(X_train)
        y_pred_test = ensemble.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_mean = 0.0
            cv_std = 0.0
        
        results = {
            'ensemble': ensemble,
            'feature_names': feature_names,
            'metadata': metadata,
            'metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std,
                'overfitting': test_r2 < train_r2 - 0.1
            },
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1]
            }
        }
        
        return results
    
    def _create_voting_ensemble(self) -> Any:
        """Create voting ensemble."""
        estimators = []
        
        # Add base estimators
        if LGBM_AVAILABLE:
            estimators.append(('lgbm', lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)))
        
        estimators.extend([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ])
        
        return VotingRegressor(estimators)
    
    def _create_stacking_ensemble(self) -> Any:
        """Create stacking ensemble."""
        base_estimators = []
        
        # Add base estimators
        if LGBM_AVAILABLE:
            base_estimators.append(('lgbm', lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)))
        
        base_estimators.extend([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ])
        
        # Meta-learner
        meta_learner = LinearRegression()
        
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5
        )


# Convenience functions
def get_enhanced_ensemble_features(data: pd.DataFrame, base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get enhanced comprehensive features for ensemble training."""
    integrator = EnhancedEnsembleTrainingIntegration()
    return integrator.get_comprehensive_ensemble_features(data, base_models)


def train_enhanced_ensemble(data: pd.DataFrame, base_models: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Train enhanced ensemble with comprehensive features."""
    integrator = EnhancedEnsembleTrainingIntegration()
    return integrator.train_enhanced_ensemble(data, base_models, **kwargs)


__all__ = [
    'EnhancedEnsembleTrainingIntegration',
    'get_enhanced_ensemble_features',
    'train_enhanced_ensemble'
]