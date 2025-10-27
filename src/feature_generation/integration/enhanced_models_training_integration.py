"""
Enhanced Models Training Integration

This module provides comprehensive models training integration that combines
existing feature bank features (volume, trend, volatility, momentum) with
regime-specific features for optimal ML model training.

Target: 30-60 comprehensive features optimized for ML model training
Uses LGBM-SHAP RFE for feature selection when > 60 features available
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
import json

# Import feature bank integration
from .feature_bank_integration import (
    FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory,
    get_comprehensive_models_training_features
)

# Import LGBM-SHAP RFE selector
from ...feature_selection.vectorbt_extensions.lgbm_shap_rfe_selector import (
    LGBMSHAPRFESelector, 
    LGBMSHAPRFEConfig,
    create_lgbm_shap_rfe_selector
)

# Import LGBM and SHAP for feature selection
try:
    import lightgbm as lgb
    import shap
    LGBM_SHAP_AVAILABLE = True
except ImportError:
    LGBM_SHAP_AVAILABLE = False
    warnings.warn("LGBM or SHAP not available. Install with: pip install lightgbm shap")

# Import ML models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)


class EnhancedModelsTrainingIntegration:
    """
    Enhanced Models Training Integration.
    
    Provides 30-60 comprehensive features optimized for ML model training
    by combining existing feature bank features with regime-specific features.
    Uses LGBM-SHAP for feature selection when > 60 features available.
    """
    
    def __init__(self, 
                 min_features: int = 30,
                 max_features: int = 60,
                 enable_comprehensive_features: bool = True,
                 enable_lgbm_shap_rfe: bool = True,
                 removal_percentage: float = 0.25,
                 lgbm_params: Optional[Dict[str, Any]] = None,
                 enable_detailed_logging: bool = True,
                 training_config: Optional[Dict[str, Any]] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_lgbm_shap_rfe = enable_lgbm_shap_rfe and LGBM_SHAP_AVAILABLE
        self.removal_percentage = removal_percentage
        self.enable_detailed_logging = enable_detailed_logging
        self.training_config = training_config or {}
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            # Configure for models training
            config = FeatureBankConfig()
            config.models_training_min_features = min_features
            config.models_training_max_features = max_features
            # Balanced weights for ML training
            config.models_training_weights = {
                FeatureBankCategory.REGIME: 0.3,      # Regime features
                FeatureBankCategory.VOLUME: 0.2,      # Volume patterns
                FeatureBankCategory.TREND: 0.2,       # Trend patterns
                FeatureBankCategory.VOLATILITY: 0.2,  # Volatility patterns
                FeatureBankCategory.MOMENTUM: 0.1     # Momentum patterns
            }
            self.feature_integrator = FeatureBankIntegrator(config)
        else:
            self.feature_integrator = None
        
        # Initialize LGBM-SHAP RFE selector
        if self.enable_lgbm_shap_rfe:
            rfe_config = LGBMSHAPRFEConfig(
                target_features=max_features,
                removal_percentage=removal_percentage,
                enable_detailed_logging=enable_detailed_logging
            )
            
            # Override LGBM parameters if provided
            if lgbm_params:
                rfe_config.lgb_params.update(lgbm_params)
            
            self.rfe_selector = create_lgbm_shap_rfe_selector(rfe_config)
        else:
            self.rfe_selector = None
    
    def get_comprehensive_training_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for ML model training.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        if self.enable_comprehensive_features:
            # Use comprehensive feature bank integration
            result = self.feature_integrator.get_comprehensive_features_for_task(
                'regime_models_training', data
            )
            
            # Add training-specific metadata
            result.update({
                'training_optimized': True,
                'comprehensive_features': True,
                'feature_categories': self._get_feature_category_breakdown(result['features']),
                'training_readiness': self._assess_training_readiness(result['features'])
            })
            
            return result
        else:
            # Fallback to basic training features
            return self._get_basic_training_features(data)
    
    def _get_basic_training_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to basic training features if comprehensive features are disabled."""
        # This would use the original training features only
        # For now, return a basic implementation
        return {
            'features': {},
            'feature_names': [],
            'feature_count': 0,
            'target_range': (self.min_features, self.max_features),
            'training_optimized': True,
            'comprehensive_features': False,
            'description': 'Basic training features (comprehensive disabled)'
        }
    
    def _get_feature_category_breakdown(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Get breakdown of features by category."""
        breakdown = {
            'regime': 0,
            'volume': 0,
            'trend': 0,
            'volatility': 0,
            'momentum': 0,
            'clustering': 0,
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
            elif any(keyword in feature_name.lower() for keyword in ['cluster', 'distance', 'separation', 'stability']):
                breakdown['clustering'] += 1
            else:
                breakdown['other'] += 1
        
        return breakdown
    
    def _assess_training_readiness(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess how well-suited the features are for ML model training."""
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
        category_breakdown = self._get_feature_category_breakdown(features)
        unique_categories = sum(1 for count in category_breakdown.values() if count > 0)
        if unique_categories < 3:
            issues.append(f'Low feature diversity: only {unique_categories} categories')
            score -= 20
        
        return {
            'score': max(0, score),
            'issues': issues,
            'feature_count': feature_count,
            'category_diversity': unique_categories,
            'quality_issues': quality_issues
        }
    
    def prepare_data_for_training(self, data: pd.DataFrame, 
                                target_column: Optional[str] = None,
                                create_synthetic_target: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for ML model training with comprehensive features.
        
        Args:
            data: Market data DataFrame
            target_column: Name of target column (if None, will create synthetic target)
            create_synthetic_target: Whether to create synthetic target if target_column is None
            
        Returns:
            Tuple of (X, y, feature_names, metadata)
        """
        # Get comprehensive features
        feature_result = self.get_comprehensive_training_features(data)
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
        elif create_synthetic_target:
            y = self._create_synthetic_target(data)
        else:
            raise ValueError("No target column specified and synthetic target creation disabled")
        
        # Ensure target has same length as features
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]
        
        # Apply LGBM-SHAP RFE feature selection if enabled and needed
        if self.enable_lgbm_shap_rfe and len(feature_names) > self.max_features:
            X, feature_names, selection_info = self._select_features_with_lgbm_shap_rfe(X, y, feature_names)
            feature_result['feature_selection'] = selection_info
        
        # Add preprocessing metadata
        metadata = feature_result.copy()
        metadata.update({
            'preprocessing': {
                'nan_handled': True,
                'feature_matrix_shape': X.shape,
                'target_length': len(y),
                'lgbm_shap_rfe_applied': self.enable_lgbm_shap_rfe and len(feature_names) > self.max_features
            }
        })
        
        return X, y, feature_names, metadata
    
    def select_features_for_regime_training(self, 
                                          data: pd.DataFrame,
                                          target_column: Optional[str] = None,
                                          custom_feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Select optimal features for regime models training using LGBM-SHAP RFE.
        
        Args:
            data: Input data with OHLCV columns
            target_column: Target column name (if None, will create synthetic target)
            custom_feature_names: Custom feature names (if None, will use generated names)
            
        Returns:
            Dictionary containing selection results and detailed report
        """
        tprint("🚀 Starting LGBM-SHAP RFE feature selection for regime training")
        
        # Step 1: Generate comprehensive features
        tprint("📊 Step 1: Generating comprehensive features")
        feature_result = self.get_comprehensive_training_features(data)
        
        if not feature_result.get('success', True):
            raise ValueError(f"Feature generation failed: {feature_result.get('error', 'Unknown error')}")
        
        features_df = feature_result['features']
        feature_names = feature_result['feature_names']
        
        tprint(f"✅ Generated {len(feature_names)} features")
        
        # Step 2: Prepare target variable
        tprint("🎯 Step 2: Preparing target variable")
        if target_column is None:
            # Create synthetic target (future returns)
            target = self._create_synthetic_target(data)
            tprint("📈 Created synthetic target (future returns)")
        else:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            target = data[target_column].values
            tprint(f"📈 Using provided target column: {target_column}")
        
        # Step 3: Align features and target
        tprint("🔗 Step 3: Aligning features and target")
        aligned_data = self._align_features_and_target(features_df, target, data)
        
        if aligned_data is None:
            raise ValueError("Failed to align features and target")
        
        X_aligned, y_aligned, feature_names_aligned = aligned_data
        
        tprint(f"✅ Aligned data: {X_aligned.shape[0]} samples, {X_aligned.shape[1]} features")
        
        # Step 4: Run LGBM-SHAP RFE selection
        tprint("🔍 Step 4: Running LGBM-SHAP RFE selection")
        selection_result = self.rfe_selector.select_features(
            X_aligned, 
            y_aligned, 
            feature_names_aligned,
            target_features=self.max_features
        )
        
        if not selection_result['success']:
            raise ValueError(f"Feature selection failed: {selection_result.get('error', 'Unknown error')}")
        
        # Step 5: Prepare final results
        tprint("📋 Step 5: Preparing final results")
        final_result = self._prepare_final_results(
            selection_result, 
            feature_result, 
            data,
            target_column
        )
        
        tprint_success("🎉 LGBM-SHAP RFE feature selection completed successfully!")
        
        return final_result
    
    def _align_features_and_target(self, 
                                 features_df: pd.DataFrame,
                                 target: np.ndarray,
                                 original_data: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Align features and target, handling NaN values."""
        tprint_debug("🔗 Aligning features and target")
        
        # Convert features to numpy array
        X = features_df.values
        feature_names = features_df.columns.tolist()
        
        # Find valid indices (no NaN in features or target)
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(target))
        
        if not np.any(valid_mask):
            tprint_warning("⚠️ No valid samples after alignment")
            return None
        
        # Filter valid samples
        X_valid = X[valid_mask]
        y_valid = target[valid_mask]
        
        tprint(f"📊 Valid samples after alignment: {len(y_valid)}")
        
        return X_valid, y_valid, feature_names
    
    def _prepare_final_results(self, 
                             selection_result: Dict[str, Any],
                             feature_result: Dict[str, Any],
                             original_data: pd.DataFrame,
                             target_column: Optional[str]) -> Dict[str, Any]:
        """Prepare final results with comprehensive information."""
        tprint_debug("📋 Preparing final results")
        
        # Extract key information
        selected_indices = selection_result['selected_features']
        selected_feature_names = selection_result['selected_feature_names']
        removed_features = selection_result['removed_features']
        selection_history = selection_result['selection_history']
        detailed_report = selection_result['report']
        
        # Create feature importance summary
        importance_summary = self._create_importance_summary(selection_history)
        
        # Create performance summary
        performance_summary = self._create_performance_summary(selection_history)
        
        # Prepare final results
        final_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'target_features': self.max_features,
            'removal_percentage': self.removal_percentage,
            'selected_features': {
                'indices': selected_indices,
                'names': selected_feature_names,
                'count': len(selected_feature_names)
            },
            'removed_features': {
                'names': removed_features,
                'count': len(removed_features)
            },
            'selection_process': {
                'total_iterations': len(selection_history),
                'history': selection_history,
                'importance_summary': importance_summary,
                'performance_summary': performance_summary
            },
            'detailed_report': detailed_report,
            'original_data_info': {
                'shape': original_data.shape,
                'columns': original_data.columns.tolist(),
                'target_column': target_column
            },
            'feature_generation_info': {
                'total_features_generated': len(feature_result['feature_names']),
                'feature_categories': feature_result.get('feature_categories', {}),
                'generation_metadata': feature_result.get('metadata', {})
            }
        }
        
        return final_result
    
    def _create_importance_summary(self, selection_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create importance summary from selection history."""
        if not selection_history:
            return {}
        
        # Extract importance scores from all iterations
        all_importance_scores = []
        for iteration in selection_history:
            if 'importance_scores' in iteration:
                all_importance_scores.extend(iteration['importance_scores'])
        
        if not all_importance_scores:
            return {}
        
        importance_array = np.array(all_importance_scores)
        
        return {
            'mean_importance': float(np.mean(importance_array)),
            'std_importance': float(np.std(importance_array)),
            'min_importance': float(np.min(importance_array)),
            'max_importance': float(np.max(importance_array)),
            'median_importance': float(np.median(importance_array))
        }
    
    def _create_performance_summary(self, selection_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create performance summary from selection history."""
        if not selection_history:
            return {}
        
        # Extract performance scores
        performances = [iteration['performance'] for iteration in selection_history if 'performance' in iteration]
        
        if not performances:
            return {}
        
        performance_array = np.array(performances)
        
        return {
            'mean_performance': float(np.mean(performance_array)),
            'std_performance': float(np.std(performance_array)),
            'min_performance': float(np.min(performance_array)),
            'max_performance': float(np.max(performance_array)),
            'final_performance': float(performances[-1]) if performances else None,
            'performance_trend': 'improving' if len(performances) > 1 and performances[-1] > performances[0] else 'stable'
        }
    
    def _create_synthetic_target(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic target for training (future returns)."""
        if 'close' in data.columns:
            prices = data['close']
            # Create future returns as target
            future_returns = prices.pct_change().shift(-1).fillna(0)
            return future_returns.values
        else:
            # Fallback: create random target
            return np.random.randn(len(data))
    
    def _select_features_with_lgbm_shap_rfe(self, X: np.ndarray, y: np.ndarray, 
                                          feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Select features using LGBM-SHAP RFE.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_X, selected_feature_names, selection_info)
        """
        if not self.enable_lgbm_shap_rfe or self.rfe_selector is None:
            # Fallback to variance-based selection
            return self._select_features_by_variance(X, feature_names)
        
        try:
            tprint("🔍 Starting LGBM-SHAP RFE feature selection")
            tprint(f"📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
            tprint(f"🎯 Target: {self.max_features} features")
            
            # Run LGBM-SHAP RFE selection
            selection_result = self.rfe_selector.select_features(
                X, y, feature_names, target_features=self.max_features
            )
            
            if not selection_result['success']:
                tprint_warning("⚠️ LGBM-SHAP RFE failed, falling back to variance-based selection")
                return self._select_features_by_variance(X, feature_names)
            
            # Extract results
            selected_indices = selection_result['selected_features']
            selected_feature_names = selection_result['selected_feature_names']
            selected_X = X[:, selected_indices]
            
            # Prepare selection info
            selection_info = {
                'method': 'lgbm_shap_rfe',
                'original_features': len(feature_names),
                'selected_features': len(selected_feature_names),
                'target_features': self.max_features,
                'removal_percentage': self.removal_percentage,
                'total_iterations': len(selection_result['selection_history']),
                'removed_features': selection_result['removed_features'],
                'selection_history': selection_result['selection_history'],
                'detailed_report': selection_result['report']
            }
            
            tprint_success(f"✅ LGBM-SHAP RFE completed: {len(selected_feature_names)} features selected")
            
            return selected_X, selected_feature_names, selection_info
            
        except Exception as e:
            tprint_warning(f"⚠️ LGBM-SHAP RFE failed: {e}. Falling back to variance-based selection.")
            return self._select_features_by_variance(X, feature_names)
    
    def _select_features_by_variance(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Fallback feature selection by variance."""
        tprint("📊 Using variance-based feature selection (fallback)")
        
        feature_variances = np.var(X, axis=0)
        top_indices = np.argsort(feature_variances)[-self.max_features:]
        
        selected_X = X[:, top_indices]
        selected_feature_names = [feature_names[i] for i in top_indices]
        
        selection_info = {
            'method': 'variance',
            'original_features': len(feature_names),
            'selected_features': len(selected_feature_names),
            'feature_variances': {feature_names[i]: float(feature_variances[i]) for i in top_indices}
        }
        
        tprint(f"✅ Variance selection completed: {len(selected_feature_names)} features selected")
        
        return selected_X, selected_feature_names, selection_info
    
    def train_enhanced_models(self, data: pd.DataFrame, 
                            target_column: Optional[str] = None,
                            test_size: float = 0.2,
                            models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train enhanced ML models with comprehensive features.
        
        Args:
            data: Market data DataFrame
            target_column: Name of target column
            test_size: Fraction of data to use for testing
            models: List of models to train ('lgbm', 'rf', 'gb', 'linear', 'ridge', 'lasso')
            
        Returns:
            Dictionary containing trained models and results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Prepare data
        X, y, feature_names, metadata = self.prepare_data_for_training(data, target_column)
        
        if X.size == 0:
            raise ValueError("No features available for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Default models
        if models is None:
            models = ['lgbm', 'rf', 'gb', 'linear', 'ridge', 'lasso']
        
        # Train models
        trained_models = {}
        model_results = {}
        
        for model_name in models:
            try:
                model, results = self._train_single_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                trained_models[model_name] = model
                model_results[model_name] = results
            except Exception as e:
                warnings.warn(f"Failed to train {model_name}: {e}")
                model_results[model_name] = {'error': str(e)}
        
        return {
            'models': trained_models,
            'results': model_results,
            'feature_names': feature_names,
            'metadata': metadata,
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1]
            }
        }
    
    def _train_single_model(self, model_name: str, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train a single model."""
        if model_name == 'lgbm' and LGBM_SHAP_AVAILABLE:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        elif model_name == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'gb':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_name == 'linear':
            model = LinearRegression()
        elif model_name == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_name == 'lasso':
            model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_mean = 0.0
            cv_std = 0.0
        
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'overfitting': test_r2 < train_r2 - 0.1
        }
        
        return model, results
    
    def get_feature_importance_for_training(self, data: pd.DataFrame, 
                                          model_name: str = 'lgbm',
                                          target_column: Optional[str] = None) -> Dict[str, float]:
        """
        Get feature importance for training using comprehensive features.
        
        Args:
            data: Market data DataFrame
            model_name: Name of model to use for importance calculation
            target_column: Name of target column
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Prepare data
        X, y, feature_names, metadata = self.prepare_data_for_training(data, target_column)
        
        if X.size == 0:
            return {}
        
        # Train model
        if model_name == 'lgbm' and LGBM_SHAP_AVAILABLE:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            model.fit(X, y)
            
            # Get feature importance
            importance_scores = model.feature_importances_
            
        elif model_name == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importance_scores = model.feature_importances_
            
        else:
            # Fallback to variance-based importance
            importance_scores = np.var(X, axis=0)
        
        # Create importance dictionary
        importance_dict = {}
        for i, feature_name in enumerate(feature_names):
            importance_dict[feature_name] = float(importance_scores[i])
        
        return importance_dict


# Convenience functions
def get_enhanced_training_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get enhanced comprehensive features for ML model training."""
    integrator = EnhancedModelsTrainingIntegration()
    return integrator.get_comprehensive_training_features(data)


def train_enhanced_models(data: pd.DataFrame, target_column: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Train enhanced ML models with comprehensive features."""
    integrator = EnhancedModelsTrainingIntegration()
    return integrator.train_enhanced_models(data, target_column, **kwargs)


__all__ = [
    'EnhancedModelsTrainingIntegration',
    'get_enhanced_training_features',
    'train_enhanced_models'
]