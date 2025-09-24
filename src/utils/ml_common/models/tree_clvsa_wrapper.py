"""
Tree-Specific CLVSA Wrapper for Enhanced Tree Models

This module provides specialized CLVSA (Contextual Learning with Variable Structure Adaptation)
wrappers specifically designed for tree-based models (RandomForest, XGBoost, LightGBM, CatBoost).

Key Features:
1. Tree-specific attention mechanisms
2. Feature importance weighting
3. Temporal attention patterns for time series
4. Regime-aware feature selection
5. Ensemble-specific optimizations
6. Memory-efficient implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TreeCLVSAConfig:
    """Configuration for Tree CLVSA wrapper."""
    
    def __init__(self,
                 attention_dim: int = 64,
                 use_temporal_attention: bool = True,
                 regime_aware: bool = True,
                 attention_dropout: float = 0.1,
                 feature_selection_method: str = 'mutual_info',
                 temporal_window_size: int = 20,
                 regime_embedding_dim: int = 16,
                 ensemble_attention: bool = True,
                 memory_efficient: bool = True):
        """Initialize Tree CLVSA configuration.
        
        Args:
            attention_dim: Attention dimension for feature weighting
            use_temporal_attention: Whether to use temporal attention patterns
            regime_aware: Whether to use regime-aware attention
            attention_dropout: Attention dropout rate
            feature_selection_method: Method for feature selection ('mutual_info', 'tree_importance', 'correlation')
            temporal_window_size: Window size for temporal attention
            regime_embedding_dim: Dimension for regime embeddings
            ensemble_attention: Whether to use ensemble-specific attention
            memory_efficient: Whether to use memory-efficient implementations
        """
        self.attention_dim = attention_dim
        self.use_temporal_attention = use_temporal_attention
        self.regime_aware = regime_aware
        self.attention_dropout = attention_dropout
        self.feature_selection_method = feature_selection_method
        self.temporal_window_size = temporal_window_size
        self.regime_embedding_dim = regime_embedding_dim
        self.ensemble_attention = ensemble_attention
        self.memory_efficient = memory_efficient


class TreeCLVSAWrapper(BaseEstimator, RegressorMixin):
    """Enhanced CLVSA wrapper specifically designed for tree-based models."""
    
    def __init__(self, base_model, config: Optional[TreeCLVSAConfig] = None):
        """Initialize Tree CLVSA wrapper.
        
        Args:
            base_model: Tree-based model (RandomForest, XGBoost, LightGBM, CatBoost)
            config: Tree CLVSA configuration
        """
        self.base_model = base_model
        self.config = config or TreeCLVSAConfig()
        
        # Attention components
        self.feature_attention_weights = None
        self.temporal_attention_weights = None
        self.regime_attention_weights = None
        self.ensemble_attention_weights = None
        
        # Scalers
        self.input_scaler = RobustScaler() if self.config.memory_efficient else StandardScaler()
        self.attention_scaler = StandardScaler()
        
        # Feature selection
        self.selected_features = None
        self.feature_importance_scores = None
        
        # Regime components
        self.regime_embeddings = None
        self.regime_classifier = None
        
        # Performance tracking
        self.attention_performance = {}
        self.training_time = 0.0
        
        logger.info(f"🌲 Tree CLVSA wrapper initialized for {type(base_model).__name__}")
        logger.info(f"   Attention dim: {self.config.attention_dim}")
        logger.info(f"   Temporal attention: {self.config.use_temporal_attention}")
        logger.info(f"   Regime aware: {self.config.regime_aware}")
    
    def _compute_tree_specific_attention(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute tree-specific attention weights using multiple methods."""
        try:
            logger.info("🔍 Computing tree-specific attention weights...")
            
            # Method 1: Mutual information
            if self.config.feature_selection_method == 'mutual_info':
                if len(np.unique(y)) <= 10:
                    mi_scores = mutual_info_classif(X, y, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                attention_weights = mi_scores / (np.max(mi_scores) + 1e-8)
            
            # Method 2: Tree importance
            elif self.config.feature_selection_method == 'tree_importance':
                # Use a quick RandomForest to get feature importance
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                rf.fit(X, y)
                attention_weights = rf.feature_importances_
            
            # Method 3: Correlation-based
            elif self.config.feature_selection_method == 'correlation':
                # Compute correlation with target
                if len(y.shape) == 1:
                    correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
                else:
                    correlations = np.abs([np.corrcoef(X[:, i], y[:, 0])[0, 1] for i in range(X.shape[1])])
                attention_weights = correlations / (np.max(correlations) + 1e-8)
            
            else:
                # Default: uniform weights
                attention_weights = np.ones(X.shape[1]) / X.shape[1]
            
            # Apply softmax-like transformation
            attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
            
            # Add small noise to prevent zero weights
            attention_weights = attention_weights + np.random.normal(0, 0.01, size=attention_weights.shape)
            attention_weights = attention_weights / np.sum(attention_weights)
            
            logger.info(f"✅ Tree attention weights computed: {len(attention_weights)} features")
            return attention_weights
            
        except Exception as e:
            logger.warning(f"⚠️ Tree attention computation failed: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _compute_temporal_attention(self, X: np.ndarray) -> np.ndarray:
        """Compute temporal attention patterns for time series data."""
        try:
            if not self.config.use_temporal_attention:
                return np.ones(X.shape[1]) / X.shape[1]
            
            logger.info("⏰ Computing temporal attention patterns...")
            
            temporal_weights = np.zeros(X.shape[1])
            window_size = min(self.config.temporal_window_size, len(X) // 10)
            
            for i in range(X.shape[1]):
                feature_series = X[:, i]
                
                # Calculate temporal patterns
                temporal_importance = 0.0
                
                # Autocorrelation analysis
                autocorrs = []
                for lag in range(1, min(10, len(feature_series) // 2)):
                    if len(feature_series) > lag:
                        try:
                            autocorr = np.corrcoef(feature_series[:-lag], feature_series[lag:])[0, 1]
                            if not np.isnan(autocorr):
                                autocorrs.append(abs(autocorr))
                        except:
                            continue
                
                if autocorrs:
                    temporal_importance = np.mean(autocorrs)
                
                # Rolling window variance
                if len(feature_series) > window_size:
                    rolling_var = pd.Series(feature_series).rolling(window_size).var().fillna(0)
                    temporal_importance += np.mean(rolling_var) / (np.var(feature_series) + 1e-8)
                
                temporal_weights[i] = temporal_importance
            
            # Normalize temporal weights
            if np.sum(temporal_weights) > 0:
                temporal_weights = temporal_weights / np.sum(temporal_weights)
            else:
                temporal_weights = np.ones(X.shape[1]) / X.shape[1]
            
            logger.info("✅ Temporal attention patterns computed")
            return temporal_weights
            
        except Exception as e:
            logger.warning(f"⚠️ Temporal attention computation failed: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _compute_regime_attention(self, X: np.ndarray, y: np.ndarray, 
                                regimes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Compute regime-specific attention weights."""
        try:
            if not self.config.regime_aware or regimes is None:
                return {}
            
            logger.info("🏛️ Computing regime-specific attention...")
            
            unique_regimes = np.unique(regimes)
            regime_weights = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) < 10:  # Need sufficient data
                    continue
                
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]
                
                # Compute attention weights for this regime
                regime_attention = self._compute_tree_specific_attention(X_regime, y_regime)
                regime_weights[str(regime)] = regime_attention
                
                logger.info(f"   Regime {regime}: {len(regime_attention)} features")
            
            logger.info(f"✅ Regime attention computed for {len(regime_weights)} regimes")
            return regime_weights
            
        except Exception as e:
            logger.warning(f"⚠️ Regime attention computation failed: {e}")
            return {}
    
    def _compute_ensemble_attention(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ensemble-specific attention for tree models."""
        try:
            if not self.config.ensemble_attention:
                return np.ones(X.shape[1]) / X.shape[1]
            
            logger.info("🌳 Computing ensemble attention...")
            
            # Use cross-validation to get stable feature importance
            cv_scores = []
            feature_importances = []
            
            # Multiple CV folds for stability
            n_folds = min(5, len(X) // 20)  # Adaptive fold count
            if n_folds < 2:
                n_folds = 2
            
            for fold in range(n_folds):
                # Create fold split
                fold_size = len(X) // n_folds
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(X)
                
                X_fold = X[start_idx:end_idx]
                y_fold = y[start_idx:end_idx]
                
                if len(X_fold) < 10:
                    continue
                
                # Quick tree model for feature importance
                rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
                rf.fit(X_fold, y_fold)
                feature_importances.append(rf.feature_importances_)
                
                # Cross-validation score
                try:
                    scores = cross_val_score(rf, X_fold, y_fold, cv=3, scoring='neg_mean_squared_error')
                    cv_scores.append(np.mean(scores))
                except:
                    cv_scores.append(0.0)
            
            if feature_importances:
                # Average feature importance across folds
                ensemble_weights = np.mean(feature_importances, axis=0)
                ensemble_weights = ensemble_weights / (np.sum(ensemble_weights) + 1e-8)
                
                # Weight by CV performance
                if cv_scores:
                    performance_weight = np.mean(cv_scores)
                    ensemble_weights = ensemble_weights * (1 + performance_weight)
                    ensemble_weights = ensemble_weights / (np.sum(ensemble_weights) + 1e-8)
                
                logger.info("✅ Ensemble attention computed")
                return ensemble_weights
            else:
                return np.ones(X.shape[1]) / X.shape[1]
                
        except Exception as e:
            logger.warning(f"⚠️ Ensemble attention computation failed: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _apply_attention_weights(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply all attention weights to features."""
        if self.feature_attention_weights is None:
            return X
        
        # Start with base features
        X_weighted = X.copy()
        
        # Apply feature attention
        X_weighted = X_weighted * self.feature_attention_weights[None, :]
        
        # Apply temporal attention if enabled
        if self.config.use_temporal_attention and self.temporal_attention_weights is not None:
            X_weighted = X_weighted * self.temporal_attention_weights[None, :]
        
        # Apply regime attention if available
        if self.config.regime_aware and regimes is not None and self.regime_attention_weights:
            for i, regime in enumerate(regimes):
                regime_key = str(regime)
                if regime_key in self.regime_attention_weights:
                    X_weighted[i] = X_weighted[i] * self.regime_attention_weights[regime_key]
        
        # Apply ensemble attention
        if self.config.ensemble_attention and self.ensemble_attention_weights is not None:
            X_weighted = X_weighted * self.ensemble_attention_weights[None, :]
        
        # Apply attention dropout
        if self.config.attention_dropout > 0:
            attention_mask = np.random.binomial(
                1, 1 - self.config.attention_dropout,
                size=X.shape[1]
            )
            X_weighted = X_weighted * attention_mask[None, :]
        
        return X_weighted
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            regimes: Optional[np.ndarray] = None) -> 'TreeCLVSAWrapper':
        """Fit the Tree CLVSA wrapper."""
        start_time = time.time()
        
        logger.info(f"🌲 Training Tree CLVSA wrapper with {X.shape[0]} samples, {X.shape[1]} features")
        
        try:
            # Compute feature attention weights
            self.feature_attention_weights = self._compute_tree_specific_attention(X, y)
            
            # Compute temporal attention weights
            if self.config.use_temporal_attention:
                self.temporal_attention_weights = self._compute_temporal_attention(X)
            
            # Compute regime attention weights
            if self.config.regime_aware and regimes is not None:
                self.regime_attention_weights = self._compute_regime_attention(X, y, regimes)
            
            # Compute ensemble attention weights
            if self.config.ensemble_attention:
                self.ensemble_attention_weights = self._compute_ensemble_attention(X, y)
            
            # Apply attention to training data
            X_attentioned = self._apply_attention_weights(X, regimes)
            
            # Scale features
            X_scaled = self.input_scaler.fit_transform(X_attentioned)
            
            # Fit base model
            if sample_weight is not None:
                self.base_model.fit(X_scaled, y, sample_weight=sample_weight)
            else:
                self.base_model.fit(X_scaled, y)
            
            # Store performance metrics
            self.training_time = time.time() - start_time
            self.attention_performance = {
                'feature_attention_entropy': -np.sum(self.feature_attention_weights * np.log(self.feature_attention_weights + 1e-8)),
                'temporal_attention_entropy': -np.sum(self.temporal_attention_weights * np.log(self.temporal_attention_weights + 1e-8)) if self.temporal_attention_weights is not None else 0,
                'regime_attention_count': len(self.regime_attention_weights) if self.regime_attention_weights else 0,
                'training_time': self.training_time
            }
            
            logger.info(f"✅ Tree CLVSA wrapper fitted in {self.training_time:.3f}s")
            logger.info(f"   Feature attention entropy: {self.attention_performance['feature_attention_entropy']:.3f}")
            logger.info(f"   Regime attention regimes: {self.attention_performance['regime_attention_count']}")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Tree CLVSA training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions using Tree CLVSA wrapper."""
        try:
            # Apply attention weights
            X_attentioned = self._apply_attention_weights(X, regimes)
            
            # Scale features
            X_scaled = self.input_scaler.transform(X_attentioned)
            
            # Make predictions
            return self.base_model.predict(X_scaled)
            
        except Exception as e:
            logger.error(f"❌ Tree CLVSA prediction failed: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Make probability predictions if base model supports it."""
        try:
            # Apply attention weights
            X_attentioned = self._apply_attention_weights(X, regimes)
            
            # Scale features
            X_scaled = self.input_scaler.transform(X_attentioned)
            
            # Make probability predictions
            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(X_scaled)
            else:
                # Fallback to regular predictions
                predictions = self.base_model.predict(X_scaled)
                # Convert to probabilities (simple approach)
                if len(predictions.shape) == 1:
                    # Binary classification
                    proba = np.column_stack([1 - predictions, predictions])
                    return proba
                else:
                    return predictions
                    
        except Exception as e:
            logger.error(f"❌ Tree CLVSA probability prediction failed: {e}")
            raise
    
    def get_attention_weights(self) -> Dict[str, Any]:
        """Get all attention weights for analysis."""
        return {
            'feature_attention': self.feature_attention_weights,
            'temporal_attention': self.temporal_attention_weights,
            'regime_attention': self.regime_attention_weights,
            'ensemble_attention': self.ensemble_attention_weights,
            'attention_performance': self.attention_performance
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the base model."""
        if hasattr(self.base_model, 'feature_importances_'):
            return self.base_model.feature_importances_
        elif hasattr(self.base_model, 'coef_'):
            return np.abs(self.base_model.coef_)
        else:
            # Fallback to attention weights
            return self.feature_attention_weights if self.feature_attention_weights is not None else np.ones(1)
    
    def score(self, X: np.ndarray, y: np.ndarray, regimes: Optional[np.ndarray] = None) -> float:
        """Get model score."""
        try:
            predictions = self.predict(X, regimes)
            if hasattr(self.base_model, 'score'):
                return self.base_model.score(X, y)
            else:
                # Fallback to R² score
                from sklearn.metrics import r2_score
                return r2_score(y, predictions)
        except Exception as e:
            logger.error(f"❌ Tree CLVSA scoring failed: {e}")
            return 0.0


# Factory functions for creating Tree CLVSA wrappers
def create_tree_clvsa_wrapper(base_model, config: Optional[TreeCLVSAConfig] = None) -> TreeCLVSAWrapper:
    """Create Tree CLVSA wrapper for tree-based models."""
    return TreeCLVSAWrapper(base_model, config)


def create_tree_clvsa_config(**kwargs) -> TreeCLVSAConfig:
    """Create Tree CLVSA configuration with custom parameters."""
    return TreeCLVSAConfig(**kwargs)


# Integration helpers
def wrap_tree_model_with_clvsa(base_model, **config_kwargs) -> TreeCLVSAWrapper:
    """Convenience function to wrap any tree model with CLVSA."""
    config = create_tree_clvsa_config(**config_kwargs)
    return create_tree_clvsa_wrapper(base_model, config)


# Model-specific factory functions
def create_clvsa_random_forest(**model_params) -> TreeCLVSAWrapper:
    """Create CLVSA-wrapped Random Forest."""
    from sklearn.ensemble import RandomForestRegressor
    
    base_model = RandomForestRegressor(**model_params)
    return wrap_tree_model_with_clvsa(base_model)


def create_clvsa_xgboost(**model_params) -> TreeCLVSAWrapper:
    """Create CLVSA-wrapped XGBoost."""
    import xgboost as xgb
    
    base_model = xgb.XGBRegressor(**model_params)
    return wrap_tree_model_with_clvsa(base_model)


def create_clvsa_lightgbm(**model_params) -> TreeCLVSAWrapper:
    """Create CLVSA-wrapped LightGBM."""
    import lightgbm as lgb
    
    base_model = lgb.LGBMRegressor(**model_params)
    return wrap_tree_model_with_clvsa(base_model)


def create_clvsa_catboost(**model_params) -> TreeCLVSAWrapper:
    """Create CLVSA-wrapped CatBoost."""
    from catboost import CatBoostRegressor
    
    base_model = CatBoostRegressor(**model_params)
    return wrap_tree_model_with_clvsa(base_model)