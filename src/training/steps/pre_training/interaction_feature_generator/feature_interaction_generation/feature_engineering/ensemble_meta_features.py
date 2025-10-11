"""
Ensemble Meta-Features Generator

This module provides meta-feature generation for ensemble models, including
disagreement features, that can be called from training steps and trading modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Import disagreement meta-features
from .disagreement_meta_features import DisagreementMetaFeatures

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class EnsembleMetaFeatureGenerator:
    """
    Meta-feature generator for ensemble models that can be called from
    training steps and trading modules.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ensemble meta-feature generator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.disagreement_calculator = DisagreementMetaFeatures(logger)
    
    def generate_meta_features_for_analyst_ensemble(
        self, 
        features_df: pd.DataFrame, 
        ensemble_predictions: Optional[Dict[str, Any]] = None,
        is_live: bool = False
    ) -> pd.DataFrame:
        """
        Generate meta-features for analyst ensemble models.
        
        Args:
            features_df: Input DataFrame with features
            ensemble_predictions: Optional ensemble predictions for disagreement analysis
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            DataFrame containing meta-features including disagreement features
        """
        try:
            # Initialize meta-features DataFrame
            meta_features = pd.DataFrame(index=features_df.index)
            
            # Add basic analyst-specific meta-features
            if 'close' in features_df.columns:
                meta_features['price_momentum'] = features_df['close'].pct_change(10)
                meta_features['price_acceleration'] = features_df['close'].pct_change(10).diff()
                meta_features['volatility_proxy'] = features_df['close'].pct_change().rolling(20).std()
                meta_features['price_trend'] = features_df['close'].rolling(50).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
                )
            
            if 'volume' in features_df.columns:
                meta_features['volume_momentum'] = features_df['volume'].pct_change(10)
                meta_features['volume_acceleration'] = features_df['volume'].pct_change(10).diff()
                meta_features['volume_trend'] = features_df['volume'].rolling(50).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
                )
            
            # Add regime-specific features if available
            if 'composite_cluster_id' in features_df.columns:
                meta_features['regime_stability'] = features_df['composite_cluster_id'].rolling(20).std()
                meta_features['regime_persistence'] = (
                    features_df['composite_cluster_id'] == features_df['composite_cluster_id'].shift(1)
                ).rolling(20).mean()
                meta_features['regime_transition'] = (
                    features_df['composite_cluster_id'] != features_df['composite_cluster_id'].shift(1)
                ).rolling(10).sum()
            
            # Add HMM integration features if available
            hmm_features = ['hmm_state', 'hmm_transition_prob', 'hmm_confidence']
            for feature in hmm_features:
                if feature in features_df.columns:
                    meta_features[f'{feature}_momentum'] = features_df[feature].pct_change(10)
                    meta_features[f'{feature}_stability'] = features_df[feature].rolling(20).std()
            
            # Add disagreement features if ensemble predictions are available
            if ensemble_predictions and len(ensemble_predictions) > 1:
                disagreement_features = self.disagreement_calculator.calculate_disagreement_features_for_ensemble(
                    ensemble_predictions, is_live=is_live
                )
                
                # Add disagreement features to meta-features
                for feature_name, feature_value in disagreement_features.items():
                    meta_features[feature_name] = feature_value
                
                self.logger.info(f"Added {len(disagreement_features)} disagreement features to analyst ensemble")
            else:
                # Add default disagreement features
                default_disagreement = self.disagreement_calculator._get_default_disagreement_features()
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = feature_value
            
            # Ensure all features are numeric and handle any NaN values
            meta_features = meta_features.fillna(0.0)
            
            # Convert to numeric, coercing any non-numeric values
            for col in meta_features.columns:
                meta_features[col] = pd.to_numeric(meta_features[col], errors='coerce').fillna(0.0)
            
            return meta_features
            
        except Exception as e:
            self.logger.error(f"Error generating meta-features for analyst ensemble: {e}")
            # Return basic meta-features as fallback
            try:
                meta_features = pd.DataFrame(index=features_df.index)
                if 'close' in features_df.columns:
                    meta_features['price_momentum'] = features_df['close'].pct_change(10).fillna(0)
                return meta_features.fillna(0.0)
            except Exception as fallback_error:
                self.logger.error(f"Fallback meta-feature generation also failed: {fallback_error}")
                return pd.DataFrame(index=features_df.index)
    
    def generate_meta_features_for_tactician_ensemble(
        self, 
        features_df: pd.DataFrame, 
        ensemble_predictions: Optional[Dict[str, Any]] = None,
        is_live: bool = False
    ) -> pd.DataFrame:
        """
        Generate meta-features for tactician ensemble models.
        
        Args:
            features_df: Input DataFrame with features
            ensemble_predictions: Optional ensemble predictions for disagreement analysis
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            DataFrame containing meta-features including disagreement features
        """
        try:
            # Initialize meta-features DataFrame
            meta_features = pd.DataFrame(index=features_df.index)
            
            # Add basic tactician-specific meta-features
            if 'close' in features_df.columns:
                meta_features['price_momentum'] = features_df['close'].pct_change(5)
                meta_features['price_acceleration'] = features_df['close'].pct_change(5).diff()
                meta_features['volatility_proxy'] = features_df['close'].pct_change().rolling(20).std()
            
            if 'volume' in features_df.columns:
                meta_features['volume_momentum'] = features_df['volume'].pct_change(5)
                meta_features['volume_acceleration'] = features_df['volume'].pct_change(5).diff()
            
            # Add regime-specific features if available
            if 'composite_cluster_id' in features_df.columns:
                meta_features['regime_stability'] = features_df['composite_cluster_id'].rolling(10).std()
                meta_features['regime_persistence'] = (
                    features_df['composite_cluster_id'] == features_df['composite_cluster_id'].shift(1)
                ).rolling(10).mean()
            
            # Add analyst integration features if available
            analyst_features = ['analyst_confidence', 'analyst_prediction', 'analyst_ensemble_confidence']
            for feature in analyst_features:
                if feature in features_df.columns:
                    meta_features[f'{feature}_momentum'] = features_df[feature].pct_change(5)
                    meta_features[f'{feature}_stability'] = features_df[feature].rolling(10).std()
            
            # Add disagreement features if ensemble predictions are available
            if ensemble_predictions and len(ensemble_predictions) > 1:
                disagreement_features = self.disagreement_calculator.calculate_disagreement_features_for_ensemble(
                    ensemble_predictions, is_live=is_live
                )
                
                # Add disagreement features to meta-features
                for feature_name, feature_value in disagreement_features.items():
                    meta_features[feature_name] = feature_value
                
                self.logger.info(f"Added {len(disagreement_features)} disagreement features to tactician ensemble")
            else:
                # Add default disagreement features
                default_disagreement = self.disagreement_calculator._get_default_disagreement_features()
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = feature_value
            
            # Ensure all features are numeric and handle any NaN values
            meta_features = meta_features.fillna(0.0)
            
            # Convert to numeric, coercing any non-numeric values
            for col in meta_features.columns:
                meta_features[col] = pd.to_numeric(meta_features[col], errors='coerce').fillna(0.0)
            
            return meta_features
            
        except Exception as e:
            self.logger.error(f"Error generating meta-features for tactician ensemble: {e}")
            # Return basic meta-features as fallback
            try:
                meta_features = pd.DataFrame(index=features_df.index)
                if 'close' in features_df.columns:
                    meta_features['price_momentum'] = features_df['close'].pct_change(5).fillna(0)
                return meta_features.fillna(0.0)
            except Exception as fallback_error:
                self.logger.error(f"Fallback meta-feature generation also failed: {fallback_error}")
                return pd.DataFrame(index=features_df.index)
    
    def generate_meta_features_for_volatile_regime_ensemble(
        self, 
        features_df: pd.DataFrame, 
        ensemble_predictions: Optional[Dict[str, Any]] = None,
        is_live: bool = False
    ) -> pd.DataFrame:
        """
        Generate meta-features for volatile regime ensemble models.
        
        Args:
            features_df: Input DataFrame with features
            ensemble_predictions: Optional ensemble predictions for disagreement analysis
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            DataFrame containing meta-features including disagreement features
        """
        try:
            # Initialize meta-features DataFrame
            meta_features = pd.DataFrame(index=features_df.index)
            
            # Add basic volatile regime-specific meta-features
            if 'volatility_20' in features_df.columns:
                meta_features['volatility_percentile'] = features_df['volatility_20'].rolling(100).rank(pct=True)
                meta_features['volatility_acceleration'] = features_df['volatility_20'].diff()
                meta_features['volatility_momentum'] = (
                    features_df['volatility_20'] - features_df['volatility_20'].shift(5)
                )
            
            if 'volume' in features_df.columns:
                meta_features['volume_volatility'] = features_df['volume'].rolling(20).std()
                meta_features['volume_volatility_ratio'] = (
                    meta_features['volume_volatility'] / features_df['volume'].rolling(20).mean()
                )
            
            if 'close' in features_df.columns:
                meta_features['price_volatility'] = features_df['close'].pct_change().rolling(20).std()
                meta_features['price_volatility_percentile'] = (
                    meta_features['price_volatility'].rolling(100).rank(pct=True)
                )
            
            if 'volatility_regime' in features_df.columns:
                meta_features['volatility_regime_numeric'] = features_df['volatility_regime']
            
            # Add disagreement features if ensemble predictions are available
            if ensemble_predictions and len(ensemble_predictions) > 1:
                disagreement_features = self.disagreement_calculator.calculate_disagreement_features_for_ensemble(
                    ensemble_predictions, is_live=is_live
                )
                
                # Add disagreement features to meta-features
                for feature_name, feature_value in disagreement_features.items():
                    meta_features[feature_name] = feature_value
                
                self.logger.info(f"Added {len(disagreement_features)} disagreement features to volatile regime ensemble")
            else:
                # Add default disagreement features
                default_disagreement = self.disagreement_calculator._get_default_disagreement_features()
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = feature_value
            
            # Ensure all features are numeric and handle any NaN values
            meta_features = meta_features.fillna(0.0)
            
            # Convert to numeric, coercing any non-numeric values
            for col in meta_features.columns:
                meta_features[col] = pd.to_numeric(meta_features[col], errors='coerce').fillna(0.0)
            
            return meta_features
            
        except Exception as e:
            self.logger.error(f"Error generating meta-features for volatile regime ensemble: {e}")
            # Return basic meta-features as fallback
            try:
                meta_features = pd.DataFrame(index=features_df.index)
                if 'volatility_20' in features_df.columns:
                    meta_features['volatility_percentile'] = features_df['volatility_20'].rolling(100).rank(pct=True).fillna(0)
                return meta_features.fillna(0.0)
            except Exception as fallback_error:
                self.logger.error(f"Fallback meta-feature generation also failed: {fallback_error}")
                return pd.DataFrame(index=features_df.index)
    
    def get_base_model_predictions(
        self, 
        models: Dict[str, Any], 
        features_df: pd.DataFrame, 
        is_live: bool = False
    ) -> Dict[str, Any]:
        """
        Get predictions from base models for disagreement analysis.
        
        Args:
            models: Dict of trained models
            features_df: Input DataFrame with features
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            Dict containing model predictions and probabilities
        """
        try:
            base_predictions = {}
            
            for model_name, model in models.items():
                if model is None:
                    continue
                    
                try:
                    # Get prediction from model
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_df.values)
                        prediction = np.argmax(proba, axis=1)[0] if len(proba) > 0 else 0.5
                        confidence = np.max(proba, axis=1)[0] if len(proba) > 0 else 0.5
                    elif hasattr(model, 'predict'):
                        prediction = model.predict(features_df.values)[0] if hasattr(model, 'predict') else 0.5
                        confidence = 0.7  # Default confidence
                    else:
                        prediction = 0.5
                        confidence = 0.0
                    
                    base_predictions[model_name] = {
                        'prediction': float(prediction),
                        'probability': float(prediction),
                        'confidence': float(confidence)
                    }
                    
                except Exception as model_error:
                    self.logger.warning(f"Error getting prediction from {model_name}: {model_error}")
                    base_predictions[model_name] = {
                        'prediction': 0.5,
                        'probability': 0.5,
                        'confidence': 0.0
                    }
            
            return base_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting base model predictions: {e}")
            return {}

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
