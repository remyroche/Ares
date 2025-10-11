"""
Ensemble Meta-Features Generator

This module provides meta-feature generation for ensemble models, including
disagreement features, that can be called from training steps and trading modules.

VectorBT Optimizations:
- Vectorized meta-feature calculations
- GPU acceleration for large datasets
- Parallel processing for multiple ensemble models
- Memory-efficient operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import warnings

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.utils.array_ops import rolling_apply
    from vectorbt.utils.array_ops import rolling_apply_parallel
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_apply = None
    rolling_apply_parallel = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import disagreement meta-features
from .disagreement_meta_features import DisagreementMetaFeatures

class EnsembleMetaFeatureGenerator:
    """
    Meta-feature generator for ensemble models that can be called from
    training steps and trading modules with VectorBT optimization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, use_vectorbt: bool = True, use_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize the ensemble meta-feature generator.
        
        Args:
            logger: Optional logger instance
            use_vectorbt: Whether to use VectorBT optimizations
            use_gpu: Whether to use GPU acceleration
            enable_parallel: Whether to enable parallel processing
        """
        self.logger = logger or logging.getLogger(__name__)
        self.disagreement_calculator = DisagreementMetaFeatures(logger)
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
    
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
            
            # Add basic analyst-specific meta-features with VectorBT optimization
            if 'close' in features_df.columns:
                if self.use_vectorbt and len(features_df) > 1000:
                    meta_features.update(self._generate_price_meta_features_vectorized(features_df, 'analyst'))
                else:
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
                    meta_features[feature_name] = np.asarray(feature_value, dtype=float)

                self.logger.info(f"Added {len(disagreement_features)} disagreement features to analyst ensemble")
            else:
                # Add default disagreement features
                default_disagreement = self.disagreement_calculator._get_default_disagreement_features(meta_features.index)
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = np.asarray(feature_value, dtype=float)
            
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
                    meta_features[feature_name] = np.asarray(feature_value, dtype=float)

                self.logger.info(f"Added {len(disagreement_features)} disagreement features to tactician ensemble")
            else:
                # Add default disagreement features
                default_disagreement = self.disagreement_calculator._get_default_disagreement_features(meta_features.index)
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = np.asarray(feature_value, dtype=float)
            
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
                    meta_features[feature_name] = np.asarray(feature_value, dtype=float)

                self.logger.info(f"Added {len(disagreement_features)} disagreement features to volatile regime ensemble")
            else:
                # Add default disagreement features
                default_disagreement = self.disagreement_calculator._get_default_disagreement_features(meta_features.index)
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = np.asarray(feature_value, dtype=float)
            
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
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_df.values)
                        proba = np.asarray(proba, dtype=float)
                        if proba.ndim == 1:
                            proba = proba.reshape(-1, 1)
                        prediction = proba[:, -1]
                        confidence = np.max(proba, axis=1)
                        probability_payload: Union[np.ndarray, List[float]] = proba
                    elif hasattr(model, 'predict'):
                        prediction = np.asarray(model.predict(features_df.values), dtype=float)
                        confidence = np.full_like(prediction, 0.7, dtype=float)
                        probability_payload = prediction
                    else:
                        prediction = np.full(len(features_df), 0.5)
                        confidence = np.zeros(len(features_df))
                        probability_payload = prediction

                    base_predictions[model_name] = {
                        'prediction': prediction,
                        'probability': probability_payload,
                        'confidence': confidence
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
    
    def _generate_price_meta_features_vectorized(self, features_df: pd.DataFrame, ensemble_type: str) -> Dict[str, pd.Series]:
        """Generate price-related meta-features using VectorBT optimization."""
        meta_features = {}
        
        if 'close' not in features_df.columns:
            return meta_features
        
        close_prices = features_df['close']
        
        if self.use_gpu and CUPY_AVAILABLE:
            return self._generate_price_meta_features_gpu(close_prices, ensemble_type)
        else:
            return self._generate_price_meta_features_cpu_vectorized(close_prices, ensemble_type)
    
    def _generate_price_meta_features_cpu_vectorized(self, close_prices: pd.Series, ensemble_type: str) -> Dict[str, pd.Series]:
        """CPU-optimized vectorized price meta-feature generation."""
        meta_features = {}
        
        # Vectorized calculations
        if ensemble_type == 'analyst':
            meta_features['price_momentum'] = close_prices.pct_change(10)
            meta_features['price_acceleration'] = close_prices.pct_change(10).diff()
            meta_features['volatility_proxy'] = close_prices.pct_change().rolling(20).std()
            meta_features['price_trend'] = close_prices.rolling(50).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
        elif ensemble_type == 'tactician':
            meta_features['price_momentum'] = close_prices.pct_change(5)
            meta_features['price_acceleration'] = close_prices.pct_change(5).diff()
            meta_features['volatility_proxy'] = close_prices.pct_change().rolling(20).std()
        elif ensemble_type == 'volatile_regime':
            meta_features['price_volatility'] = close_prices.pct_change().rolling(20).std()
            meta_features['price_volatility_percentile'] = (
                meta_features['price_volatility'].rolling(100).rank(pct=True)
            )
        
        return meta_features
    
    def _generate_price_meta_features_gpu(self, close_prices: pd.Series, ensemble_type: str) -> Dict[str, pd.Series]:
        """GPU-accelerated price meta-feature generation using CuPy."""
        if not CUPY_AVAILABLE:
            return self._generate_price_meta_features_cpu_vectorized(close_prices, ensemble_type)
        
        meta_features = {}
        
        # Convert to GPU arrays
        close_gpu = cp.asarray(close_prices.values, dtype=cp.float32)
        
        if ensemble_type == 'analyst':
            # GPU-accelerated calculations
            returns_gpu = cp.diff(close_gpu) / close_gpu[:-1]
            meta_features['price_momentum'] = pd.Series(
                cp.asnumpy(cp.diff(close_gpu, n=10) / close_gpu[:-10]), 
                index=close_prices.index[10:]
            )
            meta_features['price_acceleration'] = pd.Series(
                cp.asnumpy(cp.diff(cp.diff(close_gpu, n=10) / close_gpu[:-10])), 
                index=close_prices.index[11:]
            )
            # Volatility calculation on GPU
            volatility_gpu = cp.zeros_like(close_gpu)
            for i in range(20, len(close_gpu)):
                window_returns = returns_gpu[i-20:i]
                volatility_gpu[i] = cp.std(window_returns)
            meta_features['volatility_proxy'] = pd.Series(
                cp.asnumpy(volatility_gpu[20:]), 
                index=close_prices.index[20:]
            )
        elif ensemble_type == 'tactician':
            meta_features['price_momentum'] = pd.Series(
                cp.asnumpy(cp.diff(close_gpu, n=5) / close_gpu[:-5]), 
                index=close_prices.index[5:]
            )
            meta_features['price_acceleration'] = pd.Series(
                cp.asnumpy(cp.diff(cp.diff(close_gpu, n=5) / close_gpu[:-5])), 
                index=close_prices.index[6:]
            )
        elif ensemble_type == 'volatile_regime':
            returns_gpu = cp.diff(close_gpu) / close_gpu[:-1]
            volatility_gpu = cp.zeros_like(close_gpu)
            for i in range(20, len(close_gpu)):
                window_returns = returns_gpu[i-20:i]
                volatility_gpu[i] = cp.std(window_returns)
            meta_features['price_volatility'] = pd.Series(
                cp.asnumpy(volatility_gpu[20:]), 
                index=close_prices.index[20:]
            )
        
        return meta_features
    
    def _generate_volume_meta_features_vectorized(self, features_df: pd.DataFrame, ensemble_type: str) -> Dict[str, pd.Series]:
        """Generate volume-related meta-features using VectorBT optimization."""
        meta_features = {}
        
        if 'volume' not in features_df.columns:
            return meta_features
        
        volume = features_df['volume']
        
        if self.use_gpu and CUPY_AVAILABLE:
            return self._generate_volume_meta_features_gpu(volume, ensemble_type)
        else:
            return self._generate_volume_meta_features_cpu_vectorized(volume, ensemble_type)
    
    def _generate_volume_meta_features_cpu_vectorized(self, volume: pd.Series, ensemble_type: str) -> Dict[str, pd.Series]:
        """CPU-optimized vectorized volume meta-feature generation."""
        meta_features = {}
        
        if ensemble_type == 'analyst':
            meta_features['volume_momentum'] = volume.pct_change(10)
            meta_features['volume_acceleration'] = volume.pct_change(10).diff()
            meta_features['volume_trend'] = volume.rolling(50).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
        elif ensemble_type == 'tactician':
            meta_features['volume_momentum'] = volume.pct_change(5)
            meta_features['volume_acceleration'] = volume.pct_change(5).diff()
        elif ensemble_type == 'volatile_regime':
            meta_features['volume_volatility'] = volume.rolling(20).std()
            meta_features['volume_volatility_ratio'] = (
                meta_features['volume_volatility'] / volume.rolling(20).mean()
            )
        
        return meta_features
    
    def _generate_volume_meta_features_gpu(self, volume: pd.Series, ensemble_type: str) -> Dict[str, pd.Series]:
        """GPU-accelerated volume meta-feature generation using CuPy."""
        if not CUPY_AVAILABLE:
            return self._generate_volume_meta_features_cpu_vectorized(volume, ensemble_type)
        
        meta_features = {}
        volume_gpu = cp.asarray(volume.values, dtype=cp.float32)
        
        if ensemble_type == 'analyst':
            meta_features['volume_momentum'] = pd.Series(
                cp.asnumpy(cp.diff(volume_gpu, n=10) / volume_gpu[:-10]), 
                index=volume.index[10:]
            )
            meta_features['volume_acceleration'] = pd.Series(
                cp.asnumpy(cp.diff(cp.diff(volume_gpu, n=10) / volume_gpu[:-10])), 
                index=volume.index[11:]
            )
        elif ensemble_type == 'tactician':
            meta_features['volume_momentum'] = pd.Series(
                cp.asnumpy(cp.diff(volume_gpu, n=5) / volume_gpu[:-5]), 
                index=volume.index[5:]
            )
            meta_features['volume_acceleration'] = pd.Series(
                cp.asnumpy(cp.diff(cp.diff(volume_gpu, n=5) / volume_gpu[:-5])), 
                index=volume.index[6:]
            )
        elif ensemble_type == 'volatile_regime':
            # GPU-accelerated rolling calculations
            volume_volatility_gpu = cp.zeros_like(volume_gpu)
            volume_mean_gpu = cp.zeros_like(volume_gpu)
            
            for i in range(20, len(volume_gpu)):
                window = volume_gpu[i-20:i]
                volume_volatility_gpu[i] = cp.std(window)
                volume_mean_gpu[i] = cp.mean(window)
            
            meta_features['volume_volatility'] = pd.Series(
                cp.asnumpy(volume_volatility_gpu[20:]), 
                index=volume.index[20:]
            )
            meta_features['volume_volatility_ratio'] = pd.Series(
                cp.asnumpy(volume_volatility_gpu[20:] / volume_mean_gpu[20:]), 
                index=volume.index[20:]
            )
        
        return meta_features