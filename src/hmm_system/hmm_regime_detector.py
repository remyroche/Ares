"""
HMM Regime Detection System

This module implements the HMM-based regime detection system that runs every 15 minutes
on 1-hour base timeframe data, providing probabilities for 15-25 market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.core.decorators import handles_errors


@dataclass
class HMMConfig:
    """Configuration for HMM regime detection system."""
    base_timeframe: str = "1h"
    run_interval_minutes: int = 15
    n_regimes: int = 20  # Will be optimized between 15-25
    n_features: int = 100
    lookback_periods: int = 24  # 24 hours of 1h data
    min_regimes: int = 15
    max_regimes: int = 25
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    random_state: int = 42


@dataclass
class RegimeProbabilities:
    """Container for regime probabilities and metadata."""
    timestamp: datetime
    regime_probs: np.ndarray
    dominant_regime: int
    confidence: float
    regime_characteristics: Dict[str, Any]
    feature_importance: Dict[str, float]


class HMMRegimeDetector:
    """
    HMM-based regime detection system for market analysis.
    
    This system:
    - Runs every 15 minutes on 1-hour base timeframe data
    - Uses 100 features to detect 15-25 market regimes
    - Provides probabilities for each regime based on momentum, volatility, volume
    - Optimizes the number of regimes automatically
    """
    
    def __init__(self, config: HMMConfig):
        """Initialize the HMM regime detector."""
        self.config = config
        self.logger = system_logger.getChild('HMMRegimeDetector')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.models: Dict[int, GaussianMixture] = {}
        self.feature_names: List[str] = []
        self.regime_characteristics: Dict[int, Dict[str, Any]] = {}
        self.is_trained = False
        self.last_run_time: Optional[datetime] = None
        
    @handles_errors
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 100 features for regime detection.
        
        Features include:
        - Momentum indicators (RSI, MACD, etc.)
        - Volatility indicators (ATR, Bollinger Bands, etc.)
        - Volume indicators (Volume ratios, OBV, etc.)
        - Price action features (OHLC patterns, etc.)
        - Cross-timeframe features
        """
        tprint("Extracting features for HMM regime detection...")
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['open']
        features['price_range'] = data['high'] - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = data['close'].pct_change(period)
            
        # MACD features
        macd_line, macd_signal, macd_hist = self._calculate_macd(data['close'])
        features['macd'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        
        # Volatility features
        for period in [10, 20, 30]:
            features[f'atr_{period}'] = self._calculate_atr(data, period)
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = self._calculate_bollinger_upper(data['close'], period)
            features[f'bb_lower_{period}'] = self._calculate_bollinger_lower(data['close'], period)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / data['close']
            
        # Volume features
        if 'volume' in data.columns:
            features['volume_ma_5'] = data['volume'].rolling(5).mean()
            features['volume_ma_20'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma_20']
            features['obv'] = self._calculate_obv(data['close'], data['volume'])
            features['volume_price_trend'] = self._calculate_vpt(data['close'], data['volume'])
            
        # Price patterns
        features['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) < 0.1
        features['hammer'] = self._detect_hammer(data)
        features['shooting_star'] = self._detect_shooting_star(data)
        
        # Cross-timeframe features (simplified for 1h base)
        for period in [2, 4, 8, 12]:  # 2h, 4h, 8h, 12h aggregations
            if len(data) > period:
                features[f'ctf_returns_{period}h'] = data['close'].pct_change(period)
                features[f'ctf_volatility_{period}h'] = data['close'].rolling(period).std()
                if 'volume' in data.columns:
                    features[f'ctf_volume_{period}h'] = data['volume'].rolling(period).mean()
                    
        # Additional technical indicators
        features['stoch_k'] = self._calculate_stochastic_k(data)
        features['stoch_d'] = self._calculate_stochastic_d(data)
        features['williams_r'] = self._calculate_williams_r(data)
        features['cci'] = self._calculate_cci(data)
        
        # Market microstructure features
        features['bid_ask_spread'] = (data['high'] - data['low']) / data['close']  # Proxy
        features['price_impact'] = abs(data['close'] - data['open']) / data['close']
        
        # Regime persistence features
        for period in [3, 6, 12]:
            features[f'regime_persistence_{period}'] = self._calculate_regime_persistence(data, period)
            
        # Drop NaN values and select top 100 features
        features = features.dropna()
        
        # Select most important features if we have more than 100
        if len(features.columns) > self.config.n_features:
            # Use variance-based selection
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(self.config.n_features).index.tolist()
            features = features[selected_features]
            
        self.feature_names = features.columns.tolist()
        tprint(f"Extracted {len(self.feature_names)} features for HMM analysis")
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_bollinger_upper(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Bands upper."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std * std_dev)
    
    def _calculate_bollinger_lower(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Bands lower."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma - (std * std_dev)
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = prices.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0)).cumsum()
        return pd.Series(obv, index=prices.index)
    
    def _calculate_vpt(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = prices.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern."""
        body = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        return (lower_shadow > 2 * body) & (upper_shadow < body)
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect shooting star candlestick pattern."""
        body = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        return (upper_shadow > 2 * body) & (lower_shadow < body)
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        return 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    
    def _calculate_stochastic_d(self, data: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.Series:
        """Calculate Stochastic %D."""
        k = self._calculate_stochastic_k(data, period)
        return k.rolling(window=smooth).mean()
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        return -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_regime_persistence(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate regime persistence indicator."""
        returns = data['close'].pct_change()
        return returns.rolling(window=period).apply(lambda x: len(x[x > 0]) / len(x))
    
    @handles_errors
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train HMM models with optimal number of regimes.
        
        Tests different numbers of regimes (15-25) and selects the best one
        based on BIC/AIC criteria.
        """
        tprint("Training HMM regime detection models...")
        
        # Extract features
        features = self.extract_features(data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Test different numbers of regimes
        best_model = None
        best_score = float('inf')
        best_n_regimes = self.config.min_regimes
        
        regime_scores = {}
        
        for n_regimes in range(self.config.min_regimes, self.config.max_regimes + 1):
            try:
                # Train Gaussian Mixture Model (HMM approximation)
                model = GaussianMixture(
                    n_components=n_regimes,
                    covariance_type='full',
                    max_iter=self.config.max_iterations,
                    tol=self.config.convergence_threshold,
                    random_state=self.config.random_state
                )
                
                model.fit(features_pca)
                
                # Calculate BIC (lower is better)
                bic = model.bic(features_pca)
                aic = model.aic(features_pca)
                
                regime_scores[n_regimes] = {
                    'bic': bic,
                    'aic': aic,
                    'converged': model.converged_
                }
                
                # Use BIC for model selection
                if bic < best_score:
                    best_score = bic
                    best_model = model
                    best_n_regimes = n_regimes
                    
                tprint(f"Tested {n_regimes} regimes: BIC={bic:.2f}, AIC={aic:.2f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train model with {n_regimes} regimes: {e}")
                continue
        
        if best_model is None:
            raise ValueError("Failed to train any HMM models")
        
        # Update config with optimal number of regimes
        self.config.n_regimes = best_n_regimes
        self.models[best_n_regimes] = best_model
        
        # Calculate regime characteristics
        self._calculate_regime_characteristics(features, features_pca, best_model)
        
        self.is_trained = True
        
        training_results = {
            'n_regimes': best_n_regimes,
            'bic_score': best_score,
            'regime_scores': regime_scores,
            'n_features': len(self.feature_names),
            'explained_variance_ratio': self.pca.explained_variance_ratio_.sum()
        }
        
        tprint(f"HMM training completed: {best_n_regimes} regimes, BIC={best_score:.2f}")
        
        return training_results
    
    def _calculate_regime_characteristics(self, features: pd.DataFrame, 
                                        features_pca: np.ndarray, 
                                        model: GaussianMixture) -> None:
        """Calculate characteristics for each regime."""
        regime_probs = model.predict_proba(features_pca)
        regime_labels = model.predict(features_pca)
        
        for regime in range(model.n_components):
            regime_mask = regime_labels == regime
            if not np.any(regime_mask):
                continue
                
            regime_data = features[regime_mask]
            
            self.regime_characteristics[regime] = {
                'mean_returns': regime_data['returns'].mean() if 'returns' in regime_data.columns else 0,
                'volatility': regime_data['returns'].std() if 'returns' in regime_data.columns else 0,
                'mean_volume': regime_data['volume_ratio'].mean() if 'volume_ratio' in regime_data.columns else 0,
                'frequency': np.sum(regime_mask) / len(regime_mask),
                'mean_momentum': regime_data['rsi_14'].mean() if 'rsi_14' in regime_data.columns else 0,
                'mean_volatility_indicator': regime_data['atr_14'].mean() if 'atr_14' in regime_data.columns else 0
            }
    
    @handles_errors
    def predict_regime_probabilities(self, data: pd.DataFrame) -> RegimeProbabilities:
        """
        Predict regime probabilities for the latest data.
        
        Returns probabilities for each of the 15-25 regimes.
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Extract features for the latest data point
        features = self.extract_features(data.tail(self.config.lookback_periods))
        
        if len(features) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Use the latest data point
        latest_features = features.iloc[-1:].values
        
        # Scale and transform features
        features_scaled = self.scaler.transform(latest_features)
        features_pca = self.pca.transform(features_scaled)
        
        # Get regime probabilities
        model = self.models[self.config.n_regimes]
        regime_probs = model.predict_proba(features_pca)[0]
        
        # Find dominant regime
        dominant_regime = np.argmax(regime_probs)
        confidence = regime_probs[dominant_regime]
        
        # Calculate feature importance (simplified)
        feature_importance = dict(zip(self.feature_names, 
                                    np.abs(features_scaled[0])))
        
        # Get regime characteristics
        regime_chars = self.regime_characteristics.get(dominant_regime, {})
        
        return RegimeProbabilities(
            timestamp=datetime.now(),
            regime_probs=regime_probs,
            dominant_regime=dominant_regime,
            confidence=confidence,
            regime_characteristics=regime_chars,
            feature_importance=feature_importance
        )
    
    @handles_errors
    def should_run(self) -> bool:
        """Check if it's time to run the HMM analysis."""
        if self.last_run_time is None:
            return True
        
        time_since_last_run = datetime.now() - self.last_run_time
        return time_since_last_run >= timedelta(minutes=self.config.run_interval_minutes)
    
    @handles_errors
    def run_analysis(self, data: pd.DataFrame) -> RegimeProbabilities:
        """
        Run the complete HMM analysis if it's time.
        
        This method should be called every 15 minutes.
        """
        if not self.should_run():
            return None
        
        if not self.is_trained:
            self.train_models(data)
        
        result = self.predict_regime_probabilities(data)
        self.last_run_time = datetime.now()
        
        tprint(f"HMM analysis completed: Regime {result.dominant_regime} "
              f"(confidence: {result.confidence:.3f})")
        
        return result
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'pca': self.pca,
            'models': self.models,
            'feature_names': self.feature_names,
            'regime_characteristics': self.regime_characteristics,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        tprint(f"HMM models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.regime_characteristics = model_data['regime_characteristics']
        self.is_trained = model_data['is_trained']
        
        tprint(f"HMM models loaded from {filepath}")


# Factory function for easy instantiation
def create_hmm_regime_detector(config: Optional[HMMConfig] = None) -> HMMRegimeDetector:
    """Create and return a new HMM regime detector instance."""
    if config is None:
        config = HMMConfig()
    return HMMRegimeDetector(config)