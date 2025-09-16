"""
Analyst Regime Predictor System

This module implements the Analyst system that runs every 2 minutes on 5-minute base
timeframe data, using 300+ features and HMM outputs to decide IF we should trade.
Trained per-regime with comprehensive cross-timeframe features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import os

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.core.decorators import handles_errors


@dataclass
class AnalystConfig:
    """Configuration for Analyst regime prediction system."""
    base_timeframe: str = "5m"
    run_interval_minutes: int = 2
    n_features: int = 300  # 300+ features
    target_threshold: float = 0.5  # 0.5% price change threshold
    lookback_periods: int = 144  # 12 hours of 5m data
    cross_timeframe_periods: List[int] = None  # Will be set in __post_init__
    models: Dict[str, str] = None  # Will be set in __post_init__
    meta_learner: str = "elastic_net"
    train_test_split: float = 0.8
    random_state: int = 42
    
    def __post_init__(self):
        if self.cross_timeframe_periods is None:
            # Cross-timeframe periods: 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
            self.cross_timeframe_periods = [3, 6, 12, 24, 48, 72, 144, 288]
        
        if self.models is None:
            self.models = {
                "tcn": "Temporal Convolutions Network",
                "catboost": "CatBoostRegressor", 
                "lightgbm": "LGBMRegressor"
            }


@dataclass
class AnalystPrediction:
    """Container for Analyst predictions and metadata."""
    timestamp: datetime
    should_trade: bool
    confidence: float
    base_model_predictions: Dict[str, float]
    meta_learner_prediction: float
    regime_id: int
    feature_importance: Dict[str, float]
    market_conditions: Dict[str, Any]


class AnalystRegimePredictor:
    """
    Analyst system for deciding IF we should trade.
    
    This system:
    - Runs every 2 minutes on 5-minute base timeframe data
    - Uses 300+ features including cross-timeframe features
    - Integrates HMM regime outputs
    - Trained per-regime to decide trading opportunities
    - Emits green light for the Tactician when conditions are favorable
    """
    
    def __init__(self, config: AnalystConfig):
        """Initialize the Analyst regime predictor."""
        self.config = config
        self.logger = system_logger.getChild('AnalystRegimePredictor')
        self.scaler = StandardScaler()
        self.models: Dict[int, Dict[str, Any]] = {}  # Per-regime models
        self.meta_learners: Dict[int, Any] = {}  # Per-regime meta-learners
        self.feature_names: List[str] = []
        self.is_trained = False
        self.last_run_time: Optional[datetime] = None
        self.regime_models_trained: Dict[int, bool] = {}
        
    @handles_errors
    def extract_comprehensive_features(self, data: pd.DataFrame, 
                                     hmm_outputs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Extract 300+ comprehensive features including cross-timeframe analysis.
        
        Features include:
        - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - Volume analysis (OBV, VPT, Volume ratios, etc.)
        - Price action patterns (Candlestick patterns, Support/Resistance)
        - Cross-timeframe features (15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        - Market microstructure features
        - HMM regime probabilities and characteristics
        - Momentum and volatility across timeframes
        """
        tprint("Extracting comprehensive features for Analyst...")
        
        features = pd.DataFrame(index=data.index)
        
        # === BASIC PRICE FEATURES ===
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['open']
        features['price_range'] = data['high'] - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        features['body_ratio'] = features['body_size'] / features['price_range']
        features['shadow_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / features['price_range']
        
        # === MOMENTUM INDICATORS ===
        for period in [5, 10, 14, 20, 30, 50]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = data['close'].pct_change(period)
            features[f'roc_ma_{period}'] = features[f'roc_{period}'].rolling(5).mean()
            
        # MACD variations
        for fast, slow in [(12, 26), (5, 35), (19, 39)]:
            macd_line, macd_signal, macd_hist = self._calculate_macd(data['close'], fast, slow, 9)
            features[f'macd_{fast}_{slow}'] = macd_line
            features[f'macd_signal_{fast}_{slow}'] = macd_signal
            features[f'macd_hist_{fast}_{slow}'] = macd_hist
            
        # === VOLATILITY INDICATORS ===
        for period in [10, 14, 20, 30, 50]:
            features[f'atr_{period}'] = self._calculate_atr(data, period)
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = self._calculate_bollinger_upper(data['close'], period)
            features[f'bb_lower_{period}'] = self._calculate_bollinger_lower(data['close'], period)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / data['close']
            features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
            
        # === VOLUME ANALYSIS ===
        if 'volume' in data.columns:
            for period in [5, 10, 20, 30, 50]:
                features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
                features[f'volume_std_{period}'] = data['volume'].rolling(period).std()
                
            features['obv'] = self._calculate_obv(data['close'], data['volume'])
            features['vpt'] = self._calculate_vpt(data['close'], data['volume'])
            features['mfi'] = self._calculate_mfi(data, 14)
            features['eom'] = self._calculate_eom(data)
            features['vwap'] = self._calculate_vwap(data)
            
        # === OSCILLATORS ===
        features['stoch_k'] = self._calculate_stochastic_k(data)
        features['stoch_d'] = self._calculate_stochastic_d(data)
        features['williams_r'] = self._calculate_williams_r(data)
        features['cci'] = self._calculate_cci(data)
        features['roc'] = self._calculate_roc(data)
        features['ppo'] = self._calculate_ppo(data)
        
        # === PRICE PATTERNS ===
        features['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) < 0.1
        features['hammer'] = self._detect_hammer(data)
        features['shooting_star'] = self._detect_shooting_star(data)
        features['engulfing_bullish'] = self._detect_engulfing_bullish(data)
        features['engulfing_bearish'] = self._detect_engulfing_bearish(data)
        features['harami_bullish'] = self._detect_harami_bullish(data)
        features['harami_bearish'] = self._detect_harami_bearish(data)
        
        # === CROSS-TIMEFRAME FEATURES ===
        for period in self.config.cross_timeframe_periods:
            if len(data) > period:
                # Price features
                features[f'ctf_returns_{period}'] = data['close'].pct_change(period)
                features[f'ctf_volatility_{period}'] = data['close'].rolling(period).std()
                features[f'ctf_range_{period}'] = (data['high'].rolling(period).max() - data['low'].rolling(period).min()) / data['close']
                
                # Volume features
                if 'volume' in data.columns:
                    features[f'ctf_volume_{period}'] = data['volume'].rolling(period).mean()
                    features[f'ctf_volume_ratio_{period}'] = data['volume'] / features[f'ctf_volume_{period}']
                    
                # Momentum features
                features[f'ctf_rsi_{period}'] = self._calculate_rsi(data['close'], period)
                features[f'ctf_momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                
                # Volatility features
                features[f'ctf_atr_{period}'] = self._calculate_atr(data, period)
                features[f'ctf_bb_width_{period}'] = self._calculate_bb_width(data['close'], period)
                
        # === MARKET MICROSTRUCTURE ===
        features['bid_ask_spread'] = (data['high'] - data['low']) / data['close']
        features['price_impact'] = abs(data['close'] - data['open']) / data['close']
        features['intraday_volatility'] = (data['high'] - data['low']) / data['open']
        features['gap_up'] = data['open'] > data['close'].shift(1)
        features['gap_down'] = data['open'] < data['close'].shift(1)
        
        # === TREND ANALYSIS ===
        for period in [10, 20, 50, 100]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features[f'trend_{period}'] = (data['close'] - features[f'sma_{period}']) / features[f'sma_{period}']
            features[f'trend_strength_{period}'] = features[f'trend_{period}'].rolling(10).std()
            
        # === SUPPORT/RESISTANCE FEATURES ===
        features['sr_levels'] = self._calculate_sr_levels(data)
        features['sr_distance'] = self._calculate_sr_distance(data, features['sr_levels'])
        features['sr_strength'] = self._calculate_sr_strength(data, features['sr_levels'])
        
        # === HMM INTEGRATION ===
        if hmm_outputs is not None:
            if 'regime_probs' in hmm_outputs:
                regime_probs = hmm_outputs['regime_probs']
                for i, prob in enumerate(regime_probs):
                    features[f'hmm_regime_{i}_prob'] = prob
                    
            if 'dominant_regime' in hmm_outputs:
                features['hmm_dominant_regime'] = hmm_outputs['dominant_regime']
                
            if 'regime_characteristics' in hmm_outputs:
                regime_chars = hmm_outputs['regime_characteristics']
                for key, value in regime_chars.items():
                    features[f'hmm_{key}'] = value
                    
        # === ADDITIONAL TECHNICAL FEATURES ===
        features['adx'] = self._calculate_adx(data)
        features['di_plus'] = self._calculate_di_plus(data)
        features[f'di_minus'] = self._calculate_di_minus(data)
        features['aroon_up'] = self._calculate_aroon_up(data)
        features['aroon_down'] = self._calculate_aroon_down(data)
        features['aroon_oscillator'] = features['aroon_up'] - features['aroon_down']
        
        # === REGIME PERSISTENCE ===
        for period in [5, 10, 20]:
            features[f'regime_persistence_{period}'] = self._calculate_regime_persistence(data, period)
            features[f'trend_persistence_{period}'] = self._calculate_trend_persistence(data, period)
            
        # === MARKET STRENGTH INDICATORS ===
        features['market_strength'] = self._calculate_market_strength(data)
        features['volatility_regime'] = self._calculate_volatility_regime(data)
        features['momentum_regime'] = self._calculate_momentum_regime(data)
        
        # Drop NaN values
        features = features.dropna()
        
        # Ensure we have enough features
        if len(features.columns) < self.config.n_features:
            tprint(f"Warning: Only {len(features.columns)} features extracted, target is {self.config.n_features}")
        elif len(features.columns) > self.config.n_features:
            # Select most important features
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(self.config.n_features).index.tolist()
            features = features[selected_features]
            
        self.feature_names = features.columns.tolist()
        tprint(f"Extracted {len(self.feature_names)} comprehensive features for Analyst")
        
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
    
    def _calculate_bb_width(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Bands width."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return (upper - lower) / sma
    
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
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_eom(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Ease of Movement."""
        distance_moved = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
        box_height = data['volume'] / (data['high'] - data['low'])
        eom = distance_moved / box_height
        return eom.rolling(period).mean()
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
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
    
    def _calculate_roc(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Rate of Change."""
        return data['close'].pct_change(period) * 100
    
    def _calculate_ppo(self, data: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate Percentage Price Oscillator."""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        return ((ema_fast - ema_slow) / ema_slow) * 100
    
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
    
    def _detect_engulfing_bullish(self, data: pd.DataFrame) -> pd.Series:
        """Detect bullish engulfing pattern."""
        prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
        curr_body = abs(data['close'] - data['open'])
        return (data['close'].shift(1) < data['open'].shift(1)) & (data['close'] > data['open']) & (curr_body > prev_body)
    
    def _detect_engulfing_bearish(self, data: pd.DataFrame) -> pd.Series:
        """Detect bearish engulfing pattern."""
        prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
        curr_body = abs(data['close'] - data['open'])
        return (data['close'].shift(1) > data['open'].shift(1)) & (data['close'] < data['open']) & (curr_body > prev_body)
    
    def _detect_harami_bullish(self, data: pd.DataFrame) -> pd.Series:
        """Detect bullish harami pattern."""
        prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
        curr_body = abs(data['close'] - data['open'])
        return (data['close'].shift(1) < data['open'].shift(1)) & (data['close'] > data['open']) & (curr_body < prev_body)
    
    def _detect_harami_bearish(self, data: pd.DataFrame) -> pd.Series:
        """Detect bearish harami pattern."""
        prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
        curr_body = abs(data['close'] - data['open'])
        return (data['close'].shift(1) > data['open'].shift(1)) & (data['close'] < data['open']) & (curr_body < prev_body)
    
    def _calculate_sr_levels(self, data: pd.DataFrame) -> pd.Series:
        """Calculate support/resistance levels (simplified)."""
        # This is a simplified version - in practice, you'd use more sophisticated SR detection
        highs = data['high'].rolling(20).max()
        lows = data['low'].rolling(20).min()
        return (highs + lows) / 2
    
    def _calculate_sr_distance(self, data: pd.DataFrame, sr_levels: pd.Series) -> pd.Series:
        """Calculate distance to nearest support/resistance level."""
        return (data['close'] - sr_levels) / data['close']
    
    def _calculate_sr_strength(self, data: pd.DataFrame, sr_levels: pd.Series) -> pd.Series:
        """Calculate strength of support/resistance level."""
        # Simplified: count how many times price touched the level recently
        return data['close'].rolling(10).apply(lambda x: np.sum(np.abs(x - sr_levels.iloc[-len(x):]) < 0.01))
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        # Simplified ADX calculation
        high_diff = data['high'].diff()
        low_diff = data['low'].diff()
        close_diff = data['close'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / self._calculate_atr(data, period))
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / self._calculate_atr(data, period))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
    
    def _calculate_di_plus(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate +DI."""
        high_diff = data['high'].diff()
        low_diff = data['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        return 100 * (pd.Series(plus_dm).rolling(period).mean() / self._calculate_atr(data, period))
    
    def _calculate_di_minus(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate -DI."""
        high_diff = data['high'].diff()
        low_diff = data['low'].diff()
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        return 100 * (pd.Series(minus_dm).rolling(period).mean() / self._calculate_atr(data, period))
    
    def _calculate_aroon_up(self, data: pd.DataFrame, period: int = 25) -> pd.Series:
        """Calculate Aroon Up."""
        return data['high'].rolling(period).apply(lambda x: (period - x.argmax()) / period * 100)
    
    def _calculate_aroon_down(self, data: pd.DataFrame, period: int = 25) -> pd.Series:
        """Calculate Aroon Down."""
        return data['low'].rolling(period).apply(lambda x: (period - x.argmin()) / period * 100)
    
    def _calculate_regime_persistence(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate regime persistence indicator."""
        returns = data['close'].pct_change()
        return returns.rolling(window=period).apply(lambda x: len(x[x > 0]) / len(x))
    
    def _calculate_trend_persistence(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate trend persistence indicator."""
        sma = data['close'].rolling(period).mean()
        trend = data['close'] > sma
        return trend.rolling(period).apply(lambda x: len(x[x]) / len(x))
    
    def _calculate_market_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate overall market strength indicator."""
        rsi = self._calculate_rsi(data['close'], 14)
        macd, _, _ = self._calculate_macd(data['close'])
        return (rsi + macd) / 2  # Simplified combination
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime indicator."""
        volatility = data['close'].rolling(20).std()
        return pd.cut(volatility, bins=3, labels=[0, 1, 2]).astype(float)
    
    def _calculate_momentum_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum regime indicator."""
        momentum = data['close'].pct_change(20)
        return pd.cut(momentum, bins=3, labels=[0, 1, 2]).astype(float)
    
    @handles_errors
    def train_regime_models(self, data: pd.DataFrame, regime_labels: np.ndarray, 
                           hmm_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train models for each regime separately.
        
        Args:
            data: Market data
            regime_labels: Regime assignments for each data point
            hmm_outputs: HMM regime detection outputs
        """
        tprint("Training Analyst models per regime...")
        
        # Extract features
        features = self.extract_comprehensive_features(data, hmm_outputs)
        
        # Align features with regime labels
        min_len = min(len(features), len(regime_labels))
        features = features.iloc[:min_len]
        regime_labels = regime_labels[:min_len]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create target variable (0.5% price change)
        target = (data['close'].shift(-1) / data['close'] - 1) * 100  # Convert to percentage
        target = target.iloc[:min_len]
        
        # Train models for each regime
        unique_regimes = np.unique(regime_labels)
        training_results = {}
        
        for regime in unique_regimes:
            if regime == -1:  # Skip invalid regimes
                continue
                
            tprint(f"Training models for regime {regime}...")
            
            # Get data for this regime
            regime_mask = regime_labels == regime
            regime_features = features_scaled[regime_mask]
            regime_target = target[regime_mask]
            
            if len(regime_features) < 10:  # Need minimum data
                tprint(f"Insufficient data for regime {regime}, skipping...")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                regime_features, regime_target, 
                test_size=1-self.config.train_test_split,
                random_state=self.config.random_state
            )
            
            # Train base models
            base_models = {}
            
            # TCN (simplified as LSTM for now)
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                
                tcn_model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                tcn_model.compile(optimizer='adam', loss='mse')
                
                # Reshape for LSTM
                X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                tcn_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
                tcn_pred = tcn_model.predict(X_test_lstm).flatten()
                base_models['tcn'] = tcn_pred
                
            except ImportError:
                # Fallback to Random Forest if TensorFlow not available
                from sklearn.ensemble import RandomForestRegressor
                tcn_model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
                tcn_model.fit(X_train, y_train)
                tcn_pred = tcn_model.predict(X_test)
                base_models['tcn'] = tcn_pred
            
            # CatBoost
            catboost_model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=self.config.random_state,
                verbose=False
            )
            catboost_model.fit(X_train, y_train)
            catboost_pred = catboost_model.predict(X_test)
            base_models['catboost'] = catboost_pred
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_state,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            base_models['lightgbm'] = lgb_pred
            
            # Train meta-learner
            meta_features = np.column_stack(list(base_models.values()))
            meta_learner = ElasticNet(alpha=0.1, random_state=self.config.random_state)
            meta_learner.fit(meta_features, y_test)
            
            # Store models
            self.models[regime] = {
                'tcn': tcn_model,
                'catboost': catboost_model,
                'lightgbm': lgb_model
            }
            self.meta_learners[regime] = meta_learner
            
            # Calculate performance metrics
            meta_pred = meta_learner.predict(meta_features)
            mse = mean_squared_error(y_test, meta_pred)
            r2 = r2_score(y_test, meta_pred)
            
            training_results[regime] = {
                'mse': mse,
                'r2': r2,
                'n_samples': len(regime_features),
                'base_model_predictions': {k: v.tolist() for k, v in base_models.items()}
            }
            
            self.regime_models_trained[regime] = True
            tprint(f"Regime {regime} training completed: MSE={mse:.4f}, R²={r2:.4f}")
        
        self.is_trained = True
        tprint("Analyst training completed for all regimes")
        
        return training_results
    
    @handles_errors
    def predict_trading_opportunity(self, data: pd.DataFrame, regime_id: int,
                                  hmm_outputs: Optional[Dict[str, Any]] = None) -> AnalystPrediction:
        """
        Predict if we should trade based on current market conditions.
        
        Returns a green light (True) if conditions are favorable for trading.
        """
        if not self.is_trained or regime_id not in self.regime_models_trained:
            raise ValueError(f"Models not trained for regime {regime_id}")
        
        # Extract features for the latest data point
        features = self.extract_comprehensive_features(data.tail(self.config.lookback_periods), hmm_outputs)
        
        if len(features) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Use the latest data point
        latest_features = features.iloc[-1:].values
        
        # Scale features
        features_scaled = self.scaler.transform(latest_features)
        
        # Get base model predictions
        base_models = self.models[regime_id]
        base_predictions = {}
        
        for model_name, model in base_models.items():
            if model_name == 'tcn' and hasattr(model, 'predict'):
                # Handle LSTM model
                try:
                    pred = model.predict(features_scaled.reshape(1, features_scaled.shape[1], 1)).flatten()[0]
                except:
                    # Fallback for non-LSTM models
                    pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(features_scaled)[0]
            base_predictions[model_name] = pred
        
        # Get meta-learner prediction
        meta_features = np.array(list(base_predictions.values())).reshape(1, -1)
        meta_learner = self.meta_learners[regime_id]
        meta_prediction = meta_learner.predict(meta_features)[0]
        
        # Determine if we should trade
        should_trade = abs(meta_prediction) >= self.config.target_threshold
        
        # Calculate confidence based on prediction magnitude and consistency
        prediction_consistency = 1 - np.std(list(base_predictions.values())) / (np.mean(np.abs(list(base_predictions.values()))) + 1e-8)
        confidence = min(abs(meta_prediction) / self.config.target_threshold, 1.0) * prediction_consistency
        
        # Calculate feature importance (simplified)
        feature_importance = dict(zip(self.feature_names, 
                                    np.abs(features_scaled[0])))
        
        # Market conditions analysis
        market_conditions = {
            'volatility': features['volatility_20'].iloc[-1] if 'volatility_20' in features.columns else 0,
            'momentum': features['rsi_14'].iloc[-1] if 'rsi_14' in features.columns else 50,
            'volume_ratio': features['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in features.columns else 1,
            'trend_strength': features['trend_strength_20'].iloc[-1] if 'trend_strength_20' in features.columns else 0
        }
        
        return AnalystPrediction(
            timestamp=datetime.now(),
            should_trade=should_trade,
            confidence=confidence,
            base_model_predictions=base_predictions,
            meta_learner_prediction=meta_prediction,
            regime_id=regime_id,
            feature_importance=feature_importance,
            market_conditions=market_conditions
        )
    
    @handles_errors
    def should_run(self) -> bool:
        """Check if it's time to run the Analyst analysis."""
        if self.last_run_time is None:
            return True
        
        time_since_last_run = datetime.now() - self.last_run_time
        return time_since_last_run >= timedelta(minutes=self.config.run_interval_minutes)
    
    @handles_errors
    def run_analysis(self, data: pd.DataFrame, regime_id: int,
                    hmm_outputs: Optional[Dict[str, Any]] = None) -> Optional[AnalystPrediction]:
        """
        Run the complete Analyst analysis if it's time.
        
        This method should be called every 2 minutes.
        """
        if not self.should_run():
            return None
        
        if not self.is_trained or regime_id not in self.regime_models_trained:
            tprint(f"Analyst models not trained for regime {regime_id}")
            return None
        
        result = self.predict_trading_opportunity(data, regime_id, hmm_outputs)
        self.last_run_time = datetime.now()
        
        status = "GREEN LIGHT" if result.should_trade else "RED LIGHT"
        tprint(f"Analyst analysis completed: {status} "
              f"(confidence: {result.confidence:.3f}, prediction: {result.meta_learner_prediction:.3f}%)")
        
        return result
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'models': self.models,
            'meta_learners': self.meta_learners,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'regime_models_trained': self.regime_models_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        tprint(f"Analyst models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.meta_learners = model_data['meta_learners']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.regime_models_trained = model_data['regime_models_trained']
        
        tprint(f"Analyst models loaded from {filepath}")


# Factory function for easy instantiation
def create_analyst_regime_predictor(config: Optional[AnalystConfig] = None) -> AnalystRegimePredictor:
    """Create and return a new Analyst regime predictor instance."""
    if config is None:
        config = AnalystConfig()
    return AnalystRegimePredictor(config)