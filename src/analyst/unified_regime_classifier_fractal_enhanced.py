# src/analyst/unified_regime_classifier_fractal_enhanced.py
"""
Enhanced Unified Regime Classifier with Step 6 Features and Performance Optimizations

This version integrates:
1. Features from Step 6 (technical indicators with optimal lookback periods)
2. Performance optimizations (caching, vectorization, incremental updates)
3. Better ML integration through richer feature sets
"""

from src.core.decorators import handles_errors
import os
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from functools import lru_cache
import numba
from src.config import CONFIG
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
import logging
import asyncio
from src.core.decorators import (
from src.core.decorators.errors import handles_errors
    validates as validate_data_quality,
    traced as with_tracing_span,
)


class UnifiedRegimeClassifierFractalEnhanced:
    """
    Enhanced Fractal Location Classifier with:
    1. Step 6 feature integration (technical indicators)
    2. Performance optimizations (caching, vectorization)
    3. Richer ML-ready feature sets
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("UnifiedRegimeClassifierFractalEnhanced")
        self.exchange = exchange
        self.symbol = symbol
        
        # Fractal timeframes
        self.fractal_timeframes = self.config.get("fractal_timeframes", [
            {"name": "1m", "periods": 60, "weight": 0.1},
            {"name": "5m", "periods": 12, "weight": 0.15},
            {"name": "15m", "periods": 4, "weight": 0.2},
            {"name": "1h", "periods": 1, "weight": 0.25},
            {"name": "4h", "periods": 0.25, "weight": 0.2},
            {"name": "1d", "periods": 0.042, "weight": 0.1},
        ])
        
        # Distance and strength parameters
        self.distance_normalization = self.config.get("distance_normalization", "percentage")
        self.min_strength_threshold = self.config.get("min_strength_threshold", 0.3)
        self.max_relevant_distance = self.config.get("max_relevant_distance", 0.05)
        
        # Step 6 feature integration - optimal lookback periods
        self.technical_indicators_config = {
            "RSI": {"periods": [7, 21, 50]},
            "ATR": {"periods": [7, 14, 30]},
            "BB": {"periods": [10, 20, 50]},
            "SMA": {"periods": [5, 20, 100]},
            "EMA": {"periods": [8, 21, 55]},
            "MACD": {"fast": 12, "slow": 26, "signal": 9},
            "ADX": {"periods": [7, 14, 25]},
            "MFI": {"periods": [7, 14, 30]},
            "OBV": {"normalize": True},
        }
        
        # Performance optimization
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_size = self.config.get("cache_size", 1000)
        self._feature_cache = {}
        self._sr_cache = {}
        self._last_update_time = None
        
        # ML feature configuration
        self.enable_rich_features = self.config.get("enable_rich_features", True)
        self.feature_version = "v2"  # Enhanced feature set
        
        # S/R Predictor
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        self.sr_predictor = None
        
        # Model components
        self.scaler = StandardScaler()
        
        # Training status
        self.trained = False
        self.last_training_time = None
        
    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid data for location classification"),
            AttributeError: (False, "Missing required attributes"),
        },
        default_return=False,
        context="classifier initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced fractal location classifier."""
        try:
            self.logger.info("Initializing Enhanced Fractal Location Classifier...")
            
            # Initialize S/R Predictor if enabled
            if self.enable_sr_integration:
                self.sr_predictor = SRBreakoutPredictor(
                    self.global_config,
                    self.exchange,
                    self.symbol
                )
                sr_init_success = await self.sr_predictor.initialize()
                if not sr_init_success:
                    self.logger.warning("Failed to initialize S/R Predictor, will use basic analysis")
                    self.sr_predictor = None
            
            self.logger.info("✅ Enhanced Fractal Location Classifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier: {e}")
            return False
    
    async def classify_location(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced location classification with Step 6 features and optimizations.
        """
        if features_df.empty or len(features_df) < 200:
            return self._get_default_classification()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(features_df)
            if self.enable_caching and cache_key in self._feature_cache:
                cached_result = self._feature_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    return cached_result['data']
            
            # Get current price and calculate ATR
            current_price = features_df['close'].iloc[-1]
            atr = self._calculate_atr_optimized(features_df)
            
            # Calculate technical indicators from Step 6
            technical_features = self._calculate_technical_indicators(features_df)
            
            # Perform fractal S/R analysis
            fractal_sr_data = await self._analyze_fractal_sr_levels_optimized(features_df)
            
            # Aggregate S/R levels
            aggregated_levels = self._aggregate_sr_levels(fractal_sr_data, current_price)
            
            # Calculate core location metrics
            location_metrics = self._calculate_location_metrics(
                current_price, 
                aggregated_levels, 
                atr
            )
            
            # Add enhanced features
            if self.enable_rich_features:
                # Market microstructure features
                microstructure = self._calculate_microstructure_features(features_df)
                location_metrics.update(microstructure)
                
                # Price action context
                price_context = self._calculate_price_action_context(features_df)
                location_metrics.update(price_context)
                
                # Technical indicator integration
                location_metrics.update(technical_features)
                
                # Volume profile features
                volume_features = self._calculate_volume_profile_features(features_df)
                location_metrics.update(volume_features)
            
            # Add metadata
            location_metrics['timestamp'] = datetime.now().isoformat()
            location_metrics['current_price'] = current_price
            location_metrics['atr'] = atr
            location_metrics['feature_version'] = self.feature_version
            
            # Cache result
            if self.enable_caching:
                self._feature_cache[cache_key] = {
                    'data': location_metrics,
                    'timestamp': datetime.now()
                }
                self._cleanup_cache()
            
            return location_metrics
            
        except Exception as e:
            self.logger.error(f"Error in enhanced location classification: {e}")
            return self._get_default_classification()
    
    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_atr_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Vectorized ATR calculation using Numba for speed."""
        n = len(high)
        if n < period:
            return 0.0
        
        # True Range calculation
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, max(hc, lc))
        
        # ATR as EMA
        atr = np.zeros(n)
        atr[:period] = np.mean(tr[:period])
        
        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            atr[i] = (tr[i] - atr[i-1]) * multiplier + atr[i-1]
        
        return atr[-1]
    
    def _calculate_atr_optimized(self, df: pd.DataFrame) -> float:
        """Optimized ATR calculation with caching."""
        return self._calculate_atr_vectorized(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            14
        )
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from Step 6 configuration."""
        indicators = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # RSI with multiple periods
        for period in self.technical_indicators_config['RSI']['periods']:
            rsi = talib.RSI(close, timeperiod=period)
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                indicators[f'rsi_{period}'] = rsi[-1]
                # RSI divergence
                if len(rsi) > 5:
                    price_slope = (close[-1] - close[-5]) / close[-5]
                    rsi_slope = (rsi[-1] - rsi[-5]) / (rsi[-5] + 1e-8)
                    indicators[f'rsi_{period}_divergence'] = price_slope - rsi_slope
        
        # ATR normalized by price
        for period in self.technical_indicators_config['ATR']['periods']:
            atr = talib.ATR(high, low, close, timeperiod=period)
            if len(atr) > 0 and not np.isnan(atr[-1]):
                indicators[f'atr_normalized_{period}'] = atr[-1] / close[-1]
        
        # Bollinger Bands
        for period in self.technical_indicators_config['BB']['periods']:
            upper, middle, lower = talib.BBANDS(close, timeperiod=period)
            if len(upper) > 0 and not np.isnan(upper[-1]):
                # BB position (0-1)
                bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1] + 1e-8)
                indicators[f'bb_position_{period}'] = bb_position
                # BB squeeze (bandwidth)
                bb_squeeze = (upper[-1] - lower[-1]) / (middle[-1] + 1e-8)
                indicators[f'bb_squeeze_{period}'] = bb_squeeze
        
        # Moving averages and their relationships
        sma_values = {}
        for period in self.technical_indicators_config['SMA']['periods']:
            sma = talib.SMA(close, timeperiod=period)
            if len(sma) > 0 and not np.isnan(sma[-1]):
                sma_values[period] = sma[-1]
                indicators[f'price_to_sma_{period}'] = close[-1] / sma[-1]
        
        # SMA slopes
        if len(sma_values) >= 2:
            sorted_periods = sorted(sma_values.keys())
            for i in range(len(sorted_periods) - 1):
                fast_period = sorted_periods[i]
                slow_period = sorted_periods[i + 1]
                indicators[f'sma_ratio_{fast_period}_{slow_period}'] = (
                    sma_values[fast_period] / sma_values[slow_period]
                )
        
        # MACD
        macd_config = self.technical_indicators_config['MACD']
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=macd_config['fast'],
            slowperiod=macd_config['slow'],
            signalperiod=macd_config['signal']
        )
        if len(macd) > 0 and not np.isnan(macd[-1]):
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = signal[-1]
            indicators['macd_hist'] = hist[-1]
            indicators['macd_hist_normalized'] = hist[-1] / (close[-1] * 0.01)  # Normalize by 1% of price
        
        # ADX for trend strength
        for period in self.technical_indicators_config['ADX']['periods']:
            adx = talib.ADX(high, low, close, timeperiod=period)
            if len(adx) > 0 and not np.isnan(adx[-1]):
                indicators[f'adx_{period}'] = adx[-1]
        
        # MFI (Money Flow Index)
        for period in self.technical_indicators_config['MFI']['periods']:
            mfi = talib.MFI(high, low, close, volume, timeperiod=period)
            if len(mfi) > 0 and not np.isnan(mfi[-1]):
                indicators[f'mfi_{period}'] = mfi[-1]
        
        # OBV (On Balance Volume) - normalized
        obv = talib.OBV(close, volume)
        if len(obv) > 0 and not np.isnan(obv[-1]):
            # Normalize by average volume
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            indicators['obv_normalized'] = obv[-1] / (avg_volume * 20 + 1e-8)
        
        return indicators
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {}
        
        # Price acceleration
        if len(df) >= 3:
            price_velocity = df['close'].pct_change()
            price_acceleration = price_velocity.diff()
            features['price_acceleration'] = price_acceleration.iloc[-1]
            features['price_acceleration_3'] = price_acceleration.iloc[-3:].mean()
        
        # High-Low spread
        hl_spread = (df['high'] - df['low']) / df['close']
        features['hl_spread_current'] = hl_spread.iloc[-1]
        features['hl_spread_avg'] = hl_spread.iloc[-10:].mean()
        features['hl_spread_ratio'] = hl_spread.iloc[-1] / (hl_spread.iloc[-20:].mean() + 1e-8)
        
        # Close position in candle
        candle_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        features['candle_position'] = candle_position.iloc[-1]
        features['candle_position_avg'] = candle_position.iloc[-5:].mean()
        
        # Volume-weighted price momentum
        volume_weight = df['volume'] / df['volume'].rolling(20).mean()
        price_change = df['close'].pct_change()
        vw_momentum = (price_change * volume_weight).rolling(5).sum()
        features['volume_weighted_momentum'] = vw_momentum.iloc[-1]
        
        return features
    
    def _calculate_price_action_context(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price action context features."""
        features = {}
        
        # Recent price swings
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        current_price = df['close'].iloc[-1]
        
        # Position in recent range
        if recent_high != recent_low:
            features['position_in_range'] = (current_price - recent_low) / (recent_high - recent_low)
        else:
            features['position_in_range'] = 0.5
        
        # Swing detection
        swing_high_5 = df['high'].iloc[-5:].max()
        swing_low_5 = df['low'].iloc[-5:].min()
        features['near_swing_high'] = (swing_high_5 - current_price) / current_price
        features['near_swing_low'] = (current_price - swing_low_5) / current_price
        
        # Momentum persistence
        returns = df['close'].pct_change()
        positive_returns = (returns > 0).astype(int)
        features['momentum_persistence'] = positive_returns.iloc[-10:].sum() / 10
        
        # Volatility regime
        short_vol = returns.iloc[-10:].std()
        long_vol = returns.iloc[-50:].std()
        features['volatility_ratio'] = short_vol / (long_vol + 1e-8)
        
        return features
    
    def _calculate_volume_profile_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile features."""
        features = {}
        
        # Volume momentum
        volume_sma_5 = df['volume'].rolling(5).mean()
        volume_sma_20 = df['volume'].rolling(20).mean()
        features['volume_momentum'] = volume_sma_5.iloc[-1] / (volume_sma_20.iloc[-1] + 1e-8)
        
        # Volume spikes
        volume_std = df['volume'].rolling(20).std()
        volume_zscore = (df['volume'] - volume_sma_20) / (volume_std + 1e-8)
        features['volume_spike'] = volume_zscore.iloc[-1]
        features['recent_volume_spikes'] = (volume_zscore.iloc[-5:] > 2).sum()
        
        # Price-volume correlation
        if len(df) >= 20:
            price_changes = df['close'].pct_change().iloc[-20:]
            volume_changes = df['volume'].pct_change().iloc[-20:]
            if len(price_changes.dropna()) >= 10:
                features['price_volume_correlation'] = price_changes.corr(volume_changes)
            else:
                features['price_volume_correlation'] = 0.0
        
        return features
    
    async def _analyze_fractal_sr_levels_optimized(self, features_df: pd.DataFrame) -> Dict[str, Dict]:
        """Optimized fractal S/R analysis with caching."""
        # Check SR cache
        cache_key = f"sr_{len(features_df)}_{features_df.index[-1]}"
        if self.enable_caching and cache_key in self._sr_cache:
            return self._sr_cache[cache_key]
        
        fractal_sr_data = {}
        
        # Process timeframes in parallel if possible
        tasks = []
        for tf_config in self.fractal_timeframes:
            tf_name = tf_config['name']
            periods = int(tf_config['periods'] * len(features_df)) if tf_config['periods'] < 1 else int(tf_config['periods'])
            
            if periods > len(features_df):
                periods = len(features_df)
            
            # Get data window for this timeframe
            tf_data = features_df.iloc[-periods:] if periods > 0 else features_df
            
            # Analyze S/R for this timeframe
            task = self._analyze_timeframe_sr(tf_data, tf_name, tf_config['weight'])
            tasks.append(task)
        
        # Wait for all timeframe analyses
        results = await asyncio.gather(*tasks)
        
        # Combine results
        for tf_name, result in zip([tf['name'] for tf in self.fractal_timeframes], results):
            fractal_sr_data[tf_name] = result
        
        # Cache result
        if self.enable_caching:
            self._sr_cache[cache_key] = fractal_sr_data
        
        return fractal_sr_data
    
    async def _analyze_timeframe_sr(self, data: pd.DataFrame, timeframe: str, weight: float) -> Dict:
        """Analyze S/R for a single timeframe."""
        if self.sr_predictor and self.enable_sr_integration:
            sr_levels = await self._get_enhanced_sr_levels(data, timeframe)
        else:
            sr_levels = self._get_basic_sr_levels(data, timeframe)
        
        return {
            'support_levels': sr_levels['support'],
            'resistance_levels': sr_levels['resistance'],
            'weight': weight
        }
    
    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key for dataframe."""
        return f"{len(df)}_{df.index[-1]}_{df['close'].iloc[-1]}"
    
    def _is_cache_valid(self, cached_item: Dict) -> bool:
        """Check if cached item is still valid."""
        if 'timestamp' not in cached_item:
            return False
        
        age = (datetime.now() - cached_item['timestamp']).total_seconds()
        return age < 60  # Cache valid for 1 minute
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        if len(self._feature_cache) > self.cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self._feature_cache.items(),
                key=lambda x: x[1].get('timestamp', datetime.min)
            )
            
            # Keep only recent items
            self._feature_cache = dict(sorted_items[-self.cache_size:])
    
    def get_ml_features(self, classification: Dict[str, Any]) -> pd.Series:
        """
        Get comprehensive ML-ready features including Step 6 indicators.
        """
        features = {}
        
        # Core distance and strength features
        features['support_distance'] = classification.get('support_distance', 1.0)
        features['resistance_distance'] = classification.get('resistance_distance', 1.0)
        features['support_strength'] = classification.get('support_strength', 0.0)
        features['resistance_strength'] = classification.get('resistance_strength', 0.0)
        features['combined_location_score'] = classification.get('combined_location_score', 0.0)
        features['location_quality'] = classification.get('location_quality', 0.0)
        
        # Technical indicators from Step 6
        for key, value in classification.items():
            if key.startswith(('rsi_', 'atr_', 'bb_', 'sma_', 'ema_', 'macd', 'adx_', 'mfi_', 'obv_')):
                features[key] = value
        
        # Microstructure features
        for key, value in classification.items():
            if key.startswith(('price_acceleration', 'hl_spread', 'candle_position', 'volume_weighted')):
                features[key] = value
        
        # Price action features
        for key, value in classification.items():
            if key.startswith(('position_in_range', 'near_swing', 'momentum_persistence', 'volatility_ratio')):
                features[key] = value
        
        # Volume profile features
        for key, value in classification.items():
            if key.startswith(('volume_momentum', 'volume_spike', 'price_volume_correlation')):
                features[key] = value
        
        return pd.Series(features)
    
    # Keep other methods from the simplified version...
    # (aggregate_sr_levels, calculate_location_metrics, etc.)