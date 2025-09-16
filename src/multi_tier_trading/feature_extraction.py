"""
Enhanced Feature Extraction for Multi-Tier Trading System

This module provides comprehensive feature extraction for each tier:
- HMM: 100 features for regime detection
- Analyst: 300+ features including cross-timeframe analysis
- Tactician: 50+ high-frequency timing features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from src.utils.tprint import tprint


class HMMFeatureExtractor:
    """Feature extractor for HMM system (100 features, 1h base)."""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract 100 features for HMM regime detection."""
        tprint("Extracting HMM features...")
        
        features = pd.DataFrame(index=data.index)
        
        # Basic price features (20 features)
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['open']
        features['price_range'] = data['high'] - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        features['body_ratio'] = features['body_size'] / features['price_range']
        features['shadow_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / features['price_range']
        
        # Momentum indicators (30 features)
        for period in [5, 10, 20, 50]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = data['close'].pct_change(period)
            features[f'price_acceleration_{period}'] = features[f'momentum_{period}'].diff()
        
        # MACD variations (10 features)
        for fast, slow in [(12, 26), (5, 35), (19, 39)]:
            macd_line, macd_signal, macd_hist = self._calculate_macd(data['close'], fast, slow, 9)
            features[f'macd_{fast}_{slow}'] = macd_line
            features[f'macd_signal_{fast}_{slow}'] = macd_signal
            features[f'macd_hist_{fast}_{slow}'] = macd_hist
        
        # Volatility indicators (20 features)
        for period in [10, 20, 50]:
            features[f'atr_{period}'] = self._calculate_atr(data, period)
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = self._calculate_bollinger_upper(data['close'], period)
            features[f'bb_lower_{period}'] = self._calculate_bollinger_lower(data['close'], period)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / data['close']
        
        # Volume features (10 features)
        if 'volume' in data.columns:
            for period in [5, 10, 20]:
                features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
            
            features['obv'] = self._calculate_obv(data['close'], data['volume'])
            features['vpt'] = self._calculate_vpt(data['close'], data['volume'])
        
        # Cross-timeframe features (10 features)
        for period in [2, 4, 8, 12]:  # 2h, 4h, 8h, 12h
            if len(data) > period:
                features[f'ctf_returns_{period}h'] = data['close'].pct_change(period)
                features[f'ctf_volatility_{period}h'] = data['close'].rolling(period).std()
        
        # Select top 100 features
        features = features.dropna()
        if len(features.columns) > 100:
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(100).index.tolist()
            features = features[selected_features]
        
        self.feature_names = features.columns.tolist()
        tprint(f"Extracted {len(self.feature_names)} HMM features")
        
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


class AnalystFeatureExtractor:
    """Feature extractor for Analyst system (300+ features, 5m base)."""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, data: pd.DataFrame, hmm_output: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Extract 300+ features for Analyst decision making."""
        tprint("Extracting Analyst features...")
        
        features = pd.DataFrame(index=data.index)
        
        # Basic price features (30 features)
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['open']
        features['price_range'] = data['high'] - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        features['body_ratio'] = features['body_size'] / features['price_range']
        features['shadow_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / features['price_range']
        
        # Momentum indicators (60 features)
        for period in [5, 10, 14, 20, 30, 50]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = data['close'].pct_change(period)
            features[f'roc_ma_{period}'] = features[f'roc_{period}'].rolling(5).mean()
        
        # MACD variations (20 features)
        for fast, slow in [(12, 26), (5, 35), (19, 39), (8, 21)]:
            macd_line, macd_signal, macd_hist = self._calculate_macd(data['close'], fast, slow, 9)
            features[f'macd_{fast}_{slow}'] = macd_line
            features[f'macd_signal_{fast}_{slow}'] = macd_signal
            features[f'macd_hist_{fast}_{slow}'] = macd_hist
        
        # Volatility indicators (40 features)
        for period in [10, 14, 20, 30, 50]:
            features[f'atr_{period}'] = self._calculate_atr(data, period)
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = self._calculate_bollinger_upper(data['close'], period)
            features[f'bb_lower_{period}'] = self._calculate_bollinger_lower(data['close'], period)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / data['close']
            features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Volume analysis (50 features)
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
        
        # Cross-timeframe features (80 features)
        for period in [3, 6, 12, 24, 48, 72, 144, 288]:  # 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
            if len(data) > period:
                features[f'ctf_returns_{period}'] = data['close'].pct_change(period)
                features[f'ctf_volatility_{period}'] = data['close'].rolling(period).std()
                features[f'ctf_range_{period}'] = (data['high'].rolling(period).max() - data['low'].rolling(period).min()) / data['close']
                
                if 'volume' in data.columns:
                    features[f'ctf_volume_{period}'] = data['volume'].rolling(period).mean()
                    features[f'ctf_volume_ratio_{period}'] = data['volume'] / features[f'ctf_volume_{period}']
                
                features[f'ctf_rsi_{period}'] = self._calculate_rsi(data['close'], period)
                features[f'ctf_atr_{period}'] = self._calculate_atr(data, period)
        
        # HMM integration (20 features)
        if hmm_output:
            if 'regime_probs' in hmm_output:
                regime_probs = hmm_output['regime_probs']
                for i, prob in enumerate(regime_probs):
                    features[f'hmm_regime_{i}_prob'] = prob
            
            if 'dominant_regime' in hmm_output:
                features['hmm_dominant_regime'] = hmm_output['dominant_regime']
            
            if 'regime_characteristics' in hmm_output:
                regime_chars = hmm_output['regime_characteristics']
                for key, value in regime_chars.items():
                    features[f'hmm_{key}'] = value
        
        # Additional technical features (40 features)
        features['stoch_k'] = self._calculate_stochastic_k(data)
        features['stoch_d'] = self._calculate_stochastic_d(data)
        features['williams_r'] = self._calculate_williams_r(data)
        features['cci'] = self._calculate_cci(data)
        features['roc'] = self._calculate_roc(data)
        features['ppo'] = self._calculate_ppo(data)
        
        # Market microstructure (20 features)
        features['bid_ask_spread'] = (data['high'] - data['low']) / data['close']
        features['price_impact'] = abs(data['close'] - data['open']) / data['close']
        features['intraday_volatility'] = (data['high'] - data['low']) / data['open']
        features['gap_up'] = data['open'] > data['close'].shift(1)
        features['gap_down'] = data['open'] < data['close'].shift(1)
        
        # Select top 300+ features
        features = features.dropna()
        if len(features.columns) > 300:
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(300).index.tolist()
            features = features[selected_features]
        
        self.feature_names = features.columns.tolist()
        tprint(f"Extracted {len(self.feature_names)} Analyst features")
        
        return features
    
    # Include all the same helper methods as HMMFeatureExtractor
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


class TacticianFeatureExtractor:
    """Feature extractor for Tactician system (50+ features, 1m base)."""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, data: pd.DataFrame, hmm_output: Optional[Dict[str, Any]] = None, 
                        analyst_output: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Extract 50+ features for Tactician timing prediction."""
        tprint("Extracting Tactician features...")
        
        features = pd.DataFrame(index=data.index)
        
        # High-frequency price features (15 features)
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['open']
        features['price_range'] = data['high'] - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['body_ratio'] = features['body_size'] / features['price_range']
        features['price_acceleration'] = features['returns'].diff()
        features['volatility_1m'] = data['close'].rolling(5).std()
        features['volatility_5m'] = data['close'].rolling(30).std()
        features['volatility_ratio'] = features['volatility_1m'] / features['volatility_5m']
        
        # Short-term momentum (10 features)
        for period in [2, 3, 5, 10, 15]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # Volume analysis (10 features)
        if 'volume' in data.columns:
            for period in [2, 5, 10, 15]:
                features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
            
            features['obv'] = self._calculate_obv(data['close'], data['volume'])
            features['vpt'] = self._calculate_vpt(data['close'], data['volume'])
        
        # Cross-timeframe features (10 features)
        for period in [5, 15, 30, 60, 120]:  # 5m, 15m, 30m, 1h, 2h
            if len(data) > period:
                features[f'ctf_momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                features[f'ctf_volatility_{period}'] = data['close'].rolling(period).std()
        
        # HMM integration (5 features)
        if hmm_output:
            features['hmm_dominant_regime'] = hmm_output.get('dominant_regime', 0)
            features['hmm_confidence'] = hmm_output.get('confidence', 0.5)
            if 'regime_characteristics' in hmm_output:
                regime_chars = hmm_output['regime_characteristics']
                features['hmm_volatility'] = regime_chars.get('volatility', 0.02)
                features['hmm_momentum'] = regime_chars.get('mean_returns', 0)
        
        # Analyst integration (5 features)
        if analyst_output:
            features['analyst_should_trade'] = analyst_output.get('should_trade', False)
            features['analyst_confidence'] = analyst_output.get('confidence', 0.5)
            features['analyst_prediction'] = analyst_output.get('meta_learner_prediction', 0)
            features['analyst_regime_id'] = analyst_output.get('regime_id', 0)
        
        # Entry timing signals (5 features)
        features['price_breakout'] = self._detect_price_breakout(data)
        features['volume_breakout'] = self._detect_volume_breakout(data)
        features['momentum_divergence'] = self._detect_momentum_divergence(data)
        features['volatility_breakout'] = self._detect_volatility_breakout(data)
        features['entry_signal_strength'] = self._calculate_entry_signal_strength(data)
        
        # Select top 50+ features
        features = features.dropna()
        if len(features.columns) > 50:
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(50).index.tolist()
            features = features[selected_features]
        
        self.feature_names = features.columns.tolist()
        tprint(f"Extracted {len(self.feature_names)} Tactician features")
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
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
    
    def _detect_price_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detect price breakout signals."""
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        
        breakout_up = data['close'] > upper_band
        breakout_down = data['close'] < lower_band
        
        return np.where(breakout_up, 1, np.where(breakout_down, -1, 0))
    
    def _detect_volume_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detect volume breakout signals."""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
            
        volume_ma = data['volume'].rolling(20).mean()
        volume_std = data['volume'].rolling(20).std()
        volume_threshold = volume_ma + 2 * volume_std
        
        return (data['volume'] > volume_threshold).astype(int)
    
    def _detect_momentum_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Detect momentum divergence signals."""
        price_momentum = data['close'].pct_change(10)
        rsi = self._calculate_rsi(data['close'], 14)
        
        price_trend = price_momentum.rolling(5).mean()
        rsi_trend = rsi.rolling(5).mean()
        
        divergence = np.where(
            (price_trend > 0) & (rsi_trend < 0), 1,  # Bearish divergence
            np.where((price_trend < 0) & (rsi_trend > 0), -1, 0)  # Bullish divergence
        )
        
        return pd.Series(divergence, index=data.index)
    
    def _detect_volatility_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detect volatility breakout signals."""
        volatility = data['close'].rolling(20).std()
        vol_ma = volatility.rolling(20).mean()
        vol_std = volatility.rolling(20).std()
        
        vol_breakout = volatility > (vol_ma + 2 * vol_std)
        return vol_breakout.astype(int)
    
    def _calculate_entry_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate entry signal strength."""
        momentum = data['close'].pct_change(5)
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean() if 'volume' in data.columns else 1
        volatility = data['close'].rolling(10).std()
        
        signal = abs(momentum) * volume_ratio / (volatility + 1e-8)
        return signal.rolling(5).mean()


# Factory functions
def create_hmm_feature_extractor() -> HMMFeatureExtractor:
    """Create HMM feature extractor."""
    return HMMFeatureExtractor()

def create_analyst_feature_extractor() -> AnalystFeatureExtractor:
    """Create Analyst feature extractor."""
    return AnalystFeatureExtractor()

def create_tactician_feature_extractor() -> TacticianFeatureExtractor:
    """Create Tactician feature extractor."""
    return TacticianFeatureExtractor()