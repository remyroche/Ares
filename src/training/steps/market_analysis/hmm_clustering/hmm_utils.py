from ....core.decorators import handles_errors
"""Utility functions and decorators for HMM regime discovery."""

import logging
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import numpy as np
import pandas as pd

from src.utils.logger import system_logger

# Import decorators
from src.core.decorators.logging import log_execution_time
from src.core.decorators.cache import cached

# Placeholder decorators for compatibility
def monitor_feature_engineering(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def ensure_data_integrity(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def monitor_step_execution(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def secure_step_execution(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


def create_fallback_logger() -> Any:
    """Create a fallback logger if system_logger is not available."""
    logging.basicConfig(level = logging.INFO)
    return logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    path.mkdir(parents = True, exist_ok = True)
    return path


def safe_json_dump(data: Any, file_path: Path, **kwargs) -> None:
    """Safely dump data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, **kwargs)


class TechnicalIndicators:
    """Collection of technical indicator calculation methods."""
    
    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window = window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span = fast).mean()
        ema_slow = prices.ewm(span = slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        atr = tr.rolling(window = window).mean()
        return atr

    @staticmethod
    @handles_errors(fallback = pd.DataFrame())
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window = window).mean()
        std = prices.rolling(window = window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        bb_width = (bb_upper - bb_lower) / sma
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        bb_features = pd.DataFrame({
            'bb_upper': bb_upper, 
            'bb_middle': sma, 
            'bb_lower': bb_lower, 
            'bb_width': bb_width, 
            'bb_position': bb_position
        })
        return bb_features

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        tr_smooth = tr.rolling(window = window).mean()
        dm_plus_smooth = dm_plus.rolling(window = window).mean()
        dm_minus_smooth = dm_minus.rolling(window = window).mean()
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window = window).mean()
        return adx

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_sr_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate support/resistance strength indicator."""
        high_swing = df['high'].rolling(window = window, center = True).max()
        low_swing = df['low'].rolling(window = window, center = True).min()
        current_price = df['close']
        high_strength = (high_swing - current_price) / high_swing
        low_strength = (current_price - low_swing) / low_swing
        sr_strength = (high_strength + low_strength) / 2
        return sr_strength


class FeatureCalculator:
    """Handles feature calculation and preparation for HMM analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.indicators = TechnicalIndicators()

    @handles_errors(fallback = pd.DataFrame())
    def prepare_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for HMM regime discovery."""
        try:
            self.logger.info('🔧 Starting comprehensive feature preparation for HMM...')
            df = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop = True)
            
            features = pd.DataFrame()
            features['timestamp'] = df['timestamp']
            
            # Calculate momentum features
            self._add_momentum_features(features, df)
            
            # Calculate volatility features
            self._add_volatility_features(features, df)
            
            # Calculate volume features
            self._add_volume_features(features, df)
            
            # Calculate support/resistance features
            self._add_sr_features(features, df)
            
            # Calculate technical features
            self._add_technical_features(features, df)
            
            # Calculate feature interactions
            self._add_feature_interactions(features)
            
            # Clean and validate features
            hmm_features = self._clean_features(features)
            
            self.logger.info(f'✅ Comprehensive feature preparation completed: {len(hmm_features.columns)} features')
            return hmm_features
            
        except Exception as e:
            self.logger.exception(f'❌ Error preparing HMM features: {e}')
            raise

    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add momentum features."""
        self.logger.info('🚀 Calculating momentum features...')
        features['price_momentum_5'] = df['close'].pct_change(5)
        features['price_momentum_10'] = df['close'].pct_change(10)
        features['price_momentum_20'] = df['close'].pct_change(20)
        features['volume_momentum_5'] = df['volume'].pct_change(5)
        features['volume_momentum_10'] = df['volume'].pct_change(10)
        features['volume_momentum_20'] = df['volume'].pct_change(20)
        features['rsi'] = self.indicators.calculate_rsi(df['close'])
        features['rsi_momentum'] = features['rsi'].diff(5)
        features['macd'] = self.indicators.calculate_macd(df['close'])
        features['macd_momentum'] = features['macd'].diff(5)

    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volatility features."""
        self.logger.info('📈 Calculating volatility features...')
        features['volatility_5'] = df['close'].pct_change().rolling(window = 5).std()
        features['volatility_10'] = df['close'].pct_change().rolling(window = 10).std()
        features['volatility_20'] = df['close'].pct_change().rolling(window = 20).std()
        features['ewma_volatility_20'] = df['close'].pct_change().ewm(span = 20).std()
        features['volatility_acceleration'] = features['volatility_20'].diff()
        features['volatility_momentum'] = features['volatility_20'] - features['volatility_20'].shift(5)
        features['atr'] = self.indicators.calculate_atr(df)
        features['atr_normalized'] = features['atr'] / df['close']

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volume features."""
        self.logger.info('📊 Calculating volume features...')
        features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(window = 5).mean()
        features['volume_ratio_10'] = df['volume'] / df['volume'].rolling(window = 10).mean()
        features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(window = 20).mean()
        features['volume_change'] = df['volume'].pct_change()
        features['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        features['volume_price_trend_ratio'] = features['volume_price_trend'] / features['volume_price_trend'].rolling(20).mean()

    def _add_sr_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add support/resistance features."""
        self.logger.info('🎯 Calculating support/resistance features...')
        features['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        features['support_1'] = 2 * features['pivot_point'] - df['high']
        features['resistance_1'] = 2 * features['pivot_point'] - df['low']
        features['distance_to_support'] = (df['close'] - features['support_1']) / df['close']
        features['distance_to_resistance'] = (features['resistance_1'] - df['close']) / df['close']
        features['sr_strength'] = self.indicators.calculate_sr_strength(df)
        
        # Bollinger Bands
        bb_features = self.indicators.calculate_bollinger_bands(df['close'])
        features = pd.concat([features, bb_features], axis = 1)

    def _add_technical_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add technical features."""
        self.logger.info('🔧 Calculating additional technical features...')
        features['sma_20'] = df['close'].rolling(window = 20).mean()
        features['sma_50'] = df['close'].rolling(window = 50).mean()
        features['ema_12'] = df['close'].ewm(span = 12).mean()
        features['ema_26'] = df['close'].ewm(span = 26).mean()
        features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20']
        features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50']
        features['adx'] = self.indicators.calculate_adx(df)

    def _add_feature_interactions(self, features: pd.DataFrame) -> None:
        """Add feature interactions."""
        self.logger.info('🔄 Calculating feature interactions...')
        features['momentum_volume_interaction'] = features['price_momentum_10'] * features['volume_ratio_10']
        features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio_20']
        features['rsi_momentum_interaction'] = features['rsi'] * features['price_momentum_10']

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        self.logger.info('🧹 Cleaning and validating features...')
        hmm_features = features.drop('timestamp', axis = 1)
        initial_rows = len(hmm_features)
        
        # Forward fill technical indicators
        technical_cols = ['rsi', 'macd', 'adx', 'bb_position', 'bb_width']
        for col in technical_cols:
            if col in hmm_features.columns:
                hmm_features[col] = hmm_features[col].ffill()
        
        hmm_features = hmm_features.fillna(0)
        final_rows = len(hmm_features)
        removed_rows = initial_rows - final_rows
        
        self.logger.info(f'✅ Feature cleaning completed: {final_rows:,} rows, {len(hmm_features.columns)} features')
        return hmm_features


class RegimeAnalyzer:
    """Handles regime analysis and interpretation."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @handles_errors(default_return={'state_to_regime_map': {}, 'state_analysis': {}})
    def interpret_hmm_states(self, features: pd.DataFrame, state_sequence: np.ndarray, state_probs: np.ndarray) -> Dict[str, Any]:
        """Interpret HMM states based on feature characteristics."""
        try:
            self.logger.info('🔍 Interpreting HMM states...')
            state_analysis = {}
            state_to_regime_map = {}
            unique_states = sorted(set(state_sequence))
            
            for state in unique_states:
                state_mask = state_sequence == state
                state_data = features[state_mask]
                if len(state_data) == 0:
                    continue
                
                state_char = {
                    'count': len(state_data),
                    'percentage': len(state_data) / len(features) * 100
                }
                
                key_features = ['price_momentum_10', 'volatility_20', 'volume_ratio_10', 'rsi', 'adx', 'bb_position']
                for feature in key_features:
                    if feature in state_data.columns:
                        feature_data = state_data[feature].dropna()
                        if len(feature_data) > 0:
                            state_char[f'{feature}_mean'] = feature_data.mean()
                            state_char[f'{feature}_std'] = feature_data.std()
                
                state_analysis[state] = state_char
                regime_name = self._map_state_to_regime(state_char)
                state_to_regime_map[state] = regime_name
                
                self.logger.info(f"   State {state} → {regime_name}: {len(state_data)} periods ({state_char['percentage']:.1f}%)")
            
            return {'state_to_regime_map': state_to_regime_map, 'state_analysis': state_analysis}
            
        except Exception as e:
            self.logger.exception(f'❌ Error interpreting HMM states: {e}')
            return {'state_to_regime_map': {}, 'state_analysis': {}}

    @handles_errors(fallback='unknown_regime')
    def _map_state_to_regime(self, state_char: Dict[str, Any]) -> str:
        """Map state characteristics to regime name."""
        try:
            momentum = state_char.get('price_momentum_10_mean', 0)
            volatility = state_char.get('volatility_20_mean', 0)
            volume_ratio = state_char.get('volume_ratio_10_mean', 1)
            rsi = state_char.get('rsi_mean', 50)
            adx = state_char.get('adx_mean', 25)
            
            if volatility > 0.02:
                if momentum > 0.001:
                    return 'high_volatility_bull'
                elif momentum < -0.001:
                    return 'high_volatility_bear'
                else:
                    return 'high_volatility_neutral'
            elif volatility < 0.01:
                if momentum > 0.001:
                    return 'low_volatility_bull'
                elif momentum < -0.001:
                    return 'low_volatility_bear'
                else:
                    return 'low_volatility_neutral'
            elif momentum > 0.001:
                return 'medium_volatility_bull'
            elif momentum < -0.001:
                return 'medium_volatility_bear'
            else:
                return 'medium_volatility_neutral'
                
        except Exception as e:
            self.logger.warning(f'Error mapping state to regime: {e}')
            return 'unknown_regime'

    @handles_errors
    def calculate_regime_transitions(self, regimes: List[str]) -> Dict[str, Any]:
        """Calculate regime transition probabilities."""
        self.logger.info('🔄 Calculating regime transition probabilities...')
        transitions = {}
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            
            if current_regime not in transitions:
                transitions[current_regime] = {}
            if next_regime not in transitions[current_regime]:
                transitions[current_regime][next_regime] = 0
            transitions[current_regime][next_regime] += 1
        
        # Convert to probabilities
        for current_regime in transitions:
            total = sum(transitions[current_regime].values())
            for next_regime in transitions[current_regime]:
                transitions[current_regime][next_regime] /= total
        
        self.logger.info(f'✅ Transition matrix calculated for {len(transitions)} regimes')
        return transitions