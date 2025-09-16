#!/usr/bin/env python3
"""
Utility functions and decorators for HMM regime discovery.
Enhanced with common utilities integration for optimal performance.
"""

import logging
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import time
from datetime import datetime

from ....core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import common utilities for enhanced functionality
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, 
    calculate_data_quality_metrics, optimize_memory_usage
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_nan_to_num
)
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.serialization_utils import UniversalSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Import decorators

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
    try:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    except Exception as e:
        # If logging setup fails, create a minimal logger
        import sys
        class MinimalLogger:
            def info(self, msg): print(f"INFO: {msg}", file=sys.stdout)
            def warning(self, msg): print(f"WARNING: {msg}", file=sys.stderr)
            def error(self, msg): print(f"ERROR: {msg}", file=sys.stderr)
            def exception(self, msg): print(f"EXCEPTION: {msg}", file=sys.stderr)
        return MinimalLogger()

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    try:
        if path is None:
            raise ValueError("Path cannot be None")
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger = create_fallback_logger()
        logger.exception(f"Failed to create directory {path}: {e}")
        raise

def safe_json_dump(data: Any, file_path: Path, **kwargs) -> None:
    """Safely dump data to JSON file."""
    try:
        if data is None:
            raise ValueError("Data cannot be None")
        if file_path is None:
            raise ValueError("File path cannot be None")
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, **kwargs)
    except Exception as e:
        logger = create_fallback_logger()
        logger.exception(f"Failed to dump JSON to {file_path}: {e}")
        raise

class TechnicalIndicators:
    """Collection of technical indicator calculation methods."""
    
    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index with enhanced common utilities integration."""
        try:
            # Enhanced validation using common utilities
            if prices is None or prices.empty:
                raise ValueError("Prices series cannot be None or empty")
            window = validate_positive(window, "window")
            if len(prices) < window:
                raise ValueError(f"Prices length ({len(prices)}) must be >= window ({window})")
            
            # Calculate RSI with safe operations
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Use safe division to avoid division by zero
            rs = safe_divide(gain, loss, default=1.0)
            rsi = 100 - safe_divide(100, (1 + rs), default=50.0)
            
            # Apply safe conversion for any remaining issues
            rsi = safe_nan_to_num(rsi)
            return rsi
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"RSI calculation failed: {e}")
            return pd.Series()

    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            if prices is None or prices.empty:
                raise ValueError("Prices series cannot be None or empty")
            if fast < 1 or slow < 1 or signal < 1:
                raise ValueError("All parameters must be >= 1")
            if fast >= slow:
                raise ValueError("Fast period must be < slow period")
            if len(prices) < slow:
                raise ValueError(f"Prices length ({len(prices)}) must be >= slow period ({slow})")
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"MACD calculation failed: {e}")
            return pd.Series()

    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
            if window < 1:
                raise ValueError("Window must be >= 1")
            
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(df) < window:
                raise ValueError(f"DataFrame length ({len(df)}) must be >= window ({window})")
            
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"ATR calculation failed: {e}")
            return pd.Series()

    @staticmethod
    @handles_errors(fallback=pd.DataFrame())
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            if prices is None or prices.empty:
                raise ValueError("Prices series cannot be None or empty")
            if window < 1:
                raise ValueError("Window must be >= 1")
            if num_std <= 0:
                raise ValueError("num_std must be > 0")
            if len(prices) < window:
                raise ValueError(f"Prices length ({len(prices)}) must be >= window ({window})")
            
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            bb_upper = sma + std * num_std
            bb_lower = sma - std * num_std
            
            # Avoid division by zero
            bb_width = (bb_upper - bb_lower) / (sma + 1e-10)
            bb_position = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            bb_features = pd.DataFrame({
                'bb_upper': bb_upper, 
                'bb_middle': sma, 
                'bb_lower': bb_lower, 
                'bb_width': bb_width, 
                'bb_position': bb_position
            })
            return bb_features
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"Bollinger Bands calculation failed: {e}")
            return pd.DataFrame()

    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
            if window < 1:
                raise ValueError("Window must be >= 1")
            
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(df) < window + 1:  # Need at least window + 1 for shift operations
                raise ValueError(f"DataFrame length ({len(df)}) must be >= window + 1 ({window + 1})")
            
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            dm_plus = high - high.shift(1)
            dm_minus = low.shift(1) - low
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            tr_smooth = tr.rolling(window=window).mean()
            dm_plus_smooth = dm_plus.rolling(window=window).mean()
            dm_minus_smooth = dm_minus.rolling(window=window).mean()
            
            # Avoid division by zero
            di_plus = 100 * (dm_plus_smooth / (tr_smooth + 1e-10))
            di_minus = 100 * (dm_minus_smooth / (tr_smooth + 1e-10))
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
            adx = dx.rolling(window=window).mean()
            return adx
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"ADX calculation failed: {e}")
            return pd.Series()

    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_sr_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate support/resistance strength indicator."""
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
            if window < 1:
                raise ValueError("Window must be >= 1")
            
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(df) < window:
                raise ValueError(f"DataFrame length ({len(df)}) must be >= window ({window})")
            
            high_swing = df['high'].rolling(window=window, center=True).max()
            low_swing = df['low'].rolling(window=window, center=True).min()
            current_price = df['close']
            
            # Avoid division by zero
            high_strength = (high_swing - current_price) / (high_swing + 1e-10)
            low_strength = (current_price - low_swing) / (low_swing + 1e-10)
            sr_strength = (high_strength + low_strength) / 2
            return sr_strength
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"SR strength calculation failed: {e}")
            return pd.Series()

class FeatureCalculator:
    """Handles feature calculation and preparation for HMM analysis."""
    
    def __init__(self, logger: logging.Logger):
        if logger is None:
            self.logger = create_fallback_logger()
        else:
            self.logger = logger
        self.indicators = TechnicalIndicators()

    @handles_errors(fallback=pd.DataFrame())
    def prepare_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for HMM regime discovery."""
        try:
            if df is None or df.empty:
                raise ValueError("Input DataFrame cannot be None or empty")
            
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.logger.info('🔧 Starting comprehensive feature preparation for HMM...')
            df = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
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
        try:
            self.logger.info('🚀 Calculating momentum features...')
            
            if len(df) < 20:  # Need at least 20 periods for momentum calculations
                self.logger.warning(f"Insufficient data for momentum features: {len(df)} periods")
                return
            
            features['price_momentum_5'] = df['close'].pct_change(5)
            features['price_momentum_20'] = df['close'].pct_change(20)
            features['volume_momentum_5'] = df['volume'].pct_change(5)
            features['volume_momentum_20'] = df['volume'].pct_change(20)
            
            # Calculate technical indicators with error handling
            rsi = self.indicators.calculate_rsi(df['close'])
            if not rsi.empty:
                features['rsi'] = rsi
                features['rsi_momentum'] = features['rsi'].diff(5)
            
            macd = self.indicators.calculate_macd(df['close'])
            if not macd.empty:
                features['macd'] = macd
                features['macd_momentum'] = features['macd'].diff(5)
                
        except Exception as e:
            self.logger.exception(f"Error calculating momentum features: {e}")
            raise

    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volatility features."""
        try:
            self.logger.info('📈 Calculating volatility features...')
            
            if len(df) < 20:  # Need at least 20 periods for volatility calculations
                self.logger.warning(f"Insufficient data for volatility features: {len(df)} periods")
                return
            
            price_changes = df['close'].pct_change()
            features['volatility_5'] = price_changes.rolling(window=5).std()
            features['volatility_10'] = price_changes.rolling(window=10).std()
            features['volatility_20'] = price_changes.rolling(window=20).std()
            features['ewma_volatility_20'] = price_changes.ewm(span=20).std()
            
            # Calculate volatility derivatives only if we have enough data
            if 'volatility_20' in features.columns:
                features['volatility_acceleration'] = features['volatility_20'].diff()
                features['volatility_momentum'] = features['volatility_20'] - features['volatility_20'].shift(5)
            
            # Calculate ATR with error handling
            atr = self.indicators.calculate_atr(df)
            if not atr.empty:
                features['atr'] = atr
                # Avoid division by zero
                features['atr_normalized'] = features['atr'] / (df['close'] + 1e-10)
                
        except Exception as e:
            self.logger.exception(f"Error calculating volatility features: {e}")
            raise

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volume features."""
        try:
            self.logger.info('📊 Calculating volume features...')
            
            if len(df) < 20:  # Need at least 20 periods for volume calculations
                self.logger.warning(f"Insufficient data for volume features: {len(df)} periods")
                return
            
            # Calculate volume ratios with error handling
            volume_ma_5 = df['volume'].rolling(window=5).mean()
            volume_ma_10 = df['volume'].rolling(window=10).mean()
            volume_ma_20 = df['volume'].rolling(window=20).mean()
            
            # Avoid division by zero
            features['volume_ratio_5'] = df['volume'] / (volume_ma_5 + 1e-10)
            features['volume_ratio_10'] = df['volume'] / (volume_ma_10 + 1e-10)
            features['volume_ratio_20'] = df['volume'] / (volume_ma_20 + 1e-10)
            
            features['volume_change'] = df['volume'].pct_change()
            features['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
            
            # Calculate volume price trend ratio only if we have enough data
            vpt_ma = features['volume_price_trend'].rolling(20).mean()
            features['volume_price_trend_ratio'] = features['volume_price_trend'] / (vpt_ma + 1e-10)
                
        except Exception as e:
            self.logger.exception(f"Error calculating volume features: {e}")
            raise

    def _add_sr_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add support/resistance features."""
        try:
            self.logger.info('🎯 Calculating support/resistance features...')
            
            if len(df) < 20:  # Need at least 20 periods for SR calculations
                self.logger.warning(f"Insufficient data for SR features: {len(df)} periods")
                return
            
            features['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
            features['support_1'] = 2 * features['pivot_point'] - df['high']
            features['resistance_1'] = 2 * features['pivot_point'] - df['low']
            
            # Avoid division by zero
            features['distance_to_support'] = (df['close'] - features['support_1']) / (df['close'] + 1e-10)
            features['distance_to_resistance'] = (features['resistance_1'] - df['close']) / (df['close'] + 1e-10)
            
            # Calculate SR strength with error handling
            sr_strength = self.indicators.calculate_sr_strength(df)
            if not sr_strength.empty:
                features['sr_strength'] = sr_strength
            
            # Bollinger Bands with error handling
            bb_features = self.indicators.calculate_bollinger_bands(df['close'])
            if not bb_features.empty:
                features = pd.concat([features, bb_features], axis=1)
                
        except Exception as e:
            self.logger.exception(f"Error calculating SR features: {e}")
            raise

    def _add_technical_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add technical features."""
        try:
            self.logger.info('🔧 Calculating additional technical features...')
            
            if len(df) < 50:  # Need at least 50 periods for technical features
                self.logger.warning(f"Insufficient data for technical features: {len(df)} periods")
                return
            
            features['sma_20'] = df['close'].rolling(window=20).mean()
            features['sma_50'] = df['close'].rolling(window=50).mean()
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Avoid division by zero
            features['price_vs_sma20'] = (df['close'] - features['sma_20']) / (features['sma_20'] + 1e-10)
            features['price_vs_sma50'] = (df['close'] - features['sma_50']) / (features['sma_50'] + 1e-10)
            
            # Calculate ADX with error handling
            adx = self.indicators.calculate_adx(df)
            if not adx.empty:
                features['adx'] = adx
                
        except Exception as e:
            self.logger.exception(f"Error calculating technical features: {e}")
            raise

    def _add_feature_interactions(self, features: pd.DataFrame) -> None:
        """Add feature interactions."""
        try:
            self.logger.info('🔄 Calculating feature interactions...')
            
            # Only calculate interactions if the required features exist
            if 'price_momentum_5' in features.columns and 'volume_ratio_10' in features.columns:
                features['momentum_volume_interaction'] = features['price_momentum_5'] * features['volume_ratio_10']
            
            if 'volatility_20' in features.columns and 'volume_ratio_20' in features.columns:
                features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio_20']
            
            if 'rsi' in features.columns and 'price_momentum_5' in features.columns:
                features['rsi_momentum_interaction'] = features['rsi'] * features['price_momentum_5']
                
        except Exception as e:
            self.logger.exception(f"Error calculating feature interactions: {e}")
            raise

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        try:
            self.logger.info('🧹 Cleaning and validating features...')
            
            if features is None or features.empty:
                raise ValueError("Features DataFrame cannot be None or empty")
            
            hmm_features = features.drop('timestamp', axis=1)
            initial_rows = len(hmm_features)
            
            # Forward fill technical indicators
            technical_cols = ['rsi', 'macd', 'adx', 'bb_position', 'bb_width']
            for col in technical_cols:
                if col in hmm_features.columns:
                    hmm_features[col] = hmm_features[col].ffill()
            
            # Handle infinite values
            hmm_features = hmm_features.replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaN values
            hmm_features = hmm_features.fillna(0)
            
            final_rows = len(hmm_features)
            removed_rows = initial_rows - final_rows
            
            if removed_rows > 0:
                self.logger.warning(f"Removed {removed_rows} rows during cleaning")
            
            self.logger.info(f'✅ Feature cleaning completed: {final_rows:,} rows, {len(hmm_features.columns)} features')
            return hmm_features
            
        except Exception as e:
            self.logger.exception(f"Error cleaning features: {e}")
            raise

class RegimeAnalyzer:
    """Handles regime analysis and interpretation."""
    
    def __init__(self, logger: logging.Logger):
        if logger is None:
            self.logger = create_fallback_logger()
        else:
            self.logger = logger

    @handles_errors(default_return={'state_to_regime_map': {}, 'state_analysis': {}})
    def interpret_hmm_states(self, features: pd.DataFrame, state_sequence: np.ndarray, state_probs: np.ndarray) -> Dict[str, Any]:
        """Interpret HMM states based on feature characteristics."""
        try:
            if features is None or features.empty:
                raise ValueError("Features DataFrame cannot be None or empty")
            if state_sequence is None or len(state_sequence) == 0:
                raise ValueError("State sequence cannot be None or empty")
            if len(features) != len(state_sequence):
                raise ValueError(f"Features length ({len(features)}) must match state sequence length ({len(state_sequence)})")
            
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
                
                key_features = ['price_momentum_5', 'volatility_20', 'volume_ratio_10', 'rsi', 'adx', 'bb_position']
                for feature in key_features:
                    if feature in state_data.columns:
                        feature_data = state_data[feature].dropna()
                        if len(feature_data) > 0:
                            state_char[f'{feature}_mean'] = float(feature_data.mean())
                            state_char[f'{feature}_std'] = float(feature_data.std())
                
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
            if state_char is None:
                return 'unknown_regime'
            
            momentum = state_char.get('price_momentum_5_mean', 0)
            volatility = state_char.get('volatility_20_mean', 0)
            volume_ratio = state_char.get('volume_ratio_10_mean', 1)
            rsi = state_char.get('rsi_mean', 50)
            adx = state_char.get('adx_mean', 25)
            
            # Ensure values are numeric
            try:
                momentum = float(momentum) if momentum is not None else 0
                volatility = float(volatility) if volatility is not None else 0
                volume_ratio = float(volume_ratio) if volume_ratio is not None else 1
                rsi = float(rsi) if rsi is not None else 50
                adx = float(adx) if adx is not None else 25
            except (ValueError, TypeError):
                return 'unknown_regime'
            
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
        try:
            if regimes is None or len(regimes) < 2:
                self.logger.warning("Insufficient regime data for transition calculation")
                return {}
            
            self.logger.info('🔄 Calculating regime transition probabilities...')
            transitions = {}
            
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                
                if current_regime is None or next_regime is None:
                    continue
                
                if current_regime not in transitions:
                    transitions[current_regime] = {}
                if next_regime not in transitions[current_regime]:
                    transitions[current_regime][next_regime] = 0
                transitions[current_regime][next_regime] += 1
            
            # Convert to probabilities
            for current_regime in transitions:
                total = sum(transitions[current_regime].values())
                if total > 0:
                    for next_regime in transitions[current_regime]:
                        transitions[current_regime][next_regime] /= total
            
            self.logger.info(f'✅ Transition matrix calculated for {len(transitions)} regimes')
            return transitions
            
        except Exception as e:
            self.logger.exception(f"Error calculating regime transitions: {e}")
            return {}


# Enhanced utility functions with common utilities integration

class EnhancedHMMUtils:
    """Enhanced HMM utilities with common utilities integration."""
    
    def __init__(self, logger: Optional[Any] = None):
        """Initialize enhanced HMM utilities."""
        self.logger = logger or create_fallback_logger()
        self.klines_manager = KlinesParquetManager()
        self.serializer = UniversalSerializer()
        self.matrix_ops = UnifiedMatrixOperations()
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
    
    def load_market_data_enhanced(
        self, 
        symbol: str, 
        interval: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Load market data with enhanced common utilities integration."""
        try:
            self.logger.info(f"Loading market data for {symbol} {interval}")
            
            # Get data info
            data_info = self.klines_manager.get_data_info(symbol, interval)
            if not data_info['available']:
                self.logger.warning(f"No data available for {symbol} {interval}")
                return None
            
            # Load data
            data = self.klines_manager.load_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None or data.empty:
                self.logger.warning("No data loaded")
                return None
            
            # Validate data quality using common utilities
            quality_metrics = calculate_data_quality_metrics(data)
            self.logger.info(f"Data quality metrics: {quality_metrics}")
            
            # Apply memory optimization if available
            if self.memory_optimizer:
                data = optimize_memory_usage(data)
                self.logger.info("Applied memory optimization to data")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return None
    
    def engineer_features_enhanced(
        self, 
        data: pd.DataFrame, 
        lookback_windows: List[int] = [5, 10, 20, 50],
        technical_indicators: List[str] = ["rsi", "macd", "bollinger_bands", "atr"]
    ) -> pd.DataFrame:
        """Engineer features with enhanced common utilities integration."""
        try:
            self.logger.info("Engineering features with enhanced utilities")
            
            features = data.copy()
            
            # Calculate technical indicators with enhanced validation
            for window in lookback_windows:
                window = validate_positive(window, f"window_{window}")
                
                if "rsi" in technical_indicators:
                    features[f"rsi_{window}"] = TechnicalIndicators.calculate_rsi(
                        features['close'], window
                    )
                
                if "macd" in technical_indicators:
                    macd_line, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(
                        features['close'], window, window*2, window*3
                    )
                    features[f"macd_{window}"] = macd_line
                    features[f"macd_signal_{window}"] = macd_signal
                    features[f"macd_hist_{window}"] = macd_hist
                
                if "bollinger_bands" in technical_indicators:
                    bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
                        features['close'], window
                    )
                    features[f"bb_upper_{window}"] = bb_upper
                    features[f"bb_middle_{window}"] = bb_middle
                    features[f"bb_lower_{window}"] = bb_lower
                    features[f"bb_width_{window}"] = safe_divide(
                        bb_upper - bb_lower, bb_middle, default=0.0
                    )
                
                if "atr" in technical_indicators:
                    features[f"atr_{window}"] = TechnicalIndicators.calculate_atr(
                        features, window
                    )
            
            # Price-based features with safe operations
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = safe_log(features['close'] / features['close'].shift(1))
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['price_momentum'] = safe_divide(
                features['close'], features['close'].shift(20), default=1.0
            ) - 1
            
            # Volume features
            if 'volume' in features.columns:
                features['volume_ma'] = features['volume'].rolling(window=20).mean()
                features['volume_ratio'] = safe_divide(
                    features['volume'], features['volume_ma'], default=1.0
                )
            
            # Remove rows with NaN values
            features = features.dropna()
            
            # Apply safe conversion to handle any remaining issues
            for col in features.select_dtypes(include=[np.number]).columns:
                features[col] = safe_nan_to_num(features[col])
            
            self.logger.info(f"Engineered {len(features.columns)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to engineer features: {e}")
            return pd.DataFrame()
    
    def calculate_regime_metrics_enhanced(
        self,
        features: pd.DataFrame,
        state_sequence: np.ndarray,
        state_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate enhanced regime metrics using common utilities."""
        try:
            metrics = {}
            
            # Regime stability
            regime_changes = np.sum(np.diff(state_sequence) != 0)
            metrics['regime_stability'] = safe_divide(
                len(state_sequence) - regime_changes, len(state_sequence), default=0.0
            )
            
            # Regime balance
            unique_regimes, counts = np.unique(state_sequence, return_counts=True)
            if len(counts) > 1:
                regime_balance = safe_divide(
                    np.mean(counts), np.std(counts) + 1e-10, default=1.0
                )
            else:
                regime_balance = 1.0
            metrics['regime_balance'] = regime_balance
            
            # Probability confidence
            max_probs = np.max(state_probs, axis=1)
            metrics['avg_confidence'] = np.mean(max_probs)
            metrics['min_confidence'] = np.min(max_probs)
            
            # Regime duration statistics
            regime_durations = []
            current_regime = state_sequence[0]
            current_duration = 1
            
            for i in range(1, len(state_sequence)):
                if state_sequence[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = state_sequence[i]
                    current_duration = 1
            
            regime_durations.append(current_duration)
            
            if regime_durations:
                metrics['avg_regime_duration'] = np.mean(regime_durations)
                metrics['min_regime_duration'] = np.min(regime_durations)
                metrics['max_regime_duration'] = np.max(regime_durations)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate regime metrics: {e}")
            return {}
    
    def save_results_enhanced(
        self,
        results: Dict[str, Any],
        filepath: str,
        include_metadata: bool = True
    ) -> bool:
        """Save results with enhanced serialization."""
        try:
            if include_metadata:
                results['metadata'] = {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'common_utilities_integration': True
                }
            
            return self.serializer.save(results, filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False
    
    def load_results_enhanced(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load results with enhanced deserialization."""
        try:
            return self.serializer.load(filepath)
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return None


def create_enhanced_hmm_utils(logger: Optional[Any] = None) -> EnhancedHMMUtils:
    """Create enhanced HMM utils instance with common utilities integration."""
    return EnhancedHMMUtils(logger)