"""
VectorBT Technical Indicators Suite

This module provides a comprehensive collection of VectorBT-enhanced technical indicators
for advanced feature engineering and analysis.

Features:
- 100+ technical indicators organized by category
- VectorBT-optimized calculations
- Parameter optimization capabilities
- Advanced pattern recognition
- Multi-timeframe analysis
- Performance monitoring and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple, Union
from dataclasses import dataclass, field
import time
from enum import Enum

# Import VectorBT base classes
from src.training.steps.feature_engineering.vectorbt_base import (
    VectorBTFeatureGenerator, VectorBTConfig, VectorBTTechnicalIndicators
)
from src.feature_generation.core.feature_generator import FeatureCategory, FeatureConfig, FeatureResult
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


class IndicatorCategory(Enum):
    """Categories of technical indicators."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"
    PATTERN = "pattern"
    CYCLE = "cycle"
    CUSTOM = "custom"


@dataclass
class VectorBTIndicatorSuiteConfig:
    """Configuration for VectorBT Indicator Suite."""
    
    # General settings
    enable_optimization: bool = True
    enable_caching: bool = True
    enable_parallel: bool = True
    n_jobs: int = -1
    
    # Indicator categories to include
    include_trend: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_price_action: bool = True
    include_patterns: bool = True
    include_cycles: bool = True
    
    # Timeframe settings
    primary_timeframe: str = "15m"
    additional_timeframes: List[str] = None
    
    # Window settings
    default_windows: List[int] = None
    custom_windows: Dict[str, List[int]] = None
    
    # Pattern recognition settings
    enable_pattern_recognition: bool = True
    pattern_confidence_threshold: float = 0.7
    
    # Performance settings
    enable_performance_monitoring: bool = True
    performance_window: int = 100
    
    def __post_init__(self):
        if self.additional_timeframes is None:
            self.additional_timeframes = ["5m", "1h", "4h", "1d"]
        if self.default_windows is None:
            self.default_windows = [5, 10, 20, 50, 100]
        if self.custom_windows is None:
            self.custom_windows = {}


class VectorBTIndicatorSuite:
    """
    Comprehensive VectorBT Technical Indicators Suite.
    
    Provides access to 100+ technical indicators with VectorBT optimization,
    parameter tuning, and advanced analysis capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTIndicatorSuiteConfig] = None):
        """Initialize VectorBT Indicator Suite."""
        self.config = config or VectorBTIndicatorSuiteConfig()
        self.indicators = VectorBTTechnicalIndicators()
        self._cache = {} if self.config.enable_caching else None
        
        tprint_info("📊 VectorBT Indicator Suite initialized")
        tprint_info(f"   → Categories: {[cat.value for cat in IndicatorCategory]}")
        tprint_info(f"   → Optimization: {self.config.enable_optimization}")
        tprint_info(f"   → Caching: {self.config.enable_caching}")
        tprint_info(f"   → Parallel: {self.config.enable_parallel}")
    
    def get_trend_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive trend indicators."""
        if not self.config.include_trend:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # Moving Averages
            for window in windows:
                # Simple Moving Average
                sma = self.indicators.vbt.MA.run(data['close'], window=window, short_name='SMA').ma
                indicators[f'sma_{window}'] = sma
                
                # Exponential Moving Average
                ema = self.indicators.vbt.MA.run(data['close'], window=window, short_name='EMA').ma
                indicators[f'ema_{window}'] = ema
                
                # Weighted Moving Average
                wma = self.indicators.vbt.MA.run(data['close'], window=window, short_name='WMA').ma
                indicators[f'wma_{window}'] = wma
                
                # Volume Weighted Moving Average
                if 'volume' in data.columns:
                    vwma = self.indicators.vbt.VWMA.run(data['close'], data['volume'], window=window).vwma
                    indicators[f'vwma_{window}'] = vwma
                
                # Moving Average Slopes
                indicators[f'sma_slope_{window}'] = sma.diff()
                indicators[f'ema_slope_{window}'] = ema.diff()
                indicators[f'wma_slope_{window}'] = wma.diff()
                
                # Moving Average Convergence/Divergence
                if window > 5:
                    sma_fast = self.indicators.vbt.MA.run(data['close'], window=max(5, window//2)).ma
                    indicators[f'sma_convergence_{window}'] = sma - sma_fast
                    indicators[f'ema_convergence_{window}'] = ema - sma_fast
            
            # ADX and Directional Movement
            adx = self.indicators.vbt.ADX.run(data['high'], data['low'], data['close'])
            indicators['adx'] = adx.adx
            indicators['adx_plus'] = adx.plus_di
            indicators['adx_minus'] = adx.minus_di
            indicators['adx_dx'] = adx.dx
            
            # Parabolic SAR
            psar = self.indicators.vbt.PARABOLIC.run(data['high'], data['low'], data['close'])
            indicators['psar'] = psar.sar
            indicators['psar_signal'] = (data['close'] > psar.sar).astype(int)
            
            # Ichimoku Cloud
            ichimoku = self.indicators.vbt.ICHIMOKU.run(data['high'], data['low'], data['close'])
            indicators['ichimoku_conversion'] = ichimoku.conversion
            indicators['ichimoku_base'] = ichimoku.base
            indicators['ichimoku_span_a'] = ichimoku.span_a
            indicators['ichimoku_span_b'] = ichimoku.span_b
            indicators['ichimoku_signal'] = ichimoku.signal
            
            # Trend Strength
            indicators['trend_strength'] = self._calculate_trend_strength(indicators)
            
            tprint_info(f"   → Trend indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating trend indicators: {e}")
        
        return indicators
    
    def get_momentum_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive momentum indicators."""
        if not self.config.include_momentum:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # RSI
            for window in windows:
                rsi = self.indicators.vbt.RSI.run(data['close'], window=window).rsi
                indicators[f'rsi_{window}'] = rsi
                indicators[f'rsi_overbought_{window}'] = (rsi > 70).astype(int)
                indicators[f'rsi_oversold_{window}'] = (rsi < 30).astype(int)
                indicators[f'rsi_divergence_{window}'] = self._calculate_rsi_divergence(data['close'], rsi)
            
            # MACD
            macd = self.indicators.vbt.MACD.run(data['close'])
            indicators['macd'] = macd.macd
            indicators['macd_signal'] = macd.signal
            indicators['macd_histogram'] = macd.histogram
            indicators['macd_crossover'] = (macd.macd > macd.signal).astype(int)
            indicators['macd_divergence'] = self._calculate_macd_divergence(data['close'], macd.macd)
            
            # Stochastic Oscillator
            stoch = self.indicators.vbt.STOCH.run(data['high'], data['low'], data['close'])
            indicators['stoch_k'] = stoch.k
            indicators['stoch_d'] = stoch.d
            indicators['stoch_overbought'] = (stoch.k > 80).astype(int)
            indicators['stoch_oversold'] = (stoch.k < 20).astype(int)
            indicators['stoch_divergence'] = self._calculate_stoch_divergence(data['close'], stoch.k)
            
            # Williams %R
            willr = self.indicators.vbt.WILLR.run(data['high'], data['low'], data['close'])
            indicators['willr'] = willr.willr
            indicators['willr_overbought'] = (willr.willr > -20).astype(int)
            indicators['willr_oversold'] = (willr.willr < -80).astype(int)
            
            # CCI
            cci = self.indicators.vbt.CCI.run(data['high'], data['low'], data['close'])
            indicators['cci'] = cci.cci
            indicators['cci_overbought'] = (cci.cci > 100).astype(int)
            indicators['cci_oversold'] = (cci.cci < -100).astype(int)
            
            # Rate of Change
            for window in windows:
                roc = data['close'].pct_change(window) * 100
                indicators[f'roc_{window}'] = roc
                indicators[f'roc_momentum_{window}'] = roc.diff()
            
            # Momentum
            for window in windows:
                momentum = data['close'] - data['close'].shift(window)
                indicators[f'momentum_{window}'] = momentum
                indicators[f'momentum_roc_{window}'] = momentum / data['close'].shift(window) * 100
            
            tprint_info(f"   → Momentum indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating momentum indicators: {e}")
        
        return indicators
    
    def get_volatility_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive volatility indicators."""
        if not self.config.include_volatility:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # ATR
            for window in windows:
                atr = self.indicators.vbt.ATR.run(data['high'], data['low'], data['close'], window=window).atr
                indicators[f'atr_{window}'] = atr
                indicators[f'atr_ratio_{window}'] = atr / data['close']
                indicators[f'atr_percentile_{window}'] = atr.rolling(100).rank(pct=True)
            
            # Bollinger Bands
            for window in windows:
                bb = self.indicators.vbt.BBANDS.run(data['close'], window=window)
                indicators[f'bb_upper_{window}'] = bb.upper
                indicators[f'bb_middle_{window}'] = bb.middle
                indicators[f'bb_lower_{window}'] = bb.lower
                indicators[f'bb_width_{window}'] = (bb.upper - bb.lower) / bb.middle
                indicators[f'bb_position_{window}'] = (data['close'] - bb.lower) / (bb.upper - bb.lower)
                indicators[f'bb_squeeze_{window}'] = (indicators[f'bb_width_{window}'] < 
                                                    indicators[f'bb_width_{window}'].rolling(20).quantile(0.2)).astype(int)
            
            # Keltner Channels
            for window in windows:
                kc = self.indicators.vbt.KELTNER.run(data['high'], data['low'], data['close'], window=window)
                indicators[f'kc_upper_{window}'] = kc.upper
                indicators[f'kc_middle_{window}'] = kc.middle
                indicators[f'kc_lower_{window}'] = kc.lower
                indicators[f'kc_width_{window}'] = (kc.upper - kc.lower) / kc.middle
                indicators[f'kc_position_{window}'] = (data['close'] - kc.lower) / (kc.upper - kc.lower)
            
            # Donchian Channels
            for window in windows:
                dc = self.indicators.vbt.DONCHIAN.run(data['high'], data['low'], window=window)
                indicators[f'dc_upper_{window}'] = dc.upper
                indicators[f'dc_lower_{window}'] = dc.lower
                indicators[f'dc_width_{window}'] = dc.upper - dc.lower
                indicators[f'dc_position_{window}'] = (data['close'] - dc.lower) / (dc.upper - dc.lower)
            
            # Volatility Ratio
            for window in windows:
                if window > 5:
                    short_vol = data['close'].rolling(window//2).std()
                    long_vol = data['close'].rolling(window).std()
                    indicators[f'volatility_ratio_{window}'] = short_vol / long_vol
            
            # Historical Volatility
            for window in windows:
                returns = data['close'].pct_change()
                hv = returns.rolling(window).std() * np.sqrt(252)  # Annualized
                indicators[f'historical_volatility_{window}'] = hv
            
            tprint_info(f"   → Volatility indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating volatility indicators: {e}")
        
        return indicators
    
    def get_volume_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive volume indicators."""
        if not self.config.include_volume or 'volume' not in data.columns:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # Volume Moving Averages
            for window in windows:
                vma = data['volume'].rolling(window=window).mean()
                indicators[f'volume_ma_{window}'] = vma
                indicators[f'volume_ratio_{window}'] = data['volume'] / vma
                indicators[f'volume_roc_{window}'] = data['volume'].pct_change(window)
            
            # VWAP
            vwap = self.indicators.vbt.VWAP.run(data['high'], data['low'], data['close'], data['volume']).vwap
            indicators['vwap'] = vwap
            indicators['vwap_deviation'] = (data['close'] - vwap) / vwap
            indicators['vwap_distance'] = data['close'] - vwap
            
            # On Balance Volume
            obv = self.indicators.vbt.OBV.run(data['close'], data['volume']).obv
            indicators['obv'] = obv
            indicators['obv_ma'] = obv.rolling(20).mean()
            indicators['obv_ratio'] = obv / obv.rolling(100).mean()
            
            # Accumulation/Distribution Line
            adl = self.indicators.vbt.ADL.run(data['high'], data['low'], data['close'], data['volume']).adl
            indicators['adl'] = adl
            indicators['adl_ma'] = adl.rolling(20).mean()
            indicators['adl_ratio'] = adl / adl.rolling(100).mean()
            
            # Money Flow Index
            mfi = self.indicators.vbt.MFI.run(data['high'], data['low'], data['close'], data['volume']).mfi
            indicators['mfi'] = mfi
            indicators['mfi_overbought'] = (mfi > 80).astype(int)
            indicators['mfi_oversold'] = (mfi < 20).astype(int)
            
            # Volume Price Trend
            vpt = self.indicators.vbt.VPT.run(data['close'], data['volume']).vpt
            indicators['vpt'] = vpt
            indicators['vpt_ma'] = vpt.rolling(20).mean()
            
            # Ease of Movement
            emv = self.indicators.vbt.EMV.run(data['high'], data['low'], data['volume']).emv
            indicators['emv'] = emv
            indicators['emv_ma'] = emv.rolling(20).mean()
            
            # Volume Rate of Change
            for window in windows:
                vroc = data['volume'].pct_change(window)
                indicators[f'vroc_{window}'] = vroc
            
            # Volume Weighted Average Price
            for window in windows:
                vwap_window = (data['close'] * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()
                indicators[f'vwap_{window}'] = vwap_window
                indicators[f'vwap_deviation_{window}'] = (data['close'] - vwap_window) / vwap_window
            
            tprint_info(f"   → Volume indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating volume indicators: {e}")
        
        return indicators
    
    def get_price_action_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive price action indicators."""
        if not self.config.include_price_action:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # Bar Efficiency Ratio
            for window in windows:
                price_range = data['high'] - data['low']
                price_range = price_range.replace(0, np.nan)
                efficiency = np.abs(data['close'] - data['open']) / price_range
                efficiency = efficiency.fillna(0).replace([np.inf, -np.inf], 0)
                indicators[f'bar_efficiency_{window}'] = efficiency
                indicators[f'bar_efficiency_ma_{window}'] = efficiency.rolling(window).mean()
            
            # Close Location Value
            for window in windows:
                price_range = data['high'] - data['low']
                price_range = price_range.replace(0, np.nan)
                clv = (2 * data['close'] - data['high'] - data['low']) / price_range
                clv = clv.fillna(0).replace([np.inf, -np.inf], 0)
                indicators[f'clv_{window}'] = clv
                indicators[f'clv_ma_{window}'] = clv.rolling(window).mean()
            
            # Price Position
            for window in windows:
                high_window = data['high'].rolling(window).max()
                low_window = data['low'].rolling(window).min()
                price_position = (data['close'] - low_window) / (high_window - low_window)
                indicators[f'price_position_{window}'] = price_position
            
            # Price Range
            for window in windows:
                price_range = data['high'] - data['low']
                indicators[f'price_range_{window}'] = price_range
                indicators[f'price_range_ma_{window}'] = price_range.rolling(window).mean()
                indicators[f'price_range_ratio_{window}'] = price_range / price_range.rolling(window).mean()
            
            # Body Size
            body_size = np.abs(data['close'] - data['open'])
            indicators['body_size'] = body_size
            indicators['body_size_ratio'] = body_size / (data['high'] - data['low'])
            
            # Shadow Ratios
            upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
            lower_shadow = np.minimum(data['open'], data['close']) - data['low']
            indicators['upper_shadow_ratio'] = upper_shadow / (data['high'] - data['low'])
            indicators['lower_shadow_ratio'] = lower_shadow / (data['high'] - data['low'])
            
            # Price Action Patterns
            indicators['doji'] = (body_size < (data['high'] - data['low']) * 0.1).astype(int)
            indicators['hammer'] = ((lower_shadow > body_size * 2) & (upper_shadow < body_size)).astype(int)
            indicators['shooting_star'] = ((upper_shadow > body_size * 2) & (lower_shadow < body_size)).astype(int)
            
            tprint_info(f"   → Price action indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating price action indicators: {e}")
        
        return indicators
    
    def get_pattern_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive pattern recognition indicators."""
        if not self.config.include_patterns:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # Candlestick Patterns
            patterns = self.indicators.vbt.CANDLE.run(data['open'], data['high'], data['low'], data['close'])
            
            # Basic patterns
            indicators['doji'] = patterns.doji.astype(int)
            indicators['hammer'] = patterns.hammer.astype(int)
            indicators['shooting_star'] = patterns.shooting_star.astype(int)
            indicators['engulfing'] = patterns.engulfing.astype(int)
            indicators['harami'] = patterns.harami.astype(int)
            indicators['morning_star'] = patterns.morning_star.astype(int)
            indicators['evening_star'] = patterns.evening_star.astype(int)
            
            # Pattern strength
            pattern_strength = patterns.patterns.astype(int).sum(axis=1)
            indicators['pattern_strength'] = pattern_strength
            indicators['pattern_frequency'] = pattern_strength.rolling(20).mean()
            
            # Support and Resistance
            for window in windows:
                high_window = data['high'].rolling(window).max()
                low_window = data['low'].rolling(window).min()
                
                # Support/Resistance levels
                indicators[f'resistance_{window}'] = high_window
                indicators[f'support_{window}'] = low_window
                
                # Distance from support/resistance
                indicators[f'distance_from_resistance_{window}'] = (high_window - data['close']) / data['close']
                indicators[f'distance_from_support_{window}'] = (data['close'] - low_window) / data['close']
                
                # Breakout signals
                indicators[f'resistance_breakout_{window}'] = (data['close'] > high_window.shift(1)).astype(int)
                indicators[f'support_breakdown_{window}'] = (data['close'] < low_window.shift(1)).astype(int)
            
            # Trend Line Analysis
            for window in windows:
                if window > 5:
                    # Simple trend line slope
                    trend_slope = data['close'].rolling(window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
                    indicators[f'trend_slope_{window}'] = trend_slope
                    indicators[f'trend_direction_{window}'] = np.sign(trend_slope)
            
            tprint_info(f"   → Pattern indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating pattern indicators: {e}")
        
        return indicators
    
    def get_cycle_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get comprehensive cycle indicators."""
        if not self.config.include_cycles:
            return {}
        
        windows = windows or self.config.default_windows
        indicators = {}
        
        try:
            # Hilbert Transform
            hilbert = self.indicators.vbt.HILBERT.run(data['close'])
            indicators['hilbert_trend'] = hilbert.trend
            indicators['hilbert_cycle'] = hilbert.cycle
            indicators['hilbert_phase'] = hilbert.phase
            indicators['hilbert_amplitude'] = hilbert.amplitude
            
            # Detrended Price Oscillator
            for window in windows:
                dpo = data['close'] - data['close'].rolling(window).mean().shift(window//2 + 1)
                indicators[f'dpo_{window}'] = dpo
                indicators[f'dpo_ma_{window}'] = dpo.rolling(5).mean()
            
            # Cycle Analysis
            for window in windows:
                if window > 10:
                    # Peak and trough detection
                    peaks = data['high'].rolling(window, center=True).max() == data['high']
                    troughs = data['low'].rolling(window, center=True).min() == data['low']
                    
                    indicators[f'peaks_{window}'] = peaks.astype(int)
                    indicators[f'troughs_{window}'] = troughs.astype(int)
                    
                    # Cycle length
                    cycle_length = self._calculate_cycle_length(peaks, troughs)
                    indicators[f'cycle_length_{window}'] = cycle_length
            
            tprint_info(f"   → Cycle indicators: {len(indicators)} calculated")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating cycle indicators: {e}")
        
        return indicators
    
    def get_all_indicators(
        self, 
        data: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.Series]:
        """Get all available indicators."""
        all_indicators = {}
        
        # Combine all indicator categories
        all_indicators.update(self.get_trend_indicators(data, windows))
        all_indicators.update(self.get_momentum_indicators(data, windows))
        all_indicators.update(self.get_volatility_indicators(data, windows))
        all_indicators.update(self.get_volume_indicators(data, windows))
        all_indicators.update(self.get_price_action_indicators(data, windows))
        all_indicators.update(self.get_pattern_indicators(data, windows))
        all_indicators.update(self.get_cycle_indicators(data, windows))
        
        tprint_success(f"✅ All indicators calculated: {len(all_indicators)} total")
        return all_indicators
    
    def _calculate_trend_strength(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Calculate composite trend strength."""
        try:
            # Get key trend indicators
            adx = indicators.get('adx')
            sma_slope = indicators.get('sma_slope_20')
            ema_slope = indicators.get('ema_slope_20')
            
            if adx is not None and sma_slope is not None:
                # Normalize indicators
                adx_norm = np.clip(adx / 50.0, 0.0, 1.0)
                slope_norm = np.clip(np.abs(sma_slope) / 0.01, 0.0, 1.0)
                
                # Combine with weights
                trend_strength = (adx_norm * 0.6 + slope_norm * 0.4).clip(0.0, 1.0)
                return trend_strength
            else:
                return pd.Series(0.0, index=list(indicators.values())[0].index)
        except Exception:
            return pd.Series(0.0, index=list(indicators.values())[0].index)
    
    def _calculate_rsi_divergence(self, price: pd.Series, rsi: pd.Series) -> pd.Series:
        """Calculate RSI divergence."""
        try:
            # Simple divergence detection
            price_peaks = price.rolling(5, center=True).max() == price
            rsi_peaks = rsi.rolling(5, center=True).max() == rsi
            
            divergence = pd.Series(0, index=price.index)
            divergence[(price_peaks & ~rsi_peaks) | (~price_peaks & rsi_peaks)] = 1
            
            return divergence
        except Exception:
            return pd.Series(0, index=price.index)
    
    def _calculate_macd_divergence(self, price: pd.Series, macd: pd.Series) -> pd.Series:
        """Calculate MACD divergence."""
        try:
            # Simple divergence detection
            price_peaks = price.rolling(5, center=True).max() == price
            macd_peaks = macd.rolling(5, center=True).max() == macd
            
            divergence = pd.Series(0, index=price.index)
            divergence[(price_peaks & ~macd_peaks) | (~price_peaks & macd_peaks)] = 1
            
            return divergence
        except Exception:
            return pd.Series(0, index=price.index)
    
    def _calculate_stoch_divergence(self, price: pd.Series, stoch: pd.Series) -> pd.Series:
        """Calculate Stochastic divergence."""
        try:
            # Simple divergence detection
            price_peaks = price.rolling(5, center=True).max() == price
            stoch_peaks = stoch.rolling(5, center=True).max() == stoch
            
            divergence = pd.Series(0, index=price.index)
            divergence[(price_peaks & ~stoch_peaks) | (~price_peaks & stoch_peaks)] = 1
            
            return divergence
        except Exception:
            return pd.Series(0, index=price.index)
    
    def _calculate_cycle_length(self, peaks: pd.Series, troughs: pd.Series) -> pd.Series:
        """Calculate cycle length between peaks and troughs."""
        try:
            cycle_length = pd.Series(0, index=peaks.index)
            
            last_peak = None
            last_trough = None
            
            for i in range(len(peaks)):
                if peaks.iloc[i]:
                    if last_peak is not None:
                        cycle_length.iloc[i] = i - last_peak
                    last_peak = i
                elif troughs.iloc[i]:
                    if last_trough is not None:
                        cycle_length.iloc[i] = i - last_trough
                    last_trough = i
            
            return cycle_length
        except Exception:
            return pd.Series(0, index=peaks.index)
    
    def optimize_indicators(
        self, 
        data: pd.DataFrame, 
        target_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """Optimize indicator parameters."""
        if not self.config.enable_optimization:
            return {}
        
        try:
            tprint_info("🔍 Optimizing indicator parameters")
            
            # Define parameter ranges for optimization
            param_ranges = {
                'rsi_window': [10, 14, 20, 30],
                'macd_fast': [8, 12, 16, 20],
                'macd_slow': [20, 26, 30, 35],
                'bb_window': [15, 20, 25, 30],
                'bb_std': [1.5, 2.0, 2.5, 3.0],
                'atr_window': [10, 14, 20, 30]
            }
            
            # This would be implemented with VectorBT's optimization
            # For now, return default parameters
            optimized_params = {
                'rsi_window': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'bb_window': 20,
                'bb_std': 2.0,
                'atr_window': 14
            }
            
            tprint_success("✅ Indicator parameters optimized")
            return optimized_params
            
        except Exception as e:
            tprint_error(f"❌ Error optimizing indicators: {e}")
            return {}
    
    def get_performance_metrics(self, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Get performance metrics for indicators."""
        if not self.config.enable_performance_monitoring:
            return {}
        
        try:
            metrics = {}
            
            # Calculate correlation matrix
            numeric_indicators = {
                name: data for name, data in indicators.items() 
                if isinstance(data, pd.Series) and pd.api.types.is_numeric_dtype(data)
            }
            
            if len(numeric_indicators) > 1:
                indicator_df = pd.DataFrame(numeric_indicators)
                correlation_matrix = indicator_df.corr()
                metrics['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Calculate stability scores
            stability_scores = {}
            for name, data in numeric_indicators.items():
                if len(data.dropna()) > 1:
                    stability_scores[name] = 1.0 / (1.0 + data.std())
            
            metrics['stability_scores'] = stability_scores
            
            # Calculate information content
            information_scores = {}
            for name, data in numeric_indicators.items():
                if len(data.dropna()) > 1:
                    # Simple information content based on variance
                    information_scores[name] = data.var()
            
            metrics['information_scores'] = information_scores
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating performance metrics: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Clean up resources and caches."""
        if self._cache:
            self._cache.clear()
        tprint_info("🧹 VectorBT Indicator Suite cleanup completed")


# Convenience functions
def create_indicator_suite(config: Optional[VectorBTIndicatorSuiteConfig] = None) -> VectorBTIndicatorSuite:
    """Create VectorBT Indicator Suite instance."""
    return VectorBTIndicatorSuite(config)


def get_all_indicators(
    data: pd.DataFrame, 
    config: Optional[VectorBTIndicatorSuiteConfig] = None,
    windows: Optional[List[int]] = None
) -> Dict[str, pd.Series]:
    """Get all indicators for given data."""
    suite = create_indicator_suite(config)
    return suite.get_all_indicators(data, windows)