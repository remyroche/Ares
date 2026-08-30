"""
VectorBT Technical Indicators

Enhanced technical indicators using VectorBT for improved performance and functionality.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from functools import lru_cache

from .vectorbt_base import VectorBTBase, VectorBTError
from .vectorbt_config import IndicatorConfig

logger = logging.getLogger(__name__)

class VectorBTIndicators(VectorBTBase):
    """
    VectorBT Technical Indicators
    
    Provides high-performance technical indicators using VectorBT.
    """
    
    def __init__(self, config: IndicatorConfig, base_config: Optional[VectorBTConfig] = None):
        """Initialize VectorBT indicators."""
        if base_config is None:
            from .vectorbt_config import VectorBTConfig
            base_config = VectorBTConfig()
        
        super().__init__(base_config)
        self.indicator_config = config
        self._indicator_cache = {}
        
        self.logger.info("VectorBT Indicators initialized")
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all configured technical indicators."""
        start_time = time.time()
        
        try:
            # Validate data
            data = self.validate_data(data)
            
            # Create result dataframe
            result = data.copy()
            
            # Calculate moving averages
            if self.indicator_config.sma_periods:
                result = self._calculate_sma(result)
            
            if self.indicator_config.ema_periods:
                result = self._calculate_ema(result)
            
            # Calculate oscillators
            result = self._calculate_rsi(result)
            result = self._calculate_stochastic(result)
            result = self._calculate_williams_r(result)
            
            # Calculate volatility indicators
            result = self._calculate_bollinger_bands(result)
            result = self._calculate_atr(result)
            
            # Calculate momentum indicators
            result = self._calculate_macd(result)
            
            # Calculate volume indicators
            if self.indicator_config.obv_enabled:
                result = self._calculate_obv(result)
            
            if self.indicator_config.ad_line_enabled:
                result = self._calculate_ad_line(result)
            
            # Calculate custom indicators
            for name, config in self.indicator_config.custom_indicators.items():
                result = self._calculate_custom_indicator(result, name, config)
            
            duration = time.time() - start_time
            self.log_performance("calculate_all_indicators", duration)
            
            self.logger.info(f"All indicators calculated in {duration:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate indicators: {e}")
            raise VectorBTError(f"Indicator calculation failed: {e}")
    
    def _calculate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        try:
            for period in self.indicator_config.sma_periods:
                if len(data) >= period:
                    sma = vbt.MA.run(data['close'], window=period).ma
                    data[f'sma_{period}'] = sma
                    self.logger.debug(f"SMA {period} calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"SMA calculation failed: {e}")
            return data
    
    def _calculate_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        try:
            for period in self.indicator_config.ema_periods:
                if len(data) >= period:
                    ema = vbt.MA.run(data['close'], window=period, ewm=True).ma
                    data[f'ema_{period}'] = ema
                    self.logger.debug(f"EMA {period} calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"EMA calculation failed: {e}")
            return data
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        try:
            if len(data) >= self.indicator_config.rsi_period:
                rsi = vbt.RSI.run(data['close'], window=self.indicator_config.rsi_period).rsi
                data['rsi'] = rsi
                self.logger.debug(f"RSI calculated with period {self.indicator_config.rsi_period}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}")
            return data
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        try:
            k_period, d_period, smooth_k = self.indicator_config.stoch_periods
            
            if len(data) >= k_period:
                stoch = vbt.STOCH.run(
                    data['high'], 
                    data['low'], 
                    data['close'],
                    k_window=k_period,
                    d_window=d_period,
                    smooth_k=smooth_k
                )
                
                data['stoch_k'] = stoch.stoch_k
                data['stoch_d'] = stoch.stoch_d
                self.logger.debug(f"Stochastic calculated with periods {k_period}, {d_period}, {smooth_k}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Stochastic calculation failed: {e}")
            return data
    
    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R."""
        try:
            if len(data) >= self.indicator_config.williams_r_period:
                williams_r = vbt.WILLIAMS_R.run(
                    data['high'],
                    data['low'],
                    data['close'],
                    window=self.indicator_config.williams_r_period
                ).williams_r
                
                data['williams_r'] = williams_r
                self.logger.debug(f"Williams %R calculated with period {self.indicator_config.williams_r_period}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Williams %R calculation failed: {e}")
            return data
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            if len(data) >= self.indicator_config.bb_period:
                bb = vbt.BBANDS.run(
                    data['close'],
                    window=self.indicator_config.bb_period,
                    alpha=self.indicator_config.bb_std
                )
                
                data['bb_upper'] = bb.upper
                data['bb_middle'] = bb.middle
                data['bb_lower'] = bb.lower
                data['bb_width'] = bb.width
                data['bb_percent'] = bb.percent
                
                self.logger.debug(f"Bollinger Bands calculated with period {self.indicator_config.bb_period}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation failed: {e}")
            return data
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range."""
        try:
            if len(data) >= self.indicator_config.atr_period:
                atr = vbt.ATR.run(
                    data['high'],
                    data['low'],
                    data['close'],
                    window=self.indicator_config.atr_period
                ).atr
                
                data['atr'] = atr
                self.logger.debug(f"ATR calculated with period {self.indicator_config.atr_period}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {e}")
            return data
    
    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        try:
            if len(data) >= self.indicator_config.macd_slow:
                macd = vbt.MACD.run(
                    data['close'],
                    fast_window=self.indicator_config.macd_fast,
                    slow_window=self.indicator_config.macd_slow,
                    signal_window=self.indicator_config.macd_signal
                )
                
                data['macd'] = macd.macd
                data['macd_signal'] = macd.signal
                data['macd_histogram'] = macd.histogram
                
                self.logger.debug("MACD calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {e}")
            return data
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        try:
            obv = vbt.OBV.run(data['close'], data['volume']).obv
            data['obv'] = obv
            self.logger.debug("OBV calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"OBV calculation failed: {e}")
            return data
    
    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Accumulation/Distribution Line."""
        try:
            ad_line = vbt.AD.run(
                data['high'],
                data['low'],
                data['close'],
                data['volume']
            ).ad
            
            data['ad_line'] = ad_line
            self.logger.debug("A/D Line calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"A/D Line calculation failed: {e}")
            return data
    
    def _calculate_custom_indicator(self, data: pd.DataFrame, name: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate custom indicator."""
        try:
            indicator_type = config.get('type', 'unknown')
            
            if indicator_type == 'sma':
                period = config.get('period', 20)
                if len(data) >= period:
                    sma = vbt.MA.run(data['close'], window=period).ma
                    data[f'custom_{name}'] = sma
            
            elif indicator_type == 'rsi':
                period = config.get('period', 14)
                if len(data) >= period:
                    rsi = vbt.RSI.run(data['close'], window=period).rsi
                    data[f'custom_{name}'] = rsi
            
            # Add more custom indicator types as needed
            
            self.logger.debug(f"Custom indicator '{name}' calculated")
            return data
            
        except Exception as e:
            self.logger.error(f"Custom indicator '{name}' calculation failed: {e}")
            return data
    
    @lru_cache(maxsize=1000)
    def _cached_indicator(self, indicator_name: str, period: int, data_hash: str) -> Optional[pd.Series]:
        """Cached indicator calculation."""
        # This is a placeholder for caching mechanism
        # In practice, you'd implement proper caching here
        return None
    
    def generate_signals(self, data: pd.DataFrame, 
                        strategy: str = 'rsi_mean_reversion') -> Tuple[pd.Series, pd.Series]:
        """Generate trading signals based on technical indicators."""
        try:
            entries = pd.Series(False, index=data.index)
            exits = pd.Series(False, index=data.index)
            
            if strategy == 'rsi_mean_reversion':
                if 'rsi' in data.columns:
                    # RSI oversold/overbought signals
                    entries = data['rsi'] < 30
                    exits = data['rsi'] > 70
            
            elif strategy == 'bollinger_bands':
                if all(col in data.columns for col in ['bb_upper', 'bb_lower']):
                    # Bollinger Bands mean reversion
                    entries = data['close'] < data['bb_lower']
                    exits = data['close'] > data['bb_upper']
            
            elif strategy == 'macd_crossover':
                if all(col in data.columns for col in ['macd', 'macd_signal']):
                    # MACD crossover signals
                    macd_above_signal = data['macd'] > data['macd_signal']
                    entries = macd_above_signal & ~macd_above_signal.shift(1)
                    exits = ~macd_above_signal & macd_above_signal.shift(1)
            
            elif strategy == 'moving_average_crossover':
                if all(col in data.columns for col in ['sma_20', 'sma_50']):
                    # Moving average crossover
                    ma_above = data['sma_20'] > data['sma_50']
                    entries = ma_above & ~ma_above.shift(1)
                    exits = ~ma_above & ma_above.shift(1)
            
            else:
                self.logger.warning(f"Unknown strategy: {strategy}")
            
            self.logger.info(f"Signals generated: {entries.sum()} entries, {exits.sum()} exits")
            return entries, exits
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return pd.Series(False, index=data.index), pd.Series(False, index=data.index)
    
    def get_indicator_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of calculated indicators."""
        try:
            summary = {
                'total_indicators': 0,
                'indicator_types': {},
                'data_quality': {},
                'performance_metrics': {}
            }
            
            # Count indicators by type
            indicator_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            summary['total_indicators'] = len(indicator_columns)
            
            for col in indicator_columns:
                if col.startswith('sma_'):
                    summary['indicator_types']['sma'] = summary['indicator_types'].get('sma', 0) + 1
                elif col.startswith('ema_'):
                    summary['indicator_types']['ema'] = summary['indicator_types'].get('ema', 0) + 1
                elif col == 'rsi':
                    summary['indicator_types']['rsi'] = 1
                elif col.startswith('bb_'):
                    summary['indicator_types']['bollinger_bands'] = summary['indicator_types'].get('bollinger_bands', 0) + 1
                # Add more indicator types as needed
            
            # Data quality metrics
            for col in indicator_columns:
                if col in data.columns:
                    summary['data_quality'][col] = {
                        'null_count': data[col].isnull().sum(),
                        'null_percentage': (data[col].isnull().sum() / len(data)) * 100,
                        'mean': data[col].mean() if not data[col].isnull().all() else None,
                        'std': data[col].std() if not data[col].isnull().all() else None
                    }
            
            # Performance metrics
            summary['performance_metrics'] = self.get_performance_stats()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate indicator summary: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear indicator cache."""
        self._indicator_cache.clear()
        self.logger.info("Indicator cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            'cache_size': len(self._indicator_cache),
            'cache_keys': list(self._indicator_cache.keys())
        }