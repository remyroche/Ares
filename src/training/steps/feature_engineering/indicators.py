"""
Centralized Technical Indicators Module

This module provides centralized access to all technical indicators (RSI, MACD, etc.)
through the feature bank system. All other modules should import indicators from here
rather than calculating them directly.

This ensures consistency and avoids code duplication across the codebase.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass

# Import the feature bank system
try:
    from src.feature_generation.core.feature_bank import get_global_feature_bank, FeatureBank
    from src.feature_generation.core.feature_generator import FeatureCategory
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False
    FeatureBank = None
    FeatureCategory = None

logger = logging.getLogger(__name__)

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stochastic_period: int = 14
    stochastic_smooth_k: int = 3
    stochastic_smooth_d: int = 3
    williams_r_period: int = 14
    cci_period: int = 20
    adx_period: int = 14

class CentralizedIndicators:
    """
    Centralized access to technical indicators through the feature bank system.
    
    This class provides a unified interface for accessing all technical indicators,
    ensuring they are calculated consistently across the entire codebase.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize the centralized indicators system.
        
        Args:
            config: Configuration for indicator parameters
        """
        self.config = config or IndicatorConfig()
        self.feature_bank = None
        self._initialize_feature_bank()
    
    def _initialize_feature_bank(self) -> None:
        """Initialize the feature bank system."""
        if not FEATURE_BANK_AVAILABLE:
            logger.warning("Feature bank not available. Indicators will use fallback calculations.")
            return
        
        try:
            self.feature_bank = get_global_feature_bank()
            logger.info("✅ Centralized indicators initialized with feature bank")
        except Exception as e:
            logger.warning(f"Failed to initialize feature bank: {e}. Using fallback calculations.")
            self.feature_bank = None
    
    def calculate_rsi(self, 
                     data: pd.DataFrame, 
                     period: Optional[int] = None,
                     price_column: str = 'close') -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) indicator.
        
        Args:
            data: DataFrame with OHLCV data
            period: RSI period (defaults to config value)
            price_column: Column name for price data
            
        Returns:
            RSI values as pandas Series
        """
        period = period or self.config.rsi_period
        
        if self.feature_bank and FEATURE_BANK_AVAILABLE:
            try:
                # Use feature bank for RSI calculation
                rsi_features = self.feature_bank.generate_specific_features(
                    data, 
                    features=[f'rsi_{period}']
                )
                if not rsi_features.empty and f'rsi_{period}' in rsi_features.columns:
                    return rsi_features[f'rsi_{period}']
            except Exception as e:
                logger.warning(f"Feature bank RSI calculation failed: {e}. Using fallback.")
        
        # Fallback RSI calculation
        return self._calculate_rsi_fallback(data[price_column], period)
    
    def calculate_macd(self, 
                      data: pd.DataFrame,
                      fast: Optional[int] = None,
                      slow: Optional[int] = None,
                      signal: Optional[int] = None,
                      price_column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period (defaults to config value)
            slow: Slow EMA period (defaults to config value)
            signal: Signal line period (defaults to config value)
            price_column: Column name for price data
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' Series
        """
        fast = fast or self.config.macd_fast
        slow = slow or self.config.macd_slow
        signal = signal or self.config.macd_signal
        
        if self.feature_bank and FEATURE_BANK_AVAILABLE:
            try:
                # Use feature bank for MACD calculation
                macd_features = self.feature_bank.generate_specific_features(
                    data,
                    features=[f'macd_{fast}_{slow}', f'macd_signal_{fast}_{slow}_{signal}', f'macd_histogram_{fast}_{slow}_{signal}']
                )
                if not macd_features.empty:
                    result = {}
                    if f'macd_{fast}_{slow}' in macd_features.columns:
                        result['macd'] = macd_features[f'macd_{fast}_{slow}']
                    if f'macd_signal_{fast}_{slow}_{signal}' in macd_features.columns:
                        result['signal'] = macd_features[f'macd_signal_{fast}_{slow}_{signal}']
                    if f'macd_histogram_{fast}_{slow}_{signal}' in macd_features.columns:
                        result['histogram'] = macd_features[f'macd_histogram_{fast}_{slow}_{signal}']
                    
                    if result:
                        return result
            except Exception as e:
                logger.warning(f"Feature bank MACD calculation failed: {e}. Using fallback.")
        
        # Fallback MACD calculation
        return self._calculate_macd_fallback(data[price_column], fast, slow, signal)
    
    def calculate_stochastic(self, 
                           data: pd.DataFrame,
                           period: Optional[int] = None,
                           smooth_k: Optional[int] = None,
                           smooth_d: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator indicator.
        
        Args:
            data: DataFrame with OHLCV data
            period: Stochastic period (defaults to config value)
            smooth_k: K smoothing period (defaults to config value)
            smooth_d: D smoothing period (defaults to config value)
            
        Returns:
            Dictionary with 'k' and 'd' Series
        """
        period = period or self.config.stochastic_period
        smooth_k = smooth_k or self.config.stochastic_smooth_k
        smooth_d = smooth_d or self.config.stochastic_smooth_d
        
        if self.feature_bank and FEATURE_BANK_AVAILABLE:
            try:
                # Use feature bank for Stochastic calculation
                stoch_features = self.feature_bank.generate_specific_features(
                    data,
                    features=[f'stochastic_k_{period}_{smooth_k}', f'stochastic_d_{period}_{smooth_k}_{smooth_d}']
                )
                if not stoch_features.empty:
                    result = {}
                    if f'stochastic_k_{period}_{smooth_k}' in stoch_features.columns:
                        result['k'] = stoch_features[f'stochastic_k_{period}_{smooth_k}']
                    if f'stochastic_d_{period}_{smooth_k}_{smooth_d}' in stoch_features.columns:
                        result['d'] = stoch_features[f'stochastic_d_{period}_{smooth_k}_{smooth_d}']
                    
                    if result:
                        return result
            except Exception as e:
                logger.warning(f"Feature bank Stochastic calculation failed: {e}. Using fallback.")
        
        # Fallback Stochastic calculation
        return self._calculate_stochastic_fallback(data, period, smooth_k, smooth_d)
    
    def calculate_williams_r(self, 
                           data: pd.DataFrame,
                           period: Optional[int] = None) -> pd.Series:
        """
        Calculate Williams %R indicator.
        
        Args:
            data: DataFrame with OHLCV data
            period: Williams %R period (defaults to config value)
            
        Returns:
            Williams %R values as pandas Series
        """
        period = period or self.config.williams_r_period
        
        if self.feature_bank and FEATURE_BANK_AVAILABLE:
            try:
                # Use feature bank for Williams %R calculation
                williams_features = self.feature_bank.generate_specific_features(
                    data,
                    features=[f'williams_r_{period}']
                )
                if not williams_features.empty and f'williams_r_{period}' in williams_features.columns:
                    return williams_features[f'williams_r_{period}']
            except Exception as e:
                logger.warning(f"Feature bank Williams %R calculation failed: {e}. Using fallback.")
        
        # Fallback Williams %R calculation
        return self._calculate_williams_r_fallback(data, period)
    
    def calculate_cci(self, 
                     data: pd.DataFrame,
                     period: Optional[int] = None) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI) indicator.
        
        Args:
            data: DataFrame with OHLCV data
            period: CCI period (defaults to config value)
            
        Returns:
            CCI values as pandas Series
        """
        period = period or self.config.cci_period
        
        if self.feature_bank and FEATURE_BANK_AVAILABLE:
            try:
                # Use feature bank for CCI calculation
                cci_features = self.feature_bank.generate_specific_features(
                    data,
                    features=[f'cci_{period}']
                )
                if not cci_features.empty and f'cci_{period}' in cci_features.columns:
                    return cci_features[f'cci_{period}']
            except Exception as e:
                logger.warning(f"Feature bank CCI calculation failed: {e}. Using fallback.")
        
        # Fallback CCI calculation
        return self._calculate_cci_fallback(data, period)
    
    def calculate_adx(self, 
                     data: pd.DataFrame,
                     period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) indicator.
        
        Args:
            data: DataFrame with OHLCV data
            period: ADX period (defaults to config value)
            
        Returns:
            ADX values as pandas Series
        """
        period = period or self.config.adx_period
        
        if self.feature_bank and FEATURE_BANK_AVAILABLE:
            try:
                # Use feature bank for ADX calculation
                adx_features = self.feature_bank.generate_specific_features(
                    data,
                    features=[f'adx_{period}']
                )
                if not adx_features.empty and f'adx_{period}' in adx_features.columns:
                    return adx_features[f'adx_{period}']
            except Exception as e:
                logger.warning(f"Feature bank ADX calculation failed: {e}. Using fallback.")
        
        # Fallback ADX calculation
        return self._calculate_adx_fallback(data, period)
    
    def get_all_indicators(self, 
                          data: pd.DataFrame,
                          indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get all available indicators for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            indicators: List of indicator names to calculate (None for all)
            
        Returns:
            DataFrame with all indicator values
        """
        if indicators is None:
            indicators = ['rsi', 'macd', 'stochastic', 'williams_r', 'cci', 'adx']
        
        results = {}
        
        for indicator in indicators:
            try:
                if indicator == 'rsi':
                    results[f'rsi_{self.config.rsi_period}'] = self.calculate_rsi(data)
                elif indicator == 'macd':
                    macd_data = self.calculate_macd(data)
                    results.update(macd_data)
                elif indicator == 'stochastic':
                    stoch_data = self.calculate_stochastic(data)
                    results.update(stoch_data)
                elif indicator == 'williams_r':
                    results[f'williams_r_{self.config.williams_r_period}'] = self.calculate_williams_r(data)
                elif indicator == 'cci':
                    results[f'cci_{self.config.cci_period}'] = self.calculate_cci(data)
                elif indicator == 'adx':
                    results[f'adx_{self.config.adx_period}'] = self.calculate_adx(data)
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")
        
        return pd.DataFrame(results, index=data.index)
    
    # Fallback calculation methods
    def _calculate_rsi_fallback(self, prices: pd.Series, period: int) -> pd.Series:
        """Fallback RSI calculation."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def _calculate_macd_fallback(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """Fallback MACD calculation."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_stochastic_fallback(self, data: pd.DataFrame, period: int, smooth_k: int, smooth_d: int) -> Dict[str, pd.Series]:
        """Fallback Stochastic calculation."""
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        k_smooth = k.rolling(window=smooth_k).mean()
        d_smooth = k_smooth.rolling(window=smooth_d).mean()
        
        return {
            'k': k_smooth,
            'd': d_smooth
        }
    
    def _calculate_williams_r_fallback(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Fallback Williams %R calculation."""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - data['close']) / (high_max - low_min))
        return williams_r.fillna(-50)  # Fill NaN with neutral value
    
    def _calculate_cci_fallback(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Fallback CCI calculation."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci.fillna(0)  # Fill NaN with neutral value
    
    def _calculate_adx_fallback(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Fallback ADX calculation."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate smoothed values
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(25)  # Fill NaN with neutral value

# Global instance for easy access
_global_indicators: Optional[CentralizedIndicators] = None

def get_centralized_indicators(config: Optional[IndicatorConfig] = None) -> CentralizedIndicators:
    """
    Get the global centralized indicators instance.
    
    Args:
        config: Optional configuration for indicators
        
    Returns:
        CentralizedIndicators instance
    """
    global _global_indicators
    
    if _global_indicators is None:
        _global_indicators = CentralizedIndicators(config)
    
    return _global_indicators

def calculate_rsi(data: pd.DataFrame, period: int = 14, price_column: str = 'close') -> pd.Series:
    """
    Convenience function to calculate RSI.
    
    Args:
        data: DataFrame with OHLCV data
        period: RSI period
        price_column: Column name for price data
        
    Returns:
        RSI values as pandas Series
    """
    indicators = get_centralized_indicators()
    return indicators.calculate_rsi(data, period, price_column)

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_column: str = 'close') -> Dict[str, pd.Series]:
    """
    Convenience function to calculate MACD.
    
    Args:
        data: DataFrame with OHLCV data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        price_column: Column name for price data
        
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' Series
    """
    indicators = get_centralized_indicators()
    return indicators.calculate_macd(data, fast, slow, signal, price_column)

def calculate_stochastic(data: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, pd.Series]:
    """
    Convenience function to calculate Stochastic Oscillator.
    
    Args:
        data: DataFrame with OHLCV data
        period: Stochastic period
        smooth_k: K smoothing period
        smooth_d: D smoothing period
        
    Returns:
        Dictionary with 'k' and 'd' Series
    """
    indicators = get_centralized_indicators()
    return indicators.calculate_stochastic(data, period, smooth_k, smooth_d)

def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Convenience function to calculate Williams %R.
    
    Args:
        data: DataFrame with OHLCV data
        period: Williams %R period
        
    Returns:
        Williams %R values as pandas Series
    """
    indicators = get_centralized_indicators()
    return indicators.calculate_williams_r(data, period)

def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Convenience function to calculate CCI.
    
    Args:
        data: DataFrame with OHLCV data
        period: CCI period
        
    Returns:
        CCI values as pandas Series
    """
    indicators = get_centralized_indicators()
    return indicators.calculate_cci(data, period)

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Convenience function to calculate ADX.
    
    Args:
        data: DataFrame with OHLCV data
        period: ADX period
        
    Returns:
        ADX values as pandas Series
    """
    indicators = get_centralized_indicators()
    return indicators.calculate_adx(data, period)

def get_all_indicators(data: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to get all indicators.
    
    Args:
        data: DataFrame with OHLCV data
        indicators: List of indicator names to calculate (None for all)
        
    Returns:
        DataFrame with all indicator values
    """
    indicators_obj = get_centralized_indicators()
    return indicators_obj.get_all_indicators(data, indicators)

# Export the main classes and functions
__all__ = [
    'CentralizedIndicators',
    'IndicatorConfig',
    'get_centralized_indicators',
    'calculate_rsi',
    'calculate_macd',
    'calculate_stochastic',
    'calculate_williams_r',
    'calculate_cci',
    'calculate_adx',
    'get_all_indicators'
]