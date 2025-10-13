"""
Feature Registry for End-to-End Roadmap System

Defines parent features with exact formulas and metadata.
Organized by family: price/returns, volatility, mean-reversion, liquidity/micro, anchors/TOD, context.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class FeatureFamily(Enum):
    """Feature families for organization."""
    PRICE_RETURNS = "price_returns"
    VOLATILITY = "volatility"
    MEAN_REVERSION = "mean_reversion"
    TREND = "trend"
    VOLUME = "volume"
    CONTEXT = "context"


@dataclass
class FeatureMetadata:
    """Metadata for each feature."""
    fields_required: List[str]
    lookback_bars: int
    compute_cost_ms_p95: float
    causal: bool
    family: FeatureFamily
    description: str
    formula: str


class ParentFeature(ABC):
    """Abstract base class for parent features."""
    
    def __init__(self, name: str, metadata: FeatureMetadata):
        self.name = name
        self.metadata = metadata
        tprint_debug(f"🔧 Initialized feature: {name} ({metadata.family.value})")
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute the feature from market data."""
        raise NotImplementedError("Subclasses must implement compute method")
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has required fields."""
        missing_fields = set(self.metadata.fields_required) - set(data.columns)
        if missing_fields:
            raise ValueError(f"Missing required fields for {self.name}: {missing_fields}")
        return True


class PriceReturnsFeatures:
    """Price and returns features (10 total)."""
    
    @staticmethod
    def r1(data: pd.DataFrame) -> pd.Series:
        """1-bar return: log(Ct/Ct-1)"""
        return np.log(data['close'] / data['close'].shift(1))
    
    @staticmethod
    def r3(data: pd.DataFrame) -> pd.Series:
        """3-bar return: log(Ct/Ct-3)"""
        return np.log(data['close'] / data['close'].shift(3))
    
    @staticmethod
    def r5(data: pd.DataFrame) -> pd.Series:
        """5-bar return: log(Ct/Ct-5)"""
        return np.log(data['close'] / data['close'].shift(5))
    
    @staticmethod
    def r10(data: pd.DataFrame) -> pd.Series:
        """10-bar return: log(Ct/Ct-10)"""
        return np.log(data['close'] / data['close'].shift(10))
    
    @staticmethod
    def mom5(data: pd.DataFrame) -> pd.Series:
        """5-bar momentum: (Ct/Ct-5) - 1"""
        return (data['close'] / data['close'].shift(5)) - 1
    
    @staticmethod
    def mom10(data: pd.DataFrame) -> pd.Series:
        """10-bar momentum: (Ct/Ct-10) - 1"""
        return (data['close'] / data['close'].shift(10)) - 1
    
    @staticmethod
    def mom20(data: pd.DataFrame) -> pd.Series:
        """20-bar momentum: (Ct/Ct-20) - 1"""
        return (data['close'] / data['close'].shift(20)) - 1
    
    @staticmethod
    def price_ema10_pct(data: pd.DataFrame) -> pd.Series:
        """Price vs EMA10 percentage: (Ct - EMA10) / EMA10"""
        ema10 = data['close'].ewm(span=10).mean()
        return (data['close'] - ema10) / ema10
    
    @staticmethod
    def price_ema20_pct(data: pd.DataFrame) -> pd.Series:
        """Price vs EMA20 percentage: (Ct - EMA20) / EMA20"""
        ema20 = data['close'].ewm(span=20).mean()
        return (data['close'] - ema20) / ema20
    
    @staticmethod
    def bollz20(data: pd.DataFrame) -> pd.Series:
        """Bollinger z-score: (Ct - MA20) / SD20(C)"""
        ma20 = data['close'].rolling(20).mean()
        sd20 = data['close'].rolling(20).std()
        return (data['close'] - ma20) / sd20


class VolatilityFeatures:
    """Volatility features (6 total)."""
    
    @staticmethod
    def sigma_ew(data: pd.DataFrame, halflife: int = 12) -> pd.Series:
        """EW standard deviation of r1 with specified halflife."""
        r1 = np.log(data['close'] / data['close'].shift(1))
        return r1.ewm(halflife=halflife).std()
    
    @staticmethod
    def gk_w(data: pd.DataFrame, window: int = 12) -> pd.Series:
        """Garman-Klass estimator over specified window."""
        # GK = 0.5 * (log(H/L))^2 - (2*log(2)-1) * (log(C/O))^2
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window).mean())
    
    @staticmethod
    def rv_bipower_12(data: pd.DataFrame) -> pd.Series:
        """Bipower variation over 12 bars."""
        r1 = np.log(data['close'] / data['close'].shift(1))
        r1_abs = np.abs(r1)
        bipower = r1_abs * r1_abs.shift(1)
        return np.sqrt(bipower.rolling(12).mean())
    
    @staticmethod
    def rv_short_3(data: pd.DataFrame) -> pd.Series:
        """Short-term realized volatility over 3 bars."""
        r1 = np.log(data['close'] / data['close'].shift(1))
        return np.sqrt((r1**2).rolling(3).sum())
    
    @staticmethod
    def sigma_slope_6(data: pd.DataFrame) -> pd.Series:
        """Volatility slope: (σew - σew,-6) / (σew,-6 + ε)"""
        sigma_ew = VolatilityFeatures.sigma_ew(data)
        sigma_ew_lag6 = sigma_ew.shift(6)
        epsilon = 1e-8
        return (sigma_ew - sigma_ew_lag6) / (sigma_ew_lag6 + epsilon)
    
    @staticmethod
    def range_pct(data: pd.DataFrame) -> pd.Series:
        """Range percentage: (Ht - Lt) / (Ct + ε)"""
        epsilon = 1e-8
        return (data['high'] - data['low']) / (data['close'] + epsilon)


class MeanReversionFeatures:
    """Mean reversion features (4 total)."""
    
    @staticmethod
    def rsi7(data: pd.DataFrame) -> pd.Series:
        """7-period RSI (Wilder)."""
        return MeanReversionFeatures._rsi(data, 7)
    
    @staticmethod
    def rsi14(data: pd.DataFrame) -> pd.Series:
        """14-period RSI (Wilder)."""
        return MeanReversionFeatures._rsi(data, 14)
    
    @staticmethod
    def stochk14(data: pd.DataFrame) -> pd.Series:
        """14-period Stochastic %K."""
        low14 = data['low'].rolling(14).min()
        high14 = data['high'].rolling(14).max()
        return 100 * (data['close'] - low14) / (high14 - low14)
    
    @staticmethod
    def autocorr_r1_w(data: pd.DataFrame, window: int = 12) -> pd.Series:
        """Autocorrelation of r1 over specified window."""
        r1 = np.log(data['close'] / data['close'].shift(1))
        return r1.rolling(window).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    @staticmethod
    def _rsi(data: pd.DataFrame, period: int) -> pd.Series:
        """Helper function to calculate RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class TrendFeatures:
    """Trend and momentum features (8 total)."""
    
    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (ADX)."""
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
        
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Smooth the values
        atr = tr.rolling(period).mean()
        di_plus = 100 * pd.Series(dm_plus, index=data.index).rolling(period).mean() / atr
        di_minus = 100 * pd.Series(dm_minus, index=data.index).rolling(period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        return macd_line
    
    @staticmethod
    def macd_signal(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD Signal Line."""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return signal_line
    
    @staticmethod
    def macd_histogram(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD Histogram."""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    
    @staticmethod
    def ichimoku_base(data: pd.DataFrame, period: int = 26) -> pd.Series:
        """Ichimoku Base Line (Tenkan-sen)."""
        high_9 = data['high'].rolling(9).max()
        low_9 = data['low'].rolling(9).min()
        return (high_9 + low_9) / 2
    
    @staticmethod
    def ichimoku_conversion(data: pd.DataFrame, period: int = 26) -> pd.Series:
        """Ichimoku Conversion Line (Kijun-sen)."""
        high_26 = data['high'].rolling(26).max()
        low_26 = data['low'].rolling(26).min()
        return (high_26 + low_26) / 2
    
    @staticmethod
    def parabolic_sar(data: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
        """Parabolic SAR (Stop and Reverse)."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Initialize arrays
        sar = np.zeros(len(data))
        trend = np.zeros(len(data))
        af = np.zeros(len(data))
        ep = np.zeros(len(data))
        
        # Initial values
        sar[0] = low.iloc[0]
        trend[0] = 1
        af[0] = step
        ep[0] = high.iloc[0]
        
        for i in range(1, len(data)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                if low.iloc[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = step
                    ep[i] = low.iloc[i]
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + step, max_step)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                if high.iloc[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = step
                    ep[i] = high.iloc[i]
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + step, max_step)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(sar, index=data.index)
    
    @staticmethod
    def williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R indicator."""
        highest_high = data['high'].rolling(period).max()
        lowest_low = data['low'].rolling(period).min()
        wr = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        return wr


class VolumeFeatures:
    """Volume and liquidity features (8 total)."""
    
    @staticmethod
    def volume_z18(data: pd.DataFrame) -> pd.Series:
        """Volume z-score over 18 bars."""
        volume_mean = data['volume'].rolling(18).mean()
        volume_std = data['volume'].rolling(18).std()
        return (data['volume'] - volume_mean) / volume_std
    
    @staticmethod
    def volume_ma_ratio(data: pd.DataFrame) -> pd.Series:
        """Volume to MA ratio."""
        volume_ma = data['volume'].rolling(20).mean()
        return data['volume'] / volume_ma
    
    @staticmethod
    def volume_price_trend(data: pd.DataFrame) -> pd.Series:
        """Volume Price Trend (VPT)."""
        r1 = np.log(data['close'] / data['close'].shift(1))
        vpt = (r1 * data['volume']).cumsum()
        return vpt
    
    @staticmethod
    def on_balance_volume(data: pd.DataFrame) -> pd.Series:
        """On Balance Volume (OBV)."""
        price_change = data['close'].diff()
        obv = np.where(price_change > 0, data['volume'], 
                      np.where(price_change < 0, -data['volume'], 0)).cumsum()
        return pd.Series(obv, index=data.index)
    
    @staticmethod
    def money_flow_index(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index (MFI)."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    @staticmethod
    def ease_of_movement(data: pd.DataFrame) -> pd.Series:
        """Ease of Movement indicator."""
        distance_moved = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
        box_height = data['volume'] / (data['high'] - data['low'])
        return distance_moved / box_height
    
    @staticmethod
    def volume_rate_of_change(data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Volume Rate of Change."""
        return data['volume'].pct_change(period) * 100
    
    @staticmethod
    def klinger_volume_oscillator(data: pd.DataFrame, fast: int = 34, slow: int = 55) -> pd.Series:
        """Klinger Volume Oscillator."""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate trend
        trend = np.where(close > close.shift(1), 1, -1)
        
        # Calculate DM and CM
        dm = high - low
        cm = np.where(trend == trend.shift(1), dm + dm.shift(1), dm)
        
        # Calculate VF
        vf = volume * trend * abs(2 * (dm / cm) - 1)
        
        # Calculate KVO
        ema_fast = pd.Series(vf, index=data.index).ewm(span=fast).mean()
        ema_slow = pd.Series(vf, index=data.index).ewm(span=slow).mean()
        
        return ema_fast - ema_slow


class ContextFeatures:
    """Context features (2 total, optional)."""
    
    @staticmethod
    def beta30(data: pd.DataFrame) -> pd.Series:
        """Rolling OLS beta of r1 to index r1 (30 bars, requires index_close)."""
        if 'index_close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        
        r1 = np.log(data['close'] / data['close'].shift(1))
        r1_index = np.log(data['index_close'] / data['index_close'].shift(1))
        
        def calc_beta(window):
            if len(window) < 2:
                return np.nan
            return np.corrcoef(window['r1'], window['r1_index'])[0, 1]
        
        combined = pd.DataFrame({'r1': r1, 'r1_index': r1_index})
        return combined.rolling(30).apply(calc_beta, raw=False)
    
    @staticmethod
    def mkt_dispersion(data: pd.DataFrame) -> pd.Series:
        """Cross-sectional std of r1 (if universe is traded)."""
        # This would need multiple assets - simplified for now
        r1 = np.log(data['close'] / data['close'].shift(1))
        return r1.rolling(20).std()  # Simplified as single-asset std


class FeatureRegistry:
    """Registry of all parent features."""

    def __init__(self):
        self.features = {}
        self._register_features()
        self._compute_fn_map = self._build_compute_function_map()
    
    def _register_features(self):
        """Register all parent features with metadata."""
        
        # Price/Returns features (10)
        price_returns = [
            ('p/r1', FeatureMetadata(['close'], 1, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                   '1-bar return', 'log(Ct/Ct-1)')),
            ('p/r3', FeatureMetadata(['close'], 3, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                   '3-bar return', 'log(Ct/Ct-3)')),
            ('p/r5', FeatureMetadata(['close'], 5, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                   '5-bar return', 'log(Ct/Ct-5)')),
            ('p/r10', FeatureMetadata(['close'], 10, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                    '10-bar return', 'log(Ct/Ct-10)')),
            ('p/mom5', FeatureMetadata(['close'], 5, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                     '5-bar momentum', '(Ct/Ct-5) - 1')),
            ('p/mom10', FeatureMetadata(['close'], 10, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                      '10-bar momentum', '(Ct/Ct-10) - 1')),
            ('p/mom20', FeatureMetadata(['close'], 20, 0.1, True, FeatureFamily.PRICE_RETURNS, 
                                      '20-bar momentum', '(Ct/Ct-20) - 1')),
            ('p/price_ema10_pct', FeatureMetadata(['close'], 10, 0.2, True, FeatureFamily.PRICE_RETURNS, 
                                                 'Price vs EMA10%', '(Ct - EMA10) / EMA10')),
            ('p/price_ema20_pct', FeatureMetadata(['close'], 20, 0.2, True, FeatureFamily.PRICE_RETURNS, 
                                                 'Price vs EMA20%', '(Ct - EMA20) / EMA20')),
            ('p/bollz20', FeatureMetadata(['close'], 20, 0.3, True, FeatureFamily.PRICE_RETURNS, 
                                        'Bollinger z-score', '(Ct - MA20) / SD20(C)'))
        ]
        
        # Volatility features (6)
        volatility = [
            ('p/sigma_ew', FeatureMetadata(['close'], 18, 0.5, True, FeatureFamily.VOLATILITY, 
                                         'EW std of r1', 'EW std with halflife')),
            ('p/gk_w', FeatureMetadata(['open', 'high', 'low', 'close'], 24, 0.4, True, FeatureFamily.VOLATILITY, 
                                     'GK estimator', '0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2')),
            ('p/rv_bipower_12', FeatureMetadata(['close'], 12, 0.3, True, FeatureFamily.VOLATILITY, 
                                              'Bipower variation', 'sqrt(bipower over 12 bars)')),
            ('p/rv_short_3', FeatureMetadata(['close'], 3, 0.2, True, FeatureFamily.VOLATILITY, 
                                           'Short RV', 'sqrt(sum(r^2) over 3 bars)')),
            ('p/sigma_slope_6', FeatureMetadata(['close'], 18, 0.4, True, FeatureFamily.VOLATILITY, 
                                              'Vol slope', '(σew - σew,-6) / (σew,-6 + ε)')),
            ('p/range_pct', FeatureMetadata(['high', 'low', 'close'], 1, 0.1, True, FeatureFamily.VOLATILITY, 
                                          'Range %', '(Ht - Lt) / (Ct + ε)'))
        ]
        
        # Mean reversion features (4)
        mean_reversion = [
            ('p/rsi7', FeatureMetadata(['close'], 7, 0.3, True, FeatureFamily.MEAN_REVERSION, 
                                     '7-period RSI', 'Wilder RSI')),
            ('p/rsi14', FeatureMetadata(['close'], 14, 0.3, True, FeatureFamily.MEAN_REVERSION, 
                                      '14-period RSI', 'Wilder RSI')),
            ('p/stochk14', FeatureMetadata(['high', 'low', 'close'], 14, 0.2, True, FeatureFamily.MEAN_REVERSION, 
                                         'Stochastic %K', '100 * (C - L14) / (H14 - L14)')),
            ('p/autocorr_r1_w', FeatureMetadata(['close'], 12, 0.4, True, FeatureFamily.MEAN_REVERSION, 
                                              'R1 autocorr', 'autocorr(r1, lag=1)'))
        ]
        
        # Trend features (8)
        trend = [
            ('p/adx', FeatureMetadata(['high', 'low', 'close'], 14, 0.5, True, FeatureFamily.TREND, 
                                    'ADX', 'Average Directional Index')),
            ('p/macd', FeatureMetadata(['close'], 26, 0.3, True, FeatureFamily.TREND, 
                                     'MACD', 'Moving Average Convergence Divergence')),
            ('p/macd_signal', FeatureMetadata(['close'], 26, 0.3, True, FeatureFamily.TREND, 
                                            'MACD Signal', 'MACD Signal Line')),
            ('p/macd_histogram', FeatureMetadata(['close'], 26, 0.3, True, FeatureFamily.TREND, 
                                               'MACD Histogram', 'MACD - Signal Line')),
            ('p/ichimoku_base', FeatureMetadata(['high', 'low'], 26, 0.4, True, FeatureFamily.TREND, 
                                              'Ichimoku Base', 'Tenkan-sen (9-period)')),
            ('p/ichimoku_conversion', FeatureMetadata(['high', 'low'], 26, 0.4, True, FeatureFamily.TREND, 
                                                    'Ichimoku Conversion', 'Kijun-sen (26-period)')),
            ('p/parabolic_sar', FeatureMetadata(['high', 'low', 'close'], 1, 0.2, True, FeatureFamily.TREND, 
                                              'Parabolic SAR', 'Stop and Reverse')),
            ('p/williams_r', FeatureMetadata(['high', 'low', 'close'], 14, 0.2, True, FeatureFamily.TREND, 
                                           'Williams %R', 'Williams Percent Range'))
        ]
        
        # Volume features (8)
        volume = [
            ('p/volume_z18', FeatureMetadata(['volume'], 18, 0.2, True, FeatureFamily.VOLUME, 
                                           'Volume z-score', 'z-score over 18 bars')),
            ('p/volume_ma_ratio', FeatureMetadata(['volume'], 20, 0.2, True, FeatureFamily.VOLUME, 
                                                'Volume MA ratio', 'Volume to MA ratio')),
            ('p/volume_price_trend', FeatureMetadata(['close', 'volume'], 1, 0.3, True, FeatureFamily.VOLUME, 
                                                   'Volume Price Trend', 'VPT indicator')),
            ('p/on_balance_volume', FeatureMetadata(['close', 'volume'], 1, 0.2, True, FeatureFamily.VOLUME, 
                                                  'On Balance Volume', 'OBV indicator')),
            ('p/money_flow_index', FeatureMetadata(['high', 'low', 'close', 'volume'], 14, 0.4, True, FeatureFamily.VOLUME, 
                                                 'Money Flow Index', 'MFI indicator')),
            ('p/ease_of_movement', FeatureMetadata(['high', 'low', 'volume'], 1, 0.2, True, FeatureFamily.VOLUME, 
                                                 'Ease of Movement', 'EOM indicator')),
            ('p/volume_rate_of_change', FeatureMetadata(['volume'], 10, 0.2, True, FeatureFamily.VOLUME, 
                                                      'Volume ROC', 'Volume Rate of Change')),
            ('p/klinger_volume_oscillator', FeatureMetadata(['high', 'low', 'close', 'volume'], 55, 0.5, True, FeatureFamily.VOLUME, 
                                                          'Klinger Volume Osc', 'KVO indicator'))
        ]
        
        # Context features (2, optional)
        context = [
            ('p/beta30', FeatureMetadata(['close', 'index_close'], 30, 0.5, True, FeatureFamily.CONTEXT, 
                                       'Rolling beta', 'OLS beta of r1 to index r1')),
            ('p/mkt_dispersion', FeatureMetadata(['close'], 20, 0.3, True, FeatureFamily.CONTEXT, 
                                               'Market dispersion', 'Cross-sec std of r1'))
        ]
        
        # Register all features
        all_features = price_returns + volatility + mean_reversion + trend + volume + context
        
        for name, metadata in all_features:
            self.features[name] = metadata

    def _build_compute_function_map(self) -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
        """Create dispatch map from feature name to compute function."""

        compute_map: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            # Price/Returns
            'p/r1': PriceReturnsFeatures.r1,
            'p/r3': PriceReturnsFeatures.r3,
            'p/r5': PriceReturnsFeatures.r5,
            'p/r10': PriceReturnsFeatures.r10,
            'p/mom5': PriceReturnsFeatures.mom5,
            'p/mom10': PriceReturnsFeatures.mom10,
            'p/mom20': PriceReturnsFeatures.mom20,
            'p/price_ema10_pct': PriceReturnsFeatures.price_ema10_pct,
            'p/price_ema20_pct': PriceReturnsFeatures.price_ema20_pct,
            'p/bollz20': PriceReturnsFeatures.bollz20,
            # Volatility
            'p/sigma_ew': VolatilityFeatures.sigma_ew,
            'p/gk_w': VolatilityFeatures.gk_w,
            'p/rv_bipower_12': VolatilityFeatures.rv_bipower_12,
            'p/rv_short_3': VolatilityFeatures.rv_short_3,
            'p/sigma_slope_6': VolatilityFeatures.sigma_slope_6,
            'p/range_pct': VolatilityFeatures.range_pct,
            # Mean reversion
            'p/rsi7': MeanReversionFeatures.rsi7,
            'p/rsi14': MeanReversionFeatures.rsi14,
            'p/stochk14': MeanReversionFeatures.stochk14,
            'p/autocorr_r1_w': MeanReversionFeatures.autocorr_r1_w,
            # Trend features
            'p/adx': TrendFeatures.adx,
            'p/macd': TrendFeatures.macd,
            'p/macd_signal': TrendFeatures.macd_signal,
            'p/macd_histogram': TrendFeatures.macd_histogram,
            'p/ichimoku_base': TrendFeatures.ichimoku_base,
            'p/ichimoku_conversion': TrendFeatures.ichimoku_conversion,
            'p/parabolic_sar': TrendFeatures.parabolic_sar,
            'p/williams_r': TrendFeatures.williams_r,
            # Volume features
            'p/volume_z18': VolumeFeatures.volume_z18,
            'p/volume_ma_ratio': VolumeFeatures.volume_ma_ratio,
            'p/volume_price_trend': VolumeFeatures.volume_price_trend,
            'p/on_balance_volume': VolumeFeatures.on_balance_volume,
            'p/money_flow_index': VolumeFeatures.money_flow_index,
            'p/ease_of_movement': VolumeFeatures.ease_of_movement,
            'p/volume_rate_of_change': VolumeFeatures.volume_rate_of_change,
            'p/klinger_volume_oscillator': VolumeFeatures.klinger_volume_oscillator,
            # Context
            'p/beta30': ContextFeatures.beta30,
            'p/mkt_dispersion': ContextFeatures.mkt_dispersion,
        }

        missing = set(self.features) - set(compute_map)
        if missing:
            raise ValueError(
                "Missing compute function definitions for features: "
                f"{sorted(missing)}"
            )

        return compute_map
    
    def get_feature_metadata(self, name: str) -> FeatureMetadata:
        """Get metadata for a specific feature."""
        if name not in self.features:
            raise ValueError(f"Feature '{name}' not found in registry")
        return self.features[name]
    
    def get_features_by_family(self, family: FeatureFamily) -> List[str]:
        """Get all features in a specific family."""
        return [name for name, metadata in self.features.items() if metadata.family == family]
    
    def get_all_features(self) -> List[str]:
        """Get all feature names."""
        return list(self.features.keys())

    def get_compute_fn(self, name: str) -> Callable[[pd.DataFrame], pd.Series]:
        """Retrieve the compute function for a feature."""
        if name not in self._compute_fn_map:
            raise ValueError(f"Feature '{name}' not found in registry")
        return self._compute_fn_map[name]

    def compute_feature(self, name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a feature using the registered implementation."""
        metadata = self.get_feature_metadata(name)
        missing_fields = set(metadata.fields_required) - set(data.columns)
        if missing_fields:
            raise ValueError(
                f"Missing required fields for feature '{name}': {sorted(missing_fields)}"
            )

        compute_fn = self.get_compute_fn(name)
        return compute_fn(data)

    def validate_feature_gates(self, feature_name: str, lookback: int) -> bool:
        """Validate feature against gates."""
        metadata = self.get_feature_metadata(feature_name)
        
        # Check lookback ceiling (118 minutes)
        if lookback > 118:
            return False
        
        # Check compute cost (2ms p95)
        if metadata.compute_cost_ms_p95 > 2.0:
            return False
        
        # Check causal requirement
        if not metadata.causal:
            return False

        return True



    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
