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


class FeatureFamily(Enum):
    """Feature families for organization."""
    PRICE_RETURNS = "price_returns"
    VOLATILITY = "volatility"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY_MICRO = "liquidity_micro"
    ANCHORS_TOD = "anchors_tod"
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
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute the feature from market data."""
        pass
    
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


class LiquidityMicroFeatures:
    """Liquidity and microstructure features (6 total, book-optional)."""
    
    @staticmethod
    def volume_z18(data: pd.DataFrame) -> pd.Series:
        """Volume z-score over 18 bars."""
        volume_mean = data['volume'].rolling(18).mean()
        volume_std = data['volume'].rolling(18).std()
        return (data['volume'] - volume_mean) / volume_std
    
    @staticmethod
    def tradecount_z18(data: pd.DataFrame) -> pd.Series:
        """Trade count z-score over 18 bars (requires trade_count)."""
        if 'trade_count' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        tc_mean = data['trade_count'].rolling(18).mean()
        tc_std = data['trade_count'].rolling(18).std()
        return (data['trade_count'] - tc_mean) / tc_std
    
    @staticmethod
    def spread_z18(data: pd.DataFrame) -> pd.Series:
        """Spread z-score over 18 bars (requires bid/ask)."""
        if 'bid' not in data.columns or 'ask' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        spread = (data['ask'] - data['bid']) / data['close']
        spread_mean = spread.rolling(18).mean()
        spread_std = spread.rolling(18).std()
        return (spread - spread_mean) / spread_std
    
    # REMOVED: dollarvol_z18 - not recommended for use
    # Use volume_z18 or create custom dollar volume calculation if needed
    
    @staticmethod
    def ofi_proxy(data: pd.DataFrame) -> pd.Series:
        """Order flow imbalance proxy (requires bid/ask sizes)."""
        if 'bid_size' not in data.columns or 'ask_size' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        # Simple OFI proxy: (bid_size - ask_size) / (bid_size + ask_size)
        total_size = data['bid_size'] + data['ask_size']
        return (data['bid_size'] - data['ask_size']) / total_size
    
    @staticmethod
    def microprice_dev(data: pd.DataFrame) -> pd.Series:
        """Microprice deviation (requires bid/ask)."""
        if 'bid' not in data.columns or 'ask' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        microprice = (data['bid'] + data['ask']) / 2
        return (data['close'] - microprice) / data['close']


class AnchorsTODFeatures:
    """Anchors and time-of-day features (4 total)."""
    
    @staticmethod
    def vwap_session_dist(data: pd.DataFrame) -> pd.Series:
        """Session-causal VWAP distance (resets per session)."""
        # This would need session information - simplified for now
        vwap = (data['high'] + data['low'] + data['close']) / 3
        vwap_session = vwap.rolling(12).mean()  # Simplified session VWAP
        return (data['close'] - vwap_session) / vwap_session
    
    @staticmethod
    def vwap_roll12_dist(data: pd.DataFrame) -> pd.Series:
        """Rolling VWAP 12-bar distance within session."""
        vwap = (data['high'] + data['low'] + data['close']) / 3
        vwap_roll = vwap.rolling(12).mean()
        return (data['close'] - vwap_roll) / vwap_roll
    
    @staticmethod
    def open30(data: pd.DataFrame) -> pd.Series:
        """1 if within first 30 minutes of session."""
        # This would need proper session timing - simplified for now
        return pd.Series(0, index=data.index)  # Placeholder
    
    @staticmethod
    def last30(data: pd.DataFrame) -> pd.Series:
        """1 if within last 30 minutes of session."""
        # This would need proper session timing - simplified for now
        return pd.Series(0, index=data.index)  # Placeholder


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
        
        # Liquidity/Micro features (6, book-optional)
        liquidity_micro = [
            ('p/volume_z18', FeatureMetadata(['volume'], 18, 0.2, True, FeatureFamily.LIQUIDITY_MICRO, 
                                           'Volume z-score', 'z-score over 18 bars')),
            ('p/tradecount_z18', FeatureMetadata(['trade_count'], 18, 0.2, True, FeatureFamily.LIQUIDITY_MICRO, 
                                               'Trade count z-score', 'z-score over 18 bars')),
            ('p/spread_z18', FeatureMetadata(['bid', 'ask', 'close'], 18, 0.2, True, FeatureFamily.LIQUIDITY_MICRO, 
                                           'Spread z-score', 'z-score of (ask-bid)/close')),
            # REMOVED: p/dollarvol_z18 - not recommended, use volume_z18 instead
            ('p/ofi_proxy', FeatureMetadata(['bid_size', 'ask_size'], 1, 0.1, True, FeatureFamily.LIQUIDITY_MICRO, 
                                          'OFI proxy', '(bid_size - ask_size) / (bid_size + ask_size)')),
            ('p/microprice_dev', FeatureMetadata(['bid', 'ask', 'close'], 1, 0.1, True, FeatureFamily.LIQUIDITY_MICRO, 
                                               'Microprice dev', '(C - microprice) / C'))
        ]
        
        # Anchors & TOD features (4)
        anchors_tod = [
            ('p/vwap_session_dist', FeatureMetadata(['high', 'low', 'close'], 12, 0.3, True, FeatureFamily.ANCHORS_TOD, 
                                                   'Session VWAP dist', 'Distance to session VWAP')),
            ('p/vwap_roll12_dist', FeatureMetadata(['high', 'low', 'close'], 12, 0.3, True, FeatureFamily.ANCHORS_TOD, 
                                                 'Rolling VWAP dist', 'Distance to rolling VWAP')),
            ('p/open30', FeatureMetadata([], 0, 0.0, True, FeatureFamily.ANCHORS_TOD, 
                                       'First 30min', '1 if within first 30min of session')),
            ('p/last30', FeatureMetadata([], 0, 0.0, True, FeatureFamily.ANCHORS_TOD, 
                                       'Last 30min', '1 if within last 30min of session'))
        ]
        
        # Context features (2, optional)
        context = [
            ('p/beta30', FeatureMetadata(['close', 'index_close'], 30, 0.5, True, FeatureFamily.CONTEXT, 
                                       'Rolling beta', 'OLS beta of r1 to index r1')),
            ('p/mkt_dispersion', FeatureMetadata(['close'], 20, 0.3, True, FeatureFamily.CONTEXT, 
                                               'Market dispersion', 'Cross-sec std of r1'))
        ]
        
        # Register all features
        all_features = price_returns + volatility + mean_reversion + liquidity_micro + anchors_tod + context
        
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
            # Liquidity / microstructure
            'p/volume_z18': LiquidityMicroFeatures.volume_z18,
            'p/tradecount_z18': LiquidityMicroFeatures.tradecount_z18,
            'p/spread_z18': LiquidityMicroFeatures.spread_z18,
            # REMOVED: p/dollarvol_z18 - not recommended
            'p/ofi_proxy': LiquidityMicroFeatures.ofi_proxy,
            'p/microprice_dev': LiquidityMicroFeatures.microprice_dev,
            # Anchors / TOD
            'p/vwap_session_dist': AnchorsTODFeatures.vwap_session_dist,
            'p/vwap_roll12_dist': AnchorsTODFeatures.vwap_roll12_dist,
            'p/open30': AnchorsTODFeatures.open30,
            'p/last30': AnchorsTODFeatures.last30,
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

        # Check lookback ceiling (120 minutes)
        if lookback > 120:
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
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
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
