"""
Family-Diverse Feature Generation

This module generates features from different families (momentum, volume, trend, 
oscillators, etc.) to ensure diversity in feature selection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.signal import find_peaks
import warnings

logger = logging.getLogger(__name__)

class FamilyDiverseFeatureGenerator:
    """
    Generates features from different families to ensure diversity.
    """
    
    def __init__(self, enable_matrix_ops: bool = True):
        """
        Initialize family-diverse feature generator.
        
        Args:
            enable_matrix_ops: Whether to enable matrix operations
        """
        self.enable_matrix_ops = enable_matrix_ops
        
        # Initialize matrix operations if available
        if enable_matrix_ops:
            try:
                from src.utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations(enable_gpu=True, enable_parallel=True)
                self.matrix_available = True
            except ImportError:
                self.matrix_ops = None
                self.matrix_available = False
                logger.warning("Matrix operations not available, using standard operations")
        else:
            self.matrix_ops = None
            self.matrix_available = False
    
    def generate_family_diverse_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate features from different families.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with features by family
        """
        logger.info("Generating family-diverse features...")
        
        families = {}
        
        # 1. Returns Family
        families['returns'] = self._generate_returns_family(data)
        
        # 2. Momentum Family
        families['momentum'] = self._generate_momentum_family(data)
        
        # 3. Volume Family
        families['volume'] = self._generate_volume_family(data)
        
        # 4. Trend Family
        families['trend'] = self._generate_trend_family(data)
        
        # 5. Oscillators Family
        families['oscillators'] = self._generate_oscillators_family(data)
        
        # 6. Volatility Family
        families['volatility'] = self._generate_volatility_family(data)
        
        # 7. VWAP Family
        families['vwap'] = self._generate_vwap_family(data)
        
        # 8. Technical Indicators Family
        families['technical'] = self._generate_technical_family(data)
        
        # 9. Statistical Family
        families['statistical'] = self._generate_statistical_family(data)
        
        # Note: Cross-asset family removed as requested
        
        logger.info(f"Generated features for {len(families)} families")
        return families
    
    def _generate_returns_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate returns-based features."""
        df = data.copy()
        
        # Basic returns
        df['ret_t1'] = data['close'].pct_change()
        df['ret_t5'] = data['close'].pct_change(5)
        df['ret_t10'] = data['close'].pct_change(10)
        df['ret_t20'] = data['close'].pct_change(20)
        
        # Log returns
        df['log_ret_t1'] = np.log(data['close'] / data['close'].shift(1))
        df['log_ret_t5'] = np.log(data['close'] / data['close'].shift(5))
        
        # Absolute returns
        df['abs_ret_t1'] = df['ret_t1'].abs()
        df['abs_ret_t5'] = df['ret_t5'].abs()
        
        # Squared returns
        df['sq_ret_t1'] = df['ret_t1'] ** 2
        df['sq_ret_t5'] = df['ret_t5'] ** 2
        
        # Signed returns
        df['signed_ret_t1'] = np.sign(df['ret_t1']) * np.sqrt(df['abs_ret_t1'])
        df['signed_ret_t5'] = np.sign(df['ret_t5']) * np.sqrt(df['abs_ret_t5'])
        
        return df
    
    def _generate_momentum_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based features."""
        df = data.copy()
        
        # Price momentum
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            df[f'momentum_ma_{period}'] = df[f'momentum_{period}'].rolling(period).mean()
        
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = data['close'].pct_change(period) * 100
        
        # Momentum acceleration
        df['momentum_acc_5'] = df['momentum_5'].diff()
        df['momentum_acc_10'] = df['momentum_10'].diff()
        
        # Price position within range
        for period in [10, 20, 50]:
            high_period = data['high'].rolling(period).max()
            low_period = data['low'].rolling(period).min()
            df[f'price_position_{period}'] = (data['close'] - low_period) / (high_period - low_period + 1e-8)
        
        # Momentum divergence
        df['momentum_div_5'] = df['momentum_5'] - df['momentum_5'].rolling(10).mean()
        df['momentum_div_10'] = df['momentum_10'] - df['momentum_10'].rolling(20).mean()
        
        return df
    
    def _generate_volume_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        df = data.copy()
        
        if 'volume' not in data.columns:
            return df
        
        # Volume returns
        df['vol_ret_t1'] = data['volume'].pct_change()
        df['vol_ret_t5'] = data['volume'].pct_change(5)
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'vol_ma_{period}'] = data['volume'].rolling(period).mean()
            df[f'vol_std_{period}'] = data['volume'].rolling(period).std()
        
        # Volume ratios
        df['vol_ratio_5'] = data['volume'] / df['vol_ma_5']
        df['vol_ratio_20'] = data['volume'] / df['vol_ma_20']
        
        # Volume-weighted average price
        df['vwap_5'] = (data['close'] * data['volume']).rolling(5).sum() / data['volume'].rolling(5).sum()
        df['vwap_20'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        
        # Volume-price trend
        df['vpt'] = (data['close'].pct_change() * data['volume']).cumsum()
        
        # On-balance volume
        df['obv'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        
        # Volume momentum
        df['vol_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
        df['vol_momentum_10'] = data['volume'] / data['volume'].shift(10) - 1
        
        return df
    
    def _generate_trend_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features."""
        df = data.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = data['close'].rolling(period).mean()
            df[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        # Trend strength
        for short, long in [(5, 20), (10, 50), (20, 100)]:
            df[f'trend_strength_{short}_{long}'] = (df[f'ma_{short}'] - df[f'ma_{long}']) / df[f'ma_{long}']
        
        # Trend direction
        for period in [5, 10, 20]:
            df[f'trend_up_{period}'] = (df[f'ma_{period}'] > df[f'ma_{period}'].shift(1)).astype(int)
            df[f'trend_down_{period}'] = (df[f'ma_{period}'] < df[f'ma_{period}'].shift(1)).astype(int)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Parabolic SAR (simplified)
        df['psar'] = self._calculate_psar(data)
        
        # Trend channels
        for period in [20, 50]:
            high_ma = data['high'].rolling(period).mean()
            low_ma = data['low'].rolling(period).mean()
            df[f'trend_channel_{period}'] = (data['close'] - low_ma) / (high_ma - low_ma + 1e-8)
        
        return df
    
    def _generate_oscillators_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate oscillator-based features."""
        df = data.copy()
        
        # RSI
        for period in [5, 10, 14, 20]:
            df[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
        
        # Stochastic Oscillator
        for k_period, d_period in [(5, 3), (14, 3), (21, 5)]:
            df[f'stoch_k_{k_period}'] = self._calculate_stochastic_k(data, k_period)
            df[f'stoch_d_{k_period}'] = df[f'stoch_k_{k_period}'].rolling(d_period).mean()
        
        # Williams %R
        for period in [5, 10, 14, 20]:
            df[f'williams_r_{period}'] = self._calculate_williams_r(data, period)
        
        # Commodity Channel Index (CCI)
        for period in [10, 20, 30]:
            df[f'cci_{period}'] = self._calculate_cci(data, period)
        
        # Money Flow Index (MFI)
        for period in [10, 14, 20]:
            df[f'mfi_{period}'] = self._calculate_mfi(data, period)
        
        # Ultimate Oscillator
        df['ultimate_osc'] = self._calculate_ultimate_oscillator(data)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100
        
        return df
    
    def _generate_volatility_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features."""
        df = data.copy()
        
        # Rolling volatility
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            df[f'volatility_annualized_{period}'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        # Parkinson volatility
        for period in [5, 10, 20]:
            df[f'parkinson_vol_{period}'] = self._calculate_parkinson_volatility(data, period)
        
        # Garman-Klass volatility
        for period in [5, 10, 20]:
            df[f'gk_vol_{period}'] = self._calculate_garman_klass_volatility(data, period)
        
        # Volatility of volatility
        for period in [10, 20]:
            df[f'vol_of_vol_{period}'] = df[f'volatility_20'].rolling(period).std()
        
        # Average True Range (ATR)
        for period in [5, 10, 14, 20]:
            df[f'atr_{period}'] = self._calculate_atr(data, period)
        
        # Bollinger Bands
        for period in [10, 20, 50]:
            ma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = ma + (std * 2)
            df[f'bb_lower_{period}'] = ma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
            df[f'bb_position_{period}'] = (data['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)
        
        return df
    
    def _generate_vwap_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP-based features."""
        df = data.copy()
        
        if 'volume' not in data.columns:
            return df
        
        # VWAP calculations
        for period in [5, 10, 20, 50, 100]:
            df[f'vwap_{period}'] = (data['close'] * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
        
        # VWAP deviation
        for period in [20, 50]:
            df[f'vwap_dev_{period}'] = (data['close'] - df[f'vwap_{period}']) / df[f'vwap_{period}']
            df[f'vwap_dev_pct_{period}'] = df[f'vwap_dev_{period}'] * 100
        
        # VWAP bands
        for period in [20, 50]:
            vwap = df[f'vwap_{period}']
            vwap_std = (data['close'] - vwap).rolling(period).std()
            df[f'vwap_upper_{period}'] = vwap + (vwap_std * 2)
            df[f'vwap_lower_{period}'] = vwap - (vwap_std * 2)
            df[f'vwap_band_position_{period}'] = (data['close'] - df[f'vwap_lower_{period}']) / (df[f'vwap_upper_{period}'] - df[f'vwap_lower_{period}'] + 1e-8)
        
        # VWAP momentum
        for period in [5, 10, 20]:
            df[f'vwap_momentum_{period}'] = df[f'vwap_20'].pct_change(period)
        
        return df
    
    def _generate_technical_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        df = data.copy()
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(data)
        df.update(ichimoku)
        
        # Fibonacci retracements
        for period in [20, 50, 100]:
            high = data['high'].rolling(period).max()
            low = data['low'].rolling(period).min()
            df[f'fib_23.6_{period}'] = high - 0.236 * (high - low)
            df[f'fib_38.2_{period}'] = high - 0.382 * (high - low)
            df[f'fib_61.8_{period}'] = high - 0.618 * (high - low)
        
        # Support and resistance
        for period in [20, 50]:
            df[f'support_{period}'] = data['low'].rolling(period).min()
            df[f'resistance_{period}'] = data['high'].rolling(period).max()
            df[f'support_distance_{period}'] = (data['close'] - df[f'support_{period}']) / data['close']
            df[f'resistance_distance_{period}'] = (df[f'resistance_{period}'] - data['close']) / data['close']
        
        # Pivot points
        df['pivot'] = (data['high'] + data['low'] + data['close']) / 3
        df['r1'] = 2 * df['pivot'] - data['low']
        df['s1'] = 2 * df['pivot'] - data['high']
        df['r2'] = df['pivot'] + (data['high'] - data['low'])
        df['s2'] = df['pivot'] - (data['high'] - data['low'])
        
        return df
    
    def _generate_statistical_family(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        df = data.copy()
        
        # Higher moments
        for period in [5, 10, 20, 50]:
            returns = data['close'].pct_change()
            df[f'skewness_{period}'] = returns.rolling(period).skew()
            df[f'kurtosis_{period}'] = returns.rolling(period).kurt()
        
        # Autocorrelation
        for period in [5, 10, 20]:
            returns = data['close'].pct_change()
            df[f'autocorr_{period}'] = returns.rolling(period).apply(lambda x: x.autocorr(lag=1))
        
        # Hurst exponent (simplified)
        for period in [20, 50]:
            df[f'hurst_{period}'] = self._calculate_hurst_exponent(data['close'], period)
        
        # Entropy
        for period in [10, 20]:
            returns = data['close'].pct_change()
            df[f'entropy_{period}'] = returns.rolling(period).apply(lambda x: self._calculate_entropy(x))
        
        # Fractal dimension
        for period in [10, 20]:
            df[f'fractal_dim_{period}'] = self._calculate_fractal_dimension(data['close'], period)
        
        return df
    
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low + 1e-8))
        return k_percent
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        wr = -100 * ((highest_high - data['close']) / (highest_high - lowest_low + 1e-8))
        return wr
    
    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad + 1e-8)
        return cci
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        if 'volume' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
        return mfi
    
    def _calculate_ultimate_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator."""
        tr = np.maximum(data['high'] - data['low'], 
                       np.maximum(abs(data['high'] - data['close'].shift(1)),
                                 abs(data['low'] - data['close'].shift(1))))
        
        bp = data['close'] - np.minimum(data['low'], data['close'].shift(1))
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        return uo
    
    def _calculate_psar(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Parabolic SAR (simplified)."""
        # Simplified PSAR calculation
        psar = data['close'].copy()
        psar.iloc[0] = data['low'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > psar.iloc[i-1]:
                psar.iloc[i] = psar.iloc[i-1] + 0.02 * (data['high'].iloc[i] - psar.iloc[i-1])
            else:
                psar.iloc[i] = psar.iloc[i-1] - 0.02 * (psar.iloc[i-1] - data['low'].iloc[i])
        
        return psar
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Parkinson volatility."""
        return np.sqrt(0.25 * np.log(data['high'] / data['low']) ** 2).rolling(period).mean()
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Garman-Klass volatility."""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        return gk.rolling(period).mean()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return tr.rolling(period).mean()
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud indicators."""
        high_9 = data['high'].rolling(9).max()
        low_9 = data['low'].rolling(9).min()
        high_26 = data['high'].rolling(26).max()
        low_26 = data['low'].rolling(26).min()
        high_52 = data['high'].rolling(52).max()
        low_52 = data['low'].rolling(52).min()
        
        return {
            'tenkan_sen': (high_9 + low_9) / 2,
            'kijun_sen': (high_26 + low_26) / 2,
            'senkou_span_a': ((high_9 + low_9) / 2 + (high_26 + low_26) / 2) / 2,
            'senkou_span_b': (high_52 + low_52) / 2,
            'chikou_span': data['close'].shift(-26)
        }
    
    def _calculate_hurst_exponent(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Hurst exponent (simplified)."""
        def hurst(x):
            if len(x) < 2:
                return 0.5
            try:
                lags = range(2, min(20, len(x)))
                tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        return prices.rolling(period).apply(hurst)
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a series."""
        try:
            if len(series) < 2:
                return 0.0
            hist, _ = np.histogram(series.dropna(), bins=min(10, len(series)))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        except:
            return 0.0
    
    def _calculate_fractal_dimension(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate fractal dimension (simplified)."""
        def fractal_dim(x):
            if len(x) < 3:
                return 1.0
            try:
                n = len(x)
                L = np.sum(np.abs(np.diff(x)))
                return 1 + np.log(L) / np.log(n - 1)
            except:
                return 1.0
        
        return prices.rolling(period).apply(fractal_dim)
    
    def get_family_summary(self, families: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary of generated families."""
        summary = {}
        
        for family_name, family_df in families.items():
            # Count features (exclude original OHLCV columns)
            original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            feature_cols = [col for col in family_df.columns if col not in original_cols]
            
            summary[family_name] = {
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'n_samples': len(family_df),
                'memory_usage_mb': family_df.memory_usage(deep=True).sum() / 1024**2
            }
        
        return summary