"""
Standardized Feature Definitions

This module provides standardized feature definitions with explicit conventions
to avoid ambiguity in feature engineering and comparison.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.signal import find_peaks
import warnings

logger = logging.getLogger(__name__)

class StandardizedFeatureGenerator:
    """
    Standardized feature generator with explicit naming conventions.
    
    Naming Conventions:
    - ret_t(h) = log(P_t / P_{t-h})
    - vwap_t = volume-weighted average over window W (state W in name)
    - vol_t(W) = realized vol proxy (std of returns over W)
    
    Suffixes:
    - _normvolW → divided by vol_t(W)
    - _zcs → cross-sectional z-score at time t
    - _ewmA → EWMA with span A
    - _wW → rolling window W
    - _leadH / _lagH → H-step lead/lag
    """
    
    def __init__(self, data: pd.DataFrame, enable_matrix_ops: bool = True):
        """
        Initialize standardized feature generator.
        
        Args:
            data: Input DataFrame with OHLCV data
            enable_matrix_ops: Whether to enable matrix operations
        """
        self.data = data.copy()
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
    
    def generate_standardized_features(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all standardized feature versions.
        
        Returns:
            Dictionary with standardized feature versions
        """
        logger.info("Generating standardized features...")
        
        versions = {}
        
        # Version 1: Initial features (standardized)
        versions['initial'] = self._create_initial_features_standardized()
        
        # Version 2: VWAP-based features (standardized)
        versions['vwap_based'] = self._create_vwap_features_standardized()
        
        # Version 3: Volatility normalized features (standardized)
        versions['vol_normalized'] = self._create_vol_normalized_features_standardized()
        
        # Version 4: Combined VWAP + volatility normalized (standardized)
        versions['vwap_vol_normalized'] = self._create_combined_features_standardized()
        
        logger.info(f"Generated {len(versions)} standardized feature versions")
        return versions
    
    def _create_initial_features_standardized(self) -> pd.DataFrame:
        """Create initial features with standardized naming."""
        df = self.data.copy()
        
        # Core returns features
        df['ret_t1'] = self._calculate_returns(df['close'], 1)  # log(P_t / P_{t-1})
        df['ret_abs_t1'] = np.abs(df['ret_t1'])
        df['ret_sq_t1'] = df['ret_t1'] ** 2
        
        # Rolling features with explicit windows
        windows = [5, 10, 20, 50]
        for w in windows:
            # Rolling mean: ret_ma_wW
            df[f'ret_ma_w{w}'] = df['ret_t1'].rolling(w).mean()
            
            # Rolling std: ret_std_wW
            df[f'ret_std_w{w}'] = df['ret_t1'].rolling(w).std()
            
            # EWMA: ret_ewmA_wA
            df[f'ret_ewm_w{w}'] = df['ret_t1'].ewm(span=w).mean()
            
            # Higher moments
            df[f'ret_skew_w{w}'] = df['ret_t1'].rolling(w).skew()
            df[f'ret_kurt_w{w}'] = df['ret_t1'].rolling(w).kurt()
        
        # Lagged features: ret_lagH
        for h in [1, 2, 3, 5, 10]:
            df[f'ret_lag{h}'] = df['ret_t1'].shift(h)
        
        # Momentum features (explicit definition)
        for k in [1, 2, 3, 5]:
            # Momentum as cumulative log-return over k periods
            df[f'ret_mom_k{k}'] = df['ret_t1'].rolling(k).sum()
            # Alternative: RSI-style momentum
            df[f'ret_rsi_k{k}'] = self._calculate_rsi_standardized(df['ret_t1'], k)
        
        # Acceleration: Δ momentum
        df['ret_acc_k1'] = df['ret_mom_k1'].diff()
        
        # Autocorrelation features using VectorBT
        try:
            import vectorbt as vbt
            from vectorbt.generic import rolling_corr
            
            for w in [10, 20]:
                # Use VectorBT for rolling autocorrelation
                df[f'ret_ac1_w{w}'] = rolling_corr(df['ret_t1'], df['ret_t1'].shift(1), window=w)
                df[f'ret_pac1_w{w}'] = self._calculate_partial_autocorr(df['ret_t1'], w)
                
        except Exception as e:
            logger.warning(f"VectorBT rolling correlation failed, using pandas: {e}")
            # Fallback to pandas
            for w in [10, 20]:
                df[f'ret_ac1_w{w}'] = df['ret_t1'].rolling(w).apply(lambda x: x.autocorr(lag=1))
                df[f'ret_pac1_w{w}'] = self._calculate_partial_autocorr(df['ret_t1'], w)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df = self._add_volume_features_standardized(df)
        
        # Drawdown features
        df = self._add_drawdown_features_standardized(df)
        
        # Entropy features
        df = self._add_entropy_features_standardized(df)
        
        return df
    
    def _create_vwap_features_standardized(self) -> pd.DataFrame:
        """Create VWAP-based features with standardized naming."""
        df = self._create_initial_features_standardized()
        
        # VWAP calculation with explicit windows
        windows = [10, 20, 50]
        for w in windows:
            df[f'vwap_w{w}'] = self._calculate_vwap(df, w)
            df[f'vwap_ret_w{w}'] = self._calculate_returns(df[f'vwap_w{w}'], 1)
        
        # VWAP basis and relative deviation
        df['vwap_basis_w20'] = df['close'] - df['vwap_w20']
        df['rel_vwap_dev_w20'] = df['vwap_basis_w20'] / df['vwap_w20']
        
        # VWAP momentum and volatility
        for w in [5, 10, 20]:
            df[f'vwap_ret_ma_w{w}'] = df['vwap_ret_w20'].rolling(w).mean()
            df[f'vwap_ret_std_w{w}'] = df['vwap_ret_w20'].rolling(w).std()
        
        # Rolling correlation with VWAP basis using VectorBT
        try:
            import vectorbt as vbt
            from vectorbt.generic import rolling_corr
            
            for w in [10, 20]:
                df[f'ret_vwap_corr_w{w}'] = rolling_corr(df['ret_t1'], df['vwap_basis_w20'], window=w)
                
        except Exception as e:
            logger.warning(f"VectorBT rolling correlation failed, using pandas: {e}")
            # Fallback to pandas
            for w in [10, 20]:
                df[f'ret_vwap_corr_w{w}'] = df['ret_t1'].rolling(w).corr(df['vwap_basis_w20'])
        
        return df
    
    def _create_vol_normalized_features_standardized(self) -> pd.DataFrame:
        """Create volatility normalized features with standardized naming."""
        df = self._create_initial_features_standardized()
        
        # Volatility calculation with explicit windows
        vol_windows = [10, 20, 50]
        for w in vol_windows:
            df[f'vol_w{w}'] = df['ret_t1'].rolling(w).std()
        
        # Volatility-normalized features
        for w in [5, 10, 20]:
            if f'ret_ma_w{w}' in df.columns:
                df[f'ret_ma_w{w}_normvol20'] = df[f'ret_ma_w{w}'] / df['vol_w20']
            if f'ret_std_w{w}' in df.columns:
                df[f'ret_std_w{w}_normvol20'] = df[f'ret_std_w{w}'] / df['vol_w20']
        
        # Volatility of volatility
        for w1 in [10, 20]:
            for w2 in [5, 10]:
                df[f'vol_w{w1}_std_w{w2}'] = df[f'vol_w{w1}'].rolling(w2).std()
        
        # Regime features
        df = self._add_regime_features_standardized(df)
        
        # Beta normalization
        df = self._add_beta_features_standardized(df)
        
        return df
    
    def _create_combined_features_standardized(self) -> pd.DataFrame:
        """Create combined VWAP + volatility normalized features."""
        df = self._create_initial_features_standardized()
        
        # VWAP features
        df['vwap_w20'] = self._calculate_vwap(df, 20)
        df['vwap_ret_w20'] = self._calculate_returns(df['vwap_w20'], 1)
        df['vwap_basis_w20'] = df['close'] - df['vwap_w20']
        df['rel_vwap_dev_w20'] = df['vwap_basis_w20'] / df['vwap_w20']
        
        # Volatility features
        df['vol_w20'] = df['ret_t1'].rolling(20).std()
        
        # Combined features with explicit windows
        df['vwap_ret_w20_normvol20'] = df['vwap_ret_w20'] / df['vol_w20']
        df['rel_vwap_dev_w20_normvol20'] = df['rel_vwap_dev_w20'] / df['vol_w20']
        
        # Interaction features
        df['ret_vol_interact'] = df['ret_t1'] * df['vol_w20']
        df['vwap_vol_interact'] = df['rel_vwap_dev_w20'] * df['vol_w20']
        
        # Regime-based interactions
        high_vol = df['vol_w20'] > df['vol_w20'].rolling(50).mean()
        df['ret_highvol_interact'] = df['ret_t1'] * high_vol.astype(int)
        
        return df
    
    def _calculate_returns(self, prices: pd.Series, h: int) -> pd.Series:
        """Calculate log returns: ret_t(h) = log(P_t / P_{t-h})"""
        return np.log(prices / prices.shift(h))
    
    def _calculate_vwap(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate VWAP over window W: vwap_t = Σ(P_i * V_i) / Σ(V_i) over W"""
        if 'volume' not in df.columns:
            return df['close']  # Fallback to close price
        
        return (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    
    def _calculate_rsi_standardized(self, returns: pd.Series, k: int) -> pd.Series:
        """Calculate RSI-style momentum over k periods."""
        gains = returns.where(returns > 0, 0).rolling(k).mean()
        losses = (-returns.where(returns < 0, 0)).rolling(k).mean()
        rs = gains / (losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_partial_autocorr(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate partial autocorrelation."""
        def pacf(x):
            if len(x) < 3:
                return np.nan
            try:
                from statsmodels.tsa.stattools import pacf
                result = pacf(x, nlags=1, method='ywm')
                return result[1] if len(result) > 1 else np.nan
            except:
                return np.nan
        
        return series.rolling(window).apply(pacf, raw=False)
    
    def _add_volume_features_standardized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized volume features."""
        # Volume returns
        df['vol_ret_t1'] = self._calculate_returns(df['volume'], 1)
        
        # Volume moving averages
        for w in [5, 10, 20]:
            df[f'vol_ma_w{w}'] = df['volume'].rolling(w).mean()
            df[f'vol_std_w{w}'] = df['volume'].rolling(w).std()
        
        # Volume ratios
        df['vol_adv_w20'] = df['volume'] / df['volume'].rolling(20).mean()  # ADV ratio
        
        # Volume-weighted returns
        df['vw_ret_w20'] = (df['ret_t1'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume-price trend
        df['vol_price_trend'] = df['vol_ret_t1'] * df['ret_t1']
        
        return df
    
    def _add_drawdown_features_standardized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add drawdown features."""
        # Current drawdown
        df['dd_current'] = df['close'] / df['close'].expanding().max() - 1
        
        # Maximum drawdown over windows
        for w in [10, 20, 50]:
            rolling_max = df['close'].rolling(w).max()
            df[f'dd_max_w{w}'] = (df['close'] / rolling_max - 1).rolling(w).min()
        
        return df
    
    def _add_entropy_features_standardized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add entropy/complexity features."""
        for w in [10, 20]:
            df[f'ret_perm_entropy_w{w}'] = df['ret_t1'].rolling(w).apply(
                lambda x: self._calculate_permutation_entropy(x), raw=False
            )
        
        return df
    
    def _calculate_permutation_entropy(self, series: pd.Series) -> float:
        """Calculate permutation entropy of a series."""
        if len(series) < 4:
            return np.nan
        
        try:
            # Convert to ordinal patterns
            sorted_indices = np.argsort(series.values)
            pattern = ''.join(map(str, sorted_indices))
            
            # Count pattern frequencies
            patterns = {}
            for i in range(len(pattern) - 2):
                subpattern = pattern[i:i+3]
                patterns[subpattern] = patterns.get(subpattern, 0) + 1
            
            # Calculate entropy
            total = sum(patterns.values())
            entropy = 0
            for count in patterns.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            return entropy
        except:
            return np.nan
    
    def _add_regime_features_standardized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime features."""
        # High volatility regime
        vol_ma = df['vol_w20'].rolling(50).mean()
        df['regime_highvol'] = (df['vol_w20'] > vol_ma).astype(int)
        
        # Volatility regime interactions
        df['ret_highvol_interact'] = df['ret_t1'] * df['regime_highvol']
        
        return df
    
    def _add_beta_features_standardized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add beta normalization features."""
        # Rolling beta to market (using VWAP as market proxy)
        for w in [10, 20]:
            if 'vwap_ret_w20' in df.columns:
                # Calculate rolling beta
                covariance = df['ret_t1'].rolling(w).cov(df['vwap_ret_w20'])
                market_var = df['vwap_ret_w20'].rolling(w).var()
                df[f'beta_market_w{w}'] = covariance / (market_var + 1e-8)
                
                # Beta-normalized returns
                df[f'ret_normbeta_w{w}'] = df['ret_t1'] / (df[f'beta_market_w{w}'] + 1e-8)
        
        return df
    
    def get_feature_matrix(self, version: str, 
                          exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature matrix for a specific version.
        
        Args:
            version: Version name
            exclude_cols: Columns to exclude from features
            
        Returns:
            Feature matrix
        """
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")
        
        df = self.versions[version].copy()
        
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        # Remove excluded columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return df[feature_cols]
    
    def get_feature_definitions(self) -> Dict[str, str]:
        """
        Get standardized feature definitions.
        
        Returns:
            Dictionary with feature definitions
        """
        definitions = {
            # Core returns
            'ret_t1': 'Log return: log(P_t / P_{t-1})',
            'ret_abs_t1': 'Absolute return: |ret_t1|',
            'ret_sq_t1': 'Squared return: ret_t1^2',
            
            # Rolling features
            'ret_ma_wW': 'Rolling mean of returns over window W',
            'ret_std_wW': 'Rolling std of returns over window W',
            'ret_ewm_wW': 'EWMA of returns with span W',
            'ret_skew_wW': 'Rolling skewness over window W',
            'ret_kurt_wW': 'Rolling kurtosis over window W',
            
            # Lagged features
            'ret_lagH': 'H-step lagged return',
            'ret_mom_kK': 'Momentum: cumulative return over K periods',
            'ret_rsi_kK': 'RSI-style momentum over K periods',
            'ret_acc_k1': 'Acceleration: Δ momentum',
            
            # Autocorrelation
            'ret_ac1_wW': '1-lag autocorrelation over window W',
            'ret_pac1_wW': '1-lag partial autocorrelation over window W',
            
            # VWAP features
            'vwap_wW': 'VWAP over window W',
            'vwap_ret_wW': 'VWAP return over window W',
            'vwap_basis_wW': 'VWAP basis: (price - vwap)',
            'rel_vwap_dev_wW': 'Relative VWAP deviation: (price - vwap)/vwap',
            'ret_vwap_corr_wW': 'Correlation between returns and VWAP basis over W',
            
            # Volatility features
            'vol_wW': 'Realized volatility over window W',
            'ret_ma_wW_normvolW2': 'Return MA normalized by volatility over W2',
            'vol_wW1_std_wW2': 'Volatility of volatility: std(vol_wW1) over W2',
            
            # Regime features
            'regime_highvol': 'High volatility regime indicator',
            'ret_highvol_interact': 'Return × high volatility interaction',
            
            # Beta features
            'beta_market_wW': 'Rolling beta to market over window W',
            'ret_normbeta_wW': 'Beta-normalized returns over window W',
            
            # Volume features
            'vol_ret_t1': 'Volume return: log(V_t / V_{t-1})',
            'vol_ma_wW': 'Volume moving average over window W',
            'vol_adv_wW': 'Volume/ADV ratio over window W',
            'vw_ret_wW': 'Volume-weighted return over window W',
            
            # Drawdown features
            'dd_current': 'Current drawdown from peak',
            'dd_max_wW': 'Maximum drawdown over window W',
            
            # Entropy features
            'ret_perm_entropy_wW': 'Permutation entropy of returns over window W',
            
            # Interaction features
            'ret_vol_interact': 'Return × volatility interaction',
            'vwap_vol_interact': 'VWAP deviation × volatility interaction',
        }
        
        return definitions