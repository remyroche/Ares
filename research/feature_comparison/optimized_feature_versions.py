"""
Optimized Feature Versions with Matrix Operations and Hardware Acceleration

This module provides optimized feature generation using matrix operations
and hardware acceleration for the feature comparison framework.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from .feature_comparison_utils import FeatureComparisonUtils

# Try to import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations, get_vectorized_processing_core
    from src.utils.hardware import get_unified_hardware_manager
    MATRIX_OPS_AVAILABLE = True
    HARDWARE_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    HARDWARE_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizedFeatureVersions:
    """
    Optimized feature versions manager with matrix operations and hardware acceleration.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'returns', 
                 enable_matrix_ops: bool = True, enable_hardware_opt: bool = True):
        """
        Initialize optimized feature versions manager.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_col: Name of target column
            enable_matrix_ops: Whether to enable matrix operations
            enable_hardware_opt: Whether to enable hardware optimizations
        """
        self.data = data.copy()
        self.target_col = target_col
        self.enable_matrix_ops = enable_matrix_ops and MATRIX_OPS_AVAILABLE
        self.enable_hardware_opt = enable_hardware_opt and HARDWARE_AVAILABLE
        self.utils = FeatureComparisonUtils()
        self.versions = {}
        self.target = None
        
        # Initialize matrix operations
        if self.enable_matrix_ops:
            try:
                self.matrix_ops = get_unified_matrix_operations(
                    enable_gpu=True, enable_parallel=True
                )
                self.vectorized_core = get_vectorized_processing_core()
                logger.info("✅ Matrix operations initialized")
            except Exception as e:
                logger.warning(f"Matrix operations not available: {e}")
                self.enable_matrix_ops = False
        
        # Initialize hardware optimizations
        if self.enable_hardware_opt:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                logger.info("✅ Hardware optimizations initialized")
            except Exception as e:
                logger.warning(f"Hardware optimizations not available: {e}")
                self.enable_hardware_opt = False
    
    def create_target(self, method: str = 'future_returns', 
                     periods: int = 1) -> pd.Series:
        """
        Create target variable using returns instead of raw prices.
        
        Args:
            method: Method to create target ('future_returns', 'price_direction', 'volatility')
            periods: Number of periods ahead for future returns
            
        Returns:
            Target series based on returns
        """
        # Calculate returns first
        returns = self.data['close'].pct_change()
        
        if method == 'future_returns':
            # Future returns
            self.target = returns.shift(-periods)
        elif method == 'price_direction':
            # Price direction based on returns
            future_returns = returns.shift(-periods)
            self.target = (future_returns > 0).astype(int)
        elif method == 'volatility':
            # Future volatility of returns
            future_returns = returns.shift(-periods)
            self.target = future_returns.rolling(periods).std()
        else:
            raise ValueError(f"Unknown target method: {method}")
        
        return self.target
    
    def generate_all_versions(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all 4 versions of features using optimized operations.
        
        Returns:
            Dictionary with all feature versions
        """
        logger.info("Generating optimized feature versions...")
        
        # Version 1: Initial features (returns-based + technical indicators)
        self.versions['initial'] = self._create_initial_features_optimized()
        
        # Version 2: VWAP-based features (using returns)
        self.versions['vwap_based'] = self._create_vwap_features_optimized()
        
        # Version 3: Volatility normalized features (returns-based)
        self.versions['vol_normalized'] = self._create_vol_normalized_features_optimized()
        
        # Version 4: VWAP + volatility normalized features
        self.versions['vwap_vol_normalized'] = self._create_combined_features_optimized()
        
        logger.info(f"Generated {len(self.versions)} optimized feature versions")
        return self.versions
    
    def _create_initial_features_optimized(self) -> pd.DataFrame:
        """Create initial features using returns and optimized operations."""
        df = self.data.copy()
        
        # Calculate returns (primary feature)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios using returns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Returns-based features
        df['returns_abs'] = df['returns'].abs()
        df['returns_squared'] = df['returns'] ** 2
        
        # Rolling features using matrix operations if available
        if self.enable_matrix_ops:
            df = self._add_rolling_features_matrix(df)
        else:
            df = self._add_rolling_features_standard(df)
        
        # Lagged and lead features
        df = self._add_lagged_lead_features(df)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df = self._add_volume_features(df)
        
        return df
    
    def _add_rolling_features_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features using VectorBT operations."""
        try:
            import vectorbt as vbt
            from vectorbt.generic import rolling_mean, rolling_std, rolling_apply
            from vectorbt.indicators.basic import RSI
            
            # Get returns data
            returns_data = df['returns'].dropna()
            
            # Rolling windows
            windows = [5, 10, 20, 50]
            
            # Use VectorBT for rolling operations
            for window in windows:
                if len(returns_data) >= window:
                    # Rolling mean using VectorBT
                    df[f'returns_ma_{window}'] = rolling_mean(returns_data, window=window)
                    
                    # Rolling std using VectorBT
                    df[f'returns_std_{window}'] = rolling_std(returns_data, window=window)
                    
                    # EWMA using VectorBT
                    df[f'returns_ewma_{window}'] = returns_data.ewm(span=window).mean()
            
            # RSI using VectorBT
            df['rsi_14'] = RSI.run(returns_data, window=14).rsi
            
        except Exception as e:
            logger.warning(f"VectorBT operations failed, falling back to standard: {e}")
            df = self._add_rolling_features_standard(df)
        
        return df
    
    def _add_rolling_features_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features using VectorBT operations."""
        try:
            import vectorbt as vbt
            from vectorbt.generic import rolling_mean, rolling_std, rolling_apply
            from vectorbt.indicators.basic import RSI
            
            windows = [5, 10, 20, 50]
            
            for window in windows:
                # Rolling mean using VectorBT
                df[f'returns_ma_{window}'] = rolling_mean(df['returns'], window=window)
                
                # Rolling std using VectorBT
                df[f'returns_std_{window}'] = rolling_std(df['returns'], window=window)
                
                # EWMA using VectorBT
                df[f'returns_ewma_{window}'] = df['returns'].ewm(span=window).mean()
                
                # Rolling skewness and kurtosis using VectorBT
                df[f'returns_skew_{window}'] = rolling_apply(df['returns'], window=window, func=lambda x: x.skew())
                df[f'returns_kurt_{window}'] = rolling_apply(df['returns'], window=window, func=lambda x: x.kurt())
            
            # RSI using VectorBT
            df['rsi_14'] = RSI.run(df['returns'], window=14).rsi
            
        except Exception as e:
            logger.warning(f"VectorBT operations failed, using pandas fallback: {e}")
            # Fallback to pandas if VectorBT fails
            windows = [5, 10, 20, 50]
            
            for window in windows:
                df[f'returns_ma_{window}'] = df['returns'].rolling(window).mean()
                df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
                df[f'returns_ewma_{window}'] = df['returns'].ewm(span=window).mean()
                df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
                df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
            
            df['rsi_14'] = self._calculate_rsi_standard(df['returns'])
        
        return df
    
    def _add_lagged_lead_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged and lead features for predictive analysis."""
        # Lagged returns (reactive features)
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'returns_abs_lag_{lag}'] = df['returns_abs'].shift(lag)
        
        # Lead returns (predictive features)
        for lead in [1, 2, 3, 5]:
            df[f'returns_lead_{lead}'] = df['returns'].shift(-lead)
        
        # Lagged volatility
        for lag in [1, 2, 3, 5]:
            df[f'volatility_lag_{lag}'] = df['returns_std_20'].shift(lag)
        
        # Returns momentum (difference between current and lagged)
        for lag in [1, 2, 3, 5]:
            df[f'returns_momentum_{lag}'] = df['returns'] - df['returns'].shift(lag)
        
        # Returns acceleration (second difference)
        df['returns_acceleration'] = df['returns'] - 2 * df['returns'].shift(1) + df['returns'].shift(2)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume returns
        df['volume_returns'] = df['volume'].pct_change()
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume_returns'] * df['returns']
        
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # Volume-weighted returns
        df['vw_returns'] = df['returns'] * df['volume']
        
        return df
    
    def _create_vwap_features_optimized(self) -> pd.DataFrame:
        """Create VWAP-based features using returns."""
        df = self._create_initial_features_optimized()
        
        # Calculate VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # VWAP returns
        df['vwap_returns'] = df['vwap'].pct_change()
        
        # VWAP-based features using returns
        df['returns_vwap_ratio'] = df['returns'] / df['vwap_returns']
        df['returns_vwap_diff'] = df['returns'] - df['vwap_returns']
        
        # VWAP momentum using returns
        for window in [5, 10, 20]:
            df[f'vwap_returns_ma_{window}'] = df['vwap_returns'].rolling(window).mean()
            df[f'vwap_returns_std_{window}'] = df['vwap_returns'].rolling(window).std()
            df[f'vwap_returns_momentum_{window}'] = df['vwap_returns'] - df['vwap_returns'].shift(window)
        
        # VWAP volatility
        df['vwap_volatility'] = df['vwap_returns'].rolling(20).std()
        
        return df
    
    def _create_vol_normalized_features_optimized(self) -> pd.DataFrame:
        """Create volatility normalized features using returns."""
        df = self._create_initial_features_optimized()
        
        # Calculate rolling volatility of returns
        df['returns_volatility'] = df['returns'].rolling(20).std()
        
        # Volatility-normalized returns
        df['returns_vol_norm'] = df['returns'] / df['returns_volatility']
        df['returns_abs_vol_norm'] = df['returns_abs'] / df['returns_volatility']
        
        # Volatility-normalized rolling features
        for window in [5, 10, 20]:
            df[f'returns_ma_{window}_vol_norm'] = df[f'returns_ma_{window}'] / df['returns_volatility']
            df[f'returns_std_{window}_vol_norm'] = df[f'returns_std_{window}'] / df['returns_volatility']
        
        # Volatility regime features
        df['vol_regime'] = (df['returns_volatility'] > df['returns_volatility'].rolling(50).mean()).astype(int)
        df['high_vol_returns'] = df['returns'] * df['vol_regime']
        df['low_vol_returns'] = df['returns'] * (1 - df['vol_regime'])
        
        return df
    
    def _create_combined_features_optimized(self) -> pd.DataFrame:
        """Create combined VWAP + volatility normalized features."""
        df = self._create_initial_features_optimized()
        
        # VWAP features
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_returns'] = df['vwap'].pct_change()
        
        # Volatility features
        df['returns_volatility'] = df['returns'].rolling(20).std()
        
        # Combined features
        df['vwap_returns_vol_norm'] = df['vwap_returns'] / df['returns_volatility']
        df['returns_vwap_vol_norm_ratio'] = df['returns'] / df['vwap_returns_vol_norm']
        
        # Advanced combined features
        df['vwap_vol_regime'] = (df['vwap_returns'].rolling(20).std() > df['vwap_returns'].rolling(50).std()).astype(int)
        df['combined_momentum'] = df['returns'] * df['vwap_returns'] * df['returns_volatility']
        
        return df
    
    def _calculate_rsi_matrix(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using matrix operations."""
        try:
            if self.enable_matrix_ops:
                returns_data = returns.dropna().values.reshape(-1, 1)
                rsi_values = self.matrix_ops.calculate_rsi(returns_data, window)
                return pd.Series(rsi_values.flatten(), index=returns.index)
            else:
                return self._calculate_rsi_standard(returns, window)
        except Exception as e:
            logger.warning(f"Matrix RSI calculation failed: {e}")
            return self._calculate_rsi_standard(returns, window)
    
    def _calculate_rsi_standard(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using standard method."""
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'vwap']
            if self.target_col in df.columns:
                exclude_cols.append(self.target_col)
        
        # Remove excluded columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return df[feature_cols]
    
    def get_version_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about each version.
        
        Returns:
            Dictionary with version information
        """
        info = {}
        
        for version_name, version_df in self.versions.items():
            feature_cols = self.get_feature_matrix(version_name).columns
            info[version_name] = {
                'n_features': len(feature_cols),
                'feature_names': list(feature_cols),
                'n_samples': len(version_df),
                'has_nan': version_df.isna().any().any(),
                'nan_count': version_df.isna().sum().sum(),
                'matrix_ops_enabled': self.enable_matrix_ops,
                'hardware_opt_enabled': self.enable_hardware_opt
            }
        
        return info