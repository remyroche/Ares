"""
Feature Versions Manager

This module manages the different versions of features for comparison:
1. Initial features (basic OHLCV)
2. VWAP-based features
3. Volatility normalized features
4. VWAP + volatility normalized features
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from .feature_comparison_utils import FeatureComparisonUtils

logger = logging.getLogger(__name__)

class FeatureVersions:
    """
    Manages different versions of features for comparison.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'returns'):
        """
        Initialize feature versions manager.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_col: Name of target column
        """
        self.data = data.copy()
        self.target_col = target_col
        self.utils = FeatureComparisonUtils()
        self.versions = {}
        self.target = None
        
    def create_target(self, method: str = 'future_returns', 
                     periods: int = 1) -> pd.Series:
        """
        Create target variable for analysis.
        
        Args:
            method: Method to create target ('future_returns', 'price_direction', 'volatility')
            periods: Number of periods ahead for future returns
            
        Returns:
            Target series
        """
        if method == 'future_returns':
            # Future returns
            self.target = self.data['close'].pct_change(periods).shift(-periods)
        elif method == 'price_direction':
            # Price direction (1 for up, 0 for down)
            future_price = self.data['close'].shift(-periods)
            self.target = (future_price > self.data['close']).astype(int)
        elif method == 'volatility':
            # Future volatility
            future_returns = self.data['close'].pct_change(periods).shift(-periods)
            self.target = future_returns.rolling(periods).std()
        else:
            raise ValueError(f"Unknown target method: {method}")
        
        return self.target
    
    def generate_all_versions(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all 4 versions of features.
        
        Returns:
            Dictionary with all feature versions
        """
        logger.info("Generating all feature versions...")
        
        # Version 1: Initial features (basic OHLCV + basic technical indicators)
        self.versions['initial'] = self._create_initial_features()
        
        # Version 2: VWAP-based features
        self.versions['vwap_based'] = self._create_vwap_features()
        
        # Version 3: Volatility normalized features
        self.versions['vol_normalized'] = self._create_vol_normalized_features()
        
        # Version 4: VWAP + volatility normalized features
        self.versions['vwap_vol_normalized'] = self._create_combined_features()
        
        logger.info(f"Generated {len(self.versions)} feature versions")
        return self.versions
    
    def _create_initial_features(self) -> pd.DataFrame:
        """Create initial features (basic OHLCV + basic technical indicators)."""
        df = self.data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Price ratios
        for window in [5, 10, 20]:
            df[f'price_sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
            df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(20).mean()
        
        # Momentum features
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)
        
        return df
    
    def _create_vwap_features(self) -> pd.DataFrame:
        """Create VWAP-based features."""
        df = self._create_initial_features()
        df = self.utils.create_vwap_features(df)
        return df
    
    def _create_vol_normalized_features(self) -> pd.DataFrame:
        """Create volatility normalized features."""
        df = self._create_initial_features()
        df = self.utils.create_volatility_normalized_features(df)
        return df
    
    def _create_combined_features(self) -> pd.DataFrame:
        """Create combined VWAP + volatility normalized features."""
        df = self._create_initial_features()
        df = self.utils.create_combined_features(df)
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
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
            version: Version name ('initial', 'vwap_based', 'vol_normalized', 'vwap_vol_normalized')
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
                'nan_count': version_df.isna().sum().sum()
            }
        
        return info
    
    def compare_feature_counts(self) -> pd.DataFrame:
        """
        Compare feature counts across versions.
        
        Returns:
            DataFrame with feature count comparison
        """
        info = self.get_version_info()
        
        comparison = pd.DataFrame({
            'version': list(info.keys()),
            'n_features': [info[v]['n_features'] for v in info.keys()],
            'n_samples': [info[v]['n_samples'] for v in info.keys()],
            'has_nan': [info[v]['has_nan'] for v in info.keys()],
            'nan_count': [info[v]['nan_count'] for v in info.keys()]
        })
        
        return comparison
    
    def get_common_features(self) -> List[str]:
        """
        Get features that are common across all versions.
        
        Returns:
            List of common feature names
        """
        if not self.versions:
            return []
        
        # Get feature names for each version
        all_features = []
        for version_name in self.versions.keys():
            features = set(self.get_feature_matrix(version_name).columns)
            all_features.append(features)
        
        # Find intersection
        common_features = set.intersection(*all_features)
        return sorted(list(common_features))
    
    def get_version_specific_features(self) -> Dict[str, List[str]]:
        """
        Get features that are specific to each version.
        
        Returns:
            Dictionary with version-specific features
        """
        if not self.versions:
            return {}
        
        # Get all features for each version
        version_features = {}
        for version_name in self.versions.keys():
            features = set(self.get_feature_matrix(version_name).columns)
            version_features[version_name] = features
        
        # Find version-specific features
        specific_features = {}
        for version_name, features in version_features.items():
            other_features = set()
            for other_version, other_feature_set in version_features.items():
                if other_version != version_name:
                    other_features.update(other_feature_set)
            
            specific = features - other_features
            specific_features[version_name] = sorted(list(specific))
        
        return specific_features