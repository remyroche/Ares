"""
Feature Comparison Utilities

This module provides utilities to call scripts from different feature engineering modules
and manage the feature comparison process.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import importlib.util

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

class FeatureComparisonUtils:
    """
    Utility class to call scripts from different feature engineering modules
    and manage feature comparison workflows.
    """
    
    def __init__(self, base_path: str = "/workspace/src"):
        """
        Initialize the feature comparison utilities.
        
        Args:
            base_path: Base path to the src directory
        """
        self.base_path = Path(base_path)
        self.feature_modules = {
            'lookback_optimization': self.base_path / 'training' / 'steps' / 'pre_training' / 'feature_lookback_optimization',
            'feature_generation': self.base_path / 'feature_generation',
            'features_common': self.base_path / 'features_common',
            'feature_selection': self.base_path / 'feature_selection'
        }
        
    def get_feature_generator(self) -> Any:
        """Get the feature generator from feature_generation module."""
        try:
            from src.feature_generation.core.feature_generator import FeatureGenerator
            return FeatureGenerator
        except ImportError as e:
            logger.error(f"Failed to import FeatureGenerator: {e}")
            return None
    
    def get_scalers(self) -> Dict[str, Any]:
        """Get available scalers from features_common module."""
        scalers = {}
        try:
            from src.features_common.transforms.base_scaler import BaseScaler
            scalers['base_scaler'] = BaseScaler
        except ImportError as e:
            logger.warning(f"Failed to import BaseScaler: {e}")
        
        return scalers
    
    def get_lookback_optimizer(self) -> Any:
        """Get the lookback optimizer from feature_lookback_optimization module."""
        try:
            from src.training.steps.pre_training.feature_lookback_optimization.core.optimizer import Optimizer
            return Optimizer
        except ImportError as e:
            logger.error(f"Failed to import Optimizer: {e}")
            return None
    
    def get_feature_selector(self) -> Any:
        """Get feature selection utilities from feature_selection module."""
        try:
            from src.feature_selection.selector import FeatureSelector
            return FeatureSelector
        except ImportError as e:
            logger.error(f"Failed to import FeatureSelector: {e}")
            return None
    
    def create_vwap_features(self, data: pd.DataFrame, price_col: str = 'close', 
                           volume_col: str = 'volume') -> pd.DataFrame:
        """
        Create VWAP-based features.
        
        Args:
            data: Input DataFrame with OHLCV data
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            DataFrame with VWAP features
        """
        df = data.copy()
        
        # Calculate VWAP
        df['vwap'] = (df[price_col] * df[volume_col]).cumsum() / df[volume_col].cumsum()
        
        # VWAP-based features
        df['price_vwap_ratio'] = df[price_col] / df['vwap']
        df['price_vwap_diff'] = df[price_col] - df['vwap']
        df['price_vwap_pct'] = (df[price_col] - df['vwap']) / df['vwap'] * 100
        
        # VWAP momentum
        df['vwap_momentum_5'] = df['vwap'].pct_change(5)
        df['vwap_momentum_10'] = df['vwap'].pct_change(10)
        df['vwap_momentum_20'] = df['vwap'].pct_change(20)
        
        # VWAP volatility
        df['vwap_volatility_5'] = df['vwap'].rolling(5).std()
        df['vwap_volatility_10'] = df['vwap'].rolling(10).std()
        df['vwap_volatility_20'] = df['vwap'].rolling(20).std()
        
        return df
    
    def create_volatility_normalized_features(self, data: pd.DataFrame, 
                                            lookback: int = 20) -> pd.DataFrame:
        """
        Create volatility-normalized features.
        
        Args:
            data: Input DataFrame
            lookback: Lookback period for volatility calculation
            
        Returns:
            DataFrame with volatility-normalized features
        """
        df = data.copy()
        
        # Calculate rolling volatility
        df['volatility'] = df['close'].rolling(lookback).std()
        
        # Volatility-normalized features
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[f'{col}_vol_norm'] = df[col] / df['volatility']
                df[f'{col}_vol_norm_ma'] = df[f'{col}_vol_norm'].rolling(5).mean()
        
        # Volume volatility normalization
        if 'volume' in df.columns:
            df['volume_vol_norm'] = df['volume'] / df['volatility']
        
        return df
    
    def create_combined_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined VWAP + volatility normalized features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with combined features
        """
        df = data.copy()
        
        # First create VWAP features
        df = self.create_vwap_features(df)
        
        # Then create volatility normalized features
        df = self.create_volatility_normalized_features(df)
        
        # Combined features
        df['vwap_vol_norm'] = df['vwap'] / df['volatility']
        df['price_vwap_vol_norm_ratio'] = df['price_vwap_ratio'] / df['volatility']
        
        return df
    
    def prepare_feature_versions(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Prepare the 4 different feature versions for comparison.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            Dictionary with 4 versions of features
        """
        versions = {}
        
        # Version 1: Initial features (basic OHLCV)
        versions['initial'] = data.copy()
        
        # Version 2: VWAP-based features
        versions['vwap_based'] = self.create_vwap_features(data)
        
        # Version 3: Volatility normalized features
        versions['vol_normalized'] = self.create_volatility_normalized_features(data)
        
        # Version 4: VWAP + volatility normalized features
        versions['vwap_vol_normalized'] = self.create_combined_features(data)
        
        return versions
    
    def get_feature_columns(self, data: pd.DataFrame, 
                          exclude_base: bool = True) -> List[str]:
        """
        Get feature columns from the dataset.
        
        Args:
            data: Input DataFrame
            exclude_base: Whether to exclude base OHLCV columns
            
        Returns:
            List of feature column names
        """
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in data.columns if col not in base_cols]
        return feature_cols