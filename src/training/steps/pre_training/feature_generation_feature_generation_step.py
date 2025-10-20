"""
Feature Generation Feature Generation Step

This step generates base features from market data.
It creates technical indicators, statistical features, and transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.training.steps.base_step import BaseStep


class FeatureGenerationFeatureGenerationStep(BaseStep):
    """
    Generates base features from market data.
    
    Features include:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Statistical features (rolling mean, std, skew, kurtosis)
    - Price transformations (returns, log returns, etc.)
    - Volume features
    - Time-based features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature generation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_feature_generation_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature generation.
        
        Args:
            config: Configuration containing:
                - feature_types: List of feature types to generate
                - lookback_periods: List of lookback periods
                - generate_interactions: Whether to generate interaction features
        
        Returns:
            Dictionary containing:
                - success: bool
                - features_data_path: str
                - feature_list: List[str]
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("⚙️ Starting feature generation")
            
            # Load labeled data
            labeled_data = self._load_dataframe('labeled_data')
            if labeled_data is None:
                return {
                    'success': False,
                    'error': 'No labeled data found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Initialize features dataframe
            features_df = labeled_data.copy()
            feature_list = list(labeled_data.columns)
            
            # Get configuration
            feature_types = config.get('feature_types', ['momentum', 'volatility', 'volume'])
            lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
            
            # Generate momentum features
            if 'momentum' in feature_types:
                features_df, new_features = self._generate_momentum_features(
                    features_df, lookback_periods
                )
                feature_list.extend(new_features)
                self.logger.info(f"Generated {len(new_features)} momentum features")
            
            # Generate volatility features
            if 'volatility' in feature_types:
                features_df, new_features = self._generate_volatility_features(
                    features_df, lookback_periods
                )
                feature_list.extend(new_features)
                self.logger.info(f"Generated {len(new_features)} volatility features")
            
            # Generate volume features
            if 'volume' in feature_types:
                features_df, new_features = self._generate_volume_features(
                    features_df, lookback_periods
                )
                feature_list.extend(new_features)
                self.logger.info(f"Generated {len(new_features)} volume features")
            
            # Generate statistical features
            if 'statistical' in feature_types:
                features_df, new_features = self._generate_statistical_features(
                    features_df, lookback_periods
                )
                feature_list.extend(new_features)
                self.logger.info(f"Generated {len(new_features)} statistical features")
            
            # Remove any NaN values created by rolling operations
            initial_rows = len(features_df)
            features_df = features_df.dropna()
            dropped_rows = initial_rows - len(features_df)
            
            if dropped_rows > 0:
                self.logger.info(f"Dropped {dropped_rows} rows with NaN values")
            
            # Save generated features
            features_path = self._save_dataframe(
                features_df,
                'generated_features',
                metadata={'feature_count': len(feature_list)}
            )
            
            # Save feature list
            feature_list_path = self._save_metadata(
                {'features': feature_list},
                'feature_list'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'features_data_path': features_path,
                'feature_list': feature_list,
                'artifacts': [features_path, feature_list_path],
                'metrics': {
                    'total_features': len(feature_list),
                    'data_rows': len(features_df),
                    'dropped_rows': dropped_rows,
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    def _generate_momentum_features(
        self, df: pd.DataFrame, periods: List[int]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Generate momentum-based features."""
        new_features = []
        
        if 'close' not in df.columns:
            return df, new_features
        
        for period in periods:
            # Returns
            feature_name = f'returns_{period}'
            df[feature_name] = df['close'].pct_change(period)
            new_features.append(feature_name)
            
            # Momentum (rate of change)
            feature_name = f'momentum_{period}'
            df[feature_name] = df['close'] / df['close'].shift(period) - 1
            new_features.append(feature_name)
            
            # RSI-like momentum
            if period >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                feature_name = f'rsi_{period}'
                df[feature_name] = 100 - (100 / (1 + rs))
                new_features.append(feature_name)
        
        return df, new_features
    
    def _generate_volatility_features(
        self, df: pd.DataFrame, periods: List[int]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Generate volatility-based features."""
        new_features = []
        
        if 'close' not in df.columns:
            return df, new_features
        
        returns = df['close'].pct_change()
        
        for period in periods:
            # Standard deviation of returns
            feature_name = f'volatility_{period}'
            df[feature_name] = returns.rolling(window=period).std()
            new_features.append(feature_name)
            
            # Parkinson volatility (using high-low)
            if 'high' in df.columns and 'low' in df.columns:
                feature_name = f'parkinson_volatility_{period}'
                hl_ratio = np.log(df['high'] / df['low'])
                df[feature_name] = hl_ratio.rolling(window=period).std()
                new_features.append(feature_name)
        
        return df, new_features
    
    def _generate_volume_features(
        self, df: pd.DataFrame, periods: List[int]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Generate volume-based features."""
        new_features = []
        
        if 'volume' not in df.columns:
            return df, new_features
        
        for period in periods:
            # Volume moving average
            feature_name = f'volume_ma_{period}'
            df[feature_name] = df['volume'].rolling(window=period).mean()
            new_features.append(feature_name)
            
            # Volume ratio
            feature_name = f'volume_ratio_{period}'
            df[feature_name] = df['volume'] / (df['volume'].rolling(window=period).mean() + 1e-10)
            new_features.append(feature_name)
        
        return df, new_features
    
    def _generate_statistical_features(
        self, df: pd.DataFrame, periods: List[int]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Generate statistical features."""
        new_features = []
        
        if 'close' not in df.columns:
            return df, new_features
        
        for period in periods:
            # Rolling mean
            feature_name = f'close_mean_{period}'
            df[feature_name] = df['close'].rolling(window=period).mean()
            new_features.append(feature_name)
            
            # Rolling std
            feature_name = f'close_std_{period}'
            df[feature_name] = df['close'].rolling(window=period).std()
            new_features.append(feature_name)
            
            # Z-score
            feature_name = f'zscore_{period}'
            rolling_mean = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            df[feature_name] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
            new_features.append(feature_name)
        
        return df, new_features
