"""
Shared Feature Collector for Regime Detection Systems.

This module provides standardized feature collection utilities that can be used
by both NAS and TAS regime detection systems. It follows the same pattern as
hmm_regime_discovery.py for data loading and feature calculation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .standardized_features import StandardizedFeatureCalculator
from src.utils.logger import system_logger


class SharedFeatureCollector:
    """
    Shared feature collector for both NAS and TAS regime detection.

    This class provides standardized feature collection following the same
    pattern as hmm_regime_discovery.py, ensuring consistency across regime
    detection systems.
    """

    def __init__(self):
        """Initialize the shared feature collector."""
        self.logger = system_logger.getChild('SharedFeatureCollector')
        self.standardized_calc = StandardizedFeatureCalculator()
        self._resources_to_cleanup = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Clean up any allocated resources."""
        try:
            for resource in self._resources_to_cleanup:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            self._resources_to_cleanup.clear()
        except Exception as e:
            self.logger.warning(f"Error during resource cleanup: {e}")

    async def collect_features(self,
                              data: Any,
                              symbol: str,
                              timeframe: str = "15m",
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect standardized features for regime detection.

        Args:
            data: Market data (DataFrame or None to load from klines_parquet)
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            Dictionary containing standardized features and metadata
        """
        self.logger.info(f'🔍 Collecting features for {symbol} {timeframe}')

        try:
            # Load market data using same pattern as hmm_regime_discovery.py
            market_data = await self._load_market_data(data, symbol, timeframe, start_date, end_date)

            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for feature collection: {symbol} {timeframe}")

            # Calculate standardized features
            standardized_features = self.standardized_calc.calculate_all_features(market_data)

            # Get primary features by dimension
            primary_features = self.standardized_calc.get_primary_features()

            # Prepare grouped features for regime detection
            grouped_features = self._prepare_grouped_features(standardized_features, primary_features)

            # Create feature metadata
            feature_metadata = self._create_feature_metadata(standardized_features, primary_features)

            self.logger.info(f'✅ Feature collection completed: {len(standardized_features.columns)} features')

            return {
                'market_data': market_data,
                'standardized_features': standardized_features,
                'grouped_features': grouped_features,
                'primary_features': primary_features,
                'feature_metadata': feature_metadata,
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points': len(market_data),
                'execution_info': {
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            }

        except Exception as e:
            self.logger.error(f'❌ Feature collection failed: {e}')
            return {
                'market_data': None,
                'standardized_features': None,
                'grouped_features': None,
                'primary_features': None,
                'feature_metadata': None,
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points': 0,
                'execution_info': {
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error_message': str(e)
                }
            }

    async def _load_market_data(self,
                               data: Any,
                               symbol: str,
                               timeframe: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load market data using the same pattern as hmm_regime_discovery.py.

        Args:
            data: Market data (DataFrame or None)
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Market data DataFrame
        """
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                # Try to load data using klines_parquet manager (same as hmm_regime_discovery.py)
                from src.utils.data.klines_parquet import get_klines_manager

                manager = get_klines_manager()

                # Parse date filters if provided
                start_dt = None
                end_dt = None
                if start_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if end_date:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")

                # Try processed data first (better for analysis)
                market_data = manager.read_data(symbol, timeframe, start_date=start_dt, end_date=end_dt, data_type="processed")

                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    self.logger.info(f"📊 No processed data found, trying raw {symbol} {timeframe} data")
                    market_data = manager.read_data(symbol, timeframe, start_date=start_dt, end_date=end_dt, data_type="raw")

                if market_data is None or market_data.empty:
                    self.logger.error(f"❌ No data available for {symbol} {timeframe}")
                    return None

                self.logger.info(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data

            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                self.logger.info(f"📊 Using provided DataFrame with {len(data)} rows")
                return data.copy()

            return None

        except Exception as e:
            self.logger.exception(f"❌ Error loading market data: {e}")
            return None

    def _prepare_grouped_features(self,
                                 standardized_features: pd.DataFrame,
                                 primary_features: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Prepare features grouped by dimension for regime detection.

        Args:
            standardized_features: All standardized features
            primary_features: Primary features by dimension

        Returns:
            Dictionary of features grouped by dimension
        """
        try:
            grouped_features = {}

            # Prepare momentum features
            momentum_features = pd.DataFrame(index=standardized_features.index)
            for feature_name in primary_features['momentum']:
                if feature_name in standardized_features.columns:
                    momentum_features[feature_name] = standardized_features[feature_name]

            # Prepare volatility features
            volatility_features = pd.DataFrame(index=standardized_features.index)
            for feature_name in primary_features['volatility']:
                if feature_name in standardized_features.columns:
                    volatility_features[feature_name] = standardized_features[feature_name]

            # Prepare volume features
            volume_features = pd.DataFrame(index=standardized_features.index)
            for feature_name in primary_features['volume']:
                if feature_name in standardized_features.columns:
                    volume_features[feature_name] = standardized_features[feature_name]

            # Prepare trend features
            trend_features = pd.DataFrame(index=standardized_features.index)
            for feature_name in primary_features['trend']:
                if feature_name in standardized_features.columns:
                    trend_features[feature_name] = standardized_features[feature_name]

            # Clean and standardize features
            grouped_features = {
                'momentum': self._clean_dimension_features(momentum_features, 'momentum'),
                'volatility': self._clean_dimension_features(volatility_features, 'volatility'),
                'volume': self._clean_dimension_features(volume_features, 'volume'),
                'trend': self._clean_dimension_features(trend_features, 'trend')
            }

            self.logger.info("✅ Prepared grouped features by dimension")
            return grouped_features

        except Exception as e:
            self.logger.error(f"❌ Failed to prepare grouped features: {e}")
            return {}

    def _clean_dimension_features(self, features: pd.DataFrame, dimension: str) -> pd.DataFrame:
        """
        Clean and validate features for a specific dimension.

        Args:
            features: Features DataFrame for the dimension
            dimension: Name of the dimension (momentum, volatility, volume, trend)

        Returns:
            Cleaned features DataFrame
        """
        try:
            if features.empty:
                return features

            # Replace infinite values with NaN
            features = features.replace([np.inf, -np.inf], np.nan)

            # Handle NaN values based on dimension
            for col in features.columns:
                if features[col].isnull().any():
                    if dimension == 'momentum':
                        features[col] = features[col].fillna(0.0)
                    elif dimension == 'volatility':
                        # Use forward fill then median for volatility
                        features[col] = features[col].fillna(method='ffill')
                        median_val = features[col].median()
                        features[col] = features[col].fillna(median_val if not pd.isna(median_val) else 0.01)
                    elif dimension == 'volume':
                        # Volume ratios default to 1.0 (neutral)
                        if 'ratio' in col.lower():
                            features[col] = features[col].fillna(1.0)
                        else:
                            features[col] = features[col].fillna(0.0)
                    elif dimension == 'trend':
                        features[col] = features[col].fillna(0.0)

            # Ensure finite values
            features = features.astype(np.float64)
            features = features.replace([np.inf, -np.inf], 0.0)

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to clean {dimension} features: {e}")
            return features.fillna(0.0)

    def _create_feature_metadata(self,
                                standardized_features: pd.DataFrame,
                                primary_features: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create metadata about the collected features.

        Args:
            standardized_features: All standardized features
            primary_features: Primary features by dimension

        Returns:
            Feature metadata dictionary
        """
        try:
            metadata = {
                'total_features': len(standardized_features.columns),
                'feature_dimensions': {
                    'momentum': len(primary_features['momentum']),
                    'volatility': len(primary_features['volatility']),
                    'volume': len(primary_features['volume']),
                    'trend': len(primary_features['trend'])
                },
                'feature_names': {
                    'momentum': primary_features['momentum'],
                    'volatility': primary_features['volatility'],
                    'volume': primary_features['volume'],
                    'trend': primary_features['trend']
                },
                'feature_statistics': {
                    col: {
                        'mean': standardized_features[col].mean(),
                        'std': standardized_features[col].std(),
                        'min': standardized_features[col].min(),
                        'max': standardized_features[col].max(),
                        'nan_count': standardized_features[col].isnull().sum()
                    }
                    for col in standardized_features.columns
                }
            }

            return metadata

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create feature metadata: {e}")
            return {}