"""
Shared Data Pipeline for Regime Detection Systems.

This module provides a standardized data pipeline that follows the same
pattern as hmm_regime_discovery.py for data loading and preprocessing.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

from src.utils.logger import system_logger
from .data_preprocessor import DataPreprocessor


class SharedDataPipeline:
    """
    Shared data pipeline for both NAS and TAS regime detection systems.

    This class provides standardized data loading and preprocessing following
    the same pattern as hmm_regime_discovery.py, ensuring consistency across
    regime detection systems.
    """

    def __init__(self):
        """Initialize the shared data pipeline."""
        self.logger = system_logger.getChild('SharedDataPipeline')
        self.preprocessor = DataPreprocessor()
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

    async def load_and_preprocess_data(self,
                                     data: Any,
                                     symbol: str,
                                     timeframe: str = "15m",
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     preprocessing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load and preprocess data for regime detection.

        This method follows the same pattern as hmm_regime_discovery.py for
        data loading and preprocessing.

        Args:
            data: Market data (DataFrame or None to load from klines_parquet)
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            preprocessing_config: Configuration for preprocessing

        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            self.logger.info(f'🔄 Loading and preprocessing data for {symbol} {timeframe}')

            # Load market data
            market_data = await self._load_market_data(data, symbol, timeframe, start_date, end_date)

            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available: {symbol} {timeframe}")

            # Preprocess data
            processed_data = self.preprocessor.preprocess_data(market_data, preprocessing_config)

            # Create data quality report
            quality_report = self._generate_data_quality_report(processed_data)

            # Create pipeline metadata
            pipeline_metadata = self._create_pipeline_metadata(
                market_data, processed_data, symbol, timeframe, preprocessing_config
            )

            self.logger.info(f'✅ Data pipeline completed: {len(processed_data)} processed samples')

            return {
                'raw_data': market_data,
                'processed_data': processed_data,
                'quality_report': quality_report,
                'pipeline_metadata': pipeline_metadata,
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points': len(processed_data),
                'execution_info': {
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            }

        except Exception as e:
            self.logger.error(f'❌ Data pipeline failed: {e}')
            return {
                'raw_data': None,
                'processed_data': None,
                'quality_report': None,
                'pipeline_metadata': None,
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

    def _generate_data_quality_report(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a quality report for the processed data.

        Args:
            processed_data: Processed market data

        Returns:
            Dictionary containing quality metrics
        """
        try:
            if processed_data is None or processed_data.empty:
                return {'error': 'No data available for quality assessment'}

            report = {}

            # Basic data statistics
            report['total_samples'] = len(processed_data)
            report['total_features'] = len(processed_data.columns)
            report['date_range'] = {
                'start': str(processed_data.index.min()),
                'end': str(processed_data.index.max())
            }

            # Missing data analysis
            missing_data = processed_data.isnull().sum()
            report['missing_data'] = {
                'total_missing': int(missing_data.sum()),
                'missing_by_column': missing_data[missing_data > 0].to_dict()
            }

            # Data quality indicators
            report['data_completeness'] = (1 - missing_data.sum() / (len(processed_data) * len(processed_data.columns))) * 100

            # Statistical properties
            numeric_data = processed_data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                report['numeric_features'] = {
                    'mean': numeric_data.mean().to_dict(),
                    'std': numeric_data.std().to_dict(),
                    'min': numeric_data.min().to_dict(),
                    'max': numeric_data.max().to_dict()
                }

            # Data consistency checks
            report['consistency_checks'] = {
                'price_relationships_valid': self._validate_price_relationships(processed_data),
                'volume_positive': (processed_data.get('volume', 0) > 0).all(),
                'no_infinite_values': not np.any(np.isinf(processed_data.values)),
                'reasonable_price_ranges': self._validate_price_ranges(processed_data)
            }

            return report

        except Exception as e:
            self.logger.warning(f"⚠️ Data quality report generation failed: {e}")
            return {'error': str(e)}

    def _validate_price_relationships(self, data: pd.DataFrame) -> bool:
        """
        Validate price relationships (high >= low, etc.).

        Args:
            data: Market data

        Returns:
            True if relationships are valid
        """
        try:
            if 'high' not in data.columns or 'low' not in data.columns:
                return True  # Can't validate if columns missing

            return (data['high'] >= data['low']).all()
        except:
            return False

    def _validate_price_ranges(self, data: pd.DataFrame) -> bool:
        """
        Validate that price ranges are reasonable.

        Args:
            data: Market data

        Returns:
            True if price ranges are reasonable
        """
        try:
            price_columns = ['open', 'high', 'low', 'close']
            available_columns = [col for col in price_columns if col in data.columns]

            if not available_columns:
                return True

            # Check for extreme price movements (>50% in single period)
            for col in available_columns:
                if col in data.columns:
                    returns = data[col].pct_change().abs()
                    if (returns > 0.5).any():
                        return False

            return True
        except:
            return False

    def _create_pipeline_metadata(self,
                                raw_data: pd.DataFrame,
                                processed_data: pd.DataFrame,
                                symbol: str,
                                timeframe: str,
                                preprocessing_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create metadata about the data pipeline execution.

        Args:
            raw_data: Original market data
            processed_data: Processed data
            symbol: Trading symbol
            timeframe: Timeframe
            preprocessing_config: Preprocessing configuration

        Returns:
            Pipeline metadata dictionary
        """
        try:
            metadata = {
                'pipeline_info': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'raw_data_points': len(raw_data),
                    'processed_data_points': len(processed_data),
                    'data_reduction_ratio': len(processed_data) / max(len(raw_data), 1),
                    'preprocessing_applied': preprocessing_config is not None
                },
                'data_characteristics': {
                    'features': list(processed_data.columns),
                    'index_type': str(processed_data.index.dtype),
                    'memory_usage_mb': processed_data.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                'quality_indicators': {
                    'data_integrity_score': self._calculate_data_integrity_score(processed_data),
                    'feature_coverage': len(processed_data.columns) / 10,  # Assuming 10 expected features
                    'temporal_consistency': self._check_temporal_consistency(processed_data)
                }
            }

            if preprocessing_config:
                metadata['preprocessing_config'] = preprocessing_config

            return metadata

        except Exception as e:
            self.logger.warning(f"⚠️ Pipeline metadata creation failed: {e}")
            return {}

    def _calculate_data_integrity_score(self, data: pd.DataFrame) -> float:
        """
        Calculate a data integrity score.

        Args:
            data: Processed data

        Returns:
            Integrity score (0-1)
        """
        try:
            score = 1.0

            # Penalize for missing data
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            score -= missing_ratio * 0.3

            # Penalize for infinite values
            inf_count = np.isinf(data.values).sum()
            if inf_count > 0:
                score -= 0.2

            # Check for reasonable data ranges
            if 'close' in data.columns:
                price_range = data['close'].max() - data['close'].min()
                if price_range == 0:
                    score -= 0.3  # All prices same

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.warning(f"⚠️ Data integrity score calculation failed: {e}")
            return 0.5

    def _check_temporal_consistency(self, data: pd.DataFrame) -> float:
        """
        Check temporal consistency of the data.

        Args:
            data: Processed data

        Returns:
            Consistency score (0-1)
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return 0.5  # Neutral score for non-datetime index

            # Check for regular intervals
            time_diffs = data.index.to_series().diff().dropna()
            unique_diffs = time_diffs.unique()

            if len(unique_diffs) == 1:
                return 1.0  # Perfect regularity
            elif len(unique_diffs) <= 3:
                return 0.7  # Some variation but manageable
            else:
                return 0.3  # High irregularity

        except Exception as e:
            self.logger.warning(f"⚠️ Temporal consistency check failed: {e}")
            return 0.5

    def save_pipeline_state(self, pipeline_data: Dict[str, Any], output_path: str):
        """
        Save pipeline state to file.

        Args:
            pipeline_data: Pipeline data dictionary
            output_path: Path to save the data
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as JSON with metadata
            pipeline_state = {
                'timestamp': datetime.now().isoformat(),
                'pipeline_data': pipeline_data,
                'version': '1.0'
            }

            with open(output_path, 'w') as f:
                json.dump(pipeline_state, f, indent=2, default=str)

            self.logger.info(f"💾 Pipeline state saved to {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save pipeline state: {e}")

    def load_pipeline_state(self, input_path: str) -> Dict[str, Any]:
        """
        Load pipeline state from file.

        Args:
            input_path: Path to load the data from

        Returns:
            Pipeline data dictionary
        """
        try:
            with open(input_path, 'r') as f:
                pipeline_state = json.load(f)

            self.logger.info(f"📂 Pipeline state loaded from {input_path}")
            return pipeline_state.get('pipeline_data', {})

        except Exception as e:
            self.logger.error(f"❌ Failed to load pipeline state: {e}")
            return {}