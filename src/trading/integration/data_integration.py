"""
Data Integration

Integration utilities for sharing data between trading and training pipelines.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.validation import validate_market_data

logger = system_logger.getChild('DataIntegration')

class TradingDataExporter:
    """
    Exports trading data for use in training pipeline.
    """

    def __init__(self):
        self.logger = logger.getChild('TradingDataExporter')

class DataSyncManager:
    """
    Manages data synchronization between trading and training pipelines.
    """

    def __init__(self):
        self.logger = logger.getChild('DataSyncManager')
        self.sync_status: Dict[str, Any] = {}
        self.last_sync: Optional[datetime] = None

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def sync_market_data(
        self,
        trading_data: pd.DataFrame,
        symbol: str = "ETHUSDT",
        timeframe: str = "1m"
    ) -> bool:
        """
        Sync market data from trading to training pipeline.

        Args:
            trading_data: Market data from trading operations
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            True if sync successful
        """
        tprint_info(f"🔄 Syncing market data for {symbol} ({timeframe})")

        try:
            # Validate input data
            validate_market_data(trading_data)

            # Prepare data for training pipeline format
            training_format_data = await self._convert_to_training_format(
                trading_data, symbol, timeframe
            )

            # Save to training pipeline data location
            await self._save_to_training_data_store(
                training_format_data, symbol, timeframe, "market_data"
            )

            # Update sync status
            self.sync_status[f"market_data_{symbol}_{timeframe}"] = {
                'last_sync': datetime.now(),
                'records_synced': len(trading_format_data),
                'status': 'success'
            }

            tprint_success(f"✅ Synced {len(trading_format_data)} market data records")
            return True

        except Exception as e:
            tprint_error(f"❌ Market data sync failed: {e}")
            self.sync_status[f"market_data_{symbol}_{timeframe}"] = {
                'last_sync': datetime.now(),
                'status': 'failed',
                'error': str(e)
            }
            return False

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def sync_trading_decisions(
        self,
        decisions: List[Dict[str, Any]],
        symbol: str = "ETHUSDT"
    ) -> bool:
        """
        Sync trading decisions to training pipeline for model improvement.

        Args:
            decisions: List of trading decisions
            symbol: Trading symbol

        Returns:
            True if sync successful
        """
        tprint_info(f"🔄 Syncing trading decisions for {symbol}")

        try:
            if not decisions:
                tprint_warning("⚠️ No trading decisions to sync")
                return True

            # Convert to DataFrame for easier handling
            decisions_df = pd.DataFrame(decisions)

            # Add metadata
            decisions_df['sync_timestamp'] = datetime.now()
            decisions_df['data_source'] = 'trading_live'
            decisions_df['symbol'] = symbol

            # Save to training pipeline data location
            await self._save_to_training_data_store(
                decisions_df, symbol, "live", "trading_decisions"
            )

            # Update sync status
            self.sync_status[f"trading_decisions_{symbol}"] = {
                'last_sync': datetime.now(),
                'records_synced': len(decisions),
                'status': 'success'
            }

            tprint_success(f"✅ Synced {len(decisions)} trading decisions")
            return True

        except Exception as e:
            tprint_error(f"❌ Trading decisions sync failed: {e}")
            self.sync_status[f"trading_decisions_{symbol}"] = {
                'last_sync': datetime.now(),
                'status': 'failed',
                'error': str(e)
            }
            return False

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def sync_performance_metrics(
        self,
        metrics: Dict[str, Any],
        symbol: str = "ETHUSDT",
        timeframe: str = "1m"
    ) -> bool:
        """
        Sync performance metrics to training pipeline.

        Args:
            metrics: Performance metrics dictionary
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            True if sync successful
        """
        tprint_info(f"🔄 Syncing performance metrics for {symbol}")

        try:
            # Add metadata to metrics
            enhanced_metrics = {
                **metrics,
                'sync_timestamp': datetime.now().isoformat(),
                'data_source': 'trading_live',
                'symbol': symbol,
                'timeframe': timeframe
            }

            # Save to training pipeline data location
            await self._save_performance_metrics(enhanced_metrics, symbol, timeframe)

            # Update sync status
            self.sync_status[f"performance_metrics_{symbol}"] = {
                'last_sync': datetime.now(),
                'status': 'success',
                'metrics_count': len(metrics)
            }

            tprint_success(f"✅ Synced performance metrics ({len(metrics)} metrics)")
            return True

        except Exception as e:
            tprint_error(f"❌ Performance metrics sync failed: {e}")
            self.sync_status[f"performance_metrics_{symbol}"] = {
                'last_sync': datetime.now(),
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def _convert_to_training_format(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Convert trading data to training pipeline format."""
        try:
            # Create a copy to avoid modifying original
            formatted_data = data.copy()

            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in formatted_data.columns:
                    tprint_warning(f"⚠️ Missing required column: {col}")

            # Add metadata columns
            formatted_data['symbol'] = symbol
            formatted_data['timeframe'] = timeframe
            formatted_data['data_source'] = 'trading_live'
            formatted_data['created_at'] = datetime.now()

            # Ensure timestamp is datetime
            if 'timestamp' in formatted_data.columns:
                formatted_data['timestamp'] = pd.to_datetime(formatted_data['timestamp'])

            # Sort by timestamp
            if 'timestamp' in formatted_data.columns:
                formatted_data = formatted_data.sort_values('timestamp')

            return formatted_data

        except Exception as e:
            tprint_error(f"❌ Data format conversion failed: {e}")
            return data

    async def _save_to_training_data_store(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        data_type: str
    ):
        """Save data to training pipeline data store."""
        try:
            import os

            # Create training data directory structure
            base_dir = "data_cache/training_sync"
            data_dir = os.path.join(base_dir, data_type, symbol, timeframe)
            os.makedirs(data_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{data_type}_{symbol}_{timeframe}_{timestamp}.parquet"
            filepath = os.path.join(data_dir, filename)

            # Save as parquet for efficiency
            data.to_parquet(filepath, index=False)

            tprint_info(f"💾 Saved {data_type} data to {filepath}")

        except Exception as e:
            tprint_error(f"❌ Failed to save to training data store: {e}")

    async def _save_performance_metrics(
        self,
        metrics: Dict[str, Any],
        symbol: str,
        timeframe: str
    ):
        """Save performance metrics to training pipeline."""
        try:
            import json

            # Create metrics directory
            base_dir = "data_cache/training_sync"
            metrics_dir = os.path.join(base_dir, "performance_metrics", symbol)
            os.makedirs(metrics_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_metrics_{symbol}_{timeframe}_{timestamp}.json"
            filepath = os.path.join(metrics_dir, filename)

            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

            tprint_info(f"💾 Saved performance metrics to {filepath}")

        except Exception as e:
            tprint_error(f"❌ Failed to save performance metrics: {e}")

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        return {
            'sync_status': self.sync_status,
            'last_sync': self.last_sync,
            'active_syncs': len([s for s in self.sync_status.values() if s.get('status') == 'success'])
        }

class TrainingDataReader:
    """
    Reads data from training pipeline for use in trading operations.
    """

    def __init__(self):
        self.logger = logger.getChild('TrainingDataReader')

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def read_training_features(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "1m",
        lookback_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Read feature data from training pipeline.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            lookback_days: Number of days to look back

        Returns:
            DataFrame with feature data or None if not available
        """
        tprint_info(f"📖 Reading training features for {symbol} ({timeframe})")

        try:
            # Try to read from training pipeline feature store
            features_df = await self._read_from_training_store(
                "features", symbol, timeframe, lookback_days
            )

            if features_df is not None and not features_df.empty:
                tprint_success(f"✅ Read {len(features_df)} feature records")
                return features_df
            else:
                tprint_warning("⚠️ No training features found")
                return None

        except Exception as e:
            tprint_error(f"❌ Failed to read training features: {e}")
            return None

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def read_training_targets(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "1m",
        lookback_days: int = 30
    ) -> Optional[pd.Series]:
        """
        Read target data from training pipeline.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            lookback_days: Number of days to look back

        Returns:
            Series with target data or None if not available
        """
        tprint_info(f"📖 Reading training targets for {symbol} ({timeframe})")

        try:
            # Try to read from training pipeline target store
            targets_df = await self._read_from_training_store(
                "targets", symbol, timeframe, lookback_days
            )

            if targets_df is not None and not targets_df.empty:
                # Extract target column
                target_columns = ['target', 'label', 'y', 'success']
                target_column = None

                for col in target_columns:
                    if col in targets_df.columns:
                        target_column = col
                        break

                if target_column:
                    targets_series = targets_df[target_column]
                    tprint_success(f"✅ Read {len(targets_series)} target records")
                    return targets_series
                else:
                    tprint_warning("⚠️ No target column found in training targets")
                    return None
            else:
                tprint_warning("⚠️ No training targets found")
                return None

        except Exception as e:
            tprint_error(f"❌ Failed to read training targets: {e}")
            return None

    async def _read_from_training_store(
        self,
        data_type: str,
        symbol: str,
        timeframe: str,
        lookback_days: int
    ) -> Optional[pd.DataFrame]:
        """Read data from training pipeline data store."""
        try:
            import glob

            # Look for data in training pipeline directories
            possible_dirs = [
                f"data_cache/training_sync/{data_type}/{symbol}/{timeframe}",
                f"data_cache/{data_type}/{symbol}_{timeframe}",
                f"data/{data_type}/{symbol}",
                f"training_data/{data_type}"
            ]

            # Find the most recent data files
            all_files = []
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    pattern = os.path.join(dir_path, "*.parquet")
                    files = glob.glob(pattern)
                    all_files.extend(files)

            if not all_files:
                tprint_warning(f"⚠️ No {data_type} files found for {symbol}")
                return None

            # Sort by modification time (most recent first)
            all_files.sort(key=os.path.getmtime, reverse=True)

            # Read and combine recent files
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            combined_data = []

            for file_path in all_files[:10]:  # Limit to 10 most recent files
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time >= cutoff_date:
                        df = pd.read_parquet(file_path)
                        combined_data.append(df)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to read file {file_path}: {e}")
                    continue

            if combined_data:
                # Combine all dataframes
                result_df = pd.concat(combined_data, ignore_index=True)

                # Remove duplicates if timestamp column exists
                if 'timestamp' in result_df.columns:
                    result_df = result_df.drop_duplicates(subset=['timestamp'])
                    result_df = result_df.sort_values('timestamp')

                return result_df
            else:
                return None

        except Exception as e:
            tprint_error(f"❌ Failed to read from training store: {e}")
            return None

# Global instances
data_sync_manager = DataSyncManager()
training_data_reader = TrainingDataReader()

# Convenience functions
async def sync_all_trading_data(
    market_data: pd.DataFrame,
    trading_decisions: List[Dict[str, Any]],
    performance_metrics: Dict[str, Any],
    symbol: str = "ETHUSDT",
    timeframe: str = "1m"
) -> Dict[str, bool]:
    """Sync all trading data with training pipeline."""
    tprint_info("🔄 Syncing all trading data with training pipeline...")

    results = {}

    # Sync market data
    results['market_data'] = await data_sync_manager.sync_market_data(
        market_data, symbol, timeframe
    )

    # Sync trading decisions
    results['trading_decisions'] = await data_sync_manager.sync_trading_decisions(
        trading_decisions, symbol
    )

    # Sync performance metrics
    results['performance_metrics'] = await data_sync_manager.sync_performance_metrics(
        performance_metrics, symbol, timeframe
    )

    successful_syncs = sum(results.values())
    tprint_success(f"✅ Successfully synced {successful_syncs}/3 data types")

    return results

async def read_all_training_data(
    symbol: str = "ETHUSDT",
    timeframe: str = "1m",
    lookback_days: int = 30
) -> Dict[str, Optional[pd.DataFrame]]:
    """Read all available training data."""
    tprint_info("📖 Reading all available training data...")

    results = {}

    # Read features
    results['features'] = await training_data_reader.read_training_features(
        symbol, timeframe, lookback_days
    )

    # Read targets
    results['targets'] = await training_data_reader.read_training_targets(
        symbol, timeframe, lookback_days
    )

    available_data = sum(1 for v in results.values() if v is not None)
    tprint_success(f"✅ Successfully read {available_data}/2 data types")

    return results
