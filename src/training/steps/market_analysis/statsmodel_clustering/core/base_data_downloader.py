"""
Base Data Downloader for Statsmodel Clustering

This module provides a comprehensive base class for downloading market data
specifically for statsmodel clustering analysis. It integrates with the existing
data infrastructure while providing specialized functionality for clustering workflows.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np

# Import BaseStep
try:
    from src.training.base_step import BaseStep
    BASE_STEP_AVAILABLE = True
except ImportError:
    BASE_STEP_AVAILABLE = False
    BaseStep = None

# Import existing data infrastructure
try:
    from src.training.steps.data_collection.unified_data_downloader import UnifiedDataDownloader
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    DATA_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    DATA_INFRASTRUCTURE_AVAILABLE = False
    UnifiedDataDownloader = None
    HistoricalDataDownloader = None

# Import utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
    from src.utils.logger import system_logger
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    system_logger = logging.getLogger(__name__)

# Import standardized parquet handler
try:
    from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
    PARQUET_HANDLER_AVAILABLE = True
except ImportError:
    PARQUET_HANDLER_AVAILABLE = False
    standardized_parquet_handler = None


class BaseDataDownloader(BaseStep if BASE_STEP_AVAILABLE else ABC):
    """
    Abstract base class for data downloading in statsmodel clustering.
    
    This class provides a standardized interface for downloading market data
    with specific requirements for clustering analysis, including:
    - Data validation and quality checks
    - Standardized output formats
    - Integration with existing data infrastructure
    - Error handling and logging
    - Configurable data sources and timeframes
    - BaseStep inheritance for pipeline integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base data downloader.
        
        Args:
            config: Configuration dictionary with download parameters
        """
        # Initialize BaseStep if available
        if BASE_STEP_AVAILABLE:
            super().__init__(config)
        
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        
        # Extract common configuration
        self.symbol = config.get('symbol', 'ETHUSDT')
        self.exchange = config.get('exchange', 'BINANCE')
        self.timeframe = config.get('timeframe', '1h')
        self.data_dir = Path(config.get('data_dir', 'data_cache'))
        self.lookback_years = config.get('lookback_years', 2)
        self.force_download = config.get('force_download', False)
        
        # Initialize data infrastructure
        self.unified_downloader = None
        self.historical_downloader = None
        
        # Statistics tracking
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Initialize components
        self._initialize_infrastructure()
    
    def _initialize_infrastructure(self):
        """Initialize data infrastructure components."""
        tprint_info("🔧 Initializing data infrastructure components")
        
        if DATA_INFRASTRUCTURE_AVAILABLE:
            try:
                self.unified_downloader = UnifiedDataDownloader(str(self.data_dir))
                self.logger.info("✅ Unified data downloader initialized")
                tprint_success("✅ Unified data downloader initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize unified downloader: {e}")
                tprint_warning(f"⚠️ Failed to initialize unified downloader: {e}")
            
            try:
                self.historical_downloader = HistoricalDataDownloader(
                    data_dir=str(self.data_dir),
                    exchange=self.exchange.lower()
                )
                self.logger.info("✅ Historical data downloader initialized")
                tprint_success("✅ Historical data downloader initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize historical downloader: {e}")
                tprint_warning(f"⚠️ Failed to initialize historical downloader: {e}")
        else:
            self.logger.warning("⚠️ Data infrastructure not available")
            tprint_warning("⚠️ Data infrastructure not available")
    
    @abstractmethod
    async def download_data(self) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Download market data for clustering analysis.
        
        Returns:
            Tuple of (success, data, error_message)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate downloaded data for clustering requirements.
        
        Args:
            data: Downloaded market data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    def get_output_path(self) -> Path:
        """
        Get standardized output path for downloaded data.
        
        Returns:
            Path object for output file
        """
        filename = f"{self.symbol}_{self.exchange}_{self.timeframe}_clustering_data.parquet"
        output_path = self.data_dir / filename
        tprint_info(f"📁 Output path: {output_path}")
        return output_path
    
    def check_existing_data(self) -> Optional[pd.DataFrame]:
        """
        Check if valid data already exists.
        
        Returns:
            Existing data DataFrame if valid, None otherwise
        """
        tprint_info("🔍 Checking for existing data")
        output_path = self.get_output_path()
        
        if not output_path.exists() or self.force_download:
            tprint_info("📂 No existing data found or force download enabled")
            return None
        
        try:
            tprint_info(f"📖 Reading existing data from {output_path}")
            if PARQUET_HANDLER_AVAILABLE and standardized_parquet_handler:
                data = standardized_parquet_handler.read_parquet_standardized(output_path)
            else:
                data = pd.read_parquet(output_path)
            
            # Validate existing data
            tprint_info("🔍 Validating existing data")
            is_valid, errors = self.validate_data(data)
            
            if is_valid:
                tprint_info(f"📁 Found valid existing data: {len(data)} records")
                return data
            else:
                tprint_warning(f"⚠️ Existing data validation failed: {errors}")
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to read existing data: {e}")
            return None
    
    def save_data(self, data: pd.DataFrame) -> bool:
        """
        Save downloaded data to standardized format.
        
        Args:
            data: DataFrame to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving {len(data)} records to parquet format")
            output_path = self.get_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if PARQUET_HANDLER_AVAILABLE and standardized_parquet_handler:
                tprint_info("📝 Using standardized parquet handler")
                standardized_parquet_handler.write_parquet_standardized(data, output_path)
            else:
                tprint_info("📝 Using default parquet writer")
                data.to_parquet(output_path, index=True, compression='snappy')
            
            tprint_success(f"💾 Saved {len(data)} records to {output_path}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to save data: {e}")
            return False
    
    def update_stats(self, success: bool, record_count: int = 0):
        """
        Update download statistics.
        
        Args:
            success: Whether download was successful
            record_count: Number of records downloaded
        """
        self.download_stats['total_downloads'] += 1
        
        if success:
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_records'] += record_count
            tprint_info(f"📊 Updated stats: {record_count} records downloaded successfully")
        else:
            self.download_stats['failed_downloads'] += 1
            tprint_warning("📊 Updated stats: download failed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get download statistics.
        
        Returns:
            Dictionary with download statistics
        """
        tprint_info("📊 Retrieving download statistics")
        return self.download_stats.copy()
    
    def reset_stats(self):
        """Reset download statistics."""
        tprint_info("🔄 Resetting download statistics")
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }


class StandardDataDownloader(BaseDataDownloader):
    """
    Standard implementation of data downloader for statsmodel clustering.
    
    This class provides a concrete implementation that uses the existing
    data infrastructure to download market data suitable for clustering analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize standard data downloader.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Additional configuration for clustering
        self.min_data_points = config.get('min_data_points', 1000)
        self.required_columns = config.get('required_columns', ['open', 'high', 'low', 'close', 'volume'])
        self.data_quality_threshold = config.get('data_quality_threshold', 0.95)
    
    async def execute(self, data: Any) -> Any:
        """
        Execute the data download step (BaseStep interface).
        
        Args:
            data: Input data (not used for download step)
            
        Returns:
            Download result with data and metadata
        """
        success, downloaded_data, error = await self.download_data()
        
        return {
            'success': success,
            'data': downloaded_data,
            'error': error,
            'stats': self.get_stats(),
            'config': self.config
        }
    
    def validate_config(self) -> None:
        """
        Validate the configuration for the step (BaseStep interface).
        """
        tprint_info("🔍 Validating configuration")
        
        required_keys = ['symbol', 'exchange', 'timeframe']
        for key in required_keys:
            if key not in self.config:
                tprint_error(f"❌ Missing required config key: {key}")
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate symbol format
        symbol = self.config.get('symbol', '')
        if not symbol or not symbol.isupper():
            tprint_error(f"❌ Invalid symbol format: {symbol}")
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        # Validate exchange
        valid_exchanges = ['BINANCE', 'BYBIT', 'OKX', 'KRAKEN']
        exchange = self.config.get('exchange', '').upper()
        if exchange not in valid_exchanges:
            tprint_error(f"❌ Invalid exchange: {exchange}. Valid: {valid_exchanges}")
            raise ValueError(f"Invalid exchange: {exchange}. Valid: {valid_exchanges}")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        timeframe = self.config.get('timeframe', '')
        if timeframe not in valid_timeframes:
            tprint_error(f"❌ Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
        
        tprint_success("✅ Configuration validation passed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status and metrics of the step (BaseStep interface).
        
        Returns:
            Dictionary with status information
        """
        tprint_info("📊 Retrieving step status")
        status = {
            'step_name': 'data_download',
            'step_class': self.__class__.__name__,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'data_dir': str(self.data_dir),
            'lookback_years': self.lookback_years,
            'stats': self.get_stats(),
            'infrastructure_available': DATA_INFRASTRUCTURE_AVAILABLE
        }
        return status
    
    async def download_data(self) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Download market data using unified infrastructure.
        
        Returns:
            Tuple of (success, data, error_message)
        """
        self.download_stats['start_time'] = datetime.now()
        
        try:
            tprint_info(f"📥 Downloading data for {self.symbol} on {self.exchange} ({self.timeframe})")
            
            # Check existing data first
            existing_data = self.check_existing_data()
            if existing_data is not None:
                self.download_stats['end_time'] = datetime.now()
                self.update_stats(True, len(existing_data))
                return True, existing_data, None
            
            # Download new data
            if self.unified_downloader:
                success, raw_data, error = await self._download_with_unified()
            elif self.historical_downloader:
                success, raw_data, error = await self._download_with_historical()
            else:
                success, raw_data, error = False, None, "No data downloader available"
            
            if not success or raw_data is None:
                self.download_stats['end_time'] = datetime.now()
                self.update_stats(False)
                return False, None, error
            
            # Convert to DataFrame and process
            data = self._process_raw_data(raw_data)
            
            if data is None or len(data) == 0:
                self.download_stats['end_time'] = datetime.now()
                self.update_stats(False)
                return False, None, "No data after processing"
            
            # Validate data
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                self.download_stats['end_time'] = datetime.now()
                self.update_stats(False)
                return False, None, f"Data validation failed: {errors}"
            
            # Save data
            if not self.save_data(data):
                self.download_stats['end_time'] = datetime.now()
                self.update_stats(False)
                return False, None, "Failed to save data"
            
            self.download_stats['end_time'] = datetime.now()
            self.update_stats(True, len(data))
            
            tprint_success(f"✅ Successfully downloaded {len(data)} records")
            return True, data, None
            
        except Exception as e:
            self.download_stats['end_time'] = datetime.now()
            self.update_stats(False)
            error_msg = f"Download failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            return False, None, error_msg
    
    async def _download_with_unified(self) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """Download using unified data downloader."""
        tprint_info("🔄 Using unified data downloader")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)
            
            tprint_info(f"📅 Downloading data from {start_date.date()} to {end_date.date()}")
            
            success, data, error = await self.unified_downloader.download_klines(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if success:
                tprint_success(f"✅ Unified download successful: {len(data) if data else 0} records")
            else:
                tprint_error(f"❌ Unified download failed: {error}")
            
            return success, data, error
            
        except Exception as e:
            tprint_error(f"❌ Unified download exception: {e}")
            return False, None, str(e)
    
    async def _download_with_historical(self) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """Download using historical data downloader."""
        tprint_info("🔄 Using historical data downloader")
        
        try:
            tprint_info(f"📅 Downloading {self.lookback_years} years of historical data")
            
            success = await self.historical_downloader.download_historical_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                years=self.lookback_years
            )
            
            if not success:
                tprint_error("❌ Historical download failed")
                return False, None, "Historical download failed"
            
            # Read downloaded data
            tprint_info("📖 Reading downloaded files")
            files = self.historical_downloader.get_downloaded_files(self.symbol)
            if not files:
                tprint_error("❌ No downloaded files found")
                return False, None, "No downloaded files found"
            
            tprint_info(f"📁 Found {len(files)} files to process")
            
            # Combine all files
            all_data = []
            for file_path in files:
                try:
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        all_data.append(df)
                        tprint_info(f"📊 Loaded {len(df)} records from {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")
                    tprint_warning(f"⚠️ Failed to read {file_path}: {e}")
            
            if not all_data:
                tprint_error("❌ No valid data in downloaded files")
                return False, None, "No valid data in downloaded files"
            
            tprint_info("🔄 Combining data from all files")
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()
            
            # Convert to list of dicts
            tprint_info("🔄 Converting data to list format")
            data_list = []
            for idx, row in combined_data.iterrows():
                data_list.append({
                    'timestamp': int(idx.timestamp() * 1000),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            tprint_success(f"✅ Historical download successful: {len(data_list)} records")
            return True, data_list, None
            
        except Exception as e:
            tprint_error(f"❌ Historical download exception: {e}")
            return False, None, str(e)
    
    def _process_raw_data(self, raw_data: List[Dict]) -> Optional[pd.DataFrame]:
        """Process raw data into standardized DataFrame."""
        tprint_info("🔄 Processing raw data into standardized DataFrame")
        
        try:
            if not raw_data:
                tprint_warning("⚠️ No raw data to process")
                return None
            
            # Convert to DataFrame
            tprint_info(f"📊 Converting {len(raw_data)} records to DataFrame")
            df = pd.DataFrame(raw_data)
            
            # Convert timestamp to datetime and set as index
            if 'timestamp' in df.columns:
                tprint_info("🕐 Converting timestamps to datetime")
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp')
            
            # Ensure required columns exist and are numeric
            tprint_info("🔍 Validating required columns")
            for col in self.required_columns:
                if col not in df.columns:
                    self.logger.warning(f"Missing column: {col}")
                    tprint_error(f"❌ Missing required column: {col}")
                    return None
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values in required columns
            initial_count = len(df)
            df = df.dropna(subset=self.required_columns)
            removed_count = initial_count - len(df)
            if removed_count > 0:
                tprint_info(f"🧹 Removed {removed_count} rows with NaN values")
            
            # Sort by timestamp
            tprint_info("🕐 Sorting by timestamp")
            df = df.sort_index()
            
            # Remove duplicates
            initial_count = len(df)
            df = df[~df.index.duplicated(keep='last')]
            removed_count = initial_count - len(df)
            if removed_count > 0:
                tprint_info(f"🧹 Removed {removed_count} duplicate records")
            
            tprint_success(f"✅ Data processing complete: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to process raw data: {e}")
            tprint_error(f"❌ Failed to process raw data: {e}")
            return None
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data for clustering requirements."""
        tprint_info("🔍 Validating data for clustering requirements")
        errors = []
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            error_msg = f"Insufficient data points: {len(data)} < {self.min_data_points}"
            errors.append(error_msg)
            tprint_error(f"❌ {error_msg}")
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            errors.append(error_msg)
            tprint_error(f"❌ {error_msg}")
        
        # Check for NaN values
        tprint_info("🔍 Checking for NaN values")
        nan_counts = data[self.required_columns].isnull().sum()
        high_nan_cols = nan_counts[nan_counts > len(data) * (1 - self.data_quality_threshold)]
        if not high_nan_cols.empty:
            error_msg = f"High NaN ratio in columns: {dict(high_nan_cols)}"
            errors.append(error_msg)
            tprint_error(f"❌ {error_msg}")
        
        # Check for zero/negative prices
        tprint_info("💰 Checking for valid price values")
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                if (data[col] <= 0).any():
                    error_msg = f"Non-positive values in {col}"
                    errors.append(error_msg)
                    tprint_error(f"❌ {error_msg}")
        
        # Check for negative volume
        tprint_info("📊 Checking for valid volume values")
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                error_msg = "Negative volume values"
                errors.append(error_msg)
                tprint_error(f"❌ {error_msg}")
        
        # Check data continuity (gaps)
        if len(data) > 1:
            tprint_info("🕐 Checking data continuity")
            time_diffs = data.index.to_series().diff().dropna()
            expected_interval = pd.Timedelta(self._get_expected_interval())
            large_gaps = time_diffs > expected_interval * 2
            if large_gaps.sum() > len(data) * 0.01:  # More than 1% gaps
                error_msg = f"Excessive data gaps: {large_gaps.sum()} gaps found"
                errors.append(error_msg)
                tprint_error(f"❌ {error_msg}")
        
        if len(errors) == 0:
            tprint_success("✅ Data validation passed")
        else:
            tprint_error(f"❌ Data validation failed with {len(errors)} errors")
        
        return len(errors) == 0, errors
    
    def _get_expected_interval(self) -> str:
        """Get expected time interval for the timeframe."""
        interval_map = {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day'
        }
        return interval_map.get(self.timeframe, '1 hour')


def create_data_downloader(config: Dict[str, Any]) -> BaseDataDownloader:
    """
    Factory function to create appropriate data downloader.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured data downloader instance
    """
    tprint_info("🏭 Creating data downloader with factory function")
    
    downloader_type = config.get('downloader_type', 'standard')
    tprint_info(f"📊 Downloader type: {downloader_type}")
    
    if downloader_type == 'standard':
        downloader = StandardDataDownloader(config)
        tprint_success("✅ Standard data downloader created")
        return downloader
    else:
        tprint_error(f"❌ Unknown downloader type: {downloader_type}")
        raise ValueError(f"Unknown downloader type: {downloader_type}")


# Convenience function for quick usage
async def download_clustering_data(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    timeframe: str = "1h",
    lookback_years: int = 2,
    data_dir: str = "data_cache",
    force_download: bool = False
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Convenience function to download data for clustering analysis.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        lookback_years: Years of historical data
        data_dir: Data directory
        force_download: Force re-download even if data exists
        
    Returns:
        Tuple of (success, data, error_message)
    """
    tprint_info("🚀 Convenience function: downloading clustering data")
    
    config = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'lookback_years': lookback_years,
        'data_dir': data_dir,
        'force_download': force_download,
        'downloader_type': 'standard'
    }
    
    tprint_info(f"📊 Configuration: {symbol} on {exchange} ({timeframe}) for {lookback_years} years")
    
    downloader = create_data_downloader(config)
    result = await downloader.download_data()
    
    if result[0]:
        tprint_success(f"✅ Successfully downloaded clustering data: {len(result[1]) if result[1] else 0} records")
    else:
        tprint_error(f"❌ Failed to download clustering data: {result[2]}")
    
    return result