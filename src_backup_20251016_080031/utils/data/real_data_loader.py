"""
Real Data Loading Utilities

This module provides utilities to load real market data instead of using synthetic/mock data.
It integrates with the existing data collection infrastructure to ensure we always use real data.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from src.utils.logger import system_logger
from src.training.steps.data_collection.unified_data_loader import UnifiedDataLoader
from src.training.steps.data_collection.unified_data_downloader import UnifiedDataDownloader
from src.utils.data.unified_data_utils import UnifiedDataUtils

logger = logging.getLogger(__name__)

class RealDataLoader:
    """
    Real data loader that ensures we never use synthetic/mock data.
    It attempts to load real data from various sources and downloads if necessary.
    """
    
    def __init__(self, data_dir: str = 'data/training'):
        """Initialize the real data loader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.logger = system_logger.getChild('RealDataLoader')
        self.data_dir = Path(data_dir)
        self.unified_loader = UnifiedDataLoader()
        self.data_downloader = UnifiedDataDownloader(data_dir)
        self.data_utils = UnifiedDataUtils()
        
    async def load_market_data(
        self,
        symbol: str = 'ETHUSDT',
        exchange: str = 'binance',
        timeframe: str = '1m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Load real market data, downloading if necessary.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_download: Force download even if data exists
            
        Returns:
            DataFrame with real market data
        """
        try:
            # First, try to load existing data
            if not force_download:
                self.logger.info(f"🔍 Looking for existing data: {symbol}/{exchange}/{timeframe}")
                existing_data = await self._load_existing_data(symbol, exchange, timeframe, start_date, end_date)
                if existing_data is not None and len(existing_data) > 0:
                    self.logger.info(f"✅ Loaded existing data: {len(existing_data)} rows")
                    return existing_data
            
            # If no existing data or force download, download new data
            self.logger.info(f"📥 Downloading real market data: {symbol}/{exchange}/{timeframe}")
            download_success = await self._download_real_data(symbol, exchange, timeframe, start_date, end_date)
            
            if download_success:
                # Try to load the newly downloaded data
                downloaded_data = await self._load_existing_data(symbol, exchange, timeframe, start_date, end_date)
                if downloaded_data is not None and len(downloaded_data) > 0:
                    self.logger.info(f"✅ Successfully loaded downloaded data: {len(downloaded_data)} rows")
                    return downloaded_data
            
            # If all else fails, raise an error instead of using synthetic data
            raise RuntimeError(
                f"❌ Failed to load real market data for {symbol}/{exchange}/{timeframe}. "
                "No synthetic data will be used. Please check your data sources and network connection."
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error loading real market data: {e}")
            raise
    
    async def _load_existing_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Load existing data from various possible locations."""
        try:
            # Try unified data loader first
            data = await self.unified_loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=str(self.data_dir),
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and len(data) > 0:
                return data
            
            # Try consolidated klines
            consolidated_path = self.data_dir / 'consolidated' / f'klines_{exchange}_{symbol}_{timeframe}.parquet'
            if consolidated_path.exists():
                self.logger.info(f"📊 Loading from consolidated file: {consolidated_path}")
                data = pd.read_parquet(consolidated_path)
                if len(data) > 0:
                    return data
            
            # Try individual klines files
            klines_dir = self.data_dir / 'klines' / exchange / symbol / timeframe
            if klines_dir.exists():
                parquet_files = list(klines_dir.glob('*.parquet'))
                if parquet_files:
                    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"📊 Loading from individual file: {latest_file}")
                    data = pd.read_parquet(latest_file)
                    if len(data) > 0:
                        return data
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Could not load existing data: {e}")
            return None
    
    async def _download_real_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> bool:
        """Download real market data."""
        try:
            # Set default date range if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"📥 Downloading data from {start_date} to {end_date}")
            
            # Use the unified data downloader
            success = await self.data_downloader.download_klines_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if success:
                self.logger.info("✅ Data download completed successfully")
                return True
            else:
                self.logger.error("❌ Data download failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error downloading data: {e}")
            return False
    
    def process_and_validate_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Process and validate the loaded data using unified data utils.
        
        Args:
            data: Raw data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Processed and validated DataFrame
        """
        try:
            # Use unified data utils for processing and validation
            processed_data, processing_report = self.data_utils.process_and_validate(
                data=data,
                validate_quality=True,
                clean_missing_values=True,
                detect_outliers=True,
                optimize_dtypes=True,
                regularize_timestamps=True,
                context=f"real_data_loading_{symbol}_{exchange}_{timeframe}",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe
            )
            
            self.logger.info(f"✅ Data processing completed: {processing_report['steps_completed']}")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Error processing data: {e}")
            # Return original data if processing fails
            return data

# Global instance for convenience
real_data_loader = RealDataLoader()