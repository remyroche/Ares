"""
Historical Data Downloader for Binance Klines

This module provides tools to download historical klines data from Binance
and save it as optimized monthly parquet files.
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from exchange.binance import BinanceExchange
from src.utils.logger import system_logger
from src.utils.parquet_utils import ParquetUtils
from src.utils.data.processing.data_processing import DataProcessor


class HistoricalDataDownloader:
    """Download and manage historical Binance klines data."""

    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the historical data downloader.
        
        Args:
            data_dir: Base directory for storing historical data
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "binance"
        self.logger = system_logger.getChild("HistoricalDataDownloader")
        self.parquet_utils = ParquetUtils()
        self.data_processor = DataProcessor()
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    async def download_historical_klines(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        years: int = 3,
        api_key: str = "",
        api_secret: str = ""
    ) -> bool:
        """Download historical klines data for the specified symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            years: Number of years of historical data to download
            api_key: Binance API key
            api_secret: Binance API secret
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"🚀 Starting historical data download for {symbol}")
            self.logger.info(f"📅 Downloading {years} years of {interval} data")
            
            # Initialize Binance exchange
            exchange = BinanceExchange(api_key, api_secret, symbol)
            await exchange._initialize_exchange()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Create symbol directory
            symbol_dir = self.raw_data_dir / symbol.lower() / "raw"
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Download data month by month
            current_date = start_date
            total_months = years * 12
            downloaded_months = 0
            
            while current_date < end_date:
                # Calculate month end
                if current_date.month == 12:
                    month_end = current_date.replace(year=current_date.year + 1, month=1, day=1) - timedelta(days=1)
                else:
                    month_end = current_date.replace(month=current_date.month + 1, day=1) - timedelta(days=1)
                
                # Don't go beyond end_date
                month_end = min(month_end, end_date)
                
                # Download month data
                success = await self._download_month_data(
                    exchange, symbol, interval, current_date, month_end, symbol_dir
                )
                
                if success:
                    downloaded_months += 1
                    self.logger.info(f"✅ Downloaded {current_date.strftime('%Y-%m')} ({downloaded_months}/{total_months})")
                else:
                    self.logger.warning(f"⚠️ Failed to download {current_date.strftime('%Y-%m')}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
            
            await exchange.close()
            
            self.logger.info(f"🎉 Historical data download completed: {downloaded_months}/{total_months} months")
            return downloaded_months > 0
            
        except Exception as e:
            self.logger.exception(f"❌ Historical data download failed: {e}")
            return False
    
    async def _download_month_data(
        self,
        exchange: BinanceExchange,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        symbol_dir: Path
    ) -> bool:
        """Download data for a specific month.
        
        Args:
            exchange: Binance exchange instance
            symbol: Trading symbol
            interval: Kline interval
            start_date: Start date for the month
            end_date: End date for the month
            symbol_dir: Directory to save the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to milliseconds
            start_time_ms = int(start_date.timestamp() * 1000)
            end_time_ms = int(end_date.timestamp() * 1000)
            
            # Download raw klines data
            raw_data = await exchange._get_historical_klines_raw(
                symbol, interval, start_time_ms, end_time_ms, 1000
            )
            
            if not raw_data:
                self.logger.warning(f"No data received for {start_date.strftime('%Y-%m')}")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add metadata
            df['symbol'] = symbol
            df['interval'] = interval
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            
            # Optimize data types
            df = self.data_processor.optimize_dataframe_dtypes(df)
            
            # Save as parquet
            filename = f"{symbol.lower()}_{interval}_{start_date.strftime('%Y_%m')}.parquet"
            filepath = symbol_dir / filename
            
            df.to_parquet(filepath, index=True, compression='snappy')
            
            self.logger.info(f"💾 Saved {len(df)} records to {filename}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to download month data for {start_date.strftime('%Y-%m')}: {e}")
            return False
    
    def get_downloaded_files(self, symbol: str) -> List[Path]:
        """Get list of downloaded files for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of file paths
        """
        symbol_dir = self.raw_data_dir / symbol.lower() / "raw"
        if not symbol_dir.exists():
            return []
        
        return list(symbol_dir.glob("*.parquet"))
    
    def get_data_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of downloaded data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with data summary
        """
        files = self.get_downloaded_files(symbol)
        
        if not files:
            return {"files_count": 0, "total_records": 0, "date_range": None}
        
        total_records = 0
        date_ranges = []
        
        for file_path in files:
            try:
                df = self.parquet_utils.safe_read_parquet(str(file_path))
                if df is not None and not df.empty:
                    total_records += len(df)
                    date_ranges.append((df.index.min(), df.index.max()))
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")
        
        summary = {
            "files_count": len(files),
            "total_records": total_records,
            "date_range": None
        }
        
        if date_ranges:
            min_date = min(dt[0] for dt in date_ranges)
            max_date = max(dt[1] for dt in date_ranges)
            summary["date_range"] = (min_date, max_date)
        
        return summary


# Convenience functions
async def download_ethusdt_historical_data(
    years: int = 3,
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = ""
) -> bool:
    """Download 3 years of ETHUSDT historical data.
    
    Args:
        years: Number of years to download
        data_dir: Base directory for data storage
        api_key: Binance API key
        api_secret: Binance API secret
        
    Returns:
        True if successful, False otherwise
    """
    downloader = HistoricalDataDownloader(data_dir)
    return await downloader.download_historical_klines(
        symbol="ETHUSDT",
        interval="1m",
        years=years,
        api_key=api_key,
        api_secret=api_secret
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        downloader = HistoricalDataDownloader()
        success = await downloader.download_historical_klines(
            symbol="ETHUSDT",
            interval="1m",
            years=3
        )
        
        if success:
            summary = downloader.get_data_summary("ETHUSDT")
            print(f"Download completed: {summary}")
        else:
            print("Download failed")
    
    asyncio.run(main())