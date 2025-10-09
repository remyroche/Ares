"""
Ares Launcher Data Loading Utility

This module provides data loading utilities that respect the ares_launcher
configuration, ensuring that the lookback period (20 days in "light" mode)
is properly applied when loading data using KlinesParquetManager or other utilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)
from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
from src.config.pipeline_modes import get_light_mode_config, get_blank_mode_config, get_full_mode_config


class AresLauncherDataLoader:
    """
    Data loader that respects ares_launcher configuration and lookback periods.
    
    This class ensures that the lookback period specified in ares_launcher
    (e.g., 20 days in "light" mode) is properly applied when loading data.
    """
    
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the ares launcher data loader."""
        self.data_dir = data_dir
        self.klines_manager = get_klines_manager(data_dir)
        
        tprint("🚀 Initializing Ares Launcher Data Loader")
        tprint_info(f"📁 Data directory: {data_dir}")
    
    def get_lookback_dates(self, mode: str = "light") -> tuple[datetime, datetime]:
        """
        Get start and end dates based on ares_launcher mode configuration.
        
        Args:
            mode: Pipeline mode ("light", "blank", "full")
            
        Returns:
            Tuple of (start_date, end_date)
        """
        # Get mode configuration
        if mode == "light":
            config = get_light_mode_config()
        elif mode == "blank":
            config = get_blank_mode_config()
        elif mode == "full":
            config = get_full_mode_config()
        else:
            tprint_warning(f"⚠️ Unknown mode '{mode}', defaulting to light mode")
            config = get_light_mode_config()
        
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.lookback_days)
        
        tprint_info(f"📅 {mode.upper()} mode: {config.lookback_days} days lookback")
        tprint_info(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        
        return start_date, end_date
    
    def load_data_with_mode(
        self,
        symbol: str,
        interval: str,
        mode: str = "light",
        data_type: str = "raw",
        columns: Optional[List[str]] = None,
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Load data respecting ares_launcher mode configuration.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            mode: Pipeline mode ("light", "blank", "full")
            data_type: 'raw' or 'processed'
            columns: List of columns to read
            custom_start_date: Override start date
            custom_end_date: Override end date
            
        Returns:
            Loaded DataFrame or None
        """
        tprint(f"📊 Loading data for {symbol} ({interval}) in {mode.upper()} mode")
        
        # Get dates based on mode
        if custom_start_date is not None and custom_end_date is not None:
            start_date = custom_start_date
            end_date = custom_end_date
            tprint_info(f"📅 Using custom date range: {start_date.date()} to {end_date.date()}")
        else:
            start_date, end_date = self.get_lookback_dates(mode)
        
        try:
            # Load data using KlinesParquetManager
            data = self.klines_manager.read_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type,
                columns=columns
            )
            
            if data is not None and not data.empty:
                tprint_success(f"✅ Loaded {len(data)} records for {symbol} ({interval})")
                tprint_info(f"📅 Data range: {data.index.min().date()} to {data.index.max().date()}")
                return data
            else:
                tprint_warning(f"⚠️ No data found for {symbol} ({interval}) in {mode.upper()} mode")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to load data for {symbol} ({interval}): {e}")
            return None
    
    async def load_data_async(
        self,
        symbol: str,
        interval: str,
        mode: str = "light",
        data_type: str = "raw",
        columns: Optional[List[str]] = None,
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Asynchronously load data respecting ares_launcher mode configuration.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            mode: Pipeline mode ("light", "blank", "full")
            data_type: 'raw' or 'processed'
            columns: List of columns to read
            custom_start_date: Override start date
            custom_end_date: Override end date
            
        Returns:
            Loaded DataFrame or None
        """
        tprint(f"📊 Loading data asynchronously for {symbol} ({interval}) in {mode.upper()} mode")
        
        # Get dates based on mode
        if custom_start_date is not None and custom_end_date is not None:
            start_date = custom_start_date
            end_date = custom_end_date
            tprint_info(f"📅 Using custom date range: {start_date.date()} to {end_date.date()}")
        else:
            start_date, end_date = self.get_lookback_dates(mode)
        
        try:
            # Use asyncio to run the synchronous load in a thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self.klines_manager.read_data,
                symbol,
                interval,
                start_date,
                end_date,
                data_type,
                columns
            )
            
            if data is not None and not data.empty:
                tprint_success(f"✅ Loaded {len(data)} records for {symbol} ({interval})")
                tprint_info(f"📅 Data range: {data.index.min().date()} to {data.index.max().date()}")
                return data
            else:
                tprint_warning(f"⚠️ No data found for {symbol} ({interval}) in {mode.upper()} mode")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to load data for {symbol} ({interval}): {e}")
            return None
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        mode: str = "light",
        data_type: str = "raw",
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load data for multiple symbols respecting ares_launcher mode configuration.
        
        Args:
            symbols: List of trading symbols
            interval: Data interval
            mode: Pipeline mode ("light", "blank", "full")
            data_type: 'raw' or 'processed'
            columns: List of columns to read
            
        Returns:
            Dictionary mapping symbols to loaded DataFrames
        """
        tprint(f"📊 Loading data for {len(symbols)} symbols in {mode.upper()} mode")
        
        results = {}
        start_date, end_date = self.get_lookback_dates(mode)
        
        for symbol in symbols:
            tprint_debug(f"📊 Loading {symbol}...")
            data = self.load_data_with_mode(
                symbol=symbol,
                interval=interval,
                mode=mode,
                data_type=data_type,
                columns=columns,
                custom_start_date=start_date,
                custom_end_date=end_date
            )
            results[symbol] = data
        
        successful_loads = sum(1 for data in results.values() if data is not None)
        tprint_info(f"📊 Successfully loaded {successful_loads}/{len(symbols)} symbols")
        
        return results
    
    def get_available_data_info(
        self,
        symbol: str,
        interval: str,
        mode: str = "light"
    ) -> Dict[str, Any]:
        """
        Get information about available data for a symbol in the specified mode.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            mode: Pipeline mode ("light", "blank", "full")
            
        Returns:
            Dictionary with data information
        """
        tprint(f"📊 Getting data info for {symbol} ({interval}) in {mode.upper()} mode")
        
        # Get mode configuration
        if mode == "light":
            config = get_light_mode_config()
        elif mode == "blank":
            config = get_blank_mode_config()
        elif mode == "full":
            config = get_full_mode_config()
        else:
            config = get_light_mode_config()
        
        # Get basic data info
        info = self.klines_manager.get_data_info(symbol, interval, "raw")
        
        # Add mode-specific information
        start_date, end_date = self.get_lookback_dates(mode)
        
        info.update({
            'mode': mode,
            'lookback_days': config.lookback_days,
            'requested_start_date': start_date.isoformat(),
            'requested_end_date': end_date.isoformat(),
            'mode_description': config.description
        })
        
        return info
    
    def validate_data_availability(
        self,
        symbol: str,
        interval: str,
        mode: str = "light"
    ) -> bool:
        """
        Validate that sufficient data is available for the specified mode.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            mode: Pipeline mode ("light", "blank", "full")
            
        Returns:
            True if sufficient data is available, False otherwise
        """
        tprint(f"🔍 Validating data availability for {symbol} ({interval}) in {mode.upper()} mode")
        
        # Get data info
        info = self.get_available_data_info(symbol, interval, mode)
        
        if not info.get('available', False):
            tprint_warning(f"⚠️ No data available for {symbol} ({interval})")
            return False
        
        # Check if we have enough data for the requested lookback period
        config = get_light_mode_config() if mode == "light" else get_blank_mode_config() if mode == "blank" else get_full_mode_config()
        required_days = config.lookback_days
        
        if info.get('total_records', 0) == 0:
            tprint_warning(f"⚠️ No records found for {symbol} ({interval})")
            return False
        
        # Estimate days based on interval
        interval_minutes = self._get_interval_minutes(interval)
        estimated_days = (info.get('total_records', 0) * interval_minutes) / (24 * 60)
        
        if estimated_days < required_days:
            tprint_warning(f"⚠️ Insufficient data: {estimated_days:.1f} days available, {required_days} days required")
            return False
        
        tprint_success(f"✅ Sufficient data available: {estimated_days:.1f} days")
        return True
    
    def _get_interval_minutes(self, interval: str) -> int:
        """Get interval in minutes."""
        interval_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return interval_map.get(interval, 1)


# Convenience functions for easy usage
def load_data_with_ares_mode(
    symbol: str,
    interval: str,
    mode: str = "light",
    data_dir: str = "historical_data",
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to load data with ares_launcher mode configuration.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        mode: Pipeline mode ("light", "blank", "full")
        data_dir: Data directory
        **kwargs: Additional arguments for KlinesParquetManager.read_data
        
    Returns:
        Loaded DataFrame or None
    """
    loader = AresLauncherDataLoader(data_dir)
    return loader.load_data_with_mode(symbol, interval, mode, **kwargs)


async def load_data_async_with_ares_mode(
    symbol: str,
    interval: str,
    mode: str = "light",
    data_dir: str = "historical_data",
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to asynchronously load data with ares_launcher mode configuration.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        mode: Pipeline mode ("light", "blank", "full")
        data_dir: Data directory
        **kwargs: Additional arguments for KlinesParquetManager.read_data
        
    Returns:
        Loaded DataFrame or None
    """
    loader = AresLauncherDataLoader(data_dir)
    return await loader.load_data_async(symbol, interval, mode, **kwargs)


def get_ares_mode_dates(mode: str = "light") -> tuple[datetime, datetime]:
    """
    Get start and end dates for a specific ares_launcher mode.
    
    Args:
        mode: Pipeline mode ("light", "blank", "full")
        
    Returns:
        Tuple of (start_date, end_date)
    """
    loader = AresLauncherDataLoader()
    return loader.get_lookback_dates(mode)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example usage
        loader = AresLauncherDataLoader()
        
        # Load data in light mode (20 days)
        data_light = loader.load_data_with_mode("ETHUSDT", "15m", "light")
        print(f"Light mode data shape: {data_light.shape if data_light is not None else 'None'}")
        
        # Load data in blank mode (180 days)
        data_blank = loader.load_data_with_mode("ETHUSDT", "15m", "blank")
        print(f"Blank mode data shape: {data_blank.shape if data_blank is not None else 'None'}")
        
        # Load data asynchronously
        data_async = await loader.load_data_async("ETHUSDT", "15m", "light")
        print(f"Async data shape: {data_async.shape if data_async is not None else 'None'}")
        
        # Validate data availability
        is_available = loader.validate_data_availability("ETHUSDT", "15m", "light")
        print(f"Data available: {is_available}")
    
    asyncio.run(main())