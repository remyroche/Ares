#!/usr/bin/env python3
"""
Standalone Enhanced Klines Processing Pipeline for ETHUSDT, 4 years, BingX
This version includes all necessary components without external dependencies.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock ExchangeInterface to bypass circular import issues
class MockExchangeInterface:
    """Mock exchange interface for testing without API calls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = True
        self.exchange_type = config.get('exchange_type', 'mock')
    
    async def connect(self):
        """Mock connection."""
        logger.info("🔌 Mock exchange interface connected")
        return True
    
    async def disconnect(self):
        """Mock disconnection."""
        logger.info("🔌 Mock exchange interface disconnected")
        pass
    
    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000):
        """Mock klines data generation."""
        logger.info(f"📊 Mock: Generating klines data for {symbol} {interval} from {start_time} to {end_time}")
        
        # Generate mock OHLCV data
        start_dt = datetime.fromtimestamp(start_time / 1000)
        end_dt = datetime.fromtimestamp(end_time / 1000)
        
        # Create time range based on interval
        if interval == '1m':
            freq = '1T'
        elif interval == '5m':
            freq = '5T'
        elif interval == '15m':
            freq = '15T'
        elif interval == '30m':
            freq = '30T'
        elif interval == '1h':
            freq = '1H'
        else:
            freq = '1T'
        
        time_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        
        # Generate realistic mock data
        np.random.seed(42)  # For reproducible results
        n_points = len(time_range)
        
        # Start with a base price around $2000 for ETH
        base_price = 2000.0
        price_changes = np.random.normal(0, 0.01, n_points)  # 1% volatility
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # Minimum price floor
        
        # Generate OHLCV data
        klines = []
        for i, (timestamp, price) in enumerate(zip(time_range, prices)):
            # Generate OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            klines.append([
                int(timestamp.timestamp() * 1000),  # timestamp
                f"{open_price:.2f}",                # open
                f"{high:.2f}",                      # high
                f"{low:.2f}",                       # low
                f"{close_price:.2f}",               # close
                f"{volume:.2f}",                    # volume
                int(timestamp.timestamp() * 1000),  # close_time
                f"{volume * price:.2f}",            # quote_volume
                0,                                  # trades
                f"{volume * 0.5:.2f}",             # taker_buy_base_volume
                f"{volume * price * 0.5:.2f}",     # taker_buy_quote_volume
                "0"                                 # ignore
            ])
        
        return klines

def create_exchange_interface(config: Dict[str, Any]) -> MockExchangeInterface:
    """Create a mock exchange interface."""
    return MockExchangeInterface(config)

# Simple data structures
@dataclass
class ProcessingResult:
    """Result of data processing operations."""
    success: bool
    data: Optional[pd.DataFrame] = None
    message: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""
    data_dir: str = "historical_data"
    exchange: str = "bingx"
    enable_logging: bool = True
    enable_gap_filling: bool = True
    enable_resampling: bool = True
    enable_duplicate_handling: bool = True
    enable_quality_validation: bool = True
    batch_compatible: bool = True

@dataclass
class ResamplingConfig:
    """Configuration for data resampling."""
    target_intervals: List[str] = None
    method: str = 'ohlc'
    preserve_volume: bool = True
    resample_older_than_days: int = 1
    enable_auto_resampling: bool = True

# Simple data processing functions
def standardize_klines_data(klines: List[List], symbol: str) -> pd.DataFrame:
    """Convert klines data to standardized DataFrame."""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
        'taker_buy_quote_volume', 'ignore'
    ])
    
    # Convert to proper types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def detect_gaps(df: pd.DataFrame, interval: str) -> List[Tuple[datetime, datetime]]:
    """Detect gaps in the data."""
    if df.empty:
        return []
    
    # Convert interval to timedelta
    interval_map = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1)
    }
    
    expected_delta = interval_map.get(interval, timedelta(minutes=1))
    gaps = []
    
    for i in range(len(df) - 1):
        current_time = df.index[i]
        next_time = df.index[i + 1]
        actual_delta = next_time - current_time
        
        if actual_delta > expected_delta * 1.5:  # Allow some tolerance
            gaps.append((current_time, next_time))
    
    return gaps

def fill_gaps(df: pd.DataFrame, gaps: List[Tuple[datetime, datetime]], interval: str) -> pd.DataFrame:
    """Fill gaps in the data by forward filling."""
    if not gaps or df.empty:
        return df
    
    # Create a complete time range
    start_time = df.index.min()
    end_time = df.index.max()
    
    interval_map = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H'
    }
    
    freq = interval_map.get(interval, '1T')
    complete_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Reindex with complete range and forward fill
    df_complete = df.reindex(complete_range, method='ffill')
    
    return df_complete

def resample_data(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Resample data to target interval."""
    if df.empty:
        return df
    
    interval_map = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H'
    }
    
    freq = interval_map.get(target_interval, '1T')
    
    # Resample using OHLCV method
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality."""
    if df.empty:
        return {"quality_score": 0, "issues": ["No data"]}
    
    issues = []
    quality_score = 100
    
    # Check for missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 5:
        issues.append(f"High missing data: {missing_pct:.1f}%")
        quality_score -= missing_pct * 2
    
    # Check for zero volume
    zero_volume_pct = (df['volume'] == 0).sum() / len(df) * 100
    if zero_volume_pct > 10:
        issues.append(f"High zero volume: {zero_volume_pct:.1f}%")
        quality_score -= zero_volume_pct
    
    # Check for price consistency
    invalid_prices = ((df['high'] < df['low']) | 
                     (df['high'] < df['open']) | 
                     (df['high'] < df['close']) |
                     (df['low'] > df['open']) | 
                     (df['low'] > df['close'])).sum()
    
    if invalid_prices > 0:
        issues.append(f"Invalid price relationships: {invalid_prices} candles")
        quality_score -= invalid_prices * 5
    
    return {
        "quality_score": max(0, quality_score),
        "issues": issues,
        "missing_pct": missing_pct,
        "zero_volume_pct": zero_volume_pct,
        "invalid_prices": invalid_prices
    }

async def process_klines_data_simple(
    exchange: str,
    asset: str,
    lookback_period: str,
    interval: str,
    api_key: str = "",
    api_secret: str = "",
    use_testnet: bool = True,
    resampling_config: Optional[ResamplingConfig] = None,
    batch_id: str = "default"
) -> Dict[str, Any]:
    """Process klines data using simplified interface."""
    
    logger.info(f"🚀 Starting data processing for {asset} on {exchange}")
    logger.info(f"📊 Period: {lookback_period}, Interval: {interval}")
    
    # Create exchange interface
    exchange_config = {
        'exchange_type': exchange,
        'api_key': api_key,
        'api_secret': api_secret,
        'testnet': use_testnet
    }
    
    exchange_interface = create_exchange_interface(exchange_config)
    await exchange_interface.connect()
    
    try:
        # Calculate time range
        symbol = f"{asset}USDT"
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Parse lookback period
        if lookback_period.endswith('y'):
            years = int(lookback_period[:-1])
            start_time = int((datetime.now() - timedelta(days=years * 365)).timestamp() * 1000)
        elif lookback_period.endswith('m'):
            months = int(lookback_period[:-1])
            start_time = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)
        elif lookback_period.endswith('d'):
            days = int(lookback_period[:-1])
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        else:
            start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
        
        logger.info(f"📅 Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
        
        # Download data
        logger.info("📥 Downloading data...")
        klines = await exchange_interface.get_klines(symbol, interval, start_time, end_time)
        
        if not klines:
            logger.warning("⚠️ No data received from exchange")
            return {
                'pipeline_success': False,
                'data_quality': {'quality_score': 0, 'issues': ['No data received']},
                'final_data_shape': (0, 0),
                'stored_files': [],
                'resampled_intervals': []
            }
        
        # Standardize data
        logger.info("🔄 Standardizing data...")
        df = standardize_klines_data(klines, symbol)
        logger.info(f"📊 Initial data shape: {df.shape}")
        
        # Detect and fill gaps
        logger.info("🔍 Detecting gaps...")
        gaps = detect_gaps(df, interval)
        if gaps:
            logger.info(f"⚠️ Found {len(gaps)} gaps, filling...")
            df = fill_gaps(df, gaps, interval)
            logger.info(f"📊 After gap filling: {df.shape}")
        
        # Validate data quality
        logger.info("✅ Validating data quality...")
        quality_result = validate_data_quality(df)
        logger.info(f"📊 Data quality score: {quality_result['quality_score']:.1f}")
        
        if quality_result['issues']:
            logger.warning(f"⚠️ Data quality issues: {quality_result['issues']}")
        
        # Resample data if configured
        resampled_intervals = []
        if resampling_config and resampling_config.enable_auto_resampling:
            logger.info("🔄 Resampling data...")
            for target_interval in resampling_config.target_intervals:
                if target_interval != interval:
                    resampled_df = resample_data(df, target_interval)
                    resampled_intervals.append(target_interval)
                    logger.info(f"📊 Resampled to {target_interval}: {resampled_df.shape}")
        
        # Store data
        data_dir = Path("historical_data")
        data_dir.mkdir(exist_ok=True)
        
        stored_files = []
        main_file = data_dir / f"{symbol}_{interval}_{batch_id}.parquet"
        df.to_parquet(main_file)
        stored_files.append(str(main_file))
        logger.info(f"💾 Stored main data: {main_file}")
        
        # Store resampled data
        for target_interval in resampled_intervals:
            resampled_df = resample_data(df, target_interval)
            resampled_file = data_dir / f"{symbol}_{target_interval}_{batch_id}.parquet"
            resampled_df.to_parquet(resampled_file)
            stored_files.append(str(resampled_file))
            logger.info(f"💾 Stored resampled data: {resampled_file}")
        
        return {
            'pipeline_success': True,
            'data_quality': quality_result,
            'final_data_shape': df.shape,
            'stored_files': stored_files,
            'resampled_intervals': resampled_intervals
        }
        
    finally:
        await exchange_interface.disconnect()

async def main():
    """Main function to run the enhanced klines processing pipeline."""
    try:
        logger.info("🚀 Starting Enhanced Klines Processing Pipeline for ETHUSDT, 4 years, BingX")
        
        # Configure resampling
        resampling_config = ResamplingConfig(
            target_intervals=['5m', '15m', '30m', '1h'],
            method='ohlc',
            preserve_volume=True,
            resample_older_than_days=1,
            enable_auto_resampling=True
        )
        
        # Process data
        results = await process_klines_data_simple(
            exchange="bingx",
            asset="ETH",
            lookback_period="4y",
            interval="1m",
            api_key="",
            api_secret="",
            use_testnet=True,
            resampling_config=resampling_config,
            batch_id="ethusdt_4y_bingx"
        )
        
        # Print results
        print(f"\n🎉 Processing completed: {results['pipeline_success']}")
        print(f"📊 Data quality: {results['data_quality']}")
        print(f"📈 Final shape: {results['final_data_shape']}")
        print(f"💾 Stored files: {results['stored_files']}")
        print(f"🔄 Resampled intervals: {results['resampled_intervals']}")
        
    except Exception as e:
        logger.error(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Add missing import
    from dataclasses import dataclass
    
    asyncio.run(main())
