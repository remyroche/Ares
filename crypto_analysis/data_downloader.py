#!/usr/bin/env python3
"""
Cryptocurrency Data Downloader for Scalping/Swinging Analysis
Downloads 15-minute klines from Binance for multiple assets over 2 years
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def download_asset_data(self, symbol, start_date, end_date, interval='15m'):
        """
        Download complete data for a single asset

        Args:
            symbol (str): Trading pair symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            interval (str): Kline interval

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")

        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        while current_start < end_ms:
            current_end = min(current_start + (1000 * 15 * 60 * 1000), end_ms)  # 1000 * 15 minutes in ms

            klines = self.get_klines(symbol, interval, current_start, current_end)

            if not klines:
                logger.warning(f"No data received for {symbol} at {datetime.fromtimestamp(current_start/1000)}")
                current_start = current_end
                continue

            all_data.extend(klines)
            current_start = current_end

            # Rate limiting
            time.sleep(0.1)

            # Progress logging
            if len(all_data) % 10000 == 0:
                logger.info(f"Downloaded {len(all_data)} klines for {symbol}")

        if not all_data:
            logger.error(f"No data downloaded for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert types with optimal Parquet storage in mind
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamps
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Set index
        df.set_index('open_time', inplace=True)

        # Remove unnecessary columns and ensure proper data types for Parquet
        df = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades']]

        # Ensure integer type for number_of_trades (better for Parquet)
        df['number_of_trades'] = df['number_of_trades'].astype('int64')

        # Add symbol column
        df['symbol'] = symbol

        # Ensure all numeric columns are float64 for optimal Parquet storage
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
            df[col] = df[col].astype('float64')

        logger.info(f"Successfully downloaded {len(df)} klines for {symbol}")
        return df

    def download_multiple_assets(self, symbols, start_date, end_date, interval='15m'):
        """
        Download data for multiple assets

        Args:
            symbols (list): List of trading pair symbols
            start_date (datetime): Start date
            end_date (datetime): End date
            interval (str): Kline interval

        Returns:
            pd.DataFrame: Combined DataFrame with all assets
        """
        all_data = []

        for symbol in symbols:
            try:
                df = self.download_asset_data(symbol, start_date, end_date, interval)
                if not df.empty:
                    all_data.append(df)
                else:
                    logger.warning(f"Skipping {symbol} due to empty data")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                continue

        if not all_data:
            logger.error("No data downloaded for any asset")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, axis=0)
        combined_df.sort_index(inplace=True)

        logger.info(f"Total data downloaded: {len(combined_df)} klines across {len(all_data)} assets")
        return combined_df

def verify_parquet_file(parquet_file, original_df):
    """
    Verify that the Parquet file was created correctly and contains the expected data

    Args:
        parquet_file (Path): Path to the Parquet file
        original_df (pd.DataFrame): Original DataFrame for comparison
    """
    try:
        # Read the Parquet file back
        loaded_df = pd.read_parquet(parquet_file)

        # Basic verification
        if len(loaded_df) != len(original_df):
            logger.error(f"Data length mismatch: Original {len(original_df)}, Loaded {len(loaded_df)}")
            return False

        if loaded_df.columns.tolist() != original_df.columns.tolist():
            logger.error(f"Column mismatch: Original {original_df.columns.tolist()}, Loaded {loaded_df.columns.tolist()}")
            return False

        # Check data integrity for a few random rows
        sample_size = min(100, len(original_df))
        sample_indices = original_df.sample(n=sample_size).index

        for idx in sample_indices:
            if idx in loaded_df.index:
                original_row = original_df.loc[idx]
                loaded_row = loaded_df.loc[idx]

                if not original_row.equals(loaded_row):
                    logger.warning(f"Data mismatch at index {idx}")
                    return False

        # File size check
        file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
        logger.info(f"Parquet file verification successful:")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        logger.info(f"  - Records: {len(loaded_df):,}")
        logger.info(f"  - Columns: {len(loaded_df.columns)}")
        logger.info(f"  - Compression ratio: {len(original_df) * len(original_df.columns) * 8 / (file_size_mb * 1024 * 1024):.1f}x")

        return True

    except Exception as e:
        logger.error(f"Error verifying Parquet file: {e}")
        return False

def main():
    """Main function to download cryptocurrency data"""

    # Define assets to download
    assets = [
        'ETHUSDT', 'ADAUSDT', 'ALGOUSDT', 'BTCUSDT', 'BNBUSDT',
        'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT',
        'ATOMUSDT', 'UNIUSDT', 'LTCUSDT', 'XRPUSDT', 'BCHUSDT'
    ]

    # Define date range (2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)

    logger.info(f"Starting download for {len(assets)} assets from {start_date} to {end_date}")

    # Create downloader instance
    downloader = BinanceDataDownloader()

    # Download data
    df = downloader.download_multiple_assets(assets, start_date, end_date, interval='15m')

    if df.empty:
        logger.error("No data downloaded. Exiting.")
        return

    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Save to Parquet file with optimal settings
    output_file = output_dir / f"crypto_15m_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"

    try:
        # Save with optimal Parquet settings
        df.to_parquet(
            output_file,
            compression='snappy',
            engine='pyarrow',
            index=True  # Include the datetime index
        )

        logger.info(f"Data saved to {output_file}")

        # Verify the Parquet file was created correctly
        verify_parquet_file(output_file, df)

    except Exception as e:
        logger.error(f"Error saving Parquet file: {e}")
        # Fallback to CSV if Parquet fails
        csv_file = output_file.with_suffix('.csv')
        df.to_csv(csv_file)
        logger.info(f"Data saved to CSV as fallback: {csv_file}")
        return

    # Print summary statistics
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total klines: {len(df):,}")
    print(f"Assets: {df['symbol'].nunique()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    # Per-asset summary
    print("\nPer-asset summary:")
    print("-" * 30)
    for symbol in sorted(df['symbol'].unique()):
        asset_data = df[df['symbol'] == symbol]
        print(f"{symbol:10} | {len(asset_data):8,} klines | "
              f"Volume: {asset_data['volume'].sum():12.0f}")

if __name__ == "__main__":
    main()