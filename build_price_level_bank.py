#!/usr/bin/env python3
"""
Price Level Bank Builder

This script builds a comprehensive bank of price levels with their historical tags
from historical market data. The bank can then be used by feature generators and
ML training processes.

Usage:
    python build_price_level_bank.py --symbol BTCUSDT --timeframe 1h --start-date 2023-01-01 --end-date 2024-01-01
    python build_price_level_bank.py --symbols-file symbols.txt --output-dir ./custom_bank
    python build_price_level_bank.py --config config.json
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from tqdm import tqdm

from feature_generation.core.price_level_bank import (
    PriceLevelBank,
    PriceLevelData,
    PriceLevelBankConfig
)
from feature_generation.categories.support_resistance import (
    HistoricalPriceLevelCrossingGenerator,
    HistoricalPriceLevelBounceGenerator,
    HistoricalVolumeAtPriceLevelGenerator,
    HistoricalPriceLevelTouchDensityGenerator,
    HistoricalPriceLevelTimeDecayGenerator,
    HistoricalPriceLevelSuccessRateGenerator
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceLevelBankBuilder:
    """Builder class for populating the price level bank."""

    def __init__(self,
                 output_dir: str = "./data/price_level_bank",
                 chunk_size: int = 1000,
                 batch_size: int = 100):
        """
        Initialize the bank builder.

        Args:
            output_dir: Directory to store the bank
            chunk_size: Size of data chunks to process
            batch_size: Number of levels to process in each batch
        """
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize bank with custom config
        bank_config = PriceLevelBankConfig(
            storage_path=str(self.output_dir),
            auto_save_interval=50,  # Save frequently during building
            max_levels_per_symbol=50000
        )
        self.bank = PriceLevelBank(bank_config)

        logger.info(f"Bank builder initialized with output dir: {self.output_dir}")

    def load_historical_data(self,
                           symbol: str,
                           timeframe: str,
                           start_date: str,
                           end_date: str) -> pd.DataFrame:
        """
        Load historical data for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with historical OHLCV data
        """
        logger.info(f"Loading historical data for {symbol} {timeframe} from {start_date} to {end_date}")

        # This is a placeholder - in practice, you'd load from your data source
        # For now, we'll create synthetic data for demonstration
        try:
            # Try to load from your data source here
            # data = load_data_from_exchange(symbol, timeframe, start_date, end_date)

            # For demonstration, create synthetic data
            logger.warning("Using synthetic data for demonstration. Replace with real data loading.")
            data = self._generate_synthetic_data(symbol, start_date, end_date, timeframe)

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise

        logger.info(f"Loaded {len(data)} records")
        return data

    def _generate_synthetic_data(self,
                                symbol: str,
                                start_date: str,
                                end_date: str,
                                timeframe: str) -> pd.DataFrame:
        """
        Generate synthetic data for demonstration purposes.
        Replace this with actual data loading logic.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Calculate number of periods
        if timeframe == '1m':
            periods = int((end - start).total_seconds() / 60)
            freq = '1min'
        elif timeframe == '5m':
            periods = int((end - start).total_seconds() / 300)
            freq = '5min'
        elif timeframe == '15m':
            periods = int((end - start).total_seconds() / 900)
            freq = '15min'
        elif timeframe == '1h':
            periods = int((end - start).total_seconds() / 3600)
            freq = '1h'
        elif timeframe == '4h':
            periods = int((end - start).total_seconds() / 14400)
            freq = '4h'
        elif timeframe == '1d':
            periods = (end - start).days
            freq = '1d'
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Generate synthetic price data
        base_price = 50000.0 if 'BTC' in symbol else 2000.0
        dates = pd.date_range(start=start, periods=periods, freq=freq)

        # Create realistic price movement
        np.random.seed(42)  # For reproducible results
        price_changes = np.random.normal(0, 0.02, periods)  # 2% std dev
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.1))  # Don't let it go too low

        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Create OHLC with some spread
            spread = price * 0.002  # 0.2% spread
            high = price + abs(np.random.normal(0, spread/2))
            low = price - abs(np.random.normal(0, spread/2))
            open_price = prices[max(0, i-1)] if i > 0 else price

            # Ensure OHLC relationships are correct
            high = max(high, open_price, price)
            low = min(low, open_price, price)

            # Generate volume
            volume = np.random.exponential(1000) + 100

            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def calculate_price_levels(self, data: pd.DataFrame, level_pcts: List[float]) -> List[PriceLevelData]:
        """
        Calculate price levels and their tags from historical data.

        Args:
            data: Historical OHLCV data
            level_pcts: List of level percentages to calculate

        Returns:
            List of PriceLevelData objects
        """
        logger.info(f"Calculating price levels for {len(data)} data points with {len(level_pcts)} level percentages")

        levels = []

        # Process in chunks to manage memory
        for chunk_start in range(0, len(data), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(data))
            chunk_data = data.iloc[chunk_start:chunk_end]

            logger.info(f"Processing chunk {chunk_start//self.chunk_size + 1} "
                       f"({chunk_start}-{chunk_end})")

            for idx, (_, row) in enumerate(chunk_data.iterrows()):
                current_price = row['close']
                current_time = row.name

                # Calculate levels for each percentage
                for level_pct in level_pcts:
                    price_range = current_price * level_pct / 100

                    # Define levels around current price
                    levels_up = [current_price + (i + 1) * price_range for i in range(5)]
                    levels_down = [current_price - (i + 1) * price_range for i in range(5)]

                    for price in levels_up + levels_down:
                        # Get historical data up to this point
                        historical_data = data.iloc[:chunk_start + idx]

                        if len(historical_data) < 50:  # Need minimum data
                            continue

                        level_data = self._calculate_level_tags(
                            price, level_pct, current_price, current_time,
                            historical_data, row
                        )

                        if level_data:
                            levels.append(level_data)

        logger.info(f"Calculated {len(levels)} price levels")
        return levels

    def _calculate_level_tags(self,
                            level_price: float,
                            level_pct: float,
                            current_price: float,
                            current_time: pd.Timestamp,
                            historical_data: pd.DataFrame,
                            current_row: pd.Series) -> Optional[PriceLevelData]:
        """
        Calculate all tags for a specific price level.

        Args:
            level_price: The price level
            level_pct: Level percentage
            current_price: Current market price
            current_time: Current timestamp
            historical_data: Historical data up to current point
            current_row: Current OHLCV row

        Returns:
            PriceLevelData object or None if insufficient data
        """
        if len(historical_data) < 50:  # Need minimum historical data
            return None

        # Extract symbol and timeframe from data
        symbol = "BTCUSDT"  # This would come from your data source
        timeframe = "1h"    # This would come from your data source

        # Initialize level data
        level_data = PriceLevelData(
            price=level_price,
            level_pct=level_pct,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=current_time
        )

        # Calculate historical tags
        try:
            # Crossing count (past 100 periods)
            crossing_gen = HistoricalPriceLevelCrossingGenerator(level_pct=level_pct, window=100)
            level_data.historical_crossings = self._extract_scalar_feature(
                crossing_gen, historical_data
            )

            # Bounce count (past 100 periods)
            bounce_gen = HistoricalPriceLevelBounceGenerator(level_pct=level_pct, window=100)
            level_data.historical_bounces = self._extract_scalar_feature(
                bounce_gen, historical_data
            )

            # Volume at level (past 100 periods)
            volume_gen = HistoricalVolumeAtPriceLevelGenerator(level_pct=level_pct, window=100)
            level_data.historical_volume = self._extract_scalar_feature(
                volume_gen, historical_data
            )

            # Touch density (past 100 periods)
            density_gen = HistoricalPriceLevelTouchDensityGenerator(level_pct=level_pct, window=100)
            level_data.historical_touch_density = self._extract_scalar_feature(
                density_gen, historical_data
            )

            # Time decay (past 100 periods with 20 period half-life)
            decay_gen = HistoricalPriceLevelTimeDecayGenerator(
                level_pct=level_pct, window=100, decay_half_life=20
            )
            level_data.historical_time_decay = self._extract_scalar_feature(
                decay_gen, historical_data
            )

            # Success rate (past 100 periods, measured 20 periods ahead)
            success_gen = HistoricalPriceLevelSuccessRateGenerator(
                level_pct=level_pct, window=100, forward_periods=20
            )
            level_data.historical_success_rate = self._extract_scalar_feature(
                success_gen, historical_data
            )

            # Additional metadata
            level_data.day_of_week = current_time.dayofweek
            level_data.hour_of_day = current_time.hour

            # Determine session type
            level_data.session_type = self._get_session_type(current_time.hour)

            # Calculate significance (simple version)
            total_activity = (level_data.historical_crossings +
                            level_data.historical_bounces +
                            level_data.historical_touch_density)
            level_data.significance_level = min(total_activity / 10.0, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating tags for level {level_price}: {e}")
            return None

        return level_data

    def _extract_scalar_feature(self, generator, data: pd.DataFrame) -> float:
        """
        Extract a scalar value from a feature generator.

        Args:
            generator: Feature generator instance
            data: Input data

        Returns:
            Scalar feature value
        """
        try:
            result_series = generator._generate_feature(data)
            # Return the last value (most recent)
            return float(result_series.iloc[-1]) if len(result_series) > 0 else 0.0
        except Exception:
            return 0.0

    def _get_session_type(self, hour: int) -> str:
        """Determine trading session type based on hour."""
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'european'
        else:
            return 'us'

    def build_bank(self,
                   symbol: str,
                   timeframe: str,
                   start_date: str,
                   end_date: str,
                   level_pcts: List[float] = None) -> Dict[str, Any]:
        """
        Build the price level bank for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            level_pcts: List of level percentages (default: [0.1, 0.2, 0.5, 1.0, 2.0])

        Returns:
            Dictionary with build statistics
        """
        if level_pcts is None:
            level_pcts = [0.1, 0.2, 0.5, 1.0, 2.0]

        logger.info(f"Building price level bank for {symbol} {timeframe}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Level percentages: {level_pcts}")

        # Load historical data
        data = self.load_historical_data(symbol, timeframe, start_date, end_date)

        # Calculate price levels
        levels = self.calculate_price_levels(data, level_pcts)

        # Add levels to bank in batches
        logger.info(f"Adding {len(levels)} levels to bank...")

        batch_stats = []
        for i in range(0, len(levels), self.batch_size):
            batch = levels[i:i + self.batch_size]
            batch_level_ids = self.bank.add_levels(batch)

            batch_stats.append({
                'batch': i // self.batch_size + 1,
                'levels_added': len(batch),
                'level_ids': batch_level_ids
            })

            logger.info(f"Processed batch {i // self.batch_size + 1}, "
                       f"added {len(batch)} levels")

        # Save final bank
        self.bank.save_to_disk()

        # Get final statistics
        final_stats = self.bank.get_statistics()

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'date_range': [start_date, end_date],
            'level_pcts': level_pcts,
            'total_levels_calculated': len(levels),
            'total_levels_stored': final_stats['total_levels'],
            'batch_stats': batch_stats,
            'bank_stats': final_stats
        }

    def build_from_config(self, config_file: str) -> Dict[str, Any]:
        """
        Build bank from a configuration file.

        Args:
            config_file: Path to JSON configuration file

        Returns:
            Dictionary with build results
        """
        logger.info(f"Loading configuration from {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        results = []

        for build_config in config.get('builds', []):
            try:
                result = self.build_bank(**build_config)
                results.append(result)
                logger.info(f"Completed build: {build_config}")
            except Exception as e:
                logger.error(f"Failed to build for {build_config}: {e}")
                results.append({'error': str(e), 'config': build_config})

        return {'results': results, 'total_builds': len(results)}

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Build Price Level Bank from historical data')

    # Single build arguments
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')

    # Batch build arguments
    parser.add_argument('--symbols-file', type=str,
                       help='File containing list of symbols to process')
    parser.add_argument('--config', type=str,
                       help='JSON configuration file for batch processing')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./data/price_level_bank',
                       help='Output directory for the bank')

    # Processing options
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Size of data chunks to process')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for adding levels to bank')
    parser.add_argument('--level-pcts', type=str, default='0.1,0.2,0.5,1.0,2.0',
                       help='Comma-separated list of level percentages')

    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Parse level percentages
    level_pcts = [float(p) for p in args.level_pcts.split(',')]

    # Create builder
    builder = PriceLevelBankBuilder(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size
    )

    try:
        if args.config:
            # Batch build from config file
            results = builder.build_from_config(args.config)
            logger.info(f"Batch build completed: {results}")

        elif args.symbols_file:
            # Build from symbols file
            with open(args.symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]

            results = []
            for symbol in symbols:
                if args.start_date and args.end_date:
                    result = builder.build_bank(
                        symbol=symbol,
                        timeframe=args.timeframe,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        level_pcts=level_pcts
                    )
                    results.append(result)

            logger.info(f"Built banks for {len(symbols)} symbols")

        elif args.symbol and args.start_date and args.end_date:
            # Single symbol build
            result = builder.build_bank(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                level_pcts=level_pcts
            )

            logger.info(f"Bank build completed: {result}")

        else:
            logger.error("Missing required arguments. Use --help for usage information.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()