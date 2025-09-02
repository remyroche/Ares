#!/usr/bin/env python3
"""Regenerate Timeframe Files.

This script regenerates all timeframe files (5m, 15m, 30m, 1h, 4h) from the base 1m data
after data has been updated, gaps have been filled, or new data has been added.

Usage:
    python scripts/regenerate_timeframes.py --symbol ETHUSDT --exchange BINANCE
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root=Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.validation_decorators import validate_dataframe_operation

logger=system_logger.getChild("TimeframeRegenerator")


class TimeframeRegenerator:
    """Regenerates timeframe files from base 1m data."""

    SUPPORTED_TIMEFRAMES=["5m", "15m", "30m", "1h", "4h"]

    def __init__(self, data_cache_path: str="data_cache") -> None:
        self.data_cache_path=Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

    @validate_dataframe_operation("resample_to_timeframe", validate_before=True, validate_after=True)
    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1m data to target timeframe."""
        if len(df) == 0:
            return pd.DataFrame()

        # Ensure timestamp is datetime
        df=df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # type: ignore[index]
        df=df.set_index("timestamp")

        # Resample based on timeframe
        timeframe_mapping={
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
        }

        if timeframe not in timeframe_mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        resampled=(
            df.resample(timeframe_mapping[timeframe])
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

        return resampled.reset_index()

    @handle_errors(default_return=None, context="save_resampled_data")
    def save_resampled_data(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[Path]:
        """Save resampled data to parquet file."""
        if len(df) == 0:
            return None

        output_filename=f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        output_path = self.data_cache_path / output_filename

        df.to_parquet(output_path, compression="zstd", index=False)
        return output_path

    @handle_errors(default_return={"success": False, "errors": ["Unhandled error"]}, context="regenerate_timeframes")
    def regenerate_timeframes(self, symbol: str, exchange: str, timeframes: Optional[List[str]]) -> Dict[str, object]:
        """Regenerate all timeframe files from 1m data."""
        if timeframes is None:
            timeframes=self.SUPPORTED_TIMEFRAMES

        logger.info(f"🔄 Regenerating timeframe files for {exchange}_{symbol}: {timeframes}")

        results: Dict[str, object] = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframes": timeframes,
            "regenerated_files": {},
            "failed_timeframes": [],
            "success": True,
            "errors": [],
        }

        # Get all 1m klines files
        klines_files=list(self.data_cache_path.glob(f"klines_{exchange}_{symbol}_1m_*.parquet"))
        if not klines_files:
            logger.error(f"❌ No 1m klines files found for {exchange}_{symbol}")
            results["success"] = False
            results["errors"].append("No 1m klines files found")  # type: ignore[index]
            return results

        # Load and combine all 1m data
        all_1m_data: List[pd.DataFrame] = []
        for file_path in klines_files:
            try:
                df=pd.read_parquet(file_path)
                all_1m_data.append(df)
                logger.info(f"📊 Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"⚠️ Error reading {file_path}: {e}")
                continue

        if not all_1m_data:
            logger.error(f"❌ No valid 1m data found for {exchange}_{symbol}")
            results["success"] = False
            results["errors"].append("No valid 1m data found")  # type: ignore[index]
            return results

        # Combine all 1m data
        combined_1m=pd.concat(all_1m_data, ignore_index=True)
        combined_1m=combined_1m.sort_values("timestamp").drop_duplicates(subset=["timestamp"])  # type: ignore[arg-type]

        logger.info(
            f"📊 Combined 1m data: {len(combined_1m)} rows from {combined_1m['timestamp'].min()} to {combined_1m['timestamp'].max()}",
        )

        # Regenerate each timeframe
        for timeframe in timeframes:
            try:
                logger.info(f"🔄 Regenerating {timeframe} timeframe...")

                # Resample to the target timeframe
                resampled_df=self.resample_to_timeframe(combined_1m, timeframe)

                if len(resampled_df) == 0:
                    logger.warning(f"⚠️ No data after resampling to {timeframe}")
                    results["failed_timeframes"].append(timeframe)  # type: ignore[index]
                    continue

                # Save the resampled data
                output_path=self.save_resampled_data(resampled_df, symbol, exchange, timeframe)

                if output_path:
                    results["regenerated_files"][timeframe] = str(output_path)
                    logger.info(f"✅ Regenerated {timeframe}: {len(resampled_df)} rows -> {output_path}")
                else:
                    logger.error(f"❌ Failed to save {timeframe} data")
                    results["failed_timeframes"].append(timeframe)  # type: ignore[index]
                    results["errors"].append(f"Failed to save {timeframe} data")  # type: ignore[index]

            except Exception as e:  # noqa: BLE001
                logger.exception(f"❌ Error regenerating {timeframe}: {e}")
                results["failed_timeframes"].append(timeframe)  # type: ignore[index]
                results["errors"].append(f"{timeframe}: {e}")  # type: ignore[index]

        # Summary
        successful=len(results["regenerated_files"])  # type: ignore[arg-type]
        failed=len(results["failed_timeframes"])  # type: ignore[arg-type]

        logger.info(f"📊 Timeframe regeneration complete: {successful} successful, {failed} failed")

        if failed > 0:
            results["success"] = False

        return results


async def main() -> None:
    """Main function."""
    parser=argparse.ArgumentParser(description="Regenerate timeframe files from 1m data")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name (default: BINANCE)")
    parser.add_argument("--timeframes", nargs="+",
                       default=["5m", "15m", "30m", "1h", "4h"],
                       help="Timeframes to regenerate (default: all)")
    parser.add_argument("--data-cache", default="data_cache",
                       help="Data cache directory (default: data_cache)")

    args=parser.parse_args()

    logger.info("🔄 Starting timeframe regeneration")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Timeframes: {args.timeframes}")

    # Initialize regenerator
    regenerator=TimeframeRegenerator(args.data_cache)

    # Regenerate timeframes
    results=regenerator.regenerate_timeframes(args.symbol, args.exchange, args.timeframes)

    # Print results
    if results["regenerated_files"]:
        for _timeframe, _file_path in results["regenerated_files"].items():  # type: ignore[assignment]
            logger.info(f"✅ {_timeframe}: {_file_path}")
    if results["failed_timeframes"]:
        for _timeframe in results["failed_timeframes"]:  # type: ignore[assignment]
            logger.warning(f"⚠️ Failed timeframe: {_timeframe}")
    if results["errors"]:
        for _error in results["errors"]:  # type: ignore[assignment]
            logger.error(f"❌ Error: {_error}")
    if results["success"]:
        logger.info("🎉 All done!")
    else:
        sys.exit(1)


if __name__== "__main__":
    asyncio.run(main())
