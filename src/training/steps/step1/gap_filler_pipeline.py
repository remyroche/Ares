#!/usr/bin/env python3
"""Gap Filler for Pipeline Integration
Makes API calls until gaps are fully filled and regenerates timeframe files.
"""

import asyncio
import io
import ssl
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import aiohttp
import certifi
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class PipelineGapFiller:
    """Gap filler that integrates with the training pipeline."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.session = None
        self.max_api_calls_per_gap = 50  # Maximum calls to prevent infinite loops
        self.call_delay = 0.1  # Delay between API calls
        self.max_consecutive_empty = 3  # Stop if 3 consecutive calls return no data

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is available."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self) -> None:
        """Close aiohttp session."""
        if self.session:
            await self.session.close()

    def detect_gaps_in_file(
        self, file_path: Path, min_gap_seconds: int = 5,
    ) -> list[dict]:
        """Detect gaps in a single aggtrades file."""
        try:
            # Read the file (Parquet or CSV)
            if file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                return []

            if df.empty:
                return []

            # Ensure timestamp column exists
            if "timestamp" not in df.columns:
                return []

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Calculate time differences
            df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

            # Find gaps larger than threshold
            gaps = []
            gap_rows = df[df["time_diff"] > min_gap_seconds]

            for idx, row in gap_rows.iterrows():
                if idx > 0:
                    gap_start = df.loc[idx - 1, "timestamp"]
                    gap_end = row["timestamp"]
                    gap_duration = (gap_end - gap_start).total_seconds()

                    gaps.append(
                        {
                            "file": file_path.name,
                            "gap_start": gap_start,
                            "gap_end": gap_end,
                            "gap_duration_seconds": gap_duration,
                        },
                    )

            return gaps

        except Exception:
            return []

    async def _fetch_aggtrades_from_binance_vision(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        start_time_ms: int,
        end_time_ms: int,
        market_segment: str = "um",
    ) -> list[dict]:
        """Download aggregated trades from Binance Vision for a specific gap period."""
        await self._ensure_session()

        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path = f"data/futures/{market_segment}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        url = f"{base_url}/{path}"

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with self.session.get(url, ssl=ssl_context) as resp:
                if resp.status != 200:
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return []

                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=["a", "p", "q", "f", "l", "T", "m", "M"],
                        low_memory=False,
                    )

            if df.empty:
                return []

            # Process data types
            for col in ["a", "f", "l", "T"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["p", "q"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["m"] = (
                df["m"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
                .astype("boolean")
            )

            # Drop invalid timestamps and filter to gap period
            df = df.dropna(subset=["T"])
            df = df[(df["T"] >= start_time_ms) & (df["T"] < end_time_ms)]

            if df.empty:
                return []

            # Convert to list of dicts
            return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(orient="records")

        except Exception:
            return []

    def _standardize_aggtrades_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize aggtrades data format."""
        expected_columns = [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "timestamp",
            "is_buyer_maker",
        ]

        # Map Binance Vision format to expected format
        if "a" in df.columns:
            column_mapping = {
                "a": "agg_trade_id",
                "p": "price",
                "q": "quantity",
                "f": "first_trade_id",
                "l": "last_trade_id",
                "T": "timestamp",
                "m": "is_buyer_maker",
            }
            df = df.rename(columns=column_mapping)

        # Convert timestamp from milliseconds to datetime
        if "timestamp" in df.columns and df["timestamp"].dtype in ["int64", "float64"]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Ensure proper data types
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "quantity" in df.columns:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        # Select only expected columns that exist
        available_columns = [col for col in expected_columns if col in df.columns]
        return df[available_columns]

    async def fill_gap_until_complete(
        self, gap_info: dict, symbol: str = "ETHUSDT",
    ) -> dict:
        """Fill a single gap using multiple API calls until gap is fully filled."""
        try:
            gap_start = gap_info["gap_start"]
            gap_end = gap_info["gap_end"]
            file_name = gap_info["file"]
            gap_duration = gap_info["gap_duration_seconds"]


            all_missing_data = []
            successful_calls = 0
            consecutive_empty_calls = 0

            # Keep making API calls until gap is filled or we hit limits
            call_num = 0
            while call_num < self.max_api_calls_per_gap:
                call_num += 1

                # Convert to timestamps
                start_time_ms = int(gap_start.timestamp() * 1000)
                end_time_ms = int(gap_end.timestamp() * 1000)

                # Try Binance Vision
                missing_data = await self._fetch_aggtrades_from_binance_vision(
                    symbol=symbol,
                    gap_start=gap_start,
                    gap_end=gap_end,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )

                if missing_data and len(missing_data) > 0:
                    all_missing_data.extend(missing_data)
                    successful_calls += 1
                    consecutive_empty_calls = 0

                    # Check if we have enough data to fill the gap
                    # For gaps > 10 seconds, we should have multiple trades
                    expected_min_trades = max(
                        1, int(gap_duration / 2),
                    )  # Rough estimate
                    if len(all_missing_data) >= expected_min_trades:
                        break
                else:
                    consecutive_empty_calls += 1

                    # Stop if too many consecutive empty calls
                    if consecutive_empty_calls >= self.max_consecutive_empty:
                        break

                # Delay between calls
                await asyncio.sleep(self.call_delay)

            if all_missing_data:
                # Remove duplicates based on trade ID and timestamp
                unique_data = []
                seen_combinations = set()

                for trade in all_missing_data:
                    # Create unique identifier for each trade
                    trade_id = trade.get("a", 0)
                    timestamp = trade.get("T", 0)
                    unique_id = (trade_id, timestamp)

                    if unique_id not in seen_combinations:
                        seen_combinations.add(unique_id)
                        unique_data.append(trade)


                # Convert to DataFrame and standardize
                df_missing = pd.DataFrame(unique_data)
                df_missing = self._standardize_aggtrades_format(df_missing)

                # Load existing file
                file_path = self.data_cache_path / file_name
                if file_path.exists():
                    # Read existing file (Parquet or CSV)
                    if file_path.suffix.lower() == ".parquet":
                        df_existing = pd.read_parquet(file_path)
                    elif file_path.suffix.lower() == ".csv":
                        df_existing = pd.read_csv(file_path)
                    else:
                        return {
                            "success": False,
                            "error": f"Unsupported file format: {file_path.suffix}",
                            "rows_added": 0,
                            "api_calls_made": call_num,
                            "successful_calls": successful_calls,
                        }

                    # Combine data
                    df_combined = pd.concat(
                        [df_existing, df_missing], ignore_index=True,
                    )
                    df_combined = df_combined.sort_values("timestamp").drop_duplicates(
                        subset=["timestamp"],
                    )

                    # Save back in the same format
                    if file_path.suffix.lower() == ".parquet":
                        df_combined.to_parquet(
                            file_path, compression="zstd", index=False,
                        )
                    elif file_path.suffix.lower() == ".csv":
                        df_combined.to_csv(file_path, index=False)

                    return {
                        "success": True,
                        "rows_added": len(df_missing),
                        "api_calls_made": call_num,
                        "successful_calls": successful_calls,
                        "gap_duration": gap_info["gap_duration_seconds"],
                    }

            return {
                "success": False,
                "error": f"No data available after {call_num} API calls",
                "rows_added": 0,
                "api_calls_made": call_num,
                "successful_calls": successful_calls,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rows_added": 0,
                "api_calls_made": 0,
                "successful_calls": 0,
            }

    async def regenerate_timeframe_files(
        self, symbol: str, exchange: str, timeframes: list[str] | None = None,
    ) -> dict:
        """Regenerate timeframe files after data has been updated/fixed.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to regenerate (default: 5m, 15m, 30m, 1h)

        Returns:
            Dictionary with regeneration results

        """
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h"]  # Removed 4h as requested


        results = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframes": timeframes,
            "regenerated_files": {},
            "failed_timeframes": [],
            "success": True,
            "errors": [],
        }

        try:
            # Get all 1m klines files
            klines_files = list(self.data_cache_path.glob(f"klines_{exchange}_{symbol}_1m_*.parquet"))
            if not klines_files:
                results["success"] = False
                results["errors"].append("No 1m klines files found")
                return results

            # Load and combine all 1m data
            all_1m_data = []
            for file_path in klines_files:
                try:
                    df = pd.read_parquet(file_path)
                    all_1m_data.append(df)
                except Exception:
                    continue

            if not all_1m_data:
                results["success"] = False
                results["errors"].append("No valid 1m data found")
                return results

            # Combine all 1m data
            combined_1m = pd.concat(all_1m_data, ignore_index=True)
            combined_1m = combined_1m.sort_values("timestamp").drop_duplicates(subset=["timestamp"])


            # Regenerate each timeframe
            for timeframe in timeframes:
                try:

                    # Resample to the target timeframe
                    resampled_df = self._resample_to_timeframe(combined_1m, timeframe)

                    if len(resampled_df) == 0:
                        results["failed_timeframes"].append(timeframe)
                        continue

                    # Save the resampled data
                    output_path = self._save_resampled_data(resampled_df, symbol, exchange, timeframe)

                    if output_path:
                        results["regenerated_files"][timeframe] = str(output_path)
                    else:
                        results["failed_timeframes"].append(timeframe)
                        results["errors"].append(f"Failed to save {timeframe} data")

                except Exception as e:
                    results["failed_timeframes"].append(timeframe)
                    results["errors"].append(f"{timeframe}: {e}")

            # Summary
            len(results["regenerated_files"])
            failed = len(results["failed_timeframes"])


            if failed > 0:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(f"General error: {e}")

        return results

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1m data to target timeframe."""
        if len(df) == 0:
            return pd.DataFrame()

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Resample based on timeframe
        timeframe_mapping = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
        }

        resampled = df.resample(timeframe_mapping[timeframe]).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled.reset_index()

    def _save_resampled_data(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Path | None:
        """Save resampled data to parquet file."""
        if len(df) == 0:
            return None

        output_filename = f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        output_path = self.data_cache_path / output_filename

        try:
            df.to_parquet(output_path, compression="zstd", index=False)
            return output_path
        except Exception:
            return None

    async def process_all_gaps(
        self, symbol: str = "ETHUSDT", exchange: str = "BINANCE",
    ):
        """Process all gaps in all aggtrades files."""
        # Find all aggtrades files (both Parquet and CSV)
        parquet_pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"
        csv_pattern = f"aggtrades_{exchange}_{symbol}_*.csv"

        parquet_files = list(self.data_cache_path.glob(parquet_pattern))
        csv_files = list(self.data_cache_path.glob(csv_pattern))

        # Combine and prioritize Parquet files
        parquet_files + csv_files
        files = []

        # Add Parquet files first
        for pf in parquet_files:
            files.append(pf)

        # Add CSV files that don't have Parquet equivalents
        for cf in csv_files:
            parquet_equivalent = cf.with_suffix(".parquet")
            if parquet_equivalent not in parquet_files:
                files.append(cf)

        if not files:
            return None


        total_files_processed = 0
        total_files_with_gaps = 0
        total_gaps_found = 0
        total_gaps_filled = 0
        total_gaps_failed = 0
        total_api_calls = 0
        total_successful_calls = 0

        for file_path in files:

            # Detect gaps in this file
            gaps = self.detect_gaps_in_file(file_path)
            total_files_processed += 1

            if gaps:
                total_files_with_gaps += 1
                total_gaps_found += len(gaps)

                # Fill each gap with multiple API calls
                for _i, gap in enumerate(gaps):

                    result = await self.fill_gap_until_complete(gap, symbol)

                    total_api_calls += result.get("api_calls_made", 0)
                    total_successful_calls += result.get("successful_calls", 0)

                    if result["success"]:
                        total_gaps_filled += 1

                        # Regenerate timeframe files after each successful gap fill
                        timeframe_results = await self.regenerate_timeframe_files(symbol, exchange)
                        if timeframe_results.get("success"):
                            pass
                        else:
                            pass
                    else:
                        total_gaps_failed += 1

                    # Rate limiting between gaps
                    await asyncio.sleep(0.5)
            else:
                pass

        # Summary

        if total_gaps_found > 0:
            (total_gaps_filled / total_gaps_found) * 100

        if total_api_calls > 0:
            (total_successful_calls / total_api_calls) * 100


        return {
            "files_processed": total_files_processed,
            "files_with_gaps": total_files_with_gaps,
            "gaps_found": total_gaps_found,
            "gaps_filled": total_gaps_filled,
            "gaps_failed": total_gaps_failed,
            "api_calls_made": total_api_calls,
            "successful_calls": total_successful_calls,
        }


# Function to integrate with pipeline
async def run_gap_filling_pipeline(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    data_cache_path: str = "data_cache",
):
    """Run gap filling as part of the training pipeline."""
    gap_filler = PipelineGapFiller(data_cache_path)

    try:
        return await gap_filler.process_all_gaps(symbol, exchange)
    finally:
        await gap_filler.close_session()


if __name__ == "__main__":
    asyncio.run(run_gap_filling_pipeline())
