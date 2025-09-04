#!/usr/bin/env python3
"""Comprehensive Gap Filler v2 - Multiple API calls to ensure complete gap filling."""

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
project_root=Path(__file__).parent
sys.path.insert(0, str(project_root))


class ComprehensiveGapFillerV2:
    """Comprehensive gap filling with multiple API calls for complete coverage."""

    def __init__(self, data_cache_path: str="data_cache") -> None:
        self.data_cache_path=Path(data_cache_path)
        self.session=None
        self.max_api_calls_per_gap = 10  # Maximum calls to prevent infinite loops
        self.call_delay = 0.2  # Delay between API calls
        self.min_trades_per_gap = 1  # Minimum trades expected per gap

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is available."""
        if self.session is None:
            self.session=aiohttp.ClientSession()

    async def close_session(self) -> None:
        """Close aiohttp session."""
        if self.session:
            await self.session.close()

    def detect_gaps_in_file(
        self, file_path: Path, min_gap_seconds: int=10,
    ) -> list[dict]:
        """Detect gaps in a single aggtrades file."""
        try:
            # Read the parquet file
            df=pd.read_parquet(file_path)

            if df.empty:
                return []

            # Ensure timestamp column exists
            if "timestamp" not in df.columns:
                return []

            # Sort by timestamp
            df=df.sort_values("timestamp").reset_index(drop=True)

            # Calculate time differences
            df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

            # Find gaps larger than threshold
            gaps=[]
            gap_rows = df[df["time_diff"] > min_gap_seconds]

            for idx, row in gap_rows.iterrows():
                if idx > 0:
                    gap_start=df.loc[idx - 1, "timestamp"]
                    gap_end=row["timestamp"]
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
        market_segment: str="um",
    ) -> list[dict]:
        """Download aggregated trades from Binance Vision for a specific gap period."""
        await self._ensure_session()

        base_url="https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path=f"data/futures/{market_segment}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        url = f"{base_url}/{path}"

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with self.session.get(url, ssl=ssl_context) as resp:
                if resp.status != 200:
                    return []
                content=await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names=[n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return []

                with zf.open(csv_names[0]) as f:
                    df=pd.read_csv(
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
            df=df.dropna(subset=["T"])
            df=df[(df["T"] >= start_time_ms) & (df["T"] < end_time_ms)]

            if df.empty:
                return []

            # Convert to list of dicts
            return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(orient="records")

        except Exception:
            return []

    def _standardize_aggtrades_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize aggtrades data format."""
        expected_columns=[
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
            column_mapping={
                "a": "agg_trade_id",
                "p": "price",
                "q": "quantity",
                "f": "first_trade_id",
                "l": "last_trade_id",
                "T": "timestamp",
                "m": "is_buyer_maker",
            }
            df=df.rename(columns=column_mapping)

        # Convert timestamp from milliseconds to datetime
        if "timestamp" in df.columns and df["timestamp"].dtype in ["int64", "float64"]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Ensure proper data types
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "quantity" in df.columns:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        # Select only expected columns that exist
        available_columns=[col for col in expected_columns if col in df.columns]
        return df[available_columns]

    async def fill_gap_with_multiple_calls(
        self, gap_info: dict, symbol: str="ETHUSDT",
    ) -> dict:
        """Fill a single gap using multiple API calls until gap is fully filled."""
        try:
            gap_start=gap_info["gap_start"]
            gap_end = gap_info["gap_end"]
            file_name = gap_info["file"]
            gap_duration = gap_info["gap_duration_seconds"]


            all_missing_data = []
            successful_calls = 0
            consecutive_empty_calls = 0
            max_consecutive_empty = 3  # Stop if 3 consecutive calls return no data

            # Keep making API calls until gap is filled or we hit limits
            call_num = 0
            while call_num < self.max_api_calls_per_gap:
                call_num += 1

                # Convert to timestamps
                start_time_ms = int(gap_start.timestamp() * 1000)
                end_time_ms=int(gap_end.timestamp() * 1000)

                # Try Binance Vision
                missing_data=await self._fetch_aggtrades_from_binance_vision(
                    symbol=symbol,
                    gap_start=gap_start,
                    gap_end=gap_end,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )

                if missing_data and len(missing_data) > 0:
                    all_missing_data.extend(missing_data)
                    successful_calls += 1
                    consecutive_empty_calls=0

                    # Check if we have enough data to fill the gap
                    # For gaps > 10 seconds, we should have multiple trades
                    expected_min_trades=max(
                        1, int(gap_duration / 2),
                    )  # Rough estimate
                    if len(all_missing_data) >= expected_min_trades:
                        break
                else:
                    consecutive_empty_calls += 1

                    # Stop if too many consecutive empty calls
                    if consecutive_empty_calls >= max_consecutive_empty:
                        break

                # Delay between calls
                await asyncio.sleep(self.call_delay)

            if all_missing_data:
                # Remove duplicates based on trade ID and timestamp
                unique_data=[]
                seen_combinations = set()

                for trade in all_missing_data:
                    # Create unique identifier for each trade
                    trade_id=trade.get("a", 0)
                    timestamp=trade.get("T", 0)
                    unique_id=(trade_id, timestamp)

                    if unique_id not in seen_combinations:
                        seen_combinations.add(unique_id)
                        unique_data.append(trade)


                # Convert to DataFrame and standardize
                df_missing=pd.DataFrame(unique_data)
                df_missing=self._standardize_aggtrades_format(df_missing)

                # Load existing file
                file_path=self.data_cache_path / file_name
                if file_path.exists():
                    df_existing=pd.read_parquet(file_path)

                    # Combine data
                    df_combined=pd.concat(
                        [df_existing, df_missing], ignore_index=True,
                    )
                    df_combined=df_combined.sort_values("timestamp").drop_duplicates(
                        subset=["timestamp"],
                    )

                    # Save back
                    df_combined.to_parquet(file_path, compression="zstd", index=False)

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

    async def process_all_gaps(
        self, symbol: str="ETHUSDT", exchange: str="BINANCE",
    ) -> None:
        """Process all gaps in all aggtrades files with multiple API calls."""
        # Find all aggtrades files
        pattern=f"aggtrades_{exchange}_{symbol}_*.parquet"
        files = list(self.data_cache_path.glob(pattern))

        if not files:
            return total_files_processed, 0
        total_files_with_gaps = 0
        total_gaps_found = 0
        total_gaps_filled = 0
        total_gaps_failed = 0
        total_api_calls = 0
        total_successful_calls = 0

        for file_path in files:
            pass
            pass
            pass
            pass  # TODO: Add proper implementation
            # Detect gaps in this file
            gaps = self.detect_gaps_in_file(file_path)
            total_files_processed += 1

            if gaps:
                total_files_with_gaps += 1
                total_gaps_found += len(gaps)

                # Fill each gap with multiple API calls
                for _i, gap in enumerate(gaps):
    pass  # TODO: Add proper implementation
                    result=await self.fill_gap_with_multiple_calls(gap, symbol)

                    total_api_calls += result.get("api_calls_made", 0)
                    total_successful_calls += result.get("successful_calls", 0)

                    if result["success"]:
                        total_gaps_filled += 1
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



async def main() -> None:
    """Main function."""
    gap_filler=ComprehensiveGapFillerV2()

    try:
        await gap_filler.process_all_gaps()
    finally:
        await gap_filler.close_session()


if __name__== "__main__":
    asyncio.run(main())
