#!/usr/bin/env python3
"""Comprehensive Gap Filler for Pipeline Integration
Handles aggtrades, futures, and klines files with gap detection and filling.
"""

from __future__ import annotations

import asyncio
import io
import ssl
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import certifi
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ComprehensiveGapFiller:
    """Comprehensive gap filler that handles all data types."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.session: aiohttp.ClientSession | None = None
        self.max_api_calls_per_gap = 50  # Maximum calls to prevent infinite loops
        self.call_delay = 0.1  # Delay between API calls
        self.max_consecutive_empty = 3  # Stop if 3 consecutive calls return no data

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is available."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self) -> None:
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def detect_gaps_in_aggtrades_file(
        self,
        file_path: Path,
        min_gap_seconds: int = 5,
    ) -> list[dict[str, Any]]:
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
            df["time_diff"] = (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds()
            )

            # Find gaps larger than threshold
            gaps: list[dict[str, Any]] = []
            gap_rows = df[df["time_diff"] > min_gap_seconds]

            for idx, row in gap_rows.iterrows():
                if idx > 0:
                    gap_start = pd.to_datetime(
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    gap_duration = (gap_end - gap_start).total_seconds()

                    gaps.append(
                        {
                            "file": file_path.name,
                            "gap_start": gap_start,
                            "gap_end": gap_end,
                            "gap_duration_seconds": gap_duration,
                            "data_type": "aggtrades",
                        },
                    )

            return gaps

        except Exception:
            return []

    def detect_gaps_in_futures_file(
        self,
        file_path: Path,
        min_gap_hours: int = 1,
    ) -> list[dict[str, Any]]:
        """Detect gaps in a single futures file."""
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

            # Calculate time differences in hours
            df["time_diff_hours"] = (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() / 3600
            )

            # Find gaps larger than threshold
            # (futures typically have 8-hour funding intervals)
            gaps: list[dict[str, Any]] = []
            gap_rows = df[df["time_diff_hours"] > min_gap_hours]

            for idx, row in gap_rows.iterrows():
                if idx > 0:
                    gap_start = pd.to_datetime(
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    gap_duration_hours = (
                        (gap_end - gap_start).total_seconds() / 3600
                    )

                    gaps.append(
                        {
                            "file": file_path.name,
                            "gap_start": gap_start,
                            "gap_end": gap_end,
                            "gap_duration_hours": gap_duration_hours,
                            "data_type": "futures",
                        },
                    )

            return gaps

        except Exception:
            return []

    def detect_gaps_in_klines_file(
        self,
        file_path: Path,
        min_gap_minutes: int = 2,
    ) -> list[dict[str, Any]]:
        """Detect gaps in a single klines file."""
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

            # Calculate time differences in minutes
            df["time_diff_minutes"] = (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() / 60
            )

            # Find gaps larger than threshold
            gaps: list[dict[str, Any]] = []
            gap_rows = df[df["time_diff_minutes"] > min_gap_minutes]

            for idx, row in gap_rows.iterrows():
                if idx > 0:
                    gap_start = pd.to_datetime(
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    gap_duration_minutes = (
                        (gap_end - gap_start).total_seconds() / 60
                    )

                    gaps.append(
                        {
                            "file": file_path.name,
                            "gap_start": gap_start,
                            "gap_end": gap_end,
                            "gap_duration_minutes": gap_duration_minutes,
                            "data_type": "klines",
                        },
                    )

            return gaps

        except Exception:
            return []

    def _should_use_binance_vision(self, gap_start: datetime) -> bool:
        """Determine if we should use Binance Vision based on date."""
        cutoff_date = datetime.now() - timedelta(days=7)
        return gap_start < cutoff_date

    async def _fetch_aggtrades_data(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        start_time_ms: int,
        end_time_ms: int,
        market_segment: str = "um",
    ) -> list[dict[str, Any]]:
        """Fetch aggtrades data using appropriate source based on date."""
        if self._should_use_binance_vision(gap_start):
            return await self._fetch_aggtrades_from_binance_vision(
                symbol=symbol,
                gap_start=gap_start,
                gap_end=gap_end,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                market_segment=market_segment,
            )
        return await self._fetch_aggtrades_from_regular_api(
            symbol=symbol,
            gap_start=gap_start,
            gap_end=gap_end,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )

    async def _fetch_aggtrades_from_regular_api(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[dict[str, Any]]:
        """Download aggregated trades from regular Binance API for recent data."""
        await self._ensure_session()

        try:
            # Placeholder for real exchange client integration
            # Return empty list to indicate no additional data from
            # regular API in this stub
            return []
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
    ) -> list[dict[str, Any]]:
        """Download aggregated trades from Binance Vision for a specific gap period."""
        await self._ensure_session()

        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path = (
            f"data/futures/{market_segment}/daily/aggTrades/{symbol}/"
            f"{symbol}-aggTrades-{date_str}.zip"
        )
        url = f"{base_url}/{path}"

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            assert self.session is not None
            async with self.session.get(url, ssl=ssl_context) as resp:
                if resp.status != 200:
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return []

                with zf.open(csv_names[0]) as f:
                    # Binance vision aggTrades CSV has known schema; read without header
                    df = pd.read_csv(f, header=None)

            if df.empty:
                return []

            # Assign column names based on Binance schema
            if df.shape[1] >= 7:
                df.columns = [
                    "a",
                    "p",
                    "q",
                    "f",
                    "l",
                    "T",
                    "m",
                ] + list(range(7, df.shape[1]))

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
            return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(
                orient="records",
            )

        except Exception:
            return []

    async def _fetch_futures_data(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        market_segment: str = "um",
    ) -> list[dict[str, Any]]:
        """Fetch futures data using appropriate source based on date."""
        if self._should_use_binance_vision(gap_start):
            return await self._fetch_futures_from_binance_vision(
                symbol=symbol,
                gap_start=gap_start,
                gap_end=gap_end,
                market_segment=market_segment,
            )
        return await self._fetch_futures_from_regular_api(
            symbol=symbol,
            gap_start=gap_start,
            gap_end=gap_end,
        )

    async def _fetch_futures_from_regular_api(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
    ) -> list[dict[str, Any]]:
        """Download futures funding rate data from regular Binance API for
        recent data.
        """
        await self._ensure_session()

        try:
            # Placeholder for real exchange client integration
            return []
        except Exception:
            return []

    async def _fetch_futures_from_binance_vision(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        market_segment: str = "um",
    ) -> list[dict[str, Any]]:
        """Download futures funding rate data from Binance Vision for a
        specific gap period.
        """
        await self._ensure_session()

        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path = (
            f"data/futures/{market_segment}/daily/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{date_str}.zip"
        )
        url = f"{base_url}/{path}"

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            assert self.session is not None
            async with self.session.get(url, ssl=ssl_context) as resp:
                if resp.status != 200:
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return []

                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)

            if df.empty:
                return []

            # Process data types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            if "fundingRate" in df.columns:
                df["fundingRate"] = pd.to_numeric(
                    df["fundingRate"], errors="coerce",
                )

            # Filter to gap period
            df = df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]

            if df.empty:
                return []

            # Convert to list of dicts
            return df.to_dict(orient="records")

        except Exception:
            return []

    async def _fetch_klines_data(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        interval: str = "1m",
        market_segment: str = "um",
    ) -> list[dict[str, Any]]:
        """Fetch klines data using appropriate source based on date."""
        if self._should_use_binance_vision(gap_start):
            return await self._fetch_klines_from_binance_vision(
                symbol=symbol,
                gap_start=gap_start,
                gap_end=gap_end,
                interval=interval,
                market_segment=market_segment,
            )
        return await self._fetch_klines_from_regular_api(
            symbol=symbol,
            gap_start=gap_start,
            gap_end=gap_end,
            interval=interval,
        )

    async def _fetch_klines_from_regular_api(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        interval: str = "1m",
    ) -> list[dict[str, Any]]:
        """Download klines data from regular Binance API for recent data."""
        await self._ensure_session()

        try:
            # Placeholder for real exchange client integration
            return []
        except Exception:
            return []

    async def _fetch_klines_from_binance_vision(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        interval: str = "1m",
        market_segment: str = "um",
    ) -> list[dict[str, Any]]:
        """Download klines data from Binance Vision for a specific gap period."""
        await self._ensure_session()

        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path = (
            f"data/futures/{market_segment}/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_str}.zip"
        )
        url = f"{base_url}/{path}"

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            assert self.session is not None
            async with self.session.get(url, ssl=ssl_context) as resp:
                if resp.status != 200:
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return []

                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)

            if df.empty:
                return []

            # Process data types
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Filter to gap period
            df = df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]

            if df.empty:
                return []

            # Convert to list of dicts
            keep_cols = [
                c
                for c in ["timestamp", "open", "high", "low", "close", "volume"]
                if c in df.columns
            ]
            return df[keep_cols].to_dict(orient="records")

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
        if "timestamp" in df.columns and str(df["timestamp"].dtype).startswith(
            ("int", "float"),
        ):
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
        self, gap_info: dict[str, Any], symbol: str = "ETHUSDT"
    ) -> dict[str, Any]:
        """Fill a single gap using multiple API calls until gap is fully filled."""
        try:
            gap_start = gap_info["gap_start"]
            gap_end = gap_info["gap_end"]
            file_name = gap_info["file"]
            data_type = gap_info["data_type"]

            if data_type == "aggtrades":
                gap_duration = gap_info["gap_duration_seconds"]
            elif data_type == "futures":
                gap_duration = gap_info["gap_duration_hours"]
            elif data_type == "klines":
                gap_duration = gap_info["gap_duration_minutes"]
            else:
                return {
                    "success": False,
                    "error": f"Unknown data type: {data_type}",
                    "rows_added": 0,
                    "api_calls_made": 0,
                    "successful_calls": 0,
                }

            all_missing_data: list[dict[str, Any]] = []
            successful_calls = 0
            consecutive_empty_calls = 0

            # Keep making API calls until gap is filled or we hit limits
            call_num = 0
            while call_num < self.max_api_calls_per_gap:
                call_num += 1

                missing_data: list[dict[str, Any]] = []

                if data_type == "aggtrades":
                    # Convert to timestamps
                    start_time_ms = int(gap_start.timestamp() * 1000)
                    end_time_ms = int(gap_end.timestamp() * 1000)

                    missing_data = await self._fetch_aggtrades_data(
                        symbol=symbol,
                        gap_start=gap_start,
                        gap_end=gap_end,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    )
                elif data_type == "futures":
                    missing_data = await self._fetch_futures_data(
                        symbol=symbol, gap_start=gap_start, gap_end=gap_end
                    )
                elif data_type == "klines":
                    missing_data = await self._fetch_klines_data(
                        symbol=symbol,
                        gap_start=gap_start,
                        gap_end=gap_end,
                        interval="1m",
                    )

                if missing_data and len(missing_data) > 0:
                    all_missing_data.extend(missing_data)
                    successful_calls += 1
                    consecutive_empty_calls = 0
                else:
                    consecutive_empty_calls += 1

                # Check if we have enough data to fill the gap
                if data_type == "aggtrades":
                    expected_min_trades = max(1, int(gap_duration / 2))
                    if len(all_missing_data) >= expected_min_trades:
                        break
                elif data_type == "futures":
                    # Funding happens about every 8 hours
                    expected_min_records = max(1, int(gap_duration / 8))
                    if len(all_missing_data) >= expected_min_records:
                        break
                elif data_type == "klines":
                    expected_min_records = max(1, int(gap_duration))
                    if len(all_missing_data) >= expected_min_records:
                        break

                # Stop if too many consecutive empty calls
                if consecutive_empty_calls >= self.max_consecutive_empty:
                    break

                # Delay between calls
                await asyncio.sleep(self.call_delay)

            if all_missing_data:
                # Remove duplicates based on timestamp
                unique_data: list[dict[str, Any]] = []
                seen_timestamps: set[int] = set()

                for record in all_missing_data:
                    if data_type == "aggtrades":
                        timestamp = int(record.get("T", 0))
                    else:
                        ts_val = record.get("timestamp")
                        if isinstance(ts_val, (int, float)):
                            timestamp = int(ts_val)
                        elif isinstance(ts_val, str):
                            try:
                                timestamp = int(
                                    pd.to_datetime(ts_val).value // 10**6,
                                )
                            except Exception:
                                timestamp = 0
                        elif isinstance(ts_val, pd.Timestamp):
                            timestamp = int(ts_val.value // 10**6)
                        else:
                            timestamp = 0

                    if timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_data.append(record)

                # Convert to DataFrame and standardize
                df_missing = pd.DataFrame(unique_data)

                if data_type == "aggtrades":
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
                        [df_existing, df_missing], ignore_index=True
                    )
                    if "timestamp" in df_combined.columns:
                        df_combined = (
                            df_combined.sort_values("timestamp").drop_duplicates(
                                subset=["timestamp"],
                            )
                        )

                    # Save back in the same format
                    if file_path.suffix.lower() == ".parquet":
                        df_combined.to_parquet(
                            file_path, compression="zstd", index=False
                        )
                    elif file_path.suffix.lower() == ".csv":
                        df_combined.to_csv(file_path, index=False)

                    return {
                        "success": True,
                        "rows_added": int(len(df_missing)),
                        "api_calls_made": call_num,
                        "successful_calls": successful_calls,
                        "data_type": data_type,
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
        self, symbol: str, exchange: str, timeframes: list[str] | None = None
    ) -> dict[str, Any]:
        """Regenerate timeframe files after data has been updated/fixed."""
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h"]  # Removed 4h as requested

        results: dict[str, Any] = {
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
            klines_files = list(
                self.data_cache_path.glob(
                    f"klines_{exchange}_{symbol}_1m_*.parquet",
                ),
            )
            if not klines_files:
                results["success"] = False
                results["errors"].append("No 1m klines files found")
                return results

            # Load and combine all 1m data
            all_1m_data: list[pd.DataFrame] = []
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
            if "timestamp" in combined_1m.columns:
                combined_1m = combined_1m.sort_values("timestamp").drop_duplicates(
                    subset=["timestamp"],
                )

            # Regenerate each timeframe
            for timeframe in timeframes:
                try:
                    # Resample to the target timeframe
                    resampled_df = self._resample_to_timeframe(
                        combined_1m.copy(), timeframe,
                    )

                    if len(resampled_df) == 0:
                        results["failed_timeframes"].append(timeframe)
                        continue

                    # Save the resampled data
                    output_path = self._save_resampled_data(
                        resampled_df, symbol, exchange, timeframe,
                    )

                    if output_path:
                        results["regenerated_files"][timeframe] = str(output_path)
                    else:
                        results["failed_timeframes"].append(timeframe)
                        results["errors"].append(f"Failed to save {timeframe} data")

                except Exception as e:
                    results["failed_timeframes"].append(timeframe)
                    results["errors"].append(f"{timeframe}: {e}")

            # Summary
            _ = len(results["regenerated_files"])  # keep for potential logging
            failed = len(results["failed_timeframes"])

            if failed > 0 and failed == len(timeframes):
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # type: ignore[assignment]
        df = df.set_index("timestamp")

        # Resample based on timeframe
        timeframe_mapping = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
        }

        freq = timeframe_mapping.get(timeframe)
        if freq is None:
            return pd.DataFrame()

        resampled = (
            df.resample(freq)
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

    def _save_resampled_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> Path | None:
        """Save resampled data to parquet file."""
        if len(df) == 0:
            return None

        output_filename = f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        output_path = self.data_cache_path / output_filename

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, compression="zstd", index=False)
            return output_path
        except Exception:
            return None

    async def process_all_data_types(
        self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"
    ) -> dict[str, Any] | None:
        """Process all gaps in all data types (aggtrades, futures, klines)."""
        # Find all files for each data type
        aggtrades_pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"
        aggtrades_csv_pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        futures_pattern = f"futures_{exchange}_{symbol}_*.parquet"
        futures_csv_pattern = f"futures_{exchange}_{symbol}_*.csv"
        klines_pattern = f"klines_{exchange}_{symbol}_1m_*.parquet"
        klines_csv_pattern = f"klines_{exchange}_{symbol}_1m_*.csv"

        # Get all files
        aggtrades_files = list(self.data_cache_path.glob(aggtrades_pattern)) + list(
            self.data_cache_path.glob(aggtrades_csv_pattern)
        )
        futures_files = list(self.data_cache_path.glob(futures_pattern)) + list(
            self.data_cache_path.glob(futures_csv_pattern)
        )
        klines_files = list(self.data_cache_path.glob(klines_pattern)) + list(
            self.data_cache_path.glob(klines_csv_pattern)
        )

        all_files: list[tuple[Path, str]] = []

        # Add files with types
        for af in aggtrades_files:
            all_files.append((af, "aggtrades"))
        for ff in futures_files:
            all_files.append((ff, "futures"))
        for kf in klines_files:
            all_files.append((kf, "klines"))

        if not all_files:
            return None

        total_files_processed = 0
        total_files_with_gaps = 0
        total_gaps_found = 0
        total_gaps_filled = 0
        total_gaps_failed = 0
        total_api_calls = 0
        total_successful_calls = 0

        # Process each data type
        for data_type in ["aggtrades", "futures", "klines"]:
            type_files = [(f, t) for f, t in all_files if t == data_type]

            for file_path, _file_type in type_files:
                # Detect gaps based on data type
                if data_type == "aggtrades":
                    gaps = self.detect_gaps_in_aggtrades_file(file_path)
                elif data_type == "futures":
                    gaps = self.detect_gaps_in_futures_file(file_path)
                elif data_type == "klines":
                    gaps = self.detect_gaps_in_klines_file(file_path)
                else:
                    continue

                total_files_processed += 1

                if gaps:
                    total_files_with_gaps += 1
                    total_gaps_found += len(gaps)

                # Fill each gap with multiple API calls
                for _i, gap in enumerate(gaps):
                    result = await self.fill_gap_until_complete(gap, symbol)

                    total_api_calls += int(result.get("api_calls_made", 0))
                    total_successful_calls += int(result.get("successful_calls", 0))

                    if result.get("success"):
                        total_gaps_filled += 1
                    else:
                        total_gaps_failed += 1

                # Regenerate timeframe files after each successful gap fill
                # (only for aggtrades)
                if data_type == "aggtrades" and total_gaps_filled > 0:
                    timeframe_results = await self.regenerate_timeframe_files(
                        symbol, exchange,
                    )
                    if timeframe_results.get("success"):
                        # Count regenerated files as succeeded; no-op otherwise
                        _ = len(timeframe_results.get("regenerated_files", {}))
                    else:
                        # If regeneration failed, mark as failed timeframe work,
                        # not gap filling failure
                        pass

                # Rate limiting between gaps
                await asyncio.sleep(0.2)

        # Summary
        if total_gaps_found > 0:
            return {
                "files_processed": total_files_processed,
                "files_with_gaps": total_files_with_gaps,
                "gaps_found": total_gaps_found,
                "gaps_filled": total_gaps_filled,
                "gaps_failed": total_gaps_failed,
                "api_calls_made": total_api_calls,
                "successful_calls": total_successful_calls,
            }

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
async def run_comprehensive_gap_filling_pipeline(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    data_cache_path: str = "data_cache",
) -> dict[str, Any] | None:
    """Run comprehensive gap filling as part of the training pipeline."""
    gap_filler = ComprehensiveGapFiller(data_cache_path)

    try:
        return await gap_filler.process_all_data_types(symbol=symbol, exchange=exchange)
    finally:
        await gap_filler.close_session()


if __name__ == "__main__":
    asyncio.run(run_comprehensive_gap_filling_pipeline())