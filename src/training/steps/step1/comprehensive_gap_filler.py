#!/usr / bin / env python3
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
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class ComprehensiveGapFiller:
    """Comprehensive gap filler that handles all data types."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
    pass
    pass
        self.data_cache_path, Path(data_cache_path)
        self.session: aiohttp.ClientSession | None, None
        self.max_api_calls_per_gap, 50  # Maximum calls to prevent infinite loops
        self.call_delay, 0.1  # Delay between API calls
        self.max_consecutive_empty, 3  # Stop if 3 consecutive calls return no data

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is available."""
        if self.session is None:
    pass
    pass
            timeout, aiohttp.ClientTimeout(total = 60)
        self.session, aiohttp.ClientSession(timeout = timeout)

    async def close_session(self) -> None:
        """Close aiohttp session."""
        if self.session:
    pass
    pass
        await self.session.close()
        self.session, None

    def detect_gaps_in_aggtrades_file(
        self,
        file_path: Path,
        min_gap_seconds: int, 5,
    ) -> list[dict[str, Any]]:
        """Detect gaps in a single aggtrades file."""
        try:
        # Read the file (Parquet or CSV)
    except Exception as e:
        pass
    except Exception as e:
        pass
        if file_path.suffix.lower() == ".parquet":
    pass
    pass
                df, pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df, pd.read_csv(file_path)
            else:
        return []

        if df.empty:
    pass
    pass
        return []

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
    pass
    pass
        return []

        # Sort by timestamp
            df, df.sort_values("timestamp").reset_index(drop = True)

        # Calculate time differences
            df["time_diff"] = (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds()
            )

        # Find gaps larger than threshold
            gaps: list[dict[str, Any]] = []
            gap_rows, df[df["time_diff"] > min_gap_seconds]

        for idx, row in gap_rows.iterrows():
    pass
    pass
        if idx > 0:
    pass
    pass
                    gap_start, pd.to_datetime(
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end, pd.to_datetime(row["timestamp"]).to_pydatetime()
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
        min_gap_hours: int, 1,
    ) -> list[dict[str, Any]]:
        """Detect gaps in a single futures file."""
        try:
        # Read the file (Parquet or CSV)
    except Exception as e:
        pass
    except Exception as e:
        pass
        if file_path.suffix.lower() == ".parquet":
    pass
    pass
                df, pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df, pd.read_csv(file_path)
            else:
        return []

        if df.empty:
    pass
    pass
        return []

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
    pass
    pass
        return []

        # Sort by timestamp
            df, df.sort_values("timestamp").reset_index(drop = True)

        # Calculate time differences in hours
            df["time_diff_hours"] = (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() / 3600
            )

        # Find gaps larger than threshold
        # (futures typically have 8 - hour funding intervals)
            gaps: list[dict[str, Any]] = []
            gap_rows, df[df["time_diff_hours"] > min_gap_hours]

        for idx, row in gap_rows.iterrows():
    pass
    pass
        if idx > 0:
    pass
    pass
                    gap_start, pd.to_datetime(
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end, pd.to_datetime(row["timestamp"]).to_pydatetime()
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
        min_gap_minutes: int, 2,
    ) -> list[dict[str, Any]]:
        """Detect gaps in a single klines file."""
        try:
        # Read the file (Parquet or CSV)
    except Exception as e:
        pass
    except Exception as e:
        pass
        if file_path.suffix.lower() == ".parquet":
    pass
    pass
                df, pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df, pd.read_csv(file_path)
            else:
        return []

        if df.empty:
    pass
    pass
        return []

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
    pass
    pass
        return []

        # Sort by timestamp
            df, df.sort_values("timestamp").reset_index(drop = True)

        # Calculate time differences in minutes
            df["time_diff_minutes"] = (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() / 60
            )

        # Find gaps larger than threshold
            gaps: list[dict[str, Any]] = []
            gap_rows, df[df["time_diff_minutes"] > min_gap_minutes]

        for idx, row in gap_rows.iterrows():
    pass
    pass
        if idx > 0:
    pass
    pass
                    gap_start, pd.to_datetime(
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end, pd.to_datetime(row["timestamp"]).to_pydatetime()
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
    pass
    pass
        """Determine if we should use Binance Vision based on date."""
        cutoff_date, datetime.now() - timedelta(days = 7)
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
    pass
    pass
        return await self._fetch_aggtrades_from_binance_vision(
                symbol = symbol,
                gap_start = gap_start,
                gap_end = gap_end,
                start_time_ms = start_time_ms,
                end_time_ms = end_time_ms,
                market_segment = market_segment,
            )
        return await self._fetch_aggtrades_from_regular_api(
            symbol = symbol,
            gap_start = gap_start,
            gap_end = gap_end,
            start_time_ms = start_time_ms,
            end_time_ms = end_time_ms,
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
    except Exception as e:
        pass
    except Exception as e:
        pass
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
        date_str, gap_start.strftime("%Y-%m-%d")
        path = (
            f"data / futures/{market_segment}/daily / aggTrades/{symbol}/"
            f"{symbol}-aggTrades-{date_str}.zip"
        )
        url, f"{base_url}/{path}"

        try:
            ssl_context, ssl.create_default_context(cafile = certifi.where())

    except Exception as e:
        pass
    except Exception as e:
        pass
            assert self.session is not None
        async with self.session.get(url, ssl = ssl_context) as resp:
        if resp.status != 200:
    pass
    pass
        return []
                content, await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
    pass
    pass
        return []

        with zf.open(csv_names[0]) as f:
        # Binance vision aggTrades CSV has known schema; read without header
                    df, pd.read_csv(f, header = None)

        if df.empty:
    pass
    pass
        return []

        # Assign column names based on Binance schema
        if df.shape[1] >= 7:
    pass
    pass
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
    pass
    pass
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["p", "q"]:
    pass
    pass
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
            df, df.dropna(subset=["T"])
            df, df[(df["T"] >= start_time_ms) & (df["T"] < end_time_ms)]

        if df.empty:
    pass
    pass
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
    pass
    pass
        return await self._fetch_futures_from_binance_vision(
                symbol = symbol,
                gap_start = gap_start,
                gap_end = gap_end,
                market_segment = market_segment,
            )
        return await self._fetch_futures_from_regular_api(
            symbol = symbol,
            gap_start = gap_start,
            gap_end = gap_end,
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
    except Exception as e:
        pass
    except Exception as e:
        pass
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
        date_str, gap_start.strftime("%Y-%m-%d")
        path = (
            f"data / futures/{market_segment}/daily / fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{date_str}.zip"
        )
        url, f"{base_url}/{path}"

        try:
            ssl_context, ssl.create_default_context(cafile = certifi.where())

    except Exception as e:
        pass
    except Exception as e:
        pass
            assert self.session is not None
        async with self.session.get(url, ssl = ssl_context) as resp:
        if resp.status != 200:
    pass
    pass
        return []
                content, await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
    pass
    pass
        return []

        with zf.open(csv_names[0]) as f:
                    df, pd.read_csv(f)

        if df.empty:
    pass
    pass
        return []

        # Process data types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if "fundingRate" in df.columns:
    pass
    pass
                df["fundingRate"] = pd.to_numeric(
                    df["fundingRate"], errors="coerce",
                )

        # Filter to gap period
            df, df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]

        if df.empty:
    pass
    pass
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
    pass
    pass
        return await self._fetch_klines_from_binance_vision(
                symbol = symbol,
                gap_start = gap_start,
                gap_end = gap_end,
                interval = interval,
                market_segment = market_segment,
            )
        return await self._fetch_klines_from_regular_api(
            symbol = symbol,
            gap_start = gap_start,
            gap_end = gap_end,
            interval = interval,
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
    except Exception as e:
        pass
    except Exception as e:
        pass
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
        date_str, gap_start.strftime("%Y-%m-%d")
        path = (
            f"data / futures/{market_segment}/daily / klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_str}.zip"
        )
        url, f"{base_url}/{path}"

        try:
            ssl_context, ssl.create_default_context(cafile = certifi.where())

    except Exception as e:
        pass
    except Exception as e:
        pass
            assert self.session is not None
        async with self.session.get(url, ssl = ssl_context) as resp:
        if resp.status != 200:
    pass
    pass
        return []
                content, await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
    pass
    pass
        return []

        with zf.open(csv_names[0]) as f:
                    df, pd.read_csv(f)

        if df.empty:
    pass
    pass
        return []

        # Process data types
        if "timestamp" in df.columns:
    pass
    pass
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
    pass
    pass
        if col in df.columns:
    pass
    pass
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter to gap period
            df, df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]

        if df.empty:
    pass
    pass
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
    pass
    pass
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
    pass
    pass
            column_mapping = {
                "a": "agg_trade_id",
                "p": "price",
                "q": "quantity",
                "f": "first_trade_id",
                "l": "last_trade_id",
                "T": "timestamp",
                "m": "is_buyer_maker",
            }
            df, df.rename(columns = column_mapping)

        # Convert timestamp from milliseconds to datetime
        if "timestamp" in df.columns and str(df["timestamp"].dtype).startswith(
            ("int", "float"),
        ):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Ensure proper data types
        if "price" in df.columns:
    pass
    pass
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "quantity" in df.columns:
    pass
    pass
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        # Select only expected columns that exist
        available_columns = [col for col in expected_columns if col in df.columns]
        return df[available_columns]

    async def fill_gap_until_complete(
        self, gap_info: dict[str, Any], symbol: str = "ETHUSDT"
    ) -> dict[str, Any]:
        """Fill a single gap using multiple API calls until gap is fully filled."""
        try:
            gap_start, gap_info["gap_start"]
    except Exception as e:
        pass
    except Exception as e:
        pass
            gap_end, gap_info["gap_end"]
            file_name, gap_info["file"]
            data_type, gap_info["data_type"]

        if data_type == "aggtrades":
    pass
    pass
                gap_duration, gap_info["gap_duration_seconds"]
            elif data_type == "futures":
                gap_duration, gap_info["gap_duration_hours"]
            elif data_type == "klines":
                gap_duration, gap_info["gap_duration_minutes"]
            else:
        return {
                    "success": False,
                    "error": f"Unknown data type: {data_type}",
                    "rows_added": 0,
                    "api_calls_made": 0,
                    "successful_calls": 0,
                }

            all_missing_data: list[dict[str, Any]] = []
            successful_calls, 0
            consecutive_empty_calls, 0

        # Keep making API calls until gap is filled or we hit limits
            call_num, 0
        while call_num < self.max_api_calls_per_gap:
                call_num += 1

                missing_data: list[dict[str, Any]] = []

        if data_type == "aggtrades":
    pass
    pass
        # Convert to timestamps
                    start_time_ms, int(gap_start.timestamp() * 1000)
                    end_time_ms, int(gap_end.timestamp() * 1000)

                    missing_data, await self._fetch_aggtrades_data(
                        symbol = symbol,
                        gap_start = gap_start,
                        gap_end = gap_end,
                        start_time_ms = start_time_ms,
                        end_time_ms = end_time_ms,
                    )
                elif data_type == "futures":
                    missing_data, await self._fetch_futures_data(
                        symbol = symbol, gap_start = gap_start, gap_end = gap_end
                    )
                elif data_type == "klines":
                    missing_data, await self._fetch_klines_data(
                        symbol = symbol,
                        gap_start = gap_start,
                        gap_end = gap_end,
                        interval="1m",
                    )

        if missing_data and len(missing_data) > 0:
    pass
    pass
                    all_missing_data.extend(missing_data)
                    successful_calls += 1
                    consecutive_empty_calls, 0
                else:
                    consecutive_empty_calls += 1

        # Check if we have enough data to fill the gap
        if data_type == "aggtrades":
    pass
    pass
                    expected_min_trades, max(1, int(gap_duration / 2))
        if len(all_missing_data) >= expected_min_trades:
    pass
    pass
                        break
                elif data_type == "futures":
        # Funding happens about every 8 hours
                    expected_min_records, max(1, int(gap_duration / 8))
        if len(all_missing_data) >= expected_min_records:
    pass
    pass
                        break
                elif data_type == "klines":
                    expected_min_records, max(1, int(gap_duration))
        if len(all_missing_data) >= expected_min_records:
    pass
    pass
                        break

        # Stop if too many consecutive empty calls
        if consecutive_empty_calls >= self.max_consecutive_empty:
    pass
    pass
                    break

        # Delay between calls
        await asyncio.sleep(self.call_delay)

        if all_missing_data:
    pass
    pass
        # Remove duplicates based on timestamp
                unique_data: list[dict[str, Any]] = []
                seen_timestamps: set[int] = set()

        for record in all_missing_data:
    pass
    pass
        if data_type == "aggtrades":
    pass
    pass
                        timestamp, int(record.get("T", 0))
                    else:
                        ts_val, record.get("timestamp")
        if isinstance(ts_val, (int, float)):
    pass
    pass
                            timestamp, int(ts_val)
                        elif isinstance(ts_val, str):
        try:
                                timestamp, int(
                                    pd.to_datetime(ts_val).value // 10**6,
    except Exception as e:
        pass
    except Exception as e:
        pass
                                )
        except Exception:
                                timestamp, 0
                        elif isinstance(ts_val, pd.Timestamp):
                            timestamp, int(ts_val.value // 10**6)
                        else:
                            timestamp, 0

        if timestamp not in seen_timestamps:
    pass
    pass
                        seen_timestamps.add(timestamp)
                        unique_data.append(record)

        # Convert to DataFrame and standardize
                df_missing, pd.DataFrame(unique_data)

        if data_type == "aggtrades":
    pass
    pass
                    df_missing, self._standardize_aggtrades_format(df_missing)

        # Load existing file
                file_path, self.data_cache_path / file_name
        if file_path.exists():
    pass
    pass
        # Read existing file (Parquet or CSV)
        if file_path.suffix.lower() == ".parquet":
    pass
    pass
                        df_existing, pd.read_parquet(file_path)
                    elif file_path.suffix.lower() == ".csv":
                        df_existing, pd.read_csv(file_path)
                    else:
        return {
                            "success": False,
                            "error": f"Unsupported file format: {file_path.suffix}",
                            "rows_added": 0,
                            "api_calls_made": call_num,
                            "successful_calls": successful_calls,
                        }

        # Combine data
                    df_combined, pd.concat(
                        [df_existing, df_missing], ignore_index = True
                    )
        if "timestamp" in df_combined.columns:
    pass
    pass
                        df_combined = (
                            df_combined.sort_values("timestamp").drop_duplicates(
                                subset=["timestamp"],
                            )
                        )

        # Save back in the same format
        if file_path.suffix.lower() == ".parquet":
    pass
    pass
                        df_combined.to_parquet(
                            file_path, compression="zstd", index = False
                        )
                    elif file_path.suffix.lower() == ".csv":
                        df_combined.to_csv(file_path, index = False)

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
        self, symbol: str, exchange: str, timeframes: list[str] | None, None
    ) -> dict[str, Any]:
        """Regenerate timeframe files after data has been updated / fixed."""
        if timeframes is None:
    pass
    pass
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
            klines_files, list(
    except Exception as e:
        pass
    except Exception as e:
        pass
        self.data_cache_path.glob(
                    f"klines_{exchange}_{symbol}_1m_*.parquet",
                ),
            )
        if not klines_files:
    pass
    pass
                results["success"] = False
                results["errors"].append("No 1m klines files found")
        return results

        # Load and combine all 1m data
            all_1m_data: list[pd.DataFrame] = []
        for file_path in klines_files:
    pass
    pass
        try:
                    df, pd.read_parquet(file_path)
    except Exception as e:
        pass
    except Exception as e:
        pass
                    all_1m_data.append(df)
        except Exception:
                    continue

        if not all_1m_data:
    pass
    pass
                results["success"] = False
                results["errors"].append("No valid 1m data found")
        return results

        # Combine all 1m data
            combined_1m, pd.concat(all_1m_data, ignore_index = True)
        if "timestamp" in combined_1m.columns:
    pass
    pass
                combined_1m, combined_1m.sort_values("timestamp").drop_duplicates(
                    subset=["timestamp"],
                )

        # Regenerate each timeframe
        for timeframe in timeframes:
    pass
    pass
        try:
        # Resample to the target timeframe
                    resampled_df, self._resample_to_timeframe(
                        combined_1m.copy(), timeframe,
                    )

    except Exception as e:
        pass
    except Exception as e:
        pass
        if len(resampled_df) == 0:
    pass
    pass
                        results["failed_timeframes"].append(timeframe)
                        continue

        # Save the resampled data
                    output_path, self._save_resampled_data(
                        resampled_df, symbol, exchange, timeframe,
                    )

        if output_path:
    pass
    pass
                        results["regenerated_files"][timeframe] = str(output_path)
                    else:
                        results["failed_timeframes"].append(timeframe)
                        results["errors"].append(f"Failed to save {timeframe} data")

        except Exception as e:
                    results["failed_timeframes"].append(timeframe)
                    results["errors"].append(f"{timeframe}: {e}")

        # Summary
            _, len(results["regenerated_files"])  # keep for potential logging
            failed, len(results["failed_timeframes"])

        if failed > 0 and failed == len(timeframes):
    pass
    pass
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(f"General error: {e}")

        return results

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    pass
    pass
        """Resample 1m data to target timeframe."""
        if len(df) == 0:
    pass
    pass
        return pd.DataFrame()

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # type: ignore[assignment]
        df, df.set_index("timestamp")

        # Resample based on timeframe
        timeframe_mapping = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
        }

        freq, timeframe_mapping.get(timeframe)
        if freq is None:
    pass
    pass
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
    pass
    pass
        return None

        output_filename, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        output_path, self.data_cache_path / output_filename

        try:
            output_path.parent.mkdir(parents = True, exist_ok = True)
    except Exception as e:
        pass
    except Exception as e:
        pass
            df.to_parquet(output_path, compression="zstd", index = False)
        return output_path
        except Exception:
        return None

    async def process_all_data_types(
        self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"
    ) -> dict[str, Any] | None:
        """Process all gaps in all data types (aggtrades, futures, klines)."""
        from src.utils.logger import system_logger
import logger, system_logger.getChild
        logger, system_logger.getChild("ComprehensiveGapFiller")

        gap_filling_start, datetime.now()
        logger.info(f"🔧 COMPREHENSIVE GAP FILLING FOR {exchange}_{symbol}")
        logger.info(f"📁 Data cache path: {self.data_cache_path}")
        logger.info(f"⏱️  Max API calls per gap: {self.max_api_calls_per_gap}")
        logger.info(f"⏱️  Call delay: {self.call_delay}s")
        logger.info(f"⏱️  Max consecutive empty: {self.max_consecutive_empty}")
        logger.info("-" * 60)

        # Find all files for each data type
        aggtrades_pattern, f"aggtrades_{exchange}_{symbol}_*.parquet"
        aggtrades_csv_pattern, f"aggtrades_{exchange}_{symbol}_*.csv"
        futures_pattern, f"futures_{exchange}_{symbol}_*.parquet"
        futures_csv_pattern, f"futures_{exchange}_{symbol}_*.csv"
        klines_pattern, f"klines_{exchange}_{symbol}_1m_*.parquet"
        klines_csv_pattern, f"klines_{exchange}_{symbol}_1m_*.csv"

        # Get all files
        aggtrades_files, list(self.data_cache_path.glob(aggtrades_pattern)) + list(
        self.data_cache_path.glob(aggtrades_csv_pattern)
        )
        futures_files, list(self.data_cache_path.glob(futures_pattern)) + list(
        self.data_cache_path.glob(futures_csv_pattern)
        )
        klines_files, list(self.data_cache_path.glob(klines_pattern)) + list(
        self.data_cache_path.glob(klines_csv_pattern)
        )

        logger.info("📁 FILE DISCOVERY RESULTS:")
        logger.info(f"  • Aggtrades files: {len(aggtrades_files)}")
        logger.info(f"  • Futures files: {len(futures_files)}")
        logger.info(f"  • Klines files: {len(klines_files)}")

        all_files: list[tuple[Path, str]] = []

        # Add files with types
        for af in aggtrades_files:
    pass
    pass
            all_files.append((af, "aggtrades"))
        for ff in futures_files:
    pass
    pass
            all_files.append((ff, "futures"))
        for kf in klines_files:
    pass
    pass
            all_files.append((kf, "klines"))

        if not all_files:
    pass
    pass
            logger.warning("⚠️  No data files found for gap filling!")
        return None

        logger.info(f"📊 Total files to process: {len(all_files)}")
        logger.info("-" * 60)

        total_files_processed, 0
        total_files_with_gaps, 0
        total_gaps_found, 0
        total_gaps_filled, 0
        total_gaps_failed, 0
        total_api_calls, 0
        total_successful_calls, 0

        # Process each data type
        for data_type in ["aggtrades", "futures", "klines"]:
    pass
    pass
            type_files = [(f, t) for f, t in all_files if t == data_type]

        for file_path, _file_type in type_files:
    pass
    pass
        # Detect gaps based on data type
        if data_type == "aggtrades":
    pass
    pass
                    gaps, self.detect_gaps_in_aggtrades_file(file_path)
                elif data_type == "futures":
                    gaps, self.detect_gaps_in_futures_file(file_path)
                elif data_type == "klines":
                    gaps, self.detect_gaps_in_klines_file(file_path)
                else:
                    continue

                total_files_processed += 1

        if gaps:
    pass
    pass
                    total_files_with_gaps += 1
                    total_gaps_found += len(gaps)

        # Fill each gap with multiple API calls
        for _i, gap in enumerate(gaps):
    pass
    pass
                    result, await self.fill_gap_until_complete(gap, symbol)

                    total_api_calls += int(result.get("api_calls_made", 0))
                    total_successful_calls += int(result.get("successful_calls", 0))

        if result.get("success"):
    pass
    pass
                        total_gaps_filled += 1
                    else:
                        total_gaps_failed += 1

        # Regenerate timeframe files after each successful gap fill
        # (only for aggtrades)
        if data_type == "aggtrades" and total_gaps_filled > 0:
    pass
    pass
                    timeframe_results, await self.regenerate_timeframe_files(
                        symbol, exchange,
                    )
        if timeframe_results.get("success"):
    pass
    pass
        # Count regenerated files as succeeded; no - op otherwise
                        _, len(timeframe_results.get("regenerated_files", {}))
                    else:
        # If regeneration failed, mark as failed timeframe work,
        # not gap filling failure
                        pass

        # Rate limiting between gaps
        await asyncio.sleep(0.2)

        # Summary
        gap_filling_end, datetime.now()
        gap_filling_time, gap_filling_end - gap_filling_start

        logger.info("-" * 60)
        logger.info("📊 COMPREHENSIVE GAP FILLING SUMMARY")
        logger.info(f"⏱️  Total processing time: {gap_filling_time}")
        logger.info(f"📁 Files processed: {total_files_processed}")
        logger.info(f"📁 Files with gaps: {total_files_with_gaps}")
        logger.info(f"❌ Gaps found: {total_gaps_found}")
        logger.info(f"✅ Gaps filled: {total_gaps_filled}")
        logger.info(f"❌ Gaps failed: {total_gaps_failed}")
        logger.info(f"📡 API calls made: {total_api_calls}")
        logger.info(f"📡 Successful API calls: {total_successful_calls}")

        if total_gaps_found > 0:
    pass
    pass
            success_rate = (total_gaps_filled / total_gaps_found) * 100
            logger.info(f"📊 Gap filling success rate: {success_rate:.1f}%")

        if total_gaps_filled > 0:
    pass
    pass
                logger.info("✅ GAP FILLING COMPLETED SUCCESSFULLY!")
            else:
                logger.warning("⚠️  GAP FILLING COMPLETED WITH NO SUCCESSFUL FILLS!")
        else:
            logger.info("✅ NO GAPS FOUND - ALL DATA IS COMPLETE!")

        if total_gaps_found > 0:
    pass
    pass
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
    gap_filler, ComprehensiveGapFiller(data_cache_path)

    try:
        return await gap_filler.process_all_data_types(symbol = symbol, exchange = exchange)
    except Exception as e:
        pass
    except Exception as e:
        pass
    finally:
        await gap_filler.close_session()

if __name__ == "__main__":
    pass
    pass
    asyncio.run(run_comprehensive_gap_filling_pipeline())