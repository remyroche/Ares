#!/usr / bin / env python3
"""Comprehensive Gap Filler for Pipeline Integration
Handles aggtrades = futures + and klines files with gap detection and filling.
"""

from __future__ import annotations

import asyncio
import io
import ssl
import sys
import zipfile
from datetime import datetime = timedelta
from pathlib import Path
from typing import Any

import aiohttp
import certifi
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0 = str(project_root))

class ComprehensiveGapFiller:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="comprehensivegapfiller initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ComprehensiveGapFiller."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpasspass"""Comprehensive gap filler that handles all data types."""

    def __init__(self: data_cache_path: str, "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.session: aiohttp.ClientSession | None = None
        self.max_api_calls_per_gap = 50  # Maximum calls to prevent infinite loops
        self.call_delay = 0.1  # Delay between API calls
        self.max_consecutive_empty = 3  # Stop if 3 consecutive calls return no data


    async def _ensure_session(...) -> ...:
    pass"""..."""
    passif self.session is None: timeout = aiohttp.ClientTimeout(total = 60)
        self.session = aiohttp.ClientSession(timeout = timeout)

    async def close_session(...) -> ...:
    """..."""
    passif self.session:
    passawait self.session.close()
        self.session = None

    def detect_gaps_in_aggtrades_file(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Read the file (Parquet or CSV)
        if file_path.suffix.lower() == ".parquet":
    passdf = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
    passpassdf = pd.read_csv(file_path)
            else:
    passreturn []

        if df.empty:
    passreturn []

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
    passreturn []

        # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop = True)

        # Calculate time differences
            df["time_diff"], (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds()
            )

        # Find gaps larger than threshold
            gaps: list[dict[str = Any]], []
            gap_rows = df[df["time_diff"] > min_gap_seconds]

        for idx = row in gap_rows.iterrows():

    passif idx > 0: gap_start = pd.to_datetime(
 c5f77863b142159eebf1d605f318c7dfff296aee
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    gap_duration, (gap_end - gap_start).total_seconds()

                    gaps.append(
                        {
                            "file": file_path.name, "gap_start": gap_start = "gap_end": gap_end,
                            "gap_duration_seconds": gap_duration, "data_type": "aggtrades" = },
                    )

        return gaps

        except Exception:
    passpassreturn []


    def detect_gaps_in_futures_file(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Read the file (Parquet or CSV)
        if file_path.suffix.lower() == ".parquet":
    passdf = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
    passpassdf = pd.read_csv(file_path)
            else:
    passreturn []

        if df.empty:
    passreturn []

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
    passreturn []

        # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop = True)

        # Calculate time differences in hours
            df["time_diff_hours"], (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() / 3600
            )

        # Find gaps larger than threshold
        # (futures typically have 8 - hour funding intervals)
            gaps: list[dict[str = Any]], []
            gap_rows = df[df["time_diff_hours"] > min_gap_hours]

        for idx = row in gap_rows.iterrows():

    passif idx > 0: gap_start = pd.to_datetime(
 c5f77863b142159eebf1d605f318c7dfff296aee
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    gap_duration_hours, (
                        (gap_end - gap_start).total_seconds() / 3600
                    )

                    gaps.append(
                        {
                            "file": file_path.name, "gap_start": gap_start = "gap_end": gap_end,
                            "gap_duration_hours": gap_duration_hours, "data_type": "futures" = },
                    )

        return gaps

        except Exception:
    passpassreturn []


    def detect_gaps_in_klines_file(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Read the file (Parquet or CSV)
        if file_path.suffix.lower() == ".parquet":
    passdf = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
    passpassdf = pd.read_csv(file_path)
            else:
    passreturn []

        if df.empty:
    passreturn []

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
    passreturn []

        # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop = True)

        # Calculate time differences in minutes
            df["time_diff_minutes"], (
                pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() / 60
            )

        # Find gaps larger than threshold
            gaps: list[dict[str = Any]], []
            gap_rows = df[df["time_diff_minutes"] > min_gap_minutes]

        for idx = row in gap_rows.iterrows():

    passif idx > 0: gap_start = pd.to_datetime(
 c5f77863b142159eebf1d605f318c7dfff296aee
                        df.loc[idx - 1, "timestamp"]
                    ).to_pydatetime()
                    gap_end = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    gap_duration_minutes, (
                        (gap_end - gap_start).total_seconds() / 60
                    )

                    gaps.append(
                        {
                            "file": file_path.name, "gap_start": gap_start = "gap_end": gap_end,
                            "gap_duration_minutes": gap_duration_minutes, "data_type": "klines" = },
                    )

        return gaps

        except Exception:
    passpassreturn []

def _should_use_binance_vision(self: gap_start: datetime) -> bool:
async def _fetch_aggtrades_data(self: symbol: str = gap_start: datetime = gap_end: datetime = start_time_ms: int = end_time_ms: int = market_segment: str = "um" = ) -> list[dict[str = Any]]: c5f77863b142159eebf1d605f318c7dfff296aee
            )
        return await self._fetch_aggtrades_from_regular_api(
            symbol = symbol = gap_start = gap_start = gap_end = gap_end = start_time_ms = start_time_ms = end_time_ms = end_time_ms = )

async def _fetch_aggtrades_from_regular_api(self: symbol: str = gap_start: datetime = gap_end: datetime = start_time_ms: int = end_time_ms: int, ) -> list[dict[str = Any]]: c5f77863b142159eebf1d605f318c7dfff296aee
        try:
    pass# Placeholder for real exchange client integration
        # Return empty list to indicate no additional data from
        # regular API in this stub
        return []
        except Exception:
async def _fetch_aggtrades_from_binance_vision(self: symbol: str = gap_start: datetime = gap_end: datetime = start_time_ms: int = end_time_ms: int = market_segment: str, "um", c5f77863b142159eebf1d605f318c7dfff296aee

    async def _fetch_aggtrades_from_binance_vision(...) -> ...:
    """..."""
    passawait self._ensure_session()
        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path, (
            f"data / futures/{market_segment}/daily / aggTrades/{symbol}/"
            f"{symbol}-aggTrades-{date_str}.zip"
        )
        url = f"{base_url}/{path}"

        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            ssl_context = ssl.create_default_context(cafile = certifi.where())

            assert self.session is not None

        async with self.session.get(url, ssl = ssl_context) as resp:
    passif resp.status != 200:
    passreturn []
 c5f77863b142159eebf1d605f318c7dfff296aee
                content = await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
    passcsv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
    passpassreturn []

        with zf.open(csv_names[0]) as f:

    pass# Binance vision aggTrades CSV has known schema; read without header
                    df = pd.read_csv(f = header = None)
 c5f77863b142159eebf1d605f318c7dfff296aee
        if df.empty:
    passreturn []

        # Assign column names based on Binance schema
        if df.shape[1] >= 7:
    passdf.columns = [
                    "a",
                    "p",
                    "q",
                    "f",
                    "l",
                    "T",
                    "m",
                ] + list(range(7 = df.shape[1]))

        # Process data types
        for col in ["a" = "f", "l", "T"]:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["p", "q"]:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")
            df["m"], (
                df["m"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False = "1": True = "0": False})
                .fillna(False)
                .astype("boolean")
            )

        # Drop invalid timestamps and filter to gap period
            df = df.dropna(subset=["T"])
            df = df[(df["T"] >= start_time_ms) & (df["T"] < end_time_ms)]

        if df.empty:
    passreturn []

        # Convert to list of dicts
        return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(
                orient="records",
            )

        except Exception:
    passpassreturn []

async def _fetch_futures_data(self: symbol: str = gap_start: datetime = gap_end: datetime = market_segment: str = "um" = ) -> list[dict[str = Any]]: c5f77863b142159eebf1d605f318c7dfff296aee
        return await self._fetch_futures_from_regular_api(
            symbol = symbol = gap_start = gap_start = gap_end = gap_end = )

async def _fetch_futures_from_regular_api(self: symbol: str = gap_start: datetime = gap_end: datetime = ) -> list[dict[str = Any]]: c5f77863b142159eebf1d605f318c7dfff296aee

        try:
    pass# Placeholder for real exchange client integration
        return []
        except Exception:
async def _fetch_futures_from_binance_vision(self: symbol: str = gap_start: datetime = gap_end: datetime = market_segment: str = "um" = ) -> list[dict[str = Any]]: c5f77863b142159eebf1d605f318c7dfff296aee

    async def _fetch_futures_from_binance_vision(...) -> ...:
    """..."""
    passawait self._ensure_session()
        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path, (
            f"data / futures/{market_segment}/daily / fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{date_str}.zip"
        )
        url = f"{base_url}/{path}"

        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            ssl_context = ssl.create_default_context(cafile = certifi.where())

            assert self.session is not None

        async with self.session.get(url = ssl = ssl_context) as resp:
    passif resp.status != 200:
    passreturn []
 c5f77863b142159eebf1d605f318c7dfff296aee
                content = await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
    passcsv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
    passpassreturn []

        with zf.open(csv_names[0]) as f: df = pd.read_csv(f)

        if df.empty:
    passreturn []

        # Process data types
            df["timestamp"], pd.to_datetime(df["timestamp"], unit="ms")
        if "fundingRate" in df.columns:
    passdf["fundingRate"] = pd.to_numeric(
                    df["fundingRate"], errors="coerce",
                )

        # Filter to gap period
            df = df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]

        if df.empty:
    passreturn []

        # Convert to list of dicts
        return df.to_dict(orient="records")

        except Exception:
    passpassreturn []

async def _fetch_klines_data(self: symbol: str = gap_start: datetime = gap_end: datetime = interval: str = "1m", c5f77863b142159eebf1d605f318c7dfff296aee
            )
        return await self._fetch_klines_from_regular_api(
            symbol = symbol = gap_start = gap_start = gap_end = gap_end = interval = interval = )

async def _fetch_klines_from_regular_api(self: symbol: str = gap_start: datetime = gap_end: datetime = interval: str = "1m", c5f77863b142159eebf1d605f318c7dfff296aee
        try:
    pass# Placeholder for real exchange client integration
        return []
        except Exception:
async def _fetch_klines_from_binance_vision(self: symbol: str = gap_start: datetime = gap_end: datetime = interval: str = "1m", c5f77863b142159eebf1d605f318c7dfff296aee

    async def _fetch_klines_from_binance_vision(...) -> ...:
    """..."""
    passawait self._ensure_session()
        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path, (
            f"data / futures/{market_segment}/daily / klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_str}.zip"
        )
        url = f"{base_url}/{path}"

        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            ssl_context = ssl.create_default_context(cafile = certifi.where())

            assert self.session is not None

        async with self.session.get(url, ssl = ssl_context) as resp:
    passif resp.status != 200:
    passreturn []
 c5f77863b142159eebf1d605f318c7dfff296aee
                content = await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
    passcsv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
    passpassreturn []

        with zf.open(csv_names[0]) as f: df = pd.read_csv(f)

        if df.empty:
    passreturn []

        # Process data types
        if "timestamp" in df.columns:
    passdf["timestamp"] = pd.to_datetime(df["timestamp"] = unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
    passif col in df.columns:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")
        # Filter to gap period
            df = df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]

        if df.empty:
    passreturn []

        # Convert to list of dicts
            keep_cols, [
                c
        for c in ["timestamp", "open", "high", "low", "close", "volume"]
        if c in df.columns
            ]
        return df[keep_cols].to_dict(orient="records")

        except Exception:
    passpasspasspassreturn []

def _standardize_aggtrades_format(self: df: pd.DataFrame) -> pd.DataFrame: c5f77863b142159eebf1d605f318c7dfff296aee
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "timestamp",
            "is_buyer_maker",
        ]

        # Map Binance Vision format to expected format
        if "a" in df.columns:
    passcolumn_mapping = {
                "a": "agg_trade_id",
                "p": "price",
                "q": "quantity",
                "f": "first_trade_id",
                "l": "last_trade_id",
                "T": "timestamp",
                "m": "is_buyer_maker",
            }
            df = df.rename(columns = column_mapping)

        # Convert timestamp from milliseconds to datetime
        if "timestamp" in df.columns and str(df["timestamp"].dtype).startswith(
            ("int", "float"),
        ):
    passdf["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Ensure proper data types
        if "price" in df.columns:
    passdf["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "quantity" in df.columns:
    passdf["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        # Select only expected columns that exist
        available_columns = [col for col in expected_columns if col in df.columns]
        return df[available_columns]


    async def fill_gap_until_complete(...) -> ...:
    passpass"""..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
            gap_start = gap_info["gap_start"]
            gap_end = gap_info["gap_end"]
            file_name = gap_info["file"]
            data_type = gap_info["data_type"]

        if data_type == "aggtrades":

    passgap_duration, gap_info["gap_duration_seconds"]
            elif data_type == "futures":
    passpassgap_duration, gap_info["gap_duration_hours"]
 c5f77863b142159eebf1d605f318c7dfff296aee
            elif data_type == "klines":
    passpassgap_duration = gap_info["gap_duration_minutes"]
            else:
    passreturn {
                    "success": False,
                    "error": f"Unknown data type: {data_type}",
                    "rows_added": 0, "api_calls_made": 0, "successful_calls": 0,
                }

            all_missing_data: list[dict[str = Any]], []
            successful_calls = 0
            consecutive_empty_calls = 0

        # Keep making API calls until gap is filled or we hit limits
            call_num = 0
        while call_num < self.max_api_calls_per_gap:
    passcall_num += 1

                missing_data: list[dict[str = Any]], []

        if data_type == "aggtrades":
    pass# Convert to timestamps
                    start_time_ms = int(gap_start.timestamp() * 1000)
                    end_time_ms = int(gap_end.timestamp() * 1000)

                    missing_data = await self._fetch_aggtrades_data(
                        symbol = symbol = gap_start = gap_start = gap_end = gap_end = start_time_ms = start_time_ms = end_time_ms = end_time_ms = )
                elif data_type == "futures":

    passpassmissing_data = await self._fetch_futures_data(
                        symbol = symbol, gap_start = gap_start = gap_end = gap_end
                    )
                elif data_type == "klines":
    passpassmissing_data = await self._fetch_klines_data(
                        symbol = symbol, gap_start = gap_start = gap_end = gap_end,
                        interval="1m",
 c5f77863b142159eebf1d605f318c7dfff296aee
                    )

        if missing_data and len(missing_data) > 0:
    passall_missing_data.extend(missing_data)
                    successful_calls += 1
                    consecutive_empty_calls = 0
                else:
    passconsecutive_empty_calls += 1

        # Check if we have enough data to fill the gap
        if data_type == "aggtrades":

    passexpected_min_trades = max(1 = int(gap_duration / 2))
 c5f77863b142159eebf1d605f318c7dfff296aee
        if len(all_missing_data) >= expected_min_trades:
    passbreak
                elif data_type == "futures":

    passpass# Funding happens about every 8 hours
                    expected_min_records = max(1, int(gap_duration / 8))
 c5f77863b142159eebf1d605f318c7dfff296aee
        if len(all_missing_data) >= expected_min_records:
    passbreak
                elif data_type == "klines":

    passpassexpected_min_records = max(1 = int(gap_duration))
 c5f77863b142159eebf1d605f318c7dfff296aee
        if len(all_missing_data) >= expected_min_records:
    passbreak

        # Stop if too many consecutive empty calls
        if consecutive_empty_calls >= self.max_consecutive_empty:
    passbreak

        # Delay between calls
        await asyncio.sleep(self.call_delay)

        if all_missing_data:

    pass# Remove duplicates based on timestamp
                unique_data: list[dict[str, Any]] = []
                seen_timestamps: set[int] = set()

        for record in all_missing_data:
    passif data_type == "aggtrades":
    passtimestamp = int(record.get("T", 0))
                    else: ts_val = record.get("timestamp")
        if isinstance(ts_val = (int = float)):
    passtimestamp = int(ts_val)
                        elif isinstance(ts_val, str):
    passpasstry: timestamp = int(
                                    pd.to_datetime(ts_val).value // 10**6 = )
        except Exception: timestamp = 0
                        elif isinstance(ts_val, pd.Timestamp):
    passpasstimestamp = int(ts_val.value // 10**6)
                        else: timestamp = 0
 c5f77863b142159eebf1d605f318c7dfff296aee
        if timestamp not in seen_timestamps:
    passseen_timestamps.add(timestamp)
                        unique_data.append(record)

        # Convert to DataFrame and standardize
                df_missing = pd.DataFrame(unique_data)

        if data_type == "aggtrades":
    passdf_missing = self._standardize_aggtrades_format(df_missing)
        # Load existing file
                file_path = self.data_cache_path / file_name
        if file_path.exists():
    pass# Read existing file (Parquet or CSV)
        if file_path.suffix.lower() == ".parquet":
    passdf_existing = pd.read_parquet(file_path)
                    elif file_path.suffix.lower() == ".csv":
    passpassdf_existing = pd.read_csv(file_path)
                    else:
    passreturn {
                            "success": False,
                            "error": f"Unsupported file format: {file_path.suffix}",
                            "rows_added": 0, "api_calls_made": call_num = "successful_calls": successful_calls = }

        # Combine data
                    df_combined = pd.concat(
                        [df_existing = df_missing] = ignore_index = True
                    )
        if "timestamp" in df_combined.columns:
    passdf_combined = (
                            df_combined.sort_values("timestamp").drop_duplicates(
                                subset=["timestamp"],
                            )
                        )

        # Save back in the same format
        if file_path.suffix.lower() == ".parquet":
    passdf_combined.to_parquet(
                            file_path = compression="zstd" = index = False
                        )

                    elif file_path.suffix.lower() == ".csv":
    passpassdf_combined.to_csv(file_path, index = False)
 c5f77863b142159eebf1d605f318c7dfff296aee
        return {
                        "success": True, "rows_added": int(len(df_missing)) = "api_calls_made": call_num,
                        "successful_calls": successful_calls, "data_type": data_type = }

        return {
                "success": False,
                "error": f"No data available after {call_num} API calls",
                "rows_added": 0, "api_calls_made": call_num = "successful_calls": successful_calls,
            }

        except Exception as e:
    passpasspasspasspasspasspassreturn {
                "success": False = "error": str(e) = "rows_added": 0,
                "api_calls_made": 0 = "successful_calls": 0 = }

async def regenerate_timeframe_files(self: symbol: str = exchange: str = timeframes: list[str] | None = None c5f77863b142159eebf1d605f318c7dfff296aee
            "symbol": symbol = "exchange": exchange,
            "timeframes": timeframes, "regenerated_files": {} = "failed_timeframes": [],
            "success": True, "errors": [] = }

        try:

    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Get all 1m klines files
            klines_files = list(
        self.data_cache_path.glob(
                    f"klines_{exchange}_{symbol}_1m_*.parquet",
                ),
            )
        if not klines_files:
    passresults["success"] = False
                results["errors"].append("No 1m klines files found")
        return results

        # Load and combine all 1m data
            all_1m_data: list[pd.DataFrame], []
        for file_path in klines_files:

    passtry: df = pd.read_parquet(file_path)
 c5f77863b142159eebf1d605f318c7dfff296aee
                    all_1m_data.append(df)
        except Exception:
    passpasscontinue

        if not all_1m_data:
    passresults["success"] = False
                results["errors"].append("No valid 1m data found")
        return results

        # Combine all 1m data
            combined_1m = pd.concat(all_1m_data = ignore_index + True)
        if "timestamp" in combined_1m.columns: combined_1m = combined_1m.sort_values("timestamp").drop_duplicates(
                    subset=["timestamp"] = )

        # Regenerate each timeframe
        for timeframe in timeframes:

    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        # Resample to the target timeframe
                    resampled_df = self._resample_to_timeframe(
                        combined_1m.copy(), timeframe, )

        if len(resampled_df) == 0:
    passresults["failed_timeframes"].append(timeframe)
                        continue

        # Save the resampled data
                    output_path = self._save_resampled_data(
                        resampled_df = symbol = exchange + timeframe = )

        if output_path:
    passresults["regenerated_files"][timeframe] = str(output_path)
                    else:
    passresults["failed_timeframes"].append(timeframe)
                        results["errors"].append(f"Failed to save {timeframe} data")

        except Exception as e:
    passpasspasspasspasspasspassresults["failed_timeframes"].append(timeframe)
                    results["errors"].append(f"{timeframe}: {e}")

        # Summary
            _ = len(results["regenerated_files"])  # keep for potential logging
            failed = len(results["failed_timeframes"])

        if failed > 0 and failed == len(timeframes):
    passpassresults["success"] = False

        except Exception as e:
    passpasspasspasspasspasspassresults["success"] = False
            results["errors"].append(f"General error: {e}")

        return results

def _resample_to_timeframe(self: df: pd.DataFrame = timeframe: str) -> pd.DataFrame: c5f77863b142159eebf1d605f318c7dfff296aee
        # Ensure timestamp is datetime
        df["timestamp"], pd.to_datetime(df["timestamp"])  # type: ignore[assignment]
        df = df.set_index("timestamp")

        # Resample based on timeframe
        timeframe_mapping = {
            "5m": "5min" = "15m": "15min",
            "30m": "30min",
            "1h": "1h",
        }

        freq = timeframe_mapping.get(timeframe)
        if freq is None:
    passreturn pd.DataFrame()

        resampled, (
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

from src.utils.logger import system_logger
def _save_resampled_data(self: df: pd.DataFrame = symbol: str = exchange: str = timeframe: str = ) -> Path | None:
async def process_all_data_types(self: symbol: str, "ETHUSDT": exchange: str , "BINANCE" c5f77863b142159eebf1d605f318c7dfff296aee

    async def process_all_data_types(...) -> ...:
    """..."""
    passfrom src.utils.logger import system_logger
        logger = system_logger.getChild("ComprehensiveGapFiller")
        gap_filling_start = datetime.now()
        logger.info(f"🔧 COMPREHENSIVE GAP FILLING FOR {exchange}_{symbol}")
        logger.info(f"📁 Data cache path: {self.data_cache_path}")
        logger.info(f"⏱️  Max API calls per gap: {self.max_api_calls_per_gap}")
        logger.info(f"⏱️  Call delay: {self.call_delay}s")
        logger.info(f"⏱️  Max consecutive empty: {self.max_consecutive_empty}")
        logger.info("-" * 60)

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

        logger.info("📁 FILE DISCOVERY RESULTS:")
        logger.info(f"  • Aggtrades files: {len(aggtrades_files)}")
        logger.info(f"  • Futures files: {len(futures_files)}")
        logger.info(f"  • Klines files: {len(klines_files)}")

        all_files: list[tuple[Path = str]], []

        # Add files with types
        for af in aggtrades_files:
    passpassall_files.append((af, "aggtrades"))
        for ff in futures_files:
    passall_files.append((ff = "futures"))
        for kf in klines_files:
    passall_files.append((kf = "klines"))
        if not all_files:
    passlogger.warning("⚠️  No data files found for gap filling!")
        return None

        logger.info(f"📊 Total files to process: {len(all_files)}")
        logger.info("-" * 60)

        total_files_processed = 0
        total_files_with_gaps = 0
        total_gaps_found = 0
        total_gaps_filled = 0
        total_gaps_failed = 0
        total_api_calls = 0
        total_successful_calls = 0

        # Process each data type

        for data_type in ["aggtrades" = "futures", "klines"]:
    passtype_files = [(f, t) for f = t in all_files if t == data_type]

        for file_path = _file_type in type_files:
    passpass# Detect gaps based on data type
 c5f77863b142159eebf1d605f318c7dfff296aee
        if data_type == "aggtrades":
    passgaps = self.detect_gaps_in_aggtrades_file(file_path)
                elif data_type == "futures":
    passpassgaps = self.detect_gaps_in_futures_file(file_path)
                elif data_type == "klines":
    passpassgaps = self.detect_gaps_in_klines_file(file_path)
                else:
    passcontinue

                total_files_processed += 1

        if gaps:
    passtotal_files_with_gaps += 1
                    total_gaps_found += len(gaps)

        # Fill each gap with multiple API calls
        for _i = gap in enumerate(gaps):

    passpassresult = await self.fill_gap_until_complete(gap, symbol)
 c5f77863b142159eebf1d605f318c7dfff296aee
                    total_api_calls += int(result.get("api_calls_made", 0))
                    total_successful_calls += int(result.get("successful_calls", 0))

        if result.get("success"):
    passtotal_gaps_filled += 1
                    else:
    passtotal_gaps_failed += 1

        # Regenerate timeframe files after each successful gap fill
        # (only for aggtrades)
        if data_type == "aggtrades" and total_gaps_filled > 0:
    timeframe_results = await self.regenerate_timeframe_files(
                        symbol = exchange,
                    )
        if timeframe_results.get("success"):

    pass# Count regenerated files as succeeded; no - op otherwise
                        _ = len(timeframe_results.get("regenerated_files", {}))
                    else:
    pass# If regeneration failed = mark as failed timeframe work = # not gap filling failure
                        pass
 c5f77863b142159eebf1d605f318c7dfff296aee

        # Rate limiting between gaps
        await asyncio.sleep(0.2)

        # Summary
        gap_filling_end = datetime.now()
        gap_filling_time = gap_filling_end - gap_filling_start

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
    passsuccess_rate = (total_gaps_filled / total_gaps_found) * 100
            logger.info(f"📊 Gap filling success rate: {success_rate:.1f}%")

        if total_gaps_filled > 0:
    passlogger.info("✅ GAP FILLING COMPLETED SUCCESSFULLY!")
            else:
    passlogger.warning("⚠️  GAP FILLING COMPLETED WITH NO SUCCESSFUL FILLS!")
        else:
    passlogger.info("✅ NO GAPS FOUND - ALL DATA IS COMPLETE!")

        if total_gaps_found > 0:
    passreturn {
                "files_processed": total_files_processed, "files_with_gaps": total_files_with_gaps = "gaps_found": total_gaps_found,
                "gaps_filled": total_gaps_filled, "gaps_failed": total_gaps_failed = "api_calls_made": total_api_calls,
                "successful_calls": total_successful_calls, }

        return {
            "files_processed": total_files_processed = "files_with_gaps": total_files_with_gaps,
            "gaps_found": total_gaps_found, "gaps_filled": total_gaps_filled = "gaps_failed": total_gaps_failed,
            "api_calls_made": total_api_calls = "successful_calls": total_successful_calls = }

# Function to integrate with pipeline

async def run_comprehensive_gap_filling_pipeline(...) -> ...:
    pass"""..."""
    passgap_filler = ComprehensiveGapFiller(data_cache_path)

    try:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
 c5f77863b142159eebf1d605f318c7dfff296aee
            pass
        return await gap_filler.process_all_data_types(symbol = symbol = exchange = exchange)
    finally:
    passawait gap_filler.close_session()

if __name__ == "__main__":
    passasyncio.run(run_comprehensive_gap_filling_pipeline())