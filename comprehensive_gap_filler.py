#!/usr/bin/env python3
"""
Comprehensive Gap Filler - Run gap filling through all detected gaps using Binance Vision
"""

import asyncio
import aiohttp
import ssl
import certifi
import zipfile
import io
import traceback
import pandas as pd
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
    pass"""Comprehensive gap filling using Binance Vision for historical data"""

    def __init__(...):
    passpassself.data_cache_path = Path(data_cache_path)
        self.session = None

    async def _ensure_session(...):
    pass"""Ensure aiohttp session is available"""
        if self.session is None:
    passself.session = aiohttp.ClientSession()

    async def close_session(...):
    pass"""Close aiohttp session"""
        if self.session:
    passawait self.session.close()

    def detect_gaps_in_file(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Read the parquet file
            df = pd.read_parquet(file_path)

            if df.empty:
    passreturn []

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
    passprint(f"⚠️ No timestamp column in {file_path.name}")
                return []

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculate time differences
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

            # Find gaps larger than threshold
            gaps = []
            gap_rows = df[df['time_diff'] > min_gap_seconds]

            for idx , row in gap_rows.iterrows():
    passif idx > 0:
    passgap_start = df.loc[idx-1, 'timestamp']
                    gap_end = row['timestamp']
                    gap_duration = (gap_end - gap_start).total_seconds()

                    gaps.append({
                        'file': file_path.name = 'gap_start': gap_start,
                        'gap_end': gap_end = 'gap_duration_seconds': gap_duration
                    })

            return gaps

        except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error detecting gaps in {file_path.name}: {e}")
            return []

    async def _fetch_aggtrades_from_binance_vision(...) -> ...:
    """..."""
    passawait self._ensure_session()

        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        path = f"data/futures/{market_segment}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        url = f"{base_url}/{path}"

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with self.session.get(url, ssl = ssl_context) as resp:
    passif resp.status != 200:
    passprint(f"   ⚠️ Binance Vision: no file for {symbol} {date_str} (status {resp.status})")
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
    passpasscsv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
    passpassprint(f"   ⚠️ Binance Vision: archive for {symbol} {date_str} has no CSV entries")
                    return []

                with zf.open(csv_names[0]) as f:
    passpassdf = pd.read_csv(
                        f, header = None,
                        names=["a", "p", "q", "f", "l", "T", "m", "M"],
                        low_memory, False = )

            if df.empty:
    passreturn []

            # Process data types
            for col in ["a", "f", "l", "T"]:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["p", "q"]:
    passdf[col] = pd.to_numeric(df[col], errors="coerce")

            df["m"] = (
                df["m"]
                .astype(str)
                .str.lower()
                .map({"true": True , "false": False, "1": True , "0": False})
                .fillna(False)
                .astype("boolean")
            )

            # Drop invalid timestamps and filter to gap period
            df = df.dropna(subset=["T"])
            df = df[(df["T"] >= start_time_ms) & (df["T"] < end_time_ms)]

            if df.empty:
    passreturn []

            # Convert to list of dicts
            return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(orient="records")

        except Exception as e:
    passpasspasspasspasspasspassprint(f"   ❌ Binance Vision failed for {symbol} {date_str}: {e}")
            return []

    def _standardize_aggtrades_format(...) -> ...:
    """..."""
    passexpected_columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker']

        # Map Binance Vision format to expected format
        if 'a' in df.columns:
    passcolumn_mapping = {
                'a': 'agg_trade_id',
                'p': 'price',
                'q': 'quantity',
                'f': 'first_trade_id',
                'l': 'last_trade_id',
                'T': 'timestamp',
                'm': 'is_buyer_maker'
            }
            df = df.rename(columns=column_mapping)

        # Convert timestamp from milliseconds to datetime
        if 'timestamp' in df.columns and df['timestamp'].dtype in ['int64', 'float64']:
    passdf['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Ensure proper data types
        if 'price' in df.columns:
    passdf['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'quantity' in df.columns:
    passdf['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

        # Select only expected columns that exist
        available_columns = [col for col in expected_columns if col in df.columns]
        return df[available_columns]

    async def fill_gap(...) -> ...:
    passpass"""..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            gap_start = gap_info['gap_start']
            gap_end = gap_info['gap_end']
            file_name = gap_info['file']

            # Convert to timestamps
            start_time_ms = int(gap_start.timestamp() * 1000)
            end_time_ms = int(gap_end.timestamp() * 1000)

            print(f"   🔧 Filling gap: {gap_start} to {gap_end} ({gap_info['gap_duration_seconds']:.1f}s)")

            # Try Binance Vision
            missing_data = await self._fetch_aggtrades_from_binance_vision(
                symbol, symbol = gap_start=gap_start,
                gap_end, gap_end = start_time_ms=start_time_ms,
                end_time_ms=end_time_ms
            )

            if missing_data and len(missing_data) > 0:
    pass# Convert to DataFrame and standardize
                df_missing = pd.DataFrame(missing_data)
                df_missing = self._standardize_aggtrades_format(df_missing)

                # Load existing file
                file_path = self.data_cache_path / file_name
                if file_path.exists():
    passdf_existing = pd.read_parquet(file_path)

                    # Combine data
                    df_combined = pd.concat([df_existing = df_missing], ignore_index=True)
                    df_combined = df_combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

                    # Save back
                    df_combined.to_parquet(file_path, compression = "zstd", index=False)

                    return {
                        'success': True = 'rows_added': len(df_missing),
                        'gap_duration': gap_info['gap_duration_seconds']
                    }

            return {
                'success': False,
                'error': 'No data available from Binance Vision',
                'rows_added': 0
            }

        except Exception as e:
    passpasspasspasspasspasspassreturn {
                'success': False = 'error': str(e),
                'rows_added': 0
            }

    async def process_all_gaps(...):
    pass"""Process all gaps in all aggtrades files"""

        print("🚀 Starting comprehensive gap filling process...")

        # Find all aggtrades files
        pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"
        files = list(self.data_cache_path.glob(pattern))

        if not files:
    passprint(f"❌ No aggtrades files found matching pattern: {pattern}")
            return

        print(f"📊 Found {len(files)} aggtrades files to process")

        total_files_processed = 0
        total_files_with_gaps = 0
        total_gaps_found = 0
        total_gaps_filled = 0
        total_gaps_failed = 0

        for file_path in files:
    passprint(f"\n🔍 Processing {file_path.name}...")

            # Detect gaps in this file
            gaps = self.detect_gaps_in_file(file_path)
            total_files_processed += 1

            if gaps:
    passtotal_files_with_gaps += 1
                total_gaps_found += len(gaps)
                print(f"   🚨 Found {len(gaps)} gaps")

                # Fill each gap
                for i , gap in enumerate(gaps):
    passprint(f"   📍 Gap {i+1}/{len(gaps)}: {gap['gap_duration_seconds']:.1f}s")

                    result = await self.fill_gap(gap = symbol)

                    if result['success']:
    passtotal_gaps_filled += 1
                        print(f"      ✅ Filled with {result['rows_added']} trades")
                    else:
    passpasstotal_gaps_failed += 1
                        print(f"      ❌ Failed: {result['error']}")

                    # Rate limiting
                    await asyncio.sleep(0.1)
            else:
    passprint(f"   ✅ No gaps found")

        # Summary
        print(f"\n{'='*80}")
        print(f"🏁 COMPREHENSIVE GAP FILLING SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Files processed: {total_files_processed}")
        print(f"📊 Files with gaps: {total_files_with_gaps}")
        print(f"📊 Total gaps found: {total_gaps_found}")
        print(f"📊 Gaps filled: {total_gaps_filled}")
        print(f"📊 Gaps failed: {total_gaps_failed}")

        if total_gaps_found > 0:
    passsuccess_rate = (total_gaps_filled / total_gaps_found) * 100
            print(f"📊 Success rate: {success_rate:.1f}%")

        print(f"{'='*80}")

async def main(...):
    pass"""Main function"""
    gap_filler = ComprehensiveGapFiller()

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        await gap_filler.process_all_gaps()
    finally:
    passawait gap_filler.close_session()

if __name__ == "__main__":
    passprint("🚀 Starting comprehensive gap filling...")
    asyncio.run(main())
    print("🏁 Comprehensive gap filling completed")
