#!/usr/bin/env python3
"""
Fix multi-timeframe data alignment issues in the Ares trading system.

This script addresses the following issues:
1. Corrupted timestamps in timeframe files (1970 epoch)
2. Missing or invalid timeframe data
3. Data alignment issues between different timeframes
"""

import glob
import logging
import os

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_timeframe_data():
    pass
    pass
    """Fix timeframe data issues by creating proper consolidated files with correct timestamps."""
    logger.info("🔧 Fixing multi-timeframe data issues...")

    # Check if data_cache directory exists
    if not os.path.exists("data_cache"):
    pass
    pass
        logger.error("❌ data_cache directory does not exist!")
        return False

    # Look for existing klines files
    klines_patterns = ["klines_*.parquet", "klines_*.csv"]

    klines_files = []
    for pattern in klines_patterns:
    pass
    pass
        klines_files.extend(glob.glob(os.path.join("data_cache", pattern)))

    if not klines_files:
    pass
    pass
        logger.warning("⚠️ No klines files found in data_cache")
        return False

    logger.info(f"📁 Found {len(klines_files)} klines files")

    # Find the most recent consolidated 1m file
    consolidated_1m_patterns = [
        "klines_*_1m_consolidated.parquet",
        "klines_*_1m_consolidated.csv",
    ]

    consolidated_1m_files = []
    for pattern in consolidated_1m_patterns:
    pass
    pass
        consolidated_1m_files.extend(glob.glob(os.path.join("data_cache", pattern)))

    if not consolidated_1m_files:
    pass
    pass
        logger.warning("⚠️ No consolidated 1m files found")
        return False

    # Use the most recent consolidated 1m file
    latest_1m_file = max(consolidated_1m_files, key=os.path.getctime)
    logger.info(f"📁 Using 1m data from: {latest_1m_file}")

    try:
        # Load the 1m data
    except Exception as e:
        pass
    except Exception as e:
        pass
        if latest_1m_file.endswith(".parquet"):
    pass
    pass
            df_1m = pd.read_parquet(latest_1m_file)
        else:
            df_1m = pd.read_csv(latest_1m_file)

        # Ensure we have timestamp column
        if "timestamp" not in df_1m.columns:
    pass
    pass
            logger.error("❌ 1m data missing timestamp column")
            return False

        # Convert timestamp to datetime
        df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
        df_1m = df_1m.set_index("timestamp")

        # Validate timestamps
        if df_1m.index.min().year == 1970:
    pass
    pass
            logger.error("❌ 1m data has corrupted 1970 timestamps")
            return False

        logger.info(f"✅ Loaded 1m data: {len(df_1m)} records")
        logger.info(f"📅 Date range: {df_1m.index.min()} to {df_1m.index.max()}")

        # Create timeframe files for 5m = 15m, 30m = 1h, 4h = 1d
        timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]

        for timeframe in timeframes:
    pass
    pass
            logger.info(f"📝 Creating {timeframe} timeframe data...")

            try:
                # Resample 1m data to target timeframe
    except Exception as e:
        pass
    except Exception as e:
        pass
                if timeframe == "5m":
    pass
    pass
                    df_resampled = (
                        df_1m.resample("5min")
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )
                elif timeframe == "15m":
                    df_resampled = (
                        df_1m.resample("15min")
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )
                elif timeframe == "30m":
                    df_resampled = (
                        df_1m.resample("30min")
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )
                elif timeframe == "1h":
                    df_resampled = (
                        df_1m.resample("1h")
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )
                elif timeframe == "4h":
                    df_resampled = (
                        df_1m.resample("4h")
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )
                elif timeframe == "1d":
                    df_resampled = (
                        df_1m.resample("1D")
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            }
                        )
                        .dropna()
                    )

                # Reset index to make timestamp a column
                df_resampled = df_resampled.reset_index()

                # Save to file
                output_path = f"data_cache/klines_BINANCE_ETHUSDT_{timeframe}_consolidated.parquet"
                df_resampled.to_parquet(output_path, index=False)

                logger.info(f"✅ Created {timeframe} timeframe file: {output_path}")
                logger.info(f"📊 {timeframe} records: {len(df_resampled)}")
                logger.info(
                    f"📅 {timeframe} date range: {df_resampled['timestamp'].min()} to {df_resampled['timestamp'].max()}"
                )

            except Exception as e:
                logger.error(f"❌ Error creating {timeframe} timeframe data: {e}")
                continue

        return True

    except Exception as e:
        logger.error(f"❌ Error fixing timeframe data: {e}")
        return False


def validate_timeframe_files():
    pass
    pass
    """Validate that all timeframe files have correct timestamps."""
    logger.info("🔍 Validating timeframe files...")

    # Check for timeframe files
    timeframe_patterns = [
        "klines_*_5m_consolidated.parquet",
        "klines_*_15m_consolidated.parquet",
        "klines_*_30m_consolidated.parquet",
        "klines_*_1h_consolidated.parquet",
        "klines_*_4h_consolidated.parquet",
        "klines_*_1d_consolidated.parquet",
    ]

    valid_files = []
    corrupted_files = []

    for pattern in timeframe_patterns:
    pass
    pass
        files = glob.glob(os.path.join("data_cache", pattern))
        for file_path in files:
    pass
    pass
            try:
                df = pd.read_parquet(file_path)

    except Exception as e:
        pass
    except Exception as e:
        pass
                if "timestamp" not in df.columns:
    pass
    pass
                    logger.warning(
                        f"⚠️ {os.path.basename(file_path)} missing timestamp column"
                    )
                    corrupted_files.append(file_path)
                    continue

                # Convert timestamps
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Check for 1970 timestamps
                if df["timestamp"].min().year == 1970:
    pass
    pass
                    logger.error(
                        f"❌ {os.path.basename(file_path)} has corrupted 1970 timestamps"
                    )
                    corrupted_files.append(file_path)
                    continue

                # Check for reasonable date range
                if df["timestamp"].min().year < 2000:
    pass
    pass
                    logger.error(
                        f"❌ {os.path.basename(file_path)} has timestamps before 2000"
                    )
                    corrupted_files.append(file_path)
                    continue

                valid_files.append(file_path)
                logger.info(f"✅ {os.path.basename(file_path)} validated successfully")
                logger.info(
                    f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
                )

            except Exception as e:
                logger.error(f"❌ Error validating {file_path}: {e}")
                corrupted_files.append(file_path)

    logger.info(
        f"📊 Validation results: {len(valid_files)} valid files = {len(corrupted_files)} corrupted files"
    )

    if corrupted_files:
    pass
    pass
        logger.warning("⚠️ Found corrupted files that should be removed:")
        for file_path in corrupted_files:
    pass
    pass
            logger.warning(f"   - {os.path.basename(file_path)}")

    return len(valid_files) > 0


def cleanup_corrupted_files():
    pass
    pass
    """Clean up corrupted timeframe files."""
    logger.info("🧹 Cleaning up corrupted timeframe files...")

    # Look for files with 1970 timestamps
    timeframe_patterns = [
        "klines_*_5m_consolidated.parquet",
        "klines_*_15m_consolidated.parquet",
        "klines_*_30m_consolidated.parquet",
        "klines_*_1h_consolidated.parquet",
        "klines_*_4h_consolidated.parquet",
        "klines_*_1d_consolidated.parquet",
    ]

    cleaned_files = []

    for pattern in timeframe_patterns:
    pass
    pass
        files = glob.glob(os.path.join("data_cache", pattern))
        for file_path in files:
    pass
    pass
            try:
                df = pd.read_parquet(file_path)

    except Exception as e:
        pass
    except Exception as e:
        pass
                if "timestamp" in df.columns:
    pass
    pass
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Check for 1970 timestamps
                    if df["timestamp"].min().year == 1970:
    pass
    pass
                        logger.warning(
                            f"🗑️ Removing corrupted file: {os.path.basename(file_path)}"
                        )
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                        continue

            except Exception as e:
                logger.warning(
                    f"🗑️ Removing unreadable file: {os.path.basename(file_path)}"
                )
                os.remove(file_path)
                cleaned_files.append(file_path)

    logger.info(f"🧹 Cleaned up {len(cleaned_files)} corrupted files")
    return len(cleaned_files)


def main():
    pass
    pass
    """Main function to fix multi-timeframe data issues."""
    logger.info("🚀 Starting multi-timeframe data fixes...")

    # Clean up corrupted files first
    cleanup_corrupted_files()

    # Validate existing files
    if not validate_timeframe_files():
    pass
    pass
        logger.warning("⚠️ No valid timeframe files found")

    # Fix timeframe data
    if not fix_timeframe_data():
    pass
    pass
        logger.error("❌ Failed to fix timeframe data")
        return

    # Validate again after fixes
    if not validate_timeframe_files():
    pass
    pass
        logger.error("❌ Timeframe files still have issues after fixes")
        return

    logger.info("✅ Multi-timeframe data fixes completed!")


if __name__ == "__main__":
    pass
    pass
    main()
