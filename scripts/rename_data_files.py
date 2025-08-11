#!/usr/bin/env python3
"""
Script to rename existing data files in data_cache to include exchange name prefix.
This script renames files from the old format to the new format that includes exchange names.
"""

import glob
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
import shutil
from pathlib import Path


def rename_data_files():
    """Rename existing data files to include exchange name prefix."""
    data_cache_dir = Path("data_cache")

    if not data_cache_dir.exists():
        print(missing("data_cache directory not found!")))
        return False

    # Define the exchange name for existing files
    exchange_name = "BINANCE"

    # Patterns to match existing files
    patterns = [
        ("klines_ETHUSDT_1m_*.csv", f"klines_{exchange_name}_ETHUSDT_1m_*.csv"),
        ("aggtrades_ETHUSDT_*.csv", f"aggtrades_{exchange_name}_ETHUSDT_*.csv"),
        ("futures_ETHUSDT_*.csv", f"futures_{exchange_name}_ETHUSDT_*.csv"),
    ]

    total_renamed = 0

    for old_pattern, _new_pattern in patterns:
        # Find files matching the old pattern
        old_files = glob.glob(str(data_cache_dir / old_pattern))

        if not old_files:
            print(f"ℹ️  No files found matching pattern: {old_pattern}")
            continue

        print(f"📁 Found {len(old_files)} files matching: {old_pattern}")

        for old_file_path in old_files:
            old_path = Path(old_file_path)

            # Extract the date/interval part from the filename
            if "klines" in old_pattern:
                # klines_ETHUSDT_1m_2025-07.csv -> klines_BINANCE_ETHUSDT_1m_2025-07.csv
                parts = old_path.name.split("_")
                if len(parts) >= 4:
                    new_name = (
                        f"klines_{exchange_name}_{parts[1]}_{parts[2]}_{parts[3]}"
                    )
                else:
                    print(warning(" Skipping {old_path.name} - unexpected format")))
                    continue
            elif "aggtrades" in old_pattern:
                # aggtrades_ETHUSDT_2025-07-29.csv -> aggtrades_BINANCE_ETHUSDT_2025-07-29.csv
                parts = old_path.name.split("_")
                if len(parts) >= 3:
                    new_name = f"aggtrades_{exchange_name}_{parts[1]}_{parts[2]}"
                else:
                    print(warning(" Skipping {old_path.name} - unexpected format")))
                    continue
            elif "futures" in old_pattern:
                # futures_ETHUSDT_2025-07.csv -> futures_BINANCE_ETHUSDT_2025-07.csv
                parts = old_path.name.split("_")
                if len(parts) >= 3:
                    new_name = f"futures_{exchange_name}_{parts[1]}_{parts[2]}"
                else:
                    print(warning(" Skipping {old_path.name} - unexpected format")))
                    continue
            else:
                print(warning(" Skipping {old_path.name} - unknown pattern")))
                continue

            new_path = old_path.parent / new_name

            # Check if new file already exists
            if new_path.exists():
                print(warning(" Skipping {old_path.name} - {new_name} already exists")))
                continue

            try:
                # Rename the file
                shutil.move(str(old_path), str(new_path))
                print(f"✅ Renamed: {old_path.name} -> {new_name}")
                total_renamed += 1
            except Exception as e:
                print(warning("Error renaming {old_path.name}: {e}")))

    print(f"\n🎉 Renamed {total_renamed} files successfully!")
    return True


if __name__ == "__main__":
    print("🔄 Renaming existing data files to include exchange name...")
    success = rename_data_files()

    if success:
        print("✅ File renaming completed successfully!")
    else:
        print(failed("File renaming failed!")))
