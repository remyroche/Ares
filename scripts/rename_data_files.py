#!/usr/bin/env python3
"""
Script to rename existing data files in data_cache to include exchange name prefix.
This script renames files from the old format to the new format that includes exchange names.
"""

import glob
import shutil
from pathlib import Path

from src.utils.warning_symbols import failed, missing, warning


def _find_files(base_dir: Path, pattern: str) -> list[Path]:
	"""Find files in base_dir matching pattern."""
	return [Path(p) for p in glob.glob(str(base_dir / pattern))]


def _build_new_name(exchange_name: str, old_path: Path, kind: str) -> tuple[bool, str]:
	"""Build new file name based on kind and original name.

	Returns (ok, new_name)
	"""
	parts=old_path.name.split("_")
	if kind== "klines":
		# klines_ETHUSDT_1m_2025-07.csv -> klines_BINANCE_ETHUSDT_1m_2025-07.csv
		if len(parts) >= 4:
			return True, f"klines_{exchange_name}_{parts[1]}_{parts[2]}_{parts[3]}"
		return False, warning(f" Skipping {old_path.name} - unexpected format")
	if kind== "aggtrades":
		# aggtrades_ETHUSDT_2025-07-29.csv -> aggtrades_BINANCE_ETHUSDT_2025-07-29.csv
		if len(parts) >= 3:
			return True, f"aggtrades_{exchange_name}_{parts[1]}_{parts[2]}"
		return False, warning(f" Skipping {old_path.name} - unexpected format")
	if kind== "futures":
		# futures_ETHUSDT_2025-07.csv -> futures_BINANCE_ETHUSDT_2025-07.csv
		if len(parts) >= 3:
			return True, f"futures_{exchange_name}_{parts[1]}_{parts[2]}"
		return False, warning(f" Skipping {old_path.name} - unexpected format")
	return False, warning(f" Skipping {old_path.name} - unknown pattern")


def rename_data_files() -> bool:
	"""Rename existing data files to include exchange name prefix."""
	data_cache_dir=Path("data_cache")

	if not data_cache_dir.exists():
		print(missing("data_cache directory not found!"))
		return False

	# Define the exchange name for existing files
	exchange_name="BINANCE"

	# Patterns to match existing files
	patterns = [
		("klines_ETHUSDT_1m_*.csv", f"klines_{exchange_name}_ETHUSDT_1m_*.csv"),
		("aggtrades_ETHUSDT_*.csv", f"aggtrades_{exchange_name}_ETHUSDT_*.csv"),
		("futures_ETHUSDT_*.csv", f"futures_{exchange_name}_ETHUSDT_*.csv"),
	]

	total_renamed=0

	for old_pattern, _new_pattern in patterns:
		# Find files matching the old pattern
		old_files=_find_files(data_cache_dir, old_pattern)

		if not old_files:
			print(f"ℹ️  No files found matching pattern: {old_pattern}")
			continue

		print(f"📁 Found {len(old_files)} files matching: {old_pattern}")

		for old_path in old_files:
			# Determine kind from pattern
			if old_pattern.startswith("klines"):
				ok, new_name=_build_new_name(exchange_name, old_path, "klines")
			elif old_pattern.startswith("aggtrades"):
				ok, new_name=_build_new_name(exchange_name, old_path, "aggtrades")
			elif old_pattern.startswith("futures"):
				ok, new_name=_build_new_name(exchange_name, old_path, "futures")
			else:
				ok, new_name=False, warning(f" Skipping {old_path.name} - unknown pattern")

			if not ok:
				print(new_name)
				continue

			new_path=old_path.parent / new_name

			# Check if new file already exists
			if new_path.exists():
				print(warning(f" Skipping {old_path.name} - {new_name} already exists"))
				continue

			try:
				# Rename the file
				shutil.move(str(old_path), str(new_path))
				print(f"✅ Renamed: {old_path.name} -> {new_name}")
				total_renamed += 1
			except Exception as e:  # noqa: BLE001
				print(warning(f"Error renaming {old_path.name}: {e}"))

	print(f"\n🎉 Renamed {total_renamed} files successfully!")
	return True


if __name__== "__main__":
	print("🔄 Renaming existing data files to include exchange name...")
	success=rename_data_files()

	if success:
		print("✅ File renaming completed successfully!")
	else:
		print(failed("File renaming failed!"))
