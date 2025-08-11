#!/usr/bin/env python3
"""
Batch-migrate flat Parquet directories under data/training/parquet to
partitioned Hive-style datasets under data_cache/parquet/* using
ParquetDatasetManager.

Usage:
  python scripts/migrate_parquet_datasets.py \
    --exchange BINANCE --symbol ETHUSDT --timeframe 1m \
    [--src-base data/training/parquet] [--dst-base data_cache/parquet]

Notes:
  - Static columns exchange/symbol/timeframe will be added if missing.
  - Existing partitioned data will be appended/overwritten per dataset manager behavior.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.logger import system_logger


def migrate_dir(
    src_dir: Path,
    dst_base_dir: Path,
    schema_name: str,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> None:
    from src.training.enhanced_training_manager_optimized import ParquetDatasetManager

    logger = system_logger.getChild("ParquetMigration")
    pdm = ParquetDatasetManager(logger=logger)

    static_columns = {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
    }

    logger.info(f"Migrating {src_dir} -> {dst_base_dir} (schema={schema_name})")
    pdm.migrate_flat_parquet_dir_to_partitioned(
        src_dir=str(src_dir),
        dst_base_dir=str(dst_base_dir),
        schema_name=schema_name,
        static_columns=static_columns,
        compression="zstd"
        if schema_name in {"klines", "aggtrades", "futures"}
        else "snappy",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate flat Parquet dirs to partitioned datasets"
    )
    parser.add_argument("--exchange", default=os.environ.get("AresExchange", "BINANCE"))
    parser.add_argument("--symbol", default=os.environ.get("AresSymbol", "ETHUSDT"))
    parser.add_argument("--timeframe", default=os.environ.get("AresTimeframe", "1m"))
    parser.add_argument("--src-base", default="data/training/parquet")
    parser.add_argument("--dst-base", default="data_cache/parquet")
    args = parser.parse_args()

    src_base = Path(args.src_base)
    dst_base = Path(args.dst_base)

    if not src_base.exists():
        system_logger.warning(f"Source base does not exist: {src_base}")
        return 0

    # Map subdirectory names to schema names
    dataset_map = {
        "klines": "klines",
        "aggtrades": "aggtrades",
        "futures": "futures",
        "features": "split",
        "labeled": "split",
        "regime_data": "split",
        "vectorized_features": "split",
    }

    migrated_any = False
    for subdir_name, schema_name in dataset_map.items():
        src_dir = src_base / subdir_name
        if not src_dir.exists() or not any(src_dir.rglob("*.parquet")):
            continue
        dst_dir = dst_base / subdir_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        migrate_dir(
            src_dir=src_dir,
            dst_base_dir=dst_dir,
            schema_name=schema_name,
            exchange=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        migrated_any = True

    if not migrated_any:
        system_logger.info(f"No flat Parquet directories found under {src_base}")
    else:
        system_logger.info(
            f"Migration complete. Partitioned datasets available under {dst_base}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
