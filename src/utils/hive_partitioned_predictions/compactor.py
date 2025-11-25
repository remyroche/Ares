"""
Monthly compactor for Hive-partitioned predictions.

Consolidates daily files into monthly_consolidated.parquet to solve
the "small files problem" and reduce inode usage by 95%+.

Features:
- Race condition protection with lock files
- Atomic consolidation (write to temp, then rename)
- Automatic cleanup of daily files after consolidation
- Comprehensive logging and error handling
"""
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import logging
import fcntl
import os
import time

from .constants import (
    LAYER_PATHS,
    PREDICTIONS_DIR,
    COMPACTION_LOCK_FILE,
    PARQUET_COMPRESSION,
    PARQUET_ENGINE,
)


logger = logging.getLogger(__name__)


class CompactionLock:
    """
    Context manager for compaction lock files.

    Prevents race conditions when multiple processes try to compact
    the same month simultaneously.
    """

    def __init__(self, lock_path: Path, timeout: int = 300):
        """
        Initialize compaction lock.

        Args:
            lock_path: Path to lock file
            timeout: Lock acquisition timeout in seconds (default: 5 minutes)
        """
        self.lock_path = lock_path
        self.timeout = timeout
        self.lock_file = None

    def __enter__(self):
        """Acquire lock (blocking with timeout)."""
        start_time = time.time()

        while True:
            try:
                # Create lock file with exclusive access
                self.lock_file = open(self.lock_path, 'w')
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Write PID and timestamp to lock file
                self.lock_file.write(f"{os.getpid()}\n")
                self.lock_file.write(f"{datetime.now().isoformat()}\n")
                self.lock_file.flush()

                logger.debug(f"Acquired lock: {self.lock_path}")
                return self

            except IOError:
                # Lock is held by another process
                elapsed = time.time() - start_time

                if elapsed > self.timeout:
                    raise TimeoutError(
                        f"Could not acquire lock {self.lock_path} "
                        f"after {self.timeout}s"
                    )

                # Wait and retry
                logger.debug(f"Waiting for lock: {self.lock_path}")
                time.sleep(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock and clean up."""
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
                self.lock_path.unlink()
                logger.debug(f"Released lock: {self.lock_path}")
            except Exception as e:
                logger.warning(f"Error releasing lock {self.lock_path}: {e}")


class MonthlyCompactor:
    """
    Compact daily files into monthly_consolidated.parquet at month end.

    Run this as a scheduled job on the 1st of each month.

    Benefits:
    - Reduces file count by ~30x (96 rows/file -> 2880 rows/file)
    - Reduces inode usage by 95%+
    - Improves read performance (1 file vs 30 files)
    - Saves disk space with better compression

    Safety:
    - Lock files prevent race conditions
    - Atomic writes (temp file + rename)
    - Preserves daily files until consolidation succeeds
    - Comprehensive error handling
    """

    def __init__(
        self,
        layer_name: str,
        base_path: Optional[Path] = None,
        delete_daily_files: bool = True
    ):
        """
        Initialize the monthly compactor.

        Args:
            layer_name: Layer name (specialists, base_models, etc.)
            base_path: Optional base path override (for testing)
            delete_daily_files: If True, delete daily files after consolidation
                               (default: True)
        """
        if layer_name not in LAYER_PATHS and base_path is None:
            raise ValueError(
                f"Unknown layer: {layer_name}. "
                f"Must be one of: {list(LAYER_PATHS.keys())}"
            )

        self.layer_name = layer_name
        self.delete_daily_files = delete_daily_files

        if base_path is not None:
            self.base_path = base_path / PREDICTIONS_DIR
        else:
            self.base_path = LAYER_PATHS[layer_name] / PREDICTIONS_DIR

        logger.debug(
            f"Initialized MonthlyCompactor for {layer_name} "
            f"at {self.base_path}"
        )

    def compact_previous_month(self) -> Dict[str, int]:
        """
        Compact previous month's daily files.

        Example: Run on Dec 1st to compact November.

        Returns:
            Dict with compaction statistics:
            {
                'model_versions_processed': int,
                'months_compacted': int,
                'files_before': int,
                'files_after': int,
                'rows_consolidated': int
            }
        """
        # Get previous month
        today = datetime.now()
        if today.month == 1:
            target_year = today.year - 1
            target_month = 12
        else:
            target_year = today.year
            target_month = today.month - 1

        logger.info(f"🗜️ Compacting {target_year}-{target_month:02d}")

        # Find all model versions
        if not self.base_path.exists():
            logger.warning(f"Predictions path does not exist: {self.base_path}")
            return self._empty_stats()

        version_dirs = list(self.base_path.glob("model_version=*"))
        if not version_dirs:
            logger.warning(f"No model versions found in {self.base_path}")
            return self._empty_stats()

        # Compact each model version
        stats = self._empty_stats()

        for version_dir in version_dirs:
            model_version = version_dir.name.split('=')[1]

            try:
                version_stats = self._compact_month(
                    model_version, target_year, target_month
                )
                self._merge_stats(stats, version_stats)
                stats['model_versions_processed'] += 1

            except Exception as e:
                logger.error(
                    f"❌ Compaction failed for {model_version}: {e}",
                    exc_info=True
                )

        logger.info(
            f"✅ Compaction complete: {stats['months_compacted']} months, "
            f"{stats['files_before']} -> {stats['files_after']} files, "
            f"{stats['rows_consolidated']} rows"
        )

        return stats

    def compact_month(
        self,
        model_version: str,
        year: int,
        month: int
    ) -> Dict[str, int]:
        """
        Compact a specific month for a model version.

        Args:
            model_version: Model version (e.g., "v1.2.3")
            year: Year (e.g., 2025)
            month: Month (1-12)

        Returns:
            Dict with compaction statistics
        """
        logger.info(
            f"🗜️ Compacting {self.layer_name}/{model_version} "
            f"{year}-{month:02d}"
        )

        stats = self._compact_month(model_version, year, month)
        stats['model_versions_processed'] = 1

        return stats

    def _compact_month(
        self,
        model_version: str,
        year: int,
        month: int
    ) -> Dict[str, int]:
        """Compact a single month for a model version."""
        month_path = (
            self.base_path /
            f"model_version={model_version}" /
            f"year={year}" /
            f"month={month:02d}"
        )

        if not month_path.exists():
            logger.debug(f"Month path does not exist: {month_path}")
            return self._empty_stats()

        # Check if already consolidated
        consolidated_path = month_path / "monthly_consolidated.parquet"
        if consolidated_path.exists():
            logger.debug(f"✓ Already consolidated: {consolidated_path}")
            return self._empty_stats()

        # Acquire lock to prevent race conditions
        lock_path = month_path / COMPACTION_LOCK_FILE

        try:
            with CompactionLock(lock_path, timeout=300):
                return self._do_compaction(
                    month_path, consolidated_path, model_version, year, month
                )
        except TimeoutError as e:
            logger.error(f"❌ Lock timeout: {e}")
            return self._empty_stats()

    def _do_compaction(
        self,
        month_path: Path,
        consolidated_path: Path,
        model_version: str,
        year: int,
        month: int
    ) -> Dict[str, int]:
        """Perform the actual compaction (assumes lock is held)."""
        # Check again if consolidated (another process may have done it)
        if consolidated_path.exists():
            logger.debug(f"✓ Already consolidated (after lock): {consolidated_path}")
            return self._empty_stats()

        # Read all daily files
        daily_dfs = []
        daily_files = []

        for day_dir in sorted(month_path.glob("day=*")):
            data_file = day_dir / "data.parquet"

            if data_file.exists():
                try:
                    df = pd.read_parquet(data_file)
                    daily_dfs.append(df)
                    daily_files.append(data_file)
                except Exception as e:
                    logger.error(f"❌ Error reading {data_file}: {e}")
                    # Continue with other files

        if not daily_dfs:
            logger.warning(f"⚠️ No daily files found in {month_path}")
            return self._empty_stats()

        # Combine all daily files
        monthly_df = pd.concat(daily_dfs, axis=0)
        monthly_df = monthly_df.sort_index()

        # Remove duplicates (keep last)
        monthly_df = monthly_df[~monthly_df.index.duplicated(keep='last')]

        logger.info(
            f"📦 Consolidating {len(daily_files)} daily files "
            f"({len(monthly_df)} rows) into {consolidated_path.name}"
        )

        # Write consolidated file (atomic write)
        self._atomic_write(monthly_df, consolidated_path)

        # Build statistics
        stats = {
            'model_versions_processed': 0,
            'months_compacted': 1,
            'files_before': len(daily_files),
            'files_after': 1,
            'rows_consolidated': len(monthly_df)
        }

        # DELETE daily files (save 95% of inodes!)
        if self.delete_daily_files:
            deleted_count = self._delete_daily_files(month_path)
            logger.info(
                f"✅ Compaction complete: {consolidated_path.name} "
                f"(deleted {deleted_count} daily files)"
            )
        else:
            logger.info(
                f"✅ Compaction complete: {consolidated_path.name} "
                f"(daily files preserved)"
            )

        return stats

    def _atomic_write(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Atomic write: write to temp, then rename.

        Args:
            df: DataFrame to write
            filepath: Target file path
        """
        # Create temp file in same directory (for atomic rename)
        temp_path = filepath.with_suffix('.tmp')

        try:
            # Write to temp file
            df.to_parquet(
                temp_path,
                compression=PARQUET_COMPRESSION,
                index=True,
                engine=PARQUET_ENGINE
            )

            # Atomic rename (POSIX guarantees atomicity)
            temp_path.rename(filepath)

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(
                f"Failed to write consolidated file {filepath}: {e}"
            ) from e

    def _delete_daily_files(self, month_path: Path) -> int:
        """
        Delete daily files after consolidation.

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        for day_dir in month_path.glob("day=*"):
            data_file = day_dir / "data.parquet"

            if data_file.exists():
                try:
                    data_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"❌ Error deleting {data_file}: {e}")

            # Remove empty day directory
            try:
                day_dir.rmdir()
            except OSError:
                # Directory not empty (temp files or other content)
                logger.debug(f"Could not remove {day_dir} (not empty)")

        return deleted_count

    def _empty_stats(self) -> Dict[str, int]:
        """Return empty statistics dict."""
        return {
            'model_versions_processed': 0,
            'months_compacted': 0,
            'files_before': 0,
            'files_after': 0,
            'rows_consolidated': 0
        }

    def _merge_stats(self, target: Dict[str, int], source: Dict[str, int]) -> None:
        """Merge statistics dicts (in-place)."""
        for key, value in source.items():
            if key in target:
                target[key] += value
