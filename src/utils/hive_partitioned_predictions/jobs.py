"""
Scheduled compaction jobs for Hive-partitioned predictions.

Run this as a cron job on the 1st of each month to consolidate
daily files into monthly_consolidated.parquet files.

Example crontab entry:
    # Run at 02:00 AM UTC on the 1st of every month
    0 2 1 * * cd /path/to/Ares && python -m src.utils.hive_partitioned_predictions.jobs

"""
import logging
import sys
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from .compactor import MonthlyCompactor
from .constants import SUPPORTED_LAYERS


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def monthly_compaction_job(
    layers: List[str] = None,
    delete_daily_files: bool = True,
    log_file: str = None
) -> Dict[str, Dict[str, int]]:
    """
    Run compaction for all layers.

    This is the main entry point for the scheduled job.

    Args:
        layers: List of layers to compact (default: all supported layers)
        delete_daily_files: If True, delete daily files after consolidation
        log_file: Optional log file path

    Returns:
        Dict mapping layer names to compaction statistics

    Example:
        >>> stats = monthly_compaction_job()
        >>> print(f"Processed {stats['specialists']['months_compacted']} months")
    """
    logger.info("=" * 80)
    logger.info("🗜️ Starting Monthly Compaction Job")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    if layers is None:
        layers = SUPPORTED_LAYERS

    # Validate layers
    invalid_layers = set(layers) - set(SUPPORTED_LAYERS)
    if invalid_layers:
        logger.error(f"Invalid layers: {invalid_layers}")
        logger.error(f"Supported layers: {SUPPORTED_LAYERS}")
        sys.exit(1)

    results = {}
    overall_stats = {
        'total_model_versions': 0,
        'total_months': 0,
        'total_files_before': 0,
        'total_files_after': 0,
        'total_rows': 0,
        'layers_processed': 0,
        'layers_failed': 0
    }

    for layer in layers:
        logger.info("")
        logger.info(f"Processing layer: {layer}")
        logger.info("-" * 80)

        try:
            compactor = MonthlyCompactor(
                layer,
                delete_daily_files=delete_daily_files
            )
            stats = compactor.compact_previous_month()

            results[layer] = stats

            # Update overall stats
            overall_stats['total_model_versions'] += stats['model_versions_processed']
            overall_stats['total_months'] += stats['months_compacted']
            overall_stats['total_files_before'] += stats['files_before']
            overall_stats['total_files_after'] += stats['files_after']
            overall_stats['total_rows'] += stats['rows_consolidated']
            overall_stats['layers_processed'] += 1

            logger.info(
                f"✅ {layer}: {stats['months_compacted']} months, "
                f"{stats['files_before']} -> {stats['files_after']} files, "
                f"{stats['rows_consolidated']} rows"
            )

        except Exception as e:
            logger.error(f"❌ Compaction failed for {layer}: {e}", exc_info=True)
            results[layer] = None
            overall_stats['layers_failed'] += 1

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 Compaction Summary")
    logger.info("=" * 80)
    logger.info(f"Layers processed: {overall_stats['layers_processed']}")
    logger.info(f"Layers failed: {overall_stats['layers_failed']}")
    logger.info(f"Model versions: {overall_stats['total_model_versions']}")
    logger.info(f"Months compacted: {overall_stats['total_months']}")
    logger.info(
        f"Files: {overall_stats['total_files_before']} -> "
        f"{overall_stats['total_files_after']}"
    )
    logger.info(f"Rows consolidated: {overall_stats['total_rows']}")

    # Calculate space savings
    if overall_stats['total_files_before'] > 0:
        reduction_pct = (
            100 * (overall_stats['total_files_before'] - overall_stats['total_files_after'])
            / overall_stats['total_files_before']
        )
        logger.info(f"File reduction: {reduction_pct:.1f}%")

    logger.info("=" * 80)
    logger.info("✅ Monthly compaction complete!")
    logger.info("=" * 80)

    return results


def compact_specific_month(
    layer: str,
    model_version: str,
    year: int,
    month: int,
    delete_daily_files: bool = True
) -> Dict[str, int]:
    """
    Compact a specific month for a model version.

    Useful for manual compaction or backfilling.

    Args:
        layer: Layer name
        model_version: Model version (e.g., "v1.2.3")
        year: Year (e.g., 2025)
        month: Month (1-12)
        delete_daily_files: If True, delete daily files after consolidation

    Returns:
        Compaction statistics

    Example:
        >>> stats = compact_specific_month(
        ...     "specialists", "v1.2.3", 2025, 10
        ... )
    """
    logger.info(
        f"Compacting specific month: {layer}/{model_version} "
        f"{year}-{month:02d}"
    )

    compactor = MonthlyCompactor(layer, delete_daily_files=delete_daily_files)
    stats = compactor.compact_month(model_version, year, month)

    logger.info(f"✅ Compaction complete: {stats}")

    return stats


def backfill_compaction(
    layer: str,
    model_version: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    delete_daily_files: bool = True
) -> Dict[str, int]:
    """
    Backfill compaction for a date range.

    Useful when migrating to Hive partitioning or when compaction jobs
    were missed.

    Args:
        layer: Layer name
        model_version: Model version (e.g., "v1.2.3")
        start_year: Start year
        start_month: Start month (1-12)
        end_year: End year
        end_month: End month (1-12)
        delete_daily_files: If True, delete daily files after consolidation

    Returns:
        Aggregated compaction statistics

    Example:
        >>> # Backfill compaction for Q4 2024
        >>> stats = backfill_compaction(
        ...     "specialists", "v1.2.3", 2024, 10, 2024, 12
        ... )
    """
    logger.info(
        f"Backfilling compaction: {layer}/{model_version} "
        f"{start_year}-{start_month:02d} to {end_year}-{end_month:02d}"
    )

    compactor = MonthlyCompactor(layer, delete_daily_files=delete_daily_files)

    # Generate list of (year, month) tuples
    months = []
    current_year = start_year
    current_month = start_month

    while (current_year, current_month) <= (end_year, end_month):
        months.append((current_year, current_month))

        # Increment month
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1

    # Compact each month
    total_stats = {
        'model_versions_processed': 0,
        'months_compacted': 0,
        'files_before': 0,
        'files_after': 0,
        'rows_consolidated': 0
    }

    for year, month in months:
        try:
            stats = compactor.compact_month(model_version, year, month)

            # Aggregate statistics
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)

            logger.info(
                f"✅ {year}-{month:02d}: {stats['months_compacted']} months, "
                f"{stats['files_before']} -> {stats['files_after']} files"
            )

        except Exception as e:
            logger.error(
                f"❌ Compaction failed for {year}-{month:02d}: {e}",
                exc_info=True
            )

    logger.info(f"✅ Backfill complete: {total_stats}")

    return total_stats


if __name__ == "__main__":
    """
    CLI entry point for scheduled compaction jobs.

    Usage:
        # Run monthly compaction for all layers
        python -m src.utils.hive_partitioned_predictions.jobs

        # Run with specific layers
        python -m src.utils.hive_partitioned_predictions.jobs specialists base_models

        # Run with debug logging
        LOG_LEVEL=DEBUG python -m src.utils.hive_partitioned_predictions.jobs

        # Write to log file
        python -m src.utils.hive_partitioned_predictions.jobs --log-file compaction.log
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run monthly compaction for Hive-partitioned predictions"
    )
    parser.add_argument(
        "layers",
        nargs="*",
        default=None,
        help="Layers to compact (default: all supported layers)"
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Preserve daily files after consolidation (for testing)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: stdout only)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    # Run compaction
    try:
        results = monthly_compaction_job(
            layers=args.layers if args.layers else None,
            delete_daily_files=not args.no_delete,
            log_file=args.log_file
        )

        # Exit with error if any layers failed
        failed_count = sum(1 for r in results.values() if r is None)
        if failed_count > 0:
            logger.error(f"Compaction failed for {failed_count} layers")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
