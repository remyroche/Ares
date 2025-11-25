"""
CLI entry point for Hive-partitioned predictions module.

Usage:
    python -m src.utils.hive_partitioned_predictions [command] [options]

Commands:
    compact     Run monthly compaction job
    backfill    Backfill compaction for a date range
    info        Show information about stored predictions

Examples:
    # Run monthly compaction for all layers
    python -m src.utils.hive_partitioned_predictions compact

    # Compact specific layers
    python -m src.utils.hive_partitioned_predictions compact specialists base_models

    # Backfill compaction
    python -m src.utils.hive_partitioned_predictions backfill specialists v1.2.3 2024-10 2024-12

    # Show info
    python -m src.utils.hive_partitioned_predictions info specialists
"""

if __name__ == "__main__":
    from .jobs import monthly_compaction_job, setup_logging
    import sys

    # Setup default logging
    setup_logging(log_level="INFO")

    # Delegate to jobs module CLI
    from .jobs import __main__
