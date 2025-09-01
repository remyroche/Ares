#!/usr/bin/env python3
"""
Centralized CSV Export System for Monitoring Data

Provides CSV export capabilities for monitoring data.
"""


import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors
from src.utils.centralized_decorators import (
    performance_monitor,
    PerformanceLevel,
    memory_efficient,
)
from src.utils.logger import system_logger


class CSVExporter:
    """Centralized CSV export system for monitoring data."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("CSVExporter")

        # Export configuration
        self.export_config = config.get(
            "csv_exporter",
            {
                "export_directory": "exports/monitoring",
                "auto_export_interval_hours": 24,
                "max_file_size_mb": 100,
                "compression_enabled": True,
                "include_metadata": True,
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
            },
        )

        # Create export directory
        self.export_dir = Path(self.export_config["export_directory"])  # type: ignore[index]
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Export history
        self.export_history: List[Dict[str, Any]] = []

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @memory_efficient()
    @handle_errors(exceptions=(Exception,), default_return=False, context="csv_exporter.initialize")
    @performance_monitor(level=PerformanceLevel.DETAILED)
    @memory_efficient()
    @handle_errors(exceptions=(Exception,), default_return=None, context="csv_exporter.export_performance")
    async def _write_csv_file(
        self,
        filepath: Path,
        rows: List[Dict[str, Any]],
        include_metadata: bool,
    ) -> None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        if include_metadata:
            meta_path = filepath.with_suffix(".json")
            try:
                import json

                meta = {
                    "exported_at": datetime.now().isoformat(),
                    "row_count": len(rows),
                    "source": "CSVExporter",
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                # Non-fatal metadata failure
                pass
