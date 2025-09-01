#!/usr/bin/env python3
"""
Centralized CSV Export System for Monitoring Data

Provides CSV export capabilities for monitoring data.
"""

from __future__ import annotations

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
    async def initialize(self) -> bool:
        """Initialize CSV exporter."""
        self.logger.info("📊 Initializing CSV Exporter...")

        # Create subdirectories for different data types
        for data_type in [
            "performance",
            "anomalies",
            "predictions",
            "correlations",
            "risk_metrics",
            "system_health",
            "trade_data",
            "model_metrics",
        ]:
            (self.export_dir / data_type).mkdir(exist_ok=True)

        self.logger.info("✅ CSV Exporter initialized successfully")
        return True

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @memory_efficient()
    @handle_errors(exceptions=(Exception,), default_return=None, context="csv_exporter.export_performance")
    async def export_performance_metrics(
        self,
        data: List[Dict[str, Any]],
        time_range: str = "24h",
        include_metadata: bool = True,
    ) -> Optional[str]:
        """Export performance metrics to CSV."""
        if not data:
            self.logger.warning("No performance data to export")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_metrics_{time_range}_{timestamp}.csv"
        filepath = self.export_dir / "performance" / filename

        await self._write_csv_file(filepath, data, include_metadata)
        self._record_export("performance", filepath, len(data))
        self.logger.info(f"✅ Performance metrics exported to {filepath}")
        return str(filepath)

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

    def _record_export(self, data_type: str, filepath: Path, count: int) -> None:
        self.export_history.append(
            {
                "type": data_type,
                "path": str(filepath),
                "count": count,
                "timestamp": datetime.now().isoformat(),
            }
        )
