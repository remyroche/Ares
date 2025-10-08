"""Utility for persisting structured metrics for the pre-training pipeline."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Import common utilities for enhanced file operations and error handling
from src.utils.common_operations import ensure_directory, safe_file_exists, format_bytes, get_memory_usage
from src.utils.tprint import tprint, tprint_debug, tprint_warning, tprint_error

try:  # pragma: no cover - optional dependency guard
    from prometheus_client import CollectorRegistry, Gauge
except ImportError:  # pragma: no cover - handled at runtime when enabled
    CollectorRegistry = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]


@dataclass
class MetricsSinkConfig:
    """Configuration for the :class:`MetricsSink` utility."""

    output_path: Path
    output_format: str = "csv"
    enable_prometheus: bool = False
    namespace: str = "pre_training"


class MetricsSink:
    """Persist metrics to CSV/JSONL files and optionally a Prometheus registry."""

    def __init__(self, config: MetricsSinkConfig):
        self.config = config
        self.output_path = config.output_path
        self.output_format = config.output_format.lower()
        self._csv_fieldnames: Optional[Iterable[str]] = None

        tprint(f"📊 Initialized MetricsSink for {self.output_format.upper()} output")
        tprint_debug(f"📁 Output path: {self.output_path}")

        if self.output_format not in {"csv", "jsonl"}:
            tprint_error(f"❌ Unsupported metrics output format: {self.output_format}")
            raise ValueError(f"Unsupported metrics output format: {self.output_format}")

        # Use common utility for directory creation with error handling
        if not ensure_directory(self.output_path.parent):
            tprint_error(f"❌ Failed to create metrics output directory: {self.output_path.parent}")
            raise ValueError(f"Cannot create metrics output directory: {self.output_path.parent}")

        tprint_debug(f"✅ MetricsSink initialized successfully")

        self.registry: Optional[CollectorRegistry] = None
        self._prometheus_metrics: Dict[str, Gauge] = {}
        if config.enable_prometheus:
            if CollectorRegistry is None or Gauge is None:  # pragma: no cover - runtime guard
                raise ImportError("prometheus_client is required when enable_prometheus=True")
            self.registry = CollectorRegistry()

        if self.output_format == "csv" and self.output_path.exists() and self.output_path.stat().st_size > 0:
            with self.output_path.open("r", newline="") as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader, None)
                if header:
                    self._csv_fieldnames = header

    def write(self, record: Dict[str, Any]) -> None:
        """Append a metrics record to the configured sink."""

        tprint_debug(f"📝 Writing metrics record with {len(record)} fields")

        if self.registry is not None:
            self._update_prometheus(record)

        serialized_record = self._serialize_record(record)
        if self.output_format == "csv":
            self._write_csv(serialized_record)
        else:
            self._write_jsonl(serialized_record)

        tprint_debug(f"✅ Metrics record written successfully")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_csv(self, record: Dict[str, Any]) -> None:
        if self._csv_fieldnames is None:
            self._csv_fieldnames = list(record.keys())

        with self.output_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._csv_fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(record)

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        with self.output_path.open("a") as jsonl_file:
            jsonl_file.write(json.dumps(record) + "\n")

    def _serialize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (dict, list, tuple)):
                serialized[key] = json.dumps(value)
            elif value is None:
                serialized[key] = ""
            else:
                serialized[key] = value
        return serialized

    # ------------------------------------------------------------------
    # Prometheus helpers
    # ------------------------------------------------------------------
    def _update_prometheus(self, record: Dict[str, Any]) -> None:
        if self.registry is None or Gauge is None:
            return

        record_label = str(record.get("step_name", "record"))
        for key, value in record.items():
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                metric_name = self._sanitize_metric_name(f"{self.config.namespace}_{key}")
                gauge = self._prometheus_metrics.get(metric_name)
                if gauge is None:
                    gauge = Gauge(metric_name, f"{key} metric", ["record"], registry=self.registry)
                    self._prometheus_metrics[metric_name] = gauge
                gauge.labels(record=record_label).set(float(value))

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"metric_{sanitized}"
        return sanitized.lower()

