#!/usr/bin/env python3
"""
Centralized CSV Export System for Monitoring Data

This module provides comprehensive CSV export capabilities for all monitoring aspects,
enabling personalized computations and external analysis.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
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


class CSVExporter:
    """Centralized CSV export system for monitoring data."""

    def __init__(self, config: dict[str, Any]):
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
        self.export_dir = Path(self.export_config["export_directory"])
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Export history
        self.export_history: list[dict[str, Any]] = []

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="CSV exporter initialization",
    )
    async def initialize(self) -> bool:
        """Initialize CSV exporter."""
        self.logger.info("📊 Initializing CSV Exporter...")

        # Create subdirectories for different data types
        data_types = [
            "performance",
            "anomalies",
            "predictions",
            "correlations",
            "risk_metrics",
            "system_health",
            "trade_data",
            "model_metrics",
        ]

        for data_type in data_types:
            (self.export_dir / data_type).mkdir(exist_ok=True)

        self.logger.info("✅ CSV Exporter initialized successfully")
        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance metrics export",
    )
    async def export_performance_metrics(
        self,
        data: list[dict[str, Any]],
        time_range: str = "24h",
        include_metadata: bool = True,
    ) -> str | None:
        """Export performance metrics to CSV."""
        try:
            if not data:
                self.print(warning("No performance data to export"))
                return None

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "performance" / filename

            # Prepare CSV data
            csv_data = []
            for record in data:
                row = {
                    "timestamp": record.get("timestamp", ""),
                    "model_accuracy": record.get("model_accuracy", 0.0),
                    "model_precision": record.get("model_precision", 0.0),
                    "model_recall": record.get("model_recall", 0.0),
                    "model_f1_score": record.get("model_f1_score", 0.0),
                    "model_auc": record.get("model_auc", 0.0),
                    "trading_win_rate": record.get("trading_win_rate", 0.0),
                    "trading_profit_factor": record.get("trading_profit_factor", 0.0),
                    "trading_sharpe_ratio": record.get("trading_sharpe_ratio", 0.0),
                    "trading_max_drawdown": record.get("trading_max_drawdown", 0.0),
                    "trading_total_return": record.get("trading_total_return", 0.0),
                    "system_memory_usage": record.get("system_memory_usage", 0.0),
                    "system_cpu_usage": record.get("system_cpu_usage", 0.0),
                    "system_response_time": record.get("system_response_time", 0.0),
                    "system_throughput": record.get("system_throughput", 0.0),
                    "confidence_analyst": record.get("confidence_analyst", 0.0),
                    "confidence_tactician": record.get("confidence_tactician", 0.0),
                    "confidence_final": record.get("confidence_final", 0.0),
                }
                csv_data.append(row)

            # Write CSV file
            await self._write_csv_file(filepath, csv_data, include_metadata)

            # Record export
            self._record_export("performance", filepath, len(data))

            self.logger.info(f"✅ Performance metrics exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting performance metrics: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="anomaly detection export",
    )
    async def export_anomaly_data(
        self,
        data: list[dict[str, Any]],
        time_range: str = "7d",
        include_metadata: bool = True,
    ) -> str | None:
        """Export anomaly detection data to CSV."""
        try:
            if not data:
                self.print(warning("No anomaly data to export"))
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anomaly_detection_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "anomalies" / filename

            csv_data = []
            for record in data:
                row = {
                    "timestamp": record.get("timestamp", ""),
                    "metric": record.get("metric", ""),
                    "current_value": record.get("current_value", 0.0),
                    "baseline_value": record.get("baseline_value", 0.0),
                    "deviation": record.get("deviation", 0.0),
                    "severity": record.get("severity", ""),
                    "threshold": record.get("threshold", 0.0),
                    "description": record.get("description", ""),
                    "recommended_action": record.get("recommended_action", ""),
                    "estimated_impact": record.get("estimated_impact", ""),
                }
                csv_data.append(row)

            await self._write_csv_file(filepath, csv_data, include_metadata)
            self._record_export("anomalies", filepath, len(data))

            self.logger.info(f"✅ Anomaly data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting anomaly data: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="predictive analytics export",
    )
    async def export_predictive_analytics(
        self,
        data: list[dict[str, Any]],
        time_range: str = "24h",
        include_metadata: bool = True,
    ) -> str | None:
        """Export predictive analytics data to CSV."""
        try:
            if not data:
                self.print(warning("No predictive analytics data to export"))
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictive_analytics_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "predictions" / filename

            csv_data = []
            for record in data:
                row = {
                    "timestamp": record.get("timestamp", ""),
                    "metric": record.get("metric", ""),
                    "current_value": record.get("current_value", 0.0),
                    "predictions": ",".join(map(str, record.get("predictions", []))),
                    "confidence": record.get("confidence", 0.0),
                    "trend": record.get("trend", ""),
                    "slope": record.get("slope", 0.0),
                    "r_squared": record.get("r_squared", 0.0),
                }
                csv_data.append(row)

            await self._write_csv_file(filepath, csv_data, include_metadata)
            self._record_export("predictions", filepath, len(data))

            self.logger.info(f"✅ Predictive analytics exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting predictive analytics: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="correlation analysis export",
    )
    async def export_correlation_analysis(
        self,
        data: list[dict[str, Any]],
        time_range: str = "7d",
        include_metadata: bool = True,
    ) -> str | None:
        """Export correlation analysis data to CSV."""
        try:
            if not data:
                self.print(warning("No correlation data to export"))
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlation_analysis_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "correlations" / filename

            csv_data = []
            for record in data:
                row = {
                    "timestamp": record.get("timestamp", ""),
                    "metric_1": record.get("metric_1", ""),
                    "metric_2": record.get("metric_2", ""),
                    "correlation_value": record.get("correlation_value", 0.0),
                    "significance": record.get("significance", ""),
                    "p_value": record.get("p_value", 0.0),
                    "sample_size": record.get("sample_size", 0),
                    "confidence_interval_lower": record.get(
                        "confidence_interval_lower",
                        0.0,
                    ),
                    "confidence_interval_upper": record.get(
                        "confidence_interval_upper",
                        0.0,
                    ),
                }
                csv_data.append(row)

            await self._write_csv_file(filepath, csv_data, include_metadata)
            self._record_export("correlations", filepath, len(data))

            self.logger.info(f"✅ Correlation analysis exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting correlation analysis: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk metrics export",
    )
    async def export_risk_metrics(
        self,
        data: list[dict[str, Any]],
        time_range: str = "24h",
        include_metadata: bool = True,
    ) -> str | None:
        """Export risk metrics data to CSV."""
        try:
            if not data:
                self.print(warning("No risk metrics data to export"))
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_metrics_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "risk_metrics" / filename

            csv_data = []
            for record in data:
                row = {
                    "timestamp": record.get("timestamp", ""),
                    "portfolio_var": record.get("portfolio_var", 0.0),
                    "portfolio_correlation": record.get("portfolio_correlation", 0.0),
                    "portfolio_concentration": record.get(
                        "portfolio_concentration",
                        0.0,
                    ),
                    "portfolio_leverage": record.get("portfolio_leverage", 1.0),
                    "position_count": record.get("position_count", 0),
                    "max_position_size": record.get("max_position_size", 0.0),
                    "avg_position_size": record.get("avg_position_size", 0.0),
                    "position_duration": record.get("position_duration", 0.0),
                    "market_volatility": record.get("market_volatility", 0.0),
                    "market_liquidity": record.get("market_liquidity", 0.0),
                    "market_stress": record.get("market_stress", 0.0),
                    "market_regime": record.get("market_regime", "unknown"),
                }
                csv_data.append(row)

            await self._write_csv_file(filepath, csv_data, include_metadata)
            self._record_export("risk_metrics", filepath, len(data))

            self.logger.info(f"✅ Risk metrics exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting risk metrics: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="system health export",
    )
    async def export_system_health(
        self,
        data: list[dict[str, Any]],
        time_range: str = "1h",
        include_metadata: bool = True,
    ) -> str | None:
        """Export system health data to CSV."""
        try:
            if not data:
                self.print(warning("No system health data to export"))
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_health_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "system_health" / filename

            csv_data = []
            for record in data:
                row = {
                    "timestamp": record.get("timestamp", ""),
                    "health_score": record.get("health_score", 0.0),
                    "memory_usage": record.get("memory_usage", 0.0),
                    "cpu_usage": record.get("cpu_usage", 0.0),
                    "response_time": record.get("response_time", 0.0),
                    "throughput": record.get("throughput", 0.0),
                    "error_rate": record.get("error_rate", 0.0),
                    "uptime": record.get("uptime", 0.0),
                    "active_connections": record.get("active_connections", 0),
                }
                csv_data.append(row)

            await self._write_csv_file(filepath, csv_data, include_metadata)
            self._record_export("system_health", filepath, len(data))

            self.logger.info(f"✅ System health exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting system health: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trade data export",
    )
    async def export_trade_data(
        self,
        data: list[dict[str, Any]],
        time_range: str = "30d",
        include_metadata: bool = True,
    ) -> str | None:
        """Export trade data to CSV."""
        try:
            if not data:
                self.print(warning("No trade data to export"))
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_data_{time_range}_{timestamp}.csv"
            filepath = self.export_dir / "trade_data" / filename

            csv_data = []
            for record in data:
                row = {
                    "trade_id": record.get("trade_id", ""),
                    "timestamp": record.get("timestamp", ""),
                    "symbol": record.get("symbol", ""),
                    "side": record.get("side", ""),
                    "quantity": record.get("quantity", 0.0),
                    "price": record.get("price", 0.0),
                    "pnl": record.get("pnl", 0.0),
                    "win_rate": record.get("win_rate", 0.0),
                    "model_confidence": record.get("model_confidence", 0.0),
                    "ensemble_decision": record.get("ensemble_decision", ""),
                    "regime_analysis": record.get("regime_analysis", ""),
                    "risk_metrics": record.get("risk_metrics", ""),
                }
                csv_data.append(row)

            await self._write_csv_file(filepath, csv_data, include_metadata)
            self._record_export("trade_data", filepath, len(data))

            self.logger.info(f"✅ Trade data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.print(error("Error exporting trade data: {e}"))
            return None

    async def _write_csv_file(
        self,
        filepath: Path,
        data: list[dict[str, Any]],
        include_metadata: bool = True,
    ) -> None:
        """Write data to CSV file with optional metadata."""
        try:
            if not data:
                return

            # Get fieldnames from first record
            fieldnames = list(data[0].keys())

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write data rows
                for row in data:
                    writer.writerow(row)

            # Add metadata if requested
            if include_metadata:
                await self._add_metadata(filepath, data)

        except Exception as e:
            self.print(error("Error writing CSV file: {e}"))
            raise

    async def _add_metadata(self, filepath: Path, data: list[dict[str, Any]]) -> None:
        """Add metadata to CSV file."""
        try:
            metadata = {
                "export_timestamp": datetime.now().isoformat(),
                "total_records": len(data),
                "time_range": self._calculate_time_range(data),
                "data_columns": list(data[0].keys()) if data else [],
                "export_config": self.export_config,
            }

            # Create metadata file
            metadata_file = filepath.with_suffix(".json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.print(error("Error adding metadata: {e}"))

    def _calculate_time_range(self, data: list[dict[str, Any]]) -> dict[str, str]:
        """Calculate time range from data."""
        try:
            if not data:
                return {"start": "", "end": ""}

            timestamps = [
                record.get("timestamp", "")
                for record in data
                if record.get("timestamp")
            ]
            if timestamps:
                return {"start": min(timestamps), "end": max(timestamps)}
            return {"start": "", "end": ""}

        except Exception as e:
            self.print(error("Error calculating time range: {e}"))
            return {"start": "", "end": ""}

    def _record_export(self, data_type: str, filepath: Path, record_count: int) -> None:
        """Record export history."""
        try:
            export_record = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "filepath": str(filepath),
                "record_count": record_count,
                "file_size_mb": filepath.stat().st_size / (1024 * 1024),
            }
            self.export_history.append(export_record)

            # Keep only last 100 exports
            if len(self.export_history) > 100:
                self.export_history = self.export_history[-100:]

        except Exception as e:
            self.print(error("Error recording export: {e}"))

    def get_export_history(
        self,
        data_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get export history."""
        try:
            if data_type:
                return [
                    record
                    for record in self.export_history
                    if record["data_type"] == data_type
                ]
            return self.export_history

        except Exception as e:
            self.print(error("Error getting export history: {e}"))
            return []

    def get_export_summary(self) -> dict[str, Any]:
        """Get export summary statistics."""
        try:
            total_exports = len(self.export_history)
            total_size_mb = sum(
                record.get("file_size_mb", 0) for record in self.export_history
            )

            # Group by data type
            type_counts = {}
            for record in self.export_history:
                data_type = record["data_type"]
                type_counts[data_type] = type_counts.get(data_type, 0) + 1

            return {
                "total_exports": total_exports,
                "total_size_mb": round(total_size_mb, 2),
                "type_counts": type_counts,
                "export_directory": str(self.export_dir),
                "last_export": self.export_history[-1] if self.export_history else None,
            }

        except Exception as e:
            self.print(error("Error getting export summary: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="CSV exporter cleanup",
    )
    async def cleanup_old_exports(self, max_age_days: int = 30) -> int:
        """Clean up old export files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0

            for filepath in self.export_dir.rglob("*.csv"):
                if filepath.stat().st_mtime < cutoff_date.timestamp():
                    filepath.unlink()
                    deleted_count += 1

            self.logger.info(f"✅ Cleaned up {deleted_count} old export files")
            return deleted_count

        except Exception as e:
            self.print(error("Error cleaning up old exports: {e}"))
            return 0


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="CSV exporter setup",
)
async def setup_csv_exporter(config: dict[str, Any]) -> CSVExporter | None:
    """Setup CSV exporter."""
    try:
        exporter = CSVExporter(config)
        await exporter.initialize()
        return exporter
    except Exception as e:
        system_print(failed("Failed to setup CSV exporter: {e}"))
        return None
