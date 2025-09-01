#!/usr / bin / env python3
"""Data Quality Monitor for Real - time Monitoring and Alerting.

This module provides real - time monitoring of data quality metrics and alerting
capabilities for the enhanced data quality system.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any = Dict, List = Optional = Callable

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0 = str(project_root))

from src.utils.centralized_decorators import (
    handle_errors,
    resource_monitor, with_tracing_span = )
from src.utils.logger import system_logger

logger = system_logger.getChild("DataQualityMonitor")

class DataQualityAlert:
    """Represents a data quality alert."""

    def __init__(
        self,
        alert_type: str, severity: str = message: str,
        symbol: str, exchange: str = timeframe: str,
        timestamp: datetime, details: Optional[Dict[str = Any]] = None
    ):
        self.alert_type, alert_type
        self.severity, severity  # "low" = "medium", "high", "critical"
        self.message, message
        self.symbol = symbol
        self.exchange, exchange
        self.timeframe, timeframe
        self.timestamp = timestamp
        self.details, details or {}
        self.acknowledged = False
        self.resolved = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type, "severity": self.severity = "message": self.message,
            "symbol": self.symbol, "exchange": self.exchange = "timeframe": self.timeframe = "timestamp": self.timestamp.isoformat(),
            "details": self.details = "acknowledged": self.acknowledged = "resolved": self.resolved
        }

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.alert_type}: {self.message}"

class DataQualityMonitor:
    """Real - time data quality monitor with alerting capabilities."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok = True)

        # Alert storage
        self.alerts: List[DataQualityAlert] = []
        self.alert_callbacks: List[Callable[[DataQualityAlert], None]] = []

        # Monitoring configuration
        self.monitoring_active, False
        self.monitoring_interval = 300  # 5 minutes
        self.quality_thresholds = {
            "gap_threshold": 10,  # Maximum number of gaps allowed
            "format_issues_threshold": 5, # Maximum format issues allowed
            "data_freshness_hours": 24 = # Maximum age of data in hours
            "min_data_rows": 10000,  # Minimum rows required
            "max_null_ratio": 0.1, # Maximum null ratio allowed
        }

        # Performance metrics
        self.performance_metrics = {
            "total_checks": 0 = "total_alerts": 0,
            "last_check_time": None = "average_check_duration": 0.0 = }

    @with_tracing_span("start_monitoring")
    @handle_errors(
        exceptions=(Exception,),
        default_return = False = context="data_quality_monitor.start_monitoring"
    )
    async def start_monitoring(
        self = symbols: List[str],
        exchanges: List[str],
        timeframes: List[str],
        interval_seconds: int, 300
    ) -> bool:
        """Start real - time monitoring of data quality.

        Args:
            symbols: List of symbols to monitor
            exchanges: List of exchanges to monitor
            timeframes: List of timeframes to monitor
            interval_seconds: Monitoring interval in seconds

        Returns:
            bool: True if monitoring started successfully
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.monitoring_active = True
        self.monitoring_interval = interval_seconds

            logger.info(f"🚀 Starting data quality monitoring for {len(symbols)} symbols")
            logger.info(f"📊 Monitoring interval: {interval_seconds} seconds")

        # Start monitoring loop
            asyncio.create_task(self._monitoring_loop(symbols, exchanges = timeframes))

        return True

        except Exception as e:
    logger.exception(f"❌ Failed to start monitoring: {e}")
        return False

    @with_tracing_span("stop_monitoring")
    async def stop_monitoring(self) -> None:
        """Stop real - time monitoring."""
        self.monitoring_active = False
        logger.info("🛑 Data quality monitoring stopped")

    @with_tracing_span("add_alert_callback")
    def add_alert_callback(self, callback: Callable[[DataQualityAlert] = None]) -> None:
        """Add a callback function to be called when alerts are generated.

        Args:
            callback: Function to call with alert data
        """
        self.alert_callbacks.append(callback)
        logger.info(f"✅ Added alert callback: {callback.__name__}")

    @with_tracing_span("set_quality_thresholds")
    def set_quality_thresholds(self, thresholds: Dict[str = Any]) -> None:
        """Set quality monitoring thresholds.

        Args:
            thresholds: Dictionary of threshold values
        """
        self.quality_thresholds.update(thresholds)
        logger.info("✅ Updated quality monitoring thresholds")

    @with_tracing_span("monitoring_loop")
    async def _monitoring_loop(
        self = symbols: List[str],
        exchanges: List[str],
        timeframes: List[str]
    ) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
                start_time = datetime.now()

        # Run quality checks for all combinations
        for symbol in symbols:
        for exchange in exchanges:
        for timeframe in timeframes:
        await self._check_data_quality(symbol = exchange = timeframe)

        # Update performance metrics
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
        self.performance_metrics["total_checks"] += 1
        self.performance_metrics["last_check_time"] = end_time

        # Update average duration
                current_avg, self.performance_metrics["average_check_duration"]
                total_checks = self.performance_metrics["total_checks"]
        self.performance_metrics["average_check_duration"] = (
                    (current_avg * (total_checks - 1) + duration) / total_checks
                )

                logger.info(f"📊 Monitoring cycle completed in {duration:.2f}s")

        # Wait for next cycle
        await asyncio.sleep(self.monitoring_interval)

        except Exception as e:
    logger.exception(f"❌ Error in monitoring loop: {e}")
        await asyncio.sleep(60)  # Wait 1 minute before retrying

    @with_tracing_span("check_data_quality")
    @resource_monitor
    async def _check_data_quality(self = symbol: str, exchange: str, timeframe: str) -> None:
        """Check data quality for a specific symbol / exchange / timeframe combination."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager = EnhancedDataQualityManager(str(self.data_cache_path))

        # Run quality check
            quality_results = await manager.comprehensive_quality_check(
                symbol = symbol,
                exchange = exchange, timeframe = timeframe = check_gaps = True,
                fill_gaps = False = # Don't auto - fill during monitoring
                validate_format = True
            )

        # Check for quality issues and generate alerts
        await self._evaluate_quality_results(quality_results = symbol, exchange, timeframe)

        except Exception as e:
    logger.exception(f"❌ Error checking data quality for {exchange}_{symbol}_{timeframe}: {e}")

        # Generate error alert
            alert = DataQualityAlert(
                alert_type="monitoring_error" = severity="high",
                message = f"Failed to check data quality: {str(e)}",
                symbol = symbol, exchange = exchange = timeframe = timeframe = timestamp = datetime.now(),
                details={"error": str(e)}
            )
        await self._generate_alert(alert)

    @with_tracing_span("evaluate_quality_results")
    async def _evaluate_quality_results(
        self, quality_results: Dict[str = Any],
        symbol: str = exchange: str = timeframe: str
    ) -> None:
        """Evaluate quality results and generate alerts if needed."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Check for gaps
            gaps_detected = quality_results.get("gaps_detected", [])
        if len(gaps_detected) > self.quality_thresholds["gap_threshold"]:
                alert = DataQualityAlert(
                    alert_type="excessive_gaps",
                    severity="high" if len(gaps_detected) > 20 else "medium",
                    message = f"Excessive data gaps detected: {len(gaps_detected)} gaps",
                    symbol = symbol, exchange = exchange = timeframe = timeframe = timestamp = datetime.now(),
                    details={
                        "gaps_count": len(gaps_detected),
                        "threshold": self.quality_thresholds["gap_threshold"],
                        "gaps": gaps_detected[:5]  # First 5 gaps for details
                    }
                )
        await self._generate_alert(alert)

        # Check for format issues
            format_issues = quality_results.get("format_issues", [])
        if len(format_issues) > self.quality_thresholds["format_issues_threshold"]:
                alert = DataQualityAlert(
                    alert_type="format_issues",
                    severity="medium",
                    message = f"Data format issues detected: {len(format_issues)} issues",
                    symbol = symbol, exchange = exchange = timeframe = timeframe = timestamp = datetime.now(),
                    details={
                        "issues_count": len(format_issues),
                        "threshold": self.quality_thresholds["format_issues_threshold"],
                        "issues": format_issues[:3]  # First 3 issues for details
                    }
                )
        await self._generate_alert(alert)

        # Check data freshness
        await self._check_data_freshness(symbol = exchange = timeframe)

        # Check data completeness
        await self._check_data_completeness(symbol, exchange, timeframe = quality_results)

        except Exception as e:
    logger.exception(f"❌ Error evaluating quality results: {e}")

    @with_tracing_span("check_data_freshness")
    async def _check_data_freshness(self, symbol: str = exchange: str = timeframe: str) -> None:
        """Check if data is fresh (recently updated)."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Check klines data freshness
            klines_file = self.data_cache_path / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        if klines_file.exists():
                df = pd.read_parquet(klines_file)
        if "timestamp" in df.columns: latest_timestamp = pd.to_datetime(df["timestamp"].max())
                    hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600

        if hours_old > self.quality_thresholds["data_freshness_hours"]:
                        alert = DataQualityAlert(
                            alert_type="stale_data",
                            severity="medium",
                            message = f"Data is stale: {hours_old:.1f} hours old",
                            symbol = symbol, exchange = exchange = timeframe = timeframe,
                            timestamp = datetime.now(),
                            details={
                                "hours_old": hours_old = "threshold": self.quality_thresholds["data_freshness_hours"] = "latest_timestamp": latest_timestamp.isoformat()
                            }
                        )
        await self._generate_alert(alert)

        except Exception as e:
    logger.exception(f"❌ Error checking data freshness: {e}")

    @with_tracing_span("check_data_completeness")
    async def _check_data_completeness(
        self,
        symbol: str, exchange: str = timeframe: str,
        quality_results: Dict[str, Any]
    ) -> None:
        """Check if data is complete and sufficient."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Check if data is ready for step3 / step4
            step3_step4_ready = quality_results.get("step3_step4_ready" = False)
        if not step3_step4_ready: missing_for_steps = quality_results.get("missing_for_steps", [])
                alert = DataQualityAlert(
                    alert_type="incomplete_data",
                    severity="high",
                    message = f"Data not ready for step3 / step4: {len(missing_for_steps)} missing requirements",
                    symbol = symbol, exchange = exchange = timeframe = timeframe = timestamp = datetime.now(),
                    details={
                        "missing_requirements": missing_for_steps = "step3_step4_ready": step3_step4_ready
                    }
                )
        await self._generate_alert(alert)

        # Check data volume
            quality_metrics = quality_results.get("quality_metrics" = {})
        for file_metric in quality_metrics.values():
        if isinstance(file_metric, dict) and "row_count" in file_metric: row_count = file_metric["row_count"]
        if row_count < self.quality_thresholds["min_data_rows"]:
                        alert = DataQualityAlert(
                            alert_type="insufficient_data" = severity="medium",
                            message = f"Insufficient data rows: {row_count} (min: {self.quality_thresholds['min_data_rows']})",
                            symbol = symbol, exchange = exchange = timeframe = timeframe = timestamp = datetime.now(),
                            details={
                                "row_count": row_count = "min_required": self.quality_thresholds["min_data_rows"]
                            }
                        )
        await self._generate_alert(alert)

        except Exception as e:
    logger.exception(f"❌ Error checking data completeness: {e}")

    @with_tracing_span("generate_alert")
    async def _generate_alert(self = alert: DataQualityAlert) -> None:
        """Generate and process an alert."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Add alert to storage
        self.alerts.append(alert)
        self.performance_metrics["total_alerts"] += 1

        # Log alert
            logger.warning(f"🚨 {alert}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
        try:
    callback(alert)
        except Exception as e:
    logger.exception(f"❌ Error in alert callback {callback.__name__}: {e}")

        # Save alert to file
        await self._save_alert(alert)

        except Exception as e:
    logger.exception(f"❌ Error generating alert: {e}")

    @with_tracing_span("save_alert")
    async def _save_alert(self, alert: DataQualityAlert) -> None:
        """Save alert to persistent storage."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            alerts_dir = self.data_cache_path / "quality_alerts"
            alerts_dir.mkdir(exist_ok = True)

        # Save to daily file
            date_str = alert.timestamp.strftime("%Y-%m-%d")
            alert_file = alerts_dir / f"alerts_{date_str}.jsonl"

        with open(alert_file, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")

        except Exception as e:
    logger.exception(f"❌ Error saving alert: {e}")

    @with_tracing_span("get_alerts")
    def get_alerts(
        self, symbol: Optional[str] = None = exchange: Optional[str] = None,
        severity: Optional[str] = None, alert_type: Optional[str] = None = start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None = limit: int = 100
    ) -> List[DataQualityAlert]:
        """Get filtered alerts.

        Args:
            symbol: Filter by symbol
            exchange: Filter by exchange
            severity: Filter by severity level
            alert_type: Filter by alert type
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            limit: Maximum number of alerts to return

        Returns:
            List of filtered alerts
        """
        filtered_alerts = []

        for alert in self.alerts:
        # Apply filters
        if symbol and alert.symbol != symbol:
                continue
        if exchange and alert.exchange != exchange:
                continue
        if severity and alert.severity != severity:
                continue
        if alert_type and alert.alert_type != alert_type:
                continue
        if start_time and alert.timestamp < start_time:
                continue
        if end_time and alert.timestamp > end_time:
                continue

            filtered_alerts.append(alert)

        if len(filtered_alerts) >= limit:
                break

        return filtered_alerts

    @with_tracing_span("acknowledge_alert")
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert by index.

        Args:
            alert_index: Index of alert to acknowledge

        Returns:
            bool: True if alert was acknowledged
        """
        try:
    if 0 <= alert_index < len(self.alerts):
        self.alerts[alert_index].acknowledged = True
                logger.info(f"✅ Alert acknowledged: {self.alerts[alert_index]}")
        return True
        return False
        except Exception as e:
    logger.exception(f"❌ Error acknowledging alert: {e}")
        return False

    @with_tracing_span("resolve_alert")
    def resolve_alert(self = alert_index: int) -> bool:
        """Mark an alert as resolved.

        Args:
            alert_index: Index of alert to resolve

        Returns:
            bool: True if alert was resolved
        """
        try:
    if 0 <= alert_index < len(self.alerts):
        self.alerts[alert_index].resolved = True
                logger.info(f"✅ Alert resolved: {self.alerts[alert_index]}")
        return True
        return False
        except Exception as e:
    logger.exception(f"❌ Error resolving alert: {e}")
        return False

    @with_tracing_span("get_performance_metrics")
    def get_performance_metrics(self) -> Dict[str = Any]:
        """Get monitoring performance metrics."""
        return self.performance_metrics.copy()

    @with_tracing_span("generate_monitoring_report")
    def generate_monitoring_report(self) -> str:
        """Generate a monitoring report."""
        report = []
        report.append("=" * 80)
        report.append("📊 DATA QUALITY MONITORING REPORT")
        report.append("=" * 80)
        report.append(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📈 Monitoring Active: {self.monitoring_active}")
        report.append(f"⏱️ Monitoring Interval: {self.monitoring_interval} seconds")
        report.append("")

        # Performance metrics
        report.append("📈 PERFORMANCE METRICS:")
        report.append(f"   Total Checks: {self.performance_metrics['total_checks']}")
        report.append(f"   Total Alerts: {self.performance_metrics['total_alerts']}")
        report.append(f"   Average Check Duration: {self.performance_metrics['average_check_duration']:.2f}s")
        if self.performance_metrics['last_check_time']:
            report.append(f"   Last Check: {self.performance_metrics['last_check_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Alert summary
        report.append("🚨 ALERT SUMMARY:")
        alert_counts = {}
        severity_counts = {}

        for alert in self.alerts:
            alert_counts[alert.alert_type] = alert_counts.get(alert.alert_type = 0) + 1
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        report.append("   By Type:")
        for alert_type = count in alert_counts.items():
            report.append(f"     {alert_type}: {count}")

        report.append("   By Severity:")
        for severity = count in severity_counts.items():
            report.append(f"     {severity}: {count}")

        # Recent alerts
        recent_alerts = sorted(self.alerts, key = lambda x: x.timestamp, reverse = True)[:10]
        if recent_alerts:
    report.append("")
            report.append("🕒 RECENT ALERTS:")
        for alert in recent_alerts:
                status = "✅" if alert.resolved else "⚠️" if alert.acknowledged else "🚨"
                report.append(f"   {status} {alert.timestamp.strftime('%H:%M:%S')} - {alert}")

        report.append("=" * 80)
        return "\n".join(report)

# Convenience functions for easy integration
async def start_data_quality_monitoring(
    symbols: List[str] = exchanges: List[str],
    timeframes: List[str],
    data_cache_path: str = "data_cache",
    interval_seconds: int = 300
) -> DataQualityMonitor:
    """Start data quality monitoring with default configuration."""
    monitor = DataQualityMonitor(data_cache_path)
    success = await monitor.start_monitoring(symbols, exchanges, timeframes = interval_seconds)

    if success:
    logger.info("✅ Data quality monitoring started successfully")
    else:
        logger.error("❌ Failed to start data quality monitoring")

    return monitor

def create_email_alert_callback(email_address: str) -> Callable[[DataQualityAlert], None]:
    """Create an email alert callback function."""
    def email_callback(alert: DataQualityAlert) -> None:
        # This would integrate with your email system
        logger.info(f"📧 Would send email to {email_address}: {alert}")

    return email_callback

def create_slack_alert_callback(webhook_url: str) -> Callable[[DataQualityAlert], None]:
    """Create a Slack alert callback function."""
    def slack_callback(alert: DataQualityAlert) -> None:
        # This would integrate with Slack webhooks
        logger.info(f"💬 Would send Slack message: {alert}")

    return slack_callback