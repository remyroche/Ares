#!/usr/bin/env python3
"""
Error Detection and Alerting System

This module provides comprehensive error detection, anomaly detection,
and alerting capabilities for the trading system to identify issues
before they impact performance.
"""

import asyncio
import json
import smtplib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ErrorCategory(Enum):
    """Error categories for classification."""

    SYSTEM = "system"
    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    TRADING = "trading"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONFIGURATION = "configuration"


class AnomalyType(Enum):
    """Types of anomalies to detect."""

    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    VOLUME_SPIKE = "volume_spike"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_SPIKE = "error_rate_spike"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    NETWORK_ISSUES = "network_issues"
    DATA_QUALITY = "data_quality"
    FEATURE_DRIFT = "feature_drift"


@dataclass
class ErrorEvent:
    """Individual error event record."""

    error_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: ErrorCategory

    # Error details
    error_message: str
    error_code: str | None = None
    component: str = ""
    function: str = ""

    # Context
    stack_trace: str | None = None
    user_context: dict[str, Any] = field(default_factory=dict)
    system_state: dict[str, Any] = field(default_factory=dict)

    # Resolution
    is_resolved: bool = False
    resolution_time: datetime | None = None
    resolution_notes: str = ""

    # Impact assessment
    impact_score: float = 0.0  # 0.0 to 1.0
    affected_components: list[str] = field(default_factory=list)
    business_impact: str = ""


@dataclass
class AnomalyDetection:
    """Anomaly detection record."""

    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AlertSeverity

    # Detection details
    metric_name: str
    current_value: float
    expected_value: float
    threshold: float
    deviation_score: float  # How many standard deviations

    # Historical context
    historical_mean: float
    historical_std: float
    percentile_rank: float

    # Time series analysis
    trend_direction: str  # "increasing", "decreasing", "stable"
    seasonality_factor: float | None = None
    autocorrelation: float | None = None

    # Detection algorithm
    detection_method: str = ""  # "statistical", "ml", "rule_based"
    confidence: float = 0.0

    # Context
    related_metrics: dict[str, float] = field(default_factory=dict)
    potential_causes: list[str] = field(default_factory=list)

    # Action taken
    auto_resolved: bool = False
    manual_review_required: bool = True
    escalated: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration."""

    rule_id: str
    rule_name: str
    category: ErrorCategory
    is_enabled: bool = True

    # Trigger conditions
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals", "anomaly"
    threshold: float | None = None
    evaluation_window_minutes: int = 5

    # Advanced conditions
    consecutive_violations: int = 1
    percentage_threshold: float | None = None
    rate_of_change_threshold: float | None = None

    # Alert behavior
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_minutes: int = 30
    max_alerts_per_hour: int = 5

    # Notification
    notify_email: bool = False
    notify_slack: bool = False
    notify_webhook: bool = False
    email_recipients: list[str] = field(default_factory=list)

    # Auto-resolution
    auto_resolve: bool = False
    auto_resolve_condition: str = ""
    auto_resolve_threshold: float | None = None


@dataclass
class SystemHealthMetrics:
    """System health metrics snapshot."""

    timestamp: datetime

    # Performance metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_latency_ms: float

    # Application metrics
    active_connections: int
    request_rate_per_second: float
    error_rate_percent: float
    response_time_ms: float

    # Trading metrics
    prediction_accuracy: float | None = None
    trade_execution_time_ms: float | None = None
    order_success_rate: float | None = None
    pnl_performance: float | None = None

    # Data quality metrics
    data_freshness_minutes: float | None = None
    missing_data_percent: float | None = None
    data_validation_errors: int = 0

    # Model metrics
    model_confidence: float | None = None
    feature_drift_score: float | None = None
    prediction_variance: float | None = None


class ErrorDetectionSystem:
    """
    Comprehensive error detection and alerting system with anomaly detection,
    real-time monitoring, and intelligent alerting capabilities.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize error detection system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("ErrorDetectionSystem")

        # Configuration
        self.detection_config = config.get("error_detection", {})
        self.enable_anomaly_detection = self.detection_config.get(
            "enable_anomaly_detection",
            True,
        )
        self.enable_predictive_alerts = self.detection_config.get(
            "enable_predictive_alerts",
            True,
        )
        self.enable_email_alerts = self.detection_config.get(
            "enable_email_alerts",
            False,
        )
        self.enable_slack_alerts = self.detection_config.get(
            "enable_slack_alerts",
            False,
        )

        # Detection parameters
        self.monitoring_interval = self.detection_config.get(
            "monitoring_interval",
            30,
        )  # seconds
        self.anomaly_sensitivity = self.detection_config.get(
            "anomaly_sensitivity",
            0.95,
        )  # percentile
        self.min_data_points = self.detection_config.get("min_data_points", 100)
        self.lookback_window_hours = self.detection_config.get(
            "lookback_window_hours",
            24,
        )

        # Storage
        self.error_events: dict[str, ErrorEvent] = {}
        self.anomaly_detections: dict[str, AnomalyDetection] = {}
        self.alert_rules: dict[str, AlertRule] = {}
        self.health_metrics_history: list[SystemHealthMetrics] = []

        # Alert state
        self.active_alerts: dict[str, datetime] = {}  # rule_id -> last_alert_time
        self.alert_counts: dict[str, int] = {}  # rule_id -> count_in_current_hour

        # Anomaly detection state
        self.metric_baselines: dict[
            str,
            dict[str, float],
        ] = {}  # metric -> {mean, std, percentiles}
        self.time_series_data: dict[str, list[tuple[datetime, float]]] = {}

        # Statistics
        self.detection_stats = {
            "errors_detected": 0,
            "anomalies_detected": 0,
            "alerts_sent": 0,
            "false_positives": 0,
            "auto_resolved": 0,
            "last_update": datetime.now(),
        }

        self.is_initialized = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid error detection configuration"),
            AttributeError: (False, "Missing required detection parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="error detection initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the error detection system.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Error Detection System...")

            # Initialize storage backend
            await self._initialize_storage()

            # Load configuration and rules
            await self._load_alert_rules()

            # Initialize anomaly detection
            if self.enable_anomaly_detection:
                await self._initialize_anomaly_detection()

            # Initialize notification systems
            await self._initialize_notifications()

            # Start background monitoring
            await self._start_background_monitoring()

            self.is_initialized = True
            self.logger.info("✅ Error Detection System initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error Detection System initialization failed: {e}",
            )
            return False

    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        try:
            storage_backend = self.config.get("monitoring", {}).get(
                "storage_backend",
                "sqlite",
            )

            if storage_backend == "sqlite":
                from src.database.sqlite_manager import SQLiteManager

                self.storage_manager = SQLiteManager(self.config)
                await self.storage_manager.initialize()
                await self._create_error_detection_tables()

            self.logger.info(
                f"Error detection storage backend '{storage_backend}' initialized",
            )

        except Exception:
            self.logger.exception(failed("Failed to initialize storage: {e}"))
            raise

    async def _create_error_detection_tables(self) -> None:
        """Create database tables for error detection."""
        try:
            tables = [
                """
                CREATE TABLE IF NOT EXISTS error_events (
                    error_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    severity TEXT,
                    category TEXT,
                    error_message TEXT,
                    error_code TEXT,
                    component TEXT,
                    function TEXT,
                    is_resolved BOOLEAN,
                    impact_score REAL,
                    error_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    anomaly_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    anomaly_type TEXT,
                    severity TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    expected_value REAL,
                    deviation_score REAL,
                    detection_method TEXT,
                    confidence REAL,
                    auto_resolved BOOLEAN,
                    anomaly_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_name TEXT,
                    category TEXT,
                    is_enabled BOOLEAN,
                    metric_name TEXT,
                    condition TEXT,
                    threshold REAL,
                    severity TEXT,
                    rule_config TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS system_health_metrics (
                    timestamp DATETIME PRIMARY KEY,
                    cpu_usage_percent REAL,
                    memory_usage_percent REAL,
                    disk_usage_percent REAL,
                    network_latency_ms REAL,
                    active_connections INTEGER,
                    request_rate_per_second REAL,
                    error_rate_percent REAL,
                    response_time_ms REAL,
                    health_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS alert_history (
                    alert_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    rule_id TEXT,
                    severity TEXT,
                    message TEXT,
                    recipient TEXT,
                    delivery_status TEXT,
                    alert_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ]

            for table_sql in tables:
                await self.storage_manager.execute_query(table_sql)

            self.logger.info("Error detection tables created successfully")

        except Exception:
            self.logger.exception(failed("Failed to create error detection tables: {e}"))
            raise

    async def _load_alert_rules(self) -> None:
        """Load alert rules from configuration and database."""
        try:
            # Load default rules from configuration
            default_rules = self.detection_config.get("default_rules", {})

            for rule_name, rule_config in default_rules.items():
                rule = AlertRule(
                    rule_id=rule_name,
                    rule_name=rule_config.get("name", rule_name),
                    category=ErrorCategory(rule_config.get("category", "system")),
                    metric_name=rule_config.get("metric_name", ""),
                    condition=rule_config.get("condition", "greater_than"),
                    threshold=rule_config.get("threshold"),
                    severity=AlertSeverity(rule_config.get("severity", "warning")),
                    evaluation_window_minutes=rule_config.get(
                        "evaluation_window_minutes",
                        5,
                    ),
                    cooldown_minutes=rule_config.get("cooldown_minutes", 30),
                    notify_email=rule_config.get("notify_email", False),
                    email_recipients=rule_config.get("email_recipients", []),
                )

                self.alert_rules[rule_name] = rule

            # Add some default critical rules
            await self._add_default_critical_rules()

            self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")

        except Exception:
            self.logger.exception(failed("Failed to load alert rules: {e}"))
            raise

    async def _add_default_critical_rules(self) -> None:
        """Add default critical monitoring rules."""
        try:
            critical_rules = [
                {
                    "rule_id": "high_error_rate",
                    "rule_name": "High Error Rate",
                    "category": ErrorCategory.SYSTEM,
                    "metric_name": "error_rate_percent",
                    "condition": "greater_than",
                    "threshold": 5.0,  # 5% error rate
                    "severity": AlertSeverity.CRITICAL,
                    "evaluation_window_minutes": 5,
                    "consecutive_violations": 2,
                },
                {
                    "rule_id": "memory_usage_high",
                    "rule_name": "High Memory Usage",
                    "category": ErrorCategory.PERFORMANCE,
                    "metric_name": "memory_usage_percent",
                    "condition": "greater_than",
                    "threshold": 90.0,
                    "severity": AlertSeverity.ERROR,
                    "evaluation_window_minutes": 10,
                },
                {
                    "rule_id": "cpu_usage_critical",
                    "rule_name": "Critical CPU Usage",
                    "category": ErrorCategory.PERFORMANCE,
                    "metric_name": "cpu_usage_percent",
                    "condition": "greater_than",
                    "threshold": 95.0,
                    "severity": AlertSeverity.CRITICAL,
                    "evaluation_window_minutes": 3,
                    "consecutive_violations": 3,
                },
                {
                    "rule_id": "prediction_accuracy_low",
                    "rule_name": "Low Prediction Accuracy",
                    "category": ErrorCategory.MODEL,
                    "metric_name": "prediction_accuracy",
                    "condition": "less_than",
                    "threshold": 0.4,  # 40% accuracy
                    "severity": AlertSeverity.ERROR,
                    "evaluation_window_minutes": 30,
                },
                {
                    "rule_id": "network_latency_high",
                    "rule_name": "High Network Latency",
                    "category": ErrorCategory.NETWORK,
                    "metric_name": "network_latency_ms",
                    "condition": "greater_than",
                    "threshold": 1000.0,  # 1 second
                    "severity": AlertSeverity.WARNING,
                    "evaluation_window_minutes": 5,
                },
                {
                    "rule_id": "data_freshness_stale",
                    "rule_name": "Stale Data",
                    "category": ErrorCategory.DATA,
                    "metric_name": "data_freshness_minutes",
                    "condition": "greater_than",
                    "threshold": 30.0,  # 30 minutes
                    "severity": AlertSeverity.ERROR,
                    "evaluation_window_minutes": 5,
                },
            ]

            for rule_data in critical_rules:
                rule = AlertRule(**rule_data)
                self.alert_rules[rule.rule_id] = rule

        except Exception:
            self.logger.exception(failed("Failed to add default critical rules: {e}"))

    async def _initialize_anomaly_detection(self) -> None:
        """Initialize anomaly detection algorithms."""
        try:
            # Load historical data to establish baselines
            await self._load_historical_metrics()

            # Calculate baselines for key metrics
            await self._calculate_metric_baselines()

            self.logger.info("Anomaly detection initialized")

        except Exception:
            self.logger.exception(failed("Failed to initialize anomaly detection: {e}"))
            raise

    async def _load_historical_metrics(self) -> None:
        """Load historical metrics for baseline calculation."""
        try:
            if hasattr(self, "storage_manager"):
                cutoff_date = datetime.now() - timedelta(
                    hours=self.lookback_window_hours,
                )

                query = """
                SELECT timestamp, cpu_usage_percent, memory_usage_percent,
                       error_rate_percent, response_time_ms
                FROM system_health_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp
                """

                results = await self.storage_manager.execute_query(
                    query,
                    (cutoff_date,),
                )

                # Process historical data
                for _row in results:
                    # This would parse the row data and populate time_series_data
                    # For now, we'll skip this detailed implementation
                    pass

                self.logger.info(f"Loaded historical metrics from {cutoff_date}")

        except Exception:
            self.logger.exception(failed("Failed to load historical metrics: {e}"))
            # Non-critical error, continue

    async def _calculate_metric_baselines(self) -> None:
        """Calculate statistical baselines for metrics."""
        try:
            # Calculate baselines for each metric in time series data
            for metric_name, data_points in self.time_series_data.items():
                if len(data_points) >= self.min_data_points:
                    values = [point[1] for point in data_points]

                    baseline = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "p50": float(np.percentile(values, 50)),
                        "p95": float(np.percentile(values, 95)),
                        "p99": float(np.percentile(values, 99)),
                    }

                    self.metric_baselines[metric_name] = baseline

            self.logger.info(
                f"Calculated baselines for {len(self.metric_baselines)} metrics",
            )

        except Exception:
            self.logger.exception(failed("Failed to calculate metric baselines: {e}"))

    async def _initialize_notifications(self) -> None:
        """Initialize notification systems."""
        try:
            # Email configuration
            if self.enable_email_alerts:
                self.email_config = self.config.get("email", {})
                self.smtp_server = self.email_config.get("smtp_server", "")
                self.smtp_port = self.email_config.get("smtp_port", 587)
                self.email_user = self.email_config.get("username", "")
                self.email_password = self.email_config.get("password", "")

                if not all([self.smtp_server, self.email_user, self.email_password]):
                    self.enable_email_alerts = False
                    self.logger.warning(
                        "Email alerts disabled due to missing configuration",
                    )

            # Slack configuration
            if self.enable_slack_alerts:
                self.slack_config = self.config.get("slack", {})
                self.slack_webhook = self.slack_config.get("webhook_url", "")

                if not self.slack_webhook:
                    self.enable_slack_alerts = False
                    self.logger.warning(
                        "Slack alerts disabled due to missing webhook URL",
                    )

            self.logger.info("Notification systems initialized")

        except Exception:
            self.logger.exception(failed("Failed to initialize notifications: {e}"))

    async def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Start system health monitoring
            asyncio.create_task(self._monitor_system_health())

            # Start anomaly detection
            if self.enable_anomaly_detection:
                asyncio.create_task(self._monitor_anomalies())

            # Start alert rule evaluation
            asyncio.create_task(self._evaluate_alert_rules())

            self.logger.info("Background monitoring tasks started")

        except Exception:
            self.logger.exception(failed("Failed to start background monitoring: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="error event recording",
    )
    async def record_error_event(
        self,
        severity: AlertSeverity,
        category: ErrorCategory,
        error_message: str,
        component: str = "",
        function: str = "",
        error_code: str = None,
        **kwargs,
    ) -> str:
        """
        Record an error event.

        Args:
            severity: Error severity level
            category: Error category
            error_message: Error description
            component: Component where error occurred
            function: Function where error occurred
            error_code: Optional error code
            **kwargs: Additional context

        Returns:
            str: Error ID
        """
        try:
            error_id = f"error_{int(time.time() * 1000)}"

            # Calculate impact score
            impact_score = await self._calculate_error_impact(
                severity,
                category,
                component,
            )

            # Create error event
            error_event = ErrorEvent(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                error_message=error_message,
                error_code=error_code,
                component=component,
                function=function,
                stack_trace=kwargs.get("stack_trace"),
                user_context=kwargs.get("user_context", {}),
                system_state=kwargs.get("system_state", {}),
                impact_score=impact_score,
                affected_components=kwargs.get("affected_components", []),
                business_impact=kwargs.get("business_impact", ""),
            )

            # Store error event
            self.error_events[error_id] = error_event

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_error_event(error_event)

            # Check if this error should trigger alerts
            await self._check_error_alerts(error_event)

            # Update statistics
            self.detection_stats["errors_detected"] += 1
            self.detection_stats["last_update"] = datetime.now()

            self.logger.warning(
                f"Recorded {severity.value} error: {error_message} (ID: {error_id})",
            )

            return error_id

        except Exception:
            self.logger.exception(failed("Failed to record error event: {e}"))
            return ""

    async def _calculate_error_impact(
        self,
        severity: AlertSeverity,
        category: ErrorCategory,
        component: str,
    ) -> float:
        """Calculate error impact score."""
        try:
            # Base impact from severity
            severity_scores = {
                AlertSeverity.INFO: 0.1,
                AlertSeverity.WARNING: 0.3,
                AlertSeverity.ERROR: 0.6,
                AlertSeverity.CRITICAL: 0.8,
                AlertSeverity.EMERGENCY: 1.0,
            }

            impact = severity_scores.get(severity, 0.5)

            # Adjust based on category
            category_multipliers = {
                ErrorCategory.SYSTEM: 1.2,
                ErrorCategory.TRADING: 1.3,
                ErrorCategory.SECURITY: 1.5,
                ErrorCategory.MODEL: 1.1,
                ErrorCategory.DATA: 1.0,
                ErrorCategory.NETWORK: 0.9,
                ErrorCategory.CONFIGURATION: 0.8,
                ErrorCategory.PERFORMANCE: 1.0,
            }

            impact *= category_multipliers.get(category, 1.0)

            # Adjust based on critical components
            critical_components = ["tactician", "analyst", "exchange", "supervisor"]
            if any(comp in component.lower() for comp in critical_components):
                impact *= 1.2

            return min(impact, 1.0)

        except Exception:
            self.logger.exception(failed("Failed to calculate error impact: {e}"))
            return 0.5

    async def _store_error_event(self, error_event: ErrorEvent) -> None:
        """Store error event in database."""
        try:
            data = {
                "error_id": error_event.error_id,
                "timestamp": error_event.timestamp,
                "severity": error_event.severity.value,
                "category": error_event.category.value,
                "error_message": error_event.error_message,
                "error_code": error_event.error_code,
                "component": error_event.component,
                "function": error_event.function,
                "is_resolved": error_event.is_resolved,
                "impact_score": error_event.impact_score,
                "error_details": json.dumps(asdict(error_event), default=str),
            }

            await self.storage_manager.insert_data("error_events", data)

        except Exception:
            self.logger.exception(failed("Failed to store error event: {e}"))
            raise

    async def _check_error_alerts(self, error_event: ErrorEvent) -> None:
        """Check if error should trigger alerts."""
        try:
            # Immediate alerts for critical errors
            if error_event.severity in [
                AlertSeverity.CRITICAL,
                AlertSeverity.EMERGENCY,
            ]:
                await self._send_immediate_alert(error_event)

            # Check error rate spikes
            await self._check_error_rate_spike(error_event)

        except Exception:
            self.logger.exception(failed("Failed to check error alerts: {e}"))

    async def _send_immediate_alert(self, error_event: ErrorEvent) -> None:
        """Send immediate alert for critical errors."""
        try:
            alert_message = f"""
CRITICAL ERROR DETECTED

Error ID: {error_event.error_id}
Timestamp: {error_event.timestamp}
Severity: {error_event.severity.value}
Category: {error_event.category.value}
Component: {error_event.component}
Message: {error_event.error_message}
Impact Score: {error_event.impact_score:.2f}

Immediate attention required!
"""

            # Send notifications
            if self.enable_email_alerts:
                await self._send_email_alert("CRITICAL ERROR", alert_message)

            if self.enable_slack_alerts:
                await self._send_slack_alert("🚨 CRITICAL ERROR", alert_message)

            self.detection_stats["alerts_sent"] += 1

        except Exception:
            self.logger.exception(failed("Failed to send immediate alert: {e}"))

    async def _check_error_rate_spike(self, error_event: ErrorEvent) -> None:
        """Check for error rate spikes."""
        try:
            # Count recent errors
            recent_cutoff = datetime.now() - timedelta(minutes=10)
            recent_errors = [
                err
                for err in self.error_events.values()
                if err.timestamp >= recent_cutoff
                and err.category == error_event.category
            ]

            if len(recent_errors) >= 5:  # 5 errors in 10 minutes
                await self._trigger_error_rate_alert(
                    error_event.category,
                    len(recent_errors),
                )

        except Exception:
            self.logger.exception(failed("Failed to check error rate spike: {e}"))

    async def _trigger_error_rate_alert(
        self,
        category: ErrorCategory,
        error_count: int,
    ) -> None:
        """Trigger error rate spike alert."""
        try:
            alert_message = f"""
ERROR RATE SPIKE DETECTED

Category: {category.value}
Error Count: {error_count} errors in last 10 minutes
Threshold: 5 errors

This indicates a potential system issue requiring investigation.
"""

            if self.enable_email_alerts:
                await self._send_email_alert("Error Rate Spike", alert_message)

            self.detection_stats["alerts_sent"] += 1

        except Exception:
            self.logger.exception(failed("Failed to trigger error rate alert: {e}"))

    async def detect_anomaly(
        self,
        metric_name: str,
        current_value: float,
        **kwargs,
    ) -> AnomalyDetection | None:
        """
        Detect anomalies in metric values.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            **kwargs: Additional context

        Returns:
            AnomalyDetection: Anomaly detection result or None if no anomaly
        """
        try:
            # Get baseline for this metric
            baseline = self.metric_baselines.get(metric_name)
            if not baseline:
                # Not enough historical data
                return None

            # Calculate deviation
            mean = baseline["mean"]
            std = baseline["std"]

            deviation_score = 0.0 if std == 0 else abs(current_value - mean) / std

            # Check if anomalous
            if deviation_score >= 2.0:  # 2 standard deviations
                # Determine severity
                if deviation_score >= 4.0:
                    severity = AlertSeverity.CRITICAL
                elif deviation_score >= 3.0:
                    severity = AlertSeverity.ERROR
                else:
                    severity = AlertSeverity.WARNING

                # Determine anomaly type
                anomaly_type = await self._classify_anomaly_type(
                    metric_name,
                    current_value,
                    baseline,
                )

                # Calculate percentile rank
                percentile_rank = (
                    stats.percentileofscore(
                        [baseline["min"], baseline["max"]],
                        current_value,
                    )
                    / 100.0
                )

                # Create anomaly detection
                anomaly_id = f"anomaly_{metric_name}_{int(time.time())}"

                anomaly = AnomalyDetection(
                    anomaly_id=anomaly_id,
                    timestamp=datetime.now(),
                    anomaly_type=anomaly_type,
                    severity=severity,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=mean,
                    threshold=mean + 2 * std,
                    deviation_score=deviation_score,
                    historical_mean=mean,
                    historical_std=std,
                    percentile_rank=percentile_rank,
                    detection_method="statistical",
                    confidence=min(deviation_score / 4.0, 1.0),
                    trend_direction=kwargs.get("trend_direction", "unknown"),
                    related_metrics=kwargs.get("related_metrics", {}),
                    potential_causes=await self._identify_potential_causes(
                        metric_name,
                        current_value,
                        baseline,
                    ),
                )

                # Store anomaly
                self.anomaly_detections[anomaly_id] = anomaly

                # Store in database
                if hasattr(self, "storage_manager"):
                    await self._store_anomaly_detection(anomaly)

                # Check if alert should be sent
                await self._check_anomaly_alerts(anomaly)

                # Update statistics
                self.detection_stats["anomalies_detected"] += 1
                self.detection_stats["last_update"] = datetime.now()

                self.logger.warning(
                    f"Anomaly detected in {metric_name}: {current_value} (expected: {mean:.2f})",
                )

                return anomaly

            return None

        except Exception:
            self.logger.exception(failed("Failed to detect anomaly for {metric_name}: {e}"))
            return None

    async def _classify_anomaly_type(
        self,
        metric_name: str,
        current_value: float,
        baseline: dict[str, float],
    ) -> AnomalyType:
        """Classify the type of anomaly."""
        try:
            # Simple classification based on metric name and value
            if "cpu" in metric_name.lower():
                return AnomalyType.CPU_SPIKE
            if "memory" in metric_name.lower():
                return AnomalyType.MEMORY_LEAK
            if (
                "latency" in metric_name.lower()
                or "response_time" in metric_name.lower()
            ):
                return AnomalyType.LATENCY_SPIKE
            if "error_rate" in metric_name.lower():
                return AnomalyType.ERROR_RATE_SPIKE
            if "prediction" in metric_name.lower():
                return AnomalyType.PREDICTION_DRIFT
            if (
                "accuracy" in metric_name.lower()
                or "performance" in metric_name.lower()
            ):
                return AnomalyType.PERFORMANCE_DEGRADATION
            if "volume" in metric_name.lower():
                return AnomalyType.VOLUME_SPIKE
            if "feature" in metric_name.lower():
                return AnomalyType.FEATURE_DRIFT
            if "network" in metric_name.lower():
                return AnomalyType.NETWORK_ISSUES
            return AnomalyType.PERFORMANCE_DEGRADATION

        except Exception:
            self.logger.exception(failed("Failed to classify anomaly type: {e}"))
            return AnomalyType.PERFORMANCE_DEGRADATION

    async def _identify_potential_causes(
        self,
        metric_name: str,
        current_value: float,
        baseline: dict[str, float],
    ) -> list[str]:
        """Identify potential causes for anomaly."""
        try:
            causes = []

            if "cpu" in metric_name.lower():
                causes = [
                    "High computational load",
                    "Inefficient algorithm",
                    "Resource leak",
                    "External process interference",
                ]
            elif "memory" in metric_name.lower():
                causes = [
                    "Memory leak",
                    "Large data structures",
                    "Inefficient memory usage",
                    "Too many concurrent operations",
                ]
            elif "error_rate" in metric_name.lower():
                causes = [
                    "System malfunction",
                    "Network connectivity issues",
                    "Invalid input data",
                    "Configuration error",
                ]
            elif "prediction" in metric_name.lower():
                causes = [
                    "Model drift",
                    "Data distribution change",
                    "Feature engineering issues",
                    "Market regime change",
                ]
            else:
                causes = ["Unknown cause", "System performance issue"]

            return causes

        except Exception:
            self.logger.exception(failed("Failed to identify potential causes: {e}"))
            return ["Unknown cause"]

    async def _store_anomaly_detection(self, anomaly: AnomalyDetection) -> None:
        """Store anomaly detection in database."""
        try:
            data = {
                "anomaly_id": anomaly.anomaly_id,
                "timestamp": anomaly.timestamp,
                "anomaly_type": anomaly.anomaly_type.value,
                "severity": anomaly.severity.value,
                "metric_name": anomaly.metric_name,
                "current_value": anomaly.current_value,
                "expected_value": anomaly.expected_value,
                "deviation_score": anomaly.deviation_score,
                "detection_method": anomaly.detection_method,
                "confidence": anomaly.confidence,
                "auto_resolved": anomaly.auto_resolved,
                "anomaly_details": json.dumps(asdict(anomaly), default=str),
            }

            await self.storage_manager.insert_data("anomaly_detections", data)

        except Exception:
            self.logger.exception(failed("Failed to store anomaly detection: {e}"))
            raise

    async def _check_anomaly_alerts(self, anomaly: AnomalyDetection) -> None:
        """Check if anomaly should trigger alerts."""
        try:
            # Send alerts for high-severity anomalies
            if anomaly.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                await self._send_anomaly_alert(anomaly)

        except Exception:
            self.logger.exception(failed("Failed to check anomaly alerts: {e}"))

    async def _send_anomaly_alert(self, anomaly: AnomalyDetection) -> None:
        """Send anomaly alert."""
        try:
            alert_message = f"""
ANOMALY DETECTED

Anomaly ID: {anomaly.anomaly_id}
Type: {anomaly.anomaly_type.value}
Severity: {anomaly.severity.value}
Metric: {anomaly.metric_name}
Current Value: {anomaly.current_value:.2f}
Expected Value: {anomaly.expected_value:.2f}
Deviation: {anomaly.deviation_score:.2f} standard deviations
Confidence: {anomaly.confidence:.2f}

Potential Causes:
{chr(10).join(f"- {cause}" for cause in anomaly.potential_causes)}
"""

            if self.enable_email_alerts:
                await self._send_email_alert("Anomaly Detection", alert_message)

            self.detection_stats["alerts_sent"] += 1

        except Exception:
            self.logger.exception(failed("Failed to send anomaly alert: {e}"))

    async def collect_system_health_metrics(self) -> SystemHealthMetrics:
        """Collect current system health metrics."""
        try:
            import psutil

            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Network latency (placeholder - would ping a reliable server)
            network_latency = 50.0  # Placeholder

            # Application metrics (placeholders - would integrate with actual metrics)
            active_connections = 10
            request_rate = 25.0
            error_rate = 1.2
            response_time = 150.0

            return SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_latency_ms=network_latency,
                active_connections=active_connections,
                request_rate_per_second=request_rate,
                error_rate_percent=error_rate,
                response_time_ms=response_time,
            )

        except Exception:
            self.logger.exception(failed("Failed to collect system health metrics: {e}"))
            # Return basic metrics
            return SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_latency_ms=0.0,
                active_connections=0,
                request_rate_per_second=0.0,
                error_rate_percent=0.0,
                response_time_ms=0.0,
            )

    async def _monitor_system_health(self) -> None:
        """Background task to monitor system health."""
        try:
            while True:
                # Collect metrics
                metrics = await self.collect_system_health_metrics()

                # Store metrics
                self.health_metrics_history.append(metrics)

                # Keep only recent history
                cutoff_time = datetime.now() - timedelta(
                    hours=self.lookback_window_hours,
                )
                self.health_metrics_history = [
                    m for m in self.health_metrics_history if m.timestamp >= cutoff_time
                ]

                # Store in database
                if hasattr(self, "storage_manager"):
                    await self._store_health_metrics(metrics)

                # Check for anomalies
                if self.enable_anomaly_detection:
                    await self.detect_anomaly(
                        "cpu_usage_percent",
                        metrics.cpu_usage_percent,
                    )
                    await self.detect_anomaly(
                        "memory_usage_percent",
                        metrics.memory_usage_percent,
                    )
                    await self.detect_anomaly(
                        "error_rate_percent",
                        metrics.error_rate_percent,
                    )
                    await self.detect_anomaly(
                        "response_time_ms",
                        metrics.response_time_ms,
                    )

                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            self.logger.info("System health monitoring task cancelled")
        except Exception:
            self.logger.exception(error("Error in system health monitoring: {e}"))

    async def _store_health_metrics(self, metrics: SystemHealthMetrics) -> None:
        """Store health metrics in database."""
        try:
            data = {
                "timestamp": metrics.timestamp,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "memory_usage_percent": metrics.memory_usage_percent,
                "disk_usage_percent": metrics.disk_usage_percent,
                "network_latency_ms": metrics.network_latency_ms,
                "active_connections": metrics.active_connections,
                "request_rate_per_second": metrics.request_rate_per_second,
                "error_rate_percent": metrics.error_rate_percent,
                "response_time_ms": metrics.response_time_ms,
                "health_details": json.dumps(asdict(metrics), default=str),
            }

            await self.storage_manager.insert_data("system_health_metrics", data)

        except Exception:
            self.logger.exception(failed("Failed to store health metrics: {e}"))

    async def _monitor_anomalies(self) -> None:
        """Background task for anomaly detection."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute

                # Update baselines periodically
                if len(self.health_metrics_history) >= self.min_data_points:
                    await self._update_metric_baselines()

        except asyncio.CancelledError:
            self.logger.info("Anomaly monitoring task cancelled")
        except Exception:
            self.logger.exception(error("Error in anomaly monitoring: {e}"))

    async def _update_metric_baselines(self) -> None:
        """Update metric baselines with recent data."""
        try:
            # Update baselines using recent health metrics
            if self.health_metrics_history:
                metrics_data = {
                    "cpu_usage_percent": [
                        m.cpu_usage_percent for m in self.health_metrics_history
                    ],
                    "memory_usage_percent": [
                        m.memory_usage_percent for m in self.health_metrics_history
                    ],
                    "error_rate_percent": [
                        m.error_rate_percent for m in self.health_metrics_history
                    ],
                    "response_time_ms": [
                        m.response_time_ms for m in self.health_metrics_history
                    ],
                }

                for metric_name, values in metrics_data.items():
                    if len(values) >= self.min_data_points:
                        baseline = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "p95": float(np.percentile(values, 95)),
                            "p99": float(np.percentile(values, 99)),
                        }

                        self.metric_baselines[metric_name] = baseline

        except Exception:
            self.logger.exception(failed("Failed to update metric baselines: {e}"))

    async def _evaluate_alert_rules(self) -> None:
        """Background task to evaluate alert rules."""
        try:
            while True:
                await asyncio.sleep(30)  # Evaluate every 30 seconds

                # Get latest metrics
                if self.health_metrics_history:
                    latest_metrics = self.health_metrics_history[-1]

                    # Evaluate each alert rule
                    for rule in self.alert_rules.values():
                        if rule.is_enabled:
                            await self._evaluate_rule(rule, latest_metrics)

        except asyncio.CancelledError:
            self.logger.info("Alert rule evaluation task cancelled")
        except Exception:
            self.logger.exception(error("Error in alert rule evaluation: {e}"))

    async def _evaluate_rule(
        self,
        rule: AlertRule,
        metrics: SystemHealthMetrics,
    ) -> None:
        """Evaluate a single alert rule."""
        try:
            # Get metric value
            metric_value = getattr(metrics, rule.metric_name, None)
            if metric_value is None:
                return

            # Check condition
            triggered = False

            if rule.condition == "greater_than" and rule.threshold is not None:
                triggered = metric_value > rule.threshold
            elif rule.condition == "less_than" and rule.threshold is not None:
                triggered = metric_value < rule.threshold
            elif rule.condition == "equals" and rule.threshold is not None:
                triggered = metric_value == rule.threshold

            if triggered:
                await self._handle_rule_trigger(rule, metric_value)

        except Exception:
            self.logger.exception(failed("Failed to evaluate rule {rule.rule_id}: {e}"))

    async def _handle_rule_trigger(self, rule: AlertRule, metric_value: float) -> None:
        """Handle alert rule trigger."""
        try:
            # Check cooldown
            last_alert = self.active_alerts.get(rule.rule_id)
            if last_alert:
                time_since_last = (datetime.now() - last_alert).total_seconds() / 60.0
                if time_since_last < rule.cooldown_minutes:
                    return  # Still in cooldown

            # Check rate limiting
            current_hour = datetime.now().hour
            alert_count_key = f"{rule.rule_id}_{current_hour}"
            current_count = self.alert_counts.get(alert_count_key, 0)

            if current_count >= rule.max_alerts_per_hour:
                return  # Rate limit exceeded

            # Send alert
            await self._send_rule_alert(rule, metric_value)

            # Update tracking
            self.active_alerts[rule.rule_id] = datetime.now()
            self.alert_counts[alert_count_key] = current_count + 1

        except Exception:
            self.logger.exception(failed("Failed to handle rule trigger: {e}"))

    async def _send_rule_alert(self, rule: AlertRule, metric_value: float) -> None:
        """Send alert for triggered rule."""
        try:
            alert_message = f"""
ALERT: {rule.rule_name}

Rule: {rule.rule_name}
Severity: {rule.severity.value}
Metric: {rule.metric_name}
Current Value: {metric_value}
Threshold: {rule.threshold}
Condition: {rule.condition}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            if rule.notify_email and self.enable_email_alerts:
                await self._send_email_alert(
                    f"Alert: {rule.rule_name}",
                    alert_message,
                    rule.email_recipients,
                )

            if rule.notify_slack and self.enable_slack_alerts:
                await self._send_slack_alert(f"⚠️ {rule.rule_name}", alert_message)

            self.detection_stats["alerts_sent"] += 1

        except Exception:
            self.logger.exception(failed("Failed to send rule alert: {e}"))

    async def _send_email_alert(
        self,
        subject: str,
        message: str,
        recipients: list[str] = None,
    ) -> None:
        """Send email alert."""
        try:
            if not self.enable_email_alerts:
                return

            if recipients is None:
                recipients = self.email_config.get("default_recipients", [])

            if not recipients:
                return

            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.email_user
            msg["To"] = ", ".join(recipients)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)

            self.logger.info(f"Email alert sent: {subject}")

        except Exception:
            self.logger.exception(failed("Failed to send email alert: {e}"))

    async def _send_slack_alert(self, title: str, message: str) -> None:
        """Send Slack alert."""
        try:
            if not self.enable_slack_alerts or not self.slack_webhook:
                return

            import aiohttp

            payload = {
                "text": f"{title}\n\n{message}",
                "username": "Ares Trading Bot",
                "icon_emoji": ":robot_face:",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack alert sent: {title}")
                    else:
                        self.logger.error(
                            f"Failed to send Slack alert: {response.status}",
                        )

        except Exception:
            self.logger.exception(failed("Failed to send Slack alert: {e}"))

    async def get_detection_statistics(self) -> dict[str, Any]:
        """Get comprehensive detection statistics."""
        try:
            stats = self.detection_stats.copy()

            # Add current state information
            stats.update(
                {
                    "active_error_events": len(
                        [e for e in self.error_events.values() if not e.is_resolved],
                    ),
                    "total_error_events": len(self.error_events),
                    "active_anomalies": len(
                        [
                            a
                            for a in self.anomaly_detections.values()
                            if not a.auto_resolved
                        ],
                    ),
                    "total_anomalies": len(self.anomaly_detections),
                    "active_alert_rules": len(
                        [r for r in self.alert_rules.values() if r.is_enabled],
                    ),
                    "total_alert_rules": len(self.alert_rules),
                    "health_metrics_points": len(self.health_metrics_history),
                    "metric_baselines": len(self.metric_baselines),
                    "is_initialized": self.is_initialized,
                },
            )

            # Error distribution
            if self.error_events:
                error_by_severity = {}
                error_by_category = {}

                for error in self.error_events.values():
                    severity = error.severity.value
                    category = error.category.value

                    error_by_severity[severity] = error_by_severity.get(severity, 0) + 1
                    error_by_category[category] = error_by_category.get(category, 0) + 1

                stats["error_distribution_by_severity"] = error_by_severity
                stats["error_distribution_by_category"] = error_by_category

            return stats

        except Exception:
            self.logger.exception(failed("Failed to get detection statistics: {e}"))
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Error Detection System...")

            # Clear history
            self.health_metrics_history.clear()

            # Close storage connections
            if hasattr(self, "storage_manager"):
                await self.storage_manager.close()

            self.logger.info("Error Detection System cleanup completed")

        except Exception:
            self.logger.exception(failed("Failed to cleanup Error Detection System: {e}"))


# Setup function for integration
async def setup_error_detection_system(
    config: dict[str, Any],
) -> ErrorDetectionSystem | None:
    """
    Setup and return a configured Error Detection System instance.

    Args:
        config: Configuration dictionary

    Returns:
        ErrorDetectionSystem: Configured system instance or None if setup failed
    """
    try:
        system = ErrorDetectionSystem(config)
        if await system.initialize():
            return system
        return None
    except Exception:
        system_logger.exception(failed("Failed to setup Error Detection System: {e}"))
        return None
