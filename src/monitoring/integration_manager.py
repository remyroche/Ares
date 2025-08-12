#!/usr/bin/env python3
"""
Monitoring Integration Manager

This module provides unified coordination of all monitoring components including
metrics dashboard, advanced tracing, ML monitoring, report scheduling, and tracking.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    initialization_error,
)

from .advanced_tracer import AdvancedTracer, setup_advanced_tracer
from .correlation_manager import CorrelationManager, setup_correlation_manager
from .metrics_dashboard import MetricsDashboard, setup_metrics_dashboard
from .ml_monitor import MLMonitor, setup_ml_monitor
from .report_scheduler import ReportScheduler, setup_report_scheduler
from .tracking_system import TrackingSystem, setup_tracking_system


@dataclass
class MonitoringComponents:
    """Container for all monitoring components."""

    metrics_dashboard: MetricsDashboard | None = None
    advanced_tracer: AdvancedTracer | None = None
    correlation_manager: CorrelationManager | None = None
    ml_monitor: MLMonitor | None = None
    report_scheduler: ReportScheduler | None = None
    tracking_system: TrackingSystem | None = None


class MonitoringIntegrationManager:
    """
    Unified monitoring integration manager that coordinates all monitoring components.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize monitoring integration manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MonitoringIntegrationManager")

        # Integration configuration
        self.integration_config = config.get("monitoring_integration", {})
        self.enable_unified_monitoring = self.integration_config.get(
            "enable_unified_monitoring",
            True,
        )
        self.enable_cross_component_tracking = self.integration_config.get(
            "enable_cross_component_tracking",
            True,
        )
        self.enable_performance_correlation = self.integration_config.get(
            "enable_performance_correlation",
            True,
        )

        # Component storage
        self.components = MonitoringComponents()

        # Integration state
        self.is_integrated = False
        self.integration_task: asyncio.Task | None = None

        # Cross-component data
        self.cross_component_metrics: dict[str, Any] = {}
        self.performance_correlations: dict[str, dict[str, float]] = {}

        self.logger.info("🔗 Monitoring Integration Manager initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid integration configuration"),
            AttributeError: (False, "Missing required integration parameters"),
        },
        default_return=False,
        context="integration manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the monitoring integration manager."""
        try:
            self.logger.info("Initializing Monitoring Integration Manager...")

            # Initialize all monitoring components
            await self._initialize_components()

            # Initialize cross-component tracking
            if self.enable_cross_component_tracking:
                await self._initialize_cross_component_tracking()

            # Initialize performance correlation
            if self.enable_performance_correlation:
                await self._initialize_performance_correlation()

            self.logger.info(
                "✅ Monitoring Integration Manager initialization completed",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Monitoring Integration Manager initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="components initialization",
    )
    async def _initialize_components(self) -> None:
        """Initialize all monitoring components."""
        try:
            # Initialize metrics dashboard
            self.components.metrics_dashboard = await setup_metrics_dashboard(
                self.config,
            )
            if self.components.metrics_dashboard:
                self.logger.info("✅ Metrics Dashboard initialized")

            # Initialize advanced tracer
            self.components.advanced_tracer = await setup_advanced_tracer(self.config)
            if self.components.advanced_tracer:
                self.logger.info("✅ Advanced Tracer initialized")

            # Initialize correlation manager
            self.components.correlation_manager = await setup_correlation_manager(
                self.config,
            )
            if self.components.correlation_manager:
                self.logger.info("✅ Correlation Manager initialized")

            # Initialize ML monitor
            self.components.ml_monitor = await setup_ml_monitor(self.config)
            if self.components.ml_monitor:
                self.logger.info("✅ ML Monitor initialized")

            # Initialize report scheduler
            self.components.report_scheduler = await setup_report_scheduler(self.config)
            if self.components.report_scheduler:
                self.logger.info("✅ Report Scheduler initialized")

            # Initialize tracking system
            self.components.tracking_system = await setup_tracking_system(self.config)
            if self.components.tracking_system:
                self.logger.info("✅ Tracking System initialized")

        except Exception:
            self.logger.exception(initialization_error("Error initializing components: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="cross-component tracking initialization",
    )
    async def _initialize_cross_component_tracking(self) -> None:
        """Initialize cross-component tracking."""
        try:
            # Initialize cross-component tracking structures
            self.cross_component_metrics.clear()

            self.logger.info("Cross-component tracking initialized")

        except Exception:
            self.logger.exception(
                initialization_error(
                    "Error initializing cross-component tracking: {e}"
                ),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance correlation initialization",
    )
    async def _initialize_performance_correlation(self) -> None:
        """Initialize performance correlation."""
        try:
            # Initialize performance correlation structures
            self.performance_correlations.clear()

            self.logger.info("Performance correlation initialized")

        except Exception:
            self.logger.exception(
                initialization_error("Error initializing performance correlation: {e}"),
            )

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Integration failed"),
        },
        default_return=False,
        context="monitoring integration",
    )
    async def start_integration(self) -> bool:
        """Start monitoring integration."""
        try:
            self.is_integrated = True

            # Start all components
            await self._start_all_components()

            # Start integration task
            self.integration_task = asyncio.create_task(self._integration_loop())

            self.logger.info("🚀 Monitoring Integration started")
            return True

        except Exception:
            self.logger.exception(error("Error starting integration: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="all components start",
    )
    async def _start_all_components(self) -> None:
        """Start all monitoring components."""
        try:
            # Start metrics dashboard
            if self.components.metrics_dashboard:
                await self.components.metrics_dashboard.start()

            # Start advanced tracer
            if self.components.advanced_tracer:
                await self.components.advanced_tracer.start()

            # Start correlation manager
            if self.components.correlation_manager:
                await self.components.correlation_manager.start()

            # Start ML monitor
            if self.components.ml_monitor:
                await self.components.ml_monitor.start_monitoring()

            # Start report scheduler
            if self.components.report_scheduler:
                await self.components.report_scheduler.start_scheduling()

            # Start tracking system
            if self.components.tracking_system:
                await self.components.tracking_system.start_tracking()

        except Exception:
            self.logger.exception(error("Error starting all components: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="integration loop",
    )
    async def _integration_loop(self) -> None:
        """Main integration loop."""
        try:
            while self.is_integrated:
                await self._update_cross_component_metrics()
                await self._update_performance_correlations()
                await asyncio.sleep(30)  # Update every 30 seconds

        except Exception:
            self.logger.exception(error("Error in integration loop: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="cross-component metrics update",
    )
    async def _update_cross_component_metrics(self) -> None:
        """Update cross-component metrics."""
        try:
            # Collect metrics from all components
            metrics = {}

            # Metrics dashboard metrics
            if self.components.metrics_dashboard:
                dashboard_data = self.components.metrics_dashboard.get_dashboard_data()
                metrics["dashboard"] = dashboard_data

            # Advanced tracer metrics
            if self.components.advanced_tracer:
                tracer_stats = self.components.advanced_tracer.get_trace_statistics()
                metrics["tracer"] = tracer_stats

            # Correlation manager metrics
            if self.components.correlation_manager:
                correlation_stats = (
                    self.components.correlation_manager.get_correlation_statistics()
                )
                metrics["correlation"] = correlation_stats

            # ML monitor metrics
            if self.components.ml_monitor:
                ml_summary = self.components.ml_monitor.get_ml_monitoring_summary()
                metrics["ml_monitor"] = ml_summary

            # Report scheduler metrics
            if self.components.report_scheduler:
                scheduler_status = (
                    self.components.report_scheduler.get_scheduler_status()
                )
                metrics["report_scheduler"] = scheduler_status

            # Tracking system metrics
            if self.components.tracking_system:
                tracking_summary = (
                    self.components.tracking_system.get_tracking_summary()
                )
                metrics["tracking"] = tracking_summary

            self.cross_component_metrics = metrics

        except Exception:
            self.logger.exception(error("Error updating cross-component metrics: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance correlations update",
    )
    async def _update_performance_correlations(self) -> None:
        """Update performance correlations."""
        try:
            # Calculate correlations between different components
            correlations = {}

            # This would calculate actual correlations between component performance
            # For now, create sample correlations
            if self.cross_component_metrics:
                components = list(self.cross_component_metrics.keys())
                for i, comp1 in enumerate(components):
                    for comp2 in components[i + 1 :]:
                        correlation_key = f"{comp1}_vs_{comp2}"
                        correlations[correlation_key] = {
                            "correlation": 0.75,  # Sample correlation
                            "significance": 0.05,
                            "timestamp": datetime.now().isoformat(),
                        }

            self.performance_correlations = correlations

        except Exception:
            self.logger.exception(error("Error updating performance correlations: {e}"))

    def get_integration_status(self) -> dict[str, Any]:
        """Get integration status."""
        try:
            return {
                "is_integrated": self.is_integrated,
                "components": {
                    "metrics_dashboard": self.components.metrics_dashboard is not None,
                    "advanced_tracer": self.components.advanced_tracer is not None,
                    "correlation_manager": self.components.correlation_manager
                    is not None,
                    "ml_monitor": self.components.ml_monitor is not None,
                    "report_scheduler": self.components.report_scheduler is not None,
                    "tracking_system": self.components.tracking_system is not None,
                },
                "cross_component_tracking": self.enable_cross_component_tracking,
                "performance_correlation": self.enable_performance_correlation,
            }

        except Exception:
            self.logger.exception(error("Error getting integration status: {e}"))
            return {}

    def get_cross_component_metrics(self) -> dict[str, Any]:
        """Get cross-component metrics."""
        return self.cross_component_metrics

    def get_performance_correlations(self) -> dict[str, Any]:
        """Get performance correlations."""
        return self.performance_correlations

    def get_unified_dashboard_data(self) -> dict[str, Any]:
        """Get unified dashboard data from all components."""
        try:
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "integration_status": self.get_integration_status(),
                "cross_component_metrics": self.cross_component_metrics,
                "performance_correlations": self.performance_correlations,
            }

            # Add component-specific data
            if self.components.metrics_dashboard:
                dashboard_data["metrics_dashboard"] = (
                    self.components.metrics_dashboard.get_dashboard_data()
                )

            if self.components.advanced_tracer:
                dashboard_data["tracer"] = (
                    self.components.advanced_tracer.get_trace_statistics()
                )

            if self.components.correlation_manager:
                dashboard_data["correlation"] = (
                    self.components.correlation_manager.get_correlation_statistics()
                )

            if self.components.ml_monitor:
                dashboard_data["ml_monitor"] = (
                    self.components.ml_monitor.get_ml_monitoring_summary()
                )

            if self.components.report_scheduler:
                dashboard_data["report_scheduler"] = (
                    self.components.report_scheduler.get_scheduler_status()
                )

            if self.components.tracking_system:
                dashboard_data["tracking"] = (
                    self.components.tracking_system.get_tracking_summary()
                )

            return dashboard_data

        except Exception:
            self.logger.exception(error("Error getting unified dashboard data: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="integration stop",
    )
    async def stop_integration(self) -> None:
        """Stop monitoring integration."""
        try:
            self.is_integrated = False

            # Stop integration task
            if self.integration_task:
                self.integration_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.integration_task

            # Stop all components
            await self._stop_all_components()

            self.logger.info("🛑 Monitoring Integration stopped")

        except Exception:
            self.logger.exception(error("Error stopping integration: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="all components stop",
    )
    async def _stop_all_components(self) -> None:
        """Stop all monitoring components."""
        try:
            # Stop metrics dashboard
            if self.components.metrics_dashboard:
                await self.components.metrics_dashboard.stop()

            # Stop advanced tracer
            if self.components.advanced_tracer:
                await self.components.advanced_tracer.stop()

            # Stop correlation manager
            if self.components.correlation_manager:
                await self.components.correlation_manager.stop()

            # Stop ML monitor
            if self.components.ml_monitor:
                await self.components.ml_monitor.stop_monitoring()

            # Stop report scheduler
            if self.components.report_scheduler:
                await self.components.report_scheduler.stop_scheduling()

            # Stop tracking system
            if self.components.tracking_system:
                await self.components.tracking_system.stop_tracking()

        except Exception:
            self.logger.exception(error("Error stopping all components: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="monitoring integration manager setup",
)
async def setup_monitoring_integration_manager(
    config: dict[str, Any],
) -> MonitoringIntegrationManager | None:
    """
    Setup and initialize monitoring integration manager.

    Args:
        config: Configuration dictionary

    Returns:
        MonitoringIntegrationManager instance or None if setup failed
    """
    try:
        integration_manager = MonitoringIntegrationManager(config)

        if await integration_manager.initialize():
            return integration_manager
        return None

    except Exception:
        system_logger.exception(error("Error setting up monitoring integration manager: {e}"))
        return None
