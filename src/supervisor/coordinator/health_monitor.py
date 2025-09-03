"""
Health Monitor Module.

This module monitors overall system health, including resource usage,
performance metrics, and system stability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error
from copy import copy
import asyncio


class HealthMonitor:
    """Monitors overall system health and performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize health monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("HealthMonitor")
        self.health_metrics: Dict[str, Any] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.max_history: int = config.get("max_health_history", 1000)
        self.health_thresholds = config.get("health_thresholds", {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "error_rate": 0.05,
            "latency_ms": 1000,
        })

    @handles_errors(
        exceptions=(Exception,),
        default_return={},
    )
    async def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Dictionary containing health metrics
        """
        try:
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "resource_usage": await self._check_resource_usage(),
                "component_health": await self._check_component_health(),
                "performance_metrics": await self._check_performance_metrics(),
                "overall_status": "healthy",
                "warnings": [],
                "errors": [],
            }

            # Analyze health data for issues
            self._analyze_health_data(health_data)

            # Store in history
            self.health_history.append(health_data)
            if len(self.health_history) > self.max_history:
                self.health_history.pop(0)

            self.health_metrics = health_data
            return health_data

        except Exception as e:
            self.logger.error(error(f"Error checking system health: {e}"))
            return {"status": "error", "message": str(e)}

    async def _check_resource_usage(self) -> Dict[str, float]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            }
        except Exception as e:
            self.logger.error(f"Error checking resource usage: {e}")
            return {}

    async def _check_component_health(self) -> Dict[str, Any]:
        """Check health of individual components."""
        # This would integrate with actual component health checks
        return {
            "analyst": {"status": "healthy", "uptime": 3600},
            "strategist": {"status": "healthy", "uptime": 3600},
            "tactician": {"status": "healthy", "uptime": 3600},
            "sentinel": {"status": "healthy", "uptime": 3600},
        }

    async def _check_performance_metrics(self) -> Dict[str, float]:
        """Check system performance metrics."""
        # This would integrate with actual performance monitoring
        return {
            "avg_latency_ms": 50.0,
            "error_rate": 0.01,
            "throughput_per_sec": 100.0,
            "queue_depth": 10,
        }

    def _analyze_health_data(self, health_data: Dict[str, Any]) -> None:
        """Analyze health data and add warnings/errors."""
        warnings = []
        errors = []
        
        # Check resource usage
        resources = health_data.get("resource_usage", {})
        if resources.get("cpu_percent", 0) > self.health_thresholds["cpu_percent"]:
            warnings.append(f"High CPU usage: {resources['cpu_percent']:.1f}%")
        
        if resources.get("memory_percent", 0) > self.health_thresholds["memory_percent"]:
            warnings.append(f"High memory usage: {resources['memory_percent']:.1f}%")
        
        if resources.get("disk_percent", 0) > self.health_thresholds["disk_percent"]:
            errors.append(f"Critical disk usage: {resources['disk_percent']:.1f}%")
        
        # Check performance metrics
        perf = health_data.get("performance_metrics", {})
        if perf.get("error_rate", 0) > self.health_thresholds["error_rate"]:
            warnings.append(f"High error rate: {perf['error_rate']:.2%}")
        
        if perf.get("avg_latency_ms", 0) > self.health_thresholds["latency_ms"]:
            warnings.append(f"High latency: {perf['avg_latency_ms']:.0f}ms")
        
        # Update health data
        health_data["warnings"] = warnings
        health_data["errors"] = errors
        
        # Determine overall status
        if errors:
            health_data["overall_status"] = "critical"
        elif warnings:
            health_data["overall_status"] = "warning"
        else:
            health_data["overall_status"] = "healthy"

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_metrics.copy()

    def get_health_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get health history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of health check results
        """
        if limit:
            return self.health_history[-limit:]
        return self.health_history.copy()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        recent_checks = self.health_history[-10:] if len(self.health_history) >= 10 else self.health_history
        
        # Calculate averages
        avg_cpu = sum(h.get("resource_usage", {}).get("cpu_percent", 0) for h in recent_checks) / len(recent_checks)
        avg_memory = sum(h.get("resource_usage", {}).get("memory_percent", 0) for h in recent_checks) / len(recent_checks)
        avg_latency = sum(h.get("performance_metrics", {}).get("avg_latency_ms", 0) for h in recent_checks) / len(recent_checks)
        
        # Count statuses
        status_counts = {"healthy": 0, "warning": 0, "critical": 0}
        for check in recent_checks:
            status = check.get("overall_status", "unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        return {
            "current_status": self.health_metrics.get("overall_status", "unknown"),
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_latency_ms": avg_latency,
            "status_distribution": status_counts,
            "checks_analyzed": len(recent_checks),
            "last_check": recent_checks[-1]["timestamp"] if recent_checks else None,
        }