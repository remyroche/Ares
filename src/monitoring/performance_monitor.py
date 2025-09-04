from __future__ import annotations

from src.core.decorators import cached, handles_errors, log_execution_time
from src.core.domain import PerformanceLevel

# src/monitoring/performance_monitor.py


"""
Performance Monitor for Dual Model System
Comprehensive monitoring of model performance, system metrics, trading performance, and optimization opportunities.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.utils.logger import system_logger
import asyncio
from src.core.decorators.errors import handles_errors


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: datetime = field(default_factory=datetime.now)
    model_accuracy: float = 0.0
    model_precision: float = 0.0
    model_recall: float = 0.0
    model_f1_score: float = 0.0
    model_auc: float = 0.0
    trading_win_rate: float = 0.0
    trading_profit_factor: float = 0.0
    trading_sharpe_ratio: float = 0.0
    trading_max_drawdown: float = 0.0
    trading_total_return: float = 0.0
    system_memory_usage: float = 0.0
    system_cpu_usage: float = 0.0
    system_response_time: float = 0.0
    system_throughput: float = 0.0
    confidence_analyst: float = 0.0
    confidence_tactician: float = 0.0
    confidence_final: float = 0.0


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = system_logger.getChild("PerformanceMonitor")

        # Monitoring configuration
        self.monitoring_config = config.get(
            "performance_monitoring",
            {
                "enable_monitoring": True,
                "monitoring_interval_seconds": 60,
                "metrics_history_size": 1000,
            },
        )

        # Metrics storage
        self.metrics_history: deque[PerformanceMetrics] = deque(
            maxlen=int(self.monitoring_config.get("metrics_history_size", 1000)),
        )

    @log_execution_time(level=PerformanceLevel.DETAILED)
    @log_execution_time()
    @cached()
    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance_monitor.initialize",
    )
    async def initialize(self) -> bool:
        self.logger.info("📈 Initializing Performance Monitor ...")
        self.metrics_history.clear()
        self.logger.info("✅ Performance Monitor initialized successfully")
        return True

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        self.metrics_history.append(metrics)

    def latest_metrics(self) -> PerformanceMetrics | None:
        return self.metrics_history[-1] if self.metrics_history else None
