"""
Base pipeline framework for Ares trading bot (minimal scaffold).
"""


from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.centralized_decorators import (
    performance_monitor,
    PerformanceLevel,
    handle_errors,
    handle_specific_errors,
    resource_monitor,
    memory_efficient,
    pipeline_checkpoint,
)
from src.utils.logger import system_logger


@dataclass
class PipelineConfig:
    name: str
    symbol: str
    exchange: str
    environment: str  # "live", "backtest", "training"

    checkpoint_enabled: bool = True
    email_notifications: bool = True
    pid_file_enabled: bool = True
    loop_interval_seconds: int = 10
    max_retries: int = 3
    timeout_seconds: int = 3600

    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    risk_config: Dict[str, Any] = field(default_factory=dict)
    notification_config: Dict[str, Any] = field(default_factory=dict)

    parallel_execution: bool = False
    max_workers: int = 4
    continue_on_failure: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.name:
            errors.append("Pipeline name is required")
        if not self.symbol:
            errors.append("Symbol is required")
        if not self.exchange:
            errors.append("Exchange is required")
        if self.environment not in ["live", "backtest", "training"]:
            errors.append("Environment must be 'live', 'backtest', or 'training'")
        if self.loop_interval_seconds <= 0:
            errors.append("Loop interval must be positive")
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        return errors


@dataclass
class PipelineMetrics:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    stages_completed: int = 0
    stages_failed: int = 0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    def update_duration(self) -> None:
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class BasePipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("BasePipeline")
        self.pipeline_config: Dict[str, Any] = self.config.get("base_pipeline", {})
        self.metrics = PipelineMetrics()
        self.is_running: bool = False

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @resource_monitor()
    @memory_efficient()
    @pipeline_checkpoint(checkpoint_name="base_pipeline.initialize")
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid base pipeline configuration"),
            AttributeError: (False, "Missing required pipeline parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="base_pipeline.initialize",
    )
    @performance_monitor(level=PerformanceLevel.DETAILED)
    @resource_monitor()
    @memory_efficient()
    @pipeline_checkpoint(checkpoint_name="base_pipeline.shutdown")
    @handle_errors(exceptions=(Exception,), default_return=False, context="base_pipeline.shutdown")