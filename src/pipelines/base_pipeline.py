"""
Base pipeline framework for Ares trading bot.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    
    name: str
    max_retries: int = 3
    timeout_seconds: int = 300
    enable_logging: bool = True
    enable_metrics: bool = True
    checkpoint_interval: int = 100
    max_memory_mb: int = 1024
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")


@dataclass
class PipelineMetrics:
    """Metrics tracking for pipeline execution."""
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    stages_completed: int = 0
    stages_failed: int = 0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    def update_duration(self) -> None:
        """Update duration based on start and end times."""
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.start_time = None
        self.end_time = None
        self.duration_seconds = None
        self.stages_completed = 0
        self.stages_failed = 0
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0


class BasePipeline(ABC):
    """Abstract base class for all pipeline implementations."""
    
    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the base pipeline."""
        self.config = config
        self.logger = system_logger.getChild(f"Pipeline.{config.name}")
        self.metrics = PipelineMetrics()
        self.is_initialized = False
        self.is_running = False
        
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pipeline initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the pipeline. Must be implemented by subclasses."""
        try:
            self.logger.info(f"🚀 Initializing {self.__class__.__name__}...")
            await self._initialize_impl()
            self.is_initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization logic."""
        pass
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pipeline execution",
    )
    async def execute(self) -> bool:
        """Execute the pipeline. Must be implemented by subclasses."""
        if not self.is_initialized:
            self.logger.error("Pipeline not initialized")
            return False
            
        try:
            self.logger.info(f"🚀 Executing {self.__class__.__name__}...")
            self.is_running = True
            self.metrics.start_time = datetime.now()
            
            result = await self._execute_impl()
            
            self.metrics.end_time = datetime.now()
            self.metrics.update_duration()
            self.is_running = False
            
            if result:
                self.logger.info(f"✅ {self.__class__.__name__} executed successfully")
            else:
                self.logger.error(f"❌ {self.__class__.__name__} execution failed")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Error executing {self.__class__.__name__}: {e}")
            self.is_running = False
            return False
    
    @abstractmethod
    async def _execute_impl(self) -> bool:
        """Implementation-specific execution logic."""
        pass
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pipeline cleanup",
    )
    async def cleanup(self) -> bool:
        """Clean up pipeline resources."""
        try:
            self.logger.info(f"🧹 Cleaning up {self.__class__.__name__}...")
            await self._cleanup_impl()
            self.is_initialized = False
            self.logger.info(f"✅ {self.__class__.__name__} cleaned up successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error cleaning up {self.__class__.__name__}: {e}")
            return False
    
    @abstractmethod
    async def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup logic."""
        pass
    
    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "name": self.config.name,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "metrics": {
                "duration_seconds": self.metrics.duration_seconds,
                "stages_completed": self.metrics.stages_completed,
                "stages_failed": self.metrics.stages_failed,
                "total_operations": self.metrics.total_operations,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations,
            }
        }


