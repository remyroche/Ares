# src/training/factory.py

"""Factory for creating optimized training components based on configuration."""

import os
from typing import Any

from src.config.computational_optimization_config import get_optimization_config
from src.training.enhanced_training_manager_optimized import (
    EnhancedTrainingManagerOptimized,
)
from src.training.memory_profiler import MemoryLeakDetector, MemoryProfiler
from src.training.steps.optimized_step_executor import OptimizedStepExecutor
from src.utils.logger import system_logger


class OptimizedTrainingFactory:
    """Factory for creating optimized training components."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.optimization_config = get_optimization_config(
            config.get("computational_optimization", {}),
        )
        self.logger = system_logger.getChild("OptimizedTrainingFactory")

    def create_enhanced_training_manager(self) -> EnhancedTrainingManagerOptimized:
        """Create an optimized enhanced training manager."""
        self.logger.info("Creating Enhanced Training Manager with optimizations")

        # Merge optimization config with main config
        enhanced_config = self.config.copy()
        enhanced_config["computational_optimization"] = self.optimization_config

        return EnhancedTrainingManagerOptimized(enhanced_config)

    def create_memory_profiler(
        self,
        enable_continuous_monitoring: bool | None = None,
    ) -> MemoryProfiler:
        """Create a memory profiler with appropriate settings."""
        monitoring_config = self.optimization_config.get("monitoring", {})

        if enable_continuous_monitoring is None:
            enable_continuous_monitoring = monitoring_config.get(
                "continuous_monitoring",
                True,
            )

        enable_tracemalloc = monitoring_config.get("memory_leak_detection", True)

        self.logger.info(
            f"Creating Memory Profiler (continuous: {enable_continuous_monitoring})",
        )

        return MemoryProfiler(
            enable_tracemalloc=enable_tracemalloc,
            enable_continuous_monitoring=enable_continuous_monitoring,
        )

    def create_memory_leak_detector(
        self,
        profiler: MemoryProfiler,
    ) -> MemoryLeakDetector:
        """Create a memory leak detector."""
        self.logger.info("Creating Memory Leak Detector")
        return MemoryLeakDetector(profiler)

    def create_step_executor(self) -> OptimizedStepExecutor:
        """Create an optimized step executor."""
        self.logger.info("Creating Optimized Step Executor")

        executor_config = {
            "parallel_execution": self.optimization_config["parallelization"][
                "enabled"
            ],
            "max_workers": self.optimization_config["parallelization"]["max_workers"],
            "enable_caching": self.optimization_config["caching"]["enabled"],
            "enable_memory_optimization": self.optimization_config["memory_management"][
                "enabled"
            ],
            "memory_threshold": self.optimization_config["memory_management"][
                "memory_threshold"
            ],
        }

        return OptimizedStepExecutor(executor_config)

    def create_training_pipeline(self) -> dict[str, Any]:
        """Create a complete optimized training pipeline."""
        self.logger.info("Creating complete optimized training pipeline")

        # Create components
        training_manager = self.create_enhanced_training_manager()
        memory_profiler = self.create_memory_profiler()
        leak_detector = self.create_memory_leak_detector(memory_profiler)
        step_executor = self.create_step_executor()

        return {
            "training_manager": training_manager,
            "memory_profiler": memory_profiler,
            "leak_detector": leak_detector,
            "step_executor": step_executor,
            "optimization_config": self.optimization_config,
        }


def create_optimized_training_system(config: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to create a complete optimized training system.

    Args:
        config: Training configuration

    Returns:
        Dictionary containing all optimized training components

    """
    factory = OptimizedTrainingFactory(config)
    return factory.create_training_pipeline()

