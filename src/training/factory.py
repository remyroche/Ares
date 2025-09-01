# src/training/factory.py

"""Factory for creating optimized training components based on configuration."""

import os
from typing import Any

from src.config.computational_optimization_config import get_optimization_config
from src.training.enhanced_training_manager_optimized import (
    EnhancedTrainingManagerOptimized
)
from src.training.memory_profiler import MemoryLeakDetector, MemoryProfiler
from src.training.steps.optimized_step_executor import OptimizedStepExecutor
from src.utils.logger import system_logger


class OptimizedTrainingFactory:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimizedtrainingfactory initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OptimizedTrainingFactory."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpass"""Factory for creating optimized training components."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.optimization_config = get_optimization_config(
            config.get("computational_optimization", {}),
        )
        self.logger = system_logger.getChild("OptimizedTrainingFactory")

    def create_enhanced_training_manager(...) -> ...:
    """..."""
    passself.logger.info("Creating Enhanced Training Manager with optimizations")

        # Merge optimization config with main config
        enhanced_config = self.config.copy()
        enhanced_config["computational_optimization"] = self.optimization_config

        return EnhancedTrainingManagerOptimized(enhanced_config)

    def create_memory_profiler(...) -> ...:
    pass"""..."""
    passmonitoring_config = self.optimization_config.get("monitoring", {})

        if enable_continuous_monitoring is None:
    passenable_continuous_monitoring = monitoring_config.get(
                "continuous_monitoring",
                True
            )

        enable_tracemalloc = monitoring_config.get("memory_leak_detection", True)

        self.logger.info(
            f"Creating Memory Profiler (continuous: {enable_continuous_monitoring})",
        )

        return MemoryProfiler(
            enable_tracemalloc=enable_tracemalloc,
            enable_continuous_monitoring=enable_continuous_monitoring
        )

    def create_memory_leak_detector(...) -> ...:
    """..."""
    passself.logger.info("Creating Memory Leak Detector")
        return MemoryLeakDetector(profiler)

    def create_step_executor(...) -> ...:
    """..."""
    passself.logger.info("Creating Optimized Step Executor")

        executor_config = {
            "parallel_execution": self.optimization_config["parallelization"]["enabled"],
            "max_workers": self.optimization_config["parallelization"]["max_workers"],
            "enable_caching": self.optimization_config["caching"]["enabled"],
            "enable_memory_optimization": self.optimization_config["memory_management"]["enabled"],
            "memory_threshold": self.optimization_config["memory_management"]["memory_threshold"],
        }

        return OptimizedStepExecutor(executor_config)

    def create_training_pipeline(...) -> ...:
    """..."""
    passself.logger.info("Creating complete optimized training pipeline")

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
            "optimization_config": self.optimization_config
        }

    def get_optimization_summary(...) -> ...:
    """..."""
    passsummary = {
            "optimizations_enabled": {},
            "performance_expectations": {},
            "configuration": {},
        }

        # Check which optimizations are enabled
        for optimization_type in [
            "caching",
            "parallelization",
            "early_stopping",
            "memory_management",
            "data_streaming",
            "adaptive_sampling",
        ]:
    passsummary["optimizations_enabled"][optimization_type] = (
                self.optimization_config[optimization_type]["enabled"]
            )

        # Add performance expectations
        from src.config.computational_optimization_config import (
            get_performance_expectations
        )

        summary["performance_expectations"] = get_performance_expectations()

        # Add key configuration values
        summary["configuration"] = {
            "max_workers": self.optimization_config["parallelization"]["max_workers"],
            "memory_threshold": self.optimization_config["memory_management"]["memory_threshold"],
            "cache_size": self.optimization_config["caching"]["max_cache_size"],
            "monitoring_interval": self.optimization_config["monitoring"]["monitoring_interval"],
        }

        return summary


def create_optimized_training_system(...) -> ...:
    """..."""
    passfactory = OptimizedTrainingFactory(config)
    return factory.create_training_pipeline()


def get_optimization_recommendations(...) -> ...:
    """..."""
    passrecommendations = {
        "memory_optimizations": [],
        "parallelization_optimizations": [],
        "caching_optimizations": [],
        "general_optimizations": [],
    }

    # Check system resources
    cpu_count = os.cpu_count()
    import psutil

    memory_gb = psutil.virtual_memory().total / (1024**3)

    # Memory recommendations
    if memory_gb < 8:
    passrecommendations["memory_optimizations"].extend(
            [
                "Enable aggressive memory management due to limited RAM",
                "Reduce chunk sizes for data processing",
                "Enable data streaming for large datasets",
            ],
        )
    elif memory_gb > 32:
    passpasspassrecommendations["memory_optimizations"].extend(
            [
                "Can afford larger cache sizes",
                "Enable memory profiling without performance concerns",
            ],
        )

    # Parallelization recommendations
    if cpu_count >= 8:
    passrecommendations["parallelization_optimizations"].extend(
            [
                f"Enable parallel processing with up to {min(cpu_count, 16)} workers",
                "Enable parallel backtesting and feature engineering",
            ],
        )
    else:
    passpassrecommendations["parallelization_optimizations"].extend(
            [
                "Limited CPU cores - use conservative parallelization",
                "Focus on memory optimizations over parallelization",
            ],
        )

    # Caching recommendations
    training_config = config.get("training", {})
    if training_config.get("n_trials", 100) > 500:
    passrecommendations["caching_optimizations"].extend(
            [
                "Enable aggressive caching for large optimization runs",
                "Use surrogate models to reduce expensive evaluations",
            ],
        )

    # General recommendations
    recommendations["general_optimizations"].extend(
        [
            "Enable early stopping for faster iteration",
            "Use Parquet format for data storage",
            "Enable adaptive sampling for better parameter exploration",
            "Monitor memory usage continuously during long training runs",
        ],
    )

    return recommendations
