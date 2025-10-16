"""
Logging utilities for TAS

Logging and monitoring utilities for tree architecture search including:
- Tree-specific loggers
- Performance monitoring
- Search progress tracking
- Visualization integration
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TreeLogger:
    """Tree-specific logger for TAS operations."""

    def __init__(self, name: str = "TAS.TreeLogger"):
        self.logger = logging.getLogger(name)
        self.start_time = time.time()

    def log_tree_operation(self, operation: str, details: Dict[str, Any]):
        """Log tree operation with details."""
        self.logger.info(f"🌲 Tree Operation: {operation}")
        for key, value in details.items():
            self.logger.debug(f"  {key}: {value}")

    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.logger.info("📊 Performance Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

    def log_search_progress(self, iteration: int, best_score: float, population_size: int):
        """Log search progress."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"🔍 Iteration {iteration}: Best Score={best_score:.4f}, Population={population_size}, Elapsed={elapsed:.2f}s")

@dataclass
class TreeSearchLogger(TreeLogger):
    """Logger for tree search operations."""

    def log_search_start(self, config: Dict[str, Any]):
        """Log search initialization."""
        self.logger.info("🚀 Tree Search Started")
        self.logger.info(f"  Population Size: {config.get('population_size', 'N/A')}")
        self.logger.info(f"  Max Generations: {config.get('generations', 'N/A')}")
        self.logger.info(f"  Mutation Rate: {config.get('mutation_rate', 'N/A')}")

    def log_generation_complete(self, generation: int, best_fitness: float, avg_fitness: float):
        """Log completion of a generation."""
        self.logger.info(f"✅ Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

@dataclass
class TreePerformanceLogger(TreeLogger):
    """Logger for performance monitoring."""

    def log_hardware_metrics(self, metrics: Dict[str, Any]):
        """Log hardware performance metrics."""
        self.logger.info("⚡ Hardware Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")

    def log_memory_usage(self, used_mb: float, total_mb: float):
        """Log memory usage."""
        percentage = (used_mb / total_mb) * 100
        self.logger.info(f"🧠 Memory Usage: {used_mb:.1f}MB / {total_mb:.1f}MB ({percentage:.1f}%)")

    def log_execution_time(self, operation: str, duration: float):
        """Log execution time for an operation."""
        self.logger.info(f"⏱️ {operation}: {duration:.4f}s")
