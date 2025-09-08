"""
Step05 Adaptive Chunk Sizing Module

This module provides intelligent adaptive chunk sizing for Step05 processing,
dynamically adjusting chunk sizes based on available memory, data complexity,
processing speed, and historical performance patterns.
"""

import pandas as pd
import numpy as np
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates
from src.utils.common_operations import safe_mean, safe_std, safe_float, safe_int
from src.utils.math_validation import safe_divide, validate_range

logger = system_logger.getChild('AdaptiveChunkSizer')


@dataclass
class ChunkPerformanceMetrics:
    """Performance metrics for chunk processing."""
    chunk_size: int
    processing_time: float
    memory_usage_mb: float
    throughput_rows_per_sec: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data_complexity_score: float = 0.0
    success: bool = True


@dataclass
class AdaptiveChunkConfig:
    """Configuration for adaptive chunk sizing."""
    base_chunk_size: int = 10000
    min_chunk_size: int = 1000
    max_chunk_size: int = 100000
    memory_safety_factor: float = 0.7  # Use only 70% of available memory
    performance_history_size: int = 50
    adaptation_interval_seconds: int = 30
    enable_memory_adaptation: bool = True
    enable_complexity_adaptation: bool = True
    enable_performance_learning: bool = True
    learning_rate: float = 0.1


class AdaptiveChunkSizer:
    """
    Intelligent chunk size adapter that learns from processing history
    and adapts to system conditions and data characteristics.
    """

    def __init__(self, config: Optional[AdaptiveChunkConfig] = None):
        self.config = config or AdaptiveChunkConfig()
        self.logger = logger

        # Performance history for learning
        self.performance_history: deque[ChunkPerformanceMetrics] = deque(maxlen=self.config.performance_history_size)

        # Current system state
        self.current_chunk_size = self.config.base_chunk_size
        self.last_adaptation_time = datetime.now()

        # Learning state
        self.optimal_sizes_by_complexity: Dict[str, int] = {}
        self.performance_trends: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.RLock()

        self.logger.info("🚀 Initializing Adaptive Chunk Sizer")
        self.logger.info(f"📊 Base chunk size: {self.config.base_chunk_size:,}")
        self.logger.info(f"📏 Size range: {self.config.min_chunk_size:,} - {self.config.max_chunk_size:,}")
        self.logger.info(f"🧠 Performance learning: {'Enabled' if self.config.enable_performance_learning else 'Disabled'}")

    def get_optimal_chunk_size(self, data_sample: Optional[pd.DataFrame] = None,
                             available_memory_gb: Optional[float] = None,
                             system_load: Optional[float] = None) -> int:
        """
        Calculate optimal chunk size based on current conditions and learning history.

        Args:
            data_sample: Sample of data to analyze complexity
            available_memory_gb: Available system memory
            system_load: Current system load (0-1)

        Returns:
            Optimal chunk size
        """
        with self._lock:
            try:
                base_size = self.current_chunk_size

                # Factor 1: Memory-based adaptation
                if self.config.enable_memory_adaptation:
                    memory_factor = self._calculate_memory_factor(available_memory_gb)
                    base_size = int(base_size * memory_factor)

                # Factor 2: Data complexity adaptation
                if self.config.enable_complexity_adaptation and data_sample is not None:
                    complexity_factor = self._calculate_complexity_factor(data_sample)
                    base_size = int(base_size * complexity_factor)

                # Factor 3: System load adaptation
                if system_load is not None:
                    load_factor = self._calculate_load_factor(system_load)
                    base_size = int(base_size * load_factor)

                # Factor 4: Historical performance learning
                if self.config.enable_performance_learning and len(self.performance_history) > 5:
                    learning_factor = self._calculate_learning_factor()
                    base_size = int(base_size * learning_factor)

                # Apply bounds
                optimal_size = max(self.config.min_chunk_size,
                                 min(self.config.max_chunk_size, base_size))

                # Periodic adaptation check
                if self._should_adapt():
                    self._perform_periodic_adaptation()

                self.logger.debug(f"🎯 Optimal chunk size: {optimal_size:,} "
                                f"(base: {self.current_chunk_size:,})")

                return optimal_size

            except Exception as e:
                self.logger.warning(f"⚠️ Error calculating optimal chunk size: {e}")
                return self.config.base_chunk_size

    def _calculate_memory_factor(self, available_memory_gb: Optional[float] = None) -> float:
        """Calculate memory-based adjustment factor."""
        try:
            if available_memory_gb is None:
                # Get current memory info
                memory_info = psutil.virtual_memory()
                available_memory_gb = memory_info.available / (1024**3)

            # Target memory usage per chunk (rough estimate)
            estimated_mb_per_1000_rows = 50  # Conservative estimate
            target_chunk_mb = (self.current_chunk_size / 1000) * estimated_mb_per_1000_rows

            # Calculate how much memory we can safely use
            safe_memory_mb = available_memory_gb * 1024 * self.config.memory_safety_factor
            memory_factor = safe_memory_mb / target_chunk_mb

            # Bound the factor
            memory_factor = max(0.1, min(2.0, memory_factor))

            self.logger.debug(f"💾 Memory factor: {memory_factor:.2f} "
                            f"(available: {available_memory_gb:.1f}GB)")

            return memory_factor

        except Exception as e:
            self.logger.warning(f"⚠️ Memory factor calculation failed: {e}")
            return 1.0

    def _calculate_complexity_factor(self, data_sample: pd.DataFrame) -> float:
        """Calculate data complexity-based adjustment factor."""
        try:
            if len(data_sample) == 0:
                return 1.0

            # Calculate complexity score based on various factors
            complexity_score = 0.0

            # Factor 1: Column count
            column_factor = len(data_sample.columns) / 10.0  # Normalize to ~10 columns
            complexity_score += column_factor * 0.3

            # Factor 2: Data type diversity
            dtypes = data_sample.dtypes
            numeric_count = sum(pd.api.types.is_numeric_dtype(dt) for dt in dtypes)
            dtype_diversity = numeric_count / len(dtypes)
            complexity_score += (1 - dtype_diversity) * 0.2

            # Factor 3: Sparsity (null values)
            null_percentage = data_sample.isnull().mean().mean()
            sparsity_factor = 1 + (null_percentage * 0.5)  # More nulls = more complex
            complexity_score += sparsity_factor * 0.2

            # Factor 4: Value range diversity
            numeric_cols = data_sample.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Calculate coefficient of variation for numeric columns
                cv_scores = []
                for col in numeric_cols[:5]:  # Sample first 5 numeric columns
                    try:
                        mean_val = data_sample[col].mean()
                        std_val = data_sample[col].std()
                        if mean_val != 0:
                            cv = abs(std_val / mean_val)
                            cv_scores.append(min(cv, 10.0))  # Cap at 10
                    except:
                        pass

                if cv_scores:
                    avg_cv = np.mean(cv_scores)
                    complexity_score += min(avg_cv / 2.0, 1.0) * 0.3

            # Normalize complexity score and convert to factor
            complexity_score = min(complexity_score, 3.0)  # Cap at 3.0

            # Convert complexity to size factor (higher complexity = smaller chunks)
            complexity_factor = 1.0 / (1.0 + complexity_score * 0.5)
            complexity_factor = max(0.3, min(1.5, complexity_factor))

            # Cache optimal size for this complexity level
            complexity_key = f"{complexity_score:.2f}"
            self.optimal_sizes_by_complexity[complexity_key] = self.current_chunk_size

            self.logger.debug(f"🧩 Complexity factor: {complexity_factor:.2f} "
                            f"(score: {complexity_score:.2f})")

            return complexity_factor

        except Exception as e:
            self.logger.warning(f"⚠️ Complexity factor calculation failed: {e}")
            return 1.0

    def _calculate_load_factor(self, system_load: float) -> float:
        """Calculate system load-based adjustment factor."""
        try:
            # Higher load = smaller chunks to reduce system pressure
            load_factor = 1.0 - (system_load * 0.5)  # Reduce by up to 50% at full load
            load_factor = max(0.5, load_factor)  # Minimum 50% of base size

            self.logger.debug(f"⚡ Load factor: {load_factor:.2f} "
                            f"(system load: {system_load:.2f})")

            return load_factor

        except Exception as e:
            self.logger.warning(f"⚠️ Load factor calculation failed: {e}")
            return 1.0

    def _calculate_learning_factor(self) -> float:
        """Calculate learning-based adjustment factor from historical performance."""
        try:
            if len(self.performance_history) < 5:
                return 1.0

            # Analyze recent performance trends
            recent_metrics = list(self.performance_history)[-10:]  # Last 10 measurements

            # Calculate throughput trend
            throughputs = [m.throughput_rows_per_sec for m in recent_metrics]
            if len(throughputs) >= 2:
                throughput_trend = throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 1.0

                # If throughput is improving, we can try larger chunks
                # If throughput is declining, we should reduce chunk size
                learning_factor = 1.0 + (throughput_trend - 1.0) * self.config.learning_rate
                learning_factor = max(0.8, min(1.2, learning_factor))

                self.logger.debug(f"🧠 Learning factor: {learning_factor:.2f} "
                                f"(throughput trend: {throughput_trend:.2f})")

                return learning_factor

            return 1.0

        except Exception as e:
            self.logger.warning(f"⚠️ Learning factor calculation failed: {e}")
            return 1.0

    def record_chunk_performance(self, chunk_size: int, processing_time: float,
                               memory_usage_mb: float, success: bool = True,
                               data_complexity_score: float = 0.0) -> None:
        """
        Record performance metrics for a processed chunk to enable learning.

        Args:
            chunk_size: Size of the chunk that was processed
            processing_time: Time taken to process the chunk
            memory_usage_mb: Memory usage during processing
            success: Whether processing was successful
            data_complexity_score: Complexity score of the data
        """
        with self._lock:
            try:
                # Calculate throughput
                throughput = chunk_size / processing_time if processing_time > 0 else 0

                # Get CPU usage (rough estimate)
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Create metrics record
                metrics = ChunkPerformanceMetrics(
                    chunk_size=chunk_size,
                    processing_time=processing_time,
                    memory_usage_mb=memory_usage_mb,
                    throughput_rows_per_sec=throughput,
                    cpu_usage_percent=cpu_percent,
                    data_complexity_score=data_complexity_score,
                    success=success
                )

                # Add to history
                self.performance_history.append(metrics)

                # Update current chunk size based on success/failure
                if success:
                    # Slight increase for successful chunks
                    self.current_chunk_size = int(self.current_chunk_size * 1.05)
                else:
                    # Decrease for failed chunks
                    self.current_chunk_size = int(self.current_chunk_size * 0.8)

                # Apply bounds
                self.current_chunk_size = max(self.config.min_chunk_size,
                                            min(self.config.max_chunk_size, self.current_chunk_size))

                self.logger.debug(f"📊 Recorded chunk performance: size={chunk_size:,}, "
                                f"time={processing_time:.3f}s, throughput={throughput:.0f} rows/s")

            except Exception as e:
                self.logger.warning(f"⚠️ Error recording chunk performance: {e}")

    def _should_adapt(self) -> bool:
        """Check if periodic adaptation should be performed."""
        time_since_adaptation = datetime.now() - self.last_adaptation_time
        return time_since_adaptation.total_seconds() >= self.config.adaptation_interval_seconds

    def _perform_periodic_adaptation(self) -> None:
        """Perform periodic adaptation based on accumulated performance data."""
        try:
            if len(self.performance_history) < 10:
                return

            # Analyze performance patterns
            recent_metrics = list(self.performance_history)[-20:]

            # Calculate optimal chunk size based on throughput
            successful_metrics = [m for m in recent_metrics if m.success]
            if len(successful_metrics) >= 5:
                # Group by chunk size ranges and find best performing
                size_ranges = {}
                for metric in successful_metrics:
                    size_range = self._get_size_range(metric.chunk_size)
                    if size_range not in size_ranges:
                        size_ranges[size_range] = []
                    size_ranges[size_range].append(metric.throughput_rows_per_sec)

                # Find range with best average throughput
                best_range = None
                best_throughput = 0

                for size_range, throughputs in size_ranges.items():
                    avg_throughput = np.mean(throughputs)
                    if avg_throughput > best_throughput:
                        best_throughput = avg_throughput
                        best_range = size_range

                if best_range:
                    optimal_size = self._range_to_size(best_range)
                    # Gradually move toward optimal
                    adjustment_factor = 1.0 + (optimal_size / self.current_chunk_size - 1.0) * 0.1
                    self.current_chunk_size = int(self.current_chunk_size * adjustment_factor)

                    # Apply bounds
                    self.current_chunk_size = max(self.config.min_chunk_size,
                                                min(self.config.max_chunk_size, self.current_chunk_size))

                    self.logger.info(f"🔄 Periodic adaptation: new chunk size {self.current_chunk_size:,} "
                                   f"(optimal range: {best_range})")

            self.last_adaptation_time = datetime.now()

        except Exception as e:
            self.logger.warning(f"⚠️ Periodic adaptation failed: {e}")

    def _get_size_range(self, size: int) -> str:
        """Convert chunk size to size range category."""
        if size < 5000:
            return "small"
        elif size < 20000:
            return "medium"
        elif size < 50000:
            return "large"
        else:
            return "xlarge"

    def _range_to_size(self, range_name: str) -> int:
        """Convert size range to representative chunk size."""
        range_sizes = {
            "small": 2500,
            "medium": 10000,
            "large": 30000,
            "xlarge": 75000
        }
        return range_sizes.get(range_name, self.config.base_chunk_size)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics and adaptation history."""
        with self._lock:
            try:
                if not self.performance_history:
                    return {"status": "no_data"}

                metrics = list(self.performance_history)

                return {
                    "status": "active",
                    "total_measurements": len(metrics),
                    "current_chunk_size": self.current_chunk_size,
                    "avg_throughput": np.mean([m.throughput_rows_per_sec for m in metrics]),
                    "avg_processing_time": np.mean([m.processing_time for m in metrics]),
                    "avg_memory_usage_mb": np.mean([m.memory_usage_mb for m in metrics]),
                    "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                    "performance_trends": dict(self.performance_trends),
                    "optimal_sizes_by_complexity": dict(self.optimal_sizes_by_complexity),
                    "last_adaptation": self.last_adaptation_time.isoformat()
                }

            except Exception as e:
                self.logger.error(f"❌ Error generating performance summary: {e}")
                return {"status": "error", "error": str(e)}

    def reset_adaptation(self) -> None:
        """Reset adaptation state to start fresh learning."""
        with self._lock:
            self.performance_history.clear()
            self.optimal_sizes_by_complexity.clear()
            self.performance_trends.clear()
            self.current_chunk_size = self.config.base_chunk_size
            self.last_adaptation_time = datetime.now()

            self.logger.info("🔄 Adaptation state reset")
