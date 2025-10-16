"""
Enhanced Step Optimizations for Training Pipeline.

This module provides comprehensive optimization utilities that can be applied
across all training pipeline steps for maximum performance and efficiency.
"""

import time
import logging
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union, Tuple
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')

class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    PERFORMANCE_FIRST = "performance_first"
    MEMORY_FIRST = "memory_first"

class WorkloadType(Enum):
    """Types of workloads for optimization selection."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    GPU_INTENSIVE = "gpu_intensive"
    MIXED = "mixed"
    LIGHTWEIGHT = "lightweight"

@dataclass
class OptimizationProfile:
    """Profile for optimization selection."""
    workload_type: WorkloadType
    data_size_mb: float
    expected_duration: float
    priority: str = "normal"  # low, normal, high, critical
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationDecision:
    """Decision made by the optimization selector."""
    strategy: OptimizationStrategy
    enabled_optimizations: List[str]
    disabled_optimizations: List[str]
    configuration: Dict[str, Any]
    reasoning: List[str]
    expected_improvement: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

class IntelligentOptimizationSelector:
    """Intelligent optimization selector based on workload analysis and learning."""

    def __init__(self, enable_learning: bool = True, history_size: int = 1000):
        """Initialize intelligent optimization selector.

        Args:
            enable_learning: Whether to learn from past optimization decisions
            history_size: Maximum number of historical decisions to keep
        """
        self.enable_learning = enable_learning
        self.history_size = history_size

        # Decision history
        self.decision_history: deque = deque(maxlen=history_size)
        self.performance_history: deque = deque(maxlen=history_size)

        # Optimization profiles and their performance
        self.optimization_profiles: Dict[str, Dict[str, Any]] = {}

        # Current system state
        self.system_state = self._get_system_state()

        # Background monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None

        self.logger = logging.getLogger(f"{__name__}.IntelligentOptimizationSelector")
        self.logger.info("🧠 Intelligent Optimization Selector initialized")

        if enable_learning:
            self._start_monitoring()

    def select_optimizations(self, profile: OptimizationProfile) -> OptimizationDecision:
        """Select optimal optimizations for a given workload profile.

        Args:
            profile: Workload profile to optimize for

        Returns:
            Optimization decision with strategy and configuration
        """
        start_time = time.time()

        # Analyze current system state
        current_state = self._get_system_state()

        # Determine optimal strategy based on profile and system state
        strategy = self._determine_optimal_strategy(profile, current_state)

        # Select specific optimizations to enable/disable
        enabled_opts, disabled_opts = self._select_specific_optimizations(profile, strategy, current_state)

        # Generate configuration
        config = self._generate_configuration(enabled_opts, profile, current_state)

        # Calculate expected improvements
        expected_improvement = self._calculate_expected_improvement(profile, enabled_opts, strategy)

        # Create reasoning
        reasoning = self._generate_reasoning(profile, strategy, enabled_opts, disabled_opts, current_state)

        decision = OptimizationDecision(
            strategy=strategy,
            enabled_optimizations=enabled_opts,
            disabled_optimizations=disabled_opts,
            configuration=config,
            reasoning=reasoning,
            expected_improvement=expected_improvement
        )

        # Record decision for learning
        if self.enable_learning:
            self.decision_history.append({
                'profile': profile,
                'decision': decision,
                'system_state': current_state,
                'selection_time': time.time() - start_time
            })

        self.logger.info(f"🎯 Selected {strategy.value} strategy with {len(enabled_opts)} optimizations enabled")
        return decision

    def _determine_optimal_strategy(self, profile: OptimizationProfile,
                                   system_state: Dict[str, Any]) -> OptimizationStrategy:
        """Determine the optimal optimization strategy."""

        # High-level decision based on workload and system characteristics
        workload_type = profile.workload_type
        data_size = profile.data_size_mb
        priority = profile.priority

        # CPU utilization affects strategy choice
        cpu_utilization = system_state.get('cpu_percent', 50)

        # Memory availability affects strategy choice
        memory_available_percent = system_state.get('memory_available_percent', 50)

        # GPU availability
        gpu_available = system_state.get('gpu_available', False)

        # Decision logic
        if priority == "critical":
            return OptimizationStrategy.AGGRESSIVE
        elif priority == "high":
            if cpu_utilization > 80:
                return OptimizationStrategy.MEMORY_FIRST
            else:
                return OptimizationStrategy.PERFORMANCE_FIRST
        elif data_size > 1000:  # Large datasets
            if memory_available_percent < 30:
                return OptimizationStrategy.CONSERVATIVE
            else:
                return OptimizationStrategy.BALANCED
        elif workload_type == WorkloadType.GPU_INTENSIVE and gpu_available:
            return OptimizationStrategy.PERFORMANCE_FIRST
        elif workload_type == WorkloadType.MEMORY_INTENSIVE:
            return OptimizationStrategy.MEMORY_FIRST
        elif cpu_utilization > 70:
            return OptimizationStrategy.CONSERVATIVE
        else:
            return OptimizationStrategy.BALANCED

    def _select_specific_optimizations(self, profile: OptimizationProfile,
                                     strategy: OptimizationStrategy,
                                     system_state: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Select which specific optimizations to enable or disable."""

        enabled = []
        disabled = []

        # Base optimizations (always considered)
        base_optimizations = {
            'memory_cleanup': True,
            'gc_tuning': True,
            'parallel_processing': True,
            'gpu_acceleration': system_state.get('gpu_available', False),
            'batch_optimization': True,
            'caching': True,
            'pipeline_optimization': True
        }

        # Strategy-specific modifications
        if strategy == OptimizationStrategy.AGGRESSIVE:
            # Enable all optimizations
            enabled.extend([k for k, v in base_optimizations.items() if v])
            # Add aggressive options
            enabled.extend(['memory_leak_detection', 'swap_management', 'adaptive_batching'])

        elif strategy == OptimizationStrategy.PERFORMANCE_FIRST:
            enabled.extend(['gpu_acceleration', 'parallel_processing', 'pipeline_optimization'])
            disabled.extend(['memory_leak_detection'])  # Can be expensive

        elif strategy == OptimizationStrategy.MEMORY_FIRST:
            enabled.extend(['memory_cleanup', 'gc_tuning', 'caching'])
            disabled.extend(['gpu_acceleration'])  # If memory is critical

        elif strategy == OptimizationStrategy.CONSERVATIVE:
            enabled.extend(['memory_cleanup', 'gc_tuning'])
            disabled.extend(['parallel_processing', 'gpu_acceleration', 'pipeline_optimization'])

        elif strategy == OptimizationStrategy.BALANCED:
            enabled.extend([k for k, v in base_optimizations.items() if v])
            # Balance by being selective
            if system_state.get('cpu_percent', 50) > 60:
                disabled.append('parallel_processing')

        # Workload-specific adjustments
        workload_type = profile.workload_type

        if workload_type == WorkloadType.CPU_INTENSIVE:
            enabled.append('parallel_processing')
            if strategy != OptimizationStrategy.CONSERVATIVE:
                enabled.append('cpu_optimization')
        elif workload_type == WorkloadType.MEMORY_INTENSIVE:
            enabled.extend(['memory_cleanup', 'caching', 'chunked_processing'])
        elif workload_type == WorkloadType.IO_INTENSIVE:
            enabled.extend(['async_io', 'caching'])
        elif workload_type == WorkloadType.GPU_INTENSIVE:
            enabled.append('gpu_acceleration')
        elif workload_type == WorkloadType.LIGHTWEIGHT:
            # Minimal optimizations for lightweight workloads
            enabled.extend(['gc_tuning', 'memory_cleanup'])
            disabled.extend(['parallel_processing', 'gpu_acceleration'])

        return enabled, disabled

    def _generate_configuration(self, enabled_optimizations: List[str],
                               profile: OptimizationProfile,
                               system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration for enabled optimizations."""

        config = {}

        # Configure each enabled optimization
        for opt in enabled_optimizations:
            if opt == 'memory_cleanup':
                config['memory_cleanup'] = {
                    'aggressive_gc': profile.data_size_mb > 500,
                    'cache_clearing': True
                }
            elif opt == 'gpu_acceleration':
                config['gpu_acceleration'] = {
                    'batch_size': min(1000, max(100, int(profile.data_size_mb / 10))),
                    'mixed_precision': system_state.get('gpu_memory_mb', 0) > 4000
                }
            elif opt == 'parallel_processing':
                cpu_count = system_state.get('cpu_count', 4)
                config['parallel_processing'] = {
                    'max_workers': min(cpu_count, 8),
                    'chunk_size': max(1000, int(profile.data_size_mb * 1000))
                }
            elif opt == 'batch_optimization':
                config['batch_optimization'] = {
                    'dynamic_sizing': True,
                    'learning_enabled': True
                }
            elif opt == 'caching':
                config['caching'] = {
                    'max_cache_size_mb': min(1000, profile.data_size_mb),
                    'cache_strategy': 'lru'
                }

        return config

    def _calculate_expected_improvement(self, profile: OptimizationProfile,
                                       enabled_opts: List[str],
                                       strategy: OptimizationStrategy) -> Dict[str, float]:
        """Calculate expected performance improvements."""

        improvements = {
            'speedup': 1.0,
            'memory_reduction': 0.0,
            'cpu_efficiency': 1.0
        }

        # Base improvements from strategy
        strategy_multipliers = {
            OptimizationStrategy.AGGRESSIVE: {'speedup': 1.8, 'memory_reduction': 0.3},
            OptimizationStrategy.PERFORMANCE_FIRST: {'speedup': 1.5, 'memory_reduction': 0.1},
            OptimizationStrategy.MEMORY_FIRST: {'speedup': 1.2, 'memory_reduction': 0.4},
            OptimizationStrategy.CONSERVATIVE: {'speedup': 1.1, 'memory_reduction': 0.2},
            OptimizationStrategy.BALANCED: {'speedup': 1.3, 'memory_reduction': 0.15}
        }

        if strategy in strategy_multipliers:
            multipliers = strategy_multipliers[strategy]
            improvements['speedup'] *= multipliers['speedup']
            improvements['memory_reduction'] = multipliers['memory_reduction']

        # Adjust based on enabled optimizations
        if 'gpu_acceleration' in enabled_opts:
            improvements['speedup'] *= 1.5
        if 'parallel_processing' in enabled_opts:
            improvements['speedup'] *= 1.3
        if 'memory_cleanup' in enabled_opts:
            improvements['memory_reduction'] += 0.2

        # Adjust based on workload type
        workload_adjustments = {
            WorkloadType.GPU_INTENSIVE: {'speedup': 1.4},
            WorkloadType.CPU_INTENSIVE: {'speedup': 1.3},
            WorkloadType.MEMORY_INTENSIVE: {'memory_reduction': 0.3}
        }

        if profile.workload_type in workload_adjustments:
            for key, value in workload_adjustments[profile.workload_type].items():
                improvements[key] *= value

        return improvements

    def _generate_reasoning(self, profile: OptimizationProfile, strategy: OptimizationStrategy,
                           enabled_opts: List[str], disabled_opts: List[str],
                           system_state: Dict[str, Any]) -> List[str]:
        """Generate reasoning for the optimization decision."""

        reasoning = []

        # Strategy reasoning
        reasoning.append(f"Selected {strategy.value} strategy based on {profile.workload_type.value} workload")

        # System state reasoning
        if system_state.get('cpu_percent', 0) > 80:
            reasoning.append("High CPU utilization detected - selected conservative optimizations")
        if system_state.get('memory_available_percent', 100) < 30:
            reasoning.append("Low memory availability - prioritized memory optimizations")

        # Workload reasoning
        if profile.data_size_mb > 1000:
            reasoning.append(f"Large dataset ({profile.data_size_mb:.0f}MB) - enabled chunked processing")
        if profile.priority == "critical":
            reasoning.append("Critical priority - enabled aggressive optimizations")

        # Specific optimization reasoning
        if 'gpu_acceleration' in enabled_opts and system_state.get('gpu_available'):
            reasoning.append("GPU available - enabled GPU acceleration")
        if 'parallel_processing' in enabled_opts:
            reasoning.append("CPU resources available - enabled parallel processing")

        return reasoning

    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for optimization decisions."""
        try:
            state = {}

            # CPU information
            state['cpu_count'] = psutil.cpu_count()
            state['cpu_percent'] = psutil.cpu_percent(interval=0.1)

            # Memory information
            memory = psutil.virtual_memory()
            state['memory_total_mb'] = memory.total / (1024 * 1024)
            state['memory_available_mb'] = memory.available / (1024 * 1024)
            state['memory_available_percent'] = (memory.available / memory.total) * 100

            # GPU information (if available)
            try:
                import torch
                state['gpu_available'] = torch.backends.mps.is_available() or torch.cuda.is_available()
                if torch.cuda.is_available():
                    state['gpu_memory_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                elif torch.backends.mps.is_available():
                    # MPS doesn't provide memory info easily
                    state['gpu_memory_mb'] = 8000  # Estimate for M1/M2
            except ImportError:
                state['gpu_available'] = False
                state['gpu_memory_mb'] = 0

            return state

        except Exception as e:
            self.logger.warning(f"Failed to get system state: {e}")
            return {
                'cpu_count': 4,
                'cpu_percent': 50,
                'memory_available_percent': 50,
                'gpu_available': False
            }

    def record_performance_result(self, profile: OptimizationProfile,
                                decision: OptimizationDecision,
                                actual_improvement: Dict[str, float],
                                execution_time: float):
        """Record actual performance results for learning."""

        if not self.enable_learning:
            return

        performance_record = {
            'profile': profile,
            'decision': decision,
            'actual_improvement': actual_improvement,
            'execution_time': execution_time,
            'timestamp': time.time(),
            'system_state': self._get_system_state()
        }

        self.performance_history.append(performance_record)

        # Update optimization profiles based on results
        self._update_profiles_from_performance(profile, decision, actual_improvement)

    def _update_profiles_from_performance(self, profile: OptimizationProfile,
                                        decision: OptimizationDecision,
                                        actual_improvement: Dict[str, float]):
        """Update optimization profiles based on performance results."""

        profile_key = f"{profile.workload_type.value}_{profile.data_size_mb:.0f}MB"

        if profile_key not in self.optimization_profiles:
            self.optimization_profiles[profile_key] = {
                'total_runs': 0,
                'avg_improvement': {},
                'best_strategy': decision.strategy.value,
                'best_improvement': actual_improvement
            }

        prof = self.optimization_profiles[profile_key]
        prof['total_runs'] += 1

        # Update average improvements
        for key, value in actual_improvement.items():
            if key not in prof['avg_improvement']:
                prof['avg_improvement'][key] = value
            else:
                # Exponential moving average
                alpha = 0.1
                prof['avg_improvement'][key] = (
                    prof['avg_improvement'][key] * (1 - alpha) + value * alpha
                )

        # Update best strategy if this was better
        if actual_improvement.get('speedup', 1.0) > prof['best_improvement'].get('speedup', 1.0):
            prof['best_strategy'] = decision.strategy.value
            prof['best_improvement'] = actual_improvement

    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop for system state."""
        while self.monitoring_active:
            try:
                # Update system state every 30 seconds
                self.system_state = self._get_system_state()
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            'total_decisions': len(self.decision_history),
            'total_performance_records': len(self.performance_history),
            'optimization_profiles': len(self.optimization_profiles),
            'learning_enabled': self.enable_learning,
            'current_system_state': self.system_state,
            'decision_history_summary': self._get_decision_summary(),
            'performance_history_summary': self._get_performance_summary()
        }

    def _get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of decision history."""
        if not self.decision_history:
            return {}

        strategies = {}
        for record in self.decision_history:
            strategy = record['decision'].strategy.value
            strategies[strategy] = strategies.get(strategy, 0) + 1

        return {
            'strategy_distribution': strategies,
            'average_selection_time': np.mean([
                r['selection_time'] for r in self.decision_history
            ])
        }

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance history."""
        if not self.performance_history:
            return {}

        speedups = [r['actual_improvement'].get('speedup', 1.0) for r in self.performance_history]
        memory_reductions = [r['actual_improvement'].get('memory_reduction', 0.0) for r in self.performance_history]

        return {
            'average_speedup': np.mean(speedups),
            'average_memory_reduction': np.mean(memory_reductions),
            'best_speedup': max(speedups),
            'best_memory_reduction': max(memory_reductions)
        }

    def save_profiles(self, filepath: str):
        """Save optimization profiles to file."""
        try:
            # Convert dataclasses to dictionaries for JSON serialization
            serializable_profiles = {}
            for key, profile in self.optimization_profiles.items():
                serializable_profiles[key] = {
                    'total_runs': profile['total_runs'],
                    'avg_improvement': profile['avg_improvement'],
                    'best_strategy': profile['best_strategy'],
                    'best_improvement': dict(profile['best_improvement'])
                }

            with open(filepath, 'w') as f:
                json.dump(serializable_profiles, f, indent=2)

            self.logger.info(f"💾 Saved optimization profiles to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save profiles: {e}")

    def load_profiles(self, filepath: str):
        """Load optimization profiles from file."""
        try:
            with open(filepath, 'r') as f:
                loaded_profiles = json.load(f)

            self.optimization_profiles = loaded_profiles
            self.logger.info(f"📂 Loaded optimization profiles from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}")

class StepOptimizationManager:
    """Manager for optimizing training pipeline steps."""

    def __init__(self, enable_intelligent_selection: bool = True):
        """Initialize step optimization manager."""
        self.optimizations_enabled = True
        self.performance_metrics = {}
        self.enable_intelligent_selection = enable_intelligent_selection
        self.logger = logger.getChild('StepOptimizationManager')

        # Initialize optimization components
        self._init_optimization_components()

        # Initialize intelligent optimization selector
        if self.enable_intelligent_selection:
            self.optimization_selector = IntelligentOptimizationSelector(
                enable_learning=True,
                history_size=1000
            )
        else:
            self.optimization_selector = None

    def _init_optimization_components(self):
        """Initialize all optimization components."""
        try:
            from .vectorized_processing_core import get_vectorized_processing_core
            from .hardware.m1_gpu_utils import get_m1_gpu_manager
            from .hardware.m1_memory_optimizer import get_m1_memory_optimizer
            from .hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

            self.vectorized_core = get_vectorized_processing_core()
            self.data_manager = None  # Fallback since optimized_data_manager doesn't exist
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()

            self.logger.info("🚀 All optimization components initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Some optimization components not available: {e}")
            self.vectorized_core = None
            self.data_manager = None
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    @contextmanager
    def optimized_execution_context(self, step_name: str = "unknown_step"):
        """Context manager for optimized step execution."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent if psutil else 0

        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint(f"step_{step_name}"):
                try:
                    yield self
                finally:
                    self._log_step_performance(step_name, start_time, start_memory)
        else:
            try:
                yield self
            finally:
                self._log_step_performance(step_name, start_time, start_memory)

    def _log_step_performance(self, step_name: str, start_time: float, start_memory: float):
        """Log step performance metrics."""
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent if psutil else 0

        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory

        self.logger.info(
            f"📊 Step '{step_name}' completed in {execution_time:.2f}s, "
            f"memory Δ: {memory_delta:+.1f}%"
        )

        # Store metrics
        self.performance_metrics[step_name] = {
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'timestamp': end_time
        }

    def optimize_dataframe_operations(self, df: pd.DataFrame,
                                    operation: str = "general") -> pd.DataFrame:
        """Optimize DataFrame for specific operations."""
        if not self.vectorized_core:
            return df

        with self.memory_optimizer.memory_checkpoint("dataframe_optimization") if self.memory_optimizer else nullcontext():
            if operation == "feature_engineering":
                # Add rolling features
                df = self.vectorized_core.vectorized_rolling_features(df)

                # Optimize data types
                df = self.vectorized_core.optimize_dataframe_for_processing(df)

            elif operation == "matrix_operations":
                # Optimize for matrix operations
                df = self.vectorized_core.optimize_dataframe_for_processing(df)

                # Convert to float32 for GPU compatibility
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].astype(np.float32)

            elif operation == "storage":
                # Optimize for storage
                if self.data_manager:
                    df = self.data_manager.optimize_dataframe_schema(df)

            return df

    def parallel_feature_processing(self, data: pd.DataFrame,
                                  feature_generators: List[Callable[[pd.DataFrame], pd.Series]],
                                  max_workers: Optional[int] = None) -> pd.DataFrame:
        """Parallel feature processing with optimization."""
        if not self.vectorized_core or not feature_generators:
            # Fallback to sequential processing
            for generator in feature_generators:
                feature_series = generator(data)
                data = pd.concat([data, feature_series], axis=1)
            return data

        return self.vectorized_core.parallel_feature_engineering(
            data, feature_generators, max_workers
        )

    def matrix_optimized_operations(self, matrices: List[np.ndarray],
                                  operation: str = "multiply") -> Union[np.ndarray, List[np.ndarray]]:
        """Matrix operations with GPU acceleration."""
        if not self.vectorized_core or not matrices:
            # Fallback to numpy operations
            if operation == "multiply" and len(matrices) >= 2:
                result = matrices[0]
                for matrix in matrices[1:]:
                    result = np.dot(result, matrix)
                return result
            return matrices[0] if matrices else None

        if len(matrices) == 2:
            return self.vectorized_core.gpu_accelerated_matrix_ops(
                matrices[0], matrices[1], operation
            )
        else:
            # Sequential operations
            result = matrices[0]
            for matrix in matrices[1:]:
                result = self.vectorized_core.gpu_accelerated_matrix_ops(
                    result, matrix, operation
                )
            return result

    def memory_efficient_groupby(self, df: pd.DataFrame,
                               group_cols: List[str],
                               agg_dict: Dict[str, str],
                               chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Memory-efficient groupby operations."""
        if not self.vectorized_core:
            return df.groupby(group_cols).agg(agg_dict)

        return self.vectorized_core.memory_efficient_groupby(
            df, group_cols, agg_dict, chunk_size
        )

    def optimized_data_storage(self, data: Union[pd.DataFrame, np.ndarray],
                             filename: str, **kwargs) -> str:
        """Optimized data storage."""
        if not self.data_manager:
            # Fallback to basic storage
            if isinstance(data, pd.DataFrame):
                path = f"data_cache/{filename}.parquet"
                data.to_parquet(path, **kwargs)
                return path
            else:
                path = f"data_cache/{filename}.npy"
                np.save(path, data)
                return path

        if isinstance(data, pd.DataFrame):
            return self.data_manager.save_dataframe_optimized(data, filename, **kwargs)
        elif isinstance(data, np.ndarray):
            return self.data_manager.save_numpy_array_optimized(data, filename)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'optimizations_enabled': self.optimizations_enabled,
            'performance_metrics': self.performance_metrics,
            'components_available': {
                'vectorized_core': self.vectorized_core is not None,
                'data_manager': self.data_manager is not None,
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            }
        }

        # Add component-specific stats
        if self.gpu_manager:
            stats['gpu_info'] = {
                'device': str(self.gpu_manager.device),
                'memory_gb': self.gpu_manager.memory_info.get('available_gb', 0)
            }

        if self.memory_optimizer:
            stats['memory_stats'] = self.memory_optimizer.get_memory_report()

        if self.cpu_optimizer:
            stats['cpu_stats'] = self.cpu_optimizer.get_cpu_usage_report()

        return stats

    def select_intelligent_optimizations(self, workload_profile: OptimizationProfile) -> OptimizationDecision:
        """Select intelligent optimizations based on workload profile."""
        if not self.enable_intelligent_selection or not self.optimization_selector:
            # Fallback to default optimization decision
            return OptimizationDecision(
                strategy=OptimizationStrategy.BALANCED,
                enabled_optimizations=['memory_cleanup', 'gc_tuning', 'parallel_processing'],
                disabled_optimizations=[],
                configuration={},
                reasoning=['Using default balanced optimizations'],
                expected_improvement={'speedup': 1.3, 'memory_reduction': 0.15}
            )

        return self.optimization_selector.select_optimizations(workload_profile)

    def record_optimization_performance(self, profile: OptimizationProfile,
                                      decision: OptimizationDecision,
                                      actual_improvement: Dict[str, float],
                                      execution_time: float):
        """Record performance results for optimization learning."""
        if self.optimization_selector:
            self.optimization_selector.record_performance_result(
                profile, decision, actual_improvement, execution_time
            )

    def get_intelligent_optimization_stats(self) -> Dict[str, Any]:
        """Get intelligent optimization statistics."""
        if self.optimization_selector:
            return self.optimization_selector.get_optimization_stats()
        return {'intelligent_selection': False}

# Decorator for optimizing step functions
def optimized_step(operation_type: str = "general", enable_gpu: bool = True,
                  enable_parallel: bool = True, memory_efficient: bool = True):
    """Decorator for optimizing training pipeline steps."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get optimization manager
            try:
                from .enhanced_step_optimizations import get_step_optimization_manager
                opt_manager = get_step_optimization_manager()
            except ImportError:
                # Fallback without optimizations
                return func(*args, **kwargs)

            step_name = func.__name__

            with opt_manager.optimized_execution_context(step_name):
                # Pre-optimization
                if memory_efficient and opt_manager.memory_optimizer:
                    opt_manager.memory_optimizer.optimize_memory()

                # Execute with optimizations
                result = func(*args, **kwargs)

                # Post-optimization cleanup
                if memory_efficient and opt_manager.memory_optimizer:
                    opt_manager.memory_optimizer.optimize_memory()

                return result

        return wrapper
    return decorator

# Convenience functions
def get_step_optimization_manager() -> StepOptimizationManager:
    """Get global step optimization manager instance."""
    if not hasattr(get_step_optimization_manager, '_instance'):
        get_step_optimization_manager._instance = StepOptimizationManager()
    return get_step_optimization_manager._instance

def optimize_dataframe(df: pd.DataFrame, operation: str = "general") -> pd.DataFrame:
    """Optimize DataFrame for specific operations."""
    manager = get_step_optimization_manager()
    return manager.optimize_dataframe_operations(df, operation)

def parallel_features(data: pd.DataFrame,
                     generators: List[Callable[[pd.DataFrame], pd.Series]]) -> pd.DataFrame:
    """Generate features in parallel."""
    manager = get_step_optimization_manager()
    return manager.parallel_feature_processing(data, generators)

def matrix_ops(matrices: List[np.ndarray], operation: str = "multiply") -> np.ndarray:
    """Perform optimized matrix operations."""
    manager = get_step_optimization_manager()
    return manager.matrix_optimized_operations(matrices, operation)

def efficient_groupby(df: pd.DataFrame, group_cols: List[str],
                     agg_dict: Dict[str, str]) -> pd.DataFrame:
    """Memory-efficient groupby."""
    manager = get_step_optimization_manager()
    return manager.memory_efficient_groupby(df, group_cols, agg_dict)

def save_optimized(data: Union[pd.DataFrame, np.ndarray], filename: str) -> str:
    """Save data with optimizations."""
    manager = get_step_optimization_manager()
    return manager.optimized_data_storage(data, filename)

def create_optimization_profile(workload_type: WorkloadType, data_size_mb: float,
                              expected_duration: float = 60.0, priority: str = "normal") -> OptimizationProfile:
    """Create an optimization profile for intelligent selection."""
    return OptimizationProfile(
        workload_type=workload_type,
        data_size_mb=data_size_mb,
        expected_duration=expected_duration,
        priority=priority
    )

def select_intelligent_optimizations(profile: OptimizationProfile) -> OptimizationDecision:
    """Select intelligent optimizations based on workload profile."""
    manager = get_step_optimization_manager()
    return manager.select_intelligent_optimizations(profile)

def record_optimization_performance(profile: OptimizationProfile,
                                  decision: OptimizationDecision,
                                  actual_improvement: Dict[str, float],
                                  execution_time: float):
    """Record performance results for optimization learning."""
    manager = get_step_optimization_manager()
    manager.record_optimization_performance(profile, decision, actual_improvement, execution_time)

def get_intelligent_optimization_stats() -> Dict[str, Any]:
    """Get intelligent optimization statistics."""
    manager = get_step_optimization_manager()
    return manager.get_intelligent_optimization_stats()

# Import psutil for memory monitoring
try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from contextlib import nullcontext
except ImportError:
    # Python < 3.7 compatibility
    class nullcontext:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
