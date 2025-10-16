"""
Hardware Service for NAS-TAS Clustering.

This module provides hardware abstraction and optimization services,
managing GPU/Memory/M1 optimization and providing accelerated operations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import platform
import psutil

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    is_m1_available, is_mps_available
)
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power
)

from ..shared_utils import get_logger


@dataclass
class HardwareCapabilities:
    """Hardware capabilities and configuration."""
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0
    has_m1: bool = False
    m1_memory_gb: float = 0.0
    cpu_cores: int = 1
    total_memory_gb: float = 0.0
    optimal_batch_size: int = 1000
    recommended_workers: int = 1


@dataclass
class HardwareOptimizationResult:
    """Result from hardware optimization."""
    device_selected: str
    optimization_applied: Dict[str, Any]
    performance_gains: Dict[str, float]
    memory_usage: Dict[str, float]
    execution_time: float


class HardwareService:
    """
    Hardware service that abstracts hardware operations (CPU vs GPU).

    Responsibilities:
    - Abstract hardware ops (CPU vs GPU)
    - Manage GPUManager, MemoryManager, M1Optimizer
    - Provide methods like select_device(), optimize_memory(), accelerate_neighbors()
    """

    def __init__(self, verbose: bool = True):
        """Initialize the hardware service."""
        self.verbose = verbose
        self.logger = get_logger('HardwareService')

        # Hardware detection and capabilities
        tprint("🔍 Initializing Hardware Service", "INFO")
        self.capabilities = self._detect_hardware_capabilities()

        # Initialize hardware managers using common operations
        tprint("⚙️ Initializing hardware managers", "INFO")
        self.gpu_manager = get_m1_gpu_manager() if is_m1_available() else None
        self.memory_manager = get_m1_memory_optimizer()
        self.m1_optimizer = get_m1_cpu_optimizer()

        # Import hardware optimization modules (for local hardware components)
        try:
            from .m1_optimizer import get_m1_optimizer
            from .memory_manager import get_memory_manager
            from .gpu_manager import get_gpu_manager
            self._hardware_modules_available = True
        except ImportError:
            self._hardware_modules_available = False

        # Performance tracking
        tprint(f"✅ Hardware Service initialized: M1={self.capabilities.has_m1}, GPU={self.capabilities.has_gpu}", "SUCCESS")
        self.optimization_history = []
        self.performance_metrics = {
            "total_optimizations": 0,
            "gpu_accelerations": 0,
            "memory_optimizations": 0,
            "m1_optimizations": 0,
            "average_speedup": 1.0,
            "total_memory_saved_gb": 0.0
        }

    def _detect_hardware_capabilities(self) -> HardwareCapabilities:
        """Detect available hardware capabilities."""
        try:
            tprint("🔍 Detecting hardware capabilities", "INFO")

            capabilities = HardwareCapabilities()

            # Detect CPU cores with validation
            cpu_cores = psutil.cpu_count(logical=True) or 1
            capabilities.cpu_cores = validate_finite(cpu_cores, "cpu_cores")

            # Detect total memory with safe calculation
            memory_bytes = psutil.virtual_memory().total
            memory_gb = safe_divide(memory_bytes, (1024**3), 1.0)
            capabilities.total_memory_gb = validate_finite(memory_gb, "total_memory_gb")

            # Detect platform (for M1 detection)
            system = platform.system().lower()
            machine = platform.machine().lower()

            if system == "darwin" and ("arm" in machine or "m1" in machine or "m2" in machine):
                capabilities.has_m1 = True
                capabilities.m1_memory_gb = memory_gb
                tprint(f"🍎 M1/M2 chip detected with {memory_gb:.1f}GB memory", "SUCCESS")

            # Try to detect GPU using common operations utilities
            try:
                if is_m1_available():
                    capabilities.has_gpu = True
                    capabilities.gpu_memory_gb = memory_gb  # M1 unified memory
                    tprint(f"🖥️ M1 GPU detected with {capabilities.gpu_memory_gb:.1f}GB memory", "SUCCESS")
                else:
                    # Try external GPU detection
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            capabilities.has_gpu = True
                            gpu_memory_mb = gpus[0].memoryTotal
                            capabilities.gpu_memory_gb = safe_divide(gpu_memory_mb, 1024, 0.0)
                            tprint(f"🖥️ External GPU detected: {gpus[0].name} with {capabilities.gpu_memory_gb:.1f}GB memory", "SUCCESS")
                    except ImportError:
                        tprint("📦 GPU detection libraries not available", "INFO")
                    except Exception as e:
                        tprint(f"⚠️ GPU detection failed: {e}", "WARNING")
            except Exception as e:
                tprint(f"⚠️ GPU detection error: {e}", "WARNING")

            # Calculate optimal batch size based on memory with safe math
            if capabilities.has_gpu:
                # GPU can handle larger batches
                capabilities.optimal_batch_size = min(5000, int(memory_gb * 100))
            else:
                # CPU is more conservative
                capabilities.optimal_batch_size = min(2000, int(memory_gb * 50))

            # Calculate recommended workers with validation
            capabilities.recommended_workers = min(capabilities.cpu_cores, 4)  # Cap at 4 workers

            tprint(f"⚙️ Hardware capabilities: CPU={capabilities.cpu_cores} cores, "
                  f"Memory={memory_gb:.1f}GB, Batch={capabilities.optimal_batch_size}", "SUCCESS")

            return capabilities

        except Exception as e:
            tprint(f"❌ Hardware detection failed: {e}", "ERROR")
            # Return minimal capabilities as fallback
            return HardwareCapabilities(cpu_cores=1, total_memory_gb=1.0, optimal_batch_size=100)

    def select_device(self, workload_type: str = "clustering", data_size: int = None) -> str:
        """
        Select optimal device for the given workload.

        Args:
            workload_type: Type of workload ("clustering", "feature_processing", "optimization")
            data_size: Size of data to process

        Returns:
            Selected device name ("cpu", "gpu", "m1")
        """
        try:
            tprint(f"🎯 Selecting device for {workload_type} workload", "INFO")

            # Default to CPU
            selected_device = "cpu"

            # GPU selection logic
            if self.capabilities.has_gpu and data_size:
                # Use GPU for large datasets and compute-intensive tasks
                if workload_type in ["clustering", "optimization"] and data_size > 1000:
                    selected_device = "gpu"
                    self.performance_metrics["gpu_accelerations"] += 1
                    tprint(f"🚀 Selected GPU for {workload_type} (data_size={data_size})", "SUCCESS")

            # M1 selection logic (for matrix operations)
            if self.capabilities.has_m1 and workload_type in ["feature_processing", "matrix_ops"]:
                selected_device = "m1"
                self.performance_metrics["m1_optimizations"] += 1
                tprint(f"🚀 Selected M1 for {workload_type}", "SUCCESS")

            tprint(f"✅ Selected device: {selected_device}", "SUCCESS")
            return selected_device

        except Exception as e:
            tprint(f"❌ Device selection failed: {e}", "ERROR")
            return "cpu"

    def optimize_memory(self, data: np.ndarray, target_memory_gb: float = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize memory usage for the given data using hardware optimization modules.

        Args:
            data: Data array to optimize
            target_memory_gb: Target memory usage in GB

        Returns:
            Tuple of (optimized_data, optimization_info)
        """
        try:
            start_time = time.time()
            tprint("🧠 Optimizing memory usage", "INFO")

            optimization_info = {
                "original_shape": data.shape,
                "original_dtype": str(data.dtype),
                "original_memory_gb": data.nbytes / (1024**3),
                "optimizations_applied": [],
                "final_memory_gb": data.nbytes / (1024**3),
                "hardware_optimization_used": False
            }

            optimized_data = data.copy()

            # Use hardware optimization modules if available
            if self._hardware_modules_available:
                try:
                    # Get memory manager
                    if self.memory_manager is None:
                        self.memory_manager = get_memory_manager()

                    # Apply hardware-optimized memory management
                    optimized_data = self.memory_manager.optimize_dataframe(optimized_data)

                    # Update optimization info
                    optimization_info["hardware_optimization_used"] = True
                    optimization_info["final_memory_gb"] = optimized_data.nbytes / (1024**3)
                    optimization_info["optimizations_applied"].append("hardware_memory_optimization")

                    tprint(f"✅ Hardware memory optimization applied: {optimization_info['original_memory_gb']:.2f}GB → {optimization_info['final_memory_gb']:.2f}GB", "SUCCESS")

                except Exception as e:
                    tprint(f"⚠️ Hardware memory optimization failed: {e}, using fallback", "WARNING")

            # Fallback to basic optimization if needed
            if not optimization_info["hardware_optimization_used"]:
                # Determine target memory (default to 80% of available memory)
                if target_memory_gb is None:
                    target_memory_gb = self.capabilities.total_memory_gb * 0.8

                current_memory_gb = optimization_info["original_memory_gb"]

                # Apply memory optimizations if needed
                if current_memory_gb > target_memory_gb:
                    tprint(f"📊 Memory usage {current_memory_gb:.2f}GB exceeds target {target_memory_gb:.2f}GB", "INFO")

                    # Try dtype optimization (float64 -> float32)
                    if data.dtype in [np.float64, np.complex128]:
                        try:
                            optimized_data = data.astype(np.float32)
                            optimization_info["optimizations_applied"].append("dtype_reduction")
                            optimization_info["final_memory_gb"] = optimized_data.nbytes / (1024**3)
                            tprint(f"✅ Reduced precision: {current_memory_gb:.2f}GB → {optimization_info['final_memory_gb']:.2f}GB", "SUCCESS")
                        except Exception as e:
                            tprint(f"⚠️ Dtype optimization failed: {e}", "WARNING")

                    # Try chunking for very large datasets
                    if optimization_info["final_memory_gb"] > target_memory_gb and data.shape[0] > 10000:
                        try:
                            # Suggest chunked processing
                            chunk_size = min(5000, data.shape[0] // 2)
                            optimization_info["optimizations_applied"].append("chunking_suggested")
                            optimization_info["suggested_chunk_size"] = chunk_size
                            tprint(f"💡 Suggesting chunked processing with size {chunk_size}", "INFO")
                        except Exception as e:
                            tprint(f"⚠️ Chunking suggestion failed: {e}", "WARNING")

            # Record optimization
            memory_saved = optimization_info["original_memory_gb"] - optimization_info["final_memory_gb"]
            self.performance_metrics["total_memory_saved_gb"] += memory_saved
            self.performance_metrics["memory_optimizations"] += 1

            execution_time = time.time() - start_time
            optimization_info["optimization_time"] = execution_time

            tprint(f"✅ Memory optimization completed in {execution_time:.3f}s", "SUCCESS")

            return optimized_data, optimization_info

        except Exception as e:
            tprint(f"❌ Memory optimization failed: {e}", "ERROR")
            return data, {"error": str(e)}

    def accelerate_neighbors(self, data: np.ndarray, n_neighbors: int = 15) -> Tuple[Any, Dict[str, Any]]:
        """
        Accelerate nearest neighbors computation using hardware optimization modules.

        Args:
            data: Data for neighbor computation
            n_neighbors: Number of neighbors to find

        Returns:
            Tuple of (neighbors_result, acceleration_info)
        """
        try:
            start_time = time.time()
            tprint(f"🏎️ Accelerating neighbors computation (k={n_neighbors})", "INFO")

            acceleration_info = {
                "device_used": "cpu",
                "algorithm": "brute_force",
                "original_time": 0.0,
                "accelerated_time": 0.0,
                "speedup_factor": 1.0,
                "hardware_acceleration_used": False
            }

            # Use hardware optimization modules if available
            if self._hardware_modules_available:
                try:
                    # Get GPU manager for acceleration
                    if self.gpu_manager is None:
                        from .gpu_manager import get_gpu_manager
                        self.gpu_manager = get_gpu_manager()

                    # Check if GPU acceleration is available
                    from .gpu_manager import GPUAccelerationType
                    if self.gpu_manager.is_acceleration_available(GPUAccelerationType.FAISS):
                        # Create accelerated operation
                        operation_id = self.gpu_manager.create_accelerated_operation(
                            GPUAccelerationType.FAISS,
                            data,
                            {"n_neighbors": n_neighbors, "metric": "euclidean"},
                            priority=5
                        )

                        if operation_id:
                            # Execute the operation
                            neighbors_result = self.gpu_manager.execute_accelerated_operation(operation_id)
                            acceleration_info["device_used"] = "gpu"
                            acceleration_info["algorithm"] = "faiss_accelerated"
                            acceleration_info["hardware_acceleration_used"] = True

                            tprint(f"✅ GPU-accelerated neighbors computation completed", "SUCCESS")
                        else:
                            raise Exception("Failed to create GPU operation")
                    else:
                        # Fallback to CPU
                        neighbors_result = self._compute_neighbors_cpu(data, n_neighbors)

                except Exception as e:
                    tprint(f"⚠️ Hardware acceleration failed: {e}, falling back to CPU", "WARNING")
                    neighbors_result = self._compute_neighbors_cpu(data, n_neighbors)
            else:
                # Fallback to CPU implementation
                neighbors_result = self._compute_neighbors_cpu(data, n_neighbors)

            # Calculate performance metrics
            execution_time = time.time() - start_time
            acceleration_info["accelerated_time"] = execution_time

            # Estimate original time (for comparison)
            if hasattr(neighbors_result, 'execution_time'):
                original_time = neighbors_result.execution_time
                acceleration_info["original_time"] = original_time
                if original_time > 0:
                    acceleration_info["speedup_factor"] = original_time / execution_time

            # Update performance metrics
            self.performance_metrics["total_optimizations"] += 1
            if acceleration_info["speedup_factor"] > 1.0:
                self.performance_metrics["average_speedup"] = (
                    (self.performance_metrics["average_speedup"] * (self.performance_metrics["total_optimizations"] - 1) +
                     acceleration_info["speedup_factor"]) / self.performance_metrics["total_optimizations"]
                )

            tprint(f"✅ Neighbors computation completed in {execution_time:.3f}s "
                  f"(speedup: {acceleration_info['speedup_factor']:.1f}x)", "SUCCESS")

            return neighbors_result, acceleration_info

        except Exception as e:
            tprint(f"❌ Neighbors acceleration failed: {e}", "ERROR")
            # Return basic CPU implementation as fallback
            return self._compute_neighbors_cpu(data, n_neighbors), {"error": str(e)}

    def _compute_neighbors_cpu(self, data: np.ndarray, n_neighbors: int):
        """Compute neighbors using CPU (standard sklearn implementation)."""
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(data)), metric='euclidean')
            nn.fit(data)

            # Return fitted model and sample results
            distances, indices = nn.kneighbors(data[:min(100, len(data))])  # Sample for performance

            return {
                "model": nn,
                "sample_distances": distances,
                "sample_indices": indices,
                "execution_time": 0.1,  # Placeholder
                "device": "cpu"
            }

        except Exception as e:
            tprint(f"❌ CPU neighbors computation failed: {e}", "ERROR")
            raise

    def _compute_neighbors_gpu(self, data: np.ndarray, n_neighbors: int):
        """Compute neighbors using GPU acceleration."""
        try:
            # Use M1 GPU manager if available
            if self.gpu_manager and hasattr(self.gpu_manager, 'accelerate_neighbors'):
                tprint(f"🚀 Using M1 GPU manager for neighbors computation", "INFO")
                result = self.gpu_manager.accelerate_neighbors(data, n_neighbors)
                return {
                    "model": result,
                    "execution_time": getattr(result, 'execution_time', 0.05),
                    "device": "gpu"
                }

            # Try to use cuML if available (NVIDIA GPU)
            try:
                import cuml
                from cuml.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(data)))
                nn.fit(data)

                return {
                    "model": nn,
                    "execution_time": 0.05,  # Placeholder - would be measured
                    "device": "gpu"
                }

            except ImportError:
                raise Exception("cuML not available")

        except Exception as e:
            tprint(f"⚠️ GPU neighbors computation not available: {e}", "WARNING")
            raise

    def _compute_neighbors_m1(self, data: np.ndarray, n_neighbors: int):
        """Compute neighbors using M1 optimization."""
        try:
            # Use M1 CPU optimizer if available
            if self.m1_optimizer and hasattr(self.m1_optimizer, 'optimize_neighbors'):
                tprint(f"🚀 Using M1 CPU optimizer for neighbors computation", "INFO")
                result = self.m1_optimizer.optimize_neighbors(data, n_neighbors)
                return {
                    "model": result,
                    "execution_time": getattr(result, 'execution_time', 0.08),
                    "device": "m1"
                }

            # M1-specific optimizations would go here
            # For now, fall back to CPU with M1-aware batching
            tprint(f"💡 Using M1-aware CPU computation", "INFO")
            return self._compute_neighbors_cpu(data, n_neighbors)

        except Exception as e:
            tprint(f"❌ M1 neighbors computation failed: {e}", "ERROR")
            raise

    def get_hardware_recommendations(self, workload_size: int) -> Dict[str, Any]:
        """
        Get hardware recommendations for the given workload.

        Args:
            workload_size: Size of the workload (number of samples)

        Returns:
            Recommendations dictionary
        """
        try:
            recommendations = {
                "recommended_device": self.select_device("clustering", workload_size),
                "recommended_batch_size": self.capabilities.optimal_batch_size,
                "recommended_workers": self.capabilities.recommended_workers,
                "memory_estimate_gb": (workload_size * 200) / (1024**3),  # Rough estimate
                "estimated_execution_time": self._estimate_execution_time(workload_size),
                "hardware_warnings": []
            }

            # Add warnings for potential issues
            if workload_size > 100000 and not self.capabilities.has_gpu:
                recommendations["hardware_warnings"].append(
                    "Large dataset without GPU - consider chunked processing"
                )

            if recommendations["memory_estimate_gb"] > self.capabilities.total_memory_gb * 0.8:
                recommendations["hardware_warnings"].append(
                    "High memory usage - consider memory optimization"
                )

            return recommendations

        except Exception as e:
            tprint(f"❌ Recommendation generation failed: {e}", "ERROR")
            return {"error": str(e)}

    def _estimate_execution_time(self, workload_size: int) -> float:
        """Estimate execution time for the given workload size."""
        try:
            # Base time estimates (in seconds)
            base_times = {
                "small": 10,    # < 1K samples
                "medium": 60,   # 1K - 10K samples
                "large": 300,   # 10K - 100K samples
                "xlarge": 1200  # > 100K samples
            }

            if workload_size < 1000:
                base_time = base_times["small"]
            elif workload_size < 10000:
                base_time = base_times["medium"]
            elif workload_size < 100000:
                base_time = base_times["large"]
            else:
                base_time = base_times["xlarge"]

            # Apply hardware speedup factors
            speedup = 1.0
            if self.capabilities.has_gpu:
                speedup *= 0.3  # 70% faster on GPU
            if self.capabilities.has_m1:
                speedup *= 0.8  # 20% faster on M1

            return base_time * speedup

        except Exception as e:
            tprint(f"⚠️ Time estimation failed: {e}", "WARNING")
            return 60.0  # Default 1 minute

    def get_hardware_statistics(self) -> Dict[str, Any]:
        """Get hardware service statistics."""
        try:
            return {
                "capabilities": self.capabilities.__dict__,
                "performance_metrics": self.performance_metrics,
                "optimization_history_length": len(self.optimization_history),
                "recent_optimizations": self.optimization_history[-5:] if self.optimization_history else [],
                "memory_utilization": psutil.virtual_memory().percent,
                "cpu_utilization": psutil.cpu_percent(interval=1)
            }

        except Exception as e:
            tprint(f"❌ Statistics collection failed: {e}", "ERROR")
            return {"error": str(e)}

    def benchmark_hardware(self, test_data: np.ndarray) -> Dict[str, Any]:
        """
        Benchmark hardware performance with test data.

        Args:
            test_data: Test data for benchmarking

        Returns:
            Benchmark results
        """
        try:
            tprint("🏃 Running hardware benchmarks", "INFO")

            benchmark_results = {
                "matrix_operations": {},
                "memory_operations": {},
                "neighbors_computation": {},
                "overall_score": 0.0
            }

            # Benchmark matrix operations
            benchmark_results["matrix_operations"] = self._benchmark_matrix_ops(test_data)

            # Benchmark memory operations
            benchmark_results["memory_operations"] = self._benchmark_memory_ops(test_data)

            # Benchmark neighbors computation
            benchmark_results["neighbors_computation"] = self._benchmark_neighbors(test_data)

            # Calculate overall score (weighted average)
            scores = [
                benchmark_results["matrix_operations"].get("score", 50),
                benchmark_results["memory_operations"].get("score", 50),
                benchmark_results["neighbors_computation"].get("score", 50)
            ]
            benchmark_results["overall_score"] = np.mean(scores)

            tprint(f"🏆 Hardware benchmark score: {benchmark_results['overall_score']:.1f}/100", "SUCCESS")

            return benchmark_results

        except Exception as e:
            tprint(f"❌ Hardware benchmarking failed: {e}", "ERROR")
            return {"error": str(e)}

    def _benchmark_matrix_ops(self, test_data: np.ndarray) -> Dict[str, Any]:
        """Benchmark matrix operations performance."""
        try:
            import time

            # Simple matrix multiplication benchmark
            start_time = time.time()
            result = np.dot(test_data.T, test_data)  # Small matrix operation
            matrix_time = time.time() - start_time

            # Score based on performance (lower time = higher score)
            # Assume 0.01s is excellent (100 points), 0.1s is good (50 points)
            if matrix_time < 0.01:
                score = 100
            elif matrix_time < 0.1:
                score = 50
            else:
                score = 25

            return {
                "time_seconds": matrix_time,
                "score": score,
                "operations_per_second": 1.0 / matrix_time if matrix_time > 0 else 0
            }

        except Exception as e:
            return {"error": str(e), "score": 0}

    def _benchmark_memory_ops(self, test_data: np.ndarray) -> Dict[str, Any]:
        """Benchmark memory operations performance."""
        try:
            import time

            # Memory copy benchmark
            start_time = time.time()
            copied_data = test_data.copy()
            memory_time = time.time() - start_time

            # Score based on performance
            data_size_gb = test_data.nbytes / (1024**3)
            if data_size_gb > 0:
                throughput_gbps = data_size_gb / memory_time if memory_time > 0 else 0
            else:
                throughput_gbps = 0

            # Score based on throughput (assume 10 GB/s is excellent)
            score = min(100, throughput_gbps * 10)

            return {
                "time_seconds": memory_time,
                "throughput_gbps": throughput_gbps,
                "score": score
            }

        except Exception as e:
            return {"error": str(e), "score": 0}

    def _benchmark_neighbors(self, test_data: np.ndarray) -> Dict[str, Any]:
        """Benchmark neighbors computation performance."""
        try:
            # Use the accelerate_neighbors method for benchmarking
            _, acceleration_info = self.accelerate_neighbors(test_data, n_neighbors=10)

            # Extract performance metrics
            speedup = acceleration_info.get("speedup_factor", 1.0)
            time_taken = acceleration_info.get("accelerated_time", 1.0)

            # Score based on speedup and absolute performance
            base_score = 50  # Base score
            speedup_bonus = min(30, (speedup - 1) * 10)  # Up to 30 points for speedup
            time_bonus = max(0, 20 - time_taken * 10)  # Bonus for fast execution

            score = base_score + speedup_bonus + time_bonus

            return {
                "speedup_factor": speedup,
                "execution_time": time_taken,
                "score": min(100, score)
            }

        except Exception as e:
            return {"error": str(e), "score": 0}

    def reset_hardware_state(self):
        """Reset hardware service state."""
        try:
            self.optimization_history.clear()

            # Reset performance metrics
            self.performance_metrics = {
                "total_optimizations": 0,
                "gpu_accelerations": 0,
                "memory_optimizations": 0,
                "m1_optimizations": 0,
                "average_speedup": 1.0,
                "total_memory_saved_gb": 0.0
            }

            tprint("🧹 Hardware state reset", "INFO")

        except Exception as e:
            tprint(f"⚠️ State reset failed: {e}", "WARNING")

    def optimize_matrix_operations(self, enable_acceleration: bool = True) -> Dict[str, Any]:
        """
        Optimize matrix operations for M1 hardware.

        Args:
            enable_acceleration: Whether to enable hardware acceleration

        Returns:
            Optimization report
        """
        try:
            if not self._hardware_modules_available:
                return {
                    'success': False,
                    'error': 'Hardware optimization modules not available'
                }

            # Get M1 optimizer
            if self.m1_optimizer is None:
                from .m1_optimizer import get_m1_optimizer
                self.m1_optimizer = get_m1_optimizer()

            # Apply matrix operation optimizations
            optimization_result = self.m1_optimizer.optimize_matrix_operations(enable_acceleration)

            # Update performance metrics
            if optimization_result.get('success', False):
                self.performance_metrics["m1_optimizations"] += 1

            return optimization_result

        except Exception as e:
            tprint(f"❌ Matrix optimization failed: {e}", "ERROR")
            return {'success': False, 'error': str(e)}

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage report using hardware optimization modules.

        Returns:
            Memory report
        """
        try:
            if self._hardware_modules_available and self.memory_manager is not None:
                return self.memory_manager.get_memory_report()
            else:
                # Fallback to basic memory info
                memory = psutil.virtual_memory()
                return {
                    'memory_stats': {
                        'total_memory': memory.total,
                        'available_memory': memory.available,
                        'used_memory': memory.used,
                        'memory_percent': memory.percent
                    }
                }

        except Exception as e:
            tprint(f"❌ Memory report failed: {e}", "ERROR")
            return {'error': str(e)}

    def get_acceleration_report(self) -> Dict[str, Any]:
        """
        Get comprehensive acceleration report using hardware optimization modules.

        Returns:
            Acceleration report
        """
        try:
            if self._hardware_modules_available and self.gpu_manager is not None:
                return self.gpu_manager.get_acceleration_report()
            else:
                return {
                    'device_info': {'available_devices': ['CPU']},
                    'current_device': 'CPU',
                    'enhanced_mode': False,
                    'gpu_manager_available': False
                }

        except Exception as e:
            tprint(f"❌ Acceleration report failed: {e}", "ERROR")
            return {'error': str(e)}

    def get_optimization_context(self):
        """
        Get optimization context manager for M1 optimizations.

        Returns:
            Context manager for hardware optimizations
        """
        try:
            if self._hardware_modules_available and self.m1_optimizer is not None:
                return self.m1_optimizer.get_optimization_context()
            else:
                # Return dummy context manager
                class DummyContext:
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                return DummyContext()

        except Exception as e:
            tprint(f"❌ Optimization context creation failed: {e}", "ERROR")
            # Return dummy context manager as fallback
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()
