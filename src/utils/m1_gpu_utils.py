"""
M1 GPU Utilities for Training Pipeline Optimization.

This module provides comprehensive GPU utilities optimized for Apple M1/M2/M3 chips
using Metal Performance Shaders (MPS). It includes automatic device detection,
memory management, and performance optimizations for the training pipeline.
"""

import torch
import logging
import numpy as np
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from contextlib import contextmanager
import psutil
import os
import collections

logger = logging.getLogger(__name__)


class M1GPUManager:
    """Manager for M1 GPU operations with automatic optimization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize M1 GPU manager with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('M1GPUManager')
        self.device = self._detect_device()
        self.memory_info = self._get_memory_info()
        # Learned/adaptive thresholds cache per (operation_type, dtype)
        self._adaptive_thresholds: Dict[Tuple[str, Optional[torch.dtype]], int] = {}

        # Configuration defaults
        self.enable_mixed_precision = self.config.get('enable_mixed_precision', True)
        self.enable_memory_cleanup = self.config.get('enable_memory_cleanup', True)
        self.batch_size = self.config.get('batch_size', 1000)
        self.memory_threshold = self.config.get('memory_threshold', 0.8)

        # Adaptive GPU/precision policy controls (with env overrides)
        self.precision_policy = str(self.config.get(
            'precision_policy', os.getenv('ARES_PRECISION_POLICY', 'auto')
        )).lower()
        self.force_gpu = bool(self.config.get(
            'force_gpu', str(os.getenv('ARES_FORCE_GPU', 'false')).lower() in {'1', 'true', 'yes'}
        ))
        self.disable_gpu = bool(self.config.get(
            'disable_gpu', str(os.getenv('ARES_DISABLE_GPU', 'false')).lower() in {'1', 'true', 'yes'}
        ))

        # Operation base thresholds (can be overridden)
        self.gpu_base_thresholds: Dict[str, int] = {
            'matrix_mult': 1000,
            'neural_net': 5000,
            'general': 10000
        }
        self.gpu_base_thresholds.update(self.config.get('gpu_thresholds', {}))

        # Precision support and decision cache
        self.supports_fp16: bool = False
        self.supports_bf16: bool = False
        self._selected_dtype_cache: Dict[str, torch.dtype] = {}

        # Probe dtype support for the detected device
        self._probe_dtype_support()

        self.logger.info(f"🔧 M1 GPU Manager initialized with device: {self.device}")
        self.logger.info(f"📊 Memory: {self.memory_info}")

    def _detect_device(self) -> torch.device:
        """Detect the best available device for M1."""
        if torch.backends.mps.is_available():
            self.logger.info("✅ MPS (Metal Performance Shaders) is available")
            return torch.device("mps")
        elif torch.cuda.is_available():
            self.logger.warning("⚠️ CUDA detected, but MPS recommended for M1")
            return torch.device("cuda")
        else:
            self.logger.info("ℹ️ Using CPU (MPS not available)")
            return torch.device("cpu")

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the system."""
        try:
            if psutil:
                memory = psutil.virtual_memory()
                return {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percentage': memory.percent
                }
            else:
                return {'total_gb': 16.0, 'available_gb': 8.0, 'used_gb': 8.0, 'percentage': 50.0}
        except Exception as e:
            self.logger.warning(f"Failed to get memory info: {e}")
            return {'total_gb': 16.0, 'available_gb': 8.0, 'used_gb': 8.0, 'percentage': 50.0}

    def should_use_gpu(
        self,
        data_size: int,
        operation_type: str = "general",
        dtype: Optional[torch.dtype] = None,
        shape: Optional[Tuple[int, ...]] = None
    ) -> bool:
        """Determine if GPU should be used for a given operation.

        Uses fresh memory stats, dtype-aware heuristics, and adaptive thresholds.
        """
        if self.disable_gpu:
            self.logger.debug("🚫 GPU usage disabled by config/env; using CPU")
            return False

        if self.device.type == "cpu":
            return False

        # GPU memory check
        current_mem = self._get_memory_info()
        self.memory_info = current_mem  # refresh cached snapshot

        # Base threshold for operation
        base_threshold = self.gpu_base_thresholds.get(operation_type, self.gpu_base_thresholds['general'])

        # Apply simple shape heuristics (small inner dims favor CPU)
        if shape is not None and operation_type == 'matrix_mult':
            try:
                if len(shape) >= 2:
                    m = shape[-2]
                    n = shape[-1]
                    if min(m, n) < 64:
                        base_threshold = int(base_threshold * 1.5)
            except Exception:
                pass

        # Memory pressure factor
        mem_pct = float(current_mem.get('percentage', 50.0))
        if mem_pct <= 50:
            memory_factor = 0.75
        elif mem_pct <= 70:
            memory_factor = 1.0
        elif mem_pct <= 85:
            memory_factor = 1.25
        else:
            memory_factor = 1.5

        # Precision factor using preferred dtype (can differ from current dtype)
        preferred_dtype = self._select_precision_dtype(operation_type=operation_type, data_size=data_size)
        if preferred_dtype == torch.float16:
            precision_factor = 0.85
        elif preferred_dtype == getattr(torch, 'bfloat16', None):
            precision_factor = 0.90
        else:
            precision_factor = 1.12

        adaptive_threshold = int(max(1, base_threshold * memory_factor * precision_factor))

        # Apply any learned override
        adaptive_key = (operation_type, preferred_dtype)
        if adaptive_key in self._adaptive_thresholds:
            adaptive_threshold = self._adaptive_thresholds[adaptive_key]

        # Optional force GPU override for sufficiently large inputs
        if self.force_gpu and data_size >= max(256, adaptive_threshold // 4):
            self.logger.debug(
                f"⚡ Forcing GPU (size {data_size}, op {operation_type}, adaptive_th {adaptive_threshold}, mem% {mem_pct:.1f})"
            )
            return True

        should_use = data_size >= adaptive_threshold and mem_pct <= (self.memory_threshold * 100)

        if should_use:
            self.logger.debug(
                f"🎯 Using GPU for {operation_type} (size: {data_size}, base_th: {base_threshold}, mem%: {mem_pct:.1f}, "
                f"mem_factor: {memory_factor:.2f}, prec: {str(preferred_dtype).split('.')[-1]}, prec_factor: {precision_factor:.2f}, "
                f"adaptive_th: {adaptive_threshold})"
            )
        else:
            self.logger.debug(
                f"💻 Using CPU for {operation_type} (size: {data_size} < adaptive_th: {adaptive_threshold} or mem% {mem_pct:.1f} > {self.memory_threshold*100:.0f}%)"
            )

        return should_use

    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        results = {'gpu_cache_cleared': False, 'gc_collected': 0, 'memory_freed_mb': 0}

        try:
            # Clear GPU cache if available
            if self.device.type == "mps":
                torch.mps.empty_cache()
                results['gpu_cache_cleared'] = True
                self.logger.debug("🧹 MPS cache cleared")
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
                results['gpu_cache_cleared'] = True
                self.logger.debug("🧹 CUDA cache cleared")

            # Force garbage collection
            results['gc_collected'] = gc.collect()
            self.logger.debug(f"🗑️ Garbage collected {results['gc_collected']} objects")

            # Log memory status
            current_memory = psutil.virtual_memory() if psutil else None
            if current_memory:
                results['memory_freed_mb'] = self.memory_info['used_gb'] * 1024 - current_memory.used / (1024**2)
                self.logger.debug(f"💾 Memory freed: {results['memory_freed_mb']:.1f}MB")

        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")

        return results

    @contextmanager
    def gpu_context(self, operation_name: str = "unknown"):
        """Context manager for GPU operations with automatic cleanup."""
        start_memory_info = self._get_memory_info()
        start_memory_gb = start_memory_info['used_gb']
        start_time_evt = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        start_perf = None
        elapsed_ms_final = None
        if start_time_evt:
            start_time_evt.record()
        else:
            # Use wall clock for MPS/CPU
            import time as _time
            start_perf = _time.perf_counter()

        try:
            yield self.device
        finally:
            # Memory cleanup
            if self.enable_memory_cleanup:
                self.optimize_memory()

            # Timing
            if start_time_evt and self.device.type == "cuda":
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_ms = start_time_evt.elapsed_time(end_time)
                elapsed_ms_final = float(elapsed_ms)
                self.logger.debug(f"⏱️ {operation_name} took {elapsed_ms_final:.2f} ms")
            else:
                import time as _time
                if self.device.type == "mps":
                    # Ensure ops have completed
                    if hasattr(torch.mps, 'synchronize'):
                        try:
                            torch.mps.synchronize()
                        except Exception:
                            pass
                if start_perf is not None:
                    elapsed_s = _time.perf_counter() - start_perf
                    elapsed_ms_final = float(elapsed_s * 1000.0)
                    self.logger.debug(f"⏱️ {operation_name} took {elapsed_ms_final:.2f} ms")

            # Memory delta log (debug)
            end_memory_gb = self._get_memory_info().get('used_gb', start_memory_gb)
            mem_delta_mb = (end_memory_gb - start_memory_gb) * 1024
            if abs(mem_delta_mb) > 5:
                self.logger.debug(f"💾 {operation_name} memory delta: {mem_delta_mb:+.1f} MB")

            # Emit concise INFO for longer operations to keep terminal progress visible
            if elapsed_ms_final is not None and elapsed_ms_final >= 200.0:
                self.logger.info(f"✅ {operation_name} completed in {elapsed_ms_final:.0f} ms (Δmem {mem_delta_mb:+.0f} MB)")

    def to_device(self, tensor: Union[torch.Tensor, np.ndarray], operation_type: str = "general") -> torch.Tensor:
        """Move tensor to appropriate device with optimization."""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")

        # Determine if GPU should be used
        data_size = tensor.numel()
        use_gpu = self.should_use_gpu(data_size, operation_type, dtype=tensor.dtype, shape=tuple(tensor.shape))

        target_device = self.device if use_gpu else torch.device("cpu")

        # Decide precision based on policy/support
        desired_dtype: Optional[torch.dtype] = None
        if self.enable_mixed_precision and torch.is_floating_point(tensor):
            desired_dtype = self._select_precision_dtype(operation_type=operation_type, data_size=data_size)

        # Move with dtype if applicable, with safe fallback
        try:
            if desired_dtype is not None and tensor.dtype != desired_dtype:
                tensor = tensor.to(target_device, dtype=desired_dtype)
            else:
                tensor = tensor.to(target_device)
        except Exception as e:
            self.logger.warning(
                f"Precision/device transfer failed ({e}); retrying with default dtype on {target_device}."
            )
            tensor = tensor.to(target_device)

        return tensor

    # ----- Precision selection and probing -----
    def _probe_dtype_support(self) -> None:
        """Probe whether the current device supports float16 and bfloat16 reliably."""
        if self.device.type == 'cpu':
            # CPU tensors can be allocated in these dtypes; ops may upcast
            self.supports_fp16 = True
            self.supports_bf16 = True
            return

        # Try a tiny allocation and op for each dtype
        for dtype_name, dtype in (('fp16', torch.float16), ('bf16', getattr(torch, 'bfloat16', None))):
            if dtype is None:
                if dtype_name == 'fp16':
                    self.supports_fp16 = False
                else:
                    self.supports_bf16 = False
                continue
            is_supported = False
            try:
                x = torch.ones((2, 2), device=self.device, dtype=dtype)
                y = torch.ones((2, 2), device=self.device, dtype=dtype)
                _ = (x @ y).sum().item()
                is_supported = True
            except Exception as e:
                self.logger.debug(f"ℹ️ {dtype_name} not fully supported on {self.device}: {e}")

            if dtype_name == 'fp16':
                self.supports_fp16 = is_supported
            else:
                self.supports_bf16 = is_supported

        self.logger.info(
            f"🎚️ DType support on {self.device}: fp16={self.supports_fp16}, bf16={self.supports_bf16}"
        )

    def _select_precision_dtype(self, operation_type: str = 'general', data_size: Optional[int] = None) -> torch.dtype:
        """Select the preferred dtype for an operation based on policy, device support, and memory.

        Policies:
        - 'fp32' -> torch.float32
        - 'bf16' -> torch.bfloat16 (if supported else fp16/32 fallback)
        - 'fp16' -> torch.float16 (if supported else bf16/32 fallback)
        - 'auto' -> Heuristic based on operation, size, and memory pressure
        """
        # Config/env hard overrides
        if self.precision_policy in {'fp32', 'float32'}:
            return torch.float32
        if self.precision_policy in {'bf16', 'bfloat16'}:
            if self.supports_bf16:
                return getattr(torch, 'bfloat16')
            return torch.float16 if self.supports_fp16 else torch.float32
        if self.precision_policy in {'fp16', 'float16'}:
            if self.supports_fp16:
                return torch.float16
            return getattr(torch, 'bfloat16') if self.supports_bf16 else torch.float32

        # Cached decision per operation size bucket
        size_bucket = 'large' if (data_size or 0) > 1_000_000 else ('medium' if (data_size or 0) >= 250_000 else 'small')
        cache_key = f"{operation_type}:{size_bucket}"
        if cache_key in self._selected_dtype_cache:
            return self._selected_dtype_cache[cache_key]

        # Heuristic: prefer bf16 for neural nets (better range), fp16 for large matmuls, fp32 for small ops
        mem_pct = float(self.memory_info.get('percentage', 50.0))
        is_low_memory_pressure = mem_pct <= 60
        is_high_memory_pressure = mem_pct >= 85
        is_large = (data_size or 0) >= 1_000_000
        is_medium = 250_000 <= (data_size or 0) < 1_000_000

        preferred: torch.dtype = torch.float32
        if operation_type == 'matrix_mult':
            if is_large and self.supports_fp16:
                preferred = torch.float16
            elif self.supports_bf16 and (is_medium or is_large):
                preferred = getattr(torch, 'bfloat16')
            else:
                preferred = torch.float32
        elif operation_type == 'neural_net':
            if self.supports_bf16 and (is_medium or is_large or is_low_memory_pressure):
                preferred = getattr(torch, 'bfloat16')
            elif self.supports_fp16 and (is_large or is_high_memory_pressure):
                preferred = torch.float16
            else:
                preferred = torch.float32
        else:
            # general
            if is_large and (self.supports_bf16 or self.supports_fp16):
                preferred = getattr(torch, 'bfloat16') if self.supports_bf16 else torch.float16
            elif is_high_memory_pressure and self.supports_fp16:
                preferred = torch.float16
            else:
                preferred = torch.float32

        self._selected_dtype_cache[cache_key] = preferred
        return preferred

    def matrix_multiply_mps(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Optimized matrix multiplication for MPS."""
        with self.gpu_context("matrix_multiply"):
            # Ensure tensors are on the correct device
            a_gpu = self.to_device(a, "matrix_mult")
            b_gpu = self.to_device(b, "matrix_mult")

            # Perform multiplication
            result = torch.matmul(a_gpu, b_gpu)

            # Convert back to CPU if needed
            if self.device.type == "cpu":
                result = result.cpu()

            return result

    def batch_process_mps(
        self,
        data: torch.Tensor,
        batch_size: Optional[int] = None,
        op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        concat_dim: int = 0,
        stream: bool = False,
        operation_type: str = "general",
        return_cpu: bool = False
    ) -> Union[torch.Tensor, Iterator[torch.Tensor]]:
        """Batch process data with MPS optimization.

        If op is provided, applies it to each batch on the chosen device.
        If stream=True, returns an iterator that yields each processed batch.
        """
        batch_size = batch_size or self.batch_size

        def _batch_iter() -> Iterator[torch.Tensor]:
            with self.gpu_context(f"batch_process:{operation_type}"):
                for i in range(0, data.shape[0], batch_size):
                    batch = data[i:i + batch_size]
                    batch_dev = self.to_device(batch, operation_type)
                    out = op(batch_dev) if op is not None else batch_dev
                    if return_cpu and isinstance(out, torch.Tensor) and out.device.type != "cpu":
                        out = out.detach().cpu()
                    yield out
                    # Periodic memory cleanup
                    if (i // batch_size) % 3 == 0:
                        self.optimize_memory()

        if stream:
            return _batch_iter()

        # Non-streaming: collect results
        if data.shape[0] <= batch_size and op is None:
            return self.to_device(data, operation_type)

        outputs: List[torch.Tensor] = []
        for out in _batch_iter():
            outputs.append(out)

        return torch.cat(outputs, dim=concat_dim)


class M1PerformanceOptimizer:
    """Performance optimizer for M1 training operations."""

    def __init__(self, gpu_manager: M1GPUManager):
        self.gpu_manager = gpu_manager
        self.logger = logger.getChild('M1PerformanceOptimizer')

    def optimize_pytorch_settings(self):
        """Optimize PyTorch settings for M1."""
        # Set memory fraction for MPS
        if self.gpu_manager.device.type == "mps":
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
            self.logger.info("🔧 Set MPS memory watermark to 0.8")

        # Disable CUDA for MPS optimization
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.logger.info("🔧 Disabled CUDA for MPS optimization")

        # Set thread optimization
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self.logger.info(f"🔧 Set PyTorch threads to {torch.get_num_threads()}")

    def get_optimal_batch_size(self, data_shape: Tuple[int, ...], operation_type: str = "general") -> int:
        """Get optimal batch size based on data and operation type."""
        data_size = np.prod(data_shape)

        # Base batch sizes optimized for M1
        base_sizes = {
            'matrix_mult': 512,
            'neural_net': 256,
            'general': 1024
        }

        base_size = base_sizes.get(operation_type, base_sizes['general'])

        # Adjust based on memory
        memory_factor = min(1.0, self.gpu_manager.memory_info['available_gb'] / 8.0)
        optimal_size = int(base_size * memory_factor)

        # Adjust based on data size
        size_factor = min(1.0, data_size / 1000000)  # Normalize by 1M elements
        optimal_size = int(optimal_size * size_factor)

        optimal_size = max(1, min(optimal_size, data_size))  # Clamp to valid range

        self.logger.debug(f"📏 Optimal batch size for {operation_type}: {optimal_size} (data_size: {data_size})")
        return optimal_size


def create_m1_optimized_config() -> Dict[str, Any]:
    """Create M1-optimized configuration for training pipeline."""
    return {
        "m1_gpu": {
            "enable_mps": True,
            "enable_mixed_precision": True,
            "enable_memory_cleanup": True,
            "batch_size": 1000,
            "memory_threshold": 0.8,
            "precision_policy": "auto",  # auto | fp32 | bf16 | fp16
            "gpu_thresholds": {
                "matrix_mult": 1000,
                "neural_net": 5000,
                "general": 10000
            },
            "force_gpu": False,
            "disable_gpu": False,
            "enable_parallel_processing": True,
            "cpu_fallback_threshold": 10000
        },
        "performance": {
            "enable_caching": True,
            "enable_memory_pooling": True,
            "enable_chunked_processing": True,
            "chunk_size": 5000,
            "max_workers": 4
        },
        "optimization": {
            "enable_adaptive_batch_size": True,
            "enable_memory_monitoring": True,
            "enable_performance_tracking": True,
            "cleanup_interval": 100
        }
    }


def initialize_m1_gpu() -> M1GPUManager:
    """Initialize M1 GPU manager with optimal settings."""
    config = create_m1_optimized_config()
    manager = M1GPUManager(config["m1_gpu"])

    # Apply performance optimizations
    optimizer = M1PerformanceOptimizer(manager)
    optimizer.optimize_pytorch_settings()

    return manager


# Global instance for easy access
_m1_gpu_manager = None

def get_m1_gpu_manager() -> M1GPUManager:
    """Get global M1 GPU manager instance."""
    global _m1_gpu_manager
    if _m1_gpu_manager is None:
        _m1_gpu_manager = initialize_m1_gpu()
    return _m1_gpu_manager


def m1_tensor_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convenience function for M1-optimized tensor multiplication."""
    manager = get_m1_gpu_manager()
    return manager.matrix_multiply_mps(a, b)


def m1_batch_process(
    data: torch.Tensor,
    batch_size: Optional[int] = None,
    op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    concat_dim: int = 0,
    stream: bool = False,
    operation_type: str = "general",
    return_cpu: bool = False
) -> Union[torch.Tensor, Iterator[torch.Tensor]]:
    """Convenience function for M1-optimized batch processing with optional callback and streaming."""
    manager = get_m1_gpu_manager()
    return manager.batch_process_mps(
        data,
        batch_size=batch_size,
        op=op,
        concat_dim=concat_dim,
        stream=stream,
        operation_type=operation_type,
        return_cpu=return_cpu,
    )


def m1_monte_carlo_simulate(
    historical_data: np.ndarray,
    n_simulations: int,
    trading_days: int = 252,
    use_mps: bool = True
) -> Dict[str, Any]:
    """M1-optimized Monte Carlo simulation engine."""
    manager = get_m1_gpu_manager()

    # Convert to tensor for MPS processing
    data_tensor = manager.to_device(historical_data, "general")

    results = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rates': [],
        'volatilities': [],
        'var_95': [],
        'cvar_95': [],
        'convergence_history': []
    }

    # Batch processing for efficiency
    batch_size = min(1000, n_simulations // 4 + 1)

    with manager.gpu_context("monte_carlo_simulation"):
        for batch_start in range(0, n_simulations, batch_size):
            batch_end = min(batch_start + batch_size, n_simulations)

            # Generate bootstrap samples using MPS
            bootstrap_indices = torch.randint(
                0, len(data_tensor), (batch_end - batch_start, trading_days),
                device=manager.device
            )

            # Gather historical returns
            bootstrap_returns = data_tensor[bootstrap_indices]

            # Calculate cumulative returns
            cumulative_returns = torch.cumprod(1 + bootstrap_returns, dim=1)

            # Calculate performance metrics on GPU
            total_returns = cumulative_returns[:, -1] - 1

            # Annualized metrics
            returns_np = total_returns.cpu().numpy()

            for ret in returns_np:
                # Calculate Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                sharpe_ratio = (ret * 252 / trading_days - risk_free_rate) / \
                              (np.std(bootstrap_returns.cpu().numpy(), axis=1).mean() * np.sqrt(252)) \
                              if np.std(bootstrap_returns.cpu().numpy(), axis=1).mean() > 0 else 0

                # Maximum drawdown
                peak = np.maximum.accumulate(cumulative_returns.cpu().numpy(), axis=1)
                drawdown = (cumulative_returns.cpu().numpy() - peak) / peak
                max_drawdown = np.min(drawdown, axis=1)

                # Win rate
                win_rate = np.mean(bootstrap_returns.cpu().numpy() > 0, axis=1)

                # VaR and CVaR
                var_95 = np.percentile(bootstrap_returns.cpu().numpy(), 5, axis=1)
                losses = bootstrap_returns.cpu().numpy()[bootstrap_returns.cpu().numpy() <= var_95[:, np.newaxis]]
                cvar_95 = np.mean(losses, axis=1) if len(losses) > 0 else var_95

                # Store results
                results['returns'].append(float(ret))
                results['sharpe_ratios'].append(float(sharpe_ratio))
                results['max_drawdowns'].append(float(max_drawdown.mean()))
                results['win_rates'].append(float(win_rate.mean()))
                results['volatilities'].append(float(np.std(bootstrap_returns.cpu().numpy(), axis=1).mean() * np.sqrt(252)))
                results['var_95'].append(float(var_95.mean()))
                results['cvar_95'].append(float(cvar_95.mean()))

            # Track convergence
            if batch_start // batch_size % 10 == 0:
                results['convergence_history'].append({
                    'simulation': batch_start,
                    'mean_return': np.mean(results['returns']),
                    'std_return': np.std(results['returns']),
                    'mean_sharpe': np.mean(results['sharpe_ratios'])
                })

            # Memory cleanup
            if batch_start % (batch_size * 3) == 0:
                manager.optimize_memory()

    return results
