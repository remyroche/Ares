"""
Parallel Processing Optimizer for Mac M1

This module provides optimized parallel processing utilities designed
for Apple Silicon as well as other platforms. It offers a simple
parallel apply for DataFrame workloads and a convenience decorator.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial, wraps
from typing import TYPE_CHECKING, A, Callableny

import numpy as np
import pandas as pd
import psutil

if TYPE_CHECKING:
    from collections.abc import Callable as Callable_collections_abc
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class MacM1ParallelOptimizer:
    """
    Parallel processing optimizer with Apple Silicon awareness.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        *,
        chunk_size: int = 1000,
        use_process_pool: bool = True,
        memory_limit_mb: int = 2048,
    ) -> None:
        """
        Initialize the parallel optimizer.

        Args:
            max_workers: Maximum parallel workers. Defaults to cpu_count() or 4 on M1.
            chunk_size: Target chunk size for DataFrame splitting.
            use_process_pool: Use processes if True, threads if False.
            memory_limit_mb: Logical memory budget per worker (MB).
        """
        cpu_count = mp.cpu_count() or 1
        self.is_m1_mac: bool = self._detect_m1_mac()
        # On M1, favor 4 workers by default; otherwise default to cpu_count
        default_workers = 4 if self.is_m1_mac else min(8, cpu_count)
        self.max_workers: int = (
            max_workers if max_workers and max_workers > 0 else default_workers
        )
        self.chunk_size: int = max(1, chunk_size)
        self.use_process_pool: bool = bool(use_process_pool)
        self.memory_limit_mb: int = max(128, memory_limit_mb)

        if self.is_m1_mac:
            logger.info("🍎 Detected Apple Silicon - applying M1-specific limits")
            # Unified memory allows a bit more headroom per worker.
            self.memory_limit_mb = min(self.memory_limit_mb * 2, 8192)

        logger.info("🔧 Initialized MacM1ParallelOptimizer:")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Chunk size: {self.chunk_size}")
        logger.info(f"   Pool type: {'Process' if self.use_process_pool else 'Thread'}")
        logger.info(f"   Memory limit per worker: {self.memory_limit_mb} MB")
        logger.info(f"   M1 Mac detected: {self.is_m1_mac}")

    def _detect_m1_mac(self) -> bool:
        """
        Detect if running on Apple Silicon macOS.
        """
        try:
            if platform.system() != "Darwin":
                return False
            # Prefer Python's platform.machine which is fast and available
            machine = platform.machine().lower()
            if machine in {"arm64", "aarch64"}:
                return True
            # Fallback to sysctl when available
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return "apple" in result.stdout.lower()
            except Exception:
                return False
        except Exception:
            return False

    def _get_optimal_chunk_size(self, data_size: int) -> int:
        """
        Calculate optimal chunk size for parallel processing.
        """
        base_chunk_size = self.chunk_size * (2 if self.is_m1_mac else 1)
        # Aim for ~4x workers chunks at minimum
        denom = max(1, self.max_workers * 4)
        adaptive = max(1, data_size // denom)
        optimal = max(base_chunk_size, adaptive)
        return min(optimal, 10000)

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        *,
        chunk_size: int | None = None,
    ) -> list[pd.DataFrame]:
        """
        Split DataFrame into chunks.
        """
        size = len(df)
        if size == 0:
            return [df.copy()]
        if chunk_size is None:
            # Fallback implementation for chunk_size
            chunk_size = self._get_optimal_chunk_size(size)
        chunks: list[pd.DataFrame] = []
        for i in range(0, size, chunk_size):
            chunks.append(df.iloc[i : i + chunk_size].copy())
        logger.debug(
            f"📦 Split DataFrame into {len(chunks)} chunks of ~{chunk_size} rows each"
        )
        return chunks

    def _merge_chunks(self, chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge DataFrame chunks back into a single DataFrame.
        """
        chunks_list = list(chunks)
        if not chunks_list:
            return pd.DataFrame()
        merged_df = pd.concat(chunks_list, ignore_index=True, copy=False)
        logger.debug(
            f"🔗 Merged {len(chunks_list)} chunks into DataFrame with {len(merged_df)} rows",
        )
        return merged_df

    def parallel_apply(
        self,
        df: pd.DataFrame,
        func: (
            Callable[[pd.DataFrame, Any], pd.DataFrame]
            | Callable[[pd.DataFrame], pd.DataFrame]
        ),
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Apply a function to DataFrame chunks in parallel.
        """
        if not isinstance(df, pd.DataFrame):
            msg = "parallel_apply expects a pandas DataFrame as first argument"
            raise TypeError(msg)

        # For small datasets, process sequentially to avoid overhead
        if len(df) < self.chunk_size * 2:
            logger.debug("📊 Dataset small - processing sequentially")
            return func(df, *args, **kwargs)

        chunks = self._split_dataframe(df)
        partial_func = partial(func, *args, **kwargs)
        start_time = time.time()

        if self.use_process_pool:
            executor_cls = ProcessPoolExecutor
        else:
            executor_cls = ThreadPoolExecutor

        # Process chunks in parallel
        results: list[pd.DataFrame] = []
        with executor_cls(max_workers=self.max_workers) as executor:
            futures = [executor.submit(partial_func, chunk) for chunk in chunks]
            for future in as_completed(futures):
                results.append(future.result())

        processing_time = time.time() - start_time
        merged_result = self._merge_chunks(results)

        logger.info("⚡ Parallel processing completed:")
        logger.info(f"   Chunks processed: {len(chunks)}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        if processing_time > 0:
            logger.info(f"   Speed: {len(df) / processing_time:.0f} rows/second")

        return merged_result

    def parallel_feature_engineering(
        self,
        df: pd.DataFrame,
        feature_funcs: list[Callable[[pd.DataFrame], pd.DataFrame]],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Apply multiple feature engineering functions in parallel and concat columns.
        """
        if not feature_funcs:
            return df.copy()
        if len(feature_funcs) == 1:
            return self.parallel_apply(df, feature_funcs[0], *args, **kwargs)

        workers_per_func = max(1, self.max_workers // max(1, len(feature_funcs)))
        logger.info(
            f"🔧 Parallel feature engineering with {len(feature_funcs)} functions | workers per func: {workers_per_func}",
        )

        results: list[pd.DataFrame] = []
        for func in feature_funcs:
            temp_optimizer = MacM1ParallelOptimizer(
                max_workers=workers_per_func,
                chunk_size=self.chunk_size,
                use_process_pool=self.use_process_pool,
                memory_limit_mb=self.memory_limit_mb,
            )
            result = temp_optimizer.parallel_apply(df, func, *args, **kwargs)
            results.append(result)

        final_result = pd.concat(results, axis=1)
        logger.info("✅ Parallel feature engineering completed")
        return final_result

    def parallel_rolling_operations(
        self,
        df: pd.DataFrame,
        window_sizes: list[int],
        operation: str = "mean",
    ) -> pd.DataFrame:
        """
        Perform rolling operations with different window sizes in parallel.
        """

        def rolling_operation(
            chunk_df: pd.DataFrame, window_size: int, op: str
        ) -> pd.DataFrame:
            numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
            result = chunk_df.copy()
            for col in numeric_cols:
                if op == "mean":
                    result[f"{col}_rolling_{window_size}"] = (
                        chunk_df[col].rolling(window_size).mean()
                    )
                elif op == "std":
                    result[f"{col}_rolling_{window_size}_std"] = (
                        chunk_df[col].rolling(window_size).std()
                    )
                elif op == "min":
                    result[f"{col}_rolling_{window_size}_min"] = (
                        chunk_df[col].rolling(window_size).min()
                    )
                elif op == "max":
                    result[f"{col}_rolling_{window_size}_max"] = (
                        chunk_df[col].rolling(window_size).max()
                    )
            return result

        feature_funcs = [
            partial(rolling_operation, window_size=w, op=operation)
            for w in window_sizes
        ]
        return self.parallel_feature_engineering(df, feature_funcs)

    def get_system_info(self) -> dict[str, Any]:
        """
        Get system information for optimization.
        """
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        return {
            "cpu_count": cpu_count,
            "memory_gb": memory_gb,
            "is_m1_mac": self.is_m1_mac,
            "max_workers": self.max_workers,
            "chunk_size": self.chunk_size,
            "memory_limit_mb": self.memory_limit_mb,
        }

    def log_system_info(self) -> None:
        """Log system information for debugging."""
        info = self.get_system_info()
        logger.info("💻 System Information:")
        logger.info(f"   CPU cores: {info['cpu_count']}")
        logger.info(f"   Total memory: {info['memory_gb']:.1f} GB")
        logger.info(f"   M1 Mac: {info['is_m1_mac']}")
        logger.info(f"   Max workers: {info['max_workers']}")
        logger.info(f"   Chunk size: {info['chunk_size']}")
        logger.info(f"   Memory limit per worker: {info['memory_limit_mb']} MB")


# Global optimizer instance
_parallel_optimizer: MacM1ParallelOptimizer | None = None


def get_parallel_optimizer() -> MacM1ParallelOptimizer:
    """
    Get the global parallel optimizer instance.
    """
    global _parallel_optimizer
    if _parallel_optimizer is None:
        # Fallback implementation for _parallel_optimizer
        _parallel_optimizer = MacM1ParallelOptimizer()
    return _parallel_optimizer


def parallel_feature_engineering(
    max_workers: int = 4,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for parallel feature engineering functions that return a DataFrame.
    Skips parallelization for async functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # If async, return function unchanged to preserve coroutine semantics
        if asyncio.iscoroutinefunction(func):
            return func  # type: ignore[return-value]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            optimizer = get_parallel_optimizer()
            optimizer.max_workers = max(1, max_workers)

            # Identify first DataFrame arg
            df_arg: pd.DataFrame | None = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df_arg = arg
                    break
            if df_arg is None:
                # Fallback implementation for df_arg
                # Fallback implementation for df_arg
                # Try kwargs
                for v in kwargs.values():
                    if isinstance(v, pd.DataFrame):
                        df_arg = v
                        break

            if df_arg is None:
                # Fallback implementation for df_arg
                # Fallback implementation for df_arg
                return func(*args, **kwargs)

            # Run function in parallel by applying it to chunks and merging
            def apply_func(chunk: pd.DataFrame) -> pd.DataFrame:
                # type: ignore[misc]
                return func(
                    chunk,
                    *[a for a in args if not isinstance(a, pd.DataFrame)],
                    **kwargs,
                )

            return optimizer.parallel_apply(df_arg, apply_func)

        return wrapper

    return decorator


def optimize_for_m1_mac() -> None:
    """
    Apply Mac M1 specific optimizations via environment hints.
    """
    optimizer = get_parallel_optimizer()
    optimizer.log_system_info()

    if optimizer.is_m1_mac:
        os.environ["OMP_NUM_THREADS"] = str(optimizer.max_workers)
        os.environ["MKL_NUM_THREADS"] = str(optimizer.max_workers)
        os.environ["OPENBLAS_NUM_THREADS"] = str(optimizer.max_workers)
        logger.info("🍎 Applied Mac M1 specific optimizations")
        logger.info(f"   Set OMP_NUM_THREADS={optimizer.max_workers}")
        logger.info(f"   Set MKL_NUM_THREADS={optimizer.max_workers}")
        logger.info(f"   Set OPENBLAS_NUM_THREADS={optimizer.max_workers}")
