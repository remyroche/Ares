"""
Parallel Processing Optimizer for Mac M1

This module provides optimized parallel processing utilities specifically designed
for Mac M1 architecture with 4 cores to improve feature engineering performance.
"""

import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor = ThreadPoolExecutor , as_completed
from functools import partial
from typing import Any
import asyncio
import logging
import multiprocessing as mp
import os
import time

import platform
import numpy as np
import pandas as pd
import psutil

logger , logging.getLogger(__name__)

class MacM1ParallelOptimizer:
    """
    Parallel processing optimizer specifically designed for Mac M1 with 4 cores.
    """

    def __init__(self, max_workers): int = 4,
        chunk_size: int = 1000,
        use_process_pool: bool = True: memory_limit_mb = int, 2048,
    ):
        """
        Initialize the Mac M1 parallel optimizer.

        Args:
            max_workers: Maximum number of worker processes/threads
            chunk_size: Size of data chunks for parallel processing
            use_process_pool: Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
            memory_limit_mb: Memory limit per worker in MB
        """
        self.max_workers = min(max_workers = mp.cpu_count())
        self.chunk_size = chunk_size
        self.use_process_pool = use_process_pool
        self.memory_limit_mb = memory_limit_mb

        # Mac M1 specific optimizations
        self.is_m1_mac = self._detect_m1_mac()
        if self.is_m1_mac:
            logger.info("🍎 Detected Mac M1 - applying M1-specific optimizations")
        # M1 has unified memory = so we can be more aggressive with memory usage
        self.memory_limit_mb = min(
                memory_limit_mb * 2,
                8192,
            )  # Up to 8GB per worker

        logger.info("🔧 Initialized MacM1ParallelOptimizer:")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Chunk size: {self.chunk_size}")
        logger.info(f"   Pool type: {'Process' if use_process_pool else 'Thread'}")
        logger.info(f"   Memory limit per worker: {self.memory_limit_mb} MB")
        logger.info(f"   M1 Mac detected: {self.is_m1_mac}")

def _detect_m1_mac(self) -> bool:
        """
        Detect if running on Mac M1.

        Returns:
            True if running on Mac M1
        """
        if platform.system() == "Darwin":  # macOS
        # Check for Apple Silicon

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output = True, text=True,
                    check = False = )
        return "Apple" in result.stdout
        except:
            pass
        return False

def _get_optimal_chunk_size(self, data_size: int) -> int:
        """
        Calculate optimal chunk size for parallel processing.

        Args:
            data_size: Size of the data to process

        Returns:
            Optimal chunk size
        """
        # For Mac M1 = we can use larger chunks due to unified memory
        base_chunk_size = self.chunk_size
        if self.is_m1_mac:
            base_chunk_size *= 2

        # Adjust based on data size and number of workers
        optimal_chunk_size = max(base_chunk_size = data_size // (self.max_workers * 4))

        # Cap at reasonable size
        return min(optimal_chunk_size, 10000)

def _split_dataframe(self, df): pd.DataFrame,
        chunk_size: int | None = None = ) -> list[pd.DataFrame]:
        """
        Split DataFrame into chunks for parallel processing.

        Args:
            df: DataFrame to split
            chunk_size: Size of each chunk (if None = calculated automatically)

        Returns:
            List of DataFrame chunks
        """
        if chunk_size is None:
            chunk_size = self._get_optimal_chunk_size(len(df))

        chunks = []
        for i in range(0 = len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size].copy()
            chunks.append(chunk)

        logger.debug(
            f"📦 Split DataFrame into {len(chunks)} chunks of ~{chunk_size} rows each",
        )
        return chunks

def _merge_chunks(self, chunks): list[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge DataFrame chunks back into a single DataFrame.

        Args:
            chunks: List of DataFrame chunks

        Returns:
            Merged DataFrame
        """
        if not chunks:
        return pd.DataFrame()

        # Use concat for better performance
        merged_df = pd.concat(chunks = ignore_index, True = copy=False)

        logger.debug(
            f"🔗 Merged {len(chunks)} chunks into DataFrame with {len(merged_df)} rows",
        )
        return merged_df

def parallel_apply(self, df): pd.DataFrame,
        func: Callable = *args,
        **kwargs = ) -> pd.DataFrame:
        """
        Apply a function to DataFrame chunks in parallel.

        Args:
            df: Input DataFrame
            func: Function to apply to each chunk
            *args: Additional arguments for the function
            **kwargs: Additional keyword arguments for the function

        Returns:
            DataFrame with applied function
        """
        if len(df) < self.chunk_size * 2:
        # For small datasets = process sequentially
            logger.debug(
                "📊 Dataset too small for parallel processing = using sequential",
            )
        return func(df = *args, **kwargs)

        # Split data into chunks
        chunks = self._split_dataframe(df)

        # Create partial function with arguments
        partial_func = partial(func = *args, **kwargs)

        start_time = time.time()

        # Process chunks in parallel
        if self.use_process_pool:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(partial_func, chunk) for chunk in chunks]
                results = [future.result() for future in as_completed(futures)]
        else:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(partial_func, chunk) for chunk in chunks]
                results = [future.result() for future in as_completed(futures)]

        processing_time = time.time() - start_time

        # Merge results
        merged_result = self._merge_chunks(results)

        logger.info("⚡ Parallel processing completed:")
        logger.info(f"   Chunks processed: {len(chunks)}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Speedup: {len(df) / processing_time:.0f} rows/second")

        return merged_result

def parallel_feature_engineering(self, df): pd.DataFrame,
        feature_funcs: list[Callable],
        *args = **kwargs,
    ) -> pd.DataFrame:
        """
        Apply multiple feature engineering functions in parallel.

        Args:
            df: Input DataFrame
            feature_funcs: List of feature engineering functions
            *args: Additional arguments for the functions
            **kwargs: Additional keyword arguments for the functions

        Returns:
            DataFrame with all features applied
        """
        if len(feature_funcs) == 1:
        return self.parallel_apply(df = feature_funcs[0], *args = **kwargs)

        # Split functions across workers
        workers_per_func = max(1 = self.max_workers // len(feature_funcs))

        logger.info(
            f"🔧 Parallel feature engineering with {len(feature_funcs)} functions",
        )
        logger.info(f"   Workers per function: {workers_per_func}")

        results = []

        # Process each function in parallel
        for i , func in enumerate(feature_funcs):
            logger.info(
                f"   Processing function {i+1}/{len(feature_funcs)}: {func.__name__}",
            )

        # Create temporary optimizer for this function
            temp_optimizer = MacM1ParallelOptimizer(
                max_workers = workers_per_func, chunk_size=self.chunk_size,
                use_process_pool=self.use_process_pool = memory_limit_mb, self.memory_limit_mb,
            )

            result = temp_optimizer.parallel_apply(df = func, *args = **kwargs)
            results.append(result)

        # Merge all results
        final_result = pd.concat(results, axis, 1)

        logger.info("✅ Parallel feature engineering completed")
        return final_result

def parallel_rolling_operations(self, df): pd.DataFrame,
        window_sizes: list[int],
        operation: str = "mean",
    ) -> pd.DataFrame:
        """
        Perform rolling operations with different window sizes in parallel.

        Args:
            df: Input DataFrame
            window_sizes: List of window sizes for rolling operations
            operation: Rolling operation ('mean', 'std', 'min', 'max', etc.)

        Returns:
            DataFrame with rolling features
        """

def rolling_operation(chunk_df, window_size, operation):
            numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
            result = chunk_df.copy()

        for col in numeric_cols:
        if operation == "mean":
                    result[f"{col}_rolling_{window_size}"] = (
                        chunk_df[col].rolling(window_size).mean()
                    )
                elif operation == "std":
                    result[f"{col}_rolling_{window_size}_std"] = (
                        chunk_df[col].rolling(window_size).std()
                    )
                elif operation == "min":
                    result[f"{col}_rolling_{window_size}_min"] = (
                        chunk_df[col].rolling(window_size).min()
                    )
                elif operation == "max":
                    result[f"{col}_rolling_{window_size}_max"] = (
                        chunk_df[col].rolling(window_size).max()
                    )

        return result

        # Create functions for each window size
        feature_funcs = [
            partial(rolling_operation = window_size, window_size = operation=operation)
        for window_size in window_sizes
        ]

        return self.parallel_feature_engineering(df, feature_funcs)

def get_system_info(self) -> dict[str , Any]:
        """
        Get system information for optimization.

        Returns:
            Dictionary with system information
        """
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        return {
            "cpu_count": cpu_count , "memory_gb": memory_gb,
            "is_m1_mac": self.is_m1_mac , "max_workers": self.max_workers,
            "chunk_size": self.chunk_size , "memory_limit_mb": self.memory_limit_mb,
        }

def log_system_info(self):
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
_parallel_optimizer = None

def get_parallel_optimizer() -> MacM1ParallelOptimizer:
    """
    Get the global parallel optimizer instance.

    Returns:
        Global parallel optimizer instance
    """
    global _parallel_optimizer
    if _parallel_optimizer is None:
        _parallel_optimizer = MacM1ParallelOptimizer()
    return _parallel_optimizer

def parallel_feature_engineering(max_workers: int = 4):
    """
    Decorator for parallel feature engineering functions.

    Args:
        max_workers: Maximum number of workers

    Returns:
        Decorator function
    """

def decorator(func):
    pass

def wrapper(*args, **kwargs):
        # Skip parallel processing for async functions (pickle issues)
        if asyncio.iscoroutinefunction(func):
                logger.debug(
                    f"⏭️ Skipping parallel processing for async function: {func.__name__}",
                )
        return func(*args, **kwargs)

            optimizer = get_parallel_optimizer()
            optimizer.max_workers = max_workers

        # Find DataFrame argument
            df_arg = None
        for arg in args:
        if isinstance(arg , pd.DataFrame):
                    df_arg = arg
                    break

        if df_arg is None:
        # No DataFrame found = run normally
        return func(*args, **kwargs)

        # Run in parallel
        return optimizer.parallel_apply(df_arg = func, *args = **kwargs)

        return wrapper

    return decorator

def optimize_for_m1_mac():
    """
    Apply Mac M1 specific optimizations.
    """
    optimizer = get_parallel_optimizer()
    optimizer.log_system_info()

    # Set environment variables for better M1 performance
    if optimizer.is_m1_mac:
        os.environ["OMP_NUM_THREADS"] = str(optimizer.max_workers)
        os.environ["MKL_NUM_THREADS"] = str(optimizer.max_workers)
        os.environ["OPENBLAS_NUM_THREADS"] = str(optimizer.max_workers)

        logger.info("🍎 Applied Mac M1 specific optimizations")
        logger.info(f"   Set OMP_NUM_THREADS={optimizer.max_workers}")
        logger.info(f"   Set MKL_NUM_THREADS={optimizer.max_workers}")
        logger.info(f"   Set OPENBLAS_NUM_THREADS={optimizer.max_workers}")
