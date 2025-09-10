#!/usr/bin/env python3
"""Parallel I/O Operations for Step03 with Async File Loading and Processing.

This module provides advanced parallel I/O capabilities including:
1. Parallel file loading with async operations
2. Concurrent data processing
3. Async file writing and persistence
4. I/O performance monitoring
5. Error handling and retry mechanisms
"""

import asyncio
import aiofiles
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import pickle
from datetime import datetime
import hashlib
import os

logger = logging.getLogger(__name__)

@dataclass
class IOConfig:
    """Configuration for I/O operations."""
    max_concurrent_files: int = 10
    max_workers: int = 4
    chunk_size_mb: int = 100
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_compression: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_performance_monitoring: bool = True
    max_file_size_mb: int = 1000

@dataclass
class IOMetrics:
    """I/O operation metrics."""
    operation_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    file_size_mb: float
    throughput_mbps: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0

class IOPerformanceMonitor:
    """Monitor I/O performance and throughput."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IOPerformanceMonitor")
        self.metrics_history = []
        self.active_operations = {}
    
    def start_operation(self, operation_id: str, operation_name: str, file_size_mb: float = 0) -> None:
        """Start tracking an I/O operation."""
        self.active_operations[operation_id] = {
            'operation_name': operation_name,
            'start_time': time.time(),
            'file_size_mb': file_size_mb
        }
        self.logger.debug(f"🔍 Started tracking I/O operation: {operation_name} ({operation_id})")
    
    def end_operation(self, operation_id: str, success: bool, error_message: Optional[str] = None, retry_count: int = 0) -> IOMetrics:
        """End tracking an I/O operation and return metrics."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"⚠️ Operation {operation_id} not found in active operations")
            return None
        
        operation_data = self.active_operations.pop(operation_id)
        end_time = time.time()
        duration = end_time - operation_data['start_time']
        
        # Calculate throughput
        throughput_mbps = 0
        if duration > 0 and operation_data['file_size_mb'] > 0:
            throughput_mbps = operation_data['file_size_mb'] / duration
        
        metrics = IOMetrics(
            operation_name=operation_data['operation_name'],
            start_time=operation_data['start_time'],
            end_time=end_time,
            duration_seconds=duration,
            file_size_mb=operation_data['file_size_mb'],
            throughput_mbps=throughput_mbps,
            success=success,
            error_message=error_message,
            retry_count=retry_count
        )
        
        self.metrics_history.append(metrics)
        
        if success:
            self.logger.info(f"✅ I/O operation completed: {operation_data['operation_name']} "
                           f"({duration:.2f}s, {throughput_mbps:.1f} MB/s)")
        else:
            self.logger.error(f"❌ I/O operation failed: {operation_data['operation_name']} "
                            f"({duration:.2f}s) - {error_message}")
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get I/O performance summary."""
        if not self.metrics_history:
            return {'total_operations': 0}
        
        successful_ops = [m for m in self.metrics_history if m.success]
        failed_ops = [m for m in self.metrics_history if not m.success]
        
        total_duration = sum(m.duration_seconds for m in self.metrics_history)
        total_size_mb = sum(m.file_size_mb for m in self.metrics_history)
        avg_throughput = total_size_mb / total_duration if total_duration > 0 else 0
        
        return {
            'total_operations': len(self.metrics_history),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': len(successful_ops) / len(self.metrics_history),
            'total_duration_seconds': total_duration,
            'total_size_mb': total_size_mb,
            'average_throughput_mbps': avg_throughput,
            'average_duration_seconds': total_duration / len(self.metrics_history),
            'retry_count': sum(m.retry_count for m in self.metrics_history)
        }

class AsyncFileCache:
    """Async file cache with TTL support."""
    
    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.logger = logging.getLogger(f"{__name__}.AsyncFileCache")
        self._cache_metadata = {}
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(key)
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            # Check TTL
            async with aiofiles.open(metadata_path, 'r') as f:
                metadata = json.loads(await f.read())
            
            if time.time() - metadata['created_at'] > self.ttl_seconds:
                # Expired, remove cache files
                cache_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                return None
            
            # Load cached data
            if cache_path.suffix == '.json':
                async with aiofiles.open(cache_path, 'r') as f:
                    return json.loads(await f.read())
            elif cache_path.suffix == '.pkl':
                # For pickle files, we need to use sync operations
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                # Assume it's a parquet file
                return pd.read_parquet(cache_path)
                
        except Exception as e:
            self.logger.error(f"Failed to load cache for key {key}: {e}")
            # Remove corrupted cache files
            cache_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return None
    
    async def set(self, key: str, value: Any, file_format: str = 'json') -> None:
        """Set cached value."""
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(key)
        
        try:
            # Save data
            if file_format == 'json':
                cache_path = cache_path.with_suffix('.json')
                async with aiofiles.open(cache_path, 'w') as f:
                    await f.write(json.dumps(value, default=str))
            elif file_format == 'pkl':
                cache_path = cache_path.with_suffix('.pkl')
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            elif file_format == 'parquet':
                cache_path = cache_path.with_suffix('.parquet')
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(cache_path)
                else:
                    raise ValueError("Parquet format only supports DataFrame values")
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Save metadata
            metadata = {
                'created_at': time.time(),
                'file_format': file_format,
                'file_size': cache_path.stat().st_size
            }
            
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))
            
            self.logger.debug(f"💾 Cached data for key: {key} ({file_format})")
            
        except Exception as e:
            self.logger.error(f"Failed to cache data for key {key}: {e}")
            # Clean up on failure
            cache_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
    
    async def clear(self) -> None:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            for meta_file in self.cache_dir.glob("*.meta"):
                meta_file.unlink()
            self.logger.info("🧹 Cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")

class ParallelIOOperations:
    """Parallel I/O operations with async support."""
    
    def __init__(self, config: Optional[IOConfig] = None):
        self.config = config or IOConfig()
        self.logger = logging.getLogger(f"{__name__}.ParallelIOOperations")
        self.performance_monitor = IOPerformanceMonitor()
        self.cache = AsyncFileCache(Path("cache/io"), self.config.cache_ttl_seconds) if self.config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
    
    async def load_file_async(self, file_path: Path, file_format: str = 'auto') -> pd.DataFrame:
        """Load a single file asynchronously."""
        operation_id = f"load_{file_path.name}_{int(time.time())}"
        file_size_mb = file_path.stat().st_size / (1024**2)
        
        self.performance_monitor.start_operation(operation_id, f"load_{file_path.name}", file_size_mb)
        
        async with self._semaphore:
            try:
                # Check cache first
                if self.cache:
                    cache_key = f"file_{file_path}_{file_path.stat().st_mtime}"
                    cached_data = await self.cache.get(cache_key)
                    if cached_data is not None:
                        self.logger.debug(f"📦 Loaded from cache: {file_path.name}")
                        self.performance_monitor.end_operation(operation_id, True)
                        return cached_data
                
                # Determine file format
                if file_format == 'auto':
                    if file_path.suffix.lower() == '.parquet':
                        file_format = 'parquet'
                    elif file_path.suffix.lower() == '.csv':
                        file_format = 'csv'
                    elif file_path.suffix.lower() == '.json':
                        file_format = 'json'
                    else:
                        raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
                # Load file in thread pool
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    self.executor,
                    self._load_file_sync,
                    file_path,
                    file_format
                )
                
                # Cache the result
                if self.cache:
                    await self.cache.set(cache_key, data, 'parquet')
                
                self.performance_monitor.end_operation(operation_id, True)
                return data
                
            except Exception as e:
                self.performance_monitor.end_operation(operation_id, False, str(e))
                raise
    
    def _load_file_sync(self, file_path: Path, file_format: str) -> pd.DataFrame:
        """Synchronous file loading (runs in thread pool)."""
        if file_format == 'parquet':
            return pd.read_parquet(file_path)
        elif file_format == 'csv':
            return pd.read_csv(file_path)
        elif file_format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def load_files_parallel(self, file_paths: List[Path], 
                                 file_formats: Optional[List[str]] = None) -> List[pd.DataFrame]:
        """Load multiple files in parallel."""
        self.logger.info(f"📁 Loading {len(file_paths)} files in parallel...")
        
        if file_formats is None:
            file_formats = ['auto'] * len(file_paths)
        
        # Create tasks for parallel loading
        tasks = []
        for file_path, file_format in zip(file_paths, file_formats):
            task = self.load_file_async(file_path, file_format)
            tasks.append(task)
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from exceptions
            successful_results = []
            failed_files = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_files.append((file_paths[i], str(result)))
                    self.logger.error(f"❌ Failed to load {file_paths[i]}: {result}")
                else:
                    successful_results.append(result)
            
            if failed_files:
                self.logger.warning(f"⚠️ {len(failed_files)} files failed to load")
                # Optionally raise exception if too many files failed
                if len(failed_files) > len(file_paths) * 0.5:  # More than 50% failed
                    raise RuntimeError(f"Too many files failed to load: {len(failed_files)}/{len(file_paths)}")
            
            self.logger.info(f"✅ Successfully loaded {len(successful_results)}/{len(file_paths)} files")
            return successful_results
            
        except Exception as e:
            self.logger.error(f"❌ Parallel file loading failed: {e}")
            raise
    
    async def save_file_async(self, data: Any, file_path: Path, 
                            file_format: str = 'parquet', 
                            compression: Optional[str] = None) -> None:
        """Save data to file asynchronously."""
        operation_id = f"save_{file_path.name}_{int(time.time())}"
        
        # Estimate file size (rough approximation)
        if isinstance(data, pd.DataFrame):
            file_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
        else:
            file_size_mb = 1.0  # Default estimate
        
        self.performance_monitor.start_operation(operation_id, f"save_{file_path.name}", file_size_mb)
        
        async with self._semaphore:
            try:
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save file in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self._save_file_sync,
                    data,
                    file_path,
                    file_format,
                    compression
                )
                
                self.performance_monitor.end_operation(operation_id, True)
                self.logger.info(f"💾 Saved file: {file_path}")
                
            except Exception as e:
                self.performance_monitor.end_operation(operation_id, False, str(e))
                raise
    
    def _save_file_sync(self, data: Any, file_path: Path, 
                       file_format: str, compression: Optional[str]) -> None:
        """Synchronous file saving (runs in thread pool)."""
        if file_format == 'parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, compression=compression)
            else:
                raise ValueError("Parquet format only supports DataFrame data")
        elif file_format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, compression=compression)
            else:
                raise ValueError("CSV format only supports DataFrame data")
        elif file_format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def save_files_parallel(self, data_files: List[Tuple[Any, Path]], 
                                 file_formats: Optional[List[str]] = None,
                                 compressions: Optional[List[str]] = None) -> None:
        """Save multiple files in parallel."""
        self.logger.info(f"💾 Saving {len(data_files)} files in parallel...")
        
        if file_formats is None:
            file_formats = ['parquet'] * len(data_files)
        if compressions is None:
            compressions = [None] * len(data_files)
        
        # Create tasks for parallel saving
        tasks = []
        for (data, file_path), file_format, compression in zip(data_files, file_formats, compressions):
            task = self.save_file_async(data, file_path, file_format, compression)
            tasks.append(task)
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for failures
            failed_saves = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_saves.append((data_files[i][1], str(result)))
                    self.logger.error(f"❌ Failed to save {data_files[i][1]}: {result}")
            
            if failed_saves:
                self.logger.warning(f"⚠️ {len(failed_saves)} files failed to save")
                if len(failed_saves) > len(data_files) * 0.5:  # More than 50% failed
                    raise RuntimeError(f"Too many files failed to save: {len(failed_saves)}/{len(data_files)}")
            
            successful_saves = len(data_files) - len(failed_saves)
            self.logger.info(f"✅ Successfully saved {successful_saves}/{len(data_files)} files")
            
        except Exception as e:
            self.logger.error(f"❌ Parallel file saving failed: {e}")
            raise
    
    async def process_data_parallel(self, data_list: List[pd.DataFrame], 
                                  process_func: Callable[[pd.DataFrame], Any],
                                  max_workers: Optional[int] = None) -> List[Any]:
        """Process multiple DataFrames in parallel."""
        if max_workers is None:
            max_workers = self.config.max_workers
        
        self.logger.info(f"⚡ Processing {len(data_list)} datasets in parallel with {max_workers} workers...")
        
        # Create tasks for parallel processing
        tasks = []
        for i, data in enumerate(data_list):
            task = self._process_data_async(data, process_func, f"dataset_{i}")
            tasks.append(task)
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from exceptions
            successful_results = []
            failed_processing = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_processing.append((i, str(result)))
                    self.logger.error(f"❌ Failed to process dataset {i}: {result}")
                else:
                    successful_results.append(result)
            
            if failed_processing:
                self.logger.warning(f"⚠️ {len(failed_processing)} datasets failed to process")
                if len(failed_processing) > len(data_list) * 0.5:  # More than 50% failed
                    raise RuntimeError(f"Too many datasets failed to process: {len(failed_processing)}/{len(data_list)}")
            
            self.logger.info(f"✅ Successfully processed {len(successful_results)}/{len(data_list)} datasets")
            return successful_results
            
        except Exception as e:
            self.logger.error(f"❌ Parallel data processing failed: {e}")
            raise
    
    async def _process_data_async(self, data: pd.DataFrame, 
                                process_func: Callable[[pd.DataFrame], Any],
                                operation_name: str) -> Any:
        """Process a single DataFrame asynchronously."""
        operation_id = f"process_{operation_name}_{int(time.time())}"
        file_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
        
        self.performance_monitor.start_operation(operation_id, f"process_{operation_name}", file_size_mb)
        
        try:
            # Run CPU-intensive processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                process_func,
                data
            )
            
            self.performance_monitor.end_operation(operation_id, True)
            return result
            
        except Exception as e:
            self.performance_monitor.end_operation(operation_id, False, str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        if self.cache:
            await self.cache.clear()
        self.logger.info("🧹 Parallel I/O operations cleaned up")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'io_performance': self.performance_monitor.get_performance_summary(),
            'config': {
                'max_concurrent_files': self.config.max_concurrent_files,
                'max_workers': self.config.max_workers,
                'chunk_size_mb': self.config.chunk_size_mb,
                'enable_caching': self.config.enable_caching,
                'enable_compression': self.config.enable_compression
            }
        }

# Global instance
_parallel_io_operations = None

def get_parallel_io_operations(config: Optional[IOConfig] = None) -> ParallelIOOperations:
    """Get or create global parallel I/O operations instance."""
    global _parallel_io_operations
    if _parallel_io_operations is None:
        _parallel_io_operations = ParallelIOOperations(config)
    return _parallel_io_operations