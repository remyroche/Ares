"""
Streaming Data Processor with Memory Optimization

This module provides memory-efficient data processing using tools from src/utils/hardware/
for optimized memory management during data collection and processing.
"""

import asyncio
import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Generator
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.memory_optimization import MemoryMonitor, MemoryConfig

logger = system_logger.getChild('StreamingDataProcessor')

@dataclass
class StreamingConfig:
    """Configuration for streaming data processing."""
    chunk_size: int = 6000
    max_memory_mb: float = 2048.0
    temp_dir: Optional[str] = None
    compression: str = 'gzip'
    memory_pressure_threshold: float = 0.8
    gc_frequency: int = 100  # Trigger GC every N chunks

class StreamingDataProcessor:
    """Memory-optimized streaming data processor."""

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.logger = logger.getChild('StreamingDataProcessor')

        # Initialize memory optimization tools
        self.memory_optimizer = M1MemoryOptimizer(memory_limit_gb=self.config.max_memory_mb / 1024)
        self.memory_monitor = MemoryMonitor(MemoryConfig(
            max_memory_mb=self.config.max_memory_mb,
            warning_threshold=self.config.memory_pressure_threshold,
            critical_threshold=0.9,
            gc_threshold=0.7
        ))

        # Setup temporary directory
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "streaming_data"

        self.temp_dir.mkdir(exist_ok=True)

        # Processing statistics
        self.stats = {
            'chunks_processed': 0,
            'total_rows_processed': 0,
            'memory_cleanups': 0,
            'gc_triggers': 0,
            'peak_memory_mb': 0.0
        }

        # Start memory monitoring
        self.memory_optimizer.start_monitoring()

        self.logger.info(f"🚀 StreamingDataProcessor initialized with {self.config.chunk_size} chunk size")
        self.logger.info(f"   Memory limit: {self.config.max_memory_mb} MB")
        self.logger.info(f"   Temp directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.memory_optimizer.stop_monitoring()
        except:
            pass

    def process_dataframe_in_chunks(self,
                                   df: pd.DataFrame,
                                   processing_func: callable,
                                   output_path: Optional[str] = None) -> pd.DataFrame:
        """Process DataFrame in memory-efficient chunks."""
        self.logger.info(f"🔄 Processing DataFrame with {len(df)} rows in chunks of {self.config.chunk_size}")

        if len(df) <= self.config.chunk_size:
            # Small DataFrame, process directly
            return self._process_single_chunk(df, processing_func)

        # Large DataFrame, process in chunks
        processed_chunks = []
        temp_files = []

        try:
            for i in range(0, len(df), self.config.chunk_size):
                chunk = df.iloc[i:i + self.config.chunk_size]

                # Check memory pressure
                if self.memory_monitor.is_memory_pressure():
                    self._handle_memory_pressure()

                # Process chunk
                processed_chunk = self._process_single_chunk(chunk, processing_func)

                # Store chunk (in memory or temp file based on memory pressure)
                if self.memory_monitor.is_memory_pressure(0.7):
                    # High memory pressure, save to temp file
                    temp_file = self._save_chunk_to_temp(processed_chunk, i)
                    temp_files.append(temp_file)
                    processed_chunks.append(None)  # Placeholder
                else:
                    # Low memory pressure, keep in memory
                    processed_chunks.append(processed_chunk)

                self.stats['chunks_processed'] += 1
                self.stats['total_rows_processed'] += len(chunk)

                # Periodic garbage collection
                if self.stats['chunks_processed'] % self.config.gc_frequency == 0:
                    self._trigger_gc()

                # Update peak memory
                current_memory = self.memory_monitor.get_usage_mb()
                self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], current_memory)

            # Combine results
            if temp_files:
                # Load from temp files
                result = self._load_chunks_from_temp_files(temp_files, processed_chunks)
            else:
                # Combine in-memory chunks
                result = pd.concat(processed_chunks, ignore_index=True) if processed_chunks else pd.DataFrame()

            # Save to output if specified
            if output_path:
                self._save_result(result, output_path)

            self.logger.info(f"✅ Processing completed: {len(result)} rows processed")
            self.logger.info(f"   Peak memory usage: {self.stats['peak_memory_mb']:.2f} MB")

            return result

        finally:
            # Cleanup temp files
            self._cleanup_temp_files(temp_files)

    def _process_single_chunk(self, chunk: pd.DataFrame, processing_func: callable) -> pd.DataFrame:
        """Process a single chunk of data."""
        try:
            return processing_func(chunk)
        except Exception as e:
            self.logger.error(f"❌ Error processing chunk: {e}")
            return chunk  # Return original chunk on error

    def _handle_memory_pressure(self):
        """Handle memory pressure by applying optimizations."""
        self.logger.warning("⚠️ Memory pressure detected, applying optimizations...")

        # Trigger garbage collection
        self._trigger_gc()

        # Apply M1 memory optimizations
        self.memory_optimizer._apply_memory_optimizations()

        self.stats['memory_cleanups'] += 1

    def _trigger_gc(self):
        """Trigger aggressive garbage collection."""
        before_mb = self.memory_monitor.get_usage_mb()
        
        # Multiple rounds of garbage collection for better cleanup
        total_freed = 0
        for _ in range(3):
            gc.collect()
            current_mb = self.memory_monitor.get_usage_mb()
            freed = before_mb - current_mb
            total_freed += freed
            before_mb = current_mb
        
        if total_freed > 0:
            self.logger.info(f"🧹 Aggressive garbage collection freed {total_freed:.2f} MB")

        self.stats['gc_triggers'] += 1

    def _save_chunk_to_temp(self, chunk: pd.DataFrame, chunk_index: int) -> str:
        """Save chunk to temporary file."""
        temp_file = self.temp_dir / f"chunk_{chunk_index}.parquet"
        chunk.to_parquet(temp_file, compression=self.config.compression)
        return str(temp_file)

    def _load_chunks_from_temp_files(self, temp_files: List[str], in_memory_chunks: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
        """Load chunks from temporary files and combine with in-memory chunks."""
        all_chunks = []

        # Add in-memory chunks
        for chunk in in_memory_chunks:
            if chunk is not None:
                all_chunks.append(chunk)

        # Load from temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                chunk = pd.read_parquet(temp_file)
                all_chunks.append(chunk)

        return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

    def _save_result(self, result: pd.DataFrame, output_path: str):
        """Save result to output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.parquet':
            result.to_parquet(output_path, compression=self.config.compression)
        elif output_path.suffix == '.csv':
            result.to_csv(output_path, index=False)
        else:
            result.to_parquet(output_path.with_suffix('.parquet'), compression=self.config.compression)

        self.logger.info(f"💾 Result saved to {output_path}")

    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files."""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to cleanup temp file {temp_file}: {e}")

    def stream_large_file(self, file_path: str, processing_func: callable) -> Generator[pd.DataFrame, None, None]:
        """Stream large file in chunks for memory-efficient processing."""
        self.logger.info(f"📖 Streaming large file: {file_path}")

        try:
            # Read file in chunks
            chunk_iter = pd.read_parquet(file_path, chunksize=self.config.chunk_size)

            for chunk in chunk_iter:
                # Check memory pressure
                if self.memory_monitor.is_memory_pressure():
                    self._handle_memory_pressure()

                # Process chunk
                processed_chunk = self._process_single_chunk(chunk, processing_func)

                self.stats['chunks_processed'] += 1
                self.stats['total_rows_processed'] += len(chunk)

                yield processed_chunk

        except Exception as e:
            self.logger.error(f"❌ Error streaming file {file_path}: {e}")
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'current_memory_mb': self.memory_monitor.get_usage_mb(),
            'peak_memory_mb': self.memory_monitor.get_peak_usage_mb(),
            'memory_pressure': self.memory_monitor.is_memory_pressure(),
            'critical_memory': self.memory_monitor.is_critical_memory()
        }

# Global instance
_streaming_processor: Optional[StreamingDataProcessor] = None

def get_streaming_processor(config: Optional[StreamingConfig] = None) -> StreamingDataProcessor:
    """Get the global streaming processor instance."""
    global _streaming_processor
    if _streaming_processor is None:
        _streaming_processor = StreamingDataProcessor(config)
    return _streaming_processor

def with_memory_optimization(chunk_size: int = 3000, max_memory_mb: float = 2048.0):
    """Decorator for memory-optimized processing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            processor = get_streaming_processor(StreamingConfig(
                chunk_size=chunk_size,
                max_memory_mb=max_memory_mb
            ))

            # If first argument is a DataFrame, process it in chunks
            if args and isinstance(args[0], pd.DataFrame):
                df = args[0]
                other_args = args[1:]
                return processor.process_dataframe_in_chunks(df, lambda chunk: func(chunk, *other_args, **kwargs))
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator
