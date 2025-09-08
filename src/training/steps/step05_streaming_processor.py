"""
Step05 Streaming/Chunked Processing Module

This module provides streaming and chunked processing capabilities for Step05 labeling
operations, enabling efficient processing of large datasets with comprehensive logging.
"""

import pandas as pd
import numpy as np
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Iterator, Generator
from dataclasses import dataclass, field
from pathlib import Path
import logging
import gc

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates

logger = system_logger.getChild('Step05StreamingProcessor')


@dataclass
class ChunkProcessingStats:
    """Statistics for chunk processing operations."""
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_rows: int = 0
    processed_rows: int = 0
    total_computation_time: float = 0.0
    avg_chunk_time: float = 0.0
    memory_usage_peak: float = 0.0
    memory_usage_avg: float = 0.0


@dataclass
class StreamingConfig:
    """Configuration for streaming processing."""
    chunk_size: int = 10000
    max_memory_mb: float = 1000.0
    overlap_rows: int = 100
    enable_compression: bool = True
    enable_parallel_processing: bool = False
    max_workers: int = 4
    progress_reporting_interval: int = 10


class Step05StreamingProcessor:
    """Streaming processor for large dataset handling with comprehensive logging."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.streaming_config = StreamingConfig()
        self.processing_stats = ChunkProcessingStats()
        self._load_streaming_config()
        
        self.logger.info("🚀 Initializing Step05 Streaming Processor")
        self.logger.info(f"📊 Chunk size: {self.streaming_config.chunk_size:,} rows")
        self.logger.info(f"💾 Max memory: {self.streaming_config.max_memory_mb:.0f} MB")
        self.logger.info(f"🔄 Overlap rows: {self.streaming_config.overlap_rows}")
        self.logger.info(f"🗜️ Compression: {'Enabled' if self.streaming_config.enable_compression else 'Disabled'}")
        self.logger.info(f"⚡ Parallel processing: {'Enabled' if self.streaming_config.enable_parallel_processing else 'Disabled'}")
    
    def _load_streaming_config(self):
        """Load streaming configuration from config."""
        if 'streaming' in self.config:
            stream_config = self.config['streaming']
            self.streaming_config = StreamingConfig(
                chunk_size=stream_config.get('chunk_size', 10000),
                max_memory_mb=stream_config.get('max_memory_mb', 1000.0),
                overlap_rows=stream_config.get('overlap_rows', 100),
                enable_compression=stream_config.get('enable_compression', True),
                enable_parallel_processing=stream_config.get('enable_parallel_processing', False),
                max_workers=stream_config.get('max_workers', 4),
                progress_reporting_interval=stream_config.get('progress_reporting_interval', 10)
            )
            self.logger.info("✅ Streaming configuration loaded")
    
    def _log_memory_usage(self, operation_name: str):
        """Log current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            if memory_mb > self.processing_stats.memory_usage_peak:
                self.processing_stats.memory_usage_peak = memory_mb
            
            self.processing_stats.memory_usage_avg = (
                (self.processing_stats.memory_usage_avg * self.processing_stats.processed_chunks + memory_mb) /
                (self.processing_stats.processed_chunks + 1)
            )
            
            self.logger.debug(f"💾 Memory usage for {operation_name}: {memory_mb:.1f} MB")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not log memory usage: {e}")
    
    def _check_memory_availability(self) -> bool:
        """Check if sufficient memory is available for processing."""
        try:
            memory_info = psutil.virtual_memory()
            available_memory_gb = memory_info.available / (1024**3)
            required_memory_gb = self.streaming_config.max_memory_mb / 1024
            
            if available_memory_gb < required_memory_gb:
                self.logger.error(f"❌ Insufficient memory: {available_memory_gb:.1f}GB available < {required_memory_gb:.1f}GB required")
                return False
            
            self.logger.info(f"✅ Memory check passed: {available_memory_gb:.1f}GB available")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Memory check failed: {e}")
            return False
    
    @traced(span_name='load_data_streaming')
    @validates()
    @handles_errors()
    def load_data_streaming(self, file_path: Path, 
                           chunk_size: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Load data in streaming chunks with comprehensive logging.
        
        Args:
            file_path: Path to the data file
            chunk_size: Optional chunk size override
            
        Yields:
            DataFrame chunks
        """
        start_time = time.time()
        self.logger.info(f"📁 Starting streaming data load from: {file_path}")
        
        try:
            # Fast fail validation
            if not file_path.exists():
                self.logger.error(f"❌ FAST FAIL: File does not exist: {file_path}")
                return
            
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            self.logger.info(f"📏 File size: {file_size_mb:.2f} MB")
            
            # Check memory availability
            if not self._check_memory_availability():
                return
            
            # Determine chunk size
            effective_chunk_size = chunk_size or self.streaming_config.chunk_size
            
            # Adjust chunk size based on file size
            if file_size_mb > 1000:  # Large file
                effective_chunk_size = min(effective_chunk_size, 5000)
                self.logger.info(f"📊 Large file detected, reducing chunk size to {effective_chunk_size:,}")
            
            self.logger.info(f"🔄 Processing with chunk size: {effective_chunk_size:,} rows")
            
            # Load data in chunks
            chunk_count = 0
            total_rows = 0
            
            try:
                # Try to read as parquet first
                if file_path.suffix.lower() == '.parquet':
                    chunk_iterator = pd.read_parquet(file_path, chunksize=effective_chunk_size)
                else:
                    # Fallback to CSV
                    chunk_iterator = pd.read_csv(file_path, chunksize=effective_chunk_size)
                
                for chunk in chunk_iterator:
                    chunk_count += 1
                    chunk_rows = len(chunk)
                    total_rows += chunk_rows
                    
                    self.logger.info(f"📦 Processing chunk {chunk_count}: {chunk_rows:,} rows")
                    self.logger.info(f"📊 Chunk shape: {chunk.shape}")
                    self.logger.info(f"📋 Chunk columns: {list(chunk.columns)}")
                    
                    # Log chunk memory usage
                    chunk_memory = chunk.memory_usage(deep=True).sum() / (1024**2)
                    self.logger.info(f"💾 Chunk memory usage: {chunk_memory:.1f} MB")
                    
                    # Update processing stats
                    self.processing_stats.total_chunks += 1
                    self.processing_stats.total_rows += chunk_rows
                    
                    self._log_memory_usage(f"chunk_{chunk_count}")
                    
                    # Yield chunk for processing
                    yield chunk
                    
                    # Force garbage collection for large chunks
                    if chunk_memory > 100:  # >100MB
                        gc.collect()
                        self.logger.debug("🗑️ Forced garbage collection after large chunk")
                    
                    # Progress reporting
                    if chunk_count % self.streaming_config.progress_reporting_interval == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_chunk = elapsed_time / chunk_count
                        estimated_total_time = avg_time_per_chunk * (file_size_mb / (chunk_memory * chunk_count))
                        
                        self.logger.info(f"📈 Progress: {chunk_count} chunks, {total_rows:,} rows processed")
                        self.logger.info(f"⏱️ Elapsed: {elapsed_time:.1f}s, Avg per chunk: {avg_time_per_chunk:.2f}s")
                        self.logger.info(f"🔮 Estimated total time: {estimated_total_time:.1f}s")
                
                elapsed_time = time.time() - start_time
                self.logger.info(f"✅ Streaming data load completed in {elapsed_time:.1f}s")
                self.logger.info(f"📊 Total: {chunk_count} chunks, {total_rows:,} rows")
                self.logger.info(f"📈 Average chunk size: {total_rows/chunk_count:.0f} rows")
                
            except Exception as e:
                self.logger.error(f"❌ Error during streaming load: {e}")
                self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
                return
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Streaming data load failed after {elapsed_time:.1f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return
    
    @traced(span_name='process_chunks_streaming')
    @validates()
    @handles_errors()
    def process_chunks_streaming(self, chunk_generator: Generator[pd.DataFrame, None, None],
                               processing_function: callable,
                               output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process data chunks in streaming fashion with comprehensive logging.
        
        Args:
            chunk_generator: Generator yielding DataFrame chunks
            processing_function: Function to process each chunk
            output_path: Optional path to save processed chunks
            
        Returns:
            List of processing results for each chunk
        """
        start_time = time.time()
        self.logger.info("🔄 Starting streaming chunk processing...")
        
        results = []
        chunk_count = 0
        
        try:
            for chunk in chunk_generator:
                chunk_start_time = time.time()
                chunk_count += 1
                
                self.logger.info(f"⚙️ Processing chunk {chunk_count}...")
                self.logger.info(f"📊 Chunk shape: {chunk.shape}")
                
                try:
                    # Process chunk
                    chunk_result = processing_function(chunk)
                    
                    # Calculate chunk processing time
                    chunk_time = time.time() - chunk_start_time
                    
                    # Update stats
                    self.processing_stats.processed_chunks += 1
                    self.processing_stats.processed_rows += len(chunk)
                    self.processing_stats.total_computation_time += chunk_time
                    self.processing_stats.avg_chunk_time = (
                        self.processing_stats.total_computation_time / 
                        self.processing_stats.processed_chunks
                    )
                    
                    # Log chunk results
                    self.logger.info(f"✅ Chunk {chunk_count} processed in {chunk_time:.3f}s")
                    self.logger.info(f"📊 Chunk result type: {type(chunk_result).__name__}")
                    
                    if isinstance(chunk_result, dict):
                        self.logger.info(f"📋 Chunk result keys: {list(chunk_result.keys())}")
                    elif hasattr(chunk_result, 'shape'):
                        self.logger.info(f"📊 Chunk result shape: {chunk_result.shape}")
                    
                    # Store result
                    result = {
                        'chunk_id': chunk_count,
                        'chunk_size': len(chunk),
                        'processing_time': chunk_time,
                        'result': chunk_result,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    # Save chunk if output path provided
                    if output_path:
                        self._save_chunk_result(chunk_result, output_path, chunk_count)
                    
                    self._log_memory_usage(f"processed_chunk_{chunk_count}")
                    
                    # Force garbage collection
                    del chunk_result
                    gc.collect()
                    
                except Exception as e:
                    chunk_time = time.time() - chunk_start_time
                    self.processing_stats.failed_chunks += 1
                    
                    self.logger.error(f"❌ Chunk {chunk_count} processing failed after {chunk_time:.3f}s: {e}")
                    self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
                    import traceback
                    self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
                    
                    # Store error result
                    error_result = {
                        'chunk_id': chunk_count,
                        'chunk_size': len(chunk),
                        'processing_time': chunk_time,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
            
            elapsed_time = time.time() - start_time
            
            # Log final statistics
            self.logger.info(f"✅ Streaming chunk processing completed in {elapsed_time:.1f}s")
            self.logger.info(f"📊 Processing statistics:")
            self.logger.info(f"   Total chunks: {self.processing_stats.total_chunks}")
            self.logger.info(f"   Processed chunks: {self.processing_stats.processed_chunks}")
            self.logger.info(f"   Failed chunks: {self.processing_stats.failed_chunks}")
            self.logger.info(f"   Total rows: {self.processing_stats.total_rows:,}")
            self.logger.info(f"   Processed rows: {self.processing_stats.processed_rows:,}")
            self.logger.info(f"   Success rate: {self.processing_stats.processed_chunks/max(1, self.processing_stats.total_chunks)*100:.1f}%")
            self.logger.info(f"   Average chunk time: {self.processing_stats.avg_chunk_time:.3f}s")
            self.logger.info(f"   Peak memory usage: {self.processing_stats.memory_usage_peak:.1f} MB")
            self.logger.info(f"   Average memory usage: {self.processing_stats.memory_usage_avg:.1f} MB")
            
            return results
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Streaming chunk processing failed after {elapsed_time:.1f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return results
    
    def _save_chunk_result(self, chunk_result: Any, output_path: Path, chunk_id: int):
        """Save chunk processing result to file."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(chunk_result, pd.DataFrame):
                chunk_file = output_path / f"chunk_{chunk_id:06d}.parquet"
                chunk_result.to_parquet(chunk_file, compression='snappy' if self.streaming_config.enable_compression else None)
                self.logger.debug(f"💾 Saved chunk {chunk_id} to {chunk_file}")
            
            elif isinstance(chunk_result, dict):
                import json
                chunk_file = output_path / f"chunk_{chunk_id:06d}.json"
                with open(chunk_file, 'w') as f:
                    json.dump(chunk_result, f, indent=2, default=str)
                self.logger.debug(f"💾 Saved chunk {chunk_id} to {chunk_file}")
            
            else:
                self.logger.warning(f"⚠️ Unsupported chunk result type for saving: {type(chunk_result)}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save chunk {chunk_id}: {e}")
    
    @traced(span_name='merge_chunk_results')
    @validates()
    @handles_errors()
    def merge_chunk_results(self, results: List[Dict[str, Any]], 
                          merge_function: Optional[callable] = None) -> Any:
        """
        Merge chunk processing results with comprehensive logging.
        
        Args:
            results: List of chunk processing results
            merge_function: Optional custom merge function
            
        Returns:
            Merged result
        """
        start_time = time.time()
        self.logger.info("🔗 Starting chunk results merging...")
        
        try:
            if not results:
                self.logger.warning("⚠️ No results to merge")
                return None
            
            successful_results = [r for r in results if 'error' not in r]
            failed_results = [r for r in results if 'error' in r]
            
            self.logger.info(f"📊 Merging {len(successful_results)} successful results")
            if failed_results:
                self.logger.warning(f"⚠️ {len(failed_results)} failed results will be excluded")
            
            if not successful_results:
                self.logger.error("❌ No successful results to merge")
                return None
            
            # Default merge function for DataFrames
            if merge_function is None:
                def default_merge_function(results):
                    dataframes = [r['result'] for r in results if isinstance(r['result'], pd.DataFrame)]
                    if dataframes:
                        return pd.concat(dataframes, ignore_index=True)
                    else:
                        # Merge dictionaries
                        merged_dict = {}
                        for result in results:
                            if isinstance(result['result'], dict):
                                merged_dict.update(result['result'])
                        return merged_dict
                
                merge_function = default_merge_function
            
            # Merge results
            merged_result = merge_function(successful_results)
            
            elapsed_time = time.time() - start_time
            
            # Log merge results
            self.logger.info(f"✅ Chunk results merging completed in {elapsed_time:.3f}s")
            
            if isinstance(merged_result, pd.DataFrame):
                self.logger.info(f"📊 Merged DataFrame shape: {merged_result.shape}")
                self.logger.info(f"📋 Merged DataFrame columns: {list(merged_result.columns)}")
                memory_usage = merged_result.memory_usage(deep=True).sum() / (1024**2)
                self.logger.info(f"💾 Merged DataFrame memory usage: {memory_usage:.1f} MB")
            
            elif isinstance(merged_result, dict):
                self.logger.info(f"📋 Merged dictionary keys: {list(merged_result.keys())}")
            
            else:
                self.logger.info(f"📊 Merged result type: {type(merged_result).__name__}")
            
            return merged_result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Chunk results merging failed after {elapsed_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return None
    
    @traced(span_name='process_large_file_streaming')
    @validates()
    @handles_errors()
    def process_large_file_streaming(self, file_path: Path,
                                   processing_function: callable,
                                   output_path: Optional[Path] = None,
                                   chunk_size: Optional[int] = None) -> Any:
        """
        Process a large file using streaming with comprehensive logging.
        
        Args:
            file_path: Path to the input file
            processing_function: Function to process each chunk
            output_path: Optional path to save results
            chunk_size: Optional chunk size override
            
        Returns:
            Processed result
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting large file streaming processing: {file_path}")
        
        try:
            # Load data in streaming chunks
            chunk_generator = self.load_data_streaming(file_path, chunk_size)
            
            # Process chunks
            results = self.process_chunks_streaming(
                chunk_generator, 
                processing_function, 
                output_path
            )
            
            # Merge results
            merged_result = self.merge_chunk_results(results)
            
            elapsed_time = time.time() - start_time
            
            # Log final results
            self.logger.info(f"✅ Large file streaming processing completed in {elapsed_time:.1f}s")
            self.logger.info(f"📊 Final processing statistics:")
            self.logger.info(f"   Total processing time: {elapsed_time:.1f}s")
            self.logger.info(f"   Chunks processed: {self.processing_stats.processed_chunks}")
            self.logger.info(f"   Rows processed: {self.processing_stats.processed_rows:,}")
            self.logger.info(f"   Peak memory usage: {self.processing_stats.memory_usage_peak:.1f} MB")
            self.logger.info(f"   Average memory usage: {self.processing_stats.memory_usage_avg:.1f} MB")
            
            return merged_result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Large file streaming processing failed after {elapsed_time:.1f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processing_stats': {
                'total_chunks': self.processing_stats.total_chunks,
                'processed_chunks': self.processing_stats.processed_chunks,
                'failed_chunks': self.processing_stats.failed_chunks,
                'total_rows': self.processing_stats.total_rows,
                'processed_rows': self.processing_stats.processed_rows,
                'total_computation_time': self.processing_stats.total_computation_time,
                'avg_chunk_time': self.processing_stats.avg_chunk_time,
                'memory_usage_peak': self.processing_stats.memory_usage_peak,
                'memory_usage_avg': self.processing_stats.memory_usage_avg
            },
            'success_rate': (
                self.processing_stats.processed_chunks / 
                max(1, self.processing_stats.total_chunks)
            ),
            'throughput': (
                self.processing_stats.processed_rows / 
                max(1, self.processing_stats.total_computation_time)
            )
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = ChunkProcessingStats()
        self.logger.info("🔄 Processing statistics reset")
    
    def optimize_chunk_size(self, file_path: Path, sample_size: int = 1000) -> int:
        """Optimize chunk size based on file characteristics."""
        try:
            self.logger.info(f"🔍 Optimizing chunk size for: {file_path}")
            
            # Sample the file to determine optimal chunk size
            sample = pd.read_parquet(file_path, nrows=sample_size)
            sample_memory = sample.memory_usage(deep=True).sum() / (1024**2)
            
            # Calculate rows per MB
            rows_per_mb = sample_size / sample_memory
            
            # Target chunk size based on memory limit
            target_memory_mb = self.streaming_config.max_memory_mb * 0.5  # Use 50% of limit
            optimal_chunk_size = int(rows_per_mb * target_memory_mb)
            
            # Apply reasonable bounds
            optimal_chunk_size = max(1000, min(optimal_chunk_size, 50000))
            
            self.logger.info(f"📊 Sample analysis: {sample_size} rows = {sample_memory:.1f} MB")
            self.logger.info(f"📈 Rows per MB: {rows_per_mb:.0f}")
            self.logger.info(f"🎯 Optimal chunk size: {optimal_chunk_size:,} rows")
            
            return optimal_chunk_size
            
        except Exception as e:
            self.logger.error(f"❌ Chunk size optimization failed: {e}")
            return self.streaming_config.chunk_size