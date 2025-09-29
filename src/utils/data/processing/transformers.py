"""Data streaming and transformation utilities for handling large datasets efficiently."""
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Generator, Callable
from pathlib import Path

from datetime import datetime, timedelta
import gc
import psutil
import os
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
import logging
import numpy as np
import time

class DataStreamingManager:
    """Manages data streaming and chunking for large datasets."""
    
    _instance = None
    _initialized = False

    def __new__(cls, chunk_size: int = 10000, memory_threshold: float = 0.8, overlap_size: int = 100, enable_compression: bool = True):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(DataStreamingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, chunk_size: int = 10000, memory_threshold: float = 0.8, overlap_size: int = 100, enable_compression: bool = True) -> None:
        """
        Initialize data streaming manager (only once due to singleton).
        
        Args:
            chunk_size: Number of rows per chunk
            memory_threshold: Memory usage threshold (0.0-1.0)
            overlap_size: Number of overlapping rows between chunks
            enable_compression: Enable data compression for storage
        """
        if self._initialized:
            return
        
        start_time = time.time()
        self.logger = system_logger.getChild('DataStreamingManager')
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold
        self.overlap_size = overlap_size
        self.enable_compression = enable_compression
        self.standards = PipelineStandards(self.logger)
        self.performance_metrics = {'chunks_processed': 0, 'total_rows_processed': 0, 'memory_usage_peak': 0.0, 'processing_time_total': 0.0, 'compression_ratio': 0.0}
        self.logger.info(f'🚀 DataStreamingManager initialized (singleton): chunk_size={chunk_size}, memory_threshold={memory_threshold}')
        self._initialized = True
        
        # Add timing information
        duration = time.time() - start_time
        try:
            from src.utils.tprint import tprint_performance
            tprint_performance("DataStreamingManager initialization", duration)
        except ImportError:
            self.logger.info(f"⏱️ DataStreamingManager initialized in {duration:.3f}s")


class DataTransformer:
    """Data transformation utilities for feature engineering."""

    def __init__(self):
        """Initialize the DataTransformer."""
        self.logger = system_logger.getChild('DataTransformer')

    def transform_features(self, data: pd.DataFrame, transformations: List[str] = None) -> pd.DataFrame:
        """Apply transformations to features.

        Args:
            data: Input DataFrame
            transformations: List of transformation types

        Returns:
            Transformed DataFrame
        """
        if transformations is None:
            transformations = ['normalize', 'standardize']

        transformed_data = data.copy()

        for transformation in transformations:
            try:
                if transformation == 'normalize':
                    transformed_data = self._normalize_features(transformed_data)
                elif transformation == 'standardize':
                    transformed_data = self._standardize_features(transformed_data)
                elif transformation == 'log':
                    transformed_data = self._log_transform_features(transformed_data)
                elif transformation == 'sqrt':
                    transformed_data = self._sqrt_transform_features(transformed_data)
                else:
                    self.logger.warning(f"Unknown transformation: {transformation}")
            except Exception as e:
                self.logger.warning(f"Failed to apply transformation {transformation}: {e}")

        self.logger.info(f"Applied {len(transformations)} transformations to {len(data.columns)} features")
        return transformed_data

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range."""
        normalized_data = data.copy()
        for col in data.select_dtypes(include=[np.number]).columns:
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
        return normalized_data

    def _standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize features to mean=0, std=1."""
        standardized_data = data.copy()
        for col in data.select_dtypes(include=[np.number]).columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
        return standardized_data

    def _log_transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to features."""
        log_data = data.copy()
        for col in data.select_dtypes(include=[np.number]).columns:
            if (data[col] > 0).all():
                log_data[col] = np.log(data[col])
        return log_data

    def _sqrt_transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply square root transformation to features."""
        sqrt_data = data.copy()
        for col in data.select_dtypes(include=[np.number]).columns:
            if (data[col] >= 0).all():
                sqrt_data[col] = np.sqrt(data[col])
        return sqrt_data

    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            return memory_info.rss / system_memory.total
        except Exception as e:
            self.logger.warning(f'⚠️ Could not get memory usage: {e}')
            return 0.0

    def should_chunk_data(self, data: pd.DataFrame) -> bool:
        """
        Determine if data should be chunked based on size and memory usage.
        
        Args:
            data: DataFrame to evaluate
            
        Returns:
            True if data should be chunked
        """
        current_memory = self.get_memory_usage()
        if current_memory > self.memory_threshold:
            self.logger.info(f'📊 Memory usage high ({current_memory:.2%}), chunking recommended')
            return True
        if len(data) > self.chunk_size * 2:
            self.logger.info(f'📊 Large dataset ({len(data)} rows), chunking recommended')
            return True
        memory_footprint = data.memory_usage(deep = True).sum() / 1024 ** 3
        if memory_footprint > 1.0:
            self.logger.info(f'📊 Large memory footprint ({memory_footprint:.2f} GB), chunking recommended')
            return True
        return False

    def create_data_chunks(self, data: pd.DataFrame, preserve_order: bool = True, time_based_chunking: bool = True) -> Generator[pd.DataFrame, None, None]:
        """
        Create data chunks with optional overlap and time-based chunking.
        
        Args:
            data: DataFrame to chunk
            preserve_order: Whether to preserve temporal order
            time_based_chunking: Use time-based chunking if timestamp column exists
            
        Yields:
            DataFrame chunks
        """
        self.logger.info(f'🔪 Creating chunks from {len(data)} rows...')
        if not self.should_chunk_data(data):
            self.logger.info('📊 Data size acceptable, returning as single chunk')
            yield data
            return
        if time_based_chunking and 'timestamp' in data.columns:
            yield from self._create_time_based_chunks(data, preserve_order)
        else:
            yield from self._create_size_based_chunks(data, preserve_order)

    def _create_time_based_chunks(self, data: pd.DataFrame, preserve_order: bool) -> Generator[pd.DataFrame, None, None]:
        """Create chunks based on time intervals."""
        self.logger.info('⏰ Using time-based chunking...')
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        if preserve_order and (not data['timestamp'].is_monotonic_increasing):
            data = data.sort_values('timestamp').reset_index(drop = True)
        total_time_span = data['timestamp'].max() - data['timestamp'].min()
        time_per_chunk = total_time_span / (len(data) / self.chunk_size)
        current_time = data['timestamp'].min()
        chunk_index = 0
        while current_time < data['timestamp'].max():
            chunk_end_time = current_time + time_per_chunk
            chunk_mask = (data['timestamp'] >= current_time) & (data['timestamp'] < chunk_end_time)
            chunk_data = data[chunk_mask].copy()
            if len(chunk_data) > 0:
                if chunk_index > 0 and self.overlap_size > 0:
                    overlap_start = current_time - timedelta(seconds = self.overlap_size * 60)
                    overlap_mask = (data['timestamp'] >= overlap_start) & (data['timestamp'] < current_time)
                    overlap_data = data[overlap_mask].copy()
                    if len(overlap_data) > 0:
                        chunk_data = pd.concat([overlap_data, chunk_data], ignore_index = True)
                        chunk_data = chunk_data.drop_duplicates(subset=['timestamp'], keep='last')
                self.logger.info(f"📦 Created time-based chunk {chunk_index + 1}: {len(chunk_data)} rows, time range: {chunk_data['timestamp'].min()} to {chunk_data['timestamp'].max()}")
                yield chunk_data
                chunk_index += 1
            current_time = chunk_end_time

    def _create_size_based_chunks(self, data: pd.DataFrame, preserve_order: bool) -> Generator[pd.DataFrame, None, None]:
        """Create chunks based on row count."""
        self.logger.info('📊 Using size-based chunking...')
        total_rows = len(data)
        chunk_index = 0
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk_data = data.iloc[start_idx:end_idx].copy()
            if chunk_index > 0 and self.overlap_size > 0:
                overlap_start = max(0, start_idx - self.overlap_size)
                overlap_data = data.iloc[overlap_start:start_idx].copy()
                if len(overlap_data) > 0:
                    chunk_data = pd.concat([overlap_data, chunk_data], ignore_index = True)
                    if 'timestamp' in chunk_data.columns:
                        chunk_data = chunk_data.drop_duplicates(subset=['timestamp'], keep='last')
            self.logger.info(f'📦 Created size-based chunk {chunk_index + 1}: {len(chunk_data)} rows (rows {start_idx}-{end_idx - 1})')
            yield chunk_data
            chunk_index += 1

    def process_large_dataset(self, data: pd.DataFrame, processing_func: Callable[[pd.DataFrame], pd.DataFrame], combine_results: bool = True, progress_callback: Optional[Callable[[int, int], None]]=None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Process large dataset in chunks with memory management.
        
        Args:
            data: DataFrame to process
            processing_func: Function to apply to each chunk
            combine_results: Whether to combine results into single DataFrame
            progress_callback: Optional progress callback function
            
        Returns:
            Processed DataFrame or list of processed chunks
        """
        start_time = datetime.now()
        self.logger.info(f'🔄 Processing large dataset: {len(data)} rows...')
        processed_chunks = []
        chunk_count = 0
        try:
            for chunk in self.create_data_chunks(data):
                chunk_count += 1
                self.logger.info(f'⚙️ Processing chunk {chunk_count}...')
                try:
                    processed_chunk = processing_func(chunk)
                    processed_chunks.append(processed_chunk)
                    self.performance_metrics['chunks_processed'] += 1
                    self.performance_metrics['total_rows_processed'] += len(processed_chunk)
                    current_memory = self.get_memory_usage()
                    self.performance_metrics['memory_usage_peak'] = max(self.performance_metrics['memory_usage_peak'], current_memory)
                    if progress_callback:
                        progress_callback(chunk_count, len(processed_chunks))
                    if current_memory > self.memory_threshold * 0.9:
                        self.logger.info('🧹 High memory usage, triggering garbage collection...')
                        gc.collect()
                except Exception as e:
                    self.logger.error(f'❌ Error processing chunk {chunk_count}: {e}')
                    raise
            if combine_results and processed_chunks:
                self.logger.info('🔗 Combining processed chunks...')
                if self.overlap_size > 0 and 'timestamp' in processed_chunks[0].columns:
                    combined_data = self._combine_chunks_without_overlap(processed_chunks)
                else:
                    combined_data = pd.concat(processed_chunks, ignore_index = True)
                combined_data = self.standards.validate_data_quality(combined_data, 'unified')
                self.logger.info(f'✅ Dataset processing completed: {len(combined_data)} rows')
                return combined_data
            else:
                self.logger.info(f'✅ Dataset processing completed: {len(processed_chunks)} chunks')
                return processed_chunks
        except Exception as e:
            self.logger.exception(f'❌ Error processing large dataset: {e}')
            raise
        finally:
            end_time = datetime.now()
            self.performance_metrics['processing_time_total'] = (end_time - start_time).total_seconds()
            self._log_performance_summary()

    def _combine_chunks_without_overlap(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine chunks while removing overlaps."""
        if not chunks:
            return pd.DataFrame()
        if len(chunks) == 1:
            return chunks[0]
        if 'timestamp' in chunks[0].columns:
            chunks = sorted(chunks, key = lambda x: x['timestamp'].min())
        combined_data = chunks[0].copy()
        for i, chunk in enumerate(chunks[1:], 1):
            if 'timestamp' in chunk.columns:
                last_timestamp = combined_data['timestamp'].max()
                non_overlap_mask = chunk['timestamp'] > last_timestamp
                non_overlap_data = chunk[non_overlap_mask]
                if len(non_overlap_data) > 0:
                    combined_data = pd.concat([combined_data, non_overlap_data], ignore_index = True)
            else:
                combined_data = pd.concat([combined_data, chunk], ignore_index = True)
        return combined_data

    def stream_data_from_file(self, file_path: Union[str, Path], chunk_size: Optional[int]=None, file_format: str='parquet') -> Generator[pd.DataFrame, None, None]:
        """
        Stream data from file in chunks.
        
        Args:
            file_path: Path to data file
            chunk_size: Override default chunk size
            file_format: File format ('parquet', 'csv', 'json')
            
        Yields:
            DataFrame chunks
        """
        chunk_size = chunk_size or self.chunk_size
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')
        self.logger.info(f'📁 Streaming data from {file_path} ({file_format})...')
        try:
            if file_format.lower() == 'parquet':
                parquet_file = pd.read_parquet(file_path, engine='pyarrow')
                yield from self.create_data_chunks(parquet_file)
            elif file_format.lower() == 'csv':
                chunk_iter = pd.read_csv(file_path, chunksize = chunk_size)
                for chunk in chunk_iter:
                    yield chunk
            elif file_format.lower() == 'json':
                json_data = pd.read_json(file_path)
                yield from self.create_data_chunks(json_data)
            else:
                raise ValueError(f'Unsupported file format: {file_format}')
        except Exception as e:
            self.logger.exception(f'❌ Error streaming data from file: {e}')
            raise

    def _log_performance_summary(self) -> None:
        """Log performance summary."""
        metrics = self.performance_metrics
        self.logger.info('📊 Data Streaming Performance Summary:')
        self.logger.info(f"   - Chunks processed: {metrics['chunks_processed']}")
        self.logger.info(f"   - Total rows processed: {metrics['total_rows_processed']:,}")
        self.logger.info(f"   - Peak memory usage: {metrics['memory_usage_peak']:.2%}")
        self.logger.info(f"   - Total processing time: {metrics['processing_time_total']:.2f}s")
        if metrics['chunks_processed'] > 0:
            avg_chunk_time = metrics['processing_time_total'] / metrics['chunks_processed']
            self.logger.info(f'   - Average chunk processing time: {avg_chunk_time:.2f}s')

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {'chunks_processed': 0, 'total_rows_processed': 0, 'memory_usage_peak': 0.0, 'processing_time_total': 0.0, 'compression_ratio': 0.0}
data_streaming_manager = DataStreamingManager()