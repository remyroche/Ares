"""
Regime Data Processing Utilities

This module provides comprehensive utilities for processing data based on detected regimes,
including regime-based data splitting, validation, and analysis.

Key Features:
- Regime-based data splitting and validation
- Cross-regime data consistency checks
- Regime continuity validation
- Async file processing for large datasets
- Memory-efficient data processing
- Data type optimization
- Regime transition analysis
- Performance tracking and analytics
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiofiles
import json
import os
from pathlib import Path

from ..math_validation import safe_divide, safe_log, safe_sqrt, validate_positive, validate_range
from ..common_operations import create_fallback_logger
from ..m1_gpu_utils import M1GPUManager
from ..parallel_processing_optimizer import ParallelProcessor

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Available processing modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    CHUNKED = "chunked"

@dataclass
class RegimeProcessingConfig:
    """Configuration for regime data processing."""
    min_regime_samples: int = 100
    max_regime_samples: int = 10000
    chunk_size: int = 1000
    max_workers: int = 4
    memory_efficient: bool = True
    validate_continuity: bool = True
    validate_consistency: bool = True
    processing_mode: ProcessingMode = ProcessingMode.PARALLEL
    output_format: str = "parquet"  # parquet, csv, json
    compression: str = "snappy"  # snappy, gzip, bz2
    include_metadata: bool = True
    regime_column: str = "regime"
    timestamp_column: str = "timestamp"

@dataclass
class RegimeProcessingResult:
    """Result of regime data processing operation."""
    processed_data: Dict[str, pd.DataFrame]
    regime_statistics: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    validation_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class AsyncFileProcessor:
    """Async file processor for large datasets."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logger.getChild('AsyncFileProcessor')
    
    async def process_file_async(self, file_path: str, processor_func: Callable) -> Any:
        """Process a file asynchronously."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
                return await processor_func(content)
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    async def process_files_parallel(self, file_paths: List[str], processor_func: Callable) -> List[Any]:
        """Process multiple files in parallel."""
        tasks = [self.process_file_async(path, processor_func) for path in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)

class MemoryPoolManager:
    """Memory pool manager for efficient memory usage."""
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.pools = {}
        self.logger = logger.getChild('MemoryPoolManager')
    
    def get_pool(self, pool_name: str) -> List[Any]:
        """Get a memory pool by name."""
        if pool_name not in self.pools:
            self.pools[pool_name] = []
        return self.pools[pool_name]
    
    def return_pool(self, pool_name: str, objects: List[Any]) -> None:
        """Return objects to a memory pool."""
        pool = self.get_pool(pool_name)
        pool.extend(objects)
        
        # Limit pool size
        if len(pool) > self.pool_size:
            pool[:] = pool[-self.pool_size:]
    
    def clear_pool(self, pool_name: str) -> None:
        """Clear a memory pool."""
        if pool_name in self.pools:
            self.pools[pool_name].clear()

class DataTypeOptimizer:
    """Optimize data types for memory efficiency."""
    
    def __init__(self):
        self.logger = logger.getChild('DataTypeOptimizer')
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        optimized_df = df.copy()
        
        # Optimize integer columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            if optimized_df[col].min() >= 0:
                if optimized_df[col].max() < 255:
                    optimized_df[col] = optimized_df[col].astype('uint8')
                elif optimized_df[col].max() < 65535:
                    optimized_df[col] = optimized_df[col].astype('uint16')
                elif optimized_df[col].max() < 4294967295:
                    optimized_df[col] = optimized_df[col].astype('uint32')
            else:
                if optimized_df[col].min() > -128 and optimized_df[col].max() < 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif optimized_df[col].min() > -32768 and optimized_df[col].max() < 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif optimized_df[col].min() > -2147483648 and optimized_df[col].max() < 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = optimized_df[col].astype('float32')
        
        # Optimize object columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].dtype == 'object':
                try:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                except (ValueError, TypeError):
                    try:
                        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                    except (ValueError, TypeError):
                        # Keep as object if conversion fails
                        pass
        
        return optimized_df
    
    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get memory usage information for a DataFrame."""
        memory_usage = df.memory_usage(deep=True)
        return {
            'total_memory_mb': memory_usage.sum() / 1024 / 1024,
            'per_column_memory_mb': (memory_usage / 1024 / 1024).to_dict(),
            'dtypes': df.dtypes.to_dict()
        }

class RegimeDataProcessor:
    """
    Comprehensive regime data processing utilities.
    
    This class provides utilities for processing data based on detected regimes,
    including splitting, validation, and analysis with memory optimization.
    """
    
    def __init__(self, config: Optional[RegimeProcessingConfig] = None):
        """Initialize the regime data processor."""
        self.config = config or RegimeProcessingConfig()
        self.logger = logger.getChild('RegimeDataProcessor')
        
        # Initialize components
        self.gpu_manager = M1GPUManager()
        self.parallel_processor = ParallelProcessor(max_workers=self.config.max_workers)
        self.async_processor = AsyncFileProcessor(max_workers=self.config.max_workers)
        self.memory_manager = MemoryPoolManager()
        self.data_optimizer = DataTypeOptimizer()
        
        # Validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the processing configuration."""
        validate_positive(self.config.min_regime_samples, "min_regime_samples")
        validate_positive(self.config.max_regime_samples, "max_regime_samples")
        validate_positive(self.config.chunk_size, "chunk_size")
        validate_positive(self.config.max_workers, "max_workers")
        
        if self.config.min_regime_samples >= self.config.max_regime_samples:
            raise ValueError("min_regime_samples must be less than max_regime_samples")
    
    def process_regime_data(
        self,
        data: pd.DataFrame,
        regime_ids: np.ndarray,
        output_dir: Optional[str] = None
    ) -> RegimeProcessingResult:
        """
        Process data based on regime assignments.
        
        Args:
            data: Input data DataFrame
            regime_ids: Regime assignments for each data point
            output_dir: Optional output directory for saving processed data
            
        Returns:
            RegimeProcessingResult with processed data and metadata
        """
        self.logger.info("Processing regime data")
        
        start_time = datetime.now()
        
        try:
            # Validate inputs
            validation_results = self._validate_inputs(data, regime_ids)
            
            # Split data by regimes
            regime_data = self._split_data_by_regimes(data, regime_ids)
            
            # Process each regime
            processed_data = {}
            regime_statistics = {}
            
            for regime_id, regime_df in regime_data.items():
                self.logger.info(f"Processing regime {regime_id} with {len(regime_df)} samples")
                
                # Validate regime data
                regime_validation = self._validate_regime_data(regime_df, regime_id)
                
                # Process regime data
                processed_regime_data = self._process_single_regime(regime_df, regime_id)
                processed_data[regime_id] = processed_regime_data
                
                # Calculate regime statistics
                regime_statistics[regime_id] = self._calculate_regime_statistics(processed_regime_data, regime_id)
            
            # Save processed data if output directory specified
            if output_dir:
                self._save_processed_data(processed_data, regime_statistics, output_dir)
            
            # Calculate performance metrics
            end_time = datetime.now()
            performance_metrics = {
                'processing_time_seconds': (end_time - start_time).total_seconds(),
                'total_samples': len(data),
                'regimes_processed': len(processed_data),
                'memory_usage_mb': self._calculate_memory_usage(processed_data)
            }
            
            # Create processing metadata
            processing_metadata = {
                'config': {
                    'min_regime_samples': self.config.min_regime_samples,
                    'max_regime_samples': self.config.max_regime_samples,
                    'chunk_size': self.config.chunk_size,
                    'processing_mode': self.config.processing_mode.value,
                    'output_format': self.config.output_format
                },
                'input_info': {
                    'total_samples': len(data),
                    'total_features': len(data.columns),
                    'regime_column': self.config.regime_column,
                    'timestamp_column': self.config.timestamp_column
                },
                'processing_info': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'regimes_found': len(processed_data)
                }
            }
            
            return RegimeProcessingResult(
                processed_data=processed_data,
                regime_statistics=regime_statistics,
                processing_metadata=processing_metadata,
                validation_results=validation_results,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in regime data processing: {e}")
            raise
    
    def _validate_inputs(self, data: pd.DataFrame, regime_ids: np.ndarray) -> Dict[str, Any]:
        """Validate input data and regime assignments."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check data length consistency
        if len(data) != len(regime_ids):
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Data length ({len(data)}) != regime_ids length ({len(regime_ids)})")
        
        # Check for required columns
        if self.config.timestamp_column not in data.columns:
            validation_results['issues'].append(f"Timestamp column '{self.config.timestamp_column}' not found")
        
        # Check regime distribution
        unique_regimes, counts = np.unique(regime_ids, return_counts=True)
        small_regimes = [regime for regime, count in zip(unique_regimes, counts) if count < self.config.min_regime_samples]
        large_regimes = [regime for regime, count in zip(unique_regimes, counts) if count > self.config.max_regime_samples]
        
        if small_regimes:
            validation_results['issues'].append(f"Regimes with < {self.config.min_regime_samples} samples: {small_regimes}")
        
        if large_regimes:
            validation_results['issues'].append(f"Regimes with > {self.config.max_regime_samples} samples: {large_regimes}")
        
        # Statistics
        validation_results['statistics'] = {
            'total_samples': len(data),
            'total_regimes': len(unique_regimes),
            'regime_distribution': dict(zip(unique_regimes, counts)),
            'min_regime_size': np.min(counts),
            'max_regime_size': np.max(counts),
            'avg_regime_size': np.mean(counts)
        }
        
        return validation_results
    
    def _split_data_by_regimes(self, data: pd.DataFrame, regime_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
        """Split data by regime assignments."""
        regime_data = {}
        unique_regimes = np.unique(regime_ids)
        
        for regime in unique_regimes:
            regime_mask = regime_ids == regime
            regime_df = data[regime_mask].copy()
            
            # Add regime information
            regime_df[self.config.regime_column] = regime
            
            # Optimize data types
            if self.config.memory_efficient:
                regime_df = self.data_optimizer.optimize_dataframe(regime_df)
            
            regime_data[str(regime)] = regime_df
        
        return regime_data
    
    def _validate_regime_data(self, regime_df: pd.DataFrame, regime_id: str) -> Dict[str, Any]:
        """Validate data for a specific regime."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check sample count
        if len(regime_df) < self.config.min_regime_samples:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Regime {regime_id} has only {len(regime_df)} samples")
        
        # Check for missing values
        missing_counts = regime_df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(regime_df) * 0.1]
        if not high_missing.empty:
            validation_results['issues'].append(f"High missing values in columns: {high_missing.index.tolist()}")
        
        # Check data continuity if timestamp column exists
        if self.config.timestamp_column in regime_df.columns:
            timestamps = pd.to_datetime(regime_df[self.config.timestamp_column])
            time_gaps = timestamps.diff().dropna()
            large_gaps = time_gaps[time_gaps > time_gaps.quantile(0.95)]
            if not large_gaps.empty:
                validation_results['issues'].append(f"Large time gaps detected in regime {regime_id}")
        
        # Statistics
        validation_results['statistics'] = {
            'sample_count': len(regime_df),
            'feature_count': len(regime_df.columns),
            'missing_value_ratio': regime_df.isnull().sum().sum() / (len(regime_df) * len(regime_df.columns)),
            'memory_usage_mb': regime_df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return validation_results
    
    def _process_single_regime(self, regime_df: pd.DataFrame, regime_id: str) -> pd.DataFrame:
        """Process data for a single regime."""
        processed_df = regime_df.copy()
        
        # Add regime-specific features
        processed_df = self._add_regime_features(processed_df, regime_id)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Add regime metadata
        processed_df = self._add_regime_metadata(processed_df, regime_id)
        
        return processed_df
    
    def _add_regime_features(self, df: pd.DataFrame, regime_id: str) -> pd.DataFrame:
        """Add regime-specific features."""
        # Add regime duration features
        if self.config.timestamp_column in df.columns:
            timestamps = pd.to_datetime(df[self.config.timestamp_column])
            df['regime_duration'] = (timestamps - timestamps.min()).dt.total_seconds()
            df['regime_position'] = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
        # Add regime statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in [self.config.regime_column, 'regime_duration', 'regime_position']:
                df[f'{col}_regime_mean'] = df[col].mean()
                df[f'{col}_regime_std'] = df[col].std()
                df[f'{col}_regime_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in regime data."""
        # Forward fill for time series data
        if self.config.timestamp_column in df.columns:
            df = df.sort_values(self.config.timestamp_column)
            df = df.fillna(method='ffill')
        
        # Fill remaining missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value[0])
        
        return df
    
    def _add_regime_metadata(self, df: pd.DataFrame, regime_id: str) -> pd.DataFrame:
        """Add regime metadata to DataFrame."""
        df['regime_id'] = regime_id
        df['regime_sample_count'] = len(df)
        df['regime_processing_timestamp'] = datetime.now().isoformat()
        
        return df
    
    def _calculate_regime_statistics(self, df: pd.DataFrame, regime_id: str) -> Dict[str, Any]:
        """Calculate statistics for a regime."""
        statistics = {
            'regime_id': regime_id,
            'sample_count': len(df),
            'feature_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Numeric statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            statistics['numeric_summary'] = df[numeric_columns].describe().to_dict()
        
        # Time-based statistics
        if self.config.timestamp_column in df.columns:
            timestamps = pd.to_datetime(df[self.config.timestamp_column])
            statistics['time_span'] = {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat(),
                'duration_seconds': (timestamps.max() - timestamps.min()).total_seconds()
            }
        
        return statistics
    
    def _save_processed_data(
        self,
        processed_data: Dict[str, pd.DataFrame],
        regime_statistics: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Save processed data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save regime data
        for regime_id, df in processed_data.items():
            file_path = output_path / f"regime_{regime_id}_data.{self.config.output_format}"
            
            if self.config.output_format == "parquet":
                df.to_parquet(file_path, compression=self.config.compression)
            elif self.config.output_format == "csv":
                df.to_csv(file_path, index=False)
            elif self.config.output_format == "json":
                df.to_json(file_path, orient='records', date_format='iso')
        
        # Save statistics
        if self.config.include_metadata:
            stats_path = output_path / "regime_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(regime_statistics, f, indent=2, default=str)
    
    def _calculate_memory_usage(self, processed_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total memory usage of processed data."""
        total_memory = 0
        for df in processed_data.values():
            total_memory += df.memory_usage(deep=True).sum()
        return total_memory / 1024 / 1024  # Convert to MB
    
    def validate_regime_continuity(self, regime_ids: np.ndarray) -> Dict[str, Any]:
        """Validate regime continuity and detect anomalies."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Calculate regime durations
        durations = []
        current_regime = regime_ids[0]
        current_duration = 1
        
        for i in range(1, len(regime_ids)):
            if regime_ids[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime_ids[i]
                current_duration = 1
        
        durations.append(current_duration)
        
        # Check for very short regimes
        short_durations = [d for d in durations if d < 5]
        if short_durations:
            validation_results['issues'].append(f"Found {len(short_durations)} very short regimes (< 5 samples)")
        
        # Check for very long regimes
        long_durations = [d for d in durations if d > 1000]
        if long_durations:
            validation_results['issues'].append(f"Found {len(long_durations)} very long regimes (> 1000 samples)")
        
        # Statistics
        validation_results['statistics'] = {
            'total_regimes': len(durations),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'duration_std': np.std(durations),
            'short_regimes': len(short_durations),
            'long_regimes': len(long_durations)
        }
        
        return validation_results
    
    def analyze_regime_transitions(self, regime_ids: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions and patterns."""
        unique_regimes = np.unique(regime_ids)
        n_regimes = len(unique_regimes)
        
        # Create transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_to_index = {regime: i for i, regime in enumerate(unique_regimes)}
        
        for i in range(len(regime_ids) - 1):
            current_regime = regime_ids[i]
            next_regime = regime_ids[i + 1]
            current_idx = regime_to_index[current_regime]
            next_idx = regime_to_index[next_regime]
            transition_matrix[current_idx, next_idx] += 1
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i] /= row_sums[i]
        
        # Calculate transition statistics
        transitions = 0
        for i in range(len(regime_ids) - 1):
            if regime_ids[i] != regime_ids[i + 1]:
                transitions += 1
        
        transition_rate = transitions / len(regime_ids)
        
        return {
            'transition_matrix': transition_matrix.tolist(),
            'regime_mapping': regime_to_index,
            'total_transitions': transitions,
            'transition_rate': transition_rate,
            'most_common_transitions': self._find_most_common_transitions(transition_matrix, regime_to_index)
        }
    
    def _find_most_common_transitions(self, transition_matrix: np.ndarray, regime_mapping: Dict[int, int]) -> List[Dict[str, Any]]:
        """Find the most common regime transitions."""
        transitions = []
        index_to_regime = {v: k for k, v in regime_mapping.items()}
        
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                if transition_matrix[i, j] > 0:
                    transitions.append({
                        'from_regime': index_to_regime[i],
                        'to_regime': index_to_regime[j],
                        'probability': transition_matrix[i, j]
                    })
        
        # Sort by probability
        transitions.sort(key=lambda x: x['probability'], reverse=True)
        return transitions[:10]  # Return top 10 transitions

# Convenience functions
def get_regime_processor(config: Optional[RegimeProcessingConfig] = None) -> RegimeDataProcessor:
    """Get a configured regime data processor."""
    return RegimeDataProcessor(config)

def process_regime_data(
    data: pd.DataFrame,
    regime_ids: np.ndarray,
    output_dir: Optional[str] = None,
    config: Optional[RegimeProcessingConfig] = None
) -> RegimeProcessingResult:
    """Convenience function for regime data processing."""
    processor = get_regime_processor(config)
    return processor.process_regime_data(data, regime_ids, output_dir)

def validate_regime_continuity(regime_ids: np.ndarray) -> Dict[str, Any]:
    """Convenience function for regime continuity validation."""
    processor = get_regime_processor()
    return processor.validate_regime_continuity(regime_ids)

def analyze_regime_transitions(regime_ids: np.ndarray) -> Dict[str, Any]:
    """Convenience function for regime transition analysis."""
    processor = get_regime_processor()
    return processor.analyze_regime_transitions(regime_ids)