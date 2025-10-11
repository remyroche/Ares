"""
VectorBT Batch Processing Enhancement

This module provides high-performance batch processing capabilities using VectorBT's
optimized operations for feature generation, trading signals, and backtesting.

Key Features:
- Vectorized batch processing with VectorBT
- GPU acceleration for large datasets
- Memory-efficient processing with chunking
- Parallel processing for multiple symbols/timeframes
- Advanced memory management
- Progress tracking and monitoring
"""

import numpy as np
import pandas as pd
import logging
import time
import gc
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from abc import ABC, abstractmethod

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max
    from vectorbt.generic import rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Optional parallel processing
try:
    import multiprocessing as mp
    from multiprocessing import Pool, cpu_count
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    mp = None
    Pool = None
    cpu_count = None

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for VectorBT batch processing."""
    # Batch settings
    batch_size: int = 10000
    chunk_size: int = 50000
    max_memory_gb: float = 8.0
    
    # Processing settings
    enable_gpu: bool = False
    enable_parallel: bool = True
    max_workers: int = None
    use_threading: bool = True
    
    # Memory management
    enable_memory_optimization: bool = True
    memory_cleanup_frequency: int = 10  # Cleanup every N batches
    enable_garbage_collection: bool = True
    
    # Progress tracking
    enable_progress_tracking: bool = True
    progress_update_frequency: int = 100  # Update every N batches
    enable_timing: bool = True
    
    # VectorBT specific
    vectorbt_freq: str = '1min'
    vectorbt_enable_parallel: bool = True
    vectorbt_memory_limit_gb: float = 8.0
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(cpu_count() if PARALLEL_AVAILABLE else 4, 8)
        if self.enable_gpu and not CUPY_AVAILABLE:
            self.enable_gpu = False
            logger.warning("GPU acceleration requested but CuPy not available")


class BatchProcessor(ABC):
    """Abstract base class for batch processors."""
    
    @abstractmethod
    def process_batch(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process a single batch of data."""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get required columns for processing."""
        pass


class VectorBTBatchProcessor:
    """
    High-performance batch processor using VectorBT optimizations.
    
    Provides efficient batch processing for:
    - Feature generation
    - Trading signal processing
    - Backtesting operations
    - Multi-symbol processing
    """
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize VectorBT batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchProcessingConfig()
        self.use_vectorbt = VECTORBT_AVAILABLE
        self.use_gpu = self.config.enable_gpu and CUPY_AVAILABLE
        self.use_parallel = self.config.enable_parallel and PARALLEL_AVAILABLE
        
        # Initialize VectorBT settings
        if self.use_vectorbt:
            vbt.settings.parallel['enabled'] = self.config.vectorbt_enable_parallel
            vbt.settings.array_wrapper['freq'] = self.config.vectorbt_freq
        
        # Memory tracking
        self.memory_usage = []
        self.batch_count = 0
        self.processing_times = []
        
        logger.info(f"VectorBTBatchProcessor initialized: VectorBT={self.use_vectorbt}, GPU={self.use_gpu}, Parallel={self.use_parallel}")
    
    def process_features_batch(
        self, 
        data: pd.DataFrame, 
        feature_generators: List[BatchProcessor],
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process features in optimized batches using VectorBT.
        
        Args:
            data: Input data with multi-level index (symbol, timestamp) or single symbol
            feature_generators: List of feature generators
            symbols: Optional list of symbols to process
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        if not self.use_vectorbt:
            logger.warning("VectorBT not available, falling back to sequential processing")
            return self._process_sequential(data, feature_generators, **kwargs)
        
        try:
            start_time = time.time()
            results = []
            
            # Determine processing strategy
            if symbols and len(symbols) > 1:
                # Multi-symbol processing
                results = self._process_multi_symbol(data, feature_generators, symbols, **kwargs)
            else:
                # Single symbol or time-based batching
                results = self._process_single_symbol_batched(data, feature_generators, **kwargs)
            
            # Combine results
            if results:
                combined_results = pd.concat(results, ignore_index=False)
                self._log_processing_stats(start_time, len(combined_results))
                return combined_results
            else:
                logger.warning("No results generated from batch processing")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return self._process_sequential(data, feature_generators, **kwargs)
    
    def _process_multi_symbol(
        self, 
        data: pd.DataFrame, 
        feature_generators: List[BatchProcessor],
        symbols: List[str],
        **kwargs
    ) -> List[pd.DataFrame]:
        """Process multiple symbols in parallel using VectorBT."""
        results = []
        
        if self.use_parallel and len(symbols) > 1:
            # Parallel processing for multiple symbols
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for symbol in symbols:
                    symbol_data = data.xs(symbol, level=0) if hasattr(data.index, 'levels') else data
                    future = executor.submit(
                        self._process_single_symbol_vectorized,
                        symbol_data, feature_generators, symbol, **kwargs
                    )
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result()
                        if not result.empty:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing symbol: {e}")
        else:
            # Sequential processing
            for symbol in symbols:
                symbol_data = data.xs(symbol, level=0) if hasattr(data.index, 'levels') else data
                result = self._process_single_symbol_vectorized(symbol_data, feature_generators, symbol, **kwargs)
                if not result.empty:
                    results.append(result)
        
        return results
    
    def _process_single_symbol_batched(
        self, 
        data: pd.DataFrame, 
        feature_generators: List[BatchProcessor],
        **kwargs
    ) -> List[pd.DataFrame]:
        """Process single symbol data in time-based batches."""
        results = []
        total_rows = len(data)
        
        # Calculate batch boundaries
        batch_starts = range(0, total_rows, self.config.batch_size)
        
        for i, start_idx in enumerate(batch_starts):
            end_idx = min(start_idx + self.config.batch_size, total_rows)
            batch_data = data.iloc[start_idx:end_idx]
            
            # Process batch
            batch_result = self._process_single_batch_vectorized(batch_data, feature_generators, **kwargs)
            
            if not batch_result.empty:
                results.append(batch_result)
            
            # Memory management
            if self.config.enable_memory_optimization and i % self.config.memory_cleanup_frequency == 0:
                self._cleanup_memory()
            
            # Progress tracking
            if self.config.enable_progress_tracking and i % self.config.progress_update_frequency == 0:
                progress = (i + 1) / len(batch_starts) * 100
                logger.info(f"Batch processing progress: {progress:.1f}%")
        
        return results
    
    def _process_single_symbol_vectorized(
        self, 
        data: pd.DataFrame, 
        feature_generators: List[BatchProcessor],
        symbol: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process single symbol data using VectorBT optimizations."""
        try:
            # Prepare data for VectorBT processing
            if self.use_gpu:
                data = self._prepare_gpu_data(data)
            
            # Process with VectorBT optimizations
            results = []
            
            for generator in feature_generators:
                try:
                    # Use VectorBT-optimized processing
                    generator_result = self._process_generator_vectorized(generator, data, **kwargs)
                    if not generator_result.empty:
                        results.append(generator_result)
                except Exception as e:
                    logger.error(f"Error processing generator {generator.__class__.__name__}: {e}")
                    continue
            
            # Combine results
            if results:
                combined_result = pd.concat(results, axis=1)
                if symbol:
                    combined_result.index = pd.MultiIndex.from_product(
                        [[symbol], combined_result.index], 
                        names=['symbol', 'timestamp']
                    )
                return combined_result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in vectorized processing: {e}")
            return pd.DataFrame()
    
    def _process_single_batch_vectorized(
        self, 
        batch_data: pd.DataFrame, 
        feature_generators: List[BatchProcessor],
        **kwargs
    ) -> pd.DataFrame:
        """Process a single batch using VectorBT optimizations."""
        try:
            # Prepare data for VectorBT processing
            if self.use_gpu:
                batch_data = self._prepare_gpu_data(batch_data)
            
            # Process with VectorBT optimizations
            results = []
            
            for generator in feature_generators:
                try:
                    generator_result = self._process_generator_vectorized(generator, batch_data, **kwargs)
                    if not generator_result.empty:
                        results.append(generator_result)
                except Exception as e:
                    logger.error(f"Error processing generator {generator.__class__.__name__}: {e}")
                    continue
            
            # Combine results
            if results:
                return pd.concat(results, axis=1)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in batch vectorized processing: {e}")
            return pd.DataFrame()
    
    def _process_generator_vectorized(
        self, 
        generator: BatchProcessor, 
        data: pd.DataFrame, 
        **kwargs
    ) -> pd.DataFrame:
        """Process a single generator using VectorBT optimizations."""
        try:
            # Optimize data for VectorBT processing
            if hasattr(generator, '_optimize_dataframe_for_vectorbt'):
                data = generator._optimize_dataframe_for_vectorbt(data)
            
            # Check if generator has VectorBT optimization
            if hasattr(generator, 'use_vectorbt') and generator.use_vectorbt:
                # Use VectorBT-optimized processing with enhanced batch processing
                if hasattr(generator, 'process_batch_vectorized'):
                    result = generator.process_batch_vectorized(data, use_vectorbt=True, **kwargs)
                else:
                    result = generator.process_batch(data, use_vectorbt=True, **kwargs)
                
                # Apply VectorBT-specific optimizations
                if VECTORBT_AVAILABLE and not result.empty:
                    result = self._apply_vectorbt_optimizations(result)
                
                return result
            else:
                # Fallback to standard processing
                return generator.process_batch(data, **kwargs)
                
        except Exception as e:
            logger.error(f"Error processing generator {generator.__class__.__name__}: {e}")
            return pd.DataFrame()
    
    def _apply_vectorbt_optimizations(self, result: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT-specific optimizations to results."""
        try:
            # Use VectorBT's optimized data types
            optimized_result = result.copy()
            
            # Convert to VectorBT array wrapper for better performance
            if VECTORBT_AVAILABLE:
                for column in optimized_result.columns:
                    if optimized_result[column].dtype in ['float64', 'float32']:
                        # Use VectorBT's optimized float handling
                        optimized_result[column] = vbt.array_wrapper(optimized_result[column])
            
            return optimized_result
            
        except Exception as e:
            logger.warning(f"VectorBT optimization failed: {e}")
            return result
    
    def _prepare_gpu_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for GPU processing."""
        if not self.use_gpu or not CUPY_AVAILABLE:
            return data
        
        try:
            # Convert to CuPy arrays for GPU processing
            gpu_data = data.copy()
            for col in gpu_data.select_dtypes(include=[np.number]).columns:
                gpu_data[col] = cp.asarray(gpu_data[col].values)
            
            return gpu_data
            
        except Exception as e:
            logger.warning(f"Error preparing GPU data: {e}")
            return data
    
    def _process_sequential(
        self, 
        data: pd.DataFrame, 
        feature_generators: List[BatchProcessor],
        **kwargs
    ) -> pd.DataFrame:
        """Fallback sequential processing."""
        results = []
        
        for generator in feature_generators:
            try:
                generator_result = generator.process_batch(data, **kwargs)
                if not generator_result.empty:
                    results.append(generator_result)
            except Exception as e:
                logger.error(f"Error in sequential processing: {e}")
                continue
        
        if results:
            return pd.concat(results, axis=1)
        else:
            return pd.DataFrame()
    
    def _cleanup_memory(self):
        """Clean up memory usage."""
        if self.config.enable_garbage_collection:
            gc.collect()
        
        # Log memory usage
        if self.config.enable_memory_optimization:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            self.memory_usage.append(memory_usage)
            
            if memory_usage > 90:
                logger.warning(f"High memory usage: {memory_usage:.1f}%")
    
    def _log_processing_stats(self, start_time: float, result_count: int):
        """Log processing statistics."""
        if self.config.enable_timing:
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(f"Batch processing completed: {result_count} results in {processing_time:.2f}s")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                logger.info(f"Average processing time: {avg_time:.2f}s")


class VectorBTFeatureBatchProcessor(BatchProcessor):
    """VectorBT-optimized feature batch processor."""
    
    def __init__(self, feature_generator, use_vectorbt: bool = True):
        self.feature_generator = feature_generator
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
    
    def process_batch(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process a batch of data for feature generation."""
        try:
            if self.use_vectorbt and hasattr(self.feature_generator, 'generate_features'):
                return self.feature_generator.generate_features(data, **kwargs)
            else:
                # Fallback to standard processing
                return self.feature_generator._generate_feature(data, **kwargs)
        except Exception as e:
            logger.error(f"Error in feature batch processing: {e}")
            return pd.DataFrame()
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for processing."""
        if hasattr(self.feature_generator, 'config'):
            return getattr(self.feature_generator.config, 'required_columns', [])
        return []


class VectorBTSignalBatchProcessor(BatchProcessor):
    """VectorBT-optimized signal batch processor."""
    
    def __init__(self, signal_generator, use_vectorbt: bool = True):
        self.signal_generator = signal_generator
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
    
    def process_batch(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process a batch of data for signal generation."""
        try:
            if self.use_vectorbt and hasattr(self.signal_generator, 'generate_signals_batch'):
                return self.signal_generator.generate_signals_batch(data, **kwargs)
            else:
                # Fallback to standard processing
                return self.signal_generator.generate_signals(data, **kwargs)
        except Exception as e:
            logger.error(f"Error in signal batch processing: {e}")
            return pd.DataFrame()
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for processing."""
        return ['close', 'volume']  # Default required columns


def create_vectorbt_batch_processor(
    config: Optional[BatchProcessingConfig] = None
) -> VectorBTBatchProcessor:
    """
    Create a VectorBT batch processor.
    
    Args:
        config: Batch processing configuration
        
    Returns:
        VectorBTBatchProcessor instance
    """
    return VectorBTBatchProcessor(config)


def create_feature_batch_processor(
    feature_generator, 
    use_vectorbt: bool = True
) -> VectorBTFeatureBatchProcessor:
    """
    Create a feature batch processor.
    
    Args:
        feature_generator: Feature generator instance
        use_vectorbt: Whether to use VectorBT optimizations
        
    Returns:
        VectorBTFeatureBatchProcessor instance
    """
    return VectorBTFeatureBatchProcessor(feature_generator, use_vectorbt)


def create_signal_batch_processor(
    signal_generator, 
    use_vectorbt: bool = True
) -> VectorBTSignalBatchProcessor:
    """
    Create a signal batch processor.
    
    Args:
        signal_generator: Signal generator instance
        use_vectorbt: Whether to use VectorBT optimizations
        
    Returns:
        VectorBTSignalBatchProcessor instance
    """
    return VectorBTSignalBatchProcessor(signal_generator, use_vectorbt)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    returns = np.random.normal(0.001, 0.02, 5000)
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 5000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 5000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 5000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=dates)
    
    # Create batch processor
    config = BatchProcessingConfig(
        batch_size=1000,
        enable_gpu=False,
        enable_parallel=True,
        enable_progress_tracking=True
    )
    
    processor = create_vectorbt_batch_processor(config)
    
    # Create mock feature generators
    class MockFeatureGenerator:
        def __init__(self, name):
            self.name = name
            self.use_vectorbt = True
        
        def generate_features(self, data, **kwargs):
            # Mock feature generation
            features = pd.DataFrame(index=data.index)
            # Use VectorBT operations for better performance
            if VECTORBT_AVAILABLE:
                try:
                    features[f'{self.name}_feature_1'] = rolling_mean(data['close'], window=20)
                    features[f'{self.name}_feature_2'] = rolling_std(data['volume'], window=20)
                except Exception as e:
                    logger.warning(f"VectorBT operations failed: {e}, using pandas fallback")
                    features[f'{self.name}_feature_1'] = data['close'].rolling(window=20).mean()
                    features[f'{self.name}_feature_2'] = data['volume'].rolling(window=20).std()
            else:
                features[f'{self.name}_feature_1'] = data['close'].rolling(window=20).mean()
                features[f'{self.name}_feature_2'] = data['volume'].rolling(window=20).std()
            return features
    
    # Create feature generators
    feature_generators = [
        VectorBTFeatureBatchProcessor(MockFeatureGenerator('volatility')),
        VectorBTFeatureBatchProcessor(MockFeatureGenerator('volume')),
        VectorBTFeatureBatchProcessor(MockFeatureGenerator('momentum'))
    ]
    
    # Process features in batches
    results = processor.process_features_batch(data, feature_generators)
    
    print(f"Generated {len(results.columns)} features from {len(feature_generators)} generators")
    print("Feature names:", list(results.columns))
    print("\nFirst few rows:")
    print(results.head())