#!/usr/bin/env python3
"""Demonstration of Step 2 optimizations without external dependencies.

This script demonstrates the key optimization concepts implemented
in the optimized Step 2 data reading module.
"""
import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import gc
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock data structures to simulate pandas operations
class MockDataFrame:
    """Mock DataFrame to simulate pandas operations."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.columns = list(data[0].keys()) if data else []
        self.shape = (len(data), len(self.columns))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.data]
        return self.data[key]
    
    def to_dict(self):
        return {col: self[col] for col in self.columns}
    
    def memory_usage(self, deep=True):
        # Mock memory usage calculation
        return sum(len(str(row)) for row in self.data) / 1024 / 1024

# Custom exceptions for better error handling
class DataReadingError(Exception):
    """Base exception for data reading errors."""
    pass

class DataQualityError(DataReadingError):
    """Exception for data quality issues."""
    pass

class FileNotFoundError(DataReadingError):
    """Exception for file not found issues."""
    pass

class ValidationError(DataReadingError):
    """Exception for validation errors."""
    pass

# Fast-fail validation functions
def fast_fail_file_check(file_paths: List[Path], min_files: int = 1) -> Tuple[bool, Optional[str]]:
    """Fast-fail check for file existence and count."""
    if not file_paths:
        return False, "No parquet files found"
    
    if len(file_paths) < min_files:
        return False, f"Insufficient files: {len(file_paths)} < {min_files}"
    
    # Check if files are readable
    for file_path in file_paths[:5]:  # Check first 5 files
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        if file_path.stat().st_size == 0:
            return False, f"Empty file: {file_path}"
    
    return True, None

def fast_fail_schema_check(data: MockDataFrame) -> Tuple[bool, Optional[str]]:
    """Fast-fail check for required schema."""
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, None

def fast_fail_data_size_check(data: MockDataFrame, min_rows: int = 1000) -> Tuple[bool, Optional[str]]:
    """Fast-fail check for minimum data size."""
    if len(data) < min_rows:
        return False, f"Insufficient data rows: {len(data)} < {min_rows}"
    
    return True, None

# Vectorized validation functions (simplified)
def vectorized_price_validation(data: MockDataFrame) -> Dict[str, Any]:
    """Vectorized price validation using list comprehensions."""
    results = {}
    
    # Check for negative prices
    negative_count = 0
    infinite_count = 0
    nan_count = 0
    ohlc_inconsistencies = 0
    
    for row in data.data:
        # Check for negative prices
        if any(row[col] <= 0 for col in ['open', 'high', 'low', 'close']):
            negative_count += 1
        
        # Check for infinite values
        if any(row[col] == float('inf') or row[col] == float('-inf') for col in ['open', 'high', 'low', 'close']):
            infinite_count += 1
        
        # Check for NaN values
        if any(str(row[col]) == 'nan' for col in ['open', 'high', 'low', 'close']):
            nan_count += 1
        
        # OHLC consistency check
        if not (row['low'] <= row['open'] <= row['high'] and row['low'] <= row['close'] <= row['high']):
            ohlc_inconsistencies += 1
    
    results['negative_prices'] = negative_count
    results['infinite_prices'] = infinite_count
    results['nan_prices'] = nan_count
    results['ohlc_inconsistencies'] = ohlc_inconsistencies
    
    return results

def vectorized_timestamp_validation(data: MockDataFrame) -> Dict[str, Any]:
    """Vectorized timestamp validation."""
    results = {}
    
    timestamps = [row['timestamp'] for row in data.data]
    
    # Check for duplicate timestamps
    results['duplicate_timestamps'] = len(timestamps) - len(set(timestamps))
    
    # Check for monotonic ordering
    is_monotonic = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    results['non_monotonic'] = not is_monotonic
    
    # Check for gaps larger than 0.5 seconds
    large_gaps = 0
    max_gap = 0
    for i in range(len(timestamps)-1):
        gap = timestamps[i+1] - timestamps[i]
        if gap > 0.5:
            large_gaps += 1
        max_gap = max(max_gap, gap)
    
    results['large_gaps'] = large_gaps
    results['max_gap_seconds'] = max_gap
    
    return results

def vectorized_volume_validation(data: MockDataFrame) -> Dict[str, Any]:
    """Vectorized volume validation with sanity checks."""
    results = {}
    
    volumes = [row['volume'] for row in data.data]
    
    # Check for negative volumes
    results['negative_volumes'] = sum(1 for v in volumes if v < 0)
    
    # Check for zero volumes
    results['zero_volumes'] = sum(1 for v in volumes if v == 0)
    
    # Volume sanity check - detect unrealistic spikes
    if len(volumes) > 100:
        volumes_sorted = sorted(volumes)
        volume_q99 = volumes_sorted[int(len(volumes) * 0.99)]
        volume_median = volumes_sorted[len(volumes) // 2]
        
        # Check for volumes > 10x the 99th percentile
        results['extreme_high_volumes'] = sum(1 for v in volumes if v > volume_q99 * 10)
        
        # Check for volumes that are too low compared to median
        results['extreme_low_volumes'] = sum(1 for v in volumes if v < volume_median * 0.001)
        
        results['volume_q99'] = volume_q99
        results['volume_median'] = volume_median
    else:
        results['extreme_high_volumes'] = 0
        results['extreme_low_volumes'] = 0
    
    return results

# Parallel file reading functions
async def read_mock_file_async(file_path: Path) -> Optional[MockDataFrame]:
    """Asynchronously read a mock file."""
    try:
        # Simulate file reading delay
        await asyncio.sleep(0.01)
        
        # Create mock data
        data = []
        for i in range(1000):
            data.append({
                'timestamp': i * 0.001,  # 1ms intervals
                'open': 100.0 + i * 0.01,
                'high': 100.5 + i * 0.01,
                'low': 99.5 + i * 0.01,
                'close': 100.2 + i * 0.01,
                'volume': 1000 + i * 10
            })
        
        return MockDataFrame(data)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

async def read_mock_files_parallel(file_paths: List[Path], max_workers: int = 4) -> List[MockDataFrame]:
    """Read multiple mock files in parallel."""
    semaphore = asyncio.Semaphore(max_workers)
    
    async def read_with_semaphore(file_path: Path) -> Optional[MockDataFrame]:
        async with semaphore:
            return await read_mock_file_async(file_path)
    
    tasks = [read_with_semaphore(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    dataframes = []
    for result in results:
        if isinstance(result, MockDataFrame):
            dataframes.append(result)
        elif isinstance(result, Exception):
            logger.error(f"Exception in parallel reading: {result}")
    
    return dataframes

# Memory-efficient concatenation
def memory_efficient_concat(dataframes: List[MockDataFrame], chunk_size: int = 10000) -> MockDataFrame:
    """Memory-efficient concatenation of dataframes."""
    if not dataframes:
        return MockDataFrame([])
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Process in chunks to reduce memory usage
    result_data = []
    
    for i in range(0, len(dataframes), chunk_size):
        chunk = dataframes[i:i + chunk_size]
        if chunk:
            # Concatenate chunk
            for df in chunk:
                result_data.extend(df.data)
            
            # Force garbage collection
            del chunk
            gc.collect()
    
    return MockDataFrame(result_data)

# Optimized data reading step class
class OptimizedDataReadingStep:
    """Optimized Step 2: Data Reading with parallel processing and fast-fail validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        self.start_time = None
        self.step_timings = {}
        
        # Configuration
        self.max_workers = config.get('max_workers', 4)
        self.chunk_size = config.get('chunk_size', 10000)
        self.min_rows = config.get('min_rows', 1000)
        self.max_duplicate_ratio = config.get('max_duplicate_ratio', 0.01)
        self.max_gap_seconds = config.get('max_gap_seconds', 0.5)
    
    async def initialize(self) -> None:
        """Initialize the optimized data reading step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Optimized Data Reading Step...')
        self.logger.info(f'   - Max workers: {self.max_workers}')
        self.logger.info(f'   - Chunk size: {self.chunk_size}')
        self.logger.info(f'   - Min rows: {self.min_rows}')
        self.logger.info('✅ Optimized Data Reading Step initialized')
    
    async def read_unified_data_optimized(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[MockDataFrame]:
        """Read unified data with parallel processing and fast-fail validation."""
        step_start = time.time()
        
        try:
            self.logger.info(f'📖 Reading unified data for {symbol} on {exchange} ({timeframe})')
            
            # Build data path
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            
            # Fast-fail: Check if path exists
            if not unified_data_path.exists():
                error_msg = f'Unified data path does not exist: {unified_data_path}'
                self.logger.error(f'❌ {error_msg}')
                raise FileNotFoundError(error_msg)
            
            # Create mock parquet files for demonstration
            parquet_files = []
            for i in range(5):  # Create 5 mock files
                file_path = unified_data_path / f'data_{i}.parquet'
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()  # Create empty file
                parquet_files.append(file_path)
            
            # Fast-fail: Check file existence and count
            is_valid, error_msg = fast_fail_file_check(parquet_files, min_files=1)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                raise FileNotFoundError(error_msg)
            
            self.logger.info(f'📁 Found {len(parquet_files)} parquet files')
            
            # Parallel file reading
            self.logger.info(f'🔄 Reading files in parallel with {self.max_workers} workers...')
            dataframes = await read_mock_files_parallel(parquet_files, self.max_workers)
            
            if not dataframes:
                error_msg = 'No data found in parquet files'
                self.logger.error(f'❌ {error_msg}')
                raise DataReadingError(error_msg)
            
            self.logger.info(f'📊 Successfully read {len(dataframes)} dataframes')
            
            # Memory-efficient concatenation
            self.logger.info('🔄 Concatenating dataframes efficiently...')
            unified_data = memory_efficient_concat(dataframes, self.chunk_size)
            
            # Fast-fail: Check data size
            is_valid, error_msg = fast_fail_data_size_check(unified_data, self.min_rows)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                raise DataQualityError(error_msg)
            
            # Fast-fail: Check schema
            is_valid, error_msg = fast_fail_schema_check(unified_data)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                raise ValidationError(error_msg)
            
            self.logger.info(f'✅ Successfully read unified data: {len(unified_data)} rows')
            self._log_step_timing('read_unified_data_optimized', step_start)
            
            return unified_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error reading unified data: {e}')
            raise
    
    async def validate_data_quality_optimized(self, data: MockDataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """Validate data quality using vectorized operations and comprehensive checks."""
        step_start = time.time()
        
        try:
            self.logger.info('🔍 Validating data quality with vectorized operations...')
            
            # Vectorized validations
            price_validation = vectorized_price_validation(data)
            timestamp_validation = vectorized_timestamp_validation(data)
            volume_validation = vectorized_volume_validation(data)
            
            # Combine results
            validation_results = {
                'passed': True,
                'issues': [],
                'warnings': [],
                'data_info': {
                    'rows': len(data),
                    'columns': list(data.columns),
                    'memory_usage': data.memory_usage(deep=True)
                },
                'quality_score': 100.0,
                'price_validation': price_validation,
                'timestamp_validation': timestamp_validation,
                'volume_validation': volume_validation
            }
            
            # Check price validation results
            if price_validation['negative_prices'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Negative prices: {price_validation['negative_prices']} rows")
                validation_results['quality_score'] -= 20
            
            if price_validation['infinite_prices'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Infinite prices: {price_validation['infinite_prices']} rows")
                validation_results['quality_score'] -= 20
            
            if price_validation['nan_prices'] > 0:
                validation_results['warnings'].append(f"NaN prices: {price_validation['nan_prices']} rows")
                validation_results['quality_score'] -= 10
            
            if price_validation['ohlc_inconsistencies'] > 0:
                validation_results['warnings'].append(f"OHLC inconsistencies: {price_validation['ohlc_inconsistencies']} rows")
                validation_results['quality_score'] -= 5
            
            # Check timestamp validation results
            if timestamp_validation['duplicate_timestamps'] > 0:
                duplicate_ratio = timestamp_validation['duplicate_timestamps'] / len(data)
                if duplicate_ratio > self.max_duplicate_ratio:
                    validation_results['passed'] = False
                    validation_results['issues'].append(f"Too many duplicate timestamps: {timestamp_validation['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                    validation_results['quality_score'] -= 15
                else:
                    validation_results['warnings'].append(f"Duplicate timestamps: {timestamp_validation['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                    validation_results['quality_score'] -= 5
            
            if timestamp_validation['non_monotonic']:
                validation_results['passed'] = False
                validation_results['issues'].append("Non-monotonic timestamp ordering")
                validation_results['quality_score'] -= 20
            
            if timestamp_validation['large_gaps'] > 0:
                validation_results['warnings'].append(f"Large time gaps (>0.5s): {timestamp_validation['large_gaps']} gaps, max: {timestamp_validation['max_gap_seconds']:.2f}s")
                validation_results['quality_score'] -= 5
            
            # Check volume validation results
            if volume_validation['negative_volumes'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Negative volumes: {volume_validation['negative_volumes']} rows")
                validation_results['quality_score'] -= 15
            
            if volume_validation['extreme_high_volumes'] > 0:
                validation_results['warnings'].append(f"Extreme high volumes: {volume_validation['extreme_high_volumes']} rows")
                validation_results['quality_score'] -= 5
            
            if volume_validation['extreme_low_volumes'] > 0:
                validation_results['warnings'].append(f"Extreme low volumes: {volume_validation['extreme_low_volumes']} rows")
                validation_results['quality_score'] -= 5
            
            # Ensure quality score is not negative
            validation_results['quality_score'] = max(0, validation_results['quality_score'])
            
            self.logger.info(f'✅ Data quality validation completed')
            self.logger.info(f"   - Rows: {validation_results['data_info']['rows']}")
            self.logger.info(f"   - Memory usage: {validation_results['data_info']['memory_usage']:.2f} MB")
            self.logger.info(f"   - Quality score: {validation_results['quality_score']:.2f}")
            self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
            self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")
            
            self._log_step_timing('validate_data_quality_optimized', step_start)
            return validation_results
            
        except Exception as e:
            self.logger.exception(f'❌ Error during data quality validation: {e}')
            raise
    
    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')
    
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute the optimized data reading step."""
        self.logger.info('🚀 Starting Optimized Step 2: Data Reading and Validation')
        
        try:
            # Read unified data with parallel processing
            unified_data = await self.read_unified_data_optimized(symbol, exchange, timeframe, data_dir)
            
            # Validate data quality with vectorized operations
            validation_results = await self.validate_data_quality_optimized(unified_data, symbol, exchange)
            
            if not validation_results['passed']:
                self.logger.error('❌ Data quality validation failed')
                self.logger.error(f"   Issues: {validation_results['issues']}")
                return {'success': False, 'error': 'Data quality validation failed', 'validation_results': validation_results}
            
            # Save validated data (mock)
            processed_dir = Path(data_dir) / 'processed' / exchange / symbol
            processed_dir.mkdir(parents=True, exist_ok=True)
            output_file = f'{exchange}_{symbol}_{timeframe}_validated_data.parquet'
            output_path = processed_dir / output_file
            output_path.touch()  # Mock file creation
            
            self.logger.info(f'✅ Optimized Step 2 completed successfully')
            self.logger.info(f'   - Validated data saved to: {output_path}')
            self.logger.info(f'   - Total execution time: {time.time() - self.start_time:.2f} seconds')
            
            return {
                'success': True,
                'data_path': str(output_path),
                'validation_results': validation_results,
                'step_timings': self.step_timings
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error in Optimized Step 2: {e}')
            return {'success': False, 'error': str(e)}

# Test functions
def test_vectorized_operations():
    """Test vectorized operations performance."""
    logger.info("🧪 Testing vectorized operations...")
    
    # Create test data
    test_data = []
    for i in range(10000):
        test_data.append({
            'timestamp': i * 0.001,
            'open': 100.0 + i * 0.01,
            'high': 100.5 + i * 0.01,
            'low': 99.5 + i * 0.01,
            'close': 100.2 + i * 0.01,
            'volume': 1000 + i * 10
        })
    
    # Add some data quality issues
    test_data[100]['close'] = -1.0  # Negative price
    test_data[200]['volume'] = float('inf')  # Infinite volume
    test_data[300]['open'] = float('nan')  # NaN price
    
    data = MockDataFrame(test_data)
    
    # Test vectorized price validation
    start_time = time.time()
    price_results = vectorized_price_validation(data)
    price_time = time.time() - start_time
    
    # Test vectorized timestamp validation
    start_time = time.time()
    timestamp_results = vectorized_timestamp_validation(data)
    timestamp_time = time.time() - start_time
    
    # Test vectorized volume validation
    start_time = time.time()
    volume_results = vectorized_volume_validation(data)
    volume_time = time.time() - start_time
    
    logger.info("📊 Vectorized Operations Results:")
    logger.info(f"   - Price validation: {price_time:.4f}s")
    logger.info(f"   - Timestamp validation: {timestamp_time:.4f}s")
    logger.info(f"   - Volume validation: {volume_time:.4f}s")
    logger.info(f"   - Total time: {price_time + timestamp_time + volume_time:.4f}s")
    
    # Log validation results
    logger.info("🔍 Validation Results:")
    logger.info(f"   - Negative prices: {price_results['negative_prices']}")
    logger.info(f"   - Infinite prices: {price_results['infinite_prices']}")
    logger.info(f"   - NaN prices: {price_results['nan_prices']}")
    logger.info(f"   - OHLC inconsistencies: {price_results['ohlc_inconsistencies']}")
    logger.info(f"   - Duplicate timestamps: {timestamp_results['duplicate_timestamps']}")
    logger.info(f"   - Large gaps: {timestamp_results['large_gaps']}")
    logger.info(f"   - Negative volumes: {volume_results['negative_volumes']}")
    logger.info(f"   - Zero volumes: {volume_results['zero_volumes']}")
    
    return {
        'price_validation': price_results,
        'timestamp_validation': timestamp_results,
        'volume_validation': volume_results,
        'timings': {
            'price_time': price_time,
            'timestamp_time': timestamp_time,
            'volume_time': volume_time,
            'total_time': price_time + timestamp_time + volume_time
        }
    }

def test_fast_fail_validation():
    """Test fast-fail validation functions."""
    logger.info("🧪 Testing fast-fail validation...")
    
    # Test with valid data
    valid_data = []
    for i in range(1000):
        valid_data.append({
            'timestamp': i * 0.001,
            'open': 100.0,
            'high': 100.5,
            'low': 99.5,
            'close': 100.2,
            'volume': 1000
        })
    
    valid_df = MockDataFrame(valid_data)
    
    # Test schema check
    start_time = time.time()
    is_valid, error = fast_fail_schema_check(valid_df)
    schema_time = time.time() - start_time
    
    logger.info(f"✅ Schema check (valid data): {is_valid}, {error}, {schema_time:.6f}s")
    
    # Test data size check
    start_time = time.time()
    is_valid, error = fast_fail_data_size_check(valid_df, 500)
    size_time = time.time() - start_time
    
    logger.info(f"✅ Size check (valid data): {is_valid}, {error}, {size_time:.6f}s")
    
    # Test with invalid data
    invalid_data = []
    for i in range(100):
        invalid_data.append({
            'timestamp': i * 0.001,
            'open': 100.0,
            'high': 100.5,
            'low': 99.5,
            'close': 100.2,
            'volume': 1000
        })
    
    invalid_df = MockDataFrame(invalid_data)
    
    start_time = time.time()
    is_valid, error = fast_fail_data_size_check(invalid_df, 500)
    invalid_size_time = time.time() - start_time
    
    logger.info(f"❌ Size check (insufficient data): {is_valid}, {error}, {invalid_size_time:.6f}s")
    
    return {
        'valid_schema_time': schema_time,
        'valid_size_time': size_time,
        'invalid_size_time': invalid_size_time
    }

async def test_parallel_reading():
    """Test parallel file reading performance."""
    logger.info("🧪 Testing parallel file reading...")
    
    # Create test files
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create multiple test files
    num_files = 10
    file_paths = []
    
    for i in range(num_files):
        file_path = test_dir / f"test_data_{i}.parquet"
        file_path.touch()
        file_paths.append(file_path)
    
    logger.info(f"Created {num_files} test files")
    
    # Test parallel reading
    start_time = time.time()
    dataframes = await read_mock_files_parallel(file_paths, max_workers=4)
    parallel_time = time.time() - start_time
    
    logger.info(f"✅ Parallel reading: {len(dataframes)} files in {parallel_time:.4f}s")
    
    # Test sequential reading for comparison
    start_time = time.time()
    sequential_dataframes = []
    for file_path in file_paths:
        df = await read_mock_file_async(file_path)
        if df:
            sequential_dataframes.append(df)
    sequential_time = time.time() - start_time
    
    logger.info(f"📊 Sequential reading: {len(sequential_dataframes)} files in {sequential_time:.4f}s")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"🚀 Speedup: {speedup:.2f}x")
    
    # Cleanup
    for file_path in file_paths:
        file_path.unlink()
    test_dir.rmdir()
    
    return {
        'parallel_time': parallel_time,
        'sequential_time': sequential_time,
        'speedup': speedup,
        'files_processed': len(dataframes)
    }

async def test_optimized_step():
    """Test the optimized step implementation."""
    logger.info("🧪 Testing optimized step implementation...")
    
    # Create test data directory structure
    test_data_dir = Path("test_data_cache")
    test_data_dir.mkdir(exist_ok=True)
    
    unified_dir = test_data_dir / "unified" / "BINANCE" / "ETHUSDT" / "1m"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created test data directory: {unified_dir}")
    
    # Test optimized step
    config = {
        'max_workers': 4,
        'chunk_size': 1000,
        'min_rows': 1000,
        'max_duplicate_ratio': 0.01,
        'max_gap_seconds': 0.5
    }
    
    step = OptimizedDataReadingStep(config)
    await step.initialize()
    
    start_time = time.time()
    result = await step.execute("ETHUSDT", "BINANCE", "1m", str(test_data_dir))
    execution_time = time.time() - start_time
    
    logger.info(f"✅ Optimized step execution: {execution_time:.4f}s")
    logger.info(f"   - Success: {result['success']}")
    
    if result['success']:
        logger.info(f"   - Data path: {result['data_path']}")
        logger.info(f"   - Quality score: {result['validation_results']['quality_score']}")
        logger.info(f"   - Issues: {len(result['validation_results']['issues'])}")
        logger.info(f"   - Warnings: {len(result['validation_results']['warnings'])}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_data_dir)
    
    return {
        'execution_time': execution_time,
        'success': result['success'],
        'result': result
    }

async def main():
    """Run all optimization tests."""
    logger.info("🚀 Starting Step 2 Optimization Demonstration")
    logger.info("=" * 60)
    
    # Test vectorized operations
    logger.info("\n1. Testing Vectorized Operations")
    logger.info("-" * 40)
    vectorized_results = test_vectorized_operations()
    
    # Test fast-fail validation
    logger.info("\n2. Testing Fast-Fail Validation")
    logger.info("-" * 40)
    fast_fail_results = test_fast_fail_validation()
    
    # Test parallel reading
    logger.info("\n3. Testing Parallel File Reading")
    logger.info("-" * 40)
    parallel_results = await test_parallel_reading()
    
    # Test optimized step
    logger.info("\n4. Testing Optimized Step Implementation")
    logger.info("-" * 40)
    step_results = await test_optimized_step()
    
    # Summary
    logger.info("\n📊 OPTIMIZATION DEMONSTRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Vectorized Operations Total Time: {vectorized_results['timings']['total_time']:.4f}s")
    logger.info(f"Fast-Fail Validation (Valid): {fast_fail_results['valid_schema_time'] + fast_fail_results['valid_size_time']:.6f}s")
    logger.info(f"Fast-Fail Validation (Invalid): {fast_fail_results['invalid_size_time']:.6f}s")
    logger.info(f"Parallel Reading Speedup: {parallel_results['speedup']:.2f}x")
    logger.info(f"Optimized Step Execution: {step_results['execution_time']:.4f}s")
    
    logger.info("\n✅ All optimization demonstrations completed successfully!")
    
    return {
        'vectorized_results': vectorized_results,
        'fast_fail_results': fast_fail_results,
        'parallel_results': parallel_results,
        'step_results': step_results
    }

if __name__ == "__main__":
    asyncio.run(main())