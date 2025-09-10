"""Optimized Validator for Step 2: Data Reading with Fast-Fail Validation and Vectorized Operations.

This module implements optimized validation with:
- Fast-fail validation checks
- Vectorized operations for performance
- Comprehensive data quality validation
- Memory-efficient processing
- Fixed error handling and monitoring issues
"""
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import gc

# Import the optimized monitoring and validation functions
from .step02_data_reading_optimized import (
    OptimizedFunctionCallMonitor,
    DataReadingError,
    DataQualityError,
    FileNotFoundError,
    ValidationError,
    vectorized_price_validation,
    vectorized_timestamp_validation,
    vectorized_volume_validation,
    fast_fail_file_check,
    fast_fail_schema_check,
    fast_fail_data_size_check
)

# Custom exceptions for validator
class ValidatorError(Exception):
    """Base exception for validator errors."""
    pass

class ValidationTimeoutError(ValidatorError):
    """Exception for validation timeout."""
    pass

# Fast-fail validation functions for validator
def fast_fail_directory_check(data_dir: str, exchange: str, symbol: str, timeframe: str) -> Tuple[bool, Optional[str], Optional[Path]]:
    """Fast-fail check for directory structure."""
    unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
    
    if not unified_data_path.exists():
        return False, f'Unified data directory not found: {unified_data_path}', None
    
    # Look for parquet files recursively
    data_files = list(unified_data_path.rglob('*.parquet'))
    if not data_files:
        return False, f'No parquet files found in {unified_data_path}', None
    
    return True, None, unified_data_path

def fast_fail_file_metadata_check(data_files: List[Path]) -> Tuple[bool, Optional[str], Optional[Path]]:
    """Fast-fail check for file metadata and select the best file."""
    if not data_files:
        return False, "No data files provided", None
    
    # Check file sizes and select the largest file (likely most recent)
    valid_files = []
    for file_path in data_files:
        try:
            if file_path.exists() and file_path.stat().st_size > 0:
                valid_files.append(file_path)
        except Exception as e:
            logging.warning(f"Error checking file {file_path}: {e}")
    
    if not valid_files:
        return False, "No valid data files found", None
    
    # Select the largest file (most likely to contain the most data)
    best_file = max(valid_files, key=lambda x: x.stat().st_size)
    return True, None, best_file

# Vectorized validation functions for validator
def vectorized_data_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data statistics using vectorized operations."""
    stats = {}
    
    # Basic statistics
    stats['total_rows'] = len(data)
    stats['total_columns'] = len(data.columns)
    stats['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Price statistics
    price_cols = ['open', 'high', 'low', 'close']
    if all(col in data.columns for col in price_cols):
        price_data = data[price_cols]
        stats['price_stats'] = {
            'mean': price_data.mean().to_dict(),
            'std': price_data.std().to_dict(),
            'min': price_data.min().to_dict(),
            'max': price_data.max().to_dict(),
            'median': price_data.median().to_dict()
        }
        
        # Price range analysis
        stats['price_range'] = {
            'min_price': price_data.min().min(),
            'max_price': price_data.max().max(),
            'price_span': price_data.max().max() - price_data.min().min()
        }
    
    # Volume statistics
    if 'volume' in data.columns:
        volume_data = data['volume']
        stats['volume_stats'] = {
            'mean': float(volume_data.mean()),
            'std': float(volume_data.std()),
            'min': float(volume_data.min()),
            'max': float(volume_data.max()),
            'median': float(volume_data.median()),
            'zero_count': int((volume_data == 0).sum()),
            'zero_percentage': float((volume_data == 0).sum() / len(volume_data) * 100)
        }
    
    # Timestamp statistics
    if 'timestamp' in data.columns:
        timestamp_data = data['timestamp']
        stats['timestamp_stats'] = {
            'start': str(timestamp_data.min()),
            'end': str(timestamp_data.max()),
            'duration_hours': float((timestamp_data.max() - timestamp_data.min()).total_seconds() / 3600),
            'unique_timestamps': int(timestamp_data.nunique()),
            'duplicate_timestamps': int(timestamp_data.duplicated().sum())
        }
    
    return stats

def vectorized_data_quality_checks(data: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality checks using vectorized operations."""
    quality_checks = {}
    
    # Missing data analysis
    missing_data = data.isnull().sum()
    quality_checks['missing_data'] = {
        'total_missing': int(missing_data.sum()),
        'missing_percentage': float(missing_data.sum() / (len(data) * len(data.columns)) * 100),
        'columns_with_missing': missing_data[missing_data > 0].to_dict()
    }
    
    # Data type analysis
    quality_checks['data_types'] = data.dtypes.astype(str).to_dict()
    
    # Numeric data analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_data = data[numeric_cols]
        quality_checks['numeric_analysis'] = {
            'infinite_values': int(np.isinf(numeric_data).sum().sum()),
            'negative_values': int((numeric_data < 0).sum().sum()),
            'zero_values': int((numeric_data == 0).sum().sum())
        }
    
    # Price consistency checks
    price_validation = vectorized_price_validation(data)
    quality_checks['price_validation'] = price_validation
    
    # Timestamp consistency checks
    timestamp_validation = vectorized_timestamp_validation(data)
    quality_checks['timestamp_validation'] = timestamp_validation
    
    # Volume consistency checks
    volume_validation = vectorized_volume_validation(data)
    quality_checks['volume_validation'] = volume_validation
    
    return quality_checks

# Optimized validator class
class OptimizedStep2Validator:
    """Optimized validator for Step 2: Data Reading with fast-fail validation and vectorized operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.OptimizedStep2Validator")
        self.monitor = OptimizedFunctionCallMonitor()
        
        # Configuration
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.max_workers = self.config.get('max_workers', 4)
        self.min_rows = self.config.get('min_rows', 1000)
        self.max_duplicate_ratio = self.config.get('max_duplicate_ratio', 0.01)
        self.max_gap_seconds = self.config.get('max_gap_seconds', 0.5)
        
        # Performance tracking
        self.validation_timings = {}
        self.start_time = None
    
    async def initialize(self) -> None:
        """Initialize the optimized validator."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Optimized Step 2 Validator...')
        self.logger.info(f'   - Timeout: {self.timeout_seconds}s')
        self.logger.info(f'   - Max workers: {self.max_workers}')
        self.logger.info(f'   - Min rows: {self.min_rows}')
        self.logger.info('✅ Optimized Step 2 Validator initialized')
    
    def _log_timing(self, operation: str, start_time: float) -> None:
        """Log timing information for an operation."""
        elapsed = time.time() - start_time
        self.validation_timings[operation] = elapsed
        self.logger.info(f'⏱️ {operation} completed in {elapsed:.2f} seconds')
    
    async def validate_directory_structure(self, data_dir: str, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Validate directory structure with fast-fail checks."""
        step_start = time.time()
        call_id = self.monitor.start_function_call(self.validate_directory_structure, (data_dir, exchange, symbol, timeframe), {})
        
        try:
            self.logger.info('🔍 Validating directory structure...')
            
            # Fast-fail directory check
            is_valid, error_msg, unified_data_path = fast_fail_directory_check(data_dir, exchange, symbol, timeframe)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                result = {
                    'step_name': 'step02_data_reading_validation',
                    'validation_passed': False,
                    'error': error_msg
                }
                self.monitor.complete_function_call(call_id, error=FileNotFoundError(error_msg))
                return result
            
            # Get data files
            data_files = list(unified_data_path.rglob('*.parquet'))
            
            # Fast-fail file metadata check
            is_valid, error_msg, best_file = fast_fail_file_metadata_check(data_files)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                result = {
                    'step_name': 'step02_data_reading_validation',
                    'validation_passed': False,
                    'error': error_msg
                }
                self.monitor.complete_function_call(call_id, error=FileNotFoundError(error_msg))
                return result
            
            result = {
                'validation_passed': True,
                'data_files': data_files,
                'unified_data_path': unified_data_path,
                'best_file': best_file,
                'total_files': len(data_files)
            }
            
            self.logger.info(f'✅ Directory structure validation passed')
            self.logger.info(f'   - Found {len(data_files)} parquet files')
            self.logger.info(f'   - Best file: {best_file.name}')
            
            self._log_timing('validate_directory_structure', step_start)
            self.monitor.complete_function_call(call_id, result)
            return result
            
        except Exception as e:
            self.logger.exception(f'❌ Error validating directory structure: {e}')
            result = {
                'step_name': 'step02_data_reading_validation',
                'validation_passed': False,
                'error': f'Directory validation error: {e}'
            }
            self.monitor.complete_function_call(call_id, error=e)
            return result
    
    async def validate_data_content(self, data: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Validate data content using vectorized operations."""
        step_start = time.time()
        call_id = self.monitor.start_function_call(self.validate_data_content, (data, exchange, symbol, timeframe), {})
        
        try:
            self.logger.info('🔍 Validating data content with vectorized operations...')
            
            # Fast-fail schema check
            is_valid, error_msg = fast_fail_schema_check(data)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                result = {
                    'step_name': 'step02_data_reading_validation',
                    'validation_passed': False,
                    'error': error_msg
                }
                self.monitor.complete_function_call(call_id, error=ValidationError(error_msg))
                return result
            
            # Fast-fail data size check
            is_valid, error_msg = fast_fail_data_size_check(data, self.min_rows)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                result = {
                    'step_name': 'step02_data_reading_validation',
                    'validation_passed': False,
                    'error': error_msg
                }
                self.monitor.complete_function_call(call_id, error=DataQualityError(error_msg))
                return result
            
            # Vectorized data statistics
            data_stats = vectorized_data_statistics(data)
            
            # Vectorized data quality checks
            quality_checks = vectorized_data_quality_checks(data)
            
            # Determine overall validation result
            validation_passed = True
            issues = []
            warnings = []
            
            # Check for critical issues
            if quality_checks['price_validation']['negative_prices'] > 0:
                validation_passed = False
                issues.append(f"Negative prices: {quality_checks['price_validation']['negative_prices']} rows")
            
            if quality_checks['price_validation']['infinite_prices'] > 0:
                validation_passed = False
                issues.append(f"Infinite prices: {quality_checks['price_validation']['infinite_prices']} rows")
            
            if quality_checks['timestamp_validation']['duplicate_timestamps'] > 0:
                duplicate_ratio = quality_checks['timestamp_validation']['duplicate_timestamps'] / len(data)
                if duplicate_ratio > self.max_duplicate_ratio:
                    validation_passed = False
                    issues.append(f"Too many duplicate timestamps: {quality_checks['timestamp_validation']['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                else:
                    warnings.append(f"Duplicate timestamps: {quality_checks['timestamp_validation']['duplicate_timestamps']} ({duplicate_ratio:.2%})")
            
            if quality_checks['timestamp_validation']['non_monotonic']:
                validation_passed = False
                issues.append("Non-monotonic timestamp ordering")
            
            if quality_checks['timestamp_validation']['large_gaps'] > 0:
                warnings.append(f"Large time gaps (>0.5s): {quality_checks['timestamp_validation']['large_gaps']} gaps")
            
            if quality_checks['volume_validation']['negative_volumes'] > 0:
                validation_passed = False
                issues.append(f"Negative volumes: {quality_checks['volume_validation']['negative_volumes']} rows")
            
            # Check for warnings
            if quality_checks['price_validation']['nan_prices'] > 0:
                warnings.append(f"NaN prices: {quality_checks['price_validation']['nan_prices']} rows")
            
            if quality_checks['price_validation']['ohlc_inconsistencies'] > 0:
                warnings.append(f"OHLC inconsistencies: {quality_checks['price_validation']['ohlc_inconsistencies']} rows")
            
            if quality_checks['volume_validation']['extreme_high_volumes'] > 0:
                warnings.append(f"Extreme high volumes: {quality_checks['volume_validation']['extreme_high_volumes']} rows")
            
            result = {
                'validation_passed': validation_passed,
                'data': data,
                'data_stats': data_stats,
                'quality_checks': quality_checks,
                'issues': issues,
                'warnings': warnings,
                'total_issues': len(issues),
                'total_warnings': len(warnings)
            }
            
            self.logger.info(f'✅ Data content validation completed')
            self.logger.info(f"   - Validation passed: {validation_passed}")
            self.logger.info(f"   - Data shape: {data.shape}")
            self.logger.info(f"   - Issues: {len(issues)}")
            self.logger.info(f"   - Warnings: {len(warnings)}")
            
            self._log_timing('validate_data_content', step_start)
            self.monitor.complete_function_call(call_id, result)
            return result
            
        except Exception as e:
            self.logger.exception(f'❌ Error validating data content: {e}')
            result = {
                'step_name': 'step02_data_reading_validation',
                'validation_passed': False,
                'error': f'Data content validation error: {e}'
            }
            self.monitor.complete_function_call(call_id, error=e)
            return result
    
    async def run_validation(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation for Step 2: Data Reading."""
        self.logger.info('🔍 Starting Optimized Step 2 Validation')
        
        try:
            # Extract parameters
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            # Validate directory structure
            dir_result = await self.validate_directory_structure(data_dir, exchange, symbol, timeframe)
            if not dir_result['validation_passed']:
                return dir_result
            
            # Load data from the best file
            best_file = dir_result['best_file']
            self.logger.info(f'📖 Loading data from: {best_file.name}')
            
            # Load data in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                data = await loop.run_in_executor(executor, pd.read_parquet, best_file)
            
            # Validate data content
            content_result = await self.validate_data_content(data, exchange, symbol, timeframe)
            if not content_result['validation_passed']:
                return content_result
            
            # Combine results
            final_result = {
                'step_name': 'step02_data_reading_validation',
                'validation_passed': True,
                'data_file_path': str(best_file),
                'data_shape': data.shape,
                'data_stats': content_result['data_stats'],
                'quality_checks': content_result['quality_checks'],
                'issues': content_result['issues'],
                'warnings': content_result['warnings'],
                'total_issues': content_result['total_issues'],
                'total_warnings': content_result['total_warnings'],
                'validation_timings': self.validation_timings,
                'performance_summary': self.monitor.get_performance_summary()
            }
            
            self.logger.info('✅ Optimized Step 2 Validation completed successfully')
            self.logger.info(f"   - Total validation time: {time.time() - self.start_time:.2f} seconds")
            self.logger.info(f"   - Data shape: {data.shape}")
            self.logger.info(f"   - Issues: {content_result['total_issues']}")
            self.logger.info(f"   - Warnings: {content_result['total_warnings']}")
            
            return final_result
            
        except Exception as e:
            self.logger.exception(f'❌ Error in Optimized Step 2 validation: {e}')
            return {
                'step_name': 'step02_data_reading_validation',
                'validation_passed': False,
                'error': f'Validation error: {e}'
            }
    
    async def generate_validation_report(self, training_input: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        try:
            self.logger.info('📊 Generating comprehensive validation report...')
            
            # Get performance summary
            performance_summary = self.monitor.get_performance_summary()
            
            # Prepare report data
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            
            report_data = {
                'step': 'step02_data_reading_validation_optimized',
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'validation_result': validation_result,
                'performance_summary': performance_summary,
                'validation_timings': self.validation_timings,
                'total_validation_time': time.time() - self.start_time if self.start_time else 0
            }
            
            self.logger.info('📊 Validation Report Summary:')
            self.logger.info(f'   - Validation passed: {validation_result.get("validation_passed", False)}')
            self.logger.info(f'   - Data shape: {validation_result.get("data_shape", "Unknown")}')
            self.logger.info(f'   - Issues: {validation_result.get("total_issues", 0)}')
            self.logger.info(f'   - Warnings: {validation_result.get("total_warnings", 0)}')
            self.logger.info(f'   - Total validation time: {time.time() - self.start_time:.2f}s')
            
            if performance_summary:
                self.logger.info(f'   - Function calls: {performance_summary.get("total_calls", 0)}')
                self.logger.info(f'   - Success rate: {performance_summary.get("success_rate", 0):.1f}%')
                self.logger.info(f'   - Avg execution time: {performance_summary.get("avg_execution_time", 0):.3f}s')
            
            return {
                'success': True,
                'report_data': report_data,
                'validation_passed': validation_result.get('validation_passed', False)
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error generating validation report: {e}')
            return {
                'success': False,
                'error': str(e)
            }

# Entry point function
async def run_validator_optimized(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimized entry point for Step 2 validation."""
    config = {
        'timeout_seconds': 300,
        'max_workers': 4,
        'min_rows': 1000,
        'max_duplicate_ratio': 0.01,
        'max_gap_seconds': 0.5
    }
    
    validator = OptimizedStep2Validator(config)
    await validator.initialize()
    
    # Run validation
    validation_result = await validator.run_validation(training_input, pipeline_state)
    
    # Generate report
    report_result = await validator.generate_validation_report(training_input, validation_result)
    
    return {
        'validation_result': validation_result,
        'report_result': report_result
    }

if __name__ == '__main__':
    async def test():
        test_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache'
        }
        test_state = {}
        
        result = await run_validator_optimized(test_input, test_state)
        print(f'Validation result: {result}')
    
    asyncio.run(test())