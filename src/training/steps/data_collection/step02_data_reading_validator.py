"""Validator for Step 2: Data Reading.

This module validates the data reading step outputs with comprehensive quality checks.
"""
import asyncio
import sys
from pathlib import Path

from .core.decorators import handles_errors, traced, validates
from .core.decorators.errors import handles_errors

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .utils.logger import system_logger
from src.utils.trading_decorators import (
    comprehensive_data_validation, 
    handle_errors, 
    memory_efficient, 
    resource_monitor,
    secure_data_processing, 
    validate_data_structure, 
    with_tracing_span, 
    quality_gate
)
from .utils.common_operations import safe_json_load
from typing import Any
from typing import Dict
import pandas as pd

logger = system_logger.getChild('Step2DataReadingValidator')


async def _validate_directory_structure(data_dir: str, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Validate directory structure exists."""
    unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
    
    if not unified_data_path.exists():
        error_msg = f'Unified data directory not found: {unified_data_path}'
        logger.error(f'❌ {error_msg}')
        return {
            'step_name': 'step02_data_reading', 
            'validation_passed': False, 
            'error': error_msg
        }
    
    data_files = list(unified_data_path.glob('*.parquet'))
    if not data_files:
        error_msg = f'No parquet files found in {unified_data_path}'
        logger.error(f'❌ {error_msg}')
        return {
            'step_name': 'step02_data_reading', 
            'validation_passed': False, 
            'error': error_msg
        }
    
    return {
        'validation_passed': True,
        'data_files': data_files,
        'unified_data_path': unified_data_path
    }


async def _validate_data_files(data_files: list, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Validate data files and load the latest one."""
    try:
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        data = pd.read_parquet(latest_file)
        
        if len(data) == 0:
            error_msg = 'No data rows found'
            logger.error(f'❌ {error_msg}')
            return {
                'step_name': 'step02_data_reading', 
                'validation_passed': False, 
                'error': error_msg
            }
        
        return {
            'validation_passed': True,
            'data': data,
            'latest_file': latest_file
        }
        
    except Exception as e:
        error_msg = f'Error reading data files: {e}'
        logger.error(f'❌ {error_msg}')
        return {
            'step_name': 'step02_data_reading', 
            'validation_passed': False, 
            'error': error_msg
        }


async def _validate_data_content(data: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Validate data content and structure."""
    try:
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f'Missing required columns: {missing_columns}'
            logger.error(f'❌ {error_msg}')
            return {
                'step_name': 'step02_data_reading', 
                'validation_passed': False, 
                'error': error_msg
            }
        
        # Check timestamp column
        if 'timestamp' not in data.columns:
            error_msg = "Missing required 'timestamp' column"
            logger.error(f'❌ {error_msg}')
            return {
                'step_name': 'step02_data_reading', 
                'validation_passed': False, 
                'error': error_msg
            }
        
        # Validate timestamp format
        ts_is_datetime = data['timestamp'].dtype.kind == 'M'
        ts_is_numeric = str(data['timestamp'].dtype).startswith('int') or str(data['timestamp'].dtype).startswith('float')
        
        if not (ts_is_datetime or ts_is_numeric):
            error_msg = "'timestamp' must be datetime64 or numeric (ms)"
            logger.error(f'❌ {error_msg}')
            return {
                'step_name': 'step02_data_reading', 
                'validation_passed': False, 
                'error': error_msg
            }
        
        # Check for data quality issues
        nan_count = data[required_columns].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f'⚠️ Found {nan_count} NaN values in required columns')
        
        inf_count = data[required_columns].isin([float('inf'), float('-inf')]).sum().sum()
        if inf_count > 0:
            logger.warning(f'⚠️ Found {inf_count} infinite values in required columns')
        
        # Check for negative prices
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).sum().sum()
        if negative_prices > 0:
            error_msg = f'Found {negative_prices} negative price values'
            logger.error(f'❌ {error_msg}')
            return {
                'step_name': 'step02_data_reading', 
                'validation_passed': False, 
                'error': error_msg
            }
        
        # Check for zero prices
        zero_prices = (data[['open', 'high', 'low', 'close']] == 0).sum().sum()
        if zero_prices > 0:
            logger.warning(f'⚠️ Found {zero_prices} zero price values')
        
        # Generate statistics
        price_stats = data[['open', 'high', 'low', 'close']].describe()
        volume_stats = data['volume'].describe()
        logger.info(f'✅ Price statistics: {price_stats.to_dict()}')
        logger.info(f'✅ Volume statistics: {volume_stats.to_dict()}')
        
        # Check OHLC consistency
        ohlc_errors = 0
        for idx, row in data.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and row['low'] <= row['close'] <= row['high']):
                ohlc_errors += 1
        
        if ohlc_errors > 0:
            logger.warning(f'⚠️ Found {ohlc_errors} OHLC consistency errors')
        
        # Check for duplicate timestamps
        duplicate_timestamps = data['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            logger.warning(f'⚠️ Found {duplicate_timestamps} duplicate timestamps')
        
        # Calculate time differences
        data_sorted = data.sort_values('timestamp')
        time_diffs = data_sorted['timestamp'].diff().dropna()
        if len(time_diffs) > 0:
            avg_time_diff = time_diffs.mean()
            logger.info(f'✅ Average time difference: {avg_time_diff}')
        
        return {
            'validation_passed': True,
            'data': data,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'ohlc_errors': ohlc_errors,
            'price_stats': price_stats.to_dict(),
            'volume_stats': volume_stats.to_dict()
        }
        
    except Exception as e:
        error_msg = f'Error validating data content: {e}'
        logger.error(f'❌ {error_msg}')
        return {
            'step_name': 'step02_data_reading', 
            'validation_passed': False, 
            'error': error_msg
        }


@traced(span_name='validate_data_reading')
@validates()
@handles_errors
async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run validation for Step 2: Data Reading.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info('🔍 Validating Step 2: Data Reading')
    
    try:
        # Extract parameters
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'BINANCE')
        timeframe = training_input.get('timeframe', '1m')
        data_dir = training_input.get('data_dir', 'data_cache')
        
        # Validate directory structure
        validation_result = await _validate_directory_structure(data_dir, exchange, symbol, timeframe)
        if not validation_result['validation_passed']:
            return validation_result
            
        # Validate data files
        validation_result = await _validate_data_files(
            validation_result['data_files'], exchange, symbol, timeframe
        )
        if not validation_result['validation_passed']:
            return validation_result
            
        # Validate data content
        validation_result = await _validate_data_content(
            validation_result['data'], exchange, symbol, timeframe
        )
        if not validation_result['validation_passed']:
            return validation_result
        
        # Load validation metadata if available
        validation_report_path = Path(data_dir) / f'{exchange}_{symbol}_{timeframe}_validation_report.json'
        validation_metadata = {}
        
        if validation_report_path.exists():
            try:
                validation_metadata = safe_json_load(validation_report_path)
                logger.info('✅ Validation report found and loaded')
            except Exception as e:
                logger.warning(f'⚠️ Error reading validation report: {e}')
        
        # Log success information
        logger.info(f'✅ Data shape: {validation_result["data"].shape}')
        logger.info(f'✅ Number of files: {len(validation_result["data_files"])}')
        logger.info(f'✅ Latest file: {validation_result["latest_file"].name}')
        logger.info('✅ Step 2: Data Reading validation passed')
        
        return {
            'step_name': 'step02_data_reading', 
            'validation_passed': True, 
            'data_file_path': str(validation_result['latest_file']), 
            'validation_report_path': str(validation_report_path) if validation_report_path.exists() else None, 
            'data_shape': validation_result['data'].shape, 
            'nan_count': validation_result['nan_count'], 
            'inf_count': validation_result['inf_count'], 
            'ohlc_errors': validation_result['ohlc_errors'], 
            'price_stats': validation_result['price_stats'], 
            'volume_stats': validation_result['volume_stats'], 
            'validation_metadata': validation_metadata
        }
        
    except Exception as e:
        logger.exception(f'❌ Error in Step 2 validation: {e}')
        return {
            'step_name': 'step02_data_reading', 
            'validation_passed': False, 
            'error': f'Validation error: {e}'
        }


if __name__ == '__main__':
    async def test() -> None:
        test_input = {
            'symbol': 'ETHUSDT', 
            'exchange': 'BINANCE', 
            'timeframe': '1m', 
            'data_dir': 'data_cache'
        }
        test_state = {}
        result = await run_validator(test_input, test_state)
        print(f'Validation result: {result}')
    
    asyncio.run(test())