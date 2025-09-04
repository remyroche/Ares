"""Validator for Step 2: Data Reading."

This module validates the data reading step outputs with comprehensive quality checks.
"""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from src.core.decorators import handles_errors, traced, validates
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.logger import system_logger
from src.core.domain import comprehensive_data_validation, handle_errors, memory_efficient, resource_monitor, secure_data_processing, validate_data_structure, with_tracing_span, quality_gate
from src.utils.common_operations import safe_json_load
logger = system_logger.getChild('Step2DataReadingValidator')

@traced(span_name='validate_data_reading')
@validates()
@handles_errors
async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run validation for Step 2: Data Reading."

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info('🔍 Validating Step 2: Data Reading')
    try:
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'BINANCE')
        timeframe = training_input.get('timeframe', '1m')
        data_dir = training_input.get('data_dir', 'data_cache')
        unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
        if not unified_data_path.exists():
            logger.error(f'❌ Unified data directory not found: {unified_data_path}')
            return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': f'Unified data directory not found: {unified_data_path}'}
        data_files = list(unified_data_path.glob('*.parquet'))
        if not data_files:
            logger.error(f'❌ No parquet files found in {unified_data_path}')
            return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': f'No parquet files found in {unified_data_path}'}
        validation_report_path = Path(data_dir) / f'{exchange}_{symbol}_{timeframe}_validation_report.json'
        try:
            import pandas as pd
            import json
            import numpy as np
from src.core.decorators.errors import handles_errors
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            data = pd.read_parquet(latest_file)
            if len(data) == 0:
                logger.error('❌ No data rows found')
                return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': 'No data rows found'}
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f'❌ Missing required columns: {missing_columns}')
                return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': f'Missing required columns: {missing_columns}'}
            if 'timestamp' not in data.columns:
                logger.error('❌ Timestamp column not found')
                return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': "Missing required 'timestamp' column"}
            ts_is_datetime = data['timestamp'].dtype.kind == 'M'
            ts_is_numeric = str(data['timestamp'].dtype).startswith('int') or str(data['timestamp'].dtype).startswith('float')
            if not (ts_is_datetime or ts_is_numeric):
                logger.error("❌ 'timestamp' must be datetime64 or numeric (ms)")
                return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': "'timestamp' must be datetime64 or numeric (ms)"}
            nan_count = data[required_columns].isna().sum().sum()
            if nan_count > 0:
                logger.warning(f'⚠️ Found {nan_count} NaN values in required columns')
            inf_count = data[required_columns].isin([float('inf'), float('-inf')]).sum().sum()
            if inf_count > 0:
                logger.warning(f'⚠️ Found {inf_count} infinite values in required columns')
            negative_prices = (data[['open', 'high', 'low', 'close']] < 0).sum().sum()
            if negative_prices > 0:
                logger.error(f'❌ Found {negative_prices} negative price values')
                return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': f'Found {negative_prices} negative price values'}
            zero_prices = (data[['open', 'high', 'low', 'close']] == 0).sum().sum()
            if zero_prices > 0:
                logger.warning(f'⚠️ Found {zero_prices} zero price values')
            price_stats = data[['open', 'high', 'low', 'close']].describe()
            logger.info(f'✅ Price statistics: {price_stats.to_dict()}')
            volume_stats = data['volume'].describe()
            logger.info(f'✅ Volume statistics: {volume_stats.to_dict()}')
            ohlc_errors = 0
            for idx, row in data.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and row['low'] <= row['close'] <= row['high']):
                    ohlc_errors += 1
            if ohlc_errors > 0:
                logger.warning(f'⚠️ Found {ohlc_errors} OHLC consistency errors')
            if 'timestamp' in data.columns:
                duplicate_timestamps = data['timestamp'].duplicated().sum()
                if duplicate_timestamps > 0:
                    logger.warning(f'⚠️ Found {duplicate_timestamps} duplicate timestamps')
            if 'timestamp' in data.columns:
                data_sorted = data.sort_values('timestamp')
                time_diffs = data_sorted['timestamp'].diff().dropna()
                if len(time_diffs) > 0:
                    avg_time_diff = time_diffs.mean()
                    logger.info(f'✅ Average time difference: {avg_time_diff}')
            validation_metadata = {}
            if validation_report_path.exists():
                try:
                    validation_metadata = safe_json_load(validation_report_path)
                    logger.info('✅ Validation report found and loaded')
                except Exception as e:
                    logger.warning(f'⚠️ Error reading validation report: {e}')
            logger.info(f'✅ Data shape: {data.shape}')
            logger.info(f'✅ Number of files: {len(data_files)}')
            logger.info(f'✅ Latest file: {latest_file.name}')
            logger.info('✅ Step 2: Data Reading validation passed')
            return {'step_name': 'step02_data_reading', 'validation_passed': True, 'data_file_path': str(latest_file), 'validation_report_path': str(validation_report_path) if validation_report_path.exists() else None, 'data_shape': data.shape, 'nan_count': nan_count, 'inf_count': inf_count, 'ohlc_errors': ohlc_errors, 'price_stats': price_stats.to_dict(), 'volume_stats': volume_stats.to_dict(), 'validation_metadata': validation_metadata}
        except Exception as e:
            logger.error(f'❌ Error reading data files: {e}')
            return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': f'Error reading files: {e}'}
    except Exception as e:
        logger.exception(f'❌ Error in Step 2 validation: {e}')
        return {'step_name': 'step02_data_reading', 'validation_passed': False, 'error': f'Validation error: {e}'}
if __name__ == '__main__':

    async def test() -> None:
        test_input = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m', 'data_dir': 'data_cache'}
        test_state = {}
        result = await run_validator(test_input, test_state)
        print(f'Validation result: {result}')
    asyncio.run(test())