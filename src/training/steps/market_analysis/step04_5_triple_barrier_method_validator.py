from ....core.decorators import handles_errors
"""Validator for Step 4: Triple Barrier Method.
from src.utils.logger import system_logger

This module validates the triple barrier method step outputs.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Standardized imports from utils
from src.utils.common_operations import (
    safe_read_parquet,
    safe_file_exists,
    get_logger,
    safe_dict_get,
    safe_float,
    safe_int
)
from src.utils.decorators import (
    traced,
    validates
)
from src.core.decorators import log_execution_time
from src.utils.logger import system_logger

# Project setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import decorators from core location
import logging
import time

# Initialize logger using common utilities
logger = get_logger('Step4TripleBarrierMethodValidator')

@traced(span_name='validate_triple_barrier_method')
@validates()
@handles_errors()
@log_execution_time()
def _validate_file_exists(triple_barrier_path: Path) -> Optional[Dict[str, Any]]:
    """Validate that the triple barrier file exists and is not empty."""
    if not safe_file_exists(triple_barrier_path):
        logger.error(f'❌ Triple barrier labels file not found: {triple_barrier_path}')
        return {
            'step_name': 'step04_5_triple_barrier_method', 
            'validation_passed': False, 
            'error': f'Triple barrier labels file not found: {triple_barrier_path}'
        }
    
    try:
        file_size = triple_barrier_path.stat().st_size
        if file_size == 0:
            logger.error(f'❌ Triple barrier labels file is empty: {triple_barrier_path}')
            return {
                'step_name': 'step04_5_triple_barrier_method', 
                'validation_passed': False, 
                'error': 'Triple barrier labels file is empty'
            }
        
        # Log file size for monitoring
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f'📊 Triple barrier file size: {file_size_mb:.2f} MB')
        
        return None
        
    except Exception as e:
        logger.error(f'❌ Error checking file size: {e}')
        return {
            'step_name': 'step04_5_triple_barrier_method', 
            'validation_passed': False, 
            'error': f'Error checking file: {e}'
        }

@traced(span_name='validate_data_content')
@handles_errors()
def _validate_data_content(data: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate the content of the loaded data."""
    try:
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.error(f'❌ Expected DataFrame, got {type(data)}')
            return {
                'step_name': 'step04_5_triple_barrier_method', 
                'validation_passed': False, 
                'error': f'Expected DataFrame, got {type(data)}'
            }
        
        # Check for required columns (fixed column name)
        required_columns = ['triple_barrier_label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f'❌ Missing required columns: {missing_columns}')
            logger.info(f'📋 Available columns: {list(data.columns)}')
            return {
                'step_name': 'step04_5_triple_barrier_method', 
                'validation_passed': False, 
                'error': f'Missing required columns: {missing_columns}'
            }
        
        if len(data) == 0:
            logger.error('❌ No data rows found')
            return {
                'step_name': 'step04_5_triple_barrier_method', 
                'validation_passed': False, 
                'error': 'No data rows found'
            }
        
        # Log data shape for monitoring
        logger.info(f'📊 Data shape: {data.shape}')
        
        return None
        
    except Exception as e:
        logger.exception(f'❌ Error validating data content: {e}')
        return {
            'step_name': 'step04_5_triple_barrier_method', 
            'validation_passed': False, 
            'error': f'Error validating data: {e}'
        }

@traced(span_name='validate_label_distribution')
@handles_errors()
def _validate_label_distribution(data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the distribution of labels in the data."""
    try:
        if not isinstance(data, pd.DataFrame):
            logger.error(f'❌ Expected DataFrame for label validation, got {type(data)}')
            return {
                'step_name': 'step04_5_triple_barrier_method', 
                'validation_passed': False, 
                'error': f'Expected DataFrame, got {type(data)}'
            }
        
        # Get label distribution
        label_counts = data['triple_barrier_label'].value_counts()
        label_distribution = label_counts.to_dict()
        
        logger.info(f'✅ Label distribution: {label_distribution}')
        
        # Check for potential issues
        warnings = []
        
        # Check if all labels are 0 (hold)
        if 0 in label_counts and label_counts[0] == len(data):
            warning_msg = 'All labels are 0 (hold) - this might indicate an issue'
            logger.warning(f'⚠️ {warning_msg}')
            warnings.append(warning_msg)
        
        # Check for extreme imbalance
        total_labels = len(data)
        if total_labels > 0:
            for label, count in label_distribution.items():
                percentage = (count / total_labels) * 100
                if percentage > 95:
                    warning_msg = f'Label {label} represents {percentage:.1f}% of data - extreme imbalance'
                    logger.warning(f'⚠️ {warning_msg}')
                    warnings.append(warning_msg)
        
        # Check for missing labels (should have both 1 and -1 for binary classification)
        if 1 not in label_distribution or -1 not in label_distribution:
            warning_msg = 'Missing buy (1) or sell (-1) labels - check triple barrier configuration'
            logger.warning(f'⚠️ {warning_msg}')
            warnings.append(warning_msg)
        
        result = {
            'step_name': 'step04_5_triple_barrier_method', 
            'validation_passed': True, 
            'data_shape': data.shape, 
            'label_distribution': label_distribution,
            'total_labels': total_labels
        }
        
        if warnings:
            result['warnings'] = warnings
        
        logger.info('✅ Step 4: Triple Barrier Method validation passed')
        return result
        
    except Exception as e:
        logger.exception(f'❌ Error validating label distribution: {e}')
        return {
            'step_name': 'step04_5_triple_barrier_method', 
            'validation_passed': False, 
            'error': f'Error validating labels: {e}'
        }

@traced(span_name='run_validator_step04_5')
@handles_errors()
@log_execution_time()
async def run_validator(
    training_input: Dict[str, Any], 
    pipeline_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Run validation for Step 4: Triple Barrier Method.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info('🔍 Validating Step 4: Triple Barrier Method')
    
    try:
        # Extract parameters with safe defaults
        symbol = safe_dict_get(training_input, 'symbol', 'ETHUSDT')
        exchange = safe_dict_get(training_input, 'exchange', 'BINANCE')
        timeframe = safe_dict_get(training_input, 'timeframe', '1m')
        data_dir = safe_dict_get(training_input, 'data_dir', 'data_cache')
        
        # Construct file path
        triple_barrier_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
        
        logger.info(f'📁 Validating file: {triple_barrier_path}')
        
        # Step 1: Validate file exists and is not empty
        file_validation_error = _validate_file_exists(triple_barrier_path)
        if file_validation_error:
            return file_validation_error
        
        # Step 2: Load and validate data content
        try:
            data = safe_read_parquet(triple_barrier_path)
            if data is None or data.empty:
                logger.error('❌ Failed to load data or data is empty')
                return {
                    'step_name': 'step04_5_triple_barrier_method', 
                    'validation_passed': False, 
                    'error': 'Failed to load data or data is empty'
                }
            
            # Step 3: Validate data content
            content_validation_error = _validate_data_content(data)
            if content_validation_error:
                return content_validation_error
            
            # Step 4: Validate label distribution
            result = _validate_label_distribution(data)
            result['file_path'] = str(triple_barrier_path)
            result['file_size_mb'] = triple_barrier_path.stat().st_size / (1024 * 1024)
            
            logger.info('✅ Step 4: Triple Barrier Method validation completed successfully')
            return result
            
        except Exception as e:
            logger.exception(f'❌ Error reading triple barrier labels file: {e}')
            return {
                'step_name': 'step04_5_triple_barrier_method', 
                'validation_passed': False, 
                'error': f'Error reading file: {e}'
            }
            
    except Exception as e:
        logger.exception(f'❌ Error in Step 4 validation: {e}')
        return {
            'step_name': 'step04_5_triple_barrier_method', 
            'validation_passed': False, 
            'error': f'Validation error: {e}'
        }
if __name__ == '__main__':

    async def test() -> None:
        test_input = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m', 'data_dir': 'data_cache'}
        test_state = {}
        result = await run_validator(test_input, test_state)
        print(f'Validation result: {result}')
    asyncio.run(test())