"""Validator for Step 4: Triple Barrier Method.

This module validates the triple barrier method step outputs.
"""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.logger import system_logger
from src.core.domain import comprehensive_data_validation, handle_errors, memory_efficient, resource_monitor, secure_data_processing, validate_data_structure, with_tracing_span, quality_gate
from src.core.decorators import handles_errors, traced, validates
logger = system_logger.getChild('Step4TripleBarrierMethodValidator')

@traced(span_name='validate_triple_barrier_method')
@validates()
@handles_errors
async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run validation for Step 4: Triple Barrier Method.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info('🔍 Validating Step 4: Triple Barrier Method')
    try:
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'BINANCE')
        timeframe = training_input.get('timeframe', '1m')
        data_dir = training_input.get('data_dir', 'data_cache')
        triple_barrier_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
        if not triple_barrier_path.exists():
            logger.error(f'❌ Triple barrier labels file not found: {triple_barrier_path}')
            return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': False, 'error': f'Triple barrier labels file not found: {triple_barrier_path}'}
        file_size = triple_barrier_path.stat().st_size
        if file_size == 0:
            logger.error(f'❌ Triple barrier labels file is empty: {triple_barrier_path}')
            return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': False, 'error': 'Triple barrier labels file is empty'}
        try:
            import pandas as pd
from src.core.decorators.errors import handles_errors
            data = pd.read_parquet(triple_barrier_path)
            required_columns = ['triple_barrier_label']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f'❌ Missing required columns: {missing_columns}')
                return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': False, 'error': f'Missing required columns: {missing_columns}'}
            if len(data) == 0:
                logger.error('❌ No data rows found')
                return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': False, 'error': 'No data rows found'}
            label_counts = data['triple_barrier_label'].value_counts()
            logger.info(f'✅ Label distribution: {label_counts.to_dict()}')
            if 0 in label_counts and label_counts[0] == len(data):
                logger.warning('⚠️ All labels are 0 (hold) - this might indicate an issue')
                return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': True, 'warning': 'All labels are 0 (hold) - this might indicate an issue'}
            logger.info('✅ Step 4: Triple Barrier Method validation passed')
            return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': True, 'file_path': str(triple_barrier_path), 'data_shape': data.shape, 'label_distribution': label_counts.to_dict()}
        except Exception as e:
            logger.error(f'❌ Error reading triple barrier labels file: {e}')
            return {'step_name': 'step04_5_triple_barrier_method', 'validation_passed': False, 'error': f'Error reading file: {e}'}
    except Exception as e:
        logger.exception(f'❌ Error in Step 4 validation: {e}')
        return {'step_name': 'step4_triple_barrier_method', 'validation_passed': False, 'error': f'Validation error: {e}'}
if __name__ == '__main__':

    async def test() -> None:
        test_input = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m', 'data_dir': 'data_cache'}
        test_state = {}
        result = await run_validator(test_input, test_state)
        print(f'Validation result: {result}')
    asyncio.run(test())