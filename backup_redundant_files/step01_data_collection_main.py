from src.utils.tprint import tprint

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Step 1: Data Collection Pipeline.

This module provides the main interface for data collection with:
1. Raw data collection from exchanges
2. Data quality validation
3. Unified data loading
4. Data conversion and preprocessing
"""
import asyncio
import sys
from pathlib import Path
import time
import json

# Import utility modules
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
from src.utils.validation import validate_data_quality
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from .enhanced_data_collection_pipeline import run_enhanced_data_collection_pipeline

async def main() -> None:
    """Main function to run data collection pipeline."""
    tprint('🚀 Step 1: Data Collection Pipeline')
    tprint('=' * 80)
    symbol = 'ETHUSDT'
    exchange = 'BINANCE'
    timeframe = '1m'
    data_dir = 'data_cache'
    config = {'force_rerun': True, 'quality_checks': True, 'validate_data': True, 'convert_format': True, 'random_state': 42}
    tprint(f'📊 Configuration:')
    tprint(f'   Symbol: {symbol}')
    tprint(f'   Exchange: {exchange}')
    tprint(f'   Timeframe: {timeframe}')
    tprint(f'   Data directory: {data_dir}')
    tprint(f"   Force rerun: {config['force_rerun']}")
    tprint(f"   Quality checks: {config['quality_checks']}")
    tprint('=' * 80)
    start_time = time.time()
    try:
        result = await run_enhanced_data_collection_pipeline(symbol = symbol, exchange = exchange, data_dir = data_dir, config = config)
        success = result.get('success', False)
        total_time = time.time() - start_time
        if success:
            tprint('\n🎉 DATA COLLECTION COMPLETED SUCCESSFULLY!')
            tprint('=' * 80)
            tprint('✅ All data collection steps completed:')
            tprint('   ✅ Raw data collection from exchange')
            tprint('   ✅ Data quality validation')
            tprint('   ✅ Unified data loading')
            tprint('   ✅ Data conversion and preprocessing')
            tprint(f'⏱️ Total execution time: {total_time:.2f} seconds')
            tprint('=' * 80)
            config_file = Path(data_dir) / f'data_collection_config_{symbol}_{timeframe}.json'
            with open(config_file, 'w') as f:
                json.dump({'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'config': config, 'execution_time': total_time, 'success': True}, f, indent = 2)
            tprint(f'💾 Configuration saved to: {config_file}')
        else:
            tprint('\n❌ DATA COLLECTION FAILED!')
            tprint('=' * 80)
            tprint('❌ Please check the logs for error details')
            tprint(f'⏱️ Total execution time: {total_time:.2f} seconds')
            tprint('=' * 80)
    except Exception as e:
        total_time = time.time() - start_time
        tprint(f'\n💥 DATA COLLECTION FAILED WITH EXCEPTION: {e}')
        tprint('=' * 80)
        tprint(f'⏱️ Total execution time: {total_time:.2f} seconds')
        tprint('=' * 80)
        raise
if __name__ == '__main__':
    asyncio.run(main())