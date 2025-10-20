from src.utils.tprint import tprint

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Step 1: Data Collection - Refactored to use BaseStep.

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

from src.core.decorators import handles_errors
from src.training.steps.base_step import BaseStep
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
import pandas as pd
import numpy as np

class DataCollectionStep(BaseStep):
    """Step 1: Data Collection using standardized base class."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize data collection step.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '01', 'data_collection')
        self.lookback_years = config.get('lookback_years', 2)
        self.data_sources = config.get('data_sources', ['binance'])  # Default to binance for backward compatibility
        self.intervals = config.get('intervals', ['1m'])
        self.data_downloader = None
    @log_step_functions

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            from src.training.steps.data_collection.data_downloader import download_all_data_with_consolidation
            import logging
            import time

            self.data_downloader = download_all_data_with_consolidation
            self.logger.info('✅ Data downloader initialized')
        except ImportError:
            self.logger.warning('⚠️ Data downloader not available, will use mock data')
    @log_step_functions

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if not training_input.get('symbol'):
            errors.append('Symbol is required')
        if not training_input.get('exchange'):
            errors.append('Exchange is required')
        symbol = training_input.get('symbol', '')
        if symbol and (not symbol.isupper()):
            errors.append(f'Symbol should be uppercase, got: {symbol}')
        valid_exchanges = ['binance', 'bybit', 'okx', 'kraken']
        exchange = training_input.get('exchange', '').lower()
        if exchange and exchange not in valid_exchanges:
            errors.append(f'Invalid exchange: {exchange}. Valid: {valid_exchanges}')
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        timeframe = training_input.get('timeframe', '1m')
        if timeframe not in valid_timeframes:
            errors.append(f'Invalid timeframe: {timeframe}. Valid: {valid_timeframes}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='data collection execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection logic.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state
        """
        symbol = training_input['symbol']
        exchange = training_input['exchange']
        timeframe = training_input.get('timeframe', '1m')
        data_dir = training_input.get('data_dir', 'data_cache')
        self.logger.info(f'📥 Collecting data for {symbol} on {exchange} ({timeframe})')
        output_dir = Path(data_dir)
        output_dir.mkdir(parents = True, exist_ok = True)
        consolidated_file = output_dir / f'klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet'
        if consolidated_file.exists() and (not training_input.get('force_download', False)):
            self.logger.info(f'📁 Found existing data: {consolidated_file}')
            try:
                data = standardized_parquet_handler.read_parquet_standardized(consolidated_file)
                self.logger.info(f'✅ Loaded {len(data)} rows of existing data')
                pipeline_state['raw_market_data'] = consolidated_file
                pipeline_state['data_shape'] = data.shape
                pipeline_state['data_columns'] = list(data.columns)
                pipeline_state['data_date_range'] = {'start': str(data.index.min() if not data.empty else None), 'end': str(data.index.max() if not data.empty else None)}
                pipeline_state['step01_data_collection_completed'] = True
                return pipeline_state
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to load existing data: {e}')
        if self.data_downloader:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days = 365 * self.lookback_years)
                self.logger.info(f'📥 Downloading data from {start_date.date()} to {end_date.date()}')
                success = await self.data_downloader(symbol = symbol, exchange = exchange, interval = timeframe, start_date = start_date, end_date = end_date, output_dir = str(output_dir))
                if success and consolidated_file.exists():
                    data = standardized_parquet_handler.read_parquet_standardized(consolidated_file)
                    self.logger.info(f'✅ Downloaded {len(data)} rows of data')
                    pipeline_state['raw_market_data'] = str(consolidated_file)
                    pipeline_state['data_shape'] = data.shape
                    pipeline_state['data_columns'] = list(data.columns)
                    pipeline_state['data_date_range'] = {'start': str(data.index.min() if not data.empty else None), 'end': str(data.index.max() if not data.empty else None)}
                    pipeline_state['step01_data_collection_completed'] = True
                else:
                    raise RuntimeError('Data download failed')
            except Exception as e:
                self.logger.error(f'❌ Data download failed: {e}')
                raise RuntimeError(f'Data download failed: {e}')
        else:
            raise RuntimeError('Data downloader not available')
        pipeline_state['data_collection_metadata'] = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'lookback_years': self.lookback_years, 'download_timestamp': datetime.now().isoformat()}
        pipeline_state['step01_data_collection_completed'] = True
        return pipeline_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.

        Args:
            pipeline_state: Updated pipeline state

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'raw_market_data' not in pipeline_state:
            errors.append('No raw_market_data in pipeline state')
            return (False, errors)
        data_path = Path(pipeline_state['raw_market_data'])
        if not data_path.exists():
            errors.append(f'Data file does not exist: {data_path}')
            return (False, errors)
        if 'data_shape' in pipeline_state:
            rows, cols = pipeline_state['data_shape']
            if rows < 100:
                errors.append(f'Insufficient data rows: {rows} (minimum 100 required)')
            if cols < 5:
                errors.append(f'Insufficient columns: {cols} (minimum 5 required)')
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if 'data_columns' in pipeline_state:
            missing_cols = set(required_columns) - set(pipeline_state['data_columns'])
            if missing_cols:
                errors.append(f'Missing required columns: {missing_cols}')
        return (len(errors) == 0, errors)
    @log_all_calls

    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ['symbol', 'exchange', 'timeframe']

    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ['raw_market_data', 'data_shape', 'data_columns', 'data_date_range']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return []

@handles_errors(fallback=False)
async def run_step(symbol: str, exchange: str, timeframe: str = '1m', data_dir: str = None, force_rerun: bool = False) -> bool:
    """Run the data collection step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re-run even if results exist

    Returns:
        bool: True if successful, False otherwise
    """

    tprint('\n' + '=' * 80)
    tprint('🚀 STEP 1: DATA COLLECTION - STARTING EXECUTION')
    tprint('=' * 80)
    tprint(f'🎯 Symbol: {symbol}')
    tprint(f'🏢 Exchange: {exchange}')
    tprint(f'📊 Timeframe: {timeframe}')
    if data_dir is None:
        data_dir = 'data_cache'
    tprint(f'📁 Data directory: {data_dir}')
    tprint(f'🔄 Force rerun: {force_rerun}')
    tprint(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tprint('=' * 80)

    start_time = time.time()

    try:
        # Create step configuration
        config = {
            'SYMBOL': symbol,
            'EXCHANGE': exchange,
            'TIMEFRAME': timeframe,
            'DATA_DIR': data_dir,
            'lookback_years': 2,  # Default value
            'data_sources': ['binance'],  # Default value
            'intervals': [timeframe]  # Use provided timeframe
        }

        # Initialize and run the step
        step = DataCollectionStep(config)
        await step.initialize()

        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'force_rerun': force_rerun
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        elapsed_time = time.time() - start_time

        if result.get('data_collection_completed', False):
            tprint('✅ Step 1: Data Collection completed successfully')
            tprint(f'⏱️ Total execution time: {elapsed_time:.2f} seconds')
            tprint(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            tprint('=' * 80)
            return True
        else:
            tprint('❌ Step 1: Data Collection failed')
            error = result.get('data_collection_error', 'Unknown error')
            tprint(f'   Error: {error}')
            tprint(f'⏱️ Total execution time: {elapsed_time:.2f} seconds')
            tprint('=' * 80)
            return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        tprint('💥 STEP 1 EXECUTION ERROR')
        tprint('=' * 80)
        tprint(f'❌ Error: {str(e)}')
        tprint(f'⏱️ Total execution time: {elapsed_time:.2f} seconds')
        tprint('=' * 80)
        return False
