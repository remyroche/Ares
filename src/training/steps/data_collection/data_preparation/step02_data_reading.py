from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

"""Step 2: Data Reading - Refactored to use BaseStep with Hardware Optimizations.
from src.utils.logger import system_logger

This module handles reading the unified data from step1_5 and performs comprehensive
data quality validation before proceeding to HMM regime discovery with M1 hardware acceleration.
"""
from src.core.decorators import handles_errors
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from pathlib import Path
from src.training.base_step import BaseStep
from src.utils.common_operations import validate_dataframe_schema, validate_data_quality
from src.utils.parquet_utils import ParquetUtils
from src.utils.pipeline_standards import PipelineStandards
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)
from typing import Any, Dict, Tuple
import pandas as pd
from src.utils.logger import system_logger

# Import optimization utilities for enhanced performance
try:
    from src.utils.matrix_operations import get_vectorized_processing_core
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    import logging
    import os
    import time
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False

class DataReadingStep(BaseStep):
    """Step 2: Data Reading and Validation using standardized base class."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize data reading step.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '02', 'data_reading')
        self.logger = system_logger.getChild('DataReadingStep')
        self.data_quality_thresholds = config.get('data_quality_thresholds', {'min_rows': 1000, 'max_missing_pct': 0.05, 'min_unique_timestamps': 500})

        # Initialize optimization components
        if OPTIMIZATIONS_AVAILABLE:
            try:
                self.vectorized_core = get_vectorized_processing_core()
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.logger.info('🚀 Step 2 initialized with M1 hardware acceleration and vectorized processing')
            except Exception as e:
                self.logger.warning(f'Failed to initialize optimizations: {e}')
                self.vectorized_core = None
                self.gpu_manager = None
                self.memory_optimizer = None
        else:
            self.vectorized_core = None
            self.gpu_manager = None
            self.memory_optimizer = None
    @log_step_functions

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.logger.info('✅ Data reading step initialized')

    async def initialize(self) -> None:
        """Initialize the step."""
        self._initialize_step()

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step."""
        return await self.execute_logic(training_input, pipeline_state)
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
        data_path = pipeline_state.get('unified_data_path') or pipeline_state.get('raw_market_data')
        if not data_path:
            symbol = training_input.get('symbol', '').upper()
            exchange = training_input.get('exchange', '').upper()
            timeframe = training_input.get('timeframe', '1m')
            if symbol and exchange and timeframe:
                standards = PipelineStandards(self.logger)
                data_path = standards.build_path('unified_partitioned', exchange, symbol, timeframe=timeframe)
                self.logger.info(f'Constructed data path for validation: {data_path}')
            else:
                errors.append('No unified_data_path or raw_market_data in pipeline state, and cannot construct from training input')
        if data_path and (not Path(data_path).exists()):
            errors.append(f'Data file does not exist: {data_path}')
        for key in ['symbol', 'exchange', 'timeframe']:
            if key not in training_input:
                errors.append(f'Missing required input: {key}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False})
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data reading and validation logic.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state
        """
        # Initialize lookahead bias detector
        from datetime import datetime
        current_time = datetime.now()
        bias_detector = get_global_detector()
        bias_detector.set_current_timestamp(current_time)
        # Get data path - prefer partitioned data
        data_path = pipeline_state.get('unified_data_path') or pipeline_state.get('raw_market_data')
        if not data_path:
            symbol = training_input.get('symbol', '').upper()
            exchange = training_input.get('exchange', '').upper()
            timeframe = training_input.get('timeframe', '1m')

            # Try partitioned data first
            partitioned_path = standardized_parquet_handler.get_partitioned_path('unified_partitioned', exchange, symbol, timeframe)
            if os.path.exists(partitioned_path):
                self.logger.info(f'📊 Reading partitioned data from: {partitioned_path}')
                data = standardized_parquet_handler.read_partitioned_parquet(
                    base_path=partitioned_path,
                    schema_name='unified'
                )
                if data is not None and not data.empty:
                    self.logger.info(f'✅ Successfully loaded partitioned data: {len(data)} rows')
                else:
                    self.logger.warning('⚠️ Partitioned data directory exists but is empty or invalid')
                    data = None
            else:
                self.logger.warning(f'⚠️ Partitioned data directory not found at: {partitioned_path}')
                data = None

            # Fallback to single file if partitioned data not available
            if data is None or data.empty:
                standards = PipelineStandards(self.logger)
                data_path = standards.build_path('unified', exchange, symbol, timeframe=timeframe)
                self.logger.info(f'📖 Fallback: Reading unified data from: {data_path}')
                if os.path.exists(data_path):
                    data = standardized_parquet_handler.read_parquet_standardized(data_path, schema_name='unified')
                    if data is not None and not data.empty:
                        self.logger.info(f'✅ Successfully loaded unified data: {len(data)} rows')
                    else:
                        self.logger.warning('⚠️ Unified data file exists but is empty or invalid')
                else:
                    self.logger.warning(f'⚠️ Unified data file not found at: {data_path}')
        else:
            self.logger.info(f'📖 Reading data from provided path: {data_path}')
            data_path_obj = Path(data_path)
            if data_path_obj.is_file():
                data = standardized_parquet_handler.read_parquet_standardized(data_path, schema_name='unified')
            elif data_path_obj.is_dir():
                # Try partitioned reading first
                data = standardized_parquet_handler.read_partitioned_parquet(
                    base_path=data_path,
                    schema_name='unified'
                )
                if data is None or data.empty:
                    # Fallback to individual file reading
                    parquet_files = list(data_path_obj.glob('**/*.parquet'))
                    if not parquet_files:
                        raise ValueError(f'No parquet files found in directory: {data_path}')
                    self.logger.info(f'📁 Found {len(parquet_files)} parquet files in directory')
                    dataframes = []
                    for i, file_path in enumerate(parquet_files):
                        self.logger.info(f'📖 Reading file {i + 1}/{len(parquet_files)}: {file_path.name}')
                        df = standardized_parquet_handler.read_parquet_standardized(str(file_path), schema_name='unified')
                        if df is not None and (not df.empty):
                            dataframes.append(df)
                    if not dataframes:
                        raise ValueError(f'Failed to read any data from parquet files in {data_path}')
                    data = pd.concat(dataframes, ignore_index = True)
                    self.logger.info(f'📊 Concatenated {len(dataframes)} dataframes')
            else:
                raise ValueError(f'Path does not exist: {data_path}')

        try:
            if data is None or data.empty:
                # Attempt centralized auto re-collection and one retry
                self.logger.warning(f"⚠️ Empty data after read. Attempting auto re-collection and retry...")
                try:
                    from src.training.steps.data_collection.data_preparation.enhanced_data_quality_manager import EnhancedDataQualityManager
                    _qm2 = EnhancedDataQualityManager(str(Path(data_path).parents[3])) if len(Path(data_path).parts) > 3 else EnhancedDataQualityManager('data_cache')
                    symbol_q2 = training_input.get('symbol', symbol)
                    exchange_q2 = training_input.get('exchange', exchange)
                    timeframe_q2 = training_input.get('timeframe', timeframe)
                    import asyncio as _asyncio
                    _asyncio.get_event_loop()
                    _asyncio.run(_qm2.get_data_for_step3_step4(symbol_q2, exchange_q2, timeframe_q2))
                    if data_path_obj.is_dir():
                        parquet_files = list(data_path_obj.glob('**/*.parquet'))
                        if parquet_files:
                            dataframes = []
                            for file_path in parquet_files:
                                df = standardized_parquet_handler.read_parquet_standardized(str(file_path), schema_name='unified')
                                if df is not None and (not df.empty):
                                    try:
                                        df = PipelineStandards(self.logger).enforce_schema(df, 'unified')
                                    except Exception as _se2:
                                        self.logger.warning(f"Schema enforcement failed for retry {file_path.name}: {_se2}")
                                    dataframes.append(df)
                            if dataframes:
                                data = pd.concat(dataframes, ignore_index = True)
                except Exception as _qe2:
                    self.logger.warning(f"Auto re-collection retry failed: {_qe2}")
                if data is None or data.empty:
                    raise ValueError(f'Failed to read data from {data_path}')
            self.logger.info(f'✅ Loaded {len(data)} rows with {len(data.columns)} columns')

            # Validate no lookahead bias in loaded data
            try:
                if hasattr(data, 'index') and len(data) > 0:
                    data_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
                    if data_time:
                        bias_detector.set_current_timestamp(data_time)
                        data = validate_no_future_data(data, 'timestamp', data_time)
                        self.logger.info("✅ Lookahead bias validation passed")
            except LookaheadBiasError as e:
                self.logger.error(f"Lookahead bias detected: {e}")
                raise
            except Exception as e:
                self.logger.warning(f"Lookahead bias validation failed: {e}")

        except Exception as e:
            self.logger.error(f'❌ Failed to read data: {e}')
            raise
        try:
            if not isinstance(data.index, (pd.DatetimeIndex,)):
                self.logger.warning('Index is not DatetimeIndex; attempting auto-conversion from timestamp/date/time column')
                for ts_col in ['timestamp', 'date', 'time']:
                    if ts_col in data.columns:
                        data[ts_col] = pd.to_datetime(data[ts_col], errors='coerce')
                        data.set_index(ts_col, inplace = True)
                        break
        except Exception as e:
            self.logger.warning(f'⚠️ Could not normalize index: {e}')
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            self.logger.warning(f'⚠️ Missing OHLCV columns: {missing_cols}')
        validation_results = self._validate_data_quality(data)
        # Warmup zero logic for volume and key features
        try:
            warmup_rows = min(50, len(data))
            if 'volume' in data.columns:
                post_warmup_zeros = (data['volume'].iloc[warmup_rows:] == 0).sum()
                if post_warmup_zeros > 0:
                    self.logger.warning(f"⚠️ Found {post_warmup_zeros} zero volume rows beyond warmup ({warmup_rows})")
        except Exception as _zw:
            self.logger.debug(f"Zero warmup check skipped: {_zw}")
        pipeline_state['validated_data'] = data
        pipeline_state['data_validation_results'] = validation_results
        pipeline_state['data_info'] = {'shape': data.shape, 'columns': list(data.columns), 'index_type': str(type(data.index)), 'memory_usage_mb': data.memory_usage(deep = True).sum() / 1024 / 1024, 'date_range': {'start': str(data.index.min()) if hasattr(data.index, 'min') else None, 'end': str(data.index.max()) if hasattr(data.index, 'max') else None}}
        self._log_validation_summary(validation_results)
        pipeline_state['dataframe'] = data
        pipeline_state['validated_data'] = data
        pipeline_state['data_validation_results'] = validation_results
        return {'success': True, 'step02_data_reading_completed': True, 'validated_data': data, 'data_validation_results': validation_results, 'data_info': pipeline_state['data_info'], 'dataframe': data, 'step_name': 'step02_data_reading'}

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.

        Args:
            pipeline_state: Updated pipeline state

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'validated_data' not in pipeline_state and 'dataframe' not in pipeline_state:
            errors.append('No validated data in pipeline state')
            return (False, errors)
        if 'data_validation_results' not in pipeline_state:
            errors.append('No data validation results in pipeline state')
        else:
            validation_results = pipeline_state['data_validation_results']
            if not validation_results.get('has_required_columns', True):
                errors.append('Missing required columns')
            if validation_results.get('missing_data_pct', 100) > self.data_quality_thresholds['max_missing_pct'] * 100:
                errors.append(f"Too much missing data: {validation_results.get('missing_data_pct', 0):.2f}%")
            if validation_results.get('total_rows', 0) < self.data_quality_thresholds['min_rows']:
                errors.append(f"Insufficient data rows: {validation_results.get('total_rows', 0)}")
        return (len(errors) == 0, errors)
    @log_all_calls

    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality validation.

        Args:
            data: DataFrame to validate

        Returns:
            Validation results dictionary
        """
        results = {'total_rows': len(data), 'total_columns': len(data.columns), 'has_required_columns': True, 'missing_data_pct': 0, 'duplicate_rows': 0, 'data_quality_score': 100, 'issues': []}
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            results['has_required_columns'] = False
            results['issues'].append(f'Missing required columns: {missing_columns}')
            results['data_quality_score'] -= 20
        missing_count = data.isnull().sum().sum()
        total_cells = data.shape[0] * data.shape[1]
        if total_cells > 0:
            try:
                results['missing_data_pct'] = safe_divide(missing_count, total_cells, 0.0) * 100
                if results['missing_data_pct'] > 0:
                    results['issues'].append(f"Missing data: {results['missing_data_pct']:.2f}%")
                    results['data_quality_score'] -= min(20, results['missing_data_pct'] * 4)
            except MathValidationError as e:
                self.logger.warning(f"Mathematical validation error in missing data calculation: {e}")
                results['missing_data_pct'] = 0.0
        if hasattr(data.index, 'duplicated'):
            duplicate_count = data.index.duplicated().sum()
            if duplicate_count > 0:
                results['duplicate_rows'] = duplicate_count
                results['issues'].append(f'Duplicate timestamps: {duplicate_count}')
                results['data_quality_score'] -= 10
        if all((col in data.columns for col in ['high', 'low', 'open', 'close'])):
            invalid_high = (data['high'] < data[['open', 'close', 'low']].max(axis = 1)).sum()
            if invalid_high > 0:
                results['issues'].append(f'Invalid high values: {invalid_high} rows')
                results['data_quality_score'] -= 5
            invalid_low = (data['low'] > data[['open', 'close', 'high']].min(axis = 1)).sum()
            if invalid_low > 0:
                results['issues'].append(f'Invalid low values: {invalid_low} rows')
                results['data_quality_score'] -= 5
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                invalid_prices = (data[col] <= 0).sum()
                if invalid_prices > 0:
                    results['issues'].append(f'Invalid {col} prices: {invalid_prices} rows')
                    results['data_quality_score'] -= 5
        if 'volume' in data.columns:
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > 0:
                try:
                    zero_volume_pct = safe_divide(zero_volume, len(data), 0.0) * 100
                    if zero_volume_pct > 10:
                        results['issues'].append(f'Zero volume: {zero_volume} rows ({zero_volume_pct:.1f}%)')
                        results['data_quality_score'] -= min(10, zero_volume_pct)
                except MathValidationError as e:
                    self.logger.warning(f"Mathematical validation error in zero volume calculation: {e}")
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                if not data.index.is_monotonic_increasing:
                    results['issues'].append('Non-monotonic datetime index')
                    results['data_quality_score'] -= 5
        except Exception:
            pass
        results['data_quality_score'] = max(0, results['data_quality_score'])
        return results
    @log_all_calls

    def _log_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """Log a summary of validation results.

        Args:
            validation_results: Validation results dictionary
        """
        self.logger.info('📊 Data Validation Summary:')
        self.logger.info(f"   - Total rows: {validation_results['total_rows']:,}")
        self.logger.info(f"   - Total columns: {validation_results['total_columns']}")
        self.logger.info(f"   - Missing data: {validation_results['missing_data_pct']:.2f}%")
        self.logger.info(f"   - Duplicate rows: {validation_results['duplicate_rows']}")
        self.logger.info(f"   - Quality score: {validation_results['data_quality_score']}/100")
        if validation_results['issues']:
            self.logger.warning('⚠️ Data quality issues found:')
            for issue in validation_results['issues']:
                self.logger.warning(f'   - {issue}')
        else:
            self.logger.info('✅ No data quality issues found')

    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ['unified_data_path or raw_market_data']

    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ['validated_data', 'data_validation_results', 'data_info', 'dataframe']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ['01_data_collection', '01_5_data_converter']
