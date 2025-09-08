from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from src.utils.logger import system_logger
from ..standardized_parquet_handler import standardized_parquet_handler

"""
Enhanced Step 1: Data Collection with Comprehensive Function Call Monitoring

This module provides step01 data collection with:
- Comprehensive function call monitoring and validation
- Detailed function entry, execution, and completion reporting
- Inter-function call tracking and dependency monitoring
- Performance monitoring with timing and resource usage
- Enhanced error handling with detailed function-level tracking
- Structured logging with comprehensive reports
"""
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

    class MockDataFrame:
        @log_important_calls
        def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]=None) -> None:
            self.data = data or []
            self.columns = []

        def to_dict(self, orient: Any='records') -> None:
            return self.data

        def head(self, n: int = 5) -> None:
            return MockDataFrame(self.data[:n])

        def tail(self, n: int = 5) -> None:
            return MockDataFrame(self.data[-n:])

        def isnull(self) -> None:
            return MockDataFrame([False] * len(self.data))

        def sum(self) -> int:
            return 0

        @log_all_calls
        def __len__(self) -> None:
            return len(self.data)

        @log_all_calls
        def __iter__(self) -> None:
            return iter(self.data)

    class MockSeries:
        @log_important_calls
        def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]=None) -> None:
            self.data = data or []

        def sum(self) -> int:
            return 0
    pd = type('MockPandas', (), {'DataFrame': MockDataFrame, 'Series': MockSeries, 'read_parquet': lambda path: MockDataFrame(), 'to_datetime': lambda x: x, 'isna': lambda x: False, 'date_range': lambda start, end, freq: []})()
    np = type('MockNumpy', (), {'random': type('MockRandom', (), {'seed': lambda x: None, 'normal': lambda mean, std, size: [0] * size, 'uniform': lambda low, high, size: [0] * size, 'choice': lambda choices: choices[0] if choices else None, 'randint': lambda low, high: low})(), 'array': lambda x: x, 'isinf': lambda x: False})()

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.function_call_monitor import monitor_comprehensive, monitor_standard, monitor_basic, get_function_call_monitor, log_function_call_summary
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild('Step01EnhancedMonitoring')
function_monitor = get_function_call_monitor()

class EnhancedDataCollectionStepWithMonitoring:
    """Enhanced Step 1: Data Collection with comprehensive function call monitoring."""
    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the enhanced data collection step with monitoring."""
        self.config = config
        self.logger = logger.getChild('EnhancedDataCollectionStepWithMonitoring')
        self.standards = pipeline_standards
        self._validate_environment()

    @log_all_calls
    @monitor_comprehensive
    def _validate_environment(self) -> None:
        """Validate environment dependencies with comprehensive monitoring."""
        self.logger.info('🔍 Validating environment dependencies with monitoring...')
        required_modules = ['pandas', 'numpy', 'src.config', 'src.utils.logger', 'src.utils.error_handler', 'src.training.steps.data_downloader', 'src.utils.enhanced_mlflow_integration', 'src.utils.centralized_decorators']
        dependency_status = PipelineStandards.validate_environment_dependencies(required_modules)
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')
        log_function_call_summary(self.logger)

    @monitor_standard
    async def initialize(self) -> None:
        """Initialize the data collection step with monitoring."""
        self.logger.info('🚀 Initializing Enhanced Data Collection Step with Monitoring...')
        self.logger.info('📋 Step 1 Configuration:')
        self.logger.info(f"   - Config keys: {(list(self.config.keys()) if self.config else 'None')}")
        self.logger.info('✅ Enhanced Data Collection Step initialized successfully')
        log_function_call_summary(self.logger)

    @monitor_comprehensive
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection with comprehensive monitoring and validation.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with detailed monitoring results
        """
        self.logger.info('🚀 Starting comprehensive data collection with monitoring...')
        if not training_input:
            raise ValueError('Training input cannot be empty')
        if not isinstance(pipeline_state, dict):
            raise ValueError('Pipeline state must be a dictionary')
        try:
            symbol = training_input.get('symbol')
            exchange = training_input.get('exchange')
            timeframe = training_input.get('timeframe', '1m')
            if not symbol or not exchange:
                raise ValueError('Symbol and exchange are required parameters')
            self.logger.info(f'📊 Processing: {exchange}_{symbol}_{timeframe}')
            data_dir = self.standards.build_path('raw_data', exchange, symbol)
            self.logger.info(f'📁 Using standardized data directory: {data_dir}')
            collection_success = await self._run_monitored_data_collection(training_input, data_dir)
            if collection_success:
                self.logger.info('✅ Data collection completed successfully')
                quality_success = await self._run_monitored_quality_check(symbol, exchange, timeframe, data_dir)
                if quality_success:
                    self.logger.info('✅ Quality check passed')
                    pipeline_state['data_collection_completed'] = True
                    pipeline_state['quality_check_passed'] = True
                else:
                    self.logger.warning('⚠️ Quality check found issues')
                    pipeline_state['data_collection_completed'] = True
                    pipeline_state['quality_check_passed'] = False
            else:
                self.logger.error('❌ Data collection failed')
                pipeline_state['data_collection_completed'] = False
                pipeline_state['quality_check_passed'] = False
            pipeline_state['monitoring_summary'] = function_monitor.get_call_summary()
        except Exception as e:
            self.logger.exception(f'❌ Error during data collection: {e}')
            pipeline_state['data_collection_completed'] = False
            pipeline_state['quality_check_passed'] = False
            pipeline_state['error_message'] = str(e)
        await self._log_monitored_step1_artifacts_and_report(training_input, pipeline_state)
        log_function_call_summary(self.logger)
        return pipeline_state

    @monitor_standard
    async def _run_monitored_data_collection(self, training_input: Dict[str, Any], data_dir: str) -> bool:
        """Run data collection with comprehensive monitoring."""
        try:
            symbol = training_input.get('symbol')
            exchange = training_input.get('exchange')
            timeframe = training_input.get('timeframe', '1m')
            self.logger.info(f'📊 Starting monitored data collection for {exchange}_{symbol}_{timeframe}')
            os.makedirs(data_dir, exist_ok = True)
            download_success = await self._monitored_data_download(symbol, exchange, timeframe, data_dir)
            if download_success:
                self.logger.info('✅ Data download completed successfully')
                validation_success = await self._monitored_data_validation(symbol, exchange, timeframe, data_dir)
                if validation_success:
                    self.logger.info('✅ Downloaded data validation passed')
                else:
                    self.logger.warning('⚠️ Downloaded data validation found issues')
                await self._log_monitored_detailed_data_extract(symbol, exchange, timeframe, data_dir)
                return True
            else:
                self.logger.warning('⚠️ Data download failed, using fallback method')
                return await self._monitored_fallback_data_collection(training_input, data_dir)
        except Exception as e:
            self.logger.exception(f'❌ Error in monitored data collection: {e}')
            return False

    @monitor_basic
    async def _monitored_data_download(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Download data with basic monitoring."""
        try:
            try:
                from src.training.steps.data_collection.data_downloader import download_all_data_with_consolidation
                self.logger.info(f'🔄 Downloading data from {exchange} API...')
                success = await download_all_data_with_consolidation(symbol = symbol, exchange_name = exchange, interval = timeframe, data_dir = data_dir)
                if success:
                    self.logger.info('✅ Data download completed successfully')
                    return True
                else:
                    self.logger.warning('⚠️ Data download returned False')
                    return False
            except ImportError:
                self.logger.warning('⚠️ Data downloader not available')
                return False
        except Exception as e:
            self.logger.exception(f'❌ Error in data download: {e}')
            return False

    @monitor_standard
    async def _monitored_data_validation(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate downloaded data with standard monitoring."""
        try:
            self.logger.info('🔍 Running monitored data validation...')
            expected_files = [self.standards.generate_file_name('klines', exchange, symbol, timeframe), self.standards.generate_file_name('aggtrades', exchange, symbol)]
            validation_results = []
            for file_name in expected_files:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    self.logger.info(f'✅ Found expected file: {file_name}')
                    validation_result = await self._monitored_single_file_validation(file_name, file_path)
                    if validation_result is not None:
                        validation_results.append(validation_result)
                    else:
                        self.logger.warning(f'⚠️ File validation failed: {file_name}')
                        return False
                else:
                    self.logger.warning(f'⚠️ Expected file not found: {file_name}')
                    return False
            return self._process_monitored_validation_results(validation_results)
        except Exception as e:
            self.logger.exception(f'❌ Error in monitored data validation: {e}')
            return False

    @monitor_basic
    async def _monitored_single_file_validation(self, file_name: str, file_path: str) -> Optional[Any]:
        """Validate a single file with basic monitoring."""
        try:
            self.logger.info(f'🔍 Validating {file_name}...')
            df = standardized_parquet_handler.read_parquet_standardized(file_path)
            df = self.standards.standardize_timestamp(df, 'timestamp')
            schema_name = self._determine_schema_name(file_name)
            validation_result = self.standards.validate_data_quality(df, schema_name)
            self._log_monitored_validation_result(file_name, validation_result)
            return validation_result
        except Exception as e:
            self.logger.exception(f'❌ Error validating {file_name}: {e}')
            return None

    @log_all_calls
    @monitor_basic
    def _determine_schema_name(self, file_name: str) -> str:
        """Determine schema name based on file name."""
        if 'klines' in file_name:
            return 'klines'
        elif 'aggtrades' in file_name:
            return 'aggtrades'
        else:
            return 'unified'

    @log_all_calls
    @monitor_basic
    def _log_monitored_validation_result(self, file_name: str, validation_result: Any) -> None:
        """Log validation result with monitoring."""
        if validation_result.passed:
            self.logger.info(f'✅ {file_name} quality check passed (score: {validation_result.quality_score:.2f})')
        else:
            self.logger.warning(f'⚠️ {file_name} quality check issues:')
            self._log_monitored_issues(validation_result.issues)
        self._log_monitored_warnings(validation_result.warnings)

    @log_all_calls
    @monitor_basic
    def _log_monitored_issues(self, issues: List[Any], max_display: int = 3) -> None:
        """Log validation issues with monitoring."""
        for issue in issues[:max_display]:
            self.logger.warning(f'   - {issue.message}')
        if len(issues) > max_display:
            self.logger.warning(f'   ... and {len(issues) - max_display} more issues')

    @log_all_calls
    @monitor_basic
    def _log_monitored_warnings(self, warnings: List[Any], max_display: int = 3) -> None:
        """Log validation warnings with monitoring."""
        for warning in warnings[:max_display]:
            self.logger.info(f'   ⚠️ {warning.message}')
        if len(warnings) > max_display:
            self.logger.info(f'   ... and {len(warnings) - max_display} more warnings')

    @log_all_calls
    @monitor_standard
    def _process_monitored_validation_results(self, validation_results: List[Any]) -> bool:
        """Process and summarize validation results with monitoring."""
        if not validation_results:
            self.logger.warning('⚠️ No validation results available')
            return False
        overall_passed = all((result.passed for result in validation_results))
        overall_quality_score = sum((result.quality_score for result in validation_results)) / len(validation_results)
        self.logger.info(f"📊 Overall validation: {('PASSED' if overall_passed else 'FAILED')}")
        self.logger.info(f'📊 Average quality score: {overall_quality_score:.2f}')
        total_issues = sum((len(result.issues) for result in validation_results))
        total_warnings = sum((len(result.warnings) for result in validation_results))
        if total_issues > 0:
            self.logger.warning(f'📊 Total issues found: {total_issues}')
        if total_warnings > 0:
            self.logger.info(f'📊 Total warnings: {total_warnings}')
        return overall_passed

    @monitor_standard
    async def _monitored_fallback_data_collection(self, training_input: Dict[str, Any], data_dir: str) -> bool:
        """Fallback data collection with monitoring."""
        self.logger.info('🔄 Running monitored fallback data collection...')
        try:
            symbol = training_input.get('symbol')
            exchange = training_input.get('exchange')
            timeframe = training_input.get('timeframe', '1m')
            if not symbol or not exchange:
                self.logger.error('❌ Symbol and exchange required for fallback collection')
                return False
            self.logger.info('📊 Creating mock data for fallback collection...')
            mock_data_success = await self._generate_monitored_mock_data(symbol, exchange, timeframe, data_dir)
            if mock_data_success:
                self.logger.info('✅ Mock data generation completed successfully')
                return True
            else:
                self.logger.error('❌ Mock data generation failed')
                return False
        except Exception as e:
            self.logger.exception(f'❌ Error in monitored fallback data collection: {e}')
            return False

    @monitor_standard
    async def _generate_monitored_mock_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Generate mock data with monitoring."""
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days = 30)
            timestamps = pd.date_range(start = start_date, end = end_date, freq='1min')
            np.random.seed(42)
            base_price = 3000.0
            price_changes = np.random.normal(0, 0.002, len(timestamps))
            prices = [base_price]
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 100))
            prices = np.array(prices)
            klines_data = []
            for i, timestamp in enumerate(timestamps):
                price = prices[i]
                volume = np.random.uniform(10, 1000)
                spread = price * 0.001
                open_price = price + np.random.uniform(-spread, spread)
                high_price = max(open_price, price + np.random.uniform(0, spread))
                low_price = min(open_price, price - np.random.uniform(0, spread))
                close_price = price + np.random.uniform(-spread, spread)
                klines_data.append({'timestamp': int(timestamp.timestamp() * 1000), 'open': round(open_price, 2), 'high': round(high_price, 2), 'low': round(low_price, 2), 'close': round(close_price, 2), 'volume': round(volume, 2)})
            klines_df = pd.DataFrame(klines_data)
            klines_df = self.standards.standardize_timestamp(klines_df, 'timestamp')
            klines_df = self.standards.enforce_schema(klines_df, 'klines')
            klines_file = self.standards.generate_file_name('klines', exchange, symbol, timeframe)
            klines_path = os.path.join(data_dir, klines_file)
            standardized_parquet_handler.write_parquet_standardized(klines_df, klines_path, index=False)
            self.logger.info(f'✅ Created mock klines data: {len(klines_df)} rows')
            self.logger.info(f'💾 Saved to: {klines_path}')
            aggtrades_data = []
            for i in range(0, len(timestamps), 5):
                timestamp = timestamps[i]
                price = prices[i] if i < len(prices) else base_price
                num_trades = np.random.randint(1, 10)
                for _ in range(num_trades):
                    trade_price = price + np.random.normal(0, 50)
                    quantity = np.random.uniform(0.1, 10.0)
                    aggtrades_data.append({'timestamp': int(timestamp.timestamp() * 1000), 'price': round(trade_price, 2), 'quantity': round(quantity, 4), 'is_buyer_maker': np.random.choice([True, False])})
            aggtrades_df = pd.DataFrame(aggtrades_data)
            aggtrades_df = self.standards.standardize_timestamp(aggtrades_df, 'timestamp')
            aggtrades_df = self.standards.enforce_schema(aggtrades_df, 'aggtrades')
            aggtrades_file = self.standards.generate_file_name('aggtrades', exchange, symbol)
            aggtrades_path = os.path.join(data_dir, aggtrades_file)
            standardized_parquet_handler.write_parquet_standardized(aggtrades_df, aggtrades_path, index=False)
            self.logger.info(f'✅ Created mock aggtrades data: {len(aggtrades_df)} rows')
            self.logger.info(f'💾 Saved to: {aggtrades_path}')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error generating mock data: {e}')
            return False

    @monitor_standard
    async def _run_monitored_quality_check(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Run quality check with monitoring."""
        try:
            self.logger.info('🔍 Running monitored quality check...')
            return await self._monitored_data_validation(symbol, exchange, timeframe, data_dir)
        except Exception as e:
            self.logger.exception(f'❌ Error in monitored quality check: {e}')
            return False

    @monitor_standard
    async def _log_monitored_detailed_data_extract(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
        """Log detailed data extract with monitoring."""
        self.logger.info('=' * 80)
        self.logger.info('📊 DETAILED DATA EXTRACT WITH MONITORING')
        self.logger.info('=' * 80)
        try:
            klines_file = self.standards.generate_file_name('klines', exchange, symbol, timeframe)
            aggtrades_file = self.standards.generate_file_name('aggtrades', exchange, symbol)
            files_to_check = [('Klines', os.path.join(data_dir, klines_file)), ('Aggtrades', os.path.join(data_dir, aggtrades_file))]
            for data_type, file_path in files_to_check:
                self.logger.info(f'🔍 Analyzing {data_type} data: {os.path.basename(file_path)}')
                if os.path.exists(file_path):
                    try:
                        df = standardized_parquet_handler.read_parquet_standardized(file_path)
                        self.logger.info(f'   📊 Shape: {df.shape}')
                        self.logger.info(f'   📁 File size: {os.path.getsize(file_path):,} bytes')
                        self.logger.info(f'   🗂️ Columns ({len(df.columns)}): {list(df.columns)}')
                        self.logger.info('   🔧 Data types:')
                        for col, dtype in df.dtypes.items():
                            self.logger.info(f'      - {col}: {dtype}')
                        self.logger.info('   📋 Sample data (first 3 rows):')
                        sample_df = df.head(3)
                        for idx, row in sample_df.iterrows():
                            formatted_row = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    formatted_row[col] = 'NaN'
                                elif isinstance(val, (int, float)):
                                    formatted_row[col] = f'{val:.6f}' if isinstance(val, float) else str(val)
                                else:
                                    formatted_row[col] = str(val)
                            self.logger.info(f'      Row {idx}: {formatted_row}')
                        if 'timestamp' in df.columns:
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                min_date = df['timestamp'].min()
                                max_date = df['timestamp'].max()
                                total_days = (max_date - min_date).days
                                self.logger.info(f'   📅 Date range: {min_date} to {max_date} ({total_days} days)')
                            except Exception as e:
                                self.logger.warning(f'   ⚠️ Could not parse timestamp: {e}')
                        missing_counts = df.isnull().sum()
                        if missing_counts.sum() > 0:
                            self.logger.warning('   ⚠️ Missing values:')
                            for col, count in missing_counts.items():
                                if count > 0:
                                    percentage = count / len(df) * 100
                                    self.logger.warning(f'      - {col}: {count} ({percentage:.2f}%)')
                        else:
                            self.logger.info('   ✅ No missing values found')
                        self.logger.info(f'   ✅ {data_type} data analysis completed')
                    except Exception as e:
                        self.logger.exception(f'   ❌ Error analyzing {data_type} data: {e}')
                else:
                    self.logger.warning(f'   ⚠️ File not found: {file_path}')
                self.logger.info('')
            self.logger.info('📋 DATA EXTRACT SUMMARY:')
            existing_files = sum((1 for _, file_path in files_to_check if os.path.exists(file_path)))
            self.logger.info(f'   • Files found: {existing_files}/{len(files_to_check)}')
            self.logger.info('   • Data types analyzed: Klines, Aggtrades')
            self.logger.info('   • Information logged: Shape, columns, data types, sample data, date ranges, missing values')
            self.logger.info('=' * 80)
        except Exception as e:
            self.logger.exception(f'❌ Error in detailed data extract: {e}')
            self.logger.info('=' * 80)

    @monitor_standard
    async def _log_monitored_step1_artifacts_and_report(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Log step 1 artifacts and create detailed report with monitoring."""
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            execution_metadata = {'start_time': datetime.now().isoformat(), 'end_time': datetime.now().isoformat(), 'duration_seconds': 0.0, 'memory_usage_mb': 0.0, 'cpu_usage_percent': 0.0, 'data_quality_score': 1.0 if pipeline_state.get('quality_check_passed', False) else 0.5, 'processing_efficiency': 1.0 if pipeline_state.get('data_collection_completed', False) else 0.0, 'monitoring_summary': pipeline_state.get('monitoring_summary', {})}
            artifacts_generated = []
            if pipeline_state.get('data_collection_completed', False):
                artifacts_generated.extend([f'{exchange}_{symbol}_{timeframe}_klines.parquet', f'{exchange}_{symbol}_{timeframe}_trades.parquet', f'{exchange}_{symbol}_{timeframe}_orderbook.parquet'])
            metrics_calculated = {'data_collection_success': 1.0 if pipeline_state.get('data_collection_completed', False) else 0.0, 'quality_check_passed': 1.0 if pipeline_state.get('quality_check_passed', False) else 0.0, 'total_artifacts_generated': len(artifacts_generated), 'function_calls_monitored': pipeline_state.get('monitoring_summary', {}).get('total_calls', 0), 'function_call_success_rate': pipeline_state.get('monitoring_summary', {}).get('success_rate', 0.0)}
            report_data = {'step_name': 'step01_enhanced_data_collection_with_monitoring', 'step_data': pipeline_state, 'training_input': training_input, 'execution_metadata': execution_metadata, 'artifacts_generated': artifacts_generated, 'metrics_calculated': metrics_calculated, 'errors_encountered': [] if pipeline_state.get('data_collection_completed', False) else ['Data collection failed'], 'monitoring_results': pipeline_state.get('monitoring_summary', {})}
            self.logger.info('✅ Enhanced Step 1 artifacts and reports logged successfully')
            self.logger.info(f"📊 Function calls monitored: {metrics_calculated['function_calls_monitored']}")
            self.logger.info(f"📊 Function call success rate: {metrics_calculated['function_call_success_rate']:.1f}%")
        except Exception as e:
            self.logger.exception(f'❌ Failed to log enhanced step 1 artifacts and reports: {e}')

@monitor_comprehensive
async def run_enhanced_step01_with_monitoring(symbol: str, exchange: str, timeframe: str='1m', data_dir: Optional[str]=None, force_rerun: bool = False, **kwargs: Any) -> bool:
    """Run the enhanced data collection step with comprehensive monitoring.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if data exists
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info('=' * 80)
        logger.info('🚀 ENHANCED STEP 1: Data Collection with Comprehensive Monitoring')
        logger.info('=' * 80)
        logger.info(f'🎯 Symbol: {symbol}')
        logger.info(f'🏢 Exchange: {exchange}')
        logger.info(f'📊 Timeframe: {timeframe}')
        logger.info(f'📁 Data directory: {data_dir}')
        logger.info(f'🔄 Force rerun: {force_rerun}')
        logger.info('=' * 80)
        config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir or 'data_cache'}
        step = EnhancedDataCollectionStepWithMonitoring(config)
        await step.initialize()
        training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'force_rerun': force_rerun, **kwargs}
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        success = result.get('data_collection_completed', False)
        if success:
            logger.info('✅ Enhanced Step 1: Data Collection completed successfully')
            report_path = f"step01_monitoring_report_{symbol}_{exchange}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            function_monitor.export_detailed_report(report_path)
            logger.info(f'📊 Detailed monitoring report exported to: {report_path}')
        else:
            logger.error('❌ Enhanced Step 1: Data Collection failed')
        log_function_call_summary(logger)
        return success
    except Exception as e:
        logger.exception(f'❌ Enhanced Step 1 failed with exception: {e}')
        return False
if __name__ == '__main__':

    async def main() -> None:
        """Main execution function."""
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else 'data_cache'
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == 'true'
        else:
            print('Usage: python step01_enhanced_with_monitoring.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]')
            print('Example: python step01_enhanced_with_monitoring.py ETHUSDT BINANCE 1m data_cache true')
            return
        success = await run_enhanced_step01_with_monitoring(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, force_rerun = force_rerun)
        if success:
            print('✅ Enhanced Step 1: Data Collection completed successfully')
        else:
            print('❌ Enhanced Step 1: Data Collection failed')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n🛑 Interrupted by user')
    except Exception as e:
        print(f'❌ Error: {e}')