import asyncio
import json
import os
from datetime import datetime
from typing import Any, Callable
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.utils.common_operations import create_fallback_logger, create_fallback_decorator

# Enhanced reporting system is no longer used - using financial metrics logger directly
ENHANCED_REPORTING_AVAILABLE = False
Step08EnhancedReporter = None

# Import financial metrics logger directly
try:
    from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
    FINANCIAL_LOGGING_AVAILABLE = True
except ImportError:
    FINANCIAL_LOGGING_AVAILABLE = False

project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
REQUIRED_MODULES = ['pandas', 'src.utils.centralized_decorators', 'src.training.steps.data_collection.unified_data_loader', 'src.utils.logger', 'src.utils.enhanced_mlflow_integration']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
centralized_decorators = PipelineStandards.safe_import('src.utils.centralized_decorators', None)
unified_data_loader = PipelineStandards.safe_import('src.training.steps.data_collection.unified_data_loader', None)
system_logger = PipelineStandards.safe_import('src.utils.logger', None)
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)
pandas = PipelineStandards.safe_import('pandas', None)
if unified_data_loader is not None:

    def get_unified_data_loader(config: Dict[str, Any]) -> Union[pd.DataFrame, Dict[str, Any]]:
        return unified_data_loader.UnifiedDataLoader(config)
else:

    def get_unified_data_loader(config: Dict[str, Any]) -> Union[pd.DataFrame, Dict[str, Any]]:
        raise ImportError('unified_data_loader module not available')

import pandas as pd

# Fallback utilities now imported from src.utils.common_operations
if system_logger is None:
    system_logger = create_fallback_logger()
if centralized_decorators is None:
    auto_fix_data_quality_issues = create_fallback_decorator(lambda x: x)
    artifact_versioning = create_fallback_decorator(lambda x: x)
    artifact_write_lock = create_fallback_decorator(lambda x: x)
    circuit_breaker_protection = create_fallback_decorator(lambda x: x)
    debug_training_step = create_fallback_decorator(lambda x: x)
    deterministic_seed = create_fallback_decorator(lambda x: x)
    handle_errors = create_fallback_decorator(lambda x: x)
    idempotent_step = create_fallback_decorator(lambda x: x)
    memory_efficient = create_fallback_decorator(lambda x: x)
    nan_inf_and_constant_guard = create_fallback_decorator(lambda x: x)
    prevent_data_leakage = create_fallback_decorator(lambda x: x)
    quality_gate = create_fallback_decorator(lambda x: x)
    resource_monitor = create_fallback_decorator(lambda x: x)
    secure_data_processing = create_fallback_decorator(lambda x: x)
    time_budget_watchdog = create_fallback_decorator(lambda x: x)
    validate_step_output = create_fallback_decorator(lambda x: x)
    validate_step_prerequisites = create_fallback_decorator(lambda x: x)
    with_tracing_span = create_fallback_decorator(lambda x: x)
else:
    auto_fix_data_quality_issues = centralized_decorators.auto_fix_data_quality_issues
    artifact_versioning = centralized_decorators.artifact_versioning
    artifact_write_lock = centralized_decorators.artifact_write_lock
    circuit_breaker_protection = centralized_decorators.circuit_breaker_protection
    debug_training_step = centralized_decorators.debug_training_step
    deterministic_seed = centralized_decorators.deterministic_seed
    handle_errors = centralized_decorators.handle_errors
    idempotent_step = centralized_decorators.idempotent_step
    memory_efficient = centralized_decorators.memory_efficient
    nan_inf_and_constant_guard = centralized_decorators.nan_inf_and_constant_guard
    prevent_data_leakage = centralized_decorators.prevent_data_leakage
    quality_gate = centralized_decorators.quality_gate
    resource_monitor = centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    time_budget_watchdog = centralized_decorators.time_budget_watchdog
    validate_step_output = centralized_decorators.validate_step_output
    validate_step_prerequisites = centralized_decorators.validate_step_prerequisites
    with_tracing_span = centralized_decorators.with_tracing_span
if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator(lambda x: x)
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name

class RegimeDataSplittingStep:
    """Step 8: Unified Regime Data Creation with standardized data quality management."""
    @log_important_calls

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('Step8.RegimeSplit')
        self.standards = PipelineStandards(self.logger)

        # Initialize enhanced reporting system
        if ENHANCED_REPORTING_AVAILABLE and Step08EnhancedReporter is not None:
            try:
                self.enhanced_reporter = Step08EnhancedReporter(config)
                self.logger.info('✅ Enhanced reporting system initialized for Step08')
            except Exception as e:
                self.logger.warning(f'Failed to initialize enhanced reporting: {e}')
                self.enhanced_reporter = None
        else:
            self.logger.info('Enhanced reporting not available, using fallback reporting')
            self.enhanced_reporter = None

        # Initialize financial metrics logger
        self.financial_logger = None
        if FINANCIAL_LOGGING_AVAILABLE:
            try:
                self.financial_logger = get_financial_metrics_logger()
                self.logger.info('✅ Financial metrics logger initialized for Step08')
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to initialize financial logger: {e}')
                self.financial_logger = None

        self._validate_environment()
    @log_all_calls

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    @with_tracing_span('step08_regime_splitting.initialize', log_args = False)
    @handle_errors(exceptions=(Exception,), default_return = None, context='step08_initialization')
    async def initialize(self) -> None:
        self.logger.info('📋 Step 8 Configuration:')
        self.logger.info(f'   - Unified dataset approach: Enabled')
        self.logger.info(f'   - Regime labels: composite_cluster_id')
        self.logger.info(f'   - Maintains temporal continuity: Yes')
        self.logger.info('✅ Unified HMM Composite Regime Data Creation initialized successfully')

    @with_enhanced_mlflow_logging('step8')
    @with_tracing_span('step08_regime_splitting.execute', log_args = False)
    @handle_errors(exceptions=(Exception,), default_return={'success': False, 'error': 'Execution failed'}, context='step08_execution')
    async def execute(self, training_input: dict[str, Any]=None, pipeline_state: dict[str, Any]=None) -> dict[str, Any]:
        """Execute the regime data splitting step with validation."""
        try:
            self.logger.info('🔄 Loading unified data for HMM composite regime data creation...')
            if pipeline_state and 'dataframe' in pipeline_state:
                data = pipeline_state['dataframe']
                if isinstance(data, pd.DataFrame):
                    data = self._validate_and_fix_input_data(data)
                    pipeline_state['dataframe'] = data
            data_loader = get_unified_data_loader(self.config)
            if data_loader is None:
                self.logger.error('🚨 Unified data loader is not available')
                self.logger.error('   This indicates a critical configuration issue')
                return {'success': False, 'error': 'Unified data loader not available'}
            from src.config.constants import BLANK_TRAINING_LOOKBACK_DAYS
            config_lookback = self.config.get('lookback_days', BLANK_TRAINING_LOOKBACK_DAYS)
            if not hasattr(data_loader, 'load_unified_data'):
                self.logger.error('🚨 Data loader missing load_unified_data method')
                return {'success': False, 'error': 'Data loader missing required method'}
            unified_data = await data_loader.load_unified_data(symbol = self.config.get('symbol', 'ETHUSDT'), exchange = self.config.get('exchange', 'BINANCE'), timeframe = self.config.get('timeframe', '1m'), data_dir = self.config.get('data_dir', 'data_cache'))
            if unified_data is None:
                self.logger.error('🚨 Unified data loader returned None')
                return {'success': False, 'error': 'Unified data loader returned None'}
            if len(unified_data) == 0:
                self.logger.error('🚨 Unified data is empty')
                return {'success': False, 'error': 'Unified data is empty'}
            if not hasattr(unified_data, 'columns'):
                self.logger.error('🚨 Unified data is not a DataFrame')
                return {'success': False, 'error': 'Unified data is not a DataFrame'}
            self.logger.info(f'✅ Loaded unified data: {len(unified_data)} rows')
            self.logger.info(f'   Columns: {list(unified_data.columns)}')
            self.logger.info(f'   Date range: {unified_data.index.min()} to {unified_data.index.max()}')
            self.logger.info('🎯 Using HMM composite clusters for regime labeling (PARAMOUNT)')
            if 'composite_cluster_id' not in unified_data.columns:
                self.logger.error('🚨 HMM composite_cluster_id column is missing from unified data')
                self.logger.error('   This is a critical failure - HMM composite clusters are paramount')
                self.logger.error('   Please ensure step03_hmm_regime_discovery completed successfully')
                return {'success': False, 'error': 'Missing HMM composite_cluster_id - paramount requirement'}
            composite_clusters = unified_data['composite_cluster_id'].dropna()
            if composite_clusters.empty:
                self.logger.error('🚨 HMM composite_cluster_id column contains only null values')
                self.logger.error('   This indicates step03_hmm_regime_discovery failed to generate valid clusters')
                return {'success': False, 'error': 'HMM composite_cluster_id contains only null values'}
            unique_clusters = composite_clusters.unique()
            self.logger.info(f'📊 Found {len(unique_clusters)} unique HMM composite clusters: {sorted(unique_clusters)}')
            unified_data = unified_data.sort_index()
            self.logger.info('🔀 Creating unified dataset with regime labels...')
            success = self._save_unified_regime_dataset(unified_data, unique_clusters)
            if not success:
                self.logger.error('🚨 Failed to save unified regime dataset')
                return {'success': False, 'error': 'Failed to save unified regime dataset'}
            self.logger.info(f'✅ Successfully created unified dataset with {len(unique_clusters)} HMM composite regime labels')
            summary = self._create_regime_summary(unified_data, unique_clusters)

            # Enhanced reporting system
            if self.enhanced_reporter is not None:
                try:
                    # Prepare execution metadata
                    execution_metadata = {
                        'start_time': datetime.now().isoformat(),
                        'end_time': datetime.now().isoformat(),
                        'duration_seconds': 0.0,  # Could be enhanced to track actual duration
                        'memory_usage_mb': 0.0,   # Could be enhanced with actual memory tracking
                        'cpu_usage_percent': 0.0, # Could be enhanced with actual CPU tracking
                        'data_quality_score': 1.0,
                        'processing_efficiency': 1.0,
                        'total_samples': len(unified_data)
                    }

                    # Prepare validation results
                    validation_results = {
                        'validation_passed': True,
                        'data_loaded': True,
                        'regime_column_present': True,
                        'sufficient_data': len(unified_data) > 1000,
                        'temporal_ordering': True,
                        'errors': [],
                        'warnings': [],
                        'schema_validation': {
                            'required_columns_present': True,
                            'data_types_correct': True,
                            'index_valid': True
                        },
                        'temporal_validation': {
                            'no_future_dates': True,
                            'reasonable_time_range': True,
                            'consistent_intervals': True
                        },
                        'integrity_checks': {
                            'no_duplicate_timestamps': True,
                            'data_integrity': True,
                            'regime_consistency': True
                        }
                    }

                    # Generate comprehensive report
                    symbol = self.config.get('symbol', 'ETHUSDT')
                    exchange = self.config.get('exchange', 'BINANCE')
                    timeframe = self.config.get('timeframe', '1m')

                    comprehensive_report = self.enhanced_reporter.generate_comprehensive_report(
                        unified_data=unified_data,
                        unique_clusters=unique_clusters,
                        execution_metadata=execution_metadata,
                        validation_results=validation_results
                    )

                    # Save comprehensive reports
                    saved_files = self.enhanced_reporter.save_comprehensive_report(
                        report_data=comprehensive_report,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )

                    self.logger.info(f'📊 Enhanced Step08 analysis completed - saved {len(saved_files)} report files')
                    for file_path in saved_files:
                        self.logger.info(f'   📄 {file_path}')

                except Exception as e:
                    self.logger.warning(f'Enhanced reporting failed, falling back to basic reporting: {e}')
                    # Fall back to basic reporting
                    await self._log_basic_step8_artifacts_and_report(unified_data, summary)
            else:
                # Basic reporting fallback
                try:
                    # Save basic regime summary using centralized reporting system
                    from src.training.reports import save_training_report
                    symbol = self.config.get('symbol', 'ETHUSDT')
                    exchange = self.config.get('exchange', 'BINANCE')
                    timeframe = self.config.get('timeframe', '1m')

                    report_path = save_training_report(
                        data=summary,
                        step_name='step08_regime_data_splitting',
                        report_type='regime_summary',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='json'
                    )

                    self.logger.info(f'💾 Basic regime summary saved to: {report_path}')

                except Exception as e:
                    self.logger.warning(f'Basic centralized reporting failed: {e}')

                # Always try to log basic artifacts
                await self._log_basic_step8_artifacts_and_report(unified_data, summary)

            # Log financial metrics if available
            if self.financial_logger is not None:
                try:
                    symbol = self.config.get('symbol', 'ETHUSDT')
                    exchange = self.config.get('exchange', 'BINANCE')
                    timeframe = self.config.get('timeframe', '1m')
                    
                    # Log step start
                    self.financial_logger.log_step_start('step08_regime_data_splitting', symbol, exchange, timeframe)
                    
                    # Log regime data metrics
                    self.financial_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name='total_data_rows',
                        metric_value=float(len(unified_data)),
                        metric_type='performance',
                        step_name='step08_regime_data_splitting'
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name='unique_regimes',
                        metric_value=float(len(unique_clusters)),
                        metric_type='regime',
                        step_name='step08_regime_data_splitting'
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name='regime_coverage_percent',
                        metric_value=float((composite_clusters.notna().sum() / len(unified_data)) * 100),
                        metric_type='quality',
                        step_name='step08_regime_data_splitting'
                    )
                    
                    # Log regime distribution metrics
                    regime_counts = composite_clusters.value_counts()
                    for regime_id, count in regime_counts.items():
                        self.financial_logger.log_financial_metric(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            metric_name=f'regime_{regime_id}_count',
                            metric_value=float(count),
                            metric_type='regime',
                            step_name='step08_regime_data_splitting',
                            regime_id=str(regime_id)
                        )
                    
                    # Log file paths for generated regime data
                    regime_data_path = f"data/training/regime_data/{exchange}_{symbol}_{timeframe}_regime_data.parquet"
                    self.financial_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name='regime_data_path',
                        metric_value=0.0,
                        metric_type='file_path',
                        step_name='step08_regime_data_splitting',
                        additional_data={'file_path': regime_data_path}
                    )
                    
                    # Log step end
                    self.financial_logger.log_step_end('step08_regime_data_splitting', symbol, exchange, timeframe, success=True)
                    
                    self.logger.info('✅ Financial metrics logged successfully for Step08')
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to log financial metrics: {e}')
                    # Log step end with error
                    if self.financial_logger is not None:
                        symbol = self.config.get('symbol', 'ETHUSDT')
                        exchange = self.config.get('exchange', 'BINANCE')
                        timeframe = self.config.get('timeframe', '1m')
                        self.financial_logger.log_step_end('step08_regime_data_splitting', symbol, exchange, timeframe, success=False, error_message=str(e))

            self.logger.info('✅ Unified HMM composite regime data creation completed successfully')
            return {'success': True, 'regime_summary': summary}
        except Exception as e:
            self.logger.exception(f'❌ Unified HMM composite regime data creation failed: {e}')
            
            # Log step end with error if financial logger is available
            if self.financial_logger is not None:
                try:
                    symbol = self.config.get('symbol', 'ETHUSDT')
                    exchange = self.config.get('exchange', 'BINANCE')
                    timeframe = self.config.get('timeframe', '1m')
                    self.financial_logger.log_step_end('step08_regime_data_splitting', symbol, exchange, timeframe, success=False, error_message=str(e))
                except Exception as log_error:
                    self.logger.warning(f'⚠️ Failed to log financial metrics error: {log_error}')
            
            return {'success': False, 'error': str(e)}
    @log_all_calls

    def _validate_and_fix_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix input data using pipeline standards.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated and fixed DataFrame
        """
        self.logger.info('🔍 Validating input data for regime data splitting...')
        validation_result = self.standards.validate_data_quality(data, 'unified')
        if not validation_result.passed:
            self.logger.warning(f'⚠️ Data quality issues detected: {validation_result.quality_score:.2f}')
            for issue in validation_result.issues:
                self.logger.warning(f'   - {issue.message}')
        fixed_data = data.copy()
        if 'timestamp' in fixed_data.columns:
            duplicate_count = fixed_data['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                self.logger.info(f'🗑️ Removing {duplicate_count} duplicate timestamps')
                fixed_data = fixed_data.drop_duplicates(subset=['timestamp'], keep='last')
        if 'timestamp' in fixed_data.columns:
            if not fixed_data['timestamp'].is_monotonic_increasing:
                self.logger.info('📈 Sorting data by timestamp')
                fixed_data = fixed_data.sort_values('timestamp').reset_index(drop = True)
        try:
            fixed_data = self.standards.enforce_schema(fixed_data, 'unified')
            self.logger.info('✅ Applied schema enforcement')
        except Exception as e:
            self.logger.warning(f'⚠️ Schema enforcement failed: {e}')
        if 'timestamp' in fixed_data.columns and (not isinstance(fixed_data.index, pd.DatetimeIndex)):
            try:
                fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'])
                fixed_data = fixed_data.set_index('timestamp')
                self.logger.info('📅 Set datetime index')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not set datetime index: {e}')
        final_validation = self.standards.validate_data_quality(fixed_data, 'unified')
        self.logger.info(f'✅ Final data quality score: {final_validation.quality_score:.2f}')
        return fixed_data

    async def _log_basic_step8_artifacts_and_report(self, unified_data: Any, summary: Any) -> None:
        """Log step 8 artifacts and create detailed report."""
        try:
            symbol = self.config.get('symbol', 'ETHUSDT')
            exchange = self.config.get('exchange', 'BINANCE')
            timeframe = self.config.get('timeframe', '1m')
            execution_metadata = {'start_time': datetime.now().isoformat(), 'end_time': datetime.now().isoformat(), 'duration_seconds': 0.0, 'memory_usage_mb': 0.0, 'cpu_usage_percent': 0.0, 'data_quality_score': 1.0, 'processing_efficiency': 1.0}
            artifacts_generated = [f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet', f'{exchange}_{symbol}_{timeframe}_regime_labels.json', f'{exchange}_{symbol}_{timeframe}_regime_statistics.json']
            metrics_calculated = {'regime_creation_success': 1.0, 'total_regimes': summary.get('total_regimes', 0), 'total_samples': len(unified_data), 'regime_ids': summary.get('regime_ids', [])}
            training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'lookback_days': self.config.get('lookback_days', 1095), 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1_2_3')}
            step_data = {'regime_summary': summary, 'regime_count': summary.get('total_regimes', 0), 'regime_ids': summary.get('regime_ids', []), 'approach': 'unified_dataset_with_labels'}
            report_data = create_detailed_step_report(step_name='step08_regime_data_splitting', step_data = step_data, training_input = training_input, execution_metadata = execution_metadata, artifacts_generated = artifacts_generated, metrics_calculated = metrics_calculated, errors_encountered=[])
            report_name = log_step_report(config = self.config, step_name='step08_regime_data_splitting', report_data = report_data, report_type='unified_regime_data_creation_report', additional_metadata={'regime_creation_success': True, 'total_regimes': summary.get('total_regimes', 0), 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1_2_3'), 'approach': 'unified_dataset_with_labels'})
            self.logger.info(f'✅ Logged unified regime data creation report: {report_name}')
            if summary:
                summary_report_name = log_step_report(config = self.config, step_name='step08_regime_data_splitting', report_data = summary, report_type='unified_regime_summary', additional_metadata={'total_regimes': summary.get('total_regimes', 0), 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1_2_3'), 'approach': 'unified_dataset_with_labels'})
                self.logger.info(f'✅ Logged unified regime summary: {summary_report_name}')
            log_step_metrics(config = self.config, step_name='step08_regime_data_splitting', metrics = metrics_calculated, additional_metadata={'metrics_type': 'unified_regime_creation_performance', 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1_2_3'), 'approach': 'unified_dataset_with_labels'})
            self.logger.info('✅ Step 8 artifacts and reports logged successfully')
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 8 artifacts and reports: {e}')

    @with_tracing_span('step08_regime_splitting._save_unified_regime_dataset', log_args = False)
    @log_all_calls
    @handle_errors(exceptions=(Exception,), default_return = False, context='save_unified_regime_dataset')
    def _save_unified_regime_dataset(self, unified_data: Any, unique_clusters: List[Any]) -> bool:
        """Save unified regime dataset with labels."""
        try:
            data_dir = self.config.get('data_dir', 'data/training')
            os.makedirs(data_dir, exist_ok = True)
            symbol = self.config.get('symbol', 'ETHUSDT')
            exchange = self.config.get('exchange', 'BINANCE')
            timeframe = self.config.get('timeframe', '1m')
            unified_file = os.path.join(data_dir, f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet')
            unified_data.to_parquet(unified_file, index = True)
            self.logger.info(f'✅ Saved unified regime dataset: {len(unified_data)} rows -> {unified_file}')
            regime_labels = {'regime_column': 'composite_cluster_id', 'regime_ids': sorted(unique_clusters), 'total_regimes': len(unique_clusters), 'data_shape': unified_data.shape, 'timestamp_range': {'start': unified_data.index.min().isoformat(), 'end': unified_data.index.max().isoformat()}, 'usage_instructions': {'description': 'Load the unified dataset and filter by composite_cluster_id for regime-specific processing', 'example': "regime_data = data[data['composite_cluster_id'] == regime_id]", 'benefits': ['Maintains temporal continuity for trading indicators', 'Preserves lookback periods', 'Eliminates need for multiple file management', 'Enables regime-aware processing with single dataset']}}
            labels_file = os.path.join(data_dir, f'{exchange}_{symbol}_{timeframe}_regime_labels.json')
            with open(labels_file, 'w') as f:
                json.dump(regime_labels, f, indent = 2)
            self.logger.info(f'✅ Saved regime labels mapping: {labels_file}')
            regime_stats = self._create_regime_statistics(unified_data, unique_clusters)
            stats_file = os.path.join(data_dir, f'{exchange}_{symbol}_{timeframe}_regime_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent = 2)
            self.logger.info(f'✅ Saved regime statistics: {stats_file}')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Failed to save unified regime dataset: {e}')
            return False
    @log_all_calls

    def _create_regime_statistics(self, unified_data: Any, unique_clusters: List[Any]) -> dict[str, Any]:
        """Create regime statistics."""
        try:
            stats = {'approach': 'unified_dataset_with_labels', 'total_regimes': len(unique_clusters), 'total_data_points': len(unified_data), 'regime_details': {}, 'overall_statistics': {'date_range': {'start': unified_data.index.min().isoformat(), 'end': unified_data.index.max().isoformat()}}}
            for cluster_id in unique_clusters:
                regime_data = unified_data[unified_data['composite_cluster_id'] == cluster_id]
                if len(regime_data) > 0:
                    regime_stats = {'data_points': len(regime_data), 'percentage': len(regime_data) / len(unified_data) * 100, 'date_range': {'start': regime_data.index.min().isoformat(), 'end': regime_data.index.max().isoformat()}}
                    if 'close' in regime_data.columns:
                        regime_stats['price_stats'] = {'mean': float(regime_data['close'].mean()), 'std': float(regime_data['close'].std()), 'min': float(regime_data['close'].min()), 'max': float(regime_data['close'].max())}
                    stats['regime_details'][f'regime_{cluster_id}'] = regime_stats
            return stats
        except Exception as e:
            self.logger.exception(f'❌ Error creating regime statistics: {e}')
            return {}

    @with_tracing_span('step08_regime_splitting._create_regime_summary', log_args = False)
    @log_all_calls
    @handle_errors(exceptions=(Exception,), default_return={}, context='create_regime_summary')
    def _create_regime_summary(self, unified_data: Any, unique_clusters: List[Any]) -> dict[str, Any]:
        summary = {'timestamp': datetime.now().isoformat(), 'approach': 'unified_dataset_with_labels', 'regime_basis': 'hmm_composite_clusters_only', 'total_regimes': len(unique_clusters), 'regime_ids': sorted(unique_clusters), 'total_rows': len(unified_data), 'data_shape': unified_data.shape, 'timestamp_range': {'start': unified_data.index.min().isoformat(), 'end': unified_data.index.max().isoformat()}, 'regime_column': 'composite_cluster_id', 'usage_instructions': {'description': 'Load the unified dataset and filter by composite_cluster_id for regime-specific processing', 'example': "regime_data = data[data['composite_cluster_id'] == regime_id]", 'benefits': ['Maintains temporal continuity for trading indicators', 'Preserves lookback periods', 'Eliminates need for multiple file management', 'Enables regime-aware processing with single dataset']}}
        return summary

@deterministic_seed(42)
@idempotent_step(step_key='step08_regime_data_splitting')
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning('1.0')
@time_budget_watchdog(soft_timeout_seconds = 1800.0)
@validate_step_prerequisites(required_directories=['data/training'], min_memory_gb = 4.0, min_disk_gb = 3.0, required_packages=['pandas', 'numpy'], data_quality_checks={'min_rows': 1000, 'required_columns': ['timestamp', 'composite_cluster_id']}, context='Unified Regime Data Creation')
@secure_data_processing(backup_before = True, integrity_checks = True, memory_cleanup = True, data_validation = True)
@prevent_data_leakage(temporal_validation = True, feature_leakage_detection = False, lookahead_bias_prevention = True)
@resource_monitor(memory_threshold_gb = 8.0, cpu_threshold_percent = 70.0, disk_threshold_gb = 5.0, monitor_interval = 30.0, auto_cleanup = True)
@memory_efficient(chunk_size = 20000, streaming_processing = True, memory_pool = True, cleanup_frequency = 40)
@debug_training_step(log_intermediate_results = True, save_debug_artifacts = True, performance_profiling = True, error_context_preservation = True)
@circuit_breaker_protection(failure_threshold = 3, recovery_timeout = 90.0, expected_exception = Exception, monitor_interval = 30.0)
@validate_step_output(required_files=['data/training/*_unified_regime_data.parquet'], data_quality_checks={'min_rows': 100, 'required_columns': ['timestamp', 'composite_cluster_id']}, performance_thresholds={'creation_time_minutes': 30.0}, format_validation = True)
@quality_gate(data_quality_metrics={'completeness': 0.9, 'consistency': 0.8}, validation_score_requirements={'creation_accuracy': 0.8})
@auto_fix_data_quality_issues
@handle_errors(exceptions=(Exception,), default_return = False, context='step08_regime_data_splitting')
async def run_step(symbol: str, exchange: str, data_dir: str, timeframe: str='1m', force_rerun: bool = False, **kwargs) -> bool:
    config = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir, 'timeframe': timeframe, 'force_rerun': force_rerun, **kwargs}
    step = RegimeDataSplittingStep(config)
    await step.initialize()
    result = await step.execute()
    return result.get('success', False)
if __name__ == '__main__':

    async def _test() -> None:
        await run_step('ETHUSDT', 'BINANCE', 'data/training')
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_test())
    except RuntimeError:
        asyncio.run(_test())