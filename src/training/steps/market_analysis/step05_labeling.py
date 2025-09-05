"""Step 5: Labeling with Simplified Architecture.

This module provides a simplified, well-structured labeling step that maintains
all functionality while dramatically reducing complexity through modular design.

Key Simplifications:
- Extracted monitoring systems into separate modules
- Extracted decorator system with fallback mechanisms  
- Extracted labeling components into focused classes
- Centralized dependency management
- Simplified main class focused on core functionality
"""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Import our simplified modules
from .dependencies import (
    dependency_manager,
    get_ensure_directory,
    get_pipeline_standards,
    get_safe_json_dump,
    get_system_logger,
)
from .decorators import (
    cached,
    comprehensive_data_validation,
    handles_errors,
    log_execution_time,
    log_step_artifact_with_standardized_name,
    log_step_dataframe_with_standardized_name,
    log_step_metrics,
    log_step_report,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    traced,
    validate_data_structure,
    validates,
    with_enhanced_mlflow_logging,
)
from .labeling_components import ComprehensiveLabeling
from .monitoring import (
    ComprehensiveValidationFramework,
    EnhancedErrorHandler,
    FunctionCallMonitor,
    PerformanceMonitor,
    comprehensive_function_monitor,
    comprehensive_validation,
    enhanced_error_handler,
    performance_monitor,
)

# Get system components
system_logger = get_system_logger()
pipeline_standards = get_pipeline_standards()
ensure_directory = get_ensure_directory()
safe_json_dump = get_safe_json_dump()


class LabelingStep:
    """Simplified Step 5: Labeling with modular architecture.

    This class focuses on core labeling functionality while delegating
    monitoring, validation, and complex logic to specialized modules.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('LabelingStep')
        self.start_time: Optional[float] = None
        self.step_timings: Dict[str, float] = {}
        
        # Initialize core components
        self.comprehensive_labeling = ComprehensiveLabeling(config, self.logger)
        
        # Initialize monitoring systems (optional)
        self._initialize_monitoring_systems()
        
        # Validate environment
        self._validate_environment()
        
        self.logger.info('✅ LabelingStep initialized with simplified architecture')

    def _initialize_monitoring_systems(self) -> None:
        """Initialize monitoring systems if available."""
        try:
            # Initialize monitoring systems
            self.function_monitor = FunctionCallMonitor(self.logger)
            self.error_handler = EnhancedErrorHandler(self.logger)
            self.performance_monitor = PerformanceMonitor(self.logger)
            self.validation_framework = ComprehensiveValidationFramework(self.logger)
            
            # Setup function monitoring
            self._setup_function_monitoring()
            
            self.logger.info('✅ Monitoring systems initialized')
        except Exception as e:
            self.logger.warning(f'⚠️ Monitoring systems not available: {e}')
            self.function_monitor = None
            self.error_handler = None
            self.performance_monitor = None
            self.validation_framework = None

    def _setup_function_monitoring(self) -> None:
        """Setup function monitoring with validation rules and performance thresholds."""
        if self.function_monitor is None:
            return
        
        # Set performance thresholds for key functions
        self.function_monitor.performance_thresholds = {
            'execute_labeling': 300.0,  # 5 minutes
            'generate_comprehensive_labels': 180.0,  # 3 minutes
        }
        
        # Set custom validation rules
        self.function_monitor.validation_rules = {
            'execute_labeling': self._validate_execute_labeling_result,
            'generate_comprehensive_labels': self._validate_labeling_result,
        }

    def _validate_execute_labeling_result(self, call_record) -> bool:
        """Validate execute_labeling function result."""
        if call_record.return_value is None:
            return False
        return isinstance(call_record.return_value, bool)

    def _validate_labeling_result(self, call_record) -> bool:
        """Validate labeling function result."""
        if call_record.return_value is None:
            return False
        return isinstance(call_record.return_value, pd.DataFrame) and len(call_record.return_value) > 0

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [k for k, ok in dependency_manager.get_dependency_status().items() if not ok]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    async def initialize(self) -> None:
        """Initialize the labeling step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Labeling Step...')
        self.logger.info('📋 Step 5 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Labeling Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    def _compute_labeling_fingerprint(self, triple_barrier_path: Path) -> Dict[str, Any]:
        """Compute a stable fingerprint of source labeling inputs to ensure idempotence."""
        try:
            stat = triple_barrier_path.stat()
            relevant_cfg = {
                'vectorized_labelling_orchestrator': self.config.get('vectorized_labelling_orchestrator', {}),
                'labeling': self.config.get('labeling', {}),
            }
            relevant_cfg_json = json.dumps(relevant_cfg, sort_keys=True, default=str)
            import hashlib
            cfg_hash = hashlib.sha256(relevant_cfg_json.encode('utf-8')).hexdigest()
            return {
                'source_path': str(triple_barrier_path),
                'source_size': stat.st_size,
                'source_mtime': int(stat.st_mtime),
                'config_hash': cfg_hash
            }
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to compute labeling fingerprint: {e}')
            return {}

    @comprehensive_function_monitor
    @comprehensive_validation
    @performance_monitor
    @enhanced_error_handler
    @traced(span_name='execute_labeling')
    @validates()
    @handles_errors()
    @cached()
    @log_execution_time()
    @comprehensive_data_validation
    @memory_efficient
    @resource_monitor
    @secure_data_processing
    @validate_data_structure
    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = 'data_cache',
        force_rerun: bool = False,
    ) -> bool:
        """Execute the labeling step with comprehensive monitoring."""
        step_start = time.time()
        self.logger.info(f'🚀 Executing Labeling for {symbol} on {exchange}')
        
        try:
            # Setup paths
            triple_barrier_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            if not triple_barrier_path.exists():
                self.logger.error(f'❌ Triple barrier labels not found at {triple_barrier_path}')
                return False
            
            self.logger.info(f'📁 Loading triple barrier labels from {triple_barrier_path}')
            labeled_dir = ensure_directory(Path(data_dir) / 'training' / 'labeled_data')
            output_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeled_data.parquet'
            metadata_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeling_metadata.json'
            
            # Check for idempotence
            current_fp = self._compute_labeling_fingerprint(triple_barrier_path)
            if not force_rerun and output_path.exists() and metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        existing_meta = json.load(f)
                    existing_fp = existing_meta.get('source_fingerprint', {})
                    if existing_fp == current_fp and existing_meta.get('total_samples', 0) > 0:
                        self.logger.info('🟢 Labeling is idempotent: existing outputs match current inputs. Skipping recomputation.')
                        self._log_step_timing('Labeling (skipped)', step_start)
                        return True
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to read existing labeling metadata: {e}')
            
            # Load data
            data = pd.read_parquet(triple_barrier_path)
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            
            # Ensure regime labels are present/consistent
            try:
                from .utils.regime_data_access import ensure_regime_labels, get_regime_column
                data = ensure_regime_labels(
                    data,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_dir=data_dir,
                )
                detected_col = get_regime_column(data)
                if detected_col and detected_col != self.comprehensive_labeling.regime_aware_labeling.regime_col:
                    self.logger.info(f"🔁 Using detected regime column '{detected_col}' instead of '{self.comprehensive_labeling.regime_aware_labeling.regime_col}'")
                    self.comprehensive_labeling.regime_aware_labeling.regime_col = detected_col
            except Exception:
                pass
            
            # Generate comprehensive labels
            data = await self.comprehensive_labeling.generate_comprehensive_labels(data, symbol, exchange, timeframe)
            if data is None:
                self.logger.error('❌ Comprehensive labeling failed')
                return False
            
            # Save labeled data
            data.to_parquet(output_path)
            self.logger.info(f'✅ Labeled data saved to {output_path}')
            
            # Generate metadata
            label_distribution = {}
            if 'label' in data.columns:
                label_distribution = data['label'].value_counts().to_dict()
            
            metadata = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'total_samples': int(len(data)),
                'label_distribution': label_distribution,
                'created_at': pd.Timestamp.now().isoformat(),
                'labeling_config': self.config.get('labeling', {}),
                'source_fingerprint': current_fp
            }
            safe_json_dump(metadata, metadata_path, indent=2, default=str)
            
            self._log_step_timing('execute_labeling', step_start)
            
            # Log artifacts and reports
            await self._log_step5_artifacts_and_report(symbol, exchange, timeframe, data_dir, data, output_path, metadata_path)
            
            # Generate monitoring reports if available
            if self.function_monitor:
                await self._generate_and_log_monitoring_reports()
            
            return True
            
        except Exception as e:
            self.logger.exception(f'❌ Error in labeling: {e}')
            
            # Generate monitoring reports even on failure
            if self.function_monitor:
                await self._generate_and_log_monitoring_reports()
            
            return False

    async def _generate_and_log_monitoring_reports(self) -> None:
        """Generate and log comprehensive monitoring reports."""
        try:
            if not self.function_monitor:
                return
            
            self.logger.info('📊 Generating comprehensive monitoring reports...')
            
            # Generate function call report
            report = self.function_monitor.generate_comprehensive_report()
            self.function_monitor.log_detailed_report(report)
            
            # Generate error summary
            if self.error_handler:
                error_summary = self.error_handler.generate_error_summary_report()
                self.error_handler.log_error_summary_report(error_summary)
            
            # Generate performance report
            if self.performance_monitor:
                performance_report = self.performance_monitor.generate_performance_report()
                self.performance_monitor.log_performance_report(performance_report)
            
            # Generate validation report
            if self.validation_framework:
                validation_report = self.validation_framework.generate_validation_report()
                self.validation_framework.log_validation_report(validation_report)
            
            self.logger.info('✅ Monitoring reports generated and logged successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to generate monitoring reports: {e}')

    async def _log_step5_artifacts_and_report(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        labeled_data: pd.DataFrame,
        output_path: Path,
        metadata_path: Path,
    ) -> None:
        """Log step 5 artifacts and create detailed report."""
        try:
            execution_metadata = {
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0.0,
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'data_quality_score': 1.0,
                'processing_efficiency': 1.0
            }
            
            artifacts_generated = [
                str(output_path),
                str(metadata_path),
                f'{exchange}_{symbol}_{timeframe}_labeling_metrics.json'
            ]
            
            metrics_calculated = {
                'labeling_success': 1.0,
                'total_samples': len(labeled_data) if labeled_data is not None else 0,
                'labeled_samples': len(labeled_data[labeled_data['label'].notna()]) if labeled_data is not None else 0,
                'label_distribution': labeled_data['label'].value_counts().to_dict() if labeled_data is not None and 'label' in labeled_data.columns else {},
                'triple_barrier_distribution': labeled_data['triple_barrier_label'].value_counts().to_dict() if labeled_data is not None and 'triple_barrier_label' in labeled_data.columns else {}
            }
            
            training_input = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir
            }
            
            step_data = {
                'output_path': str(output_path),
                'metadata_path': str(metadata_path),
                'data_shape': list(labeled_data.shape) if labeled_data is not None else [],
                'label_columns': list(labeled_data.columns) if labeled_data is not None else []
            }
            
            report_data = {
                'step_name': 'step05_labeling',
                'step_data': step_data,
                'training_input': training_input,
                'execution_metadata': execution_metadata,
                'artifacts_generated': artifacts_generated,
                'metrics_calculated': metrics_calculated,
                'errors_encountered': []
            }
            
            report_name = log_step_report(
                config=self.config,
                step_name='step05_labeling',
                report_data=report_data,
                report_type='labeling_report',
                additional_metadata={
                    'labeling_success': True,
                    'timeframe': timeframe,
                    'asset': symbol,
                    'lookback_period': self.config.get('lookback_days', 1095),
                    'project_version': self.config.get('project_version', '1.0.0')
                }
            )
            self.logger.info(f'✅ Logged labeling report: {report_name}')
            
            if labeled_data is not None:
                artifact_name = log_step_dataframe_with_standardized_name(
                    config=self.config,
                    step_name='step05_labeling',
                    df=labeled_data,
                    artifact_type='labeled_data',
                    additional_metadata={
                        'artifact_type': 'labeled_data',
                        'dataframe_shape': list(labeled_data.shape),
                        'label_distribution': labeled_data['label'].value_counts().to_dict() if 'label' in labeled_data.columns else {},
                        'asset': symbol,
                        'lookback_period': self.config.get('lookback_days', 1095),
                        'project_version': self.config.get('project_version', '1.0.0'),
                        'timeframe': timeframe
                    }
                )
                self.logger.info(f'✅ Logged labeled data: {artifact_name}')
            
            if metadata_path.exists():
                metadata_artifact_name = log_step_artifact_with_standardized_name(
                    config=self.config,
                    step_name='step05_labeling',
                    artifact_path=str(metadata_path),
                    artifact_type='labeling_metadata',
                    additional_metadata={
                        'metadata_type': 'labeling_metadata',
                        'timeframe': timeframe,
                        'asset': symbol,
                        'lookback_period': self.config.get('lookback_days', 1095),
                        'project_version': self.config.get('project_version', '1.0.0')
                    }
                )
                self.logger.info(f'✅ Logged labeling metadata: {metadata_artifact_name}')
            
            log_step_metrics(
                config=self.config,
                step_name='step05_labeling',
                metrics=metrics_calculated,
                additional_metadata={
                    'metrics_type': 'labeling_performance',
                    'timeframe': timeframe,
                    'asset': symbol,
                    'lookback_period': self.config.get('lookback_days', 1095),
                    'project_version': self.config.get('project_version', '1.0.0')
                }
            )
            self.logger.info('✅ Step 5 artifacts and reports logged successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 5 artifacts and reports: {e}')

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute labeling step with validation."""
        try:
            self.logger.info("🏷️ Starting labeling step with validation...")
            
            # Validate input data if available
            data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
            if data is not None and isinstance(data, pd.DataFrame):
                data = self._validate_and_fix_input_data(data)
                pipeline_state['dataframe'] = data
            
            # Execute labeling
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data')
            
            success = await self.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            return {
                'success': success,
                'step_name': 'step05_labeling',
                'message': 'Labeling completed successfully' if success else 'Labeling failed'
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Labeling step failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'step05_labeling'
            }
    
    def _validate_and_fix_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix input data using pipeline standards."""
        if pipeline_standards is None:
            self.logger.warning("⚠️ Pipeline standards not available, skipping validation")
            return data
        
        self.logger.info("🔍 Validating input data for labeling...")
        
        # Validate data quality using pipeline standards
        validation_result = pipeline_standards.validate_data_quality(data, 'unified')
        
        if not validation_result.passed:
            self.logger.warning(f"⚠️ Data quality issues detected: {validation_result.quality_score:.2f}")
            for issue in validation_result.issues:
                self.logger.warning(f"   - {issue.message}")
        
        # Apply fixes for common issues
        fixed_data = data.copy()
        
        # Fix duplicate timestamps
        if 'timestamp' in fixed_data.columns:
            duplicate_count = fixed_data['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                self.logger.info(f"🗑️ Removing {duplicate_count} duplicate timestamps")
                fixed_data = fixed_data.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Fix non-monotonic index
        if 'timestamp' in fixed_data.columns:
            if not fixed_data['timestamp'].is_monotonic_increasing:
                self.logger.info("📈 Sorting data by timestamp")
                fixed_data = fixed_data.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure proper data types using pipeline standards
        try:
            fixed_data = pipeline_standards.enforce_schema(fixed_data, 'unified')
            self.logger.info("✅ Applied schema enforcement")
        except Exception as e:
            self.logger.warning(f"⚠️ Schema enforcement failed: {e}")
        
        # Set datetime index if timestamp column exists
        if 'timestamp' in fixed_data.columns and not isinstance(fixed_data.index, pd.DatetimeIndex):
            try:
                fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'])
                fixed_data = fixed_data.set_index('timestamp')
                self.logger.info("📅 Set datetime index")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not set datetime index: {e}")
        
        # Final validation
        final_validation = pipeline_standards.validate_data_quality(fixed_data, 'unified')
        self.logger.info(f"✅ Final data quality score: {final_validation.quality_score:.2f}")
        
        return fixed_data


async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the labeling step with simplified architecture.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = {}
    if data_dir is None:
        if pipeline_standards:
            data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
        else:
            data_dir = 'data_cache'
    
    step_config = {
        'SYMBOL': symbol,
        'EXCHANGE': exchange,
        'TIMEFRAME': timeframe,
        'DATA_DIR': data_dir,
        'labeling': {
            'enable_meta_labeling': True,
            'enable_trend_labels': True,
            'enable_volatility_labels': True,
            'composite_label_strategy': 'weighted_combination'
        },
        'vectorized_labelling_orchestrator': {
            'auto_recalculate_hmm_barriers': True,
            'hmm_barrier_regime_column': 'hmm_regime',
            'time_barrier_minutes': 30,
            'max_lookahead': 100,
            'profit_take_multiplier': 0.002,
            'stop_loss_multiplier': 0.001
        },
        **config
    }
    
    step = LabelingStep(step_config)
    await step.initialize()
    return await step.execute_labeling(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )


if __name__ == '__main__':
    async def test() -> None:
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Step 5 result: {success}')
    
    asyncio.run(test())