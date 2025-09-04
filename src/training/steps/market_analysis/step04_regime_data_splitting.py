from typing import Dict, List, Optional, Union, Any, Tuple
"""Step 4: Regime Data Splitting with Standardized Data Quality Management."

This module creates a unified dataset with regime labels for regime-aware processing.
Uses labels to differentiate regimes instead of creating separate files per regime.
This ensures trading indicators have the necessary lookback periods.
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.core.decorators import handles_errors, traced, validates, cached, log_execution_time, monitor_feature_engineering
import numpy as np
import pandas as pd
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
REQUIRED_MODULES = ['pandas', 'numpy', 'src.utils.centralized_decorators', 'src.utils.logger', 'src.utils.enhanced_mlflow_integration']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
centralized_decorators = PipelineStandards.safe_import('src.utils.centralized_decorators', None)
system_logger = PipelineStandards.safe_import('src.utils.logger', None)
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)
pandas = PipelineStandards.safe_import('pandas', None)
numpy = PipelineStandards.safe_import('numpy', None)

def create_fallback_logger() -> Any:
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:

    def decorator(func: Callable) -> None:
        return func
    return decorator
if system_logger is None:
    system_logger = create_fallback_logger()
if centralized_decorators is None:
    comprehensive_data_validation = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    monitor_feature_engineering = create_fallback_decorator()
    validates = create_fallback_decorator()
    cached = create_fallback_decorator()
    log_execution_time = create_fallback_decorator()
else:
    comprehensive_data_validation = centralized_decorators.comprehensive_data_validation
    handle_errors = centralized_decorators.handle_errors
    memory_efficient = centralized_decorators.memory_efficient
    resource_monitor = centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    validate_data_structure = centralized_decorators.validate_data_structure
    with_tracing_span = centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate
    monitor_feature_engineering = centralized_decorators.monitor_feature_engineering
    validates = centralized_decorators.validates
    cached = centralized_decorators.cached
    log_execution_time = centralized_decorators.log_execution_time
if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
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
logger = system_logger.getChild('Step4RegimeDataSplitting')

class RegimeDataSplittingStep:
    """Step 4: Regime Data Splitting with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('RegimeDataSplittingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Regime Data Splitting Step...')
        self.logger.info('📋 Step 4 Configuration:')
        self.logger.info(f'   - Unified dataset approach: Enabled')
        self.logger.info(f'   - Regime labels: composite_cluster_id')
        self.logger.info(f'   - Memory management: Optimized')
        self.logger.info('✅ Regime Data Splitting Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @traced(span_name='split_data_by_regimes')
    @validates(min_quality_score=0.8, max_correlation=0.95, required_grade='B')
    @cached
    async def split_data_by_regimes(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Create unified dataset with regime labels for regime-aware processing."""
        step_start = time.time()
        self.logger.info(f'🔀 Creating unified dataset with regime labels for {symbol} on {exchange} ({timeframe})')
        try:
            regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                return False
            regime_ids = regime_data['composite_cluster_id'].unique()
            num_regimes = len(regime_ids)
            self.logger.info(f'📊 Found {num_regimes} regimes: {sorted(regime_ids)}')
            if num_regimes < 3:
                self.logger.error(f'❌ Too few regimes: {num_regimes} (minimum 3 required)')
                return False
            if num_regimes > 20:
                self.logger.warning(f'⚠️ Many regimes detected: {num_regimes} (maximum 20 supported)')
            success = await self._create_unified_regime_dataset(regime_data, regime_ids, symbol, exchange, timeframe, data_dir)
            if success:
                self._log_step_timing('Regime Data Splitting', step_start)
                self.logger.info(f'✅ Successfully created unified dataset with {num_regimes} regime labels')
                await self._save_regime_metadata(regime_ids, data_dir, symbol, exchange, timeframe)
                return True
            else:
                self.logger.error('❌ Failed to create unified regime dataset')
                return False
        except Exception as e:
            self.logger.exception(f'❌ Error in regime data splitting: {e}')
            return False

    async def _load_regime_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load HMM regime data with standardized validation."""
        try:
            unified_data_path = Path(self.standards.build_path('unified_data', exchange, symbol)) / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data path not found: {unified_data_path}')
                return None
            regime_file = Path(data_dir) / 'hmm_regimes' / f'{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            if not regime_file.exists():
                self.logger.error(f'❌ Regime file not found: {regime_file}')
                return None
            unified_files = list(unified_data_path.glob('**/*.parquet'))
            if not unified_files:
                self.logger.error(f'❌ No unified data files found in {unified_data_path}')
                return None
            unified_data = []
            for file_path in sorted(unified_files):
                df = pd.read_parquet(file_path)
                df = self.standards.standardize_timestamp(df, 'timestamp')
                df = self.standards.enforce_schema(df, 'unified')
                unified_data.append(df)
            unified_df = pd.concat(unified_data, ignore_index=True)
            regime_df = pd.read_parquet(regime_file)
            regime_df = self.standards.standardize_timestamp(regime_df, 'timestamp')
            merged_data = pd.merge(unified_df, regime_df[['timestamp', 'composite_cluster_id']], on='timestamp', how='inner')
            try:
                retention_ratio = len(merged_data) / max(len(unified_df), 1) if len(unified_df) else 0.0
                self.logger.info(f'📈 Merge retention ratio: {retention_ratio:.3f}')
                min_retention = float(self.config.get('regime_merge_min_retention', 0.8))
                if retention_ratio < min_retention:
                    self.logger.warning(f'⚠️ Low retention after regime merge: {retention_ratio:.3f} (< {min_retention:.2f}). Check timestamp alignment and data coverage.')
            except Exception:
                pass
            self.logger.info(f'✅ Loaded {len(merged_data)} data points with regime information')
            return merged_data
        except Exception as e:
            self.logger.exception(f'❌ Error loading regime data: {e}')
            return None

    async def _create_unified_regime_dataset(self, data: pd.DataFrame, regime_ids: List[int], symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Create unified dataset with regime labels."""
        try:
            data = data.sort_values('timestamp').reset_index(drop=True)
            training_dir = ensure_directory(Path(data_dir) / 'training')
            unified_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            data.to_parquet(unified_file, index=False)
            self.logger.info(f'✅ Saved unified regime dataset: {len(data)} rows -> {unified_file}')
            regime_stats = self._calculate_regime_statistics(data, regime_ids)
            stats_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_statistics.json'
            import json
from src.core.decorators.errors import handles_errors
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent=2)
            self.logger.info(f'✅ Saved regime statistics: {stats_file}')
            regime_labels = {'regime_column': 'composite_cluster_id', 'regime_ids': sorted(regime_ids), 'total_regimes': len(regime_ids), 'data_shape': data.shape, 'timestamp_range': {'start': data['timestamp'].min().isoformat(), 'end': data['timestamp'].max().isoformat()}}
            labels_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_labels.json'
            safe_json_dump(regime_labels, labels_file, indent=2)
            self.logger.info(f'✅ Saved regime labels mapping: {labels_file}')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error creating unified regime dataset: {e}')
            return False

    def _calculate_regime_statistics(self, data: pd.DataFrame, regime_ids: List[int]) -> Dict[str, Any]:
        """Calculate statistics for each regime."""
        try:
            stats = {'total_regimes': len(regime_ids), 'total_data_points': len(data), 'regime_details': {}, 'overall_statistics': {'date_range': {'start': data['timestamp'].min().isoformat(), 'end': data['timestamp'].max().isoformat()}, 'price_stats': {'mean': float(data['close'].mean()) if 'close' in data.columns else None, 'std': float(data['close'].std()) if 'close' in data.columns else None, 'min': float(data['close'].min()) if 'close' in data.columns else None, 'max': float(data['close'].max()) if 'close' in data.columns else None}}}
            for regime_id in regime_ids:
                regime_data = data[data['composite_cluster_id'] == regime_id]
                if len(regime_data) > 0:
                    regime_stats = {'data_points': len(regime_data), 'percentage': len(regime_data) / len(data) * 100, 'date_range': {'start': regime_data['timestamp'].min().isoformat(), 'end': regime_data['timestamp'].max().isoformat()}}
                    if 'close' in regime_data.columns:
                        regime_stats['price_stats'] = {'mean': float(regime_data['close'].mean()), 'std': float(regime_data['close'].std()), 'min': float(regime_data['close'].min()), 'max': float(regime_data['close'].max())}
                    stats['regime_details'][f'regime_{regime_id}'] = regime_stats
            return stats
        except Exception as e:
            self.logger.exception(f'❌ Error calculating regime statistics: {e}')
            return {}

    async def _save_regime_metadata(self, regime_ids: List[int], data_dir: str, symbol: str, exchange: str, timeframe: str) -> None:
        """Save metadata about the unified regime dataset."""
        try:
            metadata = {'approach': 'unified_dataset_with_labels', 'total_regimes': len(regime_ids), 'regime_ids': sorted(regime_ids), 'created_at': time.time(), 'data_structure': {'main_file': f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet', 'regime_column': 'composite_cluster_id', 'regime_labels_file': f'{exchange}_{symbol}_{timeframe}_regime_labels.json', 'regime_statistics_file': f'{exchange}_{symbol}_{timeframe}_regime_statistics.json'}, 'usage_instructions': {'description': 'Load the unified dataset and filter by composite_cluster_id for regime-specific processing', 'example': "regime_data = data[data['composite_cluster_id'] == regime_id]", 'benefits': ['Maintains temporal continuity for trading indicators', 'Preserves lookback periods', 'Eliminates need for multiple file management', 'Enables regime-aware processing with single dataset']}}
            metadata_file = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_metadata.json'
            safe_json_dump(metadata, metadata_file, indent=2)
            self.logger.info(f'✅ Regime metadata saved: {metadata_file}')
        except Exception as e:
            self.logger.exception(f'❌ Error saving regime metadata: {e}')

@traced(span_name='execute_regime_data_splitting')
@validates()
@handles_errors
@cached
@log_execution_time
@monitor_feature_engineering
async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str=None, force_rerun: bool=False, config: dict[str, Any]=None) -> bool:
    """Run Step 4: Regime Data Splitting with standardized data quality management."
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun flag
        config: Configuration dictionary
        
    Returns:
        bool: Success status
    """
    logger.info('🚀 Starting Step 4: Regime Data Splitting with Standardized Data Quality Management')
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    try:
        step = RegimeDataSplittingStep(config or {})
        await step.initialize()
        success = await step.split_data_by_regimes(symbol, exchange, timeframe, data_dir)
        if success:
            logger.info('✅ Step 4: Regime Data Splitting completed successfully')
        else:
            logger.error('❌ Step 4: Regime Data Splitting failed')
        return success
    except Exception as e:
        logger.exception(f'❌ Error in Step 4: {e}')
        return False
if __name__ == '__main__':

    async def test() -> None:
        test_config = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m'}
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache', force_rerun=False, config=test_config)
        print(f'Test result: {success}')
    asyncio.run(test())