"""Step 4: Regime Data Splitting with Standardized Data Quality Management."

This module creates a unified dataset with regime labels for regime-aware processing.
Uses labels to differentiate regimes instead of creating separate files per regime.
This ensures trading indicators have the necessary lookback periods.
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from src.core.decorators import handles_errors, traced, validates, cached, log_execution_time
try:
    from src.core.domain.decorators_extended import monitor_feature_engineering
except Exception:
    def monitor_feature_engineering(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator
import pandas as pd
import numpy as np
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
REQUIRED_MODULES = ['pandas', 'numpy', 'src.utils.logger', 'src.utils.enhanced_mlflow_integration']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
from src.utils.logger import system_logger
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)
pandas = pd
numpy = np

def create_fallback_logger() -> Any:
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
if system_logger is None:
    system_logger = create_fallback_logger()
# Use the real decorators that are already imported
comprehensive_data_validation = validates
handle_errors = handles_errors
memory_efficient = cached
resource_monitor = log_execution_time
secure_data_processing = handles_errors
validate_data_structure = validates
with_tracing_span = traced
quality_gate = validates
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
    @validates()
    @cached()
    async def split_data_by_regimes(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Dict[str, Any]:
        """Create unified dataset with regime labels for regime-aware processing.

        Returns a dictionary suitable for tests and downstream usage:
        {"success": bool, "unified_data": pd.DataFrame | None, "regime_stats": dict | None, "saved_path": str | None}
        """
        step_start = time.time()
        self.logger.info(f'🔀 Creating unified dataset with regime labels for {symbol} on {exchange} ({timeframe})')
        try:
            regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                return {"success": False, "error": "regime_data_not_found", "unified_data": None, "regime_stats": None, "saved_path": None}
            regime_ids = regime_data['composite_cluster_id'].unique()
            num_regimes = len(regime_ids)
            self.logger.info(f'📊 Found {num_regimes} regimes: {sorted(regime_ids)}')
            if num_regimes < 3:
                self.logger.error(f'❌ Too few regimes: {num_regimes} (minimum 3 required)')
                return {"success": False, "error": "too_few_regimes", "unified_data": None, "regime_stats": None, "saved_path": None}
            if num_regimes > 20:
                self.logger.warning(f'⚠️ Many regimes detected: {num_regimes} (maximum 20 supported)')
            dataset_info = await self._create_unified_regime_dataset(regime_data, regime_ids, data_dir, symbol, exchange, timeframe)
            if isinstance(dataset_info, dict):
                self._log_step_timing('Regime Data Splitting', step_start)
                self.logger.info(f'✅ Successfully created unified dataset with {num_regimes} regime labels')
                await self._save_regime_metadata(regime_ids, data_dir, symbol, exchange, timeframe)
                return {
                    "success": True,
                    "unified_data": dataset_info.get("unified_data"),
                    "regime_stats": dataset_info.get("regime_stats"),
                    "saved_path": dataset_info.get("saved_path"),
                }
            else:
                self.logger.error('❌ Failed to create unified regime dataset')
                return {"success": False, "error": "creation_failed", "unified_data": None, "regime_stats": None, "saved_path": None}
        except Exception as e:
            self.logger.exception(f'❌ Error in regime data splitting: {e}')
            return {"success": False, "error": str(e), "unified_data": None, "regime_stats": None, "saved_path": None}

    async def _load_regime_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load HMM regime data with standardized validation."""
        try:
            unified_data_path = Path(self.standards.build_path('unified_data', exchange, symbol)) / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data path not found: {unified_data_path}')
                return None
            regime_primary = Path('data') / 'hmm_regimes' / f'{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            regime_alternative = Path(data_dir) / 'hmm_regimes' / f'{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            regime_file = regime_primary if regime_primary.exists() else regime_alternative
            if not regime_file.exists():
                self.logger.error(f'❌ Regime file not found: {regime_primary} or {regime_alternative}')
                return None
            unified_files = list(unified_data_path.glob('**/*.parquet'))
            if not unified_files:
                self.logger.error(f'❌ No unified data files found in {unified_data_path}')
                return None
            unified_data = []
            for file_path in sorted(unified_files):
                df = pandas.read_parquet(file_path)
                df = self.standards.standardize_timestamp(df, 'timestamp')
                df = self.standards.enforce_schema(df, 'unified')
                unified_data.append(df)
            unified_df = pd.concat(unified_data, ignore_index=True)
            regime_df = pandas.read_parquet(regime_file)
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

    def _save_unified_dataset(self, data: pd.DataFrame, training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save the unified regime dataset to parquet file."""
        try:
            unified_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            data.to_parquet(unified_file, index=False)
            self.logger.info(f'✅ Saved unified regime dataset: {len(data)} rows -> {unified_file}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error saving unified dataset: {e}')
            return False

    def _save_regime_statistics(self, data: pd.DataFrame, regime_ids: List[int], training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save regime statistics to JSON file."""
        try:
            regime_stats = self._calculate_regime_statistics(data, regime_ids)
            stats_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_statistics.json'
            import json
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent=2)
            self.logger.info(f'✅ Saved regime statistics: {stats_file}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error saving regime statistics: {e}')
            return False

    def _save_regime_labels(self, data: pd.DataFrame, regime_ids: List[int], training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save regime labels mapping to JSON file."""
        try:
            regime_labels = {
                'regime_column': 'composite_cluster_id',
                'regime_ids': sorted(regime_ids),
                'total_regimes': len(regime_ids),
                'data_shape': data.shape,
                'timestamp_range': {
                    'start': data['timestamp'].min().isoformat(),
                    'end': data['timestamp'].max().isoformat()
                }
            }
            labels_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_labels.json'
            safe_json_dump(regime_labels, labels_file, indent=2)
            self.logger.info(f'✅ Saved regime labels mapping: {labels_file}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error saving regime labels: {e}')
            return False

    async def _create_unified_regime_dataset(self, data: pd.DataFrame, regime_ids: List[int], data_dir: str, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any] | None:
        """Create unified dataset with regime labels and return dataset info."""
        try:
            # Prepare data
            data = data.sort_values('timestamp').reset_index(drop=True)
            training_dir = ensure_directory(Path(data_dir) / 'training' / 'regime_splits')
            
            # Save unified dataset
            if not self._save_unified_dataset(data, training_dir, exchange, symbol, timeframe):
                return None
            
            # Save regime statistics
            if not self._save_regime_statistics(data, regime_ids, training_dir, exchange, symbol, timeframe):
                return None
            
            # Save regime labels
            if not self._save_regime_labels(data, regime_ids, training_dir, exchange, symbol, timeframe):
                return None

            saved_path = str(training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet')
            return {
                "unified_data": data,
                "regime_stats": self._calculate_regime_statistics(data, regime_ids),
                "saved_path": saved_path,
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error creating unified regime dataset: {e}')
            return None

    def _calculate_regime_statistics(self, data: pd.DataFrame, regime_ids: List[int]) -> Dict[str, Any]:
        """Calculate simple per-regime statistics compatible with tests."""
        try:
            stats: Dict[int, Dict[str, Any]] = {}
            for regime_id in regime_ids:
                regime_data = data[data['composite_cluster_id'] == regime_id]
                if len(regime_data) == 0:
                    stats[int(regime_id)] = {"count": 0, "duration_minutes": 0, "mean_volume": 0.0}
                    continue
                start_ts = regime_data['timestamp'].min()
                end_ts = regime_data['timestamp'].max()
                try:
                    # If timestamps are int64 ms
                    duration_minutes = int((int(end_ts) - int(start_ts)) / 60000)
                except Exception:
                    # If timestamps are datetime
                    duration_minutes = int((pd.to_datetime(end_ts) - pd.to_datetime(start_ts)).total_seconds() / 60)
                mean_volume = float(regime_data['volume'].mean()) if 'volume' in regime_data.columns else 0.0
                stats[int(regime_id)] = {
                    "count": int(len(regime_data)),
                    "duration_minutes": duration_minutes,
                    "mean_volume": mean_volume,
                }
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
@handles_errors()
@cached()
@log_execution_time()
@monitor_feature_engineering()
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
        result = await step.split_data_by_regimes(symbol, exchange, timeframe, data_dir)
        success = bool(result.get("success", bool(result))) if isinstance(result, dict) else bool(result)
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