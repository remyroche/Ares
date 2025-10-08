"""
Pre-Training Sub-Pipeline - Feature Engineering Steps

This module provides the pre-training sub-pipeline with the 4 feature engineering steps
that were moved from market_analysis:

1. multi_horizon_profit_labeler - Apply multi-horizon profit labeling
2. feature_lookback_optimization - Optimize feature lookback periods
3. interactive_feature_generation - End-to-end interactive feature generation with comprehensive approach
4. final_feature_selection - Final multi-stage feature selection (120→100→80→60)

Each step can receive a timeframe parameter, with default 15m.
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import json

try:  # pragma: no cover - platform specific import
    import resource
except ImportError:  # pragma: no cover
    resource = None


class PipelineResultDict(TypedDict, total=False):
    """Type definition for pipeline execution results."""
    success: bool
    execution_time: float
    total_steps: int
    completed_steps: int
    results: Dict[str, Any]
    error_message: Optional[str]

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from src.utils.tprint import tprint, tprint_error, tprint_warning

# Import component system
from .components import ComponentFactory, ComponentConfig
from .metrics_sink import MetricsSink, MetricsSinkConfig

logger = system_logger.getChild('PreTrainingSubPipeline')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class LoggingConfig:
    """Logging configuration for the sub-pipeline."""
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = False
    log_file: Optional[str] = None

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"  # Default timeframe for pre-training steps (analyst)
    data_dir: str = "historical_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    skip_next_pipeline: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    label_imbalance_warning_threshold: float = 0.75
    nan_rate_warning_threshold: float = 0.05
    duplicate_index_warning_threshold: float = 0.02
    metrics_output_path: Optional[str] = None
    metrics_output_format: str = "csv"
    metrics_prometheus_enabled: bool = False
    """
    Metrics capture configuration.

    Defaults:
        metrics_output_path: ``artifacts/pre_training_metrics.<format>``
        metrics_output_format: ``csv``
        metrics_prometheus_enabled: ``False``
    """

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = False
    output_files: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class PreTrainingSubPipeline:
    """
    Pre-Training Sub-Pipeline for Feature Engineering Steps.

    Executes the 4 feature engineering steps in sequence:
    1. multi_horizon_profit_labeler
    2. feature_lookback_optimization
    3. interactive_feature_generation
    4. final_feature_selection
    """

    def __init__(self):
        """Initialize the pre-training sub-pipeline."""
        self.logger = logger.getChild('PreTrainingSubPipeline')
        self.results: List[SubPipelineResult] = []
        self._current_pipeline_state: Dict[str, Any] = {}
        self._metrics_sink: Optional[MetricsSink] = None

    async def execute_pipeline(self, config: SubPipelineConfig) -> PipelineResultDict:
        """
        Execute the complete pre-training pipeline.

        Args:
            config: Configuration for pipeline execution

        Returns:
            PipelineResultDict containing execution results with typed fields
        """
        self.logger.info('🚀 Starting Pre-Training Sub-Pipeline execution')
        self.logger.info(f'📊 Symbol: {config.symbol}, Exchange: {config.exchange}')
        self.logger.info(f'⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}')

        start_time = datetime.now()
        tprint(f"🚀 Starting Pre-Training Sub-Pipeline execution for {config.symbol} on {config.exchange}")
        tprint(f"⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}")
        tprint(f"📊 Configuration: force_rerun={config.force_rerun}, parallel={config.parallel_processing}")

        metrics_sink = self._create_metrics_sink(config)
        self._metrics_sink = metrics_sink
        step_metric_records: List[Dict[str, Any]] = []

        results = {
            'success': False,
            'execution_time': 0.0,
            'total_steps': 4,
            'completed_steps': 0,
            'results': {},
            'error_message': None,
        }

        try:
            # Step 1: Multi-Horizon Profit Labeler
            tprint("🎯 Step 1: Multi-Horizon Profit Labeler")
            self.logger.info('🎯 Step 1: Multi-Horizon Profit Labeler')
            mh_result = await self._execute_multi_horizon_profit_labeler(config)
            if mh_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('multi_horizon_profit_labeler', mh_result, results, metrics_sink, step_metric_records)
            if not mh_result.success:
                tprint(f"❌ Multi-horizon profit labeling failed: {mh_result.error_message}")
                self.logger.error(f'❌ Multi-horizon profit labeling failed: {mh_result.error_message}')
                results['error_message'] = mh_result.error_message
                return self._finalize_results(results, start_time, metrics_sink, step_metric_records)

            tprint(f"✅ Multi-horizon profit labeling completed for {config.symbol}")

            # Validate artifacts before updating state
            if 'multi_horizon_labeling_result' in mh_result.artifacts:
                labeled_data = mh_result.artifacts.get('multi_horizon_labeling_result', {}).get('labeled_data', pd.DataFrame())
                if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
                    tprint(f"   → Labels generated: {len(labeled_data.columns)} columns")
                    results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
                    self._current_pipeline_state.update(mh_result.artifacts)
                else:
                    tprint_error("❌ Multi-horizon labeling artifact validation failed: labeled_data is empty or invalid")
                    results['error_message'] = "Multi-horizon labeling artifact validation failed"
                    return self._finalize_results(results, start_time, metrics_sink, step_metric_records)
            else:
                tprint_error("❌ Multi-horizon labeling artifact validation failed: missing 'multi_horizon_labeling_result'")
                results['error_message'] = "Missing multi_horizon_labeling_result artifact"
                return self._finalize_results(results, start_time, metrics_sink, step_metric_records)

            # Step 2: Feature Lookback Optimization
            tprint("⚙️ Step 2: Feature Lookback Optimization")
            self.logger.info('⚙️ Step 2: Feature Lookback Optimization')
            flo_result = await self._execute_feature_lookback_optimization(config)
            if flo_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('feature_lookback_optimization', flo_result, results, metrics_sink, step_metric_records)
            if not flo_result.success:
                tprint(f"❌ Feature lookback optimization failed: {flo_result.error_message}")
                self.logger.error(f'❌ Feature lookback optimization failed: {flo_result.error_message}')
                results['error_message'] = flo_result.error_message
                return self._finalize_results(results, start_time, metrics_sink, step_metric_records)

            tprint(f"✅ Feature lookback optimization completed for {config.symbol}")

            # Validate artifacts before updating state
            if 'feature_lookback_optimization_result' in flo_result.artifacts:
                optimized_features = flo_result.artifacts.get('feature_lookback_optimization_result', {}).get('optimized_features', {})
                tprint(f"   → Features optimized: {len(optimized_features)}")
                results['results']['feature_lookback_optimization'] = flo_result.artifacts
                self._current_pipeline_state.update(flo_result.artifacts)
            else:
                tprint_warning("⚠️ Feature lookback optimization completed but artifact structure unexpected")
                results['results']['feature_lookback_optimization'] = flo_result.artifacts
                self._current_pipeline_state.update(flo_result.artifacts)

            # Step 3: Interactive Feature Generation
            tprint("🔧 Step 3: Interactive Feature Generation")
            self.logger.info('🔧 Step 3: Interactive Feature Generation')
            interactive_result = await self._execute_interactive_feature_generation(config)
            if interactive_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('interactive_feature_generation', interactive_result, results, metrics_sink, step_metric_records)
            if not interactive_result.success:
                tprint(f"❌ Interactive feature generation failed: {interactive_result.error_message}")
                self.logger.error(f'❌ Interactive feature generation failed: {interactive_result.error_message}')
                results['error_message'] = interactive_result.error_message
                return self._finalize_results(results, start_time, metrics_sink, step_metric_records)

            tprint(f"✅ Interactive feature generation completed for {config.symbol}")

            # Validate artifacts before updating state
            if 'interactive_feature_generation_result' in interactive_result.artifacts:
                features = interactive_result.artifacts.get('interactive_feature_generation_result', {}).get('features', {})
                tprint(f"   → Features generated: {len(features)}")
                results['results']['interactive_feature_generation'] = interactive_result.artifacts
                self._current_pipeline_state.update(interactive_result.artifacts)
            else:
                tprint_warning("⚠️ Interactive feature generation completed but artifact structure unexpected")
                results['results']['interactive_feature_generation'] = interactive_result.artifacts
                self._current_pipeline_state.update(interactive_result.artifacts)

            # Step 4: Final Feature Selection
            tprint("🎯 Step 4: Final Feature Selection")
            self.logger.info('🎯 Step 4: Final Feature Selection')
            ffs_result = await self._execute_final_feature_selection(config)
            if ffs_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('final_feature_selection', ffs_result, results, metrics_sink, step_metric_records)
            if not ffs_result.success:
                tprint(f"❌ Final feature selection failed: {ffs_result.error_message}")
                self.logger.error(f'❌ Final feature selection failed: {ffs_result.error_message}')
                results['error_message'] = ffs_result.error_message
                return self._finalize_results(results, start_time, metrics_sink, step_metric_records)

            tprint(f"✅ Final feature selection completed for {config.symbol}")

            # Validate artifacts before updating state
            if 'final_feature_selection_result' in ffs_result.artifacts:
                selected_features = ffs_result.artifacts.get('final_feature_selection_result', {}).get('selected_features', [])
                tprint(f"   → Final features: {len(selected_features)}")
                results['results']['final_feature_selection'] = ffs_result.artifacts
                self._current_pipeline_state.update(ffs_result.artifacts)
            else:
                tprint_warning("⚠️ Final feature selection completed but artifact structure unexpected")
                results['results']['final_feature_selection'] = ffs_result.artifacts
                self._current_pipeline_state.update(ffs_result.artifacts)

            # Success
            end_time = datetime.now()
            results['success'] = True
            results['execution_time'] = (end_time - start_time).total_seconds()
            results['completed_steps'] = 4

            tprint(f"🎉 Pre-Training Sub-Pipeline execution completed successfully for {config.symbol}")
            tprint(f"⏱️ Total execution time: {results['execution_time']:.2f} seconds")
            tprint(f"📊 All {results['completed_steps']} steps completed successfully")
            tprint(f"📋 Pipeline summary:")
            tprint(f"   🎯 Multi-horizon labeling: ✅ Complete")
            tprint(f"   ⚙️ Feature optimization: ✅ Complete")
            tprint(f"   🔧 Roadmap features: ✅ Complete")
            tprint(f"   🎯 Final selection: ✅ Complete")

            self.logger.info(f'🎉 Pre-Training Sub-Pipeline completed successfully in {results["execution_time"]:.2f}s')

        except ImportError as e:
            self.logger.error(f'❌ Pre-Training Sub-Pipeline failed due to missing dependencies: {e}')
            tprint_error(f"❌ Missing dependencies: {e}")
            results['error_message'] = f"Missing dependencies: {str(e)}"
        except FileNotFoundError as e:
            self.logger.error(f'❌ Pre-Training Sub-Pipeline failed due to missing files: {e}')
            tprint_error(f"❌ Missing files: {e}")
            results['error_message'] = f"Missing files: {str(e)}"
        except MemoryError as e:
            self.logger.error(f'❌ Pre-Training Sub-Pipeline failed due to memory issues: {e}')
            tprint_error(f"❌ Memory error: {e}")
            results['error_message'] = f"Memory error: {str(e)}"
        except Exception as e:
            self.logger.error(f'❌ Pre-Training Sub-Pipeline failed with unexpected error: {e}')
            tprint_error(f"❌ Unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")
            results['error_message'] = f"Unexpected error: {str(e)}"

        return self._finalize_results(results, start_time, metrics_sink, step_metric_records, end_time if results.get('success') else None)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _create_metrics_sink(self, config: SubPipelineConfig) -> Optional[MetricsSink]:
        output_path: Optional[Path] = None
        if config.metrics_output_path:
            output_path = Path(config.metrics_output_path)
        elif config.metrics_output_path is None:
            extension = 'jsonl' if config.metrics_output_format.lower() == 'jsonl' else 'csv'
            output_path = Path('artifacts') / f'pre_training_metrics.{extension}'

        if output_path is None and not config.metrics_prometheus_enabled:
            return None

        if output_path is None:
            output_path = Path('artifacts') / f'pre_training_metrics.{config.metrics_output_format.lower()}'

        sink_config = MetricsSinkConfig(
            output_path=output_path,
            output_format=config.metrics_output_format,
            enable_prometheus=config.metrics_prometheus_enabled,
        )
        return MetricsSink(sink_config)

    def _record_step_metrics(
        self,
        step_name: str,
        result: SubPipelineResult,
        pipeline_results: Dict[str, Any],
        metrics_sink: Optional[MetricsSink],
        step_metric_records: List[Dict[str, Any]],
    ) -> None:
        if metrics_sink is None:
            return

        record = self._base_metrics_record()
        row_counts = self._extract_row_counts(result.artifacts)
        row_count_total = sum(row_counts.values()) if row_counts else 0
        label_skew = self._compute_label_distribution_skew(result.metadata)
        memory_peak_mb = self._get_memory_usage_mb()

        record.update({
            'record_type': 'step',
            'step_name': step_name,
            'status': result.status.value,
            'success': result.success,
            'duration_seconds': result.duration_seconds,
            'row_count_total': row_count_total,
            'row_count_details': json.dumps(row_counts, sort_keys=True),
            'memory_peak_mb': memory_peak_mb,
            'label_distribution_skew': label_skew,
            'timestamp': datetime.utcnow().isoformat(),
            'artifact_count': len(result.artifacts),
            'metadata_keys': ','.join(sorted(result.metadata.keys())) if result.metadata else '',
            'total_steps': pipeline_results.get('total_steps'),
            'completed_steps': pipeline_results.get('completed_steps'),
            'total_row_count': row_count_total,
            'max_memory_peak_mb': memory_peak_mb,
            'average_label_distribution_skew': label_skew,
            'error_message': result.error_message or '',
        })

        step_metric_records.append(record)
        metrics_sink.write(record)

    def _emit_pipeline_metrics(
        self,
        metrics_sink: MetricsSink,
        step_metric_records: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> None:
        total_row_count = sum(record.get('row_count_total') or 0 for record in step_metric_records)
        max_memory_peak = max(
            (record.get('memory_peak_mb') for record in step_metric_records if record.get('memory_peak_mb') is not None),
            default=None,
        )
        label_skew_values = [
            record.get('label_distribution_skew')
            for record in step_metric_records
            if record.get('label_distribution_skew') is not None
        ]
        average_label_skew = (sum(label_skew_values) / len(label_skew_values)) if label_skew_values else None
        artifact_count = sum(record.get('artifact_count') or 0 for record in step_metric_records)
        row_detail_map = {
            record['step_name']: record.get('row_count_total', 0)
            for record in step_metric_records
            if record.get('step_name')
        }
        metadata_keys = sorted({
            key
            for record in step_metric_records
            for key in (record.get('metadata_keys', '') or '').split(',')
            if key
        })

        pipeline_record = self._base_metrics_record()
        pipeline_record.update({
            'record_type': 'pipeline',
            'step_name': 'pipeline_total',
            'status': 'completed' if results.get('success') else 'failed',
            'success': results.get('success', False),
            'duration_seconds': results.get('execution_time'),
            'row_count_total': total_row_count,
            'row_count_details': json.dumps(row_detail_map, sort_keys=True),
            'memory_peak_mb': max_memory_peak,
            'label_distribution_skew': average_label_skew,
            'timestamp': datetime.utcnow().isoformat(),
            'artifact_count': artifact_count,
            'metadata_keys': ','.join(metadata_keys),
            'total_steps': results.get('total_steps'),
            'completed_steps': results.get('completed_steps'),
            'total_row_count': total_row_count,
            'max_memory_peak_mb': max_memory_peak,
            'average_label_distribution_skew': average_label_skew,
            'error_message': results.get('error_message') or '',
        })

        metrics_sink.write(pipeline_record)

    def _finalize_results(
        self,
        results: Dict[str, Any],
        start_time: datetime,
        metrics_sink: Optional[MetricsSink],
        step_metric_records: List[Dict[str, Any]],
        end_time: Optional[datetime] = None,
    ) -> PipelineResultDict:
        end_time = end_time or datetime.now()
        results['execution_time'] = (end_time - start_time).total_seconds()
        if metrics_sink is not None:
            self._emit_pipeline_metrics(metrics_sink, step_metric_records, results)
        return results

    @staticmethod
    def _base_metrics_record() -> Dict[str, Any]:
        fields = [
            'record_type',
            'step_name',
            'status',
            'success',
            'duration_seconds',
            'row_count_total',
            'row_count_details',
            'memory_peak_mb',
            'label_distribution_skew',
            'timestamp',
            'artifact_count',
            'metadata_keys',
            'total_steps',
            'completed_steps',
            'total_row_count',
            'max_memory_peak_mb',
            'average_label_distribution_skew',
            'error_message',
        ]
        return {field: None for field in fields}

    @staticmethod
    def _extract_row_counts(artifacts: Dict[str, Any]) -> Dict[str, int]:
        row_counts: Dict[str, int] = {}

        def _walk(prefix: str, value: Any) -> None:
            key_name = prefix or 'root'
            if isinstance(value, pd.DataFrame):
                row_counts[key_name] = int(value.shape[0])
            elif isinstance(value, pd.Series):
                row_counts[key_name] = int(value.shape[0])
            elif isinstance(value, np.ndarray):
                row_counts[key_name] = int(value.shape[0])
            elif isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = f"{key_name}.{key}" if prefix else str(key)
                    _walk(next_prefix, nested_value)
            elif isinstance(value, (list, tuple)):
                for index, nested_value in enumerate(value):
                    next_prefix = f"{key_name}[{index}]"
                    _walk(next_prefix, nested_value)

        for key, value in artifacts.items():
            _walk(key, value)

        return row_counts

    @staticmethod
    def _compute_label_distribution_skew(metadata: Dict[str, Any]) -> Optional[float]:
        if not metadata:
            return None

        label_distribution = metadata.get('label_distribution')
        if not isinstance(label_distribution, dict):
            return None

        values: List[float] = []

        def _collect_values(data: Any) -> None:
            if isinstance(data, dict):
                for nested in data.values():
                    _collect_values(nested)
            elif isinstance(data, (int, float)):
                values.append(float(data))

        _collect_values(label_distribution)

        if not values:
            return None

        total = sum(values)
        if total > 0:
            normalized = [value / total for value in values]
        else:
            normalized = values

        return max(normalized) - min(normalized) if normalized else None

    @staticmethod
    def _get_memory_usage_mb() -> Optional[float]:
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = getattr(usage, 'ru_maxrss', None)
        if max_rss is None:
            return None
        # On Linux ru_maxrss is reported in kilobytes.
        return max_rss / 1024.0

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pre-training pipeline with backward compatible interface.

        Args:
            training_input: Input data for the pipeline
            pipeline_state: Current pipeline state

        Returns:
            Dictionary containing execution results
        """
        # Extract configuration from pipeline state
        config = SubPipelineConfig(
            symbol=pipeline_state.get('symbol', 'ETHUSDT'),
            exchange=pipeline_state.get('exchange', 'binance'),
            timeframe=pipeline_state.get('timeframe', '1h'),  # Default 1h for pre-training (analyst)
            data_dir=pipeline_state.get('data_dir', 'historical_data'),
            mode=ExecutionMode.FULL,  # Default to full mode
            custom_params=pipeline_state.get('custom_params', {})
        )

        # Execute the pipeline
        return await self.execute_pipeline(config)

    def _prepare_component_pipeline_state(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Construct the pipeline state passed to individual components."""
        pipeline_state: Dict[str, Any] = {
            'symbol': config.symbol,
            'exchange': config.exchange,
            'timeframe': config.timeframe,
            'data_dir': config.data_dir,
            'custom_params': self._build_component_custom_params(config),
            'quality_thresholds': self._get_quality_thresholds(config),
        }

        regime_cache_path = config.custom_params.get('regime_cache_path') if config.custom_params else None
        if not regime_cache_path:
            data_cache_dir = config.custom_params.get('data_cache_dir') if config.custom_params else None
            if data_cache_dir:
                regime_cache_path = str((Path(data_cache_dir).expanduser() / 'nas_tas_clustering').resolve(strict=False))

        if regime_cache_path:
            pipeline_state['regime_cache_path'] = regime_cache_path

        regime_split = config.custom_params.get('regime_data_splitting_result')
        if regime_split is None:
            regime_split = self._current_pipeline_state.get('regime_data_splitting_result')

        if regime_split is not None:
            pipeline_state['regime_data_splitting_result'] = regime_split
            self._current_pipeline_state['regime_data_splitting_result'] = regime_split

        return pipeline_state

    def _get_quality_thresholds(self, config: SubPipelineConfig) -> Dict[str, float]:
        """Return the data quality thresholds configured for the pipeline."""
        return {
            'label_imbalance': float(config.label_imbalance_warning_threshold),
            'nan_rate': float(config.nan_rate_warning_threshold),
            'duplicate_index': float(config.duplicate_index_warning_threshold),
        }

    def _build_component_custom_params(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Augment component custom parameters with quality thresholds."""
        params = dict(config.custom_params or {})
        params.setdefault('quality_thresholds', self._get_quality_thresholds(config))
        return params

    def _extend_with_quality_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        alerts: List[str],
        config: SubPipelineConfig,
    ) -> Dict[str, Any]:
        """Merge computed quality metrics and alerts into metadata."""
        merged_metadata: Dict[str, Any] = dict(metadata or {})
        if metrics:
            merged_metadata['quality_metrics'] = metrics
        if alerts:
            merged_metadata['quality_alerts'] = alerts
        merged_metadata.setdefault('quality_thresholds', self._get_quality_thresholds(config))
        return merged_metadata

    def _analyze_component_quality(
        self,
        component_name: str,
        artifacts: Dict[str, Any],
        config: SubPipelineConfig,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Inspect component artifacts and compute quality metrics and alerts."""
        thresholds = self._get_quality_thresholds(config)
        metrics: Dict[str, Any] = {}
        alerts: List[str] = []
        visited_frames: Dict[int, Dict[str, Any]] = {}

        def log_warning(message: str) -> None:
            alerts.append(message)
            tprint_warning(message)
            self.logger.warning(message)

        def handle_dataframe(dataset_name: str, df: pd.DataFrame) -> None:
            if df is None or df.empty:
                return
            df_id = id(df)
            if df_id in visited_frames:
                metrics[dataset_name] = visited_frames[df_id]
                return

            dataset_metrics, dataset_alerts = self._compute_dataframe_quality_metrics(
                component_name,
                dataset_name,
                df,
                thresholds,
            )
            visited_frames[df_id] = dataset_metrics
            metrics[dataset_name] = dataset_metrics
            for alert in dataset_alerts:
                log_warning(alert)

        def traverse(prefix: str, value: Any) -> None:
            if isinstance(value, pd.DataFrame):
                handle_dataframe(prefix, value)
            elif isinstance(value, dict):
                for key, nested_value in value.items():
                    nested_prefix = f"{prefix}.{key}" if prefix else key
                    traverse(nested_prefix, nested_value)

        traverse('', artifacts)
        return metrics, alerts

    def _compute_dataframe_quality_metrics(
        self,
        component_name: str,
        dataset_name: str,
        df: pd.DataFrame,
        thresholds: Dict[str, float],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Compute quality statistics for a DataFrame and build alert messages."""
        dataset_metrics: Dict[str, Any] = {}
        alerts: List[str] = []

        total_cells = int(df.size)
        nan_rate = float(df.isna().sum().sum() / total_cells) if total_cells else 0.0
        dataset_metrics['nan_rate'] = nan_rate
        if nan_rate >= thresholds['nan_rate'] > 0:
            alerts.append(
                f"⚠️ [{component_name}] {dataset_name} NaN rate {nan_rate:.2%} exceeds threshold {thresholds['nan_rate']:.2%}"
            )

        duplicate_share = 0.0
        if len(df.index) > 0:
            duplicate_mask = df.index.duplicated()
            duplicate_share = float(duplicate_mask.mean()) if duplicate_mask.any() else 0.0
        dataset_metrics['duplicate_index_share'] = duplicate_share
        if duplicate_share > thresholds['duplicate_index'] > 0:
            alerts.append(
                f"⚠️ [{component_name}] {dataset_name} duplicate index share {duplicate_share:.2%} exceeds threshold {thresholds['duplicate_index']:.2%}"
            )

        column_metrics: Dict[str, Any] = {}
        max_dominant_share = 0.0
        max_dominant_column: Optional[str] = None
        for column in df.columns:
            series = df[column].dropna()
            unique_count = series.nunique(dropna=True)
            if unique_count == 0 or unique_count > 20:
                continue
            counts = series.value_counts(dropna=True, normalize=True)
            if counts.empty:
                continue
            dominant_value = counts.index[0]
            dominant_share = float(counts.iloc[0])
            column_metrics[str(column)] = {
                'dominant_value': str(dominant_value),
                'dominant_share': dominant_share,
                'distribution': {str(k): float(v) for k, v in counts.items()},
            }
            if dominant_share > max_dominant_share:
                max_dominant_share = dominant_share
                max_dominant_column = str(column)
            if dominant_share >= thresholds['label_imbalance'] > 0:
                alerts.append(
                    f"⚠️ [{component_name}] {dataset_name}.{column} dominant label share {dominant_share:.2%} exceeds threshold {thresholds['label_imbalance']:.2%}"
                )

        if column_metrics:
            dataset_metrics['label_balance'] = {
                'columns': column_metrics,
                'max_dominant_share': max_dominant_share,
                'max_dominant_column': max_dominant_column,
            }

        return dataset_metrics, alerts

    async def _execute_multi_horizon_profit_labeler(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute multi-horizon profit labeler with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='multi_horizon_profit_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            custom_params = config.custom_params or {}
            precomputed_result = custom_params.get('precomputed_labeling_result')

            if precomputed_result:
                tprint('📥 Using precomputed entry labeling result for tactician pipeline')
                result.status = SubPipelineStatus.COMPLETED
                result.success = True
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                result.artifacts = precomputed_result
                base_metadata = {
                    'component_type': 'multi_horizon_profit_labeler',
                    'source': 'precomputed',
                    'labeling_method': precomputed_result.get('multi_horizon_labeling_result', {}).get('method', 'tactician_entry_labeling')
                }
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'multi_horizon_profit_labeler',
                    precomputed_result,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    base_metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
                return result

            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('multi_horizon_profit_labeler', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'multi_horizon_profit_labeler',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
            else:
                result.metadata = component_result.metadata or {
                    'component_type': 'multi_horizon_profit_labeler'
                }

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Multi-horizon profit labeler failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Multi-horizon profit labeler failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Multi-horizon profit labeler failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        return result

    async def _execute_feature_lookback_optimization(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute feature lookback optimization with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_lookback_optimization',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('feature_lookback_optimization', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'feature_lookback_optimization',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
            else:
                result.metadata = component_result.metadata or {}

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Feature lookback optimization failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Feature lookback optimization failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Feature lookback optimization failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        return result

    async def _execute_interactive_feature_generation(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute interactive feature generation with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='interactive_feature_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Import the new interactive feature generation component
            try:
                from .interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
                    create_interactive_feature_generation_component, InteractiveFeatureGenerationConfig
                )
                tprint("🔧 Using optimized interactive feature generation component")
            except ImportError as import_error:
                tprint_error(f"❌ Required component not found: {import_error}")
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Missing interactive feature generation component: {str(import_error)}"
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                return result
            
            # Create component configuration
            component_config = InteractiveFeatureGenerationConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                feature_budget_pre=config.custom_params.get('feature_budget_pre', 120),
                feature_budget_post=config.custom_params.get('feature_budget_post', (30, 60)),
                interactions_cap=config.custom_params.get('interactions_cap', 15),
                enable_matrix_optimization=config.custom_params.get('enable_matrix_optimization', True),
                enable_hardware_optimization=config.custom_params.get('enable_hardware_optimization', True),
                enable_parallel_processing=config.parallel_processing,
                max_workers=config.max_workers,
                verbose_logging=config.custom_params.get('verbose_logging', True)
            )

            # Create component
            component = create_interactive_feature_generation_component(component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.output_files = component_result.output_files
            result.error_message = component_result.error_message
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'interactive_feature_generation',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
            else:
                result.metadata = component_result.metadata or {}

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Interactive feature generation failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Interactive feature generation failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Interactive feature generation failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        return result

    async def _execute_optimized_lookback_generation(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute optimized lookback generation with matrix operations and hardware acceleration."""
        result = SubPipelineResult(
            sub_pipeline_name='optimized_lookback_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('optimized_lookback_generation', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'optimized_lookback_generation',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
            else:
                result.metadata = component_result.metadata or {}

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Optimized lookback generation failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Optimized lookback generation failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Optimized lookback generation failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        return result

    async def _execute_final_feature_selection(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute final feature selection with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='final_feature_selection',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('final_feature_selection', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'final_feature_selection',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
            else:
                result.metadata = component_result.metadata

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Final feature selection failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Final feature selection failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Final feature selection failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        return result

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines for pre-training stage."""
        return [
            'multi_horizon_profit_labeler',
            'feature_lookback_optimization', 
            'interactive_feature_generation',
            'final_feature_selection'
        ]

    async def execute_sub_pipeline(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline."""
        if sub_pipeline_name == 'multi_horizon_profit_labeler':
            return await self._execute_multi_horizon_profit_labeler(config)
        elif sub_pipeline_name == 'feature_lookback_optimization':
            return await self._execute_feature_lookback_optimization(config)
        elif sub_pipeline_name == 'optimized_lookback_generation':
            return await self._execute_optimized_lookback_generation(config)
        elif sub_pipeline_name == 'interactive_feature_generation':
            return await self._execute_interactive_feature_generation(config)
        elif sub_pipeline_name == 'final_feature_selection':
            return await self._execute_final_feature_selection(config)
        else:
            tprint_error(f"❌ Unknown sub-pipeline requested: {sub_pipeline_name}")
            tprint(f"📋 Available sub-pipelines: {self.get_available_sub_pipelines()}")
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")

    async def execute_sub_pipeline_with_next(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline and automatically trigger subsequent sub-pipelines."""
        # For pre-training, we execute all 4 steps in sequence
        available_steps = self.get_available_sub_pipelines()
        
        try:
            start_index = available_steps.index(sub_pipeline_name)
        except ValueError:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
        
        # Execute all steps starting from the specified one
        for i in range(start_index, len(available_steps)):
            step_name = available_steps[i]
            self.logger.info(f"🚀 Executing pre-training step: {step_name}")
            
            result = await self.execute_sub_pipeline(step_name, config)
            self.results.append(result)
            
            # If this step failed, stop the sequence
            if not result.success:
                self.logger.error(f"❌ Step {step_name} failed, stopping execution sequence")
                break
        
        # Return the first result (the one that was requested)
        return self.results[0] if self.results else None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with all results."""
        return {
            'total_sub_pipelines': len(self.results),
            'successful_sub_pipelines': len([r for r in self.results if r.success]),
            'failed_sub_pipelines': len([r for r in self.results if not r.success]),
            'total_execution_time': sum(r.duration_seconds for r in self.results),
            'sub_pipeline_results': [
                {
                    'name': r.sub_pipeline_name,
                    'status': r.status.value,
                    'success': r.success,
                    'execution_time': r.duration_seconds,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }

# Convenience function for direct execution
async def execute_pre_training_pipeline(config: SubPipelineConfig) -> Dict[str, Any]:
    """
    Execute the pre-training pipeline with the given configuration.

    Args:
        config: Configuration for pipeline execution

    Returns:
        Dictionary containing execution results
    """
    pipeline = PreTrainingSubPipeline()
    return await pipeline.execute_pipeline(config)