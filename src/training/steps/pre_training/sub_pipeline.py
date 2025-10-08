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
import hashlib
import json
import socket
import subprocess
import pandas as pd
import numpy as np


class PipelineResultDict(TypedDict, total=False):
    """Type definition for pipeline execution results."""
    success: bool
    execution_time: float
    total_steps: int
    completed_steps: int
    results: Dict[str, Any]
    error_message: Optional[str]
    run_metadata: Dict[str, Any]

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_error, tprint_warning

# Import component system
from .components import ComponentFactory, ComponentConfig

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

    def _initialize_run_metadata(self, config: SubPipelineConfig) -> Tuple[Dict[str, Any], datetime]:
        """Collect reproducibility metadata for the pipeline run."""
        start_time = datetime.utcnow()
        run_metadata = {
            'git_sha': self._get_git_sha(),
            'config_hash': self._compute_config_hash(config),
            'data_snapshot_id': self._extract_data_snapshot_id(config),
            'rng_seed': self._extract_rng_seed(config),
            'host_name': self._get_host_name(),
            'start_timestamp': start_time.isoformat() + 'Z',
            'end_timestamp': None,
        }
        return run_metadata, start_time

    def _get_git_sha(self) -> str:
        """Return the current git SHA for the repository."""
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        except Exception:
            return 'unknown'

    def _get_host_name(self) -> str:
        """Return the host name for the current machine."""
        try:
            return socket.gethostname()
        except Exception:
            return 'unknown-host'

    def _safe_serialize_for_hash(self, value: Any) -> Any:
        """Safely serialize nested objects for hashing."""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(k): self._safe_serialize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple, set)):
            return [self._safe_serialize_for_hash(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _compute_config_hash(self, config: SubPipelineConfig) -> str:
        """Compute a stable hash for the pipeline configuration."""
        config_payload = {
            'mode': config.mode.value,
            'symbol': config.symbol,
            'exchange': config.exchange,
            'timeframe': config.timeframe,
            'data_dir': config.data_dir,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'force_rerun': config.force_rerun,
            'parallel_processing': config.parallel_processing,
            'max_workers': config.max_workers,
            'validation_enabled': config.validation_enabled,
            'monitoring_enabled': config.monitoring_enabled,
            'fast_mode': config.fast_mode,
            'skip_next_pipeline': config.skip_next_pipeline,
            'custom_params': self._safe_serialize_for_hash(config.custom_params or {}),
        }
        try:
            config_json = json.dumps(config_payload, sort_keys=True)
            return hashlib.sha256(config_json.encode()).hexdigest()
        except Exception:
            return 'unknown-config-hash'

    def _extract_data_snapshot_id(self, config: SubPipelineConfig) -> Optional[Any]:
        """Extract data snapshot identifier from configuration."""
        params = config.custom_params or {}
        for key in ('data_snapshot_id', 'data_snapshot', 'snapshot_id'):
            if key in params:
                return params.get(key)
        return None

    def _extract_rng_seed(self, config: SubPipelineConfig) -> Optional[Any]:
        """Extract RNG seed from configuration."""
        params = config.custom_params or {}
        for key in ('rng_seed', 'seed', 'random_seed'):
            if key in params:
                return params.get(key)
        return None

    def _log_run_metadata_block(self, run_metadata: Dict[str, Any], *, phase: str, duration_seconds: Optional[float] = None) -> None:
        """Log the reproducibility metadata block to both logger and tprint."""
        header = f"🧾 Run metadata snapshot ({phase})"
        tprint(header)
        self.logger.info(header)

        fields = [
            ('git_sha', run_metadata.get('git_sha')),
            ('config_hash', run_metadata.get('config_hash')),
            ('data_snapshot_id', run_metadata.get('data_snapshot_id')),
            ('rng_seed', run_metadata.get('rng_seed')),
            ('host_name', run_metadata.get('host_name')),
            ('start_timestamp', run_metadata.get('start_timestamp')),
            ('end_timestamp', run_metadata.get('end_timestamp')),
        ]

        if duration_seconds is not None:
            fields.append(('duration_seconds', f"{duration_seconds:.2f}"))

        for label, value in fields:
            line = f"   • {label}: {value}"
            tprint(line)
            self.logger.info(line)

    def _merge_run_metadata(self, metadata: Optional[Dict[str, Any]], run_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Merge run metadata into an existing metadata mapping."""
        merged_metadata: Dict[str, Any] = dict(metadata or {})
        existing_run_metadata = {}
        if isinstance(merged_metadata.get('run_metadata'), dict):
            existing_run_metadata = dict(merged_metadata.get('run_metadata'))
        merged_metadata['run_metadata'] = {**run_metadata, **existing_run_metadata}
        return merged_metadata

    def _prepare_component_for_execution(self, component: Any, run_metadata: Dict[str, Any]) -> None:
        """Attach run metadata to a component if supported."""
        try:
            if hasattr(component, 'set_run_metadata') and callable(getattr(component, 'set_run_metadata')):
                component.set_run_metadata(run_metadata)
            else:
                setattr(component, 'run_metadata', run_metadata)
        except Exception:
            # Components that cannot accept run metadata continue without interruption.
            pass

    def _finalize_pipeline_results(
        self,
        results: PipelineResultDict,
        run_metadata: Dict[str, Any],
        pipeline_start_time: datetime,
    ) -> PipelineResultDict:
        """Finalize pipeline results with reproducibility metadata and timing."""
        end_time = datetime.utcnow()
        duration_seconds = (end_time - pipeline_start_time).total_seconds()
        run_metadata['end_timestamp'] = end_time.isoformat() + 'Z'
        run_metadata['duration_seconds'] = duration_seconds

        if not results.get('execution_time'):
            results['execution_time'] = duration_seconds

        results['run_metadata'] = dict(run_metadata)

        self._log_run_metadata_block(run_metadata, phase="end", duration_seconds=duration_seconds)

        return results

    async def execute_pipeline(self, config: SubPipelineConfig) -> PipelineResultDict:
        """
        Execute the complete pre-training pipeline.

        Args:
            config: Configuration for pipeline execution

        Returns:
            PipelineResultDict containing execution results with typed fields
        """
        run_metadata, pipeline_start_time = self._initialize_run_metadata(config)

        self._log_run_metadata_block(run_metadata, phase="start")

        self.logger.info('🚀 Starting Pre-Training Sub-Pipeline execution')
        self.logger.info(f'📊 Symbol: {config.symbol}, Exchange: {config.exchange}')
        self.logger.info(f'⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}')

        start_time = pipeline_start_time
        tprint(f"🚀 Starting Pre-Training Sub-Pipeline execution for {config.symbol} on {config.exchange}")
        tprint(f"⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}")
        tprint(f"📊 Configuration: force_rerun={config.force_rerun}, parallel={config.parallel_processing}")

        results = {
            'success': False,
            'execution_time': 0.0,
            'total_steps': 4,
            'completed_steps': 0,
            'results': {},
            'run_metadata': run_metadata,
        }

        try:
            # Step 1: Multi-Horizon Profit Labeler
            tprint("🎯 Step 1: Multi-Horizon Profit Labeler")
            self.logger.info('🎯 Step 1: Multi-Horizon Profit Labeler')
            mh_result = await self._execute_multi_horizon_profit_labeler(config, run_metadata)
            if not mh_result.success:
                tprint(f"❌ Multi-horizon profit labeling failed: {mh_result.error_message}")
                self.logger.error(f'❌ Multi-horizon profit labeling failed: {mh_result.error_message}')
                return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)

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
                    return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)
            else:
                tprint_error("❌ Multi-horizon labeling artifact validation failed: missing 'multi_horizon_labeling_result'")
                return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)

            # Step 2: Feature Lookback Optimization
            tprint("⚙️ Step 2: Feature Lookback Optimization")
            self.logger.info('⚙️ Step 2: Feature Lookback Optimization')
            flo_result = await self._execute_feature_lookback_optimization(config, run_metadata)
            if not flo_result.success:
                tprint(f"❌ Feature lookback optimization failed: {flo_result.error_message}")
                self.logger.error(f'❌ Feature lookback optimization failed: {flo_result.error_message}')
                return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)

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
            interactive_result = await self._execute_interactive_feature_generation(config, run_metadata)
            if not interactive_result.success:
                tprint(f"❌ Interactive feature generation failed: {interactive_result.error_message}")
                self.logger.error(f'❌ Interactive feature generation failed: {interactive_result.error_message}')
                return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)

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
            ffs_result = await self._execute_final_feature_selection(config, run_metadata)
            if not ffs_result.success:
                tprint(f"❌ Final feature selection failed: {ffs_result.error_message}")
                self.logger.error(f'❌ Final feature selection failed: {ffs_result.error_message}')
                return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)

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
            end_time = datetime.utcnow()
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

        return self._finalize_pipeline_results(results, run_metadata, pipeline_start_time)

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
            'custom_params': config.custom_params,
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

    async def _execute_multi_horizon_profit_labeler(
        self,
        config: SubPipelineConfig,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> SubPipelineResult:
        """Execute multi-horizon profit labeler with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='multi_horizon_profit_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=self._merge_run_metadata({}, run_metadata or {})
        )

        try:
            custom_params = config.custom_params or {}
            precomputed_result = custom_params.get('precomputed_labeling_result')

            if precomputed_result:
                tprint('📥 Using precomputed entry labeling result for tactician pipeline')
                result.status = SubPipelineStatus.COMPLETED
                result.success = True
                result.end_time = datetime.utcnow()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                result.artifacts = precomputed_result
                result.metadata.update({
                    'component_type': 'multi_horizon_profit_labeler',
                    'source': 'precomputed',
                    'labeling_method': precomputed_result.get('multi_horizon_labeling_result', {}).get('method', 'tactician_entry_labeling')
                })
                return result

            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('multi_horizon_profit_labeler', component_config)
            self._prepare_component_for_execution(component, run_metadata or {})

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            component_result.metadata = self._merge_run_metadata(component_result.metadata, run_metadata or {})
            result.metadata = dict(component_result.metadata)

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Multi-horizon profit labeler failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Multi-horizon profit labeler failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Multi-horizon profit labeler failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        result.metadata = self._merge_run_metadata(result.metadata, run_metadata or {})
        return result

    async def _execute_feature_lookback_optimization(
        self,
        config: SubPipelineConfig,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> SubPipelineResult:
        """Execute feature lookback optimization with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_lookback_optimization',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=self._merge_run_metadata({}, run_metadata or {})
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('feature_lookback_optimization', component_config)
            self._prepare_component_for_execution(component, run_metadata or {})

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            component_result.metadata = self._merge_run_metadata(component_result.metadata, run_metadata or {})
            result.metadata = dict(component_result.metadata)

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Feature lookback optimization failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Feature lookback optimization failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Feature lookback optimization failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        result.metadata = self._merge_run_metadata(result.metadata, run_metadata or {})
        return result

    async def _execute_interactive_feature_generation(
        self,
        config: SubPipelineConfig,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> SubPipelineResult:
        """Execute interactive feature generation with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='interactive_feature_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=self._merge_run_metadata({}, run_metadata or {})
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
            self._prepare_component_for_execution(component, run_metadata or {})

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.output_files = component_result.output_files
            component_result.metadata = self._merge_run_metadata(getattr(component_result, 'metadata', {}), run_metadata or {})
            result.metadata = dict(component_result.metadata)
            result.error_message = component_result.error_message

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Interactive feature generation failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Interactive feature generation failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Interactive feature generation failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        result.metadata = self._merge_run_metadata(result.metadata, run_metadata or {})
        return result

    async def _execute_optimized_lookback_generation(
        self,
        config: SubPipelineConfig,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> SubPipelineResult:
        """Execute optimized lookback generation with matrix operations and hardware acceleration."""
        result = SubPipelineResult(
            sub_pipeline_name='optimized_lookback_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=self._merge_run_metadata({}, run_metadata or {})
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('optimized_lookback_generation', component_config)
            self._prepare_component_for_execution(component, run_metadata or {})

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            component_result.metadata = self._merge_run_metadata(component_result.metadata, run_metadata or {})
            result.metadata = dict(component_result.metadata)

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Optimized lookback generation failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Optimized lookback generation failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Optimized lookback generation failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        result.metadata = self._merge_run_metadata(result.metadata, run_metadata or {})
        return result

    async def _execute_final_feature_selection(
        self,
        config: SubPipelineConfig,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> SubPipelineResult:
        """Execute final feature selection with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='final_feature_selection',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=self._merge_run_metadata({}, run_metadata or {})
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('final_feature_selection', component_config)
            self._prepare_component_for_execution(component, run_metadata or {})

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            component_result.metadata = self._merge_run_metadata(component_result.metadata, run_metadata or {})
            result.metadata = dict(component_result.metadata)

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Final feature selection failed - missing dependencies: {e}")
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Final feature selection failed - missing files: {e}")
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Final feature selection failed with unexpected error: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

        result.metadata = self._merge_run_metadata(result.metadata, run_metadata or {})
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