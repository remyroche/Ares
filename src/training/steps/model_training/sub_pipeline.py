"""
Model Training Sub-Pipeline - Orchestration of Analyst and Tactician Training

This module orchestrates the complete model training pipeline with distinct
workflows for Analyst and Tactician models:

ANALYST PIPELINE (60m timeframe - IF we trade):
1. analyst_pre_ml_orchestration - Feature engineering on 60m data
2. analyst_models_training - Train base models (per-regime)
3. analyst_ensemble_training - Train ensemble models

TACTICIAN PIPELINE (15m timeframe - WHEN we trade):
4. tactician_pre_ml_orchestration - Feature engineering on 15m data (includes Analyst outputs as features)
5. tactician_models_training - Train base models (includes Analyst predictions as features)
6. tactician_ensemble_training - Train ensemble models

Each model type (short/long) is trained separately.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_success,
    tprint_error,
    tprint_warning,
)

try:
    from src.utils.common_operations import (
        ensure_directory,
        safe_read_parquet,
        safe_to_parquet,
        safe_json_dump,
        cleanup_m1_optimizers,
        get_current_datetime
    )
    COMMON_IO_AVAILABLE = True
except ImportError:
    COMMON_IO_AVAILABLE = False
    ensure_directory = None
    safe_read_parquet = None
    safe_to_parquet = None
    safe_json_dump = None
    cleanup_m1_optimizers = None
    get_current_datetime = None

# Import orchestration and training steps
try:
    from .analyst_pre_ml_orchestration import (
        AnalystPreMLOrchestrator, AnalystPreMLConfig, AnalystPreMLResult
    )
    ANALYST_PRE_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_pre_ml_orchestration not available: {e}")
    ANALYST_PRE_ML_AVAILABLE = False

# Per-regime training integration is no longer used
PER_REGIME_TRAINING_AVAILABLE = False
PerRegimeTrainingIntegration = None
PerRegimeTrainingResult = None
get_per_regime_integration = None
train_analyst_per_regime_models = None
train_tactician_per_regime_models = None
get_model_selector_for_trading = None

try:
    from .analyst_training_pipeline import (
        AnalystTrainingPipeline, AnalystTrainingPipelineConfig, AnalystTrainingPipelineResult
    )
    ANALYST_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_training_pipeline not available: {e}")
    ANALYST_TRAINING_AVAILABLE = False

try:
    from .analyst_ensemble_training import (
        AnalystEnsembleTrainingStep,
        AnalystEnsembleTrainingConfig,
    )
    ANALYST_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_ensemble_training not available: {e}")
    ANALYST_ENSEMBLE_AVAILABLE = False

try:
    from .tactician_pre_ml_orchestration import (
        TacticianPreMLOrchestrator, TacticianPreMLConfig, TacticianPreMLResult
    )
    TACTICIAN_PRE_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: tactician_pre_ml_orchestration not available: {e}")
    TACTICIAN_PRE_ML_AVAILABLE = False

try:
    from .tactician_training_pipeline import (
        TacticianTrainingPipeline, TacticianTrainingPipelineConfig, TacticianTrainingPipelineResult
    )
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: tactician_training_pipeline not available: {e}")
    TACTICIAN_TRAINING_AVAILABLE = False

try:
    from .tactician_ensemble_training import (
        TacticianEnsembleTrainingStep,
        TacticianEnsembleTrainingConfig,
    )
    TACTICIAN_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: tactician_ensemble_training not available: {e}")
    TACTICIAN_ENSEMBLE_AVAILABLE = False

logger = system_logger.getChild('ModelTrainingSubPipeline')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"
    LIGHT = "light"
    BLANK = "blank"

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class SubPipelineConfig:
    """Configuration for model training sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    analyst_timeframe: str = "60m"  # Analyst uses 60m
    tactician_timeframe: str = "15m"  # Tactician uses 15m
    timeframe: Optional[str] = None  # Generic timeframe (used if analyst_timeframe/tactician_timeframe not explicitly set)
    data_dir: str = "historical_data"
    start_date: Optional[str] = None  # Optional date filtering
    end_date: Optional[str] = None    # Optional date filtering

    # Training configuration
    train_analyst: bool = True
    train_tactician: bool = True
    train_short_models: bool = True
    train_long_models: bool = True

    # Execution parameters
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True  # Hardware monitoring
    single_stage_only: bool = False   # Pipeline chaining control

    # Output configuration
    output_directory: str = "generated/model_training"
    save_models: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # Direction control for trading (inherited from main pipeline config)
    enable_long_positions: bool = True
    enable_short_positions: bool = True

    # Additional compatibility parameters
    use_existing_data: bool = False  # Use pre-existing data artifacts
    logging: Optional[Any] = None     # Logging configuration (optional)

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

class ModelTrainingSubPipeline:
    """
    Model Training Sub-Pipeline.

    Orchestrates the complete training workflow for both Analyst and Tactician models
    with proper timeframe separation and data filtering.
    """

    def __init__(self):
        """Initialize the model training sub-pipeline."""
        self.logger = logger.getChild('ModelTrainingSubPipeline')
        self.results: List[SubPipelineResult] = []
        self._current_pipeline_state: Dict[str, Any] = {}

        # Initialize orchestrators
        if ANALYST_PRE_ML_AVAILABLE:
            from .analyst_pre_ml_orchestration import AnalystPreMLConfig
            config = AnalystPreMLConfig()
            self.analyst_pre_ml = AnalystPreMLOrchestrator(config)
        else:
            self.analyst_pre_ml = None

        if ANALYST_TRAINING_AVAILABLE:
            analyst_training_config = AnalystTrainingPipelineConfig(
                ensemble_models=False
            )
            self.analyst_training = AnalystTrainingPipeline(analyst_training_config)
        else:
            self.analyst_training = None

        if ANALYST_ENSEMBLE_AVAILABLE:
            self.analyst_ensemble_trainer = AnalystEnsembleTrainingStep(
                AnalystEnsembleTrainingConfig()
            )
        else:
            self.analyst_ensemble_trainer = None

        if TACTICIAN_PRE_ML_AVAILABLE:
            from .tactician_pre_ml_orchestration import TacticianPreMLConfig
            config = TacticianPreMLConfig()
            self.tactician_pre_ml = TacticianPreMLOrchestrator(config)
        else:
            self.tactician_pre_ml = None

        if TACTICIAN_TRAINING_AVAILABLE:
            tactician_training_config = TacticianTrainingPipelineConfig(
                ensemble_models=False
            )
            self.tactician_training = TacticianTrainingPipeline(tactician_training_config)
        else:
            self.tactician_training = None

        if TACTICIAN_ENSEMBLE_AVAILABLE:
            self.tactician_ensemble_trainer = TacticianEnsembleTrainingStep(
                TacticianEnsembleTrainingConfig()
            )
        else:
            self.tactician_ensemble_trainer = None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _get_step_directory(self, config: SubPipelineConfig, step_name: str, create: bool = False) -> Path:
        """Return the directory used to persist artifacts for a given step."""
        base_dir = Path(config.output_directory)
        step_dir = base_dir / step_name
        if create:
            step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def _save_step_artifacts(
        self,
        config: SubPipelineConfig,
        step_name: str,
        artifacts: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Persist step artifacts and metadata so later executions can reload them."""
        if not artifacts:
            return []

        saved_paths: List[str] = []
        try:
            step_dir = self._get_step_directory(config, step_name, create=True)

            if COMMON_IO_AVAILABLE and isinstance(artifacts, dict):
                # Persist any dataframes explicitly referenced by path keys
                for key, value in list(artifacts.items()):
                    if isinstance(value, pd.DataFrame):
                        dataframe_path = step_dir / f"{key}.parquet"
                        safe_to_parquet(value, dataframe_path)
                        artifacts[key + '_path'] = str(dataframe_path)
                        artifacts.pop(key)

            artifact_path = step_dir / "artifacts.pkl"
            with artifact_path.open('wb') as artifact_file:
                pickle.dump(artifacts, artifact_file)
            saved_paths.append(str(artifact_path))

            metadata_path = step_dir / "metadata.pkl"
            with metadata_path.open('wb') as metadata_file:
                pickle.dump(metadata or {}, metadata_file)
            saved_paths.append(str(metadata_path))

            self.logger.debug(f"💾 Saved artifacts for {step_name} to {artifact_path}")
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to persist artifacts for {step_name}: {exc}")

        return saved_paths

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _ensure_directory(self, directory: Path) -> None:
        """Ensure a directory exists on disk."""
        if COMMON_IO_AVAILABLE:
            ensure_directory(directory)
        else:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from parquet with graceful fallback."""
        if COMMON_IO_AVAILABLE and path.exists():
            df = safe_read_parquet(path)
            if df is not None:
                return df
        if not path.exists():
            raise FileNotFoundError(f"Dataframe file not found: {path}")
        return pd.read_parquet(path)

    def _serialize_dataframe(
        self,
        df: pd.DataFrame,
        directory: Path,
        filename: str
    ) -> Path:
        """Serialize a DataFrame to parquet and return the path."""
        self._ensure_directory(directory)
        file_path = directory / filename
        if COMMON_IO_AVAILABLE:
            safe_to_parquet(df, file_path)
        else:
            df.to_parquet(file_path)
        return file_path

    def _persist_metadata(self, metadata: Dict[str, Any], directory: Path, filename: str) -> Path:
        """Persist metadata dictionary to JSON."""
        self._ensure_directory(directory)
        file_path = directory / filename
        if COMMON_IO_AVAILABLE:
            safe_json_dump(metadata, file_path)
        else:
            with file_path.open('w', encoding='utf-8') as fh:
                json.dump(metadata, fh, indent=2, default=str)
        return file_path

    def _candidate_paths(self, directory: Path, patterns: Sequence[str]) -> List[Path]:
        """Return candidate files matching the given patterns ordered by recency."""
        candidates: List[Tuple[float, Path]] = []
        for pattern in patterns:
            for path in directory.glob(pattern):
                try:
                    candidates.append((path.stat().st_mtime, path))
                except FileNotFoundError:
                    continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in candidates]

    def _load_market_data(
        self,
        config: SubPipelineConfig,
        timeframe: str
    ) -> pd.DataFrame:
        """Load market data for the requested timeframe."""
        base_dir = Path(config.data_dir)
        exchange_dir = base_dir / config.exchange.lower()
        symbol_dir = exchange_dir / config.symbol.lower()

        candidate_dirs = [symbol_dir / 'training', symbol_dir]
        patterns = [
            f"*{config.exchange.upper()}*{config.symbol.upper()}*{timeframe}*features*.parquet",
            f"*{config.symbol.upper()}*{timeframe}*features*.parquet",
            f"*{timeframe}*unified_regime_data.parquet",
        ]

        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for candidate in self._candidate_paths(directory, patterns):
                try:
                    df = self._load_dataframe(candidate)
                    if 'timestamp' not in df.columns and df.index.name in ('timestamp', 'datetime'):
                        df = df.reset_index()
                    return df
                except Exception as exc:
                    self.logger.debug(f"⚠️ Failed to load market data from {candidate}: {exc}")

        raise FileNotFoundError(
            f"Market data for timeframe {timeframe} not found in {candidate_dirs}."
        )

    def _load_regime_assignments(
        self,
        config: SubPipelineConfig,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load NAS/TAS regime assignments generated by market analysis."""
        search_directories = [
            Path('generated/market_analysis'),
            Path(config.data_dir) / config.exchange.lower() / config.symbol.lower(),
        ]
        patterns = [
            f"**/*{timeframe}*regime_assignments*.parquet",
            "**/nas_tas_regime_assignments_*.parquet",
        ]

        for base_dir in search_directories:
            if not base_dir.exists():
                continue
            for pattern in patterns:
                for candidate in sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                    try:
                        regime_df = self._load_dataframe(candidate)
                        if 'timestamp' not in regime_df.columns:
                            if 'datetime' in regime_df.columns:
                                regime_df['timestamp'] = pd.to_datetime(regime_df['datetime'])
                            elif regime_df.index.name in ('timestamp', 'datetime'):
                                regime_df = regime_df.reset_index()
                        if 'timestamp' not in regime_df.columns:
                            # Ensure timestamp column exists for merging
                            regime_df['timestamp'] = regime_df.index
                        regime_df['timestamp'] = pd.to_datetime(regime_df['timestamp'])
                        return regime_df
                    except Exception as exc:
                        self.logger.debug(f"⚠️ Failed to load regime assignments from {candidate}: {exc}")
                        continue

        self.logger.warning(
            f"⚠️ Regime assignments not found for timeframe {timeframe}."
        )
        return None

    def _extract_regime_feature_columns(self, regime_df: Optional[pd.DataFrame]) -> List[str]:
        """Extract columns that should be treated as regime features."""
        if regime_df is None:
            return []

        ignore_columns = {'timestamp', 'datetime', 'regime_id', 'cluster_id', 'composite_cluster_id'}
        return [
            column for column in regime_df.columns
            if column not in ignore_columns and np.issubdtype(regime_df[column].dtype, np.number)
        ]

    def _merge_features_with_regime(
        self,
        features_df: pd.DataFrame,
        regime_df: Optional[pd.DataFrame],
        regime_feature_columns: Sequence[str]
    ) -> pd.DataFrame:
        """Merge selected features with regime feature columns on timestamp."""
        if regime_df is None or not regime_feature_columns:
            return features_df

        if 'timestamp' not in features_df.columns:
            if features_df.index.name == 'timestamp':
                features_df = features_df.reset_index()
            else:
                raise ValueError('Features dataframe is missing timestamp column required for regime merge')

        merge_columns = ['timestamp'] + list(regime_feature_columns)
        available_columns = [col for col in merge_columns if col in regime_df.columns]
        if len(available_columns) <= 1:
            return features_df

        merged = features_df.merge(
            regime_df[available_columns].drop_duplicates(subset=['timestamp']),
            on='timestamp',
            how='left'
        )
        return merged

    def _filter_analyst_predictions(
        self,
        predictions: Optional[Any],
        threshold: float
    ) -> Optional[pd.DataFrame]:
        """Filter analyst predictions using the configured confidence threshold."""
        if predictions is None:
            return None

        if isinstance(predictions, pd.DataFrame):
            predictions_df = predictions.copy()
        elif isinstance(predictions, dict):
            predictions_df = pd.DataFrame(predictions)
        else:
            predictions_df = pd.DataFrame(predictions)

        if predictions_df.empty:
            return predictions_df

        if 'timestamp' not in predictions_df.columns:
            predictions_df['timestamp'] = predictions_df.index

        signal_columns = [
            column for column in predictions_df.columns
            if any(keyword in column.lower() for keyword in ['signal', 'score', 'prob', 'prediction'])
            and column != 'timestamp'
        ]

        if not signal_columns:
            return predictions_df

        mask = np.zeros(len(predictions_df), dtype=bool)
        for column in signal_columns:
            try:
                values = pd.to_numeric(predictions_df[column], errors='coerce')
                mask |= values.abs() >= threshold
            except Exception as exc:
                self.logger.debug(f"⚠️ Failed to evaluate prediction column {column}: {exc}")

        if not mask.any():
            return predictions_df

        return predictions_df.loc[mask].copy()

    def _load_step_artifacts(
        self,
        config: SubPipelineConfig,
        step_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load previously saved artifacts for a step, if available."""
        step_dir = self._get_step_directory(config, step_name)
        artifact_path = step_dir / "artifacts.pkl"
        if not artifact_path.exists():
            self.logger.info(
                f"📂 No persisted artifacts found for {step_name} in {artifact_path.parent}"
            )
            return None

        try:
            with artifact_path.open('rb') as artifact_file:
                artifacts = pickle.load(artifact_file)
            self.logger.debug(f"📥 Loaded artifacts for {step_name} from {artifact_path}")
            return artifacts
        except Exception as exc:
            self.logger.error(f"❌ Failed to load artifacts for {step_name}: {exc}")
            return None

    def _load_step_metadata(
        self,
        config: SubPipelineConfig,
        step_name: str
    ) -> Dict[str, Any]:
        """Load previously saved metadata for a step, if available."""
        step_dir = self._get_step_directory(config, step_name)
        metadata_path = step_dir / "metadata.pkl"
        if not metadata_path.exists():
            return {}

        try:
            with metadata_path.open('rb') as metadata_file:
                metadata = pickle.load(metadata_file)
            return metadata if isinstance(metadata, dict) else {}
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to load metadata for {step_name}: {exc}")
            return {}

    def _build_loaded_result(
        self,
        step_name: str,
        artifacts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> SubPipelineResult:
        """Create a SubPipelineResult object from persisted artifacts."""
        now = datetime.now()
        return SubPipelineResult(
            sub_pipeline_name=step_name,
            status=SubPipelineStatus.COMPLETED,
            start_time=now,
            end_time=now,
            duration_seconds=0.0,
            success=True,
            artifacts=artifacts or {},
            metadata=metadata or {}
        )

    async def execute_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """
        Execute the complete model training pipeline.

        Args:
            config: Configuration for pipeline execution

        Returns:
            Dictionary containing execution results
        """
        self.logger.info('🚀 Starting Model Training Sub-Pipeline execution')
        self.logger.info(f'📊 Symbol: {config.symbol}, Exchange: {config.exchange}')
        self.logger.info(f'⏰ Analyst timeframe: {config.analyst_timeframe}, Tactician timeframe: {config.tactician_timeframe}')

        start_time = datetime.now()
        results = {
            'success': False,
            'execution_time': 0.0,
            'analyst_results': {},
            'tactician_results': {},
            'completed_steps': 0,
            'total_steps': 0
        }

        # Count total steps
        total_steps = 0
        if config.train_analyst:
            total_steps += 3  # pre_ml, models, ensemble
        if config.train_tactician:
            total_steps += 3  # pre_ml, models, ensemble
        results['total_steps'] = total_steps

        try:
            # ==================== ANALYST PIPELINE (60m) ====================
            if config.train_analyst:
                self.logger.info('=' * 80)
                self.logger.info('🎯 ANALYST PIPELINE (60m timeframe - IF we trade)')
                self.logger.info('=' * 80)

                # Step 1: Analyst Pre-ML Orchestration
                analyst_pre_ml_result = await self._execute_analyst_pre_ml_orchestration(config)
                if not analyst_pre_ml_result.success:
                    self.logger.error(f'❌ Analyst pre-ML orchestration failed: {analyst_pre_ml_result.error_message}')
                    return results

                results['analyst_results']['pre_ml'] = analyst_pre_ml_result.artifacts
                self._current_pipeline_state['analyst_features'] = analyst_pre_ml_result.artifacts
                results['completed_steps'] += 1

                # Step 2: Analyst Models Training
                analyst_models_result = await self._execute_analyst_models_training(config, analyst_pre_ml_result)
                if not analyst_models_result.success:
                    self.logger.error(f'❌ Analyst models training failed: {analyst_models_result.error_message}')
                    return results

                results['analyst_results']['models'] = analyst_models_result.artifacts
                self._current_pipeline_state['analyst_models'] = analyst_models_result.artifacts
                results['completed_steps'] += 1

                # Step 3: Analyst Ensemble Training
                analyst_ensemble_result = await self._execute_analyst_ensemble_training(config, analyst_models_result)
                if not analyst_ensemble_result.success:
                    self.logger.error(f'❌ Analyst ensemble training failed: {analyst_ensemble_result.error_message}')
                    return results

                results['analyst_results']['ensemble'] = analyst_ensemble_result.artifacts
                self._current_pipeline_state['analyst_ensemble'] = analyst_ensemble_result.artifacts
                results['completed_steps'] += 1

                self.logger.info('✅ Analyst pipeline completed successfully')

            # ==================== TACTICIAN PIPELINE (15m) ====================
            if config.train_tactician:
                self.logger.info('=' * 80)
                self.logger.info('🎯 TACTICIAN PIPELINE (15m timeframe - WHEN we trade)')
                self.logger.info('=' * 80)

                # Get Analyst predictions for filtering
                analyst_predictions = self._current_pipeline_state.get('analyst_ensemble', {}).get('predictions')

                # Step 4: Tactician Pre-ML Orchestration (with Analyst filtering)
                tactician_pre_ml_result = await self._execute_tactician_pre_ml_orchestration(
                    config, analyst_predictions
                )
                if not tactician_pre_ml_result.success:
                    self.logger.error(f'❌ Tactician pre-ML orchestration failed: {tactician_pre_ml_result.error_message}')
                    return results

                results['tactician_results']['pre_ml'] = tactician_pre_ml_result.artifacts
                self._current_pipeline_state['tactician_features'] = tactician_pre_ml_result.artifacts
                results['completed_steps'] += 1

                # Step 5: Tactician Models Training
                tactician_models_result = await self._execute_tactician_models_training(
                    config, tactician_pre_ml_result, analyst_predictions
                )
                if not tactician_models_result.success:
                    self.logger.error(f'❌ Tactician models training failed: {tactician_models_result.error_message}')
                    return results

                results['tactician_results']['models'] = tactician_models_result.artifacts
                self._current_pipeline_state['tactician_models'] = tactician_models_result.artifacts
                results['completed_steps'] += 1

                # Step 6: Tactician Ensemble Training
                tactician_ensemble_result = await self._execute_tactician_ensemble_training(
                    config, tactician_models_result, analyst_predictions
                )
                if not tactician_ensemble_result.success:
                    self.logger.error(f'❌ Tactician ensemble training failed: {tactician_ensemble_result.error_message}')
                    return results

                results['tactician_results']['ensemble'] = tactician_ensemble_result.artifacts
                self._current_pipeline_state['tactician_ensemble'] = tactician_ensemble_result.artifacts
                results['completed_steps'] += 1

                self.logger.info('✅ Tactician pipeline completed successfully')

            # Success
            end_time = datetime.now()
            results['success'] = True
            results['execution_time'] = (end_time - start_time).total_seconds()

            self.logger.info(f'🎉 Model Training Sub-Pipeline completed successfully in {results["execution_time"]:.2f}s')
            self.logger.info(f'📊 Completed steps: {results["completed_steps"]}/{results["total_steps"]}')

        except Exception as e:
            self.logger.error(f'❌ Model Training Sub-Pipeline failed with exception: {e}')
            results['error_message'] = str(e)

        return results

    async def _execute_analyst_pre_ml_orchestration(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute Analyst pre-ML orchestration (60m timeframe)."""
        result = SubPipelineResult(
            sub_pipeline_name='analyst_pre_ml_orchestration',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if not self.analyst_pre_ml:
                raise RuntimeError("Analyst pre-ML orchestrator not available")

            self.logger.info('🔧 Executing Analyst Pre-ML Orchestration (60m)...')

            training_data = self._load_market_data(config, config.analyst_timeframe)
            regime_assignments = self._load_regime_assignments(config, config.analyst_timeframe)

            regime_feature_columns = self._extract_regime_feature_columns(regime_assignments)

            # Execute orchestration
            orchestration_result = await self.analyst_pre_ml.orchestrate(
                training_data=training_data,
                regime_assignments=regime_assignments,
            )

            result.success = orchestration_result.success
            result.status = SubPipelineStatus.COMPLETED if orchestration_result.success else SubPipelineStatus.FAILED
            result.error_message = orchestration_result.error_message
            artifacts: Dict[str, Any] = {
                'selected_features': orchestration_result.selected_feature_names or [],
                'feature_count': orchestration_result.final_feature_count,
                'regime_feature_columns': regime_feature_columns,
            }

            step_dir = self._get_step_directory(config, 'analyst_pre_ml_orchestration', create=True)

            if orchestration_result.final_features is not None:
                feature_path = self._serialize_dataframe(
                    orchestration_result.final_features,
                    step_dir,
                    'final_features.parquet'
                )
                artifacts['final_features_path'] = str(feature_path)

            if regime_assignments is not None:
                regime_path = self._serialize_dataframe(
                    regime_assignments,
                    step_dir,
                    'regime_assignments.parquet'
                )
                artifacts['regime_assignments_path'] = str(regime_path)

            metadata_payload = {
                'selected_feature_count': len(orchestration_result.selected_feature_names or []),
                'regime_feature_columns': regime_feature_columns,
                'total_samples': getattr(orchestration_result, 'total_samples', len(training_data)),
                'final_feature_count': getattr(orchestration_result, 'final_feature_count', None),
            }
            metadata_path = self._persist_metadata(
                metadata_payload,
                step_dir,
                'metadata.json'
            )
            artifacts['metadata_path'] = str(metadata_path)
            result.artifacts = artifacts
            result.metadata = {
                'total_samples': getattr(orchestration_result, 'total_samples', None),
                'final_feature_count': getattr(orchestration_result, 'final_feature_count', None),
                'selection_phase': getattr(orchestration_result, 'phase', None)
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'analyst_pre_ml_orchestration',
                    artifacts,
                    result.metadata
                )

                self._current_pipeline_state['analyst_regime_assignments'] = regime_assignments
                self._current_pipeline_state['analyst_feature_frame_path'] = artifacts.get('final_features_path')

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f'❌ Analyst pre-ML orchestration failed: {e}')

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_analyst_models_training(
        self, config: SubPipelineConfig, pre_ml_result: SubPipelineResult
    ) -> SubPipelineResult:
        """Execute Analyst models training (base models)."""
        result = SubPipelineResult(
            sub_pipeline_name='analyst_models_training',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if not self.analyst_training:
                raise RuntimeError("Analyst training pipeline not available")

            self.logger.info('📈 Executing Analyst Models Training...')

            artifacts = pre_ml_result.artifacts
            feature_path = artifacts.get('final_features_path')
            if feature_path:
                features_df = self._load_dataframe(Path(feature_path))
            else:
                features_df = artifacts.get('final_features')
                if isinstance(features_df, pd.DataFrame):
                    features_df = features_df.copy()
                else:
                    raise ValueError('Final features not available for analyst training')

            regime_assignments_path = artifacts.get('regime_assignments_path')
            regime_assignments = None
            if regime_assignments_path:
                try:
                    regime_assignments = self._load_dataframe(Path(regime_assignments_path))
                except FileNotFoundError:
                    regime_assignments = self._current_pipeline_state.get('analyst_regime_assignments')
            else:
                regime_assignments = self._current_pipeline_state.get('analyst_regime_assignments')

            regime_feature_columns = artifacts.get('regime_feature_columns', [])
            if regime_feature_columns:
                features_df = self._merge_features_with_regime(
                    features_df,
                    regime_assignments,
                    regime_feature_columns
                )

            selected_features = artifacts.get('selected_features', [])
            feature_columns = list(dict.fromkeys(selected_features + regime_feature_columns))

            if not feature_columns:
                raise ValueError('No feature columns available for analyst training')

            # Execute training
            training_result = await self.analyst_training.train_analyst_models(
                training_data=features_df,
                feature_columns=feature_columns,
                target_columns=['target_long', 'target_short'],
                regime_assignments=regime_assignments,
            )

            # Execute per-regime training alongside base model training
            per_regime_result = None
            if PER_REGIME_TRAINING_AVAILABLE and training_result.base_training_completed:
                try:
                    self.logger.info('🎯 Executing Analyst per-regime training...')
                    per_regime_result = train_analyst_per_regime_models(
                        training_data=features_df,
                        feature_columns=feature_columns,
                        target_columns=['target_long', 'target_short'],
                        regime_assignments=regime_assignments
                    )

                    if per_regime_result.success:
                        self.logger.info('✅ Analyst per-regime training completed successfully')
                    else:
                        self.logger.warning(f'⚠️ Analyst per-regime training failed: {per_regime_result.error_message}')

                except Exception as e:
                    self.logger.warning(f'⚠️ Analyst per-regime training failed: {e}')

            result.success = training_result.base_training_completed
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED
            result.artifacts = {
                'base_models': training_result.base_models,
                'metrics': training_result.base_training_metrics,
                'feature_columns': feature_columns,
                'training_data_path': feature_path,
                'regime_feature_columns': regime_feature_columns,
                'regime_assignments_path': regime_assignments_path,
                'per_regime_models': per_regime_result.regime_models if per_regime_result else {},
                'per_regime_metadata': per_regime_result.regime_metadata if per_regime_result else {},
                'model_selector': per_regime_result.model_selector if per_regime_result else None,
            }
            result.metadata = {
                'training_summary': getattr(training_result, 'training_summary', None),
                'metrics': training_result.base_training_metrics,
                'ensemble_available': getattr(training_result, 'ensemble_models', None) is not None,
                'direction_settings': {
                    'enable_long_positions': config.enable_long_positions,
                    'enable_short_positions': config.enable_short_positions,
                }
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'analyst_models_training',
                    result.artifacts,
                    result.metadata
                )

                self._current_pipeline_state['analyst_base_models'] = training_result.base_models
                self._current_pipeline_state['analyst_feature_columns'] = feature_columns
                self._current_pipeline_state['analyst_training_frame'] = feature_path

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f'❌ Analyst models training failed: {e}')

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_analyst_ensemble_training(
        self, config: SubPipelineConfig, models_result: SubPipelineResult
    ) -> SubPipelineResult:
        """Execute Analyst ensemble training."""
        result = SubPipelineResult(
            sub_pipeline_name='analyst_ensemble_training',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if not self.analyst_ensemble_trainer:
                raise RuntimeError("Analyst ensemble training step not available")

            self.logger.info('🔄 Executing Analyst Ensemble Training...')

            base_models = models_result.artifacts.get('base_models')
            feature_columns = models_result.artifacts.get('feature_columns', [])
            feature_path = models_result.artifacts.get('training_data_path')

            if not base_models:
                raise ValueError('Analyst base models missing for ensemble training')
            if not feature_columns:
                raise ValueError('Feature columns missing for analyst ensemble training')

            if feature_path:
                training_data = self._load_dataframe(Path(feature_path))
            else:
                training_data = self._current_pipeline_state.get('analyst_features')
                if isinstance(training_data, dict):
                    training_data = training_data.get('final_features')
            if training_data is None:
                raise ValueError('Training data not available for analyst ensemble training')

            if 'timestamp' not in training_data.columns and training_data.index.name == 'timestamp':
                training_data = training_data.reset_index()

            ensemble_result = await self.analyst_ensemble_trainer.train_analyst_ensemble(
                training_data=training_data,
                base_models=base_models,
                feature_columns=feature_columns,
                target_columns=['target_long', 'target_short'],
            )

            ensemble_models = ensemble_result.get('models', {})
            predictions_df = None
            if ensemble_models:
                feature_frame = training_data[feature_columns].copy()
                timestamps = training_data['timestamp'] if 'timestamp' in training_data.columns else pd.RangeIndex(len(feature_frame))
                predictions_data: Dict[str, Any] = {'timestamp': timestamps}
                for name, model in ensemble_models.items():
                    try:
                        model_obj = model.get('model') if isinstance(model, dict) else model
                        if hasattr(model_obj, 'predict'):
                            preds = model_obj.predict(feature_frame)
                            predictions_data[f'{name}_prediction'] = preds
                    except Exception as pred_exc:
                        self.logger.warning(f"⚠️ Failed to generate predictions for {name}: {pred_exc}")
                if len(predictions_data) > 1:
                    predictions_df = pd.DataFrame(predictions_data)

            step_dir = self._get_step_directory(config, 'analyst_ensemble_training', create=True)
            artifacts: Dict[str, Any] = {
                'ensemble_models': ensemble_models,
                'metrics': ensemble_result.get('metrics', {}),
            }

            if predictions_df is not None:
                predictions_path = self._serialize_dataframe(
                    predictions_df,
                    step_dir,
                    'ensemble_predictions.parquet'
                )
                artifacts['predictions_path'] = str(predictions_path)
                artifacts['predictions'] = predictions_df
            else:
                artifacts['predictions'] = None

            result.success = bool(ensemble_models)
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED
            result.artifacts = artifacts
            result.metadata = {
                'samples_used': ensemble_result.get('samples_used'),
                'feature_integration_complete': ensemble_result.get('feature_integration_complete', False),
                'direction_settings': {
                    'enable_long_positions': config.enable_long_positions,
                    'enable_short_positions': config.enable_short_positions,
                }
            }

            result.output_files = self._save_step_artifacts(
                config,
                'analyst_ensemble_training',
                artifacts,
                result.metadata
            )

            if predictions_df is not None:
                self._current_pipeline_state['analyst_ensemble_predictions'] = predictions_df
            self._current_pipeline_state['analyst_ensemble_models'] = ensemble_models

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f'❌ Analyst ensemble training failed: {e}')

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_tactician_pre_ml_orchestration(
        self, config: SubPipelineConfig, analyst_predictions: Optional[pd.DataFrame]
    ) -> SubPipelineResult:
        """Execute Tactician pre-ML orchestration (15m timeframe, includes Analyst outputs as features)."""
        result = SubPipelineResult(
            sub_pipeline_name='tactician_pre_ml_orchestration',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if not self.tactician_pre_ml:
                raise RuntimeError("Tactician pre-ML orchestrator not available")

            self.logger.info('🔧 Executing Tactician Pre-ML Orchestration (15m, with Analyst features)...')

            training_data = self._load_market_data(config, config.tactician_timeframe)
            regime_assignments = self._load_regime_assignments(config, config.tactician_timeframe)
            regime_feature_columns = self._extract_regime_feature_columns(regime_assignments)

            regime_data_splitting_result = (
                config.custom_params.get('regime_data_splitting_result')
                if config.custom_params else None
            )
            if regime_data_splitting_result is None:
                regime_data_splitting_result = self._current_pipeline_state.get('regime_data_splitting_result')
            if regime_data_splitting_result is not None:
                self._current_pipeline_state['regime_data_splitting_result'] = regime_data_splitting_result

            # Execute orchestration with Analyst predictions as features (no filtering)
            orchestration_result = await self.tactician_pre_ml.orchestrate(
                training_data=training_data,
                analyst_predictions=analyst_predictions,
                regime_assignments=regime_assignments,
                regime_data_splitting_result=regime_data_splitting_result,
            )

            result.success = orchestration_result.success
            result.status = SubPipelineStatus.COMPLETED if orchestration_result.success else SubPipelineStatus.FAILED
            result.error_message = orchestration_result.error_message
            artifacts: Dict[str, Any] = {
                'selected_features': orchestration_result.selected_feature_names or [],
                'feature_count': orchestration_result.final_feature_count,
                'regime_feature_columns': regime_feature_columns,
            }

            step_dir = self._get_step_directory(config, 'tactician_pre_ml_orchestration', create=True)

            if orchestration_result.final_features is not None:
                feature_path = self._serialize_dataframe(
                    orchestration_result.final_features,
                    step_dir,
                    'final_features.parquet'
                )
                artifacts['final_features_path'] = str(feature_path)

            if regime_assignments is not None:
                regime_path = self._serialize_dataframe(
                    regime_assignments,
                    step_dir,
                    'regime_assignments.parquet'
                )
                artifacts['regime_assignments_path'] = str(regime_path)

            if filtered_predictions is not None:
                predictions_path = self._serialize_dataframe(
                    filtered_predictions,
                    step_dir,
                    'analyst_predictions.parquet'
                )
                artifacts['analyst_predictions_path'] = str(predictions_path)

            result.artifacts = artifacts
            result.metadata = {
                'total_samples': getattr(orchestration_result, 'total_samples', None)
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'tactician_pre_ml_orchestration',
                    artifacts,
                    result.metadata
                )

                self._current_pipeline_state['tactician_regime_assignments'] = regime_assignments
                self._current_pipeline_state['tactician_filtered_predictions_path'] = artifacts.get('analyst_predictions_path')
                self._current_pipeline_state['tactician_feature_frame_path'] = artifacts.get('final_features_path')

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f'❌ Tactician pre-ML orchestration failed: {e}')

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_tactician_models_training(
        self, config: SubPipelineConfig, pre_ml_result: SubPipelineResult, analyst_predictions: Optional[pd.DataFrame]
    ) -> SubPipelineResult:
        """Execute Tactician models training (base models with Analyst features)."""
        result = SubPipelineResult(
            sub_pipeline_name='tactician_models_training',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if not self.tactician_training:
                raise RuntimeError("Tactician training pipeline not available")

            self.logger.info('📈 Executing Tactician Models Training...')

            artifacts = pre_ml_result.artifacts
            feature_path = artifacts.get('final_features_path')
            if feature_path:
                features_df = self._load_dataframe(Path(feature_path))
            else:
                features_df = artifacts.get('final_features')
                if isinstance(features_df, pd.DataFrame):
                    features_df = features_df.copy()
                else:
                    raise ValueError('Final tactician features not available')

            regime_assignments_path = artifacts.get('regime_assignments_path')
            regime_assignments = None
            if regime_assignments_path:
                try:
                    regime_assignments = self._load_dataframe(Path(regime_assignments_path))
                except FileNotFoundError:
                    regime_assignments = self._current_pipeline_state.get('tactician_regime_assignments')
            else:
                regime_assignments = self._current_pipeline_state.get('tactician_regime_assignments')

            regime_feature_columns = artifacts.get('regime_feature_columns', [])
            if regime_feature_columns:
                features_df = self._merge_features_with_regime(
                    features_df,
                    regime_assignments,
                    regime_feature_columns
                )

            analyst_predictions_df = None
            predictions_path = artifacts.get('analyst_predictions_path')
            if analyst_predictions is not None:
                analyst_predictions_df = analyst_predictions
            elif predictions_path:
                analyst_predictions_df = self._load_dataframe(Path(predictions_path))
            elif self._current_pipeline_state.get('analyst_ensemble_predictions') is not None:
                analyst_predictions_df = self._current_pipeline_state.get('analyst_ensemble_predictions')

            analyst_feature_columns: List[str] = []
            if analyst_predictions_df is not None and not analyst_predictions_df.empty:
                if 'timestamp' not in analyst_predictions_df.columns:
                    analyst_predictions_df['timestamp'] = analyst_predictions_df.index
                value_columns = [
                    column for column in analyst_predictions_df.columns
                    if column != 'timestamp'
                ]
                analyst_feature_columns = value_columns
                features_df = features_df.merge(
                    analyst_predictions_df[['timestamp'] + value_columns],
                    on='timestamp',
                    how='left'
                )

            feature_columns = list(dict.fromkeys(
                artifacts.get('selected_features', []) + regime_feature_columns + analyst_feature_columns
            ))

            if not feature_columns:
                raise ValueError('No features available for tactician training')

            # Execute training
            training_result = await self.tactician_training.train_tactician_models(
                training_data=features_df,
                feature_columns=feature_columns,
                target_columns=['target_long', 'target_short'],
                regime_assignments=regime_assignments,
            )

            # Execute per-regime training alongside base model training
            per_regime_result = None
            if PER_REGIME_TRAINING_AVAILABLE and training_result.base_training_completed:
                try:
                    self.logger.info('🎯 Executing Tactician per-regime training...')
                    per_regime_result = train_tactician_per_regime_models(
                        training_data=features_df,
                        feature_columns=feature_columns,
                        target_columns=['target_long', 'target_short'],
                        regime_assignments=regime_assignments
                    )

                    if per_regime_result.success:
                        self.logger.info('✅ Tactician per-regime training completed successfully')
                    else:
                        self.logger.warning(f'⚠️ Tactician per-regime training failed: {per_regime_result.error_message}')

                except Exception as e:
                    self.logger.warning(f'⚠️ Tactician per-regime training failed: {e}')

            result.success = training_result.base_training_completed
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED
            result.artifacts = {
                'base_models': training_result.base_models,
                'metrics': training_result.base_training_metrics,
                'feature_columns': feature_columns,
                'training_data_path': feature_path,
                'regime_feature_columns': regime_feature_columns,
                'analyst_feature_columns': analyst_feature_columns,
                'regime_assignments_path': regime_assignments_path,
                'per_regime_models': per_regime_result.regime_models if per_regime_result else {},
                'per_regime_metadata': per_regime_result.regime_metadata if per_regime_result else {},
                'model_selector': per_regime_result.model_selector if per_regime_result else None,
                'analyst_predictions_path': predictions_path,
            }
            result.metadata = {
                'training_summary': getattr(training_result, 'training_summary', None),
                'metrics': training_result.base_training_metrics,
                'direction_settings': {
                    'enable_long_positions': config.enable_long_positions,
                    'enable_short_positions': config.enable_short_positions,
                }
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'tactician_models_training',
                    result.artifacts,
                    result.metadata
                )

                self._current_pipeline_state['tactician_base_models'] = training_result.base_models
                self._current_pipeline_state['tactician_feature_columns'] = feature_columns
                self._current_pipeline_state['tactician_training_frame'] = feature_path

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f'❌ Tactician models training failed: {e}')

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_tactician_ensemble_training(
        self, config: SubPipelineConfig, models_result: SubPipelineResult, analyst_predictions: Optional[pd.DataFrame]
    ) -> SubPipelineResult:
        """Execute Tactician ensemble training (with Analyst features)."""
        result = SubPipelineResult(
            sub_pipeline_name='tactician_ensemble_training',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if not self.tactician_ensemble_trainer:
                raise RuntimeError("Tactician ensemble training step not available")

            self.logger.info('🔄 Executing Tactician Ensemble Training...')

            base_models = models_result.artifacts.get('base_models')
            feature_columns = models_result.artifacts.get('feature_columns', [])
            feature_path = models_result.artifacts.get('training_data_path')

            if not base_models:
                raise ValueError('Tactician base models missing for ensemble training')
            if not feature_columns:
                raise ValueError('Tactician feature columns missing for ensemble training')

            if feature_path:
                training_data = self._load_dataframe(Path(feature_path))
            else:
                training_data = self._current_pipeline_state.get('tactician_features')
                if isinstance(training_data, dict):
                    training_data = training_data.get('final_features')
            if training_data is None:
                raise ValueError('Training data not available for tactician ensemble training')

            if 'timestamp' not in training_data.columns and training_data.index.name == 'timestamp':
                training_data = training_data.reset_index()

            training_frame = training_data.copy()
            base_prediction_columns: List[str] = []
            for regime_key, model_mapping in (base_models or {}).items():
                if not isinstance(model_mapping, dict):
                    continue
                for model_name, model_container in model_mapping.items():
                    try:
                        model_obj = model_container.get('model') if isinstance(model_container, dict) else model_container
                        if not hasattr(model_obj, 'predict'):
                            continue
                        preds = model_obj.predict(training_frame[feature_columns])
                        prediction_column = f"{model_name}_regime_{regime_key}_prediction"
                        training_frame[prediction_column] = preds
                        base_prediction_columns.append(prediction_column)
                    except Exception as prediction_exc:
                        self.logger.debug(f"⚠️ Failed to compute predictions for {model_name} ({regime_key}): {prediction_exc}")

            augmented_feature_columns = list(dict.fromkeys(feature_columns + base_prediction_columns))

            ensemble_result = await self.tactician_ensemble_trainer.train_tactician_ensemble(
                training_data=training_frame,
                base_models=base_models,
                feature_columns=augmented_feature_columns,
                target_columns=['target_long', 'target_short'],
            )

            ensemble_models = ensemble_result.get('models', {})
            predictions_df = None
            if ensemble_models:
                feature_subset = training_frame[augmented_feature_columns]
                timestamps = training_frame['timestamp'] if 'timestamp' in training_frame.columns else pd.RangeIndex(len(feature_subset))
                predictions_data: Dict[str, Any] = {'timestamp': timestamps}
                for name, model in ensemble_models.items():
                    try:
                        model_obj = model.get('model') if isinstance(model, dict) else model
                        if hasattr(model_obj, 'predict'):
                            preds = model_obj.predict(feature_subset)
                            predictions_data[f'{name}_prediction'] = preds
                    except Exception as pred_exc:
                        self.logger.warning(f"⚠️ Failed to generate tactician ensemble predictions for {name}: {pred_exc}")
                if len(predictions_data) > 1:
                    predictions_df = pd.DataFrame(predictions_data)

            step_dir = self._get_step_directory(config, 'tactician_ensemble_training', create=True)
            artifacts: Dict[str, Any] = {
                'ensemble_models': ensemble_models,
                'metrics': ensemble_result.get('metrics', {}),
                'feature_columns': augmented_feature_columns,
                'base_prediction_columns': base_prediction_columns,
            }

            if predictions_df is not None:
                predictions_path = self._serialize_dataframe(
                    predictions_df,
                    step_dir,
                    'ensemble_predictions.parquet'
                )
                artifacts['predictions_path'] = str(predictions_path)
                artifacts['predictions'] = predictions_df
            else:
                artifacts['predictions'] = None

            result.success = bool(ensemble_models)
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED
            result.artifacts = artifacts
            result.metadata = {
                'samples_used': ensemble_result.get('samples_used'),
                'feature_integration_complete': ensemble_result.get('feature_integration_complete', False),
                'direction_settings': {
                    'enable_long_positions': config.enable_long_positions,
                    'enable_short_positions': config.enable_short_positions,
                }
            }

            result.output_files = self._save_step_artifacts(
                config,
                'tactician_ensemble_training',
                artifacts,
                result.metadata
            )

            if predictions_df is not None:
                self._current_pipeline_state['tactician_ensemble_predictions'] = predictions_df
            self._current_pipeline_state['tactician_ensemble_models'] = ensemble_models

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f'❌ Tactician ensemble training failed: {e}')

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return [
            'analyst_pre_ml_orchestration',
            'analyst_models_training',
            'analyst_ensemble_training',
            'tactician_pre_ml_orchestration',
            'tactician_models_training',
            'tactician_ensemble_training'
        ]

    async def execute_sub_pipeline(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline."""
        result: Optional[SubPipelineResult] = None

        if sub_pipeline_name == 'analyst_pre_ml_orchestration':
            result = await self._execute_analyst_pre_ml_orchestration(config)
        elif sub_pipeline_name == 'analyst_models_training':
            pre_ml_artifacts = self._load_step_artifacts(config, 'analyst_pre_ml_orchestration')
            if pre_ml_artifacts is None:
                raise FileNotFoundError(
                    "Analyst pre-ML artifacts not found. Run 'analyst_pre_ml_orchestration' first or provide persisted artifacts."
                )

            pre_ml_metadata = self._load_step_metadata(config, 'analyst_pre_ml_orchestration')
            pre_ml_result = self._build_loaded_result(
                'analyst_pre_ml_orchestration',
                pre_ml_artifacts,
                pre_ml_metadata
            )

            result = await self._execute_analyst_models_training(config, pre_ml_result)
        elif sub_pipeline_name == 'analyst_ensemble_training':
            models_artifacts = self._load_step_artifacts(config, 'analyst_models_training')
            if models_artifacts is None:
                raise FileNotFoundError(
                    "Analyst model artifacts not found. Run 'analyst_models_training' first or provide persisted artifacts."
                )

            models_metadata = self._load_step_metadata(config, 'analyst_models_training')
            models_result = self._build_loaded_result(
                'analyst_models_training',
                models_artifacts,
                models_metadata
            )

            result = await self._execute_analyst_ensemble_training(config, models_result)
        elif sub_pipeline_name == 'tactician_pre_ml_orchestration':
            analyst_artifacts = self._load_step_artifacts(config, 'analyst_ensemble_training')
            analyst_metadata = self._load_step_metadata(config, 'analyst_ensemble_training') if analyst_artifacts is not None else {}
            analyst_result = None
            if analyst_artifacts is not None:
                analyst_result = self._build_loaded_result(
                    'analyst_ensemble_training',
                    analyst_artifacts,
                    analyst_metadata
                )

            analyst_predictions = None
            if analyst_result:
                analyst_predictions = analyst_result.artifacts.get('predictions')

            result = await self._execute_tactician_pre_ml_orchestration(
                config,
                analyst_predictions
            )
        elif sub_pipeline_name == 'tactician_models_training':
            pre_ml_artifacts = self._load_step_artifacts(config, 'tactician_pre_ml_orchestration')
            if pre_ml_artifacts is None:
                raise FileNotFoundError(
                    "Tactician pre-ML artifacts not found. Run 'tactician_pre_ml_orchestration' first or provide persisted artifacts."
                )

            pre_ml_metadata = self._load_step_metadata(config, 'tactician_pre_ml_orchestration')
            pre_ml_result = self._build_loaded_result(
                'tactician_pre_ml_orchestration',
                pre_ml_artifacts,
                pre_ml_metadata
            )

            analyst_artifacts = self._load_step_artifacts(config, 'analyst_ensemble_training')
            analyst_metadata = self._load_step_metadata(config, 'analyst_ensemble_training') if analyst_artifacts is not None else {}
            analyst_result = None
            if analyst_artifacts is not None:
                analyst_result = self._build_loaded_result(
                    'analyst_ensemble_training',
                    analyst_artifacts,
                    analyst_metadata
                )

            analyst_predictions = analyst_result.artifacts.get('predictions') if analyst_result else None

            result = await self._execute_tactician_models_training(
                config,
                pre_ml_result,
                analyst_predictions
            )
        elif sub_pipeline_name == 'tactician_ensemble_training':
            models_artifacts = self._load_step_artifacts(config, 'tactician_models_training')
            if models_artifacts is None:
                raise FileNotFoundError(
                    "Tactician model artifacts not found. Run 'tactician_models_training' first or provide persisted artifacts."
                )

            models_metadata = self._load_step_metadata(config, 'tactician_models_training')
            models_result = self._build_loaded_result(
                'tactician_models_training',
                models_artifacts,
                models_metadata
            )

            analyst_artifacts = self._load_step_artifacts(config, 'analyst_ensemble_training')
            analyst_predictions = None
            if analyst_artifacts is not None:
                analyst_predictions = analyst_artifacts.get('predictions')

            result = await self._execute_tactician_ensemble_training(
                config,
                models_result,
                analyst_predictions
            )
        else:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")

        if result:
            self.results.append(result)

            if result.success:
                if sub_pipeline_name == 'analyst_pre_ml_orchestration':
                    self._current_pipeline_state['analyst_features'] = result.artifacts
                elif sub_pipeline_name == 'analyst_models_training':
                    self._current_pipeline_state['analyst_models'] = result.artifacts
                elif sub_pipeline_name == 'analyst_ensemble_training':
                    self._current_pipeline_state['analyst_ensemble'] = result.artifacts
                elif sub_pipeline_name == 'tactician_pre_ml_orchestration':
                    self._current_pipeline_state['tactician_features'] = result.artifacts
                elif sub_pipeline_name == 'tactician_models_training':
                    self._current_pipeline_state['tactician_models'] = result.artifacts
                elif sub_pipeline_name == 'tactician_ensemble_training':
                    self._current_pipeline_state['tactician_ensemble'] = result.artifacts

            return result

        raise RuntimeError(f"Sub-pipeline '{sub_pipeline_name}' did not return a result")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
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
async def execute_model_training_pipeline(config: SubPipelineConfig) -> Dict[str, Any]:
    """Execute the model training pipeline."""
    pipeline = ModelTrainingSubPipeline()
    return await pipeline.execute_pipeline(config)
