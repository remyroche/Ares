"""
Model Training Sub-Pipeline - Orchestration of Analyst and Tactician Training

This module orchestrates the complete model training pipeline with distinct
workflows for Analyst and Tactician models:

ANALYST PIPELINE (15m timeframe - IF we trade):
1. analyst_pre_ml_orchestration - Feature engineering on 15m data
2. analyst_models_training - Train base models (per-regime)
3. analyst_ensemble_training - Train ensemble models

TACTICIAN PIPELINE (5m timeframe - WHEN we trade):
4. tactician_pre_ml_orchestration - Feature engineering on 5m data (filtered on Analyst signals)
5. tactician_models_training - Train base models
6. tactician_ensemble_training - Train ensemble models

Each model type (short/long) is trained separately.
"""

from typing import Any, Dict, List, Optional, Iterable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd
import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    JOBLIB_AVAILABLE = False
    joblib = None

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import orchestration and training steps
try:
    from ..models_training.analyst_pre_ml_orchestration import (
        AnalystPreMLOrchestrator, AnalystPreMLConfig, AnalystPreMLResult
    )
    ANALYST_PRE_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_pre_ml_orchestration not available: {e}")
    ANALYST_PRE_ML_AVAILABLE = False

try:
    from ..models_training.analyst_training_pipeline import (
        AnalystTrainingPipeline, AnalystTrainingPipelineConfig, AnalystTrainingPipelineResult
    )
    ANALYST_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_training_pipeline not available: {e}")
    ANALYST_TRAINING_AVAILABLE = False

try:
    from ..models_training.tactician_pre_ml_orchestration import (
        TacticianPreMLOrchestrator, TacticianPreMLConfig, TacticianPreMLResult
    )
    TACTICIAN_PRE_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: tactician_pre_ml_orchestration not available: {e}")
    TACTICIAN_PRE_ML_AVAILABLE = False

try:
    from ..models_training.tactician_training_pipeline import (
        TacticianTrainingPipeline, TacticianTrainingPipelineConfig, TacticianTrainingPipelineResult
    )
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: tactician_training_pipeline not available: {e}")
    TACTICIAN_TRAINING_AVAILABLE = False

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
    analyst_timeframe: str = "15m"  # Analyst uses 15m
    tactician_timeframe: str = "5m"  # Tactician uses 5m
    data_dir: str = "historical_data"
    
    # Training configuration
    train_analyst: bool = True
    train_tactician: bool = True
    train_short_models: bool = True
    train_long_models: bool = True
    
    # Analyst configuration
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for Tactician filtering
    
    # Execution parameters
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    
    # Output configuration
    output_directory: str = "generated/model_training"
    save_models: bool = True
    
    # Custom parameters
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
            self.analyst_pre_ml = AnalystPreMLOrchestrator()
        else:
            self.analyst_pre_ml = None
            
        if ANALYST_TRAINING_AVAILABLE:
            self.analyst_training = AnalystTrainingPipeline()
        else:
            self.analyst_training = None
            
        if TACTICIAN_PRE_ML_AVAILABLE:
            self.tactician_pre_ml = TacticianPreMLOrchestrator()
        else:
            self.tactician_pre_ml = None
            
        if TACTICIAN_TRAINING_AVAILABLE:
            self.tactician_training = TacticianTrainingPipeline()
        else:
            self.tactician_training = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _json_default(self, obj: Any) -> Any:
        """Fallback serializer for JSON dumps."""
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='list')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return str(obj)

    def _ensure_directory(self, path: Path) -> None:
        """Ensure that a directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read a parquet file into a DataFrame."""
        self.logger.debug(f"Loading parquet file: {path}")
        return pd.read_parquet(path)

    def _find_latest_file(
        self,
        directories: Iterable[Path],
        include_tokens: Optional[List[str]] = None,
        suffixes: Optional[List[str]] = None
    ) -> Optional[Path]:
        """Find the most recent file matching the include tokens and suffixes."""

        include_tokens = include_tokens or []
        suffixes = suffixes or ['.parquet']

        candidates: List[Path] = []
        for directory in directories:
            if not directory or not directory.exists():
                continue

            for suffix in suffixes:
                candidates.extend([
                    path for path in directory.glob(f"*{suffix}")
                    if all(token.lower() in path.name.lower() for token in include_tokens)
                ])

        if not candidates:
            return None

        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_input_dataframe(
        self,
        config: SubPipelineConfig,
        *,
        role: str,
        timeframe: str,
        kind: str,
        required: bool = False,
        custom_key: Optional[str] = None,
        default_directories: Optional[List[Path]] = None
    ) -> Optional[pd.DataFrame]:
        """Load an input DataFrame from pipeline state, custom path, or default locations."""

        state_key = f"{role}_{kind}"
        if state_key in self._current_pipeline_state:
            state_value = self._current_pipeline_state[state_key]
            if isinstance(state_value, pd.DataFrame):
                return state_value.copy()

        # Custom configuration override
        custom_keys = [
            custom_key or f"{role}_{kind}_path",
            f"{role}_{timeframe}_{kind}_path",
            f"{kind}_path"
        ]
        for key in custom_keys:
            if not key:
                continue
            path_value = config.custom_params.get(key)
            if path_value:
                path = Path(path_value)
                if path.exists():
                    return self._read_parquet(path)
                else:
                    self.logger.warning(f"Configured path does not exist for {role} {kind}: {path}")

        # Search default directories
        default_directories = default_directories or []
        search_directories = [
            Path(config.output_directory) / role / 'pre_ml',
            Path('generated/market_analysis'),
            Path('generated/market_analysis/regime_data_splitting'),
            Path('generated/market_analysis/pid_features')
        ] + default_directories

        include_tokens = [role, timeframe, kind]
        candidate = self._find_latest_file(search_directories, include_tokens=include_tokens)
        if candidate:
            return self._read_parquet(candidate)

        if required:
            raise FileNotFoundError(
                f"Unable to locate required {role} {kind} data for timeframe {timeframe}. "
                f"Provide a path via config.custom_params or ensure generated artifacts exist."
            )

        return None

    def _save_dataframe_artifact(self, df: pd.DataFrame, path: Path) -> str:
        """Persist a DataFrame to parquet and return the saved path."""
        self._ensure_directory(path.parent)
        df.to_parquet(path)
        return str(path)

    def _save_json_artifact(self, payload: Dict[str, Any], path: Path) -> str:
        """Persist a JSON payload and return the saved path."""
        self._ensure_directory(path.parent)
        with path.open('w') as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)
        return str(path)

    def _get_target_columns(self, data: pd.DataFrame) -> List[str]:
        """Infer target columns from a training DataFrame."""
        candidate_targets = [col for col in data.columns if col.startswith('target_')]
        if not candidate_targets:
            raise ValueError("No target columns found in training data (expected columns starting with 'target_')")
        return candidate_targets

    def _merge_regime_features(
        self,
        base_frame: pd.DataFrame,
        regime_features: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge regime features into the base feature frame if available."""
        if regime_features is None or regime_features.empty:
            return base_frame

        aligned = base_frame.join(regime_features, how='left')
        self.logger.info(
            f"Merged regime features: base columns={base_frame.shape[1]}, "
            f"regime columns={regime_features.shape[1]}, result columns={aligned.shape[1]}"
        )
        return aligned

    def _collect_regime_features(self, artifacts: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract regime feature DataFrame from orchestration artifacts if present."""
        regime_features = artifacts.get('regime_features') or artifacts.get('metadata', {}).get('regime_features')
        if isinstance(regime_features, pd.DataFrame):
            return regime_features
        if isinstance(regime_features, dict):
            try:
                return pd.DataFrame(regime_features)
            except Exception:  # pragma: no cover - defensive
                return None
        if isinstance(regime_features, np.ndarray):
            return pd.DataFrame(regime_features)
        return None

    def _generate_model_predictions(
        self,
        models: Dict[str, Any],
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions for each model given a feature matrix."""
        predictions: Dict[str, np.ndarray] = {}
        for model_name, model in (models or {}).items():
            if not hasattr(model, 'predict'):
                continue
            try:
                model_predictions = model.predict(features.values)
                if isinstance(model_predictions, (list, tuple)):
                    model_predictions = np.asarray(model_predictions)
                if model_predictions.ndim == 1:
                    predictions[model_name] = model_predictions
                else:
                    for idx in range(model_predictions.shape[1]):
                        predictions[f"{model_name}_{idx}"] = model_predictions[:, idx]
            except Exception as exc:  # pragma: no cover - model specific edge cases
                self.logger.warning(f"Failed to generate predictions for model {model_name}: {exc}")

        if not predictions:
            return pd.DataFrame(index=features.index)

        return pd.DataFrame(predictions, index=features.index)

    def _persist_models(self, models: Dict[str, Any], destination: Path, name_filter: Optional[List[str]] = None) -> List[str]:
        """Persist selected models to disk using joblib."""
        saved_paths: List[str] = []
        if not models:
            return saved_paths

        if not JOBLIB_AVAILABLE:
            self.logger.warning("joblib not available - skipping model persistence")
            return saved_paths

        self._ensure_directory(destination)
        for model_name, model in models.items():
            if name_filter and not any(token in model_name.lower() for token in name_filter):
                continue
            try:
                model_path = destination / f"{model_name}.joblib"
                joblib.dump(model, model_path)
                saved_paths.append(str(model_path))
            except Exception as exc:  # pragma: no cover - serialization edge cases
                self.logger.warning(f"Failed to persist model {model_name}: {exc}")
        return saved_paths

    def _filter_predictions_by_threshold(
        self,
        predictions: Optional[pd.DataFrame],
        threshold: float
    ) -> Optional[pd.DataFrame]:
        """Filter predictions based on absolute value threshold across any column."""
        if predictions is None or predictions.empty:
            return predictions

        mask = (predictions.abs() >= threshold).any(axis=1)
        return predictions.loc[mask]
    
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
            # ==================== ANALYST PIPELINE (15m) ====================
            if config.train_analyst:
                self.logger.info('=' * 80)
                self.logger.info('🎯 ANALYST PIPELINE (15m timeframe - IF we trade)')
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
            
            # ==================== TACTICIAN PIPELINE (5m) ====================
            if config.train_tactician:
                self.logger.info('=' * 80)
                self.logger.info('🎯 TACTICIAN PIPELINE (5m timeframe - WHEN we trade)')
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
        """Execute Analyst pre-ML orchestration (15m timeframe)."""
        result = SubPipelineResult(
            sub_pipeline_name='analyst_pre_ml_orchestration',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            if not self.analyst_pre_ml:
                raise RuntimeError("Analyst pre-ML orchestrator not available")

            self.logger.info('🔧 Executing Analyst Pre-ML Orchestration (15m)...')

            training_data = self._load_input_dataframe(
                config,
                role='analyst',
                timeframe=config.analyst_timeframe,
                kind='training_data',
                required=True
            )

            regime_assignments = self._load_input_dataframe(
                config,
                role='analyst',
                timeframe=config.analyst_timeframe,
                kind='regime_assignments',
                required=False
            )

            orchestration_result = await self.analyst_pre_ml.orchestrate(
                training_data=training_data,
                regime_assignments=regime_assignments,
            )

            result.success = orchestration_result.success
            result.status = SubPipelineStatus.COMPLETED if orchestration_result.success else SubPipelineStatus.FAILED
            result.error_message = orchestration_result.error_message

            if not orchestration_result.success:
                return result

            final_features = orchestration_result.final_features
            if final_features is None:
                raise ValueError('Analyst pre-ML orchestration did not return final features')

            regime_features = self._collect_regime_features(orchestration_result.feature_selection_result or {})
            metadata = {
                'selected_features': orchestration_result.selected_feature_names or [],
                'feature_count': orchestration_result.final_feature_count,
                'total_samples': orchestration_result.total_samples,
                'regime_features': regime_features,
                'regime_assignments_provided': regime_assignments is not None
            }

            output_dir = Path(config.output_directory) / 'analyst' / 'pre_ml'
            saved_files: List[str] = []
            saved_files.append(self._save_dataframe_artifact(final_features, output_dir / 'final_features.parquet'))
            if regime_features is not None and not regime_features.empty:
                saved_files.append(self._save_dataframe_artifact(regime_features, output_dir / 'regime_features.parquet'))
                metadata['regime_feature_columns'] = list(regime_features.columns)

            saved_files.append(self._save_json_artifact({
                'selected_features': metadata['selected_features'],
                'feature_count': metadata['feature_count'],
                'total_samples': metadata['total_samples'],
                'regime_feature_columns': metadata.get('regime_feature_columns', [])
            }, output_dir / 'metadata.json'))

            result.artifacts = {
                'final_features': final_features,
                'selected_features': metadata['selected_features'],
                'feature_count': metadata['feature_count'],
                'regime_features': regime_features,
                'metadata': metadata
            }
            result.output_files = saved_files

            self._current_pipeline_state['analyst_training_frame'] = final_features
            self._current_pipeline_state['analyst_regime_features'] = regime_features
            self._current_pipeline_state['analyst_pre_ml_metadata'] = metadata

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

            final_features = pre_ml_result.artifacts.get('final_features')
            if final_features is None or final_features.empty:
                raise ValueError('Analyst pre-ML artifacts missing final features')

            selected_features = list(pre_ml_result.artifacts.get('selected_features', []))
            if not selected_features:
                raise ValueError('Analyst pre-ML artifacts missing selected features')

            regime_features = self._collect_regime_features(pre_ml_result.artifacts)
            if regime_features is None:
                regime_features = self._current_pipeline_state.get('analyst_regime_features')

            training_frame = self._merge_regime_features(final_features, regime_features)
            feature_columns = list(dict.fromkeys(selected_features + ([
                col for col in (regime_features.columns if isinstance(regime_features, pd.DataFrame) else [])
                if col not in selected_features
            ])))

            target_columns = self._get_target_columns(training_frame)

            output_dir = Path(config.output_directory) / 'analyst' / 'models'
            saved_files: List[str] = []
            saved_files.append(self._save_dataframe_artifact(training_frame, output_dir / 'training_frame.parquet'))

            training_result = await self.analyst_training.train_analyst_models(
                training_data=training_frame,
                feature_columns=feature_columns,
                target_columns=target_columns,
            )

            result.success = training_result.base_training_completed
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED

            base_models = training_result.base_models or {}
            base_metrics = training_result.base_training_metrics or {}
            ensemble_models = training_result.ensemble_models or {}
            ensemble_metrics = training_result.ensemble_metrics or {}

            predictions_df = self._generate_model_predictions(
                ensemble_models or base_models,
                training_frame[feature_columns]
            )

            metrics_payload = {
                'base': base_metrics,
                'ensemble': ensemble_metrics,
                'feature_columns': feature_columns,
                'target_columns': target_columns
            }
            saved_files.append(self._save_json_artifact(metrics_payload, output_dir / 'metrics.json'))

            if not predictions_df.empty:
                saved_files.append(self._save_dataframe_artifact(predictions_df, output_dir / 'ensemble_predictions.parquet'))

            model_paths = self._persist_models(
                base_models,
                destination=output_dir / 'nas_tas_models',
                name_filter=['nas', 'tas']
            )
            saved_files.extend(model_paths)

            result.artifacts = {
                'base_models': base_models,
                'ensemble_models': ensemble_models,
                'metrics': metrics_payload,
                'predictions': predictions_df,
                'feature_columns': feature_columns,
                'target_columns': target_columns
            }
            result.output_files = saved_files

            self._current_pipeline_state['analyst_training_frame'] = training_frame
            self._current_pipeline_state['analyst_feature_columns'] = feature_columns
            self._current_pipeline_state['analyst_models'] = base_models
            self._current_pipeline_state['analyst_ensemble_models'] = ensemble_models
            self._current_pipeline_state['analyst_ensemble_predictions'] = predictions_df

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
            if not self.analyst_training or not getattr(self.analyst_training, 'ensemble_trainer', None):
                raise RuntimeError("Analyst ensemble trainer not available")

            self.logger.info('🔄 Executing Analyst Ensemble Training...')

            training_frame: Optional[pd.DataFrame] = self._current_pipeline_state.get('analyst_training_frame')
            feature_columns: List[str] = models_result.artifacts.get('feature_columns') or self._current_pipeline_state.get('analyst_feature_columns', [])
            base_models = models_result.artifacts.get('base_models') or self._current_pipeline_state.get('analyst_models')

            if training_frame is None or training_frame.empty:
                raise ValueError('Analyst training frame unavailable for ensemble step')
            if not feature_columns:
                raise ValueError('Analyst feature columns unavailable for ensemble step')
            if not base_models:
                raise ValueError('Analyst base models unavailable for ensemble step')

            target_columns: List[str] = models_result.artifacts.get('target_columns') or self._get_target_columns(training_frame)

            ensemble_trainer = self.analyst_training.ensemble_trainer
            ensemble_output = await ensemble_trainer.train_analyst_ensemble(
                training_data=training_frame,
                base_models=base_models,
                feature_columns=feature_columns,
                target_columns=target_columns,
            )

            ensemble_models = ensemble_output.get('models', {})
            ensemble_metrics = ensemble_output.get('metrics', {})
            ensemble_metadata = ensemble_output.get('metadata', {})

            predictions_df = self._generate_model_predictions(ensemble_models, training_frame[feature_columns])

            output_dir = Path(config.output_directory) / 'analyst' / 'ensemble'
            saved_files: List[str] = []
            if not predictions_df.empty:
                saved_files.append(self._save_dataframe_artifact(predictions_df, output_dir / 'ensemble_predictions.parquet'))
            saved_files.append(self._save_json_artifact({
                'metrics': ensemble_metrics,
                'metadata': ensemble_metadata
            }, output_dir / 'ensemble_metrics.json'))

            result.success = True
            result.status = SubPipelineStatus.COMPLETED
            result.artifacts = {
                'ensemble_models': ensemble_models,
                'ensemble_metrics': ensemble_metrics,
                'metadata': ensemble_metadata,
                'predictions': predictions_df
            }
            result.output_files = saved_files

            self._current_pipeline_state['analyst_ensemble_models'] = ensemble_models
            if not predictions_df.empty:
                self._current_pipeline_state['analyst_ensemble_predictions'] = predictions_df

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
        """Execute Tactician pre-ML orchestration (5m timeframe, filtered on Analyst signals)."""
        result = SubPipelineResult(
            sub_pipeline_name='tactician_pre_ml_orchestration',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            if not self.tactician_pre_ml:
                raise RuntimeError("Tactician pre-ML orchestrator not available")

            self.logger.info('🔧 Executing Tactician Pre-ML Orchestration (5m, filtered)...')

            training_data = self._load_input_dataframe(
                config,
                role='tactician',
                timeframe=config.tactician_timeframe,
                kind='training_data',
                required=True
            )

            analyst_predictions_df = analyst_predictions
            if analyst_predictions_df is None:
                analyst_predictions_df = self._current_pipeline_state.get('analyst_ensemble_predictions')

            if isinstance(analyst_predictions_df, pd.DataFrame):
                analyst_predictions_df = self._filter_predictions_by_threshold(
                    analyst_predictions_df,
                    config.analyst_confidence_threshold
                )

            regime_assignments = self._load_input_dataframe(
                config,
                role='tactician',
                timeframe=config.tactician_timeframe,
                kind='regime_assignments',
                required=False
            )

            orchestration_result = await self.tactician_pre_ml.orchestrate(
                training_data=training_data,
                analyst_predictions=analyst_predictions_df,
                regime_assignments=regime_assignments,
            )

            result.success = orchestration_result.success
            result.status = SubPipelineStatus.COMPLETED if orchestration_result.success else SubPipelineStatus.FAILED
            result.error_message = orchestration_result.error_message

            if not orchestration_result.success:
                return result

            final_features = orchestration_result.final_features
            if final_features is None:
                raise ValueError('Tactician pre-ML orchestration did not return final features')

            regime_features = self._collect_regime_features(orchestration_result.feature_selection_result or {})
            metadata = {
                'selected_features': orchestration_result.selected_feature_names or [],
                'feature_count': orchestration_result.final_feature_count,
                'filter_ratio': orchestration_result.filter_ratio,
                'total_samples_before_filter': orchestration_result.total_samples_before_filter,
                'total_samples_after_filter': orchestration_result.total_samples_after_filter,
                'regime_features': regime_features
            }

            output_dir = Path(config.output_directory) / 'tactician' / 'pre_ml'
            saved_files: List[str] = []
            saved_files.append(self._save_dataframe_artifact(final_features, output_dir / 'final_features.parquet'))
            if regime_features is not None and not regime_features.empty:
                saved_files.append(self._save_dataframe_artifact(regime_features, output_dir / 'regime_features.parquet'))
                metadata['regime_feature_columns'] = list(regime_features.columns)

            saved_files.append(self._save_json_artifact({
                'selected_features': metadata['selected_features'],
                'feature_count': metadata['feature_count'],
                'filter_ratio': metadata['filter_ratio'],
                'total_samples_before_filter': metadata['total_samples_before_filter'],
                'total_samples_after_filter': metadata['total_samples_after_filter'],
                'regime_feature_columns': metadata.get('regime_feature_columns', [])
            }, output_dir / 'metadata.json'))

            result.artifacts = {
                'final_features': final_features,
                'selected_features': metadata['selected_features'],
                'feature_count': metadata['feature_count'],
                'filter_ratio': metadata['filter_ratio'],
                'regime_features': regime_features,
                'metadata': metadata
            }
            result.output_files = saved_files

            self._current_pipeline_state['tactician_training_frame'] = final_features
            self._current_pipeline_state['tactician_regime_features'] = regime_features
            self._current_pipeline_state['tactician_pre_ml_metadata'] = metadata

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

            final_features = pre_ml_result.artifacts.get('final_features')
            if final_features is None or final_features.empty:
                raise ValueError('Tactician pre-ML artifacts missing final features')

            selected_features = list(pre_ml_result.artifacts.get('selected_features', []))
            if not selected_features:
                raise ValueError('Tactician pre-ML artifacts missing selected features')

            regime_features = self._collect_regime_features(pre_ml_result.artifacts)
            if regime_features is None:
                regime_features = self._current_pipeline_state.get('tactician_regime_features')

            analyst_predictions_df: Optional[pd.DataFrame] = self._current_pipeline_state.get('analyst_ensemble_predictions')
            if isinstance(analyst_predictions_df, pd.DataFrame) and not analyst_predictions_df.empty:
                analyst_predictions_df = analyst_predictions_df.reindex(final_features.index).add_prefix('analyst_')

            training_frame = self._merge_regime_features(final_features, regime_features)
            if isinstance(analyst_predictions_df, pd.DataFrame):
                training_frame = training_frame.join(analyst_predictions_df, how='left')

            additional_features: List[str] = []
            if isinstance(regime_features, pd.DataFrame):
                additional_features.extend([
                    col for col in regime_features.columns if col not in selected_features
                ])
            if isinstance(analyst_predictions_df, pd.DataFrame):
                additional_features.extend(list(analyst_predictions_df.columns))

            feature_columns = list(dict.fromkeys(selected_features + additional_features))
            target_columns = self._get_target_columns(training_frame)

            output_dir = Path(config.output_directory) / 'tactician' / 'models'
            saved_files: List[str] = []
            saved_files.append(self._save_dataframe_artifact(training_frame, output_dir / 'training_frame.parquet'))

            training_result = await self.tactician_training.train_tactician_models(
                training_data=training_frame,
                feature_columns=feature_columns,
                target_columns=target_columns,
            )

            result.success = training_result.base_training_completed
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED

            base_models = training_result.base_models or {}
            base_metrics = training_result.base_training_metrics or {}
            ensemble_models = training_result.ensemble_models or {}
            ensemble_metrics = training_result.ensemble_metrics or {}

            predictions_df = self._generate_model_predictions(base_models, training_frame[feature_columns])

            metrics_payload = {
                'base': base_metrics,
                'ensemble': ensemble_metrics,
                'feature_columns': feature_columns,
                'target_columns': target_columns
            }
            saved_files.append(self._save_json_artifact(metrics_payload, output_dir / 'metrics.json'))

            if not predictions_df.empty:
                saved_files.append(self._save_dataframe_artifact(predictions_df, output_dir / 'base_model_predictions.parquet'))

            result.artifacts = {
                'base_models': base_models,
                'ensemble_models': ensemble_models,
                'metrics': metrics_payload,
                'base_predictions': predictions_df,
                'feature_columns': feature_columns,
                'target_columns': target_columns
            }
            result.output_files = saved_files

            self._current_pipeline_state['tactician_training_frame'] = training_frame
            self._current_pipeline_state['tactician_feature_columns'] = feature_columns
            self._current_pipeline_state['tactician_models'] = base_models
            self._current_pipeline_state['tactician_base_predictions'] = predictions_df

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
            if not self.tactician_training or not getattr(self.tactician_training, 'ensemble_trainer', None):
                raise RuntimeError("Tactician ensemble trainer not available")

            self.logger.info('🔄 Executing Tactician Ensemble Training...')

            training_frame: Optional[pd.DataFrame] = self._current_pipeline_state.get('tactician_training_frame')
            feature_columns: List[str] = models_result.artifacts.get('feature_columns') or self._current_pipeline_state.get('tactician_feature_columns', [])
            base_models = models_result.artifacts.get('base_models') or self._current_pipeline_state.get('tactician_models')
            base_predictions = models_result.artifacts.get('base_predictions') or self._current_pipeline_state.get('tactician_base_predictions')

            if training_frame is None or training_frame.empty:
                raise ValueError('Tactician training frame unavailable for ensemble step')
            if not feature_columns:
                raise ValueError('Tactician feature columns unavailable for ensemble step')
            if not base_models:
                raise ValueError('Tactician base models unavailable for ensemble step')

            target_columns: List[str] = models_result.artifacts.get('target_columns') or self._get_target_columns(training_frame)

            ensemble_trainer = self.tactician_training.ensemble_trainer
            ensemble_output = await ensemble_trainer.train_tactician_ensemble(
                training_data=training_frame,
                base_models=base_models,
                feature_columns=feature_columns,
                target_columns=target_columns,
                base_model_predictions=base_predictions,
                analyst_predictions=self._current_pipeline_state.get('analyst_ensemble_predictions')
            )

            ensemble_models = ensemble_output.get('models', {})
            ensemble_metrics = ensemble_output.get('metrics', {})
            ensemble_metadata = ensemble_output.get('metadata', {})

            predictions_df = self._generate_model_predictions(ensemble_models, training_frame[feature_columns])

            output_dir = Path(config.output_directory) / 'tactician' / 'ensemble'
            saved_files: List[str] = []
            if not predictions_df.empty:
                saved_files.append(self._save_dataframe_artifact(predictions_df, output_dir / 'ensemble_predictions.parquet'))
            saved_files.append(self._save_json_artifact({
                'metrics': ensemble_metrics,
                'metadata': ensemble_metadata
            }, output_dir / 'ensemble_metrics.json'))

            result.success = True
            result.status = SubPipelineStatus.COMPLETED
            result.artifacts = {
                'ensemble_models': ensemble_models,
                'ensemble_metrics': ensemble_metrics,
                'metadata': ensemble_metadata,
                'predictions': predictions_df
            }
            result.output_files = saved_files

            self._current_pipeline_state['tactician_ensemble_models'] = ensemble_models
            if not predictions_df.empty:
                self._current_pipeline_state['tactician_ensemble_predictions'] = predictions_df

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
        if sub_pipeline_name == 'analyst_pre_ml_orchestration':
            return await self._execute_analyst_pre_ml_orchestration(config)
        elif sub_pipeline_name == 'analyst_models_training':
            # Need pre-ML result, TODO: load from artifacts
            raise NotImplementedError("Individual execution not yet supported, use full pipeline")
        elif sub_pipeline_name == 'analyst_ensemble_training':
            raise NotImplementedError("Individual execution not yet supported, use full pipeline")
        elif sub_pipeline_name == 'tactician_pre_ml_orchestration':
            return await self._execute_tactician_pre_ml_orchestration(config, None)
        elif sub_pipeline_name == 'tactician_models_training':
            raise NotImplementedError("Individual execution not yet supported, use full pipeline")
        elif sub_pipeline_name == 'tactician_ensemble_training':
            raise NotImplementedError("Individual execution not yet supported, use full pipeline")
        else:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
    
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
