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

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import orchestration and training steps
try:
    from .analyst_pre_ml_orchestration import (
        AnalystPreMLOrchestrator, AnalystPreMLConfig, AnalystPreMLResult
    )
    ANALYST_PRE_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_pre_ml_orchestration not available: {e}")
    ANALYST_PRE_ML_AVAILABLE = False

try:
    from .analyst_training_pipeline import (
        AnalystTrainingPipeline, AnalystTrainingPipelineConfig, AnalystTrainingPipelineResult
    )
    ANALYST_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: analyst_training_pipeline not available: {e}")
    ANALYST_TRAINING_AVAILABLE = False

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
            
            # Execute orchestration
            orchestration_result = await self.analyst_pre_ml.orchestrate(
                training_data=None,  # TODO: Load from artifacts
                regime_assignments=None,  # TODO: Load from market_analysis
            )

            result.success = orchestration_result.success
            result.status = SubPipelineStatus.COMPLETED if orchestration_result.success else SubPipelineStatus.FAILED
            result.error_message = orchestration_result.error_message
            result.artifacts = {
                'final_features': orchestration_result.final_features,
                'selected_features': orchestration_result.selected_feature_names,
                'feature_count': orchestration_result.final_feature_count
            }
            result.metadata = {
                'total_samples': getattr(orchestration_result, 'total_samples', None),
                'final_feature_count': getattr(orchestration_result, 'final_feature_count', None),
                'selection_phase': getattr(orchestration_result, 'phase', None)
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'analyst_pre_ml_orchestration',
                    result.artifacts,
                    result.metadata
                )

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
            
            # Execute training
            training_result = await self.analyst_training.train_analyst_models(
                training_data=pre_ml_result.artifacts.get('final_features'),
                feature_columns=pre_ml_result.artifacts.get('selected_features', []),
                target_columns=['target_long', 'target_short'],  # TODO: Get from config
            )

            result.success = training_result.base_training_completed
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED
            result.artifacts = {
                'base_models': training_result.base_models,
                'metrics': training_result.base_training_metrics
            }
            result.metadata = {
                'training_summary': getattr(training_result, 'training_summary', None),
                'metrics': training_result.base_training_metrics
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'analyst_models_training',
                    result.artifacts,
                    result.metadata
                )

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
            if not self.analyst_training:
                raise RuntimeError("Analyst training pipeline not available")
            
            self.logger.info('🔄 Executing Analyst Ensemble Training...')
            
            # Ensemble training is handled within the analyst_training pipeline
            # The result is already available from the models training step
            result.success = True
            result.status = SubPipelineStatus.COMPLETED
            result.artifacts = {
                'ensemble_models': models_result.artifacts.get('base_models'),  # Placeholder
                'predictions': None  # TODO: Generate predictions
            }
            result.metadata = {
                'source_models_available': bool(models_result.artifacts.get('base_models'))
            }

            result.output_files = self._save_step_artifacts(
                config,
                'analyst_ensemble_training',
                result.artifacts,
                result.metadata
            )

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
            
            # Execute orchestration with Analyst filtering
            orchestration_result = await self.tactician_pre_ml.orchestrate(
                training_data=None,  # TODO: Load from artifacts
                analyst_predictions=analyst_predictions,
                regime_assignments=None,  # TODO: Load from market_analysis
            )
            
            result.success = orchestration_result.success
            result.status = SubPipelineStatus.COMPLETED if orchestration_result.success else SubPipelineStatus.FAILED
            result.error_message = orchestration_result.error_message
            result.artifacts = {
                'final_features': orchestration_result.final_features,
                'selected_features': orchestration_result.selected_feature_names,
                'feature_count': orchestration_result.final_feature_count,
                'filter_ratio': orchestration_result.filter_ratio
            }
            result.metadata = {
                'total_samples': getattr(orchestration_result, 'total_samples', None),
                'filter_ratio': getattr(orchestration_result, 'filter_ratio', None)
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'tactician_pre_ml_orchestration',
                    result.artifacts,
                    result.metadata
                )

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
            
            # Execute training
            training_result = await self.tactician_training.train_tactician_models(
                training_data=pre_ml_result.artifacts.get('final_features'),
                feature_columns=pre_ml_result.artifacts.get('selected_features', []),
                target_columns=['target_long', 'target_short'],  # TODO: Get from config
            )

            result.success = training_result.base_training_completed
            result.status = SubPipelineStatus.COMPLETED if result.success else SubPipelineStatus.FAILED
            result.artifacts = {
                'base_models': training_result.base_models,
                'metrics': training_result.base_training_metrics
            }
            result.metadata = {
                'training_summary': getattr(training_result, 'training_summary', None),
                'metrics': training_result.base_training_metrics
            }

            if result.success:
                result.output_files = self._save_step_artifacts(
                    config,
                    'tactician_models_training',
                    result.artifacts,
                    result.metadata
                )

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
            if not self.tactician_training:
                raise RuntimeError("Tactician training pipeline not available")
            
            self.logger.info('🔄 Executing Tactician Ensemble Training...')
            
            # Ensemble training is handled within the tactician_training pipeline
            # The result is already available from the models training step
            result.success = True
            result.status = SubPipelineStatus.COMPLETED
            result.artifacts = {
                'ensemble_models': models_result.artifacts.get('base_models'),  # Placeholder
                'predictions': None  # TODO: Generate predictions
            }
            result.metadata = {
                'source_models_available': bool(models_result.artifacts.get('base_models'))
            }

            result.output_files = self._save_step_artifacts(
                config,
                'tactician_ensemble_training',
                result.artifacts,
                result.metadata
            )

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
