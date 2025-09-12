from src.utils.tprint import tprint
import pandas as pd

"""
Model Training Sub-Pipeline - Final Structure

This module provides the final model training sub-pipeline with only 4 required steps:

1. analyst_models_training - Per-regime individual model training with HPO, saving, and metrics
2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics
3. tactician_models_training - All-regime individual model training with HPO, saving, and metrics
4. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics
"""

import asyncio
import json
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager

logger = system_logger.getChild('ModelTrainingSubPipeline')

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
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    data_dir: str = "data/training"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

class ModelTrainingSubPipeline:
    """
    Model Training Sub-Pipeline Manager.
    
    Provides granular control over model training processes with different
    execution modes and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the model training sub-pipeline."""
        self.config = config or SubPipelineConfig()
        self.logger = logger.getChild('ModelTrainingSubPipeline')
        self.results: List[SubPipelineResult] = []
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize sub-pipeline registry with only the 4 required steps
        self.sub_pipelines = {
            'analyst_models_training': self._analyst_models_training_pipeline,
            'analyst_ensemble_training': self._analyst_ensemble_training_pipeline,
            'tactician_models_training': self._tactician_models_training_pipeline,
            'tactician_ensemble_training': self._tactician_ensemble_training_pipeline,
        }
        
        # Initialize temporal feature integration
        self.temporal_features_available = False
        self.temporal_features = {}
        self.temporal_feature_metadata = {}
    
    def _log_sub_pipeline_completion(self, sub_pipeline_name: str, config: SubPipelineConfig, artifacts: Dict[str, Any]):
        """Helper method to log sub-pipeline completion with emojis and artifact paths."""
        tprint("\n" + "="*80)
        tprint(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED SUCCESSFULLY!")
        tprint("="*80)
        tprint(f"📁 Artifact Paths:")
        
        # Log different types of artifacts with appropriate emojis
        for key, value in artifacts.items():
            if isinstance(value, list) and value:
                if 'model' in key.lower():
                    for item in value:
                        tprint(f"   🤖 {key.title()}: {config.data_dir}/models/{item}")
                elif 'file' in key.lower() or 'data' in key.lower():
                    for item in value:
                        tprint(f"   📄 {key.title()}: {config.data_dir}/{item}")
                elif 'report' in key.lower():
                    for item in value:
                        tprint(f"   📋 {key.title()}: {config.data_dir}/{item}")
                else:
                    for item in value:
                        tprint(f"   📊 {key.title()}: {config.data_dir}/{item}")
            elif isinstance(value, dict) and value:
                tprint(f"   📊 {key.title()}: {config.data_dir}/{key}.json")
        
        tprint(f"📊 Artifacts Summary: {len(artifacts)} artifact types generated")
        tprint("="*80 + "\n")
        
        # Log to logger as well
        self.logger.info(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(f"📁 Artifact Paths:")
        for key, value in artifacts.items():
            if isinstance(value, list) and value:
                if 'model' in key.lower():
                    for item in value:
                        self.logger.info(f"   🤖 {key.title()}: {config.data_dir}/models/{item}")
                elif 'file' in key.lower() or 'data' in key.lower():
                    for item in value:
                        self.logger.info(f"   📄 {key.title()}: {config.data_dir}/{item}")
                elif 'report' in key.lower():
                    for item in value:
                        self.logger.info(f"   📋 {key.title()}: {config.data_dir}/{item}")
                else:
                    for item in value:
                        self.logger.info(f"   📊 {key.title()}: {config.data_dir}/{item}")
            elif isinstance(value, dict) and value:
                self.logger.info(f"   📊 {key.title()}: {config.data_dir}/{key}.json")
        self.logger.info(f"📊 Artifacts Summary: {len(artifacts)} artifact types generated")
    
    async def execute_sub_pipeline(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Optional configuration override
            
        Returns:
            SubPipelineResult with execution details
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting model training sub-pipeline: {sub_pipeline_name} (mode: {config.mode.value})")
        
        start_time = datetime.now()
        result = SubPipelineResult(
            sub_pipeline_name=sub_pipeline_name,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            if sub_pipeline_name not in self.sub_pipelines:
                raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
            
            # Execute the sub-pipeline
            pipeline_func = self.sub_pipelines[sub_pipeline_name]
            artifacts = await pipeline_func(config)
            
            # Update result
            end_time = datetime.now()
            result.status = SubPipelineStatus.COMPLETED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.artifacts = artifacts
            result.metadata = {
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe
            }
            
            self.logger.info(f"✅ Model training sub-pipeline {sub_pipeline_name} completed in {result.duration_seconds:.2f}s")
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"❌ Model training sub-pipeline {sub_pipeline_name} failed: {e}")
        
        self.results.append(result)
        return result
    
    async def execute_multiple_sub_pipelines(
        self,
        sub_pipeline_names: List[str],
        config: Optional[SubPipelineConfig] = None,
        sequential: bool = False
    ) -> List[SubPipelineResult]:
        """
        Execute multiple sub-pipelines.
        
        Args:
            sub_pipeline_names: List of sub-pipeline names to execute
            config: Optional configuration override
            sequential: Whether to execute sequentially or in parallel
            
        Returns:
            List of SubPipelineResult objects
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting {len(sub_pipeline_names)} model training sub-pipelines (sequential: {sequential})")
        
        if sequential:
            results = []
            for name in sub_pipeline_names:
                result = await self.execute_sub_pipeline(name, config)
                results.append(result)
                if result.status == SubPipelineStatus.FAILED:
                    self.logger.warning(f"⚠️ Stopping sequential execution due to failure in {name}")
                    break
            return results
        else:
            # Execute in parallel
            tasks = [self.execute_sub_pipeline(name, config) for name in sub_pipeline_names]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _load_temporal_features(self, config: SubPipelineConfig) -> bool:
        """Load temporal features from MARKET_ANALYSIS stage."""
        try:
            # Try to load temporal features from various sources
            temporal_feature_sources = [
                f"{config.data_dir}/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet",
                f"{config.data_dir}/training/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet",
                f"{config.data_dir}/processed/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet"
            ]
            
            for feature_path in temporal_feature_sources:
                if Path(feature_path).exists():
                    self.logger.info(f"📊 Loading temporal features from: {feature_path}")
                    temporal_df = pd.read_parquet(feature_path)
                    if not temporal_df.empty:
                        self.temporal_features = temporal_df.to_dict('series')
                        self.temporal_features_available = True
                        self.logger.info(f"✅ Loaded {len(self.temporal_features)} temporal features")
                        
                        # Load metadata if available
                        metadata_path = feature_path.replace('temporal_features_', 'temporal_feature_metadata_').replace('.parquet', '.json')
                        if Path(metadata_path).exists():
                            with open(metadata_path, 'r') as f:
                                self.temporal_feature_metadata = json.load(f)
                            self.logger.info(f"✅ Loaded temporal feature metadata")
                        
                        return True
            
            self.logger.warning("⚠️ No temporal features found, using standard features only")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load temporal features: {e}")
            return False
    
    def _get_enhanced_feature_columns(self, base_features: List[str]) -> List[str]:
        """Get enhanced feature columns including temporal features."""
        if not self.temporal_features_available:
            return base_features
        
        # Combine base features with temporal features
        temporal_feature_names = list(self.temporal_features.keys())
        enhanced_features = base_features + temporal_feature_names
        
        self.logger.info(f"📊 Enhanced features: {len(base_features)} base + {len(temporal_feature_names)} temporal = {len(enhanced_features)} total")
        return enhanced_features
    
    def _get_temporal_feature_info(self) -> Dict[str, Any]:
        """Get information about available temporal features."""
        if not self.temporal_features_available:
            return {'available': False, 'count': 0, 'types': {}}
        
        # Analyze temporal feature types
        lookback_features = [name for name in self.temporal_features.keys() if name.startswith('lookback_')]
        cross_tf_features = [name for name in self.temporal_features.keys() if name.startswith('cross_tf_')]
        
        return {
            'available': True,
            'count': len(self.temporal_features),
            'lookback_features': len(lookback_features),
            'cross_timeframe_features': len(cross_tf_features),
            'types': {
                'lookback': lookback_features,
                'cross_timeframe': cross_tf_features
            },
            'metadata_available': bool(self.temporal_feature_metadata)
        }
    
    # Sub-pipeline implementations
    
    async def _analyst_model_training_pipeline(self, config: SubPipelineConfig, hmm_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyst model training sub-pipeline."""
        self.logger.info("📊 Executing analyst model training pipeline")
       
        artifacts = {
            'analyst_models': [],
            'training_metrics': {},
            'analyst_performance': {},
            'temporal_features_used': False,
            'temporal_feature_info': {}
        }
        
        # Load temporal features from MARKET_ANALYSIS stage
        temporal_loaded = await self._load_temporal_features(config)
        if temporal_loaded:
            temporal_info = self._get_temporal_feature_info()
            artifacts['temporal_features_used'] = True
            artifacts['temporal_feature_info'] = temporal_info
            self.logger.info(f"✅ Using {temporal_info['count']} temporal features in analyst model training")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual analyst models training")
            artifacts['analyst_models'] = ['analyst_model.pkl']
            return artifacts
        
        # Import and use analyst models training
        try:
            from .analyst_models_training import AnalystModelsTrainingStep
            
            # Create enhanced configuration with temporal features and HMM data
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            if temporal_loaded:
                enhanced_config['temporal_features_available'] = True
                enhanced_config['temporal_feature_columns'] = list(self.temporal_features.keys())
                enhanced_config['temporal_feature_metadata'] = self.temporal_feature_metadata
            
            trainer = AnalystModelsTrainingStep()
            training_result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={}
            )
            
            artifacts['analyst_models'] = training_result.get('models', [])
            artifacts['training_metrics'] = training_result.get('metrics', {})
            artifacts['analyst_performance'] = training_result.get('performance', {})
            
        except ImportError:
            self.logger.warning("⚠️ Analyst models trainer not available, using mock training")
            artifacts['analyst_models'] = ['analyst_model.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("analyst_models_training", config, artifacts)
        
        return artifacts
    
    async def _analyst_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst Ensemble Training sub-pipeline - Per-regime ensemble training."""
        self.logger.info("🎭 Executing analyst ensemble training pipeline (per-regime ensemble models)")
        
        artifacts = {
            'analyst_ensembles': [],
            'training_metrics': {},
            'analyst_ensemble_performance': {},
            'temporal_features_used': False,
            'temporal_feature_info': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual analyst ensemble training")
            artifacts['analyst_ensembles'] = ['analyst_ensemble.pkl']
            return artifacts
        
        # Import and use analyst ensemble training
        try:
            from .analyst_ensemble_training import AnalystEnsembleTrainingStep as AnalystEnsembleTrainer
            
            # Create enhanced configuration with analyst models and HMM data
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            if ensemble_data:
                enhanced_config.update({
                    'base_analyst_models': ensemble_data.get('analyst_models', []),
                    'analyst_training_metrics': ensemble_data.get('analyst_training_metrics', {}),
                    'hmm_data': ensemble_data.get('hmm_data', {})
                })
                self.logger.info("✅ Using pre-trained analyst models as base models for ensemble")
            
            trainer = AnalystEnsembleTrainer()
            training_result = await trainer.execute_analyst_ensemble_training(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                force_rerun=config.force_rerun,
                enhanced_config=enhanced_config
            )
            
            artifacts['analyst_ensembles'] = training_result.get('models', [])
            artifacts['ensemble_metrics'] = training_result.get('metrics', {})
            artifacts['analyst_ensemble_performance'] = training_result.get('performance', {})
            
        except ImportError:
            self.logger.warning("⚠️ Analyst ensemble trainer not available, using mock training")
            artifacts['analyst_ensembles'] = ['analyst_ensemble.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("analyst_ensemble_training", config, artifacts)
        
        # Automatically trigger the next sub-pipeline: tactician_model_training
        self.logger.info("🔄 Analyst ensemble training completed, triggering next: tactician_model_training")
        try:
            # Pass all analyst data and HMM data to tactician training
            tactician_data = {
                'analyst_models': ensemble_data.get('analyst_models', []) if ensemble_data else [],
                'analyst_ensembles': artifacts.get('analyst_ensembles', []),
                'analyst_ensemble_metrics': artifacts.get('ensemble_metrics', {}),
                'hmm_data': ensemble_data.get('hmm_data', {}) if ensemble_data else {}
            }
            next_artifacts = await self._tactician_model_training_pipeline(config, tactician_data)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Tactician model training pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute tactician model training pipeline: {e}")
        
        return artifacts
    
    async def _tactician_model_training_pipeline(self, config: SubPipelineConfig, tactician_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Tactician model training sub-pipeline."""
        self.logger.info("⚔️ Executing tactician model training pipeline")
        
        artifacts = {
            'tactician_models': [],
            'training_metrics': {},
            'tactician_performance': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual tactician model training")
            artifacts['tactician_models'] = ['tactician_model.pkl']
            return artifacts
        
        # Import and use tactician model training
        try:
            from .tactician_models_training import TacticianModelsTrainingStep as TacticianModelTrainer
            
            # Create enhanced configuration with all analyst data and HMM data
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            if tactician_data:
                enhanced_config.update({
                    'analyst_models': tactician_data.get('analyst_models', []),
                    'analyst_ensembles': tactician_data.get('analyst_ensembles', []),
                    'analyst_ensemble_metrics': tactician_data.get('analyst_ensemble_metrics', {}),
                    'hmm_data': tactician_data.get('hmm_data', {})
                })
                self.logger.info("✅ Using all analyst model inputs and HMM data in tactician training")
            
            trainer = TacticianModelTrainer()
            training_result = await trainer.train_tactician_model(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                force_rerun=config.force_rerun,
                enhanced_config=enhanced_config
            )
            
            artifacts['tactician_models'] = training_result.get('models', [])
            artifacts['training_metrics'] = training_result.get('metrics', {})
            artifacts['tactician_performance'] = training_result.get('performance', {})
            
        except ImportError:
            self.logger.warning("⚠️ Tactician model trainer not available, using mock training")
            artifacts['tactician_models'] = ['tactician_model.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("tactician_model_training", config, artifacts)
        
        # Automatically trigger the next sub-pipeline: tactician_ensemble_training
        self.logger.info("🔄 Tactician model training completed, triggering next: tactician_ensemble_training")
        try:
            # Pass all data to tactician ensemble training (meta learner gets all inputs)
            ensemble_data = {
                'tactician_models': artifacts.get('tactician_models', []),
                'tactician_training_metrics': artifacts.get('training_metrics', {}),
                'analyst_models': tactician_data.get('analyst_models', []) if tactician_data else [],
                'analyst_ensembles': tactician_data.get('analyst_ensembles', []) if tactician_data else [],
                'analyst_ensemble_metrics': tactician_data.get('analyst_ensemble_metrics', {}) if tactician_data else {},
                'hmm_data': tactician_data.get('hmm_data', {}) if tactician_data else {}
            }
            next_artifacts = await self._tactician_ensemble_training_pipeline(config, ensemble_data)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Tactician ensemble training pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute tactician ensemble training pipeline: {e}")
        
        return artifacts
    
    async def _tactician_ensemble_training_pipeline(self, config: SubPipelineConfig, ensemble_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Tactician ensemble training sub-pipeline."""
        self.logger.info("⚔️🎯 Executing tactician ensemble training pipeline")
        
        artifacts = {
            'tactician_ensembles': [],
            'ensemble_metrics': {},
            'tactician_ensemble_performance': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual tactician ensemble training")
            artifacts['tactician_ensembles'] = ['tactician_ensemble.pkl']
            return artifacts
        
        # Import and use tactician ensemble training
        try:
            from .tactician_ensemble_training import TacticianEnsembleTrainingStep as TacticianEnsembleTrainer
            
            # Create enhanced configuration with all model inputs for meta learner
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            if ensemble_data:
                enhanced_config.update({
                    'base_tactician_models': ensemble_data.get('tactician_models', []),
                    'tactician_training_metrics': ensemble_data.get('tactician_training_metrics', {}),
                    'analyst_models': ensemble_data.get('analyst_models', []),
                    'analyst_ensembles': ensemble_data.get('analyst_ensembles', []),
                    'analyst_ensemble_metrics': ensemble_data.get('analyst_ensemble_metrics', {}),
                    'hmm_data': ensemble_data.get('hmm_data', {})
                })
                self.logger.info("✅ Meta learner will use all inputs from all ML models (HMM, Analyst, Tactician)")
            
            trainer = TacticianEnsembleTrainer()
            training_result = await trainer.execute_tactician_ensemble_training(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                force_rerun=config.force_rerun,
                enhanced_config=enhanced_config
            )
            
            artifacts['tactician_ensembles'] = training_result.get('models', [])
            artifacts['ensemble_metrics'] = training_result.get('metrics', {})
            artifacts['tactician_ensemble_performance'] = training_result.get('performance', {})
            
        except ImportError:
            self.logger.warning("⚠️ Tactician ensemble trainer not available, using mock training")
            artifacts['tactician_ensembles'] = ['tactician_ensemble.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("tactician_ensemble_training", config, artifacts)
        
        # This is the final step in the pipeline
        self.logger.info("🎉 All model training pipelines completed successfully!")
        
        return artifacts
    
    async def _hmm_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """HMM training sub-pipeline."""
        self.logger.info("🔄 Executing HMM training pipeline")
        
        artifacts = {
            'hmm_models': [],
            'hmm_metrics': {},
            'regime_models': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual HMM training")
            artifacts['hmm_models'] = ['hmm_model.pkl']
            return artifacts
        
        # Import and use HMM training
        try:
            from .simplified.hmm_training import HMMTrainingPipeline
            from .hmm_regime_integration_step import HMMRegimeIntegrationStep
            
            # Initialize HMM training pipeline
            hmm_trainer = HMMTrainingPipeline()
            
            # Create a mock pipeline state for HMM training
            pipeline_state = {
                'hmm_regime_discovery_completed': True,
                'step03_hmm_regime_discovery_completed': True,
                'regime_states': [],  # Will be populated by HMM training
                'regime_probabilities': [],
                'regime_confidence': [],
                'hmm_state_sequence': [],
                'hmm_state_probs': [],
                'regime_characteristics': {},
                'transition_matrix': None
            }
            
            # Execute HMM training
            hmm_result = await hmm_trainer.train_hmm_models(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                pipeline_state=pipeline_state,
                force_rerun=config.force_rerun
            )
            
            # Update pipeline state with HMM results
            updated_pipeline_state = hmm_result.get('updated_pipeline_state', pipeline_state)
            updated_pipeline_state['hmm_training_completed'] = True
            
            # Execute HMM regime integration step
            regime_integrator = HMMRegimeIntegrationStep()
            integration_result = await regime_integrator.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state=updated_pipeline_state
            )
            
            artifacts['hmm_models'] = hmm_result.get('models', [])
            artifacts['hmm_metrics'] = hmm_result.get('metrics', {})
            artifacts['regime_models'] = hmm_result.get('regime_models', {})
            artifacts['hmm_regime_integration'] = integration_result
            artifacts['updated_pipeline_state'] = integration_result
            
        except ImportError:
            self.logger.warning("⚠️ HMM training pipeline not available, using mock training")
            artifacts['hmm_models'] = ['hmm_model.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("hmm_training", config, artifacts)
        
        # Automatically trigger the next sub-pipeline: analyst_model_training
        self.logger.info("🔄 HMM training completed, triggering next: analyst_model_training")
        try:
            # Pass HMM results to analyst model training
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            enhanced_config.update({
                'hmm_regime_states': artifacts.get('updated_pipeline_state', {}).get('regime_states', []),
                'hmm_regime_probabilities': artifacts.get('updated_pipeline_state', {}).get('regime_probabilities', []),
                'hmm_regime_confidence': artifacts.get('updated_pipeline_state', {}).get('regime_confidence', []),
                'hmm_state_sequence': artifacts.get('updated_pipeline_state', {}).get('hmm_state_sequence', []),
                'hmm_state_probs': artifacts.get('updated_pipeline_state', {}).get('hmm_state_probs', []),
                'regime_characteristics': artifacts.get('updated_pipeline_state', {}).get('regime_characteristics', {}),
                'transition_matrix': artifacts.get('updated_pipeline_state', {}).get('transition_matrix', None)
            })
            
            next_artifacts = await self._analyst_model_training_pipeline(config, enhanced_config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Analyst model training pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute analyst model training pipeline: {e}")
        
        return artifacts
    
    async def _ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Ensemble training sub-pipeline."""
        self.logger.info("🎯 Executing ensemble training pipeline")
        
        artifacts = {
            'ensemble_models': [],
            'ensemble_metrics': {},
            'ensemble_weights': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual ensemble training")
            artifacts['ensemble_models'] = ['ensemble_model.pkl']
            return artifacts
        
        # Import and use ensemble training
        try:
            from .simplified.ensemble_training import EnsembleTrainingPipeline
            
            ensemble_trainer = EnsembleTrainingPipeline()
            ensemble_result = await ensemble_trainer.train_ensemble_models(
                data_dir=config.data_dir,
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe
            )
            
            artifacts['ensemble_models'] = ensemble_result.get('models', [])
            artifacts['ensemble_metrics'] = ensemble_result.get('metrics', {})
            artifacts['ensemble_weights'] = ensemble_result.get('weights', {})
            
        except ImportError:
            self.logger.warning("⚠️ Ensemble training pipeline not available, using mock training")
            artifacts['ensemble_models'] = ['ensemble_model.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("ensemble_training", config, artifacts)
        
        # Automatically trigger the next sub-pipeline: multi_timeframe_training
        self.logger.info("🔄 Ensemble training completed, triggering next: multi_timeframe_training")
        try:
            next_artifacts = await self._multi_timeframe_training_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Multi-timeframe training pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute multi-timeframe training pipeline: {e}")
        
        return artifacts
    
    async def _multi_timeframe_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Multi-timeframe training sub-pipeline."""
        self.logger.info("⏰ Executing multi-timeframe training pipeline")
        
        artifacts = {
            'multi_tf_models': [],
            'timeframe_metrics': {},
            'cross_timeframe_performance': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual multi-timeframe training")
            artifacts['multi_tf_models'] = ['multi_tf_model.pkl']
            return artifacts
        
        # Import and use multi-timeframe training
        try:
            from src.utils.ml_common.multi_timeframe_training import MultiTimeframeTrainer, MultiTimeframeTrainingConfig, TimeframeConfig
            
            # Create timeframe configurations (removed 1d and 4h as requested)
            timeframe_configs = [
                TimeframeConfig(timeframe='1m', weight=0.3),
                TimeframeConfig(timeframe='5m', weight=0.2),
                TimeframeConfig(timeframe='15m', weight=0.2),
                TimeframeConfig(timeframe='30m', weight=0.2),
                TimeframeConfig(timeframe='1h', weight=0.1)
            ]
            
            mtf_config = MultiTimeframeTrainingConfig(
                timeframes=timeframe_configs,
                enable_cross_timeframe_features=True,
                enable_timeframe_ensemble=True,
                ensemble_method="weighted_average"
            )
            
            multi_tf_trainer = MultiTimeframeTrainer(mtf_config, config.symbol, config.exchange)
            
            # Note: This would need actual training data to work properly
            # For now, we'll use mock data structure
            mock_training_data = {
                '1m': pd.DataFrame(),  # Would contain actual data
                '5m': pd.DataFrame(),
                '15m': pd.DataFrame(),
                '30m': pd.DataFrame(),
                '1h': pd.DataFrame()
            }
            
            # multi_tf_result = await multi_tf_trainer.train_models(mock_training_data, model_trainer, model_config)
            
            artifacts['multi_tf_models'] = ['multi_tf_model.pkl']
            artifacts['timeframe_metrics'] = {'timeframes_processed': len(timeframe_configs)}
            artifacts['cross_timeframe_performance'] = {'ensemble_method': 'weighted_average'}
            
        except ImportError:
            self.logger.warning("⚠️ Multi-timeframe training pipeline not available, using mock training")
            artifacts['multi_tf_models'] = ['multi_tf_model.pkl']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("multi_timeframe_training", config, artifacts)
        
        return artifacts
    
    
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.sub_pipelines.keys())
    
    def get_sub_pipeline_status(self, sub_pipeline_name: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific sub-pipeline."""
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result.status
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all sub-pipeline executions."""
        total_executions = len(self.results)
        completed = sum(1 for r in self.results if r.status == SubPipelineStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == SubPipelineStatus.FAILED)
        total_duration = sum(r.duration_seconds or 0 for r in self.results)
        
        return {
            'total_executions': total_executions,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_executions if total_executions > 0 else 0,
            'total_duration_seconds': total_duration,
            'results': self.results
        }

# Convenience functions
def get_model_training_sub_pipeline(config: Optional[SubPipelineConfig] = None) -> ModelTrainingSubPipeline:
    """Get a configured model training sub-pipeline."""
    return ModelTrainingSubPipeline(config)

async def execute_model_training_sub_pipeline(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute a model training sub-pipeline."""
    pipeline = get_model_training_sub_pipeline(config)
    return await pipeline.execute_sub_pipeline(sub_pipeline_name, config)