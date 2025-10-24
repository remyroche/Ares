"""
ML Model Trainer Step - Integrated with Base Step System

This step integrates the unified ML model trainer with the base step system,
providing seamless integration with the ares launcher and artifact management.
"""

import asyncio
import logging
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Import base step
from src.training.steps.base_step import BaseStep

# Import ML model trainer
from src.training.ml_model_trainer import (
    MLModelTrainer, MLModelTrainerConfig, ModelType, TrainingResult
)

# Import step registry
from src.training.steps.base_step import step_registry

# Import pipeline mode configuration
from src.config.pipeline_modes import get_mode_config, get_mode_lookback_days


@dataclass
class MLModelTrainerStepConfig:
    """Configuration for the ML Model Trainer Step."""
    # Model types to train
    model_types: List[str] = field(default_factory=lambda: [
        "analyst_base", "analyst_ensemble", "tactician_base", "tactician_ensemble"
    ])
    
    # Data configuration
    timeframe: str = "15m"
    random_state: int = 42
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    cv_folds: int = 5
    
    # Performance configuration
    enable_parallel_training: bool = True
    max_workers: int = 4
    enable_gpu: bool = False
    
    # Output configuration
    output_dir: str = "results/ml_model_trainer"
    save_models: bool = True
    save_predictions: bool = True
    save_reports: bool = True
    
    # Monitoring configuration
    enable_monitoring: bool = True
    log_level: str = "INFO"
    verbose: bool = True


class MLModelTrainerStep(BaseStep):
    """
    ML Model Trainer Step that integrates with the base step system.
    
    This step provides a unified interface for training all ML models:
    - Analyst Base Models
    - Analyst Ensemble Models  
    - Tactician Base Models
    - Tactician Ensemble Models
    """
    
    def __init__(self, step_name: str, config: Dict[str, Any]):
        """Initialize the ML Model Trainer Step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.{step_name}")
        
        # Initialize configuration
        self.step_config = self._create_step_config(config)
        
        # Initialize ML model trainer
        self.ml_trainer = None
        
    def _create_step_config(self, config: Dict[str, Any]) -> MLModelTrainerStepConfig:
        """Create step configuration from input config."""
        return MLModelTrainerStepConfig(
            model_types=config.get('model_types', [
                "analyst_base", "analyst_ensemble", "tactician_base", "tactician_ensemble"
            ]),
            timeframe=config.get('timeframe', '15m'),
            random_state=config.get('random_state', 42),
            validation_split=config.get('validation_split', 0.2),
            test_split=config.get('test_split', 0.1),
            cv_folds=config.get('cv_folds', 5),
            enable_parallel_training=config.get('enable_parallel_training', True),
            max_workers=config.get('max_workers', 4),
            enable_gpu=config.get('enable_gpu', False),
            output_dir=config.get('output_dir', 'results/ml_model_trainer'),
            save_models=config.get('save_models', True),
            save_predictions=config.get('save_predictions', True),
            save_reports=config.get('save_reports', True),
            enable_monitoring=config.get('enable_monitoring', True),
            log_level=config.get('log_level', 'INFO'),
            verbose=config.get('verbose', True)
        )
    
    def _convert_model_types(self, model_types: List[str]) -> List[ModelType]:
        """Convert string model types to ModelType enum."""
        type_mapping = {
            "analyst_base": ModelType.ANALYST_BASE,
            "analyst_ensemble": ModelType.ANALYST_ENSEMBLE,
            "tactician_base": ModelType.TACTICIAN_BASE,
            "tactician_ensemble": ModelType.TACTICIAN_ENSEMBLE
        }
        
        converted_types = []
        for model_type in model_types:
            if model_type in type_mapping:
                converted_types.append(type_mapping[model_type])
            else:
                self.logger.warning(f"Unknown model type: {model_type}")
        
        return converted_types
    
    def _get_config_paths(self) -> Dict[ModelType, str]:
        """Get configuration file paths for each model type."""
        base_config_dir = Path("config/ml_model_trainer")
        
        config_paths = {
            ModelType.ANALYST_BASE: str(base_config_dir / "analyst_base_config.yaml"),
            ModelType.ANALYST_ENSEMBLE: str(base_config_dir / "analyst_ensemble_config.yaml"),
            ModelType.TACTICIAN_BASE: str(base_config_dir / "tactician_base_config.yaml"),
            ModelType.TACTICIAN_ENSEMBLE: str(base_config_dir / "tactician_ensemble_config.yaml")
        }
        
        # Check if config files exist, create defaults if not
        for model_type, config_path in config_paths.items():
            if not Path(config_path).exists():
                self.logger.warning(f"Config file not found: {config_path}, creating default")
                self._create_default_config(config_path, model_type)
        
        return config_paths
    
    def _create_default_config(self, config_path: str, model_type: ModelType):
        """Create default configuration file for a model type by copying from existing YAML configs."""
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Map model types to existing YAML config files
        config_mapping = {
            ModelType.ANALYST_BASE: "config/ml_model_trainer/analyst_base_config.yaml",
            ModelType.ANALYST_ENSEMBLE: "config/ml_model_trainer/analyst_ensemble_config.yaml",
            ModelType.TACTICIAN_BASE: "config/ml_model_trainer/tactician_base_config.yaml",
            ModelType.TACTICIAN_ENSEMBLE: "config/ml_model_trainer/tactician_ensemble_config.yaml"
        }
        
        source_config = config_mapping.get(model_type)
        if source_config and Path(source_config).exists():
            # Copy existing YAML config
            import shutil
            shutil.copy2(source_config, config_path)
            self.logger.info(f"Copied existing config from {source_config} to {config_path}")
        else:
            # Create minimal config if source doesn't exist
            minimal_config = {
                "model_type": model_type.value,
                "timeframe": self.step_config.timeframe,
                "description": f"Configuration for {model_type.value} models",
                "models": [],
                "targets": {
                    "primary": "target",
                    "target_type": "binary_classification"
                },
                "training": {
                    "validation_split": self.step_config.validation_split,
                    "cv_folds": self.step_config.cv_folds,
                    "random_state": 42
                },
                "metrics": {
                    "primary": "f1_score",
                    "secondary": ["precision", "recall", "accuracy"]
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(minimal_config, f, default_flow_style=False, indent=2)
            
            self.logger.warning(f"Created minimal config for {model_type.value} at {config_path}")
            self.logger.info(f"Please update {config_path} with proper model configurations")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the ML model training step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('longs', 'shorts', 'both')
                - execution_mode: Execution mode ('full', 'light', 'blank')
                - model_types: List of model types to train (optional)
        
        Returns:
            Dictionary containing execution results
        """
        try:
            self.logger.info("🚀 Starting ML Model Trainer Step")
            
            # Get execution mode and validate
            execution_mode = config.get('execution_mode', 'light')
            if execution_mode not in ['full', 'light', 'blank']:
                self.logger.warning(f"Invalid execution mode '{execution_mode}', defaulting to 'light'")
                execution_mode = 'light'
            
            # Get mode configuration for lookback periods and other parameters
            mode_config = get_mode_config(execution_mode)
            lookback_days = mode_config.lookback_days
            
            self.logger.info(f"📊 Execution Mode: {execution_mode.upper()}")
            self.logger.info(f"📅 Lookback Period: {lookback_days} days ({mode_config.lookback_years} years)")
            self.logger.info(f"⚡ Computational Intensity: {mode_config.computational_intensity}")
            self.logger.info(f"🔄 Max Trials: {mode_config.max_trials}")
            
            # Set context for enhanced file naming with execution mode
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'longs'),
                model=config.get('model', 'ML_Trainer'),
                execution_mode=execution_mode
            )
            
            # Determine which model types to train
            requested_model_types = config.get('model_types', self.step_config.model_types)
            model_types = self._convert_model_types(requested_model_types)
            
            if not model_types:
                return {
                    'success': False,
                    'error': 'No valid model types specified',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Create ML model trainer configuration with mode-specific parameters
            ml_config = MLModelTrainerConfig(
                model_types=model_types,
                timeframe=config.get('timeframe', self.step_config.timeframe),
                random_state=self.step_config.random_state,
                validation_split=self.step_config.validation_split,
                test_split=self.step_config.test_split,
                cv_folds=mode_config.cross_validation_folds,  # Use mode-specific CV folds
                enable_parallel_training=mode_config.enable_parallelization,  # Use mode-specific parallelization
                max_workers=self.step_config.max_workers,
                enable_gpu=self.step_config.enable_gpu,
                output_dir=self.step_config.output_dir,
                save_models=self.step_config.save_models,
                save_predictions=self.step_config.save_predictions,
                save_reports=self.step_config.save_reports,
                enable_monitoring=self.step_config.enable_monitoring,
                log_level=self.step_config.log_level,
                verbose=self.step_config.verbose
            )
            
            # Store mode-specific parameters for use in training
            mode_params = {
                'execution_mode': execution_mode,
                'lookback_days': lookback_days,
                'lookback_years': mode_config.lookback_years,
                'intensity_percentage': mode_config.intensity_percentage,
                'computational_intensity': mode_config.computational_intensity,
                'max_trials': mode_config.max_trials,
                'n_trials': mode_config.n_trials,
                'optuna_trials': mode_config.optuna_trials,
                'optuna_timeout': mode_config.optuna_timeout,
                'batch_size': mode_config.batch_size,
                'epochs': mode_config.epochs,
                'early_stopping_patience': mode_config.early_stopping_patience,
                'enable_advanced_features': mode_config.enable_advanced_features,
                'enable_ensemble_training': mode_config.enable_ensemble_training,
                'enable_multi_timeframe_training': mode_config.enable_multi_timeframe_training,
                'enable_adaptive_training': mode_config.enable_adaptive_training
            }
            
            # Save mode parameters as an artifact for reference
            self._save_dataframe(
                pd.DataFrame([mode_params]), 
                'mode_parameters',
                metadata={
                    'execution_mode': execution_mode,
                    'lookback_days': lookback_days,
                    'description': f'Mode-specific parameters for {execution_mode} execution'
                }
            )
            
            # Initialize ML model trainer
            self.ml_trainer = MLModelTrainer(ml_config, self.logger)
            
            # Load training data for each model type
            training_data = await self._load_training_data(config)
            if not training_data:
                return {
                    'success': False,
                    'error': 'Failed to load training data',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Prepare data for ML trainer - combine all features for each model type
            all_features = []
            for model_type_str, features in training_data['model_data'].items():
                if features is not None:
                    all_features.append(features)
                    self.logger.info(f"Added {features.shape[1]} features for {model_type_str}")
            
            if not all_features:
                self.logger.error("No features available for training")
                return {
                    'success': False,
                    'error': 'No features available for training',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Combine all features
            import numpy as np
            combined_features = np.hstack(all_features)
            self.logger.info(f"Combined features shape: {combined_features.shape}")
            
            # Prepare data for ML trainer
            data = {
                'features': combined_features,
                'targets': training_data['targets'],
                'model_data': training_data['model_data'],
                'regime_outputs': training_data['regime_outputs'],
                'metadata': training_data['metadata']
            }
            
            # Add mode parameters to training data metadata and as separate fields
            data['metadata'].update(mode_params)
            data['execution_mode'] = execution_mode
            data['lookback_days'] = lookback_days
            data['mode_config'] = mode_params
            
            # Get configuration paths
            config_paths = self._get_config_paths()
            
            # Filter config paths to only include requested model types
            filtered_config_paths = {
                model_type: path for model_type, path in config_paths.items()
                if model_type in model_types
            }
            
            # Train models
            self.logger.info(f"Training models: {[mt.value for mt in model_types]}")
            self.logger.info(f"Mode: {execution_mode.upper()} | Lookback: {lookback_days} days | CV Folds: {mode_config.cross_validation_folds}")
            self.logger.info(f"Max Trials: {mode_config.max_trials} | Parallel: {mode_config.enable_parallelization}")
            results = await self.ml_trainer.train_models(data, filtered_config_paths)
            
            # Process results and save artifacts
            artifacts = []
            metrics = {}
            
            for model_type, model_results in results.items():
                if isinstance(model_results, list):
                    # Handle list of TrainingResult objects
                    for result in model_results:
                        if result.success:
                            artifacts.append(f"{model_type.value}_{result.model_name}")
                            metrics[f"{model_type.value}_{result.model_name}"] = result.metrics
                else:
                    # Handle single result dictionary
                    if model_results.get('success', False):
                        artifacts.append(f"{model_type.value}_ensemble")
                        metrics[f"{model_type.value}_ensemble"] = model_results.get('metrics', {})
            
            # Save results as artifacts
            self._save_metadata({
                'model_types': [mt.value for mt in model_types],
                'results': results,
                'artifacts': artifacts,
                'metrics': metrics
            }, 'ml_training_results')
            
            self.logger.info(f"✅ ML Model Trainer Step completed successfully")
            self.logger.info(f"Trained {len(artifacts)} models")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'model_types': [mt.value for mt in model_types],
                'results': results
            }
            
        except Exception as e:
            error_msg = f"ML Model Trainer Step failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {}
            }
    
    async def _load_training_data(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load training data from artifacts based on model type and direction."""
        try:
            model_types = config.get('model_types', [])
            direction = config.get('direction', 'longs')
            
            # Load targets from labeling integration step
            targets = await self._load_targets(config)
            if targets is None:
                self.logger.error("Failed to load targets from labeling integration step")
                return None
            
            # Load regime model outputs
            regime_outputs = await self._load_regime_outputs(config)
            
            # Load model-specific data
            model_data = {}
            for model_type in model_types:
                model_type_str = model_type.value if hasattr(model_type, 'value') else str(model_type)
                model_features = await self._load_model_features(model_type_str, direction, regime_outputs, config)
                if model_features is not None:
                    model_data[model_type_str] = model_features
                else:
                    self.logger.warning(f"Failed to load features for {model_type_str}")
            
            if not model_data:
                self.logger.error("Failed to load features for any model type")
                return None
            
            return {
                'targets': targets,
                'model_data': model_data,
                'regime_outputs': regime_outputs,
                'metadata': {
                    'symbol': config.get('symbol'),
                    'exchange': config.get('exchange'),
                    'timeframe': config.get('timeframe'),
                    'direction': direction,
                    'execution_mode': config.get('execution_mode', 'light'),
                    'model_types': [mt.value if hasattr(mt, 'value') else str(mt) for mt in model_types],
                    'target_shape': targets.shape if hasattr(targets, 'shape') else (len(targets),),
                    'data_points': len(targets)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            return None

    async def _load_targets(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load targets from feature_generation_labeling_integration_step."""
        try:
            # Load targets from labeling integration step
            targets = self._load_dataframe('targets')
            if targets is not None and not targets.empty:
                self.logger.info("Loaded targets from labeling integration step")
                return targets.values if hasattr(targets, 'values') else np.array(targets)
            
            # Fallback: try other target artifacts
            target_artifacts = ['processed_targets', 'final_targets', 'profit_labels', 'risk_labels']
            for artifact_name in target_artifacts:
                targets = self._load_dataframe(artifact_name)
                if targets is not None and not targets.empty:
                    self.logger.info(f"Loaded targets from artifact: {artifact_name}")
                    return targets.values if hasattr(targets, 'values') else np.array(targets)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load targets: {e}")
            return None

    async def _load_regime_outputs(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load regime model outputs from regime_data_splitting step."""
        try:
            regime_outputs = {}
            
            # Load regime data splits
            regime_artifacts = [
                'low_vol_data', 'normal_data', 'high_vol_data',
                'regime_classification', 'regime_probabilities',
                'regime_transition_matrix', 'regime_volatility_estimates'
            ]
            
            for artifact_name in regime_artifacts:
                try:
                    data = self._load_dataframe(artifact_name)
                    if data is not None and not data.empty:
                        regime_outputs[artifact_name] = data
                        self.logger.debug(f"Loaded regime artifact: {artifact_name}")
                except Exception as e:
                    self.logger.debug(f"Failed to load regime artifact {artifact_name}: {e}")
                    continue
            
            if regime_outputs:
                self.logger.info(f"Loaded {len(regime_outputs)} regime outputs")
            else:
                self.logger.warning("No regime outputs found")
            
            return regime_outputs
            
        except Exception as e:
            self.logger.error(f"Failed to load regime outputs: {e}")
            return None

    async def _load_model_features(self, model_type: str, direction: str, regime_outputs: Dict[str, Any], config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load features specific to each model type."""
        try:
            import numpy as np
            import pandas as pd
            
            features_list = []
            
            if model_type in ['analyst_base', 'analyst_ensemble']:
                # Analyst models: use features from feature_generation_final_feature_selection_step for Analyst
                analyst_features = await self._load_analyst_features(direction)
                if analyst_features is not None:
                    features_list.append(analyst_features)
                    self.logger.info(f"Loaded analyst features for {model_type}")
                else:
                    self.logger.warning(f"Failed to load analyst features for {model_type}")
            
            if model_type in ['tactician_base', 'tactician_ensemble']:
                # Tactician models: use features from feature_generation_final_feature_selection_step for Tactician
                tactician_features = await self._load_tactician_features(direction)
                if tactician_features is not None:
                    features_list.append(tactician_features)
                    self.logger.info(f"Loaded tactician features for {model_type}")
                else:
                    self.logger.warning(f"Failed to load tactician features for {model_type}")
            
            # Add regime outputs to all models
            if regime_outputs:
                regime_features = await self._prepare_regime_features(regime_outputs)
                if regime_features is not None:
                    features_list.append(regime_features)
                    self.logger.info(f"Added regime features to {model_type}")
            
            # Add previous model outputs for ensemble models
            if model_type == 'analyst_ensemble':
                analyst_base_outputs = await self._load_analyst_base_outputs()
                if analyst_base_outputs is not None:
                    features_list.append(analyst_base_outputs)
                    self.logger.info("Added analyst base outputs to analyst ensemble")
            
            if model_type == 'tactician_base':
                analyst_ensemble_outputs = await self._load_analyst_ensemble_outputs()
                if analyst_ensemble_outputs is not None:
                    features_list.append(analyst_ensemble_outputs)
                    self.logger.info("Added analyst ensemble outputs to tactician base")
            
            if model_type == 'tactician_ensemble':
                analyst_ensemble_outputs = await self._load_analyst_ensemble_outputs()
                tactician_base_outputs = await self._load_tactician_base_outputs()
                
                if analyst_ensemble_outputs is not None:
                    features_list.append(analyst_ensemble_outputs)
                    self.logger.info("Added analyst ensemble outputs to tactician ensemble")
                
                if tactician_base_outputs is not None:
                    features_list.append(tactician_base_outputs)
                    self.logger.info("Added tactician base outputs to tactician ensemble")
            
            if not features_list:
                self.logger.error(f"No features loaded for {model_type}")
                return None
            
            # Combine all features
            combined_features = np.hstack(features_list)
            self.logger.info(f"Combined {len(features_list)} feature sets for {model_type}, shape: {combined_features.shape}")
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Failed to load features for {model_type}: {e}")
            return None

    async def _load_analyst_features(self, direction: str) -> Optional[np.ndarray]:
        """Load analyst features from feature_generation_final_feature_selection_step."""
        try:
            # Look for analyst features with direction
            artifact_name = f'selected_feature_dataframe_60_analyst_{direction}'
            features = self._load_dataframe(artifact_name)
            
            if features is None or features.empty:
                # Fallback to general analyst features
                fallback_artifacts = [
                    f'selected_feature_dataframe_60_analyst',
                    'selected_feature_dataframe_60',
                    'analyst_features'
                ]
                
                for fallback_name in fallback_artifacts:
                    features = self._load_dataframe(fallback_name)
                    if features is not None and not features.empty:
                        break
            
            if features is not None and not features.empty:
                return features.values if hasattr(features, 'values') else np.array(features)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load analyst features: {e}")
            return None

    async def _load_tactician_features(self, direction: str) -> Optional[np.ndarray]:
        """Load tactician features from feature_generation_final_feature_selection_step."""
        try:
            # Look for tactician features with direction
            artifact_name = f'selected_feature_dataframe_60_tactician_{direction}'
            features = self._load_dataframe(artifact_name)
            
            if features is None or features.empty:
                # Fallback to general tactician features
                fallback_artifacts = [
                    f'selected_feature_dataframe_60_tactician',
                    'selected_feature_dataframe_60',
                    'tactician_features'
                ]
                
                for fallback_name in fallback_artifacts:
                    features = self._load_dataframe(fallback_name)
                    if features is not None and not features.empty:
                        break
            
            if features is not None and not features.empty:
                return features.values if hasattr(features, 'values') else np.array(features)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load tactician features: {e}")
            return None

    async def _prepare_regime_features(self, regime_outputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare regime features from regime outputs."""
        try:
            import numpy as np
            import pandas as pd
            
            regime_features = []
            
            # Add regime classification if available
            if 'regime_classification' in regime_outputs:
                regime_class = regime_outputs['regime_classification']
                if hasattr(regime_class, 'values'):
                    regime_features.append(regime_class.values)
                else:
                    regime_features.append(np.array(regime_class))
            
            # Add regime probabilities if available
            if 'regime_probabilities' in regime_outputs:
                regime_probs = regime_outputs['regime_probabilities']
                if hasattr(regime_probs, 'values'):
                    regime_features.append(regime_probs.values)
                else:
                    regime_features.append(np.array(regime_probs))
            
            if not regime_features:
                return None
            
            # Combine regime features
            combined_regime = np.hstack(regime_features)
            return combined_regime
            
        except Exception as e:
            self.logger.error(f"Failed to prepare regime features: {e}")
            return None

    async def _load_analyst_base_outputs(self) -> Optional[np.ndarray]:
        """Load analyst base model outputs."""
        try:
            # Look for analyst base model outputs
            output_artifacts = [
                'analyst_base_predictions',
                'analyst_base_probabilities',
                'analyst_base_outputs'
            ]
            
            for artifact_name in output_artifacts:
                outputs = self._load_dataframe(artifact_name)
                if outputs is not None and not outputs.empty:
                    return outputs.values if hasattr(outputs, 'values') else np.array(outputs)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to load analyst base outputs: {e}")
            return None

    async def _load_analyst_ensemble_outputs(self) -> Optional[np.ndarray]:
        """Load analyst ensemble model outputs."""
        try:
            # Look for analyst ensemble model outputs
            output_artifacts = [
                'analyst_ensemble_predictions',
                'analyst_ensemble_probabilities',
                'analyst_ensemble_outputs'
            ]
            
            for artifact_name in output_artifacts:
                outputs = self._load_dataframe(artifact_name)
                if outputs is not None and not outputs.empty:
                    return outputs.values if hasattr(outputs, 'values') else np.array(outputs)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to load analyst ensemble outputs: {e}")
            return None

    async def _load_tactician_base_outputs(self) -> Optional[np.ndarray]:
        """Load tactician base model outputs."""
        try:
            # Look for tactician base model outputs
            output_artifacts = [
                'tactician_base_predictions',
                'tactician_base_outputs'
            ]
            
            for artifact_name in output_artifacts:
                outputs = self._load_dataframe(artifact_name)
                if outputs is not None and not outputs.empty:
                    return outputs.values if hasattr(outputs, 'values') else np.array(outputs)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to load tactician base outputs: {e}")
            return None


# Register the step
step_registry.register("ml_model_trainer", MLModelTrainerStep)
step_registry.register("train_analyst_base", MLModelTrainerStep)
step_registry.register("train_analyst_ensemble", MLModelTrainerStep)
step_registry.register("train_tactician_base", MLModelTrainerStep)
step_registry.register("train_tactician_ensemble", MLModelTrainerStep)