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
            
            # Load data from artifacts
            data = await self._load_training_data(config)
            if not data:
                return {
                    'success': False,
                    'error': 'Failed to load training data',
                    'artifacts': [],
                    'metrics': {}
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
        """Load training data from artifacts."""
        try:
            # Try to load features and targets from artifacts
            features = None
            targets = None
            
            # Look for features in various artifact locations
            feature_artifacts = [
                'features', 'processed_features', 'final_features',
                'analyst_features', 'tactician_features'
            ]
            
            for artifact_name in feature_artifacts:
                try:
                    features = self._load_dataframe(artifact_name)
                    if features is not None and not features.empty:
                        self.logger.info(f"Loaded features from artifact: {artifact_name}")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load {artifact_name}: {e}")
                    continue
            
            # Look for targets in various artifact locations
            target_artifacts = [
                'targets', 'processed_targets', 'final_targets',
                'profit_labels', 'risk_labels', 'entry_labels', 'exit_labels'
            ]
            
            for artifact_name in target_artifacts:
                try:
                    targets = self._load_dataframe(artifact_name)
                    if targets is not None and not targets.empty:
                        self.logger.info(f"Loaded targets from artifact: {artifact_name}")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load {artifact_name}: {e}")
                    continue
            
            if features is None or targets is None:
                self.logger.error("Failed to load required training data")
                return None
            
            # Convert to numpy arrays
            import numpy as np
            features_array = features.values if hasattr(features, 'values') else np.array(features)
            targets_array = targets.values if hasattr(targets, 'values') else np.array(targets)
            
            return {
                'features': features_array,
                'targets': targets_array,
                'metadata': {
                    'symbol': config.get('symbol'),
                    'exchange': config.get('exchange'),
                    'timeframe': config.get('timeframe'),
                    'direction': config.get('direction'),
                    'execution_mode': config.get('execution_mode', 'light'),
                    'feature_shape': features_array.shape,
                    'target_shape': targets_array.shape,
                    'data_points': len(features_array),
                    'feature_count': features_array.shape[1] if len(features_array.shape) > 1 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            return None


# Register the step
step_registry.register("ml_model_trainer", MLModelTrainerStep)
step_registry.register("train_analyst_base", MLModelTrainerStep)
step_registry.register("train_analyst_ensemble", MLModelTrainerStep)
step_registry.register("train_tactician_base", MLModelTrainerStep)
step_registry.register("train_tactician_ensemble", MLModelTrainerStep)