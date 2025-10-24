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
        """Create default configuration file for a model type."""
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default config based on model type
        if model_type == ModelType.ANALYST_BASE:
            default_config = {
                "timeframe": self.step_config.timeframe,
                "targets": {
                    "target_type": "binary_classification",
                    "primary_target": "profit_label",
                    "secondary_target": "risk_label"
                },
                "inputs": {
                    "analyst_features": {
                        "enable_patchtst_features": True,
                        "enable_regime_features": True,
                        "enable_multi_timeframe": True
                    }
                },
                "models": [
                    {
                        "name": "LightGBM_Analyst_Base",
                        "type": "LIGHTGBM",
                        "enabled": True,
                        "parameters": {
                            "n_estimators": 1000,
                            "learning_rate": 0.1,
                            "num_leaves": 31,
                            "max_depth": 6,
                            "random_state": 42
                        }
                    }
                ],
                "training": {
                    "validation_split": self.step_config.validation_split,
                    "cv_folds": self.step_config.cv_folds,
                    "early_stopping": {
                        "enabled": True,
                        "patience": 50
                    },
                    "hyperparameter_optimization": {
                        "enabled": False,
                        "n_trials": 100,
                        "timeout": 3600
                    }
                },
                "metrics": {
                    "primary": "f1_score",
                    "secondary": ["precision", "recall", "auc_roc"]
                }
            }
        elif model_type == ModelType.ANALYST_ENSEMBLE:
            default_config = {
                "timeframe": self.step_config.timeframe,
                "targets": {
                    "target_type": "binary_classification",
                    "primary_target": "profit_label",
                    "secondary_target": "risk_label"
                },
                "inputs": {
                    "analyst_features": {
                        "enable_patchtst_features": True,
                        "enable_regime_features": True,
                        "enable_multi_timeframe": True
                    }
                },
                "base_models": [
                    {
                        "name": "LightGBM_Base",
                        "type": "LIGHTGBM",
                        "enabled": True,
                        "parameters": {
                            "n_estimators": 1000,
                            "learning_rate": 0.1,
                            "random_state": 42
                        }
                    },
                    {
                        "name": "CatBoost_Base",
                        "type": "CATBOOST",
                        "enabled": True,
                        "parameters": {
                            "iterations": 1000,
                            "learning_rate": 0.1,
                            "random_seed": 42
                        }
                    }
                ],
                "models": [
                    {
                        "name": "Stacking_Ensemble",
                        "type": "STACKING",
                        "enabled": True,
                        "parameters": {
                            "meta_learner_type": "LIGHTGBM",
                            "meta_learner_params": {
                                "n_estimators": 100,
                                "learning_rate": 0.1,
                                "random_state": 42
                            },
                            "cv_folds": 5,
                            "use_features_in_secondary": True,
                            "use_proba_as_level1": True
                        }
                    }
                ],
                "training": {
                    "validation_split": self.step_config.validation_split,
                    "cv_folds": self.step_config.cv_folds,
                    "ensemble_training": {
                        "stacking": {
                            "meta_learner_cv": 5
                        }
                    }
                },
                "metrics": {
                    "primary": "f1_score",
                    "secondary": ["precision", "recall", "auc_roc"]
                }
            }
        elif model_type == ModelType.TACTICIAN_BASE:
            default_config = {
                "timeframe": self.step_config.timeframe,
                "targets": {
                    "target_type": "regression",
                    "primary_target": "entry_timing",
                    "secondary_target": "exit_timing",
                    "tertiary_target": "position_sizing"
                },
                "inputs": {
                    "tactician_features": {
                        "enable_entry_timing": True,
                        "enable_exit_timing": True,
                        "enable_position_sizing": True
                    }
                },
                "models": [
                    {
                        "name": "LightGBM_Tactician_Base",
                        "type": "LIGHTGBM",
                        "enabled": True,
                        "parameters": {
                            "n_estimators": 1000,
                            "learning_rate": 0.1,
                            "num_leaves": 31,
                            "max_depth": 6,
                            "random_state": 42
                        }
                    }
                ],
                "training": {
                    "validation_split": self.step_config.validation_split,
                    "cv_folds": self.step_config.cv_folds,
                    "early_stopping": {
                        "enabled": True,
                        "patience": 50
                    }
                },
                "metrics": {
                    "primary": "r2_score",
                    "secondary": ["mse", "mae", "explained_variance"]
                }
            }
        elif model_type == ModelType.TACTICIAN_ENSEMBLE:
            default_config = {
                "timeframe": self.step_config.timeframe,
                "targets": {
                    "target_type": "regression",
                    "primary_target": "entry_timing",
                    "secondary_target": "exit_timing",
                    "tertiary_target": "position_sizing"
                },
                "inputs": {
                    "tactician_features": {
                        "enable_entry_timing": True,
                        "enable_exit_timing": True,
                        "enable_position_sizing": True
                    }
                },
                "base_models": [
                    {
                        "name": "LightGBM_Base",
                        "type": "LIGHTGBM",
                        "enabled": True,
                        "parameters": {
                            "n_estimators": 1000,
                            "learning_rate": 0.1,
                            "random_state": 42
                        }
                    },
                    {
                        "name": "XGBoost_Base",
                        "type": "XGBOOST",
                        "enabled": True,
                        "parameters": {
                            "n_estimators": 1000,
                            "learning_rate": 0.1,
                            "random_state": 42
                        }
                    }
                ],
                "models": [
                    {
                        "name": "Stacking_Ensemble",
                        "type": "STACKING",
                        "enabled": True,
                        "parameters": {
                            "meta_learner_type": "LIGHTGBM",
                            "meta_learner_params": {
                                "n_estimators": 100,
                                "learning_rate": 0.1,
                                "random_state": 42
                            },
                            "cv_folds": 5,
                            "use_features_in_secondary": True,
                            "use_proba_as_level1": False
                        }
                    }
                ],
                "training": {
                    "validation_split": self.step_config.validation_split,
                    "cv_folds": self.step_config.cv_folds,
                    "ensemble_training": {
                        "stacking": {
                            "meta_learner_cv": 5
                        }
                    }
                },
                "metrics": {
                    "primary": "r2_score",
                    "secondary": ["mse", "mae", "explained_variance"]
                }
            }
        
        # Write config file
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created default config: {config_path}")
    
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
            
            # Set context for enhanced file naming
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'longs'),
                model=config.get('model', 'ML_Trainer')
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
            
            # Create ML model trainer configuration
            ml_config = MLModelTrainerConfig(
                model_types=model_types,
                timeframe=self.step_config.timeframe,
                random_state=self.step_config.random_state,
                validation_split=self.step_config.validation_split,
                test_split=self.step_config.test_split,
                cv_folds=self.step_config.cv_folds,
                enable_parallel_training=self.step_config.enable_parallel_training,
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
            
            # Get configuration paths
            config_paths = self._get_config_paths()
            
            # Filter config paths to only include requested model types
            filtered_config_paths = {
                model_type: path for model_type, path in config_paths.items()
                if model_type in model_types
            }
            
            # Train models
            self.logger.info(f"Training models: {[mt.value for mt in model_types]}")
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
                    'feature_shape': features_array.shape,
                    'target_shape': targets_array.shape
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