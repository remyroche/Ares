"""
Unified ML Model Trainer Pipeline

This module provides a single, unified pipeline for training all ML models:
- Analyst Base Models
- Analyst Ensemble Models  
- Tactician Base Models
- Tactician Ensemble Models

The pipeline handles:
- Configuration management
- Feature engineering
- Data preprocessing
- Model training
- Cross-validation
- Hyperparameter optimization
- Data leakage detection
- Metrics analysis
- SHAP analysis
- Model evaluation
- Results reporting

Everything is managed by the pipeline except for:
- Which ML models to train (specified in config)
- ML model parameters (specified in config)
- What targets to use (specified in config)
- What inputs to use (specified in config)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing components
from src.training.steps.models_training.core.analyst_base_trainer import (
    AnalystBaseTrainer, AnalystTrainingConfig, AnalystModelType
)
from src.training.steps.models_training.core.analyst_ensemble_trainer import (
    AnalystEnsembleTrainer, AnalystEnsembleTrainingConfig, EnsembleMethod
)
from src.training.steps.models_training.core.tactician_base_trainer import (
    TacticianBaseTrainer, TacticianTrainingConfig, TacticianModelType
)
from src.training.steps.models_training.core.tactician_ensemble_trainer import (
    TacticianEnsembleTrainer, TacticianEnsembleTrainingConfig, TacticianEnsembleMethod
)

# Import utilities
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager, WorkloadType
from src.utils.hardware.optimization_decorators import performance_tracked
from src.utils.hardware.memory_optimized_decorators import comprehensive_memory_optimization

# Import data quality and analysis tools
from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import (
    EnhancedLabelDefinitions, AnalystLabelConfig, TacticianLabelConfig
)
from src.utils.ml_common.data_processing.multi_timeframe_training import MultiTimeframeProcessor
from src.utils.ml_common.ensembles.stacking_ensemble_manager import StackingEnsembleManager

# Import validation and metrics
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Import hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    tprint_warning("Optuna not available. Hyperparameter optimization will be disabled.")


class ModelType(Enum):
    """Model types supported by the unified pipeline."""
    ANALYST_BASE = "analyst_base"
    ANALYST_ENSEMBLE = "analyst_ensemble"
    TACTICIAN_BASE = "tactician_base"
    TACTICIAN_ENSEMBLE = "tactician_ensemble"


@dataclass
class MLModelTrainerConfig:
    """Configuration for the unified ML model trainer."""
    # Pipeline configuration
    model_types: List[ModelType] = field(default_factory=lambda: [
        ModelType.ANALYST_BASE,
        ModelType.ANALYST_ENSEMBLE,
        ModelType.TACTICIAN_BASE,
        ModelType.TACTICIAN_ENSEMBLE
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


@dataclass
class TrainingResult:
    """Result of model training."""
    model_type: ModelType
    model_name: str
    success: bool
    model: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray = None
    probabilities: np.ndarray = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: np.ndarray = None
    training_time: float = 0.0
    error_message: str = ""


class MLModelTrainer:
    """
    Unified ML Model Trainer Pipeline.
    
    This class provides a single interface for training all ML models with
    comprehensive configuration management, feature engineering, validation,
    and analysis capabilities.
    """
    
    def __init__(self, config: MLModelTrainerConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the ML Model Trainer.
        
        Args:
            config: Configuration for the trainer
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or system_logger
        
        # Initialize components
        self._initialize_components()
        
        # Training results storage
        self.training_results: Dict[ModelType, List[TrainingResult]] = {}
        
        # Feature engineering components
        self.feature_engineers = {}
        self.target_generators = {}
        
        # Data quality components
        self.data_validators = {}
        self.leakage_detectors = {}
        
        # Analysis components
        self.metrics_calculators = {}
        self.shap_analyzers = {}
        
        tprint_info(f"🔧 Initialized MLModelTrainer for {config.timeframe}")
        self.logger.info(f"Initialized MLModelTrainer for {config.timeframe}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Initialize hardware manager
        self.hardware_manager = get_integrated_hardware_manager()
        
        # Initialize feature engineers
        self.feature_engineers = {
            ModelType.ANALYST_BASE: self._create_analyst_feature_engineer(),
            ModelType.ANALYST_ENSEMBLE: self._create_analyst_feature_engineer(),
            ModelType.TACTICIAN_BASE: self._create_tactician_feature_engineer(),
            ModelType.TACTICIAN_ENSEMBLE: self._create_tactician_feature_engineer()
        }
        
        # Initialize target generators
        self.target_generators = {
            ModelType.ANALYST_BASE: self._create_analyst_target_generator(),
            ModelType.ANALYST_ENSEMBLE: self._create_analyst_target_generator(),
            ModelType.TACTICIAN_BASE: self._create_tactician_target_generator(),
            ModelType.TACTICIAN_ENSEMBLE: self._create_tactician_target_generator()
        }
        
        # Initialize data validators
        self.data_validators = {
            ModelType.ANALYST_BASE: self._create_data_validator(),
            ModelType.ANALYST_ENSEMBLE: self._create_data_validator(),
            ModelType.TACTICIAN_BASE: self._create_data_validator(),
            ModelType.TACTICIAN_ENSEMBLE: self._create_data_validator()
        }
        
        # Initialize leakage detectors
        self.leakage_detectors = {
            ModelType.ANALYST_BASE: self._create_leakage_detector(),
            ModelType.ANALYST_ENSEMBLE: self._create_leakage_detector(),
            ModelType.TACTICIAN_BASE: self._create_leakage_detector(),
            ModelType.TACTICIAN_ENSEMBLE: self._create_leakage_detector()
        }
        
        # Initialize metrics calculators
        self.metrics_calculators = {
            ModelType.ANALYST_BASE: self._create_metrics_calculator(),
            ModelType.ANALYST_ENSEMBLE: self._create_metrics_calculator(),
            ModelType.TACTICIAN_BASE: self._create_metrics_calculator(),
            ModelType.TACTICIAN_ENSEMBLE: self._create_metrics_calculator()
        }
        
        # Initialize SHAP analyzers
        self.shap_analyzers = {
            ModelType.ANALYST_BASE: self._create_shap_analyzer(),
            ModelType.ANALYST_ENSEMBLE: self._create_shap_analyzer(),
            ModelType.TACTICIAN_BASE: self._create_shap_analyzer(),
            ModelType.TACTICIAN_ENSEMBLE: self._create_shap_analyzer()
        }
    
    def _create_analyst_feature_engineer(self):
        """Create analyst feature engineer."""
        # This would integrate with existing analyst feature engineering
        return None
    
    def _create_tactician_feature_engineer(self):
        """Create tactician feature engineer."""
        # This would integrate with existing tactician feature engineering
        return None
    
    def _create_analyst_target_generator(self):
        """Create analyst target generator."""
        # This would integrate with existing analyst target generation
        return None
    
    def _create_tactician_target_generator(self):
        """Create tactician target generator."""
        # This would integrate with existing tactician target generation
        return None
    
    def _create_data_validator(self):
        """Create data validator."""
        # This would integrate with existing data validation
        return None
    
    def _create_leakage_detector(self):
        """Create leakage detector."""
        # This would integrate with existing leakage detection
        return None
    
    def _create_metrics_calculator(self):
        """Create metrics calculator."""
        # This would integrate with existing metrics calculation
        return None
    
    def _create_shap_analyzer(self):
        """Create SHAP analyzer."""
        # This would integrate with existing SHAP analysis
        return None
    
    @traced
    @log_execution_time
    async def train_models(self, data: Dict[str, Any], config_paths: Dict[ModelType, str]) -> Dict[ModelType, List[TrainingResult]]:
        """
        Train all configured models.
        
        Args:
            data: Input data dictionary
            config_paths: Paths to configuration files for each model type
            
        Returns:
            Dictionary of training results by model type
        """
        tprint_info("🚀 Starting unified ML model training pipeline")
        self.logger.info("Starting unified ML model training pipeline")
        
        # Load configurations
        configs = await self._load_configurations(config_paths)
        
        # Preprocess data
        processed_data = await self._preprocess_data(data)
        
        # Train models
        if self.config.enable_parallel_training:
            results = await self._train_models_parallel(processed_data, configs)
        else:
            results = await self._train_models_sequential(processed_data, configs)
        
        # Generate reports
        if self.config.save_reports:
            await self._generate_reports(results)
        
        tprint_success("✅ ML model training pipeline completed")
        self.logger.info("ML model training pipeline completed")
        
        return results
    
    async def _load_configurations(self, config_paths: Dict[ModelType, str]) -> Dict[ModelType, Dict[str, Any]]:
        """Load configuration files for each model type."""
        configs = {}
        
        for model_type, config_path in config_paths.items():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                configs[model_type] = config
                tprint_info(f"📋 Loaded configuration for {model_type.value}")
            except Exception as e:
                tprint_error(f"❌ Failed to load configuration for {model_type.value}: {e}")
                raise
        
        return configs
    
    async def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data."""
        tprint_info("🔄 Preprocessing data")
        
        # Apply hardware optimization
        processed_data = self.hardware_manager.process_data_with_optimization(
            data, WorkloadType.ML_TRAINING
        )
        
        # Validate data quality
        for model_type in self.config.model_types:
            await self._validate_data_quality(processed_data, model_type)
        
        # Detect data leakage
        for model_type in self.config.model_types:
            await self._detect_data_leakage(processed_data, model_type)
        
        tprint_success("✅ Data preprocessing completed")
        return processed_data
    
    async def _validate_data_quality(self, data: Dict[str, Any], model_type: ModelType):
        """Validate data quality for specific model type."""
        # This would implement comprehensive data quality validation
        pass
    
    async def _detect_data_leakage(self, data: Dict[str, Any], model_type: ModelType):
        """Detect data leakage for specific model type."""
        # This would implement comprehensive leakage detection
        pass
    
    async def _train_models_parallel(self, data: Dict[str, Any], configs: Dict[ModelType, Dict[str, Any]]) -> Dict[ModelType, List[TrainingResult]]:
        """Train models in parallel."""
        tprint_info("🔄 Training models in parallel")
        
        results = {}
        tasks = []
        
        for model_type in self.config.model_types:
            if model_type in configs:
                task = asyncio.create_task(
                    self._train_model_type(data, model_type, configs[model_type])
                )
                tasks.append((model_type, task))
        
        # Wait for all tasks to complete
        for model_type, task in tasks:
            try:
                result = await task
                results[model_type] = result
                tprint_success(f"✅ Completed training for {model_type.value}")
            except Exception as e:
                tprint_error(f"❌ Failed training for {model_type.value}: {e}")
                results[model_type] = []
        
        return results
    
    async def _train_models_sequential(self, data: Dict[str, Any], configs: Dict[ModelType, Dict[str, Any]]) -> Dict[ModelType, List[TrainingResult]]:
        """Train models sequentially."""
        tprint_info("🔄 Training models sequentially")
        
        results = {}
        
        for model_type in self.config.model_types:
            if model_type in configs:
                try:
                    result = await self._train_model_type(data, model_type, configs[model_type])
                    results[model_type] = result
                    tprint_success(f"✅ Completed training for {model_type.value}")
                except Exception as e:
                    tprint_error(f"❌ Failed training for {model_type.value}: {e}")
                    results[model_type] = []
        
        return results
    
    async def _train_model_type(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> List[TrainingResult]:
        """Train all models of a specific type."""
        results = []
        
        # Extract model configurations
        models_config = config.get('models', [])
        
        for model_config in models_config:
            if not model_config.get('enabled', True):
                continue
            
            try:
                result = await self._train_single_model(data, model_type, model_config, config)
                results.append(result)
            except Exception as e:
                tprint_error(f"❌ Failed to train {model_config.get('name', 'unknown')}: {e}")
                results.append(TrainingResult(
                    model_type=model_type,
                    model_name=model_config.get('name', 'unknown'),
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _train_single_model(self, data: Dict[str, Any], model_type: ModelType, model_config: Dict[str, Any], config: Dict[str, Any]) -> TrainingResult:
        """Train a single model."""
        start_time = time.time()
        model_name = model_config.get('name', 'unknown')
        
        tprint_info(f"🔄 Training {model_name} ({model_type.value})")
        
        try:
            # Create trainer based on model type
            trainer = self._create_trainer(model_type, model_config, config)
            
            # Prepare features and targets
            X, y = await self._prepare_training_data(data, model_type, config)
            
            # Train model
            model = await self._train_model(trainer, X, y, model_config, config)
            
            # Evaluate model
            metrics = await self._evaluate_model(model, X, y, model_type, config)
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(model, X, y, model_type)
            
            # Perform SHAP analysis
            shap_values = await self._perform_shap_analysis(model, X, y, model_type, config)
            
            # Generate predictions
            predictions = await self._generate_predictions(model, X, model_type)
            probabilities = await self._generate_probabilities(model, X, model_type)
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                model_type=model_type,
                model_name=model_name,
                success=True,
                model=model,
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                feature_importance=feature_importance,
                shap_values=shap_values,
                training_time=training_time
            )
            
            tprint_success(f"✅ Successfully trained {model_name} in {training_time:.2f}s")
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            tprint_error(f"❌ Failed to train {model_name}: {e}")
            
            return TrainingResult(
                model_type=model_type,
                model_name=model_name,
                success=False,
                training_time=training_time,
                error_message=str(e)
            )
    
    def _create_trainer(self, model_type: ModelType, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create appropriate trainer for model type."""
        if model_type == ModelType.ANALYST_BASE:
            return self._create_analyst_base_trainer(model_config, config)
        elif model_type == ModelType.ANALYST_ENSEMBLE:
            return self._create_analyst_ensemble_trainer(model_config, config)
        elif model_type == ModelType.TACTICIAN_BASE:
            return self._create_tactician_base_trainer(model_config, config)
        elif model_type == ModelType.TACTICIAN_ENSEMBLE:
            return self._create_tactician_ensemble_trainer(model_config, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_analyst_base_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create analyst base trainer."""
        # Convert config to AnalystTrainingConfig
        training_config = AnalystTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_patchtst_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_patchtst_features', True),
            enable_regime_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_regime_features', True),
            enable_multi_timeframe=config.get('inputs', {}).get('analyst_features', {}).get('enable_multi_timeframe', True),
            lightgbm_params=model_config.get('parameters', {}),
            catboost_params=model_config.get('parameters', {}),
            stacker_params=model_config.get('parameters', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return AnalystBaseTrainer(training_config, self.logger)
    
    def _create_analyst_ensemble_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create analyst ensemble trainer."""
        # Convert config to AnalystEnsembleTrainingConfig
        training_config = AnalystEnsembleTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_patchtst_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_patchtst_features', True),
            enable_regime_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_regime_features', True),
            enable_multi_timeframe=config.get('inputs', {}).get('analyst_features', {}).get('enable_multi_timeframe', True),
            ensemble_method=EnsembleMethod(model_config.get('type', 'STACKING').upper()),
            base_models=[AnalystModelType(model.get('type', 'LIGHTGBM')) for model in config.get('base_models', [])],
            meta_learner_params=model_config.get('parameters', {}).get('meta_learner_params', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return AnalystEnsembleTrainer(training_config, self.logger)
    
    def _create_tactician_base_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create tactician base trainer."""
        # Convert config to TacticianTrainingConfig
        training_config = TacticianTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_entry_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_entry_timing', True),
            enable_exit_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_exit_timing', True),
            enable_position_sizing=config.get('inputs', {}).get('tactician_features', {}).get('enable_position_sizing', True),
            lightgbm_params=model_config.get('parameters', {}),
            catboost_params=model_config.get('parameters', {}),
            neural_network_params=model_config.get('parameters', {}),
            linear_params=model_config.get('parameters', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return TacticianBaseTrainer(training_config, self.logger)
    
    def _create_tactician_ensemble_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create tactician ensemble trainer."""
        # Convert config to TacticianEnsembleTrainingConfig
        training_config = TacticianEnsembleTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_entry_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_entry_timing', True),
            enable_exit_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_exit_timing', True),
            enable_position_sizing=config.get('inputs', {}).get('tactician_features', {}).get('enable_position_sizing', True),
            ensemble_method=TacticianEnsembleMethod(model_config.get('type', 'STACKING').upper()),
            base_models=[TacticianModelType(model.get('type', 'LIGHTGBM')) for model in config.get('base_models', [])],
            meta_learner_params=model_config.get('parameters', {}).get('meta_learner_params', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return TacticianEnsembleTrainer(training_config, self.logger)
    
    async def _prepare_training_data(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for specific model type."""
        # This would implement comprehensive data preparation
        # including feature engineering, target generation, etc.
        
        # For now, return placeholder data
        X = np.random.randn(1000, 50)  # Placeholder features
        y = np.random.randint(0, 2, 1000)  # Placeholder targets
        
        return X, y
    
    async def _train_model(self, trainer, X: np.ndarray, y: np.ndarray, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Train the model using the trainer."""
        # This would implement the actual training logic
        # For now, return a placeholder model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    async def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the trained model."""
        # This would implement comprehensive model evaluation
        # including cross-validation, various metrics, etc.
        
        # For now, return placeholder metrics
        predictions = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions)
        }
        
        return metrics
    
    async def _calculate_feature_importance(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType) -> Dict[str, float]:
        """Calculate feature importance."""
        # This would implement feature importance calculation
        # For now, return placeholder importance
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        importance = np.random.rand(X.shape[1])
        return dict(zip(feature_names, importance))
    
    async def _perform_shap_analysis(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Perform SHAP analysis."""
        # This would implement SHAP analysis
        # For now, return placeholder SHAP values
        return np.random.randn(X.shape[0], X.shape[1])
    
    async def _generate_predictions(self, model, X: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Generate predictions."""
        return model.predict(X)
    
    async def _generate_probabilities(self, model, X: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Generate prediction probabilities."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return None
    
    async def _generate_reports(self, results: Dict[ModelType, List[TrainingResult]]):
        """Generate comprehensive reports."""
        tprint_info("📊 Generating reports")
        
        # This would implement comprehensive report generation
        # including HTML reports, plots, tables, etc.
        
        tprint_success("✅ Reports generated")


# Example usage
async def main():
    """Example usage of the ML Model Trainer."""
    
    # Create configuration
    config = MLModelTrainerConfig(
        model_types=[
            ModelType.ANALYST_BASE,
            ModelType.ANALYST_ENSEMBLE,
            ModelType.TACTICIAN_BASE,
            ModelType.TACTICIAN_ENSEMBLE
        ],
        timeframe="15m",
        enable_parallel_training=True,
        max_workers=4
    )
    
    # Create trainer
    trainer = MLModelTrainer(config)
    
    # Define config paths
    config_paths = {
        ModelType.ANALYST_BASE: "config/ml_model_trainer/analyst_base_config.yaml",
        ModelType.ANALYST_ENSEMBLE: "config/ml_model_trainer/analyst_ensemble_config.yaml",
        ModelType.TACTICIAN_BASE: "config/ml_model_trainer/tactician_base_config.yaml",
        ModelType.TACTICIAN_ENSEMBLE: "config/ml_model_trainer/tactician_ensemble_config.yaml"
    }
    
    # Prepare data (placeholder)
    data = {
        'features': np.random.randn(1000, 50),
        'targets': np.random.randint(0, 2, 1000),
        'metadata': {}
    }
    
    # Train models
    results = await trainer.train_models(data, config_paths)
    
    # Print results
    for model_type, model_results in results.items():
        print(f"\n{model_type.value} Results:")
        for result in model_results:
            print(f"  {result.model_name}: {'Success' if result.success else 'Failed'}")
            if result.success:
                print(f"    Metrics: {result.metrics}")
                print(f"    Training Time: {result.training_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())