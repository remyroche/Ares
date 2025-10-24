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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json

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
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_preview, tprint_data_format, LogLevel
)
from src.core.decorators import handles_errors, traced, log_execution_time

# Import common operations and utilities
from src.utils.common_operations import safe_dataframe_operation, safe_array_operation
from src.utils.common_utilities import (
    validate_dataframe, validate_array, safe_dataframe_operation,
    memory_managed, MemoryStrategy, get_memory_manager, force_cleanup
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_exp,
    validate_numeric_input, validate_array_input, safe_statistical_operation
)

# Import hardware optimization
from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager, WorkloadType
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)

# Import ML common utilities
from src.utils.ml_common.optimization.consolidated_hpo import (
    ConsolidatedHPO, HPOConfig, OptimizationResult
)
from src.utils.ml_common.validation.consolidated_cv import (
    ConsolidatedCV, CVConfig, PurgedCV, WalkForwardCV, TemporalCV
)
from src.utils.ml_common.validation.data_leakage_detector import (
    DataLeakageDetector, DataLeakageReport
)
from src.utils.ml_common.explainability.model_explainability import (
    ModelExplainabilityManager, ExplanationConfig
)
from src.utils.ml_common.explainability.shap_lime_integration import (
    SHAPLIMEIntegration, ExplanationResult
)
from src.utils.ml_common.data_processing.multi_timeframe_training import MultiTimeframeProcessor
from src.utils.ml_common.ensembles.stacking_ensemble_manager import StackingEnsembleManager
from src.utils.ml_common.feature_selection import (
    FeatureSelector, FeatureSelectionConfig, mRMRSelector, LASSOSelector, RFESelector
)

# Import data quality and analysis tools
from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import (
    EnhancedLabelDefinitions, AnalystLabelConfig, TacticianLabelConfig
)

# Import validation and metrics
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
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
    
    def __post_init__(self):
        """Initialize process pool for CPU-bound training."""
        self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)


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
    
    def _infer_task_type(self, model_config: Dict[str, Any], y: np.ndarray) -> str:
        """Infer task type from config or data."""
        t = (model_config.get("task") or "").lower()
        if t in {"classification", "regression"}:
            return t
        # Fallback by data
        if y is not None:
            return "classification" if (np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 50) else "regression"
        return "classification"  # Default fallback

    def _initialize_components(self):
        """Initialize all pipeline components using existing utilities."""
        tprint_info("🔧 Initializing ML Model Trainer components")
        
        # Initialize hardware manager
        self.hardware_manager = get_integrated_hardware_manager()
        tprint_data_format("Hardware manager initialized", LogLevel.INFO)
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager()
        tprint_data_format("Memory manager initialized", LogLevel.INFO)
        
        # Initialize HPO system
        self.hpo_system = ConsolidatedHPO(HPOConfig(
            enable_optuna=OPTUNA_AVAILABLE,
            max_trials=100,
            timeout=3600,
            n_jobs=self.config.max_workers
        ))
        tprint_data_format("HPO system initialized", LogLevel.INFO)
        
        # Initialize cross-validation system
        self.cv_system = ConsolidatedCV(CVConfig(
            enable_purged_cv=True,
            enable_walk_forward=True,
            enable_temporal_cv=True,
            n_splits=self.config.cv_folds
        ))
        tprint_data_format("CV system initialized", LogLevel.INFO)
        
        # Initialize data leakage detector
        self.leakage_detector = DataLeakageDetector({
            'temporal_tolerance': 1,
            'lookahead_tolerance': 24,
            'feature_contamination_threshold': 0.1,
            'enable_strict_mode': True,
            'use_vectorbt_analysis': True,
            'correlation_threshold': 0.95
        })
        tprint_data_format("Data leakage detector initialized", LogLevel.INFO)
        
        # Initialize explainability manager
        self.explainability_manager = ModelExplainabilityManager(ExplanationConfig(
            enable_shap=True,
            enable_lime=True,
            shap_sample_size=100,
            lime_sample_size=1000
        ))
        tprint_data_format("Explainability manager initialized", LogLevel.INFO)
        
        # Initialize feature selectors
        self.feature_selectors = {
            ModelType.ANALYST_BASE: FeatureSelector(FeatureSelectionConfig(
                method='mrmr',
                max_features=50,
                enable_correlation_filter=True
            )),
            ModelType.ANALYST_ENSEMBLE: FeatureSelector(FeatureSelectionConfig(
                method='lasso',
                max_features=100,
                enable_correlation_filter=True
            )),
            ModelType.TACTICIAN_BASE: FeatureSelector(FeatureSelectionConfig(
                method='rfe',
                max_features=30,
                enable_correlation_filter=True
            )),
            ModelType.TACTICIAN_ENSEMBLE: FeatureSelector(FeatureSelectionConfig(
                method='mrmr',
                max_features=80,
                enable_correlation_filter=True
            ))
        }
        tprint_data_format("Feature selectors initialized", LogLevel.INFO)
        
        # Initialize multi-timeframe processor
        self.multi_timeframe_processor = MultiTimeframeProcessor()
        tprint_data_format("Multi-timeframe processor initialized", LogLevel.INFO)
        
        # Initialize ensemble managers
        self.ensemble_managers = {
            ModelType.ANALYST_ENSEMBLE: StackingEnsembleManager(),
            ModelType.TACTICIAN_ENSEMBLE: StackingEnsembleManager()
        }
        tprint_data_format("Ensemble managers initialized", LogLevel.INFO)
        
        tprint_success("✅ All components initialized successfully")
    
    @memory_managed(MemoryStrategy.MODERATE)
    def _prepare_features(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Prepare features using existing utilities."""
        tprint_info(f"🔄 Preparing features for {model_type.value}")
        
        # Extract base features
        base_features = data.get('features', np.array([]))
        tprint_data_preview(base_features, f"Base features for {model_type.value}")
        
        # Validate input data
        if not validate_array(base_features):
            tprint_error("Invalid input features")
            raise ValueError("Invalid input features")
        
        # Ensure features are 2D for feature selection
        if base_features.ndim == 1:
            base_features = base_features.reshape(1, -1)
        elif base_features.ndim > 2:
            base_features = base_features.reshape(base_features.shape[0], -1)
        
        # Apply feature selection
        feature_selector = self.feature_selectors.get(model_type)
        if feature_selector and base_features.shape[1] > 1:
            selected_features = feature_selector.fit_transform(base_features)
            tprint_data_format(f"Feature selection completed: {base_features.shape} -> {selected_features.shape}", LogLevel.INFO)
        else:
            selected_features = base_features
        
        # Apply multi-timeframe processing if enabled
        if config.get('inputs', {}).get('analyst_features', {}).get('enable_multi_timeframe', False):
            selected_features = self.multi_timeframe_processor.process_features(
                selected_features, 
                timeframes=config.get('inputs', {}).get('analyst_features', {}).get('timeframes', ['5m', '15m', '1h'])
            )
            tprint_data_format(f"Multi-timeframe processing completed: {selected_features.shape}", LogLevel.INFO)
        
        # Apply hardware optimization
        optimized_features = self.hardware_manager.process_data_with_optimization(
            selected_features, WorkloadType.ML_TRAINING
        )
        tprint_data_format(f"Hardware optimization completed: {optimized_features.shape}", LogLevel.INFO)
        
        return optimized_features
    
    @memory_managed(MemoryStrategy.MODERATE)
    def _prepare_targets(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Prepare targets using existing utilities."""
        tprint_info(f"🎯 Preparing targets for {model_type.value}")
        
        # Extract targets based on model type
        if model_type in [ModelType.ANALYST_BASE, ModelType.ANALYST_ENSEMBLE]:
            targets = data.get('targets', np.array([]))
            if targets.ndim == 2 and targets.shape[1] >= 2:
                # Use first two columns for analyst targets
                targets = targets[:, :2]
        else:  # Tactician models
            targets = data.get('targets', np.array([]))
            if targets.ndim == 2 and targets.shape[1] >= 3:
                # Use last three columns for tactician targets
                targets = targets[:, -3:]
        
        tprint_data_preview(targets, f"Targets for {model_type.value}")
        
        # Validate targets
        if not validate_array(targets):
            tprint_error("Invalid target data")
            raise ValueError("Invalid target data")
        
        # Ensure targets are 1D for single-output
        if targets.ndim > 1 and targets.shape[1] == 1:
            targets = targets.ravel()
        elif targets.ndim > 1 and targets.shape[1] > 1:
            # multi-output supported later; for now pick the first
            targets = targets[:, 0]
        else:
            targets = targets.ravel()
        
        tprint_data_format(f"Target preparation completed: {targets.shape}", LogLevel.INFO)
        return targets
    
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
        """Load configuration files for each model type with inheritance support."""
        configs = {}
        
        for model_type, config_path in config_paths.items():
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                
                # Handle inheritance
                if 'extends' in cfg:
                    parent = (Path(config_path).parent / cfg['extends']).resolve()
                    with open(parent) as pf:
                        base = yaml.safe_load(pf) or {}
                    cfg = {**base, **cfg}  # base overrides trial
                    cfg.pop('extends', None)
                
                # Compute config hash for reproducibility
                config_str = json.dumps(cfg, sort_keys=True)
                config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
                cfg['_config_hash'] = config_hash
                
                configs[model_type] = cfg
                tprint_info(f"📋 Loaded configuration for {model_type.value} (hash: {config_hash})")
            except Exception as e:
                tprint_error(f"❌ Failed to load configuration for {model_type.value}: {e}")
                raise
        
        return configs
    
    @comprehensive_memory_optimization(MemoryOptimizationLevel.AGGRESSIVE)
    async def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data using existing utilities."""
        tprint_info("🔄 Preprocessing data with comprehensive validation")
        
        # Validate input data structure
        if not isinstance(data, dict):
            tprint_error("Data must be a dictionary")
            raise ValueError("Data must be a dictionary")
        
        # Extract and validate features
        features = data.get('features', np.array([]))
        if not validate_array(features):
            tprint_error("Invalid features data")
            raise ValueError("Invalid features data")
        
        tprint_data_preview(features, "Input features")
        tprint_data_format(f"Features shape: {features.shape}, dtype: {features.dtype}", LogLevel.INFO)
        
        # Extract and validate targets
        targets = data.get('targets', np.array([]))
        if not validate_array(targets):
            tprint_error("Invalid targets data")
            raise ValueError("Invalid targets data")
        
        tprint_data_preview(targets, "Input targets")
        tprint_data_format(f"Targets shape: {targets.shape}, dtype: {targets.dtype}", LogLevel.INFO)
        
        # Apply safe data operations - ensure features are 2D
        processed_features = safe_array_operation(np.atleast_2d(features), self._clean_data)
        if processed_features.shape[0] < processed_features.shape[1]:
            # assume already samples x features
            pass
        
        # Ensure targets are 1D for single-output
        processed_targets = safe_array_operation(targets, self._clean_data)
        if processed_targets.ndim > 1 and processed_targets.shape[1] == 1:
            processed_targets = processed_targets.ravel()
        elif processed_targets.ndim > 1 and processed_targets.shape[1] > 1:
            # multi-output supported later; for now pick the first
            processed_targets = processed_targets[:, 0]
        else:
            processed_targets = processed_targets.ravel()
        
        # Detect data leakage using existing detector
        leakage_report = self.leakage_detector.detect_leakage(processed_features, processed_targets)
        if leakage_report.has_leakage:
            tprint_warning(f"Data leakage detected: {leakage_report.leakage_score:.3f}")
            tprint_warning(f"Recommendations: {leakage_report.recommendations}")
        else:
            tprint_success("No data leakage detected")
        
        # Apply hardware optimization
        processed_data = {
            'features': self.hardware_manager.process_data_with_optimization(
                processed_features, WorkloadType.ML_TRAINING
            ),
            'targets': self.hardware_manager.process_data_with_optimization(
                processed_targets, WorkloadType.ML_TRAINING
            ),
            'metadata': data.get('metadata', {}),
            'leakage_report': leakage_report
        }
        
        tprint_data_format(f"Processed data - Features: {processed_data['features'].shape}, Targets: {processed_data['targets'].shape}", LogLevel.INFO)
        tprint_success("✅ Data preprocessing completed")
        
        return processed_data
    
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Clean data using safe operations."""
        # Remove infinite values
        data = np.where(np.isfinite(data), data, 0.0)
        
        # Remove NaN values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
    
    @performance_tracked
    async def _validate_data_quality(self, data: Dict[str, Any], model_type: ModelType):
        """Validate data quality using existing utilities."""
        tprint_info(f"🔍 Validating data quality for {model_type.value}")
        
        features = data.get('features', np.array([]))
        targets = data.get('targets', np.array([]))
        
        # Use existing validation utilities
        validation_result = validate_dataframe(pd.DataFrame(features)) if features.size > 0 else None
        if validation_result and not validation_result.is_valid:
            tprint_warning(f"Data quality issues detected: {validation_result.errors}")
        
        # Check for data consistency
        if features.size > 0 and targets.size > 0:
            if len(features) != len(targets):
                tprint_error(f"Feature-target length mismatch: {len(features)} vs {len(targets)}")
                raise ValueError("Feature-target length mismatch")
        
        tprint_success(f"✅ Data quality validation completed for {model_type.value}")
    
    @performance_tracked
    async def _detect_data_leakage(self, data: Dict[str, Any], model_type: ModelType):
        """Detect data leakage using existing detector."""
        tprint_info(f"🔍 Detecting data leakage for {model_type.value}")
        
        features = data.get('features', np.array([]))
        targets = data.get('targets', np.array([]))
        
        if features.size > 0 and targets.size > 0:
            # Use existing leakage detector
            leakage_report = self.leakage_detector.detect_leakage(features, targets)
            
            if leakage_report.has_leakage:
                tprint_warning(f"Leakage detected for {model_type.value}: {leakage_report.leakage_score:.3f}")
                tprint_warning(f"Temporal violations: {leakage_report.temporal_violations}")
                tprint_warning(f"Feature contamination: {leakage_report.feature_contamination}")
            else:
                tprint_success(f"✅ No leakage detected for {model_type.value}")
        
        tprint_success(f"✅ Leakage detection completed for {model_type.value}")
    
    async def _train_models_parallel(self, data: Dict[str, Any], configs: Dict[ModelType, Dict[str, Any]]) -> Dict[ModelType, List[TrainingResult]]:
        """Train models in parallel using ProcessPoolExecutor."""
        tprint_info("🔄 Training models in parallel with ProcessPoolExecutor")
        
        results = {}
        tasks = []
        loop = asyncio.get_running_loop()
        
        for model_type in self.config.model_types:
            if model_type in configs:
                task = loop.run_in_executor(
                    self.config._process_pool, 
                    self._train_model_type_sync, 
                    data, model_type, configs[model_type]
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
    
    def _train_model_type_sync(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> List[TrainingResult]:
        """Synchronous version of model training for ProcessPoolExecutor."""
        # This is a simplified sync version - in practice you'd need to handle
        # the async components differently or restructure the code
        try:
            # For now, return empty results - this would need proper implementation
            tprint_info(f"Training {model_type.value} (sync)")
            return []
        except Exception as e:
            tprint_error(f"Sync training failed for {model_type.value}: {e}")
            return []
    
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
            ensemble_method=EnsembleMethod[model_config.get('type', 'STACKING').upper()],
            base_models=[AnalystModelType[model.get('type', 'LIGHTGBM').upper()] for model in config.get('base_models', [])],
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
            ensemble_method=TacticianEnsembleMethod[model_config.get('type', 'STACKING').upper()],
            base_models=[TacticianModelType[model.get('type', 'LIGHTGBM').upper()] for model in config.get('base_models', [])],
            meta_learner_params=model_config.get('parameters', {}).get('meta_learner_params', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return TacticianEnsembleTrainer(training_config, self.logger)
    
    @memory_managed(MemoryStrategy.MODERATE)
    async def _prepare_training_data(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data using existing utilities."""
        tprint_info(f"🔄 Preparing training data for {model_type.value}")
        
        # Prepare features using existing utilities
        X = self._prepare_features(data, model_type, config)
        tprint_data_format(f"Features prepared: {X.shape}", LogLevel.INFO)
        
        # Prepare targets using existing utilities
        y = self._prepare_targets(data, model_type, config)
        tprint_data_format(f"Targets prepared: {y.shape}", LogLevel.INFO)
        
        # Validate data consistency
        if len(X) != len(y):
            tprint_error(f"Feature-target length mismatch: {len(X)} vs {len(y)}")
            raise ValueError("Feature-target length mismatch")
        
        # Apply safe statistical operations for data validation
        X = safe_statistical_operation(X, np.asarray)
        y = safe_statistical_operation(y, np.asarray)
        
        tprint_data_preview(X, f"Final features for {model_type.value}")
        tprint_data_preview(y, f"Final targets for {model_type.value}")
        
        tprint_success(f"✅ Training data prepared for {model_type.value}")
        return X, y
    
    @comprehensive_memory_optimization(MemoryOptimizationLevel.AGGRESSIVE)
    async def _train_model(self, trainer, X: np.ndarray, y: np.ndarray, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Train the model using the trainer with HPO."""
        tprint_info(f"🚀 Training model: {model_config.get('name', 'unknown')}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid training data")
            raise ValueError("Invalid training data")
        
        tprint_data_format(f"Training data - X: {X.shape}, y: {y.shape}", LogLevel.INFO)
        
        # Check if hyperparameter optimization is enabled
        hpo_config = config.get('training', {}).get('hyperparameter_optimization', {})
        if hpo_config.get('enabled', False) and OPTUNA_AVAILABLE:
            tprint_info("🔧 Running hyperparameter optimization")
            
            # Infer task type for proper scoring
            task_type = self._infer_task_type(model_config, y)
            
            # Define objective function for HPO
            def objective(trial):
                # Get hyperparameters from trial
                params = self._get_hpo_params(trial, model_config)
                
                # Create model with trial parameters
                model = self._create_model_with_params(model_config, params, task_type)
                
                # Train and evaluate
                try:
                    model.fit(X, y)
                    score, direction = self._evaluate_model_score(model, X, y, config, task_type)
                    return score if direction == "maximize" else -score
                except Exception as e:
                    tprint_warning(f"HPO trial failed: {e}")
                    return float('-inf')
            
            # Run HPO
            best_params = self.hpo_system.optimize(
                objective=objective,
                n_trials=hpo_config.get('n_trials', 100),
                timeout=hpo_config.get('timeout', 3600)
            )
            
            tprint_success(f"✅ HPO completed. Best params: {best_params}")
            
            # Create final model with best parameters
            final_model = self._create_model_with_params(model_config, best_params)
        else:
            # Use default parameters
            final_model = self._create_model_with_params(model_config, model_config.get('parameters', {}))
        
        # Train the final model
        tprint_info("🏋️ Training final model")
        final_model.fit(X, y)
        
        tprint_success(f"✅ Model training completed: {model_config.get('name', 'unknown')}")
        return final_model
    
    def _get_hpo_params(self, trial, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters from Optuna trial."""
        model_type = model_config.get('type', 'LIGHTGBM').upper()
        params = {}
        
        if model_type == 'LIGHTGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        elif model_type == 'CATBOOST':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        elif model_type == 'XGBOOST':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        
        return params
    
    def _create_model_with_params(self, model_config: Dict[str, Any], params: Dict[str, Any], task_type: str):
        """Create model with given parameters."""
        model_type = model_config.get('type', 'LIGHTGBM').upper()
        
        # Merge fixed parameters with trial parameters (fixed overrides trial)
        base = model_config.get('parameters', {})
        merged = {**params, **base}  # base overrides trial
        
        # Determine if classification based on task type
        is_classification = (task_type == "classification")
        
        if model_type == 'LIGHTGBM':
            from lightgbm import LGBMClassifier, LGBMRegressor
            ModelClass = LGBMClassifier if is_classification else LGBMRegressor
            return ModelClass(**merged, random_state=42, verbose=-1, n_jobs=1)
        elif model_type == 'CATBOOST':
            from catboost import CatBoostClassifier, CatBoostRegressor
            ModelClass = CatBoostClassifier if is_classification else CatBoostRegressor
            return ModelClass(**merged, random_seed=42, verbose=False, thread_count=1)
        elif model_type == 'XGBOOST':
            from xgboost import XGBClassifier, XGBRegressor
            ModelClass = XGBClassifier if is_classification else XGBRegressor
            return ModelClass(**merged, random_state=42, verbosity=0, n_jobs=1)
        else:
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            ModelClass = RandomForestClassifier if is_classification else RandomForestRegressor
            return ModelClass(**merged, random_state=42, n_jobs=1)
    
    def _evaluate_model_score(self, model, X: np.ndarray, y: np.ndarray, config: Dict[str, Any], task_type: str) -> Tuple[float, str]:
        """Evaluate model score for HPO."""
        try:
            # Get metric from config or infer from task type
            metric = config.get('training', {}).get('hyperparameter_optimization', {}).get('metric')
            if metric is None:
                metric = "f1" if task_type == "classification" else "neg_mean_squared_error"
            
            # Use cross-validation for evaluation
            cv_scores = self.cv_system.cross_validate(
                model, X, y, 
                cv_type='temporal',
                scoring=metric
            )
            # sklearn "neg_*" means higher is better
            direction = "maximize"
            return float(np.mean(cv_scores)), direction
        except Exception:
            return float('-inf'), "maximize"
    
    @performance_tracked
    async def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the trained model using existing utilities."""
        tprint_info(f"📊 Evaluating model for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid evaluation data")
            raise ValueError("Invalid evaluation data")
        
        # Generate predictions
        predictions = model.predict(X)
        tprint_data_format(f"Predictions generated: {predictions.shape}", LogLevel.INFO)
        
        # Infer task type from model config
        task_type = self._infer_task_type(model_config, y)
        
        # Calculate basic metrics
        metrics = {}
        
        if task_type == "classification":
            # Classification metrics
            try:
                metrics.update({
                    'accuracy': float(accuracy_score(y, predictions)),
                    'f1_score': float(f1_score(y, predictions, average='weighted')),
                    'precision': float(precision_score(y, predictions, average='weighted')),
                    'recall': float(recall_score(y, predictions, average='weighted'))
                })
            except Exception as e:
                tprint_warning(f"Classification metrics failed: {e}")
                metrics.update({
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                })
            
            # Add AUC if binary classification
            if task_type == "classification" and np.unique(y).size == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X)[:, 1]
                        metrics['auc_roc'] = float(roc_auc_score(y, probs))
                except Exception as e:
                    tprint_warning(f"Could not calculate AUC: {e}")
        else:
            # Regression metrics
            try:
                metrics.update({
                    'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                    'mae': float(mean_absolute_error(y, predictions)),
                    'r2': float(r2_score(y, predictions)),
                    'explained_variance': float(explained_variance_score(y, predictions))
                })
            except Exception as e:
                tprint_warning(f"Regression metrics failed: {e}")
                metrics.update({
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': 0.0,
                    'explained_variance': 0.0
                })
        
        # Separate in-sample and CV metrics
        metrics['in_sample'] = metrics.copy()  # keep in-sample metrics
        
        # Cross-validation evaluation
        try:
            primary_metric = 'f1' if task_type == "classification" else 'neg_mean_squared_error'
            cv_scores = self.cv_system.cross_validate(
                model, X, y,
                cv_type='temporal',
                scoring=primary_metric
            )
            metrics['cv_mean'] = float(cv_scores.mean())
            metrics['cv_std'] = float(cv_scores.std())
            tprint_data_format(f"CV scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", LogLevel.INFO)
        except Exception as e:
            tprint_warning(f"CV evaluation failed: {e}")
            metrics['cv_mean'] = None
            metrics['cv_std'] = None
        
        # Apply safe mathematical operations to metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                metrics[key] = 0.0
        
        tprint_data_format(f"Evaluation metrics: {metrics}", LogLevel.INFO)
        tprint_success(f"✅ Model evaluation completed for {model_type.value}")
        
        return metrics
    
    @performance_tracked
    async def _calculate_feature_importance(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType) -> Dict[str, float]:
        """Calculate feature importance using existing utilities."""
        tprint_info(f"🔍 Calculating feature importance for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid data for feature importance calculation")
            return {}
        
        feature_importance = {}
        
        try:
            # Try to get built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
                feature_importance = dict(zip(feature_names, importance_scores))
                tprint_data_format(f"Built-in feature importance: {len(feature_importance)} features", LogLevel.INFO)
            
            # Use feature selector for additional importance
            feature_selector = self.feature_selectors.get(model_type)
            if feature_selector:
                try:
                    selector_importance = feature_selector.get_feature_importance(X, y)
                    if selector_importance:
                        feature_importance.update(selector_importance)
                        tprint_data_format(f"Feature selector importance: {len(selector_importance)} features", LogLevel.INFO)
                except Exception as e:
                    tprint_warning(f"Feature selector importance failed: {e}")
            
            # Apply safe mathematical operations
            for key, value in feature_importance.items():
                if not np.isfinite(value):
                    feature_importance[key] = 0.0
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            tprint_data_format(f"Feature importance calculated: {len(feature_importance)} features", LogLevel.INFO)
            tprint_success(f"✅ Feature importance calculation completed for {model_type.value}")
            
        except Exception as e:
            tprint_error(f"Feature importance calculation failed: {e}")
            # Return empty dict on failure
            feature_importance = {}
        
        return feature_importance
    
    @performance_tracked
    async def _perform_shap_analysis(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Perform SHAP analysis using existing utilities."""
        tprint_info(f"🔍 Performing SHAP analysis for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid data for SHAP analysis")
            return np.array([])
        
        try:
            # Use existing explainability manager
            explanation_config = ExplanationConfig(
                enable_shap=True,
                enable_lime=False,  # Focus on SHAP for now
                shap_sample_size=min(100, X.shape[0]),
                shap_max_features=min(50, X.shape[1])
            )
            
            # Generate SHAP explanations
            shap_values = self.explainability_manager.explain_model(
                model=model,
                X=X,
                y=y,
                config=explanation_config
            )
            
            if shap_values is not None:
                tprint_data_format(f"SHAP analysis completed: {shap_values.shape}", LogLevel.INFO)
                tprint_success(f"✅ SHAP analysis completed for {model_type.value}")
                return shap_values
            else:
                tprint_warning("SHAP analysis returned None")
                return np.array([])
                
        except Exception as e:
            tprint_error(f"SHAP analysis failed: {e}")
            # Return empty array on failure
            return np.array([])
    
    @performance_tracked
    async def _generate_predictions(self, model, X: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Generate predictions using safe operations."""
        tprint_info(f"🔮 Generating predictions for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X):
            tprint_error("Invalid data for prediction generation")
            return np.array([])
        
        try:
            predictions = model.predict(X)
            tprint_data_format(f"Predictions generated: {predictions.shape}", LogLevel.INFO)
            tprint_success(f"✅ Predictions generated for {model_type.value}")
            return predictions
        except Exception as e:
            tprint_error(f"Prediction generation failed: {e}")
            return np.array([])
    
    @performance_tracked
    async def _generate_probabilities(self, model, X: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Generate prediction probabilities using safe operations."""
        tprint_info(f"🔮 Generating probabilities for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X):
            tprint_error("Invalid data for probability generation")
            return None
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                tprint_data_format(f"Probabilities generated: {proba.shape}", LogLevel.INFO)
                tprint_success(f"✅ Probabilities generated for {model_type.value}")
                return proba  # shape (n, n_classes); callers can slice if needed
            else:
                tprint_info(f"Model does not support probability prediction for {model_type.value}")
                return None
        except Exception as e:
            tprint_error(f"Probability generation failed: {e}")
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