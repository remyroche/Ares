"""
Stacking Ensemble Training for Multi-Output Models

This module provides comprehensive training integration for the Analyst (5m) and
Tactician (1m) multi-output stacking ensemble system.

Key Features:
- AnalystStackingTrainer for 5m models
- TacticianStackingTrainer for 1m models
- StackingEnsembleConfig dataclass
- Integration with existing training pipeline
- M1 hardware optimization integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

# M1 Optimization imports
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.memory_optimization import get_memory_manager, MemoryMonitor

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError
)

# Import multi-output models and ensemble manager
from src.utils.ml_common.model_factory import EnhancedModelFactory, ModelConfig, ModelType
from src.utils.ml_common.multi_output_models import MultiOutputConfig, MultiOutputStackingModel
from src.utils.ml_common.stacking_ensemble_manager import StackingEnsembleManager, StackingEnsembleConfig
from src.utils.ml_common.stacking_confidence_calibration import StackingConfidenceCalibrator, StackingCalibrationConfig

logger = logging.getLogger(__name__)


@dataclass
class StackingEnsembleConfig:
    """Configuration for stacking ensemble training."""
    # Basic configuration
    trainer_name: str
    output_dir: str
    
    # Model configuration
    model_type: str  # "analyst" or "tactician"
    n_outputs: int = 4
    output_names: List[str] = field(default_factory=lambda: ["output_1", "output_2", "output_3", "output_4"])
    
    # Base model configuration
    base_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta_models: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Stacking configuration
    stacking_method: str = "blending"
    enable_meta_learning: bool = True
    meta_learning_rate: float = 0.01
    meta_learning_iterations: int = 1000
    
    # Multi-output specific settings
    output_weights: Optional[List[float]] = None
    output_loss_weights: Optional[List[float]] = None
    enable_output_correlation: bool = True
    correlation_threshold: float = 0.7
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_online_learning: bool = False
    
    # Output settings
    save_models: bool = True
    save_predictions: bool = True
    generate_reports: bool = True


@dataclass
class StackingTrainingResult:
    """Result from stacking ensemble training."""
    # Basic info
    trainer_name: str
    model_type: str
    n_outputs: int
    output_names: List[str]
    created_at: datetime
    total_duration: float
    
    # Training results
    ensemble_performance: Dict[str, float] = field(default_factory=dict)
    per_output_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Model information
    base_model_count: int = 0
    meta_model_count: int = 0
    base_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    meta_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_probabilities: Optional[np.ndarray] = None
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Model characteristics
    model_weights: Optional[np.ndarray] = None
    output_correlations: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, Any]] = None
    
    # Metadata
    config: StackingEnsembleConfig = field(default_factory=StackingEnsembleConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class AnalystStackingTrainer:
    """Analyst (5m) stacking ensemble trainer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Analyst stacking trainer."""
        self.logger = logger.getChild('AnalystStackingTrainer')
        self.logger.info("🚀 Initializing AnalystStackingTrainer...")
        start_time = time.time()
        
        self.config = config or {}
        
        # Initialize M1 optimizers
        self.logger.debug("🔧 Initializing M1 optimizers...")
        self.m1_gpu = get_m1_memory_optimizer() if self.config.get('enable_gpu_acceleration', True) else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        ) if self.config.get('enable_memory_optimization', True) else None
        self.m1_cpu = get_memory_manager() if self.config.get('enable_parallel_processing', True) else None
        
        self.logger.debug("✅ M1 optimizers initialized")
        
        # Initialize model factory
        self.model_factory = EnhancedModelFactory(self.config)
        
        # Analyst-specific configuration
        self.output_names = ["signal_strength", "confidence", "risk_score", "regime_label"]
        self.n_outputs = len(self.output_names)
        
        # Initialize base models
        self.base_models = self._initialize_analyst_base_models()
        
        # Initialize meta models
        self.meta_models = self._initialize_analyst_meta_models()
        
        # Initialize ensemble manager
        self.ensemble_manager = self._initialize_ensemble_manager()
        
        # Initialize confidence calibrator
        self.confidence_calibrator = self._initialize_confidence_calibrator()
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ AnalystStackingTrainer initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Outputs: {self.n_outputs} ({self.output_names})")
        self.logger.info(f"🎯 Base models: {len(self.base_models)}")
        self.logger.info(f"🎯 Meta models: {len(self.meta_models)}")
    
    def _initialize_analyst_base_models(self) -> Dict[str, Any]:
        """Initialize Analyst base models."""
        
        self.logger.debug("🔧 Initializing Analyst base models...")
        
        # Analyst fixed models
        analyst_models = {
            "transformer": ModelType.TIME_SERIES_TRANSFORMER,
            "catboost": ModelType.CATBOOST,
            "lightgbm": ModelType.LIGHTGBM,
            "ensemble_rf": ModelType.RANDOM_FOREST
        }
        
        base_models = {}
        
        for output_name in self.output_names:
            base_models[output_name] = {}
            
            for model_name, model_type in analyst_models.items():
                config = ModelConfig(
                    model_type=model_type,
                    model_name=f"analyst_{model_name}_{output_name}",
                    is_multi_output=False,  # Base models are single-output
                    n_outputs=1,
                    output_names=[output_name],
                    enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', True),
                    enable_memory_optimization=self.config.get('enable_memory_optimization', True),
                    enable_parallel_processing=self.config.get('enable_parallel_processing', True),
                    memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
                )
                
                model = self.model_factory.create_model(config)
                base_models[output_name][model_name] = model
                self.logger.debug(f"✅ Created {model_name} for {output_name}")
        
        self.logger.info(f"✅ Initialized {sum(len(models) for models in base_models.values())} base models")
        return base_models
    
    def _initialize_analyst_meta_models(self) -> Dict[str, Any]:
        """Initialize Analyst meta models."""
        
        self.logger.debug("🔧 Initializing Analyst meta models...")
        
        meta_models = {}
        
        for output_name in self.output_names:
            # Use Ridge regression as meta model for each output
            config = ModelConfig(
                model_type=ModelType.RIDGE,
                model_name=f"analyst_meta_{output_name}",
                is_multi_output=False,
                n_outputs=1,
                output_names=[output_name],
                model_params={
                    'alpha': 1.0,
                    'random_state': 42
                }
            )
            
            model = self.model_factory.create_model(config)
            meta_models[output_name] = model
            self.logger.debug(f"✅ Created meta model for {output_name}")
        
        self.logger.info(f"✅ Initialized {len(meta_models)} meta models")
        return meta_models
    
    def _initialize_ensemble_manager(self) -> StackingEnsembleManager:
        """Initialize the ensemble manager."""
        
        self.logger.debug("🔧 Initializing ensemble manager...")
        
        config = StackingEnsembleConfig(
            ensemble_name="analyst_ensemble",
            output_dir=self.config.get('output_dir', './analyst_ensemble'),
            n_outputs=self.n_outputs,
            output_names=self.output_names,
            base_models=self.base_models,
            meta_models=self.meta_models,
            stacking_method=self.config.get('stacking_method', 'blending'),
            enable_meta_learning=self.config.get('enable_meta_learning', True),
            enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', True),
            enable_memory_optimization=self.config.get('enable_memory_optimization', True),
            enable_parallel_processing=self.config.get('enable_parallel_processing', True),
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        )
        
        manager = StackingEnsembleManager(config)
        self.logger.debug("✅ Ensemble manager initialized")
        return manager
    
    def _initialize_confidence_calibrator(self) -> StackingConfidenceCalibrator:
        """Initialize the confidence calibrator."""
        
        self.logger.debug("🔧 Initializing confidence calibrator...")
        
        config = StackingCalibrationConfig(
            calibrator_name="analyst_calibrator",
            n_outputs=self.n_outputs,
            output_names=self.output_names,
            enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', True),
            enable_memory_optimization=self.config.get('enable_memory_optimization', True),
            enable_parallel_processing=self.config.get('enable_parallel_processing', True),
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        )
        
        calibrator = StackingConfidenceCalibrator(config)
        self.logger.debug("✅ Confidence calibrator initialized")
        return calibrator
    
    @traced(span_name='train_analyst_ensemble')
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.DataFrame] = None) -> StackingTrainingResult:
        """Train the Analyst stacking ensemble."""
        
        self.logger.info("🚀 Training Analyst stacking ensemble...")
        start_time = time.time()
        
        self.logger.info(f"📊 Training data shape: {X_train.shape}")
        self.logger.info(f"📊 Target data shape: {y_train.shape}")
        if X_val is not None:
            self.logger.info(f"📊 Validation data shape: {X_val.shape}")
        if y_val is not None:
            self.logger.info(f"📊 Validation target shape: {y_val.shape}")
        
        try:
            # Train the ensemble
            self.logger.info("🔄 Training ensemble...")
            ensemble_result = self.ensemble_manager.train_ensemble(X_train, y_train, X_val, y_val)
            
            # Calibrate confidence if validation data is available
            if X_val is not None and y_val is not None:
                self.logger.info("🔄 Calibrating confidence...")
                y_pred = ensemble_result.predictions
                calibration_result = self.confidence_calibrator.calibrate_confidence(y_val.values, y_pred)
                
                # Update ensemble result with calibrated predictions
                ensemble_result.predictions = calibration_result.calibrated_predictions
                ensemble_result.confidence_scores = calibration_result.calibrated_predictions
            else:
                self.logger.warning("⚠️ No validation data provided, skipping confidence calibration")
            
            # Create training result
            result = StackingTrainingResult(
                trainer_name="analyst_stacking_trainer",
                model_type="analyst",
                n_outputs=self.n_outputs,
                output_names=self.output_names,
                created_at=datetime.now(),
                total_duration=0.0,  # Will be set by caller
                ensemble_performance=ensemble_result.ensemble_performance,
                per_output_performance=ensemble_result.per_output_performance,
                base_model_count=ensemble_result.base_model_count,
                meta_model_count=ensemble_result.meta_model_count,
                base_model_performance=ensemble_result.base_model_performance,
                meta_model_performance=ensemble_result.meta_model_performance,
                predictions=ensemble_result.predictions,
                prediction_probabilities=ensemble_result.prediction_probabilities,
                confidence_scores=ensemble_result.confidence_scores,
                model_weights=ensemble_result.model_weights,
                output_correlations=ensemble_result.output_correlations,
                feature_importance=ensemble_result.feature_importance,
                config=StackingEnsembleConfig(
                    trainer_name="analyst_stacking_trainer",
                    output_dir=self.config.get('output_dir', './analyst_ensemble'),
                    model_type="analyst",
                    n_outputs=self.n_outputs,
                    output_names=self.output_names
                ),
                optimization_used=ensemble_result.optimization_used
            )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Log memory usage
            if self.m1_memory:
                result.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
                self.logger.info(f"🧠 Memory usage: {result.memory_usage_mb:.1f} MB")
            
            self.logger.info(f"✅ Analyst stacking ensemble trained in {execution_time:.2f}s")
            self.logger.info(f"📊 Ensemble performance: {result.ensemble_performance}")
            self.logger.info(f"🎯 Base models: {result.base_model_count}")
            self.logger.info(f"🎯 Meta models: {result.meta_model_count}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Failed to train Analyst ensemble after {execution_time:.3f}s: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Make predictions using the trained ensemble."""
        
        if not self.ensemble_manager.stacking_model.is_fitted:
            raise ValueError("Ensemble not trained yet")
        
        return self.ensemble_manager.predict(X)
    
    def save_ensemble(self, file_path: str) -> None:
        """Save the trained ensemble."""
        
        try:
            # Save ensemble manager
            ensemble_path = file_path.replace('.pkl', '_ensemble.pkl')
            self.ensemble_manager.save_ensemble(ensemble_path)
            
            # Save confidence calibrator
            calibrator_path = file_path.replace('.pkl', '_calibrator.pkl')
            self.confidence_calibrator.save_calibrator(calibrator_path)
            
            self.logger.info(f"💾 Analyst ensemble saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save Analyst ensemble: {e}")
            raise
    
    def load_ensemble(self, file_path: str) -> None:
        """Load the trained ensemble."""
        
        try:
            # Load ensemble manager
            ensemble_path = file_path.replace('.pkl', '_ensemble.pkl')
            self.ensemble_manager.load_ensemble(ensemble_path)
            
            # Load confidence calibrator
            calibrator_path = file_path.replace('.pkl', '_calibrator.pkl')
            self.confidence_calibrator.load_calibrator(calibrator_path)
            
            self.logger.info(f"📂 Analyst ensemble loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Analyst ensemble: {e}")
            raise


class TacticianStackingTrainer:
    """Tactician (1m) stacking ensemble trainer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Tactician stacking trainer."""
        self.logger = logger.getChild('TacticianStackingTrainer')
        self.logger.info("🚀 Initializing TacticianStackingTrainer...")
        start_time = time.time()
        
        self.config = config or {}
        
        # Initialize M1 optimizers
        self.logger.debug("🔧 Initializing M1 optimizers...")
        self.m1_gpu = get_m1_memory_optimizer() if self.config.get('enable_gpu_acceleration', True) else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        ) if self.config.get('enable_memory_optimization', True) else None
        self.m1_cpu = get_memory_manager() if self.config.get('enable_parallel_processing', True) else None
        
        self.logger.debug("✅ M1 optimizers initialized")
        
        # Initialize model factory
        self.model_factory = EnhancedModelFactory(self.config)
        
        # Tactician-specific configuration
        self.output_names = ["entry_timing", "position_size", "stop_loss", "take_profit"]
        self.n_outputs = len(self.output_names)
        
        # Initialize base models
        self.base_models = self._initialize_tactician_base_models()
        
        # Initialize meta models
        self.meta_models = self._initialize_tactician_meta_models()
        
        # Initialize ensemble manager
        self.ensemble_manager = self._initialize_ensemble_manager()
        
        # Initialize confidence calibrator
        self.confidence_calibrator = self._initialize_confidence_calibrator()
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ TacticianStackingTrainer initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Outputs: {self.n_outputs} ({self.output_names})")
        self.logger.info(f"🎯 Base models: {len(self.base_models)}")
        self.logger.info(f"🎯 Meta models: {len(self.meta_models)}")
    
    def _initialize_tactician_base_models(self) -> Dict[str, Any]:
        """Initialize Tactician base models."""
        
        self.logger.debug("🔧 Initializing Tactician base models...")
        
        # Tactician fixed models
        tactician_models = {
            "tabnet": ModelType.TABNET,
            "catboost": ModelType.CATBOOST,
            "lightgbm": ModelType.LIGHTGBM,
            "linear_ridge": ModelType.RIDGE
        }
        
        base_models = {}
        
        for output_name in self.output_names:
            base_models[output_name] = {}
            
            for model_name, model_type in tactician_models.items():
                config = ModelConfig(
                    model_type=model_type,
                    model_name=f"tactician_{model_name}_{output_name}",
                    is_multi_output=False,  # Base models are single-output
                    n_outputs=1,
                    output_names=[output_name],
                    enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', True),
                    enable_memory_optimization=self.config.get('enable_memory_optimization', True),
                    enable_parallel_processing=self.config.get('enable_parallel_processing', True),
                    memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
                )
                
                model = self.model_factory.create_model(config)
                base_models[output_name][model_name] = model
                self.logger.debug(f"✅ Created {model_name} for {output_name}")
        
        self.logger.info(f"✅ Initialized {sum(len(models) for models in base_models.values())} base models")
        return base_models
    
    def _initialize_tactician_meta_models(self) -> Dict[str, Any]:
        """Initialize Tactician meta models."""
        
        self.logger.debug("🔧 Initializing Tactician meta models...")
        
        meta_models = {}
        
        for output_name in self.output_names:
            # Use Ridge regression as meta model for each output
            config = ModelConfig(
                model_type=ModelType.RIDGE,
                model_name=f"tactician_meta_{output_name}",
                is_multi_output=False,
                n_outputs=1,
                output_names=[output_name],
                model_params={
                    'alpha': 1.0,
                    'random_state': 42
                }
            )
            
            model = self.model_factory.create_model(config)
            meta_models[output_name] = model
            self.logger.debug(f"✅ Created meta model for {output_name}")
        
        self.logger.info(f"✅ Initialized {len(meta_models)} meta models")
        return meta_models
    
    def _initialize_ensemble_manager(self) -> StackingEnsembleManager:
        """Initialize the ensemble manager."""
        
        self.logger.debug("🔧 Initializing ensemble manager...")
        
        config = StackingEnsembleConfig(
            ensemble_name="tactician_ensemble",
            output_dir=self.config.get('output_dir', './tactician_ensemble'),
            n_outputs=self.n_outputs,
            output_names=self.output_names,
            base_models=self.base_models,
            meta_models=self.meta_models,
            stacking_method=self.config.get('stacking_method', 'blending'),
            enable_meta_learning=self.config.get('enable_meta_learning', True),
            enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', True),
            enable_memory_optimization=self.config.get('enable_memory_optimization', True),
            enable_parallel_processing=self.config.get('enable_parallel_processing', True),
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        )
        
        manager = StackingEnsembleManager(config)
        self.logger.debug("✅ Ensemble manager initialized")
        return manager
    
    def _initialize_confidence_calibrator(self) -> StackingConfidenceCalibrator:
        """Initialize the confidence calibrator."""
        
        self.logger.debug("🔧 Initializing confidence calibrator...")
        
        config = StackingCalibrationConfig(
            calibrator_name="tactician_calibrator",
            n_outputs=self.n_outputs,
            output_names=self.output_names,
            enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', True),
            enable_memory_optimization=self.config.get('enable_memory_optimization', True),
            enable_parallel_processing=self.config.get('enable_parallel_processing', True),
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        )
        
        calibrator = StackingConfidenceCalibrator(config)
        self.logger.debug("✅ Confidence calibrator initialized")
        return calibrator
    
    @traced(span_name='train_tactician_ensemble')
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.DataFrame] = None) -> StackingTrainingResult:
        """Train the Tactician stacking ensemble."""
        
        self.logger.info("🚀 Training Tactician stacking ensemble...")
        start_time = time.time()
        
        self.logger.info(f"📊 Training data shape: {X_train.shape}")
        self.logger.info(f"📊 Target data shape: {y_train.shape}")
        if X_val is not None:
            self.logger.info(f"📊 Validation data shape: {X_val.shape}")
        if y_val is not None:
            self.logger.info(f"📊 Validation target shape: {y_val.shape}")
        
        try:
            # Train the ensemble
            self.logger.info("🔄 Training ensemble...")
            ensemble_result = self.ensemble_manager.train_ensemble(X_train, y_train, X_val, y_val)
            
            # Calibrate confidence if validation data is available
            if X_val is not None and y_val is not None:
                self.logger.info("🔄 Calibrating confidence...")
                y_pred = ensemble_result.predictions
                calibration_result = self.confidence_calibrator.calibrate_confidence(y_val.values, y_pred)
                
                # Update ensemble result with calibrated predictions
                ensemble_result.predictions = calibration_result.calibrated_predictions
                ensemble_result.confidence_scores = calibration_result.calibrated_predictions
            else:
                self.logger.warning("⚠️ No validation data provided, skipping confidence calibration")
            
            # Create training result
            result = StackingTrainingResult(
                trainer_name="tactician_stacking_trainer",
                model_type="tactician",
                n_outputs=self.n_outputs,
                output_names=self.output_names,
                created_at=datetime.now(),
                total_duration=0.0,  # Will be set by caller
                ensemble_performance=ensemble_result.ensemble_performance,
                per_output_performance=ensemble_result.per_output_performance,
                base_model_count=ensemble_result.base_model_count,
                meta_model_count=ensemble_result.meta_model_count,
                base_model_performance=ensemble_result.base_model_performance,
                meta_model_performance=ensemble_result.meta_model_performance,
                predictions=ensemble_result.predictions,
                prediction_probabilities=ensemble_result.prediction_probabilities,
                confidence_scores=ensemble_result.confidence_scores,
                model_weights=ensemble_result.model_weights,
                output_correlations=ensemble_result.output_correlations,
                feature_importance=ensemble_result.feature_importance,
                config=StackingEnsembleConfig(
                    trainer_name="tactician_stacking_trainer",
                    output_dir=self.config.get('output_dir', './tactician_ensemble'),
                    model_type="tactician",
                    n_outputs=self.n_outputs,
                    output_names=self.output_names
                ),
                optimization_used=ensemble_result.optimization_used
            )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Log memory usage
            if self.m1_memory:
                result.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
                self.logger.info(f"🧠 Memory usage: {result.memory_usage_mb:.1f} MB")
            
            self.logger.info(f"✅ Tactician stacking ensemble trained in {execution_time:.2f}s")
            self.logger.info(f"📊 Ensemble performance: {result.ensemble_performance}")
            self.logger.info(f"🎯 Base models: {result.base_model_count}")
            self.logger.info(f"🎯 Meta models: {result.meta_model_count}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Failed to train Tactician ensemble after {execution_time:.3f}s: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Make predictions using the trained ensemble."""
        
        if not self.ensemble_manager.stacking_model.is_fitted:
            raise ValueError("Ensemble not trained yet")
        
        return self.ensemble_manager.predict(X)
    
    def save_ensemble(self, file_path: str) -> None:
        """Save the trained ensemble."""
        
        try:
            # Save ensemble manager
            ensemble_path = file_path.replace('.pkl', '_ensemble.pkl')
            self.ensemble_manager.save_ensemble(ensemble_path)
            
            # Save confidence calibrator
            calibrator_path = file_path.replace('.pkl', '_calibrator.pkl')
            self.confidence_calibrator.save_calibrator(calibrator_path)
            
            self.logger.info(f"💾 Tactician ensemble saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save Tactician ensemble: {e}")
            raise
    
    def load_ensemble(self, file_path: str) -> None:
        """Load the trained ensemble."""
        
        try:
            # Load ensemble manager
            ensemble_path = file_path.replace('.pkl', '_ensemble.pkl')
            self.ensemble_manager.load_ensemble(ensemble_path)
            
            # Load confidence calibrator
            calibrator_path = file_path.replace('.pkl', '_calibrator.pkl')
            self.confidence_calibrator.load_calibrator(calibrator_path)
            
            self.logger.info(f"📂 Tactician ensemble loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Tactician ensemble: {e}")
            raise


# Convenience functions for creating trainers
def create_analyst_trainer(config: Optional[Dict[str, Any]] = None) -> AnalystStackingTrainer:
    """Create an Analyst stacking trainer."""
    return AnalystStackingTrainer(config)


def create_tactician_trainer(config: Optional[Dict[str, Any]] = None) -> TacticianStackingTrainer:
    """Create a Tactician stacking trainer."""
    return TacticianStackingTrainer(config)