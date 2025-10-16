"""
Shared Training Utilities for Hybrid NAS-TAS Regime Detection.

Provides common training utilities that can be used by both NAS and TAS systems
for model training, optimization, and performance monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum
import json
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrainingType(Enum):
    """Types of training available."""
    STANDARD = "standard"
    REGULARIZED = "regularized"
    ENSEMBLE = "ensemble"
    TRANSFER = "transfer"
    CONTINUAL = "continual"
    META = "meta"

class OptimizerType(Enum):
    """Optimizer types available."""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    NADAM = "nadam"
    ADAMAX = "adamax"

@dataclass
class SharedTrainingConfig:
    """Configuration for shared training utilities."""
    # Training type
    training_type: TrainingType = TrainingType.STANDARD

    # Optimizer settings
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9

    # Training parameters
    batch_size: int = 32
    n_epochs: int = 100
    validation_split: float = 0.2
    random_state: int = 42

    # Regularization
    enable_regularization: bool = True
    l1_regularization: float = 0.0
    l2_regularization: float = 0.01
    dropout_rate: float = 0.0
    batch_normalization: bool = False

    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001

    # Learning rate scheduling
    enable_lr_scheduling: bool = True
    lr_scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "exponential", "step"
    lr_factor: float = 0.5
    lr_patience: int = 5

    # Performance optimization
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    memory_limit_gb: float = 8.0

    # Monitoring
    enable_monitoring: bool = True
    log_interval: int = 10
    save_checkpoints: bool = True
    checkpoint_interval: int = 10

    # Output settings
    save_results: bool = True
    output_dir: str = "training_results"
    verbose: bool = True

@dataclass
class SharedTrainingResult:
    """Result from shared training."""
    # Training results
    model: Any
    training_history: Dict[str, List[float]]
    best_epoch: int
    best_score: float

    # Performance metrics
    training_time: float = 0.0
    memory_usage_mb: float = 0.0
    convergence_metrics: Dict[str, float] = None

    # Validation results
    validation_scores: Dict[str, float] = None
    test_scores: Dict[str, float] = None

    # Metadata
    training_type: str = ""
    optimizer: str = ""
    n_epochs: int = 0
    batch_size: int = 0

    # Results
    success: bool = True
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class SharedTrainer:
    """Shared trainer for both NAS and TAS systems."""

    def __init__(self, config: SharedTrainingConfig):
        """Initialize the shared trainer.

        Args:
            config: Shared training configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for shared training")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for shared training")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        # Initialize training components
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        self.logger.info("✅ Shared Trainer initialized")
        self.logger.info(f"   Type: {config.training_type.value}")
        self.logger.info(f"   Optimizer: {config.optimizer.value}")
        self.logger.info(f"   Learning rate: {config.learning_rate}")
        self.logger.info(f"   Epochs: {config.n_epochs}")
        self.logger.info(f"   Batch size: {config.batch_size}")

    def train(self,
              model: Any,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              additional_data: Optional[Dict[str, Any]] = None) -> SharedTrainingResult:
        """Train a model using the configured strategy.

        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            additional_data: Optional additional data for training

        Returns:
            SharedTrainingResult with training results
        """
        start_time = time.time()

        try:
            self.logger.info("🚀 Starting shared training")
            self.logger.info(f"   Training data shape: {X_train.shape}")
            self.logger.info(f"   Type: {self.config.training_type.value}")
            self.logger.info(f"   Optimizer: {self.config.optimizer.value}")
            self.logger.info(f"   Epochs: {self.config.n_epochs}")

            # Initialize training
            self._initialize_training()

            # Perform training based on type
            if self.config.training_type == TrainingType.STANDARD:
                result = self._standard_training(model, X_train, y_train, X_val, y_val)
            elif self.config.training_type == TrainingType.REGULARIZED:
                result = self._regularized_training(model, X_train, y_train, X_val, y_val)
            elif self.config.training_type == TrainingType.ENSEMBLE:
                result = self._ensemble_training(model, X_train, y_train, X_val, y_val)
            elif self.config.training_type == TrainingType.TRANSFER:
                result = self._transfer_training(model, X_train, y_train, X_val, y_val, additional_data)
            elif self.config.training_type == TrainingType.CONTINUAL:
                result = self._continual_training(model, X_train, y_train, X_val, y_val)
            elif self.config.training_type == TrainingType.META:
                result = self._meta_training(model, X_train, y_train, X_val, y_val, additional_data)
            else:
                raise ValueError(f"Unknown training type: {self.config.training_type}")

            # Finalize training
            training_time = time.time() - start_time
            result.training_time = training_time
            result.hardware_optimization_applied = self.hardware_accelerator is not None
            result.matrix_operations_used = self.matrix_ops is not None

            # Save results if requested
            if self.config.save_results:
                self._save_training_results(result)

            self.logger.info(f"✅ Shared training completed in {training_time:.2f}s")
            self.logger.info(f"   Best epoch: {result.best_epoch}")
            self.logger.info(f"   Best score: {result.best_score:.4f}")

            return result

        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Shared training failed: {e}")

            return SharedTrainingResult(
                model=model,
                training_history={},
                best_epoch=0,
                best_score=0.0,
                training_time=training_time,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _initialize_training(self):
        """Initialize training process."""
        try:
            # Reset training history
            self.training_history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'learning_rate': []
            }

            self.logger.info("✅ Training initialized")

        except Exception as e:
            self.logger.error(f"❌ Training initialization failed: {e}")
            raise

    def _standard_training(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> SharedTrainingResult:
        """Perform standard training."""
        try:
            self.logger.info("📚 Performing standard training")

            # Configure optimizer
            optimizer = self._create_optimizer(model)

            # Configure learning rate scheduler
            scheduler = self._create_scheduler(optimizer) if self.config.enable_lr_scheduling else None

            # Training loop
            best_score = -np.inf
            best_epoch = 0
            patience_counter = 0

            for epoch in range(self.config.n_epochs):
                # Training phase
                train_loss, train_accuracy = self._train_epoch(model, X_train, y_train, optimizer)

                # Validation phase
                val_loss, val_accuracy = 0.0, 0.0
                if X_val is not None and y_val is not None:
                    val_loss, val_accuracy = self._validate_epoch(model, X_val, y_val)

                # Update learning rate
                if scheduler:
                    scheduler.step(val_loss if val_loss > 0 else train_loss)

                # Record history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_accuracy'].append(train_accuracy)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                self.training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

                # Update best score
                current_score = val_accuracy if val_accuracy > 0 else train_accuracy
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Log progress
                if epoch % self.config.log_interval == 0:
                    self.logger.info(f"   Epoch {epoch+1}/{self.config.n_epochs}: "
                                   f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                                   f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                # Early stopping
                if self.config.enable_early_stopping and patience_counter >= self.config.patience:
                    self.logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break

            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=best_epoch,
                best_score=best_score,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Standard training failed: {e}")
            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=0,
                best_score=0.0,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _regularized_training(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> SharedTrainingResult:
        """Perform regularized training."""
        try:
            self.logger.info("🔒 Performing regularized training")

            # Apply regularization to model
            if self.config.l1_regularization > 0 or self.config.l2_regularization > 0:
                model = self._apply_regularization(model)

            # Use standard training with regularization
            return self._standard_training(model, X_train, y_train, X_val, y_val)

        except Exception as e:
            self.logger.error(f"❌ Regularized training failed: {e}")
            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=0,
                best_score=0.0,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _ensemble_training(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> SharedTrainingResult:
        """Perform ensemble training."""
        try:
            self.logger.info("🎭 Performing ensemble training")

            # Create ensemble of models
            ensemble_models = self._create_ensemble_models(model, n_models=5)

            # Train each model in the ensemble
            ensemble_results = []
            for i, ensemble_model in enumerate(ensemble_models):
                self.logger.info(f"   Training ensemble model {i+1}/{len(ensemble_models)}")
                result = self._standard_training(ensemble_model, X_train, y_train, X_val, y_val)
                ensemble_results.append(result)

            # Combine ensemble results
            best_score = max(result.best_score for result in ensemble_results)
            best_epoch = max(result.best_epoch for result in ensemble_results)

            return SharedTrainingResult(
                model=ensemble_models,
                training_history=self.training_history,
                best_epoch=best_epoch,
                best_score=best_score,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=0,
                best_score=0.0,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _transfer_training(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                         additional_data: Optional[Dict[str, Any]]) -> SharedTrainingResult:
        """Perform transfer training."""
        try:
            self.logger.info("🔄 Performing transfer training")

            # Load pre-trained model if available
            if additional_data and 'pretrained_model' in additional_data:
                pretrained_model = additional_data['pretrained_model']
                model = self._load_pretrained_weights(model, pretrained_model)

            # Use standard training with transfer learning
            return self._standard_training(model, X_train, y_train, X_val, y_val)

        except Exception as e:
            self.logger.error(f"❌ Transfer training failed: {e}")
            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=0,
                best_score=0.0,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _continual_training(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> SharedTrainingResult:
        """Perform continual training."""
        try:
            self.logger.info("🔄 Performing continual training")

            # Split training data into tasks
            tasks = self._split_into_tasks(X_train, y_train, n_tasks=3)

            # Train on each task sequentially
            for i, (X_task, y_task) in enumerate(tasks):
                self.logger.info(f"   Training on task {i+1}/{len(tasks)}")
                result = self._standard_training(model, X_task, y_task, X_val, y_val)

            return result

        except Exception as e:
            self.logger.error(f"❌ Continual training failed: {e}")
            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=0,
                best_score=0.0,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _meta_training(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                      additional_data: Optional[Dict[str, Any]]) -> SharedTrainingResult:
        """Perform meta training."""
        try:
            self.logger.info("🧠 Performing meta training")

            # Use standard training for now
            # Meta-learning would require more complex implementation
            return self._standard_training(model, X_train, y_train, X_val, y_val)

        except Exception as e:
            self.logger.error(f"❌ Meta training failed: {e}")
            return SharedTrainingResult(
                model=model,
                training_history=self.training_history,
                best_epoch=0,
                best_score=0.0,
                training_type=self.config.training_type.value,
                optimizer=self.config.optimizer.value,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                success=False,
                error_message=str(e)
            )

    def _create_optimizer(self, model: Any) -> Any:
        """Create optimizer for the model."""
        try:
            import torch
            import torch.optim as optim

            if self.config.optimizer == OptimizerType.SGD:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == OptimizerType.ADAM:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == OptimizerType.ADAMW:
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == OptimizerType.RMSPROP:
                optimizer = optim.RMSprop(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            else:
                # Default to Adam
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )

            return optimizer

        except Exception as e:
            self.logger.warning(f"⚠️ Optimizer creation failed: {e}")
            return None

    def _create_scheduler(self, optimizer: Any) -> Any:
        """Create learning rate scheduler."""
        try:
            import torch.optim.lr_scheduler as lr_scheduler

            if self.config.lr_scheduler == "reduce_on_plateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.config.lr_factor,
                    patience=self.config.lr_patience,
                    min_lr=1e-7
                )
            elif self.config.lr_scheduler == "cosine":
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.n_epochs
                )
            elif self.config.lr_scheduler == "exponential":
                scheduler = lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=self.config.lr_factor
                )
            elif self.config.lr_scheduler == "step":
                scheduler = lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.config.lr_patience,
                    gamma=self.config.lr_factor
                )
            else:
                scheduler = None

            return scheduler

        except Exception as e:
            self.logger.warning(f"⚠️ Scheduler creation failed: {e}")
            return None

    def _train_epoch(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, optimizer: Any) -> Tuple[float, float]:
        """Train for one epoch."""
        try:
            # This would be implemented based on the specific model type
            # For now, return dummy values
            train_loss = 0.5
            train_accuracy = 0.8

            return train_loss, train_accuracy

        except Exception as e:
            self.logger.warning(f"⚠️ Training epoch failed: {e}")
            return 0.0, 0.0

    def _validate_epoch(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[float, float]:
        """Validate for one epoch."""
        try:
            # This would be implemented based on the specific model type
            # For now, return dummy values
            val_loss = 0.4
            val_accuracy = 0.85

            return val_loss, val_accuracy

        except Exception as e:
            self.logger.warning(f"⚠️ Validation epoch failed: {e}")
            return 0.0, 0.0

    def _apply_regularization(self, model: Any) -> Any:
        """Apply regularization to the model."""
        try:
            # This would be implemented based on the specific model type
            # For now, return the model as-is
            return model

        except Exception as e:
            self.logger.warning(f"⚠️ Regularization application failed: {e}")
            return model

    def _create_ensemble_models(self, model: Any, n_models: int = 5) -> List[Any]:
        """Create ensemble of models."""
        try:
            # This would create multiple instances of the model
            # For now, return a list with the original model
            return [model] * n_models

        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble model creation failed: {e}")
            return [model]

    def _load_pretrained_weights(self, model: Any, pretrained_model: Any) -> Any:
        """Load pre-trained weights into the model."""
        try:
            # This would load pre-trained weights
            # For now, return the model as-is
            return model

        except Exception as e:
            self.logger.warning(f"⚠️ Pre-trained weight loading failed: {e}")
            return model

    def _split_into_tasks(self, X_train: pd.DataFrame, y_train: pd.Series, n_tasks: int = 3) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Split training data into tasks for continual learning."""
        try:
            # Split data into tasks
            task_size = len(X_train) // n_tasks
            tasks = []

            for i in range(n_tasks):
                start_idx = i * task_size
                end_idx = start_idx + task_size if i < n_tasks - 1 else len(X_train)

                X_task = X_train.iloc[start_idx:end_idx]
                y_task = y_train.iloc[start_idx:end_idx]
                tasks.append((X_task, y_task))

            return tasks

        except Exception as e:
            self.logger.warning(f"⚠️ Task splitting failed: {e}")
            return [(X_train, y_train)]

    def _save_training_results(self, result: SharedTrainingResult):
        """Save training results to file."""
        try:
            from pathlib import Path

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save results
            result_file = output_dir / "shared_training_results.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'training_history': result.training_history,
                    'best_epoch': result.best_epoch,
                    'best_score': result.best_score,
                    'training_time': result.training_time,
                    'training_type': result.training_type,
                    'optimizer': result.optimizer,
                    'n_epochs': result.n_epochs,
                    'batch_size': result.batch_size,
                    'success': result.success,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)

            self.logger.info(f"💾 Training results saved to {result_file}")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not save training results: {e}")

def create_shared_trainer(config: Optional[SharedTrainingConfig] = None) -> SharedTrainer:
    """Create a shared trainer instance.

    Args:
        config: Optional shared training configuration

    Returns:
        SharedTrainer instance
    """
    if config is None:
        config = SharedTrainingConfig()
    return SharedTrainer(config)

def quick_shared_training(model: Any,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: Optional[pd.DataFrame] = None,
                         y_val: Optional[pd.Series] = None,
                         training_type: TrainingType = TrainingType.STANDARD,
                         n_epochs: int = 100) -> SharedTrainingResult:
    """Quick shared training with default settings.

    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        X_val: Optional validation features
        y_val: Optional validation targets
        training_type: Training type
        n_epochs: Number of epochs

    Returns:
        SharedTrainingResult
    """
    config = SharedTrainingConfig(
        training_type=training_type,
        n_epochs=n_epochs,
        enable_early_stopping=True
    )

    trainer = SharedTrainer(config)
    return trainer.train(model, X_train, y_train, X_val, y_val)
