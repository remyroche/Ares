"""
Enhanced Early Stopping System for ML Common

Comprehensive early stopping implementation that integrates with existing modules
and supports all model types including neural networks, tree-based models, and others.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.base import clone
import warnings

logger = logging.getLogger(__name__)

@dataclass
class EarlyStoppingConfig:
    """Comprehensive early stopping configuration."""
    # Basic settings
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    mode: str = 'min'  # 'min' for loss, 'max' for accuracy

    # Model-specific settings
    monitor: str = 'validation_loss'
    restore_best_weights: bool = True

    # Neural network settings
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100
    nn_validation_split: float = 0.2

    # Tree-based model settings
    tree_eval_metric: str = 'auto'
    tree_early_stopping_rounds: int = 50

    # Generic model settings
    generic_check_frequency: int = 1
    generic_max_iterations: int = 100

    # Advanced settings
    start_from_epoch: int = 0
    verbose: bool = True
    save_best_model: bool = True
    checkpoint_dir: str = "checkpoints"

@dataclass
class EarlyStoppingResult:
    """Result from early stopping training."""
    # Basic information
    model_type: str = "unknown"
    training_stopped: bool = False
    best_epoch: int = 0
    best_score: float = 0.0

    # Training history
    history: Dict[str, List[float]] = None
    stopped_epoch: int = 0
    total_epochs: int = 0

    # Model information
    best_model: Any = None
    final_model: Any = None

    # Performance metrics
    training_scores: List[float] = None
    validation_scores: List[float] = None
    test_score: float = 0.0

    # Early stopping details
    early_stopping_applied: bool = False
    reason: str = ""
    patience_counter: int = 0

    # Metadata
    config_used: Dict[str, Any] = None
    training_time: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.history is None:
            self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        if self.training_scores is None:
            self.training_scores = []
        if self.validation_scores is None:
            self.validation_scores = []
        if self.config_used is None:
            self.config_used = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class EnhancedEarlyStopping:
    """Enhanced early stopping system supporting all model types."""

    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        """
        Initialize enhanced early stopping system.

        Args:
            config: Early stopping configuration
        """
        self.config = config or EarlyStoppingConfig()

        # Track training state
        self.best_score = float('inf') if self.config.mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

        # Create checkpoint directory
        if self.config.save_best_model:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Enhanced Early Stopping initialized")

    def apply_early_stopping(self,
                           model: Any,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           model_type: str,
                           **kwargs) -> Tuple[Any, EarlyStoppingResult]:
        """
        Apply early stopping based on model type.

        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (trained_model, early_stopping_result)
        """
        model_type_lower = model_type.lower()

        # Route to appropriate early stopping method
        if model_type_lower in ['xgboost', 'xgb', 'lightgbm', 'lgbm', 'catboost']:
            return self._apply_tree_based_early_stopping(model, X_train, y_train, X_val, y_val, model_type, **kwargs)
        elif model_type_lower in ['neural_network', 'neural_net', 'nn', 'torch', 'pytorch', 'tensorflow', 'keras']:
            return self._apply_neural_network_early_stopping(model, X_train, y_train, X_val, y_val, model_type, **kwargs)
        else:
            return self._apply_generic_early_stopping(model, X_train, y_train, X_val, y_val, model_type, **kwargs)

    def _apply_tree_based_early_stopping(self,
                                       model: Any,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_val: np.ndarray,
                                       model_type: str,
                                       **kwargs) -> Tuple[Any, EarlyStoppingResult]:
        """Apply early stopping for tree-based models."""
        result = EarlyStoppingResult(model_type=model_type)

        try:
            # Configure early stopping parameters
            eval_metric = self.config.tree_eval_metric
            if eval_metric == 'auto':
                eval_metric = 'logloss' if len(np.unique(y_train)) <= 10 else 'rmse'

            # Set up model parameters
            params = model.get_params()

            # XGBoost configuration
            if 'xgb' in model_type.lower():
                params.update({
                    'eval_set': [(X_val, y_val)],
                    'early_stopping_rounds': self.config.tree_early_stopping_rounds,
                    'eval_metric': eval_metric,
                    'verbose': False
                })

            # LightGBM configuration
            elif 'lgbm' in model_type.lower():
                params.update({
                    'eval_set': [(X_val, y_val)],
                    'early_stopping_rounds': self.config.tree_early_stopping_rounds,
                    'eval_metric': eval_metric,
                    'callbacks': ['early_stopping'],
                    'verbose': -1
                })

            # CatBoost configuration
            elif 'catboost' in model_type.lower():
                params.update({
                    'eval_set': (X_val, y_val),
                    'early_stopping_rounds': self.config.tree_early_stopping_rounds,
                    'verbose': False,
                    'use_best_model': True
                })

            # Update model with new parameters
            model.set_params(**params)

            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            result.training_time = (datetime.now() - start_time).total_seconds()

            # Extract training history if available
            if hasattr(model, 'evals_result_'):
                evals = model.evals_result_
                if evals:
                    result.history['val_loss'] = evals['validation_0'][eval_metric]

            # Get best iteration if available
            if hasattr(model, 'best_iteration'):
                result.best_epoch = model.best_iteration
                result.early_stopping_applied = model.best_iteration < model.get_params().get('n_estimators', 100)

            # Make predictions for scoring
            y_pred = model.predict(X_val)

            # Calculate final scores
            if len(np.unique(y_train)) <= 10:  # Classification
                result.best_score = accuracy_score(y_val, y_pred)
                result.test_score = result.best_score
            else:  # Regression
                result.best_score = r2_score(y_val, y_pred)
                result.test_score = result.best_score

            result.training_stopped = result.early_stopping_applied
            result.final_model = model
            result.best_model = model

            if result.early_stopping_applied:
                result.reason = f"Early stopping at iteration {result.best_epoch}"

            logger.info(f"✅ Tree-based early stopping completed for {model_type}")
            return model, result

        except Exception as e:
            logger.error(f"Tree-based early stopping failed: {e}")
            # Fallback to standard training
            model.fit(X_train, y_train)
            result.reason = f"Fallback training after early stopping failure: {e}"
            result.final_model = model
            result.best_model = model
            return model, result

    def _apply_neural_network_early_stopping(self,
                                          model: Any,
                                          X_train: np.ndarray,
                                          y_train: np.ndarray,
                                          X_val: np.ndarray,
                                          y_val: np.ndarray,
                                          model_type: str,
                                          **kwargs) -> Tuple[Any, EarlyStoppingResult]:
        """Apply early stopping for neural network models."""
        result = EarlyStoppingResult(model_type=model_type)

        try:
            # Check if model is a PyTorch model
            if isinstance(model, nn.Module):
                return self._apply_pytorch_early_stopping(model, X_train, y_train, X_val, y_val, model_type, **kwargs)
            else:
                # For other neural network frameworks, use generic approach
                return self._apply_generic_early_stopping(model, X_train, y_train, X_val, y_val, model_type, **kwargs)

        except Exception as e:
            logger.error(f"Neural network early stopping failed: {e}")
            # Fallback to generic approach
            return self._apply_generic_early_stopping(model, X_train, y_train, X_val, y_val, model_type, **kwargs)

    def _apply_pytorch_early_stopping(self,
                                    model: nn.Module,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray,
                                    model_type: str,
                                    **kwargs) -> Tuple[Any, EarlyStoppingResult]:
        """Apply early stopping for PyTorch neural networks."""
        result = EarlyStoppingResult(model_type=model_type)

        try:
            # Convert numpy arrays to torch tensors
            X_train_tensor = torch.from_numpy(X_train).float()
            y_train_tensor = torch.from_numpy(y_train).float()
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).float()

            # Determine if it's classification or regression
            is_classification = len(np.unique(y_train)) <= 10

            if is_classification:
                # For classification, convert to long tensor
                y_train_tensor = torch.from_numpy(y_train).long()
                y_val_tensor = torch.from_numpy(y_val).long()

                # Determine number of classes
                n_classes = len(np.unique(y_train))
                if n_classes == 2:
                    criterion = nn.CrossEntropyLoss()
                    output_activation = None
                else:
                    criterion = nn.CrossEntropyLoss()
                    output_activation = None
            else:
                criterion = nn.MSELoss()
                output_activation = None

            # Setup optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.nn_learning_rate)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.config.nn_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.nn_batch_size)

            # Training loop with early stopping
            best_val_loss = float('inf') if self.config.mode == 'min' else float('-inf')
            best_model_state = None

            for epoch in range(self.config.nn_epochs):
                # Training phase
                model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_predictions = []
                val_targets = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        val_predictions.extend(outputs.detach().numpy())
                        val_targets.extend(batch_y.detach().numpy())

                val_loss /= len(val_loader)
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)

                # Calculate validation metric
                if is_classification:
                    val_acc = accuracy_score(val_targets, np.argmax(val_predictions, axis=1) if val_predictions.ndim > 1 else np.round(val_predictions))
                    current_score = val_acc if self.config.mode == 'max' else -val_loss
                else:
                    val_r2 = r2_score(val_targets, val_predictions)
                    current_score = val_r2 if self.config.mode == 'max' else -val_loss

                # Update history
                result.history['train_loss'].append(train_loss)
                result.history['val_loss'].append(val_loss)

                # Check early stopping condition
                if self.config.mode == 'min':
                    improved = current_score < best_val_loss - self.config.min_delta
                else:
                    improved = current_score > best_val_loss + self.config.min_delta

                if improved:
                    best_val_loss = current_score
                    best_model_state = model.state_dict().copy()
                    result.best_epoch = epoch
                    result.counter = 0
                else:
                    result.counter += 1

                # Early stopping check
                if result.counter >= self.config.patience:
                    result.training_stopped = True
                    result.stopped_epoch = epoch
                    result.reason = f"Early stopping at epoch {epoch} (no improvement for {self.config.patience} epochs)"
                    break

            result.total_epochs = len(result.history['train_loss'])
            result.best_score = best_val_loss

            # Restore best weights
            if self.config.restore_best_weights and best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Calculate final test score
            model.eval()
            with torch.no_grad():
                if is_classification:
                    test_outputs = model(torch.from_numpy(X_val).float())
                    test_preds = np.argmax(test_outputs.detach().numpy(), axis=1) if test_outputs.ndim > 1 else np.round(test_outputs.detach().numpy())
                    result.test_score = accuracy_score(y_val, test_preds)
                else:
                    test_outputs = model(torch.from_numpy(X_val).float())
                    result.test_score = r2_score(y_val, test_outputs.detach().numpy())

            result.early_stopping_applied = result.training_stopped
            result.final_model = model
            result.best_model = model if best_model_state is None else model

            logger.info(f"✅ PyTorch early stopping completed for {model_type}")
            return model, result

        except Exception as e:
            logger.error(f"PyTorch early stopping failed: {e}")
            # Fallback to basic training
            model.fit(X_train, y_train)
            result.reason = f"Fallback training after PyTorch early stopping failure: {e}"
            result.final_model = model
            result.best_model = model
            return model, result

    def _apply_generic_early_stopping(self,
                                    model: Any,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray,
                                    model_type: str,
                                    **kwargs) -> Tuple[Any, EarlyStoppingResult]:
        """Apply generic early stopping for other models."""
        result = EarlyStoppingResult(model_type=model_type)

        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y_train)) <= 10

            # Simple iterative early stopping
            best_val_score = float('inf') if self.config.mode == 'min' else float('-inf')
            best_model_state = None

            for epoch in range(self.config.generic_max_iterations):
                # Train model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)

                # Evaluate on validation set
                y_pred = model_clone.predict(X_val)

                # Calculate score
                if is_classification:
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = r2_score(y_val, y_pred)

                # Update history
                result.history['val_loss'].append(1 - score if is_classification else -score)

                # Check improvement
                if self.config.mode == 'min':
                    improved = score < best_val_score - self.config.min_delta
                else:
                    improved = score > best_val_score + self.config.min_delta

                if improved:
                    best_val_score = score
                    best_model_state = model_clone
                    result.best_epoch = epoch
                    result.counter = 0
                else:
                    result.counter += 1

                # Early stopping check
                if result.counter >= self.config.patience:
                    result.training_stopped = True
                    result.stopped_epoch = epoch
                    result.reason = f"Early stopping at iteration {epoch} (no improvement for {self.config.patience} iterations)"
                    break

            result.total_epochs = epoch + 1
            result.best_score = best_val_score
            result.test_score = best_val_score
            result.early_stopping_applied = result.training_stopped

            if best_model_state is not None:
                result.best_model = best_model_state
                result.final_model = best_model_state
            else:
                result.final_model = model
                result.best_model = model

            logger.info(f"✅ Generic early stopping completed for {model_type}")
            return result.final_model, result

        except Exception as e:
            logger.error(f"Generic early stopping failed: {e}")
            # Fallback to standard training
            model.fit(X_train, y_train)
            result.reason = f"Fallback training after generic early stopping failure: {e}"
            result.final_model = model
            result.best_model = model
            return model, result

# Convenience functions
def apply_enhanced_early_stopping(model: Any,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 model_type: str,
                                 config: Optional[EarlyStoppingConfig] = None) -> Tuple[Any, EarlyStoppingResult]:
    """Convenience function to apply enhanced early stopping."""
    early_stopper = EnhancedEarlyStopping(config)
    return early_stopper.apply_early_stopping(model, X_train, y_train, X_val, y_val, model_type)

def get_early_stopping_config(**kwargs) -> EarlyStoppingConfig:
    """Get early stopping configuration with defaults."""
    return EarlyStoppingConfig(**kwargs)
