"""
Neural Architecture Search Trainer

This module provides training functionality for NAS models,
including training loops, loss functions, and optimization strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path

from ..search.search_space import ArchitectureConfig
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000

    # Loss function
    loss_function: str = "cross_entropy"  # "cross_entropy", "mse", "hmm_loss"

    # Optimizer
    optimizer: str = "adam"  # "adam", "sgd", "adamw"

    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"

    # Hardware
    use_gpu: bool = True
    mixed_precision: bool = True
    num_workers: int = 4

@dataclass
class TrainingResult:
    """Result of model training."""
    model: nn.Module
    final_loss: float
    best_loss: float
    final_accuracy: float
    best_accuracy: float
    training_history: Dict[str, List[float]]
    execution_time: float
    epochs_trained: int
    converged: bool

class NASTrainer:
    """
    Neural Architecture Search Trainer

    Handles training of NAS models with different architectures,
    loss functions, and optimization strategies.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize NAS trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Setup device
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🔧 Training device: {self.device}")

        # Initialize components
        self.loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'mse': nn.MSELoss(),
            'bce': nn.BCEWithLogitsLoss(),
            'hmm_loss': self._hmm_loss,
            'regime_loss': self._regime_loss
        }

        self.optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD
        }

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.no_improvement_count = 0
        self.converged = False

    def train(self,
              model: nn.Module,
              train_dataset: Dataset,
              val_dataset: Optional[Dataset] = None,
              problem_type: str = "classification") -> TrainingResult:
        """
        Train a NAS model.

        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            problem_type: Type of problem

        Returns:
            TrainingResult with trained model and metrics
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting training for {model.__class__.__name__}")

        # Move model to device
        model = model.to(self.device)

        # Setup training components
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )

        # Setup loss function and optimizer
        criterion = self._get_loss_function(problem_type)
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)

        # Mixed precision setup
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == 'cuda' else None

        # Training history
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Training loop
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch

                # Train epoch
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, criterion, optimizer, scaler
                )

                # Validate epoch
                val_loss, val_acc = self._validate_epoch(
                    model, val_loader, criterion
                )

                # Update learning rate
                if scheduler:
                    if self.config.scheduler == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # Track metrics
                current_lr = optimizer.param_groups[0]['lr']
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['train_accuracy'].append(train_acc)
                training_history['val_accuracy'].append(val_acc)
                training_history['learning_rate'].append(current_lr)

                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(
                        f"📈 Epoch {epoch+1}/{self.config.epochs} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                        f"LR: {current_lr:.6f}"
                    )

                # Check early stopping
                if self._check_early_stopping(val_loss, val_acc):
                    self.logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break

                # Check convergence
                if self._check_convergence():
                    self.logger.info(f"✅ Training converged at epoch {epoch+1}")
                    break

        except KeyboardInterrupt:
            self.logger.info("🛑 Training interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")

        execution_time = time.time() - start_time
        epochs_trained = self.current_epoch + 1

        # Create training result
        result = TrainingResult(
            model=model,
            final_loss=training_history['val_loss'][-1] if training_history['val_loss'] else 0.0,
            best_loss=self.best_loss,
            final_accuracy=training_history['val_accuracy'][-1] if training_history['val_accuracy'] else 0.0,
            best_accuracy=self.best_accuracy,
            training_history=training_history,
            execution_time=execution_time,
            epochs_trained=epochs_trained,
            converged=self.converged
        )

        self.logger.info(f"✅ Training completed in {execution_time:.2f}s")
        self.logger.info(f"🏆 Final validation loss: {result.final_loss:.4f}")
        self.logger.info(f"🎯 Final validation accuracy: {result.final_accuracy:.4f}")

        return result

    def _train_epoch(self,
                     model: nn.Module,
                     train_loader: DataLoader,
                     criterion: Callable,
                     optimizer: optim.Optimizer,
                     scaler: Optional[torch.cuda.amp.GradScaler]) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scaler: Gradient scaler for mixed precision

        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            if scaler and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = self._calculate_loss(output, target, criterion, model)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = self._calculate_loss(output, target, criterion, model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            if isinstance(criterion, nn.CrossEntropyLoss):
                pred = output.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
                total_predictions += target.size(0)
            elif isinstance(criterion, nn.MSELoss):
                # For regression, calculate R² or similar
                pred = output.squeeze()
                correct_predictions += torch.mean((pred - target) ** 2).item()
                total_predictions += 1

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return avg_loss, accuracy

    def _validate_epoch(self,
                       model: nn.Module,
                       val_loader: Optional[DataLoader],
                       criterion: Callable) -> Tuple[float, float]:
        """Validate for one epoch.

        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        if val_loader is None:
            return 0.0, 0.0

        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                loss = self._calculate_loss(output, target, criterion, model)
                total_loss += loss.item()

                # Calculate accuracy
                if isinstance(criterion, nn.CrossEntropyLoss):
                    pred = output.argmax(dim=1)
                    correct_predictions += pred.eq(target).sum().item()
                    total_predictions += target.size(0)
                elif isinstance(criterion, nn.MSELoss):
                    pred = output.squeeze()
                    correct_predictions += torch.mean((pred - target) ** 2).item()
                    total_predictions += 1

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Track best performance
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return avg_loss, accuracy

    def _calculate_loss(self, output: torch.Tensor, target: torch.Tensor,
                       criterion: Callable, model: nn.Module) -> torch.Tensor:
        """Calculate loss for given output and target.

        Args:
            output: Model output
            target: Ground truth
            criterion: Loss function
            model: Model instance

        Returns:
            Loss tensor
        """
        if isinstance(criterion, nn.CrossEntropyLoss):
            return criterion(output, target)
        elif isinstance(criterion, nn.MSELoss):
            return criterion(output.squeeze(), target.float())
        elif callable(criterion):
            return criterion(output, target)
        else:
            return criterion(output, target)

    def _hmm_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """HMM-specific loss function.

        Args:
            output: Model output (state probabilities and transitions)
            target: Ground truth states

        Returns:
            HMM loss
        """
        if isinstance(output, tuple):
            state_probs, transition_probs = output
            # HMM loss combines state prediction and transition smoothness
            state_loss = F.nll_loss(state_probs, target)
            return state_loss
        else:
            return F.nll_loss(output, target)

    def _regime_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Regime detection loss function.

        Args:
            output: Model output (regime probabilities)
            target: Ground truth regimes

        Returns:
            Regime detection loss
        """
        return F.nll_loss(output, target)

    def _get_loss_function(self, problem_type: str) -> Callable:
        """Get appropriate loss function for problem type.

        Args:
            problem_type: Type of problem

        Returns:
            Loss function
        """
        loss_map = {
            'classification': self.loss_functions['cross_entropy'],
            'regression': self.loss_functions['mse'],
            'hmm': self.loss_functions['hmm_loss'],
            'regime_detection': self.loss_functions['regime_loss']
        }
        return loss_map.get(problem_type, self.loss_functions['cross_entropy'])

    def _get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get optimizer for model.

        Args:
            model: Model to optimize

        Returns:
            Optimizer
        """
        optimizer_class = self.optimizers.get(self.config.optimizer, optim.Adam)

        if self.config.optimizer in ['adam', 'adamw']:
            return optimizer_class(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optimizer_class(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            return optimizer_class(model.parameters(), lr=self.config.learning_rate)

    def _get_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule

        Returns:
            Learning rate scheduler or None
        """
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        else:
            return None

    def _check_early_stopping(self, val_loss: float, val_acc: float) -> bool:
        """Check if training should stop early.

        Args:
            val_loss: Validation loss
            val_acc: Validation accuracy

        Returns:
            True if should stop
        """
        if val_loss < self.config.early_stopping_threshold:
            return True

        if self.no_improvement_count >= self.config.early_stopping_patience:
            return True

        return False

    def _check_convergence(self) -> bool:
        """Check if training has converged.

        Returns:
            True if converged
        """
        # Simple convergence check based on loss stabilization
        if len(self.converged) > 10:
            recent_losses = self.converged[-10:]
            std_loss = np.std(recent_losses)
            if std_loss < 0.001:  # Very small variation
                return True
        return False

    def save_model(self, model: nn.Module, path: str):
        """Save trained model.

        Args:
            model: Model to save
            path: Save path
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config,
                'training_result': None  # Could add training result here
            }, path)
            self.logger.info(f"💾 Model saved to {path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")

    def load_model(self, path: str, model_class: type) -> nn.Module:
        """Load trained model.

        Args:
            path: Model path
            model_class: Class to instantiate model

        Returns:
            Loaded model
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            self.logger.info(f"📁 Model loaded from {path}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise