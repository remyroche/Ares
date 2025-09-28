"""
Regime-Aware Training System

This module provides training capabilities that adapt to market regimes,
including regime-specific model training, regime transition handling,
and regime-aware validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from .financial_architecture_primitives import RegimeType, FinancialActivationType
from .financial_loss_functions import FinancialLossType, create_financial_loss_function, CompositeFinancialLoss
from .financial_optimizers import FinancialOptimizerConfig, create_financial_optimizer

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Training modes for regime-aware training."""
    REGIME_SPECIFIC = "regime_specific"
    REGIME_ADAPTIVE = "regime_adaptive"
    REGIME_ENSEMBLE = "regime_ensemble"
    REGIME_TRANSITION = "regime_transition"
    REGIME_CONTINUAL = "regime_continual"


class RegimeTransitionStrategy(Enum):
    """Strategies for handling regime transitions."""
    GRADUAL_ADAPTATION = "gradual_adaptation"
    SUDDEN_SWITCH = "sudden_switch"
    ENSEMBLE_FUSION = "ensemble_fusion"
    META_LEARNING = "meta_learning"
    CONTINUAL_LEARNING = "continual_learning"


@dataclass
class RegimeAwareTrainingConfig:
    """Configuration for regime-aware training."""
    # Base training settings
    training_mode: TrainingMode = TrainingMode.REGIME_ADAPTIVE
    max_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_detection_frequency: int = 10  # Detect regimes every N epochs
    regime_adaptation_rate: float = 0.1
    regime_memory_size: int = 1000
    
    # Regime transition handling
    transition_strategy: RegimeTransitionStrategy = RegimeTransitionStrategy.GRADUAL_ADAPTATION
    transition_detection_threshold: float = 0.3
    transition_adaptation_rate: float = 0.2
    transition_memory_size: int = 500
    
    # Regime-specific training
    enable_regime_specific_training: bool = True
    regime_specific_epochs: int = 100
    regime_specific_learning_rate: float = 0.0005
    regime_specific_batch_size: int = 16
    
    # Regime ensemble
    enable_regime_ensemble: bool = False
    ensemble_size: int = 3
    ensemble_fusion_method: str = "weighted_average"  # weighted_average, majority_vote, stacking
    
    # Regime continual learning
    enable_continual_learning: bool = False
    continual_learning_rate: float = 0.0001
    memory_replay_size: int = 1000
    knowledge_distillation_weight: float = 0.5
    
    # Loss function configuration
    primary_loss_type: FinancialLossType = FinancialLossType.SHARPE_LOSS
    secondary_loss_types: List[FinancialLossType] = field(default_factory=lambda: [
        FinancialLossType.DRAWDOWN_LOSS, FinancialLossType.WIN_RATE_LOSS
    ])
    loss_weights: Dict[FinancialLossType, float] = field(default_factory=lambda: {
        FinancialLossType.SHARPE_LOSS: 0.5,
        FinancialLossType.DRAWDOWN_LOSS: 0.3,
        FinancialLossType.WIN_RATE_LOSS: 0.2
    })
    
    # Optimizer configuration
    optimizer_type: str = "adam"  # adam, sgd, rmsprop
    optimizer_momentum: float = 0.9
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    
    # Learning rate scheduling
    enable_lr_scheduling: bool = True
    lr_scheduler_type: str = "reduce_on_plateau"  # reduce_on_plateau, cosine_annealing, step
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-6
    
    # Validation
    validation_frequency: int = 10
    validation_metrics: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"
    ])
    
    # Performance tracking
    performance_window: int = 50
    min_performance_samples: int = 10


@dataclass
class RegimeTrainingResult:
    """Result from regime-aware training."""
    best_model_state: Dict[str, Any]
    best_score: float
    training_history: List[Dict[str, Any]]
    regime_performance: Dict[int, Dict[str, float]]
    regime_transitions: List[Dict[str, Any]]
    financial_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    execution_time: float
    n_epochs: int


class RegimeAwareTrainer:
    """Regime-aware trainer for financial models."""
    
    def __init__(self, config: RegimeAwareTrainingConfig):
        tprint("🎯 [REGIME_AWARE_TRAINER] Initializing Regime-Aware Trainer", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Training state
        tprint("📊 [REGIME_AWARE_TRAINER] Setting up training state", color="cyan")
        self.current_regime = None
        self.regime_history = []
        self.regime_models = {}
        self.regime_performance = {}
        self.regime_transitions = []
        
        # Performance tracking
        tprint("📈 [REGIME_AWARE_TRAINER] Setting up performance tracking", color="cyan")
        self.training_history = []
        self.validation_history = []
        self.regime_adaptation_history = []
        
        # Regime detection
        tprint("🔍 [REGIME_AWARE_TRAINER] Setting up regime detection", color="cyan")
        self.regime_detector = None
        self.regime_detection_history = []
        
        # Loss function
        tprint("📉 [REGIME_AWARE_TRAINER] Creating loss function", color="cyan")
        self.loss_function = self._create_loss_function()
        
        # Optimizer and scheduler
        tprint("⚙️ [REGIME_AWARE_TRAINER] Setting up optimizer and scheduler", color="cyan")
        self.optimizer = None
        self.scheduler = None
        
        tprint_success("✅ [REGIME_AWARE_TRAINER] Regime-Aware Trainer initialized")
        self.logger.info("✅ Regime-Aware Trainer initialized")
        self.logger.info(f"   Training Mode: {config.training_mode.value}")
        self.logger.info(f"   Regime Awareness: {config.enable_regime_awareness}")
        self.logger.info(f"   Transition Strategy: {config.transition_strategy.value}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        tprint(f"📉 [REGIME_AWARE_TRAINER] _create_loss_function() called", color="blue")
        tprint(f"📊 [REGIME_AWARE_TRAINER] Primary loss: {self.config.primary_loss_type.value}, Secondary losses: {len(self.config.secondary_loss_types)}", color="cyan")
        if len(self.config.secondary_loss_types) > 0:
            # Create composite loss function
            tprint("📉 [REGIME_AWARE_TRAINER] Creating composite loss function", color="yellow")
            loss_weights = self.config.loss_weights
            loss_function = CompositeFinancialLoss(
                FinancialLossConfig(
                    loss_type=self.config.primary_loss_type,
                    enable_regime_awareness=self.config.enable_regime_awareness
                ),
                loss_weights
            )
        else:
            # Create single loss function
            tprint("📉 [REGIME_AWARE_TRAINER] Creating single loss function", color="yellow")
            loss_config = FinancialLossConfig(
                loss_type=self.config.primary_loss_type,
                enable_regime_awareness=self.config.enable_regime_awareness
            )
            loss_function = create_financial_loss_function(loss_config)
        
        tprint_success("✅ [REGIME_AWARE_TRAINER] _create_loss_function() completed successfully")
        tprint(f"📊 [REGIME_AWARE_TRAINER] _create_loss_function() outcome: {type(loss_function).__name__}", color="green")
        return loss_function
    
    def train(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
              validation_data: Tuple[torch.Tensor, torch.Tensor],
              regime_data: Optional[Dict[str, Any]] = None) -> RegimeTrainingResult:
        """Train model with regime awareness."""
        start_time = time.time()
        tprint("🚀 [REGIME_AWARE_TRAINER] Starting Regime-Aware Training", color="cyan", bold=True)
        tprint(f"📊 [REGIME_AWARE_TRAINER] Training mode: {self.config.training_mode.value}", color="blue")
        self.logger.info("🚀 Starting Regime-Aware Training...")
        
        try:
            # Initialize training components
            tprint("🔧 [REGIME_AWARE_TRAINER] Initializing training components", color="yellow")
            self._initialize_training_components(model)
            
            # Detect initial regimes
            if regime_data is not None:
                tprint("🔍 [REGIME_AWARE_TRAINER] Detecting initial regimes", color="yellow")
                self._detect_initial_regimes(regime_data)
            
            # Training loop
            tprint(f"🔄 [REGIME_AWARE_TRAINER] Starting training loop for {self.config.max_epochs} epochs", color="yellow")
            best_model_state = None
            best_score = -np.inf
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Detect regime changes
                if self.config.enable_regime_awareness and epoch % self.config.regime_detection_frequency == 0:
                    tprint(f"🔍 [REGIME_AWARE_TRAINER] Detecting regime changes at epoch {epoch}", color="blue")
                    self._detect_regime_changes(epoch, regime_data)
                
                # Training step
                train_metrics = self._training_step(model, train_data, regime_data)
                
                # Validation step
                val_metrics = self._validation_step(model, validation_data, regime_data)
                
                # Regime-specific training
                if self.config.enable_regime_specific_training:
                    self._regime_specific_training(model, train_data, regime_data)
                
                # Regime transition handling
                if self.config.enable_regime_awareness:
                    self._handle_regime_transitions(model, epoch, regime_data)
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                
                # Track training history
                self._update_training_history(epoch, train_metrics, val_metrics)
                
                # Update best model
                current_score = val_metrics.get('sharpe_ratio', val_metrics.get('loss', 0.0))
                if current_score > best_score:
                    best_score = current_score
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if self.config.enable_early_stopping and patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log progress
                if epoch % 100 == 0:
                    self.logger.debug(f"Epoch {epoch}: Score = {current_score:.4f}, Regime = {self.current_regime}")
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            execution_time = time.time() - start_time
            
            # Calculate final metrics
            financial_metrics = self._calculate_financial_metrics()
            risk_metrics = self._calculate_risk_metrics()
            regime_performance = self._calculate_regime_performance()
            
            return RegimeTrainingResult(
                best_model_state=best_model_state,
                best_score=best_score,
                training_history=self.training_history,
                regime_performance=regime_performance,
                regime_transitions=self.regime_transitions,
                financial_metrics=financial_metrics,
                risk_metrics=risk_metrics,
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_epochs=len(self.training_history)
            )
            
        except Exception as e:
            self.logger.error(f"Regime-aware training failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _initialize_training_components(self, model: nn.Module):
        """Initialize training components."""
        # Create optimizer
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.optimizer_beta1, self.config.optimizer_beta2),
                eps=self.config.optimizer_epsilon
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.optimizer_momentum
            )
        else:
            self.optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Create scheduler
        if self.config.enable_lr_scheduling:
            if self.config.lr_scheduler_type == "reduce_on_plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=self.config.lr_factor,
                    patience=self.config.lr_patience,
                    min_lr=self.config.lr_min
                )
            elif self.config.lr_scheduler_type == "cosine_annealing":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.max_epochs,
                    eta_min=self.config.lr_min
                )
            elif self.config.lr_scheduler_type == "step":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=self.config.lr_patience,
                    gamma=self.config.lr_factor
                )
    
    def _detect_initial_regimes(self, regime_data: Dict[str, Any]):
        """Detect initial regimes from regime data."""
        if 'regime_predictions' in regime_data:
            regime_predictions = regime_data['regime_predictions']
            unique_regimes = np.unique(regime_predictions)
            self.regime_history = unique_regimes.tolist()
            self.current_regime = unique_regimes[0] if len(unique_regimes) > 0 else 0
            self.logger.info(f"Detected initial regimes: {unique_regimes}")
    
    def _detect_regime_changes(self, epoch: int, regime_data: Optional[Dict[str, Any]]):
        """Detect regime changes during training."""
        if regime_data is None:
            return
        
        # Simplified regime change detection
        # In practice, this would use actual regime detection
        if epoch > 0 and epoch % 50 == 0:
            # Simulate regime change
            new_regime = (epoch // 50) % 4
            if new_regime != self.current_regime:
                self._handle_regime_transition(self.current_regime, new_regime, epoch)
                self.current_regime = new_regime
    
    def _handle_regime_transition(self, from_regime: int, to_regime: int, epoch: int):
        """Handle regime transition."""
        transition = {
            'from_regime': from_regime,
            'to_regime': to_regime,
            'epoch': epoch,
            'timestamp': datetime.now(),
            'strategy': self.config.transition_strategy.value
        }
        
        self.regime_transitions.append(transition)
        
        # Apply transition strategy
        if self.config.transition_strategy == RegimeTransitionStrategy.GRADUAL_ADAPTATION:
            self._gradual_adaptation(from_regime, to_regime)
        elif self.config.transition_strategy == RegimeTransitionStrategy.SUDDEN_SWITCH:
            self._sudden_switch(from_regime, to_regime)
        elif self.config.transition_strategy == RegimeTransitionStrategy.ENSEMBLE_FUSION:
            self._ensemble_fusion(from_regime, to_regime)
        elif self.config.transition_strategy == RegimeTransitionStrategy.META_LEARNING:
            self._meta_learning_adaptation(from_regime, to_regime)
        elif self.config.transition_strategy == RegimeTransitionStrategy.CONTINUAL_LEARNING:
            self._continual_learning_adaptation(from_regime, to_regime)
        
        self.logger.info(f"Regime transition: {from_regime} -> {to_regime} at epoch {epoch}")
    
    def _gradual_adaptation(self, from_regime: int, to_regime: int):
        """Gradual adaptation to new regime."""
        # Gradually adjust learning rate
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= (1 + self.config.transition_adaptation_rate)
        
        # Store regime-specific performance
        self.regime_performance[from_regime] = self._get_current_performance()
    
    def _sudden_switch(self, from_regime: int, to_regime: int):
        """Sudden switch to new regime."""
        # Reset optimizer state
        if self.optimizer:
            self.optimizer.zero_grad()
        
        # Store regime-specific model
        self.regime_models[from_regime] = self._get_current_model_state()
    
    def _ensemble_fusion(self, from_regime: int, to_regime: int):
        """Ensemble fusion of regime models."""
        # Store current model for ensemble
        if from_regime not in self.regime_models:
            self.regime_models[from_regime] = self._get_current_model_state()
        
        # Create ensemble if we have multiple regime models
        if len(self.regime_models) > 1:
            self._create_regime_ensemble()
    
    def _meta_learning_adaptation(self, from_regime: int, to_regime: int):
        """Meta-learning adaptation to new regime."""
        # Simplified meta-learning adaptation
        # In practice, this would use actual meta-learning techniques
        if from_regime in self.regime_performance:
            previous_performance = self.regime_performance[from_regime]
            # Use previous performance to adapt to new regime
            self._adapt_based_on_previous_performance(previous_performance)
    
    def _continual_learning_adaptation(self, from_regime: int, to_regime: int):
        """Continual learning adaptation to new regime."""
        # Store knowledge from previous regime
        if from_regime not in self.regime_models:
            self.regime_models[from_regime] = self._get_current_model_state()
        
        # Apply knowledge distillation
        if self.config.enable_continual_learning:
            self._apply_knowledge_distillation(from_regime, to_regime)
    
    def _training_step(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                      regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform training step."""
        model.train()
        self.optimizer.zero_grad()
        
        X_train, y_train = train_data
        
        # Forward pass
        predictions = model(X_train)
        
        # Calculate loss
        loss = self.loss_function(predictions, y_train, regime_data)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Calculate additional metrics
        mse = torch.nn.functional.mse_loss(predictions, y_train)
        mae = torch.nn.functional.l1_loss(predictions, y_train)
        
        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'mae': mae.item()
        }
    
    def _validation_step(self, model: nn.Module, validation_data: Tuple[torch.Tensor, torch.Tensor],
                        regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform validation step."""
        model.eval()
        
        with torch.no_grad():
            X_val, y_val = validation_data
            predictions = model(X_val)
            
            # Calculate validation metrics
            val_loss = self.loss_function(predictions, y_val, regime_data)
            
            # Calculate additional metrics
            mse = torch.nn.functional.mse_loss(predictions, y_val)
            mae = torch.nn.functional.l1_loss(predictions, y_val)
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics_from_predictions(predictions, y_val)
            
            return {
                'loss': val_loss.item(),
                'mse': mse.item(),
                'mae': mae.item(),
                **financial_metrics
            }
    
    def _regime_specific_training(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                                 regime_data: Optional[Dict[str, Any]] = None):
        """Perform regime-specific training."""
        if not self.config.enable_regime_specific_training:
            return
        
        # Create regime-specific optimizer
        regime_optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.regime_specific_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Regime-specific training loop
        for epoch in range(self.config.regime_specific_epochs):
            # Training step
            train_metrics = self._training_step(model, train_data, regime_data)
            
            # Update regime-specific performance
            if self.current_regime is not None:
                if self.current_regime not in self.regime_performance:
                    self.regime_performance[self.current_regime] = []
                self.regime_performance[self.current_regime].append(train_metrics)
    
    def _handle_regime_transitions(self, model: nn.Module, epoch: int, regime_data: Optional[Dict[str, Any]]):
        """Handle regime transitions during training."""
        if not self.config.enable_regime_awareness:
            return
        
        # Check for regime transitions
        if len(self.regime_transitions) > 0:
            latest_transition = self.regime_transitions[-1]
            if latest_transition['epoch'] == epoch:
                # Apply transition-specific adjustments
                self._apply_transition_adjustments(model, latest_transition)
    
    def _apply_transition_adjustments(self, model: nn.Module, transition: Dict[str, Any]):
        """Apply transition-specific adjustments."""
        from_regime = transition['from_regime']
        to_regime = transition['to_regime']
        
        # Adjust learning rate based on regime transition
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= (1 + self.config.transition_adaptation_rate)
        
        # Store regime-specific model state
        if from_regime not in self.regime_models:
            self.regime_models[from_regime] = model.state_dict().copy()
    
    def _update_training_history(self, epoch: int, train_metrics: Dict[str, float],
                                val_metrics: Dict[str, float]):
        """Update training history."""
        history_entry = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'regime': self.current_regime,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
            'timestamp': datetime.now()
        }
        
        self.training_history.append(history_entry)
        
        # Keep only recent history
        if len(self.training_history) > self.config.performance_window:
            self.training_history.pop(0)
    
    def _calculate_financial_metrics_from_predictions(self, predictions: torch.Tensor, 
                                                     targets: torch.Tensor) -> Dict[str, float]:
        """Calculate financial metrics from predictions."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate financial metrics
        mean_return = torch.mean(returns).item()
        std_return = torch.std(returns).item()
        
        # Sharpe ratio
        sharpe_ratio = mean_return / (std_return + 1e-8)
        
        # Win rate
        win_rate = torch.mean((returns > 0).float()).item()
        
        # Maximum drawdown
        cumulative_returns = torch.cumsum(returns, dim=0)
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        drawdown = running_max - cumulative_returns
        max_drawdown = torch.max(drawdown).item()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'mean_return': mean_return,
            'std_return': std_return
        }
    
    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics from training history."""
        if not self.training_history:
            return {}
        
        # Extract metrics from training history
        sharpe_ratios = [entry['val_metrics'].get('sharpe_ratio', 0.0) for entry in self.training_history]
        win_rates = [entry['val_metrics'].get('win_rate', 0.0) for entry in self.training_history]
        max_drawdowns = [entry['val_metrics'].get('max_drawdown', 0.0) for entry in self.training_history]
        
        return {
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'max_sharpe_ratio': np.max(sharpe_ratios),
            'mean_win_rate': np.mean(win_rates),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'final_sharpe_ratio': sharpe_ratios[-1] if sharpe_ratios else 0.0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics from training history."""
        if not self.training_history:
            return {}
        
        # Extract loss metrics
        losses = [entry['train_metrics']['loss'] for entry in self.training_history]
        
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'loss_volatility': np.std(losses)
        }
    
    def _calculate_regime_performance(self) -> Dict[int, Dict[str, float]]:
        """Calculate performance by regime."""
        regime_performance = {}
        
        for regime, performance_list in self.regime_performance.items():
            if performance_list:
                # Calculate average performance for this regime
                avg_performance = {}
                for metric in performance_list[0].keys():
                    values = [p[metric] for p in performance_list]
                    avg_performance[metric] = np.mean(values)
                
                regime_performance[regime] = avg_performance
        
        return regime_performance
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence."""
        if len(self.training_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        # Analyze loss convergence
        recent_losses = [entry['train_metrics']['loss'] for entry in self.training_history[-10:]]
        loss_std = np.std(recent_losses)
        
        # Analyze Sharpe ratio convergence
        recent_sharpe = [entry['val_metrics'].get('sharpe_ratio', 0.0) for entry in self.training_history[-10:]]
        sharpe_std = np.std(recent_sharpe)
        
        # Analyze regime stability
        recent_regimes = [entry['regime'] for entry in self.training_history[-10:]]
        regime_stability = len(set(recent_regimes)) / len(recent_regimes)
        
        return {
            'converged': loss_std < 0.01 and sharpe_std < 0.01,
            'loss_std': loss_std,
            'sharpe_std': sharpe_std,
            'regime_stability': regime_stability,
            'n_regime_transitions': len(self.regime_transitions)
        }
    
    def _get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.training_history:
            return {}
        
        return self.training_history[-1]['val_metrics']
    
    def _get_current_model_state(self) -> Dict[str, Any]:
        """Get current model state."""
        # This would return the current model state
        # For now, return empty dict
        return {}
    
    def _create_regime_ensemble(self):
        """Create ensemble of regime-specific models."""
        # Simplified ensemble creation
        # In practice, this would create an actual ensemble
        pass
    
    def _adapt_based_on_previous_performance(self, previous_performance: Dict[str, float]):
        """Adapt based on previous regime performance."""
        # Simplified adaptation
        # In practice, this would use actual adaptation techniques
        pass
    
    def _apply_knowledge_distillation(self, from_regime: int, to_regime: int):
        """Apply knowledge distillation between regimes."""
        # Simplified knowledge distillation
        # In practice, this would use actual knowledge distillation techniques
        pass
    
    def _create_error_result(self, error_message: str, execution_time: float) -> RegimeTrainingResult:
        """Create error result."""
        return RegimeTrainingResult(
            best_model_state={},
            best_score=0.0,
            training_history=[],
            regime_performance={},
            regime_transitions=[],
            financial_metrics={},
            risk_metrics={},
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_epochs=0
        )


def create_regime_aware_trainer(config: RegimeAwareTrainingConfig) -> RegimeAwareTrainer:
    """Create regime-aware trainer instance."""
    return RegimeAwareTrainer(config)