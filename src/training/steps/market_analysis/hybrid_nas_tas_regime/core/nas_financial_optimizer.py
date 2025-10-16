"""
NAS-Specific Financial Optimizer for Neural Architecture Search

This module provides a custom optimizer specifically designed for financial neural
architecture search, featuring adaptive learning rates, financial loss functions,
and regularization techniques optimized for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam, SGD, AdamW
from torch.optim.lr_scheduler import _LRScheduler, StepLR, CosineAnnealingLR, ReduceLROnPlateau
import warnings
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

logger = logging.getLogger(__name__)

class LossFunction(Enum):
    """Financial-specific loss functions."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    COMBINED_FINANCIAL = "combined_financial"
    ASYMMETRIC = "asymmetric"
    ROBUST_FINANCIAL = "robust_financial"

class LearningRateSchedule(Enum):
    """Learning rate scheduling strategies."""
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    VOLATILITY_ADAPTIVE = "volatility_adaptive"
    REGIME_ADAPTIVE = "regime_adaptive"
    PERFORMANCE_BASED = "performance_based"

@dataclass
class NASOptimizerConfig:
    """Configuration for NAS financial optimizer."""
    # Base optimizer settings
    base_optimizer: str = "adam"  # adam, sgd, adamw
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999

    # Financial loss function
    loss_function: LossFunction = LossFunction.COMBINED_FINANCIAL
    loss_weights: Dict[LossFunction, float] = field(default_factory=lambda: {
        LossFunction.SHARPE_RATIO: 1.0,
        LossFunction.MAX_DRAWDOWN: 0.5,
        LossFunction.SORTINO_RATIO: 0.3,
        LossFunction.CALMAR_RATIO: 0.3
    })

    # Learning rate scheduling
    lr_schedule: LearningRateSchedule = LearningRateSchedule.PERFORMANCE_BASED
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    min_lr: float = 1e-6
    max_lr: float = 0.1

    # Financial-specific settings
    volatility_adjustment: bool = True
    regime_adaptation: bool = True
    asymmetric_loss: bool = True
    early_stopping_patience: int = 20
    gradient_clip_norm: float = 1.0

    # Regularization
    dropout_rate: float = 0.1
    l1_regularization: float = 0.0
    l2_regularization: float = 0.01

    # Trading-specific
    transaction_cost_penalty: float = 0.001
    position_limit_penalty: float = 0.01
    risk_adjustment_factor: float = 0.1

class FinancialLossFunctions:
    """Collection of financial-specific loss functions."""

    @staticmethod
    def sharpe_ratio_loss(returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Loss based on negative Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        mean_excess = torch.mean(excess_returns)
        std_excess = torch.std(excess_returns, unbiased=False)

        if std_excess == 0:
            return torch.tensor(0.0, device=returns.device)

        sharpe = mean_excess / (std_excess + 1e-8)
        return -sharpe  # Negative because we want to maximize Sharpe

    @staticmethod
    def max_drawdown_loss(returns: torch.Tensor) -> torch.Tensor:
        """Loss based on maximum drawdown."""
        cumulative = torch.cumprod(1 + returns, dim=0)
        peak = torch.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-8)
        max_dd = torch.max(drawdown)
        return max_dd

    @staticmethod
    def sortino_ratio_loss(returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Loss based on negative Sortino ratio."""
        excess_returns = returns - risk_free_rate
        negative_returns = torch.where(excess_returns < 0, excess_returns, torch.zeros_like(excess_returns))
        downside_deviation = torch.sqrt(torch.mean(negative_returns ** 2) + 1e-8)
        mean_excess = torch.mean(excess_returns)

        sortino = mean_excess / (downside_deviation + 1e-8)
        return -sortino

    @staticmethod
    def calmar_ratio_loss(returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Loss based on negative Calmar ratio."""
        annual_return = torch.mean(returns) * 252
        max_dd = FinancialLossFunctions.max_drawdown_loss(returns)

        if max_dd == 0:
            return torch.tensor(0.0, device=returns.device)

        calmar = annual_return / max_dd
        return -calmar

    @staticmethod
    def profit_factor_loss(returns: torch.Tensor) -> torch.Tensor:
        """Loss based on negative profit factor."""
        positive_returns = torch.sum(torch.where(returns > 0, returns, torch.zeros_like(returns)))
        negative_returns = torch.sum(torch.where(returns < 0, -returns, torch.zeros_like(returns)))

        if negative_returns == 0:
            profit_factor = float('inf')
        else:
            profit_factor = positive_returns / negative_returns

        return -profit_factor

    @staticmethod
    def combined_financial_loss(returns: torch.Tensor,
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               config: NASOptimizerConfig) -> torch.Tensor:
        """Combined financial loss function."""
        total_loss = 0.0

        # Sharpe ratio loss
        if LossFunction.SHARPE_RATIO in config.loss_weights:
            sharpe_loss = FinancialLossFunctions.sharpe_ratio_loss(returns)
            total_loss += config.loss_weights[LossFunction.SHARPE_RATIO] * sharpe_loss

        # Max drawdown loss
        if LossFunction.MAX_DRAWDOWN in config.loss_weights:
            dd_loss = FinancialLossFunctions.max_drawdown_loss(returns)
            total_loss += config.loss_weights[LossFunction.MAX_DRAWDOWN] * dd_loss

        # Sortino ratio loss
        if LossFunction.SORTINO_RATIO in config.loss_weights:
            sortino_loss = FinancialLossFunctions.sortino_ratio_loss(returns)
            total_loss += config.loss_weights[LossFunction.SORTINO_RATIO] * sortino_loss

        # Calmar ratio loss
        if LossFunction.CALMAR_RATIO in config.loss_weights:
            calmar_loss = FinancialLossFunctions.calmar_ratio_loss(returns)
            total_loss += config.loss_weights[LossFunction.CALMAR_RATIO] * calmar_loss

        return total_loss

    @staticmethod
    def asymmetric_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Asymmetric loss that penalizes different types of errors differently."""
        errors = targets - predictions

        # Penalize overestimation more than underestimation for financial predictions
        loss = torch.where(
            errors > 0,  # Underestimation (predicted < actual)
            0.5 * errors ** 2,  # Smaller penalty for underestimation
            2.0 * errors ** 2   # Larger penalty for overestimation
        )

        return torch.mean(loss)

    @staticmethod
    def robust_financial_loss(returns: torch.Tensor, predictions: torch.Tensor,
                            targets: torch.Tensor) -> torch.Tensor:
        """Robust financial loss with outlier handling."""
        # Huber loss for robustness
        huber_loss = F.huber_loss(predictions, targets, reduction='mean', delta=1.0)

        # Financial performance loss
        financial_loss = FinancialLossFunctions.combined_financial_loss(
            returns, predictions, targets, NASOptimizerConfig()
        )

        return huber_loss + 0.1 * financial_loss

class VolatilityAdaptiveScheduler(_LRScheduler):
    """Learning rate scheduler that adapts based on market volatility."""

    def __init__(self, optimizer: Optimizer, volatility_window: int = 20,
                 volatility_threshold: float = 0.02, last_epoch: int = -1):
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
        self.volatility_history = []
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rate based on recent volatility."""
        if len(self.volatility_history) < self.volatility_window:
            return [base_lr * self.optimizer.param_groups[0]['lr'] for base_lr in self.base_lrs]

        recent_volatility = np.mean(self.volatility_history[-self.volatility_window:])

        if recent_volatility > self.volatility_threshold:
            # High volatility - reduce learning rate
            lr_multiplier = 0.5
        else:
            # Low volatility - increase learning rate
            lr_multiplier = 1.0

        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

class RegimeAdaptiveScheduler(_LRScheduler):
    """Learning rate scheduler that adapts based on market regimes."""

    def __init__(self, optimizer: Optimizer, regime_data: Dict[str, Any] = None,
                 last_epoch: int = -1):
        self.regime_data = regime_data or {}
        self.regime_multipliers = {
            'trending': 1.0,
            'ranging': 0.8,
            'volatile': 0.6,
            'stable': 1.2
        }
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rate based on current market regime."""
        if not self.regime_data:
            return [base_lr for base_lr in self.base_lrs]

        # Determine current regime (simplified)
        current_regime = self._get_current_regime()

        if current_regime in self.regime_multipliers:
            lr_multiplier = self.regime_multipliers[current_regime]
        else:
            lr_multiplier = 1.0

        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

    def _get_current_regime(self) -> str:
        """Determine current market regime."""
        # Simplified regime detection
        # In practice, this would use actual regime classification
        return 'trending'  # Default

class PerformanceBasedScheduler(_LRScheduler):
    """Learning rate scheduler based on model performance."""

    def __init__(self, optimizer: Optimizer, performance_history: List[float] = None,
                 improvement_threshold: float = 0.01, last_epoch: int = -1):
        self.performance_history = performance_history or []
        self.improvement_threshold = improvement_threshold
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rate based on performance improvement."""
        if len(self.performance_history) < 10:
            return [base_lr for base_lr in self.base_lrs]

        recent_performance = np.mean(self.performance_history[-5:])
        previous_performance = np.mean(self.performance_history[-10:-5])

        if recent_performance > previous_performance + self.improvement_threshold:
            # Performance improving - increase LR
            lr_multiplier = 1.1
        elif recent_performance < previous_performance - self.improvement_threshold:
            # Performance degrading - decrease LR
            lr_multiplier = 0.9
        else:
            # Stable performance - keep LR
            lr_multiplier = 1.0

        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

class FinancialAdam(Optimizer):
    """Adam optimizer with financial-specific enhancements."""

    def __init__(self, params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False,
                 volatility_adjustment: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, volatility_adjustment=volatility_adjustment)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Volatility adjustment
                if group['volatility_adjustment']:
                    grad = self._apply_volatility_adjustment(grad, state)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * np.sqrt(bias_correction2) / bias_correction1

                p.data.add_(exp_avg / denom, alpha=-step_size)

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

        return loss

    def _apply_volatility_adjustment(self, grad: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """Apply volatility adjustment to gradients."""
        # Simplified volatility adjustment
        # In practice, this would use market volatility data
        return grad

class NASFinancialOptimizer:
    """
    NAS-specific financial optimizer for neural architecture search.

    Provides advanced optimization techniques specifically designed for financial
    neural networks, including adaptive learning rates, financial loss functions,
    and regularization optimized for financial time series.
    """

    def __init__(self, model: nn.Module, config: NASOptimizerConfig):
        """Initialize the NAS financial optimizer."""
        tprint("🚀 [NAS_FINANCIAL_OPTIMIZER] Initializing NAS Financial Optimizer", color="cyan", bold=True)
        tprint(f"📊 [NAS_FINANCIAL_OPTIMIZER] Base Optimizer: {config.base_optimizer}", color="blue")
        tprint(f"📊 [NAS_FINANCIAL_OPTIMIZER] Loss Function: {config.loss_function.value}", color="blue")
        tprint(f"📊 [NAS_FINANCIAL_OPTIMIZER] LR Schedule: {config.lr_schedule.value}", color="blue")
        self.config = config
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize base optimizer
        tprint("🔧 [NAS_FINANCIAL_OPTIMIZER] Creating base optimizer", color="yellow")
        self.optimizer = self._create_base_optimizer()

        # Initialize learning rate scheduler
        tprint("📈 [NAS_FINANCIAL_OPTIMIZER] Creating learning rate scheduler", color="yellow")
        self.scheduler = self._create_scheduler()

        # Loss function
        tprint("🎯 [NAS_FINANCIAL_OPTIMIZER] Creating loss function", color="yellow")
        self.criterion = self._create_loss_function()

        # Training state
        tprint("📊 [NAS_FINANCIAL_OPTIMIZER] Initializing training state", color="blue")
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.performance_history = []
        self.volatility_history = []

        tprint("✅ [NAS_FINANCIAL_OPTIMIZER] NAS Financial Optimizer initialized successfully", color="green", bold=True)
        self.logger.info("✅ NAS Financial Optimizer initialized")
        self.logger.info(f"   Base Optimizer: {config.base_optimizer}")
        self.logger.info(f"   Loss Function: {config.loss_function.value}")
        self.logger.info(f"   LR Schedule: {config.lr_schedule.value}")

    def _create_base_optimizer(self) -> Optimizer:
        """Create base optimizer based on configuration."""
        if self.config.base_optimizer == "adam":
            return FinancialAdam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                volatility_adjustment=self.config.volatility_adjustment
            )
        elif self.config.base_optimizer == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.base_optimizer == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )

    def _create_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler."""
        if self.config.lr_schedule == LearningRateSchedule.STEP:
            return StepLR(self.optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma)
        elif self.config.lr_schedule == LearningRateSchedule.COSINE:
            return CosineAnnealingLR(self.optimizer, T_max=self.config.lr_step_size)
        elif self.config.lr_schedule == LearningRateSchedule.VOLATILITY_ADAPTIVE:
            return VolatilityAdaptiveScheduler(self.optimizer)
        elif self.config.lr_schedule == LearningRateSchedule.REGIME_ADAPTIVE:
            return RegimeAdaptiveScheduler(self.optimizer)
        elif self.config.lr_schedule == LearningRateSchedule.PERFORMANCE_BASED:
            return PerformanceBasedScheduler(self.optimizer, self.performance_history)
        else:
            return StepLR(self.optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma)

    def _create_loss_function(self) -> Callable:
        """Create financial-specific loss function."""
        if self.config.loss_function == LossFunction.SHARPE_RATIO:
            return lambda returns, pred, target: FinancialLossFunctions.sharpe_ratio_loss(returns)
        elif self.config.loss_function == LossFunction.MAX_DRAWDOWN:
            return lambda returns, pred, target: FinancialLossFunctions.max_drawdown_loss(returns)
        elif self.config.loss_function == LossFunction.COMBINED_FINANCIAL:
            return lambda returns, pred, target: FinancialLossFunctions.combined_financial_loss(
                returns, pred, target, self.config
            )
        elif self.config.loss_function == LossFunction.ASYMMETRIC:
            return lambda returns, pred, target: FinancialLossFunctions.asymmetric_loss(pred, target)
        elif self.config.loss_function == LossFunction.ROBUST_FINANCIAL:
            return lambda returns, pred, target: FinancialLossFunctions.robust_financial_loss(
                returns, pred, target
            )
        else:
            return F.mse_loss  # Default to MSE

    def step(self, returns: torch.Tensor, predictions: torch.Tensor,
             targets: torch.Tensor, market_volatility: Optional[float] = None) -> float:
        """Perform optimization step with financial data."""
        self.optimizer.zero_grad()

        # Calculate loss
        if self.config.loss_function in [LossFunction.SHARPE_RATIO, LossFunction.MAX_DRAWDOWN,
                                       LossFunction.COMBINED_FINANCIAL, LossFunction.ASYMMETRIC,
                                       LossFunction.ROBUST_FINANCIAL]:
            loss = self.criterion(returns, predictions, targets)
        else:
            loss = self.criterion(predictions, targets)

        # Add regularization
        loss = self._apply_regularization(loss)

        # Add financial penalties
        loss = self._apply_financial_penalties(loss, returns, predictions)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

        # Optimizer step
        self.optimizer.step()

        # Update scheduler
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        # Track performance
        current_loss = loss.item()
        self.performance_history.append(current_loss)

        # Update volatility history
        if market_volatility is not None:
            self.volatility_history.append(market_volatility)

        # Early stopping check
        self._check_early_stopping(current_loss)

        self.current_epoch += 1

        return current_loss

    def _apply_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply regularization techniques."""
        if self.config.l1_regularization > 0:
            l1_loss = sum(torch.sum(torch.abs(p)) for p in self.model.parameters())
            loss += self.config.l1_regularization * l1_loss

        if self.config.l2_regularization > 0:
            l2_loss = sum(torch.sum(p ** 2) for p in self.model.parameters())
            loss += self.config.l2_regularization * l2_loss

        return loss

    def _apply_financial_penalties(self, loss: torch.Tensor, returns: torch.Tensor,
                                 predictions: torch.Tensor) -> torch.Tensor:
        """Apply financial-specific penalties."""
        # Transaction cost penalty
        if self.config.transaction_cost_penalty > 0:
            transaction_changes = torch.abs(predictions[1:] - predictions[:-1])
            transaction_penalty = torch.mean(transaction_changes) * self.config.transaction_cost_penalty
            loss += transaction_penalty

        # Position limit penalty
        if self.config.position_limit_penalty > 0:
            position_magnitude = torch.mean(torch.abs(predictions))
            position_penalty = torch.relu(position_magnitude - 1.0) * self.config.position_limit_penalty
            loss += position_penalty

        return loss

    def _check_early_stopping(self, current_loss: float):
        """Check early stopping conditions."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.config.early_stopping_patience:
            self.logger.info(f"Early stopping at epoch {self.current_epoch}")
            return True

        return False

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adapt_to_market_conditions(self, market_volatility: float, market_regime: str):
        """Adapt optimizer to current market conditions."""
        # Volatility-based learning rate adjustment
        if self.config.volatility_adjustment:
            if market_volatility > 0.02:  # High volatility
                self.set_learning_rate(self.config.learning_rate * 0.5)
            else:
                self.set_learning_rate(self.config.learning_rate)

        # Regime-based adjustments
        if self.config.regime_adaptation:
            if market_regime == 'volatile':
                self.set_learning_rate(self.config.learning_rate * 0.7)
            elif market_regime == 'stable':
                self.set_learning_rate(self.config.learning_rate * 1.2)

    def get_optimizer_state(self) -> Dict[str, Any]:
        """Get optimizer state information."""
        return {
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'learning_rate': self.get_learning_rate(),
            'performance_history_length': len(self.performance_history),
            'volatility_history_length': len(self.volatility_history)
        }

    def save_optimizer_state(self, filepath: str) -> bool:
        """Save optimizer state to disk."""
        try:
            state = {
                'config': self.config.__dict__,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                'current_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'patience_counter': self.patience_counter,
                'performance_history': self.performance_history,
                'volatility_history': self.volatility_history
            }

            torch.save(state, filepath)
            self.logger.info(f"✅ Optimizer state saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save optimizer state: {e}")
            return False

    def load_optimizer_state(self, filepath: str) -> bool:
        """Load optimizer state from disk."""
        try:
            state = torch.load(filepath)

            self.config = NASOptimizerConfig(**state['config'])
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])

            if state['scheduler_state'] and hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(state['scheduler_state'])

            self.current_epoch = state.get('current_epoch', 0)
            self.best_loss = state.get('best_loss', float('inf'))
            self.patience_counter = state.get('patience_counter', 0)
            self.performance_history = state.get('performance_history', [])
            self.volatility_history = state.get('volatility_history', [])

            self.logger.info(f"✅ Optimizer state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load optimizer state: {e}")
            return False

def create_nas_financial_optimizer(model: nn.Module, config: NASOptimizerConfig) -> NASFinancialOptimizer:
    """Create NAS financial optimizer instance."""
    return NASFinancialOptimizer(model, config)

def quick_financial_optimization(model: nn.Module,
                               train_data: torch.Tensor,
                               train_targets: torch.Tensor,
                               config: Optional[NASOptimizerConfig] = None) -> NASFinancialOptimizer:
    """Quick financial optimization with default settings."""
    if config is None:
        config = NASOptimizerConfig(
            base_optimizer="adam",
            learning_rate=0.001,
            loss_function=LossFunction.COMBINED_FINANCIAL,
            lr_schedule=LearningRateSchedule.PERFORMANCE_BASED
        )

    optimizer = NASFinancialOptimizer(model, config)

    # Simple training loop
    for epoch in range(10):
        predictions = model(train_data)
        loss = F.mse_loss(predictions, train_targets)
        optimizer.step(torch.zeros_like(train_targets), predictions, train_targets)

    return optimizer
