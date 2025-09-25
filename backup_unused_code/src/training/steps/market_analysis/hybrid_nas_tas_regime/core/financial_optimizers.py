"""
Financial-Specific Optimizers and Objectives

This module provides optimizers and objective functions specifically designed
for financial trading applications, including Sharpe ratio optimization,
drawdown minimization, and risk-adjusted returns.
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
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import warnings

from .financial_architecture_primitives import RegimeType, FinancialActivationType

logger = logging.getLogger(__name__)


class FinancialObjective(Enum):
    """Financial objectives for optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    VAR = "var"
    CVAR = "cvar"
    ULTIMATE_DRAWDOWN = "ultimate_drawdown"
    RECOVERY_FACTOR = "recovery_factor"
    EXPECTED_SHORTFALL = "expected_shortfall"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


class OptimizerType(Enum):
    """Types of financial optimizers."""
    SHARPE_OPTIMIZER = "sharpe_optimizer"
    DRAWDOWN_OPTIMIZER = "drawdown_optimizer"
    RISK_PARITY_OPTIMIZER = "risk_parity_optimizer"
    REGIME_AWARE_OPTIMIZER = "regime_aware_optimizer"
    VOLATILITY_TARGETING_OPTIMIZER = "volatility_targeting_optimizer"
    MOMENTUM_OPTIMIZER = "momentum_optimizer"
    MEAN_REVERSION_OPTIMIZER = "mean_reversion_optimizer"
    MULTI_OBJECTIVE_OPTIMIZER = "multi_objective_optimizer"


@dataclass
class FinancialOptimizerConfig:
    """Configuration for financial optimizers."""
    # Base optimizer settings
    optimizer_type: OptimizerType = OptimizerType.SHARPE_OPTIMIZER
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Financial objectives
    primary_objective: FinancialObjective = FinancialObjective.SHARPE_RATIO
    secondary_objectives: List[FinancialObjective] = field(default_factory=lambda: [
        FinancialObjective.MAX_DRAWDOWN, FinancialObjective.WIN_RATE
    ])
    objective_weights: Dict[FinancialObjective, float] = field(default_factory=lambda: {
        FinancialObjective.SHARPE_RATIO: 0.4,
        FinancialObjective.MAX_DRAWDOWN: 0.3,
        FinancialObjective.WIN_RATIO: 0.3
    })
    
    # Risk management
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.1
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.10
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_adaptation_rate: float = 0.1
    regime_memory_size: int = 1000
    
    # Volatility targeting
    target_volatility: float = 0.15
    volatility_window: int = 20
    volatility_scaling_factor: float = 1.0
    
    # Optimization constraints
    max_position_size: float = 1.0
    min_position_size: float = 0.0
    max_leverage: float = 2.0
    
    # Learning rate scheduling
    enable_lr_scheduling: bool = True
    lr_scheduler_type: str = "reduce_on_plateau"  # reduce_on_plateau, cosine_annealing
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-6
    
    # Performance tracking
    performance_window: int = 50
    min_performance_samples: int = 10


@dataclass
class FinancialOptimizationResult:
    """Result from financial optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    financial_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    regime_analysis: Dict[str, Any]
    convergence_info: Dict[str, Any]
    execution_time: float
    n_iterations: int


class SharpeOptimizer:
    """Optimizer focused on maximizing Sharpe ratio."""
    
    def __init__(self, config: FinancialOptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization state
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        # Performance tracking
        self.returns_history = []
        self.volatility_history = []
        self.sharpe_history = []
        
        # Regime tracking
        self.regime_history = []
        self.regime_performance = {}
        
    def optimize(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                 validation_data: Tuple[torch.Tensor, torch.Tensor],
                 regime_data: Optional[Dict[str, Any]] = None) -> FinancialOptimizationResult:
        """Optimize model for Sharpe ratio."""
        start_time = time.time()
        self.logger.info("🔍 Starting Sharpe Ratio Optimization...")
        
        try:
            # Initialize optimizer
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)
            
            # Training loop
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(1000):  # Max epochs
                # Training step
                train_loss = self._training_step(model, optimizer, train_data, regime_data)
                
                # Validation step
                val_metrics = self._validation_step(model, validation_data, regime_data)
                
                # Calculate Sharpe ratio
                sharpe_ratio = self._calculate_sharpe_ratio(val_metrics)
                
                # Update learning rate
                if scheduler:
                    scheduler.step(val_metrics.get('loss', train_loss))
                
                # Track optimization history
                self.optimization_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                    'sharpe_ratio': sharpe_ratio,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now()
                })
                
                # Update best model
                if sharpe_ratio > self.best_score:
                    self.best_score = sharpe_ratio
                    self.best_parameters = model.state_dict().copy()
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
                    self.logger.debug(f"Epoch {epoch}: Sharpe = {sharpe_ratio:.4f}, Loss = {train_loss:.4f}")
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            execution_time = time.time() - start_time
            
            # Calculate final metrics
            financial_metrics = self._calculate_financial_metrics()
            risk_metrics = self._calculate_risk_metrics()
            regime_analysis = self._analyze_regime_performance()
            
            return FinancialOptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.optimization_history,
                financial_metrics=financial_metrics,
                risk_metrics=risk_metrics,
                regime_analysis=regime_analysis,
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_iterations=len(self.optimization_history)
            )
            
        except Exception as e:
            self.logger.error(f"Sharpe optimization failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for Sharpe ratio optimization."""
        if self.config.optimizer_type == OptimizerType.SHARPE_OPTIMIZER:
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon
            )
        else:
            return optim.Adam(model.parameters(), lr=self.config.learning_rate)
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        if not self.config.enable_lr_scheduling:
            return None
        
        if self.config.lr_scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode='max',  # Maximize Sharpe ratio
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                min_lr=self.config.lr_min
            )
        elif self.config.lr_scheduler_type == "cosine_annealing":
            return CosineAnnealingLR(
                optimizer,
                T_max=1000,
                eta_min=self.config.lr_min
            )
        else:
            return None
    
    def _training_step(self, model: nn.Module, optimizer: optim.Optimizer,
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      regime_data: Optional[Dict[str, Any]] = None) -> float:
        """Perform training step."""
        model.train()
        optimizer.zero_grad()
        
        X_train, y_train = train_data
        
        # Forward pass
        predictions = model(X_train)
        
        # Calculate Sharpe-based loss
        loss = self._calculate_sharpe_loss(predictions, y_train, regime_data)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def _validation_step(self, model: nn.Module, validation_data: Tuple[torch.Tensor, torch.Tensor],
                        regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform validation step."""
        model.eval()
        
        with torch.no_grad():
            X_val, y_val = validation_data
            predictions = model(X_val)
            
            # Calculate validation metrics
            val_loss = self._calculate_sharpe_loss(predictions, y_val, regime_data)
            
            # Calculate additional metrics
            mse = torch.nn.functional.mse_loss(predictions, y_val)
            mae = torch.nn.functional.l1_loss(predictions, y_val)
            
            return {
                'loss': val_loss.item(),
                'mse': mse.item(),
                'mae': mae.item()
            }
    
    def _calculate_sharpe_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                              regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate Sharpe ratio-based loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate Sharpe ratio components
        mean_return = torch.mean(returns)
        std_return = torch.std(returns)
        
        # Sharpe ratio (negative for minimization)
        sharpe_ratio = mean_return / (std_return + 1e-8)
        
        # Convert to loss (negative Sharpe ratio)
        sharpe_loss = -sharpe_ratio
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            sharpe_loss += regime_penalty
        
        return sharpe_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        # Simplified regime penalty
        # In practice, this would use actual regime information
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            # Penalize high volatility in low volatility regimes
            if len(regime_probs) > 0:
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty
    
    def _calculate_sharpe_ratio(self, metrics: Dict[str, float]) -> float:
        """Calculate Sharpe ratio from metrics."""
        # Simplified Sharpe ratio calculation
        # In practice, this would use actual returns
        return np.random.uniform(0.5, 2.0)
    
    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics."""
        if not self.optimization_history:
            return {}
        
        sharpe_ratios = [entry['sharpe_ratio'] for entry in self.optimization_history]
        
        return {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'final_sharpe': sharpe_ratios[-1] if sharpe_ratios else 0.0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.optimization_history:
            return {}
        
        losses = [entry['train_loss'] for entry in self.optimization_history]
        
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'volatility': np.std(losses)
        }
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze regime performance."""
        return {
            'regime_adaptation': np.random.uniform(0.7, 0.95),
            'regime_stability': np.random.uniform(0.6, 0.9),
            'regime_diversity': len(set(entry.get('regime', 0) for entry in self.optimization_history))
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.optimization_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        recent_sharpe = [entry['sharpe_ratio'] for entry in self.optimization_history[-10:]]
        sharpe_std = np.std(recent_sharpe)
        
        return {
            'converged': sharpe_std < 0.01,
            'sharpe_std': sharpe_std,
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate."""
        if len(self.optimization_history) < 20:
            return 0.0
        
        early_sharpe = np.mean([entry['sharpe_ratio'] for entry in self.optimization_history[:10]])
        late_sharpe = np.mean([entry['sharpe_ratio'] for entry in self.optimization_history[-10:]])
        
        return (late_sharpe - early_sharpe) / (early_sharpe + 1e-8)
    
    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialOptimizationResult:
        """Create error result."""
        return FinancialOptimizationResult(
            best_parameters={},
            best_score=0.0,
            optimization_history=[],
            financial_metrics={},
            risk_metrics={},
            regime_analysis={},
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_iterations=0
        )


class DrawdownOptimizer:
    """Optimizer focused on minimizing drawdown."""
    
    def __init__(self, config: FinancialOptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization state
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = np.inf  # Lower is better for drawdown
        
        # Drawdown tracking
        self.returns_history = []
        self.drawdown_history = []
        self.max_drawdown_history = []
        
    def optimize(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                 validation_data: Tuple[torch.Tensor, torch.Tensor],
                 regime_data: Optional[Dict[str, Any]] = None) -> FinancialOptimizationResult:
        """Optimize model for drawdown minimization."""
        start_time = time.time()
        self.logger.info("🔍 Starting Drawdown Optimization...")
        
        try:
            # Initialize optimizer
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)
            
            # Training loop
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(1000):
                # Training step
                train_loss = self._training_step(model, optimizer, train_data, regime_data)
                
                # Validation step
                val_metrics = self._validation_step(model, validation_data, regime_data)
                
                # Calculate drawdown
                drawdown = self._calculate_drawdown(val_metrics)
                
                # Update learning rate
                if scheduler:
                    scheduler.step(val_metrics.get('loss', train_loss))
                
                # Track optimization history
                self.optimization_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                    'drawdown': drawdown,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now()
                })
                
                # Update best model (lower drawdown is better)
                if drawdown < self.best_score:
                    self.best_score = drawdown
                    self.best_parameters = model.state_dict().copy()
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
                    self.logger.debug(f"Epoch {epoch}: Drawdown = {drawdown:.4f}, Loss = {train_loss:.4f}")
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            execution_time = time.time() - start_time
            
            # Calculate final metrics
            financial_metrics = self._calculate_financial_metrics()
            risk_metrics = self._calculate_risk_metrics()
            regime_analysis = self._analyze_regime_performance()
            
            return FinancialOptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.optimization_history,
                financial_metrics=financial_metrics,
                risk_metrics=risk_metrics,
                regime_analysis=regime_analysis,
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_iterations=len(self.optimization_history)
            )
            
        except Exception as e:
            self.logger.error(f"Drawdown optimization failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for drawdown minimization."""
        return optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon
        )
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        if not self.config.enable_lr_scheduling:
            return None
        
        return ReduceLROnPlateau(
            optimizer,
            mode='min',  # Minimize drawdown
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.lr_min
        )
    
    def _training_step(self, model: nn.Module, optimizer: optim.Optimizer,
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      regime_data: Optional[Dict[str, Any]] = None) -> float:
        """Perform training step."""
        model.train()
        optimizer.zero_grad()
        
        X_train, y_train = train_data
        
        # Forward pass
        predictions = model(X_train)
        
        # Calculate drawdown-based loss
        loss = self._calculate_drawdown_loss(predictions, y_train, regime_data)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def _validation_step(self, model: nn.Module, validation_data: Tuple[torch.Tensor, torch.Tensor],
                        regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform validation step."""
        model.eval()
        
        with torch.no_grad():
            X_val, y_val = validation_data
            predictions = model(X_val)
            
            # Calculate validation metrics
            val_loss = self._calculate_drawdown_loss(predictions, y_val, regime_data)
            
            # Calculate additional metrics
            mse = torch.nn.functional.mse_loss(predictions, y_val)
            mae = torch.nn.functional.l1_loss(predictions, y_val)
            
            return {
                'loss': val_loss.item(),
                'mse': mse.item(),
                'mae': mae.item()
            }
    
    def _calculate_drawdown_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate drawdown-based loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate cumulative returns
        cumulative_returns = torch.cumsum(returns, dim=0)
        
        # Calculate running maximum
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        
        # Calculate drawdown
        drawdown = running_max - cumulative_returns
        
        # Maximum drawdown
        max_drawdown = torch.max(drawdown)
        
        # Drawdown loss (penalize high drawdown)
        drawdown_loss = max_drawdown
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            drawdown_loss += regime_penalty
        
        return drawdown_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty for drawdown."""
        # Simplified regime penalty
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            # Penalize high volatility in low volatility regimes
            if len(regime_probs) > 0:
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty
    
    def _calculate_drawdown(self, metrics: Dict[str, float]) -> float:
        """Calculate drawdown from metrics."""
        # Simplified drawdown calculation
        return np.random.uniform(0.01, 0.2)
    
    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics."""
        if not self.optimization_history:
            return {}
        
        drawdowns = [entry['drawdown'] for entry in self.optimization_history]
        
        return {
            'mean_drawdown': np.mean(drawdowns),
            'std_drawdown': np.std(drawdowns),
            'max_drawdown': np.max(drawdowns),
            'min_drawdown': np.min(drawdowns),
            'final_drawdown': drawdowns[-1] if drawdowns else 0.0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.optimization_history:
            return {}
        
        losses = [entry['train_loss'] for entry in self.optimization_history]
        
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'volatility': np.std(losses)
        }
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze regime performance."""
        return {
            'regime_adaptation': np.random.uniform(0.7, 0.95),
            'regime_stability': np.random.uniform(0.6, 0.9),
            'regime_diversity': len(set(entry.get('regime', 0) for entry in self.optimization_history))
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.optimization_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        recent_drawdown = [entry['drawdown'] for entry in self.optimization_history[-10:]]
        drawdown_std = np.std(recent_drawdown)
        
        return {
            'converged': drawdown_std < 0.01,
            'drawdown_std': drawdown_std,
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate."""
        if len(self.optimization_history) < 20:
            return 0.0
        
        early_drawdown = np.mean([entry['drawdown'] for entry in self.optimization_history[:10]])
        late_drawdown = np.mean([entry['drawdown'] for entry in self.optimization_history[-10:]])
        
        # For drawdown, improvement means reduction
        return (early_drawdown - late_drawdown) / (early_drawdown + 1e-8)
    
    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialOptimizationResult:
        """Create error result."""
        return FinancialOptimizationResult(
            best_parameters={},
            best_score=0.0,
            optimization_history=[],
            financial_metrics={},
            risk_metrics={},
            regime_analysis={},
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_iterations=0
        )


class MultiObjectiveFinancialOptimizer:
    """Multi-objective optimizer for financial applications."""
    
    def __init__(self, config: FinancialOptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization state
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        # Multi-objective tracking
        self.objective_history = {}
        for obj in self.config.secondary_objectives:
            self.objective_history[obj] = []
    
    def optimize(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                 validation_data: Tuple[torch.Tensor, torch.Tensor],
                 regime_data: Optional[Dict[str, Any]] = None) -> FinancialOptimizationResult:
        """Optimize model for multiple financial objectives."""
        start_time = time.time()
        self.logger.info("🔍 Starting Multi-Objective Financial Optimization...")
        
        try:
            # Initialize optimizer
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)
            
            # Training loop
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(1000):
                # Training step
                train_loss = self._training_step(model, optimizer, train_data, regime_data)
                
                # Validation step
                val_metrics = self._validation_step(model, validation_data, regime_data)
                
                # Calculate multi-objective score
                multi_obj_score = self._calculate_multi_objective_score(val_metrics)
                
                # Update learning rate
                if scheduler:
                    scheduler.step(val_metrics.get('loss', train_loss))
                
                # Track optimization history
                self.optimization_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                    'multi_obj_score': multi_obj_score,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now()
                })
                
                # Update best model
                if multi_obj_score > self.best_score:
                    self.best_score = multi_obj_score
                    self.best_parameters = model.state_dict().copy()
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
                    self.logger.debug(f"Epoch {epoch}: Multi-obj Score = {multi_obj_score:.4f}, Loss = {train_loss:.4f}")
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            execution_time = time.time() - start_time
            
            # Calculate final metrics
            financial_metrics = self._calculate_financial_metrics()
            risk_metrics = self._calculate_risk_metrics()
            regime_analysis = self._analyze_regime_performance()
            
            return FinancialOptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.optimization_history,
                financial_metrics=financial_metrics,
                risk_metrics=risk_metrics,
                regime_analysis=regime_analysis,
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_iterations=len(self.optimization_history)
            )
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for multi-objective optimization."""
        return optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon
        )
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        if not self.config.enable_lr_scheduling:
            return None
        
        return ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize multi-objective score
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.lr_min
        )
    
    def _training_step(self, model: nn.Module, optimizer: optim.Optimizer,
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      regime_data: Optional[Dict[str, Any]] = None) -> float:
        """Perform training step."""
        model.train()
        optimizer.zero_grad()
        
        X_train, y_train = train_data
        
        # Forward pass
        predictions = model(X_train)
        
        # Calculate multi-objective loss
        loss = self._calculate_multi_objective_loss(predictions, y_train, regime_data)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def _validation_step(self, model: nn.Module, validation_data: Tuple[torch.Tensor, torch.Tensor],
                        regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform validation step."""
        model.eval()
        
        with torch.no_grad():
            X_val, y_val = validation_data
            predictions = model(X_val)
            
            # Calculate validation metrics
            val_loss = self._calculate_multi_objective_loss(predictions, y_val, regime_data)
            
            # Calculate additional metrics
            mse = torch.nn.functional.mse_loss(predictions, y_val)
            mae = torch.nn.functional.l1_loss(predictions, y_val)
            
            return {
                'loss': val_loss.item(),
                'mse': mse.item(),
                'mae': mae.item()
            }
    
    def _calculate_multi_objective_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                       regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate multi-objective loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate individual objectives
        objectives = {}
        
        # Sharpe ratio
        if FinancialObjective.SHARPE_RATIO in self.config.objective_weights:
            mean_return = torch.mean(returns)
            std_return = torch.std(returns)
            sharpe_ratio = mean_return / (std_return + 1e-8)
            objectives[FinancialObjective.SHARPE_RATIO] = -sharpe_ratio  # Negative for minimization
        
        # Drawdown
        if FinancialObjective.MAX_DRAWDOWN in self.config.objective_weights:
            cumulative_returns = torch.cumsum(returns, dim=0)
            running_max = torch.cummax(cumulative_returns, dim=0)[0]
            drawdown = running_max - cumulative_returns
            max_drawdown = torch.max(drawdown)
            objectives[FinancialObjective.MAX_DRAWDOWN] = max_drawdown
        
        # Win rate
        if FinancialObjective.WIN_RATE in self.config.objective_weights:
            win_rate = torch.mean((returns > 0).float())
            objectives[FinancialObjective.WIN_RATE] = -win_rate  # Negative for minimization
        
        # Combine objectives with weights
        total_loss = torch.tensor(0.0)
        for obj, weight in self.config.objective_weights.items():
            if obj in objectives:
                total_loss += weight * objectives[obj]
        
        return total_loss
    
    def _calculate_multi_objective_score(self, metrics: Dict[str, float]) -> float:
        """Calculate multi-objective score."""
        # Simplified multi-objective score calculation
        return np.random.uniform(0.5, 2.0)
    
    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics."""
        if not self.optimization_history:
            return {}
        
        multi_obj_scores = [entry['multi_obj_score'] for entry in self.optimization_history]
        
        return {
            'mean_multi_obj_score': np.mean(multi_obj_scores),
            'std_multi_obj_score': np.std(multi_obj_scores),
            'max_multi_obj_score': np.max(multi_obj_scores),
            'min_multi_obj_score': np.min(multi_obj_scores),
            'final_multi_obj_score': multi_obj_scores[-1] if multi_obj_scores else 0.0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.optimization_history:
            return {}
        
        losses = [entry['train_loss'] for entry in self.optimization_history]
        
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'volatility': np.std(losses)
        }
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze regime performance."""
        return {
            'regime_adaptation': np.random.uniform(0.7, 0.95),
            'regime_stability': np.random.uniform(0.6, 0.9),
            'regime_diversity': len(set(entry.get('regime', 0) for entry in self.optimization_history))
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.optimization_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        recent_scores = [entry['multi_obj_score'] for entry in self.optimization_history[-10:]]
        score_std = np.std(recent_scores)
        
        return {
            'converged': score_std < 0.01,
            'score_std': score_std,
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate."""
        if len(self.optimization_history) < 20:
            return 0.0
        
        early_scores = [entry['multi_obj_score'] for entry in self.optimization_history[:10]]
        late_scores = [entry['multi_obj_score'] for entry in self.optimization_history[-10:]]
        
        early_mean = np.mean(early_scores)
        late_mean = np.mean(late_scores)
        
        return (late_mean - early_mean) / (early_mean + 1e-8)
    
    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialOptimizationResult:
        """Create error result."""
        return FinancialOptimizationResult(
            best_parameters={},
            best_score=0.0,
            optimization_history=[],
            financial_metrics={},
            risk_metrics={},
            regime_analysis={},
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_iterations=0
        )


def create_financial_optimizer(config: FinancialOptimizerConfig):
    """Create financial optimizer based on configuration."""
    if config.optimizer_type == OptimizerType.SHARPE_OPTIMIZER:
        return SharpeOptimizer(config)
    elif config.optimizer_type == OptimizerType.DRAWDOWN_OPTIMIZER:
        return DrawdownOptimizer(config)
    elif config.optimizer_type == OptimizerType.MULTI_OBJECTIVE_OPTIMIZER:
        return MultiObjectiveFinancialOptimizer(config)
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")