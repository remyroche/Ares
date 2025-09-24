"""
Financial-Specific Loss Functions

This module provides loss functions specifically designed for financial trading
applications, including Sharpe ratio loss, drawdown loss, and risk-adjusted losses.
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
import warnings

from .financial_architecture_primitives import RegimeType, FinancialActivationType

logger = logging.getLogger(__name__)


class FinancialLossType(Enum):
    """Types of financial loss functions."""
    SHARPE_LOSS = "sharpe_loss"
    DRAWDOWN_LOSS = "drawdown_loss"
    SORTINO_LOSS = "sortino_loss"
    CALMAR_LOSS = "calmar_loss"
    VAR_LOSS = "var_loss"
    CVAR_LOSS = "cvar_loss"
    WIN_RATE_LOSS = "win_rate_loss"
    PROFIT_FACTOR_LOSS = "profit_factor_loss"
    REGIME_AWARE_LOSS = "regime_aware_loss"
    VOLATILITY_TARGETING_LOSS = "volatility_targeting_loss"
    MOMENTUM_LOSS = "momentum_loss"
    MEAN_REVERSION_LOSS = "mean_reversion_loss"
    RISK_PARITY_LOSS = "risk_parity_loss"
    MAXIMUM_DRAWDOWN_LOSS = "maximum_drawdown_loss"
    ULTIMATE_DRAWDOWN_LOSS = "ultimate_drawdown_loss"


@dataclass
class FinancialLossConfig:
    """Configuration for financial loss functions."""
    # Base loss settings
    loss_type: FinancialLossType = FinancialLossType.SHARPE_LOSS
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    target_volatility: float = 0.15
    max_drawdown_threshold: float = 0.1
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_weights: Dict[RegimeType, float] = field(default_factory=lambda: {
        RegimeType.BULL: 1.0,
        RegimeType.BEAR: 0.8,
        RegimeType.SIDEWAYS: 0.9,
        RegimeType.HIGH_VOLATILITY: 0.7,
        RegimeType.LOW_VOLATILITY: 1.1,
        RegimeType.TRENDING: 1.0,
        RegimeType.MEAN_REVERTING: 0.9
    })
    
    # Volatility targeting
    enable_volatility_targeting: bool = True
    volatility_window: int = 20
    volatility_scaling_factor: float = 1.0
    
    # Risk management
    max_risk_per_trade: float = 0.02
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.10
    
    # Loss function parameters
    sharpe_weight: float = 1.0
    drawdown_weight: float = 0.5
    volatility_weight: float = 0.3
    regime_weight: float = 0.2
    
    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    regime_regularization: float = 0.0


class SharpeLoss(nn.Module):
    """Sharpe ratio-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate Sharpe ratio loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate Sharpe ratio components
        mean_return = torch.mean(returns)
        std_return = torch.std(returns)
        
        # Sharpe ratio
        sharpe_ratio = (mean_return - self.risk_free_rate) / (std_return + 1e-8)
        
        # Convert to loss (negative Sharpe ratio)
        sharpe_loss = -sharpe_ratio
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            sharpe_loss += regime_penalty
        
        # Add volatility targeting
        if self.config.enable_volatility_targeting:
            vol_penalty = self._calculate_volatility_penalty(returns)
            sharpe_loss += vol_penalty
        
        return sharpe_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Calculate regime-weighted penalty
                regime_weights = torch.tensor([
                    self.config.regime_weights.get(RegimeType.BULL, 1.0),
                    self.config.regime_weights.get(RegimeType.BEAR, 1.0),
                    self.config.regime_weights.get(RegimeType.SIDEWAYS, 1.0),
                    self.config.regime_weights.get(RegimeType.HIGH_VOLATILITY, 1.0)
                ])
                
                # Weight penalty by regime probabilities
                regime_penalty = torch.std(returns) * torch.mean(regime_weights) * 0.1
        
        return regime_penalty
    
    def _calculate_volatility_penalty(self, returns: torch.Tensor) -> torch.Tensor:
        """Calculate volatility targeting penalty."""
        current_volatility = torch.std(returns)
        target_volatility = self.config.target_volatility
        
        # Penalize deviation from target volatility
        vol_deviation = torch.abs(current_volatility - target_volatility)
        vol_penalty = vol_deviation * self.config.volatility_weight
        
        return vol_penalty


class DrawdownLoss(nn.Module):
    """Drawdown-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate drawdown loss."""
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
        
        # Drawdown loss
        drawdown_loss = max_drawdown * self.config.drawdown_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            drawdown_loss += regime_penalty
        
        # Add threshold penalty
        if max_drawdown > self.config.max_drawdown_threshold:
            threshold_penalty = (max_drawdown - self.config.max_drawdown_threshold) * 2.0
            drawdown_loss += threshold_penalty
        
        return drawdown_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty for drawdown."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize high volatility in low volatility regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class SortinoLoss(nn.Module):
    """Sortino ratio-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate Sortino ratio loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate Sortino ratio components
        mean_return = torch.mean(returns)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = torch.where(returns < 0, returns, torch.tensor(0.0))
        downside_deviation = torch.sqrt(torch.mean(negative_returns ** 2))
        
        # Sortino ratio
        sortino_ratio = (mean_return - self.risk_free_rate) / (downside_deviation + 1e-8)
        
        # Convert to loss (negative Sortino ratio)
        sortino_loss = -sortino_ratio
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            sortino_loss += regime_penalty
        
        return sortino_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize high downside risk in stable regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class VaRLoss(nn.Module):
    """Value at Risk (VaR) based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        self.confidence_level = config.confidence_level
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate VaR loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate VaR
        var_percentile = (1 - self.confidence_level) * 100
        var = torch.quantile(returns, var_percentile / 100)
        
        # VaR loss (penalize high VaR)
        var_loss = -var * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            var_loss += regime_penalty
        
        return var_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize high VaR in low volatility regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class CVaRLoss(nn.Module):
    """Conditional Value at Risk (CVaR) based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        self.confidence_level = config.confidence_level
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate CVaR loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate VaR
        var_percentile = (1 - self.confidence_level) * 100
        var = torch.quantile(returns, var_percentile / 100)
        
        # Calculate CVaR (expected shortfall)
        tail_returns = torch.where(returns <= var, returns, torch.tensor(0.0))
        cvar = torch.mean(tail_returns)
        
        # CVaR loss (penalize high CVaR)
        cvar_loss = -cvar * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            cvar_loss += regime_penalty
        
        return cvar_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize high CVaR in low volatility regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class WinRateLoss(nn.Module):
    """Win rate-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate win rate loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate win rate
        win_rate = torch.mean((returns > 0).float())
        
        # Win rate loss (penalize low win rate)
        win_rate_loss = (1.0 - win_rate) * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            win_rate_loss += regime_penalty
        
        return win_rate_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize low win rate in favorable regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class ProfitFactorLoss(nn.Module):
    """Profit factor-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate profit factor loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate profit factor
        positive_returns = torch.where(returns > 0, returns, torch.tensor(0.0))
        negative_returns = torch.where(returns < 0, -returns, torch.tensor(0.0))
        
        total_profit = torch.sum(positive_returns)
        total_loss = torch.sum(negative_returns)
        
        # Profit factor
        profit_factor = total_profit / (total_loss + 1e-8)
        
        # Profit factor loss (penalize low profit factor)
        profit_factor_loss = (1.0 / (profit_factor + 1e-8)) * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            profit_factor_loss += regime_penalty
        
        return profit_factor_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize low profit factor in favorable regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class RegimeAwareLoss(nn.Module):
    """Regime-aware loss function that adapts based on market regimes."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        self.base_loss = SharpeLoss(config)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate regime-aware loss."""
        # Calculate base loss
        base_loss = self.base_loss(predictions, targets, regime_data)
        
        # Add regime-specific adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_adjustment = self._calculate_regime_adjustment(predictions, targets, regime_data)
            base_loss += regime_adjustment
        
        return base_loss
    
    def _calculate_regime_adjustment(self, predictions: torch.Tensor, targets: torch.Tensor,
                                   regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-specific adjustment."""
        regime_adjustment = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Calculate regime-specific weights
                regime_weights = torch.tensor([
                    self.config.regime_weights.get(RegimeType.BULL, 1.0),
                    self.config.regime_weights.get(RegimeType.BEAR, 1.0),
                    self.config.regime_weights.get(RegimeType.SIDEWAYS, 1.0),
                    self.config.regime_weights.get(RegimeType.HIGH_VOLATILITY, 1.0)
                ])
                
                # Weight adjustment by regime probabilities
                regime_adjustment = torch.mean(regime_weights) * 0.1
        
        return regime_adjustment


class VolatilityTargetingLoss(nn.Module):
    """Volatility targeting loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        self.target_volatility = config.target_volatility
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate volatility targeting loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate current volatility
        current_volatility = torch.std(returns)
        
        # Volatility targeting loss
        vol_deviation = torch.abs(current_volatility - self.target_volatility)
        vol_loss = vol_deviation * self.config.volatility_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            vol_loss += regime_penalty
        
        return vol_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize volatility deviation in stable regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class MomentumLoss(nn.Module):
    """Momentum-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate momentum loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate momentum
        momentum = torch.mean(returns)
        
        # Momentum loss (penalize negative momentum)
        momentum_loss = -momentum * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            momentum_loss += regime_penalty
        
        return momentum_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize negative momentum in trending regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class MeanReversionLoss(nn.Module):
    """Mean reversion-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate mean reversion loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate mean reversion (negative autocorrelation)
        if len(returns) > 1:
            autocorr = torch.corrcoef(torch.stack([returns[:-1], returns[1:]]))[0, 1]
            mean_reversion = -autocorr  # Negative autocorrelation is good for mean reversion
        else:
            mean_reversion = torch.tensor(0.0)
        
        # Mean reversion loss (penalize positive autocorrelation)
        mean_reversion_loss = -mean_reversion * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            mean_reversion_loss += regime_penalty
        
        return mean_reversion_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize positive autocorrelation in mean-reverting regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class RiskParityLoss(nn.Module):
    """Risk parity-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate risk parity loss."""
        # Calculate returns
        returns = predictions - targets
        
        # Calculate risk contribution (simplified)
        risk_contribution = torch.var(returns)
        
        # Risk parity loss (penalize high risk contribution)
        risk_parity_loss = risk_contribution * self.config.sharpe_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            risk_parity_loss += regime_penalty
        
        return risk_parity_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize high risk in low volatility regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


class MaximumDrawdownLoss(nn.Module):
    """Maximum drawdown-based loss function."""
    
    def __init__(self, config: FinancialLossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate maximum drawdown loss."""
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
        
        # Maximum drawdown loss
        max_drawdown_loss = max_drawdown * self.config.drawdown_weight
        
        # Add regime-aware adjustments
        if regime_data and self.config.enable_regime_awareness:
            regime_penalty = self._calculate_regime_penalty(returns, regime_data)
            max_drawdown_loss += regime_penalty
        
        return max_drawdown_loss
    
    def _calculate_regime_penalty(self, returns: torch.Tensor, regime_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate regime-aware penalty."""
        regime_penalty = torch.tensor(0.0)
        
        if 'regime_probabilities' in regime_data:
            regime_probs = regime_data['regime_probabilities']
            if len(regime_probs) > 0:
                # Penalize high maximum drawdown in stable regimes
                regime_penalty = torch.std(returns) * 0.1
        
        return regime_penalty


def create_financial_loss_function(config: FinancialLossConfig) -> nn.Module:
    """Create financial loss function based on configuration."""
    if config.loss_type == FinancialLossType.SHARPE_LOSS:
        return SharpeLoss(config)
    elif config.loss_type == FinancialLossType.DRAWDOWN_LOSS:
        return DrawdownLoss(config)
    elif config.loss_type == FinancialLossType.SORTINO_LOSS:
        return SortinoLoss(config)
    elif config.loss_type == FinancialLossType.VAR_LOSS:
        return VaRLoss(config)
    elif config.loss_type == FinancialLossType.CVAR_LOSS:
        return CVaRLoss(config)
    elif config.loss_type == FinancialLossType.WIN_RATE_LOSS:
        return WinRateLoss(config)
    elif config.loss_type == FinancialLossType.PROFIT_FACTOR_LOSS:
        return ProfitFactorLoss(config)
    elif config.loss_type == FinancialLossType.REGIME_AWARE_LOSS:
        return RegimeAwareLoss(config)
    elif config.loss_type == FinancialLossType.VOLATILITY_TARGETING_LOSS:
        return VolatilityTargetingLoss(config)
    elif config.loss_type == FinancialLossType.MOMENTUM_LOSS:
        return MomentumLoss(config)
    elif config.loss_type == FinancialLossType.MEAN_REVERSION_LOSS:
        return MeanReversionLoss(config)
    elif config.loss_type == FinancialLossType.RISK_PARITY_LOSS:
        return RiskParityLoss(config)
    elif config.loss_type == FinancialLossType.MAXIMUM_DRAWDOWN_LOSS:
        return MaximumDrawdownLoss(config)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


class CompositeFinancialLoss(nn.Module):
    """Composite financial loss function combining multiple objectives."""
    
    def __init__(self, config: FinancialLossConfig, loss_weights: Dict[FinancialLossType, float]):
        super().__init__()
        self.config = config
        self.loss_weights = loss_weights
        
        # Create individual loss functions
        self.loss_functions = {}
        for loss_type, weight in loss_weights.items():
            if weight > 0:
                loss_config = FinancialLossConfig(
                    loss_type=loss_type,
                    risk_free_rate=config.risk_free_rate,
                    confidence_level=config.confidence_level,
                    target_volatility=config.target_volatility,
                    enable_regime_awareness=config.enable_regime_awareness,
                    regime_weights=config.regime_weights
                )
                self.loss_functions[loss_type] = create_financial_loss_function(loss_config)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                regime_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Calculate composite loss."""
        total_loss = torch.tensor(0.0)
        
        for loss_type, loss_fn in self.loss_functions.items():
            weight = self.loss_weights[loss_type]
            loss = loss_fn(predictions, targets, regime_data)
            total_loss += weight * loss
        
        return total_loss


def create_composite_financial_loss(config: FinancialLossConfig,
                                   loss_weights: Dict[FinancialLossType, float]) -> CompositeFinancialLoss:
    """Create composite financial loss function."""
    return CompositeFinancialLoss(config, loss_weights)