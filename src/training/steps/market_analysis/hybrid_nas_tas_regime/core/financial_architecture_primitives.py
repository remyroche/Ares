"""
Financial-Specific Architecture Primitives

This module provides financial domain-specific neural network layers, activations,
and tree components optimized for financial time series and trading applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class FinancialActivationType(Enum):
    """Financial-specific activation functions."""
    VOLATILITY_SENSITIVE = "volatility_sensitive"
    REGIME_AWARE = "regime_aware"
    SHARPE_OPTIMIZED = "sharpe_optimized"
    DRAWDOWN_AWARE = "drawdown_aware"
    MOMENTUM_BASED = "momentum_based"
    MEAN_REVERSION = "mean_reversion"

class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

@dataclass
class FinancialLayerConfig:
    """Configuration for financial-specific layers."""
    regime_aware: bool = True
    volatility_sensitive: bool = True
    momentum_based: bool = True
    risk_adjusted: bool = True
    regime_embedding_dim: int = 16
    volatility_window: int = 20
    momentum_window: int = 10

class VolatilitySensitiveActivation(nn.Module):
    """Activation function that adapts based on market volatility."""

    def __init__(self, base_activation: str = "relu", volatility_threshold: float = 0.02):
        super().__init__()
        self.base_activation = base_activation
        self.volatility_threshold = volatility_threshold
        self.volatility_estimator = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with volatility adaptation."""
        if volatility is None:
            # Estimate volatility from input
            volatility = torch.std(x, dim=-1, keepdim=True)

        # Apply base activation
        if self.base_activation == "relu":
            base_output = F.relu(x)
        elif self.base_activation == "tanh":
            base_output = torch.tanh(x)
        elif self.base_activation == "sigmoid":
            base_output = torch.sigmoid(x)
        else:
            base_output = x

        # Adapt based on volatility
        volatility_factor = torch.sigmoid(self.volatility_estimator(volatility))
        high_vol_mask = volatility > self.volatility_threshold

        # Reduce sensitivity in high volatility
        adapted_output = torch.where(
            high_vol_mask,
            base_output * (1 - volatility_factor * 0.3),
            base_output * (1 + volatility_factor * 0.1)
        )

        return adapted_output

class RegimeAwareActivation(nn.Module):
    """Activation function that adapts based on market regime."""

    def __init__(self, base_activation: str = "relu", n_regimes: int = 4):
        super().__init__()
        self.base_activation = base_activation
        self.n_regimes = n_regimes
        self.regime_embeddings = nn.Embedding(n_regimes, 1)
        self.regime_weights = nn.Parameter(torch.ones(n_regimes))

    def forward(self, x: torch.Tensor, regime_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with regime adaptation."""
        # Apply base activation
        if self.base_activation == "relu":
            base_output = F.relu(x)
        elif self.base_activation == "tanh":
            base_output = torch.tanh(x)
        elif self.base_activation == "sigmoid":
            base_output = torch.sigmoid(x)
        else:
            base_output = x

        if regime_probs is not None:
            # Calculate regime-weighted output
            regime_weights = torch.softmax(self.regime_weights, dim=0)
            weighted_regime_probs = regime_probs * regime_weights.unsqueeze(0)
            regime_factor = torch.sum(weighted_regime_probs, dim=-1, keepdim=True)

            # Adapt output based on regime
            adapted_output = base_output * (0.8 + 0.4 * regime_factor)
        else:
            adapted_output = base_output

        return adapted_output

class SharpeOptimizedActivation(nn.Module):
    """Activation function optimized for Sharpe ratio maximization."""

    def __init__(self, risk_free_rate: float = 0.02):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.return_estimator = nn.Linear(1, 1)
        self.volatility_estimator = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass optimized for Sharpe ratio."""
        # Apply base activation
        base_output = torch.tanh(x)

        if returns is not None:
            # Calculate Sharpe ratio components
            expected_return = self.return_estimator(returns.unsqueeze(-1))
            volatility = torch.abs(self.volatility_estimator(returns.unsqueeze(-1)))

            # Sharpe ratio = (return - risk_free_rate) / volatility
            sharpe_ratio = (expected_return - self.risk_free_rate) / (volatility + 1e-8)

            # Adapt output based on Sharpe ratio
            sharpe_factor = torch.sigmoid(sharpe_ratio)
            adapted_output = base_output * sharpe_factor
        else:
            adapted_output = base_output

        return adapted_output

class DrawdownAwareActivation(nn.Module):
    """Activation function that considers drawdown risk."""

    def __init__(self, max_drawdown_threshold: float = 0.1):
        super().__init__()
        self.max_drawdown_threshold = max_drawdown_threshold
        self.drawdown_estimator = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, cumulative_returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with drawdown awareness."""
        # Apply base activation
        base_output = torch.tanh(x)

        if cumulative_returns is not None:
            # Calculate drawdown
            peak = torch.cummax(cumulative_returns, dim=-1)[0]
            drawdown = (peak - cumulative_returns) / (peak + 1e-8)
            max_drawdown = torch.max(drawdown, dim=-1, keepdim=True)[0]

            # Adapt output based on drawdown
            drawdown_factor = torch.sigmoid(self.drawdown_estimator(max_drawdown))
            high_drawdown_mask = max_drawdown > self.max_drawdown_threshold

            adapted_output = torch.where(
                high_drawdown_mask,
                base_output * (1 - drawdown_factor * 0.5),  # Reduce output in high drawdown
                base_output * (1 + drawdown_factor * 0.1)   # Increase output in low drawdown
            )
        else:
            adapted_output = base_output

        return adapted_output

class MomentumBasedActivation(nn.Module):
    """Activation function that adapts based on momentum."""

    def __init__(self, momentum_window: int = 10):
        super().__init__()
        self.momentum_window = momentum_window
        self.momentum_estimator = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, momentum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with momentum adaptation."""
        # Apply base activation
        base_output = torch.tanh(x)

        if momentum is not None:
            # Calculate momentum factor
            momentum_factor = torch.sigmoid(self.momentum_estimator(momentum.unsqueeze(-1)))

            # Adapt output based on momentum
            positive_momentum_mask = momentum > 0
            adapted_output = torch.where(
                positive_momentum_mask,
                base_output * (1 + momentum_factor * 0.2),  # Amplify positive momentum
                base_output * (1 - momentum_factor * 0.1)    # Dampen negative momentum
            )
        else:
            adapted_output = base_output

        return adapted_output

class MeanReversionActivation(nn.Module):
    """Activation function for mean reversion strategies."""

    def __init__(self, reversion_threshold: float = 2.0):
        super().__init__()
        self.reversion_threshold = reversion_threshold
        self.zscore_estimator = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, zscore: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with mean reversion awareness."""
        # Apply base activation
        base_output = torch.tanh(x)

        if zscore is not None:
            # Calculate mean reversion factor
            abs_zscore = torch.abs(zscore)
            reversion_factor = torch.sigmoid(self.zscore_estimator(abs_zscore.unsqueeze(-1)))

            # Adapt output based on z-score
            extreme_zscore_mask = abs_zscore > self.reversion_threshold
            adapted_output = torch.where(
                extreme_zscore_mask,
                base_output * (1 + reversion_factor * 0.3),  # Amplify extreme values
                base_output * (1 - reversion_factor * 0.1)   # Dampen normal values
            )
        else:
            adapted_output = base_output

        return adapted_output

class RegimeAwareLinear(nn.Module):
    """Linear layer that adapts weights based on market regime."""

    def __init__(self, in_features: int, out_features: int, n_regimes: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_regimes = n_regimes

        # Base weights
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.randn(out_features))

        # Regime-specific adjustments
        self.regime_weights = nn.Parameter(torch.randn(n_regimes, out_features, in_features))
        self.regime_biases = nn.Parameter(torch.randn(n_regimes, out_features))

        # Regime attention
        self.regime_attention = nn.Linear(in_features, n_regimes)

    def forward(self, x: torch.Tensor, regime_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with regime-aware weights."""
        batch_size = x.size(0)

        if regime_probs is not None:
            # Calculate regime-weighted weights and biases
            regime_weights = torch.sum(
                regime_probs.unsqueeze(-1).unsqueeze(-1) * self.regime_weights.unsqueeze(0),
                dim=1
            )
            regime_biases = torch.sum(
                regime_probs.unsqueeze(-1) * self.regime_biases.unsqueeze(0),
                dim=1
            )

            # Combine base and regime-specific weights
            final_weights = self.base_weight + regime_weights
            final_bias = self.base_bias + regime_biases
        else:
            # Use base weights only
            final_weights = self.base_weight
            final_bias = self.base_bias

        # Apply linear transformation
        output = F.linear(x, final_weights, final_bias)
        return output

class VolatilitySensitiveLinear(nn.Module):
    """Linear layer that adapts based on volatility."""

    def __init__(self, in_features: int, out_features: int, volatility_window: int = 20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.volatility_window = volatility_window

        # Base weights
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.randn(out_features))

        # Volatility adaptation
        self.volatility_adapter = nn.Linear(1, out_features * in_features)
        self.volatility_bias_adapter = nn.Linear(1, out_features)

    def forward(self, x: torch.Tensor, volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with volatility adaptation."""
        if volatility is None:
            # Estimate volatility from input
            volatility = torch.std(x, dim=-1, keepdim=True)

        # Calculate volatility adaptation
        vol_adaptation = self.volatility_adapter(volatility)
        vol_bias_adaptation = self.volatility_bias_adapter(volatility)

        # Reshape adaptation to weight matrix
        vol_adaptation = vol_adaptation.view(-1, self.out_features, self.in_features)

        # Apply adapted weights
        adapted_weights = self.base_weight + vol_adaptation
        adapted_bias = self.base_bias + vol_bias_adaptation

        # Linear transformation
        output = F.linear(x, adapted_weights, adapted_bias)
        return output

class FinancialLSTM(nn.Module):
    """LSTM layer optimized for financial time series."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 regime_aware: bool = True, volatility_sensitive: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.regime_aware = regime_aware
        self.volatility_sensitive = volatility_sensitive

        # Base LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Financial adaptations
        if regime_aware:
            self.regime_gate = nn.Linear(input_size, hidden_size)
            self.regime_attention = nn.Linear(hidden_size, 1)

        if volatility_sensitive:
            self.volatility_gate = nn.Linear(input_size, hidden_size)
            self.volatility_attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, regime_probs: Optional[torch.Tensor] = None,
                volatility: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with financial adaptations."""
        # Base LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply regime awareness
        if self.regime_aware and regime_probs is not None:
            regime_gate = torch.sigmoid(self.regime_gate(x))
            regime_attention = torch.softmax(self.regime_attention(lstm_out), dim=1)
            lstm_out = lstm_out * regime_gate * regime_attention

        # Apply volatility sensitivity
        if self.volatility_sensitive and volatility is not None:
            vol_gate = torch.sigmoid(self.volatility_gate(x))
            vol_attention = torch.softmax(self.volatility_attention(lstm_out), dim=1)
            lstm_out = lstm_out * vol_gate * vol_attention

        return lstm_out, (h_n, c_n)

class FinancialTransformer(nn.Module):
    """Transformer layer optimized for financial time series."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 1,
                 regime_aware: bool = True, volatility_sensitive: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.regime_aware = regime_aware
        self.volatility_sensitive = volatility_sensitive

        # Base transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Financial adaptations
        if regime_aware:
            self.regime_embedding = nn.Linear(1, d_model)
            self.regime_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        if volatility_sensitive:
            self.volatility_embedding = nn.Linear(1, d_model)
            self.volatility_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, x: torch.Tensor, regime_probs: Optional[torch.Tensor] = None,
                volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with financial adaptations."""
        # Base transformer forward pass
        output = self.transformer(x)

        # Apply regime awareness
        if self.regime_aware and regime_probs is not None:
            regime_embedding = self.regime_embedding(regime_probs.unsqueeze(-1))
            regime_out, _ = self.regime_attention(output, regime_embedding, regime_embedding)
            output = output + regime_out

        # Apply volatility sensitivity
        if self.volatility_sensitive and volatility is not None:
            vol_embedding = self.volatility_embedding(volatility.unsqueeze(-1))
            vol_out, _ = self.volatility_attention(output, vol_embedding, vol_embedding)
            output = output + vol_out

        return output

class FinancialTreePrimitive:
    """Financial-specific tree primitive for TAS."""

    def __init__(self, feature_type: str, regime_aware: bool = True):
        self.feature_type = feature_type
        self.regime_aware = regime_aware
        self.regime_weights = {}

    def evaluate(self, data: np.ndarray, regime: Optional[int] = None) -> float:
        """Evaluate tree primitive with regime awareness."""
        if self.feature_type == "volatility_ratio":
            return self._volatility_ratio(data, regime)
        elif self.feature_type == "momentum_score":
            return self._momentum_score(data, regime)
        elif self.feature_type == "mean_reversion":
            return self._mean_reversion(data, regime)
        elif self.feature_type == "regime_stability":
            return self._regime_stability(data, regime)
        else:
            return 0.0

    def _volatility_ratio(self, data: np.ndarray, regime: Optional[int] = None) -> float:
        """Calculate volatility ratio with regime awareness."""
        if len(data) < 2:
            return 0.0

        returns = np.diff(data) / data[:-1]
        current_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        historical_vol = np.std(returns) if len(returns) > 10 else current_vol

        ratio = current_vol / (historical_vol + 1e-8)

        # Regime adjustment
        if regime is not None and regime in self.regime_weights:
            ratio *= self.regime_weights[regime]

        return ratio

    def _momentum_score(self, data: np.ndarray, regime: Optional[int] = None) -> float:
        """Calculate momentum score with regime awareness."""
        if len(data) < 2:
            return 0.0

        returns = np.diff(data) / data[:-1]
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)

        # Regime adjustment
        if regime is not None and regime in self.regime_weights:
            momentum *= self.regime_weights[regime]

        return momentum

    def _mean_reversion(self, data: np.ndarray, regime: Optional[int] = None) -> float:
        """Calculate mean reversion score with regime awareness."""
        if len(data) < 3:
            return 0.0

        mean_price = np.mean(data)
        current_price = data[-1]
        std_price = np.std(data)

        zscore = (current_price - mean_price) / (std_price + 1e-8)

        # Regime adjustment
        if regime is not None and regime in self.regime_weights:
            zscore *= self.regime_weights[regime]

        return zscore

    def _regime_stability(self, data: np.ndarray, regime: Optional[int] = None) -> float:
        """Calculate regime stability score."""
        if len(data) < 10:
            return 0.0

        # Calculate regime consistency
        returns = np.diff(data) / data[:-1]
        positive_returns = np.sum(returns > 0)
        total_returns = len(returns)

        stability = max(positive_returns, total_returns - positive_returns) / total_returns

        # Regime adjustment
        if regime is not None and regime in self.regime_weights:
            stability *= self.regime_weights[regime]

        return stability

def create_financial_activation(activation_type: FinancialActivationType, **kwargs) -> nn.Module:
    """Create financial-specific activation function."""
    if activation_type == FinancialActivationType.VOLATILITY_SENSITIVE:
        return VolatilitySensitiveActivation(**kwargs)
    elif activation_type == FinancialActivationType.REGIME_AWARE:
        return RegimeAwareActivation(**kwargs)
    elif activation_type == FinancialActivationType.SHARPE_OPTIMIZED:
        return SharpeOptimizedActivation(**kwargs)
    elif activation_type == FinancialActivationType.DRAWDOWN_AWARE:
        return DrawdownAwareActivation(**kwargs)
    elif activation_type == FinancialActivationType.MOMENTUM_BASED:
        return MomentumBasedActivation(**kwargs)
    elif activation_type == FinancialActivationType.MEAN_REVERSION:
        return MeanReversionActivation(**kwargs)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

def create_financial_layer(layer_type: str, in_features: int, out_features: int, **kwargs) -> nn.Module:
    """Create financial-specific layer."""
    if layer_type == "regime_aware_linear":
        return RegimeAwareLinear(in_features, out_features, **kwargs)
    elif layer_type == "volatility_sensitive_linear":
        return VolatilitySensitiveLinear(in_features, out_features, **kwargs)
    elif layer_type == "financial_lstm":
        return FinancialLSTM(in_features, out_features, **kwargs)
    elif layer_type == "financial_transformer":
        return FinancialTransformer(in_features, out_features, **kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

def create_financial_tree_primitive(feature_type: str, **kwargs) -> FinancialTreePrimitive:
    """Create financial-specific tree primitive."""
    return FinancialTreePrimitive(feature_type, **kwargs)
