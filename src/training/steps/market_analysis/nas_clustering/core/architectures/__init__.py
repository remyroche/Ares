"""
Neural Network Architectures for Regime Detection

This module provides specialized neural network architectures optimized for different
types of regime detection in financial time series data.
"""

from .regime_networks import (
    VolatilityRegimeNetwork,
    TrendRegimeNetwork,
    VolumeRegimeNetwork,
    HybridRegimeNetwork,
    RegimeNetworkFactory
)

from .temporal_layers import (
    TemporalConvolutionLayer,
    RegimeLSTMLayer,
    RegimeGRULayer,
    MultiScaleTemporalLayer,
    TemporalAttentionLayer
)

from .attention_mechanisms import (
    RegimeAttention,
    MultiHeadRegimeAttention,
    TemporalAttention,
    CrossRegimeAttention,
    SelfRegimeAttention
)

__all__ = [
    # Regime networks
    'VolatilityRegimeNetwork',
    'TrendRegimeNetwork',
    'VolumeRegimeNetwork',
    'HybridRegimeNetwork',
    'RegimeNetworkFactory',
    
    # Temporal layers
    'TemporalConvolutionLayer',
    'RegimeLSTMLayer',
    'RegimeGRULayer',
    'MultiScaleTemporalLayer',
    'TemporalAttentionLayer',
    
    # Attention mechanisms
    'RegimeAttention',
    'MultiHeadRegimeAttention',
    'TemporalAttention',
    'CrossRegimeAttention',
    'SelfRegimeAttention'
]