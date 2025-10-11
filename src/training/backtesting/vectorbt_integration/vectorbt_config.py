"""
VectorBT Configuration

Configuration classes and utilities for VectorBT integration.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VectorBTMode(Enum):
    """VectorBT execution modes."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"

class PortfolioMode(Enum):
    """Portfolio simulation modes."""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    CUSTOM = "custom"

@dataclass
class VectorBTConfig:
    """Configuration for VectorBT integration."""
    
    # Basic configuration
    mode: VectorBTMode = VectorBTMode.AUTO
    portfolio_mode: PortfolioMode = PortfolioMode.ADVANCED
    
    # Performance settings
    enable_gpu: bool = True
    enable_parallel: bool = True
    chunk_size: int = 10000
    memory_limit_mb: int = 1024
    
    # Portfolio settings
    init_cash: float = 100000.0
    fees: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    freq: str = 'D'  # Daily frequency
    
    # Risk management
    max_position_size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Technical indicators
    enable_indicators: bool = True
    indicator_cache_size: int = 1000
    
    # Simulation settings
    n_simulations: int = 1000
    simulation_horizon: int = 252
    confidence_level: float = 0.95
    
    # Visualization
    enable_plots: bool = True
    plot_style: str = 'darkgrid'
    figure_size: Tuple[int, int] = (12, 8)
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.init_cash <= 0:
            raise ValueError("Initial cash must be positive")
        
        if not 0 <= self.fees <= 1:
            raise ValueError("Fees must be between 0 and 1")
        
        if not 0 <= self.slippage <= 1:
            raise ValueError("Slippage must be between 0 and 1")
        
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        
        if self.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mode': self.mode.value,
            'portfolio_mode': self.portfolio_mode.value,
            'enable_gpu': self.enable_gpu,
            'enable_parallel': self.enable_parallel,
            'chunk_size': self.chunk_size,
            'memory_limit_mb': self.memory_limit_mb,
            'init_cash': self.init_cash,
            'fees': self.fees,
            'slippage': self.slippage,
            'freq': self.freq,
            'max_position_size': self.max_position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'enable_indicators': self.enable_indicators,
            'indicator_cache_size': self.indicator_cache_size,
            'n_simulations': self.n_simulations,
            'simulation_horizon': self.simulation_horizon,
            'confidence_level': self.confidence_level,
            'enable_plots': self.enable_plots,
            'plot_style': self.plot_style,
            'figure_size': self.figure_size,
            'custom_params': self.custom_params
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VectorBTConfig':
        """Create configuration from dictionary."""
        # Convert enum values back to enums
        if 'mode' in config_dict:
            config_dict['mode'] = VectorBTMode(config_dict['mode'])
        if 'portfolio_mode' in config_dict:
            config_dict['portfolio_mode'] = PortfolioMode(config_dict['portfolio_mode'])
        
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'VectorBTConfig':
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_params[key] = value
        
        self._validate_config()
        return self

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    
    # Moving averages
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    
    # Oscillators
    rsi_period: int = 14
    stoch_periods: Tuple[int, int, int] = (14, 3, 3)
    williams_r_period: int = 14
    
    # Volatility
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Momentum
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volume
    obv_enabled: bool = True
    ad_line_enabled: bool = True
    
    # Custom indicators
    custom_indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sma_periods': self.sma_periods,
            'ema_periods': self.ema_periods,
            'rsi_period': self.rsi_period,
            'stoch_periods': self.stoch_periods,
            'williams_r_period': self.williams_r_period,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'atr_period': self.atr_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'obv_enabled': self.obv_enabled,
            'ad_line_enabled': self.ad_line_enabled,
            'custom_indicators': self.custom_indicators
        }

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    
    # Simulation parameters
    n_simulations: int = 1000
    simulation_horizon: int = 252
    confidence_level: float = 0.95
    
    # Distribution parameters
    distribution: str = 'normal'  # 'normal', 't', 'skewed_t'
    bootstrap_sample_size: float = 0.8
    
    # Risk metrics
    var_confidence: float = 0.05
    expected_shortfall_confidence: float = 0.01
    
    # Performance settings
    enable_parallel: bool = True
    chunk_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_simulations': self.n_simulations,
            'simulation_horizon': self.simulation_horizon,
            'confidence_level': self.confidence_level,
            'distribution': self.distribution,
            'bootstrap_sample_size': self.bootstrap_sample_size,
            'var_confidence': self.var_confidence,
            'expected_shortfall_confidence': self.expected_shortfall_confidence,
            'enable_parallel': self.enable_parallel,
            'chunk_size': self.chunk_size
        }

def create_default_config() -> VectorBTConfig:
    """Create default VectorBT configuration."""
    return VectorBTConfig()

def create_high_performance_config() -> VectorBTConfig:
    """Create high-performance VectorBT configuration."""
    return VectorBTConfig(
        mode=VectorBTMode.GPU,
        enable_gpu=True,
        enable_parallel=True,
        chunk_size=50000,
        memory_limit_mb=4096,
        n_simulations=10000
    )

def create_memory_efficient_config() -> VectorBTConfig:
    """Create memory-efficient VectorBT configuration."""
    return VectorBTConfig(
        mode=VectorBTMode.CPU,
        enable_gpu=False,
        enable_parallel=False,
        chunk_size=1000,
        memory_limit_mb=512,
        n_simulations=1000
    )