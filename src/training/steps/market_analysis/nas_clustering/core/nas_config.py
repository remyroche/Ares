"""
Configuration classes for NAS-driven clustering.

This module provides configuration classes optimized for short-term trading
regime detection with micro-regime detection capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np


class NASArchitectureType(Enum):
    """NAS architecture types for regime detection."""
    VOLATILITY_FOCUSED = "volatility_focused"
    TREND_FOCUSED = "trend_focused"
    VOLUME_FOCUSED = "volume_focused"
    MOMENTUM_FOCUSED = "momentum_focused"
    HYBRID = "hybrid"


class MicroRegimeType(Enum):
    """Micro-regime types for subtle market changes."""
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class NASClusteringConfig:
    """Configuration for NAS-driven clustering."""
    
    # Timeframe configuration
    timeframe: str = "15m"  # Primary timeframe
    micro_timeframe: str = "5m"  # Micro-regime detection timeframe
    
    # Regime configuration
    n_regimes: int = 12  # Target number of regimes (will be optimized if data_driven=True)
    min_regime_duration: int = 15  # Minimum 15 minutes for actionable states
    max_regime_duration: int = 180  # Maximum 3 hours for short-term trading
    data_driven_regimes: bool = True  # Enable data-driven regime count determination
    
    # NAS architecture configuration
    nas_architecture_type: NASArchitectureType = NASArchitectureType.HYBRID
    enable_micro_regime_detection: bool = True
    micro_regime_sensitivity: float = 0.7  # Sensitivity for micro-regime detection
    
    # Feature configuration
    feature_extraction_config: Dict[str, Any] = field(default_factory=dict)
    exclude_complex_features: bool = True  # Exclude polynomial, wavelet features
    include_technical_indicators: bool = True
    include_volume_features: bool = True
    include_volatility_features: bool = True
    include_momentum_features: bool = True
    include_trend_features: bool = True
    
    # NAS optimization configuration
    nas_search_space: Dict[str, Any] = field(default_factory=dict)
    nas_objectives: List[str] = field(default_factory=lambda: [
        "regime_stability", "economic_significance", "trading_viability"
    ])
    nas_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Short-term trading optimization
    short_term_optimization: bool = True
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    regime_transition_cost: float = 0.05
    
    # Micro-regime configuration
    micro_regime_types: List[MicroRegimeType] = field(default_factory=lambda: [
        MicroRegimeType.BREAKOUT,
        MicroRegimeType.CONSOLIDATION,
        MicroRegimeType.REVERSAL,
        MicroRegimeType.ACCELERATION,
        MicroRegimeType.VOLUME_SPIKE,
        MicroRegimeType.VOLATILITY_SPIKE
    ])
    
    # Performance configuration
    enable_hardware_acceleration: bool = True
    enable_matrix_optimization: bool = True
    batch_size: int = 1000
    max_memory_usage: float = 0.8
    
    # Validation configuration
    validation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_regime_stability': 0.6,
        'min_economic_significance': 0.7,
        'min_trading_viability': 0.6,
        'max_regime_volatility': 0.3
    })
    
    @classmethod
    def create_short_term_trading_config(cls) -> 'NASClusteringConfig':
        """Create configuration optimized for short-term trading."""
        return cls(
            timeframe="15m",
            micro_timeframe="5m",
            n_regimes=12,
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            nas_architecture_type=NASArchitectureType.HYBRID,
            enable_micro_regime_detection=True,
            micro_regime_sensitivity=0.7,
            short_term_optimization=True,
            economic_significance_threshold=0.7,
            trading_viability_threshold=0.6,
            regime_transition_cost=0.05,
            feature_extraction_config={
                'volume_features': True,
                'volatility_features': True,
                'momentum_features': True,
                'trend_features': True,
                'technical_indicators': True,
                'exclude_complex_features': True
            },
            nas_search_space={
                'architecture_depth': [3, 5, 7, 9],
                'hidden_units': [32, 64, 128, 256],
                'activation_functions': ['relu', 'tanh', 'swish'],
                'dropout_rates': [0.1, 0.2, 0.3, 0.4],
                'learning_rates': [0.001, 0.01, 0.1]
            },
            nas_objectives=[
                'regime_stability',
                'economic_significance',
                'trading_viability',
                'micro_regime_detection_accuracy'
            ],
            nas_constraints={
                'max_architecture_complexity': 1000,
                'min_regime_persistence': 3,
                'max_regime_transition_frequency': 0.1
            }
        )
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return {
            'timeframe': self.timeframe,
            'micro_timeframe': self.micro_timeframe,
            'exclude_complex_features': self.exclude_complex_features,
            'include_technical_indicators': self.include_technical_indicators,
            'include_volume_features': self.include_volume_features,
            'include_volatility_features': self.include_volatility_features,
            'include_momentum_features': self.include_momentum_features,
            'include_trend_features': self.include_trend_features,
            **self.feature_extraction_config
        }
    
    def get_nas_config(self) -> Dict[str, Any]:
        """Get NAS optimization configuration."""
        return {
            'search_space': self.nas_search_space,
            'objectives': self.nas_objectives,
            'constraints': self.nas_constraints,
            'architecture_type': self.nas_architecture_type.value,
            'short_term_optimization': self.short_term_optimization,
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'regime_transition_cost': self.regime_transition_cost
        }
    
    def get_micro_regime_config(self) -> Dict[str, Any]:
        """Get micro-regime detection configuration."""
        return {
            'enable_micro_regime_detection': self.enable_micro_regime_detection,
            'micro_regime_sensitivity': self.micro_regime_sensitivity,
            'micro_regime_types': [t.value for t in self.micro_regime_types],
            'micro_timeframe': self.micro_timeframe
        }


@dataclass
class NASConfig:
    """Main NAS configuration class."""
    
    clustering_config: NASClusteringConfig = field(default_factory=NASClusteringConfig)
    
    # Pipeline integration
    pipeline_compatibility: bool = True
    output_format: str = "hmm_clustering_compatible"
    
    # Logging and monitoring
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_visualization: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: [
        'execution_time', 'memory_usage', 'regime_detection_accuracy',
        'micro_regime_detection_accuracy', 'economic_significance_score'
    ])
    
    @classmethod
    def create_default_config(cls) -> 'NASConfig':
        """Create default NAS configuration."""
        return cls(
            clustering_config=NASClusteringConfig.create_short_term_trading_config(),
            pipeline_compatibility=True,
            output_format="hmm_clustering_compatible",
            enable_logging=True,
            log_level="INFO",
            enable_metrics=True,
            enable_visualization=True,
            enable_performance_monitoring=True
        )
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.clustering_config, key):
                setattr(self.clustering_config, key, value)
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate timeframe
            if self.clustering_config.timeframe not in ['5m', '15m', '30m']:
                return False
            
            # Validate regime count
            if not (10 <= self.clustering_config.n_regimes <= 15):
                return False
            
            # Validate duration constraints
            if self.clustering_config.min_regime_duration < 15:
                return False
            
            # Validate thresholds
            if not (0 <= self.clustering_config.economic_significance_threshold <= 1):
                return False
            
            if not (0 <= self.clustering_config.trading_viability_threshold <= 1):
                return False
            
            return True
            
        except Exception:
            return False