"""
Regime Feature Integration Module

This module provides regime-aware feature integration capabilities for the feature generation system.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..core.feature_generator import FeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ..base_calculations.base_calculator import BaseCalculationType


class RegimeType(Enum):
    """Enumeration of regime types."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class RegimeFeatureConfig:
    """Configuration for regime feature integration."""
    
    name: str = "regime_feature_integration"
    enable_regime_detection: bool = True
    regime_threshold: float = 0.5
    lookback_period: int = 20
    min_samples_per_regime: int = 50
    enable_adaptive_features: bool = True
    enable_regime_transitions: bool = True
    vectorbt_threshold: int = 1000
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.regime_threshold < 0 or self.regime_threshold > 1:
            raise ValueError("regime_threshold must be between 0 and 1")
        if self.lookback_period < 5:
            raise ValueError("lookback_period must be at least 5")


class RegimeFeatureIntegration(FeatureGenerator):
    """Regime-aware feature integration generator."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        """Initialize regime feature integration."""
        if config is None:
            config = RegimeFeatureConfig()
            
        feature_config = FeatureConfig(
            name=config.name,
            category=FeatureCategory.REGIME,
            description="Regime-aware feature integration",
            required_columns=["close", "volume"],
            default_lookback=config.lookback_period,
            min_lookback=10,
            max_lookback=200,
            parameters=config.__dict__,
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(feature_config)
        self.regime_config = config
        self.current_regime = RegimeType.UNKNOWN
        self.regime_history = []
        
    def _detect_regime(self, data: pd.DataFrame) -> RegimeType:
        """Detect current market regime."""
        try:
            if len(data) < self.regime_config.min_samples_per_regime:
                return RegimeType.UNKNOWN
                
            # Calculate regime indicators
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=self.regime_config.lookback_period).std()
            trend_strength = abs(returns.rolling(window=self.regime_config.lookback_period).mean())
            
            # Simple regime detection logic
            recent_vol = volatility.iloc[-1] if not volatility.empty else 0
            recent_trend = trend_strength.iloc[-1] if not trend_strength.empty else 0
            
            if recent_vol > self.regime_config.regime_threshold:
                return RegimeType.VOLATILE
            elif recent_trend > self.regime_config.regime_threshold:
                return RegimeType.TRENDING
            elif recent_vol < 0.1:  # Low volatility threshold
                return RegimeType.STABLE
            else:
                return RegimeType.MEAN_REVERTING
                
        except Exception:
            return RegimeType.UNKNOWN
    
    def _generate_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate regime-specific features."""
        regime = self._detect_regime(data)
        self.current_regime = regime
        self.regime_history.append(regime)
        
        # Keep only recent regime history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
            
        features = {
            'regime_type': regime.value,
            'regime_confidence': 0.8,  # Placeholder
            'regime_duration': len(self.regime_history),
            'regime_stability': self._calculate_regime_stability()
        }
        
        if self.regime_config.enable_adaptive_features:
            features.update(self._generate_adaptive_features(data, regime))
            
        if self.regime_config.enable_regime_transitions:
            features.update(self._generate_transition_features())
            
        return features
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability score."""
        if len(self.regime_history) < 2:
            return 1.0
            
        # Count regime changes
        changes = sum(1 for i in range(1, len(self.regime_history)) 
                     if self.regime_history[i] != self.regime_history[i-1])
        
        stability = 1.0 - (changes / len(self.regime_history))
        return max(0.0, min(1.0, stability))
    
    def _generate_adaptive_features(self, data: pd.DataFrame, regime: RegimeType) -> Dict[str, Any]:
        """Generate regime-adaptive features."""
        features = {}
        
        if regime == RegimeType.TRENDING:
            # Trending regime features
            features['trend_strength'] = self._calculate_trend_strength(data)
            features['trend_persistence'] = self._calculate_trend_persistence(data)
            
        elif regime == RegimeType.MEAN_REVERTING:
            # Mean reverting regime features
            features['mean_reversion_strength'] = self._calculate_mean_reversion_strength(data)
            features['reversion_speed'] = self._calculate_reversion_speed(data)
            
        elif regime == RegimeType.VOLATILE:
            # Volatile regime features
            features['volatility_clustering'] = self._calculate_volatility_clustering(data)
            features['volatility_persistence'] = self._calculate_volatility_persistence(data)
            
        return features
    
    def _generate_transition_features(self) -> Dict[str, Any]:
        """Generate regime transition features."""
        if len(self.regime_history) < 2:
            return {}
            
        current_regime = self.regime_history[-1]
        previous_regime = self.regime_history[-2]
        
        return {
            'regime_transition': current_regime != previous_regime,
            'transition_from': previous_regime.value,
            'transition_to': current_regime.value
        }
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength indicator."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.regime_config.lookback_period:
                return 0.0
                
            # Simple trend strength calculation
            recent_returns = returns.tail(self.regime_config.lookback_period)
            trend_strength = abs(recent_returns.mean()) / recent_returns.std()
            return min(1.0, trend_strength)
        except:
            return 0.0
    
    def _calculate_trend_persistence(self, data: pd.DataFrame) -> float:
        """Calculate trend persistence indicator."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.regime_config.lookback_period:
                return 0.0
                
            # Count consecutive positive/negative returns
            recent_returns = returns.tail(self.regime_config.lookback_period)
            signs = np.sign(recent_returns)
            persistence = np.sum(np.diff(signs) == 0) / len(signs)
            return persistence
        except:
            return 0.0
    
    def _calculate_mean_reversion_strength(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion strength."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.regime_config.lookback_period:
                return 0.0
                
            # Simple mean reversion calculation
            recent_returns = returns.tail(self.regime_config.lookback_period)
            autocorr = recent_returns.autocorr(lag=1)
            return abs(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _calculate_reversion_speed(self, data: pd.DataFrame) -> float:
        """Calculate reversion speed indicator."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.regime_config.lookback_period:
                return 0.0
                
            # Calculate how quickly prices revert to mean
            recent_returns = returns.tail(self.regime_config.lookback_period)
            mean_return = recent_returns.mean()
            deviations = abs(recent_returns - mean_return)
            reversion_speed = 1.0 / (deviations.mean() + 1e-8)
            return min(1.0, reversion_speed)
        except:
            return 0.0
    
    def _calculate_volatility_clustering(self, data: pd.DataFrame) -> float:
        """Calculate volatility clustering indicator."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.regime_config.lookback_period:
                return 0.0
                
            # Calculate volatility clustering
            recent_returns = returns.tail(self.regime_config.lookback_period)
            squared_returns = recent_returns ** 2
            autocorr = squared_returns.autocorr(lag=1)
            return abs(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _calculate_volatility_persistence(self, data: pd.DataFrame) -> float:
        """Calculate volatility persistence indicator."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.regime_config.lookback_period:
                return 0.0
                
            # Calculate volatility persistence
            recent_returns = returns.tail(self.regime_config.lookback_period)
            volatility = recent_returns.rolling(window=5).std()
            autocorr = volatility.autocorr(lag=1)
            return abs(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime integration features."""
        try:
            regime_features = self._generate_regime_features(data)
            
            # Convert to pandas Series
            feature_series = pd.Series(regime_features, index=[data.index[-1]])
            return feature_series
            
        except Exception as e:
            # Return default features on error
            default_features = {
                'regime_type': 'unknown',
                'regime_confidence': 0.0,
                'regime_duration': 0,
                'regime_stability': 0.0
            }
            return pd.Series(default_features, index=[data.index[-1]])


def generate_regime_features(
    data: pd.DataFrame,
    config: Optional[RegimeFeatureConfig] = None
) -> Dict[str, Any]:
    """
    Generate regime features for given data.
    
    Args:
        data: Market data DataFrame
        config: Regime feature configuration
        
    Returns:
        Dictionary of regime features
    """
    if config is None:
        config = RegimeFeatureConfig()
        
    generator = RegimeFeatureIntegration(config)
    return generator._generate_regime_features(data)


# Default regime feature generators
def create_default_regime_feature_generators() -> List[RegimeFeatureIntegration]:
    """Create default regime feature generators."""
    generators = []
    
    # Basic regime detection
    config = RegimeFeatureConfig(
        name="basic_regime_detection",
        enable_regime_detection=True,
        enable_adaptive_features=False,
        enable_regime_transitions=False
    )
    generators.append(RegimeFeatureIntegration(config))
    
    # Advanced regime features
    config = RegimeFeatureConfig(
        name="advanced_regime_features",
        enable_regime_detection=True,
        enable_adaptive_features=True,
        enable_regime_transitions=True
    )
    generators.append(RegimeFeatureIntegration(config))
    
    return generators
