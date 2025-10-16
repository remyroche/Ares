"""
Dynamic Search Space for Financial Architecture Search

This module provides an adaptive search space that evolves based on market conditions,
regime changes, and performance feedback for both NAS and TAS systems.
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

from .financial_architecture_primitives import (
    FinancialActivationType, RegimeType, FinancialLayerConfig,
    create_financial_activation, create_financial_layer, create_financial_tree_primitive
)

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market condition types."""
    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class SearchSpaceEvolutionStrategy(Enum):
    """Strategies for search space evolution."""
    PERFORMANCE_BASED = "performance_based"
    REGIME_BASED = "regime_based"
    VOLATILITY_BASED = "volatility_based"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class DynamicSearchSpaceConfig:
    """Configuration for dynamic search space."""
    # Base configuration
    initial_architecture_types: List[str] = field(default_factory=lambda: [
        "regime_aware_linear", "volatility_sensitive_linear", "financial_lstm", "financial_transformer"
    ])
    initial_activation_types: List[FinancialActivationType] = field(default_factory=lambda: [
        FinancialActivationType.VOLATILITY_SENSITIVE,
        FinancialActivationType.REGIME_AWARE,
        FinancialActivationType.SHARPE_OPTIMIZED
    ])
    initial_tree_primitives: List[str] = field(default_factory=lambda: [
        "volatility_ratio", "momentum_score", "mean_reversion", "regime_stability"
    ])
    
    # Evolution parameters
    evolution_strategy: SearchSpaceEvolutionStrategy = SearchSpaceEvolutionStrategy.ADAPTIVE
    evolution_frequency: int = 100  # Evolve every N evaluations
    performance_window: int = 50   # Look back N evaluations for performance
    regime_window: int = 20        # Look back N periods for regime analysis
    
    # Market condition thresholds
    volatility_threshold: float = 0.02
    trend_strength_threshold: float = 0.6
    regime_stability_threshold: float = 0.7
    
    # Search space constraints
    max_architecture_types: int = 10
    max_activation_types: int = 8
    max_tree_primitives: int = 12
    min_architecture_types: int = 3
    min_activation_types: int = 2
    min_tree_primitives: int = 4
    
    # Performance thresholds
    performance_improvement_threshold: float = 0.05
    performance_degradation_threshold: float = -0.03
    diversity_threshold: float = 0.3


@dataclass
class MarketConditionAnalysis:
    """Analysis of current market conditions."""
    condition: MarketCondition
    volatility: float
    trend_strength: float
    regime_stability: float
    market_efficiency: float
    risk_level: float
    confidence: float
    timestamp: datetime


@dataclass
class SearchSpaceState:
    """Current state of the search space."""
    available_architectures: List[str]
    available_activations: List[FinancialActivationType]
    available_tree_primitives: List[str]
    architecture_weights: Dict[str, float]
    activation_weights: Dict[FinancialActivationType, float]
    tree_primitive_weights: Dict[str, float]
    performance_history: List[float]
    regime_history: List[int]
    market_condition_history: List[MarketConditionAnalysis]
    last_evolution: datetime
    evolution_count: int


class MarketConditionAnalyzer:
    """Analyzes market conditions for search space evolution."""
    
    def __init__(self, config: DynamicSearchSpaceConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_market_condition(self, market_data: pd.DataFrame, 
                                regime_predictions: Optional[np.ndarray] = None) -> MarketConditionAnalysis:
        """Analyze current market conditions."""
        try:
            # Calculate basic metrics
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            trend_strength = self._calculate_trend_strength(market_data)
            regime_stability = self._calculate_regime_stability(regime_predictions)
            market_efficiency = self._calculate_market_efficiency(returns)
            risk_level = self._calculate_risk_level(returns)
            
            # Determine market condition
            condition = self._determine_market_condition(
                volatility, trend_strength, regime_stability, market_efficiency
            )
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence(
                volatility, trend_strength, regime_stability
            )
            
            return MarketConditionAnalysis(
                condition=condition,
                volatility=volatility,
                trend_strength=trend_strength,
                regime_stability=regime_stability,
                market_efficiency=market_efficiency,
                risk_level=risk_level,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Market condition analysis failed: {e}")
            return MarketConditionAnalysis(
                condition=MarketCondition.RANGING,
                volatility=0.02,
                trend_strength=0.5,
                regime_stability=0.5,
                market_efficiency=0.5,
                risk_level=0.5,
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength."""
        try:
            prices = market_data['close'].values
            if len(prices) < 10:
                return 0.5
            
            # Calculate trend using linear regression
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope
            price_range = np.max(prices) - np.min(prices)
            trend_strength = abs(slope) / (price_range / len(prices))
            
            return min(trend_strength, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_regime_stability(self, regime_predictions: Optional[np.ndarray]) -> float:
        """Calculate regime stability."""
        if regime_predictions is None or len(regime_predictions) < 5:
            return 0.5
        
        # Calculate regime consistency
        unique_regimes, counts = np.unique(regime_predictions, return_counts=True)
        max_count = np.max(counts)
        total_count = len(regime_predictions)
        
        stability = max_count / total_count
        return stability
    
    def _calculate_market_efficiency(self, returns: pd.Series) -> float:
        """Calculate market efficiency."""
        try:
            if len(returns) < 10:
                return 0.5
            
            # Autocorrelation test
            autocorr = returns.autocorr(lag=1)
            efficiency = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
            
            return max(0.0, min(1.0, efficiency))
            
        except Exception:
            return 0.5
    
    def _calculate_risk_level(self, returns: pd.Series) -> float:
        """Calculate risk level."""
        try:
            if len(returns) < 10:
                return 0.5
            
            # Calculate VaR (Value at Risk)
            var_95 = np.percentile(returns, 5)
            risk_level = abs(var_95)
            
            # Normalize to 0-1 range
            return min(risk_level * 10, 1.0)
            
        except Exception:
            return 0.5
    
    def _determine_market_condition(self, volatility: float, trend_strength: float,
                                  regime_stability: float, market_efficiency: float) -> MarketCondition:
        """Determine market condition based on metrics."""
        # High volatility conditions
        if volatility > self.config.volatility_threshold:
            if regime_stability < 0.3:
                return MarketCondition.CRISIS
            else:
                return MarketCondition.HIGH_VOLATILITY
        
        # Low volatility conditions
        if volatility < self.config.volatility_threshold * 0.5:
            return MarketCondition.LOW_VOLATILITY
        
        # Trending conditions
        if trend_strength > self.config.trend_strength_threshold:
            if trend_strength > 0.8:
                return MarketCondition.BULL_MARKET
            else:
                return MarketCondition.TRENDING
        
        # Ranging conditions
        if trend_strength < 0.3:
            return MarketCondition.RANGING
        
        # Default to ranging
        return MarketCondition.RANGING
    
    def _calculate_analysis_confidence(self, volatility: float, trend_strength: float,
                                    regime_stability: float) -> float:
        """Calculate confidence in market condition analysis."""
        # Higher confidence with more extreme conditions
        volatility_confidence = min(volatility * 20, 1.0)
        trend_confidence = min(trend_strength * 1.5, 1.0)
        regime_confidence = regime_stability
        
        # Average confidence
        confidence = (volatility_confidence + trend_confidence + regime_confidence) / 3
        return confidence


class DynamicSearchSpace:
    """Dynamic search space that evolves based on market conditions."""
    
    def __init__(self, config: DynamicSearchSpaceConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.market_analyzer = MarketConditionAnalyzer(config)
        self.search_space_state = self._initialize_search_space()
        
        # Evolution tracking
        self.evolution_history = []
        self.performance_tracker = PerformanceTracker()
        
        self.logger.info("✅ Dynamic Search Space initialized")
        self.logger.info(f"   Evolution Strategy: {config.evolution_strategy.value}")
        self.logger.info(f"   Initial Architectures: {len(self.search_space_state.available_architectures)}")
        self.logger.info(f"   Initial Activations: {len(self.search_space_state.available_activations)}")
        self.logger.info(f"   Initial Tree Primitives: {len(self.search_space_state.available_tree_primitives)}")
    
    def _initialize_search_space(self) -> SearchSpaceState:
        """Initialize the search space state."""
        return SearchSpaceState(
            available_architectures=self.config.initial_architecture_types.copy(),
            available_activations=self.config.initial_activation_types.copy(),
            available_tree_primitives=self.config.initial_tree_primitives.copy(),
            architecture_weights={arch: 1.0 for arch in self.config.initial_architecture_types},
            activation_weights={act: 1.0 for act in self.config.initial_activation_types},
            tree_primitive_weights={prim: 1.0 for prim in self.config.initial_tree_primitives},
            performance_history=[],
            regime_history=[],
            market_condition_history=[],
            last_evolution=datetime.now(),
            evolution_count=0
        )
    
    def update_market_condition(self, market_data: pd.DataFrame,
                               regime_predictions: Optional[np.ndarray] = None):
        """Update market condition analysis."""
        try:
            # Analyze current market condition
            condition_analysis = self.market_analyzer.analyze_market_condition(
                market_data, regime_predictions
            )
            
            # Update state
            self.search_space_state.market_condition_history.append(condition_analysis)
            
            # Keep only recent history
            if len(self.search_space_state.market_condition_history) > self.config.regime_window:
                self.search_space_state.market_condition_history.pop(0)
            
            # Check if evolution is needed
            if self._should_evolve():
                self._evolve_search_space()
            
            self.logger.debug(f"Market condition updated: {condition_analysis.condition.value}")
            
        except Exception as e:
            self.logger.error(f"Market condition update failed: {e}")
    
    def update_performance(self, architecture: str, performance: float, 
                          regime: Optional[int] = None):
        """Update performance tracking."""
        try:
            # Update performance history
            self.search_space_state.performance_history.append(performance)
            
            # Update regime history
            if regime is not None:
                self.search_space_state.regime_history.append(regime)
            
            # Keep only recent history
            if len(self.search_space_state.performance_history) > self.config.performance_window:
                self.search_space_state.performance_history.pop(0)
            if len(self.search_space_state.regime_history) > self.config.regime_window:
                self.search_space_state.regime_history.pop(0)
            
            # Update performance tracker
            self.performance_tracker.update(architecture, performance, regime)
            
            # Check if evolution is needed
            if self._should_evolve():
                self._evolve_search_space()
            
        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")
    
    def _should_evolve(self) -> bool:
        """Check if search space should evolve."""
        # Check evolution frequency
        time_since_evolution = datetime.now() - self.search_space_state.last_evolution
        if time_since_evolution.total_seconds() < 3600:  # 1 hour minimum
            return False
        
        # Check if enough data for evolution
        if len(self.search_space_state.performance_history) < self.config.performance_window:
            return False
        
        # Check performance trends
        recent_performance = self.search_space_state.performance_history[-self.config.performance_window:]
        performance_trend = np.mean(recent_performance[-10:]) - np.mean(recent_performance[:10])
        
        # Evolve if performance is degrading
        if performance_trend < self.config.performance_degradation_threshold:
            return True
        
        # Evolve if market conditions have changed significantly
        if len(self.search_space_state.market_condition_history) >= 2:
            recent_condition = self.search_space_state.market_condition_history[-1]
            previous_condition = self.search_space_state.market_condition_history[-2]
            
            if recent_condition.condition != previous_condition.condition:
                return True
        
        return False
    
    def _evolve_search_space(self):
        """Evolve the search space based on current conditions."""
        try:
            self.logger.info("🔄 Evolving search space...")
            
            # Get current market condition
            current_condition = self.search_space_state.market_condition_history[-1]
            
            # Apply evolution strategy
            if self.config.evolution_strategy == SearchSpaceEvolutionStrategy.PERFORMANCE_BASED:
                self._evolve_based_on_performance()
            elif self.config.evolution_strategy == SearchSpaceEvolutionStrategy.REGIME_BASED:
                self._evolve_based_on_regime()
            elif self.config.evolution_strategy == SearchSpaceEvolutionStrategy.VOLATILITY_BASED:
                self._evolve_based_on_volatility(current_condition)
            elif self.config.evolution_strategy == SearchSpaceEvolutionStrategy.ADAPTIVE:
                self._evolve_adaptively(current_condition)
            else:  # HYBRID
                self._evolve_hybrid(current_condition)
            
            # Update evolution tracking
            self.search_space_state.last_evolution = datetime.now()
            self.search_space_state.evolution_count += 1
            
            # Record evolution
            self.evolution_history.append({
                'timestamp': datetime.now(),
                'condition': current_condition.condition.value,
                'architectures': len(self.search_space_state.available_architectures),
                'activations': len(self.search_space_state.available_activations),
                'tree_primitives': len(self.search_space_state.available_tree_primitives)
            })
            
            self.logger.info(f"✅ Search space evolved (count: {self.search_space_state.evolution_count})")
            
        except Exception as e:
            self.logger.error(f"Search space evolution failed: {e}")
    
    def _evolve_based_on_performance(self):
        """Evolve search space based on performance feedback."""
        # Get performance by architecture type
        performance_by_arch = self.performance_tracker.get_performance_by_architecture()
        
        # Remove poorly performing architectures
        poor_performers = [
            arch for arch, perf in performance_by_arch.items()
            if perf < np.mean(list(performance_by_arch.values())) - 0.1
        ]
        
        for arch in poor_performers:
            if arch in self.search_space_state.available_architectures:
                self.search_space_state.available_architectures.remove(arch)
                self.logger.info(f"Removed poor performer: {arch}")
        
        # Add new architectures if space is too small
        if len(self.search_space_state.available_architectures) < self.config.min_architecture_types:
            self._add_new_architectures()
    
    def _evolve_based_on_regime(self):
        """Evolve search space based on regime analysis."""
        if not self.search_space_state.regime_history:
            return
        
        # Analyze regime patterns
        regime_counts = {}
        for regime in self.search_space_state.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Adjust weights based on regime frequency
        for regime, count in regime_counts.items():
            weight = count / len(self.search_space_state.regime_history)
            # Update architecture weights based on regime performance
            # This is a simplified implementation
            pass
    
    def _evolve_based_on_volatility(self, condition: MarketConditionAnalysis):
        """Evolve search space based on volatility conditions."""
        if condition.volatility > self.config.volatility_threshold:
            # High volatility - add volatility-sensitive components
            if FinancialActivationType.VOLATILITY_SENSITIVE not in self.search_space_state.available_activations:
                self.search_space_state.available_activations.append(FinancialActivationType.VOLATILITY_SENSITIVE)
                self.logger.info("Added volatility-sensitive activation for high volatility")
        else:
            # Low volatility - add trend-following components
            if FinancialActivationType.MOMENTUM_BASED not in self.search_space_state.available_activations:
                self.search_space_state.available_activations.append(FinancialActivationType.MOMENTUM_BASED)
                self.logger.info("Added momentum-based activation for low volatility")
    
    def _evolve_adaptively(self, condition: MarketConditionAnalysis):
        """Adaptive evolution based on multiple factors."""
        # Combine performance and market condition evolution
        self._evolve_based_on_performance()
        self._evolve_based_on_volatility(condition)
        
        # Add regime-aware components if needed
        if condition.regime_stability < 0.5:
            if FinancialActivationType.REGIME_AWARE not in self.search_space_state.available_activations:
                self.search_space_state.available_activations.append(FinancialActivationType.REGIME_AWARE)
                self.logger.info("Added regime-aware activation for unstable regimes")
    
    def _evolve_hybrid(self, condition: MarketConditionAnalysis):
        """Hybrid evolution combining all strategies."""
        self._evolve_adaptively(condition)
        
        # Add financial-specific components based on market condition
        if condition.condition == MarketCondition.TRENDING:
            if "financial_lstm" not in self.search_space_state.available_architectures:
                self.search_space_state.available_architectures.append("financial_lstm")
        elif condition.condition == MarketCondition.HIGH_VOLATILITY:
            if "financial_transformer" not in self.search_space_state.available_architectures:
                self.search_space_state.available_architectures.append("financial_transformer")
    
    def _add_new_architectures(self):
        """Add new architectures to maintain diversity."""
        all_architectures = [
            "regime_aware_linear", "volatility_sensitive_linear", "financial_lstm", "financial_transformer",
            "regime_aware_conv", "volatility_sensitive_conv", "financial_gru", "financial_attention"
        ]
        
        for arch in all_architectures:
            if arch not in self.search_space_state.available_architectures:
                self.search_space_state.available_architectures.append(arch)
                self.logger.info(f"Added new architecture: {arch}")
                break
    
    def sample_architecture(self, architecture_type: str = "neural") -> Dict[str, Any]:
        """Sample an architecture from the current search space."""
        try:
            if architecture_type == "neural":
                return self._sample_neural_architecture()
            elif architecture_type == "tree":
                return self._sample_tree_architecture()
            else:
                return self._sample_hybrid_architecture()
        except Exception as e:
            self.logger.error(f"Architecture sampling failed: {e}")
            return self._get_default_architecture()
    
    def _sample_neural_architecture(self) -> Dict[str, Any]:
        """Sample a neural architecture."""
        # Select architecture type
        arch_type = np.random.choice(
            self.search_space_state.available_architectures,
            p=self._get_architecture_probabilities()
        )
        
        # Select activation
        activation = np.random.choice(
            self.search_space_state.available_activations,
            p=self._get_activation_probabilities()
        )
        
        # Generate architecture parameters
        architecture = {
            'type': 'neural',
            'architecture_type': arch_type,
            'activation': activation.value,
            'layers': self._generate_layer_config(arch_type),
            'regime_aware': True,
            'volatility_sensitive': True
        }
        
        return architecture
    
    def _sample_tree_architecture(self) -> Dict[str, Any]:
        """Sample a tree architecture."""
        # Select tree primitives
        primitives = np.random.choice(
            self.search_space_state.available_tree_primitives,
            size=min(3, len(self.search_space_state.available_tree_primitives)),
            replace=False,
            p=self._get_tree_primitive_probabilities()
        )
        
        architecture = {
            'type': 'tree',
            'primitives': primitives.tolist(),
            'max_depth': np.random.randint(3, 8),
            'min_samples_split': np.random.randint(10, 50),
            'regime_aware': True
        }
        
        return architecture
    
    def _sample_hybrid_architecture(self) -> Dict[str, Any]:
        """Sample a hybrid architecture."""
        neural_arch = self._sample_neural_architecture()
        tree_arch = self._sample_tree_architecture()
        
        return {
            'type': 'hybrid',
            'neural_component': neural_arch,
            'tree_component': tree_arch,
            'fusion_method': np.random.choice(['weighted', 'attention', 'ensemble'])
        }
    
    def _get_architecture_probabilities(self) -> np.ndarray:
        """Get probabilities for architecture selection."""
        weights = [self.search_space_state.architecture_weights.get(arch, 1.0) 
                  for arch in self.search_space_state.available_architectures]
        return np.array(weights) / np.sum(weights)
    
    def _get_activation_probabilities(self) -> np.ndarray:
        """Get probabilities for activation selection."""
        weights = [self.search_space_state.activation_weights.get(act, 1.0) 
                  for act in self.search_space_state.available_activations]
        return np.array(weights) / np.sum(weights)
    
    def _get_tree_primitive_probabilities(self) -> np.ndarray:
        """Get probabilities for tree primitive selection."""
        weights = [self.search_space_state.tree_primitive_weights.get(prim, 1.0) 
                  for prim in self.search_space_state.available_tree_primitives]
        return np.array(weights) / np.sum(weights)
    
    def _generate_layer_config(self, arch_type: str) -> List[Dict[str, Any]]:
        """Generate layer configuration for architecture."""
        n_layers = np.random.randint(2, 6)
        layers = []
        
        for i in range(n_layers):
            layer = {
                'type': arch_type,
                'hidden_size': np.random.randint(32, 256),
                'dropout': np.random.uniform(0.1, 0.5),
                'batch_norm': np.random.choice([True, False])
            }
            layers.append(layer)
        
        return layers
    
    def _get_default_architecture(self) -> Dict[str, Any]:
        """Get default architecture if sampling fails."""
        return {
            'type': 'neural',
            'architecture_type': 'regime_aware_linear',
            'activation': 'volatility_sensitive',
            'layers': [{'type': 'regime_aware_linear', 'hidden_size': 64, 'dropout': 0.2}],
            'regime_aware': True,
            'volatility_sensitive': True
        }
    
    def get_search_space_info(self) -> Dict[str, Any]:
        """Get information about current search space."""
        return {
            'available_architectures': self.search_space_state.available_architectures,
            'available_activations': [act.value for act in self.search_space_state.available_activations],
            'available_tree_primitives': self.search_space_state.available_tree_primitives,
            'evolution_count': self.search_space_state.evolution_count,
            'last_evolution': self.search_space_state.last_evolution.isoformat(),
            'performance_history_length': len(self.search_space_state.performance_history),
            'market_condition_history_length': len(self.search_space_state.market_condition_history)
        }


class PerformanceTracker:
    """Tracks performance by architecture and regime."""
    
    def __init__(self):
        self.performance_data = {}
        self.regime_performance = {}
        
    def update(self, architecture: str, performance: float, regime: Optional[int] = None):
        """Update performance tracking."""
        if architecture not in self.performance_data:
            self.performance_data[architecture] = []
        self.performance_data[architecture].append(performance)
        
        if regime is not None:
            if regime not in self.regime_performance:
                self.regime_performance[regime] = []
            self.regime_performance[regime].append(performance)
    
    def get_performance_by_architecture(self) -> Dict[str, float]:
        """Get average performance by architecture."""
        return {
            arch: np.mean(perfs) if perfs else 0.0
            for arch, perfs in self.performance_data.items()
        }
    
    def get_performance_by_regime(self) -> Dict[int, float]:
        """Get average performance by regime."""
        return {
            regime: np.mean(perfs) if perfs else 0.0
            for regime, perfs in self.regime_performance.items()
        }


def create_dynamic_search_space(config: DynamicSearchSpaceConfig) -> DynamicSearchSpace:
    """Create a dynamic search space instance."""
    return DynamicSearchSpace(config)