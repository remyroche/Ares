"""
Regime Weight Manager

Manages regime-specific weights and importance scores
for trading decisions based on market conditions and performance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from ..config.regime_config import RegimeConfig, RegimeType, RegimeWeight

logger = system_logger.getChild('RegimeWeightManager')

class RegimeWeightManager:
    """
    Manages regime-specific weights and importance scores.
    
    Dynamically adjusts regime weights based on:
    - Historical performance
    - Market conditions
    - Regime stability
    - Trading success rates
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logger.getChild('RegimeWeightManager')
        
        # Regime weights (importance scores)
        self.regime_weights: Dict[RegimeType, RegimeWeight] = {}
        
        # Performance tracking
        self.performance_history: Dict[RegimeType, List[Dict[str, Any]]] = {}
        
        # Weight adjustment parameters
        self.learning_rate = 0.1
        self.min_weight = 0.1
        self.max_weight = 2.0
        self.decay_factor = 0.95  # For exponential decay of old performance
        
        # Market condition weights
        self.condition_multipliers: Dict[str, float] = {
            'high_volatility': 1.2,
            'low_volatility': 0.8,
            'trending': 1.1,
            'sideways': 0.9,
            'breakout': 1.3,
            'reversal': 1.0
        }
        
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize regime weight manager."""
        try:
            tprint_info("🔄 Initializing Regime Weight Manager...")
            
            # Initialize default weights
            await self._initialize_default_weights()
            
            # Load historical performance data
            await self._load_performance_history()
            
            # Calculate initial weights based on historical data
            await self._calculate_initial_weights()
            
            tprint_success("✅ Regime Weight Manager initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Regime Weight Manager: {e}")
            return False
    
    async def _initialize_default_weights(self):
        """Initialize default regime weights."""
        try:
            # Set default weights based on regime characteristics
            default_weights = {
                RegimeType.TRENDING_UP: RegimeWeight(
                    base_weight=1.2,
                    confidence_multiplier=1.1,
                    stability_factor=0.9,
                    performance_adjustment=1.0
                ),
                RegimeType.TRENDING_DOWN: RegimeWeight(
                    base_weight=1.2,
                    confidence_multiplier=1.1,
                    stability_factor=0.9,
                    performance_adjustment=1.0
                ),
                RegimeType.SIDEWAYS: RegimeWeight(
                    base_weight=0.8,
                    confidence_multiplier=0.9,
                    stability_factor=1.1,
                    performance_adjustment=1.0
                ),
                RegimeType.HIGH_VOLATILITY: RegimeWeight(
                    base_weight=1.3,
                    confidence_multiplier=0.8,
                    stability_factor=0.7,
                    performance_adjustment=1.0
                ),
                RegimeType.LOW_VOLATILITY: RegimeWeight(
                    base_weight=0.9,
                    confidence_multiplier=1.2,
                    stability_factor=1.3,
                    performance_adjustment=1.0
                ),
                RegimeType.BREAKOUT: RegimeWeight(
                    base_weight=1.5,
                    confidence_multiplier=0.7,
                    stability_factor=0.6,
                    performance_adjustment=1.0
                ),
                RegimeType.REVERSAL: RegimeWeight(
                    base_weight=1.1,
                    confidence_multiplier=0.8,
                    stability_factor=0.8,
                    performance_adjustment=1.0
                ),
                RegimeType.MOMENTUM: RegimeWeight(
                    base_weight=1.3,
                    confidence_multiplier=1.0,
                    stability_factor=0.8,
                    performance_adjustment=1.0
                ),
                RegimeType.MEAN_REVERSION: RegimeWeight(
                    base_weight=1.0,
                    confidence_multiplier=1.0,
                    stability_factor=1.0,
                    performance_adjustment=1.0
                )
            }
            
            self.regime_weights = default_weights
            
            # Initialize performance history
            for regime in RegimeType:
                self.performance_history[regime] = []
            
            self.logger.info("✅ Default regime weights initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize default weights: {e}")
    
    async def _load_performance_history(self):
        """Load historical performance data if available."""
        try:
            import os
            import json
            
            cache_file = "data_cache/regime_performance_history.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Load performance history
                for regime_str, performances in data.get('performance_history', {}).items():
                    regime = RegimeType(regime_str)
                    self.performance_history[regime] = performances
                
                # Load regime weights
                for regime_str, weight_data in data.get('regime_weights', {}).items():
                    regime = RegimeType(regime_str)
                    self.regime_weights[regime] = RegimeWeight(**weight_data)
                
                self.logger.info("✅ Performance history loaded")
            else:
                self.logger.info("📝 No performance history found, using defaults")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load performance history: {e}")
    
    async def _calculate_initial_weights(self):
        """Calculate initial weights based on historical performance."""
        try:
            for regime in RegimeType:
                performances = self.performance_history.get(regime, [])
                
                if performances:
                    # Calculate performance-based adjustment
                    recent_performances = performances[-50:]  # Last 50 performances
                    
                    if recent_performances:
                        success_rate = sum(1 for p in recent_performances if p.get('success', False)) / len(recent_performances)
                        avg_return = np.mean([p.get('return', 0.0) for p in recent_performances])
                        
                        # Adjust performance factor based on success rate and returns
                        performance_factor = 0.5 + success_rate * 0.5 + np.clip(avg_return * 10, -0.5, 0.5)
                        
                        # Update regime weight
                        current_weight = self.regime_weights.get(regime, RegimeWeight())
                        current_weight.performance_adjustment = performance_factor
                        self.regime_weights[regime] = current_weight
            
            self.logger.info("✅ Initial weights calculated based on historical performance")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate initial weights: {e}")
    
    @handles_errors
    @traced(span_name="regime_weight_calculation")
    async def get_regime_weight(
        self,
        regime: RegimeType,
        market_conditions: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0,
        stability_score: float = 1.0
    ) -> float:
        """
        Get effective weight for a specific regime.
        
        Args:
            regime: The regime type
            market_conditions: Current market conditions
            confidence_score: Confidence in regime detection
            stability_score: Regime stability score
            
        Returns:
            Effective weight for the regime
        """
        try:
            # Get base regime weight
            regime_weight = self.regime_weights.get(regime, RegimeWeight())
            
            # Calculate effective weight
            base_weight = regime_weight.base_weight
            confidence_mult = regime_weight.confidence_multiplier * confidence_score
            stability_mult = regime_weight.stability_factor * stability_score
            performance_mult = regime_weight.performance_adjustment
            
            # Apply market condition multipliers
            condition_mult = await self._get_condition_multiplier(regime, market_conditions)
            
            # Calculate final weight
            effective_weight = (
                base_weight * 
                confidence_mult * 
                stability_mult * 
                performance_mult * 
                condition_mult
            )
            
            # Clamp to min/max bounds
            effective_weight = max(self.min_weight, min(self.max_weight, effective_weight))
            
            return effective_weight
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get regime weight for {regime}: {e}")
            return 1.0  # Default weight
    
    async def _get_condition_multiplier(
        self,
        regime: RegimeType,
        market_conditions: Optional[Dict[str, Any]]
    ) -> float:
        """Get market condition multiplier for regime weight."""
        try:
            if not market_conditions:
                return 1.0
            
            multiplier = 1.0
            
            # Volatility-based adjustments
            volatility = market_conditions.get('volatility', {})
            if volatility:
                vol_ratio = volatility.get('ratio', 1.0)
                
                if regime in [RegimeType.HIGH_VOLATILITY, RegimeType.BREAKOUT]:
                    # High volatility regimes benefit from high volatility conditions
                    multiplier *= (1.0 + (vol_ratio - 1.0) * 0.3)
                elif regime in [RegimeType.LOW_VOLATILITY, RegimeType.SIDEWAYS]:
                    # Low volatility regimes benefit from low volatility conditions
                    multiplier *= (1.0 - (vol_ratio - 1.0) * 0.3)
            
            # Trend-based adjustments
            trend = market_conditions.get('trend', {})
            if trend:
                trend_consistency = trend.get('consistency', 0.5)
                
                if regime in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN, RegimeType.MOMENTUM]:
                    # Trending regimes benefit from consistent trends
                    multiplier *= (0.8 + trend_consistency * 0.4)
                elif regime in [RegimeType.SIDEWAYS, RegimeType.MEAN_REVERSION]:
                    # Sideways regimes benefit from inconsistent trends
                    multiplier *= (1.2 - trend_consistency * 0.4)
            
            # Volume-based adjustments
            volume = market_conditions.get('volume', {})
            if volume:
                volume_strength = volume.get('strength', 'normal')
                
                if regime in [RegimeType.BREAKOUT, RegimeType.MOMENTUM]:
                    # Breakout regimes benefit from high volume
                    if volume_strength == 'high':
                        multiplier *= 1.2
                    elif volume_strength == 'low':
                        multiplier *= 0.8
            
            return max(0.5, min(2.0, multiplier))  # Clamp multiplier
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get condition multiplier: {e}")
            return 1.0
    
    @handles_errors
    async def update_regime_performance(
        self,
        regime: RegimeType,
        performance_data: Dict[str, Any]
    ):
        """
        Update regime performance and adjust weights.
        
        Args:
            regime: The regime type
            performance_data: Performance metrics (success, return, duration, etc.)
        """
        try:
            # Add performance data to history
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'success': performance_data.get('success', False),
                'return': performance_data.get('return', 0.0),
                'duration': performance_data.get('duration', 1),
                'confidence': performance_data.get('confidence', 0.5),
                'market_conditions': performance_data.get('market_conditions', {}),
                'metadata': performance_data.get('metadata', {})
            }
            
            self.performance_history[regime].append(performance_record)
            
            # Maintain history size (keep last 200 records)
            if len(self.performance_history[regime]) > 200:
                self.performance_history[regime] = self.performance_history[regime][-200:]
            
            # Update regime weight based on performance
            await self._update_regime_weight(regime)
            
            self.logger.debug(f"Updated performance for regime {regime.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update regime performance: {e}")
    
    async def _update_regime_weight(self, regime: RegimeType):
        """Update regime weight based on recent performance."""
        try:
            performances = self.performance_history.get(regime, [])
            if not performances:
                return
            
            # Get recent performances (last 20)
            recent_performances = performances[-20:]
            
            # Calculate performance metrics
            success_rate = sum(1 for p in recent_performances if p.get('success', False)) / len(recent_performances)
            avg_return = np.mean([p.get('return', 0.0) for p in recent_performances])
            avg_confidence = np.mean([p.get('confidence', 0.5) for p in recent_performances])
            
            # Calculate new performance adjustment
            performance_score = (
                success_rate * 0.5 +  # 50% weight on success rate
                np.clip(avg_return * 10, -0.3, 0.3) +  # 30% weight on returns (clamped)
                avg_confidence * 0.2  # 20% weight on confidence
            )
            
            # Apply exponential smoothing to current performance adjustment
            current_weight = self.regime_weights.get(regime, RegimeWeight())
            old_adjustment = current_weight.performance_adjustment
            
            new_adjustment = (
                self.learning_rate * performance_score +
                (1 - self.learning_rate) * old_adjustment
            )
            
            # Clamp adjustment
            new_adjustment = max(0.3, min(1.7, new_adjustment))
            
            # Update regime weight
            current_weight.performance_adjustment = new_adjustment
            self.regime_weights[regime] = current_weight
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update regime weight for {regime}: {e}")
    
    def get_all_regime_weights(
        self,
        market_conditions: Optional[Dict[str, Any]] = None,
        confidence_scores: Optional[Dict[RegimeType, float]] = None,
        stability_scores: Optional[Dict[RegimeType, float]] = None
    ) -> Dict[RegimeType, float]:
        """Get weights for all regimes."""
        try:
            weights = {}
            
            for regime in RegimeType:
                confidence = confidence_scores.get(regime, 1.0) if confidence_scores else 1.0
                stability = stability_scores.get(regime, 1.0) if stability_scores else 1.0
                
                weight = asyncio.run(self.get_regime_weight(
                    regime, market_conditions, confidence, stability
                ))
                weights[regime] = weight
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get all regime weights: {e}")
            return {regime: 1.0 for regime in RegimeType}
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get weight statistics and performance metrics."""
        try:
            stats = {
                'regime_weights': {},
                'performance_summary': {},
                'weight_distribution': {},
                'total_performances': 0
            }
            
            # Regime weight statistics
            for regime, weight in self.regime_weights.items():
                stats['regime_weights'][regime.value] = {
                    'base_weight': weight.base_weight,
                    'confidence_multiplier': weight.confidence_multiplier,
                    'stability_factor': weight.stability_factor,
                    'performance_adjustment': weight.performance_adjustment,
                    'effective_weight': weight.base_weight * weight.performance_adjustment
                }
            
            # Performance summary
            for regime, performances in self.performance_history.items():
                if performances:
                    recent_performances = performances[-50:]  # Last 50
                    
                    success_rate = sum(1 for p in recent_performances if p.get('success', False)) / len(recent_performances)
                    avg_return = np.mean([p.get('return', 0.0) for p in recent_performances])
                    total_trades = len(performances)
                    
                    stats['performance_summary'][regime.value] = {
                        'success_rate': success_rate,
                        'avg_return': avg_return,
                        'total_trades': total_trades,
                        'recent_trades': len(recent_performances)
                    }
                    
                    stats['total_performances'] += total_trades
            
            # Weight distribution
            effective_weights = [w.base_weight * w.performance_adjustment for w in self.regime_weights.values()]
            if effective_weights:
                stats['weight_distribution'] = {
                    'mean': np.mean(effective_weights),
                    'std': np.std(effective_weights),
                    'min': np.min(effective_weights),
                    'max': np.max(effective_weights)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get weight statistics: {e}")
            return {}
    
    async def save_performance_data(self):
        """Save performance data and weights to cache."""
        try:
            
            # Prepare data for saving
            save_data = {
                'performance_history': {
                    regime.value: performances
                    for regime, performances in self.performance_history.items()
                },
                'regime_weights': {
                    regime.value: {
                        'base_weight': weight.base_weight,
                        'confidence_multiplier': weight.confidence_multiplier,
                        'stability_factor': weight.stability_factor,
                        'performance_adjustment': weight.performance_adjustment
                    }
                    for regime, weight in self.regime_weights.items()
                },
                'learning_rate': self.learning_rate,
                'last_updated': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs("data_cache", exist_ok=True)
            
            # Save to file
            cache_file = "data_cache/regime_performance_history.json"
            with open(cache_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.logger.info("✅ Performance data saved to cache")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save performance data: {e}")
    
    async def reset_regime_weights(self, regime: Optional[RegimeType] = None):
        """Reset regime weights to defaults."""
        try:
            if regime:
                # Reset specific regime
                await self._initialize_default_weights()
                self.performance_history[regime] = []
                self.logger.info(f"✅ Reset weights for regime {regime.value}")
            else:
                # Reset all regimes
                await self._initialize_default_weights()
                for r in RegimeType:
                    self.performance_history[r] = []
                self.logger.info("✅ Reset all regime weights")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to reset regime weights: {e}")
    
    async def stop(self):
        """Stop regime weight manager."""
        try:
            self.logger.info("🛑 Stopping Regime Weight Manager...")
            
            # Save performance data
            await self.save_performance_data()
            
            # Clear data
            self.performance_history.clear()
            
            self.logger.info("✅ Regime Weight Manager stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Regime Weight Manager: {e}")