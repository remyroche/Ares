"""Dynamic SR Level Management for Step02_5.

This module provides suggestions and implementations for making SR levels
more dynamic and adaptive to changing market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from src.utils.logger import system_logger
from datetime import datetime, timedelta

logger = system_logger.getChild('DynamicSRSuggestions')

class DynamicSRManager:
    """Manages dynamic SR levels that adapt to market conditions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dynamic SR manager."""
        self.config = config
        self.logger = system_logger.getChild('DynamicSRManager')
        
        # Configuration parameters
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.time_decay_factor = config.get('time_decay_factor', 0.95)
        self.regime_change_threshold = config.get('regime_change_threshold', 0.3)
        self.min_level_age = config.get('min_level_age', 100)  # bars
        self.max_level_age = config.get('max_level_age', 1000)  # bars
        
    def adapt_sr_levels_to_market_conditions(
        self, 
        sr_levels: Dict[str, List[float]], 
        market_data: pd.DataFrame,
        current_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adapt SR levels based on current market conditions.
        
        Args:
            sr_levels: Dictionary with 'support_levels' and 'resistance_levels'
            market_data: Recent market data for analysis
            current_regime: Current market regime (if available)
            
        Returns:
            Dictionary with adapted SR levels and metadata
        """
        self.logger.info('🔄 Adapting SR levels to current market conditions...')
        
        # Analyze current market conditions
        market_conditions = self._analyze_market_conditions(market_data)
        
        # Apply dynamic adjustments
        adapted_levels = self._apply_dynamic_adjustments(sr_levels, market_conditions, current_regime)
        
        # Calculate level strength with time decay
        enhanced_levels = self._calculate_dynamic_strength(adapted_levels, market_data)
        
        # Filter levels based on current relevance
        filtered_levels = self._filter_relevant_levels(enhanced_levels, market_conditions)
        
        self.logger.info(f'✅ SR levels adapted: {len(filtered_levels["support_levels"])} support, {len(filtered_levels["resistance_levels"])} resistance')
        
        return {
            'adapted_levels': filtered_levels,
            'market_conditions': market_conditions,
            'adaptation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'regime': current_regime,
                'volatility_regime': market_conditions['volatility_regime'],
                'volume_regime': market_conditions['volume_regime']
            }
        }
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions."""
        recent_data = market_data.tail(50)  # Last 50 bars
        
        # Volatility analysis
        returns = recent_data['close'].pct_change().dropna()
        current_volatility = returns.std()
        volatility_ma = returns.rolling(20).std().iloc[-1]
        
        # Volume analysis
        current_volume = recent_data['volume'].iloc[-1]
        volume_ma = recent_data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / volume_ma
        
        # Price momentum
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-20]) / recent_data['close'].iloc[-20]
        
        # Trend strength
        sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = recent_data['close'].rolling(50).mean().iloc[-1] if len(recent_data) >= 50 else sma_20
        trend_strength = (sma_20 - sma_50) / sma_50
        
        return {
            'volatility': current_volatility,
            'volatility_regime': 'high' if current_volatility > self.volatility_threshold else 'low',
            'volume_ratio': volume_ratio,
            'volume_regime': 'high' if volume_ratio > self.volume_threshold else 'normal',
            'price_momentum': price_change,
            'trend_strength': trend_strength,
            'trend_direction': 'up' if trend_strength > 0 else 'down'
        }
    
    def _apply_dynamic_adjustments(
        self, 
        sr_levels: Dict[str, List[float]], 
        market_conditions: Dict[str, Any],
        current_regime: Optional[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Apply dynamic adjustments to SR levels."""
        adapted_levels = {'support_levels': [], 'resistance_levels': []}
        
        # Adjust support levels
        for level in sr_levels.get('support_levels', []):
            adapted_level = self._adjust_single_level(level, 'support', market_conditions, current_regime)
            adapted_levels['support_levels'].append(adapted_level)
        
        # Adjust resistance levels
        for level in sr_levels.get('resistance_levels', []):
            adapted_level = self._adjust_single_level(level, 'resistance', market_conditions, current_regime)
            adapted_levels['resistance_levels'].append(adapted_level)
        
        return adapted_levels
    
    def _adjust_single_level(
        self, 
        level: float, 
        level_type: str, 
        market_conditions: Dict[str, Any],
        current_regime: Optional[str]
    ) -> Dict[str, Any]:
        """Adjust a single SR level based on market conditions."""
        base_level = level
        adjustment_factor = 1.0
        
        # Volatility-based adjustment
        if market_conditions['volatility_regime'] == 'high':
            # Widen levels in high volatility
            adjustment_factor *= 1.1
        else:
            # Tighten levels in low volatility
            adjustment_factor *= 0.95
        
        # Volume-based adjustment
        if market_conditions['volume_regime'] == 'high':
            # Strengthen levels with high volume
            adjustment_factor *= 1.05
        
        # Regime-based adjustment
        if current_regime:
            if current_regime == 'trending' and level_type == 'support':
                # Strengthen support in uptrend
                adjustment_factor *= 1.02
            elif current_regime == 'trending' and level_type == 'resistance':
                # Strengthen resistance in downtrend
                adjustment_factor *= 1.02
            elif current_regime == 'ranging':
                # Maintain levels in ranging market
                adjustment_factor *= 1.0
        
        adjusted_level = base_level * adjustment_factor
        
        return {
            'original_price': base_level,
            'adjusted_price': adjusted_level,
            'adjustment_factor': adjustment_factor,
            'level_type': level_type,
            'strength': min(adjustment_factor, 1.0),
            'market_conditions': market_conditions
        }
    
    def _calculate_dynamic_strength(
        self, 
        adapted_levels: Dict[str, List[Dict[str, Any]]], 
        market_data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate dynamic strength for SR levels with time decay."""
        enhanced_levels = {'support_levels': [], 'resistance_levels': []}
        
        for level_type in ['support_levels', 'resistance_levels']:
            for level_info in adapted_levels[level_type]:
                # Calculate time-based strength decay
                time_strength = self._calculate_time_strength(level_info, market_data)
                
                # Calculate volume-based strength
                volume_strength = self._calculate_volume_strength(level_info, market_data)
                
                # Calculate proximity strength (how close current price is)
                proximity_strength = self._calculate_proximity_strength(level_info, market_data)
                
                # Combine strength factors
                total_strength = (
                    level_info['strength'] * 0.4 +
                    time_strength * 0.3 +
                    volume_strength * 0.2 +
                    proximity_strength * 0.1
                )
                
                enhanced_level = level_info.copy()
                enhanced_level.update({
                    'time_strength': time_strength,
                    'volume_strength': volume_strength,
                    'proximity_strength': proximity_strength,
                    'total_strength': total_strength,
                    'is_active': total_strength > 0.3  # Threshold for active levels
                })
                
                enhanced_levels[level_type].append(enhanced_level)
        
        return enhanced_levels
    
    def _calculate_time_strength(self, level_info: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate time-based strength with decay."""
        # Simulate level age (in practice, this would be tracked from level creation)
        simulated_age = np.random.randint(self.min_level_age, self.max_level_age)
        
        # Apply time decay
        time_strength = self.time_decay_factor ** (simulated_age / 100)
        
        return max(0.1, time_strength)  # Minimum strength
    
    def _calculate_volume_strength(self, level_info: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate volume-based strength."""
        recent_volume = market_data['volume'].tail(10).mean()
        avg_volume = market_data['volume'].mean()
        
        volume_ratio = recent_volume / avg_volume
        volume_strength = min(volume_ratio / 2.0, 1.0)  # Cap at 1.0
        
        return volume_strength
    
    def _calculate_proximity_strength(self, level_info: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate proximity-based strength."""
        current_price = market_data['close'].iloc[-1]
        level_price = level_info['adjusted_price']
        
        # Calculate distance as percentage
        distance = abs(current_price - level_price) / current_price
        
        # Closer levels are stronger
        proximity_strength = max(0, 1.0 - distance * 10)  # Decay over 10% distance
        
        return proximity_strength
    
    def _filter_relevant_levels(
        self, 
        enhanced_levels: Dict[str, List[Dict[str, Any]]], 
        market_conditions: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Filter levels to keep only relevant ones."""
        filtered_levels = {'support_levels': [], 'resistance_levels': []}
        
        for level_type in ['support_levels', 'resistance_levels']:
            # Sort by total strength
            sorted_levels = sorted(
                enhanced_levels[level_type], 
                key=lambda x: x['total_strength'], 
                reverse=True
            )
            
            # Keep only active levels and top performers
            active_levels = [level for level in sorted_levels if level['is_active']]
            
            # Limit to top 5 levels
            filtered_levels[level_type] = active_levels[:5]
        
        return filtered_levels

class SRLevelSuggestions:
    """Provides suggestions for improving SR level detection and management."""
    
    @staticmethod
    def get_dynamic_sr_suggestions() -> List[Dict[str, Any]]:
        """Get suggestions for making SR levels more dynamic."""
        return [
            {
                'category': 'Time Decay',
                'suggestion': 'Implement time-based strength decay for SR levels',
                'implementation': 'Apply exponential decay factor based on level age',
                'benefit': 'Prevents stale levels from dominating analysis',
                'priority': 'High'
            },
            {
                'category': 'Volatility Adaptation',
                'suggestion': 'Adjust SR level sensitivity based on market volatility',
                'implementation': 'Widen levels in high volatility, tighten in low volatility',
                'benefit': 'Better adaptation to changing market conditions',
                'priority': 'High'
            },
            {
                'category': 'Volume Confirmation',
                'suggestion': 'Weight SR levels by volume confirmation',
                'implementation': 'Boost levels that show volume spikes on touches',
                'benefit': 'More reliable level identification',
                'priority': 'Medium'
            },
            {
                'category': 'Regime Awareness',
                'suggestion': 'Adapt SR detection based on market regime',
                'implementation': 'Different parameters for trending vs ranging markets',
                'benefit': 'Regime-specific level optimization',
                'priority': 'Medium'
            },
            {
                'category': 'Multi-Timeframe',
                'suggestion': 'Combine SR levels from multiple timeframes',
                'implementation': 'Weight levels by timeframe confluence',
                'benefit': 'More robust level identification',
                'priority': 'Low'
            },
            {
                'category': 'Machine Learning',
                'suggestion': 'Use ML to predict SR level effectiveness',
                'implementation': 'Train models on historical level performance',
                'benefit': 'Data-driven level selection',
                'priority': 'Low'
            }
        ]
    
    @staticmethod
    def get_implementation_roadmap() -> List[Dict[str, Any]]:
        """Get implementation roadmap for dynamic SR improvements."""
        return [
            {
                'phase': 'Phase 1 - Immediate',
                'tasks': [
                    'Implement time decay for existing levels',
                    'Add volatility-based level adjustment',
                    'Create dynamic strength calculation'
                ],
                'timeline': '1-2 weeks',
                'effort': 'Medium'
            },
            {
                'phase': 'Phase 2 - Short Term',
                'tasks': [
                    'Add volume confirmation weighting',
                    'Implement regime-aware parameters',
                    'Create level filtering system'
                ],
                'timeline': '2-4 weeks',
                'effort': 'High'
            },
            {
                'phase': 'Phase 3 - Long Term',
                'tasks': [
                    'Multi-timeframe level integration',
                    'ML-based level prediction',
                    'Real-time level adaptation'
                ],
                'timeline': '1-3 months',
                'effort': 'Very High'
            }
        ]