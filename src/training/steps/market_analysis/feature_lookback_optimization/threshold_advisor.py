"""
Variance Threshold Advisor

This module provides intelligent threshold recommendations for period consolidation
based on trading strategy, market conditions, and feature types.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class TradingTimeframe(Enum):
    """Trading timeframe categories."""
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"

class MarketVolatility(Enum):
    """Market volatility levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class FeatureType(Enum):
    """Feature type categories."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    UNKNOWN = "unknown"

@dataclass
class ThresholdRecommendation:
    """Threshold recommendation with reasoning."""
    recommended_threshold: float
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    alternative_thresholds: Dict[str, float]
    considerations: List[str]

class ThresholdAdvisor:
    """Provides intelligent threshold recommendations."""
    
    # Base thresholds by trading timeframe
    TIMEFRAME_BASE_THRESHOLDS = {
        TradingTimeframe.INTRADAY: 0.12,   # 12% - precision critical for intraday/scalping
        TradingTimeframe.SWING: 0.15,      # 15% - balanced
        TradingTimeframe.POSITION: 0.25    # 25% - trends similar
    }
    
    # Adjustments by market volatility
    VOLATILITY_MULTIPLIERS = {
        MarketVolatility.HIGH: 0.7,    # Stricter in volatile markets
        MarketVolatility.MEDIUM: 1.0,  # No adjustment
        MarketVolatility.LOW: 1.4      # More lenient in stable markets
    }
    
    # Feature-specific thresholds
    FEATURE_TYPE_THRESHOLDS = {
        FeatureType.TREND: 0.25,       # Trends work similarly both ways
        FeatureType.MOMENTUM: 0.15,    # Momentum can differ significantly
        FeatureType.VOLATILITY: 0.10,  # Volatility often asymmetric
        FeatureType.VOLUME: 0.30,      # Volume often similar
        FeatureType.UNKNOWN: 0.15      # Conservative default
    }
    
    def __init__(self):
        """Initialize the threshold advisor."""
        pass
    
    def recommend_threshold(self,
                          trading_timeframe: TradingTimeframe = TradingTimeframe.SWING,
                          market_volatility: MarketVolatility = MarketVolatility.MEDIUM,
                          primary_feature_types: Optional[List[FeatureType]] = None,
                          custom_requirements: Optional[Dict[str, Any]] = None) -> ThresholdRecommendation:
        """
        Recommend optimal threshold based on trading characteristics.
        
        Args:
            trading_timeframe: Primary trading timeframe
            market_volatility: Expected market volatility
            primary_feature_types: Main types of features being optimized
            custom_requirements: Custom requirements (precision_critical, feature_count_critical, etc.)
            
        Returns:
            ThresholdRecommendation with detailed analysis
        """
        
        reasoning = []
        considerations = []
        
        # 1. Start with timeframe-based threshold
        base_threshold = self.TIMEFRAME_BASE_THRESHOLDS[trading_timeframe]
        reasoning.append(f"Base threshold {base_threshold:.1%} for {trading_timeframe.value} trading")
        
        # 2. Adjust for market volatility
        volatility_multiplier = self.VOLATILITY_MULTIPLIERS[market_volatility]
        adjusted_threshold = base_threshold * volatility_multiplier
        
        if volatility_multiplier != 1.0:
            reasoning.append(f"Adjusted to {adjusted_threshold:.1%} for {market_volatility.value} volatility market")
        
        # 3. Consider feature types
        if primary_feature_types:
            feature_thresholds = [self.FEATURE_TYPE_THRESHOLDS[ft] for ft in primary_feature_types]
            avg_feature_threshold = sum(feature_thresholds) / len(feature_thresholds)
            
            # Blend with timeframe/volatility adjusted threshold
            blended_threshold = (adjusted_threshold + avg_feature_threshold) / 2
            reasoning.append(f"Blended with feature-type average {avg_feature_threshold:.1%} → {blended_threshold:.1%}")
            adjusted_threshold = blended_threshold
        
        # 4. Apply custom requirements
        if custom_requirements:
            if custom_requirements.get('precision_critical', False):
                adjusted_threshold *= 0.7
                reasoning.append("Reduced by 30% for precision-critical application")
            
            if custom_requirements.get('feature_count_critical', False):
                adjusted_threshold *= 1.3
                reasoning.append("Increased by 30% for feature-count-critical application")
            
            if custom_requirements.get('performance_sensitive', False):
                adjusted_threshold *= 0.8
                reasoning.append("Reduced by 20% for performance-sensitive application")
        
        # 5. Ensure reasonable bounds
        final_threshold = max(0.05, min(0.45, adjusted_threshold))
        if final_threshold != adjusted_threshold:
            reasoning.append(f"Bounded to reasonable range: {final_threshold:.1%}")
        
        # 6. Generate alternatives
        alternatives = {
            'conservative': final_threshold * 0.7,
            'aggressive': final_threshold * 1.4,
            'very_strict': 0.08,
            'very_lenient': 0.35
        }
        
        # 7. Add considerations
        considerations.extend(self._generate_considerations(
            trading_timeframe, market_volatility, primary_feature_types, final_threshold
        ))
        
        # 8. Calculate confidence
        confidence = self._calculate_confidence(
            trading_timeframe, market_volatility, primary_feature_types, custom_requirements
        )
        
        return ThresholdRecommendation(
            recommended_threshold=final_threshold,
            confidence=confidence,
            reasoning=reasoning,
            alternative_thresholds=alternatives,
            considerations=considerations
        )
    
    def _generate_considerations(self,
                               timeframe: TradingTimeframe,
                               volatility: MarketVolatility,
                               feature_types: Optional[List[FeatureType]],
                               threshold: float) -> List[str]:
        """Generate important considerations for the recommendation."""
        considerations = []
        
        # Timeframe considerations
        if timeframe == TradingTimeframe.INTRADAY:
            considerations.append("Intraday trading: Small period differences can significantly impact performance")
            considerations.append("Consider monitoring consolidation impact on signal quality")
        
        elif timeframe == TradingTimeframe.POSITION:
            considerations.append("Position trading: Long-term trends often behave similarly for long/short")
            considerations.append("Higher consolidation rates are typically acceptable")
        
        # Volatility considerations
        if volatility == MarketVolatility.HIGH:
            considerations.append("High volatility: Long/short strategies may need different optimal periods")
            considerations.append("Monitor for asymmetric volatility effects")
        
        elif volatility == MarketVolatility.LOW:
            considerations.append("Low volatility: Similar optimal periods expected for both directions")
            considerations.append("Higher consolidation rates often beneficial")
        
        # Feature type considerations
        if feature_types:
            if FeatureType.VOLATILITY in feature_types:
                considerations.append("Volatility features: Often exhibit asymmetric behavior between long/short")
            
            if FeatureType.TREND in feature_types:
                considerations.append("Trend features: Usually work similarly for both directions")
            
            if FeatureType.MOMENTUM in feature_types:
                considerations.append("Momentum features: Can have different optimal periods for long/short")
        
        # Threshold-specific considerations
        if threshold < 0.10:
            considerations.append("Very strict threshold: Expect minimal consolidation, maximum precision")
        elif threshold > 0.30:
            considerations.append("Lenient threshold: Expect significant consolidation, potential precision loss")
        
        return considerations
    
    def _calculate_confidence(self,
                            timeframe: TradingTimeframe,
                            volatility: MarketVolatility,
                            feature_types: Optional[List[FeatureType]],
                            custom_requirements: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in the recommendation."""
        
        confidence = 0.8  # Base confidence
        
        # Higher confidence for well-established combinations
        if timeframe == TradingTimeframe.SWING and volatility == MarketVolatility.MEDIUM:
            confidence += 0.1
        
        # Lower confidence for extreme cases
        if timeframe == TradingTimeframe.INTRADAY and volatility == MarketVolatility.HIGH:
            confidence -= 0.1
        
        # Higher confidence when feature types are specified
        if feature_types and len(feature_types) > 0:
            confidence += 0.05
        
        # Lower confidence for conflicting requirements
        if custom_requirements:
            conflicting = 0
            if custom_requirements.get('precision_critical') and custom_requirements.get('feature_count_critical'):
                conflicting += 1
            if conflicting > 0:
                confidence -= 0.1 * conflicting
        
        return max(0.5, min(0.95, confidence))
    
    def analyze_feature_types(self, feature_names: List[str]) -> Dict[FeatureType, int]:
        """Analyze feature names to determine their types."""
        
        type_counts = {ft: 0 for ft in FeatureType}
        
        for feature_name in feature_names:
            feature_type = self._classify_feature(feature_name)
            type_counts[feature_type] += 1
        
        return type_counts
    
    def _classify_feature(self, feature_name: str) -> FeatureType:
        """Classify a feature based on its name."""
        name_lower = feature_name.lower()
        
        # Trend features
        if any(term in name_lower for term in ['sma', 'ema', 'ma', 'trend', 'cross']):
            return FeatureType.TREND
        
        # Momentum features
        elif any(term in name_lower for term in ['rsi', 'macd', 'roc', 'momentum', 'stoch']):
            return FeatureType.MOMENTUM
        
        # Volatility features
        elif any(term in name_lower for term in ['atr', 'volatility', 'bb', 'bollinger', 'vix']):
            return FeatureType.VOLATILITY
        
        # Volume features
        elif any(term in name_lower for term in ['volume', 'obv', 'vwap']):
            return FeatureType.VOLUME
        
        else:
            return FeatureType.UNKNOWN
    
    def get_threshold_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined threshold configurations for common scenarios."""
        
        return {
            'crypto_day_trading': {
                'threshold': 0.08,
                'description': 'High-frequency crypto trading',
                'config': {
                    'trading_timeframe': TradingTimeframe.INTRADAY,
                    'market_volatility': MarketVolatility.HIGH,
                    'custom_requirements': {'precision_critical': True}
                }
            },
            
            'forex_swing_trading': {
                'threshold': 0.15,
                'description': 'Forex swing trading',
                'config': {
                    'trading_timeframe': TradingTimeframe.SWING,
                    'market_volatility': MarketVolatility.MEDIUM,
                    'primary_feature_types': [FeatureType.TREND, FeatureType.MOMENTUM]
                }
            },
            
            'stock_position_trading': {
                'threshold': 0.25,
                'description': 'Long-term stock position trading',
                'config': {
                    'trading_timeframe': TradingTimeframe.POSITION,
                    'market_volatility': MarketVolatility.LOW,
                    'primary_feature_types': [FeatureType.TREND, FeatureType.VOLUME]
                }
            },
            
            'mixed_strategy_balanced': {
                'threshold': 0.15,
                'description': 'Balanced approach for mixed strategies',
                'config': {
                    'trading_timeframe': TradingTimeframe.SWING,
                    'market_volatility': MarketVolatility.MEDIUM,
                    'primary_feature_types': [FeatureType.TREND, FeatureType.MOMENTUM, FeatureType.VOLATILITY]
                }
            },
            
            'feature_count_optimized': {
                'threshold': 0.30,
                'description': 'Optimized for maximum feature consolidation',
                'config': {
                    'trading_timeframe': TradingTimeframe.SWING,
                    'market_volatility': MarketVolatility.MEDIUM,
                    'custom_requirements': {'feature_count_critical': True}
                }
            }
        }

# Convenience functions
def get_threshold_recommendation(trading_style: str = "swing_trading",
                               market_type: str = "medium_volatility",
                               feature_names: Optional[List[str]] = None) -> float:
    """
    Get a quick threshold recommendation.
    
    Args:
        trading_style: "intraday", "swing_trading", "position_trading"
        market_type: "low_volatility", "medium_volatility", "high_volatility"  
        feature_names: Optional list of feature names for analysis
        
    Returns:
        Recommended threshold value
    """
    advisor = ThresholdAdvisor()
    
    # Map string inputs to enums
    timeframe_map = {
        "intraday": TradingTimeframe.INTRADAY,
        "swing_trading": TradingTimeframe.SWING,
        "position_trading": TradingTimeframe.POSITION
    }
    
    volatility_map = {
        "low_volatility": MarketVolatility.LOW,
        "medium_volatility": MarketVolatility.MEDIUM,
        "high_volatility": MarketVolatility.HIGH
    }
    
    timeframe = timeframe_map.get(trading_style, TradingTimeframe.SWING)
    volatility = volatility_map.get(market_type, MarketVolatility.MEDIUM)
    
    # Analyze feature types if provided
    feature_types = None
    if feature_names:
        type_counts = advisor.analyze_feature_types(feature_names)
        # Get most common feature types
        feature_types = [ft for ft, count in type_counts.items() if count > 0 and ft != FeatureType.UNKNOWN]
    
    recommendation = advisor.recommend_threshold(
        trading_timeframe=timeframe,
        market_volatility=volatility,
        primary_feature_types=feature_types
    )
    
    return recommendation.recommended_threshold

def print_threshold_analysis(trading_style: str = "swing_trading",
                           market_type: str = "medium_volatility", 
                           feature_names: Optional[List[str]] = None):
    """Print detailed threshold analysis."""
    
    advisor = ThresholdAdvisor()
    
    # Map inputs
    timeframe_map = {
        "intraday": TradingTimeframe.INTRADAY,
        "swing_trading": TradingTimeframe.SWING, 
        "position_trading": TradingTimeframe.POSITION
    }
    
    volatility_map = {
        "low_volatility": MarketVolatility.LOW,
        "medium_volatility": MarketVolatility.MEDIUM,
        "high_volatility": MarketVolatility.HIGH
    }
    
    timeframe = timeframe_map.get(trading_style, TradingTimeframe.SWING)
    volatility = volatility_map.get(market_type, MarketVolatility.MEDIUM)
    
    # Analyze features
    feature_types = None
    if feature_names:
        type_counts = advisor.analyze_feature_types(feature_names)
        feature_types = [ft for ft, count in type_counts.items() if count > 0 and ft != FeatureType.UNKNOWN]
        
        print(f"📊 Feature Analysis ({len(feature_names)} features):")
        for ft, count in type_counts.items():
            if count > 0:
                print(f"   {ft.value}: {count}")
    
    # Get recommendation
    recommendation = advisor.recommend_threshold(
        trading_timeframe=timeframe,
        market_volatility=volatility,
        primary_feature_types=feature_types
    )
    
    # Print results
    print(f"\n🎯 Threshold Recommendation: {recommendation.recommended_threshold:.1%}")
    print(f"🔍 Confidence: {recommendation.confidence:.0%}")
    
    print(f"\n📋 Reasoning:")
    for reason in recommendation.reasoning:
        print(f"   • {reason}")
    
    print(f"\n🔄 Alternative Thresholds:")
    for name, threshold in recommendation.alternative_thresholds.items():
        print(f"   {name}: {threshold:.1%}")
    
    print(f"\n⚠️ Considerations:")
    for consideration in recommendation.considerations:
        print(f"   • {consideration}")