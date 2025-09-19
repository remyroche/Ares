#!/usr/bin/env python3
"""
Variance Threshold Analysis for Period Consolidation

This script analyzes appropriate variance thresholds for consolidating
long/short lookback periods based on financial market characteristics.
"""

from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_lookback_period_characteristics():
    """Analyze typical lookback period characteristics in financial markets."""
    
    print("📊 Financial Market Lookback Period Analysis")
    print("=" * 50)
    
    # Common financial indicators and their typical periods
    financial_indicators = {
        'SMA_short': [5, 9, 10, 12, 15, 20],
        'SMA_medium': [21, 26, 30, 50, 55, 60],
        'SMA_long': [100, 120, 150, 200, 250],
        'RSI': [14, 21, 30],
        'MACD': [12, 26, 9],  # Fast, slow, signal
        'Bollinger': [20, 21, 25],
        'Stochastic': [14, 21],
        'Williams_R': [14, 21],
        'ATR': [14, 20, 21],
        'ADX': [14, 21],
        'CCI': [20, 21],
        'Volume_MA': [10, 20, 30, 50]
    }
    
    print("\n🔍 Typical Financial Indicator Periods:")
    for indicator, periods in financial_indicators.items():
        print(f"   {indicator}: {periods}")
    
    return financial_indicators

def calculate_variance_scenarios(base_periods: List[int]) -> Dict[str, Any]:
    """Calculate variance scenarios for different threshold values."""
    
    scenarios = {}
    variance_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    print(f"\n📈 Variance Analysis for Base Periods: {base_periods}")
    print("-" * 60)
    
    for threshold in variance_thresholds:
        consolidation_count = 0
        examples = []
        
        for base_period in base_periods:
            # Simulate various long/short period combinations
            variance_factor = 0.01
            while variance_factor < 0.50:
                long_period = int(base_period * (1 + variance_factor))
                short_period = int(base_period * (1 - variance_factor))
                
                if short_period < 1:
                    continue
                
                # Calculate relative variance
                avg_period = (long_period + short_period) / 2
                if avg_period == 0:
                    continue
                    
                variance = abs(long_period - short_period) / avg_period
                
                if variance < threshold:
                    consolidation_count += 1
                    if len(examples) < 3:  # Keep first few examples
                        examples.append({
                            'base': base_period,
                            'long': long_period,
                            'short': short_period,
                            'variance': variance,
                            'consolidated': int(avg_period)
                        })
                
                variance_factor += 0.02
        
        scenarios[threshold] = {
            'consolidation_count': consolidation_count,
            'examples': examples
        }
        
        print(f"Threshold {threshold:4.1%}: {consolidation_count:3d} consolidations")
        for ex in examples[:2]:  # Show first 2 examples
            print(f"   Example: {ex['long']:2d}↔{ex['short']:2d} → {ex['consolidated']:2d} "
                  f"(variance: {ex['variance']:.1%})")
    
    return scenarios

def analyze_market_specific_thresholds():
    """Analyze appropriate thresholds for different market conditions."""
    
    print("\n🎯 Market-Specific Threshold Recommendations")
    print("=" * 50)
    
    market_conditions = {
        'High Volatility Markets': {
            'description': 'Crypto, emerging markets, individual stocks',
            'characteristics': 'Rapid price changes, high noise',
            'optimal_periods_differ': True,
            'suggested_threshold': 0.10,  # 10% - stricter
            'reasoning': 'Long/short strategies need different periods due to volatility asymmetry'
        },
        
        'Medium Volatility Markets': {
            'description': 'Major forex pairs, large-cap stocks, commodities',
            'characteristics': 'Moderate price changes, balanced trends',
            'optimal_periods_differ': False,
            'suggested_threshold': 0.25,  # 25% - moderate
            'reasoning': 'Similar optimal periods for both directions, allow more consolidation'
        },
        
        'Low Volatility Markets': {
            'description': 'Government bonds, stable currencies, utilities',
            'characteristics': 'Slow price changes, clear trends',
            'optimal_periods_differ': False,
            'suggested_threshold': 0.35,  # 35% - lenient
            'reasoning': 'Long/short periods often similar, maximize consolidation'
        },
        
        'Intraday Trading': {
            'description': 'High-frequency, scalping, day trading',
            'characteristics': 'Short periods (5-60 minutes), noise sensitive',
            'optimal_periods_differ': True,
            'suggested_threshold': 0.08,  # 8% - very strict
            'reasoning': 'Small period differences matter significantly'
        },
        
        'Swing Trading': {
            'description': 'Multi-day holds, trend following',
            'characteristics': 'Medium periods (hours to days)',
            'optimal_periods_differ': False,
            'suggested_threshold': 0.20,  # 20% - original default
            'reasoning': 'Balanced approach for medium-term strategies'
        },
        
        'Position Trading': {
            'description': 'Long-term holds, fundamental analysis',
            'characteristics': 'Long periods (days to months)',
            'optimal_periods_differ': False,
            'suggested_threshold': 0.30,  # 30% - lenient
            'reasoning': 'Long-term trends similar for both directions'
        }
    }
    
    for market_type, details in market_conditions.items():
        print(f"\n📋 {market_type}:")
        print(f"   Description: {details['description']}")
        print(f"   Characteristics: {details['characteristics']}")
        print(f"   Suggested Threshold: {details['suggested_threshold']:.0%}")
        print(f"   Reasoning: {details['reasoning']}")
    
    return market_conditions

def calculate_practical_examples():
    """Show practical examples of different thresholds."""
    
    print("\n💡 Practical Examples of Threshold Impact")
    print("=" * 50)
    
    # Real-world scenarios
    examples = [
        {'name': 'SMA Cross Strategy', 'long_period': 21, 'short_period': 19},
        {'name': 'RSI Overbought', 'long_period': 14, 'short_period': 16},
        {'name': 'Bollinger Bands', 'long_period': 20, 'short_period': 25},
        {'name': 'MACD Signal', 'long_period': 12, 'short_period': 9},
        {'name': 'ATR Volatility', 'long_period': 14, 'short_period': 21},
        {'name': 'Volume MA', 'long_period': 30, 'short_period': 20},
    ]
    
    thresholds_to_test = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    print(f"{'Strategy':<20} {'Long':<4} {'Short':<5} {'Var%':<5}", end="")
    for threshold in thresholds_to_test:
        print(f" {threshold:.0%}", end="")
    print()
    print("-" * 60)
    
    for example in examples:
        long_p = example['long_period']
        short_p = example['short_period']
        avg_p = (long_p + short_p) / 2
        variance = abs(long_p - short_p) / avg_p
        
        print(f"{example['name']:<20} {long_p:<4} {short_p:<5} {variance:<5.1%}", end="")
        
        for threshold in thresholds_to_test:
            consolidate = "✓" if variance < threshold else "✗"
            print(f"  {consolidate}", end="")
        print()
    
    print("\n✓ = Would consolidate, ✗ = Keep separate")

def suggest_adaptive_thresholds():
    """Suggest adaptive threshold strategies."""
    
    print("\n🧠 Adaptive Threshold Strategies")
    print("=" * 40)
    
    strategies = {
        'Feature-Based Adaptive': {
            'description': 'Different thresholds based on feature type',
            'implementation': {
                'trend_features': 0.25,  # SMA, EMA - trends similar both ways
                'momentum_features': 0.15,  # RSI, MACD - momentum differs
                'volatility_features': 0.10,  # ATR, BB - volatility asymmetric
                'volume_features': 0.30,  # Volume indicators - often similar
            }
        },
        
        'Period-Length Adaptive': {
            'description': 'Threshold based on absolute period length',
            'implementation': {
                'short_periods_1_10': 0.08,   # Very sensitive to small changes
                'medium_periods_11_50': 0.20,  # Balanced
                'long_periods_51_plus': 0.35,  # Less sensitive to changes
            }
        },
        
        'Performance-Based Adaptive': {
            'description': 'Threshold based on historical performance difference',
            'implementation': {
                'high_performance_diff': 0.05,  # Keep separate if big difference
                'medium_performance_diff': 0.15,  # Moderate threshold
                'low_performance_diff': 0.40,   # Consolidate aggressively
            }
        },
        
        'Market Regime Adaptive': {
            'description': 'Threshold based on current market conditions',
            'implementation': {
                'trending_market': 0.30,     # Trends work both ways
                'sideways_market': 0.15,     # Need precision
                'volatile_market': 0.08,     # Asymmetric behavior
            }
        }
    }
    
    for strategy_name, details in strategies.items():
        print(f"\n📊 {strategy_name}:")
        print(f"   {details['description']}")
        for key, threshold in details['implementation'].items():
            print(f"   - {key}: {threshold:.0%}")
    
    return strategies

def recommend_optimal_thresholds():
    """Provide final threshold recommendations."""
    
    print("\n🎯 Final Threshold Recommendations")
    print("=" * 40)
    
    recommendations = {
        'Conservative (Strict)': {
            'threshold': 0.10,
            'use_case': 'High-frequency trading, volatile markets, when precision matters',
            'trade_off': 'Fewer consolidations, more features, better precision'
        },
        
        'Balanced (Default)': {
            'threshold': 0.15,
            'use_case': 'General trading, mixed markets, balanced approach',
            'trade_off': 'Good balance between consolidation and precision'
        },
        
        'Moderate': {
            'threshold': 0.25,
            'use_case': 'Swing trading, stable markets, feature count concerns',
            'trade_off': 'More consolidations, fewer features, slight precision loss'
        },
        
        'Aggressive (Lenient)': {
            'threshold': 0.35,
            'use_case': 'Long-term trading, low volatility, maximum consolidation',
            'trade_off': 'Maximum consolidation, minimum features, potential precision loss'
        }
    }
    
    print("\n📋 Recommended Thresholds:")
    for approach, details in recommendations.items():
        print(f"\n🔹 {approach}: {details['threshold']:.0%}")
        print(f"   Use Case: {details['use_case']}")
        print(f"   Trade-off: {details['trade_off']}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • 20% threshold is reasonable for general use but not optimal")
    print(f"   • 15% provides better balance for most trading strategies")
    print(f"   • 10% recommended for high-frequency/volatile markets")
    print(f"   • 25-35% suitable for long-term/stable markets")
    print(f"   • Consider adaptive thresholds based on feature type")
    
    return recommendations

def main():
    """Run complete variance threshold analysis."""
    
    print("🔍 Variance Threshold Analysis for Period Consolidation")
    print("=" * 60)
    
    # 1. Analyze typical financial periods
    financial_indicators = analyze_lookback_period_characteristics()
    
    # 2. Calculate variance scenarios
    typical_periods = [5, 10, 14, 20, 21, 26, 30, 50]
    scenarios = calculate_variance_scenarios(typical_periods)
    
    # 3. Market-specific analysis
    market_conditions = analyze_market_specific_thresholds()
    
    # 4. Practical examples
    calculate_practical_examples()
    
    # 5. Adaptive strategies
    adaptive_strategies = suggest_adaptive_thresholds()
    
    # 6. Final recommendations
    recommendations = recommend_optimal_thresholds()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📋 Summary: 20% threshold should be adjusted based on:")
    print(f"   • Market volatility (lower threshold for volatile markets)")
    print(f"   • Trading timeframe (lower threshold for shorter timeframes)")
    print(f"   • Feature type (different thresholds for different indicators)")
    print(f"   • Performance requirements (lower threshold for precision)")
    
    return {
        'financial_indicators': financial_indicators,
        'scenarios': scenarios,
        'market_conditions': market_conditions,
        'adaptive_strategies': adaptive_strategies,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = main()