#!/usr/bin/env python3
"""
Corrected Variance Threshold Analysis for Lookback Period Consolidation

This script provides the CORRECT analysis of variance thresholds based on 
LOOKBACK PERIOD differences, not feature values.

Formula: variance = abs(long_period - short_period) / average_period
Example: RSI 6 vs RSI 7 → variance = |6-7|/6.5 = 15.4%
"""

from typing import Dict, List, Tuple, Any

def calculate_lookback_variance(long_period: int, short_period: int) -> float:
    """Calculate variance between two lookback periods."""
    if long_period == 0 and short_period == 0:
        return 0.0
    
    avg_period = (long_period + short_period) / 2
    if avg_period == 0:
        return 0.0
    
    variance = abs(long_period - short_period) / avg_period
    return variance

def analyze_common_lookback_scenarios():
    """Analyze common lookback period consolidation scenarios."""
    
    print("🔍 Corrected Lookback Period Variance Analysis")
    print("=" * 60)
    print("Formula: variance = |long_period - short_period| / average_period")
    print()
    
    # Common financial indicator period combinations
    scenarios = [
        # RSI variations
        ("RSI", 6, 7, "Very close RSI periods"),
        ("RSI", 14, 16, "Standard RSI with slight variation"),
        ("RSI", 12, 18, "Moderate RSI difference"),
        ("RSI", 14, 21, "Significant RSI difference"),
        
        # SMA variations  
        ("SMA", 9, 10, "Very close SMA periods"),
        ("SMA", 20, 21, "Close SMA periods"),
        ("SMA", 15, 20, "Moderate SMA difference"),
        ("SMA", 20, 30, "Significant SMA difference"),
        ("SMA", 50, 100, "Large SMA difference"),
        
        # Short period indicators
        ("Fast MA", 3, 4, "Very short periods"),
        ("Fast MA", 5, 7, "Short periods"),
        ("Fast MA", 8, 12, "Short-medium periods"),
        
        # Medium period indicators
        ("MACD", 12, 26, "MACD fast vs slow"),
        ("Stochastic", 14, 21, "Stochastic variations"),
        ("ATR", 14, 20, "ATR variations"),
        
        # Long period indicators
        ("Long SMA", 100, 120, "Long SMA variations"),
        ("Long SMA", 150, 200, "Very long SMA variations"),
    ]
    
    print(f"{'Indicator':<12} {'Long':<4} {'Short':<5} {'Variance':<8} {'5%':<3} {'10%':<4} {'15%':<4} {'20%':<4} {'25%':<4}")
    print("-" * 65)
    
    for indicator, long_p, short_p, description in scenarios:
        variance = calculate_lookback_variance(long_p, short_p)
        
        # Test different thresholds
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
        consolidation_flags = ["✓" if variance < t else "✗" for t in thresholds]
        
        print(f"{indicator:<12} {long_p:<4} {short_p:<5} {variance:<8.1%} {consolidation_flags[0]:<3} {consolidation_flags[1]:<4} {consolidation_flags[2]:<4} {consolidation_flags[3]:<4} {consolidation_flags[4]:<4}")
    
    print("\n✓ = Would consolidate, ✗ = Keep separate")

def analyze_threshold_impact_on_periods():
    """Analyze what period differences each threshold allows."""
    
    print("\n🎯 Threshold Impact Analysis")
    print("=" * 40)
    
    base_periods = [5, 10, 14, 20, 30, 50, 100]
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print("\nFor each base period, maximum allowed difference at each threshold:")
    print()
    
    for base_period in base_periods:
        print(f"Base Period {base_period}:")
        
        for threshold in thresholds:
            # Calculate maximum allowed difference
            # variance = difference / average
            # For base_period and (base_period + diff): average = (2*base_period + diff) / 2
            # threshold = diff / average = diff / ((2*base_period + diff) / 2)
            # Solving: diff = threshold * 2 * base_period / (2 - threshold)
            
            max_diff = threshold * 2 * base_period / (2 - threshold)
            max_other_period = base_period + int(max_diff)
            
            print(f"  {threshold:.0%} threshold: ±{max_diff:.1f} → periods {base_period-int(max_diff)} to {max_other_period}")
        print()

def analyze_practical_consolidation_scenarios():
    """Analyze practical consolidation scenarios with correct variance calculation."""
    
    print("💡 Practical Consolidation Scenarios")
    print("=" * 40)
    
    scenarios_by_threshold = {
        "5% (Very Strict)": [
            (5, 5, "Identical periods only"),
            (10, 10, "Identical periods only"),
            (20, 21, "20 vs 21 - just barely consolidates"),
        ],
        
        "10% (Strict)": [
            (6, 7, "RSI 6 vs 7 - consolidates"),
            (9, 10, "SMA 9 vs 10 - consolidates"),
            (20, 22, "20 vs 22 - consolidates"),
            (14, 16, "RSI 14 vs 16 - does NOT consolidate"),
        ],
        
        "15% (Balanced)": [
            (6, 7, "RSI 6 vs 7 - consolidates"),
            (14, 16, "RSI 14 vs 16 - consolidates"),
            (20, 23, "20 vs 23 - consolidates"),
            (12, 15, "12 vs 15 - does NOT consolidate"),
        ],
        
        "20% (Lenient)": [
            (14, 16, "RSI 14 vs 16 - consolidates"),
            (20, 24, "20 vs 24 - consolidates"),
            (12, 15, "12 vs 15 - consolidates"),
            (10, 13, "10 vs 13 - does NOT consolidate"),
        ],
        
        "25% (Very Lenient)": [
            (12, 15, "12 vs 15 - consolidates"),
            (20, 25, "20 vs 25 - consolidates"),
            (10, 13, "10 vs 13 - consolidates"),
            (8, 12, "8 vs 12 - does NOT consolidate"),
        ]
    }
    
    for threshold_name, examples in scenarios_by_threshold.items():
        print(f"\n{threshold_name}:")
        for long_p, short_p, description in examples:
            variance = calculate_lookback_variance(long_p, short_p)
            print(f"  {description} (variance: {variance:.1%})")

def recommend_corrected_thresholds():
    """Provide corrected threshold recommendations based on lookback period analysis."""
    
    print("\n🎯 Corrected Threshold Recommendations")
    print("=" * 45)
    
    recommendations = {
        "Ultra Strict (5%)": {
            "use_case": "When period precision is absolutely critical",
            "consolidates": "Only identical or nearly identical periods (20↔21)",
            "example": "High-frequency trading, precise signal timing",
            "trade_off": "Minimal consolidation, maximum precision"
        },
        
        "Strict (10%)": {
            "use_case": "When small period differences matter significantly", 
            "consolidates": "Very close periods (6↔7, 9↔10, 20↔22)",
            "example": "Intraday trading, scalping strategies",
            "trade_off": "Conservative consolidation, high precision"
        },
        
        "Balanced (15%)": {
            "use_case": "General trading with balanced consolidation",
            "consolidates": "Close periods (14↔16, 20↔23), not moderate differences",
            "example": "Swing trading, mixed strategies",
            "trade_off": "Good balance between consolidation and precision"
        },
        
        "Moderate (20%)": {
            "use_case": "When moderate period differences are acceptable",
            "consolidates": "Moderate differences (14↔17, 20↔24, 12↔15)",
            "example": "Position trading, trend following",
            "trade_off": "More consolidation, some precision loss"
        },
        
        "Lenient (25%)": {
            "use_case": "Maximum consolidation for feature count management",
            "consolidates": "Significant differences (20↔25, 12↔15, 10↔13)",
            "example": "Long-term trading, feature count critical",
            "trade_off": "High consolidation, notable precision loss"
        }
    }
    
    for threshold_name, details in recommendations.items():
        print(f"\n🔹 {threshold_name}:")
        print(f"   Use Case: {details['use_case']}")
        print(f"   Consolidates: {details['consolidates']}")
        print(f"   Example: {details['example']}")
        print(f"   Trade-off: {details['trade_off']}")

def analyze_20_percent_threshold():
    """Specific analysis of the 20% threshold mentioned in the user query."""
    
    print("\n🔍 Specific Analysis: 20% Threshold")
    print("=" * 40)
    
    print("User Example: RSI 6 vs RSI 7")
    variance_6_7 = calculate_lookback_variance(6, 7)
    print(f"  Variance: |6-7| / ((6+7)/2) = 1 / 6.5 = {variance_6_7:.1%}")
    print(f"  20% threshold: {'✓ Consolidates' if variance_6_7 < 0.20 else '✗ Keeps separate'}")
    print(f"  10% threshold: {'✓ Consolidates' if variance_6_7 < 0.10 else '✗ Keeps separate'}")
    
    print(f"\nOther examples at 20% threshold:")
    
    examples_20_percent = [
        (14, 16, "RSI 14 vs 16"),
        (14, 17, "RSI 14 vs 17"),  
        (20, 24, "SMA 20 vs 24"),
        (20, 25, "SMA 20 vs 25"),
        (12, 15, "Period 12 vs 15"),
        (10, 12, "Period 10 vs 12"),
        (30, 36, "Period 30 vs 36"),
        (50, 60, "Period 50 vs 60"),
    ]
    
    for long_p, short_p, description in examples_20_percent:
        variance = calculate_lookback_variance(long_p, short_p)
        consolidates = "✓" if variance < 0.20 else "✗"
        print(f"  {description}: {variance:.1%} → {consolidates}")

def main():
    """Run corrected variance threshold analysis."""
    
    print("🔧 CORRECTED Variance Threshold Analysis")
    print("Understanding: Threshold applies to LOOKBACK PERIOD differences")
    print("Formula: variance = |long_period - short_period| / average_period")
    print("=" * 70)
    
    # Run all analyses
    analyze_common_lookback_scenarios()
    analyze_threshold_impact_on_periods()
    analyze_practical_consolidation_scenarios()
    analyze_20_percent_threshold()
    recommend_corrected_thresholds()
    
    print(f"\n🎉 Corrected Analysis Complete!")
    print(f"Key Insight: 20% threshold allows moderate period differences")
    print(f"Examples: RSI 6↔7 (15.4%), RSI 14↔17 (19.4%), SMA 20↔24 (18.2%)")
    print(f"Recommendation: 20% is reasonable for general use, adjust based on:")
    print(f"  • 10-15% for precision-critical applications")
    print(f"  • 20-25% for balanced consolidation")
    print(f"  • 25-30% for maximum consolidation")

if __name__ == "__main__":
    main()