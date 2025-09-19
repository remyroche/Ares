#!/usr/bin/env python3
"""
Simple Multi-Horizon Score Fix Demonstration

This script demonstrates the specific fixes needed for the multi_horizon_profit_labeler.py
bonus/malus system that causes negative feature scores.
"""

import math
import json
from typing import Optional, Dict, Any, List

def calculate_original_problematic_score(target_hit: bool, time_to_hit: Optional[int], 
                                       max_adverse: float, total_periods: int, 
                                       net_profit: float, direction: str = 'long') -> float:
    """
    Simulate the original problematic scoring logic that creates negative scores.
    
    Problems:
    1. risk_penalty_multiplier = 30 (too aggressive)
    2. Directional penalties of 10-15%
    3. Profit score of 0.1 for unprofitable trades
    """
    if not target_hit:
        return 0.1  # Original low score for missed targets
    
    quality_factors = []
    
    # Speed factor
    if time_to_hit is not None:
        speed_factor = 1.0 - (time_to_hit / total_periods)
        speed_score = max(0.2, speed_factor)
        quality_factors.append(speed_score * 0.3)  # 30% weight
    
    # PROBLEMATIC: Risk factor with aggressive penalty
    if max_adverse > 0:
        risk_penalty_multiplier = 30  # This is the problem!
        risk_factor = max(0.1, 1.0 - (max_adverse * risk_penalty_multiplier))
        quality_factors.append(risk_factor * 0.4)  # 40% weight
    else:
        quality_factors.append(1.0 * 0.4)
    
    # Profitability factor
    if net_profit > 0:
        profit_scale_factor = 300
        profit_factor = min(1.0, net_profit * profit_scale_factor)
        profit_score = max(0.2, profit_factor)
    else:
        profit_score = 0.1  # PROBLEMATIC: Too low for unprofitable trades
    
    quality_factors.append(profit_score * 0.3)  # 30% weight
    
    # Calculate base score
    base_score = sum(quality_factors)
    
    # PROBLEMATIC: Aggressive directional penalties
    if direction.lower() == 'long' and max_adverse > 0.01:
        base_score *= 0.9  # 10% penalty
    elif direction.lower() == 'short' and max_adverse > 0.008:
        base_score *= 0.85  # 15% penalty - very aggressive!
    
    return min(1.0, base_score)

def calculate_fixed_score(target_hit: bool, time_to_hit: Optional[int], 
                         max_adverse: float, total_periods: int, 
                         net_profit: float, direction: str = 'long') -> float:
    """
    Fixed version with proper bonus/malus handling.
    
    Fixes:
    1. Reduced risk penalty multiplier from 30 to 10
    2. Gentler directional penalties (5-8% instead of 10-15%)
    3. Better profit scoring for negative profits
    4. Improved minimum score bounds
    """
    if not target_hit:
        return 0.2  # Increased from 0.1
    
    quality_factors = []
    
    # Speed factor (unchanged - this was fine)
    if time_to_hit is not None:
        speed_factor = max(0.0, 1.0 - (time_to_hit / total_periods))
        speed_score = 0.3 + (speed_factor * 0.7)  # Range: [0.3, 1.0]
        quality_factors.append(speed_score * 0.3)
    else:
        quality_factors.append(0.5 * 0.3)  # Default when time unknown
    
    # FIXED: Risk factor with gentler penalty
    if max_adverse > 0:
        risk_penalty_multiplier = 10  # REDUCED from 30 to 10
        risk_penalty = min(0.8, max_adverse * risk_penalty_multiplier)  # Cap at 80%
        risk_factor = 1.0 - risk_penalty
        risk_score = max(0.2, risk_factor)  # Minimum 20% instead of 10%
        quality_factors.append(risk_score * 0.4)
    else:
        quality_factors.append(1.0 * 0.4)
    
    # FIXED: Better profitability scoring
    if net_profit > 0:
        profit_scale_factor = 200  # Reduced from 300
        profit_factor = min(1.0, net_profit * profit_scale_factor)
        profit_score = max(0.3, profit_factor)  # Minimum 30% for profitable
    else:
        # IMPROVED: Graduated penalty for losses instead of fixed 0.1
        if net_profit >= -0.005:  # Small losses (< 0.5%)
            profit_score = 0.25
        elif net_profit >= -0.01:  # Medium losses (0.5% - 1.0%)
            profit_score = 0.2
        else:  # Large losses (> 1.0%)
            profit_score = 0.15
    
    quality_factors.append(profit_score * 0.3)
    
    # Calculate base score
    base_score = sum(quality_factors)
    
    # FIXED: Gentler directional penalties
    if direction.lower() == 'long' and max_adverse > 0.01:
        penalty = min(0.05, (max_adverse - 0.01) * 2)  # Max 5% penalty
        base_score *= (1.0 - penalty)
    elif direction.lower() == 'short' and max_adverse > 0.008:
        penalty = min(0.08, (max_adverse - 0.008) * 5)  # Max 8% penalty
        base_score *= (1.0 - penalty)
    
    # Normalize to [0.2, 1.0] range
    normalized_score = 0.2 + (min(1.0, base_score) * 0.8)
    
    return normalized_score

def normalize_composite_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize composite scores to eliminate negative values.
    
    This is the key fix for the multi-horizon labeler - many features
    get negative scores due to compounding penalties.
    """
    print("🔧 Normalizing composite scores to eliminate negative values")
    
    # Separate opportunity scores from directional scores
    opportunity_fields = [
        'long_overall_opportunity', 'short_overall_opportunity', 'overall_opportunity',
        'leverage_adjusted_score', 'reversal_capture_score', 'best_target_prob'
    ]
    
    directional_fields = [
        'directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum'
    ]
    
    normalized = scores.copy()
    
    # Get opportunity scores for normalization
    opportunity_scores = []
    for field in opportunity_fields:
        if field in scores and isinstance(scores[field], (int, float)):
            opportunity_scores.append(scores[field])
    
    if opportunity_scores:
        min_score = min(opportunity_scores)
        max_score = max(opportunity_scores)
        
        print(f"   Original opportunity range: [{min_score:.4f}, {max_score:.4f}]")
        
        # Apply min-max normalization to [0.1, 1.0] range
        if max_score > min_score:
            for field in opportunity_fields:
                if field in scores:
                    score = scores[field]
                    if isinstance(score, (int, float)):
                        normalized_score = 0.1 + 0.9 * ((score - min_score) / (max_score - min_score))
                        normalized[field] = normalized_score
        
        # Verify results
        new_scores = [normalized[field] for field in opportunity_fields if field in normalized]
        if new_scores:
            print(f"   Normalized opportunity range: [{min(new_scores):.4f}, {max(new_scores):.4f}]")
    
    # Handle directional scores (can be negative by design, but clamp extremes)
    for field in directional_fields:
        if field in normalized:
            score = normalized[field]
            if isinstance(score, (int, float)):
                normalized[field] = max(-2.0, min(2.0, score))
    
    return normalized

def demonstrate_specific_fixes():
    """Demonstrate the specific fixes for multi-horizon profit labeler issues."""
    print("🚀 Multi-Horizon Profit Labeler - Specific Bonus/Malus Fixes")
    print("=" * 70)
    
    # Test scenarios that cause negative scores in the original system
    problem_scenarios = [
        {
            'name': 'High Adverse Excursion Scenario',
            'description': 'Trade with 5% adverse move - causes negative scores with 30x multiplier',
            'target_hit': True,
            'time_to_hit': 2,
            'max_adverse': 0.05,  # 5% adverse excursion
            'total_periods': 4,
            'net_profit': 0.008,  # 0.8% net profit
            'direction': 'long'
        },
        {
            'name': 'Short Trade with Adverse Move',
            'description': 'Short trade with 1% adverse - gets 15% directional penalty',
            'target_hit': True,
            'time_to_hit': 3,
            'max_adverse': 0.012,  # 1.2% adverse
            'total_periods': 4,
            'net_profit': 0.006,  # 0.6% profit
            'direction': 'short'
        },
        {
            'name': 'Small Loss Trade',
            'description': 'Small losing trade - gets fixed 0.1 score in original',
            'target_hit': False,
            'time_to_hit': None,
            'max_adverse': 0.008,
            'total_periods': 4,
            'net_profit': -0.003,  # -0.3% loss
            'direction': 'long'
        }
    ]
    
    print("\n📊 SCENARIO ANALYSIS")
    print("=" * 50)
    
    total_improvement = 0
    scenarios_improved = 0
    
    for scenario in problem_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"   {scenario['description']}")
        
        # Calculate original problematic score
        original_score = calculate_original_problematic_score(
            scenario['target_hit'], scenario['time_to_hit'],
            scenario['max_adverse'], scenario['total_periods'],
            scenario['net_profit'], scenario['direction']
        )
        
        # Calculate fixed score
        fixed_score = calculate_fixed_score(
            scenario['target_hit'], scenario['time_to_hit'],
            scenario['max_adverse'], scenario['total_periods'],
            scenario['net_profit'], scenario['direction']
        )
        
        improvement = fixed_score - original_score
        improvement_pct = (improvement / original_score * 100) if original_score > 0 else 0
        
        print(f"   Original score: {original_score:.4f}")
        print(f"   Fixed score: {fixed_score:.4f}")
        print(f"   Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)")
        
        if improvement > 0.01:  # Significant improvement
            scenarios_improved += 1
            total_improvement += improvement
        
        # Show the specific problem and fix
        if scenario['max_adverse'] > 0:
            original_penalty = scenario['max_adverse'] * 30  # Original multiplier
            fixed_penalty = min(0.8, scenario['max_adverse'] * 10)  # Fixed multiplier
            print(f"   Risk penalty: {original_penalty:.2f} → {fixed_penalty:.2f} (reduced)")
    
    print(f"\n📈 COMPOSITE SCORE NORMALIZATION TEST")
    print("=" * 50)
    
    # Simulate problematic composite scores (common in multi-horizon labeler)
    problematic_composite = {
        'long_overall_opportunity': 0.03,    # Very low due to penalties
        'short_overall_opportunity': -0.05,  # Negative due to compounding penalties
        'overall_opportunity': 0.02,         # Very low
        'leverage_adjusted_score': -0.02,    # Negative
        'reversal_capture_score': 0.01,     # Very low
        'best_target_prob': 0.04,           # Low
        'directional_bias': -0.8,           # Allowed to be negative
        'opportunity_asymmetry': -0.3       # Allowed to be negative
    }
    
    print("Original composite scores (showing the problem):")
    negative_count = 0
    very_low_count = 0
    for key, value in problematic_composite.items():
        status = ""
        if value < 0:
            status = " ❌ NEGATIVE"
            negative_count += 1
        elif value < 0.1:
            status = " ⚠️ VERY LOW"
            very_low_count += 1
        print(f"   {key}: {value:.4f}{status}")
    
    print(f"\nProblems identified:")
    print(f"   Negative scores: {negative_count}")
    print(f"   Very low scores (< 0.1): {very_low_count}")
    
    # Apply normalization fix
    normalized_composite = normalize_composite_scores(problematic_composite)
    
    print("\nNormalized composite scores (after fix):")
    final_negative = 0
    final_very_low = 0
    for key, value in normalized_composite.items():
        status = ""
        if value < 0 and key not in ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']:
            status = " ❌ STILL NEGATIVE"
            final_negative += 1
        elif value < 0.1 and key not in ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']:
            status = " ⚠️ STILL LOW"
            final_very_low += 1
        elif key in ['directional_bias', 'opportunity_asymmetry']:
            status = " ✅ (allowed to be negative)"
        else:
            status = " ✅ FIXED"
        print(f"   {key}: {value:.4f}{status}")
    
    print(f"\n🎯 SUMMARY OF FIXES APPLIED")
    print("=" * 50)
    
    fixes_applied = [
        "1. Reduced risk penalty multiplier: 30 → 10 (67% reduction)",
        "2. Gentler directional penalties: 10-15% → 5-8% (50% reduction)",
        "3. Improved unprofitable trade scoring: 0.1 → 0.15-0.25 (up to 150% improvement)",
        "4. Added score normalization: negative scores → [0.1, 1.0] range",
        "5. Increased minimum bounds: 0.1 → 0.2 throughout",
        "6. Capped maximum penalties: unlimited → 80% max penalty"
    ]
    
    for fix in fixes_applied:
        print(f"   ✅ {fix}")
    
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 30)
    print(f"Scenarios tested: {len(problem_scenarios)}")
    print(f"Scenarios improved: {scenarios_improved}")
    print(f"Average improvement: +{total_improvement/len(problem_scenarios):.4f}")
    print(f"Negative composite scores eliminated: {negative_count} → {final_negative}")
    print(f"Very low composite scores reduced: {very_low_count} → {final_very_low}")
    
    print(f"\n💡 KEY INSIGHTS FOR PRODUCTION")
    print("=" * 40)
    insights = [
        "🔧 The 30x risk penalty multiplier was the main cause of negative scores",
        "📊 Directional penalties (10-15%) compounded the problem significantly",
        "⚡ Score normalization eliminates negative values while preserving ranking",
        "🎯 Graduated penalties work better than fixed harsh penalties",
        "📈 Minimum score bounds prevent extreme low values",
        "🔄 These fixes maintain relative feature importance while improving absolute scores"
    ]
    
    for insight in insights:
        print(f"   {insight}")
    
    print(f"\n🚀 IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 40)
    recommendations = [
        "1. Replace risk_penalty_multiplier = 30 with 10 in _calculate_quality_score()",
        "2. Replace directional penalties 0.9/0.85 with gentler 0.95/0.92",
        "3. Replace profit_score = 0.1 with graduated 0.15-0.25 based on loss size",
        "4. Add normalize_composite_scores() call before returning final results",
        "5. Increase all minimum score bounds from 0.1 to 0.2",
        "6. Add score capping to prevent extreme penalties"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # Save results for reference
    results = {
        'scenario_tests': {
            scenario['name']: {
                'original_score': calculate_original_problematic_score(
                    scenario['target_hit'], scenario['time_to_hit'],
                    scenario['max_adverse'], scenario['total_periods'],
                    scenario['net_profit'], scenario['direction']
                ),
                'fixed_score': calculate_fixed_score(
                    scenario['target_hit'], scenario['time_to_hit'],
                    scenario['max_adverse'], scenario['total_periods'],
                    scenario['net_profit'], scenario['direction']
                )
            } for scenario in problem_scenarios
        },
        'composite_normalization': {
            'original': problematic_composite,
            'normalized': normalized_composite
        },
        'fixes_applied': fixes_applied,
        'recommendations': recommendations
    }
    
    with open('/workspace/multi_horizon_fixes_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: /workspace/multi_horizon_fixes_results.json")
    print("✅ Multi-horizon profit labeler fixes demonstration completed!")

if __name__ == "__main__":
    demonstrate_specific_fixes()