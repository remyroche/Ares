#!/usr/bin/env python3
"""
Fix Negative Scores in Multi-Horizon Profit Labeler

This script fixes the specific bonus/malus issues in the multi_horizon_profit_labeler.py
that cause negative or excessively low feature scores.
"""

import numpy as np
from typing import Optional, Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiHorizonScoreFixer:
    """
    Fix the scoring issues in Multi-Horizon Profit Labeler.
    
    Problems identified:
    1. Risk penalty multiplier (30) is too aggressive - causes negative scores
    2. Profit scores of 0.1 for unprofitable trades are too low
    3. Directional penalties (0.9, 0.85) compound the problem
    4. No normalization of final scores
    """
    
    def __init__(self):
        self.logger = logger
        
    def calculate_fixed_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                                    max_adverse: float, total_periods: int, net_profit: float,
                                    speed_weight: float = 0.3, risk_weight: float = 0.4, 
                                    profitability_weight: float = 0.3) -> float:
        """
        Fixed version of _calculate_quality_score with proper bonus/malus handling.
        
        Key fixes:
        1. Reduced risk penalty multiplier from 30 to 10
        2. Improved profit scoring for negative profits
        3. Better normalization and score bounds
        4. Smoother bonus/malus transitions
        """
        if not target_hit:
            return 0.2  # Increased from 0.1 for model uncertainty
        
        quality_factors = []
        
        # 1. Speed factor (faster = better) - 30% weight
        if time_to_hit is not None:
            # Smoother speed scoring
            speed_factor = max(0.0, 1.0 - (time_to_hit / total_periods))
            speed_score = 0.3 + (speed_factor * 0.7)  # Range: [0.3, 1.0]
            quality_factors.append(speed_score * speed_weight)
            
            # Bonus for very fast moves (within 50% of time window)
            if time_to_hit < total_periods * 0.5:
                speed_bonus = min(0.1, (0.5 - time_to_hit/total_periods) * 0.2)
                quality_factors.append(speed_bonus)
        else:
            # Default speed score when time is unknown
            quality_factors.append(0.5 * speed_weight)
        
        # 2. FIXED Risk factor (lower adverse excursion = better) - 40% weight
        if max_adverse > 0:
            # REDUCED penalty multiplier from 30 to 10 to prevent negative scores
            risk_penalty_multiplier = 10  # Much more reasonable penalty
            
            # Smoother risk penalty curve
            risk_penalty = min(0.8, max_adverse * risk_penalty_multiplier)  # Cap penalty at 80%
            risk_factor = 1.0 - risk_penalty
            risk_score = max(0.2, risk_factor)  # Minimum 20% score instead of 10%
        else:
            risk_score = 1.0  # Perfect score if no adverse excursion
        
        quality_factors.append(risk_score * risk_weight)
        
        # 3. FIXED Profitability factor - 30% weight
        if net_profit > 0:
            # Scale net profit for short-term moves
            profit_scale_factor = 200  # Reduced from 300 for smoother scaling
            profit_factor = min(1.0, net_profit * profit_scale_factor)
            profit_score = max(0.3, profit_factor)  # Minimum 30% for profitable trades
            
            # Bonus for high profitability relative to risk
            if max_adverse > 0:
                profit_risk_ratio = net_profit / max_adverse
                if profit_risk_ratio > 1.5:  # Lowered threshold from 2.0 to 1.5
                    profit_bonus = min(0.15, (profit_risk_ratio - 1.5) * 0.08)
                    quality_factors.append(profit_bonus)
        else:
            # IMPROVED handling of unprofitable trades
            # Instead of fixed 0.1, use graduated penalty based on loss size
            if net_profit >= -0.005:  # Small losses (< 0.5%)
                profit_score = 0.25  # Increased from 0.1
            elif net_profit >= -0.01:  # Medium losses (0.5% - 1.0%)
                profit_score = 0.2
            else:  # Large losses (> 1.0%)
                profit_score = 0.15
        
        quality_factors.append(profit_score * profitability_weight)
        
        # Calculate total with proper bounds
        total_quality = np.sum(quality_factors)
        
        # IMPROVED: Normalize to [0.2, 1.0] range instead of capping at 1.0
        normalized_quality = 0.2 + (min(1.0, total_quality) * 0.8)
        
        return normalized_quality
    
    def calculate_fixed_directional_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                                                max_adverse: float, total_periods: int, 
                                                net_profit: float, direction: str) -> float:
        """
        Fixed version of _calculate_directional_quality_score with gentler penalties.
        
        Key fixes:
        1. Reduced directional penalties from 10%/15% to 5%/8%
        2. Added directional bonuses to balance penalties
        3. Smoother penalty curves
        """
        if not target_hit:
            return 0.2  # Increased base score
        
        # Start with fixed base quality score
        base_quality = self.calculate_fixed_quality_score(
            target_hit, time_to_hit, max_adverse, total_periods, net_profit
        )
        
        # FIXED: Gentler directional adjustments
        directional_multiplier = 1.0
        
        if direction.lower() == 'long':
            # Long trades bonuses and penalties
            if time_to_hit is not None and time_to_hit < total_periods * 0.3:
                directional_multiplier *= 1.05  # Reduced from 1.1 to 1.05 (5% bonus)
            
            # GENTLER adverse excursion penalty
            if max_adverse > 0.01:  # More than 1% adverse for longs
                penalty = min(0.05, (max_adverse - 0.01) * 2)  # Max 5% penalty
                directional_multiplier *= (1.0 - penalty)
                
        else:  # direction == 'short'
            # Short trades bonuses and penalties
            if time_to_hit is not None and time_to_hit > total_periods * 0.5:
                directional_multiplier *= 1.03  # Reduced from 1.05 to 1.03 (3% bonus)
            
            # GENTLER adverse excursion penalty for shorts
            if max_adverse > 0.008:  # More than 0.8% adverse for shorts
                penalty = min(0.08, (max_adverse - 0.008) * 5)  # Max 8% penalty
                directional_multiplier *= (1.0 - penalty)
        
        # Apply directional adjustment with bounds
        adjusted_quality = base_quality * directional_multiplier
        
        # Ensure result stays within reasonable bounds
        return max(0.15, min(1.0, adjusted_quality))
    
    def calculate_fixed_reversal_capture_score(self, probability_scores: Dict[str, float], 
                                             sample_labels: Dict[str, float]) -> float:
        """
        Fixed version of _calculate_reversal_capture_score with gentler penalties.
        
        Key fixes:
        1. Reduced adverse penalty multiplier from 50 to 20
        2. Improved minimum score bounds
        3. Better factor weighting
        """
        reversal_factors = []
        
        # Factor 1: Speed of opportunity (faster = better for reversals)
        time_values = [v for k, v in sample_labels.items() if k.endswith('_time_to_hit') and v >= 0]
        if time_values:
            avg_time = np.mean(time_values)
            # Smoother speed factor calculation
            speed_factor = max(0.2, 1.0 - (avg_time / 4.0))  # Increased minimum from 0.1
            reversal_factors.append(speed_factor * 0.4)  # 40% weight
        
        # Factor 2: FIXED adverse excursion penalty
        adverse_values = [v for k, v in sample_labels.items() if k.endswith('_max_adverse')]
        if adverse_values:
            avg_adverse = np.mean(adverse_values)
            # REDUCED penalty multiplier from 50 to 20
            clean_factor = max(0.2, 1.0 - (avg_adverse * 20))  # Gentler penalty
            reversal_factors.append(clean_factor * 0.3)  # 30% weight
        
        # Factor 3: Immediate vs short-term probability ratio
        immediate_prob = probability_scores.get('micro_immediate', 0.0) + probability_scores.get('small_immediate', 0.0)
        short_prob = probability_scores.get('micro_short', 0.0) + probability_scores.get('small_short', 0.0)
        
        if short_prob > 0:
            ratio_factor = min(1.0, immediate_prob / short_prob)
            reversal_factors.append(ratio_factor * 0.3)  # 30% weight
        else:
            # Add default factor when no short-term probabilities
            reversal_factors.append(0.5 * 0.3)
        
        # Calculate final score with improved bounds
        final_score = np.sum(reversal_factors) if reversal_factors else 0.2
        return max(0.15, min(1.0, final_score))  # Improved bounds: [0.15, 1.0]
    
    def normalize_composite_scores(self, composite_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize all composite scores to prevent negative values and improve distribution.
        
        This addresses the core issue where multiple penalties compound to create
        very low or negative scores.
        """
        self.logger.info("🔧 Normalizing composite scores to fix negative values")
        
        normalized_scores = composite_scores.copy()
        
        # List of score fields that should be normalized
        score_fields = [
            'long_overall_opportunity', 'short_overall_opportunity', 'overall_opportunity',
            'long_immediate_opportunity', 'short_immediate_opportunity',
            'long_short_opportunity', 'short_short_opportunity',
            'leverage_adjusted_score', 'long_leverage_adjusted_score', 'short_leverage_adjusted_score',
            'best_target_prob', 'net_profitability_score', 'reversal_capture_score',
            'long_directional_strength', 'short_directional_strength'
        ]
        
        # Apply min-max normalization to opportunity scores
        opportunity_scores = []
        for field in score_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not np.isnan(score):
                    opportunity_scores.append(score)
        
        if opportunity_scores:
            min_score = min(opportunity_scores)
            max_score = max(opportunity_scores)
            
            self.logger.info(f"   Original score range: [{min_score:.4f}, {max_score:.4f}]")
            
            # Normalize opportunity scores to [0.1, 1.0] range
            if max_score > min_score:
                for field in score_fields:
                    if field in normalized_scores:
                        score = normalized_scores[field]
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            normalized_score = 0.1 + 0.9 * ((score - min_score) / (max_score - min_score))
                            normalized_scores[field] = normalized_score
            
            # Verify normalization
            new_opportunity_scores = []
            for field in score_fields:
                if field in normalized_scores:
                    score = normalized_scores[field]
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        new_opportunity_scores.append(score)
            
            if new_opportunity_scores:
                new_min = min(new_opportunity_scores)
                new_max = max(new_opportunity_scores)
                self.logger.info(f"   Normalized score range: [{new_min:.4f}, {new_max:.4f}]")
        
        # Handle directional scores separately (they can be negative by design)
        directional_fields = ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']
        for field in directional_fields:
            if field in normalized_scores:
                # These fields are allowed to be negative, but clamp extreme values
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not np.isnan(score):
                    normalized_scores[field] = max(-2.0, min(2.0, score))
        
        # Ensure confidence and consistency scores are in [0, 1] range
        bounded_fields = ['directional_confidence', 'long_directional_consistency', 'short_directional_consistency']
        for field in bounded_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not np.isnan(score):
                    normalized_scores[field] = max(0.0, min(1.0, score))
        
        return normalized_scores
    
    def generate_fix_report(self, original_scores: Dict[str, float], 
                          fixed_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate a report showing the improvements made."""
        
        # Find scores that were improved
        improvements = {}
        for key in original_scores:
            if key in fixed_scores:
                original = original_scores[key]
                fixed = fixed_scores[key]
                if isinstance(original, (int, float)) and isinstance(fixed, (int, float)):
                    if not (np.isnan(original) or np.isnan(fixed)):
                        improvement = fixed - original
                        if improvement > 0.01:  # Significant improvement
                            improvements[key] = {
                                'original': original,
                                'fixed': fixed,
                                'improvement': improvement
                            }
        
        # Calculate statistics
        original_values = [v for v in original_scores.values() 
                          if isinstance(v, (int, float)) and not np.isnan(v)]
        fixed_values = [v for v in fixed_scores.values() 
                       if isinstance(v, (int, float)) and not np.isnan(v)]
        
        report = {
            'summary': {
                'total_scores': len(original_values),
                'improved_scores': len(improvements),
                'improvement_percentage': len(improvements) / len(original_values) * 100 if original_values else 0
            },
            'statistics': {
                'original': {
                    'min': min(original_values) if original_values else 0,
                    'max': max(original_values) if original_values else 0,
                    'mean': np.mean(original_values) if original_values else 0,
                    'negative_count': sum(1 for v in original_values if v < 0)
                },
                'fixed': {
                    'min': min(fixed_values) if fixed_values else 0,
                    'max': max(fixed_values) if fixed_values else 0,
                    'mean': np.mean(fixed_values) if fixed_values else 0,
                    'negative_count': sum(1 for v in fixed_values if v < 0)
                }
            },
            'top_improvements': sorted(improvements.items(), 
                                     key=lambda x: x[1]['improvement'], 
                                     reverse=True)[:10],
            'fixes_applied': [
                'Reduced risk penalty multiplier from 30 to 10',
                'Improved profit scoring for negative profits',
                'Gentler directional penalties (5-8% instead of 10-15%)',
                'Reduced adverse excursion penalty in reversal capture',
                'Applied min-max normalization to all opportunity scores',
                'Increased minimum score bounds throughout'
            ]
        }
        
        return report

def demonstrate_multi_horizon_fixes():
    """Demonstrate the multi-horizon profit labeler fixes."""
    logger.info("🚀 Multi-Horizon Profit Labeler Score Fixes")
    logger.info("=" * 60)
    
    # Create sample problematic scenarios
    test_scenarios = [
        {
            'name': 'High Adverse Excursion',
            'target_hit': True,
            'time_to_hit': 2,
            'max_adverse': 0.05,  # 5% adverse - causes negative scores with multiplier 30
            'total_periods': 4,
            'net_profit': 0.008,  # 0.8% profit
            'direction': 'long'
        },
        {
            'name': 'Unprofitable Trade',
            'target_hit': False,
            'time_to_hit': None,
            'max_adverse': 0.02,
            'total_periods': 4,
            'net_profit': -0.003,  # -0.3% loss
            'direction': 'short'
        },
        {
            'name': 'Slow Profitable Trade',
            'target_hit': True,
            'time_to_hit': 3,
            'max_adverse': 0.01,
            'total_periods': 4,
            'net_profit': 0.015,  # 1.5% profit
            'direction': 'long'
        }
    ]
    
    fixer = MultiHorizonScoreFixer()
    
    print("\n📊 SCENARIO TESTING")
    print("=" * 40)
    
    original_scores = {}
    fixed_scores = {}
    
    for scenario in test_scenarios:
        name = scenario['name']
        
        # Simulate original problematic scoring
        if scenario['target_hit']:
            # Original aggressive penalty
            if scenario['max_adverse'] > 0:
                risk_penalty = 1.0 - (scenario['max_adverse'] * 30)  # Original multiplier
                original_score = max(0.1, risk_penalty)
            else:
                original_score = 1.0
            
            # Apply directional penalties
            if scenario['direction'] == 'long' and scenario['max_adverse'] > 0.01:
                original_score *= 0.9  # 10% penalty
            elif scenario['direction'] == 'short' and scenario['max_adverse'] > 0.008:
                original_score *= 0.85  # 15% penalty
        else:
            original_score = 0.1  # Original low score for missed targets
        
        # Calculate fixed score
        fixed_score = fixer.calculate_fixed_directional_quality_score(
            scenario['target_hit'], scenario['time_to_hit'], 
            scenario['max_adverse'], scenario['total_periods'],
            scenario['net_profit'], scenario['direction']
        )
        
        original_scores[name] = original_score
        fixed_scores[name] = fixed_score
        
        improvement = fixed_score - original_score
        print(f"\n{name}:")
        print(f"   Original score: {original_score:.4f}")
        print(f"   Fixed score: {fixed_score:.4f}")
        print(f"   Improvement: +{improvement:.4f} ({improvement/original_score*100:.1f}%)")
    
    # Test composite score normalization
    print(f"\n📈 COMPOSITE SCORE NORMALIZATION")
    print("=" * 40)
    
    sample_composite_scores = {
        'long_overall_opportunity': 0.05,  # Very low
        'short_overall_opportunity': -0.1,  # Negative!
        'overall_opportunity': 0.08,
        'leverage_adjusted_score': 0.03,
        'reversal_capture_score': -0.05,  # Negative!
        'directional_bias': -0.3,  # Allowed to be negative
        'directional_confidence': 1.2  # Over 1.0
    }
    
    print("Original composite scores:")
    for key, value in sample_composite_scores.items():
        print(f"   {key}: {value:.4f}")
    
    normalized_scores = fixer.normalize_composite_scores(sample_composite_scores)
    
    print("\nNormalized composite scores:")
    for key, value in normalized_scores.items():
        print(f"   {key}: {value:.4f}")
    
    # Generate comprehensive report
    all_original = {**original_scores, **sample_composite_scores}
    all_fixed = {**fixed_scores, **normalized_scores}
    
    report = fixer.generate_fix_report(all_original, all_fixed)
    
    print(f"\n📋 COMPREHENSIVE REPORT")
    print("=" * 40)
    print(f"Total scores analyzed: {report['summary']['total_scores']}")
    print(f"Scores improved: {report['summary']['improved_scores']}")
    print(f"Improvement rate: {report['summary']['improvement_percentage']:.1f}%")
    
    print(f"\nOriginal stats:")
    print(f"   Range: [{report['statistics']['original']['min']:.4f}, {report['statistics']['original']['max']:.4f}]")
    print(f"   Mean: {report['statistics']['original']['mean']:.4f}")
    print(f"   Negative count: {report['statistics']['original']['negative_count']}")
    
    print(f"\nFixed stats:")
    print(f"   Range: [{report['statistics']['fixed']['min']:.4f}, {report['statistics']['fixed']['max']:.4f}]")
    print(f"   Mean: {report['statistics']['fixed']['mean']:.4f}")
    print(f"   Negative count: {report['statistics']['fixed']['negative_count']}")
    
    print(f"\n🔧 Fixes Applied:")
    for fix in report['fixes_applied']:
        print(f"   ✅ {fix}")
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"   ✅ Eliminated {report['statistics']['original']['negative_count']} negative scores")
    print(f"   ✅ Improved {report['summary']['improved_scores']} scores significantly")
    print(f"   ✅ Raised minimum score bounds from 0.1 to 0.15-0.2")
    print(f"   ✅ Applied gentler penalty curves throughout")
    print(f"   ✅ Added proper score normalization")
    
    # Save results
    import json
    output_data = {
        'scenario_results': {
            'original_scores': original_scores,
            'fixed_scores': fixed_scores
        },
        'composite_normalization': {
            'original': sample_composite_scores,
            'normalized': normalized_scores
        },
        'report': report
    }
    
    with open('/workspace/multi_horizon_score_fixes.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: /workspace/multi_horizon_score_fixes.json")
    print("✅ Multi-horizon score fixing demonstration completed!")

if __name__ == "__main__":
    demonstrate_multi_horizon_fixes()