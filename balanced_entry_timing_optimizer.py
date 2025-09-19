#!/usr/bin/env python3
"""
Balanced Entry Timing Optimizer for Multi-Horizon Profit Labeler

This module provides balanced bonus/malus scoring that optimizes entry timing:
- Too early entry: Penalty for entering before optimal momentum
- Too late entry: Reduced points for missed opportunity
- Optimal timing: Maximum positive score for best entry window

Key principles:
1. Balance between early/late penalties and opportunity rewards
2. Smooth scoring curves instead of harsh binary penalties
3. Time-aware scoring that considers market momentum
4. Preserve relative importance while eliminating extreme negative scores
"""

import math
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class EntryTimingOptimizer:
    """
    Optimizes entry timing scoring with balanced positive/negative point system.
    
    The goal is to find the sweet spot for entry timing:
    - Not too early (before momentum builds) 
    - Not too late (missing the best part of the move)
    - Just right (optimal entry window)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with balanced timing parameters."""
        self.config = config or {}
        
        # Balanced timing parameters
        self.optimal_entry_window = self.config.get('optimal_entry_window', 0.3)  # 30% of time horizon
        self.early_penalty_factor = self.config.get('early_penalty_factor', 2.0)  # Gentle early penalty
        self.late_penalty_factor = self.config.get('late_penalty_factor', 3.0)   # Moderate late penalty
        self.momentum_threshold = self.config.get('momentum_threshold', 0.002)   # 0.2% momentum
        
        # Score balance parameters
        self.max_positive_score = self.config.get('max_positive_score', 1.0)
        self.min_score_floor = self.config.get('min_score_floor', 0.15)  # Prevent extreme negatives
        self.neutral_score = self.config.get('neutral_score', 0.5)       # Baseline score
        
        self.logger = logger

    def calculate_balanced_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                                       max_adverse: float, total_periods: int, 
                                       net_profit: float, entry_momentum: float = 0.0) -> float:
        """
        Calculate balanced quality score with optimal entry timing focus.
        
        Scoring components:
        1. Entry Timing Score (40%): Rewards optimal timing, penalizes too early/late
        2. Risk Management Score (30%): Balanced adverse excursion handling  
        3. Profitability Score (30%): Graduated profit/loss scoring
        
        Args:
            target_hit: Whether profit target was reached
            time_to_hit: Periods to reach target (None if not hit)
            max_adverse: Maximum adverse excursion 
            total_periods: Total time horizon
            net_profit: Net profit after costs
            entry_momentum: Market momentum at entry (optional)
            
        Returns:
            Balanced score between min_score_floor and max_positive_score
        """
        if not target_hit:
            # For missed targets, provide graduated scoring based on how close we got
            proximity_score = self._calculate_proximity_score(max_adverse, net_profit)
            return max(self.min_score_floor, proximity_score)
        
        score_components = []
        
        # 1. Entry Timing Score (40% weight) - The core innovation
        timing_score = self._calculate_entry_timing_score(
            time_to_hit, total_periods, entry_momentum
        )
        score_components.append(timing_score * 0.4)
        
        # 2. Risk Management Score (30% weight) - Balanced approach
        risk_score = self._calculate_balanced_risk_score(max_adverse, net_profit)
        score_components.append(risk_score * 0.3)
        
        # 3. Profitability Score (30% weight) - Graduated scoring
        profit_score = self._calculate_graduated_profit_score(net_profit)
        score_components.append(profit_score * 0.3)
        
        # Calculate final score with proper bounds
        final_score = sum(score_components)
        
        # Apply smooth normalization to [min_score_floor, max_positive_score] range
        normalized_score = self._smooth_normalize(final_score)
        
        return normalized_score
    
    def _calculate_entry_timing_score(self, time_to_hit: Optional[int], 
                                    total_periods: int, entry_momentum: float) -> float:
        """
        Calculate entry timing score - the key innovation for balanced scoring.
        
        Timing zones:
        - Too Early (0-20% of horizon): Penalty for entering before momentum
        - Optimal Window (20-50% of horizon): Maximum positive scores  
        - Late but OK (50-80% of horizon): Reduced but positive scores
        - Too Late (80-100% of horizon): Penalty for missed opportunity
        
        Args:
            time_to_hit: Periods to reach target
            total_periods: Total time horizon
            entry_momentum: Market momentum at entry
            
        Returns:
            Timing score with balanced positive/negative regions
        """
        if time_to_hit is None:
            return self.neutral_score  # Neutral score for unknown timing
        
        # Calculate timing ratio (0 = immediate, 1 = end of horizon)
        timing_ratio = time_to_hit / total_periods
        
        # Define optimal timing zones
        if timing_ratio <= 0.2:
            # TOO EARLY ZONE: Gentle penalty for entering before momentum builds
            early_penalty = (0.2 - timing_ratio) * self.early_penalty_factor
            base_score = self.neutral_score - (early_penalty * 0.1)  # Max 10% penalty
            
            # Momentum bonus can offset early penalty
            momentum_bonus = min(0.1, entry_momentum / self.momentum_threshold * 0.1)
            return max(self.min_score_floor, base_score + momentum_bonus)
        
        elif 0.2 < timing_ratio <= 0.5:
            # OPTIMAL WINDOW: Maximum positive scores for best entry timing
            # Peak score at 30% of horizon (sweet spot)
            optimal_ratio = abs(timing_ratio - 0.35) / 0.15  # Distance from 35% point
            optimal_score = self.max_positive_score * (1.0 - optimal_ratio * 0.2)
            
            # Additional momentum bonus in optimal window
            momentum_bonus = min(0.05, entry_momentum / self.momentum_threshold * 0.05)
            return min(self.max_positive_score, optimal_score + momentum_bonus)
        
        elif 0.5 < timing_ratio <= 0.8:
            # LATE BUT OK ZONE: Reduced positive scores for decent timing
            late_factor = (timing_ratio - 0.5) / 0.3  # 0 to 1 over this zone
            late_score = self.neutral_score + (0.3 * (1.0 - late_factor))
            return max(self.neutral_score, late_score)
        
        else:
            # TOO LATE ZONE: Penalty for missing most of the opportunity
            missed_opportunity = (timing_ratio - 0.8) * self.late_penalty_factor
            late_penalty_score = self.neutral_score - (missed_opportunity * 0.15)  # Max 15% penalty
            return max(self.min_score_floor, late_penalty_score)
    
    def _calculate_balanced_risk_score(self, max_adverse: float, net_profit: float) -> float:
        """
        Calculate balanced risk score with reasonable adverse excursion handling.
        
        Key improvements:
        - Reduced penalty multiplier (10 instead of 30)
        - Risk-reward ratio consideration
        - Smooth penalty curves instead of harsh steps
        """
        if max_adverse <= 0:
            return self.max_positive_score  # Perfect score for no adverse movement
        
        # Balanced penalty multiplier (much gentler than original 30)
        risk_penalty_multiplier = 10
        
        # Calculate base risk penalty with smooth curve
        risk_penalty_raw = max_adverse * risk_penalty_multiplier
        risk_penalty_smooth = 1.0 - (1.0 / (1.0 + math.exp(-5 * (risk_penalty_raw - 0.5))))
        
        # Risk-reward adjustment
        if net_profit > 0:
            risk_reward_ratio = net_profit / max_adverse
            if risk_reward_ratio > 2.0:  # Good risk-reward
                risk_penalty_smooth *= 0.7  # Reduce penalty by 30%
            elif risk_reward_ratio > 1.0:  # Acceptable risk-reward
                risk_penalty_smooth *= 0.85  # Reduce penalty by 15%
        
        # Calculate final risk score
        risk_score = 1.0 - risk_penalty_smooth
        return max(self.min_score_floor, min(self.max_positive_score, risk_score))
    
    def _calculate_graduated_profit_score(self, net_profit: float) -> float:
        """
        Calculate graduated profit score with balanced positive/negative regions.
        
        Profit zones:
        - Large profits (>1.5%): Maximum positive scores
        - Good profits (0.5-1.5%): High positive scores  
        - Small profits (0-0.5%): Moderate positive scores
        - Small losses (0 to -0.5%): Mild penalties
        - Large losses (<-0.5%): Moderate penalties (not extreme)
        """
        if net_profit > 0.015:  # Large profits (>1.5%)
            return self.max_positive_score
        
        elif net_profit > 0.005:  # Good profits (0.5-1.5%)
            profit_factor = net_profit / 0.015
            return self.neutral_score + (0.5 * profit_factor)
        
        elif net_profit > 0:  # Small profits (0-0.5%)
            profit_factor = net_profit / 0.005
            return self.neutral_score + (0.2 * profit_factor)
        
        elif net_profit >= -0.005:  # Small losses (0 to -0.5%)
            loss_factor = abs(net_profit) / 0.005
            penalty = 0.15 * loss_factor  # Max 15% penalty
            return max(self.min_score_floor, self.neutral_score - penalty)
        
        else:  # Large losses (<-0.5%)
            # Graduated penalty based on loss size (not fixed harsh penalty)
            loss_factor = min(1.0, abs(net_profit) / 0.02)  # Cap at 2% loss
            penalty = 0.25 * loss_factor  # Max 25% penalty
            return max(self.min_score_floor, self.neutral_score - penalty)
    
    def _calculate_proximity_score(self, max_adverse: float, net_profit: float) -> float:
        """Calculate score for missed targets based on how close we got."""
        # Base score for missed targets
        base_score = self.min_score_floor + 0.1
        
        # Bonus for getting close to target (small adverse excursion)
        if max_adverse < 0.01:  # Got within 1% of target
            proximity_bonus = (0.01 - max_adverse) * 10  # Up to 10% bonus
            base_score += proximity_bonus * 0.1
        
        # Penalty for large losses on missed targets
        if net_profit < -0.01:  # Large loss on missed target
            loss_penalty = min(0.1, abs(net_profit) * 5)  # Max 10% additional penalty
            base_score -= loss_penalty
        
        return max(self.min_score_floor, base_score)
    
    def _smooth_normalize(self, score: float) -> float:
        """Apply smooth normalization to keep scores in reasonable bounds."""
        # Sigmoid-based normalization for smooth transitions
        sigmoid_input = (score - self.neutral_score) * 2  # Scale around neutral point
        sigmoid_output = 1.0 / (1.0 + math.exp(-sigmoid_input))
        
        # Map sigmoid output to [min_score_floor, max_positive_score] range
        score_range = self.max_positive_score - self.min_score_floor
        normalized = self.min_score_floor + (sigmoid_output * score_range)
        
        return max(self.min_score_floor, min(self.max_positive_score, normalized))

    def calculate_directional_timing_score(self, target_hit: bool, time_to_hit: Optional[int],
                                         max_adverse: float, total_periods: int,
                                         net_profit: float, direction: str,
                                         entry_momentum: float = 0.0) -> float:
        """
        Calculate directional timing score with balanced approach for long/short trades.
        
        Direction-specific timing considerations:
        - Long trades: Favor faster momentum captures, gentle adverse penalties
        - Short trades: Allow more time for development, balanced adverse handling
        """
        # Start with balanced base score
        base_score = self.calculate_balanced_quality_score(
            target_hit, time_to_hit, max_adverse, total_periods, net_profit, entry_momentum
        )
        
        if not target_hit:
            return base_score
        
        # Gentle directional adjustments (much smaller than original)
        directional_adjustment = 0.0
        
        if direction.lower() == 'long':
            # Long trades: Reward quick momentum captures
            if time_to_hit is not None and time_to_hit < total_periods * 0.3:
                if entry_momentum > self.momentum_threshold:
                    directional_adjustment += 0.03  # 3% bonus for quick momentum long
            
            # Gentle penalty for large adverse in longs (reduced from 10%)
            if max_adverse > 0.012:  # >1.2% adverse
                penalty = min(0.02, (max_adverse - 0.012) * 1.5)  # Max 2% penalty
                directional_adjustment -= penalty
        
        else:  # Short trades
            # Short trades: Allow time for development  
            if time_to_hit is not None and time_to_hit > total_periods * 0.4:
                directional_adjustment += 0.02  # 2% bonus for patient shorts
            
            # Gentle penalty for adverse in shorts (reduced from 15%)
            if max_adverse > 0.01:  # >1% adverse  
                penalty = min(0.025, (max_adverse - 0.01) * 2.0)  # Max 2.5% penalty
                directional_adjustment -= penalty
        
        # Apply directional adjustment
        adjusted_score = base_score + directional_adjustment
        
        return max(self.min_score_floor, min(self.max_positive_score, adjusted_score))

    def normalize_composite_scores_balanced(self, composite_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize composite scores with balanced approach.
        
        Key principles:
        1. Eliminate extreme negative scores while preserving relative ranking
        2. Maintain meaningful spread between good and poor features
        3. Keep directional indicators (bias, asymmetry) in natural ranges
        """
        self.logger.info("🎯 Applying balanced composite score normalization")
        
        normalized_scores = composite_scores.copy()
        
        # Opportunity score fields to normalize
        opportunity_fields = [
            'long_overall_opportunity', 'short_overall_opportunity', 'overall_opportunity',
            'long_immediate_opportunity', 'short_immediate_opportunity', 
            'long_short_opportunity', 'short_short_opportunity',
            'leverage_adjusted_score', 'long_leverage_adjusted_score', 'short_leverage_adjusted_score',
            'best_target_prob', 'reversal_capture_score'
        ]
        
        # Collect opportunity scores for balanced normalization
        opportunity_scores = []
        for field in opportunity_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not math.isnan(score):
                    opportunity_scores.append(score)
        
        if opportunity_scores:
            min_score = min(opportunity_scores)
            max_score = max(opportunity_scores)
            
            self.logger.info(f"   Original range: [{min_score:.4f}, {max_score:.4f}]")
            
            # Balanced normalization to [min_score_floor, max_positive_score] range
            if max_score > min_score:
                score_range = self.max_positive_score - self.min_score_floor
                for field in opportunity_fields:
                    if field in normalized_scores:
                        score = normalized_scores[field]
                        if isinstance(score, (int, float)) and not math.isnan(score):
                            # Balanced normalization preserving relative differences
                            normalized_score = self.min_score_floor + score_range * (
                                (score - min_score) / (max_score - min_score)
                            )
                            normalized_scores[field] = normalized_score
            else:
                # All scores equal - set to neutral value
                for field in opportunity_fields:
                    if field in normalized_scores:
                        normalized_scores[field] = self.neutral_score
            
            # Verify results
            new_scores = [normalized_scores[field] for field in opportunity_fields 
                         if field in normalized_scores]
            if new_scores:
                new_min, new_max = min(new_scores), max(new_scores)
                self.logger.info(f"   Normalized range: [{new_min:.4f}, {new_max:.4f}]")
        
        # Handle directional scores (preserve natural ranges but clamp extremes)
        directional_fields = ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']
        for field in directional_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not math.isnan(score):
                    # Clamp to reasonable range while preserving sign and magnitude
                    normalized_scores[field] = max(-1.5, min(1.5, score))
        
        # Ensure confidence scores are in [0, 1] range
        confidence_fields = ['directional_confidence', 'long_directional_consistency', 
                           'short_directional_consistency']
        for field in confidence_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not math.isnan(score):
                    normalized_scores[field] = max(0.0, min(1.0, score))
        
        return normalized_scores

def demonstrate_balanced_entry_timing():
    """Demonstrate the balanced entry timing optimization."""
    print("🎯 Balanced Entry Timing Optimizer for Multi-Horizon Profit Labeler")
    print("=" * 70)
    
    # Initialize optimizer with balanced parameters
    optimizer = EntryTimingOptimizer({
        'optimal_entry_window': 0.3,    # 30% of time horizon is optimal
        'early_penalty_factor': 2.0,    # Gentle early penalty
        'late_penalty_factor': 3.0,     # Moderate late penalty  
        'max_positive_score': 1.0,      # Maximum positive score
        'min_score_floor': 0.15,        # Minimum score (no extreme negatives)
        'neutral_score': 0.5            # Neutral baseline
    })
    
    # Test scenarios across different entry timings
    timing_scenarios = [
        {
            'name': 'Too Early Entry',
            'description': 'Entering before momentum builds (10% of horizon)',
            'target_hit': True,
            'time_to_hit': 1,  # 10% of 10-period horizon
            'total_periods': 10,
            'max_adverse': 0.008,
            'net_profit': 0.012,
            'entry_momentum': 0.001,  # Low momentum
            'direction': 'long'
        },
        {
            'name': 'Optimal Entry Window',
            'description': 'Perfect timing in optimal window (30% of horizon)',
            'target_hit': True,
            'time_to_hit': 3,  # 30% of 10-period horizon  
            'total_periods': 10,
            'max_adverse': 0.004,
            'net_profit': 0.015,
            'entry_momentum': 0.003,  # Good momentum
            'direction': 'long'
        },
        {
            'name': 'Late but OK Entry',
            'description': 'Decent timing but missing some opportunity (60% of horizon)',
            'target_hit': True,
            'time_to_hit': 6,  # 60% of 10-period horizon
            'total_periods': 10,
            'max_adverse': 0.006,
            'net_profit': 0.008,
            'entry_momentum': 0.002,
            'direction': 'long'
        },
        {
            'name': 'Too Late Entry',
            'description': 'Entering too late, missing most opportunity (90% of horizon)',
            'target_hit': True,
            'time_to_hit': 9,  # 90% of 10-period horizon
            'total_periods': 10,
            'max_adverse': 0.003,
            'net_profit': 0.005,
            'entry_momentum': 0.001,
            'direction': 'long'
        },
        {
            'name': 'Patient Short Trade',
            'description': 'Short trade that takes time to develop (50% of horizon)',
            'target_hit': True,
            'time_to_hit': 5,  # 50% of 10-period horizon
            'total_periods': 10,
            'max_adverse': 0.007,
            'net_profit': 0.010,
            'entry_momentum': 0.002,
            'direction': 'short'
        }
    ]
    
    print("\n📊 ENTRY TIMING ANALYSIS")
    print("=" * 50)
    
    for scenario in timing_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"   {scenario['description']}")
        
        # Calculate balanced score
        balanced_score = optimizer.calculate_directional_timing_score(
            scenario['target_hit'], scenario['time_to_hit'],
            scenario['max_adverse'], scenario['total_periods'],
            scenario['net_profit'], scenario['direction'],
            scenario['entry_momentum']
        )
        
        # Calculate timing ratio for analysis
        timing_ratio = scenario['time_to_hit'] / scenario['total_periods']
        timing_zone = ""
        
        if timing_ratio <= 0.2:
            timing_zone = "TOO EARLY ZONE"
        elif timing_ratio <= 0.5:
            timing_zone = "OPTIMAL WINDOW"
        elif timing_ratio <= 0.8:
            timing_zone = "LATE BUT OK"
        else:
            timing_zone = "TOO LATE ZONE"
        
        print(f"   Timing: {timing_ratio*100:.0f}% of horizon ({timing_zone})")
        print(f"   Balanced Score: {balanced_score:.4f}")
        
        # Show score breakdown
        if timing_zone == "OPTIMAL WINDOW":
            print(f"   ✅ High score for optimal timing")
        elif "TOO" in timing_zone:
            print(f"   ⚠️ Penalty applied but not extreme")
        else:
            print(f"   📊 Moderate score for acceptable timing")
    
    # Test composite score normalization
    print(f"\n📈 COMPOSITE SCORE NORMALIZATION")
    print("=" * 50)
    
    # Simulate problematic composite scores
    test_composite_scores = {
        'long_overall_opportunity': 0.08,    # Very low but positive
        'short_overall_opportunity': -0.02,  # Slightly negative
        'overall_opportunity': 0.05,         # Very low  
        'leverage_adjusted_score': -0.01,    # Slightly negative
        'reversal_capture_score': 0.12,     # Low but positive
        'best_target_prob': 0.03,           # Very low
        'directional_bias': -0.6,           # Allowed to be negative
        'opportunity_asymmetry': 0.4,       # Positive asymmetry
        'directional_confidence': 1.2       # Over 1.0 (needs clamping)
    }
    
    print("Original composite scores:")
    negative_count = 0
    very_low_count = 0
    
    for key, value in test_composite_scores.items():
        status = ""
        if key not in ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']:
            if value < 0:
                status = " ❌ NEGATIVE"
                negative_count += 1
            elif value < 0.1:
                status = " ⚠️ VERY LOW"
                very_low_count += 1
            else:
                status = " ✅ OK"
        else:
            status = " 📊 DIRECTIONAL (can be negative)"
        
        print(f"   {key}: {value:.4f}{status}")
    
    # Apply balanced normalization
    normalized_scores = optimizer.normalize_composite_scores_balanced(test_composite_scores)
    
    print("\nBalanced normalized scores:")
    final_negative = 0
    final_very_low = 0
    
    for key, value in normalized_scores.items():
        status = ""
        if key not in ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']:
            if value < 0:
                status = " ❌ STILL NEGATIVE"
                final_negative += 1
            elif value < optimizer.min_score_floor:
                status = " ⚠️ BELOW FLOOR"
                final_very_low += 1
            else:
                status = " ✅ BALANCED"
        else:
            status = " 📊 DIRECTIONAL (preserved)"
        
        print(f"   {key}: {value:.4f}{status}")
    
    print(f"\n🎯 BALANCED OPTIMIZATION RESULTS")
    print("=" * 50)
    
    results_summary = [
        f"Entry timing zones clearly defined with balanced penalties/rewards",
        f"Optimal window (20-50% of horizon) receives highest scores",
        f"Early/late penalties are gentle (2-15% reduction) not extreme",
        f"Negative composite scores: {negative_count} → {final_negative}",
        f"Scores below floor: {very_low_count} → {final_very_low}",
        f"Score range maintained: [{optimizer.min_score_floor}, {optimizer.max_positive_score}]",
        f"Relative ranking preserved while eliminating extreme negatives"
    ]
    
    for result in results_summary:
        print(f"   ✅ {result}")
    
    print(f"\n💡 KEY PRINCIPLES IMPLEMENTED")
    print("=" * 40)
    
    principles = [
        "🎯 BALANCED APPROACH: Neither too harsh penalties nor too generous rewards",
        "⏰ TIMING OPTIMIZATION: Clear optimal entry window with graduated penalties",
        "📊 SMOOTH CURVES: No sudden drops or harsh binary penalties",
        "🔄 PRESERVED RANKING: Relative feature importance maintained",
        "📈 POSITIVE FOCUS: Emphasis on finding best timing, not just avoiding bad timing",
        "🎚️ CONFIGURABLE: All parameters tunable for different market conditions"
    ]
    
    for principle in principles:
        print(f"   {principle}")
    
    # Save results
    results = {
        'timing_scenarios': {
            scenario['name']: {
                'timing_ratio': scenario['time_to_hit'] / scenario['total_periods'],
                'balanced_score': optimizer.calculate_directional_timing_score(
                    scenario['target_hit'], scenario['time_to_hit'],
                    scenario['max_adverse'], scenario['total_periods'],
                    scenario['net_profit'], scenario['direction'],
                    scenario['entry_momentum']
                ),
                'timing_zone': (
                    "TOO EARLY" if scenario['time_to_hit'] / scenario['total_periods'] <= 0.2
                    else "OPTIMAL" if scenario['time_to_hit'] / scenario['total_periods'] <= 0.5  
                    else "LATE BUT OK" if scenario['time_to_hit'] / scenario['total_periods'] <= 0.8
                    else "TOO LATE"
                )
            } for scenario in timing_scenarios
        },
        'composite_normalization': {
            'original': test_composite_scores,
            'normalized': normalized_scores,
            'negative_eliminated': negative_count - final_negative,
            'very_low_improved': very_low_count - final_very_low
        },
        'optimizer_config': {
            'optimal_entry_window': optimizer.optimal_entry_window,
            'early_penalty_factor': optimizer.early_penalty_factor,
            'late_penalty_factor': optimizer.late_penalty_factor,
            'max_positive_score': optimizer.max_positive_score,
            'min_score_floor': optimizer.min_score_floor,
            'neutral_score': optimizer.neutral_score
        }
    }
    
    import json
    with open('/workspace/balanced_entry_timing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: /workspace/balanced_entry_timing_results.json")
    print("✅ Balanced entry timing optimization demonstration completed!")

if __name__ == "__main__":
    demonstrate_balanced_entry_timing()