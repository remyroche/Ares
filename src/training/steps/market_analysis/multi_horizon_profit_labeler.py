"""
Multi-Horizon Profit Probability Labeling - Replacement for Triple Barrier Method

This module provides a superior alternative to triple barrier labeling by generating
probability distributions for different profit scenarios across multiple time horizons.

Key advantages over triple barrier:
- No arbitrary parameter setting
- Fee-aware by design
- Rich training signals (20+ targets vs 3)
- High-leverage optimized
- Market-driven probabilities
- Multiple time horizons
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time

@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon profit labeling."""
    # Profit targets (fee-aware, 0.3% minimum) - SHORT-TERM FOCUSED
    profit_targets: Dict[str, float] = field(default_factory=lambda: {
        'micro': 0.003,    # 0.3% (net: 0.22% after 0.08% fees)
        'small': 0.005,    # 0.5% (net: 0.42% after fees)
        'medium': 0.007,   # 0.7% (net: 0.62% after fees)
        'good': 0.010      # 1.0% (net: 0.92% after fees)
    })
    
    # Time horizons (SHORT-TERM ONLY for regular reassessment)
    time_horizons: Dict[str, int] = field(default_factory=lambda: {
        'immediate': 2,    # 10 minutes (2 * 5m) - capture quick moves
        'short': 4         # 20 minutes (4 * 5m) - capture short-term moves
    })
    
    # Fee consideration
    transaction_cost: float = 0.0008  # 0.08%
    
    # Quality scoring parameters
    enable_quality_scoring: bool = True
    speed_weight: float = 0.3
    risk_weight: float = 0.4
    profitability_weight: float = 0.3
    
    # High-leverage optimization
    leverage_aware: bool = True
    small_move_emphasis: float = 0.4  # Emphasize smaller moves for high leverage

class MultiHorizonProfitLabeler:
    """
    Multi-horizon profit probability labeler - superior alternative to triple barrier.
    
    Generates probability distributions for different profit scenarios across
    multiple time horizons, providing rich training signals for ML models.
    """
    
    def __init__(self, config: Optional[MultiHorizonConfig] = None):
        """Initialize the multi-horizon profit labeler."""
        self.config = config or MultiHorizonConfig()
        self.logger = get_logger('MultiHorizonProfitLabeler')
        
        # Validate configuration
        self._validate_config()
        
        # Pre-calculate combinations for efficiency
        self.target_horizon_combinations = self._generate_combinations()
        
        self.logger.info(f'🚀 Multi-Horizon Profit Labeler initialized')
        self.logger.info(f'   → Profit targets: {list(self.config.profit_targets.keys())}')
        self.logger.info(f'   → Time horizons: {list(self.config.time_horizons.keys())}')
        self.logger.info(f'   → Total combinations: {len(self.target_horizon_combinations)}')
        
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check minimum profit targets (0.3% minimum)
        min_target = min(self.config.profit_targets.values())
        if min_target < 0.003:
            raise ValueError(f"Minimum profit target must be >= 0.3%, got {min_target*100:.2f}%")
        
        # Check all targets are profitable after fees
        for name, target in self.config.profit_targets.items():
            net_profit = target - self.config.transaction_cost
            if net_profit <= 0:
                raise ValueError(f"Target '{name}' ({target*100:.2f}%) not profitable after fees")
        
        self.logger.info('✅ Configuration validation passed')
    
    def _generate_combinations(self) -> List[Tuple[str, str, float, int]]:
        """Generate all target/horizon combinations."""
        combinations = []
        for target_name, target_pct in self.config.profit_targets.items():
            for horizon_name, horizon_periods in self.config.time_horizons.items():
                combinations.append((target_name, horizon_name, target_pct, horizon_periods))
        return combinations
    
    @traced(span_name='generate_multi_horizon_labels')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    @log_execution_time()
    def generate_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-horizon profit probability labels.
        
        Args:
            data: OHLCV data with 5m timeframe
            
        Returns:
            DataFrame with probability columns for each target/horizon combination
        """
        self.logger.info(f'🔍 Generating multi-horizon labels for {len(data)} samples')
        
        if len(data) < max(self.config.time_horizons.values()) + 1:
            self.logger.warning(f'⚠️ Insufficient data for labeling')
            return data.copy()
        
        labeled_data = data.copy()
        max_horizon = max(self.config.time_horizons.values())
        
        # Initialize all probability columns
        self._initialize_columns(labeled_data)
        
        # Generate labels for each valid sample
        valid_samples = len(data) - max_horizon
        self.logger.info(f'📊 Processing {valid_samples} valid samples')
        
        for i in range(valid_samples):
            if i % 1000 == 0:
                self.logger.info(f'   → Progress: {i}/{valid_samples} ({i/valid_samples*100:.1f}%)')
            
            current_price = data.iloc[i]['close']
            sample_labels = self._generate_sample_labels(data, i, current_price)
            
            # Store all labels for this sample
            for col_name, value in sample_labels.items():
                labeled_data.loc[i, col_name] = value
        
        # Calculate summary statistics
        self._log_labeling_statistics(labeled_data, valid_samples)
        
        return labeled_data
    
    def _initialize_columns(self, labeled_data: pd.DataFrame):
        """Initialize all probability and metadata columns."""
        columns_to_add = []
        
        # Individual probability columns
        for target_name, horizon_name, _, _ in self.target_horizon_combinations:
            col_name = f'{target_name}_{horizon_name}_prob'
            columns_to_add.extend([
                col_name,
                f'{col_name}_time_to_hit',
                f'{col_name}_max_adverse',
                f'{col_name}_net_profit',
                f'{col_name}_quality_score'
            ])
        
        # Composite score columns (SHORT-TERM FOCUSED)
        composite_columns = [
            'immediate_opportunity',
            'short_term_opportunity', 
            'overall_opportunity',
            'leverage_adjusted_score',
            'best_target_prob',
            'best_target_name',
            'avg_time_to_target',
            'avg_max_adverse',
            'net_profitability_score',
            'reversal_capture_score',    # NEW: Score for capturing reversals
            'reassessment_frequency'     # NEW: Optimal reassessment frequency
        ]
        columns_to_add.extend(composite_columns)
        
        # Initialize all columns with zeros
        for col in columns_to_add:
            labeled_data[col] = 0.0
    
    def _generate_sample_labels(self, data: pd.DataFrame, index: int, current_price: float) -> Dict[str, float]:
        """Generate all labels for a single sample."""
        sample_labels = {}
        probability_scores = {}
        
        # Generate labels for each target/horizon combination
        for target_name, horizon_name, target_pct, horizon_periods in self.target_horizon_combinations:
            window_end = min(index + horizon_periods + 1, len(data))
            window_data = data.iloc[index:window_end]
            
            # Calculate probability for this target/horizon
            prob_result = self._calculate_profit_probability(
                window_data, current_price, target_pct, horizon_periods
            )
            
            # Store individual results
            col_base = f'{target_name}_{horizon_name}'
            sample_labels[f'{col_base}_prob'] = prob_result['probability']
            sample_labels[f'{col_base}_time_to_hit'] = prob_result['time_to_hit'] or -1
            sample_labels[f'{col_base}_max_adverse'] = prob_result['max_adverse_excursion']
            sample_labels[f'{col_base}_net_profit'] = prob_result['net_profit']
            sample_labels[f'{col_base}_quality_score'] = prob_result['quality_score']
            
            # Store for composite calculations
            probability_scores[f'{target_name}_{horizon_name}'] = prob_result['probability']
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(probability_scores, sample_labels)
        sample_labels.update(composite_scores)
        
        return sample_labels
    
    def _calculate_profit_probability(self, window_data: pd.DataFrame, 
                                    entry_price: float, 
                                    profit_target: float,
                                    horizon_periods: int) -> Dict[str, Any]:
        """Calculate probability and quality metrics for a profit target."""
        if len(window_data) < 2:
            return {
                'probability': 0.0,
                'time_to_hit': None,
                'max_adverse_excursion': 0.0,
                'net_profit': 0.0,
                'quality_score': 0.0
            }
        
        target_price = entry_price * (1 + profit_target)
        highs = window_data['high'].values
        lows = window_data['low'].values
        
        # Check if target is hit
        target_hit = np.any(highs >= target_price)
        time_to_hit = None
        max_adverse = 0.0
        
        if target_hit:
            hit_index = np.where(highs >= target_price)[0][0]
            time_to_hit = hit_index
            
            # Calculate max adverse excursion before hit
            if hit_index > 0:
                max_adverse = (entry_price - np.min(lows[:hit_index+1])) / entry_price
            else:
                max_adverse = 0.0
        else:
            # Target not hit - calculate max adverse over whole period
            max_adverse = (entry_price - np.min(lows)) / entry_price
        
        # Calculate net profit after fees
        gross_profit = profit_target if target_hit else 0.0
        net_profit = gross_profit - self.config.transaction_cost
        
        # Base probability
        base_prob = 1.0 if target_hit else 0.1  # Small base probability for uncertainty
        
        # Quality adjustments if enabled
        if self.config.enable_quality_scoring:
            quality_score = self._calculate_quality_score(
                target_hit, time_to_hit, max_adverse, horizon_periods, net_profit
            )
            final_probability = base_prob * quality_score
        else:
            quality_score = 1.0 if target_hit else 0.1
            final_probability = base_prob
        
        return {
            'probability': np.clip(final_probability, 0.0, 1.0),
            'time_to_hit': time_to_hit,
            'max_adverse_excursion': max_adverse,
            'net_profit': net_profit,
            'quality_score': quality_score
        }
    
    def _calculate_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                               max_adverse: float, total_periods: int, net_profit: float) -> float:
        """
        Calculate quality score for the profit opportunity.
        
        Quality scoring based on three factors:
        1. Speed Factor (30% weight): How quickly the target is reached
           - Faster moves get higher scores (less time risk)
           - Formula: 1.0 - (time_to_hit / total_periods)
           
        2. Risk Factor (40% weight): Maximum adverse excursion before target
           - Lower drawdown before profit = higher quality
           - Formula: 1.0 - (max_adverse_excursion * penalty_multiplier)
           
        3. Profitability Factor (30% weight): Net profit after fees
           - Higher net profit = higher quality
           - Formula: min(1.0, net_profit * scale_factor)
        """
        if not target_hit:
            return 0.1  # Small probability for model uncertainty
        
        quality_factors = []
        
        # 1. Speed factor (faster = better) - 30% weight
        if time_to_hit is not None:
            speed_factor = 1.0 - (time_to_hit / total_periods)
            speed_score = max(0.2, speed_factor)  # Minimum 20% score
            quality_factors.append(speed_score * self.config.speed_weight)
            
            # Bonus for very fast moves (within 50% of time window)
            if time_to_hit < total_periods * 0.5:
                speed_bonus = 0.1
                quality_factors.append(speed_bonus)
        
        # 2. Risk factor (lower adverse excursion = better) - 40% weight
        if max_adverse > 0:
            # Penalize adverse excursion heavily for short-term moves
            risk_penalty_multiplier = 30  # Higher penalty for short-term trades
            risk_factor = max(0.1, 1.0 - (max_adverse * risk_penalty_multiplier))
            risk_score = risk_factor
        else:
            risk_score = 1.0  # Perfect score if no adverse excursion
        quality_factors.append(risk_score * self.config.risk_weight)
        
        # 3. Profitability factor (after fees) - 30% weight
        if net_profit > 0:
            # Scale net profit for short-term moves
            profit_scale_factor = 300  # Higher scaling for small profits
            profit_factor = min(1.0, net_profit * profit_scale_factor)
            profit_score = max(0.2, profit_factor)
            
            # Bonus for high profitability relative to risk
            if max_adverse > 0:
                profit_risk_ratio = net_profit / max_adverse
                if profit_risk_ratio > 2.0:  # 2:1 profit:risk ratio
                    profit_bonus = min(0.2, (profit_risk_ratio - 2.0) * 0.1)
                    quality_factors.append(profit_bonus)
        else:
            profit_score = 0.1  # Low quality if not profitable after fees
        quality_factors.append(profit_score * self.config.profitability_weight)
        
        # Cap total quality score at 1.0
        total_quality = min(1.0, np.sum(quality_factors))
        
        return total_quality
    
    def _calculate_composite_scores(self, probability_scores: Dict[str, float], 
                                  sample_labels: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite opportunity scores."""
        composite_scores = {}
        
        # Horizon-based scores
        for horizon_name in self.config.time_horizons.keys():
            horizon_probs = [prob for key, prob in probability_scores.items() 
                           if key.endswith(f'_{horizon_name}')]
            if horizon_probs:
                composite_scores[f'{horizon_name}_opportunity'] = np.mean(horizon_probs)
        
        # Overall opportunity score
        all_probs = list(probability_scores.values())
        if all_probs:
            composite_scores['overall_opportunity'] = np.mean(all_probs)
        
        # High-leverage adjusted score (emphasize smaller moves)
        if self.config.leverage_aware:
            leverage_weights = {
                'micro': 0.4,
                'small': 0.3,
                'medium': 0.2,
                'good': 0.1,
                'great': 0.05,
                'excellent': 0.05
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for target_name in self.config.profit_targets.keys():
                weight = leverage_weights.get(target_name, 0.1)
                target_probs = [prob for key, prob in probability_scores.items() 
                               if key.startswith(f'{target_name}_')]
                if target_probs:
                    weighted_score += np.mean(target_probs) * weight
                    total_weight += weight
            
            if total_weight > 0:
                composite_scores['leverage_adjusted_score'] = weighted_score / total_weight
        
        # Best target identification
        if probability_scores:
            best_key = max(probability_scores.keys(), key=lambda k: probability_scores[k])
            composite_scores['best_target_prob'] = probability_scores[best_key]
            composite_scores['best_target_name'] = hash(best_key) % 1000  # Simple encoding
        
        # Average metrics
        time_values = [v for k, v in sample_labels.items() if k.endswith('_time_to_hit') and v >= 0]
        adverse_values = [v for k, v in sample_labels.items() if k.endswith('_max_adverse')]
        net_profit_values = [v for k, v in sample_labels.items() if k.endswith('_net_profit')]
        
        composite_scores['avg_time_to_target'] = np.mean(time_values) if time_values else -1
        composite_scores['avg_max_adverse'] = np.mean(adverse_values) if adverse_values else 0
        composite_scores['net_profitability_score'] = np.mean([1 if p > 0 else 0 for p in net_profit_values])
        
        # NEW: Reversal capture score (for capturing small reversals)
        composite_scores['reversal_capture_score'] = self._calculate_reversal_capture_score(
            probability_scores, sample_labels
        )
        
        # NEW: Optimal reassessment frequency (in minutes)
        composite_scores['reassessment_frequency'] = self._calculate_optimal_reassessment_frequency(
            time_values, probability_scores
        )
        
        return composite_scores
    
    def _log_labeling_statistics(self, labeled_data: pd.DataFrame, valid_samples: int):
        """Log labeling statistics."""
        self.logger.info('📊 Labeling Statistics:')
        
        # Overall opportunity distribution
        overall_opp = labeled_data['overall_opportunity'].iloc[:valid_samples]
        self.logger.info(f'   → Overall opportunity: mean={overall_opp.mean():.3f}, std={overall_opp.std():.3f}')
        
        # High opportunity samples
        high_opp_count = (overall_opp > 0.7).sum()
        self.logger.info(f'   → High opportunity samples (>0.7): {high_opp_count} ({high_opp_count/valid_samples*100:.1f}%)')
        
        # Leverage-adjusted scores
        leverage_scores = labeled_data['leverage_adjusted_score'].iloc[:valid_samples]
        self.logger.info(f'   → Leverage-adjusted: mean={leverage_scores.mean():.3f}, std={leverage_scores.std():.3f}')
        
        # Average time to targets
        avg_times = labeled_data['avg_time_to_target'].iloc[:valid_samples]
        valid_times = avg_times[avg_times >= 0]
        if len(valid_times) > 0:
            self.logger.info(f'   → Avg time to target: {valid_times.mean():.1f} periods')
        
        self.logger.info('✅ Multi-horizon labeling completed successfully')
    
    def _calculate_reversal_capture_score(self, probability_scores: Dict[str, float], 
                                        sample_labels: Dict[str, float]) -> float:
        """
        Calculate reversal capture score for small reversals and corrections.
        
        This score measures how well the system can capture small price reversals
        that allow for close/reopen strategies around minor corrections.
        """
        reversal_factors = []
        
        # Factor 1: Speed of opportunity (faster = better for reversals)
        time_values = [v for k, v in sample_labels.items() if k.endswith('_time_to_hit') and v >= 0]
        if time_values:
            avg_time = np.mean(time_values)
            # Shorter time horizons get higher reversal scores
            speed_factor = max(0.1, 1.0 - (avg_time / 4.0))  # Normalize by max 4 periods
            reversal_factors.append(speed_factor * 0.4)  # 40% weight
        
        # Factor 2: Low adverse excursion (clean moves without drawdown)
        adverse_values = [v for k, v in sample_labels.items() if k.endswith('_max_adverse')]
        if adverse_values:
            avg_adverse = np.mean(adverse_values)
            # Lower adverse excursion = better reversal capture
            clean_factor = max(0.1, 1.0 - (avg_adverse * 50))  # Heavy penalty for adverse moves
            reversal_factors.append(clean_factor * 0.3)  # 30% weight
        
        # Factor 3: Immediate vs short-term probability ratio
        immediate_prob = probability_scores.get('micro_immediate', 0.0) + probability_scores.get('small_immediate', 0.0)
        short_prob = probability_scores.get('micro_short', 0.0) + probability_scores.get('small_short', 0.0)
        
        if short_prob > 0:
            # Higher immediate vs short ratio = better for reversals
            ratio_factor = min(1.0, immediate_prob / short_prob)
            reversal_factors.append(ratio_factor * 0.3)  # 30% weight
        
        return np.sum(reversal_factors) if reversal_factors else 0.1
    
    def _calculate_optimal_reassessment_frequency(self, time_values: List[float], 
                                                probability_scores: Dict[str, float]) -> float:
        """
        Calculate optimal reassessment frequency in minutes.
        
        Determines how often positions should be reassessed based on
        the speed of opportunities and probability patterns.
        """
        if not time_values:
            return 5.0  # Default 5-minute reassessment
        
        avg_time_to_target = np.mean(time_values)
        
        # Base reassessment frequency on average time to target
        # Faster opportunities need more frequent reassessment
        if avg_time_to_target <= 1.0:  # Very fast (within 5 minutes)
            base_frequency = 2.0  # Every 2 minutes
        elif avg_time_to_target <= 2.0:  # Fast (within 10 minutes)
            base_frequency = 3.0  # Every 3 minutes
        elif avg_time_to_target <= 3.0:  # Medium (within 15 minutes)
            base_frequency = 4.0  # Every 4 minutes
        else:  # Slower opportunities
            base_frequency = 5.0  # Every 5 minutes
        
        # Adjust based on probability distribution
        immediate_probs = [v for k, v in probability_scores.items() if 'immediate' in k]
        if immediate_probs and np.mean(immediate_probs) > 0.7:
            # High immediate probabilities = more frequent reassessment
            base_frequency *= 0.8  # 20% more frequent
        
        return max(1.0, min(10.0, base_frequency))  # Cap between 1-10 minutes

# Convenience functions for backward compatibility
def create_multi_horizon_labeler(config: Optional[MultiHorizonConfig] = None) -> MultiHorizonProfitLabeler:
    """Create multi-horizon profit labeler."""
    return MultiHorizonProfitLabeler(config)

def apply_multi_horizon_labeling(data: pd.DataFrame, 
                                config: Optional[MultiHorizonConfig] = None) -> pd.DataFrame:
    """Apply multi-horizon profit labeling to data."""
    labeler = MultiHorizonProfitLabeler(config)
    return labeler.generate_labels(data)

# Test function
if __name__ == '__main__':
    # Test the labeler
    tprint('🧪 Testing Multi-Horizon Profit Labeler')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)
    
    # Generate realistic price data with trends
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, 1000)  # Small returns with volatility
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Test labeling
    tprint('\n🔍 Testing multi-horizon labeling...')
    config = MultiHorizonConfig()
    labeled_data = apply_multi_horizon_labeling(data, config)
    
    tprint(f'✅ Labeling completed:')
    tprint(f'   → Input shape: {data.shape}')
    tprint(f'   → Output shape: {labeled_data.shape}')
    tprint(f'   → New columns added: {labeled_data.shape[1] - data.shape[1]}')
    
    # Show sample results
    sample_cols = ['overall_opportunity', 'leverage_adjusted_score', 'immediate_opportunity']
    sample_data = labeled_data[sample_cols].head(10)
    tprint(f'\n📊 Sample results:')
    for col in sample_cols:
        tprint(f'   → {col}: mean={sample_data[col].mean():.3f}')
    
    tprint('✅ Multi-Horizon Profit Labeler test completed successfully!')