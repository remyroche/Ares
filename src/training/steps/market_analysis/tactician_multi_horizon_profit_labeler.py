"""
Tactician Multi-Horizon Profit Labeler - With Long/Short Differentiation

This module provides a version of the multi-horizon profit labeler
specifically designed for the Tactician model on 1m timeframe with long/short differentiation.

Key features:
- Long/short differentiation (separate analysis)
- Optimized for 1m timeframe
- Enhanced target generation for directional trading
- Focus on separate long and short opportunity assessment
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import utilities from src level
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.math_validation import safe_divide, validate_finite
from src.utils.matrix_operations import UnifiedMatrixOperations
from src.feature_generation.utils.enhanced_matrix_operations import EnhancedMatrixOperations
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

@dataclass
class TacticianMultiHorizonConfig:
    """Configuration for Tactician multi-horizon profit labeling (with long/short differentiation)."""
    # Profit targets (fee-aware, 0.3% minimum) - TACTICIAN FOCUSED
    profit_targets: Dict[str, float] = field(default_factory=lambda: {
        'micro': 0.003,    # 0.3% (net: 0.22% after 0.08% fees)
        'small': 0.005,    # 0.5% (net: 0.42% after fees)
        'medium': 0.007,   # 0.7% (net: 0.62% after fees)
        'good': 0.010      # 1.0% (net: 0.92% after fees)
    })
    
    # Time horizons (TACTICIAN FOCUSED - 1m timeframe)
    time_horizons: Dict[str, int] = field(default_factory=lambda: {
        'immediate': 5,    # 5 minutes (5 * 1m) - capture quick moves
        'short': 10,       # 10 minutes (10 * 1m) - capture short-term moves
        'medium': 20       # 20 minutes (20 * 1m) - capture medium-term moves
    })
    
    # Fee consideration
    transaction_cost: float = 0.0008  # 0.08%
    
    # Quality scoring parameters (enhanced for Tactician)
    enable_quality_scoring: bool = True
    speed_weight: float = 0.4  # Higher weight for speed in 1m timeframe
    risk_weight: float = 0.3
    profitability_weight: float = 0.3
    
    # Long/short differentiation parameters
    enable_long_short_differentiation: bool = True
    long_short_balance_weight: float = 0.5  # Weight for balancing long/short opportunities
    directional_confidence_threshold: float = 0.1  # Minimum confidence for directional bias
    
    # High-leverage optimization
    leverage_aware: bool = True
    small_move_emphasis: float = 0.5  # Higher emphasis for 1m timeframe

    # Memory optimization settings
    memory_optimization: bool = True
    enable_streaming: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size: int = 20000  # Larger batch size for 1m data
    enable_m1_optimization: bool = True

    # Quality validation settings
    enable_quality_validation: bool = True
    outlier_detection_enabled: bool = True
    outlier_threshold: float = 3.0
    min_sample_quality_score: float = 0.7

class TacticianMultiHorizonProfitLabeler:
    """
    Tactician Multi-horizon profit probability labeler - WITH LONG/SHORT DIFFERENTIATION.
    
    Generates probability distributions for different profit scenarios across
    multiple time horizons, providing separate long and short training signals for Tactician ML models.
    """
    
    def __init__(self, config: Optional[TacticianMultiHorizonConfig] = None):
        """Initialize the Tactician multi-horizon profit labeler."""
        self.config = config or TacticianMultiHorizonConfig()
        self.logger = get_logger('TacticianMultiHorizonProfitLabeler')

        # Initialize matrix operations for performance
        self.matrix_ops = UnifiedMatrixOperations()
        self.enhanced_ops = EnhancedMatrixOperations()

        # Initialize hardware optimizers
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if self.config.enable_m1_optimization:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()

            if self.config.max_memory_usage_gb and self.memory_optimizer:
                self.memory_optimizer.set_memory_limit(self.config.max_memory_usage_gb)

        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
            self.cpu_optimizer.optimize_pandas_operations()

        # Validate configuration
        self._validate_config()

        # Pre-calculate combinations for efficiency
        self.target_horizon_combinations = self._generate_combinations()

        self.logger.info(f'🚀 Tactician Multi-Horizon Profit Labeler initialized (WITH LONG/SHORT DIFFERENTIATION)')
        self.logger.info(f'   → Profit targets: {list(self.config.profit_targets.keys())}')
        self.logger.info(f'   → Time horizons: {list(self.config.time_horizons.keys())}')
        self.logger.info(f'   → Total combinations: {len(self.target_horizon_combinations)}')
        self.logger.info(f'   → Long/Short differentiation: {"Enabled" if self.config.enable_long_short_differentiation else "Disabled"}')
        self.logger.info(f'   → Matrix operations: Enabled')
        self.logger.info(f'   → Memory optimization: {"Enabled" if self.config.memory_optimization else "Disabled"}')
        self.logger.info(f'   → M1 optimization: {"Enabled" if self.config.enable_m1_optimization else "Disabled"}')
        self.logger.info(f'   → Quality validation: {"Enabled" if self.config.enable_quality_validation else "Disabled"}')
        
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
    
    @traced(span_name='generate_tactician_multi_horizon_labels')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    @log_execution_time()
    def generate_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-horizon profit probability labels for Tactician (WITH LONG/SHORT DIFFERENTIATION).
        
        Args:
            data: OHLCV data with 1m timeframe
            
        Returns:
            DataFrame with probability columns for each target/horizon combination
        """
        self.logger.info(f'🔍 Generating Tactician multi-horizon labels for {len(data)} samples (WITH LONG/SHORT DIFFERENTIATION)')
        
        if len(data) < max(self.config.time_horizons.values()) + 1:
            self.logger.warning(f'⚠️ Insufficient data for labeling')
            return data.copy()

        # Enhanced data quality validation and preprocessing
        data_quality_result = self._validate_and_preprocess_data(data)
        if not data_quality_result['is_valid']:
            self.logger.error(f'❌ Data validation failed: {data_quality_result["errors"]}')
            return data.copy()

        # Apply data quality recommendations
        data = data_quality_result['processed_data']
        self.logger.info(f'✅ Data preprocessing completed: {len(data)} rows validated')

        # Memory optimization and data preparation
        if self.config.memory_optimization and self.memory_optimizer:
            labeled_data = self.memory_optimizer.optimize_dataframe_memory(data.copy())
            self.logger.info(f'🧠 Memory optimization applied to {len(data)} rows')
        else:
            labeled_data = self.enhanced_ops.optimize_dataframe(data.copy())

        max_horizon = max(self.config.time_horizons.values())
        
        # Initialize all probability columns (LONG/SHORT DIFFERENTIATED)
        self._initialize_columns(labeled_data)
        
        # Generate labels for each valid sample
        valid_samples = len(data) - max_horizon
        self.logger.info(f'📊 Processing {valid_samples} valid samples with matrix operations')

        # Choose processing strategy based on dataset size and configuration
        if len(data) > self.config.batch_size and self.config.enable_streaming:
            self.logger.info(f'📦 Large dataset detected - using batch processing ({self.config.batch_size} samples per batch)')
            self._generate_labels_batched(labeled_data, data, valid_samples, max_horizon)
        else:
            # Use vectorized operations where possible
            self._generate_labels_vectorized(labeled_data, data, valid_samples, max_horizon)
        
        # Apply quality validation if enabled
        if self.config.enable_quality_validation:
            labeled_data = self._apply_quality_validation(labeled_data, data, valid_samples)
            self.logger.info('✅ Quality validation completed')

        # Calculate summary statistics
        self._log_labeling_statistics(labeled_data, valid_samples)

        return labeled_data

    def _generate_labels_vectorized(self, labeled_data: pd.DataFrame, data: pd.DataFrame,
                                  valid_samples: int, max_horizon: int):
        """Vectorized label generation using matrix operations for performance."""
        # Pre-allocate arrays for better performance
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Process in batches for memory efficiency
        batch_size = min(2000, valid_samples)  # Larger batch size for 1m data
        
        for batch_start in range(0, valid_samples, batch_size):
            batch_end = min(batch_start + batch_size, valid_samples)
            batch_indices = range(batch_start, batch_end)
            
            if batch_start % 10000 == 0:
                self.logger.info(f'   → Progress: {batch_start}/{valid_samples} ({batch_start/valid_samples*100:.1f}%)')
            
            # Process batch with vectorized operations
            self._process_batch_vectorized(labeled_data, close_prices, high_prices, low_prices, 
                                         batch_indices, max_horizon)
    
    def _process_batch_vectorized(self, labeled_data: pd.DataFrame, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                batch_indices: range, max_horizon: int):
        """Process a batch of samples using vectorized operations."""
        for i in batch_indices:
            current_price = close_prices[i]
            sample_labels = self._generate_sample_labels_vectorized(
                close_prices, high_prices, low_prices, i, current_price, max_horizon
            )
            
            # Store all labels for this sample
            for col_name, value in sample_labels.items():
                labeled_data.iloc[i, labeled_data.columns.get_loc(col_name)] = value
    
    def _generate_sample_labels_vectorized(self, close_prices: np.ndarray, high_prices: np.ndarray,
                                         low_prices: np.ndarray, index: int, current_price: float,
                                         max_horizon: int) -> Dict[str, float]:
        """Generate sample labels using vectorized operations (LONG/SHORT DIFFERENTIATED)."""
        sample_labels = {}
        probability_scores = {}
        
        # Generate labels for each target/horizon combination - BOTH DIRECTIONS
        for target_name, horizon_name, target_pct, horizon_periods in self.target_horizon_combinations:
            window_end = min(index + horizon_periods + 1, len(close_prices))
            
            # Extract window data as numpy arrays for vectorized operations
            window_highs = high_prices[index:window_end]
            window_lows = low_prices[index:window_end]
            
            # Calculate probability for LONG direction
            long_result = self._calculate_profit_probability_vectorized(
                window_highs, window_lows, current_price, target_pct, horizon_periods, direction='long'
            )
            
            # Calculate probability for SHORT direction  
            short_result = self._calculate_profit_probability_vectorized(
                window_highs, window_lows, current_price, target_pct, horizon_periods, direction='short'
            )
            
            # Store LONG results
            long_base = f'{target_name}_{horizon_name}_long'
            sample_labels[f'{long_base}_prob'] = long_result['probability']
            sample_labels[f'{long_base}_time_to_hit'] = long_result['time_to_hit'] or -1
            sample_labels[f'{long_base}_max_adverse'] = long_result['max_adverse_excursion']
            sample_labels[f'{long_base}_net_profit'] = long_result['net_profit']
            sample_labels[f'{long_base}_quality_score'] = long_result['quality_score']
            
            # Store SHORT results
            short_base = f'{target_name}_{horizon_name}_short'
            sample_labels[f'{short_base}_prob'] = short_result['probability']
            sample_labels[f'{short_base}_time_to_hit'] = short_result['time_to_hit'] or -1
            sample_labels[f'{short_base}_max_adverse'] = short_result['max_adverse_excursion']
            sample_labels[f'{short_base}_net_profit'] = short_result['net_profit']
            sample_labels[f'{short_base}_quality_score'] = short_result['quality_score']
            
            # Store for composite calculations (both directions)
            probability_scores[f'{target_name}_{horizon_name}_long'] = long_result['probability']
            probability_scores[f'{target_name}_{horizon_name}_short'] = short_result['probability']
        
        # Calculate composite scores (LONG/SHORT DIFFERENTIATED)
        composite_scores = self._calculate_composite_scores(probability_scores, sample_labels)
        sample_labels.update(composite_scores)
        
        return sample_labels
    
    def _calculate_profit_probability_vectorized(self, highs: np.ndarray, lows: np.ndarray,
                                               entry_price: float, profit_target: float,
                                               horizon_periods: int, direction: str = 'long') -> Dict[str, Any]:
        """Vectorized calculation of profit probability (LONG/SHORT DIFFERENTIATED)."""
        if len(highs) < 2:
            return {
                'probability': 0.0,
                'time_to_hit': None,
                'max_adverse_excursion': 0.0,
                'net_profit': 0.0,
                'quality_score': 0.0
            }
        
        # Calculate directional target prices and check hits using vectorized operations
        if direction.lower() == 'long':
            target_price = entry_price * (1 + profit_target)
            target_hit_mask = highs >= target_price
            target_hit = np.any(target_hit_mask)
            
            if target_hit:
                hit_index = np.where(target_hit_mask)[0][0]
                # For longs, adverse move is price going down
                max_adverse = (entry_price - np.min(lows[:hit_index+1])) / entry_price if hit_index > 0 else 0.0
            else:
                max_adverse = (entry_price - np.min(lows)) / entry_price
                
        else:  # direction == 'short'
            target_price = entry_price * (1 - profit_target)  # Short target is below entry
            target_hit_mask = lows <= target_price
            target_hit = np.any(target_hit_mask)
            
            if target_hit:
                hit_index = np.where(target_hit_mask)[0][0]
                # For shorts, adverse move is price going up
                max_adverse = (np.max(highs[:hit_index+1]) - entry_price) / entry_price if hit_index > 0 else 0.0
            else:
                max_adverse = (np.max(highs) - entry_price) / entry_price
        
        time_to_hit = hit_index if target_hit else None
        
        # Calculate net profit after fees
        gross_profit = profit_target if target_hit else 0.0
        net_profit = gross_profit - self.config.transaction_cost
        
        # Base probability
        base_prob = 1.0 if target_hit else 0.1  # Small base probability for uncertainty
        
        # Quality adjustments if enabled
        if self.config.enable_quality_scoring:
            quality_score = self._calculate_directional_quality_score(
                target_hit, time_to_hit, max_adverse, horizon_periods, net_profit, direction
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
    
    def _initialize_columns(self, labeled_data: pd.DataFrame):
        """Initialize all probability and metadata columns (LONG/SHORT DIFFERENTIATED)."""
        columns_to_add = []
        
        # Individual probability columns - BOTH DIRECTIONS
        for target_name, horizon_name, _, _ in self.target_horizon_combinations:
            # LONG columns
            long_base = f'{target_name}_{horizon_name}_long'
            columns_to_add.extend([
                f'{long_base}_prob',
                f'{long_base}_time_to_hit',
                f'{long_base}_max_adverse',
                f'{long_base}_net_profit',
                f'{long_base}_quality_score'
            ])
            
            # SHORT columns
            short_base = f'{target_name}_{horizon_name}_short'
            columns_to_add.extend([
                f'{short_base}_prob',
                f'{short_base}_time_to_hit',
                f'{short_base}_max_adverse',
                f'{short_base}_net_profit',
                f'{short_base}_quality_score'
            ])
        
        # Composite score columns (LONG/SHORT DIFFERENTIATED)
        composite_columns = [
            # Long opportunity scores
            'long_immediate_opportunity',
            'long_short_opportunity',
            'long_medium_opportunity',
            'long_overall_opportunity',
            'long_leverage_adjusted_score',
            'long_directional_strength',
            'long_directional_consistency',
            'long_momentum',
            
            # Short opportunity scores
            'short_immediate_opportunity',
            'short_short_opportunity',
            'short_medium_opportunity',
            'short_overall_opportunity',
            'short_leverage_adjusted_score',
            'short_directional_strength',
            'short_directional_consistency',
            'short_momentum',
            
            # Directional analysis
            'directional_bias',           # 1.0 = long, -1.0 = short, 0.0 = neutral
            'directional_confidence',     # How strong the directional bias is
            'best_direction',            # Direction with highest opportunity (1.0/-1.0/0.0)
            'opportunity_asymmetry',     # Difference between long and short opportunities
            'long_short_balance',        # Balance between long and short opportunities
            
            # Enhanced directional indicators
            'long_short_ratio',          # Ratio of long to short opportunities
            'directional_volatility',    # Volatility of directional bias
            'directional_persistence',   # Persistence of directional bias
            'directional_reversal_risk'  # Risk of directional reversal
        ]
        columns_to_add.extend(composite_columns)
        
        # Initialize all columns with zeros
        for col in columns_to_add:
            labeled_data[col] = 0.0
    
    def _calculate_composite_scores(self, probability_scores: Dict[str, float], 
                                  sample_labels: Dict[str, float]) -> Dict[str, float]:
        """Calculate long/short differentiated composite opportunity scores."""
        composite_scores = {}
        
        # Separate long and short probability scores
        long_scores = {k: v for k, v in probability_scores.items() if '_long' in k}
        short_scores = {k: v for k, v in probability_scores.items() if '_short' in k}
        
        # LONG opportunity scores
        for horizon_name in self.config.time_horizons.keys():
            long_horizon_probs = [prob for key, prob in long_scores.items() 
                                if key.endswith(f'_{horizon_name}_long')]
            if long_horizon_probs:
                composite_scores[f'long_{horizon_name}_opportunity'] = np.mean(long_horizon_probs)
        
        if long_scores:
            long_avg = np.mean(list(long_scores.values()))
            composite_scores['long_overall_opportunity'] = long_avg
        
        # SHORT opportunity scores  
        for horizon_name in self.config.time_horizons.keys():
            short_horizon_probs = [prob for key, prob in short_scores.items() 
                                 if key.endswith(f'_{horizon_name}_short')]
            if short_horizon_probs:
                composite_scores[f'short_{horizon_name}_opportunity'] = np.mean(short_horizon_probs)
        
        if short_scores:
            short_avg = np.mean(list(short_scores.values()))
            composite_scores['short_overall_opportunity'] = short_avg
        
        # High-leverage adjusted score (long/short differentiated)
        if self.config.leverage_aware:
            leverage_weights = {
                'micro': 0.4, 'small': 0.3, 'medium': 0.2, 'good': 0.1
            }
            
            # Calculate for both directions
            for direction, dir_scores in [('long', long_scores), ('short', short_scores)]:
                weighted_score = 0.0
                total_weight = 0.0
                
                for target_name in self.config.profit_targets.keys():
                    weight = leverage_weights.get(target_name, 0.1)
                    target_probs = [prob for key, prob in dir_scores.items() 
                                   if key.startswith(f'{target_name}_')]
                    if target_probs:
                        weighted_score += np.mean(target_probs) * weight
                        total_weight += weight
                
                if total_weight > 0:
                    composite_scores[f'{direction}_leverage_adjusted_score'] = weighted_score / total_weight
        
        # Directional analysis with enhanced logic
        long_avg = composite_scores.get('long_overall_opportunity', 0.0)
        short_avg = composite_scores.get('short_overall_opportunity', 0.0)
        
        # Calculate directional strength for each horizon
        long_immediate = composite_scores.get('long_immediate_opportunity', 0.0)
        long_short = composite_scores.get('long_short_opportunity', 0.0)
        long_medium = composite_scores.get('long_medium_opportunity', 0.0)
        short_immediate = composite_scores.get('short_immediate_opportunity', 0.0)
        short_short = composite_scores.get('short_short_opportunity', 0.0)
        short_medium = composite_scores.get('short_medium_opportunity', 0.0)
        
        # Weighted directional score (immediate gets higher weight for 1m timeframe)
        long_weighted = (long_immediate * 0.5) + (long_short * 0.3) + (long_medium * 0.2)
        short_weighted = (short_immediate * 0.5) + (short_short * 0.3) + (short_medium * 0.2)
        
        # Determine directional bias with adaptive threshold
        confidence_threshold = max(0.05, min(0.15, (long_avg + short_avg) * 0.1))  # Dynamic threshold
        
        if long_weighted > short_weighted + confidence_threshold:
            composite_scores['directional_bias'] = 1.0  # Long bias
            composite_scores['best_direction'] = 1.0
        elif short_weighted > long_weighted + confidence_threshold:
            composite_scores['directional_bias'] = -1.0  # Short bias
            composite_scores['best_direction'] = -1.0
        else:
            composite_scores['directional_bias'] = 0.0  # Neutral
            composite_scores['best_direction'] = 0.0
        
        # Enhanced directional confidence and asymmetry
        composite_scores['directional_confidence'] = abs(long_weighted - short_weighted)
        composite_scores['opportunity_asymmetry'] = long_weighted - short_weighted  # Positive = long bias, Negative = short bias
        
        # Directional consistency score (how consistent the directional bias is across horizons)
        long_consistency = 1.0 - abs(long_immediate - long_short) if (long_immediate + long_short) > 0 else 0.0
        short_consistency = 1.0 - abs(short_immediate - short_short) if (short_immediate + short_short) > 0 else 0.0
        composite_scores['long_directional_consistency'] = max(0.0, long_consistency)
        composite_scores['short_directional_consistency'] = max(0.0, short_consistency)
        
        # Overall directional strength (combines opportunity with consistency)
        composite_scores['long_directional_strength'] = long_weighted * composite_scores['long_directional_consistency']
        composite_scores['short_directional_strength'] = short_weighted * composite_scores['short_directional_consistency']
        
        # Directional momentum indicator
        composite_scores['long_momentum'] = safe_divide(
            (long_immediate - long_short), 
            long_short, 
            0.0
        )
        
        composite_scores['short_momentum'] = safe_divide(
            (short_immediate - short_short), 
            short_short, 
            0.0
        )
        
        # Enhanced directional indicators
        composite_scores['long_short_ratio'] = safe_divide(long_avg, short_avg, 1.0)
        composite_scores['long_short_balance'] = 1.0 - abs(long_avg - short_avg) / (long_avg + short_avg + 1e-8)
        
        # Directional volatility (how much the directional bias changes)
        directional_values = [long_immediate, long_short, long_medium, short_immediate, short_short, short_medium]
        composite_scores['directional_volatility'] = np.std(directional_values) if len(directional_values) > 1 else 0.0
        
        # Directional persistence (how persistent the directional bias is)
        composite_scores['directional_persistence'] = (composite_scores['long_directional_consistency'] + 
                                                      composite_scores['short_directional_consistency']) / 2
        
        # Directional reversal risk (risk of the directional bias reversing)
        composite_scores['directional_reversal_risk'] = 1.0 - composite_scores['directional_persistence']
        
        return composite_scores
    
    def _calculate_directional_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                                           max_adverse: float, total_periods: int, net_profit: float, 
                                           direction: str) -> float:
        """Calculate directional-aware quality score for profit opportunities."""
        if not target_hit:
            return 0.2  # Base score for missed targets
        
        # Start with base quality score
        base_quality = self._calculate_quality_score(target_hit, time_to_hit, max_adverse, total_periods, net_profit)
        
        # Directional adjustments
        directional_multiplier = 1.0
        
        if direction.lower() == 'long':
            # Long trades: reward speed, penalize adverse excursion
            if time_to_hit is not None and time_to_hit < total_periods * 0.3:
                directional_multiplier *= 1.1  # 10% bonus for fast long moves
            
            if max_adverse > 0.01:  # More than 1% adverse for longs
                penalty = min(0.1, (max_adverse - 0.01) * 5)  # Max 10% penalty
                directional_multiplier *= (1.0 - penalty)
                
        else:  # direction == 'short'
            # Short trades: reward persistence, gentle adverse penalties
            if time_to_hit is not None and time_to_hit > total_periods * 0.5:
                directional_multiplier *= 1.05  # 5% bonus for persistent short moves
            
            if max_adverse > 0.008:  # More than 0.8% adverse for shorts
                penalty = min(0.08, (max_adverse - 0.008) * 5)  # Max 8% penalty
                directional_multiplier *= (1.0 - penalty)
        
        # Apply directional adjustment with proper bounds
        adjusted_quality = base_quality * directional_multiplier
        
        # Ensure result stays within reasonable bounds
        return max(0.15, min(1.0, adjusted_quality))
    
    def _calculate_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                               max_adverse: float, total_periods: int, net_profit: float) -> float:
        """Calculate quality score for the profit opportunity."""
        if not target_hit:
            return 0.2  # Base score for missed targets
        
        quality_factors = []
        
        # Speed factor (faster = better) - 40% weight for 1m timeframe
        if time_to_hit is not None:
            speed_factor = max(0.0, 1.0 - (time_to_hit / total_periods))
            speed_score = 0.3 + (speed_factor * 0.7)  # Range: [0.3, 1.0]
            quality_factors.append(speed_score * self.config.speed_weight)
        else:
            quality_factors.append(0.5 * self.config.speed_weight)
        
        # Risk factor (lower adverse excursion = better) - 30% weight
        if max_adverse > 0:
            risk_penalty = min(0.8, max_adverse * 10)  # Reduced penalty
            risk_factor = 1.0 - risk_penalty
            risk_score = max(0.2, risk_factor)
        else:
            risk_score = 1.0  # Perfect score if no adverse excursion
        
        quality_factors.append(risk_score * self.config.risk_weight)
        
        # Profitability factor (after fees) - 30% weight
        if net_profit > 0:
            profit_factor = min(1.0, net_profit * 200)  # Reduced scale factor
            profit_score = max(0.3, profit_factor)
        else:
            if net_profit >= -0.005:  # Small losses
                profit_score = 0.25
            elif net_profit >= -0.01:  # Medium losses
                profit_score = 0.2
            else:  # Large losses
                profit_score = 0.15
        
        quality_factors.append(profit_score * self.config.profitability_weight)
        
        # Calculate total
        total_quality = np.sum(quality_factors)
        
        # Normalize to [0.2, 1.0] range
        normalized_quality = 0.2 + (min(1.0, total_quality) * 0.8)
        
        return normalized_quality
    
    def _log_labeling_statistics(self, labeled_data: pd.DataFrame, valid_samples: int):
        """Log labeling statistics with enhanced directional analysis."""
        self.logger.info('📊 Tactician Multi-Horizon Labeling Statistics (LONG/SHORT DIFFERENTIATED):')
        
        # Long opportunity analysis
        long_opp = labeled_data['long_overall_opportunity'].iloc[:valid_samples]
        self.logger.info(f'   → Long opportunities: mean={long_opp.mean():.3f}, std={long_opp.std():.3f}')
        
        # Short opportunity analysis
        short_opp = labeled_data['short_overall_opportunity'].iloc[:valid_samples]
        self.logger.info(f'   → Short opportunities: mean={short_opp.mean():.3f}, std={short_opp.std():.3f}')
        
        # Directional bias analysis
        directional_bias = labeled_data['directional_bias'].iloc[:valid_samples]
        long_bias_count = (directional_bias > 0.5).sum()
        short_bias_count = (directional_bias < -0.5).sum()
        neutral_count = valid_samples - long_bias_count - short_bias_count
        
        self.logger.info(f'   → Directional bias distribution:')
        self.logger.info(f'     • Long bias: {long_bias_count} ({long_bias_count/valid_samples*100:.1f}%)')
        self.logger.info(f'     • Short bias: {short_bias_count} ({short_bias_count/valid_samples*100:.1f}%)')
        self.logger.info(f'     • Neutral: {neutral_count} ({neutral_count/valid_samples*100:.1f}%)')
        
        # Directional strength analysis
        long_strength = labeled_data['long_directional_strength'].iloc[:valid_samples]
        short_strength = labeled_data['short_directional_strength'].iloc[:valid_samples]
        
        self.logger.info(f'   → Directional strength:')
        self.logger.info(f'     • Long strength: mean={long_strength.mean():.3f}, max={long_strength.max():.3f}')
        self.logger.info(f'     • Short strength: mean={short_strength.mean():.3f}, max={short_strength.max():.3f}')
        
        # High opportunity samples (enhanced)
        high_long_count = (long_opp > 0.7).sum()
        high_short_count = (short_opp > 0.7).sum()
        self.logger.info(f'   → High opportunity samples (>0.7):')
        self.logger.info(f'     • Long: {high_long_count} ({high_long_count/valid_samples*100:.1f}%)')
        self.logger.info(f'     • Short: {high_short_count} ({high_short_count/valid_samples*100:.1f}%)')
        
        # Directional momentum analysis
        long_momentum = labeled_data['long_momentum'].iloc[:valid_samples]
        short_momentum = labeled_data['short_momentum'].iloc[:valid_samples]
        self.logger.info(f'   → Momentum indicators:')
        self.logger.info(f'     • Long momentum: mean={long_momentum.mean():.3f}')
        self.logger.info(f'     • Short momentum: mean={short_momentum.mean():.3f}')
        
        # Long/short balance analysis
        long_short_balance = labeled_data['long_short_balance'].iloc[:valid_samples]
        self.logger.info(f'   → Long/Short balance: mean={long_short_balance.mean():.3f}')
        
        self.logger.info('✅ Tactician multi-horizon directional labeling completed successfully')
    
    def _validate_and_preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data validation and preprocessing."""
        try:
            self.logger.info('🔍 Starting comprehensive data validation and preprocessing')

            # Basic validation
            if data is None or data.empty:
                return {'is_valid': False, 'errors': ["DataFrame is None or empty"], 'processed_data': data}

            if len(data) < max(self.config.time_horizons.values()) + 1:
                return {'is_valid': False, 'errors': [f"Insufficient data: {len(data)} rows"], 'processed_data': data}

            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                # Create fallback columns if we have at least one price column
                price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
                if price_columns:
                    reference_price = data[price_columns[0]]
                    for col in missing_columns:
                        if col == 'volume':
                            data[col] = 1000  # Default volume
                        elif col in ['open', 'high', 'low', 'close']:
                            data[col] = reference_price  # Use existing price as fallback

            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove non-numeric columns
            data = data.select_dtypes(include=[np.number])
            
            # Ensure all data is float
            for col in data.columns:
                if data[col].dtype != np.float64:
                    try:
                        data[col] = data[col].astype(np.float64)
                    except (ValueError, TypeError):
                        data = data.drop(columns=[col])

            return {
                'is_valid': True,
                'processed_data': data,
                'errors': [],
                'warnings': []
            }

        except Exception as e:
            return {
                'is_valid': False,
                'processed_data': data,
                'errors': [str(e)],
                'warnings': []
            }
    
    def _apply_quality_validation(self, labeled_data: pd.DataFrame, original_data: pd.DataFrame,
                                valid_samples: int) -> pd.DataFrame:
        """Apply quality validation to labeling results."""
        try:
            self.logger.info('🔍 Starting quality validation of labeling results')

            # Detect and handle outliers
            if self.config.outlier_detection_enabled:
                labeled_data = self._detect_and_handle_outliers(labeled_data, valid_samples)

            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in quality validation: {e}')
            return labeled_data
    
    def _detect_and_handle_outliers(self, labeled_data: pd.DataFrame, valid_samples: int) -> pd.DataFrame:
        """Detect and handle outliers in probability scores."""
        try:
            prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]

            if not prob_columns:
                return labeled_data

            for col in prob_columns:
                try:
                    values = labeled_data[col].iloc[:valid_samples].dropna()

                    if len(values) < 10:
                        continue

                    # Use IQR method for outlier detection
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR

                    # Identify outliers
                    outlier_mask = (values < lower_bound) | (values > upper_bound)

                    if outlier_mask.any():
                        outlier_count = outlier_mask.sum()
                        outlier_ratio = outlier_count / len(values)

                        if outlier_ratio < 0.1:  # Less than 10% outliers
                            median_val = values.median()
                            labeled_data.loc[labeled_data[col].iloc[:valid_samples][outlier_mask].index, col] = median_val
                            self.logger.info(f'🧹 Corrected {outlier_count} outliers in {col}')

                except Exception as e:
                    self.logger.warning(f'⚠️ Error processing outliers in {col}: {e}')
                    continue

            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in outlier detection: {e}')
            return labeled_data

# Convenience functions
def create_tactician_multi_horizon_labeler(config: Optional[TacticianMultiHorizonConfig] = None) -> TacticianMultiHorizonProfitLabeler:
    """Create Tactician multi-horizon profit labeler."""
    return TacticianMultiHorizonProfitLabeler(config)

def apply_tactician_multi_horizon_labeling(data: pd.DataFrame, 
                                         config: Optional[TacticianMultiHorizonConfig] = None) -> pd.DataFrame:
    """Apply Tactician multi-horizon profit labeling to data."""
    labeler = TacticianMultiHorizonProfitLabeler(config)
    return labeler.generate_labels(data)