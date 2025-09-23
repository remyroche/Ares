"""
Analyst Multi-Horizon Profit Labeler - No Long/Short Differentiation

This module provides a simplified version of the multi-horizon profit labeler
specifically designed for the Analyst model on 5m timeframe without long/short differentiation.

Key features:
- No long/short differentiation (unified approach)
- Optimized for 5m timeframe
- Simplified target generation
- Focus on overall opportunity assessment
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
class AnalystMultiHorizonConfig:
    """Configuration for Analyst multi-horizon profit labeling (no long/short differentiation)."""
    # Profit targets (fee-aware, 0.3% minimum) - ANALYST FOCUSED
    profit_targets: Dict[str, float] = field(default_factory=lambda: {
        'micro': 0.003,    # 0.3% (net: 0.22% after 0.08% fees)
        'small': 0.005,    # 0.5% (net: 0.42% after fees)
        'medium': 0.007,   # 0.7% (net: 0.62% after fees)
        'good': 0.010      # 1.0% (net: 0.92% after fees)
    })
    
    # Time horizons (ANALYST FOCUSED - 5m timeframe)
    time_horizons: Dict[str, int] = field(default_factory=lambda: {
        'immediate': 2,    # 10 minutes (2 * 5m) - capture quick moves
        'short': 4         # 20 minutes (4 * 5m) - capture short-term moves
    })
    
    # Fee consideration
    transaction_cost: float = 0.0008  # 0.08%
    
    # Quality scoring parameters (simplified for Analyst)
    enable_quality_scoring: bool = True
    speed_weight: float = 0.3
    risk_weight: float = 0.4
    profitability_weight: float = 0.3
    
    # High-leverage optimization
    leverage_aware: bool = True
    small_move_emphasis: float = 0.4  # Emphasize smaller moves for high leverage

    # Memory optimization settings
    memory_optimization: bool = True
    enable_streaming: bool = True
    max_memory_usage_gb: float = 8.0  # Maximum memory usage in GB
    batch_size: int = 10000  # Processing batch size for large datasets
    enable_m1_optimization: bool = True  # Enable M1-specific optimizations

    # Quality validation settings
    enable_quality_validation: bool = True
    outlier_detection_enabled: bool = True
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    min_sample_quality_score: float = 0.7  # Minimum quality score for samples

class AnalystMultiHorizonProfitLabeler:
    """
    Analyst Multi-horizon profit probability labeler - NO LONG/SHORT DIFFERENTIATION.
    
    Generates probability distributions for different profit scenarios across
    multiple time horizons, providing unified training signals for Analyst ML models.
    """
    
    def __init__(self, config: Optional[AnalystMultiHorizonConfig] = None):
        """Initialize the Analyst multi-horizon profit labeler."""
        self.config = config or AnalystMultiHorizonConfig()
        self.logger = get_logger('AnalystMultiHorizonProfitLabeler')

        # Initialize matrix operations for performance
        self.matrix_ops = UnifiedMatrixOperations()
        self.enhanced_ops = EnhancedMatrixOperations()

        # Initialize hardware optimizers
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if self.config.enable_m1_optimization:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()

            # Set memory limits if specified
            if self.config.max_memory_usage_gb and self.memory_optimizer:
                self.memory_optimizer.set_memory_limit(self.config.max_memory_usage_gb)

        # Optimize CPU for data processing
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
            self.cpu_optimizer.optimize_pandas_operations()

        # Validate configuration
        self._validate_config()

        # Pre-calculate combinations for efficiency
        self.target_horizon_combinations = self._generate_combinations()

        self.logger.info(f'🚀 Analyst Multi-Horizon Profit Labeler initialized (NO LONG/SHORT DIFFERENTIATION)')
        self.logger.info(f'   → Profit targets: {list(self.config.profit_targets.keys())}')
        self.logger.info(f'   → Time horizons: {list(self.config.time_horizons.keys())}')
        self.logger.info(f'   → Total combinations: {len(self.target_horizon_combinations)}')
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
    
    @traced(span_name='generate_analyst_multi_horizon_labels')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    @log_execution_time()
    def generate_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-horizon profit probability labels for Analyst (NO LONG/SHORT DIFFERENTIATION).
        
        Args:
            data: OHLCV data with 5m timeframe
            
        Returns:
            DataFrame with probability columns for each target/horizon combination
        """
        self.logger.info(f'🔍 Generating Analyst multi-horizon labels for {len(data)} samples (NO LONG/SHORT DIFFERENTIATION)')
        
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
        
        # Initialize all probability columns (UNIFIED - NO LONG/SHORT)
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
        batch_size = min(1000, valid_samples)
        
        for batch_start in range(0, valid_samples, batch_size):
            batch_end = min(batch_start + batch_size, valid_samples)
            batch_indices = range(batch_start, batch_end)
            
            if batch_start % 5000 == 0:
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
        """Generate sample labels using vectorized operations (UNIFIED APPROACH)."""
        sample_labels = {}
        probability_scores = {}
        
        # Generate labels for each target/horizon combination - UNIFIED (NO LONG/SHORT)
        for target_name, horizon_name, target_pct, horizon_periods in self.target_horizon_combinations:
            window_end = min(index + horizon_periods + 1, len(close_prices))
            
            # Extract window data as numpy arrays for vectorized operations
            window_highs = high_prices[index:window_end]
            window_lows = low_prices[index:window_end]
            
            # Calculate probability for UNIFIED direction (both long and short opportunities)
            unified_result = self._calculate_profit_probability_vectorized(
                window_highs, window_lows, current_price, target_pct, horizon_periods
            )
            
            # Store UNIFIED results (no long/short differentiation)
            base_name = f'{target_name}_{horizon_name}'
            sample_labels[f'{base_name}_prob'] = unified_result['probability']
            sample_labels[f'{base_name}_time_to_hit'] = unified_result['time_to_hit'] or -1
            sample_labels[f'{base_name}_max_adverse'] = unified_result['max_adverse_excursion']
            sample_labels[f'{base_name}_net_profit'] = unified_result['net_profit']
            sample_labels[f'{base_name}_quality_score'] = unified_result['quality_score']
            
            # Store for composite calculations
            probability_scores[f'{target_name}_{horizon_name}'] = unified_result['probability']
        
        # Calculate composite scores (UNIFIED)
        composite_scores = self._calculate_composite_scores(probability_scores, sample_labels)
        sample_labels.update(composite_scores)
        
        return sample_labels
    
    def _calculate_profit_probability_vectorized(self, highs: np.ndarray, lows: np.ndarray,
                                               entry_price: float, profit_target: float,
                                               horizon_periods: int) -> Dict[str, Any]:
        """Vectorized calculation of profit probability (UNIFIED APPROACH)."""
        if len(highs) < 2:
            return {
                'probability': 0.0,
                'time_to_hit': None,
                'max_adverse_excursion': 0.0,
                'net_profit': 0.0,
                'quality_score': 0.0
            }
        
        # Calculate BOTH long and short opportunities and take the better one
        # Long opportunity
        long_target_price = entry_price * (1 + profit_target)
        long_hit_mask = highs >= long_target_price
        long_hit = np.any(long_hit_mask)
        
        # Short opportunity  
        short_target_price = entry_price * (1 - profit_target)
        short_hit_mask = lows <= short_target_price
        short_hit = np.any(short_hit_mask)
        
        # Choose the better opportunity (higher probability)
        if long_hit and short_hit:
            # Both hit - choose based on which happened first
            long_hit_index = np.where(long_hit_mask)[0][0] if long_hit else horizon_periods
            short_hit_index = np.where(short_hit_mask)[0][0] if short_hit else horizon_periods
            
            if long_hit_index <= short_hit_index:
                # Long hit first or at same time
                target_hit = True
                time_to_hit = long_hit_index
                max_adverse = (entry_price - np.min(lows[:long_hit_index+1])) / entry_price if long_hit_index > 0 else 0.0
            else:
                # Short hit first
                target_hit = True
                time_to_hit = short_hit_index
                max_adverse = (np.max(highs[:short_hit_index+1]) - entry_price) / entry_price if short_hit_index > 0 else 0.0
        elif long_hit:
            # Only long hit
            target_hit = True
            time_to_hit = np.where(long_hit_mask)[0][0]
            max_adverse = (entry_price - np.min(lows[:time_to_hit+1])) / entry_price if time_to_hit > 0 else 0.0
        elif short_hit:
            # Only short hit
            target_hit = True
            time_to_hit = np.where(short_hit_mask)[0][0]
            max_adverse = (np.max(highs[:time_to_hit+1]) - entry_price) / entry_price if time_to_hit > 0 else 0.0
        else:
            # Neither hit
            target_hit = False
            time_to_hit = None
            # Calculate adverse excursion for the full period
            max_adverse_long = (entry_price - np.min(lows)) / entry_price
            max_adverse_short = (np.max(highs) - entry_price) / entry_price
            max_adverse = max(max_adverse_long, max_adverse_short)
        
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
    
    def _initialize_columns(self, labeled_data: pd.DataFrame):
        """Initialize all probability and metadata columns (UNIFIED APPROACH)."""
        columns_to_add = []
        
        # Individual probability columns - UNIFIED (NO LONG/SHORT)
        for target_name, horizon_name, _, _ in self.target_horizon_combinations:
            base_name = f'{target_name}_{horizon_name}'
            columns_to_add.extend([
                f'{base_name}_prob',
                f'{base_name}_time_to_hit',
                f'{base_name}_max_adverse',
                f'{base_name}_net_profit',
                f'{base_name}_quality_score'
            ])
        
        # Composite score columns (UNIFIED)
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
            'reversal_capture_score',
            'reassessment_frequency'
        ]
        columns_to_add.extend(composite_columns)
        
        # Initialize all columns with zeros
        for col in columns_to_add:
            labeled_data[col] = 0.0
    
    def _calculate_composite_scores(self, probability_scores: Dict[str, float], 
                                  sample_labels: Dict[str, float]) -> Dict[str, float]:
        """Calculate unified composite opportunity scores."""
        composite_scores = {}
        
        # Unified opportunity scores (no long/short differentiation)
        for horizon_name in self.config.time_horizons.keys():
            horizon_probs = [prob for key, prob in probability_scores.items() 
                           if key.endswith(f'_{horizon_name}')]
            if horizon_probs:
                composite_scores[f'{horizon_name}_opportunity'] = np.mean(horizon_probs)
        
        if probability_scores:
            overall_avg = np.mean(list(probability_scores.values()))
            composite_scores['overall_opportunity'] = overall_avg
        
        # High-leverage adjusted score
        if self.config.leverage_aware:
            leverage_weights = {
                'micro': 0.4, 'small': 0.3, 'medium': 0.2, 'good': 0.1
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
        
        # Reversal capture score
        composite_scores['reversal_capture_score'] = self._calculate_reversal_capture_score(
            probability_scores, sample_labels
        )
        
        # Optimal reassessment frequency
        composite_scores['reassessment_frequency'] = self._calculate_optimal_reassessment_frequency(
            time_values, probability_scores
        )
        
        return composite_scores
    
    def _calculate_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                               max_adverse: float, total_periods: int, net_profit: float) -> float:
        """Calculate quality score for the profit opportunity."""
        if not target_hit:
            return 0.2  # Base score for missed targets
        
        quality_factors = []
        
        # Speed factor (faster = better) - 30% weight
        if time_to_hit is not None:
            speed_factor = max(0.0, 1.0 - (time_to_hit / total_periods))
            speed_score = 0.3 + (speed_factor * 0.7)  # Range: [0.3, 1.0]
            quality_factors.append(speed_score * self.config.speed_weight)
        else:
            quality_factors.append(0.5 * self.config.speed_weight)
        
        # Risk factor (lower adverse excursion = better) - 40% weight
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
    
    def _calculate_reversal_capture_score(self, probability_scores: Dict[str, float], 
                                        sample_labels: Dict[str, float]) -> float:
        """Calculate reversal capture score for small reversals."""
        reversal_factors = []
        
        # Speed factor
        time_values = [v for k, v in sample_labels.items() if k.endswith('_time_to_hit') and v >= 0]
        if time_values:
            avg_time = np.mean(time_values)
            speed_factor = max(0.2, 1.0 - (avg_time / 4.0))
            reversal_factors.append(speed_factor * 0.4)
        else:
            reversal_factors.append(0.5 * 0.4)
        
        # Adverse excursion factor
        adverse_values = [v for k, v in sample_labels.items() if k.endswith('_max_adverse')]
        if adverse_values:
            avg_adverse = np.mean(adverse_values)
            clean_factor = max(0.2, 1.0 - (avg_adverse * 20))  # Reduced penalty
            reversal_factors.append(clean_factor * 0.3)
        else:
            reversal_factors.append(0.6 * 0.3)
        
        # Immediate vs short-term ratio
        immediate_prob = probability_scores.get('micro_immediate', 0.0) + probability_scores.get('small_immediate', 0.0)
        short_prob = probability_scores.get('micro_short', 0.0) + probability_scores.get('small_short', 0.0)
        
        if short_prob > 0:
            ratio_factor = min(1.0, immediate_prob / short_prob)
            reversal_factors.append(ratio_factor * 0.3)
        else:
            reversal_factors.append(0.5 * 0.3)
        
        # Calculate final score
        final_score = np.sum(reversal_factors) if reversal_factors else 0.2
        return max(0.15, min(1.0, final_score))
    
    def _calculate_optimal_reassessment_frequency(self, time_values: List[float], 
                                                probability_scores: Dict[str, float]) -> float:
        """Calculate optimal reassessment frequency in minutes."""
        if not time_values:
            return 5.0  # Default 5-minute reassessment
        
        avg_time_to_target = np.mean(time_values)
        
        # Base reassessment frequency on average time to target
        if avg_time_to_target <= 1.0:  # Very fast
            base_frequency = 2.0  # Every 2 minutes
        elif avg_time_to_target <= 2.0:  # Fast
            base_frequency = 3.0  # Every 3 minutes
        elif avg_time_to_target <= 3.0:  # Medium
            base_frequency = 4.0  # Every 4 minutes
        else:  # Slower opportunities
            base_frequency = 5.0  # Every 5 minutes
        
        # Adjust based on probability distribution
        immediate_probs = [v for k, v in probability_scores.items() if 'immediate' in k]
        if immediate_probs and np.mean(immediate_probs) > 0.7:
            base_frequency *= 0.8  # 20% more frequent
        
        return max(1.0, min(10.0, base_frequency))
    
    def _log_labeling_statistics(self, labeled_data: pd.DataFrame, valid_samples: int):
        """Log labeling statistics."""
        self.logger.info('📊 Analyst Multi-Horizon Labeling Statistics:')
        
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
        
        self.logger.info('✅ Analyst multi-horizon labeling completed successfully')
    
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
def create_analyst_multi_horizon_labeler(config: Optional[AnalystMultiHorizonConfig] = None) -> AnalystMultiHorizonProfitLabeler:
    """Create Analyst multi-horizon profit labeler."""
    return AnalystMultiHorizonProfitLabeler(config)

def apply_analyst_multi_horizon_labeling(data: pd.DataFrame, 
                                       config: Optional[AnalystMultiHorizonConfig] = None) -> pd.DataFrame:
    """Apply Analyst multi-horizon profit labeling to data."""
    labeler = AnalystMultiHorizonProfitLabeler(config)
    return labeler.generate_labels(data)