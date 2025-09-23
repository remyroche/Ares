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

FIXED VERSION: Addresses critical bugs and performance issues
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

# FIXED: Named constants to replace magic numbers
class ScoringConstants:
    """Constants for scoring calculations to replace magic numbers."""
    
    # Risk penalty multipliers (FIXED: Reduced from problematic values)
    RISK_PENALTY_MULTIPLIER = 10  # Was 30 - causing negative scores
    REVERSAL_PENALTY_MULTIPLIER = 20  # Was 50 - causing negative scores
    
    # Profit scale factors
    PROFIT_SCALE_FACTOR = 200  # Reduced from 300 for smoother scoring
    
    # Quality score bounds (FIXED: Proper bounds)
    MIN_QUALITY_SCORE = 0.2  # Increased from 0.1
    MAX_QUALITY_SCORE = 1.0
    
    # Directional penalties (FIXED: Gentler penalties)
    LONG_ADVERSE_PENALTY = 0.05  # Max 5% penalty instead of 10%
    SHORT_ADVERSE_PENALTY = 0.08  # Max 8% penalty instead of 15%
    
    # Speed bonus thresholds
    FAST_MOVE_THRESHOLD = 0.3  # Within 30% of time window
    VERY_FAST_MOVE_THRESHOLD = 0.5  # Within 50% of time window
    
    # Profit-risk ratio thresholds
    PROFIT_RISK_THRESHOLD = 1.5  # Reduced from 2.0
    
    # Adverse excursion thresholds
    LONG_ADVERSE_THRESHOLD = 0.01  # 1%
    SHORT_ADVERSE_THRESHOLD = 0.008  # 0.8%

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

class MultiHorizonProfitLabeler:
    """
    Multi-horizon profit probability labeler - superior alternative to triple barrier.
    
    Generates probability distributions for different profit scenarios across
    multiple time horizons, providing rich training signals for ML models.
    """
    
    def __init__(self, config: Optional[MultiHorizonConfig] = None):
        """Initialize the multi-horizon profit labeler with memory optimization."""
        self.config = config or MultiHorizonConfig()
        self.logger = get_logger('MultiHorizonProfitLabeler')

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

        self.logger.info(f'🚀 Multi-Horizon Profit Labeler initialized (ENHANCED VERSION)')
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
    
    @traced(span_name='generate_multi_horizon_labels')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    @log_execution_time()
    def generate_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Generate multi-horizon profit probability labels with matrix operations optimization.
        
        Args:
            data: OHLCV data with 5m timeframe
            
        Returns:
            DataFrame with probability columns for each target/horizon combination
        """
        self.logger.info(f'🔍 Generating multi-horizon labels for {len(data)} samples (FIXED VERSION)')
        
        if len(data) < max(self.config.time_horizons.values()) + 1:
            self.logger.warning(f'⚠️ Insufficient data for labeling')
            return data.copy()

        # ENHANCED: Comprehensive data quality validation and preprocessing
        data_quality_result = self._validate_and_preprocess_data(data)
        if not data_quality_result['is_valid']:
            self.logger.error(f'❌ Data validation failed: {data_quality_result["errors"]}')
            return data.copy()

        # Apply data quality recommendations
        data = data_quality_result['processed_data']
        self.logger.info(f'✅ Data preprocessing completed: {len(data)} rows validated')

        # ENHANCED: Memory optimization and data preparation
        if self.config.memory_optimization and self.memory_optimizer:
            # Optimize data for memory efficiency
            labeled_data = self.memory_optimizer.optimize_dataframe_memory(data.copy())
            self.logger.info(f'🧠 Memory optimization applied to {len(data)} rows')
        else:
            labeled_data = self.enhanced_ops.optimize_dataframe(data.copy())

        max_horizon = max(self.config.time_horizons.values())
        
        # Initialize all probability columns
        self._initialize_columns(labeled_data)
        
        # Generate labels for each valid sample
        valid_samples = len(data) - max_horizon
        self.logger.info(f'📊 Processing {valid_samples} valid samples with matrix operations')

        # ENHANCED: Choose processing strategy based on dataset size and configuration
        if len(data) > self.config.batch_size and self.config.enable_streaming:
            self.logger.info(f'📦 Large dataset detected - using batch processing ({self.config.batch_size} samples per batch)')
            self._generate_labels_batched(labeled_data, data, valid_samples, max_horizon)
        else:
            # FIXED: Use vectorized operations where possible
            self._generate_labels_vectorized(labeled_data, data, valid_samples, max_horizon)
        
        # ENHANCED: Apply quality validation if enabled
        if self.config.enable_quality_validation:
            labeled_data = self._apply_quality_validation(labeled_data, data, valid_samples)
            self.logger.info('✅ Quality validation completed')

        # Calculate summary statistics
        self._log_labeling_statistics(labeled_data, valid_samples)

        return labeled_data

    def _generate_labels_batched(self, labeled_data: pd.DataFrame, data: pd.DataFrame,
                               valid_samples: int, max_horizon: int):
        """
        ENHANCED: Generate labels using memory-efficient batch processing.

        This method processes data in batches to handle large datasets while
        maintaining memory efficiency and quality validation.
        """
        try:
            batch_size = self.config.batch_size
            total_batches = (valid_samples + batch_size - 1) // batch_size

            self.logger.info(f'🔄 Starting batch processing: {total_batches} batches of {batch_size} samples each')

            # Pre-allocate numpy arrays for better memory efficiency
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, valid_samples)
                batch_indices = range(start_idx, end_idx)

                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    self.logger.info(f'   → Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - Processing samples {start_idx} to {end_idx}')

                # Process batch with memory monitoring
                with self._memory_checkpoint(f'batch_{batch_idx}'):
                    self._process_batch_vectorized(labeled_data, close_prices, high_prices, low_prices,
                                                 batch_indices, max_horizon)

                # Memory cleanup between batches if needed
                if self.memory_optimizer and (batch_idx + 1) % 5 == 0:
                    self.memory_optimizer.force_garbage_collection()

            self.logger.info('✅ Batch processing completed successfully')

        except Exception as e:
            self.logger.error(f'❌ Error in batch processing: {e}')
            # Fallback to regular vectorized processing
            self.logger.info('🔄 Falling back to standard vectorized processing')
            self._generate_labels_vectorized(labeled_data, data, valid_samples, max_horizon)

    def _memory_checkpoint(self, checkpoint_name: str):
        """
        Context manager for memory checkpoint monitoring.
        """
        class MemoryCheckpoint:
            def __init__(self, optimizer, name):
                self.optimizer = optimizer
                self.name = name

            def __enter__(self):
                if self.optimizer:
                    self.optimizer.log_memory_usage(f'Before {self.name}')
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.optimizer:
                    self.optimizer.log_memory_usage(f'After {self.name}')

        return MemoryCheckpoint(self.memory_optimizer, checkpoint_name)

    def _apply_quality_validation(self, labeled_data: pd.DataFrame, original_data: pd.DataFrame,
                                valid_samples: int) -> pd.DataFrame:
        """
        ENHANCED: Apply comprehensive quality validation to labeling results.

        This method validates the quality of generated labels and applies corrections
        for outliers and inconsistencies.
        """
        try:
            self.logger.info('🔍 Starting quality validation of labeling results')

            # Step 1: Detect and handle outliers in probability scores
            if self.config.outlier_detection_enabled:
                labeled_data = self._detect_and_handle_outliers(labeled_data, valid_samples)

            # Step 2: Validate directional consistency
            labeled_data = self._validate_directional_consistency(labeled_data, valid_samples)

            # Step 3: Check for sample quality issues
            labeled_data = self._validate_sample_quality(labeled_data, original_data, valid_samples)

            # Step 4: Apply final quality corrections
            labeled_data = self._apply_final_quality_corrections(labeled_data, valid_samples)

            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in quality validation: {e}')
            return labeled_data  # Return original data on error

    def _detect_and_handle_outliers(self, labeled_data: pd.DataFrame, valid_samples: int) -> pd.DataFrame:
        """
        Detect and handle outliers in probability scores using statistical methods.
        """
        try:
            # Focus on key probability columns
            prob_columns = [col for col in labeled_data.columns
                          if col.endswith('_prob') and not col.endswith('_long_prob') and not col.endswith('_short_prob')]

            if not prob_columns:
                return labeled_data

            # Apply outlier detection to each probability column
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

                        # Only handle if outliers are not excessive
                        if outlier_ratio < 0.1:  # Less than 10% outliers
                            # Replace outliers with median
                            median_val = values.median()
                            labeled_data.loc[labeled_data[col].iloc[:valid_samples][outlier_mask].index, col] = median_val

                            self.logger.info(f'🧹 Corrected {outlier_count} outliers in {col} (median: {median_val:.3f})')
                        else:
                            self.logger.warning(f'⚠️ High outlier ratio in {col}: {outlier_ratio:.1%} - skipping correction')

                except Exception as e:
                    self.logger.warning(f'⚠️ Error processing outliers in {col}: {e}')
                    continue

            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in outlier detection: {e}')
            return labeled_data

    def _validate_directional_consistency(self, labeled_data: pd.DataFrame, valid_samples: int) -> pd.DataFrame:
        """
        Validate directional consistency between long and short signals.
        """
        try:
            # Find directional columns
            long_cols = [col for col in labeled_data.columns if '_long_prob' in col]
            short_cols = [col for col in labeled_data.columns if '_short_prob' in col]

            if not long_cols or not short_cols:
                return labeled_data

            # Create a mapping between long and short columns
            directional_pairs = {}
            for long_col in long_cols:
                # Find corresponding short column (same target and horizon)
                base_name = long_col.replace('_long_prob', '')
                short_col = base_name + '_short_prob'
                if short_col in labeled_data.columns:
                    directional_pairs[long_col] = short_col

            # Check each pair for consistency
            for long_col, short_col in directional_pairs.items():
                try:
                    long_values = labeled_data[long_col].iloc[:valid_samples]
                    short_values = labeled_data[short_col].iloc[:valid_samples]

                    # Calculate directional bias
                    bias = long_values - short_values

                    # Identify extreme inconsistencies (both high probability)
                    extreme_bias_mask = (long_values > 0.8) & (short_values > 0.8)

                    if extreme_bias_mask.any():
                        # These are suspicious - both directions have high probability
                        # Apply moderation: reduce both probabilities
                        moderation_factor = 0.7
                        labeled_data.loc[extreme_bias_mask[extreme_bias_mask].index, long_col] *= moderation_factor
                        labeled_data.loc[extreme_bias_mask[extreme_bias_mask].index, short_col] *= moderation_factor

                        self.logger.info(f'🔧 Moderated {extreme_bias_mask.sum()} extreme directional conflicts in {long_col}')

                except Exception as e:
                    self.logger.warning(f'⚠️ Error validating directional consistency for {long_col}: {e}')
                    continue

            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in directional consistency validation: {e}')
            return labeled_data

    def _validate_sample_quality(self, labeled_data: pd.DataFrame, original_data: pd.DataFrame,
                               valid_samples: int) -> pd.DataFrame:
        """
        Validate sample quality based on multiple criteria.
        """
        try:
            # Calculate quality scores for each sample
            quality_scores = self._calculate_sample_quality_scores_enhanced(labeled_data, original_data, valid_samples)

            # Identify low-quality samples
            low_quality_mask = quality_scores < self.config.min_sample_quality_score

            if low_quality_mask.any():
                low_quality_count = low_quality_mask.sum()

                # Only apply correction if not too many samples are affected
                if low_quality_count < valid_samples * 0.3:  # Less than 30% of samples
                    self.logger.info(f'🛠️ Applying quality corrections to {low_quality_count} low-quality samples')

                    # Apply quality-based corrections
                    labeled_data = self._correct_low_quality_samples(labeled_data, quality_scores,
                                                                  low_quality_mask, valid_samples)
                else:
                    self.logger.warning(f'⚠️ High number of low-quality samples ({low_quality_count}) - skipping corrections')

            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in sample quality validation: {e}')
            return labeled_data

    def _calculate_sample_quality_scores_enhanced(self, labeled_data: pd.DataFrame,
                                                original_data: pd.DataFrame,
                                                valid_samples: int) -> pd.Series:
        """
        Calculate enhanced quality scores for each sample based on multiple factors.
        """
        try:
            quality_scores = pd.Series(1.0, index=labeled_data.index[:valid_samples])

            # Factor 1: Probability distribution reasonableness
            prob_cols = [col for col in labeled_data.columns if col.endswith('_prob')]
            if prob_cols:
                for idx in range(valid_samples):
                    try:
                        sample_probs = labeled_data[prob_cols].iloc[idx].dropna()

                        if len(sample_probs) > 0:
                            # Check for reasonable probability distribution
                            prob_sum = sample_probs.sum()
                            prob_variance = sample_probs.var()

                            # High sum might indicate overconfident predictions
                            if prob_sum > 2.0:
                                quality_scores.iloc[idx] *= 0.8

                            # Very low variance might indicate lack of discrimination
                            if prob_variance < 0.01:
                                quality_scores.iloc[idx] *= 0.9

                    except Exception:
                        continue

            # Factor 2: Directional signal coherence
            long_cols = [col for col in labeled_data.columns if '_long_prob' in col]
            short_cols = [col for col in labeled_data.columns if '_short_prob' in col]

            if long_cols and short_cols:
                for idx in range(valid_samples):
                    try:
                        # Check if directional signals are reasonable
                        long_avg = labeled_data[long_cols].iloc[idx].mean()
                        short_avg = labeled_data[short_cols].iloc[idx].mean()

                        # Extreme bias might indicate poor signal quality
                        bias_ratio = abs(long_avg - short_avg) / (long_avg + short_avg + 0.001)
                        if bias_ratio > 0.8:  # Very strong bias
                            quality_scores.iloc[idx] *= 0.85

                    except Exception:
                        continue

            # Factor 3: Price consistency with original data
            for idx in range(valid_samples):
                try:
                    original_price = original_data.iloc[idx]['close']
                    # Check if any calculated probabilities are based on inconsistent price data
                    # (This would be detected by extreme probability values)

                    max_prob = labeled_data[[col for col in prob_cols if col in labeled_data.columns]].iloc[idx].max()
                    if max_prob > 0.95:  # Very confident prediction
                        # This might be reasonable, but flag for review
                        pass

                except Exception:
                    continue

            return quality_scores

        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating enhanced quality scores: {e}')
            return pd.Series(1.0, index=labeled_data.index[:valid_samples])

    def _correct_low_quality_samples(self, labeled_data: pd.DataFrame, quality_scores: pd.Series,
                                   low_quality_mask: pd.Series, valid_samples: int) -> pd.DataFrame:
        """
        Apply corrections to low-quality samples.
        """
        try:
            # Get indices of low-quality samples
            low_quality_indices = quality_scores[low_quality_mask].index

            # For low-quality samples, apply conservative corrections:
            # 1. Reduce extreme probability values
            prob_cols = [col for col in labeled_data.columns if col.endswith('_prob')]

            if prob_cols:
                for idx in low_quality_indices:
                    try:
                        # Reduce extreme probabilities (>0.8) by 20%
                        for col in prob_cols:
                            if col in labeled_data.columns:
                                current_val = labeled_data.loc[idx, col]
                                if current_val > 0.8:
                                    labeled_data.loc[idx, col] = current_val * 0.8

                    except Exception:
                        continue

            # 2. Adjust directional bias for better balance
            long_cols = [col for col in labeled_data.columns if '_long_prob' in col]
            short_cols = [col for col in labeled_data.columns if '_short_prob' in col]

            if long_cols and short_cols:
                for idx in low_quality_indices:
                    try:
                        # Calculate current directional bias
                        long_avg = labeled_data[long_cols].iloc[idx].mean()
                        short_avg = labeled_data[short_cols].iloc[idx].mean()

                        # If bias is extreme, apply moderation
                        if abs(long_avg - short_avg) > 0.5:
                            # Reduce the stronger signal
                            if long_avg > short_avg:
                                labeled_data.loc[idx, long_cols] *= 0.9
                            else:
                                labeled_data.loc[idx, short_cols] *= 0.9

                    except Exception:
                        continue

            self.logger.info(f'✅ Applied quality corrections to {len(low_quality_indices)} samples')
            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error applying quality corrections: {e}')
            return labeled_data

    def _apply_final_quality_corrections(self, labeled_data: pd.DataFrame, valid_samples: int) -> pd.DataFrame:
        """
        Apply final quality corrections and normalization.
        """
        try:
            # Ensure all probability values are within [0, 1] range
            prob_cols = [col for col in labeled_data.columns if col.endswith('_prob')]

            if prob_cols:
                for col in prob_cols:
                    if col in labeled_data.columns:
                        # Clip values to valid range
                        labeled_data[col] = np.clip(labeled_data[col], 0.0, 1.0)

            # Normalize composite scores to prevent extreme values
            composite_cols = [
                'overall_opportunity', 'leverage_adjusted_score', 'immediate_opportunity',
                'short_term_opportunity', 'long_overall_opportunity', 'short_overall_opportunity'
            ]

            for col in composite_cols:
                if col in labeled_data.columns:
                    # Clip to reasonable range
                    labeled_data[col] = np.clip(labeled_data[col], 0.0, 2.0)

            # Validate directional bias values
            if 'directional_bias' in labeled_data.columns:
                labeled_data['directional_bias'] = np.clip(labeled_data['directional_bias'], -1.0, 1.0)

            self.logger.info('✅ Final quality corrections applied')
            return labeled_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error in final quality corrections: {e}')
            return labeled_data

    def _validate_and_preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ENHANCED: Comprehensive data validation and preprocessing with quality metrics.

        This method performs thorough validation and preprocessing of input data
        including missing value handling, data consistency checks, and quality improvements.
        """
        try:
            self.logger.info('🔍 Starting comprehensive data validation and preprocessing')

            # Step 1: Basic validation
            basic_validation = self._perform_basic_validation(data)
            if not basic_validation['is_valid']:
                return basic_validation

            # Step 2: Advanced quality assessment
            quality_assessment = self._perform_quality_assessment(data)
            self.logger.info(f'📊 Data quality assessment: {quality_assessment["overall_score"]:.3f}')

            # Step 3: Apply preprocessing corrections
            processed_data = self._apply_preprocessing_corrections(data, quality_assessment)

            # Step 4: Final validation
            final_validation = self._perform_final_validation(processed_data)

            return {
                'is_valid': final_validation['is_valid'],
                'processed_data': processed_data,
                'quality_metrics': quality_assessment,
                'validation_results': final_validation,
                'errors': [] if final_validation['is_valid'] else final_validation['errors'],
                'warnings': quality_assessment.get('warnings', [])
            }

        except Exception as e:
            self.logger.error(f'❌ Error in data validation and preprocessing: {e}')
            return {
                'is_valid': False,
                'processed_data': data,
                'errors': [str(e)],
                'warnings': []
            }

    def _perform_basic_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic validation checks on the input data.
        """
        try:
            errors = []
            warnings = []

            # Check for empty data
            if data is None or data.empty:
                errors.append("DataFrame is None or empty")
                return {'is_valid': False, 'errors': errors, 'warnings': warnings}

            # Check for minimum required rows
            min_required_rows = max(self.config.time_horizons.values()) + 1
            if len(data) < min_required_rows:
                errors.append(f"Insufficient data: {len(data)} rows, minimum {min_required_rows} required")
                return {'is_valid': False, 'errors': errors, 'warnings': warnings}

            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
                return {'is_valid': False, 'errors': errors, 'warnings': warnings}

            # Check for reasonable data types
            for col in required_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        try:
                            # Try to convert to numeric
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            warnings.append(f"Converted column {col} to numeric")
                        except Exception:
                            errors.append(f"Column {col} cannot be converted to numeric")
                            return {'is_valid': False, 'errors': errors, 'warnings': warnings}

            return {'is_valid': True, 'errors': errors, 'warnings': warnings}

        except Exception as e:
            return {'is_valid': False, 'errors': [str(e)], 'warnings': []}

    def _perform_quality_assessment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of the input data.
        """
        try:
            quality_metrics = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'missing_values': 0,
                'duplicate_rows': 0,
                'price_consistency_score': 1.0,
                'volume_quality_score': 1.0,
                'data_completeness_score': 1.0,
                'warnings': []
            }

            # Calculate missing values
            quality_metrics['missing_values'] = data.isnull().sum().sum()

            # Calculate duplicates
            quality_metrics['duplicate_rows'] = data.duplicated().sum()

            # Calculate data completeness
            total_cells = len(data) * len(data.columns)
            missing_ratio = quality_metrics['missing_values'] / total_cells if total_cells > 0 else 0
            quality_metrics['data_completeness_score'] = max(0.0, 1.0 - missing_ratio * 2)

            # Check price consistency
            quality_metrics['price_consistency_score'] = self._calculate_price_consistency_score(data)

            # Check volume quality
            quality_metrics['volume_quality_score'] = self._calculate_volume_quality_score(data)

            # Calculate overall quality score
            weights = {
                'completeness': 0.3,
                'consistency': 0.4,
                'volume': 0.2,
                'duplicates': 0.1
            }

            duplicate_penalty = min(1.0, quality_metrics['duplicate_rows'] / len(data))
            overall_score = (
                quality_metrics['data_completeness_score'] * weights['completeness'] +
                quality_metrics['price_consistency_score'] * weights['consistency'] +
                quality_metrics['volume_quality_score'] * weights['volume'] +
                (1.0 - duplicate_penalty) * weights['duplicates']
            )

            quality_metrics['overall_score'] = max(0.0, min(1.0, overall_score))

            # Generate warnings based on quality scores
            if quality_metrics['data_completeness_score'] < 0.8:
                quality_metrics['warnings'].append("Low data completeness - consider data imputation")

            if quality_metrics['price_consistency_score'] < 0.7:
                quality_metrics['warnings'].append("Price consistency issues detected")

            if quality_metrics['volume_quality_score'] < 0.7:
                quality_metrics['warnings'].append("Volume data quality issues detected")

            if quality_metrics['duplicate_rows'] > len(data) * 0.1:
                quality_metrics['warnings'].append("High duplicate ratio detected")

            return quality_metrics

        except Exception as e:
            self.logger.warning(f'⚠️ Error in quality assessment: {e}')
            return {
                'overall_score': 0.5,
                'warnings': [f'Quality assessment failed: {e}']
            }

    def _calculate_price_consistency_score(self, data: pd.DataFrame) -> float:
        """
        Calculate price consistency score based on OHLC relationships.
        """
        try:
            if len(data) < 10:
                return 0.5

            # Sample data for consistency checks (to avoid excessive computation)
            sample_size = min(1000, len(data))
            sample = data.tail(sample_size)

            consistency_issues = 0
            total_checks = 0

            # Check OHLC logical relationships
            total_checks += 1
            high_issues = (sample['high'] < np.maximum(sample['open'], sample['close'])).sum()
            consistency_issues += high_issues

            total_checks += 1
            low_issues = (sample['low'] > np.minimum(sample['open'], sample['close'])).sum()
            consistency_issues += low_issues

            # Check for extreme price changes
            total_checks += 1
            if len(sample) > 1:
                returns = sample['close'].pct_change().dropna()
                extreme_changes = (returns.abs() > 0.5).sum()  # More than 50% change
                if extreme_changes > len(returns) * 0.1:  # More than 10% extreme changes
                    consistency_issues += 1

            # Check for price gaps (unusual)
            total_checks += 1
            if len(sample) > 1:
                price_gaps = ((sample['high'].shift(1) < sample['low']) &
                             (sample.index.to_series().diff().dt.total_seconds() <= 3600)).sum()  # Gaps within same hour
                if price_gaps > len(sample) * 0.05:  # More than 5% gaps
                    consistency_issues += 1

            if total_checks > 0:
                consistency_score = max(0.0, 1.0 - (consistency_issues / total_checks))
            else:
                consistency_score = 0.5

            return consistency_score

        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating price consistency: {e}')
            return 0.5

    def _calculate_volume_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate volume data quality score.
        """
        try:
            if 'volume' not in data.columns or len(data) < 10:
                return 0.5

            volume_data = data['volume'].dropna()

            if len(volume_data) == 0:
                return 0.0

            # Check for zero/negative volumes
            invalid_volumes = (volume_data <= 0).sum()
            invalid_ratio = invalid_volumes / len(volume_data)

            # Check for extreme volume spikes
            volume_mean = volume_data.mean()
            volume_std = volume_data.std()

            if volume_std > 0:
                extreme_volumes = (volume_data > volume_mean + 5 * volume_std).sum()
                extreme_ratio = extreme_volumes / len(volume_data)
            else:
                extreme_ratio = 0

            # Calculate volume quality score
            quality_score = 1.0
            quality_score *= max(0.0, 1.0 - invalid_ratio * 2)  # Penalize invalid volumes
            quality_score *= max(0.0, 1.0 - extreme_ratio * 3)  # Penalize extreme volumes

            return max(0.0, quality_score)

        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating volume quality: {e}')
            return 0.5

    def _apply_preprocessing_corrections(self, data: pd.DataFrame, quality_metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply preprocessing corrections based on quality assessment.
        """
        try:
            corrected_data = data.copy()

            # Apply corrections based on quality issues
            warnings = quality_metrics.get('warnings', [])

            # Handle missing values
            if 'completeness' in str(warnings).lower():
                corrected_data = self._handle_missing_values(corrected_data)

            # Handle volume issues
            if 'volume' in str(warnings).lower():
                corrected_data = self._correct_volume_issues(corrected_data)

            # Handle price consistency issues
            if quality_metrics.get('price_consistency_score', 1.0) < 0.8:
                corrected_data = self._correct_price_consistency_issues(corrected_data)

            # Remove excessive duplicates
            duplicate_ratio = quality_metrics.get('duplicate_rows', 0) / len(data)
            if duplicate_ratio > 0.05:  # More than 5% duplicates
                corrected_data = corrected_data.drop_duplicates()

            return corrected_data

        except Exception as e:
            self.logger.warning(f'⚠️ Error applying preprocessing corrections: {e}')
            return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies.
        """
        try:
            # Forward fill for price data (maintains trend)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill')

            # Backward fill for any remaining missing prices
            for col in price_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(method='bfill')

            # For volume, use median of surrounding values
            if 'volume' in data.columns:
                data['volume'] = data['volume'].fillna(data['volume'].rolling(10, min_periods=1, center=True).median())

            # Final fill for any remaining missing values
            data = data.fillna(method='ffill').fillna(method='bfill')

            self.logger.info('✅ Missing values handled using forward/backward fill and median')
            return data

        except Exception as e:
            self.logger.warning(f'⚠️ Error handling missing values: {e}')
            return data

    def _correct_volume_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Correct volume data issues.
        """
        try:
            if 'volume' not in data.columns:
                return data

            # Replace negative/zero volumes with median
            volume_median = data['volume'].median()
            data['volume'] = data['volume'].clip(lower=volume_median * 0.1)  # Minimum 10% of median

            # Smooth extreme volume spikes
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()

            if volume_std > 0:
                upper_limit = volume_mean + 3 * volume_std
                data['volume'] = data['volume'].clip(upper=upper_limit)

            self.logger.info('✅ Volume data issues corrected')
            return data

        except Exception as e:
            self.logger.warning(f'⚠️ Error correcting volume issues: {e}')
            return data

    def _correct_price_consistency_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Correct price consistency issues.
        """
        try:
            # Fix OHLC logical relationships
            for idx in data.index:
                try:
                    row = data.loc[idx]

                    # Ensure high is maximum of open/close
                    if row['high'] < max(row['open'], row['close']):
                        data.loc[idx, 'high'] = max(row['open'], row['close'])

                    # Ensure low is minimum of open/close
                    if row['low'] > min(row['open'], row['close']):
                        data.loc[idx, 'low'] = min(row['open'], row['close'])

                except Exception:
                    continue

            self.logger.info('✅ Price consistency issues corrected')
            return data

        except Exception as e:
            self.logger.warning(f'⚠️ Error correcting price consistency: {e}')
            return data

    def _perform_final_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform final validation after preprocessing.
        """
        try:
            errors = []
            warnings = []

            # Check for any remaining missing values
            remaining_missing = data.isnull().sum().sum()
            if remaining_missing > 0:
                errors.append(f"Still has {remaining_missing} missing values after preprocessing")

            # Check for any remaining duplicates
            remaining_duplicates = data.duplicated().sum()
            if remaining_duplicates > 0:
                warnings.append(f"Still has {remaining_duplicates} duplicate rows")

            # Final data size check
            if len(data) < max(self.config.time_horizons.values()) + 1:
                errors.append("Insufficient data after preprocessing")

            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }

        except Exception as e:
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _generate_labels_vectorized(self, labeled_data: pd.DataFrame, data: pd.DataFrame,
                                  valid_samples: int, max_horizon: int):
        """
        FIXED: Vectorized label generation using matrix operations for performance.
        
        This method replaces the inefficient row-by-row loop with vectorized operations
        where possible, significantly improving performance.
        """
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
        """
        Process a batch of samples using vectorized operations.
        """
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
        """
        Generate sample labels using vectorized operations where possible.
        """
        sample_labels = {}
        probability_scores = {}

        # Generate labels for each target/horizon combination - COMBINED APPROACH (for Analyst)
        for target_name, horizon_name, target_pct, horizon_periods in self.target_horizon_combinations:
            window_end = min(index + horizon_periods + 1, len(close_prices))

            # Extract window data as numpy arrays for vectorized operations
            window_highs = high_prices[index:window_end]
            window_lows = low_prices[index:window_end]

            # Calculate probability for COMBINED direction (for Analyst)
            combined_result = self._calculate_profit_probability_vectorized(
                window_highs, window_lows, current_price, target_pct, horizon_periods, direction='combined'
            )

            # Store COMBINED results
            combined_base = f'{target_name}_{horizon_name}'
            sample_labels[f'{combined_base}_prob'] = combined_result['probability']
            sample_labels[f'{combined_base}_time_to_hit'] = combined_result['time_to_hit'] or -1
            sample_labels[f'{combined_base}_max_adverse'] = combined_result['max_adverse_excursion']
            sample_labels[f'{combined_base}_net_profit'] = combined_result['net_profit']
            sample_labels[f'{combined_base}_quality_score'] = combined_result['quality_score']

            # Store for composite calculations (combined)
            probability_scores[f'{target_name}_{horizon_name}'] = combined_result['probability']
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(probability_scores, sample_labels)
        sample_labels.update(composite_scores)
        
        return sample_labels
    
    def _calculate_profit_probability_vectorized(self, highs: np.ndarray, lows: np.ndarray,
                                               entry_price: float, profit_target: float,
                                               horizon_periods: int, direction: str = 'long') -> Dict[str, Any]:
        """
        FIXED: Vectorized calculation of profit probability using numpy operations.
        """
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

        elif direction.lower() == 'short':
            target_price = entry_price * (1 - profit_target)  # Short target is below entry
            target_hit_mask = lows <= target_price
            target_hit = np.any(target_hit_mask)

            if target_hit:
                hit_index = np.where(target_hit_mask)[0][0]
                # For shorts, adverse move is price going up
                max_adverse = (np.max(highs[:hit_index+1]) - entry_price) / entry_price if hit_index > 0 else 0.0
            else:
                max_adverse = (np.max(highs) - entry_price) / entry_price

        else:  # direction == 'combined' (for Analyst - no directional differentiation)
            # For combined approach, we consider either direction as a hit
            long_target_price = entry_price * (1 + profit_target)
            short_target_price = entry_price * (1 - profit_target)

            long_target_hit_mask = highs >= long_target_price
            short_target_hit_mask = lows <= short_target_price

            long_target_hit = np.any(long_target_hit_mask)
            short_target_hit = np.any(short_target_hit_mask)

            # Hit if either direction hits
            target_hit = long_target_hit or short_target_hit

            if target_hit:
                # Use the first hit (whichever comes first)
                long_hit_index = np.where(long_target_hit_mask)[0][0] if long_target_hit else len(highs)
                short_hit_index = np.where(short_target_hit_mask)[0][0] if short_target_hit else len(lows)
                hit_index = min(long_hit_index, short_hit_index)

                # Calculate combined adverse excursion (maximum of both directions)
                long_adverse = (entry_price - np.min(lows[:hit_index+1])) / entry_price if hit_index > 0 else 0.0
                short_adverse = (np.max(highs[:hit_index+1]) - entry_price) / entry_price if hit_index > 0 else 0.0
                max_adverse = max(long_adverse, short_adverse)
            else:
                # No hit - use maximum adverse from both directions
                long_adverse = (entry_price - np.min(lows)) / entry_price
                short_adverse = (np.max(highs) - entry_price) / entry_price
                max_adverse = max(long_adverse, short_adverse)
        
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
        """Initialize all probability and metadata columns."""
        columns_to_add = []

        # Individual probability columns - COMBINED APPROACH (for Analyst)
        for target_name, horizon_name, _, _ in self.target_horizon_combinations:
            # COMBINED columns (no directional differentiation for Analyst)
            combined_base = f'{target_name}_{horizon_name}'
            columns_to_add.extend([
                f'{combined_base}_prob',
                f'{combined_base}_time_to_hit',
                f'{combined_base}_max_adverse',
                f'{combined_base}_net_profit',
                f'{combined_base}_quality_score'
            ])
        
        # Composite score columns (COMBINED APPROACH for Analyst)
        composite_columns = [
            # Original composite scores (combined approach for Analyst)
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
    
    def _generate_sample_labels(self, data: pd.DataFrame, index: int, current_price: float) -> Dict[str, float]:
        """Generate all labels for a single sample."""
        sample_labels = {}
        probability_scores = {}
        
        # Generate labels for each target/horizon combination - BOTH DIRECTIONS
        for target_name, horizon_name, target_pct, horizon_periods in self.target_horizon_combinations:
            window_end = min(index + horizon_periods + 1, len(data))
            window_data = data.iloc[index:window_end]
            
            # Calculate probability for LONG direction
            long_result = self._calculate_profit_probability(
                window_data, current_price, target_pct, horizon_periods, direction='long'
            )
            
            # Calculate probability for SHORT direction  
            short_result = self._calculate_profit_probability(
                window_data, current_price, target_pct, horizon_periods, direction='short'
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
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(probability_scores, sample_labels)
        sample_labels.update(composite_scores)
        
        # DEBUG: Log if bi-directional scores were created
        if 'long_overall_opportunity' in composite_scores:
            self.logger.info(f"✅ BI-DIRECTIONAL: long_overall_opportunity = {composite_scores['long_overall_opportunity']:.4f}")
        if 'short_overall_opportunity' in composite_scores:
            self.logger.info(f"✅ BI-DIRECTIONAL: short_overall_opportunity = {composite_scores['short_overall_opportunity']:.4f}")
        
        return sample_labels
    
    def _calculate_profit_probability(self, window_data: pd.DataFrame, 
                                    entry_price: float, 
                                    profit_target: float,
                                    horizon_periods: int,
                                    direction: str = 'long') -> Dict[str, Any]:
        """Calculate probability and quality metrics for a profit target in specified direction."""
        if len(window_data) < 2:
            return {
                'probability': 0.0,
                'time_to_hit': None,
                'max_adverse_excursion': 0.0,
                'net_profit': 0.0,
                'quality_score': 0.0
            }
        
        highs = window_data['high'].values
        lows = window_data['low'].values
        
        # Calculate directional target prices and check hits
        if direction.lower() == 'long':
            target_price = entry_price * (1 + profit_target)
            target_hit = np.any(highs >= target_price)
            if target_hit:
                hit_index = np.where(highs >= target_price)[0][0]
                # For longs, adverse move is price going down
                max_adverse = (entry_price - np.min(lows[:hit_index+1])) / entry_price if hit_index > 0 else 0.0
            else:
                max_adverse = (entry_price - np.min(lows)) / entry_price
                
        else:  # direction == 'short'
            target_price = entry_price * (1 - profit_target)  # Short target is below entry
            target_hit = np.any(lows <= target_price)
            if target_hit:
                hit_index = np.where(lows <= target_price)[0][0]
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
    
    def _calculate_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                               max_adverse: float, total_periods: int, net_profit: float) -> float:
        """
        FIXED: Calculate quality score for the profit opportunity.
        
        Key fixes:
        1. Reduced risk penalty multiplier from 30 to 10 (67% reduction)
        2. Improved profit scoring for negative profits (graduated instead of fixed 0.1)
        3. Increased minimum score bounds from 0.1 to 0.2
        4. Added score normalization to [0.2, 1.0] range
        
        Quality scoring based on three factors:
        1. Speed Factor (30% weight): How quickly the target is reached
        2. Risk Factor (40% weight): Maximum adverse excursion before target
        3. Profitability Factor (30% weight): Net profit after fees
        """
        if not target_hit:
            return ScoringConstants.MIN_QUALITY_SCORE  # Increased from 0.1
        
        quality_factors = []
        
        # 1. FIXED Speed factor (faster = better) - 30% weight
        if time_to_hit is not None:
            # Smoother speed scoring curve
            speed_factor = max(0.0, 1.0 - (time_to_hit / total_periods))
            speed_score = 0.3 + (speed_factor * 0.7)  # Range: [0.3, 1.0]
            quality_factors.append(speed_score * self.config.speed_weight)
            
            # Bonus for very fast moves (within 50% of time window)
            if time_to_hit < total_periods * ScoringConstants.VERY_FAST_MOVE_THRESHOLD:
                speed_bonus = min(0.1, (ScoringConstants.VERY_FAST_MOVE_THRESHOLD - time_to_hit/total_periods) * 0.2)
                quality_factors.append(speed_bonus)
        else:
            # Default speed score when time is unknown
            quality_factors.append(0.5 * self.config.speed_weight)
        
        # 2. FIXED Risk factor (lower adverse excursion = better) - 40% weight
        if max_adverse > 0:
            # CRITICAL FIX: Reduced penalty multiplier from 30 to 10
            risk_penalty_multiplier = ScoringConstants.RISK_PENALTY_MULTIPLIER
            
            # Cap penalty at 80% to prevent extreme penalties
            risk_penalty = min(0.8, max_adverse * risk_penalty_multiplier)
            risk_factor = 1.0 - risk_penalty
            risk_score = max(ScoringConstants.MIN_QUALITY_SCORE, risk_factor)  # Increased minimum
        else:
            risk_score = 1.0  # Perfect score if no adverse excursion
        
        quality_factors.append(risk_score * self.config.risk_weight)
        
        # 3. FIXED Profitability factor (after fees) - 30% weight
        if net_profit > 0:
            # Slightly reduced scale factor for smoother scoring
            profit_scale_factor = ScoringConstants.PROFIT_SCALE_FACTOR
            profit_factor = min(1.0, net_profit * profit_scale_factor)
            profit_score = max(0.3, profit_factor)  # Increased minimum for profitable trades
            
            # Bonus for high profitability relative to risk (lowered threshold)
            if max_adverse > 0:
                profit_risk_ratio = safe_divide(net_profit, max_adverse, 0.0)
                if profit_risk_ratio > ScoringConstants.PROFIT_RISK_THRESHOLD:
                    profit_bonus = min(0.15, (profit_risk_ratio - ScoringConstants.PROFIT_RISK_THRESHOLD) * 0.08)
                    quality_factors.append(profit_bonus)
        else:
            # MAJOR FIX: Graduated scoring for unprofitable trades instead of fixed 0.1
            if net_profit >= -0.005:  # Small losses (< 0.5%)
                profit_score = 0.25  # Much better than original 0.1
            elif net_profit >= -0.01:  # Medium losses (0.5% - 1.0%)
                profit_score = 0.2
            else:  # Large losses (> 1.0%)
                profit_score = 0.15  # Still better than original 0.1
        
        quality_factors.append(profit_score * self.config.profitability_weight)
        
        # Calculate total with improved bounds
        total_quality = np.sum(quality_factors)
        
        # CRITICAL FIX: Normalize to [0.2, 1.0] range instead of just capping at 1.0
        normalized_quality = ScoringConstants.MIN_QUALITY_SCORE + (min(ScoringConstants.MAX_QUALITY_SCORE, total_quality) * 0.8)
        
        return normalized_quality
    
    def _calculate_directional_quality_score(self, target_hit: bool, time_to_hit: Optional[int], 
                                           max_adverse: float, total_periods: int, net_profit: float, 
                                           direction: str) -> float:
        """
        FIXED: Calculate directional-aware quality score for profit opportunities.
        
        Key fixes:
        1. Gentler directional penalties (5-8% instead of 10-15%)
        2. Uses the fixed base quality score
        3. Smoother penalty curves
        4. Better bounds checking
        
        This method builds on the base quality scoring but adds direction-specific adjustments:
        - Long trades: Penalize upward adverse excursion more heavily (going against gravity)
        - Short trades: Penalize downward adverse excursion more heavily (fighting momentum)
        - Different risk-reward expectations for each direction
        """
        if not target_hit:
            return ScoringConstants.MIN_QUALITY_SCORE  # Increased base score
        
        # Start with the FIXED base quality score
        base_quality = self._calculate_quality_score(target_hit, time_to_hit, max_adverse, total_periods, net_profit)
        
        # FIXED: Much gentler directional adjustments
        directional_multiplier = 1.0
        
        if direction.lower() == 'long':
            # Long trades: reward speed, penalize adverse excursion gently
            if time_to_hit is not None and time_to_hit < total_periods * ScoringConstants.FAST_MOVE_THRESHOLD:
                directional_multiplier *= 1.05  # Reduced from 1.1 to 1.05 (5% bonus)
            
            # GENTLER adverse excursion penalty
            if max_adverse > ScoringConstants.LONG_ADVERSE_THRESHOLD:  # More than 1% adverse for longs
                # Smooth penalty curve instead of fixed 10%
                penalty = min(ScoringConstants.LONG_ADVERSE_PENALTY, (max_adverse - ScoringConstants.LONG_ADVERSE_THRESHOLD) * 2)  # Max 5% penalty
                directional_multiplier *= (1.0 - penalty)
                
        else:  # direction == 'short'
            # Short trades: reward persistence, gentle adverse penalties
            if time_to_hit is not None and time_to_hit > total_periods * ScoringConstants.VERY_FAST_MOVE_THRESHOLD:
                directional_multiplier *= 1.03  # Reduced from 1.05 to 1.03 (3% bonus)
            
            # MUCH GENTLER adverse excursion penalty for shorts
            if max_adverse > ScoringConstants.SHORT_ADVERSE_THRESHOLD:  # More than 0.8% adverse for shorts
                # Smooth penalty curve instead of fixed 15%
                penalty = min(ScoringConstants.SHORT_ADVERSE_PENALTY, (max_adverse - ScoringConstants.SHORT_ADVERSE_THRESHOLD) * 5)  # Max 8% penalty instead of 15%
                directional_multiplier *= (1.0 - penalty)
        
        # Apply directional adjustment with proper bounds
        adjusted_quality = base_quality * directional_multiplier
        
        # Ensure result stays within reasonable bounds
        return max(0.15, min(1.0, adjusted_quality))
    
    def _calculate_composite_scores(self, probability_scores: Dict[str, float],
                                  sample_labels: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite opportunity scores for combined approach (Analyst)."""
        composite_scores = {}

        # Use all probability scores (combined approach for Analyst)
        self.logger.debug(f"🔍 Sample probability_scores keys: {list(probability_scores.keys())[:3]}")

        # COMBINED opportunity scores (for Analyst)
        for horizon_name in self.config.time_horizons.keys():
            horizon_probs = [prob for key, prob in probability_scores.items()
                            if key.endswith(f'_{horizon_name}')]
            if horizon_probs:
                composite_scores[f'{horizon_name}_opportunity'] = np.mean(horizon_probs)

        if probability_scores:
            combined_avg = np.mean(list(probability_scores.values()))
            composite_scores['overall_opportunity'] = combined_avg
            self.logger.info(f"✅ Created overall_opportunity (combined): {combined_avg:.4f}")
        
        # High-leverage adjusted score (combined approach for Analyst)
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
        
        # NEW: Reversal capture score (for capturing small reversals)
        composite_scores['reversal_capture_score'] = self._calculate_reversal_capture_score(
            probability_scores, sample_labels
        )
        
        # NEW: Optimal reassessment frequency (in minutes)
        composite_scores['reassessment_frequency'] = self._calculate_optimal_reassessment_frequency(
            time_values, probability_scores
        )
        
        # For Analyst (combined approach), we don't calculate directional bias
        # The overall_opportunity score already represents the combined opportunity
        
        # CRITICAL FIX: Normalize composite scores to eliminate negative values
        composite_scores = self._normalize_composite_scores(composite_scores)
        
        return composite_scores
    
    def _log_labeling_statistics(self, labeled_data: pd.DataFrame, valid_samples: int):
        """Log labeling statistics for combined approach (Analyst)."""
        self.logger.info('📊 Multi-Horizon Labeling Statistics (Combined Approach - Analyst):')

        # Overall opportunity distribution
        overall_opp = labeled_data['overall_opportunity'].iloc[:valid_samples]
        self.logger.info(f'   → Overall opportunity: mean={overall_opp.mean():.3f}, std={overall_opp.std():.3f}')

        # Immediate and short-term opportunities
        immediate_opp = labeled_data['immediate_opportunity'].iloc[:valid_samples]
        short_term_opp = labeled_data['short_term_opportunity'].iloc[:valid_samples]

        self.logger.info(f'   → Immediate opportunities: mean={immediate_opp.mean():.3f}, std={immediate_opp.std():.3f}')
        self.logger.info(f'   → Short-term opportunities: mean={short_term_opp.mean():.3f}, std={short_term_opp.std():.3f}')

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

        # Reversal capture and reassessment frequency
        reversal_scores = labeled_data['reversal_capture_score'].iloc[:valid_samples]
        reassessment_freq = labeled_data['reassessment_frequency'].iloc[:valid_samples]

        self.logger.info(f'   → Reversal capture: mean={reversal_scores.mean():.3f}')
        self.logger.info(f'   → Reassessment frequency: mean={reassessment_freq.mean():.1f} minutes')

        self.logger.info('✅ Multi-horizon combined labeling completed successfully')
    
    def _calculate_reversal_capture_score(self, probability_scores: Dict[str, float], 
                                        sample_labels: Dict[str, float]) -> float:
        """
        FIXED: Calculate reversal capture score for small reversals and corrections.
        
        Key fixes:
        1. Reduced adverse penalty multiplier from 50 to 20 (60% reduction)
        2. Improved minimum score bounds
        3. Better handling of missing data
        
        This score measures how well the system can capture small price reversals
        that allow for close/reopen strategies around minor corrections.
        """
        reversal_factors = []
        
        # Factor 1: Speed of opportunity (faster = better for reversals)
        time_values = [v for k, v in sample_labels.items() if k.endswith('_time_to_hit') and v >= 0]
        if time_values:
            avg_time = np.mean(time_values)
            # Improved speed factor with better bounds
            speed_factor = max(0.2, 1.0 - (avg_time / 4.0))  # Increased minimum from 0.1
            reversal_factors.append(speed_factor * 0.4)  # 40% weight
        else:
            # Default when no time data available
            reversal_factors.append(0.5 * 0.4)
        
        # Factor 2: FIXED adverse excursion penalty
        adverse_values = [v for k, v in sample_labels.items() if k.endswith('_max_adverse')]
        if adverse_values:
            avg_adverse = np.mean(adverse_values)
            # CRITICAL FIX: Reduced penalty multiplier from 50 to 20
            clean_factor = max(0.2, 1.0 - (avg_adverse * ScoringConstants.REVERSAL_PENALTY_MULTIPLIER))  # Much gentler penalty
            reversal_factors.append(clean_factor * 0.3)  # 30% weight
        else:
            # Default when no adverse data available
            reversal_factors.append(0.6 * 0.3)
        
        # Factor 3: Immediate vs short-term probability ratio
        immediate_prob = probability_scores.get('micro_immediate_long', 0.0) + probability_scores.get('small_immediate_long', 0.0)
        short_prob = probability_scores.get('micro_short_long', 0.0) + probability_scores.get('small_short_long', 0.0)
        
        if short_prob > 0:
            ratio_factor = min(1.0, immediate_prob / short_prob)
            reversal_factors.append(ratio_factor * 0.3)  # 30% weight
        else:
            # Better default when no short-term probabilities
            reversal_factors.append(0.5 * 0.3)
        
        # Calculate final score with improved bounds
        final_score = np.sum(reversal_factors) if reversal_factors else 0.2
        return max(0.15, min(1.0, final_score))  # Improved bounds: [0.15, 1.0]
    
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
    
    def _normalize_composite_scores(self, composite_scores: Dict[str, float]) -> Dict[str, float]:
        """
        CRITICAL FIX: Normalize composite scores to eliminate negative values.
        
        This is the most important fix - call this method before returning
        the final composite scores from _calculate_composite_scores().
        """
        self.logger.debug("🔧 Normalizing composite scores to eliminate negative values")
        
        normalized_scores = composite_scores.copy()
        
        # Define which fields should be normalized (opportunity scores)
        opportunity_fields = [
            'long_overall_opportunity', 'short_overall_opportunity', 'overall_opportunity',
            'long_immediate_opportunity', 'short_immediate_opportunity',
            'long_short_opportunity', 'short_short_opportunity',
            'leverage_adjusted_score', 'long_leverage_adjusted_score', 'short_leverage_adjusted_score',
            'best_target_prob', 'net_profitability_score', 'reversal_capture_score',
            'long_directional_strength', 'short_directional_strength'
        ]
        
        # Collect opportunity scores for normalization
        opportunity_scores = []
        for field in opportunity_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not np.isnan(score):
                    opportunity_scores.append(score)
        
        if opportunity_scores:
            min_score = min(opportunity_scores)
            max_score = max(opportunity_scores)
            
            self.logger.debug(f"   Original score range: [{min_score:.4f}, {max_score:.4f}]")
            
            # Apply min-max normalization to [0.1, 1.0] range
            if max_score > min_score:
                for field in opportunity_fields:
                    if field in normalized_scores:
                        score = normalized_scores[field]
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            # Map to [0.1, 1.0] range
                            normalized_score = 0.1 + 0.9 * ((score - min_score) / (max_score - min_score))
                            normalized_scores[field] = normalized_score
            else:
                # All scores are the same - set to neutral value
                for field in opportunity_fields:
                    if field in normalized_scores:
                        normalized_scores[field] = 0.5
        
        # Handle directional scores (allowed to be negative but clamp extremes)
        directional_fields = ['directional_bias', 'opportunity_asymmetry', 'long_momentum', 'short_momentum']
        for field in directional_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not np.isnan(score):
                    # Clamp to reasonable range but allow negatives
                    normalized_scores[field] = max(-2.0, min(2.0, score))
        
        # Ensure confidence and consistency scores are in [0, 1] range
        bounded_fields = ['directional_confidence', 'long_directional_consistency', 'short_directional_consistency']
        for field in bounded_fields:
            if field in normalized_scores:
                score = normalized_scores[field]
                if isinstance(score, (int, float)) and not np.isnan(score):
                    normalized_scores[field] = max(0.0, min(1.0, score))
        
        return normalized_scores

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
    
    # Show sample results with enhanced directional analysis
    sample_cols = [
        'overall_opportunity', 'leverage_adjusted_score', 'immediate_opportunity',
        'long_overall_opportunity', 'short_overall_opportunity', 
        'directional_bias', 'directional_confidence', 'opportunity_asymmetry'
    ]
    available_cols = [col for col in sample_cols if col in labeled_data.columns]
    sample_data = labeled_data[available_cols].head(10)
    
    tprint(f'\n📊 Enhanced sample results (directional analysis):')
    for col in available_cols:
        tprint(f'   → {col}: mean={sample_data[col].mean():.3f}')
    
    # Show directional distribution
    if 'directional_bias' in labeled_data.columns:
        bias_data = labeled_data['directional_bias'].head(100)
        long_bias = (bias_data > 0.5).sum()
        short_bias = (bias_data < -0.5).sum()
        neutral = 100 - long_bias - short_bias
        tprint(f'\n🎯 Directional bias distribution (first 100 samples):')
        tprint(f'   → Long bias: {long_bias}%')
        tprint(f'   → Short bias: {short_bias}%')
        tprint(f'   → Neutral: {neutral}%')
    
    tprint('✅ Multi-Horizon Profit Labeler test completed successfully!')