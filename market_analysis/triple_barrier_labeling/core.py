"""
Core Triple Barrier Labeling Implementation

This module provides the core triple barrier labeling functionality with multiple
implementations and comprehensive integration with the existing utility infrastructure.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import warnings
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path

# Import common utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_dataframe_operation, validate_dataframe_columns,
    create_summary_statistics, safe_convert_dtypes,
    optimize_dataframe_dtypes, safe_timestamp_conversion
)
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import MathValidation
from src.utils.serialization_utils import UniversalSerializer
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Import ML common utilities
from src.utils.ml_common.data_processing.data_labeling import EnhancedDataLabeler
from src.utils.ml_common.validation.cv_utils import TemporalCrossValidator, PurgedKFold

# Import hardware optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class BarrierType(Enum):
    """Types of barriers for triple barrier method."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    REGIME_AWARE = "regime_aware"
    FRACTIONAL = "fractional"
    # Note: Volatility-based methods removed as per requirements

class LabelingMethod(Enum):
    """Available labeling methods."""
    TRIPLE_BARRIER = "triple_barrier"
    REGIME_AWARE_TRIPLE_BARRIER = "regime_aware_triple_barrier"
    FRACTIONAL_TRIPLE_BARRIER = "fractional_triple_barrier"
    PROFIT_BASED = "profit_based"
    # Note: Volatility-based methods removed as per requirements
    CUSTOM = "custom"

@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling.
    
    Barrier Value Calculation:
    - Profit Target Price = entry_price * (1 + pt_mult)
    - Stop Loss Price = entry_price * (1 - sl_mult)
    
    Where:
    - pt_mult: Profit target multiplier (e.g., 0.002 = 0.2%)
    - sl_mult: Stop loss multiplier (e.g., 0.001 = 0.1%)
    - entry_price: The price at which the position is entered
    
    Transaction Cost:
    - Global standard transaction cost of 0.08% (0.0008) applied to all trades
    - Includes both entry and exit costs combined
    """
    # Core parameters
    pt_mult: float = 1.0  # Profit target multiplier
    sl_mult: float = 1.0  # Stop loss multiplier
    min_holding_period: int = 1
    max_holding_period: int = 100
    
    # Global transaction cost (0.08% standard)
    transaction_cost: float = 0.0008  # 0.08% - includes entry and exit costs combined
    
    # Barrier configuration
    barrier_type: BarrierType = BarrierType.FIXED
    regime_aware: bool = False
    fractional_support: bool = False
    # Note: Volatility adjustment removed as per requirements
    
    # Quality and validation
    quality_threshold: float = 0.7
    min_samples_per_label: int = 10
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 10000
    chunk_size: int = 1000
    
    # Hardware optimization
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    
    # Output settings
    save_intermediate_results: bool = True
    verbose_logging: bool = True
    output_directory: Optional[str] = None

@dataclass
class LabelingResult:
    """Result of triple barrier labeling operation."""
    labels: pd.DataFrame
    config: TripleBarrierConfig
    quality_metrics: Dict[str, Any]
    performance_stats: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    processing_time: float

class TripleBarrierLabeler:
    """Core triple barrier labeler with comprehensive functionality."""
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """Initialize the triple barrier labeler.
        
        Args:
            config: Configuration for the labeler
        """
        self.config = config or TripleBarrierConfig()
        self.logger = logging.getLogger(f"{__name__}.TripleBarrierLabeler")
        
        # Initialize utility managers
        self._initialize_utilities()
        
        # Performance tracking
        self.performance_stats = {
            'total_labels_generated': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'quality_scores': [],
            'method_usage': {}
        }
        
        # Quality metrics
        self.quality_metrics = None
        
        self.logger.info("✅ TripleBarrierLabeler initialized successfully")

    def _initialize_utilities(self):
        """Initialize utility managers and hardware optimizations."""
        self.logger.info("🔄 Initializing utility managers...")
        start_time = time.time()
        
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
            self.klines_manager = KlinesParquetManager()
            self.matrix_ops = UnifiedMatrixOperations()
            
            # Initialize enhanced data labeler
            self.enhanced_labeler = EnhancedDataLabeler()
            
            # Initialize hardware optimizations if available
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                try:
                    self.gpu_manager = get_m1_gpu_manager()
                    self.memory_optimizer = get_m1_memory_optimizer()
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    
                    # Optimize for M1 if available
                    if self.cpu_optimizer:
                        self.cpu_optimizer.optimize_numpy_operations()
                    
                    self.logger.info("✅ Hardware optimization utilities initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Hardware optimization failed: {e}")
                    self.gpu_manager = None
                    self.memory_optimizer = None
                    self.cpu_optimizer = None
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.logger.info("ℹ️ Hardware optimization not available")
            
            init_time = time.time() - start_time
            self.logger.info(f"✅ Utility managers initialized in {init_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize utilities: {e}")
            raise

    def create_labels(
        self,
        data: pd.DataFrame,
        method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER,
        regime_data: Optional[pd.DataFrame] = None,
        config: Optional[TripleBarrierConfig] = None
    ) -> LabelingResult:
        """Create triple barrier labels for market data.
        
        Args:
            data: Market data with OHLC columns
            method: Labeling method to use
            regime_data: Optional regime information
            config: Optional configuration override
            
        Returns:
            LabelingResult with labels and metadata
        """
        config = config or self.config
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting triple barrier labeling with method: {method.value}")
        self.logger.info(f"📊 Input data shape: {data.shape}")
        self.logger.info(f"⚙️ Configuration: PT={config.pt_mult}, SL={config.sl_mult}")
        
        warnings_list = []
        errors_list = []
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Select implementation based on method
            if method == LabelingMethod.TRIPLE_BARRIER:
                labels_df = self._create_standard_triple_barrier(data, config)
            elif method == LabelingMethod.REGIME_AWARE_TRIPLE_BARRIER:
                labels_df = self._create_regime_aware_triple_barrier(data, regime_data, config)
            elif method == LabelingMethod.FRACTIONAL_TRIPLE_BARRIER:
                labels_df = self._create_fractional_triple_barrier(data, config)
            elif method == LabelingMethod.PROFIT_BASED:
                labels_df = self._create_profit_based_labels(data, config)
            # Note: Volatility-based labeling removed as per requirements
            else:
                raise ValueError(f"Unsupported labeling method: {method}")
            
            # Assess label quality
            quality_metrics = self._assess_label_quality(labels_df, data)
            
            # Update performance stats
            self._update_performance_stats(start_time, len(labels_df), method.value)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = LabelingResult(
                labels=labels_df,
                config=config,
                quality_metrics=quality_metrics,
                performance_stats=self.performance_stats.copy(),
                warnings=warnings_list,
                errors=errors_list,
                processing_time=processing_time
            )
            
            self.logger.info(f"✅ Generated {len(labels_df)} labels in {processing_time:.3f}s")
            self.logger.info(f"📊 Quality score: {quality_metrics.get('overall_quality', 0.0):.3f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to create labels after {processing_time:.3f}s: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            errors_list.append(error_msg)
            
            # Return empty result with error
            return LabelingResult(
                labels=pd.DataFrame(),
                config=config,
                quality_metrics={},
                performance_stats=self.performance_stats.copy(),
                warnings=warnings_list,
                errors=errors_list,
                processing_time=processing_time
            )

    def _create_standard_triple_barrier(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create standard triple barrier labels with comprehensive validation."""
        self.logger.debug("🔄 Creating standard triple barrier labels with validation")
        
        # Calculate and validate end indices
        end_indices = self._calculate_end_indices_with_validation(data, config)
        
        result = data.copy()
        labels = []
        profit_pcts = []
        barrier_hits = []
        
        # Process in batches for memory efficiency
        batch_size = config.batch_size
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch_data = data.iloc[start_idx:end_idx]
            
            self.logger.debug(f"🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_data)} samples)")
            
            for i, (_, row) in enumerate(batch_data.iterrows()):
                global_idx = start_idx + i
                entry_price = row['close']
                
                # Validate end index for this position
                position_end_idx = end_indices[global_idx]
                if not self._validate_end_index_bounds(global_idx, position_end_idx, len(data)):
                    # Skip this position if validation fails
                    labels.append(0)
                    profit_pcts.append(0.0)
                    barrier_hits.append("validation_failed")
                    continue
                
                # Calculate barriers with numerical stability
                if entry_price <= 0:
                    labels.append(0)
                    profit_pcts.append(0.0)
                    barrier_hits.append("invalid_entry_price")
                    continue
                
                pt_price = entry_price * (1 + config.pt_mult)
                sl_price = entry_price * (1 - config.sl_mult)
                
                # Use validated future data window
                future_data = data.iloc[global_idx:position_end_idx]
                
                # Find barrier hit with improved logic
                label, profit_pct, barrier_type = self._find_barrier_hit(
                    future_data, 
                    entry_price, 
                    pt_price, 
                    sl_price, 
                    config
                )
                
                labels.append(label)
                profit_pcts.append(profit_pct)
                barrier_hits.append(barrier_type)
        
        # Add labels to result
        result['label'] = labels
        result['profit_pct'] = profit_pcts
        result['barrier_type'] = barrier_hits
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult
        result['transaction_cost'] = config.transaction_cost
        
        # Log statistics with validation info
        label_counts = pd.Series(labels).value_counts()
        validation_failures = sum(1 for bt in barrier_hits if 'validation_failed' in bt or 'invalid_entry_price' in bt)
        
        self.logger.info(f"📊 Label distribution: {label_counts.to_dict()}")
        self.logger.info(f"💰 Profit range: {min(profit_pcts):.4f} - {max(profit_pcts):.4f}")
        self.logger.info(f"🔍 Validation failures: {validation_failures}/{len(labels)} ({validation_failures/len(labels)*100:.1f}%)")
        
        return result

    def _create_regime_aware_triple_barrier(
        self, 
        data: pd.DataFrame, 
        regime_data: Optional[pd.DataFrame],
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create regime-aware triple barrier labels."""
        self.logger.info("🎯 Creating regime-aware triple barrier labels")
        
        if regime_data is None:
            self.logger.warning("⚠️ No regime data provided, falling back to standard triple barrier")
            return self._create_standard_triple_barrier(data, config)
        
        # Merge regime data
        merged_data = data.merge(regime_data, left_index=True, right_index=True, how='left')
        
        # Get unique regimes
        regimes = merged_data['regime'].unique()
        self.logger.info(f"📈 Found {len(regimes)} unique regimes: {regimes}")
        
        regime_results = []
        
        for regime in regimes:
            regime_mask = merged_data['regime'] == regime
            regime_data_subset = merged_data[regime_mask]
            
            self.logger.debug(f"🔄 Processing regime {regime} with {len(regime_data_subset)} samples")
            
            # Create labels for this regime
            regime_labels = self._create_standard_triple_barrier(regime_data_subset, config)
            regime_labels['regime'] = regime
            regime_results.append(regime_labels)
        
        # Combine results
        result = pd.concat(regime_results, ignore_index=True)
        result = result.sort_index()
        result['barrier_type'] = 'regime_aware_triple_barrier'
        
        return result

    def _create_fractional_triple_barrier(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create fractional triple barrier labels for continuous targets."""
        self.logger.debug("🔄 Creating fractional triple barrier labels")
        
        result = data.copy()
        labels = []
        profit_pcts = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            entry_price = row['close']
            
            # Calculate barriers
            pt_price = entry_price * (1 + config.pt_mult)
            sl_price = entry_price * (1 - config.sl_mult)
            
            # Find fractional barrier hit
            label, profit_pct = self._find_fractional_barrier_hit(
                data.iloc[i:], 
                entry_price, 
                pt_price, 
                sl_price, 
                config
            )
            
            labels.append(label)
            profit_pcts.append(profit_pct)
        
        result['label'] = labels
        result['profit_pct'] = profit_pcts
        result['barrier_type'] = 'fractional_triple_barrier'
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult
        
        return result

    def _create_profit_based_labels(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create profit-based labels with transaction costs."""
        self.logger.debug("🔄 Creating profit-based labels")
        
        result = data.copy()
        labels = []
        profit_pcts = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            entry_price = row['close']
            
            # Calculate profit-based barriers (including transaction costs)
            pt_price = entry_price * (1 + config.pt_mult + config.transaction_cost)
            sl_price = entry_price * (1 - config.sl_mult - config.transaction_cost)
            
            # Find barrier hit
            label, profit_pct, _ = self._find_barrier_hit(
                data.iloc[i:], 
                entry_price, 
                pt_price, 
                sl_price, 
                config
            )
            
            # Adjust for transaction costs
            if label != 0:
                profit_pct -= config.transaction_cost
            
            labels.append(label)
            profit_pcts.append(profit_pct)
        
        result['label'] = labels
        result['profit_pct'] = profit_pcts
        result['barrier_type'] = 'profit_based'
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult
        result['transaction_cost'] = config.transaction_cost
        
        return result

    # Note: Volatility-based labeling method removed as per requirements

    def _find_barrier_hit(
        self, 
        future_data: pd.DataFrame, 
        entry_price: float, 
        pt_price: float, 
        sl_price: float, 
        config: TripleBarrierConfig
    ) -> Tuple[int, float, str]:
        """Find which barrier is hit first with proper intra-bar priority logic.
        
        Barrier values are calculated as:
        - Profit Target: entry_price * (1 + pt_mult)
        - Stop Loss: entry_price * (1 - sl_mult)
        
        Where pt_mult and sl_mult are multipliers (e.g., 0.002 = 0.2%)
        
        Intra-bar priority logic:
        1. If both barriers are hit in the same bar, determine which was hit first
        2. Use opening price proximity as tie-breaker
        3. Apply global transaction cost of 0.08%
        """
        for j, (_, row) in enumerate(future_data.iterrows()):
            if j >= config.max_holding_period:
                return 0, 0.0, "time_barrier"
            
            # Check if both barriers are hit in the same bar
            pt_hit = row['high'] >= pt_price
            sl_hit = row['low'] <= sl_price
            
            if pt_hit and sl_hit:
                # Both barriers hit - use intra-bar priority logic
                return self._resolve_intra_bar_conflict(
                    row, entry_price, pt_price, sl_price, config.transaction_cost
                )
            elif pt_hit:
                # Only profit target hit
                gross_profit_pct = safe_divide(pt_price - entry_price, entry_price)
                net_profit_pct = gross_profit_pct - config.transaction_cost
                return 1, net_profit_pct, "profit_target"
            elif sl_hit:
                # Only stop loss hit
                gross_loss_pct = safe_divide(sl_price - entry_price, entry_price)
                net_loss_pct = gross_loss_pct - config.transaction_cost
                return -1, net_loss_pct, "stop_loss"
        
        return 0, 0.0, "no_hit"

    def _resolve_intra_bar_conflict(
        self, 
        row: pd.Series, 
        entry_price: float, 
        pt_price: float, 
        sl_price: float, 
        transaction_cost: float
    ) -> Tuple[int, float, str]:
        """Resolve conflicts when both barriers are hit in the same bar.
        
        Priority logic:
        1. Calculate distance from open to each barrier
        2. The closer barrier is assumed to be hit first
        3. If distances are equal, use timestamp-based tie-breaking (favor stop loss for safety)
        4. Apply transaction costs to final result
        
        Args:
            row: OHLC data for the current bar
            entry_price: Entry price for the position
            pt_price: Profit target barrier price
            sl_price: Stop loss barrier price
            transaction_cost: Global transaction cost (0.08%)
            
        Returns:
            Tuple of (label, net_profit_pct, barrier_type)
        """
        open_price = row['open']
        
        # Calculate distances from open price to each barrier
        pt_distance = abs(open_price - pt_price)
        sl_distance = abs(open_price - sl_price)
        
        # Determine which barrier is hit first based on proximity to open
        if pt_distance < sl_distance:
            # Profit target is closer to open, likely hit first
            gross_profit_pct = safe_divide(pt_price - entry_price, entry_price)
            net_profit_pct = gross_profit_pct - transaction_cost
            return 1, net_profit_pct, "profit_target_priority"
        elif sl_distance < pt_distance:
            # Stop loss is closer to open, likely hit first
            gross_loss_pct = safe_divide(sl_price - entry_price, entry_price)
            net_loss_pct = gross_loss_pct - transaction_cost
            return -1, net_loss_pct, "stop_loss_priority"
        else:
            # Equal distances - use conservative tie-breaking (favor stop loss for risk management)
            self.logger.debug(f"Equal barrier distances detected - applying conservative tie-breaking")
            gross_loss_pct = safe_divide(sl_price - entry_price, entry_price)
            net_loss_pct = gross_loss_pct - transaction_cost
            return -1, net_loss_pct, "stop_loss_tie_break"

    def _find_fractional_barrier_hit(
        self, 
        future_data: pd.DataFrame, 
        entry_price: float, 
        pt_price: float, 
        sl_price: float, 
        config: TripleBarrierConfig
    ) -> Tuple[float, float]:
        """Find fractional barrier hit for continuous targets with consistent transaction cost handling."""
        for j, (_, row) in enumerate(future_data.iterrows()):
            if j >= config.max_holding_period:
                return 0.0, 0.0
            
            # Check if both barriers are hit in the same bar
            pt_hit = row['high'] >= pt_price
            sl_hit = row['low'] <= sl_price
            
            if pt_hit and sl_hit:
                # Both barriers hit - use same priority logic as regular method
                _, net_profit_pct, barrier_type = self._resolve_intra_bar_conflict(
                    row, entry_price, pt_price, sl_price, config.transaction_cost
                )
                # Convert to fractional based on barrier type
                if "profit_target" in barrier_type:
                    hit_ratio = safe_divide(pt_price - entry_price, row['high'] - entry_price)
                    return min(1.0, hit_ratio), net_profit_pct
                else:
                    hit_ratio = safe_divide(entry_price - sl_price, entry_price - row['low'])
                    return max(-1.0, -hit_ratio), net_profit_pct
            elif pt_hit:
                # Only profit target hit
                hit_ratio = safe_divide(pt_price - entry_price, row['high'] - entry_price)
                gross_profit_pct = safe_divide(pt_price - entry_price, entry_price)
                net_profit_pct = gross_profit_pct - config.transaction_cost
                return min(1.0, hit_ratio), net_profit_pct
            elif sl_hit:
                # Only stop loss hit
                hit_ratio = safe_divide(entry_price - sl_price, entry_price - row['low'])
                gross_loss_pct = safe_divide(sl_price - entry_price, entry_price)
                net_loss_pct = gross_loss_pct - config.transaction_cost
                return max(-1.0, -hit_ratio), net_loss_pct
        
        return 0.0, 0.0

    def _assess_label_quality(
        self, 
        labels_df: pd.DataFrame, 
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess the quality of generated labels."""
        try:
            # Label distribution
            label_counts = labels_df['label'].value_counts()
            total_labels = len(labels_df)
            label_distribution = {
                'positive': safe_divide(label_counts.get(1, 0), total_labels),
                'negative': safe_divide(label_counts.get(-1, 0), total_labels),
                'neutral': safe_divide(label_counts.get(0, 0), total_labels)
            }
            
            # Regime balance (if regime-aware)
            regime_balance = {}
            if 'regime' in labels_df.columns:
                regime_counts = labels_df['regime'].value_counts()
                regime_balance = {
                    regime: safe_divide(count, total_labels) 
                    for regime, count in regime_counts.items()
                }
            
            # Temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(labels_df)
            
            # Profit consistency
            profit_consistency = self._calculate_profit_consistency(labels_df)
            
            # Overall quality score
            overall_quality = (
                temporal_consistency * 0.3 +
                profit_consistency * 0.3 +
                (1 - abs(label_distribution['positive'] - label_distribution['negative'])) * 0.4
            )
            
            return {
                'label_distribution': label_distribution,
                'regime_balance': regime_balance,
                'temporal_consistency': temporal_consistency,
                'profit_consistency': profit_consistency,
                'overall_quality': overall_quality,
                'total_labels': total_labels,
                'label_counts': label_counts.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to assess label quality: {e}")
            return {
                'label_distribution': {},
                'regime_balance': {},
                'temporal_consistency': 0.0,
                'profit_consistency': 0.0,
                'overall_quality': 0.0,
                'total_labels': 0,
                'label_counts': {},
                'error': str(e)
            }

    def _calculate_temporal_consistency(self, labels_df: pd.DataFrame) -> float:
        """Calculate temporal consistency of labels."""
        try:
            labels = labels_df['label'].values
            transitions = np.diff(labels)
            consistent_transitions = np.sum(transitions == 0)
            total_transitions = len(transitions)
            return safe_divide(consistent_transitions, total_transitions)
        except Exception:
            return 0.0

    def _calculate_profit_consistency(self, labels_df: pd.DataFrame) -> float:
        """Calculate profit consistency of labels."""
        try:
            profits = labels_df['profit_pct'].values
            positive_profits = profits[profits > 0]
            negative_profits = profits[profits < 0]
            
            if len(positive_profits) == 0 or len(negative_profits) == 0:
                return 0.0
            
            # Calculate coefficient of variation
            pos_cv = safe_divide(np.std(positive_profits), np.mean(positive_profits))
            neg_cv = safe_divide(np.std(negative_profits), np.mean(negative_profits))
            
            # Lower CV indicates higher consistency
            consistency = 1.0 - min(1.0, (pos_cv + neg_cv) / 2)
            return max(0.0, consistency)
        except Exception:
            return 0.0

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data for labeling."""
        self.logger.debug("🔍 Validating input data...")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < 10:
            raise ValueError("Insufficient data for labeling (minimum 10 rows required)")
        
        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found in price data: {null_counts.to_dict()}")
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            self.logger.warning(f"⚠️ Found {invalid_count} rows with invalid OHLC relationships")
            if invalid_count > len(data) * 0.01:  # More than 1% invalid
                raise ValueError(f"Too many invalid OHLC relationships: {invalid_count} out of {len(data)} rows")
        
        self.logger.debug("✅ Input data validation passed")

    def _calculate_end_indices_with_validation(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> np.ndarray:
        """Calculate end indices with comprehensive validation and temporal leakage detection.
        
        Args:
            data: Input market data
            config: Triple barrier configuration
            
        Returns:
            Array of validated end indices
            
        Raises:
            ValueError: If temporal leakage is detected or validation fails
        """
        n = len(data)
        
        # Calculate base end indices
        end_indices = np.minimum(
            np.arange(n) + config.max_holding_period,
            n
        )
        
        # Validate end indices for temporal consistency
        self._validate_temporal_consistency(end_indices, n, config)
        
        # Detect potential temporal leakage
        self._detect_temporal_leakage(data, end_indices, config)
        
        return end_indices
    
    def _validate_temporal_consistency(
        self, 
        end_indices: np.ndarray, 
        data_length: int, 
        config: TripleBarrierConfig
    ):
        """Validate temporal consistency of end indices.
        
        Args:
            end_indices: Array of end indices
            data_length: Length of the data
            config: Configuration object
            
        Raises:
            ValueError: If validation fails
        """
        # Check bounds
        if np.any(end_indices < 0):
            raise ValueError("Negative end indices detected")
        
        if np.any(end_indices > data_length):
            raise ValueError(f"End indices exceed data length: max={np.max(end_indices)}, data_length={data_length}")
        
        # Check for reasonable lookahead
        max_lookahead = np.max(end_indices - np.arange(len(end_indices)))
        if max_lookahead > config.max_holding_period * 1.1:  # Allow 10% tolerance
            self.logger.warning(f"⚠️ Unusually large lookahead detected: {max_lookahead} > {config.max_holding_period}")
        
        # Check for minimum lookahead
        min_lookahead = np.min(end_indices - np.arange(len(end_indices)))
        if min_lookahead < config.min_holding_period:
            self.logger.warning(f"⚠️ Lookahead below minimum: {min_lookahead} < {config.min_holding_period}")
        
        self.logger.debug(f"✅ Temporal consistency validation passed: lookahead range [{min_lookahead}, {max_lookahead}]")
    
    def _detect_temporal_leakage(
        self, 
        data: pd.DataFrame, 
        end_indices: np.ndarray, 
        config: TripleBarrierConfig
    ):
        """Detect potential temporal leakage in the labeling process.
        
        Args:
            data: Input market data
            end_indices: Array of end indices
            config: Configuration object
            
        Raises:
            ValueError: If temporal leakage is detected
        """
        n = len(data)
        leakage_detected = False
        leakage_issues = []
        
        # Check for future information usage
        for i in range(min(100, n - 1)):  # Sample check first 100 points
            end_idx = end_indices[i]
            
            # Ensure we're not using future information beyond the specified lookahead
            expected_max_end = i + config.max_holding_period
            if end_idx > expected_max_end + 1:  # Allow 1 bar tolerance
                leakage_detected = True
                leakage_issues.append(f"Row {i}: end_idx={end_idx} > expected_max={expected_max_end}")
            
            # Check that we have sufficient future data for labeling
            if end_idx <= i + config.min_holding_period:
                if i < n - config.max_holding_period:  # Only flag if we should have more data
                    leakage_issues.append(f"Row {i}: insufficient lookahead, end_idx={end_idx} <= {i + config.min_holding_period}")
        
        if leakage_detected:
            error_msg = f"Temporal leakage detected in {len(leakage_issues)} cases. Examples: {leakage_issues[:3]}"
            self.logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Check for systematic issues
        avg_lookahead = np.mean(end_indices[:n-1] - np.arange(n-1))
        if avg_lookahead > config.max_holding_period * 0.9:
            self.logger.warning(f"⚠️ Average lookahead suspiciously high: {avg_lookahead:.2f}")
        
        self.logger.debug(f"✅ Temporal leakage detection passed: avg_lookahead={avg_lookahead:.2f}")
    
    def _validate_end_index_bounds(self, i: int, end_idx: int, data_length: int) -> bool:
        """Validate that end index is within acceptable bounds for position i.
        
        Args:
            i: Current position index
            end_idx: Calculated end index
            data_length: Total length of data
            
        Returns:
            True if valid, False otherwise
        """
        # Basic bounds check
        if end_idx <= i:
            return False
        
        if end_idx > data_length:
            return False
        
        # Minimum future data requirement
        if end_idx <= i + 1:  # Need at least 1 future bar
            return False
        
        return True

    def _update_performance_stats(self, start_time: float, num_labels: int, method: str):
        """Update performance statistics."""
        processing_time = time.time() - start_time
        
        self.performance_stats['total_labels_generated'] += num_labels
        self.performance_stats['processing_time'] += processing_time
        
        if method not in self.performance_stats['method_usage']:
            self.performance_stats['method_usage'][method] = 0
        self.performance_stats['method_usage'][method] += 1
        
        self.logger.debug(f"📊 Performance stats updated: {num_labels} labels in {processing_time:.3f}s")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        labels_per_second = safe_divide(
            self.performance_stats['total_labels_generated'],
            self.performance_stats['processing_time']
        )
        
        return {
            'total_labels_generated': self.performance_stats['total_labels_generated'],
            'total_processing_time': self.performance_stats['processing_time'],
            'labels_per_second': labels_per_second,
            'method_usage': self.performance_stats['method_usage']
        }

    def save_labels(self, result: LabelingResult, filepath: str) -> bool:
        """Save labeling result to file."""
        try:
            if self.serializer:
                return self.serializer.save(result.labels, filepath)
            else:
                result.labels.to_parquet(filepath)
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save labels: {e}")
            return False

    def load_labels(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load labels from file."""
        try:
            if self.serializer:
                return self.serializer.load(filepath)
            else:
                return pd.read_parquet(filepath)
        except Exception as e:
            self.logger.error(f"❌ Failed to load labels: {e}")
            return None