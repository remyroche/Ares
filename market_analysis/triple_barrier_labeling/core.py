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
    VOLATILITY_ADJUSTED = "volatility_adjusted"

class LabelingMethod(Enum):
    """Available labeling methods."""
    TRIPLE_BARRIER = "triple_barrier"
    REGIME_AWARE_TRIPLE_BARRIER = "regime_aware_triple_barrier"
    FRACTIONAL_TRIPLE_BARRIER = "fractional_triple_barrier"
    PROFIT_BASED = "profit_based"
    VOLATILITY_BASED = "volatility_based"
    CUSTOM = "custom"

@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    # Core parameters
    pt_mult: float = 1.0  # Profit target multiplier
    sl_mult: float = 1.0  # Stop loss multiplier
    min_holding_period: int = 1
    max_holding_period: int = 100
    
    # Transaction costs and fees
    transaction_cost: float = 0.001
    spread_cost: float = 0.0005
    
    # Barrier configuration
    barrier_type: BarrierType = BarrierType.FIXED
    regime_aware: bool = False
    fractional_support: bool = False
    volatility_adjusted: bool = False
    
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
            elif method == LabelingMethod.VOLATILITY_BASED:
                labels_df = self._create_volatility_based_labels(data, config)
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
        """Create standard triple barrier labels."""
        self.logger.debug("🔄 Creating standard triple barrier labels")
        
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
                entry_price = row['close']
                
                # Calculate barriers
                pt_price = entry_price * (1 + config.pt_mult)
                sl_price = entry_price * (1 - config.sl_mult)
                
                # Find barrier hit
                label, profit_pct, barrier_type = self._find_barrier_hit(
                    data.iloc[start_idx + i:], 
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
        
        # Log statistics
        label_counts = pd.Series(labels).value_counts()
        self.logger.info(f"📊 Label distribution: {label_counts.to_dict()}")
        self.logger.info(f"💰 Profit range: {min(profit_pcts):.4f} - {max(profit_pcts):.4f}")
        
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

    def _create_volatility_based_labels(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create volatility-adjusted triple barrier labels."""
        self.logger.debug("🔄 Creating volatility-based labels")
        
        # Calculate rolling volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std()
        
        result = data.copy()
        labels = []
        profit_pcts = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            if i < 20:  # Skip first 20 periods for volatility calculation
                labels.append(0)
                profit_pcts.append(0.0)
                continue
                
            entry_price = row['close']
            current_volatility = volatility.iloc[i]
            
            # Adjust barriers based on volatility
            vol_multiplier = 1 + current_volatility
            pt_mult = config.pt_mult * vol_multiplier
            sl_mult = config.sl_mult * vol_multiplier
            
            pt_price = entry_price * (1 + pt_mult)
            sl_price = entry_price * (1 - sl_mult)
            
            # Find barrier hit
            label, profit_pct, _ = self._find_barrier_hit(
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
        result['barrier_type'] = 'volatility_based'
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult
        result['volatility'] = volatility
        
        return result

    def _find_barrier_hit(
        self, 
        future_data: pd.DataFrame, 
        entry_price: float, 
        pt_price: float, 
        sl_price: float, 
        config: TripleBarrierConfig
    ) -> Tuple[int, float, str]:
        """Find which barrier is hit first."""
        for j, (_, row) in enumerate(future_data.iterrows()):
            if j >= config.max_holding_period:
                return 0, 0.0, "time_barrier"
            
            if row['high'] >= pt_price:
                profit_pct = safe_divide(pt_price - entry_price, entry_price)
                return 1, profit_pct, "profit_target"
            elif row['low'] <= sl_price:
                profit_pct = safe_divide(sl_price - entry_price, entry_price)
                return -1, profit_pct, "stop_loss"
        
        return 0, 0.0, "no_hit"

    def _find_fractional_barrier_hit(
        self, 
        future_data: pd.DataFrame, 
        entry_price: float, 
        pt_price: float, 
        sl_price: float, 
        config: TripleBarrierConfig
    ) -> Tuple[float, float]:
        """Find fractional barrier hit for continuous targets."""
        for j, (_, row) in enumerate(future_data.iterrows()):
            if j >= config.max_holding_period:
                return 0.0, 0.0
            
            # Calculate fractional hit
            if row['high'] >= pt_price:
                hit_ratio = safe_divide(pt_price - entry_price, row['high'] - entry_price)
                profit_pct = safe_divide(pt_price - entry_price, entry_price)
                return min(1.0, hit_ratio), profit_pct
            elif row['low'] <= sl_price:
                hit_ratio = safe_divide(entry_price - sl_price, entry_price - row['low'])
                profit_pct = safe_divide(sl_price - entry_price, entry_price)
                return max(-1.0, -hit_ratio), profit_pct
        
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
        
        self.logger.debug("✅ Input data validation passed")

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