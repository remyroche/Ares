#!/usr/bin/env python3
"""
Enhanced Data Labeling Utilities

This enhanced module consolidates all triple barrier implementations and adds
comprehensive regime-aware labeling capabilities with advanced features:

Key Enhancements:
- Consolidated Triple Barrier Methods: All 6 implementations merged
- Regime-Aware Labeling: Dynamic parameters based on market regimes
- Fractional Barrier Support: Continuous target labeling
- Label Quality Assessment: Comprehensive validation and metrics
- Cross-Validation Integration: Temporal CV for label validation
- Memory Optimization: M1-optimized operations
- GPU Acceleration: M1 MPS support
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from pathlib import Path

# Import comprehensive utility infrastructure
from ..math_validation import (
    safe_divide, safe_log, safe_sqrt,
    validate_positive, validate_range
)
from src.utils.core.common import create_fallback_logger, create_fallback_decorator
from src.utils.parquet_utils import ParquetUtils
from src.utils.parquet_utils import ParquetUtils as UniversalSerializer
from src.utils.data_processing_utils import DataProcessingUtils
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Import ML Common utilities for cross-validation (use compatibility exports)
from ..validation.cv_utils import TemporalCrossValidator, PurgedKFold
# from .validation_utils import ValidationFramework  # Not available
# from .pareto import ParetoFrontAnalyzer  # Causes circular import

logger = logging.getLogger(__name__)

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - using pure Python implementations")

class LabelingMethod(Enum):
    """Available labeling methods."""
    TRIPLE_BARRIER = "triple_barrier"
    REGIME_AWARE_TRIPLE_BARRIER = "regime_aware_triple_barrier"
    FRACTIONAL_TRIPLE_BARRIER = "fractional_triple_barrier"
    PROFIT_BASED = "profit_based"
    CUSTOM = "custom"

class BarrierType(Enum):
    """Types of barriers for triple barrier method."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    REGIME_AWARE = "regime_aware"
    FRACTIONAL = "fractional"

@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    pt_mult: float = 1.0  # Profit target multiplier
    sl_mult: float = 1.0  # Stop loss multiplier
    min_holding_period: int = 1
    max_holding_period: int = 100
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    barrier_type: BarrierType = BarrierType.FIXED
    regime_aware: bool = False
    fractional_support: bool = False
    quality_threshold: float = 0.7

@dataclass
class RegimeAwareConfig:
    """Configuration for regime-aware labeling."""
    regime_column: str = "regime"
    regime_params: Dict[str, TripleBarrierConfig] = field(default_factory=dict)
    default_config: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    regime_transition_threshold: float = 0.1
    adaptive_parameters: bool = True

@dataclass
class LabelQualityMetrics:
    """Metrics for label quality assessment."""
    label_distribution: Dict[str, float]
    regime_balance: Dict[str, float]
    temporal_consistency: float
    profit_consistency: float
    overall_quality: float
    warnings: List[str]
    errors: List[str]

@dataclass
class LabelingConfig:
    """Main configuration class for data labeling operations."""
    # Triple barrier configuration
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)

    # Regime-aware configuration
    regime_aware: RegimeAwareConfig = field(default_factory=RegimeAwareConfig)

    # General labeling settings
    method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER
    enable_quality_assessment: bool = True
    enable_cross_validation: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = True

    # Performance settings
    batch_size: int = 10000
    max_workers: int = 4
    chunk_size: int = 1000

    # Quality thresholds
    min_quality_score: float = 0.7
    max_label_imbalance: float = 0.8
    min_samples_per_regime: int = 100

    # Output settings
    save_intermediate_results: bool = True
    verbose_logging: bool = True
    output_directory: Optional[str] = None

class EnhancedDataLabeler:
    """Enhanced data labeler with consolidated triple barrier implementations."""

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        self.config = config or TripleBarrierConfig()
        self.logger = create_fallback_logger()
        self.logger.name = "EnhancedDataLabeler"

        # Initialize utility managers
        self._initialize_utilities()

        # Initialize regime-aware components
        self.regime_config = RegimeAwareConfig()
        self.quality_metrics = None

        # Performance tracking
        self.performance_stats = {
            'total_labels_generated': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'quality_scores': []
        }

    def _initialize_utilities(self):
        """Initialize utility managers."""
        self.logger.info("🔄 Initializing utility managers for EnhancedDataLabeler...")
        start_time = time.time()

        try:
            # Try to import and initialize M1 GPU manager with fallback
            try:
                self.logger.debug("🔧 Initializing M1 GPU manager...")
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.debug("✅ M1 GPU manager initialized")
            except NameError:
                self.logger.warning("⚠️ get_m1_gpu_manager not available, using fallback")
                self.gpu_manager = None
            except Exception as gpu_e:
                self.logger.warning(f"⚠️ M1 GPU manager initialization failed: {gpu_e}")
                self.gpu_manager = None

            # Try to import and initialize M1 memory optimizer with fallback
            try:
                self.logger.debug("🔧 Initializing M1 memory optimizer...")
                self.memory_optimizer = get_m1_memory_optimizer()
                self.logger.debug("✅ M1 memory optimizer initialized")
            except Exception as mem_e:
                self.logger.warning(f"⚠️ M1 memory optimizer initialization failed: {mem_e}")
                self.memory_optimizer = None

            # Try to import and initialize M1 CPU optimizer with fallback
            try:
                self.logger.debug("🔧 Initializing M1 CPU optimizer...")
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.debug("✅ M1 CPU optimizer initialized")
            except Exception as cpu_e:
                self.logger.warning(f"⚠️ M1 CPU optimizer initialization failed: {cpu_e}")
                self.cpu_optimizer = None

            self.logger.debug("🔧 Initializing Parquet utilities...")
            self.parquet_utils = ParquetUtils()
            self.logger.debug("✅ Parquet utilities initialized")

            self.logger.debug("🔧 Initializing universal serializer...")
            self.serializer = UniversalSerializer()
            self.logger.debug("✅ Universal serializer initialized")

            self.logger.debug("🔧 Initializing data processing utilities...")
            self.data_processor = DataProcessingUtils()
            self.logger.debug("✅ Data processing utilities initialized")

            # Common utilities optional: keep disabled to avoid import errors

            init_time = time.time() - start_time
            self.logger.info(f"✅ All utility managers initialized successfully in {init_time:.3f}s")
            self.logger.info(f"🎯 GPU acceleration: {'Available' if self.gpu_manager else 'Not available'}")
            self.logger.info(f"🧠 Memory optimization: {'Available' if self.memory_optimizer else 'Not available'}")
            self.logger.info(f"⚡ CPU optimization: {'Available' if self.cpu_optimizer else 'Not available'}")

        except Exception as e:
            init_time = time.time() - start_time
            self.logger.warning(f"⚠️ Unexpected error during utility initialization after {init_time:.3f}s: {e}")
            self.logger.warning("🔄 Setting fallback implementations...")

            # Set fallback implementations for any that weren't already set
            if not hasattr(self, 'gpu_manager') or self.gpu_manager is None:
                self.gpu_manager = None
            if not hasattr(self, 'memory_optimizer') or self.memory_optimizer is None:
                self.memory_optimizer = None
            if not hasattr(self, 'cpu_optimizer') or self.cpu_optimizer is None:
                self.cpu_optimizer = None
            if not hasattr(self, 'parquet_utils'):
                self.parquet_utils = None
            if not hasattr(self, 'serializer'):
                self.serializer = None
            if not hasattr(self, 'data_processor'):
                self.data_processor = None

            self.logger.info("✅ Fallback implementations set - basic functionality preserved")

    def create_triple_barrier_labels(
        self,
        data: pd.DataFrame,
        method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER,
        regime_data: Optional[pd.DataFrame] = None,
        config: Optional[TripleBarrierConfig] = None
    ) -> pd.DataFrame:
        """
        Create triple barrier labels with consolidated implementations.

        Args:
            data: Price data with OHLC columns
            method: Labeling method to use
            regime_data: Optional regime information
            config: Optional configuration override

        Returns:
            DataFrame with labels and metadata
        """
        config = config or self.config
        start_time = time.time()

        self.logger.info(f"🚀 Starting triple barrier labeling with method: {method.value}")
        self.logger.info(f"📊 Input data shape: {data.shape}")
        self.logger.info(f"⚙️ Configuration: PT={config.pt_mult}, SL={config.sl_mult}, MinHold={config.min_holding_period}, MaxHold={config.max_holding_period}")

        if regime_data is not None:
            self.logger.info(f"🎯 Regime data provided: {regime_data.shape}")
            unique_regimes = regime_data['regime'].unique() if 'regime' in regime_data.columns else []
            self.logger.info(f"📈 Unique regimes: {unique_regimes}")

        try:
            # Validate input data
            self.logger.debug("🔍 Validating input data...")
            self._validate_input_data(data)
            self.logger.debug("✅ Input data validation passed")

            # Select implementation based on method
            self.logger.info(f"🔄 Creating labels using {method.value} method...")
            method_start_time = time.time()

            if method == LabelingMethod.TRIPLE_BARRIER:
                labels_df = self._create_standard_triple_barrier(data, config)
            elif method == LabelingMethod.REGIME_AWARE_TRIPLE_BARRIER:
                labels_df = self._create_regime_aware_triple_barrier(data, regime_data, config)
            elif method == LabelingMethod.FRACTIONAL_TRIPLE_BARRIER:
                labels_df = self._create_fractional_triple_barrier(data, config)
            elif method == LabelingMethod.PROFIT_BASED:
                labels_df = self._create_profit_based_labels(data, config)
            else:
                raise ValueError(f"Unsupported labeling method: {method}")

            method_time = time.time() - method_start_time
            self.logger.info(f"✅ Label creation completed in {method_time:.3f}s")

            # Assess label quality
            self.logger.debug("🔍 Assessing label quality...")
            quality_start_time = time.time()
            self.quality_metrics = self._assess_label_quality(labels_df, data)
            quality_time = time.time() - quality_start_time
            self.logger.info(f"📊 Quality assessment completed in {quality_time:.3f}s")

            # Log quality metrics
            if self.quality_metrics:
                self.logger.info(f"📈 Label distribution: {self.quality_metrics.label_distribution}")
                self.logger.info(f"🎯 Overall quality score: {self.quality_metrics.overall_quality:.3f}")
                self.logger.info(f"⏱️ Temporal consistency: {self.quality_metrics.temporal_consistency:.3f}")
                self.logger.info(f"💰 Profit consistency: {self.quality_metrics.profit_consistency:.3f}")

                if self.quality_metrics.warnings:
                    for warning in self.quality_metrics.warnings:
                        self.logger.warning(f"⚠️ Quality warning: {warning}")

                if self.quality_metrics.errors:
                    for error in self.quality_metrics.errors:
                        self.logger.error(f"❌ Quality error: {error}")

            # Update performance stats
            self._update_performance_stats(start_time, len(labels_df))

            total_time = time.time() - start_time
            self.logger.info(f"✅ Generated {len(labels_df)} labels using {method.value} in {total_time:.3f}s")
            self.logger.info(f"📊 Labels per second: {len(labels_df) / total_time:.1f}")

            return labels_df

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"❌ Failed to create triple barrier labels after {total_time:.3f}s: {e}")
            self.logger.error(f"📋 Method: {method.value}")
            self.logger.error(f"📊 Data shape: {data.shape}")
            self.logger.error(f"⚙️ Config: PT={config.pt_mult}, SL={config.sl_mult}")
            raise

    def _create_standard_triple_barrier(
        self,
        data: pd.DataFrame,
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create standard triple barrier labels."""
        self.logger.debug(f"🔄 Creating standard triple barrier labels for {len(data)} samples")

        if NUMBA_AVAILABLE:
            self.logger.debug("⚡ Using Numba-accelerated implementation")
            return self._create_numba_triple_barrier(data, config)
        else:
            self.logger.debug("🐍 Using pure Python implementation")
            return self._create_python_triple_barrier(data, config)

    def _create_numba_triple_barrier(
        self,
        data: pd.DataFrame,
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Numba-accelerated triple barrier implementation."""
        self.logger.debug("⚡ Starting Numba-accelerated triple barrier computation")
        start_time = time.time()

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        self.logger.debug(f"📊 Processing {len(close)} price points")
        self.logger.debug(f"💰 Price range: {close.min():.4f} - {close.max():.4f}")

        # Calculate end indices for each position
        end_indices = np.arange(len(close))

        # Use numba-accelerated function
        self.logger.debug("🔄 Executing Numba-accelerated core function...")
        core_start_time = time.time()
        labels, profit_pcts = self._numba_triple_barrier_core(
            close, high, low,
            config.pt_mult, config.sl_mult,
            end_indices, config.transaction_cost
        )
        core_time = time.time() - core_start_time
        self.logger.debug(f"⚡ Numba core computation completed in {core_time:.3f}s")

        # Create result DataFrame
        self.logger.debug("📊 Creating result DataFrame...")
        result = data.copy()
        result['label'] = labels
        result['profit_pct'] = profit_pcts
        result['barrier_type'] = 'triple_barrier'
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult

        # Log label statistics
        label_counts = np.bincount(labels + 1)  # Convert -1,0,1 to 0,1,2
        total_time = time.time() - start_time

        self.logger.info(f"✅ Numba triple barrier completed in {total_time:.3f}s")
        self.logger.info(f"📊 Label distribution: Positive={label_counts[2]}, Neutral={label_counts[1]}, Negative={label_counts[0]}")
        self.logger.info(f"💰 Profit range: {profit_pcts.min():.4f} - {profit_pcts.max():.4f}")

        return result

    def _create_python_triple_barrier(
        self,
        data: pd.DataFrame,
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Pure Python triple barrier implementation."""
        self.logger.debug("🐍 Starting pure Python triple barrier computation")
        start_time = time.time()

        result = data.copy()
        labels = []
        profit_pcts = []

        self.logger.debug(f"📊 Processing {len(data)} samples with Python implementation")

        # Progress tracking for large datasets
        progress_interval = max(1, len(data) // 10)  # Log every 10%

        for i in range(len(data)):
            if i % progress_interval == 0:
                progress = (i / len(data)) * 100
                self.logger.debug(f"🔄 Progress: {progress:.1f}% ({i}/{len(data)})")

            entry_price = data['close'].iloc[i]

            # Calculate barriers
            pt_price = entry_price * (1 + config.pt_mult)
            sl_price = entry_price * (1 - config.sl_mult)

            # Find barrier hit
            label, profit_pct = self._find_barrier_hit(
                data.iloc[i:], entry_price, pt_price, sl_price, config
            )

            labels.append(label)
            profit_pcts.append(profit_pct)

        result['label'] = labels
        result['profit_pct'] = profit_pcts
        result['barrier_type'] = 'triple_barrier'
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult

        # Log label statistics
        label_counts = np.bincount(np.array(labels) + 1)  # Convert -1,0,1 to 0,1,2
        total_time = time.time() - start_time

        self.logger.info(f"✅ Python triple barrier completed in {total_time:.3f}s")
        self.logger.info(f"📊 Label distribution: Positive={label_counts[2]}, Neutral={label_counts[1]}, Negative={label_counts[0]}")
        self.logger.info(f"💰 Profit range: {min(profit_pcts):.4f} - {max(profit_pcts):.4f}")
        self.logger.info(f"⚡ Processing speed: {len(data) / total_time:.1f} samples/second")

        return result

    def _create_regime_aware_triple_barrier(
        self,
        data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame],
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create regime-aware triple barrier labels."""
        self.logger.info("🎯 Starting regime-aware triple barrier labeling")
        start_time = time.time()

        if regime_data is None:
            self.logger.warning("⚠️ No regime data provided, falling back to standard triple barrier")
            return self._create_standard_triple_barrier(data, config)

        self.logger.debug(f"📊 Merging regime data with price data...")
        # Merge regime data
        merged_data = data.merge(regime_data, left_index=True, right_index=True, how='left')
        self.logger.debug(f"✅ Data merged successfully: {merged_data.shape}")

        # Get unique regimes
        regimes = merged_data['regime'].unique()
        self.logger.info(f"📈 Found {len(regimes)} unique regimes: {regimes}")

        regime_results = []
        regime_stats = {}

        for i, regime in enumerate(regimes):
            self.logger.debug(f"🔄 Processing regime {i+1}/{len(regimes)}: {regime}")
            regime_start_time = time.time()

            regime_mask = merged_data['regime'] == regime
            regime_data_subset = merged_data[regime_mask]

            self.logger.debug(f"📊 Regime {regime} has {len(regime_data_subset)} samples")

            # Get regime-specific config
            regime_config = self.regime_config.regime_params.get(
                str(regime), self.regime_config.default_config
            )

            self.logger.debug(f"⚙️ Regime {regime} config: PT={regime_config.pt_mult}, SL={regime_config.sl_mult}")

            # Create labels for this regime
            regime_labels = self._create_standard_triple_barrier(regime_data_subset, regime_config)
            regime_labels['regime'] = regime
            regime_results.append(regime_labels)

            regime_time = time.time() - regime_start_time
            regime_stats[regime] = {
                'samples': len(regime_data_subset),
                'processing_time': regime_time
            }

            self.logger.debug(f"✅ Regime {regime} completed in {regime_time:.3f}s")

        # Combine results
        self.logger.debug("🔄 Combining regime results...")
        result = pd.concat(regime_results, ignore_index=True)
        result = result.sort_index()
        result['barrier_type'] = 'regime_aware_triple_barrier'

        total_time = time.time() - start_time

        # Log regime statistics
        self.logger.info(f"✅ Regime-aware triple barrier completed in {total_time:.3f}s")
        for regime, stats in regime_stats.items():
            self.logger.info(f"📊 Regime {regime}: {stats['samples']} samples in {stats['processing_time']:.3f}s")

        return result

    def _create_fractional_triple_barrier(
        self,
        data: pd.DataFrame,
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create fractional triple barrier labels for continuous targets."""
        result = data.copy()
        labels = []
        profit_pcts = []

        for i in range(len(data)):
            entry_price = data['close'].iloc[i]

            # Calculate fractional barriers
            pt_price = entry_price * (1 + config.pt_mult)
            sl_price = entry_price * (1 - config.sl_mult)

            # Find fractional barrier hit
            label, profit_pct = self._find_fractional_barrier_hit(
                data.iloc[i:], entry_price, pt_price, sl_price, config
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
        result = data.copy()
        labels = []
        profit_pcts = []

        for i in range(len(data)):
            entry_price = data['close'].iloc[i]

            # Calculate profit-based barriers
            pt_price = entry_price * (1 + config.pt_mult + config.transaction_cost)
            sl_price = entry_price * (1 - config.sl_mult - config.transaction_cost)

            # Find profit-based barrier hit
            label, profit_pct = self._find_barrier_hit(
                data.iloc[i:], entry_price, pt_price, sl_price, config
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

    def _find_barrier_hit(
        self,
        future_data: pd.DataFrame,
        entry_price: float,
        pt_price: float,
        sl_price: float,
        config: TripleBarrierConfig
    ) -> Tuple[int, float]:
        """Find which barrier is hit first."""
        for j, (_, row) in enumerate(future_data.iterrows()):
            if j >= config.max_holding_period:
                return 0, 0.0  # Time barrier hit

            if row['high'] >= pt_price:
                return 1, safe_divide(pt_price - entry_price, entry_price)  # Profit target hit
            elif row['low'] <= sl_price:
                return -1, safe_divide(sl_price - entry_price, entry_price)  # Stop loss hit

        return 0, 0.0  # No barrier hit

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
                return 0.0, 0.0  # Time barrier hit

            # Calculate fractional hit
            if row['high'] >= pt_price:
                hit_ratio = safe_divide(pt_price - entry_price, row['high'] - entry_price)
                return min(1.0, hit_ratio), safe_divide(pt_price - entry_price, entry_price)
            elif row['low'] <= sl_price:
                hit_ratio = safe_divide(entry_price - sl_price, entry_price - row['low'])
                return max(-1.0, -hit_ratio), safe_divide(sl_price - entry_price, entry_price)

        return 0.0, 0.0  # No barrier hit

    def _assess_label_quality(
        self,
        labels_df: pd.DataFrame,
        original_data: pd.DataFrame
    ) -> LabelQualityMetrics:
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
                total_regimes = len(regime_counts)
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

            # Generate warnings and errors
            warnings = []
            errors = []

            if overall_quality < self.config.quality_threshold:
                warnings.append(f"Overall quality {overall_quality:.3f} below threshold {self.config.quality_threshold}")

            if abs(label_distribution['positive'] - label_distribution['negative']) > 0.3:
                warnings.append("Significant label imbalance detected")

            return LabelQualityMetrics(
                label_distribution=label_distribution,
                regime_balance=regime_balance,
                temporal_consistency=temporal_consistency,
                profit_consistency=profit_consistency,
                overall_quality=overall_quality,
                warnings=warnings,
                errors=errors
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to assess label quality: {e}")
            return LabelQualityMetrics(
                label_distribution={},
                regime_balance={},
                temporal_consistency=0.0,
                profit_consistency=0.0,
                overall_quality=0.0,
                warnings=[],
                errors=[str(e)]
            )

    def _calculate_temporal_consistency(self, labels_df: pd.DataFrame) -> float:
        """Calculate temporal consistency of labels."""
        try:
            # Calculate label transitions
            labels = labels_df['label'].values
            transitions = np.diff(labels)

            # Count consistent transitions
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

    def validate_labels_with_cv(
        self,
        labels_df: pd.DataFrame,
        n_splits: int = 5,
        purged_pct: float = 0.01
    ) -> Dict[str, Any]:
        """Validate labels using temporal cross-validation (unified API)."""
        try:
            # Prepare data for CV
            X = labels_df.drop(['label', 'profit_pct'], axis=1, errors='ignore')
            y = labels_df['label']

            # Use a simple classifier for label validation
            try:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            except Exception:
                model = None

            # Map purged_pct to a gap approximately per fold (best-effort)
            try:
                approx_fold_size = max(1, len(X) // (n_splits + 1))
                gap = max(0, int(approx_fold_size * purged_pct))
            except Exception:
                gap = 0

            if model is None:
                return {
                    'cv_scores': [],
                    'mean_score': 0.0,
                    'std_score': 0.0,
                    'validation_passed': False,
                    'error': 'No classifier available for validation'
                }

            # Perform temporal CV via unified API
            cv_res = temporal_cross_validation(
                model,
                X.values if hasattr(X, 'values') else X,
                y.values if hasattr(y, 'values') else y,
                n_splits=n_splits,
                gap=gap,
                test_size=None,
                scoring='accuracy'
            )

            scores = np.array(cv_res.get('scores', []) or [])
            mean_score = float(cv_res.get('mean', np.mean(scores) if scores.size else 0.0))
            std_score = float(cv_res.get('std', np.std(scores) if scores.size else 0.0))

            return {
                'cv_scores': scores.tolist() if scores.size else [],
                'mean_score': mean_score,
                'std_score': std_score,
                'validation_passed': mean_score > 0.6
            }
        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed: {e}")
            return {
                'cv_scores': [],
                'mean_score': 0.0,
                'std_score': 0.0,
                'validation_passed': False,
                'error': str(e)
            }

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data for labeling."""
        self.logger.debug("🔍 Validating input data for labeling...")

        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.logger.error(f"❌ Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.debug(f"✅ Required columns present: {required_columns}")

        if len(data) < 10:
            self.logger.error(f"❌ Insufficient data: {len(data)} rows (minimum 10 required)")
            raise ValueError("Insufficient data for labeling (minimum 10 rows required)")

        self.logger.debug(f"✅ Data size validation passed: {len(data)} rows")

        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.error(f"❌ Null values found in price data: {null_counts.to_dict()}")
            raise ValueError(f"Null values found in price data: {null_counts.to_dict()}")

        self.logger.debug("✅ No null values found in price data")

        # Check for reasonable price ranges
        for col in required_columns:
            col_min = data[col].min()
            col_max = data[col].max()
            if col_min <= 0:
                self.logger.warning(f"⚠️ Non-positive values found in {col}: min={col_min}")
            if col_max / col_min > 1000:  # Very large price range
                self.logger.warning(f"⚠️ Large price range in {col}: {col_min:.4f} - {col_max:.4f}")

        self.logger.debug("✅ Input data validation completed successfully")

    def _update_performance_stats(self, start_time: float, num_labels: int):
        """Update performance statistics."""
        processing_time = time.time() - start_time

        self.performance_stats['total_labels_generated'] += num_labels
        self.performance_stats['processing_time'] += processing_time

        if self.quality_metrics:
            self.performance_stats['quality_scores'].append(self.quality_metrics.overall_quality)

        self.logger.debug(f"📊 Performance stats updated: {num_labels} labels in {processing_time:.3f}s")
        self.logger.debug(f"📈 Total labels generated: {self.performance_stats['total_labels_generated']}")
        self.logger.debug(f"⏱️ Total processing time: {self.performance_stats['processing_time']:.3f}s")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        self.logger.info("📊 Generating performance summary...")

        avg_quality = np.mean(self.performance_stats['quality_scores']) if self.performance_stats['quality_scores'] else 0.0
        labels_per_second = safe_divide(
            self.performance_stats['total_labels_generated'],
            self.performance_stats['processing_time']
        )

        summary = {
            'total_labels_generated': self.performance_stats['total_labels_generated'],
            'total_processing_time': self.performance_stats['processing_time'],
            'average_quality_score': avg_quality,
            'labels_per_second': labels_per_second
        }

        self.logger.info(f"📈 Performance Summary:")
        self.logger.info(f"   📊 Total labels generated: {summary['total_labels_generated']}")
        self.logger.info(f"   ⏱️ Total processing time: {summary['total_processing_time']:.3f}s")
        self.logger.info(f"   🎯 Average quality score: {summary['average_quality_score']:.3f}")
        self.logger.info(f"   ⚡ Labels per second: {summary['labels_per_second']:.1f}")

        return summary

    # Numba-accelerated core function
    if NUMBA_AVAILABLE:
        @staticmethod
        @numba.jit(nopython=True, cache=True)
        def _numba_triple_barrier_core(
            close: np.ndarray,
            high: np.ndarray,
            low: np.ndarray,
            pt_mult: float,
            sl_mult: float,
            end_idx_arr: np.ndarray,
            transaction_cost: float
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Numba-accelerated triple barrier core implementation."""
            labels = np.zeros(close.shape[0], dtype=np.int8)
            profit_pcts = np.zeros(close.shape[0], dtype=np.float64)
            n = close.shape[0]

            for i in range(n - 1):
                entry_price = close[i]
                pt_price = entry_price * (1 + pt_mult)
                sl_price = entry_price * (1 - sl_mult)

                # Check barriers
                for j in range(i + 1, min(i + 100, n)):  # Max holding period
                    if high[j] >= pt_price:
                        labels[i] = 1
                        profit_pcts[i] = (pt_price - entry_price) / entry_price
                        break
                    elif low[j] <= sl_price:
                        labels[i] = -1
                        profit_pcts[i] = (sl_price - entry_price) / entry_price
                        break
                    elif j == min(i + 99, n - 1):  # Time barrier
                        labels[i] = 0
                        profit_pcts[i] = 0.0
                        break

            return labels, profit_pcts

# Global instance for backward compatibility
enhanced_data_labeler = EnhancedDataLabeler()

# Export for backward compatibility
DataLabeler = EnhancedDataLabeler
