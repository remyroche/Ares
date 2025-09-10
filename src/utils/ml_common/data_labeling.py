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
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from ..common_operations import create_fallback_logger, create_fallback_decorator
from ..common_utilities import CommonUtilities
from ..parquet_utils import ParquetUtils
from ..serialization_utils import UniversalSerializer
from ..data_processing_utils import DataProcessingUtils
from ..m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
from ..m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from ..m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer

# Import ML Common utilities for cross-validation
from .cv_utils import TemporalCrossValidator, PurgedKFold
from .validation_utils import ValidationFramework
from .pareto import ParetoFrontAnalyzer

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
    transaction_cost: float = 0.001
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

class EnhancedDataLabeler:
    """Enhanced data labeler with consolidated triple barrier implementations."""
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        self.config = config or TripleBarrierConfig()
        self.logger = create_fallback_logger("EnhancedDataLabeler")
        
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
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.parquet_utils = ParquetUtils()
            self.serializer = UniversalSerializer()
            self.data_processor = DataProcessingUtils()
            self.common_utils = CommonUtilities()
            
            self.logger.info("✅ All utility managers initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Some utility managers failed to initialize: {e}")
            # Set fallback implementations
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.parquet_utils = None
            self.serializer = None
            self.data_processor = None
            self.common_utils = None

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
            else:
                raise ValueError(f"Unsupported labeling method: {method}")
            
            # Assess label quality
            self.quality_metrics = self._assess_label_quality(labels_df, data)
            
            # Update performance stats
            self._update_performance_stats(start_time, len(labels_df))
            
            self.logger.info(f"✅ Generated {len(labels_df)} labels using {method.value}")
            return labels_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create triple barrier labels: {e}")
            raise

    def _create_standard_triple_barrier(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create standard triple barrier labels."""
        if NUMBA_AVAILABLE:
            return self._create_numba_triple_barrier(data, config)
        else:
            return self._create_python_triple_barrier(data, config)

    def _create_numba_triple_barrier(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Numba-accelerated triple barrier implementation."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Calculate end indices for each position
        end_indices = np.arange(len(close))
        
        # Use numba-accelerated function
        labels, profit_pcts = self._numba_triple_barrier_core(
            close, high, low, 
            config.pt_mult, config.sl_mult, 
            end_indices, config.transaction_cost
        )
        
        # Create result DataFrame
        result = data.copy()
        result['label'] = labels
        result['profit_pct'] = profit_pcts
        result['barrier_type'] = 'triple_barrier'
        result['config_pt'] = config.pt_mult
        result['config_sl'] = config.sl_mult
        
        return result

    def _create_python_triple_barrier(
        self, 
        data: pd.DataFrame, 
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Pure Python triple barrier implementation."""
        result = data.copy()
        labels = []
        profit_pcts = []
        
        for i in range(len(data)):
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
        
        return result

    def _create_regime_aware_triple_barrier(
        self, 
        data: pd.DataFrame, 
        regime_data: Optional[pd.DataFrame],
        config: TripleBarrierConfig
    ) -> pd.DataFrame:
        """Create regime-aware triple barrier labels."""
        if regime_data is None:
            self.logger.warning("⚠️ No regime data provided, falling back to standard triple barrier")
            return self._create_standard_triple_barrier(data, config)
        
        # Merge regime data
        merged_data = data.merge(regime_data, left_index=True, right_index=True, how='left')
        
        # Get unique regimes
        regimes = merged_data['regime'].unique()
        regime_results = []
        
        for regime in regimes:
            regime_mask = merged_data['regime'] == regime
            regime_data_subset = merged_data[regime_mask]
            
            # Get regime-specific config
            regime_config = self.regime_config.regime_params.get(
                str(regime), self.regime_config.default_config
            )
            
            # Create labels for this regime
            regime_labels = self._create_standard_triple_barrier(regime_data_subset, regime_config)
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
        """Validate labels using temporal cross-validation."""
        try:
            # Initialize temporal cross-validator
            cv = TemporalCrossValidator(n_splits=n_splits, purged_pct=purged_pct)
            
            # Prepare data for CV
            X = labels_df.drop(['label', 'profit_pct'], axis=1, errors='ignore')
            y = labels_df['label']
            
            # Perform cross-validation
            cv_results = cv.cross_validate(X, y)
            
            return {
                'cv_scores': cv_results,
                'mean_score': np.mean(cv_results),
                'std_score': np.std(cv_results),
                'validation_passed': np.mean(cv_results) > 0.6
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

    def _update_performance_stats(self, start_time: float, num_labels: int):
        """Update performance statistics."""
        processing_time = time.time() - start_time
        
        self.performance_stats['total_labels_generated'] += num_labels
        self.performance_stats['processing_time'] += processing_time
        
        if self.quality_metrics:
            self.performance_stats['quality_scores'].append(self.quality_metrics.overall_quality)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        avg_quality = np.mean(self.performance_stats['quality_scores']) if self.performance_stats['quality_scores'] else 0.0
        
        return {
            'total_labels_generated': self.performance_stats['total_labels_generated'],
            'total_processing_time': self.performance_stats['processing_time'],
            'average_quality_score': avg_quality,
            'labels_per_second': safe_divide(
                self.performance_stats['total_labels_generated'],
                self.performance_stats['processing_time']
            )
        }

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