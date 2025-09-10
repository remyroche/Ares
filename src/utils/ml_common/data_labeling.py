"""
Data Labeling Utilities

This module provides comprehensive data labeling utilities for trading data,
including triple barrier method implementations, regime-aware labeling,
and various labeling strategies optimized for financial time series.

Key Features:
- Triple barrier method with multiple variants
- Regime-aware labeling with dynamic parameters
- Fractional barrier labeling for continuous targets
- Profit-based feature engineering
- Label quality assessment and validation
- Performance tracking and analytics
- GPU acceleration support via M1 MPS
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

from ..math_validation import safe_divide, safe_log, safe_sqrt, validate_positive, validate_range
from ..common_operations import create_fallback_logger
from ..m1_gpu_utils import M1GPUManager
from ..parallel_processing_optimizer import ParallelProcessor

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

@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    profit_take_multiplier: float = 0.02
    stop_loss_multiplier: float = 0.01
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = 0.001
    min_holding_period: int = 1
    max_holding_period: int = 50
    regime_aware: bool = False
    regime_column: Optional[str] = None
    regime_parameters: Optional[Dict[str, Dict[str, float]]] = None

@dataclass
class LabelingResult:
    """Result of data labeling operation."""
    labels: np.ndarray
    profit_pcts: np.ndarray
    barrier_hit_types: np.ndarray  # 1=profit, -1=stop_loss, 0=time
    hit_indices: np.ndarray
    entry_prices: np.ndarray
    exit_prices: np.ndarray
    holding_periods: np.ndarray
    regime_ids: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataLabelingUtilities:
    """
    Comprehensive data labeling utilities for trading data.
    
    This class provides various labeling methods optimized for financial time series,
    with support for regime-aware labeling, GPU acceleration, and performance tracking.
    """
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """Initialize the data labeling utilities."""
        self.config = config or TripleBarrierConfig()
        self.logger = logger.getChild('DataLabelingUtilities')
        
        # Initialize components
        self.gpu_manager = M1GPUManager()
        self.parallel_processor = ParallelProcessor(max_workers=4)
        
        # Validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the labeling configuration."""
        validate_positive(self.config.profit_take_multiplier, "profit_take_multiplier")
        validate_positive(self.config.stop_loss_multiplier, "stop_loss_multiplier")
        validate_positive(self.config.time_barrier_minutes, "time_barrier_minutes")
        validate_positive(self.config.max_lookahead, "max_lookahead")
        validate_range(self.config.transaction_cost, 0.0, 1.0, "transaction_cost")
        
        if self.config.regime_aware and not self.config.regime_column:
            raise ValueError("regime_column must be specified when regime_aware=True")
    
    def label_data(
        self,
        data: pd.DataFrame,
        method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER,
        config: Optional[TripleBarrierConfig] = None
    ) -> LabelingResult:
        """
        Label data using the specified method.
        
        Args:
            data: Input data DataFrame with OHLCV columns
            method: Labeling method to use
            config: Optional configuration override
            
        Returns:
            LabelingResult with labels and metadata
        """
        config = config or self.config
        self.logger.info(f"Labeling data using method: {method.value}")
        
        try:
            if method == LabelingMethod.TRIPLE_BARRIER:
                return self._triple_barrier_labeling(data, config)
            elif method == LabelingMethod.REGIME_AWARE_TRIPLE_BARRIER:
                return self._regime_aware_triple_barrier_labeling(data, config)
            elif method == LabelingMethod.FRACTIONAL_TRIPLE_BARRIER:
                return self._fractional_triple_barrier_labeling(data, config)
            elif method == LabelingMethod.PROFIT_BASED:
                return self._profit_based_labeling(data, config)
            else:
                raise ValueError(f"Unsupported labeling method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error in data labeling: {e}")
            raise
    
    def _triple_barrier_labeling(self, data: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
        """Perform standard triple barrier labeling."""
        self.logger.info("Performing triple barrier labeling")
        
        # Extract OHLCV data
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Calculate end indices for each point
        end_indices = self._calculate_end_indices(data, config)
        
        # Perform labeling
        if NUMBA_AVAILABLE:
            labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods = self._numba_triple_barrier_labels(
                close, high, low, config.profit_take_multiplier, config.stop_loss_multiplier,
                end_indices, config.transaction_cost
            )
        else:
            labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods = self._python_triple_barrier_labels(
                close, high, low, config.profit_take_multiplier, config.stop_loss_multiplier,
                end_indices, config.transaction_cost
            )
        
        # Create metadata
        metadata = {
            'method': 'triple_barrier',
            'config': {
                'profit_take_multiplier': config.profit_take_multiplier,
                'stop_loss_multiplier': config.stop_loss_multiplier,
                'time_barrier_minutes': config.time_barrier_minutes,
                'transaction_cost': config.transaction_cost
            },
            'statistics': self._calculate_labeling_statistics(labels, profit_pcts, barrier_types)
        }
        
        return LabelingResult(
            labels=labels,
            profit_pcts=profit_pcts,
            barrier_hit_types=barrier_types,
            hit_indices=hit_indices,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            holding_periods=holding_periods,
            metadata=metadata
        )
    
    def _regime_aware_triple_barrier_labeling(self, data: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
        """Perform regime-aware triple barrier labeling."""
        self.logger.info("Performing regime-aware triple barrier labeling")
        
        if not config.regime_aware or not config.regime_column:
            self.logger.warning("Regime-aware labeling requested but no regime column specified, falling back to standard labeling")
            return self._triple_barrier_labeling(data, config)
        
        # Extract OHLCV data
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        regime_ids = data[config.regime_column].values
        
        # Get regime parameters
        regime_params = self._prepare_regime_parameters(config)
        
        # Calculate end indices for each point
        end_indices = self._calculate_end_indices(data, config)
        
        # Perform regime-aware labeling
        if NUMBA_AVAILABLE:
            labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods = self._numba_regime_aware_triple_barrier_labels(
                close, high, low, regime_ids, regime_params['pt_multipliers'], 
                regime_params['sl_multipliers'], end_indices
            )
        else:
            labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods = self._python_regime_aware_triple_barrier_labels(
                close, high, low, regime_ids, regime_params, end_indices
            )
        
        # Create metadata
        metadata = {
            'method': 'regime_aware_triple_barrier',
            'config': {
                'regime_column': config.regime_column,
                'regime_parameters': regime_params,
                'default_profit_take_multiplier': config.profit_take_multiplier,
                'default_stop_loss_multiplier': config.stop_loss_multiplier
            },
            'statistics': self._calculate_labeling_statistics(labels, profit_pcts, barrier_types),
            'regime_statistics': self._calculate_regime_statistics(labels, profit_pcts, regime_ids)
        }
        
        return LabelingResult(
            labels=labels,
            profit_pcts=profit_pcts,
            barrier_hit_types=barrier_types,
            hit_indices=hit_indices,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            holding_periods=holding_periods,
            regime_ids=regime_ids,
            metadata=metadata
        )
    
    def _fractional_triple_barrier_labeling(self, data: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
        """Perform fractional triple barrier labeling for continuous targets."""
        self.logger.info("Performing fractional triple barrier labeling")
        
        # Extract OHLCV data
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Calculate end indices for each point
        end_indices = self._calculate_end_indices(data, config)
        
        # Perform fractional labeling
        labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods = self._fractional_triple_barrier_labels(
            close, high, low, config.profit_take_multiplier, config.stop_loss_multiplier,
            end_indices, config.transaction_cost
        )
        
        # Create metadata
        metadata = {
            'method': 'fractional_triple_barrier',
            'config': {
                'profit_take_multiplier': config.profit_take_multiplier,
                'stop_loss_multiplier': config.stop_loss_multiplier,
                'time_barrier_minutes': config.time_barrier_minutes,
                'transaction_cost': config.transaction_cost
            },
            'statistics': self._calculate_labeling_statistics(labels, profit_pcts, barrier_types)
        }
        
        return LabelingResult(
            labels=labels,
            profit_pcts=profit_pcts,
            barrier_hit_types=barrier_types,
            hit_indices=hit_indices,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            holding_periods=holding_periods,
            metadata=metadata
        )
    
    def _profit_based_labeling(self, data: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
        """Perform profit-based labeling."""
        self.logger.info("Performing profit-based labeling")
        
        # Extract OHLCV data
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Calculate end indices for each point
        end_indices = self._calculate_end_indices(data, config)
        
        # Perform profit-based labeling
        labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods = self._profit_based_labels(
            close, high, low, config.profit_take_multiplier, config.stop_loss_multiplier,
            end_indices, config.transaction_cost
        )
        
        # Create metadata
        metadata = {
            'method': 'profit_based',
            'config': {
                'profit_take_multiplier': config.profit_take_multiplier,
                'stop_loss_multiplier': config.stop_loss_multiplier,
                'time_barrier_minutes': config.time_barrier_minutes,
                'transaction_cost': config.transaction_cost
            },
            'statistics': self._calculate_labeling_statistics(labels, profit_pcts, barrier_types)
        }
        
        return LabelingResult(
            labels=labels,
            profit_pcts=profit_pcts,
            barrier_hit_types=barrier_types,
            hit_indices=hit_indices,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            holding_periods=holding_periods,
            metadata=metadata
        )
    
    def _calculate_end_indices(self, data: pd.DataFrame, config: TripleBarrierConfig) -> np.ndarray:
        """Calculate end indices for each data point based on time barrier."""
        n = len(data)
        end_indices = np.zeros(n, dtype=np.int32)
        
        # Convert time barrier to number of periods
        if 'timestamp' in data.columns:
            # Time-based calculation
            timestamps = pd.to_datetime(data['timestamp'])
            time_delta = pd.Timedelta(minutes=config.time_barrier_minutes)
            
            for i in range(n):
                end_time = timestamps.iloc[i] + time_delta
                end_idx = timestamps.searchsorted(end_time, side='right')
                end_indices[i] = min(end_idx, i + config.max_lookahead)
        else:
            # Index-based calculation
            for i in range(n):
                end_indices[i] = min(i + config.time_barrier_minutes, i + config.max_lookahead)
        
        return end_indices
    
    def _prepare_regime_parameters(self, config: TripleBarrierConfig) -> Dict[str, Any]:
        """Prepare regime-specific parameters."""
        if not config.regime_parameters:
            # Use default parameters for all regimes
            return {
                'pt_multipliers': np.array([config.profit_take_multiplier]),
                'sl_multipliers': np.array([config.stop_loss_multiplier])
            }
        
        # Extract regime parameters
        regimes = list(config.regime_parameters.keys())
        pt_multipliers = []
        sl_multipliers = []
        
        for regime in regimes:
            regime_config = config.regime_parameters[regime]
            pt_multipliers.append(regime_config.get('profit_take_multiplier', config.profit_take_multiplier))
            sl_multipliers.append(regime_config.get('stop_loss_multiplier', config.stop_loss_multiplier))
        
        return {
            'pt_multipliers': np.array(pt_multipliers),
            'sl_multipliers': np.array(sl_multipliers),
            'regime_mapping': {regime: i for i, regime in enumerate(regimes)}
        }
    
    def _calculate_labeling_statistics(self, labels: np.ndarray, profit_pcts: np.ndarray, barrier_types: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for labeling results."""
        total_labels = len(labels)
        long_labels = np.sum(labels == 1)
        short_labels = np.sum(labels == -1)
        hold_labels = np.sum(labels == 0)
        
        profit_hits = np.sum(barrier_types == 1)
        stop_hits = np.sum(barrier_types == -1)
        time_hits = np.sum(barrier_types == 0)
        
        avg_profit = np.mean(profit_pcts[profit_pcts != 0]) if np.any(profit_pcts != 0) else 0
        avg_holding_period = np.mean(np.abs(profit_pcts[profit_pcts != 0])) if np.any(profit_pcts != 0) else 0
        
        return {
            'total_labels': total_labels,
            'long_labels': long_labels,
            'short_labels': short_labels,
            'hold_labels': hold_labels,
            'long_ratio': long_labels / total_labels if total_labels > 0 else 0,
            'short_ratio': short_labels / total_labels if total_labels > 0 else 0,
            'hold_ratio': hold_labels / total_labels if total_labels > 0 else 0,
            'profit_hits': profit_hits,
            'stop_hits': stop_hits,
            'time_hits': time_hits,
            'profit_hit_ratio': profit_hits / total_labels if total_labels > 0 else 0,
            'stop_hit_ratio': stop_hits / total_labels if total_labels > 0 else 0,
            'time_hit_ratio': time_hits / total_labels if total_labels > 0 else 0,
            'avg_profit': avg_profit,
            'avg_holding_period': avg_holding_period
        }
    
    def _calculate_regime_statistics(self, labels: np.ndarray, profit_pcts: np.ndarray, regime_ids: np.ndarray) -> Dict[str, Any]:
        """Calculate regime-specific statistics."""
        unique_regimes = np.unique(regime_ids)
        regime_stats = {}
        
        for regime in unique_regimes:
            regime_mask = regime_ids == regime
            regime_labels = labels[regime_mask]
            regime_profits = profit_pcts[regime_mask]
            
            regime_stats[str(regime)] = {
                'count': np.sum(regime_mask),
                'long_ratio': np.sum(regime_labels == 1) / len(regime_labels) if len(regime_labels) > 0 else 0,
                'short_ratio': np.sum(regime_labels == -1) / len(regime_labels) if len(regime_labels) > 0 else 0,
                'avg_profit': np.mean(regime_profits[regime_profits != 0]) if np.any(regime_profits != 0) else 0
            }
        
        return regime_stats
    
    # Numba-accelerated implementations
    if NUMBA_AVAILABLE:
        @staticmethod
        @numba.jit(nopython=True, cache=True)
        def _numba_triple_barrier_labels(
            close: np.ndarray, high: np.ndarray, low: np.ndarray,
            pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray,
            transaction_cost: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Numba-accelerated triple barrier labeling."""
            n = close.shape[0]
            labels = np.zeros(n, dtype=numba.int8)
            profit_pcts = np.zeros(n, dtype=numba.float64)
            barrier_types = np.zeros(n, dtype=numba.int8)
            hit_indices = np.zeros(n, dtype=numba.int32)
            entry_prices = np.zeros(n, dtype=numba.float64)
            exit_prices = np.zeros(n, dtype=numba.float64)
            holding_periods = np.zeros(n, dtype=numba.int32)
            
            for i in range(n - 1):
                entry_price = close[i]
                entry_prices[i] = entry_price
                profit_barrier = entry_price * (1.0 + pt_mult)
                stop_barrier = entry_price * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i])
                
                if end_idx <= i + 1:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    barrier_types[i] = 0
                    hit_indices[i] = i
                    exit_prices[i] = entry_price
                    holding_periods[i] = 0
                    continue
                
                lab = 0
                profit_pct = 0.0
                barrier_type = 0
                hit_idx = i
                exit_price = entry_price
                holding_period = 0
                
                for j in range(i + 1, end_idx):
                    holding_period += 1
                    if high[j] >= profit_barrier:
                        lab = 1
                        profit_pct = pt_mult - transaction_cost
                        barrier_type = 1
                        hit_idx = j
                        exit_price = profit_barrier
                        break
                    if low[j] <= stop_barrier:
                        lab = -1
                        profit_pct = -sl_mult - transaction_cost
                        barrier_type = -1
                        hit_idx = j
                        exit_price = stop_barrier
                        break
                
                labels[i] = lab
                profit_pcts[i] = profit_pct
                barrier_types[i] = barrier_type
                hit_indices[i] = hit_idx
                exit_prices[i] = exit_price
                holding_periods[i] = holding_period
            
            return labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods
        
        @staticmethod
        @numba.jit(nopython=True, cache=True)
        def _numba_regime_aware_triple_barrier_labels(
            close: np.ndarray, high: np.ndarray, low: np.ndarray,
            regime_ids: np.ndarray, pt_multipliers: np.ndarray, sl_multipliers: np.ndarray,
            end_idx_arr: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Numba-accelerated regime-aware triple barrier labeling."""
            n = close.shape[0]
            labels = np.zeros(n, dtype=numba.int8)
            profit_pcts = np.zeros(n, dtype=numba.float64)
            barrier_types = np.zeros(n, dtype=numba.int8)
            hit_indices = np.zeros(n, dtype=numba.int32)
            entry_prices = np.zeros(n, dtype=numba.float64)
            exit_prices = np.zeros(n, dtype=numba.float64)
            holding_periods = np.zeros(n, dtype=numba.int32)
            
            for i in range(n - 1):
                entry_price = close[i]
                entry_prices[i] = entry_price
                regime_id = int(regime_ids[i])
                
                # Get regime-specific parameters
                pt_mult = pt_multipliers[regime_id] if regime_id < len(pt_multipliers) else pt_multipliers[0]
                sl_mult = sl_multipliers[regime_id] if regime_id < len(sl_multipliers) else sl_multipliers[0]
                
                profit_barrier = entry_price * (1.0 + pt_mult)
                stop_barrier = entry_price * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i])
                
                if end_idx <= i + 1:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    barrier_types[i] = 0
                    hit_indices[i] = i
                    exit_prices[i] = entry_price
                    holding_periods[i] = 0
                    continue
                
                lab = 0
                profit_pct = 0.0
                barrier_type = 0
                hit_idx = i
                exit_price = entry_price
                holding_period = 0
                
                for j in range(i + 1, end_idx):
                    holding_period += 1
                    if high[j] >= profit_barrier:
                        lab = 1
                        profit_pct = pt_mult
                        barrier_type = 1
                        hit_idx = j
                        exit_price = profit_barrier
                        break
                    if low[j] <= stop_barrier:
                        lab = -1
                        profit_pct = -sl_mult
                        barrier_type = -1
                        hit_idx = j
                        exit_price = stop_barrier
                        break
                
                labels[i] = lab
                profit_pcts[i] = profit_pct
                barrier_types[i] = barrier_type
                hit_indices[i] = hit_idx
                exit_prices[i] = exit_price
                holding_periods[i] = holding_period
            
            return labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods
    
    # Python fallback implementations
    def _python_triple_barrier_labels(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Python implementation of triple barrier labeling."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        barrier_types = np.zeros(n, dtype=np.int8)
        hit_indices = np.zeros(n, dtype=np.int32)
        entry_prices = np.zeros(n, dtype=np.float64)
        exit_prices = np.zeros(n, dtype=np.float64)
        holding_periods = np.zeros(n, dtype=np.int32)
        
        for i in range(n - 1):
            entry_price = close[i]
            entry_prices[i] = entry_price
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                barrier_types[i] = 0
                hit_indices[i] = i
                exit_prices[i] = entry_price
                holding_periods[i] = 0
                continue
            
            lab = 0
            profit_pct = 0.0
            barrier_type = 0
            hit_idx = i
            exit_price = entry_price
            holding_period = 0
            
            for j in range(i + 1, end_idx):
                holding_period += 1
                if high[j] >= profit_barrier:
                    lab = 1
                    profit_pct = pt_mult - transaction_cost
                    barrier_type = 1
                    hit_idx = j
                    exit_price = profit_barrier
                    break
                if low[j] <= stop_barrier:
                    lab = -1
                    profit_pct = -sl_mult - transaction_cost
                    barrier_type = -1
                    hit_idx = j
                    exit_price = stop_barrier
                    break
            
            labels[i] = lab
            profit_pcts[i] = profit_pct
            barrier_types[i] = barrier_type
            hit_indices[i] = hit_idx
            exit_prices[i] = exit_price
            holding_periods[i] = holding_period
        
        return labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods
    
    def _python_regime_aware_triple_barrier_labels(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        regime_ids: np.ndarray, regime_params: Dict[str, Any], end_idx_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Python implementation of regime-aware triple barrier labeling."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        barrier_types = np.zeros(n, dtype=np.int8)
        hit_indices = np.zeros(n, dtype=np.int32)
        entry_prices = np.zeros(n, dtype=np.float64)
        exit_prices = np.zeros(n, dtype=np.float64)
        holding_periods = np.zeros(n, dtype=np.int32)
        
        pt_multipliers = regime_params['pt_multipliers']
        sl_multipliers = regime_params['sl_multipliers']
        
        for i in range(n - 1):
            entry_price = close[i]
            entry_prices[i] = entry_price
            regime_id = int(regime_ids[i])
            
            # Get regime-specific parameters
            pt_mult = pt_multipliers[regime_id] if regime_id < len(pt_multipliers) else pt_multipliers[0]
            sl_mult = sl_multipliers[regime_id] if regime_id < len(sl_multipliers) else sl_multipliers[0]
            
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                barrier_types[i] = 0
                hit_indices[i] = i
                exit_prices[i] = entry_price
                holding_periods[i] = 0
                continue
            
            lab = 0
            profit_pct = 0.0
            barrier_type = 0
            hit_idx = i
            exit_price = entry_price
            holding_period = 0
            
            for j in range(i + 1, end_idx):
                holding_period += 1
                if high[j] >= profit_barrier:
                    lab = 1
                    profit_pct = pt_mult
                    barrier_type = 1
                    hit_idx = j
                    exit_price = profit_barrier
                    break
                if low[j] <= stop_barrier:
                    lab = -1
                    profit_pct = -sl_mult
                    barrier_type = -1
                    hit_idx = j
                    exit_price = stop_barrier
                    break
            
            labels[i] = lab
            profit_pcts[i] = profit_pct
            barrier_types[i] = barrier_type
            hit_indices[i] = hit_idx
            exit_prices[i] = exit_price
            holding_periods[i] = holding_period
        
        return labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods
    
    def _fractional_triple_barrier_labels(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fractional triple barrier labeling for continuous targets."""
        n = len(close)
        labels = np.zeros(n, dtype=np.float64)
        profit_pcts = np.zeros(n, dtype=np.float64)
        barrier_types = np.zeros(n, dtype=np.int8)
        hit_indices = np.zeros(n, dtype=np.int32)
        entry_prices = np.zeros(n, dtype=np.float64)
        exit_prices = np.zeros(n, dtype=np.float64)
        holding_periods = np.zeros(n, dtype=np.int32)
        
        for i in range(n - 1):
            entry_price = close[i]
            entry_prices[i] = entry_price
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0.0
                profit_pcts[i] = 0.0
                barrier_types[i] = 0
                hit_indices[i] = i
                exit_prices[i] = entry_price
                holding_periods[i] = 0
                continue
            
            max_profit = 0.0
            max_loss = 0.0
            hit_idx = i
            exit_price = entry_price
            holding_period = 0
            
            for j in range(i + 1, end_idx):
                holding_period += 1
                
                # Calculate current profit/loss
                current_profit = (high[j] - entry_price) / entry_price
                current_loss = (entry_price - low[j]) / entry_price
                
                max_profit = max(max_profit, current_profit)
                max_loss = max(max_loss, current_loss)
                
                # Check barriers
                if high[j] >= profit_barrier:
                    hit_idx = j
                    exit_price = profit_barrier
                    break
                if low[j] <= stop_barrier:
                    hit_idx = j
                    exit_price = stop_barrier
                    break
            
            # Calculate fractional label
            if max_profit >= pt_mult:
                labels[i] = 1.0
                profit_pcts[i] = pt_mult - transaction_cost
                barrier_types[i] = 1
            elif max_loss >= sl_mult:
                labels[i] = -1.0
                profit_pcts[i] = -sl_mult - transaction_cost
                barrier_types[i] = -1
            else:
                # Fractional label based on max profit/loss
                if max_profit > max_loss:
                    labels[i] = max_profit / pt_mult
                    profit_pcts[i] = max_profit - transaction_cost
                else:
                    labels[i] = -max_loss / sl_mult
                    profit_pcts[i] = -max_loss - transaction_cost
                barrier_types[i] = 0
            
            hit_indices[i] = hit_idx
            exit_prices[i] = exit_price
            holding_periods[i] = holding_period
        
        return labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods
    
    def _profit_based_labels(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Profit-based labeling focusing on actual profit/loss."""
        n = len(close)
        labels = np.zeros(n, dtype=np.float64)
        profit_pcts = np.zeros(n, dtype=np.float64)
        barrier_types = np.zeros(n, dtype=np.int8)
        hit_indices = np.zeros(n, dtype=np.int32)
        entry_prices = np.zeros(n, dtype=np.float64)
        exit_prices = np.zeros(n, dtype=np.float64)
        holding_periods = np.zeros(n, dtype=np.int32)
        
        for i in range(n - 1):
            entry_price = close[i]
            entry_prices[i] = entry_price
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0.0
                profit_pcts[i] = 0.0
                barrier_types[i] = 0
                hit_indices[i] = i
                exit_prices[i] = entry_price
                holding_periods[i] = 0
                continue
            
            best_profit = 0.0
            best_loss = 0.0
            hit_idx = i
            exit_price = entry_price
            holding_period = 0
            
            for j in range(i + 1, end_idx):
                holding_period += 1
                
                # Calculate potential profit/loss
                profit = (high[j] - entry_price) / entry_price
                loss = (entry_price - low[j]) / entry_price
                
                if profit > best_profit:
                    best_profit = profit
                if loss > best_loss:
                    best_loss = loss
            
            # Determine label based on best profit vs loss
            if best_profit > best_loss and best_profit > pt_mult:
                labels[i] = 1.0
                profit_pcts[i] = best_profit - transaction_cost
                barrier_types[i] = 1
            elif best_loss > best_profit and best_loss > sl_mult:
                labels[i] = -1.0
                profit_pcts[i] = -best_loss - transaction_cost
                barrier_types[i] = -1
            else:
                labels[i] = 0.0
                profit_pcts[i] = 0.0
                barrier_types[i] = 0
            
            hit_indices[i] = hit_idx
            exit_prices[i] = exit_price
            holding_periods[i] = holding_period
        
        return labels, profit_pcts, barrier_types, hit_indices, entry_prices, exit_prices, holding_periods

# Convenience functions
def get_data_labeler(config: Optional[TripleBarrierConfig] = None) -> DataLabelingUtilities:
    """Get a configured data labeling utility."""
    return DataLabelingUtilities(config)

def label_triple_barrier(
    data: pd.DataFrame,
    config: Optional[TripleBarrierConfig] = None,
    method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER
) -> LabelingResult:
    """Convenience function for triple barrier labeling."""
    labeler = get_data_labeler(config)
    return labeler.label_data(data, method)

def label_regime_aware(
    data: pd.DataFrame,
    regime_column: str,
    config: Optional[TripleBarrierConfig] = None
) -> LabelingResult:
    """Convenience function for regime-aware labeling."""
    if config is None:
        config = TripleBarrierConfig()
    config.regime_aware = True
    config.regime_column = regime_column
    
    labeler = get_data_labeler(config)
    return labeler.label_data(data, LabelingMethod.REGIME_AWARE_TRIPLE_BARRIER, config)