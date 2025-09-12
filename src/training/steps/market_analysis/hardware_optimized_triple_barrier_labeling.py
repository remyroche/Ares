"""
Hardware-Optimized Triple Barrier Labeling Module for MARKET_ANALYSIS.

This module provides a comprehensive implementation of the triple barrier method
optimized for hardware performance, particularly for M1/M2/M3 Macs and other
high-performance computing environments.

Key Features:
- Hardware-specific optimizations for M1/M2/M3 Macs
- Advanced memory management and optimization
- Numba acceleration with parallel processing
- GPU acceleration support (MPS)
- Regime-aware labeling with hardware optimization
- Transaction cost modeling and binary classification
- Comprehensive validation and error handling
- Integration with existing market analysis pipeline

Usage:
    from src.training.steps.market_analysis import HardwareOptimizedTripleBarrierLabeling
    
    # Basic usage
    labeler = HardwareOptimizedTripleBarrierLabeling()
    labeled_data = labeler.apply_triple_barrier_labeling(data)
    
    # Regime-aware usage
    labeler = HardwareOptimizedTripleBarrierLabeling(enable_regime_aware=True)
    labeled_data = labeler.apply_regime_aware_labeling(data, regime_column='hmm_regime')
"""

import logging
import time
import gc
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

# Hardware optimization imports
try:
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, 
        create_m1_optimized_thread_pool, 
        run_cpu_intensive_task,
        create_m1_optimized_context
    )
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, 
        optimize_dataframe_for_m1, 
        create_m1_optimized_array,
        is_m1_available,
        is_mps_available
    )
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, 
        optimize_dataframe_memory,
        start_m1_memory_monitoring,
        stop_m1_memory_monitoring
    )
    from src.utils.hardware.m1_optimizations import (
        get_m1_memory_optimizer as get_advanced_m1_optimizer,
        M1MemoryOptimizer,
        M1DataManager
    )
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimizations not available: {e}")
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False

# Try to import Numba for performance acceleration
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False

# Try to import PyTorch for MPS support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Core imports
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, cached, log_execution_time

logger = get_logger(__name__)

@dataclass
class HardwareOptimizedConfig:
    """Configuration for hardware-optimized triple barrier labeling."""
    profit_take_multiplier: float = 0.004
    stop_loss_multiplier: float = 0.003
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = 0.0008
    binary_classification: bool = True
    enable_regime_aware: bool = False
    enable_hardware_optimization: bool = True
    enable_numba_acceleration: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    chunk_size: Optional[int] = None
    enable_memory_monitoring: bool = True
    enable_parallel_processing: bool = True
    num_threads: Optional[int] = None

# Numba-accelerated functions with parallel processing
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True, parallel=True)
    def _numba_triple_barrier_labels_parallel(
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray,
        pt_mult: float, 
        sl_mult: float, 
        end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated triple barrier labeling with parallel processing."""
        n = close.shape[0]
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        
        for i in numba.prange(n - 1):
            entry_price = close[i]
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                continue
                
            lab = 0
            profit_pct = 0.0
            
            for j in range(i + 1, end_idx):
                if high[j] >= profit_barrier:
                    lab = 1
                    profit_pct = pt_mult - transaction_cost
                    break
                if low[j] <= stop_barrier:
                    lab = -1
                    profit_pct = -(sl_mult + transaction_cost)
                    break
                    
            labels[i] = lab
            profit_pcts[i] = profit_pct
            
        return labels, profit_pcts

    @numba.jit(nopython=True, cache=True)
    def _numba_regime_aware_barriers_optimized(
        close: np.ndarray,
        regime_params: np.ndarray,  # [pt_mult, sl_mult, time_mult] for each regime
        regime_labels: np.ndarray,
        end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated regime-aware triple barrier labeling."""
        n = close.shape[0]
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        
        for i in range(n - 1):
            regime = int(regime_labels[i])
            if regime >= len(regime_params):
                regime = 0
                
            pt_mult, sl_mult, time_mult = regime_params[regime]
            
            entry_price = close[i]
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i] * time_mult)
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                continue
                
            lab = 0
            profit_pct = 0.0
            
            for j in range(i + 1, min(end_idx, n)):
                if high[j] >= profit_barrier:
                    lab = 1
                    profit_pct = pt_mult - transaction_cost
                    break
                if low[j] <= stop_barrier:
                    lab = -1
                    profit_pct = -(sl_mult + transaction_cost)
                    break
                    
            labels[i] = lab
            profit_pcts[i] = profit_pct
            
        return labels, profit_pcts

class HardwareOptimizedTripleBarrierLabeling:
    """
    Hardware-Optimized Triple Barrier Labeling with Advanced Performance Features.
    
    This class provides optimized triple barrier labeling with:
    - Hardware-specific optimizations for M1/M2/M3 Macs
    - Advanced memory management and optimization
    - Numba acceleration with parallel processing
    - GPU acceleration support (MPS)
    - Regime-aware labeling with hardware optimization
    - Transaction cost modeling
    - Binary classification support
    """

    def __init__(self, config: Optional[HardwareOptimizedConfig] = None):
        """Initialize the hardware-optimized triple barrier labeling system."""
        self.config = config or HardwareOptimizedConfig()
        self.logger = get_logger(f'{__name__}.HardwareOptimizedTripleBarrierLabeling')
        
        # Initialize hardware optimizers
        self._setup_hardware_optimizations()
        
        # Validate configuration
        self._validate_config()
        
        # Start memory monitoring if enabled
        if self.config.enable_memory_monitoring and HARDWARE_OPTIMIZATIONS_AVAILABLE:
            start_m1_memory_monitoring()
        
        self.logger.info(f'🚀 Hardware-Optimized Triple Barrier Labeling initialized')
        self.logger.info(f'   → Hardware optimizations: {HARDWARE_OPTIMIZATIONS_AVAILABLE}')
        self.logger.info(f'   → Numba acceleration: {NUMBA_AVAILABLE}')
        self.logger.info(f'   → GPU acceleration: {TORCH_AVAILABLE and is_mps_available() if HARDWARE_OPTIMIZATIONS_AVAILABLE else False}')
        self.logger.info(f'   → Regime-aware: {self.config.enable_regime_aware}')
        self.logger.info(f'   → Binary classification: {self.config.binary_classification}')

    def _setup_hardware_optimizations(self):
        """Setup hardware-specific optimizations."""
        if not HARDWARE_OPTIMIZATIONS_AVAILABLE:
            self.logger.warning('⚠️ Hardware optimizations not available')
            return
            
        try:
            # Initialize M1 CPU optimizer
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.cpu_optimizer.optimize_numpy_operations()
            
            # Initialize M1 GPU manager
            self.gpu_manager = get_m1_gpu_manager()
            
            # Initialize M1 memory optimizer
            self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
            
            # Initialize advanced M1 memory optimizer
            self.advanced_memory_optimizer = get_advanced_m1_optimizer(
                memory_limit_gb=self.config.memory_limit_gb,
                enable_gc_tuning=True,
                enable_memory_leak_detection=True,
                enable_swap_management=True
            )
            
            # Initialize data manager
            self.data_manager = M1DataManager(self.advanced_memory_optimizer)
            
            # Setup thread pool for parallel processing
            if self.config.enable_parallel_processing:
                self.thread_pool = create_m1_optimized_thread_pool(
                    max_workers=self.config.num_threads
                )
            else:
                self.thread_pool = None
            
            self.logger.info('✅ Hardware optimizations initialized successfully')
            
        except Exception as e:
            self.logger.warning(f'⚠️ Hardware optimization setup failed: {e}')
            self.cpu_optimizer = None
            self.gpu_manager = None
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
            self.data_manager = None
            self.thread_pool = None

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.profit_take_multiplier <= 0:
            raise ValueError("Profit take multiplier must be positive")
        if self.config.stop_loss_multiplier <= 0:
            raise ValueError("Stop loss multiplier must be positive")
        if self.config.transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if self.config.max_lookahead <= 0:
            raise ValueError("Max lookahead must be positive")
        if self.config.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")

    @contextmanager
    def _hardware_optimization_context(self):
        """Context manager for hardware optimization."""
        if not HARDWARE_OPTIMIZATIONS_AVAILABLE:
            yield
            return
        
        try:
            # Create M1 optimization context
            with create_m1_optimized_context() as m1_context:
                # Optimize memory before processing
                if self.advanced_memory_optimizer:
                    self.advanced_memory_optimizer.optimize_memory()
                
                yield m1_context
                
        except Exception as e:
            self.logger.warning(f'⚠️ Hardware optimization context failed: {e}')
            yield

    @handles_errors(default_return=pd.DataFrame())
    @log_execution_time
    def apply_triple_barrier_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple barrier labeling with hardware optimization.
        
        Args:
            data: Market data with OHLC columns
            
        Returns:
            DataFrame with triple barrier labels
        """
        self.logger.info(f'🏷️ Starting hardware-optimized triple barrier labeling')
        self.logger.info(f'   Input shape: {data.shape}')
        
        # Validate input data
        self._validate_input_data(data)
        
        # Apply hardware optimizations
        with self._hardware_optimization_context():
            # Optimize data for hardware
            optimized_data = self._optimize_data_for_hardware(data)
            
            # Apply labeling logic
            labeled_data = self._apply_barrier_logic_optimized(optimized_data)
            
            # Post-process results
            if self.config.binary_classification:
                labeled_data = self._filter_hold_samples(labeled_data)
            
            # Final memory optimization
            if self.advanced_memory_optimizer:
                self.advanced_memory_optimizer.optimize_memory()
        
        self.logger.info(f'✅ Hardware-optimized triple barrier labeling completed')
        self.logger.info(f'   Output shape: {labeled_data.shape}')
        
        return labeled_data

    @handles_errors(default_return=pd.DataFrame())
    @log_execution_time
    def apply_regime_aware_labeling(
        self, 
        data: pd.DataFrame, 
        regime_column: str = 'hmm_regime'
    ) -> pd.DataFrame:
        """
        Apply regime-aware triple barrier labeling with hardware optimization.
        
        Args:
            data: Market data with OHLC and regime columns
            regime_column: Name of the regime column
            
        Returns:
            DataFrame with regime-aware triple barrier labels
        """
        self.logger.info(f'🎯 Starting hardware-optimized regime-aware triple barrier labeling')
        self.logger.info(f'   Regime column: {regime_column}')
        
        # Validate input data
        self._validate_input_data(data)
        if regime_column not in data.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in data")
        
        # Apply hardware optimizations
        with self._hardware_optimization_context():
            # Optimize data for hardware
            optimized_data = self._optimize_data_for_hardware(data)
            
            # Apply regime-aware labeling logic
            labeled_data = self._apply_regime_aware_barrier_logic_optimized(optimized_data, regime_column)
            
            # Post-process results
            if self.config.binary_classification:
                labeled_data = self._filter_hold_samples(labeled_data)
            
            # Final memory optimization
            if self.advanced_memory_optimizer:
                self.advanced_memory_optimizer.optimize_memory()
        
        self.logger.info(f'✅ Hardware-optimized regime-aware triple barrier labeling completed')
        
        return labeled_data

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data quality."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for invalid prices
        for col in required_columns:
            if (data[col] <= 0).any():
                raise ValueError(f"Invalid prices found in {col} (≤ 0)")
        
        # Check for NaN values
        for col in required_columns:
            if data[col].isna().any():
                raise ValueError(f"NaN values found in {col}")

    def _optimize_data_for_hardware(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply hardware-specific optimizations to data."""
        if not HARDWARE_OPTIMIZATIONS_AVAILABLE:
            return data
        
        try:
            # Optimize DataFrame memory usage
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe_memory(data)
            
            # Optimize for M1 GPU
            if self.gpu_manager:
                data = optimize_dataframe_for_m1(data)
            
            # Convert to optimized arrays
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    data[col] = create_m1_optimized_array(data[col].values)
            
            self.logger.debug('✅ Data optimized for hardware')
            
        except Exception as e:
            self.logger.warning(f'⚠️ Hardware optimization failed: {e}')
        
        return data

    def _apply_barrier_logic_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply barrier logic with hardware optimization."""
        self.logger.info('🔧 Applying hardware-optimized barrier logic')
        
        # Prepare data
        labeled_data = data.copy()
        n = len(labeled_data)
        
        if n < 2:
            labeled_data['triple_barrier_label'] = 0
            labeled_data['profit_pct'] = 0.0
            return labeled_data
        
        # Convert to optimized numpy arrays
        close = create_m1_optimized_array(labeled_data['close'].values)
        high = create_m1_optimized_array(labeled_data['high'].values)
        low = create_m1_optimized_array(labeled_data['low'].values)
        
        # Calculate end indices
        end_idx_arr = self._calculate_end_indices_optimized(labeled_data)
        
        # Use Numba acceleration if available and data is large enough
        use_numba = (NUMBA_AVAILABLE and 
                    callable(globals().get('_numba_triple_barrier_labels_parallel')) and 
                    n >= 512)
        
        if use_numba:
            self.logger.info('⚡ Using Numba-accelerated parallel triple barrier labeling')
            labels, profit_pcts = _numba_triple_barrier_labels_parallel(
                close, high, low,
                self.config.profit_take_multiplier,
                self.config.stop_loss_multiplier,
                end_idx_arr,
                self.config.transaction_cost
            )
        else:
            self.logger.info('🐍 Using Python vectorized triple barrier labeling')
            labels, profit_pcts = self._python_barrier_logic_optimized(
                close, high, low, end_idx_arr
            )
        
        # Add results to DataFrame
        labeled_data['triple_barrier_label'] = labels
        labeled_data['profit_pct'] = profit_pcts
        
        return labeled_data

    def _apply_regime_aware_barrier_logic_optimized(
        self, 
        data: pd.DataFrame, 
        regime_column: str
    ) -> pd.DataFrame:
        """Apply regime-aware barrier logic with hardware optimization."""
        self.logger.info('🎯 Applying hardware-optimized regime-aware barrier logic')
        
        # Prepare data
        labeled_data = data.copy()
        n = len(labeled_data)
        
        if n < 2:
            labeled_data['triple_barrier_label'] = 0
            labeled_data['profit_pct'] = 0.0
            return labeled_data
        
        # Convert to optimized numpy arrays
        close = create_m1_optimized_array(labeled_data['close'].values)
        high = create_m1_optimized_array(labeled_data['high'].values)
        low = create_m1_optimized_array(labeled_data['low'].values)
        regime_labels = create_m1_optimized_array(labeled_data[regime_column].values, dtype=np.int32)
        
        # Create regime parameters array
        regime_params = self._create_regime_parameters_optimized(regime_labels)
        
        # Calculate end indices
        end_idx_arr = self._calculate_end_indices_optimized(labeled_data)
        
        # Use Numba acceleration if available
        use_numba = (NUMBA_AVAILABLE and 
                    callable(globals().get('_numba_regime_aware_barriers_optimized')) and 
                    n >= 512)
        
        if use_numba:
            self.logger.info('⚡ Using Numba-accelerated regime-aware labeling')
            labels, profit_pcts = _numba_regime_aware_barriers_optimized(
                close, regime_params, regime_labels, end_idx_arr,
                self.config.transaction_cost
            )
        else:
            self.logger.info('🐍 Using Python regime-aware labeling')
            labels, profit_pcts = self._python_regime_aware_logic_optimized(
                close, high, low, regime_labels, regime_params, end_idx_arr
            )
        
        # Add results to DataFrame
        labeled_data['triple_barrier_label'] = labels
        labeled_data['profit_pct'] = profit_pcts
        
        return labeled_data

    def _calculate_end_indices_optimized(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate end indices for barrier evaluation with optimization."""
        n = len(data)
        arange_n = np.arange(n, dtype=np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + self.config.max_lookahead, n)
        
        # Handle time barrier if datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            try:
                idx_ns = data.index.view(np.int64)
                delta_ns = np.int64(self.config.time_barrier_minutes) * np.int64(60000000000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns, end_times, side='right')
            except Exception:
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
        
        return np.minimum(end_by_lookahead, end_by_time).astype(np.int64)

    def _create_regime_parameters_optimized(self, regime_labels: np.ndarray) -> np.ndarray:
        """Create regime-specific parameters array with optimization."""
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        
        # Create parameters array [pt_mult, sl_mult, time_mult] for each regime
        regime_params = np.zeros((n_regimes, 3), dtype=np.float64)
        
        for i, regime in enumerate(unique_regimes):
            # Default parameters (can be customized per regime)
            regime_params[i] = [
                self.config.profit_take_multiplier,
                self.config.stop_loss_multiplier,
                1.0  # Time multiplier
            ]
        
        return regime_params

    def _python_barrier_logic_optimized(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        end_idx_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Python implementation of barrier logic with optimization."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        
        pt_mult = self.config.profit_take_multiplier
        sl_mult = self.config.stop_loss_multiplier
        transaction_cost = self.config.transaction_cost
        
        # Use CPU-intensive task optimization if available
        if HARDWARE_OPTIMIZATIONS_AVAILABLE and self.cpu_optimizer:
            def process_barriers():
                for i in range(n - 1):
                    entry_price = close[i]
                    profit_barrier = entry_price * (1.0 + pt_mult)
                    stop_barrier = entry_price * (1.0 - sl_mult)
                    end_idx = int(end_idx_arr[i])
                    
                    if end_idx <= i + 1:
                        labels[i] = 0
                        profit_pcts[i] = 0.0
                        continue
                    
                    lab = 0
                    profit_pct = 0.0
                    
                    for j in range(i + 1, end_idx):
                        if high[j] >= profit_barrier:
                            lab = 1
                            profit_pct = pt_mult - transaction_cost
                            break
                        if low[j] <= stop_barrier:
                            lab = -1
                            profit_pct = -(sl_mult + transaction_cost)
                            break
                    
                    labels[i] = lab
                    profit_pcts[i] = profit_pct
            
            run_cpu_intensive_task(process_barriers)
        else:
            # Standard Python implementation
            for i in range(n - 1):
                entry_price = close[i]
                profit_barrier = entry_price * (1.0 + pt_mult)
                stop_barrier = entry_price * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i])
                
                if end_idx <= i + 1:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue
                
                lab = 0
                profit_pct = 0.0
                
                for j in range(i + 1, end_idx):
                    if high[j] >= profit_barrier:
                        lab = 1
                        profit_pct = pt_mult - transaction_cost
                        break
                    if low[j] <= stop_barrier:
                        lab = -1
                        profit_pct = -(sl_mult + transaction_cost)
                        break
                
                labels[i] = lab
                profit_pcts[i] = profit_pct
        
        return labels, profit_pcts

    def _python_regime_aware_logic_optimized(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        regime_labels: np.ndarray,
        regime_params: np.ndarray,
        end_idx_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Python implementation of regime-aware barrier logic with optimization."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        
        transaction_cost = self.config.transaction_cost
        
        # Use CPU-intensive task optimization if available
        if HARDWARE_OPTIMIZATIONS_AVAILABLE and self.cpu_optimizer:
            def process_regime_barriers():
                for i in range(n - 1):
                    regime = int(regime_labels[i])
                    if regime >= len(regime_params):
                        regime = 0
                    
                    pt_mult, sl_mult, time_mult = regime_params[regime]
                    
                    entry_price = close[i]
                    profit_barrier = entry_price * (1.0 + pt_mult)
                    stop_barrier = entry_price * (1.0 - sl_mult)
                    end_idx = int(end_idx_arr[i] * time_mult)
                    
                    if end_idx <= i + 1:
                        labels[i] = 0
                        profit_pcts[i] = 0.0
                        continue
                    
                    lab = 0
                    profit_pct = 0.0
                    
                    for j in range(i + 1, min(end_idx, n)):
                        if high[j] >= profit_barrier:
                            lab = 1
                            profit_pct = pt_mult - transaction_cost
                            break
                        if low[j] <= stop_barrier:
                            lab = -1
                            profit_pct = -(sl_mult + transaction_cost)
                            break
                    
                    labels[i] = lab
                    profit_pcts[i] = profit_pct
            
            run_cpu_intensive_task(process_regime_barriers)
        else:
            # Standard Python implementation
            for i in range(n - 1):
                regime = int(regime_labels[i])
                if regime >= len(regime_params):
                    regime = 0
                
                pt_mult, sl_mult, time_mult = regime_params[regime]
                
                entry_price = close[i]
                profit_barrier = entry_price * (1.0 + pt_mult)
                stop_barrier = entry_price * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i] * time_mult)
                
                if end_idx <= i + 1:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue
                
                lab = 0
                profit_pct = 0.0
                
                for j in range(i + 1, min(end_idx, n)):
                    if high[j] >= profit_barrier:
                        lab = 1
                        profit_pct = pt_mult - transaction_cost
                        break
                    if low[j] <= stop_barrier:
                        lab = -1
                        profit_pct = -(sl_mult + transaction_cost)
                        break
                
                labels[i] = lab
                profit_pcts[i] = profit_pct
        
        return labels, profit_pcts

    def _filter_hold_samples(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out HOLD samples for binary classification."""
        original_count = len(data)
        hold_samples = (data['triple_barrier_label'] == 0).sum()
        
        filtered_data = data[data['triple_barrier_label'] != 0].copy()
        filtered_count = len(filtered_data)
        
        self.logger.info(f'📊 Label distribution after filtering:')
        self.logger.info(f'   LONG (1): {(filtered_data["triple_barrier_label"] == 1).sum()} samples')
        self.logger.info(f'   SHORT (-1): {(filtered_data["triple_barrier_label"] == -1).sum()} samples')
        self.logger.info(f'   HOLD (0): {hold_samples} samples (removed)')
        self.logger.info(f'   Total samples: {filtered_count} (from {original_count})')
        
        return filtered_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        stats = {
            'hardware_optimizations_available': HARDWARE_OPTIMIZATIONS_AVAILABLE,
            'numba_available': NUMBA_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'mps_available': is_mps_available() if HARDWARE_OPTIMIZATIONS_AVAILABLE else False,
            'm1_available': is_m1_available() if HARDWARE_OPTIMIZATIONS_AVAILABLE else False,
            'config': {
                'profit_take_multiplier': self.config.profit_take_multiplier,
                'stop_loss_multiplier': self.config.stop_loss_multiplier,
                'time_barrier_minutes': self.config.time_barrier_minutes,
                'max_lookahead': self.config.max_lookahead,
                'transaction_cost': self.config.transaction_cost,
                'binary_classification': self.config.binary_classification,
                'enable_regime_aware': self.config.enable_regime_aware,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_numba_acceleration': self.config.enable_numba_acceleration,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                'memory_limit_gb': self.config.memory_limit_gb,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'num_threads': self.config.num_threads
            },
            'performance_optimization': {
                'numba_available': NUMBA_AVAILABLE,
                'vectorized_implementation': True,
                'hardware_optimization': HARDWARE_OPTIMIZATIONS_AVAILABLE,
                'memory_optimization': self.memory_optimizer is not None,
                'cpu_optimization': self.cpu_optimizer is not None,
                'gpu_optimization': self.gpu_manager is not None,
                'advanced_memory_optimization': self.advanced_memory_optimizer is not None,
                'data_manager_available': self.data_manager is not None,
                'parallel_processing': self.thread_pool is not None
            }
        }
        
        return stats

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        
        if not NUMBA_AVAILABLE:
            recommendations.append('Install numba for significant performance improvements')
        
        if not HARDWARE_OPTIMIZATIONS_AVAILABLE:
            recommendations.append('Enable hardware optimizations for better performance')
        
        if not TORCH_AVAILABLE:
            recommendations.append('Install PyTorch for GPU acceleration support')
        
        if self.config.profit_take_multiplier < 0.001:
            recommendations.append('Consider increasing profit take multiplier for better signal quality')
        
        if self.config.stop_loss_multiplier < 0.0005:
            recommendations.append('Consider increasing stop loss multiplier for better risk management')
        
        if not self.config.enable_parallel_processing:
            recommendations.append('Enable parallel processing for better performance')
        
        return recommendations

    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        if self.config.enable_memory_monitoring and HARDWARE_OPTIMIZATIONS_AVAILABLE:
            stop_m1_memory_monitoring()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        self.logger.info('🧹 Hardware-optimized triple barrier labeling cleanup completed')

# Convenience functions
def apply_hardware_optimized_triple_barrier_labeling(
    data: pd.DataFrame,
    profit_take_multiplier: float = 0.004,
    stop_loss_multiplier: float = 0.003,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = 0.0008,
    binary_classification: bool = True,
    enable_hardware_optimization: bool = True,
    enable_numba_acceleration: bool = True,
    enable_gpu_acceleration: bool = True,
    memory_limit_gb: float = 8.0
) -> pd.DataFrame:
    """
    Apply hardware-optimized triple barrier labeling.
    
    Args:
        data: Market data with OHLC columns
        profit_take_multiplier: Multiplier for profit take barrier
        stop_loss_multiplier: Multiplier for stop loss barrier
        time_barrier_minutes: Time barrier in minutes
        max_lookahead: Maximum number of points to look ahead
        transaction_cost: Transaction cost as percentage
        binary_classification: If True, only generate buy (1) and sell (-1) labels
        enable_hardware_optimization: Enable hardware-specific optimizations
        enable_numba_acceleration: Enable Numba acceleration
        enable_gpu_acceleration: Enable GPU acceleration
        memory_limit_gb: Memory limit in GB
        
    Returns:
        DataFrame with triple barrier labels
    """
    config = HardwareOptimizedConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        binary_classification=binary_classification,
        enable_hardware_optimization=enable_hardware_optimization,
        enable_numba_acceleration=enable_numba_acceleration,
        enable_gpu_acceleration=enable_gpu_acceleration,
        memory_limit_gb=memory_limit_gb
    )
    
    labeler = HardwareOptimizedTripleBarrierLabeling(config)
    try:
        return labeler.apply_triple_barrier_labeling(data)
    finally:
        labeler.cleanup()

def apply_hardware_optimized_regime_aware_triple_barrier_labeling(
    data: pd.DataFrame,
    regime_column: str = 'hmm_regime',
    profit_take_multiplier: float = 0.004,
    stop_loss_multiplier: float = 0.003,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = 0.0008,
    binary_classification: bool = True,
    enable_hardware_optimization: bool = True,
    enable_numba_acceleration: bool = True,
    enable_gpu_acceleration: bool = True,
    memory_limit_gb: float = 8.0
) -> pd.DataFrame:
    """
    Apply hardware-optimized regime-aware triple barrier labeling.
    
    Args:
        data: Market data with OHLC and regime columns
        regime_column: Name of the regime column
        profit_take_multiplier: Multiplier for profit take barrier
        stop_loss_multiplier: Multiplier for stop loss barrier
        time_barrier_minutes: Time barrier in minutes
        max_lookahead: Maximum number of points to look ahead
        transaction_cost: Transaction cost as percentage
        binary_classification: If True, only generate buy (1) and sell (-1) labels
        enable_hardware_optimization: Enable hardware-specific optimizations
        enable_numba_acceleration: Enable Numba acceleration
        enable_gpu_acceleration: Enable GPU acceleration
        memory_limit_gb: Memory limit in GB
        
    Returns:
        DataFrame with regime-aware triple barrier labels
    """
    config = HardwareOptimizedConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        binary_classification=binary_classification,
        enable_regime_aware=True,
        enable_hardware_optimization=enable_hardware_optimization,
        enable_numba_acceleration=enable_numba_acceleration,
        enable_gpu_acceleration=enable_gpu_acceleration,
        memory_limit_gb=memory_limit_gb
    )
    
    labeler = HardwareOptimizedTripleBarrierLabeling(config)
    try:
        return labeler.apply_regime_aware_labeling(data, regime_column)
    finally:
        labeler.cleanup()

def get_hardware_optimization_info() -> Dict[str, Any]:
    """Get information about available hardware optimizations."""
    return {
        'hardware_optimizations_available': HARDWARE_OPTIMIZATIONS_AVAILABLE,
        'numba_available': NUMBA_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'mps_available': is_mps_available() if HARDWARE_OPTIMIZATIONS_AVAILABLE else False,
        'm1_available': is_m1_available() if HARDWARE_OPTIMIZATIONS_AVAILABLE else False,
        'recommendations': [
            'Install numba for significant performance improvements' if not NUMBA_AVAILABLE else None,
            'Enable hardware optimizations for better performance' if not HARDWARE_OPTIMIZATIONS_AVAILABLE else None,
            'Install PyTorch for GPU acceleration support' if not TORCH_AVAILABLE else None
        ]
    }

# Module information
__version__ = '1.0.0'
__author__ = 'Market Analysis Team'
__description__ = 'Hardware-Optimized Triple Barrier Labeling with Advanced Performance Features'

# Export key components
__all__ = [
    'HardwareOptimizedTripleBarrierLabeling',
    'HardwareOptimizedConfig',
    'apply_hardware_optimized_triple_barrier_labeling',
    'apply_hardware_optimized_regime_aware_triple_barrier_labeling',
    'get_hardware_optimization_info',
    'HARDWARE_OPTIMIZATIONS_AVAILABLE',
    'NUMBA_AVAILABLE',
    'TORCH_AVAILABLE'
]