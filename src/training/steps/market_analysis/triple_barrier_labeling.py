"""
MARKET_ANALYSIS Triple Barrier Labeling Implementation

This module provides comprehensive triple barrier labeling functionality for the market analysis pipeline.
It integrates regime-aware labeling, performance optimization, and comprehensive validation.

Key Features:
- Regime-aware triple barrier labeling with HMM integration
- Performance optimization with Numba acceleration
- Comprehensive validation and error handling
- Transaction cost modeling
- Binary and ternary classification support
- Integration with existing market analysis pipeline

DEPRECATED: This module is deprecated. Use unified_triple_barrier_labeler.py instead.
This file is kept for backward compatibility and will be removed in a future version.
"""

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time, cached
from src.utils.math_validation import safe_divide, validate_positive, MathValidationError

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import contextlib

# Hardware optimization imports
try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, create_m1_optimized_thread_pool, run_cpu_intensive_task
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, optimize_dataframe_for_m1, create_m1_optimized_array
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_dataframe_memory
    from src.utils.hardware.m1_optimizations import get_m1_memory_optimizer as get_advanced_m1_optimizer
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

# Try to import validation framework
try:
    from src.training.steps.step06_enhanced_validation_framework import (
        step06_function_validator, 
        step06_validation_context, 
        get_step06_validation_summary, 
        ValidationLevel
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    # Create fallback decorators
    def step06_function_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def step06_validation_context(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
    
    def get_step06_validation_summary():
        return {'error': 'Validation framework not available'}
    
    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'
    
    VALIDATION_AVAILABLE = False

# Constants for risk management
DEFAULT_PROFIT_TAKE_MULTIPLIER = 0.002  # 0.2% - conservative
DEFAULT_STOP_LOSS_MULTIPLIER = 0.001    # 0.1% - conservative
DEFAULT_TRANSACTION_COST = 0.0008       # 0.08% transaction cost
MIN_BARRIER_MULTIPLIER = 0.0005         # Minimum 0.05% barrier
MAX_BARRIER_MULTIPLIER = 0.05           # Maximum 5% barrier
EPSILON = 1e-10                         # Numerical stability constant

@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling parameters."""
    profit_take_multiplier: float = DEFAULT_PROFIT_TAKE_MULTIPLIER
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    binary_classification: bool = True
    regime_aware: bool = True
    regime_column: str = 'hmm_regime'
    enable_validation: bool = True
    enable_profiling: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.profit_take_multiplier < MIN_BARRIER_MULTIPLIER:
            self.profit_take_multiplier = MIN_BARRIER_MULTIPLIER
        if self.stop_loss_multiplier < MIN_BARRIER_MULTIPLIER:
            self.stop_loss_multiplier = MIN_BARRIER_MULTIPLIER
        if self.transaction_cost < 0:
            self.transaction_cost = 0.0

# Numba-accelerated triple barrier labeling function
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def _numba_triple_barrier_labels(
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        pt_mult: float, 
        sl_mult: float, 
        end_idx_arr: np.ndarray,
        transaction_cost: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated triple barrier labeling with profit tracking and transaction costs.
        
        Returns:
            labels: 1 for LONG position, -1 for SHORT position, 0 for HOLD
            profit_pcts: Actual profit/loss percentages at barrier hits (net of transaction costs)
            transaction_costs: Transaction costs incurred
        """
        labels = np.zeros(close.shape[0], dtype=np.int8)
        profit_pcts = np.zeros(close.shape[0], dtype=np.float64)
        transaction_costs = np.zeros(close.shape[0], dtype=np.float64)
        n = close.shape[0]
        
        for i in range(n - 1):
            entry_price = close[i]
            
            # Numerical stability check
            if entry_price <= EPSILON:
                labels[i] = 0
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
                continue
                
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
                continue
                
            lab = 0
            profit_pct = 0.0
            tx_cost = 0.0
            
            for j in range(i + 1, end_idx):
                if high[j] >= profit_barrier:
                    lab = 1
                    # Net profit after transaction costs
                    gross_profit = pt_mult
                    tx_cost = transaction_cost
                    profit_pct = gross_profit - tx_cost
                    break
                    
                if low[j] <= stop_barrier:
                    lab = -1
                    # Net loss including transaction costs
                    gross_loss = -sl_mult
                    tx_cost = transaction_cost
                    profit_pct = gross_loss - tx_cost
                    break
                    
            labels[i] = lab
            profit_pcts[i] = profit_pct
            transaction_costs[i] = tx_cost
            
        return (labels, profit_pcts, transaction_costs)

class MarketAnalysisTripleBarrierLabeling:
    """
    Comprehensive Triple Barrier Method for Market Analysis Pipeline.
    
    This implementation provides:
    - Regime-aware labeling with HMM integration
    - Performance optimization with Numba acceleration
    - Comprehensive validation and error handling
    - Transaction cost modeling
    - Binary and ternary classification support
    """
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """Initialize the triple barrier labeling system.
        
        Args:
            config: Configuration object with labeling parameters
        """
        self.config = config or TripleBarrierConfig()
        self.logger = get_logger('MarketAnalysisTripleBarrierLabeling')
        
        # Initialize hardware optimizers
        self._setup_hardware_optimizations()
        
        # Validate configuration
        self._validate_configuration()
        
        # Log initialization
        self._log_initialization()
        
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
            self.memory_optimizer = get_m1_memory_optimizer(8.0)  # 8GB limit
            
            # Initialize advanced M1 memory optimizer
            self.advanced_memory_optimizer = get_advanced_m1_optimizer(
                memory_limit_gb=8.0,
                enable_gc_tuning=True,
                enable_memory_leak_detection=True,
                enable_swap_management=True
            )
            
            self.logger.info('✅ Hardware optimizations initialized successfully')
            
        except Exception as e:
            self.logger.warning(f'⚠️ Hardware optimization setup failed: {e}')
            self.cpu_optimizer = None
            self.gpu_manager = None
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
        
    def _validate_configuration(self):
        """Validate configuration parameters."""
        try:
            # Validate profit take multiplier
            if self.config.profit_take_multiplier < MIN_BARRIER_MULTIPLIER:
                raise MathValidationError(f"Profit take too small ({self.config.profit_take_multiplier:.4f} < {MIN_BARRIER_MULTIPLIER:.4f})")
            if self.config.profit_take_multiplier > MAX_BARRIER_MULTIPLIER:
                raise MathValidationError(f"Profit take too large ({self.config.profit_take_multiplier:.4f} > {MAX_BARRIER_MULTIPLIER:.4f})")
            
            # Validate stop loss multiplier
            if self.config.stop_loss_multiplier < MIN_BARRIER_MULTIPLIER:
                raise MathValidationError(f"Stop loss too small ({self.config.stop_loss_multiplier:.4f} < {MIN_BARRIER_MULTIPLIER:.4f})")
            if self.config.stop_loss_multiplier > MAX_BARRIER_MULTIPLIER:
                raise MathValidationError(f"Stop loss too large ({self.config.stop_loss_multiplier:.4f} > {MAX_BARRIER_MULTIPLIER:.4f})")
            
            # Check risk-reward ratio
            risk_reward_ratio = safe_divide(
                self.config.profit_take_multiplier, 
                self.config.stop_loss_multiplier, 
                default=0.0
            )
            if risk_reward_ratio < 1.0:
                self.logger.warning(f"⚠️ Risk-reward ratio < 1.0 ({risk_reward_ratio:.2f}) - may be unprofitable")
            
            # Check if barriers are too close
            barrier_diff = abs(self.config.profit_take_multiplier - self.config.stop_loss_multiplier)
            if barrier_diff < 0.0005:
                raise MathValidationError(f"Profit take and stop loss too close (diff: {barrier_diff:.4f} < 0.05%)")
            
            self.logger.info(f"✅ Configuration validated successfully")
            
        except MathValidationError as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _log_initialization(self):
        """Log initialization parameters."""
        self.logger.info('🚀 Initializing Market Analysis Triple Barrier Labeling')
        self.logger.info(f'📋 Configuration:')
        self.logger.info(f'   → Profit take: {self.config.profit_take_multiplier:.4f} ({self.config.profit_take_multiplier*100:.2f}%)')
        self.logger.info(f'   → Stop loss: {self.config.stop_loss_multiplier:.4f} ({self.config.stop_loss_multiplier*100:.2f}%)')
        self.logger.info(f'   → Transaction cost: {self.config.transaction_cost:.4f} ({self.config.transaction_cost*100:.2f}%)')
        self.logger.info(f'   → Time barrier: {self.config.time_barrier_minutes} minutes')
        self.logger.info(f'   → Max lookahead: {self.config.max_lookahead}')
        self.logger.info(f'   → Binary classification: {self.config.binary_classification}')
        self.logger.info(f'   → Regime aware: {self.config.regime_aware}')
        self.logger.info(f'   → Numba acceleration: {NUMBA_AVAILABLE}')
        self.logger.info(f'   → Hardware optimizations: {HARDWARE_OPTIMIZATIONS_AVAILABLE}')
        self.logger.info(f'   → Validation framework: {VALIDATION_AVAILABLE}')

    @step06_function_validator(function_type='labeling', validation_level=ValidationLevel.COMPREHENSIVE)
    @traced(span_name='apply_triple_barrier_labeling')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    @log_execution_time()
    def apply_triple_barrier_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier labeling to market data.
        
        Args:
            data: DataFrame with OHLCV data and optional regime information
            
        Returns:
            DataFrame with triple barrier labels and profit tracking
        """
        with step06_validation_context('apply_triple_barrier_labeling', 'labeling'):
            self.logger.info(f'🏷️ Starting triple barrier labeling')
            self.logger.info(f'   Input data shape: {data.shape}')
            self.logger.info(f'   Available columns: {list(data.columns)}')
        
        # Validate and prepare data
        validated_data = self._validate_and_prepare_data(data)
        if validated_data is None:
            return pd.DataFrame()
        
        # Apply hardware optimizations
        validated_data = self._optimize_data_for_hardware(validated_data)
        
        # Apply regime-aware labeling if enabled
        if self.config.regime_aware and self.config.regime_column in validated_data.columns:
            self.logger.info('🎯 Applying regime-aware triple barrier labeling')
            labeled_data = self._apply_regime_aware_labeling(validated_data)
        else:
            self.logger.info('📊 Applying standard triple barrier labeling')
            labeled_data = self._apply_standard_labeling(validated_data)
        
        # Apply post-processing
        final_data = self._apply_post_processing(labeled_data)
        
        # Log results
        self._log_labeling_results(final_data)
        
        return final_data
    
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
    
    def _validate_and_prepare_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate input data and prepare for processing."""
        if data is None or data.empty:
            self.logger.error('❌ Input data is None or empty')
            return None
        
        # Standardize column names
        rename_map = self._get_column_rename_map(data)
        if rename_map:
            data = data.rename(columns=rename_map)
            self.logger.info(f'📝 Renamed columns: {rename_map}')
        
        # Check required columns
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f'❌ Missing required OHLC columns {missing_columns}')
            return None
        
        # Validate data quality
        if not self._validate_data_quality(data):
            return None
        
        # Create working copy
        labeled_data = data.copy()
        
        return labeled_data
    
    def _get_column_rename_map(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get column rename mapping for standardization."""
        rename_map = {}
        canonical_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume'
        }
        
        for original, canonical in canonical_map.items():
            if original in data.columns and canonical not in data.columns:
                rename_map[original] = canonical
                
        return rename_map
    
    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate data quality with comprehensive checks."""
        # Check for sufficient data
        if len(data) < 2:
            self.logger.error('❌ Insufficient data for labeling (need at least 2 rows)')
            return False
        
        # Check for numerical stability
        for col in ['close', 'high', 'low']:
            if data[col].isna().all():
                self.logger.error(f'❌ Column {col} contains only NaN values')
                return False
            if (data[col] <= 0).any():
                self.logger.warning(f'⚠️ Column {col} contains non-positive values')
        
        # Check OHLC consistency
        if not self._validate_ohlc_consistency(data):
            return False
        
        return True
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> bool:
        """Validate OHLC consistency."""
        # High should be >= max(open, close)
        high_consistent = (data['high'] >= np.maximum(data['open'], data['close'])).all()
        if not high_consistent:
            self.logger.warning('⚠️ OHLC consistency issue: high < max(open, close)')
        
        # Low should be <= min(open, close)
        low_consistent = (data['low'] <= np.minimum(data['open'], data['close'])).all()
        if not low_consistent:
            self.logger.warning('⚠️ OHLC consistency issue: low > min(open, close)')
        
        return high_consistent and low_consistent
    
    def _apply_regime_aware_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-aware triple barrier labeling."""
        try:
            # Get unique regimes
            regimes = data[self.config.regime_column].unique()
            self.logger.info(f'📊 Found {len(regimes)} unique regimes: {regimes}')
            
            # Initialize result arrays
            n = len(data)
            labels = np.zeros(n, dtype=np.int8)
            profit_pcts = np.zeros(n, dtype=np.float64)
            transaction_costs = np.zeros(n, dtype=np.float64)
            
            # Process each regime separately
            for regime in regimes:
                regime_mask = data[self.config.regime_column] == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) < 2:
                    continue
                
                # Get regime-specific parameters (could be optimized per regime)
                regime_pt_mult = self.config.profit_take_multiplier
                regime_sl_mult = self.config.stop_loss_multiplier
                
                # Apply triple barrier logic to regime data
                regime_labels, regime_profits, regime_costs = self._calculate_barriers_for_regime(
                    regime_data, regime_pt_mult, regime_sl_mult
                )
                
                # Store results
                labels[regime_mask] = regime_labels
                profit_pcts[regime_mask] = regime_profits
                transaction_costs[regime_mask] = regime_costs
            
            # Add results to dataframe
            data['label'] = labels
            data['potential_profit_pct'] = profit_pcts
            data['transaction_cost'] = transaction_costs
            data['net_profit_pct'] = profit_pcts  # Net profit after transaction costs
            data['labeling_method'] = 'regime_aware'
            
            return data
            
        except Exception as e:
            self.logger.error(f'❌ Error in regime-aware labeling: {e}')
            return self._apply_standard_labeling(data)
    
    def _apply_standard_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard triple barrier labeling."""
        n = len(data)
        close = data['close'].to_numpy()
        high = data['high'].to_numpy()
        low = data['low'].to_numpy()
        idx = data.index
        
        # Calculate end indices
        end_idx_arr = self._calculate_end_indices(n, idx)
        
        # Apply barrier logic
        labels, profit_pcts, transaction_costs = self._apply_barrier_logic(
            close, high, low, end_idx_arr
        )
        
        # Add results to dataframe
        data['label'] = labels
        data['potential_profit_pct'] = profit_pcts
        data['transaction_cost'] = transaction_costs
        data['net_profit_pct'] = profit_pcts  # Net profit after transaction costs
        data['labeling_method'] = 'standard'
        
        return data
    
    def _calculate_barriers_for_regime(self, regime_data: pd.DataFrame, pt_mult: float, sl_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate barriers for a specific regime."""
        n = len(regime_data)
        close = regime_data['close'].to_numpy()
        high = regime_data['high'].to_numpy()
        low = regime_data['low'].to_numpy()
        idx = regime_data.index
        
        # Calculate end indices
        end_idx_arr = self._calculate_end_indices(n, idx)
        
        # Apply barrier logic
        return self._apply_barrier_logic(close, high, low, end_idx_arr, pt_mult, sl_mult)
    
    def _calculate_end_indices(self, n: int, idx: pd.Index) -> np.ndarray:
        """Calculate end indices for barrier evaluation."""
        arange_n = np.arange(n, dtype=np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + int(self.config.max_lookahead), n)
        
        if isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing:
            try:
                idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.config.time_barrier_minutes) * np.int64(60000000000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns, end_times, side='right')
            except Exception as e:
                self.logger.warning(f'⚠️ Time barrier calculation failed: {e}, using lookahead only')
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
            
        return np.minimum(end_by_lookahead, end_by_time).astype(np.int64)
    
    def _apply_barrier_logic(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        end_idx_arr: np.ndarray,
        pt_mult: Optional[float] = None,
        sl_mult: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply barrier logic with performance optimization."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        transaction_costs = np.zeros(n, dtype=np.float64)
        
        # Use provided multipliers or defaults
        pt_mult = pt_mult or self.config.profit_take_multiplier
        sl_mult = sl_mult or self.config.stop_loss_multiplier
        tx_cost = self.config.transaction_cost
        
        # Use Numba acceleration if available and data is large enough
        use_numba = (NUMBA_AVAILABLE and 
                    callable(globals().get('_numba_triple_barrier_labels')) and 
                    n >= 512)
        
        if use_numba:
            self.logger.info('⚡ Using Numba-accelerated triple barrier labeling')
            labels, profit_pcts, transaction_costs = _numba_triple_barrier_labels(
                close.astype(np.float64), 
                high.astype(np.float64), 
                low.astype(np.float64), 
                pt_mult, 
                sl_mult, 
                end_idx_arr.astype(np.int64),
                tx_cost
            )
        else:
            self.logger.info('🐍 Using Python triple barrier labeling')
            labels, profit_pcts, transaction_costs = self._apply_barrier_logic_python(
                close, high, low, end_idx_arr, pt_mult, sl_mult, tx_cost
            )
            
        return labels, profit_pcts, transaction_costs
    
    def _apply_barrier_logic_python(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        end_idx_arr: np.ndarray,
        pt_mult: float,
        sl_mult: float,
        tx_cost: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply barrier logic in Python."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        transaction_costs = np.zeros(n, dtype=np.float64)
        
        for i in range(n - 1):
            entry_price = close[i]
            
            # Numerical stability check
            if entry_price <= EPSILON:
                labels[i] = 0
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
                continue
                
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
                continue
                
            # Get window data
            win_high = high[i + 1:end_idx]
            win_low = low[i + 1:end_idx]
            
            # Find barrier hits
            profit_hits = np.where(win_high >= profit_barrier)[0]
            stop_hits = np.where(win_low <= stop_barrier)[0]
            
            # Determine label and profit
            if profit_hits.size == 0 and stop_hits.size == 0:
                # No barriers hit - time barrier
                labels[i] = 0
                profit_pcts[i] = 0.0
                transaction_costs[i] = 0.0
            elif profit_hits.size == 0:
                # Only stop loss hit
                labels[i] = -1
                profit_pcts[i] = -sl_mult - tx_cost
                transaction_costs[i] = tx_cost
            elif stop_hits.size == 0:
                # Only profit take hit
                labels[i] = 1
                profit_pcts[i] = pt_mult - tx_cost
                transaction_costs[i] = tx_cost
            else:
                # Both hit - use first one
                if profit_hits[0] <= stop_hits[0]:
                    labels[i] = 1
                    profit_pcts[i] = pt_mult - tx_cost
                    transaction_costs[i] = tx_cost
                else:
                    labels[i] = -1
                    profit_pcts[i] = -sl_mult - tx_cost
                    transaction_costs[i] = tx_cost
                    
        return labels, profit_pcts, transaction_costs
    
    def _apply_post_processing(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing and filtering."""
        original_count = len(labeled_data)
        
        # Filter out HOLD samples if binary classification
        if self.config.binary_classification:
            hold_samples = (labeled_data['label'] == 0).sum()
            labeled_data = labeled_data[labeled_data['label'] != 0].copy()
            self.logger.info(f'📊 Filtered {hold_samples} HOLD samples for binary classification')
        
        return labeled_data
    
    def _log_labeling_results(self, labeled_data: pd.DataFrame):
        """Log labeling results."""
        if len(labeled_data) == 0:
            self.logger.warning('⚠️ No labeled data produced')
            return
        
        # Label distribution
        label_counts = labeled_data['label'].value_counts()
        self.logger.info('📊 Label distribution:')
        for label, count in label_counts.items():
            label_name = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}.get(label, f'LABEL_{label}')
            self.logger.info(f'   {label_name}: {count} samples')
        
        # Profit statistics
        if 'net_profit_pct' in labeled_data.columns:
            long_profits = labeled_data[labeled_data['label'] == 1]['net_profit_pct']
            short_profits = labeled_data[labeled_data['label'] == -1]['net_profit_pct']
            
            self.logger.info('💰 Profit statistics (net of transaction costs):')
            if len(long_profits) > 0:
                self.logger.info(f'   LONG signals - Avg: {long_profits.mean():.4f}, Max: {long_profits.max():.4f}, Min: {long_profits.min():.4f}')
            if len(short_profits) > 0:
                self.logger.info(f'   SHORT signals - Avg: {short_profits.mean():.4f}, Max: {short_profits.max():.4f}, Min: {short_profits.min():.4f}')
            
            total_profits = labeled_data['net_profit_pct']
            self.logger.info(f'   Overall - Avg: {total_profits.mean():.4f}, Std: {total_profits.std():.4f}')
            
            # Transaction cost analysis
            if 'transaction_cost' in labeled_data.columns:
                total_tx_costs = labeled_data['transaction_cost'].sum()
                self.logger.info(f'   Total transaction costs: {total_tx_costs:.4f} ({total_tx_costs*100:.2f}%)')
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive labeling report."""
        self.logger.info('📋 Generating comprehensive triple barrier labeling report...')
        
        validation_summary = {}
        if VALIDATION_AVAILABLE:
            try:
                validation_summary = get_step06_validation_summary()
            except Exception as e:
                self.logger.warning(f'Could not get validation summary: {e}')
        
        internal_stats = {
            'labeling_configuration': {
                'profit_take_multiplier': self.config.profit_take_multiplier,
                'stop_loss_multiplier': self.config.stop_loss_multiplier,
                'transaction_cost': self.config.transaction_cost,
                'time_barrier_minutes': self.config.time_barrier_minutes,
                'max_lookahead': self.config.max_lookahead,
                'binary_classification': self.config.binary_classification,
                'regime_aware': self.config.regime_aware,
                'regime_column': self.config.regime_column
            },
            'performance_optimization': {
                'numba_available': NUMBA_AVAILABLE,
                'hardware_optimizations_available': HARDWARE_OPTIMIZATIONS_AVAILABLE,
                'vectorized_implementation': True,
                'regime_aware_implementation': self.config.regime_aware
            },
            'validation_status': {
                'validation_framework_available': VALIDATION_AVAILABLE,
                'comprehensive_validation_enabled': self.config.enable_validation
            }
        }
        
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': validation_summary,
            'internal_statistics': internal_stats,
            'recommendations': self._generate_recommendations(internal_stats),
            'performance_analysis': self._analyze_performance()
        }
        
        self.logger.info('✅ Comprehensive triple barrier labeling report generated')
        return comprehensive_report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on execution statistics."""
        recommendations = []
        
        config = stats['labeling_configuration']
        
        # Risk parameter recommendations
        if config['profit_take_multiplier'] < 0.001:
            recommendations.append('Consider increasing profit take multiplier for better signal quality')
        if config['stop_loss_multiplier'] < 0.0005:
            recommendations.append('Consider increasing stop loss multiplier for better risk management')
        
        # Transaction cost recommendations
        if config['transaction_cost'] > 0.001:
            recommendations.append('High transaction costs detected - consider optimizing execution')
        
        # Performance recommendations
        if not stats['performance_optimization']['numba_available']:
            recommendations.append('Install numba for significant performance improvements')
        
        if not stats['performance_optimization']['hardware_optimizations_available']:
            recommendations.append('Enable hardware optimizations for better performance')
        
        # Validation recommendations
        if not stats['validation_status']['validation_framework_available']:
            recommendations.append('Enable validation framework for better error tracking and reporting')
        
        return recommendations
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics."""
        return {
            'implementation_type': 'market_analysis_triple_barrier',
            'numba_acceleration': NUMBA_AVAILABLE,
            'hardware_optimizations': HARDWARE_OPTIMIZATIONS_AVAILABLE,
            'binary_classification_optimized': self.config.binary_classification,
            'regime_aware_enabled': self.config.regime_aware,
            'profit_tracking_enabled': True,
            'transaction_cost_modeling': True,
            'comprehensive_validation': self.config.enable_validation
        }

# Convenience functions for easy integration
def create_triple_barrier_labeler(
    profit_take_multiplier: float = DEFAULT_PROFIT_TAKE_MULTIPLIER,
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    binary_classification: bool = True,
    regime_aware: bool = True,
    regime_column: str = 'hmm_regime'
) -> MarketAnalysisTripleBarrierLabeling:
    """Create a triple barrier labeler with specified parameters.
    
    Args:
        profit_take_multiplier: Profit take multiplier (default: 0.2%)
        stop_loss_multiplier: Stop loss multiplier (default: 0.1%)
        time_barrier_minutes: Time barrier in minutes (default: 30)
        max_lookahead: Maximum lookahead (default: 100)
        transaction_cost: Transaction cost percentage (default: 0.08%)
        binary_classification: Whether to use binary classification (default: True)
        regime_aware: Whether to use regime-aware labeling (default: True)
        regime_column: Column name for regime information (default: 'hmm_regime')
    
    Returns:
        Configured MarketAnalysisTripleBarrierLabeling instance
    """
    config = TripleBarrierConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        binary_classification=binary_classification,
        regime_aware=regime_aware,
        regime_column=regime_column
    )
    
    return MarketAnalysisTripleBarrierLabeling(config)

def apply_triple_barrier_labeling(
    data: pd.DataFrame,
    profit_take_multiplier: float = DEFAULT_PROFIT_TAKE_MULTIPLIER,
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    binary_classification: bool = True,
    regime_aware: bool = True,
    regime_column: str = 'hmm_regime'
) -> pd.DataFrame:
    """Apply triple barrier labeling to data.
    
    Args:
        data: DataFrame with OHLCV data
        profit_take_multiplier: Profit take multiplier (default: 0.2%)
        stop_loss_multiplier: Stop loss multiplier (default: 0.1%)
        time_barrier_minutes: Time barrier in minutes (default: 30)
        max_lookahead: Maximum lookahead (default: 100)
        transaction_cost: Transaction cost percentage (default: 0.08%)
        binary_classification: Whether to use binary classification (default: True)
        regime_aware: Whether to use regime-aware labeling (default: True)
        regime_column: Column name for regime information (default: 'hmm_regime')
    
    Returns:
        DataFrame with triple barrier labels
    """
    labeler = create_triple_barrier_labeler(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        binary_classification=binary_classification,
        regime_aware=regime_aware,
        regime_column=regime_column
    )
    
    return labeler.apply_triple_barrier_labeling(data)

# Benchmark function
@handles_errors(exceptions=(Exception,), default_return={})
def benchmark_triple_barrier_methods(data: pd.DataFrame) -> Dict[str, float]:
    """Benchmark triple barrier labeling methods."""
    start_time = time.time()
    
    # Test standard implementation
    labeler = create_triple_barrier_labeler()
    labeled_data = labeler.apply_triple_barrier_labeling(data)
    
    standard_time = time.time() - start_time
    
    # Test regime-aware implementation if regime data is available
    regime_time = 0.0
    if 'hmm_regime' in data.columns:
        start_time = time.time()
        regime_labeler = create_triple_barrier_labeler(regime_aware=True)
        regime_labeled_data = regime_labeler.apply_triple_barrier_labeling(data)
        regime_time = time.time() - start_time
    
    return {
        'standard_time': standard_time,
        'regime_aware_time': regime_time,
        'data_size': len(data),
        'labeled_samples': len(labeled_data),
        'numba_available': NUMBA_AVAILABLE,
        'hardware_optimizations_available': HARDWARE_OPTIMIZATIONS_AVAILABLE,
        'validation_available': VALIDATION_AVAILABLE
    }

# DEPRECATION WARNING AND MIGRATION
import warnings

def _deprecation_warning():
    """Show deprecation warning."""
    warnings.warn(
        "triple_barrier_labeling.py is deprecated. Use unified_triple_barrier_labeler.py instead. "
        "This module will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Override the main classes to show deprecation warnings
class MarketAnalysisTripleBarrierLabeling:
    """DEPRECATED: Use UnifiedTripleBarrierLabeler from unified_triple_barrier_labeler.py"""
    
    def __init__(self, *args, **kwargs):
        _deprecation_warning()
        # Import and use the unified implementation
        from .unified_triple_barrier_labeler import UnifiedTripleBarrierLabeler, TripleBarrierConfig
        self._unified_labeler = UnifiedTripleBarrierLabeler(*args, **kwargs)
    
    def apply_triple_barrier_labeling(self, data):
        """DEPRECATED: Use UnifiedTripleBarrierLabeler.apply_labeling() instead."""
        _deprecation_warning()
        result = self._unified_labeler.apply_labeling(data)
        return result.labeled_data if result.success else pd.DataFrame()

# Override convenience functions
def create_triple_barrier_labeler(*args, **kwargs):
    """DEPRECATED: Use unified_triple_barrier_labeler.create_triple_barrier_labeler() instead."""
    _deprecation_warning()
    from .unified_triple_barrier_labeler import create_triple_barrier_labeler as unified_create
    return unified_create(*args, **kwargs)

def apply_triple_barrier_labeling(*args, **kwargs):
    """DEPRECATED: Use unified_triple_barrier_labeler.apply_triple_barrier_labeling() instead."""
    _deprecation_warning()
    from .unified_triple_barrier_labeler import apply_triple_barrier_labeling as unified_apply
    result = unified_apply(*args, **kwargs)
    return result.labeled_data if result.success else pd.DataFrame()

if __name__ == '__main__':
    # Test the implementation
    tprint('🧪 Testing Market Analysis Triple Barrier Labeling (DEPRECATED)')
    tprint('⚠️  This module is deprecated. Use unified_triple_barrier_labeler.py instead.')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'hmm_regime': np.random.choice([0, 1, 2], 1000)  # Add regime data
    }, index=dates)
    
    # Test with deprecation warnings
    tprint('\n📊 Testing deprecated triple barrier labeling...')
    try:
        standard_labeler = create_triple_barrier_labeler(regime_aware=False)
        standard_labeled = standard_labeler.apply_triple_barrier_labeling(data)
        tprint(f'Standard labeling completed: {len(standard_labeled)} samples labeled')
        
        # Test regime-aware labeling
        tprint('\n🎯 Testing regime-aware triple barrier labeling...')
        regime_labeler = create_triple_barrier_labeler(regime_aware=True)
        regime_labeled = regime_labeler.apply_triple_barrier_labeling(data)
        tprint(f'Regime-aware labeling completed: {len(regime_labeled)} samples labeled')
        
        tprint('✅ Deprecated Market Analysis Triple Barrier Labeling test completed!')
        tprint('⚠️  Please migrate to unified_triple_barrier_labeler.py for better performance and reliability.')
        
    except Exception as e:
        tprint(f'❌ Test failed: {e}')
        tprint('⚠️  This is expected as the deprecated module may have issues.')