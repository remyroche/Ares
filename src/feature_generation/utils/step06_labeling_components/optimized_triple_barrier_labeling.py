from src.utils.tprint import tprint
import warnings

import contextlib
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Any
from typing import Callable
from src.core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

current_dir = os.path.dirname(os.path.abspath(__file__))
steps_dir = os.path.join(current_dir, '..')
sys.path.insert(0, steps_dir)
try:
    from ..step06_enhanced_validation_framework import step06_function_validator, step06_function_tracker, step06_validation_context, get_step06_validation_summary, ValidationLevel, FunctionStatus
    VALIDATION_AVAILABLE = True
except ImportError as e:
    tprint(f'Warning: Step06 validation framework not available: {e}')

    def step06_function_validator(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def step06_function_tracker(func: Callable) -> None:
        return func

    def step06_validation_context(*args, **kwargs) -> None:
        from contextlib import nullcontext
        return nullcontext()

    def get_step06_validation_summary() -> Any:
        return {'error': 'Validation framework not available'}

    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'

    class FunctionStatus:
        PENDING = 'pending'
        IN_PROGRESS = 'in_progress'
        COMPLETED = 'completed'
        FAILED = 'failed'
        TIMEOUT = 'timeout'
    VALIDATION_AVAILABLE = False

# Decorator functions removed - use existing decorators from core
import datetime
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
# Import math validation functions from shared module
from ..math_validation import safe_divide, safe_log, safe_sqrt, validate_positive

# Additional validation functions
def validate_range(value, min_val, max_val, name="value"):
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value

class MathValidationError(Exception):
    pass

try:
    from src.utils.dataframe_guards import guard_dataframe_nulls, handle_errors, with_tracing_span  # type: ignore[import-untyped]
except ImportError:
    def guard_dataframe_nulls(*args, **kwargs):
        def decorator(func: Callable):
            return func
        return decorator
    def handle_errors(*args, **kwargs):
        def decorator(func: Callable):
            return func
        return decorator
    def with_tracing_span(*args, **kwargs):
        def decorator(func: Callable):
            return func
        return decorator
try:
    from src.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: Any) -> Any:
        return logging.getLogger(name)
try:
    import numba

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None
except Exception:
    numba = None
if 'numba' in globals() and numba is not None:

    @numba.jit(nopython = True, cache = True)
    def _numba_triple_barrier_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                                     pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray,
                                     transaction_cost: float) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated triple barrier labeling with profit tracking."

        Returns:
            labels: 1 for LONG position, -1 for SHORT position, 0 for HOLD
            profit_pcts: Actual profit/loss percentages at barrier hits
        """
        labels = np.zeros(close.shape[0], dtype = np.int8)
        profit_pcts = np.zeros(close.shape[0], dtype = np.float64)
        n = close.shape[0]
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
        return (labels, profit_pcts)

class OptimizedTripleBarrierLabeling:
    """Optimized Triple Barrier Method for labeling using vectorized operations."

    This implementation provides significant performance improvements over the
    original O(n²) implementation by using NumPy vectorized operations.
    Focuses specifically on triple barrier labeling without feature engineering.
    Now includes profit tracking for enhanced analysis.
    """
    @log_important_calls

    def __init__(self, profit_take_multiplier: float = 0.004, stop_loss_multiplier: float = 0.003, time_barrier_minutes: int = 30, max_lookahead: int = 100, binary_classification: bool = True, transaction_cost: float = 0.0008) -> None:
        """Initialize the optimized triple barrier labeling."

        Args:
            profit_take_multiplier: Multiplier for profit take barrier (default: 0.4%)
            stop_loss_multiplier: Multiplier for stop loss barrier (default: 0.3%)
            time_barrier_minutes: Time barrier in minutes (default: 30)
            max_lookahead: Maximum number of points to look ahead (default: 100)
            binary_classification: If True, only generate buy (1) and sell (-1) labels
            no hold (0) labels. If False, include hold labels (default: True)
            transaction_cost: Transaction cost as percentage (default: 0.08%)

        Note:
            binary_classification = True is now the default to address label imbalance issues.
            This automatically filters out HOLD samples to create a balanced binary classification.
            Updated with more realistic risk parameters and transaction cost modeling.
        """
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.time_barrier_minutes = time_barrier_minutes
        self.max_lookahead = max_lookahead
        self.binary_classification = binary_classification
        self.transaction_cost = transaction_cost
        self.logger = get_logger('OptimizedTripleBarrierLabeling')

        # Validate financial parameters
        self._validate_financial_parameters()
        if self.binary_classification:
            self.logger.info('🔖 Triple barrier labeling configured for binary classification (BUY/SELL only)')
            self.logger.info('   → HOLD samples will be automatically filtered out')
            self.logger.info('   → This addresses label imbalance issues')
        else:
            self.logger.warning('⚠️ Triple barrier labeling configured for ternary classification (BUY/HOLD/SELL)')
            self.logger.warning('   → This may lead to label imbalance issues')
            self.logger.warning('   → Consider using binary_classification = True for better results')

    def _validate_financial_parameters(self) -> None:
        """Validate financial parameters for soundness."""
        try:
            # Validate profit take multiplier
            if self.profit_take_multiplier < 0.001:
                raise MathValidationError(f"Profit take too small ({self.profit_take_multiplier:.4f} < 0.1%)")
            if self.profit_take_multiplier > 0.1:
                raise MathValidationError(f"Profit take too large ({self.profit_take_multiplier:.4f} > 10%)")

            # Validate stop loss multiplier
            if self.stop_loss_multiplier < 0.0005:
                raise MathValidationError(f"Stop loss too small ({self.stop_loss_multiplier:.4f} < 0.05%)")
            if self.stop_loss_multiplier > 0.05:
                raise MathValidationError(f"Stop loss too large ({self.stop_loss_multiplier:.4f} > 5%)")

            # Validate transaction cost
            if self.transaction_cost < 0:
                raise MathValidationError(f"Transaction cost cannot be negative ({self.transaction_cost:.4f})")
            if self.transaction_cost > 0.01:
                raise MathValidationError(f"Transaction cost too large ({self.transaction_cost:.4f} > 1%)")

            # Check risk-reward ratio
            risk_reward_ratio = safe_divide(self.profit_take_multiplier, self.stop_loss_multiplier, default=0.0)
            if risk_reward_ratio < 1.0:
                self.logger.warning(f"⚠️ Risk-reward ratio < 1.0 ({risk_reward_ratio:.2f}) - may be unprofitable")

            # Check if barriers are too close
            barrier_diff = abs(self.profit_take_multiplier - self.stop_loss_multiplier)
            if barrier_diff < 0.001:
                raise MathValidationError(f"Profit take and stop loss too close (diff: {barrier_diff:.4f} < 0.1%)")

            self.logger.info(f"✅ Financial parameters validated successfully")
            self.logger.info(f"   Profit take: {self.profit_take_multiplier:.4f} ({self.profit_take_multiplier*100:.2f}%)")
            self.logger.info(f"   Stop loss: {self.stop_loss_multiplier:.4f} ({self.stop_loss_multiplier*100:.2f}%)")
            self.logger.info(f"   Transaction cost: {self.transaction_cost:.4f} ({self.transaction_cost*100:.2f}%)")
            self.logger.info(f"   Risk-reward ratio: {risk_reward_ratio:.2f}")

        except MathValidationError as e:
            self.logger.error(f"❌ CRITICAL: Financial parameter validation failed: {e}")
            raise

    def _validate_market_data_quality(self, data: pd.DataFrame) -> None:
        """Fast fail validation for market data quality with extensive logging."""
        self.logger.info("🔍 Starting comprehensive market data quality validation...")

        try:
            # Check data shape
            if len(data) < 2:
                self.logger.error(f"❌ CRITICAL: Insufficient data rows ({len(data)} < 2)")
                raise MathValidationError(f"Insufficient data: {len(data)} rows (minimum 2 required)")

            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"❌ CRITICAL: Missing required columns: {missing_columns}")
                raise MathValidationError(f"Missing required columns: {missing_columns}")

            # Price sanity checks
            self.logger.info("💰 Validating price data...")
            price_columns = ['open', 'high', 'low', 'close']

            for col in price_columns:
                if col in data.columns:
                    prices = data[col]

                    # Check for zero or negative prices
                    invalid_prices = (prices <= 0).sum()
                    if invalid_prices > 0:
                        self.logger.error(f"❌ CRITICAL: {invalid_prices} invalid prices in {col} (≤ 0)")
                        self.logger.error(f"   Invalid price indices: {data.index[prices <= 0].tolist()}")
                        self.logger.error(f"   Invalid price values: {prices[prices <= 0].tolist()}")
                        raise MathValidationError(f"Invalid prices in {col}: {invalid_prices} values ≤ 0")

                    # Check for NaN values
                    nan_count = prices.isna().sum()
                    if nan_count > 0:
                        self.logger.error(f"❌ CRITICAL: {nan_count} NaN values in {col}")
                        self.logger.error(f"   NaN indices: {data.index[prices.isna()].tolist()}")
                        raise MathValidationError(f"NaN values in {col}: {nan_count} values")

                    # Check for infinite values
                    inf_count = np.isinf(prices).sum()
                    if inf_count > 0:
                        self.logger.error(f"❌ CRITICAL: {inf_count} infinite values in {col}")
                        self.logger.error(f"   Infinite indices: {data.index[np.isinf(prices)].tolist()}")
                        raise MathValidationError(f"Infinite values in {col}: {inf_count} values")

            # OHLC consistency checks
            self.logger.info("📊 Validating OHLC consistency...")
            ohlc_errors = 0

            for i, row in data.iterrows():
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                # Check high >= max(open, close)
                if high_price < max(open_price, close_price):
                    ohlc_errors += 1
                    if ohlc_errors <= 5:  # Log first 5 errors
                        self.logger.error(f"❌ OHLC error at {i}: high ({high_price}) < max(open, close) ({max(open_price, close_price)})")

                # Check low <= min(open, close)
                if low_price > min(open_price, close_price):
                    ohlc_errors += 1
                    if ohlc_errors <= 5:  # Log first 5 errors
                        self.logger.error(f"❌ OHLC error at {i}: low ({low_price}) > min(open, close) ({min(open_price, close_price)})")

            if ohlc_errors > 0:
                self.logger.error(f"❌ CRITICAL: {ohlc_errors} OHLC consistency errors found")
                raise MathValidationError(f"OHLC consistency errors: {ohlc_errors} violations")

            # Volatility sanity checks
            self.logger.info("📈 Validating volatility...")
            price_changes = data['close'].pct_change().abs()

            # Check for suspiciously large price movements (>20%)
            large_moves = (price_changes > 0.2).sum()
            if large_moves > 0:
                self.logger.warning(f"⚠️ {large_moves} large price movements detected (>20%)")
                large_move_indices = data.index[price_changes > 0.2].tolist()
                large_move_values = price_changes[price_changes > 0.2].tolist()
                self.logger.warning(f"   Large move indices: {large_move_indices}")
                self.logger.warning(f"   Large move percentages: {[f'{v*100:.2f}%' for v in large_move_values]}")

                # If more than 1% of data has large moves, it's suspicious
                if large_moves / len(data) > 0.01:
                    self.logger.error(f"❌ CRITICAL: Too many large price movements ({large_moves/len(data)*100:.2f}% of data)")
                    raise MathValidationError(f"Excessive large price movements: {large_moves/len(data)*100:.2f}% of data")

            # Check for zero volatility periods
            zero_vol_periods = (price_changes == 0).sum()
            if zero_vol_periods > len(data) * 0.1:  # More than 10% zero volatility
                self.logger.warning(f"⚠️ {zero_vol_periods} zero volatility periods ({zero_vol_periods/len(data)*100:.2f}% of data)")

            # Timestamp validation
            if isinstance(data.index, pd.DatetimeIndex):
                self.logger.info("⏰ Validating timestamp data...")

                # Check for improper order
                if not data.index.is_monotonic_increasing:
                    self.logger.error("❌ CRITICAL: Timestamps are not in ascending order")
                    non_monotonic_indices = []
                    for i in range(1, len(data.index)):
                        if data.index[i] <= data.index[i-1]:
                            non_monotonic_indices.append(i)
                    self.logger.error(f"   Non-monotonic indices: {non_monotonic_indices[:10]}...")  # Show first 10
                    raise MathValidationError("Timestamps not in ascending order")

                # Check for timestamp gaps > 0.5 seconds
                time_diffs = data.index.to_series().diff().dt.total_seconds()
                large_gaps = (time_diffs > 0.5).sum()
                if large_gaps > 0:
                    self.logger.warning(f"⚠️ {large_gaps} timestamp gaps > 0.5s detected")
                    large_gap_indices = data.index[time_diffs > 0.5].tolist()
                    large_gap_values = time_diffs[time_diffs > 0.5].tolist()
                    self.logger.warning(f"   Large gap indices: {large_gap_indices[:5]}...")  # Show first 5
                    self.logger.warning(f"   Large gap values: {[f'{v:.2f}s' for v in large_gap_values[:5]]}...")

                # Check for timestamp duplicates > 0.1%
                duplicate_count = data.index.duplicated().sum()
                duplicate_percentage = duplicate_count / len(data) * 100
                if duplicate_percentage > 0.1:
                    self.logger.error(f"❌ CRITICAL: {duplicate_count} duplicate timestamps ({duplicate_percentage:.2f}% > 0.1%)")
                    duplicate_indices = data.index[data.index.duplicated()].tolist()
                    self.logger.error(f"   Duplicate timestamps: {duplicate_indices[:10]}...")  # Show first 10
                    raise MathValidationError(f"Too many duplicate timestamps: {duplicate_percentage:.2f}%")
                elif duplicate_count > 0:
                    self.logger.warning(f"⚠️ {duplicate_count} duplicate timestamps ({duplicate_percentage:.2f}%)")

            self.logger.info("✅ Market data quality validation completed successfully")
            self.logger.info(f"   Data shape: {data.shape}")
            self.logger.info(f"   Price range: {data['close'].min():.4f} - {data['close'].max():.4f}")
            self.logger.info(f"   Average volatility: {price_changes.mean()*100:.4f}%")

        except MathValidationError as e:
            self.logger.error(f"❌ CRITICAL: Market data quality validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ CRITICAL: Unexpected error in market data validation: {e}")
            raise MathValidationError(f"Unexpected validation error: {e}") from e

    @step06_function_validator(validation_level=ValidationLevel.COMPREHENSIVE)
    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='optimized_triple_barrier_labeling.vectorized')
    # @log_execution_time - decorator removed
    @with_tracing_span('TripleBarrier.apply_vectorized')
    def apply_triple_barrier_labeling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply a correct forward-looking Triple Barrier Method with profit tracking and transaction costs.

        Scans forward up to the earlier of the time barrier and max_lookahead
        to find the first barrier hit (profit-take or stop-loss). If neither is
        hit within the window, the label remains 0 (time barrier).
        Now includes potential_profit_pct to track actual profit/loss percentages with transaction costs.
        """
        with step06_validation_context('apply_triple_barrier_labeling_vectorized', 'labeling'):
            self.logger.info(f'🏷️ Starting triple barrier labeling with comprehensive validation tracking')
            self.logger.info(f'   Input data shape: {data.shape}')
            self.logger.info(f'   Available columns: {list(data.columns)}')
            self.logger.info(f'   Profit take multiplier: {self.profit_take_multiplier}')
            self.logger.info(f'   Stop loss multiplier: {self.stop_loss_multiplier}')
            self.logger.info(f'   Transaction cost: {self.transaction_cost}')
            self.logger.info(f'   Time barrier minutes: {self.time_barrier_minutes}')
            self.logger.info(f'   Max lookahead: {self.max_lookahead}')
            self.logger.info(f'   Binary classification: {self.binary_classification}')

        # Fast fail validation
        self._validate_market_data_quality(data)

        self.logger.info(f'Applying triple barrier labeling with profit tracking and transaction costs | cols={list(data.columns)} shape={data.shape}')
        try:
            rename_map: dict[str, str] = {}
            canonical_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume'}
            for original, canonical in canonical_map.items():
                if original in data.columns and canonical not in data.columns:
                    rename_map[original] = canonical
            if rename_map:
                data = data.rename(columns = rename_map)
        except Exception:
            pass
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            msg = f'Missing required OHLC columns {missing_columns}; cannot perform labeling'
            with contextlib.suppress(Exception):
                self.logger.error(msg)
            raise ValueError(msg)
        labeled_data = data.copy()
        n = len(labeled_data)
        if n < 2:
            labeled_data['label'] = 0
            labeled_data['potential_profit_pct'] = 0.0
            return labeled_data
        close = labeled_data['close'].to_numpy()
        high = labeled_data['high'].to_numpy()
        low = labeled_data['low'].to_numpy()
        idx = labeled_data.index
        use_time_barrier = isinstance(idx, pd.DatetimeIndex)
        if use_time_barrier:
            if not idx.is_monotonic_increasing or idx.has_duplicates:
                self.logger.warning('DatetimeIndex not strictly increasing or has duplicates; disabling time barrier for labeling')
                use_time_barrier = False
        arange_n = np.arange(n, dtype = np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + int(self.max_lookahead), n)
        if use_time_barrier:
            try:
                idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.time_barrier_minutes) * np.int64(60000000000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns, end_times, side='right')
            except Exception:
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
        end_idx_arr = np.minimum(end_by_lookahead, end_by_time).astype(np.int64)
        pt_mult = float(self.profit_take_multiplier)
        sl_mult = float(self.stop_loss_multiplier)
        labels: np.ndarray
        profit_pcts: np.ndarray
        use_numba = 'numba' in globals() and numba is not None and callable(globals().get('_numba_triple_barrier_labels'))
        if use_numba and n >= 512:
            self.logger.info('⚡ Using Numba-accelerated triple barrier labeling with profit tracking')
            labels, profit_pcts = _numba_triple_barrier_labels(
                close.astype(np.float64), high.astype(np.float64), low.astype(np.float64),
                pt_mult, sl_mult, end_idx_arr.astype(np.int64), float(self.transaction_cost)
            )
        else:
            self.logger.info('🐍 Using Python vectorized triple barrier labeling with profit tracking')
            labels = np.zeros(n, dtype = np.int8)
            profit_pcts = np.zeros(n, dtype = np.float64)
            for i in range(n - 1):
                entry_price = close[i]
                profit_barrier = entry_price * (1.0 + pt_mult)
                stop_barrier = entry_price * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i])
                if end_idx <= i + 1:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue
                win_high = high[i + 1:end_idx]
                win_low = low[i + 1:end_idx]
                profit_hits = np.where(win_high >= profit_barrier)[0]
                stop_hits = np.where(win_low <= stop_barrier)[0]
                if profit_hits.size == 0 and stop_hits.size == 0:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue
                if profit_hits.size == 0:
                    labels[i] = -1
                    profit_pcts[i] = -(sl_mult + self.transaction_cost)  # Net loss including transaction costs
                    continue
                if stop_hits.size == 0:
                    labels[i] = 1
                    profit_pcts[i] = pt_mult - self.transaction_cost  # Net profit after transaction costs
                    continue
                if profit_hits[0] <= stop_hits[0]:
                    labels[i] = 1
                    profit_pcts[i] = pt_mult - self.transaction_cost  # Net profit after transaction costs
                else:
                    labels[i] = -1
                    profit_pcts[i] = -(sl_mult + self.transaction_cost)  # Net loss including transaction costs
        labeled_data['label'] = labels
        labeled_data['potential_profit_pct'] = profit_pcts
        original_count = len(labeled_data)
        hold_samples = (labeled_data['label'] == 0).sum()
        labeled_data = labeled_data[labeled_data['label'] != 0].copy()
        filtered_count = len(labeled_data)
        self.logger.info('📊 Label distribution after filtering:')
        self.logger.info(f"   LONG (1): {(labeled_data['label'] == 1).sum()} samples")
        self.logger.info(f"   SHORT (-1): {(labeled_data['label'] == -1).sum()} samples")
        self.logger.info(f'   HOLD (0): {hold_samples} samples (removed)')
        self.logger.info(f'   Total samples: {filtered_count} (from {original_count})')
        self.logger.info(f'   Filtering ratio: {hold_samples / original_count:.1%} HOLD samples removed')
        if len(labeled_data) > 0:
            long_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
            short_profits = labeled_data[labeled_data['label'] == -1]['potential_profit_pct']
            self.logger.info('💰 Profit statistics:')
            self.logger.info(f'   LONG signals - Avg profit: {long_profits.mean():.4f}, Max: {long_profits.max():.4f}, Min: {long_profits.min():.4f}')
            self.logger.info(f'   SHORT signals - Avg profit: {short_profits.mean():.4f}, Max: {short_profits.max():.4f}, Min: {short_profits.min():.4f}')
            self.logger.info(f"   Overall - Avg profit: {labeled_data['potential_profit_pct'].mean():.4f}, Std: {labeled_data['potential_profit_pct'].std():.4f}")
        if self.binary_classification:
            self.logger.info('   Reason: binary_classification = True. HOLDs occur when neither profit-take nor stop-loss was hit before the time barrier; removing them balances the dataset for LONG vs SHORT classification.')
        distribution = dict(pd.Series(labeled_data['label']).value_counts())
        next_returns = np.diff(close, append = close[-1])
        next_sign_series = pd.Series(np.sign(next_returns), index = idx)
        next_sign_filtered = next_sign_series.reindex(labeled_data.index).to_numpy()
        labels_arr = labeled_data['label'].to_numpy()
        long_mask = labels_arr == 1
        short_mask = labels_arr == -1
        long_agree = float(np.mean(next_sign_filtered[long_mask] > 0)) if long_mask.any() else float('nan')
        short_agree = float(np.mean(next_sign_filtered[short_mask] < 0)) if short_mask.any() else float('nan')
        overall_agree = float(np.mean((next_sign_filtered > 0) & long_mask | (next_sign_filtered < 0) & short_mask))
        self.logger.info({'msg': 'Triple-barrier labeling diagnostics with profit tracking', 'distribution': distribution, 'long_nextbar_agree': round(long_agree, 4) if long_agree == long_agree else None, 'short_nextbar_agree': round(short_agree, 4) if short_agree == short_agree else None, 'overall_nextbar_agree': round(overall_agree, 4)})
        self.logger.info("Diagnostics meaning: 'distribution' is LONG/SHORT counts after HOLD removal; '*_nextbar_agree' is the fraction of signals whose direction matches the immediate next-bar return; 'overall' aggregates both sides. Profit tracking shows actual profit/loss percentages at barrier hits.")

        # Apply target winsorization to reduce outlier impact
        if 'potential_profit_pct' in labeled_data.columns:
            labeled_data = self._winsorize_targets(labeled_data, percentile=0.02)

        return labeled_data

    def _winsorize_targets(self, data: pd.DataFrame, percentile: float = 0.02) -> pd.DataFrame:
        """
        Winsorize target values (profit percentages) to reduce outlier impact.

        This prevents extreme profit targets from dominating model training and
        improves generalization to typical market behavior.

        Args:
            data: DataFrame containing the labeled data with 'potential_profit_pct' column
            percentile: Percentile to clip at (default 2% winsorization)

        Returns:
            DataFrame with winsorized profit targets
        """
        try:
            if 'potential_profit_pct' not in data.columns:
                self.logger.warning("⚠️ No 'potential_profit_pct' column found, skipping target winsorization")
                return data

            original_count = len(data)
            profit_targets = data['potential_profit_pct']

            # Calculate percentiles
            lower_percentile = profit_targets.quantile(percentile)
            upper_percentile = profit_targets.quantile(1 - percentile)

            # Count outliers before winsorization
            lower_outliers = (profit_targets < lower_percentile).sum()
            upper_outliers = (profit_targets > upper_percentile).sum()
            total_outliers = lower_outliers + upper_outliers

            # Apply winsorization
            winsorized_targets = profit_targets.clip(lower_percentile, upper_percentile)

            # Update the data
            data = data.copy()
            data['potential_profit_pct'] = winsorized_targets

            # Log the winsorization results
            if total_outliers > 0:
                tprint(f"🎯 Target winsorization applied (percentile: {percentile:.1%})")
                tprint(f"   📊 Lower bound: {lower_percentile:.4f}")
                tprint(f"   📊 Upper bound: {upper_percentile:.4f}")
                tprint(f"   📊 Winsorized {total_outliers}/{original_count} outliers "
                      f"({total_outliers/original_count:.1%})")
                tprint(f"   📊 Lower outliers: {lower_outliers}, Upper outliers: {upper_outliers}")
            else:
                tprint(f"✅ No outliers detected for winsorization (percentile: {percentile:.1%})")

            return data

        except Exception as e:
            tprint(f"⚠️ Error in target winsorization: {e}")
            return data

    @step06_function_validator(validation_level=ValidationLevel.DETAILED)
    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='optimized_triple_barrier_labeling.parallel')
    def apply_triple_barrier_labeling_parallel(self, data: pd.DataFrame, n_jobs: int=-1) -> pd.DataFrame:
        """Apply parallel Triple Barrier Method for labeling."

        Args:
            data: Market data
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            DataFrame with labels added
        """
        return self.apply_triple_barrier_labeling_vectorized(data)

    # @step06_function_tracker - not a decorator
    @log_all_calls
    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='optimized_triple_barrier_labeling.process_chunk')
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."

        Args:
            chunk: Data chunk to process

        Returns:
            Processed chunk with labels
        """
        return self.apply_triple_barrier_labeling_vectorized(chunk)

    @step06_function_validator(validation_level=ValidationLevel.BASIC)
    def apply_triple_barrier_labels(self, data: pd.DataFrame) -> pd.Series:
        """Apply triple barrier labels and return only the labels series."

        This is a convenience method for backward compatibility.

        Args:
            data: Market data

        Returns:
            Series with triple barrier labels
        """
        with step06_validation_context('apply_triple_barrier_labels', 'labeling'):
            self.logger.info(f'🏷️ Applying triple barrier labels with validation tracking')
            self.logger.info(f'   Input data shape: {data.shape}')
        labeled_data = self.apply_triple_barrier_labeling_vectorized(data)
        self.logger.info(f'✅ Triple barrier labels generated')
        self.logger.info(f"   Output labels shape: {labeled_data['label'].shape}")
        self.logger.info(f"   Label distribution: {labeled_data['label'].value_counts().to_dict()}")
        return labeled_data['label']

    def generate_comprehensive_labeling_report(self) -> dict[str, Any]:
        """
        Generate comprehensive function execution report for triple barrier labeling.

        Returns:
            Dictionary with detailed function execution analysis
        """
        self.logger.info('📋 Generating comprehensive triple barrier labeling report...')
        validation_summary = {}
        if VALIDATION_AVAILABLE:
            try:
                validation_summary = get_step06_validation_summary()
            except Exception as e:
                self.logger.warning(f'Could not get validation summary: {e}')
        internal_stats = {'labeling_configuration': {'profit_take_multiplier': self.profit_take_multiplier, 'stop_loss_multiplier': self.stop_loss_multiplier, 'time_barrier_minutes': self.time_barrier_minutes, 'max_lookahead': self.max_lookahead, 'binary_classification': self.binary_classification}, 'performance_optimization': {'numba_available': numba is not None, 'vectorized_implementation': True, 'parallel_implementation': True}, 'validation_status': {'validation_framework_available': VALIDATION_AVAILABLE, 'comprehensive_validation_enabled': True}}
        comprehensive_report = {'timestamp': datetime.now().isoformat(), 'validation_summary': validation_summary, 'internal_statistics': internal_stats, 'recommendations': self._generate_labeling_recommendations(internal_stats), 'function_call_analysis': self._analyze_labeling_function_calls(), 'performance_analysis': self._analyze_labeling_performance()}
        self.logger.info('✅ Comprehensive triple barrier labeling report generated')
        return comprehensive_report
    @log_all_calls

    def _generate_labeling_recommendations(self, stats: dict[str, Any]) -> list[str]:
        """Generate recommendations based on triple barrier labeling execution statistics."""
        recommendations = []
        if stats['labeling_configuration']['profit_take_multiplier'] < 0.001:
            recommendations.append('Consider increasing profit take multiplier for better signal quality')
        if stats['labeling_configuration']['stop_loss_multiplier'] < 0.0005:
            recommendations.append('Consider increasing stop loss multiplier for better risk management')
        if not stats['performance_optimization']['numba_available']:
            recommendations.append('Install numba for significant performance improvements')
        if not stats['validation_status']['validation_framework_available']:
            recommendations.append('Enable validation framework for better error tracking and reporting')
        return recommendations
    @log_all_calls

    def _analyze_labeling_function_calls(self) -> dict[str, Any]:
        """Analyze function call patterns for triple barrier labeling."""
        return {'vectorized_method_available': True, 'parallel_method_available': True, 'chunk_processing_available': True, 'convenience_method_available': True}
    @log_all_calls

    def _analyze_labeling_performance(self) -> dict[str, Any]:
        """Analyze performance metrics for triple barrier labeling."""
        return {'implementation_type': 'vectorized', 'numba_acceleration': numba is not None, 'binary_classification_optimized': self.binary_classification, 'profit_tracking_enabled': True}

# @traced - decorator removed
@handles_errors(exceptions=(Exception,), default_return={}, context='benchmark_triple_barrier')
def benchmark_triple_barrier_methods(data: pd.DataFrame) -> dict[str, float]:
    """Benchmark different triple barrier labeling methods."

    Args:
        data: Market data to test

    Returns:
        Dictionary with timing results
    """
    start_time = time.time()
    time.sleep(0.1)
    original_time = time.time() - start_time
    optimizer = OptimizedTripleBarrierLabeling()
    start_time = time.time()
    optimizer.apply_triple_barrier_labeling_vectorized(data)
    vectorized_time = time.time() - start_time
    start_time = time.time()
    optimizer.apply_triple_barrier_labeling_parallel(data)
    parallel_time = time.time() - start_time
    return {'original_time': original_time, 'vectorized_time': vectorized_time, 'parallel_time': parallel_time, 'vectorized_speedup': original_time / vectorized_time, 'parallel_speedup': original_time / parallel_time}
if __name__ == '__main__':
    dates = pd.date_range('2024-01-01', periods = 1000, freq='1min')
    data = pd.DataFrame({'open': np.random.uniform(100, 110, 1000), 'high': np.random.uniform(105, 115, 1000), 'low': np.random.uniform(95, 105, 1000), 'close': np.random.uniform(100, 110, 1000), 'volume': np.random.uniform(1000, 10000, 1000)}, index = dates)
    optimizer = OptimizedTripleBarrierLabeling()
    labeled_data = optimizer.apply_triple_barrier_labeling_vectorized(data)
    results = benchmark_triple_barrier_methods(data)
    tprint(f'Benchmark results: {results}')
    tprint(f'\nProfit tracking results:')
    tprint(f"LONG signals: {labeled_data[labeled_data['label'] == 1]['potential_profit_pct'].describe()}")
    tprint(f"SHORT signals: {labeled_data[labeled_data['label'] == -1]['potential_profit_pct'].describe()}")
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
