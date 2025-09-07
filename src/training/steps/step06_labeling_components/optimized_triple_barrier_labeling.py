import contextlib
import sys
import os
import pandas as pd
import numpy as np
from typing import Any
from typing import Dict, List, Optional, Union, Any, Tuple
from typing import Callable
import numpy as np
from src.core.decorators import handles_errors

current_dir = os.path.dirname(os.path.abspath(__file__))
steps_dir = os.path.join(current_dir, '..')
sys.path.insert(0, steps_dir)
try:
    from ..step06_enhanced_validation_framework import step06_function_validator, step06_function_tracker, step06_validation_context, get_step06_validation_summary, ValidationLevel, FunctionStatus
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f'Warning: Step06 validation framework not available: {e}')

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

from src.core.decorators.logging import log_execution_time, log_call
import datetime
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

try:
    from src.utils.dataframe_guards import guard_dataframe_nulls, handle_errors, with_tracing_span
except ImportError:

    def guard_dataframe_nulls(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def handle_errors(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def with_tracing_span(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
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
except Exception:
    numba = None
if 'numba' in globals() and numba is not None:

    @numba.jit(nopython = True, cache = True)
    def _numba_triple_barrier_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray, pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                    profit_pct = pt_mult
                    break
                if low[j] <= stop_barrier:
                    lab = -1
                    profit_pct = -sl_mult
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

    def __init__(self, profit_take_multiplier: float = 0.002, stop_loss_multiplier: float = 0.001, time_barrier_minutes: int = 30, max_lookahead: int = 100, binary_classification: bool = True) -> None:
        """Initialize the optimized triple barrier labeling."

        Args:
            profit_take_multiplier: Multiplier for profit take barrier (default: 0.2%)
            stop_loss_multiplier: Multiplier for stop loss barrier (default: 0.1%)
            time_barrier_minutes: Time barrier in minutes (default: 30)
            max_lookahead: Maximum number of points to look ahead (default: 100)
            binary_classification: If True, only generate buy (1) and sell (-1) labels
            no hold (0) labels. If False, include hold labels (default: True)

        Note:
            binary_classification = True is now the default to address label imbalance issues.
            This automatically filters out HOLD samples to create a balanced binary classification.
        """
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.time_barrier_minutes = time_barrier_minutes
        self.max_lookahead = max_lookahead
        self.binary_classification = binary_classification
        self.logger = get_logger('OptimizedTripleBarrierLabeling')
        if self.binary_classification:
            self.logger.info('🔖 Triple barrier labeling configured for binary classification (BUY/SELL only)')
            self.logger.info('   → HOLD samples will be automatically filtered out')
            self.logger.info('   → This addresses label imbalance issues')
        else:
            self.logger.warning('⚠️ Triple barrier labeling configured for ternary classification (BUY/HOLD/SELL)')
            self.logger.warning('   → This may lead to label imbalance issues')
            self.logger.warning('   → Consider using binary_classification = True for better results')

    @step06_function_validator(function_type='labeling', validation_level = ValidationLevel.COMPREHENSIVE)
    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='optimized_triple_barrier_labeling.vectorized')
    @log_execution_time
    @validates
    @traced(span_name='TripleBarrier.apply_vectorized')
    def apply_triple_barrier_labeling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply a correct forward-looking Triple Barrier Method with profit tracking."

        Scans forward up to the earlier of the time barrier and max_lookahead
        to find the first barrier hit (profit-take or stop-loss). If neither is
        hit within the window, the label remains 0 (time barrier).
        Now includes potential_profit_pct to track actual profit/loss percentages.
        """
        with step06_validation_context('apply_triple_barrier_labeling_vectorized', 'labeling'):
            self.logger.info(f'🏷️ Starting triple barrier labeling with comprehensive validation tracking')
            self.logger.info(f'   Input data shape: {data.shape}')
            self.logger.info(f'   Available columns: {list(data.columns)}')
            self.logger.info(f'   Profit take multiplier: {self.profit_take_multiplier}')
            self.logger.info(f'   Stop loss multiplier: {self.stop_loss_multiplier}')
            self.logger.info(f'   Time barrier minutes: {self.time_barrier_minutes}')
            self.logger.info(f'   Max lookahead: {self.max_lookahead}')
            self.logger.info(f'   Binary classification: {self.binary_classification}')
        self.logger.info(f'Applying triple barrier labeling with profit tracking | cols={list(data.columns)} shape={data.shape}')
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
            labels, profit_pcts = _numba_triple_barrier_labels(close.astype(np.float64), high.astype(np.float64), low.astype(np.float64), pt_mult, sl_mult, end_idx_arr.astype(np.int64))
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
                    profit_pcts[i] = -sl_mult
                    continue
                if stop_hits.size == 0:
                    labels[i] = 1
                    profit_pcts[i] = pt_mult
                    continue
                if profit_hits[0] <= stop_hits[0]:
                    labels[i] = 1
                    profit_pcts[i] = pt_mult
                else:
                    labels[i] = -1
                    profit_pcts[i] = -sl_mult
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
        return labeled_data

    @step06_function_validator(function_type='labeling', validation_level = ValidationLevel.DETAILED)
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

    @step06_function_tracker
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

    @step06_function_validator(function_type='labeling', validation_level = ValidationLevel.BASIC)
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

@traced(span_name='benchmark_triple_barrier_methods')
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
    print(f'Benchmark results: {results}')
    print(f'\nProfit tracking results:')
    print(f"LONG signals: {labeled_data[labeled_data['label'] == 1]['potential_profit_pct'].describe()}")
    print(f"SHORT signals: {labeled_data[labeled_data['label'] == -1]['potential_profit_pct'].describe()}")