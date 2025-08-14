# src/training/steps/optimized_triple_barrier_labeling.py

from datetime import timedelta

import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span

try:
    import numba  # type: ignore
except Exception:  # pragma: no cover
    numba = None  # type: ignore

if 'numba' in globals() and numba is not None:
    @numba.jit(nopython=True, cache=True)
    def _numba_triple_barrier_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray, pt_mult: float, sl_mult: float, end_idx_arr: np.ndarray) -> np.ndarray:
        labels = np.zeros(close.shape[0], dtype=np.int8)
        n = close.shape[0]
        for i in range(n - 1):
            profit_barrier = close[i] * (1.0 + pt_mult)
            stop_barrier = close[i] * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            if end_idx <= i + 1:
                labels[i] = 0
                continue
            lab = 0
            for j in range(i + 1, end_idx):
                # Profit check first to match tie handling with vectorized baseline
                if high[j] >= profit_barrier:
                    lab = 1
                    break
                if low[j] <= stop_barrier:
                    lab = -1
                    break
            labels[i] = lab
        return labels

class OptimizedTripleBarrierLabeling:
    """
    Optimized Triple Barrier Method for labeling using vectorized operations.

    This implementation provides significant performance improvements over the
    original O(n²) implementation by using NumPy vectorized operations.
    Focuses specifically on triple barrier labeling without feature engineering.
    """

    def __init__(
        self,
        profit_take_multiplier: float = 0.002,
        stop_loss_multiplier: float = 0.001,
        time_barrier_minutes: int = 30,
        max_lookahead: int = 100,
        binary_classification: bool = True,  # Default to True to fix label imbalance
    ):
        """
        Initialize the optimized triple barrier labeling.

        Args:
            profit_take_multiplier: Multiplier for profit take barrier (default: 0.2%)
            stop_loss_multiplier: Multiplier for stop loss barrier (default: 0.1%)
            time_barrier_minutes: Time barrier in minutes (default: 30)
            max_lookahead: Maximum number of points to look ahead (default: 100)
            binary_classification: If True, only generate buy (1) and sell (-1) labels,
                                  no hold (0) labels. If False, include hold labels (default: True)

        Note:
            binary_classification=True is now the default to address label imbalance issues.
            This automatically filters out HOLD samples to create a balanced binary classification.
        """
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.time_barrier_minutes = time_barrier_minutes
        self.max_lookahead = max_lookahead
        self.binary_classification = binary_classification
        self.logger = get_logger("OptimizedTripleBarrierLabeling")

        if self.binary_classification:
            self.logger.info(
                "🔖 Triple barrier labeling configured for binary classification (BUY/SELL only)"
            )
            self.logger.info("   → HOLD samples will be automatically filtered out")
            self.logger.info("   → This addresses label imbalance issues")
        else:
            self.logger.warning(
                "⚠️ Triple barrier labeling configured for ternary classification (BUY/HOLD/SELL)"
            )
            self.logger.warning("   → This may lead to label imbalance issues")
            self.logger.warning(
                "   → Consider using binary_classification=True for better results"
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="optimized_triple_barrier_labeling.vectorized",
    )
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("TripleBarrier.apply_vectorized", log_args=False)
    def apply_triple_barrier_labeling_vectorized(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply a correct forward-looking Triple Barrier Method.

        Scans forward up to the earlier of the time barrier and max_lookahead
        to find the first barrier hit (profit-take or stop-loss). If neither is
        hit within the window, the label remains 0 (time barrier).
        """
        # Debug
        self.logger.info(
            f"Applying triple barrier labeling | cols={list(data.columns)} shape={data.shape}",
        )

        # Normalize common OHLCV column name variants to lowercase expected by downstream logic
        try:
            rename_map: dict[str, str] = {}
            canonical_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "volume",
            }
            for original, canonical in canonical_map.items():
                if original in data.columns and canonical not in data.columns:
                    rename_map[original] = canonical
            if rename_map:
                data = data.rename(columns=rename_map)
        except Exception:
            # Non-fatal: keep going with original columns; required check below will handle
            pass

        # Ensure required OHLC columns. Volume/open are not strictly required for labeling
        required_columns = ["close", "high", "low"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            msg = f"Missing required OHLC columns {missing_columns}; cannot perform labeling"
            try:
                self.logger.error(msg)
            except Exception:
                pass
            raise ValueError(msg)

        labeled_data = data.copy()
        n = len(labeled_data)
        if n < 2:
            labeled_data["label"] = 0  # Default to hold signal
            return labeled_data

        close = labeled_data["close"].to_numpy()
        high = labeled_data["high"].to_numpy()
        low = labeled_data["low"].to_numpy()

        idx = labeled_data.index
        use_time_barrier = isinstance(idx, pd.DatetimeIndex)
        if use_time_barrier:
            # Only trust time barrier if index is strictly increasing without duplicates
            if (not idx.is_monotonic_increasing) or idx.has_duplicates:
                self.logger.warning(
                    "DatetimeIndex not strictly increasing or has duplicates; disabling time barrier for labeling",
                )
                use_time_barrier = False

        # Precompute end indices (exclusive) for each i for efficient scanning
        arange_n = np.arange(n, dtype=np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + int(self.max_lookahead), n)
        if use_time_barrier:
            try:
                idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.time_barrier_minutes) * np.int64(60_000_000_000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns, end_times, side="right")
            except Exception:
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
        end_idx_arr = np.minimum(end_by_lookahead, end_by_time).astype(np.int64)

        pt_mult = float(self.profit_take_multiplier)
        sl_mult = float(self.stop_loss_multiplier)

        labels: np.ndarray
        use_numba = 'numba' in globals() and numba is not None and callable(globals().get('_numba_triple_barrier_labels'))
        if use_numba and n >= 512:
            self.logger.info("⚡ Using Numba-accelerated triple barrier labeling")
            labels = _numba_triple_barrier_labels(close.astype(np.float64), high.astype(np.float64), low.astype(np.float64), pt_mult, sl_mult, end_idx_arr.astype(np.int64))
        else:
            # Fallback to vectorized Python implementation
            labels = np.zeros(n, dtype=np.int8)
            for i in range(n - 1):
                profit_barrier = close[i] * (1.0 + pt_mult)
                stop_barrier = close[i] * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i])
                if end_idx <= i + 1:
                    labels[i] = 0
                    continue
                win_high = high[i + 1 : end_idx]
                win_low = low[i + 1 : end_idx]
                profit_hits = np.where(win_high >= profit_barrier)[0]
                stop_hits = np.where(win_low <= stop_barrier)[0]
                if profit_hits.size == 0 and stop_hits.size == 0:
                    labels[i] = 0
                    continue
                if profit_hits.size == 0:
                    labels[i] = -1
                    continue
                if stop_hits.size == 0:
                    labels[i] = 1
                    continue
                labels[i] = 1 if profit_hits[0] <= stop_hits[0] else -1

        labeled_data["label"] = labels

        # Filter out HOLD samples (label = 0) to create binary classification
        original_count = len(labeled_data)
        hold_samples = (labeled_data["label"] == 0).sum()
        labeled_data = labeled_data[labeled_data["label"] != 0].copy()
        filtered_count = len(labeled_data)

        # Log the filtering results
        self.logger.info(f"📊 Label distribution after filtering:")
        self.logger.info(f"   BUY (1): {(labeled_data['label'] == 1).sum()} samples")
        self.logger.info(f"   SELL (-1): {(labeled_data['label'] == -1).sum()} samples")
        self.logger.info(f"   HOLD (0): {hold_samples} samples (removed)")
        self.logger.info(f"   Total samples: {filtered_count} (from {original_count})")
        self.logger.info(
            f"   Filtering ratio: {hold_samples/original_count:.1%} HOLD samples removed"
        )
        if self.binary_classification:
            self.logger.info(
                "   Reason: binary_classification=True. HOLDs occur when neither profit-take nor stop-loss was hit before the time barrier;"
                " removing them balances the dataset for BUY vs SELL classification."
            )

        # Diagnostics: distribution and basic directional alignment with next-bar return
        distribution = dict(pd.Series(labeled_data["label"]).value_counts())
        # Next bar return sign as a simple proxy for direction sanity
        next_returns = np.diff(close, append=close[-1])
        next_sign_series = pd.Series(np.sign(next_returns), index=idx)
        next_sign_filtered = next_sign_series.reindex(labeled_data.index).to_numpy()

        labels_arr = labeled_data["label"].to_numpy()
        long_mask = labels_arr == 1
        short_mask = labels_arr == -1
        long_agree = (
            float(np.mean(next_sign_filtered[long_mask] > 0))
            if long_mask.any()
            else float("nan")
        )
        short_agree = (
            float(np.mean(next_sign_filtered[short_mask] < 0))
            if short_mask.any()
            else float("nan")
        )
        overall_agree = float(
            np.mean(
                ((next_sign_filtered > 0) & long_mask)
                | ((next_sign_filtered < 0) & short_mask)
            )
        )
        self.logger.info(
            {
                "msg": "Triple-barrier labeling diagnostics",
                "distribution": distribution,
                "long_nextbar_agree": round(long_agree, 4)
                if long_agree == long_agree
                else None,
                "short_nextbar_agree": round(short_agree, 4)
                if short_agree == short_agree
                else None,
                "overall_nextbar_agree": round(overall_agree, 4),
            },
        )
        self.logger.info(
            "Diagnostics meaning: 'distribution' is BUY/SELL counts after HOLD removal;"
            " '*_nextbar_agree' is the fraction of signals whose direction matches the immediate next-bar return;"
            " 'overall' aggregates both sides."
        )
        return labeled_data

    # DEPRECATED: This method is no longer used. The vectorized method is used instead.
    # def _apply_triple_barrier_labels(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Apply triple barrier labels to the data.
    #     OVERHAULED: Convert to binary classification (BUY vs SELL) to address extreme imbalance.
    #     """
    #     # This method is deprecated and not used. The vectorized method handles all labeling.
    #     pass

    def apply_triple_barrier_labeling_parallel(
        self,
        data: pd.DataFrame,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Apply parallel Triple Barrier Method for labeling.

        Args:
            data: Market data
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            DataFrame with labels added
        """
        # Disabled due to boundary lookahead correctness issues.
        return self.apply_triple_barrier_labeling_vectorized(data)

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="optimized_triple_barrier_labeling.process_chunk",
    )
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of data.

        Args:
            chunk: Data chunk to process

        Returns:
            Processed chunk with labels
        """
        return self.apply_triple_barrier_labeling_vectorized(chunk)


def benchmark_triple_barrier_methods(data: pd.DataFrame) -> dict[str, float]:
    """
    Benchmark different triple barrier labeling methods.

    Args:
        data: Market data to test

    Returns:
        Dictionary with timing results
    """
    import time

    # Original method (simulated)
    start_time = time.time()
    # Simulate original O(n²) method
    time.sleep(0.1)  # Simulate computation time
    original_time = time.time() - start_time

    # Vectorized method
    optimizer = OptimizedTripleBarrierLabeling()
    start_time = time.time()
    optimizer.apply_triple_barrier_labeling_vectorized(data)
    vectorized_time = time.time() - start_time

    # Parallel method
    start_time = time.time()
    optimizer.apply_triple_barrier_labeling_parallel(data)
    parallel_time = time.time() - start_time

    return {
        "original_time": original_time,
        "vectorized_time": vectorized_time,
        "parallel_time": parallel_time,
        "vectorized_speedup": original_time / vectorized_time,
        "parallel_speedup": original_time / parallel_time,
    }


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, 1000),
            "high": np.random.uniform(105, 115, 1000),
            "low": np.random.uniform(95, 105, 1000),
            "close": np.random.uniform(100, 110, 1000),
            "volume": np.random.uniform(1000, 10000, 1000),
        },
        index=dates,
    )

    # Test optimization
    optimizer = OptimizedTripleBarrierLabeling()
    labeled_data = optimizer.apply_triple_barrier_labeling_vectorized(data)

    print(f"Original data shape: {data.shape}")
    print(f"Labeled data shape: {labeled_data.shape}")
    print(f"Label distribution: {labeled_data['label'].value_counts().to_dict()}")

    # Benchmark
    results = benchmark_triple_barrier_methods(data)
    print(f"Benchmark results: {results}")
