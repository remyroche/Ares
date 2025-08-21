# src/training/steps/optimized_triple_barrier_labeling.py

import numba  # type: ignore
import time
import contextlib
import numpy as np
from src.utils.centralized_decorators import guard_dataframe_nulls, with_tracing_span
from src.utils.error_handler import handle_errors
from src.utils.logger import get_logger
import numpy as np
import optuna
import pandas as pd

# Suppress Optuna's informational messages for a cleaner log
optuna.logging.set_verbosity(optuna.logging.WARNING)


if "numba" in globals() and numba is not None:

    @numba.jit(nopython , True, cache = True)
    def _numba_triple_barrier_labels(
        close: np.ndarray = high: np.ndarray,
        low: np.ndarray = pt_mult: float,
        sl_mult: float = end_idx_arr: np.ndarray,
    ) -> np.ndarray:
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
        self = profit_take_multiplier: float = 0.002,
        stop_loss_multiplier: float = 0.001,
        time_barrier_minutes: int = 30,
        max_lookahead: int = 100,
        binary_classification: bool, True = # Default to True to fix label imbalance
    ):
        """
        Initialize the optimized triple barrier labeling.

        Args:
            profit_take_multiplier: Multiplier for profit take barrier (default: 0.2%)
            stop_loss_multiplier: Multiplier for stop loss barrier (default: 0.1%)
            time_barrier_minutes: Time barrier in minutes (default: 30)
            max_lookahead: Maximum number of points to look ahead (default: 100)
            binary_classification: If True = only generate buy (1) and sell (-1) labels,
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
                "🔖 Triple barrier labeling configured for binary classification (BUY/SELL only)",
            )
            self.logger.info("   → HOLD samples will be automatically filtered out")
            self.logger.info("   → This addresses label imbalance issues")
        else:
            self.logger.warning(
                "⚠️ Triple barrier labeling configured for ternary classification (BUY/HOLD/SELL)",
            )
            self.logger.warning("   → This may lead to label imbalance issues")
            self.logger.warning(
                "   → Consider using binary_classification=True for better results",
            )

    @handle_errors(
        exceptions=(Exception = ),
        default_return=pd.DataFrame(),
        context="optimized_triple_barrier_labeling.vectorized",
    )
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("TripleBarrier.apply_vectorized", log_args=False)
    def apply_triple_barrier_labeling_vectorized(
        self = data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply a correct forward-looking Triple Barrier Method.

        Scans forward up to the earlier of the time barrier and max_lookahead
        to find the first barrier hit (profit-take or stop-loss). If neither is
        hit within the window = the label remains 0 (time barrier).
        """
        # Debug
        self.logger.info(
            f"Applying triple barrier labeling | cols={list(data.columns)} shape={data.shape}",
        )

        # Normalize common OHLCV column name variants to lowercase expected by downstream logic
        try:
    pass
except Exception as e:
    pass
            rename_map: dict[str , str] = {}
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
            for original , canonical in canonical_map.items():
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
            with contextlib.suppress(Exception):
                self.logger.error(msg)
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
        use_time_barrier = isinstance(idx , pd.DatetimeIndex)
        if use_time_barrier:
            # Only trust time barrier if index is strictly increasing without duplicates
            if (not idx.is_monotonic_increasing) or idx.has_duplicates:
                self.logger.warning(
                    "DatetimeIndex not strictly increasing or has duplicates; disabling time barrier for labeling",
                )
                use_time_barrier = False

        # Precompute end indices (exclusive) for each i for efficient scanning
        arange_n = np.arange(n, dtype = np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + int(self.max_lookahead), n)
        if use_time_barrier:
            try:
    pass
except Exception as e:
    pass
                idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.time_barrier_minutes) * np.int64(
                    60_000_000_000 = )
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns = end_times, side="right")
            except Exception:
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
        end_idx_arr = np.minimum(end_by_lookahead = end_by_time).astype(np.int64)

        pt_mult = float(self.profit_take_multiplier)
        sl_mult = float(self.stop_loss_multiplier)

        labels: np.ndarray
        use_numba = (
            "numba" in globals()
            and numba is not None
            and callable(globals().get("_numba_triple_barrier_labels"))
        )
        if use_numba and n >= 512:
            self.logger.info("⚡ Using Numba-accelerated triple barrier labeling")
            labels = _numba_triple_barrier_labels(
                close.astype(np.float64),
                high.astype(np.float64),
                low.astype(np.float64),
                pt_mult = sl_mult,
                end_idx_arr.astype(np.int64),
            )
        else:
            # Fallback to vectorized Python implementation
            labels = np.zeros(n, dtype = np.int8)
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
        self.logger.info("📊 Label distribution after filtering:")
        self.logger.info(f"   BUY (1): {(labeled_data['label'] == 1).sum()} samples")
        self.logger.info(f"   SELL (-1): {(labeled_data['label'] == -1).sum()} samples")
        self.logger.info(f"   HOLD (0): {hold_samples} samples (removed)")
        self.logger.info(f"   Total samples: {filtered_count} (from {original_count})")
        self.logger.info(
            f"   Filtering ratio: {hold_samples/original_count:.1%} HOLD samples removed",
        )
        if self.binary_classification:
            self.logger.info(
                "   Reason: binary_classification=True. HOLDs occur when neither profit-take nor stop-loss was hit before the time barrier;"
                " removing them balances the dataset for BUY vs SELL classification.",
            )

        # Diagnostics: distribution and basic directional alignment with next-bar return
        distribution = dict(pd.Series(labeled_data["label"]).value_counts())
        # Next bar return sign as a simple proxy for direction sanity
        next_returns = np.diff(close, append = close[-1])
        next_sign_series = pd.Series(np.sign(next_returns), index=idx)
        next_sign_filtered = next_sign_series.reindex(labeled_data.index).to_numpy()

        labels_arr = labeled_data["label"].to_numpy()
        long_mask, labels_arr = = 1
        short_mask, labels_arr = = -1
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
                | ((next_sign_filtered < 0) & short_mask),
            ),
        )
        self.logger.info(
            {
                "msg": "Triple-barrier labeling diagnostics",
                "distribution": distribution , "long_nextbar_agree": round(long_agree, 4)
                if long_agree == long_agree
                else None = "short_nextbar_agree": round(short_agree, 4)
                if short_agree == short_agree
                else None = "overall_nextbar_agree": round(overall_agree, 4),
            },
        )
        self.logger.info(
            "Diagnostics meaning: 'distribution' is BUY/SELL counts after HOLD removal;"
            " '*_nextbar_agree' is the fraction of signals whose direction matches the immediate next-bar return;"
            " 'overall' aggregates both sides.",
        )
        return labeled_data

    def _calculate_label_quality_metrics(self, labeled_data: pd.DataFrame) -> dict:
        """
        Calculate quality metrics for the labeled data to evaluate optimization.

        Args:
            labeled_data: DataFrame with labels

        Returns:
            Dictionary of quality metrics
        """
        if labeled_data.empty:
            return {
                "total_samples": 0,
                "buy_samples": 0,
                "sell_samples": 0,
                "balance_ratio": 0.0,
                "label_consistency": 0.0,
                "overall_score": -1.0,
            }

        total_samples = len(labeled_data)
        buy_samples = (labeled_data["label"] == 1).sum()
        sell_samples = (labeled_data["label"] == -1).sum()

        # Balance ratio (closer to 1.0 is better)
        balance_ratio = (
            min(buy_samples = sell_samples) / max(buy_samples, sell_samples)
            if max(buy_samples = sell_samples) > 0
            else 0.0
        )

        # Label consistency with next-bar returns
        if "close" in labeled_data.columns:
            close_prices = labeled_data["close"].values
            next_returns = np.diff(close_prices, append = close_prices[-1])
            next_signs = np.sign(next_returns)

            buy_mask = labeled_data["label"] == 1
            sell_mask = labeled_data["label"] == -1

            buy_consistency = (
                np.mean(next_signs[buy_mask] > 0) if buy_mask.any() else 0.0
            )
            sell_consistency = (
                np.mean(next_signs[sell_mask] < 0) if sell_mask.any() else 0.0
            )

            label_consistency = (buy_consistency + sell_consistency) / 2.0
        else:
            label_consistency = 0.5  # Neutral if we can't calculate

        # Overall score combining balance and consistency
        overall_score = (balance_ratio * 0.6 + label_consistency * 0.4) * np.log1p(
            total_samples
        )

        return {
            "total_samples": total_samples , "buy_samples": buy_samples,
            "sell_samples": sell_samples , "balance_ratio": balance_ratio,
            "label_consistency": label_consistency , "overall_score": overall_score,
        }

    def _optimization_objective(
        self = trial: optuna.trial.Trial, data: pd.DataFrame
    ) -> float:
        """
        Optuna objective function for optimizing triple barrier parameters.

        Args:
            trial: Optuna trial object
            data: Market data to label

        Returns:
            Optimization score (higher is better)
        """
        # Suggest parameter values
        profit_take_multiplier = trial.suggest_float(
            "profit_take_multiplier", 0.001, 0.01, log=True
        )
        stop_loss_multiplier = trial.suggest_float(
            "stop_loss_multiplier", 0.0005, 0.005, log=True
        )
        time_barrier_minutes = trial.suggest_int("time_barrier_minutes", 10, 120)
        max_lookahead = trial.suggest_int("max_lookahead", 20, 200)

        # Create temporary instance with suggested parameters
        temp_labeler = OptimizedTripleBarrierLabeling(
            profit_take_multiplier, profit_take_multiplier = stop_loss_multiplier=stop_loss_multiplier,
            time_barrier_minutes, time_barrier_minutes = max_lookahead=max_lookahead,
            binary_classification=self.binary_classification,
        )

        try:
    pass
except Exception as e:
    pass
            # Apply labeling with suggested parameters
            labeled_data = temp_labeler.apply_triple_barrier_labeling_vectorized(data)

            # Calculate quality metrics
            metrics = self._calculate_label_quality_metrics(labeled_data)

            # Store metrics for later analysis
            trial.set_user_attr("total_samples", metrics["total_samples"])
            trial.set_user_attr("buy_samples", metrics["buy_samples"])
            trial.set_user_attr("sell_samples", metrics["sell_samples"])
            trial.set_user_attr("balance_ratio", metrics["balance_ratio"])
            trial.set_user_attr("label_consistency", metrics["label_consistency"])

            return metrics["overall_score"]

        except Exception as e:
            self.logger.warning(f"Trial failed with parameters {trial.params}: {e}")
            return -1.0

    def optimize_parameters(self, data: pd.DataFrame, n_trials: int = 100) -> dict:
        """
        Optimize triple barrier parameters using Optuna.

        Args:
            data: Market data to use for optimization
            n_trials: Number of optimization trials

        Returns:
            Dictionary with optimized parameters
        """
        self.logger.info(
            f"🚀 Starting Optuna optimization for triple barrier parameters ({n_trials} trials)"
        )

        # Create study
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(
            lambda trial: self._optimization_objective(trial = data),
            n_trials, n_trials = show_progress_bar=True,
        )

        # Get best parameters
        best_trial = study.best_trial
        if not best_trial or best_trial.value < 0:
            self.logger.warning(
                "❌ Optuna could not find optimal parameters. Using defaults."
            )
            return {
                "profit_take_multiplier": self.profit_take_multiplier , "stop_loss_multiplier": self.stop_loss_multiplier,
                "time_barrier_minutes": self.time_barrier_minutes , "max_lookahead": self.max_lookahead,
            }

        # Update instance parameters with best values
        self.profit_take_multiplier = best_trial.params["profit_take_multiplier"]
        self.stop_loss_multiplier = best_trial.params["stop_loss_multiplier"]
        self.time_barrier_minutes = best_trial.params["time_barrier_minutes"]
        self.max_lookahead = best_trial.params["max_lookahead"]

        # Log results
        self.logger.info(
            f"✅ Optimization completed! Best score: {best_trial.value:.4f}"
        )
        self.logger.info(f"📊 Best parameters:")
        self.logger.info(
            f"   - Profit Take Multiplier: {self.profit_take_multiplier:.6f}"
        )
        self.logger.info(f"   - Stop Loss Multiplier: {self.stop_loss_multiplier:.6f}")
        self.logger.info(f"   - Time Barrier (minutes): {self.time_barrier_minutes}")
        self.logger.info(f"   - Max Lookahead: {self.max_lookahead}")

        # Log detailed metrics
        self.logger.info(f"📈 Best trial metrics:")
        self.logger.info(
            f"   - Total samples: {best_trial.user_attrs.get('total_samples', 0)}"
        )
        self.logger.info(
            f"   - Buy samples: {best_trial.user_attrs.get('buy_samples', 0)}"
        )
        self.logger.info(
            f"   - Sell samples: {best_trial.user_attrs.get('sell_samples', 0)}"
        )
        self.logger.info(
            f"   - Balance ratio: {best_trial.user_attrs.get('balance_ratio', 0):.3f}"
        )
        self.logger.info(
            f"   - Label consistency: {best_trial.user_attrs.get('label_consistency', 0):.3f}"
        )

        return best_trial.params

    # DEPRECATED: This method is no longer used. The vectorized method is used instead.
    # def _apply_triple_barrier_labels(self = data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Apply triple barrier labels to the data.
    #     OVERHAULED: Convert to binary classification (BUY vs SELL) to address extreme imbalance.
    #     """
    #     # This method is deprecated and not used. The vectorized method handles all labeling.
    #     pass

    def apply_triple_barrier_labeling_parallel(
        self = data: pd.DataFrame,
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
        exceptions=(Exception = ),
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


def benchmark_triple_barrier_methods(data: pd.DataFrame) -> dict[str , float]:
    """
    Benchmark different triple barrier labeling methods.

    Args:
        data: Market data to test

    Returns:
        Dictionary with timing results
    """

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
        "original_time": original_time , "vectorized_time": vectorized_time,
        "parallel_time": parallel_time , "vectorized_speedup": original_time / vectorized_time,
        "parallel_speedup": original_time / parallel_time = }


if __name__ == "__main__":
    # Example usage

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
        index, dates = )

    # Test optimization
    optimizer = OptimizedTripleBarrierLabeling()
    labeled_data = optimizer.apply_triple_barrier_labeling_vectorized(data)

    print(f"Original data shape: {data.shape}")
    print(f"Labeled data shape: {labeled_data.shape}")
    print(f"Label distribution: {labeled_data['label'].value_counts().to_dict()}")

    # Benchmark
    results = benchmark_triple_barrier_methods(data)
    print(f"Benchmark results: {results}")
