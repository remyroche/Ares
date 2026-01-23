"""
Causal Surprise Events Module

Implements causal surprise event detection based on specialist prediction errors
and structural breaks in causal relationships.

Key Features:
1. Causal surprise detection from specialist prediction errors
2. Structural break detection in causal relationships
3. Event generation and scoring
4. Integration with existing event systems
"""

import hashlib
import numpy as np
import pandas as pd
from src.utils.numba_funcs import _numba_rolling_mad, _numba_rolling_entropy
from typing import Dict, Optional, Any
from scipy import stats
from .detection_utils import detect_rolling_quantile_surprises

# Import tprint functions
try:
    from src.utils.tprint import (
        tprint_info,
        tprint_success,
        tprint_warning,
        tprint_error,
    )
except ImportError:
    # Fallback print functions
    def tprint_info(msg):
        print(f"[INFO] {msg}")

    def tprint_success(msg):
        print(f"[SUCCESS] {msg}")

    def tprint_warning(msg):
        print(f"[WARNING] {msg}")

    def tprint_error(msg):
        print(f"[ERROR] {msg}")


def _hash_pandas(obj: Any, sample_size: int = 1000) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                return "empty"
            sample = obj.head(sample_size)
            hashed = pd.util.hash_pandas_object(sample, index=True).values
        elif isinstance(obj, pd.Series):
            if obj.empty:
                return "empty"
            sample = obj.head(sample_size)
            hashed = pd.util.hash_pandas_object(sample, index=True).values
        else:
            return "na"
        return hashlib.md5(hashed.tobytes()).hexdigest()[:8]
    except Exception:
        return "na"


def _summarize_obj(obj: Any) -> str:
    if isinstance(obj, pd.DataFrame):
        idx_min = obj.index.min() if not obj.empty else None
        idx_max = obj.index.max() if not obj.empty else None
        return (
            f"DataFrame(shape={obj.shape}, cols={len(obj.columns)}, "
            f"range={idx_min}->{idx_max}, hash={_hash_pandas(obj)})"
        )
    if isinstance(obj, pd.Series):
        idx_min = obj.index.min() if not obj.empty else None
        idx_max = obj.index.max() if not obj.empty else None
        return (
            f"Series(len={len(obj)}, name={obj.name}, "
            f"range={idx_min}->{idx_max}, hash={_hash_pandas(obj)})"
        )
    if isinstance(obj, np.ndarray):
        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    if isinstance(obj, dict):
        return f"dict(len={len(obj)})"
    return f"{type(obj).__name__}({obj})"


class CausalSurpriseDetector:
    """
    Detects causal surprise events based on specialist prediction errors.

    Causal surprise occurs when specialist models make large prediction errors,
    indicating potential mechanism breaks or regime changes.
    """

    def __init__(
        self,
        surprise_threshold: float = 1.2,
        rolling_window: int = 100,
        min_specialists: int = 2,
        structural_break_window: int = 50,
        verbose: bool = True,
        zone_score_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Causal Surprise Detector.

        Args:
            surprise_threshold: Z-score threshold for surprise detection (default 1.8)
            rolling_window: Window for rolling statistics (Adaptive Volatility Filter, default 500)
            min_specialists: Minimum number of specialists required
            structural_break_window: Window for structural break detection
            verbose: Whether to print progress information
        """
        self.surprise_threshold = surprise_threshold
        self.rolling_window = rolling_window
        self.min_specialists = min_specialists
        self.structural_break_window = structural_break_window
        self.verbose = verbose

        self.specialist_predictions_ = {}
        self.specialist_errors_ = {}
        self.specialist_metadata_ = {}
        self.surprise_events_ = {}
        self.structural_breaks_ = {}
        self.surprise_density_ = 0.0

        # Zone score configuration and storage
        self.zone_score_config = zone_score_config or {}
        self.zone_score_power = max(
            1.0, float(self.zone_score_config.get("power", 2.0))
        )
        self.zone_score_cap = float(self.zone_score_config.get("cap", 0.99))
        self.zone3_floor = float(self.zone_score_config.get("zone3_floor", 0.85))
        self.zone3_ratio_boost = float(
            self.zone_score_config.get("zone3_ratio_boost", 0.5)
        )
        self.zone2_ratio_boost = float(
            self.zone_score_config.get("zone2_ratio_boost", 0.2)
        )

        # Core vs conditional specialist contract
        self.core_min_share_multiplier = float(
            self.zone_score_config.get("core_min_share_multiplier", 0.5)
        )
        self.core_max_dispersion_multiplier = float(
            self.zone_score_config.get("core_max_dispersion_multiplier", 1.0)
        )
        self.conditional_min_share_multiplier = float(
            self.zone_score_config.get("conditional_min_share_multiplier", 2.0)
        )
        self.conditional_weight_cap = float(
            self.zone_score_config.get("conditional_weight_cap", 0.5)
        )
        self.conditional_weight_multiplier = float(
            self.zone_score_config.get("conditional_weight_multiplier", 0.7)
        )
        self.weak_weight_multiplier = float(
            self.zone_score_config.get("weak_weight_multiplier", 0.3)
        )

        self.specialist_zone_scores_: pd.DataFrame = pd.DataFrame()
        self.specialist_zone_levels_: pd.DataFrame = pd.DataFrame()
        self.surprise_aggregates_df_: pd.DataFrame = pd.DataFrame()
        self.specialist_reliability_: Dict[str, Dict[str, float]] = {}
        self.detector_reliability_: Dict[str, float] = {}
        self.specialist_role_classification_: Dict[str, str] = {}

    def _log_call(self, func_name: str, **kwargs: Any) -> None:
        if not self.verbose:
            return
        details = ", ".join(
            [f"{key}={_summarize_obj(value)}" for key, value in kwargs.items()]
        )
        suffix = f" | {details}" if details else ""
        tprint_info(f"▶️ {func_name}{suffix}")

    def _log_exit(self, func_name: str, result: Any = None, extra: str = "") -> None:
        if not self.verbose:
            return
        result_info = f" result={_summarize_obj(result)}" if result is not None else ""
        extra_info = f" {extra}" if extra else ""
        tprint_success(f"✅ {func_name} complete.{result_info}{extra_info}")

    def register_specialist(
        self, specialist_name: str, predictions: pd.Series, targets: pd.Series
    ) -> None:
        """
        Register a specialist model with predictions and targets.

        Args:
            specialist_name: Name of the specialist
            predictions: Specialist predictions
            targets: True targets
        """
        self._log_call(
            "register_specialist",
            specialist_name=specialist_name,
            predictions=predictions,
            targets=targets,
        )
        try:
            if len(predictions) != len(targets):
                raise ValueError("Predictions and targets must have same length")

            # Compute prediction errors
            errors = targets - predictions

            # Compute Global MAD for this specialist (Robust baseline)
            median_error = np.median(errors)
            global_mad = np.median(np.abs(errors - median_error))

            # Store specialist data
            self.specialist_predictions_[specialist_name] = predictions
            self.specialist_errors_[specialist_name] = errors
            self.specialist_metadata_[specialist_name] = {
                "global_mad": global_mad,
                "mean_error": errors.mean(),
                "std_error": errors.std(),
            }

            if self.verbose:
                tprint_info(f"📝 Registered specialist: {specialist_name}")
                tprint_info(f"   - Samples: {len(predictions)}")
                tprint_info(f"   - Global MAD: {global_mad:.6f}")
                tprint_info(f"   - Mean error: {errors.mean():.6f}")
                tprint_info(f"   - Error std: {errors.std():.6f}")

            self._log_exit(
                "register_specialist",
                result=self.specialist_metadata_.get(specialist_name, {}),
            )

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Failed to register specialist {specialist_name}: {e}")
            raise

    def adaptive_calibration(
        self, target_density: float, duration_days: float
    ) -> float:
        """
        Adjust surprise_threshold to target a specific event density.

        Args:
            target_density: Target number of events per day
            duration_days: Total duration of the dataset in days

        Returns:
            The newly calibrated threshold
        """
        self._log_call(
            "adaptive_calibration",
            target_density=target_density,
            duration_days=duration_days,
        )
        if self.surprise_events_ is None or len(self.surprise_events_) == 0:
            if self.verbose:
                tprint_warning(
                    "   ⚠️ Adaptive Calibration: No surprise events aggregated yet."
                )
            return self.surprise_threshold

        # Extract zone_scores if available
        if "zone_score" in self.surprise_events_.columns:
            scores = self.surprise_events_["zone_score"].values
        else:
            # Fallback to max_surprise if zone_score not yet aggregated
            scores = self.surprise_events_["max_surprise"].values

        target_count = max(1, int(target_density * duration_days))

        if len(scores) <= target_count:
            self.surprise_threshold = 0.01  # Very loose if we have very little data
        else:
            # Find threshold that gives target_count events
            threshold = np.partition(scores, -target_count)[-target_count]
            self.surprise_threshold = float(threshold)

        if self.verbose:
            tprint_info(
                f"🎯 Adaptive Calibration: Target {target_density:.2f} events/day ({target_count} total)"
            )
            tprint_info(f"   - New threshold: {self.surprise_threshold:.4f}")

        self._log_exit("adaptive_calibration", result=self.surprise_threshold)
        return self.surprise_threshold

    def compute_soft_surprise(
        self, errors: pd.DataFrame = None, q: float = 0.975
    ) -> pd.DataFrame:
        """
        Compute continuous surprise scores per specialist in (0,1) range
        using a logistic sigmoid mapping.
        """
        self._log_call("compute_soft_surprise", errors=errors, q=q)
        if errors is None:
            errors = self._build_error_frame()

        if errors.empty:
            result = pd.DataFrame()
            self._log_exit("compute_soft_surprise", result=result, extra="empty_errors")
            return result

        # Extract Global MADs for scaling
        mads = (
            pd.Series(
                {
                    k: v.get("global_mad", 1.0)
                    for k, v in self.specialist_metadata_.items()
                }
            )
            .reindex(errors.columns)
            .fillna(1.0)
        )

        # Normalize errors by global MAD
        norm_error = errors.abs().divide(mads, axis=1)

        # Calculate sigmoid parameters (alpha, mu) based on data distribution or fixed targets
        # User reference: mu = norm_error.quantile(q), alpha = 1.0 / sigma
        mu = norm_error.quantile(q)
        sigma = norm_error.sub(mu).std().replace(0, 1.0)
        alpha = 1.0 / sigma

        # Sigmoid mapping: 1 / (1 + exp(-alpha * (x - mu)))
        soft_surprise = 1 / (1 + np.exp(-alpha * (norm_error - mu)))
        soft_surprise = soft_surprise.clip(0.0, 1.0)
        self._log_exit("compute_soft_surprise", result=soft_surprise)
        return soft_surprise

    def compute_zone_score(
        self, soft_surprise: pd.DataFrame, chaos_emphasis: float = 2.0
    ) -> pd.Series:
        """
        Compute a global ZoneScore (0-1 gradient) using reliability-weighted
        aggregation of soft surprise scores.
        """
        self._log_call(
            "compute_zone_score",
            soft_surprise=soft_surprise,
            chaos_emphasis=chaos_emphasis,
        )
        if soft_surprise.empty:
            result = pd.Series(0.0, index=soft_surprise.index)
            self._log_exit("compute_zone_score", result=result, extra="empty_soft_surprise")
            return result

        # Extract reliability weights
        # If not computed yet, use uniform weights
        reliability = (
            pd.Series(
                {
                    k: v.get("composite_reliability", 1.0)
                    for k, v in self.specialist_reliability_.items()
                }
            )
            .reindex(soft_surprise.columns)
            .fillna(1.0)
        )

        weights = reliability / reliability.sum()

        # Power weighting for chaos emphasis
        weighted = soft_surprise.pow(chaos_emphasis).mul(weights, axis=1)
        zone_score = weighted.sum(axis=1)
        zone_score = zone_score.clip(0.0, 1.0)
        self._log_exit("compute_zone_score", result=zone_score)
        return zone_score

    def compute_specialist_surprise(
        self, specialist_name: str, method: str = "zscore"
    ) -> pd.Series:
        """
        Compute surprise scores for a specialist.

        Args:
            specialist_name: Name of the specialist
            method: Method for surprise computation ("zscore", "magnitude", "combined")

        Returns:
            Surprise scores
        """
        self._log_call(
            "compute_specialist_surprise",
            specialist_name=specialist_name,
            method=method,
        )
        try:
            if specialist_name not in self.specialist_errors_:
                raise ValueError(f"Specialist {specialist_name} not registered")

            errors = self.specialist_errors_[specialist_name]

            if method == "zscore" or method == "robust_zscore":
                # Robust Z-score based surprise using Median Absolute Deviation (MAD)
                # Surprise = |Y_actual - Y_specialist| / sigma_residual

                # Use MAD for robustness as requested
                def get_mad(x):
                    median = np.median(x)
                    return np.median(np.abs(x - median))

                rolling_median = errors.rolling(
                    self.rolling_window, min_periods=min(self.rolling_window, 20)
                ).median()
                rolling_mad = errors.rolling(
                    self.rolling_window, min_periods=min(self.rolling_window, 20)
                ).apply(get_mad)

                # Standardize: current error / Rolling MAD (robust sigma)
                # Apply floor of Global MAD + absolute unit floor (1e-6 for small scale metrics)
                global_mad = self.specialist_metadata_.get(specialist_name, {}).get(
                    "global_mad", 1e-6
                )
                sigma_floor = np.maximum(global_mad, 1e-6)
                surprise_scores = np.abs(errors - rolling_median) / (
                    np.maximum(rolling_mad, sigma_floor)
                )

            elif method == "magnitude":
                # Magnitude-based surprise
                rolling_mad = errors.rolling(
                    self.rolling_window, min_periods=min(self.rolling_window, 20)
                ).apply(lambda x: np.median(np.abs(x - np.median(x))))
                global_mad = self.specialist_metadata_.get(specialist_name, {}).get(
                    "global_mad", 1e-6
                )
                sigma_floor = np.maximum(global_mad, 1e-6)
                surprise_scores = np.abs(errors) / (
                    np.maximum(rolling_mad, sigma_floor)
                )

            elif method == "combined":
                # Combined robust z-score and magnitude
                rolling_median = errors.rolling(
                    self.rolling_window, min_periods=min(self.rolling_window, 20)
                ).median()
                rolling_mad = errors.rolling(
                    self.rolling_window, min_periods=min(self.rolling_window, 20)
                ).apply(lambda x: np.median(np.abs(x - np.median(x))))

                global_mad = self.specialist_metadata_.get(specialist_name, {}).get(
                    "global_mad", 1e-6
                )
                sigma_floor = np.maximum(global_mad, 1e-6)

                zscore_surprise = np.abs(errors - rolling_median) / (
                    np.maximum(rolling_mad, sigma_floor)
                )
                magnitude_surprise = np.abs(errors) / (
                    np.maximum(rolling_mad, sigma_floor)
                )
                surprise_scores = 0.6 * zscore_surprise + 0.4 * magnitude_surprise

            else:
                raise ValueError(f"Unknown surprise method: {method}")

            surprise_scores = surprise_scores.fillna(0)
            self._log_exit("compute_specialist_surprise", result=surprise_scores)
            return surprise_scores

        except Exception as e:
            if self.verbose:
                tprint_error(
                    f"❌ Surprise computation failed for {specialist_name}: {e}"
                )
            return pd.Series(0, index=self.specialist_errors_[specialist_name].index)

    def detect_structural_breaks(
        self, specialist_name: str, method: str = "chow"
    ) -> pd.Series:
        """
        Detect structural breaks in specialist prediction errors.

        Args:
            specialist_name: Name of the specialist
            method: Method for break detection ("chow", "cusum", "variance")

        Returns:
            Structural break indicators
        """
        self._log_call(
            "detect_structural_breaks",
            specialist_name=specialist_name,
            method=method,
        )
        try:
            if specialist_name not in self.specialist_errors_:
                raise ValueError(f"Specialist {specialist_name} not registered")

            errors = self.specialist_errors_[specialist_name].values
            n_samples = len(errors)
            break_indicators = np.zeros(n_samples)

            if method == "chow":
                # Simplified Chow test for structural breaks
                for i in range(
                    self.structural_break_window,
                    n_samples - self.structural_break_window,
                ):
                    # Split data at potential break point
                    errors_before = errors[i - self.structural_break_window : i]
                    errors_after = errors[i : i + self.structural_break_window]

                    if len(errors_before) > 10 and len(errors_after) > 10:
                        # Compare means and variances
                        mean_before, var_before = np.mean(errors_before), np.var(
                            errors_before
                        )
                        mean_after, var_after = np.mean(errors_after), np.var(
                            errors_after
                        )

                        # Simple break test statistic
                        mean_diff = abs(mean_before - mean_after)
                        var_ratio = max(var_before, var_after) / (
                            min(var_before, var_after) + 1e-8
                        )

                        # Break if significant difference
                        if mean_diff > 2 * np.sqrt(var_before) or var_ratio > 3:
                            break_indicators[i] = 1

            elif method == "cusum":
                # CUSUM-based break detection
                cumulative_errors = np.cumsum(errors - np.mean(errors))
                std_cumulative = np.std(cumulative_errors)

                # Break if cumulative error exceeds threshold
                threshold = 3 * std_cumulative
                break_indicators[np.abs(cumulative_errors) > threshold] = 1

            elif method == "variance":
                # Variance-based break detection
                rolling_var = (
                    pd.Series(errors).rolling(self.structural_break_window).var()
                )
                var_threshold = rolling_var.quantile(0.95)
                break_indicators[rolling_var > var_threshold] = 1

            break_series = pd.Series(
                break_indicators, index=self.specialist_errors_[specialist_name].index
            )
            self._log_exit("detect_structural_breaks", result=break_series)
            return break_series

        except Exception as e:
            if self.verbose:
                tprint_error(
                    f"❌ Structural break detection failed for {specialist_name}: {e}"
                )
            return pd.Series(0, index=self.specialist_errors_[specialist_name].index)

    def _compute_rolling_entropy(
        self, series: pd.Series, window: int = 100, bins: int = 10
    ) -> pd.Series:
        """Compute rolling Shannon entropy."""
        self._log_call(
            "_compute_rolling_entropy",
            series=series,
            window=window,
            bins=bins,
        )

        values = series.values.astype(np.float64)
        entropy_vals = _numba_rolling_entropy(values, window=window, bins=bins)
        # _numba_rolling_entropy uses log10, convert to natural log units
        entropy_vals = entropy_vals * np.log(10)
        result = pd.Series(entropy_vals, index=series.index).fillna(0)
        self._log_exit("_compute_rolling_entropy", result=result)
        return result

    def set_zone_score_weights(
        self,
        zone3_boost: float = 0.5,
        zone2_boost: float = 0.2,
        exposure_scalar: float = 1.0,
    ) -> None:
        """Update zone score emphasis parameters on the fly."""
        self._log_call(
            "set_zone_score_weights",
            zone3_boost=zone3_boost,
            zone2_boost=zone2_boost,
            exposure_scalar=exposure_scalar,
        )
        self.zone3_ratio_boost = float(zone3_boost)
        self.zone2_ratio_boost = float(zone2_boost)
        self.zone_score_exposure = float(exposure_scalar)
        if self.verbose:
            tprint_info(
                f"🎚️ Zone score weights updated: "
                f"zone3={self.zone3_ratio_boost:.2f}, "
                f"zone2={self.zone2_ratio_boost:.2f}, "
                f"exposure={self.zone_score_exposure:.2f}"
            )
        self._log_exit("set_zone_score_weights")

    def _weighted_spearmanr(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Compute posterior-weighted Spearman correlation."""
        self._log_call("_weighted_spearmanr", x=x, y=y, w=w)
        try:
            from scipy.stats import rankdata

            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
            if m.sum() < 20:
                return 0.0

            xr = rankdata(x[m])
            yr = rankdata(y[m])
            ww = w[m]

            # Weighted Pearson on ranks
            sw = np.sum(ww)
            xm = np.sum(ww * xr) / sw
            ym = np.sum(ww * yr) / sw

            cov = np.sum(ww * (xr - xm) * (yr - ym)) / sw
            vx = np.sum(ww * (xr - xm) ** 2) / sw
            vy = np.sum(ww * (yr - ym) ** 2) / sw

            denom = np.sqrt(vx) * np.sqrt(vy)
            result = 0.0 if denom == 0 else float(cov / denom)
            self._log_exit("_weighted_spearmanr", result=result)
            return result
        except Exception:
            result = 0.0
            self._log_exit("_weighted_spearmanr", result=result, extra="exception")
            return result

    def _weighted_lift(
        self, signal: np.ndarray, ret: np.ndarray, w: np.ndarray
    ) -> float:
        """Compute posterior-weighted Lift (Top decile return - Bottom decile return)."""
        self._log_call("_weighted_lift", signal=signal, ret=ret, w=w)
        try:
            m = np.isfinite(signal) & np.isfinite(ret) & np.isfinite(w) & (w > 0)
            if m.sum() < 20:
                result = 0.0
                self._log_exit("_weighted_lift", result=result, extra="insufficient_data")
                return result

            v = signal[m]
            r = ret[m]
            ww = w[m]

            # Sort by signal
            s_idx = np.argsort(v)
            r_s = r[s_idx]
            w_s = ww[s_idx]

            cum_w = np.cumsum(w_s) / np.sum(w_s)

            # Masks for top/bottom 10% mass
            mask_lo = cum_w <= 0.10
            mask_hi = cum_w >= 0.90

            if mask_lo.sum() == 0 or mask_hi.sum() == 0:
                result = 0.0
                self._log_exit("_weighted_lift", result=result, extra="empty_deciles")
                return result

            mean_lo = np.average(r_s[mask_lo], weights=w_s[mask_lo])
            mean_hi = np.average(r_s[mask_hi], weights=w_s[mask_hi])

            result = float(mean_hi - mean_lo)
            self._log_exit("_weighted_lift", result=result)
            return result
        except Exception:
            result = 0.0
            self._log_exit("_weighted_lift", result=result, extra="exception")
            return result

    def compute_regime_specific_reliability(
        self,
        regime_posteriors: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute specialist reliability scores per regime using weighted statistics (IC, Lift, Precision).
        Returns a DataFrame (Specialist x Regime) of reliability scores.
        Q(s, r) = softplus(10*IC) * softplus(5*Lift) * (Precision^0.25)
        """
        self._log_call(
            "compute_regime_specific_reliability",
            regime_posteriors=regime_posteriors,
            market_data=market_data,
        )
        if (
            not self.specialist_errors_
            or regime_posteriors is None
            or regime_posteriors.empty
        ):
            result = pd.DataFrame()
            self._log_exit(
                "compute_regime_specific_reliability",
                result=result,
                extra="empty_inputs",
            )
            return result

        errors_df = self._build_error_frame()
        # Ensure errors_df is float
        errors_df = errors_df.astype(float)

        common_idx = errors_df.index.intersection(regime_posteriors.index)

        if len(common_idx) < 50:
            result = pd.DataFrame()
            self._log_exit(
                "compute_regime_specific_reliability",
                result=result,
                extra="insufficient_overlap",
            )
            return result

        errors_aligned = errors_df.loc[common_idx]
        posteriors_aligned = regime_posteriors.loc[common_idx]
        errors_matrix = errors_aligned.values

        # Calculate forward returns if market data available
        fwd_returns = None
        if market_data is not None and "close" in market_data.columns:
            # Align market data first
            mkt_aligned = market_data.reindex(common_idx).ffill()
            # Use horizon consistent with typical events (e.g. 24 bars)
            fwd_returns = mkt_aligned["close"].pct_change(24).shift(-24).fillna(0.0)

        reliability = np.zeros(
            (errors_matrix.shape[1], posteriors_aligned.shape[1]), dtype=float
        )

        if fwd_returns is not None:
            ret_values = fwd_returns.values.astype(np.float64)
        else:
            ret_values = None

        from scipy.stats import rankdata

        for j, regime_col in enumerate(posteriors_aligned.columns):
            weights = posteriors_aligned[regime_col].values.astype(np.float64)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            weight_sum = float(np.sum(weights))

            if weight_sum < 1e-6:
                reliability[:, j] = 1.0
                continue

            weights = weights / weight_sum
            weighted_mean = (errors_matrix * weights[:, None]).sum(axis=0)
            weighted_second = ((errors_matrix ** 2) * weights[:, None]).sum(axis=0)
            weighted_var = weighted_second - weighted_mean ** 2
            precision = 1.0 / (weighted_var + 1e-4)

            ic = np.zeros(errors_matrix.shape[1], dtype=float)
            lift = np.zeros(errors_matrix.shape[1], dtype=float)

            if ret_values is not None:
                ranks_signal = rankdata(errors_matrix, axis=0)
                ranks_ret = rankdata(ret_values)
                w = weights[:, None]
                mean_signal = (w * ranks_signal).sum(axis=0)
                mean_ret = float((weights * ranks_ret).sum())
                cov = (w * (ranks_signal - mean_signal) * (ranks_ret - mean_ret)[:, None]).sum(axis=0)
                var_signal = (w * (ranks_signal - mean_signal) ** 2).sum(axis=0)
                var_ret = float((weights * (ranks_ret - mean_ret) ** 2).sum())
                denom = np.sqrt(var_signal * var_ret)
                ic = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)

                sort_idx = np.argsort(errors_matrix, axis=0)
                ret_sorted = np.take_along_axis(ret_values[:, None], sort_idx, axis=0)
                w_sorted = np.take_along_axis(weights[:, None], sort_idx, axis=0)
                cum_w = np.cumsum(w_sorted, axis=0)
                mask_lo = cum_w <= 0.10
                mask_hi = cum_w >= 0.90

                sum_lo = np.sum(w_sorted * ret_sorted * mask_lo, axis=0)
                w_lo = np.sum(w_sorted * mask_lo, axis=0)
                sum_hi = np.sum(w_sorted * ret_sorted * mask_hi, axis=0)
                w_hi = np.sum(w_sorted * mask_hi, axis=0)

                mean_lo = np.divide(sum_lo, w_lo, out=np.zeros_like(sum_lo), where=w_lo > 0)
                mean_hi = np.divide(sum_hi, w_hi, out=np.zeros_like(sum_hi), where=w_hi > 0)
                lift = mean_hi - mean_lo

            ic_val = np.clip(10.0 * ic, -20.0, 20.0)
            lift_val = np.clip(5.0 * lift, -20.0, 20.0)
            ic_score = np.log1p(np.exp(ic_val))
            lift_score = np.log1p(np.exp(lift_val))
            prec_score = precision ** 0.25

            reliability[:, j] = ic_score * lift_score * prec_score

        reliability_matrix = pd.DataFrame(
            reliability,
            index=errors_aligned.columns,
            columns=posteriors_aligned.columns,
        )

        # Normalize columns (sum to 1 per regime)
        col_sums = reliability_matrix.sum(axis=0).replace(0.0, np.nan)
        reliability_matrix = reliability_matrix.div(col_sums, axis=1)
        reliability_matrix = reliability_matrix.fillna(
            1.0 / len(errors_aligned.columns)
        )

        self._log_exit("compute_regime_specific_reliability", result=reliability_matrix)
        return reliability_matrix

    def aggregate_specialist_surprise(
        self,
        spectral_reliability: Optional[Dict[str, Dict[str, Any]]] = None,
        exposure_scalar: float = 1.0,
        regime_vol: Optional[pd.Series] = None,
        market_data: Optional[pd.DataFrame] = None,
        regime_posteriors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate surprise scores across all specialists.
        Uses regime-conditional weighting if regime_posteriors is provided.

        Args:
            regime_posteriors: DataFrame of regime posterior probabilities (soft regimes)
        """
        self._log_call(
            "aggregate_specialist_surprise",
            spectral_reliability=spectral_reliability or {},
            exposure_scalar=exposure_scalar,
            regime_vol=regime_vol,
            market_data=market_data,
            regime_posteriors=regime_posteriors,
        )
        try:
            if self.verbose:
                tprint_info(
                    "🔄 Aggregating Specialist Surprise (Continuous Framework)..."
                )

            if len(self.specialist_errors_) < self.min_specialists:
                raise ValueError(
                    f"Insufficient specialists for surprise aggregation: {len(self.specialist_errors_)} < {self.min_specialists}"
                )

            errors_df = self._build_error_frame()
            if errors_df.empty:
                raise ValueError("Empty error frame - no specialist data available")

            # Step 1: Compute Soft Surprises (Logistic Sigmoid Mapping)
            soft_surprise_df = self.compute_soft_surprise(errors_df)
            self.specialist_soft_surprise_ = soft_surprise_df

            # Step 2: Compute Discrete Metrics + NORMALIZATION
            surprise_df = self._compute_batch_surprises(
                errors_df, method="combined", market_data=market_data
            )
            self.specialist_surprises_ = surprise_df

            # Step 4: Integrate regime_vol if provided
            if regime_vol is not None and not regime_vol.empty:
                regime_vol_aligned = regime_vol.reindex(surprise_df.index).fillna(0.0)
                regime_weight = 1.0 + 0.5 * np.tanh(regime_vol_aligned)
                surprise_df = surprise_df.multiply(regime_weight, axis=0)
                if self.verbose:
                    tprint_info(
                        f"📊 Integrated regime volatility weighting (mean factor: {regime_weight.mean():.3f})"
                    )

            # Step 5: Specialist Weighting (Regime-Conditional vs Global)

            # 5a. Global Weights (Inverse-Variance + Reliability) - Fallback/Base
            inv_var_weights = {}
            for spec_name, meta in self.specialist_metadata_.items():
                std_err = meta.get("std_error", 1.0)
                inv_var_weights[spec_name] = 1.0 / (std_err**2 + 1e-4)

            inv_var_series = (
                pd.Series(inv_var_weights).reindex(surprise_df.columns).fillna(0.1)
            )
            inv_var_series /= inv_var_series.sum()

            # Global Reliability weights
            spectral_scores = pd.Series()
            if spectral_reliability:
                spectral_scores = pd.Series(
                    {
                        spec: float(metrics.get("composite_reliability", np.nan))
                        for spec, metrics in spectral_reliability.items()
                    }
                )
            detector_scores = pd.Series(
                {
                    spec: metrics.get("composite_reliability", np.nan)
                    for spec, metrics in (self.specialist_reliability_ or {}).items()
                }
            )
            reliability = pd.concat([spectral_scores, detector_scores], axis=1)
            if reliability.empty:
                reliability_weight = pd.Series(1.0, index=surprise_df.columns)
            else:
                reliability.columns = ["spectral", "detector"]
                reliability = reliability.reindex(surprise_df.columns).fillna(0.0)
                blended = 0.6 * reliability["spectral"] + 0.4 * reliability["detector"]
                reliability_weight = blended.fillna(
                    reliability[["spectral", "detector"]].max(axis=1)
                ).replace(0.0, 0.1)

            reliability_weight = reliability_weight.reindex(surprise_df.columns).fillna(
                0.1
            )

            # 5b. Regime-Conditional Weighting (The "Contract")
            weight_matrix = None
            regime_reliability = pd.DataFrame()

            if regime_posteriors is not None:
                # Compute Q(s, r) matrix (Passing market_data for IC/Lift)
                regime_reliability = self.compute_regime_specific_reliability(
                    regime_posteriors, market_data=market_data
                )

                if not regime_reliability.empty:
                    if self.verbose:
                        tprint_info(
                            "   🧬 Applying Regime-Conditional Specialist Weighting..."
                        )

                    # Align indices & Normalize Posteriors (prevent 0-mass rows)
                    aligned_posteriors = regime_posteriors.reindex(surprise_df.index)
                    aligned_posteriors = aligned_posteriors.fillna(
                        1.0 / aligned_posteriors.shape[1]
                    )
                    row_sums = aligned_posteriors.sum(axis=1).replace(0.0, np.nan)
                    aligned_posteriors = aligned_posteriors.div(
                        row_sums, axis=0
                    ).fillna(1.0 / aligned_posteriors.shape[1])

                    # Compute dynamic weights W(t, s) = sum_r( P(r|t) * Q(s, r) )
                    # Matrix multiplication: (T x R) @ (R x S) -> (T x S)
                    # regime_reliability is (S x R), so we use its transpose (R x S)
                    dynamic_weights = aligned_posteriors @ regime_reliability.T

                    # Blend with Global Reliability (Stability Anchor)
                    # W_final(t, s) = 0.7 * Dynamic(t, s) + 0.3 * Global(s)
                    # Broadcast global weights across time
                    global_weights_df = pd.DataFrame(
                        [reliability_weight.values] * len(surprise_df),
                        index=surprise_df.index,
                        columns=surprise_df.columns,
                    )

                    # Ensure columns match
                    dynamic_weights = dynamic_weights.reindex(
                        columns=surprise_df.columns
                    ).fillna(0.0)

                    # Combine
                    combined_weights = 0.7 * dynamic_weights + 0.3 * global_weights_df

                    # Normalize rows to sum to 1
                    weight_matrix = combined_weights.div(
                        combined_weights.sum(axis=1), axis=0
                    ).fillna(0.0)

            if weight_matrix is None:
                # Pure Global Fallback
                final_weights = inv_var_series * reliability_weight
                weight_series = final_weights / final_weights.sum()
                weight_matrix = pd.DataFrame(
                    [weight_series.values] * len(surprise_df),
                    index=surprise_df.index,
                    columns=surprise_df.columns,
                )

            # Step 5c: Core vs Conditional Contract (Regime Reliability)
            if not regime_reliability.empty:
                uniform_share = 1.0 / max(1, len(surprise_df.columns))
                core_min_share = self.core_min_share_multiplier * uniform_share
                core_max_dispersion = self.core_max_dispersion_multiplier * uniform_share
                conditional_min_share = (
                    self.conditional_min_share_multiplier * uniform_share
                )

                role_classification: Dict[str, str] = {}
                for spec in regime_reliability.index:
                    rel = regime_reliability.loc[spec]
                    min_share = float(rel.min())
                    max_share = float(rel.max())
                    dispersion = float(rel.std())
                    if min_share >= core_min_share and dispersion <= core_max_dispersion:
                        role = "core"
                    elif max_share >= conditional_min_share:
                        role = "conditional"
                    else:
                        role = "weak"
                    role_classification[spec] = role

                self.specialist_role_classification_ = role_classification
                if self.verbose and role_classification:
                    counts = (
                        pd.Series(role_classification).value_counts().to_dict()
                    )
                    tprint_info(
                        "   🧭 Specialist roles: "
                        f"{counts} (core_min={core_min_share:.4f}, "
                        f"cond_min={conditional_min_share:.4f}, "
                        f"core_dispersion_max={core_max_dispersion:.4f})"
                    )

                role_weights = pd.Series(1.0, index=surprise_df.columns)
                for spec, role in role_classification.items():
                    if role == "conditional":
                        role_weights.loc[spec] = self.conditional_weight_multiplier
                    elif role == "weak":
                        role_weights.loc[spec] = self.weak_weight_multiplier

                weight_matrix = weight_matrix.mul(role_weights, axis=1)

                conditional_cols = [
                    spec for spec, role in role_classification.items()
                    if role == "conditional"
                ]
                if conditional_cols:
                    conditional_sum = weight_matrix[conditional_cols].sum(axis=1)
                    cap = self.conditional_weight_cap
                    scale = (cap / conditional_sum).clip(upper=1.0).fillna(1.0)
                    weight_matrix.loc[:, conditional_cols] = (
                        weight_matrix.loc[:, conditional_cols].mul(scale, axis=0)
                    )
                    if self.verbose and (conditional_sum > cap).any():
                        tprint_info(
                            "   🧱 Conditional cap applied: "
                            f"cap={cap:.2f}, max_pre_cap={conditional_sum.max():.3f}"
                        )

                weight_matrix = weight_matrix.div(
                    weight_matrix.sum(axis=1), axis=0
                ).fillna(0.0)

            # Aggregation logic using time-varying weights
            aggregated = pd.DataFrame(index=surprise_df.index)

            # Raw Intensity (unweighted max) vs Weighted Aggregates
            aggregated["max_surprise"] = surprise_df.max(axis=1)  # Raw intensity

            # Weighted Mean/Consensus
            # Element-wise multiplication: Surprise(t, s) * Weight(t, s)
            weighted_surprise = surprise_df * weight_matrix
            aggregated["mean_surprise"] = weighted_surprise.sum(axis=1)

            # Weighted Consensus: sum_s( I(surprise > thresh) * weight(t, s) )
            is_surprised = (surprise_df > self.surprise_threshold).astype(float)
            weighted_consensus = (is_surprised * weight_matrix).sum(axis=1)
            # Scale consensus to be comparable to "number of specialists"
            # Since weights sum to 1, we multiply by N_spec to get an "effective count"
            n_specs = float(len(surprise_df.columns))
            aggregated["surprise_consensus"] = weighted_consensus * n_specs

            # Aggregate Signed Errors for Directionality
            signed_errors = errors_df.reindex(surprise_df.columns, axis=1).fillna(0)
            aggregated["mean_signed_error"] = (signed_errors * weight_matrix).sum(
                axis=1
            )

            # Simple break stats (unweighted for raw count)
            break_df = (surprise_df > self.surprise_threshold).astype(int)
            aggregated["total_breaks"] = break_df.sum(axis=1)
            aggregated["has_break"] = (aggregated["total_breaks"] > 0).astype(int)

            # Add Relative Severity (Max Weight across specialists)
            # This satisfies the requirement: "Downstream consumers... receive a weight metric"
            if hasattr(self, "zone_weights_") and not self.zone_weights_.empty:
                max_severity = self.zone_weights_.max(axis=1)
                aggregated["max_severity"] = max_severity.fillna(0.0)
            else:
                aggregated["max_severity"] = 0.0

            zone_levels = self._compute_specialist_zone_levels(surprise_df)
            zone_scores = self._compute_specialist_zone_scores(
                zone_levels, getattr(self, "zone_weights_", surprise_df)
            )
            combined_zone_score = self._compute_combined_zone_score(
                zone_scores, zone_levels
            )
            adjusted_zone_score = combined_zone_score.reindex(aggregated.index).fillna(
                0.0
            )
            adjusted_zone_score *= float(
                getattr(self, "zone_score_exposure", 1.0)
            ) * float(exposure_scalar)
            aggregated["zone_score"] = adjusted_zone_score.clip(
                0.0, self.zone_score_cap
            )

            # Surprise Density
            if len(aggregated) > 0:
                surprise_density = (aggregated["zone_score"] > 0.33).mean()
                self.surprise_density_ = surprise_density

            aggregated["causal_surprise"] = (aggregated["zone_score"] > 0.25).astype(
                int
            )

            # Discrete Zones
            aggregated["surprise_zone"] = (
                pd.cut(
                    aggregated["zone_score"],
                    bins=[-np.inf, 0.25, 0.66, np.inf],
                    labels=[1, 2, 3],
                )
                .astype(float)
                .fillna(1)
            )

            # --- REGIME-CONDITIONAL PROBABILITY CALIBRATION ---
            # Map consensus surprise to event probability using soft regime weights
            if regime_posteriors is not None:
                try:
                    aligned_posteriors = regime_posteriors.reindex(
                        aggregated.index
                    ).fillna(1.0 / regime_posteriors.shape[1])
                    base_prob = self._calibrate_event_probability(
                        aggregated["max_surprise"]
                    )

                    # Regime-specific adjustments (heuristic: Volatile regimes have higher noise floor)
                    # We assume regimes are ordered 0..K (Low..High Vol) - verify in production!
                    # For now, we apply a sigmoid scaling based on regime index
                    n_regimes = aligned_posteriors.shape[1]
                    regime_factors = np.linspace(
                        1.2, 0.8, n_regimes
                    )  # Low Vol -> Boost Prob, High Vol -> Dampen Prob (Noise)

                    weighted_prob = pd.Series(0.0, index=aggregated.index)
                    for k in range(n_regimes):
                        regime_col = aligned_posteriors.columns[k]
                        weighted_prob += aligned_posteriors[regime_col] * (
                            base_prob * regime_factors[k]
                        )

                    aggregated["event_probability"] = weighted_prob.clip(0.0, 1.0)
                except Exception as e:
                    tprint_warning(f"   ⚠️ Probability calibration failed: {e}")
                    aggregated["event_probability"] = self._calibrate_event_probability(
                        aggregated["max_surprise"]
                    )
            else:
                aggregated["event_probability"] = self._calibrate_event_probability(
                    aggregated["max_surprise"]
                )

            self.surprise_events_ = aggregated
            self.surprise_aggregates_df_ = aggregated.copy()

            if self.verbose:
                tprint_success("✅ Continuous surprise aggregation complete:")
                tprint_info(
                    f"   - ZoneScore Mean: {aggregated['zone_score'].mean():.4f}"
                )
                tprint_info(
                    f"   - Event Prob Mean: {aggregated['event_probability'].mean():.4f}"
                )

            self._log_exit(
                "aggregate_specialist_surprise",
                result=aggregated,
                extra=f"events={len(aggregated)}",
            )
            return aggregated
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ FAIL FAST: Surprise aggregation failed: {e}")
            raise ValueError(f"Causal surprise aggregation failed: {e}") from e

    def _build_error_frame(self) -> pd.DataFrame:
        """
        Build a time-aligned DataFrame of specialist errors.
        """
        self._log_call("_build_error_frame", specialist_errors=self.specialist_errors_)
        if not self.specialist_errors_:
            result = pd.DataFrame()
            self._log_exit("_build_error_frame", result=result)
            return result

        errors_df = pd.DataFrame(self.specialist_errors_)
        errors_df = errors_df.sort_index()

        # Drop rows that are completely empty
        errors_df = errors_df.dropna(how="all")
        self._log_exit("_build_error_frame", result=errors_df)
        return errors_df

    def _compute_batch_surprises(
        self,
        errors_df: pd.DataFrame,
        method: str = "combined",
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute surprise scores for all specialists using vectorized rolling stats.
        Includes optional Normalization (Volatility, Entropy, Liquidity).
        """
        self._log_call(
            "_compute_batch_surprises",
            errors_df=errors_df,
            method=method,
            market_data=market_data,
        )
        window = self.rolling_window
        min_periods = min(window, 20)

        rolling_median = errors_df.rolling(
            window=window, min_periods=min_periods
        ).median()

        # 1. Compute Global MAD (Robust Normalization)
        # Replacing Rolling MAD to prevent normalizing away volatility clusters
        try:
            mads_dict = {}
            for col in errors_df.columns:
                values = errors_df[col].dropna()
                if len(values) == 0:
                    mads_dict[col] = 1.0
                    continue
                    
                median = values.median()
                mad = (values - median).abs().median()
                # Apply metadata floor if available
                meta = self.specialist_metadata_.get(col, {})
                floor = max(meta.get("global_mad", 1e-6), 1e-6)
                mads_dict[col] = max(mad, floor)
            
            # Broadcast global MADs to dataframe shape
            rolling_mad = pd.DataFrame(
                {col: [val]*len(errors_df) for col, val in mads_dict.items()}, 
                index=errors_df.index
            )
        except Exception as e:
            tprint_warning(f"   ⚠️ Global MAD calculation failed: {e}")
            rolling_mad = pd.DataFrame(1.0, index=errors_df.index, columns=errors_df.columns)

        # 2. Compute Raw Surprise
        if method == "zscore" or method == "robust_zscore":
            surprise_df = (errors_df - rolling_median).abs() / rolling_mad
        elif method == "magnitude":
            surprise_df = errors_df.abs() / rolling_mad
        elif method == "combined":
            zscore_surprise = (errors_df - rolling_median).abs() / rolling_mad
            magnitude_surprise = errors_df.abs() / rolling_mad
            surprise_df = 0.6 * zscore_surprise + 0.4 * magnitude_surprise
        else:
            raise ValueError(f"Unknown surprise method: {method}")

        # 3. Apply Advanced Normalization (Volatility, Entropy, Liquidity) if Market Data available
        if market_data is not None:
            try:
                # Align market data
                mkt = market_data.reindex(errors_df.index).ffill()

                # A) Volatility Normalization
                if "close" in mkt.columns:
                    returns = mkt["close"].pct_change().fillna(0)
                    vol = returns.rolling(window=window).std().fillna(1.0)
                    vol_factor = vol.replace(0, 0.001)
                    surprise_df = surprise_df.div(vol_factor * 100, axis=0)

                # B) Entropy Normalization (H)
                if "close" in mkt.columns:
                    returns = mkt["close"].pct_change().fillna(0)
                    entropy = self._compute_rolling_entropy(returns, window=window)
                    entropy_factor = 1.0 + entropy
                    surprise_df = surprise_df.div(entropy_factor, axis=0)

                # C) Liquidity Normalization (L)
                if "close" in mkt.columns and "volume" in mkt.columns:
                    returns = mkt["close"].pct_change().fillna(0)
                    vol = returns.rolling(window=window).std().fillna(1e-6)
                    volume_ma = mkt["volume"].rolling(window=window).mean().fillna(1e-6)
                    L = volume_ma / (vol + 1e-9)
                    liq_score = L / (L + 1.0)
                    surprise_df = surprise_df.mul(liq_score, axis=0)

            except Exception as e:
                tprint_warning(f"⚠️ Market Data Normalization failed: {e}")

        result = surprise_df.fillna(0)
        self._log_exit("_compute_batch_surprises", result=result)
        return result

    def _compute_specialist_zone_levels(
        self, surprise_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Map specialist surprise values to discrete zones per specialist."""
        self._log_call("_compute_specialist_zone_levels", surprise_df=surprise_df)
        if surprise_df.empty:
            result = pd.DataFrame(index=surprise_df.index)
            self._log_exit("_compute_specialist_zone_levels", result=result)
            return result

        zone_levels = {}
        zone_weights = {} # Store continuous weights
        window = 500
        
        for col in surprise_df.columns:
            series = surprise_df[col]
            try:
                details = detect_rolling_quantile_surprises(
                    series, 
                    window=window, 
                    quantiles=(0.96, 0.98),
                    return_details=True
                )
                levels = details['level']
                weights = details['weight']
            except Exception as e:
                tprint_warning(f"   ⚠️ Detection util failed for {col}: {e}")
                levels = pd.Series(1.0, index=series.index)
                weights = pd.Series(0.0, index=series.index)
            
            zone_levels[col] = levels
            zone_weights[col] = weights
            
        self.zone_weights_ = pd.DataFrame(zone_weights, index=surprise_df.index)
        result = pd.DataFrame(zone_levels, index=surprise_df.index)
        self._log_exit("_compute_specialist_zone_levels", result=result)
        return result

    def _compute_specialist_zone_scores(
        self, zone_levels: pd.DataFrame, weights_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert zone levels to soft 0-1 scores using adaptive weights."""
        self._log_call(
            "_compute_specialist_zone_scores",
            zone_levels=zone_levels,
            weights_df=weights_df,
        )
        if zone_levels.empty:
            self._log_exit("_compute_specialist_zone_scores", result=zone_levels)
            return zone_levels
            
        # Base discrete scores
        score_map = {1.0: 0.0, 2.0: 0.5, 3.0: 1.0}
        zone_scores = zone_levels.replace(score_map)
        
        # Apply continuous weighting if enabled
        # The 'weights_df' contains (Value / Threshold_Q1), so >1.0 means event
        if self.zone_score_config.get("use_continuous_mapping", True):
            # Clip weight to reasonable range (e.g. 2x threshold = extreme)
            normalized = weights_df.clip(0.0, 2.0)
            
            # Combine: Soft score = min(1.0, max(Discrete, Normalized))
            # This ensures that if an event is technically in Zone 2 but barely (weight=1.01),
            # it gets score ~0.5. If it's deep in Zone 2 (weight=1.5), it gets higher score.
            # Using the existing logic:
            zone_scores = zone_scores.combine(
                normalized, lambda z, n: np.minimum(1.0, np.maximum(z, z * n))
            )
            
        result = zone_scores.fillna(0.0)
        self._log_exit("_compute_specialist_zone_scores", result=result)
        return result

    def _compute_combined_zone_score(
        self, zone_scores: pd.DataFrame, zone_levels: pd.DataFrame
    ) -> pd.Series:
        """Aggregate specialist zone scores into a single scalar with chaos emphasis."""
        self._log_call(
            "_compute_combined_zone_score",
            zone_scores=zone_scores,
            zone_levels=zone_levels,
        )
        if zone_scores.empty:
            result = pd.Series(0.0, index=zone_scores.index)
            self._log_exit("_compute_combined_zone_score", result=result)
            return result
        powered = zone_scores.pow(self.zone_score_power)
        base_score = powered.mean(axis=1).pow(1.0 / self.zone_score_power)
        zone3_ratio = (zone_levels == 3.0).sum(axis=1) / np.maximum(
            1, zone_levels.shape[1]
        )
        zone2_ratio = (zone_levels == 2.0).sum(axis=1) / np.maximum(
            1, zone_levels.shape[1]
        )
        boost = (
            1.0
            + self.zone3_ratio_boost * zone3_ratio
            + self.zone2_ratio_boost * zone2_ratio
        )
        boosted = base_score * boost
        boosted = boosted.clip(lower=0.0, upper=self.zone_score_cap)
        boosted[zone3_ratio >= self.zone3_floor] = self.zone_score_cap
        result = boosted.fillna(0.0)
        self._log_exit("_compute_combined_zone_score", result=result)
        return result

    def _compute_batch_breaks(
        self, errors_df: pd.DataFrame, method: str = "chow"
    ) -> pd.DataFrame:
        """
        Compute structural break indicators for all specialists.
        """
        self._log_call(
            "_compute_batch_breaks",
            errors_df=errors_df,
            method=method,
        )
        window = self.structural_break_window
        if method != "chow":
            # Fallback to per-specialist computation for other methods
            break_data = {}
            for specialist_name in errors_df.columns:
                break_indicators = self.detect_structural_breaks(
                    specialist_name, method=method
                )
                break_data[specialist_name] = break_indicators
            result = pd.DataFrame(break_data).fillna(0)
            self._log_exit("_compute_batch_breaks", result=result)
            return result

        if window <= 0:
            result = pd.DataFrame(
                np.zeros_like(errors_df),
                index=errors_df.index,
                columns=errors_df.columns,
            )
            self._log_exit("_compute_batch_breaks", result=result)
            return result

        mean_before = errors_df.rolling(window=window, min_periods=window).mean()
        var_before = errors_df.rolling(window=window, min_periods=window).var()

        shifted_errors = errors_df.shift(-window)
        mean_after = shifted_errors.rolling(window=window, min_periods=window).mean()
        var_after = shifted_errors.rolling(window=window, min_periods=window).var()

        mean_diff = (mean_before - mean_after).abs()

        var_ratio = pd.DataFrame(
            np.maximum(var_before, var_after)
            / (np.minimum(var_before, var_after) + 1e-8),
            index=errors_df.index,
            columns=errors_df.columns,
        ).replace([np.inf, -np.inf], np.nan)

        variance_threshold = 2 * np.sqrt(var_before.clip(lower=1e-8))
        mean_breaks = mean_diff > variance_threshold
        variance_breaks = var_ratio > 3

        break_df = (mean_breaks | variance_breaks).astype(int)
        result = break_df.fillna(0)
        self._log_exit("_compute_batch_breaks", result=result)
        return result

    def generate_causal_events(
        self,
        event_threshold: float = 0.25,
        min_event_separation: float = 0.25,
        market_data: Optional[pd.DataFrame] = None,
        regime_posteriors: Optional[pd.DataFrame] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate causal surprise events from aggregated data using Regime-Conditional Thresholds.

        Args:
            market_data: DataFrame with price/volume columns for entropy + volume spike gating.
            regime_posteriors: DataFrame of GMM posterior probabilities for soft thresholding.
        """
        self._log_call(
            "generate_causal_events",
            event_threshold=event_threshold,
            min_event_separation=min_event_separation,
            market_data=market_data,
            regime_posteriors=regime_posteriors,
        )
        try:
            if self.verbose:
                tprint_info("🎯 Generating Causal Surprise Events...")

            if self.surprise_events_ is None or len(self.surprise_events_) == 0:
                self.aggregate_specialist_surprise(regime_posteriors=regime_posteriors)

            if self.surprise_events_ is None or len(self.surprise_events_) == 0:
                tprint_warning(
                    "   ⚠️ No surprise events found after aggregation - check specialist registration"
                )
                result: Dict[int, Dict[str, Any]] = {}
                self._log_exit(
                    "generate_causal_events",
                    result=result,
                    extra="empty_surprise_events",
                )
                return result

            scores = self.surprise_events_["max_surprise"]
            target_quantile = 1.0 - float(event_threshold)
            target_quantile = float(np.clip(target_quantile, 0.5, 0.999))

            # --- 1. Regime-Conditional Adaptive Thresholding ---
            # Theta(t) = sum_r( P(r|t) * Theta_r )
            # Theta_r uses quantile = 1 - event_threshold when event_threshold is a density target.
            quantile_target = 0.96
            if 0.0 < event_threshold < 0.5:
                quantile_target = 1.0 - event_threshold
            if self.verbose:
                tprint_info(
                    f"   📐 Surprise quantile target: {quantile_target:.3f} (event_threshold={event_threshold})"
                )

            if regime_posteriors is not None:
                aligned_posteriors = (
                    regime_posteriors.reindex(scores.index).ffill().fillna(0.0)
                )
                n_regimes = aligned_posteriors.shape[1]

                # Calculate Theta_r for each regime
                # Since rolling weighted quantile is expensive, we approximate:
                # 1. Calculate global rolling quantile baseline
                # 2. Calculate regime-specific scaling factors based on historical volatility of scores in that regime

                # FIX for MultiIndex (Asset Boundaries)
                if isinstance(scores.index, pd.MultiIndex):
                    ticker_level = 'ticker' if 'ticker' in scores.index.names else 1
                    base_threshold = (
                        scores.groupby(level=ticker_level, group_keys=False)
                        .rolling(window=2880, min_periods=100)
                        .quantile(quantile_target)
                        .reset_index(level=0, drop=True)
                        .reindex(scores.index)
                        .fillna(event_threshold)
                    )
                else:
                    base_threshold = (
                        scores.rolling(window=2880, min_periods=100)
                        .quantile(quantile_target)
                        .fillna(event_threshold)
                    )

                # Determine Score Volatility per Regime
                regime_score_scales = []
                for k in range(n_regimes):
                    regime_col = aligned_posteriors.columns[k]
                    weights = aligned_posteriors[regime_col].values
                    if weights.sum() > 0:
                        # Weighted mean of scores
                        w_mean = np.average(scores.values, weights=weights)
                        # Weighted std of scores
                        w_std = np.sqrt(
                            np.average((scores.values - w_mean) ** 2, weights=weights)
                        )
                        regime_score_scales.append(w_std)
                    else:
                        regime_score_scales.append(1.0)

                # Normalize scales relative to global mean scale
                global_scale = np.mean(regime_score_scales)
                regime_multipliers = [
                    s / (global_scale + 1e-9) for s in regime_score_scales
                ]

                # Blend multipliers: Multiplier(t) = sum_r( P(r|t) * Multiplier_r )
                effective_multiplier = pd.Series(0.0, index=scores.index)
                for k in range(n_regimes):
                    col = aligned_posteriors.columns[k]
                    effective_multiplier += (
                        aligned_posteriors[col] * regime_multipliers[k]
                    )

                adaptive_threshold = base_threshold * effective_multiplier
                # NOTE: Removed clip(lower=event_threshold) as it conflicts with normalized scores
                # The rolling quantile already provides appropriate thresholds for the score distribution

                if self.verbose:
                    tprint_info(
                        f"   🧬 Using Data-Driven Regime Thresholding (Mean Mult: {effective_multiplier.mean():.2f})"
                    )
            else:
                # Fallback: Global Adaptive Threshold
                # FIX for MultiIndex
                if isinstance(scores.index, pd.MultiIndex):
                    ticker_level = 'ticker' if 'ticker' in scores.index.names else 1
                    rolling_threshold = (
                        scores.groupby(level=ticker_level, group_keys=False)
                        .rolling(window=2880, min_periods=100)
                        .quantile(quantile_target)
                        .reset_index(level=0, drop=True)
                        .reindex(scores.index)
                    )
                else:
                    rolling_threshold = scores.rolling(
                        window=2880, min_periods=100
                    ).quantile(quantile_target)
                
                # NOTE: Removed clip(lower=event_threshold) as it conflicts with normalized scores
                adaptive_threshold = rolling_threshold.fillna(0.0)  # Use 0 fallback for NaN (early bars)

            # Use adaptive threshold
            is_surprised = scores > adaptive_threshold
            
            # DIAGNOSTIC: Show how many bars passed the threshold
            if self.verbose:
                passed_count = is_surprised.sum()
                total_count = len(is_surprised)
                pct = 100.0 * passed_count / total_count if total_count > 0 else 0
                tprint_info(
                    f"   📊 Threshold Check: {passed_count}/{total_count} bars passed ({pct:.2f}%) "
                    f"(target_q={target_quantile:.3f})"
                )
                tprint_info(f"   📊 Score stats: mean={scores.mean():.4f}, std={scores.std():.4f}, max={scores.max():.4f}")
                tprint_info(f"   📊 Threshold stats: mean={adaptive_threshold.mean():.4f}, min={adaptive_threshold.min():.4f}")


            # --- 1b. Regime Gating (Noise Filtering) ---
            # User Policy: Filter bottom 15% using (Entropy × VolSpike) metric
            # Entropy: Local Shannon Entropy of returns (captures information content)
            # VolSpike: Volume spike ratio (current bar / prior 10-bar mean)
            if market_data is not None:
                mkt = market_data if isinstance(market_data, pd.DataFrame) else None

                if (
                    mkt is not None
                    and "close" in mkt.columns
                    and "volume" in mkt.columns
                ):
                    # Align to surprise scores index
                    mkt_aligned = mkt.reindex(scores.index).ffill()

                    # A) Compute Local Shannon Entropy (from returns)
                    returns = mkt_aligned["close"].pct_change().fillna(0)
                    entropy = self._compute_rolling_entropy(returns, window=100)

                    # B) Compute Volume Spike Ratio (current / prior 10-bar mean)
                    volume = mkt_aligned["volume"].fillna(0)
                    vol_mean_10 = volume.rolling(window=10, min_periods=1).mean()
                    vol_spike = volume / (vol_mean_10 + 1e-9)

                    # C) Composite Noise Metric: Entropy × VolSpike
                    # Higher = more information + volume activity = NOT noise
                    # Lower  = low entropy + low volume = NOISE
                    noise_metric = entropy * vol_spike

                    # Calculate 10th percentile threshold (User: 10% instead of 15%)
                    noise_threshold = noise_metric.quantile(0.10)

                    # Identify Noise Regime (Bottom 15%)
                    is_noise = noise_metric < noise_threshold

                    # Filter
                    original_count = is_surprised.sum()
                    is_surprised = is_surprised & (~is_noise)
                    filtered_count = is_surprised.sum()

                    if self.verbose and (original_count != filtered_count):
                        removed = original_count - filtered_count
                        tprint_info(
                            f"   🛡️ Regime Gating (Entropy×VolSpike): Filtered {removed} events (threshold: {noise_threshold:.4f})"
                        )
                else:
                    # Fallback: If market_data is a Series (old behavior), use it directly
                    aligned_vol = (
                        market_data.reindex(scores.index).fillna(0)
                        if isinstance(market_data, pd.Series)
                        else pd.Series(0, index=scores.index)
                    )
                    vol_threshold = aligned_vol.quantile(0.15)
                    is_noise = aligned_vol < vol_threshold
                    original_count = is_surprised.sum()
                    is_surprised = is_surprised & (~is_noise)
                    filtered_count = is_surprised.sum()
                    if self.verbose and (original_count != filtered_count):
                        removed = original_count - filtered_count
                        tprint_info(
                            f"   🛡️ Regime Gating (Vol Fallback): Filtered {removed} events (< {vol_threshold:.4f})"
                        )

            if isinstance(self.surprise_aggregates_df_, pd.DataFrame) and not self.surprise_aggregates_df_.empty:
                aligned = self.surprise_aggregates_df_.reindex(scores.index)
                aligned["causal_surprise"] = is_surprised.astype(int)
                aligned["adaptive_threshold"] = adaptive_threshold
                self.surprise_aggregates_df_ = aligned

            # --- 2. Regime Break Retention (Fix for "Slow-Moving") ---
            # Max duration: 96 bars (24 hours on 15m) - previously 48
            # If > max_duration, we flag as "Structural Break" but KEEP the start
            max_duration = 96

            # Find contiguous blocks
            block_id = (is_surprised != is_surprised.shift(1).fillna(False)).cumsum()
            block_lengths = is_surprised.groupby(block_id).transform("count")

            # Determine type: 'novelty' (short) vs 'structural_break' (long)
            # Only consider "on" blocks
            event_types = pd.Series("none", index=is_surprised.index)
            event_types[is_surprised & (block_lengths <= max_duration)] = "novelty"
            event_types[is_surprised & (block_lengths > max_duration)] = (
                "structural_break"
            )

            # Generate mask for STARTS of events (Novelty OR Structural Break)
            surprise_start = is_surprised & (~is_surprised.shift(1).fillna(False))
            surprise_mask = surprise_start  # We keep both!

            # DIAGNOSTIC
            if self.verbose:
                novelty_count = (event_types[surprise_mask] == "novelty").sum()
                break_count = (event_types[surprise_mask] == "structural_break").sum()
                tprint_info(
                    f"   📊 Events detected: {surprise_mask.sum()} (Novelty: {novelty_count}, Break: {break_count})"
                )

            event_candidates = self.surprise_events_[surprise_mask].index

            # --- 3. Clustering & Min Separation ---
            # Aggregate close events into clusters
            cluster_events = []

            cached_events = list(event_candidates)

            if not cached_events:
                tprint_warning("   ⚠️ No events passed thresholds")
                result: Dict[int, Dict[str, Any]] = {}
                self._log_exit(
                    "generate_causal_events",
                    result=result,
                    extra="no_candidates",
                )
                return result

            # Simple clustering: if within window, add to current cluster, else start new
            current_cluster_start = cached_events[0]
            current_cluster_count = 1
            current_cluster_max_score = scores.loc[current_cluster_start]

            for i in range(1, len(cached_events)):
                evt_time = cached_events[i]
                time_diff = (evt_time - current_cluster_start).total_seconds() / 3600.0

                if time_diff < min_event_separation:
                    # Within cluster
                    current_cluster_count += 1
                    current_cluster_max_score = max(
                        current_cluster_max_score, scores.loc[evt_time]
                    )
                else:
                    # Close previous cluster
                    cluster_events.append(
                        {
                            "time": current_cluster_start,
                            "count": current_cluster_count,
                            "score": current_cluster_max_score,
                            "label_type": event_types.loc[current_cluster_start],
                        }
                    )
                    # Start new
                    current_cluster_start = evt_time
                    current_cluster_count = 1
                    current_cluster_max_score = scores.loc[evt_time]

            # Close last
            cluster_events.append(
                {
                    "time": current_cluster_start,
                    "count": current_cluster_count,
                    "score": current_cluster_max_score,
                    "label_type": event_types.loc[current_cluster_start],
                }
            )

            # Generate event dictionary
            causal_events = {}

            for evt in cluster_events:
                event_time = evt["time"]
                event_data = self.surprise_events_.loc[event_time]

                # --- 4. Directional Split ---
                # Determine direction from mean_signed_error
                # If mean_signed_error > 0 => Prediction < Target => Price Shifted UP => UPSIDE SURPRISE
                # If mean_signed_error < 0 => Prediction > Target => Price Shifted DOWN => DOWNSIDE SURPRISE
                signed_err = event_data.get("mean_signed_error", 0.0)
                direction = 1 if signed_err >= 0 else -1

                causal_events[event_time] = {
                    "type": "causal_surprise",
                    "label_type": evt[
                        "label_type"
                    ],  # 'novelty' or 'structural_break' (Semantic Ambiguity Fix)
                    "strength": evt["score"],  # Max score in cluster
                    "cluster_size": evt["count"],  # Clustering count
                    "direction": direction,  # Directional Split
                    "zone": int(event_data["surprise_zone"]),
                    "consensus": event_data["surprise_consensus"],
                    "mean_surprise": event_data["mean_surprise"],
                    "zone_score": event_data["zone_score"],
                    "has_structural_break": event_data["has_break"],
                    "total_breaks": event_data["total_breaks"],
                    "specialist_count": len(self.specialist_errors_),
                    "source": "specialist_prediction_errors",
                    "sigma_method": "rolling_mad_normalized",
                    "event_id": f"causal_surprise_{event_time.isoformat()}",
                }

            self.surprise_events_ = causal_events

            if self.verbose:
                tprint_success(
                    f"✅ Generated {len(causal_events)} causal surprise events (merged from clusters):"
                )
                tprint_info("   - Event threshold: Adaptive (Quantile 96%)")
                tprint_info(f"   - Min separation: {min_event_separation} hours")

            self._log_exit(
                "generate_causal_events",
                result=causal_events,
                extra=f"events={len(causal_events)}",
            )
            return causal_events

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Event generation failed: {e}")
            return {}

    def _filter_slow_moving_events(
        self, is_surprised: pd.Series, max_duration: int
    ) -> pd.Series:
        """
        Filter out events that are part of a long-duration surprise sequence.
        Returns mask where True = keep (not slow moving).
        """
        self._log_call(
            "_filter_slow_moving_events",
            is_surprised=is_surprised,
            max_duration=max_duration,
        )
        # Identify blocks of True
        block_id = (is_surprised != is_surprised.shift()).cumsum()
        duration = is_surprised.groupby(block_id).transform("count")

        # Keep only if duration <= max_duration OR it's not a surprise anyway
        # We only want to filter out existing surprises that are too long
        keep_mask = ~is_surprised | (duration <= max_duration)
        self._log_exit("_filter_slow_moving_events", result=keep_mask)
        return keep_mask

    def analyze_surprise_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in causal surprise events.

        Returns:
            Dictionary with pattern analysis
        """
        self._log_call("analyze_surprise_patterns")
        try:
            if self.surprise_events_ is None or len(self.surprise_events_) == 0:
                result = {}
                self._log_exit("analyze_surprise_patterns", result=result, extra="no_events")
                return result

            # Convert to DataFrame for analysis
            events_df = pd.DataFrame.from_dict(self.surprise_events_, orient="index")

            if events_df.empty:
                result = {}
                self._log_exit("analyze_surprise_patterns", result=result, extra="empty_dataframe")
                return result

            analysis = {
                "total_events": len(events_df),
                "avg_strength": events_df["strength"].mean(),
                "max_strength": events_df["strength"].max(),
                "avg_consensus": events_df["consensus"].mean(),
                "break_events": events_df["has_structural_break"].sum(),
                "break_ratio": events_df["has_structural_break"].mean(),
                "avg_zone_score": (
                    events_df["zone_score"].mean()
                    if "zone_score" in events_df.columns
                    else 0.0
                ),
                "avg_zone3_ratio": (
                    events_df["zone3_ratio"].mean()
                    if "zone3_ratio" in events_df.columns
                    else 0.0
                ),
                "avg_zone2_ratio": (
                    events_df["zone2_ratio"].mean()
                    if "zone2_ratio" in events_df.columns
                    else 0.0
                ),
                "strength_distribution": events_df["strength"].describe().to_dict(),
                "consensus_distribution": events_df["consensus"].describe().to_dict(),
            }

            # Temporal patterns
            if len(events_df) > 1:
                event_times = events_df.index
                time_diffs = event_times[1:] - event_times[:-1]
                analysis["avg_time_between_events"] = time_diffs.mean()
                analysis["event_frequency"] = len(events_df) / (
                    time_diffs.sum().total_seconds() / 3600
                )  # events per hour

            self._log_exit("analyze_surprise_patterns", result=analysis)
            return analysis

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Pattern analysis failed: {e}")
            return {}

    def compute_reliability_metrics(
        self, realized_outcomes: pd.Series, binary_labels: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Compute advanced reliability metrics for specialists and detector.

        Args:
            realized_outcomes: Continuous future returns / outcome magnitude
            binary_labels: Ground truth meta-labels (optional)

        Returns:
            Dictionary with reliability metrics
        """
        self._log_call(
            "compute_reliability_metrics",
            realized_outcomes=realized_outcomes,
            binary_labels=binary_labels,
        )
        try:
            if self.verbose:
                tprint_info("🔬 Computing Advanced Causal Reliability Metrics...")

            events_df_aligned = self._get_events_dataframe()
            if self.specialist_surprises_.empty or events_df_aligned.empty:
                tprint_warning(
                    "⚠️ No surprise data available for reliability computation"
                )
                result = {}
                self._log_exit("compute_reliability_metrics", result=result, extra="no_surprise_data")
                return result

            if not isinstance(realized_outcomes, pd.Series):
                realized_outcomes = pd.Series(
                    realized_outcomes,
                    index=self.specialist_surprises_.index[: len(realized_outcomes)],
                )
            if binary_labels is not None and not isinstance(binary_labels, pd.Series):
                binary_labels = pd.Series(
                    binary_labels, index=realized_outcomes.index[: len(binary_labels)]
                )

            common_idx = self.specialist_surprises_.index.intersection(
                realized_outcomes.index
            )
            if binary_labels is not None:
                common_idx = common_idx.intersection(binary_labels.index)
            
            # DIAGNOSTIC: Check common_idx size
            if self.verbose:
                tprint_info(f"   📊 Overlap Check: surprises={len(self.specialist_surprises_)}, outcomes={len(realized_outcomes)}, common={len(common_idx)}")

            if common_idx.empty:
                tprint_warning("⚠️ No overlapping samples for reliability computation")
                result = {}
                self._log_exit("compute_reliability_metrics", result=result, extra="no_overlap")
                return result

            surprises = self.specialist_surprises_.reindex(common_idx)
            zones = self.specialist_zone_levels_.reindex(common_idx)
            outcomes = realized_outcomes.reindex(common_idx)
            labels = (
                binary_labels.reindex(common_idx) if binary_labels is not None else None
            )

            specialist_metrics = {}

            # 1. Specialist Reliability Metrics
            for spec_name in surprises.columns:
                spec_surprise = surprises[spec_name]
                if zones is None or spec_name not in zones.columns:
                    continue
                spec_zones = zones[spec_name]

                # 1.1 Surprise Responsiveness (Correlation between surprise and magnitude)
                responsiveness, _ = stats.spearmanr(spec_surprise.abs(), outcomes.abs())

                # 1.2 Zone-Conditioned Precision
                # For meta-labeling, precision is accuracy of predicting positive meta-labels
                z2_mask = spec_zones == 2.0
                z3_mask = spec_zones == 3.0

                z2_precision = 0.0
                z3_precision = 0.0

                if labels is not None:
                    if z2_mask.sum() > 5:
                        z2_precision = labels.loc[z2_mask[z2_mask].index].mean()
                    if z3_mask.sum() > 5:
                        z3_precision = labels.loc[z3_mask[z3_mask].index].mean()

                # 1.3 Confidence Calibration (Brier Score equivalent for surprise)
                # If surprise > threshold, we predict a move.
                calibration_score = 0.0
                if labels is not None:
                    preds = (spec_surprise > self.surprise_threshold).astype(float)
                    calibration_score = 1.0 - np.mean(
                        (preds - labels) ** 2
                    )  # Accuracy-like Brier

                # 1.4 Marginal Value (Leave-One-Out)
                # Proxy: Correlation of total consensus vs LOO consensus
                total_consensus = (surprises > self.surprise_threshold).sum(axis=1)
                loo_surprises = surprises.drop(columns=[spec_name])
                loo_consensus = (loo_surprises > self.surprise_threshold).sum(axis=1)

                full_corr, _ = stats.spearmanr(total_consensus, outcomes.abs())
                loo_corr, _ = stats.spearmanr(loo_consensus, outcomes.abs())
                marginal_value = full_corr - loo_corr

                # 1.5 Consensus Alignment (Correlation with other specialists)
                consensus_corr = 0.0
                if loo_consensus.std() > 0:
                    consensus_corr, _ = stats.spearmanr(
                        spec_surprise.abs(), loo_consensus
                    )

                specialist_metrics[spec_name] = {
                    "responsiveness": responsiveness,
                    "z2_precision": z2_precision,
                    "z3_precision": z3_precision,
                    "calibration": calibration_score,
                    "marginal_value": marginal_value,
                    "consensus_corr": consensus_corr,
                }

                # 3. Composite Reliability Score
                specialist_metrics[spec_name]["composite_reliability"] = (
                    self._compute_composite_reliability(specialist_metrics[spec_name])
                )

            self.specialist_reliability_ = specialist_metrics

            # 2. Detector Reliability Metrics
            # Ensure we're using the aligned DataFrame for consensus/chaos
            # Use 'events_df_aligned' which we prepared earlier

            detector_metrics = {"filtered_event_density": self.surprise_density_}
            consensus_chaos_corr = 0.0
            if {"surprise_consensus", "total_breaks"}.issubset(
                events_df_aligned.columns
            ):
                aligned_slice = events_df_aligned.reindex(common_idx)
                if (
                    aligned_slice[["surprise_consensus", "total_breaks"]]
                    .dropna(how="all")
                    .shape[0]
                    > 1
                ):
                    consensus_chaos_corr = stats.spearmanr(
                        aligned_slice["surprise_consensus"],
                        aligned_slice["total_breaks"],
                    )[0]
            detector_metrics["consensus_chaos_correlation"] = consensus_chaos_corr

            if labels is not None:
                aligned = events_df_aligned.reindex(common_idx)
                if "causal_surprise" in events_df_aligned.columns:
                    surprise_mask = (
                        aligned["causal_surprise"].fillna(0).astype(int) == 1
                    )
                else:
                    # Deriving mask from presence in events DataFrame
                    surprise_mask = pd.Series(False, index=common_idx)
                    common_events = events_df_aligned.index.intersection(common_idx)
                    surprise_mask.loc[common_events] = True

                # 2.1 Precision
                active_labels = labels[surprise_mask]
                precision = active_labels.mean() if not active_labels.empty else 0.0
                detector_metrics["precision"] = precision

                # 2.2 Recall (Capture Rate of Profitable Opportunities)
                profitable_mask = labels == 1
                captured = (surprise_mask & profitable_mask).sum()
                recall = captured / max(1, profitable_mask.sum())
                detector_metrics["recall"] = recall

                # DIAGNOSTIC: Explain recall computation
                if self.verbose:
                    tprint_info("   📊 Recall Diagnostics:")
                    tprint_info(
                        f"      - Total profitable opportunities (labels==1): {profitable_mask.sum()}"
                    )
                    tprint_info(f"      - Surprise-flagged bars: {surprise_mask.sum()}")
                    tprint_info(f"      - Overlap (captured): {captured}")
                    if (
                        profitable_mask.sum() > 0
                        and surprise_mask.sum() > 0
                        and captured == 0
                    ):
                        tprint_warning(
                            "      ⚠️ Zero overlap: Surprise events may not align with ground truth labels"
                        )

                # 2.3 F1 Score
                detector_metrics["f1"] = (
                    2 * (precision * recall) / max(1e-8, precision + recall)
                )

                # 2.4 Stability Across Time (Split Half Reliability)
                mid_point = len(common_idx) // 2
                first_half = common_idx[:mid_point]
                second_half = common_idx[mid_point:]

                def _precision_for_slice(idx_slice: pd.Index) -> float:
                    if len(idx_slice) == 0:
                        return 0.0
                    slice_mask = surprise_mask.loc[idx_slice]
                    if not slice_mask.any():
                        return 0.0
                    slice_labels = labels.loc[idx_slice][slice_mask]
                    return float(slice_labels.mean()) if not slice_labels.empty else 0.0

                prec_h1 = _precision_for_slice(first_half)
                prec_h2 = _precision_for_slice(second_half)
                detector_metrics["stability_index"] = 1.0 - abs(
                    prec_h1 - prec_h2
                )  # 1.0 is perfectly stable

            self.detector_reliability_ = detector_metrics

            if self.verbose:
                tprint_success("✅ Causal Reliability Metrics computed")
                if "f1" in detector_metrics:
                    tprint_info(
                        f"   - Detector: F1={detector_metrics['f1']:.3f}, Recall={detector_metrics['recall']:.3f}, Precision={detector_metrics.get('precision', 0.0):.3f}"
                    )
                for spec, m in specialist_metrics.items():
                    tprint_info(
                        f"   - {spec}: Reliability {m['composite_reliability']:.3f} (Resp: {m['responsiveness']:.3f})"
                    )

            result = {"specialists": specialist_metrics, "detector": detector_metrics}
            self._log_exit("compute_reliability_metrics", result=result)
            return result

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Reliability computation failed: {e}")
            return {}

    def _get_events_dataframe(self) -> pd.DataFrame:
        """Return the most informative representation of surprise events as a DataFrame."""
        self._log_call("_get_events_dataframe")
        if (
            hasattr(self, "surprise_aggregates_df_")
            and not self.surprise_aggregates_df_.empty
        ):
            result = self.surprise_aggregates_df_.copy()
            self._log_exit("_get_events_dataframe", result=result)
            return result
        if isinstance(self.surprise_events_, pd.DataFrame):
            result = self.surprise_events_.copy()
            self._log_exit("_get_events_dataframe", result=result)
            return result
        if isinstance(self.surprise_events_, dict) and self.surprise_events_:
            df = pd.DataFrame.from_dict(self.surprise_events_, orient="index")
            result = df.sort_index()
            self._log_exit("_get_events_dataframe", result=result)
            return result
        result = pd.DataFrame()
        self._log_exit("_get_events_dataframe", result=result)
        return result

    def _calibrate_event_probability(self, scores: pd.Series) -> pd.Series:
        """
        Map raw surprise scores to [0, 1] probability using Isotonic Regression proxy (Sigmoid).
        P(Event) = Sigmoid( (Score - Median) / MAD )
        """
        self._log_call("_calibrate_event_probability", scores=scores)
        median = scores.rolling(2000, min_periods=100).median()
        mad = scores.rolling(2000, min_periods=100).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        z = (scores - median) / (mad + 1e-9)
        # Calibrated to 50% prob at 2 sigma
        calibrated = 1.0 / (1.0 + np.exp(-(z - 2.0)))
        self._log_exit("_calibrate_event_probability", result=calibrated)
        return calibrated

    def _compute_composite_reliability(self, metrics: Dict[str, float]) -> float:
        """Combine metrics into a single reliability score."""
        self._log_call("_compute_composite_reliability", metrics=metrics)
        # Weights for the 2026 Pro Secret Reliability Formula:
        # Precision in Zone2 (Opportunity) is the primary driver (40%)
        # Response to market magnitude is secondary (30%)
        # Alignment with the committee (consensus) is tertiary (20%)
        # Marginal value adds the final alpha bump (10%)
        w1, w2, w3, w4 = 0.4, 0.2, 0.3, 0.1

        # Normalize and combine
        score = (
            w1 * max(0, metrics["z2_precision"])
            + w2 * max(0, metrics.get("consensus_corr", 0.5))
            + w3 * max(0, metrics["responsiveness"])
            + w4
            * np.tanh(
                max(0, metrics["marginal_value"] * 20)
            )  # Non-linear marginal lift
        )
        result = float(np.clip(score, 0, 1))
        self._log_exit("_compute_composite_reliability", result=result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of causal surprise detection.

        Returns:
            Summary dictionary
        """
        self._log_call("get_summary")
        summary = {
            "specialists_registered": len(self.specialist_errors_),
            "surprise_threshold": self.surprise_threshold,
            "rolling_window": self.rolling_window,
            "structural_break_window": self.structural_break_window,
            "events_generated": (
                len(self.surprise_events_) if self.surprise_events_ else 0
            ),
            "has_surprise_data": self.surprise_events_ is not None,
        }
        self._log_exit("get_summary", result=summary)
        return summary


# Convenience functions
def quick_causal_surprise(
    specialist_predictions: Dict[str, pd.Series],
    specialist_targets: Dict[str, pd.Series],
    **kwargs,
) -> CausalSurpriseDetector:
    """
    Quick causal surprise detection.

    Args:
        specialist_predictions: Dictionary of specialist predictions
        specialist_targets: Dictionary of specialist targets
        **kwargs: Additional parameters

    Returns:
        CausalSurpriseDetector instance
    """
    tprint_info(
        "▶️ quick_causal_surprise | "
        f"specialist_predictions={_summarize_obj(specialist_predictions)}, "
        f"specialist_targets={_summarize_obj(specialist_targets)}"
    )
    detector = CausalSurpriseDetector(**kwargs)

    # Register all specialists
    for spec_name, predictions in specialist_predictions.items():
        if spec_name in specialist_targets:
            detector.register_specialist(
                spec_name, predictions, specialist_targets[spec_name]
            )

    # Generate events
    detector.aggregate_specialist_surprise()
    detector.generate_causal_events()

    tprint_success(
        "✅ quick_causal_surprise complete | "
        f"specialists_registered={len(detector.specialist_errors_)}"
    )
    return detector


def detect_mechanism_breaks(
    specialist_errors: Dict[str, pd.Series], threshold: float = 2.0, **kwargs
) -> pd.DataFrame:
    """
    Detect mechanism breaks from specialist errors.

    Args:
        specialist_errors: Dictionary of specialist prediction errors
        threshold: Surprise threshold
        **kwargs: Additional parameters

    Returns:
        DataFrame with mechanism break indicators
    """
    tprint_info(
        "▶️ detect_mechanism_breaks | "
        f"specialist_errors={_summarize_obj(specialist_errors)}, "
        f"threshold={threshold}"
    )
    detector = CausalSurpriseDetector(surprise_threshold=threshold, **kwargs)

    # Register specialists with errors
    for spec_name, errors in specialist_errors.items():
        # Create mock predictions (zeros) and targets (errors)
        predictions = pd.Series(0, index=errors.index)
        targets = errors
        detector.register_specialist(spec_name, predictions, targets)

    # Aggregate and return surprise events
    result = detector.aggregate_specialist_surprise()
    tprint_success(
        "✅ detect_mechanism_breaks complete | "
        f"events={len(result) if isinstance(result, pd.DataFrame) else 0}"
    )
    return result
