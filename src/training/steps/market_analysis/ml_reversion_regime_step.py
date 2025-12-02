"""Mean-reversion regime detection step (OU/Hurst teacher + XGB student).

IMPROVED VERSION with:
- Relaxed teacher thresholds for realistic mean-reversion detection
- Classification target: predicts directional moves (0=up, 1=down)
- Enhanced features: momentum divergence, reversion speed, persistence
- Isotonic calibration for proper probability estimates
- Simplified signal generation without overly strict gating
- Comprehensive diagnostics and walk-forward validation

Output: calibrated probability where:
  - 0.0 = bullish (price will increase)
  - 1.0 = bearish (price will decrease)
  - 0.5 = neutral/uncertain
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

try:
    from statsmodels.tsa.stattools import adfuller
    STATIONARITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    STATIONARITY_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.features_common.transforms.scaling_normalization import (
    zscore_normalize,
    winsorized_zscore_normalize,
    rolling_adaptive_normalize,
)
from src.training.steps.market_analysis.shared_utils.balanced_feature_extractor import (
    BalancedFeatureExtractor,
    BalancedFeatureConfig,
    FeatureCategory as BFCategory,
)
from src.utils.ml_common.trading_grid_backtester import (
    run_simple_long_grid_backtest,
    run_simple_short_grid_backtest,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
    XGBTrainingResults,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Optimized teacher feature calculations (Numba-compiled when available)
# ============================================================================

def _rolling_hurst_python(series: np.ndarray, window: int) -> np.ndarray:
    """Python fallback for rolling Hurst exponent calculation."""
    h = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        x = series[i - window : i]
        x = x[~np.isnan(x)]
        if len(x) < 10:
            continue
        r = np.diff(x)
        if len(r) < 5:
            continue
        n = len(r)
        mean_r = r.mean()
        dev = r - mean_r
        cum = np.cumsum(dev)
        R = cum.max() - cum.min()
        S = r.std()
        if S <= 0 or R <= 0:
            h[i] = 0.5
        else:
            h[i] = max(0.0, min(1.0, np.log(R / S) / np.log(n)))
    return h


def _rolling_ou_params_python(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Python fallback for rolling OU parameters calculation."""
    half = np.full(len(series), np.nan)
    theta = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        x = series[i - window : i]
        x = x[~np.isnan(x)]
        if len(x) < 10:
            continue
        x0, x1 = x[:-1], x[1:]
        x0c = x0 - x0.mean()
        x1c = x1 - x1.mean()
        denom = np.dot(x0c, x0c)
        if denom <= 0:
            continue
        phi = float(np.dot(x0c, x1c) / denom)
        if phi <= 0 or phi >= 1:
            continue
        hl = -np.log(2.0) / np.log(phi)
        half[i] = hl
        theta[i] = 1.0 / max(hl, 1e-6)
    return half, theta


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def _rolling_hurst_numba(series: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized rolling Hurst exponent calculation."""
        n_samples = len(series)
        h = np.full(n_samples, np.nan)

        for i in range(window, n_samples):
            # Extract window, removing NaNs
            x = series[i - window : i]
            # Remove NaN values
            valid_mask = ~np.isnan(x)
            x_clean = x[valid_mask]

            if len(x_clean) < 10:
                continue

            # Compute returns
            r = np.diff(x_clean)
            if len(r) < 5:
                continue

            n = len(r)
            mean_r = np.mean(r)
            dev = r - mean_r
            cum = np.cumsum(dev)
            R = np.max(cum) - np.min(cum)
            S = np.std(r)

            if S <= 0 or R <= 0:
                h[i] = 0.5
            else:
                hurst_val = np.log(R / S) / np.log(n)
                h[i] = max(0.0, min(1.0, hurst_val))

        return h

    @numba.jit(nopython=True, cache=True)
    def _rolling_ou_params_numba(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized rolling OU parameters calculation."""
        n_samples = len(series)
        half = np.full(n_samples, np.nan)
        theta = np.full(n_samples, np.nan)

        for i in range(window, n_samples):
            # Extract window, removing NaNs
            x = series[i - window : i]
            # Remove NaN values
            valid_mask = ~np.isnan(x)
            x_clean = x[valid_mask]

            if len(x_clean) < 10:
                continue

            x0 = x_clean[:-1]
            x1 = x_clean[1:]
            x0c = x0 - np.mean(x0)
            x1c = x1 - np.mean(x1)
            denom = np.dot(x0c, x0c)

            if denom <= 0:
                continue

            phi = np.dot(x0c, x1c) / denom

            if phi <= 0 or phi >= 1:
                continue

            hl = -np.log(2.0) / np.log(phi)
            half[i] = hl
            theta[i] = 1.0 / max(hl, 1e-6)

        return half, theta
else:
    # If Numba not available, point to Python versions
    _rolling_hurst_numba = _rolling_hurst_python
    _rolling_ou_params_numba = _rolling_ou_params_python


class MLMeanReversionRegimeStep(BaseStep):
    """Ornstein–Uhlenbeck / Hurst teacher → XGB classifier for mean reversion.

    Predicts directional moves (up=0, down=1) using mean-reversion signals.
    """

    def __init__(self, step_name: str = "ml_mean_reversion_step") -> None:
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLMeanReversionRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data: Optional[pd.DataFrame] = None
        self._cached_market_source: Optional[str] = None
        self._cached_market_cache_key: Optional[Tuple[str, str, str, str]] = None
        numba_status = "✅ ENABLED (Numba-optimized)" if NUMBA_AVAILABLE else "⚠️  DISABLED (Python fallback)"
        tprint(f"✅ Initialized {step_name} step (IMPROVED with classification, Teacher features: {numba_status})", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        start_time = time.time()
        tprint_info("=" * 80)
        tprint_info("🎯 MLMeanReversionRegimeStep.execute() - START")
        tprint_info("=" * 80)

        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for MLMeanReversionRegimeStep")

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "15m")))
            direction = str(config.get("direction", "long"))
            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe}, direction={direction})"
            )
            tprint_info(f"⏱️  Step start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # 1) Load OHLCV (no light-mode filter, with caching)
            tprint_info("📊 [1/9] Loading market data...")
            load_start = time.time()
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            cache_key = (symbol, exchange, regime_timeframe, exec_mode_cfg)
            if self._cached_market_data is not None and self._cached_market_cache_key == cache_key:
                market_data = self._cached_market_data.copy()
                market_source = self._cached_market_source
                tprint_info("♻️ Reusing cached market data for mean-reversion step")
            else:
                market_data, market_source = self.load_market_data_or_fail(
                    {**config, "timeframe": regime_timeframe},
                    pipeline_state={},
                    allow_config_override=True,
                    light_mode_filter=False,
                    skip_artifacts=True,
                )
                self._cached_market_data = market_data.copy() if isinstance(market_data, pd.DataFrame) else market_data
                self._cached_market_source = market_source
                self._cached_market_cache_key = cache_key
            tprint_info(f"✅ Market data loaded in {time.time() - load_start:.2f}s")

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if "timestamp" in market_data.columns:
                ts = market_data["timestamp"]
                if np.issubdtype(ts.dtype, np.datetime64):
                    market_data = market_data.copy()
                    market_data.index = ts
                elif np.issubdtype(ts.dtype, np.number):
                    market_data = market_data.copy()
                    try:
                        market_data.index = pd.to_datetime(ts.astype("int64"), unit="ms")
                    except (OverflowError, ValueError):
                        market_data.index = pd.to_datetime(ts.astype("int64"), unit="s")
                else:
                    market_data = market_data.copy()
                    market_data.index = pd.to_datetime(ts)
            elif not isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                try:
                    market_data.index = pd.to_datetime(market_data.index)
                except (TypeError, ValueError):
                    market_data.index = pd.to_datetime(market_data.index, utc=True)
                    market_data.index = market_data.index.tz_convert(None)
            if isinstance(market_data.index, pd.DatetimeIndex) and market_data.index.tz is not None:
                market_data.index = market_data.index.tz_convert(None)

            if isinstance(market_data.index, pd.DatetimeIndex):
                idx_min = market_data.index.min()
                idx_max = market_data.index.max()
                span_days = max(1, (idx_max - idx_min).days)
                timeframe_lower = str(regime_timeframe).lower()
                minutes_per_bar_map = {
                    "1m": 1,
                    "3m": 3,
                    "5m": 5,
                    "15m": 15,
                    "30m": 30,
                    "1h": 60,
                    "4h": 240,
                    "1d": 1440,
                }
                minutes_per_bar = minutes_per_bar_map.get(timeframe_lower)
                approx_days_from_bars = None
                if minutes_per_bar is not None:
                    approx_days_from_bars = max(
                        1.0,
                        float(len(market_data)) * float(minutes_per_bar) / (60.0 * 24.0),
                    )
                needs_rebuild = False
                if idx_min.year < 2000 or idx_max.year < 2000:
                    needs_rebuild = True
                elif approx_days_from_bars is not None and span_days < approx_days_from_bars * 0.2:
                    needs_rebuild = True
                if needs_rebuild and minutes_per_bar is not None:
                    freq_map = {
                        "1m": "1T",
                        "3m": "3T",
                        "5m": "5T",
                        "15m": "15T",
                        "30m": "30T",
                        "1h": "1H",
                        "4h": "4H",
                        "1d": "1D",
                    }
                    freq = freq_map.get(timeframe_lower)
                    if freq is not None:
                        base_ts = pd.Timestamp("2020-01-01 00:00:00")
                        new_index = pd.date_range(start=base_ts, periods=len(market_data), freq=freq)
                        market_data = market_data.copy()
                        market_data.index = new_index

            tprint_info(
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )

            market_data = market_data.sort_index()
            required_cols = {"open", "high", "low", "close", "volume"}
            missing = [c for c in required_cols if c not in market_data.columns]
            if missing:
                raise ValueError(f"Market data missing OHLCV columns: {missing}")

            # Create temporal split config with 6-month burn-in for indicator stabilization
            tprint_info("📊 Creating temporal split configuration...")
            split_start = time.time()
            split_config = create_temporal_split_config_for_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                data_start=market_data.index.min(),
                data_end=market_data.index.max(),
                enable_burnin=True,
                # Use default burnin_pct=1/12 (3 months)
            )
            tprint_info(
                f"📊 Temporal split config created in {time.time() - split_start:.2f}s with burn-in: "
                f"Burn-in {split_config.burnin.start if split_config.burnin else 'N/A'} → "
                f"{split_config.burnin.effective_end if split_config.burnin else 'N/A'}, "
                f"Train {split_config.training.start} → {split_config.training.effective_end}, "
                f"Val {split_config.validation.start} → {split_config.validation.effective_end}, "
                f"Test {split_config.test.start} → {split_config.test.effective_end}"
            )

            # 2) Teacher features + GMM labels + continuous reversion score
            tprint_info("🧮 [2/9] Building teacher features (Hurst, OU, variance ratio, ADF)...")
            teacher_start = time.time()
            teacher_df = self._build_teacher_features(market_data, config)
            tprint_info(f"✅ Teacher features built in {time.time() - teacher_start:.2f}s (shape={teacher_df.shape})")
            tprint_info("🎓 [3/9] Training teacher GMM (Gaussian Mixture Model)...")
            gmm_start = time.time()
            (
                gmm,
                teacher_clusters,
                teacher_binary,
                teacher_score,
                teacher_metrics,
            ) = self._train_teacher_gmm(teacher_df, config)
            tprint_info(f"✅ Teacher GMM trained in {time.time() - gmm_start:.2f}s (positive_rate={teacher_metrics.get('teacher_positive_rate', 0):.4f})")

            # 3) Student features (ENHANCED with momentum divergence, reversion speed, persistence)
            tprint_info("🎓 [4/9] Building student features (enhanced with momentum divergence, reversion speed, persistence)...")
            student_start = time.time()
            student_df = self._build_student_features(market_data, config)
            # Augment with teacher interactions so the model can learn
            # regime- and teacher-score-dependent behaviour directly.
            student_df = self._augment_with_teacher_interactions(
                student_df=student_df,
                teacher_binary=teacher_binary,
                teacher_score=teacher_score,
                config=config,
            )
            tprint_info(f"✅ Student features built in {time.time() - student_start:.2f}s (shape={student_df.shape})")

            # 3.5) Calculate dynamic ATR-based TPSL multipliers
            tprint_info("📏 [5/9] Calculating dynamic ATR-based TPSL multipliers...")
            atr_start = time.time()
            atr_14, atr_300, dynamic_tp_sl_multiplier = self._calculate_atr_multipliers(market_data, config)
            tprint_info(f"✅ ATR multipliers calculated in {time.time() - atr_start:.2f}s")

            # 4) Build classification target: forward price direction
            #    0 = price will go up (bullish)
            #    1 = price will go down (bearish)
            tprint_info("🎯 [6/9] Building classification target (forward price direction)...")
            target_start = time.time()
            y_direction_all = self._build_direction_target(market_data, config)
            tprint_info(f"✅ Target built in {time.time() - target_start:.2f}s")

            # Determine minimum number of aligned samples required for training.
            # Use a configurable threshold with a sane lower bound so that we can
            # still train in smaller windows while avoiding degenerate runs.
            try:
                min_aligned_samples = int(config.get("mr_min_aligned_samples", 300))
            except Exception:
                min_aligned_samples = 300
            if min_aligned_samples < 200:
                min_aligned_samples = 200

            # Align indices and drop any samples without a valid direction label
            valid_target_idx = y_direction_all.dropna().index
            common_idx = (
                teacher_score.index
                .intersection(student_df.index)
                .intersection(valid_target_idx)
                .sort_values()
            )
            if len(common_idx) < min_aligned_samples:
                raise ValueError(
                    f"Not enough aligned samples for training ({len(common_idx)} < {min_aligned_samples})"
                )

            X_all = student_df.loc[common_idx]
            y_target_all = y_direction_all.loc[common_idx].astype(int)
            y_teacher_binary = teacher_binary.loc[common_idx].astype(int)
            teacher_score_aligned = teacher_score.loc[common_idx]

            # If we have an explicit temporal split, restrict to the exact
            # union of train/validation/test windows used by the student model.
            # This ensures that the length of X_all matches the length of the
            # concatenated prediction arrays returned by _train_xgb_student.
            if split_config is not None:
                train_mask = (
                    (X_all.index >= split_config.training.start)
                    & (X_all.index <= split_config.training.effective_end)
                )
                val_mask = (
                    (X_all.index >= split_config.validation.start)
                    & (X_all.index <= split_config.validation.effective_end)
                )
                test_mask = (
                    (X_all.index >= split_config.test.start)
                    & (X_all.index <= split_config.test.effective_end)
                )
                union_mask = train_mask | val_mask | test_mask
                if union_mask.sum() < min_aligned_samples:
                    raise ValueError(
                        f"Not enough samples within temporal split windows ({union_mask.sum()} < {min_aligned_samples})"
                    )

                X_all = X_all.loc[union_mask]
                y_target_all = y_target_all.loc[union_mask]
                y_teacher_binary = y_teacher_binary.loc[union_mask]
                teacher_score_aligned = teacher_score_aligned.loc[union_mask]

            # Allow global launcher-level HPO toggle to control this step when
            # mr_enable_hpo is not explicitly provided in the config.
            if bool(config.get("enable_hpo", False)) and "mr_enable_hpo" not in config:
                config["mr_enable_hpo"] = True

            # 5) Run HPO if enabled
            hpo_enabled = bool(config.get("mr_enable_hpo", False))
            if hpo_enabled:
                tprint_info("🎯 [7/9] HPO enabled - optimizing XGBoost hyperparameters...")
                hpo_start = time.time()
                try:
                    hpo_best_params = self._run_hierarchical_hpo(
                        X_all,
                        y_target_all,
                        config,
                        split_config=split_config,
                    )
                    # Merge HPO results into config for training
                    for key, value in hpo_best_params.items():
                        config[f"mr_{key}"] = value
                    tprint_success(f"✅ HPO complete in {time.time() - hpo_start:.2f}s - using optimized parameters for training")
                except Exception as hpo_exc:
                    tprint_error(f"❌ HPO failed after {time.time() - hpo_start:.2f}s: {hpo_exc}")
                    raise
            else:
                tprint_info("⏭️  [7/9] HPO disabled (mr_enable_hpo=False) - using default parameters")

            # 6) Train XGB classifier and generate artifacts per direction
            tprint_info("🤖 [8/9] Training XGBoost classifier and generating artifacts...")
            direction_lower = direction.lower()
            if direction_lower == "both":
                directions_to_run: List[str] = ["long", "short"]
            elif direction_lower in {"long", "short"}:
                directions_to_run = [direction_lower]
            else:
                directions_to_run = [direction]

            tprint_info(f"📋 Processing {len(directions_to_run)} direction(s): {directions_to_run}")

            all_artifacts: Dict[str, Dict[str, str]] = {}
            all_reports: Dict[str, Dict[str, str]] = {}
            all_student_metrics: Dict[str, Dict[str, Any]] = {}
            all_fwd_metrics: Dict[str, Dict[Any, Any]] = {}

            for idx, dir_ in enumerate(directions_to_run):
                tprint_info(f"🔄 Processing direction {idx+1}/{len(directions_to_run)}: {dir_}")
                dir_start = time.time()
                # Always use the standard target (0=bullish, 1=bearish)
                # This ensures mr_probability is consistently P(bearish)
                # Note: This means we train practically identical models for both directions
                # if both are requested, but it ensures correct signal logic (avoiding
                # inverted probabilities).
                y_dir = y_target_all.copy()

                tprint_info(f"  🎓 Training XGBoost with OOF predictions for {dir_} direction...")
                train_start = time.time()
                try:
                    # Use new OOF trainer (no data leakage!)
                    oof_results = self._train_xgb_oof(
                        X_all,
                        y_dir,
                        config,
                        market_data,
                        direction=dir_
                    )
                    tprint_info(f"  ✅ XGBoost OOF training complete in {time.time() - train_start:.2f}s")

                    # Extract OOF predictions
                    oof_predictions = oof_results.oof_predictions

                    # Create student metrics from OOF metadata
                    student_metrics = {
                        "oof_windows": len(oof_results.metadata),
                        "hpo_runs": sum(1 for m in oof_results.metadata if m.get('used_hpo', False)),
                        "total_oof_predictions": len(oof_predictions),
                        "prediction_method": "oof",  # IMPORTANT: Mark as OOF
                    }
                    if oof_results.metadata:
                        student_metrics.update({
                            "first_window": oof_results.metadata[0],
                            "last_window": oof_results.metadata[-1],
                        })

                except Exception as train_exc:
                    tprint_error(f"  ❌ XGBoost OOF training failed after {time.time() - train_start:.2f}s: {train_exc}")
                    raise

                # Attach outputs to main frame for this direction
                output_df = market_data.copy()
                for c in teacher_df.columns:
                    output_df[c] = teacher_df[c]
                output_df["mr_teacher_cluster"] = teacher_clusters
                output_df["mr_teacher_mean_reversion"] = teacher_binary
                output_df["mr_teacher_score"] = teacher_score
                for c in student_df.columns:
                    output_df[c] = student_df[c]

                # Align per-direction classification target to the full
                # market_data index so that diagnostics and artifact
                # saving can always rely on mr_direction_target existing.
                aligned_target_full = y_dir.reindex(output_df.index)
                output_df["mr_direction_target"] = aligned_target_full

                # Join OOF predictions (only OOF, no training set!) as raw scores.
                # This will have NaN for non-OOF periods. Be robust to empty
                # predictions and different column names.
                raw_df = None
                if isinstance(oof_predictions, pd.DataFrame) and not oof_predictions.empty:
                    prob_col = None
                    if 'probability' in oof_predictions.columns:
                        prob_col = 'probability'
                    elif 'prediction' in oof_predictions.columns:
                        prob_col = 'prediction'
                    else:
                        # Fallback: use the first numeric column if available
                        numeric_cols = [
                            c
                            for c in oof_predictions.columns
                            if np.issubdtype(oof_predictions[c].dtype, np.number)
                        ]
                        if len(numeric_cols) == 1:
                            prob_col = numeric_cols[0]

                    if prob_col is not None:
                        raw_df = oof_predictions[[prob_col]].rename(columns={prob_col: 'mr_raw_score'})

                if raw_df is None:
                    # No usable OOF probabilities – create an all-NaN column
                    raw_df = pd.DataFrame({
                        'mr_raw_score': pd.Series(index=output_df.index, dtype=float)
                    })

                output_df = output_df.join(raw_df, how='left')

                # Mark which samples are OOF vs. filled based on raw scores
                output_df['mr_is_oof'] = ~output_df['mr_raw_score'].isna()

                # Add target for OOF samples only (safe even if index is empty)
                if isinstance(oof_predictions, pd.DataFrame) and not oof_predictions.empty:
                    common_oof_idx = oof_predictions.index.intersection(output_df.index)
                    if len(common_oof_idx) > 0:
                        aligned_oof_target = y_dir.reindex(common_oof_idx)
                        output_df.loc[common_oof_idx, "mr_direction_target"] = aligned_oof_target

                # Calibrate raw scores into probabilities using isotonic regression when possible.
                # This keeps mr_raw_score as the uncalibrated OOF score and mr_probability as
                # the calibrated probability used for downstream metrics and backtests.
                raw_series_full = output_df["mr_raw_score"].astype(float)
                target_series_full = output_df["mr_direction_target"].astype(float)

                raw_values = raw_series_full.values
                target_values = target_series_full.values
                is_oof = output_df["mr_is_oof"].astype(bool).values

                # cal_mask: OOF samples with both a valid raw score and a valid label
                cal_mask = (
                    is_oof
                    & np.isfinite(raw_values)
                    & np.isfinite(target_values)
                )
                # pred_mask: all OOF samples with a valid raw score (may or may not have labels)
                pred_mask = is_oof & np.isfinite(raw_values)

                calib_method = str(config.get("mr_oof_calibration_method", "isotonic"))
                min_cal_samples = int(config.get("mr_oof_min_calibration_samples", 200))

                if calib_method == "isotonic" and cal_mask.sum() >= max(50, min_cal_samples):
                    x_cal = raw_values[cal_mask]
                    y_cal = target_values[cal_mask]

                    # Require both classes for meaningful calibration
                    if np.unique(y_cal).size >= 2:
                        try:
                            ir = IsotonicRegression(y_min=0.0, y_max=1.0)
                            ir.fit(x_cal, y_cal)

                            # Apply calibrated mapping to all OOF predictions produced
                            # by the standardized trainer, not just the labeled subset.
                            prob_full = np.full_like(raw_values, np.nan, dtype=float)
                            if pred_mask.any():
                                prob_full[pred_mask] = ir.transform(raw_values[pred_mask])
                            output_df["mr_probability"] = prob_full
                            student_metrics["calibration_method"] = "isotonic_oof_full"
                        except Exception:
                            # Fall back to identity mapping if calibration fails
                            output_df["mr_probability"] = raw_series_full.values
                            student_metrics["calibration_method"] = "identity_oof_fallback"
                    else:
                        # Not enough class diversity to calibrate
                        output_df["mr_probability"] = raw_series_full.values
                        student_metrics["calibration_method"] = "identity_oof_insufficient_classes"
                else:
                    # Identity calibration (no change or insufficient samples)
                    output_df["mr_probability"] = raw_series_full.values
                    if "calibration_method" not in student_metrics:
                        student_metrics["calibration_method"] = "identity_oof"

                prob_dense = output_df["mr_probability"].astype(float)
                if prob_dense.notna().any():
                    prob_dense = prob_dense.ffill().bfill()
                output_df["mr_probability_dense"] = prob_dense

                output_df["mr_atr_14"] = atr_14
                output_df["mr_atr_300"] = atr_300
                output_df["mr_dynamic_tpsl_multiplier"] = dynamic_tp_sl_multiplier

                # Forward-return diagnostics at multiple horizons
                tprint_info(f"  📊 Computing forward-return diagnostics for {dir_} direction...")
                fwd_start = time.time()
                horizons_cfg = config.get("mr_forward_horizons", [4, 8, 12])
                fwd_metrics: Dict[int, Dict[str, Any]] = {}
                for h in horizons_cfg:
                    try:
                        h_int = int(h)
                    except (TypeError, ValueError):
                        continue
                    m = self._compute_forward_metrics(
                        output_df,
                        prob_col="mr_probability",
                        horizon=h_int,
                        target_col="mr_direction_target",
                    )
                    if m:
                        fwd_metrics[h_int] = m
                tprint_info(f"  ✅ Forward metrics computed in {time.time() - fwd_start:.2f}s for {len(fwd_metrics)} horizons")

                # Persist artifacts + reports for this direction
                tprint_info(f"  💾 Saving artifacts and reports for {dir_} direction...")
                save_start = time.time()
                try:
                    self.set_context(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=regime_timeframe,
                        direction=dir_,
                        model="mean_reversion",
                    )

                    artifacts, reports = self._save_artifacts_and_reports(
                        output_df=output_df,
                        X_all=X_all,
                        y_target=y_dir,
                        y_teacher=y_teacher_binary,
                        model=oof_results.models[-1] if oof_results.models else None,  # Use last trained model
                        calibrated_model=None,  # OOF trainer doesn't use separate calibrated model
                        teacher_metrics=teacher_metrics,
                        student_metrics=student_metrics,
                        fwd_metrics=fwd_metrics,
                        split_config=split_config,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=regime_timeframe,
                        market_source=str(market_source),
                        oof_metadata=oof_results.metadata,  # Pass OOF metadata
                        config=config,
                    )

                    all_artifacts[dir_] = artifacts
                    all_reports[dir_] = reports
                    all_student_metrics[dir_] = student_metrics
                    all_fwd_metrics[dir_] = fwd_metrics
                    tprint_info(f"  ✅ Artifacts saved in {time.time() - save_start:.2f}s")
                except Exception as save_exc:
                    tprint_error(f"  ❌ Failed to save artifacts after {time.time() - save_start:.2f}s: {save_exc}")
                    raise

                tprint_success(f"✅ Direction {dir_} completed in {time.time() - dir_start:.2f}s")

            exec_time = time.time() - start_time
            tprint_info("=" * 80)
            tprint_success(
                f"✅ {self.step_name} completed in {exec_time:.2f}s ({exec_time/60:.2f} minutes) with {len(X_all)} samples"
            )
            tprint_info(f"⏱️  Step end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            tprint_info("=" * 80)

            if len(directions_to_run) == 1:
                dir_key = directions_to_run[0]
                return {
                    "success": True,
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "n_samples": int(len(X_all)),
                    "metrics": {
                        "teacher": teacher_metrics,
                        "student": all_student_metrics.get(dir_key, {}),
                        "forward": all_fwd_metrics.get(dir_key, {}),
                    },
                    "artifacts": all_artifacts.get(dir_key, {}),
                    "reports": all_reports.get(dir_key, {}),
                    "execution_time": exec_time,
                }

            return {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "n_samples": int(len(X_all)),
                "metrics": {
                    dir_: {
                        "teacher": teacher_metrics,
                        "student": all_student_metrics.get(dir_, {}),
                        "forward": all_fwd_metrics.get(dir_, {}),
                    }
                    for dir_ in directions_to_run
                },
                "artifacts": all_artifacts,
                "reports": all_reports,
                "execution_time": exec_time,
            }

        except Exception as exc:  # noqa: BLE001
            exec_time = time.time() - start_time
            tprint_error("=" * 80)
            tprint_error(f"❌ {self.step_name} FAILED after {exec_time:.2f}s ({exec_time/60:.2f} minutes)")
            tprint_error(f"❌ Error: {exc}")
            tprint_error(f"⏱️  Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            tprint_error("=" * 80)
            logger.exception("Mean reversion step failed")
            return {"success": False, "error": str(exc), "execution_time": exec_time}

    # ---------------- Teacher -----------------
    def _build_teacher_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        tprint_info("  🔢 Computing log price and returns...")
        close = df["close"].astype(float)
        log_price = np.log(close.replace(0.0, np.nan)).ffill()
        returns = log_price.diff().fillna(0.0)

        hurst_window = int(config.get("mr_hurst_window", 200))
        ou_window = int(config.get("mr_ou_window", 200))
        vr_window = int(config.get("mr_variance_ratio_window", 200))
        vr_h = int(config.get("mr_variance_ratio_horizon", 5))

        tprint_info(f"  📊 Computing Hurst exponent (window={hurst_window})...")
        hurst_start = time.time()
        hurst = self._rolling_hurst(log_price.values, hurst_window)
        tprint_info(f"  ✅ Hurst computed in {time.time() - hurst_start:.2f}s")

        tprint_info(f"  📊 Computing OU parameters (window={ou_window})...")
        ou_start = time.time()
        ou_half_life, ou_theta = self._rolling_ou_params(log_price.values, ou_window)
        tprint_info(f"  ✅ OU parameters computed in {time.time() - ou_start:.2f}s")

        # Simple rolling variance ratio VR(k) using log returns
        tprint_info(f"  📊 Computing variance ratio (window={vr_window}, horizon={vr_h})...")
        vr_start = time.time()
        vr = np.full(len(returns), np.nan)
        if vr_window > vr_h + 5:
            var1 = returns.rolling(vr_window).var(ddof=1)
            sum_k = returns.rolling(vr_h).sum()
            var_k = sum_k.rolling(vr_window).var(ddof=1)
            vr_series = var_k / (vr_h * var1)
            vr_series[~np.isfinite(vr_series)] = np.nan
            vr = vr_series.to_numpy()
        tprint_info(f"  ✅ Variance ratio computed in {time.time() - vr_start:.2f}s")

        adf_p = np.full(len(close), np.nan)
        if STATIONARITY_AVAILABLE:
            adf_start = time.time()
            adf_w = int(config.get("mr_adf_window", 200))
            adf_stride = max(1, int(config.get("mr_adf_stride", 4)))
            tprint_info(
                f"  📊 Computing ADF p-values (window={adf_w}, stride={adf_stride})..."
            )
            for i in range(adf_w, len(returns), adf_stride):
                seg = returns.iloc[i - adf_w : i]
                try:
                    adf_p[i] = float(adfuller(seg.values, maxlag=0, autolag=None)[1])
                except Exception:
                    adf_p[i] = np.nan
            # Forward-fill between stride steps so every bar has a p-value.
            adf_p = pd.Series(adf_p, index=df.index).ffill().to_numpy()
            tprint_info(
                f"  ✅ ADF p-values computed in {time.time() - adf_start:.2f}s (stride={adf_stride})"
            )
        else:
            tprint_warning("  ⚠️  Statsmodels not available, skipping ADF p-values")
        teacher_df = pd.DataFrame(
            {
                "mr_hurst": hurst,
                "mr_ou_half_life": ou_half_life,
                "mr_ou_theta": ou_theta,
                "mr_variance_ratio": vr,
                "mr_adf_pvalue": adf_p,
            },
            index=df.index,
        )
        return teacher_df

    @staticmethod
    def _rolling_hurst(series: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling Hurst exponent.

        Uses vectorized Numba version if available, otherwise falls back to Python loop.
        """
        if NUMBA_AVAILABLE:
            return _rolling_hurst_numba(series, window)
        else:
            return _rolling_hurst_python(series, window)

    @staticmethod
    def _rolling_ou_params(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rolling Ornstein-Uhlenbeck parameters (half-life and theta).

        Uses vectorized Numba version if available, otherwise falls back to Python loop.
        """
        if NUMBA_AVAILABLE:
            return _rolling_ou_params_numba(series, window)
        else:
            return _rolling_ou_params_python(series, window)

    def _train_teacher_gmm(
        self, teacher_df: pd.DataFrame, config: Dict[str, Any]
    ) -> Tuple[GaussianMixture, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        """Train GMM on teacher features and identify mean-reversion regime.

        IMPROVED: Relaxed thresholds for 15m timeframe, OR logic for auxiliary features.
        """
        feat_cols = [
            "mr_hurst",
            "mr_ou_half_life",
            "mr_ou_theta",
            "mr_variance_ratio",
            "mr_adf_pvalue",
        ]
        df = teacher_df[feat_cols].copy()

        # Require only core OU/Hurst features for GMM validity
        core_gmm_cols = ["mr_hurst", "mr_ou_half_life", "mr_ou_theta"]
        mask = teacher_df[core_gmm_cols].notna().all(axis=1)

        n_valid = int(mask.sum())
        min_teacher = int(config.get("mr_min_teacher_samples", 100))
        n_comp = int(config.get("mr_teacher_n_components", 3))
        abs_min = int(config.get("mr_min_teacher_abs_min", 50))
        per_comp_min = 10 * max(1, n_comp)
        effective_min = min(min_teacher, max(abs_min, per_comp_min))

        if n_valid < effective_min:
            msg = (
                "Not enough valid teacher samples for stable GMM: "
                f"{n_valid} < {effective_min} "
                f"(configured mr_min_teacher_samples={min_teacher}, "
                f"abs_min={abs_min}, n_components={n_comp})"
            )
            raise ValueError(msg)
        if n_valid < min_teacher:
            tprint_warning(
                f"  ⚠️ Using reduced teacher sample count {n_valid} < configured "
                f"mr_min_teacher_samples={min_teacher} (effective_min={effective_min})"
            )

        # Use rolling window normalization to prevent look-ahead bias.
        # For performance on multi-year 15m data (~100k samples), prefer
        # standard rolling z-score normalization here instead of the more
        # expensive winsorized variant.
        window_size = int(config.get("mr_normalization_window", 500))
        X = zscore_normalize(
            teacher_df.loc[mask, core_gmm_cols],
            window=window_size,
        ).values.astype(float)
        n_comp = int(config.get("mr_teacher_n_components", 3))
        cov_type = str(config.get("mr_teacher_covariance_type", "diag"))
        if cov_type not in {"full", "tied", "diag", "spherical"}:
            cov_type = "diag"
        tprint_info(
            f"  📊 Fitting teacher GMM on {int(mask.sum())} samples "
            f"(n_components={n_comp}, covariance_type={cov_type})..."
        )
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=cov_type,
            random_state=42,
        )
        gmm.fit(X)
        clusters_clean = gmm.predict(X)
        clusters = pd.Series(-1, index=teacher_df.index, dtype=int)
        clusters.loc[mask.index[mask]] = clusters_clean

        # Identify mean-reversion cluster: high theta, low hurst
        stats = (
            teacher_df.loc[mask, ["mr_hurst", "mr_ou_theta"]]
            .groupby(clusters_clean)
            .mean()
        )
        if stats.empty:
            raise ValueError("GMM stats empty")
        # Normalize then score
        h_norm = (stats["mr_hurst"] - stats["mr_hurst"].mean()) / (stats["mr_hurst"].std() + 1e-8)
        th_norm = (stats["mr_ou_theta"] - stats["mr_ou_theta"].mean()) / (stats["mr_ou_theta"].std() + 1e-8)
        score = -h_norm + th_norm
        mr_cluster = int(score.idxmax())

        # IMPROVED: Relaxed thresholds for 15m timeframe (trades last 2-12 bars)
        # For 15m: half-life of 4-10 bars = 1-2.5h is reasonable for mean reversion
        h_thr = float(config.get("mr_hurst_threshold", 0.5))       # Relaxed from 0.4
        hl_thr = float(config.get("mr_half_life_threshold", 12.0)) # Relaxed from 5.0, ~3h for 15m
        adf_thr = float(config.get("mr_adf_p_threshold", 0.15))    # Relaxed from 0.1
        vr_thr = float(config.get("mr_vr_threshold", 1.2))         # Relaxed from 0.9

        h_arr = teacher_df.loc[mask, "mr_hurst"].astype(float).values
        hl_arr = teacher_df.loc[mask, "mr_ou_half_life"].astype(float).values
        vr_arr = teacher_df.loc[mask, "mr_variance_ratio"].astype(float).values
        adf_arr = teacher_df.loc[mask, "mr_adf_pvalue"].astype(float).values

        h_finite = np.isfinite(h_arr)
        hl_finite = np.isfinite(hl_arr)
        vr_finite = np.isfinite(vr_arr)
        adf_finite = np.isfinite(adf_arr)

        # Core conditions (must satisfy)
        cond_h = np.zeros_like(h_arr, dtype=bool)
        cond_h[h_finite] = h_arr[h_finite] < h_thr
        cond_hl = np.zeros_like(hl_arr, dtype=bool)
        cond_hl[hl_finite] = hl_arr[hl_finite] < hl_thr
        cond_cluster = clusters_clean == mr_cluster

        # IMPROVED: Auxiliary conditions (at least one should be true)
        cond_vr = np.zeros_like(vr_arr, dtype=bool)
        if vr_finite.any():
            cond_vr[vr_finite] = vr_arr[vr_finite] < vr_thr
        cond_adf = np.zeros_like(adf_arr, dtype=bool)
        if adf_finite.any():
            cond_adf[adf_finite] = adf_arr[adf_finite] < adf_thr

        # At least one auxiliary feature should support mean-reversion
        # If both unavailable, allow through (don't penalize)
        has_vr = vr_finite.any()
        has_adf = adf_finite.any()
        if has_vr and has_adf:
            cond_aux = cond_vr | cond_adf  # OR logic
        elif has_vr:
            cond_aux = cond_vr
        elif has_adf:
            cond_aux = cond_adf
        else:
            cond_aux = np.ones_like(h_arr, dtype=bool)  # Pass if neither available

        # Final: core conditions AND at least one auxiliary
        cond_all = cond_cluster & cond_h & cond_hl & cond_aux

        binary = pd.Series(0, index=teacher_df.index, dtype=int)
        binary.loc[mask.index[mask]] = cond_all.astype(int)

        # Continuous teacher reversion score in [0, 1]
        h_score = np.zeros_like(h_arr, dtype=float)
        h_score[h_finite] = np.clip((h_thr - h_arr[h_finite]) / max(h_thr, 1e-6), 0.0, 1.0)
        hl_score = np.zeros_like(hl_arr, dtype=float)
        hl_score[hl_finite] = np.clip((hl_thr - hl_arr[hl_finite]) / max(hl_thr, 1e-6), 0.0, 1.0)
        vr_score = np.zeros_like(vr_arr, dtype=float)
        vr_score[vr_finite] = np.clip((vr_thr - vr_arr[vr_finite]) / max(vr_thr, 1e-6), 0.0, 1.0)
        adf_score = np.zeros_like(adf_arr, dtype=float)
        adf_score[adf_finite] = np.clip((adf_thr - adf_arr[adf_finite]) / max(adf_thr, 1e-6), 0.0, 1.0)

        comp_stack = np.vstack([h_score, hl_score, vr_score, adf_score])
        base_score = np.nanmean(comp_stack, axis=0)
        # Gate by MR cluster membership
        base_score = base_score * cond_cluster.astype(float)

        teacher_score = pd.Series(0.0, index=teacher_df.index, dtype=float)
        teacher_score.loc[mask.index[mask]] = base_score

        metrics: Dict[str, Any] = {
            "n_components": n_comp,
            "mean_reversion_cluster": mr_cluster,
            "cluster_counts": clusters.value_counts().to_dict(),
            "cluster_stats": stats.to_dict(),
            "thresholds": {
                "hurst": h_thr,
                "half_life": hl_thr,
                "adf_p": adf_thr,
                "variance_ratio": vr_thr,
            },
            "teacher_positive_rate": float(binary.mean()),
        }
        return gmm, clusters, binary, teacher_score, metrics

    # ---------------- Student -----------------
    def _build_student_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Build student features with ENHANCED mean-reversion indicators:
        - Momentum divergence
        - Reversion speed
        - Regime persistence
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        vol = df["volume"].astype(float)

        ma_fast = int(config.get("mr_ma_fast_window", 20))
        ma_slow = int(config.get("mr_ma_slow_window", 50))
        vwap_w = int(config.get("mr_vwap_window", 30))

        ma_f = close.rolling(ma_fast, min_periods=ma_fast // 2).mean()
        ma_s = close.rolling(ma_slow, min_periods=ma_slow // 2).mean()
        vwap = (close * vol).rolling(vwap_w, min_periods=vwap_w // 2).sum() / (
            vol.rolling(vwap_w, min_periods=vwap_w // 2).sum() + 1e-8
        )

        dist_ma = (close - ma_s) / (ma_s.replace(0.0, np.nan))
        dist_vwap = (close - vwap) / (vwap.replace(0.0, np.nan))

        # RSI
        rsi_w = int(config.get("mr_rsi_window", 14))
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_w).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_w).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # Fix flat line case (gain=0, loss=0) -> RSI=50
        flat_mask = (gain < 1e-9) & (loss < 1e-9)
        if flat_mask.any():
            rsi[flat_mask] = 50.0

        # Bollinger width
        bb_w = int(config.get("mr_bb_window", ma_slow))
        mid = close.rolling(bb_w, min_periods=bb_w // 2).mean()
        std = close.rolling(bb_w, min_periods=bb_w // 2).std()
        bb_width = (2.0 * std) / (mid.replace(0.0, np.nan))

        # Simple volatility + volume state
        ret = close.pct_change().fillna(0.0)
        vol_std = ret.rolling(20, min_periods=10).std()
        vol_atr = (high - low).rolling(20, min_periods=10).mean() / close.replace(0.0, np.nan)

        vol_ma = vol.rolling(30, min_periods=10).mean()
        vol_std_ = vol.rolling(30, min_periods=10).std()
        vol_cv = vol_std_ / (vol_ma + 1e-8)
        vol_rel = vol / (vol_ma + 1e-8)
        log_vol = np.log1p(vol)

        # NEW: Momentum divergence features
        price_roc_5 = close.pct_change(5)
        price_roc_10 = close.pct_change(10)
        ma_roc_5 = ma_s.pct_change(5)
        ma_roc_10 = ma_s.pct_change(10)
        momentum_div_5 = price_roc_5 - ma_roc_5
        momentum_div_10 = price_roc_10 - ma_roc_10

        # RSI divergence from price position
        rsi_centered = (rsi - 50) / 50  # Normalize to [-1, 1]
        rsi_divergence = rsi_centered * dist_ma  # Positive when aligned, negative when diverging

        # NEW: Mean reversion speed indicators
        # How fast is price converging to/diverging from mean?
        dist_ma_change_2 = dist_ma.diff(2)   # 30m change for 15m bars
        dist_ma_change_4 = dist_ma.diff(4)   # 1h change
        dist_vwap_change_2 = dist_vwap.diff(2)
        dist_vwap_change_4 = dist_vwap.diff(4)

        # Acceleration toward mean (second derivative)
        dist_ma_accel = dist_ma_change_2.diff(2)

        # NEW: Regime persistence features
        # How long has price been in current regime?
        below_ma = (dist_ma < 0).astype(int)
        below_vwap = (dist_vwap < 0).astype(int)
        oversold_rsi = (rsi < 30).astype(int)
        overbought_rsi = (rsi > 70).astype(int)

        # Count consecutive periods in regime
        below_ma_periods = below_ma.rolling(20, min_periods=1).sum()
        below_vwap_periods = below_vwap.rolling(20, min_periods=1).sum()
        oversold_periods = oversold_rsi.rolling(20, min_periods=1).sum()
        overbought_periods = overbought_rsi.rolling(20, min_periods=1).sum()

        # Extreme distance (potential reversal zones)
        extreme_below = (dist_ma < -0.02).astype(int)  # >2% below MA
        extreme_above = (dist_ma > 0.02).astype(int)   # >2% above MA
        extreme_below_periods = extreme_below.rolling(10, min_periods=1).sum()
        extreme_above_periods = extreme_above.rolling(10, min_periods=1).sum()

        # Approximate higher-timeframe (HTF) context using multi-bar windows
        # 1h ~ 4 bars, 4h ~ 16 bars, 1d ~ 96 bars on 15m timeframe.
        vol_1h = ret.rolling(4, min_periods=2).std()
        vol_4h = ret.rolling(16, min_periods=4).std()
        vol_ratio_4h_1h = vol_4h / (vol_1h.replace(0.0, np.nan))

        ma_1h = close.rolling(4, min_periods=2).mean()
        ma_4h = close.rolling(16, min_periods=4).mean()
        trend_ma_1h = ma_1h - ma_1h.shift(4)
        trend_ma_4h = ma_4h - ma_4h.shift(16)

        hh_1d = close.rolling(96, min_periods=16).max()
        ll_1d = close.rolling(96, min_periods=16).min()
        range_pos_1d = (close - ll_1d) / (hh_1d - ll_1d + 1e-8)

        feats = pd.DataFrame(
            {
                # Original features
                "z_price_ma_slow": dist_ma,
                "z_price_vwap": dist_vwap,
                "rsi": rsi,
                "bb_width": bb_width,
                "ret_std_20": vol_std,
                "atr_rel_20": vol_atr,
                "log_volume": log_vol,
                "volume_rel_ma": vol_rel,
                "volume_cv_30": vol_cv,
                # HTF-inspired context (approximate 1h/4h/1d using bar windows)
                "htf_vol_1h": vol_1h,
                "htf_vol_4h": vol_4h,
                "htf_vol_ratio_4h_1h": vol_ratio_4h_1h,
                "htf_trend_ma_1h": trend_ma_1h,
                "htf_trend_ma_4h": trend_ma_4h,
                "htf_range_pos_1d": range_pos_1d,
                # NEW: Momentum divergence
                "momentum_div_5": momentum_div_5,
                "momentum_div_10": momentum_div_10,
                "rsi_divergence": rsi_divergence,
                # NEW: Reversion speed
                "dist_ma_change_2": dist_ma_change_2,
                "dist_ma_change_4": dist_ma_change_4,
                "dist_vwap_change_2": dist_vwap_change_2,
                "dist_vwap_change_4": dist_vwap_change_4,
                "dist_ma_accel": dist_ma_accel,
                # NEW: Regime persistence
                "below_ma_periods": below_ma_periods,
                "below_vwap_periods": below_vwap_periods,
                "oversold_periods": oversold_periods,
                "overbought_periods": overbought_periods,
                "extreme_below_periods": extreme_below_periods,
                "extreme_above_periods": extreme_above_periods,
            },
            index=df.index,
        )
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.dropna()

        # Optional: augment with balanced feature extractor
        if bool(config.get("mr_enable_balanced_features", True)):
            try:
                bf_config = BalancedFeatureConfig()
                bf_config.enabled_categories = [
                    BFCategory.PRICE,
                    BFCategory.VOLUME,
                    BFCategory.VOLATILITY,
                    BFCategory.MOMENTUM,
                    BFCategory.TREND,
                    BFCategory.REGIME,
                ]
                bf_config.enable_temporal_features = False
                bf_config.enable_micro_regime_features = False
                bf_config.enable_feature_selection = True
                bf_config.total_max_features = int(
                    config.get("mr_balanced_total_max_features", 64)
                )
                bf_config.max_features_per_category = int(
                    config.get("mr_balanced_max_features_per_category", 12)
                )

                extractor = BalancedFeatureExtractor(bf_config)
                bf_result = extractor.extract_balanced_features(
                    df[["open", "high", "low", "close", "volume"]]
                )
                if bf_result.success and bf_result.features.size > 0:
                    bf_df = pd.DataFrame(
                        bf_result.features,
                        index=df.index,
                        columns=[f"bf_{name}" for name in bf_result.feature_names],
                    )
                    # Align with existing feature index after dropna
                    bf_df = bf_df.loc[feats.index]
                    feats = pd.concat([feats, bf_df], axis=1)
            except Exception as e:  # noqa: BLE001
                tprint_warning(f"Balanced feature extraction failed: {e}")

        # Normalise most features with adaptive normalization (ATR for spatial
        # distance/level features, log1p+zscore for pure volume where applicable,
        # winsorized z-score for the rest). Keep a few core level features raw.
        exclude = {"z_price_ma_slow", "z_price_vwap", "rsi", "bb_width"}
        norm_cols = [c for c in feats.columns if c not in exclude]
        if norm_cols:
            window_size = int(config.get("mr_normalization_window", 500))

            # Restrict OHLC series to the feature index for ATR calculation
            high = df["high"].reindex(feats.index) if "high" in df.columns else None
            low = df["low"].reindex(feats.index) if "low" in df.columns else None
            close = df["close"].reindex(feats.index) if "close" in df.columns else None

            feats[norm_cols] = rolling_adaptive_normalize(
                feats[norm_cols],
                window=window_size,
                min_periods=window_size // 2,
                high=high,
                low=low,
                close=close,
            )
        return feats

    def _augment_with_teacher_interactions(
        self,
        student_df: pd.DataFrame,
        teacher_binary: pd.Series,
        teacher_score: pd.Series,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Augment student features with teacher score/regime and simple interactions.

        This explicitly exposes:
        - mr_teacher_score and mr_teacher_mean_reversion on the student grid
        - interactions between teacher signals and key mean-reversion features.
        """

        if student_df is None or student_df.empty:
            return student_df

        if not bool(config.get("mr_enable_teacher_interactions", True)):
            return student_df

        try:
            df_aug = student_df.copy()
            idx = df_aug.index

            score = teacher_score.reindex(idx).astype(float)
            binary = teacher_binary.reindex(idx).astype(float)

            df_aug["mr_teacher_score"] = score
            df_aug["mr_teacher_mean_reversion"] = binary

            base_cols = [
                "z_price_ma_slow",
                "z_price_vwap",
                "ret_std_20",
                "atr_rel_20",
            ]
            for col in base_cols:
                if col in df_aug.columns:
                    df_aug[f"{col}_x_mr_teacher"] = (
                        df_aug[col] * df_aug["mr_teacher_score"]
                    )
                    df_aug[f"{col}_x_mr_regime"] = (
                        df_aug[col] * df_aug["mr_teacher_mean_reversion"]
                    )

            df_aug = df_aug.replace([np.inf, -np.inf], np.nan)
            return df_aug
        except Exception:
            # If anything goes wrong, fall back to the original student features.
            return student_df

    def _build_direction_target(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Build classification target: forward price direction.

        Returns:
            0 = bullish (price will go up)
            1 = bearish (price will go down)

        For 15m bars with trades lasting 30m-3h (2-12 bars), we use a forward
        horizon that captures the typical trade duration.
        """
        close = df["close"].astype(float)

        # For 15m timeframe, prefer a longer horizon to capture mean reversion
        # (default 12 bars ≈ 3 hours)
        forward_horizon = int(config.get("mr_forward_target_horizon", 12))
        min_fixed_target = float(config.get("mr_direction_min_threshold", 0.015))  # 1.5% default
        atr_mult = float(config.get("mr_dynamic_target_atr_multiplier", 2.0))

        # Fee calc
        fee_rate = float(config.get("mr_fee_rate", 0.0015))
        effective_fee = float(config.get("mr_effective_roundtrip_fee", 2.0 * fee_rate))

        # Need ATR for dynamic target. Calculate if not already present,
        # otherwise approximate or use re-calculation.
        # Since this method might be called independently, let's re-calculate ATR-14 efficiently.
        high = df["high"].values
        low = df["low"].values
        # close is already defined as float array

        # Simple ATR calculation (Wilder's smoothing not strictly necessary for target generation, rolling mean is fine)
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        # Handle first element
        tr[0] = high[0] - low[0]

        # Use pandas for easy rolling
        atr_series = pd.Series(tr).rolling(14).mean().fillna(0.0).values

        y_direction = np.full(len(close), np.nan)

        # Triple Barrier Logic with Dynamic Threshold
        # Iterate through bars (performance note: this loop is generally fast enough for <1M bars,
        # but could be vectorized if needed. For clarity/correctness of triple-barrier, loops are safer).

        # Pre-calculate lookahead windows to avoid slicing in loop?
        # For simplicity and correctness, we'll loop standardly.

        n_bars = len(close)

        for i in range(n_bars - forward_horizon):
            entry_price = close[i]
            if entry_price <= 0:
                continue

            # Dynamic Target: max(1.5%, 2.0 * ATR/Price)
            current_atr = atr_series[i]
            vol_target_pct = 0.0
            if entry_price > 0:
                vol_target_pct = atr_mult * (current_atr / entry_price)

            base_threshold_pct = max(min_fixed_target, vol_target_pct)

            # Hurdle includes fee
            hurdle_pct = base_threshold_pct + effective_fee

            upper_barrier = entry_price * (1.0 + hurdle_pct)
            lower_barrier = entry_price * (1.0 - hurdle_pct)

            # Check horizon for touches
            # Slice strictly forward: i+1 to i+horizon (inclusive)
            # indices: i+1 ... i+horizon+1 (exclusive for python slice)
            start_idx = i + 1
            end_idx = i + 1 + forward_horizon

            w_high = high[start_idx:end_idx]
            w_low = low[start_idx:end_idx]

            # Boolean arrays of touches
            hit_upper = w_high >= upper_barrier
            hit_lower = w_low <= lower_barrier

            has_upper = hit_upper.any()
            has_lower = hit_lower.any()

            if not has_upper and not has_lower:
                continue

            # Find first index
            idx_upper = np.argmax(hit_upper) if has_upper else 999999
            idx_lower = np.argmax(hit_lower) if has_lower else 999999

            if idx_upper < idx_lower:
                # Bullish win (hit upper barrier first)
                y_direction[i] = 0
            elif idx_lower < idx_upper:
                # Bearish win (hit lower barrier first)
                # Note: "Bearish" label means price dropped.
                y_direction[i] = 1
            else:
                # Simultaneous touch in the same bar (volatility) -> Ambiguous, skip
                pass

        y_series = pd.Series(y_direction, index=df.index)

        # Optional gating: only keep labels where the OU/Hurst teacher indicates
        # a mean-reversion regime (core MR cluster or high teacher_score band)
        # and RSI is outside a neutral band. This focuses the target on
        # economically meaningful MR opportunities when the relevant columns
        # are available.
        try:
            teacher_mask = None
            if "mr_teacher_mean_reversion" in df.columns:
                mr_core = df["mr_teacher_mean_reversion"].astype(float) == 1.0
                teacher_mask = mr_core

            if "mr_teacher_score" in df.columns:
                score = df["mr_teacher_score"].astype(float)
                # Relaxed gating from 0.8 to 0.65 to expand scope
                q_conf = float(config.get("mr_teacher_score_quantile", 0.65))
                try:
                    thr = float(score.quantile(q_conf))
                    hi_band = score >= thr
                except Exception:
                    hi_band = score.notna()
                if teacher_mask is None:
                    teacher_mask = hi_band
                else:
                    teacher_mask = teacher_mask | hi_band

            if teacher_mask is None:
                gating_mask = y_series.notna().to_numpy()
            else:
                gating_mask = (teacher_mask & y_series.notna()).to_numpy()

            # RSI gating: drop labels where RSI is in a neutral range.
            # Relaxed neutral band (40-60 instead of 35-65) to expand scope
            if "rsi" in df.columns:
                rsi = df["rsi"].astype(float)
                rsi_norm = rsi / 100.0
                rsi_low = float(config.get("mr_rsi_neutral_low", 0.40))
                rsi_high = float(config.get("mr_rsi_neutral_high", 0.60))
                rsi_tradable = (rsi_norm < rsi_low) | (rsi_norm > rsi_high)
                gating_mask = gating_mask & rsi_tradable.to_numpy()

            # Apply gating mask: set all non-selected labels to NaN
            y_series.loc[~gating_mask] = np.nan
        except Exception:
            # If anything goes wrong with gating, fall back to the raw labels.
            pass

        return y_series

    def _calculate_atr_multipliers(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate dynamic ATR-based TPSL multipliers.

        Formula: target_multiplier = (ATR_14 / ATR_300)^α
        where α = 0.5 (configurable)

        Returns:
            - ATR_14: 14-bar ATR series
            - ATR_300: 300-bar ATR series
            - dynamic_tp_sl_multiplier: Multiplier series to apply to base TPSL
        """
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        # Calculate True Range
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs()
        }).max(axis=1)

        # Calculate ATR with different windows
        atr_14_window = int(config.get("mr_atr_short_window", 14))
        atr_300_window = int(config.get("mr_atr_long_window", 300))
        alpha = float(config.get("mr_atr_multiplier_alpha", 0.5))

        atr_14 = tr.rolling(atr_14_window, min_periods=atr_14_window // 2).mean()
        atr_300 = tr.rolling(atr_300_window, min_periods=atr_300_window // 2).mean()

        # Calculate dynamic multiplier: (ATR_14 / ATR_300)^α
        # When ATR_14 > ATR_300 (higher recent volatility), multiplier > 1 (wider TPSL)
        # When ATR_14 < ATR_300 (lower recent volatility), multiplier < 1 (tighter TPSL)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = atr_14 / atr_300.replace(0.0, np.nan)
            dynamic_tp_sl_multiplier = ratio ** alpha

        # Clip multiplier to reasonable range (0.5x to 2.0x)
        min_mult = float(config.get("mr_atr_multiplier_min", 0.5))
        max_mult = float(config.get("mr_atr_multiplier_max", 2.0))
        dynamic_tp_sl_multiplier = dynamic_tp_sl_multiplier.clip(lower=min_mult, upper=max_mult)

        # Fill NaN values with 1.0 (no adjustment)
        dynamic_tp_sl_multiplier = dynamic_tp_sl_multiplier.fillna(1.0)

        return atr_14, atr_300, dynamic_tp_sl_multiplier

    def _run_hierarchical_hpo(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        split_config: Optional[TemporalSplitConfig] = None,
    ) -> Dict[str, Any]:
        """Run hierarchical HPO for XGBoost parameters with tied parameter optimization.

        Optimizes parameters in groups with tied values to reduce search space:
        - reg_lambda and reg_alpha use the same value (regularization strength)
        - subsample and colsample_bytree use the same value (sampling rate)

        Returns:
            Best parameters from HPO
        """
        tprint_info("🔍 Starting Hierarchical HPO for XGBoost parameters")

        # ====================================================================
        # OPTIMIZATIONS: Warm start and dynamic subsampling for HPO
        # ====================================================================
        try:
            from src.utils.ml_common.training_efficiency import WarmStartManager, DynamicSubsampler
            
            # Setup warm start
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            model_id = f"{symbol}_{timeframe}_reversion_regime"
            warm_manager = WarmStartManager(model_id=model_id, model_type='reversion_xgb')
            warm_params = warm_manager.load_params()
            
            if warm_params:
                tprint_info(f"🔄 Loaded warm start params for reversion: {list(warm_params.keys())}")
            
            # Dynamic subsampling for HPO
            subsampler = DynamicSubsampler()
            warm_start_enabled = True
        except ImportError:
            tprint_warning("training_efficiency module not available")
            warm_params = None
            warm_manager = None
            subsampler = None
            warm_start_enabled = False
        # ====================================================================

        # Use temporal split if available
        if split_config is not None:
            train_mask = (X.index >= split_config.training.start) & (X.index <= split_config.training.effective_end)
            val_mask = (X.index >= split_config.validation.start) & (X.index <= split_config.validation.effective_end)

            X_train = X.loc[train_mask]
            X_val = X.loc[val_mask]
            y_train = y.loc[train_mask]
            y_val = y.loc[val_mask]
        else:
            # Fallback to percentage split
            n = len(X)
            n_train = int(n * 0.7)
            X_train = X.iloc[:n_train]
            X_val = X.iloc[n_train:]
            y_train = y.iloc[:n_train]
            y_val = y.iloc[n_train:]

        # Apply dynamic subsampling for HPO (10-50% based on data size)
        if subsampler is not None:
            sample_info = subsampler.get_subsample_info(len(X_train))
            if sample_info['will_subsample']:
                X_train_hpo, y_train_hpo = subsampler.sample(X_train, y_train, stratify=True)
                tprint_info(
                    f"🎯 Dynamic subsampling for HPO: {sample_info['original_samples']} -> "
                    f"{len(X_train_hpo)} ({sample_info['sample_pct']:.1%})"
                )
            else:
                X_train_hpo = X_train
                y_train_hpo = y_train
        else:
            X_train_hpo = X_train
            y_train_hpo = y_train

        # Convert to numpy with float32 for memory efficiency
        X_train_np = X_train_hpo.astype(np.float32).values
        X_val_np = X_val.astype(np.float32).values
        y_train_np = y_train_hpo.astype(np.int32).values
        y_val_np = y_val.astype(np.int32).values

        # Calculate class balance
        n_neg = (y_train_np == 0).sum()
        n_pos = (y_train_np == 1).sum()
        auto_scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        if (n_pos + n_neg) > 0:
            hpo_base_score = float(n_pos / float(n_pos + n_neg))
            eps = 1e-6
            if hpo_base_score <= 0.0:
                hpo_base_score = eps
            elif hpo_base_score >= 1.0:
                hpo_base_score = 1.0 - eps
        else:
            hpo_base_score = 0.5

        # Define parameter groups with tied parameters
        param_groups = [
            # Group 1: Model Structure (optimize first)
            ParameterGroup(
                name="structure",
                params={
                    "max_depth": {"type": "int", "low": 4, "high": 7},
                    "min_child_weight": {"type": "float", "low": 2.0, "high": 10.0},
                },
                priority=1,
                description="Model structure parameters"
            ),

            # Group 2: Regularization (tied: reg_lambda = reg_alpha)
            ParameterGroup(
                name="regularization",
                params={
                    "reg_strength": {"type": "float", "low": 0.1, "high": 2.0},  # Shared for both L1 and L2
                    "gamma": {"type": "float", "low": 0.0, "high": 0.2},
                },
                priority=2,
                depends_on=["structure"],
                description="Regularization parameters (reg_lambda=reg_alpha=reg_strength)"
            ),

            # Group 3: Sampling (tied: subsample = colsample_bytree)
            ParameterGroup(
                name="sampling",
                params={
                    "sampling_rate": {"type": "float", "low": 0.6, "high": 1.0},  # Shared for both row and column sampling
                },
                priority=3,
                depends_on=["regularization"],
                description="Sampling parameters (subsample=colsample_bytree=sampling_rate)"
            ),

            # Group 4: Learning Rate
            ParameterGroup(
                name="learning",
                params={
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                },
                priority=4,
                depends_on=["sampling"],
                description="Learning rate"
            ),
        ]

        # Define objective function
        def objective(
            params: Dict[str, Any],
            X_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            model: Optional[Any] = None,
            cv_folds: Optional[int] = None,
            scoring_metric: Optional[str] = None,
            **kwargs: Any,
        ) -> float:
            """Objective function for HPO."""
            # Expand tied parameters
            if "reg_strength" in params:
                params["reg_alpha"] = params["reg_strength"]
                params["reg_lambda"] = params["reg_strength"]
            if "sampling_rate" in params:
                params["subsample"] = params["sampling_rate"]
                params["colsample_bytree"] = params["sampling_rate"]

            # Build XGBoost params
            xgb_params = {
                "tree_method": "hist",
                "learning_rate": float(params.get("learning_rate", 0.03)),
                "max_depth": int(params.get("max_depth", 5)),
                "min_child_weight": float(params.get("min_child_weight", 5.0)),
                "subsample": float(params.get("subsample", 0.8)),
                "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
                "gamma": float(params.get("gamma", 0.05)),
                "reg_alpha": float(params.get("reg_alpha", 0.5)),
                "reg_lambda": float(params.get("reg_lambda", 0.5)),
                "n_estimators": int(config.get("mr_n_estimators", 500)),
                "scale_pos_weight": auto_scale_pos_weight,
                "eval_metric": "logloss",
                "random_state": 42,
                "base_score": hpo_base_score,
            }

            try:
                # Train model
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(
                    X_train_np,
                    y_train_np,
                    eval_set=[(X_val_np, y_val_np)],
                    verbose=False
                )

                # Predict on validation set
                y_pred_proba = model.predict_proba(X_val_np)[:, 1]

                # Calculate AUC as primary metric
                try:
                    auc = float(roc_auc_score(y_val_np, y_pred_proba))
                except ValueError:
                    auc = 0.5

                # Calculate accuracy
                y_pred = (y_pred_proba >= 0.5).astype(int)
                acc = float(accuracy_score(y_val_np, y_pred))

                # Combined score: 70% AUC + 30% ACC
                score = 0.7 * auc + 0.3 * acc

                return score

            except Exception as e:
                tprint_warning(f"HPO trial failed: {e}")
                return 0.0

        # Create optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.TPE
            ],
            cv_folds=3,  # Reduced for speed
            scoring_metric='custom',
            direction='maximize',
            n_rounds=1,  # Single round for speed
            enable_final_refinement=False,
            random_state=42,
            verbose=True,
        )

        # Run optimization
        tprint_info("🚀 Running HPO optimization...")
        result = optimizer.optimize(
            X_train=X_train_np,
            y_train=y_train_np,
            X_val=X_val_np,
            y_val=y_val_np,
        )

        # Expand tied parameters in best params
        best_params = result.best_params.copy()
        if "reg_strength" in best_params:
            best_params["reg_alpha"] = best_params["reg_strength"]
            best_params["reg_lambda"] = best_params["reg_strength"]
        if "sampling_rate" in best_params:
            best_params["subsample"] = best_params["sampling_rate"]
            best_params["colsample_bytree"] = best_params["sampling_rate"]

        tprint_success(
            f"✅ HPO Complete! Best score: {result.best_score:.4f}, "
            f"Total trials: {result.total_trials}, Time: {result.total_time:.1f}s"
        )
        tprint_info(f"📊 Best parameters: {best_params}")

        # Save best params for future warm start
        if warm_start_enabled and warm_manager is not None:
            try:
                warm_manager.save_params(best_params, metrics={'best_score': result.best_score})
                tprint_info("💾 Saved best params for future warm start")
            except Exception as e:
                tprint_warning(f"Failed to save warm start params: {e}")

        return best_params

    def _train_xgb_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        market_data: pd.DataFrame,
        direction: str = "long"
    ) -> XGBTrainingResults:
        """Train XGBoost with OOF predictions using standardized trainer.

        This replaces the old _train_xgb_student method with proper OOF predictions.
        No data leakage - only returns predictions on data the model hasn't seen.

        Args:
            X: Feature dataframe with DatetimeIndex
            y: Target series with DatetimeIndex
            config: Configuration dictionary
            market_data: Original market data for date range
            direction: Trading direction (long/short)

        Returns:
            XGBTrainingResults with OOF predictions, models, and metadata
        """
        # Create model ID
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("regime_timeframe", config.get("timeframe", "15m"))
        model_id = f"{symbol}_{exchange}_{timeframe}_mean_reversion_{direction}"

        # Allow step-level override of the minimum number of samples required
        # for each OOF training window so that shorter histories can still
        # produce usable predictions. Default to a moderately conservative
        # value (400) and clamp extreme values to avoid very small windows.
        min_samples_cfg = config.get("mr_oof_min_samples_for_training", 400)
        try:
            min_samples_int = int(min_samples_cfg)
        except Exception:
            min_samples_int = 400
        if min_samples_int < 200:
            min_samples_int = 200

        tprint_info(
            "📊 OOF configuration: "
            f"min_samples_for_training={min_samples_int}, "
            "retrain_interval_days=10, burnin_pct≈1/12"
        )

        tprint_info(f"🚀 Using StandardizedXGBTrainer for OOF predictions (model_id={model_id})")

        # Create custom config
        training_config = XGBTrainingConfig(
            model_id=model_id,
            retrain_interval_days=10,  # OOF window every 10 days of historical data
            hpo_interval_days=30,  # HPO every 30 days of historical data
            burnin_pct=1/12,  # 3 months
            min_samples_for_training=min_samples_int,

            # XGBoost parameters
            tree_method="hist",
            n_estimators=int(config.get("mr_n_estimators", 500)),
            learning_rate=float(config.get("mr_learning_rate", 0.03)),
            max_depth=int(config.get("mr_max_depth", 5)),
            min_child_weight=float(config.get("mr_min_child_weight", 5.0)),
            subsample=float(config.get("mr_subsample", 0.8)),
            colsample_bytree=float(config.get("mr_colsample_bytree", 0.8)),
            gamma=float(config.get("mr_gamma", 0.05)),
            reg_lambda=float(config.get("mr_reg_lambda", 0.5)),
            early_stopping_rounds=20,

            # HPO config
            hpo_n_estimators=300,
            hpo_n_trials=50,
            enable_warm_start=True,

            # Sparse matrices
            enable_sparse_matrices=True,
            sparsity_threshold=0.5,
        )

        # Create trainer
        trainer = StandardizedXGBTrainer(
            model_id=model_id,
            config=training_config
        )

        # Train and get OOF predictions
        results = trainer.train_and_predict(
            X=X,
            y=y,
            data_start=market_data.index.min(),
            data_end=market_data.index.max(),
            eval_metric="logloss",
            verbose=True
        )

        tprint_success(
            f"✅ OOF training complete: {len(results.oof_predictions)} predictions, "
            f"{len(results.models)} models, "
            f"{sum(1 for m in results.metadata if m.get('used_hpo', False))} HPO runs"
        )

        return results

    @staticmethod
    def _compute_forward_metrics(
        df: pd.DataFrame,
        prob_col: str,
        horizon: int,
        target_col: str = "mr_direction_target"
    ) -> Dict[str, Any]:
        """Compute forward-looking metrics for model validation."""
        if "close" not in df.columns or prob_col not in df.columns:
            return {}

        close = df["close"].astype(float).values
        fwd = np.full(len(close), np.nan)
        for i in range(len(close) - horizon):
            if close[i] > 0 and close[i + horizon] > 0:
                fwd[i] = (close[i + horizon] - close[i]) / close[i]

        probs = df[prob_col].values
        mask = np.isfinite(fwd) & np.isfinite(probs)
        if mask.sum() < 50:
            return {}

        # Correlation: higher prob (bearish) should correlate with negative returns
        corr = float(np.corrcoef(probs[mask], fwd[mask])[0, 1])

        # Directional accuracy: prob > 0.5 predicts down (negative return)
        pred_down = (probs[mask] > 0.5).astype(int)
        actual_down = (fwd[mask] < 0).astype(int)
        dir_acc = float(accuracy_score(actual_down, pred_down))

        # Returns by probability bucket
        buckets = pd.qcut(probs[mask], q=5, labels=False, duplicates='drop')
        bucket_returns = {}
        for b in range(5):
            bucket_mask = (buckets == b)
            if bucket_mask.sum() > 0:
                bucket_returns[f"bucket_{b}"] = float(np.mean(fwd[mask][bucket_mask]))

        return {
            "horizon": horizon,
            "n_samples": int(mask.sum()),
            "mean_fwd_return": float(np.mean(fwd[mask])),
            "std_fwd_return": float(np.std(fwd[mask])),
            "corr_prob_fwd": corr,
            "directional_accuracy": dir_acc,
            "bucket_returns": bucket_returns,
        }

    def _save_artifacts_and_reports(
        self,
        output_df: pd.DataFrame,
        X_all: pd.DataFrame,
        y_target: pd.Series,
        y_teacher: pd.Series,
        model: Optional[xgb.XGBClassifier],
        calibrated_model: Optional[CalibratedClassifierCV],
        teacher_metrics: Dict[str, Any],
        student_metrics: Dict[str, Any],
        fwd_metrics: Dict[Any, Any],
        split_config: TemporalSplitConfig,
        symbol: str,
        exchange: str,
        timeframe: str,
        market_source: str,
        oof_metadata: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Save artifacts and generate comprehensive reports with improved diagnostics and burn-in metadata."""
        artifacts: Dict[str, str] = {}
        reports: Dict[str, str] = {}

        # Use current context direction (long/short) for naming and grid behaviour
        direction = str(self._current_context.get("direction", "long"))
        suffix = f"_{direction}" if direction in {"long", "short"} else ""

        # Save training data with all scores
        to_save = output_df[[
            "mr_teacher_cluster",
            "mr_teacher_mean_reversion",
            "mr_teacher_score",
            "mr_raw_score",
            "mr_probability",
            "mr_probability_dense",
            "mr_direction_target",
        ]].copy()
        to_save = to_save.reset_index().rename(columns={output_df.index.name or "index": "timestamp"})
        try:
            # Prepare metadata with temporal split information
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "source_market_data": market_source,
                "version": "v2_classification_with_burnin",
                "training_start": str(split_config.training.start),
                "training_end": str(split_config.training.effective_end),
                "validation_start": str(split_config.validation.start),
                "validation_end": str(split_config.validation.effective_end),
                "test_start": str(split_config.test.start),
                "test_end": str(split_config.test.effective_end),
                "prediction_method": "oof" if oof_metadata else "traditional",
                "oof_windows": len(oof_metadata) if oof_metadata else 0,
                "hpo_runs": sum(1 for m in oof_metadata if m.get('used_hpo', False)) if oof_metadata else 0,
                "retrain_interval_days": 10,
                "hpo_interval_days": 30,
            }
            if split_config.burnin is not None:
                metadata["burnin_start"] = str(split_config.burnin.start)
                metadata["burnin_end"] = str(split_config.burnin.effective_end)

            artifacts["training_data"] = self._save_artifact(
                data=to_save,
                artifact_name=f"ml_mean_reversion_training_data_{timeframe}",
                artifact_type="data",
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save training data artifact: {exc}")

        # Save base XGB model
        try:
            artifacts["model_base"] = self._save_artifact(
                data=model,
                artifact_name=f"ml_mean_reversion_model_base_{timeframe}{suffix}",
                artifact_type="model",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "model_type": "xgboost_classifier",
                    "version": "v2"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save base model artifact: {exc}")

        # Save calibrated model
        try:
            artifacts["model_calibrated"] = self._save_artifact(
                data=calibrated_model,
                artifact_name=f"ml_mean_reversion_model_calibrated_{timeframe}{suffix}",
                artifact_type="model",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "model_type": "calibrated_classifier",
                    "calibration_method": student_metrics.get("calibration_method", "isotonic"),
                    "version": "v2"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save calibrated model artifact: {exc}")

        # Save metrics
        # Enrich student metrics with OOF-based classification metrics if available.
        try:
            if "mr_probability" in output_df.columns and "mr_direction_target" in output_df.columns:
                prob_series_full = output_df["mr_probability"].astype(float)
                target_series_full = output_df["mr_direction_target"].astype(float)

                # Use only finite probabilities with known targets
                base_mask = np.isfinite(prob_series_full.values) & np.isfinite(target_series_full.values)

                # Restrict to true OOF samples when available
                if "mr_is_oof" in output_df.columns:
                    base_mask &= output_df["mr_is_oof"].astype(bool).values

                if bool(base_mask.any()):
                    idx_all = output_df.index[base_mask]
                    y_all = target_series_full.loc[idx_all].astype(int)
                    p_cal_all = prob_series_full.loc[idx_all]

                    if "mr_raw_score" in output_df.columns:
                        raw_series_full = output_df["mr_raw_score"].astype(float)
                        p_raw_all = raw_series_full.loc[idx_all]
                    else:
                        p_raw_all = p_cal_all

                    def _split_indices(start_ts: datetime, end_ts: datetime) -> pd.DatetimeIndex:
                        if split_config is None:
                            return idx_all
                        return idx_all[(idx_all >= start_ts) & (idx_all <= end_ts)]

                    def _compute_split_metrics(idx_subset: pd.DatetimeIndex, p_all: pd.Series) -> Dict[str, float]:
                        if idx_subset is None or len(idx_subset) < 50:
                            return {}
                        y_true = y_all.loc[idx_subset]
                        p = p_all.loc[idx_subset]
                        if y_true.nunique() < 2:
                            return {}
                        y_pred = (p >= 0.5).astype(int)
                        m: Dict[str, float] = {
                            "acc": float(accuracy_score(y_true, y_pred)),
                            "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
                            "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
                            "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
                        }
                        try:
                            m["auc"] = float(roc_auc_score(y_true, p))
                        except ValueError:
                            m["auc"] = float("nan")
                        try:
                            m["logloss"] = float(log_loss(y_true, p))
                        except ValueError:
                            m["logloss"] = float("nan")
                        return m

                    if split_config is not None:
                        train_idx = _split_indices(split_config.training.start, split_config.training.effective_end)
                        val_idx = _split_indices(split_config.validation.start, split_config.validation.effective_end)
                        test_idx = _split_indices(split_config.test.start, split_config.test.effective_end)
                    else:
                        train_idx = idx_all
                        val_idx = idx_all
                        test_idx = idx_all

                    # Raw metrics (from mr_raw_score when available)
                    train_raw = _compute_split_metrics(train_idx, p_raw_all)
                    val_raw = _compute_split_metrics(val_idx, p_raw_all)
                    test_raw = _compute_split_metrics(test_idx, p_raw_all)

                    if train_raw:
                        student_metrics["train_raw"] = train_raw
                    if val_raw:
                        student_metrics["val_raw"] = val_raw
                    if test_raw:
                        student_metrics["test_raw"] = test_raw

                    # Calibrated metrics (from mr_probability)
                    train_cal = _compute_split_metrics(train_idx, p_cal_all)
                    val_cal = _compute_split_metrics(val_idx, p_cal_all)
                    test_cal = _compute_split_metrics(test_idx, p_cal_all)

                    if train_cal:
                        student_metrics["train_calibrated"] = train_cal
                    if val_cal:
                        student_metrics["val_calibrated"] = val_cal
                    if test_cal:
                        student_metrics["test_calibrated"] = test_cal

                    # Class balance and split sizes
                    cb = student_metrics.get("class_balance", {})
                    if len(train_idx) > 0:
                        cb["train_pos_rate"] = float(y_all.loc[train_idx].mean())
                    if len(val_idx) > 0:
                        cb["val_pos_rate"] = float(y_all.loc[val_idx].mean())
                    if len(test_idx) > 0:
                        cb["test_pos_rate"] = float(y_all.loc[test_idx].mean())
                    if cb:
                        student_metrics["class_balance"] = cb

                    split_sizes = student_metrics.get("split_sizes", {})
                    split_sizes["train"] = int(len(train_idx))
                    split_sizes["val"] = int(len(val_idx))
                    split_sizes["test"] = int(len(test_idx))
                    student_metrics["split_sizes"] = split_sizes

                    # Mark calibration method to avoid misleading defaults
                    student_metrics.setdefault("calibration_method", "oof")
        except Exception as metrics_exc:  # noqa: BLE001
            tprint_warning(f"Failed to compute OOF classification metrics: {metrics_exc}")

        try:
            artifacts["metrics"] = self._save_artifact(
                data={
                    "teacher": teacher_metrics,
                    "student": student_metrics,
                    "forward": fwd_metrics,
                },
                artifact_name=f"ml_mean_reversion_metrics_{timeframe}{suffix}",
                artifact_type="metadata",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "version": "v2"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save metrics artifact: {exc}")

        # Also export key metrics to a flat CSV in outcomes/ for easier inspection.
        try:
            flat_metrics: Dict[str, Any] = {}

            # Teacher metrics (top-level scalars)
            for k, v in teacher_metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    flat_metrics[f"teacher_{k}"] = v

            thresholds = teacher_metrics.get("thresholds", {})
            if isinstance(thresholds, dict):
                for k, v in thresholds.items():
                    if isinstance(v, (int, float, str, bool)):
                        flat_metrics[f"teacher_thresholds_{k}"] = v

            # Student metrics (top-level scalars)
            for k, v in student_metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    flat_metrics[f"student_{k}"] = v

            # Nested student metrics: raw and calibrated splits
            for split_key in [
                "train_raw",
                "val_raw",
                "test_raw",
                "train_calibrated",
                "val_calibrated",
                "test_calibrated",
            ]:
                split_metrics = student_metrics.get(split_key, {})
                if isinstance(split_metrics, dict):
                    for mk, mv in split_metrics.items():
                        if isinstance(mv, (int, float, str, bool)):
                            flat_metrics[f"student_{split_key}_{mk}"] = mv

            class_balance = student_metrics.get("class_balance", {})
            if isinstance(class_balance, dict):
                for mk, mv in class_balance.items():
                    if isinstance(mv, (int, float, str, bool)):
                        flat_metrics[f"student_class_balance_{mk}"] = mv

            split_sizes = student_metrics.get("split_sizes", {})
            if isinstance(split_sizes, dict):
                for mk, mv in split_sizes.items():
                    if isinstance(mv, (int, float, str, bool)):
                        flat_metrics[f"student_split_sizes_{mk}"] = mv

            # Forward-return diagnostics, keyed by horizon
            for h, m in fwd_metrics.items():
                if isinstance(m, dict):
                    for mk, mv in m.items():
                        if isinstance(mv, (int, float, str, bool)):
                            flat_metrics[f"forward_h{h}_{mk}"] = mv

            # Basic identifiers
            flat_metrics["symbol"] = symbol
            flat_metrics["exchange"] = exchange
            flat_metrics["timeframe"] = timeframe
            flat_metrics["direction"] = direction
            flat_metrics["created_at"] = datetime.now().isoformat()

            metrics_df = pd.DataFrame([flat_metrics])
            ts_metrics = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"outcomes/ml_mean_reversion_metrics_{symbol}_{timeframe}_{direction}_{ts_metrics}.csv"
            metrics_df.to_csv(csv_path, index=False)
            tprint_info(f"  💾 Saved metrics CSV to {csv_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save metrics CSV: {exc}")

        # Generate comprehensive Markdown report
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            md_path = f"outcomes/ml_mean_reversion_summary_{symbol}_{timeframe}_{direction}_{ts}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# ML Mean-Reversion (v2) Summary for {symbol} ({timeframe})\n\n")
                f.write("**Model Type**: XGBoost Classifier with Isotonic Calibration\n")
                f.write("**Target**: Directional (0=up, 1=down)\n")
                f.write("**Version**: v2 with relaxed thresholds, enhanced features, and proper calibration\n\n")

                # Teacher metrics
                f.write("## Teacher (OU/Hurst GMM) - IMPROVED\n\n")
                f.write(f"- Components: {teacher_metrics.get('n_components')}\n")
                f.write(f"- Mean-reversion cluster: {teacher_metrics.get('mean_reversion_cluster')}\n")
                f.write(f"- Cluster counts: {teacher_metrics.get('cluster_counts')}\n")
                thresholds = teacher_metrics.get('thresholds', {})
                f.write(f"- Thresholds (RELAXED for 15m):\n")
                f.write(f"  - Hurst: {thresholds.get('hurst', 'N/A')}\n")
                f.write(f"  - Half-life: {thresholds.get('half_life', 'N/A')} bars\n")
                f.write(f"  - ADF p-value: {thresholds.get('adf_p', 'N/A')}\n")
                f.write(f"  - Variance ratio: {thresholds.get('variance_ratio', 'N/A')}\n")
                f.write(f"- **Teacher positive rate: {teacher_metrics.get('teacher_positive_rate', 0.0):.4f}** (IMPROVED from ~0.0)\n\n")

                # Student metrics
                f.write("## Student (XGB Classifier) - RAW vs CALIBRATED\n\n")

                f.write("### Raw Model Performance\n\n")
                for split in ["train", "val", "test"]:
                    m = student_metrics.get(f"{split}_raw", {})
                    f.write(f"**{split.upper()}**: ")
                    f.write(f"ACC={m.get('acc', 0):.4f}, ")
                    f.write(f"F1={m.get('f1', 0):.4f}, ")
                    f.write(f"Precision={m.get('precision', 0):.4f}, ")
                    f.write(f"Recall={m.get('recall', 0):.4f}, ")
                    f.write(f"AUC={m.get('auc', 0):.4f}, ")
                    f.write(f"LogLoss={m.get('logloss', 0):.4f}\n")
                f.write("\n")

                f.write("### Calibrated Model Performance\n\n")
                for split in ["train", "val", "test"]:
                    m = student_metrics.get(f"{split}_calibrated", {})
                    f.write(f"**{split.upper()}**: ")
                    f.write(f"ACC={m.get('acc', 0):.4f}, ")
                    f.write(f"F1={m.get('f1', 0):.4f}, ")
                    f.write(f"Precision={m.get('precision', 0):.4f}, ")
                    f.write(f"Recall={m.get('recall', 0):.4f}, ")
                    f.write(f"AUC={m.get('auc', 0):.4f}, ")
                    f.write(f"LogLoss={m.get('logloss', 0):.4f}\n")
                f.write("\n")

                # Class balance
                f.write("### Class Balance\n\n")
                cb = student_metrics.get("class_balance", {})
                f.write(f"- Train positive rate (bearish): {cb.get('train_pos_rate', 0):.4f}\n")
                f.write(f"- Val positive rate (bearish): {cb.get('val_pos_rate', 0):.4f}\n")
                f.write(f"- Test positive rate (bearish): {cb.get('test_pos_rate', 0):.4f}\n\n")

                # Walk-forward stability
                wf = student_metrics.get("walkforward")
                if isinstance(wf, dict) and wf.get("folds", 0) > 0:
                    f.write("## Walk-Forward Stability (OOF Calibrated)\n\n")
                    f.write(f"- Folds: {wf.get('folds')}\n")
                    f.write(f"- **ACC**: mean={wf.get('acc_mean', 0):.4f}, std={wf.get('acc_std', 0):.4f}\n")
                    f.write(f"- **F1**: mean={wf.get('f1_mean', 0):.4f}, std={wf.get('f1_std', 0):.4f}\n")
                    f.write(f"- **AUC**: mean={wf.get('auc_mean', 0):.4f}, std={wf.get('auc_std', 0):.4f}\n")
                    f.write(f"- **LogLoss**: mean={wf.get('logloss_mean', 0):.4f}, std={wf.get('logloss_std', 0):.4f}\n\n")

                # Forward diagnostics
                if fwd_metrics:
                    f.write("## Forward-Return Diagnostics\n\n")
                    for h, m in sorted(fwd_metrics.items()):
                        f.write(f"### Horizon {h} bars ({h * 15} minutes)\n\n")
                        f.write(f"- n_samples: {m.get('n_samples')}\n")
                        f.write(f"- mean_fwd_return: {m.get('mean_fwd_return', 0):.6f}\n")
                        f.write(f"- std_fwd_return: {m.get('std_fwd_return', 0):.6f}\n")
                        f.write(f"- **corr_prob_fwd**: {m.get('corr_prob_fwd', 0):.4f} (negative = good, higher prob → lower returns)\n")
                        f.write(f"- **directional_accuracy**: {m.get('directional_accuracy', 0):.4f}\n")

                        bucket_returns = m.get('bucket_returns', {})
                        if bucket_returns:
                            f.write(f"- Returns by probability bucket:\n")
                            for bucket, ret in sorted(bucket_returns.items()):
                                f.write(f"  - {bucket}: {ret:.6f}\n")
                        f.write("\n")

                # Signal statistics
                f.write("## Signal Statistics\n\n")
                prob_series = output_df.loc[X_all.index, "mr_probability"]
                raw_series = output_df.loc[X_all.index, "mr_raw_score"]
                target_series = y_target

                bullish_rate = float((prob_series < 0.4).mean())
                neutral_rate = float(((prob_series >= 0.4) & (prob_series <= 0.6)).mean())
                bearish_rate = float((prob_series > 0.6).mean())

                f.write(f"- Bullish signals (prob < 0.4): {bullish_rate:.4f}\n")
                f.write(f"- Neutral signals (0.4 ≤ prob ≤ 0.6): {neutral_rate:.4f}\n")
                f.write(f"- Bearish signals (prob > 0.6): {bearish_rate:.4f}\n")
                f.write(f"- Mean calibrated probability: {prob_series.mean():.4f}\n")
                f.write(f"- Std calibrated probability: {prob_series.std():.4f}\n\n")

                # Feature importance
                try:
                    f.write("## Top 15 Feature Importances\n\n")
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:15]
                    for i, idx in enumerate(indices):
                        col_name = X_all.columns[idx]
                        imp = importances[idx]
                        f.write(f"{i+1}. {col_name}: {imp:.4f}\n")
                    f.write("\n")
                except Exception:
                    pass

            reports["markdown"] = md_path
            tprint_success(f"✅ Saved markdown report: {md_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to write Markdown report: {exc}")

        # Save probabilities CSV
        try:
            idx = X_all.index
            # Align all series explicitly to the feature index to guarantee equal lengths
            output_aligned = output_df.reindex(idx)
            y_aligned = y_target.reindex(idx)
            csv_df = pd.DataFrame(
                {
                    "timestamp": idx.to_numpy(),
                    "mr_teacher_score": output_aligned["mr_teacher_score"].to_numpy(),
                    "mr_raw_score": output_aligned["mr_raw_score"].to_numpy(),
                    "mr_probability": output_aligned["mr_probability"].to_numpy(),
                    "mr_direction_target": y_aligned.to_numpy(),
                    "close": output_aligned["close"].to_numpy(),
                }
            )
            csv_path = f"outcomes/ml_mean_reversion_probabilities_{symbol}_{timeframe}_{direction}_{ts}.csv"
            csv_df.to_csv(csv_path, index=False)
            reports["probabilities_csv"] = csv_path
            tprint_success(f"✅ Saved probabilities CSV: {csv_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to write probabilities CSV: {exc}")

        # Grid backtest with SIMPLIFIED signal generation
        try:
            idx = X_all.index
            # Align all series (teacher labels, targets, features) to the
            # same feature index used for probabilities.
            output_aligned = output_df.reindex(idx)
            y_teacher_aligned = y_teacher.reindex(idx)
            y_target_aligned = y_target.reindex(idx)

            # Use a clean RangeIndex for grid simulation to avoid duplicate-label
            # reindex issues inside the grid backtester while preserving
            # temporal ordering.
            grid_index = pd.RangeIndex(len(idx))

            close = pd.Series(
                output_aligned["close"].astype(float).to_numpy(), index=grid_index
            )
            high = pd.Series(
                output_aligned["high"].astype(float).to_numpy(), index=grid_index
            )
            low = pd.Series(
                output_aligned["low"].astype(float).to_numpy(), index=grid_index
            )
            raw_returns = close.pct_change().fillna(0.0)

            prob = pd.Series(
                output_aligned["mr_probability"].astype(float).to_numpy(),
                index=grid_index,
            )

            # SIMPLIFIED: Use continuous probability directly.
            # For long strategy:
            #   - High bearish prob (close to 1) = avoid/short
            #   - Low bearish prob (close to 0) = strong long signal
            #   → long_confidence = 1 - prob
            # For short strategy (reverse grid):
            #   - High bearish prob (close to 1) = strong short signal
            #   → short_confidence = prob
            z_ma = pd.Series(
                output_aligned["z_price_ma_slow"].astype(float).to_numpy(),
                index=grid_index,
            )
            z_vwap = pd.Series(
                output_aligned["z_price_vwap"].astype(float).to_numpy(),
                index=grid_index,
            )

            if direction == "short":
                base_confidence = prob
                # Boost confidence when *strongly* overbought (above mean) for
                # mean-reversion shorts. Use stricter AND condition and a
                # slightly larger z-threshold by default.
                short_z_thr = float(config.get("mr_short_overbought_z_threshold", 0.02))
                short_boost = float(config.get("mr_short_confidence_boost", 0.5))
                overbought = ((z_ma > short_z_thr) & (z_vwap > short_z_thr)).astype(float)
                confidence_boost = 1.0 + overbought * short_boost
                preds = -base_confidence * confidence_boost
                preds = preds.clip(-1, 0)
                grid_confidence = base_confidence
                grid_fn = run_simple_short_grid_backtest
            else:
                base_confidence = 1.0 - prob
                # Boost confidence when clearly oversold (below mean) for
                # mean-reversion longs. Use symmetric AND condition.
                long_z_thr = float(config.get("mr_long_oversold_z_threshold", -0.02))
                long_boost = float(config.get("mr_long_confidence_boost", 0.5))
                oversold = ((z_ma < long_z_thr) & (z_vwap < long_z_thr)).astype(float)
                confidence_boost = 1.0 + oversold * long_boost
                preds = base_confidence * confidence_boost
                preds = preds.clip(0, 1)
                grid_confidence = base_confidence
                grid_fn = run_simple_long_grid_backtest

            ml_df_grid = pd.DataFrame(
                {
                    "mr_teacher_mean_reversion": y_teacher_aligned.astype(int).to_numpy(),
                    "mr_teacher_score": output_aligned["mr_teacher_score"]
                    .astype(float)
                    .to_numpy(),
                    "mr_probability": prob.to_numpy(),
                    "mr_direction_target": y_target_aligned.astype(int).to_numpy(),
                },
                index=grid_index,
            )

            # Attempt to load meta-labeling HPO parameters and apply dynamic ATR multiplier
            tp_override = None
            sl_override = None
            try:
                from pathlib import Path
                import json

                base_dir = Path("outcomes")
                hpo_pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
                hpo_candidates = sorted(base_dir.glob(hpo_pattern))
                if not hpo_candidates:
                    fallback_pattern = f"meta_labeling_hpo_best_params_{symbol}_*_*.json"
                    hpo_candidates = sorted(base_dir.glob(fallback_pattern))
                if hpo_candidates:
                    hpo_path = hpo_candidates[-1]
                    with open(hpo_path, "r", encoding="utf-8") as f_hpo:
                        hpo_cfg = json.load(f_hpo)
                    params = {}
                    if isinstance(hpo_cfg, dict):
                        knee = hpo_cfg.get("knee_params")
                        best = hpo_cfg.get("best_params")
                        if isinstance(knee, dict) and knee:
                            params = knee
                        elif isinstance(best, dict) and best:
                            params = best
                    profit_thr = float(params.get("profit_thr_base")) if params.get("profit_thr_base") is not None else None
                    stop_ratio = float(params.get("stop_to_profit_ratio")) if params.get("stop_to_profit_ratio") is not None else None
                    if profit_thr is not None and stop_ratio is not None:
                        # Base TPSL values
                        tp_base = max(0.0005, profit_thr)
                        sl_base = max(0.0005, profit_thr * stop_ratio)

                        # Apply dynamic ATR multiplier (use mean multiplier across the test period)
                        # Get multiplier for the backtest period
                        multiplier_series = output_df.loc[idx, "mr_dynamic_tpsl_multiplier"].astype(float)
                        mean_multiplier = float(multiplier_series.mean())

                        # Apply multiplier to base TPSL
                        tp_override = tp_base * mean_multiplier
                        sl_override = sl_base * mean_multiplier

                        tprint_info(
                            f"📊 Dynamic TPSL: Base TP={tp_base*100:.3f}%, SL={sl_base*100:.3f}% | "
                            f"Multiplier={mean_multiplier:.3f} | "
                            f"Adjusted TP={tp_override*100:.3f}%, SL={sl_override*100:.3f}%"
                        )
            except Exception as e:
                tprint_warning(f"Failed to load HPO params or apply ATR multiplier: {e}")
                tp_override = None
                sl_override = None

            local_config: Dict[str, Any] = config or {}
            dir_suffix = "_short" if direction == "short" else "_long"

            # Optionally constrain TPSL overrides to a configurable band around
            # average ATR_14-relative volatility so that profit targets and
            # stops remain in a reasonable range relative to recent price
            # movement and trading fees.
            if tp_override is not None and sl_override is not None and "mr_atr_14" in output_df.columns:
                try:
                    atr_14_series = output_df.loc[idx, "mr_atr_14"].astype(float)
                    close_series = close
                    atr_rel = (atr_14_series / close_series).replace([np.inf, -np.inf], np.nan)
                    avg_atr_rel = float(atr_rel.mean())
                except Exception:
                    avg_atr_rel = None

                if avg_atr_rel is not None and avg_atr_rel > 0:
                    tp_min_mult = float(local_config.get("mr_tp_min_atr_mult", 1.0))
                    tp_max_mult = float(local_config.get("mr_tp_max_atr_mult", 3.0))
                    sl_min_mult = float(local_config.get("mr_sl_min_atr_mult", 0.5))
                    sl_max_mult = float(local_config.get("mr_sl_max_atr_mult", 2.0))

                    tp_floor = tp_min_mult * avg_atr_rel
                    tp_cap = tp_max_mult * avg_atr_rel
                    sl_floor = sl_min_mult * avg_atr_rel
                    sl_cap = sl_max_mult * avg_atr_rel

                    tp_override = max(tp_floor, min(tp_override, tp_cap))
                    sl_override = max(sl_floor, min(sl_override, sl_cap))

            holding_candidates = (
                local_config.get(f"mr_grid_max_holding_bars_list{dir_suffix}")
                or local_config.get("mr_grid_max_holding_bars_list")
            )
            if holding_candidates is None:
                base_default = int(local_config.get("mr_grid_max_holding_bars_default", 6))
                base_dir = int(local_config.get(f"mr_grid_max_holding_bars_default{dir_suffix}", base_default))
                base_hold = max(1, base_dir)
                # Use symmetric holding-period candidates for long and short by
                # default so that trade intensity is comparable across sides.
                holding_candidates = [base_hold, base_hold * 2, base_hold * 4]

            try:
                holds_unique = sorted(
                    {int(h) for h in holding_candidates if h is not None and int(h) > 0}
                )
            except Exception:
                holds_unique = [6]

            grid_frames: List[pd.DataFrame] = []
            for max_hold in holds_unique:
                if tp_override is not None and sl_override is not None:
                    grid_local = grid_fn(
                        close=close,
                        high=high,
                        low=low,
                        raw_returns=raw_returns,
                        predictions=preds,
                        confidence=grid_confidence,
                        ml_df=ml_df_grid,
                        timeframe=timeframe,
                        regime_col="mr_teacher_mean_reversion",
                        tp_values=[tp_override],
                        sl_values=[sl_override],
                        max_holding_bars=max_hold,
                    )
                else:
                    grid_local = grid_fn(
                        close=close,
                        high=high,
                        low=low,
                        raw_returns=raw_returns,
                        predictions=preds,
                        confidence=grid_confidence,
                        ml_df=ml_df_grid,
                        timeframe=timeframe,
                        regime_col="mr_teacher_mean_reversion",
                        max_holding_bars=max_hold,
                    )

                if isinstance(grid_local, pd.DataFrame) and not grid_local.empty:
                    grid_local = grid_local.copy()
                    grid_local["max_holding_bars"] = int(max_hold)
                    if "grid_config" in grid_local.columns:
                        grid_local["grid_config"] = (
                            grid_local["grid_config"].astype(str) + f",max_hold={max_hold}"
                        )
                    grid_frames.append(grid_local)

            grid_df_all: Optional[pd.DataFrame]
            if grid_frames:
                grid_df_all = pd.concat(grid_frames, ignore_index=True)
            else:
                grid_df_all = None

            best_summary = None
            if isinstance(grid_df_all, pd.DataFrame) and not grid_df_all.empty:
                grid_path = f"outcomes/ml_mean_reversion_grid_backtest_{symbol}_{timeframe}_{direction}_{ts}.csv"
                tprint_info(
                    f"Writing grid backtest CSV with shape={grid_df_all.shape} to {grid_path}"
                )
                grid_df_all.to_csv(grid_path, index=False)
                reports["grid_backtest_csv"] = grid_path
                tprint_success(f"✅ Saved grid backtest CSV: {grid_path}")

                try:
                    candidates = grid_df_all.copy()
                    min_trades = int(
                        local_config.get(
                            f"mr_min_grid_trades{dir_suffix}",
                            local_config.get("mr_min_grid_trades", 30),
                        )
                    )
                    min_bars = int(
                        local_config.get(
                            f"mr_min_grid_bars{dir_suffix}",
                            local_config.get("mr_min_grid_bars", 200),
                        )
                    )
                    if "number_of_trades" in candidates.columns:
                        candidates = candidates[candidates["number_of_trades"] >= min_trades]
                    if "bars" in candidates.columns:
                        candidates = candidates[candidates["bars"] >= min_bars]

                    # Optional trade-intensity band based on avg_trades_per_day
                    # when configured. This allows constraining the selected
                    # grid to a desired activity level without hard-coding a
                    # specific range here.
                    max_tpd_cfg = local_config.get(
                        f"mr_max_grid_trades_per_day{dir_suffix}",
                        local_config.get("mr_max_grid_trades_per_day"),
                    )
                    min_tpd_cfg = local_config.get(
                        f"mr_min_grid_trades_per_day{dir_suffix}",
                        local_config.get("mr_min_grid_trades_per_day"),
                    )
                    if "avg_trades_per_day" in candidates.columns:
                        if min_tpd_cfg is not None:
                            try:
                                min_tpd = float(min_tpd_cfg)
                                candidates = candidates[candidates["avg_trades_per_day"] >= min_tpd]
                            except Exception:
                                pass
                        if max_tpd_cfg is not None:
                            try:
                                max_tpd = float(max_tpd_cfg)
                                candidates = candidates[candidates["avg_trades_per_day"] <= max_tpd]
                            except Exception:
                                pass

                    if not candidates.empty:
                        sort_cols = [
                            "sharpe_ratio_with_fees",
                            "calmar_ratio_with_fees",
                            "strategy_total_return_with_fees_%",
                        ]
                        available_cols = [c for c in sort_cols if c in candidates.columns]
                        if available_cols:
                            candidates = candidates.sort_values(
                                by=available_cols,
                                ascending=[False] * len(available_cols),
                            )
                            best_row = candidates.iloc[0]
                            best_summary = {
                                "grid_config": best_row.get("grid_config"),
                                "take_profit_pct": float(best_row.get("take_profit_pct", 0.0)),
                                "stop_loss_pct": float(best_row.get("stop_loss_pct", 0.0)),
                                "confidence_quantile": float(best_row.get("confidence_quantile", 0.0)),
                                "max_holding_bars": int(best_row.get("max_holding_bars", 0)),
                                "sharpe_ratio_with_fees": float(
                                    best_row.get("sharpe_ratio_with_fees", 0.0)
                                ),
                                "calmar_ratio_with_fees": float(
                                    best_row.get("calmar_ratio_with_fees", 0.0)
                                ),
                                "strategy_total_return_with_fees_%": float(
                                    best_row.get("strategy_total_return_with_fees_%", 0.0)
                                ),
                                "number_of_trades": int(best_row.get("number_of_trades", 0)),
                            }
                            tprint_info(
                                "✨ Best fee-aware grid config: "
                                f"{best_summary['grid_config']} | "
                                f"TP={best_summary['take_profit_pct']*100:.3f}% "
                                f"SL={best_summary['stop_loss_pct']*100:.3f}% | "
                                f"Sharpe_with_fees={best_summary['sharpe_ratio_with_fees']:.3f}, "
                                f"Calmar_with_fees={best_summary['calmar_ratio_with_fees']:.3f}, "
                                f"Total_return_with_fees={best_summary['strategy_total_return_with_fees_%']:.2f}% "
                                f"(trades={best_summary['number_of_trades']}, "
                                f"max_hold={best_summary['max_holding_bars']} bars)"
                            )
                except Exception as sel_exc:
                    tprint_warning(f"Failed to select best grid configuration: {sel_exc}")

                if best_summary is not None:
                    try:
                        artifacts["grid_backtest_best"] = self._save_artifact(
                            data=best_summary,
                            artifact_name=(
                                f"ml_mean_reversion_grid_best_with_fees_{symbol}_{timeframe}_{direction}"
                            ),
                            artifact_type="metadata",
                            metadata={
                                "symbol": symbol,
                                "exchange": exchange,
                                "timeframe": timeframe,
                                "direction": direction,
                            },
                        )
                    except Exception as exc:
                        tprint_warning(f"Failed to save best grid config artifact: {exc}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to run/write grid backtest: {exc}")

        return artifacts, reports

    async def run_config_batch(self, configs: List[Dict[str, Any]], symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""
        results = []
        total_configs = len(configs)

        for i, base_config in enumerate(configs):
            # Ensure proper HPO control flags are set for the sweep
            config = dict(base_config)

            # Use cached market data where possible to speed up repeated runs
            config.setdefault("use_cached_market_data", True)

            # Setup logging for this trial
            config_sig = self.get_config_signature(config)
            tprint_info(f"🚀 Running config {i+1}/{total_configs}: {config_sig}")

            try:
                # Run the step with this configuration
                start_time = time.time()

                # Execute step
                result = await self.execute(config)

                execution_time = time.time() - start_time

                if not result.get("success", False):
                    tprint_warning(f"⚠️ Config {i+1} failed: {result.get('error')}")
                    results.append({
                        "config_signature": config_sig,
                        "config_id": i + 1,
                        "execution_time": execution_time,
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                    })
                    continue

                # Extract key metrics
                metrics = result.get("metrics", {})
                student_metrics = metrics.get("student", {})
                teacher_metrics = metrics.get("teacher", {})
                forward_metrics = metrics.get("forward", {})

                # We prioritize OOF (test) metrics if available, else standard test split
                test_cal = student_metrics.get("test_calibrated", {})

                trial_result = {
                    "config_signature": config_sig,
                    "config_id": i + 1,
                    "execution_time": execution_time,
                    "success": True,

                    # Primary ranking metrics (OOF/Test performance)
                    "student_test_auc": float(test_cal.get("auc", 0.0) or 0.0),
                    "student_test_acc": float(test_cal.get("acc", 0.0) or 0.0),
                    "student_test_logloss": float(test_cal.get("logloss", 0.0) or 0.0),

                    # Teacher stats
                    "teacher_positive_rate": float(teacher_metrics.get("teacher_positive_rate", 0.0) or 0.0),
                    "n_regimes": teacher_metrics.get("n_components"),
                    "mean_reversion_cluster": teacher_metrics.get("mean_reversion_cluster"),

                    # Forward Analysis (using horizon 4 or 12 as available)
                    "fwd_dir_acc": 0.0,
                    "fwd_corr": 0.0,
                }

                # Populate forward metrics (prefer horizon 12, then 8, then 4)
                for h in [12, 8, 4]:
                    if h in forward_metrics:
                        fm = forward_metrics[h]
                        trial_result["fwd_dir_acc"] = float(fm.get("directional_accuracy", 0.0) or 0.0)
                        trial_result["fwd_corr"] = float(fm.get("corr_prob_fwd", 0.0) or 0.0)
                        break

                # Add configuration details
                # Capture all mr_ keys that are not huge objects
                trial_result.update({
                    f"config_{k}": v for k, v in config.items()
                    if k.startswith("mr_") and not callable(v) and not isinstance(v, (list, dict))
                })

                results.append(trial_result)

                tprint_info(
                    f"✅ Config {i+1} done: AUC={trial_result['student_test_auc']:.4f}, "
                    f"ACC={trial_result['student_test_acc']:.4f}, "
                    f"TeacherRate={trial_result['teacher_positive_rate']:.4f}"
                )

            except Exception as e:
                tprint_error(f"❌ Config {i+1} crashed: {e}")
                results.append({
                    "config_signature": config_sig,
                    "config_id": i + 1,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e),
                })

        return results

    def get_config_signature(self, config: Dict[str, Any]) -> str:
        """Generate a compact signature for configuration identification."""
        key_params = [
            "mr_hurst_threshold",
            "mr_half_life_threshold",
            "mr_adf_p_threshold",
            "mr_ma_fast_window",
            "mr_ma_slow_window",
            "mr_rsi_window",
            "mr_enable_hpo"
        ]

        parts = []
        for param in key_params:
            if param in config:
                val = config[param]
                # Shorten key name
                short_key = param.replace("mr_", "").replace("threshold", "thr").replace("window", "win")
                parts.append(f"{short_key}={val}")

        if not parts:
            return "default_config"

        return "|".join(parts)

    def analyze_and_rank_results(self, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze results and rank configurations by quality."""

        if not results:
            return pd.DataFrame(), {}

        df = pd.DataFrame(results)

        # Filter successful runs
        successful = df[df["success"] == True].copy()
        failed = df[df["success"] == False].copy()

        tprint_info(f"📊 Analysis: {len(successful)} successful, {len(failed)} failed runs")

        if len(successful) == 0:
            tprint_warning("⚠️ No successful configurations to analyze")
            return df, {"best_config": None, "analysis": "no_successful_runs"}

        # Calculate Composite Score
        # Priority:
        # 1. AUC (discrimination capability) - 50%
        # 2. Forward Directional Accuracy (real-world predictive power) - 30%
        # 3. Test Accuracy (binary classification correctness) - 20%
        # Penalty: Teacher Positive Rate < 0.01 (too rare) or > 0.5 (too common)

        def calculate_score(row):
            auc = row.get("student_test_auc", 0.5)
            fwd_acc = row.get("fwd_dir_acc", 0.5)
            acc = row.get("student_test_acc", 0.5)
            tp_rate = row.get("teacher_positive_rate", 0.0)

            base_score = (0.5 * auc) + (0.3 * fwd_acc) + (0.2 * acc)

            # Penalize extreme teacher rates (regime is either non-existent or trivial)
            penalty = 1.0
            if tp_rate < 0.01 or tp_rate > 0.6:
                penalty = 0.8
            if tp_rate < 0.001:
                penalty = 0.5

            return base_score * penalty

        successful["composite_score"] = successful.apply(calculate_score, axis=1)

        # Sort by composite score
        successful = successful.sort_values("composite_score", ascending=False)

        # Get best configuration
        best_row = successful.iloc[0]
        best_config = best_row.to_dict()

        # Analysis summary
        analysis = {
            "best_config": best_config,
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "best_composite_score": float(best_row["composite_score"]),
            "best_auc": float(best_row.get("student_test_auc", 0)),
            "top_5_configs": successful.head(5).to_dict("records"),
        }

        # Helper to print summary
        print("\n" + "="*80)
        print("🏆 MEAN REVERSION THRESHOLD SWEEP RESULTS")
        print("="*80)
        print(f"\n🥇 BEST CONFIGURATION (Score: {best_row['composite_score']:.4f})")
        print(f"   Signature: {best_row.get('config_signature', 'N/A')}")
        print(f"   AUC: {best_row.get('student_test_auc', 0):.4f} | Fwd Acc: {best_row.get('fwd_dir_acc', 0):.4f}")
        print(f"   Teacher Rate: {best_row.get('teacher_positive_rate', 0):.4f}")

        print(f"\n🏅 TOP 5 CONFIGURATIONS:")
        cols = ["config_id", "composite_score", "student_test_auc", "fwd_dir_acc", "teacher_positive_rate", "config_signature"]
        available_cols = [c for c in cols if c in successful.columns]
        print(successful[available_cols].head(5).to_string(index=False))
        print("\n" + "="*80)

        return pd.concat([successful, failed], ignore_index=True), analysis
