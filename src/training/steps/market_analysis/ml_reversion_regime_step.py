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

            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                market_data.index = pd.to_datetime(market_data.index)
                if market_data.index.tz is not None:
                    market_data.index = market_data.index.tz_convert(None)

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

            # Align indices and drop any samples without a valid direction label
            valid_target_idx = y_direction_all.dropna().index
            common_idx = (
                teacher_score.index
                .intersection(student_df.index)
                .intersection(valid_target_idx)
                .sort_values()
            )
            if len(common_idx) < 500:
                raise ValueError(f"Not enough aligned samples for training ({len(common_idx)} < 500)")

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
                if union_mask.sum() < 500:
                    raise ValueError(
                        f"Not enough samples within temporal split windows ({union_mask.sum()} < 500)"
                    )

                X_all = X_all.loc[union_mask]
                y_target_all = y_target_all.loc[union_mask]
                y_teacher_binary = y_teacher_binary.loc[union_mask]
                teacher_score_aligned = teacher_score_aligned.loc[union_mask]

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
                if dir_ == "short":
                    y_dir = (1 - y_target_all).astype(int)
                else:
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

                # Join OOF predictions (only OOF, no training set!)
                # This will have NaN for non-OOF periods
                output_df = output_df.join(oof_predictions.rename(columns={'probability': 'mr_probability'}), how='left')

                # Mark which samples are OOF vs. filled
                output_df['mr_is_oof'] = ~output_df['mr_probability'].isna()

                # Add target for OOF samples only
                output_df.loc[oof_predictions.index, "mr_direction_target"] = y_dir.loc[oof_predictions.index].values

                # For backward compatibility, also add mr_raw_score (same as mr_probability for OOF)
                output_df['mr_raw_score'] = output_df['mr_probability']

                output_df["mr_atr_14"] = atr_14
                output_df["mr_atr_300"] = atr_300
                output_df["mr_dynamic_tpsl_multiplier"] = dynamic_tp_sl_multiplier

                # Forward-return diagnostics at multiple horizons
                tprint_info(f"  📊 Computing forward-return diagnostics for {dir_} direction...")
                fwd_start = time.time()
                horizons_cfg = config.get("mr_forward_horizons", [2, 4, 8, 12])
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
            for i in range(vr_window, len(returns)):
                win = returns.iloc[i - vr_window : i]
                if win.isna().all():
                    continue
                var1 = float(win.var(ddof=1))
                if not np.isfinite(var1) or var1 <= 0:
                    continue
                win_k = win.rolling(vr_h).sum().dropna()
                if len(win_k) < 10:
                    continue
                var_k = float(win_k.var(ddof=1))
                if not np.isfinite(var_k) or var_k <= 0:
                    continue
                vr[i] = var_k / (vr_h * var1)
        tprint_info(f"  ✅ Variance ratio computed in {time.time() - vr_start:.2f}s")

        adf_p = np.full(len(close), np.nan)
        if STATIONARITY_AVAILABLE:
            tprint_info(f"  📊 Computing ADF p-values...")
            adf_start = time.time()
            adf_w = int(config.get("mr_adf_window", 200))
            for i in range(adf_w, len(returns)):
                seg = returns.iloc[i - adf_w : i]
                try:
                    adf_p[i] = float(adfuller(seg.values, maxlag=0, autolag=None)[1])
                except Exception:
                    adf_p[i] = np.nan
            tprint_info(f"  ✅ ADF p-values computed in {time.time() - adf_start:.2f}s")
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

        min_teacher = int(config.get("mr_min_teacher_samples", 100))
        if mask.sum() < min_teacher:
            raise ValueError("Not enough valid teacher samples")

        # Use rolling window normalization to prevent look-ahead bias
        window_size = int(config.get("mr_normalization_window", 500))
        X = winsorized_zscore_normalize(
            teacher_df.loc[mask, core_gmm_cols],
            window=window_size,
        ).values.astype(float)
        n_comp = int(config.get("mr_teacher_n_components", 3))
        gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=42)
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

    def _build_direction_target(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Build classification target: forward price direction.

        Returns:
            0 = bullish (price will go up)
            1 = bearish (price will go down)

        For 15m bars with trades lasting 30m-3h (2-12 bars), we use a forward
        horizon that captures the typical trade duration.
        """
        close = df["close"].astype(float)

        # For 15m timeframe, use 4-6 bar horizon (1-1.5h)
        forward_horizon = int(config.get("mr_forward_target_horizon", 6))
        min_threshold = float(config.get("mr_direction_min_threshold", 0.002))  # 0.2% minimum move

        fwd_returns = np.full(len(close), np.nan)
        for i in range(len(close) - forward_horizon):
            if close.iloc[i] > 0 and close.iloc[i + forward_horizon] > 0:
                fwd_returns[i] = (close.iloc[i + forward_horizon] - close.iloc[i]) / close.iloc[i]

        # Classification:
        # - If forward return > +min_threshold: label = 0 (bullish, price went up)
        # - If forward return < -min_threshold: label = 1 (bearish, price went down)
        # - If |forward return| < min_threshold: look at sign (slight bias up vs down)
        y_direction = np.full(len(close), np.nan)
        finite_mask = np.isfinite(fwd_returns)
        y_direction[finite_mask] = (fwd_returns[finite_mask] < 0).astype(int)  # 1 if down, 0 if up

        return pd.Series(y_direction, index=df.index)

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

        # Define parameter groups with tied parameters
        param_groups = [
            # Group 1: Model Structure (optimize first)
            ParameterGroup(
                name="structure",
                params={
                    "max_depth": {"type": "int", "low": 3, "high": 7},
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
        def objective(params: Dict[str, Any]) -> float:
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
            objective_func=lambda params: objective(params),
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

    def _train_xgb_student(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        split_config: Optional[TemporalSplitConfig] = None,
        y_teacher: Optional[pd.Series] = None
    ) -> Tuple[xgb.XGBClassifier, CalibratedClassifierCV, Dict[str, Any], np.ndarray, np.ndarray]:
        """Train XGBoost classifier with isotonic calibration and walk-forward validation.

        Args:
            X: Feature dataframe with DatetimeIndex
            y: Target series with DatetimeIndex
            config: Configuration dictionary
            split_config: Temporal split configuration (train/val/test boundaries)
            y_teacher: Optional teacher labels

        Returns:
            - Base XGB model
            - Calibrated model (isotonic)
            - Metrics dict
            - Raw scores (uncalibrated probabilities)
            - Calibrated scores
        """
        # Use temporal split config for proper train/val/test separation
        if split_config is not None:
            # Filter to train/val/test periods using temporal boundaries
            train_mask = (X.index >= split_config.training.start) & (X.index <= split_config.training.effective_end)
            val_mask = (X.index >= split_config.validation.start) & (X.index <= split_config.validation.effective_end)
            test_mask = (X.index >= split_config.test.start) & (X.index <= split_config.test.effective_end)

            X_train = X.loc[train_mask]
            X_val = X.loc[val_mask]
            X_test = X.loc[test_mask]

            y_train = y.loc[train_mask]
            y_val = y.loc[val_mask]
            y_test = y.loc[test_mask]

            # Convert to numpy for XGBoost
            X_train_np = X_train.astype(np.float32).values
            X_val_np = X_val.astype(np.float32).values
            X_test_np = X_test.astype(np.float32).values
            y_train_np = y_train.astype(np.int32).values
            y_val_np = y_val.astype(np.int32).values
            y_test_np = y_test.astype(np.int32).values

            n = len(X)
            n_train = len(X_train)
            n_val = len(X_val)
            n_test = len(X_test)

            tprint_info(
                f"📊 Using temporal splits: "
                f"Train {n_train} samples ({split_config.training.start} → {split_config.training.effective_end}), "
                f"Val {n_val} samples ({split_config.validation.start} → {split_config.validation.effective_end}), "
                f"Test {n_test} samples ({split_config.test.start} → {split_config.test.effective_end})"
            )
        else:
            # Fallback to percentage-based splits (legacy behavior)
            tprint_warning("⚠️  Using legacy percentage-based splits - consider using temporal split config for better results")
            X_np_full = X.astype(np.float32).values
            y_np_full = y.astype(np.int32).values
            n = len(X_np_full)

            train_frac = float(config.get("mr_train_fraction", 0.6))
            val_frac = float(config.get("mr_val_fraction", 0.2))
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            n_train = max(100, min(n_train, n - 200))
            n_val = max(50, min(n_val, n - n_train - 50))
            n_test = n - n_train - n_val
            if n_test < 50:
                n_test = 50
                n_train = n - n_val - n_test

            idx_train = slice(0, n_train)
            idx_val = slice(n_train, n_train + n_val)
            idx_test = slice(n_train + n_val, n)

            X_train_np = X_np_full[idx_train]
            X_val_np = X_np_full[idx_val]
            X_test_np = X_np_full[idx_test]
            y_train_np = y_np_full[idx_train]
            y_val_np = y_np_full[idx_val]
            y_test_np = y_np_full[idx_test]

            tprint_info(f"✅ Percentage splits prepared in {time.time() - split_start:.2f}s: Train {n_train}, Val {n_val}, Test {n_test}")

        # Calculate class weights for better balance
        # If classes are imbalanced, adjust scale_pos_weight to balance predictions
        n_neg = (y_train_np == 0).sum()
        n_pos = (y_train_np == 1).sum()
        if n_pos > 0:
            auto_scale_pos_weight = float(n_neg / n_pos)
        else:
            auto_scale_pos_weight = 1.0

        # IMPROVED: Reduced regularization to allow more diverse predictions
        # Previous settings were too conservative, leading to compressed probabilities
        params = dict(
            tree_method="hist",
            learning_rate=float(config.get("mr_learning_rate", 0.03)),  # Increased from 0.02
            max_depth=int(config.get("mr_max_depth", 5)),  # Increased from 4 for more complexity
            min_child_weight=float(config.get("mr_min_child_weight", 5.0)),  # Reduced from 10.0
            subsample=float(config.get("mr_subsample", 0.8)),  # Increased from 0.7
            colsample_bytree=float(config.get("mr_colsample_bytree", 0.8)),  # Increased from 0.6
            gamma=float(config.get("mr_gamma", 0.05)),  # Reduced from 0.1
            reg_alpha=float(config.get("mr_reg_alpha", 0.5)),  # Reduced from 1.0
            reg_lambda=float(config.get("mr_reg_lambda", 0.5)),  # Reduced from 1.0
            n_estimators=int(config.get("mr_n_estimators", 500)),
            # Use auto-calculated scale_pos_weight for class balance
            scale_pos_weight=float(config.get("mr_scale_pos_weight", auto_scale_pos_weight)),
            eval_metric="logloss",
        )

        tprint_info(
            f"📊 XGBoost class balance: n_neg={n_neg}, n_pos={n_pos}, "
            f"scale_pos_weight={params['scale_pos_weight']:.3f}"
        )

        tprint_info(f"🤖 Training base XGBoost model ({params['n_estimators']} trees)...")
        xgb_start = time.time()
        model = xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=30)
        model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False
        )
        tprint_info(f"✅ Base XGBoost trained in {time.time() - xgb_start:.2f}s (early_stopping_rounds=30)")

        # Get raw predictions (uncalibrated) for all splits
        tprint_info("📊 Generating raw (uncalibrated) predictions...")
        pred_start = time.time()
        raw_train = model.predict_proba(X_train_np)[:, 1]  # Probability of class 1 (bearish)
        raw_val = model.predict_proba(X_val_np)[:, 1]
        raw_test = model.predict_proba(X_test_np)[:, 1]
        tprint_info(f"✅ Raw predictions generated in {time.time() - pred_start:.2f}s")

        # Calibrate on validation set
        # IMPROVED: Default to sigmoid calibration for better probability distribution
        # Isotonic can be too aggressive and compress probabilities to narrow range
        calibration_method = config.get("mr_calibration_method", "sigmoid")
        if calibration_method not in ["isotonic", "sigmoid"]:
            calibration_method = "sigmoid"

        tprint_info(f"🎯 Calibrating model with {calibration_method} method on validation set...")
        calib_start = time.time()
        calibrated_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv="prefit"
        )
        calibrated_model.fit(X_val_np, y_val_np)
        tprint_info(f"✅ Calibration complete in {time.time() - calib_start:.2f}s")

        # Get calibrated predictions for all splits
        tprint_info("📊 Generating calibrated predictions...")
        calib_pred_start = time.time()
        calib_train = calibrated_model.predict_proba(X_train_np)[:, 1]
        calib_val = calibrated_model.predict_proba(X_val_np)[:, 1]
        calib_test = calibrated_model.predict_proba(X_test_np)[:, 1]
        tprint_info(f"✅ Calibrated predictions generated in {time.time() - calib_pred_start:.2f}s")

        def _metrics(y_true, y_pred_proba, prefix="") -> Dict[str, float]:
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            metrics = {
                f"{prefix}acc": float(accuracy_score(y_true, y_pred_binary)),
                f"{prefix}f1": float(f1_score(y_true, y_pred_binary, zero_division=0.0)),
                f"{prefix}precision": float(precision_score(y_true, y_pred_binary, zero_division=0.0)),
                f"{prefix}recall": float(recall_score(y_true, y_pred_binary, zero_division=0.0)),
            }
            try:
                metrics[f"{prefix}auc"] = float(roc_auc_score(y_true, y_pred_proba))
            except ValueError:
                metrics[f"{prefix}auc"] = float("nan")
            try:
                metrics[f"{prefix}logloss"] = float(log_loss(y_true, y_pred_proba))
            except ValueError:
                metrics[f"{prefix}logloss"] = float("nan")
            return metrics

        metrics: Dict[str, Any] = {
            "train_raw": _metrics(y_train_np, raw_train, ""),
            "val_raw": _metrics(y_val_np, raw_val, ""),
            "test_raw": _metrics(y_test_np, raw_test, ""),
            "train_calibrated": _metrics(y_train_np, calib_train, ""),
            "val_calibrated": _metrics(y_val_np, calib_val, ""),
            "test_calibrated": _metrics(y_test_np, calib_test, ""),
            "calibration_method": calibration_method,
            "class_balance": {
                "train_pos_rate": float(y_train_np.mean()),
                "val_pos_rate": float(y_val_np.mean()),
                "test_pos_rate": float(y_test_np.mean()),
            },
            "split_sizes": {
                "train": n_train,
                "val": n_val,
                "test": n_test,
            }
        }

        # Walk-forward validation for OOF calibration (using concatenated data)
        # Combine train+val+test back for walk-forward analysis
        tprint_info("🔄 [9/9] Running walk-forward validation for stability assessment...")
        X_full_np = np.vstack([X_train_np, X_val_np, X_test_np])
        y_full_np = np.concatenate([y_train_np, y_val_np, y_test_np])

        try:
            wf_metrics = self._run_walkforward_validation(
                X_full_np,
                y_full_np,
                config,
                calibration_method=calibration_method,
            )
            if wf_metrics:
                metrics["walkforward"] = wf_metrics
        except Exception as e:
            tprint_warning(f"⚠️  Walk-forward validation failed: {e}")

        # Combine predictions back into full arrays (in temporal order)
        raw_proba_full = np.concatenate([raw_train, raw_val, raw_test])
        calibrated_proba_full = np.concatenate([calib_train, calib_val, calib_test])

        return model, calibrated_model, metrics, raw_proba_full, calibrated_proba_full

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

        tprint_info(f"🚀 Using StandardizedXGBTrainer for OOF predictions (model_id={model_id})")

        # Create custom config
        training_config = XGBTrainingConfig(
            model_id=model_id,
            retrain_interval_days=10,  # OOF window every 10 days of historical data
            hpo_interval_days=30,  # HPO every 30 days of historical data
            burnin_pct=1/12,  # 3 months
            min_samples_for_training=1000,

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

    def _run_walkforward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict[str, Any],
        calibration_method: str = "isotonic",
    ) -> Dict[str, Any]:
        """Rolling walk-forward validation with OOF calibration.

        Each fold:
        - Train on expanding window
        - Calibrate on small validation window
        - Test on forward window
        """
        n = int(X.shape[0])
        if n < 600:
            return {}

        n_folds = int(config.get("mr_walkforward_folds", 3))  # Reduced from 5 to 3
        if n_folds < 2:
            tprint_warning(f"⚠️  Walk-forward validation requires at least 2 folds (got {n_folds}), skipping")
            return {}

        min_train = int(config.get("mr_walkforward_min_train_size", max(200, n // 4)))
        if min_train >= n - 100:
            tprint_warning(f"⚠️  Insufficient data for walk-forward validation (min_train={min_train}, n={n}), skipping")
            return {}

        step = (n - min_train) // n_folds
        if step < 50:
            tprint_warning(f"⚠️  Step size too small for walk-forward validation (step={step}), skipping")
            return {}

        tprint_info(f"🔄 Starting {n_folds}-fold walk-forward validation (n_samples={n}, min_train={min_train}, step={step})")
        tprint_info(f"⚠️  NOTE: The LAST fold will take 2-3x longer as it trains on the largest window (~75-80% of data)")

        acc_list: List[float] = []
        f1_list: List[float] = []
        auc_list: List[float] = []
        logloss_list: List[float] = []

        base_estimators = int(config.get("mr_n_estimators", 500))
        wf_estimators = max(100, min(base_estimators, 200))  # Reduced from 400 to 200
        tprint_info(f"📊 Using {wf_estimators} estimators per fold (reduced from base {base_estimators} for speed)")

        wf_start = time.time()
        for fold in range(n_folds):
            fold_start = time.time()
            tprint_info(f"  🔄 [Fold {fold+1}/{n_folds}] Starting walk-forward validation fold...")
            train_end = min_train + fold * step
            val_size = min(100, train_end // 10)
            val_start = train_end - val_size
            test_start = train_end
            test_end = min(train_end + step, n)
            if test_end - test_start < 50:
                tprint_warning(f"    ⚠️  Fold {fold+1} skipped: insufficient test samples ({test_end - test_start} < 50)")
                continue

            tprint_info(f"    📊 Fold {fold+1} data splits: train={val_start}, val={val_size}, test={test_end-test_start}")

            X_tr = X[:val_start]
            y_tr = y[:val_start]
            X_val = X[val_start:train_end]
            y_val = y[val_start:train_end]
            X_te = X[test_start:test_end]
            y_te = y[test_start:test_end]

            params = dict(
                tree_method="hist",
                learning_rate=float(config.get("mr_learning_rate", 0.02)),
                max_depth=int(config.get("mr_max_depth", 4)),
                min_child_weight=float(config.get("mr_min_child_weight", 10.0)),
                subsample=float(config.get("mr_subsample", 0.7)),
                colsample_bytree=float(config.get("mr_colsample_bytree", 0.6)),
                gamma=float(config.get("mr_gamma", 0.1)),
                reg_alpha=float(config.get("mr_reg_alpha", 1.0)),
                reg_lambda=float(config.get("mr_reg_lambda", 1.0)),
                n_estimators=wf_estimators,
                eval_metric="logloss",
            )
            try:
                tprint_info(f"    🤖 Training XGBoost model ({wf_estimators} trees, train_size={len(X_tr)})...")
                train_start_fold = time.time()
                model = xgb.XGBClassifier(**params, random_state=42)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                tprint_info(f"    ✅ XGBoost trained in {time.time() - train_start_fold:.2f}s")

                # Calibrate on val set
                tprint_info(f"    🎯 Calibrating with {calibration_method} method...")
                calib_start = time.time()
                calibrated = CalibratedClassifierCV(model, method=calibration_method, cv="prefit")
                calibrated.fit(X_val, y_val)
                tprint_info(f"    ✅ Calibration complete in {time.time() - calib_start:.2f}s")

                # Predict on test set
                y_pred_proba = calibrated.predict_proba(X_te)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            except Exception as fold_exc:
                tprint_error(f"    ❌ Fold {fold+1} FAILED after {time.time() - fold_start:.2f}s: {fold_exc}")
                continue

            try:
                acc = float(accuracy_score(y_te, y_pred))
                f1 = float(f1_score(y_te, y_pred, zero_division=0.0))
                auc = float(roc_auc_score(y_te, y_pred_proba))
                ll = float(log_loss(y_te, y_pred_proba))

                acc_list.append(acc)
                f1_list.append(f1)
                auc_list.append(auc)
                logloss_list.append(ll)

                fold_time = time.time() - fold_start
                tprint_success(
                    f"  ✅ [Fold {fold+1}/{n_folds}] Complete in {fold_time:.2f}s - "
                    f"ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}"
                )
            except Exception as metric_exc:
                tprint_error(f"    ❌ Failed to compute metrics for fold {fold+1}: {metric_exc}")
                pass

        total_wf_time = time.time() - wf_start
        if not acc_list:
            tprint_warning(f"⚠️  Walk-forward validation produced no valid results after {total_wf_time:.2f}s")
            return {}

        tprint_success(f"✅ Walk-forward validation complete in {total_wf_time:.2f}s ({len(acc_list)}/{n_folds} folds succeeded)")

        return {
            "folds": len(acc_list),
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "f1_mean": float(np.mean(f1_list)),
            "f1_std": float(np.std(f1_list)),
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list)),
            "logloss_mean": float(np.mean(logloss_list)),
            "logloss_std": float(np.std(logloss_list)),
            "acc_per_fold": acc_list,
            "f1_per_fold": f1_list,
            "auc_per_fold": auc_list,
            "logloss_per_fold": logloss_list,
        }

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
            csv_df = pd.DataFrame(
                {
                    "timestamp": X_all.index,
                    "mr_teacher_score": output_df.loc[X_all.index, "mr_teacher_score"],
                    "mr_raw_score": output_df.loc[X_all.index, "mr_raw_score"],
                    "mr_probability": output_df.loc[X_all.index, "mr_probability"],
                    "mr_direction_target": y_target,
                    "close": output_df.loc[X_all.index, "close"],
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
            close = output_df.loc[idx, "close"].astype(float)
            high = output_df.loc[idx, "high"].astype(float)
            low = output_df.loc[idx, "low"].astype(float)
            raw_returns = close.pct_change().fillna(0.0)

            prob = output_df.loc[idx, "mr_probability"].astype(float)

            # SIMPLIFIED: Use continuous probability directly.
            # For long strategy:
            #   - High bearish prob (close to 1) = avoid/short
            #   - Low bearish prob (close to 0) = strong long signal
            #   → long_confidence = 1 - prob
            # For short strategy (reverse grid):
            #   - High bearish prob (close to 1) = strong short signal
            #   → short_confidence = prob
            z_ma = output_df.loc[idx, "z_price_ma_slow"].astype(float)
            z_vwap = output_df.loc[idx, "z_price_vwap"].astype(float)

            if direction == "short":
                base_confidence = prob
                # Boost confidence when overbought (above mean) for mean-reversion shorts
                overbought = ((z_ma > 0.01) | (z_vwap > 0.01)).astype(float)
                confidence_boost = 1.0 + overbought * 0.5
                preds = -base_confidence * confidence_boost
                preds = preds.clip(-1, 0)
                grid_confidence = base_confidence
                grid_fn = run_simple_short_grid_backtest
            else:
                base_confidence = 1.0 - prob
                # Boost confidence when oversold (below mean) for mean-reversion longs
                oversold = ((z_ma < -0.01) | (z_vwap < -0.01)).astype(float)
                confidence_boost = 1.0 + oversold * 0.5
                preds = base_confidence * confidence_boost
                preds = preds.clip(0, 1)
                grid_confidence = base_confidence
                grid_fn = run_simple_long_grid_backtest

            ml_df_grid = pd.DataFrame(
                {
                    "mr_teacher_mean_reversion": y_teacher.loc[idx].astype(int),
                    "mr_teacher_score": output_df.loc[idx, "mr_teacher_score"].astype(float),
                    "mr_probability": prob,
                    "mr_direction_target": y_target.loc[idx].astype(int),
                },
                index=idx,
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

            if tp_override is not None and sl_override is not None:
                grid_df = grid_fn(
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
                )
            else:
                grid_df = grid_fn(
                    close=close,
                    high=high,
                    low=low,
                    raw_returns=raw_returns,
                    predictions=preds,
                    confidence=grid_confidence,
                    ml_df=ml_df_grid,
                    timeframe=timeframe,
                    regime_col="mr_teacher_mean_reversion",
                )

            if isinstance(grid_df, pd.DataFrame):
                grid_path = f"outcomes/ml_mean_reversion_grid_backtest_{symbol}_{timeframe}_{direction}_{ts}.csv"
                tprint_info(
                    f"Writing grid backtest CSV with shape={grid_df.shape} to {grid_path}"
                )
                grid_df.to_csv(grid_path, index=False)
                reports["grid_backtest_csv"] = grid_path
                tprint_success(f"✅ Saved grid backtest CSV: {grid_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to run/write grid backtest: {exc}")

        return artifacts, reports
