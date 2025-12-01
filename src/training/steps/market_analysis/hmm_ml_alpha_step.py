"""
HMM ML Alpha Step

This step consumes 1h Rolling HMM regime outputs plus OHLCV data to
construct forward-return-based alpha labels and a cleaned training
DataFrame for downstream models (e.g., regime-aware 15m models).

Responsibilities (initial version):
- Load 1h HMM artifacts from versioned HDF5 (labels, probabilities,
  economic features) using the same context as
  RollingHMMRegimeDiscoveryStep.
- Load 1h OHLCV market data.
- Align all series on a common DatetimeIndex.
- Compute forward 1h log returns and a simple binary alpha target.
- Save the resulting bar-level dataset into a dedicated
  versioned_artifacts store for later consumption.

Model training and regime-level alpha statistics will be added in
follow-up iterations.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
)
from src.features_common.transforms.scaling_normalization import (
    ScalingNormalizer,
    rolling_winsorized_zscore_normalize,
    rolling_adaptive_normalize,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
)
from src.training.steps.market_analysis.rolling_hmm_clustering.rolling_hmm_regime_discovery_step import (
    RollingHMMRegimeDiscoveryStep,
)
from src.utils.ml_common.optimization import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
)
from src.utils.ml_common.feature_engineering.feature_smoothing import apply_ewm_smoothing

# Feature analysis and selection tools
try:
    from src.feature_selection.advanced.permutation_importance import (
        PermutationImportanceCalculator,
        PermutationConfig,
    )
    PERMUTATION_IMPORTANCE_AVAILABLE = True
except ImportError:
    PERMUTATION_IMPORTANCE_AVAILABLE = False

try:
    from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR
    IMPROVED_MRMR_AVAILABLE = True
except ImportError:
    IMPROVED_MRMR_AVAILABLE = False

try:
    from src.utils.ml_common.evaluation.enhanced_learning_curve_analysis import (
        EnhancedLearningCurveAnalyzer,
    )
    ENHANCED_LEARNING_CURVE_AVAILABLE = True
except ImportError:
    ENHANCED_LEARNING_CURVE_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


logger = logging.getLogger(__name__)


class HMMMLAlphaStep(BaseStep):
    """Pipeline step to construct alpha labels from 1h Rolling HMM regimes."""

    def __init__(self, step_name: str = "hmm_ml_alpha_step"):
        """Initialize the HMM ML alpha step with versioned artifacts enabled."""
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("HMMMLAlphaStep") if hasattr(logger, "getChild") else logger
        tprint(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the alpha label construction from 1h HMM regimes.

        Expected config keys (minimum):
            - symbol: Trading symbol (e.g., 'ETHUSDT')
            - exchange: Exchange name (e.g., 'binance')
            - regime_timeframe: Timeframe used for HMM regimes (default: '1h')
            - direction: Trading direction (default: 'long')
            - execution_mode: 'full', 'light', 'blank', etc. (used for data loading)

        Optional alpha configuration:
            - alpha_horizon_bars: Forward horizon in bars (default: 1)
            - alpha_return_type: 'log' or 'simple' (default: 'log')
            - alpha_target_type: 'classification' or 'regression'
              (default: 'classification')
        """
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(
                config.get("regime_timeframe", config.get("timeframe", "15m"))
            )
            direction = str(config.get("direction", "long"))

            # Set default alpha-specific configuration if not provided
            # Smoothing defaults are chosen to target ~3ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ5 bar regime persistence
            # while allowing overrides via config/CLI when needed.
            alpha_defaults: Dict[str, Any] = {
                "alpha_target_smoothing_method": "ewm",
                "alpha_target_smoothing_window": 8,  # Adjusted for 4-8h regimes (was 5)
                "alpha_score_smoothing_method": "ewm",
                "alpha_score_smoothing_window": 16,  # Further increased window for smoother 4-8h regimes (was 12)
                # Keep HPO opt-in; these defaults are used when explicitly enabled
                "alpha_enable_hpo": False,
                "alpha_hpo_cv_folds": 3,
                "alpha_hpo_final_trials": 20,
                "alpha_early_stopping_rounds": 30,
                "alpha_enable_regression_calibration": True,
                "alpha_enable_expectation_calibration": True,
                "alpha_expectation_positive_threshold": 0.0,
                "alpha_expectation_min_samples": 200,
                "alpha_expectation_ema_period": 6,  # Adjusted for 4-8h regimes (was 4)
                "alpha_expectation_ema_weight_recent": 0.15,  # Moderate EMA for 4-8h regimes (was None ~0.4)
                "alpha_target_vol_window": 400,  # Adjusted rolling window (was 320)
                # Enable trend feature engineering on aligned market data
                "alpha_enable_trend_features": True,
                "alpha_trend_ema_fast_window": 56,  # Adjusted for 4-8h regimes (was 48)
                "alpha_trend_ema_slow_window": 120,  # Adjusted for 4-8h regimes (was 104)
                "alpha_trend_slope_window": 96,  # Adjusted for 4-8h regimes (was 80)
                # Auto-pruning experiment enabled by default with a small R^2 threshold
                "alpha_enable_auto_prune_rerun": True,
                "alpha_auto_prune_min_delta": 0.0005,
                # Quantile-based auto-prune thresholds over permutation importance.
                # Try slightly more aggressive thresholds; still require a small
                # positive improvement in validation R^2 before adopting.
                "alpha_auto_prune_quantiles": [0.15, 0.25, 0.35, 0.45],
                # Minimum run length for regime persistence (NOT scaled)
                # Leave at 0 by default; prefer HMM/smoothing to control persistence
                "alpha_regime_min_run_bars": 0,
                # Enable quality report generation
                "alpha_enable_quality_report": True,
                # Regime configuration: default to 4 regimes, with 2–4 explored in
                # flexible optimization / experimentation.
                "alpha_regime_counts": [2, 3, 4],
                "alpha_regime_bins": 4,
            }
            for k, v in alpha_defaults.items():
                config.setdefault(k, v)

            alpha_scaling_factor = 4

            for key, default in [
                ("alpha_max_horizon_bars", 3),
                ("alpha_horizon_bars", 1),
                ("alpha_target_smoothing_window", 8),  # Updated for 4-8h regimes
                ("alpha_score_smoothing_window", 16),  # Updated for slightly smoother 4-8h regimes
                ("alpha_target_vol_window", 400),  # Updated for 4-8h regimes
                ("alpha_expectation_ema_period", 6),  # Updated for 4-8h regimes
                ("alpha_normalization_window", 500),
                ("alpha_vol_of_vol_window", 10),
                ("alpha_trend_ema_fast_window", 56),  # Updated for 4-8h regimes
                ("alpha_trend_ema_slow_window", 120),  # Updated for 4-8h regimes
                ("alpha_trend_slope_window", 96),  # Updated for 4-8h regimes
            ]:
                value = config.get(key, default)
                try:
                    config[key] = int(value) * alpha_scaling_factor
                except Exception:
                    config[key] = value

            def _scale_periods(name: str, default_periods: List[int]) -> None:
                raw = config.get(name, default_periods)
                if isinstance(raw, (list, tuple)):
                    scaled: List[int] = []
                    for p in raw:
                        try:
                            scaled.append(int(p) * alpha_scaling_factor)
                        except Exception:
                            scaled.append(p)  # type: ignore[arg-type]
                    config[name] = scaled
                else:
                    try:
                        config[name] = int(raw) * alpha_scaling_factor
                    except Exception:
                        config[name] = raw

            _scale_periods("alpha_ewm_periods", [2, 6, 10])
            _scale_periods("alpha_score_ewm_periods", [3, 5])

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # ------------------------------------------------------------------
            # 1) Load HMM artifacts (labels/probabilities/economic features)
            # ------------------------------------------------------------------
            # Use the same context as RollingHMMRegimeDiscoveryStep when loading
            # its artifacts (model='regime').
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime",
            )

            use_internal_hmm = bool(config.get("alpha_generate_hmm_internally", True))

            if use_internal_hmm:
                labels_df, probs_df, economic_df = await self._generate_hmm_artifacts_internal(
                    symbol=symbol,
                    exchange=exchange,
                    regime_timeframe=regime_timeframe,
                    direction=direction,
                    config=config,
                )
            else:
                await self._ensure_hmm_economic_features(
                    symbol=symbol,
                    exchange=exchange,
                    regime_timeframe=regime_timeframe,
                    direction=direction,
                    config=config,
                )

                labels_df, probs_df, economic_df = self._load_hmm_artifacts(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=regime_timeframe,
                    config=config,
                )

            # ------------------------------------------------------------------
            # 2) Load 1h OHLCV market data
            # ------------------------------------------------------------------
            # Use the centralized execution_mode lookback (blank/light/full)
            # when loading market data so that blank mode uses the configured
            # ~1 year window instead of a hard-coded sample cap. Light-mode
            # filtering is still applied *after* alignment.
            market_data_config = {
                **config,
                "timeframe": regime_timeframe,
            }
            market_data_load_config = dict(market_data_config)
            market_data_load_config["execution_mode"] = str(
                config.get("execution_mode", "full")
            ).lower()

            market_data, market_source = self.load_market_data_or_fail(
                market_data_load_config,
                pipeline_state={},
                allow_config_override=True,
            )

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data.index, pd.DatetimeIndex):
                try:
                    market_data = market_data.copy()
                    # Handle tz-aware objects by converting to UTC then dropping tz
                    try:
                        market_data.index = pd.to_datetime(market_data.index)
                    except (TypeError, ValueError):
                        market_data.index = pd.to_datetime(market_data.index, utc=True)
                        market_data.index = market_data.index.tz_convert(None)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        "Market data index could not be converted to DatetimeIndex"
                    ) from exc

            tprint_info(
                f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {market_data.index.max()})"
            )

            # Create temporal split config with 6-month burn-in for indicator stabilization
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
                f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Temporal split config: "
                f"burn-in={split_config.burnin.start if split_config.burnin else 'None'} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ "
                f"{split_config.burnin.effective_end if split_config.burnin else 'None'}, "
                f"train={split_config.training.start} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {split_config.training.effective_end}, "
                f"val={split_config.validation.start} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {split_config.validation.effective_end}, "
                f"test={split_config.test.start} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {split_config.test.effective_end}"
            )

            # ------------------------------------------------------------------
            # 3) Align all inputs on common DatetimeIndex
            # ------------------------------------------------------------------
            aligned_df = self._align_inputs(
                market_data=market_data,
                labels_df=labels_df,
                probs_df=probs_df,
                economic_df=economic_df,
            )

            if aligned_df.empty:
                raise ValueError("Aligned dataset is empty after merging inputs")

            # Optional light-mode filtering applied on the *aligned* dataset so
            # that we always operate on the overlap between market data and HMM
            # artifacts.
            execution_mode = str(config.get("execution_mode", "full")).lower()
            if execution_mode == "light":
                aligned_df = self._apply_light_mode_filter(
                    aligned_df,
                    config,
                    timeframe=regime_timeframe,
                )

                if aligned_df.empty:
                    raise ValueError(
                        "Aligned dataset became empty after light-mode filtering; check HMM and market data coverage"
                    )

            if bool(config.get("alpha_enable_trend_features", True)):
                try:
                    if "close" in aligned_df.columns:
                        price_series = aligned_df["close"].astype(float)

                        fast_w = int(config.get("alpha_trend_ema_fast_window", 12))
                        slow_w = int(config.get("alpha_trend_ema_slow_window", 26))
                        slope_w = int(config.get("alpha_trend_slope_window", 20))

                        if fast_w > 0:
                            ema_fast = price_series.ewm(span=fast_w, adjust=False).mean()
                            aligned_df["trend_ema_fast"] = ema_fast

                        if slow_w > 0:
                            ema_slow = price_series.ewm(span=slow_w, adjust=False).mean()
                            aligned_df["trend_ema_slow"] = ema_slow

                        if slope_w > 1:
                            # Compute rolling log-price slope using fully
                            # vectorized NumPy operations (no per-window
                            # Python callbacks) for efficiency.
                            log_price = np.log(price_series.clip(lower=1e-8))
                            y = log_price.to_numpy(dtype=float, copy=False)
                            n = y.shape[0]

                            if n >= slope_w:
                                x = np.arange(slope_w, dtype=float)
                                x_mean = x.mean()
                                denom_loc = float(((x - x_mean) ** 2).sum())

                                if denom_loc <= 0.0:
                                    slope_vals = np.zeros_like(y, dtype=float)
                                else:
                                    # Rolling sums via convolution
                                    kernel_ones = np.ones(slope_w, dtype=float)
                                    sum_y = np.convolve(y, kernel_ones, mode="valid")
                                    sum_xy = np.convolve(y, x, mode="valid")

                                    # Covariance numerator: ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ(x*y) - x_mean * ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ(y)
                                    num = sum_xy - x_mean * sum_y
                                    slope_core = num / denom_loc

                                    slope_vals = np.full(n, np.nan, dtype=float)
                                    slope_vals[slope_w - 1 :] = slope_core

                                aligned_df["trend_price_slope"] = pd.Series(
                                    slope_vals,
                                    index=aligned_df.index,
                                    name="trend_price_slope",
                                )
                except Exception as trend_exc:
                    tprint_warning(f"Trend feature engineering failed (ignored): {trend_exc}")

            # ------------------------------------------------------------------
            # 4) Compute alpha labels
            # ------------------------------------------------------------------
            alpha_df = self._compute_alpha_labels(aligned_df, config)

            if alpha_df.empty:
                raise ValueError("Alpha dataset is empty after label construction")

            # ------------------------------------------------------------------
            # 5) Train LightGBM alpha model and derive alpha regimes
            # ------------------------------------------------------------------
            model = None
            alpha_scores = None
            training_metrics: Dict[str, Any] = {}
            training_metrics["alpha_fallback_used"] = False
            regime_stats_df: Optional[pd.DataFrame] = None
            model_path: Optional[str] = None
            regime_stats_path: Optional[str] = None
            regime_col_name: Optional[str] = None
            alpha_quality_metrics: Optional[ClusterQualityMetrics] = None
            alpha_quality_path: Optional[str] = None
            feature_pipeline_artifacts: Optional[Dict[str, Any]] = None
            feature_pipeline_path: Optional[str] = None

            try:
                alpha_enable_hpo = bool(config.get("alpha_enable_hpo", False))
                tprint_info(
                    f" Starting alpha model training: samples={len(alpha_df)}, "
                    f"features={len([c for c in alpha_df.select_dtypes(include=[np.number]).columns if c != 'alpha_target' and not c.startswith('alpha_forward_return_')])}, "
                    f"target_type={config.get('alpha_target_type', 'regression')}, "
                    f"alpha_enable_hpo={alpha_enable_hpo}"
                )

                model, alpha_scores, pred_col_name, training_metrics, feature_pipeline_artifacts = self._train_alpha_model(
                    alpha_df,
                    config,
                    split_config=split_config,
                )

                # Mark successful model training in metrics for downstream diagnostics
                training_metrics["alpha_model_training_failed"] = 0

                tprint_info(
                    f" Alpha model training complete: model_type="
                    f"{training_metrics.get('model_type', 'unknown')} | "
                    f"alpha_hpo_used={training_metrics.get('alpha_hpo_used', False)}"
                )
                if alpha_scores is not None:
                    alpha_df[pred_col_name] = alpha_scores
                    # Derive calibrated 01 expectation score from alpha predictions
                    # and make this the canonical alpha_score_continuous signal.
                    canonical_score: Optional[pd.Series] = None
                    try:
                        target_type_local = str(config.get("alpha_target_type", "regression")).lower()
                        expectation_series = None

                        if target_type_local == "classification":
                            # Classification branch already produces calibrated probabilities
                            expectation_series = alpha_scores.clip(0.0, 1.0)
                            canonical_score = expectation_series
                        else:
                            expectation_calibration_enabled = bool(
                                config.get("alpha_enable_expectation_calibration", True)
                            )
                            positive_threshold = float(
                                config.get("alpha_expectation_positive_threshold", 0.0)
                            )
                            min_samples = int(
                                config.get("alpha_expectation_min_samples", 200)
                            )

                            if expectation_calibration_enabled:
                                try:
                                    from sklearn.isotonic import IsotonicRegression

                                    y_target = alpha_df["alpha_target"].reindex(alpha_scores.index)
                                    base_mask = (
                                        alpha_scores.notna()
                                        & y_target.notna()
                                        & np.isfinite(alpha_scores)
                                        & np.isfinite(y_target)
                                    )

                                    # Use an out-of-fold style calibration: fit isotonic only on
                                    # the validation tail defined by alpha_train_fraction.
                                    n_valid = int(base_mask.sum())
                                    if n_valid >= max(min_samples, 50):
                                        train_frac_local = float(config.get("alpha_train_fraction", 0.8))
                                        train_frac_local = min(max(train_frac_local, 0.5), 0.95)

                                        # Chronological ordering for effective validation split
                                        valid_index = alpha_scores.index[base_mask]
                                        split_idx_local = int(n_valid * train_frac_local)
                                        split_idx_local = max(min(split_idx_local, n_valid - 1), 1)

                                        val_index = valid_index[split_idx_local:]
                                        if len(val_index) >= max(50, min_samples // 2):
                                            scores_val = alpha_scores.loc[val_index]
                                            y_val = y_target.loc[val_index]
                                            y_bin_val = (y_val > positive_threshold).astype(float)

                                            iso = IsotonicRegression(
                                                out_of_bounds="clip",
                                                y_min=0.0,
                                                y_max=1.0,
                                            )
                                            iso.fit(
                                                scores_val.to_numpy(
                                                    dtype=float,
                                                    copy=False,
                                                ),
                                                y_bin_val.to_numpy(
                                                    dtype=float,
                                                    copy=False,
                                                ),
                                            )

                                            calibrated_vals = iso.predict(
                                                alpha_scores.to_numpy(
                                                    dtype=float,
                                                    copy=False,
                                                )
                                            )
                                            expectation_series = pd.Series(
                                                calibrated_vals,
                                                index=alpha_scores.index,
                                                name="alpha_expectation_raw_01",
                                            ).clip(0.0, 1.0)

                                            training_metrics["alpha_expectation_calibration_used"] = True
                                            training_metrics[
                                                "alpha_expectation_positive_threshold"
                                            ] = positive_threshold
                                            training_metrics[
                                                "alpha_expectation_calibration_method"
                                            ] = "isotonic_regression_binary_oof"
                                        else:
                                            training_metrics["alpha_expectation_calibration_used"] = False
                                    else:
                                        training_metrics["alpha_expectation_calibration_used"] = False
                                except ImportError:
                                    training_metrics["alpha_expectation_calibration_used"] = False
                                except Exception as exp_calib_exc:
                                    training_metrics["alpha_expectation_calibration_used"] = False
                                    training_metrics["alpha_expectation_calibration_error"] = str(
                                        exp_calib_exc
                                    )

                            if expectation_series is None:
                                # Monotonic fallback: logistic mapping of centered alpha_scores
                                try:
                                    scores_clean = alpha_scores.replace(
                                        [np.inf, -np.inf], np.nan
                                    ).dropna()
                                    if not scores_clean.empty:
                                        median_val = float(scores_clean.median())
                                        iqr = float(
                                            scores_clean.quantile(0.75)
                                            - scores_clean.quantile(0.25)
                                        )
                                        scale = iqr if iqr > 1e-8 else max(
                                            float(scores_clean.std()), 1e-8
                                        )
                                        z = (alpha_scores - median_val) / scale
                                        expectation_series = 1.0 / (1.0 + np.exp(-z))
                                        expectation_series = expectation_series.clip(
                                            0.0, 1.0
                                        )
                                except Exception:
                                    expectation_series = None

                            if expectation_series is not None:
                                alpha_df["alpha_expectation_raw_01"] = expectation_series
                                canonical_score = expectation_series

                                # EMA smoothing on expectation score
                                try:
                                    ema_weight_raw = config.get(
                                        "alpha_expectation_ema_weight_recent",
                                        None,
                                    )
                                    ema_period = int(
                                        config.get(
                                            "alpha_expectation_ema_period",
                                            4,
                                        )
                                    )
                                    if ema_weight_raw is None:
                                        period_eff = max(ema_period, 1)
                                        ema_weight = 2.0 / float(period_eff + 1.0)
                                    else:
                                        ema_weight = float(ema_weight_raw)

                                    ema_weight = min(max(ema_weight, 0.01), 1.0)
                                    expectation_ema = expectation_series.ewm(
                                        alpha=ema_weight,
                                        adjust=False,
                                    ).mean()
                                    alpha_df[
                                        "alpha_expectation_ema_01"
                                    ] = expectation_ema
                                    canonical_score = expectation_ema
                                    training_metrics[
                                        "alpha_expectation_ema_weight_recent"
                                    ] = ema_weight
                                    training_metrics[
                                        "alpha_expectation_ema_period"
                                    ] = ema_period
                                except Exception as ema_exc:
                                    training_metrics["alpha_expectation_ema_error"] = str(
                                        ema_exc
                                    )

                    except Exception as expectation_exc:
                        training_metrics["alpha_expectation_error"] = str(
                            expectation_exc
                        )

                    # Finalize canonical alpha_score_continuous: prefer calibrated expectation,
                    # fall back to the underlying alpha_scores if calibration was unavailable.
                    if canonical_score is not None:
                        alpha_df["alpha_score_continuous"] = canonical_score
                    else:
                        alpha_df["alpha_score_continuous"] = alpha_scores

                    # Add EWM-smoothed variants of alpha_score_continuous as additional features
                    try:
                        score_base = (
                            alpha_df["alpha_score_continuous"]
                            .astype(float)
                            .replace([np.inf, -np.inf], np.nan)
                        )
                        ewm_periods_cfg = config.get("alpha_score_ewm_periods", [3, 5])
                        try:
                            ewm_periods = [int(p) for p in ewm_periods_cfg if int(p) > 0]
                        except Exception:
                            ewm_periods = [3, 5]

                        for period in ewm_periods:
                            col_name = f"alpha_score_continuous_ewm_{period}"
                            alpha_df[col_name] = score_base.ewm(
                                span=period,
                                adjust=False,
                            ).mean()
                    except Exception:
                        pass

                    alpha_df, regime_stats_df, regime_col_name = self._assign_alpha_regimes(
                        alpha_df,
                        alpha_df["alpha_score_continuous"],
                        config,
                    )

                    # Record whether primary regime assignment produced a usable regime column
                    if regime_col_name is not None and regime_col_name in alpha_df.columns:
                        try:
                            n_regimes_primary = int(
                                pd.Series(alpha_df[regime_col_name].dropna().astype(int)).nunique()
                            )
                        except Exception:
                            n_regimes_primary = -1
                        training_metrics["alpha_primary_regimes_success"] = 1
                        training_metrics["alpha_primary_regimes_n_regimes"] = n_regimes_primary
                    else:
                        training_metrics["alpha_primary_regimes_success"] = 0
            except ImportError as lgb_err:
                tprint_warning(
                    f"LightGBM not available; skipping alpha model training: {lgb_err}"
                )
            except Exception as model_exc:
                tprint_warning(
                    f"Alpha model training failed; continuing with labels only: {model_exc}"
                )
                training_metrics["alpha_model_training_failed"] = 1
                training_metrics["alpha_model_training_error"] = str(model_exc)

            # Only trigger fallback if we genuinely have no regime column.
            # This avoids marking alpha_fallback_used=True when primary
            # regime assignment has already succeeded.
            if regime_col_name is None or regime_col_name not in alpha_df.columns:
                raise RuntimeError(
                    "Alpha regime assignment failed: no regime column produced from primary "
                    "alpha model / score. Fallback to forward returns has been disabled to "
                    "force explicit investigation of alpha model/regime issues."
                )

            # ------------------------------------------------------------------
            # 5b) Diagnostics for alpha_score_continuous distribution & economics
            # ------------------------------------------------------------------
            try:
                if "alpha_score_continuous" in alpha_df.columns:
                    score_series = (
                        alpha_df["alpha_score_continuous"]
                        .astype(float)
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )

                    if not score_series.empty:
                        # Basic distribution statistics
                        dist = {
                            "mean": float(score_series.mean()),
                            "std": float(score_series.std()),
                            "min": float(score_series.min()),
                            "max": float(score_series.max()),
                            "q05": float(score_series.quantile(0.05)),
                            "q50": float(score_series.quantile(0.50)),
                            "q95": float(score_series.quantile(0.95)),
                            "n": int(score_series.shape[0]),
                        }
                        training_metrics["alpha_score_continuous_distribution"] = dist

                        # Economic deciles on 1h forward return
                        fwd_col = "alpha_forward_return_1h" if "alpha_forward_return_1h" in alpha_df.columns else None
                        if fwd_col is not None:
                            fwd_ret = (
                                alpha_df.loc[score_series.index, fwd_col]
                                .astype(float)
                                .replace([np.inf, -np.inf], np.nan)
                                .dropna()
                            )
                            common_idx = score_series.index.intersection(fwd_ret.index)
                            if len(common_idx) >= 50:
                                s = score_series.loc[common_idx]
                                r = fwd_ret.loc[common_idx]
                                try:
                                    ranks = s.rank(method="first")
                                    deciles = pd.qcut(ranks, 10, labels=False, duplicates="drop")
                                    decile_stats = []
                                    for d in sorted(deciles.dropna().unique()):
                                        mask = deciles == d
                                        if not mask.any():
                                            continue
                                        r_d = r[mask]
                                        if r_d.empty:
                                            continue
                                        mean_ret = float(r_d.mean())
                                        vol = float(r_d.std()) if len(r_d) > 1 else 0.0
                                        sharpe = mean_ret / (vol + 1e-8) if vol > 0 else 0.0
                                        decile_stats.append(
                                            {
                                                "decile": int(d),
                                                "n": int(mask.sum()),
                                                "mean_forward_return": mean_ret,
                                                "vol_forward_return": vol,
                                                "sharpe_forward_return": sharpe,
                                            }
                                        )
                                    if decile_stats:
                                        training_metrics[
                                            "alpha_score_decile_forward_returns"
                                        ] = decile_stats
                                except Exception:
                                    # Decile diagnostics are best-effort only
                                    pass
            except Exception:
                # Diagnostics are best-effort only; never fail the pipeline
                pass

            # ------------------------------------------------------------------
            # 6) Extract and save regime thresholds for production use
            # ------------------------------------------------------------------
            regime_thresholds: Optional[Dict[str, Any]] = None
            regime_thresholds_path: Optional[str] = None

            if alpha_scores is not None and regime_col_name is not None and regime_col_name in alpha_df.columns:
                try:
                    regime_thresholds = self._extract_and_save_regime_thresholds(
                        alpha_scores=alpha_scores,
                        regime_labels=alpha_df[regime_col_name],
                        regime_col_name=regime_col_name,
                        symbol=symbol,
                        config=config,
                    )

                    if regime_thresholds and "extraction_error" not in regime_thresholds:
                        # Save thresholds as artifact
                        try:
                            # Convert regime_thresholds to DataFrame format for HDF5 compatibility
                            # Extract the regime_thresholds dict and convert to DataFrame
                            if "regime_thresholds" in regime_thresholds and isinstance(regime_thresholds["regime_thresholds"], dict):
                                regime_df = pd.DataFrame.from_dict(regime_thresholds["regime_thresholds"], orient='index')
                                regime_df.index.name = 'regime'
                                regime_thresholds_for_save = {
                                    "regime_thresholds_df": regime_df,
                                    "metadata": {
                                        "extraction_timestamp": regime_thresholds.get("extraction_timestamp"),
                                        "symbol": regime_thresholds.get("symbol"),
                                        "regime_col_name": regime_thresholds.get("regime_col_name"),
                                        "total_samples": regime_thresholds.get("total_samples"),
                                        "n_regimes": regime_thresholds.get("n_regimes"),
                                    }
                                }
                            else:
                                regime_thresholds_for_save = regime_thresholds

                            regime_thresholds_path = self._save_artifact(
                                data=regime_thresholds_for_save,
                                artifact_name="hmm_alpha_regime_thresholds_15m",
                                artifact_type="model",
                                data_category="config",
                                metadata={
                                    "symbol": symbol,
                                    "exchange": exchange,
                                    "timeframe": regime_timeframe,
                                    "n_regimes": regime_thresholds.get("n_regimes", 0),
                                },
                            )
                            tprint_info(f" Saved regime thresholds artifact: {regime_thresholds_path}")
                        except Exception as thresholds_save_exc:
                            tprint_warning(f"Failed to save regime thresholds artifact: {thresholds_save_exc}")

                        # Add thresholds to training metrics for reporting
                        training_metrics["regime_thresholds"] = regime_thresholds
                except Exception as thresholds_exc:
                    tprint_warning(f"Regime threshold extraction failed (non-fatal): {thresholds_exc}")

            # ------------------------------------------------------------------
            # 7) Switch context to dedicated alpha namespace and assess quality
            # ------------------------------------------------------------------
            # Switch context to a dedicated alpha model namespace so we do not
            # pollute the original HMM regime store.
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime_alpha",
            )

            # Run unified cluster quality assessment on alpha regimes (if any)
            try:
                alpha_quality_metrics, alpha_quality_path = self._assess_alpha_regime_quality(
                    alpha_df=alpha_df,
                    regime_col=regime_col_name,
                    config=config,
                )
            except Exception as quality_exc:
                tprint_warning(f"Alpha regime quality assessment failed: {quality_exc}")

            alpha_to_save = alpha_df.reset_index().rename(
                columns={alpha_df.index.name or "index": "timestamp"}
            )

            tprint_info(
                f" Saving alpha training dataset with shape {alpha_to_save.shape} "
                f"to versioned HDF5 store"
            )
            # Build metadata with temporal split information
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "source_market_data": market_source,
                "version": "v2_with_burnin",
                "training_start": str(split_config.training.start),
                "training_end": str(split_config.training.effective_end),
                "validation_start": str(split_config.validation.start),
                "validation_end": str(split_config.validation.effective_end),
                "test_start": str(split_config.test.start),
                "test_end": str(split_config.test.effective_end),
            }
            if split_config.burnin is not None:
                metadata["burnin_start"] = str(split_config.burnin.start)
                metadata["burnin_end"] = str(split_config.burnin.effective_end)

            training_data_path = self._save_artifact(
                data=alpha_to_save,
                artifact_name="hmm_alpha_training_data_15m",
                artifact_type="data",
                metadata=metadata,
            )

            # Save trained model if available
            if model is not None:
                try:
                    tprint_info(" Saving LightGBM alpha model via artifact router")
                    model_metadata = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": regime_timeframe,
                        "model_type": "lightgbm",
                        "version": "v2_with_burnin",
                        "training_start": str(split_config.training.start),
                        "training_end": str(split_config.training.effective_end),
                    }
                    if split_config.burnin is not None:
                        model_metadata["burnin_start"] = str(split_config.burnin.start)
                        model_metadata["burnin_end"] = str(split_config.burnin.effective_end)

                    model_path = self._save_artifact(
                        data=model,
                        artifact_name="hmm_alpha_model_15m",
                        artifact_type="model",
                        metadata=model_metadata,
                    )
                except Exception as save_model_exc:
                    tprint_warning(f"Failed to save alpha model artifact: {save_model_exc}")

            # Persist feature pipeline (feature list + scaler state) for live usage
            if feature_pipeline_artifacts is not None:
                try:
                    feature_pipeline_path = self._save_artifact(
                        data=feature_pipeline_artifacts,
                        artifact_name="hmm_alpha_feature_pipeline_15m",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "feature_names": feature_pipeline_artifacts.get("feature_names", []),
                        },
                    )
                except Exception as save_fp_exc:
                    tprint_warning(f"Failed to save alpha feature pipeline artifact: {save_fp_exc}")

            # Save regime-level alpha statistics if available
            if regime_stats_df is not None and not regime_stats_df.empty:
                try:
                    regime_stats_to_save = regime_stats_df.reset_index()
                    regime_stats_path = self._save_artifact(
                        data=regime_stats_to_save,
                        artifact_name="hmm_alpha_regime_stats_15m",
                        artifact_type="data",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                        },
                    )
                except Exception as save_stats_exc:
                    tprint_warning(
                        f"Failed to save alpha regime statistics artifact: {save_stats_exc}"
                    )

            # ------------------------------------------------------------------
            # Generate comprehensive quality report if enabled
            # ------------------------------------------------------------------
            enable_quality_report = bool(config.get("alpha_enable_quality_report", True))
            if not enable_quality_report:
                tprint_info(
                    "alpha_enable_quality_report is False in config; "
                    "overriding to True for alpha regime diagnostics"
                )
                enable_quality_report = True

            if enable_quality_report:
                try:
                    report_result = self._generate_hmm_alpha_quality_report(
                        alpha_df=alpha_df,
                        regime_col=regime_col_name,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=regime_timeframe,
                        training_metrics=training_metrics,
                        config=config,
                    )
                    if report_result is not None:
                        csv_path, md_path = report_result
                        tprint_info(f"📄 Generated HMM alpha quality report: {md_path}")
                except Exception as report_exc:
                    tprint_warning(f"Failed to generate HMM alpha quality report: {report_exc}")

            execution_time = time.time() - start_time
            tprint_info(
                f" {self.step_name} completed in {execution_time:.2f}s "
                f"with {len(alpha_df)} samples"
            )

            result: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "n_samples": int(len(alpha_df)),
                "training_data_path": training_data_path,
                "model_path": model_path,
                "training_metrics": training_metrics,
            }

            return result

        except Exception as exc:
            tprint_error(f" {self.step_name} failed: {exc}")
            return {"success": False, "error": str(exc)}

    def _compute_alpha_labels(
        self,
        aligned_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute forward-return-based alpha labels on the aligned dataset.

        For regression:
            - Compute forward returns for horizons 1ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ3h.
            - Use the average of these horizons as a smoother macro alpha target.

        For classification:
            - Use the sign of the 1h forward return.
        """

        return_type = str(config.get("alpha_return_type", "log")).lower()
        target_type = str(config.get("alpha_target_type", "regression")).lower()

        df = aligned_df.copy()
        if "close" not in df.columns:
            raise ValueError("Aligned dataset must contain a 'close' column for returns")

        close = df["close"].astype(float)

        # Always compute 1h forward return
        if return_type == "simple":
            fwd_ret_1h = close.shift(-4) / close - 1.0
        else:
            fwd_ret_1h = np.log(close.shift(-4) / close)
        df["alpha_forward_return_1h"] = fwd_ret_1h

        # This prevents extreme events (flash crashes, +50% bars) from dominating
        # the Loss Function (MSE) of LightGBM/XGBoost models.
        winsorize_enabled = config.get("alpha_winsorize_targets", True)
        winsorize_lower = config.get("alpha_winsorize_lower_quantile", 0.01)
        winsorize_upper = config.get("alpha_winsorize_upper_quantile", 0.99)

        # Build forward-return columns for multiple horizons
        max_h = int(config.get("alpha_max_horizon_bars", 3))
        max_h = max(max_h, 1)
        fwd_cols: Dict[int, str] = {}

        for h in range(1, max_h + 1):
            if return_type == "simple":
                fwd_ret = close.shift(-4 * h) / close - 1.0
            else:
                fwd_ret = np.log(close.shift(-4 * h) / close)
            col_name = f"alpha_forward_return_{h}h"
            df[col_name] = fwd_ret
            fwd_cols[h] = col_name

        if winsorize_enabled:
            try:
                tprint_info(
                    f" Winsorizing forward returns at {winsorize_lower:.1%} and {winsorize_upper:.1%} quantiles"
                )
                for col_name in fwd_cols.values():
                    col_data = df[col_name].dropna()
                    if len(col_data) > 0:
                        lower_bound = col_data.quantile(winsorize_lower)
                        upper_bound = col_data.quantile(winsorize_upper)
                        df[col_name] = df[col_name].clip(lower=lower_bound, upper=upper_bound)
                        tprint_info(
                            f"  {col_name}: clipped to [{lower_bound:.6f}, {upper_bound:.6f}]"
                        )
            except Exception as wins_exc:
                tprint_warning(
                    f" Failed to winsorize forward returns (best-effort only): {wins_exc}"
                )

        if target_type == "regression":
            # Average of 1ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ3h forward returns as macro target
            horizon_keys = [h for h in fwd_cols.keys() if 1 <= h <= max_h]
            fwd_stack = [df[fwd_cols[h]] for h in horizon_keys]
            multi_target = pd.concat(fwd_stack, axis=1).mean(axis=1)

            # Volatility-normalized target: scale by recent realized volatility
            # computed from underlying returns (not forward returns) to reduce
            # heteroskedasticity and stabilize the alpha target.
            if return_type == "simple":
                base_returns = close.pct_change()
            else:
                base_returns = np.log(close).diff()

            vol_window_cfg = int(config.get("alpha_target_vol_window", 320))
            vol_window_cfg = max(vol_window_cfg, 1)
            n_returns = int(base_returns.shape[0])

            if n_returns <= 0:
                realized_vol = pd.Series(index=base_returns.index, dtype=float)
            else:
                if vol_window_cfg > n_returns:
                    # Adapt the window to the available history so that we still
                    # get a meaningful rolling volatility estimate on short
                    # aligned datasets.
                    adaptive_base = max(n_returns // 3, 10)
                    effective_window = max(10, min(vol_window_cfg, adaptive_base))
                    tprint_warning(
                        f"alpha_target_vol_window={vol_window_cfg} exceeds available samples ({n_returns}); using adaptive window={effective_window}"
                    )
                else:
                    effective_window = vol_window_cfg

                realized_vol = base_returns.rolling(effective_window).std()

            df["alpha_target_raw"] = multi_target
            df["alpha_target_vol"] = realized_vol

            alpha_target = multi_target / (realized_vol + 1e-8)

            # Optional robust clipping on the normalized target itself
            target_winsorize_enabled = bool(
                config.get("alpha_target_winsorize_enabled", True)
            )
            target_q_lower = float(
                config.get("alpha_target_winsorize_lower_quantile", 0.01)
            )
            target_q_upper = float(
                config.get("alpha_target_winsorize_upper_quantile", 0.99)
            )

            if target_winsorize_enabled:
                try:
                    tgt_clean = (
                        pd.Series(alpha_target)
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    if not tgt_clean.empty:
                        t_low = float(tgt_clean.quantile(target_q_lower))
                        t_high = float(tgt_clean.quantile(target_q_upper))
                        alpha_target = alpha_target.clip(t_low, t_high)
                        tprint_info(
                            f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ§ Winsorizing alpha_target at {target_q_lower:.1%}/{target_q_upper:.1%}: "
                            f"[{t_low:.6f}, {t_high:.6f}]"
                        )
                except Exception as tgt_wins_exc:
                    tprint_warning(
                        f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¸ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Failed to winsorize alpha_target (vol-normalized): {tgt_wins_exc}"
                    )

            df["alpha_target"] = alpha_target
            effective_horizon = f"1-{max_h}h_mean_vol_norm"
        else:
            # Classification: sign of 1h forward return
            target = (fwd_ret_1h > 0).astype(float)
            target = target.where(~fwd_ret_1h.isna())
            df["alpha_target"] = target
            effective_horizon = "1h_classification"

        # Drop rows where we cannot compute all required forward returns
        required_cols = ["alpha_target"] + [fwd_cols[h] for h in sorted(fwd_cols.keys())]
        before = len(df)
        df = df.dropna(subset=required_cols)
        dropped = before - len(df)
        if dropped > 0:
            tprint_warning(
                f"Dropped {dropped} rows with NaN forward returns when building alpha labels"
            )

        tprint_info(
            f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂĂÂĂÂĂĹĄĂÂĂÂĂÂÄÂĂÂ Alpha label dataset shape: {df.shape} "
            f"(target_type={target_type}, effective_horizon={effective_horizon}, return_type={return_type})"
        )

        return df

    async def _ensure_hmm_economic_features(
        self,
        *,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
        direction: str,
        config: Dict[str, Any],
    ) -> None:
        try:
            economic = self._get_artifact(
                artifact_name="rolling_hmm_economic_features",
                artifact_type="data",
            )
        except Exception as exc:
            tprint_warning(
                f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¸ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Failed to check rolling_hmm_economic_features artifact (non-fatal): {exc}"
            )
            return
        if economic is None:
            tprint_warning(
                "No rolling_hmm_economic_features artifact found; proceeding without dedicated economic features"
            )

    def _load_hmm_artifacts(
        self,
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        labels = self._get_artifact(
            artifact_name="rolling_hmm_regime_labels",
            artifact_type="data",
        )
        if labels is None:
            raise FileNotFoundError(
                "rolling_hmm_regime_labels artifact not found in versioned store"
            )

        if not isinstance(labels, pd.DataFrame):
            labels = pd.DataFrame(labels)

        if "timestamp" in labels.columns:
            labels = labels.copy()
            labels["timestamp"] = pd.to_datetime(labels["timestamp"])
            labels.set_index("timestamp", inplace=True)
        elif isinstance(labels.index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError(
                "HMM labels artifact must have a 'timestamp' column or DatetimeIndex"
            )

        tprint_info(
            f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Loaded HMM labels: {labels.shape} "
            f"({labels.index.min()} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {labels.index.max()})"
        )

        probs = self._get_artifact(
            artifact_name="rolling_hmm_regime_probabilities",
            artifact_type="data",
        )
        if probs is not None:
            if not isinstance(probs, pd.DataFrame):
                probs = pd.DataFrame(probs)
            if "timestamp" in probs.columns:
                probs = probs.copy()
                probs["timestamp"] = pd.to_datetime(probs["timestamp"])
                probs.set_index("timestamp", inplace=True)
            elif not isinstance(probs.index, pd.DatetimeIndex):
                tprint_warning(
                    "HMM probabilities artifact has no DatetimeIndex; "
                    "dropping probability features"
                )
                probs = None

        if probs is not None and not probs.empty:
            tprint_info(
                f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Loaded HMM probabilities: {probs.shape} "
                f"({probs.index.min()} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {probs.index.max()})"
            )
        else:
            tprint_warning("No HMM probabilities found; proceeding without them")
            probs = None

        economic = self._get_artifact(
            artifact_name="rolling_hmm_economic_features",
            artifact_type="data",
        )
        if economic is not None:
            if not isinstance(economic, pd.DataFrame):
                economic = pd.DataFrame(economic)
            if "timestamp" in economic.columns:
                economic = economic.copy()
                economic["timestamp"] = pd.to_datetime(economic["timestamp"])
                economic.set_index("timestamp", inplace=True)
            elif not isinstance(economic.index, pd.DatetimeIndex):
                tprint_warning(
                    "Economic features artifact has no DatetimeIndex; "
                    "dropping economic features"
                )
                economic = None

        return labels, probs, economic

    async def _generate_hmm_artifacts_internal(
        self,
        *,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
        direction: str,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Run RollingHMMRegimeDiscoveryStep to generate fresh HMM artifacts.

        This avoids relying on potentially stale rolling_hmm_regime_* artifacts
        and instead computes regimes directly from current market data.
        """

        hmm_config: Dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "regime_timeframe": regime_timeframe,
            "timeframe": regime_timeframe,
            "execution_mode": str(config.get("execution_mode", "full")).lower(),
        }

        reuse_hpo_best = bool(config.get("alpha_hmm_reuse_best_params", True))
        force_hpo = bool(config.get("alpha_hmm_force_hpo", False))
        if bool(config.get("enable_hpo", False)):
            force_hpo = True

        hmm_config["reuse_hpo_best_params"] = reuse_hpo_best
        hmm_config["force_hpo"] = force_hpo

        # Start from any caller-provided rolling_hmm_params, then apply
        # optimized defaults for internal usage where not explicitly set.
        rolling_params = dict(config.get("rolling_hmm_params", {})) if "rolling_hmm_params" in config else {}

        if "max_samples_for_hpo" not in rolling_params:
            rolling_params["max_samples_for_hpo"] = int(
                config.get("alpha_hmm_max_samples_for_hpo", 8000)
            )
        if "hpo_sample_fraction" not in rolling_params:
            rolling_params["hpo_sample_fraction"] = float(
                config.get("alpha_hmm_hpo_sample_fraction", 0.3)
            )
        if "hpo_stratified_sampling" not in rolling_params:
            rolling_params["hpo_stratified_sampling"] = bool(
                config.get("alpha_hmm_hpo_stratified_sampling", True)
            )

        if rolling_params:
            hmm_config["rolling_hmm_params"] = rolling_params

        # Allow advanced users to pass through feature/hpo configs when needed.
        if "feature_config" in config:
            hmm_config["feature_config"] = config.get("feature_config", {})
        if "hpo_config" in config:
            hmm_config["hpo_config"] = config.get("hpo_config", {})

        hmm_step = RollingHMMRegimeDiscoveryStep()
        hmm_result = await hmm_step.execute(hmm_config)

        if not hmm_result.get("success", False):
            raise RuntimeError(
                f"Internal Rolling HMM regime discovery failed inside {self.step_name}: "
                f"{hmm_result.get('error', 'unknown error')}"
            )

        artifacts = hmm_result.get("artifacts", {})
        labels = artifacts.get("labels")
        probs = artifacts.get("probabilities")

        if labels is None or probs is None:
            raise RuntimeError(
                "Internal Rolling HMM regime discovery did not return labels/probabilities DataFrames"
            )

        # Try to load economic features that Rolling HMM saves as a separate artifact.
        economic: Optional[pd.DataFrame]
        try:
            economic = self._get_artifact(
                artifact_name="rolling_hmm_economic_features",
                artifact_type="data",
            )
            if economic is not None and not isinstance(economic, pd.DataFrame):
                economic = pd.DataFrame(economic)
        except Exception:
            economic = None

        return labels, probs, economic

    def _align_inputs(
        self,
        *,
        market_data: pd.DataFrame,
        labels_df: Optional[pd.DataFrame],
        probs_df: Optional[pd.DataFrame],
        economic_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        def _prepare(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            return df.sort_index()

        market_data = _prepare(market_data)
        if market_data is None:
            raise ValueError("Market data is required for alignment")

        frames: List[pd.DataFrame] = []

        base_ohlcv = market_data[[
            col
            for col in market_data.columns
            if col.lower() in {"open", "high", "low", "close", "volume"}
        ]].rename(columns=lambda c: c.lower())
        frames.append(base_ohlcv)

        labels_df_prepared = _prepare(labels_df)
        if labels_df_prepared is not None and not labels_df_prepared.empty:
            frames.append(labels_df_prepared)

        probs_df_prepared = _prepare(probs_df)
        if probs_df_prepared is not None and not probs_df_prepared.empty:
            frames.append(probs_df_prepared)

        economic_df_prepared = _prepare(economic_df)
        if economic_df_prepared is not None and not economic_df_prepared.empty:
            frames.append(economic_df_prepared)

        aligned = frames[0]
        for extra in frames[1:]:
            aligned = aligned.join(extra, how="inner")

        aligned = aligned.dropna(how="all")

        tprint_info(
            f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Aligned dataset shape: {aligned.shape} "
            f"({aligned.index.min()} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {aligned.index.max()})"
        )

        return aligned

    def _train_alpha_model(
        self,
        alpha_df: pd.DataFrame,
        config: Dict[str, Any],
        split_config: Optional[TemporalSplitConfig] = None,
    ) -> Tuple[Any, Optional[pd.Series], str, Dict[str, Any], Dict[str, Any]]:
        """Train a LightGBM model to predict alpha targets."""
        try:
            import lightgbm as lgb  # type: ignore[import]
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError("lightgbm is required for alpha model training") from e

        try:
            from sklearn.metrics import (
                accuracy_score,
                roc_auc_score,
                r2_score,
                mean_squared_error,
            )
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.isotonic import IsotonicRegression
        except ImportError:  # pragma: no cover - optional metrics
            accuracy_score = None  # type: ignore[assignment]
            roc_auc_score = None  # type: ignore[assignment]
            r2_score = None  # type: ignore[assignment]
            mean_squared_error = None  # type: ignore[assignment]
            CalibratedClassifierCV = None  # type: ignore[assignment]
            IsotonicRegression = None  # type: ignore[assignment]

        def _safe_rmse(y_true: pd.Series, y_pred: np.ndarray) -> Optional[float]:
            if mean_squared_error is None:
                return None
            try:
                return float(mean_squared_error(y_true, y_pred, squared=False))
            except TypeError:
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))  # type: ignore[call-arg]

        def _alpha_hpo_objective(
            params: Dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            model: Optional[Any] = None,
            cv_folds: int = 5,
            scoring_metric: str = "r2",
        ) -> float:
            """Custom HPO objective emphasizing IC/rank correlation and smoothness.

            This follows the HierarchicalParameterOptimizer objective_func signature
            but focuses on holdout validation (X_val, y_val) rather than CV.
            """
            if model is None or X_val is None or y_val is None or X_val.shape[0] == 0:
                return float("-inf")

            try:
                model.set_params(**params)
                es_rounds = int(config.get("alpha_early_stopping_rounds", 0))

                if es_rounds > 0 and X_val is not None and y_val is not None and X_val.shape[0] > 0:
                    try:
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)],
                            eval_metric="l2",
                            early_stopping_rounds=es_rounds,
                            verbose=False,
                        )
                    except TypeError:
                        # Fallback if early_stopping_rounds or eval_set are unsupported
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
            except Exception:
                return float("-inf")

            y_val_arr = np.asarray(y_val, dtype=float).ravel()
            y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
            if y_val_arr.shape[0] != y_pred_arr.shape[0] or y_val_arr.shape[0] == 0:
                return float("-inf")

            # Core metrics
            ic_pearson = 0.0
            ic_spearman = 0.0
            r2_val = 0.0
            smooth_penalty = 0.0

            try:
                if y_val_arr.shape[0] >= 2:
                    ic_pearson = float(np.corrcoef(y_val_arr, y_pred_arr)[0, 1])
                    if not np.isfinite(ic_pearson):
                        ic_pearson = 0.0
            except Exception:
                ic_pearson = 0.0

            try:
                if y_val_arr.shape[0] >= 2:
                    ic_spearman = float(
                        pd.Series(y_val_arr).corr(pd.Series(y_pred_arr), method="spearman")
                    )
                    if not np.isfinite(ic_spearman):
                        ic_spearman = 0.0
            except Exception:
                ic_spearman = 0.0

            if r2_score is not None:
                try:
                    r2_val = float(r2_score(y_val_arr, y_pred_arr))
                    if not np.isfinite(r2_val):
                        r2_val = 0.0
                except Exception:
                    r2_val = 0.0

            try:
                if y_pred_arr.shape[0] >= 3:
                    diffs = np.diff(y_pred_arr)
                    smooth_penalty = float(np.std(diffs))
                    if not np.isfinite(smooth_penalty):
                        smooth_penalty = 0.0
            except Exception:
                smooth_penalty = 0.0

            # Model complexity from num_leaves (higher -> larger penalty)
            try:
                num_leaves_val = float(
                    params.get("num_leaves", config.get("alpha_num_leaves", 64))
                )
            except Exception:
                num_leaves_val = float(config.get("alpha_num_leaves", 64))
            complexity_penalty = float(np.log1p(max(num_leaves_val, 1.0)))

            # Weights are configurable via config with sensible defaults
            w_ic = float(config.get("alpha_hpo_weight_ic_pearson", 1.0))
            w_ic_spearman = float(config.get("alpha_hpo_weight_ic_spearman", 0.5))
            w_r2 = float(config.get("alpha_hpo_weight_r2", 0.3))
            w_smooth = float(config.get("alpha_hpo_weight_smoothness", 0.1))
            w_complex = float(config.get("alpha_hpo_weight_complexity", 0.05))

            score = (
                w_ic * ic_pearson
                + w_ic_spearman * ic_spearman
                + w_r2 * r2_val
                - w_smooth * smooth_penalty
                - w_complex * complexity_penalty
            )

            return float(score)

        # Default to regression model for continuous alpha
        target_type = str(config.get("alpha_target_type", "regression")).lower()
        horizon = int(config.get("alpha_horizon_bars", 1))

        df = alpha_df.copy()
        if "alpha_target" not in df.columns:
            raise ValueError("alpha_target column not found in dataset")

        df = df.dropna(subset=["alpha_target"])
        if df.empty:
            raise ValueError("No valid samples for alpha model training after dropping NaNs")

        y = df["alpha_target"]

        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [
            col
            for col in numeric_df.columns
            if col != "alpha_target" and not col.startswith("alpha_forward_return_")
        ]

        # Treat regime probability channels as model outputs, not training inputs
        regime_prob_cols = [
            col
            for col in feature_cols
            if col.startswith("regime_") and col.endswith("_prob")
        ]
        if regime_prob_cols:
            tprint_info(
                f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¸ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Excluding regime probability columns from alpha features: {regime_prob_cols}"
            )
            feature_cols = [c for c in feature_cols if c not in regime_prob_cols]

        if not feature_cols:
            raise ValueError("No numeric features available for alpha model training")

        X = numeric_df[feature_cols]

        min_samples = int(config.get("alpha_min_samples", 200))
        if len(X) < max(min_samples, 20):
            raise ValueError(
                f"Insufficient samples for alpha model training: {len(X)} < {min_samples}"
            )

        # Temporal train / validation / test split using split_config
        if split_config is not None:
            # Use temporal split config for proper train/val/test separation
            train_mask = (X.index >= split_config.training.start) & \
                         (X.index <= split_config.training.effective_end)
            val_mask = (X.index >= split_config.validation.start) & \
                       (X.index <= split_config.validation.effective_end)

            X_train_raw = X.loc[train_mask].copy()
            y_train = y.loc[train_mask]
            X_val_raw = X.loc[val_mask].copy()
            y_val = y.loc[val_mask]

            tprint_info(
                f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Temporal splits: train={len(X_train_raw)}, val={len(X_val_raw)}"
            )

            if len(X_train_raw) == 0 or len(X_val_raw) == 0:
                tprint_warning(
                    "Temporal split config produced empty train/val segments for alpha_df; "
                    "falling back to percentage-based split on available alpha samples"
                )
                split_config = None
                train_frac = float(config.get("alpha_train_fraction", 0.8))
                train_frac = min(max(train_frac, 0.5), 0.95)
                split_idx = int(len(X) * train_frac)
                split_idx = max(min(split_idx, len(X) - 1), 1)

                X_train_raw = X.iloc[:split_idx].copy()
                y_train = y.iloc[:split_idx]
                X_val_raw = X.iloc[split_idx:].copy()
                y_val = y.iloc[split_idx:]
        else:
            # Fallback to percentage-based split if no split_config provided
            tprint_warning("No split_config provided, using percentage-based split (legacy fallback)")
            train_frac = float(config.get("alpha_train_fraction", 0.8))
            train_frac = min(max(train_frac, 0.5), 0.95)
            split_idx = int(len(X) * train_frac)
            split_idx = max(min(split_idx, len(X) - 1), 1)

            X_train_raw = X.iloc[:split_idx].copy()
            y_train = y.iloc[:split_idx]
            X_val_raw = X.iloc[split_idx:].copy()
            y_val = y.iloc[split_idx:]

        # Apply rolling window normalization to features to prevent look-ahead bias.
        # Use adaptive routing so spatial distance/level features get ATR normalization,
        # pure volume series can use log1p+zscore, and the rest keep winsorized z-score.
        window_size = int(config.get("alpha_normalization_window", 500))

        # Use OHLC from alpha_df when available for ATR calculation
        high = alpha_df["high"] if "high" in alpha_df.columns else None
        low = alpha_df["low"] if "low" in alpha_df.columns else None
        close = alpha_df["close"] if "close" in alpha_df.columns else None

        # Prepare robust-scaling configuration up front so it is always defined
        outlier_threshold = float(config.get("alpha_outlier_threshold", 3.0))
        normalizer_config: Dict[str, Any] = {
            "default_strategy": "robust",
            "auto_select": False,
            "handle_outliers": True,
            "outlier_threshold": outlier_threshold,
            "use_vectorbt": False,
        }
        scaler = ScalingNormalizer(normalizer_config)
        scaling_strategy = "rolling_window"

        try:
            X_train_scaled = rolling_adaptive_normalize(
                X_train_raw,
                window=window_size,
                min_periods=window_size // 2,
                high=high,
                low=low,
                close=close,
            )
            X_val_scaled = rolling_adaptive_normalize(
                X_val_raw,
                window=window_size,
                min_periods=window_size // 2,
                high=high,
                low=low,
                close=close,
            )
            X_scaled_full = rolling_adaptive_normalize(
                X,
                window=window_size,
                min_periods=window_size // 2,
                high=high,
                low=low,
                close=close,
            )
            tprint_info("ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Applied rolling window normalization to features")
        except Exception as norm_exc:
            tprint_warning(f"Rolling normalization failed, using ScalingNormalizer fallback: {norm_exc}")
            # Fallback to ScalingNormalizer
            scaling_strategy = "robust"
            X_train_scaled = scaler.fit_transform(X_train_raw, strategy="robust")
            X_val_scaled = scaler.transform(X_val_raw)
            X_scaled_full = scaler.transform(X)

        # Optionally apply EWMA temporal smoothing on the scaled space
        use_ewm_features = bool(config.get("alpha_use_ewm_features", True))
        ewma_periods_cfg = config.get("alpha_ewm_periods", [2, 6, 10])
        try:
            ewma_periods = [int(p) for p in ewma_periods_cfg if int(p) > 0]
        except Exception:
            ewma_periods = [2, 6, 10]

        if use_ewm_features and ewma_periods:
            base_df = X_scaled_full.copy()
            feature_names_seq: List[str] = list(base_df.columns)

            aggregated_ewm: Optional[np.ndarray] = None
            n_features = base_df.shape[1]

            for period in ewma_periods:
                alpha_val = 2.0 / float(period + 1)
                try:
                    smoothed_array, _ = apply_ewm_smoothing(
                        base_df.values,
                        alpha=alpha_val,
                        feature_names=feature_names_seq,
                        use_vectorization_optimization=False,
                    )

                    # apply_ewm_smoothing returns [original, ewm] for our usage;
                    # take only the EWM-smoothed block so dimensionality stays constant.
                    if smoothed_array.shape[1] < 2 * n_features:
                        raise ValueError(
                            f"Unexpected smoothed_array shape {smoothed_array.shape} for n_features={n_features}"
                        )

                    ewm_block = smoothed_array[:, n_features:]

                    if aggregated_ewm is None:
                        aggregated_ewm = ewm_block.astype(float)
                    else:
                        aggregated_ewm = aggregated_ewm + ewm_block.astype(float)
                except Exception as e:
                    tprint_warning(
                        f"EWMA temporal smoothing failed for period={period} (using unsmoothed features): {e}"
                    )
                    aggregated_ewm = None
                    break

            if aggregated_ewm is not None:
                aggregated_ewm = aggregated_ewm / float(len(ewma_periods))
                features_df = pd.DataFrame(
                    aggregated_ewm,
                    index=base_df.index,
                    columns=pd.Index(feature_names_seq),
                )

                X_features_full = features_df
                # Preserve the temporal split semantics: when a TemporalSplitConfig
                # is provided, use the train/val masks; otherwise fall back to the
                # legacy percentage-based split via split_idx.
                if split_config is not None:
                    # Use index intersection to avoid KeyError or empty DataFrames
                    # when EWM/normalization drops rows
                    train_idx = X_features_full.index.intersection(X_train_raw.index)
                    val_idx = X_features_full.index.intersection(X_val_raw.index)

                    if len(train_idx) == 0:
                        raise ValueError(
                            f"No training samples remain after feature processing. "
                            f"Original train size: {len(X_train_raw)}, "
                            f"Features available: {len(X_features_full)}"
                        )
                    if len(val_idx) == 0:
                        raise ValueError(
                            f"No validation samples remain after feature processing. "
                            f"Original val size: {len(X_val_raw)}, "
                            f"Features available: {len(X_features_full)}"
                        )

                    X_train = X_features_full.loc[train_idx].copy()
                    X_val = X_features_full.loc[val_idx].copy()
                else:
                    X_train = X_features_full.iloc[:split_idx].copy()
                    X_val = X_features_full.iloc[split_idx:].copy()

                # Validate shapes before proceeding
                if X_train.shape[0] == 0 or X_train.shape[1] == 0:
                    raise ValueError(
                        f"Invalid X_train shape after feature processing: {X_train.shape}"
                    )
                if X_val.shape[0] == 0 or X_val.shape[1] == 0:
                    raise ValueError(
                        f"Invalid X_val shape after feature processing: {X_val.shape}"
                    )

                X_scaled_full = X_features_full
                extended_feature_names = feature_names_seq
            else:
                # Fallback: use robust-scaled features without EWMA smoothing
                X_train = X_train_scaled
                X_val = X_val_scaled
                extended_feature_names = list(X_scaled_full.columns)
        else:
            X_train = X_train_scaled
            X_val = X_val_scaled
            extended_feature_names = list(X_scaled_full.columns)

        # Final validation of data shapes before model training
        if X_train.shape[0] == 0 or X_train.shape[1] == 0:
            raise ValueError(
                f"Invalid X_train shape before model training: {X_train.shape}. "
                f"Cannot proceed with empty or zero-feature data."
            )
        if X_val.shape[0] == 0 or X_val.shape[1] == 0:
            raise ValueError(
                f"Invalid X_val shape before model training: {X_val.shape}. "
                f"Cannot proceed with empty or zero-feature validation data."
            )

        # Initialize training metrics for this model; this dict is returned to the caller
        # and also enriched throughout the training pipeline.
        training_metrics: Dict[str, Any] = {}
        training_metrics["scaling_strategy"] = scaling_strategy
        training_metrics["alpha_outlier_threshold"] = outlier_threshold
        training_metrics["alpha_use_ewm_features"] = use_ewm_features
        training_metrics["alpha_ewm_periods"] = ewma_periods

        # Prepare feature pipeline artifacts for persistence (feature names + scaler state)
        feature_pipeline_artifacts: Dict[str, Any] = {
            "feature_names": extended_feature_names,
            "scaler": scaler,
            "normalizer_config": normalizer_config,
        }

        # Optional hierarchical HPO for LightGBM hyperparameters (regression only, config-gated)
        best_hpo_params: Dict[str, Any] = {}
        enable_hpo = bool(config.get("alpha_enable_hpo", False)) and target_type == "regression"
        if enable_hpo:
            try:
                hpo_param_groups = [
                    ParameterGroup(
                        name="lgbm_core",
                        params={
                            "num_leaves": {"type": "int", "low": 16, "high": 128},
                            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                        },
                        priority=1,
                    ),
                ]

                base_model_for_hpo = lgb.LGBMRegressor(
                    n_estimators=int(config.get("alpha_n_estimators", 300)),
                    random_state=int(config.get("alpha_random_state", 42)),
                )

                optimizer = HierarchicalParameterOptimizer(
                    param_groups=hpo_param_groups,
                    objective_func=_alpha_hpo_objective,
                    stages=[OptimizationStage.COARSE_GRID, OptimizationStage.TPE],
                    cv_folds=int(config.get("alpha_hpo_cv_folds", 3)),
                    scoring_metric="r2",
                    direction="maximize",
                    n_rounds=1,
                    enable_final_refinement=False,
                    final_refinement_trials=int(config.get("alpha_hpo_final_trials", 20)),
                    verbose=True,
                    use_custom_balanced_score=False,
                )

                X_train_hpo = X_train.to_numpy(dtype=float, copy=False)
                y_train_hpo = y_train.to_numpy(dtype=float, copy=False)
                X_val_hpo = X_val.to_numpy(dtype=float, copy=False) if len(X_val) > 0 else None
                y_val_hpo = y_val.to_numpy(dtype=float, copy=False) if len(X_val) > 0 else None

                hpo_result = optimizer.optimize(
                    X_train=X_train_hpo,
                    y_train=y_train_hpo,
                    X_val=X_val_hpo,
                    y_val=y_val_hpo,
                    model=base_model_for_hpo,
                )

                best_hpo_params = hpo_result.best_params or {}
                training_metrics["alpha_hpo_best_score"] = float(hpo_result.best_score)
                training_metrics["alpha_hpo_best_params"] = best_hpo_params
                training_metrics["alpha_hpo_used"] = True
            except Exception as hpo_exc:
                tprint_warning(f"Alpha HPO failed; proceeding with default hyperparameters: {hpo_exc}")
                best_hpo_params = {}
                training_metrics["alpha_hpo_used"] = False
        else:
            training_metrics["alpha_hpo_used"] = False

        if target_type == "regression":
            base_params: Dict[str, Any] = {
                "n_estimators": int(config.get("alpha_n_estimators", 300)),
                "learning_rate": float(config.get("alpha_learning_rate", 0.05)),
                "num_leaves": int(config.get("alpha_num_leaves", 64)),
                "subsample": float(config.get("alpha_subsample", 0.8)),
                "colsample_bytree": float(config.get("alpha_colsample_bytree", 0.8)),
                "random_state": int(config.get("alpha_random_state", 42)),
            }

            if best_hpo_params:
                for key, value in best_hpo_params.items():
                    if key in base_params:
                        base_params[key] = value

            model = lgb.LGBMRegressor(**base_params)
            es_rounds = int(config.get("alpha_early_stopping_rounds", 0))
            if es_rounds > 0 and len(X_val) > 0:
                try:
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="l2",
                        early_stopping_rounds=es_rounds,
                    )
                except TypeError:
                    # Fallback if early stopping or eval_set are unsupported
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            if r2_score is not None:
                training_metrics["train_r2"] = float(r2_score(y_train, train_pred))
            if mean_squared_error is not None:
                rmse_val = _safe_rmse(y_train, train_pred)
                if rmse_val is not None:
                    training_metrics["train_rmse"] = rmse_val

            regression_calibration_enabled = bool(config.get("alpha_enable_regression_calibration", True))
            training_metrics["regression_calibration_enabled"] = regression_calibration_enabled

            calibrator = None

            if len(X_val) > 0:
                val_pred = model.predict(X_val)
                if r2_score is not None:
                    training_metrics["val_r2"] = float(r2_score(y_val, val_pred))
                if mean_squared_error is not None:
                    rmse_val = _safe_rmse(y_val, val_pred)
                    if rmse_val is not None:
                        training_metrics["val_rmse"] = rmse_val

                if regression_calibration_enabled and IsotonicRegression is not None:
                    try:
                        tprint_info(f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ§ Starting regression calibration (Isotonic Regression) on {len(y_val)} validation samples...")
                        calibrator = IsotonicRegression(out_of_bounds="clip")
                        calibrator.fit(val_pred, y_val.to_numpy(dtype=float, copy=False))

                        if mean_squared_error is not None:
                            rmse_uncal = _safe_rmse(y_val, val_pred)
                            val_pred_cal = calibrator.predict(val_pred)
                            rmse_cal = _safe_rmse(y_val, val_pred_cal) if mean_squared_error is not None else None
                            if rmse_uncal is not None:
                                training_metrics["val_rmse_uncalibrated"] = rmse_uncal
                            if rmse_cal is not None:
                                training_metrics["val_rmse_calibrated"] = rmse_cal

                            # Log calibration improvement
                            if rmse_uncal is not None and rmse_cal is not None:
                                improvement = ((rmse_uncal - rmse_cal) / rmse_uncal) * 100 if rmse_uncal > 0 else 0.0
                                tprint_info(f"  ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Regression calibration complete: RMSE {rmse_uncal:.6f} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {rmse_cal:.6f} (improvement: {improvement:.2f}%)")
                            else:
                                tprint_info(f"  ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Regression calibration complete")

                        training_metrics["regression_calibration_method"] = "isotonic_regression"
                        training_metrics["regression_calibration_used"] = True
                    except Exception as calib_err:
                        calibrator = None
                        training_metrics["regression_calibration_used"] = False
                        training_metrics["regression_calibration_failed"] = True
                        training_metrics["regression_calibration_error"] = str(calib_err)
                        tprint_warning(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¸ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Regression calibration failed: {calib_err}")
                elif not regression_calibration_enabled:
                    training_metrics["regression_calibration_used"] = False
                    tprint_info("ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¸ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ  Regression calibration disabled (alpha_enable_regression_calibration=False)")
                elif IsotonicRegression is None:
                    training_metrics["regression_calibration_used"] = False
                    tprint_warning("ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¸ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Regression calibration unavailable (IsotonicRegression not imported)")

            if calibrator is not None:
                full_raw_pred = model.predict(X_scaled_full)
                full_scores = calibrator.predict(full_raw_pred)
            else:
                full_scores = model.predict(X_scaled_full)

            scores = pd.Series(full_scores, index=df.index, name="alpha_pred_return")
            pred_col_name = "alpha_pred_return"
            training_metrics["model_type"] = "lightgbm_regression"

        else:
            base_model = lgb.LGBMClassifier(
                n_estimators=int(config.get("alpha_n_estimators", 300)),
                learning_rate=float(config.get("alpha_learning_rate", 0.05)),
                num_leaves=int(config.get("alpha_num_leaves", 64)),
                subsample=float(config.get("alpha_subsample", 0.8)),
                colsample_bytree=float(config.get("alpha_colsample_bytree", 0.8)),
                random_state=int(config.get("alpha_random_state", 42)),
            )
            base_model.fit(X_train, y_train)

            train_proba = base_model.predict_proba(X_train)[:, 1]
            train_pred = (train_proba > 0.5).astype(float)

            if roc_auc_score is not None:
                training_metrics["train_auc_uncalibrated"] = float(roc_auc_score(y_train, train_proba))
            if accuracy_score is not None:
                training_metrics["train_accuracy_uncalibrated"] = float(accuracy_score(y_train, train_pred))

            # Probability calibration using CalibratedClassifierCV with Isotonic Regression
            calibration_enabled = bool(config.get("alpha_enable_probability_calibration", True))
            training_metrics["probability_calibration_enabled"] = calibration_enabled

            model = base_model
            calibration_metrics = {}

            if calibration_enabled and CalibratedClassifierCV is not None and len(X_val) > 0:
                try:
                    # Wrap base model with CalibratedClassifierCV using Isotonic Regression
                    model = CalibratedClassifierCV(
                        base_model,
                        method='isotonic',
                        cv='prefit'  # Use the already-trained model
                    )
                    # Fit calibration on validation set
                    model.fit(X_val, y_val)

                    tprint_info(
                        f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Probability calibration (Isotonic Regression) fitted on {len(X_val)} validation samples"
                    )

                    # Evaluate calibration improvement
                    val_proba_calibrated = model.predict_proba(X_val)[:, 1]
                    val_proba_uncalibrated = base_model.predict_proba(X_val)[:, 1]

                    # Calibration quality metrics
                    if roc_auc_score is not None:
                        auc_cal = float(roc_auc_score(y_val, val_proba_calibrated))
                        auc_uncal = float(roc_auc_score(y_val, val_proba_uncalibrated))
                        calibration_metrics["val_auc_calibrated"] = auc_cal
                        calibration_metrics["val_auc_uncalibrated"] = auc_uncal
                        calibration_metrics["auc_improvement"] = auc_cal - auc_uncal
                        training_metrics.update(calibration_metrics)

                    # Expected Calibration Error (ECE) - simpler alternative to Brier score
                    # Divide probabilities into bins and measure gap between average prob and accuracy
                    try:
                        n_bins = 10
                        bin_edges = np.linspace(0, 1, n_bins + 1)
                        bin_indices = np.digitize(val_proba_uncalibrated, bin_edges) - 1
                        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                        ece_uncal = 0.0
                        ece_cal = 0.0
                        for bin_idx in range(n_bins):
                            mask = bin_indices == bin_idx
                            if mask.sum() > 0:
                                bin_acc_uncal = float((y_val[mask] == 1).mean())
                                bin_prob_uncal = float(val_proba_uncalibrated[mask].mean())
                                ece_uncal += abs(bin_acc_uncal - bin_prob_uncal) * (mask.sum() / len(y_val))

                                bin_acc_cal = float((y_val[mask] == 1).mean())
                                bin_prob_cal = float(val_proba_calibrated[mask].mean())
                                ece_cal += abs(bin_acc_cal - bin_prob_cal) * (mask.sum() / len(y_val))

                        training_metrics["ece_uncalibrated"] = float(ece_uncal)
                        training_metrics["ece_calibrated"] = float(ece_cal)
                        training_metrics["ece_improvement"] = float(ece_uncal - ece_cal)
                    except Exception as ece_err:
                        tprint_warning(f"ECE calculation failed: {ece_err}")

                    training_metrics["calibration_method"] = "isotonic_regression"

                except Exception as calib_err:
                    tprint_warning(f"Probability calibration failed, using uncalibrated model: {calib_err}")
                    model = base_model
                    training_metrics["calibration_failed"] = True
                    training_metrics["calibration_error"] = str(calib_err)
            elif not calibration_enabled:
                tprint_info("Probability calibration disabled via config")
            elif CalibratedClassifierCV is None:
                tprint_warning("CalibratedClassifierCV not available; skipping probability calibration")
            elif len(X_val) == 0:
                tprint_warning("No validation set available; skipping probability calibration")

            # Evaluate on validation set with calibrated probabilities
            if len(X_val) > 0:
                val_proba = model.predict_proba(X_val)[:, 1]
                val_pred = (val_proba > 0.5).astype(float)
                if roc_auc_score is not None:
                    training_metrics["val_auc"] = float(roc_auc_score(y_val, val_proba))
                if accuracy_score is not None:
                    training_metrics["val_accuracy"] = float(accuracy_score(y_val, val_pred))

                # Walk-Forward Validation on validation set to detect concept drift
                try:
                    wfv_metrics = self._calculate_walk_forward_validation_classification(
                        X_val=X_val.to_numpy(dtype=float, copy=False) if hasattr(X_val, 'to_numpy') else X_val,
                        y_val=y_val.to_numpy(dtype=float, copy=False) if hasattr(y_val, 'to_numpy') else y_val,
                        model=base_model,
                        config=config,
                        accuracy_score=accuracy_score
                    )
                    if wfv_metrics:
                        training_metrics.update(wfv_metrics)
                        tprint_info(
                            f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Walk-Forward Validation completed: "
                            f"avg_val_accuracy={wfv_metrics.get('wfv_avg_val_accuracy', 0.0):.3f}, "
                            f"avg_test_accuracy={wfv_metrics.get('wfv_avg_test_accuracy', 0.0):.3f}, "
                            f"accuracy_degradation={wfv_metrics.get('wfv_accuracy_degradation', 0.0):.3f}"
                        )
                except Exception as wfv_err:
                    tprint_warning(f"Walk-Forward Validation failed: {wfv_err}")

            # Get predictions on full dataset using calibrated model
            proba_all = model.predict_proba(X_scaled_full)[:, 1]
            scores = pd.Series(proba_all, index=df.index, name="alpha_pred_prob")
            pred_col_name = "alpha_pred_prob"
            training_metrics["model_type"] = "lightgbm_classification"

        full_scores = scores.reindex(alpha_df.index)

        # Optional SHAP analysis add-on (config-gated)
        if bool(config.get("alpha_enable_shap", False)):
            try:
                from src.utils.ml_common.explainability.model_explanations import (
                    explain_model_with_shap_lime,
                )

                X_train_arr = X_train.to_numpy(dtype=float, copy=False)
                if len(X_val) > 0:
                    X_test_arr = X_val.to_numpy(dtype=float, copy=False)
                else:
                    X_test_arr = X_train_arr

                shap_cfg: Dict[str, Any] = {
                    "enable_shap": True,
                    "enable_lime": False,
                    "shap_sample_size": int(config.get("alpha_shap_sample_size", 128)),
                }

                shap_results = explain_model_with_shap_lime(
                    model=model,
                    X_train=X_train_arr,
                    X_test=X_test_arr,
                    feature_names=extended_feature_names,
                    model_name="hmm_alpha_model",
                    config=shap_cfg,
                )

                shap_expl = shap_results.get("shap_explanations", {}) if isinstance(shap_results, dict) else {}
                if isinstance(shap_expl, dict):
                    training_metrics["alpha_shap_top_features"] = shap_expl.get("top_features", [])
                    training_metrics["alpha_shap_mean_importance"] = shap_expl.get("mean_importance")
            except Exception as shap_exc:
                tprint_warning(f"Alpha SHAP analysis failed (ignored): {shap_exc}")

        tprint_info(
            f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ¤ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Trained LightGBM alpha model ({training_metrics.get('model_type', 'unknown')}) "
            f"on {len(X_train)} train / {len(X_val)} val samples"
        )

        training_metrics["alpha_horizon_bars"] = horizon
        training_metrics["target_type"] = target_type

        # Comprehensive feature analysis (IC, importance metrics, mRMR, learning curves)
        try:
            feature_analysis = self._perform_comprehensive_feature_analysis(
                model=model if target_type == "regression" else base_model if 'base_model' in locals() else model,
                X_train=X_train_scaled if hasattr(X_train_scaled, 'index') else pd.DataFrame(X_train_scaled, columns=extended_feature_names),
                y_train=y_train,
                X_val=X_val_scaled if len(X_val) > 0 and (hasattr(X_val_scaled, 'index') or True) else (pd.DataFrame(X_val_scaled, columns=extended_feature_names) if len(X_val) > 0 else None),
                y_val=y_val if len(y_val) > 0 else None,
                X_full=X_scaled_full if hasattr(X_scaled_full, 'index') else pd.DataFrame(X_scaled_full, columns=extended_feature_names),
                y_full=y,
                feature_names=extended_feature_names,
                config=config,
                is_classification=(target_type == "classification"),
            )
            if feature_analysis.get("feature_analysis_completed"):
                training_metrics.update(feature_analysis)
                tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Comprehensive feature analysis completed and integrated into metrics")

                # Optional auto-prune experiment: try multiple quantile-based thresholds
                # over permutation importance and adopt the best pruned model if it
                # improves validation R^2 by more than alpha_auto_prune_min_delta.
                if (
                    target_type == "regression"
                    and bool(config.get("alpha_enable_auto_prune_rerun", False))
                ):
                    try:
                        perm_info = feature_analysis.get("permutation_importance") or training_metrics.get("permutation_importance")
                        if isinstance(perm_info, dict) and perm_info:
                            imp_values = {}
                            for fname, finfo in perm_info.items():
                                try:
                                    imp_values[fname] = float(finfo.get("importance_mean", 0.0))
                                except Exception:
                                    continue

                            if imp_values:
                                imp_series = pd.Series(imp_values)

                                # Ensure we have a proper DataFrame view of features
                                if hasattr(X_scaled_full, "columns"):
                                    X_full_df = X_scaled_full
                                else:
                                    X_full_df = pd.DataFrame(
                                        X_scaled_full,
                                        columns=extended_feature_names,
                                    )

                                # Configuration for quantile-based pruning
                                quantiles_cfg = config.get("alpha_auto_prune_quantiles", [0.1, 0.2, 0.3])
                                try:
                                    quantiles = sorted(
                                        {
                                            float(q)
                                            for q in quantiles_cfg
                                            if 0.0 < float(q) < 1.0
                                        }
                                    )
                                except Exception:
                                    quantiles = [0.1, 0.2, 0.3]

                                baseline_val_r2 = training_metrics.get("val_r2")
                                best_val_r2 = baseline_val_r2
                                best_model = None
                                best_drop_features: List[str] = []
                                best_quantile: Optional[float] = None
                                best_X_pruned: Optional[pd.DataFrame] = None

                                # Reuse gentle minimum-keep heuristics from pruning config
                                min_fraction = float(
                                    config.get("alpha_feature_pruning_min_fraction", 0.6)
                                )
                                min_fraction = min(max(min_fraction, 0.2), 0.9)
                                min_absolute = int(
                                    config.get("alpha_feature_pruning_min_absolute", 5)
                                )
                                n_features_total = len(extended_feature_names)
                                min_keep = int(
                                    max(n_features_total * min_fraction, float(min_absolute), 1.0)
                                )

                                candidate_summaries = []

                                for q in quantiles:
                                    try:
                                        threshold = imp_series.quantile(q)
                                        drop_features = [
                                            name
                                            for name, val in imp_series.items()
                                            if val <= threshold and name in extended_feature_names
                                        ]

                                        # Ensure we keep enough features
                                        if not drop_features:
                                            continue
                                        if n_features_total - len(drop_features) < min_keep:
                                            continue

                                        X_pruned = X_full_df.drop(columns=drop_features, errors="ignore")
                                        X_train_pruned = X_pruned.iloc[:split_idx]
                                        X_val_pruned = X_pruned.iloc[split_idx:]

                                        pruned_model = lgb.LGBMRegressor(**base_params)
                                        pruned_model.fit(X_train_pruned, y_train)

                                        pruned_val_r2 = None
                                        if len(X_val_pruned) > 0 and r2_score is not None:
                                            try:
                                                val_pred_pruned = pruned_model.predict(X_val_pruned)
                                                pruned_val_r2 = float(r2_score(y_val, val_pred_pruned))
                                            except Exception:
                                                pruned_val_r2 = None

                                        candidate_summaries.append(
                                            {
                                                "quantile": q,
                                                "threshold": float(threshold),
                                                "n_dropped": len(drop_features),
                                                "dropped_features": drop_features,
                                                "val_r2": pruned_val_r2,
                                            }
                                        )

                                        if pruned_val_r2 is not None and (
                                            best_val_r2 is None
                                            or pruned_val_r2 > best_val_r2
                                        ):
                                            best_val_r2 = pruned_val_r2
                                            best_model = pruned_model
                                            best_drop_features = drop_features
                                            best_quantile = q
                                            best_X_pruned = X_pruned
                                    except Exception as cand_err:
                                        tprint_warning(
                                            f"Auto-prune candidate at quantile={q} failed (ignored): {cand_err}"
                                        )

                                training_metrics["auto_prune_enabled"] = True
                                training_metrics["auto_prune_candidates"] = candidate_summaries
                                training_metrics["auto_prune_baseline_val_r2"] = (
                                    float(baseline_val_r2)
                                    if baseline_val_r2 is not None
                                    else None
                                )

                                epsilon = float(
                                    config.get("alpha_auto_prune_min_delta", 0.0)
                                )

                                adopt_pruned = (
                                    best_model is not None
                                    and baseline_val_r2 is not None
                                    and best_val_r2 is not None
                                    and best_val_r2 > baseline_val_r2 + epsilon
                                )

                                training_metrics["auto_prune_adopted"] = bool(adopt_pruned)
                                training_metrics["auto_prune_best_quantile"] = (
                                    float(best_quantile) if best_quantile is not None else None
                                )
                                training_metrics["auto_prune_best_val_r2"] = (
                                    float(best_val_r2)
                                    if best_val_r2 is not None
                                    else None
                                )
                                training_metrics["auto_prune_best_dropped_features"] = (
                                    best_drop_features
                                )

                                if adopt_pruned and best_model is not None and best_X_pruned is not None:
                                    # Adopt pruned model and feature set as primary
                                    model = best_model
                                    X_scaled_full = best_X_pruned
                                    extended_feature_names = [
                                        n for n in extended_feature_names if n not in best_drop_features
                                    ]
                                    feature_pipeline_artifacts["feature_names"] = (
                                        extended_feature_names
                                    )

                                    try:
                                        # Recompute full scores with pruned model
                                        full_raw_scores_pruned = model.predict(X_scaled_full)
                                        scores = pd.Series(
                                            full_raw_scores_pruned,
                                            index=df.index,
                                            name=pred_col_name,
                                        )
                                        full_scores = scores.reindex(alpha_df.index)
                                    except Exception as pred_err:
                                        tprint_warning(
                                            f"Auto-prune adoption prediction failed (non-fatal): {pred_err}"
                                        )

                                    tprint_info(
                                        "ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Auto-prune ADOPTED: quantile="
                                        f"{best_quantile}, dropped {len(best_drop_features)} features | "
                                        f"val_r2 {baseline_val_r2 if baseline_val_r2 is not None else 'N/A'} ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {best_val_r2:.6f}"
                                    )
                                else:
                                    tprint_info(
                                        "ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Auto-prune experiment completed: "
                                        f"baseline val_r2={baseline_val_r2 if baseline_val_r2 is not None else 'N/A'}, "
                                        f"best_pruned_val_r2={best_val_r2 if best_val_r2 is not None else 'N/A'} (no adoption)"
                                    )
                        else:
                            training_metrics["auto_prune_enabled"] = False
                    except Exception as auto_prune_err:
                        tprint_warning(f"Auto-prune experiment failed (non-fatal): {auto_prune_err}")
        except Exception as feature_analysis_err:
            tprint_warning(f"Comprehensive feature analysis integration failed (non-fatal): {feature_analysis_err}")

        return model, full_scores, pred_col_name, training_metrics, feature_pipeline_artifacts

    def _calculate_iqr_winsorization_percentiles(
        self, data: pd.Series
    ) -> Tuple[float, float]:
        """Calculate winsorization percentiles using the IQR method.

        Returns fence values at Q1 - 1.5*IQR and Q3 + 1.5*IQR, then converts
        them to percentiles in the data distribution.

        Args:
            data: Series of numeric values

        Returns:
            Tuple of (lower_percentile, upper_percentile) for winsorization
        """
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # Convert fence values to percentiles
        lower_pct = (data <= lower_fence).sum() / len(data)
        upper_pct = (data >= upper_fence).sum() / len(data)

        # Ensure we have reasonable bounds (at least 1%, at most 20%)
        lower_pct = max(0.01, min(lower_pct, 0.20))
        upper_pct = max(0.01, min(upper_pct, 0.20))

        return lower_pct, upper_pct

    def _calculate_winsorized_cv(
        self, data: pd.Series, lower_pct: Optional[float] = None, upper_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """Calculate both standard CV and Winsorized CV.

        If percentiles are not provided, uses IQR method to determine them.

        Args:
            data: Series of numeric values
            lower_pct: Lower percentile for winsorization (optional)
            upper_pct: Upper percentile for winsorization (optional)

        Returns:
            Tuple of (standard_cv, winsorized_cv)
        """
        data_clean = data.dropna()
        if len(data_clean) < 2:
            return 0.0, 0.0

        # Standard CV
        mean_val = data_clean.mean()
        std_val = data_clean.std()
        cv = abs(std_val / (mean_val + 1e-8))

        # Winsorized CV using IQR method if percentiles not provided
        if lower_pct is None or upper_pct is None:
            lower_pct, upper_pct = self._calculate_iqr_winsorization_percentiles(data_clean)

        lower_bound = data_clean.quantile(lower_pct)
        upper_bound = data_clean.quantile(1.0 - upper_pct)

        # Winsorize: cap values at bounds
        winsorized = data_clean.clip(lower=lower_bound, upper=upper_bound)

        win_mean = winsorized.mean()
        win_std = winsorized.std()
        wcv = abs(win_std / (win_mean + 1e-8))

        return cv, wcv

    def _calculate_economic_cv_ratio(
        self,
        regime_labels: np.ndarray,
        forward_returns: pd.Series,
        use_winsorized: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate Economic CV Ratio: Between-Regime CV / Within-Regime CV.

        This measures signal-to-noise: how distinct regime means are (signal)
        versus how noisy returns are within each regime (noise).

        Args:
            regime_labels: Array of regime assignments
            forward_returns: Series of forward returns aligned with labels
            use_winsorized: Whether to use winsorized CV (default: True)

        Returns:
            Tuple of (cv_ratio, metrics_dict) where metrics_dict contains:
                - between_cv / between_wcv
                - within_cv / within_wcv (sample-size weighted)
                - cv_ratio / wcv_ratio
        """
        unique_regimes = np.unique(regime_labels)
        if len(unique_regimes) < 2:
            return 0.0, {}

        # Calculate per-regime statistics
        regime_means = []
        regime_sizes = []
        within_cvs = []
        within_wcvs = []

        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_returns = forward_returns[mask].dropna()

            if len(regime_returns) < 2:
                continue

            regime_mean = regime_returns.mean()
            regime_means.append(regime_mean)
            regime_sizes.append(len(regime_returns))

            # Within-regime CV and WCV
            cv, wcv = self._calculate_winsorized_cv(regime_returns)
            within_cvs.append(cv)
            within_wcvs.append(wcv)

        if len(regime_means) < 2:
            return 0.0, {}

        # Between-Regime CV: variability of regime means
        regime_means_series = pd.Series(regime_means)
        between_mean = regime_means_series.mean()
        between_std = regime_means_series.std()
        between_cv = abs(between_std / (between_mean + 1e-8))

        # For between WCV, we winsorize the regime means themselves
        _, between_wcv = self._calculate_winsorized_cv(regime_means_series)

        # Within-Regime CV: sample-size weighted average of within-regime CVs
        total_samples = sum(regime_sizes)
        weights = [size / total_samples for size in regime_sizes]

        within_cv_weighted = sum(cv * w for cv, w in zip(within_cvs, weights))
        within_wcv_weighted = sum(wcv * w for wcv, w in zip(within_wcvs, weights))

        # CV Ratio: signal-to-noise
        cv_ratio = between_cv / (within_cv_weighted + 1e-8)
        wcv_ratio = between_wcv / (within_wcv_weighted + 1e-8)

        metrics = {
            "between_cv": between_cv,
            "between_wcv": between_wcv,
            "within_cv": within_cv_weighted,
            "within_wcv": within_wcv_weighted,
            "cv_ratio": cv_ratio,
            "wcv_ratio": wcv_ratio,
            "n_regimes": len(unique_regimes),
            "total_samples": total_samples,
        }

        # Use winsorized ratio if requested
        final_ratio = wcv_ratio if use_winsorized else cv_ratio

        return final_ratio, metrics

    def _optimize_regime_boundaries(
        self,
        *,
        alpha_scores: pd.Series,
        forward_returns: pd.Series,
        n_regimes: int,
        min_bin_pct: float = 0.10,
        max_bin_pct: float = 0.35,
        jiggle_pct: float = 0.01,
        max_iterations: int = 100,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Optimize regime boundaries to maximize Economic CV Ratio.

        Uses iterative "jiggle" search: starts with quantile boundaries,
        then systematically shifts each boundary to improve the objective.

        Args:
            alpha_scores: Predicted alpha scores (feature)
            forward_returns: Forward returns (target)
            n_regimes: Number of regimes to create
            min_bin_pct: Minimum percentage of samples per bin (default: 10%)
            max_bin_pct: Maximum percentage of samples per bin (default: 35%)
            jiggle_pct: Percentage of data to shift boundaries by (default: 1%)
            max_iterations: Maximum optimization iterations (default: 100)

        Returns:
            Tuple of (optimal_labels, best_score, metrics_dict)
        """
        # Sort data by alpha scores
        sorted_indices = alpha_scores.argsort()
        sorted_scores = alpha_scores.iloc[sorted_indices].values
        sorted_returns = forward_returns.iloc[sorted_indices]

        n_samples = len(sorted_scores)
        min_bin_size = int(n_samples * min_bin_pct)
        max_bin_size = int(n_samples * max_bin_pct)
        jiggle_size = max(1, int(n_samples * jiggle_pct))

        # Initialize with quantile boundaries
        quantile_indices = [int(n_samples * i / n_regimes) for i in range(1, n_regimes)]
        current_boundaries = np.array(quantile_indices, dtype=int)

        # Ensure boundaries respect min/max bin size
        current_boundaries = self._enforce_bin_constraints(
            current_boundaries, n_samples, min_bin_size, max_bin_size
        )

        # Calculate initial score
        current_labels = self._boundaries_to_labels(current_boundaries, n_samples)
        current_score, current_metrics = self._calculate_economic_cv_ratio(
            current_labels, sorted_returns, use_winsorized=True
        )

        best_boundaries = current_boundaries.copy()
        best_score = current_score
        best_metrics = current_metrics

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Lightweight iteration progress (log every 10 iterations and first/last)
            if iteration == 1 or iteration == max_iterations or iteration % 10 == 0:
                tprint_info(
                    f"    ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ regime boundary optimization iter={iteration}/{max_iterations} "
                    f"(current_best_WCV_ratio={best_score:.4f})"
                )

            # Try jiggling each boundary
            for i in range(len(current_boundaries)):
                # Try moving left
                test_boundaries = current_boundaries.copy()
                test_boundaries[i] = max(
                    test_boundaries[i] - jiggle_size,
                    test_boundaries[i - 1] + min_bin_size if i > 0 else min_bin_size
                )

                # Check if this configuration satisfies constraints
                if self._check_bin_constraints(test_boundaries, n_samples, min_bin_size, max_bin_size):
                    test_labels = self._boundaries_to_labels(test_boundaries, n_samples)
                    test_score, test_metrics = self._calculate_economic_cv_ratio(
                        test_labels, sorted_returns, use_winsorized=True
                    )

                    if test_score > best_score:
                        best_boundaries = test_boundaries.copy()
                        best_score = test_score
                        best_metrics = test_metrics
                        improved = True

                # Try moving right
                test_boundaries = current_boundaries.copy()
                test_boundaries[i] = min(
                    test_boundaries[i] + jiggle_size,
                    (test_boundaries[i + 1] - min_bin_size if i < len(current_boundaries) - 1
                     else n_samples - min_bin_size)
                )

                if self._check_bin_constraints(test_boundaries, n_samples, min_bin_size, max_bin_size):
                    test_labels = self._boundaries_to_labels(test_boundaries, n_samples)
                    test_score, test_metrics = self._calculate_economic_cv_ratio(
                        test_labels, sorted_returns, use_winsorized=True
                    )

                    if test_score > best_score:
                        best_boundaries = test_boundaries.copy()
                        best_score = test_score
                        best_metrics = test_metrics
                        improved = True

            if improved:
                current_boundaries = best_boundaries.copy()
                current_score = best_score

        # Convert sorted labels back to original order
        optimal_labels_sorted = self._boundaries_to_labels(best_boundaries, n_samples)
        optimal_labels = np.empty(n_samples, dtype=int)
        optimal_labels[sorted_indices] = optimal_labels_sorted

        best_metrics["optimization_iterations"] = iteration
        best_metrics["converged"] = not improved or iteration < max_iterations

        return optimal_labels, best_score, best_metrics

    def _enforce_bin_constraints(
        self,
        boundaries: np.ndarray,
        n_samples: int,
        min_bin_size: int,
        max_bin_size: int,
    ) -> np.ndarray:
        """Enforce minimum and maximum bin size constraints on boundaries."""
        adjusted = boundaries.copy()

        # Ensure first bin is large enough
        adjusted[0] = max(adjusted[0], min_bin_size)

        # Ensure spacing between boundaries
        for i in range(1, len(adjusted)):
            adjusted[i] = max(adjusted[i], adjusted[i-1] + min_bin_size)

        # Ensure last bin is large enough
        adjusted[-1] = min(adjusted[-1], n_samples - min_bin_size)

        return adjusted

    def _check_bin_constraints(
        self,
        boundaries: np.ndarray,
        n_samples: int,
        min_bin_size: int,
        max_bin_size: int,
    ) -> bool:
        """Check if boundaries satisfy bin size constraints."""
        # Check first bin
        if boundaries[0] < min_bin_size or boundaries[0] > max_bin_size:
            return False

        # Check middle bins
        for i in range(1, len(boundaries)):
            bin_size = boundaries[i] - boundaries[i-1]
            if bin_size < min_bin_size or bin_size > max_bin_size:
                return False

        # Check last bin
        last_bin_size = n_samples - boundaries[-1]
        if last_bin_size < min_bin_size or last_bin_size > max_bin_size:
            return False

        return True

    def _boundaries_to_labels(self, boundaries: np.ndarray, n_samples: int) -> np.ndarray:
        """Convert boundary indices to regime labels."""
        labels = np.zeros(n_samples, dtype=int)

        for i, boundary in enumerate(boundaries):
            labels[boundary:] = i + 1

        return labels

    def _assign_alpha_regimes(
        self,
        alpha_df: pd.DataFrame,
        alpha_scores: pd.Series,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str]]:
        """Derive alpha regimes using flexible quantile optimization.

        This method now:
        1. Tests multiple regime counts (4, 5, 6)
        2. Optimizes boundaries to maximize Economic CV Ratio
        3. Enforces 10-35% bin size constraints
        4. Uses sample-size weighted Within-CV
        5. Reports both CV and Winsorized CV metrics
        """
        # Get configuration for regime optimization
        use_flexible_regimes = bool(config.get("alpha_use_flexible_regimes", True))
        regime_counts_to_test = config.get("alpha_regime_counts", [2, 3, 4])
        min_bin_pct = float(config.get("alpha_min_bin_pct", 0.15))
        max_bin_pct = float(config.get("alpha_max_bin_pct", 0.35))

        # Validate regime counts within compact 2–4 range
        if isinstance(regime_counts_to_test, int):
            regime_counts_to_test = [regime_counts_to_test]
        regime_counts_to_test = [n for n in regime_counts_to_test if 2 <= n <= 4]
        if not regime_counts_to_test:
            regime_counts_to_test = [4]  # Fallback to 4 regimes

        tprint_info(
            "HMM alpha regime assignment starting: "
            f"samples={len(alpha_scores)}, "
            f"alpha_df_rows={len(alpha_df)}, "
            f"use_flexible_regimes={use_flexible_regimes}, "
            f"regime_counts_to_test={regime_counts_to_test}"
        )

        # Optional smoothing of alpha scores before constructing regimes
        score_smoothing_method = str(
            config.get("alpha_score_smoothing_method", "none")
        ).lower()
        score_smoothing_window = int(config.get("alpha_score_smoothing_window", 1))
        scores_for_binning = alpha_scores.copy()
        if score_smoothing_method != "none" and score_smoothing_window > 1:
            try:
                if score_smoothing_method == "ewm":
                    # Use halflife-based EWM so that older samples retain more weight
                    # than with the default span-based formulation.
                    halflife = float(max(score_smoothing_window, 1))
                    scores_for_binning = scores_for_binning.ewm(
                        halflife=halflife,
                        adjust=False,
                    ).mean()
                elif score_smoothing_method == "rolling_median":
                    scores_for_binning = (
                        scores_for_binning
                        .rolling(window=score_smoothing_window, min_periods=1)
                        .median()
                    )
                elif score_smoothing_method == "rolling_mean":
                    scores_for_binning = (
                        scores_for_binning
                        .rolling(window=score_smoothing_window, min_periods=1)
                        .mean()
                    )
                tprint_info(
                    f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ§ÄÂĂÂÄÂĂÂÄÂÄšÄÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Applied {score_smoothing_method} smoothing to alpha scores "
                    f"(window={score_smoothing_window}) before regime binning"
                )
            except Exception as score_smooth_exc:
                tprint_warning(
                    "Alpha score smoothing failed (ignored, using raw scores): "
                    f"{score_smooth_exc}"
                )
                scores_for_binning = alpha_scores

        valid_scores = scores_for_binning.dropna()
        if valid_scores.empty:
            tprint_warning(
                f"Not enough valid alpha scores ({len(valid_scores)}) to define regimes"
            )
            tprint_info(
                "HMM alpha regime assignment diagnostics: "
                f"total_scores={len(alpha_scores)}, "
                f"non_nan_scores={len(valid_scores)}, "
                f"nan_scores={int(len(alpha_scores) - len(valid_scores))}"
            )
            return alpha_df, None, None

        # ------------------------------------------------------------------
        # SIMPLE SCORE-ONLY REGIME ASSIGNMENT (robust default)
        # ------------------------------------------------------------------
        try:
            num_bins_simple = int(config.get("alpha_regime_bins", 4))
        except Exception:
            num_bins_simple = 4
        # Clamp to a compact 2–4 regime range to avoid overly fragmented regimes
        num_bins_simple = max(2, min(num_bins_simple, 4))

        if len(valid_scores) < num_bins_simple:
            if len(valid_scores) >= 3:
                num_bins_simple = len(valid_scores)
                tprint_warning(
                    f"Reducing alpha regime bins to {num_bins_simple} due to limited samples "
                    "for score-only regimes"
                )
            else:
                tprint_warning(
                    f"Not enough valid alpha scores ({len(valid_scores)}) to define score-only regimes"
                )
                return alpha_df, None, None

        ranks_simple = valid_scores.rank(method="first")
        try:
            bucket_codes_simple = pd.qcut(
                ranks_simple,
                q=num_bins_simple,
                labels=False,
                duplicates="drop",
            )
            effective_bins_simple = int(pd.Series(bucket_codes_simple).nunique())
            if effective_bins_simple < 2:
                median_rank = ranks_simple.median()
                bucket_codes_simple = (ranks_simple > median_rank).astype(int)
                num_bins_simple = 2
        except ValueError as e:
            tprint_warning(
                f"Failed to compute score-only quantile alpha regimes with qcut: {e}; "
                "falling back to binary median split"
            )
            median_rank = ranks_simple.median()
            bucket_codes_simple = (ranks_simple > median_rank).astype(int)
            num_bins_simple = 2

        bucket_col_simple = f"alpha_regime_bucket_{num_bins_simple}"
        alpha_df[bucket_col_simple] = bucket_codes_simple.reindex(alpha_df.index)

        # For now, skip regime_stats_df from this simplified path; downstream
        # consumers already guard on regime_stats_df being None.
        return alpha_df, None, bucket_col_simple

        # Get forward returns for optimization
        fwd_cols = [col for col in alpha_df.columns if col.startswith("alpha_forward_return_")]
        if not fwd_cols:
            tprint_warning("No alpha_forward_return column found for regime optimization")
            tprint_info(
                "HMM alpha regime assignment diagnostics: "
                f"available_columns={list(alpha_df.columns)}"
            )
            return alpha_df, None, None

        fwd_col = fwd_cols[0]
        forward_returns = alpha_df[fwd_col].dropna()

        # Align scores and returns
        common_idx = valid_scores.index.intersection(forward_returns.index)
        if len(common_idx) < 20:
            # If we don't have enough overlapping samples to run the
            # forward-return-driven optimization, still proceed with
            # score-only quantile regimes instead of returning None.
            tprint_warning(
                f"Not enough valid samples ({len(common_idx)}) for regime optimization; "
                "falling back to score-only quantile regimes"
            )
            tprint_info(
                "HMM alpha regime assignment diagnostics: "
                f"valid_scores={len(valid_scores)}, "
                f"valid_forward_returns={len(forward_returns)}, "
                f"intersection={len(common_idx)}, "
                f"min_required=20"
            )
            aligned_scores = valid_scores
            aligned_returns = forward_returns.reindex(valid_scores.index)
            use_flexible_regimes = False
        else:
            aligned_scores = valid_scores.loc[common_idx]
            aligned_returns = forward_returns.loc[common_idx]

        # Multi-regime competition: test multiple regime counts
        if use_flexible_regimes:
            tprint_info(
                f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Testing flexible regime optimization for {regime_counts_to_test} regime counts "
                f"(min_bin={min_bin_pct*100:.0f}%, max_bin={max_bin_pct*100:.0f}%)"
            )

            best_n_regimes = None
            best_score = -np.inf
            best_labels = None
            best_metrics = None
            competition_results = []

            for n_regimes in regime_counts_to_test:
                if len(aligned_scores) < n_regimes * int(len(aligned_scores) * min_bin_pct):
                    tprint_warning(
                        f"Skipping {n_regimes} regimes: insufficient samples for bin constraints"
                    )
                    continue

                try:
                    labels, score, metrics = self._optimize_regime_boundaries(
                        alpha_scores=aligned_scores,
                        forward_returns=aligned_returns,
                        n_regimes=n_regimes,
                        min_bin_pct=min_bin_pct,
                        max_bin_pct=max_bin_pct,
                        jiggle_pct=float(config.get("alpha_jiggle_pct", 0.01)),
                        max_iterations=int(config.get("alpha_max_opt_iterations", 100)),
                    )

                    competition_results.append({
                        "n_regimes": n_regimes,
                        "wcv_ratio": score,
                        "cv_ratio": metrics.get("cv_ratio", 0.0),
                        "between_cv": metrics.get("between_cv", 0.0),
                        "between_wcv": metrics.get("between_wcv", 0.0),
                        "within_cv": metrics.get("within_cv", 0.0),
                        "within_wcv": metrics.get("within_wcv", 0.0),
                        "iterations": metrics.get("optimization_iterations", 0),
                        "converged": metrics.get("converged", False),
                    })

                    if score > best_score:
                        best_score = score
                        best_n_regimes = n_regimes
                        best_labels = labels
                        best_metrics = metrics

                    tprint_info(
                        f"  ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {n_regimes} regimes: WCV Ratio={score:.4f}, "
                        f"CV Ratio={metrics.get('cv_ratio', 0.0):.4f}, "
                        f"iterations={metrics.get('optimization_iterations', 0)}"
                    )

                except Exception as opt_exc:
                    tprint_warning(f"Optimization failed for {n_regimes} regimes: {opt_exc}")
                    continue

            if best_labels is None:
                tprint_warning("All regime optimization attempts failed; falling back to simple quantiles")
                use_flexible_regimes = False
            else:
                num_bins = best_n_regimes
                bucket_codes = pd.Series(best_labels, index=common_idx)
                tprint_info(
                    f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Selected {num_bins} regimes with WCV Ratio={best_score:.4f} "
                    f"(Between WCV={best_metrics.get('between_wcv', 0.0):.4f}, "
                    f"Within WCV={best_metrics.get('within_wcv', 0.0):.4f})"
                )

        # Fallback to simple quantile binning if flexible optimization is disabled or failed
        if not use_flexible_regimes:
            num_bins = int(config.get("alpha_regime_bins", 5))
            num_bins = max(3, min(num_bins, 6))

            if len(aligned_scores) < num_bins:
                if len(aligned_scores) >= 3:
                    num_bins = len(aligned_scores)
                    tprint_warning(
                        f"Reducing alpha regime bins to {num_bins} due to limited samples"
                    )
                else:
                    tprint_warning(
                        f"Not enough valid alpha scores ({len(aligned_scores)}) to define regimes"
                    )
                    return alpha_df, None, None

            try:
                ranks = aligned_scores.rank(method="first")
                bucket_codes = pd.qcut(
                    ranks,
                    q=num_bins,
                    labels=False,
                    duplicates="drop",
                )

                # If heavy duplication collapses bins, fall back to a binary median split
                effective_bins = int(pd.Series(bucket_codes).nunique())
                if effective_bins < 2:
                    tprint_warning(
                        "Quantile-based alpha regimes collapsed to a single bin; "
                        "falling back to binary median split"
                    )
                    median_rank = ranks.median()
                    bucket_codes = (ranks > median_rank).astype(int)
                    num_bins = 2
                else:
                    num_bins = effective_bins

                tprint_info(
                    f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Using simple quantile binning with {num_bins} effective regimes"
                )
            except ValueError as e:
                tprint_warning(
                    f"Failed to compute quantile-based alpha regimes with qcut: {e}; "
                    "falling back to binary median split"
                )
                ranks = aligned_scores.rank(method="first")
                median_rank = ranks.median()
                bucket_codes = (ranks > median_rank).astype(int)
                num_bins = 2

        bucket_col = f"alpha_regime_bucket_{num_bins}"
        alpha_df[bucket_col] = bucket_codes.reindex(alpha_df.index)

        # Compute comprehensive regime statistics with CV and WCV metrics
        fwd_col = fwd_cols[0]

        stats_records = []
        for bucket in sorted(bucket_codes.unique()):
            mask = alpha_df[bucket_col] == bucket
            group = alpha_df.loc[mask]
            if group.empty:
                continue

            ret = group[fwd_col]
            if ret.isna().all():
                continue

            ret = ret.astype(float)
            mean_ret = float(ret.mean())
            std_ret = float(ret.std()) if len(ret) > 1 else 0.0
            sharpe = mean_ret / (std_ret + 1e-8) if std_ret > 0 else 0.0

            # CV and Winsorized CV for this regime
            cv, wcv = self._calculate_winsorized_cv(ret)

            # Realized volatility (same units as std_ret, kept separate for clarity)
            realized_vol = std_ret

            # Downside volatility: std of negative returns only
            downside = ret[ret < 0.0]
            downside_vol = float(downside.std()) if len(downside) > 1 else 0.0
            downside_sharpe = mean_ret / (downside_vol + 1e-8) if downside_vol > 0 else 0.0

            # Higher-moment and tail metrics
            skewness = float(ret.skew()) if len(ret) > 2 else 0.0
            kurtosis = float(ret.kurtosis()) if len(ret) > 3 else 0.0

            var_level = float(config.get("alpha_var_quantile", 0.05))
            var_level = min(max(var_level, 0.001), 0.2)
            try:
                var_ret = float(ret.quantile(var_level))
            except Exception:
                var_ret = 0.0
            downside_slice = ret[ret <= var_ret]
            cvar_ret = float(downside_slice.mean()) if len(downside_slice) > 0 else var_ret

            # Tail ratio: right tail vs left tail magnitude
            try:
                q_low = float(ret.quantile(var_level))
                q_high = float(ret.quantile(1.0 - var_level))
                right_tail = ret[ret >= q_high]
                left_tail = ret[ret <= q_low]
                right_mean = float(right_tail.mean()) if len(right_tail) > 0 else 0.0
                left_mean = float(left_tail.mean()) if len(left_tail) > 0 else 0.0
                tail_ratio = right_mean / (abs(left_mean) + 1e-8) if right_mean != 0.0 or left_mean != 0.0 else 0.0
            except Exception:
                tail_ratio = 0.0

            # Vol-of-vol: dispersion of rolling volatility within the regime
            vol_of_vol_window = int(config.get("alpha_vol_of_vol_window", 10))
            if vol_of_vol_window >= 2 and len(ret) >= vol_of_vol_window:
                rolling_vol = ret.rolling(vol_of_vol_window).std()
                vol_of_vol = float(rolling_vol.std(skipna=True))
            else:
                vol_of_vol = 0.0

            # Trendiness score: R^2 of linear fit on cumulative returns
            if len(ret) > 2:
                try:
                    t_idx = np.arange(len(ret), dtype=float)
                    cum_ret = ret.cumsum().to_numpy()
                    t_mean = float(t_idx.mean())
                    y_mean = float(cum_ret.mean())
                    cov_ty = float(np.dot(t_idx - t_mean, cum_ret - y_mean))
                    var_t = float(np.dot(t_idx - t_mean, t_idx - t_mean))
                    if var_t > 0.0:
                        slope = cov_ty / var_t
                        intercept = y_mean - slope * t_mean
                        y_hat = intercept + slope * t_idx
                        ss_res = float(np.sum((cum_ret - y_hat) ** 2))
                        ss_tot = float(np.sum((cum_ret - y_mean) ** 2))
                        trendiness = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
                    else:
                        trendiness = 0.0
                except Exception:
                    trendiness = 0.0
            else:
                trendiness = 0.0

            hit_rate = float((ret > 0).mean())
            mean_target = float(group["alpha_target"].mean())

            # Calculate bin percentage
            bin_pct = float(len(group)) / float(len(alpha_df))

            stats_records.append(
                {
                    "alpha_regime_bucket": int(bucket),
                    "n_samples": int(len(group)),
                    "bin_percentage": bin_pct,
                    "mean_forward_return": mean_ret,
                    "std_forward_return": std_ret,
                    "cv_forward_return": cv,
                    "wcv_forward_return": wcv,
                    "sharpe_forward_return": sharpe,
                    "realized_vol_forward_return": realized_vol,
                    "downside_vol_forward_return": downside_vol,
                    "downside_sharpe_forward_return": downside_sharpe,
                    "skewness_forward_return": skewness,
                    "kurtosis_forward_return": kurtosis,
                    "tail_ratio_forward_return": tail_ratio,
                    "var_forward_return": var_ret,
                    "cvar_forward_return": cvar_ret,
                    "vol_of_vol_forward_return": vol_of_vol,
                    "trendiness_forward_return": trendiness,
                    "hit_rate_positive_return": hit_rate,
                    "mean_alpha_target": mean_target,
                }
            )

        if not stats_records:
            tprint_warning("No stats records generated for alpha regimes")
            return alpha_df, None, bucket_col

        regime_stats_df = pd.DataFrame(stats_records).set_index("alpha_regime_bucket").sort_index()

        # Calculate overall economic metrics (Between/Within CV ratios)
        if use_flexible_regimes and best_metrics is not None:
            overall_metrics = {
                "overall_between_cv": best_metrics.get("between_cv", 0.0),
                "overall_between_wcv": best_metrics.get("between_wcv", 0.0),
                "overall_within_cv": best_metrics.get("within_cv", 0.0),
                "overall_within_wcv": best_metrics.get("within_wcv", 0.0),
                "overall_cv_ratio": best_metrics.get("cv_ratio", 0.0),
                "overall_wcv_ratio": best_metrics.get("wcv_ratio", 0.0),
                "optimization_iterations": best_metrics.get("optimization_iterations", 0),
                "optimization_converged": best_metrics.get("converged", False),
            }
            # Add as a row in the stats dataframe for easy reporting
            for key, value in overall_metrics.items():
                regime_stats_df[key] = value

        tprint_info(
            f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Computed alpha regime statistics for {len(regime_stats_df)} regimes "
            f"(bins={num_bins})"
        )

        if use_flexible_regimes and best_metrics is not None:
            tprint_info(
                f"ĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Overall Economic Metrics: "
                f"CV Ratio={best_metrics.get('cv_ratio', 0.0):.4f}, "
                f"WCV Ratio={best_metrics.get('wcv_ratio', 0.0):.4f}, "
                f"Between WCV={best_metrics.get('between_wcv', 0.0):.4f}, "
                f"Within WCV={best_metrics.get('within_wcv', 0.0):.4f}"
            )

        return alpha_df, regime_stats_df, bucket_col

    def _extract_and_save_regime_thresholds(
        self,
        *,
        alpha_scores: pd.Series,
        regime_labels: pd.Series,
        regime_col_name: str,
        symbol: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract regime bin thresholds from assigned regimes for live/production use.

        This is a CRITICAL production artifact: it captures the score boundaries that define
        each regime. Without these thresholds, you cannot consistently assign new predictions
        to regime buckets in live trading.

        Args:
            alpha_scores: Series of alpha predictions (continuous)
            regime_labels: Series of assigned regime labels (discrete)
            regime_col_name: Name of the regime column
            symbol: Trading symbol for artifact metadata
            config: Configuration dictionary

        Returns:
            Dictionary with:
            - thresholds: Dict mapping regime ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ {min_score, max_score}
            - thresholds_by_quantile: Percentile-based thresholds
            - regime_counts: How many samples per regime
            - sortable_thresholds: Sorted list for efficient lookup in production
        """
        try:
            threshold_data = {
                "extraction_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "regime_col_name": regime_col_name,
                "total_samples": len(regime_labels),
            }

            # Calculate score thresholds for each regime
            unique_regimes = sorted(regime_labels.unique())
            threshold_data["n_regimes"] = len(unique_regimes)

            # Thresholds indexed by regime label
            regimes_dict = {}
            regime_counts = {}
            sortable_thresholds = []

            for regime in unique_regimes:
                mask = regime_labels == regime
                regime_scores = alpha_scores[mask]

                if len(regime_scores) > 0:
                    min_score = float(regime_scores.min())
                    max_score = float(regime_scores.max())
                    mean_score = float(regime_scores.mean())
                    median_score = float(regime_scores.median())
                    std_score = float(regime_scores.std()) if len(regime_scores) > 1 else 0.0

                    regimes_dict[int(regime)] = {
                        "min_score": min_score,
                        "max_score": max_score,
                        "mean_score": mean_score,
                        "median_score": median_score,
                        "std_score": std_score,
                        "n_samples": int(len(regime_scores)),
                        "sample_percentage": float(len(regime_scores) / len(regime_labels) * 100),
                    }

                    regime_counts[int(regime)] = len(regime_scores)
                    sortable_thresholds.append((min_score, max_score, int(regime)))

            threshold_data["regime_thresholds"] = regimes_dict
            threshold_data["regime_counts"] = regime_counts

            # Sort thresholds for efficient binary search in production
            sortable_thresholds = sorted(sortable_thresholds, key=lambda x: x[0])
            threshold_data["sortable_thresholds"] = [
                {
                    "regime": t[2],
                    "min_score": t[0],
                    "max_score": t[1],
                }
                for t in sortable_thresholds
            ]

            # Calculate percentile-based thresholds (0%, 25%, 50%, 75%, 100%)
            percentiles = [0, 25, 50, 75, 100]
            percentile_thresholds = {}
            for p in percentiles:
                percentile_thresholds[p] = float(np.percentile(alpha_scores, p))
            threshold_data["percentile_thresholds"] = percentile_thresholds

            tprint_info(
                f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Regime thresholds extracted: {len(unique_regimes)} regimes, "
                f"min_score={min(alpha_scores):.6f}, max_score={max(alpha_scores):.6f}"
            )

            return threshold_data

        except Exception as e:
            tprint_warning(f"Regime threshold extraction failed: {e}")
            return {"extraction_error": str(e)}

    def _assign_alpha_regimes_with_thresholds(
        self,
        *,
        alpha_scores: np.ndarray,
        regime_thresholds: Dict[int, Dict[str, float]],
    ) -> np.ndarray:
        """Apply saved regime thresholds to new alpha predictions (for live trading).

        This function replicates the regime assignment logic during backtest/live without
        needing the full training dataset. Given a set of score thresholds, it assigns
        each prediction to the appropriate regime.

        Args:
            alpha_scores: Array of new alpha predictions
            regime_thresholds: Dict from _extract_and_save_regime_thresholds()["regime_thresholds"]

        Returns:
            Array of regime assignments matching the input scores
        """
        assignments = np.full(len(alpha_scores), -1, dtype=int)

        for regime_id, threshold_info in regime_thresholds.items():
            min_score = threshold_info["min_score"]
            max_score = threshold_info["max_score"]

            # Assign to this regime if score falls within bounds
            # Use <= for max to handle boundary scores
            mask = (alpha_scores >= min_score) & (alpha_scores <= max_score)
            assignments[mask] = regime_id

        # For scores outside all ranges (shouldn't happen in production), assign to nearest regime
        unassigned_mask = assignments == -1
        if unassigned_mask.any():
            unassigned_scores = alpha_scores[unassigned_mask]
            for idx in np.where(unassigned_mask)[0]:
                # Assign to regime with closest mean score
                closest_regime = min(
                    regime_thresholds.keys(),
                    key=lambda r: abs(unassigned_scores[idx] - regime_thresholds[r]["mean_score"])
                )
                assignments[idx] = closest_regime

        return assignments

    def _perform_comprehensive_feature_analysis(
        self,
        *,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        X_full: pd.DataFrame,
        y_full: pd.Series,
        feature_names: List[str],
        config: Dict[str, Any],
        is_classification: bool = False,
    ) -> Dict[str, Any]:
        """Perform comprehensive feature analysis including IC, importance metrics, mRMR, and learning curves.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            X_full: Full dataset features for full model predictions
            y_full: Full dataset targets
            feature_names: Feature names
            config: Configuration dictionary
            is_classification: Whether this is a classification task

        Returns:
            Dictionary with comprehensive feature analysis results
        """
        feature_analysis_results = {
            "feature_analysis_enabled": bool(config.get("alpha_enable_comprehensive_feature_analysis", True)),
            "features_analyzed": len(feature_names),
        }

        if not feature_analysis_results["feature_analysis_enabled"]:
            return feature_analysis_results

        try:
            # 1. Information Coefficient (IC) - Correlation between predictions and targets
            try:
                y_pred = model.predict(X_full)

                # Ensure y_pred is 1D for IC calculation
                if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
                    y_pred = y_pred.ravel() if y_pred.shape[1] == 1 else y_pred[:, 1] if is_classification else y_pred[:, 0]

                y_full_vals = y_full.values if isinstance(y_full, pd.Series) else y_full

                # Pearson correlation
                ic_pearson_corr, ic_pearson_pval = pearsonr(y_full_vals, y_pred)
                feature_analysis_results["ic_pearson_correlation"] = float(ic_pearson_corr)
                feature_analysis_results["ic_pearson_pvalue"] = float(ic_pearson_pval)

                # Spearman correlation (rank-based)
                ic_spearman_corr, ic_spearman_pval = spearmanr(y_full_vals, y_pred)
                feature_analysis_results["ic_spearman_correlation"] = float(ic_spearman_corr)
                feature_analysis_results["ic_spearman_pvalue"] = float(ic_spearman_pval)

                # Hit rate for classification
                if is_classification:
                    hits = (np.sign(y_pred) == np.sign(y_full_vals)).mean()
                    feature_analysis_results["ic_hit_rate"] = float(hits)

                tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Information Coefficient (IC) calculated: Pearson r={ic_pearson_corr:.4f}, Spearman r={ic_spearman_corr:.4f}")
            except Exception as ic_err:
                tprint_warning(f"IC calculation failed: {ic_err}")

            # 2. LGBM built-in feature importance (split and gain)
            try:
                if hasattr(model, 'feature_importances_'):
                    # LightGBM split importance (number of times feature is used)
                    split_importance = model.feature_importances_

                    lgbm_importance_data = []
                    for i, name in enumerate(feature_names):
                        lgbm_importance_data.append({
                            "feature_name": name,
                            "split_importance": float(split_importance[i]) if i < len(split_importance) else 0.0,
                        })

                    feature_analysis_results["lgbm_importance"] = lgbm_importance_data
                    feature_analysis_results["lgbm_top_features"] = sorted(
                        lgbm_importance_data,
                        key=lambda x: x["split_importance"],
                        reverse=True
                    )[:10]

                    tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ LGBM importance calculated for {len(feature_names)} features")
            except Exception as lgbm_err:
                tprint_warning(f"LGBM importance calculation failed: {lgbm_err}")

            # 3. Permutation Importance with stability checking
            if PERMUTATION_IMPORTANCE_AVAILABLE and bool(config.get("alpha_enable_permutation_importance", True)):
                try:
                    X_train_arr = X_train.to_numpy(dtype=float, copy=False) if hasattr(X_train, 'to_numpy') else X_train
                    y_train_arr = y_train.to_numpy(dtype=float, copy=False) if hasattr(y_train, 'to_numpy') else y_train

                    perm_config = PermutationConfig(
                        n_repeats=int(config.get("alpha_permutation_n_repeats", 5)),
                        scoring='r2' if not is_classification else 'accuracy',
                        enable_stability_check=True,
                        n_jobs=int(config.get("alpha_n_jobs", -1)),
                    )

                    perm_calc = PermutationImportanceCalculator(perm_config)
                    perm_results = perm_calc.calculate_importance(
                        model, X_train_arr, y_train_arr, feature_names
                    )

                    if perm_results.get("success"):
                        feature_analysis_results["permutation_importance"] = perm_results.get("feature_importance", {})
                        feature_analysis_results["permutation_importance_mean"] = perm_results.get("importance_mean", [])
                        feature_analysis_results["permutation_importance_std"] = perm_results.get("importance_std", [])
                        feature_analysis_results["permutation_importance_execution_time"] = perm_results.get("execution_time", 0.0)

                        # Top features by permutation importance
                        top_features = sorted(
                            [(name, perm_results["feature_importance"].get(name, {}).get("importance_mean", 0.0))
                             for name in feature_names],
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]
                        feature_analysis_results["permutation_top_features"] = [
                            {"feature_name": name, "importance_mean": imp} for name, imp in top_features
                        ]

                        tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Permutation importance calculated in {perm_results.get('execution_time', 0.0):.3f}s")
                except Exception as perm_err:
                    tprint_warning(f"Permutation importance calculation failed: {perm_err}")

            # 4. Improved mRMR for redundancy analysis
            # Disabled by default for production/fast runs; enable explicitly via
            # alpha_enable_mrmr_analysis=True in the config when heavy diagnostics
            # are desired.
            if IMPROVED_MRMR_AVAILABLE and bool(config.get("alpha_enable_mrmr_analysis", False)):
                try:
                    X_full_arr = X_full.to_numpy(dtype=float, copy=False) if hasattr(X_full, 'to_numpy') else X_full
                    y_full_arr = y_full.to_numpy(dtype=float, copy=False) if hasattr(y_full, 'to_numpy') else y_full

                    mrmr_config = {
                        'mi_weight': 0.7,
                        'spearman_weight': 0.3,
                        'target_ratio': float(config.get("alpha_mrmr_target_ratio", 0.7)),
                        'use_mi_proxy': True,
                        'enable_hardware_optimization': True,
                    }

                    mrmr = ImprovedMRMR(mrmr_config)
                    mrmr_results = mrmr.select_features(X_full_arr, y_full_arr, feature_names)

                    feature_analysis_results["mrmr_selected_features"] = mrmr_results.get("selected_features", [])
                    feature_analysis_results["mrmr_n_selected"] = len(mrmr_results.get("selected_features", []))
                    feature_analysis_results["mrmr_relevance_scores"] = mrmr_results.get("relevance_scores", {})
                    feature_analysis_results["mrmr_execution_time"] = mrmr_results.get("execution_time", 0.0)

                    tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ mRMR selected {feature_analysis_results['mrmr_n_selected']} features (ratio: {feature_analysis_results['mrmr_n_selected']/len(feature_names):.2%})")
                except Exception as mrmr_err:
                    tprint_warning(f"mRMR analysis failed: {mrmr_err}")

            # 5. Learning curve analysis for overfitting detection
            if ENHANCED_LEARNING_CURVE_AVAILABLE and X_val is not None and y_val is not None and bool(config.get("alpha_enable_learning_curve_analysis", True)):
                try:
                    X_train_arr = X_train.to_numpy(dtype=float, copy=False) if hasattr(X_train, 'to_numpy') else X_train
                    y_train_arr = y_train.to_numpy(dtype=float, copy=False) if hasattr(y_train, 'to_numpy') else y_train
                    X_val_arr = X_val.to_numpy(dtype=float, copy=False) if hasattr(X_val, 'to_numpy') else X_val
                    y_val_arr = y_val.to_numpy(dtype=float, copy=False) if hasattr(y_val, 'to_numpy') else y_val

                    lc_analyzer = EnhancedLearningCurveAnalyzer(random_state=int(config.get("alpha_random_state", 42)))
                    lc_result = lc_analyzer.analyze_learning_curve(
                        model,
                        X_train_arr,
                        y_train_arr,
                        X_val_arr,
                        y_val_arr,
                        cv_folds=int(config.get("alpha_hpo_cv_folds", 3)),
                        scoring='r2' if not is_classification else 'accuracy',
                    )

                    feature_analysis_results["learning_curve_analysis"] = {
                        "learning_rate": lc_result.learning_rate,
                        "convergence_stability": lc_result.convergence_stability,
                        "overfitting_risk": lc_result.overfitting_risk,
                        "training_efficiency": lc_result.training_efficiency,
                        "max_score_gap": float(lc_result.max_score_gap),
                        "final_score_gap": float(lc_result.final_score_gap),
                        "early_learning_slope": float(lc_result.early_learning_slope),
                        "convergence_stability_score": float(lc_result.convergence_stability_score),
                        "final_train_score": float(lc_result.final_train_score) if lc_result.final_train_score else None,
                        "final_validation_score": float(lc_result.final_validation_score) if lc_result.final_validation_score else None,
                        "anomalies": lc_result.anomalies,
                        "recommendations": lc_result.recommendations,
                    }

                    tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ Learning curve analysis completed: {lc_result.learning_rate} learning rate, {lc_result.overfitting_risk} overfitting risk")
                except Exception as lc_err:
                    tprint_warning(f"Learning curve analysis failed: {lc_err}")

            # 6. SHAP analysis (if available)
            if SHAP_AVAILABLE and bool(config.get("alpha_enable_shap_importance", False)):
                try:
                    X_train_sample = X_train.head(min(100, len(X_train))) if isinstance(X_train, pd.DataFrame) else X_train[:min(100, len(X_train))]
                    X_train_sample_arr = X_train_sample.to_numpy(dtype=float, copy=False) if hasattr(X_train_sample, 'to_numpy') else X_train_sample

                    # Use TreeExplainer for tree-based models
                    if hasattr(model, 'booster'):  # LightGBM
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.KernelExplainer(model.predict, X_train_sample_arr)

                    shap_values = explainer.shap_values(X_train_sample_arr)

                    # Handle multi-output case
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                    # Calculate mean absolute SHAP values
                    shap_importance = np.abs(shap_values).mean(axis=0)

                    shap_data = []
                    for i, name in enumerate(feature_names):
                        if i < len(shap_importance):
                            shap_data.append({
                                "feature_name": name,
                                "shap_importance": float(shap_importance[i]),
                            })

                    feature_analysis_results["shap_importance"] = shap_data
                    feature_analysis_results["shap_top_features"] = sorted(shap_data, key=lambda x: x["shap_importance"], reverse=True)[:10]

                    tprint_info(f"ÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂÄÂĂÂÄÂĂÂĂÂĂÂÄÂĂÂ SHAP importance calculated for {len(feature_names)} features")
                except Exception as shap_err:
                    tprint_warning(f"SHAP analysis failed (SHAP may have issues with this data): {shap_err}")

            # 7. Soft feature-pruning recommendations (non-destructive)
            #
            # Use permutation importance stability and mRMR relevance to
            # recommend a pruned feature set, but do NOT alter the training
            # feature matrix for this run. Downstream consumers can choose to
            # use this recommended set.
            try:
                enable_pruning = bool(config.get("alpha_enable_feature_pruning", True))
                if enable_pruning and feature_names:
                    selected_feature_names = list(feature_names)

                    perm_info = feature_analysis_results.get("permutation_importance", {})
                    mrmr_selected = feature_analysis_results.get("mrmr_selected_features", [])

                    if isinstance(perm_info, dict) and perm_info:
                        # Build importance and stability series
                        imp_values = []
                        stability_flags = []
                        for name in feature_names:
                            info = perm_info.get(name, {})
                            imp_values.append(float(info.get("importance_mean", 0.0)))
                            stability_flags.append(bool(info.get("stability", True)))

                        imp_series = pd.Series(imp_values, index=feature_names)
                        stable_series = pd.Series(stability_flags, index=feature_names)

                        # Focus on strictly positive importance values
                        positive_imp = imp_series[imp_series > 0.0]
                        if not positive_imp.empty:
                            # Gentle floor: near the bottom tail of positive importances
                            q_low = float(config.get("alpha_feature_pruning_importance_q_low", 0.05))
                            q_low = min(max(q_low, 0.0), 0.25)
                            floor_val = positive_imp.quantile(q_low)

                            # Candidate drop: low importance AND unstable across repeats
                            drop_mask = (imp_series <= floor_val) & (~stable_series)

                            # Protect mRMR-selected features from being dropped
                            if isinstance(mrmr_selected, list) and mrmr_selected:
                                protect = imp_series.index.isin(mrmr_selected)
                                drop_mask = drop_mask & (~protect)

                            drop_candidates = [
                                name for name, flag in drop_mask.items() if bool(flag)
                            ]

                            if drop_candidates:
                                # Do not be harsh: keep at least a minimum fraction
                                min_fraction = float(
                                    config.get("alpha_feature_pruning_min_fraction", 0.6)
                                )
                                min_fraction = min(max(min_fraction, 0.2), 0.9)
                                min_keep = int(
                                    max(
                                        len(feature_names) * min_fraction,
                                        float(config.get("alpha_feature_pruning_min_absolute", 5)),
                                    )
                                )

                                max_drop = max(len(feature_names) - min_keep, 0)
                                if max_drop > 0:
                                    # Rank drop candidates by importance (ascending)
                                    drop_candidates_sorted = sorted(
                                        drop_candidates,
                                        key=lambda n: imp_series.get(n, 0.0),
                                    )
                                    final_drop = drop_candidates_sorted[:max_drop]
                                    selected_feature_names = [
                                        n for n in feature_names if n not in final_drop
                                    ]

                    # Record recommendations
                    feature_analysis_results[
                        "alpha_selected_features_for_pruning"
                    ] = selected_feature_names
                    feature_analysis_results[
                        "alpha_n_selected_for_pruning"
                    ] = len(selected_feature_names)
                    feature_analysis_results[
                        "alpha_n_original_features"
                    ] = len(feature_names)
            except Exception as pruning_err:
                tprint_warning(f"Feature pruning recommendation step failed (non-fatal): {pruning_err}")

            feature_analysis_results["feature_analysis_completed"] = True

            return feature_analysis_results

        except Exception as e:
            tprint_warning(f"Comprehensive feature analysis failed: {e}")
            feature_analysis_results["feature_analysis_error"] = str(e)
            return feature_analysis_results

    def _calculate_walk_forward_validation_classification(
        self,
        *,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model: Any,
        config: Dict[str, Any],
        accuracy_score: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Calculate Walk-Forward Validation metrics for classification model.

        Uses rolling windows on validation set to detect concept drift and
        accuracy degradation over time.

        Args:
            X_val: Validation feature matrix (n_samples, n_features)
            y_val: Validation labels (n_samples,)
            model: Trained classifier with predict_proba method
            config: Configuration dictionary
            accuracy_score: Optional sklearn accuracy_score function

        Returns:
            Dictionary with WFV metrics including:
            - wfv_window_size: Size of test window
            - wfv_n_windows: Number of rolling windows tested
            - wfv_avg_val_accuracy: Average accuracy on validation windows
            - wfv_avg_test_accuracy: Average accuracy on test windows
            - wfv_accuracy_degradation: Test accuracy - Validation accuracy
            - wfv_max_degradation: Maximum degradation in any window
            - wfv_stability_score: Consistency metric (higher is better)
            - wfv_window_metrics: Per-window detailed metrics
        """
        if accuracy_score is None or X_val is None or len(X_val) < 20:
            return {}

        try:
            # Configure rolling window sizes
            val_set_size = len(X_val)
            # Use approximately 60% for validation window, 40% for test window
            window_size = max(int(val_set_size * 0.6), 10)
            step_size = max(int(val_set_size * 0.2), 5)

            # Ensure we have at least 2 windows
            n_windows = (val_set_size - window_size) // step_size
            if n_windows < 2:
                tprint_warning(f"Insufficient validation set size ({val_set_size}) for walk-forward validation")
                return {}

            window_metrics_list = []
            val_accuracies = []
            test_accuracies = []

            for window_idx in range(n_windows):
                start_idx = window_idx * step_size
                val_end_idx = start_idx + window_size
                test_start_idx = val_end_idx
                test_end_idx = min(test_start_idx + int(window_size * 0.4), val_set_size)

                # Skip if test window is too small
                if test_end_idx - test_start_idx < 3:
                    continue

                # Split validation window chronologically
                val_split_idx = int(window_size * 0.75)
                X_val_train = X_val[start_idx:start_idx + val_split_idx]
                y_val_train = y_val[start_idx:start_idx + val_split_idx]
                X_val_val = X_val[start_idx + val_split_idx:val_end_idx]
                y_val_val = y_val[start_idx + val_split_idx:val_end_idx]

                # Test window
                X_test = X_val[test_start_idx:test_end_idx]
                y_test = y_val[test_start_idx:test_end_idx]

                # Train model on validation training portion
                try:
                    # Create a fresh model instance for this window
                    window_model = model.__class__(**model.get_params())
                    window_model.fit(X_val_train, y_val_train)

                    # Evaluate on validation validation portion
                    val_pred = window_model.predict(X_val_val)
                    val_acc = float(accuracy_score(y_val_val, val_pred))
                    val_accuracies.append(val_acc)

                    # Evaluate on test portion
                    test_pred = window_model.predict(X_test)
                    test_acc = float(accuracy_score(y_test, test_pred))
                    test_accuracies.append(test_acc)

                    # Calculate degradation for this window
                    degradation = test_acc - val_acc

                    window_metrics_list.append({
                        "window_idx": window_idx,
                        "window_start": start_idx,
                        "window_end": val_end_idx,
                        "test_start": test_start_idx,
                        "test_end": test_end_idx,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                        "accuracy_degradation": degradation,
                        "n_val_samples": len(X_val_val),
                        "n_test_samples": len(X_test),
                    })
                except Exception as window_err:
                    tprint_warning(f"WFV window {window_idx} failed: {window_err}")
                    continue

            if len(val_accuracies) < 1:
                tprint_warning("No valid walk-forward validation windows")
                return {}

            # Calculate aggregate metrics
            val_acc_array = np.array(val_accuracies)
            test_acc_array = np.array(test_accuracies)

            avg_val_acc = float(np.mean(val_acc_array))
            avg_test_acc = float(np.mean(test_acc_array))
            accuracy_degradation = avg_test_acc - avg_val_acc

            # Stability score: 1 - coefficient of variation of accuracies (higher is better)
            if len(val_acc_array) > 1:
                val_cv = float(np.std(val_acc_array) / (np.mean(val_acc_array) + 1e-8))
                test_cv = float(np.std(test_acc_array) / (np.mean(test_acc_array) + 1e-8))
                stability_score = max(0.0, 1.0 - (val_cv + test_cv) / 2.0)
            else:
                stability_score = 1.0

            # Max degradation
            max_degradation = float(np.max(test_acc_array - val_acc_array))
            min_degradation = float(np.min(test_acc_array - val_acc_array))

            # Trend analysis: is degradation getting worse over time?
            degradation_trend = 0.0
            if len(window_metrics_list) > 1:
                degradations = [w["accuracy_degradation"] for w in window_metrics_list]
                # Simple linear fit to detect trend
                x = np.arange(len(degradations))
                if len(x) > 1:
                    z = np.polyfit(x, degradations, 1)
                    degradation_trend = float(z[0])  # Slope

            return {
                "wfv_enabled": True,
                "wfv_window_size": int(window_size),
                "wfv_n_windows": len(window_metrics_list),
                "wfv_avg_val_accuracy": avg_val_acc,
                "wfv_avg_test_accuracy": avg_test_acc,
                "wfv_accuracy_degradation": accuracy_degradation,
                "wfv_max_degradation": max_degradation,
                "wfv_min_degradation": min_degradation,
                "wfv_stability_score": stability_score,
                "wfv_val_accuracy_std": float(np.std(val_acc_array)),
                "wfv_test_accuracy_std": float(np.std(test_acc_array)),
                "wfv_degradation_trend": degradation_trend,
                "wfv_window_metrics": window_metrics_list,
            }

        except Exception as e:
            tprint_warning(f"Walk-Forward Validation calculation failed: {e}")
            return {}

    def _assess_alpha_regime_quality(
        self,
        *,
        alpha_df: pd.DataFrame,
        regime_col: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[Optional[ClusterQualityMetrics], Optional[str]]:
        """Assess quality of alpha regimes using ClusterQualityAssessor.

        This uses the unified assess_quality interface and persists metrics as a
        dedicated artifact. The minimum regime size defaults to 3 but can be
        overridden via config["alpha_min_regime_size"].
        """
        if regime_col is None or regime_col not in alpha_df.columns:
            tprint_warning("No alpha regime column provided; skipping regime quality assessment")
            return None, None

        regime_series = alpha_df[regime_col]
        valid_mask = regime_series.notna()
        if valid_mask.sum() == 0:
            tprint_warning("No valid alpha regime labels for quality assessment")
            return None, None

        regime_labels = np.asarray(regime_series[valid_mask].astype(int), dtype=int)

        numeric_df = alpha_df.select_dtypes(include=[np.number])
        drop_cols = ["alpha_target", regime_col]
        drop_cols.extend([c for c in numeric_df.columns if c.startswith("alpha_forward_return_")])
        feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
        if not feature_cols:
            forward_ret_feature_cols = [
                c for c in numeric_df.columns if c.startswith("alpha_forward_return_")
            ]
            if forward_ret_feature_cols:
                feature_cols = forward_ret_feature_cols
            else:
                tprint_warning("No numeric features available for alpha regime quality assessment")
                return None, None

        feature_data = numeric_df[feature_cols].loc[valid_mask]

        forward_ret_cols = [c for c in alpha_df.columns if c.startswith("alpha_forward_return_")]
        forward_returns = None
        if forward_ret_cols:
            forward_returns = alpha_df[forward_ret_cols[0]].loc[valid_mask]

        timestamps = alpha_df.index[valid_mask]
        min_regime_size = int(config.get("alpha_min_regime_size", 3))
        temporal_mode = str(config.get("alpha_temporal_sensitivity_mode", "regime_persistence_focused"))
        fast_mode = bool(config.get("alpha_quality_fast_mode", False))

        try:
            metrics = self.quality_assessor.assess_quality(
                regime_labels=regime_labels,
                feature_data=feature_data,
                forward_returns=forward_returns,
                timestamps=timestamps,
                min_regime_size=min_regime_size,
                temporal_sensitivity_mode=temporal_mode,
                fast_mode=fast_mode,
                standardize_for_metrics=True,
            )
        except Exception as exc:
            tprint_warning(f"Alpha regime quality assessment failed: {exc}")
            return None, None

        metrics_dict: Dict[str, Any]
        if hasattr(metrics, "to_dict"):
            metrics_dict = metrics.to_dict()  # type: ignore[assignment]
        elif is_dataclass(metrics) and not isinstance(metrics, type):
            metrics_dict = asdict(metrics)
        else:
            metrics_dict = {"metrics": metrics}

        quality_df = pd.DataFrame([metrics_dict])
        try:
            quality_path = self._save_artifact(
                data=quality_df,
                artifact_name="hmm_alpha_regime_quality_1h",
                artifact_type="data",
                metadata={
                    "min_regime_size": min_regime_size,
                },
            )
        except Exception as save_exc:
            tprint_warning(f"Failed to save alpha regime quality artifact: {save_exc}")
            quality_path = None

        return metrics, quality_path

    def _build_fallback_regime_column_for_report(
        self,
        *,
        alpha_df: pd.DataFrame,
        config: Dict[str, Any],
        prefix: str = "alpha_regime_bucket_report_",
    ) -> Optional[str]:
        """Best-effort helper to ensure a regime column exists for quality reporting."""
        try:
            score_series = None
            if "alpha_score_continuous" in alpha_df.columns:
                score_series = alpha_df["alpha_score_continuous"].astype(float)
            elif "alpha_target" in alpha_df.columns:
                score_series = alpha_df["alpha_target"].astype(float)
            else:
                fwd_cols_local = [
                    c for c in alpha_df.columns if c.startswith("alpha_forward_return_")
                ]
                if fwd_cols_local:
                    score_series = alpha_df[fwd_cols_local[0]].astype(float)

            if score_series is None:
                return None

            score_series = score_series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(score_series) < 50:
                return None

            num_bins = int(config.get("alpha_regime_bins", 5))
            num_bins = max(3, min(num_bins, 6))

            ranks = score_series.rank(method="first")
            try:
                bucket_codes = pd.qcut(
                    ranks,
                    q=num_bins,
                    labels=False,
                    duplicates="drop",
                )
            except Exception:
                bucket_codes = pd.cut(
                    ranks,
                    bins=num_bins,
                    labels=False,
                    duplicates="drop",
                )

            if bucket_codes is None or bucket_codes.empty:
                return None

            effective_bins = int(bucket_codes.nunique())
            if effective_bins < 2:
                return None

            bucket_col = f"{prefix}{effective_bins}"
            alpha_df[bucket_col] = bucket_codes.reindex(alpha_df.index)
            return bucket_col
        except Exception as exc:
            tprint_warning(
                f"Failed to build fallback regime column for quality report: {exc}"
            )
            return None

    def _generate_hmm_alpha_quality_report(
        self,
        *,
        alpha_df: pd.DataFrame,
        regime_col: Optional[str],
        symbol: str,
        exchange: str,
        timeframe: str,
        training_metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        """Generate comprehensive quality report for HMM ML alpha regimes.

        Creates CSV and Markdown reports in outcomes/ with:
        - Per-regime metrics: returns on different timeframes, Sharpe, temporal smoothness
        - Per-quantile (based on 0-1 scalar) metrics
        - Global metrics: transition matrix, overall Sharpe, regime duration stats

        Args:
            alpha_df: DataFrame with regime assignments and alpha_score_continuous
            regime_col: Name of the regime column
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe string
            training_metrics: Dict of training metrics from model
            config: Configuration dictionary

        Returns:
            Tuple of (csv_path, md_path) or None if generation fails
        """
        import os
        from datetime import datetime

        if regime_col is None or regime_col not in alpha_df.columns:
            tprint_warning(
                "No regime column for quality report generation; "
                "skipping HMM alpha quality report"
            )
            return None

        try:
            os.makedirs("outcomes", exist_ok=True)
            now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_name = f"hmm_ml_alpha_quality_{symbol}_{timeframe}_{now_str}"
            csv_path = os.path.join("outcomes", base_name + ".csv")
            md_path = os.path.join("outcomes", base_name + ".md")
            
            regime_series = alpha_df[regime_col].dropna().astype(int)
            if regime_series.empty:
                tprint_warning("No valid regime labels for quality report")
                return None
            
            # ================================================================
            # 1. Per-Regime Metrics
            # ================================================================
            regime_metrics_list = []
            fwd_cols = [c for c in alpha_df.columns if c.startswith("alpha_forward_return_")]
            score_col = "alpha_score_continuous" if "alpha_score_continuous" in alpha_df.columns else None
            
            for regime_id in sorted(regime_series.unique()):
                regime_mask = alpha_df[regime_col] == regime_id
                regime_data = alpha_df[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Duration statistics
                run_lengths = []
                in_run = False
                run_start = None
                for i, (idx, val) in enumerate(regime_series.items()):
                    if val == regime_id:
                        if not in_run:
                            in_run = True
                            run_start = i
                    else:
                        if in_run:
                            run_lengths.append(i - run_start)
                            in_run = False
                if in_run:
                    run_lengths.append(len(regime_series) - run_start)
                
                mean_duration = np.mean(run_lengths) if run_lengths else 0
                median_duration = np.median(run_lengths) if run_lengths else 0
                
                # Return metrics for multiple timeframes
                ret_metrics = {}
                for fwd_col in fwd_cols[:3]:  # Limit to first 3 forward return columns
                    horizon = fwd_col.split("_")[-1]
                    ret = regime_data[fwd_col].dropna().astype(float)
                    if len(ret) > 0:
                        mean_ret = float(ret.mean())
                        std_ret = float(ret.std()) if len(ret) > 1 else 0.0
                        sharpe = mean_ret / (std_ret + 1e-8) if std_ret > 0 else 0.0
                        ret_metrics[f"mean_return_{horizon}"] = mean_ret
                        ret_metrics[f"sharpe_{horizon}"] = sharpe
                
                # Score distribution
                score_mean = float(regime_data[score_col].mean()) if score_col else 0.0
                score_std = float(regime_data[score_col].std()) if score_col and len(regime_data) > 1 else 0.0
                
                regime_metrics = {
                    "regime_id": int(regime_id),
                    "n_samples": len(regime_data),
                    "n_runs": len(run_lengths),
                    "mean_duration_bars": mean_duration,
                    "median_duration_bars": median_duration,
                    "mean_duration_hours": mean_duration * 0.25,  # 15m bars
                    "median_duration_hours": median_duration * 0.25,
                    "score_mean": score_mean,
                    "score_std": score_std,
                    **ret_metrics
                }
                regime_metrics_list.append(regime_metrics)
            
            regime_df = pd.DataFrame(regime_metrics_list)
            
            # ================================================================
            # 2. Per-Quantile Metrics (based on 0-1 scalar)
            # ================================================================
            quantile_metrics_list = []
            if score_col and score_col in alpha_df.columns:
                score_series = alpha_df[score_col].dropna()
                if len(score_series) > 0:
                    quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    for i in range(len(quantiles) - 1):
                        q_low = quantiles[i]
                        q_high = quantiles[i + 1]
                        q_mask = (score_series >= q_low) & (score_series < q_high)
                        if i == len(quantiles) - 2:  # Include upper bound for last quantile
                            q_mask = (score_series >= q_low) & (score_series <= q_high)
                        
                        q_data = alpha_df.loc[q_mask.index[q_mask]]
                        if len(q_data) == 0:
                            continue
                        
                        q_metrics = {
                            "quantile_range": f"{q_low:.1f}-{q_high:.1f}",
                            "n_samples": len(q_data),
                        }
                        
                        # Returns by quantile
                        for fwd_col in fwd_cols[:3]:
                            horizon = fwd_col.split("_")[-1]
                            ret = q_data[fwd_col].dropna().astype(float)
                            if len(ret) > 0:
                                q_metrics[f"mean_return_{horizon}"] = float(ret.mean())
                                q_metrics[f"sharpe_{horizon}"] = float(ret.mean() / (ret.std() + 1e-8)) if len(ret) > 1 else 0.0
                        
                        quantile_metrics_list.append(q_metrics)
            
            quantile_df = pd.DataFrame(quantile_metrics_list) if quantile_metrics_list else pd.DataFrame()
            
            # ================================================================
            # 3. Global Metrics
            # ================================================================
            # Transition matrix
            transition_counts = pd.crosstab(
                regime_series.iloc[:-1].values,
                regime_series.iloc[1:].values,
                normalize='index'
            )
            
            # Overall duration stats
            all_run_lengths = []
            current_regime = regime_series.iloc[0]
            run_start = 0
            for i in range(1, len(regime_series)):
                if regime_series.iloc[i] != current_regime:
                    all_run_lengths.append(i - run_start)
                    run_start = i
                    current_regime = regime_series.iloc[i]
            all_run_lengths.append(len(regime_series) - run_start)
            
            global_metrics = {
                "n_regimes": len(regime_series.unique()),
                "total_samples": len(regime_series),
                "mean_run_length_bars": np.mean(all_run_lengths),
                "median_run_length_bars": np.median(all_run_lengths),
                "mean_run_length_hours": np.mean(all_run_lengths) * 0.25,
                "median_run_length_hours": np.median(all_run_lengths) * 0.25,
                "n_regime_changes": len(all_run_lengths) - 1,
            }
            
            # Temporal smoothness (1 - normalized transition frequency)
            regime_change_rate = (len(all_run_lengths) - 1) / len(regime_series) if len(regime_series) > 0 else 0
            global_metrics["temporal_smoothness"] = 1.0 - regime_change_rate
            
            # ================================================================
            # 4. Save CSV
            # ================================================================
            with open(csv_path, 'w') as f:
                f.write("# HMM ML Alpha Quality Report\n")
                f.write(f"# Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}\n")
                f.write(f"# Generated: {now_str}\n\n")
                
                f.write("## Global Metrics\n")
                for key, value in global_metrics.items():
                    f.write(f"{key},{value}\n")
                f.write("\n")
                
                f.write("## Per-Regime Metrics\n")
                regime_df.to_csv(f, index=False)
                f.write("\n")
                
                if not quantile_df.empty:
                    f.write("## Per-Quantile Metrics\n")
                    quantile_df.to_csv(f, index=False)
                    f.write("\n")
                
                f.write("## Transition Matrix\n")
                transition_counts.to_csv(f)
            
            # ================================================================
            # 5. Save Markdown Report
            # ================================================================
            with open(md_path, 'w') as f:
                f.write("# HMM ML Alpha Quality Report\n\n")
                f.write(f"**Symbol**: {symbol} | **Exchange**: {exchange} | **Timeframe**: {timeframe}\n\n")
                f.write(f"**Generated**: {now_str}\n\n")
                f.write("---\n\n")
                
                # Global metrics
                f.write("## Global Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in global_metrics.items():
                    if isinstance(value, float):
                        f.write(f"| {key} | {value:.4f} |\n")
                    else:
                        f.write(f"| {key} | {value} |\n")
                f.write("\n")
                
                # Regime duration interpretation
                f.write("### Regime Duration Analysis\n\n")
                mean_hours = global_metrics['mean_run_length_hours']
                median_hours = global_metrics['median_run_length_hours']
                f.write(f"- **Mean run length**: {global_metrics['mean_run_length_bars']:.1f} bars â‰ˆ {mean_hours:.2f}h\n")
                f.write(f"- **Median run length**: {global_metrics['median_run_length_bars']:.1f} bars â‰ˆ {median_hours:.2f}h\n")
                
                target_min, target_max = 4, 8  # Target for hmm_ml_alpha_step
                if target_min <= mean_hours <= target_max:
                    f.write(f"- â… **Target achieved**: {target_min}-{target_max}h regime duration\n")
                else:
                    f.write(f"- â ï¸ **Off target**: Current {mean_hours:.1f}h vs target {target_min}-{target_max}h\n")
                f.write("\n")
                
                # Per-regime table
                f.write("## Per-Regime Metrics\n\n")
                if not regime_df.empty:
                    try:
                        f.write(regime_df.to_markdown(index=False))
                    except ImportError:
                        tprint_warning(
                            "tabulate not available; falling back to plain-text regime metrics table"
                        )
                        f.write(regime_df.to_string(index=False))
                    f.write("\n\n")
                
                # Per-quantile table
                if not quantile_df.empty:
                    f.write("## Per-Quantile Metrics (0-1 Scalar)\n\n")
                    try:
                        f.write(quantile_df.to_markdown(index=False))
                    except ImportError:
                        tprint_warning(
                            "tabulate not available; falling back to plain-text quantile metrics table"
                        )
                        f.write(quantile_df.to_string(index=False))
                    f.write("\n\n")
                
                # Transition matrix
                f.write("## Transition Matrix\n\n")
                f.write("Probability of transitioning from row regime to column regime:\n\n")
                try:
                    f.write(transition_counts.to_markdown())
                except ImportError:
                    tprint_warning(
                        "tabulate not available; falling back to plain-text transition matrix"
                    )
                    f.write(transition_counts.to_string())
                f.write("\n\n")
                
                # Training metrics summary
                if training_metrics:
                    f.write("## Training Metrics Summary\n\n")
                    for key, value in training_metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"- **{key}**: {value}\n")
                        elif isinstance(value, str) and key.endswith("_error"):
                            f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            tprint_info(f"âœ“ Generated HMM ML alpha quality report: {md_path}")
            return csv_path, md_path
            
        except Exception as e:
            tprint_warning(f"Failed to generate HMM ML alpha quality report: {e}")
            import traceback
            traceback.print_exc()
            return None
