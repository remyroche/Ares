"""
ML Risk Regime Step

This step consumes 1h OHLCV data to construct risk-based regime labels using
forward volatility and tail risk metrics with GMM + Simulated Annealing optimization.

Primary Goal: Distinguish between turbulent, calm, crash-prone, volatile but trending,
and recovering markets using unsupervised learning on risk features.

Responsibilities:
- Load 1h OHLCV market data.
- Align all series on a common DatetimeIndex.
- Generate comprehensive risk features (volatility, tail risk, acceleration).
- Create optimal regime labels using GMM + Simulated Annealing (100% risk CV optimization).
- Train XGBoost multi-class classifier on optimized labels.
- Apply asymmetric hysteresis (instant danger detection, delayed safety confirmation).
- Save risk regime outputs to versioned_artifacts for downstream consumption.
"""

import logging
import time
import json
import re
from typing import Any, Dict, Optional, Tuple, List, Union
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from sklearn.preprocessing import MinMaxScaler

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.features_common.transforms.scaling_normalization import ScalingNormalizer
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
)
from src.utils.ml_common.optimization import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    create_param_group,
    default_objective_function,
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


class MLRiskRegimeStep(BaseStep):
    """Pipeline step to construct risk-based regime labels from 1h Rolling HMM regimes."""

    def __init__(self, step_name: str = "ml_risk_regime_step"):
        """Initialize the ML Risk Regime step with versioned artifacts enabled."""
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLRiskRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data_15m = None
        self._cached_market_source_15m = None
        self._cached_market_cache_key_15m = None
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

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
            # Smoothing defaults are chosen to target ~3–5 bar regime persistence
            # while allowing overrides via config/CLI when needed.
            alpha_defaults: Dict[str, Any] = {
                "alpha_target_smoothing_method": "ewm",
                "alpha_target_smoothing_window": 5,
                "alpha_score_smoothing_method": "ewm",
                "alpha_score_smoothing_window": 8,
                # Keep HPO opt-in; these defaults are used when explicitly enabled
                "alpha_enable_hpo": False,
                "alpha_hpo_cv_folds": 3,
                "alpha_hpo_final_trials": 20,
                "alpha_enable_regression_calibration": True,
            }
            for k, v in alpha_defaults.items():
                config.setdefault(k, v)

            risk_config_defaults: Dict[str, Any] = {
                "risk_hpo_balance_strength": 26.0,
                "risk_hpo_min_regime_pct": 0.05,
                "risk_hpo_max_regime_pct": 0.40,
                "risk_hpo_wcov_ratio_cap": 10.0,
                "risk_hpo_within_floor": 0.3,
                "risk_hpo_within_spread_penalty_weight": 0.2,
                "risk_n_regimes": 4,
                "risk_iterative_prune_weight_quadrant": 0.55,
                "risk_iterative_prune_weight_balance": 0.45,
                "risk_hpo_enable_balanced_sampling": True,
                "risk_hpo_balanced_max_per_regime": 6000,
                # Optional per-regime caps for the final classifier training
                # dataset. Disabled by default (max_per_regime = 0) so current
                # behavior is preserved unless explicitly enabled.
                "risk_train_enable_balanced_sampling": True,
                "risk_train_balanced_max_per_regime": 3500,
                "risk_train_balanced_random_state": 45,
                "risk_enable_temperature_scaling": True,
                # Default SA configuration: enable subsampling with a 10k cap
                # so full-mode runs do not process the entire 3-year window in
                # every SA iteration unless explicitly overridden.
                "risk_sa_enable_subsampling": True,
                "risk_sa_max_samples": 10000,
                # Expose GMM tolerance so we can approximate early stopping via
                # a slightly looser convergence threshold.
                "risk_gmm_tol": 5e-3,
                # GMM runtime controls: default initialization/iterations and
                # optional subsampling of the feature matrix used for GMM fit.
                "risk_gmm_n_init": 10,
                "risk_gmm_max_iter": 150,
                "risk_gmm_enable_subsampling": True,
                "risk_gmm_max_samples": 12000,
                "risk_gmm_subsample_random_state": 43,
                "risk_gmm_max_features": 10,
                "risk_gmm_corr_threshold": 0.97,
                # HPO subsampling defaults: evaluate candidate configs on a
                # representative subset of samples to control runtime.
                "risk_hpo_enable_subsampling": True,
                "risk_hpo_max_samples": 8000,
                # Diagnostics configuration for noise/WCoV analysis.
                "risk_noise_max_diag_features": 200,
                "risk_noise_diag_sampling_mode": "top_importance",
                "risk_feature_wcov_max_features": 256,
                "risk_noise_diag_enable_subsampling": True,
                "risk_noise_diag_subsample_frac": 0.66,
                "risk_low_importance_quantile": 0.05,
                "risk_noise_harm_score_min": 0.02,
                # EWMA correlation pruning threshold (applied only to smoothed
                # features whose names contain 'ewma').
                "risk_ewma_corr_threshold": 0.97,
                # Hard cap on the number of features used by the classifier
                # after scaling. A value <= 0 disables this cap.
                "risk_classifier_max_features": 70,
                # Maximum allowed absolute correlation between selected
                # classifier features when applying the top-K variance cap.
                "risk_classifier_corr_threshold": 0.97,
            }
            for k, v in risk_config_defaults.items():
                config.setdefault(k, v)

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # ------------------------------------------------------------------
            # 1) Load HMM artifacts (labels/probabilities/economic features)
            # ------------------------------------------------------------------
            # For ML risk regimes we now operate purely on klines_data (OHLCV)
            # and do not require pre-computed HMM regime labels or economics.
            # Keep placeholders so downstream alignment logic can remain generic.
            labels_df, probs_df, economic_df = None, None, None

            # ------------------------------------------------------------------
            # 2) Load 15m OHLCV market data
            # ------------------------------------------------------------------
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            cache_key_15m = (symbol, exchange, regime_timeframe, exec_mode_cfg)

            market_data = None
            market_source = None

            if (
                getattr(self, "_cached_market_data_15m", None) is not None
                and getattr(self, "_cached_market_cache_key_15m", None) == cache_key_15m
            ):
                try:
                    market_data = self._cached_market_data_15m.copy()
                except Exception:
                    market_data = self._cached_market_data_15m
                market_source = self._cached_market_source_15m
                tprint_info("♻️ Reusing cached 15m market data for ML risk regimes")
            else:
                market_data, market_source = self.load_market_data_or_fail(
                    {
                        **config,
                        "timeframe": regime_timeframe,
                    },
                    pipeline_state={},
                    allow_config_override=True,
                    light_mode_filter=False,  # ✅ FIX #1: Load full data, not limited by execution_mode
                    skip_artifacts=True,
                )
                if isinstance(market_data, pd.DataFrame):
                    self._cached_market_data_15m = market_data.copy()
                else:
                    self._cached_market_data_15m = market_data
                self._cached_market_source_15m = market_source
                self._cached_market_cache_key_15m = cache_key_15m

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
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )

            # ------------------------------------------------------------------
            # 3) Align all inputs on common DatetimeIndex
            # ------------------------------------------------------------------
            # With HMM artifacts disabled, this alignment effectively prepares
            # a clean OHLCV frame from klines_data while remaining compatible
            # with optional future inputs.
            aligned_df = self._align_inputs(
                market_data=market_data,
                labels_df=labels_df,
                probs_df=probs_df,
                economic_df=economic_df,
            )

            if aligned_df.empty:
                raise ValueError("Aligned dataset is empty after after merging inputs")

            # ------------------------------------------------------------------
            # 4) Generate comprehensive risk features with EWMA smoothing
            # ------------------------------------------------------------------
            aligned_df = self._generate_risk_features(aligned_df, config)

            # ------------------------------------------------------------------
            # 5) Prepare risk features for regime classification
            # ------------------------------------------------------------------
            risk_df = aligned_df.copy()

            if risk_df.empty:
                raise ValueError("Risk dataset is empty after feature generation")

            # ------------------------------------------------------------------
            # 6) NEW: XGBoost Multi-Class Regime Classifier (100% Risk-Driven)
            # ------------------------------------------------------------------
            model = None
            regime_probs: Optional[np.ndarray] = None
            regime_labels: Optional[np.ndarray] = None
            training_metrics: Dict[str, Any] = {}
            regime_stats_df: Optional[pd.DataFrame] = None
            model_path: Optional[str] = None
            regime_stats_path: Optional[str] = None
            regime_col_name: Optional[str] = None
            risk_quality_metrics: Optional[ClusterQualityMetrics] = None
            risk_quality_path: Optional[str] = None
            feature_pipeline_artifacts: Optional[Dict[str, Any]] = None
            feature_pipeline_path: Optional[str] = None

            tprint_info("=" * 80)
            tprint_info("🎯 NEW APPROACH: XGBoost Multi-Class Regime Classifier (100% Risk CV)")
            tprint_info("=" * 80)

            try:
                # 6a) Create optimal regime labels (GMM + SA, NO temporal smoothing)
                tprint_info("📊 Step 1/2: Creating optimal regime labels...")
                regime_labels_optimized, label_metrics = self._create_optimal_regime_labels(
                    risk_df=risk_df,
                    config=config
                )

                # Store label quality metrics
                training_metrics['label_quality'] = label_metrics

                # 6b) Train XGBoost multi-class classifier on optimized labels
                tprint_info("🤖 Step 2/2: Training XGBoost regime classifier...")
                model, regime_probs, classifier_metrics = self._train_regime_classifier(
                    risk_df=risk_df,
                    regime_labels=regime_labels_optimized,
                    config=config,
                    label_metrics=label_metrics,
                )

                training_metrics.update(classifier_metrics)

                # 6b.1) Save feature importance data as artifacts
                try:
                    if 'feature_importance_detailed' in classifier_metrics:
                        importance_data = classifier_metrics['feature_importance_detailed']

                        # Save global importance
                        global_importance_df = pd.DataFrame(importance_data['global'])
                        global_importance_path = self._save_artifact(
                            data={'importance': global_importance_df},
                            artifact_name="xgboost_feature_importance_global_15m",
                            artifact_type="model",
                            data_category="analysis",
                            metadata={
                                "symbol": symbol,
                                "exchange": exchange,
                                "timeframe": regime_timeframe,
                                "n_features": len(global_importance_df),
                                "model_type": "xgboost_multiclass"
                            }
                        )
                        tprint_info(f"💾 Saved global feature importance: {global_importance_path}")

                        # Save per-regime importance
                        per_regime_dfs = {}
                        for regime_id, regime_data in importance_data['per_regime'].items():
                            per_regime_dfs[f'regime_{regime_id}'] = pd.DataFrame(regime_data)

                        if per_regime_dfs:
                            per_regime_importance_path = self._save_artifact(
                                data=per_regime_dfs,
                                artifact_name="xgboost_feature_importance_per_regime_15m",
                                artifact_type="model",
                                data_category="analysis",
                                metadata={
                                    "symbol": symbol,
                                    "exchange": exchange,
                                    "timeframe": regime_timeframe,
                                    "n_regimes": len(per_regime_dfs),
                                    "model_type": "xgboost_multiclass"
                                }
                            )
                            tprint_info(f"💾 Saved per-regime feature importance: {per_regime_importance_path}")

                        # Generate comprehensive markdown report in outcomes/
                        tprint_info("📊 Generating comprehensive feature importance report...")

                        # Reconstruct importance_data from detailed dict
                        importance_data_for_report = {
                            'global': pd.DataFrame(importance_data['global']),
                            'per_regime': {
                                regime_id: pd.DataFrame(regime_data)
                                for regime_id, regime_data in importance_data['per_regime'].items()
                            },
                            'n_regimes': len(importance_data['per_regime'])
                        }

                        report_path = self._generate_feature_importance_report(
                            importance_data=importance_data_for_report,
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=regime_timeframe,
                            classifier_metrics=classifier_metrics,
                            label_metrics=label_metrics
                        )
                        tprint_success(f"📄 Feature importance report saved: {report_path}")

                        # Update main diagnostics markdown with feature importance from THIS run
                        self._update_risk_diagnostics_feature_importance(
                            symbol=symbol,
                            exchange=exchange,
                            regime_timeframe=regime_timeframe,
                            classifier_metrics=classifier_metrics,
                        )

                except Exception as save_exc:
                    tprint_warning(f"Failed to save feature importance artifacts/report: {save_exc}")

                # 6c) Hard predictions (argmax of probabilities)
                regime_labels = np.argmax(regime_probs, axis=1)

                # 6d) Add predictions to dataframe
                risk_df['risk_regime'] = regime_labels
                regime_col_name = 'risk_regime'

                # Add probabilities
                n_regimes = regime_probs.shape[1]
                for i in range(n_regimes):
                    risk_df[f'risk_regime_{i}_prob'] = regime_probs[:, i]

                # Also store training labels for comparison/analysis
                risk_df['risk_regime_training_label'] = regime_labels_optimized

                tprint_success(
                    f"=" * 80 + "\n"
                    f"✅ REGIME CLASSIFICATION COMPLETE\n"
                    f"=" * 80 + "\n"
                    f"  Classifier Accuracy: {classifier_metrics['val_accuracy']:.3f}\n"
                    f"  Label Quality Score: {label_metrics.get('quality_score', 0):.3f}\n"
                    f"  Risk CV Ratio: {label_metrics.get('risk_cv_ratio', 0):.3f}\n"
                    f"  Wasserstein Distance: {label_metrics.get('wasserstein_distance', 0):.3f}\n"
                    f"  KL Divergence: {label_metrics.get('kl_divergence', 0):.3f}\n"
                    f"  Regime Distribution: {label_metrics.get('regime_distribution', {})}\n"
                    f"=" * 80
                )

                # Calculate regime statistics using a simple volatility-based risk score proxy.
                # Prefer realized volatility; fall back to absolute returns if needed.
                if "vol_realized_20" in risk_df.columns:
                    risk_scores = risk_df["vol_realized_20"]
                elif "returns_1h" in risk_df.columns:
                    risk_scores = risk_df["returns_1h"].abs()
                else:
                    # As a last resort, use a constant score so statistics still compute.
                    risk_scores = pd.Series(1.0, index=risk_df.index)

                # Align labels and scores on the same index
                regime_labels_series = risk_df[regime_col_name]
                common_index = regime_labels_series.index.intersection(risk_scores.index)

                regime_stats_df = self._calculate_regime_statistics(
                    risk_df.loc[common_index],
                    regime_labels_series.loc[common_index],
                    risk_scores.loc[common_index],
                )

            except Exception as exc:
                tprint_error(f"❌ XGBoost regime classification failed: {exc}")
                import traceback
                traceback.print_exc()
                # Fast-fail: do not fall back to heuristic volatility-based regimes
                # to avoid producing inconsistent artifacts without proper probabilities.
                raise

            # Ensure we have a valid regime column name for downstream quality assessment
            if regime_col_name is None and "risk_regime" in risk_df.columns:
                regime_col_name = "risk_regime"

            # ------------------------------------------------------------------
            # 7) Extract and save regime thresholds for production use
            # ------------------------------------------------------------------
            regime_thresholds: Optional[Dict[str, Any]] = None
            regime_thresholds_path: Optional[str] = None

            if regime_col_name is not None and regime_col_name in risk_df.columns:
                try:
                    # Get regime data for threshold extraction
                    regime_data = risk_df[regime_col_name].dropna()
                    if regime_data.empty:
                        tprint_warning("No valid regime data for threshold extraction")
                    else:
                        # Extract KDE breakpoints and sigma as regime thresholds
                        regime_thresholds = {
                            "extraction_timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "regime_col_name": regime_col_name,
                            "total_samples": len(regime_data),
                            "n_regimes": int(config.get("risk_regime_n_regimes", 5)),
                            "risk_regime_breakpoints": config.get("risk_regime_breakpoints", []),
                            "risk_regime_sigma": config.get("risk_regime_sigma", 0.05),
                            "kde_bandwidth": config.get("risk_kde_bandwidth", 0.05),
                        }

                    if regime_thresholds and "extraction_error" not in regime_thresholds:
                        # Save thresholds as artifact
                        try:
                            # Convert breakpoints to DataFrame format for HDF5 compatibility
                            breakpoints = regime_thresholds.get("risk_regime_breakpoints", [])
                            if breakpoints:
                                breakpoints_df = pd.DataFrame({
                                    "breakpoint_idx": range(len(breakpoints)),
                                    "breakpoint_value": breakpoints,
                                })
                                regime_thresholds_for_save = {
                                    "breakpoints_df": breakpoints_df,
                                    "metadata": {
                                        "extraction_timestamp": regime_thresholds.get("extraction_timestamp"),
                                        "symbol": regime_thresholds.get("symbol"),
                                        "regime_col_name": regime_thresholds.get("regime_col_name"),
                                        "total_samples": regime_thresholds.get("total_samples"),
                                        "n_regimes": regime_thresholds.get("n_regimes"),
                                        "sigma": regime_thresholds.get("risk_regime_sigma"),
                                        "kde_bandwidth": regime_thresholds.get("kde_bandwidth"),
                                    }
                                }
                            else:
                                regime_thresholds_for_save = regime_thresholds

                            regime_thresholds_path = self._save_artifact(
                                data=regime_thresholds_for_save,
                                artifact_name="ml_risk_regime_thresholds_15m",
                                artifact_type="model",
                                data_category="config",
                                metadata={
                                    "symbol": symbol,
                                    "exchange": exchange,
                                    "timeframe": regime_timeframe,
                                    "n_regimes": regime_thresholds.get("n_regimes", 0),
                                },
                            )
                            tprint_info(f"💾 Saved risk regime thresholds artifact: {regime_thresholds_path}")
                        except Exception as thresholds_save_exc:
                            tprint_warning(f"Failed to save risk regime thresholds artifact: {thresholds_save_exc}")

                        # Add thresholds to training metrics for reporting
                        training_metrics["regime_thresholds"] = regime_thresholds
                except Exception as thresholds_exc:
                    tprint_warning(f"Risk regime threshold extraction failed (non-fatal): {thresholds_exc}")

            # ------------------------------------------------------------------
            # 8) Switch context to dedicated risk namespace and assess quality
            # ------------------------------------------------------------------
            # Switch context to a dedicated risk model namespace so we do not
            # pollute the original HMM regime store.
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime_risk",
            )

            # Run risk-specific cluster quality assessment on risk regimes (if any)
            try:
                risk_quality_metrics, risk_quality_path = self._assess_risk_regime_quality(
                    risk_df=risk_df,
                    regime_col=regime_col_name,
                    config=config,
                )
            except Exception as quality_exc:
                tprint_warning(f"Risk regime quality assessment failed: {quality_exc}")

            # Compute feature-level WCoV diagnostics for teacher vs predicted regimes
            try:
                max_wcov_feats = int(config.get("risk_feature_wcov_max_features", 0))
                wcov_top_k: Optional[int]
                wcov_top_k = max_wcov_feats if max_wcov_feats > 0 else None

                wcov_diag_df = self._compute_feature_wcov_diagnostics(
                    df=risk_df,
                    regime_col_pred=regime_col_name,
                    regime_col_teacher="risk_regime_training_label",
                    top_k=wcov_top_k,
                )

                if wcov_diag_df is not None and not wcov_diag_df.empty:
                    ts_diag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    symbol_d = str(config.get("symbol", symbol or ""))
                    exchange_d = str(config.get("exchange", exchange or ""))
                    regime_tf_d = str(
                        config.get("regime_timeframe", config.get("timeframe", "15m"))
                    )

                    diag_path = (
                        f"outcomes/ml_risk_feature_wcov_diagnostics_"
                        f"{symbol_d or 'UNKNOWN'}_{exchange_d or 'UNKNOWN'}_{regime_tf_d}_{ts_diag}.csv"
                    )
                    wcov_diag_df.to_csv(diag_path, index=False)
                    tprint_info(
                        f"💾 Saved feature WCoV diagnostics (teacher vs predicted regimes): {diag_path}"
                    )
            except Exception as diag_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Feature WCoV diagnostics computation failed (non-fatal): {diag_exc}"
                )

            risk_to_save = risk_df.reset_index().rename(columns={risk_df.index.name or "index": "timestamp"})

            tprint_info(
                f"💾 Saving risk training dataset with shape {risk_to_save.shape} "
                f"to versioned HDF5 store"
            )
            training_data_path = self._save_artifact(
                data=risk_to_save,
                artifact_name="ml_risk_training_data_15m",
                artifact_type="data",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )

            # Save risk regime probabilities on the native regime timeframe (expected to be 15m)
            try:
                risk_prob_cols = [
                    col
                    for col in risk_df.columns
                    if col.startswith("risk_regime") or col.startswith("risk_score")
                ]
                if risk_prob_cols and isinstance(risk_df.index, pd.DatetimeIndex):
                    risk_probs = risk_df[risk_prob_cols].copy()
                    risk_probs_save = risk_probs.reset_index().rename(
                        columns={risk_probs.index.name or "index": "timestamp"}
                    )

                    # Canonical native-timeframe probabilities artifact
                    self._save_artifact(
                        data=risk_probs_save,
                        artifact_name=f"ml_risk_regime_probabilities_{regime_timeframe}",
                        artifact_type="data",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "source_regime_timeframe": regime_timeframe,
                        },
                    )

                    # For 15m specifically, also emit a standardized
                    # ml_risk_regime_probs_15m artifact to mirror the
                    # liquidity regime step convention.
                    if regime_timeframe == "15m":
                        self._save_artifact(
                            data=risk_probs_save,
                            artifact_name="ml_risk_regime_probs_15m",
                            artifact_type="data",
                            metadata={
                                "symbol": symbol,
                                "exchange": exchange,
                                "timeframe": regime_timeframe,
                                "source_regime_timeframe": regime_timeframe,
                            },
                        )
            except Exception as probs_exc:
                tprint_warning(
                    f"Failed to save ML risk regimes on timeframe {regime_timeframe}: {probs_exc}"
                )

            # Save trained model if available
            if model is not None:
                try:
                    tprint_info("💾 Saving XGBoost risk model via artifact router")
                    model_path = self._save_artifact(
                        data=model,
                        artifact_name="ml_risk_model_15m",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "model_type": "xgboost",
                        },
                    )
                except Exception as save_model_exc:
                    tprint_warning(f"Failed to save risk model artifact: {save_model_exc}")

            # Persist feature pipeline (feature list + scaler state) for live usage
            if feature_pipeline_artifacts is not None:
                try:
                    feature_pipeline_path = self._save_artifact(
                        data=feature_pipeline_artifacts,
                        artifact_name="ml_risk_feature_pipeline_15m",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "feature_names": feature_pipeline_artifacts.get("feature_names", []),
                        },
                    )
                except Exception as save_fp_exc:
                    tprint_warning(f"Failed to save risk feature pipeline artifact: {save_fp_exc}")

            # Save regime-level risk statistics if available
            if regime_stats_df is not None and not regime_stats_df.empty:
                try:
                    regime_stats_to_save = regime_stats_df.reset_index()
                    regime_stats_path = self._save_artifact(
                        data=regime_stats_to_save,
                        artifact_name="ml_risk_regime_stats_15m",
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

            execution_time = time.time() - start_time
            tprint_info(
                f"✅ {self.step_name} completed in {execution_time:.2f}s "
                f"with {len(risk_df)} samples"
            )

            # ------------------------------------------------------------------
            # 7) Generate alpha-specific reports in outcomes/
            # ------------------------------------------------------------------
            try:
                # Alpha quality markdown + CSV reports via ClusterQualityAssessor.
                # Only run this path when we have a ClusterQualityMetrics instance;
                # RiskClusterQualityMetrics is handled by the dedicated
                # risk_cluster_quality_assessor and does not implement the same
                # attributes (timestamp, quality_score, etc.).
                if isinstance(risk_quality_metrics, ClusterQualityMetrics):
                    try:
                        method_config = {
                            "alpha_config": {
                                "alpha_horizon_bars": config.get("alpha_horizon_bars", 1),
                                "alpha_regime_bins": config.get("alpha_regime_bins", 5),
                                "alpha_target_type": config.get("alpha_target_type", "regression"),
                            }
                        }

                        target_type = str(config.get("alpha_target_type", "regression")).lower()
                        calibration_config: Dict[str, Any] = {}

                        if target_type == "regression":
                            calibration_config = {
                                "target_type": target_type,
                                "regression_calibration_enabled": training_metrics.get("regression_calibration_enabled"),
                                "regression_calibration_used": training_metrics.get("regression_calibration_used"),
                                "regression_calibration_method": training_metrics.get("regression_calibration_method"),
                                "val_rmse_uncalibrated": training_metrics.get("val_rmse_uncalibrated", training_metrics.get("val_rmse")),
                                "val_rmse_calibrated": training_metrics.get("val_rmse_calibrated"),
                            }
                        elif target_type == "classification":
                            calibration_config = {
                                "target_type": target_type,
                                "probability_calibration_enabled": training_metrics.get("probability_calibration_enabled"),
                                "calibration_method": training_metrics.get("calibration_method"),
                                "val_auc_uncalibrated": training_metrics.get("val_auc_uncalibrated", training_metrics.get("val_auc")),
                                "val_auc_calibrated": training_metrics.get("val_auc_calibrated"),
                                "ece_uncalibrated": training_metrics.get("ece_uncalibrated"),
                                "ece_calibrated": training_metrics.get("ece_calibrated"),
                                "ece_improvement": training_metrics.get("ece_improvement"),
                            }

                        if calibration_config:
                            method_config["alpha_calibration"] = calibration_config

                        report_prefix = "hmm_alpha_quality"
                        self.quality_assessor.generate_markdown_report(
                            risk_quality_metrics,
                            symbol=symbol,
                            output_dir="outcomes",
                            method_specific_config=method_config,
                            report_prefix=report_prefix,
                        )

                        self.quality_assessor.generate_comprehensive_csv_report(
                            risk_quality_metrics,
                            all_trials=None,
                            symbol=symbol,
                            output_dir="outcomes",
                            method_specific_config=method_config,
                        )
                    except Exception as report_exc:
                        tprint_warning(f"Alpha quality report generation failed (ignored): {report_exc}")

                # Regime stats CSV for alpha regimes
                if regime_stats_df is not None and not regime_stats_df.empty:
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # XGBoost-derived per-regime statistics for the ML risk
                        # regimes, saved under a clear name to differentiate from
                        # legacy HMM alpha stats.
                        stats_csv_name = f"ml_risk_regime_xgb_stats_{symbol}_{ts}.csv"
                        stats_csv_path = f"outcomes/{stats_csv_name}"
                        regime_stats_df.to_csv(stats_csv_path)
                        tprint_info(
                            f"📊 Saved ML risk XGB regime statistics CSV: {stats_csv_path}"
                        )
                    except Exception as stats_csv_exc:  # pragma: no cover - defensive
                        tprint_warning(
                            f"Failed to save ML risk XGB regime statistics CSV (non-fatal): {stats_csv_exc}"
                        )

                # Walk-Forward Validation window metrics CSV (if available)
                wfv_window_metrics = training_metrics.get("wfv_window_metrics", [])
                if wfv_window_metrics and isinstance(wfv_window_metrics, list) and len(wfv_window_metrics) > 0:
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wfv_csv_name = f"hmm_alpha_wfv_window_metrics_{symbol}_{ts}.csv"
                        wfv_csv_path = f"outcomes/{wfv_csv_name}"
                        wfv_df = pd.DataFrame(wfv_window_metrics)
                        wfv_df.to_csv(wfv_csv_path, index=False)
                        tprint_info(
                            f"📊 Saved Walk-Forward Validation window metrics CSV: {wfv_csv_path}"
                        )
                    except Exception as wfv_csv_exc:
                        tprint_warning(
                            f"Failed to save WFV metrics CSV (ignored): {wfv_csv_exc}"
                        )

                # Regime thresholds CSV (if available) - CRITICAL for production deployment
                regime_thresholds_data = training_metrics.get("regime_thresholds", {})
                if regime_thresholds_data and "regime_thresholds" in regime_thresholds_data:
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        thresholds_csv_name = f"hmm_alpha_regime_thresholds_{symbol}_{ts}.csv"
                        thresholds_csv_path = f"outcomes/{thresholds_csv_name}"

                        # Convert thresholds to DataFrame for easier viewing
                        thresholds_dict = regime_thresholds_data["regime_thresholds"]
                        threshold_rows = []
                        for regime_id, threshold_info in sorted(thresholds_dict.items()):
                            threshold_rows.append({
                                "regime_id": regime_id,
                                "min_score": threshold_info.get("min_score", 0.0),
                                "max_score": threshold_info.get("max_score", 0.0),
                                "mean_score": threshold_info.get("mean_score", 0.0),
                                "median_score": threshold_info.get("median_score", 0.0),
                                "std_score": threshold_info.get("std_score", 0.0),
                                "n_samples": threshold_info.get("n_samples", 0),
                                "sample_percentage": threshold_info.get("sample_percentage", 0.0),
                            })

                        thresholds_df = pd.DataFrame(threshold_rows)
                        thresholds_df.to_csv(thresholds_csv_path, index=False)
                        tprint_info(
                            f"📊 Saved regime thresholds CSV: {thresholds_csv_path}"
                        )

                        # Also export percentile thresholds
                        percentile_data = regime_thresholds_data.get("percentile_thresholds", {})
                        if percentile_data:
                            percentile_csv_name = f"hmm_alpha_percentile_thresholds_{symbol}_{ts}.csv"
                            percentile_csv_path = f"outcomes/{percentile_csv_name}"
                            percentile_rows = [
                                {"percentile": p, "threshold_score": score}
                                for p, score in sorted(percentile_data.items())
                            ]
                            percentile_df = pd.DataFrame(percentile_rows)
                            percentile_df.to_csv(percentile_csv_path, index=False)
                            tprint_info(
                                f"📊 Saved percentile thresholds CSV: {percentile_csv_path}"
                            )
                    except Exception as thresholds_csv_exc:
                        tprint_warning(
                            f"Failed to save regime thresholds CSV (ignored): {thresholds_csv_exc}"
                        )

                # Feature importance analysis CSV (if available)
                try:
                    feature_importance_dfs = []

                    # LGBM importance
                    lgbm_importance = training_metrics.get("lgbm_importance", [])
                    if lgbm_importance:
                        feature_importance_dfs.append(pd.DataFrame(lgbm_importance))

                    # Permutation importance
                    perm_importance = training_metrics.get("permutation_importance", {})
                    if perm_importance:
                        perm_data = []
                        for feat_name, feat_data in perm_importance.items():
                            perm_data.append({
                                "feature_name": feat_name,
                                "permutation_importance_mean": feat_data.get("importance_mean", 0.0),
                                "permutation_importance_std": feat_data.get("importance_std", 0.0),
                                "permutation_stability": feat_data.get("stability", True),
                            })
                        feature_importance_dfs.append(pd.DataFrame(perm_data))

                    # mRMR analysis
                    mrmr_relevance = training_metrics.get("mrmr_relevance_scores", {})
                    if mrmr_relevance and isinstance(mrmr_relevance, dict):
                        mrmr_selected = training_metrics.get("mrmr_selected_features", [])
                        mrmr_data = []
                        for feat_name, relevance_score in mrmr_relevance.items():
                            mrmr_data.append({
                                "feature_name": feat_name,
                                "mrmr_relevance_score": relevance_score if isinstance(relevance_score, (int, float)) else float(relevance_score),
                                "mrmr_selected": feat_name in mrmr_selected,
                            })
                        feature_importance_dfs.append(pd.DataFrame(mrmr_data))

                    # SHAP importance
                    shap_importance = training_metrics.get("shap_importance", [])
                    if shap_importance:
                        feature_importance_dfs.append(pd.DataFrame(shap_importance))

                    # Merge all importance metrics
                    if feature_importance_dfs:
                        # Start with first dataframe
                        merged_df = feature_importance_dfs[0].copy()

                        # Merge remaining dataframes on feature_name
                        for df in feature_importance_dfs[1:]:
                            if "feature_name" in df.columns:
                                merged_df = merged_df.merge(df, on="feature_name", how="outer")

                        # Export combined importance CSV
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        importance_csv_name = f"hmm_alpha_feature_importance_{symbol}_{ts}.csv"
                        importance_csv_path = f"outcomes/{importance_csv_name}"
                        merged_df = merged_df.fillna(0.0)
                        merged_df.to_csv(importance_csv_path, index=False)
                        tprint_info(
                            f"📊 Saved feature importance metrics CSV: {importance_csv_path}"
                        )
                except Exception as importance_csv_exc:
                    tprint_warning(
                        f"Failed to save feature importance CSV (ignored): {importance_csv_exc}"
                    )
            except Exception as report_outer_exc:
                tprint_warning(
                    f"Alpha report generation encountered a non-fatal error: {report_outer_exc}"
                )

            return {
                "success": True,
                "artifacts": {
                    "alpha_training_data": risk_df,
                    "alpha_training_data_path": training_data_path,
                    "alpha_model_path": model_path,
                    "alpha_regime_stats": regime_stats_df,
                    "alpha_regime_stats_path": regime_stats_path,
                    "alpha_regime_thresholds": regime_thresholds,
                    "alpha_regime_thresholds_path": regime_thresholds_path,
                    "alpha_regime_quality_metrics": risk_quality_metrics,
                    "alpha_regime_quality_path": risk_quality_path,
                    "alpha_feature_pipeline": feature_pipeline_artifacts,
                    "alpha_feature_pipeline_path": feature_pipeline_path,
                },
                "metrics": training_metrics,
                "execution_time": execution_time,
            }

        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = f"{self.step_name} failed: {exc}"
            self.logger.error(error_msg, exc_info=True)
            tprint_error(error_msg)
            return {
                "success": False,
                "artifacts": {},
                "metrics": {},
                "error": str(exc),
                "execution_time": execution_time,
            }

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _load_hmm_artifacts(
        self,
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load HMM labels, probabilities, and economic features from HDF5.

        Uses the same context that RollingHMMRegimeDiscoveryStep used when
        saving its artifacts: model='regime' and the specified timeframe.
        """
        # Labels
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
            f"✅ Loaded HMM labels: {labels.shape} "
            f"({labels.index.min()} → {labels.index.max()})"
        )

        # Probabilities (optional)
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
                f"✅ Loaded HMM probabilities: {probs.shape} "
                f"({probs.index.min()} → {probs.index.max()})"
            )
        else:
            tprint_warning("No HMM probabilities found; proceeding without them")
            probs = None

        # Economic features (optional)
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

        if economic is not None and not economic.empty:
            tprint_info(
                f"✅ Loaded economic features: {economic.shape} "
                f"({economic.index.min()} → {economic.index.max()})"
            )
        else:
            tprint_warning("No economic features found; proceeding without them")
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
        """Align market data, labels, probabilities, and economic features."""
        # Ensure all indices are datetime and sorted
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

        # Base OHLCV frame from klines_data
        base_ohlcv = market_data[[col for col in market_data.columns if col.lower() in {"open", "high", "low", "close", "volume"}]].rename(
            columns=lambda c: c.lower()
        )
        frames.append(base_ohlcv)

        # Optional additional frames (labels / probabilities / economic features)
        labels_df_prepared = _prepare(labels_df)
        if labels_df_prepared is not None and not labels_df_prepared.empty:
            frames.append(labels_df_prepared)

        probs_df_prepared = _prepare(probs_df)
        if probs_df_prepared is not None and not probs_df_prepared.empty:
            frames.append(probs_df_prepared)

        economic_df_prepared = _prepare(economic_df)
        if economic_df_prepared is not None and not economic_df_prepared.empty:
            frames.append(economic_df_prepared)

        # Inner join on time to ensure all required information is present
        aligned = frames[0]
        for extra in frames[1:]:
            aligned = aligned.join(extra, how="inner")

        aligned = aligned.dropna(how="all")

        tprint_info(
            f"🔗 Aligned dataset shape: {aligned.shape} "
            f"({aligned.index.min()} → {aligned.index.max()})"
        )

        return aligned

    def _generate_risk_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate comprehensive risk features with EWMA smoothing.

        Features are generated with:
        - Last bar value
        - EWMA with period 2
        - EWMA with period 6

        Special metrics (Fragility, Shock, Desperation, Divergence) are NOT smoothed.
        """
        from src.feature_generation.categories.advanced_statistical import (
            HurstExponentGenerator,
            CVaRGenerator,
            JumpIndicatorsGenerator,
            MaxDrawdownGenerator,
        )
        from src.feature_generation.categories.volatility import (
            VectorBTVolatilityExpansionGenerator,
            VectorBTGarmanKlassVolatilityGenerator,
            VectorBTParkinsonVolatilityGenerator,
            VectorBTRogersSatchellVolatilityGenerator,
            VectorBTYangZhangVolatilityGenerator,
            AdvancedVolatilityFeatures,
        )
        from src.feature_generation.categories.advanced_regime_features import (
            RegimeHurstExponentGenerator,
            RegimeFractalDimensionGenerator,
        )
        from src.feature_generation.categories.microstructure_features import (
            VectorBTOrderFlowVolatilityGenerator,
        )
        from src.feature_generation.categories.returns import (
            ReturnsVolatilityGenerator,
        )
        from src.feature_generation.categories.cross_timeframe import (
            CrossTimeframeVolatilityGenerator,
        )
        from src.feature_generation.categories.regime_features import (
            RegimeVolatilityFeatureGenerator,
            RegimeIntermediateVolatilityFeatureGenerator,
        )

        tprint_info("🎯 Generating comprehensive risk features...")

        result_df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in result_df.columns]
        if missing:
            tprint_warning(f"Missing columns for feature generation: {missing}")
            return result_df

        # Calculate returns for various metrics
        returns = np.log(result_df['close'] / result_df['close'].shift(1))
        result_df['returns_1h'] = returns

        # EWMA periods
        ewma_periods = [8, 24]

        # ============ 1. Core Volatility Features with EWMA ============

        # Simple realized volatility (baseline)
        vol_20 = returns.rolling(window=80).std()
        result_df['vol_realized_20'] = vol_20
        for period in ewma_periods:
            result_df[f'vol_realized_20_ewma{period}'] = vol_20.ewm(span=period, adjust=False).mean()

        # High-Low range volatility (Parkinson-style)
        parkinson_vol = np.sqrt(np.log(result_df['high'] / result_df['low']).rolling(window=80).var() / (4 * np.log(2)))
        result_df['parkinson_vol'] = parkinson_vol
        for period in ewma_periods:
            result_df[f'parkinson_vol_ewma{period}'] = parkinson_vol.ewm(span=period, adjust=False).mean()

        # Garman-Klass volatility (uses OHLC)
        hl = np.log(result_df['high'] / result_df['low'])
        co = np.log(result_df['close'] / result_df['open'])
        gk_vol = np.sqrt(0.5 * hl.rolling(window=80).var() - (2 * np.log(2) - 1) * co.rolling(window=80).var())
        result_df['garman_klass_vol'] = gk_vol
        for period in ewma_periods:
            result_df[f'garman_klass_vol_ewma{period}'] = gk_vol.ewm(span=period, adjust=False).mean()

        # ============ 2. Tail Risk Features ============

        # CVaR (5% and 10% levels)
        for alpha in [0.05, 0.10]:
            window = 80
            cvar_vals = []
            for i in range(len(returns)):
                if i < window:
                    cvar_vals.append(np.nan)
                else:
                    window_returns = returns.iloc[i-window:i].dropna()
                    if len(window_returns) > 0:
                        var_threshold = window_returns.quantile(alpha)
                        tail_losses = window_returns[window_returns <= var_threshold]
                        cvar = abs(tail_losses.mean()) if len(tail_losses) > 0 else abs(var_threshold)
                        cvar_vals.append(cvar)
                    else:
                        cvar_vals.append(np.nan)

            cvar_series = pd.Series(cvar_vals, index=returns.index)
            alpha_pct = int(alpha * 100)
            result_df[f'cvar_{alpha_pct}pct'] = cvar_series
            for period in ewma_periods:
                result_df[f'cvar_{alpha_pct}pct_ewma{period}'] = cvar_series.ewm(span=period, adjust=False).mean()

        # Skewness and Kurtosis (tail shape)
        skew_20 = returns.rolling(window=80).skew()
        kurt_20 = returns.rolling(window=80).kurt()
        result_df['skewness_20'] = skew_20
        result_df['kurtosis_20'] = kurt_20
        for period in ewma_periods:
            result_df[f'skewness_20_ewma{period}'] = skew_20.ewm(span=period, adjust=False).mean()
            result_df[f'kurtosis_20_ewma{period}'] = kurt_20.ewm(span=period, adjust=False).mean()

        # ============ 3. Drawdown/Runup Features ============

        # Max Drawdown (rolling 50-bar window)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(window=200, min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.rolling(window=200).min()
        result_df['max_drawdown_50'] = max_dd
        for period in ewma_periods:
            result_df[f'max_drawdown_50_ewma{period}'] = max_dd.ewm(span=period, adjust=False).mean()

        # Max Run-Up (for short squeeze risk)
        running_min = cumulative.rolling(window=200, min_periods=1).min()
        runup = (cumulative - running_min) / (running_min + 1e-9)
        max_ru = runup.rolling(window=200).max()
        result_df['max_runup_50'] = max_ru
        for period in ewma_periods:
            result_df[f'max_runup_50_ewma{period}'] = max_ru.ewm(span=period, adjust=False).mean()

        # ============ 4. Jump Detection ============

        # Jump indicator (returns exceeding 3 std devs)
        vol_rolling = returns.rolling(window=80).std()
        jump_threshold = 3.0
        jumps = (abs(returns) > jump_threshold * vol_rolling).astype(float)
        result_df['jump_indicator'] = jumps
        jump_freq = jumps.rolling(window=80).mean()
        result_df['jump_frequency_20'] = jump_freq
        for period in ewma_periods:
            result_df[f'jump_frequency_20_ewma{period}'] = jump_freq.ewm(span=period, adjust=False).mean()

        # ============ 5. Volatility Expansion & Acceleration ============

        # Vol expansion (rate of change)
        vol_roc = vol_20.pct_change(periods=20)
        result_df['vol_expansion_5'] = vol_roc
        for period in ewma_periods:
            result_df[f'vol_expansion_5_ewma{period}'] = vol_roc.ewm(span=period, adjust=False).mean()

        # Vol acceleration (already in target, but also as feature)
        vol_accel = (vol_20 - vol_20.shift(24)) / (vol_20.shift(24) + 1e-9)
        result_df['vol_acceleration'] = vol_accel
        for period in ewma_periods:
            result_df[f'vol_acceleration_ewma{period}'] = vol_accel.ewm(span=period, adjust=False).mean()

        # ============ 6. Hurst Exponent (Mean Reversion vs Trending) ============

        # Simplified Hurst calculation (R/S method)
        window = 200
        hurst_vals = []
        for i in range(len(returns)):
            if i < window:
                hurst_vals.append(0.5)
            else:
                window_returns = returns.iloc[i-window:i].dropna().values
                if len(window_returns) > 10:
                    # R/S analysis
                    mean_ret = np.mean(window_returns)
                    deviations = window_returns - mean_ret
                    cumulative_deviations = np.cumsum(deviations)
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    S = np.std(window_returns)
                    if S > 0 and R > 0:
                        hurst = np.log(R/S) / np.log(len(window_returns))
                        hurst_vals.append(np.clip(hurst, 0, 1))
                    else:
                        hurst_vals.append(0.5)
                else:
                    hurst_vals.append(0.5)

        hurst_series = pd.Series(hurst_vals, index=returns.index)
        result_df['hurst_exponent_50'] = hurst_series
        for period in ewma_periods:
            result_df[f'hurst_exponent_50_ewma{period}'] = hurst_series.ewm(span=period, adjust=False).mean()

        # ============ 7. Downside/Upside Deviation ============

        # Downside deviation (risk of negative returns)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_dev = downside_returns.rolling(window=80).std()
        result_df['downside_deviation_20'] = downside_dev
        for period in ewma_periods:
            result_df[f'downside_deviation_20_ewma{period}'] = downside_dev.ewm(span=period, adjust=False).mean()

        # Upside deviation (squeeze risk for shorts)
        upside_returns = returns.copy()
        upside_returns[upside_returns < 0] = 0
        upside_dev = upside_returns.rolling(window=80).std()
        result_df['upside_deviation_20'] = upside_dev
        for period in ewma_periods:
            result_df[f'upside_deviation_20_ewma{period}'] = upside_dev.ewm(span=period, adjust=False).mean()

        # ============ 8. Cross-Timeframe Volatility Ratios ============

        # Vol ratio: short-term vs medium-term
        vol_6 = returns.rolling(window=24).std()
        vol_24 = returns.rolling(window=96).std()
        vol_ratio = vol_6 / (vol_24 + 1e-9)
        result_df['vol_ratio_6_24'] = vol_ratio
        for period in ewma_periods:
            result_df[f'vol_ratio_6_24_ewma{period}'] = vol_ratio.ewm(span=period, adjust=False).mean()

        # ============ 9. Fragility Metrics (NO EWMA) ============

        # Fragility Ratio (Modified Amihud): |Return| / Volume
        fragility_raw = abs(returns) / (result_df['volume'] + 1e-9)
        fragility_ewma24 = fragility_raw.ewm(span=96, adjust=False).mean()
        fragility_ratio = fragility_raw / (fragility_ewma24 + 1e-9)
        result_df['fragility_ratio'] = fragility_ratio  # NO EWMA

        # Shock Ratio: (High - Low) / EWMA_Vol
        vol_ewma6 = vol_20.ewm(span=24, adjust=False).mean()
        shock_ratio = (result_df['high'] - result_df['low']) / (vol_ewma6 + 1e-9)
        result_df['shock_ratio'] = shock_ratio  # NO EWMA

        # Desperation Metric (CLV): (Close - Low) / (High - Low) - 0.5
        hl_range = result_df['high'] - result_df['low']
        clv = ((result_df['close'] - result_df['low']) / (hl_range + 1e-9)) - 0.5
        result_df['desperation_clv'] = clv  # NO EWMA

        # Volume-Vol Divergence (Trap Detector)
        vol_1h = vol_20
        vol_ewma6_for_div = vol_1h.ewm(span=24, adjust=False).mean()
        volume_1h = result_df['volume']
        volume_ewma6 = volume_1h.ewm(span=24, adjust=False).mean()

        vol_component = vol_1h / (vol_ewma6_for_div + 1e-9)
        volume_component = volume_1h / (volume_ewma6 + 1e-9)
        divergence = vol_component / (volume_component + 1e-9)
        result_df['volume_vol_divergence'] = divergence  # NO EWMA

        # ============ 10. ENHANCED: Multi-Scale Volatility Analysis ============
        # Use multiple volatility windows to separate regimes by persistence
        # Windows aligned to trading duration: 1h (immediate), 3h (trade-matched), 6h (structural context)

        vol_1h = returns.rolling(window=4).std()
        vol_3h = returns.rolling(window=12).std()
        vol_6h = returns.rolling(window=24).std()

        result_df['vol_1h'] = vol_1h
        result_df['vol_3h'] = vol_3h
        result_df['vol_6h'] = vol_6h

        # Vol ratios for regime separation
        vol_ratio_1h_3h = vol_1h / (vol_3h + 1e-9)  # Stress detection - immediate vs trade-matched
        vol_ratio_3h_6h = vol_3h / (vol_6h + 1e-9)  # Persistence detection - trade-matched vs structural
        vol_ratio_1h_6h = vol_1h / (vol_6h + 1e-9)  # Overall urgency - immediate vs structural

        result_df['vol_ratio_1h_3h'] = vol_ratio_1h_3h
        result_df['vol_ratio_3h_6h'] = vol_ratio_3h_6h
        result_df['vol_ratio_1h_6h'] = vol_ratio_1h_6h

        # EWMA smoothed ratios for stability
        for period in ewma_periods:
            result_df[f'vol_ratio_1h_3h_ewma{period}'] = vol_ratio_1h_3h.ewm(span=period, adjust=False).mean()
            result_df[f'vol_ratio_3h_6h_ewma{period}'] = vol_ratio_3h_6h.ewm(span=period, adjust=False).mean()
            result_df[f'vol_ratio_1h_6h_ewma{period}'] = vol_ratio_1h_6h.ewm(span=period, adjust=False).mean()

        # ============ 10b. Quadrant-Aligned Vol/Autocorr Features ============

        try:
            # RSI(14) on 1h returns
            rsi_window = 56
            gains = returns.clip(lower=0.0)
            losses = -returns.clip(upper=0.0)
            avg_gain = gains.rolling(rsi_window).mean()
            avg_loss = losses.rolling(rsi_window).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            rsi_14 = 100.0 - (100.0 / (1.0 + rs))
            result_df['rsi_14'] = rsi_14

            # Short-horizon volatility-of-volatility (vol_of_vol) based on
            # immediate volatility. This provides the "instability" axis.
            vol_short = returns.rolling(window=12).std()
            vol_of_vol_6 = vol_short.rolling(window=24).std()
            result_df['vol_of_vol_6'] = vol_of_vol_6

            # Short-horizon autocorrelation (lag-1) on returns to capture
            # mean-reversion vs trending behaviour.
            def _rolling_autocorr(x: pd.Series) -> float:
                if len(x) < 3:
                    return np.nan
                try:
                    return float(x.autocorr(lag=1))
                except Exception:
                    return np.nan

            autocorr_3 = returns.rolling(window=12).apply(
                lambda arr: _rolling_autocorr(pd.Series(arr)), raw=True
            )
            result_df['autocorrelation_3'] = autocorr_3
        except Exception as quad_feat_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Quadrant-aligned RSI/vol_of_vol/autocorr feature generation failed (non-fatal): {quad_feat_exc}"
            )

        # ============ 11. ENHANCED: Price Action Features ============

        # True Range (ATR building block)
        tr1 = result_df['high'] - result_df['low']
        tr2 = abs(result_df['high'] - result_df['close'].shift(1))
        tr3 = abs(result_df['low'] - result_df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result_df['true_range'] = true_range
        atr_14 = true_range.rolling(window=56).mean()
        result_df['atr_14'] = atr_14
        result_df['tr_atr_ratio'] = true_range / (atr_14 + 1e-9)  # Spikes = regime change

        for period in ewma_periods:
            result_df[f'tr_atr_ratio_ewma{period}'] = result_df['tr_atr_ratio'].ewm(span=period, adjust=False).mean()

        # Open-Close Gap (gapping risk)
        oc_gap = abs(result_df['close'] - result_df['open'])
        result_df['open_close_gap'] = oc_gap
        result_df['gap_atr_ratio'] = oc_gap / (atr_14 + 1e-9)  # Large gaps = stress

        for period in ewma_periods:
            result_df[f'gap_atr_ratio_ewma{period}'] = result_df['gap_atr_ratio'].ewm(span=period, adjust=False).mean()

        # Intrabar reversal patterns (wick/body ratios)
        body = abs(result_df['close'] - result_df['open'])
        upper_wick = result_df['high'] - np.maximum(result_df['close'], result_df['open'])
        lower_wick = np.minimum(result_df['close'], result_df['open']) - result_df['low']

        result_df['upper_wick_ratio'] = upper_wick / (body + 1e-9)  # High = rejection at top
        result_df['lower_wick_ratio'] = lower_wick / (body + 1e-9)  # High = rejection at bottom
        result_df['total_wick_ratio'] = (upper_wick + lower_wick) / (body + 1e-9)  # High = indecision

        for period in ewma_periods:
            result_df[f'upper_wick_ratio_ewma{period}'] = result_df['upper_wick_ratio'].ewm(span=period, adjust=False).mean()
            result_df[f'total_wick_ratio_ewma{period}'] = result_df['total_wick_ratio'].ewm(span=period, adjust=False).mean()

        # ============ 12. ENHANCED: Momentum Features ============

        # Rate of price change (not just volatility)
        price_change_1 = result_df['close'].diff(4)
        price_change_6 = result_df['close'].diff(24)
        result_df['price_change_1h'] = price_change_1
        result_df['price_change_6h'] = price_change_6

        # Momentum (rate of rate)
        momentum_6 = price_change_1.rolling(window=6).mean()
        momentum_24 = price_change_1.rolling(window=24).mean()
        result_df['momentum_6h'] = momentum_6
        result_df['momentum_24h'] = momentum_24

        # Momentum drift (persistence)
        momentum_drift = (momentum_6 - momentum_24) / (abs(momentum_24) + 1e-9)
        result_df['momentum_drift'] = momentum_drift  # High = accelerating

        for period in ewma_periods:
            result_df[f'momentum_drift_ewma{period}'] = momentum_drift.ewm(span=period, adjust=False).mean()

        # Acceleration (change in momentum)
        accel_1h_6h = (momentum_6 - momentum_6.shift(1)) / (abs(momentum_6) + 1e-9)
        result_df['acceleration_1h_6h'] = accel_1h_6h

        for period in ewma_periods:
            result_df[f'acceleration_1h_6h_ewma{period}'] = accel_1h_6h.ewm(span=period, adjust=False).mean()

        # ============ 13. ENHANCED: Volume Features ============

        if 'volume' in result_df.columns:
            volume = result_df['volume'].astype(float)
            volume_ma_20 = volume.rolling(window=80).mean()
            volume_ma_6 = volume.rolling(window=24).mean()

            # Volume spikes (sudden increases)
            volume_spike = volume / (volume_ma_20 + 1e-9)
            result_df['volume_spike_ratio'] = volume_spike

            for period in ewma_periods:
                result_df[f'volume_spike_ratio_ewma{period}'] = volume_spike.ewm(span=period, adjust=False).mean()

            # Volume drying up (sudden decreases)
            volume_dry = volume_ma_20 / (volume + 1e-9)
            result_df['volume_dry_ratio'] = volume_dry

            for period in ewma_periods:
                result_df[f'volume_dry_ratio_ewma{period}'] = volume_dry.ewm(span=period, adjust=False).mean()

            # Volume-Price trend (divergence detection)
            price_trend_6h = (result_df['close'] - result_df['close'].shift(6)) / (result_df['close'].shift(6) + 1e-9)
            volume_trend_6h = (volume_ma_6 - volume_ma_6.shift(6)) / (volume_ma_6.shift(6) + 1e-9)
            volume_price_divergence = price_trend_6h - volume_trend_6h
            result_df['volume_price_divergence'] = volume_price_divergence  # High = potential reversal

            try:
                if 'cvar_5pct' in result_df.columns:
                    tail_liquidity_cvar5_dry = result_df['cvar_5pct'] * volume_dry
                    result_df['tail_liquidity_cvar5_dry'] = tail_liquidity_cvar5_dry
                    tail_volume_cvar5_spike = result_df['cvar_5pct'] * volume_spike
                    result_df['tail_volume_cvar5_spike'] = tail_volume_cvar5_spike
                if 'cvar_10pct' in result_df.columns:
                    tail_liquidity_cvar10_dry = result_df['cvar_10pct'] * volume_dry
                    result_df['tail_liquidity_cvar10_dry'] = tail_liquidity_cvar10_dry
            except Exception as joint_tail_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Joint tail-liquidity feature generation failed (non-fatal): {joint_tail_exc}"
                )

        try:
            underwater = (drawdown < 0).astype(int)
            dd_streak = np.zeros(len(underwater), dtype=float)
            current_underwater = 0.0
            for i, flag in enumerate(underwater.values):
                if flag:
                    current_underwater += 1.0
                else:
                    current_underwater = 0.0
                dd_streak[i] = current_underwater
            result_df['drawdown_duration_bars'] = dd_streak

            underwater_float = underwater.astype(float)
            drawdown_underwater_frac_20 = underwater_float.rolling(window=80, min_periods=1).mean()
            result_df['drawdown_underwater_frac_20'] = drawdown_underwater_frac_20

            dd_recovery_ratio_50 = max_ru / (max_dd.abs() + 1e-9)
            result_df['dd_recovery_ratio_50'] = dd_recovery_ratio_50
        except Exception as dd_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Drawdown path / recovery feature generation failed (non-fatal): {dd_exc}"
            )

        try:
            vol_for_tail = vol_20
            tail_event = ((returns < -2.0 * (vol_for_tail + 1e-9)) & vol_for_tail.notna()).astype(int)
            tail_since = np.full(len(tail_event), np.nan, dtype=float)
            last_idx_tail = -1
            for i, flag in enumerate(tail_event.values):
                if flag:
                    last_idx_tail = i
                    tail_since[i] = 0.0
                elif last_idx_tail >= 0:
                    tail_since[i] = float(i - last_idx_tail)
            result_df['bars_since_tail_event'] = tail_since

            vol_spike_event = (vol_20 > 1.5 * vol_ewma6).astype(int)
            spike_since = np.full(len(vol_spike_event), np.nan, dtype=float)
            last_idx_spike = -1
            for i, flag in enumerate(vol_spike_event.values):
                if flag:
                    last_idx_spike = i
                    spike_since[i] = 0.0
                elif last_idx_spike >= 0:
                    spike_since[i] = float(i - last_idx_spike)
            result_df['bars_since_vol_spike'] = spike_since
        except Exception as pers_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Risk persistence feature generation failed (non-fatal): {pers_exc}"
            )

        try:
            return_3h = returns.rolling(window=12).sum()
            result_df['return_3h'] = return_3h

            sharpe_like_3h = return_3h / (downside_dev + 1e-9)
            result_df['sharpe_like_3h'] = sharpe_like_3h
            for period in ewma_periods:
                result_df[f'sharpe_like_3h_ewma{period}'] = sharpe_like_3h.ewm(span=period, adjust=False).mean()

            if 'cvar_5pct' in result_df.columns:
                reward_to_cvar_5pct = return_3h / (result_df['cvar_5pct'] + 1e-9)
                result_df['reward_to_cvar_5pct'] = reward_to_cvar_5pct
                for period in ewma_periods:
                    result_df[f'reward_to_cvar_5pct_ewma{period}'] = reward_to_cvar_5pct.ewm(span=period, adjust=False).mean()

            reward_to_dd_50 = return_3h / (max_dd.abs() + 1e-9)
            result_df['reward_to_dd_50'] = reward_to_dd_50
            for period in ewma_periods:
                result_df[f'reward_to_dd_50_ewma{period}'] = reward_to_dd_50.ewm(span=period, adjust=False).mean()
        except Exception as rr_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Risk-normalized return / reward-to-risk feature generation failed (non-fatal): {rr_exc}"
            )

        try:
            vol_slope_1h_3h = vol_3h - vol_1h
            vol_slope_3h_6h = vol_6h - vol_3h
            vol_convexity_1h_3h_6h = (vol_6h - vol_3h) - (vol_3h - vol_1h)

            result_df['vol_slope_1h_3h'] = vol_slope_1h_3h
            result_df['vol_slope_3h_6h'] = vol_slope_3h_6h
            result_df['vol_convexity_1h_3h_6h'] = vol_convexity_1h_3h_6h
        except Exception as shape_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Multi-horizon risk shape feature generation failed (non-fatal): {shape_exc}"
            )

        # Optional correlation-based pruning of EWMA/smoothed features to
        # reduce dimensionality before downstream steps. This is applied only
        # to columns whose names contain 'ewma' and leaves raw risk features
        # untouched.
        try:
            ewma_cols = [c for c in result_df.columns if 'ewma' in c.lower()]
            ewma_threshold = float(config.get("risk_ewma_corr_threshold", 0.97))

            if ewma_cols and len(ewma_cols) > 1 and 0.0 < ewma_threshold < 1.0:
                ewma_df = result_df[ewma_cols].select_dtypes(include=[np.number])
                if not ewma_df.empty and ewma_df.shape[1] > 1:
                    ewma_pruned, _ = self._drop_correlated_features(
                        ewma_df,
                        threshold=ewma_threshold,
                    )
                    kept_ewma_cols = set(ewma_pruned.columns.tolist())
                    dropped_ewma = [c for c in ewma_cols if c not in kept_ewma_cols]
                    if dropped_ewma:
                        result_df = result_df.drop(columns=dropped_ewma, errors="ignore")
                        tprint_info(
                            f"📉 EWMA correlation pruning dropped {len(dropped_ewma)} / {len(ewma_cols)} "
                            f"smoothed features (>{ewma_threshold:.2f} corr)."
                        )
        except Exception as ewma_exc:  # pragma: no cover - defensive
            tprint_warning(f"EWMA correlation pruning failed (non-fatal): {ewma_exc}")

        tprint_info(
            f"✅ Generated {len([c for c in result_df.columns if c not in df.columns])} risk features "
            f"({len([c for c in result_df.columns if 'ewma' in c])} with EWMA smoothing)"
        )

        return result_df

    
    # ========================================================================
    # XGBoost Multi-Class Classifier Approach for Regime Detection
    # ========================================================================

    def _drop_correlated_features(
        self,
        features_df: pd.DataFrame,
        threshold: float = 0.95,
        keep_priority_patterns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Drop highly correlated features to reduce dimensionality.

        Args:
            features_df: DataFrame of features
            threshold: Correlation threshold (default 0.95)
            keep_priority_patterns: List of regex patterns for features to keep

        Returns:
            filtered_df: DataFrame with correlated features removed
            dropped_features: List of dropped feature names
        """
        if keep_priority_patterns is None:
            keep_priority_patterns = [
                r'.*_raw_scaled$',  # Keep all _raw_scaled features
                r'^risk_fwd_vol',   # Keep forward vol features
                r'^risk_tail_cvar', # Keep tail risk features
            ]

        corr_matrix = features_df.corr().abs()

        # Upper triangle of correlation matrix
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation > threshold
        to_drop = set()
        for column in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()

            if correlated_features:
                # Among correlated features, keep the one matching priority patterns
                features_to_compare = [column] + correlated_features

                # Check priority
                priority_feature = None
                for feat in features_to_compare:
                    if any(pd.Series([feat]).str.match(pattern).any() for pattern in keep_priority_patterns):
                        priority_feature = feat
                        break

                # Drop all except priority feature
                for feat in features_to_compare:
                    if feat != priority_feature:
                        to_drop.add(feat)

        dropped_features = list(to_drop)
        kept_features = [col for col in features_df.columns if col not in to_drop]

        tprint_info(f"📉 Dropped {len(dropped_features)} correlated features (>{threshold}): {dropped_features[:10]}...")
        tprint_info(f"✅ Kept {len(kept_features)} features")

        return features_df[kept_features], dropped_features

    def _select_discriminative_features(
        self,
        features_df: pd.DataFrame,
        regime_labels: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Select features that maximize between/within regime variance ratio.

        Args:
            features_df: DataFrame of features
            regime_labels: Regime assignments
            top_k: Number of top features to select (None = keep all)

        Returns:
            selected_features: List of selected feature names
        """
        # Calculate variance ratio for each feature
        feature_scores = []

        for col in features_df.columns:
            feature_data = features_df[col].values

            # Skip if too many NaNs
            if np.isnan(feature_data).sum() > len(feature_data) * 0.5:
                continue

            # Between-regime variance
            regime_means = []
            for regime_id in np.unique(regime_labels):
                if regime_id < 0:  # Skip invalid labels
                    continue
                regime_mask = regime_labels == regime_id
                regime_data = feature_data[regime_mask]
                regime_data_clean = regime_data[~np.isnan(regime_data)]
                if len(regime_data_clean) > 0:
                    regime_means.append(np.mean(regime_data_clean))

            if len(regime_means) < 2:
                continue

            between_var = np.var(regime_means)

            # Within-regime variance
            within_vars = []
            for regime_id in np.unique(regime_labels):
                if regime_id < 0:
                    continue
                regime_mask = regime_labels == regime_id
                regime_data = feature_data[regime_mask]
                regime_data_clean = regime_data[~np.isnan(regime_data)]
                if len(regime_data_clean) > 1:
                    within_vars.append(np.var(regime_data_clean))

            if not within_vars:
                continue

            within_var = np.mean(within_vars)

            # Variance ratio
            var_ratio = between_var / (within_var + 1e-8)

            feature_scores.append({
                'feature': col,
                'var_ratio': var_ratio,
                'between_var': between_var,
                'within_var': within_var
            })

        # Sort by variance ratio
        feature_scores_df = pd.DataFrame(feature_scores).sort_values('var_ratio', ascending=False)

        if top_k is not None and top_k < len(feature_scores_df):
            selected_df = feature_scores_df.head(top_k)
            tprint_info(f"🎯 Selected top {top_k} discriminative features (by variance ratio)")
        else:
            selected_df = feature_scores_df
            tprint_info(f"🎯 All {len(feature_scores_df)} features ranked by discriminative power")

        # Log top features
        for idx, row in selected_df.head(10).iterrows():
            tprint_info(f"  {row['feature']}: ratio={row['var_ratio']:.3f}")

        return selected_df['feature'].tolist()

    def _apply_umap_reduction(
        self,
        features_df: pd.DataFrame,
        n_components: int = 8,
        n_neighbors: int = 30,
        min_dist: float = 0.0,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Apply UMAP dimensionality reduction preserving cluster structure.

        Args:
            features_df: Input features
            n_components: Number of UMAP components
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random seed

        Returns:
            reduced_df: DataFrame with UMAP components
            umap_reducer: Fitted UMAP object
        """
        try:
            import umap
        except ImportError:
            tprint_warning("⚠️ UMAP not installed, skipping dimensionality reduction")
            return features_df, None

        # Remove NaNs
        features_clean = features_df.dropna()

        if len(features_clean) < 100:
            tprint_warning("⚠️ Insufficient samples for UMAP, skipping")
            return features_df, None

        tprint_info(f"🔬 Applying UMAP: {features_df.shape[1]} → {n_components} dimensions")

        reducer = umap.UMAP(
            n_components=min(n_components, features_df.shape[1] - 2),
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=random_state,
            verbose=False
        )

        embedding = reducer.fit_transform(features_clean)

        reduced_df = pd.DataFrame(
            embedding,
            index=features_clean.index,
            columns=[f'umap_{i}' for i in range(embedding.shape[1])]
        )

        tprint_success(f"✅ UMAP reduction complete: {features_df.shape[1]} → {reduced_df.shape[1]} features")

        return reduced_df, reducer

    def _calculate_winsorized_cv_between(
        self,
        regime_labels: np.ndarray,
        features: Union[pd.DataFrame, pd.Series],
        lower_pct: float = 0.05,
        upper_pct: float = 0.95
    ) -> float:
        """
        Calculate between-regime CV using WINSORIZED means.
        More robust to outliers than standard CV.

        Args:
            regime_labels: Regime assignments
            features: Feature data
            lower_pct: Lower quantile for winsorization
            upper_pct: Upper quantile for winsorization

        Returns:
            between_cv: Coefficient of variation between regimes
        """
        if isinstance(features, pd.Series):
            features = features.to_frame()

        regime_means = []
        for regime_id in np.unique(regime_labels):
            if regime_id < 0:  # Skip invalid labels
                continue

            regime_mask = regime_labels == regime_id
            regime_data = features[regime_mask]

            # Winsorize each feature
            regime_means_winsorized = []
            for col in features.columns:
                col_data = regime_data[col].dropna()
                if len(col_data) > 0:
                    lower_bound = col_data.quantile(lower_pct)
                    upper_bound = col_data.quantile(upper_pct)
                    col_winsorized = col_data.clip(lower=lower_bound, upper=upper_bound)
                    regime_means_winsorized.append(col_winsorized.mean())

            if regime_means_winsorized:
                regime_means.append(np.mean(regime_means_winsorized))

        # CV of regime means
        if len(regime_means) < 2:
            return 0.0

        regime_means_array = np.array(regime_means)
        cv_between = regime_means_array.std() / (np.abs(regime_means_array.mean()) + 1e-8)

        return float(cv_between)

    def _calculate_winsorized_cv_within(
        self,
        regime_labels: np.ndarray,
        features: Union[pd.DataFrame, pd.Series],
        lower_pct: float = 0.05,
        upper_pct: float = 0.95
    ) -> float:
        """
        Calculate within-regime CV using WINSORIZED standard deviations.
        Sample-size weighted average across regimes.

        Args:
            regime_labels: Regime assignments
            features: Feature data
            lower_pct: Lower quantile for winsorization
            upper_pct: Upper quantile for winsorization

        Returns:
            within_cv: Weighted average within-regime CV
        """
        if isinstance(features, pd.Series):
            features = features.to_frame()

        within_cvs = []
        regime_sizes = []

        for regime_id in np.unique(regime_labels):
            if regime_id < 0:
                continue

            regime_mask = regime_labels == regime_id
            regime_data = features[regime_mask]
            regime_sizes.append(regime_mask.sum())

            # Winsorize each feature
            feature_cvs = []
            for col in features.columns:
                col_data = regime_data[col].dropna()
                if len(col_data) > 1:
                    lower_bound = col_data.quantile(lower_pct)
                    upper_bound = col_data.quantile(upper_pct)
                    col_winsorized = col_data.clip(lower=lower_bound, upper=upper_bound)

                    cv = col_winsorized.std() / (np.abs(col_winsorized.mean()) + 1e-8)
                    feature_cvs.append(cv)

            if feature_cvs:
                within_cvs.append(np.mean(feature_cvs))

        # Sample-size weighted average
        if not within_cvs:
            return 1.0  # Prevent division by zero

        within_cvs_array = np.array(within_cvs, dtype=float)
        regime_sizes_array = np.array(regime_sizes, dtype=float)

        total = float(regime_sizes_array.sum())
        if total <= 0.0 or within_cvs_array.size == 0:
            return 1.0

        # Align lengths defensively in case some regimes produced no CVs.
        if regime_sizes_array.shape[0] != within_cvs_array.shape[0]:
            regime_sizes_array = regime_sizes_array[: within_cvs_array.shape[0]]

        # Weight each regime by the square of its sample proportion (p_k^2),
        # so that very small regimes contribute only weakly to the overall
        # within-regime CV.
        p = regime_sizes_array / total
        weights = p ** 2
        if np.all(weights <= 0.0):
            return 1.0

        weighted_cv = np.average(within_cvs_array, weights=weights)

        return float(weighted_cv)

    def _calculate_wasserstein_distance(
        self,
        regime_labels: np.ndarray,
        features: Union[pd.DataFrame, pd.Series]
    ) -> float:
        """
        Calculate average Wasserstein distance between regime distributions.

        Args:
            regime_labels: Regime assignments
            features: Feature data

        Returns:
            avg_wasserstein: Average Wasserstein distance across all regime pairs
        """
        from scipy.stats import wasserstein_distance

        if isinstance(features, pd.Series):
            features = features.to_frame()

        unique_regimes = [r for r in np.unique(regime_labels) if r >= 0]

        if len(unique_regimes) < 2:
            return 0.0

        wasserstein_distances = []

        # Compare all pairs of regimes
        for i, regime_i in enumerate(unique_regimes):
            for regime_j in unique_regimes[i+1:]:
                mask_i = regime_labels == regime_i
                mask_j = regime_labels == regime_j

                data_i = features[mask_i].values.flatten()
                data_j = features[mask_j].values.flatten()

                # Remove NaNs
                data_i = data_i[~np.isnan(data_i)]
                data_j = data_j[~np.isnan(data_j)]

                if len(data_i) > 0 and len(data_j) > 0:
                    wd = wasserstein_distance(data_i, data_j)
                    wasserstein_distances.append(wd)

        if not wasserstein_distances:
            return 0.0

        return float(np.mean(wasserstein_distances))

    def _calculate_kl_divergence(
        self,
        regime_labels: np.ndarray,
        features: Union[pd.DataFrame, pd.Series],
        n_bins: int = 50
    ) -> float:
        """
        Calculate average KL divergence between regime distributions.

        Args:
            regime_labels: Regime assignments
            features: Feature data
            n_bins: Number of bins for histogram estimation

        Returns:
            avg_kl: Average KL divergence across all regime pairs
        """
        from scipy.stats import entropy

        if isinstance(features, pd.Series):
            features = features.to_frame()

        unique_regimes = [r for r in np.unique(regime_labels) if r >= 0]

        if len(unique_regimes) < 2:
            return 0.0

        kl_divergences = []

        # Get global range for consistent binning
        all_data = features.values.flatten()
        all_data = all_data[~np.isnan(all_data)]
        data_min, data_max = all_data.min(), all_data.max()
        bins = np.linspace(data_min, data_max, n_bins + 1)

        # Compare all pairs of regimes
        for i, regime_i in enumerate(unique_regimes):
            for regime_j in enumerate(unique_regimes[i+1:]):
                mask_i = regime_labels == regime_i
                mask_j = regime_labels == regime_j

                data_i = features[mask_i].values.flatten()
                data_j = features[mask_j].values.flatten()

                # Remove NaNs
                data_i = data_i[~np.isnan(data_i)]
                data_j = data_j[~np.isnan(data_j)]

                if len(data_i) > 10 and len(data_j) > 10:
                    # Create histograms
                    hist_i, _ = np.histogram(data_i, bins=bins, density=True)
                    hist_j, _ = np.histogram(data_j, bins=bins, density=True)

                    # Add small epsilon to avoid log(0)
                    hist_i = hist_i + 1e-10
                    hist_j = hist_j + 1e-10

                    # Normalize
                    hist_i = hist_i / hist_i.sum()
                    hist_j = hist_j / hist_j.sum()

                    # Calculate KL divergence
                    kl = entropy(hist_i, hist_j)
                    if np.isfinite(kl):
                        kl_divergences.append(kl)

        if not kl_divergences:
            return 0.0

        return float(np.mean(kl_divergences))

    def _calculate_regime_quality_score(
        self,
        regime_labels: np.ndarray,
        risk_features: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate regime quality score based on 100% RISK CV ratio.

        NO economic component - focuses entirely on risk feature distinctiveness.

        Args:
            regime_labels: Regime assignments
            risk_features: Risk feature DataFrame
            forward_returns: NOT USED (kept for compatibility)

        Returns:
            quality_score: Pure risk-based quality score
        """
        # Component 1: Risk feature CV ratio (WINSORIZED) - 100% weight
        risk_cv_between = self._calculate_winsorized_cv_between(regime_labels, risk_features)
        risk_cv_within = self._calculate_winsorized_cv_within(regime_labels, risk_features)
        risk_cv_ratio = risk_cv_between / (risk_cv_within + 1e-8)

        # Pure risk score (100% weight)
        score = risk_cv_ratio

        # Penalty for imbalanced regimes (variable-width: 5-45%)
        label_counts = pd.Series(regime_labels).value_counts()
        min_pct = label_counts.min() / len(regime_labels)
        max_pct = label_counts.max() / len(regime_labels)

        target_min = 0.05
        target_max = 0.45

        violation_low = max(0.0, target_min - min_pct)
        violation_high = max(0.0, max_pct - target_max)
        total_violation = violation_low + violation_high

        if total_violation > 0.0:
            balance_penalty = float(np.exp(-8.0 * total_violation))
            score *= balance_penalty  # Heavy penalty for extreme imbalance

        return float(score)

    def _calculate_regime_quality_metrics(
        self,
        regime_labels: np.ndarray,
        risk_features: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive regime quality metrics.

        Args:
            regime_labels: Regime assignments
            risk_features: Risk features
            forward_returns: Optional forward returns

        Returns:
            metrics: Dictionary of quality metrics
        """
        metrics = {}

        # Risk CV ratio (primary metric)
        risk_cv_between = self._calculate_winsorized_cv_between(regime_labels, risk_features)
        risk_cv_within = self._calculate_winsorized_cv_within(regime_labels, risk_features)
        metrics['risk_cv_ratio'] = risk_cv_between / (risk_cv_within + 1e-8)
        metrics['risk_cv_between'] = risk_cv_between
        metrics['risk_cv_within'] = risk_cv_within

        # Economic CV ratio (for validation only)
        if forward_returns is not None:
            econ_cv_between = self._calculate_winsorized_cv_between(regime_labels, forward_returns)
            econ_cv_within = self._calculate_winsorized_cv_within(regime_labels, forward_returns)
            metrics['econ_cv_ratio'] = econ_cv_between / (econ_cv_within + 1e-8)
            metrics['econ_cv_between'] = econ_cv_between
            metrics['econ_cv_within'] = econ_cv_within
        else:
            metrics['econ_cv_ratio'] = 0.0

        # Wasserstein distance
        try:
            metrics['wasserstein_distance'] = self._calculate_wasserstein_distance(regime_labels, risk_features)
        except Exception:
            metrics['wasserstein_distance'] = 0.0

        # KL divergence
        try:
            metrics['kl_divergence'] = self._calculate_kl_divergence(regime_labels, risk_features)
        except Exception:
            metrics['kl_divergence'] = 0.0

        # Regime balance
        label_counts = pd.Series(regime_labels).value_counts()
        metrics['min_regime_pct'] = label_counts.min() / len(regime_labels)
        metrics['max_regime_pct'] = label_counts.max() / len(regime_labels)
        metrics['regime_distribution'] = label_counts.to_dict()

        # Quality score
        metrics['quality_score'] = self._calculate_regime_quality_score(
            regime_labels, risk_features, forward_returns
        )

        return metrics

    def _compute_feature_wcov_diagnostics(
        self,
        df: pd.DataFrame,
        regime_col_pred: str,
        regime_col_teacher: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if regime_col_pred not in df.columns:
            return pd.DataFrame()

        regime_pred = np.asarray(df[regime_col_pred].values)
        teacher_labels: Optional[np.ndarray] = None
        if regime_col_teacher is not None and regime_col_teacher in df.columns:
            try:
                teacher_labels = np.asarray(df[regime_col_teacher].values)
            except Exception:
                teacher_labels = None

        exclude_prefixes = ("risk_target", "risk_regime", "alpha_")
        feature_cols: List[str] = [
            col
            for col in numeric_df.columns
            if not any(col.startswith(p) for p in exclude_prefixes)
        ]

        records: List[Dict[str, Any]] = []
        for col in feature_cols:
            series = numeric_df[col]
            if series.isna().all():
                continue

            try:
                between_pred = self._calculate_winsorized_cv_between(regime_pred, series)
                within_pred = self._calculate_winsorized_cv_within(regime_pred, series)
                ratio_pred = between_pred / (within_pred + 1e-8)
            except Exception:
                between_pred = np.nan
                within_pred = np.nan
                ratio_pred = np.nan

            between_teacher = np.nan
            within_teacher = np.nan
            ratio_teacher = np.nan
            if teacher_labels is not None:
                try:
                    between_teacher = self._calculate_winsorized_cv_between(
                        teacher_labels,
                        series,
                    )
                    within_teacher = self._calculate_winsorized_cv_within(
                        teacher_labels,
                        series,
                    )
                    ratio_teacher = between_teacher / (within_teacher + 1e-8)
                except Exception:
                    between_teacher = np.nan
                    within_teacher = np.nan
                    ratio_teacher = np.nan

            improvement = np.nan
            if np.isfinite(ratio_pred) and np.isfinite(ratio_teacher):
                improvement = ratio_pred - ratio_teacher

            records.append(
                {
                    "feature": col,
                    "wcov_between_pred": float(between_pred) if np.isfinite(between_pred) else np.nan,
                    "wcov_within_pred": float(within_pred) if np.isfinite(within_pred) else np.nan,
                    "wcov_ratio_pred": float(ratio_pred) if np.isfinite(ratio_pred) else np.nan,
                    "wcov_between_teacher": float(between_teacher) if np.isfinite(between_teacher) else np.nan,
                    "wcov_within_teacher": float(within_teacher) if np.isfinite(within_teacher) else np.nan,
                    "wcov_ratio_teacher": float(ratio_teacher) if np.isfinite(ratio_teacher) else np.nan,
                    "wcov_ratio_improvement": float(improvement) if np.isfinite(improvement) else np.nan,
                }
            )

        if not records:
            return pd.DataFrame()

        diag_df = pd.DataFrame(records)
        diag_df = diag_df.sort_values("wcov_ratio_pred", ascending=False)
        if top_k is not None and top_k > 0 and len(diag_df) > top_k:
            diag_df = diag_df.head(top_k)
        return diag_df.reset_index(drop=True)

    # REMOVED: Temporal smoothing methods (per user request)
    # Temporal smoothing is no longer applied to labels before training
    # Labels are used directly from GMM + SA optimization

    def _refine_labels_simulated_annealing(
        self,
        initial_labels: np.ndarray,
        risk_features: pd.DataFrame,
        forward_returns: Optional[pd.Series],
        n_regimes: int,
        max_iterations: int = 500,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.995,
        flip_fraction: float = 0.03,
        neighborhood_mode: str = "any",
        early_stop_patience: Optional[int] = None,
        good_enough_improvement: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Refine regime labels using simulated annealing to maximize risk CV ratio.

        Args:
            initial_labels: Starting regime labels
            risk_features: Risk feature DataFrame
            forward_returns: NOT USED (100% risk optimization)
            n_regimes: Number of regimes
            max_iterations: Maximum SA iterations
            initial_temp: Initial temperature
            cooling_rate: Cooling rate per iteration

        Returns:
            best_labels: Optimized regime labels
            best_score: Best quality score achieved
        """
        current_labels = initial_labels.copy()
        current_score = self._calculate_regime_quality_score(
            current_labels, risk_features, None
        )

        best_labels = current_labels.copy()
        best_score = current_score

        temperature = initial_temp
        accept_count = 0
        plateau_iterations = 0

        tprint_info(f"🔥 Starting Simulated Annealing (100% risk CV): initial_score={current_score:.4f}")

        for iteration in range(max_iterations):
            # Propose modification: flip 1-2% of samples to neighboring regime
            candidate_labels = current_labels.copy()
            n_flips = max(1, int(flip_fraction * len(candidate_labels)))
            flip_indices = np.random.choice(len(candidate_labels), size=n_flips, replace=False)

            for idx in flip_indices:
                current_regime = candidate_labels[idx]
                # Flip to another regime
                if neighborhood_mode == "adjacent":
                    neighbors = []
                    if current_regime > 0:
                        neighbors.append(current_regime - 1)
                    if current_regime < n_regimes - 1:
                        neighbors.append(current_regime + 1)
                else:
                    neighbors = [r for r in range(n_regimes) if r != current_regime]

                if neighbors:
                    candidate_labels[idx] = np.random.choice(neighbors)

            # Evaluate candidate
            candidate_score = self._calculate_regime_quality_score(
                candidate_labels, risk_features, None
            )

            # Accept or reject (Metropolis criterion)
            delta = candidate_score - current_score
            improved = False
            if delta > 0:
                # Always accept improvements
                current_labels = candidate_labels
                current_score = candidate_score
                accept_count += 1

                if current_score > best_score:
                    best_labels = current_labels.copy()
                    best_score = current_score
                    improved = True
            elif np.random.random() < np.exp(delta / temperature):
                # Sometimes accept worse solutions (escape local optima)
                current_labels = candidate_labels
                current_score = candidate_score
                accept_count += 1

            if improved:
                plateau_iterations = 0
            else:
                plateau_iterations += 1

            # Cool down
            temperature *= cooling_rate

            # Progress logging
            if iteration % 50 == 0 or iteration == max_iterations - 1:
                accept_rate = accept_count / (iteration + 1)
                tprint_info(
                    f"  SA iter {iteration}/{max_iterations}: "
                    f"score={current_score:.4f}, best={best_score:.4f}, "
                    f"temp={temperature:.4f}, accept_rate={accept_rate:.2%}"
                )

            # Early stopping: no improvement in best score for N iterations
            if early_stop_patience is not None and early_stop_patience > 0 and plateau_iterations >= early_stop_patience:
                tprint_info(
                    f"  SA early stop: no improvement in best score for "
                    f"{early_stop_patience} iterations (last iter={iteration}); stopping SA."
                )
                break

        improvement = best_score - self._calculate_regime_quality_score(initial_labels, risk_features, None)
        tprint_success(
            f"✅ SA completed: best_score={best_score:.4f} "
            f"(improvement: +{improvement:.4f})"
        )

        return best_labels, best_score

    def _create_optimal_regime_labels(
        self,
        risk_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create 4 regime labels optimized for risk feature distinctiveness.
        NO temporal smoothing (removed per user request).

        Flow:
            1. Select RAW risk features (no EWMA)
            2. Drop correlated features (>0.95)
            3. Optional: Select discriminative features
            4. Optional: UMAP reduction
            5. GMM initialization
            6. Simulated annealing (100% risk CV)

        Returns:
            regime_labels: Regime assignments (0-3, -1 for invalid)
            metrics: Quality metrics
        """
        n_regimes = int(config.get("risk_n_regimes", 4))

        # ========== STEP 1: Select RAW Risk Features ONLY ==========
        # Prefer unsmoothed core risk features actually produced by _generate_risk_features.
        primary_risk_cols = [
            "vol_1h",
            "vol_3h",
            "vol_6h",
            "vol_ratio_1h_3h",
            "vol_ratio_3h_6h",
            "vol_ratio_1h_6h",
            "cvar_5pct",
            "cvar_10pct",
            "max_drawdown_50",
            "max_runup_50",
            "fragility_ratio",
            "shock_ratio",
            "desperation_clv",
            "volume_vol_divergence",
            "vol_expansion_5",
            "vol_acceleration",
            "downside_deviation_20",
            "upside_deviation_20",
            "volume_spike_ratio",
            "volume_dry_ratio",
            "volume_price_divergence",
        ]

        # Filter to available primary risk columns
        available_risk_cols = [c for c in primary_risk_cols if c in risk_df.columns]

        # Fallback: if none of the primary columns are present, use all numeric
        # risk features except raw OHLCV to keep the pipeline robust.
        if not available_risk_cols:
            numeric_cols = risk_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_prefixes = ["open", "high", "low", "close", "volume"]
            available_risk_cols = [
                c for c in numeric_cols
                if not any(c.startswith(pref) for pref in exclude_prefixes)
            ]

        if not available_risk_cols:
            raise ValueError(
                "No suitable risk features found for GMM initialization; "
                "verify that _generate_risk_features produced numeric columns."
            )

        risk_features = risk_df[available_risk_cols].copy()
        # Ensure purely numeric dtypes for downstream sklearn validation
        risk_features = risk_features.apply(pd.to_numeric, errors="coerce")

        # Drop columns that are entirely NaN (no usable information)
        non_empty_cols = [c for c in risk_features.columns if risk_features[c].notna().any()]
        if not non_empty_cols:
            raise ValueError(
                "All selected risk features are NaN; check risk feature generation configuration."
            )

        if len(non_empty_cols) < len(risk_features.columns):
            dropped_cols = [c for c in risk_features.columns if c not in non_empty_cols]
            tprint_warning(
                f"Dropping {len(dropped_cols)} all-NaN risk features before GMM: "
                f"{dropped_cols[:10]}..."
            )

        risk_features = risk_features[non_empty_cols]

        tprint_info(f"📊 Using {len(risk_features.columns)} RAW risk features (no smoothing)")

        # Median-impute remaining NaNs so we do not lose all samples due to window effects
        col_medians = risk_features.median(axis=0, skipna=True)
        risk_features_filled = risk_features.fillna(col_medians)

        # Track which rows are fully valid after imputation (should typically be all)
        valid_mask = risk_features_filled.notna().all(axis=1)
        risk_features_clean = risk_features_filled[valid_mask]

        tprint_info(
            f"  Valid samples after imputation: {len(risk_features_clean)}/{len(risk_df)}"
        )

        # ========== STEP 2: Drop Correlated Features ==========
        use_corr_filter = bool(config.get("risk_use_corr_filter", True))
        if use_corr_filter:
            risk_features_clean, dropped = self._drop_correlated_features(
                risk_features_clean,
                threshold=0.95
            )

        # ========== STEP 3: Feature Selection (Optional) ==========
        use_feature_selection = bool(config.get("risk_use_feature_selection", False))
        if use_feature_selection:
            # Do preliminary GMM to get labels for feature selection
            from sklearn.mixture import GaussianMixture
            gmm_temp = GaussianMixture(n_components=n_regimes, random_state=42)
            temp_labels = gmm_temp.fit_predict(risk_features_clean)

            top_k = int(config.get("risk_top_k_features", 20))
            selected_features = self._select_discriminative_features(
                risk_features_clean, temp_labels, top_k=top_k
            )
            risk_features_clean = risk_features_clean[selected_features]

        # ========== STEP 4: UMAP Reduction (Optional) ==========
        use_umap = bool(config.get("risk_use_umap", False))
        umap_reducer = None
        if use_umap and len(risk_features_clean.columns) > 10:
            risk_features_clean, umap_reducer = self._apply_umap_reduction(
                risk_features_clean,
                n_components=int(config.get("risk_umap_components", 8))
            )

        # ========== STEP 4.5: Restrict to structural risk features for GMM/SA ==========
        structural_feature_candidates = [
            "cvar_5pct",
            "cvar_10pct",
            "max_drawdown_50",
            "downside_deviation_20",
            "max_runup_50",
            "upside_deviation_20",
            "jump_frequency_20",
            "jump_frequency_20_ewma20",
            "vol_expansion_5",
            "vol_expansion_5_ewma20",
            "vol_acceleration",
            "vol_acceleration_ewma20",
            "risk_tail_cvar_raw_scaled",
            "risk_vol_acceleration_raw_scaled",
        ]
        structural_features = [
            col for col in structural_feature_candidates
            if col in risk_features_clean.columns
        ]

        if structural_features:
            gmm_sa_features = risk_features_clean[structural_features]
        else:
            gmm_sa_features = risk_features_clean

        # Ensure GMM input is float32 for efficiency
        gmm_sa_features = gmm_sa_features.astype(np.float32)

        max_gmm_feats = int(config.get("risk_gmm_max_features", 0))
        if max_gmm_feats > 0 and gmm_sa_features.shape[1] > max_gmm_feats:
            try:
                var_series_gmm = gmm_sa_features.var(axis=0)
                var_series_gmm = var_series_gmm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                ordered_gmm_cols = var_series_gmm.sort_values(ascending=False).index.tolist()

                corr_thr_gmm = float(config.get("risk_gmm_corr_threshold", 0.97))
                corr_mat_gmm = gmm_sa_features.corr().abs()
                corr_mat_gmm = corr_mat_gmm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                selected_gmm: List[str] = []
                for col in ordered_gmm_cols:
                    if len(selected_gmm) >= max_gmm_feats:
                        break
                    if not selected_gmm:
                        selected_gmm.append(col)
                        continue

                    try:
                        corr_with_sel = corr_mat_gmm.loc[col, selected_gmm]
                        if corr_with_sel.max() >= corr_thr_gmm:
                            continue
                    except Exception:
                        pass

                    selected_gmm.append(col)

                top_gmm_cols = [str(c) for c in selected_gmm]
                if top_gmm_cols:
                    gmm_sa_features = gmm_sa_features[top_gmm_cols]
                    tprint_info(
                        f"  GMM feature cap: using {len(top_gmm_cols)} structural features "
                        f"after variance/correlation pruning (out of {len(var_series_gmm)})."
                    )
            except Exception as gmm_feat_exc:
                tprint_warning(
                    f"GMM top-K feature cap failed (non-fatal); using all structural features: {gmm_feat_exc}"
                )

        # ========== STEP 5: GMM Initialization ==========
        # Validate that we still have a non-empty, numeric feature matrix
        if gmm_sa_features.empty or gmm_sa_features.shape[1] == 0:
            raise ValueError(
                "Risk feature matrix for GMM is empty after cleaning/correlation "
                "filtering; reduce filtering or check feature generation."
            )

        from sklearn.mixture import GaussianMixture

        # Optional subsampling: fit GMM on a representative subset of rows
        # while still predicting labels for the full feature matrix.
        gmm_fit_features = gmm_sa_features
        gmm_full_n = len(gmm_sa_features)
        gmm_enable_subsampling = bool(
            config.get("risk_gmm_enable_subsampling", False)
        )
        gmm_max_samples = int(config.get("risk_gmm_max_samples", 0))

        if gmm_enable_subsampling and gmm_max_samples > 0 and gmm_full_n > gmm_max_samples:
            try:
                rng_gmm = np.random.RandomState(
                    int(config.get("risk_gmm_subsample_random_state", 43))
                )
            except Exception:
                rng_gmm = np.random.RandomState(43)

            subset_idx = rng_gmm.choice(gmm_full_n, size=gmm_max_samples, replace=False)
            gmm_fit_features = gmm_sa_features.iloc[subset_idx]

            tprint_info(
                f"  GMM subsampling enabled: fitting on {len(gmm_fit_features)} samples "
                f"out of {gmm_full_n}"
            )

        tprint_info(f"Initializing {n_regimes} regimes with GMM using "
                    f"{gmm_sa_features.shape[1]} features and "
                    f"{len(gmm_sa_features)} samples...")

        exec_mode_gmm = str(config.get("execution_mode", "")).lower()

        # Use stronger defaults in blank mode now that we use diag covariance,
        # while still preventing extreme settings via hard caps.
        default_n_init = 10 if exec_mode_gmm == "blank" else 20
        default_max_iter = 200 if exec_mode_gmm == "blank" else 200

        raw_n_init = int(config.get("risk_gmm_n_init", default_n_init))
        raw_max_iter = int(config.get("risk_gmm_max_iter", default_max_iter))

        if exec_mode_gmm == "blank":
            # Hard caps in blank mode so config cannot accidentally request
            # extremely heavy GMM runs.
            n_init = min(raw_n_init, 10)
            max_iter = min(raw_max_iter, 200)
        else:
            n_init = raw_n_init
            max_iter = raw_max_iter

        # Use diagonal covariance by default in all modes, but allow override
        # via risk_gmm_covariance_type.
        default_covariance_type = "diag"
        covariance_type = str(config.get("risk_gmm_covariance_type", default_covariance_type))

        # Configurable GMM convergence tolerance so we can approximate early
        # stopping via a slightly looser threshold in full-mode runs. This
        # defaults to 5e-3 (less strict than sklearn's 1e-3) but can be
        # overridden via risk_gmm_tol in the launcher/step config.
        gmm_tol = float(config.get("risk_gmm_tol", 5e-3))

        tprint_info(
            "  GMM configuration: "
            f"covariance_type={covariance_type}, n_init={n_init}, max_iter={max_iter}, tol={gmm_tol}"
        )

        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            tol=gmm_tol,
            random_state=42,
        )

        gmm_start = time.time()
        tprint_info(
            f"  Starting GMM fit on matrix shape={gmm_fit_features.shape} "
            f"(n_init={n_init}, max_iter={max_iter}, tol={gmm_tol})"
        )
        gmm.fit(gmm_fit_features)
        initial_labels = gmm.predict(gmm_sa_features)
        gmm_duration = time.time() - gmm_start
        tprint_info(
            f"  GMM fit completed in {gmm_duration:.2f}s; proceeding to regime profiling "
            f"and simulated annealing (if enabled)..."
        )

        # DO NOT rank/reorder regimes - let GMM find distinct profiles
        # Each regime represents a different risk PROFILE, not a continuum
        # Characterize each regime by its feature profile
        regime_profiles = {}
        for regime_id in range(n_regimes):
            regime_mask = initial_labels == regime_id
            regime_data = risk_features_clean[regime_mask]

            if len(regime_data) > 0:
                # Calculate regime characteristics
                profile = {
                    'regime_id': regime_id,
                    'size': len(regime_data),
                    'mean_vol_1h': regime_data.get('risk_fwd_vol_1h_raw_scaled', pd.Series([0])).mean(),
                    'mean_vol_4h': regime_data.get('risk_fwd_vol_4h_raw_scaled', pd.Series([0])).mean(),
                    'mean_tail_risk': regime_data.get('risk_tail_cvar_raw_scaled', pd.Series([0])).mean(),
                    'mean_vol_accel': regime_data.get('risk_vol_acceleration_raw_scaled', pd.Series([0])).mean(),
                    'mean_features': regime_data.mean().mean(),
                    'std_features': regime_data.std().mean(),
                }
                regime_profiles[regime_id] = profile

                tprint_info(
                    f"  Regime {regime_id}: n={profile['size']}, "
                    f"vol_1h={profile['mean_vol_1h']:.3f}, "
                    f"tail_risk={profile['mean_tail_risk']:.3f}"
                )

        score_start = time.time()
        initial_score = self._calculate_regime_quality_score(
            initial_labels, gmm_sa_features, None
        )
        score_duration = time.time() - score_start
        tprint_info(
            f"  GMM initialization: score={initial_score:.4f} "
            f"(quality score computed in {score_duration:.2f}s)"
        )

        # ========== STEP 6: Simulated Annealing Refinement ==========
        use_sa_refinement = bool(config.get("risk_use_sa_refinement", True))

        if use_sa_refinement:
            sa_flip_fraction = float(config.get("risk_sa_flip_fraction", 0.10))
            sa_neighborhood_mode = str(config.get("risk_sa_neighborhood_mode", "any"))
            exec_mode_sa = str(config.get("execution_mode", "")).lower()

            # Subsampling is now opt-in in all modes; default is disabled.
            sa_enable_subsampling = bool(
                config.get("risk_sa_enable_subsampling", False)
            )
            sa_max_samples = int(config.get("risk_sa_max_samples", 6000))
            sa_max_iterations = int(config.get("risk_sa_iterations", 2000))

            # Early stopping: enabled by default in blank mode with
            # conservative defaults, but fully configurable via config.
            early_stop_patience_raw = config.get("risk_sa_early_stop_patience")
            good_enough_improvement_raw = config.get("risk_sa_good_enough_improvement")

            if exec_mode_sa == "blank":
                if early_stop_patience_raw is None:
                    early_stop_patience_raw = 300
                if good_enough_improvement_raw is None:
                    good_enough_improvement_raw = 0.015

            early_stop_patience = early_stop_patience_raw
            if early_stop_patience is not None:
                try:
                    early_stop_patience = int(early_stop_patience)
                except Exception:
                    early_stop_patience = None

            good_enough_improvement = good_enough_improvement_raw
            if good_enough_improvement is not None:
                try:
                    good_enough_improvement = float(good_enough_improvement)
                except Exception:
                    good_enough_improvement = None

            tprint_info(
                "  SA configuration: "
                f"iterations={sa_max_iterations}, flip_fraction={sa_flip_fraction:.3f}, "
                f"subsampling={sa_enable_subsampling} (max_samples={sa_max_samples}), "
                f"early_stop_patience={early_stop_patience}, "
                f"good_enough_improvement={good_enough_improvement}"
            )

            sa_features = risk_features_clean
            sa_labels = initial_labels
            subset_indices: Optional[np.ndarray] = None

            if sa_enable_subsampling and len(risk_features_clean) > sa_max_samples:
                try:
                    rng = np.random.RandomState(
                        int(config.get("risk_sa_subsample_random_state", 42))
                    )
                except Exception:
                    rng = np.random.RandomState(42)

                subset_indices = rng.choice(
                    len(risk_features_clean), size=sa_max_samples, replace=False
                )
                sa_features = risk_features_clean.iloc[subset_indices]
                sa_labels = initial_labels[subset_indices]

                tprint_info(
                    f"  SA subsampling enabled: using {len(sa_features)} samples "
                    f"out of {len(risk_features_clean)} for optimization"
                )

            sa_start = time.time()
            tprint_info(
                "  Starting SA refinement (simulated annealing) "
                f"for up to {sa_max_iterations} iterations..."
            )
            refined_labels_sa, refined_score = self._refine_labels_simulated_annealing(
                initial_labels=sa_labels,
                risk_features=sa_features,
                forward_returns=None,
                n_regimes=n_regimes,
                max_iterations=sa_max_iterations,
                initial_temp=float(config.get("risk_sa_initial_temp", 2.0)),
                cooling_rate=float(config.get("risk_sa_cooling_rate", 0.995)),
                flip_fraction=sa_flip_fraction,
                neighborhood_mode=sa_neighborhood_mode,
                early_stop_patience=early_stop_patience,
                good_enough_improvement=good_enough_improvement,
            )
            sa_duration = time.time() - sa_start
            tprint_info(
                f"  SA refinement completed in {sa_duration:.2f}s; "
                f"best_score={refined_score:.4f}"
            )

            if subset_indices is not None:
                refined_labels = initial_labels.copy()
                refined_labels[subset_indices] = refined_labels_sa
            else:
                refined_labels = refined_labels_sa

            final_labels = refined_labels
            final_score = refined_score
        else:
            final_labels = initial_labels
            final_score = initial_score

        # ========== STEP 7: Calculate Final Metrics ==========
        metrics = self._calculate_regime_quality_metrics(
            final_labels, risk_features_clean, None
        )
        tprint_info(
            "  Final label quality: "
            f"risk_cv_ratio={float(metrics.get('risk_cv_ratio', 0.0)):.4f}, "
            f"min_regime_pct={float(metrics.get('min_regime_pct', 0.0)):.3f}, "
            f"max_regime_pct={float(metrics.get('max_regime_pct', 0.0)):.3f}"
        )

        # Persist a compact, human-readable summary of label quality metrics so
        # WCoV and separation diagnostics are easy to inspect across runs.
        try:
            symbol_q = str(config.get("symbol", ""))
            exchange_q = str(config.get("exchange", ""))
            regime_tf_q = str(config.get("regime_timeframe", config.get("timeframe", "15m")))

            quality_row = {
                "symbol": symbol_q,
                "exchange": exchange_q,
                "timeframe": regime_tf_q,
                "risk_cv_ratio": float(metrics.get("risk_cv_ratio", 0.0)),
                "risk_cv_between": float(metrics.get("risk_cv_between", 0.0)),
                "risk_cv_within": float(metrics.get("risk_cv_within", 0.0)),
                "wasserstein_distance": float(metrics.get("wasserstein_distance", 0.0)),
                "kl_divergence": float(metrics.get("kl_divergence", 0.0)),
                "min_regime_pct": float(metrics.get("min_regime_pct", 0.0)),
                "max_regime_pct": float(metrics.get("max_regime_pct", 0.0)),
                "quality_score": float(metrics.get("quality_score", 0.0)),
            }

            # Also include regime distribution as a string for quick inspection.
            regime_dist = metrics.get("regime_distribution", {})
            try:
                quality_row["regime_distribution"] = json.dumps(regime_dist)
            except Exception:
                quality_row["regime_distribution"] = str(regime_dist)

            quality_df = pd.DataFrame([quality_row])
            ts_q = datetime.now().strftime("%Y%m%d_%H%M%S")
            quality_path = (
                f"outcomes/ml_risk_label_quality_"
                f"{symbol_q or 'UNKNOWN'}_{regime_tf_q}_{ts_q}.csv"
            )
            quality_df.to_csv(quality_path, index=False)
            tprint_info(f"💾 Saved risk label quality summary: {quality_path}")
        except Exception as quality_exc:  # pragma: no cover - defensive
            tprint_warning(f"Failed to persist risk label quality summary (non-fatal): {quality_exc}")

        # Expand labels back to full dataframe
        full_labels = np.full(len(risk_df), -1, dtype=int)
        full_labels[valid_mask] = final_labels

        teacher_probs_full: Optional[np.ndarray]
        teacher_probs_full = None
        try:
            responsibilities = gmm.predict_proba(risk_features_clean)
            if responsibilities.shape[0] == valid_mask.sum():
                teacher_probs_full = np.zeros((len(risk_df), n_regimes), dtype=float)
                teacher_probs_full[valid_mask] = responsibilities
        except Exception as resp_exc:
            tprint_warning(f"Teacher probability extraction from GMM failed; skipping distillation helpers: {resp_exc}")

        # Store feature selection artifacts
        metrics['selected_features'] = list(risk_features_clean.columns)
        metrics['umap_reducer'] = umap_reducer
        metrics['n_features_used'] = len(risk_features_clean.columns)
        if teacher_probs_full is not None:
            metrics['teacher_probs'] = teacher_probs_full

        tprint_success(
            f"✅ Created {n_regimes} regime labels (NO temporal smoothing):\n"
            f"   Risk CV Ratio={metrics['risk_cv_ratio']:.3f}, "
            f"Wasserstein={metrics['wasserstein_distance']:.3f}, "
            f"KL Divergence={metrics['kl_divergence']:.3f}\n"
            f"   Regime Distribution: {metrics['regime_distribution']}"
        )

        return full_labels, metrics

    def _generate_feature_importance_report(
        self,
        importance_data: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        classifier_metrics: Dict[str, Any],
        label_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive markdown report for feature importance analysis.

        Args:
            importance_data: Dict with 'global' and 'per_regime' DataFrames
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            classifier_metrics: XGBoost training metrics
            label_metrics: GMM label creation metrics

        Returns:
            Path to saved report file
        """
        from datetime import datetime
        import os

        # Ensure outcomes directory exists
        os.makedirs("outcomes", exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"outcomes/{symbol}_{exchange}_{timeframe}_XGBoost_Feature_Importance_{timestamp}.md"

        global_df = importance_data['global']
        per_regime_importance = importance_data['per_regime']
        n_regimes = importance_data['n_regimes']

        with open(report_path, 'w') as f:
            # Header
            f.write(f"# XGBoost Risk Regime Feature Importance Report\n\n")
            f.write(f"**Symbol**: {symbol} | **Exchange**: {exchange} | **Timeframe**: {timeframe}\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")

            # Model Performance Summary
            f.write(f"## Model Performance Summary\n\n")
            f.write(f"### Classifier Metrics\n")
            f.write(f"- **Validation Accuracy**: {classifier_metrics.get('val_accuracy', 0):.3f}\n")
            f.write(f"- **Validation Log Loss**: {classifier_metrics.get('val_log_loss', 0):.4f}\n")
            f.write(f"- **Number of Regimes**: {n_regimes}\n")
            f.write(f"- **Number of Features**: {classifier_metrics.get('n_features', 0)}\n\n")

            f.write(f"### Label Quality Metrics\n")
            f.write(f"- **Risk CV Ratio**: {label_metrics.get('risk_cv_ratio', 0):.3f}\n")
            f.write(f"- **Wasserstein Distance**: {label_metrics.get('wasserstein_distance', 0):.3f}\n")
            f.write(f"- **KL Divergence**: {label_metrics.get('kl_divergence', 0):.3f}\n")
            f.write(f"- **Quality Score**: {label_metrics.get('quality_score', 0):.3f}\n")
            f.write(f"- **Regime Distribution**: {label_metrics.get('regime_distribution', {})}\n\n")

            f.write(f"---\n\n")

            # Global Feature Importance
            f.write(f"## Global Feature Importance\n\n")
            f.write(f"Top features across all regimes, ranked by combined score (average of weight, gain, and cover).\n\n")
            f.write(f"### Metrics Explanation\n")
            f.write(f"- **Weight**: Number of times feature is used in tree splits (normalized)\n")
            f.write(f"- **Gain**: Average improvement in loss when feature is used (normalized)\n")
            f.write(f"- **Cover**: Average number of samples affected by splits using this feature (normalized)\n")
            f.write(f"- **Combined**: Average of weight, gain, and cover (normalized)\n\n")

            # Top 30 global features
            f.write(f"### Top 30 Features (Global)\n\n")
            f.write(f"| Rank | Feature | Weight | Gain | Cover | Combined |\n")
            f.write(f"|------|---------|--------|------|-------|----------|\n")
            for rank, (idx, row) in enumerate(global_df.head(30).iterrows(), start=1):
                f.write(
                    f"| {rank:3d} | {row['feature'][:40]:40s} | "
                    f"{row['weight_norm']:.4f} | {row['gain_norm']:.4f} | "
                    f"{row['cover_norm']:.4f} | {row['combined_score']:.4f} |\n"
                )

            f.write(f"\n---\n\n")

            # Per-Regime Feature Importance
            f.write(f"## Per-Regime Feature Distinctiveness\n\n")
            f.write(f"Features that best distinguish each regime from others.\n\n")
            f.write(f"### Metrics Explanation\n")
            f.write(f"- **Regime Mean**: Average feature value in this regime\n")
            f.write(f"- **Other Mean**: Average feature value in all other regimes\n")
            f.write(f"- **Mean Sep**: Normalized separation between regime and others (in std units)\n")
            f.write(f"- **CV Ratio**: Between-regime variance / within-regime variance\n")
            f.write(f"- **Global Gain**: Feature's global gain importance (0-1)\n")
            f.write(f"- **Regime Imp**: Combined regime-specific importance (MeanSep × CVRatio × GlobalGain)\n\n")

            for regime_id in sorted(per_regime_importance.keys()):
                regime_df = per_regime_importance[regime_id]

                f.write(f"### Regime {regime_id} - Top 20 Distinguishing Features\n\n")
                f.write(f"| Rank | Feature | Regime Mean | Other Mean | Mean Sep | CV Ratio | Global Gain | Regime Imp |\n")
                f.write(f"|------|---------|-------------|------------|----------|----------|-------------|------------|\n")

                for rank, (idx, row) in enumerate(regime_df.head(20).iterrows(), start=1):
                    f.write(
                        f"| {rank:3d} | {row['feature'][:35]:35s} | "
                        f"{row['regime_mean']:11.4f} | {row['other_mean']:10.4f} | "
                        f"{row['mean_separation']:8.2f} | {row['cv_ratio']:8.2f} | "
                        f"{row['global_gain']:11.4f} | {row['regime_importance']:10.2f} |\n"
                    )

                f.write(f"\n")

            f.write(f"---\n\n")

            # Per-Regime Classification Performance
            if 'classification_report' in classifier_metrics:
                f.write(f"## Per-Regime Classification Performance\n\n")
                f.write(f"| Regime | Precision | Recall | F1-Score | Support |\n")
                f.write(f"|--------|-----------|--------|----------|---------|\n")

                report = classifier_metrics['classification_report']
                for regime_id in range(n_regimes):
                    regime_key = f'Regime_{regime_id}'
                    if regime_key in report:
                        r = report[regime_key]
                        f.write(
                            f"| {regime_id} | {r.get('precision', 0):.3f} | "
                            f"{r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | "
                            f"{int(r.get('support', 0))} |\n"
                        )

                f.write(f"\n")

            # Feature Categories Summary
            f.write(f"---\n\n")
            f.write(f"## Feature Category Analysis\n\n")

            # Categorize features by type
            feature_categories = {
                'Volatility': ['vol', 'parkinson', 'garman_klass'],
                'Tail Risk': ['cvar', 'drawdown', 'downside'],
                'Distribution': ['skewness', 'kurtosis'],
                'Dynamics': ['acceleration', 'expansion', 'jump', 'momentum'],
                'Cross-Timeframe': ['ratio', 'ewma'],
                'Hurst/Persistence': ['hurst'],
                'Divergence': ['correlation', 'divergence', 'fragility', 'shock', 'desperation']
            }

            for category, keywords in feature_categories.items():
                category_features = global_df[
                    global_df['feature'].str.lower().str.contains('|'.join(keywords), na=False)
                ]

                if len(category_features) > 0:
                    f.write(f"### {category} Features (Top 10)\n\n")
                    f.write(f"| Feature | Combined Score |\n")
                    f.write(f"|---------|----------------|\n")

                    for idx, row in category_features.head(10).iterrows():
                        f.write(f"| {row['feature'][:50]:50s} | {row['combined_score']:.4f} |\n")

                    f.write(f"\n")

            f.write(f"---\n\n")
            f.write(f"*Report generated by ml_risk_regime_step with XGBoost multi-class classifier*\n")

        return report_path

    def _update_risk_diagnostics_feature_importance(
        self,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
        classifier_metrics: Dict[str, Any],
    ) -> None:
        """Update ml_risk_regime_diagnostics_* markdown with feature importance from this run.

        This ensures the diagnostics file uses the per-run feature_importance_detailed
        structure produced during the current ml_risk_regime_step execution.
        """

        if 'feature_importance_detailed' not in classifier_metrics:
            return

        importance_data = classifier_metrics['feature_importance_detailed']

        try:
            global_df = pd.DataFrame(importance_data['global'])
            per_regime_importance = {
                regime_id: pd.DataFrame(records)
                for regime_id, records in importance_data['per_regime'].items()
            }
        except Exception as exc:
            tprint_warning(f"Failed to reconstruct feature importance for diagnostics: {exc}")
            return

        diagnostics_path = f"outcomes/ml_risk_regime_diagnostics_{symbol}_{regime_timeframe}.md"

        try:
            with open(diagnostics_path, 'r') as f:
                existing = f.read()
        except FileNotFoundError:
            existing = ""

        header = "## XGBoost Feature Importance by Regime"
        footer = "## Generated CI Overlap Plots"

        # Build new feature-importance section
        lines: List[str] = []
        lines.append(header)
        lines.append("")

        # Global top 20
        lines.append("### Global Top 20 Features")
        lines.append("")
        lines.append("| Rank | Feature | Weight | Gain | Cover | Combined |")
        lines.append("|------|---------|--------|------|-------|----------|")

        if not global_df.empty:
            top_global = global_df.sort_values('combined_score', ascending=False).head(20)
            for rank, (_, row) in enumerate(top_global.iterrows(), start=1):
                lines.append(
                    f"| {rank:3d} | {str(row['feature'])[:40]:40s} | "
                    f"{float(row.get('weight_norm', 0.0)):.4f} | "
                    f"{float(row.get('gain_norm', 0.0)):.4f} | "
                    f"{float(row.get('cover_norm', 0.0)):.4f} | "
                    f"{float(row.get('combined_score', 0.0)):.4f} |"
                )

        lines.append("")
        lines.append("### Per-Regime Top 10 Distinguishing Features")
        lines.append("")

        for regime_id in sorted(per_regime_importance.keys()):
            regime_df = per_regime_importance[regime_id]
            if regime_df.empty:
                continue

            lines.append(f"#### Regime {regime_id}")
            lines.append("")
            lines.append("| Rank | Feature | Regime Mean | Other Mean | Mean Sep | CV Ratio | Regime Imp |")
            lines.append("|------|---------|-------------|------------|----------|----------|------------|")

            top_regime = regime_df.sort_values('regime_importance', ascending=False).head(10)
            for rank, (_, row) in enumerate(top_regime.iterrows(), start=1):
                lines.append(
                    f"| {rank:3d} | {str(row['feature'])[:35]:35s} | "
                    f"{float(row.get('regime_mean', 0.0)):.4f} | "
                    f"{float(row.get('other_mean', 0.0)):.4f} | "
                    f"{float(row.get('mean_separation', 0.0)):.2f} | "
                    f"{float(row.get('cv_ratio', 0.0)):.2f} | "
                    f"{float(row.get('regime_importance', 0.0)):.2f} |"
                )

            lines.append("")

        new_section = "\n".join(lines) + "\n"

        if existing:
            if header in existing:
                start = existing.find(header)
                end = existing.find(footer, start) if footer in existing else -1
                if end != -1:
                    updated = existing[:start] + new_section + existing[end:]
                else:
                    updated = existing[:start] + new_section
            else:
                # Insert before footer if present, otherwise append
                if footer in existing:
                    end = existing.find(footer)
                    updated = existing[:end] + new_section + existing[end:]
                else:
                    updated = existing.rstrip() + "\n\n" + new_section
        else:
            # Create a minimal diagnostics file if none exists
            updated = (
                f"# ML Risk Regime Diagnostics for {symbol} {regime_timeframe}\n\n" +
                new_section
            )

        try:
            with open(diagnostics_path, 'w') as f:
                f.write(updated)
        except Exception as exc:
            tprint_warning(f"Failed to update diagnostics markdown with feature importance: {exc}")

    def _calculate_comprehensive_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive feature importance metrics from XGBoost model.

        Returns:
            - Global importance: weight, gain, cover for all features
            - Per-regime importance: feature distinctiveness for each regime

        Args:
            model: Trained XGBoost classifier
            X: Feature matrix (scaled)
            y: Regime labels
            feature_names: List of feature names

        Returns:
            Dict with global and per-regime importance metrics
        """
        import xgboost as xgb

        n_regimes = int(y.max() + 1)

        # ========== GLOBAL FEATURE IMPORTANCE ==========
        booster = model.get_booster()

        # Get all three importance types
        importance_weight = booster.get_score(importance_type='weight')
        importance_gain = booster.get_score(importance_type='gain')
        importance_cover = booster.get_score(importance_type='cover')

        # Create comprehensive global importance dataframe
        global_importance = []
        for feat in feature_names:
            # XGBoost uses f0, f1, f2... internally, map back to feature names
            feat_idx = feature_names.index(feat)
            feat_key = f'f{feat_idx}'

            global_importance.append({
                'feature': feat,
                'weight': importance_weight.get(feat_key, 0),
                'gain': importance_gain.get(feat_key, 0.0),
                'cover': importance_cover.get(feat_key, 0.0),
            })

        global_df = pd.DataFrame(global_importance)

        # Normalize each metric to sum to 1
        if global_df['weight'].sum() > 0:
            global_df['weight_norm'] = global_df['weight'] / global_df['weight'].sum()
        else:
            global_df['weight_norm'] = 0.0

        if global_df['gain'].sum() > 0:
            global_df['gain_norm'] = global_df['gain'] / global_df['gain'].sum()
        else:
            global_df['gain_norm'] = 0.0

        if global_df['cover'].sum() > 0:
            global_df['cover_norm'] = global_df['cover'] / global_df['cover'].sum()
        else:
            global_df['cover_norm'] = 0.0

        # Combined score (average of normalized metrics)
        global_df['combined_score'] = (
            global_df['weight_norm'] +
            global_df['gain_norm'] +
            global_df['cover_norm']
        ) / 3.0

        global_df = global_df.sort_values('combined_score', ascending=False)

        # ========== PER-REGIME FEATURE IMPORTANCE ==========
        # Calculate feature distinctiveness for each regime
        # A feature is important for a regime if its distribution differs significantly from other regimes

        per_regime_importance = {}

        for regime_id in range(n_regimes):
            regime_mask = (y == regime_id)
            other_mask = (y != regime_id)

            if regime_mask.sum() == 0 or other_mask.sum() == 0:
                continue

            regime_features = []

            for feat in feature_names:
                feat_idx = X.columns.get_loc(feat)
                feat_values = X.iloc[:, feat_idx]

                # Calculate distinctiveness metrics
                regime_mean = feat_values[regime_mask].mean()
                other_mean = feat_values[other_mask].mean()
                regime_std = feat_values[regime_mask].std()
                other_std = feat_values[other_mask].std()

                # Mean separation (normalized)
                mean_separation = abs(regime_mean - other_mean) / (regime_std + other_std + 1e-8)

                # Coefficient of variation ratio (between/within)
                pooled_mean = feat_values.mean()
                between_var = (regime_mean - pooled_mean)**2 * regime_mask.sum() + \
                              (other_mean - pooled_mean)**2 * other_mask.sum()
                within_var = regime_std**2 * regime_mask.sum() + other_std**2 * other_mask.sum()
                cv_ratio = between_var / (within_var + 1e-8)

                # Get global importance for this feature
                global_weight = global_df[global_df['feature'] == feat]['weight_norm'].values[0] if len(global_df[global_df['feature'] == feat]) > 0 else 0
                global_gain = global_df[global_df['feature'] == feat]['gain_norm'].values[0] if len(global_df[global_df['feature'] == feat]) > 0 else 0
                global_cover = global_df[global_df['feature'] == feat]['cover_norm'].values[0] if len(global_df[global_df['feature'] == feat]) > 0 else 0

                regime_features.append({
                    'feature': feat,
                    'regime_mean': float(regime_mean),
                    'other_mean': float(other_mean),
                    'mean_separation': float(mean_separation),
                    'cv_ratio': float(cv_ratio),
                    'global_weight': float(global_weight),
                    'global_gain': float(global_gain),
                    'global_cover': float(global_cover),
                    # Combined regime-specific importance
                    'regime_importance': float(mean_separation * cv_ratio * (global_gain + 0.1))
                })

            regime_df = pd.DataFrame(regime_features).sort_values('regime_importance', ascending=False)
            per_regime_importance[regime_id] = regime_df

        return {
            'global': global_df,
            'per_regime': per_regime_importance,
            'n_regimes': n_regimes
        }

    def _train_regime_classifier(
        self,
        risk_df: pd.DataFrame,
        regime_labels: np.ndarray,
        config: Dict[str, Any],
        label_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """
        Train XGBoost multi-class classifier to predict regimes with probabilities.
        Uses RAW features (no EWMA smoothing).

        Args:
            risk_df: Feature dataframe
            regime_labels: Target regime labels (0-3, -1 for invalid)
            config: Configuration dict

        Returns:
            model: Trained XGBoost classifier
            regime_probs: Predicted probabilities (n_samples x 4)
            training_metrics: Performance metrics
        """
        import xgboost as xgb
        from sklearn.metrics import classification_report, log_loss, accuracy_score

        # Filter valid samples
        valid_mask = regime_labels >= 0
        df_clean = risk_df[valid_mask].copy()
        y = regime_labels[valid_mask]

        teacher_probs = None
        teacher_conf_all = None
        if label_metrics is not None and "teacher_probs" in label_metrics:
            try:
                teacher_array = np.asarray(label_metrics["teacher_probs"])
                if teacher_array.shape[0] == len(regime_labels):
                    teacher_probs = teacher_array[valid_mask]
            except Exception as teacher_exc:
                tprint_warning(f"Teacher probabilities unavailable; skipping distillation helpers: {teacher_exc}")
                teacher_probs = None

        if teacher_probs is not None:
            try:
                teacher_conf_all = teacher_probs.max(axis=1)
            except Exception:
                teacher_conf_all = None

        # Select features (exclude risk targets and intermediate components)
        numeric_df = df_clean.select_dtypes(include=[np.number])
        feature_cols = [
            col for col in numeric_df.columns
            if not col.startswith("risk_target")
            and not col.startswith("risk_regime")
            and not col.startswith("alpha_")
        ]

        X = numeric_df[feature_cols]

        tprint_info(f"🤖 Training XGBoost classifier on {len(feature_cols)} RAW features")

        # Chronological split
        train_frac = float(config.get("risk_train_fraction", 0.8))
        split_idx = int(len(X) * train_frac)

        X_train_raw, y_train = X.iloc[:split_idx], y[:split_idx]
        X_val_raw, y_val = X.iloc[split_idx:], y[split_idx:]

        teacher_conf_train = None
        if teacher_conf_all is not None and len(teacher_conf_all) == len(y):
            teacher_conf_train = teacher_conf_all[:split_idx]

        # Robust scaling ONLY (no EWMA smoothing)
        from src.features_common.transforms.scaling_normalization import ScalingNormalizer

        normalizer_config = {
            "default_strategy": "robust",
            "auto_select": False,
            "handle_outliers": True,
            "outlier_threshold": 3.0,
            "use_vectorbt": False,
        }
        scaler = ScalingNormalizer(normalizer_config)

        X_train = scaler.fit_transform(X_train_raw, strategy="robust")
        X_val = scaler.transform(X_val_raw)
        X_full = scaler.transform(X)

        # Use float32 features for XGBoost
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_full = X_full.astype(np.float32)

        # Optional: restrict classifier to top-K features by variance, while
        # also pruning highly correlated features to keep the set compact and
        # diverse.
        max_clf_feats = int(config.get("risk_classifier_max_features", 0))
        if max_clf_feats > 0 and X_full.shape[1] > max_clf_feats:
            try:
                var_series = X_train.var(axis=0)
                var_series = var_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # Rank features by variance (descending)
                ordered_cols = var_series.sort_values(ascending=False).index.tolist()

                # Compute absolute correlation matrix once on the scaled
                # training data.
                corr_thr = float(
                    config.get("risk_classifier_corr_threshold", 0.97)
                )
                corr_mat = X_train.corr().abs()
                corr_mat = corr_mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                selected: List[str] = []
                for col in ordered_cols:
                    if len(selected) >= max_clf_feats:
                        break
                    if not selected:
                        selected.append(col)
                        continue

                    # Skip if this feature is too correlated with any
                    # already-selected feature.
                    try:
                        corr_with_selected = corr_mat.loc[col, selected]
                        if corr_with_selected.max() >= corr_thr:
                            continue
                    except Exception:
                        # Fallback: if correlation lookup fails, keep the
                        # feature to avoid silently dropping too many.
                        pass

                    selected.append(col)

                top_cols = [str(c) for c in selected]
                if top_cols:
                    X_train = X_train[top_cols]
                    X_val = X_val[top_cols]
                    X_full = X_full[top_cols]
                    feature_cols = top_cols
                    tprint_info(
                        f"🧪 Restricting classifier to {len(top_cols)} features by variance/correlation "
                        f"out of {len(var_series)} total (corr_thr={corr_thr:.3f})"
                    )
            except Exception as feat_cap_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Top-K feature cap for classifier failed (non-fatal); using all features: {feat_cap_exc}"
                )

        # Define monotonic constraints for risk features
        monotone_constraints = []
        for feat in X_full.columns:
            feat_lower = feat.lower()
            if any(kw in feat_lower for kw in [
                'vol', 'cvar', 'drawdown', 'jump', 'acceleration',
                'fragility', 'shock', 'tail', 'kurtosis', 'correlation'
            ]):
                monotone_constraints.append(1)  # Risk-increasing
            else:
                monotone_constraints.append(0)  # No constraint

        # Convert to XGBoost-compatible monotone constraints string
        monotone_constraints_param = "(" + ",".join(str(c) for c in monotone_constraints) + ")"

        # XGBoost Classifier Parameters
        n_regimes = int(regime_labels.max() + 1)

        base_params = {
            'objective': 'multi:softprob',
            'num_class': n_regimes,
            'tree_method': 'hist',
            'n_jobs': -1,

            # Structure (shallower trees for classification)
            'max_depth': int(config.get("risk_classifier_max_depth", 5)),
            'min_child_weight': int(config.get("risk_classifier_min_child_weight", 30)),

            # Learning dynamics
            'learning_rate': float(config.get("risk_classifier_learning_rate", 0.05)),
            'n_estimators': int(config.get("risk_classifier_n_estimators", 800)),

            # Regularization (stronger for classification)
            'subsample': float(config.get("risk_classifier_subsample", 0.7)),
            'colsample_bytree': float(config.get("risk_classifier_colsample_bytree", 0.7)),
            'gamma': float(config.get("risk_classifier_gamma", 2.0)),
            'reg_alpha': float(config.get("risk_classifier_reg_alpha", 1.0)),
            'reg_lambda': float(config.get("risk_classifier_reg_lambda", 2.0)),

            # Monotonic constraints
            'monotone_constraints': monotone_constraints_param,

            # Evaluation
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,

            'random_state': 42,
        }

        best_params = base_params.copy()

        # Optional high-confidence training configuration
        enable_high_conf = bool(config.get("risk_enable_high_confidence_training", False))
        high_conf_threshold = float(config.get("risk_high_confidence_threshold", 0.7))
        min_high_conf_fraction = float(config.get("risk_min_high_conf_fraction", 0.3))
        min_high_conf_per_regime = int(config.get("risk_min_high_conf_per_regime", 30))

        def _select_quadrant_feature_cols(all_cols: List[str]) -> List[str]:
            selected: List[str] = []

            regime_tf = str(
                config.get("regime_timeframe", config.get("timeframe", "1h"))
            ).lower()

            if regime_tf in ("1h", "60m"):
                # 1 bar ≈ 1h → 30m–3h ≈ 0.5–3 bars.
                # Use target ~3 bars with a slightly wider band [2, 8]
                # to capture both slightly shorter and longer horizons.
                default_target = 3
                default_min = 2
                default_max = 8
            elif regime_tf in ("30m", "0.5h"):
                # 1 bar ≈ 30m → 30m–3h ≈ 1–6 bars → target ~4, band [2, 8]
                default_target = 4
                default_min = 2
                default_max = 8
            elif regime_tf in ("15m",):
                # 1 bar ≈ 15m → 30m–3h ≈ 2–12 bars → target ~8, band [4, 16]
                default_target = 8
                default_min = 4
                default_max = 16
            else:
                # Fallback: generic intraday windows
                default_target = 12
                default_min = 6
                default_max = 24

            target_window = int(
                config.get("risk_quadrant_target_window_bars", default_target)
            )
            min_window = int(config.get("risk_quadrant_min_window_bars", default_min))
            max_window = int(config.get("risk_quadrant_max_window_bars", default_max))

            def _pick_closest(
                cols: List[str],
                target_n: int,
            ) -> Optional[str]:
                best_col: Optional[str] = None
                best_dist: float = float("inf")
                for col in cols:
                    matches = re.findall(r"\d+", col)
                    if not matches:
                        continue
                    try:
                        n_val = int(matches[-1])
                    except Exception:
                        continue
                    if n_val < min_window or n_val > max_window:
                        continue
                    dist = abs(n_val - target_n)
                    if dist < best_dist:
                        best_dist = dist
                        best_col = col
                if best_col is not None:
                    return best_col
                return cols[0] if cols else None

            # Prefer canonical RSI_14 when available, otherwise fall back to
            # other RSI windows near the target bar count.
            rsi_candidates = [
                c for c in all_cols
                if c.upper() == "RSI_14" or c.lower().endswith("rsi_14")
            ]
            if not rsi_candidates:
                rsi_candidates = [c for c in all_cols if "rsi" in c.lower()]
            rsi_col = _pick_closest(rsi_candidates, target_window) if rsi_candidates else None
            if rsi_col is not None:
                selected.append(rsi_col)

            # Volatility-of-volatility (vol_of_vol) family
            vov_candidates = [
                c for c in all_cols
                if "vol_of_vol" in c.lower() or "volofvol" in c.lower()
            ]
            if vov_candidates:
                selected.append(vov_candidates[0])

            # Prefer canonical parkinson_volatility name when present
            park_candidates = [
                c for c in all_cols
                if c.lower() == "parkinson_volatility" or "parkinson" in c.lower()
            ]
            if park_candidates:
                selected.append(park_candidates[0])

            # Autocorrelation: prefer standard autocorrelation_* features when
            # present (e.g. autocorrelation_5, autocorrelation_20), otherwise
            # fall back to any feature mentioning autocorr.
            autocorr_candidates = [
                c for c in all_cols
                if c.lower().startswith("autocorrelation_")
            ]
            if not autocorr_candidates:
                autocorr_candidates = [
                    c for c in all_cols
                    if "autocorr" in c.lower() or "autocorrelation" in c.lower()
                ]
            autocorr_col = _pick_closest(autocorr_candidates, target_window) if autocorr_candidates else None
            if autocorr_col is not None:
                selected.append(autocorr_col)

            return list(dict.fromkeys(selected))

        enable_hpo = bool(config.get("risk_enable_hpo", False) or config.get("enable_hpo", False))
        enable_quadrant_objective = bool(
            config.get("risk_hpo_enable_quadrant_objective", True)
        )

        # Make HPO mode-aware: in blank execution mode, keep HPO enabled by
        # default, but allow explicit opt-out via risk_disable_hpo_in_blank.
        exec_mode_cfg = str(config.get("execution_mode", "")).lower()
        if exec_mode_cfg == "blank" and bool(config.get("risk_disable_hpo_in_blank", False)):
            enable_hpo = False

        if enable_hpo:
            try:
                hpo_param_groups = [
                    create_param_group(
                        name="structure",
                        params={
                            # Allow moderately deep trees and smaller child
                            # weights so the classifier can carve out sharper
                            # quadrants when the data supports it.
                            "max_depth": {"type": "int", "low": 4, "high": 8},
                            "min_child_weight": {"type": "int", "low": 5, "high": 80},
                            "n_estimators": {"type": "int", "low": 300, "high": 1600},
                        },
                        priority=1,
                        description="Tree depth, leaf size, and capacity",
                    ),
                    create_param_group(
                        name="learning",
                        params={
                            # Slightly wider learning-rate range; HPO will
                            # settle on stable values based on WCoV objective.
                            "learning_rate": {"type": "float", "low": 0.01, "high": 0.20},
                        },
                        priority=2,
                        depends_on=["structure"],
                        description="Learning rate",
                    ),
                    create_param_group(
                        name="regularization",
                        params={
                            "gamma": {"type": "float", "low": 0.0, "high": 7.0},
                            "reg_alpha": {"type": "float", "low": 1e-6, "high": 20.0, "log": True},
                            "reg_lambda": {"type": "float", "low": 0.05, "high": 20.0},
                        },
                        priority=3,
                        depends_on=["structure"],
                        description="Regularization strength",
                    ),
                    create_param_group(
                        name="sampling",
                        params={
                            # Allow a broader range of subsampling so HPO can
                            # find robust yet expressive models.
                            "subsample": {"type": "float", "low": 0.4, "high": 0.95},
                            "colsample_bytree": {"type": "float", "low": 0.4, "high": 0.95},
                        },
                        priority=4,
                        depends_on=["regularization"],
                        description="Row/feature subsampling ratios",
                    ),
                ]

                base_model_for_hpo = xgb.XGBClassifier(**base_params)

                hpo_cv_folds = int(config.get("risk_hpo_cv_folds", 3))
                hpo_rounds = int(config.get("risk_hpo_rounds", 1))
                hpo_final_trials = int(config.get("risk_hpo_final_trials", 20))
                hpo_enable_final = bool(config.get("risk_hpo_enable_final_refinement", False))

                scoring_metric = str(config.get("risk_hpo_scoring_metric", "neg_log_loss"))

                # Optional HPO subsampling: evaluate candidate configs on a
                # representative subset of train/validation samples to control
                # runtime without changing the final classifier fit.
                X_train_hpo = X_train
                y_train_hpo = y_train
                X_val_hpo = X_val
                y_val_hpo = y_val

                hpo_enable_subsampling = bool(config.get("risk_hpo_enable_subsampling", True))
                hpo_max_samples = int(config.get("risk_hpo_max_samples", 8000))
                if hpo_enable_subsampling and hpo_max_samples > 0:
                    try:
                        rng_hpo = np.random.RandomState(
                            int(config.get("risk_hpo_subsample_random_state", 42))
                        )
                    except Exception:
                        rng_hpo = np.random.RandomState(42)

                    def _subsample_for_hpo(X_in: pd.DataFrame, y_in: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
                        if len(X_in) <= hpo_max_samples:
                            return X_in, y_in
                        idx = rng_hpo.choice(len(X_in), size=hpo_max_samples, replace=False)
                        return X_in.iloc[idx], y_in[idx]

                    X_train_hpo, y_train_hpo = _subsample_for_hpo(X_train, y_train)
                    X_val_hpo, y_val_hpo = _subsample_for_hpo(X_val, y_val)

                # Optional: apply per-regime caps so HPO does not over-focus on a
                # single dominant regime when exploring the search space.
                hpo_enable_balanced_sampling = bool(
                    config.get("risk_hpo_enable_balanced_sampling", True)
                )
                hpo_max_per_regime = int(config.get("risk_hpo_balanced_max_per_regime", 0))
                if hpo_enable_balanced_sampling and hpo_max_per_regime > 0:
                    try:
                        try:
                            rng_bal = np.random.RandomState(
                                int(config.get("risk_hpo_balanced_random_state", 44))
                            )
                        except Exception:
                            rng_bal = np.random.RandomState(44)

                        def _cap_per_regime(X_in, y_in):
                            if y_in is None or len(y_in) == 0:
                                return X_in, y_in
                            y_arr = np.asarray(y_in)
                            classes, counts = np.unique(y_arr, return_counts=True)
                            keep_indices: List[int] = []
                            for cls, cnt in zip(classes, counts):
                                cls_idx = np.where(y_arr == cls)[0]
                                if cnt > hpo_max_per_regime:
                                    sel = rng_bal.choice(
                                        cls_idx,
                                        size=hpo_max_per_regime,
                                        replace=False,
                                    )
                                else:
                                    sel = cls_idx
                                if sel.size > 0:
                                    keep_indices.append(sel)

                            if not keep_indices:
                                return X_in, y_in

                            idx_all = np.concatenate(keep_indices)
                            rng_bal.shuffle(idx_all)

                            if hasattr(X_in, "iloc"):
                                X_out = X_in.iloc[idx_all]
                            else:
                                X_out = X_in[idx_all]
                            y_out = y_arr[idx_all]
                            return X_out, y_out

                        X_train_hpo, y_train_hpo = _cap_per_regime(X_train_hpo, y_train_hpo)
                        X_val_hpo, y_val_hpo = _cap_per_regime(X_val_hpo, y_val_hpo)

                        tprint_info(
                            f"⚖️ HPO per-regime cap enabled: max {hpo_max_per_regime} samples "
                            f"per regime (train={len(y_train_hpo)}, "
                            f"val={len(y_val_hpo) if y_val_hpo is not None else 0})"
                        )
                    except Exception as hpo_bal_exc:  # pragma: no cover - defensive
                        tprint_warning(
                            f"HPO balanced sampling failed (non-fatal); using unbalanced HPO data: {hpo_bal_exc}"
                        )

                quadrant_cols = _select_quadrant_feature_cols(feature_cols)

                # Log which quadrant features will actually drive WCoV HPO and
                # pruning, and warn if canonical candidates are missing from the
                # classifier feature set.
                if quadrant_cols:
                    tprint_info(
                        f"🧭 Quadrant features selected for WCoV HPO/pruning: {quadrant_cols}"
                    )
                else:
                    tprint_warning(
                        "No quadrant features found in classifier features; WCoV "
                        "HPO/pruning will have no effect. Ensure RSI, vol_of_vol, "
                        "Parkinson volatility, and autocorrelation features are "
                        "present in the risk feature set."
                    )

                def quadrant_objective(
                    params: Dict[str, Any],
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    model: Optional[Any] = None,
                    cv_folds: int = 5,
                    scoring_metric: str = "neg_log_loss",
                    **kwargs: Any,
                ) -> float:
                    """Pure WCoV-based HPO objective on quadrant features.

                    Classifier validation metrics are deliberately ignored; HPO
                    optimizes only for between/within separation on RSI, vol-of-
                    vol, Parkinson volatility, and autocorr features.
                    """
                    try:
                        if X_val is None or y_val is None:
                            return float("-inf")

                        base_model = model if model is not None else xgb.XGBClassifier(**base_params)
                        base_model.set_params(**params)

                        # Ensure early stopping has a validation set when enabled.
                        # XGBoost's sklearn API expects at least one eval_set
                        # whenever early_stopping_rounds > 0. Also silence
                        # per-iteration logging during HPO for cleaner output.
                        if X_val is not None and y_val is not None:
                            base_model.fit(
                                X_train,
                                y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False,
                            )
                        else:
                            base_model.fit(
                                X_train,
                                y_train,
                                verbose=False,
                            )

                        val_probs = base_model.predict_proba(X_val)
                        val_pred = np.argmax(val_probs, axis=1)

                        # Previous classifier-based component (disabled: WCoV-only HPO)
                        # metric_name = (scoring_metric or "neg_log_loss").lower()
                        # if metric_name == "neg_log_loss":
                        #     try:
                        #         clf_score = -log_loss(y_val, val_probs, labels=np.arange(n_regimes))
                        #     except Exception:
                        #         clf_score = 0.0
                        # elif metric_name in ("accuracy", "acc"):
                        #     clf_score = accuracy_score(y_val, val_pred)
                        # else:
                        #     clf_score = accuracy_score(y_val, val_pred)

                        wcov_log_term = 0.0
                        within_spread_penalty = 0.0
                        if quadrant_cols:
                            try:
                                val_df = pd.DataFrame(X_val, columns=feature_cols)
                                quad_df = val_df[quadrant_cols]

                                # Compute per-feature WCoV between/within ratios,
                                # apply a log transform to each so that no single
                                # feature dominates, then average.
                                ratio_cap = float(
                                    config.get("risk_hpo_wcov_ratio_cap", 10.0)
                                )
                                feature_scores: List[float] = []
                                for col in quadrant_cols:
                                    col_df = quad_df[[col]]
                                    between_col = self._calculate_winsorized_cv_between(
                                        val_pred,
                                        col_df,
                                    )
                                    within_col = self._calculate_winsorized_cv_within(
                                        val_pred,
                                        col_df,
                                    )

                                    if not np.isfinite(between_col) or not np.isfinite(within_col):
                                        continue

                                    ratio_col = between_col / (within_col + 1e-8)
                                    if ratio_col <= 0.0:
                                        continue

                                    ratio_capped = min(ratio_col, ratio_cap)
                                    feature_scores.append(float(np.log1p(ratio_capped)))

                                if feature_scores:
                                    wcov_log_term = float(np.mean(feature_scores))
                                else:
                                    wcov_log_term = 0.0

                                # Small heterogeneity penalty based on the
                                # spread of per-regime within-WCoV values
                                # across the quadrant features.
                                try:
                                    unique_regimes = [r for r in np.unique(val_pred) if r >= 0]
                                    per_regime_cvs: List[float] = []
                                    for rid in unique_regimes:
                                        regime_mask = val_pred == rid
                                        regime_data = quad_df[regime_mask]
                                        if regime_data.shape[0] < 2:
                                            continue

                                        feature_cvs = []
                                        for col in quad_df.columns:
                                            col_data = regime_data[col].dropna()
                                            if len(col_data) > 1:
                                                lower_bound = col_data.quantile(0.05)
                                                upper_bound = col_data.quantile(0.95)
                                                col_winsorized = col_data.clip(
                                                    lower=lower_bound,
                                                    upper=upper_bound,
                                                )
                                                cv = col_winsorized.std() / (
                                                    np.abs(col_winsorized.mean()) + 1e-8
                                                )
                                                feature_cvs.append(cv)

                                        if feature_cvs:
                                            per_regime_cvs.append(float(np.mean(feature_cvs)))

                                    if len(per_regime_cvs) >= 2:
                                        spread = float(max(per_regime_cvs) - min(per_regime_cvs))
                                        spread_weight = float(
                                            config.get(
                                                "risk_hpo_within_spread_penalty_weight",
                                                0.2,
                                            )
                                        )
                                        within_spread_penalty = spread_weight * spread
                                except Exception:
                                    within_spread_penalty = 0.0
                            except Exception:
                                wcov_log_term = 0.0
                                within_spread_penalty = 0.0

                        # Predicted-regime balance term: discourage models that
                        # collapse almost all samples into a single regime while
                        # still primarily optimizing for WCoV separation.
                        balance_penalty = 0.0
                        try:
                            if val_pred is not None:
                                counts = np.bincount(val_pred, minlength=n_regimes)
                                total = float(counts.sum()) if counts.sum() > 0 else 0.0
                                if total > 0.0:
                                    p = counts.astype(float) / total
                                    min_pct = float(p.min())
                                    max_pct = float(p.max())

                                    # Default band for HPO: aim for each
                                    # regime to have at least ~3% and at most
                                    # ~75% of samples, unless overridden via
                                    # config.
                                    min_target = float(config.get("risk_hpo_min_regime_pct", 0.05))
                                    max_target = float(config.get("risk_hpo_max_regime_pct", 0.65))

                                    violation_low = max(0.0, min_target - min_pct)
                                    violation_high = max(0.0, max_pct - max_target)
                                    total_violation = violation_low + violation_high

                                    # Additive balance penalty: larger when the
                                    # predicted regime distribution is more
                                    # imbalanced relative to the target band.
                                    balance_strength = float(
                                        config.get("risk_hpo_balance_strength", 12.0)
                                    )
                                    balance_penalty = float(
                                        balance_strength * total_violation
                                    )
                        except Exception as balance_exc:  # pragma: no cover - defensive
                            tprint_warning(
                                f"Quadrant HPO balance term failed (non-fatal): {balance_exc}"
                            )
                            balance_penalty = 0.0

                        lambda_wcov = float(config.get("risk_hpo_wcov_weight", 1.0))
                        # Additive log-style objective: prioritize high WCoV
                        # separation while subtracting balance and
                        # within-spread penalties. Higher values are better.
                        return float(
                            lambda_wcov
                            * (wcov_log_term - balance_penalty - within_spread_penalty)
                        )
                    except Exception as obj_exc:
                        tprint_warning(f"Quadrant-aware HPO objective failed: {obj_exc}")
                        return float("-inf")

                objective_func = (
                    quadrant_objective if enable_quadrant_objective else default_objective_function
                )

                optimizer = HierarchicalParameterOptimizer(
                    param_groups=hpo_param_groups,
                    objective_func=objective_func,
                    cv_folds=hpo_cv_folds,
                    scoring_metric=scoring_metric,
                    direction="maximize",
                    n_rounds=hpo_rounds,
                    enable_final_refinement=hpo_enable_final,
                    final_refinement_trials=hpo_final_trials,
                    cache_dir=None,
                    random_state=42,
                    verbose=bool(config.get("risk_hpo_verbose", False)),
                    use_custom_balanced_score=False,
                )

                X_train_np = X_train_hpo.values if hasattr(X_train_hpo, "values") else X_train_hpo
                y_train_np = y_train_hpo
                X_val_np = X_val_hpo.values if hasattr(X_val_hpo, "values") else X_val_hpo
                y_val_np = y_val_hpo

                hpo_result = optimizer.optimize(
                    X_train=X_train_np,
                    y_train=y_train_np,
                    X_val=X_val_np,
                    y_val=y_val_np,
                    model=base_model_for_hpo,
                    initial_params=base_params,
                )

                best_params.update(hpo_result.best_params or {})
            except Exception as hpo_exc:
                tprint_warning(f"Classifier HPO failed; proceeding with default params: {hpo_exc}")

        enable_regime_weighting = bool(config.get("risk_enable_regime_weighting", True))

        base_sample_weight = np.ones_like(y_train, dtype=float)
        if enable_regime_weighting:
            try:
                class_counts = pd.Series(y_train).value_counts()
                n_classes = len(class_counts)
                total = float(len(y_train)) if len(y_train) > 0 else 1.0
                class_weight = {
                    int(cls): (total / (n_classes * float(cnt)))
                    for cls, cnt in class_counts.items()
                    if cnt > 0
                }
                base_sample_weight = np.array(
                    [class_weight.get(int(lbl), 1.0) for lbl in y_train],
                    dtype=float,
                )

                if teacher_conf_train is not None:
                    lambda_conf = float(config.get("risk_teacher_confidence_weight", 0.5))
                    centered = teacher_conf_train - float(teacher_conf_train.mean())
                    base_sample_weight *= (1.0 + lambda_conf * centered)
            except Exception as weight_exc:
                tprint_warning(
                    f"Regime-aware weighting failed; using uniform weights: {weight_exc}"
                )
                base_sample_weight = np.ones_like(y_train, dtype=float)

        base_sample_weight = np.maximum(base_sample_weight, 0.0)
        if float(base_sample_weight.mean()) > 0.0:
            base_sample_weight = base_sample_weight / float(base_sample_weight.mean())

        # Optional: per-regime caps for the final training dataset to prevent
        # the classifier from being dominated by a single massive regime.
        train_enable_balanced = bool(
            config.get("risk_train_enable_balanced_sampling", False)
        )
        train_max_per_regime = int(config.get("risk_train_balanced_max_per_regime", 0))
        if train_enable_balanced and train_max_per_regime > 0:
            try:
                try:
                    rng_train_bal = np.random.RandomState(
                        int(config.get("risk_train_balanced_random_state", 45))
                    )
                except Exception:
                    rng_train_bal = np.random.RandomState(45)

                y_arr = np.asarray(y_train)
                classes, counts = np.unique(y_arr, return_counts=True)
                keep_indices: List[int] = []
                for cls, cnt in zip(classes, counts):
                    cls_idx = np.where(y_arr == cls)[0]
                    if cnt > train_max_per_regime:
                        sel = rng_train_bal.choice(
                            cls_idx,
                            size=train_max_per_regime,
                            replace=False,
                        )
                    else:
                        sel = cls_idx
                    if sel.size > 0:
                        keep_indices.append(sel)

                if keep_indices:
                    idx_all = np.concatenate(keep_indices)
                    rng_train_bal.shuffle(idx_all)

                    if hasattr(X_train, "iloc"):
                        X_train = X_train.iloc[idx_all]
                    else:
                        X_train = X_train[idx_all]
                    y_train = y_arr[idx_all]
                    base_sample_weight = base_sample_weight[idx_all]
                    if (
                        teacher_conf_train is not None
                        and len(teacher_conf_train) == len(y_arr)
                    ):
                        teacher_conf_train = teacher_conf_train[idx_all]

                    tprint_info(
                        f"⚖️ Training per-regime cap enabled: max {train_max_per_regime} samples "
                        f"per regime (train={len(y_train)})"
                    )
            except Exception as train_bal_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Training balanced sampling failed (non-fatal); using full training set: {train_bal_exc}"
                )

        # Train classifier (optionally with high-confidence filtering)
        X_train_final = X_train
        y_train_final = y_train
        sample_weight_final = base_sample_weight

        if enable_high_conf:
            try:
                temp_model = xgb.XGBClassifier(**best_params)
                temp_model.fit(
                    X_train,
                    y_train,
                    sample_weight=base_sample_weight,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                train_probs = temp_model.predict_proba(X_train)
                max_conf = train_probs.max(axis=1)
                high_conf_mask = max_conf >= high_conf_threshold

                n_high_conf = int(high_conf_mask.sum())
                frac_high_conf = (
                    float(n_high_conf) / float(len(y_train)) if len(y_train) > 0 else 0.0
                )

                if n_high_conf > 0:
                    y_train_high_conf = y_train[high_conf_mask]
                    per_regime_counts = (
                        pd.Series(y_train_high_conf).value_counts().to_dict()
                    )
                else:
                    per_regime_counts = {}

                min_required = int(len(y_train) * min_high_conf_fraction)
                regime_ok = (
                    all(
                        count >= min_high_conf_per_regime
                        for count in per_regime_counts.values()
                    )
                    if per_regime_counts
                    else False
                )

                if n_high_conf >= max(min_required, n_regimes) and regime_ok:
                    X_train_final = X_train.iloc[high_conf_mask]
                    y_train_final = y_train_high_conf
                    sample_weight_final = base_sample_weight[high_conf_mask]
                    tprint_info(
                        f"🧪 High-confidence training enabled: using {n_high_conf} / {len(y_train)} "
                        f"samples ({frac_high_conf:.1%}), min_per_regime>={min_high_conf_per_regime}"
                    )
                else:
                    tprint_info(
                        "🧪 High-confidence filter skipped (insufficient high-confidence points or "
                        "per-regime counts too small); training on full dataset instead."
                    )
            except Exception as high_conf_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"High-confidence training path failed; falling back to full dataset: {high_conf_exc}"
                )
                X_train_final = X_train
                y_train_final = y_train
                sample_weight_final = base_sample_weight

        # Final classifier trained on either full or high-confidence subset (baseline model)
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_final,
            y_train_final,
            sample_weight=sample_weight_final,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        regime_probs = model.predict_proba(X_full)
        y_val_pred = model.predict(X_val)
        y_val_probs = model.predict_proba(X_val)

        def _apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
            eps = 1e-12
            clipped = np.clip(probs, eps, 1.0)
            scaled = np.power(clipped, 1.0 / max(float(temperature), eps))
            row_sums = scaled.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0.0] = 1.0
            return scaled / row_sums

        def _fit_temperature_scaling(
            y_true: np.ndarray,
            probs: np.ndarray,
        ) -> Tuple[float, np.ndarray, float, float]:
            base_loss = float(
                log_loss(y_true, probs, labels=np.arange(n_regimes))
            )
            best_T = 1.0
            best_loss = base_loss
            for T in np.linspace(0.5, 3.0, 11):
                calibrated = _apply_temperature_scaling(probs, float(T))
                loss_T = float(
                    log_loss(y_true, calibrated, labels=np.arange(n_regimes))
                )
                if loss_T < best_loss:
                    best_loss = loss_T
                    best_T = float(T)
            calibrated_best = _apply_temperature_scaling(probs, best_T)
            return best_T, calibrated_best, base_loss, best_loss

        temp_scaling_enabled = bool(
            config.get("risk_enable_temperature_scaling", True)
        )
        temperature = 1.0
        val_log_loss_uncalibrated = None
        val_log_loss_calibrated = None

        if temp_scaling_enabled and y_val_probs is not None:
            try:
                (
                    temperature,
                    y_val_probs_cal,
                    base_loss,
                    best_loss,
                ) = _fit_temperature_scaling(y_val, y_val_probs)
                val_log_loss_uncalibrated = base_loss
                val_log_loss_calibrated = best_loss
                regime_probs = _apply_temperature_scaling(regime_probs, temperature)
                y_val_probs = y_val_probs_cal
            except Exception as temp_exc:
                tprint_warning(
                    f"Temperature scaling failed (non-fatal): {temp_exc}"
                )
                temperature = 1.0

        # Quadrant WCoV quality metrics (teacher vs predicted regimes)
        regime_pred_all = np.argmax(regime_probs, axis=1)
        quadrant_quality: Dict[str, Any] = {}
        if quadrant_cols:
            try:
                full_df = pd.DataFrame(X_full, columns=feature_cols)
                quad_full = full_df[quadrant_cols]

                teacher_between = self._calculate_winsorized_cv_between(y, quad_full)
                teacher_within = self._calculate_winsorized_cv_within(y, quad_full)
                pred_between = self._calculate_winsorized_cv_between(regime_pred_all, quad_full)
                pred_within = self._calculate_winsorized_cv_within(regime_pred_all, quad_full)

                quadrant_quality = {
                    "quadrant_features": list(quadrant_cols),
                    "teacher_cv_between": float(teacher_between),
                    "teacher_cv_within": float(teacher_within),
                    "teacher_cv_ratio": float(teacher_between / (teacher_within + 1e-8)),
                    "pred_cv_between": float(pred_between),
                    "pred_cv_within": float(pred_within),
                    "pred_cv_ratio": float(pred_between / (pred_within + 1e-8)),
                }
            except Exception as quad_exc:
                tprint_warning(
                    f"Quadrant WCoV quality computation failed (non-fatal): {quad_exc}"
                )
                quadrant_quality = {}

        val_accuracy = accuracy_score(y_val, y_val_pred)
        if val_log_loss_calibrated is not None:
            val_log_loss = float(val_log_loss_calibrated)
        else:
            val_log_loss = float(log_loss(y_val, y_val_probs))

        training_metrics = {
            'val_accuracy': float(val_accuracy),
            'val_log_loss': float(val_log_loss),
            'n_regimes': n_regimes,
            'feature_names': list(X_full.columns),
            'scaler': scaler,
            'monotone_constraints': monotone_constraints,
            'n_features': len(X_full.columns),
        }

        if temp_scaling_enabled:
            training_metrics['temperature_scaling_enabled'] = True
            training_metrics['temperature'] = float(temperature)
            if val_log_loss_uncalibrated is not None:
                training_metrics['val_log_loss_uncalibrated'] = float(
                    val_log_loss_uncalibrated
                )
                training_metrics['val_log_loss_calibrated'] = float(val_log_loss)

        if quadrant_quality:
            training_metrics['quadrant_quality'] = quadrant_quality

            # Persist a dedicated quadrant WCoV quality summary reflecting HPO goals.
            try:
                symbol_q = str(config.get("symbol", ""))
                exchange_q = str(config.get("exchange", ""))
                regime_tf_q = str(
                    config.get("regime_timeframe", config.get("timeframe", "1h"))
                )
                lambda_wcov = float(config.get("risk_hpo_wcov_weight", 1.0))

                quality_row = {
                    "symbol": symbol_q,
                    "exchange": exchange_q,
                    "timeframe": regime_tf_q,
                    "risk_hpo_wcov_weight": lambda_wcov,
                    "quadrant_teacher_cv_ratio": quadrant_quality["teacher_cv_ratio"],
                    "quadrant_teacher_cv_between": quadrant_quality["teacher_cv_between"],
                    "quadrant_teacher_cv_within": quadrant_quality["teacher_cv_within"],
                    "quadrant_pred_cv_ratio": quadrant_quality["pred_cv_ratio"],
                    "quadrant_pred_cv_between": quadrant_quality["pred_cv_between"],
                    "quadrant_pred_cv_within": quadrant_quality["pred_cv_within"],
                }

                try:
                    quality_row["quadrant_features"] = json.dumps(
                        quadrant_quality["quadrant_features"]
                    )
                except Exception:
                    quality_row["quadrant_features"] = str(
                        quadrant_quality.get("quadrant_features", [])
                    )

                ts_q = datetime.now().strftime("%Y%m%d_%H%M%S")
                quadrant_quality_path = (
                    f"outcomes/ml_risk_quadrant_quality_"
                    f"{symbol_q or 'UNKNOWN'}_{regime_tf_q}_{ts_q}.csv"
                )
                pd.DataFrame([quality_row]).to_csv(quadrant_quality_path, index=False)
                tprint_info(
                    f"💾 Saved quadrant WCoV quality summary: {quadrant_quality_path}"
                )
            except Exception as quad_csv_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Failed to persist quadrant WCoV quality summary (non-fatal): {quad_csv_exc}"
                )

        # Per-regime risk profile diagnostics: summarize mean values of key
        # volatility/tail/jump features for each predicted regime.
        try:
            regime_ids = sorted(np.unique(regime_pred_all))
            total_samples = len(regime_pred_all)

            # Focus on a compact set of core risk features. Only keep those
            # that are actually present in the numeric DataFrame.
            profile_feature_candidates = [
                "parkinson_vol",
                "vol_realized_20",
                "cvar_5pct",
                "cvar_10pct",
                "downside_deviation_20",
                "vol_acceleration",
                "vol_acceleration_ewma2",
                "vol_acceleration_ewma6",
                "vol_expansion_5",
                "vol_expansion_5_ewma2",
                "jump_frequency_20",
                "jump_frequency_20_ewma2",
                "jump_frequency_20_ewma6",
                "drawdown_underwater_frac_20",
            ]

            available_profile_feats = [
                f for f in profile_feature_candidates if f in numeric_df.columns
            ]

            profile_records: List[Dict[str, Any]] = []
            if available_profile_feats and total_samples > 0:
                # Use positional boolean masks to align regimes with rows in
                # numeric_df. Both are defined on the same cleaned dataset.
                for rid in regime_ids:
                    mask = regime_pred_all == rid
                    n_k = int(mask.sum())
                    if n_k == 0:
                        continue

                    row: Dict[str, Any] = {
                        "regime_id": int(rid),
                        "n_samples": n_k,
                        "share": float(n_k / float(total_samples)),
                    }

                    regime_numeric = numeric_df.loc[mask, available_profile_feats]
                    for feat in available_profile_feats:
                        row[f"{feat}_mean"] = float(regime_numeric[feat].mean())

                    profile_records.append(row)

            if profile_records:
                profile_df = pd.DataFrame(profile_records)

                symbol_p = str(config.get("symbol", ""))
                exchange_p = str(config.get("exchange", ""))
                regime_tf_p = str(
                    config.get("regime_timeframe", config.get("timeframe", "1h"))
                )
                ts_prof = datetime.now().strftime("%Y%m%d_%H%M%S")

                profile_path = (
                    f"outcomes/ml_risk_regime_risk_profile_"
                    f"{symbol_p or 'UNKNOWN'}_{regime_tf_p}_{ts_prof}.csv"
                )
                profile_df.to_csv(profile_path, index=False)
                tprint_info(f"💾 Saved per-regime risk profile summary: {profile_path}")
        except Exception as profile_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Per-regime risk profile diagnostics failed (non-fatal): {profile_exc}"
            )

        tprint_info("🔍 Calculating comprehensive feature importance (weight, gain, cover)...")

        importance_data = self._calculate_comprehensive_feature_importance(
            model=model,
            X=X_full,
            y=y,
            feature_names=list(X_full.columns)
        )

        training_metrics['feature_importance_detailed'] = {
            'global': importance_data['global'].to_dict('records'),
            'per_regime': {
                regime_id: df.to_dict('records')
                for regime_id, df in importance_data['per_regime'].items()
            }
        }

        global_imp = importance_data['global'].head(15)
        tprint_success("📊 Top 15 Global Feature Importance:")
        for idx, row in global_imp.iterrows():
            tprint_info(
                f"  {row['feature'][:40]:40s} | "
                f"Weight: {row['weight_norm']:.3f} | "
                f"Gain: {row['gain_norm']:.3f} | "
                f"Cover: {row['cover_norm']:.3f} | "
                f"Combined: {row['combined_score']:.3f}"
            )

        tprint_success("📊 Per-Regime Feature Importance (Top 10 per regime):")
        for regime_id, regime_df in importance_data['per_regime'].items():
            tprint_info(f"\n  === Regime {regime_id} ===")
            top_regime_features = regime_df.head(10)
            for idx, row in top_regime_features.iterrows():
                tprint_info(
                    f"    {row['feature'][:35]:35s} | "
                    f"MeanSep: {row['mean_separation']:6.2f} | "
                    f"CVRatio: {row['cv_ratio']:6.2f} | "
                    f"Gain: {row['global_gain']:.3f} | "
                    f"RegImp: {row['regime_importance']:7.2f}"
                )

        tprint_success(
            f"✅ XGBoost Classifier trained (baseline):\n"
            f"   Val Accuracy={val_accuracy:.3f}, Val LogLoss={val_log_loss:.4f}\n"
            f"   Best Iteration={model.best_iteration}, Features={len(X_full.columns)}"
        )

        report = classification_report(
            y_val, y_val_pred,
            target_names=[f'Regime_{i}' for i in range(n_regimes)],
            output_dict=True,
            zero_division=0
        )
        training_metrics['classification_report'] = report

        regime_calibration = {}
        try:
            for regime_id in range(n_regimes):
                mask_regime = y_val == regime_id
                if mask_regime.sum() == 0:
                    continue
                probs_k = y_val_probs[:, regime_id]
                mean_pred = float(np.mean(probs_k[mask_regime]))
                empirical_freq = float(mask_regime.mean())
                regime_calibration[int(regime_id)] = {
                    'mean_predicted': mean_pred,
                    'empirical_freq': empirical_freq,
                }
        except Exception as calib_exc:
            tprint_warning(f"Regime calibration summary failed: {calib_exc}")

        if regime_calibration:
            training_metrics['regime_calibration'] = regime_calibration

        for regime_id in range(n_regimes):
            regime_report = report.get(f'Regime_{regime_id}', {})
            precision = regime_report.get('precision', 0)
            recall = regime_report.get('recall', 0)
            f1 = regime_report.get('f1-score', 0)
            tprint_info(
                f"  Regime {regime_id}: Precision={precision:.3f}, "
                f"Recall={recall:.3f}, F1={f1:.3f}"
            )

        baseline_model = model
        baseline_regime_probs = regime_probs
        baseline_training_metrics = dict(training_metrics)

        noise_feature_table = None
        noisy_features = []
        min_keep = max(15, int(len(feature_cols) * 0.3))

        try:
            global_df = importance_data['global'].copy()
            eps = 1e-8

            diag_enable_subsampling = bool(
                config.get("risk_noise_diag_enable_subsampling", True)
            )
            diag_frac = float(config.get("risk_noise_diag_subsample_frac", 0.66))
            n_samples_diag = len(y)
            diag_mask = np.ones(n_samples_diag, dtype=bool)
            if (
                diag_enable_subsampling
                and 0.0 < diag_frac < 1.0
                and n_samples_diag > 0
            ):
                try:
                    rng_diag_samples = np.random.RandomState(
                        int(config.get("risk_noise_diag_random_state", 41))
                    )
                except Exception:
                    rng_diag_samples = np.random.RandomState(41)

                sample_size = max(1, int(n_samples_diag * diag_frac))
                idx_diag = rng_diag_samples.choice(
                    n_samples_diag, size=sample_size, replace=False
                )
                diag_mask = np.zeros(n_samples_diag, dtype=bool)
                diag_mask[idx_diag] = True
                tprint_info(
                    f"  Noise diagnostics subsampling: using {sample_size} / {n_samples_diag} samples"
                )

            # Restrict expensive diagnostics to a subset of features to keep
            # runtime manageable on large feature sets.
            max_diag_feats = int(config.get("risk_noise_max_diag_features", 0))
            sampling_mode = str(config.get("risk_noise_diag_sampling_mode", "top_importance")).lower()

            diag_feature_candidates: List[str] = list(feature_cols)
            if max_diag_feats > 0 and len(diag_feature_candidates) > max_diag_feats:
                try:
                    if sampling_mode == "random":
                        rng_diag = np.random.RandomState(int(config.get("risk_noise_diag_random_state", 41)))
                        diag_feature_candidates = list(
                            rng_diag.choice(diag_feature_candidates, size=max_diag_feats, replace=False)
                        )
                    else:
                        # Default: top-N by global importance (gain_norm if available)
                        if "gain_norm" in global_df.columns:
                            ordered = (
                                global_df.sort_values("gain_norm", ascending=False)["feature"].astype(str).tolist()
                            )
                        else:
                            ordered = global_df["feature"].astype(str).tolist()
                        # Preserve original order within the top-N pool
                        top_set = set(ordered[:max_diag_feats])
                        diag_feature_candidates = [f for f in feature_cols if f in top_set]
                except Exception:
                    # Fallback: just truncate the original feature list
                    diag_feature_candidates = list(feature_cols)[:max_diag_feats]

            distinctiveness_records = []
            regime_ids = sorted(np.unique(y))

            for feat in diag_feature_candidates:
                if feat not in numeric_df.columns:
                    continue
                series = numeric_df[feat]
                if series.isna().all():
                    continue

                regime_means = []
                within_covs = []

                for rid in regime_ids:
                    rid_mask = (y == rid) & diag_mask
                    vals = series[rid_mask].dropna()
                    if len(vals) < 10:
                        continue
                    _, wcv = self._calculate_winsorized_cv(vals)
                    within_covs.append(wcv)
                    regime_means.append(vals.mean())

                if not within_covs or len(regime_means) < 2:
                    continue

                overall_mean = float(np.mean(regime_means))
                between_std = float(np.std(regime_means))
                if abs(overall_mean) > eps:
                    between_cov = abs(between_std / (overall_mean + eps))
                else:
                    between_cov = 0.0

                within_cov = float(np.mean(within_covs))
                if within_cov > 0.0:
                    distinctiveness = float(between_cov / (within_cov + eps))
                else:
                    distinctiveness = 0.0

                distinctiveness_records.append(
                    {
                        'feature': feat,
                        'between_cov': between_cov,
                        'within_cov': within_cov,
                        'wcov_distinctiveness': distinctiveness,
                    }
                )

            if distinctiveness_records:
                distinctiveness_df = pd.DataFrame(distinctiveness_records)
                noise_df = global_df.merge(distinctiveness_df, on='feature', how='left')
            else:
                noise_df = global_df.copy()
                noise_df['between_cov'] = np.nan
                noise_df['within_cov'] = np.nan
                noise_df['wcov_distinctiveness'] = np.nan

            conf = baseline_regime_probs.max(axis=1)
            high_conf_thr = float(config.get("risk_diagnostics_high_conf_threshold", 0.8))
            low_conf_thr = float(config.get("risk_diagnostics_low_conf_threshold", 0.5))

            high_mask = (conf >= high_conf_thr) & diag_mask
            low_mask = (conf <= low_conf_thr) & diag_mask

            var_ratios = []
            for feat in feature_cols:
                if feat not in numeric_df.columns:
                    continue
                vals = numeric_df[feat].values
                if len(vals) != len(conf):
                    continue

                hi_vals = vals[high_mask]
                lo_vals = vals[low_mask]

                if len(hi_vals) < 10 or len(lo_vals) < 10:
                    var_ratios.append((feat, np.nan, np.nan, np.nan))
                    continue

                sigma_hi = float(np.std(hi_vals))
                sigma_lo = float(np.std(lo_vals))
                var_ratio = (sigma_lo + eps) / (sigma_hi + eps)
                var_ratios.append((feat, sigma_hi, sigma_lo, var_ratio))

            var_df = pd.DataFrame(
                var_ratios,
                columns=['feature', 'sigma_high_conf', 'sigma_low_conf', 'var_ratio_confidence'],
            )

            noise_df = noise_df.merge(var_df, on='feature', how='left')

            correct_mask = y_val_pred == y_val
            wrong_mask = ~correct_mask

            misclass_records = []
            for feat in feature_cols:
                if feat not in X_val_raw.columns:
                    continue
                vals = X_val_raw[feat].values

                corr_vals = vals[correct_mask]
                wrong_vals = vals[wrong_mask]

                if len(corr_vals) < 10 or len(wrong_vals) < 10:
                    misclass_records.append(
                        (feat, np.nan, np.nan, np.nan, np.nan)
                    )
                    continue

                mu_corr = float(np.mean(corr_vals))
                mu_wrong = float(np.mean(wrong_vals))
                sigma_corr = float(np.std(corr_vals))
                sigma_wrong = float(np.std(wrong_vals))
                pooled = np.sqrt(
                    (sigma_corr ** 2 + sigma_wrong ** 2) / 2.0
                )
                if pooled > 0.0:
                    effect_size = abs(mu_wrong - mu_corr) / (pooled + eps)
                else:
                    effect_size = 0.0

                misclass_records.append(
                    (feat, mu_corr, mu_wrong, pooled, effect_size)
                )

            misclass_df = pd.DataFrame(
                misclass_records,
                columns=[
                    'feature',
                    'mean_correct',
                    'mean_wrong',
                    'pooled_std',
                    'misclass_effect_size',
                ],
            )

            noise_df = noise_df.merge(misclass_df, on='feature', how='left')

            noise_df['gain_norm'] = noise_df['gain_norm'].fillna(0.0)
            noise_df['wcov_distinctiveness'] = noise_df['wcov_distinctiveness'].fillna(0.0)
            noise_df['var_ratio_confidence'] = noise_df['var_ratio_confidence'].fillna(0.0)
            noise_df['misclass_effect_size'] = noise_df['misclass_effect_size'].fillna(0.0)

            try:
                try:
                    baseline_pred_full = np.argmax(baseline_regime_probs, axis=1)
                except Exception:
                    baseline_pred_full = None

                harm_diag_df = None
                if baseline_pred_full is not None and len(baseline_pred_full) == len(y):
                    wcov_tmp_df = numeric_df.copy()
                    wcov_tmp_df["risk_regime_pred_tmp"] = baseline_pred_full
                    wcov_tmp_df["risk_regime_teacher_tmp"] = y
                    harm_diag_df = self._compute_feature_wcov_diagnostics(
                        df=wcov_tmp_df,
                        regime_col_pred="risk_regime_pred_tmp",
                        regime_col_teacher="risk_regime_teacher_tmp",
                        top_k=None,
                    )

                if harm_diag_df is not None and not harm_diag_df.empty:
                    harm_df = harm_diag_df[["feature", "wcov_ratio_improvement"]]
                    noise_df = noise_df.merge(harm_df, on="feature", how="left")

                if "wcov_ratio_improvement" not in noise_df.columns:
                    noise_df["wcov_ratio_improvement"] = 0.0
                noise_df["wcov_ratio_improvement"] = noise_df["wcov_ratio_improvement"].fillna(0.0)

                gains_all = noise_df["gain_norm"].astype(float).clip(lower=0.0)
                wcov_improvement = noise_df["wcov_ratio_improvement"].astype(float)
                harm_raw = np.maximum(0.0, -wcov_improvement) * gains_all
                noise_df["harm_score"] = harm_raw

                low_imp_q = float(config.get("risk_low_importance_quantile", 0.05))
                if 0.0 < low_imp_q < 1.0:
                    positive_gains = gains_all[gains_all > 0.0]
                    if positive_gains.size >= 10:
                        thr_low = float(np.quantile(positive_gains, low_imp_q))
                        noise_df["low_importance_flag"] = gains_all <= thr_low
                    else:
                        noise_df["low_importance_flag"] = False
                else:
                    noise_df["low_importance_flag"] = False
            except Exception:
                noise_df["harm_score"] = 0.0
                noise_df["low_importance_flag"] = False

            # ------------------------------------------------------------------
            # Collapse-driver and clarity-enhancer diagnostics
            #
            # Idea: identify features that strongly drive assignment into a
            # single dominant predicted regime (collapse-drivers) vs features
            # that are structurally distinctive across teacher regimes without
            # just feeding the dominant state (clarity-enhancers).
            # ------------------------------------------------------------------
            clarity_protect: List[str] = []
            try:
                # Use hard predictions on the full dataset to detect the
                # dominant regime under the student (XGB) classifier.
                baseline_pred = np.argmax(baseline_regime_probs, axis=1)
                dom_counts = np.bincount(baseline_pred, minlength=n_regimes)
                dominant_regime = int(dom_counts.argmax())

                dom_mask = baseline_pred == dominant_regime
                other_mask = ~dom_mask

                collapse_records = []
                for feat in feature_cols:
                    if feat not in numeric_df.columns:
                        continue

                    vals = numeric_df[feat].values
                    if len(vals) != len(baseline_pred):
                        continue

                    valid_dom = dom_mask & np.isfinite(vals)
                    valid_other = other_mask & np.isfinite(vals)
                    dom_vals = vals[valid_dom]
                    other_vals = vals[valid_other]

                    if len(dom_vals) < 10 or len(other_vals) < 10:
                        effect_dom = 0.0
                    else:
                        mu_dom = float(np.mean(dom_vals))
                        mu_other = float(np.mean(other_vals))
                        sigma_dom = float(np.std(dom_vals))
                        sigma_other = float(np.std(other_vals))
                        pooled = np.sqrt((sigma_dom ** 2 + sigma_other ** 2) / 2.0)
                        if pooled > 0.0:
                            effect_dom = float(abs(mu_dom - mu_other) / (pooled + eps))
                        else:
                            effect_dom = 0.0

                    collapse_records.append((feat, effect_dom))

                if collapse_records:
                    collapse_df = pd.DataFrame(
                        collapse_records,
                        columns=["feature", "collapse_effect_size"],
                    )
                    noise_df = noise_df.merge(collapse_df, on="feature", how="left")
                else:
                    noise_df["collapse_effect_size"] = 0.0

                # Derive collapse-driver and clarity-enhancer scores using
                # teacher distinctiveness (wcov_distinctiveness) and the
                # collapse effect size.
                wcov_vals = noise_df["wcov_distinctiveness"].values.astype(float)
                gain_vals = noise_df["gain_norm"].values.astype(float)
                collapse_vals = (
                    noise_df["collapse_effect_size"].fillna(0.0).values.astype(float)
                )

                wcov_max = float(np.nanmax(wcov_vals)) if np.isfinite(np.nanmax(wcov_vals)) and np.nanmax(wcov_vals) > 0.0 else 0.0
                collapse_max = float(np.nanmax(collapse_vals)) if np.isfinite(np.nanmax(collapse_vals)) and np.nanmax(collapse_vals) > 0.0 else 0.0

                if collapse_max > 0.0:
                    collapse_norm = collapse_vals / (collapse_max + eps)
                else:
                    collapse_norm = np.zeros_like(collapse_vals)

                if wcov_max > 0.0:
                    teacher_norm = wcov_vals / (wcov_max + eps)
                else:
                    teacher_norm = np.zeros_like(wcov_vals)

                # High collapse-driver score: important for XGB, low teacher
                # distinctiveness, and strongly associated with the dominant
                # regime.
                teacher_lowness = 1.0 / (1.0 + wcov_vals)
                collapse_driver_score = gain_vals * teacher_lowness * collapse_norm
                noise_df["collapse_driver_score"] = collapse_driver_score

                # High clarity score: structurally distinctive across teacher
                # regimes but not primarily a collapse-driver.
                clarity_score = teacher_norm * (1.0 - collapse_norm)
                noise_df["clarity_enhancer_score"] = clarity_score

                top_k = int(config.get("risk_noise_protect_top_clarity", 12))
                if top_k > 0 and len(noise_df) > 0:
                    order = np.argsort(-clarity_score)
                    top_idx = order[: min(top_k, len(order))]
                    clarity_protect = (
                        noise_df["feature"].iloc[top_idx].dropna().astype(str).tolist()
                    )
            except Exception as clarity_exc:
                tprint_warning(
                    f"Collapse/clarity diagnostics failed (non-fatal); continuing without them: {clarity_exc}"
                )
                if "collapse_driver_score" not in noise_df.columns:
                    noise_df["collapse_driver_score"] = 0.0
                if "clarity_enhancer_score" not in noise_df.columns:
                    noise_df["clarity_enhancer_score"] = 0.0
                clarity_protect = []

            # Safety guard: always keep at least 15 features and ~30% of originals
            min_keep = max(15, int(len(feature_cols) * 0.3))

            # Base thresholds (slightly more permissive defaults so we naturally
            # surface candidates on smaller samples)
            base_gain_min = float(config.get("risk_noise_gain_min", 0.007))
            base_wcov_max = float(config.get("risk_noise_wcov_max", 1.1))
            base_var_ratio_min = float(config.get("risk_noise_var_ratio_min", 1.7))
            base_eff_size_min = float(config.get("risk_noise_effect_size_min", 0.45))

            noise_hpo_enable = bool(config.get("risk_noise_hpo_enable", True))
            noise_hpo_max_trials = int(config.get("risk_noise_hpo_max_trials", 6))
            if noise_hpo_enable and noise_hpo_max_trials < 2:
                noise_hpo_max_trials = 2

            def _compute_flags(gain_min: float, wcov_max: float, var_ratio_min: float, eff_size_min: float):
                flags_importance = noise_df['gain_norm'] >= gain_min
                flags_wcov = noise_df['wcov_distinctiveness'] <= wcov_max
                flags_conf = noise_df['var_ratio_confidence'] >= var_ratio_min
                flags_misclass = noise_df['misclass_effect_size'] >= eff_size_min
                base_noisy = (
                    flags_importance
                    & (
                        (flags_wcov & flags_conf)
                        | (flags_wcov & flags_misclass)
                    )
                )

                harm_series = noise_df['harm_score'] if 'harm_score' in noise_df.columns else None
                if harm_series is not None:
                    harm_thr = float(config.get("risk_noise_harm_score_min", 0.0))
                    if harm_thr > 0.0:
                        flags_harm = harm_series >= harm_thr
                    else:
                        flags_harm = harm_series > 0.0
                    base_noisy = base_noisy | flags_harm

                if 'low_importance_flag' in noise_df.columns:
                    low_imp_flags = noise_df['low_importance_flag'].astype(bool)
                    base_noisy = base_noisy | low_imp_flags

                is_noisy = base_noisy
                noisy_list = noise_df.loc[is_noisy, 'feature'].dropna().tolist()

                # Never treat quadrant-defining features (RSI, vol_of_vol,
                # Parkinson volatility, autocorr) as noisy, even if their
                # diagnostics would flag them. This ensures the core axes of
                # the quadrant system remain present for both HPO and pruning.
                protect = set()
                if 'quadrant_cols' in locals() and quadrant_cols:
                    protect.update(quadrant_cols)
                if 'clarity_protect' in locals() and clarity_protect:
                    protect.update(clarity_protect)
                if protect:
                    noisy_list = [f for f in noisy_list if f not in protect]

                kept_list = [f for f in feature_cols if f not in noisy_list]
                return is_noisy, noisy_list, kept_list

            def _structural_score(is_noisy_mask: pd.Series, noisy_list, kept_list) -> float:
                """Score a candidate purely on structure; min_keep is enforced later.

                We allow candidates that would keep < min_keep features here so that
                we can later cap the number of removed features to satisfy min_keep
                while still pruning the worst offenders.
                """
                if len(kept_list) == len(feature_cols) or not noisy_list:
                    return float('-inf')

                subset = noise_df.loc[is_noisy_mask]
                if subset.empty:
                    return float('-inf')

                wcov_penalty = np.maximum(0.0, 1.5 - subset['wcov_distinctiveness'].values)
                raw = (
                    subset['gain_norm'].values
                    * subset['var_ratio_confidence'].values
                    * subset['misclass_effect_size'].values
                    * wcov_penalty
                )

                n_noisy = len(noisy_list)
                total = len(feature_cols)
                frac_noisy = n_noisy / float(total) if total > 0 else 0.0

                coverage_weight = 1.0
                if frac_noisy <= 0.0:
                    coverage_weight = 0.0
                elif frac_noisy < 0.1:
                    coverage_weight = 0.5
                elif frac_noisy < 0.4:
                    coverage_weight = 1.0 + frac_noisy
                elif frac_noisy <= 0.6:
                    coverage_weight = 1.4
                else:
                    coverage_weight = 0.8

                base_score = float(raw.mean()) if len(raw) > 0 else float('-inf')
                if not np.isfinite(base_score) or coverage_weight <= 0.0:
                    return float('-inf')
                return base_score * coverage_weight

            # Start from base thresholds
            best_gain_min = base_gain_min
            best_wcov_max = base_wcov_max
            best_var_ratio_min = base_var_ratio_min
            best_eff_size_min = base_eff_size_min

            best_is_noisy, best_noisy_features, best_kept_features = _compute_flags(
                best_gain_min, best_wcov_max, best_var_ratio_min, best_eff_size_min
            )
            best_score = _structural_score(best_is_noisy, best_noisy_features, best_kept_features)

            # Optional lightweight HPO over noise thresholds (structure-based, no extra model fits)
            if noise_hpo_enable and noise_hpo_max_trials > 1:
                rng = np.random.RandomState(int(config.get("risk_noise_hpo_random_state", 42)))

                gain_min_low = max(0.0005, base_gain_min * 0.2)
                gain_min_high = max(gain_min_low * 1.1, base_gain_min * 3.0)
                wcov_max_low = max(0.5, base_wcov_max * 0.5)
                wcov_max_high = max(wcov_max_low * 1.1, base_wcov_max * 2.0)
                var_ratio_min_low = max(1.0, base_var_ratio_min * 0.3)
                var_ratio_min_high = max(var_ratio_min_low * 1.2, base_var_ratio_min * 2.0)
                eff_size_min_low = max(0.1, base_eff_size_min * 0.5)
                eff_size_min_high = max(eff_size_min_low * 1.2, base_eff_size_min * 2.0)

                for _ in range(noise_hpo_max_trials - 1):
                    cand_gain = float(rng.uniform(gain_min_low, gain_min_high))
                    cand_wcov = float(rng.uniform(wcov_max_low, wcov_max_high))
                    cand_var_ratio = float(rng.uniform(var_ratio_min_low, var_ratio_min_high))
                    cand_eff_size = float(rng.uniform(eff_size_min_low, eff_size_min_high))

                    cand_is_noisy, cand_noisy_features, cand_kept_features = _compute_flags(
                        cand_gain, cand_wcov, cand_var_ratio, cand_eff_size
                    )
                    cand_score = _structural_score(
                        cand_is_noisy, cand_noisy_features, cand_kept_features
                    )

                    if cand_score > best_score:
                        best_score = cand_score
                        best_gain_min = cand_gain
                        best_wcov_max = cand_wcov
                        best_var_ratio_min = cand_var_ratio
                        best_eff_size_min = cand_eff_size
                        best_is_noisy = cand_is_noisy
                        best_noisy_features = cand_noisy_features
                        best_kept_features = cand_kept_features

                if best_score == float('-inf'):
                    best_noisy_features = []
                    best_kept_features = feature_cols

                if best_noisy_features:
                    tprint_info(
                        f"🧪 Noise-HPO selected thresholds: "
                        f"gain_min={best_gain_min:.4f}, "
                        f"wcov_max={best_wcov_max:.3f}, "
                        f"var_ratio_min={best_var_ratio_min:.2f}, "
                        f"eff_size_min={best_eff_size_min:.2f}; "
                        f"removed={len(best_noisy_features)} features, "
                        f"kept={len(best_kept_features)}"
                    )

            # Apply best thresholds to noise_df for diagnostics and downstream pruning
            if best_noisy_features:
                noise_df['flag_importance'] = noise_df['gain_norm'] >= best_gain_min
                noise_df['flag_wcov'] = noise_df['wcov_distinctiveness'] <= best_wcov_max
                noise_df['flag_conf'] = noise_df['var_ratio_confidence'] >= best_var_ratio_min
                noise_df['flag_misclass'] = noise_df['misclass_effect_size'] >= best_eff_size_min
                noise_df['is_noisy_feature'] = best_is_noisy

                noisy_features = best_noisy_features
                kept_features = best_kept_features

                tprint_success(
                    f"🧹 Identified {len(noisy_features)} noisy / ambiguity-driving features "
                    f"out of {len(feature_cols)} candidates (pre min_keep cap)"
                )
                top_noisy = noise_df[noise_df['is_noisy_feature']].sort_values(
                    ['gain_norm'], ascending=False
                ).head(15)
                for _, row in top_noisy.iterrows():
                    tprint_info(
                        f"  NOISY {row['feature'][:40]:40s} | "
                        f"Gain={row['gain_norm']:.3f} | "
                        f"WCoV={row['wcov_distinctiveness']:.3f} | "
                        f"VarRatio={row['var_ratio_confidence']:.2f} | "
                        f"MisclassEff={row['misclass_effect_size']:.2f}"
                    )
            else:
                tprint_info("🧹 Noise feature diagnostics found no strong pruning candidates")
                noisy_features = []
                kept_features = feature_cols

            noise_feature_table = noise_df
        except Exception as noise_exc:
            tprint_warning(f"Noise feature diagnostics failed; skipping pruning: {noise_exc}")
            noisy_features = []
            noise_feature_table = None

        final_model = baseline_model
        final_regime_probs = baseline_regime_probs

        final_training_metrics = dict(baseline_training_metrics)
        final_training_metrics['baseline_classifier'] = baseline_training_metrics

        # Record selected noise thresholds (even if HPO disabled)
        final_training_metrics['noise_hpo_selected_thresholds'] = {
            'gain_min': float(best_gain_min) if 'best_gain_min' in locals() else None,
            'wcov_max': float(best_wcov_max) if 'best_wcov_max' in locals() else None,
            'var_ratio_min': float(best_var_ratio_min) if 'best_var_ratio_min' in locals() else None,
            'eff_size_min': float(best_eff_size_min) if 'best_eff_size_min' in locals() else None,
            'min_keep': int(min_keep),
            'n_noisy_features_initial': int(len(noisy_features)),
            'n_features_total': int(len(feature_cols)),
        }

        best_model = final_model
        best_regime_probs_local = final_regime_probs
        best_training_metrics = dict(final_training_metrics)

        # Baseline quadrant separation quality from the full model, if available.
        baseline_quadrant_quality = best_training_metrics.get("quadrant_quality", {}) or {}
        best_quadrant_pred_cv_ratio = float(
            baseline_quadrant_quality.get("pred_cv_ratio", 0.0)
            if isinstance(baseline_quadrant_quality, dict)
            else 0.0
        )

        baseline_report_full = best_training_metrics.get("classification_report", {}) or {}
        baseline_macro_f1 = (
            baseline_report_full.get("macro avg", {}).get("f1-score", 0.0)
            if isinstance(baseline_report_full, dict)
            else 0.0
        )
        best_macro_f1 = float(baseline_macro_f1)
        best_log_loss_local = float(best_training_metrics.get("val_log_loss", val_log_loss))

        if noisy_features and noise_feature_table is not None:
            candidate_df = noise_feature_table.copy()
            if "is_noisy_feature" in candidate_df.columns:
                candidate_df = candidate_df[candidate_df["is_noisy_feature"]]

            if not candidate_df.empty:
                wcov_penalty_all = np.maximum(
                    0.0, 1.5 - candidate_df["wcov_distinctiveness"].values
                )

                base_severity = (
                    candidate_df["gain_norm"].values
                    * candidate_df["var_ratio_confidence"].values
                    * candidate_df["misclass_effect_size"].values
                    * wcov_penalty_all
                )

                collapse_vals = (
                    candidate_df.get("collapse_driver_score", 0.0)
                    .fillna(0.0)
                    .values.astype(float)
                )
                clarity_vals = (
                    candidate_df.get("clarity_enhancer_score", 0.0)
                    .fillna(0.0)
                    .values.astype(float)
                )

                collapse_max = float(np.nanmax(np.abs(collapse_vals))) if np.isfinite(np.nanmax(np.abs(collapse_vals))) and np.nanmax(np.abs(collapse_vals)) > 0.0 else 0.0
                clarity_max = float(np.nanmax(np.abs(clarity_vals))) if np.isfinite(np.nanmax(np.abs(clarity_vals))) and np.nanmax(np.abs(clarity_vals)) > 0.0 else 0.0

                if collapse_max > 0.0:
                    collapse_norm = collapse_vals / (collapse_max + eps)
                else:
                    collapse_norm = np.zeros_like(collapse_vals)

                if clarity_max > 0.0:
                    clarity_norm = clarity_vals / (clarity_max + eps)
                else:
                    clarity_norm = np.zeros_like(clarity_vals)

                collapse_boost = 1.0 + 2.0 * collapse_norm
                clarity_penalty = 1.0 / (1.0 + 2.0 * clarity_norm)

                candidate_df["severity"] = (
                    base_severity * collapse_boost * clarity_penalty
                )
                candidate_df = candidate_df.sort_values("severity", ascending=False)
                candidate_features_order = (
                    candidate_df["feature"].dropna().tolist()
                )
            else:
                candidate_features_order = []
        else:
            candidate_features_order = []

        removed_features_iter: List[str] = []

        def _train_candidate_with_features(kept: List[str]) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
            X_train_pruned = X_train[kept]
            X_val_pruned = X_val[kept]
            X_full_pruned = X_full[kept]

            monotone_pruned = []
            for feat in X_full_pruned.columns:
                feat_lower = feat.lower()
                if any(kw in feat_lower for kw in [
                    'vol', 'cvar', 'drawdown', 'jump', 'acceleration',
                    'fragility', 'shock', 'tail', 'kurtosis', 'correlation'
                ]):
                    monotone_pruned.append(1)
                else:
                    monotone_pruned.append(0)

            monotone_constraints_param_pruned = "(" + ",".join(
                str(c) for c in monotone_pruned
            ) + ")"

            pruned_params = best_params.copy()
            pruned_params['monotone_constraints'] = monotone_constraints_param_pruned

            X_train_use = X_train_pruned
            y_train_use = y_train
            sample_weight_use = base_sample_weight

            if enable_high_conf:
                try:
                    temp_model_pruned = xgb.XGBClassifier(**pruned_params)
                    temp_model_pruned.fit(
                        X_train_pruned,
                        y_train,
                        sample_weight=base_sample_weight,
                        eval_set=[(X_val_pruned, y_val)],
                        verbose=False,
                    )

                    train_probs_pruned = temp_model_pruned.predict_proba(X_train_pruned)
                    max_conf_pruned = train_probs_pruned.max(axis=1)
                    high_conf_mask_pruned = max_conf_pruned >= high_conf_threshold

                    n_high_conf_pruned = int(high_conf_mask_pruned.sum())
                    frac_high_conf_pruned = (
                        float(n_high_conf_pruned) / float(len(y_train)) if len(y_train) > 0 else 0.0
                    )

                    if n_high_conf_pruned > 0:
                        y_train_high_conf_pruned = y_train[high_conf_mask_pruned]
                        per_regime_counts_pruned = pd.Series(
                            y_train_high_conf_pruned
                        ).value_counts().to_dict()
                    else:
                        per_regime_counts_pruned = {}

                    min_required_pruned = int(len(y_train) * min_high_conf_fraction)
                    regime_ok_pruned = (
                        all(
                            count >= min_high_conf_per_regime
                            for count in per_regime_counts_pruned.values()
                        )
                        if per_regime_counts_pruned
                        else False
                    )

                    if n_high_conf_pruned >= max(min_required_pruned, n_regimes) and regime_ok_pruned:
                        X_train_use = X_train_pruned.iloc[high_conf_mask_pruned]
                        y_train_use = y_train_high_conf_pruned
                        sample_weight_use = base_sample_weight[high_conf_mask_pruned]
                        tprint_info(
                            f"🧪 High-confidence training (iterative pruned) enabled: using "
                            f"{n_high_conf_pruned} / {len(y_train)} samples "
                            f"({frac_high_conf_pruned:.1%}), "
                            f"min_per_regime>={min_high_conf_per_regime}"
                        )
                    else:
                        tprint_info(
                            "🧪 High-confidence filter (iterative pruned) skipped (insufficient "
                            "high-confidence points or per-regime counts too small); "
                            "training on full dataset instead."
                        )
                except Exception as high_conf_exc_pruned:
                    tprint_warning(
                        f"High-confidence training path (iterative pruned) failed; "
                        f"falling back to full dataset: {high_conf_exc_pruned}"
                    )
                    X_train_use = X_train_pruned
                    y_train_use = y_train
                    sample_weight_use = base_sample_weight

            model_candidate = xgb.XGBClassifier(**pruned_params)
            model_candidate.fit(
                X_train_use,
                y_train_use,
                sample_weight=sample_weight_use,
                eval_set=[(X_val_pruned, y_val)],
                verbose=False,
            )

            regime_probs_candidate = model_candidate.predict_proba(X_full_pruned)
            y_val_pred_candidate = model_candidate.predict(X_val_pruned)
            y_val_probs_candidate = model_candidate.predict_proba(X_val_pruned)

            val_accuracy_candidate = accuracy_score(y_val, y_val_pred_candidate)
            val_log_loss_candidate = log_loss(y_val, y_val_probs_candidate)

            tprint_success(
                f"✅ XGBoost Classifier trained (iterative noise-pruned):\n"
                f"   Val Accuracy={val_accuracy_candidate:.3f} (baseline {val_accuracy:.3f})\n"
                f"   Val LogLoss={val_log_loss_candidate:.4f} (baseline {val_log_loss:.4f})\n"
                f"   Features={len(kept)} (baseline {len(feature_cols)})"
            )

            report_candidate = classification_report(
                y_val,
                y_val_pred_candidate,
                target_names=[f'Regime_{i}' for i in range(n_regimes)],
                output_dict=True,
                zero_division=0,
            )

            importance_data_candidate = self._calculate_comprehensive_feature_importance(
                model=model_candidate,
                X=X_full_pruned,
                y=y,
                feature_names=list(X_full_pruned.columns),
            )

            candidate_metrics = {
                'val_accuracy': float(val_accuracy_candidate),
                'val_log_loss': float(val_log_loss_candidate),
                'n_regimes': n_regimes,
                'feature_names': list(X_full_pruned.columns),
                'scaler': scaler,
                'monotone_constraints': monotone_pruned,
                'n_features': len(X_full_pruned.columns),
                'feature_importance_detailed': {
                    'global': importance_data_candidate['global'].to_dict('records'),
                    'per_regime': {
                        regime_id: df.to_dict('records')
                        for regime_id, df in importance_data_candidate['per_regime'].items()
                    },
                },
                'classification_report': report_candidate,
                'baseline_classifier': baseline_training_metrics,
            }

            # Quadrant WCoV quality for the pruned candidate (predicted regimes
            # on the full feature set restricted to pruned columns).
            if quadrant_cols:
                try:
                    candidate_quadrant_cols = [
                        c for c in quadrant_cols if c in X_full_pruned.columns
                    ]
                    if candidate_quadrant_cols:
                        full_pruned_df = pd.DataFrame(
                            X_full_pruned, columns=X_full_pruned.columns
                        )
                        quad_full_pruned = full_pruned_df[candidate_quadrant_cols]

                        regime_pred_all_pruned = np.argmax(regime_probs_candidate, axis=1)

                        teacher_between_pruned = self._calculate_winsorized_cv_between(
                            y, quad_full_pruned
                        )
                        teacher_within_pruned = self._calculate_winsorized_cv_within(
                            y, quad_full_pruned
                        )
                        pred_between_pruned = self._calculate_winsorized_cv_between(
                            regime_pred_all_pruned, quad_full_pruned
                        )
                        pred_within_pruned = self._calculate_winsorized_cv_within(
                            regime_pred_all_pruned, quad_full_pruned
                        )

                        quadrant_quality_candidate = {
                            "quadrant_features": list(candidate_quadrant_cols),
                            "teacher_cv_between": float(teacher_between_pruned),
                            "teacher_cv_within": float(teacher_within_pruned),
                            "teacher_cv_ratio": float(
                                teacher_between_pruned / (teacher_within_pruned + 1e-8)
                            ),
                            "pred_cv_between": float(pred_between_pruned),
                            "pred_cv_within": float(pred_within_pruned),
                            "pred_cv_ratio": float(
                                pred_between_pruned / (pred_within_pruned + 1e-8)
                            ),
                        }
                        candidate_metrics["quadrant_quality"] = quadrant_quality_candidate
                except Exception as quad_prune_exc:
                    tprint_warning(
                        f"Quadrant WCoV quality (iterative pruned) failed (non-fatal): {quad_prune_exc}"
                    )

            return model_candidate, regime_probs_candidate, candidate_metrics

        epsilon_quadrant = float(config.get('risk_iterative_prune_min_delta_quadrant', 0.0))
        max_rounds = int(config.get('risk_iterative_prune_max_rounds', 10))

        # Minimum improvement thresholds for the two components that drive the
        # combined objective below; these are applied after normalization so
        # they are effectively in [0, 1] space.
        epsilon_balance = float(config.get("risk_iterative_prune_min_delta_balance", 0.0))
        lambda_quadrant = float(config.get("risk_iterative_prune_weight_quadrant", 0.5))
        lambda_balance = float(config.get("risk_iterative_prune_weight_balance", 0.5))
        if lambda_quadrant < 0.0:
            lambda_quadrant = 0.0
        if lambda_balance < 0.0:
            lambda_balance = 0.0
        if lambda_quadrant + lambda_balance <= 0.0:
            lambda_quadrant, lambda_balance = 0.5, 0.5

        best_balance_score: Optional[float] = None
        try:
            base_pred_full = np.argmax(best_regime_probs_local, axis=1)
            base_counts = np.bincount(base_pred_full, minlength=n_regimes)
            base_total = float(base_counts.sum()) if base_counts.sum() > 0 else 0.0
            if base_total > 0.0:
                p_base = base_counts.astype(float) / base_total
                best_balance_score = float(p_base.max() - p_base.min())
        except Exception as base_balance_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Iterative pruning baseline balance diagnostics failed (non-fatal): {base_balance_exc}"
            )
            best_balance_score = None

        for round_idx in range(max_rounds):
            remaining = [f for f in candidate_features_order if f not in removed_features_iter]
            if not remaining:
                break

            chunk = remaining[:5]
            candidate_removed = removed_features_iter + chunk
            kept_round = [f for f in feature_cols if f not in candidate_removed]

            if len(kept_round) < min_keep:
                tprint_warning(
                    f"🧹 Iterative pruning stopped due to min_keep={min_keep}: "
                    f"attempted kept={len(kept_round)}"
                )
                break

            tprint_info(
                f"🧪 Iterative pruning round {round_idx + 1}: testing removal of "
                f"{len(chunk)} features (total removed would be {len(candidate_removed)})"
            )

            cand_model, cand_probs, cand_metrics = _train_candidate_with_features(kept_round)

            cand_report_full = cand_metrics.get('classification_report', {}) or {}
            cand_macro_f1 = (
                cand_report_full.get('macro avg', {}).get('f1-score', 0.0)
                if isinstance(cand_report_full, dict)
                else 0.0
            )
            cand_log_loss = float(cand_metrics.get('val_log_loss', best_log_loss_local))

            # Quadrant separation quality for the candidate
            cand_quadrant_quality = cand_metrics.get('quadrant_quality', {}) or {}
            cand_quadrant_pred_cv_ratio = float(
                cand_quadrant_quality.get('pred_cv_ratio', best_quadrant_pred_cv_ratio)
                if isinstance(cand_quadrant_quality, dict)
                else best_quadrant_pred_cv_ratio
            )

            cand_balance_score = None
            try:
                cand_pred_full = np.argmax(cand_probs, axis=1)
                counts = np.bincount(cand_pred_full, minlength=n_regimes)
                total = float(counts.sum()) if counts.sum() > 0 else 0.0
                if total > 0.0:
                    p = counts.astype(float) / total
                    cand_min_pct = float(p.min())
                    cand_max_pct = float(p.max())
                    cand_balance_score = float(cand_max_pct - cand_min_pct)

                    # Soft band diagnostics: report when regime shares fall
                    # outside the preferred range, but do not enforce a hard
                    # rejection. The actual acceptance decision is driven by
                    # the combined score and improvement thresholds below.
                    max_pref = float(
                        config.get("risk_prune_max_regime_pct", 0.6)
                    )
                    min_pref = float(
                        config.get("risk_prune_min_regime_pct", 0.03)
                    )

                    if cand_max_pct > max_pref or cand_min_pct < min_pref:
                        tprint_info(
                            "🧹 Iterative pruning candidate outside preferred "
                            f"regime share band: min_pct={cand_min_pct:.3f}, "
                            f"max_pct={cand_max_pct:.3f}"
                        )
            except Exception as dist_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Iterative pruning balance diagnostics failed (non-fatal): {dist_exc}"
                )

            # Normalized improvements for a combined scalar objective.
            #
            # Quadrant term: prefer larger increases in pred_cv_ratio.
            quad_gain_raw = (
                cand_quadrant_pred_cv_ratio - best_quadrant_pred_cv_ratio
            )
            quad_gain_norm = max(0.0, quad_gain_raw)

            # Balance term: prefer *reductions* in max-min spread
            # (smaller is better). Map into a positive direction where
            # improvements are positive.
            if (
                cand_balance_score is not None
                and best_balance_score is not None
                and best_balance_score > 0.0
            ):
                balance_gain_raw = best_balance_score - cand_balance_score
                balance_gain_norm = max(0.0, balance_gain_raw)
            else:
                balance_gain_norm = 0.0

            # Simple weighted sum; both components are non-negative after the
            # max(0, ·) clamps. Users can tilt toward structure (quadrant) or
            # distribution (balance) via config weights.
            combined_score = (
                lambda_quadrant * quad_gain_norm
                + lambda_balance * balance_gain_norm
            )

            improved = (
                quad_gain_norm > epsilon_quadrant
                or balance_gain_norm > epsilon_balance
                or combined_score > 0.0
            )

            if improved:
                removed_features_iter = candidate_removed
                best_model = cand_model
                best_regime_probs_local = cand_probs
                best_training_metrics = cand_metrics
                best_macro_f1 = cand_macro_f1
                best_log_loss_local = cand_log_loss
                best_quadrant_pred_cv_ratio = cand_quadrant_pred_cv_ratio
                if cand_balance_score is not None:
                    best_balance_score = cand_balance_score
                tprint_success(
                    f"🧹 Iterative pruning round {round_idx + 1} accepted: "
                    f"macroF1={cand_macro_f1:.3f}, "
                    f"logLoss={cand_log_loss:.4f}, "
                    f"quadrant_pred_cv_ratio={cand_quadrant_pred_cv_ratio:.3f}"
                )
            else:
                tprint_info(
                    "🧹 Iterative pruning did not improve metrics; "
                    "stopping further removals."
                )
                break

        if removed_features_iter:
            best_training_metrics['removed_noisy_features'] = list(removed_features_iter)

        final_model = best_model
        final_regime_probs = best_regime_probs_local
        final_training_metrics = best_training_metrics

        if noise_feature_table is not None and 'noise_feature_diagnostics' not in final_training_metrics:
            final_training_metrics['noise_feature_diagnostics'] = noise_feature_table.to_dict('records')

        try:
            symbol = str(config.get("symbol", "UNKNOWN"))
            exchange = str(config.get("exchange", "UNKNOWN"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "1h")))

            # Build compact summary row from training_metrics
            baseline = final_training_metrics.get("baseline_classifier", {}) or {}
            baseline_report = baseline.get("classification_report", {}) or {}
            baseline_macro_f1 = (
                baseline_report.get("macro avg", {}).get("f1-score", 0.0)
                if isinstance(baseline_report, dict)
                else 0.0
            )

            final_report = final_training_metrics.get("classification_report", {}) or {}
            final_macro_f1 = (
                final_report.get("macro avg", {}).get("f1-score", 0.0)
                if isinstance(final_report, dict)
                else 0.0
            )

            removed_features = final_training_metrics.get("removed_noisy_features", [])
            if isinstance(removed_features, (list, tuple, set)):
                n_removed_features = len(removed_features)
            else:
                n_removed_features = 0

            row = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "val_accuracy_final": final_training_metrics.get("val_accuracy"),
                "val_log_loss_final": final_training_metrics.get("val_log_loss"),
                "macro_f1_final": final_macro_f1,
                "val_accuracy_baseline": baseline.get("val_accuracy"),
                "val_log_loss_baseline": baseline.get("val_log_loss"),
                "macro_f1_baseline": baseline_macro_f1,
                "n_features_final": final_training_metrics.get("n_features"),
                "n_features_baseline": baseline.get("n_features"),
                "n_removed_noisy_features": n_removed_features,
            }

            metrics_df = pd.DataFrame([row])

            # 1) Always emit a compact, human-readable CSV summary in outcomes/
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                metrics_report_path = (
                    f"outcomes/ml_risk_classifier_metrics_"
                    f"{symbol}_{regime_timeframe}_{ts}.csv"
                )
                metrics_df.to_csv(metrics_report_path, index=False)
                final_training_metrics["metrics_report_path"] = metrics_report_path
            except Exception as human_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Failed to write human-readable risk classifier metrics report (non-fatal): {human_exc}"
                )

            # 2) Optionally save the same summary via artifact router for programmatic use
            try:
                metrics_artifact_name = f"ml_risk_classifier_metrics_{regime_timeframe}"
                metrics_artifact_path = self._save_artifact(
                    data=metrics_df,
                    artifact_name=metrics_artifact_name,
                    artifact_type="data",
                    data_category="analysis",
                    metadata={
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": regime_timeframe,
                        "n_regimes": n_regimes,
                        "n_features_final": final_training_metrics.get("n_features", len(feature_cols)),
                        "n_features_baseline": baseline.get("n_features"),
                        "has_baseline_classifier": "baseline_classifier" in final_training_metrics,
                        "has_noise_diagnostics": "noise_feature_diagnostics" in final_training_metrics,
                    },
                )
                final_training_metrics["metrics_artifact_path"] = metrics_artifact_path
            except Exception as metrics_save_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Failed to save compact risk classifier metrics artifact (non-fatal): {metrics_save_exc}"
                )

        except Exception as outer_metrics_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Risk classifier metrics persistence encountered a non-fatal error: {outer_metrics_exc}"
            )

        full_probs = np.full((len(risk_df), n_regimes), np.nan)
        full_probs[valid_mask] = final_regime_probs

        return final_model, full_probs, final_training_metrics

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
                    f"    ↪ regime boundary optimization iter={iteration}/{max_iterations} "
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
        risk_df: pd.DataFrame,
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
        regime_counts_to_test = config.get("alpha_regime_counts", [4, 5, 6])
        min_bin_pct = float(config.get("alpha_min_bin_pct", 0.15))
        max_bin_pct = float(config.get("alpha_max_bin_pct", 0.35))

        # Validate regime counts
        if isinstance(regime_counts_to_test, int):
            regime_counts_to_test = [regime_counts_to_test]
        regime_counts_to_test = [n for n in regime_counts_to_test if 3 <= n <= 6]
        if not regime_counts_to_test:
            regime_counts_to_test = [5]  # Fallback to default

        # Optional smoothing of alpha scores before constructing regimes
        score_smoothing_method = str(
            config.get("alpha_score_smoothing_method", "none")
        ).lower()
        score_smoothing_window = int(config.get("alpha_score_smoothing_window", 1))
        scores_for_binning = alpha_scores.copy()
        if score_smoothing_method != "none" and score_smoothing_window > 1:
            try:
                if score_smoothing_method == "ewm":
                    scores_for_binning = scores_for_binning.ewm(
                        span=score_smoothing_window, adjust=False
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
                    f"🧹 Applied {score_smoothing_method} smoothing to alpha scores "
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
            return risk_df, None, None

        # Get forward returns for optimization
        fwd_cols = [col for col in risk_df.columns if col.startswith("alpha_forward_return_")]
        if not fwd_cols:
            tprint_warning("No alpha_forward_return column found for regime optimization")
            return risk_df, None, None

        fwd_col = fwd_cols[0]
        forward_returns = risk_df[fwd_col].dropna()

        # Align scores and returns
        common_idx = valid_scores.index.intersection(forward_returns.index)
        if len(common_idx) < 20:
            tprint_warning(
                f"Not enough valid samples ({len(common_idx)}) for regime optimization"
            )
            return risk_df, None, None

        aligned_scores = valid_scores.loc[common_idx]
        aligned_returns = forward_returns.loc[common_idx]

        # Multi-regime competition: test multiple regime counts
        if use_flexible_regimes:
            tprint_info(
                f"🔍 Testing flexible regime optimization for {regime_counts_to_test} regime counts "
                f"(min_bin={min_bin_pct*100:.0f}%, max_bin={max_bin_pct*100:.0f}%)"
            )

            best_n_regimes = None
            best_score = -np.inf
            best_labels = None
            best_metrics = None
            competition_results = []

            for n_regimes in regime_counts_to_test:
                if len(aligned_scores) < n_regimes:
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
                        f"  → {n_regimes} regimes: WCV Ratio={score:.4f}, "
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
                    f"✅ Selected {num_bins} regimes with WCV Ratio={best_score:.4f} "
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
                    return risk_df, None, None

            try:
                ranks = aligned_scores.rank(method="first")
                bucket_codes = pd.qcut(ranks, q=num_bins, labels=False)
                tprint_info(
                    f"📊 Using simple quantile binning with {num_bins} equal-sized regimes"
                )
            except ValueError as e:
                tprint_warning(f"Failed to compute quantile-based alpha regimes: {e}")
                return risk_df, None, None

        bucket_col = f"alpha_regime_bucket_{num_bins}"
        risk_df[bucket_col] = bucket_codes.reindex(risk_df.index)

        # Compute comprehensive regime statistics with CV and WCV metrics
        fwd_col = fwd_cols[0]

        stats_records = []
        for bucket in sorted(bucket_codes.unique()):
            mask = risk_df[bucket_col] == bucket
            group = risk_df.loc[mask]
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

            hit_rate = float((ret > 0).mean())
            mean_target = float(group["alpha_target"].mean())

            # Calculate bin percentage
            bin_pct = float(len(group)) / float(len(risk_df))

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
            return risk_df, None, bucket_col

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
            f"📊 Computed alpha regime statistics for {len(regime_stats_df)} regimes "
            f"(bins={num_bins})"
        )

        if use_flexible_regimes and best_metrics is not None:
            tprint_info(
                f"📈 Overall Economic Metrics: "
                f"CV Ratio={best_metrics.get('cv_ratio', 0.0):.4f}, "
                f"WCV Ratio={best_metrics.get('wcv_ratio', 0.0):.4f}, "
                f"Between WCV={best_metrics.get('between_wcv', 0.0):.4f}, "
                f"Within WCV={best_metrics.get('within_wcv', 0.0):.4f}"
            )

        return risk_df, regime_stats_df, bucket_col

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
            - thresholds: Dict mapping regime → {min_score, max_score}
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
                f"✅ Regime thresholds extracted: {len(unique_regimes)} regimes, "
                f"min_score={min(alpha_scores):.6f}, max_score={max(alpha_scores):.6f}"
            )

            return threshold_data

        except Exception as e:
            tprint_warning(f"Regime threshold extraction failed: {e}")
            return {"extraction_error": str(e)}

    def _assign_risk_regimes_with_kde(
        self,
        risk_df: pd.DataFrame,
        risk_scores: pd.Series,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str]]:
        """Assign risk regimes using KDE-based binning with adaptive bandwidth selection.

        Workflow:
        1. Apply MinMaxScaler to risk scores → [0, 1]
        2. Adaptively select KDE bandwidth using cross-validation
        3. Find local minima (valleys) in KDE curve with refined detection
        4. Place bin walls at deepest valleys
        5. Apply regime size constraints to prevent extreme imbalance
        6. Assign samples to bins
        7. Calculate probabilistic inference using CDF

        Args:
            risk_df: DataFrame with risk features and targets
            risk_scores: Predicted risk scores from XGBoost
            config: Configuration dict

        Returns:
            Tuple of (updated_df, regime_stats_df, regime_col_name)
        """
        from scipy.signal import argrelextrema
        from scipy.stats import norm

        tprint_info("🔍 Assigning risk regimes using adaptive KDE binning with multimodal detection...")

        # Validate inputs
        valid_scores = risk_scores.dropna()
        if len(valid_scores) < 20:
            tprint_warning(f"Insufficient valid scores ({len(valid_scores)}) for KDE binning")
            return risk_df, None, None

        # 1. MinMaxScale scores to [0, 1] (should already be done, but ensure)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores_scaled = scaler.fit_transform(valid_scores.values.reshape(-1, 1)).flatten()
        scores_series = pd.Series(scores_scaled, index=valid_scores.index)

        # 2. Adaptive Kernel Density Estimation with cross-validated bandwidth
        # Try multiple bandwidth values to find one that reveals multimodal structure
        kde_bandwidth_candidates = float(config.get("risk_kde_bandwidth", 0.08))
        try:
            # Use cross-validation to find optimal bandwidth
            kde = self._fit_kde_with_adaptive_bandwidth(scores_scaled, config)
            tprint_info(f"✅ KDE bandwidth selected via adaptive method")
        except Exception as e:
            tprint_warning(f"Adaptive KDE bandwidth failed, using scott method: {e}")
            try:
                kde = gaussian_kde(scores_scaled, bw_method='scott')
            except Exception as e2:
                tprint_warning(f"KDE failed with scott method: {e2}, using default 0.08")
                kde = gaussian_kde(scores_scaled, bw_method=0.08)

        # Evaluate KDE on fine grid
        x_grid = np.linspace(0, 1, 1000)
        kde_values = kde(x_grid)

        # 3. Find meaningful minima (valleys where we'll place bin walls)
        # with improved multimodal detection and regime balance constraints

        # ENHANCED: Stricter defaults for minimum regime size to prevent extreme imbalance
        # Minimum 5% of data per regime (prevents 996 vs 4 split)
        min_bin_samples = int(config.get("risk_min_bin_samples_pct", 0.05) * len(scores_scaled))
        max_bin_samples = int(config.get("risk_max_bin_samples_pct", 0.95) * len(scores_scaled))

        # Target number of regimes (try to get 4)
        target_regimes = int(
            config.get("risk_target_regimes", config.get("risk_n_regimes", 4))
        )
        max_regimes = int(config.get("risk_max_regimes", 6))
        max_regimes = max(max_regimes, 2)  # At least 2 regimes

        # Minimum separation between consecutive breakpoints in score space
        min_sep = float(config.get("risk_min_breakpoint_separation", 0.02))

        # Find all local minima with refined order (more sensitive detection)
        minima_indices = argrelextrema(kde_values, np.less, order=5)[0]  # Reduced order for better detection

        tprint_info(f"🔍 Found {len(minima_indices)} local minima in KDE curve")

        if len(minima_indices) == 0:
            tprint_warning("No local minima found in KDE, using quantile-based binning (equal-probability)")
            # Fallback: Equal-probability bins targeting 4 regimes
            quantiles = np.linspace(0, 1, target_regimes + 1)[1:-1]
            breakpoints = [np.quantile(scores_scaled, q) for q in quantiles]
        else:
            # ENHANCED: Evaluate minima with importance scoring
            candidate_x = x_grid[minima_indices]
            candidate_vals = kde_values[minima_indices]
            max_kde = float(kde_values.max())
            depths = max_kde - candidate_vals

            # Normalize depths to [0, 1] for better comparison
            min_depth = depths.min()
            max_depth = depths.max()
            if max_depth > min_depth:
                normalized_depths = (depths - min_depth) / (max_depth - min_depth)
            else:
                normalized_depths = np.ones_like(depths)

            # Order minima by depth (deepest = most meaningful valley)
            order_idx = np.argsort(-normalized_depths)

            selected_x: List[float] = []
            for idx in order_idx:
                x_val = float(candidate_x[idx])
                depth_score = float(normalized_depths[idx])

                # Enforce minimum separation between selected breakpoints
                if selected_x:
                    closest_sep = min(abs(x_val - np.array(selected_x)))
                    if closest_sep < min_sep:
                        tprint_info(f"  Skipping breakpoint at {x_val:.4f} (too close to existing, sep={closest_sep:.4f})")
                        continue

                selected_x.append(x_val)
                tprint_info(f"  ✓ Selected breakpoint at {x_val:.4f} (depth_score={depth_score:.4f})")

                # We want (target_regimes - 1) interior breakpoints, but allow up to max_regimes
                if len(selected_x) >= target_regimes - 1:
                    break

            if not selected_x:
                # Fallback to the single deepest minimum
                tprint_warning("All minima filtered out by separation check, using deepest minimum")
                best_idx = int(np.argmax(normalized_depths))
                selected_x = [float(candidate_x[best_idx])]

            breakpoints_raw = np.array(selected_x, dtype=float)

            # ENHANCED: Iteratively refine breakpoints to ensure balanced regime sizes
            breakpoints = self._refine_breakpoints_with_balance(
                scores_scaled,
                breakpoints_raw,
                min_samples=min_bin_samples,
                max_samples=max_bin_samples,
                target_regimes=target_regimes,
            )

            # If refinement removes all breakpoints, fall back to quantile binning
            if breakpoints.size == 0:
                tprint_warning("Refined KDE breakpoints empty; falling back to equal-probability quantile binning")
                quantiles = np.linspace(0, 1, target_regimes + 1)[1:-1]
                breakpoints = np.array([np.quantile(scores_scaled, q) for q in quantiles], dtype=float)

        # Add boundaries
        breakpoints = np.sort(np.concatenate([[0.0], breakpoints, [1.0]]))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        n_regimes = len(breakpoints) - 1
        tprint_info(f"📊 KDE binning: {n_regimes} regimes with breakpoints at {breakpoints}")

        # 4. Assign samples to bins (hard assignment)
        regime_labels = np.digitize(scores_scaled, bins=breakpoints[1:-1], right=False)
        regime_series = pd.Series(regime_labels, index=valid_scores.index, name="risk_regime")

        # 5. Calculate probabilistic inference (soft assignment using CDF)
        # Get residual sigma from training metrics
        sigma = float(config.get("val_residual_sigma", 0.05))  # Fallback if not in config

        # For each sample, calculate P(Bin) using CDF
        prob_cols = []
        for i in range(n_regimes):
            lower = breakpoints[i]
            upper = breakpoints[i + 1]

            # P(Bin_i) = CDF(upper) - CDF(lower)
            # Using normal distribution centered at predicted score with std=sigma
            probs = []
            for score in scores_scaled:
                cdf_upper = norm.cdf(upper, loc=score, scale=sigma)
                cdf_lower = norm.cdf(lower, loc=score, scale=sigma)
                prob = cdf_upper - cdf_lower
                probs.append(prob)

            prob_col_name = f"risk_regime_{i}_prob"
            prob_series = pd.Series(probs, index=valid_scores.index, name=prob_col_name)
            prob_cols.append(prob_col_name)

            # Add to dataframe
            risk_df.loc[valid_scores.index, prob_col_name] = prob_series

        # 6. Apply asymmetric hysteresis
        regime_series_hysteresis = self._apply_asymmetric_hysteresis(
            regime_series,
            prob_cols,
            risk_df.loc[valid_scores.index],
            config
        )

        # Add regime labels to dataframe
        regime_col_name = "risk_regime"
        risk_df.loc[valid_scores.index, regime_col_name] = regime_series_hysteresis

        # 7. Calculate regime statistics
        regime_stats = self._calculate_regime_statistics(
            risk_df.loc[valid_scores.index],
            regime_series_hysteresis,
            risk_scores.loc[valid_scores.index],
        )

        # Save breakpoints and sigma to config for production use
        config["risk_regime_breakpoints"] = breakpoints.tolist()
        config["risk_regime_sigma"] = sigma
        config["risk_regime_n_regimes"] = n_regimes

        tprint_success(
            f"✅ KDE binning complete: {n_regimes} regimes, "
            f"bin sizes: {regime_series_hysteresis.value_counts().to_dict()}"
        )

        return risk_df, regime_stats, regime_col_name

    def _refine_breakpoints(
        self,
        scores: np.ndarray,
        breakpoints_raw: np.ndarray,
        min_samples: int,
        max_samples: int,
    ) -> np.ndarray:
        """Refine breakpoints to enforce bin size constraints.

        Iteratively remove breakpoints that create bins with too few or too many samples.
        """
        breakpoints = np.sort(breakpoints_raw)

        # Iterative refinement
        max_iterations = 20
        for iteration in range(max_iterations):
            # Add boundaries
            full_breaks = np.concatenate([[0.0], breakpoints, [1.0]])

            # Check bin sizes
            bin_sizes = []
            bins_to_remove = []

            for i in range(len(full_breaks) - 1):
                lower = full_breaks[i]
                upper = full_breaks[i + 1]
                count = np.sum((scores >= lower) & (scores < upper))
                bin_sizes.append(count)

                # If bin too small and not first/last, mark for removal
                if count < min_samples and i > 0 and i < len(full_breaks) - 2:
                    bins_to_remove.append(i)

            # Remove problematic breakpoints
            if len(bins_to_remove) > 0:
                # Remove breakpoint (merge with neighboring bin)
                mask = np.ones(len(breakpoints), dtype=bool)
                for idx in bins_to_remove:
                    if idx - 1 < len(mask):
                        mask[idx - 1] = False
                breakpoints = breakpoints[mask]
            else:
                break  # Converged

        return breakpoints

    def _refine_breakpoints_with_balance(
        self,
        scores: np.ndarray,
        breakpoints_raw: np.ndarray,
        min_samples: int,
        max_samples: int,
        target_regimes: int = 4,
    ) -> np.ndarray:
        """Refine breakpoints to enforce bin size constraints with balance optimization.

        Iteratively remove or adjust breakpoints to ensure:
        1. Each regime has at least min_samples
        2. No regime exceeds max_samples
        3. Regimes are roughly balanced
        4. Target number of regimes is achieved if possible

        Args:
            scores: Scaled risk scores
            breakpoints_raw: Initial breakpoints
            min_samples: Minimum samples per regime
            max_samples: Maximum samples per regime
            target_regimes: Target number of regimes

        Returns:
            Refined breakpoints array
        """
        breakpoints = np.sort(breakpoints_raw)

        tprint_info(
            f"🔧 Refining breakpoints for balance: "
            f"target={target_regimes} regimes, min={min_samples}, max={max_samples} samples"
        )

        # Iterative refinement
        max_iterations = 30
        for iteration in range(max_iterations):
            # Add boundaries
            full_breaks = np.concatenate([[0.0], breakpoints, [1.0]])

            # Check bin sizes and balance
            bin_sizes = []
            bins_to_remove = []

            for i in range(len(full_breaks) - 1):
                lower = full_breaks[i]
                upper = full_breaks[i + 1]
                count = np.sum((scores >= lower) & (scores <= upper))
                bin_sizes.append(count)

                # Mark bins that violate constraints for removal
                if count < min_samples and i > 0 and i < len(full_breaks) - 2:
                    bins_to_remove.append(i)
                    tprint_info(
                        f"  Bin {i}: {count} samples < min({min_samples}), marked for removal"
                    )

            # Remove problematic breakpoints
            if len(bins_to_remove) > 0:
                # Sort in descending order to avoid index shifting issues
                for idx in sorted(set(bins_to_remove), reverse=True):
                    if idx - 1 < len(breakpoints):
                        breakpoints = np.delete(breakpoints, idx - 1)
            else:
                # Check if we have too many regimes
                n_current_regimes = len(full_breaks) - 1
                if n_current_regimes > target_regimes + 2:
                    # Find smallest bin and remove its breakpoint
                    min_bin_idx = np.argmin(bin_sizes[1:-1]) + 1  # Skip first/last
                    if min_bin_idx > 0 and min_bin_idx < len(full_breaks) - 2:
                        breakpoints = np.delete(breakpoints, min_bin_idx - 1)
                        tprint_info(
                            f"  Too many regimes ({n_current_regimes}), "
                            f"removing breakpoint at index {min_bin_idx - 1}"
                        )
                else:
                    # Check for balance (no bin > 2x the smallest)
                    non_zero_bins = [b for b in bin_sizes if b > 0]
                    if len(non_zero_bins) >= 2:
                        max_bin = max(non_zero_bins)
                        min_bin = min(non_zero_bins)
                        if min_bin > 0 and max_bin / min_bin > 3.0:
                            # Unbalanced, try to rebalance by removing a breakpoint
                            max_bin_idx = np.argmax(bin_sizes[1:-1]) + 1  # Skip first/last
                            if max_bin_idx > 0 and max_bin_idx < len(full_breaks) - 2:
                                breakpoints = np.delete(breakpoints, max_bin_idx - 1)
                                tprint_info(
                                    f"  Imbalanced (ratio {max_bin/min_bin:.1f}), "
                                    f"removing breakpoint to rebalance"
                                )
                        else:
                            break  # Converged
                    else:
                        break  # Converged

        final_breaks = np.concatenate([[0.0], breakpoints, [1.0]])
        final_sizes = []
        for i in range(len(final_breaks) - 1):
            lower = final_breaks[i]
            upper = final_breaks[i + 1]
            count = np.sum((scores >= lower) & (scores <= upper))
            final_sizes.append(count)

        tprint_info(
            f"✅ Breakpoint refinement complete: {len(breakpoints)} breakpoints, "
            f"{len(final_breaks) - 1} regimes, sizes={final_sizes}"
        )

        return breakpoints

    def _fit_kde_with_adaptive_bandwidth(
        self,
        scores: np.ndarray,
        config: Dict[str, Any],
    ):
        """Fit KDE with adaptive bandwidth selection for multimodal detection.

        Uses multiple candidate bandwidths and selects the one that best reveals
        the multimodal structure (most local minima within reasonable limits).

        Args:
            scores: Scaled risk scores [0, 1]
            config: Configuration dict

        Returns:
            KDE object with selected bandwidth
        """
        from scipy.signal import argrelextrema
        from scipy.stats import gaussian_kde

        # Try multiple bandwidth candidates
        # Smaller bandwidths reveal more modes, larger smooth them out
        bandwidth_candidates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

        best_kde = None
        best_n_modes = 0
        best_bandwidth = None

        tprint_info("🔍 Selecting optimal KDE bandwidth for multimodal detection...")

        for bw in bandwidth_candidates:
            try:
                kde = gaussian_kde(scores, bw_method=bw)
                x_grid = np.linspace(0, 1, 1000)
                kde_values = kde(x_grid)

                # Count local minima (modes are separated by minima)
                minima_indices = argrelextrema(kde_values, np.less, order=5)[0]
                n_minima = len(minima_indices)
                n_modes = n_minima + 1  # Number of modes = minima + 1

                tprint_info(f"  Bandwidth {bw:.3f}: {n_modes} modes ({n_minima} minima)")

                # Prefer bandwidths that give 3-4 modes (4-5 regimes)
                # But accept 2+ modes
                if n_modes >= 2:
                    # Score preference: modes closer to target (4) are better
                    mode_score = max(0, 1.0 - abs(n_modes - 4) / 4.0)

                    if best_kde is None or mode_score > best_n_modes:
                        best_kde = kde
                        best_n_modes = mode_score
                        best_bandwidth = bw

            except Exception as e:
                tprint_info(f"  Bandwidth {bw:.3f}: Failed - {str(e)[:50]}")
                continue

        if best_kde is not None:
            tprint_info(f"✅ Selected bandwidth {best_bandwidth:.3f} (mode_score={best_n_modes:.3f})")
            return best_kde
        else:
            # Fallback to scott method if all candidates fail
            tprint_warning("All bandwidth candidates failed, using scott method")
            return gaussian_kde(scores, bw_method='scott')

    def _apply_asymmetric_hysteresis(
        self,
        regime_labels: pd.Series,
        prob_cols: List[str],
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.Series:
        """Apply asymmetric hysteresis to regime transitions.

        Rules:
        - Switching to Danger (high regime): Instant if Prob > 25%
        - Switching back to Safe (low regime): Delayed, require 3 consecutive candles
        """
        danger_threshold = float(config.get("risk_danger_threshold", 0.25))
        safety_confirmation_bars = int(config.get("risk_safety_confirmation_bars", 12))

        n_regimes = regime_labels.max() + 1  # Highest regime = most dangerous
        calm_regime = 0  # Lowest regime = calmest

        result_labels = regime_labels.copy()

        # Iterate through time series
        for i in range(1, len(regime_labels)):
            current_hard_regime = regime_labels.iloc[i]
            previous_final_regime = result_labels.iloc[i - 1]

            # Check if transitioning to danger
            if current_hard_regime > previous_final_regime:
                # Moving to higher risk: check if prob > danger_threshold
                danger_prob_col = f"risk_regime_{n_regimes - 1}_prob"
                if danger_prob_col in df.columns:
                    danger_prob = df[danger_prob_col].iloc[i]
                    if danger_prob > danger_threshold:
                        # Instant transition to danger
                        result_labels.iloc[i] = current_hard_regime
                    else:
                        # Stay in previous regime (not enough confidence)
                        result_labels.iloc[i] = previous_final_regime
                else:
                    # No prob available, use hard assignment
                    result_labels.iloc[i] = current_hard_regime

            # Check if transitioning to safety
            elif current_hard_regime < previous_final_regime:
                # Moving to lower risk: require confirmation
                # Check if safe signal persists for N bars
                lookback_start = max(0, i - safety_confirmation_bars + 1)
                recent_labels = regime_labels.iloc[lookback_start:i+1]

                if len(recent_labels) >= safety_confirmation_bars:
                    # Check if all recent bars indicate safe regime
                    if (recent_labels <= current_hard_regime).all():
                        # Confirmed safe transition
                        result_labels.iloc[i] = current_hard_regime
                    else:
                        # Not confirmed, stay in danger
                        result_labels.iloc[i] = previous_final_regime
                else:
                    # Not enough history, stay in previous regime
                    result_labels.iloc[i] = previous_final_regime
            else:
                # No transition, keep current assignment
                result_labels.iloc[i] = current_hard_regime

        n_transitions = (result_labels.diff() != 0).sum()
        tprint_info(f"⚡ Asymmetric hysteresis applied: {n_transitions} regime transitions")

        return result_labels

    def _calculate_regime_statistics(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
        risk_scores: pd.Series,
    ) -> pd.DataFrame:
        """Calculate statistics for each regime."""
        unique_regimes = sorted(regime_labels.unique())
        stats_records = []

        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_scores = risk_scores[regime_mask]

            stats = {
                "regime_id": int(regime_id),
                "n_samples": int(regime_mask.sum()),
                "mean_risk_score": float(regime_scores.mean()),
                "std_risk_score": float(regime_scores.std()),
                "min_risk_score": float(regime_scores.min()),
                "max_risk_score": float(regime_scores.max()),
            }

            # Calculate returns if available
            if 'returns_1h' in df.columns:
                regime_returns = df.loc[regime_mask, 'returns_1h'].dropna()
                if len(regime_returns) > 0:
                    stats["mean_return"] = float(regime_returns.mean())
                    stats["std_return"] = float(regime_returns.std())

            stats_records.append(stats)

        regime_stats_df = pd.DataFrame(stats_records)
        return regime_stats_df

    def _assess_risk_regime_quality(
        self,
        risk_df: pd.DataFrame,
        regime_col: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Assess risk regime quality using RiskClusterQualityAssessor.

        Args:
            risk_df: DataFrame with risk features and regime assignments
            regime_col: Name of regime column
            config: Configuration dict

        Returns:
            Tuple of (risk_quality_metrics, risk_quality_path)
        """
        from src.training.steps.market_analysis.risk_cluster_quality_assessor import (
            RiskClusterQualityAssessor,
            RiskClusterQualityMetrics,
        )

        # Fallback: if no regime column name was provided, use the standard risk_regime label
        if (regime_col is None or regime_col not in risk_df.columns) and "risk_regime" in risk_df.columns:
            regime_col = "risk_regime"

        if regime_col is None or regime_col not in risk_df.columns:
            tprint_warning("No regime column available for risk quality assessment")
            return None, None

        regime_labels = risk_df[regime_col].dropna()
        if len(regime_labels) < 20:
            tprint_warning(f"Insufficient regime samples ({len(regime_labels)}) for quality assessment")
            return None, None

        try:
            assessor = RiskClusterQualityAssessor(config=config)
            risk_quality_metrics = assessor.assess_risk_clusters(
                risk_df=risk_df,
                regime_labels=regime_labels,
                config=config
            )

            # Save assessment report
            symbol = config.get("symbol", "UNKNOWN")
            report_path = assessor.save_assessment_report(
                metrics=risk_quality_metrics,
                symbol=symbol,
                output_dir="outcomes"
            )

            # Save metrics as artifact
            quality_metrics_dict = {
                "var_stratification_score": risk_quality_metrics.var_stratification_score,
                "cvar_stratification_score": risk_quality_metrics.cvar_stratification_score,
                "var_monotonicity": risk_quality_metrics.var_monotonicity,
                "cvar_monotonicity": risk_quality_metrics.cvar_monotonicity,
                "volatility_clustering_coeff": risk_quality_metrics.volatility_clustering_coeff,
                "within_vol_cv": risk_quality_metrics.within_vol_cv,
                "between_vol_cv": risk_quality_metrics.between_vol_cv,
                "vol_separation_ratio": risk_quality_metrics.vol_separation_ratio,
                "calm_to_crash_direct_prob": risk_quality_metrics.calm_to_crash_direct_prob,
                "calm_to_turbulent_to_crash_prob": risk_quality_metrics.calm_to_turbulent_to_crash_prob,
                "transition_stability_score": risk_quality_metrics.transition_stability_score,
                "overall_quality_score": risk_quality_metrics.overall_quality_score,
                "n_regimes": risk_quality_metrics.n_regimes,
                "n_samples": risk_quality_metrics.n_samples,
                "regime_metrics": risk_quality_metrics.regime_metrics,
            }

            quality_path = self._save_artifact(
                data=quality_metrics_dict,
                artifact_name="ml_risk_quality_metrics_1h",
                artifact_type="data",
                metadata={
                    "overall_quality_score": risk_quality_metrics.overall_quality_score,
                    "n_regimes": risk_quality_metrics.n_regimes,
                    "assessment_timestamp": risk_quality_metrics.assessment_timestamp,
                },
            )

            tprint_success(
                f"✅ Risk quality assessment complete: "
                f"overall_score={risk_quality_metrics.overall_quality_score:.3f}, "
                f"report={report_path}"
            )

            return risk_quality_metrics, quality_path

        except Exception as e:
            tprint_error(f"Risk quality assessment failed: {e}")
            return None, None
    def _assess_alpha_regime_quality(
        self,
        *,
        risk_df: pd.DataFrame,
        regime_col: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[Optional[ClusterQualityMetrics], Optional[str]]:
        """Assess quality of alpha regimes using ClusterQualityAssessor.

        This uses the unified assess_quality interface and persists metrics as a
        dedicated artifact. The minimum regime size defaults to 3 but can be
        overridden via config["alpha_min_regime_size"].
        """
        if regime_col is None or regime_col not in risk_df.columns:
            tprint_warning("No alpha regime column provided; skipping regime quality assessment")
            return None, None

        regime_series = risk_df[regime_col]
        valid_mask = regime_series.notna()
        if valid_mask.sum() == 0:
            tprint_warning("No valid alpha regime labels for quality assessment")
            return None, None

        regime_labels = np.asarray(regime_series[valid_mask].astype(int), dtype=int)

        numeric_df = risk_df.select_dtypes(include=[np.number])
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

        forward_ret_cols = [c for c in risk_df.columns if c.startswith("alpha_forward_return_")]
        forward_returns = None
        if forward_ret_cols:
            forward_returns = risk_df[forward_ret_cols[0]].loc[valid_mask]

        timestamps = risk_df.index[valid_mask]
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
