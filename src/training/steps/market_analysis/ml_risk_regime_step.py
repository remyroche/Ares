"""
ML Risk Regime Step

This step consumes 1h Rolling HMM regime outputs plus OHLCV data to
construct risk-based regime labels using forward volatility and tail risk metrics.

Primary Goal: Distinguish between turbulent, calm, crash-prone, volatile but trending,
and recovering markets.

Responsibilities:
- Load 1h HMM artifacts from versioned HDF5 (labels, probabilities,
  economic features) using the same context as RollingHMMRegimeDiscoveryStep.
- Load 1h OHLCV market data.
- Align all series on a common DatetimeIndex.
- Compute composite risk target from 4 components:
  * Forward Vol 1h (30%): Short-term tactical risk
  * Forward Vol 4h (20%): Medium-term persistent risk
  * Tail Risk Probability (30%): Crash protection via CVaR
  * Vol Acceleration (20%): Regime transition detection
- Train XGBoost regression model with monotonic constraints on risk features.
- Use KDE-based binning to identify natural risk regimes.
- Apply asymmetric hysteresis (instant danger detection, delayed safety confirmation).
- Save risk regime outputs to versioned_artifacts for downstream consumption.
"""

import logging
import time
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
                config.get("regime_timeframe", config.get("timeframe", "1h"))
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
            # 2) Load 1h OHLCV market data
            # ------------------------------------------------------------------
            market_data, market_source = self.load_market_data_or_fail(
                {
                    **config,
                    "timeframe": regime_timeframe,
                },
                pipeline_state={},
                allow_config_override=True,
                light_mode_filter=False,  # ✅ FIX #1: Load full data, not limited by execution_mode
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
            # 5) Compute risk targets (4-component composite)
            # ------------------------------------------------------------------
            risk_df = self._compute_risk_targets(aligned_df, config)

            if risk_df.empty:
                raise ValueError("Risk dataset is empty after target construction")

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
                    config=config
                )

                training_metrics.update(classifier_metrics)

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

                # Store continuous score from max probability for compatibility
                risk_df['risk_score_continuous'] = np.max(regime_probs, axis=1)

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

                # Calculate regime statistics
                regime_stats_df = self._calculate_regime_statistics(risk_df, regime_col_name)

            except Exception as exc:
                tprint_error(f"❌ XGBoost regime classification failed: {exc}")
                import traceback
                traceback.print_exc()

                # Fall back to simple quantile-based regimes
                tprint_warning("⚠️ Falling back to simple quantile-based regimes")
                if 'risk_target' in risk_df.columns:
                    risk_scores = risk_df['risk_target'].dropna()
                    if len(risk_scores) >= 20:
                        regime_labels = pd.qcut(risk_scores.rank(method='first'), q=4, labels=False)
                        risk_df['risk_regime'] = np.nan
                        risk_df.loc[risk_scores.index, 'risk_regime'] = regime_labels
                        regime_col_name = 'risk_regime'
                        risk_df['risk_score_continuous'] = risk_scores
                        training_metrics['fallback_used'] = True

            # Ensure we have a valid regime column name for downstream quality assessment
            if regime_col_name is None and "risk_regime" in risk_df.columns:
                regime_col_name = "risk_regime"

            # ------------------------------------------------------------------
            # 7) Extract and save regime thresholds for production use
            # ------------------------------------------------------------------
            regime_thresholds: Optional[Dict[str, Any]] = None
            regime_thresholds_path: Optional[str] = None

            if risk_scores is not None and regime_col_name is not None and regime_col_name in risk_df.columns:
                try:
                    # Extract KDE breakpoints and sigma as regime thresholds
                    regime_thresholds = {
                        "extraction_timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "regime_col_name": regime_col_name,
                        "total_samples": len(risk_scores),
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
                                artifact_name="ml_risk_regime_thresholds_1h",
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

            risk_to_save = risk_df.reset_index().rename(columns={risk_df.index.name or "index": "timestamp"})

            tprint_info(
                f"💾 Saving risk training dataset with shape {risk_to_save.shape} "
                f"to versioned HDF5 store"
            )
            training_data_path = self._save_artifact(
                data=risk_to_save,
                artifact_name="ml_risk_training_data_1h",
                artifact_type="data",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )

            # Optionally upsample risk regimes to the base (e.g. 15m) timeframe for downstream steps
            base_timeframe = str(config.get("timeframe", "15m"))
            if base_timeframe and base_timeframe != regime_timeframe:
                try:
                    # Select regime probability and risk score columns
                    risk_prob_cols = [
                        col
                        for col in risk_df.columns
                        if col.startswith("risk_regime") or col.startswith("risk_score")
                    ]
                    if risk_prob_cols and isinstance(risk_df.index, pd.DatetimeIndex):
                        risk_probs = risk_df[risk_prob_cols].copy()

                        # Map timeframe string to pandas frequency
                        freq_map = {
                            "1m": "1T",
                            "3m": "3T",
                            "5m": "5T",
                            "15m": "15T",
                            "30m": "30T",
                            "45m": "45T",
                            "1h": "1H",
                            "2h": "2H",
                            "4h": "4H",
                            "6h": "6H",
                            "8h": "8H",
                            "12h": "12H",
                            "1d": "1D",
                        }
                        freq = freq_map.get(base_timeframe)
                        if freq is not None:
                            target_index = pd.date_range(
                                start=risk_probs.index.min(),
                                end=risk_probs.index.max(),
                                freq=freq,
                            )
                            risk_probs_resampled = risk_probs.reindex(target_index, method="ffill")
                            risk_probs_save = risk_probs_resampled.reset_index().rename(columns={"index": "timestamp"})

                            self._save_artifact(
                                data=risk_probs_save,
                                artifact_name=f"ml_risk_regime_probabilities_{base_timeframe}",
                                artifact_type="data",
                                metadata={
                                    "symbol": symbol,
                                    "exchange": exchange,
                                    "timeframe": base_timeframe,
                                    "source_regime_timeframe": regime_timeframe,
                                },
                            )
                except Exception as resample_exc:
                    tprint_warning(
                        f"Failed to upsample ML risk regimes to base timeframe {base_timeframe}: {resample_exc}"
                    )

            # Save trained model if available
            if model is not None:
                try:
                    tprint_info("💾 Saving XGBoost risk model via artifact router")
                    model_path = self._save_artifact(
                        data=model,
                        artifact_name="ml_risk_model_1h",
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
                        artifact_name="ml_risk_feature_pipeline_1h",
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
                        artifact_name="ml_risk_regime_stats_1h",
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
                        stats_csv_name = f"hmm_alpha_regime_stats_{symbol}_{ts}.csv"
                        stats_csv_path = f"outcomes/{stats_csv_name}"
                        regime_stats_df.to_csv(stats_csv_path)
                        tprint_info(
                            f"📊 Saved alpha regime statistics CSV: {stats_csv_path}"
                        )
                    except Exception as stats_csv_exc:
                        tprint_warning(
                            f"Failed to save alpha regime statistics CSV (ignored): {stats_csv_exc}"
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
            OptimizedGARCHFeatureGenerator,
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
        ewma_periods = [2, 6]

        # ============ 1. Core Volatility Features with EWMA ============

        # Simple realized volatility (baseline)
        vol_20 = returns.rolling(window=20).std()
        result_df['vol_realized_20'] = vol_20
        for period in ewma_periods:
            result_df[f'vol_realized_20_ewma{period}'] = vol_20.ewm(span=period, adjust=False).mean()

        # High-Low range volatility (Parkinson-style)
        parkinson_vol = np.sqrt(np.log(result_df['high'] / result_df['low']).rolling(window=20).var() / (4 * np.log(2)))
        result_df['parkinson_vol'] = parkinson_vol
        for period in ewma_periods:
            result_df[f'parkinson_vol_ewma{period}'] = parkinson_vol.ewm(span=period, adjust=False).mean()

        # Garman-Klass volatility (uses OHLC)
        hl = np.log(result_df['high'] / result_df['low'])
        co = np.log(result_df['close'] / result_df['open'])
        gk_vol = np.sqrt(0.5 * hl.rolling(window=20).var() - (2 * np.log(2) - 1) * co.rolling(window=20).var())
        result_df['garman_klass_vol'] = gk_vol
        for period in ewma_periods:
            result_df[f'garman_klass_vol_ewma{period}'] = gk_vol.ewm(span=period, adjust=False).mean()

        # ============ 2. Tail Risk Features ============

        # CVaR (5% and 10% levels)
        for alpha in [0.05, 0.10]:
            window = 20
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
        skew_20 = returns.rolling(window=20).skew()
        kurt_20 = returns.rolling(window=20).kurt()
        result_df['skewness_20'] = skew_20
        result_df['kurtosis_20'] = kurt_20
        for period in ewma_periods:
            result_df[f'skewness_20_ewma{period}'] = skew_20.ewm(span=period, adjust=False).mean()
            result_df[f'kurtosis_20_ewma{period}'] = kurt_20.ewm(span=period, adjust=False).mean()

        # ============ 3. Drawdown/Runup Features ============

        # Max Drawdown (rolling 50-bar window)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(window=50, min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.rolling(window=50).min()
        result_df['max_drawdown_50'] = max_dd
        for period in ewma_periods:
            result_df[f'max_drawdown_50_ewma{period}'] = max_dd.ewm(span=period, adjust=False).mean()

        # Max Run-Up (for short squeeze risk)
        running_min = cumulative.rolling(window=50, min_periods=1).min()
        runup = (cumulative - running_min) / (running_min + 1e-9)
        max_ru = runup.rolling(window=50).max()
        result_df['max_runup_50'] = max_ru
        for period in ewma_periods:
            result_df[f'max_runup_50_ewma{period}'] = max_ru.ewm(span=period, adjust=False).mean()

        # ============ 4. Jump Detection ============

        # Jump indicator (returns exceeding 3 std devs)
        vol_rolling = returns.rolling(window=20).std()
        jump_threshold = 3.0
        jumps = (abs(returns) > jump_threshold * vol_rolling).astype(float)
        result_df['jump_indicator'] = jumps
        jump_freq = jumps.rolling(window=20).mean()
        result_df['jump_frequency_20'] = jump_freq
        for period in ewma_periods:
            result_df[f'jump_frequency_20_ewma{period}'] = jump_freq.ewm(span=period, adjust=False).mean()

        # ============ 5. Volatility Expansion & Acceleration ============

        # Vol expansion (rate of change)
        vol_roc = vol_20.pct_change(periods=5)
        result_df['vol_expansion_5'] = vol_roc
        for period in ewma_periods:
            result_df[f'vol_expansion_5_ewma{period}'] = vol_roc.ewm(span=period, adjust=False).mean()

        # Vol acceleration (already in target, but also as feature)
        vol_accel = (vol_20 - vol_20.shift(6)) / (vol_20.shift(6) + 1e-9)
        result_df['vol_acceleration'] = vol_accel
        for period in ewma_periods:
            result_df[f'vol_acceleration_ewma{period}'] = vol_accel.ewm(span=period, adjust=False).mean()

        # ============ 6. Hurst Exponent (Mean Reversion vs Trending) ============

        # Simplified Hurst calculation (R/S method)
        window = 50
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
        downside_dev = downside_returns.rolling(window=20).std()
        result_df['downside_deviation_20'] = downside_dev
        for period in ewma_periods:
            result_df[f'downside_deviation_20_ewma{period}'] = downside_dev.ewm(span=period, adjust=False).mean()

        # Upside deviation (squeeze risk for shorts)
        upside_returns = returns.copy()
        upside_returns[upside_returns < 0] = 0
        upside_dev = upside_returns.rolling(window=20).std()
        result_df['upside_deviation_20'] = upside_dev
        for period in ewma_periods:
            result_df[f'upside_deviation_20_ewma{period}'] = upside_dev.ewm(span=period, adjust=False).mean()

        # ============ 8. Cross-Timeframe Volatility Ratios ============

        # Vol ratio: short-term vs medium-term
        vol_6 = returns.rolling(window=6).std()
        vol_24 = returns.rolling(window=24).std()
        vol_ratio = vol_6 / (vol_24 + 1e-9)
        result_df['vol_ratio_6_24'] = vol_ratio
        for period in ewma_periods:
            result_df[f'vol_ratio_6_24_ewma{period}'] = vol_ratio.ewm(span=period, adjust=False).mean()

        # ============ 9. Fragility Metrics (NO EWMA) ============

        # Fragility Ratio (Modified Amihud): |Return| / Volume
        fragility_raw = abs(returns) / (result_df['volume'] + 1e-9)
        fragility_ewma24 = fragility_raw.ewm(span=24, adjust=False).mean()
        fragility_ratio = fragility_raw / (fragility_ewma24 + 1e-9)
        result_df['fragility_ratio'] = fragility_ratio  # NO EWMA

        # Shock Ratio: (High - Low) / EWMA_Vol
        vol_ewma6 = vol_20.ewm(span=6, adjust=False).mean()
        shock_ratio = (result_df['high'] - result_df['low']) / (vol_ewma6 + 1e-9)
        result_df['shock_ratio'] = shock_ratio  # NO EWMA

        # Desperation Metric (CLV): (Close - Low) / (High - Low) - 0.5
        hl_range = result_df['high'] - result_df['low']
        clv = ((result_df['close'] - result_df['low']) / (hl_range + 1e-9)) - 0.5
        result_df['desperation_clv'] = clv  # NO EWMA

        # Volume-Vol Divergence (Trap Detector)
        vol_1h = vol_20
        vol_ewma6_for_div = vol_1h.ewm(span=6, adjust=False).mean()
        volume_1h = result_df['volume']
        volume_ewma6 = volume_1h.ewm(span=6, adjust=False).mean()

        vol_component = vol_1h / (vol_ewma6_for_div + 1e-9)
        volume_component = volume_1h / (volume_ewma6 + 1e-9)
        divergence = vol_component / (volume_component + 1e-9)
        result_df['volume_vol_divergence'] = divergence  # NO EWMA

        # ============ 10. ENHANCED: Multi-Scale Volatility Analysis ============
        # Use multiple volatility windows to separate regimes by persistence
        # Windows aligned to trading duration: 1h (immediate), 3h (trade-matched), 6h (structural context)

        vol_1h = returns.rolling(window=1).std()
        vol_3h = returns.rolling(window=3).std()
        vol_6h = returns.rolling(window=6).std()

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

        # ============ 11. ENHANCED: Price Action Features ============

        # True Range (ATR building block)
        tr1 = result_df['high'] - result_df['low']
        tr2 = abs(result_df['high'] - result_df['close'].shift(1))
        tr3 = abs(result_df['low'] - result_df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result_df['true_range'] = true_range
        atr_14 = true_range.rolling(window=14).mean()
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
        price_change_1 = result_df['close'].diff()
        price_change_6 = result_df['close'].diff(6)
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
            volume_ma_20 = volume.rolling(window=20).mean()
            volume_ma_6 = volume.rolling(window=6).mean()

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

        tprint_info(
            f"✅ Generated {len([c for c in result_df.columns if c not in df.columns])} risk features "
            f"({len([c for c in result_df.columns if 'ewma' in c])} with EWMA smoothing)"
        )

        return result_df

    def _compute_risk_targets(
        self,
        aligned_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute risk-based composite target from 4 components.

        Risk Target Composition (scaled individually then weighted):
            - Forward Vol 1h (30%): Short-term tactical risk
            - Forward Vol 4h (20%): Medium-term persistent risk
            - Tail Risk Probability (30%): Crash protection via CVaR
            - Vol Acceleration (20%): Regime transition detection

        Each component is winsorized and scaled with RobustScaler before weighting.
        """
        from src.features_common.transforms.scaling_normalization import (
            winsorized_zscore_normalize,
            robust_normalize
        )

        df = aligned_df.copy()
        required_cols = ["close", "high", "low"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Aligned dataset must contain {required_cols}, missing: {missing}")

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # Winsorization parameters (0.05-0.95 for regime preservation)
        winsorize_lower = float(config.get("risk_winsorize_lower_quantile", 0.05))
        winsorize_upper = float(config.get("risk_winsorize_upper_quantile", 0.95))

        tprint_info("🎯 Computing 4-component risk target...")

        # =============== Component 1: Forward Vol 1h (30%) ===============
        # Calculate forward 1h realized volatility (rolling std of returns)
        returns_1h = np.log(close / close.shift(1))
        fwd_vol_1h = returns_1h.shift(-1).rolling(window=1).std()
        df["risk_fwd_vol_1h_raw"] = fwd_vol_1h

        # =============== Component 2: Forward Vol 4h (20%) ===============
        # Calculate forward 4h realized volatility
        fwd_vol_4h = returns_1h.shift(-4).rolling(window=4).std()
        df["risk_fwd_vol_4h_raw"] = fwd_vol_4h

        # =============== Component 3: Tail Risk Probability (30%) ===============
        # Use CVaR (Expected Shortfall) as tail risk proxy
        # Calculate forward 1h worst-case tail losses
        window_tail = int(config.get("risk_tail_window", 20))
        confidence_level = float(config.get("risk_cvar_confidence", 0.05))

        def rolling_cvar(returns_series, window, alpha=0.05):
            """Calculate rolling CVaR (Expected Shortfall)."""
            cvar_vals = []
            for i in range(len(returns_series)):
                if i < window:
                    cvar_vals.append(np.nan)
                else:
                    window_returns = returns_series.iloc[i-window:i].dropna()
                    if len(window_returns) > 0:
                        var_threshold = window_returns.quantile(alpha)
                        tail_losses = window_returns[window_returns <= var_threshold]
                        cvar = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
                        cvar_vals.append(abs(cvar))  # Absolute value for risk magnitude
                    else:
                        cvar_vals.append(np.nan)

            return pd.Series(cvar_vals, index=returns_series.index)

        fwd_cvar = rolling_cvar(returns_1h.shift(-1), window=window_tail, alpha=confidence_level)
        df["risk_tail_cvar_raw"] = fwd_cvar

        # =============== Component 4: Vol Acceleration (20%) ===============
        # Rate of change in volatility (vol momentum)
        vol_current = returns_1h.rolling(window=6).std()
        vol_prev = returns_1h.rolling(window=6).std().shift(6)
        vol_accel = (vol_current - vol_prev) / (vol_prev + 1e-9)
        df["risk_vol_acceleration_raw"] = vol_accel.shift(-1)  # Forward-looking

        # =============== ENHANCED: Additional Regime Signals for Better Separation ===============
        # These signals capture different aspects of regime dynamics for orthogonal separation

        # Vol Clustering: Autocorrelation of vol changes (persistent vol = turbulent regime)
        vol_changes = vol_current.pct_change().fillna(0)
        vol_clustering = vol_changes.rolling(window=12).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        df["risk_vol_clustering_raw"] = vol_clustering.fillna(0)

        # Vol Persistence: How many consecutive bars had vol > MA
        vol_ma = vol_current.rolling(window=6).mean()
        vol_persistent = (vol_current > vol_ma).astype(int).rolling(window=6).sum()
        df["risk_vol_persistence_raw"] = vol_persistent / 6.0  # Normalized to [0, 1]

        # Price-Vol Efficiency (Separation metric): Price move per unit of vol
        price_moves = close.diff().abs()
        price_vol_efficiency = price_moves / (vol_current + 1e-9)
        df["risk_price_vol_efficiency_raw"] = price_vol_efficiency

        # Tail Risk Density: Concentration of tail losses
        window_tail_density = int(config.get("risk_tail_window", 20))
        tail_density = []
        for i in range(len(returns_1h)):
            if i < window_tail_density:
                tail_density.append(0.0)
            else:
                window_returns = returns_1h.iloc[i-window_tail_density:i].dropna()
                if len(window_returns) > 0:
                    percentile_5 = window_returns.quantile(0.05)
                    tail_count = (window_returns <= percentile_5).sum()
                    tail_density.append(float(tail_count) / len(window_returns))
                else:
                    tail_density.append(0.0)
        df["risk_tail_density_raw"] = tail_density

        # =============== NEW: Regime-Specific Divergence Features ===============
        # These features capture regime transitions and regime characteristics

        # Vol-Return Correlation: Rolling correlation between vol and returns
        # Negative in trending regimes, positive in mean-reverting/crash regimes
        vol_return_corr = returns_1h.rolling(window=20).corr(vol_current)
        df["vol_return_correlation_raw"] = vol_return_corr.fillna(0)

        # Cross-Timeframe Vol Ratio: Local vs global volatility
        # High ratio = locally elevated vol (regime transition)
        vol_short = returns_1h.rolling(window=6).std()
        vol_long = returns_1h.rolling(window=24).std()
        vol_cross_ratio = vol_short / (vol_long + 1e-9)
        df["vol_cross_timeframe_ratio_raw"] = vol_cross_ratio

        # Drawdown Duration: How long current drawdown has lasted
        # Long durations indicate persistent distress regimes
        cumulative_returns = (1 + returns_1h).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-9)

        # Calculate duration of current drawdown
        drawdown_duration = pd.Series(0, index=df.index)
        in_drawdown = drawdown < -0.01  # 1% threshold
        duration_counter = 0
        for i in range(len(in_drawdown)):
            if in_drawdown.iloc[i]:
                duration_counter += 1
                drawdown_duration.iloc[i] = duration_counter
            else:
                duration_counter = 0
        df["drawdown_duration_raw"] = drawdown_duration

        # Vol Regime Momentum: Rate of change in vol regime
        # Captures whether vol is accelerating or decelerating
        vol_roc = vol_current.pct_change(periods=3).fillna(0)
        df["vol_regime_momentum_raw"] = vol_roc

        # Return Skewness-Vol Interaction: Captures crash vs euphoria regimes
        # High vol + negative skew = crash regime
        # High vol + positive skew = euphoria/recovery regime
        skew_20 = returns_1h.rolling(window=20).skew().fillna(0)
        skew_vol_interaction = skew_20 * vol_current
        df["skew_vol_interaction_raw"] = skew_vol_interaction

        # Winsorize each component individually
        risk_components = {
            "risk_fwd_vol_1h_raw": 0.30,
            "risk_fwd_vol_4h_raw": 0.20,
            "risk_tail_cvar_raw": 0.30,
            "risk_vol_acceleration_raw": 0.20,
        }

        tprint_info(f"🔧 Winsorizing risk components at {winsorize_lower:.1%} and {winsorize_upper:.1%} quantiles")

        scaled_components = []
        for comp_name, weight in risk_components.items():
            if comp_name not in df.columns:
                tprint_warning(f"Skipping missing component: {comp_name}")
                continue

            comp_data = df[comp_name].copy()

            # Winsorize
            comp_clean = comp_data.dropna()
            if len(comp_clean) > 0:
                lower_bound = comp_clean.quantile(winsorize_lower)
                upper_bound = comp_clean.quantile(winsorize_upper)
                comp_data = comp_data.clip(lower=lower_bound, upper=upper_bound)
                tprint_info(f"  ✓ {comp_name}: clipped to [{lower_bound:.6f}, {upper_bound:.6f}]")

            # Robust scale each component with winsorization built-in
            try:
                comp_scaled = winsorized_zscore_normalize(
                    comp_data,
                    ddof=0,
                    lower_quantile=winsorize_lower,
                    upper_quantile=winsorize_upper
                )
                df[f"{comp_name}_scaled"] = comp_scaled

                # Weight the scaled component
                comp_weighted = comp_scaled * weight
                scaled_components.append(comp_weighted)

                tprint_info(f"  ✓ {comp_name}: weight={weight:.1%}, mean={comp_scaled.mean():.4f}, std={comp_scaled.std():.4f}")
            except Exception as e:
                tprint_warning(f"Failed to scale {comp_name}: {e}")
                continue

        # Combine weighted components
        if len(scaled_components) == 0:
            raise ValueError("No valid risk components could be computed")

        risk_target_raw = pd.concat(scaled_components, axis=1).sum(axis=1)
        df["risk_target_raw"] = risk_target_raw

        # Apply MinMaxScaler to get final target in [0, 1] range for KDE binning
        scaler = MinMaxScaler(feature_range=(0, 1))
        risk_target_scaled = risk_target_raw.dropna()
        if len(risk_target_scaled) > 0:
            risk_target_normalized = scaler.fit_transform(risk_target_scaled.values.reshape(-1, 1)).flatten()
            df.loc[risk_target_scaled.index, "risk_target"] = risk_target_normalized
        else:
            df["risk_target"] = np.nan

        # Drop rows with missing risk target
        before = len(df)
        df = df.dropna(subset=["risk_target"])
        dropped = before - len(df)
        if dropped > 0:
            tprint_warning(f"Dropped {dropped} rows with NaN risk target")

        tprint_info(
            f"🎯 Risk target dataset shape: {df.shape} "
            f"(target range: [{df['risk_target'].min():.4f}, {df['risk_target'].max():.4f}], "
            f"mean: {df['risk_target'].mean():.4f})"
        )

        return df

    def _train_risk_model(
        self,
        risk_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[Any, Optional[pd.Series], str, Dict[str, Any], Dict[str, Any]]:
        """Train XGBoost regression model to predict risk targets with monotonic constraints.

        Returns:
            - model: Trained XGBoost model
            - scores: Risk predictions on full dataset
            - pred_col_name: Name of prediction column
            - training_metrics: Dict of training performance metrics
            - feature_pipeline_artifacts: Dict containing scaler and feature names
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required for risk model training") from e

        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            mean_squared_error = None
            mean_absolute_error = None
            r2_score = None

        df = risk_df.copy()
        if "risk_target" not in df.columns:
            raise ValueError("risk_target column not found in dataset")

        df = df.dropna(subset=["risk_target"])
        if df.empty:
            raise ValueError("No valid samples for risk model training after dropping NaNs")

        y = df["risk_target"]

        # Select numeric features, excluding risk target and intermediate risk components
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [
            col
            for col in numeric_df.columns
            if col not in ["risk_target", "risk_target_raw"]
            and not col.startswith("risk_")
            and not col.startswith("alpha_")
        ]

        if not feature_cols:
            raise ValueError("No numeric features available for risk model training")

        X = numeric_df[feature_cols]

        min_samples = int(config.get("risk_min_samples", 200))
        if len(X) < max(min_samples, 20):
            raise ValueError(
                f"Insufficient samples for risk model training: {len(X)} < {min_samples}"
            )

        train_frac = float(config.get("risk_train_fraction", 0.8))
        train_frac = min(max(train_frac, 0.5), 0.95)
        split_idx = int(len(X) * train_frac)
        split_idx = max(min(split_idx, len(X) - 1), 1)

        # Chronological split to preserve temporal ordering
        X_train_raw, y_train = X.iloc[:split_idx].copy(), y.iloc[:split_idx]
        X_val_raw, y_val = X.iloc[split_idx:].copy(), y.iloc[split_idx:]

        # Robust scaling with winsorization
        outlier_threshold = float(config.get("risk_outlier_threshold", 3.0))
        normalizer_config: Dict[str, Any] = {
            "default_strategy": "robust",
            "auto_select": False,
            "handle_outliers": True,
            "outlier_threshold": outlier_threshold,
            "use_vectorbt": False,
        }
        scaler = ScalingNormalizer(normalizer_config)

        X_train_scaled = scaler.fit_transform(X_train_raw, strategy="robust")
        X_val_scaled = scaler.transform(X_val_raw)
        X_scaled_full = scaler.transform(X)

        # Apply EWMA temporal smoothing on scaled features (periods: 2, 6)
        use_ewm_features = bool(config.get("risk_use_ewm_features", True))
        ewma_periods = [2, 6]  # Fixed periods for risk features

        if use_ewm_features:
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
                X_train = X_features_full.iloc[:split_idx].copy()
                X_val = X_features_full.iloc[split_idx:].copy()
                X_scaled_full = X_features_full
                extended_feature_names = feature_names_seq
            else:
                X_train = X_train_scaled
                X_val = X_val_scaled
                extended_feature_names = list(X_scaled_full.columns)
        else:
            X_train = X_train_scaled
            X_val = X_val_scaled
            extended_feature_names = list(X_scaled_full.columns)

        training_metrics: Dict[str, Any] = {}
        training_metrics["scaling_strategy"] = "robust"
        training_metrics["risk_outlier_threshold"] = outlier_threshold
        training_metrics["risk_use_ewm_features"] = use_ewm_features
        training_metrics["risk_ewm_periods"] = ewma_periods

        # Prepare feature pipeline artifacts
        feature_pipeline_artifacts: Dict[str, Any] = {
            "feature_names": extended_feature_names,
            "scaler": scaler,
            "normalizer_config": normalizer_config,
        }

        # Define monotonic constraints for risk features
        # +1 means feature increases risk, -1 means feature decreases risk, 0 means no constraint
        monotone_constraints = {}
        for feat in extended_feature_names:
            feat_lower = feat.lower()
            # Risk-increasing features (+1 monotonic constraint)
            if any(keyword in feat_lower for keyword in [
                'vol', 'volatility', 'garch', 'cvar', 'drawdown', 'jump',
                'expansion', 'acceleration', 'parkinson', 'garman', 'rogers', 'yang',
                'fragility', 'shock', 'tail', 'skew', 'kurtosis', 'divergence'
            ]):
                monotone_constraints[feat] = 1
            # No strong prior for other features
            else:
                monotone_constraints[feat] = 0

        # Convert monotone constraints to list format for XGBoost
        monotone_constraints_list = [monotone_constraints.get(feat, 0) for feat in extended_feature_names]

        tprint_info(f"🔒 Monotonic constraints: {sum(c == 1 for c in monotone_constraints_list)} risk-increasing features")

        # XGBoost base parameters (optimized for risk modeling)
        base_params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'n_jobs': -1,

            # Structural constraints
            'max_depth': int(config.get("risk_max_depth", 4)),
            'min_child_weight': int(config.get("risk_min_child_weight", 20)),

            # Learning dynamics
            'learning_rate': float(config.get("risk_learning_rate", 0.05)),
            'n_estimators': int(config.get("risk_n_estimators", 1000)),

            # Randomness (anti-overfitting)
            'subsample': float(config.get("risk_subsample", 0.7)),
            'colsample_bytree': float(config.get("risk_colsample_bytree", 0.8)),

            # Regularization
            'gamma': float(config.get("risk_gamma", 1.0)),
            'reg_alpha': float(config.get("risk_reg_alpha", 0.5)),
            'reg_lambda': float(config.get("risk_reg_lambda", 1.0)),

            # Monotonic constraints
            'monotone_constraints': monotone_constraints_list,

            'random_state': int(config.get("risk_random_state", 42)),
        }

        # Hierarchical HPO (optional, config-gated)
        enable_hpo = bool(config.get("risk_enable_hpo", False))
        best_params = base_params.copy()

        if enable_hpo:
            try:
                # Define parameter groups (2–3 parameters per group)
                param_groups = [
                    create_param_group(
                        name="structure",
                        params={
                            "max_depth": {"type": "int", "low": 3, "high": 6},
                            "min_child_weight": {"type": "int", "low": 10, "high": 100},
                            "n_estimators": {"type": "int", "low": 500, "high": 1500},
                        },
                        priority=1,
                        description="Tree depth, leaf size, and capacity",
                    ),
                    create_param_group(
                        name="regularization",
                        params={
                            "gamma": {"type": "float", "low": 0.1, "high": 5.0},
                            "reg_alpha": {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
                            "reg_lambda": {"type": "float", "low": 0.1, "high": 5.0},
                        },
                        priority=2,
                        depends_on=["structure"],
                        description="Regularization strength",
                    ),
                    create_param_group(
                        name="sampling",
                        params={
                            "subsample": {"type": "float", "low": 0.5, "high": 0.9},
                            "colsample_bytree": {"type": "float", "low": 0.5, "high": 0.9},
                        },
                        priority=3,
                        depends_on=["regularization"],
                        description="Row/feature subsampling ratios",
                    ),
                ]

                # Initialize XGBoost model for objective evaluation
                base_model = xgb.XGBRegressor(**base_params)

                # Configure hierarchical optimizer (use neg MAE so direction is maximize)
                hpo_cv_folds = int(config.get("risk_hpo_cv_folds", 3))
                hpo_rounds = int(config.get("risk_hpo_rounds", 1))
                hpo_final_trials = int(config.get("risk_hpo_final_trials", 20))
                hpo_enable_final = bool(config.get("risk_hpo_enable_final_refinement", False))

                optimizer = HierarchicalParameterOptimizer(
                    param_groups=param_groups,
                    objective_func=default_objective_function,
                    cv_folds=hpo_cv_folds,
                    scoring_metric="neg_mean_absolute_error",
                    direction="maximize",
                    n_rounds=hpo_rounds,
                    enable_final_refinement=hpo_enable_final,
                    final_refinement_trials=hpo_final_trials,
                    cache_dir=None,
                    random_state=int(config.get("risk_random_state", 42)),
                    verbose=bool(config.get("risk_hpo_verbose", False)),
                    use_custom_balanced_score=False,
                )

                # Run optimization (use holdout validation when available)
                X_train_np = X_train.values if hasattr(X_train, "values") else X_train
                y_train_np = y_train.values if hasattr(y_train, "values") else y_train
                X_val_np = X_val.values if hasattr(X_val, "values") else X_val
                y_val_np = y_val.values if hasattr(y_val, "values") else y_val

                hpo_result = optimizer.optimize(
                    X_train=X_train_np,
                    y_train=y_train_np,
                    X_val=X_val_np,
                    y_val=y_val_np,
                    model=base_model,
                    initial_params=base_params,
                )

                # Hierarchical optimizer maximizes negative MAE; convert back to MAE for reporting
                best_neg_mae = float(hpo_result.best_score)
                best_mae = float(-best_neg_mae) if np.isfinite(best_neg_mae) else float("nan")

                best_params.update(hpo_result.best_params or {})
                training_metrics["risk_hpo_best_score"] = best_mae
                training_metrics["risk_hpo_best_params"] = hpo_result.best_params
                training_metrics["risk_hpo_used"] = True
                training_metrics["risk_hpo_total_trials"] = int(hpo_result.total_trials)
                tprint_info(f"✅ Hierarchical HPO completed: best MAE = {best_mae:.6f}")
            except Exception as hpo_exc:
                tprint_warning(f"Hierarchical HPO failed; proceeding with default params: {hpo_exc}")
                training_metrics["risk_hpo_used"] = False
        else:
            training_metrics["risk_hpo_used"] = False

        # Train final XGBoost model
        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='mae',
            early_stopping_rounds=int(config.get("risk_early_stopping_rounds", 50)),
            verbose=False
        )

        # Evaluate on train set
        train_pred = model.predict(X_train)
        if mean_squared_error:
            training_metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        if mean_absolute_error:
            training_metrics["train_mae"] = float(mean_absolute_error(y_train, train_pred))
        if r2_score:
            training_metrics["train_r2"] = float(r2_score(y_train, train_pred))

        # Evaluate on validation set
        if len(X_val) > 0:
            val_pred = model.predict(X_val)
            if mean_squared_error:
                training_metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            if mean_absolute_error:
                training_metrics["val_mae"] = float(mean_absolute_error(y_val, val_pred))
            if r2_score:
                training_metrics["val_r2"] = float(r2_score(y_val, val_pred))

            # Calculate residual standard deviation (sigma) for probabilistic inference
            residuals = y_val.values - val_pred
            sigma = float(np.std(residuals))
            training_metrics["val_residual_sigma"] = sigma
            tprint_info(f"📊 Validation residual σ = {sigma:.6f} (for probabilistic inference)")

        # Get predictions on full dataset
        full_raw_pred = model.predict(X_scaled_full)

        # Calibrate predictions to use full [0, 1] range (MinMaxScaler on predictions)
        pred_scaler = MinMaxScaler(feature_range=(0, 1))
        full_scores_calibrated = pred_scaler.fit_transform(full_raw_pred.reshape(-1, 1)).flatten()

        scores = pd.Series(full_scores_calibrated, index=df.index, name="risk_pred_score")
        pred_col_name = "risk_pred_score"
        training_metrics["model_type"] = "xgboost_regression"
        training_metrics["n_features"] = len(extended_feature_names)
        training_metrics["n_train_samples"] = len(X_train)
        training_metrics["n_val_samples"] = len(X_val)

        tprint_info(
            f"🤖 Trained XGBoost risk model on {len(X_train)} train / {len(X_val)} val samples "
            f"with {len(extended_feature_names)} features"
        )

        # Store calibration scaler in artifacts
        feature_pipeline_artifacts["prediction_scaler"] = pred_scaler

        return model, scores, pred_col_name, training_metrics, feature_pipeline_artifacts

    # ========================================================================
    # NEW: XGBoost Multi-Class Classifier Approach for Regime Detection
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

        within_cvs_array = np.array(within_cvs)
        regime_sizes_array = np.array(regime_sizes)
        weighted_cv = np.average(within_cvs_array, weights=regime_sizes_array)

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

        if min_pct < 0.05 or max_pct > 0.45:
            score *= 0.5  # Heavy penalty for extreme imbalance

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
        cooling_rate: float = 0.995
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

        tprint_info(f"🔥 Starting Simulated Annealing (100% risk CV): initial_score={current_score:.4f}")

        for iteration in range(max_iterations):
            # Propose modification: flip 1-2% of samples to neighboring regime
            candidate_labels = current_labels.copy()
            n_flips = max(1, int(0.015 * len(candidate_labels)))
            flip_indices = np.random.choice(len(candidate_labels), size=n_flips, replace=False)

            for idx in flip_indices:
                current_regime = candidate_labels[idx]
                # Flip to neighboring regime (maintain ordinality)
                neighbors = []
                if current_regime > 0:
                    neighbors.append(current_regime - 1)
                if current_regime < n_regimes - 1:
                    neighbors.append(current_regime + 1)

                if neighbors:
                    candidate_labels[idx] = np.random.choice(neighbors)

            # Evaluate candidate
            candidate_score = self._calculate_regime_quality_score(
                candidate_labels, risk_features, None
            )

            # Accept or reject (Metropolis criterion)
            delta = candidate_score - current_score
            if delta > 0:
                # Always accept improvements
                current_labels = candidate_labels
                current_score = candidate_score
                accept_count += 1

                if current_score > best_score:
                    best_labels = current_labels.copy()
                    best_score = current_score
            elif np.random.random() < np.exp(delta / temperature):
                # Sometimes accept worse solutions (escape local optima)
                current_labels = candidate_labels
                current_score = candidate_score
                accept_count += 1

            # Cool down
            temperature *= cooling_rate

            # Progress logging
            if iteration % 100 == 0 or iteration == max_iterations - 1:
                accept_rate = accept_count / (iteration + 1)
                tprint_info(
                    f"  SA iter {iteration}/{max_iterations}: "
                    f"score={current_score:.4f}, best={best_score:.4f}, "
                    f"temp={temperature:.4f}, accept_rate={accept_rate:.2%}"
                )

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
        risk_features_cols = [
            'risk_fwd_vol_1h_raw_scaled',
            'risk_fwd_vol_4h_raw_scaled',
            'risk_tail_cvar_raw_scaled',
            'risk_vol_acceleration_raw_scaled',
            'vol_clustering_raw',
            'vol_persistence_raw',
            'tail_density_raw',
            'vol_return_correlation_raw',
            'vol_cross_timeframe_ratio_raw',
            'drawdown_duration_raw',
            'price_vol_efficiency_raw',
            'vol_regime_momentum_raw',
            'skew_vol_interaction_raw',
        ]

        # Filter to available columns
        available_risk_cols = [c for c in risk_features_cols if c in risk_df.columns]
        risk_features = risk_df[available_risk_cols].copy()

        tprint_info(f"📊 Using {len(available_risk_cols)} RAW features (no smoothing)")

        # Remove NaNs
        valid_mask = risk_features.notna().all(axis=1)
        risk_features_clean = risk_features[valid_mask]

        tprint_info(f"  Valid samples: {len(risk_features_clean)}/{len(risk_df)}")

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

        # ========== STEP 5: GMM Initialization ==========
        from sklearn.mixture import GaussianMixture

        tprint_info(f"🎯 Initializing {n_regimes} regimes with GMM...")

        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            n_init=20,
            max_iter=200,
            random_state=42
        )
        gmm.fit(risk_features_clean)
        initial_labels = gmm.predict(risk_features_clean)

        # Rank regimes by average risk level (0 = lowest, n-1 = highest)
        regime_means = []
        for regime_id in range(n_regimes):
            regime_mask = initial_labels == regime_id
            regime_mean = risk_features_clean[regime_mask].mean().mean()
            regime_means.append(regime_mean)

        regime_ranking = np.argsort(regime_means)
        label_mapping = {old: new for new, old in enumerate(regime_ranking)}
        initial_labels = np.array([label_mapping[lbl] for lbl in initial_labels])

        initial_score = self._calculate_regime_quality_score(
            initial_labels, risk_features_clean, None
        )
        tprint_info(f"  GMM initialization: score={initial_score:.4f}")

        # ========== STEP 6: Simulated Annealing Refinement ==========
        use_sa_refinement = bool(config.get("risk_use_sa_refinement", True))

        if use_sa_refinement:
            refined_labels, refined_score = self._refine_labels_simulated_annealing(
                initial_labels=initial_labels,
                risk_features=risk_features_clean,
                forward_returns=None,  # NOT USED - 100% risk optimization
                n_regimes=n_regimes,
                max_iterations=int(config.get("risk_sa_iterations", 500)),
                initial_temp=float(config.get("risk_sa_initial_temp", 1.0)),
                cooling_rate=float(config.get("risk_sa_cooling_rate", 0.995))
            )
            final_labels = refined_labels
            final_score = refined_score
        else:
            final_labels = initial_labels
            final_score = initial_score

        # ========== STEP 7: Calculate Final Metrics ==========
        metrics = self._calculate_regime_quality_metrics(
            final_labels, risk_features_clean, None
        )

        # Expand labels back to full dataframe
        full_labels = np.full(len(risk_df), -1, dtype=int)
        full_labels[valid_mask] = final_labels

        # Store feature selection artifacts
        metrics['selected_features'] = list(risk_features_clean.columns)
        metrics['umap_reducer'] = umap_reducer
        metrics['n_features_used'] = len(risk_features_clean.columns)

        tprint_success(
            f"✅ Created {n_regimes} regime labels (NO temporal smoothing):\n"
            f"   Risk CV Ratio={metrics['risk_cv_ratio']:.3f}, "
            f"Wasserstein={metrics['wasserstein_distance']:.3f}, "
            f"KL Divergence={metrics['kl_divergence']:.3f}\n"
            f"   Regime Distribution: {metrics['regime_distribution']}"
        )

        return full_labels, metrics

    def _train_regime_classifier(
        self,
        risk_df: pd.DataFrame,
        regime_labels: np.ndarray,
        config: Dict[str, Any]
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

        # XGBoost Classifier Parameters
        n_regimes = int(regime_labels.max() + 1)

        params = {
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
            'colsample_bytree': float(config.get("risk_classifier_colsample_bytree", 0.8)),
            'gamma': float(config.get("risk_classifier_gamma", 2.0)),
            'reg_alpha': float(config.get("risk_classifier_reg_alpha", 1.0)),
            'reg_lambda': float(config.get("risk_classifier_reg_lambda", 2.0)),

            # Monotonic constraints
            'monotone_constraints': monotone_constraints,

            # Evaluation
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,

            'random_state': 42,
        }

        # Train classifier
        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict probabilities on full dataset
        regime_probs = model.predict_proba(X_full)

        # Calculate training metrics
        y_val_pred = model.predict(X_val)
        y_val_probs = model.predict_proba(X_val)

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_log_loss = log_loss(y_val, y_val_probs)

        training_metrics = {
            'val_accuracy': float(val_accuracy),
            'val_log_loss': float(val_log_loss),
            'n_regimes': n_regimes,
            'feature_names': list(X_full.columns),
            'scaler': scaler,
            'monotone_constraints': monotone_constraints,
            'n_features': len(X_full.columns),
        }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_full.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        training_metrics['feature_importance'] = feature_importance.to_dict('records')

        tprint_success(
            f"✅ XGBoost Classifier trained:\n"
            f"   Val Accuracy={val_accuracy:.3f}, Val LogLoss={val_log_loss:.4f}\n"
            f"   Best Iteration={model.best_iteration}, Features={len(X_full.columns)}"
        )

        # Classification report
        report = classification_report(
            y_val, y_val_pred,
            target_names=[f'Regime_{i}' for i in range(n_regimes)],
            output_dict=True,
            zero_division=0
        )
        training_metrics['classification_report'] = report

        # Log per-regime accuracy
        for regime_id in range(n_regimes):
            regime_report = report.get(f'Regime_{regime_id}', {})
            precision = regime_report.get('precision', 0)
            recall = regime_report.get('recall', 0)
            f1 = regime_report.get('f1-score', 0)
            tprint_info(
                f"  Regime {regime_id}: Precision={precision:.3f}, "
                f"Recall={recall:.3f}, F1={f1:.3f}"
            )

        # Expand probabilities to full dataframe
        full_probs = np.full((len(risk_df), n_regimes), np.nan)
        full_probs[valid_mask] = regime_probs

        return model, full_probs, training_metrics

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
        target_regimes = int(config.get("risk_target_regimes", 4))
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
        safety_confirmation_bars = int(config.get("risk_safety_confirmation_bars", 3))

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
