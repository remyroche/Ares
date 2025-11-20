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
from typing import Any, Dict, Optional, Tuple, List
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
            # Use the same context as RollingHMMRegimeDiscoveryStep when loading
            # its artifacts (model='regime').
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime",
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
            market_data, market_source = self.load_market_data_or_fail(
                {
                    **config,
                    "timeframe": regime_timeframe,
                },
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
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
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

            # ------------------------------------------------------------------
            # 4) Compute risk targets (4-component composite)
            # ------------------------------------------------------------------
            risk_df = self._compute_risk_targets(aligned_df, config)

            if risk_df.empty:
                raise ValueError("Risk dataset is empty after target construction")

            # ------------------------------------------------------------------
            # 5) Train XGBoost risk model and derive risk regimes
            # ------------------------------------------------------------------
            model = None
            risk_scores = None
            training_metrics: Dict[str, Any] = {}
            regime_stats_df: Optional[pd.DataFrame] = None
            model_path: Optional[str] = None
            regime_stats_path: Optional[str] = None
            regime_col_name: Optional[str] = None
            risk_quality_metrics: Optional[ClusterQualityMetrics] = None
            risk_quality_path: Optional[str] = None
            feature_pipeline_artifacts: Optional[Dict[str, Any]] = None
            feature_pipeline_path: Optional[str] = None

            try:
                model, risk_scores, pred_col_name, training_metrics, feature_pipeline_artifacts = self._train_risk_model(
                    risk_df,
                    config,
                )
                if risk_scores is not None:
                    risk_df[pred_col_name] = risk_scores
                    risk_df["risk_score_continuous"] = risk_scores
                    risk_df, regime_stats_df, regime_col_name = self._assign_alpha_regimes(
                        risk_df,
                        risk_scores,
                        config,
                    )
            except ImportError as xgb_err:
                tprint_warning(
                    f"XGBoost not available; skipping risk model training: {xgb_err}"
                )
            except Exception as model_exc:
                tprint_warning(
                    f"Risk model training failed; continuing with targets only: {model_exc}"
                )

            if risk_scores is None or regime_col_name is None:
                try:
                    forward_ret_cols = [
                        col
                        for col in risk_df.columns
                        if col.startswith("alpha_forward_return_")
                    ]
                    if forward_ret_cols:
                        fallback_series = risk_df[forward_ret_cols[0]].astype(float)
                        valid_count = fallback_series.notna().sum()
                        if valid_count >= 3:
                            pred_col_name_fallback = f"risk_fallback_score_{forward_ret_cols[0].split('_')[-1]}"
                            risk_scores = fallback_series
                            risk_df[pred_col_name_fallback] = risk_scores
                            risk_df["risk_score_continuous"] = risk_scores
                            risk_df, regime_stats_df, regime_col_name = self._assign_alpha_regimes(
                                risk_df,
                                risk_scores,
                                config,
                            )
                            training_metrics["risk_fallback_used"] = True
                            training_metrics["risk_fallback_source"] = "forward_return"
                        else:
                            tprint_warning(
                                f"Not enough samples ({valid_count}) for fallback risk regime assignment"
                            )
                except Exception as fallback_exc:
                    tprint_warning(
                        f"Risk fallback regime assignment failed; proceeding without regimes: {fallback_exc}"
                    )

            # ------------------------------------------------------------------
            # 6) Extract and save regime thresholds for production use
            # ------------------------------------------------------------------
            regime_thresholds: Optional[Dict[str, Any]] = None
            regime_thresholds_path: Optional[str] = None

            if risk_scores is not None and regime_col_name is not None and regime_col_name in risk_df.columns:
                try:
                    regime_thresholds = self._extract_and_save_regime_thresholds(
                        alpha_scores=alpha_scores,
                        regime_labels=risk_df[regime_col_name],
                        regime_col_name=regime_col_name,
                        symbol=symbol,
                        config=config,
                    )

                    if regime_thresholds and "extraction_error" not in regime_thresholds:
                        # Save thresholds as artifact
                        try:
                            regime_thresholds_path = self._save_artifact(
                                data=regime_thresholds,
                                artifact_name="hmm_alpha_regime_thresholds_1h",
                                artifact_type="model",
                                data_category="config",
                                metadata={
                                    "symbol": symbol,
                                    "exchange": exchange,
                                    "timeframe": regime_timeframe,
                                    "n_regimes": regime_thresholds.get("n_regimes", 0),
                                },
                            )
                            tprint_info(f"💾 Saved regime thresholds artifact: {regime_thresholds_path}")
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
                risk_quality_metrics, risk_quality_path = self._assess_alpha_regime_quality(
                    risk_df=risk_df,
                    regime_col=regime_col_name,
                    config=config,
                )
            except Exception as quality_exc:
                tprint_warning(f"Alpha regime quality assessment failed: {quality_exc}")

            alpha_to_save = risk_df.reset_index().rename(columns={risk_df.index.name or "index": "timestamp"})

            tprint_info(
                f"💾 Saving alpha training dataset with shape {alpha_to_save.shape} "
                f"to versioned HDF5 store"
            )
            training_data_path = self._save_artifact(
                data=alpha_to_save,
                artifact_name="hmm_alpha_training_data_1h",
                artifact_type="data",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )

            # Save trained model if available
            if model is not None:
                try:
                    tprint_info("💾 Saving LightGBM alpha model via artifact router")
                    model_path = self._save_artifact(
                        data=model,
                        artifact_name="hmm_alpha_model_1h",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "model_type": "lightgbm",
                        },
                    )
                except Exception as save_model_exc:
                    tprint_warning(f"Failed to save alpha model artifact: {save_model_exc}")

            # Persist feature pipeline (feature list + scaler state) for live usage
            if feature_pipeline_artifacts is not None:
                try:
                    feature_pipeline_path = self._save_artifact(
                        data=feature_pipeline_artifacts,
                        artifact_name="hmm_alpha_feature_pipeline_1h",
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
                        artifact_name="hmm_alpha_regime_stats_1h",
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
                # Alpha quality markdown + CSV reports via ClusterQualityAssessor
                if risk_quality_metrics is not None:
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
        labels_df: pd.DataFrame,
        probs_df: Optional[pd.DataFrame],
        economic_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Align market data, labels, probabilities, and economic features."""
        # Ensure all indices are datetime and sorted
        def _prepare(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            return df.sort_index()

        market_data = _prepare(market_data)
        labels_df = _prepare(labels_df)

        frames = [
            market_data[[col for col in market_data.columns if col.lower() in {"open", "high", "low", "close", "volume"}]].rename(
                columns=lambda c: c.lower()
            ),
            labels_df,
        ]

        if probs_df is not None and not probs_df.empty:
            probs_df = _prepare(probs_df)
            frames.append(probs_df)

        if economic_df is not None and not economic_df.empty:
            economic_df = _prepare(economic_df)
            frames.append(economic_df)

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

        # Winsorization parameters
        winsorize_lower = float(config.get("risk_winsorize_lower_quantile", 0.01))
        winsorize_upper = float(config.get("risk_winsorize_upper_quantile", 0.99))

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

        # Optuna HPO (optional, config-gated)
        enable_hpo = bool(config.get("risk_enable_hpo", False))
        best_params = base_params.copy()

        if enable_hpo:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    params = {
                        **base_params,
                        'max_depth': trial.suggest_int('max_depth', 3, 6),
                        'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
                        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                    }

                    model_trial = xgb.XGBRegressor(**params)
                    model_trial.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )

                    y_pred_val = model_trial.predict(X_val)
                    mae = mean_absolute_error(y_val, y_pred_val) if mean_absolute_error else np.mean(np.abs(y_val - y_pred_val))
                    return mae

                n_trials = int(config.get("risk_hpo_trials", 30))
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                best_params.update(study.best_params)
                training_metrics["risk_hpo_best_score"] = float(study.best_value)
                training_metrics["risk_hpo_best_params"] = study.best_params
                training_metrics["risk_hpo_used"] = True
                tprint_info(f"✅ HPO completed: best MAE = {study.best_value:.6f}")
            except Exception as hpo_exc:
                tprint_warning(f"HPO failed; proceeding with default params: {hpo_exc}")
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
            else:
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

                tprint_info(f"✅ Information Coefficient (IC) calculated: Pearson r={ic_pearson_corr:.4f}, Spearman r={ic_spearman_corr:.4f}")
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

                    tprint_info(f"✅ LGBM importance calculated for {len(feature_names)} features")
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

                        tprint_info(f"✅ Permutation importance calculated in {perm_results.get('execution_time', 0.0):.3f}s")
                except Exception as perm_err:
                    tprint_warning(f"Permutation importance calculation failed: {perm_err}")

            # 4. Improved mRMR for redundancy analysis
            if IMPROVED_MRMR_AVAILABLE and bool(config.get("alpha_enable_mrmr_analysis", True)):
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

                    tprint_info(f"✅ mRMR selected {feature_analysis_results['mrmr_n_selected']} features (ratio: {feature_analysis_results['mrmr_n_selected']/len(feature_names):.2%})")
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

                    tprint_info(f"✅ Learning curve analysis completed: {lc_result.learning_rate} learning rate, {lc_result.overfitting_risk} overfitting risk")
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

                    tprint_info(f"✅ SHAP importance calculated for {len(feature_names)} features")
                except Exception as shap_err:
                    tprint_warning(f"SHAP analysis failed (SHAP may have issues with this data): {shap_err}")

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
