"""
ML Path Regime Step (HMM Architecture with Temporal Dynamics)

This step constructs Price Path Geometry regime labels using Hidden Markov Models (HMM)
with computational optimizations and temporal structure learning.

Primary Goal: Distinguish between different price path structures (roughness, linearity,
directness, shape, steepness, timing, morphology) using HMM with learned regime transitions.

Key Features:
- Uses ONLY 7 path geometry features (no PnL chase)
- HMM-Direct inference with full covariance (NO XGBoost student classifier)
- Temporal structure learning via transition matrix
- Strided training for computational efficiency (every 10th point)
- GMM warm-start initialization for faster convergence
- LiveHMM wrapper for O(1) production updates
- Zero risk of class collapse
- Probabilistic regime assignments with temporal consistency

Computational Optimizations:
1. Strided Training: Train on every 10th data point to reduce training time
2. GMM Injection: Pre-compute cluster centers with fast GMM, inject into HMM
3. Warm Start Retraining: Use GMM means to initialize HMM parameters
4. O(1) Live Updates: LiveHMM class for production with pre-extracted matrices

Responsibilities:
- Load 15m OHLCV market data
- Generate 7 core path geometry features:
  * hurst_exponent_path (Roughness)
  * path_trend_r2 (Linearity)
  * path_efficiency_return_3h (Directness)
  * quadratic_fit_curvature (Shape/Bend)
  * linear_reg_slope (Steepness)
  * path_center_of_gravity (Timing)
  * body_range_ratio (Morphology)
- Create optimal regime labels using HMM with temporal structure
- Calculate temporal metrics (transition matrix, flip-flop rate, duration)
- Calculate forward returns and Sharpe ratios per regime (15m, 1h, 3h)
- Persist HMM model and LiveHMM wrapper for live trading
- Generate comprehensive regime quality and temporal reports
- Save regime outputs to versioned_artifacts
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
from scipy.stats import spearmanr, pearsonr, gaussian_kde, multivariate_normal
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM

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
from src.feature_generation.categories.entropy import PermutationEntropyGenerator

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


class LiveHMM:
    """
    Production-grade O(1) HMM state update for live trading.

    Extracts transition matrix, means, and covariances from a trained HMM model
    and provides efficient forward algorithm updates without retraining.

    Usage:
        # After training your HMM
        live_hmm = LiveHMM(trained_hmm_model)

        # In live trading loop
        new_observation = np.array([...])  # Single observation vector
        regime_probs = live_hmm.update(new_observation)
        current_regime = np.argmax(regime_probs)
    """

    def __init__(self, trained_model: GaussianHMM):
        """
        Initialize LiveHMM from a trained hmmlearn GaussianHMM model.

        Args:
            trained_model: Fitted GaussianHMM instance
        """
        # Extract parameters from trained model
        self.n_components = trained_model.n_components
        self.startprob = trained_model.startprob_
        self.transmat = trained_model.transmat_
        self.means = trained_model.means_
        self.covars = trained_model.covars_
        self.covariance_type = trained_model.covariance_type

        # Initialize state probabilities (uniform or use startprob)
        self.state_probs = self.startprob.copy()

        # Pre-compute multivariate normal distributions for each state
        self._setup_distributions()

    def _setup_distributions(self):
        """Pre-compute scipy multivariate normal distributions for each state."""
        self.distributions = []
        for i in range(self.n_components):
            mean = self.means[i]
            if self.covariance_type == 'full':
                cov = self.covars[i]
            elif self.covariance_type == 'diag':
                cov = np.diag(self.covars[i])
            elif self.covariance_type == 'tied':
                cov = self.covars
            elif self.covariance_type == 'spherical':
                cov = np.eye(len(mean)) * self.covars[i]
            else:
                raise ValueError(f"Unknown covariance type: {self.covariance_type}")

            self.distributions.append(multivariate_normal(mean=mean, cov=cov))

    def update(self, observation: np.ndarray) -> np.ndarray:
        """
        O(1) forward algorithm update for a single new observation.

        Args:
            observation: 1D array of shape (n_features,) representing the new observation

        Returns:
            Updated state probabilities: array of shape (n_components,)
        """
        observation = np.asarray(observation).flatten()

        # Compute emission probabilities for this observation
        emission_probs = np.array([
            dist.pdf(observation) for dist in self.distributions
        ])

        # Prevent numerical underflow
        emission_probs = np.maximum(emission_probs, 1e-300)

        # Forward step: P(S_t | obs_1:t) = P(obs_t | S_t) * sum_i P(S_t | S_{t-1}=i) * P(S_{t-1}=i | obs_1:t-1)
        self.state_probs = emission_probs * (self.transmat.T @ self.state_probs)

        # Normalize to get probabilities
        prob_sum = self.state_probs.sum()
        if prob_sum > 0:
            self.state_probs /= prob_sum
        else:
            # Fallback to uniform if underflow
            self.state_probs = np.ones(self.n_components) / self.n_components

        return self.state_probs.copy()

    def predict(self, observation: np.ndarray) -> int:
        """
        Update and return the most likely regime.

        Args:
            observation: 1D array of shape (n_features,)

        Returns:
            Integer regime index (0 to n_components-1)
        """
        probs = self.update(observation)
        return int(np.argmax(probs))

    def reset(self):
        """Reset state probabilities to initial distribution."""
        self.state_probs = self.startprob.copy()


class MLPathRegimeStep(BaseStep):
    """Pipeline step to construct risk-based regime labels from 15m Rolling HMM regimes."""

    def __init__(self, step_name: str = "ml_path_regime_step"):
        """Initialize the ML Risk Regime step with versioned artifacts enabled."""
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLPathRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data_15m = None
        self._cached_market_source_15m = None
        self._cached_market_cache_key_15m = None
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the alpha label construction from 15m HMM regimes.

        Expected config keys (minimum):
            - symbol: Trading symbol (e.g., 'ETHUSDT')
            - exchange: Exchange name (e.g., 'binance')
            - regime_timeframe: Timeframe used for HMM regimes (default: '15m')
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
                "risk_hpo_balance_strength": 12.0,
                "risk_hpo_min_regime_pct": 0.05,
                "risk_hpo_max_regime_pct": 0.65,
                "risk_iterative_prune_weight_quadrant": 0.55,
                "risk_iterative_prune_weight_balance": 0.45,
            }
            for k, v in risk_config_defaults.items():
                config.setdefault(k, v)

            path_config_defaults: Dict[str, Any] = {
                "path_ker_window_bars": 48,  # 12 * 4
                "path_trend_r2_window_bars": 96,  # 24 * 4
                "path_permutation_entropy_window": 320,  # 80 * 4
                "path_permutation_embedding_dim": 3,
                "path_permutation_delay": 1,
                "path_fractal_window_bars": 384,  # 96 * 4
                "path_hurst_window_bars": 384,  # 96 * 4
                "path_efficiency_high_threshold": 0.6,
                "path_efficiency_drop_threshold": 0.05,
            }
            for k, v in path_config_defaults.items():
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
            # 6) GMM-Direct Regime Detection (No XGBoost Student)
            # ------------------------------------------------------------------
            regime_labels: Optional[np.ndarray] = None
            training_metrics: Dict[str, Any] = {}
            regime_stats_df: Optional[pd.DataFrame] = None
            regime_stats_path: Optional[str] = None
            regime_col_name: Optional[str] = None
            risk_quality_metrics: Optional[ClusterQualityMetrics] = None
            risk_quality_path: Optional[str] = None

            tprint_info("=" * 80)
            tprint_info("🎯 HMM REGIME DETECTION (Temporal Structure Learning)")
            tprint_info("=" * 80)

            try:
                # 6a) Create optimal regime labels using HMM with GMM warm-start
                tprint_info("📊 Creating optimal regime labels via HMM...")
                regime_labels_optimized, label_metrics = self._create_optimal_regime_labels(
                    risk_df=risk_df,
                    config=config
                )

                # Store label quality metrics
                training_metrics['label_quality'] = label_metrics
                regime_labels = regime_labels_optimized

                # 6b) Calculate forward returns and Sharpe ratios per regime
                try:
                    tprint_info("📊 Calculating forward returns and Sharpe ratios per regime...")
                    forward_returns_sharpe = self._calculate_forward_returns_and_sharpe(
                        df=risk_df,
                        regime_labels=regime_labels_optimized,
                        horizons=[1, 4, 12]  # 15m, 1h, 3h
                    )
                    training_metrics['forward_returns_sharpe'] = forward_returns_sharpe

                    # Log forward returns and Sharpe ratios
                    tprint_info("=" * 80)
                    tprint_info("📈 FORWARD RETURNS & SHARPE RATIOS PER REGIME:")
                    tprint_info("=" * 80)
                    for horizon_label, regime_stats in forward_returns_sharpe.items():
                        tprint_info(f"\n  Horizon: {horizon_label}")
                        for regime_id, stats in regime_stats.items():
                            tprint_info(
                                f"    Regime {regime_id}: "
                                f"Return={stats['mean_return']:.6f}, "
                                f"Sharpe={stats['sharpe_ratio']:.4f} "
                                f"(Ann: {stats['sharpe_annualized']:.4f}), "
                                f"n={stats['count']}"
                            )
                    tprint_info("=" * 80)

                except Exception as fwd_exc:
                    tprint_warning(f"Failed to calculate forward returns/Sharpe (non-fatal): {fwd_exc}")
                    training_metrics['forward_returns_sharpe'] = {}

                # 6c) Generate HMM-based feature impact analysis
                try:
                    tprint_info("📊 Analyzing HMM feature impact on regime separation...")
                    gmm_feature_impact = self._analyze_gmm_feature_impact(
                        risk_df=risk_df,
                        regime_labels=regime_labels_optimized,
                        label_metrics=label_metrics,
                        config=config,
                    )
                    training_metrics['gmm_feature_impact'] = gmm_feature_impact

                    # Save feature impact report
                    impact_report_path = self._generate_gmm_feature_impact_report(
                        gmm_feature_impact=gmm_feature_impact,
                        label_metrics=label_metrics,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=regime_timeframe,
                    )
                    tprint_success(f"📄 HMM feature impact report saved: {impact_report_path}")

                except Exception as impact_exc:
                    tprint_warning(f"Failed to generate HMM feature impact analysis (non-fatal): {impact_exc}")

                try:
                    tprint_info("📊 Generating detailed WCoV regime quality report...")
                    wcov_csv_path, wcov_md_path = self._generate_regime_wcov_report(
                        df=risk_df,
                        regime_labels=regime_labels_optimized,
                        config=config,
                        classifier_metrics=training_metrics,  # Pass training metrics with forward returns
                    )
                    if wcov_md_path:
                        tprint_success(
                            f"📄 WCoV regime quality report saved: {wcov_md_path} (CSV: {wcov_csv_path})"
                        )
                    else:
                        tprint_success(
                            f"📄 WCoV regime quality CSV report saved: {wcov_csv_path}"
                        )
                except Exception as wcov_exc:
                    tprint_warning(
                        f"Failed to generate WCoV regime quality report (non-fatal): {wcov_exc}"
                    )

                # 6d) Add HMM labels to dataframe
                risk_df['risk_regime'] = regime_labels_optimized
                regime_col_name = 'risk_regime'

                # Extract temporal metrics from label_metrics
                temporal_metrics = label_metrics.get('temporal_metrics', {})

                tprint_success(
                    f"=" * 80 + "\n"
                    f"✅ HMM REGIME DETECTION COMPLETE (Temporal Structure Learned)\n"
                    f"=" * 80 + "\n"
                    f"  Label Quality Score: {label_metrics.get('quality_score', 0):.3f}\n"
                    f"  Risk CV Ratio: {label_metrics.get('risk_cv_ratio', 0):.3f}\n"
                    f"  Wasserstein Distance: {label_metrics.get('wasserstein_distance', 0):.3f}\n"
                    f"  KL Divergence: {label_metrics.get('kl_divergence', 0):.3f}\n"
                    f"  Avg Duration: {temporal_metrics.get('avg_duration_bars', 0):.2f} bars\n"
                    f"  Flip-Flop Rate: {temporal_metrics.get('flip_flop_rate', 0):.4f}\n"
                    f"  Regime Distribution: {label_metrics.get('regime_distribution', {})}\n"
                    f"=" * 80
                )

                # Calculate regime statistics using a Path-based score proxy
                # Prefer KER and path straightness; fall back to a constant score.
                if "path_ker_3h" in risk_df.columns:
                    path_scores = risk_df["path_ker_3h"]
                elif "path_ker_6h" in risk_df.columns:
                    path_scores = risk_df["path_ker_6h"]
                elif "path_trend_r2" in risk_df.columns:
                    path_scores = risk_df["path_trend_r2"]
                else:
                    path_scores = pd.Series(1.0, index=risk_df.index)

                # Align labels and scores on the same index
                regime_labels_series = risk_df[regime_col_name]
                common_index = regime_labels_series.index.intersection(path_scores.index)

                regime_stats_df = self._calculate_regime_statistics(
                    risk_df.loc[common_index],
                    regime_labels_series.loc[common_index],
                    path_scores.loc[common_index],
                )

            except Exception as exc:
                tprint_error(f"❌ HMM regime detection failed: {exc}")
                import traceback
                traceback.print_exc()
                # Fast-fail: raise exception to avoid producing inconsistent artifacts
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
                model="regime_path",
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
                wcov_diag_df = self._compute_feature_wcov_diagnostics(
                    df=risk_df,
                    regime_col_pred=regime_col_name,
                    regime_col_teacher="risk_regime_training_label",
                    top_k=None,
                )

                if wcov_diag_df is not None and not wcov_diag_df.empty:
                    ts_diag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    symbol_d = str(config.get("symbol", symbol or ""))
                    exchange_d = str(config.get("exchange", exchange or ""))
                    regime_tf_d = str(
                        config.get("regime_timeframe", config.get("timeframe", "15m"))
                    )

                    diag_path = (
                        f"outcomes/ml_path_feature_wcov_diagnostics_"
                        f"{symbol_d or 'UNKNOWN'}_{exchange_d or 'UNKNOWN'}_{regime_tf_d}_{ts_diag}.csv"
                    )
                    wcov_diag_df.to_csv(diag_path, index=False)
                    tprint_info(
                        f"💾 Saved Path feature WCoV diagnostics (teacher vs predicted regimes): {diag_path}"
                    )
            except Exception as diag_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Feature WCoV diagnostics computation failed (non-fatal): {diag_exc}"
                )

            risk_to_save = risk_df.reset_index().rename(columns={risk_df.index.name or "index": "timestamp"})

            tprint_info(
                f"💾 Saving path training dataset with shape {risk_to_save.shape} "
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

            # Save path regimes on the native regime timeframe (expected to be 15m)
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
                    f"Failed to save ML path regimes on timeframe {regime_timeframe}: {probs_exc}"
                )

            # Note: GMM model is already saved in _create_optimal_regime_labels
            # No XGBoost model to save in GMM-Direct architecture

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
                    tprint_warning(f"Failed to save path feature pipeline artifact: {save_fp_exc}")

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

                # Regime stats CSV for Path regimes (XGBoost classifier over Path labels)
                if regime_stats_df is not None and not regime_stats_df.empty:
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # XGBoost-derived per-regime statistics for the ML Path
                        # regimes, saved under a clear name to differentiate from
                        # legacy HMM alpha stats.
                        stats_csv_name = f"ml_path_regime_xgb_stats_{symbol}_{ts}.csv"
                        stats_csv_path = f"outcomes/{stats_csv_name}"
                        regime_stats_df.to_csv(stats_csv_path)
                        tprint_info(
                            f"📊 Saved ML Path XGB regime statistics CSV: {stats_csv_path}"
                        )
                    except Exception as stats_csv_exc:  # pragma: no cover - defensive
                        tprint_warning(
                            f"Failed to save ML Path XGB regime statistics CSV (non-fatal): {stats_csv_exc}"
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
        tprint_info("🎯 Generating path-oriented features...")

        result_df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in result_df.columns]
        if missing:
            tprint_warning(f"Missing columns for feature generation: {missing}")
            return result_df

        # Calculate base 1h returns
        returns = np.log(result_df['close'] / result_df['close'].shift(1))
        result_df['returns_1h'] = returns

        # Minimal 3h return and Sharpe-like ratio used downstream and by Path alpha
        try:
            return_3h = returns.rolling(window=12).sum()
            result_df['return_3h'] = return_3h

            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_dev = downside_returns.rolling(window=80).std()
            sharpe_like_3h = return_3h / (downside_dev + 1e-9)
            result_df['sharpe_like_3h'] = sharpe_like_3h
        except Exception as rr_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Path return / Sharpe-like feature generation failed (non-fatal): {rr_exc}"
            )

        try:
            ker_window_main = int(config.get("path_ker_window_bars", 3))
            ker_windows = sorted({ker_window_main, 24})
            for n in ker_windows:
                if n > 1:
                    price_change_n = (result_df["close"] - result_df["close"].shift(n)).abs()
                    path_length_n = result_df["close"].diff().abs().rolling(window=n, min_periods=2).sum()
                    ker_series = price_change_n / (path_length_n + 1e-9)
                    result_df[f"path_ker_{n}h"] = ker_series
        except Exception as ker_exc:  # pragma: no cover - defensive
            tprint_warning(f"Path efficiency (KER) feature generation failed (non-fatal): {ker_exc}")

        try:
            ker_col = f"path_ker_{ker_window_main}h"
            if ker_col in result_df.columns and "return_3h" in result_df.columns:
                ker_series_eff = result_df[ker_col]
                ret_3h_eff = result_df["return_3h"]
                result_df["path_efficiency_return_3h"] = ker_series_eff * ret_3h_eff
                result_df["path_directional_eff_3h"] = np.sign(ret_3h_eff) * ker_series_eff
        except Exception as path_eff_exc:  # pragma: no cover - defensive
            tprint_warning(
                f"Path efficiency-return feature generation failed (non-fatal): {path_eff_exc}"
            )

        try:
            body = (result_df["close"] - result_df["open"]).abs()
            range_bar = (result_df["high"] - result_df["low"]).replace(0, np.nan)
            brr = body / (range_bar + 1e-9)
            result_df["body_range_ratio"] = brr
        except Exception as brr_exc:  # pragma: no cover - defensive
            tprint_warning(f"Body-to-range ratio feature generation failed (non-fatal): {brr_exc}")

        try:
            range_bar = (result_df["high"] - result_df["low"]).replace(0, np.nan)
            overlap_high = np.minimum(result_df["high"], result_df["high"].shift(1))
            overlap_low = np.maximum(result_df["low"], result_df["low"].shift(1))
            overlap = (overlap_high - overlap_low).clip(lower=0.0)
            traffic = overlap / (range_bar + 1e-9)
            result_df["traffic_overlap"] = traffic
            result_df["traffic_overlap_3h"] = traffic.rolling(window=12, min_periods=1).mean()
        except Exception as traffic_exc:  # pragma: no cover - defensive
            tprint_warning(f"Traffic/overlap feature generation failed (non-fatal): {traffic_exc}")

        try:
            ker_col = f"path_ker_{ker_window_main}h"
            if ker_col in result_df.columns:
                trend_window = int(config.get("path_trend_r2_window_bars", ker_window_main * 2))

                def _rolling_r2(arr: np.ndarray) -> float:
                    mask = np.isfinite(arr)
                    if mask.sum() < 3:
                        return np.nan
                    y = arr[mask]
                    x = np.arange(len(y), dtype=float)
                    if y.std() == 0.0 or x.std() == 0.0:
                        return 0.0
                    corr = np.corrcoef(x, y)[0, 1]
                    if not np.isfinite(corr):
                        return 0.0
                    return float(corr * corr)

                r2_series = result_df["close"].rolling(
                    window=trend_window,
                    min_periods=3,
                ).apply(_rolling_r2, raw=True)
                result_df["path_trend_r2"] = r2_series
        except Exception as r2_exc:  # pragma: no cover - defensive
            tprint_warning(f"Path trend R2 feature generation failed (non-fatal): {r2_exc}")

        try:
            perm_window = int(config.get("path_permutation_entropy_window", 20))
            embedding_dim = int(config.get("path_permutation_embedding_dim", 3))
            delay = int(config.get("path_permutation_delay", 1))
            pe_gen = PermutationEntropyGenerator(
                window=perm_window,
                embedding_dim=embedding_dim,
                delay=delay,
            )
            pe_series = pe_gen._generate_feature(result_df[["close"]].copy())
            result_df["path_permutation_entropy"] = pe_series
        except Exception as pe_exc:  # pragma: no cover - defensive
            tprint_warning(f"Permutation entropy feature generation failed (non-fatal): {pe_exc}")

        try:
            fd_window = int(config.get("path_fractal_window_bars", 24))

            def _fractal_window(seq: np.ndarray) -> float:
                seq = seq[np.isfinite(seq)]
                if len(seq) < 10:
                    return 1.0

                # Katz fractal dimension on cumulative returns path
                path = np.cumsum(seq)
                diffs = np.diff(path)
                if len(diffs) == 0:
                    return 1.0

                total_length = float(np.sum(np.abs(diffs)))
                if total_length <= 0.0:
                    return 1.0

                max_dist = float(np.max(np.abs(path - path[0])))
                if max_dist <= 0.0:
                    return 1.0

                n = float(len(path))
                fd = np.log10(n) / (np.log10(n) + np.log10(max_dist / total_length))
                return float(max(1.0, min(2.0, fd)))

            fd_series = returns.rolling(
                window=fd_window,
                min_periods=10,
            ).apply(_fractal_window, raw=True)
            result_df["path_fractal_dimension"] = fd_series
        except Exception as fd_exc:  # pragma: no cover - defensive
            tprint_warning(f"Fractal dimension feature generation failed (non-fatal): {fd_exc}")

        try:
            hurst_window = int(config.get("path_hurst_window_bars", 24))

            def _hurst_window(seq: np.ndarray) -> float:
                if len(seq) < 10:
                    return 0.5
                n = len(seq)
                mean_seq = float(np.mean(seq))
                deviations = seq - mean_seq
                cumulative = np.cumsum(deviations)
                r = float(np.max(cumulative) - np.min(cumulative))
                s = float(np.std(seq))
                if s == 0.0 or r <= 0.0:
                    return 0.5
                rs = r / s
                if rs <= 0.0:
                    return 0.5
                return float(np.log(rs) / np.log(n))

            hurst_series = returns.rolling(
                window=hurst_window,
                min_periods=10,
            ).apply(_hurst_window, raw=True)
            result_df["hurst_exponent_path"] = hurst_series
        except Exception as hurst_exc:  # pragma: no cover - defensive
            tprint_warning(f"Path Hurst exponent feature generation failed (non-fatal): {hurst_exc}")

        try:
            ker_col = f"path_ker_{ker_window_main}h"
            if ker_col in result_df.columns and "return_3h" in result_df.columns:
                ker_series = result_df[ker_col]
                ker_diff = ker_series.diff()
                path_trend_up = (result_df["return_3h"] > 0).astype(int)
                eff_high_thr = float(config.get("path_efficiency_high_threshold", 0.6))
                eff_drop_thr = float(config.get("path_efficiency_drop_threshold", 0.05))
                eff_high = (ker_series >= eff_high_thr).astype(int)
                eff_dropping = (ker_diff <= -eff_drop_thr).astype(int)
                alpha_state = np.zeros(len(result_df), dtype=int)
                hold_mask = (path_trend_up == 1) & (eff_high == 1) & (eff_dropping == 0)
                tp_mask = (path_trend_up == 1) & (eff_dropping == 1)
                alpha_state[hold_mask.values] = 1
                alpha_state[tp_mask.values] = 2
                result_df["path_trend_up"] = path_trend_up
                result_df["path_efficiency_high"] = eff_high
                result_df["path_efficiency_dropping"] = eff_dropping
                result_df["path_alpha_state"] = alpha_state
        except Exception as alpha_exc:  # pragma: no cover - defensive
            tprint_warning(f"Path alpha helper feature generation failed (non-fatal): {alpha_exc}")

        # Restrict outputs to base OHLCV + Path/return features; drop legacy risk/volatility columns
        base_cols = [c for c in df.columns]
        path_keep = [
            "returns_1h",
            "path_ker_3h",
            "path_ker_6h",
            "path_efficiency_return_3h",
            "path_directional_eff_3h",
            "body_range_ratio",
            "traffic_overlap",
            "traffic_overlap_3h",
            "path_permutation_entropy",
            "path_fractal_dimension",
            "hurst_exponent_path",
            "path_trend_r2",
            "path_trend_up",
            "path_efficiency_high",
            "path_efficiency_dropping",
            "path_alpha_state",
        ]
        keep_cols = base_cols + [c for c in path_keep if c in result_df.columns]
        keep_cols = list(dict.fromkeys(keep_cols))
        result_df = result_df[keep_cols]

        tprint_info(
            f"✅ Generated {len([c for c in result_df.columns if c not in df.columns])} path features"
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

        # Core 7 Path Geometry Features (user-specified)
        # These are the ONLY features used for GMM regime detection
        primary_risk_cols = [
            "hurst_exponent_path",         # Roughness
            "path_trend_r2",               # Linearity
            "path_efficiency_return_3h",   # Directness
            "quadratic_fit_curvature",     # Shape/Bend
            "linear_reg_slope",            # Steepness
            "path_center_of_gravity",      # Timing
            "body_range_ratio",            # Morphology
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
        # Use float32 to reduce memory footprint before clustering/SA
        exec_mode_dtype = str(config.get("execution_mode", "")).lower()
        if exec_mode_dtype != "blank":
            risk_features_clean = risk_features_clean.astype("float32")
        else:
            # In blank mode keep higher precision to improve numerical stability for GMM
            risk_features_clean = risk_features_clean.astype("float32")

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

        # ========== STEP 5: HMM Initialization with GMM Warm Start ==========
        # Validate that we still have a non-empty, numeric feature matrix
        if gmm_sa_features.empty or gmm_sa_features.shape[1] == 0:
            raise ValueError(
                "Risk feature matrix for HMM is empty after cleaning/correlation "
                "filtering; reduce filtering or check feature generation."
            )

        from sklearn.mixture import GaussianMixture

        exec_mode_hmm = str(config.get("execution_mode", "")).lower()

        # ===== OPTIMIZATION 1: Strided Training (Downsample every 10th point) =====
        hmm_stride = int(config.get("hmm_stride", 10))
        tprint_info(f"🔄 Applying strided training: Using every {hmm_stride}th data point")

        strided_indices = np.arange(0, len(gmm_sa_features), hmm_stride)
        hmm_features_strided = gmm_sa_features.iloc[strided_indices].copy()

        tprint_info(
            f"  Strided data: {len(hmm_features_strided)} samples "
            f"(down from {len(gmm_sa_features)})"
        )

        # ===== OPTIMIZATION 2: GMM Pre-computation for HMM Initialization =====
        tprint_info(
            f"🚀 Step 1: Fast GMM pre-computation for HMM warm-start "
            f"(features: {hmm_features_strided.shape[1]})"
        )

        # Fast GMM with minimal iterations for initialization only
        gmm_for_init = GaussianMixture(
            n_components=n_regimes,
            covariance_type="diag",  # Fast diagonal for initialization
            n_init=5,
            max_iter=50,
            reg_covar=1e-3,
            random_state=42
        )

        gmm_start = time.time()
        gmm_for_init.fit(hmm_features_strided)
        gmm_duration = time.time() - gmm_start

        # Extract GMM means for HMM initialization
        gmm_means = gmm_for_init.means_
        gmm_labels_init = gmm_for_init.predict(hmm_features_strided)

        tprint_success(
            f"  ✅ GMM pre-computation completed in {gmm_duration:.2f}s "
            f"(warm-start initialization ready)"
        )

        # ===== OPTIMIZATION 3: HMM Training with GMM Injection =====
        tprint_info(
            f"🧠 Step 2: Training HMM with {n_regimes} regimes "
            f"(full covariance for geometric features)"
        )

        # HMM configuration optimized for 30m-3h trading timeframes
        hmm_n_iter = int(config.get("hmm_n_iter", 200))
        hmm_tol = float(config.get("hmm_tol", 1e-3))
        hmm_min_covar = float(config.get("hmm_min_covar", 0.001))

        # Initialize HMM with user-specified "Magic Number" geometry regimes
        hmm = GaussianHMM(
            n_components=n_regimes,          # The "Magic Number" for Geometry
            covariance_type="full",          # MANDATORY for geometric features
            n_iter=hmm_n_iter,               # High iteration count for convergence
            tol=hmm_tol,                     # Stricter tolerance
            init_params='stmc',              # Initialize all params (Start, Trans, Means, Covs)
            min_covar=hmm_min_covar,         # Floor to prevent instability
            random_state=42
        )

        # Inject GMM means into HMM (warm start)
        hmm.means_ = gmm_means.copy()
        hmm.init_params = 'stc'  # Only init start, trans, covars (means already set)

        tprint_info(
            f"  HMM configuration: n_components={n_regimes}, "
            f"covariance_type='full', n_iter={hmm_n_iter}, "
            f"tol={hmm_tol}, min_covar={hmm_min_covar}"
        )
        tprint_info(
            f"  🔥 GMM means injected into HMM for warm-start initialization"
        )

        # Convert to numpy array for HMM (requires shape: [n_samples, n_features])
        hmm_features_strided_array = hmm_features_strided.values

        # Reshape for HMM: needs [n_samples, n_features] and lengths array
        hmm_start = time.time()
        try:
            hmm.fit(
                hmm_features_strided_array,
                lengths=[len(hmm_features_strided_array)]  # Single sequence
            )

            # Get initial labels using trained HMM on full data
            hmm_full_array = gmm_sa_features.values
            initial_labels = hmm.predict(
                hmm_full_array,
                lengths=[len(hmm_full_array)]
            )
        except ValueError as hmm_exc:
            msg = str(hmm_exc)
            tprint_warning(
                f"HMM fit failed: {msg}. Retrying with increased min_covar "
                f"and float64 inputs for additional stability."
            )

            # Retry with more stable parameters
            hmm = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                init_params='stmc',
                min_covar=hmm_min_covar * 10.0,
                random_state=42
            )

            hmm_features_strided_retry = hmm_features_strided.astype("float64").values
            hmm_full_retry = gmm_sa_features.astype("float64").values

            hmm.fit(
                hmm_features_strided_retry,
                lengths=[len(hmm_features_strided_retry)]
            )
            initial_labels = hmm.predict(
                hmm_full_retry,
                lengths=[len(hmm_full_retry)]
            )
            gmm_sa_features = gmm_sa_features.astype("float64")

        hmm_duration = time.time() - hmm_start
        tprint_success(
            f"  ✅ HMM training completed in {hmm_duration:.2f}s"
        )

        # Extract transition matrix for reporting
        transition_matrix = hmm.transmat_

        tprint_info("=" * 80)
        tprint_info("📊 HMM TRANSITION MATRIX (Regime Dynamics):")
        tprint_info("=" * 80)
        for i in range(n_regimes):
            trans_probs = " ".join([f"{p:.3f}" for p in transition_matrix[i]])
            tprint_info(f"  Regime {i} → [{trans_probs}]")
        tprint_info("=" * 80)

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
        labels_for_score = initial_labels
        features_for_score = gmm_sa_features

        initial_score = self._calculate_regime_quality_score(
            labels_for_score, features_for_score, None
        )
        score_duration = time.time() - score_start
        tprint_info(
            f"  HMM initialization: score={initial_score:.4f} "
            f"(quality score computed in {score_duration:.2f}s)"
        )

        # ========== STEP 6: Simulated Annealing Refinement (Optional for HMM) ==========
        # Note: HMM already learns temporal structure through transition matrix,
        # so SA is less critical. Disabled by default for HMM.
        use_sa_refinement = bool(config.get("risk_use_sa_refinement", False))

        if use_sa_refinement:
            sa_flip_fraction = float(config.get("risk_sa_flip_fraction", 0.10))
            sa_neighborhood_mode = str(config.get("risk_sa_neighborhood_mode", "any"))
            exec_mode_sa = str(config.get("execution_mode", "")).lower()

            # Subsampling is now opt-in in all modes; default is enabled.
            sa_enable_subsampling = bool(
                config.get("risk_sa_enable_subsampling", True)
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
            tprint_info(
                "  SA refinement disabled; using pure GMM labels without "
                "additional optimization."
            )

        # ========== STEP 7: Calculate Final Metrics ==========
        metrics_features = risk_features_clean
        metrics_labels = final_labels
        metrics_enable_subsampling = bool(config.get("risk_metrics_enable_subsampling", True))
        metrics_max_samples = int(config.get("risk_metrics_max_samples", 8000))
        if metrics_enable_subsampling and len(metrics_features) > metrics_max_samples:
            try:
                rng_metrics = np.random.RandomState(
                    int(config.get("risk_metrics_subsample_random_state", 1337))
                )
            except Exception:
                rng_metrics = np.random.RandomState(1337)

            metrics_indices = rng_metrics.choice(
                len(metrics_features), size=metrics_max_samples, replace=False
            )
            metrics_features = metrics_features.iloc[metrics_indices]
            metrics_labels = metrics_labels[metrics_indices]

            tprint_info(
                f"  Metrics subsampling enabled: using {len(metrics_features)} samples "
                f"out of {len(risk_features_clean)} for regime quality metrics"
            )

        metrics = self._calculate_regime_quality_metrics(
            metrics_labels, metrics_features, None
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
                f"outcomes/ml_path_label_quality_"
                f"{symbol_q or 'UNKNOWN'}_{regime_tf_q}_{ts_q}.csv"
            )
            quality_df.to_csv(quality_path, index=False)
            tprint_info(f"💾 Saved Path label quality summary: {quality_path}")
        except Exception as quality_exc:  # pragma: no cover - defensive
            tprint_warning(f"Failed to persist risk label quality summary (non-fatal): {quality_exc}")

        # Expand labels back to full dataframe
        full_labels = np.full(len(risk_df), -1, dtype=int)
        full_labels[valid_mask] = final_labels

        teacher_probs_full: Optional[np.ndarray]
        teacher_probs_full = None
        try:
            # Use HMM score_samples to get log likelihood, then compute posteriors
            hmm_full_array = risk_features_clean.values
            posteriors = hmm.predict_proba(
                hmm_full_array,
                lengths=[len(hmm_full_array)]
            )
            if posteriors.shape[0] == valid_mask.sum():
                teacher_probs_full = np.zeros((len(risk_df), n_regimes), dtype=float)
                teacher_probs_full[valid_mask] = posteriors
        except Exception as resp_exc:
            tprint_warning(f"Teacher probability extraction from HMM failed; skipping distillation helpers: {resp_exc}")

        # Store feature selection artifacts
        metrics['selected_features'] = list(risk_features_clean.columns)
        metrics['umap_reducer'] = umap_reducer
        metrics['n_features_used'] = len(risk_features_clean.columns)
        if teacher_probs_full is not None:
            metrics['teacher_probs'] = teacher_probs_full

        # ========== STEP 7: Calculate Temporal Metrics ==========
        temporal_metrics = self._calculate_temporal_metrics(
            labels=final_labels,
            transition_matrix=transition_matrix,
            n_regimes=n_regimes
        )
        metrics['temporal_metrics'] = temporal_metrics

        # Log temporal metrics
        tprint_info("=" * 80)
        tprint_info("📊 TEMPORAL METRICS:")
        tprint_info("=" * 80)
        tprint_info(f"  Average Duration per Regime: {temporal_metrics['avg_duration_bars']:.2f} bars")
        tprint_info(f"  Flip-Flop Rate: {temporal_metrics['flip_flop_rate']:.4f}")
        tprint_info(f"  Regime Stability Score: {temporal_metrics['stability_score']:.4f}")
        tprint_info("=" * 80)

        # ========== STEP 8: Persist HMM Model for Direct Inference ==========
        # Save the optimized HMM model and LiveHMM wrapper for production O(1) updates
        try:
            import joblib
            from pathlib import Path

            symbol_hmm = str(config.get("symbol", "UNKNOWN"))
            timeframe_hmm = str(config.get("regime_timeframe", config.get("timeframe", "15m")))
            ts_hmm = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create regime metadata from final labels and features
            regime_metadata = {}
            for regime_id in range(n_regimes):
                regime_mask = final_labels == regime_id
                regime_count = np.sum(regime_mask)

                if regime_count > 0:
                    regime_features = risk_features_clean.iloc[regime_mask]
                    regime_means = regime_features.mean().to_dict()

                    regime_metadata[regime_id] = {
                        "name": f"Path Regime {regime_id}",
                        "description": f"Geometric path structure cluster {regime_id}",
                        "count": int(regime_count),
                        "percentage": float(regime_count / len(final_labels) * 100),
                        "feature_means": {k: float(v) for k, v in regime_means.items()},
                    }

            # Prepare model metadata
            model_metadata = {
                "symbol": symbol_hmm,
                "exchange": str(config.get("exchange", "UNKNOWN")),
                "timeframe": timeframe_hmm,
                "trained_at": ts_hmm,
                "model_type": "HMM",
                "n_regimes": n_regimes,
                "n_features": len(risk_features_clean.columns),
                "n_samples": len(final_labels),
                "risk_cv_ratio": float(metrics.get("risk_cv_ratio", 0.0)),
                "wasserstein_distance": float(metrics.get("wasserstein_distance", 0.0)),
                "covariance_type": "full",
                "hmm_n_iter": hmm_n_iter,
                "hmm_stride": hmm_stride,
                "transition_matrix": transition_matrix.tolist(),
                "temporal_metrics": temporal_metrics,
            }

            # 1. Save HMM-Direct detector (recommended for production)
            hmm_detector_path = f"versioned_artifacts/regime_models/path_hmm_{symbol_hmm}_{timeframe_hmm}_{ts_hmm}.pkl"
            Path(hmm_detector_path).parent.mkdir(parents=True, exist_ok=True)

            hmm_model_data = {
                "hmm_model": hmm,
                "feature_names": list(risk_features_clean.columns),
                "scaler": None,  # No scaling used in current implementation
                "regime_metadata": regime_metadata,
                "model_metadata": model_metadata,
                "transition_matrix": transition_matrix,
            }
            joblib.dump(hmm_model_data, hmm_detector_path)
            tprint_success(f"💾 Saved HMM-Direct detector: {hmm_detector_path}")

            # 2. Create and save LiveHMM wrapper for O(1) production updates
            live_hmm = LiveHMM(hmm)
            live_hmm_path = f"versioned_artifacts/regime_models/path_live_hmm_{symbol_hmm}_{timeframe_hmm}_{ts_hmm}.pkl"
            joblib.dump(live_hmm, live_hmm_path)
            tprint_success(f"💾 Saved LiveHMM wrapper (O(1) updates): {live_hmm_path}")

            # 2. Save Centroid-Based detector (simpler backup option)
            try:
                # Compute regime centroids and covariances
                regime_centroids_dict = {}
                regime_covariances_dict = {}

                for regime_id in range(n_regimes):
                    regime_mask = final_labels == regime_id
                    if np.sum(regime_mask) > 0:
                        regime_data = risk_features_clean.iloc[regime_mask]
                        regime_centroids_dict[regime_id] = regime_data.mean().values
                        regime_covariances_dict[regime_id] = regime_data.cov().values

                # Create centroids DataFrame
                centroids_df = pd.DataFrame(
                    regime_centroids_dict,
                    index=risk_features_clean.columns
                ).T

                centroid_detector_path = f"versioned_artifacts/regime_models/path_centroid_{symbol_gmm}_{timeframe_gmm}_{ts_gmm}.pkl"

                centroid_model_data = {
                    "regime_centroids": centroids_df,
                    "regime_covariances": regime_covariances_dict,
                    "feature_names": list(risk_features_clean.columns),
                    "scaler": None,
                    "regime_metadata": regime_metadata,
                    "distance_metric": "mahalanobis",
                }
                joblib.dump(centroid_model_data, centroid_detector_path)
                tprint_info(f"💾 Saved Centroid detector (backup): {centroid_detector_path}")

                metrics['hmm_detector_path'] = hmm_detector_path
                metrics['live_hmm_path'] = live_hmm_path
                metrics['centroid_detector_path'] = centroid_detector_path

            except Exception as centroid_exc:
                tprint_warning(f"Failed to create centroid detector (non-fatal): {centroid_exc}")
                metrics['hmm_detector_path'] = hmm_detector_path
                metrics['live_hmm_path'] = live_hmm_path
                metrics['centroid_detector_path'] = None

        except Exception as persist_exc:
            tprint_warning(f"Failed to persist HMM detectors (non-fatal): {persist_exc}")
            metrics['hmm_detector_path'] = None
            metrics['live_hmm_path'] = None
            metrics['centroid_detector_path'] = None

        tprint_success(
            f"✅ Created {n_regimes} HMM regime labels with temporal structure:\n"
            f"   Risk CV Ratio={metrics['risk_cv_ratio']:.3f}, "
            f"Wasserstein={metrics['wasserstein_distance']:.3f}, "
            f"KL Divergence={metrics['kl_divergence']:.3f}\n"
            f"   Avg Duration={temporal_metrics['avg_duration_bars']:.2f} bars, "
            f"Flip-Flop Rate={temporal_metrics['flip_flop_rate']:.4f}\n"
            f"   Regime Distribution: {metrics['regime_distribution']}"
        )

        return full_labels, metrics

    def _calculate_temporal_metrics(
        self,
        labels: np.ndarray,
        transition_matrix: np.ndarray,
        n_regimes: int
    ) -> Dict[str, Any]:
        """
        Calculate temporal metrics for regime analysis.

        Metrics calculated:
        - Average duration per regime (in bars)
        - Flip-flop rate (frequent regime changes)
        - Regime stability score (derived from transition matrix diagonals)
        - Transition probabilities summary

        Args:
            labels: Regime labels array
            transition_matrix: HMM transition matrix (n_regimes x n_regimes)
            n_regimes: Number of regimes

        Returns:
            Dictionary with temporal metrics
        """
        # Calculate regime durations
        regime_durations = []
        current_regime = labels[0]
        current_duration = 1

        for i in range(1, len(labels)):
            if labels[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = labels[i]
                current_duration = 1

        # Add last duration
        regime_durations.append(current_duration)

        avg_duration_bars = np.mean(regime_durations) if regime_durations else 0.0

        # Calculate flip-flop rate (regime changes / total samples)
        regime_changes = np.sum(labels[1:] != labels[:-1])
        flip_flop_rate = regime_changes / len(labels) if len(labels) > 0 else 0.0

        # Calculate stability score from transition matrix diagonal
        # Higher diagonal values = more likely to stay in same regime = more stable
        diagonal_probs = np.diag(transition_matrix)
        stability_score = np.mean(diagonal_probs)

        # Per-regime self-transition probabilities
        regime_stability = {}
        for regime_id in range(n_regimes):
            regime_stability[regime_id] = float(transition_matrix[regime_id, regime_id])

        # Duration statistics per regime
        regime_duration_stats = {}
        for regime_id in range(n_regimes):
            regime_mask = (labels == regime_id)
            regime_indices = np.where(regime_mask)[0]

            if len(regime_indices) == 0:
                regime_duration_stats[regime_id] = {
                    'mean_duration': 0.0,
                    'median_duration': 0.0,
                    'max_duration': 0,
                    'min_duration': 0,
                    'count': 0
                }
                continue

            # Calculate durations for this specific regime
            regime_durations_specific = []
            i = 0
            while i < len(regime_indices):
                start = regime_indices[i]
                duration = 1
                while (i + 1 < len(regime_indices) and
                       regime_indices[i + 1] == regime_indices[i] + 1):
                    duration += 1
                    i += 1
                regime_durations_specific.append(duration)
                i += 1

            regime_duration_stats[regime_id] = {
                'mean_duration': float(np.mean(regime_durations_specific)),
                'median_duration': float(np.median(regime_durations_specific)),
                'max_duration': int(np.max(regime_durations_specific)),
                'min_duration': int(np.min(regime_durations_specific)),
                'count': len(regime_durations_specific)
            }

        return {
            'avg_duration_bars': float(avg_duration_bars),
            'flip_flop_rate': float(flip_flop_rate),
            'stability_score': float(stability_score),
            'regime_changes': int(regime_changes),
            'total_samples': int(len(labels)),
            'regime_stability': regime_stability,
            'regime_duration_stats': regime_duration_stats,
            'transition_matrix': transition_matrix.tolist(),
        }

    def _calculate_forward_returns_and_sharpe(
        self,
        df: pd.DataFrame,
        regime_labels: np.ndarray,
        horizons: List[int] = [1, 4, 12]  # 15m, 1h, 3h for 15m data
    ) -> Dict[str, Any]:
        """
        Calculate forward returns and Sharpe ratios per regime at multiple horizons.

        Args:
            df: DataFrame with 'close' prices
            regime_labels: Regime labels aligned with df
            horizons: List of forward horizons in bars (default: [1, 4, 12] for 15m, 1h, 3h)

        Returns:
            Dictionary with forward returns and Sharpe ratios per regime
        """
        if 'close' not in df.columns:
            tprint_warning("No 'close' column found; skipping forward returns calculation")
            return {}

        close_prices = df['close'].values
        results = {}

        for horizon in horizons:
            horizon_name = f"{horizon}bar"
            if horizon == 1:
                horizon_label = "15m"
            elif horizon == 4:
                horizon_label = "1h"
            elif horizon == 12:
                horizon_label = "3h"
            else:
                horizon_label = f"{horizon}bars"

            # Calculate forward returns
            forward_returns = np.full(len(close_prices), np.nan)
            for i in range(len(close_prices) - horizon):
                forward_returns[i] = np.log(close_prices[i + horizon] / close_prices[i])

            # Calculate per-regime statistics
            regime_stats = {}
            for regime_id in np.unique(regime_labels):
                if regime_id < 0:
                    continue

                regime_mask = (regime_labels == regime_id) & np.isfinite(forward_returns)
                regime_returns = forward_returns[regime_mask]

                if len(regime_returns) < 2:
                    regime_stats[int(regime_id)] = {
                        'mean_return': 0.0,
                        'std_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'count': 0
                    }
                    continue

                mean_return = np.mean(regime_returns)
                std_return = np.std(regime_returns)
                sharpe_ratio = mean_return / (std_return + 1e-8)

                # Annualize Sharpe ratio (assuming 15m bars)
                # 15m = 4 bars/hour, 96 bars/day, ~35040 bars/year
                bars_per_year = 35040
                annualization_factor = np.sqrt(bars_per_year / horizon)
                sharpe_annualized = sharpe_ratio * annualization_factor

                regime_stats[int(regime_id)] = {
                    'mean_return': float(mean_return),
                    'std_return': float(std_return),
                    'sharpe_ratio': float(sharpe_ratio),
                    'sharpe_annualized': float(sharpe_annualized),
                    'count': int(len(regime_returns))
                }

            results[horizon_label] = regime_stats

        return results

    def _analyze_gmm_feature_impact(
        self,
        risk_df: pd.DataFrame,
        regime_labels: np.ndarray,
        label_metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze how each feature impacts GMM regime separation.

        This method computes:
        1. Between-regime variance (how much each feature varies across regimes)
        2. Within-regime variance (how stable each feature is within regimes)
        3. Separation ratio (between/within)
        4. Feature contribution to regime definition

        Args:
            risk_df: DataFrame with features
            regime_labels: GMM-assigned regime labels
            label_metrics: Metrics from GMM label creation
            config: Configuration dict

        Returns:
            Dict with feature impact metrics
        """
        # Get the features used in GMM
        primary_features = [
            "hurst_exponent_path",
            "path_trend_r2",
            "path_efficiency_return_3h",
            "quadratic_fit_curvature",
            "linear_reg_slope",
            "path_center_of_gravity",
            "body_range_ratio",
        ]

        available_features = [f for f in primary_features if f in risk_df.columns]

        if not available_features:
            tprint_warning("No primary features found for GMM impact analysis")
            return {}

        feature_impact_list = []

        # Filter to valid regimes
        valid_mask = regime_labels >= 0
        labels_valid = regime_labels[valid_mask]

        for feature in available_features:
            try:
                feature_data = risk_df[feature].iloc[valid_mask].values

                # Skip if all NaN
                if np.all(np.isnan(feature_data)):
                    continue

                # Remove NaN values
                feat_mask = ~np.isnan(feature_data)
                feat_clean = feature_data[feat_mask]
                labels_clean = labels_valid[feat_mask]

                if len(feat_clean) == 0:
                    continue

                # Compute between-regime variance
                regime_means = []
                regime_counts = []
                for regime_id in np.unique(labels_clean):
                    regime_data = feat_clean[labels_clean == regime_id]
                    if len(regime_data) > 0:
                        regime_means.append(np.mean(regime_data))
                        regime_counts.append(len(regime_data))

                if len(regime_means) < 2:
                    continue

                # Weighted mean of means
                global_mean = np.sum(np.array(regime_means) * np.array(regime_counts)) / np.sum(regime_counts)

                # Between-regime variance (variance of regime means)
                between_var = np.sum(
                    np.array(regime_counts) * (np.array(regime_means) - global_mean) ** 2
                ) / np.sum(regime_counts)

                # Within-regime variance (average variance within each regime)
                within_vars = []
                for regime_id in np.unique(labels_clean):
                    regime_data = feat_clean[labels_clean == regime_id]
                    if len(regime_data) > 1:
                        within_vars.append(np.var(regime_data, ddof=1))

                within_var = np.mean(within_vars) if within_vars else 0.0

                # Separation ratio (higher is better)
                if within_var > 0:
                    separation_ratio = between_var / within_var
                else:
                    separation_ratio = np.inf if between_var > 0 else 0.0

                # Coefficient of variation for between-regime means
                cv_between = np.std(regime_means) / np.abs(global_mean) if global_mean != 0 else 0.0

                # Store feature impact
                feature_impact_list.append({
                    "feature": feature,
                    "between_var": float(between_var),
                    "within_var": float(within_var),
                    "separation_ratio": float(separation_ratio),
                    "cv_between": float(cv_between),
                    "global_mean": float(global_mean),
                    "global_std": float(np.std(feat_clean)),
                    "regime_means": {int(i): float(m) for i, m in enumerate(regime_means)},
                    "n_samples": int(len(feat_clean)),
                })

            except Exception as feat_exc:
                tprint_warning(f"Failed to analyze feature {feature}: {feat_exc}")
                continue

        # Sort by separation ratio (descending)
        feature_impact_list = sorted(
            feature_impact_list,
            key=lambda x: x["separation_ratio"],
            reverse=True
        )

        return {
            "feature_impact": feature_impact_list,
            "n_features_analyzed": len(feature_impact_list),
            "label_metrics": label_metrics,
        }

    def _generate_gmm_feature_impact_report(
        self,
        gmm_feature_impact: Dict[str, Any],
        label_metrics: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> str:
        """
        Generate markdown report for GMM feature impact analysis.

        Args:
            gmm_feature_impact: Output from _analyze_gmm_feature_impact
            label_metrics: GMM label quality metrics
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Path to saved report file
        """
        import os
        from datetime import datetime

        # Ensure outcomes directory exists
        os.makedirs("outcomes", exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"outcomes/{symbol}_{exchange}_{timeframe}_GMM_Feature_Impact_{timestamp}.md"

        feature_impact_list = gmm_feature_impact.get("feature_impact", [])

        with open(report_path, 'w') as f:
            # Header
            f.write(f"# GMM-Direct Path Geometry Feature Impact Report\n\n")
            f.write(f"**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Overview
            f.write("## Overview\n\n")
            f.write("This report analyzes how each of the 7 core path geometry features impacts\n")
            f.write("GMM regime separation. Features are ranked by their **separation ratio**\n")
            f.write("(between-regime variance / within-regime variance).\n\n")

            # Label quality metrics
            f.write("## GMM Label Quality\n\n")
            f.write(f"- **Risk CV Ratio**: {label_metrics.get('risk_cv_ratio', 0):.4f}\n")
            f.write(f"- **Between-Regime CV**: {label_metrics.get('risk_cv_between', 0):.4f}\n")
            f.write(f"- **Within-Regime CV**: {label_metrics.get('risk_cv_within', 0):.4f}\n")
            f.write(f"- **Wasserstein Distance**: {label_metrics.get('wasserstein_distance', 0):.4f}\n")
            f.write(f"- **KL Divergence**: {label_metrics.get('kl_divergence', 0):.4f}\n")
            f.write(f"- **Quality Score**: {label_metrics.get('quality_score', 0):.4f}\n\n")

            regime_dist = label_metrics.get('regime_distribution', {})
            if regime_dist:
                f.write("**Regime Distribution:**\n")
                for regime_id, count in sorted(regime_dist.items()):
                    f.write(f"- Regime {regime_id}: {count} samples\n")
                f.write("\n")

            # Feature impact table
            f.write("## Feature Impact on Regime Separation\n\n")
            f.write("Features ranked by separation ratio (higher = better regime discrimination):\n\n")

            f.write("| Rank | Feature | Separation Ratio | Between Var | Within Var | CV Between |\n")
            f.write("|------|---------|------------------|-------------|------------|------------|\n")

            for rank, feat_info in enumerate(feature_impact_list, 1):
                feature = feat_info["feature"]
                sep_ratio = feat_info["separation_ratio"]
                between_var = feat_info["between_var"]
                within_var = feat_info["within_var"]
                cv_between = feat_info["cv_between"]

                # Format separation ratio
                if sep_ratio == np.inf:
                    sep_str = "∞"
                elif sep_ratio > 1000:
                    sep_str = f"{sep_ratio:.2e}"
                else:
                    sep_str = f"{sep_ratio:.2f}"

                f.write(
                    f"| {rank} | `{feature}` | {sep_str} | "
                    f"{between_var:.4f} | {within_var:.4f} | {cv_between:.4f} |\n"
                )

            f.write("\n")

            # Detailed per-feature analysis
            f.write("## Detailed Feature Analysis\n\n")

            for feat_info in feature_impact_list:
                feature = feat_info["feature"]
                f.write(f"### {feature}\n\n")

                f.write(f"- **Global Mean**: {feat_info['global_mean']:.4f}\n")
                f.write(f"- **Global Std**: {feat_info['global_std']:.4f}\n")
                f.write(f"- **Separation Ratio**: {feat_info['separation_ratio']:.4f}\n")
                f.write(f"- **Between-Regime Variance**: {feat_info['between_var']:.4f}\n")
                f.write(f"- **Within-Regime Variance**: {feat_info['within_var']:.4f}\n")
                f.write(f"- **CV Between Regimes**: {feat_info['cv_between']:.4f}\n\n")

                # Regime means
                regime_means = feat_info.get("regime_means", {})
                if regime_means:
                    f.write("**Mean Value by Regime:**\n")
                    for regime_id, mean_val in sorted(regime_means.items()):
                        f.write(f"- Regime {regime_id}: {mean_val:.4f}\n")
                    f.write("\n")

            # Interpretation
            f.write("## Interpretation Guide\n\n")
            f.write("**Separation Ratio**: Higher values indicate the feature strongly separates regimes.\n")
            f.write("- **> 1.0**: Feature contributes to regime separation\n")
            f.write("- **> 5.0**: Strong regime discriminator\n")
            f.write("- **> 10.0**: Dominant regime driver\n\n")

            f.write("**Between-Regime Variance**: Variance of regime means (higher = more separation).\n\n")
            f.write("**Within-Regime Variance**: Average variance within each regime (lower = more stability).\n\n")
            f.write("**CV Between**: Coefficient of variation of regime means (higher = relative separation).\n\n")

            # Footer
            f.write("---\n\n")
            f.write(f"*Report generated by ml_path_regime_step with GMM-Direct architecture*\n")

        return report_path

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
        f.write(f"*Report generated by ml_path_regime_step with XGBoost multi-class classifier*\n")

        return report_path

    def _generate_regime_wcov_report(
        self,
        df: pd.DataFrame,
        regime_labels: np.ndarray,
        config: Dict[str, Any],
        classifier_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        from datetime import datetime
        import os

        os.makedirs("outcomes", exist_ok=True)

        labels_array = np.asarray(regime_labels)
        if len(labels_array) != len(df):
            min_len = min(len(labels_array), len(df))
            labels_local = labels_array[:min_len]
            df_local = df.iloc[:min_len].copy()
        else:
            labels_local = labels_array
            df_local = df.copy()

        label_series = pd.Series(labels_local)
        valid_mask = label_series >= 0
        if valid_mask.sum() == 0:
            raise ValueError("No valid regime labels available for WCoV report")

        labels_local = labels_local[valid_mask.values]
        df_local = df_local[valid_mask.values]

        numeric_df = df_local.select_dtypes(include=[np.number]).copy()

        try:
            if "close" in df_local.columns:
                returns = np.log(df_local["close"] / df_local["close"].shift(1))
                if "returns_1h" not in numeric_df.columns:
                    numeric_df["returns_1h"] = returns
                return_3h = returns.rolling(window=12).sum()
                numeric_df["return_3h"] = return_3h
                downside_returns = returns.copy()
                downside_returns[downside_returns > 0] = 0
                downside_dev = downside_returns.rolling(window=80).std()
                sharpe_like_3h = return_3h / (downside_dev + 1e-9)
                numeric_df["sharpe_like_3h"] = sharpe_like_3h
        except Exception:
            pass

        base_metrics = [
            "returns_1h",
            "return_3h",
            "sharpe_like_3h",
            "path_ker_3h",
            "path_ker_6h",
            "body_range_ratio",
            "traffic_overlap_3h",
            "path_fractal_dimension",
            "hurst_exponent_path",
            "path_efficiency_return_3h",
            "path_directional_eff_3h",
            "path_trend_r2",
            "path_trend_up",
            "path_efficiency_dropping",
            "path_alpha_state",
        ]
        base_metrics_available = [
            c for c in base_metrics if c in numeric_df.columns
        ]

        exclude_metrics = {"path_permutation_entropy", "path_efficiency_high"}

        importance_features: List[str] = []
        quadrant_features: List[str] = []

        if classifier_metrics is not None:
            fi = classifier_metrics.get("feature_importance_detailed")
            if fi is not None:
                try:
                    global_imp = pd.DataFrame(fi.get("global", []))
                    if not global_imp.empty and "feature" in global_imp.columns:
                        global_imp = global_imp.sort_values(
                            "combined_score",
                            ascending=False,
                        )
                        importance_features = [
                            str(f) for f in global_imp["feature"].head(20).tolist()
                        ]
                except Exception:
                    importance_features = []

            quadrant_quality = classifier_metrics.get("quadrant_quality")
            if isinstance(quadrant_quality, dict):
                qf = quadrant_quality.get("quadrant_features")
                if isinstance(qf, (list, tuple)):
                    quadrant_features = [str(c) for c in qf]

        candidate_cols: List[str] = []
        for col in base_metrics_available + importance_features + quadrant_features:
            if (
                col in numeric_df.columns
                and col not in candidate_cols
                and col not in exclude_metrics
            ):
                candidate_cols.append(col)

        if not candidate_cols:
            candidate_cols = list(numeric_df.columns)

        rows: List[Dict[str, Any]] = []
        lower_pct = 0.05
        upper_pct = 0.95
        unique_regimes = [r for r in np.unique(labels_local) if r >= 0]

        for col in candidate_cols:
            series = numeric_df[col]
            series_nonnull = series.dropna()
            if series_nonnull.empty:
                continue

            try:
                between = self._calculate_winsorized_cv_between(labels_local, series)
                within = self._calculate_winsorized_cv_within(labels_local, series)
                ratio = between / (within + 1e-8)
                if np.isfinite(ratio) and ratio > 0.0:
                    ratio = np.log1p(ratio)
            except Exception:
                between = np.nan
                within = np.nan
                ratio = np.nan

            rows.append(
                {
                    "metric": col,
                    "scope": "GLOBAL",
                    "regime": "GLOBAL",
                    "n": int(series_nonnull.shape[0]),
                    "winsorized_mean": np.nan,
                    "winsorized_std": np.nan,
                    "wcov_regime": np.nan,
                    "wcov_between": float(between) if np.isfinite(between) else np.nan,
                    "wcov_within": float(within) if np.isfinite(within) else np.nan,
                    "wcov_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
                }
            )

            for regime_id in unique_regimes:
                mask = labels_local == regime_id
                data = series[mask].dropna()
                if len(data) < 2:
                    continue
                try:
                    lower = data.quantile(lower_pct)
                    upper = data.quantile(upper_pct)
                    winsorized = data.clip(lower=lower, upper=upper)
                    mean_val = float(winsorized.mean())
                    std_val = float(winsorized.std())
                    wcov_regime = std_val / (abs(mean_val) + 1e-8)
                except Exception:
                    mean_val = np.nan
                    std_val = np.nan
                    wcov_regime = np.nan

                rows.append(
                    {
                        "metric": col,
                        "scope": "REGIME",
                        "regime": int(regime_id),
                        "n": int(len(data)),
                        "winsorized_mean": mean_val,
                        "winsorized_std": std_val,
                        "wcov_regime": wcov_regime,
                        "wcov_between": np.nan,
                        "wcov_within": np.nan,
                        "wcov_ratio": np.nan,
                    }
                )

        if not rows:
            raise ValueError("No numeric metrics available for WCoV report")

        report_df = pd.DataFrame(rows)

        symbol_q = str(config.get("symbol", ""))
        exchange_q = str(config.get("exchange", ""))
        regime_tf_q = str(
            config.get("regime_timeframe", config.get("timeframe", "15m"))
        )

        ts_q = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = (
            f"outcomes/ml_path_regime_wcov_"
            f"{symbol_q or 'UNKNOWN'}_{regime_tf_q}_{ts_q}.csv"
        )
        md_path = (
            f"outcomes/ml_path_regime_wcov_"
            f"{symbol_q or 'UNKNOWN'}_{regime_tf_q}_{ts_q}.md"
        )

        report_df.to_csv(csv_path, index=False)

        try:
            global_df = report_df[report_df["scope"] == "GLOBAL"].copy()
            global_df = global_df.sort_values("wcov_ratio", ascending=False)

            with open(md_path, "w") as f:
                f.write("# Path Regime WCoV Report\n\n")
                f.write(
                    f"**Symbol**: {symbol_q} | **Exchange**: {exchange_q} | "
                    f"**Timeframe**: {regime_tf_q}\n\n"
                )
                f.write(
                    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                f.write("## Global WCoV Metrics (per metric)\n\n")
                f.write(
                    "| Metric | WCoV Between | WCoV Within | WCoV Ratio | Samples |\n"
                )
                f.write("|--------|--------------|-------------|------------|---------|\n")
                for _, row in global_df.iterrows():
                    f.write(
                        f"| {str(row['metric'])} | "
                        f"{float(row.get('wcov_between', np.nan)):.4f} | "
                        f"{float(row.get('wcov_within', np.nan)):.4f} | "
                        f"{float(row.get('wcov_ratio', np.nan)):.4f} | "
                        f"{int(row.get('n', 0))} |\n"
                    )

                # Regime-level summary focusing on structural Path metrics
                per_regime_df = report_df[report_df["scope"] == "REGIME"].copy()
                core_metric_candidates = [
                    "path_directional_eff_3h",
                    "path_efficiency_return_3h",
                    "path_trend_up",
                    "path_ker_3h",
                    "path_ker_6h",
                    "body_range_ratio",
                    "traffic_overlap_3h",
                    "hurst_exponent_path",
                    "path_fractal_dimension",
                    "path_efficiency_dropping",
                    "path_alpha_state",
                    "path_trend_r2",
                ]
                available_core = [
                    m for m in core_metric_candidates
                    if m in per_regime_df["metric"].unique()
                ]
                if not available_core:
                    available_core = sorted(
                        {
                            m
                            for m in per_regime_df["metric"].unique()
                            if str(m).startswith("path_")
                            or str(m).startswith("traffic_")
                            or str(m).startswith("body_")
                        }
                    )

                if available_core:
                    f.write("\n## Regime Summary (Core Path Metrics)\n\n")
                    f.write(
                        "| Regime | Metric | Winsorized Mean | WCoV (Regime) | Sign | Magnitude Rank (|mean| across regimes) |\n"
                    )
                    f.write(
                        "|--------|--------|-----------------|--------------|------|----------------------------------------|\n"
                    )

                    rank_lookup = {}
                    unique_regimes_summary = sorted(per_regime_df["regime"].unique())
                    for metric_name in available_core:
                        m_df = per_regime_df[
                            per_regime_df["metric"] == metric_name
                        ].copy()
                        if m_df.empty:
                            continue
                        m_df["abs_mean"] = m_df["winsorized_mean"].abs()
                        m_df = m_df.sort_values("abs_mean", ascending=False)
                        rank_counter = 1
                        for _, row_m in m_df.iterrows():
                            try:
                                regime_id = int(row_m["regime"])
                            except Exception:
                                continue
                            key = (regime_id, metric_name)
                            if key not in rank_lookup:
                                rank_lookup[key] = rank_counter
                                rank_counter += 1

                    for regime_id in unique_regimes_summary:
                        for metric_name in available_core:
                            subset = per_regime_df[
                                (per_regime_df["regime"] == regime_id)
                                & (per_regime_df["metric"] == metric_name)
                            ]
                            if subset.empty:
                                continue
                            row_m = subset.iloc[0]
                            mean_val = float(
                                row_m.get("winsorized_mean", np.nan)
                            )
                            wcov_val = float(
                                row_m.get("wcov_regime", np.nan)
                            )
                            if np.isfinite(mean_val):
                                if mean_val > 0:
                                    sign_symbol = "↑"
                                elif mean_val < 0:
                                    sign_symbol = "↓"
                                else:
                                    sign_symbol = "0"
                            else:
                                sign_symbol = "?"
                            rank_val = rank_lookup.get(
                                (int(regime_id), metric_name), np.nan
                            )
                            f.write(
                                f"| {int(regime_id)} | {metric_name} | "
                                f"{mean_val:.6f} | "
                                f"{wcov_val:.4f} | "
                                f"{sign_symbol} | "
                                f"{'' if not np.isfinite(rank_val) else int(rank_val)} |\n"
                            )

                    f.write("\n## Composite Trend Scores\n\n")
                    f.write("| Regime | Trend Up Score | Trend Down Score |\n")
                    f.write("|--------|----------------|------------------|\n")
                    up_metrics = [
                        "path_directional_eff_3h",
                        "path_efficiency_return_3h",
                        "path_trend_up",
                        "path_ker_3h",
                        "path_alpha_state",
                    ]
                    down_metrics = [
                        "path_directional_eff_3h",
                        "path_efficiency_return_3h",
                    ]
                    eps = 1e-8
                    for regime_id in unique_regimes_summary:
                        up_score = 0.0
                        down_score = 0.0
                        for metric_name in up_metrics:
                            if metric_name not in available_core:
                                continue
                            subset = per_regime_df[
                                (per_regime_df["regime"] == regime_id)
                                & (per_regime_df["metric"] == metric_name)
                            ]
                            if subset.empty:
                                continue
                            row_m = subset.iloc[0]
                            mean_val = float(
                                row_m.get("winsorized_mean", np.nan)
                            )
                            wcov_val = float(
                                row_m.get("wcov_regime", np.nan)
                            )
                            if (
                                not np.isfinite(mean_val)
                                or not np.isfinite(wcov_val)
                                or wcov_val <= 0.0
                            ):
                                continue
                            up_score += max(mean_val, 0.0) / (wcov_val + eps)

                        for metric_name in down_metrics:
                            if metric_name not in available_core:
                                continue
                            subset = per_regime_df[
                                (per_regime_df["regime"] == regime_id)
                                & (per_regime_df["metric"] == metric_name)
                            ]
                            if subset.empty:
                                continue
                            row_m = subset.iloc[0]
                            mean_val = float(
                                row_m.get("winsorized_mean", np.nan)
                            )
                            wcov_val = float(
                                row_m.get("wcov_regime", np.nan)
                            )
                            if (
                                not np.isfinite(mean_val)
                                or not np.isfinite(wcov_val)
                                or wcov_val <= 0.0
                            ):
                                continue
                            down_score += max(-mean_val, 0.0) / (wcov_val + eps)

                        f.write(
                            f"| {int(regime_id)} | {up_score:.4f} | {down_score:.4f} |\n"
                        )

                f.write("\n## Per-Regime Winsorized CV (per metric)\n\n")
                for metric in candidate_cols:
                    subset = report_df[
                        (report_df["scope"] == "REGIME")
                        & (report_df["metric"] == metric)
                    ].copy()
                    if subset.empty:
                        continue
                    subset = subset.sort_values("regime")

                    f.write(f"### Metric: {metric}\n\n")
                    f.write(
                        "| Regime | Samples | Winsorized Mean | Winsorized Std | WCoV (Regime) |\n"
                    )
                    f.write(
                        "|--------|---------|-----------------|----------------|---------------|\n"
                    )
                    for _, row in subset.iterrows():
                        f.write(
                            f"| {int(row['regime'])} | "
                            f"{int(row.get('n', 0))} | "
                            f"{float(row.get('winsorized_mean', np.nan)):.6f} | "
                            f"{float(row.get('winsorized_std', np.nan)):.6f} | "
                            f"{float(row.get('wcov_regime', np.nan)):.4f} |\n"
                        )

                    f.write("\n")

                # Add Temporal Metrics section
                temporal_metrics = classifier_metrics.get('temporal_metrics') if classifier_metrics else None
                if temporal_metrics:
                    f.write("---\n\n")
                    f.write("## Temporal Metrics (HMM Transition Dynamics)\n\n")
                    f.write(f"**Average Duration**: {temporal_metrics.get('avg_duration_bars', 0):.2f} bars\n\n")
                    f.write(f"**Flip-Flop Rate**: {temporal_metrics.get('flip_flop_rate', 0):.4f}\n\n")
                    f.write(f"**Stability Score**: {temporal_metrics.get('stability_score', 0):.4f}\n\n")
                    f.write(f"**Regime Changes**: {temporal_metrics.get('regime_changes', 0)}\n\n")
                    f.write(f"**Total Samples**: {temporal_metrics.get('total_samples', 0)}\n\n")

                    # Transition Matrix
                    trans_matrix = temporal_metrics.get('transition_matrix', [])
                    if trans_matrix:
                        f.write("### Transition Matrix\n\n")
                        f.write("Probability of transitioning from regime i (row) to regime j (column):\n\n")
                        f.write("| From → To | " + " | ".join([f"Regime {i}" for i in range(len(trans_matrix))]) + " |\n")
                        f.write("|" + "|".join(["-----------"] * (len(trans_matrix) + 1)) + "|\n")
                        for i, row in enumerate(trans_matrix):
                            f.write(f"| **Regime {i}** | " + " | ".join([f"{p:.4f}" for p in row]) + " |\n")
                        f.write("\n")

                    # Per-regime duration stats
                    duration_stats = temporal_metrics.get('regime_duration_stats', {})
                    if duration_stats:
                        f.write("### Regime Duration Statistics\n\n")
                        f.write("| Regime | Mean Duration (bars) | Median | Min | Max | Occurrences |\n")
                        f.write("|--------|---------------------|--------|-----|-----|-------------|\n")
                        for regime_id in sorted(duration_stats.keys()):
                            stats = duration_stats[regime_id]
                            f.write(
                                f"| {regime_id} | {stats['mean_duration']:.2f} | "
                                f"{stats['median_duration']:.1f} | {stats['min_duration']} | "
                                f"{stats['max_duration']} | {stats['count']} |\n"
                            )
                        f.write("\n")

                # Add Forward Returns and Sharpe Ratios section
                forward_returns_sharpe = classifier_metrics.get('forward_returns_sharpe') if classifier_metrics else None
                if forward_returns_sharpe:
                    f.write("---\n\n")
                    f.write("## Forward Returns & Sharpe Ratios Per Regime\n\n")

                    for horizon_label in ['15m', '1h', '3h']:
                        if horizon_label not in forward_returns_sharpe:
                            continue

                        regime_stats = forward_returns_sharpe[horizon_label]
                        f.write(f"### {horizon_label} Horizon\n\n")
                        f.write("| Regime | Mean Return | Std Return | Sharpe | Sharpe (Ann.) | Samples |\n")
                        f.write("|--------|-------------|------------|--------|---------------|-------|\n")

                        for regime_id in sorted(regime_stats.keys()):
                            stats = regime_stats[regime_id]
                            f.write(
                                f"| {regime_id} | {stats['mean_return']:.6f} | "
                                f"{stats['std_return']:.6f} | {stats['sharpe_ratio']:.4f} | "
                                f"{stats['sharpe_annualized']:.4f} | {stats['count']} |\n"
                            )
                        f.write("\n")

                f.write("---\n\n")
                f.write("*Report generated by ml_path_regime_step with HMM architecture*\n")

        except Exception:
            md_path = ""

        return csv_path, md_path

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
        DEPRECATED: This method is no longer used in GMM-Direct architecture.

        Previously trained XGBoost multi-class classifier to predict regimes.
        Now replaced with direct GMM inference (no student classifier).

        Kept for reference only. Remove in future cleanup.

        Args:
            risk_df: Feature dataframe
            regime_labels: Target regime labels (0-3, -1 for invalid)
            config: Configuration dict

        Returns:
            model: Trained XGBoost classifier
            regime_probs: Predicted probabilities (n_samples x 4)
            training_metrics: Performance metrics
        """
        raise NotImplementedError(
            "XGBoost student classifier has been removed. "
            "Use GMM-Direct architecture instead (_create_optimal_regime_labels)."
        )
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

        # Select features (Path-only allowlist, with basic returns/helpers; fallback to generic numeric)
        numeric_df = df_clean.select_dtypes(include=[np.number])

        path_feature_allowlist: List[str] = [
            "path_ker_3h",
            "path_ker_6h",
            "body_range_ratio",
            "traffic_overlap_3h",
            "path_fractal_dimension",
            "hurst_exponent_path",
            "path_trend_r2",
            "returns_1h",
            "path_efficiency_return_3h",
            "path_directional_eff_3h",
            # Legacy return-based helpers kept for backward compatibility if present
            "return_3h",
            "sharpe_like_3h",
            "path_trend_up",
            "path_efficiency_dropping",
            "path_alpha_state",
        ]

        feature_cols = [
            col for col in path_feature_allowlist
            if col in numeric_df.columns
        ]

        if not feature_cols:
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

        # Use float32 for all classifier inputs to reduce memory and improve throughput
        if hasattr(X_train, "astype"):
            X_train = X_train.astype("float32")
        if hasattr(X_val, "astype"):
            X_val = X_val.astype("float32")
        if hasattr(X_full, "astype"):
            X_full = X_full.astype("float32")

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
        es_rounds = int(config.get("risk_classifier_early_stopping_rounds", 50))

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

            # Core 7 Path Geometry Features (user-specified)
            # These are the ONLY features used for GMM regime detection
            path_priority = [
                "hurst_exponent_path",         # Roughness
                "path_trend_r2",               # Linearity
                "path_efficiency_return_3h",   # Directness
                "quadratic_fit_curvature",     # Shape/Bend
                "linear_reg_slope",            # Steepness
                "path_center_of_gravity",      # Timing
                "body_range_ratio",            # Morphology
            ]

            path_candidates: List[str] = [
                col for col in path_priority
                if col in all_cols
            ]

            if path_candidates:
                return list(dict.fromkeys(path_candidates))
            return []

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
                            # Allow slightly deeper trees and smaller child
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
                            "subsample": {"type": "float", "low": 0.7, "high": 0.95},
                            "colsample_bytree": {"type": "float", "low": 0.7, "high": 0.95},
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

                quadrant_cols = _select_quadrant_feature_cols(feature_cols)

                # Log which quadrant features will actually drive WCoV HPO and
                # pruning, and warn if no suitable path features are present in
                # the classifier feature set.
                if quadrant_cols:
                    tprint_info(
                        f"🧭 Quadrant features selected for WCoV HPO/pruning: {quadrant_cols}"
                    )
                else:
                    tprint_warning(
                        "No quadrant features found in classifier features; WCoV "
                        "HPO/pruning will have no effect. Ensure the core Path "
                        "features (KER, body/range, traffic, entropy, fractal, "
                        "path Hurst, 3h returns) are present in the feature set."
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

                        wcov_bonus = 0.0
                        if quadrant_cols:
                            try:
                                val_df = pd.DataFrame(X_val, columns=feature_cols)
                                quad_df = val_df[quadrant_cols]

                                labels_arr = np.asarray(val_pred, dtype=int)
                                total_n = float(len(labels_arr))
                                if total_n <= 0.0:
                                    raise ValueError("Empty validation set for quadrant objective")

                                # Down-weight direct PnL-like features in the
                                # WCoV computation by scaling their columns,
                                # then compute global between/within WCoV.
                                pnl_weight = float(
                                    config.get("path_hpo_pnl_feature_weight", 0.5)
                                )
                                pnl_keywords = (
                                    "return_3h",
                                    "returns_1h",
                                    "sharpe_like_3h",
                                    "path_efficiency_return_3h",
                                    "path_directional_eff_3h",
                                )

                                quad_df_weighted = quad_df.copy()
                                if pnl_weight < 1.0:
                                    for col in quad_df_weighted.columns:
                                        col_lower = str(col).lower()
                                        if any(kw in col_lower for kw in pnl_keywords):
                                            try:
                                                quad_df_weighted[col] = (
                                                    quad_df_weighted[col] * pnl_weight
                                                )
                                            except Exception:
                                                # Defensive: skip columns that cannot be scaled
                                                pass

                                # Global between- and within-regime WCoV on the
                                # weighted Path quadrant features.
                                wcov_between = self._calculate_winsorized_cv_between(
                                    labels_arr,
                                    quad_df_weighted,
                                )
                                wcov_within = self._calculate_winsorized_cv_within(
                                    labels_arr,
                                    quad_df_weighted,
                                )

                                within_weighted = float(wcov_within)
                                within_spread = 0.0

                                # Log-transform between-regime WCoV
                                between_clamped = max(float(wcov_between), 0.0)
                                between_log = float(np.log1p(between_clamped))

                                # Reverse-log transform for regime-weighted within WCoV
                                within_clamped = max(float(within_weighted), 1e-6)
                                reverse_log_within = float(np.log1p(1.0 / within_clamped))

                                base_score = between_log * reverse_log_within

                                # Penalty for large spread in within-regime WCoV (max - min)
                                spread_penalty = max(0.0, within_spread)
                                spread_factor = float(1.0 / (1.0 + spread_penalty))

                                wcov_bonus = max(0.0, base_score * spread_factor)
                            except Exception:
                                wcov_bonus = 0.0

                        # Predicted-regime balance term: discourage models that
                        # collapse almost all samples into a single regime while
                        # still primarily optimizing for WCoV separation.
                        balance_bonus = 1.0
                        entropy_bonus = 1.0
                        try:
                            if val_pred is not None:
                                counts = np.bincount(val_pred, minlength=n_regimes)
                                total = float(counts.sum()) if counts.sum() > 0 else 0.0
                                if total > 0.0:
                                    p = counts.astype(float) / total
                                    min_pct = float(p.min())
                                    max_pct = float(p.max())

                                    # Default band for HPO: aim for each
                                    # regime to have at least ~8% and at most
                                    # ~60% of samples, unless overridden via
                                    # config.

                                    violation_low = max(0.0, min_target - min_pct)
                                    violation_high = max(0.0, max_pct - max_target)
                                    total_violation = violation_low + violation_high

                                    # Stronger default penalty on imbalance so
                                    # HPO meaningfully prefers more balanced
                                    # predicted regime distributions. Default
                                    # increased from 8.0 to 25.0 so that HPO
                                    # more aggressively avoids extreme
                                    # collapse/vanishing regimes unless
                                    # explicitly overridden via config.
                                    balance_strength = float(
                                        config.get("risk_hpo_balance_strength", 25.0)
                                    )
                                    # Exponential decay from 1.0 toward 0.0 as
                                    # imbalance grows; no violation -> 1.0.
                                    balance_bonus = float(
                                        np.exp(-balance_strength * total_violation)
                                    )
                                    # Normalized label entropy bonus: encourage
                                    # higher-entropy (more balanced) predicted
                                    # regime distributions.
                                    valid_mask = p > 0.0
                                    p_valid = p[valid_mask]
                                    entropy_norm = 0.0
                                    if p_valid.size > 0:
                                        entropy_val = float(-np.sum(p_valid * np.log(p_valid)))
                                        max_entropy = float(np.log(max(float(n_regimes), 1.0)))
                                        if max_entropy > 0.0 and np.isfinite(entropy_val):
                                            entropy_norm = float(
                                                np.clip(entropy_val / max_entropy, 0.0, 1.0)
                                            )

                                    lambda_entropy = float(
                                        config.get("risk_hpo_entropy_weight", 2.0)
                                    )
                                    # Centered at 0.5: low-entropy distributions
                                    # get a strong penalty, while high-entropy
                                    # ones can receive a moderate boost.
                                    entropy_bonus = float(
                                        max(0.1, 1.0 + lambda_entropy * (entropy_norm - 0.5))
                                    )
                        except Exception as balance_exc:  # pragma: no cover - defensive
                            tprint_warning(
                                f"Quadrant HPO balance term failed (non-fatal): {balance_exc}"
                            )
                            balance_bonus = 1.0
                            entropy_bonus = 1.0

                        lambda_wcov = float(config.get("risk_hpo_wcov_weight", 1.0))
                        return float(lambda_wcov * wcov_bonus * balance_bonus * entropy_bonus)
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

                X_train_np = X_train.values if hasattr(X_train, "values") else X_train
                y_train_np = y_train
                X_val_np = X_val.values if hasattr(X_val, "values") else X_val
                y_val_np = y_val

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

        if quadrant_quality:
            training_metrics['quadrant_quality'] = quadrant_quality

            # Persist a dedicated quadrant WCoV quality summary reflecting HPO goals.
            try:
                symbol_q = str(config.get("symbol", ""))
                exchange_q = str(config.get("exchange", ""))
                regime_tf_q = str(
                    config.get("regime_timeframe", config.get("timeframe", "15m"))
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
                    f"outcomes/ml_path_quadrant_quality_"
                    f"{symbol_q or 'UNKNOWN'}_{regime_tf_q}_{ts_q}.csv"
                )
                pd.DataFrame([quality_row]).to_csv(quadrant_quality_path, index=False)
                tprint_info(
                    f"💾 Saved Path quadrant WCoV quality summary: {quadrant_quality_path}"
                )
            except Exception as quad_csv_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Failed to persist quadrant WCoV quality summary (non-fatal): {quad_csv_exc}"
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

        try:
            global_df = importance_data['global'].copy()
            eps = 1e-8

            distinctiveness_records = []
            regime_ids = sorted(np.unique(y))

            for feat in feature_cols:
                if feat not in numeric_df.columns:
                    continue
                series = numeric_df[feat]
                if series.isna().all():
                    continue

                regime_means = []
                within_covs = []

                for rid in regime_ids:
                    rid_mask = y == rid
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

            high_mask = conf >= high_conf_thr
            low_mask = conf <= low_conf_thr

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
            base_gain_min = float(config.get("risk_noise_gain_min", 0.005))
            base_wcov_max = float(config.get("risk_noise_wcov_max", 1.3))
            base_var_ratio_min = float(config.get("risk_noise_var_ratio_min", 1.5))
            base_eff_size_min = float(config.get("risk_noise_effect_size_min", 0.4))

            noise_hpo_enable = bool(config.get("risk_noise_hpo_enable", False))
            noise_hpo_max_trials = int(config.get("risk_noise_hpo_max_trials", 0))
            if noise_hpo_enable and noise_hpo_max_trials < 2:
                noise_hpo_max_trials = 2

            def _compute_flags(gain_min: float, wcov_max: float, var_ratio_min: float, eff_size_min: float):
                flags_importance = noise_df['gain_norm'] >= gain_min
                flags_wcov = noise_df['wcov_distinctiveness'] <= wcov_max
                flags_conf = noise_df['var_ratio_confidence'] >= var_ratio_min
                flags_misclass = noise_df['misclass_effect_size'] >= eff_size_min
                is_noisy = (
                    flags_importance
                    & (
                        (flags_wcov & flags_conf)
                        | (flags_wcov & flags_misclass)
                    )
                )
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
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "15m")))

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

        # ------------------------------------------------------------------
        # Temperature scaling probability calibration (multi-class)
        # ------------------------------------------------------------------
        calibration_enabled_cfg = bool(config.get("path_enable_prob_calibration", True))
        skip_calib_hpo = bool(config.get("path_skip_calibration_for_hpo", False))
        calibration_enabled = calibration_enabled_cfg and not skip_calib_hpo and len(X_val) > 0

        final_training_metrics["probability_calibration_enabled"] = calibration_enabled

        if calibration_enabled:
            try:
                from scipy.optimize import minimize_scalar
                try:
                    from sklearn.metrics import roc_auc_score
                except Exception:  # pragma: no cover - defensive
                    roc_auc_score = None  # type: ignore[assignment]

                # Uncalibrated validation probabilities from the final model
                val_probs_uncal = final_model.predict_proba(X_val)

                # Optional macro AUC on uncalibrated probabilities (one-vs-rest)
                val_auc_uncal: Optional[float] = None
                if "val_auc_uncalibrated" in final_training_metrics:
                    try:
                        val_auc_uncal = float(final_training_metrics["val_auc_uncalibrated"])
                    except Exception:
                        val_auc_uncal = None
                elif roc_auc_score is not None:
                    try:
                        val_auc_uncal = float(
                            roc_auc_score(y_val, val_probs_uncal, multi_class="ovr")
                        )
                        final_training_metrics["val_auc_uncalibrated"] = val_auc_uncal
                    except Exception:
                        val_auc_uncal = None

                # Optimize a single temperature parameter on NLL of the true
                # class, using logits derived from probabilities.
                eps = 1e-12
                probs_val_clipped = np.clip(val_probs_uncal, eps, 1.0 - eps)
                logits_val = np.log(probs_val_clipped)

                def _nll_for_temperature(T: float) -> float:
                    if T <= 0.0:
                        return float("inf")
                    scaled = logits_val / T
                    scaled = scaled - scaled.max(axis=1, keepdims=True)
                    exp_s = np.exp(scaled)
                    probs_s = exp_s / exp_s.sum(axis=1, keepdims=True)
                    idx = np.arange(len(y_val))
                    p_true = probs_s[idx, y_val]
                    p_true = np.clip(p_true, eps, 1.0 - eps)
                    return float(-np.mean(np.log(p_true)))

                opt_result = minimize_scalar(
                    _nll_for_temperature,
                    bounds=(0.05, 5.0),
                    method="bounded",
                )

                T_opt = float(opt_result.x if opt_result.success else 1.0)

                def _apply_temperature(probs: np.ndarray) -> np.ndarray:
                    probs_local = np.clip(probs, eps, 1.0 - 1e-8)
                    logits_local = np.log(probs_local)
                    scaled_local = logits_local / T_opt
                    scaled_local = scaled_local - scaled_local.max(axis=1, keepdims=True)
                    exp_scaled = np.exp(scaled_local)
                    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

                # Calibrated probabilities on validation and full datasets
                val_probs_cal = _apply_temperature(val_probs_uncal)
                full_probs_cal = _apply_temperature(final_regime_probs)

                # Update final regime probabilities with calibrated values
                final_regime_probs = full_probs_cal

                # AUC after calibration (macro one-vs-rest when available)
                if roc_auc_score is not None:
                    try:
                        val_auc_cal = float(
                            roc_auc_score(y_val, val_probs_cal, multi_class="ovr")
                        )
                        final_training_metrics["val_auc_calibrated"] = val_auc_cal
                        # Also expose a generic val_auc for downstream reports
                        final_training_metrics["val_auc"] = val_auc_cal
                        if (
                            "val_auc_uncalibrated" not in final_training_metrics
                            and val_auc_uncal is not None
                        ):
                            final_training_metrics["val_auc_uncalibrated"] = val_auc_uncal
                    except Exception:
                        pass

                # Expected Calibration Error (ECE) before/after temperature scaling
                def _expected_calibration_error(
                    probs: np.ndarray,
                    y_true: np.ndarray,
                    n_bins: int = 15,
                ) -> float:
                    confidences = probs.max(axis=1)
                    predictions = probs.argmax(axis=1)
                    correct = (predictions == y_true).astype(float)
                    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
                    ece_val = 0.0
                    n = len(y_true)
                    for i in range(n_bins):
                        lo, hi = bin_edges[i], bin_edges[i + 1]
                        if i < n_bins - 1:
                            mask = (confidences > lo) & (confidences <= hi)
                        else:
                            mask = (confidences > lo) & (confidences <= hi)
                        if not np.any(mask):
                            continue
                        bin_conf = float(confidences[mask].mean())
                        bin_acc = float(correct[mask].mean())
                        ece_val += abs(bin_acc - bin_conf) * (mask.sum() / float(n))
                    return float(ece_val)

                try:
                    ece_uncal = _expected_calibration_error(val_probs_uncal, y_val)
                    ece_cal = _expected_calibration_error(val_probs_cal, y_val)
                    final_training_metrics["ece_uncalibrated"] = ece_uncal
                    final_training_metrics["ece_calibrated"] = ece_cal
                    final_training_metrics["ece_improvement"] = ece_uncal - ece_cal
                except Exception:
                    pass

                final_training_metrics["calibration_method"] = "temperature_scaling"
            except Exception as calib_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Temperature scaling calibration failed (non-fatal); using uncalibrated probabilities: {calib_exc}"
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
                artifact_name="ml_risk_quality_metrics_15m",
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
                artifact_name="hmm_alpha_regime_quality_15m",
                artifact_type="data",
                metadata={
                    "min_regime_size": min_regime_size,
                },
            )
        except Exception as save_exc:
            tprint_warning(f"Failed to save alpha regime quality artifact: {save_exc}")
            quality_path = None

        return metrics, quality_path
