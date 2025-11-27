"""
ML Risk Regime Step (HMM Architecture with Mahalanobis Distance Scoring)

This step constructs Risk-based regime labels using Hidden Markov Models (HMM)
with Mahalanobis distance scoring from a "safe" state for geometric anomaly detection.

Primary Goal: Distinguish between turbulent, calm, crash-prone, and volatile markets
using HMM with learned regime transitions and risk feature analysis.

Key Features:
- Uses ONLY 5 core risk features (parkinson_volatility, hurst_exponent, rolling_kurtosis,
  rolling_skewness, volatility_of_volatility)
- All features scaled with winsorized_zscore_normalize
- HMM-Direct inference with full covariance (NO XGBoost classifier)
- Temporal structure learning via transition matrix
- Mahalanobis distance scoring from "safe" state
- Log-stabilized distance: ln(1 + raw)
- Strided training for computational efficiency (every 10th point)
- GMM warm-start initialization for faster convergence
- LiveHMM wrapper for O(1) production updates

Computational Optimizations:
1. Strided Training: Train on every 10th data point to reduce training time
2. GMM Injection: Pre-compute cluster centers with fast GMM, inject into HMM
3. Warm Start Retraining: Use GMM means to initialize HMM parameters
4. O(1) Live Updates: LiveHMM class for production with pre-extracted matrices

Risk Features (30-50 bar windows for 30m-3h trades on 1h data):
- parkinson_volatility (window: 48 bars = 48h)
- hurst_exponent (window: 48 bars = 48h)
- rolling_kurtosis (window: 36 bars = 36h)
- rolling_skewness (window: 36 bars = 36h)
- volatility_of_volatility (window: 30 bars = 30h)

Responsibilities:
- Load 1h OHLCV market data
- Generate 5 core risk features with appropriate windows
- Apply winsorized zscore normalization
- Create optimal regime labels using HMM with temporal structure
- Calculate Mahalanobis distance from "safe" state (log-stabilized)
- Calculate temporal metrics (transition matrix, flip-flop rate, duration)
- Calculate forward returns and Sharpe ratios at 1h horizon
- Persist HMM model and LiveHMM wrapper for live trading
- Generate comprehensive risk regime quality and temporal reports
- Save regime outputs with Mahalanobis scores to versioned_artifacts
"""

import logging
import time
import json
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from src.utils.ml_common.hmm_warm_start import HMMWarmStarter

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.features_common.transforms.scaling_normalization import (
    ScalingNormalizer,
    winsorized_zscore_normalize,
)
from src.feature_generation.categories.risk_regime_features import (
    generate_risk_regime_features,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)


logger = logging.getLogger(__name__)


class LiveHMM:
    """
    Production-grade O(1) HMM state update for live trading with Mahalanobis distance scoring.

    Extracts transition matrix, means, and covariances from a trained HMM model
    and provides efficient forward algorithm updates without retraining.

    Additionally computes log-stabilized Mahalanobis distance from the "safe" state.

    Usage:
        # After training your HMM
        live_hmm = LiveHMM(trained_hmm_model, safe_state_id=0)

        # In live trading loop
        new_observation = np.array([...])  # Single observation vector
        regime_probs = live_hmm.update(new_observation)
        current_regime = np.argmax(regime_probs)
        mahal_distance = live_hmm.get_mahalanobis_distance(new_observation)
    """

    def __init__(
        self,
        trained_model: GaussianHMM,
        safe_state_id: int = 0,
        mahal_scaler_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize LiveHMM from a trained hmmlearn GaussianHMM model.

        Args:
            trained_model: Fitted GaussianHMM instance
            safe_state_id: Index of the "safe" state for Mahalanobis distance (default: 0)
            mahal_scaler_params: Dict with 'min' and 'max' for [0, 1] normalization
        """
        # Extract parameters from trained model
        self.n_components = trained_model.n_components
        self.startprob = trained_model.startprob_
        self.transmat = trained_model.transmat_
        self.means = trained_model.means_
        self.covars = trained_model.covars_
        self.covariance_type = trained_model.covariance_type

        # Safe state configuration
        self.safe_state_id = safe_state_id
        self.safe_mean = self.means[safe_state_id]
        self.safe_cov = self._get_covariance(safe_state_id)
        self.safe_cov_inv = np.linalg.inv(self.safe_cov)

        # Scaler parameters for [0, 1] normalization
        self.mahal_scaler_params = mahal_scaler_params or {}

        # Initialize state probabilities
        self.state_probs = self.startprob.copy()

        # Pre-compute multivariate normal distributions for each state
        self._setup_distributions()

    def _get_covariance(self, state_id: int) -> np.ndarray:
        """Get covariance matrix for a specific state."""
        if self.covariance_type == 'full':
            return self.covars[state_id]
        elif self.covariance_type == 'diag':
            return np.diag(self.covars[state_id])
        elif self.covariance_type == 'tied':
            return self.covars
        elif self.covariance_type == 'spherical':
            return np.eye(len(self.means[state_id])) * self.covars[state_id]
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")

    def _setup_distributions(self):
        """Pre-compute scipy multivariate normal distributions for each state."""
        self.distributions = []
        for i in range(self.n_components):
            mean = self.means[i]
            cov = self._get_covariance(i)
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

    def get_mahalanobis_distance(self, observation: np.ndarray, log_stabilize: bool = True) -> float:
        """
        Calculate Mahalanobis distance from the "safe" state.

        The Mahalanobis distance is a measure of how far the observation is from the
        safe state center, accounting for the covariance structure.

        Formula: D_M(x) = sqrt((x - μ_safe)^T * Σ_safe^-1 * (x - μ_safe))

        Args:
            observation: 1D array of shape (n_features,)
            log_stabilize: If True, return ln(1 + raw_distance) (default: True)

        Returns:
            Mahalanobis distance (log-stabilized if log_stabilize=True)
        """
        observation = np.asarray(observation).flatten()

        # Calculate Mahalanobis distance
        diff = observation - self.safe_mean
        raw_distance = np.sqrt(diff.T @ self.safe_cov_inv @ diff)

        if log_stabilize:
            return np.log1p(raw_distance)  # ln(1 + raw_distance)
        else:
            return raw_distance

    def get_risk_score(self, observation: np.ndarray) -> float:
        """
        Calculate normalized risk score in [0, 1] range.

        This is the production-ready risk score that should be used in live trading.

        Args:
            observation: 1D array of shape (n_features,)

        Returns:
            Normalized risk score in [0, 1]
            - 0.0 = At safe regime center (lowest risk)
            - 1.0 = Maximum distance from safe regime (highest risk)
        """
        # Get log-stabilized Mahalanobis distance
        mahal_distance = self.get_mahalanobis_distance(observation, log_stabilize=True)

        # Normalize using training data statistics
        if self.mahal_scaler_params:
            min_val = self.mahal_scaler_params.get('min', 0.0)
            max_val = self.mahal_scaler_params.get('max', 1.0)

            if max_val > min_val:
                risk_score = (mahal_distance - min_val) / (max_val - min_val)
            else:
                risk_score = 0.5  # Fallback if constant
        else:
            # No scaler params: return raw distance (not recommended for production)
            risk_score = mahal_distance

        # Clip to [0, 1]
        risk_score = np.clip(risk_score, 0.0, 1.0)

        return float(risk_score)

    def reset(self):
        """Reset state probabilities to initial distribution."""
        self.state_probs = self.startprob.copy()


class MLRiskRegimeStepHMM(BaseStep):
    """Pipeline step to construct risk-based regime labels using HMM with Mahalanobis scoring."""

    def __init__(self, step_name: str = "ml_risk_regime_step_hmm"):
        """Initialize the ML Risk Regime HMM step with versioned artifacts enabled."""
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLRiskRegimeStepHMM") if hasattr(logger, "getChild") else logger
        self._cached_market_data = None
        self._cached_market_source = None
        self._cached_market_cache_key = None
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute risk regime detection using HMM.

        Expected config keys:
            - symbol: Trading symbol (e.g., 'ETHUSDT')
            - exchange: Exchange name (e.g., 'binance')
            - regime_timeframe: Timeframe for regime detection (default: '1h')
            - n_regimes: Number of regimes (default: 3)
            - safe_state_id: ID of the "safe" state for Mahalanobis distance (default: 0)
        """
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "1h")))
            n_regimes = int(config.get("n_regimes", 3))
            safe_state_id = int(config.get("safe_state_id", 0))

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(timeframe={regime_timeframe}, n_regimes={n_regimes})"
            )

            # Load market data
            market_data = await self._load_market_data(config, regime_timeframe)

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
                f"📅 Temporal split config: "
                f"burn-in={split_config.burnin.start if split_config.burnin else 'None'} → "
                f"{split_config.burnin.effective_end if split_config.burnin else 'None'}, "
                f"train={split_config.training.start} → {split_config.training.effective_end}, "
                f"val={split_config.validation.start} → {split_config.validation.effective_end}, "
                f"test={split_config.test.start} → {split_config.test.effective_end}"
            )

            # Generate risk features
            risk_df = self._generate_risk_features(market_data, config)

            # Train HMM with GMM warm-start
            hmm_model, regime_labels, training_metrics = self._train_hmm_regimes(
                risk_df, config, n_regimes
            )

            # Calculate normalized Mahalanobis distances [0, 1]
            mahal_distances, mahal_scaler_params = self._calculate_mahalanobis_distances(
                risk_df, hmm_model, safe_state_id, split_config
            )

            # Add regime labels and normalized Mahalanobis risk score to dataframe
            risk_df['risk_regime'] = regime_labels
            risk_df['risk_score'] = mahal_distances  # Normalized [0, 1] risk score

            # Persist training data for downstream specialist loaders via versioned artifacts
            try:
                self.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=regime_timeframe,
                    direction=config.get("direction", "long"),
                    model="regime_risk_hmm",
                )

                risk_to_save = risk_df.reset_index().rename(
                    columns={risk_df.index.name or "index": "timestamp"}
                )
                training_metadata = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "safe_state_id": safe_state_id,
                }
                risk_hmm_artifact_name = f"ml_risk_hmm_training_data_{regime_timeframe}"
                self._save_artifact(
                    data=risk_to_save,
                    artifact_name=risk_hmm_artifact_name,
                    artifact_type="data",
                    metadata=training_metadata,
                )
            except Exception as save_exc:
                tprint_warning(
                    f"Failed to save ML risk HMM training data artifact (non-fatal): {save_exc}"
                )

            # Calculate forward returns and Sharpe ratios
            forward_metrics = self._calculate_forward_returns_and_sharpe(
                risk_df, regime_labels, horizons=[4]  # 4x15m = 1h
            )

            # Calculate temporal metrics
            temporal_metrics = self._calculate_temporal_metrics(
                labels=regime_labels,
                transition_matrix=hmm_model.transmat_,
                n_regimes=n_regimes
            )

            # Generate comprehensive reports
            self._generate_reports(
                risk_df=risk_df,
                regime_labels=regime_labels,
                hmm_model=hmm_model,
                forward_metrics=forward_metrics,
                temporal_metrics=temporal_metrics,
                training_metrics=training_metrics,
                config=config,
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                safe_state_id=safe_state_id
            )

            # Save HMM model and LiveHMM wrapper
            model_paths = self._save_models(
                hmm_model=hmm_model,
                risk_features=risk_df,
                regime_labels=regime_labels,
                temporal_metrics=temporal_metrics,
                config=config,
                symbol=symbol,
                timeframe=regime_timeframe,
                safe_state_id=safe_state_id,
                mahal_scaler_params=mahal_scaler_params,
                split_config=split_config
            )

            duration = time.time() - start_time
            tprint_success(f"✅ {self.step_name} completed in {duration:.2f}s")

            # BaseStep.run expects a boolean 'success' flag to determine
            # whether the step completed successfully. Without it, the step
            # is treated as a failure even if all work was done correctly.
            return {
                "success": True,
                "status": "success",
                "duration": duration,
                "regime_labels": regime_labels,
                "mahal_distances": mahal_distances,
                "forward_metrics": forward_metrics,
                "temporal_metrics": temporal_metrics,
                "model_paths": model_paths,
            }

        except Exception as exc:
            tprint_error(f"❌ {self.step_name} failed: {exc}")
            import traceback
            traceback.print_exc()
            raise

    async def _load_market_data(self, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Load market data for the specified timeframe."""
        symbol = config.get("symbol")
        exchange = config.get("exchange")
        exec_mode = str(config.get("execution_mode", "")).lower()

        cache_key = (symbol, exchange, timeframe, exec_mode)

        if (self._cached_market_data is not None and
            self._cached_market_cache_key == cache_key):
            tprint_info("♻️ Reusing cached market data")
            return self._cached_market_data.copy()

        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
            light_mode_filter=False,
            skip_artifacts=True,
        )

        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            raise ValueError("Loaded market data is empty or not a DataFrame")

        # Ensure DatetimeIndex
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)
            if market_data.index.tz is not None:
                market_data.index = market_data.index.tz_convert(None)

        self._cached_market_data = market_data.copy()
        self._cached_market_source = market_source
        self._cached_market_cache_key = cache_key

        tprint_info(
            f"✅ Loaded market data from {market_source}: {market_data.shape} "
            f"({market_data.index.min()} → {market_data.index.max()})"
        )

        return market_data

    def _generate_risk_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate the 5 core risk features via the feature bank.

        Delegates to ``generate_risk_regime_features`` in the
        ``feature_generation.categories`` package so that feature
        definitions are centralized.
        """

        risk_df = df.copy()

        feature_frame = generate_risk_regime_features(risk_df, config)
        feature_cols = [
            'parkinson_volatility',
            'hurst_exponent',
            'rolling_kurtosis',
            'rolling_skewness',
            'volatility_of_volatility',
        ]

        for col in feature_cols:
            if col in feature_frame.columns:
                risk_df[col] = feature_frame[col].values

        tprint_success(f"✅ Generated {len(feature_cols)} risk features with windows 30-50 bars")

        return risk_df

    def _calculate_hurst_exponent(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Hurst exponent using R/S analysis on a price series."""
        values = series.astype(float).values
        hurst_values: List[float] = []

        for i in range(len(values)):
            if i < window:
                hurst_values.append(np.nan)
                continue

            window_data = values[i - window:i]

            try:
                # R/S analysis
                mean_val = np.mean(window_data)
                deviations = window_data - mean_val
                cumulative_deviations = np.cumsum(deviations)

                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                S = np.std(window_data)

                if S > 0 and R > 0:
                    hurst = np.log(R / S) / np.log(window)
                    hurst_values.append(hurst)
                else:
                    hurst_values.append(0.5)
            except Exception:
                hurst_values.append(0.5)

        return pd.Series(hurst_values, index=series.index)

    def _train_hmm_regimes(
        self,
        risk_df: pd.DataFrame,
        config: Dict[str, Any],
        n_regimes: int
    ) -> Tuple[GaussianHMM, np.ndarray, Dict[str, Any]]:
        """
        Train HMM model on risk features with GMM warm-start and strided training.

        Returns:
            (trained_hmm_model, regime_labels, training_metrics)
        """
        tprint_info("=" * 80)
        tprint_info("🎯 HMM RISK REGIME DETECTION (Temporal Structure Learning)")
        tprint_info("=" * 80)

        # Select risk features
        feature_cols = [
            'parkinson_volatility',
            'rolling_kurtosis',
            'rolling_skewness',
            'volatility_of_volatility',
        ]

        # Filter to available features
        available_features = [c for c in feature_cols if c in risk_df.columns]
        if not available_features:
            raise ValueError("No risk features available for HMM training")

        risk_features = risk_df[available_features].copy()

        # Drop rows with NaN
        valid_mask = risk_features.notna().all(axis=1)
        risk_features_clean = risk_features[valid_mask].copy()

        tprint_info(
            f"📊 Using {len(available_features)} risk features: {available_features}"
        )
        tprint_info(
            f"  Valid samples: {len(risk_features_clean)}/{len(risk_df)}"
        )

        # ===== OPTIMIZATION 1: Strided Training =====
        hmm_stride = int(config.get("hmm_stride", 10))
        tprint_info(f"🔄 Applying strided training: Using every {hmm_stride}th data point")

        strided_indices = np.arange(0, len(risk_features_clean), hmm_stride)
        risk_features_strided = risk_features_clean.iloc[strided_indices].copy()

        tprint_info(
            f"  Strided data: {len(risk_features_strided)} samples "
            f"(down from {len(risk_features_clean)})"
        )

        # ===== OPTIMIZATION 2: GMM Pre-computation for HMM Initialization =====
        tprint_info(
            f"🚀 Step 1: Fast GMM pre-computation for HMM warm-start "
            f"(features: {risk_features_strided.shape[1]})"
        )

        gmm_for_init = GaussianMixture(
            n_components=n_regimes,
            covariance_type="diag",
            n_init=5,
            max_iter=50,
            reg_covar=1e-3,
            random_state=42
        )

        gmm_start = time.time()
        gmm_for_init.fit(risk_features_strided)
        gmm_duration = time.time() - gmm_start

        gmm_means = gmm_for_init.means_

        tprint_success(
            f"  ✅ GMM pre-computation completed in {gmm_duration:.2f}s "
            f"(warm-start initialization ready)"
        )

        # Optional: HMM warm-start from latest saved model
        warm_start_enabled = bool(config.get("hmm_enable_warm_start", True))
        warm_start_model_path = config.get("hmm_warm_start_model_path")
        previous_hmm: Optional[GaussianHMM] = None

        if warm_start_enabled:
            try:
                candidate_paths: List[Path] = []
                if isinstance(warm_start_model_path, str) and warm_start_model_path:
                    candidate_paths = [Path(warm_start_model_path)]
                else:
                    symbol_ws = str(config.get("symbol", "ETHUSDT"))
                    timeframe_ws = str(config.get("regime_timeframe", config.get("timeframe", "1h")))
                    base_dir = Path("versioned_artifacts/risk_regime_models")
                    pattern = f"risk_hmm_{symbol_ws}_{timeframe_ws}_*.pkl"
                    if base_dir.exists():
                        candidate_paths = sorted(base_dir.glob(pattern))

                for path in reversed(candidate_paths):
                    try:
                        loaded = joblib.load(path)
                        if isinstance(loaded, dict) and "hmm_model" in loaded:
                            candidate = loaded["hmm_model"]
                        else:
                            candidate = loaded
                        if isinstance(candidate, GaussianHMM):
                            previous_hmm = candidate
                            tprint_info(f"♻️ Using previous HMM for warm-start: {path}")
                            break
                    except Exception as load_exc:
                        tprint_warning(f"HMM warm-start: failed to load {path}: {load_exc}")
            except Exception as ws_exc:
                tprint_warning(f"HMM warm-start disabled due to error, falling back to GMM init: {ws_exc}")
                previous_hmm = None

        # ===== OPTIMIZATION 3: HMM Training with GMM Injection =====
        tprint_info(
            f"🧠 Step 2: Training HMM with {n_regimes} regimes "
            f"(full covariance for risk features)"
        )

        # OPTIMIZED: Reduced n_iter from 200 to 50 for faster training
        hmm_n_iter = int(config.get("hmm_n_iter", 50))  # Reduced from 200
        hmm_tol = float(config.get("hmm_tol", 1e-3))
        hmm_min_covar = float(config.get("hmm_min_covar", 0.001))

        hmm: GaussianHMM

        if (
            previous_hmm is not None
            and getattr(previous_hmm, "n_components", None) == n_regimes
            and getattr(previous_hmm, "n_features", risk_features_strided.shape[1]) == risk_features_strided.shape[1]
        ):
            warm_starter = HMMWarmStarter()
            # OPTIMIZED: Use 'diag' covariance for 5-10x faster training
            hmm = warm_starter.create_warm_started_hmm(
                n_components=n_regimes,
                n_features=risk_features_strided.shape[1],
                previous_hmm=previous_hmm,
                covariance_type="diag",  # OPTIMIZED from 'full'
                n_iter=hmm_n_iter,
                random_state=42,
            )
            try:
                hmm.min_covar = hmm_min_covar
            except Exception:
                pass
            tprint_info("  ♻️ Warm-starting HMM from previous trained model")
        else:
            # OPTIMIZED: Use 'diag' covariance for 5-10x faster training
            hmm = GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",  # OPTIMIZED from 'full'
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                init_params='stmc',
                min_covar=hmm_min_covar,
                random_state=42
            )

            # Inject GMM means into HMM (warm start)
            hmm.means_ = gmm_means.copy()
            hmm.init_params = 'stc'  # Only init start, trans, covars

        tprint_info(
            f"  HMM configuration: n_components={n_regimes}, "
            f"covariance_type='diag' (OPTIMIZED), n_iter={hmm_n_iter}, "
            f"tol={hmm_tol}, min_covar={hmm_min_covar}"
        )

        # Train HMM
        hmm_features_strided_array = risk_features_strided.values

        hmm_start = time.time()
        try:
            hmm.fit(
                hmm_features_strided_array,
                lengths=[len(hmm_features_strided_array)]
            )

            # Get labels on full data
            hmm_full_array = risk_features_clean.values
            regime_labels_clean = hmm.predict(
                hmm_full_array,
                lengths=[len(hmm_full_array)]
            )

        except ValueError as hmm_exc:
            tprint_warning(
                f"HMM fit failed: {hmm_exc}. Retrying with increased min_covar"
            )

            # OPTIMIZED: Use 'diag' covariance for 5-10x faster training
            hmm = GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",  # OPTIMIZED from 'full'
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                init_params='stmc',
                min_covar=hmm_min_covar * 10.0,
                random_state=42
            )

            hmm_features_strided_retry = risk_features_strided.astype("float64").values
            hmm_full_retry = risk_features_clean.astype("float64").values

            hmm.fit(
                hmm_features_strided_retry,
                lengths=[len(hmm_features_strided_retry)]
            )
            regime_labels_clean = hmm.predict(
                hmm_full_retry,
                lengths=[len(hmm_full_retry)]
            )

        hmm_duration = time.time() - hmm_start
        tprint_success(f"  ✅ HMM training completed in {hmm_duration:.2f}s")

        # Expand labels back to full dataframe
        regime_labels = np.full(len(risk_df), -1, dtype=int)
        regime_labels[valid_mask] = regime_labels_clean

        # Log transition matrix
        transition_matrix = hmm.transmat_
        tprint_info("=" * 80)
        tprint_info("📊 HMM TRANSITION MATRIX (Regime Dynamics):")
        tprint_info("=" * 80)
        for i in range(n_regimes):
            trans_probs = " ".join([f"{p:.3f}" for p in transition_matrix[i]])
            tprint_info(f"  Regime {i} → [{trans_probs}]")
        tprint_info("=" * 80)

        # Calculate regime distribution
        unique, counts = np.unique(regime_labels[regime_labels >= 0], return_counts=True)
        regime_distribution = {int(r): int(c) for r, c in zip(unique, counts)}

        training_metrics = {
            'n_regimes': n_regimes,
            'n_samples': len(risk_features_clean),
            'n_features': len(available_features),
            'feature_names': available_features,
            'regime_distribution': regime_distribution,
            'training_duration_gmm': gmm_duration,
            'training_duration_hmm': hmm_duration,
            'hmm_converged': hmm.monitor_.converged,
            'hmm_n_iter': hmm.monitor_.iter,
        }

        tprint_success(
            f"✅ HMM training complete: {n_regimes} regimes, "
            f"converged in {hmm.monitor_.iter} iterations"
        )

        return hmm, regime_labels, training_metrics

    def _calculate_mahalanobis_distances(
        self,
        risk_df: pd.DataFrame,
        hmm_model: GaussianHMM,
        safe_state_id: int,
        split_config: Optional[TemporalSplitConfig] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate log-stabilized Mahalanobis distance from the "safe" state.
        Normalized to [0, 1] range using training data statistics (no data leakage).

        Formula: D_M(x) = sqrt((x - μ_safe)^T * Σ_safe^-1 * (x - μ_safe))
        Log-stabilized: ln(1 + D_M(x))
        Normalized: (distance - min_train) / (max_train - min_train)

        Args:
            risk_df: DataFrame with risk features
            hmm_model: Trained HMM model
            safe_state_id: ID of the "safe" state
            split_config: Temporal split config (to fit scaler on training data only)

        Returns:
            Tuple of (normalized_distances, scaler_params)
            - normalized_distances: Array of distances in [0, 1]
            - scaler_params: Dict with min/max from training data
        """
        tprint_info(f"📐 Calculating Mahalanobis distances from safe state (regime {safe_state_id})...")

        feature_cols = [
            'parkinson_volatility',
            'rolling_kurtosis',
            'rolling_skewness',
            'volatility_of_volatility',
        ]

        available_features = [c for c in feature_cols if c in risk_df.columns]
        risk_features = risk_df[available_features].values

        # Extract safe state parameters
        safe_mean = hmm_model.means_[safe_state_id]

        if hmm_model.covariance_type == 'full':
            safe_cov = hmm_model.covars_[safe_state_id]
        elif hmm_model.covariance_type == 'diag':
            safe_cov = np.diag(hmm_model.covars_[safe_state_id])
        else:
            raise ValueError(f"Unsupported covariance type: {hmm_model.covariance_type}")

        # Calculate inverse covariance
        try:
            safe_cov_inv = np.linalg.inv(safe_cov)
        except np.linalg.LinAlgError:
            # Add regularization if singular
            tprint_warning("Safe state covariance is singular, adding regularization")
            safe_cov += np.eye(safe_cov.shape[0]) * 1e-6
            safe_cov_inv = np.linalg.inv(safe_cov)

        # Calculate raw Mahalanobis distances (log-stabilized)
        mahal_distances_raw = np.full(len(risk_features), np.nan)

        for i in range(len(risk_features)):
            if np.any(np.isnan(risk_features[i])):
                continue

            try:
                diff = risk_features[i] - safe_mean
                raw_distance = np.sqrt(diff.T @ safe_cov_inv @ diff)

                # Log-stabilize: ln(1 + raw_distance)
                mahal_distances_raw[i] = np.log1p(raw_distance)
            except:
                mahal_distances_raw[i] = np.nan

        valid_count = np.isfinite(mahal_distances_raw).sum()
        tprint_info(
            f"  Raw distances calculated: {valid_count}/{len(mahal_distances_raw)} valid samples"
        )

        # Normalize to [0, 1] using training data only (prevent data leakage)
        if split_config is not None:
            # Get training data mask
            train_start = split_config.training.start
            train_end = split_config.training.effective_end
            train_mask = (risk_df.index >= train_start) & (risk_df.index <= train_end)

            # Fit scaler on training data only
            train_distances = mahal_distances_raw[train_mask]
            train_distances_valid = train_distances[np.isfinite(train_distances)]

            if len(train_distances_valid) > 0:
                min_val = float(np.min(train_distances_valid))
                max_val = float(np.max(train_distances_valid))
            else:
                tprint_warning("No valid training distances, using global statistics")
                valid_distances = mahal_distances_raw[np.isfinite(mahal_distances_raw)]
                min_val = float(np.min(valid_distances))
                max_val = float(np.max(valid_distances))
        else:
            # No split config: use all data (less ideal, but backward compatible)
            tprint_warning("No split_config provided, fitting scaler on all data")
            valid_distances = mahal_distances_raw[np.isfinite(mahal_distances_raw)]
            min_val = float(np.min(valid_distances))
            max_val = float(np.max(valid_distances))

        # Normalize all distances to [0, 1]
        mahal_distances_normalized = np.full(len(mahal_distances_raw), np.nan)
        valid_mask = np.isfinite(mahal_distances_raw)

        if max_val > min_val:
            mahal_distances_normalized[valid_mask] = (
                (mahal_distances_raw[valid_mask] - min_val) / (max_val - min_val)
            )
        else:
            # All distances are the same (edge case)
            tprint_warning(f"All distances are constant ({min_val:.6f}), setting to 0.5")
            mahal_distances_normalized[valid_mask] = 0.5

        # Clip to [0, 1] to handle any out-of-range values in val/test
        mahal_distances_normalized = np.clip(mahal_distances_normalized, 0.0, 1.0)

        # Store scaler parameters
        scaler_params = {
            'min': min_val,
            'max': max_val,
            'fitted_on': 'training_data' if split_config is not None else 'all_data'
        }

        tprint_success(
            f"✅ Normalized Mahalanobis distances to [0, 1]: "
            f"min={min_val:.6f}, max={max_val:.6f}"
        )
        tprint_info(
            f"  Range in data: [{np.nanmin(mahal_distances_normalized):.3f}, "
            f"{np.nanmax(mahal_distances_normalized):.3f}]"
        )
        tprint_info(
            f"  Interpretation: 0.0 = At safe regime center, 1.0 = Maximum risk/distance"
        )

        return mahal_distances_normalized, scaler_params

    def _calculate_forward_returns_and_sharpe(
        self,
        df: pd.DataFrame,
        regime_labels: np.ndarray,
        horizons: List[int] = [4]  # 4x15m = 1h for 1h data
    ) -> Dict[str, Any]:
        """
        Calculate forward returns and Sharpe ratios per regime at specified horizons.

        For 1h data:
        - horizon=4 means 4 hours forward

        Args:
            df: DataFrame with 'close' prices
            regime_labels: Regime labels aligned with df
            horizons: List of forward horizons in bars (default: [4] for 4h)

        Returns:
            Dictionary with forward returns and Sharpe ratios per regime
        """
        if 'close' not in df.columns:
            tprint_warning("No 'close' column found; skipping forward returns calculation")
            return {}

        tprint_info("📈 Calculating forward returns and Sharpe ratios per regime...")

        close_prices = df['close'].values
        results = {}

        for horizon in horizons:
            # For 1h data, horizon=4 means 4h forward
            horizon_label = f"{horizon}h"

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
                        'sharpe_annualized': 0.0,
                        'count': 0
                    }
                    continue

                mean_return = np.mean(regime_returns)
                std_return = np.std(regime_returns)
                sharpe_ratio = mean_return / (std_return + 1e-8)

                # Annualize Sharpe ratio (1h bars: 24 bars/day, ~8760 bars/year)
                bars_per_year = 8760
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

        tprint_success(f"✅ Calculated forward returns and Sharpe ratios for {len(horizons)} horizons")

        return results

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
        - Per-regime duration statistics
        - Transition probabilities

        Args:
            labels: Regime labels array
            transition_matrix: HMM transition matrix (n_regimes x n_regimes)
            n_regimes: Number of regimes

        Returns:
            Dictionary with temporal metrics
        """
        tprint_info("⏱️  Calculating temporal metrics...")

        # Filter out invalid labels
        valid_labels = labels[labels >= 0]

        # Calculate regime durations
        regime_durations = []
        if len(valid_labels) > 0:
            current_regime = valid_labels[0]
            current_duration = 1

            for i in range(1, len(valid_labels)):
                if valid_labels[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = valid_labels[i]
                    current_duration = 1

            regime_durations.append(current_duration)

        avg_duration_bars = np.mean(regime_durations) if regime_durations else 0.0

        # Calculate flip-flop rate
        regime_changes = np.sum(valid_labels[1:] != valid_labels[:-1])
        flip_flop_rate = regime_changes / len(valid_labels) if len(valid_labels) > 0 else 0.0

        # Calculate stability score from transition matrix diagonal
        diagonal_probs = np.diag(transition_matrix)
        stability_score = np.mean(diagonal_probs)

        # Per-regime self-transition probabilities
        regime_stability = {}
        for regime_id in range(n_regimes):
            regime_stability[regime_id] = float(transition_matrix[regime_id, regime_id])

        # Duration statistics per regime
        regime_duration_stats = {}
        for regime_id in range(n_regimes):
            regime_mask = (valid_labels == regime_id)
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

        tprint_success(
            f"✅ Temporal metrics calculated: "
            f"Avg Duration={avg_duration_bars:.2f} bars, "
            f"Flip-Flop={flip_flop_rate:.4f}"
        )

        return {
            'avg_duration_bars': float(avg_duration_bars),
            'flip_flop_rate': float(flip_flop_rate),
            'stability_score': float(stability_score),
            'regime_changes': int(regime_changes),
            'total_samples': int(len(valid_labels)),
            'regime_stability': regime_stability,
            'regime_duration_stats': regime_duration_stats,
            'transition_matrix': transition_matrix.tolist(),
        }

    def _generate_reports(
        self,
        risk_df: pd.DataFrame,
        regime_labels: np.ndarray,
        hmm_model: GaussianHMM,
        forward_metrics: Dict[str, Any],
        temporal_metrics: Dict[str, Any],
        training_metrics: Dict[str, Any],
        config: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        safe_state_id: int
    ):
        """Generate comprehensive CSV and MD reports."""
        tprint_info("📄 Generating comprehensive reports...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("outcomes").mkdir(exist_ok=True)

        # 1. Generate CSV report with WCoV metrics
        csv_path = self._generate_csv_report(
            risk_df=risk_df,
            regime_labels=regime_labels,
            forward_metrics=forward_metrics,
            temporal_metrics=temporal_metrics,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp
        )

        # 2. Generate MD report with all metrics
        md_path = self._generate_md_report(
            risk_df=risk_df,
            regime_labels=regime_labels,
            hmm_model=hmm_model,
            forward_metrics=forward_metrics,
            temporal_metrics=temporal_metrics,
            training_metrics=training_metrics,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            timestamp=timestamp,
            safe_state_id=safe_state_id
        )

        tprint_success(f"✅ Reports generated:")
        tprint_success(f"  📊 CSV: {csv_path}")
        tprint_success(f"  📝 MD: {md_path}")

    def _generate_csv_report(
        self,
        risk_df: pd.DataFrame,
        regime_labels: np.ndarray,
        forward_metrics: Dict[str, Any],
        temporal_metrics: Dict[str, Any],
        symbol: str,
        timeframe: str,
        timestamp: str
    ) -> str:
        """Generate CSV report with WCoV metrics and regime statistics."""

        csv_rows = []

        # Feature columns for WCoV analysis
        feature_cols = [
            'parkinson_volatility',
            'rolling_kurtosis',
            'rolling_skewness',
            'volatility_of_volatility',
            'risk_score',  # Normalized [0, 1] risk score
        ]

        available_features = [c for c in feature_cols if c in risk_df.columns]

        valid_mask = regime_labels >= 0
        df_valid = risk_df[valid_mask].copy()
        labels_valid = regime_labels[valid_mask]

        unique_regimes = np.unique(labels_valid)

        # Calculate WCoV metrics for each feature
        for feature in available_features:
            if feature not in df_valid.columns:
                continue

            feature_values = df_valid[feature].values

            # Global metrics
            wcov_between = self._calculate_wcov_between(labels_valid, feature_values)
            wcov_within = self._calculate_wcov_within(labels_valid, feature_values)
            wcov_ratio = wcov_between / (wcov_within + 1e-8)

            csv_rows.append({
                'metric': feature,
                'scope': 'GLOBAL',
                'regime': 'ALL',
                'n_samples': len(feature_values),
                'wcov_between': wcov_between,
                'wcov_within': wcov_within,
                'wcov_ratio': wcov_ratio,
                'mean': np.nanmean(feature_values),
                'std': np.nanstd(feature_values),
            })

            # Per-regime metrics
            for regime_id in unique_regimes:
                regime_mask = labels_valid == regime_id
                regime_values = feature_values[regime_mask]

                if len(regime_values) < 2:
                    continue

                csv_rows.append({
                    'metric': feature,
                    'scope': 'REGIME',
                    'regime': int(regime_id),
                    'n_samples': len(regime_values),
                    'wcov_between': np.nan,
                    'wcov_within': np.nan,
                    'wcov_ratio': np.nan,
                    'mean': np.nanmean(regime_values),
                    'std': np.nanstd(regime_values),
                })

        csv_df = pd.DataFrame(csv_rows)
        csv_path = f"outcomes/ml_risk_regime_hmm_wcov_{symbol}_{timeframe}_{timestamp}.csv"
        csv_df.to_csv(csv_path, index=False)

        return csv_path

    def _calculate_wcov_between(self, labels: np.ndarray, values: np.ndarray) -> float:
        """Calculate between-regime coefficient of variation."""
        unique_regimes = np.unique(labels)
        regime_means = []

        for regime_id in unique_regimes:
            regime_mask = labels == regime_id
            regime_values = values[regime_mask]
            if len(regime_values) > 0:
                regime_means.append(np.nanmean(regime_values))

        if len(regime_means) < 2:
            return 0.0

        mean_of_means = np.mean(regime_means)
        std_of_means = np.std(regime_means)

        return std_of_means / (abs(mean_of_means) + 1e-8)

    def _calculate_wcov_within(self, labels: np.ndarray, values: np.ndarray) -> float:
        """Calculate within-regime coefficient of variation."""
        unique_regimes = np.unique(labels)
        within_cvs = []

        for regime_id in unique_regimes:
            regime_mask = labels == regime_id
            regime_values = values[regime_mask]

            if len(regime_values) > 1:
                regime_mean = np.nanmean(regime_values)
                regime_std = np.nanstd(regime_values)
                cv = regime_std / (abs(regime_mean) + 1e-8)
                within_cvs.append(cv)

        if len(within_cvs) == 0:
            return 1.0

        return np.mean(within_cvs)

    def _generate_md_report(
        self,
        risk_df: pd.DataFrame,
        regime_labels: np.ndarray,
        hmm_model: GaussianHMM,
        forward_metrics: Dict[str, Any],
        temporal_metrics: Dict[str, Any],
        training_metrics: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        timestamp: str,
        safe_state_id: int
    ) -> str:
        """Generate comprehensive Markdown report."""

        md_path = f"outcomes/ml_risk_regime_hmm_report_{symbol}_{timeframe}_{timestamp}.md"

        with open(md_path, "w") as f:
            f.write("# ML Risk Regime HMM Report\n\n")
            f.write(f"**Symbol**: {symbol} | **Exchange**: {exchange} | **Timeframe**: {timeframe}\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Safe State**: Regime {safe_state_id}\n\n")
            f.write("---\n\n")

            # Training Summary
            f.write("## Training Summary\n\n")
            f.write(f"- **Model**: Hidden Markov Model (HMM) with full covariance\n")
            f.write(f"- **Number of Regimes**: {training_metrics['n_regimes']}\n")
            f.write(f"- **Number of Features**: {training_metrics['n_features']}\n")
            f.write(f"- **Features**: {', '.join(training_metrics['feature_names'])}\n")
            f.write(f"- **Training Samples**: {training_metrics['n_samples']}\n")
            f.write(f"- **HMM Converged**: {training_metrics['hmm_converged']}\n")
            f.write(f"- **HMM Iterations**: {training_metrics['hmm_n_iter']}\n")
            f.write(f"- **Training Duration (GMM)**: {training_metrics['training_duration_gmm']:.2f}s\n")
            f.write(f"- **Training Duration (HMM)**: {training_metrics['training_duration_hmm']:.2f}s\n\n")

            # Regime Distribution
            f.write("## Regime Distribution\n\n")
            f.write("| Regime | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")
            total_samples = sum(training_metrics['regime_distribution'].values())
            for regime_id in sorted(training_metrics['regime_distribution'].keys()):
                count = training_metrics['regime_distribution'][regime_id]
                pct = 100.0 * count / total_samples
                f.write(f"| {regime_id} | {count} | {pct:.2f}% |\n")
            f.write("\n---\n\n")

            # Temporal Metrics
            f.write("## Temporal Metrics (HMM Transition Dynamics)\n\n")
            f.write(f"**Average Duration**: {temporal_metrics['avg_duration_bars']:.2f} bars\n\n")
            f.write(f"**Flip-Flop Rate**: {temporal_metrics['flip_flop_rate']:.4f}\n\n")
            f.write(f"**Stability Score**: {temporal_metrics['stability_score']:.4f}\n\n")
            f.write(f"**Regime Changes**: {temporal_metrics['regime_changes']}\n\n")
            f.write(f"**Total Samples**: {temporal_metrics['total_samples']}\n\n")

            # Transition Matrix
            trans_matrix = temporal_metrics['transition_matrix']
            f.write("### Transition Matrix\n\n")
            f.write("Probability of transitioning from regime i (row) to regime j (column):\n\n")
            f.write("| From → To | " + " | ".join([f"Regime {i}" for i in range(len(trans_matrix))]) + " |\n")
            f.write("|" + "|".join(["-----------"] * (len(trans_matrix) + 1)) + "|\n")
            for i, row in enumerate(trans_matrix):
                f.write(f"| **Regime {i}** | " + " | ".join([f"{p:.4f}" for p in row]) + " |\n")
            f.write("\n")

            # Duration Statistics
            duration_stats = temporal_metrics['regime_duration_stats']
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
            f.write("\n---\n\n")

            # Forward Returns & Sharpe Ratios
            f.write("## Forward Returns & Sharpe Ratios Per Regime\n\n")
            for horizon_label, regime_stats in forward_metrics.items():
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

            # Risk Score Statistics
            f.write("## Normalized Risk Score [0, 1]\n\n")
            if 'risk_score' in risk_df.columns:
                risk_values = risk_df['risk_score'].dropna()
                f.write(f"**Safe State ID**: {safe_state_id}\n\n")
                f.write(f"**Mean Risk Score**: {risk_values.mean():.4f}\n\n")
                f.write(f"**Std Risk Score**: {risk_values.std():.4f}\n\n")
                f.write(f"**Median Risk Score**: {risk_values.median():.4f}\n\n")
                f.write(f"**95th Percentile**: {risk_values.quantile(0.95):.4f}\n\n")
                f.write(f"**99th Percentile**: {risk_values.quantile(0.99):.4f}\n\n")

                # Per-regime risk score statistics
                f.write("### Risk Score by Regime\n\n")
                f.write("| Regime | Mean Risk | Std Risk | Median | 95th Pct | Samples |\n")
                f.write("|--------|-----------|----------|--------|----------|-------|\n")

                valid_mask = regime_labels >= 0
                for regime_id in np.unique(regime_labels[valid_mask]):
                    regime_mask = (regime_labels == regime_id) & risk_df['risk_score'].notna()
                    regime_risk = risk_df.loc[regime_mask, 'risk_score']

                    if len(regime_risk) > 0:
                        f.write(
                            f"| {regime_id} | {regime_risk.mean():.4f} | "
                            f"{regime_risk.std():.4f} | {regime_risk.median():.4f} | "
                            f"{regime_risk.quantile(0.95):.4f} | {len(regime_risk)} |\n"
                        )
                f.write("\n")

            f.write("---\n\n")
            f.write("*Report generated by ml_risk_regime_step_hmm with normalized [0,1] risk scores*\n")

        return md_path

    def _save_models(
        self,
        hmm_model: GaussianHMM,
        risk_features: pd.DataFrame,
        regime_labels: np.ndarray,
        temporal_metrics: Dict[str, Any],
        config: Dict[str, Any],
        symbol: str,
        timeframe: str,
        safe_state_id: int,
        mahal_scaler_params: Dict[str, float],
        split_config: Optional[TemporalSplitConfig] = None
    ) -> Dict[str, str]:
        """Save HMM model, LiveHMM wrapper, and artifacts with scaler parameters."""
        import joblib

        tprint_info("💾 Saving HMM model and artifacts...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("versioned_artifacts/risk_regime_models").mkdir(parents=True, exist_ok=True)

        # 1. Save HMM model
        hmm_path = f"versioned_artifacts/risk_regime_models/risk_hmm_{symbol}_{timeframe}_{timestamp}.pkl"

        feature_cols = [
            'parkinson_volatility',
            'hurst_exponent',
            'rolling_kurtosis',
            'rolling_skewness',
            'volatility_of_volatility'
        ]
        available_features = [c for c in feature_cols if c in risk_features.columns]

        hmm_model_data = {
            "hmm_model": hmm_model,
            "feature_names": available_features,
            "safe_state_id": safe_state_id,
            "transition_matrix": hmm_model.transmat_,
            "temporal_metrics": temporal_metrics,
            "mahal_scaler_params": mahal_scaler_params,  # Scaler for [0, 1] normalization
            "symbol": symbol,
            "timeframe": timeframe,
            "trained_at": timestamp,
            "version": "v3_normalized_risk_score",
        }

        # Add burn-in metadata if available
        if split_config is not None:
            hmm_model_data["training_start"] = str(split_config.training.start)
            hmm_model_data["training_end"] = str(split_config.training.effective_end)
            hmm_model_data["validation_start"] = str(split_config.validation.start)
            hmm_model_data["validation_end"] = str(split_config.validation.effective_end)
            hmm_model_data["test_start"] = str(split_config.test.start)
            hmm_model_data["test_end"] = str(split_config.test.effective_end)
            if split_config.burnin is not None:
                hmm_model_data["burnin_start"] = str(split_config.burnin.start)
                hmm_model_data["burnin_end"] = str(split_config.burnin.effective_end)

        joblib.dump(hmm_model_data, hmm_path)
        tprint_success(f"✅ Saved HMM model: {hmm_path}")

        # 2. Save LiveHMM wrapper with scaler parameters
        live_hmm = LiveHMM(
            hmm_model,
            safe_state_id=safe_state_id,
            mahal_scaler_params=mahal_scaler_params
        )
        live_hmm_path = f"versioned_artifacts/risk_regime_models/risk_live_hmm_{symbol}_{timeframe}_{timestamp}.pkl"
        joblib.dump(live_hmm, live_hmm_path)
        tprint_success(f"✅ Saved LiveHMM wrapper with [0, 1] normalization: {live_hmm_path}")

        # 3. Save artifact with Mahalanobis distances
        artifact_path = f"versioned_artifacts/risk_regime_models/risk_regime_artifact_{symbol}_{timeframe}_{timestamp}.parquet"

        artifact_df = risk_features.copy()
        artifact_df['regime'] = regime_labels

        # Save as parquet if pandas supports it
        try:
            artifact_df.to_parquet(artifact_path)
            tprint_success(f"✅ Saved artifact with Mahalanobis distances: {artifact_path}")
        except:
            # Fallback to CSV
            artifact_path = artifact_path.replace('.parquet', '.csv')
            artifact_df.to_csv(artifact_path)
            tprint_success(f"✅ Saved artifact as CSV: {artifact_path}")

        return {
            "hmm_model_path": hmm_path,
            "live_hmm_path": live_hmm_path,
            "artifact_path": artifact_path,
        }
