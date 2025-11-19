"""Meta-Labeling HPO Experiment Step.

This offline step performs hierarchical hyperparameter optimization over
labeling-specific parameters (triple-barrier / TPSL, horizon, and target
clipping) using the HierarchicalParameterOptimizer.

It is intentionally decoupled from standard training runs. Invoke it
explicitly via the launcher with an appropriate config. A simple config
flag `enable_labeling_hpo` can be used to disable the optimization and
exit early if desired.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success

# Reuse core labeling utilities from the production meta-labeling step
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    kalman_smooth_labels,
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    generate_primary_signals,
    DEFAULT_TRANSACTION_COST,
    create_meta_features,
    compute_learnability_score,
    compute_label_entropy_score,
    generate_diagnostics_report,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
)

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)
from src.utils.ml_common.optimization.pareto import (
    Solution,
    ParetoFront,
    compute_pareto_front,
    select_knee_point,
)


logger = system_logger.getChild("MetaLabelingHPOExperiment")

# Optional diagnostics for the recommended configuration can be useful but are
# not required for the HPO step to function. They have occasionally triggered
# pandas categorical setitem issues in some environments. To keep the HPO step
# robust, we disable these diagnostics by default and gate them behind this
# constant, which can be flipped to True if deeper investigation is needed.
GENERATE_RECOMMENDED_DIAGNOSTICS: bool = False


class MetaLabelingHPOExperimentStep(BaseStep):
    """Offline HPO step to optimize labeling parameters.

    This step does *not* run as part of standard training. It must be
    invoked explicitly (e.g. via the launcher). It reuses the existing
    labeling utilities but searches over a small parameter space:

    - Event definition / TPSL (profit threshold, stop ratio, horizon 2–12,
      min spacing)
    - Target transformation (symmetric probability clipping and
      symmetric quantile clipping of target magnitudes)

    The objective focuses on label quality and economic separation
    between positive/negative labels using realized returns.
    """

    def __init__(self, step_name: str = "meta_labeling_hpo_experiment") -> None:
        super().__init__(step_name, use_versioned_artifacts=False)
        self.logger = logger

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hierarchical HPO over labeling parameters.

        Config keys (non-exhaustive):
        - symbol, exchange, timeframe: market context
        - enable_labeling_hpo: if False, step exits early
        - execution_mode: 'full'/'light'/'blank' for data loading scope
        """
        if not config.get("enable_labeling_hpo", True):
            tprint("ℹ️ Labeling HPO disabled via config.enable_labeling_hpo", "INFO")
            return {"success": True, "metrics": {}, "artifacts": {}, "skipped": True}

        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")

        tprint_info(
            f"🚀 Starting Meta-Labeling HPO experiment for {symbol}/{exchange} [{timeframe}]"
        )

        # ------------------------------------------------------------------
        # 1) Load market data once and generate primary signals
        # ------------------------------------------------------------------
        pipeline_state: Dict[str, Any] = {}
        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=True,
        )

        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            msg = "❌ No market data available for labeling HPO"
            tprint(msg, "ERROR")
            return {"success": False, "error": msg, "metrics": {}, "artifacts": {}}

        tprint_info(f"📊 Loaded market data from: {source} | rows={len(market_data)}")

        # Generate primary consensus signals using the production helper
        tprint_info("⚙️ Generating primary signals for HPO labeling runs…")
        primary_signals = generate_primary_signals(market_data.copy())

        # Precompute volatility for Kalman smoothing
        log_ret = np.log(market_data["close"]).diff()
        volatility_1d = log_ret.rolling(96).std()

        # Precompute span in days once (used by density penalty in objective)
        try:
            days_span = max(
                1,
                (market_data.index.max() - market_data.index.min()).days,
            )
        except Exception:
            days_span = 1

        # Build simple arrays for the optimizer API (they are not used in
        # the objective itself but provide shapes/logging)
        X_dummy = market_data[["close"]].dropna().values.astype("float32")
        y_dummy = np.zeros(len(X_dummy), dtype="float32")

        # ------------------------------------------------------------------
        # 2) Define parameter groups for hierarchical HPO
        # ------------------------------------------------------------------
        param_groups = [
            # Event / TPSL definition group
            create_param_group(
                name="event_definition",
                params={
                    "profit_thr_base": {
                        "type": "float",
                        "low": 0.008,
                        "high": 0.03,
                    },
                    "stop_to_profit_ratio": {
                        "type": "float",
                        "low": 0.3,
                        "high": 0.67,  # CONSTRAINT: profit must be >= 1.5x stop
                    },
                    "horizon_bars": {
                        "type": "int",
                        "low": 8,  # Changed from 2 to 8
                        "high": 56,  # Expanded upper bound for longer horizons
                        "step": 2,  # Increments of 2 or 4
                    },
                    "min_event_spacing": {
                        "type": "int",
                        "low": 2,
                        "high": 8,
                    },
                },
                priority=1,
                description="Triple-barrier / TPSL event definition",
            ),
            # Target transformation & probability clipping group
            create_param_group(
                name="target_transform",
                params={
                    "iso_min_prob": {
                        "type": "float",
                        "low": 0.0,
                        "high": 0.1,
                    },
                    "target_clip_high_q": {
                        "type": "float",
                        "low": 0.90,
                        "high": 0.99,
                    },
                },
                priority=2,
                depends_on=["event_definition"],
                description="Symmetric clipping for meta probabilities and targets",
            ),
            create_param_group(
                name="kalman_smoothing",
                params={
                    "kalman_Q": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-3,
                        "log": True,
                    },
                    "kalman_R": {
                        "type": "float",
                        "low": 1e-3,
                        "high": 0.1,
                        "log": True,
                    },
                },
                priority=3,
                depends_on=["event_definition"],
                description="Kalman smoothing noise parameters",
            ),
            create_param_group(
                name="volatility_adaptation",
                params={
                    "vol_baseline_window": {
                        "type": "int",
                        "low": 48,
                        "high": 192,
                    },
                    "profit_mult_min": {
                        "type": "float",
                        "low": 0.5,
                        "high": 1.0,
                    },
                    "profit_mult_max": {
                        "type": "float",
                        "low": 1.0,
                        "high": 2.0,
                    },
                    "stop_mult_min": {
                        "type": "float",
                        "low": 0.5,
                        "high": 1.0,
                    },
                    "stop_mult_max": {
                        "type": "float",
                        "low": 1.0,
                        "high": 2.0,
                    },
                },
                priority=4,
                depends_on=["event_definition"],
                description="Volatility adaptation baseline and multipliers",
            ),
        ]

        # Storage for candidate label configurations
        candidate_pool: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # 3) Define objective function for labeling quality (with learnability)
        # ------------------------------------------------------------------

        def labeling_objective(
            params: Dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            model: Any | None = None,
            cv_folds: int = 1,
            scoring_metric: str = "custom_balanced_score",
            **kwargs: Any,
        ) -> Dict[str, float]:
            """Evaluate one labeling configuration with multi-objective scoring.

            This function:
            - Recomputes realized returns & binary labels with candidate TPSL parameters
            - Smooths labels via Kalman filter
            - Creates meta-features for learnability assessment
            - Computes learnability score (cross-validated AUC)
            - Computes profitability score (economic metrics)
            - Applies regularization checks (temporal stability, regime consistency)
            - Returns dict of objectives for Pareto frontier

            Returns:
                Dict with keys: 'learnability', 'profitability', 'combined'
            """

            try:
                # Enforce profit >= 1.5x stop constraint
                profit_thr_base = float(params["profit_thr_base"])
                stop_ratio = float(params["stop_to_profit_ratio"])

                # CONSTRAINT: Ensure profit is at least 1.5x stop
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)
                if profit_thr_base < 1.5 * stop_thr_base:
                    tprint_warning(f"⚠️ Config rejected: profit {profit_thr_base:.4f} < 1.5x stop {stop_thr_base:.4f}")
                    return {'learnability': 0.0, 'profitability': -1e9, 'combined': -1e9}

                # Extract parameters
                horizon = int(params["horizon_bars"])
                min_spacing = int(params["min_event_spacing"])

                kalman_Q = float(params.get("kalman_Q", 1e-4))
                kalman_R = float(params.get("kalman_R", 0.01))
                vol_baseline_window = int(params.get("vol_baseline_window", 96))
                profit_mult_min = float(params.get("profit_mult_min", 0.5))
                profit_mult_max = float(params.get("profit_mult_max", 2.0))
                stop_mult_min = float(params.get("stop_mult_min", 0.5))
                stop_mult_max = float(params.get("stop_mult_max", 2.0))

                # Enforce horizon is in 8-56 range with steps of 2
                horizon = max(8, min(56, horizon))
                if horizon % 2 != 0:
                    horizon = (horizon // 2) * 2  # Round down to even
                min_spacing = max(1, min(16, min_spacing))
                vol_baseline_window = max(8, min(512, vol_baseline_window))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                # Use safe defaults when target_transform params are not part of the current group
                iso_min_prob = float(params.get("iso_min_prob", 0.0))
                iso_min_prob = max(0.0, min(0.1, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.9, min(1.0, iso_max_prob))

                q_high = float(params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.99, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                # --- Recompute realized returns ---
                vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
                vol_factor = volatility_1d / (vol_baseline + 1e-8)
                adaptive_profit = profit_thr_base * vol_factor
                adaptive_stop = stop_thr_base * vol_factor
                adaptive_profit = adaptive_profit.clip(
                    lower=profit_thr_base * profit_mult_min,
                    upper=profit_thr_base * profit_mult_max,
                )
                adaptive_stop = adaptive_stop.clip(
                    lower=stop_thr_base * stop_mult_min,
                    upper=stop_thr_base * stop_mult_max,
                )
                (
                    realized_returns,
                    binary_labels,
                    exit_reasons,
                    event_durations,
                    mfe_series,
                    mae_series,
                ) = compute_realized_returns(
                    market_data,
                    primary_signals,
                    profit_threshold=adaptive_profit,
                    stop_threshold=adaptive_stop,
                    horizon=horizon,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=min_spacing,
                )

                # Replace legacy R-multiple based labels with quantile-based labels
                # derived from volatility-scaled realized returns, to improve label
                # balance and economic relevance in HPO scoring.
                vol_scaled_returns = compute_vol_scaled_returns_for_events(
                    realized_returns=realized_returns,
                    volatility=volatility_1d,
                )
                quantile_labels = create_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_returns,
                    low_q=0.3,
                    high_q=0.7,
                )
                binary_labels = quantile_labels

                # Guard: if we got very few labeled events, return a poor score
                labeled_mask = ~binary_labels.isna()
                n_events = int(labeled_mask.sum())
                if n_events < 200:
                    tprint_warning(
                        f"⚠️ HPO config produced too few events (n={n_events}), penalizing",
                    )
                    return -1e9

                # --- Kalman smoothing for meta probability proxy ---
                smoothed_labels, _ = kalman_smooth_labels(
                    binary_labels,
                    Q=kalman_Q,
                    R=kalman_R,
                    volatility=volatility_1d,
                )

                # Probabilities limited to [0, 1]
                prob_series = smoothed_labels.clip(0.0, 1.0)

                # Symmetric clipping before isotonic regression
                prob_clipped = prob_series.clip(iso_min_prob, iso_max_prob)

                # Fit probability→expected-return mapping on labeled events
                iso_reg = fit_probability_to_return_mapping(
                    probabilities=prob_clipped.values,
                    realized_returns=realized_returns.values,
                    method="isotonic",
                )

                # Translate to long/short targets using existing helper
                target_long, target_short = translate_to_targets_with_isotonic(
                    realized_returns=realized_returns,
                    probabilities=prob_clipped.values,
                    signals=primary_signals,
                    iso_regressor=iso_reg,
                )

                # Construct a unified target magnitude (as in diagnostics)
                target_mag = pd.Series(0.0, index=market_data.index)
                long_mask = target_long > 0
                short_mask = target_short > 0
                target_mag[long_mask] = target_long[long_mask]
                target_mag[short_mask] = target_short[short_mask]

                # Quantile clipping of non-zero targets (symmetric tails)
                target_nz = target_mag[target_mag > 0]
                if len(target_nz) >= 100:
                    low_val = target_nz.quantile(q_low)
                    high_val = target_nz.quantile(q_high)
                    if low_val < high_val:
                        target_nz = target_nz.clip(low_val, high_val)

                # ===== LEARNABILITY ASSESSMENT =====
                # Create meta-features for this labeling configuration
                from src.training.steps.labeling.feature_generation_meta_labeling_step import (
                    create_meta_features,
                    compute_learnability_score,
                    compute_label_entropy_score,
                )

                meta_features = create_meta_features(
                    market_data,
                    primary_signals,
                    volume_available=True,
                    include_raw_signals=False,
                    use_kalman=True
                )

                # Compute learnability score (cross-validated AUC)
                learnability_score, mean_auc = compute_learnability_score(
                    X=meta_features,
                    y=binary_labels,
                    cv_splits=3,
                    time_aware_cv=True
                )

                # PENALTY: if mean_auc < 0.7, heavily penalize
                if mean_auc < 0.7:
                    auc_penalty = (0.7 - mean_auc) * 5.0  # Large penalty for poor learnability
                    learnability_score -= auc_penalty

                # Compute label entropy/balance score
                balance_score = compute_label_entropy_score(binary_labels)

                # ===== ECONOMIC PROFITABILITY =====
                # Compute economic separation metrics on labeled events
                returns_labeled = realized_returns[labeled_mask]
                labels_labeled = binary_labels[labeled_mask]

                # Mean return for label=1 vs label=0
                r_pos = returns_labeled[labels_labeled == 1]
                r_neg = returns_labeled[labels_labeled == 0]

                if len(r_pos) == 0 or len(r_neg) == 0:
                    return {'learnability': learnability_score, 'profitability': -1e9, 'combined': -1e9}

                mean_pos = float(r_pos.mean())
                mean_neg = float(r_neg.mean())
                sep = mean_pos - mean_neg

                # Simple Sharpe for label=1 trades
                std_pos = float(r_pos.std())
                sharpe_pos = mean_pos / (std_pos + 1e-8)

                tx = float(DEFAULT_TRANSACTION_COST)
                if mean_pos <= tx:
                    return {'learnability': learnability_score, 'profitability': -1e9, 'combined': -1e9}

                # Penalize configurations dominated by economically trivial events
                returns_labeled_nonnull = returns_labeled.dropna()
                if len(returns_labeled_nonnull) > 0:
                    small_band = tx
                    frac_small = float((returns_labeled_nonnull.abs() < small_band).mean())
                else:
                    frac_small = 1.0

                # ===== PRE- VS POST-FILTER DIAGNOSTICS (RETENTION & SNR) =====
                try:
                    pre_mask = ~realized_returns.isna()
                    n_pre_total = int(pre_mask.sum())

                    if n_pre_total > 0:
                        pre_returns = realized_returns[pre_mask]
                        raw_label_pre = (pre_returns > tx).astype(int)

                        n_pre_pos = int((raw_label_pre == 1).sum())
                        n_pre_neg = int((raw_label_pre == 0).sum())

                        n_post_total = int(n_events)
                        n_post_pos = int((labels_labeled == 1).sum())
                        n_post_neg = int((labels_labeled == 0).sum())

                        retention_total = n_post_total / max(n_pre_total, 1)
                        retention_pos = n_post_pos / max(n_pre_pos, 1) if n_pre_pos > 0 else 0.0
                        retention_neg = n_post_neg / max(n_pre_neg, 1) if n_pre_neg > 0 else 0.0

                        pre_pos_ret = pre_returns[raw_label_pre == 1]
                        pre_neg_ret = pre_returns[raw_label_pre == 0]

                        def _safe_stats(x: pd.Series) -> tuple[float, float]:
                            return (
                                float(x.mean()) if len(x) > 0 else 0.0,
                                float(x.std() if len(x) > 1 else 0.0),
                            )

                        pre_pos_mean, pre_pos_std = _safe_stats(pre_pos_ret)
                        pre_neg_mean, pre_neg_std = _safe_stats(pre_neg_ret)
                        post_pos_mean, post_pos_std = _safe_stats(r_pos)
                        post_neg_mean, post_neg_std = _safe_stats(r_neg)

                        def _cohens_d(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
                            if n1 <= 1 or n2 <= 1:
                                return float('nan')
                            pooled = ((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1)
                            if pooled <= 0:
                                return float('nan')
                            return (m1 - m2) / np.sqrt(pooled)

                        d_pre = _cohens_d(
                            pre_pos_mean,
                            pre_pos_std,
                            max(len(pre_pos_ret), 1),
                            pre_neg_mean,
                            pre_neg_std,
                            max(len(pre_neg_ret), 1),
                        )
                        d_post = _cohens_d(
                            post_pos_mean,
                            post_pos_std,
                            max(len(r_pos), 1),
                            post_neg_mean,
                            post_neg_std,
                            max(len(r_neg), 1),
                        )

                        snr_pre = pre_pos_mean / (pre_pos_std + 1e-8) if pre_pos_std > 0 else 0.0
                        snr_post = post_pos_mean / (post_pos_std + 1e-8) if post_pos_std > 0 else 0.0
                    else:
                        n_pre_total = 0
                        retention_total = 0.0
                        retention_pos = 0.0
                        retention_neg = 0.0
                        d_pre = float('nan')
                        d_post = float('nan')
                        snr_pre = 0.0
                        snr_post = 0.0
                except Exception:
                    n_pre_total = 0
                    retention_total = 0.0
                    retention_pos = 0.0
                    retention_neg = 0.0
                    d_pre = float('nan')
                    d_post = float('nan')
                    snr_pre = 0.0
                    snr_post = 0.0

                # Event density penalty: we want neither too few nor too many
                trades_per_day = n_events / days_span
                # Target range roughly 1–5 trades/day
                penalty_density = 0.0
                if trades_per_day < 1.0:
                    penalty_density += (1.0 - trades_per_day) * 5.0
                if trades_per_day > 5.0:
                    penalty_density += (trades_per_day - 5.0) * 0.5

                penalty_noise = frac_small * 10.0

                # Top-bucket economics (top 10% by smoothed probability) to capture
                # how good the very best signals are.
                top_bucket_mean = 0.0
                top_bucket_sharpe = 0.0
                try:
                    prob_array = prob_clipped.values.astype(float)
                    if np.isfinite(prob_array).any():
                        q90 = np.nanquantile(prob_array, 0.9)
                        top_mask = (prob_array >= q90) & labeled_mask.to_numpy()
                        top_returns = realized_returns[top_mask]
                        top_returns = top_returns.dropna()
                        if len(top_returns) >= 20:
                            top_bucket_mean = float(top_returns.mean())
                            top_std = float(top_returns.std())
                            top_bucket_sharpe = top_bucket_mean / (top_std + 1e-8) if top_std > 0 else 0.0
                except Exception:
                    top_bucket_mean = 0.0
                    top_bucket_sharpe = 0.0

                # Profitability score: emphasize separation and Sharpe, subtract penalties,
                # and reward strong top-bucket performance.
                profitability_score = (
                    sep * 100.0
                    + sharpe_pos * 10.0
                    + top_bucket_sharpe * 15.0
                    + top_bucket_mean * 1000.0
                    - penalty_density
                    - penalty_noise
                )

                # Extra penalty when label balance is extreme (balance_score == 0)
                if balance_score <= 0.0:
                    learnability_score -= 0.5

                # Simple power heuristic: required samples for ~80% power based on post-filter effect size
                try:
                    if np.isfinite(d_post) and d_post != 0.0:
                        n_required_80 = 16.0 / (d_post ** 2)
                    else:
                        n_required_80 = float('inf')
                except Exception:
                    n_required_80 = float('inf')

                # ===== REGULARIZATION CHECKS =====
                # Temporal stability check (rolling window AUC variance)
                try:
                    window_size = max(100, n_events // 5)
                    n_windows = min(5, n_events // window_size)
                    auc_variance = 0.0

                    if n_windows >= 2:
                        window_aucs = []
                        for w in range(n_windows):
                            start_idx = w * window_size
                            end_idx = min((w + 1) * window_size, n_events)
                            window_labels = labels_labeled.iloc[start_idx:end_idx]

                            if len(window_labels.unique()) >= 2 and len(window_labels) >= 20:
                                # Compute simple correlation as AUC proxy
                                window_returns = returns_labeled.iloc[start_idx:end_idx]
                                try:
                                    window_auc = abs(window_labels.corr(window_returns))
                                    window_aucs.append(window_auc if not np.isnan(window_auc) else 0.5)
                                except:
                                    window_aucs.append(0.5)

                        if len(window_aucs) >= 2:
                            auc_variance = float(np.var(window_aucs))
                            # Penalize high variance (instability across time)
                            temporal_stability_penalty = auc_variance * 10.0
                            profitability_score -= temporal_stability_penalty
                except Exception:
                    pass  # Skip if temporal check fails

                # ===== COMBINED OBJECTIVE (70% learnability, 30% profitability) =====
                combined_score = (0.7 * learnability_score) + (0.3 * profitability_score / 100.0)

                # Store candidate configuration for later persistence
                candidate_config = {
                    'params': params.copy(),
                    'learnability': float(learnability_score),
                    'mean_auc': float(mean_auc),
                    'profitability': float(profitability_score),
                    'combined': float(combined_score),
                    'mean_pos': float(mean_pos),
                    'mean_neg': float(mean_neg),
                    'sharpe_pos': float(sharpe_pos),
                    'n_events': int(n_events),
                    'balance_score': float(balance_score),
                    'trades_per_day': float(trades_per_day),
                    'n_pre_events': int(n_pre_total),
                    'retention_total': float(retention_total),
                    'retention_pos': float(retention_pos),
                    'retention_neg': float(retention_neg),
                    'snr_pre': float(snr_pre),
                    'snr_post': float(snr_post),
                    'effect_size_pre': float(d_pre) if np.isfinite(d_pre) else 0.0,
                    'effect_size_post': float(d_post) if np.isfinite(d_post) else 0.0,
                    'n_required_80pct_power': float(n_required_80),
                }
                candidate_pool.append(candidate_config)

                return {
                    'learnability': float(learnability_score),
                    'profitability': float(profitability_score),
                    'combined': float(combined_score)
                }

            except Exception as exc:  # Defensive: never crash HPO on one config
                tprint_warning(f"⚠️ Labeling objective failed: {exc}")
                import traceback
                traceback.print_exc()
                return {'learnability': 0.0, 'profitability': -1e9, 'combined': -1e9}

        # ------------------------------------------------------------------
        # 4) Multi-objective wrapper for single-objective optimizers
        # ------------------------------------------------------------------
        def scalar_objective_wrapper(params: Dict[str, Any]) -> float:
            """Wrapper to extract combined score for single-objective optimizers.

            BayesianTPEOptimizer always calls the objective as objective(params), so
            we close over X_dummy / y_dummy instead of expecting them as arguments.
            """
            result = labeling_objective(params, X_dummy, y_dummy)
            if isinstance(result, dict):
                return float(result.get('combined', 0.0))
            return float(result)

        # ------------------------------------------------------------------
        # 5) Instantiate Bayesian TPE optimizer (replaces hierarchical grid search)
        # ------------------------------------------------------------------
        tprint_info("🚀 Using Bayesian TPE optimization for efficient search")

        # Convert param_groups (ParameterGroup instances) to a flat Optuna-style search space
        search_space: Dict[str, Dict[str, Any]] = {}
        for group in param_groups:
            for param_name, param_spec in group.params.items():
                search_space[param_name] = param_spec

        # Configure Bayesian optimizer (aligned with OptimizationConfig signature)
        bayesian_config = OptimizationConfig(
            n_trials=100,  # Total trials
            direction='maximize',
            # Staged optimization settings
            enable_staged_optimization=True,
            coarse_grid_trials=20,
            fine_grid_trials=20,
            tpe_trials=60,
            # Disable hardware/VectorBT-specific acceleration for compatibility in this step
            enable_hardware_optimization=False,
            enable_vectorbt_optimization=False,
            # Early stopping configuration
            early_stopping_patience=15,
            early_stopping_threshold=None,
            # Reproducibility
            seed=42,
        )

        optimizer = BayesianTPEOptimizer(config=bayesian_config)

        # ------------------------------------------------------------------
        # 6) Run Bayesian TPE optimization
        # ------------------------------------------------------------------
        tprint_info("🔍 Running Bayesian TPE optimization...")

        result = optimizer.optimize(
            objective=scalar_objective_wrapper,
            search_space=search_space,
        )

        best_params = result.get('best_params', {})
        best_score = result.get('best_value', 0.0)

        tprint_success(f"✅ Labeling HPO completed. Best combined score={best_score:.6f}")
        tprint_info(f"Best parameters: {best_params}")

        # ------------------------------------------------------------------
        # 7) Compute Pareto Frontier from all candidate configurations
        # ------------------------------------------------------------------
        tprint_info("📊 Computing Pareto frontier...")

        # Convert candidate pool to Solution objects
        pareto_solutions = []
        for candidate in candidate_pool:
            solution = Solution(
                metrics={
                    'learnability': candidate['learnability'],
                    'profitability': candidate['profitability'],
                    'combined': candidate['combined'],
                    'mean_auc': candidate['mean_auc'],
                    'sharpe_pos': candidate['sharpe_pos'],
                    'n_events': candidate['n_events'],
                },
                params=candidate['params']
            )
            pareto_solutions.append(solution)

        # Compute Pareto front for learnability vs profitability
        objectives = {
            'learnability': 'max',
            'profitability': 'max',
        }

        pareto_front = compute_pareto_front(
            solutions=pareto_solutions,
            objectives=objectives,
            use_gpu=False,
            use_vectorbt=False,
        )

        tprint_success(f"✅ Pareto frontier: {len(pareto_front)}/{len(pareto_solutions)} non-dominated solutions")

        # Select knee point as recommended solution
        knee_solution = select_knee_point(
            pareto_solutions=pareto_front,
            objectives=objectives,
            weights={'learnability': 0.7, 'profitability': 0.3}
        )

        if knee_solution:
            tprint_info(f"📍 Knee point (recommended): learnability={knee_solution.metrics['learnability']:.4f}, "
                       f"profitability={knee_solution.metrics['profitability']:.4f}")
            # Update best_params to knee point if it's better balanced
            knee_params = knee_solution.params
        else:
            knee_params = best_params

        # Compact run summary for quick log scanning
        try:
            round_results = getattr(optimizer, "round_results", [])
            n_rounds = len(round_results) if isinstance(round_results, list) else None
            total_trials = sum(r.get("trials", 0) for r in round_results) if isinstance(round_results, list) else None
        except Exception:
            n_rounds = None
            total_trials = None

        tprint_info(
            "HPO summary → "
            f"symbol={symbol}, timeframe={timeframe}, "
            f"best_score={best_score:.6f}, "
            f"rounds={n_rounds}, trials={total_trials}, "
            f"params={best_params}",
        )

        # Persist best parameters and candidate pool to outcomes/
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # ===== OPTIONAL: Generate diagnostics for recommended configuration =====
        diagnostics_path: str | None = None
        try:
            # Prefer knee point if available, otherwise fall back to best_params
            diag_params: Dict[str, Any] = knee_params if knee_solution else best_params
            # Only run diagnostics when explicitly enabled via the module-level
            # constant to avoid categorical setitem issues in some environments.
            if diag_params and GENERATE_RECOMMENDED_DIAGNOSTICS:
                tprint_info("📊 Generating meta-labeling diagnostics for recommended configuration...")

                # Reconstruct labeling parameters (consistent with labeling_objective)
                profit_thr_base = float(diag_params["profit_thr_base"])
                stop_ratio = float(diag_params["stop_to_profit_ratio"])
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)

                horizon = int(diag_params["horizon_bars"])
                min_spacing = int(diag_params["min_event_spacing"])

                kalman_Q = float(diag_params.get("kalman_Q", 1e-4))
                kalman_R = float(diag_params.get("kalman_R", 0.01))
                vol_baseline_window = int(diag_params.get("vol_baseline_window", 96))
                profit_mult_min = float(diag_params.get("profit_mult_min", 0.5))
                profit_mult_max = float(diag_params.get("profit_mult_max", 2.0))
                stop_mult_min = float(diag_params.get("stop_mult_min", 0.5))
                stop_mult_max = float(diag_params.get("stop_mult_max", 2.0))

                # Apply same constraints as HPO objective
                horizon = max(8, min(28, horizon))
                if horizon % 2 != 0:
                    horizon = (horizon // 2) * 2
                min_spacing = max(1, min(16, min_spacing))
                vol_baseline_window = max(8, min(512, vol_baseline_window))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                iso_min_prob = float(diag_params.get("iso_min_prob", 0.0))
                iso_min_prob = max(0.0, min(0.1, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.9, min(1.0, iso_max_prob))

                q_high = float(diag_params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.99, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                # Recompute adaptive profit/stop thresholds
                vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
                vol_factor = volatility_1d / (vol_baseline + 1e-8)
                adaptive_profit = profit_thr_base * vol_factor
                adaptive_stop = stop_thr_base * vol_factor
                adaptive_profit = adaptive_profit.clip(
                    lower=profit_thr_base * profit_mult_min,
                    upper=profit_thr_base * profit_mult_max,
                )
                adaptive_stop = adaptive_stop.clip(
                    lower=stop_thr_base * stop_mult_min,
                    upper=stop_thr_base * stop_mult_max,
                )

                (
                    realized_returns,
                    binary_labels,
                    exit_reasons,
                    event_durations,
                    mfe_series_diag,
                    mae_series_diag,
                ) = compute_realized_returns(
                    market_data,
                    primary_signals,
                    profit_threshold=adaptive_profit,
                    stop_threshold=adaptive_stop,
                    horizon=horizon,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=min_spacing,
                )

                # Guard: if too few events, skip diagnostics
                labeled_mask = ~binary_labels.isna()
                if int(labeled_mask.sum()) < 100:
                    tprint_warning(
                        "⚠️ Recommended config produced too few events for diagnostics; skipping report generation",
                        "WARNING",
                    )
                else:
                    # Kalman smoothing
                    smoothed_labels, _ = kalman_smooth_labels(
                        binary_labels,
                        Q=kalman_Q,
                        R=kalman_R,
                        volatility=volatility_1d,
                    )

                    prob_series = smoothed_labels.clip(0.0, 1.0)
                    prob_clipped = prob_series.clip(iso_min_prob, iso_max_prob)

                    # Fit probability→expected-return mapping
                    iso_reg = fit_probability_to_return_mapping(
                        probabilities=prob_clipped.values,
                        realized_returns=realized_returns.values,
                        method="isotonic",
                    )

                    # Translate to long/short targets
                    target_long, target_short = translate_to_targets_with_isotonic(
                        realized_returns=realized_returns,
                        probabilities=prob_clipped.values,
                        signals=primary_signals,
                        iso_regressor=iso_reg,
                    )

                    # Build labeled_data in the same spirit as production step
                    labeled_data = market_data.copy()

                    # Drop any existing derived columns that might carry categorical dtypes
                    # so that we can safely assign fresh non-categorical Series
                    derived_cols = [
                        "log_ret",
                        "volatility_1d",
                        "realized_return",
                        "binary_label",
                        "smoothed_label",
                        "meta_probability",
                        "exit_reason",
                        "event_duration_bars",
                        "target_long",
                        "target_short",
                        "primary_signal",
                    ]
                    labeled_data = labeled_data.drop(columns=[c for c in derived_cols if c in labeled_data.columns], errors="ignore")

                    log_ret = np.log(market_data["close"]).diff()
                    labeled_data["log_ret"] = log_ret
                    labeled_data["volatility_1d"] = volatility_1d
                    labeled_data["realized_return"] = realized_returns
                    labeled_data["binary_label"] = binary_labels
                    labeled_data["smoothed_label"] = smoothed_labels
                    labeled_data["meta_probability"] = prob_clipped.values
                    labeled_data["exit_reason"] = exit_reasons
                    labeled_data["event_duration_bars"] = event_durations
                    labeled_data["target_long"] = target_long
                    labeled_data["target_short"] = target_short
                    labeled_data["primary_signal"] = primary_signals["consensus"]

                    # Meta-features for diagnostics (same helper as production)
                    meta_features = create_meta_features(
                        market_data,
                        primary_signals,
                        volume_available=True,
                        include_raw_signals=False,
                        use_kalman=True,
                    )

                    # Simple RF meta-model for feature importances
                    X_diag = meta_features[labeled_mask].fillna(0)
                    y_diag = binary_labels[labeled_mask]
                    final_model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=42,
                    )
                    if len(y_diag.unique()) >= 2 and len(y_diag) >= 50:
                        final_model.fit(X_diag, y_diag)
                    else:
                        # Fallback: still fit to avoid attribute errors in diagnostics
                        final_model.fit(X_diag, y_diag)

                    # Slightly enriched config for diagnostics
                    diag_config = dict(config)
                    diag_config["horizon"] = horizon
                    diag_config["profit_thr_base"] = profit_thr_base
                    diag_config["stop_thr_base"] = stop_thr_base

                    # Sanitize any categorical columns to avoid setitem/category issues
                    labeled_data_for_diag = labeled_data.copy()
                    cat_cols = labeled_data_for_diag.select_dtypes(include=["category"]).columns
                    if len(cat_cols) > 0:
                        for col in cat_cols:
                            labeled_data_for_diag[col] = labeled_data_for_diag[col].astype(object)

                    # Also ensure core Series are not categorical
                    binary_labels_diag = pd.Series(
                        binary_labels.astype(float).values,
                        index=binary_labels.index,
                    )
                    exit_reasons_diag = None
                    event_durations_diag = None
                    if exit_reasons is not None:
                        exit_reasons_diag = pd.Series(
                            exit_reasons.astype(object).values,
                            index=exit_reasons.index,
                        )
                    if event_durations is not None:
                        event_durations_diag = pd.Series(
                            event_durations.astype(float).values,
                            index=event_durations.index,
                        )

                    diagnostics_path_obj = generate_diagnostics_report(
                        labeled_data=labeled_data_for_diag,
                        meta_features=meta_features,
                        binary_labels=binary_labels_diag,
                        realized_returns=realized_returns,
                        smoothed_labels=smoothed_labels,
                        probabilities=prob_clipped.values,
                        final_model=final_model,
                        config=diag_config,
                        output_dir=outcomes_dir,
                        exit_reasons=exit_reasons_diag,
                        event_durations=event_durations_diag,
                        mfe_series=mfe_series_diag,
                        mae_series=mae_series_diag,
                        target_long=target_long,
                        target_short=target_short,
                    )
                    diagnostics_path = str(diagnostics_path_obj)

                    tprint_success(
                        f"📊 Saved diagnostics for recommended labeling configuration to {diagnostics_path}",
                    )
        except Exception as diag_exc:
            tprint_warning(f"⚠️ Failed to generate diagnostics for recommended configuration: {diag_exc}")

        # ===== SAVE BEST PARAMS JSON =====
        json_name = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{timestamp}.json"
        json_path = outcomes_dir / json_name

        try:
            with open(json_path, "w") as f:
                json.dump({
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "best_score": best_score,
                    "best_params": best_params,
                    "knee_params": knee_params,
                    "pareto_front_size": len(pareto_front),
                }, f, indent=2)
            tprint_success(f"💾 Saved best labeling HPO params to {json_path}")
        except Exception as save_exc:
            tprint_warning(f"⚠️ Failed to save best_params JSON: {save_exc}")
            json_path = None

        # ===== SAVE CANDIDATE POOL CSV =====
        csv_name = f"meta_labeling_hpo_candidate_pool_{symbol}_{timeframe}_{timestamp}.csv"
        csv_path = outcomes_dir / csv_name

        try:
            if not candidate_pool:
                tprint_warning("⚠️ Candidate pool is empty; skipping candidate CSV export")
                csv_path = None
            else:
                candidate_df = pd.DataFrame(candidate_pool)

                if 'params' in candidate_df.columns:
                    params_df = pd.json_normalize(candidate_df['params'])
                    candidate_df = candidate_df.drop(columns=['params'])
                    candidate_df = pd.concat([candidate_df, params_df], axis=1)

                if 'combined' in candidate_df.columns:
                    candidate_df = candidate_df.sort_values('combined', ascending=False)

                candidate_df.to_csv(csv_path, index=False, float_format='%.6f')
                tprint_success(f"💾 Saved {len(candidate_pool)} candidate configs to {csv_path}")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save candidate pool CSV: {csv_exc}")
            csv_path = None

        # ===== SAVE PARETO FRONTIER CSV =====
        pareto_csv_name = f"meta_labeling_hpo_pareto_front_{symbol}_{timeframe}_{timestamp}.csv"
        pareto_csv_path = outcomes_dir / pareto_csv_name

        try:
            if not pareto_front:
                tprint_warning("⚠️ Pareto frontier is empty; skipping Pareto CSV export")
                pareto_csv_path = None
            else:
                pareto_data: list[dict[str, Any]] = []
                for sol in pareto_front:
                    row = dict(sol.metrics)
                    if sol.params:
                        row.update(sol.params)
                    pareto_data.append(row)

                pareto_df = pd.DataFrame(pareto_data)
                if 'combined' in pareto_df.columns:
                    pareto_df = pareto_df.sort_values('combined', ascending=False)

                pareto_df.to_csv(pareto_csv_path, index=False, float_format='%.6f')
                tprint_success(f"💾 Saved {len(pareto_front)} Pareto solutions to {pareto_csv_path}")
        except Exception as pareto_exc:
            tprint_warning(f"⚠️ Failed to save Pareto frontier CSV: {pareto_exc}")
            pareto_csv_path = None

        # ===== SAVE COMPREHENSIVE MARKDOWN REPORT =====
        md_name = f"meta_labeling_hpo_report_{symbol}_{timeframe}_{timestamp}.md"
        md_path = outcomes_dir / md_name

        try:
            with open(md_path, "w") as f:
                f.write(f"# Meta-Labeling HPO Report\n\n")
                f.write(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
                f.write(f"**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe}\n\n")
                f.write(f"---\n\n")

                # Summary
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Configurations Evaluated:** {len(candidate_pool)}\n")
                f.write(f"- **Pareto Frontier Size:** {len(pareto_front)}\n")
                f.write(f"- **Best Combined Score:** {best_score:.6f}\n")
                f.write(f"- **Optimization Method:** Bayesian TPE with Pareto Frontier\n\n")

                # Best Parameters
                f.write(f"## Best Parameters (Highest Combined Score)\n\n")
                f.write(f"```json\n")
                f.write(json.dumps(best_params, indent=2))
                f.write(f"\n```\n\n")

                # Knee Point Parameters
                if knee_solution:
                    f.write(f"## Recommended Parameters (Pareto Knee Point)\n\n")
                    f.write(f"Balanced trade-off between learnability and profitability:\n\n")
                    f.write(f"- **Learnability:** {knee_solution.metrics['learnability']:.4f}\n")
                    f.write(f"- **Profitability:** {knee_solution.metrics['profitability']:.4f}\n")
                    f.write(f"- **Mean AUC:** {knee_solution.metrics.get('mean_auc', 0):.4f}\n")
                    f.write(f"- **Sharpe (Winners):** {knee_solution.metrics.get('sharpe_pos', 0):.4f}\n\n")
                    f.write(f"```json\n")
                    f.write(json.dumps(knee_params, indent=2))
                    f.write(f"\n```\n\n")

                # Pareto Frontier Summary
                f.write(f"## Pareto Frontier Analysis\n\n")
                f.write(f"The Pareto frontier contains {len(pareto_front)} non-dominated solutions "
                       f"representing optimal trade-offs between learnability and profitability.\n\n")

                # Top 10 Pareto Solutions
                f.write(f"### Top 10 Pareto Solutions\n\n")
                f.write(f"| Rank | Learnability | Profitability | Combined | Mean AUC | Sharpe | N Events |\n")
                f.write(f"|------|-------------|--------------|----------|----------|--------|----------|\n")

                sorted_pareto = sorted(pareto_front, key=lambda s: s.metrics['combined'], reverse=True)
                for i, sol in enumerate(sorted_pareto[:10], 1):
                    m = sol.metrics
                    f.write(f"| {i} | {m['learnability']:.4f} | {m['profitability']:.4f} | "
                           f"{m['combined']:.4f} | {m.get('mean_auc', 0):.4f} | "
                           f"{m.get('sharpe_pos', 0):.4f} | {m.get('n_events', 0)} |\n")

                f.write(f"\n")

                # Regularization Checks Summary
                f.write(f"## Regularization Checks\n\n")
                f.write(f"All configurations were evaluated with:\n\n")
                f.write(f"1. **Temporal Stability:** Rolling window AUC variance penalty\n")
                f.write(f"2. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized\n")
                f.write(f"3. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold\n")
                f.write(f"4. **Label Balance:** Entropy-based balance scoring\n\n")

                # Artifacts
                f.write(f"## Artifacts\n\n")
                f.write(f"- **Best Params JSON:** `{json_path.name if json_path else 'N/A'}`\n")
                f.write(f"- **Candidate Pool CSV:** `{csv_path.name if csv_path else 'N/A'}`\n")
                f.write(f"- **Pareto Frontier CSV:** `{pareto_csv_path.name if pareto_csv_path else 'N/A'}`\n")
                f.write(f"- **This Report:** `{md_name}`\n\n")

            tprint_success(f"📄 Saved comprehensive report to {md_path}")
        except Exception as md_exc:
            tprint_warning(f"⚠️ Failed to save markdown report: {md_exc}")
            md_path = None

        # Persist per-round HPO metrics to CSV for analysis
        csv_path = None
        try:
            round_results = getattr(optimizer, "round_results", [])
            if isinstance(round_results, list) and round_results:
                rows: list[dict[str, Any]] = []
                for rr in round_results:
                    rows.append(
                        {
                            "round": rr.get("round"),
                            "best_score": rr.get("best_score"),
                            "improvement": rr.get("improvement"),
                            "time_seconds": rr.get("time"),
                            "trials": rr.get("trials"),
                        }
                    )

                df_rounds = pd.DataFrame(rows)
                csv_name = (
                    f"meta_labeling_hpo_round_metrics_{symbol}_{timeframe}_{timestamp}.csv"
                )
                csv_path = outcomes_dir / csv_name
                df_rounds.to_csv(csv_path, index=False)
                tprint_success(f"💾 Saved HPO round metrics to {csv_path}")
            else:
                tprint_warning("⚠️ No round_results available on optimizer; skipping CSV export")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save HPO round metrics CSV: {csv_exc}")

        metrics: Dict[str, Any] = {
            "best_score": best_score,
            "best_params": best_params,
            "best_params_json": str(json_path) if json_path is not None else None,
            "round_metrics_csv": str(csv_path) if csv_path is not None else None,
            "recommended_diagnostics_path": diagnostics_path,
        }

        artifacts: Dict[str, Any] = {}
        if json_path is not None:
            artifacts["best_params_json"] = str(json_path)
        if csv_path is not None:
            artifacts["round_metrics_csv"] = str(csv_path)
        if diagnostics_path is not None:
            artifacts["recommended_diagnostics_path"] = diagnostics_path

        return {
            "success": True,
            "metrics": metrics,
            "artifacts": artifacts,
        }


def register_meta_labeling_hpo_experiment_step() -> None:
    """Register the meta-labeling HPO experiment step in the registry."""
    from src.training.steps.base_step import step_registry

    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)
    tprint("✅ Meta-labeling HPO experiment step registered", "SUCCESS")


# Auto-register when module is imported
register_meta_labeling_hpo_experiment_step()
