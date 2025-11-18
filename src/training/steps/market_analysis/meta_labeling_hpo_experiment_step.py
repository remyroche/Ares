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

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success

# Reuse core labeling utilities from the production meta-labeling step
from src.training.steps.market_analysis.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    kalman_smooth_labels,
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    generate_primary_signals,
    DEFAULT_TRANSACTION_COST,
)

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
)


logger = system_logger.getChild("MetaLabelingHPOExperiment")


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
                        "high": 1.0,
                    },
                    "horizon_bars": {
                        "type": "int",
                        "low": 2,
                        "high": 24,
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

        # ------------------------------------------------------------------
        # 3) Define objective function for labeling quality
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
        ) -> float:
            """Evaluate one labeling configuration.

            This function:
            - Recomputes realized returns & binary labels with candidate
              TPSL parameters (profit_thr_base, stop_to_profit_ratio,
              horizon_bars, min_event_spacing)
            - Smooths labels via Kalman filter
            - Applies symmetric probability clipping and isotonic mapping
              to construct targets
            - Applies symmetric quantile clipping to target magnitudes
            - Computes an economic/label-quality score to maximize
            """

            try:
                profit_thr_base = float(params["profit_thr_base"])
                stop_ratio = float(params["stop_to_profit_ratio"])
                horizon = int(params["horizon_bars"])
                min_spacing = int(params["min_event_spacing"])

                kalman_Q = float(params.get("kalman_Q", 1e-4))
                kalman_R = float(params.get("kalman_R", 0.01))
                vol_baseline_window = int(params.get("vol_baseline_window", 96))
                profit_mult_min = float(params.get("profit_mult_min", 0.5))
                profit_mult_max = float(params.get("profit_mult_max", 2.0))
                stop_mult_min = float(params.get("stop_mult_min", 0.5))
                stop_mult_max = float(params.get("stop_mult_max", 2.0))

                horizon = max(2, min(24, horizon))
                min_spacing = max(1, min(16, min_spacing))
                vol_baseline_window = max(8, min(512, vol_baseline_window))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)

                # Use safe defaults when target_transform params are not part of the current group
                iso_min_prob = float(params.get("iso_min_prob", 0.0))
                iso_min_prob = max(0.0, min(0.1, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.9, min(1.0, iso_max_prob))

                q_high = float(params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.99, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                # --- Recompute realized returns and labels ---
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
                realized_returns, binary_labels, exit_reasons, event_durations = (
                    compute_realized_returns(
                        market_data,
                        primary_signals,
                        profit_threshold=adaptive_profit,
                        stop_threshold=adaptive_stop,
                        horizon=horizon,
                        transaction_cost=DEFAULT_TRANSACTION_COST,
                        min_event_spacing=min_spacing,
                    )
                )

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

                # Compute economic separation metrics on labeled events
                returns_labeled = realized_returns[labeled_mask]
                labels_labeled = binary_labels[labeled_mask]

                # Mean return for label=1 vs label=0
                r_pos = returns_labeled[labels_labeled == 1]
                r_neg = returns_labeled[labels_labeled == 0]

                if len(r_pos) == 0 or len(r_neg) == 0:
                    return -1e9

                mean_pos = float(r_pos.mean())
                mean_neg = float(r_neg.mean())
                sep = mean_pos - mean_neg

                # Simple Sharpe for label=1 trades
                std_pos = float(r_pos.std())
                sharpe_pos = mean_pos / (std_pos + 1e-8)

                tx = float(DEFAULT_TRANSACTION_COST)
                if mean_pos <= tx:
                    return -1e9

                # Penalize configurations dominated by economically trivial events
                returns_labeled_nonnull = returns_labeled.dropna()
                if len(returns_labeled_nonnull) > 0:
                    small_band = tx
                    frac_small = float((returns_labeled_nonnull.abs() < small_band).mean())
                else:
                    frac_small = 1.0

                # Event density penalty: we want neither too few nor too many
                trades_per_day = n_events / days_span
                # Target range roughly 1–5 trades/day
                penalty_density = 0.0
                if trades_per_day < 1.0:
                    penalty_density += (1.0 - trades_per_day) * 5.0
                if trades_per_day > 5.0:
                    penalty_density += (trades_per_day - 5.0) * 0.5

                penalty_noise = frac_small * 10.0

                # Final score: emphasize separation and Sharpe, subtract density and noise penalties
                score = sep * 100.0 + sharpe_pos * 10.0 - penalty_density - penalty_noise

                return float(score)

            except Exception as exc:  # Defensive: never crash HPO on one config
                tprint_warning(f"⚠️ Labeling objective failed: {exc}")
                return -1e9

        # ------------------------------------------------------------------
        # 4) Instantiate hierarchical optimizer
        # ------------------------------------------------------------------
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=labeling_objective,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
            ],
            cv_folds=1,  # we perform our own temporal reasoning inside objective
            scoring_metric="custom_balanced_score",
            direction="maximize",
            n_rounds=2,
            enable_final_refinement=False,
            final_refinement_trials=0,
            cache_dir=None,
            random_state=42,
            verbose=True,
            use_custom_balanced_score=False,
        )

        # ------------------------------------------------------------------
        # 5) Run optimization (X_dummy / y_dummy are placeholders)
        # ------------------------------------------------------------------
        result = optimizer.optimize(
            X_train=X_dummy,
            y_train=y_dummy,
            X_val=None,
            y_val=None,
            model=None,
            initial_params=None,
        )

        best_params = result.best_params
        tprint_success(f"✅ Labeling HPO completed. Best score={result.best_score:.6f}")
        tprint_info(f"Best parameters: {best_params}")

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
            f"best_score={result.best_score:.6f}, "
            f"rounds={n_rounds}, trials={total_trials}, "
            f"params={best_params}",
        )

        # Persist best parameters to outcomes/ for later reuse
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_name = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{timestamp}.json"
        json_path = outcomes_dir / json_name

        try:
            with open(json_path, "w") as f:
                json.dump({
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "best_score": result.best_score,
                    "best_params": best_params,
                }, f, indent=2)
            tprint_success(f"💾 Saved best labeling HPO params to {json_path}")
        except Exception as save_exc:
            tprint_warning(f"⚠️ Failed to save best_params JSON: {save_exc}")
            json_path = None

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
            "best_score": result.best_score,
            "best_params": best_params,
            "best_params_json": str(json_path) if json_path is not None else None,
            "round_metrics_csv": str(csv_path) if csv_path is not None else None,
        }

        artifacts: Dict[str, Any] = {}
        if json_path is not None:
            artifacts["best_params_json"] = str(json_path)
        if csv_path is not None:
            artifacts["round_metrics_csv"] = str(csv_path)

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
