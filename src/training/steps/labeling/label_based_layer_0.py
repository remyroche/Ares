
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.steps.labeling.advanced_gating_logic import AdvancedGatingPipeline, compute_regime_labels_for_events
from src.training.steps.labeling.feature_generation_meta_labeling_step import DEFAULT_TRANSACTION_COST
from src.training.steps.labeling.multi_label_voting_utils import (
    TripleBarrierConfig,
    compute_kalman_smoothed_price_and_volatility,
    compute_multi_triple_barrier_outcomes_vectorized,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
from src.utils.tprint import tprint_info, tprint_success, tprint_warning


DEFAULT_RANDOM_SEED = 42


def get_reproducible_random_state(base_seed: int = DEFAULT_RANDOM_SEED, offset: int = 0) -> int:
    try:
        base_seed_i = int(base_seed)
    except Exception:
        base_seed_i = DEFAULT_RANDOM_SEED
    try:
        offset_i = int(offset)
    except Exception:
        offset_i = 0
    return int((base_seed_i + offset_i) % (2**31 - 1))


def rts_smoother_1d(
    prices: np.ndarray,
    volume: Optional[np.ndarray],
    Q: float,
    R: float,
    init_val: float = None,
    init_cov: float = 1.0,
) -> tuple:
    n = len(prices)
    obs = np.asarray(prices, dtype=np.float64)

    # Adaptive R scaling based on volume
    # High Volume -> Low Noise (Trust Price) -> Low R
    # R_t = R_base * (MedianVol / Vol_t)
    if volume is not None and len(volume) == n:
        vol = np.asarray(volume, dtype=np.float64)
        median_vol = np.nanmedian(vol)
        if median_vol > 0:
            # Avoid division by zero
            vol_safe = np.where(vol < 1e-8, 1e-8, vol)
            # Scaling factor: High vol -> Factor < 1 -> R decreases
            scale_factor = median_vol / vol_safe
            # Clip to prevent extreme scaling (e.g. 0.1x to 10x)
            scale_factor = np.clip(scale_factor, 0.1, 10.0)
            R_t = R * scale_factor
        else:
            R_t = np.full(n, R, dtype=np.float64)
    else:
        R_t = np.full(n, R, dtype=np.float64)

    m = np.zeros(n)
    P = np.zeros(n)

    m[0] = init_val if init_val is not None else obs[0]
    P[0] = init_cov

    for t in range(1, n):
        m_minus = m[t - 1]
        P_minus = P[t - 1] + Q

        # Use time-varying R
        r_val = R_t[t]

        K = P_minus / (P_minus + r_val)
        m[t] = m_minus + K * (obs[t] - m_minus)
        P[t] = (1 - K) * P_minus

    s_m = np.zeros(n)
    s_P = np.zeros(n)
    s_m[-1] = m[-1]
    s_P[-1] = P[-1]

    for t in range(n - 2, -1, -1):
        P_pred = P[t] + Q
        J = P[t] / P_pred if P_pred > 1e-12 else 0.0
        s_m[t] = m[t] + J * (s_m[t + 1] - m[t])
        s_P[t] = P[t] + (J**2) * (s_P[t + 1] - P_pred)

    return s_m, s_P


def robust_labeling_loss(
    smoothed: np.ndarray,
    raw: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    is_acausal: bool = True,
) -> tuple:
    s = np.asarray(smoothed, dtype=np.float64)
    r = np.asarray(raw, dtype=np.float64)

    s_ret = np.diff(s)
    r_ret = np.diff(r)
    raw_vol = np.std(r_ret) + 1e-9

    second_diff = np.diff(s, n=2)
    smooth_error = np.mean(second_diff**2) / (raw_vol**2)

    if is_acausal:
        rmse = np.sqrt(np.mean((s - r) ** 2))
        tracking_error = rmse / raw_vol
    else:
        tau = 1
        rmse = np.sqrt(np.mean((s[:-tau] - r[tau:]) ** 2))
        tracking_error = rmse / raw_vol

    std_s = np.std(s_ret)
    std_r = np.std(r_ret)
    amp_ratio = std_s / (std_r + 1e-9)
    amp_error = (amp_ratio - 0.95) ** 2

    total_loss = (alpha * smooth_error) + (beta * tracking_error) + (gamma * amp_error)
    return total_loss, {
        "loss": total_loss,
        "smooth": smooth_error,
        "track": tracking_error,
        "amp": amp_error,
        "amp_ratio": amp_ratio,
    }


@dataclass
class Layer0Output:
    best_kalman_params: Dict[str, Any]
    enable_committee_voting_hpo: bool
    enable_committee_weight_factor: bool
    enable_committee_pre_step: bool
    best_committee_params: Dict[str, Any]
    committee_loaded_from: Optional[str]
    committee_configs: List[TripleBarrierConfig]
    committee_names: List[str]
    committee_event_idx: Optional[pd.DatetimeIndex]
    committee_label_matrix_values: Optional[np.ndarray]
    committee_returns_matrix_values: Optional[np.ndarray]
    committee_durations_matrix_values: Optional[np.ndarray]
    committee_confidence_matrix_values: Optional[np.ndarray]
    advanced_gating_pipeline: Optional[AdvancedGatingPipeline]


def run_layer_0(
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    config: Dict[str, Any],
    outcomes_dir: Path,
    start_rank: int,
    stage_rank: Dict[str, int],
    start_at_canonical: str,
    load_stage_best_params: Callable[[str], Tuple[Dict[str, Any], Optional[Path]]],
) -> Layer0Output:
    close_series = market_data["close"]
    volume_series = market_data.get("volume", None)

    volume_values = None
    if volume_series is not None:
        volume_values = volume_series.values

    tprint_info("🧪 Stage 0: Optimizing Kalman Signal Parameters...")
    kalman_search_space = {
        "kalman_Q": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "kalman_R": {"type": "float", "low": 1e-4, "high": 2e-1, "log": True},
    }

    def kalman_objective(params: Dict[str, Any]) -> float:
        Q = float(params.get("kalman_Q", 1e-4))
        R = float(params.get("kalman_R", 0.01))
        raw_close = close_series.values
        if len(raw_close) < 100:
            return -10.0
        try:
            smoothed_close, _ = rts_smoother_1d(
                prices=raw_close,
                volume=volume_values,
                Q=Q,
                R=R,
                init_val=None,
                init_cov=1.0,
            )
            loss, details = robust_labeling_loss(
                smoothed=smoothed_close,
                raw=raw_close,
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
                is_acausal=True,
            )
            amp_ratio = float(details.get("amp_ratio", 0.95))
            amp_bonus = max(0.0, 0.1 - abs(amp_ratio - 0.95))
            score = -float(loss) + float(amp_bonus)
            return float(score) if np.isfinite(score) else -10.0
        except Exception as e:
            tprint_warning(f"[KALMAN_OBJ_ERROR] {e}")
            return -10.0

    stage0_loaded_from: Optional[str] = None
    if stage_rank.get("stage0", 0) < int(start_rank):
        loaded_params, loaded_path = load_stage_best_params("stage0")
        best_kalman_params = dict(loaded_params or {})
        stage0_loaded_from = str(loaded_path) if loaded_path is not None else None
        if not best_kalman_params:
            best_kalman_params = {"kalman_Q": 1e-4, "kalman_R": 0.01}
        tprint_info(
            f"♻️ Stage 0 skipped (start_at={start_at_canonical}); loaded best params from {stage0_loaded_from}"
        )
    else:
        kalman_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(
                n_trials=int(config.get("stage0_n_trials", 60)),
                execution_mode="full",
                direction="maximize",
                seed=get_reproducible_random_state(42, offset=0),
            )
        )
        kalman_result = kalman_optimizer.optimize(objective=kalman_objective, search_space=kalman_search_space)
        best_kalman_params = dict(kalman_result.get("best_params", {}) or {})

    try:
        best_Q = float(best_kalman_params.get("kalman_Q", 1e-4))
        best_R = float(best_kalman_params.get("kalman_R", 0.01))
    except Exception:
        best_Q, best_R = 1e-4, 0.01
    if (not np.isfinite(best_Q)) or best_Q <= 0.0:
        best_Q = 1e-4
    if (not np.isfinite(best_R)) or best_R <= 0.0:
        best_R = 0.01
    best_kalman_params["kalman_Q"] = float(best_Q)
    best_kalman_params["kalman_R"] = float(best_R)
    tprint_info(f"   Best RTS/Kalman Params: Q={best_Q:.2e}, R={best_R:.2e}")

    enable_committee_voting_hpo = bool(config.get("enable_committee_voting_hpo", True))
    enable_committee_weight_factor = bool(config.get("enable_committee_weight_factor", True))
    enable_committee_pre_step = bool(config.get("enable_committee_pre_step", True))
    enable_committee_pre_step = bool(
        enable_committee_pre_step and (enable_committee_voting_hpo or enable_committee_weight_factor)
    )

    best_committee_params: Dict[str, Any] = {
        "w_scalp": 1.0,
        "w_swing": 1.0,
        "w_trend": 1.0,
        "w_breakout": 0.5,
        "w_vwap_rev": 0.5,
        "w_vol_shock": 0.5,
        "consensus_quantile": float(config.get("committee_consensus_quantile_default", 0.90)),
        "consensus_threshold": float(config.get("consensus_threshold", 0.5)),
    }
    committee_loaded_from: Optional[str] = None

    committee_configs: List[TripleBarrierConfig] = []
    committee_names: List[str] = []

    committee_event_idx: Optional[pd.DatetimeIndex] = None
    committee_label_matrix_values: Optional[np.ndarray] = None
    committee_returns_matrix_values: Optional[np.ndarray] = None
    committee_durations_matrix_values: Optional[np.ndarray] = None
    committee_confidence_matrix_values: Optional[np.ndarray] = None

    if enable_committee_pre_step:
        tprint_info("🧪 Committee pre-step: building committee matrices...")

        base_profiles = {
            "scalp": (1.2, 0.6, 8),
            "swing": (1.8, 0.9, 12),
            "trend": (2.4, 1.2, 24),
        }
        vol_scalars = {"lower": 0.8, "upper": 1.2}
        for p_name, (tp_base, sl_base, h_base) in base_profiles.items():
            for v_name, v_scalar in vol_scalars.items():
                committee_configs.append(
                    TripleBarrierConfig(
                        tp_multiplier=tp_base * v_scalar,
                        sl_multiplier=sl_base * v_scalar,
                        horizon=h_base,
                    )
                )
                committee_names.append(f"{p_name}_{v_name}")

        try:
            # Pass volume_values here to align with RTS logic
            kalman_price_smooth, kalman_vol_smooth = compute_kalman_smoothed_price_and_volatility(
                prices=market_data["close"],
                volume=volume_series,  # Updated to pass volume
                vwap=market_data.get("vwap", None), # Updated to pass VWAP
                process_noise=float(best_Q),
                measurement_noise=float(best_R),
                vol_window=20,
            )
            mk_data_voting = market_data.copy()
            mk_data_voting["kalman_price"] = kalman_price_smooth
            mk_data_voting["kalman_volatility"] = kalman_vol_smooth

            committee_results = compute_multi_triple_barrier_outcomes_vectorized(
                market_data=mk_data_voting,
                primary_signals=primary_signals,
                configs=committee_configs,
                transaction_cost=DEFAULT_TRANSACTION_COST,
            )

            event_mask = primary_signals["consensus"] != 0
            committee_event_idx = pd.DatetimeIndex(primary_signals[event_mask].index)

            new_expert_scores = None
            new_expert_conf = None
            try:
                from src.training.steps.labeling.layer2_advanced_logic import (
                    NEW_EXPERT_NAMES,
                    compute_new_experts_matrix,
                )

                dir_raw = str(direction).lower()
                dir_sign = 1
                if dir_raw in {"short", "sell", "-1", "s"}:
                    dir_sign = -1

                new_expert_scores, new_expert_conf = compute_new_experts_matrix(
                    market_data=mk_data_voting,
                    event_idx=pd.DatetimeIndex(committee_event_idx),
                    direction=dir_sign,
                    breakout_lookback=20,
                    vwap_lookback=20,
                    vol_lookback=20,
                )
                committee_names.extend(list(NEW_EXPERT_NAMES))
            except Exception:
                new_expert_scores = None
                new_expert_conf = None

            n_base_experts = int(len(committee_configs))
            n_new_experts = 3 if new_expert_scores is not None else 0
            n_total_experts = int(n_base_experts + n_new_experts)

            committee_label_matrix_values = np.zeros((len(committee_event_idx), n_total_experts), dtype=np.int8)
            committee_returns_matrix_values = np.full(
                (len(committee_event_idx), n_total_experts), np.nan, dtype=np.float32
            )
            committee_durations_matrix_values = np.full(
                (len(committee_event_idx), n_total_experts), np.nan, dtype=np.float32
            )
            committee_confidence_matrix_values = np.full(
                (len(committee_event_idx), n_total_experts), np.nan, dtype=np.float32
            )

            for i, res in enumerate(committee_results):
                lbls = res["labels"].reindex(committee_event_idx).fillna(0).values.astype(int)
                rets = res["returns"].reindex(committee_event_idx).values.astype(np.float32)
                durs = res.get("durations")
                if not isinstance(durs, pd.Series):
                    durs = res.get("event_durations")
                if isinstance(durs, pd.Series):
                    dur_vals = durs.reindex(committee_event_idx).values.astype(np.float32)
                else:
                    dur_vals = np.full(int(len(committee_event_idx)), float(getattr(committee_configs[i], "horizon", 1.0)))
                conf = res.get("confidence")
                if isinstance(conf, pd.Series):
                    conf_vals = conf.reindex(committee_event_idx).values.astype(np.float32)
                else:
                    conf_vals = np.full(int(len(committee_event_idx)), 1.0, dtype=np.float32)

                committee_label_matrix_values[:, i] = lbls
                committee_returns_matrix_values[:, i] = rets
                committee_durations_matrix_values[:, i] = dur_vals
                committee_confidence_matrix_values[:, i] = conf_vals

            if new_expert_scores is not None and new_expert_conf is not None and n_new_experts == 3:
                try:
                    avg_base_ret = float(
                        np.nanmean(np.abs(committee_returns_matrix_values[:, :n_base_experts]))
                    )
                    if (not np.isfinite(avg_base_ret)) or avg_base_ret < 1e-6:
                        avg_base_ret = 0.001
                except Exception:
                    avg_base_ret = 0.001

                try:
                    med_dur = float(np.nanmedian(committee_durations_matrix_values[:, :n_base_experts]))
                    if (not np.isfinite(med_dur)) or med_dur < 1.0:
                        med_dur = 12.0
                except Exception:
                    med_dur = 12.0

                for j in range(3):
                    col_idx = n_base_experts + j
                    scores_j = np.asarray(new_expert_scores[:, j], dtype=float)
                    conf_j = np.asarray(new_expert_conf[:, j], dtype=float)
                    committee_label_matrix_values[:, col_idx] = np.sign(scores_j).astype(np.int8)
                    committee_returns_matrix_values[:, col_idx] = (scores_j * avg_base_ret).astype(np.float32)
                    committee_durations_matrix_values[:, col_idx] = np.full(
                        int(len(committee_event_idx)), med_dur, dtype=np.float32
                    )
                    committee_confidence_matrix_values[:, col_idx] = np.clip(conf_j, 0.0, 1.0).astype(np.float32)

            tprint_success(
                f"✅ Committee pre-step matrices: {committee_label_matrix_values.shape} (Events x Experts)"
            )
        except Exception as committee_matrix_exc:
            tprint_warning(f"⚠️ Committee pre-step matrix build failed: {committee_matrix_exc}")
            import traceback
            traceback.print_exc()
            committee_event_idx = None
            committee_label_matrix_values = None
            committee_returns_matrix_values = None
            committee_durations_matrix_values = None
            committee_confidence_matrix_values = None

        loaded_params, loaded_path = load_stage_best_params("committee")
        if isinstance(loaded_params, dict) and loaded_params:
            best_committee_params.update(dict(loaded_params))
            committee_loaded_from = str(loaded_path) if loaded_path is not None else None
            tprint_info(f"♻️ Loaded committee best params from {committee_loaded_from}")

    advanced_gating_pipeline: Optional[AdvancedGatingPipeline] = None
    try:
        enable_advanced_gating = bool(config.get("enable_advanced_gating", True))
        if (
            enable_advanced_gating
            and committee_label_matrix_values is not None
            and committee_returns_matrix_values is not None
            and committee_confidence_matrix_values is not None
            and committee_event_idx is not None
        ):
            tprint_info("🧪 Fitting Advanced Gating Pipeline...")
            n_experts_adv = int(committee_label_matrix_values.shape[1])

            adv_cfg = config.get("advanced_gating", {})
            if not isinstance(adv_cfg, dict):
                adv_cfg = {}

            advanced_gating_pipeline = AdvancedGatingPipeline(
                n_experts=n_experts_adv,
                enable_regime_barriers=bool(adv_cfg.get("enable_regime_barriers", True)),
                enable_meta_gate=bool(adv_cfg.get("enable_meta_gate", True)),
                enable_calibration=bool(adv_cfg.get("enable_calibration", True)),
                enable_abstention_aware=bool(adv_cfg.get("enable_abstention_aware", True)),
                enable_specialization=bool(adv_cfg.get("enable_specialization", True)),
                enable_diversity=bool(adv_cfg.get("enable_diversity", True)),
                meta_gate_mode=str(adv_cfg.get("meta_gate_mode", "weights")),
                calibration_method=str(adv_cfg.get("calibration_method", "isotonic")),
                coverage_min=float(adv_cfg.get("coverage_min", 0.3)),
                consensus_threshold=float(adv_cfg.get("consensus_threshold", 0.5)),
                specialization_strength=float(adv_cfg.get("specialization_strength", 0.5)),
                diversity_lambda=float(adv_cfg.get("diversity_lambda", 0.1)),
            )

            regime_labels_train = compute_regime_labels_for_events(
                market_data=market_data,
                event_idx=pd.DatetimeIndex(committee_event_idx),
            )

            w_scalp_adv = float(best_committee_params.get("w_scalp", 1.0))
            w_swing_adv = float(best_committee_params.get("w_swing", 1.0))
            w_trend_adv = float(best_committee_params.get("w_trend", 1.0))
            if n_experts_adv > 6:
                w_breakout_adv = float(best_committee_params.get("w_breakout", 0.5))
                w_vwap_adv = float(best_committee_params.get("w_vwap_rev", 0.5))
                w_vol_shock_adv = float(best_committee_params.get("w_vol_shock", 0.5))
                base_weights_adv = np.array(
                    [
                        w_scalp_adv,
                        w_scalp_adv,
                        w_swing_adv,
                        w_swing_adv,
                        w_trend_adv,
                        w_trend_adv,
                        w_breakout_adv,
                        w_vwap_adv,
                        w_vol_shock_adv,
                    ],
                    dtype=float,
                )
            else:
                base_weights_adv = np.array(
                    [
                        w_scalp_adv,
                        w_scalp_adv,
                        w_swing_adv,
                        w_swing_adv,
                        w_trend_adv,
                        w_trend_adv,
                    ],
                    dtype=float,
                )
            base_weights_adv = base_weights_adv / (np.sum(base_weights_adv) + 1e-8)

            lbl_train = np.asarray(committee_label_matrix_values, dtype=float)
            conf_train = np.asarray(committee_confidence_matrix_values, dtype=float)
            fired_train = lbl_train != 0
            sign_w = (
                np.where(fired_train, np.sign(lbl_train), 0.0)
                * conf_train
                * base_weights_adv.reshape(1, -1)
            )
            denom_train = (
                np.sum(fired_train.astype(float) * conf_train * base_weights_adv.reshape(1, -1), axis=1)
                + 1e-8
            )
            consensus_train = np.sum(sign_w, axis=1) / denom_train

            advanced_gating_pipeline.fit(
                market_data=market_data,
                event_idx=pd.DatetimeIndex(committee_event_idx),
                expert_returns=np.asarray(committee_returns_matrix_values, dtype=float),
                expert_labels=lbl_train,
                expert_confidences=conf_train,
                consensus_scores=consensus_train,
                regime_labels=regime_labels_train,
            )
            tprint_success(f"✅ Advanced Gating Pipeline fitted (n_experts={n_experts_adv})")
    except Exception as adv_exc:
        tprint_warning(f"⚠️ Advanced Gating Pipeline fitting failed: {adv_exc}")
        advanced_gating_pipeline = None

    return Layer0Output(
        best_kalman_params=dict(best_kalman_params),
        enable_committee_voting_hpo=bool(enable_committee_voting_hpo),
        enable_committee_weight_factor=bool(enable_committee_weight_factor),
        enable_committee_pre_step=bool(enable_committee_pre_step),
        best_committee_params=dict(best_committee_params),
        committee_loaded_from=committee_loaded_from,
        committee_configs=list(committee_configs),
        committee_names=list(committee_names),
        committee_event_idx=committee_event_idx,
        committee_label_matrix_values=committee_label_matrix_values,
        committee_returns_matrix_values=committee_returns_matrix_values,
        committee_durations_matrix_values=committee_durations_matrix_values,
        committee_confidence_matrix_values=committee_confidence_matrix_values,
        advanced_gating_pipeline=advanced_gating_pipeline,
    )
