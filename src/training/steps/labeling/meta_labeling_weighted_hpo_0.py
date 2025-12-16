"""Meta-Labeling HPO Layer 0: Kalman/RTS Smoother Optimization.

This module handles Stage 0 of the hierarchical HPO process:
- Optimizes Kalman filter parameters (Q, R) for signal smoothing
- Uses RTS (Rauch-Tung-Striebel) smoother for acausal label generation
- Optimizes for Signal-to-Noise Ratio (SNR) of price series

The optimized parameters are used for:
1. RTS (acausal) smoother for generating training labels
2. Standard Kalman filter (causal) for live feature generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.training.steps.labeling.multi_label_voting_utils import (
    TripleBarrierConfig,
    compute_multi_triple_barrier_outcomes_vectorized,
    compute_kalman_smoothed_price_and_volatility,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)

# Import shared utilities
try:
    from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
        rts_smoother_1d,
        robust_labeling_loss,
        _write_hpo_stage_report,
        DEFAULT_RANDOM_SEED,
        get_reproducible_random_state,
        tprint_info,
        tprint_success,
        tprint_warning,
    )
except ImportError:
    # Fallback for standalone testing
    DEFAULT_RANDOM_SEED = 42
    def get_reproducible_random_state(base_seed: int = 42, offset: int = 0) -> int:
        return base_seed + offset
    def tprint_info(msg: str) -> None:
        print(f"[INFO] {msg}")
    def tprint_success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")
    def tprint_warning(msg: str) -> None:
        print(f"[WARNING] {msg}")


# Kalman search space for Layer 0
LAYER0_KALMAN_SEARCH_SPACE = {
    "kalman_Q": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "kalman_R": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
}


def create_kalman_objective(
    close_series: pd.Series,
    *,
    debug_limit: int = 5,
) -> callable:
    """Create the Kalman/RTS smoother optimization objective function.
    
    Args:
        close_series: Price series to optimize smoothing for
        debug_limit: Maximum number of debug messages to print
        
    Returns:
        Objective function that takes params dict and returns score
    """
    debug_count = 0
    
    def kalman_objective(params: Dict[str, Any]) -> float:
        """
        Stage 0: RTS Smoother Optimization for Label Generation.
        
        Objective: Maximize Signal-to-Noise Ratio (SNR) of the raw price series.
        Uses RTS (Rauch-Tung-Striebel) smoother which is ACAUSAL (zero-lag) - 
        ideal for generating training labels.
        
        Loss components:
        1. Smoothness: Minimal "wiggle" (2nd derivative)
        2. Tracking Error: RMSE from raw prices (bias/oversmoothing penalty)
        3. Amplitude Fidelity: Preserve ~95% of price volatility
        """
        nonlocal debug_count
        
        Q = params.get("kalman_Q", 1e-4)
        R = params.get("kalman_R", 0.01)
        
        # Get raw close prices
        raw_close = close_series.values
        
        if len(raw_close) < 100:
            return -10.0  # Reject if insufficient data
        
        try:
            # Run RTS Smoother (acausal, zero-lag)
            smoothed_close, smoothed_cov = rts_smoother_1d(
                prices=raw_close,
                Q=Q,
                R=R,
                init_val=None,
                init_cov=1.0,
            )
            
            # Compute robust labeling loss
            loss, details = robust_labeling_loss(
                smoothed=smoothed_close,
                raw=raw_close,
                alpha=1.0,   # Smoothness weight
                beta=1.0,    # Tracking error weight
                gamma=1.0,   # Amplitude fidelity weight
                is_acausal=True,
            )
            
            # Optimizer maximizes, so return negative loss
            # Add bonus for amplitude ratio being close to 0.95
            amp_ratio = details.get("amp_ratio", 0.95)
            amp_bonus = max(0, 0.1 - abs(amp_ratio - 0.95))
            
            score = -loss + amp_bonus
            
            return float(score) if np.isfinite(score) else -10.0
            
        except Exception as e:
            if debug_count < debug_limit:
                tprint_warning(f"[KALMAN_OBJ_ERROR] {e}")
                debug_count += 1
            return -10.0
    
    return kalman_objective


def run_layer0_kalman_optimization(
    *,
    close_series: pd.Series,
    config: Dict[str, Any],
    outcomes_dir: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    start_rank: int = 0,
    stage_rank: Dict[str, int] = None,
    load_stage_best_params_fn: Optional[callable] = None,
    start_at_canonical: str = "layer0",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run Layer 0 Kalman/RTS smoother optimization.
    
    Args:
        close_series: Price series for optimization
        config: HPO configuration
        outcomes_dir: Directory for saving results
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe string
        direction: Trading direction
        start_rank: Current start rank for layer skipping
        stage_rank: Dictionary mapping stage names to ranks
        load_stage_best_params_fn: Function to load best params for a stage
        start_at_canonical: Canonical name of starting stage
        
    Returns:
        Tuple of (best_kalman_params, kalman_result, stage0_report)
    """
    if stage_rank is None:
        stage_rank = {"stage0": 0}
    
    kalman_search_space = LAYER0_KALMAN_SEARCH_SPACE.copy()
    
    # Check if we should skip this layer
    stage0_loaded_from: Optional[str] = None
    if stage_rank.get("stage0", 0) < start_rank and load_stage_best_params_fn is not None:
        loaded_params, loaded_path = load_stage_best_params_fn("stage0")
        best_kalman_params = dict(loaded_params or {})
        stage0_loaded_from = str(loaded_path) if loaded_path is not None else None
        if not best_kalman_params:
            best_kalman_params = {"kalman_Q": 1e-4, "kalman_R": 0.01}
        kalman_result = {"best_params": dict(best_kalman_params), "best_value": 0.0, "history": []}
        tprint_info(
            f"♻️ Stage 0 skipped (start_at={start_at_canonical}); loaded best params from {stage0_loaded_from}"
        )
    else:
        # Create objective and run optimization
        kalman_objective = create_kalman_objective(close_series)
        
        n_trials = int(config.get("layer0_n_trials", 60))
        kalman_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(
                n_trials=n_trials,
                execution_mode="full",
                direction="maximize",
                seed=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=0)
            )
        )
        kalman_result = kalman_optimizer.optimize(
            objective=kalman_objective, 
            search_space=kalman_search_space
        )
        best_kalman_params = kalman_result.get("best_params", {})
    
    # Extract and validate best parameters
    best_Q = best_kalman_params.get("kalman_Q")
    best_R = best_kalman_params.get("kalman_R")
    
    try:
        best_Q = float(best_Q) if best_Q is not None else float("nan")
    except Exception:
        best_Q = float("nan")
    try:
        best_R = float(best_R) if best_R is not None else float("nan")
    except Exception:
        best_R = float("nan")
    
    if not np.isfinite(best_Q) or best_Q <= 0.0:
        best_Q = 1e-4
    if not np.isfinite(best_R) or best_R <= 0.0:
        best_R = 0.01
    
    # Update best params with validated values
    best_kalman_params["kalman_Q"] = best_Q
    best_kalman_params["kalman_R"] = best_R
    
    # Compute final loss details for logging
    kalman_loss: float = float("nan")
    kalman_loss_details: Dict[str, Any] = {}
    try:
        final_smoothed, _ = rts_smoother_1d(close_series.values, Q=best_Q, R=best_R)
        final_loss, final_details = robust_labeling_loss(
            final_smoothed, close_series.values, is_acausal=True
        )
        kalman_loss = float(final_loss)
        kalman_loss_details = final_details or {}
        tprint_success(
            f"✅ Layer 0 Complete. Loss: {final_loss:.4f} "
            f"(smooth={final_details['smooth']:.4f}, track={final_details['track']:.4f}, "
            f"amp={final_details['amp']:.4f}, amp_ratio={final_details['amp_ratio']:.3f})"
        )
    except Exception:
        try:
            bv = kalman_result.get("best_value", 0.0) if isinstance(kalman_result, dict) else 0.0
            bv = float(bv) if bv is not None and np.isfinite(float(bv)) else 0.0
        except Exception:
            bv = 0.0
        tprint_success(f"✅ Layer 0 Complete. Best Score: {bv:.4f}")
    
    tprint_info(f"   Best RTS/Kalman Params: Q={best_Q:.2e}, R={best_R:.2e}")
    tprint_info("   Note: RTS (acausal) for labels, Kalman (causal) for live features")
    
    # Save trial diagnostics
    stage0_csv: Optional[Path] = None
    try:
        kalman_history = kalman_result.get("history", []) if isinstance(kalman_result, dict) else []
        stage0_rows = []
        for trial in kalman_history:
            params = trial.get("params", {}) if isinstance(trial, dict) else {}
            q_val = float(params.get("kalman_Q", 1e-4))
            r_val = float(params.get("kalman_R", 0.01))
            
            # Recompute loss components for this (Q, R) pair
            try:
                smoothed_trial, _ = rts_smoother_1d(
                    prices=close_series.values,
                    Q=q_val,
                    R=r_val,
                    init_val=None,
                    init_cov=1.0,
                )
                loss_trial, details_trial = robust_labeling_loss(
                    smoothed=smoothed_trial,
                    raw=close_series.values,
                    is_acausal=True,
                )
            except Exception:
                loss_trial, details_trial = float("nan"), {}
            
            row = {
                "trial_number": trial.get("trial_number") if isinstance(trial, dict) else None,
                "kalman_Q": q_val,
                "kalman_R": r_val,
                "score": float(trial.get("value", float("nan"))) if isinstance(trial, dict) else float("nan"),
                "loss": float(loss_trial),
                "smooth": float(details_trial.get("smooth", float("nan"))),
                "track": float(details_trial.get("track", float("nan"))),
                "amp": float(details_trial.get("amp", float("nan"))),
                "amp_ratio": float(details_trial.get("amp_ratio", float("nan"))),
            }
            stage0_rows.append(row)
        
        if stage0_rows:
            ts_stage0 = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            stage0_csv = outcomes_dir / f"hpo_layer0_kalman_trials_{symbol}_{timeframe}_{ts_stage0}.csv"
            pd.DataFrame(stage0_rows).to_csv(stage0_csv, index=False)
            tprint_info(f"   💾 Saved Layer 0 Kalman trial diagnostics to {stage0_csv}")
    except Exception as stage0_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 0 Kalman trial diagnostics: {stage0_exc}")
    
    # Write stage report
    stage0_report: Dict[str, Any] = {}
    try:
        stage0_report = _write_hpo_stage_report(
            outcomes_dir=outcomes_dir,
            run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            stage_id="layer0_kalman",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            best_params=dict(best_kalman_params) if isinstance(best_kalman_params, dict) else {},
            metrics={
                "best_value": kalman_result.get("best_value", None),
                "loss": kalman_loss,
                "loss_details": kalman_loss_details,
            },
            search_space=kalman_search_space,
            trials_csv_path=stage0_csv,
            history_json_path=None,
        )
    except Exception as stage0_report_exc:
        tprint_warning(f"   ⚠️ Failed to write Layer 0 report: {stage0_report_exc}")
    
    return best_kalman_params, kalman_result, stage0_report


def run_committee_pre_step(
    *,
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    best_kalman_params: Dict[str, Any],
    config: Dict[str, Any],
    direction: str,
) -> Tuple[
    List[TripleBarrierConfig],
    List[str],
    Optional[pd.DatetimeIndex],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, Any],
]:
    """Run committee pre-step to compute expert matrices.
    
    Args:
        market_data: Market OHLCV data
        primary_signals: Primary trading signals
        best_kalman_params: Optimized Kalman parameters from Layer 0
        config: HPO configuration
        direction: Trading direction ('long' or 'short')
        
    Returns:
        Tuple containing:
        - committee_configs: List of TripleBarrierConfig objects
        - committee_names: List of expert names
        - committee_event_idx: Event index (DatetimeIndex)
        - committee_label_matrix_values: Label matrix (events x experts)
        - committee_returns_matrix_values: Returns matrix (events x experts)
        - committee_durations_matrix_values: Durations matrix (events x experts)
        - committee_confidence_matrix_values: Confidence matrix (events x experts)
        - best_committee_params: Best committee voting parameters
    """
    DEFAULT_TRANSACTION_COST = 0.0003
    
    # Initialize defaults
    committee_configs: List[TripleBarrierConfig] = []
    committee_names: List[str] = []
    committee_event_idx: Optional[pd.DatetimeIndex] = None
    committee_label_matrix_values: Optional[np.ndarray] = None
    committee_returns_matrix_values: Optional[np.ndarray] = None
    committee_durations_matrix_values: Optional[np.ndarray] = None
    committee_confidence_matrix_values: Optional[np.ndarray] = None
    
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
    
    enable_committee_pre_step = bool(config.get("enable_committee_pre_step", True))
    if not enable_committee_pre_step:
        return (
            committee_configs,
            committee_names,
            committee_event_idx,
            committee_label_matrix_values,
            committee_returns_matrix_values,
            committee_durations_matrix_values,
            committee_confidence_matrix_values,
            best_committee_params,
        )
    
    tprint_info("🧪 Committee pre-step: computing committee voting matrices...")
    
    # Build committee configs (6 base experts)
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
    
    # Pre-compute committee matrices
    try:
        best_Q_c = best_kalman_params.get("kalman_Q", 1e-4)
        best_R_c = best_kalman_params.get("kalman_R", 0.01)
        
        kalman_price_smooth_c, kalman_vol_smooth_c = compute_kalman_smoothed_price_and_volatility(
            prices=market_data["close"],
            process_noise=best_Q_c,
            measurement_noise=best_R_c,
            vol_window=20,
        )
        mk_data_voting_c = market_data.copy()
        mk_data_voting_c["kalman_price"] = kalman_price_smooth_c
        mk_data_voting_c["kalman_volatility"] = kalman_vol_smooth_c
        
        committee_results_c = compute_multi_triple_barrier_outcomes_vectorized(
            market_data=mk_data_voting_c,
            primary_signals=primary_signals,
            configs=committee_configs,
            transaction_cost=DEFAULT_TRANSACTION_COST,
        )
        
        event_mask_c = primary_signals["consensus"] != 0
        committee_event_idx = pd.DatetimeIndex(primary_signals[event_mask_c].index)
        
        # Try to add new experts
        new_expert_scores = None
        new_expert_conf = None
        try:
            from src.training.steps.labeling.layer2_advanced_logic import (
                compute_new_experts_matrix, 
                NEW_EXPERT_NAMES
            )
            
            dir_raw = str(direction).lower()
            dir_sign = 1
            if dir_raw in {"short", "sell", "-1", "s"}:
                dir_sign = -1
            
            new_expert_scores, new_expert_conf = compute_new_experts_matrix(
                market_data=mk_data_voting_c,
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
        
        # Initialize matrices
        committee_label_matrix_values = np.zeros(
            (len(committee_event_idx), n_total_experts),
            dtype=np.int8,
        )
        committee_returns_matrix_values = np.full(
            (len(committee_event_idx), n_total_experts),
            np.nan,
            dtype=np.float32,
        )
        committee_durations_matrix_values = np.full(
            (len(committee_event_idx), n_total_experts),
            np.nan,
            dtype=np.float32,
        )
        committee_confidence_matrix_values = np.full(
            (len(committee_event_idx), n_total_experts),
            np.nan,
            dtype=np.float32,
        )
        
        # Fill base expert columns
        for i, res in enumerate(committee_results_c):
            lbls = res["labels"].reindex(committee_event_idx).fillna(0).values.astype(int)
            rets = res["returns"].reindex(committee_event_idx).values.astype(np.float32)
            durs_s = res.get("durations")
            if not isinstance(durs_s, pd.Series):
                durs_s = res.get("event_durations")
            if isinstance(durs_s, pd.Series):
                dur_vals = durs_s.reindex(committee_event_idx).values.astype(np.float32)
            else:
                try:
                    h = float(getattr(committee_configs[i], "horizon", 1.0))
                except Exception:
                    h = 1.0
                dur_vals = np.full(int(len(committee_event_idx)), float(h), dtype=np.float32)
            conf = res.get("confidence")
            if isinstance(conf, pd.Series):
                conf_vals = conf.reindex(committee_event_idx).values.astype(np.float32)
            else:
                conf_vals = np.full(int(len(committee_event_idx)), 1.0, dtype=np.float32)
            
            committee_label_matrix_values[:, i] = lbls
            committee_returns_matrix_values[:, i] = rets
            committee_durations_matrix_values[:, i] = dur_vals
            committee_confidence_matrix_values[:, i] = conf_vals
        
        # Add new expert columns if available
        if new_expert_scores is not None and new_expert_conf is not None and n_new_experts == 3:
            try:
                avg_base_ret = float(np.nanmean(np.abs(committee_returns_matrix_values[:, :n_base_experts])))
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
        
        # Log new expert integration status
        if n_new_experts > 0:
            tprint_info(
                f"   [committee pre-step] New experts integrated: {n_new_experts} "
                f"(total={n_total_experts}, names={committee_names[-n_new_experts:]})"
            )
            
    except Exception as committee_matrix_exc:
        tprint_warning(f"⚠️ Committee pre-step matrix build failed: {committee_matrix_exc}")
        committee_event_idx = None
        committee_label_matrix_values = None
        committee_returns_matrix_values = None
        committee_durations_matrix_values = None
        committee_confidence_matrix_values = None
    
    return (
        committee_configs,
        committee_names,
        committee_event_idx,
        committee_label_matrix_values,
        committee_returns_matrix_values,
        committee_durations_matrix_values,
        committee_confidence_matrix_values,
        best_committee_params,
    )
