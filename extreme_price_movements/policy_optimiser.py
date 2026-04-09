"""
policy_optimiser.py

Sequential 4-step policy optimisation using held-out data (10% tail from
SlicePlanner's outer-test set, reserved via the ``policy_optimiser`` consumer).

Steps (each reports metrics before & after):
1. TP/SL as ATR%        — grid search over tp_mult * barrier_pct, sl_mult * barrier_pct
2. Limit-order offset   — sweep base/max ATR% via simple_offset_generator helpers
3. MFE / time early exit — grid search over mfe_exit_frac, min_hold_bars
4. Trailing profit      — activation ATR%, giveback ATR%, MAE-gated variants;
                          trailing vs take-profit rule comparison

Best params are saved to ``policy_params/best_policy_params.json`` for
``holdout_strategy_eval.py`` to load during OOS evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.simple_offset_generator import (
    compute_offset_atr_pct_from_confidence,
    evaluate_offset_strategy,
)
from extreme_price_movements.simple_position_sizer import (
    evaluate_selection_profit_proxy,
)
from extreme_price_movements.utils import tprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensible default grids
# ---------------------------------------------------------------------------

TP_SL_CONFIG = {
    "tp_mult_grid": [0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
    "sl_mult_grid": [0.10, 0.15, 0.18, 0.25, 0.30, 0.40, 0.50],
}

OFFSET_CONFIG = {
    "base_offset_atr_pcts": [0.0, 0.0005, 0.001, 0.0015, 0.002],
    "max_offset_atr_pcts": [0.001, 0.002, 0.003, 0.005, 0.008],
    "offset_scalings": ["linear", "convex"],
}

MFE_EARLY_EXIT_CONFIG = {
    "mfe_exit_fracs": [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2],
    "min_hold_bars_grid": [2, 4, 6, 8, 12, 16],
}

TRAILING_CONFIG = {
    "activation_fracs": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "giveback_fracs": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    "mae_gate_fracs": [0.3, 0.5, 0.7, 1.0, 1.5],
}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _metric_score(
    rets: np.ndarray,
    timestamps: np.ndarray,
    cost_pct: float = 0.003,
) -> Dict[str, float]:
    if len(rets) == 0:
        return {
            "net_pnl": 0.0,
            "sortino": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
        }
    net_rets = rets - cost_pct
    pnl = float(np.sum(net_rets))
    hit = float(np.mean(net_rets > 0))
    gross_win = float(np.sum(net_rets[net_rets > 0]))
    gross_loss = float(np.abs(np.sum(net_rets[net_rets < 0])))
    pf = gross_win / gross_loss if gross_loss > 1e-9 else float(gross_win)
    downside = net_rets[net_rets < 0]
    ds_std = float(np.std(downside)) if len(downside) > 1 else 1e-6
    sortino = float(np.mean(net_rets)) / ds_std
    _, dd = _stable_equity_and_drawdown(net_rets)
    mdd = float(np.max(dd)) if len(dd) > 0 else 0.0
    return {
        "net_pnl": pnl,
        "sortino": sortino,
        "hit_rate": hit,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "n_trades": len(rets),
    }


def _pick_best(rows: List[Dict[str, Any]], metric: str = "net_pnl") -> Dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda r: r.get("metrics", {}).get(metric, -1e9))


def _report_step(
    step: int,
    label: str,
    best: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    m = best.get("metrics", {})
    parts = [f"  Step {step}/4 ({label}):"]
    for k, v in best.items():
        if k == "metrics":
            continue
        if isinstance(v, float):
            parts.append(f"  {k}={v:.4f}")
        else:
            parts.append(f"  {k}={v}")
    parts.append(f"  pnl={m.get('net_pnl', 0):.4f}")
    parts.append(f"  sortino={m.get('sortino', 0):.4f}")
    parts.append(f"  hit_rate={m.get('hit_rate', 0):.2%}")
    parts.append(f"  pf={m.get('profit_factor', 0):.3f}")
    parts.append(f"  mdd={m.get('max_drawdown', 0):.4f}")
    parts.append(f"  n={m.get('n_trades', 0)}")
    if extra:
        for ek, ev in extra.items():
            if isinstance(ev, float):
                parts.append(f"  {ek}={ev:.4f}")
            else:
                parts.append(f"  {ek}={ev}")
    tprint(" ".join(parts))


# ---------------------------------------------------------------------------
# Step 1: TP / SL as ATR%
# ---------------------------------------------------------------------------


def optimise_tp_sl(
    returns: np.ndarray,
    barrier_pct: np.ndarray,
    is_longs: np.ndarray,
    mfe_raw: np.ndarray,
    mae_raw: np.ndarray,
    timestamps: np.ndarray,
    tp_mult_grid: Optional[List[float]] = None,
    sl_mult_grid: Optional[List[float]] = None,
    cost_pct: float = 0.003,
) -> Dict[str, Any]:
    tp_grid = tp_mult_grid or TP_SL_CONFIG["tp_mult_grid"]
    sl_grid = sl_mult_grid or TP_SL_CONFIG["sl_mult_grid"]
    rows: List[Dict[str, Any]] = []

    for tp_m in tp_grid:
        for sl_m in sl_grid:
            if tp_m <= sl_m:
                continue
            tp_pct = tp_m * barrier_pct
            sl_pct = sl_m * barrier_pct
            tp_hit = mfe_raw >= tp_pct
            sl_hit = mae_raw >= sl_pct
            sim_rets = np.where(
                tp_hit & ~sl_hit,
                tp_pct,
                np.where(sl_hit & ~tp_hit, -sl_pct, returns),
            )
            metrics = _metric_score(sim_rets, timestamps, cost_pct)
            rows.append({"tp_mult": tp_m, "sl_mult": sl_m, "metrics": metrics})

    best = _pick_best(rows)
    _report_step(1, "TP/SL", best)
    return {"best": best, "all_results": rows}


# ---------------------------------------------------------------------------
# Step 2: Limit-order offset  (wired through simple_offset_generator)
# ---------------------------------------------------------------------------


def optimise_offset(
    returns: np.ndarray,
    confidence_scores: np.ndarray,
    timestamps: np.ndarray,
    barrier_pct: np.ndarray,
    cost_pct: float = 0.003,
    base_offsets: Optional[List[float]] = None,
    max_offsets: Optional[List[float]] = None,
    scalings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    base_grid = base_offsets or OFFSET_CONFIG["base_offset_atr_pcts"]
    max_grid = max_offsets or OFFSET_CONFIG["max_offset_atr_pcts"]
    scale_grid = scalings or OFFSET_CONFIG["offset_scalings"]
    rows: List[Dict[str, Any]] = []

    for base_o in base_grid:
        for max_o in max_grid:
            if max_o <= base_o:
                continue
            for scaling in scale_grid:
                offset_arr = compute_offset_atr_pct_from_confidence(
                    confidence_scores=confidence_scores,
                    base_offset_atr_pct=base_o,
                    max_offset_atr_pct=max_o,
                    confidence_threshold=0.0,
                    scaling=scaling,
                )
                fill_prob = np.exp(-offset_arr / (barrier_pct + 1e-9) * 2.0)
                fill_prob = np.clip(fill_prob, 0.3, 0.95)
                rng = np.random.RandomState(42)
                executed = rng.random(len(returns)) < fill_prob
                entry_improvement = offset_arr * 0.5
                adj_rets = np.where(executed, returns + entry_improvement, 0.0)
                if executed.sum() == 0:
                    continue
                metrics = _metric_score(
                    adj_rets[executed], timestamps[executed], cost_pct
                )
                metrics["fill_rate"] = float(np.mean(executed))
                metrics["avg_offset_atr_pct"] = float(np.mean(offset_arr))
                metrics["scaling"] = scaling
                rows.append(
                    {
                        "base_offset_atr_pct": base_o,
                        "max_offset_atr_pct": max_o,
                        "offset_scaling": scaling,
                        "metrics": metrics,
                    }
                )

    best = _pick_best(rows)
    _report_step(
        2,
        "Offset",
        best,
        {
            "fill_rate": best.get("metrics", {}).get("fill_rate", 0),
            "scaling": best.get("offset_scaling", "linear"),
        },
    )
    return {"best": best, "all_results": rows}


# ---------------------------------------------------------------------------
# Step 3: MFE / time-based early exit
# ---------------------------------------------------------------------------


def optimise_mfe_early_exit(
    returns: np.ndarray,
    mfe_raw: np.ndarray,
    mae_raw: np.ndarray,
    hold_bars: np.ndarray,
    timestamps: np.ndarray,
    barrier_pct: np.ndarray,
    is_longs: np.ndarray,
    cost_pct: float = 0.003,
    mfe_exit_fracs: Optional[List[float]] = None,
    min_hold_bars_grid: Optional[List[int]] = None,
) -> Dict[str, Any]:
    mfe_grid = mfe_exit_fracs or MFE_EARLY_EXIT_CONFIG["mfe_exit_fracs"]
    bars_grid = min_hold_bars_grid or MFE_EARLY_EXIT_CONFIG["min_hold_bars_grid"]
    rows: List[Dict[str, Any]] = []

    for mfe_frac in mfe_grid:
        for min_bars in bars_grid:
            mfe_threshold = mfe_frac * barrier_pct
            early_exit = (mfe_raw >= mfe_threshold) & (hold_bars >= min_bars)
            exit_rets = np.where(
                early_exit,
                mfe_threshold,
                returns,
            )
            exit_rets = np.clip(exit_rets, -barrier_pct * 2.0, barrier_pct * 2.0)
            metrics = _metric_score(exit_rets, timestamps, cost_pct)
            metrics["early_exit_rate"] = float(np.mean(early_exit))
            metrics["early_exit_n"] = int(np.sum(early_exit))
            rows.append(
                {
                    "mfe_exit_frac": mfe_frac,
                    "min_hold_bars": min_bars,
                    "metrics": metrics,
                }
            )

    best = _pick_best(rows)
    _report_step(
        3,
        "MFE early exit",
        best,
        {"exit_rate": best.get("metrics", {}).get("early_exit_rate", 0)},
    )
    return {"best": best, "all_results": rows}


# ---------------------------------------------------------------------------
# Step 4: Trailing profit  (activation ATR% + giveback ATR%, MAE-gated)
#
# Grid dimensions:
#   activation_frac : when MFE reaches this * barrier_pct  => trail starts
#   giveback_frac   : once activated, exit when peak-current >= this * barrier_pct
#   mae_gate_frac   : only activate if MAE <= this * barrier_pct
#                     None  => no MAE gate (pure trailing)
#
# Also evaluates a take-profit-only rule at activation_frac for comparison.
# ---------------------------------------------------------------------------


def optimise_trailing_profit(
    returns: np.ndarray,
    mfe_raw: np.ndarray,
    mae_raw: np.ndarray,
    barrier_pct: np.ndarray,
    timestamps: np.ndarray,
    is_longs: np.ndarray,
    cost_pct: float = 0.003,
    activation_fracs: Optional[List[float]] = None,
    giveback_fracs: Optional[List[float]] = None,
    mae_gate_fracs: Optional[List[Optional[float]]] = None,
) -> Dict[str, Any]:
    act_grid = activation_fracs or TRAILING_CONFIG["activation_fracs"]
    give_grid = giveback_fracs or TRAILING_CONFIG["giveback_fracs"]
    mae_grid = mae_gate_fracs or TRAILING_CONFIG["mae_gate_fracs"] + [None]
    rows: List[Dict[str, Any]] = []

    for act_f in act_grid:
        for give_f in give_grid:
            for mae_f in mae_grid:
                act_pct = act_f * barrier_pct
                give_pct = give_f * barrier_pct

                activated = mfe_raw >= act_pct
                if mae_f is not None:
                    mae_pct = mae_f * barrier_pct
                    activated = activated & (mae_raw <= mae_pct)

                running_peak = np.maximum(mfe_raw, act_pct)
                giveback = running_peak - np.maximum(returns, 0.0)
                trail_exit = activated & (giveback >= give_pct)

                trail_rets = np.where(
                    trail_exit,
                    np.maximum(act_pct - give_pct, -barrier_pct),
                    returns,
                )

                tp_only_exit = mfe_raw >= act_pct
                tp_rets = np.where(tp_only_exit, act_pct, returns)

                trail_metrics = _metric_score(trail_rets, timestamps, cost_pct)
                tp_metrics = _metric_score(tp_rets, timestamps, cost_pct)

                trail_metrics["trail_activation_rate"] = float(np.mean(activated))
                trail_metrics["trail_exit_rate"] = float(np.mean(trail_exit))
                trail_metrics["tp_exit_rate"] = float(np.mean(tp_only_exit))

                better_rule = (
                    "trailing"
                    if trail_metrics["net_pnl"] >= tp_metrics["net_pnl"]
                    else "tp_only"
                )
                trail_metrics["better_rule"] = better_rule
                trail_metrics["tp_only_pnl"] = tp_metrics["net_pnl"]
                trail_metrics["trail_vs_tp_delta"] = (
                    trail_metrics["net_pnl"] - tp_metrics["net_pnl"]
                )

                rows.append(
                    {
                        "activation_frac": act_f,
                        "giveback_frac": give_f,
                        "mae_gate_frac": mae_f,
                        "mae_gated": mae_f is not None,
                        "better_rule": better_rule,
                        "metrics": trail_metrics,
                    }
                )

    best = _pick_best(rows)
    best_rule = best.get("better_rule", "trailing")
    _report_step(
        4,
        "Trailing profit",
        best,
        {
            "better_rule": best_rule,
            "mae_gate": best.get("mae_gate_frac"),
            "trail_act_rate": best.get("metrics", {}).get("trail_activation_rate", 0),
        },
    )
    return {"best": best, "all_results": rows}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_policy_optimisation(
    data_root: str,
    run_id: str,
    sizer_results: Optional[Dict[str, Any]] = None,
    holdout_frac: float = 0.10,
    cost_pct: float = 0.003,
) -> Dict[str, Any]:
    tprint(f"POLICY OPTIMISER START (holdout_frac={holdout_frac})")

    output_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    output_dir.mkdir(parents=True, exist_ok=True)

    from extreme_price_movements.run_ridge_sizer import (
        load_meta_oof_predictions,
        load_trade_outcomes,
    )

    try:
        meta_oofs = load_meta_oof_predictions(data_root, run_id)
    except FileNotFoundError:
        tprint("No meta OOF predictions found for policy optimiser, skipping.")
        return {}

    all_strategy_params: Dict[str, Any] = {}

    for bucket_name, oof_df in meta_oofs.items():
        tprint(f"\n--- Policy optimisation for {bucket_name} ---")
        try:
            trade_outcomes = load_trade_outcomes(data_root, run_id, oof_df)
        except FileNotFoundError:
            tprint(f"  No trade outcomes for {bucket_name}, skipping.")
            continue

        ret_col = "return" if "return" in trade_outcomes.columns else "net_return"
        if ret_col not in trade_outcomes.columns:
            tprint(f"  No return column for {bucket_name}, skipping.")
            continue

        returns = trade_outcomes[ret_col].values.astype(np.float32)
        timestamps = (
            pd.to_datetime(trade_outcomes["timestamp"]).values
            if "timestamp" in trade_outcomes.columns
            else np.arange(len(returns))
        )
        is_longs = (
            trade_outcomes["is_long"].values.astype(bool)
            if "is_long" in trade_outcomes.columns
            else np.ones(len(returns), dtype=bool)
        )
        barrier_pct = np.full(len(returns), 0.04, dtype=np.float32)
        if "mae_ret" in trade_outcomes.columns:
            barrier_pct = np.clip(
                np.abs(trade_outcomes["mae_ret"].values.astype(np.float32)) * 2.5,
                0.01,
                0.20,
            )

        mfe_raw = (
            trade_outcomes["mfe_ret"].values.astype(np.float32)
            if "mfe_ret" in trade_outcomes.columns
            else np.abs(returns) * 1.5
        )
        mae_raw = (
            trade_outcomes["mae_ret"].values.astype(np.float32)
            if "mae_ret" in trade_outcomes.columns
            else np.abs(returns) * 0.8
        )
        hold_bars = np.full(len(returns), 16, dtype=np.int32)
        if "exit_code" in trade_outcomes.columns:
            hold_bars = np.where(
                trade_outcomes["exit_code"].values == "tp", 8, 16
            ).astype(np.int32)

        confidence = (
            trade_outcomes["oof_u_hat"].values.astype(np.float32)
            if "oof_u_hat" in trade_outcomes.columns
            else np.zeros(len(returns), dtype=np.float32)
        )

        n_hold = max(1, int(len(returns) * holdout_frac))
        if len(returns) < 100:
            tprint(f"  Too few samples ({len(returns)}) for {bucket_name}, skipping.")
            continue

        sorted_idx = (
            np.argsort(timestamps)
            if timestamps is not None
            else np.arange(len(returns))
        )
        hold_idx = sorted_idx[-n_hold:]

        h_returns = returns[hold_idx]
        h_ts = timestamps[hold_idx]
        h_barrier = barrier_pct[hold_idx]
        h_is_longs = is_longs[hold_idx]
        h_mfe = mfe_raw[hold_idx]
        h_mae = mae_raw[hold_idx]
        h_hold_bars = hold_bars[hold_idx]
        h_confidence = confidence[hold_idx]

        tprint(f"  Holdout slice: {len(hold_idx)} rows")

        # ---------- baseline (no policy changes) ----------
        baseline = _metric_score(h_returns, h_ts, cost_pct)
        tprint(
            f"  Baseline: pnl={baseline['net_pnl']:.4f} "
            f"sortino={baseline['sortino']:.4f} "
            f"hit={baseline['hit_rate']:.2%} n={baseline['n_trades']}"
        )

        # --- Step 1: TP/SL ---
        tprint("  Running Step 1/4: TP/SL optimisation...")
        step1 = optimise_tp_sl(
            h_returns,
            h_barrier,
            h_is_longs,
            h_mfe,
            h_mae,
            h_ts,
            cost_pct=cost_pct,
        )

        # Apply step-1 best TP/SL for subsequent steps
        best_tp_m = step1["best"].get("tp_mult", 1.0)
        best_sl_m = step1["best"].get("sl_mult", 0.5)
        tp_pct = best_tp_m * h_barrier
        sl_pct = best_sl_m * h_barrier
        tp_hit = h_mfe >= tp_pct
        sl_hit = h_mae >= sl_pct
        rets_after_1 = np.where(
            tp_hit & ~sl_hit,
            tp_pct,
            np.where(sl_hit & ~tp_hit, -sl_pct, h_returns),
        )

        # --- Step 2: Offset ---
        tprint("  Running Step 2/4: Offset optimisation...")
        step2 = optimise_offset(
            rets_after_1,
            h_confidence,
            h_ts,
            h_barrier,
            cost_pct=cost_pct,
        )

        # --- Step 3: MFE early exit ---
        tprint("  Running Step 3/4: MFE early exit optimisation...")
        step3 = optimise_mfe_early_exit(
            rets_after_1,
            h_mfe,
            h_mae,
            h_hold_bars,
            h_ts,
            h_barrier,
            h_is_longs,
            cost_pct=cost_pct,
        )

        # Apply step-3 early exit for step-4 input
        best_mfe_frac = step3["best"].get("mfe_exit_frac", 0.0)
        best_min_bars = step3["best"].get("min_hold_bars", 0)
        mfe_threshold = best_mfe_frac * h_barrier
        early_exit = (h_mfe >= mfe_threshold) & (h_hold_bars >= best_min_bars)
        rets_after_3 = np.where(early_exit, mfe_threshold, rets_after_1)
        rets_after_3 = np.clip(rets_after_3, -h_barrier * 2.0, h_barrier * 2.0)

        # --- Step 4: Trailing profit ---
        tprint("  Running Step 4/4: Trailing profit optimisation...")
        step4 = optimise_trailing_profit(
            rets_after_3,
            h_mfe,
            h_mae,
            h_barrier,
            h_ts,
            h_is_longs,
            cost_pct=cost_pct,
        )

        best_params = {
            "strategy_id": bucket_name,
            "tp_mult": best_tp_m,
            "sl_mult": best_sl_m,
            "base_offset_atr_pct": step2["best"].get("base_offset_atr_pct", 0.0),
            "max_offset_atr_pct": step2["best"].get("max_offset_atr_pct", 0.0),
            "offset_scaling": step2["best"].get("offset_scaling", "linear"),
            "mfe_exit_frac": best_mfe_frac,
            "min_hold_bars": best_min_bars,
            "activation_frac": step4["best"].get("activation_frac", 0.5),
            "giveback_frac": step4["best"].get("giveback_frac", 0.3),
            "mae_gate_frac": step4["best"].get("mae_gate_frac"),
            "mae_gated": step4["best"].get("mae_gated", False),
            "better_rule": step4["best"].get("better_rule", "trailing"),
            "metrics_baseline": baseline,
            "metrics_step1": step1["best"].get("metrics", {}),
            "metrics_step2": step2["best"].get("metrics", {}),
            "metrics_step3": step3["best"].get("metrics", {}),
            "metrics_step4": step4["best"].get("metrics", {}),
            "holdout_n": int(len(hold_idx)),
        }
        all_strategy_params[bucket_name] = best_params

    if not all_strategy_params:
        tprint("POLICY OPTIMISER: no strategies optimised.")
        return {}

    payload = {
        "schema_version": "v2",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "holdout_frac": holdout_frac,
        "cost_pct": cost_pct,
        "strategies": list(all_strategy_params.values()),
    }
    params_path = output_dir / "best_policy_params.json"
    params_path.write_text(json.dumps(payload, indent=2, default=str))
    tprint(
        f"POLICY OPTIMISER COMPLETE: {len(all_strategy_params)} strategies -> {params_path}"
    )

    sym_link_path = Path(data_root) / "artifacts" / run_id / "best_policy_params.json"
    try:
        if sym_link_path.exists():
            sym_link_path.unlink()
        sym_link_path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass

    return payload
