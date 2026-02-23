from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

import numpy as np


def load_best_policy_params_from_optimise(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load optimize output params for policy-aligned ML targets.

    Expected path: {data_root}/artifacts/{run_id}/models/bucket_params.json
    Returns flat/default params when bucket-specific params are unavailable.
    """
    path = os.path.join(data_root, "artifacts", run_id, "models", "bucket_params.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            blob = json.load(f)
        if isinstance(blob, dict):
            return blob
    except Exception:
        return {}
    return {}


def load_optimise_best_policy_params(run_dir: str) -> Dict[str, Any]:
    """Compatibility helper: load optimise best-policy params from a run directory.

    Args:
        run_dir: Path like ``{data_root}/artifacts/{run_id}``.
    """
    path = os.path.join(str(run_dir), "models", "bucket_params.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            blob = json.load(f)
        return blob if isinstance(blob, dict) else {}
    except Exception:
        return {}


@dataclass
class PolicyOutcome:
    r_policy: float
    bars_held: int
    exit_code: int  # 0=SL, 1=TO, 2=TP
    mae: float
    mfe: float
    reason: str


def _reason_to_exit_code(reason: str, r_policy: float) -> int:
    rr = str(reason or "")
    if rr in {"stop_loss", "early_invalidation"}:
        return 0
    if rr in {"trailing_stop", "take_profit", "tp", "giveback_exit"}:
        return 2
    if rr in {"time_exit", "timeout", "no_entry", "limit_not_filled"}:
        return 1
    # Fallback by realized sign
    return 2 if float(r_policy) > 0.0 else (0 if float(r_policy) < 0.0 else 1)


def policy_rollout_engine(
    ohlc: pd.DataFrame,
    atr_pct: pd.Series,
    t0: int,
    direction: int,
    policy_params: Dict[str, Any],
    max_hold_hours: int,
) -> PolicyOutcome:
    """Roll out trade with the *same* codepath used by engine/backtest."""
    from extreme_price_movements.engine import simulate_trade_hourly

    idx = ohlc.index
    ts_entry = idx[int(t0)]
    side = "long" if int(direction) > 0 else "short"
    entry_px = float(ohlc.loc[ts_entry, "open"])
    ret, exit_ts, reason, extras = simulate_trade_hourly(
        o_s=ohlc["open"],
        h_s=ohlc["high"],
        l_s=ohlc["low"],
        c_s=ohlc["close"],
        feats_s=atr_pct,
        ts_entry=ts_entry,
        entry_px=entry_px,
        side=side,
        cfg=dict(policy_params),
        max_hold_hours=int(max_hold_hours),
        exchange=None,
        symbol=None,
        cost=None,
    )
    bars = int(max(0, ((pd.Timestamp(exit_ts) - pd.Timestamp(ts_entry)).total_seconds() // 3600)))
    mae = float((extras or {}).get("mae_pct", 0.0))
    mfe = float((extras or {}).get("mfe_pct", 0.0))
    return PolicyOutcome(
        r_policy=float(ret),
        bars_held=bars,
        exit_code=_reason_to_exit_code(reason, float(ret)),
        mae=mae,
        mfe=mfe,
        reason=str(reason),
    )


def policy_rollout_ml(
    ohlc: pd.DataFrame,
    atr_pct: pd.Series,
    t0: int,
    direction: int,
    policy_params: Dict[str, Any],
    max_hold_hours: int,
) -> PolicyOutcome:
    """ML rollout wrapper; intentionally delegates to engine rollout path."""
    return policy_rollout_engine(
        ohlc=ohlc,
        atr_pct=atr_pct,
        t0=t0,
        direction=direction,
        policy_params=policy_params,
        max_hold_hours=max_hold_hours,
    )


def compute_u_policy_labels(
    ohlc: pd.DataFrame,
    atr_pct: pd.Series,
    event_index: np.ndarray,
    direction: int,
    policy_params: Dict[str, Any],
    max_hold_hours: int,
    fee_rt: float = 0.003,
) -> Dict[str, np.ndarray]:
    """Compute engine-aligned policy labels for provided event indices.

    Returns arrays aligned to ``event_index`` with keys:
    ``r_policy``, ``u_policy``, ``exit_code``, ``mae``, ``mfe``, ``duration``.
    """
    idx = np.asarray(event_index, dtype=int)
    n = len(idx)
    r_policy = np.zeros(n, dtype=np.float32)
    r_policy_net = np.zeros(n, dtype=np.float32)
    u_policy = np.zeros(n, dtype=np.float32)
    u_policy_net = np.zeros(n, dtype=np.float32)
    exit_code = np.ones(n, dtype=np.int8)
    early_inval = np.zeros(n, dtype=np.int8)
    mae = np.zeros(n, dtype=np.float32)
    mfe = np.zeros(n, dtype=np.float32)
    duration = np.zeros(n, dtype=np.int16)
    for i, t0 in enumerate(idx):
        if int(t0) < 0 or int(t0) >= len(ohlc.index):
            continue
        out = policy_rollout_engine(
            ohlc=ohlc,
            atr_pct=atr_pct,
            t0=int(t0),
            direction=int(direction),
            policy_params=policy_params,
            max_hold_hours=int(max_hold_hours),
        )
        gross = float(out.r_policy)
        net = (1.0 + gross) * (1.0 - float(fee_rt)) - 1.0
        r_policy[i] = gross
        r_policy_net[i] = net
        u_policy[i] = float(np.log1p(max(-0.999999, gross)))
        u_policy_net[i] = float(np.log1p(max(-0.999999, net)))
        exit_code[i] = int(out.exit_code)
        early_inval[i] = np.int8(str(out.reason) == "early_invalidation")
        mae[i] = float(out.mae)
        mfe[i] = float(out.mfe)
        duration[i] = int(out.bars_held)

    # Sanity checks: with non-negative round-trip fee, net utility should not exceed gross utility.
    _gross_u = np.log1p(np.clip(r_policy.astype(np.float64), -0.999999, None))
    _net_u = np.log1p(np.clip(r_policy_net.astype(np.float64), -0.999999, None))
    assert bool(np.all(np.isfinite(_net_u))), "u_policy_net contains non-finite values"
    assert bool(np.all(r_policy_net <= r_policy + 1e-12)), "r_policy_net must be <= r_policy for fee_rt >= 0"
    assert bool(np.all(_net_u <= _gross_u + 1e-12)), "u_policy_net must be <= u_policy for fee_rt >= 0"
    return {
        "r_policy": r_policy,
        "r_policy_net": r_policy_net,
        "u_policy": u_policy,
        "u_policy_net": u_policy_net,
        "exit_code": exit_code,
        "early_inval": early_inval,
        "mae": mae,
        "mfe": mfe,
        "duration": duration,
    }


def build_base_tp_vs_sl(exit_code: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build TP-vs-SL labels while excluding timeout rows.

    exit_code semantics: 0=SL, 1=TO, 2=TP.
    Returns (y_bin, mask_over_original_rows).
    """
    exit_code = np.asarray(exit_code, np.int32)
    mask = (exit_code == 0) | (exit_code == 2)
    y = (exit_code[mask] == 2).astype(np.int32)
    return y, mask


def log_mae(mae_ret: np.ndarray) -> np.ndarray:
    mae_ret = np.asarray(mae_ret, np.float64)
    return np.log1p(np.maximum(mae_ret, 0.0))


def sizing_score(mu_hat: np.ndarray, mae_ret_hat: np.ndarray, p_work: np.ndarray,
                 p_sl: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Dimension-consistent sizing score in utility/risk space."""
    mu_hat = np.asarray(mu_hat, np.float64)
    risk_hat = log_mae(mae_ret_hat)
    s = (mu_hat / (risk_hat + eps)) * np.asarray(p_work, np.float64)
    if p_sl is not None:
        s *= (1.0 - np.asarray(p_sl, np.float64))
    return s


@dataclass
class MetaClassifierSelectionConfig:
    max_logloss: float = 1.10
    # If True, utility weights are estimated from fold-level realized utility by class.
    dynamic_utility_from_realized: bool = True
    u_tp: float = 1.0
    u_to: float = 0.0
    u_sl: float = -3.0
    top_frac: float = 0.30
    min_top_n: int = 50
    min_lift_vs_baseline: float = 0.0
    require_positive_oof_utility: bool = True


def expected_utility(p_pred: np.ndarray, u_tp: float, u_to: float, u_sl: float) -> np.ndarray:
    """Expected utility from multiclass [SL, TO, TP] probabilities."""
    return p_pred[:, 2] * u_tp + p_pred[:, 1] * u_to + p_pred[:, 0] * u_sl


def pick_meta_classifier_by_utility_top30(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    realized_u_policy: np.ndarray,
    cfg: MetaClassifierSelectionConfig,
    trade_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    from sklearn.metrics import log_loss

    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred, dtype=float)
    realized_u_policy = np.asarray(realized_u_policy, dtype=float)
    tm = np.ones(len(y_true), dtype=bool) if trade_mask is None else np.asarray(trade_mask, dtype=bool)
    tm = tm[:len(y_true)]

    ll = float(log_loss(y_true[tm], p_pred[tm], labels=[0, 1, 2])) if np.any(tm) else 999.0
    passed_gate = ll <= float(cfg.max_logloss)

    # Dynamic utility weights from realized outcomes on this OOF vector.
    if bool(cfg.dynamic_utility_from_realized):
        def _class_mean(k: int, dflt: float) -> float:
            m = (y_true == k) & tm
            if np.any(m):
                v = float(np.nanmean(realized_u_policy[m]))
                if np.isfinite(v):
                    return v
            return dflt
        u_sl = _class_mean(0, float(cfg.u_sl))
        u_to = _class_mean(1, float(cfg.u_to))
        u_tp = _class_mean(2, float(cfg.u_tp))
    else:
        u_tp, u_to, u_sl = float(cfg.u_tp), float(cfg.u_to), float(cfg.u_sl)

    U = expected_utility(p_pred, u_tp, u_to, u_sl)
    valid_idx = np.where(tm)[0]
    if len(valid_idx) == 0:
        return {"logloss": ll, "passed_gate": 0.0, "topU_mean": float("nan"), "top_realized_u_mean": float("nan"), "baseline_realized_u_mean": float("nan"), "realized_lift_vs_baseline": float("nan"), "top_n": 0.0, "top_n_ok": 0.0, "lift_ok": 0.0, "u_tp": float(u_tp), "u_to": float(u_to), "u_sl": float(u_sl), "passed_econ": 0.0, "selection_score": float("-inf") }
    U = U[valid_idx]
    realized_u_policy = realized_u_policy[valid_idx]
    n = len(U)
    k = max(1, int(np.ceil(float(cfg.top_frac) * n)))
    idx = np.argsort(-U)[:k]

    topU_mean = float(np.mean(U[idx]))
    top_realized_u_mean = float(np.mean(realized_u_policy[idx]))
    baseline_realized_u = float(np.mean(realized_u_policy))
    realized_lift = top_realized_u_mean - baseline_realized_u

    top_n_ok = bool(k >= int(cfg.min_top_n))
    lift_ok = bool(realized_lift >= float(cfg.min_lift_vs_baseline))
    passed_econ = (top_realized_u_mean > 0.0) if bool(cfg.require_positive_oof_utility) else True
    passed_econ = bool(passed_econ and top_n_ok and lift_ok)

    return {
        "logloss": ll,
        "passed_gate": float(passed_gate),
        "topU_mean": topU_mean,
        "top_realized_u_mean": top_realized_u_mean,
        "baseline_realized_u_mean": baseline_realized_u,
        "realized_lift_vs_baseline": realized_lift,
        "top_n": float(k),
        "top_n_ok": float(top_n_ok),
        "lift_ok": float(lift_ok),
        "u_tp": float(u_tp),
        "u_to": float(u_to),
        "u_sl": float(u_sl),
        "passed_econ": float(passed_econ),
        "selection_score": topU_mean,
    }
