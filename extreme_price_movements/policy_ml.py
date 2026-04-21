from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

import numpy as np
from extreme_price_movements.utils import tprint


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


def _get_static_sizer_policy_params(
    policy_params: Dict[str, Any],
    max_hold_hours: int,
) -> Dict[str, float]:
    """Build a fixed TP/SL policy aligned with the ridge sizer utility."""
    sl_atr_mult = float(policy_params.get("policy_label_sl_atr_mult", 1.2))
    tp_sl_ratio = float(policy_params.get("policy_label_tp_sl_ratio", 2.0))
    trailing_pct = float(policy_params.get("policy_label_trailing_pct", 0.35))
    hold_hours = int(policy_params.get("policy_label_max_hold_hours", max_hold_hours))
    return {
        "sl_atr_mult": max(sl_atr_mult, 1e-6),
        "tp_sl_ratio": max(tp_sl_ratio, 1e-6),
        "trailing_pct": float(np.clip(trailing_pct, 0.0, 0.95)),
        "max_hold_hours": max(1, hold_hours),
    }


def _policy_rollout_sizer_aligned(
    ohlc: pd.DataFrame,
    atr_pct: pd.Series,
    t0: int,
    direction: int,
    policy_params: Dict[str, Any],
    max_hold_hours: int,
) -> PolicyOutcome:
    """Roll out trade with the same TP/SL/trailing semantics as the ridge sizer."""
    from extreme_price_movements.ridge_position_sizer import simulate_trade_exit

    if int(t0) < 0 or int(t0) >= len(ohlc.index):
        return PolicyOutcome(0.0, 0, 1, 0.0, 0.0, "timeout")

    static_policy = _get_static_sizer_policy_params(policy_params, max_hold_hours)
    hold_hours = int(static_policy["max_hold_hours"])
    idx = ohlc.index
    ts_entry = idx[int(t0)]
    entry_px = float(ohlc["open"].iloc[int(t0)])
    if not np.isfinite(entry_px) or entry_px <= 0.0:
        return PolicyOutcome(0.0, 0, 1, 0.0, 0.0, "timeout")

    atr_here = float(atr_pct.loc[ts_entry]) if ts_entry in atr_pct.index else float(atr_pct.iloc[int(t0)]) if int(t0) < len(atr_pct) else 0.02
    atr_here = max(float(np.nan_to_num(atr_here, nan=0.02)), 1e-6)
    sl_abs = static_policy["sl_atr_mult"] * atr_here * entry_px
    tp_abs = static_policy["tp_sl_ratio"] * sl_abs
    is_long = int(direction) > 0
    if is_long:
        tp_price = entry_px + tp_abs
        sl_price = entry_px - sl_abs
    else:
        tp_price = entry_px - tp_abs
        sl_price = entry_px + sl_abs

    start = int(t0) + 1
    stop = min(len(ohlc), start + hold_hours)
    if stop <= start:
        return PolicyOutcome(0.0, 0, 1, 0.0, 0.0, "timeout")

    o_arr = ohlc["open"].iloc[start:stop].to_numpy(dtype=np.float64, copy=False)
    h_arr = ohlc["high"].iloc[start:stop].to_numpy(dtype=np.float64, copy=False)
    l_arr = ohlc["low"].iloc[start:stop].to_numpy(dtype=np.float64, copy=False)
    c_arr = ohlc["close"].iloc[start:stop].to_numpy(dtype=np.float64, copy=False)
    if len(h_arr) == 0:
        return PolicyOutcome(0.0, 0, 1, 0.0, 0.0, "timeout")

    exit_price, exit_bar, exit_reason = simulate_trade_exit(
        highs=h_arr,
        lows=l_arr,
        opens=o_arr,
        closes=c_arr,
        entry_price=float(entry_px),
        is_long=bool(is_long),
        tp_price=float(tp_price),
        sl_price=float(sl_price),
        trailing_pct=float(static_policy["trailing_pct"]),
        max_bars=len(h_arr),
    )
    bars_held = int(max(1, exit_bar + 1))
    if is_long:
        gross = float(exit_price / (entry_px + 1e-12) - 1.0)
        favorable = (h_arr / (entry_px + 1e-12)) - 1.0
        adverse = (l_arr / (entry_px + 1e-12)) - 1.0
        mfe = float(np.nanmax(np.clip(favorable, 0.0, None))) if len(favorable) else 0.0
        mae = float(np.nanmax(np.clip(-adverse, 0.0, None))) if len(adverse) else 0.0
    else:
        gross = float(entry_px / (exit_price + 1e-12) - 1.0)
        favorable = (entry_px / np.clip(l_arr, 1e-12, None)) - 1.0
        adverse = (h_arr / (entry_px + 1e-12)) - 1.0
        mfe = float(np.nanmax(np.clip(favorable, 0.0, None))) if len(favorable) else 0.0
        mae = float(np.nanmax(np.clip(adverse, 0.0, None))) if len(adverse) else 0.0

    if exit_reason == 1:
        exit_code, reason = 0, "stop_loss"
    elif exit_reason == 3:
        exit_code, reason = 1, "timeout"
    elif exit_reason in (0, 2):
        exit_code, reason = 2, ("trailing_stop" if exit_reason == 2 else "take_profit")
    else:
        exit_code, reason = 1, "timeout"
    return PolicyOutcome(
        r_policy=float(gross),
        bars_held=bars_held,
        exit_code=int(exit_code),
        mae=float(mae),
        mfe=float(mfe),
        reason=str(reason),
    )


def _reason_to_exit_code(reason: str, r_policy: float) -> int:
    rr = str(reason or "")
    if rr in {"stop_loss", "early_invalidation"}:
        return 0
    if rr in {"take_profit", "tp"}:
        return 2
    if rr in {"trailing_stop", "giveback_exit"}:
        # Profitable trailing / giveback exits → TP; loss-making → SL.
        # A trailing stop that ends in the red means the hard SL was hit after
        # break-even ratcheted up but price reversed below entry — counts as SL
        # from a sizing/risk perspective.
        return 2 if float(r_policy) >= 0.0 else 0
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
    """ML rollout wrapper aligned with the ridge-sizer TP/SL utility family."""
    return _policy_rollout_sizer_aligned(
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
    """Compute ridge-sizer-aligned policy labels for provided event indices.

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
        out = policy_rollout_ml(
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
    """Build TP-vs-Rest labels (including timeouts).

    exit_code semantics: 0=SL, 1=TO, 2=TP.
    Returns (y_bin, mask_over_original_rows).
    """
    exit_code = np.asarray(exit_code, np.int32)
    # Include all valid outcomes in the dataset (aligns Hit Rate with Win Rate)
    mask = np.ones(len(exit_code), dtype=bool)
    y = (exit_code == 2).astype(np.int32)
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


@dataclass
class MetaMoveSelectionConfig:
    """Selection config for the binary move classifier."""

    min_roc_auc: float = 0.56
    min_pr_auc: float = 0.0
    min_balanced_accuracy: float = 0.0
    min_ic: float = 0.0
    top_frac: float = 0.10
    top_fracs: Tuple[float, ...] = (0.05, 0.10, 0.20)
    min_top_n: int = 50
    min_lift_vs_baseline: float = 0.0
    require_positive_top_lift: bool = True
    require_positive_base_rate: bool = True


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

    # Sanitize probabilities before ALL metric computations (including log_loss).
    p_pred = np.where(np.isfinite(p_pred), p_pred, 0.0)
    p_pred = np.clip(p_pred, 0.0, None)
    row_sum = p_pred.sum(axis=1, keepdims=True)
    safe = row_sum[:, 0] > 1e-12
    p_pred[safe] = p_pred[safe] / row_sum[safe]
    p_pred[~safe] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    p_pred = np.clip(p_pred, 1e-12, 1.0)
    p_pred = p_pred / np.clip(p_pred.sum(axis=1, keepdims=True), 1e-12, None)

    try:
        if y_true.ndim == 2:
            if np.any(tm):
                p_safe = np.clip(p_pred[tm], 1e-15, 1 - 1e-15)
                ll = float(-np.mean(np.sum(y_true[tm] * np.log(p_safe), axis=1)))
            else:
                ll = 999.0
        else:
            ll = float(log_loss(y_true[tm], p_pred[tm], labels=[0, 1, 2])) if np.any(tm) else 999.0
    except Exception as exc:
        tprint(f"Warning: failed to compute classifier log_loss after probability sanitation: {exc}")
        ll = 999.0
    passed_gate = ll <= float(cfg.max_logloss)

    # Dynamic utility weights from realized outcomes on this OOF vector.
    if bool(cfg.dynamic_utility_from_realized):
        def _class_mean(k: int, dflt: float) -> float:
            if y_true.ndim == 2:
                # Soft labels mapping: k_hard -> k_soft (0:SL->1, 1:TO->2, 2:TP->0)
                k_soft = 1 if k == 0 else (2 if k == 1 else 0)
                # Soft labels: weight realized utility by probability
                weights = y_true[:, k_soft] * tm.astype(float)
                w_sum = np.sum(weights)
                if w_sum > 1e-9:
                    v = float(np.sum(realized_u_policy * weights) / w_sum)
                    if np.isfinite(v):
                        return v
            else:
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

    if y_true.ndim == 2:
        # p_pred has columns [TP, SL, TO]. We need them as [SL, TO, TP] for expected_utility
        p_pred_mapped = np.column_stack([p_pred[:, 1], p_pred[:, 2], p_pred[:, 0]])
        U = expected_utility(p_pred_mapped, u_tp, u_to, u_sl)
    else:
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


def pick_meta_move_by_topq(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    realized_abs_return: np.ndarray,
    cfg: MetaMoveSelectionConfig,
    trade_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Binary move-head selection metrics and economic gate.

    The target is p_move = Pr(|net return| > k * vol_scale).
    Selection is based on ranking rows by p_move and checking whether the
    top slice shows higher realized absolute return and positive lift.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
        roc_curve,
    )
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
    realized_abs_return = np.asarray(realized_abs_return, dtype=float).reshape(-1)
    tm = np.ones(len(y_true), dtype=bool) if trade_mask is None else np.asarray(trade_mask, dtype=bool)
    tm = tm[: len(y_true)]
    valid = tm & np.isfinite(y_true) & np.isfinite(p_pred) & np.isfinite(realized_abs_return)

    if valid.sum() == 0:
        return {
            "n": 0.0,
            "logloss": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "balanced_accuracy_0p5": float("nan"),
            "balanced_accuracy_best": float("nan"),
            "best_threshold": float("nan"),
            "brier": float("nan"),
            "move_ic": float("nan"),
            "top_decile_absret_mean": float("nan"),
            "top_decile_lift": float("nan"),
            "top_decile_hit_rate": float("nan"),
            "base_rate": float("nan"),
            "passed_gate": 0.0,
            "passed_econ": 0.0,
            "selection_score": float("-inf"),
        }

    y = y_true[valid].astype(int)
    p = np.clip(p_pred[valid].astype(float), 1e-6, 1 - 1e-6)
    r = realized_abs_return[valid].astype(float)
    base_rate = float(np.mean(y))
    try:
        ll = float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        ll = float("nan")
    try:
        roc = float(roc_auc_score(y, p))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y, p))
    except Exception:
        pr = float("nan")
    try:
        brier = float(brier_score_loss(y, p))
    except Exception:
        brier = float("nan")

    pred_05 = (p >= 0.5).astype(int)
    bal_05 = float(balanced_accuracy_score(y, pred_05))
    try:
        fpr, tpr, thr = roc_curve(y, p)
        youden = tpr - fpr
        best_idx = int(np.argmax(youden))
        best_thr = float(thr[best_idx])
    except Exception:
        best_thr = 0.5
    pred_best = (p >= best_thr).astype(int)
    bal_best = float(balanced_accuracy_score(y, pred_best))

    rho = 0.0
    if np.std(p) > 1e-12 and np.std(r) > 1e-12:
        rho_val, _ = spearmanr(p, r)
        if np.isfinite(rho_val):
            rho = float(rho_val)

    k = max(1, int(np.ceil(float(cfg.top_frac) * len(p))))
    top_idx = np.argsort(-p)[:k]
    top_mean = float(np.mean(r[top_idx]))
    top_hit = float(np.mean(y[top_idx]))
    baseline_mean = float(np.mean(r))
    lift = top_mean - baseline_mean

    topq_metrics = {}
    for frac in cfg.top_fracs:
        kk = max(1, int(np.ceil(float(frac) * len(p))))
        idx = np.argsort(-p)[:kk]
        topq_metrics[f"top{int(round(frac*100)):02d}_absret_mean"] = float(np.mean(r[idx]))
        topq_metrics[f"top{int(round(frac*100)):02d}_lift"] = float(np.mean(r[idx]) - baseline_mean)
        topq_metrics[f"top{int(round(frac*100)):02d}_hit_rate"] = float(np.mean(y[idx]))

    passed_gate = (
        np.isfinite(roc)
        and np.isfinite(pr)
        and np.isfinite(bal_best)
        and roc >= float(cfg.min_roc_auc)
        and pr >= float(cfg.min_pr_auc)
        and bal_best >= float(cfg.min_balanced_accuracy)
        and rho >= float(cfg.min_ic)
    )
    passed_econ = bool(k >= int(cfg.min_top_n) and lift >= float(cfg.min_lift_vs_baseline))
    if bool(cfg.require_positive_top_lift):
        passed_econ = bool(passed_econ and top_mean > baseline_mean)
    if bool(cfg.require_positive_base_rate):
        passed_gate = bool(passed_gate and base_rate > 0.0)

    return {
        "n": float(len(p)),
        "logloss": ll,
        "roc_auc": roc,
        "pr_auc": pr,
        "balanced_accuracy_0p5": bal_05,
        "balanced_accuracy_best": bal_best,
        "best_threshold": best_thr,
        "brier": brier,
        "move_ic": rho,
        "top_decile_absret_mean": top_mean,
        "top_decile_lift": lift,
        "top_decile_hit_rate": top_hit,
        "base_rate": base_rate,
        "passed_gate": float(passed_gate),
        "passed_econ": float(passed_econ),
        "selection_score": float(lift),
        **topq_metrics,
    }
