"""Policy optimisation stage for extreme_price_movements.

This module consumes the already-selected best strategy (from simple_position_sizer),
keeps TP/SL fixed, runs offset generation first, then sequentially optimises richer
exit-policy parameters. The resulting params are persisted for holdout OOS replay.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.run_ridge_sizer import (
    load_meta_oof_predictions,
    load_trade_outcomes,
)
from extreme_price_movements.simple_offset_generator import (
    build_policy_path_state_bundle,
    run_simple_offset_generator_from_sizer,
)
from extreme_price_movements.utils import tprint

EPS = 1e-9
logger = logging.getLogger(__name__)


def _strategy_description(row: Dict[str, Any]) -> str:
    side = str(row.get("side", "") or "").strip().lower()
    source_target = str(row.get("source_target", "") or "").strip()
    source_horizon = row.get("source_horizon", np.nan)
    threshold_pct = float(row.get("threshold_pct", np.nan))
    selection_frac = float(row.get("selection_frac", np.nan))
    parts: List[str] = []
    if side:
        parts.append(side)
    if source_target:
        parts.append(source_target)
    if np.isfinite(source_horizon):
        parts.append(f"h{float(source_horizon):g}")
    if np.isfinite(threshold_pct):
        parts.append(f"thr={threshold_pct:.1f}%")
    if np.isfinite(selection_frac):
        parts.append(f"sel={selection_frac:.3f}")
    return " | ".join(parts) if parts else str(row.get("strategy_id", ""))


def _safe_log(x: float) -> float:
    return float(math.log(max(float(x), EPS)))


def _compute_strategy_D(row: Dict[str, Any]) -> float:
    pf = float(row.get("profit_factor", float("nan")))
    stability = float(row.get("stability", float("nan")))
    monthly_sortino = float(row.get("monthly_sortino", float("nan")))
    hit_rate = float(row.get("hit_rate", float("nan")))
    trades_per_day = float(row.get("trades_per_day", float("nan")))
    max_drawdown = float(row.get("max_drawdown", float("nan")))
    wallet_pnl = float(row.get("wallet_pnl", float("nan")))
    if not all(
        np.isfinite(v)
        for v in (pf, stability, monthly_sortino, hit_rate, trades_per_day, max_drawdown, wallet_pnl)
    ):
        return float("nan")
    if wallet_pnl <= 0.0:
        return float("nan")
    return (
        0.35 * _safe_log(pf)
        + 0.20 * stability
        + 0.15 * monthly_sortino
        + 0.10 * hit_rate
        + 0.10 * _safe_log(1.0 + trades_per_day)
        + 0.10 * (1.0 - max_drawdown / wallet_pnl)
    )


def _load_ridge_strategy_rows(data_root: str, run_id: str) -> Dict[str, Dict[str, Any]]:
    path = Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    rows = payload.get("strategies", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    by_strategy_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("strategy_id", "") or "")
        if not sid:
            continue
        by_strategy_id[sid] = dict(row)
    return by_strategy_id


def strategy_final_acceptation(
    data_root: str,
    run_id: str,
    strategy_rows: List[Dict[str, Any]],
    *,
    output_filename: str = "strategy_final_acceptation.json",
) -> Dict[str, Any]:
    """Filter policy candidates by hard gates and write the acceptance artifact."""
    ridge_rows = _load_ridge_strategy_rows(data_root, run_id)
    accepted: List[Dict[str, Any]] = []
    for row in strategy_rows:
        if not isinstance(row, dict):
            continue
        strategy_id = str(row.get("strategy_id", "") or "")
        source_row = ridge_rows.get(strategy_id, row)
        accepted_row = dict(source_row)
        accepted_row["strategy_id"] = strategy_id
        accepted_row["strategy_description"] = _strategy_description(source_row)
        accepted_row["D"] = _compute_strategy_D(source_row)
        pf = float(source_row.get("profit_factor", float("nan")))
        stability = float(source_row.get("stability", float("nan")))
        monthly_sortino = float(source_row.get("monthly_sortino", float("nan")))
        max_drawdown = float(source_row.get("max_drawdown", float("nan")))
        wallet_pnl = float(source_row.get("wallet_pnl", float("nan")))
        if not all(
            np.isfinite(v)
            for v in (pf, stability, monthly_sortino, max_drawdown, wallet_pnl)
        ):
            continue
        if (
            pf > 1.3
            and stability > 0.7
            and monthly_sortino > 1.0
            and wallet_pnl > 0.0
            and max_drawdown < wallet_pnl
        ):
            accepted.append(accepted_row)

    accepted.sort(
        key=lambda r: (
            float(r.get("D", float("-inf"))),
            float(r.get("wallet_pnl", float("-inf"))),
            float(r.get("profit_factor", float("-inf"))),
        ),
        reverse=True,
    )

    payload = {
        "schema_version": "v1",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "hard_gates": {
            "profit_factor_min": 1.3,
            "stability_min": 0.7,
            "monthly_sortino_min": 1.0,
            "max_drawdown_lt_wallet_pnl": True,
        },
        "strategies": [
            {
                "strategy_id": r.get("strategy_id", ""),
                "strategy_description": r.get("strategy_description", ""),
                "metrics": r,
                "n_trades": int(
                    r.get(
                        "trades_selected",
                        r.get("n_trades", r.get("trades_total", 0)),
                    )
                    or 0
                ),
            }
            for r in accepted
        ],
    }
    out_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / output_filename
    path.write_text(json.dumps(payload, indent=2, default=float))
    (Path(data_root) / "artifacts" / run_id / output_filename).write_text(
        json.dumps(payload, indent=2, default=float)
    )
    tprint(f"Wrote strategy acceptance gate output -> {path}")
    return payload


def _metric_score(
    rets: np.ndarray,
    cost_pct: float = 0.003,
    already_net: bool = False,
    executed_mask: Optional[np.ndarray] = None,
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
    if executed_mask is not None:
        executed_mask = np.asarray(executed_mask, dtype=bool)
        if executed_mask.shape[0] == len(rets):
            rets = rets[executed_mask]
    # Avoid double-counting costs if returns already include them
    if already_net:
        net_rets = rets.astype(np.float64)
    else:
        net_rets = rets.astype(np.float64) - float(cost_pct)
    downside = net_rets[net_rets < 0]
    ds_std = float(np.std(downside)) if len(downside) > 1 else 1e-6
    gross_win = float(np.sum(net_rets[net_rets > 0]))
    gross_loss = float(np.abs(np.sum(net_rets[net_rets < 0])))
    _, dd = _stable_equity_and_drawdown(net_rets)
    return {
        "net_pnl": float(np.sum(net_rets)),
        "sortino": float(np.mean(net_rets)) / ds_std,
        "hit_rate": float(np.mean(net_rets > 0)),
        "profit_factor": gross_win / gross_loss
        if gross_loss > EPS
        else float(gross_win),
        "max_drawdown": float(np.max(dd)) if len(dd) else 0.0,
        "n_trades": int(len(rets)),
    }


def _resolve_selection_score(outcomes: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Return the sizer ranking score for trade selection."""
    candidate_cols = ["sizer_score"]
    for col in candidate_cols:
        if col not in outcomes.columns:
            continue
        arr = np.asarray(outcomes[col].values, dtype=np.float32)
        if np.any(np.isfinite(arr)) and float(np.nanstd(arr)) > 1e-12:
            return arr, col
    return np.zeros(len(outcomes), dtype=np.float32), "flat_zero"


def _robust_fit(arr: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(arr, dtype=np.float64)
    tr = x[mask]
    if tr.size == 0:
        return 0.0, 1.0, -5.0, 5.0
    med = float(np.nanmedian(tr))
    q1 = float(np.nanpercentile(tr, 25))
    q3 = float(np.nanpercentile(tr, 75))
    iqr = max(q3 - q1, 1e-6)
    lo = float(np.nanpercentile((tr - med) / iqr, 1))
    hi = float(np.nanpercentile((tr - med) / iqr, 99))
    return med, iqr, lo, hi


def _robust_apply(
    arr: np.ndarray, params: Tuple[float, float, float, float]
) -> np.ndarray:
    med, iqr, lo, hi = params
    z = (np.asarray(arr, dtype=np.float64) - med) / max(iqr, 1e-6)
    return np.clip(z, lo, hi).astype(np.float32)


def _load_best_strategy(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load the post head-to-head winner from sizer artifacts.

    Preference order:
      1) explicit winner from ridge_sizer/head_to_head_comparison.json
      2) best_strategy_id from et/ridge strategy_params.json (higher net_pnl)
    """
    comparison_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "head_to_head_comparison.json"
    )
    if comparison_path.exists():
        try:
            rows = json.loads(comparison_path.read_text())
            et_winners = [r for r in rows if str(r.get("winner", "")).lower() == "et"]
            if et_winners:
                et_params = (
                    Path(data_root)
                    / "artifacts"
                    / run_id
                    / "et_sizer"
                    / "strategy_params.json"
                )
                if et_params.exists():
                    payload = json.loads(et_params.read_text())
                    sid = str(payload.get("best_strategy_id", ""))
                    bucket = (payload.get("buckets", {}) or {}).get(sid, {})
                    return {
                        "strategy_id": sid,
                        "threshold_pct": float(
                            bucket.get(
                                "threshold_pct", payload.get("best_threshold_pct", 90.0)
                            )
                        ),
                        "model": "et",
                        "tp_mult": float(bucket.get("tp_mult", 1.0)),
                        "sl_mult": float(bucket.get("sl_mult", 1.0)),
                    }
        except Exception:
            pass

    candidates = [
        Path(data_root) / "artifacts" / run_id / "et_sizer" / "strategy_params.json",
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json",
    ]
    best: Dict[str, Any] = {}
    best_pnl = -1e18
    for pth in candidates:
        if not pth.exists():
            continue
        payload = json.loads(pth.read_text())
        buckets = payload.get("buckets", {})
        sid = str(payload.get("best_strategy_id", ""))
        row = buckets.get(sid, {}) if isinstance(buckets, dict) else {}
        pnl = float(row.get("net_pnl", -1e18))
        if pnl > best_pnl:
            best_pnl = pnl
            best = {
                "strategy_id": sid,
                "threshold_pct": float(
                    row.get("threshold_pct", payload.get("best_threshold_pct", 90.0))
                ),
                "model": "et" if "et_sizer" in str(pth) else "ridge",
                "tp_mult": float(row.get("tp_mult", 1.0)),
                "sl_mult": float(row.get("sl_mult", 1.0)),
            }
    return best


def _load_strategy_candidates(data_root: str, run_id: str) -> List[Dict[str, Any]]:
    """Load all strategy rows from ridge/et sizer artifacts.

    When the same strategy_id appears in both artifacts, keep the row with the
    higher baseline net_pnl so policy optimisation starts from the stronger
    source model.
    """
    candidates: List[Dict[str, Any]] = []
    for model_source, rel_path in [
        ("ridge", "ridge_sizer/strategy_params.json"),
        ("et", "et_sizer/strategy_params.json"),
    ]:
        path = Path(data_root) / "artifacts" / run_id / rel_path
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        rows = payload.get("strategies", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("strategy_id", "") or "")
            if not sid:
                continue
            cand = dict(row)
            cand["strategy_id"] = sid
            cand["model"] = model_source
            cand["source_artifact"] = "et_sizer" if model_source == "et" else "ridge_sizer"
            candidates.append(cand)

    merged: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        sid = str(cand.get("strategy_id", ""))
        if not sid:
            continue
        prev = merged.get(sid)
        if prev is None:
            merged[sid] = cand
            continue
        prev_pnl = float(prev.get("net_pnl", float("-inf")))
        cand_pnl = float(cand.get("net_pnl", float("-inf")))
        if cand_pnl > prev_pnl:
            merged[sid] = cand
    out = list(merged.values())
    out.sort(
        key=lambda r: (
            float(r.get("net_pnl", float("-inf"))),
            float(r.get("profit_factor", float("-inf"))),
            float(r.get("hit_rate", float("-inf"))),
        ),
        reverse=True,
    )
    return out


def _load_sizer_oof_scores(data_root: str, run_id: str) -> pd.DataFrame:
    """Load the simple_position_sizer OOF score matrix."""
    path = Path(data_root) / "artifacts" / run_id / "oof" / "sizer_oof_all.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No sizer OOF score matrix found at {path}")
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        raise ValueError(f"sizer OOF score matrix {path} is missing timestamp/symbol")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(
        None
    )
    out["symbol"] = out["symbol"].astype(str)
    return out


def _attach_sizer_score(
    outcomes: pd.DataFrame, sizer_scores: pd.DataFrame, strategy_id: str
) -> pd.DataFrame:
    """Attach a strategy-specific score column from the sizer OOF matrix."""
    if strategy_id not in sizer_scores.columns:
        return outcomes
    score_df = sizer_scores[["timestamp", "symbol", strategy_id]].rename(
        columns={strategy_id: "sizer_score"}
    )
    out = outcomes.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(
        None
    )
    out["symbol"] = out["symbol"].astype(str)
    merged = out.merge(score_df, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    return merged


def resolve_optimised_selection_frac(
    *, data_root: str, run_id: str, selected: Dict[str, Any]
) -> float:
    """Return selection fraction from ranked threshold with optimised pnl.

    Uses the selected strategy threshold from `strategy_params.json` generated by
    simple_position_sizer (which stores the optimal ranked threshold). If the selected
    row has non-positive pnl, fallback to the highest positive-pnl threshold in the
    same model artifact.
    """
    model = str(selected.get("model", "ridge")).lower()
    params_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / ("et_sizer" if model == "et" else "ridge_sizer")
        / "strategy_params.json"
    )
    threshold_pct = float(selected.get("threshold_pct", 90.0))
    if params_path.exists():
        try:
            payload = json.loads(params_path.read_text())
            buckets = payload.get("buckets", {}) or {}
            sid = str(selected.get("strategy_id", ""))
            row = buckets.get(sid, {}) if isinstance(buckets, dict) else {}
            row_pnl = float(row.get("net_pnl", 0.0))
            threshold_pct = float(row.get("threshold_pct", threshold_pct))
            if row_pnl <= 0.0 and isinstance(buckets, dict):
                positive_rows = [
                    r for r in buckets.values() if float(r.get("net_pnl", -1e18)) > 0.0
                ]
                if positive_rows:
                    best_pos = max(
                        positive_rows, key=lambda r: float(r.get("net_pnl", -1e18))
                    )
                    threshold_pct = float(best_pos.get("threshold_pct", threshold_pct))
        except Exception:
            pass

    frac = max(0.01, min(1.0, 1.0 - threshold_pct / 100.0))
    return float(frac)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def build_replay_context(
    *,
    returns: np.ndarray,
    mfe_ret: np.ndarray,
    mae_ret: np.ndarray,
    delta_mfe: Optional[np.ndarray] = None,
    delta_mae: Optional[np.ndarray] = None,
    bars_since_entry: np.ndarray,
    barrier_pct: np.ndarray,
    confidence: np.ndarray,
    trend: Optional[np.ndarray] = None,
    choppiness: Optional[np.ndarray] = None,
    asym: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build canonical replay context shared by optimiser and OOS evaluation."""
    rets = np.asarray(returns, dtype=np.float32)
    mfe = np.asarray(mfe_ret, dtype=np.float32)
    mae = np.asarray(mae_ret, dtype=np.float32)
    bars = np.maximum(1, np.asarray(bars_since_entry, dtype=np.int32))
    barr = np.maximum(np.asarray(barrier_pct, dtype=np.float32), 1e-4)
    conf = np.asarray(confidence, dtype=np.float32)
    delta_mfe_arr = (
        np.asarray(delta_mfe, dtype=np.float32)
        if delta_mfe is not None
        else np.maximum(mfe / np.maximum(bars, 1), 1e-6)
    )
    delta_mae_arr = (
        np.asarray(delta_mae, dtype=np.float32)
        if delta_mae is not None
        else np.maximum(mae / np.maximum(bars, 1), 0.0)
    )

    ae_vel = delta_mae_arr / np.maximum(bars, 1)
    pressure = np.clip(delta_mae_arr / (delta_mfe_arr + 1e-6), 0.0, 20.0)
    path_quality = delta_mfe_arr - delta_mae_arr
    progress_per_bar = np.maximum(rets, 0.0) / np.maximum(barr * bars, 1e-4)

    asym_raw = np.log((np.maximum(delta_mfe_arr, 0.0) + 1e-6) / (np.maximum(delta_mae_arr, 0.0) + 1e-6))
    if asym is None:
        asym_vals = asym_raw
    else:
        asym_vals = np.asarray(asym, dtype=np.float32)

    return {
        "mfe_ret": mfe,
        "mae_ret": mae,
        "bars_since_entry": bars,
        "barrier_pct": barr,
        "confidence": conf,
        "AE_vel": ae_vel.astype(np.float32),
        "pressure": pressure.astype(np.float32),
        "path_quality": path_quality.astype(np.float32),
        "progress_per_bar": progress_per_bar.astype(np.float32),
        "trend": np.asarray(
            np.zeros_like(rets) if trend is None else trend, dtype=np.float32
        ),
        "choppiness": np.asarray(
            np.zeros_like(rets) if choppiness is None else choppiness, dtype=np.float32
        ),
        "asym": np.asarray(asym_vals, dtype=np.float32),
        "asym_raw": asym_raw.astype(np.float32),
    }


def _pack_future_path_matrices(
    outcomes: pd.DataFrame, selected_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """Pack selected 15m futures into dense padded matrices for replay."""
    if len(selected_idx) == 0:
        return {}
    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in outcomes.columns for col in required):
        return {}

    idx = np.asarray(selected_idx, dtype=np.int64)
    raw_paths = {col: outcomes[col].values[idx] for col in required}
    lengths = np.zeros(len(idx), dtype=np.int32)
    for i, arr in enumerate(raw_paths["future_opens"]):
        if arr is None:
            lengths[i] = 0
        else:
            lengths[i] = int(np.asarray(arr).shape[0])
    max_len = int(lengths.max()) if len(lengths) else 0
    if max_len <= 0:
        return {}

    packed: Dict[str, np.ndarray] = {}
    for col in required:
        mat = np.full((len(idx), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(raw_paths[col]):
            if arr is None:
                continue
            arr_np = np.asarray(arr, dtype=np.float32).reshape(-1)
            if arr_np.size == 0:
                continue
            n_copy = min(arr_np.size, max_len)
            mat[i, :n_copy] = arr_np[:n_copy]
        packed[col] = mat

    if "entry_price" in outcomes.columns:
        entry_price = np.asarray(outcomes["entry_price"].values[idx], dtype=np.float32)
    else:
        entry_price = packed["future_opens"][:, 0].copy()
        entry_price = np.where(
            np.isfinite(entry_price) & (entry_price > 0.0), entry_price, 1.0
        ).astype(np.float32)

    if "is_long" in outcomes.columns:
        is_long = np.asarray(outcomes["is_long"].values[idx], dtype=bool)
    elif "side" in outcomes.columns:
        side = np.asarray(outcomes["side"].values[idx], dtype=str)
        is_long = np.char.lower(side.astype("U")) != "short"
    else:
        is_long = np.ones(len(idx), dtype=bool)

    packed["future_lengths"] = lengths
    packed["entry_price"] = entry_price.astype(np.float32)
    packed["is_long"] = np.asarray(is_long, dtype=bool)
    return packed


def _simulate_baseline_tpsl_from_paths(
    context: Dict[str, np.ndarray],
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
) -> Optional[np.ndarray]:
    """Compute baseline returns using simple TP/SL logic from raw price paths.

    This ensures fair comparison: both baseline and policy optimization
    start from the same raw paths, with baseline using simple TP/SL exit.
    """
    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in context for col in required):
        return None

    future_opens = np.asarray(context["future_opens"], dtype=np.float32)
    future_highs = np.asarray(context["future_highs"], dtype=np.float32)
    future_lows = np.asarray(context["future_lows"], dtype=np.float32)
    future_closes = np.asarray(context["future_closes"], dtype=np.float32)
    future_lengths = np.asarray(
        context.get("future_lengths", np.full(len(future_opens), future_opens.shape[1])),
        dtype=np.int32,
    )

    if future_opens.ndim != 2:
        return None

    n_trades, max_bars = future_opens.shape
    entry = np.asarray(
        context.get("entry_price", future_opens[:, 0]), dtype=np.float32
    )
    is_long = np.asarray(context.get("is_long", np.ones(n_trades, dtype=bool)), dtype=bool)
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    barrier = np.maximum(
        np.asarray(context.get("barrier_pct", np.full(n_trades, 0.02)), dtype=np.float32),
        1e-4,
    )

    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier

    exited = np.zeros(n_trades, dtype=bool)
    exit_rets = np.zeros(n_trades, dtype=np.float32)

    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)
        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        ent = entry[idx]
        side_a = side[idx]
        bar_open = future_opens[idx, bar]
        bar_high = future_highs[idx, bar]
        bar_low = future_lows[idx, bar]
        bar_close = future_closes[idx, bar]

        valid = (
            np.isfinite(ent)
            & np.isfinite(bar_open)
            & np.isfinite(bar_high)
            & np.isfinite(bar_low)
            & np.isfinite(bar_close)
            & (ent > 0.0)
        )
        if not np.any(valid):
            continue

        idx = idx[valid]
        ent = ent[valid]
        side_a = side_a[valid]
        bar_high = bar_high[valid]
        bar_low = bar_low[valid]
        bar_close = bar_close[valid]

        tp_dist_a = tp_dist[idx]
        sl_dist_a = sl_dist[idx]

        # Compute returns at each price level
        high_ret = side_a * (bar_high / ent - 1.0)
        low_ret = side_a * (bar_low / ent - 1.0)
        close_ret = side_a * (bar_close / ent - 1.0)
        best_ret = np.maximum(high_ret, low_ret)
        worst_ret = np.minimum(high_ret, low_ret)

        # Check if TP or SL hit
        tp_hit = best_ret >= tp_dist_a
        sl_hit = worst_ret <= -sl_dist_a

        # Exit at TP/SL or hold
        bar_exit = np.full(len(idx), np.nan, dtype=np.float32)
        bar_exit = np.where(sl_hit, -sl_dist_a, bar_exit)
        bar_exit = np.where(tp_hit & np.isnan(bar_exit), tp_dist_a, bar_exit)

        # If no hit, use close for last bar
        is_last = bar >= future_lengths[idx] - 1
        bar_exit = np.where(np.isnan(bar_exit) & is_last, close_ret, bar_exit)

        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True

    # For any not exited, compute final return from last available bar
    not_exited = ~exited
    if np.any(not_exited):
        for i in np.flatnonzero(not_exited):
            last_bar = min(future_lengths[i] - 1, max_bars - 1)
            if last_bar >= 0 and np.isfinite(future_closes[i, last_bar]):
                exit_rets[i] = side[i] * (future_closes[i, last_bar] / entry[i] - 1.0)

    return exit_rets.astype(np.float32)


def _simulate_barwise_path_policy(
    returns: np.ndarray, context: Dict[str, np.ndarray], params: Dict[str, Any]
) -> Optional[np.ndarray]:
    """Replay the policy using actual 15m OHLC paths and causal bar-by-bar logic."""
    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in context for col in required):
        return None

    future_opens = np.asarray(context["future_opens"], dtype=np.float32)
    future_highs = np.asarray(context["future_highs"], dtype=np.float32)
    future_lows = np.asarray(context["future_lows"], dtype=np.float32)
    future_closes = np.asarray(context["future_closes"], dtype=np.float32)
    future_lengths = np.asarray(
        context.get("future_lengths", np.full(len(returns), future_opens.shape[1])),
        dtype=np.int32,
    )
    if future_opens.ndim != 2 or future_highs.shape != future_opens.shape:
        return None
    if future_lows.shape != future_opens.shape or future_closes.shape != future_opens.shape:
        return None

    n_trades, max_bars = future_opens.shape
    if n_trades != len(returns):
        return None

    entry = np.asarray(
        context.get("entry_price", future_opens[:, 0]), dtype=np.float32
    )
    is_long = np.asarray(context.get("is_long", np.ones(n_trades, dtype=bool)), dtype=bool)
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    barrier = np.maximum(
        np.asarray(context.get("barrier_pct", np.full(n_trades, 0.02)), dtype=np.float32),
        1e-4,
    )
    conf = np.asarray(context.get("confidence", np.zeros(n_trades)), dtype=np.float32)
    p_tp = np.asarray(context.get("p_tp", np.full(n_trades, np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(n_trades, np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0.0)

    trend = np.asarray(context.get("trend", np.zeros(n_trades)), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(n_trades, 0.5)), dtype=np.float32)
    choppy = np.asarray(context.get("choppiness", np.zeros(n_trades)), dtype=np.float32)
    s = 1.0 * trend + 0.6 * asym - 0.9 * choppy
    m_raw = 0.7 + 0.6 * _sigmoid(s)
    m = np.clip(
        m_raw,
        float(params.get("multiplier_band_min", 0.8)),
        float(params.get("multiplier_band_max", 1.2)),
    )

    tp_mult = float(params.get("tp_mult", 1.0))
    sl_mult = float(params.get("sl_mult", 1.0))
    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier
    a1 = float(params.get("a1", 0.5))
    a2 = float(params.get("a2", 1.0 - a1))
    b1 = float(params.get("b1", 0.5))
    b2 = float(params.get("b2", 1.0 - b1))
    theta_fail = float(params.get("theta_fail", 0.0))
    theta_path = float(params.get("theta_path", 0.0))
    d_path = int(params.get("d_path", 1))
    k_early = int(params.get("K_early", 3))
    tp_gate = float(params.get("tp_conf_threshold", 0.5))
    sl_gate = float(params.get("sl_early_exit_threshold", 0.4))
    progress_threshold = float(params.get("progress_threshold", 0.0))
    trail_activation_atr = float(params.get("trail_activation_atr", 1.0))
    trail_giveback_atr = float(params.get("trail_giveback_atr", 0.5))

    rets = np.asarray(returns, dtype=np.float32).copy()
    exited = np.zeros(n_trades, dtype=bool)
    exit_rets = rets.copy()
    mfe = np.zeros(n_trades, dtype=np.float32)
    mae = np.zeros(n_trades, dtype=np.float32)
    prev_mfe = np.zeros(n_trades, dtype=np.float32)
    prev_mae = np.zeros(n_trades, dtype=np.float32)

    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)
        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        ent = entry[idx]
        side_a = side[idx]
        bar_open = future_opens[idx, bar]
        bar_high = future_highs[idx, bar]
        bar_low = future_lows[idx, bar]
        bar_close = future_closes[idx, bar]

        valid = (
            np.isfinite(ent)
            & np.isfinite(bar_open)
            & np.isfinite(bar_high)
            & np.isfinite(bar_low)
            & np.isfinite(bar_close)
            & (ent > 0.0)
        )
        if not np.any(valid):
            continue

        idx = idx[valid]
        ent = ent[valid]
        side_a = side_a[valid]
        bar_open = bar_open[valid]
        bar_high = bar_high[valid]
        bar_low = bar_low[valid]
        bar_close = bar_close[valid]
        barrier_a = barrier[idx]
        tp_dist_a = tp_dist[idx]
        sl_dist_a = sl_dist[idx]
        m_a = m[idx]

        open_ret = side_a * (bar_open / ent - 1.0)
        high_ret = side_a * (bar_high / ent - 1.0)
        low_ret = side_a * (bar_low / ent - 1.0)
        close_ret = side_a * (bar_close / ent - 1.0)
        best_ret = np.maximum(high_ret, low_ret)
        worst_ret = np.minimum(high_ret, low_ret)

        mfe[idx] = np.maximum(mfe[idx], np.maximum(best_ret, 0.0))
        mae[idx] = np.maximum(mae[idx], np.maximum(-worst_ret, 0.0))
        delta_mfe = np.maximum(mfe[idx] - prev_mfe[idx], 0.0)
        delta_mae = np.maximum(mae[idx] - prev_mae[idx], 0.0)
        prev_mfe[idx] = mfe[idx]
        prev_mae[idx] = mae[idx]

        ae_vel = delta_mae / float(bar + 1)
        pressure = np.clip(delta_mae / (delta_mfe + 1e-6), 0.0, 20.0)
        path_quality = delta_mfe - delta_mae
        progress = np.maximum(close_ret, 0.0) / np.maximum(
            barrier_a * float(bar + 1), 1e-4
        )
        fail_scale = 1.0 - 0.5 * (m_a - 1.0)
        s_fail = (a1 * ae_vel + a2 * pressure) * fail_scale
        s_path = b1 * path_quality + b2 * progress

        tp_price = ent * (1.0 + side_a * tp_dist_a)
        sl_price = ent * (1.0 - side_a * sl_dist_a)

        trail_active = mfe[idx] >= trail_activation_atr * barrier_a
        trail_floor = mfe[idx] - trail_giveback_atr * barrier_a
        trail_price = ent * (1.0 + side_a * trail_floor)

        open_tp = open_ret >= tp_dist_a
        open_sl = open_ret <= -sl_dist_a
        open_trail = trail_active & (open_ret <= trail_floor)

        tp_hit = best_ret >= tp_dist_a
        sl_hit = worst_ret <= -sl_dist_a
        trail_hit = trail_active & (worst_ret <= trail_floor)

        bar_exit = np.full(len(idx), np.nan, dtype=np.float32)
        bar_exit = np.where(open_sl, -sl_dist_a, bar_exit)
        bar_exit = np.where(open_trail & np.isnan(bar_exit), trail_floor, bar_exit)
        bar_exit = np.where(open_tp & np.isnan(bar_exit), tp_dist_a, bar_exit)
        hard_hit = sl_hit | trail_hit | tp_hit | open_sl | open_trail | open_tp
        bar_exit = np.where(
            np.isnan(bar_exit),
            np.where(sl_hit, -sl_dist_a, np.where(trail_hit, trail_floor, np.where(tp_hit, tp_dist_a, np.nan))),
            bar_exit,
        )

        # MFE protection: don't trigger fail_exit if trade has made significant progress toward TP
        # This prevents cutting winning trades during temporary pullbacks
        mfe_progress = mfe[idx] / np.maximum(tp_dist_a, 1e-6)
        has_made_progress = mfe_progress >= 0.25  # Has made 25% of TP distance

        fail_exit = np.zeros(len(idx), dtype=bool)
        # Never trigger fail_exit at bar 0 (first bar) - give trade time to develop
        if bar >= 1:
            if has_barrier_proba:
                p_sl_safe = np.where(np.isfinite(p_sl[idx]), p_sl[idx], 0.5)
                # Allow fail_exit only if trade hasn't made significant progress
                fail_exit = (bar + 1 <= k_early) & (p_sl_safe >= sl_gate) & (~has_made_progress)
            else:
                # Scale theta_fail threshold by progress - harder to trigger as MFE increases
                progress_factor = np.where(has_made_progress, 2.0, 1.0)  # 2x harder to trigger if progressed
                fail_exit = (bar + 1 <= k_early) & (s_fail > theta_fail * progress_factor)


        path_exit = (
            (bar + 1 >= max(3, d_path))
            & (progress >= progress_threshold)
            & (s_path < theta_path)
        )

        discretionary_exit = ~(hard_hit) & (fail_exit | path_exit)
        if np.any(discretionary_exit):
            floor_ret = -0.25 * sl_dist_a
            discretionary_ret = np.where(
                close_ret > 0.0, close_ret, np.minimum(close_ret, floor_ret)
            )
            bar_exit = np.where(
                discretionary_exit,
                discretionary_ret,
                bar_exit,
            )

        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True

    return exit_rets.astype(np.float32)


def replay_exit_policy(
    returns: np.ndarray, context: Dict[str, np.ndarray], params: Dict[str, Any]
) -> np.ndarray:
    """Shared replay logic for optimisation and OOS evaluation."""
    rets = np.asarray(returns, dtype=np.float32).copy()
    barwise_rets = _simulate_barwise_path_policy(rets, context, params)
    if barwise_rets is not None:
        return barwise_rets
    mfe = np.asarray(context.get("mfe_ret", np.abs(rets)), dtype=np.float32)
    mae = np.asarray(
        context.get("mae_ret", np.abs(np.minimum(rets, 0.0))), dtype=np.float32
    )
    bars = np.maximum(
        1,
        np.asarray(
            context.get("bars_since_entry", np.full(len(rets), 4)), dtype=np.int32
        ),
    )
    conf = np.asarray(context.get("confidence", np.zeros(len(rets))), dtype=np.float32)
    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(len(rets), 0.02)), dtype=np.float32
        ),
        1e-4,
    )

    tp_mult = float(params.get("tp_mult", 1.0))
    sl_mult = float(params.get("sl_mult", 1.0))
    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier

    ae_vel = np.asarray(
        context.get("AE_vel", mae / np.maximum(bars, 1)), dtype=np.float32
    )
    # NOTE: pressure uses cumulative formula here for backward compatibility,
    # but the policy_optimiser overwrites this with delta-based pressure from path_bundle.
    # Ensure OOS replay uses path_bundle-based context to get consistent delta-pressure.
    pressure = np.asarray(context.get("pressure", mae / (mfe + EPS)), dtype=np.float32)
    path_quality = np.asarray(context.get("path_quality", mfe - mae), dtype=np.float32)
    progress = np.asarray(
        context.get(
            "progress_per_bar", np.maximum(rets, 0.0) / np.maximum(bars * barrier, 1e-4)
        ),
        dtype=np.float32,
    )

    a1 = float(params.get("a1", 0.5))
    a2 = float(params.get("a2", 1.0 - a1))
    b1 = float(params.get("b1", 0.5))
    b2 = float(params.get("b2", 1.0 - b1))
    theta_fail = float(params.get("theta_fail", 0.0))
    theta_path = float(params.get("theta_path", 0.0))
    d_path = int(params.get("d_path", 1))
    k_early = int(params.get("K_early", 3))

    p_tp = np.asarray(context.get("p_tp", np.full(len(rets), np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(len(rets), np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0)

    trend = np.asarray(context.get("trend", np.zeros(len(rets))), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(len(rets), 0.5)), dtype=np.float32)
    choppy = np.asarray(
        context.get("choppiness", np.zeros(len(rets))), dtype=np.float32
    )
    s = 1.0 * trend + 0.6 * asym - 0.9 * choppy
    m_raw = 0.7 + 0.6 * _sigmoid(s)
    m = np.clip(
        m_raw,
        float(params.get("multiplier_band_min", 0.8)),
        float(params.get("multiplier_band_max", 1.2)),
    )

    fail_scale = 1.0 - 0.5 * (m - 1.0)
    s_fail = (a1 * ae_vel + a2 * pressure) * fail_scale
    s_path = b1 * path_quality + b2 * progress

    tp_gate = float(params.get("tp_conf_threshold", 0.5))
    sl_gate = float(params.get("sl_early_exit_threshold", 0.4))

    # MFE protection: don't trigger fail_exit if trade has made significant progress toward TP
    mfe_progress = mfe / np.maximum(tp_dist, 1e-6)
    has_made_progress = mfe_progress >= 0.25  # Has made 25% of TP distance

    if has_barrier_proba:
        p_sl_safe = np.where(np.isfinite(p_sl), p_sl, 0.5)
        fail_exit = (bars <= k_early) & (p_sl_safe >= sl_gate) & (~has_made_progress)
    else:
        # Scale theta_fail threshold by progress - harder to trigger as MFE increases
        progress_factor = np.where(has_made_progress, 2.0, 1.0)
        fail_exit = (bars <= k_early) & (s_fail > theta_fail * progress_factor)

    path_exit = (
        (bars >= max(3, d_path))
        & (progress >= float(params.get("progress_threshold", 0.0)))
        & (s_path < theta_path)
    )
    floor_ret = -0.25 * sl_dist
    disc_ret = np.where(rets > 0.0, rets, np.minimum(rets, floor_ret))
    rets = np.where(fail_exit | path_exit, disc_ret, rets)

    mfe_norm = mfe / np.maximum(tp_dist, 1e-4)
    c_start = float(params.get("compression_start", 0.5))
    c_full_raw = float(params.get("compression_full", 1.0))
    if c_full_raw <= c_start:
        import warnings

        warnings.warn(
            f"compression_full ({c_full_raw}) <= compression_start ({c_start}). "
            f"Adjusting to compression_start + 1e-6. Review your parameter grid.",
            UserWarning,
        )
    c_full = max(c_start + 1e-6, c_full_raw)
    c_max = float(params.get("compression_max_fraction", 0.5))
    c_alpha = np.clip((mfe_norm - c_start) / (c_full - c_start), 0.0, 1.0) * c_max
    sl_eff = sl_dist * (1.0 - c_alpha)
    rets = np.maximum(rets, -sl_eff)

    trail_act = float(params.get("trail_activation_atr", 1.0)) * (1.0 + 0.8 * (m - 1.0))
    trail_gb = float(params.get("trail_giveback_atr", 0.5)) * (1.0 + 0.8 * (m - 1.0))
    trail_on = mfe >= trail_act * barrier
    trail_floor = mfe - trail_gb * barrier

    if has_barrier_proba:
        p_tp_safe = np.where(np.isfinite(p_tp), p_tp, 0.5)
        high_cont = p_tp_safe >= tp_gate
    else:
        cont_thr = float(params.get("continuation_conf_threshold", 0.5))
        high_cont = conf >= cont_thr

    with_tp = np.minimum(np.maximum(rets, trail_floor), tp_dist)
    no_tp = np.maximum(rets, trail_floor)
    rets = np.where(trail_on & high_cont, no_tp, np.where(trail_on, with_tp, rets))

    sl_scale = 1.0 + 0.4 * (m - 1.0)
    tp_scale = 1.0 + 1.0 * (m - 1.0)
    rets = np.clip(rets, -sl_dist * sl_scale, tp_dist * tp_scale)
    return rets.astype(np.float32)


def _sequential_optimise(
    base_returns: np.ndarray,
    context: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    fixed: Dict[str, Any],
    cost_pct: float,
    executed_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    params = dict(fixed)
    search_plan: List[Tuple[str, List[Any]]] = [
        ("theta_fail", np.linspace(-1.0, 1.5, 11).tolist()),
        ("theta_path", np.linspace(-1.5, 1.0, 11).tolist()),
        ("K_early", [2, 3, 4, 5]),
        ("d_path", [1, 2, 3]),
        ("progress_threshold", [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]),
        ("tp_conf_threshold", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("sl_early_exit_threshold", [0.2, 0.3, 0.4, 0.5]),
        ("compression_start", [0.30, 0.50, 0.70]),
        ("compression_full", [0.80, 1.00, 1.20]),
        ("compression_max_fraction", [0.20, 0.40, 0.60]),
        ("trail_activation_atr", [0.5, 0.8, 1.0, 1.2]),
        ("trail_giveback_atr", [0.2, 0.3, 0.4, 0.5]),
        ("continuation_conf_threshold", [0.4, 0.5, 0.6, 0.7]),
        ("multiplier_band", [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15)]),
        ("a1", np.linspace(0.0, 1.0, 11).tolist()),
        ("b1", np.linspace(0.0, 1.0, 11).tolist()),
    ]

    # Check if future paths are available for fair baseline computation
    has_future_paths_ctx = all(
        col in context for col in ("future_opens", "future_highs", "future_lows", "future_closes")
    )
    fixed_tp_mult = float(fixed.get("tp_mult", 1.0))
    fixed_sl_mult = float(fixed.get("sl_mult", 1.0))

    # Track best validation performance and corresponding params
    best_val_metric = -1e18
    best_val_params = dict(params)
    param_history: List[Tuple[str, Any, float, float]] = []  # (param_name, value, train_pnl, val_pnl)

    for name, grid in search_plan:
        best_train_metric = -1e18
        best_val_for_param: Any = grid[0]
        val_scores: List[Tuple[float, Any]] = []

        for cand in grid:
            trial = dict(params)
            if name == "multiplier_band":
                trial["multiplier_band_min"], trial["multiplier_band_max"] = cand
            elif name == "a1":
                trial["a1"] = float(cand)
                trial["a2"] = 1.0 - float(cand)
            elif name == "b1":
                trial["b1"] = float(cand)
                trial["b2"] = 1.0 - float(cand)
            else:
                trial[name] = cand
            rets = replay_exit_policy(base_returns, context, trial)
            train_score = _metric_score(
                rets[train_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(executed_mask[train_mask] if executed_mask is not None else None),
            )["net_pnl"]
            val_score = _metric_score(
                rets[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(executed_mask[val_mask] if executed_mask is not None else None),
            )["net_pnl"]
            val_scores.append((val_score, cand))
            if train_score > best_train_metric:
                best_train_metric = train_score
                best_val_for_param = cand

        # Pick value that maximizes validation score (not just training)
        val_scores.sort(reverse=True)
        best_val_for_param = val_scores[0][1]

        if name == "multiplier_band":
            params["multiplier_band_min"], params["multiplier_band_max"] = best_val_for_param
        elif name == "a1":
            params["a1"] = float(best_val_for_param)
            params["a2"] = 1.0 - float(best_val_for_param)
        elif name == "b1":
            params["b1"] = float(best_val_for_param)
            params["b2"] = 1.0 - float(best_val_for_param)
        else:
            params[name] = best_val_for_param

        # Track if these params are best on validation so far
        current_rets = replay_exit_policy(base_returns, context, params)
        current_train_metric = _metric_score(
            current_rets[train_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(executed_mask[train_mask] if executed_mask is not None else None),
        )["net_pnl"]
        current_val_metric = _metric_score(
            current_rets[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(executed_mask[val_mask] if executed_mask is not None else None),
        )["net_pnl"]
        param_history.append((name, best_val_for_param, current_train_metric, current_val_metric))
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_val_params = dict(params)

    # Compute fair baseline from paths when available
    if has_future_paths_ctx:
        baseline_rets_raw = _simulate_baseline_tpsl_from_paths(
            context,
            tp_mult=fixed_tp_mult,
            sl_mult=fixed_sl_mult,
        )
        if baseline_rets_raw is not None:
            params["metrics_baseline"] = _metric_score(
                baseline_rets_raw[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
            )
        else:
            params["metrics_baseline"] = _metric_score(
                base_returns[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
            )
    else:
        params["metrics_baseline"] = _metric_score(
            base_returns[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
        )
    # Use the params that achieved best validation performance, not just final accumulated
    # This prevents overfitting to training set during sequential optimization
    final_rets = replay_exit_policy(base_returns, context, best_val_params)
    best_val_params["metrics_final"] = _metric_score(
        final_rets[val_mask],
        cost_pct=cost_pct,
        already_net=False,
        executed_mask=(
            executed_mask[val_mask] if executed_mask is not None else None
        ),
    )
    best_val_params["_param_history_"] = param_history  # For diagnostics
    return best_val_params


def run_policy_optimisation(
    data_root: str,
    run_id: str,
    sizer_results: Optional[Dict[str, Any]] = None,
    holdout_frac: float = 0.30,
    cost_pct: float = 0.003,
    use_offset_optimiser: bool = False,
) -> Dict[str, Any]:
    tprint("POLICY OPTIMISER START")
    selected_candidates = _load_strategy_candidates(data_root, run_id)
    if not selected_candidates:
        selected = _load_best_strategy(data_root, run_id)
        if selected.get("strategy_id"):
            selected_candidates = [selected]
    if not selected_candidates:
        tprint("No selected strategy found; skipping policy optimisation.")
        return {}

    meta_oofs = load_meta_oof_predictions(data_root, run_id)
    sizer_scores = _load_sizer_oof_scores(data_root, run_id)
    strategy_rows: List[Dict[str, Any]] = []
    for selected in selected_candidates:
        key = next((k for k in meta_oofs.keys() if selected["strategy_id"] in k), None)
        if key is None:
            scored_keys = []
            for mk, mdf in meta_oofs.items():
                if hasattr(mdf, "columns") and "oof_u_hat" in mdf.columns:
                    scored_keys.append(
                        (
                            float(
                                np.nanmean(np.asarray(mdf["oof_u_hat"], dtype=np.float32))
                            ),
                            mk,
                        )
                    )
            if scored_keys:
                scored_keys.sort(reverse=True)
                key = scored_keys[0][1]
                tprint(
                    f"Strategy {selected['strategy_id']} not found in meta OOF; "
                    f"falling back to best available bucket {key}."
                )
            else:
                key = next(iter(meta_oofs.keys()), None)
                if key is None:
                    tprint("No meta OOF buckets available; skipping policy optimisation.")
                    return {}

        outcomes = load_trade_outcomes(data_root, run_id, meta_oofs[key])
        if outcomes.empty:
            continue
        outcomes = _attach_sizer_score(outcomes, sizer_scores, selected["strategy_id"])
        if "sizer_score" not in outcomes.columns:
            raise RuntimeError(
                f"Could not attach simple_position_sizer scores for {selected['strategy_id']}"
            )

        conf, conf_name = _resolve_selection_score(outcomes)
        frac = resolve_optimised_selection_frac(
            data_root=data_root,
            run_id=run_id,
            selected=selected,
        )
        k = max(1, int(len(conf) * frac))
        selected_idx = np.argsort(conf, kind="stable")[-k:]

        sizer_stub = {
            "best_simple_score_": conf,
            "best_simple_score_name_": "Ridge_Head_Sizer",
            "ridge_profit_proxy_table_": pd.DataFrame(
                [
                    {
                        "selection_frac": frac,
                        "wallet_min": 0.05,
                        "wallet_max": 0.15,
                        "sizing_mode": "linear",
                        "is_optimal": True,
                    }
                ]
            ),
            "opt_rets_": np.asarray(
                outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
                dtype=np.float32,
            )[selected_idx],
            "opt_ts_": np.asarray(
                outcomes.get("timestamp", pd.Series(np.arange(len(outcomes)))).values
            )[selected_idx],
        }

        path_bundle = build_policy_path_state_bundle(outcomes, selected_idx=selected_idx)
        path_matrices = _pack_future_path_matrices(outcomes, selected_idx=selected_idx)

        _label_tp_sl = float(np.nanmedian(path_bundle.get("label_tp_sl_ratio", np.array([3.0]))))
        _has_label_policy = bool(path_bundle.get("has_label_policy", 0.0) > 0.5)

        base_rets = np.asarray(
            outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
            dtype=np.float32,
        )[selected_idx]
        executed_mask = np.ones(len(base_rets), dtype=bool)

        if use_offset_optimiser:
            offset_result = run_simple_offset_generator_from_sizer(
                sizer_results=sizer_stub, trade_outcomes=outcomes, cost_pct=cost_pct
            )
            base_rets = np.asarray(
                outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
                dtype=np.float32,
            )[offset_result.get("above_threshold_idx")]
            executed_mask = np.asarray(
                offset_result.get("executed", np.ones(len(base_rets), dtype=bool)),
                dtype=bool,
            )

        ts = pd.to_datetime(
            np.asarray(path_bundle.get("timestamps", np.arange(len(base_rets)))),
            utc=True,
            errors="coerce",
        )
        order = np.argsort(ts.view("int64"))
        n = len(base_rets)
        n_val = max(1, int(n * holdout_frac))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[order[-n_val:]] = True
        train_mask = ~val_mask

        ae_fit = _robust_fit(path_bundle["AE_vel"], train_mask)
        pr_fit = _robust_fit(path_bundle["pressure"], train_mask)
        pq_fit = _robust_fit(path_bundle["path_quality"], train_mask)
        pg_fit = _robust_fit(path_bundle["progress_per_bar"], train_mask)
        tr_fit = _robust_fit(path_bundle["trend"], train_mask)
        as_fit = _robust_fit(path_bundle["asym_raw"], train_mask)
        ch_fit = _robust_fit(path_bundle["choppiness"], train_mask)

        asym_z = _robust_apply(path_bundle["asym_raw"], as_fit)
        asym_01 = (asym_z - float(np.nanmin(asym_z[train_mask]))) / (
            float(np.nanmax(asym_z[train_mask]) - np.nanmin(asym_z[train_mask])) + EPS
        )
        context = build_replay_context(
            returns=base_rets,
            mfe_ret=path_bundle["mfe_ret"],
            mae_ret=path_bundle["mae_ret"],
            delta_mfe=path_bundle.get("delta_mfe"),
            delta_mae=path_bundle.get("delta_mae"),
            bars_since_entry=path_bundle["bars_since_entry"],
            barrier_pct=path_bundle["barrier_pct"],
            confidence=path_bundle["confidence"],
            trend=_robust_apply(path_bundle["trend"], tr_fit),
            choppiness=_robust_apply(path_bundle["choppiness"], ch_fit),
            asym=np.clip(asym_01, 0.0, 1.0).astype(np.float32),
        )
        context["AE_vel"] = _robust_apply(path_bundle["AE_vel"], ae_fit)
        context["pressure"] = _robust_apply(path_bundle["pressure"], pr_fit)
        context["path_quality"] = _robust_apply(path_bundle["path_quality"], pq_fit)
        context["progress_per_bar"] = _robust_apply(path_bundle["progress_per_bar"], pg_fit)
        context["p_tp"] = path_bundle.get("p_tp", np.full(len(base_rets), np.nan))
        context["p_sl"] = path_bundle.get("p_sl", np.full(len(base_rets), np.nan))
        context.update(path_matrices)

        fixed = {
            "strategy_id": selected["strategy_id"],
            "tp_mult": float(_label_tp_sl) if _has_label_policy else float(selected.get("tp_mult", 1.0)),
            "sl_mult": 1.0 if _has_label_policy else float(selected.get("sl_mult", 1.0)),
            "k_recent": 3,
            "K_early": 3,
            "lambda_path": 1.0,
            "a1": 0.5,
            "a2": 0.5,
            "b1": 0.5,
            "b2": 0.5,
            "score_weight_trend": 1.0,
            "score_weight_asym": 0.6,
            "score_weight_choppiness": 0.9,
        }

        best = _sequential_optimise(
            base_rets,
            context,
            train_mask,
            val_mask,
            fixed=fixed,
            cost_pct=cost_pct,
            executed_mask=executed_mask,
        )
        best["source_model"] = selected.get("model", "")
        best["source_artifact"] = selected.get("source_artifact", "")
        best["selected_key"] = key
        best["selection_score"] = conf_name

        # Compute fair baseline from raw paths when available, instead of using
        # pre-capped returns that already have TP/SL applied.
        has_future_paths = all(
            col in context for col in ("future_opens", "future_highs", "future_lows", "future_closes")
        )
        if has_future_paths:
            # Compute baseline using simple TP/SL from raw paths for fair comparison
            baseline_rets_raw = _simulate_baseline_tpsl_from_paths(
                context,
                tp_mult=float(_label_tp_sl) if _has_label_policy else float(selected.get("tp_mult", 1.0)),
                sl_mult=1.0 if _has_label_policy else float(selected.get("sl_mult", 1.0)),
            )
            if baseline_rets_raw is not None:
                baseline_assess = _metric_score(
                    baseline_rets_raw,
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=executed_mask,
                )
                tprint(
                    f"Using path-computed baseline: net_pnl={baseline_assess.get('net_pnl', 0.0):.6f} "
                    f"(vs pre-capped={float(np.mean(base_rets) * len(base_rets)):.6f})"
                )
            else:
                # Fall back to pre-capped returns if path computation fails
                baseline_assess = _metric_score(
                    base_rets,
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=executed_mask,
                )
        else:
            # No future paths available - use pre-capped returns (less accurate)
            baseline_assess = _metric_score(
                base_rets,
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=executed_mask,
            )
            logger.warning(
                "Future path data not available in context; using pre-capped returns as baseline. "
                "Policy optimization may not find improvements because returns are already TP/SL-capped."
            )

        final_rets_assess = replay_exit_policy(base_rets, context, best)
        final_assess = _metric_score(
            final_rets_assess,
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=executed_mask,
        )

        # Only revert if policy degraded performance AND we had fair baseline
        # When using pre-capped baseline, this check is overly strict since baseline is already "optimized"
        if float(final_assess.get("net_pnl", 0.0)) < float(
            baseline_assess.get("net_pnl", 0.0)
        ):
            if has_future_paths:
                logger.warning(
                    "Policy optimisation degraded assessment net_pnl "
                    f"({float(final_assess.get('net_pnl', 0.0)):.6f} < "
                    f"{float(baseline_assess.get('net_pnl', 0.0)):.6f}); "
                    "reverting to baseline params."
                )
                best = dict(fixed)
                final_rets_assess = np.asarray(base_rets, dtype=np.float32)
                final_assess = dict(baseline_assess)
            else:
                # Without future paths, the comparison is unfair - log but don't revert
                logger.info(
                    "Policy optimisation result lower than pre-capped baseline "
                    f"({float(final_assess.get('net_pnl', 0.0)):.6f} < "
                    f"{float(baseline_assess.get('net_pnl', 0.0)):.6f}), "
                    "but future paths unavailable for fair comparison - keeping optimised params."
                )

        best["metrics_baseline"] = baseline_assess
        best["metrics_final"] = final_assess
        best["baseline_net_pnl"] = float(baseline_assess.get("net_pnl", 0.0))
        best["final_net_pnl"] = float(final_assess.get("net_pnl", 0.0))
        best["net_pnl_delta"] = float(best["final_net_pnl"] - best["baseline_net_pnl"])
        strategy_rows.append(best)

    if not strategy_rows:
        tprint("No strategy candidates produced valid policy optimisation results.")
        return {}

    strategy_rows.sort(
        key=lambda row: (
            float(row.get("final_net_pnl", float("-inf"))),
            float(row.get("net_pnl_delta", float("-inf"))),
            float(row.get("metrics_final", {}).get("profit_factor", float("-inf"))),
        ),
        reverse=True,
    )
    best = strategy_rows[0]
    payload = {
        "schema_version": "v4",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "cost_pct": float(cost_pct),
        "strategies": strategy_rows,
        "best_strategy_id": strategy_rows[0].get("strategy_id", ""),
        "best_source_model": strategy_rows[0].get("source_model", ""),
    }
    out_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_policy_params.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    (Path(data_root) / "artifacts" / run_id / "best_policy_params.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    strategy_final_acceptation(data_root, run_id, strategy_rows)
    tprint(f"POLICY OPTIMISER COMPLETE -> {path}")
    return payload
