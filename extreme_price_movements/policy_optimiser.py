"""Policy optimisation stage for extreme_price_movements.

This module consumes the already-selected best strategy (from simple_position_sizer),
keeps TP/SL fixed, runs offset generation first, then sequentially optimises richer
exit-policy parameters. The resulting params are persisted for holdout OOS replay.
"""

from __future__ import annotations

import itertools
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
        for v in (
            pf,
            stability,
            monthly_sortino,
            hit_rate,
            trades_per_day,
            max_drawdown,
            wallet_pnl,
        )
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
    path = (
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json"
    )
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
    sizes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if len(rets) == 0:
        return {
            "net_pnl": 0.0,
            "sortino": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "ulcer": 0.0,
            "tuw": 0.0,
            "pct_negative_trades": 0.0,
            "n_trades": 0,
        }
    if executed_mask is not None:
        executed_mask = np.asarray(executed_mask, dtype=bool)
        if executed_mask.shape[0] == len(rets):
            rets = rets[executed_mask]
            if sizes is not None and sizes.shape[0] == executed_mask.shape[0]:
                sizes = sizes[executed_mask]

    if sizes is None:
        sizes = np.ones_like(rets)

    # Apply sizing and exact fee calculation. Fee is proportional to size * 0.003
    # rets are raw returns here, we multiply by size then subtract fee * size
    if already_net:
        net_rets = rets.astype(np.float64) * sizes
    else:
        net_rets = (rets.astype(np.float64) - float(cost_pct)) * sizes

    downside = net_rets[net_rets < 0]
    ds_std = float(np.std(downside)) if len(downside) > 1 else 1e-6
    gross_win = float(np.sum(net_rets[net_rets > 0]))
    gross_loss = float(np.abs(np.sum(net_rets[net_rets < 0])))
    _, dd = _stable_equity_and_drawdown(net_rets)

    ulcer = float(np.sqrt(np.mean(np.square(dd * 100.0)))) if dd.size else 100.0
    tuw = float(np.mean(dd > 1e-12)) if dd.size else 1.0
    pct_negative_trades = float(np.mean(net_rets < 0)) if len(net_rets) > 0 else 0.0

    return {
        "net_pnl": float(np.sum(net_rets)),
        "sortino": float(np.mean(net_rets)) / ds_std,
        "hit_rate": float(np.mean(net_rets > 0)),
        "profit_factor": gross_win / gross_loss
        if gross_loss > EPS
        else float(gross_win),
        "max_drawdown": float(np.max(dd)) if len(dd) else 0.0,
        "ulcer": ulcer,
        "tuw": tuw,
        "pct_negative_trades": pct_negative_trades,
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
            cand["source_artifact"] = (
                "et_sizer" if model_source == "et" else "ridge_sizer"
            )
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
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
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
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    out["symbol"] = out["symbol"].astype(str)
    merged = out.merge(
        score_df, on=["timestamp", "symbol"], how="left", validate="one_to_one"
    )
    return merged


def _filter_df_by_stage_view(
    df: pd.DataFrame, stage_view: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Apply symbol/time restrictions from a materialized stage view."""
    if stage_view is None or df is None or df.empty:
        return df

    out = df.copy()
    symbol_col = next((c for c in ("symbol", "__symbol__") if c in out.columns), None)
    ts_col = next((c for c in ("timestamp", "__ts__", "t0") if c in out.columns), None)

    allowed_symbols = stage_view.get("symbols")
    if allowed_symbols is not None and symbol_col is not None:
        out = out[out[symbol_col].astype(str).isin([str(s) for s in allowed_symbols])]

    if ts_col is not None and not out.empty:
        ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

        periods = stage_view.get("allowed_periods") or None
        if periods:
            mask = np.zeros(len(out), dtype=bool)
            for period in periods:
                if isinstance(period, dict):
                    p_start = period.get("start_ts") or period.get("start")
                    p_end = period.get("end_ts") or period.get("end")
                elif isinstance(period, (list, tuple)) and len(period) >= 2:
                    p_start, p_end = period[0], period[1]
                else:
                    continue
                p_start = pd.to_datetime(p_start, utc=True, errors="coerce")
                p_end = pd.to_datetime(p_end, utc=True, errors="coerce")
                if pd.isna(p_start) or pd.isna(p_end) or p_end <= p_start:
                    continue
                mask |= (ts >= p_start) & (ts < p_end)
            out = out.loc[mask]
            ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

        start_ts = stage_view.get("allowed_start_ts")
        end_ts = stage_view.get("allowed_end_ts")
        if start_ts:
            out = out[ts >= pd.to_datetime(start_ts, utc=True, errors="coerce")]
            ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        if end_ts:
            out = out[ts <= pd.to_datetime(end_ts, utc=True, errors="coerce")]

    return out


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

            # Find the threshold just above the one that allows a positive PnL for this specific strategy
            strategy_buckets = payload.get("strategy_buckets", {}).get(sid, [])
            if not strategy_buckets:
                # Fallback to the old logic if strategy_buckets detailed list is missing
                row_pnl = float(row.get("net_pnl", 0.0))
                threshold_pct = float(row.get("threshold_pct", threshold_pct))
                if row_pnl <= 0.0 and isinstance(buckets, dict):
                    positive_rows = [
                        r
                        for r in buckets.values()
                        if float(r.get("net_pnl", -1e18)) > 0.0
                    ]
                    if positive_rows:
                        best_pos = max(
                            positive_rows, key=lambda r: float(r.get("net_pnl", -1e18))
                        )
                        threshold_pct = float(
                            best_pos.get("threshold_pct", threshold_pct)
                        )
            else:
                # We have detailed bucket evaluation.
                # Sort by threshold ascending (wider selection to tighter selection)
                sorted_buckets = sorted(
                    strategy_buckets, key=lambda r: float(r.get("threshold_pct", 100.0))
                )
                positive_threshold = None

                # Find the first (widest) threshold that produces positive net PnL
                for b in sorted_buckets:
                    if float(b.get("net_pnl", -1e18)) > 0.0:
                        positive_threshold = float(b.get("threshold_pct", 100.0))
                        break

                if positive_threshold is not None:
                    # We want the threshold "just above" the one that allows a positive PnL.
                    # This gives us a wider base of trades for optimization.
                    # Find the first bucket with threshold slightly higher (more selective) than positive_threshold.
                    just_above = None
                    for b in sorted_buckets:
                        cand_thr = float(b.get("threshold_pct", 100.0))
                        if cand_thr > positive_threshold + 1e-6:
                            just_above = cand_thr
                            break

                    threshold_pct = (
                        just_above if just_above is not None else positive_threshold
                    )
                else:
                    threshold_pct = float(row.get("threshold_pct", threshold_pct))

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

    asym_raw = np.log(
        (np.maximum(delta_mfe_arr, 0.0) + 1e-6)
        / (np.maximum(delta_mae_arr, 0.0) + 1e-6)
    )
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


def _simulate_tpsl_from_paths_unified(
    context: Dict[str, np.ndarray],
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
    enable_trailing: bool = False,
    is_baseline: bool = True,  # If True, this is baseline simulation
) -> Optional[np.ndarray]:
    """UNIFIED simulation for both baseline and policy - ensures identical results.

    Exit Logic:
    1. SL: Hard stop at -sl_mult * barrier (capped, as in real trading)
    2. TP: Trailing profit mechanism (if enabled):
       - Track highest high (for longs) or lowest low (for shorts)
       - When new extreme is reached, update trailing threshold
       - Exit when price retraces trail_mult * distance from extreme to activation
       - If disabled: simple exit at TP level

    Args:
        context: Dictionary with future OHLC paths and entry prices
        tp_mult: Take-profit activation level (relative to barrier)
        sl_mult: Stop-loss level (relative to barrier)
        scores: Optional sizer confidence scores for position sizing (5-15% wallet allocation)
        wallet_range: (min_wallet%, max_wallet%) allocation per trade
        trail_mult: Fraction of profit to give back before trailing exit (0.5 = 50% giveback)
        enable_trailing: If True, use trailing profit; if False, use simple TP exit
        is_baseline: If True, use baseline-specific logic (simpler); if False, use policy logic
    """
    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in context for col in required):
        return None

    future_opens = np.asarray(context["future_opens"], dtype=np.float32)
    future_highs = np.asarray(context["future_highs"], dtype=np.float32)
    future_lows = np.asarray(context["future_lows"], dtype=np.float32)
    future_closes = np.asarray(context["future_closes"], dtype=np.float32)
    future_lengths = np.asarray(
        context.get(
            "future_lengths", np.full(len(future_opens), future_opens.shape[1])
        ),
        dtype=np.int32,
    )

    if future_opens.ndim != 2:
        return None

    n_trades, max_bars = future_opens.shape
    entry = np.asarray(context.get("entry_price", future_opens[:, 0]), dtype=np.float32)
    is_long = np.asarray(
        context.get("is_long", np.ones(n_trades, dtype=bool)), dtype=bool
    )
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(n_trades, 0.02)), dtype=np.float32
        ),
        1e-4,
    )

    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier

    exited = np.zeros(n_trades, dtype=bool)
    exit_rets = np.zeros(n_trades, dtype=np.float32)
    
    # Trailing profit state tracking
    tp_activated = np.zeros(n_trades, dtype=bool)  # Has TP level been reached?
    extreme_price = entry.copy()  # Highest high (long) or lowest low (short) seen
    trailing_thresh = np.zeros(n_trades, dtype=np.float32)  # Trailing stop threshold

    # We will keep track of exactly what bar a trade exits to compute time-in-market

    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)

        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        ent = entry[idx]
        side_a = side[idx]
        bar_high = future_highs[idx, bar]
        bar_low = future_lows[idx, bar]
        bar_close = future_closes[idx, bar]

        valid = (
            np.isfinite(ent)
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
        worst_ret = np.minimum(high_ret, low_ret)

        # --- STOP LOSS (hard capped) ---
        sl_hit = worst_ret <= -sl_dist_a
        
        # --- TAKE PROFIT (simple hit detection) ---
        best_ret = np.maximum(high_ret, low_ret)
        tp_hit = best_ret >= tp_dist_a
        
        # --- TRAILING PROFIT LOGIC ---
        # Update extreme price tracking
        for i_idx, trade_idx in enumerate(idx):
            if side_a[i_idx] > 0:  # Long
                if bar_high[i_idx] > extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_high[i_idx]
            else:  # Short
                if bar_low[i_idx] < extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_low[i_idx]
        
        # Check if TP activation level reached
        for i_idx, trade_idx in enumerate(idx):
            if not tp_activated[trade_idx]:
                extreme_ret = side_a[i_idx] * (extreme_price[trade_idx] / ent[i_idx] - 1.0)
                if extreme_ret >= tp_dist_a[i_idx]:
                    tp_activated[trade_idx] = True

        # No trailing logic available here - this is the simple fallback / baseline simulator.
        # Trailing should be off here anyway.
        
        # Check exit conditions
        bar_exit = np.full(len(idx), np.nan, dtype=np.float32)
        for i_idx, trade_idx in enumerate(idx):
            if sl_hit[i_idx]:
                # SL hit: use hard stop (capped)
                bar_exit[i_idx] = -sl_dist_a[i_idx]
            elif tp_activated[trade_idx]:
                if enable_trailing:
                    # Check if trailing stop hit
                    if side_a[i_idx] > 0:  # Long
                        if bar_low[i_idx] <= trailing_thresh[trade_idx]:
                            # Exit at trailing threshold (or better)
                            exit_price = max(bar_low[i_idx], trailing_thresh[trade_idx])
                            bar_exit[i_idx] = (exit_price / ent[i_idx] - 1.0)
                    else:  # Short
                        if bar_high[i_idx] >= trailing_thresh[trade_idx]:
                            exit_price = min(bar_high[i_idx], trailing_thresh[trade_idx])
                            bar_exit[i_idx] = (ent[i_idx] / exit_price - 1.0)
                else:
                    # Simple TP exit (no trailing)
                    # Exit when we first hit the TP level
                    bar_exit[i_idx] = tp_dist_a[i_idx]
        
        # If no hit, use close for last bar
        is_last = bar >= future_lengths[idx] - 1
        close_ret = side_a * (bar_close / ent - 1.0)
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
    # DELEGATE to baseline function for simple TP/SL (ensures identical results)
    enable_trailing = bool(params.get("enable_trailing", True))
    enable_early_exit = bool(params.get("enable_early_exit", True))
    enable_compression = bool(params.get("enable_compression", True))
    enable_barrier_conf = bool(params.get("enable_barrier_conf", True))
    
    # If only simple TP/SL (no advanced features), use baseline for identical results
    if not any([enable_trailing, enable_early_exit, enable_compression, enable_barrier_conf]):
        tp_mult = float(params.get("tp_mult", 1.0))
        sl_mult = float(params.get("sl_mult", 1.0))
        # Call the baseline simulation directly for identical results
        return _simulate_tpsl_from_paths_unified(
            context,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            enable_trailing=False,
            is_baseline=True,
        )
    
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
    if (
        future_lows.shape != future_opens.shape
        or future_closes.shape != future_opens.shape
    ):
        return None

    n_trades, max_bars = future_opens.shape
    if n_trades != len(returns):
        return None

    entry = np.asarray(context.get("entry_price", future_opens[:, 0]), dtype=np.float32)
    is_long = np.asarray(
        context.get("is_long", np.ones(n_trades, dtype=bool)), dtype=bool
    )
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(n_trades, 0.02)), dtype=np.float32
        ),
        1e-4,
    )
    np.asarray(context.get("confidence", np.zeros(n_trades)), dtype=np.float32)
    p_tp = np.asarray(context.get("p_tp", np.full(n_trades, np.nan)), dtype=np.float32)
    conf = np.asarray(context.get("confidence", np.zeros(n_trades)), dtype=np.float32)
    p_tp = np.asarray(context.get("p_tp", np.full(n_trades, np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(n_trades, np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0.0)

    trend = np.asarray(context.get("trend", np.zeros(n_trades)), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(n_trades, 0.5)), dtype=np.float32)
    choppy = np.asarray(context.get("choppiness", np.zeros(n_trades)), dtype=np.float32)
    score_weight_trend = float(params.get("score_weight_trend", 1.0))
    score_weight_asym = float(params.get("score_weight_asym", 0.6))
    score_weight_choppiness = float(params.get("score_weight_choppiness", 0.9))
    s = (
        score_weight_trend * trend
        + score_weight_asym * asym
        - score_weight_choppiness * choppy
    )
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
    mfe_progress_threshold = float(params.get("mfe_progress_threshold", 0.25))

    trailing_power = float(params.get("trailing_power", 1.0))
    trailing_squash_divisor = float(params.get("trailing_squash_divisor", 1.0))
    trailing_override_alpha = float(params.get("trailing_override_alpha", 0.0))

    enable_early_exit = bool(params.get("enable_early_exit", True))
    enable_trailing = bool(params.get("enable_trailing", True))
    enable_barrier_conf = bool(params.get("enable_barrier_conf", True))
    enable_compression = bool(params.get("enable_compression", True))

    rets = np.asarray(returns, dtype=np.float32).copy()
    exited = np.zeros(n_trades, dtype=bool)
    exit_rets = rets.copy()
    mfe = np.zeros(n_trades, dtype=np.float32)
    mae = np.zeros(n_trades, dtype=np.float32)
    prev_mfe = np.zeros(n_trades, dtype=np.float32)
    prev_mae = np.zeros(n_trades, dtype=np.float32)

    tp_hit_ever = np.zeros(n_trades, dtype=bool)
    
    # TP-anchored trailing state (matching baseline)
    extreme_price = entry.copy()  # Track highest high / lowest low
    trailing_thresh = np.zeros(n_trades, dtype=np.float32)

    import pandas as pd
    # Strict causal expanding median grouped by symbol to avoid cross-asset contamination
    if "symbol" in context:
        df_atr = pd.DataFrame({"barrier": barrier, "symbol": context["symbol"]})
        full_median_atr = df_atr.groupby("symbol")["barrier"].expanding(min_periods=1).median().reset_index(level=0, drop=True).sort_index().values.astype(np.float32)
    else:
        full_median_atr = pd.Series(barrier).expanding(min_periods=1).median().values.astype(np.float32)

    # We will keep track of exactly what bar a trade exits to compute time-in-market
    exit_lengths = np.full(n_trades, max_bars, dtype=np.int32)

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

        # TP hit logic
        tp_hit = best_ret >= tp_dist_a
        tp_hit_ever[idx] = tp_hit_ever[idx] | tp_hit
        
        # --- NEW TRAILING LOGIC ---
        # Update extreme price tracking
        for i_idx, trade_idx in enumerate(idx):
            if side_a[i_idx] > 0:  # Long
                if bar_high[i_idx] > extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_high[i_idx]
            else:  # Short
                if bar_low[i_idx] < extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_low[i_idx]
        
        # Calculate ATR-based median
        atr_ratio_a = barrier_a / np.maximum(full_median_atr[idx], 1e-6)
        trailing_override_band_a = trailing_override_alpha * atr_ratio_a
        giveback_beta_param = float(params.get("giveback_beta", 0.5))

        # Check activation and set trailing threshold
        for i_idx, trade_idx in enumerate(idx):
            extreme_ret = side_a[i_idx] * (extreme_price[trade_idx] / ent[i_idx] - 1.0)

            # Activation condition based on unrealized PnL (extreme_ret) exceeding the band
            if extreme_ret > trailing_override_band_a[i_idx]:
                tp_hit_ever[trade_idx] = True

                activation_price = ent[i_idx] * (1.0 + trailing_override_band_a[i_idx]) if side_a[i_idx] > 0 else ent[i_idx] * (1.0 - trailing_override_band_a[i_idx])
                profit_above_activation = abs(extreme_price[trade_idx] - activation_price)

                # Apply power curve
                profit_above_activation_ret = profit_above_activation / ent[i_idx]
                trail_dist_ret = (profit_above_activation_ret ** trailing_power) / trailing_squash_divisor
                trail_dist = trail_dist_ret * ent[i_idx]

                # Giveback constraint
                D = giveback_beta_param * atr_ratio_a[i_idx] / 100.0

                if side_a[i_idx] > 0:
                    power_stop = extreme_price[trade_idx] - trail_dist
                    giveback_stop = extreme_price[trade_idx] * (1.0 - D)
                    trailing_thresh[trade_idx] = min(power_stop, giveback_stop)
                else:
                    power_stop = extreme_price[trade_idx] + trail_dist
                    giveback_stop = extreme_price[trade_idx] * (1.0 + D)
                    trailing_thresh[trade_idx] = max(power_stop, giveback_stop)
            else:
                # Keep active flag as false if not breached
                pass

        trail_active = tp_hit_ever[idx]
        # Convert price threshold to return for exit logic
        # trail_floor_price = trailing_thresh[idx]
        # trail_floor_return = side_a * (trail_floor_price / ent - 1.0)
        trail_floor_ret = side_a * (trailing_thresh[idx] / ent - 1.0)

        sl_eff = sl_dist_a.copy()
        if enable_compression:
            mfe_norm = mfe[idx] / np.maximum(tp_dist_a, 1e-4)
            c_start = float(params.get("compression_start", 0.5))
            c_full_raw = float(params.get("compression_full", 1.0))
            c_full = max(c_start + 1e-6, c_full_raw)
            c_max = float(params.get("compression_max_fraction", 0.5))
            c_alpha = (
                np.clip((mfe_norm - c_start) / (c_full - c_start), 0.0, 1.0) * c_max
            )
            sl_eff = sl_dist_a * (1.0 - c_alpha)

        open_sl = open_ret <= -sl_eff

        sl_hit = worst_ret <= -sl_eff
        # Check trail breach symmetrically: Longs check Low, Shorts check High
        trail_hit_long = (side_a > 0) & (bar_low <= trailing_thresh[idx])
        trail_hit_short = (side_a < 0) & (bar_high >= trailing_thresh[idx])
        trail_hit = trail_active & (trail_hit_long | trail_hit_short) & enable_trailing

        # For simplicity, open_trail is checked on open_ret vs trail_floor_ret to be completely symmetric
        # regardless of long/short since return handles side inversion
        open_trail = trail_active & (open_ret <= trail_floor_ret) & enable_trailing

        simple_tp_hit = tp_hit & (~enable_trailing)

        bar_exit = np.full(len(idx), np.nan, dtype=np.float32)
        bar_exit = np.where(open_sl, -sl_eff, bar_exit)
        bar_exit = np.where(open_trail & np.isnan(bar_exit), trail_floor_ret, bar_exit)
        bar_exit = np.where(simple_tp_hit & np.isnan(bar_exit), tp_dist_a, bar_exit)

        hard_hit = sl_hit | trail_hit | open_sl | open_trail | simple_tp_hit
        bar_exit = np.where(
            np.isnan(bar_exit),
            np.where(
                simple_tp_hit,
                tp_dist_a,
                np.where(
                    trail_hit,
                    trail_floor_ret,
                    np.where(sl_hit, -sl_eff, np.nan),
                )
            ),
            bar_exit,
        )

        # MFE protection: don't trigger fail_exit if trade has made significant progress toward TP
        # This prevents cutting winning trades during temporary pullbacks
        mfe_progress = mfe[idx] / np.maximum(tp_dist_a, 1e-6)
        has_made_progress = mfe_progress >= mfe_progress_threshold

        fail_exit = np.zeros(len(idx), dtype=bool)
        path_exit = np.zeros(len(idx), dtype=bool)

        if enable_early_exit:
            # Determine if we have high continuation confidence to gate early exits
            high_cont = np.zeros(len(idx), dtype=bool)
            if enable_barrier_conf:
                if has_barrier_proba:
                    p_tp_safe = np.where(np.isfinite(p_tp[idx]), p_tp[idx], 0.5)
                    high_cont = p_tp_safe >= tp_gate
                else:
                    cont_thr = float(params.get("continuation_conf_threshold", 0.5))
                    high_cont = conf[idx] >= cont_thr

            # Never trigger fail_exit at bar 0 (first bar) - give trade time to develop
            if bar >= 1:
                if has_barrier_proba and enable_barrier_conf:
                    p_sl_safe = np.where(np.isfinite(p_sl[idx]), p_sl[idx], 0.5)
                    # Allow fail_exit only if trade hasn't made significant progress
                    fail_exit = (
                        (bar + 1 <= k_early)
                        & (p_sl_safe >= sl_gate)
                        & (~has_made_progress)
                        & (~high_cont)
                    )
                else:
                    # Scale theta_fail threshold by progress - harder to trigger as MFE increases
                    progress_factor = np.where(
                        has_made_progress, 2.0, 1.0
                    )  # 2x harder to trigger if progressed
                    fail_exit = (
                        (bar + 1 <= k_early)
                        & (s_fail > theta_fail * progress_factor)
                        & (~high_cont)
                    )

            path_exit = (
                (bar + 1 >= max(3, d_path))
                & (progress >= progress_threshold)
                & (s_path < theta_path)
                & (~high_cont)
            )

        discretionary_exit = ~(hard_hit) & (fail_exit | path_exit)
        if np.any(discretionary_exit):
            bar_exit = np.where(
                discretionary_exit,
                close_ret,
                bar_exit,
            )

        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True
            exit_lengths[exit_idx] = bar + 1

    # Inject lengths into context so downstream components can use them
    context["_cached_lengths_"] = exit_lengths

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
    np.asarray(context.get("confidence", np.zeros(len(rets))), dtype=np.float32)
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

    conf = np.asarray(context.get("confidence", np.zeros(len(rets))), dtype=np.float32)
    p_tp = np.asarray(context.get("p_tp", np.full(len(rets), np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(len(rets), np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0)

    trend = np.asarray(context.get("trend", np.zeros(len(rets))), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(len(rets), 0.5)), dtype=np.float32)
    choppy = np.asarray(
        context.get("choppiness", np.zeros(len(rets))), dtype=np.float32
    )
    score_weight_trend = float(params.get("score_weight_trend", 1.0))
    score_weight_asym = float(params.get("score_weight_asym", 0.6))
    score_weight_choppiness = float(params.get("score_weight_choppiness", 0.9))
    s = (
        score_weight_trend * trend
        + score_weight_asym * asym
        - score_weight_choppiness * choppy
    )
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

    mfe_progress_threshold = float(params.get("mfe_progress_threshold", 0.25))

    # MFE protection: don't trigger fail_exit if trade has made significant progress toward TP
    mfe_progress = mfe / np.maximum(tp_dist, 1e-6)
    has_made_progress = mfe_progress >= mfe_progress_threshold

    enable_early_exit = bool(params.get("enable_early_exit", True))
    enable_trailing = bool(params.get("enable_trailing", True))
    enable_barrier_conf = bool(params.get("enable_barrier_conf", True))
    enable_compression = bool(params.get("enable_compression", True))

    if enable_early_exit:
        # Determine if we have high continuation confidence to gate early exits
        high_cont = np.zeros(len(rets), dtype=bool)
        if enable_barrier_conf:
            if has_barrier_proba:
                p_tp_safe = np.where(np.isfinite(p_tp), p_tp, 0.5)
                high_cont = p_tp_safe >= tp_gate
            else:
                cont_thr = float(params.get("continuation_conf_threshold", 0.5))
                high_cont = conf >= cont_thr

        if has_barrier_proba and enable_barrier_conf:
            p_sl_safe = np.where(np.isfinite(p_sl), p_sl, 0.5)
            fail_exit = (
                (bars <= k_early)
                & (p_sl_safe >= sl_gate)
                & (~has_made_progress)
                & (~high_cont)
            )
        else:
            # Scale theta_fail threshold by progress - harder to trigger as MFE increases
            progress_factor = np.where(has_made_progress, 2.0, 1.0)
            fail_exit = (
                (bars <= k_early)
                & (s_fail > theta_fail * progress_factor)
                & (~high_cont)
            )

        path_exit = (
            (bars >= max(3, d_path))
            & (progress >= float(params.get("progress_threshold", 0.0)))
            & (s_path < theta_path)
            & (~high_cont)
        )
        # Note: in vectorize replay we don't have close_ret easily accessible,
        # but the fallback is just whatever return the trade had at that point
        rets = np.where(fail_exit | path_exit, rets, rets)

    if enable_compression:
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

    trailing_power = float(params.get("trailing_power", 1.0))
    trailing_squash_divisor = float(params.get("trailing_squash_divisor", 1.0))
    trailing_override_alpha = float(params.get("trailing_override_alpha", 0.0))
    giveback_beta_param = float(params.get("giveback_beta", 0.5))

    # Compute expanding median ATR strictly causally across the mask to prevent future leakage
    # Using a fast pandas rolling median grouped by symbol to prevent cross-asset leakage
    import pandas as pd
    if "symbol" in context:
        df_atr = pd.DataFrame({"barrier": barrier, "symbol": context["symbol"]})
        median_atr = df_atr.groupby("symbol")["barrier"].expanding(min_periods=1).median().reset_index(level=0, drop=True).sort_index().values.astype(np.float32)
    else:
        median_atr = pd.Series(barrier).expanding(min_periods=1).median().values.astype(np.float32)

    atr_ratio = barrier / np.maximum(median_atr, 1e-6)
    trailing_override_band = trailing_override_alpha * atr_ratio

    trail_on = (mfe > trailing_override_band) & enable_trailing

    # Activation price return is exactly trailing_override_band
    # Profit above activation (in return terms)
    profit_above_activation_ret = np.maximum(0.0, mfe - trailing_override_band)

    # Apply power curve
    trail_dist_ret = (profit_above_activation_ret ** trailing_power) / trailing_squash_divisor

    # Giveback constraint
    D = giveback_beta_param * atr_ratio / 100.0

    # Trail floor is the max profit minus the giveback distance
    power_stop = mfe - trail_dist_ret
    giveback_stop = mfe - D
    trail_floor = np.minimum(power_stop, giveback_stop)

    # Apply trail floor if trailing is active
    rets = np.where(trail_on, np.maximum(rets, trail_floor), rets)

    if any([enable_trailing, enable_early_exit, enable_compression, enable_barrier_conf]):
        sl_scale = 1.0 + 0.4 * (m - 1.0)
    else:
        sl_scale = 1.0

    # We no longer hard-clip at TP. It's just an activation threshold.
    rets = np.maximum(rets, -sl_dist * sl_scale)
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
    params["enable_early_exit"] = False
    params["enable_compression"] = False
    params["enable_trailing"] = False  # BASIC RUN: Disable trailing for simple TP/SL
    params["enable_barrier_conf"] = False

    # Group parameters into families for incremental testing
    # Each family is tested separately - if it degrades performance, it's disabled
    param_families: List[Tuple[str, List[Tuple[str, List[Any]]]]] = [
        (
            "position_sizing",
            [
                ("size_power", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
            ],
        ),
        (
            "trailing_stop",
            [
                ("trailing_power", [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
                ("trailing_squash_divisor", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]),
                ("trailing_override_alpha", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                ("giveback_beta", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            ],
        ),
        (
            "multiplier_band",
            [
                ("multiplier_band", [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15)]),
            ],
        ),
    ]

    # Check if future paths are available for fair baseline computation
    has_future_paths_ctx = all(
        col in context
        for col in ("future_opens", "future_highs", "future_lows", "future_closes")
    )
    fixed_tp_mult = float(fixed.get("tp_mult", 1.0))
    fixed_sl_mult = float(fixed.get("sl_mult", 1.0))

    # Track best validation performance and corresponding params
    best_val_params = dict(params)
    param_history: List[
        Tuple[str, Any, float, float]
    ] = []  # (param_name, value, train_pnl, val_pnl)

    # Use path-based simulation when future paths available - this is the FAIR comparison
    def _get_position_sizes(rets: np.ndarray, params_to_use: Dict[str, Any], mask_idx: np.ndarray) -> np.ndarray:
        sizes = np.ones_like(rets)
        size_power = float(params_to_use.get("size_power", 1.0))
        if size_power == 1.0 and "size_power" not in params_to_use:
            return sizes
        if len(rets) == 0:
            return sizes

        scores = context.get("confidence")
        if scores is None or len(scores) != len(rets):
            return sizes

        subset_scores = scores[mask_idx]
        if len(subset_scores) > 0:
            # Rank percentile among approved trades
            ranks = np.argsort(np.argsort(subset_scores))
            approved_rank_pct = ranks / float(max(1, len(subset_scores) - 1))

            size_min = 0.05
            size_max = 0.15
            position_sizes = size_min + (size_max - size_min) * (approved_rank_pct ** size_power)

            sizes[mask_idx] = position_sizes

        return sizes

    # Both baseline and policy should be evaluated on raw price paths
    def _simulate_policy(params_to_use: Dict[str, Any], evaluate_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns raw returns and sizing array."""
        if has_future_paths_ctx:
            rets = _simulate_barwise_path_policy(base_returns, context, params_to_use)
        else:
            rets = replay_exit_policy(base_returns, context, params_to_use)

        sizes = None
        if rets is not None:
            # Apply concurrency logic
            if "timestamps_ms" in context and "symbol" in context:
                # We need exit timestamps for concurrency. We can approximate them using lengths.
                # If length is nan, trade wasn't taken or didn't finish.
                lengths = context.get("_cached_lengths_")
                if lengths is None:
                    # Very rough fallback if length calculation is unavailable in simulation scope
                    lengths = np.full(len(rets), 24, dtype=np.float32)

                # Approximate exit ms
                bar_ms = 3600000  # assuming 1h bars, rough heuristic
                exit_ms = context["timestamps_ms"] + (np.nan_to_num(lengths, nan=0.0) * bar_ms).astype(np.int64)

                conc_mask = _fast_concurrency_mask(
                    context["timestamps_ms"],
                    exit_ms,
                    context["symbol"],
                    context.get("confidence", np.zeros(len(rets))),
                )

                # Zero out returns for trades rejected by concurrency
                rets = np.where(conc_mask, rets, np.nan)

            sizes = _get_position_sizes(rets, params_to_use, evaluate_mask)
        return rets, sizes

    # Test each parameter family incrementally
    # If a family degrades performance, disable it and move to next
    baseline_rets, baseline_sizes = _simulate_policy(params, val_mask)
    if baseline_rets is not None:
        baseline_val_metrics = _metric_score(
            baseline_rets[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            sizes=baseline_sizes[val_mask] if baseline_sizes is not None else None,
            ts_vals=context.get("timestamps_ms", np.arange(len(baseline_rets)))[val_mask] if "timestamps_ms" in context else None,
            scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
        )
        baseline_val_metrics["robust_downside_ratio"] = baseline_val_metrics["net_pnl"] / np.sqrt(baseline_val_metrics["robust_downside_semi_variance"])

        baseline_val_metric = 0.0
    else:
        baseline_val_metric = -1e18

    # Track best validation score and corresponding params across families
    best_overall_val_metric = baseline_val_metric
    best_overall_val_metrics = baseline_val_metrics if baseline_rets is not None else {}
    best_overall_params = dict(params)

    disabled_families: List[str] = []

    for family_name, family_params in param_families:
        # Skip families that were previously disabled
        if family_name in disabled_families:
            tprint(f"  Skipping disabled family: {family_name}")
            continue

        family_best_params = dict(best_overall_params)  # Start from best known
        if family_name == "tp_anchored_giveback_combo":
            family_best_params["enable_trailing"] = True
        elif family_name == "early_exit":
            family_best_params["enable_early_exit"] = True
        elif family_name == "barrier_conf":
            family_best_params["enable_barrier_conf"] = True
        elif family_name == "compression":
            family_best_params["enable_compression"] = True

        tprint(
            f"  Testing family: {family_name} (best_val_pnl={best_overall_val_metric:.4f})"
        )

        import optuna

        # We evaluate scores on the current stage/family.
        # This will be useful to perform Z-score normalisation over all trials.

        family_train_scores_all_metrics = []
        family_trial_configs = []

        is_large_family = family_name in ("early_exit", "trailing_stop")

        if "_progression_" not in best_overall_params:
            best_overall_params["_progression_"] = []
            # we record baseline
            best_overall_params["_progression_"].append({
                "step_name": "Baseline",
                "params": dict(best_overall_params),
                "metrics": dict(best_overall_val_metrics),
                "score": float(best_overall_val_metric)
            })

        # Precompute baseline reference for static Z-score normalization
        baseline_metrics = _metric_score(
            baseline_rets[train_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(executed_mask[train_mask] if executed_mask is not None else None),
            sizes=baseline_sizes[train_mask] if baseline_sizes is not None else None,
            ts_vals=context.get("timestamps_ms", np.arange(len(baseline_rets)))[train_mask] if "timestamps_ms" in context else None,
            scores=context.get("confidence")[train_mask] if context.get("confidence") is not None else None,
        )
        baseline_metrics["robust_downside_ratio"] = baseline_metrics["net_pnl"] / np.sqrt(baseline_metrics["robust_downside_semi_variance"])

        # We need a small delta to prevent division by zero in IQR
        def _calculate_custom_score(metrics: Dict[str, float]) -> float:

            def z(val: float, key: str) -> float:
                # We use the baseline metric as the median, and a fixed percentage (e.g. 10% of abs baseline or a small epsilon) as IQR
                # to keep normalization completely stable across Optuna trials.
                med = float(baseline_metrics.get(key, 0.0))
                iqr = max(abs(med) * 0.1, 1e-5)
                return (val - med) / iqr

            return (
                0.15 * z(metrics.get("net_pnl", 0.0), "net_pnl") +
                0.10 * z(metrics.get("net_pnl", 0.0) / np.sqrt(metrics.get("robust_downside_semi_variance", 1e-6)), "robust_downside_ratio") +
                0.10 * z(metrics.get("pnl_top_25pct_taken_trades", 0.0), "pnl_top_25pct_taken_trades") +
                0.14 * z(metrics.get("weekly_sortino", 0.0), "weekly_sortino") +
                0.11 * z(metrics.get("monthly_sortino", 0.0), "monthly_sortino") +
                0.06 * z(metrics.get("median_pnl_per_winning_trade", 0.0), "median_pnl_per_winning_trade") +
                0.05 * z(metrics.get("weekly_gtp", 0.0), "weekly_gtp") +
                0.05 * z(metrics.get("trade_return_skew", 0.0), "trade_return_skew") -
                0.16 * z(metrics.get("ulcer", 0.0), "ulcer") -
                0.08 * z(metrics.get("tuw", 0.0), "tuw") -
                0.08 * z(metrics.get("neg_months_pct", 0.0), "neg_months_pct") -
                0.06 * z(metrics.get("worst_week_dd_mag", 0.0), "worst_week_dd_mag")
            )

        if is_large_family:
            # Use Optuna for large families
            current_rets, current_sizes = _simulate_policy(family_best_params, train_mask)
            if current_rets is not None:
                current_metrics = _metric_score(
                    current_rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    sizes=current_sizes[train_mask] if current_sizes is not None else None,
                    ts_vals=context.get("timestamps_ms", np.arange(len(current_rets)))[train_mask] if "timestamps_ms" in context else None,
                    scores=context.get("confidence")[train_mask] if context.get("confidence") is not None else None,
                )
                # Compute robust downside ratio
                current_metrics["robust_downside_ratio"] = current_metrics["net_pnl"] / np.sqrt(current_metrics["robust_downside_semi_variance"])
                family_train_scores_all_metrics.append(current_metrics)
            else:
                current_metrics = {}

            tprint(f"  Using Optuna to optimize {family_name}...")

            def objective(trial: optuna.Trial) -> float:
                trial_params = dict(family_best_params)
                for name, grid in family_params:
                    # Convert to tuple for mutability safety, optuna handles numerical lists nicely
                    cand = trial.suggest_categorical(name, grid)
                    if name == "multiplier_band":
                        (
                            trial_params["multiplier_band_min"],
                            trial_params["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        trial_params["a1"] = float(cand)
                        trial_params["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        trial_params["b1"] = float(cand)
                        trial_params["b2"] = 1.0 - float(cand)
                    else:
                        trial_params[name] = cand

                # Split train mask into 5 folds
                valid_train_idx = np.where(train_mask)[0]
                fold_size = len(valid_train_idx) // 5

                fold_scores = []
                for i in range(5):
                    start = i * fold_size
                    end = (i + 1) * fold_size if i < 4 else len(valid_train_idx)
                    fold_mask = np.zeros(len(train_mask), dtype=bool)
                    fold_mask[valid_train_idx[start:end]] = True

                    f_rets, f_sizes = _simulate_policy(trial_params, fold_mask)
                    if f_rets is None:
                        trial.report(-1e18, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        fold_scores.append(-1e18)
                        continue

                    f_metrics = _metric_score(
                        f_rets[fold_mask],
                        cost_pct=cost_pct,
                        already_net=False,
                        executed_mask=(executed_mask[fold_mask] if executed_mask is not None else None),
                        sizes=f_sizes[fold_mask] if f_sizes is not None else None,
                        ts_vals=context.get("timestamps_ms", np.arange(len(f_rets)))[fold_mask] if "timestamps_ms" in context else None,
                        scores=context.get("confidence")[fold_mask] if context.get("confidence") is not None else None,
                    )
                    f_metrics["robust_downside_ratio"] = f_metrics["net_pnl"] / np.sqrt(f_metrics["robust_downside_semi_variance"])

                    # Store metrics for dynamic calculation
                    family_train_scores_all_metrics.append(f_metrics)
                    f_score = _calculate_custom_score(f_metrics, family_train_scores_all_metrics)

                    trial.report(f_score, i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    fold_scores.append(f_score)

                # Full train evaluation
                rets, sizes = _simulate_policy(trial_params, train_mask)
                if rets is None:
                    return -1e18

                metrics = _metric_score(
                    rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    sizes=sizes[train_mask] if sizes is not None else None,
                    ts_vals=context.get("timestamps_ms", np.arange(len(rets)))[train_mask] if "timestamps_ms" in context else None,
                    scores=context.get("confidence")[train_mask] if context.get("confidence") is not None else None,
                )
                metrics["robust_downside_ratio"] = metrics["net_pnl"] / np.sqrt(metrics["robust_downside_semi_variance"])

                family_train_scores_all_metrics.append(metrics)
                family_trial_configs.append(trial_params)

                return _calculate_custom_score(metrics, family_train_scores_all_metrics)

            pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

            # Enqueue the current baseline to ensure we don't regress
            baseline_trial = {}
            for name, grid in family_params:
                if name == "multiplier_band":
                    baseline_trial[name] = (
                        family_best_params.get("multiplier_band_min"),
                        family_best_params.get("multiplier_band_max"),
                    )
                else:
                    baseline_trial[name] = family_best_params.get(
                        name, grid[0]
                    )  # fallback to first grid val if missing
                # if baseline val not in grid, append it to the grid so optuna doesn't crash on categorical
                if baseline_trial[name] not in grid:
                    grid.append(baseline_trial[name])

            study.enqueue_trial(baseline_trial)
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def early_stopping_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
                if len(study.trials) > 100:
                    best_value = study.best_value
                    # check if no improvement in last 50 trials
                    recent_trials = study.trials[-50:]
                    improved_recently = any(t.value is not None and t.value > best_value - 1e-6 for t in recent_trials)
                    if not improved_recently:
                        study.stop()
                        return

                    if len(study.trials) > 75:
                        older_best = max([t.value for t in study.trials[:-75] if t.value is not None] + [-1e18])
                        if best_value > 0 and older_best > 0:
                            if (best_value - older_best) / older_best < 0.005:
                                study.stop()

            study.optimize(objective, n_trials=400, callbacks=[early_stopping_callback])  # Use 400 trials for trailing stop family with TPE

            # Re-evaluate best with final custom score
            best_combo = study.best_params
            best_custom_score = -1e18

            # Select top 10 unique configs from trials
            valid_trials = [t for t in study.trials if t.value is not None]
            valid_trials.sort(key=lambda t: t.value, reverse=True)
            top_10_configs = []
            seen_configs = set()
            for t in valid_trials:
                config_str = str(sorted(t.params.items()))
                if config_str not in seen_configs:
                    seen_configs.add(config_str)
                    top_10_configs.append(t.params)
                if len(top_10_configs) == 10:
                    break

            # Re-evaluate top 10 across 5 folds (using chronologically partitioned train_mask blocks)
            # This requires actual chronological splitting of the train_mask
            valid_train_idx = np.where(train_mask)[0]
            fold_size = len(valid_train_idx) // 5
            folds = []
            for i in range(5):
                start = i * fold_size
                end = (i + 1) * fold_size if i < 4 else len(valid_train_idx)
                fold_mask = np.zeros(len(train_mask), dtype=bool)
                fold_mask[valid_train_idx[start:end]] = True
                folds.append(fold_mask)

            top_10_scores = []
            for config in top_10_configs:
                fold_scores = []
                for f_mask in folds:
                    f_rets, f_sizes = _simulate_policy(config, f_mask)
                    if f_rets is not None:
                        f_metrics = _metric_score(
                            f_rets[f_mask],
                            cost_pct=cost_pct,
                            already_net=False,
                            executed_mask=(executed_mask[f_mask] if executed_mask is not None else None),
                            sizes=f_sizes[f_mask] if f_sizes is not None else None,
                            ts_vals=context.get("timestamps_ms", np.arange(len(f_rets)))[f_mask] if "timestamps_ms" in context else None,
                            scores=context.get("confidence")[f_mask] if context.get("confidence") is not None else None,
                        )
                        f_metrics["robust_downside_ratio"] = f_metrics["net_pnl"] / np.sqrt(f_metrics["robust_downside_semi_variance"])
                        f_score = _calculate_custom_score(f_metrics, family_train_scores_all_metrics)
                        fold_scores.append(f_score)
                    else:
                        fold_scores.append(-1e18)

                # selection_score = median_5fold(Score) - 0.4 * IQR_5fold(Score) + 0.1 * min_5fold(Score)
                f_arr = np.array(fold_scores)
                if len(f_arr) > 0 and np.all(f_arr > -1e17):
                    f_med = np.median(f_arr)
                    f_iqr = np.percentile(f_arr, 75) - np.percentile(f_arr, 25)
                    f_min = np.min(f_arr)
                    sel_score = f_med - 0.4 * f_iqr + 0.1 * f_min
                else:
                    sel_score = -1e18
                top_10_scores.append((sel_score, config, f_arr))

            top_10_scores.sort(key=lambda x: x[0], reverse=True)
            if top_10_scores:
                best_custom_score = top_10_scores[0][0]
                best_combo = top_10_scores[0][1]

                tprint("    Top 10 Finalist Selection:")
                for rank, (s_score, cfg, f_arr) in enumerate(top_10_scores):
                    tprint(f"      {rank+1}. Score: {s_score:.4f} | Folds: {[f'{x:.4f}' for x in f_arr]}")
                    # Re-evaluate global metrics for this specific config to log them
                    if rank < 10:
                        g_rets, g_sizes = _simulate_policy(cfg, train_mask)
                        if g_rets is not None:
                            g_metrics = _metric_score(
                                g_rets[train_mask],
                                cost_pct=cost_pct,
                                already_net=False,
                                executed_mask=(executed_mask[train_mask] if executed_mask is not None else None),
                                sizes=g_sizes[train_mask] if g_sizes is not None else None,
                                ts_vals=context.get("timestamps_ms", np.arange(len(g_rets)))[train_mask] if "timestamps_ms" in context else None,
                                scores=context.get("confidence")[train_mask] if context.get("confidence") is not None else None,
                            )
                            g_metrics["robust_downside_ratio"] = g_metrics["net_pnl"] / np.sqrt(g_metrics["robust_downside_semi_variance"])
                            tprint(f"         -> Global PnL: {g_metrics['net_pnl']:.4f} | Global Sortino: {g_metrics['weekly_sortino']:.4f}")

                    if rank == 0:
                        tprint(f"         Winner Config: {cfg}")

            baseline_custom_score = _calculate_custom_score(family_train_scores_all_metrics[0], family_train_scores_all_metrics)

            if best_custom_score > baseline_custom_score + 1e-6:
                tprint(
                    f"    Optuna {family_name}: best_params={best_combo}, delta Score={best_custom_score - baseline_custom_score:+.4f}"
                )
                for name, _ in family_params:
                    cand = best_combo[name]
                    param_history.append(
                        (name, cand, float(best_custom_score), float("nan"))
                    )  # val appended later
                    if name == "multiplier_band":
                        (
                            family_best_params["multiplier_band_min"],
                            family_best_params["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        family_best_params["a1"] = float(cand)
                        family_best_params["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        family_best_params["b1"] = float(cand)
                        family_best_params["b2"] = 1.0 - float(cand)
                    else:
                        family_best_params[name] = cand
            else:
                tprint(f"    Optuna {family_name}: no improvement over baseline")
        else:
            # Grid Search (Cartesian product) for small families
            current_rets, current_sizes = _simulate_policy(family_best_params, train_mask)
            if current_rets is not None:
                current_train_score = _metric_score(
                    current_rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    sizes=current_sizes[train_mask] if current_sizes is not None else None,
                )["net_pnl"]
            else:
                current_train_score = -1e18

            # Build grids and ensure baseline is included
            safe_grids = []
            for name, grid in family_params:
                if name == "multiplier_band":
                    current_val = (
                        family_best_params.get("multiplier_band_min"),
                        family_best_params.get("multiplier_band_max"),
                    )
                    if current_val[0] is None or current_val[1] is None:
                        current_val = None
                else:
                    current_val = family_best_params.get(name)

                test_grid = list(grid)
                if current_val is not None and current_val not in test_grid:
                    test_grid.append(current_val)
                safe_grids.append(test_grid)

            all_combos = list(itertools.product(*safe_grids))
            tprint(f"  Testing {len(all_combos)} combinations for {family_name}...")

            best_combo_score = current_train_score
            best_combo = None

            for combo in all_combos:
                trial = dict(family_best_params)
                for i, (name, _) in enumerate(family_params):
                    cand = combo[i]
                    if name == "multiplier_band":
                        trial["multiplier_band_min"], trial["multiplier_band_max"] = (
                            cand
                        )
                    elif name == "a1":
                        trial["a1"] = float(cand)
                        trial["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        trial["b1"] = float(cand)
                        trial["b2"] = 1.0 - float(cand)
                    else:
                        trial[name] = cand

                rets, sizes = _simulate_policy(trial, train_mask)
                if rets is None:
                    continue
                train_score = _metric_score(
                    rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    sizes=sizes[train_mask] if sizes is not None else None,
                )["net_pnl"]

                if train_score > best_combo_score + 1e-6:
                    best_combo_score = train_score
                    best_combo = combo

            if best_combo is not None:
                delta_pnl = best_combo_score - current_train_score
                tprint(
                    f"    Grid {family_name}: best_combo={best_combo}, delta PnL={delta_pnl:+.4f}"
                )
                for i, (name, _) in enumerate(family_params):
                    cand = best_combo[i]
                    param_history.append(
                        (name, cand, float(best_combo_score), float("nan"))
                    )
                    if name == "multiplier_band":
                        (
                            family_best_params["multiplier_band_min"],
                            family_best_params["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        family_best_params["a1"] = float(cand)
                        family_best_params["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        family_best_params["b1"] = float(cand)
                        family_best_params["b2"] = 1.0 - float(cand)
                    else:
                        family_best_params[name] = cand
            else:
                tprint(f"    Grid {family_name}: no improvement over baseline")

        # Test if this family improved over baseline
        family_rets, family_sizes = _simulate_policy(family_best_params, val_mask)
        if family_rets is not None:
            family_val_metrics = _metric_score(
                family_rets[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
                sizes=family_sizes[val_mask] if family_sizes is not None else None,
                ts_vals=context.get("timestamps_ms", np.arange(len(family_rets)))[val_mask] if "timestamps_ms" in context else None,
                scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
            )
            family_val_metrics["robust_downside_ratio"] = family_val_metrics["net_pnl"] / np.sqrt(family_val_metrics["robust_downside_semi_variance"])

            # Recalculate baseline val metrics for comparison
            baseline_val_rets, baseline_val_sizes = _simulate_policy(best_overall_params, val_mask)
            baseline_val_metrics = _metric_score(
                baseline_val_rets[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
                sizes=baseline_val_sizes[val_mask] if baseline_val_sizes is not None else None,
                ts_vals=context.get("timestamps_ms", np.arange(len(baseline_val_rets)))[val_mask] if "timestamps_ms" in context else None,
                scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
            )
            baseline_val_metrics["robust_downside_ratio"] = baseline_val_metrics["net_pnl"] / np.sqrt(baseline_val_metrics["robust_downside_semi_variance"])

            # Since baseline_metrics is used as a static ref, we can just call it on the dict
            validation_history = [baseline_val_metrics, family_val_metrics]
            family_val_metric = _calculate_custom_score(family_val_metrics, validation_history)
            best_overall_val_metric_current = _calculate_custom_score(baseline_val_metrics, validation_history)

            if (
                family_val_metric > best_overall_val_metric_current + 0.001
            ):  # Significant improvement threshold
                tprint(
                    f"  ✓ Family '{family_name}' improved validation Score: {best_overall_val_metric_current:.4f} -> {family_val_metric:.4f}"
                )

                # End of family tprint
                tprint(f"  [End of Family: {family_name} Metrics]")
                tprint(f"    PnL: {family_val_metrics['net_pnl']:.4f}")
                tprint(f"    Robust Downside Ratio: {family_val_metrics['robust_downside_ratio']:.4f}")
                tprint(f"    PnL Top 25% Taken Trades: {family_val_metrics['pnl_top_25pct_taken_trades']:.4f}")
                tprint(f"    Weekly Sortino: {family_val_metrics['weekly_sortino']:.4f}")
                tprint(f"    Monthly Sortino: {family_val_metrics['monthly_sortino']:.4f}")
                tprint(f"    Median PnL / Win: {family_val_metrics['median_pnl_per_winning_trade']:.4f}")
                tprint(f"    Weekly GtP: {family_val_metrics['weekly_gtp']:.4f}")
                tprint(f"    Trade Return Skew: {family_val_metrics['trade_return_skew']:.4f}")
                tprint(f"    Ulcer: {family_val_metrics['ulcer']:.4f}")
                tprint(f"    TUW: {family_val_metrics['tuw']:.4f}")
                tprint(f"    Neg Months %: {family_val_metrics['neg_months_pct']:.4%}")
                tprint(f"    Abs Worst Week DD: {family_val_metrics['worst_week_dd_mag']:.4f}")
                tprint(f"    Trades/Day: {family_val_metrics['n_trades'] / 725.0:.2f}") # Approx
                if family_val_metrics['n_trades'] > 0:
                    tprint(f"    Avg PnL/Trade (net): {family_val_metrics['net_pnl']/family_val_metrics['n_trades']:.6f}")

                best_overall_params = dict(family_best_params)
                best_overall_val_metric = family_val_metric
                best_overall_val_metrics = family_val_metrics

                best_overall_params["_progression_"].append({
                    "step_name": family_name,
                    "params": dict(best_overall_params),
                    "metrics": dict(best_overall_val_metrics),
                    "score": float(best_overall_val_metric)
                })

                # Update param_history val scores
                for i in range(len(param_history) - 1, -1, -1):
                    if np.isnan(param_history[i][3]):
                        param_history[i] = (
                            param_history[i][0],
                            param_history[i][1],
                            param_history[i][2],
                            float(family_val_metric),
                        )
                    else:
                        break  # Stop at previous family
            else:
                tprint(
                    f"  ✗ Family '{family_name}' did not improve (or degraded): {best_overall_val_metric:.4f} -> {family_val_metric:.4f}"
                )
                disabled_families.append(family_name)

                # Remove rejected family params from param_history
                for i in range(len(param_history) - 1, -1, -1):
                    if np.isnan(param_history[i][3]):
                        param_history.pop()
                    else:
                        break

    # Return the best params found across all families
    best_val_params = dict(best_overall_params)

    tprint(f"Optimization complete: best_overall_val_score={best_overall_val_metric:.4f}")

    # Compute fair baseline from paths when available
    baseline_rets_raw = None
    if has_future_paths_ctx:
        baseline_rets_raw = _simulate_tpsl_from_paths_unified(
            context,
            tp_mult=fixed_tp_mult,
            sl_mult=fixed_sl_mult,
            enable_trailing=False,  # BASIC RUN: Simple TP/SL only, no trailing
            is_baseline=True,
        )

    if baseline_rets_raw is not None:
        params["metrics_baseline"] = _metric_score(
            baseline_rets_raw[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            ts_vals=context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))[val_mask] if "timestamps_ms" in context else None,
            scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
        )
    else:
        params["metrics_baseline"] = _metric_score(
            base_returns[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            ts_vals=context.get("timestamps_ms", np.arange(len(base_returns)))[val_mask] if "timestamps_ms" in context else None,
            scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
        )
    # Use the params that achieved best validation performance, not just final accumulated
    # This prevents overfitting to training set during sequential optimization
    final_rets, final_sizes = _simulate_policy(best_val_params, val_mask)

    best_val_params["metrics_final"] = _metric_score(
        final_rets[val_mask] if final_rets is not None else np.zeros(len(val_mask)),
        cost_pct=cost_pct,
        already_net=False,
        executed_mask=(executed_mask[val_mask] if executed_mask is not None else None),
        sizes=final_sizes[val_mask] if final_sizes is not None else None,
        ts_vals=context.get("timestamps_ms", np.arange(len(val_mask)))[val_mask] if "timestamps_ms" in context else None,
        scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
    )
    best_val_params["_param_history_"] = param_history  # For diagnostics
    return best_val_params




def _fast_concurrency_mask(
    entry_timestamps: np.ndarray,
    exit_timestamps: np.ndarray,
    symbols: np.ndarray,
    scores: np.ndarray,
    max_global_concurrent: int = 3,
) -> np.ndarray:
    """
    Generates a boolean mask indicating which trades should be executed,
    enforcing that no two trades on the same symbol overlap, and that
    a maximum of `max_global_concurrent` trades can be open globally at any time.
    Trades are processed chronologically. In case of exact timestamp ties, higher confidence wins.
    """
    n_trades = len(entry_timestamps)
    mask = np.zeros(n_trades, dtype=bool)

    if n_trades == 0:
        return mask

    # Sort trades chronologically, then by score (descending)
    sort_idx = np.lexsort((-scores, entry_timestamps))

    active_trades = []  # List to store tuples of (exit_time, symbol)

    for idx in sort_idx:
        entry_t = entry_timestamps[idx]
        exit_t = exit_timestamps[idx]
        sym = symbols[idx]

        # Remove trades that have already exited before or exactly at current entry time
        active_trades = [t for t in active_trades if t[0] > entry_t]

        # Check symbol constraint
        symbol_already_active = any(t[1] == sym for t in active_trades)

        # Check global constraint
        global_cap_reached = len(active_trades) >= max_global_concurrent

        if not symbol_already_active and not global_cap_reached:
            mask[idx] = True
            active_trades.append((exit_t, sym))

    return mask

def run_policy_optimisation(
    data_root: str,
    run_id: str,
    sizer_results: Optional[Dict[str, Any]] = None,
    holdout_frac: float = 0.30,
    cost_pct: float = 0.003,
    use_offset_optimiser: bool = False,
    stage_view: Optional[Dict[str, Any]] = None,
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
    if stage_view:
        filtered_meta: Dict[str, pd.DataFrame] = {}
        for key, mdf in meta_oofs.items():
            mdf_f = _filter_df_by_stage_view(mdf, stage_view)
            if not mdf_f.empty:
                filtered_meta[key] = mdf_f
        meta_oofs = filtered_meta
        if not meta_oofs:
            tprint("Policy optimiser stage filtering removed all meta OOF rows.")
            return {}

    sizer_scores = _load_sizer_oof_scores(data_root, run_id)
    sizer_scores = _filter_df_by_stage_view(sizer_scores, stage_view)
    if sizer_scores.empty:
        tprint("Policy optimiser stage filtering removed all sizer OOF rows.")
        return {}
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
                                np.nanmean(
                                    np.asarray(mdf["oof_u_hat"], dtype=np.float32)
                                )
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
                    tprint(
                        "No meta OOF buckets available; skipping policy optimisation."
                    )
                    return {}

        outcomes = load_trade_outcomes(data_root, run_id, meta_oofs[key])
        outcomes = _filter_df_by_stage_view(outcomes, stage_view)
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

        path_bundle = build_policy_path_state_bundle(
            outcomes, selected_idx=selected_idx
        )
        path_matrices = _pack_future_path_matrices(outcomes, selected_idx=selected_idx)

        _label_tp_sl = float(
            np.nanmedian(path_bundle.get("label_tp_sl_ratio", np.array([3.0])))
        )
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

        # The user wants the splits to be fully applied dynamically based on the holdout_frac.
        # Assuming a default base/meta OOF training portion, position sizer portion for validation, and a test/holdout tail.
        # If n is the total number of trades across the entire period (train+val+test):
        # We calculate the splits such that:
        # Holdout = holdout_frac
        # Train = 2/3 of remaining data
        # Val = 1/3 of remaining data

        n_holdout = int(n * holdout_frac)
        n_train = int(n * (1.0 - holdout_frac) * (2.0 / 3.0))
        n_val = n - n_train - n_holdout

        n_holdout = max(1, n_holdout)
        n_train = max(1, n_train)
        n_val = max(1, n_val)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[order[:n_train]] = True

        val_mask = np.zeros(n, dtype=bool)
        val_mask[order[n_train : n_train + n_val]] = True

        # If n_train + n_val < n, the rest is the holdout, which we ignore here during parameter search
        # If we need to ensure all data is used up to the split, we could adjust.

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
        context["progress_per_bar"] = _robust_apply(
            path_bundle["progress_per_bar"], pg_fit
        )
        context["p_tp"] = path_bundle.get("p_tp", np.full(len(base_rets), np.nan))
        context["p_sl"] = path_bundle.get("p_sl", np.full(len(base_rets), np.nan))
        context.update(path_matrices)

        # CRITICAL FIX: Prioritize sizer's profitable TP/SL geometry over label-derived values
        # The sizer already found profitable parameters - don't overwrite with potentially different label geometry
        sizer_tp_mult = float(selected.get("tp_mult", 1.0))
        sizer_sl_mult = float(selected.get("sl_mult", 1.0))
        label_tp_mult = float(_label_tp_sl) if _has_label_policy else sizer_tp_mult
        label_sl_mult = 1.0  # Label policy typically uses SL=1.0

        # Use sizer values as primary, label only as fallback or for ratio guidance
        # If label and sizer differ significantly, trust the sizer (it was profitable)
        tp_mult_to_use = sizer_tp_mult if sizer_tp_mult > 0 else label_tp_mult
        sl_mult_to_use = sizer_sl_mult if sizer_sl_mult > 0 else label_sl_mult

        fixed = {
            "strategy_id": selected["strategy_id"],
            "tp_mult": tp_mult_to_use,
            "sl_mult": sl_mult_to_use,
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
            # DISABLE all advanced features for basic run
            "enable_early_exit": False,
            "enable_compression": False,
            "enable_barrier_conf": False,
            "enable_trailing": False,  # BASIC RUN: Simple TP/SL only
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

        # Get sizer's reported wallet_pnl early for logging and skip logic
        sizer_wallet_pnl = float(selected.get("wallet_pnl", 0.0))

        # Compute fair baseline from raw paths when available, instead of using
        # pre-capped returns that already have TP/SL applied.
        has_future_paths = all(
            col in context
            for col in ("future_opens", "future_highs", "future_lows", "future_closes")
        )
        baseline_rets_raw = None
        if has_future_paths:
            # Compute baseline using simple TP + hard SL from raw paths for fair comparison
            # Use the SAME sizer-derived TP/SL
            baseline_rets_raw = _simulate_tpsl_from_paths_unified(
                context,
                tp_mult=tp_mult_to_use,
                sl_mult=sl_mult_to_use,
                enable_trailing=False,  # BASIC RUN: Simple TP/SL only, no trailing
                is_baseline=True,
            )
            if baseline_rets_raw is not None:
                baseline_assess = _metric_score(
                    baseline_rets_raw,
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=executed_mask,
                    ts_vals=context.get("timestamps_ms", np.arange(len(baseline_rets_raw))) if "timestamps_ms" in context else None,
                    scores=context.get("confidence") if context.get("confidence") is not None else None,
                )
                # Pre-capped returns = original returns from triple-barrier labeling during training
                # Path-computed returns = re-simulated from raw OHLC using same TP/SL geometry
                # They may differ due to: (1) different exit bar detection, (2) position sizing now applied
                pre_capped_pnl = float(np.mean(base_rets) * len(base_rets)) if len(base_rets) > 0 else 0.0
                tprint(
                    f"Using path-computed baseline: net_pnl={baseline_assess.get('net_pnl', 0.0):.6f} "
                    f"(vs pre-capped={pre_capped_pnl:.6f}, sizer wallet_pnl={sizer_wallet_pnl:.6f})"
                )
            else:
                # Fall back to pre-capped returns if path computation fails
                baseline_assess = _metric_score(
                    base_rets,
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=executed_mask,
                    ts_vals=context.get("timestamps_ms", np.arange(len(base_rets))) if "timestamps_ms" in context else None,
                    scores=context.get("confidence") if context.get("confidence") is not None else None,
                )
        else:
            # No future paths available - use original training returns (pre-capped by triple-barrier)
            baseline_assess = _metric_score(
                base_rets,
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=executed_mask,
                ts_vals=context.get("timestamps_ms", np.arange(len(base_rets))) if "timestamps_ms" in context else None,
                scores=context.get("confidence") if context.get("confidence") is not None else None,
            )
            logger.warning(
                "Future path data not available in context; using original training returns as baseline. "
                "These returns were capped by the triple-barrier labeling during training."
            )

        # Only optimize strategies with positive sizer-reported PnL
        # The baseline can be artificially pessimistic due to path simulation differences,
        # so we trust the sizer's wallet_pnl which already proved profitability
        baseline_pnl = float(baseline_assess.get("net_pnl", 0.0))
        if sizer_wallet_pnl <= 0:
            tprint(
                f"Strategy {selected['strategy_id'][:40]}... has non-positive sizer PnL ({sizer_wallet_pnl:.4f}); skipping optimization. "
                f"(path-computed baseline was {baseline_pnl:.4f})"
            )
            continue

        # FAIR COMPARISON: Compare baseline vs policy on VALIDATION SET ONLY
        # This is the same period used to select policy parameters
        if has_future_paths:
            final_rets_assess = _simulate_barwise_path_policy(base_rets, context, best)
            if final_rets_assess is None:
                final_rets_assess = replay_exit_policy(base_rets, context, best)
        else:
            final_rets_assess = replay_exit_policy(base_rets, context, best)

        def _apply_global_sizing(rets_arr: np.ndarray, strat_params: Dict[str, Any]) -> np.ndarray:
            size_power = float(strat_params.get("size_power", 1.0))
            if size_power == 1.0 and "size_power" not in strat_params:
                return rets_arr
            if len(rets_arr) == 0:
                return rets_arr
            scores_arr = context.get("confidence")
            if scores_arr is None or len(scores_arr) != len(rets_arr):
                return rets_arr
            sized = rets_arr.copy()
            ranks = np.argsort(np.argsort(scores_arr))
            pcts = ranks / float(max(1, len(scores_arr) - 1))
            sz = 0.05 + (0.15 - 0.05) * (pcts ** size_power)
            return sized * sz

        if final_rets_assess is not None:
            final_rets_assess = _apply_global_sizing(final_rets_assess, best)

        # Compute metrics on validation set for fair comparison
        if baseline_rets_raw is not None:
            baseline_val_assess = _metric_score(
                baseline_rets_raw[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
                ts_vals=context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))[val_mask] if "timestamps_ms" in context else None,
                scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
            )
        else:
            baseline_val_assess = _metric_score(
                base_rets[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
                ts_vals=context.get("timestamps_ms", np.arange(len(base_rets)))[val_mask] if "timestamps_ms" in context else None,
                scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
            )

        final_val_assess = _metric_score(
            final_rets_assess[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            ts_vals=context.get("timestamps_ms", np.arange(len(final_rets_assess)))[val_mask] if "timestamps_ms" in context else None,
            scores=context.get("confidence")[val_mask] if context.get("confidence") is not None else None,
        )

        # Also compute full-set metrics for reporting
        final_assess = _metric_score(
            final_rets_assess,
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=executed_mask,
            ts_vals=context.get("timestamps_ms", np.arange(len(final_rets_assess))) if "timestamps_ms" in context else None,
            scores=context.get("confidence") if context.get("confidence") is not None else None,
        )

        # Compare on VALIDATION SET (fair - same period used for optimization)
        baseline_val_pnl = float(baseline_val_assess.get("net_pnl", 0.0))
        final_val_pnl = float(final_val_assess.get("net_pnl", 0.0))

        tprint(
            f"Fair comparison (validation set): baseline={baseline_val_pnl:.4f}, policy={final_val_pnl:.4f}, "
            f"delta={final_val_pnl - baseline_val_pnl:+.4f}"
        )

        # Only revert if policy degraded performance on validation set
        if final_val_pnl < baseline_val_pnl - 0.001:  # Small tolerance for noise
            if has_future_paths:
                logger.warning(
                    "Policy optimisation degraded validation net_pnl "
                    f"({final_val_pnl:.6f} < {baseline_val_pnl:.6f}); "
                    "reverting to baseline params."
                )
                # Preserve param_history for diagnostics even when reverting
                param_history = best.get("_param_history_", [])
                best = dict(fixed)
                best["_param_history_"] = param_history  # Keep for analysis
                best["_reverted_to_baseline_"] = True  # Flag for diagnostics
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
        best["metrics_baseline_val"] = baseline_val_assess
        best["metrics_final"] = final_assess
        best["metrics_final_val"] = final_val_assess
        # Use VALIDATION SET metrics for fair delta comparison
        best["baseline_net_pnl"] = float(baseline_val_assess.get("net_pnl", 0.0))
        best["final_net_pnl"] = float(final_val_assess.get("net_pnl", 0.0))
        best["net_pnl_delta"] = float(best["final_net_pnl"] - best["baseline_net_pnl"])
        # Also store full-set metrics for reference
        best["baseline_net_pnl_full"] = float(baseline_assess.get("net_pnl", 0.0))
        best["final_net_pnl_full"] = float(final_assess.get("net_pnl", 0.0))
        best["net_pnl_delta_full"] = float(
            best["final_net_pnl_full"] - best["baseline_net_pnl_full"]
        )
        # Store selection parameters for comparison with sizer
        best["selection_frac"] = frac
        best["threshold_pct"] = float(selected.get("threshold_pct", 90.0))
        best["cost_pct"] = cost_pct
        best["wallet_range"] = [0.05, 0.15]
        best["sizer_wallet_pnl"] = sizer_wallet_pnl
        best["sizer_net_pnl"] = float(selected.get("net_pnl", 0.0))

        # Build progression table for this strategy
        if "_progression_" in best:
            try:
                import datetime
                import os

                dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                prog_filename = f"policy_progression_{dt_str}.md"
                prog_path = Path(data_root) / "artifacts" / run_id / "policy_params" / prog_filename
                os.makedirs(prog_path.parent, exist_ok=True)

                md_content = [f"# Policy Progression for {selected['strategy_id']}", ""]
                md_content.append(f"Run ID: {run_id}")
                md_content.append("")

                headers = ["Step", "Score", "PnL", "Rb_DownRatio", "PnL_25pct", "WkSortino", "MoSortino", "MedPnL/Win", "WkGtP", "Skew", "Ulcer", "TUW", "NegMo%", "DD_Mag", "Trades/Day", "AvgNetPnL", "Params"]

                # We format a text table manually
                row_fmt = "{:<15} | {:>8} | {:>8} | {:>12} | {:>9} | {:>9} | {:>9} | {:>10} | {:>8} | {:>6} | {:>6} | {:>5} | {:>6} | {:>8} | {:>10} | {:>9} | {}"

                md_content.append(row_fmt.format(*headers))
                md_content.append("-" * 180)

                for step in best["_progression_"]:
                    step_name = step.get("step_name", "")
                    score = step.get("score", 0.0)
                    m = step.get("metrics", {})
                    p = step.get("params", {})

                    # keep only active parameters from param_families + standard tp/sl
                    active_params = {}
                    if "size_power" in p:
                        active_params["size_power"] = p["size_power"]
                    if "trailing_power" in p:
                        active_params["trailing_power"] = p["trailing_power"]
                    if "trailing_squash_divisor" in p:
                        active_params["trailing_squash_divisor"] = p["trailing_squash_divisor"]
                    if "trailing_override_alpha" in p:
                        active_params["trailing_override_alpha"] = p["trailing_override_alpha"]
                    if "giveback_beta" in p:
                        active_params["giveback_beta"] = p["giveback_beta"]
                    if "multiplier_band_min" in p:
                        active_params["mult_min"] = p["multiplier_band_min"]
                    if "multiplier_band_max" in p:
                        active_params["mult_max"] = p["multiplier_band_max"]
                    if "tp_mult" in p:
                        active_params["tp"] = p["tp_mult"]
                    if "sl_mult" in p:
                        active_params["sl"] = p["sl_mult"]

                    def _f(val, fmt=":.4f"):
                        try:
                            return format(val, fmt)
                        except Exception:
                            return str(val)

                    n_trades = m.get('n_trades', 0)
                    trades_per_day = n_trades / 725.0
                    avg_pnl = m.get('net_pnl', 0.0) / max(1, n_trades) if n_trades > 0 else 0.0

                    row_str = row_fmt.format(
                        step_name[:15],
                        f"{score:.4f}",
                        f"{m.get('net_pnl', 0.0):.4f}",
                        f"{m.get('robust_downside_ratio', 0.0):.4f}",
                        f"{m.get('pnl_top_25pct_taken_trades', 0.0):.4f}",
                        f"{m.get('weekly_sortino', 0.0):.4f}",
                        f"{m.get('monthly_sortino', 0.0):.4f}",
                        f"{m.get('median_pnl_per_winning_trade', 0.0):.4f}",
                        f"{m.get('weekly_gtp', 0.0):.4f}",
                        f"{m.get('trade_return_skew', 0.0):.4f}",
                        f"{m.get('ulcer', 0.0):.4f}",
                        f"{m.get('tuw', 0.0):.4f}",
                        f"{m.get('neg_months_pct', 0.0):.1%}",
                        f"{m.get('worst_week_dd_mag', 0.0):.4f}",
                        f"{trades_per_day:.2f}",
                        f"{avg_pnl:.5f}",
                        str(active_params)
                    )
                    md_content.append(row_str)

                md_text = "\n".join(md_content)
                tprint("\n" + "="*120)
                tprint(md_text)
                tprint("="*120 + "\n")

                with open(prog_path, "a") as f:
                    f.write(md_text + "\n\n")

                tprint(f"Appended progression table to {prog_path}")
            except Exception as e:
                logger.warning(f"Failed to generate progression table: {e}")

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
