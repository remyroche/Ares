#!/usr/bin/env python3
"""Feature-aware head gate ablation for contextual TP/SL candidates.

This is a report-only development tool.  It tests whether deployable diagnostic
signals already present in the contextual TP/SL candidate table can identify
periods/rows where a single head should be suppressed.  It does not write
deployment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)


HEAD_ORDER = ("long_bars", "long_dist", "short_asset", "short_bollinger")
ARM_LABELS = {
    "S": "static",
    "R": "rank_only",
    "P": "performance_only",
    "J": "joint_all",
    "I": "independent_all",
}

DIAGNOSTIC_GROUPS: Mapping[str, Sequence[tuple[str, float]]] = {
    "uncertainty": (
        ("generated_score_uncertainty_p1mp", 1.0),
        ("generated_score_entropy", 1.0),
        ("oof_prob_uncertainty", 1.0),
        ("oof_contrib_entropy", 1.0),
        ("oof_rank_bin_se_oof", 1.0),
        ("oof_score_path_std", 1.0),
        ("oof_score_path_volatility", 1.0),
        ("oof_rank_path_std", 1.0),
        ("oof_score_reversal_count", 1.0),
        ("generated_score_abs_distance_from_half", -1.0),
        ("oof_score_margin_top10", -1.0),
        ("oof_score_margin_top20", -1.0),
        ("oof_score_margin_top30", -1.0),
        ("oof_rank_margin_top10", -1.0),
        ("oof_rank_margin_top20", -1.0),
        ("oof_rank_margin_top30", -1.0),
    ),
    "drift": (
        ("generated_score_abs_diff_1", 1.0),
        ("generated_score_abs_diff_4", 1.0),
        ("generated_score_abs_diff_24", 1.0),
        ("generated_score_abs_minus_prev24_mean", 1.0),
        ("generated_score_prev24_std", 1.0),
        ("generated_strategy_score_shift_abs_z", 1.0),
        ("oof_feature_drift_psi_core", 1.0),
        ("oof_feature_drift_ks_core", 1.0),
        ("oof_feature_drift_cov_shift", 1.0),
    ),
    "ood": (
        ("generated_strategy_score_ood_abs_z", 1.0),
        ("generated_strategy_barrier_ood_abs_z", 1.0),
        ("generated_strategy_friction_ood_abs_z", 1.0),
        ("oof_dae_reconstruction_error", 1.0),
        ("oof_dae_reconstruction_error_zscore", 1.0),
        ("oof_latent_mahalanobis_drift", 1.0),
        ("oof_support_gap", 1.0),
        ("oof_rare_leaf_fraction", 1.0),
        ("oof_leaf_count_mean", -1.0),
        ("oof_leaf_count_median", -1.0),
        ("oof_leaf_count_q25", -1.0),
        ("oof_leaf_count_p10", -1.0),
        ("oof_leaf_count_min", -1.0),
        ("oof_leaf_train_freq_mean", -1.0),
        ("oof_leaf_train_freq_p10", -1.0),
        ("oof_leaf_train_freq_min", -1.0),
    ),
    "performance": (
        ("generated_hr_surprise_24", -1.0),
        ("generated_hr_surprise_96", -1.0),
        ("generated_weighted_hr_surprise_24", -1.0),
        ("generated_weighted_hr_surprise_96", -1.0),
        ("generated_loss_rate_24", 1.0),
        ("generated_loss_rate_96", 1.0),
        ("generated_matured_count_24", 1.0),
        ("generated_matured_count_96", 1.0),
    ),
    "recent_hr_surprise": (
        ("generated_hr_surprise_24", -1.0),
        ("generated_hr_surprise_96", -1.0),
        ("generated_weighted_hr_surprise_24", -1.0),
        ("generated_weighted_hr_surprise_96", -1.0),
    ),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _head_name(strategy_id: str) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _parse_combo(combo_id: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    chunks = str(combo_id).split("_")
    i = 0
    while i < len(chunks):
        if i + 1 < len(chunks) and ":" not in chunks[i] and ":" in chunks[i + 1]:
            head_prefix = f"{chunks[i]}_{chunks[i + 1].split(':', 1)[0]}"
            label = chunks[i + 1].split(":", 1)[1]
            if head_prefix in HEAD_ORDER and label in ARM_LABELS:
                mapping[head_prefix] = ARM_LABELS[label]
                i += 2
                continue
        if ":" in chunks[i]:
            head, label = chunks[i].split(":", 1)
            if head in HEAD_ORDER and label in ARM_LABELS:
                mapping[head] = ARM_LABELS[label]
        i += 1
    missing = [head for head in HEAD_ORDER if head not in mapping]
    if missing:
        raise ValueError(f"Combo {combo_id!r} is missing heads: {missing}")
    return mapping


def _load_arm_tables(source_dir: Path, arms: Sequence[str]) -> tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    tables: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, Path] = {}
    for arm in sorted(set(arms)):
        path = source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate table: {path}")
        frame = pd.read_parquet(path)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        frame["head"] = frame["strategy_id"].map(_head_name)
        tables[arm] = frame
        paths[arm] = path
    return tables, paths


def _combo_candidates(source_dir: Path, combo_id: str) -> tuple[pd.DataFrame, Dict[str, str], Dict[str, Path]]:
    mapping = _parse_combo(combo_id)
    tables, paths = _load_arm_tables(source_dir, tuple(mapping.values()))
    frames = []
    for head, arm in mapping.items():
        frames.append(tables[arm].loc[tables[arm]["head"].eq(head)].copy())
    candidates = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )
    return candidates, mapping, paths


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame(), pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    exit_reason = accepted.get("position_exit_reason", pd.Series("", index=accepted.index))
    accepted["is_full_sl"] = exit_reason.astype(str).str.contains("full_sl|sl", case=False, na=False)
    accepted["is_timeout"] = exit_reason.astype(str).str.contains("timeout", case=False, na=False)

    frames = []
    for cols in (["day"], ["day", "head"], ["week"], ["week", "head"]):
        cur = (
            accepted.groupby(cols, as_index=False)
            .agg(
                net_pnl=("net_pnl_amount", "sum"),
                gross_pnl=("gross_pnl_amount", "sum"),
                trades=("accepted", "size"),
                hit_rate=("is_win", "mean"),
                full_sl_rate=("is_full_sl", "mean"),
                timeout_rate=("is_timeout", "mean"),
            )
            .sort_values(cols)
        )
        cur.insert(0, "period_type", "_".join(cols))
        frames.append(cur)
    return pd.concat(frames[:2], ignore_index=True), pd.concat(frames[2:], ignore_index=True)


def _robust_percentile_by_head(
    values: pd.Series,
    heads: pd.Series,
    *,
    reference_values: pd.Series | None = None,
    reference_heads: pd.Series | None = None,
) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce")
    if reference_values is None or reference_heads is None:
        return raw.groupby(heads, sort=False).rank(pct=True).astype("float32")
    ref_raw = pd.to_numeric(reference_values, errors="coerce")
    out = pd.Series(np.nan, index=raw.index, dtype="float32")
    for head, idx in heads.groupby(heads, sort=False).groups.items():
        ref = ref_raw.loc[reference_heads.eq(head)].dropna().to_numpy(dtype=np.float64)
        if ref.size == 0:
            continue
        ref.sort()
        vals = raw.loc[idx].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        pct = np.full(vals.shape, np.nan, dtype=np.float32)
        pct[finite] = np.searchsorted(ref, vals[finite], side="right") / float(ref.size)
        out.loc[idx] = pct
    return out.astype("float32")


def _diagnostic_scores(
    candidates: pd.DataFrame,
    groups: Sequence[str],
    *,
    reference_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = candidates.copy()
    out["head"] = out["strategy_id"].astype(str).map(_head_name)
    ref = reference_candidates.copy() if reference_candidates is not None else None
    if ref is not None:
        ref["head"] = ref["strategy_id"].astype(str).map(_head_name)
    group_scores: List[str] = []
    coverage_records: List[Dict[str, Any]] = []
    for group in groups:
        features = DIAGNOSTIC_GROUPS.get(group)
        if not features:
            raise ValueError(f"Unknown diagnostic group: {group}")
        cols = [(col, sign) for col, sign in features if col in out.columns]
        if not cols:
            out[f"diagnostic_{group}_risk"] = np.nan
            coverage_records.append({"group": group, "available_feature_count": 0})
            continue
        parts = []
        for col, sign in cols:
            scaled = _robust_percentile_by_head(
                sign * pd.to_numeric(out[col], errors="coerce"),
                out["head"],
                reference_values=sign * pd.to_numeric(ref[col], errors="coerce")
                if ref is not None and col in ref.columns
                else None,
                reference_heads=ref["head"] if ref is not None and col in ref.columns else None,
            )
            parts.append(scaled)
        mat = pd.concat(parts, axis=1)
        out[f"diagnostic_{group}_risk"] = mat.mean(axis=1, skipna=True).astype("float32")
        group_scores.append(f"diagnostic_{group}_risk")
        coverage_records.append({"group": group, "available_feature_count": len(cols)})
    if group_scores:
        out["diagnostic_composite_risk"] = out[group_scores].mean(axis=1, skipna=True).astype("float32")
    else:
        out["diagnostic_composite_risk"] = np.nan
    out.attrs["diagnostic_feature_coverage"] = coverage_records
    return out


def _filter_candidates(
    candidates: pd.DataFrame,
    *,
    gate_head: str,
    risk_column: str,
    threshold: float,
    action: str,
    size_multiplier: float,
    priority_multiplier: float,
) -> pd.DataFrame:
    out = candidates.copy()
    out["head"] = out["strategy_id"].astype(str).map(_head_name)
    risk = pd.to_numeric(out[risk_column], errors="coerce")
    gate = out["head"].eq(gate_head) & risk.ge(float(threshold))
    if action == "drop":
        out = out.loc[~gate].copy()
    elif action == "size":
        if "portfolio_size_multiplier" not in out.columns:
            out["portfolio_size_multiplier"] = 1.0
        current = pd.to_numeric(out["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
        out.loc[gate, "portfolio_size_multiplier"] = current.loc[gate] * float(size_multiplier)
    elif action == "priority":
        if "portfolio_priority_multiplier" not in out.columns:
            out["portfolio_priority_multiplier"] = 1.0
        current = pd.to_numeric(out["portfolio_priority_multiplier"], errors="coerce").fillna(1.0)
        out.loc[gate, "portfolio_priority_multiplier"] = current.loc[gate] * float(priority_multiplier)
    else:
        raise ValueError(f"Unknown action: {action}")
    return out.drop(columns=["head"], errors="ignore").reset_index(drop=True)


def _apply_weekly_gate(
    candidates: pd.DataFrame,
    *,
    weekly_gate_path: Path | None,
    weekly_gate_head: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if weekly_gate_path is None:
        return candidates, {
            "weekly_gate_path": None,
            "weekly_gate_head": None,
            "weekly_gate_removed_rows": 0,
            "weekly_gate_closed_weeks": 0,
            "weekly_gate_total_weeks": 0,
        }
    if not weekly_gate_path.exists():
        raise FileNotFoundError(f"Missing weekly gate file: {weekly_gate_path}")
    gate = pd.read_csv(weekly_gate_path)
    missing = sorted({"week", "gate_closed"} - set(gate.columns))
    if missing:
        raise ValueError(f"Weekly gate file {weekly_gate_path} is missing columns: {missing}")
    out = candidates.copy()
    heads = out["strategy_id"].astype(str).map(_head_name)
    weeks = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.to_period("W").astype(str)
    gate_map = gate.set_index("week")["gate_closed"].astype(bool)
    closed = weeks.map(gate_map).fillna(False).astype(bool)
    remove = heads.eq(str(weekly_gate_head)) & closed
    return out.loc[~remove].copy().reset_index(drop=True), {
        "weekly_gate_path": str(weekly_gate_path),
        "weekly_gate_head": str(weekly_gate_head),
        "weekly_gate_removed_rows": int(remove.sum()),
        "weekly_gate_closed_weeks": int(gate["gate_closed"].astype(bool).sum()),
        "weekly_gate_total_weeks": int(len(gate)),
    }


def _run_one(
    *,
    candidates: pd.DataFrame,
    source_dir: Path,
    out_dir: Path,
    combo_id: str,
    mapping: Mapping[str, str],
    input_paths: Mapping[str, Path],
    gate_head: str,
    risk_column: str,
    threshold: float,
    action: str,
    size_multiplier: float,
    priority_multiplier: float,
    market_mode: str,
    weekly_gate_path: Path | None,
    weekly_gate_head: str,
    start: str,
    end: str,
    risk_reference_end: str | None,
    risk_reference_rows: int | None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = _filter_candidates(
        candidates,
        gate_head=gate_head,
        risk_column=risk_column,
        threshold=threshold,
        action=action,
        size_multiplier=size_multiplier,
        priority_multiplier=priority_multiplier,
    )
    filtered, weekly_gate_meta = _apply_weekly_gate(
        filtered,
        weekly_gate_path=weekly_gate_path,
        weekly_gate_head=weekly_gate_head,
    )
    filtered.to_parquet(out_dir / "combo_candidates.parquet", index=False)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(filtered)
    decisions, equity, metrics = replay_candidates(
        filtered,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    decisions.to_parquet(out_dir / "combo_replay_decisions.parquet", index=False)
    equity.to_parquet(out_dir / "combo_replay_equity.parquet", index=False)
    daily, weekly = _period_tables(decisions)
    daily.to_csv(out_dir / "combo_replay_daily_metrics.csv", index=False)
    weekly.to_csv(out_dir / "combo_replay_weekly_metrics.csv", index=False)
    gated = _filter_candidates(
        candidates,
        gate_head=gate_head,
        risk_column=risk_column,
        threshold=threshold,
        action="drop",
        size_multiplier=0.0,
        priority_multiplier=0.0,
    )
    gate_rows = int(len(candidates) - len(gated))
    gate_head_rows = int(candidates["strategy_id"].astype(str).map(_head_name).eq(gate_head).sum())
    manifest = {
        "generated_by": "ablate_contextual_tp_sl_diagnostic_head_gate",
        "source_dir": str(source_dir),
        "combo_id": combo_id,
        "head_arm_mapping": dict(mapping),
        "gate_head": gate_head,
        "risk_column": risk_column,
        "threshold": float(threshold),
        "action": action,
        "size_multiplier": float(size_multiplier),
        "priority_multiplier": float(priority_multiplier),
        "start_filter": str(start),
        "end_filter": str(end),
        "risk_reference_end": risk_reference_end,
        "risk_reference_rows": risk_reference_rows,
        **weekly_gate_meta,
        "gate_rows": gate_rows,
        "gate_head_rows": gate_head_rows,
        "gate_row_share_within_head": gate_rows / gate_head_rows if gate_head_rows else None,
        "market_mode": market_mode,
        "candidate_rows": int(len(filtered)),
        "policy_params": asdict(params),
        "input_hashes": {arm: _sha256(path) for arm, path in sorted(input_paths.items())},
        "metrics": dict(metrics),
    }
    (out_dir / "combo_replay_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    return {
        "label": out_dir.name,
        "out_dir": str(out_dir),
        "gate_head": gate_head,
        "risk_column": risk_column,
        "threshold": float(threshold),
        "action": action,
        "size_multiplier": float(size_multiplier),
        "priority_multiplier": float(priority_multiplier),
        "start_filter": str(start),
        "end_filter": str(end),
        "risk_reference_end": risk_reference_end,
        "risk_reference_rows": risk_reference_rows,
        **weekly_gate_meta,
        "gate_rows": gate_rows,
        "gate_head_rows": gate_head_rows,
        "gate_row_share_within_head": gate_rows / gate_head_rows if gate_head_rows else np.nan,
        "candidate_rows": int(len(filtered)),
        "net_pnl": metrics.get("net_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "trade_count": metrics.get("trade_count"),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
    }


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _timestamp_mask(frame: pd.DataFrame, *, start: str, end: str) -> pd.Series:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= ts.ge(pd.Timestamp(start, tz="UTC"))
    if end:
        mask &= ts.le(pd.Timestamp(end, tz="UTC"))
    return mask


def _label(risk_column: str, threshold: float, action: str, size: float, priority: float) -> str:
    base = risk_column.replace("diagnostic_", "").replace("_risk", "")
    thr = int(round(float(threshold) * 1000))
    if action == "size":
        suffix = f"sizex{int(round(size * 1000)):03d}"
    elif action == "priority":
        suffix = f"priorityx{int(round(priority * 1000)):03d}"
    else:
        suffix = "drop"
    return f"{base}_gte{thr}_{suffix}"


def _append_weekly_gate_suffix(label: str, weekly_gate_path: Path | None, weekly_gate_head: str) -> str:
    if weekly_gate_path is None:
        return label
    return f"{label}__{weekly_gate_head}_{weekly_gate_path.parent.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--combo-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate-head", default="long_bars", choices=HEAD_ORDER)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--groups", default="uncertainty,drift,ood,recent_hr_surprise")
    parser.add_argument("--risk-columns", default="diagnostic_composite_risk")
    parser.add_argument("--thresholds", default="0.75,0.85,0.90,0.95")
    parser.add_argument("--actions", default="drop")
    parser.add_argument("--size-multiplier", type=float, default=0.50)
    parser.add_argument("--priority-multiplier", type=float, default=0.50)
    parser.add_argument("--start", default="", help="Optional inclusive replay start timestamp.")
    parser.add_argument("--end", default="", help="Optional inclusive replay end timestamp.")
    parser.add_argument(
        "--risk-reference-end",
        default="",
        help=(
            "Optional inclusive end timestamp for percentile-reference rows. "
            "Defaults to --start when --start is provided; otherwise all rows are used."
        ),
    )
    parser.add_argument(
        "--weekly-gate-path",
        type=Path,
        default=None,
        help="Optional head_gate_weeks.csv produced by a causal weekly head-gate ablation.",
    )
    parser.add_argument("--weekly-gate-head", default="long_bars", choices=HEAD_ORDER)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates, mapping, input_paths = _combo_candidates(args.source_dir, str(args.combo_id))
    groups = _parse_csv(args.groups)
    reference_candidates = None
    reference_end = str(args.risk_reference_end or args.start or "")
    if reference_end:
        ref_ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
        reference_candidates = candidates.loc[ref_ts.lt(pd.Timestamp(reference_end, tz="UTC"))].copy()
        if reference_candidates.empty:
            raise ValueError(f"No percentile-reference rows before {reference_end!r}")
    scored = _diagnostic_scores(candidates, groups, reference_candidates=reference_candidates)
    if args.start or args.end:
        scored = scored.loc[_timestamp_mask(scored, start=str(args.start), end=str(args.end))].copy()
        if scored.empty:
            raise ValueError(f"No candidate rows remain after start/end filtering: {args.start!r}, {args.end!r}")
    coverage = scored.attrs.get("diagnostic_feature_coverage", [])
    pd.DataFrame(coverage).to_csv(args.out_dir / "diagnostic_feature_coverage.csv", index=False)

    risk_columns = _parse_csv(args.risk_columns)
    risk_columns = [
        col if col.startswith("diagnostic_") else f"diagnostic_{col}_risk"
        for col in risk_columns
    ]
    missing = [col for col in risk_columns if col not in scored.columns]
    if missing:
        raise KeyError(f"Missing requested risk columns: {missing}")

    rows: List[Dict[str, Any]] = []
    for risk_column in risk_columns:
        for threshold in _parse_floats(args.thresholds):
            for action in _parse_csv(args.actions):
                run_label = _append_weekly_gate_suffix(
                    _label(
                        risk_column,
                        threshold,
                        action,
                        float(args.size_multiplier),
                        float(args.priority_multiplier),
                    ),
                    args.weekly_gate_path,
                    str(args.weekly_gate_head),
                )
                run_dir = args.out_dir / "materialized" / run_label
                print(f"RUN {run_dir.name}", flush=True)
                rows.append(
                    _run_one(
                        candidates=scored,
                        source_dir=args.source_dir,
                        out_dir=run_dir,
                        combo_id=str(args.combo_id),
                        mapping=mapping,
                        input_paths=input_paths,
                        gate_head=str(args.gate_head),
                        risk_column=risk_column,
                        threshold=float(threshold),
                        action=str(action),
                        size_multiplier=float(args.size_multiplier),
                        priority_multiplier=float(args.priority_multiplier),
                        market_mode=str(args.market_mode),
                        weekly_gate_path=args.weekly_gate_path,
                        weekly_gate_head=str(args.weekly_gate_head),
                        start=str(args.start),
                        end=str(args.end),
                        risk_reference_end=reference_end or None,
                        risk_reference_rows=int(len(reference_candidates))
                        if reference_candidates is not None
                        else None,
                    )
                )

    summary = pd.DataFrame(rows).sort_values(["net_pnl", "max_drawdown"], ascending=[False, False])
    summary.to_csv(args.out_dir / "diagnostic_head_gate_summary.csv", index=False)
    manifest = {
        "generated_by": "ablate_contextual_tp_sl_diagnostic_head_gate",
        "source_dir": str(args.source_dir),
        "combo_id": str(args.combo_id),
        "out_dir": str(args.out_dir),
        "gate_head": str(args.gate_head),
        "groups": groups,
        "risk_columns": risk_columns,
        "thresholds": _parse_floats(args.thresholds),
        "actions": _parse_csv(args.actions),
        "start": str(args.start),
        "end": str(args.end),
        "risk_reference_end": reference_end or None,
        "risk_reference_rows": int(len(reference_candidates)) if reference_candidates is not None else None,
        "size_multiplier": float(args.size_multiplier),
        "priority_multiplier": float(args.priority_multiplier),
        "weekly_gate_path": str(args.weekly_gate_path) if args.weekly_gate_path is not None else None,
        "weekly_gate_head": str(args.weekly_gate_head) if args.weekly_gate_path is not None else None,
        "run_count": int(len(summary)),
    }
    (args.out_dir / "diagnostic_head_gate_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Diagnostic Head Gate Ablation",
        "",
        f"Source: `{args.source_dir}`",
        f"Combo: `{args.combo_id}`",
        f"Gate head: `{args.gate_head}`",
        "",
        (
            "This is a long-window development/proxy replay. It uses live-available "
            "diagnostic features but threshold selection is not OOS."
        ),
        "",
        "## Diagnostic Feature Coverage",
        "",
        pd.DataFrame(coverage).to_markdown(index=False),
        "",
        "## Runs",
        "",
        summary.to_markdown(index=False),
    ]
    (args.out_dir / "diagnostic_head_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "run_count": int(len(summary))}), indent=2))


if __name__ == "__main__":
    main()
