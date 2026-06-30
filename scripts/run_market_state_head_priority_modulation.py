#!/usr/bin/env python3
"""Replay market-state head-priority modulation.

This is a portfolio-routing ablation.  It does not change labels, meta scores,
base thresholds, q-fail, HeadHealth, or market-state threshold control.  It can
apply bounded per-head, per-timestamp pre-filter rank-prior adjustments and/or
auction-priority actions used by the global portfolio manager.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_SCORE_DIR = Path(
    "data_perp/reports/market_state_controller_bundle_score_t1_lgbm_maturity_noop_20260626"
)
DEFAULT_TRAIN_DEPLOYABLE = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070/"
    "simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625/"
    "A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_head_priority_modulation_20260626"
)

BASELINE_ARM = "P0_static_priority"
ARM_SPECS: dict[str, dict[str, Any]] = {
    "P1_lcb_priority": {
        "kind": "single",
        "column": "pred_lcb_utility",
        "description": "Head priority from median predicted lower-confidence utility.",
    },
    "P2_mean_minus_fullsl_priority": {
        "kind": "composite",
        "description": "Head priority from predicted utility net of full-SL and timeout risk.",
    },
    "P3_fullsl_relief_priority": {
        "kind": "single",
        "column": "neg_pred_full_sl",
        "description": "Head priority favors heads with lower predicted full-SL probability.",
    },
}

PRIORITY_ACTIONS = {"adjustment", "multiplier", "both"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_candidates(path: Path) -> pd.DataFrame:
    out = pd.read_parquet(path)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].astype(str).map(mstc._strategy_head)
    return mstc.normalise_candidate_table(out)


def _load_controller_predictions(path: Path) -> pd.DataFrame:
    out = pd.read_parquet(path)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].astype(str).map(mstc._strategy_head)
    return out


def priority_action_values(
    centered_score: pd.Series,
    *,
    scale: float,
    max_adjustment: float,
    max_priority_multiplier: float,
    priority_action: str,
) -> tuple[pd.Series, pd.Series]:
    """Map centered head-state scores into bounded auction actions.

    The additive channel can cross small frontiers.  The multiplicative channel
    preserves within-head ordering while changing how much the global auction
    should value a head in the current market state.
    """
    action = str(priority_action or "adjustment").strip().lower()
    if action not in PRIORITY_ACTIONS:
        raise ValueError(f"unknown priority action: {priority_action!r}")
    safe_scale = max(float(scale), 1e-12)
    signed = np.tanh(pd.to_numeric(centered_score, errors="coerce") / safe_scale)
    if action in {"adjustment", "both"}:
        adjustment = float(max_adjustment) * signed
        adjustment = adjustment.clip(lower=-abs(float(max_adjustment)), upper=abs(float(max_adjustment)))
    else:
        adjustment = pd.Series(0.0, index=centered_score.index, dtype=float)
    if action in {"multiplier", "both"}:
        max_mult = max(float(max_priority_multiplier), 1.0)
        log_cap = float(np.log(max_mult)) if max_mult > 1.0 else 0.0
        multiplier = np.exp(log_cap * signed)
        multiplier = pd.Series(multiplier, index=centered_score.index, dtype=float).clip(
            lower=1.0 / max(max_mult, 1e-12),
            upper=max_mult,
        )
    else:
        multiplier = pd.Series(1.0, index=centered_score.index, dtype=float)
    return adjustment.astype(float), multiplier.astype(float)


def rank_prior_values(
    centered_score: pd.Series,
    *,
    scale: float,
    max_rank_adjustment: float,
) -> pd.Series:
    """Map centered head-state scores into bounded pre-filter rank priors."""
    max_rank = abs(float(max_rank_adjustment))
    if max_rank <= 0.0:
        return pd.Series(0.0, index=centered_score.index, dtype=float)
    safe_scale = max(float(scale), 1e-12)
    signed = np.tanh(pd.to_numeric(centered_score, errors="coerce") / safe_scale)
    return (max_rank * signed).clip(lower=-max_rank, upper=max_rank).astype(float)


def _candidate_universe_signature(candidates: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    by_head = {
        str(k): int(v)
        for k, v in candidates.groupby(candidates["head"].astype(str), sort=True).size().items()
    }
    return {
        "rows": int(len(candidates)),
        "timestamp_count": int(timestamps.nunique()),
        "timestamp_min": timestamps.min(),
        "timestamp_max": timestamps.max(),
        "heads": sorted(map(str, candidates["head"].dropna().unique())),
        "by_head_rows": by_head,
    }


def _prediction_feature_frame(
    predictions: pd.DataFrame,
    *,
    sl_weight: float,
    timeout_weight: float,
) -> pd.DataFrame:
    out = predictions.copy()
    for col in (
        "pred_lcb_utility",
        "pred_mean_utility",
        "pred_full_sl",
        "pred_timeout",
        "_rank",
        "_threshold",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "pred_full_sl" in out.columns:
        out["neg_pred_full_sl"] = -pd.to_numeric(out["pred_full_sl"], errors="coerce")
    else:
        out["neg_pred_full_sl"] = np.nan
    out["mean_minus_fullsl_timeout"] = (
        pd.to_numeric(out.get("pred_mean_utility"), errors="coerce")
        - float(sl_weight) * pd.to_numeric(out.get("pred_full_sl"), errors="coerce")
        - float(timeout_weight) * pd.to_numeric(out.get("pred_timeout"), errors="coerce")
    )
    return out


def build_head_priority_schedule(
    predictions: pd.DataFrame,
    *,
    arm: str,
    min_rank: float,
    max_adjustment: float,
    max_priority_multiplier: float = 1.0,
    max_rank_adjustment: float = 0.0,
    priority_action: str = "adjustment",
    sl_weight: float = 0.10,
    timeout_weight: float = 0.03,
    min_rows_per_head_timestamp: int = 1,
) -> pd.DataFrame:
    """Build a bounded timestamp x head auction-priority schedule."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown priority arm: {arm}")
    spec = ARM_SPECS[arm]
    work = _prediction_feature_frame(
        predictions,
        sl_weight=float(sl_weight),
        timeout_weight=float(timeout_weight),
    )
    rank = pd.to_numeric(work.get("_rank"), errors="coerce")
    if np.isfinite(float(min_rank)):
        work = work.loc[(rank >= float(min_rank)).fillna(False)].copy()
    if spec["kind"] == "single":
        score_col = str(spec["column"])
    elif spec["kind"] == "composite":
        score_col = "mean_minus_fullsl_timeout"
    else:
        raise ValueError(f"unsupported arm kind: {spec['kind']}")
    if score_col not in work.columns:
        raise ValueError(f"missing controller prediction column for {arm}: {score_col}")
    work["_raw_head_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["timestamp", "head", "_raw_head_score"])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "head",
                "raw_head_score",
                "head_rows",
                "centered_head_score",
                "portfolio_priority_adjustment",
                "portfolio_priority_multiplier",
                "portfolio_rank_adjustment",
                "priority_arm",
            ]
        )
    grouped = (
        work.groupby(["timestamp", "head"], observed=True)
        .agg(raw_head_score=("_raw_head_score", "median"), head_rows=("_raw_head_score", "size"))
        .reset_index()
    )
    grouped = grouped.loc[
        pd.to_numeric(grouped["head_rows"], errors="coerce").fillna(0)
        >= int(min_rows_per_head_timestamp)
    ].copy()
    if grouped.empty:
        return grouped
    grouped["_timestamp_mean_score"] = grouped.groupby("timestamp", observed=True)[
        "raw_head_score"
    ].transform("mean")
    grouped["centered_head_score"] = (
        pd.to_numeric(grouped["raw_head_score"], errors="coerce")
        - pd.to_numeric(grouped["_timestamp_mean_score"], errors="coerce")
    )
    centered = pd.to_numeric(grouped["centered_head_score"], errors="coerce").to_numpy(dtype=float)
    finite = centered[np.isfinite(centered)]
    if finite.size:
        scale = float(np.nanpercentile(np.abs(finite), 75))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanstd(finite))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
    else:
        scale = 1.0
    adjustment, multiplier = priority_action_values(
        grouped["centered_head_score"],
        scale=scale,
        max_adjustment=float(max_adjustment),
        max_priority_multiplier=float(max_priority_multiplier),
        priority_action=str(priority_action),
    )
    rank_adjustment = rank_prior_values(
        grouped["centered_head_score"],
        scale=scale,
        max_rank_adjustment=float(max_rank_adjustment),
    )
    grouped["portfolio_priority_adjustment"] = adjustment
    grouped["portfolio_priority_multiplier"] = multiplier
    grouped["portfolio_rank_adjustment"] = rank_adjustment
    grouped["priority_arm"] = arm
    grouped["score_column"] = score_col
    grouped["priority_scale"] = scale
    grouped["priority_action"] = str(priority_action)
    return grouped[
        [
            "timestamp",
            "head",
            "raw_head_score",
            "head_rows",
            "centered_head_score",
            "portfolio_priority_adjustment",
            "portfolio_priority_multiplier",
            "portfolio_rank_adjustment",
            "priority_arm",
            "score_column",
            "priority_scale",
            "priority_action",
        ]
    ].reset_index(drop=True)


def apply_head_priority_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    fail_closed: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = candidates.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if schedule.empty:
        if fail_closed and not out.empty:
            raise ValueError("empty priority schedule")
        out["portfolio_priority_adjustment"] = 0.0
        return out, {"missing_rows": int(len(out)), "coverage": 0.0}
    cols = ["timestamp", "head", "portfolio_priority_adjustment", "priority_arm"]
    if "portfolio_priority_multiplier" in schedule.columns:
        cols.append("portfolio_priority_multiplier")
    if "portfolio_rank_adjustment" in schedule.columns:
        cols.append("portfolio_rank_adjustment")
    sched = schedule[cols].copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    if sched.duplicated(["timestamp", "head"]).any():
        raise ValueError("priority schedule has duplicate timestamp/head rows")
    out = out.merge(sched, on=["timestamp", "head"], how="left", validate="many_to_one")
    adj = pd.to_numeric(out["portfolio_priority_adjustment"], errors="coerce")
    mult = (
        pd.to_numeric(out["portfolio_priority_multiplier"], errors="coerce")
        if "portfolio_priority_multiplier" in out.columns
        else pd.Series(1.0, index=out.index)
    )
    rank_adj = (
        pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce")
        if "portfolio_rank_adjustment" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    missing = int(adj.isna().sum())
    missing_mult = int(mult.isna().sum())
    missing_rank = int(rank_adj.isna().sum())
    if (missing or missing_mult or missing_rank) and fail_closed:
        raise ValueError(
            f"missing priority schedule values for {max(missing, missing_mult, missing_rank)} candidate rows"
        )
    out["portfolio_priority_adjustment"] = adj.fillna(0.0)
    out["portfolio_priority_multiplier"] = mult.fillna(1.0).clip(lower=0.0)
    out["portfolio_rank_adjustment"] = rank_adj.fillna(0.0).clip(lower=-1.0, upper=1.0)
    return out, {
        "rows": int(len(out)),
        "missing_rows": int(max(missing, missing_mult, missing_rank)),
        "coverage": float(1.0 - max(missing, missing_mult, missing_rank) / max(len(out), 1)),
        "mean_adjustment": float(out["portfolio_priority_adjustment"].mean()) if len(out) else 0.0,
        "min_adjustment": float(out["portfolio_priority_adjustment"].min()) if len(out) else 0.0,
        "max_adjustment": float(out["portfolio_priority_adjustment"].max()) if len(out) else 0.0,
        "mean_multiplier": float(out["portfolio_priority_multiplier"].mean()) if len(out) else 1.0,
        "min_multiplier": float(out["portfolio_priority_multiplier"].min()) if len(out) else 1.0,
        "max_multiplier": float(out["portfolio_priority_multiplier"].max()) if len(out) else 1.0,
        "mean_rank_adjustment": float(out["portfolio_rank_adjustment"].mean()) if len(out) else 0.0,
        "min_rank_adjustment": float(out["portfolio_rank_adjustment"].min()) if len(out) else 0.0,
        "max_rank_adjustment": float(out["portfolio_rank_adjustment"].max()) if len(out) else 0.0,
    }


def _replay(
    *,
    arm: str,
    candidates: pd.DataFrame,
    train_deployable: pd.DataFrame,
    params: Any,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = mstc._accepted_trades(candidates, decisions)
    if not accepted.empty:
        accepted["arm"] = arm
    summary = pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule=None)])
    by_head = mstc._by_head(arm, accepted)
    return decisions, equity, metrics, accepted, pd.concat([summary, by_head], axis=0, ignore_index=True)


def _summary_by_head(arm: str, accepted: pd.DataFrame) -> pd.DataFrame:
    return mstc._by_head(arm, accepted)


def _decision_key(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in ("timestamp", "symbol", "strategy_id", "side", "head") if col in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df)), index=df.index).astype(str)
    values: list[pd.Series] = []
    for col in cols:
        if col == "timestamp":
            values.append(pd.to_datetime(df[col], utc=True, errors="coerce").astype(str))
        else:
            values.append(df[col].astype(str))
    out = values[0]
    for value in values[1:]:
        out = out.str.cat(value, sep="|")
    return out


def _accepted_overlap(accepted: dict[str, pd.DataFrame], baseline_arm: str) -> pd.DataFrame:
    base = accepted.get(baseline_arm, pd.DataFrame())
    base_keys = set(_decision_key(base)) if not base.empty else set()
    rows = []
    for arm, frame in accepted.items():
        keys = set(_decision_key(frame)) if not frame.empty else set()
        union = base_keys | keys
        inter = base_keys & keys
        rows.append(
            {
                "arm": arm,
                "baseline_accepted": int(len(base_keys)),
                "arm_accepted": int(len(keys)),
                "intersection": int(len(inter)),
                "union": int(len(union)),
                "jaccard_vs_baseline": float(len(inter) / len(union)) if union else 1.0,
                "baseline_only": int(len(base_keys - keys)),
                "arm_only": int(len(keys - base_keys)),
            }
        )
    return pd.DataFrame(rows)


def _head_delta(summary_by_head: pd.DataFrame, baseline_arm: str) -> pd.DataFrame:
    if summary_by_head.empty:
        return pd.DataFrame()
    base = summary_by_head.loc[summary_by_head["arm"].eq(baseline_arm)].copy()
    rows = []
    for _, row in summary_by_head.iterrows():
        head = str(row.get("head"))
        base_row = base.loc[base["head"].astype(str).eq(head)]
        rec: dict[str, Any] = {"arm": row.get("arm"), "head": head}
        if base_row.empty:
            for col in ("trade_count", "net_pnl", "gross_pnl", "cost_pnl", "full_sl_rate", "timeout_rate"):
                rec[f"delta_{col}"] = float(pd.to_numeric(pd.Series([row.get(col, 0.0)]), errors="coerce").iloc[0])
        else:
            b = base_row.iloc[0]
            for col in ("trade_count", "net_pnl", "gross_pnl", "cost_pnl", "full_sl_rate", "timeout_rate"):
                rv = pd.to_numeric(pd.Series([row.get(col, 0.0)]), errors="coerce").iloc[0]
                bv = pd.to_numeric(pd.Series([b.get(col, 0.0)]), errors="coerce").iloc[0]
                rec[f"delta_{col}"] = float(rv - bv)
        rows.append(rec)
    return pd.DataFrame(rows)


def _render_report(
    *,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    overlap: pd.DataFrame,
    head_delta: pd.DataFrame,
    swap_attribution: pd.DataFrame,
    schedule_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Market-State Head-Priority Modulation",
        "",
        "This ablation keeps the T1 candidate universe, scores, thresholds, sizing, q-fail state, and HeadHealth state fixed.",
        (
            "It applies bounded pre-filter rank-prior adjustments plus auction-priority actions by head and timestamp."
            if dict(manifest.get("contract") or {}).get("rank_prior_layer") == "pre_filter_head_prior"
            else "Only bounded priority action columns change the global auction ordering by head and timestamp."
        ),
        "",
        "## Contract",
        "",
        f"- Candidates: `{manifest['inputs']['candidates']}`",
        f"- Controller predictions: `{manifest['inputs']['controller_predictions']}`",
        f"- Market-state backend: `{manifest['market_state_backend']}`",
        f"- Candidate rows: `{manifest['candidate_universe']['rows']}`",
        f"- Timestamp range: `{manifest['candidate_universe']['timestamp_min']}` to `{manifest['candidate_universe']['timestamp_max']}`",
        f"- q-fail active: `{manifest['contract']['qfail_active']}`",
        f"- HeadHealth active: `{manifest['contract']['head_health_active']}`",
        f"- Threshold controller active: `{manifest['contract']['market_state_threshold_controller_active']}`",
        f"- Rank-prior layer: `{manifest['contract'].get('rank_prior_layer', 'disabled')}`",
        "",
        "## Replay Summary",
        "",
        "| arm | trades | net_pnl | gross_pnl | cost_pnl | full_sl_rate | timeout_rate | worst_24h_net_pnl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['trade_count'])} | "
            f"{float(row['net_pnl']):.6f} | {float(row['gross_pnl']):.6f} | "
            f"{float(row['cost_pnl']):.6f} | {float(row['full_sl_rate']):.6f} | "
            f"{float(row['timeout_rate']):.6f} | {float(row['worst_24h_net_pnl']):.6f} |"
        )
    lines.extend(["", "## By Head", ""])
    if by_head.empty:
        lines.append("No accepted trades.")
    else:
        lines.extend(
            [
                "| arm | head | trades | win_rate | net_pnl | gross_pnl | full_sl_rate | timeout_rate |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in by_head.iterrows():
            lines.append(
                f"| {row['arm']} | {row['head']} | {int(row['trade_count'])} | "
                f"{float(row['win_rate']):.6f} | {float(row['net_pnl']):.6f} | "
                f"{float(row['gross_pnl']):.6f} | {float(row['full_sl_rate']):.6f} | "
                f"{float(row['timeout_rate']):.6f} |"
            )
    lines.extend(["", "## Accepted Overlap Vs Baseline", ""])
    lines.append(overlap.to_markdown(index=False) if not overlap.empty else "_No overlap rows._")
    lines.extend(["", "## Accepted Swap Utility Vs Baseline", ""])
    if swap_attribution.empty:
        lines.append("_No accepted-set swaps versus baseline._")
    else:
        view_cols = [
            "arm",
            "scope",
            "scope_value",
            "entrants",
            "removed",
            "entrant_net_pnl",
            "removed_net_pnl",
            "net_replacement_pnl",
            "same_key_net_pnl_delta",
            "net_action_pnl_delta",
            "removed_loss_avoided",
            "removed_winner_pnl_sacrificed",
            "defensive_success",
        ]
        view = swap_attribution[[c for c in view_cols if c in swap_attribution.columns]].copy()
        lines.append(view.to_markdown(index=False))
    lines.extend(["", "## By-Head Deltas Vs Baseline", ""])
    lines.append(head_delta.to_markdown(index=False) if not head_delta.empty else "_No head deltas._")
    lines.extend(["", "## Priority Schedule Summary", ""])
    lines.append(schedule_summary.to_markdown(index=False) if not schedule_summary.empty else "_No schedule rows._")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--controller-predictions", type=Path)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--min-rank", type=float, default=0.70)
    parser.add_argument("--max-adjustment", type=float, default=0.20)
    parser.add_argument("--max-priority-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--max-rank-adjustment",
        type=float,
        default=0.0,
        help="Optional bounded pre-filter rank-prior adjustment emitted from market state.",
    )
    parser.add_argument(
        "--priority-action",
        choices=sorted(PRIORITY_ACTIONS),
        default="adjustment",
    )
    parser.add_argument("--sl-weight", type=float, default=0.10)
    parser.add_argument("--timeout-weight", type=float, default=0.03)
    parser.add_argument("--min-rows-per-head-timestamp", type=int, default=1)
    parser.add_argument("--allow-missing-schedule", action="store_true")
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    candidates_path = args.candidates or (args.score_dir / "controller_scored_candidates.parquet")
    predictions_path = args.controller_predictions or (args.score_dir / "controller_predictions.parquet")
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(candidates_path)
    train_deployable = _load_candidates(args.train_deployable_candidates)
    predictions = _load_controller_predictions(predictions_path)
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)

    requested_arms = list(ARM_SPECS)
    if str(args.arms).strip().lower() != "all":
        requested_arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    unknown = sorted(set(requested_arms).difference(ARM_SPECS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")

    accepted_by_arm: dict[str, pd.DataFrame] = {}
    summary_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    base_decisions, base_equity, base_metrics, base_accepted, _ = _replay(
        arm=BASELINE_ARM,
        candidates=candidates.assign(
            portfolio_priority_adjustment=0.0,
            portfolio_priority_multiplier=1.0,
            portfolio_rank_adjustment=0.0,
        ),
        train_deployable=train_deployable,
        params=params,
        market_mode=str(args.market_mode),
    )
    accepted_by_arm[BASELINE_ARM] = base_accepted
    summary_frames.append(pd.DataFrame([mstc._metrics_row(BASELINE_ARM, base_metrics, base_accepted, schedule=None)]))
    by_head_frames.append(_summary_by_head(BASELINE_ARM, base_accepted))
    base_decisions.to_parquet(args.output_dir / f"{BASELINE_ARM}_decisions.parquet", index=False)
    base_equity.to_parquet(args.output_dir / f"{BASELINE_ARM}_equity.parquet", index=False)
    base_accepted.to_parquet(args.output_dir / f"{BASELINE_ARM}_accepted_trades.parquet", index=False)

    for arm in requested_arms:
        schedule = build_head_priority_schedule(
            predictions,
            arm=arm,
            min_rank=float(args.min_rank),
            max_adjustment=float(args.max_adjustment),
            max_priority_multiplier=float(args.max_priority_multiplier),
            max_rank_adjustment=float(args.max_rank_adjustment),
            priority_action=str(args.priority_action),
            sl_weight=float(args.sl_weight),
            timeout_weight=float(args.timeout_weight),
            min_rows_per_head_timestamp=int(args.min_rows_per_head_timestamp),
        )
        arm_candidates, coverage = apply_head_priority_schedule(
            candidates,
            schedule,
            fail_closed=not bool(args.allow_missing_schedule),
        )
        coverage["arm"] = arm
        coverage_rows.append(coverage)
        schedule_frames.append(schedule)
        decisions, equity, metrics, accepted, _ = _replay(
            arm=arm,
            candidates=arm_candidates,
            train_deployable=train_deployable,
            params=params,
            market_mode=str(args.market_mode),
        )
        accepted_by_arm[arm] = accepted
        summary_frames.append(pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule=None)]))
        by_head_frames.append(_summary_by_head(arm, accepted))
        arm_candidates.to_parquet(args.output_dir / f"{arm}_candidates.parquet", index=False)
        decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
        equity.to_parquet(args.output_dir / f"{arm}_equity.parquet", index=False)
        accepted.to_parquet(args.output_dir / f"{arm}_accepted_trades.parquet", index=False)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    schedules = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    overlap = _accepted_overlap(accepted_by_arm, BASELINE_ARM)
    accepted_all = (
        pd.concat([frame for frame in accepted_by_arm.values() if frame is not None], ignore_index=True)
        if accepted_by_arm
        else pd.DataFrame()
    )
    swap_attribution = mstc._threshold_action_utility(accepted_all, BASELINE_ARM)
    head_delta = _head_delta(by_head, BASELINE_ARM)
    coverage = pd.DataFrame(coverage_rows)
    schedule_summary = (
        schedules.groupby(["priority_arm", "head"], observed=True)
        .agg(
            schedule_rows=("portfolio_priority_adjustment", "size"),
            mean_adjustment=("portfolio_priority_adjustment", "mean"),
            min_adjustment=("portfolio_priority_adjustment", "min"),
            max_adjustment=("portfolio_priority_adjustment", "max"),
            mean_multiplier=("portfolio_priority_multiplier", "mean"),
            min_multiplier=("portfolio_priority_multiplier", "min"),
            max_multiplier=("portfolio_priority_multiplier", "max"),
            mean_rank_adjustment=("portfolio_rank_adjustment", "mean"),
            min_rank_adjustment=("portfolio_rank_adjustment", "min"),
            max_rank_adjustment=("portfolio_rank_adjustment", "max"),
            mean_raw_head_score=("raw_head_score", "mean"),
        )
        .reset_index()
        if not schedules.empty
        else pd.DataFrame()
    )

    summary.to_csv(args.output_dir / "head_priority_replay_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "head_priority_by_head.csv", index=False)
    overlap.to_csv(args.output_dir / "head_priority_accepted_overlap.csv", index=False)
    swap_attribution.to_csv(args.output_dir / "head_priority_accepted_swap_utility.csv", index=False)
    head_delta.to_csv(args.output_dir / "head_priority_by_head_delta.csv", index=False)
    schedules.to_parquet(args.output_dir / "head_priority_schedule.parquet", index=False)
    coverage.to_csv(args.output_dir / "head_priority_schedule_coverage.csv", index=False)
    schedule_summary.to_csv(args.output_dir / "head_priority_schedule_summary.csv", index=False)

    score_manifest = _load_json(args.score_dir / "manifest.json")
    manifest = {
        "generated_by": "run_market_state_head_priority_modulation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "market_state_based_head_priority_modulation_global_portfolio_ablation",
        "market_state_backend": score_manifest.get("forecast_model_kind")
        or score_manifest.get("forecast_model_kind_resolution", {}).get("value")
        or "unknown",
        "contract": {
            "changes_scores_or_ranks": bool(abs(float(args.max_rank_adjustment)) > 0.0),
            "changes_thresholds": False,
            "changes_position_sizing": False,
            "changes_auction_ordering": True,
            "rank_prior_layer": (
                "pre_filter_head_prior" if abs(float(args.max_rank_adjustment)) > 0.0 else "disabled"
            ),
            "qfail_active": False,
            "head_health_active": False,
            "market_state_threshold_controller_active": False,
            "operational_status": "shadow_only",
            "execution_enabled": False,
            "production_eligible": False,
            "requires_promotion_gate": True,
            "market_state_encoder_uses_candidate_features": False,
            "priority_adjustment_column": "portfolio_priority_adjustment",
            "priority_multiplier_column": "portfolio_priority_multiplier",
            "rank_adjustment_column": "portfolio_rank_adjustment",
            "priority_action": str(args.priority_action),
        },
        "params": {
            "arms": requested_arms,
            "min_rank": float(args.min_rank),
            "max_adjustment": float(args.max_adjustment),
            "max_priority_multiplier": float(args.max_priority_multiplier),
            "max_rank_adjustment": float(args.max_rank_adjustment),
            "priority_action": str(args.priority_action),
            "sl_weight": float(args.sl_weight),
            "timeout_weight": float(args.timeout_weight),
            "min_rows_per_head_timestamp": int(args.min_rows_per_head_timestamp),
            "allow_missing_schedule": bool(args.allow_missing_schedule),
        },
        "inputs": {
            "score_dir": str(args.score_dir),
            "score_dir_manifest": str(args.score_dir / "manifest.json"),
            "score_dir_manifest_sha256": _sha256(args.score_dir / "manifest.json"),
            "candidates": str(candidates_path),
            "candidates_sha256": _sha256(candidates_path),
            "controller_predictions": str(predictions_path),
            "controller_predictions_sha256": _sha256(predictions_path),
            "train_deployable_candidates": str(args.train_deployable_candidates),
            "train_deployable_candidates_sha256": _sha256(args.train_deployable_candidates),
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": _sha256(args.policy_manifest),
            "policy_manifest_run_id": policy_payload.get("run_id"),
        },
        "candidate_universe": _candidate_universe_signature(candidates),
        "prediction_rows": int(len(predictions)),
        "summary": summary.to_dict("records"),
        "by_head": by_head.to_dict("records"),
        "accepted_swap_utility": swap_attribution.to_dict("records"),
        "schedule_coverage": coverage.to_dict("records"),
        "outputs": {
            "summary": str(args.output_dir / "head_priority_replay_summary.csv"),
            "by_head": str(args.output_dir / "head_priority_by_head.csv"),
            "overlap": str(args.output_dir / "head_priority_accepted_overlap.csv"),
            "accepted_swap_utility": str(args.output_dir / "head_priority_accepted_swap_utility.csv"),
            "head_delta": str(args.output_dir / "head_priority_by_head_delta.csv"),
            "schedule": str(args.output_dir / "head_priority_schedule.parquet"),
            "schedule_coverage": str(args.output_dir / "head_priority_schedule_coverage.csv"),
            "schedule_summary": str(args.output_dir / "head_priority_schedule_summary.csv"),
            "report": str(args.output_dir / "market_state_head_priority_modulation_report.md"),
            "manifest": str(args.output_dir / "manifest.json"),
        },
    }
    report = _render_report(
        manifest=manifest,
        summary=summary,
        by_head=by_head,
        overlap=overlap,
        head_delta=head_delta,
        swap_attribution=swap_attribution,
        schedule_summary=schedule_summary,
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "market_state_head_priority_modulation_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(args.output_dir),
                    "summary": summary.to_dict("records"),
                    "schedule_coverage": coverage.to_dict("records"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
