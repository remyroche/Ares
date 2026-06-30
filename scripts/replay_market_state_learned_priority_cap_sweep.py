#!/usr/bin/env python3
"""Replay a learned market-state head-priority schedule under smaller caps."""

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

from scripts.run_market_state_head_priority_learning import (  # noqa: E402
    BASELINE_ARM,
    _accepted_overlap,
    _load_candidates,
    _load_json,
    _load_static_baseline_artifacts,
    _replay_arm,
    load_train_deployable_for_static_contract,
    replay_selection_metrics,
)
from scripts.run_market_state_head_priority_modulation import (  # noqa: E402
    apply_head_priority_schedule,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_s2_replayaware_forced_shadow_20260626_jun15_22_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_head_priority_cap_sweep_s2_20260626_jun15_22_v1"
)


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


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid cap value: {text!r}")
        values.append(value)
    if not values:
        raise ValueError("at least one cap is required")
    return sorted(set(float(v) for v in values))


def _arm_for_cap(base_arm: str, cap: float) -> str:
    safe = f"{float(cap):.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{base_arm}_cap_{safe}"


def rescale_learned_schedule(
    schedule: pd.DataFrame,
    *,
    max_adjustment: float,
    arm: str,
    min_abs_z: float = 0.0,
) -> pd.DataFrame:
    """Recompute bounded priority adjustment from stored raw learned scores."""
    if schedule.empty:
        raise ValueError("learned priority schedule is empty")
    required = {"timestamp", "head", "centered_head_score", "priority_scale"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError(f"learned priority schedule missing required columns: {missing}")
    out = schedule.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    scale = pd.to_numeric(out["priority_scale"], errors="coerce").replace(0.0, np.nan)
    centered = pd.to_numeric(out["centered_head_score"], errors="coerce")
    z_score = centered / scale
    adjustment = pd.Series(float(max_adjustment) * np.tanh(z_score), index=out.index)
    threshold = max(float(min_abs_z), 0.0)
    if threshold > 0.0:
        adjustment = adjustment.where(z_score.abs() >= threshold, 0.0)
    adjustment = adjustment - adjustment.groupby(out["timestamp"], dropna=False).transform("mean")
    out["priority_z_score"] = z_score
    out["min_abs_priority_z"] = threshold
    out["portfolio_priority_adjustment"] = adjustment.clip(
        lower=-abs(float(max_adjustment)),
        upper=abs(float(max_adjustment)),
    )
    # Learned schedules can contain multiple action channels.  A cap sweep is
    # intended to isolate the additive auction-priority channel only, so stale
    # multiplier/rank-prior values from the source schedule must be neutralized.
    out["portfolio_priority_multiplier"] = 1.0
    out["portfolio_rank_adjustment"] = 0.0
    if bool(out["portfolio_priority_adjustment"].isna().any()):
        raise ValueError("rescaled learned priority schedule produced non-finite adjustments")
    out["priority_arm"] = str(arm)
    return out


def select_shadow_challenger(
    metrics: pd.DataFrame,
    *,
    min_delta_net_pnl: float = 0.0,
    near_best_abs_tolerance: float = 1.0,
    near_best_rel_tolerance: float = 0.05,
    min_accepted_jaccard: float = 0.95,
    max_full_sl_delta: float = 0.005,
    max_timeout_delta: float = 0.0,
) -> dict[str, Any]:
    """Select the conservative bounded shadow challenger from a cap sweep.

    The cap sweep is an attribution surface, not a production optimiser.  When
    multiple rows pass replay gates, first prefer arms that keep the accepted
    set close to the static baseline and avoid worsening execution-risk rates.
    Within that safer surface, prefer the least aggressive action among rows
    whose PnL delta is practically tied with the best passing row.  This keeps
    market-state routing from becoming a June-specific PnL maximizer.
    """
    if metrics.empty:
        return {
            "selected": False,
            "reason": "empty_metrics",
            "arm": None,
        }
    required = {"arm", "gate_passed", "delta_net_pnl", "max_adjustment", "min_abs_z"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        return {
            "selected": False,
            "reason": "missing_required_columns",
            "missing_columns": missing,
            "arm": None,
        }
    work = metrics.copy()
    gate = work["gate_passed"].astype(str).str.lower().isin({"true", "1", "yes"})
    delta = pd.to_numeric(work["delta_net_pnl"], errors="coerce")
    work = work.loc[gate & (delta > float(min_delta_net_pnl))].copy()
    if work.empty:
        return {
            "selected": False,
            "reason": "no_gate_passing_positive_delta_arm",
            "arm": None,
            "gate_passing_count": 0,
        }
    work["_delta"] = pd.to_numeric(work["delta_net_pnl"], errors="coerce")
    numeric_cols = [
        "max_adjustment",
        "min_abs_z",
        "active_schedule_share",
        "accepted_jaccard",
        "delta_full_sl_rate",
        "delta_timeout_rate",
        "net_replacement_pnl",
        "net_action_pnl_delta",
        "defensive_success",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    safe = work.copy()
    safe_filters: dict[str, Any] = {
        "min_accepted_jaccard": float(min_accepted_jaccard),
        "max_full_sl_delta": float(max_full_sl_delta),
        "max_timeout_delta": float(max_timeout_delta),
    }
    if "accepted_jaccard" in safe.columns and np.isfinite(float(min_accepted_jaccard)):
        safe = safe.loc[safe["accepted_jaccard"] >= float(min_accepted_jaccard)].copy()
    if "delta_full_sl_rate" in safe.columns and np.isfinite(float(max_full_sl_delta)):
        safe = safe.loc[safe["delta_full_sl_rate"] <= float(max_full_sl_delta)].copy()
    if "delta_timeout_rate" in safe.columns and np.isfinite(float(max_timeout_delta)):
        safe = safe.loc[safe["delta_timeout_rate"] <= float(max_timeout_delta)].copy()
    selection_pool = safe if not safe.empty else work
    pool_reason = "risk_safe_gate_passing_positive_delta_arm" if not safe.empty else "fallback_gate_passing_positive_delta_arm"

    best_delta = float(selection_pool["_delta"].max())
    tolerance = max(float(near_best_abs_tolerance), abs(best_delta) * float(near_best_rel_tolerance))
    near_best = selection_pool.loc[selection_pool["_delta"] >= best_delta - tolerance].copy()
    sort_plan = [
        ("max_adjustment", True),
        ("min_abs_z", True),
        ("active_schedule_share", True),
        ("_delta", False),
        ("accepted_jaccard", False),
    ]
    sort_cols = [col for col, _ascending in sort_plan if col in near_best.columns]
    sort_ascending = [_ascending for col, _ascending in sort_plan if col in near_best.columns]
    near_best = near_best.sort_values(
        sort_cols,
        ascending=sort_ascending,
        na_position="last",
    )
    selected = near_best.iloc[0].drop(labels=["_delta"], errors="ignore").to_dict()
    return {
        "selected": True,
        "reason": f"selected_conservative_near_best_{pool_reason}",
        "arm": str(selected.get("arm")),
        "best_delta_net_pnl": best_delta,
        "near_best_tolerance": tolerance,
        "gate_passing_count": int(len(work)),
        "risk_safe_gate_passing_count": int(len(safe)),
        "near_best_count": int(len(near_best)),
        "selection_policy": {
            "min_delta_net_pnl": float(min_delta_net_pnl),
            "near_best_abs_tolerance": float(near_best_abs_tolerance),
            "near_best_rel_tolerance": float(near_best_rel_tolerance),
            "risk_safe_filters": safe_filters,
            "tie_break_order": [
                "risk-safe rows before fallback rows",
                "lower max_adjustment",
                "lower min_abs_z",
                "lower active_schedule_share",
                "higher delta_net_pnl",
                "higher accepted_jaccard",
            ],
        },
        "selected_row": _json_safe(selected),
    }


def _render_report(
    *,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    by_head: pd.DataFrame,
    overlap: pd.DataFrame,
    swap: pd.DataFrame,
    selected: dict[str, Any],
) -> str:
    lines = [
        "# Learned Market-State Head-Priority Cap Sweep",
        "",
        "This replay keeps the learned priority model fixed and varies only the maximum `portfolio_priority_adjustment` cap.",
        "",
        "## Contract",
        "",
        f"- Priority dir: `{manifest['inputs']['priority_dir']}`",
        f"- Candidates: `{manifest['inputs']['candidates']}`",
        f"- Caps: `{manifest['params']['caps']}`",
        "- Scores, rank references, thresholds, sizing, q-fail, HeadHealth and threshold controller remain unchanged.",
        "",
        "## Cap Metrics",
        "",
    ]
    view_cols = [
        "arm",
        "max_adjustment",
        "min_abs_z",
        "active_schedule_share",
        "trade_count",
        "net_pnl",
        "delta_net_pnl",
        "full_sl_rate",
        "delta_full_sl_rate",
        "timeout_rate",
        "delta_timeout_rate",
        "accepted_jaccard",
        "entrants",
        "removed",
        "net_replacement_pnl",
        "net_action_pnl_delta",
        "gate_passed",
    ]
    lines.append(metrics[[c for c in view_cols if c in metrics.columns]].to_markdown(index=False))
    lines.extend(["", "## Selected Shadow Challenger", ""])
    if selected.get("selected"):
        row = dict(selected.get("selected_row") or {})
        selection_view = {
            "arm": selected.get("arm"),
            "max_adjustment": row.get("max_adjustment"),
            "min_abs_z": row.get("min_abs_z"),
            "delta_net_pnl": row.get("delta_net_pnl"),
            "delta_full_sl_rate": row.get("delta_full_sl_rate"),
            "delta_timeout_rate": row.get("delta_timeout_rate"),
            "accepted_jaccard": row.get("accepted_jaccard"),
            "entrants": row.get("entrants"),
            "removed": row.get("removed"),
            "gate_passing_count": selected.get("gate_passing_count"),
            "risk_safe_gate_passing_count": selected.get("risk_safe_gate_passing_count"),
            "near_best_count": selected.get("near_best_count"),
        }
        lines.append(pd.DataFrame([selection_view]).to_markdown(index=False))
        lines.append("")
        lines.append(
            "Selection is conservative among risk-safe, gate-passing near-best rows; it does not activate production routing."
        )
    else:
        lines.append(f"No selected challenger: `{selected.get('reason')}`.")
    lines.extend(["", "## By Head", ""])
    lines.append(by_head.to_markdown(index=False) if not by_head.empty else "_No by-head rows._")
    lines.extend(["", "## Accepted Overlap", ""])
    lines.append(overlap.to_markdown(index=False) if not overlap.empty else "_No overlap rows._")
    lines.extend(["", "## Accepted Swap Utility", ""])
    lines.append(swap.to_markdown(index=False) if not swap.empty else "_No swap rows._")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--caps", default="0.02,0.03,0.05,0.08,0.10,0.15,0.20")
    parser.add_argument("--min-abs-z-thresholds", default="0.0")
    parser.add_argument(
        "--selection-gate-mode",
        choices=["defensive", "opportunity"],
        default="defensive",
        help=(
            "Replay gate used for cap selection. Use 'opportunity' when this "
            "priority schedule is evaluated as cross-head allocation rather "
            "than threshold suppression."
        ),
    )
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument(
        "--static-baseline-manifest",
        type=Path,
        help=(
            "Optional materialized T1 manifest used to load the static P0 "
            "baseline exactly instead of recomputing it from candidates."
        ),
    )
    parser.add_argument("--selection-min-accepted-jaccard", type=float, default=0.95)
    parser.add_argument("--selection-max-full-sl-delta", type=float, default=0.005)
    parser.add_argument("--selection-max-timeout-delta", type=float, default=0.0)
    args = parser.parse_args()

    priority_dir = args.priority_dir
    manifest = _load_json(priority_dir / "manifest.json")
    inputs = dict(manifest.get("inputs") or {})
    candidates_path = Path(inputs.get("candidates") or priority_dir / "L0_selected_lgbm_priority_candidates.parquet")
    train_deployable_path = Path(inputs.get("train_deployable_candidates") or "")
    policy_manifest_path = Path(inputs.get("policy_manifest") or "")
    schedule_path = priority_dir / "head_priority_learned_schedule.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    if not train_deployable_path.exists():
        raise FileNotFoundError(train_deployable_path)
    if not policy_manifest_path.exists():
        raise FileNotFoundError(policy_manifest_path)
    if not schedule_path.exists():
        raise FileNotFoundError(schedule_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    caps = _parse_float_list(args.caps)
    min_abs_z_thresholds = _parse_float_list(args.min_abs_z_thresholds)
    base_priority_arm = str(
        pd.read_parquet(schedule_path)["priority_arm"].dropna().astype(str).iloc[0]
    )
    learned_schedule = pd.read_parquet(schedule_path)
    candidates = _load_candidates(candidates_path)
    train_deployable, train_deployable_contract = load_train_deployable_for_static_contract(
        train_deployable_path,
        static_baseline_manifest=args.static_baseline_manifest,
    )
    params, policy_payload = mstc._load_policy_params(policy_manifest_path, str(args.policy_variant))

    accepted_by_arm: dict[str, pd.DataFrame] = {}
    summary_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []

    static_baseline_info: dict[str, Any] | None = None
    static_baseline = _load_static_baseline_artifacts(
        args.static_baseline_manifest,
        arm=BASELINE_ARM,
    )
    if static_baseline is None:
        base_candidates = candidates.assign(portfolio_priority_adjustment=0.0)
        base_decisions, base_equity, base_accepted, base_summary, base_by_head = _replay_arm(
            arm=BASELINE_ARM,
            candidates=base_candidates,
            train_deployable=train_deployable,
            params=params,
            market_mode=str(args.market_mode),
        )
    else:
        (
            base_decisions,
            base_equity,
            base_accepted,
            base_summary,
            base_by_head,
            static_baseline_info,
        ) = static_baseline
    accepted_by_arm[BASELINE_ARM] = base_accepted
    summary_frames.append(base_summary)
    by_head_frames.append(base_by_head)
    base_decisions.to_parquet(args.output_dir / f"{BASELINE_ARM}_decisions.parquet", index=False)
    base_equity.to_parquet(args.output_dir / f"{BASELINE_ARM}_equity.parquet", index=False)
    base_accepted.to_parquet(args.output_dir / f"{BASELINE_ARM}_accepted_trades.parquet", index=False)

    metric_rows: list[dict[str, Any]] = []
    base_row = base_summary.iloc[0].to_dict()
    for cap in caps:
        for min_abs_z in min_abs_z_thresholds:
            arm = _arm_for_cap(base_priority_arm, cap)
            if float(min_abs_z) > 0.0:
                z_safe = f"{float(min_abs_z):.3f}".rstrip("0").rstrip(".").replace(".", "p")
                arm = f"{arm}_zge_{z_safe}"
            schedule = rescale_learned_schedule(
                learned_schedule,
                max_adjustment=float(cap),
                min_abs_z=float(min_abs_z),
                arm=arm,
            )
            arm_candidates, coverage = apply_head_priority_schedule(
                candidates,
                schedule,
                fail_closed=True,
            )
            decisions, equity, accepted, summary_part, by_head_part = _replay_arm(
                arm=arm,
                candidates=arm_candidates,
                train_deployable=train_deployable,
                params=params,
                market_mode=str(args.market_mode),
            )
            accepted_by_arm[arm] = accepted
            summary_frames.append(summary_part)
            by_head_frames.append(by_head_part)
            schedule_frames.append(schedule)
            arm_candidates.to_parquet(args.output_dir / f"{arm}_candidates.parquet", index=False)
            decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
            equity.to_parquet(args.output_dir / f"{arm}_equity.parquet", index=False)
            accepted.to_parquet(args.output_dir / f"{arm}_accepted_trades.parquet", index=False)

            replay_metrics = replay_selection_metrics(
                arm=arm,
                candidate_summary=summary_part,
                candidate_accepted=accepted,
                base_summary=base_summary,
                base_accepted=base_accepted,
                gate_mode=str(args.selection_gate_mode),
            )
            cand_row = summary_part.iloc[0].to_dict()
            metric_rows.append(
                {
                    "arm": arm,
                    "max_adjustment": float(cap),
                    "min_abs_z": float(min_abs_z),
                    "active_schedule_share": float(
                        (
                            pd.to_numeric(schedule["portfolio_priority_adjustment"], errors="coerce").abs()
                            > 1e-12
                        ).mean()
                    ),
                    "coverage": float(coverage.get("coverage", np.nan)),
                    "trade_count": int(cand_row.get("trade_count", 0) or 0),
                    "net_pnl": float(cand_row.get("net_pnl", np.nan)),
                    "delta_net_pnl": float(cand_row.get("net_pnl", 0.0) or 0.0)
                    - float(base_row.get("net_pnl", 0.0) or 0.0),
                    "full_sl_rate": float(cand_row.get("full_sl_rate", np.nan)),
                    "delta_full_sl_rate": float(cand_row.get("full_sl_rate", np.nan))
                    - float(base_row.get("full_sl_rate", np.nan)),
                    "timeout_rate": float(cand_row.get("timeout_rate", np.nan)),
                    "delta_timeout_rate": float(cand_row.get("timeout_rate", np.nan))
                    - float(base_row.get("timeout_rate", np.nan)),
                    "accepted_jaccard": float(replay_metrics.get("replay_accepted_jaccard", np.nan)),
                    "entrants": int(float(replay_metrics.get("replay_entrants", 0) or 0)),
                    "removed": int(float(replay_metrics.get("replay_removed", 0) or 0)),
                    "net_replacement_pnl": float(replay_metrics.get("replay_net_replacement_pnl", np.nan)),
                    "net_action_pnl_delta": float(replay_metrics.get("replay_net_action_pnl_delta", np.nan)),
                    "defensive_success": float(replay_metrics.get("replay_defensive_success", np.nan)),
                    "gate_passed": bool(replay_metrics.get("replay_selection_gate_passed", False)),
                }
            )

    summary = pd.concat(summary_frames, ignore_index=True)
    by_head = pd.concat(by_head_frames, ignore_index=True)
    schedules = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    overlap = _accepted_overlap(accepted_by_arm)
    accepted_all = pd.concat(list(accepted_by_arm.values()), ignore_index=True)
    swap = mstc._threshold_action_utility(accepted_all, BASELINE_ARM)

    summary.to_csv(args.output_dir / "head_priority_cap_sweep_replay_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "head_priority_cap_sweep_by_head.csv", index=False)
    metrics.to_csv(args.output_dir / "head_priority_cap_sweep_metrics.csv", index=False)
    overlap.to_csv(args.output_dir / "head_priority_cap_sweep_accepted_overlap.csv", index=False)
    swap.to_csv(args.output_dir / "head_priority_cap_sweep_accepted_swap_utility.csv", index=False)
    schedules.to_parquet(args.output_dir / "head_priority_cap_sweep_schedules.parquet", index=False)
    selected_challenger = select_shadow_challenger(
        metrics,
        min_accepted_jaccard=float(args.selection_min_accepted_jaccard),
        max_full_sl_delta=float(args.selection_max_full_sl_delta),
        max_timeout_delta=float(args.selection_max_timeout_delta),
    )
    (args.output_dir / "selected_shadow_challenger.json").write_text(
        json.dumps(_json_safe(selected_challenger), indent=2) + "\n",
        encoding="utf-8",
    )

    out_manifest = {
        "generated_by": "replay_market_state_learned_priority_cap_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fixed_model_market_state_head_priority_cap_sweep",
        "contract": {
            "changes_scores_or_ranks": False,
            "changes_thresholds": False,
            "changes_position_sizing": False,
            "changes_auction_ordering": True,
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
            "neutralizes_source_multiplier_and_rank_adjustment": True,
            "static_baseline_source": (
                "materialized_t1_manifest"
                if static_baseline_info is not None
                else "recomputed_from_candidates"
            ),
        },
        "params": {
            "caps": caps,
            "min_abs_z_thresholds": min_abs_z_thresholds,
            "selection_gate_mode": str(args.selection_gate_mode),
            "selection_min_accepted_jaccard": float(args.selection_min_accepted_jaccard),
            "selection_max_full_sl_delta": float(args.selection_max_full_sl_delta),
            "selection_max_timeout_delta": float(args.selection_max_timeout_delta),
            "policy_variant": str(args.policy_variant),
            "market_mode": str(args.market_mode),
        },
        "inputs": {
            "priority_dir": str(priority_dir),
            "priority_manifest": str(priority_dir / "manifest.json"),
            "priority_manifest_sha256": _sha256(priority_dir / "manifest.json"),
            "learned_schedule": str(schedule_path),
            "learned_schedule_sha256": _sha256(schedule_path),
            "candidates": str(candidates_path),
            "candidates_sha256": _sha256(candidates_path),
            "train_deployable_candidates": str(train_deployable_path),
            "train_deployable_candidates_sha256": _sha256(train_deployable_path),
            "train_deployable_rank_contract": train_deployable_contract,
            "policy_manifest": str(policy_manifest_path),
            "policy_manifest_sha256": _sha256(policy_manifest_path),
            "policy_manifest_run_id": policy_payload.get("run_id"),
            "static_baseline_manifest": (
                str(args.static_baseline_manifest)
                if args.static_baseline_manifest is not None
                else None
            ),
            "static_baseline_manifest_sha256": _sha256(args.static_baseline_manifest),
        },
        "static_baseline": static_baseline_info,
        "summary": metrics.to_dict("records"),
        "selected_shadow_challenger": selected_challenger,
        "outputs": {
            "metrics": str(args.output_dir / "head_priority_cap_sweep_metrics.csv"),
            "summary": str(args.output_dir / "head_priority_cap_sweep_replay_summary.csv"),
            "by_head": str(args.output_dir / "head_priority_cap_sweep_by_head.csv"),
            "overlap": str(args.output_dir / "head_priority_cap_sweep_accepted_overlap.csv"),
            "swap": str(args.output_dir / "head_priority_cap_sweep_accepted_swap_utility.csv"),
            "selected_shadow_challenger": str(args.output_dir / "selected_shadow_challenger.json"),
            "report": str(args.output_dir / "market_state_head_priority_cap_sweep_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(out_manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_head_priority_cap_sweep_report.md").write_text(
        _render_report(
            manifest=out_manifest,
            metrics=metrics,
            by_head=by_head,
            overlap=overlap,
            swap=swap,
            selected=selected_challenger,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(args.output_dir),
                    "metrics": metrics.to_dict("records"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
