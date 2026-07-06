#!/usr/bin/env python3
"""Run the repeatable contextual TP/SL A/B evidence workflow.

The workflow is deliberately non-tuning:

1. Build a cumulative flat candidate ledger.
2. Check frozen dual-scoring readiness and run the frozen gate only if ready.
3. Rebuild the A/B promotion dashboard from development and frozen evidence.

It exists so future post-freeze candidate ledgers can be appended and evaluated
without changing candidate definitions or manually stitching reports together.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMOTION_DIR = ROOT / "data_perp/reports/contextual_tp_sl_current_candidate_promotion_table_v1_20260701"
DEFAULT_SCORECARD_DIR = ROOT / "data_perp/reports/contextual_tp_sl_reliability_feature_scorecard_v1_20260701"
DEFAULT_FROZEN_CANDIDATE_READINESS_DIR = (
    ROOT / "data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_readiness_v3_materialized_20260701"
)
LIVE_OUTCOME_SUMMARY_COLUMNS = (
    "prediction_rows",
    "trade_log_rows",
    "traded_rows",
    "realized_traded_rows",
    "unresolved_traded_rows",
    "realized_timestamps",
    "realized_active_heads",
    "timestamp_min",
    "timestamp_max",
    "trade_log_timestamp_max",
    "prediction_to_trade_log_lag_minutes",
    "prediction_ledger_stale_vs_trade_log",
    "realized_timestamp_min",
    "realized_timestamp_max",
)
HEAD_PATTERN = r"^(short_bollinger|short_boll|long_bars|long_dist|short_asset)"
DIAGNOSTIC_GROUPS = (
    "uncertainty",
    "drift",
    "ood",
    "recent_hit_rate_surprise",
)
LOG_TS_PATTERN = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]")
LEDGER_APPEND_PATTERN = re.compile(r"Prediction ledger appended: rows=(?P<rows>\d+) path=(?P<path>\S+)")
MONITOR_PATTERN = re.compile(r"Monitoring (?P<active>\d+) active positions for price action")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run(cmd: list[str], command_log: list[dict[str, Any]]) -> None:
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=True)
    command_log.append({"cmd": cmd, "returncode": int(completed.returncode)})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _live_outcome_summary_frame(summary: dict[str, Any]) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame()
    return pd.DataFrame([{col: summary.get(col, "") for col in LIVE_OUTCOME_SUMMARY_COLUMNS}])


def _parse_log_timestamp(line: str) -> pd.Timestamp | pd.NaT:
    match = LOG_TS_PATTERN.search(line)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group("ts") + "+00:00", utc=True, errors="coerce")


def _live_runtime_health_frame(log_path: Path | None, live_summary: dict[str, Any]) -> pd.DataFrame:
    if log_path is None or not log_path.exists():
        return pd.DataFrame()
    append_count = 0
    append_rows_total = 0
    last_append_ts = pd.NaT
    last_append_rows = 0
    last_append_path = ""
    last_monitor_ts = pd.NaT
    last_active_positions = None
    heartbeat_count = 0
    last_heartbeat_ts = pd.NaT
    try:
        with log_path.open(errors="replace") as handle:
            for line in handle:
                ts = _parse_log_timestamp(line)
                append_match = LEDGER_APPEND_PATTERN.search(line)
                if append_match:
                    append_count += 1
                    rows = int(append_match.group("rows"))
                    append_rows_total += rows
                    last_append_ts = ts
                    last_append_rows = rows
                    last_append_path = append_match.group("path")
                monitor_match = MONITOR_PATTERN.search(line)
                if monitor_match:
                    last_monitor_ts = ts
                    last_active_positions = int(monitor_match.group("active"))
                if "INFERENCE_MONITOR_HEARTBEAT" in line:
                    heartbeat_count += 1
                    last_heartbeat_ts = ts
    except OSError:
        return pd.DataFrame()

    ledger_ts = pd.to_datetime(live_summary.get("timestamp_max"), utc=True, errors="coerce")
    trade_ts = pd.to_datetime(live_summary.get("trade_log_timestamp_max"), utc=True, errors="coerce")

    def _minutes_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float | None:
        if pd.isna(later) or pd.isna(earlier):
            return None
        return float((later - earlier).total_seconds() / 60.0)

    row = {
        "log_path": str(log_path),
        "ledger_append_events": append_count,
        "ledger_append_rows_total": append_rows_total,
        "last_ledger_append_ts": last_append_ts.isoformat() if pd.notna(last_append_ts) else "",
        "last_ledger_append_rows": last_append_rows,
        "last_ledger_append_path": last_append_path,
        "last_monitor_ts": last_monitor_ts.isoformat() if pd.notna(last_monitor_ts) else "",
        "last_monitor_active_positions": last_active_positions if last_active_positions is not None else "",
        "monitor_heartbeat_events": heartbeat_count,
        "last_monitor_heartbeat_ts": last_heartbeat_ts.isoformat() if pd.notna(last_heartbeat_ts) else "",
        "minutes_heartbeat_after_last_append": _minutes_between(last_heartbeat_ts, last_append_ts),
        "minutes_monitor_after_last_append": _minutes_between(last_monitor_ts, last_append_ts),
        "minutes_trade_log_after_last_append": _minutes_between(trade_ts, last_append_ts),
        "minutes_trade_log_after_ledger_row_ts": _minutes_between(trade_ts, ledger_ts),
    }
    return pd.DataFrame([row])


def _loads_count_map(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(val) for key, val in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): int(val) for key, val in parsed.items()}


def _policy_evidence_by_head_frame(source: dict[str, Any]) -> pd.DataFrame:
    action_counts = _loads_count_map(source.get("policy_action_head_counts"))
    outcome_counts = _loads_count_map(source.get("policy_outcome_head_counts"))
    heads = sorted(set(action_counts) | set(outcome_counts))
    rows = []
    for head in heads:
        action_count = int(action_counts.get(head, 0))
        outcome_count = int(outcome_counts.get(head, 0))
        rows.append(
            {
                "head": head,
                "policy_action_rows": action_count,
                "matured_outcome_rows": outcome_count,
                "matured_per_action_rate": float(outcome_count / action_count) if action_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _policy_outcome_deficit_frame(source: dict[str, Any], requirements: dict[str, Any]) -> pd.DataFrame:
    rows = []
    action_threshold = int(requirements.get("min_policy_outcome_rows_per_action_head") or 0)
    for head, count in _loads_count_map(source.get("policy_outcome_low_action_head_counts")).items():
        rows.append(
            {
                "gate": "action_head_minimum",
                "head": head,
                "observed_matured_outcomes": int(count),
                "required_matured_outcomes": action_threshold,
            }
        )
    required_threshold = int(requirements.get("min_policy_outcome_rows_per_required_head") or 0)
    for head, count in _loads_count_map(source.get("policy_outcome_low_required_head_counts")).items():
        rows.append(
            {
                "gate": "required_head_minimum",
                "head": head,
                "observed_matured_outcomes": int(count),
                "required_matured_outcomes": required_threshold,
            }
        )
    return pd.DataFrame(rows)


def _diagnostic_family_coverage_frame(source: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for group in DIAGNOSTIC_GROUPS:
        rows.append(
            {
                "family": group,
                "columns_present": int(source.get(f"{group}_columns_present") or 0),
                "columns_required": int(source.get(f"{group}_columns_required") or 0),
                "finite_rows": int(source.get(f"{group}_finite_rows") or 0),
                "finite_row_rate": float(source.get(f"{group}_finite_row_rate") or 0.0),
                "finite_cells": int(source.get(f"{group}_finite_cells") or 0),
                "finite_cell_rate": float(source.get(f"{group}_finite_cell_rate") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _eligible_head_gate_summary_frame(gate_dir: Path) -> pd.DataFrame:
    path = gate_dir / "frozen_dual_scoring_gate_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    keep_cols = [
        "bundle",
        "tested_feature_families",
        "baseline_trade_count",
        "best_delta_pnl_variant",
        "best_delta_net_pnl",
        "best_delta_full_sl_rate",
        "max_adjusted_rows",
        "max_adjusted_share",
        "min_accepted_jaccard",
        "total_entrants",
        "total_removed",
        "max_adjusted_acceptance_changed",
        "promotion_ready",
        "failed_checks",
    ]
    return frame[[col for col in keep_cols if col in frame.columns]].copy()


def _head_eligibility_frame(source: dict[str, Any], requirements: dict[str, Any]) -> pd.DataFrame:
    action_counts = _loads_count_map(source.get("policy_action_head_counts"))
    outcome_counts = _loads_count_map(source.get("policy_outcome_head_counts"))
    required_heads_raw = requirements.get("required_policy_outcome_head") or []
    required_heads = {str(head) for head in required_heads_raw if str(head)}
    heads = sorted(set(action_counts) | set(outcome_counts) | required_heads)
    if not heads:
        return pd.DataFrame()
    min_required = int(requirements.get("min_policy_outcome_rows_per_required_head") or 0)
    min_action = int(requirements.get("min_policy_outcome_rows_per_action_head") or 0)
    rows = []
    for head in heads:
        action_rows = int(action_counts.get(head, 0))
        outcome_rows = int(outcome_counts.get(head, 0))
        required_min = min_required if head in required_heads else min_action
        if action_rows <= 0 and head in required_heads:
            status = "needs_action_evidence"
        elif outcome_rows >= required_min:
            status = "eligible"
        else:
            status = "needs_more_outcomes"
        rows.append(
            {
                "head": head,
                "required": bool(head in required_heads),
                "policy_action_rows": action_rows,
                "matured_outcome_rows": outcome_rows,
                "required_matured_outcomes": required_min,
                "matured_outcomes_needed": max(0, required_min - outcome_rows),
                "head_evidence_status": status,
            }
        )
    return pd.DataFrame(rows)


def _head_series(strategy_id: pd.Series) -> pd.Series:
    return strategy_id.astype(str).str.extract(HEAD_PATTERN, expand=False).replace(
        {"short_boll": "short_bollinger"}
    )


def _write_head_subset_ledger(ledger_path: Path, output_path: Path, heads: list[str]) -> dict[str, Any]:
    keep_heads = sorted({str(head) for head in heads if str(head)})
    if not keep_heads:
        return {"enabled": False, "reason": "no_eligible_heads", "output": str(output_path)}
    frame = pd.read_parquet(ledger_path)
    if "strategy_id" not in frame.columns:
        return {"enabled": False, "reason": "missing_strategy_id", "output": str(output_path)}
    head_values = _head_series(frame["strategy_id"])
    subset = frame.loc[head_values.isin(keep_heads)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_path, index=False)
    return {
        "enabled": True,
        "output": str(output_path),
        "eligible_heads": keep_heads,
        "input_rows": int(len(frame)),
        "output_rows": int(len(subset)),
        "dropped_rows": int(len(frame) - len(subset)),
        "timestamp_min": subset["timestamp"].min().isoformat()
        if "timestamp" in subset.columns and len(subset)
        else "",
        "timestamp_max": subset["timestamp"].max().isoformat()
        if "timestamp" in subset.columns and len(subset)
        else "",
    }


def _run_frozen_gate(
    *,
    baseline: Path,
    output_dir: Path,
    eval_start: str,
    eval_end: str,
    market_mode: str,
    bundles: list[str],
    readiness_dir: Path,
    command_log: list[dict[str, Any]],
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_frozen_dual_scoring_gate.py",
        "--baseline",
        str(baseline),
        "--output-dir",
        str(output_dir),
        "--eval-start",
        str(eval_start),
        "--market-mode",
        str(market_mode),
        "--readiness-dir",
        str(readiness_dir),
    ]
    for bundle in bundles:
        cmd.extend(["--bundle", str(bundle)])
    if eval_end:
        cmd.extend(["--eval-end", str(eval_end)])
    _run(cmd, command_log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=None, help="Flat candidate parquet. Repeatable.")
    parser.add_argument("--root", action="append", default=None, help="Optional root scanned for flat ledgers.")
    parser.add_argument("--bundle", action="append", required=True, help="label=path or path for frozen gate runner.")
    parser.add_argument(
        "--materialize-live-outcomes",
        action="store_true",
        help="Attach inference_trades.csv lifecycle outcomes to a run-scoped live prediction ledger before accumulation.",
    )
    parser.add_argument(
        "--live-prediction-ledger",
        type=Path,
        default=(
            ROOT
            / "data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260629_050000_lgbm_mda/"
            "prediction_ledger.parquet"
        ),
    )
    parser.add_argument("--trade-log", type=Path, default=ROOT / "inference_trades.csv")
    parser.add_argument(
        "--live-log",
        type=Path,
        default=None,
        help="Optional live inference log used to report last prediction-ledger append and monitor heartbeat.",
    )
    parser.add_argument("--default-expected-fee-bps", type=float, default=0.0)
    parser.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--min-post-cutoff-rows", type=int, default=2000)
    parser.add_argument("--min-post-cutoff-timestamps", type=int, default=40)
    parser.add_argument("--min-post-cutoff-active-heads", type=int, default=3)
    parser.add_argument("--min-policy-action-rows", type=int, default=50)
    parser.add_argument("--min-policy-action-timestamps", type=int, default=10)
    parser.add_argument("--min-policy-outcome-rows", type=int, default=50)
    parser.add_argument("--min-policy-outcome-timestamps", type=int, default=10)
    parser.add_argument("--min-policy-outcome-rows-per-action-head", type=int, default=0)
    parser.add_argument("--required-policy-outcome-head", action="append", default=None)
    parser.add_argument("--min-policy-outcome-rows-per-required-head", type=int, default=1)
    parser.add_argument("--min-diagnostic-group-features", type=int, default=1)
    parser.add_argument("--min-diagnostic-group-finite-rate", type=float, default=0.25)
    parser.add_argument("--promotion-dir", type=Path, default=DEFAULT_PROMOTION_DIR)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument(
        "--run-eligible-head-gate",
        action="store_true",
        help="Run a clearly separated frozen dual-scoring gate on heads with enough matured policy evidence.",
    )
    parser.add_argument(
        "--eligible-head-gate-readiness-dir",
        type=Path,
        default=DEFAULT_FROZEN_CANDIDATE_READINESS_DIR,
        help="Frozen candidate-selection readiness directory used by run_frozen_dual_scoring_gate.py.",
    )
    parser.add_argument("--force-gate", action="store_true", help="Run frozen gate on best source even if not ready.")
    parser.add_argument(
        "--skip-rematerialize-diagnostics",
        action="store_true",
        help="Do not recompute generated reliability diagnostics after ledger accumulation.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    ledger_dir = out_dir / "cumulative_ledger"
    readiness_dir = out_dir / "readiness"
    dashboard_dir = out_dir / "dashboard"
    live_outcome_dir = out_dir / "live_outcomes"
    eligible_head_dir = out_dir / "eligible_heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []

    candidates = list(args.candidate or [])
    if args.materialize_live_outcomes:
        live_outcome_path = live_outcome_dir / "prediction_ledger_with_live_outcomes.parquet"
        live_outcome_cmd = [
            sys.executable,
            "scripts/materialize_live_prediction_ledger_outcomes.py",
            "--prediction-ledger",
            str(args.live_prediction_ledger),
            "--trade-log",
            str(args.trade_log),
            "--output",
            str(live_outcome_path),
            "--report-dir",
            str(live_outcome_dir),
            "--default-expected-fee-bps",
            str(args.default_expected_fee_bps),
        ]
        _run(live_outcome_cmd, command_log)
        candidates.append(str(live_outcome_path))

    ledger_path = ledger_dir / "cumulative_flat_candidates.parquet"
    ledger_cmd = [
        sys.executable,
        "scripts/build_cumulative_flat_frozen_gate_ledger.py",
        "--output",
        str(ledger_path),
        "--report-dir",
        str(ledger_dir),
        "--cutoff",
        str(args.cutoff),
    ]
    for candidate in candidates:
        ledger_cmd.extend(["--candidate", str(candidate)])
    for root in args.root or []:
        ledger_cmd.extend(["--root", str(root)])
    if not args.skip_rematerialize_diagnostics:
        ledger_cmd.append("--rematerialize-diagnostics")
    _run(ledger_cmd, command_log)

    readiness_cmd = [
        sys.executable,
        "scripts/run_latest_frozen_dual_scoring_gate_if_ready.py",
        "--out-dir",
        str(readiness_dir),
        "--candidate",
        str(ledger_path),
        "--cutoff",
        str(args.cutoff),
        "--market-mode",
        str(args.market_mode),
        "--min-post-cutoff-rows",
        str(args.min_post_cutoff_rows),
        "--min-post-cutoff-timestamps",
        str(args.min_post_cutoff_timestamps),
        "--min-post-cutoff-active-heads",
        str(args.min_post_cutoff_active_heads),
        "--min-policy-action-rows",
        str(args.min_policy_action_rows),
        "--min-policy-action-timestamps",
        str(args.min_policy_action_timestamps),
        "--min-policy-outcome-rows",
        str(args.min_policy_outcome_rows),
        "--min-policy-outcome-timestamps",
        str(args.min_policy_outcome_timestamps),
        "--min-policy-outcome-rows-per-action-head",
        str(args.min_policy_outcome_rows_per_action_head),
        "--min-policy-outcome-rows-per-required-head",
        str(args.min_policy_outcome_rows_per_required_head),
        "--min-diagnostic-group-features",
        str(args.min_diagnostic_group_features),
        "--min-diagnostic-group-finite-rate",
        str(args.min_diagnostic_group_finite_rate),
    ]
    for head in args.required_policy_outcome_head or []:
        readiness_cmd.extend(["--required-policy-outcome-head", str(head)])
    for bundle in args.bundle:
        readiness_cmd.extend(["--bundle", str(bundle)])
    if args.eval_end:
        readiness_cmd.extend(["--eval-end", str(args.eval_end)])
    if args.force_gate:
        readiness_cmd.append("--force")
    _run(readiness_cmd, command_log)

    dashboard_cmd = [
        sys.executable,
        "scripts/build_contextual_tp_sl_ablation_dashboard.py",
        "--promotion-dir",
        str(args.promotion_dir),
        "--scorecard-dir",
        str(args.scorecard_dir),
        "--readiness-dir",
        str(readiness_dir),
        "--ledger-dir",
        str(ledger_dir),
        "--output-dir",
        str(dashboard_dir),
    ]
    _run(dashboard_cmd, command_log)

    ledger_manifest = _read_json(ledger_dir / "cumulative_flat_ledger_manifest.json")
    readiness = _read_json(readiness_dir / "latest_flat_frozen_gate_readiness.json")
    dashboard_manifest = _read_json(dashboard_dir / "manifest.json")
    live_outcome_manifest = _read_json(live_outcome_dir / "live_prediction_ledger_outcome_manifest.json")
    candidate_dashboard = _read_csv(dashboard_dir / "candidate_deployment_dashboard.csv")
    top_rows = candidate_dashboard.head(10) if not candidate_dashboard.empty else pd.DataFrame()
    live_outcome_summary = _live_outcome_summary_frame(live_outcome_manifest)
    live_runtime_health = _live_runtime_health_frame(args.live_log, live_outcome_manifest)
    post = ledger_manifest.get("post_cutoff") or {}
    req = readiness.get("requirements") or {}
    source = readiness.get("selected_source") or readiness.get("nearest_source") or {}
    policy_evidence_by_head = _policy_evidence_by_head_frame(source)
    policy_outcome_deficits = _policy_outcome_deficit_frame(source, req)
    diagnostic_family_coverage = _diagnostic_family_coverage_frame(source)
    head_eligibility = _head_eligibility_frame(source, req)
    eligible_heads = (
        head_eligibility.loc[head_eligibility["head_evidence_status"].eq("eligible"), "head"]
        .dropna()
        .astype(str)
        .tolist()
        if not head_eligibility.empty
        else []
    )
    eligible_head_subset = _write_head_subset_ledger(
        ledger_path,
        eligible_head_dir / "eligible_head_candidates.parquet",
        eligible_heads,
    )
    eligible_head_gate_dir = out_dir / "eligible_head_gate"
    eligible_head_gate_ran = False
    if args.run_eligible_head_gate and bool(eligible_head_subset.get("enabled")):
        _run_frozen_gate(
            baseline=Path(str(eligible_head_subset["output"])),
            output_dir=eligible_head_gate_dir,
            eval_start=str(args.cutoff),
            eval_end=str(args.eval_end),
            market_mode=str(args.market_mode),
            bundles=list(args.bundle),
            readiness_dir=args.eligible_head_gate_readiness_dir,
            command_log=command_log,
        )
        eligible_head_gate_ran = True
    eligible_head_gate_summary = _eligible_head_gate_summary_frame(eligible_head_gate_dir)
    manifest = {
        "generated_by": Path(__file__).name,
        "cutoff": str(args.cutoff),
        "eval_end": str(args.eval_end),
        "output_dir": str(out_dir),
        "materialized_live_outcomes": bool(args.materialize_live_outcomes),
        "live_outcome_manifest": live_outcome_manifest,
        "eligible_head_subset": eligible_head_subset,
        "eligible_head_gate": {
            "enabled": bool(args.run_eligible_head_gate),
            "ran": bool(eligible_head_gate_ran),
            "output_dir": str(eligible_head_gate_dir),
            "readiness_dir": str(args.eligible_head_gate_readiness_dir),
        },
        "commands": command_log,
        "ledger_manifest": ledger_manifest,
        "readiness": readiness,
        "dashboard_manifest": dashboard_manifest,
    }
    (out_dir / "contextual_tp_sl_ablation_workflow_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Contextual TP/SL A/B Workflow Run",
        "",
        f"Cutoff: `{args.cutoff}`",
        f"Ledger: `{ledger_path}`",
        f"Materialized live outcomes: `{bool(args.materialize_live_outcomes)}`",
        f"Readiness report: `{readiness_dir / 'latest_flat_frozen_gate_readiness.md'}`",
        f"Dashboard: `{dashboard_dir / 'contextual_tp_sl_ablation_dashboard.md'}`",
        "",
        "## Status",
        "",
        f"- Development candidates passing: `{dashboard_manifest.get('development_pass_count', 0)}`",
        f"- Deployment candidates passing: `{dashboard_manifest.get('deployment_pass_count', 0)}`",
        f"- Frozen ready sources: `{readiness.get('ready_sources', 0)}`",
        f"- Frozen gate ran: `{bool(readiness.get('ran_gate'))}`",
        f"- Post-cutoff rows: `{post.get('post_cutoff_rows', 0)}` / `{req.get('min_post_cutoff_rows', args.min_post_cutoff_rows)}`",
        f"- Post-cutoff timestamps: `{post.get('post_cutoff_timestamps', 0)}` / `{req.get('min_post_cutoff_timestamps', args.min_post_cutoff_timestamps)}`",
        f"- Post-cutoff active heads: `{post.get('post_cutoff_active_heads', 0)}` / `{req.get('min_post_cutoff_active_heads', args.min_post_cutoff_active_heads)}`",
        f"- Estimated policy-action rows: `{source.get('policy_action_rows_estimate', 0)}` / `{req.get('min_policy_action_rows', args.min_policy_action_rows)}`",
        f"- Estimated policy-action timestamps: `{source.get('policy_action_timestamps_estimate', 0)}` / `{req.get('min_policy_action_timestamps', args.min_policy_action_timestamps)}`",
        f"- Policy-action estimate source: `{source.get('policy_action_estimate_source', '')}`",
        f"- Estimated matured policy-outcome rows: `{source.get('policy_outcome_rows_estimate', 0)}` / `{req.get('min_policy_outcome_rows', args.min_policy_outcome_rows)}`",
        f"- Estimated matured policy-outcome timestamps: `{source.get('policy_outcome_timestamps_estimate', 0)}` / `{req.get('min_policy_outcome_timestamps', args.min_policy_outcome_timestamps)}`",
        f"- Policy-outcome estimate source: `{source.get('policy_outcome_estimate_source', '')}`",
        f"- Minimum matured outcomes per action head: `{req.get('min_policy_outcome_rows_per_action_head', args.min_policy_outcome_rows_per_action_head)}`",
        f"- Required matured-outcome heads: `{', '.join(req.get('required_policy_outcome_head', args.required_policy_outcome_head or []))}`",
        f"- Minimum matured outcomes per required head: `{req.get('min_policy_outcome_rows_per_required_head', args.min_policy_outcome_rows_per_required_head)}`",
        "",
        "## Live Outcome Materialization",
        "",
        live_outcome_summary.to_markdown(index=False)
        if not live_outcome_summary.empty
        else "_Live outcome materialization was not run._",
        "",
        "## Diagnostic Family Coverage",
        "",
        diagnostic_family_coverage.to_markdown(index=False)
        if not diagnostic_family_coverage.empty
        else "_No diagnostic family coverage was available._",
        "",
        "## Live Runtime Health",
        "",
        live_runtime_health.to_markdown(index=False)
        if not live_runtime_health.empty
        else "_No live runtime log was provided or readable._",
        "",
        "## Policy Evidence By Head",
        "",
        policy_evidence_by_head.to_markdown(index=False)
        if not policy_evidence_by_head.empty
        else "_No policy action/outcome head counts were available._",
        "",
        "## Policy Outcome Gate Deficits",
        "",
        policy_outcome_deficits.to_markdown(index=False)
        if not policy_outcome_deficits.empty
        else "_No per-head policy outcome deficits._",
        "",
        "## Head Evidence Eligibility",
        "",
        head_eligibility.to_markdown(index=False)
        if not head_eligibility.empty
        else "_No head-level eligibility data was available._",
        "",
        "## Eligible Head Subset Ledger",
        "",
        pd.DataFrame([eligible_head_subset]).to_markdown(index=False),
        "",
        "## Eligible Head Frozen Gate",
        "",
        pd.DataFrame(
            [
                {
                    "enabled": bool(args.run_eligible_head_gate),
                    "ran": bool(eligible_head_gate_ran),
                    "output_dir": str(eligible_head_gate_dir),
                    "readiness_dir": str(args.eligible_head_gate_readiness_dir),
                }
            ]
        ).to_markdown(index=False),
        "",
        eligible_head_gate_summary.to_markdown(index=False)
        if not eligible_head_gate_summary.empty
        else "_No eligible-head gate summary was available._",
        "",
        "## Top Candidate Status",
        "",
        top_rows.to_markdown(index=False) if not top_rows.empty else "_No candidate rows._",
        "",
        "## Commands",
        "",
        "\n".join(f"- `{' '.join(row['cmd'])}`" for row in command_log),
    ]
    (out_dir / "contextual_tp_sl_ablation_workflow_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
