#!/usr/bin/env python3
"""Run the flat-ledger frozen dual-scoring gate only when evidence is ready.

This script is intentionally conservative.  It scans known flat candidate
ledgers, checks whether the requested post-cutoff window is large enough and
whether the reliability diagnostic families are present, then either runs
`run_frozen_dual_scoring_gate.py` or writes a readiness report explaining why
the gate was skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency fallback
    pq = None


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATES = (
    "data_perp/reports/contextual_tp_sl_flat_diagnostics_jun28_with_history_20260701/"
    "combo_candidates_history_jun28_with_diagnostics.parquet",
    "data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/"
    "combo_candidates.parquet",
    "data_perp/reports/contextual_tp_sl_latest_jun26_28_static_20260701/combo_candidates.parquet",
    "data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet",
)

DIAGNOSTIC_GROUPS: dict[str, tuple[str, ...]] = {
    "uncertainty": (
        "generated_score_uncertainty_p1mp",
        "generated_score_entropy",
        "generated_score_abs_distance_from_half",
    ),
    "drift": (
        "generated_score_abs_diff_1",
        "generated_score_abs_diff_4",
        "generated_score_abs_diff_24",
        "generated_score_abs_minus_prev24_mean",
        "generated_score_prev24_std",
        "generated_strategy_score_shift_abs_z",
    ),
    "ood": (
        "generated_strategy_score_ood_abs_z",
        "generated_strategy_barrier_ood_abs_z",
        "generated_strategy_friction_ood_abs_z",
    ),
    "recent_hit_rate_surprise": (
        "generated_hr_surprise_24",
        "generated_hr_surprise_96",
        "generated_weighted_hr_surprise_24",
        "generated_weighted_hr_surprise_96",
        "generated_loss_rate_24",
        "generated_loss_rate_96",
    ),
}

BASE_COLUMNS = ("timestamp", "strategy_id", "symbol")
ACTION_SCORE_COLUMNS = (
    "auction_rank_score",
    "adjusted_rank_score",
    "normalized_rank_score",
)
ACTION_DECISION_COLUMNS = (
    "was_traded",
    "portfolio_decision",
)
OUTCOME_COLUMNS = (
    "net_return",
    "live_replay_net_return",
    "simple_policy_net_return",
    "realized_net_return",
    "closed_trade_net_pnl",
    "net_pnl",
)


def _json_safe(value: Any) -> Any:
    if not isinstance(value, (dict, list, tuple)):
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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


def _head_name(strategy_id: pd.Series) -> pd.Series:
    text = strategy_id.astype(str)
    return text.str.extract(
        r"^(short_bollinger|short_boll|long_bars|long_dist|short_asset)",
        expand=False,
    ).replace({"short_boll": "short_bollinger"})


def _parquet_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    if pq is not None:
        try:
            return list(pq.read_schema(path).names)
        except Exception:
            pass
    try:
        return list(pd.read_parquet(path).columns)
    except Exception:
        return []


def _read_probe(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        present = [col for col in columns if col in _parquet_columns(path)]
        return pd.read_parquet(path, columns=present)
    except Exception:
        return pd.DataFrame(columns=columns)


def _truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y", "traded", "accepted"})


def _policy_action_mask(probe: pd.DataFrame, post_mask: pd.Series) -> tuple[pd.Series, str]:
    mask = pd.Series(False, index=probe.index)
    has_decision_evidence = pd.Series(False, index=probe.index)
    sources: list[str] = []
    if "was_traded" in probe.columns:
        has_decision_evidence |= probe["was_traded"].notna()
        traded = _truthy_series(probe["was_traded"])
        if bool((post_mask & traded).any()):
            mask |= traded
            sources.append("was_traded_true")
    if "portfolio_decision" in probe.columns:
        decision_raw = probe["portfolio_decision"]
        decision = decision_raw.astype(str).str.strip().str.lower()
        has_decision_evidence |= decision_raw.notna() & ~decision.isin({"", "none", "nan", "nat"})
        traded = decision.eq("traded")
        if bool((post_mask & traded).any()):
            mask |= traded
            sources.append("portfolio_decision_traded")
    if "auction_rank_score" in probe.columns:
        score = pd.to_numeric(probe["auction_rank_score"], errors="coerce")
        auction_mask = score.notna() & ~has_decision_evidence
        if bool((post_mask & auction_mask).any()):
            mask |= auction_mask
            sources.append("finite_auction_rank_score")
    if bool((post_mask & mask).any()):
        return post_mask & mask, "+".join(sources)
    for col in ("adjusted_rank_score", "normalized_rank_score"):
        if col in probe.columns:
            score = pd.to_numeric(probe[col], errors="coerce")
            return post_mask & score.notna(), f"finite_{col}"
    return pd.Series(False, index=probe.index), "missing_policy_action_columns"


def _policy_outcome_mask(probe: pd.DataFrame, action_mask: pd.Series) -> tuple[pd.Series, str]:
    mask = pd.Series(False, index=probe.index)
    sources: list[str] = []
    for col in OUTCOME_COLUMNS:
        if col not in probe.columns:
            continue
        values = pd.to_numeric(probe[col], errors="coerce")
        finite = values.notna()
        if bool((action_mask & finite).any()):
            mask |= finite
            sources.append(f"finite_{col}")
    if sources:
        return action_mask & mask, "+".join(sources)
    return pd.Series(False, index=probe.index), "missing_policy_outcome_columns"


def _head_counts(heads: pd.Series, mask: pd.Series) -> dict[str, int]:
    if len(heads) == 0 or len(mask) == 0:
        return {}
    counts = heads.loc[mask].dropna().astype(str).value_counts().sort_index()
    return {str(head): int(count) for head, count in counts.items()}


def _candidate_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in (args.candidate or DEFAULT_CANDIDATES)]
    for root in args.root or []:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in ("**/combo_candidates*.parquet", "**/*with_diagnostics*.parquet"):
            paths.extend(root_path.glob(pattern))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _scan_one(path: Path, cutoff: pd.Timestamp, args: argparse.Namespace) -> dict[str, Any]:
    columns = set(_parquet_columns(path))
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "ready": False,
            "rejection_reasons": "missing_file",
        }
    missing_base = [col for col in BASE_COLUMNS if col not in columns]
    group_counts = {
        group: int(len(set(cols).intersection(columns)))
        for group, cols in DIAGNOSTIC_GROUPS.items()
    }
    missing_groups = [
        group
        for group, count in group_counts.items()
        if count < int(args.min_diagnostic_group_features)
    ]
    diagnostic_cols = sorted({col for cols in DIAGNOSTIC_GROUPS.values() for col in cols if col in columns})
    probe_cols = [
        col
        for col in ("timestamp", "strategy_id", *ACTION_SCORE_COLUMNS, *ACTION_DECISION_COLUMNS)
        + OUTCOME_COLUMNS
        + tuple(diagnostic_cols)
        if col in columns
    ]
    probe = _read_probe(path, probe_cols)
    if "timestamp" in probe.columns:
        timestamp = pd.to_datetime(probe["timestamp"], utc=True, errors="coerce")
    else:
        timestamp = pd.Series(pd.NaT, index=probe.index)
    if "strategy_id" in probe.columns:
        heads = _head_name(probe["strategy_id"])
    else:
        heads = pd.Series(index=probe.index, dtype=object)
    post_mask = timestamp.ge(cutoff) if len(timestamp) else pd.Series(dtype=bool)
    post_rows = int(post_mask.sum()) if len(post_mask) else 0
    post_timestamps = int(timestamp.loc[post_mask].nunique()) if len(post_mask) else 0
    post_heads = int(heads.loc[post_mask].dropna().nunique()) if len(post_mask) else 0
    action_mask, policy_action_estimate_source = _policy_action_mask(probe, post_mask)
    policy_action_rows = int(action_mask.sum())
    policy_action_timestamps = int(timestamp.loc[action_mask].nunique())
    policy_action_active_heads = int(heads.loc[action_mask].dropna().nunique())
    policy_action_head_counts = _head_counts(heads, action_mask)
    outcome_mask, policy_outcome_estimate_source = _policy_outcome_mask(probe, action_mask)
    policy_outcome_rows = int(outcome_mask.sum())
    policy_outcome_timestamps = int(timestamp.loc[outcome_mask].nunique())
    policy_outcome_active_heads = int(heads.loc[outcome_mask].dropna().nunique())
    policy_outcome_head_counts = _head_counts(heads, outcome_mask)
    min_outcomes_per_action_head = int(args.min_policy_outcome_rows_per_action_head)
    low_action_head_outcomes = {
        head: int(policy_outcome_head_counts.get(head, 0))
        for head in policy_action_head_counts
        if min_outcomes_per_action_head > 0
        and int(policy_outcome_head_counts.get(head, 0)) < min_outcomes_per_action_head
    }
    required_outcome_heads = sorted({str(head) for head in (args.required_policy_outcome_head or [])})
    min_outcomes_per_required_head = int(args.min_policy_outcome_rows_per_required_head)
    low_required_head_outcomes = {
        head: int(policy_outcome_head_counts.get(head, 0))
        for head in required_outcome_heads
        if int(policy_outcome_head_counts.get(head, 0)) < min_outcomes_per_required_head
    }
    group_finite_rows: dict[str, int] = {}
    group_finite_rates: dict[str, float] = {}
    group_finite_cells: dict[str, int] = {}
    group_finite_cell_rates: dict[str, float] = {}
    post_denominator = max(post_rows, 1)
    for group, cols in DIAGNOSTIC_GROUPS.items():
        present_cols = [col for col in cols if col in probe.columns]
        if not present_cols or post_rows <= 0:
            group_finite_rows[group] = 0
            group_finite_rates[group] = 0.0
            group_finite_cells[group] = 0
            group_finite_cell_rates[group] = 0.0
            continue
        finite_any = pd.Series(False, index=probe.index)
        finite_cells = 0
        for col in present_cols:
            values = pd.to_numeric(probe[col], errors="coerce")
            finite = values.notna()
            finite_any |= finite
            finite_cells += int((post_mask & finite).sum())
        finite_rows = int((post_mask & finite_any).sum())
        group_finite_rows[group] = finite_rows
        group_finite_rates[group] = float(finite_rows / post_denominator)
        cell_denominator = max(post_rows * len(present_cols), 1)
        group_finite_cells[group] = finite_cells
        group_finite_cell_rates[group] = float(finite_cells / cell_denominator)
    low_finite_groups = [
        group
        for group, rate in group_finite_rates.items()
        if rate < float(args.min_diagnostic_group_finite_rate)
    ]
    reasons: list[str] = []
    if missing_base:
        reasons.append("missing_base_columns:" + ",".join(missing_base))
    if post_rows < int(args.min_post_cutoff_rows):
        reasons.append(f"post_cutoff_rows_lt_{args.min_post_cutoff_rows}")
    if post_timestamps < int(args.min_post_cutoff_timestamps):
        reasons.append(f"post_cutoff_timestamps_lt_{args.min_post_cutoff_timestamps}")
    if post_heads < int(args.min_post_cutoff_active_heads):
        reasons.append(f"post_cutoff_active_heads_lt_{args.min_post_cutoff_active_heads}")
    if policy_action_rows < int(args.min_policy_action_rows):
        reasons.append(f"policy_action_rows_lt_{args.min_policy_action_rows}")
    if policy_action_timestamps < int(args.min_policy_action_timestamps):
        reasons.append(f"policy_action_timestamps_lt_{args.min_policy_action_timestamps}")
    if policy_outcome_rows < int(args.min_policy_outcome_rows):
        reasons.append(f"policy_outcome_rows_lt_{args.min_policy_outcome_rows}")
    if policy_outcome_timestamps < int(args.min_policy_outcome_timestamps):
        reasons.append(f"policy_outcome_timestamps_lt_{args.min_policy_outcome_timestamps}")
    if low_action_head_outcomes:
        reasons.append(
            "policy_outcome_rows_per_action_head_lt_"
            + str(min_outcomes_per_action_head)
            + ":"
            + ",".join(sorted(low_action_head_outcomes))
        )
    if low_required_head_outcomes:
        reasons.append(
            "policy_outcome_rows_per_required_head_lt_"
            + str(min_outcomes_per_required_head)
            + ":"
            + ",".join(sorted(low_required_head_outcomes))
        )
    if missing_groups:
        reasons.append("missing_diagnostic_groups:" + ",".join(missing_groups))
    if low_finite_groups:
        reasons.append("low_diagnostic_group_finite_rate:" + ",".join(low_finite_groups))
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(probe)),
        "columns": int(len(columns)),
        "timestamp_min": timestamp.min().isoformat() if timestamp.notna().any() else "",
        "timestamp_max": timestamp.max().isoformat() if timestamp.notna().any() else "",
        "timestamp_count": int(timestamp.nunique(dropna=True)),
        "post_cutoff_rows": post_rows,
        "post_cutoff_timestamps": post_timestamps,
        "post_cutoff_active_heads": post_heads,
        "policy_action_rows_estimate": policy_action_rows,
        "policy_action_timestamps_estimate": policy_action_timestamps,
        "policy_action_active_heads_estimate": policy_action_active_heads,
        "policy_action_head_counts": json.dumps(policy_action_head_counts, sort_keys=True),
        "policy_action_estimate_source": policy_action_estimate_source,
        "policy_outcome_rows_estimate": policy_outcome_rows,
        "policy_outcome_timestamps_estimate": policy_outcome_timestamps,
        "policy_outcome_active_heads_estimate": policy_outcome_active_heads,
        "policy_outcome_head_counts": json.dumps(policy_outcome_head_counts, sort_keys=True),
        "policy_outcome_low_action_head_counts": json.dumps(low_action_head_outcomes, sort_keys=True),
        "policy_outcome_required_heads": json.dumps(required_outcome_heads),
        "policy_outcome_low_required_head_counts": json.dumps(low_required_head_outcomes, sort_keys=True),
        "policy_outcome_estimate_source": policy_outcome_estimate_source,
        "ready": len(reasons) == 0,
        "rejection_reasons": ";".join(reasons) if reasons else "none",
        **{f"{group}_columns_present": count for group, count in group_counts.items()},
        **{f"{group}_columns_required": len(cols) for group, cols in DIAGNOSTIC_GROUPS.items()},
        **{f"{group}_finite_rows": group_finite_rows[group] for group in DIAGNOSTIC_GROUPS},
        **{f"{group}_finite_row_rate": group_finite_rates[group] for group in DIAGNOSTIC_GROUPS},
        **{f"{group}_finite_cells": group_finite_cells[group] for group in DIAGNOSTIC_GROUPS},
        **{f"{group}_finite_cell_rate": group_finite_cell_rates[group] for group in DIAGNOSTIC_GROUPS},
    }


def _select_source(scan: pd.DataFrame, force: bool) -> pd.Series | None:
    if scan.empty:
        return None
    frame = scan.copy()
    for col in (
        "post_cutoff_rows",
        "post_cutoff_timestamps",
        "post_cutoff_active_heads",
        "policy_action_rows_estimate",
        "policy_action_timestamps_estimate",
        "policy_outcome_rows_estimate",
        "policy_outcome_timestamps_estimate",
    ):
        frame[col] = pd.to_numeric(frame.get(col, 0), errors="coerce").fillna(0)
    if force:
        eligible = frame[frame["exists"].astype(bool)].copy()
    else:
        eligible = frame[frame["ready"].astype(bool)].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        [
            "policy_action_rows_estimate",
            "policy_outcome_rows_estimate",
            "post_cutoff_rows",
            "post_cutoff_timestamps",
            "post_cutoff_active_heads",
            "timestamp_max",
        ],
        ascending=[False, False, False, False, False, False],
    )
    return eligible.iloc[0]


def _run_gate(args: argparse.Namespace, selected: pd.Series, out_dir: Path) -> Path:
    gate_dir = out_dir / "gate_run"
    cmd = [
        sys.executable,
        "scripts/run_frozen_dual_scoring_gate.py",
        "--baseline",
        str(selected["path"]),
        "--output-dir",
        str(gate_dir),
        "--eval-start",
        str(args.cutoff),
        "--market-mode",
        str(args.market_mode),
    ]
    for bundle in args.bundle:
        cmd.extend(["--bundle", str(bundle)])
    if args.eval_end:
        cmd.extend(["--eval-end", str(args.eval_end)])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return gate_dir


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    scan: pd.DataFrame,
    selected: pd.Series | None,
    gate_dir: Path | None,
) -> None:
    scan.to_csv(out_dir / "flat_frozen_gate_source_scan.csv", index=False)
    ready_count = int(scan["ready"].sum()) if "ready" in scan.columns else 0
    nearest = _select_source(scan, force=True)
    payload = {
        "generated_by": Path(__file__).name,
        "cutoff": str(args.cutoff),
        "eval_end": str(args.eval_end),
        "ready_sources": ready_count,
        "ran_gate": gate_dir is not None,
        "selected_source": selected.to_dict() if selected is not None else None,
        "nearest_source": nearest.to_dict() if nearest is not None else None,
        "gate_dir": str(gate_dir) if gate_dir is not None else None,
        "requirements": {
            "min_post_cutoff_rows": int(args.min_post_cutoff_rows),
            "min_post_cutoff_timestamps": int(args.min_post_cutoff_timestamps),
            "min_post_cutoff_active_heads": int(args.min_post_cutoff_active_heads),
            "min_policy_action_rows": int(args.min_policy_action_rows),
            "min_policy_action_timestamps": int(args.min_policy_action_timestamps),
            "min_policy_outcome_rows": int(args.min_policy_outcome_rows),
            "min_policy_outcome_timestamps": int(args.min_policy_outcome_timestamps),
            "min_policy_outcome_rows_per_action_head": int(args.min_policy_outcome_rows_per_action_head),
            "required_policy_outcome_head": list(args.required_policy_outcome_head or []),
            "min_policy_outcome_rows_per_required_head": int(args.min_policy_outcome_rows_per_required_head),
            "min_diagnostic_group_features": int(args.min_diagnostic_group_features),
            "min_diagnostic_group_finite_rate": float(args.min_diagnostic_group_finite_rate),
        },
        "bundles": list(args.bundle),
    }
    (out_dir / "latest_flat_frozen_gate_readiness.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Latest Flat Frozen Dual-Scoring Gate Readiness",
        "",
        f"Cutoff: `{args.cutoff}`",
        f"Ready sources: `{ready_count}`",
        f"Ran gate: `{gate_dir is not None}`",
        f"Gate directory: `{gate_dir or ''}`",
        "",
        "## Requirements",
        "",
        f"- Minimum post-cutoff rows: `{args.min_post_cutoff_rows}`",
        f"- Minimum post-cutoff timestamps: `{args.min_post_cutoff_timestamps}`",
        f"- Minimum post-cutoff active heads: `{args.min_post_cutoff_active_heads}`",
        f"- Minimum estimated policy-action rows: `{args.min_policy_action_rows}`",
        f"- Minimum estimated policy-action timestamps: `{args.min_policy_action_timestamps}`",
        f"- Minimum estimated matured policy-outcome rows: `{args.min_policy_outcome_rows}`",
        f"- Minimum estimated matured policy-outcome timestamps: `{args.min_policy_outcome_timestamps}`",
        f"- Minimum matured outcome rows per action head: `{args.min_policy_outcome_rows_per_action_head}`",
        f"- Required matured-outcome heads: `{', '.join(args.required_policy_outcome_head or [])}`",
        f"- Minimum matured outcome rows per required head: `{args.min_policy_outcome_rows_per_required_head}`",
        f"- Minimum diagnostic columns per reliability family: `{args.min_diagnostic_group_features}`",
        f"- Minimum post-cutoff finite row rate per reliability family: `{args.min_diagnostic_group_finite_rate}`",
        "",
        "## Selected Source",
        "",
    ]
    if selected is None:
        lines.append("_No source met the readiness requirements._")
    else:
        lines.append(pd.DataFrame([selected.to_dict()]).to_markdown(index=False))
    if nearest is not None:
        lines.extend(["", "## Nearest Source", "", pd.DataFrame([nearest.to_dict()]).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Source Scan",
            "",
            scan.sort_values("post_cutoff_rows", ascending=False).head(30).to_markdown(index=False)
            if not scan.empty
            else "_No sources scanned._",
        ]
    )
    (out_dir / "latest_flat_frozen_gate_readiness.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=None, help="Flat candidate parquet. Repeatable.")
    parser.add_argument("--root", action="append", default=None, help="Optional root to scan for flat candidates.")
    parser.add_argument("--bundle", action="append", required=True, help="label=path or path passed to the gate runner.")
    parser.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--min-post-cutoff-rows", type=int, default=2000)
    parser.add_argument("--min-post-cutoff-timestamps", type=int, default=40)
    parser.add_argument("--min-post-cutoff-active-heads", type=int, default=3)
    parser.add_argument(
        "--min-policy-action-rows",
        type=int,
        default=50,
        help=(
            "Minimum post-cutoff rows with replay auction evidence or live traded decisions before running replay. "
            "This is a preflight proxy for policy actions, not a substitute for replay-confirmed accepted trades."
        ),
    )
    parser.add_argument("--min-policy-action-timestamps", type=int, default=10)
    parser.add_argument(
        "--min-policy-outcome-rows",
        type=int,
        default=50,
        help="Minimum post-cutoff policy-action rows with finite return/PnL outcomes for PnL/tail evaluation.",
    )
    parser.add_argument("--min-policy-outcome-timestamps", type=int, default=10)
    parser.add_argument(
        "--min-policy-outcome-rows-per-action-head",
        type=int,
        default=0,
        help=(
            "Optional minimum matured outcome rows for every head that has post-cutoff policy-action evidence. "
            "Use this to prevent a total-row pass from hiding an unsupported active head."
        ),
    )
    parser.add_argument(
        "--required-policy-outcome-head",
        action="append",
        default=None,
        help="Head that must have matured outcome evidence before the gate can run. Repeatable.",
    )
    parser.add_argument(
        "--min-policy-outcome-rows-per-required-head",
        type=int,
        default=1,
        help="Minimum matured outcome rows required for each --required-policy-outcome-head.",
    )
    parser.add_argument("--min-diagnostic-group-features", type=int, default=1)
    parser.add_argument(
        "--min-diagnostic-group-finite-rate",
        type=float,
        default=0.25,
        help=(
            "Minimum post-cutoff row share with at least one finite value in each diagnostic family. "
            "This prevents empty drift/OOD/uncertainty/recent-HR columns from satisfying the contract."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Run on the best existing source even if not ready.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    rows = [_scan_one(path, cutoff, args) for path in _candidate_paths(args)]
    scan = pd.DataFrame(rows)
    selected = _select_source(scan, force=bool(args.force))
    gate_dir = None
    if selected is not None and (bool(selected.get("ready", False)) or args.force):
        gate_dir = _run_gate(args, selected, out_dir)
    _write_report(out_dir, args, scan, selected, gate_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
