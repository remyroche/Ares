#!/usr/bin/env python3
"""Scan candidate ledgers for wf_recent combo prospective dual-scoring readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_apply_wfrecent_smooth_penalty_bundle import _raw_columns, _sha256_file  # noqa: E402
from scripts.validate_wfrecent_row_guard_walkforward import RAW_GROUPS, _fmt_table, _json_safe  # noqa: E402


DEFAULT_PATHS = [
    "data_perp/reports/contextual_tp_sl_latest_jun26_28_wf_recent_20260701/combo_candidates.parquet",
    "data_perp/reports/contextual_tp_sl_latest_jun26_28_static_20260701/combo_candidates.parquet",
    "data_perp/reports/contextual_tp_sl_latest_jun26_28_best_net_20260701/combo_candidates.parquet",
    "data_perp/reports/contextual_tp_sl_latest_jun26_28_best_balanced_20260701/combo_candidates.parquet",
    "data_perp/reports/portfolio_marginal_utility_ablation_20260701_jun21_27_inputs/simple_policy_candidates.parquet",
    "data_perp/reports/portfolio_marginal_utility_ablation_20260701_jun21_27_inputs/simple_policy_candidates_broad.parquet",
    "data_perp/reports/portfolio_marginal_utility_ablation_20260701_jun21_27_inputs/simple_policy_candidates_deployable.parquet",
    "data_perp/reports/exact_state_size_action_learning_20260628_postjun26_c3el_inputs/simple_policy_candidates_broad_postjun26_08_to_jun27_12.parquet",
    "data_perp/reports/exact_state_size_action_learning_20260628_postjun26_c3el_inputs/simple_policy_candidates_deployable_hist_to_postjun26_08_to_jun27_12.parquet",
    "data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet",
]

BASE_COLUMNS = ("timestamp", "strategy_id", "symbol")
RANK_COLUMNS = ("rank_pct", "policy_rank_pct", "normalized_rank_score")


def _group_cols(group: str) -> list[str]:
    return [col for col, _invert in RAW_GROUPS[group]]


def _read_head(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _scan_one(path: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "eligible_for_selected_bundle": False,
            "rejection_reasons": "missing_file",
        }
    try:
        frame = _read_head(path)
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "eligible_for_selected_bundle": False,
            "rejection_reasons": f"read_error:{type(exc).__name__}:{exc}",
        }

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") if "timestamp" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    post_mask = timestamp.ge(cutoff)
    raw_cols = _raw_columns()
    present_raw = [col for col in raw_cols if col in frame.columns]
    missing_raw = [col for col in raw_cols if col not in frame.columns]
    base_missing = [col for col in BASE_COLUMNS if col not in frame.columns]
    has_rank = any(col in frame.columns for col in RANK_COLUMNS)
    generated_cols = [col for col in frame.columns if str(col).startswith("generated_")]

    group_present_counts = {}
    group_missing = {}
    for group in RAW_GROUPS:
        cols = _group_cols(group)
        group_present_counts[group] = int(sum(col in frame.columns for col in cols))
        group_missing[group] = [col for col in cols if col not in frame.columns]

    required_selected_groups = ("drift_risk",)
    selected_group_ok = all(group_present_counts[group] == len(_group_cols(group)) for group in required_selected_groups)
    composite_contract_ok = all(group_present_counts[group] > 0 for group in RAW_GROUPS)
    raw_contract_ok = len(missing_raw) == 0
    post_rows = int(post_mask.sum()) if len(frame) else 0
    reasons = []
    if len(frame) == 0:
        reasons.append("empty")
    if base_missing:
        reasons.append("missing_base_columns:" + ",".join(base_missing))
    if not has_rank:
        reasons.append("missing_rank_column")
    if post_rows <= 0:
        reasons.append("no_rows_at_or_after_cutoff")
    if not selected_group_ok:
        reasons.append("missing_selected_drift_columns")
    if not composite_contract_ok:
        reasons.append("composite_risk_has_empty_raw_groups")
    if not raw_contract_ok:
        reasons.append("missing_raw_diagnostic_columns")

    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "timestamp_min": timestamp.min().isoformat() if timestamp.notna().any() else "",
        "timestamp_max": timestamp.max().isoformat() if timestamp.notna().any() else "",
        "timestamp_count": int(timestamp.nunique(dropna=True)),
        "post_cutoff_rows": post_rows,
        "generated_column_count": int(len(generated_cols)),
        "raw_diagnostic_columns_present": int(len(present_raw)),
        "raw_diagnostic_columns_required": int(len(raw_cols)),
        "missing_raw_diagnostic_count": int(len(missing_raw)),
        "drift_columns_present": group_present_counts["drift_risk"],
        "drift_columns_required": len(_group_cols("drift_risk")),
        "composite_nonempty_groups": int(sum(count > 0 for count in group_present_counts.values())),
        "composite_total_groups": int(len(RAW_GROUPS)),
        "has_rank_column": bool(has_rank),
        "base_columns_missing": ",".join(base_missing),
        "eligible_for_selected_bundle": len(reasons) == 0,
        "rejection_reasons": ";".join(reasons),
        "missing_raw_diagnostic_columns": ",".join(missing_raw[:40]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_prospective_source_scan_20260701"),
    )
    parser.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    rows = [_scan_one(Path(path), cutoff) for path in args.paths]
    scan = pd.DataFrame(rows)
    scan.to_csv(args.output_dir / "prospective_source_scan.csv", index=False)

    eligible = scan[scan["eligible_for_selected_bundle"].eq(True)].copy() if not scan.empty else pd.DataFrame()
    manifest = {
        "generated_by": "scan_wfrecent_combo_prospective_sources",
        "cutoff": cutoff.isoformat(),
        "paths_scanned": int(len(scan)),
        "eligible_sources": int(len(eligible)),
        "default_paths": DEFAULT_PATHS,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Combo Prospective Source Scan",
        "",
        "Scans candidate/prediction ledgers for the columns required to apply the selected frozen smooth-penalty challenger. This report does not score or replay anything.",
        "",
        f"Cutoff: `{cutoff.isoformat()}`",
        f"Sources scanned: `{len(scan)}`",
        f"Eligible sources: `{len(eligible)}`",
        "",
        "## Source Status",
        "",
        _fmt_table(
            scan,
            [
                "path",
                "rows",
                "timestamp_min",
                "timestamp_max",
                "post_cutoff_rows",
                "generated_column_count",
                "raw_diagnostic_columns_present",
                "raw_diagnostic_columns_required",
                "drift_columns_present",
                "drift_columns_required",
                "eligible_for_selected_bundle",
                "rejection_reasons",
            ],
        ),
        "",
        "## Readout",
        "",
    ]
    if eligible.empty:
        lines.extend(
            [
                "- No currently scanned post-cutoff source can be used for selected-candidate dual scoring.",
                "- The available post-cutoff candidate ledgers lack the generated diagnostic columns used by the frozen smooth-penalty bundle.",
                "- Next implementation step: add the generated diagnostic columns to live/prospective candidate materialization, then apply the frozen v2 bundle without changing thresholds.",
            ]
        )
    else:
        lines.append("- At least one source is eligible for selected-candidate dual scoring.")
    (args.output_dir / "prospective_source_scan_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
