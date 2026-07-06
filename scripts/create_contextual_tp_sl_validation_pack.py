#!/usr/bin/env python3
"""Create a fixed validation pack for contextual TP/SL diagnostic candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = [
    {
        "label": "longbars_drift_only",
        "role": "risk_controlled_default_and_fallback",
        "variant": "longbars_drift_only",
        "reason": "Best production-pre-OOS and drift-fallback balance; positive Apr-Jun months 3/3.",
    },
    {
        "label": "longbars_weekgate_only",
        "role": "high_upside_challenger",
        "variant": "longbars_weekgate_only",
        "reason": "Highest raw daily-weekly and monthly PnL objective; weaker recurrence.",
    },
    {
        "label": "longbars_uncertainty_only",
        "role": "secondary_pnl_tail_challenger",
        "variant": "longbars_uncertainty_only",
        "reason": "Best all-window gated daily-weekly PnL-tail candidate; less stable in temporal selection.",
    },
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
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


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _candidate_metrics(root: Path) -> pd.DataFrame:
    final_path = root / "final_candidates" / "final_candidate_comparison.csv"
    monthly_path = root / "monthly_selector_drift_fallback" / "monthly_selector_static_variant_summary.csv"
    final = _read_csv(final_path)
    monthly = _read_csv(monthly_path)
    keep = [c["label"] for c in DEFAULT_CANDIDATES]
    if final.empty:
        return pd.DataFrame()
    final = final[final["label"].isin(keep)].copy()
    if not monthly.empty:
        monthly = monthly[monthly["label"].isin(keep)].add_prefix("apr_jun_")
        final = final.merge(monthly, left_on="label", right_on="apr_jun_label", how="left")
    final["role"] = final["label"].map({c["label"]: c["role"] for c in DEFAULT_CANDIDATES})
    return final


def _source_scan(scan_dir: Path) -> Dict[str, Any]:
    scan_json = scan_dir / "contextual_tp_sl_candidate_source_scan.json"
    if scan_json.exists():
        return json.loads(scan_json.read_text(encoding="utf-8"))
    scan_csv = scan_dir / "contextual_tp_sl_candidate_source_scan.csv"
    if scan_csv.exists():
        frame = pd.read_csv(scan_csv)
        return {"sources": frame.to_dict(orient="records"), "source_count": int(len(frame))}
    return {"sources": [], "source_count": 0}


def _closest_source_frame(source_scan: Dict[str, Any]) -> pd.DataFrame:
    sources = source_scan.get("sources") or []
    if not sources:
        return pd.DataFrame()
    frame = pd.DataFrame(sources)
    for col in [
        "post_cutoff_rows",
        "post_cutoff_timestamps",
        "post_cutoff_active_heads",
        "post_cutoff_rows_needed",
        "post_cutoff_timestamps_needed",
        "post_cutoff_active_heads_needed",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    sort_cols = [
        col
        for col in [
            "usable_post_cutoff",
            "has_required_diagnostic_groups",
            "has_required_columns",
            "post_cutoff_rows",
            "post_cutoff_timestamps",
            "candidate_end",
        ]
        if col in frame.columns
    ]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    keep = [
        "source_dir",
        "candidate_end",
        "has_required_columns",
        "missing_required_columns",
        "has_required_diagnostic_groups",
        "missing_required_diagnostic_groups",
        "diagnostic_group_coverage",
        "post_cutoff_rows",
        "post_cutoff_timestamps",
        "post_cutoff_active_heads",
        "post_cutoff_rows_needed",
        "post_cutoff_timestamps_needed",
        "post_cutoff_active_heads_needed",
        "usable_post_cutoff",
    ]
    return frame[[col for col in keep if col in frame.columns]].head(5)


def _readiness_status(source_scan: Dict[str, Any], closest_sources: pd.DataFrame) -> tuple[str, List[str]]:
    if int(source_scan.get("usable_post_cutoff_count") or 0) > 0:
        return "ready_for_forward_replay", []
    blockers: List[str] = []
    if closest_sources.empty:
        return "blocked_no_candidate_sources_found", ["no_candidate_sources_found"]
    all_sources = pd.DataFrame(source_scan.get("sources") or [])
    needed_cols = [
        "post_cutoff_rows_needed",
        "post_cutoff_timestamps_needed",
        "post_cutoff_active_heads_needed",
    ]
    if any(col in closest_sources.columns and pd.to_numeric(closest_sources[col], errors="coerce").fillna(0).gt(0).any() for col in needed_cols):
        blockers.append("insufficient_post_cutoff_candidate_coverage")
    if (
        "has_required_columns" in all_sources.columns
        and not all_sources["has_required_columns"].astype(str).str.lower().eq("true").any()
    ):
        blockers.append("missing_required_columns")
    if (
        "has_required_diagnostic_groups" in all_sources.columns
        and not all_sources["has_required_diagnostic_groups"].astype(str).str.lower().eq("true").any()
    ):
        blockers.append("missing_required_diagnostic_group_coverage")
    if not blockers:
        blockers.append("no_usable_post_cutoff_source")
    return "blocked_" + "_and_".join(blockers), blockers


def _preferred_source_dir(args: argparse.Namespace, closest_sources: pd.DataFrame) -> str:
    if closest_sources.empty or "source_dir" not in closest_sources.columns:
        return str(args.source_dir)
    candidates = closest_sources.copy()
    if "has_required_diagnostic_groups" in candidates.columns:
        ready = candidates["has_required_diagnostic_groups"].astype(str).str.lower().eq("true")
        if ready.any():
            return str(candidates.loc[ready, "source_dir"].iloc[0])
    return str(candidates["source_dir"].iloc[0])


def _command_lines(args: argparse.Namespace, variants: List[str], source_dir: str) -> Dict[str, str]:
    variant_csv = ",".join(variants)
    base = (
        "python3 scripts/run_contextual_tp_sl_combined_monthly_walkforward.py "
        f"--source-dir {source_dir} "
        f"--out-dir {args.next_replay_out_dir} "
        f"--start-month {args.next_start_month} "
        f"--end-month {args.next_end_month} "
        f"--combo-id {args.combo_id} "
        f"--weekly-gate-path {args.weekly_gate_path} "
        f"--market-mode {args.market_mode} "
        f"--variants {variant_csv}"
    )
    summarize = base + " --summarize-only"
    return {
        "run_next_monthly_replay": base,
        "summarize_existing_next_replay": summarize,
        "run_daily_weekly_gate_after_replay": (
            "python3 scripts/gate_contextual_tp_sl_daily_weekly.py "
            f"--global-csv {args.next_replay_out_dir}/monthly_walkforward_global.csv "
            f"--out-dir {args.next_replay_out_dir}/daily_weekly_gate"
        ),
        "run_drift_fallback_selector_after_replay": (
            "python3 scripts/validate_contextual_tp_sl_monthly_selector.py "
            f"--daily-csv {args.next_replay_out_dir}/daily_weekly_gate/daily_all_variant_metrics.csv "
            f"--weekly-csv {args.next_replay_out_dir}/daily_weekly_gate/weekly_all_variant_metrics.csv "
            f"--out-dir {args.next_replay_out_dir}/monthly_selector_drift_fallback "
            "--fallback-label longbars_drift_only"
        ),
    }


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        vals: List[str] = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--source-dir", default="data_perp/reports/contextual_tp_sl_ablation_q35w07_q20w03_6mo_diagperf_20260630")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weekly-gate-path", default="data_perp/reports/contextual_tp_sl_dynamic_longbars_gate_6mo_20260701/materialized/net_lt_1000_lb2/head_gate_weeks.csv")
    parser.add_argument("--combo-id", default="long_bars:S_long_dist:R_short_asset:R_short_bollinger:J")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--next-start-month", default="2026-07")
    parser.add_argument("--next-end-month", default="2026-07")
    parser.add_argument("--next-replay-out-dir", default="data_perp/reports/contextual_tp_sl_diagnostic_family_forward_replay_20260701")
    parser.add_argument("--source-scan-dir", default="")
    args = parser.parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_scan_dir = Path(args.source_scan_dir) if args.source_scan_dir else out_dir / "source_scan"

    metrics = _candidate_metrics(root)
    metrics.to_csv(out_dir / "fixed_candidate_metrics.csv", index=False)
    variants = [c["variant"] for c in DEFAULT_CANDIDATES]
    source_scan = _source_scan(source_scan_dir)
    closest_sources = _closest_source_frame(source_scan)
    if not closest_sources.empty:
        closest_sources.to_csv(out_dir / "forward_readiness_closest_sources.csv", index=False)
    ready_count = int(source_scan.get("usable_post_cutoff_count") or 0)
    forward_ready = ready_count > 0
    status, readiness_blockers = _readiness_status(source_scan, closest_sources)
    effective_source_dir = _preferred_source_dir(args, closest_sources)
    commands = _command_lines(args, variants, effective_source_dir)

    evidence_files = [
        root / "final_candidates" / "final_candidate_comparison_report.md",
        root / "daily_weekly_gate" / "daily_weekly_gate_report.md",
        root / "promotion_gate" / "promotion_gate_report.md",
        root / "temporal_selection" / "temporal_selection_report.md",
        root / "monthly_selector_drift_fallback" / "monthly_selector_report.md",
    ]
    evidence = [
        {"path": str(path), "sha256": _sha256(path), "exists": path.exists()}
        for path in evidence_files
    ]
    pack: Dict[str, Any] = {
        "generated_by": "create_contextual_tp_sl_validation_pack",
        "root_dir": str(root),
        "source_dir": str(args.source_dir),
        "effective_replay_source_dir": str(effective_source_dir),
        "out_dir": str(out_dir),
        "combo_id": str(args.combo_id),
        "weekly_gate_path": str(args.weekly_gate_path),
        "market_mode": str(args.market_mode),
        "candidates": DEFAULT_CANDIDATES,
        "variants_for_next_replay": variants,
        "next_replay_window": {
            "start_month": str(args.next_start_month),
            "end_month": str(args.next_end_month),
            "out_dir": str(args.next_replay_out_dir),
        },
        "source_scan": source_scan,
        "closest_sources": closest_sources.to_dict(orient="records") if not closest_sources.empty else [],
        "forward_ready": forward_ready,
        "readiness_blockers": readiness_blockers,
        "commands": commands,
        "evidence": evidence,
        "status": status,
    }
    (out_dir / "fixed_validation_pack.json").write_text(
        json.dumps(_json_safe(pack), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metric_cols = [
        "label",
        "role",
        "daily_weekly_objective",
        "sum_delta_net_pnl",
        "positive_week_count",
        "q20_day_delta_net_pnl",
        "mean_day_full_sl_delta",
        "apr_jun_sum_eval_delta_net_pnl",
        "apr_jun_positive_eval_months",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
    lines = [
        "# Contextual TP/SL Fixed Validation Pack",
        "",
        "This freezes the current diagnostic TP/SL candidates for the next replay. It does not run a replay.",
        "",
        f"Status: `{pack['status']}`",
        f"Forward ready: `{forward_ready}`",
        f"Source directory: `{args.source_dir}`",
        f"Effective replay source directory: `{effective_source_dir}`",
        f"Next replay window: `{args.next_start_month}` to `{args.next_end_month}`",
        "",
        "## Candidates",
        "",
        _markdown_table(metrics, metric_cols),
        "",
        "## Commands",
        "",
    ]
    for name, command in commands.items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend(
        [
        "## Source Readiness",
        "",
        f"Usable post-cutoff source count: `{ready_count}`",
        f"Readiness blockers: `{', '.join(readiness_blockers) or 'none'}`",
        "",
            "Closest sources by post-cutoff coverage:",
            "",
            _markdown_table(closest_sources, list(closest_sources.columns)) if not closest_sources.empty else "_No sources found._",
            "",
            "## Evidence Files",
            "",
            _markdown_table(pd.DataFrame(evidence), ["path", "exists", "sha256"]),
            "",
        ]
    )
    (out_dir / "fixed_validation_pack.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "fixed_validation_pack.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
