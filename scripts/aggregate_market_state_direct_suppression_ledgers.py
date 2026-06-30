#!/usr/bin/env python3
"""Aggregate direct suppression ledgers across walk-forward and shadow windows.

The first direct-suppression ledger came from the controller walk-forward
directory, where accepted trades include an explicit S0 baseline arm and the
threshold schedule includes multiple controller arms. Later shadow-scored
windows have a different but valid contract: ``accepted_trades.parquet`` is the
baseline accepted population for that window and the schedule is a single
selected controller arm without an ``arm`` column.

This script normalizes both shapes into one chronological training ledger. It
does not select or promote a controller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

from scripts.build_market_state_direct_suppression_ledger import (  # noqa: E402
    BASELINE_ARM,
    build_direct_suppression_ledger,
)


DEFAULT_CONFIG = Path("config/reliability_blend_production_stack.json")
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_direct_suppression_ledger_globalrank_no_backfill_combined"
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _manifest_selected_arm(score_dir: Path) -> str | None:
    manifest = score_dir / "manifest.json"
    if not manifest.exists():
        return None
    payload = _load_json(manifest)
    selected = payload.get("selected_arm")
    return str(selected) if selected else None


def source_specs_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    controller = dict(config.get("market_state_controller_validation") or {})
    no_backfill = dict(controller.get("global_rank_threshold_controller_no_backfill_walkforward") or {})
    specs: list[dict[str, Any]] = []
    artifact_dir = no_backfill.get("artifact_dir")
    if artifact_dir:
        specs.append(
            {
                "source_dir": str(artifact_dir),
                "source_kind": "walkforward",
                "source_window_id": "walkforward_globalrank_no_backfill",
                "accepted_arm_mode": "filter_baseline_arm",
                "controller_arm_fallback": None,
            }
        )
    monitor = dict(controller.get("global_rank_threshold_controller_no_backfill_shadow_monitor") or {})
    windows = monitor.get("windows")
    if isinstance(windows, list):
        for idx, window in enumerate(windows, start=1):
            if not isinstance(window, dict) or not window.get("score_dir"):
                continue
            score_dir = Path(str(window["score_dir"]))
            specs.append(
                {
                    "source_dir": str(score_dir),
                    "source_kind": "later_shadow",
                    "source_window_id": f"later_shadow_{idx}",
                    "period_start": window.get("period_start"),
                    "period_end": window.get("period_end"),
                    "accepted_arm_mode": "all_accepted_as_baseline",
                    "controller_arm_fallback": window.get("selected_arm")
                    or _manifest_selected_arm(score_dir),
                }
            )
    return specs


def _safe_num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _assign_combined_folds(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["source_order"] = pd.to_numeric(out["source_order"], errors="coerce").fillna(0).astype(int)
    existing = pd.to_numeric(out.get("fold"), errors="coerce")
    out["fold"] = existing
    max_existing = int(existing.dropna().max()) if existing.notna().any() else 0
    for offset, source_order in enumerate(
        sorted(out.loc[out["fold"].isna(), "source_order"].unique()),
        start=1,
    ):
        out.loc[out["source_order"].eq(source_order) & out["fold"].isna(), "fold"] = (
            max_existing + offset
        )
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce").fillna(1).astype(int)
    out["fold_source"] = np.where(
        out["source_kind"].astype(str).eq("walkforward"),
        "source_schedule",
        "combined_chronological_source_window",
    )
    return out.sort_values(
        ["fold", "timestamp", "source_order", "controller_arm", "decision_key"]
    ).reset_index(drop=True)


def _direct_suppression_metrics(grouped: Any) -> pd.DataFrame:
    return grouped.agg(
        frontier_rows=("decision_key", "count"),
        unique_decision_keys=("decision_key", "nunique"),
        source_windows=("source_window_id", "nunique"),
        timestamp_count=("timestamp", "nunique"),
        direct_profitable_rate=("direct_suppression_profitable", "mean"),
        full_sl_rate=("direct_suppression_full_sl", "mean"),
        timeout_rate=("direct_suppression_timeout", "mean"),
        mean_direct_defensive_utility=("direct_defensive_utility", "mean"),
        total_direct_defensive_utility=("direct_defensive_utility", "sum"),
        current_schedule_suppressed_rows=("would_suppress_at_state_threshold", "sum"),
        current_schedule_defensive_utility=(
            "suppressed_defensive_utility_under_current_schedule",
            "sum",
        ),
        min_fold=("fold", "min"),
        max_fold=("fold", "max"),
    ).reset_index()


def aggregate_direct_suppression_ledgers(
    specs: list[dict[str, Any]],
    *,
    baseline_arm: str = BASELINE_ARM,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledgers: list[pd.DataFrame] = []
    source_summaries: list[dict[str, Any]] = []
    for source_order, spec in enumerate(specs, start=1):
        source_dir = Path(str(spec["source_dir"]))
        ledger, _by_group, summary = build_direct_suppression_ledger(
            source_dir,
            baseline_arm=baseline_arm,
            accepted_arm_mode=str(spec.get("accepted_arm_mode") or "filter_baseline_arm"),
            controller_arm_fallback=spec.get("controller_arm_fallback"),
            source_kind=spec.get("source_kind"),
            source_window_id=spec.get("source_window_id") or source_dir.name,
        )
        source_summary = {
            **summary,
            "source_order": source_order,
            "period_start": spec.get("period_start"),
            "period_end": spec.get("period_end"),
        }
        source_summaries.append(source_summary)
        if ledger.empty:
            continue
        ledger = ledger.copy()
        ledger["source_order"] = source_order
        ledger["period_start"] = spec.get("period_start")
        ledger["period_end"] = spec.get("period_end")
        ledgers.append(ledger)
    if not ledgers:
        empty = pd.DataFrame()
        summary = {
            "generated_by": "aggregate_market_state_direct_suppression_ledgers",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_contract": "direct_accepted_frontier_training_ledger_v1",
            "aggregation_contract": "combined_direct_accepted_frontier_training_ledger_v1",
            "row_count": 0,
            "source_count": len(specs),
            "source_summaries": source_summaries,
            "reason": "no_source_ledgers_with_rows",
        }
        return empty, empty, empty, empty, empty, summary

    combined = _assign_combined_folds(pd.concat(ledgers, ignore_index=True, sort=False))
    duplicate_key_cols = ["source_window_id", "controller_arm", "decision_key"]
    duplicate_rows = int(combined.duplicated(duplicate_key_cols).sum())
    by_group = _direct_suppression_metrics(
        combined.groupby(["controller_arm", "head"], dropna=False, sort=True)
    )
    by_strategy = _direct_suppression_metrics(
        combined.groupby(["controller_arm", "head", "strategy_id"], dropna=False, sort=True)
    )
    by_source = combined.groupby(
        ["source_order", "source_kind", "source_window_id"],
        dropna=False,
        sort=True,
    ).agg(
        rows=("decision_key", "count"),
        unique_decision_keys=("decision_key", "nunique"),
        timestamp_count=("timestamp", "nunique"),
        controller_arm_count=("controller_arm", "nunique"),
        direct_profitable_rate=("direct_suppression_profitable", "mean"),
        full_sl_rate=("direct_suppression_full_sl", "mean"),
        timeout_rate=("direct_suppression_timeout", "mean"),
        mean_direct_defensive_utility=("direct_defensive_utility", "mean"),
        total_direct_defensive_utility=("direct_defensive_utility", "sum"),
        current_schedule_suppressed_rows=("would_suppress_at_state_threshold", "sum"),
        current_schedule_defensive_utility=(
            "suppressed_defensive_utility_under_current_schedule",
            "sum",
        ),
        min_fold=("fold", "min"),
        max_fold=("fold", "max"),
    ).reset_index()
    by_source_strategy = _direct_suppression_metrics(
        combined.groupby(
            ["source_order", "source_kind", "source_window_id", "controller_arm", "head", "strategy_id"],
            dropna=False,
            sort=True,
        )
    )
    summary = {
        "generated_by": "aggregate_market_state_direct_suppression_ledgers",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_contract": "direct_accepted_frontier_training_ledger_v1",
        "aggregation_contract": "combined_direct_accepted_frontier_training_ledger_v1",
        "baseline_arm": str(baseline_arm),
        "source_count": len(specs),
        "source_with_rows_count": len(ledgers),
        "row_count": int(len(combined)),
        "unique_decision_key_count": int(combined["decision_key"].nunique()),
        "duplicate_source_controller_decision_key_rows": duplicate_rows,
        "timestamp_count": int(combined["timestamp"].nunique()),
        "folds": sorted(int(f) for f in combined["fold"].dropna().unique()),
        "controller_arm_count": int(combined["controller_arm"].nunique()),
        "direct_profitable_rate": float(combined["direct_suppression_profitable"].mean()),
        "full_sl_rate": float(combined["direct_suppression_full_sl"].mean()),
        "timeout_rate": float(combined["direct_suppression_timeout"].mean()),
        "mean_direct_defensive_utility": float(combined["direct_defensive_utility"].mean()),
        "total_direct_defensive_utility": float(combined["direct_defensive_utility"].sum()),
        "current_schedule_suppressed_rows": int(combined["would_suppress_at_state_threshold"].sum()),
        "current_schedule_defensive_utility": float(
            combined["suppressed_defensive_utility_under_current_schedule"].sum()
        ),
        "source_summaries": source_summaries,
        "interpretation": (
            "Combined training evidence for a shadow direct accepted-frontier suppression "
            "controller. Later windows are normalized as all accepted rows equal the "
            "baseline accepted set; this remains a training artifact, not a promotion."
        ),
    }
    return combined, by_group, by_strategy, by_source, by_source_strategy, summary


def _render_report(
    summary: dict[str, Any],
    by_group: pd.DataFrame,
    by_strategy: pd.DataFrame,
    by_source: pd.DataFrame,
    by_source_strategy: pd.DataFrame,
) -> str:
    lines = [
        "# Combined Direct Suppression Training Ledger",
        "",
        "This combines the original walk-forward direct-suppression surface with later shadow-scored windows.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Unique decision keys: `{summary.get('unique_decision_key_count')}`",
        f"- Timestamps: `{summary.get('timestamp_count')}`",
        f"- Folds: `{summary.get('folds')}`",
        f"- Sources with rows: `{summary.get('source_with_rows_count')}`",
        f"- Direct profitable suppression rate: `{summary.get('direct_profitable_rate')}`",
        f"- Mean direct defensive utility: `{summary.get('mean_direct_defensive_utility')}`",
        f"- Total direct defensive utility: `{summary.get('total_direct_defensive_utility')}`",
        f"- Current schedule suppressed rows: `{summary.get('current_schedule_suppressed_rows')}`",
        f"- Current schedule defensive utility: `{summary.get('current_schedule_defensive_utility')}`",
        "",
        "## By Source",
        "",
        by_source.to_markdown(index=False) if not by_source.empty else "_No source metrics._",
        "",
        "## By Arm / Head",
        "",
        by_group.to_markdown(index=False) if not by_group.empty else "_No grouped metrics._",
        "",
        "## By Arm / Head / Strategy",
        "",
        by_strategy.to_markdown(index=False) if not by_strategy.empty else "_No strategy metrics._",
        "",
        "## By Source / Arm / Head / Strategy",
        "",
        by_source_strategy.to_markdown(index=False)
        if not by_source_strategy.empty
        else "_No source-strategy metrics._",
        "",
        "## Contract",
        "",
        "- Keeps the active T1 score/rank/auction contract unchanged.",
        "- Walk-forward sources filter explicit S0 baseline accepted rows.",
        "- Later shadow sources use the scored accepted-trades file as the baseline accepted set.",
        "- Controller action remains threshold-raise-only and shadow-only.",
    ]
    return "\n".join(lines) + "\n"


def write_combined_ledger(
    ledger: pd.DataFrame,
    by_group: pd.DataFrame,
    by_strategy: pd.DataFrame,
    by_source: pd.DataFrame,
    by_source_strategy: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ledger_parquet": output_dir / "direct_accepted_frontier_training_ledger.parquet",
        "ledger_csv": output_dir / "direct_accepted_frontier_training_ledger.csv",
        "by_group_csv": output_dir / "direct_accepted_frontier_training_by_arm_head.csv",
        "by_strategy_csv": output_dir / "direct_accepted_frontier_training_by_arm_head_strategy.csv",
        "by_source_csv": output_dir / "direct_accepted_frontier_training_by_source.csv",
        "by_source_strategy_csv": output_dir
        / "direct_accepted_frontier_training_by_source_arm_head_strategy.csv",
        "summary_json": output_dir / "direct_accepted_frontier_training_summary.json",
        "report_md": output_dir / "direct_accepted_frontier_training_report.md",
    }
    ledger.to_parquet(paths["ledger_parquet"], index=False)
    ledger.to_csv(paths["ledger_csv"], index=False)
    by_group.to_csv(paths["by_group_csv"], index=False)
    by_strategy.to_csv(paths["by_strategy_csv"], index=False)
    by_source.to_csv(paths["by_source_csv"], index=False)
    by_source_strategy.to_csv(paths["by_source_strategy_csv"], index=False)
    payload = {**summary, "outputs": {key: str(path) for key, path in paths.items()}}
    paths["summary_json"].write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    paths["report_md"].write_text(
        _render_report(payload, by_group, by_strategy, by_source, by_source_strategy),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-arm", default=BASELINE_ARM)
    args = parser.parse_args()

    specs = source_specs_from_config(_load_json(args.config))
    ledger, by_group, by_strategy, by_source, by_source_strategy, summary = aggregate_direct_suppression_ledgers(
        specs,
        baseline_arm=args.baseline_arm,
    )
    outputs = write_combined_ledger(
        ledger,
        by_group,
        by_strategy,
        by_source,
        by_source_strategy,
        summary,
        args.output_dir,
    )
    print(json.dumps(_json_safe({**summary, "outputs": outputs}), indent=2))


if __name__ == "__main__":
    main()
