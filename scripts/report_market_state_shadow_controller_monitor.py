#!/usr/bin/env python3
"""Aggregate Stage 2 market-state controller shadow evidence.

This report is deliberately monitoring-only.  It inspects one or more
materialized/scored shadow controller bundles, verifies that executed schedules
remained no-op when the controller was disabled, and summarizes the proposed
threshold raises against matured outcomes.
"""

from __future__ import annotations

import argparse
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

from scripts import audit_market_state_controller_contract as contract_audit  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_shadow_controller_monitor")
EPS = 1e-12
DEFAULT_MIN_DEFENSIVE_POSITIVE_BUNDLE_SHARE = 0.75
DEFAULT_MIN_BUNDLE_COUNT = 1


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").astype(str)
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _first_scope(frame: pd.DataFrame, scope: str = "all", scope_value: str = "all") -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    if {"scope", "scope_value"}.issubset(frame.columns):
        mask = frame["scope"].astype(str).eq(scope) & frame["scope_value"].astype(str).eq(scope_value)
        if mask.any():
            return frame.loc[mask].iloc[0]
    return frame.iloc[0]


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    if row.empty or key not in row:
        return float(default)
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(float(value)) else float(default)


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _artifact_kind(manifest: dict[str, Any]) -> str:
    generated_by = str(manifest.get("generated_by", ""))
    if generated_by == "materialize_market_state_controller_bundle":
        return "materialized_bundle"
    if generated_by == "score_market_state_controller_bundle":
        return "scored_bundle"
    return generated_by or "unknown"


def _controller_execution_enabled(manifest: dict[str, Any]) -> bool:
    controller = dict(manifest.get("controller") or {})
    return bool(
        manifest.get(
            "controller_execution_enabled",
            controller.get("controller_execution_enabled", controller.get("execution_enabled", True)),
        )
    )


def _shadow_controller_only(manifest: dict[str, Any]) -> bool:
    controller = dict(manifest.get("controller") or {})
    return bool(manifest.get("shadow_controller_only", controller.get("shadow_controller_only", False)))


def _safe_threshold_delta(frame: pd.DataFrame) -> tuple[bool, float, int]:
    if frame.empty:
        return False, 0.0, 0
    if {"base_threshold", "state_threshold"}.issubset(frame.columns):
        delta = pd.to_numeric(frame["state_threshold"], errors="coerce") - pd.to_numeric(
            frame["base_threshold"],
            errors="coerce",
        )
        finite_delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_delta.empty:
            return False, 0.0, 0
        return bool((finite_delta >= -EPS).all()), float(finite_delta.max()), int((finite_delta > EPS).sum())
    row = _first_scope(frame)
    max_delta = _num(row, "max_threshold_delta")
    raised = int(round(_num(row, "threshold_raised_count")))
    return bool(max_delta >= -EPS), float(max_delta), raised


def _time_span(*frames: pd.DataFrame) -> tuple[str | None, str | None]:
    values: list[pd.Series] = []
    for frame in frames:
        if not frame.empty and "timestamp" in frame.columns:
            values.append(pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"))
        if not frame.empty and {"first_timestamp", "last_timestamp"}.issubset(frame.columns):
            values.append(pd.to_datetime(frame["first_timestamp"], utc=True, errors="coerce"))
            values.append(pd.to_datetime(frame["last_timestamp"], utc=True, errors="coerce"))
    if not values:
        return None, None
    merged = pd.concat(values, ignore_index=True).dropna()
    if merged.empty:
        return None, None
    return merged.min().isoformat(), merged.max().isoformat()


def _head_rows(bundle_dir: Path, replay_by_head: pd.DataFrame, suppression: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not replay_by_head.empty and "head" in replay_by_head.columns:
        for _, row in replay_by_head.iterrows():
            head = str(row.get("head"))
            rec = {
                "bundle_dir": str(bundle_dir),
                "head": head,
                "trade_count": _num(row, "trade_count"),
                "win_rate": _num(row, "win_rate", np.nan),
                "net_pnl": _num(row, "net_pnl"),
                "gross_pnl": _num(row, "gross_pnl"),
                "cost_pnl": _num(row, "cost_pnl"),
                "full_sl_rate": _num(row, "full_sl_rate", np.nan),
                "timeout_rate": _num(row, "timeout_rate", np.nan),
            }
            rows.append(rec)
    if not suppression.empty and {"scope", "scope_value"}.issubset(suppression.columns):
        by_head = suppression.loc[suppression["scope"].astype(str).eq("head")].copy()
        for _, row in by_head.iterrows():
            head = str(row.get("scope_value"))
            matches = [r for r in rows if r["head"] == head]
            if not matches:
                matches = [{"bundle_dir": str(bundle_dir), "head": head}]
                rows.append(matches[0])
            rec = matches[0]
            rec.update(
                {
                    "shadow_suppressed_candidates": _num(row, "suppressed_candidates"),
                    "shadow_loss_avoided": _num(row, "suppressed_loss_avoided"),
                    "shadow_winner_pnl_sacrificed": _num(row, "suppressed_winner_pnl_sacrificed"),
                    "shadow_realized_defensive_success": _num(row, "realized_defensive_success"),
                    "shadow_suppressed_win_rate": _num(row, "suppressed_win_rate", np.nan),
                    "shadow_suppressed_full_sl_rate": _num(row, "suppressed_full_sl_rate", np.nan),
                    "shadow_suppressed_timeout_rate": _num(row, "suppressed_timeout_rate", np.nan),
                }
            )
    return pd.DataFrame(rows)


def summarize_shadow_bundle(bundle_dir: Path, *, run_artifact_audit: bool = True) -> dict[str, Any]:
    manifest = _load_json(bundle_dir / "manifest.json")
    artifact_kind = _artifact_kind(manifest)
    execution_enabled = _controller_execution_enabled(manifest)
    shadow_only = _shadow_controller_only(manifest)

    audit_failures: list[str] = []
    if run_artifact_audit:
        try:
            audit_failures = contract_audit.audit_manifest(manifest)
            audit_failures.extend(contract_audit.audit_artifacts(bundle_dir))
        except Exception as exc:  # pragma: no cover - defensive CLI guard
            audit_failures = [f"artifact audit raised {type(exc).__name__}: {exc}"]

    replay_summary = _read_frame(bundle_dir / "controller_replay_summary.csv")
    replay_by_head = _read_frame(bundle_dir / "controller_replay_by_head.csv")
    schedule = _read_frame(bundle_dir / "strategy_threshold_schedule.parquet")
    action_audit = _read_frame(bundle_dir / "strategy_threshold_action_audit.csv")
    proposed_schedule = _read_frame(bundle_dir / "shadow_controller_proposed_schedule.parquet")
    shadow_action = _read_frame(bundle_dir / "shadow_threshold_action_audit.csv")
    suppression = _read_frame(bundle_dir / "shadow_threshold_candidate_suppression_utility.csv")
    accepted = _read_frame(bundle_dir / "accepted_trades.parquet")
    timestamp_panel = _read_frame(bundle_dir / "market_state_timestamp_panel.parquet")

    replay_row = _first_scope(replay_summary)
    applied_row = _first_scope(action_audit)
    shadow_row = _first_scope(shadow_action)
    suppression_all = _first_scope(suppression)
    applied_safe, applied_max_delta_from_schedule, applied_raised_from_schedule = _safe_threshold_delta(schedule)
    shadow_safe, shadow_max_delta_from_schedule, shadow_raised_from_schedule = _safe_threshold_delta(proposed_schedule)
    start_ts, end_ts = _time_span(timestamp_panel, schedule, proposed_schedule, action_audit)

    applied_raises = max(int(round(_num(applied_row, "threshold_raised_count"))), applied_raised_from_schedule)
    applied_max_delta = max(_num(applied_row, "max_threshold_delta"), applied_max_delta_from_schedule)
    shadow_raises = max(int(round(_num(shadow_row, "threshold_raised_count"))), shadow_raised_from_schedule)
    shadow_max_delta = max(_num(shadow_row, "max_threshold_delta"), shadow_max_delta_from_schedule)
    suppressed_candidates = _num(suppression_all, "suppressed_candidates")
    loss_avoided = _num(suppression_all, "suppressed_loss_avoided")
    winner_sacrificed = _num(suppression_all, "suppressed_winner_pnl_sacrificed")
    defensive_success = _num(suppression_all, "realized_defensive_success")

    applied_noop_pass = bool(
        (execution_enabled or (applied_safe and applied_raises == 0 and abs(applied_max_delta) <= EPS))
    )
    shadow_schedule_safe = bool(shadow_only and shadow_safe and not proposed_schedule.empty and shadow_max_delta >= -EPS)
    coverage_ok = bool(shadow_only and not proposed_schedule.empty and not suppression.empty)
    defensive_positive = bool(
        coverage_ok
        and suppressed_candidates > 0
        and defensive_success > 0.0
        and loss_avoided > winner_sacrificed
    )

    summary = {
        "bundle_dir": str(bundle_dir),
        "artifact_kind": artifact_kind,
        "selected_arm": manifest.get("selected_arm"),
        "controller_execution_enabled": bool(execution_enabled),
        "shadow_controller_only": bool(shadow_only),
        "controller_enabled_heads": _str_list(manifest.get("controller_enabled_heads")),
        "shadow_controller_enabled_heads": _str_list(manifest.get("shadow_controller_enabled_heads")),
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "artifact_audit_run": bool(run_artifact_audit),
        "artifact_audit_passed": bool(not audit_failures) if run_artifact_audit else None,
        "artifact_audit_failures": audit_failures,
        "replay_trade_count": _num(replay_row, "trade_count"),
        "replay_net_pnl": _num(replay_row, "net_pnl"),
        "replay_gross_pnl": _num(replay_row, "gross_pnl"),
        "replay_cost_pnl": _num(replay_row, "cost_pnl"),
        "replay_full_sl_rate": _num(replay_row, "full_sl_rate", np.nan),
        "replay_timeout_rate": _num(replay_row, "timeout_rate", np.nan),
        "accepted_trade_count": int(len(accepted)) if not accepted.empty else 0,
        "applied_schedule_rows": int(len(schedule)),
        "applied_threshold_raises": int(applied_raises),
        "applied_mean_threshold_delta": _num(applied_row, "mean_threshold_delta"),
        "applied_max_threshold_delta": float(applied_max_delta),
        "shadow_schedule_rows": int(len(proposed_schedule)),
        "shadow_threshold_raises": int(shadow_raises),
        "shadow_mean_threshold_delta": _num(shadow_row, "mean_threshold_delta"),
        "shadow_max_threshold_delta": float(shadow_max_delta),
        "shadow_suppressed_candidates": float(suppressed_candidates),
        "shadow_loss_avoided": float(loss_avoided),
        "shadow_winner_pnl_sacrificed": float(winner_sacrificed),
        "shadow_realized_defensive_success": float(defensive_success),
        "shadow_defensive_success_per_candidate": (
            float(defensive_success / suppressed_candidates) if suppressed_candidates > 0 else 0.0
        ),
        "shadow_suppressed_win_rate": _num(suppression_all, "suppressed_win_rate", np.nan),
        "shadow_suppressed_full_sl_rate": _num(suppression_all, "suppressed_full_sl_rate", np.nan),
        "shadow_suppressed_timeout_rate": _num(suppression_all, "suppressed_timeout_rate", np.nan),
        "applied_noop_pass": bool(applied_noop_pass),
        "shadow_schedule_safe": bool(shadow_schedule_safe),
        "coverage_ok": bool(coverage_ok),
        "defensive_positive": bool(defensive_positive),
        "monitoring_only": True,
        "promotion_ready": False,
    }
    return {
        "summary": summary,
        "by_head": _head_rows(bundle_dir, replay_by_head, suppression),
        "suppression_utility": suppression.assign(bundle_dir=str(bundle_dir)) if not suppression.empty else pd.DataFrame(),
    }


def _shadow_promotion_failures(
    rollup: dict[str, Any],
    *,
    min_defensive_positive_bundle_share: float = DEFAULT_MIN_DEFENSIVE_POSITIVE_BUNDLE_SHARE,
    min_bundle_count: int = DEFAULT_MIN_BUNDLE_COUNT,
) -> list[str]:
    failures: list[str] = []
    bundle_count = int(rollup.get("bundle_count") or 0)
    if bundle_count < int(min_bundle_count):
        failures.append("insufficient_bundle_count")
    if rollup.get("artifact_audit_run") and rollup.get("all_artifact_audits_passed") is not True:
        failures.append("artifact_audit_failed")
    if int(rollup.get("applied_parity_failures") or 0) > 0:
        failures.append("applied_noop_parity_failed")
    if int(rollup.get("shadow_schedule_failures") or 0) > 0:
        failures.append("shadow_schedule_failed")
    if int(rollup.get("coverage_failures") or 0) > 0:
        failures.append("coverage_failed")
    if float(rollup.get("total_shadow_suppressed_candidates") or 0.0) <= 0.0:
        failures.append("no_shadow_suppression")
    if float(rollup.get("total_shadow_realized_defensive_success") or 0.0) <= 0.0:
        failures.append("defensive_success_not_positive")
    if (
        float(rollup.get("total_shadow_loss_avoided") or 0.0)
        <= float(rollup.get("total_shadow_winner_pnl_sacrificed") or 0.0) + EPS
    ):
        failures.append("loss_avoided_not_greater_than_winner_pnl_sacrificed")
    if float(rollup.get("defensive_positive_bundle_share") or 0.0) + EPS < float(
        min_defensive_positive_bundle_share
    ):
        failures.append("insufficient_defensive_positive_bundle_share")
    return failures


def aggregate_shadow_bundles(
    bundle_dirs: list[Path],
    *,
    run_artifact_audit: bool = True,
    min_defensive_positive_bundle_share: float = DEFAULT_MIN_DEFENSIVE_POSITIVE_BUNDLE_SHARE,
    min_bundle_count: int = DEFAULT_MIN_BUNDLE_COUNT,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    head_frames: list[pd.DataFrame] = []
    suppression_frames: list[pd.DataFrame] = []
    for bundle_dir in bundle_dirs:
        result = summarize_shadow_bundle(bundle_dir, run_artifact_audit=run_artifact_audit)
        records.append(result["summary"])
        if not result["by_head"].empty:
            head_frames.append(result["by_head"])
        if not result["suppression_utility"].empty:
            suppression_frames.append(result["suppression_utility"])
    bundles = pd.DataFrame(records)
    by_head = pd.concat(head_frames, ignore_index=True) if head_frames else pd.DataFrame()
    suppression = pd.concat(suppression_frames, ignore_index=True) if suppression_frames else pd.DataFrame()

    total_suppressed = float(pd.to_numeric(bundles.get("shadow_suppressed_candidates", 0.0), errors="coerce").fillna(0.0).sum())
    total_defensive = float(pd.to_numeric(bundles.get("shadow_realized_defensive_success", 0.0), errors="coerce").fillna(0.0).sum())
    total_loss_avoided = float(pd.to_numeric(bundles.get("shadow_loss_avoided", 0.0), errors="coerce").fillna(0.0).sum())
    total_winner_sacrificed = float(
        pd.to_numeric(bundles.get("shadow_winner_pnl_sacrificed", 0.0), errors="coerce").fillna(0.0).sum()
    )
    rollup = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_count": int(len(bundles)),
        "monitoring_only": True,
        "promotion_ready": False,
        "shadow_promotion_min_bundle_count": int(min_bundle_count),
        "shadow_promotion_min_defensive_positive_bundle_share": float(
            min_defensive_positive_bundle_share
        ),
        "artifact_audit_run": bool(run_artifact_audit),
        "all_artifact_audits_passed": (
            bool(bundles["artifact_audit_passed"].fillna(False).all())
            if run_artifact_audit and "artifact_audit_passed" in bundles
            else None
        ),
        "applied_parity_failures": int((~bundles.get("applied_noop_pass", pd.Series(dtype=bool)).astype(bool)).sum()),
        "shadow_schedule_failures": int((~bundles.get("shadow_schedule_safe", pd.Series(dtype=bool)).astype(bool)).sum()),
        "coverage_failures": int((~bundles.get("coverage_ok", pd.Series(dtype=bool)).astype(bool)).sum()),
        "defensive_positive_bundle_share": (
            float(bundles.get("defensive_positive", pd.Series(dtype=bool)).astype(bool).mean())
            if len(bundles) > 0
            else 0.0
        ),
        "total_replay_trades": float(pd.to_numeric(bundles.get("replay_trade_count", 0.0), errors="coerce").fillna(0.0).sum()),
        "total_replay_net_pnl": float(pd.to_numeric(bundles.get("replay_net_pnl", 0.0), errors="coerce").fillna(0.0).sum()),
        "total_shadow_suppressed_candidates": total_suppressed,
        "total_shadow_loss_avoided": total_loss_avoided,
        "total_shadow_winner_pnl_sacrificed": total_winner_sacrificed,
        "total_shadow_realized_defensive_success": total_defensive,
        "weighted_shadow_defensive_success_per_candidate": (
            float(total_defensive / total_suppressed) if total_suppressed > 0 else 0.0
        ),
    }
    failures = _shadow_promotion_failures(
        rollup,
        min_defensive_positive_bundle_share=float(min_defensive_positive_bundle_share),
        min_bundle_count=int(min_bundle_count),
    )
    rollup["shadow_promotion_gate_passed"] = not failures
    rollup["shadow_promotion_failures"] = failures
    rollup["controller_should_remain_disabled"] = True
    return {"rollup": rollup, "bundles": bundles, "by_head": by_head, "suppression_utility": suppression}


def _fmt(value: Any, decimals: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.{decimals}f}"
    return str(value)


def _markdown_report(result: dict[str, Any]) -> str:
    rollup = result["rollup"]
    bundles: pd.DataFrame = result["bundles"]
    by_head: pd.DataFrame = result["by_head"]
    lines = [
        "# Market-State Shadow Controller Monitor",
        "",
        "This report is monitoring-only. It does not promote or execute a controller.",
        "",
        "## Rollup",
        "",
        f"- Bundles: {rollup['bundle_count']}",
        f"- Artifact audits passed: {rollup['all_artifact_audits_passed']}",
        f"- Shadow promotion gate passed: {rollup['shadow_promotion_gate_passed']}",
        f"- Shadow promotion failures: {', '.join(rollup['shadow_promotion_failures']) if rollup['shadow_promotion_failures'] else 'none'}",
        f"- Applied no-op parity failures: {rollup['applied_parity_failures']}",
        f"- Shadow schedule failures: {rollup['shadow_schedule_failures']}",
        f"- Coverage failures: {rollup['coverage_failures']}",
        f"- Replay trades: {_fmt(rollup['total_replay_trades'], 0)}",
        f"- Replay net PnL: {_fmt(rollup['total_replay_net_pnl'])}",
        f"- Shadow suppressed candidates: {_fmt(rollup['total_shadow_suppressed_candidates'], 0)}",
        f"- Shadow loss avoided: {_fmt(rollup['total_shadow_loss_avoided'])}",
        f"- Shadow winner PnL sacrificed: {_fmt(rollup['total_shadow_winner_pnl_sacrificed'])}",
        f"- Shadow realized defensive success: {_fmt(rollup['total_shadow_realized_defensive_success'])}",
        f"- Weighted defensive success per candidate: {_fmt(rollup['weighted_shadow_defensive_success_per_candidate'])}",
        "",
        "## Bundle Summary",
        "",
    ]
    columns = [
        "selected_arm",
        "replay_trade_count",
        "replay_net_pnl",
        "applied_threshold_raises",
        "shadow_threshold_raises",
        "shadow_suppressed_candidates",
        "shadow_realized_defensive_success",
        "applied_noop_pass",
        "defensive_positive",
    ]
    if not bundles.empty:
        display = bundles[[c for c in columns if c in bundles.columns]].copy()
        display = display.replace({np.nan: ""})
        lines.append(_markdown_table(display))
    else:
        lines.append("_No bundles summarized._")
    lines.extend(["", "## By Head", ""])
    if not by_head.empty:
        head_columns = [
            "head",
            "trade_count",
            "win_rate",
            "net_pnl",
            "shadow_suppressed_candidates",
            "shadow_realized_defensive_success",
            "shadow_suppressed_full_sl_rate",
        ]
        display = by_head[[c for c in head_columns if c in by_head.columns]].copy()
        display = display.replace({np.nan: ""})
        lines.append(_markdown_table(display))
    else:
        lines.append("_No per-head rows available._")
    lines.append("")
    return "\n".join(lines)


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shadow_controller_monitor_summary.json").write_text(
        json.dumps(_json_safe(result["rollup"]), indent=2) + "\n",
        encoding="utf-8",
    )
    result["bundles"].to_csv(output_dir / "shadow_controller_monitor_bundles.csv", index=False)
    result["by_head"].to_csv(output_dir / "shadow_controller_monitor_by_head.csv", index=False)
    result["suppression_utility"].to_csv(
        output_dir / "shadow_controller_monitor_suppression_utility.csv",
        index=False,
    )
    (output_dir / "shadow_controller_monitor_report.md").write_text(
        _markdown_report(result),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-artifact-audit", action="store_true", default=False)
    parser.add_argument("--min-bundle-count", type=int, default=DEFAULT_MIN_BUNDLE_COUNT)
    parser.add_argument(
        "--min-defensive-positive-bundle-share",
        type=float,
        default=DEFAULT_MIN_DEFENSIVE_POSITIVE_BUNDLE_SHARE,
    )
    args = parser.parse_args()

    result = aggregate_shadow_bundles(
        [Path(p) for p in args.bundle_dirs],
        run_artifact_audit=not bool(args.skip_artifact_audit),
        min_defensive_positive_bundle_share=float(args.min_defensive_positive_bundle_share),
        min_bundle_count=int(args.min_bundle_count),
    )
    write_report(result, args.output_dir)
    print(json.dumps(_json_safe(result["rollup"]), indent=2))


if __name__ == "__main__":
    main()
