#!/usr/bin/env python3
"""Audit learned market-state head-priority transfer across replay windows."""

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

from scripts.audit_market_state_priority_shadow_promotion import (  # noqa: E402
    head_mix_metrics,
    opportunity_routing_gate,
    promotion_gate,
)


BASELINE_ARM = "P0_static_priority"


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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _num(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _parse_window(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        return label.strip(), Path(raw_path.strip())
    path = Path(value)
    return path.name, path


def _row_by_arm(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty or "arm" not in frame.columns:
        return {}
    rows = frame.loc[frame["arm"].astype(str).eq(str(arm))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _candidate_arm(summary: pd.DataFrame, arm_contains: str | None) -> str:
    if summary.empty or "arm" not in summary.columns:
        raise ValueError("summary is missing arm rows")
    arms = [
        str(arm)
        for arm in summary["arm"].dropna().astype(str).unique()
        if str(arm) != BASELINE_ARM
    ]
    if not arms:
        raise ValueError("summary has no candidate arm")
    if arm_contains:
        matched = [arm for arm in arms if str(arm_contains) in arm]
        if matched:
            return matched[0]
    return arms[0]


def _all_scope_swap(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    work = frame.loc[frame.get("arm", pd.Series(dtype=str)).astype(str).eq(str(arm))]
    if "scope" in work.columns:
        all_scope = work.loc[work["scope"].astype(str).eq("all")]
        if not all_scope.empty:
            work = all_scope
    return work.iloc[0].to_dict() if not work.empty else {}


def _schedule_coverage(priority_dir: Path) -> float:
    path = priority_dir / "head_priority_learned_schedule.parquet"
    if not path.exists():
        return np.nan
    schedule = pd.read_parquet(path, columns=None)
    if schedule.empty:
        return np.nan
    if "coverage" in schedule.columns:
        return float(pd.to_numeric(schedule["coverage"], errors="coerce").min())
    if {"timestamp", "head"}.issubset(schedule.columns):
        dupes = int(schedule.duplicated(["timestamp", "head"]).sum())
        return 0.0 if dupes else 1.0
    return np.nan


def _window_bounds(manifest: dict[str, Any]) -> dict[str, Any]:
    universe = manifest.get("candidate_universe") or {}
    return {
        "candidate_rows": int(universe.get("rows", 0) or 0),
        "timestamp_count": int(universe.get("timestamp_count", 0) or 0),
        "window_start": universe.get("timestamp_min"),
        "window_end": universe.get("timestamp_max"),
    }


def load_window(priority_dir: Path, *, label: str, arm_contains: str | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = _read_json(priority_dir / "manifest.json")
    summary = _read_csv(priority_dir / "head_priority_learning_replay_summary.csv")
    by_head = _read_csv(priority_dir / "head_priority_learning_by_head.csv")
    overlap = _read_csv(priority_dir / "head_priority_learning_accepted_overlap.csv")
    swap = _read_csv(priority_dir / "head_priority_learning_accepted_swap_utility.csv")
    diagnostics = _read_csv(priority_dir / "head_priority_learning_model_diagnostics.csv")
    if summary.empty:
        raise FileNotFoundError(priority_dir / "head_priority_learning_replay_summary.csv")
    arm = _candidate_arm(summary, arm_contains)
    base = _row_by_arm(summary, BASELINE_ARM)
    cand = _row_by_arm(summary, arm)
    ov = _row_by_arm(overlap, arm)
    sw = _all_scope_swap(swap, arm)
    diag = _row_by_arm(diagnostics, arm) if "arm" in diagnostics.columns else (
        diagnostics.iloc[0].to_dict() if not diagnostics.empty else {}
    )
    row = {
        "window_label": label,
        "source_dir": str(priority_dir),
        "arm": arm,
        **_window_bounds(manifest),
        "coverage": _schedule_coverage(priority_dir),
        "baseline_trade_count": int(_num(base.get("trade_count"), 0.0)),
        "trade_count": int(_num(cand.get("trade_count"), 0.0)),
        "trade_count_delta": int(_num(cand.get("trade_count"), 0.0) - _num(base.get("trade_count"), 0.0)),
        "baseline_net_pnl": _num(base.get("net_pnl"), 0.0),
        "net_pnl": _num(cand.get("net_pnl"), 0.0),
        "delta_net_pnl": _num(cand.get("net_pnl"), 0.0) - _num(base.get("net_pnl"), 0.0),
        "baseline_full_sl_rate": _num(base.get("full_sl_rate")),
        "full_sl_rate": _num(cand.get("full_sl_rate")),
        "delta_full_sl_rate": _num(cand.get("full_sl_rate")) - _num(base.get("full_sl_rate")),
        "baseline_timeout_rate": _num(base.get("timeout_rate")),
        "timeout_rate": _num(cand.get("timeout_rate")),
        "delta_timeout_rate": _num(cand.get("timeout_rate")) - _num(base.get("timeout_rate")),
        "accepted_jaccard": _num(ov.get("jaccard_vs_baseline"), np.nan),
        "entrants": int(_num(sw.get("entrants"), 0.0)),
        "removed": int(_num(sw.get("removed"), 0.0)),
        "entrant_net_pnl": _num(sw.get("entrant_net_pnl"), 0.0),
        "removed_net_pnl": _num(sw.get("removed_net_pnl"), 0.0),
        "net_replacement_pnl": _num(sw.get("net_replacement_pnl"), 0.0),
        "net_action_pnl_delta": _num(sw.get("net_action_pnl_delta"), 0.0),
        "defensive_success": _num(sw.get("defensive_success"), 0.0),
        "selection_gate_passed": bool(diag.get("selection_gate_passed", False)),
        "selection_objective": _num(diag.get("selection_objective"), np.nan),
        "config_max_adjustment": _num(diag.get("config_max_adjustment"), np.nan),
    }

    by_head_rows: list[dict[str, Any]] = []
    if not by_head.empty and "head" in by_head.columns:
        base_head = by_head.loc[by_head["arm"].astype(str).eq(BASELINE_ARM)].set_index("head")
        cand_head = by_head.loc[by_head["arm"].astype(str).eq(arm)].set_index("head")
        for head in sorted(set(base_head.index.astype(str)) | set(cand_head.index.astype(str))):
            rec: dict[str, Any] = {"window_label": label, "arm": arm, "head": head}
            for metric in ("trade_count", "win_rate", "net_pnl", "full_sl_rate", "timeout_rate"):
                b = _num(base_head[metric].get(head, 0.0) if metric in base_head.columns and head in base_head.index else 0.0)
                c = _num(cand_head[metric].get(head, 0.0) if metric in cand_head.columns and head in cand_head.index else 0.0)
                rec[f"baseline_{metric}"] = b
                rec[metric] = c
                rec[f"delta_{metric}"] = c - b
            by_head_rows.append(rec)
    return row, pd.DataFrame(by_head_rows)


def render_report(
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    *,
    defensive_gate: dict[str, Any],
    opportunity_gate: dict[str, Any],
) -> str:
    lines = [
        "# Learned Head-Priority Window Transfer Audit",
        "",
        "This aggregates the learned market-state head-priority shadow runs across replay windows.",
        "The audited action is only `portfolio_priority_adjustment`; scores, ranks, thresholds, sizing, q-fail, HeadHealth and threshold control remain unchanged.",
        "",
        "## Window Summary",
        "",
    ]
    cols = [
        "window_label",
        "timestamp_count",
        "baseline_trade_count",
        "trade_count",
        "delta_net_pnl",
        "delta_full_sl_rate",
        "delta_timeout_rate",
        "accepted_jaccard",
        "baseline_active_head_count",
        "shadow_active_head_count",
        "shadow_dominant_head_share",
        "head_trade_share_l1_delta",
        "entrants",
        "removed",
        "net_replacement_pnl",
        "defensive_success",
        "selection_gate_passed",
    ]
    lines.append(summary[[c for c in cols if c in summary.columns]].to_markdown(index=False))
    lines.extend(["", "## Opportunity-Routing Gate", ""])
    lines.append(f"- Passed: `{bool(opportunity_gate.get('passed'))}`")
    lines.append(f"- Failures: `{', '.join(opportunity_gate.get('failures') or []) or 'none'}`")
    lines.append(
        f"- Median delta net PnL: `{float(opportunity_gate.get('median_delta_net_pnl', np.nan)):.6f}`"
    )
    lines.append(
        f"- Q25 delta net PnL: `{float(opportunity_gate.get('q25_delta_net_pnl', np.nan)):.6f}`"
    )
    lines.append(
        f"- Positive delta window share: `{float(opportunity_gate.get('positive_delta_window_share', np.nan)):.2%}`"
    )
    lines.append(f"- Action windows: `{int(opportunity_gate.get('action_window_count', 0))}`")
    lines.extend(["", "## Defensive-Suppression Gate", ""])
    lines.append(f"- Passed: `{bool(defensive_gate.get('passed'))}`")
    lines.append(f"- Failures: `{', '.join(defensive_gate.get('failures') or []) or 'none'}`")
    if not by_head.empty:
        lines.extend(["", "## By Head", ""])
        cols = [
            "window_label",
            "head",
            "delta_trade_count",
            "delta_net_pnl",
            "delta_full_sl_rate",
            "delta_timeout_rate",
        ]
        lines.append(by_head[[c for c in cols if c in by_head.columns]].to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="append", required=True, help="Window label/path as label=dir.")
    parser.add_argument("--arm-contains", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    by_head_frames: list[pd.DataFrame] = []
    for item in args.window:
        label, path = _parse_window(str(item))
        row, by_head = load_window(path, label=label, arm_contains=args.arm_contains)
        rows.append(row)
        if not by_head.empty:
            by_head_frames.append(by_head)
    summary = pd.DataFrame(rows)
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    mix = head_mix_metrics(by_head)
    if not mix.empty:
        summary = summary.merge(mix, on="window_label", how="left", validate="one_to_one")
    defensive_gate = promotion_gate(summary)
    opportunity_gate = opportunity_routing_gate(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "head_priority_window_transfer_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "head_priority_window_transfer_by_head.csv", index=False)
    payload = {
        "generated_by": "audit_market_state_head_priority_window_transfer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm_contains": args.arm_contains,
        "promotion_gate": defensive_gate,
        "defensive_suppression_gate": defensive_gate,
        "opportunity_routing_gate": opportunity_gate,
        "opportunity_routing_passed": bool(opportunity_gate.get("passed")),
        "opportunity_should_remain_shadow": not bool(opportunity_gate.get("passed")),
        "summary": summary.to_dict("records"),
        "by_head": by_head.to_dict("records"),
    }
    (args.output_dir / "head_priority_window_transfer_audit.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "head_priority_window_transfer_audit.md").write_text(
        render_report(
            summary,
            by_head,
            defensive_gate=defensive_gate,
            opportunity_gate=opportunity_gate,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(args.output_dir),
                    "opportunity_routing_gate": opportunity_gate,
                    "defensive_suppression_gate": defensive_gate,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
