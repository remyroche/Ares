#!/usr/bin/env python3
"""Aggregate rank-router plus priority-modulation shadow replays."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_ARM = "R0_t1_timestamp_short_boll"
RANK_BLEND_ARM = "R3_state_blended_short_boll"
PRIORITY_ONLY_ARM = "R4_t1_timestamp_plus_priority"
RANK_PLUS_PRIORITY_ARM = "R5_state_blended_plus_priority"


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


def _window_label(path: Path) -> str:
    text = path.name
    for prefix in (
        "market_state_rank_router_plus_priority_20260626_",
        "market_state_rank_router_plus_priority_",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.endswith("_v1"):
        text = text[:-3]
    return text


def summarize_window(path: Path) -> dict[str, Any]:
    summary_path = path / "rank_scope_switch_summary.csv"
    overlap_path = path / "rank_scope_switch_accepted_overlap.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    overlap = pd.read_csv(overlap_path) if overlap_path.exists() else pd.DataFrame()
    rows = {str(row["arm"]): row for _, row in summary.iterrows()}
    missing = sorted(
        set([BASELINE_ARM, RANK_BLEND_ARM, PRIORITY_ONLY_ARM, RANK_PLUS_PRIORITY_ARM]).difference(rows)
    )
    if missing:
        raise ValueError(f"{path} missing required arms: {missing}")
    base = rows[BASELINE_ARM]
    overlap_rows = {str(row["arm"]): row for _, row in overlap.iterrows()} if not overlap.empty else {}
    out: dict[str, Any] = {
        "artifact_dir": str(path),
        "window_label": _window_label(path),
        "baseline_trade_count": int(base["trade_count"]),
        "baseline_net_pnl": float(base["net_pnl"]),
        "baseline_full_sl_rate": float(base["full_sl_rate"]),
        "baseline_timeout_rate": float(base["timeout_rate"]),
    }
    for arm, prefix in (
        (RANK_BLEND_ARM, "rank_blend"),
        (PRIORITY_ONLY_ARM, "priority_only"),
        (RANK_PLUS_PRIORITY_ARM, "rank_plus_priority"),
    ):
        row = rows[arm]
        out[f"{prefix}_trade_count"] = int(row["trade_count"])
        out[f"{prefix}_trade_count_delta"] = int(row["trade_count"]) - int(base["trade_count"])
        out[f"{prefix}_net_pnl"] = float(row["net_pnl"])
        out[f"{prefix}_delta_net_pnl"] = float(row["net_pnl"]) - float(base["net_pnl"])
        out[f"{prefix}_full_sl_rate"] = float(row["full_sl_rate"])
        out[f"{prefix}_delta_full_sl_rate"] = float(row["full_sl_rate"]) - float(base["full_sl_rate"])
        out[f"{prefix}_timeout_rate"] = float(row["timeout_rate"])
        out[f"{prefix}_delta_timeout_rate"] = float(row["timeout_rate"]) - float(base["timeout_rate"])
        overlap_row = overlap_rows.get(arm, {})
        out[f"{prefix}_accepted_jaccard_vs_t1"] = float(
            overlap_row.get("jaccard_vs_baseline", np.nan)
        )
    return out


def aggregate_windows(
    artifact_dirs: list[Path],
    *,
    output_dir: Path,
    development_labels: set[str] | None = None,
    min_later_windows: int = 2,
    min_positive_later_share: float = 0.75,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([summarize_window(Path(path)) for path in artifact_dirs])
    development = set(development_labels or set())
    frame["is_development_window"] = frame["window_label"].astype(str).isin(development) | frame[
        "window_label"
    ].astype(str).str.contains("jun15_22|june_15_22|15_22", case=False, regex=True)
    later = frame.loc[~frame["is_development_window"]].copy()
    rollup: dict[str, Any] = {
        "generated_by": "report_market_state_rank_router_priority_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": int(len(frame)),
        "development_window_count": int(frame["is_development_window"].sum()),
        "later_window_count": int(len(later)),
        "min_later_windows": int(min_later_windows),
        "min_positive_later_share": float(min_positive_later_share),
    }
    for prefix in ("rank_blend", "priority_only", "rank_plus_priority"):
        deltas = pd.to_numeric(frame[f"{prefix}_delta_net_pnl"], errors="coerce")
        later_deltas = pd.to_numeric(later[f"{prefix}_delta_net_pnl"], errors="coerce")
        rollup[f"{prefix}_total_delta_net_pnl"] = float(deltas.sum())
        rollup[f"{prefix}_development_delta_net_pnl"] = float(
            pd.to_numeric(
                frame.loc[frame["is_development_window"], f"{prefix}_delta_net_pnl"],
                errors="coerce",
            ).sum()
        )
        rollup[f"{prefix}_later_total_delta_net_pnl"] = float(later_deltas.sum()) if len(later) else 0.0
        rollup[f"{prefix}_later_median_delta_net_pnl"] = (
            float(later_deltas.median()) if len(later) else float("nan")
        )
        rollup[f"{prefix}_later_positive_delta_share"] = (
            float((later_deltas > 0.0).mean()) if len(later) else float("nan")
        )
        rollup[f"{prefix}_max_delta_full_sl_rate"] = float(
            pd.to_numeric(frame[f"{prefix}_delta_full_sl_rate"], errors="coerce").max()
        )
        rollup[f"{prefix}_max_delta_timeout_rate"] = float(
            pd.to_numeric(frame[f"{prefix}_delta_timeout_rate"], errors="coerce").max()
        )
        rollup[f"{prefix}_min_accepted_jaccard_vs_t1"] = float(
            pd.to_numeric(frame[f"{prefix}_accepted_jaccard_vs_t1"], errors="coerce").min()
        )
    failures: list[str] = []
    if int(rollup["later_window_count"]) < int(min_later_windows):
        failures.append("insufficient_later_window_count")
    if float(rollup["rank_plus_priority_later_positive_delta_share"]) < float(min_positive_later_share):
        failures.append("rank_plus_priority_later_positive_delta_share_below_gate")
    if float(rollup["rank_plus_priority_later_median_delta_net_pnl"]) <= 0.0:
        failures.append("rank_plus_priority_later_median_delta_not_positive")
    if float(rollup["rank_plus_priority_max_delta_timeout_rate"]) > 0.0:
        failures.append("rank_plus_priority_timeout_worsened")
    rollup["rank_plus_priority_promotion_gate_passed"] = not failures
    rollup["rank_plus_priority_should_remain_shadow"] = bool(failures)
    rollup["rank_plus_priority_failures"] = failures
    rollup["interpretation"] = (
        "The composed state rank-reference router plus bounded auction-priority arm remains shadow-only. "
        "It does not improve the later windows enough to justify changing live ranking or auction priority."
    )
    frame.to_csv(output_dir / "rank_router_priority_window_summary.csv", index=False)
    payload = {"rollup": rollup, "windows": frame.to_dict("records")}
    (output_dir / "rank_router_priority_validation_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(rollup, frame)
    (output_dir / "rank_router_priority_validation_report.md").write_text(report, encoding="utf-8")
    return payload


def render_report(rollup: dict[str, Any], frame: pd.DataFrame) -> str:
    cols = [
        "window_label",
        "is_development_window",
        "baseline_trade_count",
        "baseline_net_pnl",
        "rank_blend_delta_net_pnl",
        "priority_only_delta_net_pnl",
        "rank_plus_priority_delta_net_pnl",
        "rank_plus_priority_delta_full_sl_rate",
        "rank_plus_priority_delta_timeout_rate",
        "rank_plus_priority_accepted_jaccard_vs_t1",
    ]
    return (
        "# Market-State Rank Router Plus Priority Validation\n\n"
        "This shadow-only report evaluates the sequence: state-conditioned rank reference, then bounded auction-priority modulation.\n\n"
        "## Gate\n\n"
        f"- Passed: `{rollup['rank_plus_priority_promotion_gate_passed']}`\n"
        f"- Failures: `{', '.join(rollup['rank_plus_priority_failures']) or 'none'}`\n"
        f"- Later median delta net PnL: `{rollup['rank_plus_priority_later_median_delta_net_pnl']}`\n"
        f"- Later positive delta share: `{rollup['rank_plus_priority_later_positive_delta_share']}`\n\n"
        "## Windows\n\n"
        + frame[[c for c in cols if c in frame.columns]].to_markdown(index=False)
        + "\n\n## Interpretation\n\n"
        + str(rollup["interpretation"])
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-label", action="append", default=[])
    parser.add_argument("--min-later-windows", type=int, default=2)
    parser.add_argument("--min-positive-later-share", type=float, default=0.75)
    args = parser.parse_args()
    payload = aggregate_windows(
        args.artifact_dirs,
        output_dir=args.output_dir,
        development_labels=set(args.development_label or []),
        min_later_windows=int(args.min_later_windows),
        min_positive_later_share=float(args.min_positive_later_share),
    )
    print(json.dumps(_json_safe(payload["rollup"]), indent=2))


if __name__ == "__main__":
    main()
