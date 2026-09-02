#!/usr/bin/env python3
"""Build a shadow market-state rank-reference router schedule.

The router is a formal schedule artifact for the rank-modulation research path.
It converts a learned per-head market-state priority schedule into a
timestamp-level short_boll rank-reference decision:

* weight 0.0 -> use causal global-over-time rank reference;
* weight 1.0 -> use repaired within-timestamp T1 rank reference;
* values in between -> continuous rank-column blend.

This script does not replay trades and does not alter the active stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PRIORITY_SCHEDULE = Path(
    "data_perp/reports/market_state_head_priority_learning_actionaware_20260626_jun15_22"
    "/head_priority_learned_schedule.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_rank_reference_router_20260626_v1")


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


def _sigmoid(x: pd.Series) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce").clip(lower=-30.0, upper=30.0)
    return 1.0 / (1.0 + np.exp(-z))


def build_rank_reference_router_schedule(
    priority_schedule: pd.DataFrame,
    *,
    target_head: str = "short_boll",
    reference_head: str = "short_asset",
    margin: float = 0.0,
    blend_scale: float = 0.05,
    min_timestamp_weight: float = 0.0,
    max_timestamp_weight: float = 1.0,
    timestamp_decision_threshold: float = 0.5,
    fail_closed_reference: str = "timestamp_rank",
) -> pd.DataFrame:
    """Create one router row per timestamp from a per-head priority schedule."""

    if fail_closed_reference not in {"timestamp_rank", "global_rank"}:
        raise ValueError("fail_closed_reference must be timestamp_rank or global_rank")
    if not 0.0 <= float(min_timestamp_weight) <= float(max_timestamp_weight) <= 1.0:
        raise ValueError("timestamp weight bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= float(timestamp_decision_threshold) <= 1.0:
        raise ValueError("timestamp_decision_threshold must be in [0, 1]")
    schedule = priority_schedule.copy()
    required = {"timestamp", "head", "portfolio_priority_adjustment"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError(f"priority schedule missing columns: {missing}")
    schedule["timestamp"] = pd.to_datetime(schedule["timestamp"], utc=True, errors="coerce")
    if schedule["timestamp"].isna().any():
        raise ValueError("priority schedule contains nonparseable timestamps")
    if schedule.duplicated(["timestamp", "head"]).any():
        raise ValueError("priority schedule has duplicate timestamp/head rows")
    priority = schedule.pivot(
        index="timestamp",
        columns="head",
        values="portfolio_priority_adjustment",
    )
    missing_heads = [head for head in (target_head, reference_head) if head not in priority.columns]
    if missing_heads:
        raise ValueError(f"priority schedule missing heads: {missing_heads}")

    target = pd.to_numeric(priority[target_head], errors="coerce")
    reference = pd.to_numeric(priority[reference_head], errors="coerce")
    diff = target - reference
    finite = target.replace([np.inf, -np.inf], np.nan).notna() & reference.replace(
        [np.inf, -np.inf],
        np.nan,
    ).notna()
    if float(blend_scale) <= 1e-12:
        raw_weight = pd.Series(np.where(diff > float(margin), 1.0, 0.0), index=priority.index)
    else:
        raw_weight = _sigmoid((diff - float(margin)) / float(blend_scale))
    clipped_weight = raw_weight.clip(
        lower=float(min_timestamp_weight),
        upper=float(max_timestamp_weight),
    )
    fallback_weight = 1.0 if fail_closed_reference == "timestamp_rank" else 0.0
    final_weight = clipped_weight.where(finite, fallback_weight).astype(float)
    scope = np.where(final_weight >= float(timestamp_decision_threshold), "timestamp_rank", "global_rank")
    out = pd.DataFrame(
        {
            "timestamp": priority.index,
            f"{target_head}_priority": target.to_numpy(dtype=float),
            f"{reference_head}_priority": reference.to_numpy(dtype=float),
            f"{target_head}_minus_{reference_head}_priority": diff.to_numpy(dtype=float),
            f"{target_head}_timestamp_weight_raw": raw_weight.to_numpy(dtype=float),
            f"{target_head}_timestamp_weight": final_weight.to_numpy(dtype=float),
            f"{target_head}_rank_scope": scope,
            "router_valid": finite.to_numpy(dtype=bool),
            "router_fallback_reference": np.where(finite.to_numpy(dtype=bool), "", fail_closed_reference),
            "router_mode": "state_conditioned_rank_reference_blend",
            "router_layer": "rank_reference_before_threshold",
            "target_head": str(target_head),
            "reference_head": str(reference_head),
            "changes_thresholds": False,
            "changes_scores": False,
            "changes_active_stack": False,
            "promotion_status": "shadow_only",
        }
    )
    out = out.rename(
        columns={
            f"{target_head}_timestamp_weight": "short_boll_timestamp_weight",
            f"{target_head}_rank_scope": "short_boll_rank_scope",
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.sort_values("timestamp").reset_index(drop=True)


def _summary(schedule: pd.DataFrame) -> dict[str, Any]:
    weight = pd.to_numeric(schedule["short_boll_timestamp_weight"], errors="coerce")
    return {
        "row_count": int(len(schedule)),
        "timestamp_min": schedule["timestamp"].min(),
        "timestamp_max": schedule["timestamp"].max(),
        "timestamp_weight_mean": float(weight.mean()),
        "timestamp_weight_min": float(weight.min()),
        "timestamp_weight_max": float(weight.max()),
        "timestamp_weight_p25": float(weight.quantile(0.25)),
        "timestamp_weight_p50": float(weight.quantile(0.50)),
        "timestamp_weight_p75": float(weight.quantile(0.75)),
        "timestamp_rank_share": float(schedule["short_boll_rank_scope"].astype(str).eq("timestamp_rank").mean()),
        "global_rank_share": float(schedule["short_boll_rank_scope"].astype(str).eq("global_rank").mean()),
        "valid_share": float(schedule["router_valid"].astype(bool).mean()),
    }


def render_report(manifest: dict[str, Any], schedule: pd.DataFrame) -> str:
    summary = manifest["summary"]
    scope_counts = (
        schedule.groupby("short_boll_rank_scope", observed=True)
        .size()
        .reset_index(name="timestamp_count")
    )
    scope_counts["timestamp_share"] = scope_counts["timestamp_count"] / max(len(schedule), 1)
    lines = [
        "# Market-State Rank-Reference Router",
        "",
        "This artifact is shadow-only. It converts learned market-state head priority into a short_boll rank-reference weight before thresholding.",
        "",
        "## Contract",
        "",
        "- Active stack changed: `false`",
        "- Scores changed: `false`",
        "- Thresholds changed: `false`",
        "- Rank reference changed in production: `false`",
        "- Intended replay consumer: `run_market_state_short_boll_rank_scope_switch.py --router-schedule`",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary['row_count']}`",
        f"- Timestamp range: `{summary['timestamp_min']}` to `{summary['timestamp_max']}`",
        f"- Mean timestamp-rank weight: `{summary['timestamp_weight_mean']:.6f}`",
        f"- Timestamp-rank share: `{summary['timestamp_rank_share']:.6f}`",
        f"- Valid router share: `{summary['valid_share']:.6f}`",
        "",
        "## Scope Counts",
        "",
        scope_counts.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Weight near 1.0 means the market-state learner favors short_boll timestamp ranking; weight near 0.0 means it favors global-over-time ranking. This schedule must be evaluated through paired shadow replay and later-window promotion gates before it can influence live ranking.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-schedule", type=Path, default=DEFAULT_PRIORITY_SCHEDULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-head", default="short_boll")
    parser.add_argument("--reference-head", default="short_asset")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--blend-scale", type=float, default=0.05)
    parser.add_argument("--min-timestamp-weight", type=float, default=0.0)
    parser.add_argument("--max-timestamp-weight", type=float, default=1.0)
    parser.add_argument("--timestamp-decision-threshold", type=float, default=0.5)
    parser.add_argument("--fail-closed-reference", choices=["timestamp_rank", "global_rank"], default="timestamp_rank")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    priority = pd.read_parquet(args.priority_schedule)
    router = build_rank_reference_router_schedule(
        priority,
        target_head=str(args.target_head),
        reference_head=str(args.reference_head),
        margin=float(args.margin),
        blend_scale=float(args.blend_scale),
        min_timestamp_weight=float(args.min_timestamp_weight),
        max_timestamp_weight=float(args.max_timestamp_weight),
        timestamp_decision_threshold=float(args.timestamp_decision_threshold),
        fail_closed_reference=str(args.fail_closed_reference),
    )
    summary = _summary(router)
    manifest = {
        "generated_by": "build_market_state_rank_reference_router",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "shadow_market_state_rank_reference_router_schedule",
        "contract": {
            "changes_active_stack": False,
            "changes_scores": False,
            "changes_thresholds": False,
            "changes_rank_reference_in_production": False,
            "router_layer": "rank_reference_before_threshold",
            "qfail_active": False,
            "head_health_active": False,
            "production_eligible": False,
            "promotion_status": "shadow_only",
        },
        "params": {
            "target_head": str(args.target_head),
            "reference_head": str(args.reference_head),
            "margin": float(args.margin),
            "blend_scale": float(args.blend_scale),
            "min_timestamp_weight": float(args.min_timestamp_weight),
            "max_timestamp_weight": float(args.max_timestamp_weight),
            "timestamp_decision_threshold": float(args.timestamp_decision_threshold),
            "fail_closed_reference": str(args.fail_closed_reference),
        },
        "inputs": {
            "priority_schedule": str(args.priority_schedule),
            "priority_schedule_sha256": _sha256(args.priority_schedule),
        },
        "summary": summary,
        "outputs": {
            "manifest": str(args.output_dir / "rank_reference_router_manifest.json"),
            "report": str(args.output_dir / "rank_reference_router_report.md"),
            "schedule": str(args.output_dir / "rank_reference_router_schedule.parquet"),
            "schedule_csv": str(args.output_dir / "rank_reference_router_schedule.csv"),
        },
    }
    report = render_report(manifest, router)
    router.to_parquet(args.output_dir / "rank_reference_router_schedule.parquet", index=False)
    router.to_csv(args.output_dir / "rank_reference_router_schedule.csv", index=False)
    (args.output_dir / "rank_reference_router_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "rank_reference_router_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "summary": summary}), indent=2))


if __name__ == "__main__":
    main()
