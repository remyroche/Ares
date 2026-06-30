#!/usr/bin/env python3
"""Aggregate shadow short_boll rank-scope blend validation windows.

The report is intentionally conservative: rank-scope modulation changes the
portfolio auction population, so it remains shadow-only unless later windows
show recurrent benefit against the exact T1 timestamp-rank baseline.
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


BASELINE_ARM = "R0_t1_timestamp_short_boll"
GLOBAL_ARM = "R1_global_short_boll"
SWITCH_ARM = "R2_state_switch_short_boll"
BLEND_ARM = "R3_state_blended_short_boll"
ARMS = (BASELINE_ARM, GLOBAL_ARM, SWITCH_ARM, BLEND_ARM)


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
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _window_label(path: Path, manifest: dict[str, Any]) -> str:
    period = manifest.get("params", {}).get("evaluation_period")
    if period:
        return str(period)
    name = path.name
    for prefix in (
        "market_state_short_boll_rank_scope_switch_20260626_",
        "market_state_short_boll_rank_scope_switch_",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    suffixes = ("_state_blend_v1", "_broad_parity_v1")
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def summarize_rank_scope_window(path: Path) -> dict[str, Any]:
    root = Path(path)
    summary = _read_required_csv(root / "rank_scope_switch_summary.csv")
    overlap = _read_required_csv(root / "rank_scope_switch_accepted_overlap.csv")
    schedule = _read_required_csv(root / "rank_scope_switch_schedule.csv")
    manifest = _load_json(root / "rank_scope_switch_manifest.json")
    missing = sorted(set(ARMS).difference(set(summary["arm"].astype(str))))
    if missing:
        raise ValueError(f"{root} missing required arms: {missing}")
    rows = {str(row["arm"]): row for _, row in summary.iterrows()}
    base = rows[BASELINE_ARM]
    out: dict[str, Any] = {
        "artifact_dir": str(root),
        "window_label": _window_label(root, manifest),
        "baseline_trade_count": int(base["trade_count"]),
        "baseline_net_pnl": float(base["net_pnl"]),
        "baseline_full_sl_rate": float(base["full_sl_rate"]),
        "baseline_timeout_rate": float(base["timeout_rate"]),
        "baseline_worst_24h_net_pnl": float(base["worst_24h_net_pnl"]),
        "rank_contract_candidate_parity_passed": bool(
            manifest.get("rank_contract_candidate_parity", {}).get("passed", False)
        ),
        "production_eligible": bool(manifest.get("contract", {}).get("production_eligible", False)),
        "promotion_status": manifest.get("contract", {}).get("promotion_status"),
    }
    overlap_rows = {
        str(row["arm"]): row for _, row in overlap.iterrows()
    } if not overlap.empty else {}
    for arm, prefix in (
        (GLOBAL_ARM, "global"),
        (SWITCH_ARM, "switch"),
        (BLEND_ARM, "blend"),
    ):
        row = rows[arm]
        out[f"{prefix}_trade_count"] = int(row["trade_count"])
        out[f"{prefix}_net_pnl"] = float(row["net_pnl"])
        out[f"{prefix}_delta_net_pnl"] = float(row["net_pnl"]) - float(base["net_pnl"])
        out[f"{prefix}_full_sl_rate"] = float(row["full_sl_rate"])
        out[f"{prefix}_delta_full_sl_rate"] = float(row["full_sl_rate"]) - float(base["full_sl_rate"])
        out[f"{prefix}_timeout_rate"] = float(row["timeout_rate"])
        out[f"{prefix}_delta_timeout_rate"] = float(row["timeout_rate"]) - float(base["timeout_rate"])
        out[f"{prefix}_worst_24h_net_pnl"] = float(row["worst_24h_net_pnl"])
        out[f"{prefix}_delta_worst_24h_net_pnl"] = float(row["worst_24h_net_pnl"]) - float(base["worst_24h_net_pnl"])
        out[f"{prefix}_trade_count_delta"] = int(row["trade_count"]) - int(base["trade_count"])
        overlap_row = overlap_rows.get(arm, {})
        out[f"{prefix}_accepted_jaccard_vs_t1"] = float(overlap_row.get("jaccard_vs_baseline", np.nan))
        out[f"{prefix}_entrants_vs_t1"] = int(overlap_row.get("arm_only", 0) or 0)
        out[f"{prefix}_removed_vs_t1"] = int(overlap_row.get("baseline_only", 0) or 0)

    if "short_boll_timestamp_weight" in schedule.columns:
        blend_schedule = schedule.loc[schedule["arm"].astype(str).eq(BLEND_ARM)].copy()
        weights = pd.to_numeric(blend_schedule["short_boll_timestamp_weight"], errors="coerce")
        out["blend_timestamp_weight_mean"] = float(weights.mean())
        out["blend_timestamp_weight_min"] = float(weights.min())
        out["blend_timestamp_weight_max"] = float(weights.max())
    else:
        out["blend_timestamp_weight_mean"] = float("nan")
        out["blend_timestamp_weight_min"] = float("nan")
        out["blend_timestamp_weight_max"] = float("nan")
    return out


def _promotion_failures(
    rollup: dict[str, Any],
    *,
    min_later_windows: int,
    min_positive_later_share: float,
    min_accepted_jaccard: float,
    max_full_sl_delta: float,
    max_timeout_delta: float,
) -> list[str]:
    failures: list[str] = []
    if int(rollup["later_window_count"]) < int(min_later_windows):
        failures.append("insufficient_later_window_count")
    if not bool(rollup["all_candidate_parity_passed"]):
        failures.append("rank_contract_candidate_parity_failed")
    if float(rollup["later_blend_positive_delta_share"]) < float(min_positive_later_share):
        failures.append("later_positive_delta_share_below_gate")
    if float(rollup["later_blend_median_delta_net_pnl"]) <= 0.0:
        failures.append("later_median_delta_not_positive")
    if float(rollup["later_blend_q25_delta_net_pnl"]) < 0.0:
        failures.append("later_q25_delta_negative")
    if float(rollup["max_blend_delta_full_sl_rate"]) > float(max_full_sl_delta):
        failures.append("full_sl_worsened_beyond_gate")
    if float(rollup["max_blend_delta_timeout_rate"]) > float(max_timeout_delta):
        failures.append("timeout_worsened_beyond_gate")
    if float(rollup["min_blend_accepted_jaccard_vs_t1"]) < float(min_accepted_jaccard):
        failures.append("accepted_jaccard_below_gate")
    return failures


def aggregate_rank_scope_windows(
    artifact_dirs: list[Path],
    *,
    output_dir: Path,
    development_labels: set[str] | None = None,
    min_later_windows: int = 2,
    min_positive_later_share: float = 0.75,
    min_accepted_jaccard: float = 0.75,
    max_full_sl_delta: float = 0.0,
    max_timeout_delta: float = 0.0,
) -> dict[str, Any]:
    if not artifact_dirs:
        raise ValueError("at least one rank-scope artifact directory is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_rank_scope_window(Path(path)) for path in artifact_dirs]
    frame = pd.DataFrame(rows)
    development = set(development_labels or set())
    frame["is_development_window"] = frame["window_label"].astype(str).isin(development) | frame[
        "window_label"
    ].astype(str).str.contains("jun15_22|june_15_22|15_22", case=False, regex=True)
    later = frame.loc[~frame["is_development_window"]].copy()
    blend_delta = pd.to_numeric(frame["blend_delta_net_pnl"], errors="coerce")
    later_delta = pd.to_numeric(later["blend_delta_net_pnl"], errors="coerce")
    rollup = {
        "generated_by": "report_market_state_rank_scope_blend_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": int(len(frame)),
        "development_window_count": int(frame["is_development_window"].sum()),
        "later_window_count": int(len(later)),
        "all_candidate_parity_passed": bool(frame["rank_contract_candidate_parity_passed"].fillna(False).all()),
        "all_artifacts_shadow_only": bool(frame["production_eligible"].fillna(True).eq(False).all()),
        "blend_total_delta_net_pnl": float(blend_delta.sum()),
        "blend_median_delta_net_pnl": float(blend_delta.median()),
        "blend_positive_delta_share": float((blend_delta > 0.0).mean()) if len(frame) else float("nan"),
        "later_blend_total_delta_net_pnl": float(later_delta.sum()) if len(later) else 0.0,
        "later_blend_median_delta_net_pnl": float(later_delta.median()) if len(later) else float("nan"),
        "later_blend_q25_delta_net_pnl": float(later_delta.quantile(0.25)) if len(later) else float("nan"),
        "later_blend_positive_delta_share": float((later_delta > 0.0).mean()) if len(later) else float("nan"),
        "max_blend_delta_full_sl_rate": float(pd.to_numeric(frame["blend_delta_full_sl_rate"], errors="coerce").max()),
        "max_blend_delta_timeout_rate": float(pd.to_numeric(frame["blend_delta_timeout_rate"], errors="coerce").max()),
        "min_blend_accepted_jaccard_vs_t1": float(
            pd.to_numeric(frame["blend_accepted_jaccard_vs_t1"], errors="coerce").min()
        ),
        "min_later_windows": int(min_later_windows),
        "min_positive_later_share": float(min_positive_later_share),
        "min_accepted_jaccard": float(min_accepted_jaccard),
        "max_full_sl_delta": float(max_full_sl_delta),
        "max_timeout_delta": float(max_timeout_delta),
    }
    failures = _promotion_failures(
        rollup,
        min_later_windows=min_later_windows,
        min_positive_later_share=min_positive_later_share,
        min_accepted_jaccard=min_accepted_jaccard,
        max_full_sl_delta=max_full_sl_delta,
        max_timeout_delta=max_timeout_delta,
    )
    rollup["shadow_promotion_gate_passed"] = not failures
    rollup["shadow_promotion_failures"] = failures
    rollup["rank_scope_router_should_remain_disabled"] = bool(failures)
    rollup["interpretation"] = (
        "The state-blended short_boll rank router remains a shadow-only research track. "
        "It must not change active ranks or auction ordering until later matured windows "
        "show recurrent improvement without worsening full-SL/timeout rates or excessive accepted-set churn."
    )
    frame.to_csv(output_dir / "rank_scope_blend_window_summary.csv", index=False)
    payload = {"rollup": rollup, "windows": frame.to_dict("records")}
    (output_dir / "rank_scope_blend_validation_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(rollup, frame)
    (output_dir / "rank_scope_blend_validation_report.md").write_text(report, encoding="utf-8")
    return payload


def _render_report(rollup: dict[str, Any], frame: pd.DataFrame) -> str:
    lines = [
        "# Market-State Rank-Scope Blend Validation",
        "",
        "This is a shadow-only validation of the short_boll state-conditioned rank router.",
        "It compares every arm against the exact T1 timestamp-rank baseline in each window.",
        "",
        "## Gate",
        "",
        f"- Passed: `{rollup['shadow_promotion_gate_passed']}`",
        f"- Failures: `{', '.join(rollup['shadow_promotion_failures']) or 'none'}`",
        f"- Later windows: `{rollup['later_window_count']}`",
        f"- Later median delta net PnL: `{rollup['later_blend_median_delta_net_pnl']}`",
        f"- Later q25 delta net PnL: `{rollup['later_blend_q25_delta_net_pnl']}`",
        f"- Later positive-delta share: `{rollup['later_blend_positive_delta_share']}`",
        "",
        "## Windows",
        "",
    ]
    cols = [
        "window_label",
        "is_development_window",
        "baseline_trade_count",
        "baseline_net_pnl",
        "global_delta_net_pnl",
        "switch_delta_net_pnl",
        "blend_delta_net_pnl",
        "blend_delta_full_sl_rate",
        "blend_delta_timeout_rate",
        "blend_accepted_jaccard_vs_t1",
        "blend_timestamp_weight_mean",
    ]
    view = frame[[c for c in cols if c in frame.columns]].copy()
    lines.append(view.to_markdown(index=False))
    lines.extend(["", "## Interpretation", "", str(rollup["interpretation"])])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-label", action="append", default=[])
    parser.add_argument("--min-later-windows", type=int, default=2)
    parser.add_argument("--min-positive-later-share", type=float, default=0.75)
    parser.add_argument("--min-accepted-jaccard", type=float, default=0.75)
    parser.add_argument("--max-full-sl-delta", type=float, default=0.0)
    parser.add_argument("--max-timeout-delta", type=float, default=0.0)
    args = parser.parse_args()
    payload = aggregate_rank_scope_windows(
        args.artifact_dirs,
        output_dir=args.output_dir,
        development_labels=set(args.development_label or []),
        min_later_windows=int(args.min_later_windows),
        min_positive_later_share=float(args.min_positive_later_share),
        min_accepted_jaccard=float(args.min_accepted_jaccard),
        max_full_sl_delta=float(args.max_full_sl_delta),
        max_timeout_delta=float(args.max_timeout_delta),
    )
    print(json.dumps(_json_safe(payload["rollup"]), indent=2))


if __name__ == "__main__":
    main()
