#!/usr/bin/env python3
"""Attribute guarded meta-handoff replay results by source/regime/friction.

This is a diagnostic bridge between the proxy source/regime reports and the
actual simple-policy replay.  It does not refit models or choose thresholds. It
joins replay rows back to their selected handoff rows, then reports whether
source/regime cells fail because gross edge is too small for friction, exits are
too slow, or stop/full-SL behavior dominates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HANDOFF_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1/"
    "meta_regime_context_filter_oos_v1/meta_regime_handoff_candidates_v1"
)
DEFAULT_REPLAY_DIR = DEFAULT_HANDOFF_DIR / "execution_replay_primary_overlay_exec_keys_cost1pct_v1"
DEFAULT_SELECTED_ROWS = DEFAULT_HANDOFF_DIR / "meta_regime_handoff_primary_policy_overlay_execution_selected_candidates.parquet"
DEFAULT_REPLAY_CANDIDATES = DEFAULT_REPLAY_DIR / "meta_handoff_replay_attribution_candidates.parquet"
DEFAULT_OUT_DIR = DEFAULT_REPLAY_DIR / "regime_friction_attribution_v1"

CONTEXT_COLUMNS = (
    "source_tag",
    "source_family",
    "candidate_liquidity_bin",
    "candidate_activity_liquidity_bin",
    "candidate_volatility_bin",
    "candidate_volatility_zscore_bin",
    "candidate_directional_vol_imbalance_bin",
    "candidate_market_dispersion_bin",
    "candidate_aegmm_entropy_bin",
    "candidate_aegmm_distance_bin",
    "candidate_reconstruction_bin",
    "candidate_archetype_side_aegmm_entropy_bin",
    "candidate_archetype_side_aegmm_distance_bin",
    "candidate_archetype_side_liquidity_bin",
    "candidate_archetype_side_volatility_bin",
    "candidate_archetype_side_activity_liquidity_bin",
    "candidate_archetype_side_directional_vol_imbalance_bin",
    "candidate_archetype_side_market_dispersion_bin",
    "candidate_volatility_shape_bin",
    "candidate_exec_move_speed_bin",
    "candidate_archetype_side_exec_move_speed_bin",
    "candidate_exec_signal_to_spread_bin",
    "candidate_archetype_side_exec_signal_to_spread_bin",
    "candidate_exec_slow_resolution_risk_bin",
    "candidate_archetype_side_exec_slow_resolution_risk_bin",
    "candidate_exec_adverse_path_pressure_bin",
    "candidate_archetype_side_exec_adverse_path_pressure_bin",
    "candidate_exec_opportunity_pressure_bin",
    "candidate_archetype_side_exec_opportunity_pressure_bin",
    "ctx_exec_spread_bps_proxy",
    "ctx_exec_liquidity_rank_proxy",
    "ctx_exec_spread_pressure_proxy",
    "ctx_exec_volatility_rank_proxy",
    "ctx_exec_move_speed_proxy",
    "ctx_exec_signal_to_spread_proxy",
    "ctx_exec_aegmm_uncertainty_proxy",
    "ctx_exec_model_risk_pressure_proxy",
    "ctx_exec_adverse_path_pressure_proxy",
    "ctx_exec_slow_resolution_risk_proxy",
    "ctx_exec_opportunity_pressure_proxy",
    "policy_overlay",
)
GROUP_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scenario", ("scenario",)),
    ("scenario_side", ("scenario", "side_name")),
    ("scenario_source", ("scenario", "source_family")),
    ("scenario_side_source", ("scenario", "side_name", "source_family")),
    ("scenario_volatility_shape", ("scenario", "candidate_volatility_shape_bin")),
    ("scenario_liquidity", ("scenario", "candidate_liquidity_bin")),
    ("scenario_activity_liquidity", ("scenario", "candidate_activity_liquidity_bin")),
    ("scenario_volatility", ("scenario", "candidate_volatility_bin")),
    ("scenario_volatility_zscore", ("scenario", "candidate_volatility_zscore_bin")),
    ("scenario_directional_vol_imbalance", ("scenario", "candidate_directional_vol_imbalance_bin")),
    ("scenario_market_dispersion", ("scenario", "candidate_market_dispersion_bin")),
    ("scenario_aegmm_entropy", ("scenario", "candidate_aegmm_entropy_bin")),
    ("scenario_aegmm_distance", ("scenario", "candidate_aegmm_distance_bin")),
    ("scenario_reconstruction", ("scenario", "candidate_reconstruction_bin")),
    (
        "scenario_side_market_dispersion",
        ("scenario", "side_name", "candidate_archetype_side_market_dispersion_bin"),
    ),
    (
        "scenario_side_liquidity",
        ("scenario", "side_name", "candidate_archetype_side_liquidity_bin"),
    ),
    (
        "scenario_side_volatility",
        ("scenario", "side_name", "candidate_archetype_side_volatility_bin"),
    ),
    (
        "scenario_side_aegmm_entropy",
        ("scenario", "side_name", "candidate_archetype_side_aegmm_entropy_bin"),
    ),
    (
        "scenario_side_aegmm_distance",
        ("scenario", "side_name", "candidate_archetype_side_aegmm_distance_bin"),
    ),
    (
        "scenario_side_activity_liquidity",
        ("scenario", "side_name", "candidate_archetype_side_activity_liquidity_bin"),
    ),
    (
        "scenario_side_directional_vol_imbalance",
        ("scenario", "side_name", "candidate_archetype_side_directional_vol_imbalance_bin"),
    ),
    ("scenario_exec_move_speed", ("scenario", "candidate_exec_move_speed_bin")),
    ("scenario_exec_signal_to_spread", ("scenario", "candidate_exec_signal_to_spread_bin")),
    ("scenario_exec_slow_resolution_risk", ("scenario", "candidate_exec_slow_resolution_risk_bin")),
    ("scenario_exec_adverse_path_pressure", ("scenario", "candidate_exec_adverse_path_pressure_bin")),
    ("scenario_exec_opportunity_pressure", ("scenario", "candidate_exec_opportunity_pressure_bin")),
    (
        "scenario_side_exec_move_speed",
        ("scenario", "side_name", "candidate_archetype_side_exec_move_speed_bin"),
    ),
    (
        "scenario_side_exec_signal_to_spread",
        ("scenario", "side_name", "candidate_archetype_side_exec_signal_to_spread_bin"),
    ),
    (
        "scenario_side_exec_slow_resolution_risk",
        ("scenario", "side_name", "candidate_archetype_side_exec_slow_resolution_risk_bin"),
    ),
    (
        "scenario_side_exec_adverse_path_pressure",
        ("scenario", "side_name", "candidate_archetype_side_exec_adverse_path_pressure_bin"),
    ),
    (
        "scenario_side_exec_opportunity_pressure",
        ("scenario", "side_name", "candidate_archetype_side_exec_opportunity_pressure_bin"),
    ),
)

SUPPORTED_MIN_ROWS = 12
SUPPORTED_MIN_MONTHS = 2


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _sum(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.sum()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _side_name_from_replay(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        return frame["side_name"].astype(str)
    side = _num(frame, "side", 1.0).fillna(1.0)
    return pd.Series(np.where(side.lt(0.0), "short", "long"), index=frame.index)


def _selected_context(selected: pd.DataFrame) -> pd.DataFrame:
    rows = selected.reset_index(drop=True).copy()
    rows["archetype_handoff_row_id"] = np.arange(len(rows), dtype=np.int64)
    keep = ["archetype_handoff_row_id", "timestamp", "symbol", "side_name", "month"]
    keep.extend([col for col in CONTEXT_COLUMNS if col in rows.columns])
    keep.extend(
        [
            col
            for col in (
                "meta_regime_score",
                "score_rank_pct_by_month",
                "handoff_candidate_id",
                "policy_overlay_action",
            )
            if col in rows.columns
        ]
    )
    return rows.loc[:, list(dict.fromkeys(keep))].copy()


def _join_context(replay: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = replay.copy()
    work["archetype_handoff_row_id"] = _num(work, "archetype_handoff_row_id").round().astype("Int64")
    context = _selected_context(selected)
    merged = work.merge(context, on="archetype_handoff_row_id", how="left", suffixes=("", "_selected"))
    if "side_name" not in merged.columns:
        merged["side_name"] = _side_name_from_replay(merged)
    else:
        merged["side_name"] = merged["side_name"].fillna(_side_name_from_replay(merged))
    if "month" not in merged.columns or merged["month"].isna().all():
        merged["month"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce").dt.to_period("M").astype(str)
    for col in CONTEXT_COLUMNS:
        if col not in merged.columns:
            merged[col] = "missing"
        merged[col] = merged[col].fillna("missing").astype(str)
    report = {
        "replay_rows": int(len(replay)),
        "selected_rows": int(len(selected)),
        "joined_rows": int(len(merged)),
        "unmatched_context_rows": int(merged["source_family"].astype(str).eq("missing").sum())
        if "source_family" in merged.columns
        else int(len(merged)),
    }
    return merged, report


def _failure_mode(row: dict[str, Any]) -> str:
    net = float(row.get("mean_net_return", np.nan))
    gross_bps = float(row.get("mean_gross_bps", np.nan))
    friction_bps = float(row.get("mean_expected_friction_bps", np.nan))
    timeout = float(row.get("timeout_rate", np.nan))
    full_sl = float(row.get("full_sl_rate", np.nan))
    if math.isfinite(net) and net > 0.0:
        return "replay_positive"
    if math.isfinite(gross_bps) and math.isfinite(friction_bps) and gross_bps <= friction_bps:
        return "insufficient_gross_edge_vs_friction"
    if math.isfinite(full_sl) and full_sl >= 0.20:
        return "stop_path_failure"
    if math.isfinite(timeout) and timeout >= 0.35:
        return "slow_resolution_timeout"
    return "mixed_or_low_support"


def _summarize_group(frame: pd.DataFrame, group_cols: tuple[str, ...], scope: str) -> pd.DataFrame:
    present = [col for col in group_cols if col in frame.columns]
    if not present:
        return pd.DataFrame()
    work = frame.copy()
    exit_reason = work.get("simple_policy_exit_reason", pd.Series("", index=work.index)).astype(str)
    work["is_full_sl"] = exit_reason.eq("full_sl").astype(float)
    work["is_timeout"] = exit_reason.eq("timeout").astype(float)
    work["is_trailing"] = exit_reason.eq("trailing").astype(float)
    work["is_hard_tp"] = exit_reason.eq("hard_tp").astype(float)
    work["gross_bps"] = _num(work, "gross_return") * 10000.0
    work["net_bps"] = _num(work, "net_return") * 10000.0
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(present, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {"scope": scope}
        row.update({col: str(value) for col, value in zip(present, key)})
        row.update(
            {
                "rows": int(len(group)),
                "symbols": int(group["symbol"].astype(str).nunique()) if "symbol" in group.columns else 0,
                "months": int(group["month"].astype(str).nunique()) if "month" in group.columns else 0,
                "mean_gross_return": _mean(group.get("gross_return")),
                "mean_net_return": _mean(group.get("net_return")),
                "sum_net_return": _sum(group.get("net_return")),
                "mean_gross_bps": _mean(group["gross_bps"]),
                "mean_net_bps": _mean(group["net_bps"]),
                "mean_expected_friction_bps": _mean(group.get("expected_friction_bps")),
                "mean_entry_reanchor_bps": _mean(group.get("entry_reanchor_bps")),
                "mean_spread_cost_bps": _mean(group.get("spread_cost_bps")),
                "hit_net_rate": _rate(_num(group, "net_return").gt(0.0)),
                "full_sl_rate": _rate(group["is_full_sl"]),
                "timeout_rate": _rate(group["is_timeout"]),
                "trailing_rate": _rate(group["is_trailing"]),
                "hard_tp_rate": _rate(group["is_hard_tp"]),
                "mean_holding_bars": _mean(group.get("holding_bars")),
                "mean_rank_pct": _mean(group.get("rank_pct")),
                "mean_meta_regime_score": _mean(group.get("meta_regime_score")),
            }
        )
        row["mean_gross_minus_friction_bps"] = (
            row["mean_gross_bps"] - row["mean_expected_friction_bps"]
            if math.isfinite(row["mean_gross_bps"]) and math.isfinite(row["mean_expected_friction_bps"])
            else float("nan")
        )
        row["failure_mode"] = _failure_mode(row)
        rows.append(row)
    return pd.DataFrame(rows)


def run_report(*, selected_rows: Path, replay_candidates: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_parquet(selected_rows)
    replay = pd.read_parquet(replay_candidates)
    joined, join_report = _join_context(replay, selected)
    summaries = []
    for scope, group_cols in GROUP_SPECS:
        summaries.append(_summarize_group(joined, group_cols, scope))
    summary = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(
            ["scope", "mean_net_return", "rows"],
            ascending=[True, False, False],
            kind="mergesort",
        )
    paths = {
        "joined_candidates": out_dir / "meta_handoff_replay_regime_friction_candidates.parquet",
        "summary": out_dir / "meta_handoff_replay_regime_friction_summary.csv",
        "supported_summary": out_dir / "meta_handoff_replay_regime_friction_supported_summary.csv",
        "manifest": out_dir / "manifest.json",
        "report": out_dir / "meta_handoff_replay_regime_friction_report.md",
    }
    supported = pd.DataFrame()
    if not summary.empty:
        supported = summary[
            pd.to_numeric(summary["rows"], errors="coerce").ge(SUPPORTED_MIN_ROWS)
            & pd.to_numeric(summary["months"], errors="coerce").ge(SUPPORTED_MIN_MONTHS)
        ].copy()
    joined.to_parquet(paths["joined_candidates"], index=False)
    summary.to_csv(paths["summary"], index=False)
    supported.to_csv(paths["supported_summary"], index=False)
    manifest = {
        "generated_by": "report_meta_handoff_replay_regime_friction",
        "selected_rows": str(selected_rows),
        "replay_candidates": str(replay_candidates),
        "out_dir": str(out_dir),
        "join_report": join_report,
        "summary_rows": int(len(summary)),
        "supported_min_rows": SUPPORTED_MIN_ROWS,
        "supported_min_months": SUPPORTED_MIN_MONTHS,
        "supported_summary_rows": int(len(supported)),
        "failure_mode_counts": summary["failure_mode"].value_counts(dropna=False).to_dict()
        if "failure_mode" in summary.columns
        else {},
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Meta Handoff Replay Regime Friction",
        "",
        "This report joins selected handoff context back onto simple-policy replay rows.",
        "Metrics are replay outcomes after the configured execution costs; no model is refit here.",
        "",
        "## Join",
        "",
        json.dumps(_json_safe(join_report), indent=2),
        "",
        "## Top Negative Cells",
        "",
    ]
    if summary.empty:
        lines.append("No summary rows.")
    else:
        display_cols = [
            "scope",
            "scenario",
            "side_name",
            "source_family",
            "candidate_liquidity_bin",
            "candidate_activity_liquidity_bin",
            "candidate_volatility_bin",
            "candidate_volatility_zscore_bin",
            "candidate_aegmm_entropy_bin",
            "candidate_aegmm_distance_bin",
            "candidate_volatility_shape_bin",
            "rows",
            "mean_gross_bps",
            "mean_expected_friction_bps",
            "mean_net_bps",
            "full_sl_rate",
            "timeout_rate",
            "failure_mode",
        ]
        existing = [col for col in display_cols if col in summary.columns]
        negative = summary[pd.to_numeric(summary["mean_net_return"], errors="coerce").lt(0.0)]
        lines.append(negative[existing].sort_values(["mean_net_bps", "rows"], ascending=[True, False]).head(20).to_markdown(index=False))
        lines.extend(["", "## Top Positive Cells", ""])
        positive = summary[pd.to_numeric(summary["mean_net_return"], errors="coerce").gt(0.0)]
        lines.append(positive[existing].sort_values(["mean_net_bps", "rows"], ascending=[False, False]).head(20).to_markdown(index=False))
        lines.extend(["", "## Supported Cells", ""])
        if supported.empty:
            lines.append(
                f"No cells met support thresholds: rows >= {SUPPORTED_MIN_ROWS}, months >= {SUPPORTED_MIN_MONTHS}."
            )
        else:
            lines.append(
                supported[existing]
                .sort_values(["mean_net_bps", "rows"], ascending=[False, False])
                .head(30)
                .to_markdown(index=False)
            )
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-rows", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--replay-candidates", type=Path, default=DEFAULT_REPLAY_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        selected_rows=args.selected_rows,
        replay_candidates=args.replay_candidates,
        out_dir=args.out_dir,
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
