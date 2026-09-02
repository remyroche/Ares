#!/usr/bin/env python3
"""Build a promotion decision pack for no-training label proxy candidates.

This script is deliberately pre-training. It consumes the proxy reports created
by `ablate_label_candidate_acceptance_layer.py` and classifies label candidates
against explicit learnability and economic gates before any model fit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ACCEPTANCE_DIR = Path("data_perp/reports/label_candidate_acceptance_layer_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_proxy_decision_pack_v1")


GATES: dict[str, float] = {
    "min_months": 3.0,
    "min_positive_months": 3.0,
    "min_mean_u": 0.0,
    "min_worst_month_mean_u": 0.0,
    "min_selected_weeks": 8.0,
    "min_positive_week_share": 0.50,
    "min_q25_week_mean_u": 0.0,
    "min_worst_week_mean_u": -0.020,
    "max_bad_mae_1r_rate": 0.65,
    "max_wide_barrier_25bps_rate": 0.02,
    "min_mean_rows_month": 10.0,
    "max_top_symbol_share": 0.50,
}

IDENTITY_COLS = [
    "label_arm",
    "weight_arm",
    "risk_kind",
    "risk_keep_frac",
    "top_frac",
    "overlay",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(_fmt)
    return view.to_markdown(index=False)


def _candidate_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for col in IDENTITY_COLS:
        if col in frame.columns:
            parts.append(frame[col].astype(str))
        else:
            parts.append(pd.Series("", index=frame.index))
    out = parts[0]
    for part in parts[1:]:
        out = out + "::" + part
    return out


def _with_base_deltas(aggregate: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["label_arm", "weight_arm", "risk_kind", "risk_keep_frac", "top_frac"]
    base_cols = [
        "mean_u",
        "worst_month_mean_u",
        "q25_week_mean_u",
        "worst_week_mean_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
    ]
    base = aggregate[aggregate["overlay"].astype(str) == "base"].copy()
    if base.empty:
        return aggregate
    keep_cols = group_cols + [col for col in base_cols if col in base.columns]
    base = base[keep_cols].rename(columns={col: f"base_{col}" for col in base_cols if col in base.columns})
    out = aggregate.merge(base, on=group_cols, how="left")
    for col in base_cols:
        base_col = f"base_{col}"
        if col in out.columns and base_col in out.columns:
            out[f"delta_{col}_vs_base"] = _safe_numeric(out, col) - _safe_numeric(out, base_col)
    return out


def _classify_candidates(aggregate: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    out = aggregate.copy()
    out["candidate_id"] = _candidate_key(out)
    out["positive_week_share"] = (
        _safe_numeric(out, "positive_selected_weeks") / _safe_numeric(out, "selected_weeks")
    ).replace([np.inf, -np.inf], np.nan)

    monthly = monthly.copy()
    monthly["candidate_id"] = _candidate_key(monthly)
    monthly_max_symbol = (
        monthly.groupby("candidate_id", dropna=False, observed=True)["top_symbol_share"]
        .max()
        .rename("max_month_top_symbol_share")
        .reset_index()
        if "top_symbol_share" in monthly.columns
        else pd.DataFrame({"candidate_id": out["candidate_id"], "max_month_top_symbol_share": np.nan})
    )
    out = out.merge(monthly_max_symbol, on="candidate_id", how="left")

    checks = {
        "gate_months": _safe_numeric(out, "months") >= GATES["min_months"],
        "gate_positive_months": _safe_numeric(out, "positive_months") >= GATES["min_positive_months"],
        "gate_mean_u": _safe_numeric(out, "mean_u") > GATES["min_mean_u"],
        "gate_worst_month": _safe_numeric(out, "worst_month_mean_u") > GATES["min_worst_month_mean_u"],
        "gate_selected_weeks": _safe_numeric(out, "selected_weeks") >= GATES["min_selected_weeks"],
        "gate_positive_week_share": out["positive_week_share"] >= GATES["min_positive_week_share"],
        "gate_q25_week": _safe_numeric(out, "q25_week_mean_u") >= GATES["min_q25_week_mean_u"],
        "gate_worst_week": _safe_numeric(out, "worst_week_mean_u") >= GATES["min_worst_week_mean_u"],
        "gate_bad_mae": _safe_numeric(out, "bad_mae_1r_rate") <= GATES["max_bad_mae_1r_rate"],
        "gate_wide_barrier": _safe_numeric(out, "wide_barrier_25bps_rate") <= GATES["max_wide_barrier_25bps_rate"],
        "gate_rows": _safe_numeric(out, "mean_rows_month") >= GATES["min_mean_rows_month"],
        "gate_symbol_concentration": (
            _safe_numeric(out, "max_month_top_symbol_share") <= GATES["max_top_symbol_share"]
        )
        | out["max_month_top_symbol_share"].isna(),
    }
    for name, values in checks.items():
        out[name] = values.fillna(False)

    gate_cols = list(checks)
    out["failed_gates"] = out[gate_cols].apply(
        lambda row: ",".join(col.removeprefix("gate_") for col, passed in row.items() if not bool(passed)),
        axis=1,
    )
    out["passed_gate_count"] = out[gate_cols].sum(axis=1).astype(int)
    out["gate_count"] = len(gate_cols)
    all_gates = out[gate_cols].all(axis=1)
    high_upside_fragile = (
        (_safe_numeric(out, "positive_months") >= GATES["min_positive_months"])
        & (_safe_numeric(out, "mean_u") > 0.01)
        & (
            (_safe_numeric(out, "wide_barrier_25bps_rate") > GATES["max_wide_barrier_25bps_rate"])
            | (_safe_numeric(out, "q25_week_mean_u") < GATES["min_q25_week_mean_u"])
        )
    )
    near_miss = (
        (_safe_numeric(out, "positive_months") >= 2)
        & (_safe_numeric(out, "mean_u") > 0)
        & (_safe_numeric(out, "passed_gate_count") >= len(gate_cols) - 2)
    )
    out["decision"] = np.select(
        [all_gates, high_upside_fragile, near_miss],
        ["promote_to_training", "challenger_only", "near_miss_retest"],
        default="reject_proxy",
    )
    out["decision_rank_score"] = (
        1.00 * _safe_numeric(out, "mean_u").fillna(-1.0)
        + 0.75 * _safe_numeric(out, "worst_month_mean_u").fillna(-1.0)
        + 0.50 * _safe_numeric(out, "q25_week_mean_u").fillna(-1.0)
        - 0.20 * _safe_numeric(out, "wide_barrier_25bps_rate").fillna(1.0)
        - 0.10 * _safe_numeric(out, "bad_mae_1r_rate").fillna(1.0)
    )
    out = _with_base_deltas(out)
    decision_order = {
        "promote_to_training": 0,
        "challenger_only": 1,
        "near_miss_retest": 2,
        "reject_proxy": 3,
    }
    out["decision_order"] = out["decision"].map(decision_order).fillna(99).astype(int)
    return out.sort_values(
        ["decision_order", "decision_rank_score", "mean_u"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def _focus_rows(decisions: pd.DataFrame) -> pd.DataFrame:
    promoted = decisions[decisions["decision"] == "promote_to_training"].copy()
    challengers = decisions[decisions["decision"] == "challenger_only"].copy()
    near = decisions[decisions["decision"] == "near_miss_retest"].copy()
    return pd.concat(
        [
            promoted.sort_values("decision_rank_score", ascending=False),
            challengers.sort_values("mean_u", ascending=False).head(10),
            near.sort_values("decision_rank_score", ascending=False).head(10),
        ],
        ignore_index=True,
    )


def _write_markdown(
    *,
    output_dir: Path,
    decisions: pd.DataFrame,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_proxy_decision_pack.md"
    focus = _focus_rows(decisions)
    promoted = decisions[decisions["decision"] == "promote_to_training"].copy()
    challengers = decisions[decisions["decision"] == "challenger_only"].copy()
    rejected_best = decisions[decisions["decision"] == "reject_proxy"].copy().head(10)

    cols = [
        "decision",
        "label_arm",
        "weight_arm",
        "risk_kind",
        "risk_keep_frac",
        "top_frac",
        "overlay",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "selected_weeks",
        "positive_week_share",
        "q25_week_mean_u",
        "worst_week_mean_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "mean_rows_month",
        "max_month_top_symbol_share",
        "delta_mean_u_vs_base",
        "delta_q25_week_mean_u_vs_base",
        "failed_gates",
    ]
    monthly_focus = monthly.copy()
    weekly_focus = weekly.copy()
    if not focus.empty:
        ids = set(focus["candidate_id"].astype(str))
        monthly_focus["candidate_id"] = _candidate_key(monthly_focus)
        weekly_focus["candidate_id"] = _candidate_key(weekly_focus)
        monthly_focus = monthly_focus[monthly_focus["candidate_id"].astype(str).isin(ids)]
        weekly_focus = weekly_focus[weekly_focus["candidate_id"].astype(str).isin(ids)]

    monthly_cols = [
        "label_arm",
        "weight_arm",
        "risk_kind",
        "risk_keep_frac",
        "top_frac",
        "overlay",
        "period",
        "rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
    ]
    weekly_cols = [
        "label_arm",
        "weight_arm",
        "risk_kind",
        "risk_keep_frac",
        "top_frac",
        "overlay",
        "month",
        "week",
        "rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
    ]
    gate_lines = [f"- `{name}`: {value}" for name, value in GATES.items()]
    lines = [
        "# Label Proxy Decision Pack",
        "",
        "Scope: no model training. This is a development screen for label learnability within economic limits, not a clean final OOS claim.",
        "",
        "Selection evidence uses proxy-time fields from causal ledgers. Future execution outcomes are used only for evaluation.",
        "",
        "## Promotion Gates",
        "",
        *gate_lines,
        "",
        "## Promote To Training",
        "",
        _table(promoted.sort_values("decision_rank_score", ascending=False), cols, limit=20),
        "",
        "## High-Upside Challengers",
        "",
        _table(challengers.sort_values("mean_u", ascending=False), cols, limit=20),
        "",
        "## Focus Rows",
        "",
        _table(focus, cols, limit=50),
        "",
        "## Best Rejected",
        "",
        _table(rejected_best, cols, limit=10),
        "",
        "## Monthly Focus",
        "",
        _table(monthly_focus.sort_values(IDENTITY_COLS + ["period"]), monthly_cols, limit=120),
        "",
        "## Weekly Focus",
        "",
        _table(weekly_focus.sort_values(IDENTITY_COLS + ["month", "week"]), weekly_cols, limit=160),
        "",
        "## Outputs",
        "",
        f"- Decision summary: `{manifest['outputs']['decision_summary']}`",
        f"- Promotion shortlist: `{manifest['outputs']['promotion_shortlist']}`",
        f"- Focus monthly: `{manifest['outputs']['focus_monthly']}`",
        f"- Focus weekly: `{manifest['outputs']['focus_weekly']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_decision_pack(*, acceptance_dir: Path, output_dir: Path) -> dict[str, Any]:
    aggregate_path = acceptance_dir / "acceptance_aggregate_summary.csv"
    monthly_path = acceptance_dir / "acceptance_monthly_summary.csv"
    weekly_path = acceptance_dir / "acceptance_weekly_summary.csv"
    missing = [path for path in (aggregate_path, monthly_path, weekly_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    aggregate = pd.read_csv(aggregate_path)
    monthly = pd.read_csv(monthly_path)
    weekly = pd.read_csv(weekly_path)
    decisions = _classify_candidates(aggregate, monthly)
    focus = _focus_rows(decisions)
    focus_ids = set(focus["candidate_id"].astype(str))
    monthly["candidate_id"] = _candidate_key(monthly)
    weekly["candidate_id"] = _candidate_key(weekly)
    focus_monthly = monthly[monthly["candidate_id"].astype(str).isin(focus_ids)].copy()
    focus_weekly = weekly[weekly["candidate_id"].astype(str).isin(focus_ids)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "decision_summary": output_dir / "decision_summary.csv",
        "promotion_shortlist": output_dir / "promotion_shortlist.csv",
        "focus_monthly": output_dir / "focus_monthly.csv",
        "focus_weekly": output_dir / "focus_weekly.csv",
        "manifest": output_dir / "manifest.json",
    }
    decisions.to_csv(paths["decision_summary"], index=False)
    decisions[decisions["decision"] == "promote_to_training"].to_csv(
        paths["promotion_shortlist"],
        index=False,
    )
    focus_monthly.to_csv(paths["focus_monthly"], index=False)
    focus_weekly.to_csv(paths["focus_weekly"], index=False)
    manifest = {
        "acceptance_dir": str(acceptance_dir),
        "output_dir": str(output_dir),
        "scope": "no_training_proxy_label_promotion_screen",
        "gates": GATES,
        "selection_fields": ["candidate_score", "risk_score", "__ts__", "week", "month"],
        "evaluation_fields": ["u_policy_net", "barrier", "mae_norm", "hit_u", "q10_u"],
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        decisions=decisions,
        monthly=monthly,
        weekly=weekly,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-dir", type=Path, default=DEFAULT_ACCEPTANCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_decision_pack(acceptance_dir=args.acceptance_dir, output_dir=args.output_dir)
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
