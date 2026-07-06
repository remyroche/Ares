#!/usr/bin/env python3
"""Select S52 state-risk thresholds on fit months and report holdout.

The state-risk overlay grid contains both fit and holdout metrics. This report
chooses risk-filter thresholds using fit metrics only, with explicit long-side
retention floors, then reports the corresponding holdout metrics. It is a
leakage-safety bridge between diagnostic threshold sweeps and a candidate
policy layer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SUMMARY = Path(
    "data_perp/reports/s52_state_risk_penalty_overlay_thresholds_tp100_sl050_cleangross_long_20260705_v1/"
    "s52_state_risk_penalty_overlay_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_state_risk_threshold_policy_v1")


def _parse_float_csv(value: str | None, default: tuple[float, ...] = ()) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _num(frame: pd.DataFrame, col: str, default: float = float("nan")) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _retention(frame: pd.DataFrame, prefix: str, side: str = "long") -> pd.Series:
    adjusted = _num(frame, f"{prefix}_{side}_adjusted_selected_rows", 0.0)
    baseline = _num(frame, f"{prefix}_{side}_baseline_selected_rows", 0.0)
    return adjusted / baseline.clip(lower=1.0)


def _policy_objective(
    frame: pd.DataFrame,
    *,
    prefix: str,
    retention_floor: float,
    retention_penalty: float,
    mean_u_weight: float,
) -> pd.Series:
    all_evw = _num(frame, f"{prefix}_all_sides_adjusted_ev_weighted_first_touch_precision", 0.0).fillna(0.0)
    long_evw = _num(frame, f"{prefix}_long_adjusted_ev_weighted_first_touch_precision", 0.0).fillna(0.0)
    all_u = _num(frame, f"{prefix}_all_sides_adjusted_mean_u", -0.02).fillna(-0.02)
    long_mae = _num(frame, f"{prefix}_long_adjusted_mae_before_mfe_1r_rate", 1.0).fillna(1.0)
    long_underwater = _num(frame, f"{prefix}_long_adjusted_mean_underwater_bars_before_mfe", 20.0).fillna(20.0)
    ret = _retention(frame, prefix, "long").fillna(0.0)
    score = (
        all_evw
        + 0.40 * long_evw
        + float(mean_u_weight) * all_u.clip(lower=-0.02)
        - 0.30 * (long_mae - 0.25).clip(lower=0.0)
        - 0.01 * (long_underwater - 14.0).clip(lower=0.0)
        - float(retention_penalty) * (float(retention_floor) - ret).clip(lower=0.0)
    )
    return score.astype(float)


def _candidate_rows(
    summary: pd.DataFrame,
    *,
    retention_floor: float,
    fit_mean_u_floor: float,
    min_fit_long_evw: float,
    max_fit_long_mae_before_mfe: float,
    min_fit_long_rows: int,
    retention_penalty: float,
    mean_u_weight: float,
) -> pd.DataFrame:
    rows = summary[summary["overlay_action"].astype(str).eq("risk_filter")].copy()
    rows["fit_long_retention"] = _retention(rows, "fit", "long")
    rows["holdout_long_retention"] = _retention(rows, "holdout", "long")
    rows["fit_policy_objective"] = _policy_objective(
        rows,
        prefix="fit",
        retention_floor=float(retention_floor),
        retention_penalty=float(retention_penalty),
        mean_u_weight=float(mean_u_weight),
    )
    rows["holdout_policy_objective"] = _policy_objective(
        rows,
        prefix="holdout",
        retention_floor=float(retention_floor),
        retention_penalty=float(retention_penalty),
        mean_u_weight=float(mean_u_weight),
    )
    rows["fit_all_sides_delta_ev_weighted_first_touch_precision"] = (
        _num(rows, "fit_all_sides_adjusted_ev_weighted_first_touch_precision")
        - _num(rows, "fit_all_sides_baseline_ev_weighted_first_touch_precision")
    )
    rows["holdout_all_sides_delta_ev_weighted_first_touch_precision"] = (
        _num(rows, "holdout_all_sides_adjusted_ev_weighted_first_touch_precision")
        - _num(rows, "holdout_all_sides_baseline_ev_weighted_first_touch_precision")
    )
    rows["fit_long_delta_mae_before_mfe_1r_rate"] = (
        _num(rows, "fit_long_adjusted_mae_before_mfe_1r_rate")
        - _num(rows, "fit_long_baseline_mae_before_mfe_1r_rate")
    )
    rows["holdout_long_delta_mae_before_mfe_1r_rate"] = (
        _num(rows, "holdout_long_adjusted_mae_before_mfe_1r_rate")
        - _num(rows, "holdout_long_baseline_mae_before_mfe_1r_rate")
    )
    fit_mean_u = _num(rows, "fit_all_sides_adjusted_mean_u", -1.0).fillna(-1.0)
    fit_rows = _num(rows, "fit_long_adjusted_selected_rows", 0.0).fillna(0.0)
    fit_long_evw = _num(rows, "fit_long_adjusted_ev_weighted_first_touch_precision", 0.0).fillna(0.0)
    fit_long_mae = _num(rows, "fit_long_adjusted_mae_before_mfe_1r_rate", 1.0).fillna(1.0)
    rows = rows[
        rows["fit_long_retention"].ge(float(retention_floor))
        & fit_mean_u.ge(float(fit_mean_u_floor))
        & fit_long_evw.ge(float(min_fit_long_evw))
        & fit_long_mae.le(float(max_fit_long_mae_before_mfe))
        & fit_rows.ge(float(min_fit_long_rows))
    ].copy()
    return rows


def select_policy_rows(
    summary: pd.DataFrame,
    *,
    retention_floors: list[float],
    fit_mean_u_floor: float,
    min_fit_long_evw: float,
    max_fit_long_mae_before_mfe: float,
    min_fit_long_rows: int,
    retention_penalty: float,
    mean_u_weight: float,
) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for retention_floor in retention_floors:
        candidates = _candidate_rows(
            summary,
            retention_floor=float(retention_floor),
            fit_mean_u_floor=float(fit_mean_u_floor),
            min_fit_long_evw=float(min_fit_long_evw),
            max_fit_long_mae_before_mfe=float(max_fit_long_mae_before_mfe),
            min_fit_long_rows=int(min_fit_long_rows),
            retention_penalty=float(retention_penalty),
            mean_u_weight=float(mean_u_weight),
        )
        if candidates.empty:
            continue
        candidates["retention_floor"] = float(retention_floor)
        sort_cols = [
            "retention_floor",
            "selected_col",
            "fit_policy_objective",
            "fit_all_sides_delta_ev_weighted_first_touch_precision",
            "fit_long_delta_mae_before_mfe_1r_rate",
        ]
        ranked = candidates.sort_values(
            sort_cols,
            ascending=[True, True, False, False, True],
            kind="mergesort",
        )
        selected_parts.append(
            ranked.groupby(["retention_floor", "selected_col"], observed=True, dropna=False).head(1)
        )
    if not selected_parts:
        return pd.DataFrame()
    out = pd.concat(selected_parts, ignore_index=True)
    return out.sort_values(["retention_floor", "selected_col"]).reset_index(drop=True)


def run_report(
    *,
    summary_path: Path,
    output_dir: Path,
    retention_floors: list[float],
    fit_mean_u_floor: float,
    min_fit_long_evw: float,
    max_fit_long_mae_before_mfe: float,
    min_fit_long_rows: int,
    retention_penalty: float,
    mean_u_weight: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    selected = select_policy_rows(
        summary,
        retention_floors=retention_floors,
        fit_mean_u_floor=float(fit_mean_u_floor),
        min_fit_long_evw=float(min_fit_long_evw),
        max_fit_long_mae_before_mfe=float(max_fit_long_mae_before_mfe),
        min_fit_long_rows=int(min_fit_long_rows),
        retention_penalty=float(retention_penalty),
        mean_u_weight=float(mean_u_weight),
    )
    paths = {
        "selected": output_dir / "s52_state_risk_threshold_policy_selected.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_state_risk_threshold_policy.md",
    }
    selected.to_csv(paths["selected"], index=False)
    manifest = {
        "scope": "s52_state_risk_threshold_policy",
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "retention_floors": [float(v) for v in retention_floors],
        "fit_mean_u_floor": float(fit_mean_u_floor),
        "min_fit_long_evw": float(min_fit_long_evw),
        "max_fit_long_mae_before_mfe": float(max_fit_long_mae_before_mfe),
        "min_fit_long_rows": int(min_fit_long_rows),
        "retention_penalty": float(retention_penalty),
        "mean_u_weight": float(mean_u_weight),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    cols = [
        "retention_floor",
        "selected_col",
        "combine_mode",
        "risk_threshold",
        "fit_policy_objective",
        "holdout_policy_objective",
        "fit_long_retention",
        "holdout_long_retention",
        "fit_all_sides_delta_ev_weighted_first_touch_precision",
        "holdout_all_sides_delta_ev_weighted_first_touch_precision",
        "fit_all_sides_adjusted_mean_u",
        "holdout_all_sides_adjusted_mean_u",
        "fit_long_adjusted_selected_rows",
        "holdout_long_adjusted_selected_rows",
        "fit_long_adjusted_ev_weighted_first_touch_precision",
        "holdout_long_adjusted_ev_weighted_first_touch_precision",
        "fit_long_delta_mae_before_mfe_1r_rate",
        "holdout_long_delta_mae_before_mfe_1r_rate",
        "fit_long_adjusted_mean_underwater_bars_before_mfe",
        "holdout_long_adjusted_mean_underwater_bars_before_mfe",
    ]
    lines = [
        "# S52 State-Risk Threshold Policy",
        "",
        "Thresholds are selected using fit-month metrics only, then reported on holdout.",
        "",
        selected[[c for c in cols if c in selected.columns]].to_markdown(index=False)
        if not selected.empty
        else "No policy rows met the fit constraints.",
        "",
        "## Outputs",
        "",
        f"- Selected rows: `{paths['selected']}`",
        f"- Manifest: `{paths['manifest']}`",
    ]
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "selected": str(paths["selected"]),
        "report": str(paths["markdown"]),
        "rows": int(len(selected)),
        "top": _json_safe(selected.head(20).to_dict(orient="records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retention-floors", default="0.25,0.50,0.75")
    parser.add_argument("--fit-mean-u-floor", type=float, default=-0.0100)
    parser.add_argument("--min-fit-long-evw", type=float, default=0.0)
    parser.add_argument("--max-fit-long-mae-before-mfe", type=float, default=1.0)
    parser.add_argument("--min-fit-long-rows", type=int, default=50)
    parser.add_argument("--retention-penalty", type=float, default=0.08)
    parser.add_argument("--mean-u-weight", type=float, default=2.0)
    args = parser.parse_args()
    result = run_report(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        retention_floors=_parse_float_csv(args.retention_floors, ()),
        fit_mean_u_floor=float(args.fit_mean_u_floor),
        min_fit_long_evw=float(args.min_fit_long_evw),
        max_fit_long_mae_before_mfe=float(args.max_fit_long_mae_before_mfe),
        min_fit_long_rows=int(args.min_fit_long_rows),
        retention_penalty=float(args.retention_penalty),
        mean_u_weight=float(args.mean_u_weight),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
