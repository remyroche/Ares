#!/usr/bin/env python3
"""Fit-month candidate selection using a weekly lower-tail objective."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_s41_s47_selected_ledger_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_s41_s47_weekly_lower_tail_objective_v1")


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


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.min()) if len(series) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.quantile(q)) if len(series) else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str = "week_selected_rows") -> float:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_rate(frame: pd.DataFrame, value_col: str, weight_col: str = "week_selected_rows") -> float:
    return _weighted_mean(frame, value_col, weight_col=weight_col)


def _summarize_week_side(prefix: str, frame: pd.DataFrame, *, min_week_selected_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_positive_week_rate": float("nan"),
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_row_weighted_mean_u": float("nan"),
            f"{prefix}_q10_week_mean_u": float("nan"),
            f"{prefix}_q25_week_mean_u": float("nan"),
            f"{prefix}_worst_week_mean_u": float("nan"),
            f"{prefix}_row_weighted_bad_mae_1r_rate": float("nan"),
            f"{prefix}_row_weighted_wide_25bps_rate": float("nan"),
            f"{prefix}_row_weighted_timeout_rate": float("nan"),
            f"{prefix}_min_week_selected_rows": 0,
            f"{prefix}_max_week_selected_share": float("nan"),
        }
    mean_u = pd.to_numeric(frame["mean_u"], errors="coerce")
    week_rows = pd.to_numeric(frame["week_selected_rows"], errors="coerce").fillna(0).astype(int)
    material = week_rows >= int(min_week_selected_rows)
    positive = mean_u > 0.0
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_positive_week_rate": float(positive.mean()) if len(positive) else float("nan"),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_row_weighted_mean_u": _weighted_mean(frame, "mean_u"),
        f"{prefix}_q10_week_mean_u": _safe_quantile(mean_u, 0.10),
        f"{prefix}_q25_week_mean_u": _safe_quantile(mean_u, 0.25),
        f"{prefix}_worst_week_mean_u": _safe_min(mean_u),
        f"{prefix}_row_weighted_bad_mae_1r_rate": _weighted_rate(frame, "bad_mae_1r_rate"),
        f"{prefix}_row_weighted_wide_25bps_rate": _weighted_rate(frame, "wide_barrier_25bps_rate"),
        f"{prefix}_row_weighted_timeout_rate": _weighted_rate(frame, "timeout_rate"),
        f"{prefix}_min_week_selected_rows": int(week_rows.min()) if len(week_rows) else 0,
        f"{prefix}_max_week_selected_share": _safe_quantile(frame["week_selected_share"], 1.0),
    }


def _summarize_month_side(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_mean_selected_rows": float("nan"),
            f"{prefix}_bad_mae_1r_rate": float("nan"),
            f"{prefix}_wide_25bps_rate": float("nan"),
            f"{prefix}_timeout_rate": float("nan"),
        }
    mean_u = pd.to_numeric(frame["mean_u"], errors="coerce")
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int((mean_u > 0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_mean_selected_rows": _safe_mean(frame["selected_rows"]),
        f"{prefix}_bad_mae_1r_rate": _safe_mean(frame["bad_mae_1r_rate"]),
        f"{prefix}_wide_25bps_rate": _safe_mean(frame["wide_barrier_25bps_rate"]),
        f"{prefix}_timeout_rate": _safe_mean(frame["timeout_rate"]),
    }


def _fit_objective(row: dict[str, Any]) -> float:
    values = {
        key: float(row.get(key, 0.0))
        for key in [
            "fit_mean_month_u",
            "fit_worst_month_u",
            "fit_row_weighted_mean_u",
            "fit_q25_week_mean_u",
            "fit_q10_week_mean_u",
            "fit_worst_week_mean_u",
            "fit_row_weighted_bad_mae_1r_rate",
            "fit_row_weighted_wide_25bps_rate",
            "fit_row_weighted_timeout_rate",
        ]
        if pd.notna(row.get(key, float("nan")))
    }
    return float(
        1.00 * values.get("fit_mean_month_u", 0.0)
        + 0.75 * values.get("fit_row_weighted_mean_u", 0.0)
        + 0.65 * values.get("fit_q25_week_mean_u", 0.0)
        + 0.35 * values.get("fit_q10_week_mean_u", 0.0)
        + 0.25 * values.get("fit_worst_month_u", 0.0)
        + 0.20 * values.get("fit_worst_week_mean_u", 0.0)
        - 0.025 * values.get("fit_row_weighted_bad_mae_1r_rate", 0.0)
        - 0.020 * values.get("fit_row_weighted_wide_25bps_rate", 0.0)
        - 0.020 * values.get("fit_row_weighted_timeout_rate", 0.0)
    )


def summarize_candidates(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_selected_rows: int,
    min_fit_material_weeks: int,
    min_fit_positive_week_rate: float,
    max_fit_bad_mae_1r_rate: float,
    max_fit_timeout_rate: float,
    min_holdout_material_weeks: int,
    min_holdout_positive_week_rate: float,
    max_holdout_bad_mae_1r_rate: float,
    max_holdout_timeout_rate: float,
) -> pd.DataFrame:
    top_frac_values = {float(v) for v in top_fracs}
    monthly_subset = monthly[pd.to_numeric(monthly["top_frac"], errors="coerce").isin(top_frac_values)].copy()
    weekly_subset = weekly[pd.to_numeric(weekly["top_frac"], errors="coerce").isin(top_frac_values)].copy()
    group_cols = [
        "arm",
        "label_arm",
        "weight_arm",
        "selection_mode",
        "mae_penalty",
        "wide_penalty",
        "timeout_penalty",
        "mae_keep_frac",
        "wide_keep_frac",
        "timeout_keep_frac",
        "top_frac",
    ]
    rows: list[dict[str, Any]] = []
    for key, month_group in monthly_subset.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        arm = str(key_dict["arm"])
        top_frac = float(key_dict["top_frac"])
        week_group = weekly_subset[
            weekly_subset["arm"].eq(arm) & pd.to_numeric(weekly_subset["top_frac"], errors="coerce").eq(top_frac)
        ].copy()
        fit_month = month_group[month_group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = month_group[month_group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty or fit_week.empty or holdout_week.empty:
            continue
        row: dict[str, Any] = dict(key_dict)
        row.update({"fit_month_list": ",".join(fit_months), "holdout_month": str(holdout_month)})
        row.update(_summarize_month_side("fit", fit_month))
        row.update(_summarize_month_side("holdout", holdout_monthly))
        row.update(_summarize_week_side("fit", fit_week, min_week_selected_rows=min_week_selected_rows))
        row.update(_summarize_week_side("holdout", holdout_week, min_week_selected_rows=min_week_selected_rows))
        row["fit_lower_tail_objective"] = _fit_objective(row)
        row["fit_decision"] = (
            "weekly_lower_tail_fit"
            if row["fit_months"] == len(fit_months)
            and row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_month_u"] > 0.0
            and row["fit_material_weeks"] >= min_fit_material_weeks
            and row["fit_positive_week_rate"] >= min_fit_positive_week_rate
            and row["fit_row_weighted_bad_mae_1r_rate"] <= max_fit_bad_mae_1r_rate
            and row["fit_row_weighted_timeout_rate"] <= max_fit_timeout_rate
            else "fit_reject"
        )
        row["holdout_decision"] = (
            "holdout_pass"
            if row["fit_decision"] == "weekly_lower_tail_fit"
            and row["holdout_mean_month_u"] > 0.0
            and row["holdout_material_weeks"] >= min_holdout_material_weeks
            and row["holdout_positive_week_rate"] >= min_holdout_positive_week_rate
            and row["holdout_row_weighted_bad_mae_1r_rate"] <= max_holdout_bad_mae_1r_rate
            and row["holdout_row_weighted_timeout_rate"] <= max_holdout_timeout_rate
            else "holdout_fail_or_not_fit_selected"
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "fit_decision",
            "holdout_decision",
            "fit_lower_tail_objective",
            "holdout_mean_month_u",
            "holdout_positive_week_rate",
        ],
        ascending=[False, False, False, False, False],
    )


def _write_markdown(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_weekly_lower_tail_objective.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "fit_decision",
        "holdout_decision",
        "label_arm",
        "weight_arm",
        "selection_mode",
        "mae_penalty",
        "mae_keep_frac",
        "timeout_keep_frac",
        "top_frac",
        "fit_lower_tail_objective",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_positive_week_rate",
        "fit_q25_week_mean_u",
        "fit_q10_week_mean_u",
        "fit_worst_week_mean_u",
        "fit_row_weighted_bad_mae_1r_rate",
        "fit_row_weighted_timeout_rate",
        "holdout_mean_month_u",
        "holdout_positive_week_rate",
        "holdout_q25_week_mean_u",
        "holdout_worst_week_mean_u",
        "holdout_row_weighted_bad_mae_1r_rate",
        "holdout_row_weighted_timeout_rate",
        "holdout_mean_selected_rows",
    ]
    fit_selected = summary[summary["fit_decision"].eq("weekly_lower_tail_fit")]
    holdout_pass = summary[summary["holdout_decision"].eq("holdout_pass")]
    lines = [
        "# Label Weekly Lower-Tail Objective",
        "",
        "Scope: candidates are selected on fit months using monthly utility plus weekly lower-tail utility; holdout is evaluated after selection.",
        "",
        f"Input: `{manifest['input_dir']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        (
            "Fit gates: "
            f"min material weeks `{manifest['gates']['min_fit_material_weeks']}`, "
            f"positive week rate >= `{manifest['gates']['min_fit_positive_week_rate']}`, "
            f"bad-MAE <= `{manifest['gates']['max_fit_bad_mae_1r_rate']}`, "
            f"timeout <= `{manifest['gates']['max_fit_timeout_rate']}`"
        ),
        (
            "Holdout gates: "
            f"min material weeks `{manifest['gates']['min_holdout_material_weeks']}`, "
            f"positive week rate >= `{manifest['gates']['min_holdout_positive_week_rate']}`, "
            f"bad-MAE <= `{manifest['gates']['max_holdout_bad_mae_1r_rate']}`, "
            f"timeout <= `{manifest['gates']['max_holdout_timeout_rate']}`"
        ),
        "",
        "## Holdout Pass After Weekly-Lower-Tail Fit Selection",
        "",
        table(holdout_pass, cols, limit=50),
        "",
        "## Fit-Selected Candidates",
        "",
        table(fit_selected, cols, limit=50),
        "",
        "## Best Rejected Or Holdout-Failed Rows",
        "",
        table(summary, cols, limit=40),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    input_dir: Path,
    output_dir: Path,
    monthly_filename: str,
    weekly_filename: str,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_selected_rows: int,
    min_fit_material_weeks: int,
    min_fit_positive_week_rate: float,
    max_fit_bad_mae_1r_rate: float,
    max_fit_timeout_rate: float,
    min_holdout_material_weeks: int,
    min_holdout_positive_week_rate: float,
    max_holdout_bad_mae_1r_rate: float,
    max_holdout_timeout_rate: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = input_dir / monthly_filename
    weekly_path = input_dir / weekly_filename
    monthly = pd.read_csv(monthly_path)
    weekly = pd.read_csv(weekly_path)
    summary = summarize_candidates(
        monthly=monthly,
        weekly=weekly,
        fit_months=[str(v) for v in fit_months],
        holdout_month=str(holdout_month),
        top_fracs=[float(v) for v in top_fracs],
        min_week_selected_rows=int(min_week_selected_rows),
        min_fit_material_weeks=int(min_fit_material_weeks),
        min_fit_positive_week_rate=float(min_fit_positive_week_rate),
        max_fit_bad_mae_1r_rate=float(max_fit_bad_mae_1r_rate),
        max_fit_timeout_rate=float(max_fit_timeout_rate),
        min_holdout_material_weeks=int(min_holdout_material_weeks),
        min_holdout_positive_week_rate=float(min_holdout_positive_week_rate),
        max_holdout_bad_mae_1r_rate=float(max_holdout_bad_mae_1r_rate),
        max_holdout_timeout_rate=float(max_holdout_timeout_rate),
    )
    paths = {
        "summary": output_dir / "label_weekly_lower_tail_objective_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "input_dir": str(input_dir),
        "monthly_path": str(monthly_path),
        "weekly_path": str(weekly_path),
        "output_dir": str(output_dir),
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gates": {
            "min_week_selected_rows": int(min_week_selected_rows),
            "min_fit_material_weeks": int(min_fit_material_weeks),
            "min_fit_positive_week_rate": float(min_fit_positive_week_rate),
            "max_fit_bad_mae_1r_rate": float(max_fit_bad_mae_1r_rate),
            "max_fit_timeout_rate": float(max_fit_timeout_rate),
            "min_holdout_material_weeks": int(min_holdout_material_weeks),
            "min_holdout_positive_week_rate": float(min_holdout_positive_week_rate),
            "max_holdout_bad_mae_1r_rate": float(max_holdout_bad_mae_1r_rate),
            "max_holdout_timeout_rate": float(max_holdout_timeout_rate),
        },
        "rows": int(len(summary)),
        "fit_selected_rows": int(summary["fit_decision"].eq("weekly_lower_tail_fit").sum()) if not summary.empty else 0,
        "holdout_pass_rows": int(summary["holdout_decision"].eq("holdout_pass").sum()) if not summary.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, summary, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--monthly-filename", default="label_dual_target_execution_smoke_monthly.csv")
    parser.add_argument("--weekly-filename", default="label_dual_target_execution_smoke_weekly.csv")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--top-fracs", default="0.0025,0.005")
    parser.add_argument("--min-week-selected-rows", type=int, default=3)
    parser.add_argument("--min-fit-material-weeks", type=int, default=4)
    parser.add_argument("--min-fit-positive-week-rate", type=float, default=0.55)
    parser.add_argument("--max-fit-bad-mae-1r-rate", type=float, default=0.50)
    parser.add_argument("--max-fit-timeout-rate", type=float, default=0.20)
    parser.add_argument("--min-holdout-material-weeks", type=int, default=2)
    parser.add_argument("--min-holdout-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--max-holdout-bad-mae-1r-rate", type=float, default=0.50)
    parser.add_argument("--max-holdout-timeout-rate", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        monthly_filename=str(args.monthly_filename),
        weekly_filename=str(args.weekly_filename),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_week_selected_rows=int(args.min_week_selected_rows),
        min_fit_material_weeks=int(args.min_fit_material_weeks),
        min_fit_positive_week_rate=float(args.min_fit_positive_week_rate),
        max_fit_bad_mae_1r_rate=float(args.max_fit_bad_mae_1r_rate),
        max_fit_timeout_rate=float(args.max_fit_timeout_rate),
        min_holdout_material_weeks=int(args.min_holdout_material_weeks),
        min_holdout_positive_week_rate=float(args.min_holdout_positive_week_rate),
        max_holdout_bad_mae_1r_rate=float(args.max_holdout_bad_mae_1r_rate),
        max_holdout_timeout_rate=float(args.max_holdout_timeout_rate),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
