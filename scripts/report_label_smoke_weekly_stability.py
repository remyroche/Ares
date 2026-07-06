#!/usr/bin/env python3
"""Report weekly stability of month-level label-smoke selections."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_s41_s47_twostage_weekly_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_s41_s47_weekly_stability_v1")


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


def _summarize_side(prefix: str, frame: pd.DataFrame, *, min_week_selected_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_positive_weeks": 0,
            f"{prefix}_positive_week_rate": float("nan"),
            f"{prefix}_material_positive_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_mean_week_u": float("nan"),
            f"{prefix}_row_weighted_mean_u": float("nan"),
            f"{prefix}_q25_week_mean_u": float("nan"),
            f"{prefix}_worst_week_mean_u": float("nan"),
            f"{prefix}_min_week_selected_rows": 0,
            f"{prefix}_max_week_selected_share": float("nan"),
            f"{prefix}_row_weighted_bad_mae_1r_rate": float("nan"),
            f"{prefix}_max_week_bad_mae_1r_rate": float("nan"),
            f"{prefix}_row_weighted_timeout_rate": float("nan"),
            f"{prefix}_max_week_timeout_rate": float("nan"),
        }

    week_rows = pd.to_numeric(frame["week_selected_rows"], errors="coerce").fillna(0).astype(int)
    mean_u = pd.to_numeric(frame["mean_u"], errors="coerce")
    material = week_rows >= int(min_week_selected_rows)
    material_mean = mean_u[material]
    weeks = int(len(frame))
    material_weeks = int(material.sum())
    positive_weeks = int((mean_u > 0.0).sum())
    material_positive = int((material_mean > 0.0).sum())
    return {
        f"{prefix}_weeks": weeks,
        f"{prefix}_material_weeks": material_weeks,
        f"{prefix}_positive_weeks": positive_weeks,
        f"{prefix}_positive_week_rate": float(positive_weeks / weeks) if weeks else float("nan"),
        f"{prefix}_material_positive_weeks": material_positive,
        f"{prefix}_material_positive_week_rate": float(material_positive / material_weeks)
        if material_weeks
        else float("nan"),
        f"{prefix}_mean_week_u": _safe_mean(mean_u),
        f"{prefix}_row_weighted_mean_u": _weighted_mean(frame, "mean_u"),
        f"{prefix}_q25_week_mean_u": _safe_quantile(mean_u, 0.25),
        f"{prefix}_worst_week_mean_u": _safe_min(mean_u),
        f"{prefix}_min_week_selected_rows": int(week_rows.min()) if len(week_rows) else 0,
        f"{prefix}_max_week_selected_share": _safe_quantile(frame["week_selected_share"], 1.0),
        f"{prefix}_row_weighted_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate"),
        f"{prefix}_max_week_bad_mae_1r_rate": _safe_quantile(frame["bad_mae_1r_rate"], 1.0),
        f"{prefix}_row_weighted_timeout_rate": _weighted_mean(frame, "timeout_rate"),
        f"{prefix}_max_week_timeout_rate": _safe_quantile(frame["timeout_rate"], 1.0),
    }


def summarize_weekly(
    weekly: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_selected_rows: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    min_holdout_material_weeks: int,
    min_holdout_row_weighted_mean_u: float,
    max_holdout_bad_mae_1r_rate: float,
    max_holdout_timeout_rate: float,
) -> pd.DataFrame:
    top_frac_values = {float(v) for v in top_fracs}
    subset = weekly[pd.to_numeric(weekly["top_frac"], errors="coerce").isin(top_frac_values)].copy()
    rows: list[dict[str, Any]] = []
    for key, group in subset.groupby(["arm", "label_arm", "weight_arm", "top_frac"], observed=True, dropna=False):
        arm, label_arm, weight_arm, top_frac = key
        fit = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        row: dict[str, Any] = {
            "arm": arm,
            "label_arm": label_arm,
            "weight_arm": weight_arm,
            "top_frac": float(top_frac),
            "fit_months": ",".join(fit_months),
            "holdout_month": str(holdout_month),
            "selection_mode": str(group["selection_mode"].dropna().iloc[0]) if group["selection_mode"].dropna().size else "",
            "mae_penalty": float(group["mae_penalty"].dropna().iloc[0]) if group["mae_penalty"].dropna().size else float("nan"),
            "wide_penalty": float(group["wide_penalty"].dropna().iloc[0]) if group["wide_penalty"].dropna().size else float("nan"),
            "timeout_penalty": float(group["timeout_penalty"].dropna().iloc[0])
            if group["timeout_penalty"].dropna().size
            else float("nan"),
            "mae_keep_frac": float(group["mae_keep_frac"].dropna().iloc[0])
            if group["mae_keep_frac"].dropna().size
            else float("nan"),
            "wide_keep_frac": float(group["wide_keep_frac"].dropna().iloc[0])
            if group["wide_keep_frac"].dropna().size
            else float("nan"),
            "timeout_keep_frac": float(group["timeout_keep_frac"].dropna().iloc[0])
            if group["timeout_keep_frac"].dropna().size
            else float("nan"),
        }
        row.update(_summarize_side("fit", fit, min_week_selected_rows=min_week_selected_rows))
        row.update(_summarize_side("holdout", holdout, min_week_selected_rows=min_week_selected_rows))
        row["decision"] = (
            "weekly_watchlist"
            if row["fit_positive_week_rate"] >= min_fit_positive_week_rate
            and row["holdout_positive_week_rate"] >= min_holdout_positive_week_rate
            and row["holdout_material_weeks"] >= min_holdout_material_weeks
            and row["holdout_row_weighted_mean_u"] >= min_holdout_row_weighted_mean_u
            and row["holdout_row_weighted_bad_mae_1r_rate"] <= max_holdout_bad_mae_1r_rate
            and row["holdout_row_weighted_timeout_rate"] <= max_holdout_timeout_rate
            else "reject_weekly_fragile"
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["decision", "holdout_row_weighted_mean_u", "holdout_positive_week_rate", "fit_positive_week_rate"],
        ascending=[False, False, False, False],
    )


def _write_markdown(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_smoke_weekly_stability.md"

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
        "decision",
        "label_arm",
        "weight_arm",
        "selection_mode",
        "mae_penalty",
        "mae_keep_frac",
        "timeout_keep_frac",
        "top_frac",
        "fit_positive_week_rate",
        "fit_q25_week_mean_u",
        "fit_worst_week_mean_u",
        "holdout_weeks",
        "holdout_material_weeks",
        "holdout_positive_week_rate",
        "holdout_row_weighted_mean_u",
        "holdout_q25_week_mean_u",
        "holdout_worst_week_mean_u",
        "holdout_min_week_selected_rows",
        "holdout_row_weighted_bad_mae_1r_rate",
        "holdout_max_week_bad_mae_1r_rate",
        "holdout_row_weighted_timeout_rate",
    ]
    lines = [
        "# Label Smoke Weekly Stability",
        "",
        "Scope: weekly stability of month-level selected rows. Rows are not re-ranked within weeks.",
        "",
        f"Input: `{manifest['input_dir']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        (
            "Gates: "
            f"min-week-rows `{manifest['gates']['min_week_selected_rows']}`, "
            f"fit positive week rate >= `{manifest['gates']['min_fit_positive_week_rate']}`, "
            f"holdout positive week rate >= `{manifest['gates']['min_holdout_positive_week_rate']}`, "
            f"holdout material weeks >= `{manifest['gates']['min_holdout_material_weeks']}`, "
            f"holdout row-weighted mean_u >= `{manifest['gates']['min_holdout_row_weighted_mean_u']}`, "
            f"holdout bad-MAE <= `{manifest['gates']['max_holdout_bad_mae_1r_rate']}`, "
            f"holdout timeout <= `{manifest['gates']['max_holdout_timeout_rate']}`"
        ),
        "",
        "## Weekly Watchlist",
        "",
        table(summary[summary["decision"].eq("weekly_watchlist")], cols, limit=50),
        "",
        "## Top Weekly Fragile Rows",
        "",
        table(summary[summary["decision"].ne("weekly_watchlist")], cols, limit=40),
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
    weekly_filename: str,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_selected_rows: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    min_holdout_material_weeks: int,
    min_holdout_row_weighted_mean_u: float,
    max_holdout_bad_mae_1r_rate: float,
    max_holdout_timeout_rate: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly_path = input_dir / weekly_filename
    weekly = pd.read_csv(weekly_path)
    summary = summarize_weekly(
        weekly,
        fit_months=[str(v) for v in fit_months],
        holdout_month=str(holdout_month),
        top_fracs=[float(v) for v in top_fracs],
        min_week_selected_rows=int(min_week_selected_rows),
        min_fit_positive_week_rate=float(min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(min_holdout_positive_week_rate),
        min_holdout_material_weeks=int(min_holdout_material_weeks),
        min_holdout_row_weighted_mean_u=float(min_holdout_row_weighted_mean_u),
        max_holdout_bad_mae_1r_rate=float(max_holdout_bad_mae_1r_rate),
        max_holdout_timeout_rate=float(max_holdout_timeout_rate),
    )
    paths = {
        "summary": output_dir / "label_smoke_weekly_stability_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "input_dir": str(input_dir),
        "weekly_path": str(weekly_path),
        "weekly_filename": weekly_filename,
        "output_dir": str(output_dir),
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gates": {
            "min_week_selected_rows": int(min_week_selected_rows),
            "min_fit_positive_week_rate": float(min_fit_positive_week_rate),
            "min_holdout_positive_week_rate": float(min_holdout_positive_week_rate),
            "min_holdout_material_weeks": int(min_holdout_material_weeks),
            "min_holdout_row_weighted_mean_u": float(min_holdout_row_weighted_mean_u),
            "max_holdout_bad_mae_1r_rate": float(max_holdout_bad_mae_1r_rate),
            "max_holdout_timeout_rate": float(max_holdout_timeout_rate),
        },
        "rows": int(len(summary)),
        "weekly_watchlist_rows": int(summary["decision"].eq("weekly_watchlist").sum()) if not summary.empty else 0,
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
    parser.add_argument("--weekly-filename", default="label_dual_target_execution_smoke_weekly.csv")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--top-fracs", default="0.0025,0.005")
    parser.add_argument("--min-week-selected-rows", type=int, default=3)
    parser.add_argument("--min-fit-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--min-holdout-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--min-holdout-material-weeks", type=int, default=2)
    parser.add_argument("--min-holdout-row-weighted-mean-u", type=float, default=0.0)
    parser.add_argument("--max-holdout-bad-mae-1r-rate", type=float, default=0.50)
    parser.add_argument("--max-holdout-timeout-rate", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weekly_filename=str(args.weekly_filename),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_week_selected_rows=int(args.min_week_selected_rows),
        min_fit_positive_week_rate=float(args.min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(args.min_holdout_positive_week_rate),
        min_holdout_material_weeks=int(args.min_holdout_material_weeks),
        min_holdout_row_weighted_mean_u=float(args.min_holdout_row_weighted_mean_u),
        max_holdout_bad_mae_1r_rate=float(args.max_holdout_bad_mae_1r_rate),
        max_holdout_timeout_rate=float(args.max_holdout_timeout_rate),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
