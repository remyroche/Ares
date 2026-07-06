#!/usr/bin/env python3
"""Select label-smoke candidates on fit months and evaluate a later holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/label_feature_store_model_smoke_v3_seed_ensemble_focus")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_feature_store_model_smoke_v3_temporal_holdout")


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


def _safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.min()) if len(series) else float("nan")


def _summarize_selection(
    monthly: pd.DataFrame,
    *,
    fit_months: set[str],
    holdout_month: str,
    top_fracs: list[float],
    max_bad_mae_1r_rate: float,
    max_wide_barrier_25bps_rate: float,
    max_timeout_rate: float,
    max_top_symbol_share: float,
) -> pd.DataFrame:
    top_frac_values = {float(v) for v in top_fracs}
    top_frac_ser = pd.to_numeric(monthly["top_frac"], errors="coerce")
    subset = monthly[top_frac_ser.isin(top_frac_values)].copy()
    fit = subset[subset["period"].astype(str).isin(fit_months)].copy()
    holdout = subset[subset["period"].astype(str).eq(str(holdout_month))].copy()
    rows: list[dict[str, Any]] = []
    for key, group in fit.groupby(
        ["arm", "label_arm", "weight_arm", "top_frac"],
        observed=True,
        dropna=False,
    ):
        arm, label_arm, weight_arm, top_frac = key
        h = holdout[holdout["arm"].eq(arm) & pd.to_numeric(holdout["top_frac"], errors="coerce").eq(float(top_frac))]
        if h.empty:
            continue
        holdout_row = h.iloc[0]
        fit_mean = _safe_mean(group["mean_u"])
        fit_worst = _safe_min(group["mean_u"])
        fit_pos = int((pd.to_numeric(group["mean_u"], errors="coerce") > 0.0).sum())
        holdout_mean = float(holdout_row["mean_u"])
        rows.append(
            {
                "arm": arm,
                "label_arm": label_arm,
                "weight_arm": weight_arm,
                "top_frac": float(top_frac),
                "fit_months": ",".join(sorted(fit_months)),
                "fit_positive_months": fit_pos,
                "fit_mean_u": fit_mean,
                "fit_worst_month_mean_u": fit_worst,
                "fit_score_ic_u": _safe_mean(group["score_ic_u"]),
                "fit_wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "fit_bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "fit_timeout_rate": _safe_mean(group.get("timeout_rate", pd.Series(dtype=float))),
                "fit_top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "holdout_month": str(holdout_month),
                "holdout_mean_u": holdout_mean,
                "holdout_hit_u": float(holdout_row["hit_u"]),
                "holdout_q10_u": float(holdout_row["q10_u"]),
                "holdout_delta_mean_u_vs_period": float(holdout_row["delta_mean_u_vs_period"]),
                "holdout_score_ic_u": float(holdout_row["score_ic_u"]),
                "holdout_decile_spearman_u": float(holdout_row["decile_spearman_u"]),
                "holdout_bad_mae_1r_rate": float(holdout_row["bad_mae_1r_rate"]),
                "holdout_wide_barrier_25bps_rate": float(holdout_row["wide_barrier_25bps_rate"]),
                "holdout_timeout_rate": float(holdout_row.get("timeout_rate", float("nan"))),
                "holdout_top_symbol_share": float(holdout_row["top_symbol_share"]),
                "holdout_selected_rows": int(holdout_row["selected_rows"]),
                "decision": (
                    "holdout_watchlist"
                    if fit_pos == len(fit_months)
                    and fit_mean > 0.0
                    and fit_worst > 0.0
                    and holdout_mean > 0.0
                    and float(holdout_row["score_ic_u"]) > 0.0
                    and _safe_mean(group["bad_mae_1r_rate"]) <= max_bad_mae_1r_rate
                    and _safe_mean(group["wide_barrier_25bps_rate"]) <= max_wide_barrier_25bps_rate
                    and _safe_mean(group.get("timeout_rate", pd.Series(dtype=float))) <= max_timeout_rate
                    and _safe_mean(group["top_symbol_share"]) <= max_top_symbol_share
                    and float(holdout_row["bad_mae_1r_rate"]) <= max_bad_mae_1r_rate
                    and float(holdout_row["wide_barrier_25bps_rate"]) <= max_wide_barrier_25bps_rate
                    and float(holdout_row.get("timeout_rate", float("nan"))) <= max_timeout_rate
                    and float(holdout_row["top_symbol_share"]) <= max_top_symbol_share
                    else "reject_or_rework"
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "decision",
            "fit_mean_u",
            "holdout_mean_u",
            "fit_worst_month_mean_u",
        ],
        ascending=[True, False, False, False],
    )


def _write_markdown(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_smoke_temporal_holdout.md"

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
        "arm",
        "top_frac",
        "fit_positive_months",
        "fit_mean_u",
        "fit_worst_month_mean_u",
        "fit_score_ic_u",
        "fit_wide_barrier_25bps_rate",
        "fit_bad_mae_1r_rate",
        "fit_timeout_rate",
        "fit_top_symbol_share",
        "holdout_month",
        "holdout_mean_u",
        "holdout_hit_u",
        "holdout_q10_u",
        "holdout_score_ic_u",
        "holdout_bad_mae_1r_rate",
        "holdout_wide_barrier_25bps_rate",
        "holdout_timeout_rate",
        "holdout_top_symbol_share",
        "holdout_selected_rows",
    ]
    lines = [
        "# Label Smoke Temporal Holdout",
        "",
        "Scope: candidate selection on fit months only, then evaluation on the later holdout month.",
        "",
        f"Input: `{manifest['input_dir']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        f"Gates: bad-MAE <= `{manifest['gates']['max_bad_mae_1r_rate']}`, wide-25bps <= `{manifest['gates']['max_wide_barrier_25bps_rate']}`, timeout <= `{manifest['gates']['max_timeout_rate']}`, top-symbol-share <= `{manifest['gates']['max_top_symbol_share']}`",
        "",
        "## Holdout Watchlist",
        "",
        table(summary[summary["decision"].eq("holdout_watchlist")], cols, limit=50),
        "",
        "## Top Rejected/Rework",
        "",
        table(summary[summary["decision"].ne("holdout_watchlist")], cols, limit=30),
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
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    max_bad_mae_1r_rate: float,
    max_wide_barrier_25bps_rate: float,
    max_timeout_rate: float,
    max_top_symbol_share: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = input_dir / monthly_filename
    monthly = pd.read_csv(monthly_path)
    summary = _summarize_selection(
        monthly,
        fit_months=set(str(month) for month in fit_months),
        holdout_month=str(holdout_month),
        top_fracs=[float(v) for v in top_fracs],
        max_bad_mae_1r_rate=float(max_bad_mae_1r_rate),
        max_wide_barrier_25bps_rate=float(max_wide_barrier_25bps_rate),
        max_timeout_rate=float(max_timeout_rate),
        max_top_symbol_share=float(max_top_symbol_share),
    )
    paths = {
        "summary": output_dir / "label_smoke_temporal_holdout_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "input_dir": str(input_dir),
        "monthly_path": str(monthly_path),
        "monthly_filename": str(monthly_filename),
        "output_dir": str(output_dir),
        "fit_months": [str(month) for month in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gates": {
            "max_bad_mae_1r_rate": float(max_bad_mae_1r_rate),
            "max_wide_barrier_25bps_rate": float(max_wide_barrier_25bps_rate),
            "max_timeout_rate": float(max_timeout_rate),
            "max_top_symbol_share": float(max_top_symbol_share),
        },
        "rows": int(len(summary)),
        "holdout_watchlist_rows": int(summary["decision"].eq("holdout_watchlist").sum()) if not summary.empty else 0,
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
    parser.add_argument(
        "--monthly-filename",
        type=str,
        default="label_feature_store_model_smoke_monthly.csv",
    )
    parser.add_argument("--fit-months", type=str, default="2026-04,2026-05")
    parser.add_argument("--holdout-month", type=str, default="2026-06")
    parser.add_argument("--top-frac", type=float, default=None)
    parser.add_argument("--top-fracs", type=str, default="0.01")
    parser.add_argument("--max-bad-mae-1r-rate", type=float, default=0.60)
    parser.add_argument("--max-wide-barrier-25bps-rate", type=float, default=0.10)
    parser.add_argument("--max-timeout-rate", type=float, default=0.20)
    parser.add_argument("--max-top-symbol-share", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        monthly_filename=str(args.monthly_filename),
        fit_months=[part.strip() for part in args.fit_months.split(",") if part.strip()],
        holdout_month=args.holdout_month,
        top_fracs=(
            [float(args.top_frac)]
            if args.top_frac is not None
            else [float(part.strip()) for part in str(args.top_fracs).split(",") if part.strip()]
        ),
        max_bad_mae_1r_rate=float(args.max_bad_mae_1r_rate),
        max_wide_barrier_25bps_rate=float(args.max_wide_barrier_25bps_rate),
        max_timeout_rate=float(args.max_timeout_rate),
        max_top_symbol_share=float(args.max_top_symbol_share),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
