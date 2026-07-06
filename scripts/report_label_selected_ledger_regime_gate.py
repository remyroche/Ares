#!/usr/bin/env python3
"""Causal abstention-gate sweep on label-smoke selected-row ledgers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_s41_s47_selected_ledger_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_s41_s47_regime_gate_v1")


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


def _weighted_mean(frame: pd.DataFrame, col: str) -> float:
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _week_summary(frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "weeks": 0,
            "material_weeks": 0,
            "positive_weeks": 0,
            "positive_week_rate": float("nan"),
            "material_positive_week_rate": float("nan"),
            "row_mean_u": float("nan"),
            "q25_week_mean_u": float("nan"),
            "worst_week_mean_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "wide_barrier_25bps_rate": float("nan"),
            "timeout_rate": float("nan"),
            "min_week_rows": 0,
            "max_week_share": float("nan"),
        }
    weekly = (
        frame.groupby("week", observed=True)
        .agg(
            week_rows=("u_policy_net", "size"),
            mean_u=("u_policy_net", "mean"),
            bad_mae_1r_rate=("bad_mae_1r", "mean"),
            timeout_rate=("is_timeout", "mean"),
        )
        .reset_index()
    )
    material = weekly["week_rows"] >= int(min_week_rows)
    positive = pd.to_numeric(weekly["mean_u"], errors="coerce") > 0.0
    material_positive = positive & material
    rows = int(len(frame))
    return {
        "rows": rows,
        "weeks": int(len(weekly)),
        "material_weeks": int(material.sum()),
        "positive_weeks": int(positive.sum()),
        "positive_week_rate": float(positive.mean()) if len(positive) else float("nan"),
        "material_positive_week_rate": float(material_positive.sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        "row_mean_u": _weighted_mean(frame, "u_policy_net"),
        "q25_week_mean_u": _safe_quantile(weekly["mean_u"], 0.25),
        "worst_week_mean_u": _safe_min(weekly["mean_u"]),
        "bad_mae_1r_rate": _safe_mean(frame["bad_mae_1r"].astype(float)),
        "wide_barrier_25bps_rate": _safe_mean(frame["wide_barrier_25bps"].astype(float)),
        "timeout_rate": _safe_mean(frame["is_timeout"].astype(float)),
        "min_week_rows": int(weekly["week_rows"].min()) if len(weekly) else 0,
        "max_week_share": float(weekly["week_rows"].max() / rows) if rows else float("nan"),
    }


def _candidate_gates(fit: pd.DataFrame) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    out: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("no_gate", lambda frame: pd.Series(True, index=frame.index)),
    ]

    fixed_low_cols = ["bad_mae_rank", "wide_rank", "timeout_rank"]
    for col in fixed_low_cols:
        if col in fit.columns:
            for threshold in (0.40, 0.50, 0.60, 0.70, 0.80):
                out.append(
                    (
                        f"{col}<={threshold:.2f}",
                        lambda frame, c=col, t=threshold: pd.to_numeric(frame[c], errors="coerce") <= t,
                    )
                )

    if "selected_rank" in fit.columns and "selected_rows" in fit.columns:
        out.append(
            (
                "selected_rank_frac<=0.50",
                lambda frame: (
                    pd.to_numeric(frame["selected_rank"], errors="coerce")
                    / pd.to_numeric(frame["selected_rows"], errors="coerce").replace(0, np.nan)
                )
                <= 0.50,
            )
        )
        out.append(
            (
                "selected_rank_frac<=0.75",
                lambda frame: (
                    pd.to_numeric(frame["selected_rank"], errors="coerce")
                    / pd.to_numeric(frame["selected_rows"], errors="coerce").replace(0, np.nan)
                )
                <= 0.75,
            )
        )

    fixed_high_cols = ["upside_rank"]
    for col in fixed_high_cols:
        if col in fit.columns:
            for threshold in (0.70, 0.80, 0.90):
                out.append(
                    (
                        f"{col}>={threshold:.2f}",
                        lambda frame, c=col, t=threshold: pd.to_numeric(frame[c], errors="coerce") >= t,
                    )
                )

    fit_quantile_cols = ["score", "upside_pred", "bad_mae_pred", "wide_pred", "timeout_pred"]
    for col in fit_quantile_cols:
        if col not in fit.columns:
            continue
        values = pd.to_numeric(fit[col], errors="coerce").dropna()
        if values.empty:
            continue
        if col in {"score", "upside_pred"}:
            for q in (0.25, 0.50, 0.75):
                threshold = float(values.quantile(q))
                out.append(
                    (
                        f"{col}>=fit_q{int(q * 100)}({threshold:.6g})",
                        lambda frame, c=col, t=threshold: pd.to_numeric(frame[c], errors="coerce") >= t,
                    )
                )
        else:
            for q in (0.25, 0.50, 0.75):
                threshold = float(values.quantile(q))
                out.append(
                    (
                        f"{col}<=fit_q{int(q * 100)}({threshold:.6g})",
                        lambda frame, c=col, t=threshold: pd.to_numeric(frame[c], errors="coerce") <= t,
                    )
                )

    if "bad_mae_rank" in fit.columns and "timeout_rank" in fit.columns:
        for mae_threshold in (0.50, 0.60, 0.70):
            for timeout_threshold in (0.70, 0.80):
                out.append(
                    (
                        f"bad_mae_rank<={mae_threshold:.2f}&timeout_rank<={timeout_threshold:.2f}",
                        lambda frame, m=mae_threshold, t=timeout_threshold: (
                            pd.to_numeric(frame["bad_mae_rank"], errors="coerce") <= m
                        )
                        & (pd.to_numeric(frame["timeout_rank"], errors="coerce") <= t),
                    )
                )

    if "bad_mae_rank" in fit.columns and "score" in fit.columns:
        score_median = float(pd.to_numeric(fit["score"], errors="coerce").quantile(0.50))
        for mae_threshold in (0.50, 0.60, 0.70):
            out.append(
                (
                    f"bad_mae_rank<={mae_threshold:.2f}&score>=fit_q50({score_median:.6g})",
                    lambda frame, m=mae_threshold, s=score_median: (
                        pd.to_numeric(frame["bad_mae_rank"], errors="coerce") <= m
                    )
                    & (pd.to_numeric(frame["score"], errors="coerce") >= s),
                )
            )
    return out


def summarize_gates(
    ledger: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_rows: int,
    min_fit_rows: int,
    min_holdout_rows: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_bad_mae_1r_rate: float,
    max_timeout_rate: float,
) -> pd.DataFrame:
    subset = ledger[pd.to_numeric(ledger["top_frac"], errors="coerce").isin({float(v) for v in top_fracs})].copy()
    rows: list[dict[str, Any]] = []
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
    for key, group in subset.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        fit = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        for gate_name, gate_func in _candidate_gates(fit):
            fit_mask = gate_func(fit).fillna(False).astype(bool)
            holdout_mask = gate_func(holdout).fillna(False).astype(bool)
            fit_kept = fit.loc[fit_mask].copy()
            holdout_kept = holdout.loc[holdout_mask].copy()
            fit_summary = _week_summary(fit_kept, min_week_rows=min_week_rows)
            holdout_summary = _week_summary(holdout_kept, min_week_rows=min_week_rows)
            row: dict[str, Any] = dict(key_dict)
            row.update(
                {
                    "gate": gate_name,
                    "fit_months": ",".join(fit_months),
                    "holdout_month": str(holdout_month),
                    "fit_keep_frac": float(len(fit_kept) / len(fit)) if len(fit) else float("nan"),
                    "holdout_keep_frac": float(len(holdout_kept) / len(holdout)) if len(holdout) else float("nan"),
                }
            )
            for prefix, summary in (("fit", fit_summary), ("holdout", holdout_summary)):
                for name, value in summary.items():
                    row[f"{prefix}_{name}"] = value
            fit_pass = (
                row["fit_rows"] >= min_fit_rows
                and row["fit_positive_week_rate"] >= min_fit_positive_week_rate
                and row["fit_row_mean_u"] > 0.0
                and row["fit_bad_mae_1r_rate"] <= max_bad_mae_1r_rate
                and row["fit_timeout_rate"] <= max_timeout_rate
            )
            holdout_pass = (
                row["holdout_rows"] >= min_holdout_rows
                and row["holdout_positive_week_rate"] >= min_holdout_positive_week_rate
                and row["holdout_row_mean_u"] > 0.0
                and row["holdout_bad_mae_1r_rate"] <= max_bad_mae_1r_rate
                and row["holdout_timeout_rate"] <= max_timeout_rate
            )
            row["fit_decision"] = "fit_watchlist" if fit_pass else "fit_reject"
            row["holdout_decision"] = "holdout_pass" if fit_pass and holdout_pass else "holdout_fail_or_not_fit_selected"
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "fit_decision",
            "holdout_decision",
            "fit_row_mean_u",
            "holdout_row_mean_u",
            "holdout_positive_week_rate",
        ],
        ascending=[False, False, False, False, False],
    )


def _write_markdown(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_selected_ledger_regime_gate.md"

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
        "top_frac",
        "gate",
        "fit_rows",
        "fit_keep_frac",
        "fit_row_mean_u",
        "fit_positive_week_rate",
        "fit_q25_week_mean_u",
        "fit_bad_mae_1r_rate",
        "fit_timeout_rate",
        "holdout_rows",
        "holdout_keep_frac",
        "holdout_row_mean_u",
        "holdout_positive_week_rate",
        "holdout_q25_week_mean_u",
        "holdout_worst_week_mean_u",
        "holdout_bad_mae_1r_rate",
        "holdout_timeout_rate",
    ]
    fit_watch = summary[summary["fit_decision"].eq("fit_watchlist")].copy()
    holdout_pass = summary[summary["holdout_decision"].eq("holdout_pass")].copy()
    lines = [
        "# Label Selected-Ledger Abstention Gate",
        "",
        "Scope: gates are derived from fit-month selected rows, then applied unchanged to the holdout month.",
        "",
        f"Input: `{manifest['input_dir']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        (
            "Gates: "
            f"min-week-rows `{manifest['gates']['min_week_rows']}`, "
            f"min-fit-rows `{manifest['gates']['min_fit_rows']}`, "
            f"min-holdout-rows `{manifest['gates']['min_holdout_rows']}`, "
            f"fit positive week rate >= `{manifest['gates']['min_fit_positive_week_rate']}`, "
            f"holdout positive week rate >= `{manifest['gates']['min_holdout_positive_week_rate']}`, "
            f"bad-MAE <= `{manifest['gates']['max_bad_mae_1r_rate']}`, "
            f"timeout <= `{manifest['gates']['max_timeout_rate']}`"
        ),
        "",
        "## Holdout Pass After Fit Selection",
        "",
        table(holdout_pass, cols, limit=50),
        "",
        "## Fit Watchlist",
        "",
        table(fit_watch, cols, limit=50),
        "",
        "## Best Holdout Among All Fit-Swept Gates",
        "",
        table(summary.sort_values(["holdout_row_mean_u", "holdout_positive_week_rate"], ascending=[False, False]), cols, limit=30),
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
    ledger_filename: str,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_rows: int,
    min_fit_rows: int,
    min_holdout_rows: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_bad_mae_1r_rate: float,
    max_timeout_rate: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = input_dir / ledger_filename
    ledger = pd.read_csv(ledger_path)
    summary = summarize_gates(
        ledger,
        fit_months=[str(v) for v in fit_months],
        holdout_month=str(holdout_month),
        top_fracs=[float(v) for v in top_fracs],
        min_week_rows=int(min_week_rows),
        min_fit_rows=int(min_fit_rows),
        min_holdout_rows=int(min_holdout_rows),
        min_fit_positive_week_rate=float(min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(min_holdout_positive_week_rate),
        max_bad_mae_1r_rate=float(max_bad_mae_1r_rate),
        max_timeout_rate=float(max_timeout_rate),
    )
    paths = {
        "summary": output_dir / "label_selected_ledger_regime_gate_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "input_dir": str(input_dir),
        "ledger_path": str(ledger_path),
        "ledger_filename": ledger_filename,
        "output_dir": str(output_dir),
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gates": {
            "min_week_rows": int(min_week_rows),
            "min_fit_rows": int(min_fit_rows),
            "min_holdout_rows": int(min_holdout_rows),
            "min_fit_positive_week_rate": float(min_fit_positive_week_rate),
            "min_holdout_positive_week_rate": float(min_holdout_positive_week_rate),
            "max_bad_mae_1r_rate": float(max_bad_mae_1r_rate),
            "max_timeout_rate": float(max_timeout_rate),
        },
        "rows": int(len(summary)),
        "fit_watchlist_rows": int(summary["fit_decision"].eq("fit_watchlist").sum()) if not summary.empty else 0,
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
    parser.add_argument("--ledger-filename", default="label_dual_target_execution_smoke_selected_ledger.csv")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--top-fracs", default="0.0025,0.005")
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--min-fit-rows", type=int, default=20)
    parser.add_argument("--min-holdout-rows", type=int, default=8)
    parser.add_argument("--min-fit-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--min-holdout-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--max-bad-mae-1r-rate", type=float, default=0.50)
    parser.add_argument("--max-timeout-rate", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ledger_filename=str(args.ledger_filename),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_week_rows=int(args.min_week_rows),
        min_fit_rows=int(args.min_fit_rows),
        min_holdout_rows=int(args.min_holdout_rows),
        min_fit_positive_week_rate=float(args.min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(args.min_holdout_positive_week_rate),
        max_bad_mae_1r_rate=float(args.max_bad_mae_1r_rate),
        max_timeout_rate=float(args.max_timeout_rate),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
