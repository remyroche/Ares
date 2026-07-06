#!/usr/bin/env python3
"""Summarize why the timeout/holding Stage 1 readiness gate fails.

This is read-only over the Stage 1 metrics output. It does not retrain models or
modify source labels; it turns aggregate, weekly, and calibration rows into a
small failure-attribution report for the next label-definition iteration.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/timeout_holding_risk_stage1_metrics_v1"
)


def _num(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    number = _num(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(label for _, label in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in cols) + " |")
    return "\n".join(lines)


def _gate_failures(row: pd.Series) -> list[str]:
    failures: list[str] = []
    if _num(row.get("top_risk_decile_timeout_lift")) < 1.5:
        failures.append("top_risk_timeout_lift")
    if _num(row.get("timeout_rate")) > 0.20:
        failures.append("low_risk_timeout_rate")
    if _num(row.get("delta_mean_u_vs_valid")) < -0.001:
        failures.append("utility_drawdown")
    if _num(row.get("q25_week_u_delta_vs_valid")) <= 0.0:
        failures.append("q25_week")
    if _num(row.get("positive_weeks")) < _num(row.get("valid_positive_weeks")):
        failures.append("positive_weeks")
    if _num(row.get("min_week_selected_rows")) < 5.0:
        failures.append("rows_week")
    return failures


def _candidate_id(row: pd.Series) -> tuple[str, str, str, float]:
    return (
        str(row.get("label")),
        str(row.get("feature_set")),
        str(row.get("selector")),
        float(row.get("fraction")),
    )


def build_report(input_dir: Path) -> dict[str, Path]:
    aggregate = pd.read_csv(input_dir / "timeout_holding_risk_label_aggregate.csv")
    weekly = pd.read_csv(input_dir / "timeout_holding_risk_label_weekly.csv")
    calibration = pd.read_csv(input_dir / "timeout_holding_risk_calibration_deciles.csv")

    candidates = aggregate[
        aggregate["selector"].astype(str).isin(["low_risk_keep", "low_risk_keep_weekly"])
        & pd.to_numeric(aggregate["fraction"], errors="coerce").eq(0.5)
    ].copy()
    candidates = candidates.sort_values(
        ["timeout_reduction_frac_vs_valid", "top_risk_decile_timeout_lift", "target_auc"],
        ascending=False,
        na_position="last",
        kind="mergesort",
    )

    failure_rows: list[dict[str, Any]] = []
    worst_week_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        failures = _gate_failures(row)
        label, feature_set, selector, fraction = _candidate_id(row)
        failure_rows.append(
            {
                "gate": "pass" if not failures else "fail",
                "failures": ", ".join(failures),
                "label": label,
                "feature_set": feature_set,
                "selector": selector,
                "fraction": fraction,
                "AUC": row.get("target_auc"),
                "Brier": row.get("target_brier_score"),
                "top_decile_lift": row.get("top_risk_decile_timeout_lift"),
                "timeout_reduction": row.get("timeout_reduction_frac_vs_valid"),
                "delta_u": row.get("delta_mean_u_vs_valid"),
                "q25_delta": row.get("q25_week_u_delta_vs_valid"),
                "positive_weeks": row.get("positive_weeks"),
                "valid_positive_weeks": row.get("valid_positive_weeks"),
                "min_week_rows": row.get("min_week_selected_rows"),
            }
        )

        subset = weekly[
            weekly["label"].astype(str).eq(label)
            & weekly["feature_set"].astype(str).eq(feature_set)
            & weekly["selector"].astype(str).eq(selector)
            & pd.to_numeric(weekly["fraction"], errors="coerce").eq(fraction)
        ].copy()
        if not subset.empty:
            subset["low_row_depth"] = pd.to_numeric(subset["selected_rows"], errors="coerce") < 5
            subset["non_positive_u"] = pd.to_numeric(subset["mean_u"], errors="coerce") <= 0.0
            view = subset.sort_values(
                ["non_positive_u", "low_row_depth", "mean_u", "selected_rows"],
                ascending=[False, False, True, True],
                na_position="last",
                kind="mergesort",
            ).head(5)
            for _, week in view.iterrows():
                worst_week_rows.append(
                    {
                        "label": label,
                        "feature_set": feature_set,
                        "selector": selector,
                        "week_start": week.get("week_start"),
                        "rows": week.get("selected_rows"),
                        "mean_u": week.get("mean_u"),
                        "timeout": week.get("timeout_rate"),
                        "bad_MAE": week.get("bad_mae_1r_rate"),
                        "top_symbol": week.get("top_symbol_share"),
                        "low_rows": "yes" if bool(week.get("low_row_depth")) else "no",
                    }
                )

        cal = calibration[
            calibration["label"].astype(str).eq(label)
            & calibration["feature_set"].astype(str).eq(feature_set)
            & calibration["risk_decile"].isin([1, 10])
        ].copy()
        if not cal.empty:
            grouped = cal.groupby("risk_decile", dropna=False, observed=True).agg(
                rows=("rows", "sum"),
                score_mean=("score_mean", "mean"),
                target_hard_rate=("target_hard_rate", "mean"),
                timeout_rate=("timeout_rate", "mean"),
                timeout_lift_vs_valid=("timeout_lift_vs_valid", "mean"),
                mean_u=("mean_u", "mean"),
                brier_score=("brier_score", "mean"),
            )
            for decile, decile_row in grouped.reset_index().iterrows():
                calibration_rows.append(
                    {
                        "label": label,
                        "feature_set": feature_set,
                        "selector": selector,
                        "decile": int(decile_row["risk_decile"]),
                        "rows": decile_row["rows"],
                        "score_mean": decile_row["score_mean"],
                        "target_rate": decile_row["target_hard_rate"],
                        "timeout": decile_row["timeout_rate"],
                        "timeout_lift": decile_row["timeout_lift_vs_valid"],
                        "mean_u": decile_row["mean_u"],
                        "Brier": decile_row["brier_score"],
                    }
                )

    failure_csv = input_dir / "timeout_stage1_gate_failures.csv"
    worst_week_csv = input_dir / "timeout_stage1_worst_weeks.csv"
    calibration_csv = input_dir / "timeout_stage1_edge_deciles.csv"
    report_md = input_dir / "timeout_stage1_failure_attribution.md"

    pd.DataFrame(failure_rows).to_csv(failure_csv, index=False)
    pd.DataFrame(worst_week_rows).to_csv(worst_week_csv, index=False)
    pd.DataFrame(calibration_rows).to_csv(calibration_csv, index=False)

    lines = [
        "# Timeout Stage 1 Failure Attribution",
        "",
        f"Input directory: `{input_dir}`",
        "",
        "## Gate Failures",
        "",
        _table(
            failure_rows,
            [
                ("gate", "gate"),
                ("failures", "failures"),
                ("label", "label"),
                ("feature_set", "features"),
                ("selector", "selector"),
                ("AUC", "AUC"),
                ("Brier", "Brier"),
                ("top_decile_lift", "top_decile_lift"),
                ("timeout_reduction", "timeout_reduction"),
                ("delta_u", "delta_u"),
                ("q25_delta", "q25_delta"),
                ("positive_weeks", "positive_weeks"),
                ("valid_positive_weeks", "valid_positive_weeks"),
                ("min_week_rows", "min_week_rows"),
            ],
        ),
        "",
        "## Worst Selected Weeks",
        "",
        _table(
            worst_week_rows,
            [
                ("label", "label"),
                ("feature_set", "features"),
                ("selector", "selector"),
                ("week_start", "week"),
                ("rows", "rows"),
                ("mean_u", "mean_u"),
                ("timeout", "timeout"),
                ("bad_MAE", "bad_MAE"),
                ("top_symbol", "top_symbol"),
                ("low_rows", "low_rows"),
            ],
        ),
        "",
        "## Edge Deciles",
        "",
        _table(
            calibration_rows,
            [
                ("label", "label"),
                ("feature_set", "features"),
                ("selector", "selector"),
                ("decile", "decile"),
                ("rows", "rows"),
                ("score_mean", "score"),
                ("target_rate", "target_rate"),
                ("timeout", "timeout"),
                ("timeout_lift", "timeout_lift"),
                ("mean_u", "mean_u"),
                ("Brier", "Brier"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The timeout signal is directionally learnable: the top risk decile has materially higher timeout risk and the low-risk half suppresses timeout. The gate fails because the low-risk half does not preserve enough positive weeks or minimum selected rows, and the best timeout labels do not improve q25 weekly utility versus the valid universe.",
        "",
        "Next label-definition work should target week-level depth and utility retention before any production training integration.",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": report_md,
        "failures": failure_csv,
        "worst_weeks": worst_week_csv,
        "edge_deciles": calibration_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_report(args.input_dir)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
