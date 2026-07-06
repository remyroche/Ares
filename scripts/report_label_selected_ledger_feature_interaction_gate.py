#!/usr/bin/env python3
"""Causal raw-feature interaction gate sweep on selected-row ledgers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_label_selected_ledger_feature_gate import (
    DEFAULT_FEATURE_CONTRAST_CSV,
    DEFAULT_INPUT_DIR,
    _attach_features,
    _load_gate_features,
)
from scripts.report_label_selected_ledger_regime_gate import (
    _json_safe,
    _parse_csv,
    _parse_float_csv,
    _week_summary,
)
from scripts.run_label_quality_proxy_diagnostics import DEFAULT_FEATURE_DIR


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/label_dual_target_execution_smoke_s49_allfeatures_feature_interaction_gate_v1"
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _thresholds(fit: pd.DataFrame, feature: str, quantiles: list[float]) -> list[tuple[str, float, str]]:
    values = _safe_numeric(fit[feature]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 10:
        return []
    out: list[tuple[str, float, str]] = []
    seen: set[tuple[str, float]] = set()
    for q in quantiles:
        threshold = float(values.quantile(float(q)))
        if not math.isfinite(threshold):
            continue
        q_label = f"q{int(float(q) * 100)}"
        for op in ("<=", ">="):
            key = (op, round(threshold, 12))
            if key in seen:
                continue
            seen.add(key)
            out.append((op, threshold, q_label))
    return out


def _single_gate_name(feature: str, op: str, q_label: str, threshold: float) -> str:
    return f"{feature}{op}fit_{q_label}({threshold:.6g})"


def _single_mask(frame: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    values = _safe_numeric(frame[feature])
    if op == "<=":
        return values <= threshold
    return values >= threshold


def _selected_rank_frac_mask(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if not {"selected_rank", "selected_rows"}.issubset(frame.columns):
        return pd.Series(True, index=frame.index)
    frac = _safe_numeric(frame["selected_rank"]) / _safe_numeric(frame["selected_rows"]).replace(0, np.nan)
    return frac <= float(threshold)


def _candidate_gates(
    fit: pd.DataFrame,
    *,
    features: list[str],
    quantiles: list[float],
    include_score_combo: bool,
    include_rank_combo: bool,
    max_pair_gates: int,
) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    out: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = []
    usable_features = [feature for feature in features if feature in fit.columns]
    thresholds = {
        feature: _thresholds(fit, feature, quantiles)
        for feature in usable_features
    }
    thresholds = {feature: vals for feature, vals in thresholds.items() if vals}
    score_median = (
        float(_safe_numeric(fit["score"]).quantile(0.50))
        if include_score_combo and "score" in fit.columns
        else float("nan")
    )
    for f1, f2 in combinations(thresholds, 2):
        for op1, t1, q1 in thresholds[f1]:
            for op2, t2, q2 in thresholds[f2]:
                left = _single_gate_name(f1, op1, q1, t1)
                right = _single_gate_name(f2, op2, q2, t2)
                name = f"{left}&{right}"
                out.append(
                    (
                        name,
                        lambda frame, a=f1, ao=op1, at=t1, b=f2, bo=op2, bt=t2: (
                            _single_mask(frame, a, ao, at) & _single_mask(frame, b, bo, bt)
                        ),
                    )
                )
                if include_score_combo and math.isfinite(score_median):
                    out.append(
                        (
                            f"{name}&score>=fit_q50({score_median:.6g})",
                            lambda frame, a=f1, ao=op1, at=t1, b=f2, bo=op2, bt=t2, s=score_median: (
                                _single_mask(frame, a, ao, at)
                                & _single_mask(frame, b, bo, bt)
                                & (_safe_numeric(frame["score"]) >= s)
                            ),
                        )
                    )
                if include_rank_combo and {"selected_rank", "selected_rows"}.issubset(fit.columns):
                    out.append(
                        (
                            f"{name}&selected_rank_frac<=0.75",
                            lambda frame, a=f1, ao=op1, at=t1, b=f2, bo=op2, bt=t2: (
                                _single_mask(frame, a, ao, at)
                                & _single_mask(frame, b, bo, bt)
                                & _selected_rank_frac_mask(frame, 0.75)
                            ),
                        )
                    )
                if max_pair_gates > 0 and len(out) >= int(max_pair_gates):
                    return out
    return out


def summarize_interaction_gates(
    ledger: pd.DataFrame,
    *,
    gate_features: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    quantiles: list[float],
    min_week_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_rows: int,
    min_holdout_rows: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_bad_mae_1r_rate: float,
    max_timeout_rate: float,
    include_score_combo: bool,
    include_rank_combo: bool,
    max_pair_gates: int,
) -> pd.DataFrame:
    subset = ledger[_safe_numeric(ledger["top_frac"]).isin({float(v) for v in top_fracs})].copy()
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
        for gate_name, gate_func in _candidate_gates(
            fit,
            features=gate_features,
            quantiles=quantiles,
            include_score_combo=include_score_combo,
            include_rank_combo=include_rank_combo,
            max_pair_gates=max_pair_gates,
        ):
            fit_mask = gate_func(fit).fillna(False).astype(bool)
            holdout_mask = gate_func(holdout).fillna(False).astype(bool)
            fit_kept = fit.loc[fit_mask].copy()
            holdout_kept = holdout.loc[holdout_mask].copy()
            fit_summary = _week_summary(fit_kept, min_week_rows=min_week_rows)
            holdout_summary = _week_summary(holdout_kept, min_week_rows=min_week_rows)
            row: dict[str, Any] = dict(key_dict)
            row.update(
                {
                    "interaction_gate": gate_name,
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
                and row["fit_material_weeks"] >= min_fit_material_weeks
                and row["fit_positive_week_rate"] >= min_fit_positive_week_rate
                and row["fit_row_mean_u"] > 0.0
                and row["fit_bad_mae_1r_rate"] <= max_bad_mae_1r_rate
                and row["fit_timeout_rate"] <= max_timeout_rate
            )
            holdout_pass = (
                row["holdout_rows"] >= min_holdout_rows
                and row["holdout_material_weeks"] >= min_holdout_material_weeks
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
            "holdout_decision",
            "fit_decision",
            "holdout_row_mean_u",
            "holdout_positive_week_rate",
            "holdout_bad_mae_1r_rate",
            "fit_row_mean_u",
        ],
        ascending=[False, False, False, False, True, False],
    )


def _write_markdown(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_selected_ledger_feature_interaction_gate.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    cols = [
        "fit_decision",
        "holdout_decision",
        "label_arm",
        "weight_arm",
        "selection_mode",
        "top_frac",
        "interaction_gate",
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
    holdout_pass = summary[summary["holdout_decision"].eq("holdout_pass")].copy()
    fit_watch = summary[summary["fit_decision"].eq("fit_watchlist")].copy()
    best_holdout = summary.sort_values(
        ["holdout_row_mean_u", "holdout_positive_week_rate", "holdout_bad_mae_1r_rate"],
        ascending=[False, False, True],
    )
    lines = [
        "# Label Selected-Ledger Raw-Feature Interaction Gate",
        "",
        "Scope: two-feature thresholds are derived from fit-month selected rows, then applied unchanged to the holdout month.",
        "",
        f"Input: `{manifest['input_dir']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        f"Gate features: `{','.join(manifest['gate_features'])}`",
        (
            "Gates: "
            f"min-week-rows `{manifest['gates']['min_week_rows']}`, "
            f"min-fit-material-weeks `{manifest['gates']['min_fit_material_weeks']}`, "
            f"min-holdout-material-weeks `{manifest['gates']['min_holdout_material_weeks']}`, "
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
        "## Best Holdout Rows Before Fit/Strict Filtering",
        "",
        table(best_holdout, cols, limit=50),
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
    feature_dir: Path,
    feature_contrast_csv: Path | None,
    feature_list_csv: Path | None,
    gate_features: list[str],
    top_gate_features: int,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    quantiles: list[float],
    min_week_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_rows: int,
    min_holdout_rows: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_bad_mae_1r_rate: float,
    max_timeout_rate: float,
    include_score_combo: bool,
    include_rank_combo: bool,
    max_pair_gates: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = input_dir / ledger_filename
    ledger = pd.read_csv(ledger_path)
    features = _load_gate_features(
        explicit_features=gate_features,
        feature_contrast_csv=feature_contrast_csv,
        feature_list_csv=feature_list_csv,
        top_n=top_gate_features,
    )
    ledger_with_features, feature_store_report = _attach_features(
        ledger,
        feature_dir=feature_dir,
        gate_features=features,
    )
    retained_features = [feature for feature in features if feature in ledger_with_features.columns]
    summary = summarize_interaction_gates(
        ledger_with_features,
        gate_features=retained_features,
        fit_months=[str(v) for v in fit_months],
        holdout_month=str(holdout_month),
        top_fracs=[float(v) for v in top_fracs],
        quantiles=[float(v) for v in quantiles],
        min_week_rows=int(min_week_rows),
        min_fit_material_weeks=int(min_fit_material_weeks),
        min_holdout_material_weeks=int(min_holdout_material_weeks),
        min_fit_rows=int(min_fit_rows),
        min_holdout_rows=int(min_holdout_rows),
        min_fit_positive_week_rate=float(min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(min_holdout_positive_week_rate),
        max_bad_mae_1r_rate=float(max_bad_mae_1r_rate),
        max_timeout_rate=float(max_timeout_rate),
        include_score_combo=bool(include_score_combo),
        include_rank_combo=bool(include_rank_combo),
        max_pair_gates=int(max_pair_gates),
    )
    paths = {
        "summary": output_dir / "label_selected_ledger_feature_interaction_gate_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "input_dir": str(input_dir),
        "ledger_path": str(ledger_path),
        "ledger_filename": str(ledger_filename),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_contrast_csv": str(feature_contrast_csv) if feature_contrast_csv is not None else None,
        "feature_list_csv": str(feature_list_csv) if feature_list_csv is not None else None,
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "quantiles": [float(v) for v in quantiles],
        "gate_features": retained_features,
        "feature_store": feature_store_report,
        "gates": {
            "min_week_rows": int(min_week_rows),
            "min_fit_material_weeks": int(min_fit_material_weeks),
            "min_holdout_material_weeks": int(min_holdout_material_weeks),
            "min_fit_rows": int(min_fit_rows),
            "min_holdout_rows": int(min_holdout_rows),
            "min_fit_positive_week_rate": float(min_fit_positive_week_rate),
            "min_holdout_positive_week_rate": float(min_holdout_positive_week_rate),
            "max_bad_mae_1r_rate": float(max_bad_mae_1r_rate),
            "max_timeout_rate": float(max_timeout_rate),
            "include_score_combo": bool(include_score_combo),
            "include_rank_combo": bool(include_rank_combo),
            "max_pair_gates": int(max_pair_gates),
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
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-contrast-csv", type=Path, default=DEFAULT_FEATURE_CONTRAST_CSV)
    parser.add_argument("--feature-list-csv", type=Path, default=None)
    parser.add_argument("--gate-features", default="")
    parser.add_argument("--top-gate-features", type=int, default=16)
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--top-fracs", default="0.0025,0.005")
    parser.add_argument("--quantiles", default="0.35,0.50,0.65")
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--min-fit-material-weeks", type=int, default=4)
    parser.add_argument("--min-holdout-material-weeks", type=int, default=2)
    parser.add_argument("--min-fit-rows", type=int, default=20)
    parser.add_argument("--min-holdout-rows", type=int, default=5)
    parser.add_argument("--min-fit-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--min-holdout-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--max-bad-mae-1r-rate", type=float, default=0.50)
    parser.add_argument("--max-timeout-rate", type=float, default=0.20)
    parser.add_argument("--no-score-combo", action="store_true")
    parser.add_argument("--no-rank-combo", action="store_true")
    parser.add_argument("--max-pair-gates", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ledger_filename=str(args.ledger_filename),
        feature_dir=args.feature_dir,
        feature_contrast_csv=args.feature_contrast_csv,
        feature_list_csv=args.feature_list_csv,
        gate_features=_parse_csv(args.gate_features),
        top_gate_features=int(args.top_gate_features),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=_parse_float_csv(args.top_fracs),
        quantiles=_parse_float_csv(args.quantiles),
        min_week_rows=int(args.min_week_rows),
        min_fit_material_weeks=int(args.min_fit_material_weeks),
        min_holdout_material_weeks=int(args.min_holdout_material_weeks),
        min_fit_rows=int(args.min_fit_rows),
        min_holdout_rows=int(args.min_holdout_rows),
        min_fit_positive_week_rate=float(args.min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(args.min_holdout_positive_week_rate),
        max_bad_mae_1r_rate=float(args.max_bad_mae_1r_rate),
        max_timeout_rate=float(args.max_timeout_rate),
        include_score_combo=not bool(args.no_score_combo),
        include_rank_combo=not bool(args.no_rank_combo),
        max_pair_gates=int(args.max_pair_gates),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
