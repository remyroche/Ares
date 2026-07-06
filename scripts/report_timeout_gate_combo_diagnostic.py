#!/usr/bin/env python3
"""Utility-plus-timeout gate diagnostic for label/execution alignment.

This is a pre-training diagnostic. It uses fit-month evidence to choose
utility-oriented feature directions and timeout-stable gate features, then
checks whether their simple rank blends transfer to the holdout month inside
the economic path envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_timeout_feature_stability import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_LABELS_DIR,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _fmt,
    _is_clean,
    _load_frame,
    _month_summary,
    _parse_csv,
    _parse_float_csv,
    _safe_mean,
    _safe_numeric,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import _json_safe, _selection_metrics  # noqa: E402


DEFAULT_STAGE51_DIR = Path("data_perp/reports/timeout_feature_stability_stage51_slow_trade_diag_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/timeout_gate_combo_stage52_v1")
DEFAULT_TOP_FRACS = (0.0025, 0.005, 0.01)
DEFAULT_GATE_PENALTIES = (0.25, 0.50, 1.00, 1.50, 2.00)
DEFAULT_GATE_KEEP_FRACS = (1.0, 0.75, 0.50, 0.25)


def _rank_pct(values: Any) -> pd.Series:
    values = _safe_numeric(values).replace([np.inf, -np.inf], np.nan)
    return values.rank(method="average", pct=True)


def _directed_fit_summary(period_ic: pd.DataFrame, fit_months: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fit_periods = set(str(v) for v in fit_months)
    for feature, group in period_ic.groupby("feature", sort=False):
        fit = group[group["period"].astype(str).isin(fit_periods)].copy()
        if fit.empty:
            continue
        raw_u = _safe_numeric(fit["ic_utility"])
        if not raw_u.notna().any():
            continue
        direction = 1.0 if _safe_mean(raw_u) >= 0.0 else -1.0
        signed_u = direction * _safe_numeric(fit["ic_utility"])
        signed_bad = direction * _safe_numeric(fit["ic_bad_mae"])
        signed_timeout = direction * _safe_numeric(fit["ic_timeout"])
        signed_wide = direction * _safe_numeric(fit["ic_wide_25"])
        fit_utility_min = float(signed_u.min()) if signed_u.notna().any() else float("nan")
        fit_bad_max = float(signed_bad.max()) if signed_bad.notna().any() else float("nan")
        fit_timeout_max = float(signed_timeout.max()) if signed_timeout.notna().any() else float("nan")
        fit_wide_max = float(signed_wide.max()) if signed_wide.notna().any() else float("nan")
        score = (
            (fit_utility_min if math.isfinite(fit_utility_min) else -1.0)
            - max(0.0, fit_bad_max if math.isfinite(fit_bad_max) else 1.0)
            - 0.50 * max(0.0, fit_timeout_max if math.isfinite(fit_timeout_max) else 1.0)
            - 0.25 * max(0.0, fit_wide_max if math.isfinite(fit_wide_max) else 1.0)
        )
        rows.append(
            {
                "feature": str(feature),
                "utility_direction": "high" if direction > 0.0 else "low",
                "utility_direction_sign": direction,
                "fit_utility_ic_min": fit_utility_min,
                "fit_utility_ic_mean": _safe_mean(signed_u),
                "fit_bad_mae_ic_max": fit_bad_max,
                "fit_timeout_ic_max": fit_timeout_max,
                "fit_wide_ic_max": fit_wide_max,
                "fit_candidate_score": float(score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["fit_candidate_score", "fit_utility_ic_min"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _select_utility_candidates(
    *,
    period_ic: pd.DataFrame,
    feature_summary: pd.DataFrame,
    fit_months: list[str],
    max_utility_features: int,
) -> pd.DataFrame:
    directed = _directed_fit_summary(period_ic, fit_months)
    if directed.empty:
        return directed
    meta_cols = ["feature", "feature_family", "mean_finite_frac", "min_nunique"]
    merged = directed.merge(feature_summary[meta_cols], on="feature", how="left")
    strict = merged[
        merged["fit_utility_ic_min"].gt(0.0)
        & merged["fit_bad_mae_ic_max"].le(0.0)
        & merged["fit_candidate_score"].gt(-0.10)
    ].copy()
    if len(strict) < max_utility_features:
        relaxed = merged[merged["fit_utility_ic_mean"].gt(0.0)].copy()
        strict = pd.concat([strict, relaxed], ignore_index=True).drop_duplicates("feature")
    return strict.sort_values(
        ["fit_candidate_score", "fit_utility_ic_min", "fit_utility_ic_mean"],
        ascending=[False, False, False],
    ).head(int(max_utility_features)).reset_index(drop=True)


def _select_timeout_gates(feature_summary: pd.DataFrame, max_timeout_gates: int) -> pd.DataFrame:
    stable = feature_summary[feature_summary["stable_timeout_feature"].astype(bool)].copy()
    if stable.empty:
        stable = feature_summary[
            feature_summary["anti_timeout_fit_pass"].astype(bool)
            & feature_summary["anti_timeout_holdout_pass"].astype(bool)
        ].copy()
    return stable.sort_values(
        ["stable_timeout_feature", "stability_score"],
        ascending=[False, False],
    ).head(int(max_timeout_gates)).reset_index(drop=True)


def _combo_score(
    frame: pd.DataFrame,
    *,
    utility_feature: str,
    utility_direction_sign: float,
    gate_feature: str,
    gate_direction_sign: float,
    gate_penalty: float,
    gate_keep_frac: float,
) -> pd.Series:
    utility_rank = _rank_pct(float(utility_direction_sign) * _safe_numeric(frame[utility_feature]))
    gate_rank = _rank_pct(float(gate_direction_sign) * _safe_numeric(frame[gate_feature]))
    score = utility_rank + float(gate_penalty) * gate_rank
    keep_frac = float(gate_keep_frac)
    if keep_frac < 0.999:
        threshold = max(0.0, min(1.0, 1.0 - keep_frac))
        score = score.where(gate_rank >= threshold, np.nan)
    return score


def _aggregate(monthly: pd.DataFrame, fit_months: list[str], holdout_month: str) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "utility_feature",
        "utility_feature_family",
        "utility_direction",
        "gate_feature",
        "gate_feature_family",
        "gate_direction",
        "gate_penalty",
        "gate_keep_frac",
        "top_frac",
    ]
    for key, group in monthly.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key))
        fit = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        row.update(_month_summary("fit", fit))
        row.update(_month_summary("holdout", holdout))
        row["fit_clean_pass"] = _is_clean(pd.Series(row), "fit")
        row["holdout_clean_standalone_pass"] = _is_clean(pd.Series(row), "holdout")
        row["holdout_clean_pass"] = bool(row["fit_clean_pass"] and row["holdout_clean_standalone_pass"])
        row["positive_dirty_holdout"] = bool(row["holdout_mean_month_u"] > 0.0 and not row["holdout_clean_pass"])
        row["path_risk_score"] = float(
            (row["holdout_mean_month_u"] if math.isfinite(row["holdout_mean_month_u"]) else 0.0)
            + 0.25 * (row["holdout_q10_u"] if math.isfinite(row["holdout_q10_u"]) else 0.0)
            - 0.020 * (row["holdout_bad_mae_1r_rate"] if math.isfinite(row["holdout_bad_mae_1r_rate"]) else 0.0)
            - 0.003 * (row["holdout_p90_mae_norm"] if math.isfinite(row["holdout_p90_mae_norm"]) else 0.0)
            - 0.010 * (row["holdout_timeout_rate"] if math.isfinite(row["holdout_timeout_rate"]) else 0.0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "timeout_gate_combo_report.md"
    if aggregate.empty:
        best_fit = aggregate
        best_holdout = aggregate
    else:
        best_fit = aggregate.sort_values(
            ["fit_clean_pass", "fit_mean_month_u", "fit_worst_month_u", "fit_timeout_rate"],
            ascending=[False, False, False, True],
        )
        best_holdout = aggregate.sort_values(
            ["holdout_clean_pass", "positive_dirty_holdout", "path_risk_score"],
            ascending=[False, False, False],
        )
    cols = [
        "utility_feature",
        "utility_feature_family",
        "utility_direction",
        "gate_feature",
        "gate_feature_family",
        "gate_direction",
        "gate_penalty",
        "gate_keep_frac",
        "top_frac",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "fit_timeout_rate",
        "holdout_mean_month_u",
        "holdout_bad_mae_1r_rate",
        "holdout_p90_mae_norm",
        "holdout_timeout_rate",
        "holdout_clean_pass",
        "path_risk_score",
    ]
    lines = [
        "# Timeout Gate Combo Diagnostic",
        "",
        "Scope: pre-training rank-combo diagnostic. Candidate utility features are selected from fit months only; holdout is reported after selection.",
        "",
        f"Fit months: `{','.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Utility candidates: `{manifest['utility_candidate_count']}`.",
        f"Timeout gates: `{manifest['timeout_gate_count']}`.",
        f"Combos evaluated: `{manifest['combo_rows']}`.",
        "",
        "## Counts",
        "",
        f"- Fit clean rows: `{manifest['fit_clean_rows']}`",
        f"- Holdout clean rows: `{manifest['holdout_clean_rows']}`",
        f"- Positive but dirty holdout rows: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Best Fit-Selected Rows",
        "",
        _table(best_fit, cols, limit=30),
        "",
        "## Best Holdout Rows",
        "",
        _table(best_holdout, cols, limit=30),
        "",
        "## Interpretation",
        "",
        "- A holdout-clean row is useful only if it also passed the fit envelope; otherwise it is exploratory evidence, not a train decision.",
        "- Positive holdout utility with high bad-MAE, p90 MAE, or timeout remains a dirty execution label.",
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Utility candidates: `{manifest['outputs']['utility_candidates']}`",
        f"- Timeout gates: `{manifest['outputs']['timeout_gates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    stage51_dir: Path,
    output_dir: Path,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    gate_penalties: list[float],
    gate_keep_fracs: list[float],
    max_utility_features: int,
    max_timeout_gates: int,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_summary = pd.read_csv(stage51_dir / "timeout_feature_stability.csv")
    period_ic = pd.read_csv(stage51_dir / "timeout_feature_period_ic.csv")
    utility_candidates = _select_utility_candidates(
        period_ic=period_ic,
        feature_summary=feature_summary,
        fit_months=fit_months,
        max_utility_features=max_utility_features,
    )
    timeout_gates = _select_timeout_gates(feature_summary, max_timeout_gates=max_timeout_gates)
    frame, metrics, _, reports = _load_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=None,
        include_causal_outcome_priors=True,
        include_causal_state_path_priors=True,
        include_event_confirmation_features=True,
        include_slow_trade_diagnostic_features=True,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(set(fit_months + [holdout_month]))
    dummy_targets = pd.DataFrame(
        {
            "target_soft": pd.Series(0.0, index=frame.index),
            "target_hard": pd.Series(0.0, index=frame.index),
        }
    )
    monthly_rows: list[dict[str, Any]] = []
    for _, urow in utility_candidates.iterrows():
        utility_feature = str(urow["feature"])
        if utility_feature not in frame.columns:
            continue
        for _, grow in timeout_gates.iterrows():
            gate_feature = str(grow["feature"])
            if gate_feature not in frame.columns:
                continue
            gate_sign = 1.0 if str(grow["direction"]) == "high" else -1.0
            for gate_penalty in gate_penalties:
                for gate_keep_frac in gate_keep_fracs:
                    for period in months:
                        mask = month_series.eq(period)
                        if int(mask.sum()) < 100:
                            continue
                        valid_frame = frame.loc[mask].reset_index(drop=True)
                        valid_metrics = metrics.loc[mask].reset_index(drop=True)
                        valid_target = dummy_targets.loc[mask].reset_index(drop=True)
                        score = _combo_score(
                            valid_frame,
                            utility_feature=utility_feature,
                            utility_direction_sign=float(urow["utility_direction_sign"]),
                            gate_feature=gate_feature,
                            gate_direction_sign=gate_sign,
                            gate_penalty=float(gate_penalty),
                            gate_keep_frac=float(gate_keep_frac),
                        )
                        for top_frac in top_fracs:
                            metric_row = _selection_metrics(
                                frame=valid_frame,
                                metrics=valid_metrics,
                                target=valid_target,
                                score=score,
                                arm=f"{utility_feature}::{gate_feature}",
                                selector="timeout_gate_rank_combo",
                                period=str(period),
                                top_frac=float(top_frac),
                            )
                            metric_row.update(
                                {
                                    "utility_feature": utility_feature,
                                    "utility_feature_family": str(urow.get("feature_family", "")),
                                    "utility_direction": str(urow["utility_direction"]),
                                    "gate_feature": gate_feature,
                                    "gate_feature_family": str(grow.get("feature_family", "")),
                                    "gate_direction": str(grow["direction"]),
                                    "gate_penalty": float(gate_penalty),
                                    "gate_keep_frac": float(gate_keep_frac),
                                    "utility_fit_candidate_score": float(urow["fit_candidate_score"]),
                                    "gate_stability_score": float(grow["stability_score"]),
                                }
                            )
                            monthly_rows.append(metric_row)

    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly, fit_months=fit_months, holdout_month=holdout_month)
    paths = {
        "monthly": output_dir / "timeout_gate_combo_monthly.csv",
        "aggregate": output_dir / "timeout_gate_combo_fit_holdout.csv",
        "utility_candidates": output_dir / "timeout_gate_combo_utility_candidates.csv",
        "timeout_gates": output_dir / "timeout_gate_combo_timeout_gates.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    utility_candidates.to_csv(paths["utility_candidates"], index=False)
    timeout_gates.to_csv(paths["timeout_gates"], index=False)
    manifest = {
        "scope": "timeout_gate_combo_diagnostic",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "stage51_dir": str(stage51_dir),
        "output_dir": str(output_dir),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gate_penalties": [float(v) for v in gate_penalties],
        "gate_keep_fracs": [float(v) for v in gate_keep_fracs],
        "utility_candidate_count": int(len(utility_candidates)),
        "timeout_gate_count": int(len(timeout_gates)),
        "combo_rows": int(len(aggregate)),
        "fit_clean_rows": int(aggregate["fit_clean_pass"].sum()) if not aggregate.empty else 0,
        "holdout_clean_rows": int(aggregate["holdout_clean_pass"].sum()) if not aggregate.empty else 0,
        "positive_dirty_holdout_rows": int(aggregate["positive_dirty_holdout"].sum()) if not aggregate.empty else 0,
        "feature_reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--stage51-dir", type=Path, default=DEFAULT_STAGE51_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-months", type=_parse_csv, default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument(
        "--gate-penalties",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_GATE_PENALTIES),
    )
    parser.add_argument(
        "--gate-keep-fracs",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_GATE_KEEP_FRACS),
    )
    parser.add_argument("--max-utility-features", type=int, default=40)
    parser.add_argument("--max-timeout-gates", type=int, default=4)
    parser.add_argument("--prior-windows-days", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        stage51_dir=args.stage51_dir,
        output_dir=args.output_dir,
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        gate_penalties=list(args.gate_penalties),
        gate_keep_fracs=list(args.gate_keep_fracs),
        max_utility_features=int(args.max_utility_features),
        max_timeout_gates=int(args.max_timeout_gates),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
