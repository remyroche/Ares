#!/usr/bin/env python3
"""Fit-selected multi-target rank ensemble diagnostic.

This pre-training diagnostic selects causal features for separate objectives
from fit months only, then combines their period-local ranks at selection time.
It is intended to test whether utility, timeout, bad-MAE, wide-barrier, and
holding-time signals overlap enough to produce clean holdout rows before any
base/meta model training.
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
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/multitarget_rank_ensemble_stage54_v1")
DEFAULT_TOP_FRACS = (0.0025, 0.005, 0.01)


ENSEMBLE_ARMS: tuple[dict[str, Any], ...] = (
    {
        "arm": "MT0_utility_only",
        "utility_n": 6,
        "timeout_n": 0,
        "bad_mae_n": 0,
        "wide_n": 0,
        "bars_n": 0,
        "utility_weight": 1.0,
        "timeout_weight": 0.0,
        "bad_mae_weight": 0.0,
        "wide_weight": 0.0,
        "bars_weight": 0.0,
        "risk_keep_frac": 1.0,
    },
    {
        "arm": "MT1_balanced",
        "utility_n": 6,
        "timeout_n": 2,
        "bad_mae_n": 4,
        "wide_n": 2,
        "bars_n": 2,
        "utility_weight": 1.0,
        "timeout_weight": 0.5,
        "bad_mae_weight": 0.5,
        "wide_weight": 0.25,
        "bars_weight": 0.25,
        "risk_keep_frac": 1.0,
    },
    {
        "arm": "MT2_path_heavy",
        "utility_n": 6,
        "timeout_n": 1,
        "bad_mae_n": 6,
        "wide_n": 3,
        "bars_n": 1,
        "utility_weight": 1.0,
        "timeout_weight": 0.25,
        "bad_mae_weight": 1.0,
        "wide_weight": 0.50,
        "bars_weight": 0.25,
        "risk_keep_frac": 1.0,
    },
    {
        "arm": "MT3_time_path",
        "utility_n": 6,
        "timeout_n": 2,
        "bad_mae_n": 4,
        "wide_n": 2,
        "bars_n": 4,
        "utility_weight": 1.0,
        "timeout_weight": 0.75,
        "bad_mae_weight": 0.75,
        "wide_weight": 0.25,
        "bars_weight": 0.50,
        "risk_keep_frac": 1.0,
    },
    {
        "arm": "MT4_risk_heavy",
        "utility_n": 4,
        "timeout_n": 2,
        "bad_mae_n": 6,
        "wide_n": 4,
        "bars_n": 4,
        "utility_weight": 1.0,
        "timeout_weight": 1.0,
        "bad_mae_weight": 1.0,
        "wide_weight": 0.50,
        "bars_weight": 0.50,
        "risk_keep_frac": 1.0,
    },
    {
        "arm": "MT5_path_clean_gate",
        "utility_n": 6,
        "timeout_n": 1,
        "bad_mae_n": 6,
        "wide_n": 3,
        "bars_n": 1,
        "utility_weight": 1.0,
        "timeout_weight": 0.25,
        "bad_mae_weight": 1.0,
        "wide_weight": 0.50,
        "bars_weight": 0.25,
        "risk_keep_frac": 0.50,
    },
    {
        "arm": "MT6_time_clean_gate",
        "utility_n": 6,
        "timeout_n": 2,
        "bad_mae_n": 2,
        "wide_n": 1,
        "bars_n": 4,
        "utility_weight": 1.0,
        "timeout_weight": 1.0,
        "bad_mae_weight": 0.25,
        "wide_weight": 0.10,
        "bars_weight": 0.75,
        "risk_keep_frac": 0.50,
    },
    {
        "arm": "MT7_strict_risk_gate",
        "utility_n": 4,
        "timeout_n": 2,
        "bad_mae_n": 6,
        "wide_n": 4,
        "bars_n": 4,
        "utility_weight": 1.0,
        "timeout_weight": 1.0,
        "bad_mae_weight": 1.0,
        "wide_weight": 0.50,
        "bars_weight": 0.50,
        "risk_keep_frac": 0.35,
    },
)


def _rank_pct(values: Any) -> pd.Series:
    values = _safe_numeric(values).replace([np.inf, -np.inf], np.nan)
    return values.rank(method="average", pct=True)


def _fit_feature_summary(period_ic: pd.DataFrame, fit_months: list[str]) -> pd.DataFrame:
    fit_periods = set(str(v) for v in fit_months)
    rows: list[dict[str, Any]] = []
    for feature, group in period_ic.groupby("feature", sort=False):
        fit = group[group["period"].astype(str).isin(fit_periods)].copy()
        if fit.empty:
            continue

        def signed_stats(target: str, want_high: bool) -> tuple[float, str, float, float, float, float, float]:
            raw = _safe_numeric(fit[f"ic_{target}"])
            if not raw.notna().any():
                return 1.0, "high", float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            if want_high:
                sign = 1.0 if _safe_mean(raw) >= 0.0 else -1.0
            else:
                sign = -1.0 if _safe_mean(raw) > 0.0 else 1.0
            signed_target = sign * raw
            signed_u = sign * _safe_numeric(fit["ic_utility"])
            signed_timeout = sign * _safe_numeric(fit["ic_timeout"])
            signed_bad = sign * _safe_numeric(fit["ic_bad_mae"])
            target_min = float(signed_target.min()) if signed_target.notna().any() else float("nan")
            target_max = float(signed_target.max()) if signed_target.notna().any() else float("nan")
            return (
                sign,
                "high" if sign > 0.0 else "low",
                _safe_mean(signed_target),
                target_min,
                target_max,
                _safe_mean(signed_u),
                max(
                    0.0,
                    float(signed_timeout.max()) if signed_timeout.notna().any() else 1.0,
                    float(signed_bad.max()) if signed_bad.notna().any() else 1.0,
                ),
            )

        util = signed_stats("utility", True)
        timeout = signed_stats("timeout", False)
        bad = signed_stats("bad_mae", False)
        wide = signed_stats("wide_25", False)
        bars = signed_stats("bars_policy", False)
        rows.append(
            {
                "feature": str(feature),
                "utility_sign": util[0],
                "utility_direction": util[1],
                "utility_fit_mean": util[2],
                "utility_fit_min": util[3],
                "utility_fit_max": util[4],
                "utility_conflict_penalty": util[6],
                "timeout_sign": timeout[0],
                "timeout_direction": timeout[1],
                "timeout_fit_mean": timeout[2],
                "timeout_fit_min": timeout[3],
                "timeout_fit_max": timeout[4],
                "timeout_utility_fit_mean": timeout[5],
                "bad_mae_sign": bad[0],
                "bad_mae_direction": bad[1],
                "bad_mae_fit_mean": bad[2],
                "bad_mae_fit_min": bad[3],
                "bad_mae_fit_max": bad[4],
                "bad_mae_utility_fit_mean": bad[5],
                "wide_sign": wide[0],
                "wide_direction": wide[1],
                "wide_fit_mean": wide[2],
                "wide_fit_min": wide[3],
                "wide_fit_max": wide[4],
                "wide_utility_fit_mean": wide[5],
                "bars_sign": bars[0],
                "bars_direction": bars[1],
                "bars_fit_mean": bars[2],
                "bars_fit_min": bars[3],
                "bars_fit_max": bars[4],
                "bars_utility_fit_mean": bars[5],
            }
        )
    return pd.DataFrame(rows)


def _select_target_features(
    fit_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    *,
    target: str,
    max_features: int,
    require_positive_utility: bool = False,
) -> pd.DataFrame:
    if max_features <= 0:
        return pd.DataFrame()
    merged = fit_summary.merge(
        feature_summary[["feature", "feature_family", "mean_finite_frac", "min_nunique"]],
        on="feature",
        how="left",
    )
    if target == "utility":
        score = (
            merged["utility_fit_min"].fillna(-1.0)
            + 0.50 * merged["utility_fit_mean"].fillna(-1.0)
            - 0.25 * merged["utility_conflict_penalty"].fillna(1.0)
        )
        out = merged.assign(selection_target=target, selection_score=score)
        out = out[out["utility_fit_mean"].gt(0.0)]
        direction_col = "utility_direction"
        sign_col = "utility_sign"
    else:
        mean_col = f"{target}_fit_mean"
        max_col = f"{target}_fit_max"
        utility_col = f"{target}_utility_fit_mean"
        score = -2.0 * merged[max_col].fillna(1.0) - merged[mean_col].fillna(1.0)
        score = score + 0.50 * merged[utility_col].fillna(-1.0)
        out = merged.assign(selection_target=target, selection_score=score)
        out = out[out[mean_col].lt(0.0)]
        if require_positive_utility:
            out = out[out[utility_col].gt(0.0)]
        direction_col = f"{target}_direction"
        sign_col = f"{target}_sign"
    out = out.rename(
        columns={
            direction_col: "selection_direction",
            sign_col: "selection_sign",
        }
    )
    cols = [
        "selection_target",
        "feature",
        "feature_family",
        "selection_direction",
        "selection_sign",
        "selection_score",
        "utility_fit_mean",
        "utility_fit_min",
        "timeout_fit_mean",
        "timeout_fit_max",
        "bad_mae_fit_mean",
        "bad_mae_fit_max",
        "wide_fit_mean",
        "wide_fit_max",
        "bars_fit_mean",
        "bars_fit_max",
        "mean_finite_frac",
        "min_nunique",
    ]
    return out.sort_values("selection_score", ascending=False)[cols].head(int(max_features)).reset_index(drop=True)


def _mean_rank(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    if selected.empty:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    ranks: list[pd.Series] = []
    for _, row in selected.iterrows():
        feature = str(row["feature"])
        if feature not in frame.columns:
            continue
        sign = float(row["selection_sign"])
        ranks.append(_rank_pct(sign * _safe_numeric(frame[feature])))
    if not ranks:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.concat(ranks, axis=1).mean(axis=1)


def _score_for_period(frame: pd.DataFrame, groups: dict[str, pd.DataFrame], arm: dict[str, Any]) -> pd.Series:
    utility = _mean_rank(frame, groups["utility"].head(int(arm["utility_n"])))
    score = float(arm["utility_weight"]) * utility.fillna(0.5)
    risk_parts: list[pd.Series] = []
    for target, n_key, weight_key in [
        ("timeout", "timeout_n", "timeout_weight"),
        ("bad_mae", "bad_mae_n", "bad_mae_weight"),
        ("wide", "wide_n", "wide_weight"),
        ("bars", "bars_n", "bars_weight"),
    ]:
        n = int(arm[n_key])
        weight = float(arm[weight_key])
        if n <= 0 or weight <= 0.0:
            continue
        rank = _mean_rank(frame, groups[target].head(n))
        score = score + weight * rank.fillna(0.5)
        risk_parts.append(rank)
    if risk_parts and float(arm["risk_keep_frac"]) < 0.999:
        risk_composite = pd.concat(risk_parts, axis=1).mean(axis=1)
        threshold = max(0.0, min(1.0, 1.0 - float(arm["risk_keep_frac"])))
        score = score.where(risk_composite >= threshold, np.nan)
    return score


def _aggregate(monthly: pd.DataFrame, fit_months: list[str], holdout_month: str) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "arm",
        "utility_n",
        "timeout_n",
        "bad_mae_n",
        "wide_n",
        "bars_n",
        "utility_weight",
        "timeout_weight",
        "bad_mae_weight",
        "wide_weight",
        "bars_weight",
        "risk_keep_frac",
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


def _write_markdown(
    output_dir: Path,
    aggregate: pd.DataFrame,
    selected: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "multitarget_rank_ensemble_report.md"
    if aggregate.empty:
        best_fit = aggregate
        best_holdout = aggregate
    else:
        best_fit = aggregate.sort_values(
            ["fit_clean_pass", "fit_mean_month_u", "fit_worst_month_u", "fit_timeout_rate", "fit_p90_mae_norm"],
            ascending=[False, False, False, True, True],
        )
        best_holdout = aggregate.sort_values(
            ["holdout_clean_pass", "positive_dirty_holdout", "path_risk_score"],
            ascending=[False, False, False],
        )
    result_cols = [
        "arm",
        "top_frac",
        "utility_n",
        "timeout_n",
        "bad_mae_n",
        "wide_n",
        "bars_n",
        "risk_keep_frac",
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
    selected_cols = [
        "selection_target",
        "feature",
        "feature_family",
        "selection_direction",
        "selection_score",
        "utility_fit_mean",
        "timeout_fit_mean",
        "bad_mae_fit_mean",
        "wide_fit_mean",
        "bars_fit_mean",
    ]
    lines = [
        "# Multi-Target Rank Ensemble Diagnostic",
        "",
        "Scope: pre-training rank-ensemble diagnostic. Target feature groups are selected from fit months only; holdout is reported after selection.",
        "",
        f"Fit months: `{','.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Arms evaluated: `{manifest['arms_evaluated']}`.",
        f"Aggregate rows: `{manifest['aggregate_rows']}`.",
        "",
        "## Counts",
        "",
        f"- Fit clean rows: `{manifest['fit_clean_rows']}`",
        f"- Holdout clean rows: `{manifest['holdout_clean_rows']}`",
        f"- Positive but dirty holdout rows: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Selected Features",
        "",
        _table(selected, selected_cols, limit=60),
        "",
        "## Best Fit Rows",
        "",
        _table(best_fit, result_cols, limit=30),
        "",
        "## Best Holdout Rows",
        "",
        _table(best_holdout, result_cols, limit=30),
        "",
        "## Interpretation",
        "",
        "- A trainable label candidate needs the same arm to pass fit and holdout economic envelopes.",
        "- Holdout-only positives remain exploratory evidence, not a train decision.",
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Selected features: `{manifest['outputs']['selected_features']}`",
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
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    max_features_per_target: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_summary = pd.read_csv(stage51_dir / "timeout_feature_stability.csv")
    period_ic = pd.read_csv(stage51_dir / "timeout_feature_period_ic.csv")
    fit_summary = _fit_feature_summary(period_ic, fit_months)
    groups = {
        "utility": _select_target_features(
            fit_summary,
            feature_summary,
            target="utility",
            max_features=max_features_per_target,
        ),
        "timeout": _select_target_features(
            fit_summary,
            feature_summary,
            target="timeout",
            max_features=max_features_per_target,
            require_positive_utility=True,
        ),
        "bad_mae": _select_target_features(
            fit_summary,
            feature_summary,
            target="bad_mae",
            max_features=max_features_per_target,
            require_positive_utility=True,
        ),
        "wide": _select_target_features(
            fit_summary,
            feature_summary,
            target="wide",
            max_features=max_features_per_target,
            require_positive_utility=False,
        ),
        "bars": _select_target_features(
            fit_summary,
            feature_summary,
            target="bars",
            max_features=max_features_per_target,
            require_positive_utility=False,
        ),
    }
    selected = pd.concat(groups.values(), ignore_index=True)
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
    for period in months:
        mask = month_series.eq(period)
        if int(mask.sum()) < 100:
            continue
        valid_frame = frame.loc[mask].reset_index(drop=True)
        valid_metrics = metrics.loc[mask].reset_index(drop=True)
        valid_target = dummy_targets.loc[mask].reset_index(drop=True)
        for arm in ENSEMBLE_ARMS:
            score = _score_for_period(valid_frame, groups, arm)
            for top_frac in top_fracs:
                row = _selection_metrics(
                    frame=valid_frame,
                    metrics=valid_metrics,
                    target=valid_target,
                    score=score,
                    arm=str(arm["arm"]),
                    selector="multitarget_rank_ensemble",
                    period=str(period),
                    top_frac=float(top_frac),
                )
                row.update({key: value for key, value in arm.items() if key != "arm"})
                monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly, fit_months=fit_months, holdout_month=holdout_month)
    paths = {
        "monthly": output_dir / "multitarget_rank_ensemble_monthly.csv",
        "aggregate": output_dir / "multitarget_rank_ensemble_fit_holdout.csv",
        "selected_features": output_dir / "multitarget_rank_ensemble_selected_features.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    selected.to_csv(paths["selected_features"], index=False)
    manifest = {
        "scope": "multitarget_rank_ensemble_diagnostic",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "stage51_dir": str(stage51_dir),
        "output_dir": str(output_dir),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "max_features_per_target": int(max_features_per_target),
        "target_feature_counts": {key: int(len(value)) for key, value in groups.items()},
        "arms_evaluated": int(len(ENSEMBLE_ARMS)),
        "monthly_rows": int(len(monthly)),
        "aggregate_rows": int(len(aggregate)),
        "fit_clean_rows": int(aggregate["fit_clean_pass"].sum()) if not aggregate.empty else 0,
        "holdout_clean_rows": int(aggregate["holdout_clean_pass"].sum()) if not aggregate.empty else 0,
        "positive_dirty_holdout_rows": int(aggregate["positive_dirty_holdout"].sum()) if not aggregate.empty else 0,
        "feature_reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, aggregate, selected, manifest)
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
    parser.add_argument("--prior-windows-days", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument("--max-features-per-target", type=int, default=12)
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
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        max_features_per_target=int(args.max_features_per_target),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
