#!/usr/bin/env python3
"""Timestamp-local first-touch abstention proxy grid before model training.

This diagnostic tests label learnability within economic limits. It fits only
causal prior-month feature-rank proxies, rejects rows whose execution-risk
proxies are too high, then ranks the remaining rows inside each timestamp.
No LightGBM, Optuna, policy geometry, or base/meta training is run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_LABELS_DIR,
    DEFAULT_TOP_FRACS,
    _diag_columns,
    _feature_columns,
    _first_touch_metrics,
    _fit_holdout_summary,
    _format_table,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _parse_csv,
    _parse_float_csv,
    _path_metrics,
    _proxy_score,
    _read_feature_list,
    _safe_max,
    _safe_mean,
    _safe_min,
    _safe_numeric,
    _safe_quantile,
    _selection_metrics_ext,
    _spearman,
    _target_components,
    _target_for_selection,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_first_touch_abstention_proxy_grid_v1")
DEFAULT_SCORE_NAMES = (
    "utility",
    "clean",
    "fast",
    "utility_clean50",
    "utility_minus_exec_risk",
    "clean_utility_minus_dirty",
)
DEFAULT_DIRTY_MAX = (0.35, 0.50, 0.65)
DEFAULT_EARLY_MAX = (0.50, 0.65)
DEFAULT_SLOW_MAX = (0.50, 0.65)
DEFAULT_CLEAN_MIN = ("none", "0.45")
DEFAULT_FAST_MIN = ("none", "0.45")


@dataclass(frozen=True)
class AbstentionSpec:
    name: str
    score_name: str
    dirty_max: float
    early_max: float
    slow_max: float
    clean_min: float | None
    fast_min: float | None


def _parse_optional_float_csv(value: str) -> list[float | None]:
    out: list[float | None] = []
    for part in str(value).split(","):
        item = part.strip().lower()
        if not item:
            continue
        out.append(None if item in {"none", "nan", "null"} else float(item))
    return out


def _fmt_threshold(value: float | None, prefix: str) -> str:
    if value is None:
        return f"{prefix}none"
    return f"{prefix}{int(round(float(value) * 100)):02d}"


def _build_specs(
    *,
    score_names: list[str],
    dirty_max: list[float],
    early_max: list[float],
    slow_max: list[float],
    clean_min: list[float | None],
    fast_min: list[float | None],
    max_specs: int | None,
) -> list[AbstentionSpec]:
    specs: list[AbstentionSpec] = []
    for score_name, d_max, e_max, s_max, c_min, f_min in product(
        score_names,
        dirty_max,
        early_max,
        slow_max,
        clean_min,
        fast_min,
    ):
        if c_min is not None and f_min is not None:
            # Clean and fast proxies are correlated; requiring both sharply
            # collapses timestamp coverage and mostly duplicates stricter dirty gates.
            continue
        name = "_".join(
            [
                f"A{len(specs):04d}",
                str(score_name),
                _fmt_threshold(d_max, "d"),
                _fmt_threshold(e_max, "e"),
                _fmt_threshold(s_max, "t"),
                _fmt_threshold(c_min, "c"),
                _fmt_threshold(f_min, "f"),
            ]
        )
        specs.append(
            AbstentionSpec(
                name=name,
                score_name=str(score_name),
                dirty_max=float(d_max),
                early_max=float(e_max),
                slow_max=float(s_max),
                clean_min=None if c_min is None else float(c_min),
                fast_min=None if f_min is None else float(f_min),
            )
        )
        if max_specs is not None and len(specs) >= int(max_specs):
            return specs
    return specs


def _score_expression(score_name: str, proxies: dict[str, pd.Series]) -> pd.Series:
    utility = _safe_numeric(proxies["utility"])
    clean = _safe_numeric(proxies["clean_first_touch"])
    fast = _safe_numeric(proxies["fast_edge"])
    early = _safe_numeric(proxies["early_adverse"])
    slow = _safe_numeric(proxies["slow_timeout"])
    dirty = _safe_numeric(proxies["dirty"])
    if score_name == "utility":
        return utility
    if score_name == "clean":
        return clean
    if score_name == "fast":
        return fast
    if score_name == "utility_clean50":
        return 0.50 * utility + 0.50 * clean
    if score_name == "utility_clean_fast":
        return 0.40 * utility + 0.40 * clean + 0.20 * fast
    if score_name == "utility_minus_dirty025":
        return utility - 0.25 * dirty
    if score_name == "utility_minus_exec_risk":
        return utility - 0.20 * dirty - 0.20 * early - 0.10 * slow
    if score_name == "clean_utility_minus_dirty":
        return 0.50 * clean + 0.50 * utility - 0.25 * dirty
    raise ValueError(f"unknown score name: {score_name}")


def _score_from_spec(spec: AbstentionSpec, proxies: dict[str, pd.Series]) -> pd.Series:
    score = _score_expression(spec.score_name, proxies)
    mask = pd.Series(True, index=score.index)
    mask &= _safe_numeric(proxies["dirty"]) <= float(spec.dirty_max)
    mask &= _safe_numeric(proxies["early_adverse"]) <= float(spec.early_max)
    mask &= _safe_numeric(proxies["slow_timeout"]) <= float(spec.slow_max)
    if spec.clean_min is not None:
        mask &= _safe_numeric(proxies["clean_first_touch"]) >= float(spec.clean_min)
    if spec.fast_min is not None:
        mask &= _safe_numeric(proxies["fast_edge"]) >= float(spec.fast_min)
    return _safe_numeric(score).where(mask)


def _candidate_stats(frame: pd.DataFrame, score: pd.Series) -> dict[str, Any]:
    score_series = _safe_numeric(score).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    finite = np.isfinite(score_series.to_numpy(dtype=np.float64))
    rows = int(len(score_series))
    candidate_rows = int(finite.sum())
    total_ts = int(timestamps.nunique(dropna=True))
    active_counts = timestamps[finite].value_counts(dropna=True)
    active_ts = int(len(active_counts))
    return {
        "candidate_rows": candidate_rows,
        "candidate_rate": float(candidate_rows / rows) if rows else float("nan"),
        "candidate_timestamps": active_ts,
        "timestamp_count": total_ts,
        "candidate_timestamp_coverage": float(active_ts / total_ts) if total_ts else float("nan"),
        "mean_candidates_per_active_ts": _safe_mean(active_counts),
        "p10_candidates_per_active_ts": _safe_quantile(active_counts, 0.10),
        "p50_candidates_per_active_ts": _safe_quantile(active_counts, 0.50),
        "p90_candidates_per_active_ts": _safe_quantile(active_counts, 0.90),
    }


def _monthly_weekly_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    score_arm: str,
    month: str,
    top_fracs: list[float],
    component_diag: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    score_reset = _safe_numeric(score).reset_index(drop=True)
    frame_reset = valid_frame.reset_index(drop=True)
    metrics_reset = valid_metrics.reset_index(drop=True)
    target_reset = valid_target.reset_index(drop=True)
    selector = "first_touch_abstention_proxy_oos_timestamp"
    for frac in top_fracs:
        row = _selection_metrics_ext(
            frame=frame_reset,
            metrics=metrics_reset,
            target=target_reset,
            score=score_reset,
            arm=score_arm,
            selector=selector,
            period=str(month),
            top_frac=float(frac),
            selection_mode="timestamp",
        )
        row.update(_candidate_stats(frame_reset, score_reset))
        row.update(_diag_columns(component_diag))
        monthly_rows.append(row)

        weeks = frame_reset["__ts__"].dt.to_period("W-SUN").astype(str)
        for week, ids in pd.Series(np.arange(len(frame_reset)), index=frame_reset.index).groupby(weeks, dropna=False):
            pos = ids.to_numpy(dtype=np.int64)
            if len(pos) < 20:
                continue
            week_score = score_reset.iloc[pos].reset_index(drop=True)
            week_frame = frame_reset.iloc[pos].reset_index(drop=True)
            week_row = _selection_metrics_ext(
                frame=week_frame,
                metrics=metrics_reset.iloc[pos].reset_index(drop=True),
                target=target_reset.iloc[pos].reset_index(drop=True),
                score=week_score,
                arm=score_arm,
                selector=selector,
                period=str(month),
                top_frac=float(frac),
                selection_mode="timestamp",
            )
            week_row["week"] = str(week)
            week_row["week_selected_rows"] = int(week_row["selected_rows"])
            week_row["week_selected_share"] = float(week_row["selected_rows"] / len(pos)) if len(pos) else float("nan")
            week_row.update(_candidate_stats(week_frame, week_score))
            week_row.update(_diag_columns(component_diag))
            weekly_rows.append(week_row)
    return monthly_rows, weekly_rows


def _sum_metric(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return float("nan")
    return float(_safe_numeric(frame[col]).fillna(0.0).sum())


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(float(denominator)) or float(denominator) <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _coverage_summary(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_candidate_rows": 0,
            f"{prefix}_candidate_rate": float("nan"),
            f"{prefix}_candidate_timestamp_coverage": float("nan"),
            f"{prefix}_selected_from_candidate_rate": float("nan"),
            f"{prefix}_mean_candidates_per_active_ts": float("nan"),
        }
    rows = _sum_metric(frame, "rows")
    candidate_rows = _sum_metric(frame, "candidate_rows")
    selected_rows = _sum_metric(frame, "selected_rows")
    return {
        f"{prefix}_candidate_rows": int(candidate_rows),
        f"{prefix}_candidate_rate": _ratio(candidate_rows, rows),
        f"{prefix}_candidate_timestamp_coverage": _safe_mean(frame["candidate_timestamp_coverage"]),
        f"{prefix}_selected_from_candidate_rate": _ratio(selected_rows, candidate_rows),
        f"{prefix}_mean_candidates_per_active_ts": _safe_mean(frame["mean_candidates_per_active_ts"]),
        f"{prefix}_p10_candidates_per_active_ts": _safe_mean(frame["p10_candidates_per_active_ts"]),
        f"{prefix}_p50_candidates_per_active_ts": _safe_mean(frame["p50_candidates_per_active_ts"]),
    }


def _augment_fit_holdout(
    *,
    fit_holdout: pd.DataFrame,
    monthly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
) -> pd.DataFrame:
    if fit_holdout.empty:
        return fit_holdout
    rows: list[dict[str, Any]] = []
    for _, row in fit_holdout.iterrows():
        arm = str(row["score_arm"])
        frac = float(row["top_frac"])
        group = monthly[
            monthly["arm"].astype(str).eq(arm)
            & _safe_numeric(monthly["top_frac"]).eq(frac)
        ].copy()
        fit = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        extra = {}
        extra.update(_coverage_summary("fit", fit))
        extra.update(_coverage_summary("holdout", holdout))
        rows.append(extra)
    extra_frame = pd.DataFrame(rows, index=fit_holdout.index)
    out = pd.concat([fit_holdout.reset_index(drop=True), extra_frame.reset_index(drop=True)], axis=1)
    return out.sort_values(
        [
            "holdout_clean_pass",
            "holdout_bounded_pass",
            "positive_dirty_holdout",
            "exec_risk_score",
            "holdout_candidate_timestamp_coverage",
        ],
        ascending=[False, False, False, False, False],
    )


def _write_markdown(
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    components_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "first_touch_abstention_proxy_grid.md"
    cols = [
        "score_arm",
        "top_frac",
        "exec_risk_score",
        "fit_sign_pass",
        "fit_bounded_pass",
        "holdout_sign_pass",
        "holdout_bounded_pass",
        "fit_mean_month_u",
        "fit_material_positive_week_rate",
        "fit_candidate_rate",
        "fit_candidate_timestamp_coverage",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_candidate_rate",
        "holdout_candidate_timestamp_coverage",
        "holdout_selected_from_candidate_rate",
        "holdout_clean_exec_actual_rate",
        "holdout_first_touch_timeout_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_p90_first_touch_mae_to_sl",
    ]
    component_cols = [
        "period",
        "component",
        "proxy_top_abs_ic",
        "proxy_mean_top_abs_ic",
        "proxy_ic_actual_u",
        "proxy_ic_actual_clean_exec",
        "proxy_ic_actual_dirty",
        "proxy_ic_actual_fast_edge",
        "proxy_features",
    ]
    bounded_pass = fit_holdout[fit_holdout["holdout_bounded_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    sign_pass = fit_holdout[fit_holdout["holdout_sign_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    best = fit_holdout.sort_values("exec_risk_score", ascending=False) if not fit_holdout.empty else fit_holdout
    comp = (
        components_summary[
            components_summary["period"].astype(str).isin(manifest["fit_months"] + [manifest["holdout_month"]])
        ].sort_values(["period", "component"])
        if not components_summary.empty
        else components_summary
    )
    lines = [
        "# First-Touch Abstention Proxy Grid",
        "",
        "Scope: proxy tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Proxy method: `{manifest['proxy_method']}`",
        f"Selection mode: `timestamp`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        "",
        "Each row fits prior-month proxies for utility, clean first-touch execution, fast edge, early adverse risk, slow/timeout risk, and dirty execution. The grid first abstains on risk proxies, then ranks the accepted rows inside each timestamp.",
        "Candidate coverage columns measure whether the abstention rule leaves enough contemporaneous symbols to form a realistic portfolio choice.",
        "",
        "## Counts",
        "",
        f"- Specs tested: `{manifest['spec_count']}`",
        f"- Monthly rows: `{manifest['rows_monthly']}`",
        f"- Weekly rows: `{manifest['rows_weekly']}`",
        f"- Fit bounded pass: `{manifest['fit_bounded_pass_rows']}`",
        f"- Holdout bounded pass after fit selection: `{manifest['holdout_bounded_pass_rows']}`",
        f"- Positive but economically dirty holdout: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Bounded Holdout Passes",
        "",
        _format_table(bounded_pass, cols, limit=60),
        "",
        "## Positive Holdout Sign Rows",
        "",
        _format_table(sign_pass, cols, limit=60),
        "",
        "## Best Rejected Or Failed Rows",
        "",
        _format_table(best, cols, limit=60),
        "",
        "## Component Proxy ICs",
        "",
        _format_table(comp, component_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Component IC: `{manifest['outputs']['component_proxy']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    proxy_top_k: int,
    proxy_method: str,
    proxy_tail_frac: float,
    score_names: list[str],
    dirty_max: list[float],
    early_max: list[float],
    slow_max: list[float],
    clean_min: list[float | None],
    fast_min: list[float | None],
    max_specs: int | None,
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(
        score_names=score_names,
        dirty_max=dirty_max,
        early_max=early_max,
        slow_max=slow_max,
        clean_min=clean_min,
        fast_min=fast_min,
        max_specs=max_specs,
    )
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                feature_matrix.loc[:, new_cols].reset_index(drop=True),
            ],
            axis=1,
        )
    metrics = _first_touch_metrics(frame, _path_metrics(frame))
    components = _target_components(metrics)
    features = _feature_columns(frame)

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_series < str(month)
        valid_mask = month_series == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy()
        valid_components = {name: series.loc[valid_mask] for name, series in components.items()}
        proxies: dict[str, pd.Series] = {}
        diag: dict[str, dict[str, Any]] = {}
        for name in ("utility", "clean_first_touch", "fast_edge", "early_adverse", "slow_timeout", "dirty"):
            score, score_diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=components[name].loc[train_mask],
                top_k=proxy_top_k,
                method=str(proxy_method),
                tail_frac=float(proxy_tail_frac),
            )
            proxies[name] = score
            diag[name] = score_diag
            component_rows.append(
                {
                    "period": str(month),
                    "component": name,
                    "proxy_top_abs_ic": score_diag.get("top_abs_ic"),
                    "proxy_mean_top_abs_ic": score_diag.get("mean_top_abs_ic"),
                    "proxy_ic_actual_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "proxy_ic_actual_clean_exec": _spearman(score, valid_metrics["clean_exec_actual"]),
                    "proxy_ic_actual_dirty": _spearman(score, valid_components["dirty"]),
                    "proxy_ic_actual_fast_edge": _spearman(score, valid_components["fast_edge"]),
                    "proxy_features": ",".join(score_diag.get("features", [])),
                }
            )

        valid_target = _target_for_selection(valid_components, valid.index)
        for spec in specs:
            score = _score_from_spec(spec, proxies)
            m_rows, w_rows = _monthly_weekly_rows(
                valid_frame=valid,
                valid_metrics=valid_metrics,
                valid_target=valid_target,
                score=score,
                score_arm=spec.name,
                month=str(month),
                top_fracs=top_fracs,
                component_diag=diag,
            )
            for row in m_rows + w_rows:
                row["score_name"] = spec.score_name
                row["dirty_max"] = spec.dirty_max
                row["early_max"] = spec.early_max
                row["slow_max"] = spec.slow_max
                row["clean_min"] = spec.clean_min
                row["fast_min"] = spec.fast_min
            monthly_rows.extend(m_rows)
            weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    components_summary = pd.DataFrame(component_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )
    fit_holdout = _augment_fit_holdout(
        fit_holdout=fit_holdout,
        monthly=monthly,
        fit_months=fit_months,
        holdout_month=holdout_month,
    )

    paths = {
        "monthly": output_dir / "first_touch_abstention_proxy_monthly.csv",
        "weekly": output_dir / "first_touch_abstention_proxy_weekly.csv",
        "component_proxy": output_dir / "first_touch_abstention_component_ic.csv",
        "fit_holdout": output_dir / "first_touch_abstention_proxy_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    components_summary.to_csv(paths["component_proxy"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "features": features,
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_method": str(proxy_method),
        "proxy_tail_frac": float(proxy_tail_frac),
        "selection_mode": "timestamp",
        "score_names": [str(v) for v in score_names],
        "dirty_max": [float(v) for v in dirty_max],
        "early_max": [float(v) for v in early_max],
        "slow_max": [float(v) for v in slow_max],
        "clean_min": [None if v is None else float(v) for v in clean_min],
        "fast_min": [None if v is None else float(v) for v in fast_min],
        "spec_count": int(len(specs)),
        "score_specs": [spec.__dict__ for spec in specs],
        "top_fracs": [float(v) for v in top_fracs],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_clean_pass_rows": int(fit_holdout["fit_clean_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_clean_pass_rows": int(fit_holdout["holdout_clean_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, fit_holdout, components_summary, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--proxy-top-k", type=int, default=12)
    parser.add_argument("--proxy-method", choices=["ic", "tail_lift"], default="ic")
    parser.add_argument("--proxy-tail-frac", type=float, default=0.05)
    parser.add_argument("--score-names", default=",".join(DEFAULT_SCORE_NAMES))
    parser.add_argument("--dirty-max", default=",".join(str(v) for v in DEFAULT_DIRTY_MAX))
    parser.add_argument("--early-max", default=",".join(str(v) for v in DEFAULT_EARLY_MAX))
    parser.add_argument("--slow-max", default=",".join(str(v) for v in DEFAULT_SLOW_MAX))
    parser.add_argument("--clean-min", default=",".join(DEFAULT_CLEAN_MIN))
    parser.add_argument("--fast-min", default=",".join(DEFAULT_FAST_MIN))
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        proxy_top_k=int(args.proxy_top_k),
        proxy_method=str(args.proxy_method),
        proxy_tail_frac=float(args.proxy_tail_frac),
        score_names=_parse_csv(args.score_names),
        dirty_max=_parse_float_csv(args.dirty_max),
        early_max=_parse_float_csv(args.early_max),
        slow_max=_parse_float_csv(args.slow_max),
        clean_min=_parse_optional_float_csv(args.clean_min),
        fast_min=_parse_optional_float_csv(args.fast_min),
        max_specs=args.max_specs,
        top_fracs=_parse_float_csv(args.top_fracs),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
    )
    summary_keys = [
        "output_dir",
        "rows",
        "feature_count",
        "proxy_method",
        "selection_mode",
        "spec_count",
        "rows_monthly",
        "rows_weekly",
        "fit_clean_pass_rows",
        "holdout_clean_pass_rows",
        "fit_bounded_pass_rows",
        "holdout_bounded_pass_rows",
        "positive_dirty_holdout_rows",
        "outputs",
    ]
    print(json.dumps(_json_safe({key: manifest.get(key) for key in summary_keys}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
