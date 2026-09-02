#!/usr/bin/env python3
"""Diagnose path-target learnability and feature stability before model training.

This is a proxy-only diagnostic. It asks whether causal features that appear to
rank bounded/profit-low-MAE paths in the fit window keep the same direction in
the next month, and whether path-target prevalence is stable by state bucket.
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

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
    _feature_family,
    _mfe_mae,
    _table,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_proxy,
    _state_path_risk_targets,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _rank_top_indices,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/path_target_feature_stability_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_PATH_TARGETS = ("bounded", "profit_low_mae", "path_quality", "bad_mae", "dirty")
DEFAULT_LABEL_ARMS = ("S61_tpnet_strict_adverse_veto_rank", "S65_profit_inside_exec_admissible")
DEFAULT_TOP_FRACS = (0.03, 0.05)
DEFAULT_SPLIT_EMBARGO_HOURS = 24.0


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _auc_high(values: pd.Series, target: pd.Series) -> float:
    x = _safe_numeric(values)
    y = _safe_numeric(target)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 20:
        return float("nan")
    labels = y[mask] > 0.5
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos <= 5 or n_neg <= 5:
        return float("nan")
    ranks = x[mask].rank(method="average")
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)


def _target_frame(target: pd.Series) -> pd.DataFrame:
    y = _safe_numeric(target).reset_index(drop=True)
    return pd.DataFrame({"target_soft": y, "target_hard": (y > 0.5).astype(float)})


def _target_specs(
    *,
    metrics: pd.DataFrame,
    frame: pd.DataFrame,
    path_targets: list[str],
    label_arms: list[str],
) -> dict[str, dict[str, Any]]:
    path = _state_path_risk_targets(metrics)
    specs: dict[str, dict[str, Any]] = {}
    for name in path_targets:
        if name not in path:
            raise ValueError(f"Unknown path target: {name}")
        specs[f"path::{name}"] = {
            "target": path[name],
            "kind": "continuous" if name == "path_quality" else "binary",
            "desired_high": name not in {"bad_mae", "dirty"},
            "source": "path",
        }
    labels = _label_targets(frame, metrics)
    for arm in label_arms:
        if arm not in labels:
            raise ValueError(f"Unknown label arm: {arm}")
        specs[f"label::{arm}"] = {
            "target": labels[arm]["target_soft"],
            "kind": "continuous",
            "desired_high": True,
            "source": "label",
        }
    return specs


def _month_masks(
    frame: pd.DataFrame,
    month: str,
    *,
    split_embargo_hours: float,
) -> tuple[pd.Series, pd.Series]:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    month_start = pd.Period(month, freq="M").start_time.tz_localize("UTC")
    embargo = pd.Timedelta(hours=float(split_embargo_hours))
    train_mask = ts < (month_start - embargo)
    valid_mask = ts.dt.to_period("M").astype(str).eq(str(month))
    return train_mask.fillna(False), valid_mask.fillna(False)


def _select_features_for_target(
    *,
    train: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    proxy_features: list[str],
    max_features: int | None,
) -> list[str]:
    if max_features is None or int(max_features) <= 0 or int(max_features) >= len(features):
        return list(features)
    proxy_set = set(proxy_features)
    priority = {
        "state_path_prior",
        "event_confirmation",
        "adverse_path_composite",
        "outcome_prior",
        "barrier_distance",
        "liquidity_spread",
        "open_interest",
        "pullback_location",
        "trend_quality",
        "exhaustion_reversal",
    }
    rows: list[tuple[float, str]] = []
    for feature in features:
        ic = _spearman(train[feature], target_train)
        value = abs(float(ic)) if math.isfinite(ic) else 0.0
        if _feature_family(feature) in priority:
            value += 0.01
        if feature in proxy_set:
            value += 1.0
        rows.append((value, feature))
    selected: list[str] = []
    for feature in proxy_features:
        if feature in features and feature not in selected:
            selected.append(feature)
    for _, feature in sorted(rows, key=lambda item: item[0], reverse=True):
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= int(max_features):
            break
    return selected


def _feature_rows(
    *,
    month: str,
    target_name: str,
    target_kind: str,
    desired_high: bool,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    y_valid: pd.Series,
    proxy_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        fit_ic = _spearman(train[feature], y_train)
        valid_ic = _spearman(valid[feature], y_valid)
        fit_auc = _auc_high(train[feature], y_train) if target_kind == "binary" else float("nan")
        valid_auc = _auc_high(valid[feature], y_valid) if target_kind == "binary" else float("nan")
        if math.isfinite(fit_auc):
            fit_direction = "high" if fit_auc >= 0.5 else "low"
            fit_best_auc = max(float(fit_auc), 1.0 - float(fit_auc))
            valid_auc_fit_direction = (
                float(valid_auc)
                if fit_direction == "high" and math.isfinite(valid_auc)
                else (1.0 - float(valid_auc) if math.isfinite(valid_auc) else float("nan"))
            )
        else:
            fit_direction = "high" if math.isfinite(fit_ic) and fit_ic >= 0.0 else "low"
            fit_best_auc = float("nan")
            valid_auc_fit_direction = float("nan")
        fit_sign = 1 if math.isfinite(fit_ic) and fit_ic >= 0.0 else -1
        valid_sign = 1 if math.isfinite(valid_ic) and valid_ic >= 0.0 else -1
        rows.append(
            {
                "month": month,
                "target": target_name,
                "target_kind": target_kind,
                "desired_high": bool(desired_high),
                "feature": feature,
                "feature_family": _feature_family(feature),
                "is_proxy_feature": feature in set(proxy_features),
                "fit_ic": fit_ic,
                "valid_ic": valid_ic,
                "abs_fit_ic": abs(float(fit_ic)) if math.isfinite(fit_ic) else float("nan"),
                "abs_valid_ic": abs(float(valid_ic)) if math.isfinite(valid_ic) else float("nan"),
                "ic_sign_consistent": bool(
                    math.isfinite(fit_ic) and math.isfinite(valid_ic) and fit_sign == valid_sign
                ),
                "fit_auc_high": fit_auc,
                "valid_auc_high": valid_auc,
                "fit_best_auc": fit_best_auc,
                "fit_auc_direction": fit_direction,
                "valid_auc_fit_direction": valid_auc_fit_direction,
                "auc_direction_consistent": bool(
                    math.isfinite(valid_auc_fit_direction) and float(valid_auc_fit_direction) >= 0.5
                ),
                "valid_finite_frac": float(_safe_numeric(valid[feature]).notna().mean()),
            }
        )
    return rows


def _proxy_selection_rows(
    *,
    month: str,
    target_name: str,
    desired_high: bool,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    y_valid: pd.Series,
    score: pd.Series,
    top_fracs: list[float],
    proxy_features: list[str],
) -> list[dict[str, Any]]:
    selection_score = score if desired_high else -score
    rows: list[dict[str, Any]] = []
    target_valid = _target_frame(y_valid)
    mfe_mae = _mfe_mae(valid_metrics)
    for top_frac in top_fracs:
        row = _selection_metrics(
            frame=valid,
            metrics=valid_metrics,
            target=target_valid,
            score=selection_score.reset_index(drop=True),
            arm=f"path_feature_stability::{target_name}",
            selector="causal_feature_ic_proxy",
            period=month,
            top_frac=float(top_frac),
        )
        idx = _rank_top_indices(selection_score.reset_index(drop=True), float(top_frac))
        selected_metrics = valid_metrics.iloc[idx] if len(idx) else valid_metrics.iloc[:0]
        selected_target = y_valid.reset_index(drop=True).iloc[idx] if len(idx) else y_valid.iloc[:0]
        selected_mfe_mae = mfe_mae.reset_index(drop=True).iloc[idx] if len(idx) else mfe_mae.iloc[:0]
        row.update(
            {
                "month": month,
                "target": target_name,
                "desired_high": bool(desired_high),
                "top_frac": float(top_frac),
                "selected_target_mean": _safe_mean(selected_target),
                "selected_bounded_rate": _safe_mean(
                    (
                        (selected_metrics["u_policy_net"] > 0.0)
                        & (selected_metrics["mae_norm"] <= 1.0)
                        & (selected_metrics["barrier"] <= 0.025)
                        & (selected_mfe_mae >= 1.25)
                        & (~selected_metrics["is_timeout"].astype(bool))
                    ).astype(float)
                ),
                "proxy_features": ",".join(proxy_features),
            }
        )
        rows.append(row)
    return rows


def _bucket_prevalence_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    valid_mask: pd.Series,
    month: str,
    state_features: list[str],
    targets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    mfe_mae = _mfe_mae(valid_metrics)
    for feature in state_features:
        if feature not in valid.columns:
            continue
        values = _safe_numeric(valid[feature])
        ranks = values.groupby(valid["__ts__"], dropna=False).rank(method="average", pct=True)
        bucket = pd.Series("missing", index=valid.index, dtype=object)
        bucket[(ranks > 0.0) & (ranks <= (1.0 / 3.0))] = "low"
        bucket[(ranks > (1.0 / 3.0)) & (ranks <= (2.0 / 3.0))] = "mid"
        bucket[ranks > (2.0 / 3.0)] = "high"
        for bucket_name, idx in bucket.groupby(bucket, dropna=False).groups.items():
            idx_list = list(idx)
            row = {
                "month": month,
                "state_feature": feature,
                "bucket": bucket_name,
                "rows": int(len(idx_list)),
                "mean_return_net": _safe_mean(valid_metrics.iloc[idx_list]["ret_net"]),
                "bad_mae_1r_rate": _safe_mean(valid_metrics.iloc[idx_list]["mae_norm"] >= 1.0),
                "p90_mae_norm": _safe_quantile(valid_metrics.iloc[idx_list]["mae_norm"], 0.90),
                "timeout_rate": _safe_mean(valid_metrics.iloc[idx_list]["is_timeout"].astype(float)),
                "mfe_mae_mean": _safe_mean(mfe_mae.iloc[idx_list]),
            }
            for target_name, spec in targets.items():
                if spec["source"] != "path":
                    continue
                y = _safe_numeric(spec["target"]).loc[valid_mask].reset_index(drop=True)
                row[f"{target_name.replace('path::', '')}_mean"] = _safe_mean(y.iloc[idx_list])
            rows.append(row)
    return rows


def _family_summary(feature_stats: pd.DataFrame) -> pd.DataFrame:
    if feature_stats.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (target, family), group in feature_stats.groupby(["target", "feature_family"], dropna=False, observed=True):
        top = (
            group.sort_values(["valid_auc_fit_direction", "abs_valid_ic"], ascending=[False, False])["feature"]
            .drop_duplicates()
            .head(8)
            .tolist()
        )
        rows.append(
            {
                "target": target,
                "feature_family": family,
                "rows": int(len(group)),
                "months": int(group["month"].nunique()),
                "proxy_feature_share": _safe_mean(group["is_proxy_feature"].astype(float)),
                "mean_abs_fit_ic": _safe_mean(group["abs_fit_ic"]),
                "mean_abs_valid_ic": _safe_mean(group["abs_valid_ic"]),
                "ic_sign_consistency": _safe_mean(group["ic_sign_consistent"].astype(float)),
                "mean_fit_best_auc": _safe_mean(group["fit_best_auc"]),
                "mean_valid_auc_fit_direction": _safe_mean(group["valid_auc_fit_direction"]),
                "auc_direction_consistency": _safe_mean(group["auc_direction_consistent"].astype(float)),
                "top_features": ",".join(top),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "mean_valid_auc_fit_direction", "mean_abs_valid_ic"],
        ascending=[True, False, False],
    )


def _feature_drift_summary(feature_stats: pd.DataFrame) -> pd.DataFrame:
    if feature_stats.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (target, feature), group in feature_stats.groupby(["target", "feature"], dropna=False, observed=True):
        rows.append(
            {
                "target": target,
                "feature": feature,
                "feature_family": group["feature_family"].iloc[0],
                "months": int(group["month"].nunique()),
                "is_proxy_feature_any": bool(group["is_proxy_feature"].any()),
                "mean_abs_fit_ic": _safe_mean(group["abs_fit_ic"]),
                "mean_abs_valid_ic": _safe_mean(group["abs_valid_ic"]),
                "ic_sign_consistency": _safe_mean(group["ic_sign_consistent"].astype(float)),
                "mean_fit_best_auc": _safe_mean(group["fit_best_auc"]),
                "mean_valid_auc_fit_direction": _safe_mean(group["valid_auc_fit_direction"]),
                "min_valid_auc_fit_direction": _safe_quantile(group["valid_auc_fit_direction"], 0.0),
                "auc_direction_consistency": _safe_mean(group["auc_direction_consistent"].astype(float)),
                "mean_valid_finite_frac": _safe_mean(group["valid_finite_frac"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "mean_valid_auc_fit_direction", "mean_abs_valid_ic"],
        ascending=[True, False, False],
    )


def _bucket_drift_summary(bucket_prevalence: pd.DataFrame) -> pd.DataFrame:
    if bucket_prevalence.empty:
        return pd.DataFrame()
    target_cols = [col for col in bucket_prevalence.columns if col.endswith("_mean")]
    rows: list[dict[str, Any]] = []
    for (feature, bucket), group in bucket_prevalence.groupby(["state_feature", "bucket"], dropna=False, observed=True):
        row = {
            "state_feature": feature,
            "bucket": bucket,
            "months": int(group["month"].nunique()),
            "mean_rows": _safe_mean(group["rows"]),
            "mean_return_net": _safe_mean(group["mean_return_net"]),
            "return_net_range": float(group["mean_return_net"].max() - group["mean_return_net"].min()),
            "bad_mae_rate_mean": _safe_mean(group["bad_mae_1r_rate"]),
            "bad_mae_rate_range": float(group["bad_mae_1r_rate"].max() - group["bad_mae_1r_rate"].min()),
            "timeout_rate_mean": _safe_mean(group["timeout_rate"]),
        }
        for col in target_cols:
            row[f"{col}_range"] = float(group[col].max() - group[col].min())
            row[f"{col}_mean"] = _safe_mean(group[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["bad_mae_rate_range", "return_net_range"], ascending=[False, False])


def _write_markdown(
    *,
    output_dir: Path,
    target_summary: pd.DataFrame,
    proxy_selection: pd.DataFrame,
    family_summary: pd.DataFrame,
    drift_summary: pd.DataFrame,
    bucket_drift: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "path_target_feature_stability.md"
    target_cols = [
        "month",
        "target",
        "target_source",
        "target_kind",
        "train_rows",
        "valid_rows",
        "train_target_mean",
        "valid_target_mean",
        "proxy_top_abs_ic",
        "proxy_mean_top_abs_ic",
        "proxy_features",
    ]
    proxy_cols = [
        "month",
        "target",
        "top_frac",
        "selected_rows",
        "selected_target_mean",
        "mean_return_net",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "selected_bounded_rate",
        "proxy_features",
    ]
    family_cols = [
        "target",
        "feature_family",
        "rows",
        "months",
        "proxy_feature_share",
        "mean_abs_fit_ic",
        "mean_abs_valid_ic",
        "ic_sign_consistency",
        "mean_fit_best_auc",
        "mean_valid_auc_fit_direction",
        "auc_direction_consistency",
        "top_features",
    ]
    drift_cols = [
        "target",
        "feature",
        "feature_family",
        "is_proxy_feature_any",
        "mean_abs_fit_ic",
        "mean_abs_valid_ic",
        "ic_sign_consistency",
        "mean_fit_best_auc",
        "mean_valid_auc_fit_direction",
        "min_valid_auc_fit_direction",
        "auc_direction_consistency",
    ]
    bucket_cols = [
        "state_feature",
        "bucket",
        "months",
        "mean_return_net",
        "return_net_range",
        "bad_mae_rate_mean",
        "bad_mae_rate_range",
        "timeout_rate_mean",
        "bounded_mean_mean",
        "bounded_mean_range",
        "profit_low_mae_mean_mean",
        "profit_low_mae_mean_range",
    ]
    lines = [
        "# Path Target Feature Stability",
        "",
        "Scope: no model training. Proxies are fit on rows before each OOS month with a split embargo, then feature direction and path-target prevalence are evaluated in the OOS month.",
        "",
        f"Labels path: `{manifest['labels_path']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Split embargo hours: `{manifest['split_embargo_hours']}`",
        f"Path targets: `{', '.join(manifest['path_targets'])}`",
        f"Label arms: `{', '.join(manifest['label_arms'])}`",
        f"Feature count: `{manifest['feature_count']}`",
        "",
        "## Target And Proxy Summary",
        "",
        _table(target_summary, target_cols, limit=80),
        "",
        "## Proxy Selection Quality",
        "",
        _table(
            proxy_selection.sort_values(["target", "month", "top_frac"]) if not proxy_selection.empty else proxy_selection,
            proxy_cols,
            limit=120,
        ),
        "",
        "## Feature Family Stability",
        "",
        _table(family_summary, family_cols, limit=80),
        "",
        "## Best Feature Drift",
        "",
        _table(
            drift_summary.sort_values(["target", "mean_valid_auc_fit_direction", "mean_abs_valid_ic"], ascending=[True, False, False])
            if not drift_summary.empty
            else drift_summary,
            drift_cols,
            limit=120,
        ),
        "",
        "## State Bucket Drift",
        "",
        _table(bucket_drift, bucket_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Target summary: `{manifest['outputs']['target_summary']}`",
        f"- Feature stats: `{manifest['outputs']['feature_stats']}`",
        f"- Proxy selection: `{manifest['outputs']['proxy_selection']}`",
        f"- Feature family summary: `{manifest['outputs']['family_summary']}`",
        f"- Feature drift summary: `{manifest['outputs']['feature_drift_summary']}`",
        f"- Bucket prevalence: `{manifest['outputs']['bucket_prevalence']}`",
        f"- Bucket drift summary: `{manifest['outputs']['bucket_drift_summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    path_targets: list[str],
    label_arms: list[str],
    months: list[str],
    top_fracs: list[float],
    split_embargo_hours: float,
    proxy_top_k: int,
    max_rank_features: int | None,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    features = _feature_columns(frame)
    targets = _target_specs(metrics=metrics, frame=frame, path_targets=path_targets, label_arms=label_arms)

    target_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []

    for month in months:
        train_mask, valid_mask = _month_masks(frame, month, split_embargo_hours=split_embargo_hours)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid_source = frame.loc[valid_mask].copy()
        valid = valid_source.reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)

        bucket_rows.extend(
            _bucket_prevalence_rows(
                frame=frame,
                metrics=metrics,
                valid_mask=valid_mask,
                month=month,
                state_features=state_path_prior_features,
                targets=targets,
            )
        )

        for target_name, spec in targets.items():
            y = _safe_numeric(spec["target"])
            y_train = y.loc[train_mask]
            y_valid = y.loc[valid_mask].reset_index(drop=True)
            score, diag = _score_proxy(
                train=train,
                valid=valid_source,
                features=features,
                y_train=y_train,
                proxy_top_k=proxy_top_k,
            )
            score = score.reset_index(drop=True)
            proxy_features = [str(feature) for feature in diag.get("proxy_features", [])]
            target_rows.append(
                {
                    "month": month,
                    "target": target_name,
                    "target_source": spec["source"],
                    "target_kind": spec["kind"],
                    "desired_high": bool(spec["desired_high"]),
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "train_target_mean": _safe_mean(y_train),
                    "valid_target_mean": _safe_mean(y_valid),
                    "train_positive_rows": int((_safe_numeric(y_train) > 0.5).sum()),
                    "valid_positive_rows": int((_safe_numeric(y_valid) > 0.5).sum()),
                    "proxy_top_abs_ic": diag.get("proxy_top_abs_ic", float("nan")),
                    "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic", float("nan")),
                    "proxy_features": ",".join(proxy_features),
                }
            )
            ranked_features = _select_features_for_target(
                train=train,
                features=features,
                target_train=y_train,
                proxy_features=proxy_features,
                max_features=max_rank_features,
            )
            feature_rows.extend(
                _feature_rows(
                    month=month,
                    target_name=target_name,
                    target_kind=str(spec["kind"]),
                    desired_high=bool(spec["desired_high"]),
                    train=train,
                    valid=valid,
                    features=ranked_features,
                    y_train=y_train,
                    y_valid=y_valid,
                    proxy_features=proxy_features,
                )
            )
            proxy_rows.extend(
                _proxy_selection_rows(
                    month=month,
                    target_name=target_name,
                    desired_high=bool(spec["desired_high"]),
                    valid=valid,
                    valid_metrics=valid_metrics,
                    y_valid=y_valid,
                    score=score,
                    top_fracs=top_fracs,
                    proxy_features=proxy_features,
                )
            )

    target_summary = pd.DataFrame(target_rows)
    feature_stats = pd.DataFrame(feature_rows)
    proxy_selection = pd.DataFrame(proxy_rows)
    bucket_prevalence = pd.DataFrame(bucket_rows)
    family_summary = _family_summary(feature_stats)
    drift_summary = _feature_drift_summary(feature_stats)
    bucket_drift = _bucket_drift_summary(bucket_prevalence)

    paths = {
        "target_summary": output_dir / "path_target_feature_stability_target_summary.csv",
        "feature_stats": output_dir / "path_target_feature_stability_feature_stats.csv",
        "proxy_selection": output_dir / "path_target_feature_stability_proxy_selection.csv",
        "family_summary": output_dir / "path_target_feature_stability_family_summary.csv",
        "feature_drift_summary": output_dir / "path_target_feature_stability_feature_drift_summary.csv",
        "bucket_prevalence": output_dir / "path_target_feature_stability_bucket_prevalence.csv",
        "bucket_drift_summary": output_dir / "path_target_feature_stability_bucket_drift_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    target_summary.to_csv(paths["target_summary"], index=False)
    feature_stats.to_csv(paths["feature_stats"], index=False)
    proxy_selection.to_csv(paths["proxy_selection"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)
    drift_summary.to_csv(paths["feature_drift_summary"], index=False)
    bucket_prevalence.to_csv(paths["bucket_prevalence"], index=False)
    bucket_drift.to_csv(paths["bucket_drift_summary"], index=False)

    manifest = {
        "scope": "path_target_feature_stability_proxy_only",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "path_targets": list(path_targets),
        "label_arms": list(label_arms),
        "months": list(months),
        "top_fracs": [float(value) for value in top_fracs],
        "split_embargo_hours": float(split_embargo_hours),
        "proxy_top_k": int(proxy_top_k),
        "max_rank_features": None if max_rank_features is None else int(max_rank_features),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(value) for value in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "feature_count": int(len(features)),
        "reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        target_summary=target_summary,
        proxy_selection=proxy_selection,
        family_summary=family_summary,
        drift_summary=drift_summary,
        bucket_drift=bucket_drift,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "path_target_feature_stability.md")}},
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--path-targets", type=lambda value: _parse_csv(value, DEFAULT_PATH_TARGETS), default=list(DEFAULT_PATH_TARGETS))
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=list(DEFAULT_LABEL_ARMS))
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--split-embargo-hours", type=float, default=DEFAULT_SPLIT_EMBARGO_HOURS)
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--max-rank-features", type=int, default=500)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        default=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        default=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        path_targets=list(args.path_targets),
        label_arms=list(args.label_arms),
        months=list(args.months),
        top_fracs=list(args.top_fracs),
        split_embargo_hours=float(args.split_embargo_hours),
        proxy_top_k=int(args.proxy_top_k),
        max_rank_features=None if args.max_rank_features is None else int(args.max_rank_features),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe({key: value for key, value in manifest.items() if key != "reports"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
