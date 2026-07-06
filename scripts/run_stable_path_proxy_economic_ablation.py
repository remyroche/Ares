#!/usr/bin/env python3
"""Run a proxy-only economic ablation using stability-selected path features.

This tests the Stage 92 finding that many path-stable features are not selected
by the default one-shot IC proxy. No LightGBM, Optuna, or policy optimisation is
run here.
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
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    DEFAULT_TOP_FRACS,
    _aggregate,
    _fit_holdout_summary,
    _score_period,
    _slice_week_positions,
    _state_path_feature_columns,
    _state_path_risk_targets,
    _table,
    _top_gate,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stable_path_proxy_economic_ablation_v1")
DEFAULT_LABEL_ARMS = (
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
    "S65_profit_inside_exec_admissible",
)
DEFAULT_STABLE_PATH_TARGETS = ("bounded", "profit_low_mae", "path_quality")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_GATE_FRACS = (0.10, 0.20, 0.30)


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


def _rank_score_from_features(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    binary_target: bool,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    parts: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for feature in features:
        if binary_target:
            auc = _auc_high(train[feature], y_train)
            if math.isfinite(auc):
                direction = "high" if auc >= 0.5 else "low"
                strength = max(float(auc), 1.0 - float(auc))
            else:
                ic = _spearman(train[feature], y_train)
                direction = "high" if math.isfinite(ic) and ic >= 0.0 else "low"
                strength = abs(float(ic)) if math.isfinite(ic) else float("nan")
        else:
            ic = _spearman(train[feature], y_train)
            direction = "high" if math.isfinite(ic) and ic >= 0.0 else "low"
            strength = abs(float(ic)) if math.isfinite(ic) else float("nan")
        ranks = _safe_numeric(valid[feature]).rank(method="average", pct=True)
        if direction == "low":
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.5))
        rows.append({"feature": feature, "direction": direction, "fit_strength": strength})
    score = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)
    return score.reindex(valid.index), rows


def _stable_feature_stats(
    *,
    train: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    binary_target: bool,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
) -> pd.DataFrame:
    train_months = train["__ts__"].dt.to_period("M").astype(str)
    months = sorted(train_months.dropna().unique())
    rows: list[dict[str, Any]] = []
    for feature in features:
        fold_rows: list[dict[str, float]] = []
        for month in months[1:]:
            inner_train_mask = train_months.lt(str(month))
            inner_valid_mask = train_months.eq(str(month))
            if int(inner_train_mask.sum()) < int(min_inner_train_rows) or int(inner_valid_mask.sum()) < int(min_inner_valid_rows):
                continue
            fit_ic = _spearman(train.loc[inner_train_mask, feature], y_train.loc[inner_train_mask])
            valid_ic = _spearman(train.loc[inner_valid_mask, feature], y_train.loc[inner_valid_mask])
            if binary_target:
                fit_auc = _auc_high(train.loc[inner_train_mask, feature], y_train.loc[inner_train_mask])
                valid_auc = _auc_high(train.loc[inner_valid_mask, feature], y_train.loc[inner_valid_mask])
                if math.isfinite(fit_auc) and math.isfinite(valid_auc):
                    direction = 1.0 if fit_auc >= 0.5 else -1.0
                    fit_strength = max(float(fit_auc), 1.0 - float(fit_auc))
                    valid_aligned = float(valid_auc) if direction > 0 else 1.0 - float(valid_auc)
                else:
                    direction = 1.0 if math.isfinite(fit_ic) and fit_ic >= 0.0 else -1.0
                    fit_strength = abs(float(fit_ic)) if math.isfinite(fit_ic) else float("nan")
                    valid_aligned = (
                        direction * float(valid_ic)
                        if math.isfinite(valid_ic)
                        else float("nan")
                    )
            else:
                direction = 1.0 if math.isfinite(fit_ic) and fit_ic >= 0.0 else -1.0
                fit_strength = abs(float(fit_ic)) if math.isfinite(fit_ic) else float("nan")
                valid_aligned = direction * float(valid_ic) if math.isfinite(valid_ic) else float("nan")
            fold_rows.append(
                {
                    "fit_ic": fit_ic,
                    "valid_ic": valid_ic,
                    "fit_strength": fit_strength,
                    "valid_aligned": valid_aligned,
                    "consistent": float(valid_aligned >= (0.5 if binary_target else 0.0))
                    if math.isfinite(valid_aligned)
                    else float("nan"),
                }
            )
        if not fold_rows:
            continue
        fold_frame = pd.DataFrame(fold_rows)
        rows.append(
            {
                "feature": feature,
                "inner_folds": int(len(fold_frame)),
                "mean_fit_strength": _safe_mean(fold_frame["fit_strength"]),
                "mean_valid_aligned": _safe_mean(fold_frame["valid_aligned"]),
                "min_valid_aligned": _safe_quantile(fold_frame["valid_aligned"], 0.0),
                "consistency": _safe_mean(fold_frame["consistent"]),
            }
        )
    return pd.DataFrame(rows)


def _stable_proxy_score(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    binary_target: bool,
    proxy_top_k: int,
    min_inner_folds: int,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    min_consistency: float,
) -> tuple[pd.Series, dict[str, Any]]:
    stats = _stable_feature_stats(
        train=train,
        features=features,
        y_train=y_train,
        binary_target=binary_target,
        min_inner_train_rows=min_inner_train_rows,
        min_inner_valid_rows=min_inner_valid_rows,
    )
    if not stats.empty:
        stable = stats[
            stats["inner_folds"].ge(int(min_inner_folds))
            & stats["consistency"].ge(float(min_consistency))
        ].copy()
    else:
        stable = pd.DataFrame()
    if not stable.empty:
        chosen = stable.sort_values(
            ["mean_valid_aligned", "min_valid_aligned", "mean_fit_strength"],
            ascending=[False, False, False],
        ).head(int(proxy_top_k))
        chosen_features = chosen["feature"].astype(str).tolist()
        selection_mode = "inner_stability"
    else:
        fallback_rows = []
        for feature in features:
            ic = _spearman(train[feature], y_train)
            fallback_rows.append(
                {
                    "feature": feature,
                    "mean_fit_strength": abs(float(ic)) if math.isfinite(ic) else 0.0,
                    "mean_valid_aligned": float("nan"),
                    "min_valid_aligned": float("nan"),
                    "consistency": float("nan"),
                    "inner_folds": 0,
                }
            )
        chosen = pd.DataFrame(fallback_rows).sort_values("mean_fit_strength", ascending=False).head(int(proxy_top_k))
        chosen_features = chosen["feature"].astype(str).tolist()
        selection_mode = "fallback_fit_ic"
    score, direction_rows = _rank_score_from_features(
        train=train,
        valid=valid,
        features=chosen_features,
        y_train=y_train,
        binary_target=binary_target,
    )
    direction_map = {row["feature"]: row["direction"] for row in direction_rows}
    diag = {
        "proxy_features": chosen_features,
        "selection_mode": selection_mode,
        "stable_feature_rows": int(len(stable)),
        "candidate_feature_rows": int(len(stats)),
        "proxy_feature_directions": ",".join(f"{feature}:{direction_map.get(feature, '?')}" for feature in chosen_features),
        "stable_feature_stats": chosen.to_dict(orient="records"),
    }
    return score, diag


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    path_targets: dict[str, pd.Series],
    features: list[str],
    path_features: list[str],
    month: str,
    label_arms: list[str],
    stable_path_targets: list[str],
    top_fracs: list[float],
    combine_label_weight: float,
    gate_fracs: list[float],
    proxy_top_k: int,
    min_inner_folds: int,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    min_consistency: float,
) -> list[dict[str, Any]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period.lt(str(month))
    valid_mask = month_period.eq(str(month))
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return []
    train = frame.loc[train_mask].copy()
    valid_source = frame.loc[valid_mask].copy()
    valid = valid_source.reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    valid_indices = np.arange(len(valid), dtype=np.int64)

    label_scores: dict[str, pd.Series] = {}
    label_features: dict[str, str] = {}
    for label_arm in label_arms:
        score, diag = _stable_proxy_score(
            train=train,
            valid=valid_source,
            features=features,
            y_train=targets[label_arm].loc[train_mask, "target_soft"],
            binary_target=False,
            proxy_top_k=proxy_top_k,
            min_inner_folds=min_inner_folds,
            min_inner_train_rows=min_inner_train_rows,
            min_inner_valid_rows=min_inner_valid_rows,
            min_consistency=min_consistency,
        )
        label_scores[label_arm] = score.reset_index(drop=True)
        label_features[label_arm] = (
            f"{diag['selection_mode']}|{','.join(diag.get('proxy_features', []))}|{diag.get('proxy_feature_directions', '')}"
        )

    path_scores: dict[str, pd.Series] = {}
    path_features_diag: dict[str, str] = {}
    path_valid_targets: dict[str, pd.Series] = {}
    for target_name in stable_path_targets:
        if target_name not in path_targets:
            raise ValueError(f"Unknown stable path target: {target_name}")
        score, diag = _stable_proxy_score(
            train=train,
            valid=valid_source,
            features=path_features,
            y_train=path_targets[target_name].loc[train_mask],
            binary_target=target_name != "path_quality",
            proxy_top_k=proxy_top_k,
            min_inner_folds=min_inner_folds,
            min_inner_train_rows=min_inner_train_rows,
            min_inner_valid_rows=min_inner_valid_rows,
            min_consistency=min_consistency,
        )
        path_scores[target_name] = score.reset_index(drop=True)
        path_features_diag[target_name] = (
            f"{diag['selection_mode']}|{','.join(diag.get('proxy_features', []))}|{diag.get('proxy_feature_directions', '')}"
        )
        path_valid_targets[target_name] = path_targets[target_name].loc[valid_mask].copy().reset_index(drop=True)

    selector_specs: list[dict[str, Any]] = []
    for label_arm in label_arms:
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        label_score = label_scores[label_arm]
        selector_specs.append(
            {
                "selector": "stable_label_proxy_oos",
                "label_arm": label_arm,
                "economic_arm": "none",
                "score": label_score,
                "target": target_valid,
                "label_score": target_valid["target_soft"],
                "economic_score": None,
                "economic_target": None,
                "label_proxy_features": label_features[label_arm],
                "economic_proxy_features": "",
            }
        )
        for path_target_name, path_score in path_scores.items():
            economic_arm = f"stable_state_path_{path_target_name}"
            selector_specs.extend(
                [
                    {
                        "selector": f"stable_state_path_{path_target_name}_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": economic_arm,
                        "score": path_score,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": path_score,
                        "economic_target": path_valid_targets[path_target_name],
                        "label_proxy_features": label_features[label_arm],
                        "economic_proxy_features": path_features_diag[path_target_name],
                    },
                    {
                        "selector": f"stable_combined_l{combine_label_weight:.2f}_label_state_path_{path_target_name}_oos",
                        "label_arm": label_arm,
                        "economic_arm": economic_arm,
                        "score": combine_label_weight * label_score + (1.0 - combine_label_weight) * path_score,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": path_score,
                        "economic_target": path_valid_targets[path_target_name],
                        "label_proxy_features": label_features[label_arm],
                        "economic_proxy_features": path_features_diag[path_target_name],
                    },
                ]
            )
            for gate_frac in gate_fracs:
                label_gate = _top_gate(label_score, gate_frac)
                path_gate = _top_gate(path_score, gate_frac)
                selector_specs.extend(
                    [
                        {
                            "selector": f"stable_state_path_{path_target_name}_gate{gate_frac:.2f}_then_label_oos",
                            "label_arm": label_arm,
                            "economic_arm": economic_arm,
                            "score": label_score.where(path_gate),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": path_score,
                            "economic_target": path_valid_targets[path_target_name],
                            "label_proxy_features": label_features[label_arm],
                            "economic_proxy_features": path_features_diag[path_target_name],
                        },
                        {
                            "selector": f"stable_dual_label_state_path_{path_target_name}_gate{gate_frac:.2f}_oos",
                            "label_arm": label_arm,
                            "economic_arm": economic_arm,
                            "score": label_score.where(label_gate & path_gate),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": path_score,
                            "economic_target": path_valid_targets[path_target_name],
                            "label_proxy_features": label_features[label_arm],
                            "economic_proxy_features": path_features_diag[path_target_name],
                        },
                    ]
                )

    rows: list[dict[str, Any]] = []
    period_slices = [("month", month, valid_indices)]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for spec in selector_specs:
        score = pd.to_numeric(spec["score"], errors="coerce").reset_index(drop=True)
        target = spec["target"].reset_index(drop=True)
        label_score = pd.to_numeric(spec["label_score"], errors="coerce").reset_index(drop=True)
        economic_score = (
            pd.to_numeric(spec["economic_score"], errors="coerce").reset_index(drop=True)
            if spec["economic_score"] is not None
            else None
        )
        economic_target = (
            pd.to_numeric(spec["economic_target"], errors="coerce").reset_index(drop=True)
            if spec["economic_target"] is not None
            else None
        )
        for period_type, period, pos in period_slices:
            for top_frac in top_fracs:
                rows.append(
                    _score_period(
                        frame=valid.iloc[pos].reset_index(drop=True),
                        metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                        target=target.iloc[pos].reset_index(drop=True),
                        score=score.iloc[pos].reset_index(drop=True),
                        period_type=period_type,
                        period=period,
                        month=month,
                        selector=spec["selector"],
                        label_arm=spec["label_arm"],
                        economic_arm=spec["economic_arm"],
                        top_frac=float(top_frac),
                        label_score=label_score.iloc[pos].reset_index(drop=True),
                        economic_score=economic_score.iloc[pos].reset_index(drop=True)
                        if economic_score is not None
                        else None,
                        economic_target=economic_target.iloc[pos].reset_index(drop=True)
                        if economic_target is not None
                        else None,
                        label_proxy_features=spec["label_proxy_features"],
                        economic_proxy_features=spec["economic_proxy_features"],
                    )
                )
    return rows


def _write_markdown(
    *,
    output_dir: Path,
    aggregate: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "stable_path_proxy_economic_ablation.md"
    fit_cols = [
        "trainworthy_pass",
        "fit_economic_pass",
        "holdout_economic_pass",
        "fit_sign_pass",
        "holdout_sign_pass",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "fit_mean_return_net",
        "holdout_mean_return_net",
        "fit_bad_mae_1r_rate",
        "holdout_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "holdout_p90_mae_norm",
        "fit_timeout_rate",
        "holdout_timeout_rate",
        "fit_score_ic_u",
        "holdout_score_ic_u",
        "fit_selected_rows",
        "holdout_selected_rows",
    ]
    agg_cols = [
        "acceptance_gate",
        "material_acceptance_gate",
        "period_type",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "periods",
        "positive_return_period_rate",
        "mean_return_net",
        "worst_period_return_net",
        "q25_period_return_net",
        "score_ic_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "mean_selected_rows",
    ]
    lines = [
        "# Stable Path Proxy Economic Ablation",
        "",
        "Scope: proxy-only. Path features are selected by causal inner-month stability before scoring each OOS month.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Stable path targets: `{', '.join(manifest['stable_path_targets'])}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Proxy top-k: `{manifest['proxy_top_k']}`. Min inner folds: `{manifest['min_inner_folds']}`. Min consistency: `{manifest['min_consistency']}`.",
        "",
        "## Fit / Holdout",
        "",
        _table(fit_holdout, fit_cols, limit=80),
        "",
        "## Month Aggregate",
        "",
        _table(
            aggregate[aggregate["period_type"].eq("month")].sort_values(
                ["acceptance_gate", "mean_return_net"],
                ascending=[False, False],
            ),
            agg_cols,
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
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
    label_arms: list[str],
    stable_path_targets: list[str],
    months: list[str],
    top_fracs: list[float],
    combine_label_weight: float,
    gate_fracs: list[float],
    proxy_top_k: int,
    min_inner_folds: int,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    min_consistency: float,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    min_material_selected_rows: int,
    max_timeout_rate: float | None,
    fit_months: list[str],
    holdout_month: str,
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
    path_features = _state_path_feature_columns(features)
    targets = _label_targets(frame, metrics)
    path_targets = _state_path_risk_targets(metrics)

    missing_labels = sorted(set(label_arms).difference(targets))
    missing_path = sorted(set(stable_path_targets).difference(path_targets))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    if missing_path:
        raise ValueError(f"Unknown path targets: {missing_path}")

    rows: list[dict[str, Any]] = []
    for month in months:
        rows.extend(
            _run_month(
                frame=frame,
                metrics=metrics,
                targets=targets,
                path_targets=path_targets,
                features=features,
                path_features=path_features,
                month=month,
                label_arms=label_arms,
                stable_path_targets=stable_path_targets,
                top_fracs=top_fracs,
                combine_label_weight=combine_label_weight,
                gate_fracs=gate_fracs,
                proxy_top_k=proxy_top_k,
                min_inner_folds=min_inner_folds,
                min_inner_train_rows=min_inner_train_rows,
                min_inner_valid_rows=min_inner_valid_rows,
                min_consistency=min_consistency,
            )
        )

    period_rows = pd.DataFrame(rows)
    aggregate = _aggregate(
        period_rows,
        min_material_selected_rows=min_material_selected_rows,
        max_timeout_rate=max_timeout_rate,
    )
    fit_holdout = _fit_holdout_summary(
        period_rows,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_material_selected_rows,
        max_timeout_rate=max_timeout_rate,
    )

    paths = {
        "period_rows": output_dir / "stable_path_proxy_period_rows.csv",
        "aggregate": output_dir / "stable_path_proxy_aggregate.csv",
        "fit_holdout": output_dir / "stable_path_proxy_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)

    manifest = {
        "scope": "stable_path_proxy_economic_ablation",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "label_arms": list(label_arms),
        "stable_path_targets": list(stable_path_targets),
        "months": list(months),
        "top_fracs": [float(value) for value in top_fracs],
        "combine_label_weight": float(combine_label_weight),
        "gate_fracs": [float(value) for value in gate_fracs],
        "proxy_top_k": int(proxy_top_k),
        "min_inner_folds": int(min_inner_folds),
        "min_inner_train_rows": int(min_inner_train_rows),
        "min_inner_valid_rows": int(min_inner_valid_rows),
        "min_consistency": float(min_consistency),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(value) for value in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "feature_count": int(len(features)),
        "path_feature_count": int(len(path_features)),
        "min_material_selected_rows": int(min_material_selected_rows),
        "max_timeout_rate": max_timeout_rate,
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        aggregate=aggregate,
        fit_holdout=fit_holdout,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "stable_path_proxy_economic_ablation.md")}},
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
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=list(DEFAULT_LABEL_ARMS))
    parser.add_argument(
        "--stable-path-targets",
        type=lambda value: _parse_csv(value, DEFAULT_STABLE_PATH_TARGETS),
        default=list(DEFAULT_STABLE_PATH_TARGETS),
    )
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--combine-label-weight", type=float, default=0.50)
    parser.add_argument("--gate-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_GATE_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-inner-folds", type=int, default=1)
    parser.add_argument("--min-inner-train-rows", type=int, default=500)
    parser.add_argument("--min-inner-valid-rows", type=int, default=100)
    parser.add_argument("--min-consistency", type=float, default=0.50)
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
    parser.add_argument("--min-material-selected-rows", type=int, default=3)
    parser.add_argument("--max-timeout-rate", type=float, default=None)
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, ("2026-04", "2026-05")), default=["2026-04", "2026-05"])
    parser.add_argument("--holdout-month", default="2026-06")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=list(args.label_arms),
        stable_path_targets=list(args.stable_path_targets),
        months=list(args.months),
        top_fracs=list(args.top_fracs),
        combine_label_weight=float(args.combine_label_weight),
        gate_fracs=list(args.gate_fracs),
        proxy_top_k=int(args.proxy_top_k),
        min_inner_folds=int(args.min_inner_folds),
        min_inner_train_rows=int(args.min_inner_train_rows),
        min_inner_valid_rows=int(args.min_inner_valid_rows),
        min_consistency=float(args.min_consistency),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        min_material_selected_rows=int(args.min_material_selected_rows),
        max_timeout_rate=args.max_timeout_rate,
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
    )
    print(json.dumps(_json_safe({key: value for key, value in manifest.items() if key != "reports"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
