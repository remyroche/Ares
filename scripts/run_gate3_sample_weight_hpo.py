#!/usr/bin/env python3
"""Gate 3 sample-weight HPO for clean-vs-dirty path learnability.

This is a focused diagnostic for the current broad-source Gate 3 blocker. It
optimizes sample-weight formula parameters under month-forward splits and
reports trade-relevant top-k clean precision, not just AUC.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_clean_dirty_learnability_oos import (  # noqa: E402
    DEFAULT_LABEL_VARIANTS,
    _positive_clean_variant,
    _safe_auc,
    _safe_ap,
    _topk_precision_metrics,
)
from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
    _append_fold_ae_gmm_state_features,
    _apply_spread_symbol_universe,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/gate3_sample_weight_hpo_v1")
DEFAULT_LABEL_VARIANT = "positive_econ_sideaware_exec_resolution"


@dataclass(frozen=True)
class WeightConfig:
    name: str
    clean_boost: float
    dirty_boost: float
    bad_mae_positive_boost: float
    timeout_positive_boost: float
    high_spread_dirty_boost: float
    month_side_balance_power: float
    side_balance_power: float
    utility_tail_power: float
    utility_tail_weight: float
    max_weight: float


BASELINE_CONFIGS: tuple[WeightConfig, ...] = (
    WeightConfig("uniform", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
    WeightConfig("execres_mild", 1.0, 0.8, 0.8, 0.35, 0.25, 0.35, 0.0, 2.0, 0.35, 4.0),
    WeightConfig("execres_balanced", 2.0, 1.5, 1.5, 0.75, 0.50, 0.50, 0.0, 3.0, 0.50, 5.0),
    WeightConfig("execres_badmae_heavy", 1.5, 1.2, 2.8, 0.60, 0.50, 0.50, 0.0, 2.5, 0.35, 5.0),
    WeightConfig("execres_dirty_heavy", 1.6, 2.8, 1.6, 0.60, 0.75, 0.50, 0.0, 2.5, 0.35, 5.0),
    WeightConfig("execres_side_balanced", 1.6, 1.3, 1.6, 0.60, 0.50, 0.65, 0.40, 2.5, 0.35, 5.0),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _normalize_weights(
    weights: pd.Series,
    *,
    min_weight: float = 0.10,
    max_weight: float = 5.0,
) -> pd.Series:
    w = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    w = w.clip(lower=float(min_weight), upper=float(max_weight))
    mean = float(w.mean()) if len(w) else 1.0
    if not math.isfinite(mean) or mean <= 0.0:
        return pd.Series(1.0, index=w.index, dtype=np.float32)
    return (w / mean).clip(lower=float(min_weight), upper=float(max_weight)).astype(np.float32)


def _ess_frac(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = w.clip(lower=0.0).to_numpy(dtype=np.float64)
    denom = float(np.square(w).sum())
    if denom <= 0.0 or len(w) == 0:
        return 0.0
    return float((w.sum() * w.sum()) / denom / len(w))


def _p99_p50(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(w) == 0:
        return float("nan")
    p50 = float(w.quantile(0.50))
    p99 = float(w.quantile(0.99))
    return p99 / p50 if p50 > 0.0 and math.isfinite(p50) else float("nan")


def _month_side_balance(frame: pd.DataFrame, side: pd.Series, power: float) -> pd.Series:
    if float(power) <= 0.0:
        return pd.Series(1.0, index=frame.index, dtype=np.float32)
    key = (
        frame["__ts__"].dt.to_period("M").astype(str)
        + "_"
        + np.where(pd.to_numeric(side, errors="coerce").fillna(1.0).to_numpy(dtype=np.float64) < 0.0, "short", "long")
    )
    counts = pd.Series(key, index=frame.index).map(pd.Series(key, index=frame.index).value_counts(dropna=False)).astype(float)
    raw = np.power(1.0 / counts.clip(lower=1.0), float(power))
    return _normalize_weights(raw, min_weight=0.25, max_weight=3.0)


def _side_balance(side: pd.Series, power: float) -> pd.Series:
    if float(power) <= 0.0:
        return pd.Series(1.0, index=side.index, dtype=np.float32)
    side_key = np.where(pd.to_numeric(side, errors="coerce").fillna(1.0).to_numpy(dtype=np.float64) < 0.0, "short", "long")
    counts = pd.Series(side_key, index=side.index).map(pd.Series(side_key, index=side.index).value_counts(dropna=False)).astype(float)
    raw = np.power(1.0 / counts.clip(lower=1.0), float(power))
    return _normalize_weights(raw, min_weight=0.25, max_weight=3.0)


def _build_weight_values(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    population: np.ndarray,
    y_clean: pd.Series,
    config: WeightConfig,
) -> tuple[pd.Series, dict[str, float]]:
    idx = frame.index
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").reindex(idx).fillna(0.0)
    mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").reindex(idx).fillna(0.0)
    timeout = pd.to_numeric(metrics["is_timeout"], errors="coerce").reindex(idx).fillna(0.0).gt(0.5)
    side = pd.to_numeric(metrics.get("side", pd.Series(1.0, index=idx)), errors="coerce").reindex(idx).fillna(1.0)
    positive = pd.Series(np.asarray(population, dtype=bool), index=idx)
    clean = pd.to_numeric(y_clean.reindex(idx), errors="coerce").fillna(0.0).gt(0.5)
    dirty = positive & ~clean
    bad_mae_positive = positive & mae_norm.ge(1.0)
    timeout_positive = positive & timeout
    spread_proxy = pd.to_numeric(
        frame.get("median_spread_bps", frame.get("p75_spread_bps", pd.Series(np.nan, index=idx))),
        errors="coerce",
    ).reindex(idx)
    spread_cutoff = float(spread_proxy.quantile(0.75)) if int(spread_proxy.notna().sum()) else float("nan")
    high_spread_dirty = dirty & spread_proxy.ge(spread_cutoff).fillna(False) if math.isfinite(spread_cutoff) else pd.Series(False, index=idx)
    u_rank = u.rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0)
    tail = 1.0 + float(config.utility_tail_weight) * np.power(u_rank, float(config.utility_tail_power))
    raw = (
        1.0
        + float(config.clean_boost) * clean.astype(float)
        + float(config.dirty_boost) * dirty.astype(float)
        + float(config.bad_mae_positive_boost) * bad_mae_positive.astype(float)
        + float(config.timeout_positive_boost) * timeout_positive.astype(float)
        + float(config.high_spread_dirty_boost) * high_spread_dirty.astype(float)
    )
    raw = raw * tail
    raw = raw * _month_side_balance(frame, side, float(config.month_side_balance_power))
    raw = raw * _side_balance(side, float(config.side_balance_power))
    weights = _normalize_weights(pd.Series(raw, index=idx), max_weight=float(config.max_weight))
    pop_weights = weights.loc[positive]
    diagnostics = {
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_p99_p50": _p99_p50(pop_weights),
        "weight_ess_frac": _ess_frac(pop_weights),
        "weight_top1pct_mass": float(
            pop_weights.nlargest(max(1, int(math.ceil(0.01 * len(pop_weights))))).sum()
            / max(float(pop_weights.sum()), 1e-12)
        )
        if len(pop_weights)
        else float("nan"),
    }
    return weights.astype(np.float32), diagnostics


def _cap_train_rows(
    x: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    ts: pd.Series,
    *,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if int(max_rows) <= 0 or len(x) <= int(max_rows):
        return x.reset_index(drop=True), y.reset_index(drop=True), w.reset_index(drop=True)
    positions = np.linspace(0, len(x) - 1, int(max_rows), dtype=np.int64)
    positions = np.unique(positions)
    return (
        x.iloc[positions].reset_index(drop=True),
        y.iloc[positions].reset_index(drop=True),
        w.iloc[positions].reset_index(drop=True),
    )


def _fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int,
) -> pd.Series:
    model = ExtraTreesClassifier(
        n_estimators=96,
        max_depth=8,
        min_samples_leaf=45,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(
        x_train.reset_index(drop=True),
        y_train.reset_index(drop=True).astype(int),
        sample_weight=w_train.reset_index(drop=True).astype(float),
    )
    pred = model.predict_proba(x_valid.reset_index(drop=True))[:, 1]
    return pd.Series(pred.astype(np.float32), index=x_valid.reset_index(drop=True).index)


def _score_fold(
    *,
    y_valid: pd.Series,
    score: pd.Series,
    valid_metrics: pd.DataFrame,
    month: str,
) -> dict[str, Any]:
    top = _topk_precision_metrics(y_valid, score)
    side = pd.to_numeric(valid_metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    rows: dict[str, Any] = {
        "month": month,
        "rows": int(len(y_valid)),
        "clean_rate": _safe_mean(y_valid),
        "roc_auc": _safe_auc(y_valid, score),
        "average_precision": _safe_ap(y_valid, score),
        **top,
    }
    for side_name, mask in (("long", side.ge(0.0)), ("short", side.lt(0.0))):
        if int(mask.sum()) < 100 or y_valid.loc[mask].nunique(dropna=True) < 2:
            continue
        side_top = _topk_precision_metrics(y_valid.loc[mask].reset_index(drop=True), score.loc[mask].reset_index(drop=True), prefix=f"{side_name}_")
        rows[f"{side_name}_rows"] = int(mask.sum())
        rows[f"{side_name}_clean_rate"] = _safe_mean(y_valid.loc[mask])
        rows[f"{side_name}_roc_auc"] = _safe_auc(y_valid.loc[mask].reset_index(drop=True), score.loc[mask].reset_index(drop=True))
        rows.update(side_top)
    return rows


def _objective_from_folds(folds: list[dict[str, Any]], weight_diag: dict[str, float]) -> float:
    frame = pd.DataFrame(folds)
    top10 = pd.to_numeric(frame["top10_clean_rate"], errors="coerce")
    top20 = pd.to_numeric(frame["top20_clean_rate"], errors="coerce")
    top30 = pd.to_numeric(frame["top30_clean_rate"], errors="coerce")
    lift10 = pd.to_numeric(frame["top10_clean_lift"], errors="coerce")
    long10 = pd.to_numeric(frame.get("long_top10_clean_rate", pd.Series(np.nan, index=frame.index)), errors="coerce")
    short10 = pd.to_numeric(frame.get("short_top10_clean_rate", pd.Series(np.nan, index=frame.index)), errors="coerce")
    side_min = pd.concat([long10, short10], axis=1).min(axis=1)
    objective = (
        1.00 * _safe_mean(top10)
        + 0.60 * _safe_mean(top20)
        + 0.35 * _safe_mean(top30)
        + 0.20 * _safe_mean(lift10)
        + 0.70 * _safe_mean(side_min)
    )
    min_top10 = float(top10.min()) if len(top10.dropna()) else float("nan")
    if math.isfinite(min_top10):
        objective += 0.50 * min_top10
    ess = float(weight_diag.get("weight_ess_frac_mean", float("nan")))
    p99_p50 = float(weight_diag.get("weight_p99_p50_mean", float("nan")))
    top1_mass = float(weight_diag.get("weight_top1pct_mass_mean", float("nan")))
    if math.isfinite(ess) and ess < 0.30:
        objective -= 1.5 * (0.30 - ess)
    if math.isfinite(p99_p50) and p99_p50 > 8.0:
        objective -= 0.02 * (p99_p50 - 8.0)
    if math.isfinite(top1_mass) and top1_mass > 0.05:
        objective -= 2.0 * (top1_mass - 0.05)
    return float(objective) if math.isfinite(objective) else float("-inf")


def _suggest_config(trial: Any, trial_id: int) -> WeightConfig:
    return WeightConfig(
        name=f"optuna_{trial_id:03d}",
        clean_boost=float(trial.suggest_float("clean_boost", 0.5, 3.5)),
        dirty_boost=float(trial.suggest_float("dirty_boost", 0.0, 3.5)),
        bad_mae_positive_boost=float(trial.suggest_float("bad_mae_positive_boost", 0.5, 4.0)),
        timeout_positive_boost=float(trial.suggest_float("timeout_positive_boost", 0.0, 1.5)),
        high_spread_dirty_boost=float(trial.suggest_float("high_spread_dirty_boost", 0.0, 1.5)),
        month_side_balance_power=float(trial.suggest_float("month_side_balance_power", 0.0, 0.9)),
        side_balance_power=float(trial.suggest_float("side_balance_power", 0.0, 0.6)),
        utility_tail_power=float(trial.suggest_float("utility_tail_power", 1.5, 5.0)),
        utility_tail_weight=float(trial.suggest_float("utility_tail_weight", 0.0, 1.2)),
        max_weight=float(trial.suggest_float("max_weight", 3.0, 6.0)),
    )


def _prepare_folds(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    months: list[str],
    label_variant: str,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_feature_store_features: int | None,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = _load_labels(labels_path)
    frame, symbol_filter, _symbols = _apply_spread_symbol_universe(
        frame,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        target_symbol_count=target_symbol_count,
        max_spread_bps=None,
    )
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        feature_matrix = feature_matrix.astype(np.float32, copy=False).reset_index(drop=True)
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix], axis=1, copy=False)
    metrics = _path_metrics(frame)
    population, y_clean, label_diag = _positive_clean_variant(
        metrics,
        variant=label_variant,
        frame=frame,
    )
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    y_clean = y_clean.reset_index(drop=True)
    population_s = pd.Series(np.asarray(population, dtype=bool), index=frame.index)
    features = _feature_columns(frame)
    all_months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    if not months:
        months = all_months[1:]
    folds: list[dict[str, Any]] = []
    for offset, month in enumerate(months):
        valid_mask = frame["__ts__"].dt.to_period("M").astype(str).eq(month)
        train_mask = frame["__ts__"].lt(pd.Period(month).start_time)
        train = frame.loc[train_mask].reset_index(drop=True)
        valid = frame.loc[valid_mask].reset_index(drop=True)
        if train.empty or valid.empty:
            continue
        train_metrics = metrics.loc[train_mask].reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].reset_index(drop=True)
        x_train = train[features].astype(np.float32, copy=False).reset_index(drop=True)
        x_valid = valid[features].astype(np.float32, copy=False).reset_index(drop=True)
        x_train, x_valid, generated_features, ae_diag = _append_fold_ae_gmm_state_features(
            x_train=x_train,
            x_valid=x_valid,
            train_frame=train,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            enabled=bool(include_ae_gmm_state_features),
            max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            ae_max_iter=int(ae_gmm_state_feature_max_iter),
            random_state=int(seed) + offset,
        )
        train_pop = population_s.loc[train_mask].reset_index(drop=True)
        valid_pop = population_s.loc[valid_mask].reset_index(drop=True)
        train_keep = train_pop.to_numpy(dtype=bool)
        valid_keep = valid_pop.to_numpy(dtype=bool)
        if int(train_keep.sum()) < 1000 or int(valid_keep.sum()) < 200:
            continue
        folds.append(
            {
                "month": month,
                "train": train,
                "valid": valid,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "x_train": x_train.loc[train_keep].reset_index(drop=True),
                "x_valid": x_valid.loc[valid_keep].reset_index(drop=True),
                "y_train": y_clean.loc[train_mask].reset_index(drop=True).loc[train_keep].reset_index(drop=True),
                "y_valid": y_clean.loc[valid_mask].reset_index(drop=True).loc[valid_keep].reset_index(drop=True),
                "train_frame_pop": train.loc[train_keep].reset_index(drop=True),
                "valid_frame_pop": valid.loc[valid_keep].reset_index(drop=True),
                "train_metrics_pop": train_metrics.loc[train_keep].reset_index(drop=True),
                "valid_metrics_pop": valid_metrics.loc[valid_keep].reset_index(drop=True),
                "population_train_rows": int(train_keep.sum()),
                "population_valid_rows": int(valid_keep.sum()),
                "ae_gmm_generated_features": int(len(generated_features)),
                "ae_gmm_status": ae_diag.get("ae_gmm_state_feature_status"),
            }
        )
    manifest = {
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "feature_count": int(len(features)),
        "feature_store": feature_report,
        "symbol_universe_filter": symbol_filter,
        "label_diag": label_diag,
        "fold_count": int(len(folds)),
        "fold_months": [fold["month"] for fold in folds],
    }
    return folds, manifest


def _evaluate_config(
    *,
    folds: list[dict[str, Any]],
    config: WeightConfig,
    max_train_rows: int,
    seed: int,
    trial_number: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    weight_diags: list[dict[str, float]] = []
    for fold_id, fold in enumerate(folds):
        weights_full, weight_diag = _build_weight_values(
            frame=fold["train_frame_pop"],
            metrics=fold["train_metrics_pop"],
            population=np.ones(len(fold["train_frame_pop"]), dtype=bool),
            y_clean=fold["y_train"],
            config=config,
        )
        x_train, y_train, w_train = _cap_train_rows(
            fold["x_train"],
            fold["y_train"],
            weights_full,
            fold["train_frame_pop"]["__ts__"],
            max_rows=int(max_train_rows),
        )
        pred = _fit_predict(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=fold["x_valid"],
            seed=int(seed) + 1000 * int(trial_number) + int(fold_id),
        )
        fold_metric = _score_fold(
            y_valid=fold["y_valid"],
            score=pred,
            valid_metrics=fold["valid_metrics_pop"],
            month=fold["month"],
        )
        fold_metric.update(
            {
                "trial_number": int(trial_number),
                "weight_name": config.name,
                "train_rows": int(len(x_train)),
                "train_rows_uncapped": int(len(fold["x_train"])),
                "valid_rows": int(len(fold["x_valid"])),
                **{f"weight_{k}": v for k, v in asdict(config).items() if k != "name"},
                **weight_diag,
            }
        )
        fold_rows.append(fold_metric)
        weight_diags.append(weight_diag)
    weight_summary = {
        f"{key}_mean": _safe_mean(pd.Series([diag.get(key) for diag in weight_diags]))
        for key in (
            "weight_ess_frac",
            "weight_p99_p50",
            "weight_top1pct_mass",
            "weight_max",
        )
    }
    objective = _objective_from_folds(fold_rows, weight_summary)
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "trial_number": int(trial_number),
        "weight_name": config.name,
        "objective": float(objective),
        "folds": int(len(fold_rows)),
        "mean_clean_rate": _safe_mean(fold_df["clean_rate"]) if not fold_df.empty else float("nan"),
        "mean_top30_clean_rate": _safe_mean(fold_df["top30_clean_rate"]) if not fold_df.empty else float("nan"),
        "mean_top20_clean_rate": _safe_mean(fold_df["top20_clean_rate"]) if not fold_df.empty else float("nan"),
        "mean_top10_clean_rate": _safe_mean(fold_df["top10_clean_rate"]) if not fold_df.empty else float("nan"),
        "min_top10_clean_rate": float(pd.to_numeric(fold_df["top10_clean_rate"], errors="coerce").min()) if not fold_df.empty else float("nan"),
        "mean_top10_lift": _safe_mean(fold_df["top10_clean_lift"]) if not fold_df.empty else float("nan"),
        "mean_long_top10_clean_rate": _safe_mean(fold_df.get("long_top10_clean_rate", pd.Series(dtype=float))) if not fold_df.empty else float("nan"),
        "mean_short_top10_clean_rate": _safe_mean(fold_df.get("short_top10_clean_rate", pd.Series(dtype=float))) if not fold_df.empty else float("nan"),
        "min_side_top10_clean_rate": float(
            pd.concat(
                [
                    pd.to_numeric(fold_df.get("long_top10_clean_rate", pd.Series(np.nan, index=fold_df.index)), errors="coerce"),
                    pd.to_numeric(fold_df.get("short_top10_clean_rate", pd.Series(np.nan, index=fold_df.index)), errors="coerce"),
                ],
                axis=1,
            )
            .min(axis=1)
            .min()
        )
        if not fold_df.empty
        else float("nan"),
        "mean_auc": _safe_mean(fold_df["roc_auc"]) if not fold_df.empty else float("nan"),
        "mean_ap": _safe_mean(fold_df["average_precision"]) if not fold_df.empty else float("nan"),
        **weight_summary,
        **{f"weight_{k}": v for k, v in asdict(config).items() if k != "name"},
    }
    return summary, fold_rows


def _write_markdown(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "gate3_sample_weight_hpo.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "rank",
        "weight_name",
        "objective",
        "mean_top10_clean_rate",
        "mean_top20_clean_rate",
        "mean_top30_clean_rate",
        "min_top10_clean_rate",
        "mean_top10_lift",
        "mean_long_top10_clean_rate",
        "mean_short_top10_clean_rate",
        "min_side_top10_clean_rate",
        "mean_auc",
        "weight_ess_frac_mean",
        "weight_p99_p50_mean",
        "weight_top1pct_mass_mean",
    ]
    fold_cols = [
        "weight_name",
        "month",
        "top10_clean_rate",
        "top20_clean_rate",
        "top30_clean_rate",
        "top10_clean_lift",
        "long_top10_clean_rate",
        "short_top10_clean_rate",
        "roc_auc",
        "weight_ess_frac",
        "weight_p99_p50",
    ]
    best_name = str(summary.iloc[0]["weight_name"]) if not summary.empty else ""
    lines = [
        "# Gate 3 Sample-Weight HPO",
        "",
        "Scope: month-forward clean-vs-dirty path learnability HPO. The objective prioritizes top-k clean precision and side-slice stability; AUC is reported as secondary context.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Months: `{', '.join(manifest['fold_months'])}`",
        f"Label: `{manifest['label_variant']}`",
        "",
        "## Winner",
        "",
        table(summary.head(1), cols),
        "",
        "## Trial Ranking",
        "",
        table(summary, cols, limit=40),
        "",
        "## Winner Fold Detail",
        "",
        table(folds[folds["weight_name"].eq(best_name)], fold_cols),
        "",
        "## Outputs",
        "",
        f"- Trial summary: `{manifest['outputs']['trial_summary']}`",
        f"- Fold metrics: `{manifest['outputs']['fold_metrics']}`",
        f"- Best config: `{manifest['outputs']['best_config']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_hpo(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    label_variant: str,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_feature_store_features: int | None,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    max_train_rows: int,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        months=months,
        label_variant=label_variant,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        target_symbol_count=target_symbol_count,
        max_feature_store_features=max_feature_store_features,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
        seed=seed,
    )
    if not folds:
        raise RuntimeError("No valid folds prepared for sample-weight HPO")
    summaries: list[dict[str, Any]] = []
    fold_rows_all: list[dict[str, Any]] = []
    trial_counter = 0
    for config in BASELINE_CONFIGS:
        summary, fold_rows = _evaluate_config(
            folds=folds,
            config=config,
            max_train_rows=max_train_rows,
            seed=seed,
            trial_number=trial_counter,
        )
        summaries.append(summary)
        fold_rows_all.extend(fold_rows)
        trial_counter += 1
    try:
        import optuna
    except Exception:
        optuna = None
    if optuna is not None and int(n_trials) > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            nonlocal trial_counter
            config = _suggest_config(trial, trial_counter)
            summary, fold_rows = _evaluate_config(
                folds=folds,
                config=config,
                max_train_rows=max_train_rows,
                seed=seed,
                trial_number=trial_counter,
            )
            summaries.append(summary)
            fold_rows_all.extend(fold_rows)
            trial_counter += 1
            for key, value in summary.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    trial.set_user_attr(key, float(value))
            return float(summary["objective"])

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    elif int(n_trials) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_trials)):
            config = WeightConfig(
                name=f"random_{trial_counter:03d}",
                clean_boost=float(rng.uniform(0.5, 3.5)),
                dirty_boost=float(rng.uniform(0.0, 3.5)),
                bad_mae_positive_boost=float(rng.uniform(0.5, 4.0)),
                timeout_positive_boost=float(rng.uniform(0.0, 1.5)),
                high_spread_dirty_boost=float(rng.uniform(0.0, 1.5)),
                month_side_balance_power=float(rng.uniform(0.0, 0.9)),
                side_balance_power=float(rng.uniform(0.0, 0.6)),
                utility_tail_power=float(rng.uniform(1.5, 5.0)),
                utility_tail_weight=float(rng.uniform(0.0, 1.2)),
                max_weight=float(rng.uniform(3.0, 6.0)),
            )
            summary, fold_rows = _evaluate_config(
                folds=folds,
                config=config,
                max_train_rows=max_train_rows,
                seed=seed,
                trial_number=trial_counter,
            )
            summaries.append(summary)
            fold_rows_all.extend(fold_rows)
            trial_counter += 1
    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1, dtype=np.int32))
    folds_df = pd.DataFrame(fold_rows_all)
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    paths = {
        "trial_summary": output_dir / "gate3_sample_weight_hpo_trials.csv",
        "fold_metrics": output_dir / "gate3_sample_weight_hpo_folds.csv",
        "best_config": output_dir / "gate3_sample_weight_hpo_best.json",
        "manifest": output_dir / "manifest.json",
    }
    summary_df.to_csv(paths["trial_summary"], index=False)
    folds_df.to_csv(paths["fold_metrics"], index=False)
    paths["best_config"].write_text(json.dumps(_json_safe(best), indent=2), encoding="utf-8")
    manifest.update(
        {
            "scope": "gate3_sample_weight_hpo_clean_dirty_topk",
            "labels_path": labels_path,
            "feature_dir": feature_dir,
            "feature_list_csv": feature_list_csv,
            "output_dir": output_dir,
            "label_variant": label_variant,
            "n_trials_requested": int(n_trials),
            "baseline_trials": int(len(BASELINE_CONFIGS)),
            "total_trials": int(len(summary_df)),
            "max_train_rows": int(max_train_rows),
            "model": {
                "type": "ExtraTreesClassifier",
                "n_estimators": 96,
                "max_depth": 8,
                "min_samples_leaf": 45,
                "max_features": "sqrt",
            },
            "primary_metrics": [
                "top10_clean_rate",
                "top20_clean_rate",
                "top30_clean_rate",
                "side_top10_clean_rate",
                "weight_ess_frac",
            ],
            "secondary_metrics": ["roc_auc", "average_precision"],
            "best_weight_name": best.get("weight_name", ""),
            "outputs": {key: str(value) for key, value in paths.items()},
        }
    )
    markdown = _write_markdown(output_dir, summary_df, folds_df, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--label-variant", default=DEFAULT_LABEL_VARIANT, choices=DEFAULT_LABEL_VARIANTS)
    parser.add_argument("--spread-baseline-path", type=Path, default=None)
    parser.add_argument("--spread-rank-column", default="p75_spread_bps")
    parser.add_argument("--target-symbol-count", type=int, default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER)
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_hpo(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, ()),
        label_variant=str(args.label_variant),
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=str(args.spread_rank_column),
        target_symbol_count=args.target_symbol_count,
        max_feature_store_features=args.max_feature_store_features,
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        max_train_rows=int(args.max_train_rows),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
