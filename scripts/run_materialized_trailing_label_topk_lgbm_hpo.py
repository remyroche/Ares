#!/usr/bin/env python3
"""Top-k LGBM HPO for materialized trailing-profit first-touch labels."""

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

try:
    from lightgbm import LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - dependency check.
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False

from scripts.run_first_touch_label_training_smoke import (  # noqa: E402
    _first_touch_eval_metrics,
    _target_from_frame,
)
from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
    _ae_gmm_smoke_feature_policy_columns,
    _append_fold_ae_gmm_state_features,
    _fold_ae_gmm_economic_targets,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
)
from scripts.run_label_weighted_proxy_ablation import WEIGHT_ARMS, _effective_sample_size, _weight_series  # noqa: E402
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_FEATURE_COLUMNS,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/materialized_trailing_label_topk_lgbm_hpo_v1")
DEFAULT_FIXED_PARAMS_JSON = Path("docs/promoted_s58_conditioned_trailing_topk_lgbm_params.json")
TOP_FRACS = (0.10, 0.20, 0.30)
TARGET_MODES = (
    "policy_soft",
    "target_soft",
    "exec_guarded_policy",
    "clean_exec",
    "time_decay_policy",
)


def _append_single_side_ae_gmm_state_features(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    max_train_rows: int,
    ae_max_iter: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    base_features = [
        str(col)
        for col in x_train.columns
        if str(col) not in set(str(v) for v in AE_GMM_FEATURE_COLUMNS)
    ]
    if len(base_features) < 2 or len(x_train) < 500:
        return x_train, x_valid, [], {
            "ae_gmm_state_feature_status": "single_side_insufficient_rows_or_features",
            "ae_gmm_state_feature_count": 0,
            "ae_gmm_state_input_feature_count": int(len(base_features)),
        }
    x_train_base = x_train.reindex(columns=base_features).astype(np.float32, copy=False)
    x_valid_base = x_valid.reindex(columns=base_features).astype(np.float32, copy=False)
    state = fit_ae_gmm_state(
        x_train_base.reset_index(drop=True),
        economic_targets=_fold_ae_gmm_economic_targets(
            train_metrics.reset_index(drop=True),
            train_frame=train_frame.reset_index(drop=True),
        ),
        random_state=int(random_state),
        max_train_rows=int(max_train_rows),
        ae_max_iter=int(ae_max_iter),
        require_both_sides=False,
        min_side_cluster_frac=0.02,
        min_side_cluster_rows=10,
    )
    if not bool(state.get("enabled", False)):
        return x_train, x_valid, [], {
            "ae_gmm_state_feature_status": f"single_side_{state.get('reason', 'state_disabled')}",
            "ae_gmm_state_feature_count": 0,
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
        }
    valid_generated = transform_ae_gmm_features(x_valid_base, state, index=x_valid.index)
    all_generated = [str(col) for col in valid_generated.columns]
    generated = _ae_gmm_smoke_feature_policy_columns(all_generated)
    generated = list(dict.fromkeys([*generated, "ae_gmm_oof_available"]))
    train_generated = transform_ae_gmm_features(x_train_base, state, index=x_train.index).reindex(
        columns=generated,
        fill_value=0.0,
    )
    valid_generated = valid_generated.reindex(columns=generated, fill_value=0.0)
    train_generated["ae_gmm_oof_available"] = np.float32(1.0)
    valid_generated["ae_gmm_oof_available"] = np.float32(1.0)
    selected_config = dict(state.get("selected_config", {}) or {})
    return (
        pd.concat([x_train, train_generated], axis=1, copy=False).astype(np.float32, copy=False),
        pd.concat([x_valid, valid_generated], axis=1, copy=False).astype(np.float32, copy=False),
        generated,
        {
            "ae_gmm_state_feature_status": "ok_single_side_outer_train",
            "ae_gmm_state_feature_count": int(len(generated)),
            "ae_gmm_state_all_feature_count": int(len(all_generated)),
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
            "ae_gmm_state_n_components": int(state.get("gmm_n_components", 0) or 0),
            "ae_gmm_state_path_cleanliness_score": float(
                selected_config.get("path_cleanliness_score", float("nan"))
            ),
            "ae_gmm_state_temporal_concentration_score": float(
                selected_config.get("temporal_concentration_score", float("nan"))
            ),
            "ae_gmm_state_train_feature_scope": "outer_train_in_sample",
            "ae_gmm_state_validation_feature_scope": "frozen_outer_train_artifact",
        },
    )


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_fixed_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params", payload)
    if not isinstance(params, dict):
        raise ValueError(f"Fixed params payload must be a dict or contain a params dict: {path}")
    required = {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "target_mode",
        "weight_arm",
    }
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"Fixed params missing keys {missing}: {path}")
    out = dict(params)
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        out[key] = int(float(out[key]))
    for key in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        out[key] = float(out[key])
    out["target_mode"] = str(out["target_mode"])
    out["weight_arm"] = str(out["weight_arm"])
    if "trial_number" in payload:
        out["_fixed_trial_number"] = int(float(payload["trial_number"]))
    return out


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_fold_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _fold_frame_columns(frame: pd.DataFrame) -> list[str]:
    exact = {
        "__ts__",
        "__symbol__",
        "__side__",
        "side",
        "side_name",
        "month",
        "__w__",
        "__econ_sideaware_execres_clean__",
        "__econ_sideaware_execres_dirty_positive__",
        "__econ_side_resolution_clean__",
        "__econ_side_resolution_dirty_positive__",
    }
    context_tokens = (
        "archetype",
        "source",
        "regime",
        "aegmm",
        "ae_gmm",
        "gmm",
        "cluster",
        "reconstruction",
        "latent",
        "posterior",
        "entropy",
        "mahalanobis",
    )
    keep: list[str] = []
    for col in frame.columns:
        name = str(col)
        lower = name.lower()
        if name in exact or name.startswith("__") or any(token in lower for token in context_tokens):
            keep.append(name)
    return list(dict.fromkeys([col for col in keep if col in frame.columns]))


def _write_fold_payload(fold: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    fold_dir = cache_dir / _safe_fold_name(str(fold["fold"]))
    fold_dir.mkdir(parents=True, exist_ok=True)
    payload_paths = {
        "train": fold_dir / "train.parquet",
        "valid": fold_dir / "valid.parquet",
        "train_metrics": fold_dir / "train_metrics.parquet",
        "valid_metrics": fold_dir / "valid_metrics.parquet",
        "x_train": fold_dir / "x_train.parquet",
        "x_valid": fold_dir / "x_valid.parquet",
    }
    for key, path in payload_paths.items():
        frame = fold[key]
        if key in {"x_train", "x_valid"}:
            frame = frame.clip(
                lower=float(np.finfo(np.float16).min),
                upper=float(np.finfo(np.float16).max),
                axis=None,
            ).astype(np.float16, copy=False)
        frame.to_parquet(path, index=False, compression="zstd", compression_level=9)
    slim = {key: value for key, value in fold.items() if key not in payload_paths}
    slim["payload_paths"] = {key: str(path) for key, path in payload_paths.items()}
    slim["train_rows"] = int(len(fold["x_train"]))
    slim["valid_rows"] = int(len(fold["x_valid"]))
    return slim


def _load_fold_payload(fold: dict[str, Any]) -> dict[str, Any]:
    if "payload_paths" not in fold:
        return fold
    loaded = dict(fold)
    for key, path in dict(fold["payload_paths"]).items():
        frame = pd.read_parquet(path)
        if key in {"x_train", "x_valid"}:
            frame = frame.astype(np.float32, copy=False)
        loaded[key] = frame
    return loaded


def _cap_rows(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False).astype(np.int64))


def _time_spread_cap_rows(n_rows: int, max_rows: int) -> np.ndarray:
    """Deterministically sample rows from the beginning, middle, and end."""
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    n = int(n_rows)
    k = int(max_rows)
    parts: list[np.ndarray] = []
    base = k // 3
    rem = k - base * 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    windows = [(0, n // 3), (n // 3, (2 * n) // 3), ((2 * n) // 3, n)]
    for size, (start, end) in zip(sizes, windows):
        size = min(int(size), max(int(end - start), 0))
        if size <= 0:
            continue
        if size >= end - start:
            parts.append(np.arange(start, end, dtype=np.int64))
        else:
            parts.append(np.linspace(start, end - 1, size, dtype=np.int64))
    if not parts:
        return np.arange(0, min(n, k), dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64))


def _auto_mda_keep_count(records: list[dict[str, Any]], requested_top_n: int) -> tuple[int, str, float]:
    """Choose feature count from MDA scores when no explicit top-k is requested."""

    if not records:
        return 0, "auto_mda_empty", 0.0
    if int(requested_top_n) > 0:
        return min(int(requested_top_n), len(records)), "explicit_top_n", 1.0
    scores = np.asarray([max(float(row.get("score", 0.0) or 0.0), 0.0) for row in records], dtype=np.float64)
    total = float(scores.sum())
    max_score = float(scores.max(initial=0.0))
    if total <= 0.0 or max_score <= 0.0:
        return 1, "auto_mda_no_positive_scores_keep_best", 0.0
    floor = max(1e-12, max_score * 1e-4)
    positive = scores > floor
    positive_total = float(scores[positive].sum())
    if positive_total <= 0.0:
        return 1, "auto_mda_below_noise_floor_keep_best", floor
    cumulative = np.cumsum(scores)
    candidates = np.flatnonzero((cumulative >= 0.99 * positive_total) & positive)
    keep_n = int(candidates[0] + 1) if candidates.size else int(positive.sum())
    keep_n = max(1, min(keep_n, int(positive.sum()), len(records)))
    return keep_n, "auto_mda_cumulative_positive_99pct", floor


def _select_features_by_univariate(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    target: pd.Series,
    *,
    top_n: int,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    if int(top_n) <= 0 or x_train.shape[1] <= int(top_n):
        rows = [
            {
                "fold": str(fold),
                "feature": str(col),
                "score": float("nan"),
                "rank": int(i + 1),
                "selected": True,
                "feature_selection_status": "disabled_or_not_needed",
            }
            for i, col in enumerate(x_train.columns)
        ]
        return x_train, x_valid, list(x_train.columns), pd.DataFrame(rows)
    y = _safe_numeric(target).replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna()
    if int(valid_y.sum()) < 100 or int(y.loc[valid_y].nunique(dropna=True)) < 3:
        keep = list(x_train.columns[: int(top_n)])
        rows = [
            {
                "fold": str(fold),
                "feature": str(col),
                "score": float("nan"),
                "rank": int(i + 1),
                "selected": col in keep,
                "feature_selection_status": "insufficient_target_variation",
            }
            for i, col in enumerate(x_train.columns)
        ]
        return x_train.loc[:, keep], x_valid.loc[:, keep], keep, pd.DataFrame(rows)
    x = x_train.loc[valid_y].astype(np.float32, copy=False)
    yr = y.loc[valid_y].rank(method="average").to_numpy(dtype=np.float32)
    yr -= float(np.nanmean(yr))
    yr_std = float(np.nanstd(yr))
    if yr_std <= 1e-12:
        yr_std = 1.0
    yr /= yr_std
    scores: list[tuple[str, float]] = []
    for col in x.columns:
        ser = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(ser.notna().sum()) < 100 or int(ser.nunique(dropna=True)) < 3:
            scores.append((str(col), 0.0))
            continue
        xr = ser.rank(method="average").fillna(ser.rank(method="average").median()).to_numpy(dtype=np.float32)
        xr -= float(np.nanmean(xr))
        xr_std = float(np.nanstd(xr))
        score = 0.0 if xr_std <= 1e-12 else float(abs(np.nanmean((xr / xr_std) * yr)))
        scores.append((str(col), score if math.isfinite(score) else 0.0))
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    keep = [name for name, _score in ranked[: int(top_n)]]
    selected = set(keep)
    rows = [
        {
            "fold": str(fold),
            "feature": name,
            "score": float(score),
            "rank": int(rank),
            "selected": name in selected,
            "feature_selection_status": "ok",
        }
        for rank, (name, score) in enumerate(ranked, start=1)
    ]
    return x_train.loc[:, keep], x_valid.loc[:, keep], keep, pd.DataFrame(rows)


def _feature_selection_family(name: str) -> str:
    text = str(name)
    low = text.lower()
    if "gmm" in low or "cluster" in low or "mahal" in low:
        return "ae_gmm_cluster"
    if "ae_" in low or "dae_" in low or "reconstruction" in low or "latent" in low:
        return "ae_gmm_autoencoder"
    if low.startswith("ctx_") or "state_" in low or "regime" in low:
        return "context_regime"
    if "orderbook" in low or "book" in low or "depth" in low:
        return "orderbook"
    if "funding" in low or "open_interest" in low or "_oi" in low or "oi_" in low:
        return "perp_oi_funding"
    if "btc" in low or "eth" in low or "market" in low or "cross" in low:
        return "cross_market"
    if "spread" in low or "liquidity" in low or "volume" in low:
        return "liquidity_volume"
    if "residual" in low or "leaf" in low or "base_score" in low:
        return "model_residual_leaf"
    return "config_feature"


def _mda_selection_objective(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    pred: pd.Series,
    fold: str,
) -> float:
    rows = [
        _selection_metrics(
            valid=valid,
            metrics=metrics,
            target=target,
            pred=pred,
            month=str(fold),
            top_frac=float(frac),
            trial_name="feature_selection_mda",
        )
        for frac in TOP_FRACS
    ]
    return _objective_from_rows(rows)


def _fit_lgbm_model(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    params: dict[str, Any],
    seed: int,
) -> Any:
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for MDA feature selection")
    model = LGBMRegressor(
        objective="regression",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        x_train.reset_index(drop=True),
        _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=_safe_numeric(w_train).fillna(1.0).to_numpy(dtype=np.float32),
    )
    return model


def _select_features_by_mda(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target: pd.DataFrame,
    *,
    top_n: int,
    fold: str,
    seed: int,
    max_train_rows: int = 60_000,
    max_valid_rows: int = 20_000,
    repeats: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    candidate_features = list(x_train.columns)
    candidate_family_counts = pd.Series([_feature_selection_family(c) for c in candidate_features]).value_counts()
    if int(top_n) > 0 and len(candidate_features) <= int(top_n):
        rows = [
            {
                "fold": str(fold),
                "feature": str(col),
                "feature_family": _feature_selection_family(str(col)),
                "score": float("nan"),
                "rank": int(i + 1),
                "selected": True,
                "feature_selection_method": "mda_topk_permutation",
                "feature_selection_status": "disabled_or_not_needed",
                "candidate_feature_count": int(len(candidate_features)),
                "candidate_family_count": int(candidate_family_counts.get(_feature_selection_family(str(col)), 0)),
            }
            for i, col in enumerate(candidate_features)
        ]
        return x_train, x_valid, candidate_features, pd.DataFrame(rows)
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        y = target["target_soft"] if "target_soft" in target else pd.Series(dtype=float)
        xtr, xva, keep, rows = _select_features_by_univariate(
            x_train,
            x_valid,
            y,
            top_n=int(top_n),
            fold=str(fold),
        )
        rows["feature_selection_method"] = "univariate_fallback_lightgbm_unavailable"
        rows["candidate_feature_count"] = int(len(candidate_features))
        return xtr, xva, keep, rows

    y = _safe_numeric(target["target_soft"]).replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna()
    if int(valid_y.sum()) < 1_000 or int(y.loc[valid_y].nunique(dropna=True)) < 3:
        xtr, xva, keep, rows = _select_features_by_univariate(
            x_train,
            x_valid,
            y,
            top_n=int(top_n),
            fold=str(fold),
        )
        rows["feature_selection_method"] = "univariate_fallback_insufficient_target_variation"
        rows["candidate_feature_count"] = int(len(candidate_features))
        return xtr, xva, keep, rows

    valid_positions = np.flatnonzero(valid_y.to_numpy(dtype=bool))
    split_at = int(max(500, round(0.80 * len(valid_positions))))
    split_at = min(split_at, max(len(valid_positions) - 500, 1))
    fit_pos = valid_positions[:split_at]
    eval_pos = valid_positions[split_at:]
    if len(fit_pos) < 500 or len(eval_pos) < 500:
        xtr, xva, keep, rows = _select_features_by_univariate(
            x_train,
            x_valid,
            y,
            top_n=int(top_n),
            fold=str(fold),
        )
        rows["feature_selection_method"] = "univariate_fallback_insufficient_mda_split"
        rows["candidate_feature_count"] = int(len(candidate_features))
        return xtr, xva, keep, rows

    fit_idx = fit_pos[_time_spread_cap_rows(len(fit_pos), int(max_train_rows))]
    eval_idx = eval_pos[_time_spread_cap_rows(len(eval_pos), int(max_valid_rows))]
    print(
        "[feature_selection] mda_start "
        f"fold={fold} candidates={len(candidate_features)} train_rows={len(fit_idx)} eval_rows={len(eval_idx)}",
        flush=True,
    )
    fs_target = target.reset_index(drop=True)
    fs_weights = _weight_series(
        frame=train_frame.reset_index(drop=True),
        metrics=train_metrics.reset_index(drop=True),
        target=fs_target,
        arm="W0_base",
    )
    params = {
        "n_estimators": 180,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 3.0,
    }
    model = _fit_lgbm_model(
        x_train=x_train.iloc[fit_idx].reset_index(drop=True),
        y_train=fs_target["target_soft"].iloc[fit_idx].reset_index(drop=True),
        w_train=fs_weights.iloc[fit_idx].reset_index(drop=True),
        params=params,
        seed=int(seed),
    )
    x_eval = x_train.iloc[eval_idx].reset_index(drop=True).astype(np.float32, copy=False)
    valid_eval = train_frame.iloc[eval_idx].reset_index(drop=True)
    metrics_eval = train_metrics.iloc[eval_idx].reset_index(drop=True)
    target_eval = fs_target.iloc[eval_idx].reset_index(drop=True)
    baseline_pred = pd.Series(model.predict(x_eval).astype(np.float32))
    baseline_objective = _mda_selection_objective(
        valid=valid_eval,
        metrics=metrics_eval,
        target=target_eval,
        pred=baseline_pred,
        fold=str(fold),
    )
    rng = np.random.default_rng(int(seed) + 23)
    records: list[dict[str, Any]] = []
    base_values = x_eval.to_numpy(dtype=np.float32, copy=True)
    for j, feature in enumerate(candidate_features):
        drops: list[float] = []
        for rep in range(max(1, int(repeats))):
            x_perm = base_values.copy()
            order = rng.permutation(x_perm.shape[0])
            x_perm[:, j] = x_perm[order, j]
            pred_perm = pd.Series(model.predict(pd.DataFrame(x_perm, columns=candidate_features)).astype(np.float32))
            perm_objective = _mda_selection_objective(
                valid=valid_eval,
                metrics=metrics_eval,
                target=target_eval,
                pred=pred_perm,
                fold=str(fold),
            )
            drops.append(float(baseline_objective - perm_objective))
        mean_drop = float(np.nanmean(drops)) if drops else 0.0
        records.append(
            {
                "fold": str(fold),
                "feature": str(feature),
                "feature_family": _feature_selection_family(str(feature)),
                "score": mean_drop if math.isfinite(mean_drop) else 0.0,
                "mda_mean": mean_drop if math.isfinite(mean_drop) else 0.0,
                "mda_std": float(np.nanstd(drops)) if len(drops) > 1 else 0.0,
                "mda_repeats": int(max(1, int(repeats))),
                "mda_baseline_objective": float(baseline_objective),
                "mda_eval_rows": int(len(eval_idx)),
                "mda_train_rows": int(len(fit_idx)),
                "feature_selection_method": "mda_topk_permutation",
                "feature_selection_status": "ok",
                "candidate_feature_count": int(len(candidate_features)),
                "candidate_family_count": int(candidate_family_counts.get(_feature_selection_family(str(feature)), 0)),
            }
            )
    ranked = sorted(records, key=lambda row: float(row["score"]), reverse=True)
    keep_n, selection_status, score_floor = _auto_mda_keep_count(ranked, int(top_n))
    keep = [str(row["feature"]) for row in ranked[:keep_n]]
    print(
        "[feature_selection] mda_done "
        f"fold={fold} selected={len(keep)} baseline_objective={float(baseline_objective):.6f}",
        flush=True,
    )
    selected = set(keep)
    rows = []
    for rank, row in enumerate(ranked, start=1):
        out = dict(row)
        out["rank"] = int(rank)
        out["selected"] = str(row["feature"]) in selected
        out["feature_selection_status"] = selection_status
        out["feature_selection_requested_top_n"] = int(top_n)
        out["feature_selection_auto_score_floor"] = float(score_floor)
        out["feature_selection_auto_selected_count"] = int(keep_n)
        rows.append(out)
    return x_train.loc[:, keep], x_valid.loc[:, keep], keep, pd.DataFrame(rows)


def _fit_predict_lgbm(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
) -> pd.Series:
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for this HPO")
    model = LGBMRegressor(
        objective="regression",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        x_train.reset_index(drop=True),
        _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=_safe_numeric(w_train).fillna(1.0).to_numpy(dtype=np.float32),
    )
    return pd.Series(model.predict(x_valid.reset_index(drop=True)).astype(np.float32))


def _selection_metrics(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    pred: pd.Series,
    month: str,
    top_frac: float,
    trial_name: str,
) -> dict[str, Any]:
    idx = _rank_top_indices(pred, top_frac)
    selected = valid.iloc[idx].reset_index(drop=True) if len(idx) else valid.iloc[:0].copy()
    sm = metrics.iloc[idx].reset_index(drop=True) if len(idx) else metrics.iloc[:0].copy()
    st = target.iloc[idx].reset_index(drop=True) if len(idx) else target.iloc[:0].copy()
    clean = _safe_numeric(st.get("target_hard", pd.Series(dtype=float))).fillna(0.0).clip(0.0, 1.0)
    net = _safe_numeric(sm.get("first_touch_net", sm.get("u_policy_net", pd.Series(dtype=float)))).fillna(0.0)
    cost = _safe_numeric(sm.get("round_trip_cost", pd.Series(0.0, index=sm.index))).fillna(0.0)
    gross = (net + cost).clip(lower=0.0)
    gross_denom = float(gross.sum())
    side = _safe_numeric(sm.get("side", pd.Series(1.0, index=sm.index))).fillna(1.0)
    return {
        "trial_name": str(trial_name),
        "month": str(month),
        "top_frac": float(top_frac),
        "rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "selected_symbols": int(selected["__symbol__"].nunique(dropna=True)) if "__symbol__" in selected.columns else 0,
        "clean_precision": _safe_mean(clean),
        "gross_ev_weighted_clean_precision": float((clean * gross).sum() / gross_denom) if gross_denom > 0.0 else float("nan"),
        "mean_first_touch_net": _safe_mean(net),
        "mean_first_touch_gross": _safe_mean(net + cost),
        "q10_first_touch_net": _safe_quantile(net, 0.10),
        "hit_first_touch_net": _safe_mean(net > 0.0),
        "first_touch_stop_rate": _safe_mean(sm.get("first_touch_stop", pd.Series(dtype=float))),
        "first_touch_timeout_rate": _safe_mean(sm.get("first_touch_timeout", pd.Series(dtype=float))),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(
            _safe_numeric(sm.get("first_touch_mae_to_sl", pd.Series(dtype=float))).ge(1.0)
        ),
        "p90_first_touch_mae_to_sl": _safe_quantile(sm.get("first_touch_mae_to_sl", pd.Series(dtype=float)), 0.90),
        "p90_first_touch_bar": _safe_quantile(sm.get("first_touch_bar", pd.Series(dtype=float)), 0.90),
        "bad_mae_1r_rate": _safe_mean(_safe_numeric(sm.get("mae_norm", pd.Series(dtype=float))).ge(1.0)),
        "long_share": _safe_mean(side.ge(0.0)) if len(side) else float("nan"),
        "short_share": _safe_mean(side.lt(0.0)) if len(side) else float("nan"),
        "top_symbol_share": float(selected["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
        if len(selected) and "__symbol__" in selected.columns
        else float("nan"),
    }


def _prepare_folds(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    feature_selection_top_n: int,
    feature_selection_target_mode: str,
    feature_selection_method: str,
    max_oos_model_age_days: int,
    payload_max_train_rows: int,
    fold_cache_dir: Path | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = _load_labels(labels_path).reset_index(drop=True)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        if new_cols:
            frame = pd.concat(
                [frame, feature_matrix.loc[:, new_cols].reset_index(drop=True).astype(np.float32, copy=False)],
                axis=1,
                copy=False,
            )
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame)).reset_index(drop=True)
    features = _feature_columns(frame)
    fold_frame_columns = _fold_frame_columns(frame)
    if fold_cache_dir is not None:
        fold_cache_dir.mkdir(parents=True, exist_ok=True)
    all_months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    if not months:
        months = all_months[1:]
    folds: list[dict[str, Any]] = []
    ts_utc = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    validation_windows: list[dict[str, Any]] = []
    periods = sorted(pd.Period(month) for month in months)
    contiguous_months = bool(periods) and periods == [periods[0] + i for i in range(len(periods))]
    if int(max_oos_model_age_days) > 0 and contiguous_months:
        scope_start = pd.Timestamp(periods[0].start_time, tz="UTC")
        scope_end = pd.Timestamp((periods[-1] + 1).start_time, tz="UTC")
        start = scope_start
        step = pd.Timedelta(days=int(max_oos_model_age_days))
        while start < scope_end:
            end = min(start + step, scope_end)
            validation_windows.append(
                {
                    "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                    "month": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                    "valid_start": start,
                    "valid_end": end,
                }
            )
            start = end
    else:
        for month in months:
            month_start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
            month_end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
            if int(max_oos_model_age_days) > 0:
                start = month_start
                step = pd.Timedelta(days=int(max_oos_model_age_days))
                while start < month_end:
                    end = min(start + step, month_end)
                    validation_windows.append(
                        {
                            "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                            "month": str(month),
                            "valid_start": start,
                            "valid_end": end,
                        }
                    )
                    start = end
            else:
                validation_windows.append(
                    {
                        "fold": str(month),
                        "month": str(month),
                        "valid_start": month_start,
                        "valid_end": month_end,
                    }
                )
    global_selected_features: list[str] | None = None
    global_feature_selection_df: pd.DataFrame | None = None
    eligible_windows: list[dict[str, Any]] = []
    for window in validation_windows:
        train_rows = int(ts_utc.lt(window["valid_start"]).sum())
        valid_rows = int((ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])).sum())
        if train_rows < 500 or valid_rows < 100:
            continue
        enriched = dict(window)
        enriched["train_rows_estimate"] = int(train_rows)
        enriched["valid_rows_estimate"] = int(valid_rows)
        eligible_windows.append(enriched)
    fs_window_fold = None
    if eligible_windows:
        fs_window = max(eligible_windows, key=lambda w: (int(w["train_rows_estimate"]), int(w["valid_rows_estimate"])))
        fs_window_fold = str(fs_window["fold"])
        ordered_windows = [fs_window] + [w for w in eligible_windows if str(w["fold"]) != fs_window_fold]
    else:
        ordered_windows = []
    for fold_id, window in enumerate(ordered_windows):
        print(
            "[prepare_folds] start "
            f"{window['fold']} train_est={int(window.get('train_rows_estimate', 0))} "
            f"valid_est={int(window.get('valid_rows_estimate', 0))}",
            flush=True,
        )
        valid_mask = ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])
        train_mask = ts_utc.lt(window["valid_start"])
        train_full_uncapped = frame.loc[train_mask]
        valid_full = frame.loc[valid_mask]
        train_metrics_uncapped = metrics.loc[train_mask].reset_index(drop=True)
        if int(payload_max_train_rows) > 0 and len(train_full_uncapped) > int(payload_max_train_rows):
            payload_idx = _time_spread_cap_rows(len(train_full_uncapped), int(payload_max_train_rows))
            train_full = train_full_uncapped.iloc[payload_idx]
            train_metrics = train_metrics_uncapped.iloc[payload_idx].reset_index(drop=True)
        else:
            payload_idx = np.arange(len(train_full_uncapped), dtype=np.int64)
            train_full = train_full_uncapped
            train_metrics = train_metrics_uncapped
        train = train_full.loc[:, fold_frame_columns].reset_index(drop=True)
        valid = valid_full.loc[:, fold_frame_columns].reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].reset_index(drop=True)
        print(
            "[prepare_folds] build_matrix "
            f"{window['fold']} feature_count={len(features)}",
            flush=True,
        )
        x_train = train_full.loc[:, features].replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        x_valid = valid_full.loc[:, features].replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        med_idx = _time_spread_cap_rows(len(x_train), 300_000)
        med = x_train.iloc[med_idx].median(numeric_only=True)
        x_train = x_train.fillna(med).fillna(0.0).astype(np.float32, copy=False)
        x_valid = x_valid.fillna(med).fillna(0.0).astype(np.float32, copy=False)
        print(
            "[prepare_folds] ae_gmm_start "
            f"{window['fold']} train_rows={len(x_train)} valid_rows={len(x_valid)}",
            flush=True,
        )
        x_train, x_valid, generated_features, ae_diag = _append_fold_ae_gmm_state_features(
            x_train=x_train,
            x_valid=x_valid,
            train_frame=train,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            enabled=bool(include_ae_gmm_state_features),
            max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            ae_max_iter=int(ae_gmm_state_feature_max_iter),
            random_state=int(seed) + fold_id,
        )
        if (
            bool(include_ae_gmm_state_features)
            and not generated_features
            and str(ae_diag.get("ae_gmm_state_feature_status", "")).startswith("no_valid_gmm_config")
        ):
            x_train, x_valid, generated_features, ae_diag = _append_single_side_ae_gmm_state_features(
                x_train=x_train,
                x_valid=x_valid,
                train_frame=train,
                train_metrics=train_metrics,
                max_train_rows=int(ae_gmm_state_feature_max_train_rows),
                ae_max_iter=int(ae_gmm_state_feature_max_iter),
                random_state=int(seed) + fold_id + 90_000,
            )
        print(
            "[prepare_folds] ae_gmm_done "
            f"{window['fold']} generated={len(generated_features)} status={ae_diag.get('ae_gmm_state_feature_status')}",
            flush=True,
        )
        if global_selected_features is None:
            fs_target_frame = _target_from_frame(train, train_metrics, target_mode=str(feature_selection_target_mode))
            fs_fold_name = f"largest_train_before_{window['valid_start']:%Y-%m-%d}"
            if str(feature_selection_method).strip().lower() == "mda":
                x_train, x_valid, selected_features_fold, feature_selection_df = _select_features_by_mda(
                    x_train,
                    x_valid,
                    train,
                    train_metrics,
                    fs_target_frame,
                    top_n=int(feature_selection_top_n),
                    fold=fs_fold_name,
                    seed=int(seed) + fold_id,
                )
            else:
                x_train, x_valid, selected_features_fold, feature_selection_df = _select_features_by_univariate(
                    x_train,
                    x_valid,
                    fs_target_frame["target_soft"],
                    top_n=int(feature_selection_top_n),
                    fold=fs_fold_name,
                )
                feature_selection_df["feature_selection_method"] = "univariate_rank_corr"
            global_selected_features = list(selected_features_fold)
            global_feature_selection_df = feature_selection_df
        else:
            x_train = x_train.reindex(columns=global_selected_features, fill_value=0.0).astype(np.float32, copy=False)
            x_valid = x_valid.reindex(columns=global_selected_features, fill_value=0.0).astype(np.float32, copy=False)
            selected_features_fold = list(global_selected_features)
            feature_selection_df = pd.DataFrame(
                columns=["fold", "feature", "score", "rank", "selected", "feature_selection_status"]
            )
        fold_payload = {
                "fold": str(window["fold"]),
                "month": str(window["month"]),
                "valid_start": window["valid_start"],
                "valid_end": window["valid_end"],
                "max_oos_model_age_days": int(max_oos_model_age_days),
                "train_rows_uncapped": int(len(train_full_uncapped)),
                "train_rows_payload": int(len(train_full)),
                "payload_train_sampling": (
                    "beginning_middle_end_time_spread"
                    if int(payload_max_train_rows) > 0 and len(train_full_uncapped) > int(payload_max_train_rows)
                    else "full_train_rows"
                ),
                "train": train,
                "valid": valid,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "x_train": x_train,
                "x_valid": x_valid,
                "ae_gmm_generated_features": int(len(generated_features)),
                "ae_gmm_status": ae_diag.get("ae_gmm_state_feature_status"),
                "selected_features": selected_features_fold,
                "feature_selection": feature_selection_df,
            }
        folds.append(_write_fold_payload(fold_payload, fold_cache_dir) if fold_cache_dir is not None else fold_payload)
        print(
            "[prepare_folds] cached "
            f"{window['fold']} train={int(len(train_full))}/{int(len(train_full_uncapped))} valid={int(len(valid_full))} "
            f"features={int(x_train.shape[1])} ae_gmm={int(len(generated_features))}",
            flush=True,
        )
        del train_full_uncapped, train_full, valid_full, train, valid, train_metrics, train_metrics_uncapped, valid_metrics, x_train, x_valid
    selected_by_fold = {str(fold["fold"]): list(fold["selected_features"]) for fold in folds}
    selected_sets = [set(features) for features in selected_by_fold.values()]
    selected_union = sorted(set().union(*selected_sets)) if selected_sets else []
    selected_intersection = sorted(set.intersection(*selected_sets)) if selected_sets else []
    manifest = {
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "feature_count": int(len(features)),
        "feature_store": feature_report,
        "fold_payload_storage": "parquet_cache" if fold_cache_dir is not None else "memory",
        "fold_cache_dir": str(fold_cache_dir) if fold_cache_dir is not None else None,
        "fold_cache_feature_dtype_on_disk": "float16_clipped_to_finite_range",
        "fold_frame_column_count": int(len(fold_frame_columns)),
        "fold_months": sorted({str(fold["month"]) for fold in folds}),
        "fold_windows": [
            {
                "fold": str(fold["fold"]),
                "month": str(fold["month"]),
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "max_oos_model_age_days": int(fold["max_oos_model_age_days"]),
                "train_rows_uncapped": int(fold.get("train_rows_uncapped", fold.get("train_rows", 0))),
                "train_rows_payload": int(fold.get("train_rows_payload", fold.get("train_rows", 0))),
                "payload_train_sampling": str(fold.get("payload_train_sampling", "full_train_rows")),
            }
            for fold in folds
        ],
        "fold_count": int(len(folds)),
        "max_oos_model_age_days": int(max_oos_model_age_days),
        "payload_max_train_rows": int(payload_max_train_rows),
        "validation_windowing": (
            "continuous_rolling_max_age_windows"
            if int(max_oos_model_age_days) > 0 and contiguous_months
            else "calendar_month_windows"
        ),
        "oos_model_age_contract": (
            "validation windows are capped by --max-oos-model-age-days"
            if int(max_oos_model_age_days) > 0
            else "month_forward_legacy"
        ),
        "ae_gmm_generated_features_by_fold": [fold["ae_gmm_generated_features"] for fold in folds],
        "feature_selection_scope": "single_global_largest_train_window",
        "feature_selection_calibration_fold": fs_window_fold,
        "feature_selection_global_calibration_note": (
            "features are selected once on the largest train fold and reused for all OOS scoring folds"
        ),
        "feature_selection_method": str(feature_selection_method),
        "feature_selection_top_n": int(feature_selection_top_n),
        "feature_selection_target_mode": str(feature_selection_target_mode),
        "global_feature_selection_fold": (
            str(global_feature_selection_df["fold"].iloc[0])
            if global_feature_selection_df is not None and not global_feature_selection_df.empty
            else None
        ),
        "selected_features_by_fold": selected_by_fold,
        "selected_feature_union": selected_union,
        "selected_feature_intersection": selected_intersection,
        "selected_feature_union_count": int(len(selected_union)),
        "selected_feature_intersection_count": int(len(selected_intersection)),
    }
    return folds, manifest


def _suggest_params(trial: Any, rng: np.random.Generator) -> dict[str, Any]:
    if trial is None:
        return {
            "n_estimators": int(rng.integers(120, 321)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.015), np.log(0.08)))),
            "num_leaves": int(rng.choice([15, 23, 31, 47, 63])),
            "max_depth": int(rng.choice([-1, 4, 5, 6, 8])),
            "min_child_samples": int(rng.integers(25, 101)),
            "subsample": float(rng.uniform(0.65, 0.95)),
            "colsample_bytree": float(rng.uniform(0.55, 0.95)),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(3.0)))),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.3), np.log(12.0)))),
            "target_mode": str(rng.choice(TARGET_MODES)),
            "weight_arm": str(rng.choice(["W0_base", "W7_timestamp_balanced", "W8_combined_conservative", "W12_tail_timestamp_balanced"])),
        }
    return {
        "n_estimators": int(trial.suggest_int("n_estimators", 120, 360)),
        "learning_rate": float(trial.suggest_float("learning_rate", 0.015, 0.08, log=True)),
        "num_leaves": int(trial.suggest_categorical("num_leaves", [15, 23, 31, 47, 63])),
        "max_depth": int(trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 8])),
        "min_child_samples": int(trial.suggest_int("min_child_samples", 25, 110)),
        "subsample": float(trial.suggest_float("subsample", 0.65, 0.95)),
        "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.55, 0.95)),
        "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True)),
        "reg_lambda": float(trial.suggest_float("reg_lambda", 0.3, 12.0, log=True)),
        "target_mode": str(trial.suggest_categorical("target_mode", list(TARGET_MODES))),
        "weight_arm": str(
            trial.suggest_categorical(
                "weight_arm",
                ["W0_base", "W7_timestamp_balanced", "W8_combined_conservative", "W12_tail_timestamp_balanced"],
            )
        ),
    }


def _objective_from_rows(rows: list[dict[str, Any]]) -> float:
    df = pd.DataFrame(rows)
    if df.empty:
        return float("-inf")
    def m(frac: float, col: str) -> float:
        return _safe_mean(df.loc[df["top_frac"].eq(frac), col])

    top10 = m(0.10, "gross_ev_weighted_clean_precision")
    top20 = m(0.20, "gross_ev_weighted_clean_precision")
    top30 = m(0.30, "gross_ev_weighted_clean_precision")
    clean30 = m(0.30, "clean_precision")
    net30 = m(0.30, "mean_first_touch_net")
    q10_net30 = m(0.30, "q10_first_touch_net")
    timeout30 = m(0.30, "first_touch_timeout_rate")
    bad30 = m(0.30, "first_touch_bad_mae_to_sl_rate")
    objective = (
        1.00 * top30
        + 0.55 * top20
        + 0.30 * top10
        + 0.25 * clean30
        + 10.00 * net30
        + 3.00 * min(q10_net30, 0.0)
        - 0.20 * timeout30
        - 0.12 * bad30
    )
    return float(objective) if math.isfinite(float(objective)) else float("-inf")


def _run_trial(
    *,
    folds: list[dict[str, Any]],
    params: dict[str, Any],
    trial_number: int,
    max_train_rows: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fold_id, fold in enumerate(folds):
        payload = _load_fold_payload(fold)
        train_target = _target_from_frame(
            payload["train"],
            payload["train_metrics"],
            target_mode=str(params["target_mode"]),
        )
        valid_target = _target_from_frame(
            payload["valid"],
            payload["valid_metrics"],
            target_mode=str(params["target_mode"]),
        )
        if str(params["weight_arm"]) not in WEIGHT_ARMS:
            raise ValueError(f"Unknown weight arm: {params['weight_arm']}")
        weights = _weight_series(
            frame=payload["train"],
            metrics=payload["train_metrics"],
            target=train_target,
            arm=str(params["weight_arm"]),
        )
        idx = _time_spread_cap_rows(len(payload["x_train"]), int(max_train_rows))
        pred = _fit_predict_lgbm(
            x_train=payload["x_train"].iloc[idx].reset_index(drop=True),
            y_train=train_target["target_soft"].iloc[idx].reset_index(drop=True),
            w_train=weights.iloc[idx].reset_index(drop=True),
            x_valid=payload["x_valid"],
            params=params,
            seed=int(seed) + 1000 * int(trial_number) + fold_id,
        )
        trial_name = f"trial_{int(trial_number):03d}"
        for frac in TOP_FRACS:
            metric = _selection_metrics(
                valid=payload["valid"],
                metrics=payload["valid_metrics"],
                target=valid_target,
                pred=pred,
                month=str(fold["fold"]),
                top_frac=float(frac),
                trial_name=trial_name,
            )
            metric.update(
                {
                    "trial_number": int(trial_number),
                    "calendar_month": str(fold["month"]),
                    "valid_start": fold["valid_start"],
                    "valid_end": fold["valid_end"],
                    "max_oos_model_age_days": int(fold["max_oos_model_age_days"]),
                    **params,
                }
            )
            rows.append(metric)
        diagnostics.append(
            {
                "trial_number": int(trial_number),
                "month": str(fold["fold"]),
                "calendar_month": str(fold["month"]),
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "max_oos_model_age_days": int(fold["max_oos_model_age_days"]),
                "train_rows": int(len(idx)),
                "train_rows_uncapped": int(len(payload["x_train"])),
                "valid_rows": int(len(payload["x_valid"])),
                "target_train_mean": _safe_mean(train_target["target_soft"]),
                "target_valid_mean": _safe_mean(valid_target["target_soft"]),
                "weight_mean": _safe_mean(weights),
                "weight_effective_frac": _effective_sample_size(weights) / max(float(len(weights)), 1.0),
                "ae_gmm_generated_features": int(fold["ae_gmm_generated_features"]),
                "ae_gmm_status": fold.get("ae_gmm_status"),
                **params,
            }
        )
        del payload, train_target, valid_target, weights, pred
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "trial_number": int(trial_number),
        "trial_name": f"trial_{int(trial_number):03d}",
        "objective": _objective_from_rows(rows),
        "folds": int(len(folds)),
        **params,
    }
    for frac in TOP_FRACS:
        tag = f"top{int(round(frac * 100))}"
        subset = df[df["top_frac"].eq(frac)]
        for col in (
            "gross_ev_weighted_clean_precision",
            "clean_precision",
            "mean_first_touch_net",
            "mean_first_touch_gross",
            "q10_first_touch_net",
            "hit_first_touch_net",
            "first_touch_stop_rate",
            "first_touch_timeout_rate",
            "first_touch_bad_mae_to_sl_rate",
            "bad_mae_1r_rate",
            "selected_rows",
            "selected_symbols",
        ):
            summary[f"mean_{tag}_{col}"] = _safe_mean(subset[col]) if col in subset else float("nan")
    return summary, rows, diagnostics


def _best_params_from_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "target_mode",
        "weight_arm",
    )
    out = {key: row[key] for key in keys if key in row}
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        if key in out:
            out[key] = int(float(out[key]))
    for key in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        if key in out:
            out[key] = float(out[key])
    if "target_mode" in out:
        out["target_mode"] = str(out["target_mode"])
    if "weight_arm" in out:
        out["weight_arm"] = str(out["weight_arm"])
    return out


def _score_best_oos_ledger(
    *,
    folds: list[dict[str, Any]],
    params: dict[str, Any],
    trial_number: int,
    max_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold_id, fold in enumerate(folds):
        payload = _load_fold_payload(fold)
        train_target = _target_from_frame(
            payload["train"],
            payload["train_metrics"],
            target_mode=str(params["target_mode"]),
        )
        weights = _weight_series(
            frame=payload["train"],
            metrics=payload["train_metrics"],
            target=train_target,
            arm=str(params["weight_arm"]),
        )
        idx = _time_spread_cap_rows(len(payload["x_train"]), int(max_train_rows))
        pred = _fit_predict_lgbm(
            x_train=payload["x_train"].iloc[idx].reset_index(drop=True),
            y_train=train_target["target_soft"].iloc[idx].reset_index(drop=True),
            w_train=weights.iloc[idx].reset_index(drop=True),
            x_valid=payload["x_valid"],
            params=params,
            seed=int(seed) + 1000 * int(trial_number) + fold_id,
        )
        scored = payload["valid"].copy()
        scored["score"] = pred.to_numpy(dtype=np.float32, copy=False)
        scored["oos_fold"] = str(fold["fold"])
        ts_scored = pd.to_datetime(scored["__ts__"], errors="coerce", utc=True)
        scored["fold_window"] = str(fold["month"])
        scored["calendar_month"] = ts_scored.dt.strftime("%Y-%m")
        scored["month"] = scored["calendar_month"]
        scored["valid_start"] = fold["valid_start"]
        scored["valid_end"] = fold["valid_end"]
        scored["max_oos_model_age_days"] = int(fold["max_oos_model_age_days"])
        scored["base_model_trial_number"] = int(trial_number)
        scored["base_model_target_mode"] = str(params["target_mode"])
        scored["base_model_weight_arm"] = str(params["weight_arm"])
        side = pd.to_numeric(scored.get("__side__", scored.get("side", np.nan)), errors="coerce")
        if "side_name" not in scored.columns:
            scored["side_name"] = np.where(side.to_numpy(dtype=np.float64, copy=False) < 0.0, "short", "long")
        for frac in TOP_FRACS:
            col = f"selected_top{int(round(frac * 100))}"
            mask = np.zeros(len(scored), dtype=bool)
            idx_top = _rank_top_indices(pred, float(frac))
            if len(idx_top):
                mask[idx_top] = True
            scored[col] = mask
        parts.append(scored)
        del payload, train_target, weights, pred, scored
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], errors="coerce")
    out = out.sort_values(["__ts__", "__symbol__", "side_name"], kind="mergesort").reset_index(drop=True)
    return out


def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "materialized_trailing_label_topk_lgbm_hpo.md"

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
        "trial_name",
        "objective",
        "target_mode",
        "weight_arm",
        "mean_top10_gross_ev_weighted_clean_precision",
        "mean_top20_gross_ev_weighted_clean_precision",
        "mean_top30_gross_ev_weighted_clean_precision",
        "mean_top10_clean_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_q10_first_touch_net",
        "mean_top10_first_touch_timeout_rate",
        "mean_top10_first_touch_bad_mae_to_sl_rate",
        "mean_top10_selected_rows",
        "mean_top10_selected_symbols",
        "num_leaves",
        "min_child_samples",
        "learning_rate",
        "reg_lambda",
    ]
    best = str(summary.iloc[0]["trial_name"]) if not summary.empty else ""
    fold_cols = [
        "trial_name",
        "month",
        "top_frac",
        "gross_ev_weighted_clean_precision",
        "clean_precision",
        "mean_first_touch_net",
        "q10_first_touch_net",
        "hit_first_touch_net",
        "first_touch_timeout_rate",
        "first_touch_bad_mae_to_sl_rate",
        "selected_rows",
        "selected_symbols",
    ]
    lines = [
        "# Materialized Trailing Label Top-k LGBM HPO",
        "",
        "Scope: month-forward base-model HPO against already materialized trailing-profit labels. Primary metrics are top10/top20/top30 clean precision and gross-EV-weighted clean precision; net EV and path-risk rates are diagnostics.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Months: `{', '.join(manifest['fold_months'])}`",
        f"Features: `{manifest['feature_count']}` plus AE/GMM generated features `{manifest['ae_gmm_generated_features_by_fold']}` by fold.",
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
        table(folds[folds["trial_name"].eq(best)], fold_cols),
        "",
        "## Outputs",
        "",
        f"- Trial summary: `{manifest['outputs']['trial_summary']}`",
        f"- Fold metrics: `{manifest['outputs']['fold_metrics']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Best params: `{manifest['outputs']['best_params']}`",
        f"- Best OOS scored ledger: `{manifest['outputs']['best_oos_scored_ledger']}`",
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
    max_feature_store_features: int | None,
    max_train_rows: int,
    hpo_max_train_rows: int,
    n_trials: int,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    feature_selection_top_n: int,
    feature_selection_target_mode: str,
    feature_selection_method: str,
    max_oos_model_age_days: int,
    fixed_params_json: Path | None = None,
    rerun_hpo: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_cache_dir = output_dir / "_fold_cache"
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        months=months,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
        feature_selection_top_n=feature_selection_top_n,
        feature_selection_target_mode=feature_selection_target_mode,
        feature_selection_method=feature_selection_method,
        max_oos_model_age_days=int(max_oos_model_age_days),
        payload_max_train_rows=int(max_train_rows),
        fold_cache_dir=fold_cache_dir,
        seed=seed,
    )
    if not folds:
        raise RuntimeError("No valid OOS folds prepared")
    hpo_folds = [max(folds, key=lambda fold: int(fold.get("train_rows", 0)))]
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    try:
        import optuna
    except Exception:
        optuna = None

    def evaluate(params: dict[str, Any], trial_number: int) -> float:
        print(
            "[hpo] evaluating "
            f"trial={int(trial_number)} target={params.get('target_mode')} weight={params.get('weight_arm')}",
            flush=True,
        )
        summary, rows, diag = _run_trial(
            folds=hpo_folds,
            params=params,
            trial_number=trial_number,
            max_train_rows=int(hpo_max_train_rows),
            seed=seed,
        )
        summaries.append(summary)
        fold_rows.extend(rows)
        diagnostics.extend(diag)
        print(
            "[hpo] completed "
            f"trial={int(trial_number)} objective={float(summary['objective']):.6f}",
            flush=True,
        )
        return float(summary["objective"])

    trial_counter = 0
    fixed_params_path = Path(fixed_params_json) if fixed_params_json is not None else None
    if fixed_params_path is not None and not bool(rerun_hpo):
        fixed_params = _load_fixed_params(fixed_params_path)
        fixed_trial_number = int(fixed_params.pop("_fixed_trial_number", trial_counter))
        evaluate(fixed_params, fixed_trial_number)
        trial_counter += 1
    else:
        baselines = [
            {
                "n_estimators": 180,
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 45,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.05,
                "reg_lambda": 2.0,
                "target_mode": "policy_soft",
                "weight_arm": "W0_base",
            },
            {
                "n_estimators": 220,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 55,
                "subsample": 0.82,
                "colsample_bytree": 0.78,
                "reg_alpha": 0.10,
                "reg_lambda": 4.0,
                "target_mode": "exec_guarded_policy",
                "weight_arm": "W8_combined_conservative",
            },
        ]
        for params in baselines:
            evaluate(dict(params), trial_counter)
            trial_counter += 1
    if (fixed_params_path is None or bool(rerun_hpo)) and optuna is not None and int(n_trials) > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            nonlocal trial_counter
            params = _suggest_params(trial, rng)
            value = evaluate(params, trial_counter)
            if summaries:
                for key, val in summaries[-1].items():
                    if isinstance(val, (int, float)) and math.isfinite(float(val)):
                        trial.set_user_attr(key, float(val))
            trial_counter += 1
            return value

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    elif fixed_params_path is None or bool(rerun_hpo):
        for _ in range(int(n_trials)):
            evaluate(_suggest_params(None, rng), trial_counter)
            trial_counter += 1

    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1, dtype=np.int32))
    folds_df = pd.DataFrame(fold_rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    feature_selection_df = (
        pd.concat([fold["feature_selection"] for fold in folds], ignore_index=True)
        if folds
        else pd.DataFrame(columns=["fold", "feature", "score", "rank", "selected", "feature_selection_status"])
    )
    paths = {
        "trial_summary": output_dir / "topk_lgbm_hpo_trials.csv",
        "fold_metrics": output_dir / "topk_lgbm_hpo_folds.csv",
        "diagnostics": output_dir / "topk_lgbm_hpo_diagnostics.csv",
        "feature_selection": output_dir / "topk_lgbm_feature_selection_by_fold.csv",
        "best_oos_scored_ledger": output_dir / "best_oos_scored_ledger.parquet",
        "best_params": output_dir / "topk_lgbm_hpo_best.json",
        "manifest": output_dir / "manifest.json",
    }
    summary_df.to_csv(paths["trial_summary"], index=False)
    folds_df.to_csv(paths["fold_metrics"], index=False)
    diagnostics_df.to_csv(paths["diagnostics"], index=False)
    feature_selection_df.to_csv(paths["feature_selection"], index=False)
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    if best:
        best_trial_number = int(float(best.get("trial_number", 0)))
        best_params_for_scoring = _best_params_from_summary_row(best)
        best_ledger = _score_best_oos_ledger(
            folds=folds,
            params=best_params_for_scoring,
            trial_number=best_trial_number,
            max_train_rows=int(max_train_rows),
            seed=int(seed),
        )
        best_ledger.to_parquet(paths["best_oos_scored_ledger"], index=False)
    else:
        pd.DataFrame().to_parquet(paths["best_oos_scored_ledger"], index=False)
    paths["best_params"].write_text(json.dumps(_json_safe(best), indent=2), encoding="utf-8")
    manifest.update(
        {
            "scope": "materialized_trailing_label_topk_lgbm_hpo",
            "labels_path": str(labels_path),
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "output_dir": str(output_dir),
            "max_feature_store_features": max_feature_store_features,
            "max_train_rows": int(max_train_rows),
            "hpo_scope": "single_largest_train_fold",
            "hpo_sampling": "beginning_middle_end_time_spread",
            "hpo_global_calibration_note": (
                "parameters are selected once on the largest train fold and reused for all OOS scoring folds"
            ),
            "hpo_calibration_fold": str(hpo_folds[0].get("fold")) if hpo_folds else None,
            "hpo_calibration_train_rows": int(hpo_folds[0].get("train_rows", 0)) if hpo_folds else 0,
            "hpo_max_train_rows": int(hpo_max_train_rows),
            "n_trials_requested": int(n_trials),
            "fixed_params_json": str(fixed_params_path) if fixed_params_path is not None else None,
            "rerun_hpo": bool(rerun_hpo),
            "search_mode": "fixed_params_eval" if fixed_params_path is not None and not bool(rerun_hpo) else "hpo",
            "seed": int(seed),
            "top_fracs": list(TOP_FRACS),
            "target_modes": list(TARGET_MODES),
            "primary_objective": "top30_first_gross_ev_weighted_clean_precision_plus_net_ev_penalties",
            "outputs": {key: str(value) for key, value in paths.items()},
        }
    )
    report = _write_report(output_dir, summary_df, folds_df, manifest)
    manifest["outputs"]["report"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument(
        "--hpo-max-train-rows",
        type=int,
        default=300_000,
        help="Rows used during one-shot HPO, sampled from beginning/middle/end of the largest calibration fold. Final OOS scoring still uses --max-train-rows.",
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-params-json",
        type=Path,
        default=DEFAULT_FIXED_PARAMS_JSON,
        help="Evaluate this fixed parameter recipe instead of launching HPO. Pass an empty path with --rerun-hpo to search.",
    )
    parser.add_argument(
        "--rerun-hpo",
        action="store_true",
        help="Ignore --fixed-params-json for search control and run the baseline/Optuna HPO arms.",
    )
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER)
    parser.add_argument("--feature-selection-top-n", type=int, default=0)
    parser.add_argument("--feature-selection-target-mode", choices=TARGET_MODES, default="time_decay_policy")
    parser.add_argument(
        "--feature-selection-method",
        choices=("univariate", "mda"),
        default="mda",
        help="Global first-window feature selector. 'mda' uses top-k objective permutation importance.",
    )
    parser.add_argument(
        "--max-oos-model-age-days",
        type=int,
        default=0,
        help="When positive, split each requested OOS month into windows no longer than this many days.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_hpo(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, ()),
        max_feature_store_features=args.max_feature_store_features,
        max_train_rows=int(args.max_train_rows),
        hpo_max_train_rows=int(args.hpo_max_train_rows),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        feature_selection_top_n=int(args.feature_selection_top_n),
        feature_selection_target_mode=str(args.feature_selection_target_mode),
        feature_selection_method=str(args.feature_selection_method),
        max_oos_model_age_days=int(args.max_oos_model_age_days),
        fixed_params_json=args.fixed_params_json if str(args.fixed_params_json).strip() else None,
        rerun_hpo=bool(args.rerun_hpo),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
