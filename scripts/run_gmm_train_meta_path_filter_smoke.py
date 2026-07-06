#!/usr/bin/env python3
"""Small train_meta-style path filter smoke for GMM base candidate streams."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

try:
    from lightgbm import LGBMClassifier

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional smoke dependency
    LGBMClassifier = None
    _LIGHTGBM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gmm_train_base_learnability_smoke import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    _active_readiness_rows,
    _parse_int_csv,
)
from scripts.run_label_feature_store_model_smoke import run_smoke  # noqa: E402


DEFAULT_OUTPUT_SUBDIR = "gmm_train_meta_path_filter_smoke"
DEFAULT_BASE_STREAM_SUBDIR = "base_candidate_streams"
DEFAULT_CANDIDATE_STREAMS = (
    "s8_lgbm_utility_ranker_stageA_rerank_side_cap_70",
    "s16_lgbm_utility_blended_risk_lgbm_bad_cap_45_timeout_cap_15_stageA_rerank_side_cap_70",
    (
        "s18_lgbm_path_dirty_positive_aware_ts_gate"
        "_clean_ts_min_50_dirty_ts_max_55_timeout_ts_max_70"
        "_raw_dirty_cap_60_raw_timeout_cap_30_stageA_rerank_side_cap_70"
    ),
    (
        "s20_lgbm_exec_clean_ts_gate"
        "_clean_ts_min_55_dirty_ts_max_45_bad_ts_max_55_timeout_ts_max_55"
        "_raw_dirty_cap_50_raw_bad_cap_55_raw_timeout_cap_18"
        "_final_frac_020_stageA_rerank_side_cap_70"
    ),
    (
        "s20_lgbm_exec_clean_strict_ts_gate"
        "_clean_ts_min_60_dirty_ts_max_40_bad_ts_max_50_timeout_ts_max_50"
        "_raw_dirty_cap_45_raw_bad_cap_50_raw_timeout_cap_15"
        "_final_frac_020_stageA_rerank_side_cap_70"
    ),
    (
        "s20_lgbm_exec_clean_contrast_ts_gate"
        "_clean_ts_min_55_dirty_ts_max_45_bad_ts_max_55_timeout_ts_max_55"
        "_raw_dirty_cap_50_raw_bad_cap_55_raw_timeout_cap_18"
        "_clean_dirty_cap_50_final_frac_020_stageA_rerank_side_cap_70"
    ),
    (
        "s21_lgbm_positive_clean_exec_ts_gate"
        "_pos_clean_ts_min_55_raw_clean_min_25_dirty_ts_max_45_bad_ts_max_55"
        "_timeout_ts_max_55_raw_dirty_cap_50_raw_bad_cap_55_raw_timeout_cap_18"
        "_final_frac_020_stageA_rerank_side_cap_70"
    ),
)
DEFAULT_KEEP_FRACS = (0.50, 0.60, 0.70, 0.80)
DEFAULT_THRESHOLDS = {
    "min_mean_u": 0.0,
    "min_worst_month_mean_u": 0.0,
    "max_bad_mae_1r_rate": 0.50,
    "max_timeout_rate": 0.12,
    "max_month_bad_mae_1r_rate": 0.50,
    "max_month_timeout_rate": 0.12,
    "min_final_oracle_recall": 0.02,
    "max_selected_side_share": 0.70,
    "min_selected_rows": 10,
}
META_CONTEXT_FEATURE_POLICY = str(
    os.environ.get("EPM_META_CONTEXT_FEATURE_POLICY", "off")
).strip().lower()
META_CONTEXT_FEATURE_BLOCKS = tuple(
    part.strip().lower()
    for part in str(os.environ.get("EPM_META_CONTEXT_FEATURE_BLOCKS", "all")).split(",")
    if part.strip()
)
META_CONTRAST_MIN_TRAIN_ROWS = int(os.environ.get("EPM_META_CONTRAST_MIN_TRAIN_ROWS", "120"))
META_CONTEXT_AE_GMM_TOKENS = (
    "gmm",
    "cluster",
    "archetype",
    "posterior",
    "mahalanobis",
    "reconstruction",
    "latent",
)


def _meta_context_feature_blocks(name: str) -> set[str]:
    lower = str(name).lower()
    out: set[str] = set()
    if lower.startswith("ctx_long_"):
        out.add("long")
    elif lower.startswith("ctx_short_"):
        out.add("short")
    elif any(token in lower for token in META_CONTEXT_AE_GMM_TOKENS):
        out.add("global")
    else:
        out.add("market")
    if any(token in lower for token in ("gmm_prob_", "posterior_")):
        out.add("soft_prob")
    if any(token in lower for token in ("dist_center", "mahal", "density", "nll", "likelihood")):
        out.add("distance")
    if any(token in lower for token in ("delta_", "accel", "speed", "time_since", "stability", "flip_count")):
        out.add("transition")
    if "entropy" in lower:
        out.add("entropy")
    if "reconstruction" in lower:
        out.add("reconstruction")
    return out
META_FEATURE_COLUMNS = (
    "side",
    "selector_score",
    "selector_rank_pct",
    "selector_ts_rank_pct",
    "selector_ts_side_rank_pct",
    "base_model_score",
    "bad_mae_pred",
    "timeout_pred",
    "side_bad_mae_pred",
    "side_timeout_pred",
    "clean_path_pred",
    "feature_gap_risk",
    "clean_dirty_positive_risk",
    "lgbm_bad_mae_pred",
    "lgbm_timeout_pred",
    "lgbm_clean_path_pred",
    "lgbm_dirty_positive_bad_mae_pred",
    "lgbm_positive_clean_path_pred",
    "s22_bucket_quality_score",
    "s22_bucket_quality_rank_pct",
    "s22_bucket_relaxed_pass_count",
    "s22_bucket_strict_pass_count",
    "lgbm_bad_mae_ts_pct",
    "lgbm_timeout_ts_pct",
    "lgbm_clean_path_ts_pct",
    "lgbm_dirty_positive_bad_mae_ts_pct",
    "lgbm_positive_clean_path_ts_pct",
    "lgbm_ranker_score",
    "lgbm_path_ranker_score",
    "lgbm_oracle_ranker_score",
    "lgbm_clean_oracle_ranker_score",
    "lgbm_timeout_aware_clean_ranker_score",
)


def _meta_context_columns(columns: list[str], base_features: list[str]) -> list[str]:
    if META_CONTEXT_FEATURE_POLICY in {"0", "false", "no", "off", "none"}:
        return []
    candidates = sorted(
        col
        for col in columns
        if str(col).startswith("ctx_") and col not in base_features
    )
    if META_CONTEXT_FEATURE_POLICY in {"all", "all_ctx", "all_context"}:
        return candidates
    if META_CONTEXT_FEATURE_POLICY not in {"ae_gmm_only", "ae_gmm", "archetype_only"}:
        raise ValueError(
            "Unsupported EPM_META_CONTEXT_FEATURE_POLICY="
            f"{META_CONTEXT_FEATURE_POLICY!r}; expected ae_gmm_only, all, or off."
        )
    selected = [
        col
        for col in candidates
        if any(token in str(col).lower() for token in META_CONTEXT_AE_GMM_TOKENS)
    ]
    if not META_CONTEXT_FEATURE_BLOCKS or "all" in META_CONTEXT_FEATURE_BLOCKS:
        return selected
    include = {block for block in META_CONTEXT_FEATURE_BLOCKS if not block.startswith("-")}
    exclude = {block[1:] for block in META_CONTEXT_FEATURE_BLOCKS if block.startswith("-")}
    return [
        col
        for col in selected
        if (not include or bool(_meta_context_feature_blocks(str(col)) & include))
        and not bool(_meta_context_feature_blocks(str(col)) & exclude)
    ]


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
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.mean()) if series.notna().any() else float("nan")


def _safe_min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.min()) if series.notna().any() else float("nan")


def _safe_max(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.max()) if series.notna().any() else float("nan")


def _selected_variant_key(selector: Any, meta_variant: Any, keep_frac: Any) -> tuple[str, str, float]:
    return (str(selector), str(meta_variant), round(float(keep_frac), 8))


def _side_capped_meta_indices(
    score: pd.Series,
    side: pd.Series,
    *,
    keep_frac: float,
    max_side_share: float,
    eligible: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0)
    valid_mask = score_s.notna().to_numpy()
    if eligible is not None:
        eligible_s = pd.Series(eligible).reset_index(drop=True).fillna(False).astype(bool)
        valid_mask &= eligible_s.to_numpy(dtype=bool)
    valid_idx = np.flatnonzero(valid_mask)
    target_rows = max(1, int(math.ceil(float(keep_frac) * len(valid_idx)))) if len(valid_idx) else 0
    if target_rows <= 0:
        return np.asarray([], dtype=np.int64)
    max_side_rows = max(1, int(math.floor(float(max_side_share) * target_rows)))
    order = valid_idx[
        np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    ]
    counts = {1: 0, -1: 0}
    selected: list[int] = []
    for idx in order:
        key = -1 if float(side_s.iloc[int(idx)]) < 0.0 else 1
        if counts[key] >= max_side_rows:
            continue
        selected.append(int(idx))
        counts[key] += 1
        if len(selected) >= target_rows:
            break
    if selected:
        selected_sides = np.asarray(
            [-1 if float(side_s.iloc[int(idx)]) < 0.0 else 1 for idx in selected],
            dtype=np.int8,
        )
        long_count = int((selected_sides > 0).sum())
        short_count = int((selected_sides < 0).sum())
        total = long_count + short_count
        if total > 0 and max(long_count, short_count) / float(total) > float(max_side_share):
            dominant = 1 if long_count >= short_count else -1
            minority_count = short_count if dominant == 1 else long_count
            if minority_count > 0:
                max_dominant = max(
                    1,
                    int(math.floor(float(max_side_share) * minority_count / (1.0 - float(max_side_share)))),
                )
                kept: list[int] = []
                dominant_kept = 0
                for idx, side_key in zip(selected, selected_sides, strict=False):
                    if int(side_key) == dominant:
                        if dominant_kept >= max_dominant:
                            continue
                        dominant_kept += 1
                    kept.append(int(idx))
                selected = kept
    return np.asarray(selected, dtype=np.int64)


def _first_numeric(frame: pd.DataFrame, columns: tuple[str, ...], default: float = 0.0) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values.astype(np.float32)
    return pd.Series(default, index=frame.index, dtype=np.float32)


def _posterior_cols(frame: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(
        cols,
        key=lambda col: int(col.rsplit("_", 1)[-1])
        if col.rsplit("_", 1)[-1].isdigit()
        else 999,
    )


def _argmax_bucket(frame: pd.DataFrame, cols: list[str], prefix: str) -> pd.Series:
    if not cols:
        return pd.Series("missing", index=frame.index, dtype="object")
    values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(values).any(axis=1)
    filled = np.where(np.isfinite(values), values, -np.inf)
    idx = np.argmax(filled, axis=1)
    out = pd.Series([f"{prefix}_{int(i)}" for i in idx], index=frame.index, dtype="object")
    out.loc[missing] = "missing"
    return out


def _side_archetype_bucket(frame: pd.DataFrame) -> pd.Series:
    side = pd.to_numeric(
        frame.get("side", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    long_bucket = _argmax_bucket(
        frame,
        _posterior_cols(frame, "ctx_long_gmm_cluster_posterior_"),
        "long",
    )
    short_bucket = _argmax_bucket(
        frame,
        _posterior_cols(frame, "ctx_short_gmm_cluster_posterior_"),
        "short",
    )
    global_bucket = _argmax_bucket(
        frame,
        _posterior_cols(frame, "ctx_gmm_cluster_posterior_"),
        "global",
    )
    side_bucket = pd.Series(
        np.where(side.gt(0.0), long_bucket, short_bucket),
        index=frame.index,
        dtype="object",
    )
    return side_bucket.where(side_bucket.ne("missing"), global_bucket).astype(str)


def _local_archetype_priors(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    shrink_rows: int = 80,
) -> dict[str, pd.Series]:
    valid_index = valid.index
    empty = {
        "bucket": pd.Series("missing", index=valid_index, dtype="object"),
        "support": pd.Series(0.0, index=valid_index, dtype=np.float32),
        "clean": pd.Series(0.0, index=valid_index, dtype=np.float32),
        "dirty": pd.Series(0.0, index=valid_index, dtype=np.float32),
        "bad": pd.Series(0.5, index=valid_index, dtype=np.float32),
        "timeout": pd.Series(0.1, index=valid_index, dtype=np.float32),
        "mean_u": pd.Series(0.0, index=valid_index, dtype=np.float32),
        "quality": pd.Series(0.0, index=valid_index, dtype=np.float32),
        "quality_rank": pd.Series(0.5, index=valid_index, dtype=np.float32),
    }
    if train.empty or valid.empty:
        return empty
    train_bucket = _side_archetype_bucket(train)
    valid_bucket = _side_archetype_bucket(valid)
    work = pd.DataFrame(
        {
            "bucket": train_bucket.astype(str),
            "u": pd.to_numeric(train.get("u_policy_net"), errors="coerce").fillna(0.0),
            "clean": train.get("clean_positive", pd.Series(False, index=train.index))
            .astype(bool)
            .astype(float),
            "dirty": train.get("dirty_positive", pd.Series(False, index=train.index))
            .astype(bool)
            .astype(float),
            "bad": train.get("bad_mae_1r", pd.Series(False, index=train.index))
            .astype(bool)
            .astype(float),
            "timeout": (
                pd.to_numeric(train.get("is_timeout", pd.Series(0.0, index=train.index)), errors="coerce")
                .fillna(0.0)
                .gt(0.5)
                .astype(float)
            ),
        },
        index=train.index,
    )
    if work.empty:
        return empty
    global_stats = {
        "clean": float(work["clean"].mean()),
        "dirty": float(work["dirty"].mean()),
        "bad": float(work["bad"].mean()),
        "timeout": float(work["timeout"].mean()),
        "mean_u": float(work["u"].mean()),
    }
    grouped = work.groupby("bucket", sort=False)
    stats = grouped.agg(
        support=("bucket", "size"),
        clean=("clean", "mean"),
        dirty=("dirty", "mean"),
        bad=("bad", "mean"),
        timeout=("timeout", "mean"),
        mean_u=("u", "mean"),
    )
    shrink = float(max(0, int(shrink_rows)))
    for col in ("clean", "dirty", "bad", "timeout", "mean_u"):
        stats[col] = (
            stats[col].astype(float) * stats["support"].astype(float)
            + float(global_stats[col]) * shrink
        ) / (stats["support"].astype(float) + shrink)
    stats["u_score"] = np.tanh(pd.to_numeric(stats["mean_u"], errors="coerce").fillna(0.0) * 80.0)
    stats["quality"] = (
        0.70 * stats["clean"]
        - 0.55 * stats["dirty"]
        - 0.45 * stats["bad"]
        - 0.40 * stats["timeout"]
        + 0.35 * stats["u_score"]
    )
    stats["quality_rank"] = stats["quality"].rank(method="average", pct=True)
    mapped = stats.reindex(valid_bucket.astype(str)).reset_index(drop=True)
    out = {
        "bucket": valid_bucket.reset_index(drop=True).astype(str),
        "support": pd.to_numeric(mapped["support"], errors="coerce")
        .fillna(0.0)
        .astype(np.float32),
        "clean": pd.to_numeric(mapped["clean"], errors="coerce")
        .fillna(global_stats["clean"])
        .astype(np.float32),
        "dirty": pd.to_numeric(mapped["dirty"], errors="coerce")
        .fillna(global_stats["dirty"])
        .astype(np.float32),
        "bad": pd.to_numeric(mapped["bad"], errors="coerce")
        .fillna(global_stats["bad"])
        .astype(np.float32),
        "timeout": pd.to_numeric(mapped["timeout"], errors="coerce")
        .fillna(global_stats["timeout"])
        .astype(np.float32),
        "mean_u": pd.to_numeric(mapped["mean_u"], errors="coerce")
        .fillna(global_stats["mean_u"])
        .astype(np.float32),
        "quality": pd.to_numeric(mapped["quality"], errors="coerce").fillna(0.0).astype(np.float32),
        "quality_rank": pd.to_numeric(mapped["quality_rank"], errors="coerce")
        .fillna(0.5)
        .astype(np.float32),
    }
    for key, value in out.items():
        value.index = valid_index
    return out


def _fit_meta_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    min_train_rows: int,
) -> tuple[np.ndarray, str]:
    if len(train) < int(min_train_rows):
        return np.full(len(valid), np.nan, dtype=np.float32), "insufficient_train_rows"
    y = train["clean_positive"].astype(bool).astype(int)
    if int(y.sum()) <= 0 or int((1 - y).sum()) <= 0:
        return np.full(len(valid), float(y.mean()), dtype=np.float32), "single_class"
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_valid = valid[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = x_train.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    sample_weight = (
        1.0
        + 1.50 * train["dirty_positive"].astype(bool).astype(float)
        + 1.00 * train["clean_positive"].astype(bool).astype(float)
        + 0.50 * pd.to_numeric(train["oracle_top"], errors="coerce").fillna(0.0).astype(float)
    )
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=96,
            learning_rate=0.045,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=1.25,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y.to_numpy(dtype=np.float32),
            sample_weight=sample_weight.to_numpy(dtype=np.float32),
        )
        return model.predict_proba(x_valid)[:, 1].astype(np.float32), "lgbm_ok"
    model = ExtraTreesClassifier(
        n_estimators=128,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(
        x_train,
        y.to_numpy(dtype=np.float32),
        sample_weight=sample_weight.to_numpy(dtype=np.float32),
    )
    return model.predict_proba(x_valid)[:, 1].astype(np.float32), "extratrees_ok"


def _fit_clean_dirty_contrast_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    min_train_rows: int,
) -> tuple[np.ndarray, str]:
    effective_min_train_rows = max(
        20,
        min(int(min_train_rows), int(META_CONTRAST_MIN_TRAIN_ROWS)),
    )
    train_pos = train[
        train["clean_positive"].astype(bool) | train["dirty_positive"].astype(bool)
    ].copy()
    if len(train_pos) < int(effective_min_train_rows):
        return np.full(len(valid), np.nan, dtype=np.float32), (
            f"insufficient_train_rows<{effective_min_train_rows}"
        )
    y = train_pos["clean_positive"].astype(bool).astype(int)
    if int(y.sum()) <= 0 or int((1 - y).sum()) <= 0:
        return np.full(len(valid), float(y.mean()), dtype=np.float32), "single_class"
    x_train = train_pos[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_valid = valid[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = x_train.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    utility = pd.to_numeric(train_pos["u_policy_net"], errors="coerce").fillna(0.0).clip(
        lower=0.0
    )
    sample_weight = (
        1.0
        + 1.00 * train_pos["clean_positive"].astype(bool).astype(float)
        + 1.00 * train_pos["dirty_positive"].astype(bool).astype(float)
        + 0.50 * pd.to_numeric(train_pos["oracle_top"], errors="coerce").fillna(0.0).astype(float)
        + 10.0 * utility.astype(float)
    )
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=128,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=1.50,
            random_state=int(seed) + 101,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y.to_numpy(dtype=np.float32),
            sample_weight=sample_weight.to_numpy(dtype=np.float32),
        )
        return model.predict_proba(x_valid)[:, 1].astype(np.float32), "lgbm_ok"
    model = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=7,
        min_samples_leaf=18,
        max_features="sqrt",
        random_state=int(seed) + 101,
        n_jobs=2,
    )
    model.fit(
        x_train,
        y.to_numpy(dtype=np.float32),
        sample_weight=sample_weight.to_numpy(dtype=np.float32),
    )
    return model.predict_proba(x_valid)[:, 1].astype(np.float32), "extratrees_ok"


def _fit_meta_risk_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: pd.Series,
    seed: int,
    min_train_rows: int,
    seed_offset: int,
) -> tuple[np.ndarray, str]:
    if len(train) < int(min_train_rows):
        return np.full(len(valid), np.nan, dtype=np.float32), "insufficient_train_rows"
    y = pd.Series(target).reset_index(drop=True).fillna(False).astype(bool).astype(int)
    if int(y.sum()) <= 0 or int((1 - y).sum()) <= 0:
        return np.full(len(valid), float(y.mean()), dtype=np.float32), "single_class"
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_valid = valid[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = x_train.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    pos_rate = float(y.mean())
    pos_weight = min(4.0, max(1.0, (1.0 - pos_rate) / max(pos_rate, 1.0e-6)))
    utility = (
        pd.to_numeric(train["u_policy_net"], errors="coerce")
        .reset_index(drop=True)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    dirty_positive = train["dirty_positive"].reset_index(drop=True).astype(bool).astype(float)
    oracle_top = (
        pd.to_numeric(train["oracle_top"], errors="coerce")
        .reset_index(drop=True)
        .fillna(0.0)
        .astype(float)
    )
    sample_weight = (
        1.0
        + (pos_weight - 1.0) * y.astype(float)
        + 0.50 * dirty_positive
        + 0.50 * oracle_top
        + 8.0 * utility.astype(float)
    )
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=128,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=1.75,
            random_state=int(seed) + int(seed_offset),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y.to_numpy(dtype=np.float32),
            sample_weight=sample_weight.to_numpy(dtype=np.float32),
        )
        return model.predict_proba(x_valid)[:, 1].astype(np.float32), "lgbm_ok"
    model = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=7,
        min_samples_leaf=18,
        max_features="sqrt",
        random_state=int(seed) + int(seed_offset),
        n_jobs=2,
    )
    model.fit(
        x_train,
        y.to_numpy(dtype=np.float32),
        sample_weight=sample_weight.to_numpy(dtype=np.float32),
    )
    return model.predict_proba(x_valid)[:, 1].astype(np.float32), "extratrees_ok"


def _rank_by_train_distribution(train_values: pd.Series, values: pd.Series) -> pd.Series:
    train_num = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_num = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_train = np.sort(train_num.dropna().to_numpy(dtype=float))
    if len(finite_train) == 0:
        return pd.Series(0.5, index=values.index, dtype=np.float32)
    raw = valid_num.to_numpy(dtype=float)
    out = np.full(len(raw), 0.5, dtype=np.float32)
    finite = np.isfinite(raw)
    out[finite] = np.searchsorted(finite_train, raw[finite], side="right") / float(len(finite_train))
    return pd.Series(out, index=values.index, dtype=np.float32)


def _build_executable_timeout_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [col for col in feature_cols if col in train.columns and col in valid.columns]
    train_x = train[cols].apply(pd.to_numeric, errors="coerce").copy()
    valid_x = valid[cols].apply(pd.to_numeric, errors="coerce").copy()

    context_tokens = (
        "spread",
        "volume",
        "liquidity",
        "oi_rank",
        "volatility",
        "entropy",
        "mahal",
        "distance",
        "speed",
        "accel",
        "stability",
        "flip_count",
        "reconstruction",
    )
    extra_cols = [
        col
        for col in train.columns
        if col in valid.columns
        and col.startswith("ctx_")
        and any(token in col.lower() for token in context_tokens)
    ]
    for col in extra_cols:
        if col not in train_x.columns:
            train_x[col] = pd.to_numeric(train[col], errors="coerce")
            valid_x[col] = pd.to_numeric(valid[col], errors="coerce")

    train_side = pd.to_numeric(train.get("side", pd.Series(0.0, index=train.index)), errors="coerce").fillna(0.0)
    valid_side = pd.to_numeric(valid.get("side", pd.Series(0.0, index=valid.index)), errors="coerce").fillna(0.0)
    train_x["is_long"] = train_side.gt(0.0).astype(float)
    valid_x["is_long"] = valid_side.gt(0.0).astype(float)
    train_x["is_short"] = train_side.lt(0.0).astype(float)
    valid_x["is_short"] = valid_side.lt(0.0).astype(float)

    for col in ("ctx_median_spread_bps", "ctx_oi_rank", "selector_ts_rank_pct", "selector_ts_side_rank_pct"):
        if col in train.columns and col in valid.columns:
            train_x[f"{col}_train_pct"] = _rank_by_train_distribution(train[col], train[col])
            valid_x[f"{col}_train_pct"] = _rank_by_train_distribution(train[col], valid[col])

    train_local = _local_archetype_priors(train, train, shrink_rows=80)
    valid_local = _local_archetype_priors(train, valid, shrink_rows=80)
    for name in ("support", "bad", "timeout", "mean_u", "quality_rank"):
        train_x[f"local_{name}"] = pd.to_numeric(train_local[name], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )
        valid_x[f"local_{name}"] = pd.to_numeric(valid_local[name], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )

    if "ctx_median_spread_bps_train_pct" in train_x.columns:
        train_x["long_x_spread_pct"] = train_x["is_long"] * train_x["ctx_median_spread_bps_train_pct"]
        valid_x["long_x_spread_pct"] = valid_x["is_long"] * valid_x["ctx_median_spread_bps_train_pct"]
    if "local_timeout" in train_x.columns:
        train_x["long_x_local_timeout"] = train_x["is_long"] * train_x["local_timeout"]
        valid_x["long_x_local_timeout"] = valid_x["is_long"] * valid_x["local_timeout"]
        train_x["short_x_local_timeout"] = train_x["is_short"] * train_x["local_timeout"]
        valid_x["short_x_local_timeout"] = valid_x["is_short"] * valid_x["local_timeout"]

    med = train_x.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    valid_x = valid_x.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return train_x.astype(np.float32), valid_x.astype(np.float32)


def _fit_executable_timeout_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    min_train_rows: int,
) -> tuple[np.ndarray, str]:
    if len(train) < int(min_train_rows):
        return np.full(len(valid), np.nan, dtype=np.float32), "insufficient_train_rows"
    y = (
        pd.to_numeric(train.get("is_timeout", pd.Series(0.0, index=train.index)), errors="coerce")
        .fillna(0.0)
        .gt(0.5)
        .astype(int)
    )
    if int(y.sum()) <= 0 or int((1 - y).sum()) <= 0:
        return np.full(len(valid), float(y.mean()), dtype=np.float32), "single_class"
    x_train, x_valid = _build_executable_timeout_features(train, valid, feature_cols)
    side = pd.to_numeric(train.get("side", pd.Series(0.0, index=train.index)), errors="coerce").fillna(0.0)
    bad = train.get("bad_mae_1r", pd.Series(False, index=train.index)).astype(bool).astype(float)
    sample_weight = (
        1.0
        + 2.5 * y.astype(float)
        + 0.50 * side.gt(0.0).astype(float)
        + 0.25 * bad
    )
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=5,
            min_child_samples=25,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=1.75,
            random_state=int(seed) + 503,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y.to_numpy(dtype=np.float32),
            sample_weight=sample_weight.to_numpy(dtype=np.float32),
        )
        return model.predict_proba(x_valid)[:, 1].astype(np.float32), "lgbm_ok"
    model = ExtraTreesClassifier(
        n_estimators=192,
        max_depth=8,
        min_samples_leaf=14,
        max_features="sqrt",
        random_state=int(seed) + 503,
        n_jobs=2,
    )
    model.fit(
        x_train,
        y.to_numpy(dtype=np.float32),
        sample_weight=sample_weight.to_numpy(dtype=np.float32),
    )
    return model.predict_proba(x_valid)[:, 1].astype(np.float32), "extratrees_ok"


def run_meta_filter_from_ledger(
    ledger: pd.DataFrame,
    *,
    keep_fracs: list[float],
    max_side_share: float,
    min_train_rows: int,
    seed: int,
    thresholds: dict[str, float],
    include_first_period: bool = False,
    selected_variant_keys: set[tuple[str, str, float]] | None = None,
    export_selected_rows: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    export_selected_rows = bool(export_selected_rows or selected_variant_keys is not None)
    if ledger.empty:
        if export_selected_rows:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(), pd.DataFrame()
    data = ledger.copy()
    data["period"] = data["period"].astype(str)
    feature_cols = [col for col in META_FEATURE_COLUMNS if col in data.columns]
    context_cols = _meta_context_columns(list(data.columns), feature_cols)
    feature_cols.extend(context_cols)
    monthly_rows: list[dict[str, Any]] = []
    selected_exports: list[pd.DataFrame] = []
    periods = sorted(data["period"].dropna().unique())
    period_list = periods if include_first_period else periods[1:]
    for selector, selector_rows in data.groupby("selector_variant", sort=False):
        for valid_period in period_list:
            train = selector_rows[selector_rows["period"] < valid_period].copy()
            valid = selector_rows[selector_rows["period"].eq(valid_period)].copy().reset_index(drop=True)
            if valid.empty:
                continue
            train_rows = int(len(train))
            eval_status = "ok" if train_rows >= int(min_train_rows) else "insufficient_train_rows"
            meta_score, status = _fit_meta_score(
                train,
                valid,
                feature_cols,
                seed=seed,
                min_train_rows=min_train_rows,
            )
            contrast_score, contrast_status = _fit_clean_dirty_contrast_score(
                train,
                valid,
                feature_cols,
                seed=seed,
                min_train_rows=min_train_rows,
            )
            meta_bad_score, meta_bad_status = _fit_meta_risk_score(
                train,
                valid,
                feature_cols,
                target=train["bad_mae_1r"].astype(bool),
                seed=seed,
                min_train_rows=min_train_rows,
                seed_offset=211,
            )
            meta_timeout_score, meta_timeout_status = _fit_meta_risk_score(
                train,
                valid,
                feature_cols,
                target=pd.to_numeric(train["is_timeout"], errors="coerce").fillna(0.0) > 0.5,
                seed=seed,
                min_train_rows=min_train_rows,
                seed_offset=307,
            )
            exec_timeout_score, exec_timeout_status = _fit_executable_timeout_score(
                train,
                valid,
                feature_cols,
                seed=seed,
                min_train_rows=min_train_rows,
            )
            valid["meta_score"] = meta_score
            valid["meta_clean_dirty_score"] = contrast_score
            valid["meta_bad_risk"] = meta_bad_score
            valid["meta_timeout_risk"] = meta_timeout_score
            valid["meta_exec_timeout_risk"] = exec_timeout_score
            bad_risk = _first_numeric(
                valid,
                ("lgbm_bad_mae_pred", "bad_mae_pred", "side_bad_mae_pred"),
                default=0.5,
            ).clip(0.0, 1.0)
            timeout_risk = _first_numeric(
                valid,
                ("lgbm_timeout_pred", "timeout_pred", "side_timeout_pred"),
                default=0.1,
            ).clip(0.0, 1.0)
            dirty_risk = _first_numeric(
                valid,
                ("lgbm_dirty_positive_bad_mae_pred", "clean_dirty_positive_risk"),
                default=0.5,
            ).clip(0.0, 1.0)
            risk_adjusted_score = (
                pd.Series(meta_score, index=valid.index, dtype=np.float32)
                - 0.70 * bad_risk
                - 0.25 * timeout_risk
                - 0.30 * dirty_risk
            ).astype(np.float32)
            contrast_score_s = pd.Series(contrast_score, index=valid.index, dtype=np.float32)
            meta_score_s = pd.Series(meta_score, index=valid.index, dtype=np.float32)
            meta_score_rank = pd.to_numeric(meta_score_s, errors="coerce").rank(
                method="average",
                pct=True,
            )
            contrast_score_rank = pd.to_numeric(contrast_score_s, errors="coerce").rank(
                method="average",
                pct=True,
            )
            contrast_rank_score = (
                0.40 * meta_score_s
                + 0.60 * contrast_score_s
                - 0.35 * bad_risk
                - 0.15 * timeout_risk
                - 0.20 * dirty_risk
            ).astype(np.float32)
            meta_bad_risk = pd.Series(meta_bad_score, index=valid.index, dtype=np.float32).clip(
                0.0,
                1.0,
            )
            meta_timeout_risk = pd.Series(
                meta_timeout_score,
                index=valid.index,
                dtype=np.float32,
            ).clip(0.0, 1.0)
            exec_timeout_risk = pd.Series(
                exec_timeout_score,
                index=valid.index,
                dtype=np.float32,
            ).clip(0.0, 1.0)
            joint_bad_risk = (0.50 * bad_risk + 0.50 * meta_bad_risk).astype(np.float32)
            joint_timeout_risk = (
                0.50 * timeout_risk + 0.50 * meta_timeout_risk
            ).astype(np.float32)
            exec_timeout_rank = pd.to_numeric(exec_timeout_risk, errors="coerce").rank(
                method="average",
                pct=True,
            )
            joint_path_score = (
                0.45 * meta_score_s
                + 0.45 * contrast_score_s
                + 0.10
                * _first_numeric(valid, ("lgbm_clean_path_pred", "clean_path_pred"), default=0.0)
                - 0.85 * joint_bad_risk
                - 0.75 * joint_timeout_risk
                - 0.20 * dirty_risk
            ).astype(np.float32)
            joint_path_strict_timeout_score = (
                joint_path_score
                - 0.35 * pd.to_numeric(joint_timeout_risk, errors="coerce").rank(
                    method="average",
                    pct=True,
                )
            ).astype(np.float32)
            joint_bad_rank = pd.to_numeric(joint_bad_risk, errors="coerce").rank(
                method="average",
                pct=True,
            )
            meta_bad_rank = pd.to_numeric(meta_bad_risk, errors="coerce").rank(
                method="average",
                pct=True,
            )
            joint_timeout_rank = pd.to_numeric(joint_timeout_risk, errors="coerce").rank(
                method="average",
                pct=True,
            )
            joint_path_strict_both_score = (
                joint_path_score
                - 0.25 * joint_bad_rank
                - 0.30 * joint_timeout_rank
            ).astype(np.float32)
            joint_path_penalty_b10_t40_score = (
                joint_path_score - 0.10 * joint_bad_rank - 0.40 * joint_timeout_rank
            ).astype(np.float32)
            joint_path_penalty_b15_t45_score = (
                joint_path_score - 0.15 * joint_bad_rank - 0.45 * joint_timeout_rank
            ).astype(np.float32)
            joint_path_penalty_b20_t50_score = (
                joint_path_score - 0.20 * joint_bad_rank - 0.50 * joint_timeout_rank
            ).astype(np.float32)
            clean_path_timeout_penalty_score = (
                pd.to_numeric(valid["meta_score"], errors="coerce")
                - 0.30 * joint_timeout_rank
            ).astype(np.float32)
            clean_path_strong_timeout_penalty_score = (
                pd.to_numeric(valid["meta_score"], errors="coerce")
                - 0.50 * joint_timeout_rank
            ).astype(np.float32)
            side_s = pd.to_numeric(valid["side"], errors="coerce").fillna(0.0)
            is_long = side_s.gt(0.0)
            is_short = side_s.lt(0.0)
            local_priors = _local_archetype_priors(train, valid, shrink_rows=80)
            local_quality_rank = pd.to_numeric(
                local_priors["quality_rank"],
                errors="coerce",
            ).fillna(0.5)
            local_bad_prior = pd.to_numeric(local_priors["bad"], errors="coerce").fillna(0.5)
            local_timeout_prior = pd.to_numeric(
                local_priors["timeout"],
                errors="coerce",
            ).fillna(0.1)
            local_mean_u_prior = pd.to_numeric(
                local_priors["mean_u"],
                errors="coerce",
            ).fillna(0.0)
            local_support = pd.to_numeric(local_priors["support"], errors="coerce").fillna(0.0)
            side_asym_clean_path_timeout_score = pd.Series(
                np.where(
                    is_long,
                    pd.to_numeric(valid["meta_score"], errors="coerce")
                    - 0.70 * joint_timeout_rank
                    - 0.15 * joint_bad_rank,
                    pd.to_numeric(valid["meta_score"], errors="coerce")
                    - 0.20 * joint_timeout_rank,
                ),
                index=valid.index,
                dtype=np.float32,
            )
            side_arch_local_quality_score = (
                side_asym_clean_path_timeout_score
                + 0.35 * local_quality_rank
                - 0.20
                * pd.to_numeric(local_bad_prior, errors="coerce").rank(
                    method="average",
                    pct=True,
                )
                - 0.15
                * pd.to_numeric(local_timeout_prior, errors="coerce").rank(
                    method="average",
                    pct=True,
                )
            ).astype(np.float32)
            side_arch_local_bad_penalty_score = (
                side_arch_local_quality_score
                - 0.45 * meta_bad_rank
                - 0.25 * joint_bad_rank
                - 0.10
                * pd.to_numeric(local_bad_prior, errors="coerce").rank(
                    method="average",
                    pct=True,
                )
            ).astype(np.float32)
            side_arch_exec_timeout_score = (
                side_arch_local_quality_score
                - 0.35 * exec_timeout_rank
                - 0.10
                * pd.to_numeric(exec_timeout_risk, errors="coerce").rank(
                    method="average",
                    pct=True,
                )
            ).astype(np.float32)
            side_arch_strong_exec_timeout_score = (
                side_arch_local_quality_score
                - 0.55 * exec_timeout_rank
                - 0.15
                * pd.to_numeric(exec_timeout_risk, errors="coerce").rank(
                    method="average",
                    pct=True,
                )
            ).astype(np.float32)
            long_agree_50_50 = is_long & meta_score_rank.ge(0.50) & contrast_score_rank.ge(0.50)
            long_agree_60_40 = is_long & meta_score_rank.ge(0.60) & contrast_score_rank.ge(0.40)
            side_asym_long_agree_short_joint_score = pd.Series(
                np.where(is_long, joint_path_score, joint_path_strict_timeout_score),
                index=valid.index,
                dtype=np.float32,
            )
            side_asym_long_agree_short_risk_score = pd.Series(
                np.where(is_long, joint_path_score, risk_adjusted_score),
                index=valid.index,
                dtype=np.float32,
            )
            oracle_total = int(pd.to_numeric(valid["oracle_rows_total"], errors="coerce").max() or 0)
            clean_oracle_total = int(
                pd.to_numeric(valid["clean_oracle_rows_total"], errors="coerce").max() or 0
            )
            for keep_frac in keep_fracs:
                keep_label = int(round(float(keep_frac) * 100))
                specs: list[tuple[str, pd.Series, pd.Series | None]] = [
                    (f"meta_clean_path_keep_{keep_label:02d}", valid["meta_score"], None),
                    (f"meta_risk_adjusted_keep_{keep_label:02d}", risk_adjusted_score, None),
                    (f"meta_clean_dirty_contrast_keep_{keep_label:02d}", contrast_rank_score, None),
                    (f"meta_joint_path_keep_{keep_label:02d}", joint_path_score, None),
                    (
                        f"meta_joint_path_strict_timeout_keep_{keep_label:02d}",
                        joint_path_strict_timeout_score,
                        None,
                    ),
                    (
                        f"meta_joint_path_strict_both_keep_{keep_label:02d}",
                        joint_path_strict_both_score,
                        None,
                    ),
                    (
                        f"meta_joint_path_penalty_b10_t40_keep_{keep_label:02d}",
                        joint_path_penalty_b10_t40_score,
                        None,
                    ),
                    (
                        f"meta_joint_path_penalty_b15_t45_keep_{keep_label:02d}",
                        joint_path_penalty_b15_t45_score,
                        None,
                    ),
                    (
                        f"meta_joint_path_penalty_b20_t50_keep_{keep_label:02d}",
                        joint_path_penalty_b20_t50_score,
                        None,
                    ),
                    (
                        f"meta_clean_dirty_contrast_veto_35_keep_{keep_label:02d}",
                        contrast_rank_score,
                        contrast_score_s.ge(0.35),
                    ),
                    (
                        f"meta_clean_dirty_contrast_veto_45_keep_{keep_label:02d}",
                        contrast_rank_score,
                        contrast_score_s.ge(0.45),
                    ),
                    (
                        (
                            f"meta_clean_path_contrast_rank_min_40"
                            f"_keep_{keep_label:02d}"
                        ),
                        valid["meta_score"],
                        contrast_score_rank.ge(0.40),
                    ),
                    (
                        (
                            f"meta_clean_path_timeout_penalty_contrast_rank_min_40"
                            f"_keep_{keep_label:02d}"
                        ),
                        clean_path_timeout_penalty_score,
                        contrast_score_rank.ge(0.40),
                    ),
                    (
                        (
                            f"meta_clean_path_strong_timeout_penalty_contrast_rank_min_40"
                            f"_keep_{keep_label:02d}"
                        ),
                        clean_path_strong_timeout_penalty_score,
                        contrast_score_rank.ge(0.40),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c50_t20_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.50)
                            & joint_timeout_risk.le(0.20)
                        )
                        | (is_short & contrast_score_rank.ge(0.35)),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c50_t18_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.50)
                            & joint_timeout_risk.le(0.18)
                        )
                        | (is_short & contrast_score_rank.ge(0.35)),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c60_t20_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.60)
                            & joint_timeout_risk.le(0.20)
                        )
                        | (is_short & contrast_score_rank.ge(0.35)),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c50_t20_short_c40"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.50)
                            & joint_timeout_risk.le(0.20)
                        )
                        | (is_short & contrast_score_rank.ge(0.40)),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c50_t12_short_c40"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.50)
                            & joint_timeout_risk.le(0.12)
                        )
                        | (is_short & contrast_score_rank.ge(0.40)),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c60_t12_short_c40"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.60)
                            & joint_timeout_risk.le(0.12)
                        )
                        | (is_short & contrast_score_rank.ge(0.40)),
                    ),
                    (
                        (
                            f"meta_side_asym_clean_path_long_c50_t10_short_c40"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_clean_path_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.50)
                            & joint_timeout_risk.le(0.10)
                        )
                        | (is_short & contrast_score_rank.ge(0.40)),
                    ),
                    (
                        (
                            f"meta_side_arch_local_quality_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        contrast_score_rank.ge(0.35) & local_support.ge(20.0),
                    ),
                    (
                        (
                            f"meta_side_arch_local_quality_c40"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        contrast_score_rank.ge(0.40) & local_support.ge(20.0),
                    ),
                    (
                        (
                            f"meta_side_arch_local_good_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        contrast_score_rank.ge(0.35)
                        & local_support.ge(20.0)
                        & local_bad_prior.le(0.62)
                        & local_timeout_prior.le(0.18)
                        & local_mean_u_prior.ge(-0.0015),
                    ),
                    (
                        (
                            f"meta_side_arch_local_good_c40"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        contrast_score_rank.ge(0.40)
                        & local_support.ge(20.0)
                        & local_bad_prior.le(0.60)
                        & local_timeout_prior.le(0.18)
                        & local_mean_u_prior.ge(-0.0015),
                    ),
                    (
                        (
                            f"meta_side_arch_local_asym_long_q50_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.45)
                            & local_quality_rank.ge(0.50)
                            & local_timeout_prior.le(0.20)
                        )
                        | (
                            is_short
                            & contrast_score_rank.ge(0.35)
                            & local_bad_prior.le(0.64)
                            & local_timeout_prior.le(0.20)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_local_asym_long_q60_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.50)
                            & local_quality_rank.ge(0.60)
                            & local_timeout_prior.le(0.20)
                        )
                        | (
                            is_short
                            & contrast_score_rank.ge(0.35)
                            & local_bad_prior.le(0.64)
                            & local_timeout_prior.le(0.20)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_exec_timeout_asym_q50_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_exec_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.45)
                            & local_quality_rank.ge(0.50)
                            & local_timeout_prior.le(0.20)
                        )
                        | (
                            is_short
                            & contrast_score_rank.ge(0.35)
                            & local_bad_prior.le(0.64)
                            & local_timeout_prior.le(0.20)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_strong_exec_timeout_asym_q50_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_strong_exec_timeout_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.45)
                            & local_quality_rank.ge(0.50)
                            & local_timeout_prior.le(0.20)
                        )
                        | (
                            is_short
                            & contrast_score_rank.ge(0.35)
                            & local_bad_prior.le(0.64)
                            & local_timeout_prior.le(0.20)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_exec_timeout_cap55_asym_q50_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_exec_timeout_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                            )
                        )
                        & exec_timeout_risk.le(0.55),
                    ),
                    (
                        (
                            f"meta_side_arch_exec_timeout_cap60_asym_q50_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_exec_timeout_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                            )
                        )
                        & exec_timeout_risk.le(0.60),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_side15_short_lgbm21"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("lgbm_timeout_pred",), default=1.0).le(0.21)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_side15_short_lgbm24"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("lgbm_timeout_pred",), default=1.0).le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_side15_short_joint21"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.21)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_side15_short_joint24"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_badpen_long_side15_short_joint24"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_bad_penalty_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_badpen_long_side15_short_joint24_bad55"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_bad_penalty_score,
                        (
                            (
                                (
                                    is_long
                                    & contrast_score_rank.ge(0.45)
                                    & local_quality_rank.ge(0.50)
                                    & local_timeout_prior.le(0.20)
                                    & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                                )
                                | (
                                    is_short
                                    & contrast_score_rank.ge(0.35)
                                    & local_bad_prior.le(0.64)
                                    & local_timeout_prior.le(0.20)
                                    & joint_timeout_risk.le(0.24)
                                )
                            )
                            & meta_bad_risk.le(0.55)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_badpen_long_side15_short_joint24_bad50"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_bad_penalty_score,
                        (
                            (
                                (
                                    is_long
                                    & contrast_score_rank.ge(0.45)
                                    & local_quality_rank.ge(0.50)
                                    & local_timeout_prior.le(0.20)
                                    & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                                )
                                | (
                                    is_short
                                    & contrast_score_rank.ge(0.35)
                                    & local_bad_prior.le(0.64)
                                    & local_timeout_prior.le(0.20)
                                    & joint_timeout_risk.le(0.24)
                                )
                            )
                            & meta_bad_risk.le(0.50)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_side20_short_joint24"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.45)
                                & local_quality_rank.ge(0.50)
                                & local_timeout_prior.le(0.20)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.20)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_q60_side15_short_joint24"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.55)
                                & local_quality_rank.ge(0.60)
                                & local_timeout_prior.le(0.18)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.15)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.35)
                                & local_bad_prior.le(0.64)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_q70_side12_short_joint24_c45"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.60)
                                & local_quality_rank.ge(0.70)
                                & local_timeout_prior.le(0.15)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.12)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.45)
                                & local_bad_prior.le(0.60)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_q70_side12_short_joint24_c50"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.60)
                                & local_quality_rank.ge(0.70)
                                & local_timeout_prior.le(0.15)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.12)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.50)
                                & local_bad_prior.le(0.60)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_existing_timeout_long_q70_side12_short_joint24_bad55"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            (
                                is_long
                                & contrast_score_rank.ge(0.60)
                                & local_quality_rank.ge(0.70)
                                & local_timeout_prior.le(0.15)
                                & _first_numeric(valid, ("side_timeout_pred",), default=0.5).le(0.12)
                            )
                            | (
                                is_short
                                & contrast_score_rank.ge(0.45)
                                & local_bad_prior.le(0.55)
                                & local_timeout_prior.le(0.20)
                                & joint_timeout_risk.le(0.24)
                            )
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_local_asym_long_q50_t15_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.45)
                            & local_quality_rank.ge(0.50)
                            & local_timeout_prior.le(0.20)
                            & joint_timeout_risk.le(0.15)
                        )
                        | (
                            is_short
                            & contrast_score_rank.ge(0.35)
                            & local_bad_prior.le(0.64)
                            & local_timeout_prior.le(0.20)
                        ),
                    ),
                    (
                        (
                            f"meta_side_arch_local_asym_long_q50_t12_short_c35"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_arch_local_quality_score,
                        (
                            is_long
                            & contrast_score_rank.ge(0.45)
                            & local_quality_rank.ge(0.50)
                            & local_timeout_prior.le(0.20)
                            & joint_timeout_risk.le(0.12)
                        )
                        | (
                            is_short
                            & contrast_score_rank.ge(0.35)
                            & local_bad_prior.le(0.64)
                            & local_timeout_prior.le(0.20)
                        ),
                    ),
                    (
                        (
                            f"meta_clean_path_contrast_rank_min_50"
                            f"_keep_{keep_label:02d}"
                        ),
                        valid["meta_score"],
                        contrast_score_rank.ge(0.50),
                    ),
                    (
                        (
                            f"meta_clean_path_contrast_rank_min_60"
                            f"_keep_{keep_label:02d}"
                        ),
                        valid["meta_score"],
                        contrast_score_rank.ge(0.60),
                    ),
                    (
                        (
                            f"meta_clean_path_contrast_rank_min_70"
                            f"_keep_{keep_label:02d}"
                        ),
                        valid["meta_score"],
                        contrast_score_rank.ge(0.70),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta50_contrast50"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.50) & contrast_score_rank.ge(0.50),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta60_contrast40"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.60) & contrast_score_rank.ge(0.40),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta60_contrast50"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.60) & contrast_score_rank.ge(0.50),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta70_contrast40"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.70) & contrast_score_rank.ge(0.40),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta70_contrast50"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.70) & contrast_score_rank.ge(0.50),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta60_contrast50"
                            f"_bad_cap_60_timeout_cap_18_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.60)
                        & contrast_score_rank.ge(0.50)
                        & joint_bad_risk.le(0.60)
                        & joint_timeout_risk.le(0.18),
                    ),
                    (
                        (
                            f"meta_clean_contrast_agree_meta70_contrast40"
                            f"_bad_cap_60_timeout_cap_18_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        meta_score_rank.ge(0.70)
                        & contrast_score_rank.ge(0.40)
                        & joint_bad_risk.le(0.60)
                        & joint_timeout_risk.le(0.18),
                    ),
                    (
                        (
                            f"meta_side_asym_long_agree50_short_joint_timeout"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_long_agree_short_joint_score,
                        long_agree_50_50 | is_short,
                    ),
                    (
                        (
                            f"meta_side_asym_long_agree60_short_joint_timeout"
                            f"_keep_{keep_label:02d}"
                        ),
                        side_asym_long_agree_short_joint_score,
                        long_agree_60_40 | is_short,
                    ),
                    (
                        (
                            f"meta_side_asym_long_agree50_short_joint_timeout"
                            f"_bad_cap_60_timeout_cap_18_keep_{keep_label:02d}"
                        ),
                        side_asym_long_agree_short_joint_score,
                        long_agree_50_50
                        | (
                            is_short
                            & joint_bad_risk.le(0.60)
                            & joint_timeout_risk.le(0.18)
                        ),
                    ),
                    (
                        (
                            f"meta_side_asym_long_agree50_short_risk_adjusted"
                            f"_bad_cap_60_timeout_cap_18_keep_{keep_label:02d}"
                        ),
                        side_asym_long_agree_short_risk_score,
                        long_agree_50_50
                        | (
                            is_short
                            & bad_risk.le(0.60)
                            & timeout_risk.le(0.18)
                        ),
                    ),
                    (
                        (
                            f"meta_clean_dirty_contrast_veto_35"
                            f"_bad_cap_55_timeout_cap_18_keep_{keep_label:02d}"
                        ),
                        contrast_rank_score,
                        contrast_score_s.ge(0.35)
                        & bad_risk.le(0.55)
                        & timeout_risk.le(0.18),
                    ),
                    (
                        (
                            f"meta_clean_dirty_contrast_veto_35"
                            f"_bad_cap_55_timeout_cap_12_keep_{keep_label:02d}"
                        ),
                        contrast_rank_score,
                        contrast_score_s.ge(0.35)
                        & bad_risk.le(0.55)
                        & timeout_risk.le(0.12),
                    ),
                    (
                        (
                            f"meta_clean_dirty_contrast_veto_35"
                            f"_bad_cap_60_timeout_cap_12_keep_{keep_label:02d}"
                        ),
                        contrast_rank_score,
                        contrast_score_s.ge(0.35)
                        & bad_risk.le(0.60)
                        & timeout_risk.le(0.12),
                    ),
                    (
                        (
                            f"meta_joint_path_bad_cap_50_timeout_cap_12"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        joint_bad_risk.le(0.50) & joint_timeout_risk.le(0.12),
                    ),
                    (
                        (
                            f"meta_joint_path_bad_cap_55_timeout_cap_12"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        joint_bad_risk.le(0.55) & joint_timeout_risk.le(0.12),
                    ),
                    (
                        (
                            f"meta_joint_path_bad_cap_60_timeout_cap_12"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_score,
                        joint_bad_risk.le(0.60) & joint_timeout_risk.le(0.12),
                    ),
                    (
                        (
                            f"meta_joint_path_strict_timeout_bad_cap_60_timeout_cap_12"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_strict_timeout_score,
                        joint_bad_risk.le(0.60) & joint_timeout_risk.le(0.12),
                    ),
                    (
                        (
                            f"meta_joint_path_strict_both_bad_cap_60_timeout_cap_12"
                            f"_keep_{keep_label:02d}"
                        ),
                        joint_path_strict_both_score,
                        joint_bad_risk.le(0.60) & joint_timeout_risk.le(0.12),
                    ),
                ]
                for bad_cap, timeout_cap in (
                    (0.50, 0.12),
                    (0.55, 0.12),
                    (0.60, 0.12),
                    (0.50, 0.15),
                    (0.55, 0.15),
                    (0.60, 0.15),
                ):
                    clean_path_eligible = bad_risk.le(float(bad_cap)) & timeout_risk.le(
                        float(timeout_cap)
                    )
                    specs.append(
                        (
                            (
                                f"meta_clean_path_bad_cap_{int(round(bad_cap * 100)):02d}"
                                f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                f"_keep_{keep_label:02d}"
                            ),
                            valid["meta_score"],
                            clean_path_eligible,
                        )
                    )
                for bad_cap, timeout_cap in (
                    (0.50, 0.12),
                    (0.55, 0.12),
                    (0.60, 0.12),
                    (0.45, 0.15),
                    (0.50, 0.15),
                    (0.55, 0.18),
                    (0.60, 0.20),
                ):
                    eligible = bad_risk.le(float(bad_cap)) & timeout_risk.le(float(timeout_cap))
                    specs.append(
                        (
                            (
                                f"meta_risk_adjusted_bad_cap_{int(round(bad_cap * 100)):02d}"
                                f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                f"_keep_{keep_label:02d}"
                            ),
                            risk_adjusted_score,
                            eligible,
                        )
                    )
                for variant_name, variant_score, eligible in specs:
                    score_for_variant = pd.to_numeric(
                        pd.Series(variant_score).reset_index(drop=True),
                        errors="coerce",
                    )
                    selected = _side_capped_meta_indices(
                        score_for_variant,
                        valid["side"],
                        keep_frac=float(keep_frac),
                        max_side_share=float(max_side_share),
                        eligible=eligible,
                    )
                    selected_frame = valid.iloc[selected] if len(selected) else valid.iloc[:0]
                    side = pd.to_numeric(selected_frame.get("side"), errors="coerce")
                    long_rows = int((side > 0.0).sum()) if len(selected_frame) else 0
                    short_rows = int((side < 0.0).sum()) if len(selected_frame) else 0
                    selected_rows = int(len(selected_frame))
                    max_side = (
                        max(long_rows, short_rows) / float(selected_rows)
                        if selected_rows
                        else float("nan")
                    )
                    monthly_rows.append(
                        {
                            "selector_variant": selector,
                            "meta_variant": variant_name,
                            "period": valid_period,
                            "keep_frac": float(keep_frac),
                            "meta_eval_status": str(eval_status),
                            "meta_train_rows": int(train_rows),
                            "meta_status": status,
                            "contrast_meta_status": contrast_status,
                            "meta_bad_status": meta_bad_status,
                            "meta_timeout_status": meta_timeout_status,
                            "exec_timeout_status": exec_timeout_status,
                            "candidate_rows": int(len(valid)),
                            "eligible_rows": int(pd.Series(eligible).sum())
                            if eligible is not None
                            else int(len(valid)),
                            "selected_rows": selected_rows,
                            "selected_long_rows": long_rows,
                            "selected_short_rows": short_rows,
                            "max_selected_side_share": max_side,
                            "mean_u": _safe_mean(selected_frame["u_policy_net"]),
                            "bad_mae_1r_rate": _safe_mean(selected_frame["bad_mae_1r"]),
                            "timeout_rate": _safe_mean(
                                pd.to_numeric(selected_frame["is_timeout"], errors="coerce") > 0.5
                            ),
                            "clean_positive_rate": _safe_mean(selected_frame["clean_positive"]),
                            "dirty_positive_rate": _safe_mean(selected_frame["dirty_positive"]),
                            "oracle_rows_total": oracle_total,
                            "oracle_hit_rows": int(selected_frame["oracle_top"].astype(bool).sum())
                            if selected_rows
                            else 0,
                            "final_oracle_recall": (
                                float(selected_frame["oracle_top"].astype(bool).sum() / oracle_total)
                                if oracle_total
                                else float("nan")
                            ),
                            "clean_oracle_rows_total": clean_oracle_total,
                            "clean_oracle_hit_rows": int(
                                selected_frame["clean_oracle_top"].astype(bool).sum()
                            )
                            if selected_rows
                            else 0,
                            "clean_oracle_recall": (
                                float(
                                    selected_frame["clean_oracle_top"].astype(bool).sum()
                                    / clean_oracle_total
                                )
                                if clean_oracle_total
                                else float("nan")
                            ),
                            "selected_bad_risk_mean": _safe_mean(bad_risk.iloc[selected])
                            if len(selected)
                            else float("nan"),
                            "selected_timeout_risk_mean": _safe_mean(timeout_risk.iloc[selected])
                            if len(selected)
                            else float("nan"),
                            "selected_dirty_risk_mean": _safe_mean(dirty_risk.iloc[selected])
                            if len(selected)
                            else float("nan"),
                            "selected_clean_dirty_score_mean": _safe_mean(
                                contrast_score_s.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_meta_bad_risk_mean": _safe_mean(meta_bad_risk.iloc[selected])
                            if len(selected)
                            else float("nan"),
                            "selected_meta_timeout_risk_mean": _safe_mean(
                                meta_timeout_risk.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_exec_timeout_risk_mean": _safe_mean(
                                exec_timeout_risk.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_joint_bad_risk_mean": _safe_mean(joint_bad_risk.iloc[selected])
                            if len(selected)
                            else float("nan"),
                            "selected_joint_timeout_risk_mean": _safe_mean(
                                joint_timeout_risk.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_local_support_mean": _safe_mean(
                                local_support.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_local_quality_rank_mean": _safe_mean(
                                local_quality_rank.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_local_bad_prior_mean": _safe_mean(
                                local_bad_prior.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                            "selected_local_timeout_prior_mean": _safe_mean(
                                local_timeout_prior.iloc[selected]
                            )
                            if len(selected)
                            else float("nan"),
                        }
                    )
                    selected_key = _selected_variant_key(selector, variant_name, keep_frac)
                    should_export = (
                        export_selected_rows
                        and len(selected)
                        and (
                            selected_variant_keys is None
                            or selected_key in selected_variant_keys
                        )
                    )
                    if should_export:
                        selected_export = selected_frame.copy().reset_index(drop=True)
                        selected_scores = score_for_variant.iloc[selected].to_numpy(
                            dtype=np.float32,
                            copy=False,
                        )
                        score_rank_pct = score_for_variant.rank(method="average", pct=True)
                        selected_export["meta_variant"] = str(variant_name)
                        selected_export["keep_frac"] = float(keep_frac)
                        selected_export["meta_selected_score"] = selected_scores
                        selected_export["meta_selected_rank"] = np.arange(
                            1,
                            len(selected_export) + 1,
                            dtype=np.int32,
                        )
                        selected_export["meta_selected_count"] = int(len(selected_export))
                        selected_export["meta_score_rank_pct"] = score_rank_pct.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["meta_score_rank_pct_selected"] = (
                            1.0
                            - (
                                selected_export["meta_selected_rank"].astype(np.float32)
                                - 1.0
                            )
                            / max(float(len(selected_export) - 1), 1.0)
                        ).astype(np.float32)
                        selected_export["meta_bad_risk"] = meta_bad_risk.iloc[selected].to_numpy(
                            dtype=np.float32,
                            copy=False,
                        )
                        selected_export["meta_timeout_risk"] = meta_timeout_risk.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["meta_exec_timeout_risk"] = exec_timeout_risk.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["joint_bad_risk"] = joint_bad_risk.iloc[selected].to_numpy(
                            dtype=np.float32,
                            copy=False,
                        )
                        selected_export["joint_timeout_risk"] = joint_timeout_risk.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["local_side_archetype"] = local_priors["bucket"].iloc[
                            selected
                        ].astype(str).to_numpy(copy=False)
                        selected_export["local_archetype_support"] = local_support.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["local_archetype_quality"] = local_priors["quality"].iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["local_archetype_quality_rank"] = local_quality_rank.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["local_archetype_bad_prior"] = local_bad_prior.iloc[
                            selected
                        ].to_numpy(dtype=np.float32, copy=False)
                        selected_export["local_archetype_timeout_prior"] = (
                            local_timeout_prior.iloc[selected].to_numpy(
                                dtype=np.float32,
                                copy=False,
                            )
                        )
                        selected_export["local_archetype_mean_u_prior"] = (
                            local_mean_u_prior.iloc[selected].to_numpy(
                                dtype=np.float32,
                                copy=False,
                            )
                        )
                        selected_exports.append(selected_export)
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return monthly, pd.DataFrame()
    aggregate_rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(["selector_variant", "meta_variant", "keep_frac"], sort=False):
        selector, meta_variant, keep_frac = key
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        eligible_rows = pd.to_numeric(group["eligible_rows"], errors="coerce")
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        bad_mae = pd.to_numeric(group["bad_mae_1r_rate"], errors="coerce")
        timeout = pd.to_numeric(group["timeout_rate"], errors="coerce")
        max_side = pd.to_numeric(group["max_selected_side_share"], errors="coerce")
        eval_mask = group["meta_eval_status"].astype(str).eq("ok")
        mean_u_eval = mean_u.where(eval_mask, np.nan)
        bad_mae_eval = bad_mae.where(eval_mask, np.nan)
        timeout_eval = timeout.where(eval_mask, np.nan)
        selected_rows_eval = selected_rows.where(eval_mask, np.nan)
        max_side_eval = max_side.where(eval_mask, np.nan)
        oracle_hits = pd.to_numeric(group["oracle_hit_rows"], errors="coerce").sum()
        oracle_total = pd.to_numeric(group["oracle_rows_total"], errors="coerce").sum()
        max_side = pd.to_numeric(group["max_selected_side_share"], errors="coerce")
        row = {
            "selector_variant": selector,
            "meta_variant": meta_variant,
            "keep_frac": float(keep_frac),
            "meta_oos_months": int(eval_mask.sum()),
            "meta_skipped_months_due_to_insufficient_train": int((~eval_mask).sum()),
            "positive_months": int((mean_u_eval > 0.0).sum()),
            "no_trade_months": int((selected_rows_eval <= 0).sum()),
            "mean_u": _safe_mean(mean_u_eval),
            "worst_month_mean_u": _safe_min(mean_u_eval),
            "bad_mae_1r_rate": _safe_mean(bad_mae_eval),
            "timeout_rate": _safe_mean(timeout_eval),
            "max_month_bad_mae_1r_rate": _safe_max(bad_mae_eval),
            "max_month_timeout_rate": _safe_max(timeout_eval),
            "clean_positive_rate": _safe_mean(group["clean_positive_rate"]),
            "dirty_positive_rate": _safe_mean(group["dirty_positive_rate"]),
            "final_oracle_recall": (
                float(oracle_hits / oracle_total) if float(oracle_total) > 0 else float("nan")
            ),
            "mean_eligible_rows": _safe_mean(eligible_rows),
            "mean_selected_rows": _safe_mean(selected_rows_eval),
            "min_selected_rows": int(selected_rows_eval.min()) if selected_rows_eval.notna().any() else 0,
            "max_selected_side_share": _safe_mean(max_side_eval),
            "worst_month_selected_side_share": _safe_min(max_side_eval)
            if max_side_eval.notna().any()
            else float("nan"),
            "selected_bad_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_bad_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_bad_risk_mean" in group.columns
            else float("nan"),
            "selected_timeout_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_timeout_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_timeout_risk_mean" in group.columns
            else float("nan"),
            "selected_dirty_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_dirty_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_dirty_risk_mean" in group.columns
            else float("nan"),
            "selected_clean_dirty_score_mean": _safe_mean(
                pd.to_numeric(group["selected_clean_dirty_score_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_clean_dirty_score_mean" in group.columns
            else float("nan"),
            "selected_meta_bad_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_meta_bad_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_meta_bad_risk_mean" in group.columns
            else float("nan"),
            "selected_meta_timeout_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_meta_timeout_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_meta_timeout_risk_mean" in group.columns
            else float("nan"),
            "selected_exec_timeout_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_exec_timeout_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_exec_timeout_risk_mean" in group.columns
            else float("nan"),
            "selected_joint_bad_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_joint_bad_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_joint_bad_risk_mean" in group.columns
            else float("nan"),
            "selected_joint_timeout_risk_mean": _safe_mean(
                pd.to_numeric(group["selected_joint_timeout_risk_mean"], errors="coerce").where(
                    eval_mask,
                    np.nan,
                )
            )
            if "selected_joint_timeout_risk_mean" in group.columns
            else float("nan"),
        }
        row["decision"] = (
            "candidate_for_full_train_meta_oos"
            if row["positive_months"] == row["meta_oos_months"]
            and row["no_trade_months"] == 0
            and row["mean_u"] > thresholds["min_mean_u"]
            and row["worst_month_mean_u"] > thresholds["min_worst_month_mean_u"]
            and row["bad_mae_1r_rate"] <= thresholds["max_bad_mae_1r_rate"]
            and row["timeout_rate"] <= thresholds["max_timeout_rate"]
            and row["max_month_bad_mae_1r_rate"] <= thresholds["max_month_bad_mae_1r_rate"]
            and row["max_month_timeout_rate"] <= thresholds["max_month_timeout_rate"]
            and row["final_oracle_recall"] >= thresholds["min_final_oracle_recall"]
            and row["min_selected_rows"] >= thresholds["min_selected_rows"]
            and row["worst_month_selected_side_share"] <= thresholds["max_selected_side_share"]
            else "reject_or_rework"
        )
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["decision", "mean_u", "bad_mae_1r_rate"],
        ascending=[True, False, True],
    )
    if export_selected_rows:
        selected_rows = (
            pd.concat(selected_exports, ignore_index=True)
            if selected_exports
            else pd.DataFrame()
        )
        return monthly, aggregate, selected_rows
    return monthly, aggregate


def _simple_policy_handoff_from_selected(
    selected_rows: pd.DataFrame,
    *,
    barrier_multiplier: float = 1.0,
) -> pd.DataFrame:
    if selected_rows.empty:
        return pd.DataFrame()
    multiplier = max(float(barrier_multiplier), 1e-6)
    out = pd.DataFrame()
    timestamp = pd.to_datetime(selected_rows["timestamp"], utc=True, errors="coerce")
    side = pd.to_numeric(selected_rows.get("side"), errors="coerce").fillna(1.0)
    meta_score = pd.to_numeric(selected_rows["meta_selected_score"], errors="coerce")
    score_rank = pd.to_numeric(selected_rows["meta_score_rank_pct"], errors="coerce")
    selected_rank = pd.to_numeric(
        selected_rows["meta_score_rank_pct_selected"],
        errors="coerce",
    )
    rank_pct = score_rank.fillna(selected_rank).fillna(1.0).clip(0.0, 1.0)
    barrier = pd.to_numeric(
        selected_rows.get("barrier", pd.Series(0.005, index=selected_rows.index)),
        errors="coerce",
    ).fillna(0.005) * multiplier
    out["timestamp"] = timestamp
    out["symbol"] = selected_rows["symbol"].astype(str)
    out["side"] = np.where(side < 0.0, "short", "long")
    out["strategy_id"] = np.where(
        side < 0.0,
        "short_s19_meta_path_filter",
        "long_s19_meta_path_filter",
    )
    out["rank_pct"] = rank_pct.astype(np.float32)
    out["calibrated_score"] = meta_score.astype(np.float32)
    out["barrier_pct"] = barrier.astype(np.float32)
    out["simple_policy_barrier_multiplier"] = float(multiplier)
    out["base_strategy_threshold"] = 0.0
    out["best_size_power"] = 1.0
    out["oof_regime_centroid_similarity_train"] = pd.to_numeric(
        selected_rows.get("joint_bad_risk", pd.Series(np.nan, index=selected_rows.index)),
        errors="coerce",
    ).astype(np.float32)
    out["meta_selector_variant"] = selected_rows["selector_variant"].astype(str)
    out["meta_variant"] = selected_rows["meta_variant"].astype(str)
    out["meta_keep_frac"] = pd.to_numeric(selected_rows["keep_frac"], errors="coerce").astype(
        np.float32
    )
    out["meta_score_rank_pct"] = score_rank.astype(np.float32)
    out["meta_score_rank_pct_selected"] = selected_rank.astype(np.float32)
    out["meta_bad_risk"] = pd.to_numeric(
        selected_rows.get("meta_bad_risk"),
        errors="coerce",
    ).astype(np.float32)
    out["meta_timeout_risk"] = pd.to_numeric(
        selected_rows.get("meta_timeout_risk"),
        errors="coerce",
    ).astype(np.float32)
    out["joint_bad_risk"] = pd.to_numeric(
        selected_rows.get("joint_bad_risk"),
        errors="coerce",
    ).astype(np.float32)
    out["joint_timeout_risk"] = pd.to_numeric(
        selected_rows.get("joint_timeout_risk"),
        errors="coerce",
    ).astype(np.float32)
    out = out.dropna(subset=["timestamp", "symbol", "rank_pct", "barrier_pct"]).copy()
    return out.sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)


def run_meta_smoke(
    *,
    report_dir: Path,
    output_dir: Path,
    candidate_streams: list[str],
    keep_fracs: list[float],
    candidate_ledger_path: Path | None,
    seeds: list[int] | None,
    train_lookback_months: int | None,
    max_feature_store_features: int | None,
    max_side_share: float,
    thresholds: dict[str, float] | None = None,
    include_first_period: bool = False,
    min_train_rows: int,
    simple_policy_barrier_multiplier: float = 1.0,
    base_top_fracs: list[float] | None = None,
    spread_baseline_path: Path | None = None,
    spread_rank_column: str = "p75_spread_bps",
    target_symbol_count: int | None = None,
    max_spread_bps: float | None = None,
    export_meta_variants: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if candidate_ledger_path is None:
        candidates = _active_readiness_rows(report_dir)
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one active candidate, found {len(candidates)}")
        candidate = candidates.iloc[0]
        base_output = output_dir / DEFAULT_BASE_STREAM_SUBDIR
        smoke_manifest = run_smoke(
            labels_path=Path(str(candidate.get("labels_path"))),
            output_dir=base_output,
            feature_dir=Path(str(candidate.get("feature_dir"))),
            feature_list_csv=Path(str(candidate.get("feature_list_csv"))),
            evaluation_utility_column=str(candidate.get("evaluation_utility_source") or "").strip()
            or None,
            max_feature_store_features=max_feature_store_features,
            label_arms=[str(candidate.get("label_arm"))],
            weight_arms=[str(candidate.get("weight_arm"))],
            seeds=seeds or [42],
            model_feature_selector="all",
            model_feature_tail_frac=0.01,
            top_fracs=base_top_fracs or [float(candidate.get("top_frac"))],
            train_lookback_months=train_lookback_months,
            include_risk_selector_variants=True,
            side_cap_max_share=float(max_side_share),
            candidate_ledger_selector_names=candidate_streams,
            candidate_ledger_only=True,
            candidate_ledger_fast_mode=True,
            spread_baseline_path=spread_baseline_path,
            spread_rank_column=spread_rank_column,
            target_symbol_count=target_symbol_count,
            max_spread_bps=max_spread_bps,
        )
        candidate_ledger_path = Path(str(smoke_manifest["outputs"]["candidate_ledger"]))
    else:
        smoke_manifest = {"outputs": {"candidate_ledger": str(candidate_ledger_path)}}
    ledger = pd.read_csv(candidate_ledger_path) if candidate_ledger_path.exists() else pd.DataFrame()
    if not ledger.empty and candidate_streams:
        requested_streams = {str(value) for value in candidate_streams}
        before_rows = int(len(ledger))
        ledger = ledger[ledger["selector_variant"].astype(str).isin(requested_streams)].copy()
        if ledger.empty:
            raise ValueError(
                "candidate ledger contains no requested selector_variant rows: "
                f"requested={sorted(requested_streams)} path={candidate_ledger_path}"
            )
        smoke_manifest["candidate_ledger_filter"] = {
            "requested_selector_variants": sorted(requested_streams),
            "input_rows": before_rows,
            "retained_rows": int(len(ledger)),
        }
    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        merged_thresholds.update({k: float(v) for k, v in thresholds.items()})
    merged_thresholds["max_selected_side_share"] = float(max_side_share)
    monthly, aggregate = run_meta_filter_from_ledger(
        ledger,
        keep_fracs=keep_fracs,
        max_side_share=float(max_side_share),
        min_train_rows=int(min_train_rows),
        seed=int((seeds or [42])[0]),
        thresholds=merged_thresholds,
        include_first_period=include_first_period,
    )
    passing = aggregate[aggregate["decision"].eq("candidate_for_full_train_meta_oos")]
    selected_rows = pd.DataFrame()
    handoff_candidates = pd.DataFrame()
    export_tokens = [str(v).strip() for v in (export_meta_variants or []) if str(v).strip()]
    export_all_selected = any(token.lower() == "all" for token in export_tokens)
    selected_key: set[tuple[str, str, float]] = set()
    if export_tokens:
        if not export_all_selected and not aggregate.empty:
            lower_tokens = [token.lower() for token in export_tokens]
            for row in aggregate.itertuples(index=False):
                meta_variant = str(getattr(row, "meta_variant"))
                meta_lower = meta_variant.lower()
                if any(token == meta_lower or token in meta_lower for token in lower_tokens):
                    selected_key.add(
                        _selected_variant_key(
                            getattr(row, "selector_variant"),
                            meta_variant,
                            getattr(row, "keep_frac"),
                        )
                    )
    elif not passing.empty:
        best = passing.iloc[0]
        selected_key = {
            _selected_variant_key(
                best["selector_variant"],
                best["meta_variant"],
                best["keep_frac"],
            )
        }
    if export_all_selected or selected_key:
        _monthly2, _aggregate2, selected_rows = run_meta_filter_from_ledger(
            ledger,
            keep_fracs=keep_fracs,
            max_side_share=float(max_side_share),
            min_train_rows=int(min_train_rows),
            seed=int((seeds or [42])[0]),
            thresholds=merged_thresholds,
            include_first_period=include_first_period,
            selected_variant_keys=None if export_all_selected else selected_key,
            export_selected_rows=True,
        )
        handoff_candidates = _simple_policy_handoff_from_selected(
            selected_rows,
            barrier_multiplier=float(simple_policy_barrier_multiplier),
        )
    paths = {
        "monthly": output_dir / "gmm_train_meta_path_filter_smoke_monthly.csv",
        "aggregate": output_dir / "gmm_train_meta_path_filter_smoke_aggregate.csv",
        "selected_rows": output_dir / "gmm_train_meta_path_filter_smoke_selected_rows.parquet",
        "simple_policy_handoff_candidates": output_dir
        / "gmm_train_meta_path_filter_simple_policy_candidates.parquet",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    selected_rows.to_parquet(paths["selected_rows"], index=False)
    handoff_candidates.to_parquet(paths["simple_policy_handoff_candidates"], index=False)
    manifest = {
        "status": "pass" if not passing.empty else "fail",
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "candidate_streams": candidate_streams,
        "keep_fracs": [float(v) for v in keep_fracs],
        "simple_policy_barrier_multiplier": float(simple_policy_barrier_multiplier),
        "candidate_ledger_path": str(candidate_ledger_path),
        "meta_context_feature_policy": META_CONTEXT_FEATURE_POLICY,
        "meta_context_feature_blocks": list(META_CONTEXT_FEATURE_BLOCKS),
        "meta_context_feature_count": int(
            len(_meta_context_columns(list(ledger.columns), [col for col in META_FEATURE_COLUMNS if col in ledger.columns]))
        )
        if not ledger.empty
        else 0,
        "base_smoke_manifest": smoke_manifest,
        "base_top_fracs": [float(v) for v in (base_top_fracs or [])],
        "symbol_universe_filter": {
            "spread_baseline_path": str(spread_baseline_path) if spread_baseline_path else None,
            "spread_rank_column": str(spread_rank_column),
            "target_symbol_count": int(target_symbol_count)
            if target_symbol_count is not None
            else None,
            "max_spread_bps": float(max_spread_bps) if max_spread_bps is not None else None,
        },
        "thresholds": merged_thresholds,
        "best_candidate": passing.iloc[0].to_dict() if not passing.empty else None,
        "export_meta_variants": export_tokens,
        "export_all_selected_rows": bool(export_all_selected),
        "export_selected_variant_keys": [
            {
                "selector_variant": selector,
                "meta_variant": meta_variant,
                "keep_frac": keep_frac,
            }
            for selector, meta_variant, keep_frac in sorted(selected_key)
        ],
        "selected_rows": int(len(selected_rows)),
        "simple_policy_handoff_rows": int(len(handoff_candidates)),
        "simple_policy_handoff_note": (
            "Smoke-level OOS meta selected rows for exit-policy validation; "
            "not a production train_meta artifact."
        ),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-ledger-path", type=Path, default=None)
    parser.add_argument("--candidate-streams", type=str, default=",".join(DEFAULT_CANDIDATE_STREAMS))
    parser.add_argument("--keep-fracs", type=str, default=",".join(str(v) for v in DEFAULT_KEEP_FRACS))
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--max-side-share", type=float, default=0.70)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument(
        "--base-top-fracs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated top_fracs for base candidate-ledger generation. "
            "Defaults to the active base candidate top_frac."
        ),
    )
    parser.add_argument(
        "--simple-policy-barrier-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied only to simple_policy handoff barrier_pct; "
            "base/meta label-space metrics are unchanged."
        ),
    )
    parser.add_argument(
        "--spread-baseline-path",
        type=Path,
        default=None,
        help="Optional per-symbol spread baseline passed to base candidate-ledger generation.",
    )
    parser.add_argument(
        "--spread-rank-column",
        type=str,
        default="p75_spread_bps",
        help="Spread baseline column used to rank the available symbol universe.",
    )
    parser.add_argument(
        "--target-symbol-count",
        type=int,
        default=None,
        help="Keep the lowest-spread N symbols from the available label universe.",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=None,
        help="Optional absolute spread cap in bps applied before target-symbol-count.",
    )
    parser.add_argument(
        "--include-first-period",
        action="store_true",
        help=(
            "Include the earliest ledger month in OOF evaluation. "
            "Useful for diagnostics when the first month is needed for reporting."
        ),
    )
    parser.add_argument(
        "--export-meta-variants",
        type=str,
        default=None,
        help=(
            "Comma-separated meta variant names or substrings to export selected rows for. "
            "Use 'all' to export every evaluated variant. By default only a passing best "
            "candidate is exported."
        ),
    )
    parser.add_argument("--min-mean-u", type=float, default=None)
    parser.add_argument("--min-worst-month-mean-u", type=float, default=None)
    parser.add_argument("--max-bad-mae-rate", type=float, default=None)
    parser.add_argument("--max-timeout-rate", type=float, default=None)
    parser.add_argument("--max-month-bad-mae-rate", type=float, default=None)
    parser.add_argument("--max-month-timeout-rate", type=float, default=None)
    parser.add_argument("--min-final-oracle-recall", type=float, default=None)
    parser.add_argument("--max-selected-side-share", type=float, default=None)
    parser.add_argument("--min-selected-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.report_dir / DEFAULT_OUTPUT_SUBDIR)
    manifest = run_meta_smoke(
        report_dir=args.report_dir,
        output_dir=output_dir,
        candidate_streams=_parse_csv(args.candidate_streams, DEFAULT_CANDIDATE_STREAMS),
        keep_fracs=_parse_float_csv(args.keep_fracs, DEFAULT_KEEP_FRACS),
        candidate_ledger_path=args.candidate_ledger_path,
        seeds=_parse_int_csv(args.seeds, []) if args.seeds else None,
        train_lookback_months=args.train_lookback_months,
        max_feature_store_features=args.max_feature_store_features,
        max_side_share=float(args.max_side_share),
        thresholds={
            k: v
            for k, v in {
                "min_mean_u": args.min_mean_u,
                "min_worst_month_mean_u": args.min_worst_month_mean_u,
                "max_bad_mae_1r_rate": args.max_bad_mae_rate,
                "max_timeout_rate": args.max_timeout_rate,
                "max_month_bad_mae_1r_rate": args.max_month_bad_mae_rate,
                "max_month_timeout_rate": args.max_month_timeout_rate,
                "min_final_oracle_recall": args.min_final_oracle_recall,
                "max_selected_side_share": args.max_selected_side_share,
                "min_selected_rows": args.min_selected_rows,
            }.items()
            if v is not None
        },
        include_first_period=bool(args.include_first_period),
        min_train_rows=int(args.min_train_rows),
        simple_policy_barrier_multiplier=float(args.simple_policy_barrier_multiplier),
        base_top_fracs=_parse_float_csv(args.base_top_fracs, ()) if args.base_top_fracs else None,
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=str(args.spread_rank_column),
        target_symbol_count=args.target_symbol_count,
        max_spread_bps=args.max_spread_bps,
        export_meta_variants=_parse_csv(args.export_meta_variants, ())
        if args.export_meta_variants
        else None,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
