#!/usr/bin/env python3
"""Run a controlled canonical-context retraining experiment.

This script is deliberately diagnostic-only.  It tests whether the canonical
model-state and market-state context variables discovered by the bad-regime
diagnostics add recurring, economically useful information for selected
high-confidence failure targets.

The important contract is that canonical context variables are regenerated
inside each chronological fold using trailing/causal transforms.  The raw
archetype aliases, post-hoc probabilities, bad-week labels, adversarial scores,
and leaf outcome statistics are not used as training inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from extreme_price_movements.unsupervised_regime_learning.bad_regime_archetypes import (
    BadRegimeArchetypeFeatureConfig,
    build_bad_regime_archetype_feature_frame,
)
from scripts.diagnose_meta_recent_failures import (
    _assemble_selected_matrix,
    _base_models_for_head,
    _candidate_feature_contract,
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _known_export_features,
    _merge_feature_candidates,
    _normalise_keys,
    _prepare_model_matrix,
    _read_regime_features,
    _weekly_high_conf_metrics,
    _bad_recent_weeks,
    lgb,
)
from scripts.quantify_bad_regime_archetype_usefulness import _failure_targets, _pick_realized_return


CANDIDATES: dict[str, tuple[str, ...]] = {
    "long_dist": ("high_conf_miss", "high_conf_tail_loss"),
    "short_asset": ("high_conf_miss", "high_conf_tail_loss"),
    "short_boll": ("high_conf_miss", "high_conf_tail_loss"),
}

MODEL_STATE = (
    "prediction_support_quality",
    "prediction_reconstruction_anomaly",
    "prediction_path_instability",
    "regime_similarity_or_novelty",
)
MARKET_STATE = (
    "leverage_funding_crowding",
    "liquidity_participation_stress",
    "tail_volatility_stress",
    "relative_value_dislocation",
    "breadth_market_state",
    "network_concentration",
)
CANONICAL_CONTEXT = MODEL_STATE + MARKET_STATE
INTERACTIONS = (
    ("prediction_support_quality", "leverage_funding_crowding"),
    ("prediction_support_quality", "liquidity_participation_stress"),
    ("prediction_path_instability", "tail_volatility_stress"),
    ("prediction_reconstruction_anomaly", "relative_value_dislocation"),
    ("regime_similarity_or_novelty", "leverage_funding_crowding"),
)


@dataclass(frozen=True)
class FoldContext:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    fold_id: int


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(score)
    if int(mask.sum()) < 30:
        return np.nan
    yy = np.asarray(y[mask], dtype=np.int8)
    if len(np.unique(yy)) < 2:
        return np.nan
    return float(roc_auc_score(yy, score[mask]))


def _safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(score)
    if int(mask.sum()) < 30:
        return np.nan
    yy = np.asarray(y[mask], dtype=np.int8)
    if len(np.unique(yy)) < 2:
        return np.nan
    return float(average_precision_score(yy, score[mask]))


def _calibration_slope_intercept(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 50:
        return np.nan, np.nan
    yy = np.asarray(y[mask], dtype=np.float64)
    pp = np.clip(np.asarray(pred[mask], dtype=np.float64), 1e-5, 1.0 - 1e-5)
    if len(np.unique(yy.astype(np.int8))) < 2 or float(np.nanstd(pp)) <= 1e-12:
        return np.nan, np.nan
    logit = np.log(pp / (1.0 - pp))
    try:
        slope, intercept = np.polyfit(logit, yy, 1)
    except Exception:
        return np.nan, np.nan
    return float(slope), float(intercept)


def _failure_metrics(y: np.ndarray, risk: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(risk)
    yy = np.asarray(y[mask], dtype=np.int8)
    rr = np.asarray(returns[mask], dtype=np.float64)
    pp = np.clip(np.asarray(risk[mask], dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if yy.size == 0:
        return {"rows": 0, "reason": "empty"}
    out: dict[str, Any] = {
        "rows": int(yy.size),
        "failure_rate": float(np.mean(yy)) if yy.size else np.nan,
        "roc_auc": _safe_auc(yy.astype(np.float32), pp.astype(np.float32)),
        "pr_auc": _safe_pr_auc(yy.astype(np.float32), pp.astype(np.float32)),
        "log_loss": float(log_loss(yy, pp, labels=[0, 1])) if len(np.unique(yy)) >= 2 else np.nan,
        "brier": float(brier_score_loss(yy, pp)) if len(np.unique(yy)) >= 2 else np.nan,
    }
    slope, intercept = _calibration_slope_intercept(yy.astype(np.float32), pp.astype(np.float32))
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    for pct in (0.05, 0.10, 0.20):
        n = max(1, int(math.ceil(float(pct) * len(pp))))
        reject = np.zeros(len(pp), dtype=bool)
        reject[np.argsort(pp)[::-1][:n]] = True
        retain = ~reject
        suffix = f"{int(pct * 100)}pct"
        out[f"failure_capture_{suffix}"] = float(np.sum(yy[reject] > 0) / max(float(np.sum(yy > 0)), 1.0))
        out[f"retained_failure_rate_{suffix}"] = float(np.mean(yy[retain])) if retain.any() else np.nan
        ret_retain = rr[retain & np.isfinite(rr)]
        ret_all = rr[np.isfinite(rr)]
        out[f"retained_net_return_mean_{suffix}"] = float(np.nanmean(ret_retain)) if ret_retain.size else np.nan
        out[f"all_net_return_mean_{suffix}"] = float(np.nanmean(ret_all)) if ret_all.size else np.nan
        if ret_retain.size >= 20 and ret_all.size >= 20:
            out[f"tail_loss_delta_{suffix}"] = float(np.nanquantile(ret_retain, 0.05) - np.nanquantile(ret_all, 0.05))
        else:
            out[f"tail_loss_delta_{suffix}"] = np.nan
        rejected_winners = rr[reject & np.isfinite(rr) & (rr > 0.0)]
        out[f"winner_rejection_cost_{suffix}"] = float(np.nanmean(rejected_winners)) if rejected_winners.size else 0.0
        out[f"coverage_retained_{suffix}"] = float(np.mean(retain))
    for pct in (0.10, 0.20, 0.30):
        n = max(1, int(math.ceil(float(pct) * len(pp))))
        reliable = np.zeros(len(pp), dtype=bool)
        reliable[np.argsort(pp)[:n]] = True
        suffix = f"{int(pct * 100)}pct"
        top_y = yy[reliable]
        top_r = rr[reliable & np.isfinite(rr)]
        out[f"top_reliable_hit_rate_{suffix}"] = float(1.0 - np.mean(top_y)) if top_y.size else np.nan
        out[f"top_reliable_net_return_mean_{suffix}"] = float(np.nanmean(top_r)) if top_r.size else np.nan
        out[f"top_reliable_tail_loss_{suffix}"] = float(np.nanquantile(top_r, 0.05)) if top_r.size >= 20 else np.nan
    return out


def _ranking_diagnostics(y: np.ndarray, risk: np.ndarray, timestamps: pd.Series) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(risk)
    if int(mask.sum()) < 100:
        return {
            "weekly_auc_std": np.nan,
            "weekly_auc_positive_rate": np.nan,
            "weekly_rejection_turnover_10pct": np.nan,
        }
    yy = np.asarray(y[mask], dtype=np.float32)
    pp = np.asarray(risk[mask], dtype=np.float32)
    weeks = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True).iloc[np.flatnonzero(mask)], utc=True, errors="coerce")
    weeks = weeks.dt.to_period("W").dt.start_time.astype(str).to_numpy()
    aucs: list[float] = []
    rejection_sets: list[set[int]] = []
    row_ids = np.flatnonzero(mask)
    for week in pd.unique(weeks):
        ids = np.flatnonzero(weeks == week)
        if len(ids) < 50:
            continue
        auc = _safe_auc(yy[ids], pp[ids])
        if np.isfinite(auc):
            aucs.append(float(auc))
        n = max(1, int(math.ceil(0.10 * len(ids))))
        local_reject = ids[np.argsort(pp[ids])[::-1][:n]]
        rejection_sets.append(set(row_ids[local_reject].astype(int).tolist()))
    turnovers: list[float] = []
    for prev, cur in zip(rejection_sets[:-1], rejection_sets[1:]):
        union = len(prev | cur)
        if union:
            turnovers.append(1.0 - len(prev & cur) / union)
    auc_arr = np.asarray(aucs, dtype=np.float64)
    return {
        "weekly_auc_std": float(np.nanstd(auc_arr)) if auc_arr.size else np.nan,
        "weekly_auc_positive_rate": float(np.mean(auc_arr > 0.5)) if auc_arr.size else np.nan,
        "weekly_rejection_turnover_10pct": float(np.nanmean(turnovers)) if turnovers else np.nan,
    }


def _prediction_controls(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    for col in ("oof_pred", "oof_rank_pct", "oof_p_move", "oof_meta_clf", "oof_base_clf"):
        if col in panel.columns:
            out[col] = pd.to_numeric(panel[col], errors="coerce").astype("float32")
    if out.empty:
        out["constant_prediction_control"] = np.zeros(len(panel), dtype=np.float32)
    return _downcast_numeric(out)


def _load_canonical_definitions(path: Path) -> dict[str, dict[str, Any]]:
    canonical = pd.read_csv(path)
    defs: dict[str, dict[str, Any]] = {}
    for _, row in canonical.iterrows():
        name = str(row.get("canonical_variable", "")).strip()
        if not name or name == "trend_range_breakout":
            continue
        if name not in set(MODEL_STATE).union(MARKET_STATE):
            continue
        parents = [p.strip() for p in str(row.get("top_parent_features", "")).split(",") if p.strip()]
        if not parents:
            continue
        defs[name] = {
            "top_features": parents,
            "deployable_features": [name],
            "evidence_score": 1.0,
            "mechanism_channel": str(row.get("mechanism_channel", "")),
            "recommended_layer": str(row.get("state_family", "")),
            "status": "canonical_fold_fitted_experiment",
        }
    return defs


def _build_canonical_frame(
    frame: pd.DataFrame,
    definitions: dict[str, dict[str, Any]],
    *,
    trailing_window: int,
    min_periods: int,
    min_resolved_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = BadRegimeArchetypeFeatureConfig(
        trailing_window=int(trailing_window),
        min_periods=int(min_periods),
        min_resolved_features=int(min_resolved_features),
        archetype_prefix="canonical__",
        include_deployable_aliases=True,
        include_ranked_probability_aliases=False,
    )
    features, diagnostics = build_bad_regime_archetype_feature_frame(frame, definitions, config=config)
    canonical = pd.DataFrame(index=frame.index)
    for name in CANONICAL_CONTEXT:
        if name in features.columns:
            canonical[name] = pd.to_numeric(features[name], errors="coerce").astype("float32")
        else:
            canonical[name] = np.nan
    return _downcast_numeric(canonical), diagnostics


def _fold_canonical_features(
    raw: pd.DataFrame,
    folds: list[FoldContext],
    definitions: dict[str, dict[str, Any]],
    *,
    trailing_window: int,
    min_periods: int,
    min_resolved_features: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = pd.DataFrame(np.nan, index=raw.index, columns=list(MODEL_STATE + MARKET_STATE), dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []
    ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    for fold in folds:
        train_raw = raw.iloc[fold.train_idx].copy()
        valid_raw = raw.iloc[fold.valid_idx].copy()
        # Training features are computed only on the training fold.  Validation
        # features are computed on train + validation rows and use trailing
        # shifted statistics, matching live replay with prior history available.
        train_ctx, train_diag = _build_canonical_frame(
            train_raw,
            definitions,
            trailing_window=trailing_window,
            min_periods=min_periods,
            min_resolved_features=min_resolved_features,
        )
        combined = pd.concat([train_raw, valid_raw], axis=0)
        combined = combined.assign(__orig_idx=combined.index)
        combined = combined.sort_values(["timestamp", "symbol", "__orig_idx"], kind="mergesort")
        valid_ctx_all, valid_diag = _build_canonical_frame(
            combined.drop(columns=["__orig_idx"]),
            definitions,
            trailing_window=trailing_window,
            min_periods=min_periods,
            min_resolved_features=min_resolved_features,
        )
        valid_ctx_all.index = combined["__orig_idx"].to_numpy()
        out.iloc[fold.train_idx] = train_ctx.reindex(train_raw.index).to_numpy(dtype=np.float32, copy=False)
        out.iloc[fold.valid_idx] = valid_ctx_all.reindex(valid_raw.index)[out.columns].to_numpy(dtype=np.float32, copy=False)
        diagnostics.append(
            {
                "fold": int(fold.fold_id),
                "train_rows": int(len(fold.train_idx)),
                "valid_rows": int(len(fold.valid_idx)),
                "train_start": str(ts.iloc[fold.train_idx].min()),
                "train_end": str(ts.iloc[fold.train_idx].max()),
                "valid_start": str(ts.iloc[fold.valid_idx].min()),
                "valid_end": str(ts.iloc[fold.valid_idx].max()),
                "train_diagnostics": train_diag,
                "valid_diagnostics": valid_diag,
            }
        )
    return _downcast_numeric(out), diagnostics


def _parse_fresh_oos_start(value: str | None) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="raise")
    return pd.Timestamp(ts)


def _fresh_oos_indices(
    timestamps: pd.Series,
    fresh_start: pd.Timestamp,
    *,
    embargo_hours: int,
) -> dict[str, Any]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    cutoff = fresh_start - pd.Timedelta(hours=max(0, int(embargo_hours)))
    train_mask = ts.notna() & (ts < cutoff)
    test_mask = ts.notna() & (ts >= fresh_start)
    return {
        "fresh_oos_start": str(fresh_start),
        "fresh_oos_train_cutoff": str(cutoff),
        "train_idx": np.flatnonzero(train_mask.to_numpy(dtype=bool)),
        "test_idx": np.flatnonzero(test_mask.to_numpy(dtype=bool)),
        "train_start": str(ts.loc[train_mask].min()) if bool(train_mask.any()) else "",
        "train_end": str(ts.loc[train_mask].max()) if bool(train_mask.any()) else "",
        "test_start": str(ts.loc[test_mask].min()) if bool(test_mask.any()) else "",
        "test_end": str(ts.loc[test_mask].max()) if bool(test_mask.any()) else "",
    }


def _fresh_oos_canonical_features(
    raw: pd.DataFrame,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    definitions: dict[str, dict[str, Any]],
    trailing_window: int,
    min_periods: int,
    min_resolved_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(np.nan, index=raw.index, columns=list(CANONICAL_CONTEXT), dtype=np.float32)
    diagnostics: dict[str, Any] = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_output_feature_count": 0,
        "test_output_feature_count": 0,
    }
    if len(train_idx) == 0 and len(test_idx) == 0:
        diagnostics["reason"] = "empty_fresh_oos_split"
        return out, diagnostics

    train_raw = raw.iloc[np.asarray(train_idx, dtype=np.int64)].copy()
    train_ctx = pd.DataFrame(index=train_raw.index, columns=list(CANONICAL_CONTEXT), dtype=np.float32)
    train_diag: dict[str, Any] = {"output_feature_count": 0}
    if not train_raw.empty:
        train_ctx, train_diag = _build_canonical_frame(
            train_raw,
            definitions,
            trailing_window=trailing_window,
            min_periods=min_periods,
            min_resolved_features=min_resolved_features,
        )
        out.iloc[np.asarray(train_idx, dtype=np.int64)] = train_ctx.reindex(train_raw.index)[out.columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )

    test_raw = raw.iloc[np.asarray(test_idx, dtype=np.int64)].copy()
    test_diag: dict[str, Any] = {"output_feature_count": 0}
    if not test_raw.empty:
        combined = pd.concat([train_raw, test_raw], axis=0)
        combined = combined.assign(__orig_idx=combined.index)
        combined = combined.sort_values(["timestamp", "symbol", "__orig_idx"], kind="mergesort")
        test_ctx_all, test_diag = _build_canonical_frame(
            combined.drop(columns=["__orig_idx"]),
            definitions,
            trailing_window=trailing_window,
            min_periods=min_periods,
            min_resolved_features=min_resolved_features,
        )
        test_ctx_all.index = combined["__orig_idx"].to_numpy()
        out.iloc[np.asarray(test_idx, dtype=np.int64)] = test_ctx_all.reindex(test_raw.index)[out.columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )

    diagnostics.update(
        {
            "train_output_feature_count": int(train_diag.get("output_feature_count", 0)),
            "test_output_feature_count": int(test_diag.get("output_feature_count", 0)),
            "reason": "",
        }
    )
    return _downcast_numeric(out), diagnostics


def _make_chrono_folds(timestamps: pd.Series, n_splits: int, *, embargo_hours: int = 0) -> list[FoldContext]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    ordered_ts = ts.iloc[order].reset_index(drop=True)
    folds: list[FoldContext] = []
    splitter = TimeSeriesSplit(n_splits=max(2, int(n_splits)))
    for fold_id, (tr_pos, va_pos) in enumerate(splitter.split(order), start=1):
        train_pos = np.asarray(tr_pos, dtype=np.int64)
        if int(embargo_hours) > 0 and len(va_pos):
            valid_start = ordered_ts.iloc[np.asarray(va_pos, dtype=np.int64)].min()
            cutoff = valid_start - pd.Timedelta(hours=int(embargo_hours))
            train_pos = np.asarray(
                [
                    int(pos)
                    for pos in train_pos
                    if pd.notna(ordered_ts.iloc[int(pos)]) and ordered_ts.iloc[int(pos)] < cutoff
                ],
                dtype=np.int64,
            )
            if len(train_pos) == 0:
                train_pos = np.asarray(tr_pos, dtype=np.int64)
        folds.append(
            FoldContext(
                train_idx=order[train_pos].astype(np.int64, copy=False),
                valid_idx=order[va_pos].astype(np.int64, copy=False),
                fold_id=fold_id,
            )
        )
    return folds


def _period_stratified_train_sample(
    *,
    timestamps: pd.Series,
    y: np.ndarray,
    train_idx: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if int(max_rows) <= 0 or len(train_idx) <= int(max_rows):
        return np.asarray(train_idx, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    ts = pd.to_datetime(timestamps.iloc[train_idx], utc=True, errors="coerce")
    tmp = pd.DataFrame(
        {
            "idx": np.asarray(train_idx, dtype=np.int64),
            "period": ts.dt.to_period("W").astype(str).to_numpy(),
            "y": np.asarray(y, dtype=np.int8)[train_idx],
        }
    )
    target_frac = float(max_rows) / max(float(len(train_idx)), 1.0)
    pieces: list[np.ndarray] = []
    for _, group in tmp.groupby(["period", "y"], sort=False):
        take = min(len(group), max(1, int(round(len(group) * target_frac))))
        pieces.append(rng.choice(group["idx"].to_numpy(dtype=np.int64), size=take, replace=False))
    sampled = np.concatenate(pieces) if pieces else np.asarray(train_idx, dtype=np.int64)
    if len(sampled) > int(max_rows):
        sampled = rng.choice(sampled, size=int(max_rows), replace=False)
    return np.sort(sampled.astype(np.int64, copy=False))


def _fit_predict_lgbm(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[FoldContext],
    *,
    seed: int,
    max_depth: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for this experiment")
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]]
    x_prepared = _prepare_model_matrix(x)
    for fold in folds:
        tr = fold.train_idx
        va = fold.valid_idx
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            fold_rows.append({"fold": fold.fold_id, "reason": "insufficient_classes"})
            continue
        min_child = max(50, int(math.ceil(0.025 * len(tr))))
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.035,
            max_depth=int(max_depth),
            num_leaves=max(4, min(16, 2 ** int(max_depth))),
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=int(seed + fold.fold_id * 1009),
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(
                x_prepared.iloc[tr],
                y[tr],
                eval_set=[(x_prepared.iloc[va], y[va])],
                eval_metric="auc",
                callbacks=callbacks,
            )
        pred = clf.predict_proba(x_prepared.iloc[va])[:, 1].astype(np.float32, copy=False)
        oof[va] = pred
        fold_rows.append(
            {
                "fold": fold.fold_id,
                "reason": "",
                "train_rows": int(len(tr)),
                "valid_rows": int(len(va)),
                "feature_count": int(x_prepared.shape[1]),
                "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0),
                "valid_auc": _safe_auc(y[va].astype(np.float32), pred),
            }
        )
    return oof, fold_rows


def _fit_predict_lgbm_split(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    timestamps: pd.Series,
    seed: int,
    max_depth: int = 3,
    max_train_rows: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for this experiment")
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if len(train_idx) < 200 or len(test_idx) < 50:
        return np.full(len(test_idx), np.nan, dtype=np.float32), {
            "reason": "insufficient_train_or_test_rows",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return np.full(len(test_idx), np.nan, dtype=np.float32), {
            "reason": "insufficient_train_or_test_classes",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_positive_rate": float(np.mean(y[train_idx])) if len(train_idx) else np.nan,
            "test_positive_rate": float(np.mean(y[test_idx])) if len(test_idx) else np.nan,
        }
    train_idx = _period_stratified_train_sample(
        timestamps=timestamps,
        y=y,
        train_idx=train_idx,
        max_rows=int(max_train_rows),
        seed=int(seed),
    )
    x = x.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return np.full(len(test_idx), np.nan, dtype=np.float32), {
            "reason": "empty_matrix",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "feature_count": 0,
        }
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    min_child = max(50, int(math.ceil(0.025 * len(train_idx))))
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.035,
        max_depth=int(max_depth),
        num_leaves=max(4, min(16, 2 ** int(max_depth))),
        min_child_samples=min_child,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_prepared.iloc[train_idx], y[train_idx])
    pred = clf.predict_proba(x_prepared.iloc[test_idx])[:, 1].astype(np.float32, copy=False)
    return pred, {
        "reason": "",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "feature_count": int(len(keep_cols)),
        "train_positive_rate": float(np.mean(y[train_idx])),
        "test_positive_rate": float(np.mean(y[test_idx])),
    }


def _interaction_features(canonical: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=canonical.index)
    for left, right in INTERACTIONS:
        if left not in canonical.columns or right not in canonical.columns:
            out[f"{left}__x__{right}"] = np.nan
            continue
        l = pd.to_numeric(canonical[left], errors="coerce").to_numpy(dtype=np.float32)
        r = pd.to_numeric(canonical[right], errors="coerce").to_numpy(dtype=np.float32)
        out[f"{left}__x__{right}"] = (l * r).astype(np.float32, copy=False)
    return _downcast_numeric(out)


def _arm_frames(panel_high: pd.DataFrame, canonical: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    controls = _prediction_controls(panel_high)
    model_state = canonical.loc[:, list(MODEL_STATE)]
    market_state = canonical.loc[:, list(MARKET_STATE)]
    interactions = _interaction_features(canonical)
    return {
        "baseline_current_meta_unchanged": None,
        "canonical_model_state_context": pd.concat([controls, model_state], axis=1, copy=False),
        "canonical_market_state_context": pd.concat([controls, market_state], axis=1, copy=False),
        "model_state_x_market_state_interactions": pd.concat(
            [controls, model_state, market_state, interactions], axis=1, copy=False
        ),
        "auxiliary_failure_head": pd.concat([controls, model_state, market_state], axis=1, copy=False),
    }


def _episode_labels(panel: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").dt.to_period("W").dt.start_time.dt.strftime(
        "%Y-%m-%d"
    )


def _episode_effect_rows(
    *,
    head: str,
    target: str,
    arm: str,
    y: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    episodes: pd.Series,
    bad_episodes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode, idx in episodes.groupby(episodes).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        if len(ids) < 50:
            continue
        m = _failure_metrics(y[ids], pred[ids], returns[ids])
        b = _failure_metrics(y[ids], baseline_pred[ids], returns[ids])
        rows.append(
            {
                "head": head,
                "target": target,
                "arm": arm,
                "episode": str(episode),
                "is_bad_episode": str(episode) in bad_episodes,
                "rows": int(len(ids)),
                "roc_auc": m.get("roc_auc", np.nan),
                "baseline_roc_auc": b.get("roc_auc", np.nan),
                "delta_roc_auc": m.get("roc_auc", np.nan) - b.get("roc_auc", np.nan),
                "log_loss": m.get("log_loss", np.nan),
                "baseline_log_loss": b.get("log_loss", np.nan),
                "delta_log_loss_improvement": b.get("log_loss", np.nan) - m.get("log_loss", np.nan),
                "tail_loss_delta_10pct": m.get("tail_loss_delta_10pct", np.nan),
                "baseline_tail_loss_delta_10pct": b.get("tail_loss_delta_10pct", np.nan),
                "delta_tail_loss_10pct": m.get("tail_loss_delta_10pct", np.nan) - b.get("tail_loss_delta_10pct", np.nan),
            }
        )
    return rows


def _leave_one_episode_rows(
    *,
    head: str,
    target: str,
    arms: dict[str, pd.DataFrame | None],
    y: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.Series,
    bad_episodes: set[str],
    seed: int,
    max_train_rows: int,
    embargo_hours: int,
) -> list[dict[str, Any]]:
    episodes = _episode_labels(pd.DataFrame({"timestamp": timestamps}))
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    valid_y = np.isfinite(y)
    rows: list[dict[str, Any]] = []
    if len(bad_episodes) < 2:
        return [
            {
                "head": head,
                "target": target,
                "arm": "__summary__",
                "heldout_episode": "",
                "bad_episode_count": int(len(bad_episodes)),
                "transfer_reason": "insufficient_bad_episodes_for_leave_one_out",
            }
        ]
    for episode_i, episode in enumerate(sorted(bad_episodes), start=1):
        holdout = episodes.eq(str(episode)).to_numpy(dtype=bool)
        if int(holdout.sum()) < 50:
            continue
        train_mask = valid_y & ~holdout
        if int(embargo_hours) > 0:
            hold_ts = ts.loc[holdout]
            if not hold_ts.empty:
                start = hold_ts.min() - pd.Timedelta(hours=int(embargo_hours))
                end = hold_ts.max() + pd.Timedelta(hours=int(embargo_hours))
                train_mask &= ~((ts >= start) & (ts <= end)).to_numpy(dtype=bool)
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(valid_y & holdout)
        base_metrics = _failure_metrics(y[test_idx], baseline_pred[test_idx], returns[test_idx])
        for arm_i, (arm, x_arm) in enumerate(arms.items(), start=1):
            if arm == "baseline_current_meta_unchanged":
                pred = baseline_pred[test_idx].astype(np.float32, copy=True)
                fit_info = {
                    "reason": "unchanged_current_meta_reference",
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "feature_count": 1,
                }
            else:
                depth = 2 if arm == "model_state_x_market_state_interactions" else 3
                pred, fit_info = _fit_predict_lgbm_split(
                    x_arm,
                    y.astype(np.int8, copy=False),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    timestamps=timestamps.reset_index(drop=True),
                    seed=int(seed + episode_i * 1009 + arm_i * 917),
                    max_depth=depth,
                    max_train_rows=int(max_train_rows),
                )
            metrics = _failure_metrics(y[test_idx], pred, returns[test_idx])
            rows.append(
                {
                    "head": head,
                    "target": target,
                    "arm": arm,
                    "heldout_episode": str(episode),
                    "bad_episode_count": int(len(bad_episodes)),
                    "transfer_reason": str(fit_info.get("reason", "")),
                    "transfer_train_rows": int(fit_info.get("train_rows", 0)),
                    "transfer_test_rows": int(fit_info.get("test_rows", len(test_idx))),
                    "transfer_feature_count": int(fit_info.get("feature_count", 0)),
                    "roc_auc": metrics.get("roc_auc", np.nan),
                    "baseline_roc_auc": base_metrics.get("roc_auc", np.nan),
                    "delta_roc_auc": metrics.get("roc_auc", np.nan) - base_metrics.get("roc_auc", np.nan),
                    "log_loss": metrics.get("log_loss", np.nan),
                    "baseline_log_loss": base_metrics.get("log_loss", np.nan),
                    "delta_log_loss_improvement": base_metrics.get("log_loss", np.nan)
                    - metrics.get("log_loss", np.nan),
                    "pr_auc": metrics.get("pr_auc", np.nan),
                    "baseline_pr_auc": base_metrics.get("pr_auc", np.nan),
                    "delta_pr_auc": metrics.get("pr_auc", np.nan) - base_metrics.get("pr_auc", np.nan),
                    "tail_loss_delta_10pct": metrics.get("tail_loss_delta_10pct", np.nan),
                    "baseline_tail_loss_delta_10pct": base_metrics.get("tail_loss_delta_10pct", np.nan),
                    "delta_tail_loss_10pct": metrics.get("tail_loss_delta_10pct", np.nan)
                    - base_metrics.get("tail_loss_delta_10pct", np.nan),
                    "winner_rejection_cost_10pct": metrics.get("winner_rejection_cost_10pct", np.nan),
                    "baseline_winner_rejection_cost_10pct": base_metrics.get("winner_rejection_cost_10pct", np.nan),
                    "delta_winner_rejection_cost_10pct": metrics.get("winner_rejection_cost_10pct", np.nan)
                    - base_metrics.get("winner_rejection_cost_10pct", np.nan),
                }
            )
    return rows


def _summarize_leave_one(rows: list[dict[str, Any]], *, head: str, target: str, arm: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "leave_one_episode_count": 0,
            "median_leave_one_logloss_improvement": np.nan,
            "leave_one_episodes_improved_logloss": 0,
            "worst_leave_one_logloss_improvement": np.nan,
        }
    cur = df.loc[
        df.get("head", "").astype(str).eq(str(head))
        & df.get("target", "").astype(str).eq(str(target))
        & df.get("arm", "").astype(str).eq(str(arm))
        & df.get("transfer_reason", "").astype(str).eq("")
    ]
    if cur.empty or "delta_log_loss_improvement" not in cur:
        return {
            "leave_one_episode_count": 0,
            "median_leave_one_logloss_improvement": np.nan,
            "leave_one_episodes_improved_logloss": 0,
            "worst_leave_one_logloss_improvement": np.nan,
        }
    vals = pd.to_numeric(cur["delta_log_loss_improvement"], errors="coerce")
    return {
        "leave_one_episode_count": int(cur["heldout_episode"].nunique()),
        "median_leave_one_logloss_improvement": float(vals.median()),
        "leave_one_episodes_improved_logloss": int((vals > 0.0).sum()),
        "worst_leave_one_logloss_improvement": float(vals.min()),
    }


def _fresh_oos_eval_rows(
    *,
    head: str,
    target: str,
    arms: dict[str, pd.DataFrame | None],
    y: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fresh_start: pd.Timestamp,
    seed: int,
    max_train_rows: int,
) -> list[dict[str, Any]]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    if len(train_idx) < 200 or len(test_idx) < 50:
        for arm in arms:
            rows.append(
                {
                    "head": head,
                    "target": target,
                    "arm": arm,
                    "status": "not_evaluable",
                    "fresh_oos_start": str(fresh_start),
                    "reason": "insufficient_fresh_oos_train_or_test_rows",
                    "fresh_oos_train_rows": int(len(train_idx)),
                    "fresh_oos_test_rows": int(len(test_idx)),
                }
            )
        return rows

    valid_test = np.isfinite(y[test_idx]) & np.isfinite(baseline_pred[test_idx])
    eval_test_idx = test_idx[valid_test]
    if len(eval_test_idx) < 50 or len(np.unique(y[eval_test_idx].astype(np.int8))) < 2:
        for arm in arms:
            rows.append(
                {
                    "head": head,
                    "target": target,
                    "arm": arm,
                    "status": "not_evaluable",
                    "fresh_oos_start": str(fresh_start),
                    "reason": "insufficient_fresh_oos_test_classes",
                    "fresh_oos_train_rows": int(len(train_idx)),
                    "fresh_oos_test_rows": int(len(eval_test_idx)),
                }
            )
        return rows

    base_metrics = _failure_metrics(y[eval_test_idx], baseline_pred[eval_test_idx], returns[eval_test_idx])
    for arm_i, (arm, x_arm) in enumerate(arms.items(), start=1):
        if arm == "baseline_current_meta_unchanged":
            pred = baseline_pred[eval_test_idx].astype(np.float32, copy=True)
            fit_info = {
                "reason": "unchanged_current_meta_reference",
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(eval_test_idx)),
                "feature_count": 1,
            }
            status = "reference"
        else:
            depth = 2 if arm == "model_state_x_market_state_interactions" else 3
            pred, fit_info = _fit_predict_lgbm_split(
                x_arm,
                y.astype(np.int8, copy=False),
                train_idx=train_idx,
                test_idx=eval_test_idx,
                timestamps=timestamps.reset_index(drop=True),
                seed=int(seed + arm_i * 1009),
                max_depth=depth,
                max_train_rows=int(max_train_rows),
            )
            status = "evaluated" if str(fit_info.get("reason", "")) == "" else "not_evaluable"
        metrics = _failure_metrics(y[eval_test_idx], pred, returns[eval_test_idx])
        rows.append(
            {
                "head": head,
                "target": target,
                "arm": arm,
                "status": status,
                "fresh_oos_start": str(fresh_start),
                "reason": str(fit_info.get("reason", "")),
                "fresh_oos_train_rows": int(fit_info.get("train_rows", len(train_idx))),
                "fresh_oos_test_rows": int(fit_info.get("test_rows", len(eval_test_idx))),
                "fresh_oos_feature_count": int(fit_info.get("feature_count", 0)),
                "fresh_oos_train_positive_rate": float(fit_info.get("train_positive_rate", np.nan)),
                "fresh_oos_test_positive_rate": float(fit_info.get("test_positive_rate", np.nan)),
                "roc_auc": metrics.get("roc_auc", np.nan),
                "baseline_roc_auc": base_metrics.get("roc_auc", np.nan),
                "delta_roc_auc": metrics.get("roc_auc", np.nan) - base_metrics.get("roc_auc", np.nan),
                "log_loss": metrics.get("log_loss", np.nan),
                "baseline_log_loss": base_metrics.get("log_loss", np.nan),
                "delta_log_loss_improvement": base_metrics.get("log_loss", np.nan) - metrics.get("log_loss", np.nan),
                "pr_auc": metrics.get("pr_auc", np.nan),
                "baseline_pr_auc": base_metrics.get("pr_auc", np.nan),
                "delta_pr_auc": metrics.get("pr_auc", np.nan) - base_metrics.get("pr_auc", np.nan),
                "tail_loss_delta_10pct": metrics.get("tail_loss_delta_10pct", np.nan),
                "baseline_tail_loss_delta_10pct": base_metrics.get("tail_loss_delta_10pct", np.nan),
                "delta_tail_loss_10pct": metrics.get("tail_loss_delta_10pct", np.nan)
                - base_metrics.get("tail_loss_delta_10pct", np.nan),
                "winner_rejection_cost_10pct": metrics.get("winner_rejection_cost_10pct", np.nan),
                "baseline_winner_rejection_cost_10pct": base_metrics.get("winner_rejection_cost_10pct", np.nan),
                "delta_winner_rejection_cost_10pct": metrics.get("winner_rejection_cost_10pct", np.nan)
                - base_metrics.get("winner_rejection_cost_10pct", np.nan),
            }
        )
    return rows


def _summarize_fresh_oos(rows: list[dict[str, Any]], *, head: str, target: str, arm: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    empty = {
        "fresh_oos_evaluated": False,
        "fresh_oos_status": "not_evaluated",
        "fresh_oos_reason": "fresh OOS not requested or not available",
        "fresh_oos_train_rows": 0,
        "fresh_oos_test_rows": 0,
        "fresh_oos_delta_log_loss_improvement": np.nan,
        "fresh_oos_delta_pr_auc": np.nan,
        "fresh_oos_delta_tail_loss_10pct": np.nan,
        "fresh_oos_delta_winner_rejection_cost_10pct": np.nan,
    }
    if df.empty:
        return empty
    cur = df.loc[
        df.get("head", "").astype(str).eq(str(head))
        & df.get("target", "").astype(str).eq(str(target))
        & df.get("arm", "").astype(str).eq(str(arm))
    ]
    if cur.empty:
        return empty
    rec = cur.iloc[0]
    status = str(rec.get("status", "not_evaluable"))
    evaluated = status in {"reference", "evaluated"}
    return {
        "fresh_oos_evaluated": bool(evaluated),
        "fresh_oos_status": status,
        "fresh_oos_reason": str(rec.get("reason", "")),
        "fresh_oos_train_rows": int(rec.get("fresh_oos_train_rows", 0) or 0),
        "fresh_oos_test_rows": int(rec.get("fresh_oos_test_rows", 0) or 0),
        "fresh_oos_roc_auc": float(rec.get("roc_auc", np.nan)),
        "fresh_oos_pr_auc": float(rec.get("pr_auc", np.nan)),
        "fresh_oos_log_loss": float(rec.get("log_loss", np.nan)),
        "fresh_oos_baseline_log_loss": float(rec.get("baseline_log_loss", np.nan)),
        "fresh_oos_delta_log_loss_improvement": float(rec.get("delta_log_loss_improvement", np.nan)),
        "fresh_oos_delta_pr_auc": float(rec.get("delta_pr_auc", np.nan)),
        "fresh_oos_delta_tail_loss_10pct": float(rec.get("delta_tail_loss_10pct", np.nan)),
        "fresh_oos_delta_winner_rejection_cost_10pct": float(
            rec.get("delta_winner_rejection_cost_10pct", np.nan)
        ),
    }


def _score_arm(
    *,
    head: str,
    target: str,
    arm: str,
    y: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.Series,
    bad_episodes: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = np.isfinite(y) & np.isfinite(pred) & np.isfinite(baseline_pred)
    y_eval = y[common]
    pred_eval = pred[common]
    baseline_eval = baseline_pred[common]
    returns_eval = returns[common]
    timestamps_eval = pd.Series(timestamps).reset_index(drop=True).iloc[np.flatnonzero(common)].reset_index(drop=True)
    metrics = _failure_metrics(y_eval, pred_eval, returns_eval)
    baseline_metrics = _failure_metrics(y_eval, baseline_eval, returns_eval)
    ranking = _ranking_diagnostics(y_eval, pred_eval, timestamps_eval)
    baseline_ranking = _ranking_diagnostics(y_eval, baseline_eval, timestamps_eval)
    episodes = _episode_labels(pd.DataFrame({"timestamp": timestamps_eval}))
    episode_rows = _episode_effect_rows(
        head=head,
        target=target,
        arm=arm,
        y=y_eval,
        pred=pred_eval,
        baseline_pred=baseline_eval,
        returns=returns_eval,
        episodes=episodes,
        bad_episodes=bad_episodes,
    )
    episode_df = pd.DataFrame(episode_rows)
    bad_ep = episode_df.loc[episode_df.get("is_bad_episode", False).astype(bool)] if not episode_df.empty else pd.DataFrame()
    normal_ep = episode_df.loc[~episode_df.get("is_bad_episode", False).astype(bool)] if not episode_df.empty else pd.DataFrame()
    row = {
        "head": head,
        "target": target,
        "arm": arm,
        **metrics,
        "scored_coverage": float(np.mean(common)) if len(common) else 0.0,
        "delta_log_loss_improvement": baseline_metrics.get("log_loss", np.nan) - metrics.get("log_loss", np.nan),
        "delta_roc_auc": metrics.get("roc_auc", np.nan) - baseline_metrics.get("roc_auc", np.nan),
        "delta_pr_auc": metrics.get("pr_auc", np.nan) - baseline_metrics.get("pr_auc", np.nan),
        "delta_tail_loss_10pct": metrics.get("tail_loss_delta_10pct", np.nan)
        - baseline_metrics.get("tail_loss_delta_10pct", np.nan),
        "delta_winner_rejection_cost_10pct": metrics.get("winner_rejection_cost_10pct", np.nan)
        - baseline_metrics.get("winner_rejection_cost_10pct", np.nan),
        "episode_count": int(len(episode_df)),
        "bad_episode_count": int(len(bad_ep)),
        "median_bad_episode_logloss_improvement": float(bad_ep["delta_log_loss_improvement"].median())
        if not bad_ep.empty and "delta_log_loss_improvement" in bad_ep
        else np.nan,
        "episodes_improved_logloss": int((bad_ep["delta_log_loss_improvement"] > 0.0).sum())
        if not bad_ep.empty and "delta_log_loss_improvement" in bad_ep
        else 0,
        "worst_bad_episode_logloss_improvement": float(bad_ep["delta_log_loss_improvement"].min())
        if not bad_ep.empty and "delta_log_loss_improvement" in bad_ep
        else np.nan,
        "normal_episode_median_logloss_improvement": float(normal_ep["delta_log_loss_improvement"].median())
        if not normal_ep.empty and "delta_log_loss_improvement" in normal_ep
        else np.nan,
        "weekly_auc_std": ranking.get("weekly_auc_std", np.nan),
        "baseline_weekly_auc_std": baseline_ranking.get("weekly_auc_std", np.nan),
        "delta_weekly_auc_std": baseline_ranking.get("weekly_auc_std", np.nan) - ranking.get("weekly_auc_std", np.nan),
        "weekly_auc_positive_rate": ranking.get("weekly_auc_positive_rate", np.nan),
        "weekly_rejection_turnover_10pct": ranking.get("weekly_rejection_turnover_10pct", np.nan),
    }
    return row, episode_rows


def _go_no_go(row: pd.Series, *, fresh_oos_evaluated: bool) -> tuple[str, str]:
    if str(row.get("arm")) == "baseline_current_meta_unchanged":
        return "baseline", "reference arm"
    incremental = bool(
        (float(row.get("delta_log_loss_improvement", np.nan)) > 0.0)
        or (float(row.get("delta_pr_auc", np.nan)) > 0.0)
    )
    recurrence = bool(
        float(row.get("median_bad_episode_logloss_improvement", np.nan)) > 0.0
        and int(row.get("episodes_improved_logloss", 0)) >= max(1, math.ceil(0.5 * int(row.get("bad_episode_count", 0))))
    )
    leave_one_count = int(row.get("leave_one_episode_count", 0) or 0)
    leave_one_recurrence = bool(
        leave_one_count >= 2
        and float(row.get("median_leave_one_logloss_improvement", np.nan)) > 0.0
        and int(row.get("leave_one_episodes_improved_logloss", 0)) >= max(1, math.ceil(0.5 * leave_one_count))
    )
    recurrence = recurrence and leave_one_recurrence
    economic = bool(
        np.nan_to_num(float(row.get("delta_tail_loss_10pct", np.nan)), nan=-np.inf) >= 0.0
        and np.nan_to_num(float(row.get("delta_winner_rejection_cost_10pct", np.nan)), nan=np.inf) <= 0.0
    )
    normal = bool(np.nan_to_num(float(row.get("normal_episode_median_logloss_improvement", np.nan)), nan=0.0) >= -0.002)
    row_fresh_evaluated = bool(fresh_oos_evaluated or row.get("fresh_oos_evaluated", False))
    fresh_incremental = bool(
        (float(row.get("fresh_oos_delta_log_loss_improvement", np.nan)) > 0.0)
        or (float(row.get("fresh_oos_delta_pr_auc", np.nan)) > 0.0)
    )
    fresh_economic = bool(
        np.nan_to_num(float(row.get("fresh_oos_delta_tail_loss_10pct", np.nan)), nan=-np.inf) >= 0.0
        and np.nan_to_num(float(row.get("fresh_oos_delta_winner_rejection_cost_10pct", np.nan)), nan=np.inf)
        <= 0.0
    )
    fresh_ok = bool(fresh_incremental and fresh_economic)
    if incremental and recurrence and economic and normal:
        if not row_fresh_evaluated:
            return (
                "research_candidate_pending_fresh_oos",
                "incremental, recurring, economically non-worse, and normal-period-safe; fresh OOS not evaluated",
            )
        if not fresh_ok:
            return "reject", "fresh OOS confirmation failed"
        return "candidate", "incremental, recurring, economically non-worse, normal-period-safe, and fresh-OOS evaluated"
    reasons = []
    if not incremental:
        reasons.append("no incremental log-loss/PR lift")
    if not recurrence:
        reasons.append("episode/leave-one recurrence gate failed")
    if not economic:
        reasons.append("economic tail/winner gate failed")
    if not normal:
        reasons.append("normal-period damage")
    if row_fresh_evaluated and not fresh_ok:
        reasons.append("fresh OOS confirmation failed")
    return "reject", "; ".join(reasons)


def _assemble_head_context(
    *,
    head: Any,
    panel: pd.DataFrame,
    race: Any,
    base_bundle: dict[str, Any],
    feature_dir: Path,
    transform_cache: Path | None,
    symbol_columns: dict[str, set[str]],
    regime_context: Path | None,
    max_regime_columns: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_x, _coverage, _summary = _assemble_selected_matrix(
        panel=panel,
        race=race,
        feature_dir=feature_dir,
        transform_cache=transform_cache,
        symbol_columns=symbol_columns,
    )
    base_selected_x = pd.DataFrame(index=panel.index)
    base_models, base_features = _base_models_for_head(base_bundle, head)
    if base_models and base_features:
        fake_race = type("FakeRace", (), {})()
        fake_best = type("FakeBest", (), {})()
        fake_best.selected_features = list(base_features)
        fake_best.get_training_meta_features = lambda: pd.DataFrame(index=panel.index)
        fake_best.model_effectiveness_history_defaults_ = {}
        fake_best.feature_stats_train = {}
        fake_race.best_model = fake_best
        base_selected_x, _base_cov, _base_summary = _assemble_selected_matrix(
            panel=panel,
            race=fake_race,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
    parts = [selected_x]
    if not base_selected_x.empty:
        extra = [c for c in base_selected_x.columns if c not in selected_x.columns]
        if extra:
            parts.append(base_selected_x[extra])
    export_x = _known_export_features(panel)
    regime_x = (
        _read_regime_features(regime_context, panel[["timestamp", "symbol"]], max_regime_columns)
        if regime_context is not None
        else pd.DataFrame(index=panel.index)
    )
    candidate_x = _merge_feature_candidates(pd.concat(parts, axis=1, copy=False), export_x, regime_x)
    raw = pd.concat([panel[["timestamp", "symbol"]].reset_index(drop=True), candidate_x.reset_index(drop=True)], axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    return candidate_x, raw


def _audit_status(ok: bool, *, blocker: str = "", waived: bool = False) -> str:
    if waived:
        return "waived"
    if ok:
        return "passed"
    return "blocked" if blocker else "failed"


def _audit_item(
    requirement: str,
    *,
    ok: bool,
    evidence: dict[str, Any],
    blocker: str = "",
    waived: bool = False,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": _audit_status(ok, blocker=blocker, waived=waived),
        "metrics": metrics or {},
        "evidence": evidence,
        "blocker": blocker,
    }


def _jsonable_records(df: pd.DataFrame, columns: list[str], *, n: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    view = df[[c for c in columns if c in df.columns]].copy()
    if n is not None:
        view = view.head(int(n))
    records: list[dict[str, Any]] = []
    for rec in view.to_dict(orient="records"):
        out: dict[str, Any] = {}
        for key, value in rec.items():
            if isinstance(value, (np.floating, float)):
                out[key] = None if not np.isfinite(float(value)) else round(float(value), 6)
            elif isinstance(value, (np.integer, int)):
                out[key] = int(value)
            elif pd.isna(value):
                out[key] = None
            else:
                out[key] = value
        records.append(out)
    return records


def _retrain_outcomes(
    *,
    summary: pd.DataFrame,
    leave_one: pd.DataFrame,
    context_diagnostics: pd.DataFrame,
    fresh_oos: pd.DataFrame,
) -> dict[str, Any]:
    if summary.empty:
        return {"summary_rows": 0}
    recommendation_counts = (
        summary.groupby(["arm", "recommendation"]).size().reset_index(name="rows").sort_values(["arm", "recommendation"])
        if {"arm", "recommendation"} <= set(summary.columns)
        else pd.DataFrame()
    )
    non_baseline = summary.loc[
        summary.get("arm", pd.Series("", index=summary.index)).astype(str).ne("baseline_current_meta_unchanged")
    ].copy()
    metric_cols = [
        "head",
        "target",
        "arm",
        "recommendation",
        "delta_log_loss_improvement",
        "delta_pr_auc",
        "delta_tail_loss_10pct",
        "delta_winner_rejection_cost_10pct",
        "median_bad_episode_logloss_improvement",
        "episodes_improved_logloss",
        "bad_episode_count",
        "median_leave_one_logloss_improvement",
        "leave_one_episodes_improved_logloss",
        "leave_one_episode_count",
        "normal_episode_median_logloss_improvement",
        "decision_reason",
    ]
    best_overall = (
        non_baseline.sort_values("delta_log_loss_improvement", ascending=False)
        if "delta_log_loss_improvement" in non_baseline
        else non_baseline
    )
    best_by_pair = (
        best_overall.groupby(["head", "target"], as_index=False, sort=False).head(1)
        if {"head", "target"} <= set(best_overall.columns)
        else pd.DataFrame()
    )
    baseline_cols = [
        "head",
        "target",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "weekly_auc_std",
        "top_reliable_hit_rate_10pct",
        "top_reliable_net_return_mean_10pct",
    ]
    baseline = summary.loc[
        summary.get("arm", pd.Series("", index=summary.index)).astype(str).eq("baseline_current_meta_unchanged")
    ]
    reason_counts = pd.DataFrame()
    if "decision_reason" in non_baseline:
        reason_counts = (
            non_baseline["decision_reason"]
            .fillna("")
            .astype(str)
            .str.split("; ")
            .explode()
            .loc[lambda s: s.ne("")]
            .value_counts()
            .rename_axis("reason")
            .reset_index(name="rows")
        )
    context_summary = pd.DataFrame()
    if {"head", "fold", "train_rows", "valid_rows", "valid_output_feature_count"} <= set(context_diagnostics.columns):
        context_summary = (
            context_diagnostics.loc[context_diagnostics["fold"].astype(str).ne("fresh_oos")]
            .groupby("head", as_index=False)
            .agg(
                folds=("fold", "nunique"),
                min_train_rows=("train_rows", "min"),
                max_train_rows=("train_rows", "max"),
                valid_rows=("valid_rows", "min"),
                valid_features=("valid_output_feature_count", "min"),
            )
        )
    leave_one_summary = pd.DataFrame()
    if {"head", "target", "heldout_episode"} <= set(leave_one.columns):
        leave_one_summary = (
            leave_one.groupby(["head", "target"], as_index=False)
            .agg(heldout_episodes=("heldout_episode", "nunique"))
            .sort_values(["head", "target"])
        )
    fresh_status = (
        fresh_oos.get("status", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        if not fresh_oos.empty
        else []
    )
    return {
        "summary_rows": int(len(summary)),
        "candidate_rows": int(summary.get("recommendation", pd.Series(dtype=str)).astype(str).eq("candidate").sum()),
        "research_pending_rows": int(
            summary.get("recommendation", pd.Series(dtype=str))
            .astype(str)
            .eq("research_candidate_pending_fresh_oos")
            .sum()
        ),
        "recommendation_counts": _jsonable_records(recommendation_counts, ["arm", "recommendation", "rows"]),
        "baseline_metrics": _jsonable_records(baseline, baseline_cols),
        "best_non_baseline_by_logloss": _jsonable_records(best_overall, metric_cols, n=12),
        "best_non_baseline_by_head_target": _jsonable_records(best_by_pair, metric_cols),
        "rejection_reason_counts": _jsonable_records(reason_counts, ["reason", "rows"]),
        "context_fold_metrics": _jsonable_records(
            context_summary,
            ["head", "folds", "min_train_rows", "max_train_rows", "valid_rows", "valid_features"],
        ),
        "leave_one_episode_counts": _jsonable_records(leave_one_summary, ["head", "target", "heldout_episodes"]),
        "fresh_oos_status_values": sorted(fresh_status),
    }


def _build_requirement_audit(
    *,
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    leave_one: pd.DataFrame,
    fresh_oos: pd.DataFrame,
    context_diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested_heads = set(CANDIDATES)
    if args.only_head:
        requested_heads &= {str(x) for x in args.only_head}
    requested_pairs = {(head, target) for head in requested_heads for target in CANDIDATES[head]}
    expected_arms = {
        "baseline_current_meta_unchanged",
        "canonical_model_state_context",
        "canonical_market_state_context",
        "model_state_x_market_state_interactions",
        "auxiliary_failure_head",
    }
    expected_rows = len(requested_pairs) * len(expected_arms)
    items: list[dict[str, Any]] = []

    observed_heads = set(summary.get("head", pd.Series(dtype=str)).dropna().astype(str).unique())
    observed_pairs = set(
        zip(
            summary.get("head", pd.Series(dtype=str)).dropna().astype(str),
            summary.get("target", pd.Series(dtype=str)).dropna().astype(str),
        )
    )
    observed_arms = set(summary.get("arm", pd.Series(dtype=str)).dropna().astype(str).unique())
    items.append(
        _audit_item(
            "minimal_head_target_arm_matrix",
            ok=(
                len(summary) == expected_rows
                and observed_heads == requested_heads
                and observed_pairs == requested_pairs
                and observed_arms == expected_arms
                and "long_bars" not in observed_heads
            ),
            evidence={
                "expected_rows": expected_rows,
                "observed_rows": int(len(summary)),
                "requested_heads": sorted(requested_heads),
                "observed_heads": sorted(observed_heads),
                "observed_arms": sorted(observed_arms),
                "long_bars_present": "long_bars" in observed_heads,
            },
            metrics={
                "rows": f"{int(len(summary))}/{expected_rows}",
                "heads": len(observed_heads),
                "targets": int(summary.get("target", pd.Series(dtype=str)).dropna().astype(str).nunique()),
                "arms": len(observed_arms),
                "long_bars_present": "long_bars" in observed_heads,
            },
        )
    )

    required_metric_cols = {
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier",
        "calibration_slope",
        "calibration_intercept",
        "top_reliable_hit_rate_10pct",
        "top_reliable_net_return_mean_10pct",
        "top_reliable_tail_loss_10pct",
        "delta_tail_loss_10pct",
        "delta_winner_rejection_cost_10pct",
        "weekly_auc_std",
        "scored_coverage",
        "weekly_rejection_turnover_10pct",
        "normal_episode_median_logloss_improvement",
        "median_bad_episode_logloss_improvement",
        "worst_bad_episode_logloss_improvement",
        "episodes_improved_logloss",
    }
    missing_metrics = sorted(required_metric_cols - set(summary.columns))
    items.append(
        _audit_item(
            "primary_metrics_present",
            ok=not missing_metrics,
            evidence={"missing_metric_columns": missing_metrics, "checked_columns": sorted(required_metric_cols)},
            metrics={"checked": len(required_metric_cols), "missing": len(missing_metrics)},
        )
    )

    arm_series = summary.get("arm", pd.Series("", index=summary.index)).astype(str)
    non_baseline = summary.loc[arm_series.ne("baseline_current_meta_unchanged")].copy()
    contract_cols = {"fold_fitted", "causal_trailing", "live_equivalent", "raw_alias_outputs_used", "bad_contract_feature_count"}
    contract_columns_present = contract_cols <= set(summary.columns)
    contract_ok = False
    if contract_columns_present and not non_baseline.empty:
        contract_ok = bool(
            non_baseline["fold_fitted"].astype(bool).all()
            and non_baseline["causal_trailing"].astype(bool).all()
            and non_baseline["live_equivalent"].astype(bool).all()
            and non_baseline["raw_alias_outputs_used"].astype(bool).eq(False).all()
            and pd.to_numeric(non_baseline["bad_contract_feature_count"], errors="coerce").fillna(1).eq(0).all()
        )
    items.append(
        _audit_item(
            "clean_feature_contract_for_context_arms",
            ok=contract_ok,
            evidence={
                "contract_columns_present": bool(contract_columns_present),
                "fold_fitted_all": bool(non_baseline.get("fold_fitted", pd.Series(False)).astype(bool).all())
                if "fold_fitted" in non_baseline
                else False,
                "causal_trailing_all": bool(non_baseline.get("causal_trailing", pd.Series(False)).astype(bool).all())
                if "causal_trailing" in non_baseline
                else False,
                "live_equivalent_all": bool(non_baseline.get("live_equivalent", pd.Series(False)).astype(bool).all())
                if "live_equivalent" in non_baseline
                else False,
                "raw_alias_outputs_used_any": bool(
                    non_baseline.get("raw_alias_outputs_used", pd.Series(True)).astype(bool).any()
                )
                if "raw_alias_outputs_used" in non_baseline
                else True,
                "max_bad_contract_feature_count": float(
                    pd.to_numeric(non_baseline.get("bad_contract_feature_count", pd.Series([np.nan])), errors="coerce").max()
                ),
            },
            metrics={
                "context_arm_rows": int(len(non_baseline)),
                "bad_contract_max": float(
                    pd.to_numeric(non_baseline.get("bad_contract_feature_count", pd.Series([np.nan])), errors="coerce").max()
                ),
                "raw_alias_any": bool(
                    non_baseline.get("raw_alias_outputs_used", pd.Series(True, index=non_baseline.index))
                    .astype(bool)
                    .any()
                )
                if not non_baseline.empty
                else False,
            },
        )
    )

    fold_counts = (
        fold_metrics.groupby(["head", "target", "arm"])["fold"].nunique()
        if {"head", "target", "arm", "fold"} <= set(fold_metrics.columns)
        else pd.Series(dtype=np.float64)
    )
    expected_fold_groups = expected_rows
    items.append(
        _audit_item(
            "nested_chronological_oof_complete",
            ok=(
                len(fold_counts) == expected_fold_groups
                and not fold_counts.empty
                and int(fold_counts.min()) == int(args.outer_folds)
                and int(fold_counts.max()) == int(args.outer_folds)
            ),
            evidence={
                "expected_groups": expected_fold_groups,
                "observed_groups": int(len(fold_counts)),
                "expected_outer_folds": int(args.outer_folds),
                "min_folds_per_group": int(fold_counts.min()) if not fold_counts.empty else 0,
                "max_folds_per_group": int(fold_counts.max()) if not fold_counts.empty else 0,
                "embargo_hours": int(args.embargo_hours),
            },
            metrics={
                "groups": f"{int(len(fold_counts))}/{expected_fold_groups}",
                "folds_per_group": f"{int(fold_counts.min()) if not fold_counts.empty else 0}-{int(fold_counts.max()) if not fold_counts.empty else 0}",
                "embargo_hours": int(args.embargo_hours),
            },
        )
    )

    context_fold_counts = (
        context_diagnostics.loc[context_diagnostics.get("fold", "").astype(str).ne("fresh_oos")]
        .groupby("head")["fold"]
        .nunique()
        if {"head", "fold"} <= set(context_diagnostics.columns)
        else pd.Series(dtype=np.float64)
    )
    items.append(
        _audit_item(
            "fold_fitted_canonical_context_written",
            ok=(
                set(context_fold_counts.index.astype(str)) == requested_heads
                and not context_fold_counts.empty
                and int(context_fold_counts.min()) == int(args.outer_folds)
                and pd.to_numeric(context_diagnostics.get("valid_output_feature_count", pd.Series([0])), errors="coerce")
                .fillna(0)
                .ge(len(CANONICAL_CONTEXT))
                .all()
            ),
            evidence={
                "context_heads": sorted(context_fold_counts.index.astype(str).tolist()),
                "min_context_folds": int(context_fold_counts.min()) if not context_fold_counts.empty else 0,
                "expected_context_features": len(CANONICAL_CONTEXT),
                "min_valid_output_feature_count": int(
                    pd.to_numeric(context_diagnostics.get("valid_output_feature_count", pd.Series([0])), errors="coerce")
                    .fillna(0)
                    .min()
                ),
            },
            metrics={
                "heads": len(context_fold_counts),
                "min_folds": int(context_fold_counts.min()) if not context_fold_counts.empty else 0,
                "min_valid_features": int(
                    pd.to_numeric(context_diagnostics.get("valid_output_feature_count", pd.Series([0])), errors="coerce")
                    .fillna(0)
                    .min()
                ),
                "expected_context_features": len(CANONICAL_CONTEXT),
            },
        )
    )

    loo_counts = (
        leave_one.groupby(["head", "target", "arm"])["heldout_episode"].nunique()
        if {"head", "target", "arm", "heldout_episode"} <= set(leave_one.columns)
        else pd.Series(dtype=np.float64)
    )
    items.append(
        _audit_item(
            "leave_one_episode_evaluation_present",
            ok=len(loo_counts) == expected_rows and not loo_counts.empty and int(loo_counts.min()) >= 1,
            evidence={
                "expected_groups": expected_rows,
                "observed_groups": int(len(loo_counts)),
                "min_heldout_episodes": int(loo_counts.min()) if not loo_counts.empty else 0,
                "max_heldout_episodes": int(loo_counts.max()) if not loo_counts.empty else 0,
            },
            metrics={
                "groups": f"{int(len(loo_counts))}/{expected_rows}",
                "heldout_episodes": f"{int(loo_counts.min()) if not loo_counts.empty else 0}-{int(loo_counts.max()) if not loo_counts.empty else 0}",
            },
        )
    )

    fresh_requested = bool(str(args.fresh_oos_start or "").strip())
    assume_oof_final = bool(getattr(args, "assume_oof_final", False))
    fresh_evaluated = bool(summary.get("fresh_oos_evaluated", pd.Series(dtype=bool)).astype(bool).any())
    fresh_blocker = "" if fresh_evaluated else "no untouched later period supplied via --fresh-oos-start"
    if fresh_requested and not fresh_evaluated:
        fresh_blocker = "fresh OOS cutoff was supplied but no arm produced evaluable fresh-OOS rows"
    if assume_oof_final and not fresh_evaluated:
        fresh_blocker = ""
    items.append(
        _audit_item(
            "fresh_chronological_oos_confirmation",
            ok=fresh_evaluated,
            blocker=fresh_blocker,
            waived=bool(assume_oof_final and not fresh_evaluated),
            evidence={
                "fresh_oos_requested": fresh_requested,
                "assume_oof_final": assume_oof_final,
                "fresh_oos_start": str(args.fresh_oos_start or ""),
                "fresh_oos_evaluated_rows": int(
                    summary.get("fresh_oos_evaluated", pd.Series(dtype=bool)).astype(bool).sum()
                ),
                "fresh_oos_status_values": sorted(
                    fresh_oos.get("status", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
                ),
            },
            metrics={
                "evaluated_rows": int(summary.get("fresh_oos_evaluated", pd.Series(dtype=bool)).astype(bool).sum()),
                "assume_oof_final": assume_oof_final,
                "status_values": ",".join(
                    sorted(fresh_oos.get("status", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
                ),
            },
        )
    )

    recommendation_series = summary.get("recommendation", pd.Series("", index=summary.index)).astype(str)
    promoted = summary.loc[recommendation_series.eq("candidate")]
    promoted_without_fresh = promoted.loc[
        promoted.get("fresh_oos_evaluated", pd.Series(False, index=promoted.index)).astype(bool).ne(True)
    ]
    items.append(
        _audit_item(
            "promotion_requires_fresh_oos_and_economic_gates",
            ok=promoted_without_fresh.empty,
            evidence={
                "candidate_rows": int(len(promoted)),
                "candidate_rows_without_fresh_oos": int(len(promoted_without_fresh)),
                "recommendation_counts": summary.groupby("recommendation").size().to_dict()
                if "recommendation" in summary
                else {},
            },
            metrics={
                "candidates": int(len(promoted)),
                "candidates_without_fresh": int(len(promoted_without_fresh)),
                "recommendations": dict(summary.groupby("recommendation").size())
                if "recommendation" in summary
                else {},
            },
        )
    )

    recommendation_values = set(summary.get("recommendation", pd.Series(dtype=str)).dropna().astype(str).unique())
    items.append(
        _audit_item(
            "broad_risk_sizing_remains_stopped",
            ok=not bool(recommendation_values & {"candidate", "deploy", "production"}),
            evidence={
                "recommendation_values": sorted(recommendation_values),
                "diagnostic_only": True,
                "production_models_modified": False,
            },
            metrics={
                "candidate_like_values": sorted(recommendation_values & {"candidate", "deploy", "production"}),
                "production_models_modified": False,
            },
        )
    )

    statuses = [str(item["status"]) for item in items]
    if "failed" in statuses:
        overall = "failed"
    elif "blocked" in statuses:
        overall = "blocked"
    elif "waived" in statuses:
        overall = "passed_with_waiver"
    else:
        overall = "passed"
    return {
        "status": overall,
        "blocked_requirements": [item["requirement"] for item in items if item["status"] == "blocked"],
        "failed_requirements": [item["requirement"] for item in items if item["status"] == "failed"],
        "waived_requirements": [item["requirement"] for item in items if item["status"] == "waived"],
        "outcomes": _retrain_outcomes(
            summary=summary,
            leave_one=leave_one,
            context_diagnostics=context_diagnostics,
            fresh_oos=fresh_oos,
        ),
        "items": items,
    }


def _format_audit_metric_value(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            parts.append(f"{key}={_format_audit_metric_value(val)}")
        return ", ".join(parts)
    if isinstance(value, list):
        return ",".join(_format_audit_metric_value(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (np.floating, float)):
        return "nan" if not np.isfinite(float(value)) else f"{float(value):.5g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _format_audit_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    return "; ".join(f"{key}={_format_audit_metric_value(value)}" for key, value in metrics.items())


def _append_records_table(lines: list[str], title: str, records: list[dict[str, Any]], *, floatfmt: str = ".5f") -> None:
    if not records:
        return
    lines.append(f"## {title}")
    lines.append("")
    lines.append(pd.DataFrame(records).to_markdown(index=False, floatfmt=floatfmt))
    lines.append("")


def _write_requirement_audit(out_dir: Path, audit: dict[str, Any]) -> None:
    (out_dir / "canonical_context_retrain_requirement_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    rows = [
        {
            "requirement": item["requirement"],
            "status": item["status"],
            "metrics": _format_audit_metrics(item.get("metrics", {})),
            "blocker": item.get("blocker", ""),
        }
        for item in audit.get("items", [])
    ]
    lines = ["# Canonical Context Retrain Requirement Audit", ""]
    lines.append(f"Overall status: `{audit.get('status', 'unknown')}`")
    lines.append("")
    outcomes = audit.get("outcomes", {})
    if outcomes:
        lines.append("## Outcome Summary")
        lines.append("")
        lines.append(f"- summary rows: `{outcomes.get('summary_rows', 0)}`")
        lines.append(f"- promoted candidates: `{outcomes.get('candidate_rows', 0)}`")
        lines.append(f"- research pending rows: `{outcomes.get('research_pending_rows', 0)}`")
        lines.append(
            f"- fresh-OOS status values: `{', '.join(outcomes.get('fresh_oos_status_values', [])) or 'none'}`"
        )
        lines.append("")
        _append_records_table(lines, "Recommendation Counts", outcomes.get("recommendation_counts", []))
        _append_records_table(lines, "Baseline Metrics", outcomes.get("baseline_metrics", []))
        _append_records_table(
            lines,
            "Best Non-Baseline Arm Per Head/Target",
            outcomes.get("best_non_baseline_by_head_target", []),
        )
        _append_records_table(lines, "Top Non-Baseline Arms By Log-Loss Improvement", outcomes.get("best_non_baseline_by_logloss", []))
        _append_records_table(lines, "Rejection Reason Counts", outcomes.get("rejection_reason_counts", []))
        _append_records_table(lines, "Context Fold Metrics", outcomes.get("context_fold_metrics", []))
        _append_records_table(lines, "Leave-One Episode Counts", outcomes.get("leave_one_episode_counts", []))
    lines.append("## Requirement Metrics")
    lines.append("")
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")
    if audit.get("blocked_requirements"):
        lines.append("## Blocked Requirements")
        lines.append("")
        for req in audit["blocked_requirements"]:
            lines.append(f"- `{req}`")
        lines.append("")
    if audit.get("waived_requirements"):
        lines.append("## Waived Requirements")
        lines.append("")
        lines.append("These are not proven by data; they are explicitly waived for this run.")
        lines.append("")
        for req in audit["waived_requirements"]:
            lines.append(f"- `{req}`")
        lines.append("")
    if audit.get("failed_requirements"):
        lines.append("## Failed Requirements")
        lines.append("")
        for req in audit["failed_requirements"]:
            lines.append(f"- `{req}`")
        lines.append("")
    (out_dir / "canonical_context_retrain_requirement_audit.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    out_dir = _ensure_dir(Path(args.output_dir))
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    report_dir = Path(args.report_dir)
    feature_dir = Path(args.feature_dir)
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    regime_context = Path(args.regime_context) if args.regime_context else None
    canonical_defs = _load_canonical_definitions(Path(args.canonical_reduction))
    if not canonical_defs:
        raise RuntimeError("No canonical definitions could be loaded")

    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    wanted_heads = set(CANDIDATES)
    if args.only_head:
        wanted_heads &= {str(x) for x in args.only_head}
    heads = [h for h in heads if h.head in wanted_heads]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)
    symbol_columns = _feature_store_union(feature_dir)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    context_diag_rows: list[dict[str, Any]] = []
    leave_one_rows: list[dict[str, Any]] = []
    fresh_oos_rows: list[dict[str, Any]] = []
    fresh_start = _parse_fresh_oos_start(args.fresh_oos_start)
    fresh_oos_requested = fresh_start is not None

    for head in heads:
        print(f"[canonical_context_retrain] processing head={head.head}", flush=True)
        panel = _normalise_keys(pd.read_parquet(head.meta_oof_path))
        panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
        race = meta_models[head.meta_key]
        candidate_x, raw = _assemble_head_context(
            head=head,
            panel=panel,
            race=race,
            base_bundle=base_bundle,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
            regime_context=regime_context,
            max_regime_columns=int(args.max_regime_columns),
        )
        contract = _candidate_feature_contract(candidate_x)
        bad_contract = contract.loc[
            contract["allowed_by_clean_contract"].astype(bool).ne(True)
            & contract["feature"].isin(set(sum((v["top_features"] for v in canonical_defs.values()), [])))
        ]
        high_mask = pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= float(args.rank_threshold)
        panel_high = panel.loc[high_mask].reset_index(drop=True)
        raw_high = raw.loc[high_mask].reset_index(drop=True)
        if len(panel_high) < 500:
            continue

        fresh_split: dict[str, Any] | None = None
        panel_dev = panel_high
        raw_dev = raw_high
        if fresh_start is not None:
            fresh_split = _fresh_oos_indices(
                panel_high["timestamp"],
                fresh_start,
                embargo_hours=int(args.embargo_hours),
            )
            # The fresh interval must not influence model selection or leave-one
            # development metrics.  Rows in the embargo gap are also excluded
            # from development and can only provide trailing context in live use.
            panel_dev = panel_high.iloc[np.asarray(fresh_split["train_idx"], dtype=np.int64)].reset_index(drop=True)
            raw_dev = raw_high.iloc[np.asarray(fresh_split["train_idx"], dtype=np.int64)].reset_index(drop=True)
        if len(panel_dev) < 500:
            continue

        targets = {t["name"]: t for t in _failure_targets(panel_dev) if t.get("kind") == "binary"}
        fresh_targets = {t["name"]: t for t in _failure_targets(panel_high) if t.get("kind") == "binary"}
        requested_targets = set(CANDIDATES[head.head])
        weekly = _weekly_high_conf_metrics(panel_dev, float(args.rank_threshold), int(args.min_week_rows))
        bad_weeks, bad_meta = _bad_recent_weeks(
            weekly,
            recent_weeks=int(args.recent_weeks),
            min_week_rows=int(args.min_week_rows),
        )
        bad_episode_set = {pd.Timestamp(w).strftime("%Y-%m-%d") for w in bad_weeks}
        folds = _make_chrono_folds(
            panel_dev["timestamp"],
            int(args.outer_folds),
            embargo_hours=int(args.embargo_hours),
        )
        canonical, ctx_diag = _fold_canonical_features(
            raw_dev,
            folds,
            canonical_defs,
            trailing_window=int(args.trailing_window),
            min_periods=int(args.min_periods),
            min_resolved_features=int(args.min_resolved_features),
        )
        canonical.to_parquet(out_dir / f"{head.head}_fold_fitted_canonical_context.parquet", index=False)
        for diag in ctx_diag:
            context_diag_rows.append(
                {
                    "head": head.head,
                    "fold": diag["fold"],
                    "train_rows": diag["train_rows"],
                    "valid_rows": diag["valid_rows"],
                    "train_start": diag["train_start"],
                    "train_end": diag["train_end"],
                    "valid_start": diag["valid_start"],
                    "valid_end": diag["valid_end"],
                    "train_output_feature_count": diag["train_diagnostics"].get("output_feature_count", 0),
                    "valid_output_feature_count": diag["valid_diagnostics"].get("output_feature_count", 0),
                }
            )
        arms = _arm_frames(panel_dev, canonical)
        fresh_arms: dict[str, pd.DataFrame | None] | None = None
        if fresh_split is not None:
            fresh_canonical, fresh_diag = _fresh_oos_canonical_features(
                raw_high,
                train_idx=np.asarray(fresh_split["train_idx"], dtype=np.int64),
                test_idx=np.asarray(fresh_split["test_idx"], dtype=np.int64),
                definitions=canonical_defs,
                trailing_window=int(args.trailing_window),
                min_periods=int(args.min_periods),
                min_resolved_features=int(args.min_resolved_features),
            )
            fresh_canonical.to_parquet(out_dir / f"{head.head}_fresh_oos_canonical_context.parquet", index=False)
            fresh_arms = _arm_frames(panel_high, fresh_canonical)
            context_diag_rows.append(
                {
                    "head": head.head,
                    "fold": "fresh_oos",
                    "train_rows": int(len(fresh_split["train_idx"])),
                    "valid_rows": int(len(fresh_split["test_idx"])),
                    "train_start": fresh_split.get("train_start", ""),
                    "train_end": fresh_split.get("train_end", ""),
                    "valid_start": fresh_split.get("test_start", ""),
                    "valid_end": fresh_split.get("test_end", ""),
                    "train_output_feature_count": fresh_diag.get("train_output_feature_count", 0),
                    "valid_output_feature_count": fresh_diag.get("test_output_feature_count", 0),
                }
            )
        returns = _pick_realized_return(panel_dev).to_numpy(dtype=np.float32, copy=False)
        full_returns = _pick_realized_return(panel_high).to_numpy(dtype=np.float32, copy=False)
        base_pred_raw = pd.to_numeric(panel_dev.get("oof_pred", pd.Series(np.nan, index=panel_dev.index)), errors="coerce")
        full_base_pred_raw = pd.to_numeric(
            panel_high.get("oof_pred", pd.Series(np.nan, index=panel_high.index)),
            errors="coerce",
        )
        baseline_failure_score = np.clip(1.0 - base_pred_raw.to_numpy(dtype=np.float32, copy=False), 1e-6, 1.0 - 1e-6)
        full_baseline_failure_score = np.clip(
            1.0 - full_base_pred_raw.to_numpy(dtype=np.float32, copy=False),
            1e-6,
            1.0 - 1e-6,
        )
        for target_name in sorted(requested_targets):
            if target_name not in targets:
                continue
            y = np.asarray(targets[target_name]["values"], dtype=np.float32)
            y_full = (
                np.asarray(fresh_targets[target_name]["values"], dtype=np.float32)
                if target_name in fresh_targets
                else np.full(len(panel_high), np.nan, dtype=np.float32)
            )
            valid_y = np.isfinite(y)
            y_bin = np.where(valid_y, y, 0.0).astype(np.int8)
            y_full_bin = np.where(np.isfinite(y_full), y_full, 0.0).astype(np.int8)
            for arm_name, x_arm in arms.items():
                if arm_name == "baseline_current_meta_unchanged":
                    pred = baseline_failure_score.copy()
                    arm_fold_rows = [
                        {
                            "head": head.head,
                            "target": target_name,
                            "arm": arm_name,
                            "fold": f.fold_id,
                            "reason": "unchanged_current_meta_reference",
                            "feature_count": 1,
                        }
                        for f in folds
                    ]
                else:
                    depth = 2 if arm_name == "model_state_x_market_state_interactions" else 3
                    pred, raw_fold_rows = _fit_predict_lgbm(
                        x_arm,
                        y_bin,
                        folds,
                        seed=int(args.seed) + abs(hash((head.head, target_name, arm_name))) % 10000,
                        max_depth=depth,
                    )
                    arm_fold_rows = [
                        {"head": head.head, "target": target_name, "arm": arm_name, **row}
                        for row in raw_fold_rows
                    ]
                fold_rows.extend(arm_fold_rows)
                row, eps = _score_arm(
                    head=head.head,
                    target=target_name,
                    arm=arm_name,
                    y=y_bin,
                    pred=pred,
                    baseline_pred=baseline_failure_score,
                    returns=returns,
                    timestamps=panel_high["timestamp"],
                    bad_episodes=bad_episode_set,
                )
                row.update(
                    {
                        "bad_week_count": int(len(bad_episode_set)),
                        "bad_week_reason": bad_meta.get("reason", ""),
                        "canonical_feature_count": int(canonical.shape[1]),
                        "bad_contract_feature_count": int(len(bad_contract)),
                        "fold_fitted": True,
                        "causal_trailing": True,
                        "live_equivalent": True,
                        "fresh_oos_requested": bool(fresh_oos_requested),
                        "embargo_hours": int(args.embargo_hours),
                        "raw_alias_outputs_used": False,
                        "long_bars_excluded": head.head != "long_bars",
                    }
                )
                loo = _leave_one_episode_rows(
                    head=head.head,
                    target=target_name,
                    arms={arm_name: x_arm},
                    y=y_bin,
                    baseline_pred=baseline_failure_score,
                    returns=returns,
                    timestamps=panel_high["timestamp"].reset_index(drop=True),
                    bad_episodes=bad_episode_set,
                    seed=int(args.seed) + abs(hash((head.head, target_name, "leave_one", arm_name))) % 10000,
                    max_train_rows=int(args.episode_transfer_max_train_rows),
                    embargo_hours=int(args.embargo_hours),
                )
                leave_one_rows.extend(loo)
                row.update(_summarize_leave_one(loo, head=head.head, target=target_name, arm=arm_name))
                if fresh_oos_requested and fresh_split is not None and fresh_arms is not None:
                    fresh_eval = _fresh_oos_eval_rows(
                        head=head.head,
                        target=target_name,
                        arms={arm_name: fresh_arms[arm_name]},
                        y=y_full_bin,
                        baseline_pred=full_baseline_failure_score,
                        returns=full_returns,
                        timestamps=panel_high["timestamp"].reset_index(drop=True),
                        train_idx=np.asarray(fresh_split["train_idx"], dtype=np.int64),
                        test_idx=np.asarray(fresh_split["test_idx"], dtype=np.int64),
                        fresh_start=fresh_start,
                        seed=int(args.seed) + abs(hash((head.head, target_name, "fresh_oos", arm_name))) % 10000,
                        max_train_rows=int(args.fresh_oos_max_train_rows),
                    )
                    fresh_oos_rows.extend(fresh_eval)
                    row.update(_summarize_fresh_oos(fresh_eval, head=head.head, target=target_name, arm=arm_name))
                else:
                    row.update(_summarize_fresh_oos([], head=head.head, target=target_name, arm=arm_name))
                decision, reason = _go_no_go(pd.Series(row), fresh_oos_evaluated=False)
                row["recommendation"] = decision
                row["decision_reason"] = reason
                summary_rows.append(row)
                episode_rows.extend(eps)

    summary_df = pd.DataFrame(summary_rows)
    episode_df = pd.DataFrame(episode_rows)
    fold_df = pd.DataFrame(fold_rows)
    context_diag_df = pd.DataFrame(context_diag_rows)
    leave_one_df = pd.DataFrame(leave_one_rows)
    fresh_oos_df = pd.DataFrame(
        fresh_oos_rows
        or [
            {
                "status": "not_evaluated",
                "fresh_oos_start": str(args.fresh_oos_start or ""),
                "reason": "no untouched later period was supplied; production promotion is blocked",
            }
        ]
    )
    summary_df.to_csv(out_dir / "canonical_context_retrain_summary.csv", index=False)
    episode_df.to_csv(out_dir / "canonical_context_retrain_episode_effects.csv", index=False)
    leave_one_df.to_csv(out_dir / "canonical_context_retrain_leave_one_episode.csv", index=False)
    fresh_oos_df.to_csv(out_dir / "canonical_context_retrain_fresh_oos_status.csv", index=False)
    fold_df.to_csv(out_dir / "canonical_context_retrain_fold_metrics.csv", index=False)
    context_diag_df.to_csv(out_dir / "canonical_context_retrain_context_diagnostics.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=_json_default))
    _write_report(out_dir, summary_df, episode_df)
    audit = _build_requirement_audit(
        summary=summary_df,
        fold_metrics=fold_df,
        leave_one=leave_one_df,
        fresh_oos=fresh_oos_df,
        context_diagnostics=context_diag_df,
        args=args,
    )
    _write_requirement_audit(out_dir, audit)
    print(f"[canonical_context_retrain] wrote results to {out_dir}", flush=True)
    return out_dir


def _write_report(out_dir: Path, summary: pd.DataFrame, episodes: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Canonical Context Retrain Experiment")
    lines.append("")
    lines.append("Diagnostic-only controlled retraining matrix. No production models were modified.")
    lines.append("")
    if summary.empty:
        lines.append("No summary rows were produced.")
        (out_dir / "canonical_context_retrain_report.md").write_text("\n".join(lines))
        return
    lines.append("## Arms")
    lines.append("")
    lines.append("- A: `baseline_current_meta_unchanged`")
    lines.append("- B: `canonical_model_state_context`")
    lines.append("- C: `canonical_market_state_context`")
    lines.append("- D: `model_state_x_market_state_interactions`")
    lines.append("- E: `auxiliary_failure_head`")
    lines.append("")
    lines.append("## Recommendation Summary")
    lines.append("")
    lines.append(summary.groupby(["arm", "recommendation"]).size().reset_index(name="count").to_markdown(index=False))
    lines.append("")
    show_cols = [
        "head",
        "target",
        "arm",
        "recommendation",
        "decision_reason",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "scored_coverage",
        "delta_log_loss_improvement",
        "delta_pr_auc",
        "top_reliable_hit_rate_10pct",
        "top_reliable_net_return_mean_10pct",
        "median_bad_episode_logloss_improvement",
        "episodes_improved_logloss",
        "bad_episode_count",
        "median_leave_one_logloss_improvement",
        "leave_one_episodes_improved_logloss",
        "leave_one_episode_count",
        "fresh_oos_status",
        "fresh_oos_delta_log_loss_improvement",
        "fresh_oos_delta_pr_auc",
        "fresh_oos_delta_tail_loss_10pct",
        "fresh_oos_delta_winner_rejection_cost_10pct",
        "delta_tail_loss_10pct",
        "delta_winner_rejection_cost_10pct",
        "weekly_auc_std",
        "weekly_rejection_turnover_10pct",
    ]
    lines.append("## Per-Arm Metrics")
    lines.append("")
    lines.append(summary[[c for c in show_cols if c in summary.columns]].to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    if not episodes.empty:
        bad = episodes.loc[episodes["is_bad_episode"].astype(bool)].copy()
        if not bad.empty:
            lines.append("## Bad-Episode Effects")
            lines.append("")
            agg = (
                bad.groupby(["head", "target", "arm"], as_index=False)
                .agg(
                    episodes=("episode", "nunique"),
                    median_logloss_improvement=("delta_log_loss_improvement", "median"),
                    worst_logloss_improvement=("delta_log_loss_improvement", "min"),
                    median_tail_delta=("delta_tail_loss_10pct", "median"),
                )
                .sort_values(["median_logloss_improvement", "median_tail_delta"], ascending=False)
            )
            lines.append(agg.to_markdown(index=False, floatfmt=".5f"))
            lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- `long_bars` is intentionally excluded from this first wave.")
    lines.append("- Canonical context is regenerated inside chronological folds using trailing, shifted robust baselines.")
    lines.append("- Raw archetype aliases, post-hoc probabilities, direct bad-week labels, adversarial scores, and leaf outcome statistics are not used as arm inputs.")
    lines.append("- A candidate arm still requires a fresh chronological OOS period before production promotion.")
    lines.append("- If `--fresh-oos-start` is supplied, rows at or after that timestamp are excluded from nested and leave-one development scoring.")
    lines.append("- If no `--fresh-oos-start` is supplied, passing arms are labeled `research_candidate_pending_fresh_oos`, not production candidates.")
    lines.append("- If `--assume-oof-final` is supplied, fresh OOS is treated as a user-waived requirement, not as proven fresh-OOS evidence.")
    (out_dir / "canonical_context_retrain_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument(
        "--transform-cache",
        default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet",
    )
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/canonical_context_retrain_experiment_20260622")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--recent-weeks", type=int, default=8)
    parser.add_argument("--min-week-rows", type=int, default=30)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-regime-columns", type=int, default=80)
    parser.add_argument("--episode-transfer-max-train-rows", type=int, default=60000)
    parser.add_argument("--fresh-oos-max-train-rows", type=int, default=100000)
    parser.add_argument("--fresh-oos-start", default="")
    parser.add_argument(
        "--assume-oof-final",
        action="store_true",
        help="Treat nested chronological OOF plus leave-one evaluation as final evidence when fresh OOS rows are unavailable.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--only-head", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
    raise SystemExit(0)
