#!/usr/bin/env python3
"""Strict OOF execution-EV decomposition and side-local calibration ablation.

The primary comparison is deliberately narrow and uses the same frozen handoff,
cost-adjusted 12-hour net-EV target, purged outer folds, and side-local feature
matrix for every arm:

* direct E[net EV];
* P(executable positive net EV) x E[net EV | positive];
* P(clean favorable-first) x E[max(net EV, 0) | clean] minus severe-loss
  contribution; and
* the algebraically complete sign partition with all negative outcomes.

``clean`` is the exact-policy ``tb_hard_label`` (favorable-first under the
ATR-normalized triple barrier), never a net-EV-sign proxy. ``severe`` is an
explicit cost-aware tail (at least 1% or twice entry cost), not a proxy for
every losing trade. The all-loss version is retained because it is the
algebraically complete expected-value comparator. All resulting scores are
return units net of the canonical cost contract.

The post-map is a side-specific isotonic map fitted only on earlier *outer-OOF*
predictions.  It is followed by the existing daily causal recent-EV correction,
which uses only outcomes resolved before each UTC-day snapshot.  Ranking is one
pooled global top-k after those maps; side-local ranks are diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import execution_ev_metrics  # noqa: E402
from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ExecutionEVModelAblationConfig,
    _materialize_feature_matrix,
    apply_execution_ev_causal_recent_ev_correction,
    chronological_purged_splits,
    fit_train_only_isotonic_ev_mapping,
    validate_execution_ev_model_ablation_contract,
)
from scripts.run_execution_ev_model_ablation import _load_provenance  # noqa: E402


SCHEMA = "execution_ev_decomposition_calibration_ablation_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DIRECT = "direct_net_ev"
FULL = "positive_minus_all_loss_net_ev"
CLEAN_SEVERE = "clean_favorable_minus_severe_loss_net_ev"
MULTITASK_BLEND = "direct_primary_auxiliary_oof_blend_net_ev"
MULTITASK_SHARED = "direct_primary_shared_multitask_oof_net_ev"
BASE_SCORE_ARMS = (DIRECT, FULL, CLEAN_SEVERE)
MULTITASK_FEATURES = (
    DIRECT,
    "p_executable_positive_net_ev",
    "favorable_magnitude_if_positive_net_ev",
    "p_clean_favorable_first",
    "favorable_net_magnitude_if_clean",
    "p_any_net_loss",
    "conditional_loss_if_any_net_loss",
    "p_severe_net_loss",
    "conditional_loss_if_severe_net_loss",
    FULL,
    CLEAN_SEVERE,
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _classifier(*, seed: int, iterations: int, threads: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(iterations), depth=6, learning_rate=0.04,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=6.0,
        random_seed=int(seed), thread_count=int(threads), verbose=False,
        allow_writing_files=False,
    )


def _regressor(*, seed: int, iterations: int, threads: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=int(iterations), depth=6, learning_rate=0.04,
        loss_function="RMSE", l2_leaf_reg=8.0, random_seed=int(seed),
        thread_count=int(threads), verbose=False, allow_writing_files=False,
    )


def _probability(
    x: pd.DataFrame, label: np.ndarray, train: np.ndarray, valid: np.ndarray, *,
    seed: int, iterations: int, threads: int,
) -> np.ndarray:
    train_label = np.asarray(label[train], dtype=np.int8)
    # Early expanding folds can contain only one class.  A train-only prior is
    # the only causal fallback; it is not learned from validation prevalence.
    if np.unique(train_label).size < 2:
        return np.full(len(valid), float(train_label.mean()), dtype=np.float64)
    model = _classifier(seed=seed, iterations=iterations, threads=threads)
    model.fit(x.iloc[train], train_label)
    return model.predict_proba(x.iloc[valid])[:, 1]


def _conditional_magnitude(
    x: pd.DataFrame, magnitude: np.ndarray, condition: np.ndarray,
    train: np.ndarray, valid: np.ndarray, *, seed: int, iterations: int, threads: int,
) -> np.ndarray:
    fit = train[np.asarray(condition[train], dtype=bool)]
    if len(fit) < 32:
        # This is still an authorized conditional sample; do not substitute a
        # validation statistic when a rare event lacks support.
        prior = float(np.mean(magnitude[fit])) if len(fit) else 0.0
        return np.full(len(valid), prior, dtype=np.float64)
    values = np.maximum(np.asarray(magnitude[fit], dtype=np.float64), 0.0)
    cap = float(np.quantile(values, 0.995))
    scale = max(float(np.median(values[values > 0.0])) if np.any(values > 0.0) else 0.0, 1e-4)
    target = np.log1p(np.minimum(values, cap) / scale)
    model = _regressor(seed=seed, iterations=iterations, threads=threads)
    model.fit(x.iloc[fit], target)
    return np.maximum(np.expm1(model.predict(x.iloc[valid])) * scale, 0.0)


def _fit_side_outer_fold(
    x: pd.DataFrame, net_ev: np.ndarray, clean_event: np.ndarray, severe_floor: np.ndarray,
    train: np.ndarray, valid: np.ndarray, *, seed: int, iterations: int, threads: int,
) -> dict[str, np.ndarray]:
    positive = net_ev > 0.0
    loss = ~positive
    severe = net_ev <= -severe_floor
    direct = _regressor(seed=seed, iterations=iterations, threads=threads)
    direct.fit(x.iloc[train], net_ev[train])
    p_positive = _probability(
        x, positive, train, valid, seed=seed + 1, iterations=iterations, threads=threads
    )
    positive_magnitude = _conditional_magnitude(
        x, np.maximum(net_ev, 0.0), positive, train, valid,
        seed=seed + 2, iterations=iterations, threads=threads,
    )
    p_clean = _probability(
        x, clean_event, train, valid, seed=seed + 7, iterations=iterations, threads=threads
    )
    clean_favorable_magnitude = _conditional_magnitude(
        x, np.maximum(net_ev, 0.0), clean_event, train, valid,
        seed=seed + 8, iterations=iterations, threads=threads,
    )
    p_loss = _probability(
        x, loss, train, valid, seed=seed + 3, iterations=iterations, threads=threads
    )
    loss_magnitude = _conditional_magnitude(
        x, np.maximum(-net_ev, 0.0), loss, train, valid,
        seed=seed + 4, iterations=iterations, threads=threads,
    )
    p_severe = _probability(
        x, severe, train, valid, seed=seed + 5, iterations=iterations, threads=threads
    )
    severe_magnitude = _conditional_magnitude(
        x, np.maximum(-net_ev, 0.0), severe, train, valid,
        seed=seed + 6, iterations=iterations, threads=threads,
    )
    return {
        DIRECT: direct.predict(x.iloc[valid]),
        "p_executable_positive_net_ev": p_positive,
        "favorable_magnitude_if_positive_net_ev": positive_magnitude,
        "p_clean_favorable_first": p_clean,
        "favorable_net_magnitude_if_clean": clean_favorable_magnitude,
        "p_any_net_loss": p_loss,
        "conditional_loss_if_any_net_loss": loss_magnitude,
        "p_severe_net_loss": p_severe,
        "conditional_loss_if_severe_net_loss": severe_magnitude,
        FULL: p_positive * positive_magnitude - p_loss * loss_magnitude,
        CLEAN_SEVERE: p_clean * clean_favorable_magnitude - p_severe * severe_magnitude,
    }


def temporal_side_oof_isotonic(
    frame: pd.DataFrame, raw: np.ndarray, net_ev: np.ndarray, fold_id: np.ndarray, *,
    decision_col: str, resolution_col: str, side_col: str, min_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Causal per-side map: every fold sees earlier resolved outer-OOF rows only."""
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[resolution_col], utc=True, errors="raise")
    side = frame[side_col].astype(str).str.lower().to_numpy()
    result = np.asarray(raw, dtype=np.float64).copy()
    audit: list[dict[str, Any]] = []
    valid_folds = sorted(int(item) for item in np.unique(fold_id[np.isfinite(fold_id)]))
    for fold in valid_folds:
        current = fold_id == fold
        cutoff = decision[current].min()
        for side_name in ("long", "short"):
            target = current & (side == side_name) & np.isfinite(raw)
            reference = (
                (fold_id < fold) & (side == side_name) & np.isfinite(raw)
                & np.isfinite(net_ev) & resolved.lt(cutoff).to_numpy()
            )
            mapper = fit_train_only_isotonic_ev_mapping(
                raw[reference], net_ev[reference], min_rows=int(min_rows)
            )
            if target.any():
                result[target] = mapper.predict(raw[target])
            audit.append({
                "fold": int(fold), "side": side_name,
                "validation_rows": int(target.sum()), "reference_oof_rows": int(reference.sum()),
                "reference_max_resolution_utc": (
                    resolved[reference].max().isoformat() if reference.any() else None
                ),
                "validation_start_utc": cutoff.isoformat(), "status": mapper.status,
            })
    return result, audit


def temporal_multitask_oof_blend(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    net_ev: np.ndarray,
    fold_id: np.ndarray,
    *,
    decision_col: str,
    resolution_col: str,
    min_rows: int,
    ridge_alpha: float = 10.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Learn a pooled common-unit auxiliary blend from prior OOF rows only.

    The direct EV head is the mandatory fallback and first component.  This
    auxiliary combiner never sees a same-fold model prediction or outcome.
    """

    missing = [name for name in MULTITASK_FEATURES if name not in predictions]
    if missing:
        raise ValueError("multi-task blend is missing heads: " + ", ".join(missing))
    matrix = np.column_stack(
        [np.asarray(predictions[name], dtype=np.float64) for name in MULTITASK_FEATURES]
    )
    direct = np.asarray(predictions[DIRECT], dtype=np.float64)
    result = direct.copy()
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[resolution_col], utc=True, errors="raise")
    audit: list[dict[str, Any]] = []
    valid_folds = sorted(int(item) for item in np.unique(fold_id[np.isfinite(fold_id)]))
    finite_matrix = np.isfinite(matrix).all(axis=1)
    for fold in valid_folds:
        current = (fold_id == fold) & finite_matrix
        cutoff = decision[current].min()
        reference = (
            (fold_id < fold)
            & finite_matrix
            & np.isfinite(net_ev)
            & resolved.lt(cutoff).to_numpy()
        )
        status = "direct_primary_fallback_insufficient_prior_oof"
        if int(reference.sum()) >= int(min_rows):
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(ridge_alpha), fit_intercept=True),
            )
            model.fit(matrix[reference], net_ev[reference])
            result[current] = model.predict(matrix[current])
            status = "pooled_ridge_on_prior_outer_oof_heads"
        audit.append(
            {
                "fold": int(fold),
                "validation_rows": int(current.sum()),
                "reference_oof_rows": int(reference.sum()),
                "reference_max_resolution_utc": (
                    resolved[reference].max().isoformat() if reference.any() else None
                ),
                "validation_start_utc": cutoff.isoformat(),
                "status": status,
                "ridge_alpha": float(ridge_alpha),
                "features": list(MULTITASK_FEATURES),
            }
        )
    return result, audit


def temporal_shared_multitask_oof_meta(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    net_ev: np.ndarray,
    clean_event: np.ndarray,
    severe_floor: np.ndarray,
    fold_id: np.ndarray,
    *,
    decision_col: str,
    resolution_col: str,
    side_col: str,
    min_rows: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit a side-local shared-trunk multi-task meta learner on prior OOF rows.

    Four repeated standardized direct-EV outputs make direct EV the primary
    loss. Auxiliary soft event and magnitude outputs regularize the shared
    representation. Only the averaged direct outputs become the score.
    """

    missing = [name for name in MULTITASK_FEATURES if name not in predictions]
    if missing:
        raise ValueError(
            "shared multi-task meta is missing heads: " + ", ".join(missing)
        )
    matrix = np.column_stack(
        [
            np.asarray(predictions[name], dtype=np.float64)
            for name in MULTITASK_FEATURES
        ]
    )
    direct = np.asarray(predictions[DIRECT], dtype=np.float64)
    result = direct.copy()
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[resolution_col], utc=True, errors="raise")
    side = frame[side_col].astype(str).str.lower().to_numpy()
    finite = (
        np.isfinite(matrix).all(axis=1)
        & np.isfinite(net_ev)
        & np.isfinite(severe_floor)
    )
    audit: list[dict[str, Any]] = []
    valid_folds = sorted(
        int(item) for item in np.unique(fold_id[np.isfinite(fold_id)])
    )
    for fold in valid_folds:
        fold_rows = (fold_id == fold) & finite
        validation_start = decision[fold_rows].min()
        for side_index, side_name in enumerate(("long", "short")):
            current = fold_rows & (side == side_name)
            reference = (
                (fold_id < fold)
                & finite
                & (side == side_name)
                & resolved.lt(validation_start).to_numpy()
            )
            status = "direct_primary_fallback_insufficient_prior_oof"
            if int(reference.sum()) >= int(min_rows) and current.any():
                direct_mean = float(np.mean(net_ev[reference]))
                direct_scale = max(float(np.std(net_ev[reference])), 1e-4)
                direct_z = np.clip(
                    (net_ev[reference] - direct_mean) / direct_scale,
                    -8.0,
                    8.0,
                )
                positive_soft = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(net_ev[reference] / 0.005, -40.0, 40.0)
                    )
                )
                severe_soft = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            (-net_ev[reference] - severe_floor[reference])
                            / 0.005,
                            -40.0,
                            40.0,
                        )
                    )
                )
                positive_magnitude = np.clip(
                    np.maximum(net_ev[reference], 0.0) / 0.05, 0.0, 4.0
                )
                loss_magnitude = np.clip(
                    np.maximum(-net_ev[reference], 0.0) / 0.05, 0.0, 4.0
                )
                targets = np.column_stack(
                    [
                        direct_z,
                        direct_z,
                        direct_z,
                        direct_z,
                        positive_soft,
                        clean_event[reference].astype(np.float64),
                        severe_soft,
                        positive_magnitude,
                        loss_magnitude,
                    ]
                )
                model = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(24, 12),
                        activation="tanh",
                        solver="adam",
                        alpha=0.01,
                        batch_size=512,
                        learning_rate_init=0.002,
                        max_iter=60,
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=6,
                        random_state=int(
                            random_state + 100 * fold + side_index
                        ),
                    ),
                )
                model.fit(matrix[reference], targets)
                predicted = np.asarray(
                    model.predict(matrix[current]), dtype=np.float64
                )
                predicted_direct_z = np.mean(predicted[:, :4], axis=1)
                result[current] = (
                    direct_mean + direct_scale * predicted_direct_z
                )
                status = "shared_trunk_prior_outer_oof_direct_primary"
            audit.append(
                {
                    "fold": int(fold),
                    "side": side_name,
                    "validation_rows": int(current.sum()),
                    "reference_oof_rows": int(reference.sum()),
                    "reference_max_resolution_utc": (
                        resolved[reference].max().isoformat()
                        if reference.any()
                        else None
                    ),
                    "validation_start_utc": validation_start.isoformat(),
                    "status": status,
                    "input_heads": list(MULTITASK_FEATURES),
                    "loss_outputs": {
                        "direct_ev_repetitions": 4,
                        "auxiliary_outputs": [
                            "soft_positive",
                            "clean_event",
                            "soft_severe_loss",
                            "positive_magnitude",
                            "loss_magnitude",
                        ],
                    },
                }
            )
    return result, audit


def temporal_hierarchical_oof_calibration(
    frame: pd.DataFrame,
    raw: np.ndarray,
    net_ev: np.ndarray,
    fold_id: np.ndarray,
    *,
    decision_col: str,
    resolution_col: str,
    side_col: str,
    min_rows: int,
    side_fit_fraction: float = 0.65,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Nested side monotonic maps followed by a pooled common-EV anchor.

    Earlier OOF reference rows are split chronologically.  The early resolved
    segment fits the two side maps; a later, disjoint OOF segment fits the
    pooled anchor.  Current-fold outcomes are never used by either layer.
    """

    if not 0.25 <= float(side_fit_fraction) <= 0.80:
        raise ValueError("side_fit_fraction must be in [0.25, 0.80]")
    raw = np.asarray(raw, dtype=np.float64)
    result = raw.copy()
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[resolution_col], utc=True, errors="raise")
    side = frame[side_col].astype(str).str.lower().to_numpy()
    audit: list[dict[str, Any]] = []
    valid_folds = sorted(int(item) for item in np.unique(fold_id[np.isfinite(fold_id)]))
    for fold in valid_folds:
        current = (fold_id == fold) & np.isfinite(raw)
        validation_start = decision[current].min()
        reference = (
            (fold_id < fold)
            & np.isfinite(raw)
            & np.isfinite(net_ev)
            & resolved.lt(validation_start).to_numpy()
        )
        reference_times = np.sort(decision[reference].unique())
        if len(reference_times) < 2:
            audit.append(
                {
                    "fold": int(fold),
                    "status": "identity_insufficient_prior_oof",
                    "validation_rows": int(current.sum()),
                    "reference_oof_rows": int(reference.sum()),
                    "validation_start_utc": validation_start.isoformat(),
                }
            )
            continue
        split_position = min(
            len(reference_times) - 1,
            max(1, int(np.floor(float(side_fit_fraction) * len(reference_times)))),
        )
        anchor_start = pd.Timestamp(reference_times[split_position])
        side_fit = (
            reference
            & decision.lt(anchor_start).to_numpy()
            & resolved.lt(anchor_start).to_numpy()
        )
        anchor = (
            reference
            & decision.ge(anchor_start).to_numpy()
            & resolved.lt(validation_start).to_numpy()
        )
        current_side_mapped = raw[current].copy()
        anchor_side_mapped = raw[anchor].copy()
        current_positions = np.flatnonzero(current)
        anchor_positions = np.flatnonzero(anchor)
        side_rows: dict[str, int] = {}
        side_status: dict[str, str] = {}
        for side_name in ("long", "short"):
            fit_mask = side_fit & (side == side_name)
            mapper = fit_train_only_isotonic_ev_mapping(
                raw[fit_mask], net_ev[fit_mask], min_rows=int(min_rows)
            )
            current_local = side[current_positions] == side_name
            anchor_local = side[anchor_positions] == side_name
            if current_local.any():
                current_side_mapped[current_local] = mapper.predict(
                    raw[current_positions[current_local]]
                )
            if anchor_local.any():
                anchor_side_mapped[anchor_local] = mapper.predict(
                    raw[anchor_positions[anchor_local]]
                )
            side_rows[side_name] = int(fit_mask.sum())
            side_status[side_name] = mapper.status
        pooled = fit_train_only_isotonic_ev_mapping(
            anchor_side_mapped, net_ev[anchor], min_rows=int(2 * min_rows)
        )
        result[current] = pooled.predict(current_side_mapped)
        audit.append(
            {
                "fold": int(fold),
                "status": pooled.status,
                "validation_rows": int(current.sum()),
                "reference_oof_rows": int(reference.sum()),
                "side_fit_rows": side_rows,
                "side_fit_status": side_status,
                "side_fit_max_resolution_utc": (
                    resolved[side_fit].max().isoformat() if side_fit.any() else None
                ),
                "anchor_start_utc": anchor_start.isoformat(),
                "pooled_anchor_rows": int(anchor.sum()),
                "pooled_anchor_max_resolution_utc": (
                    resolved[anchor].max().isoformat() if anchor.any() else None
                ),
                "validation_start_utc": validation_start.isoformat(),
            }
        )
    return result, audit


def _probability_metrics(score: np.ndarray, label: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask & np.isfinite(score)
    y, p = label[valid].astype(np.int8), score[valid]
    if not len(y) or np.unique(y).size < 2:
        return {"rows": int(len(y)), "prevalence": float(y.mean()) if len(y) else float("nan"), "auc": float("nan"), "average_precision": float("nan"), "brier": float("nan")}
    return {
        "rows": int(len(y)), "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def pooled_global_metrics(
    score: np.ndarray, net_ev: np.ndarray, gross_ev: np.ndarray, side: np.ndarray,
    mask: np.ndarray, *, top_fraction: float,
) -> dict[str, Any]:
    valid = mask & np.isfinite(score) & np.isfinite(net_ev)
    base = execution_ev_metrics(net_ev[valid], score[valid], top_k_fraction=top_fraction)
    positions = np.flatnonzero(valid)
    top_count = max(1, int(np.ceil(len(positions) * float(top_fraction)))) if len(positions) else 0
    ranked = positions[np.argsort(-score[positions], kind="stable")[:top_count]] if top_count else np.array([], dtype=int)
    return {
        **base,
        "ranking_scope": "one_pooled_global_top_k_after_side_calibration",
        "top_k_mean_gross_ev": float(gross_ev[ranked].mean()) if len(ranked) and np.isfinite(gross_ev[ranked]).all() else float("nan"),
        "top_k_gross_minus_net_cost": float((gross_ev[ranked] - net_ev[ranked]).mean()) if len(ranked) and np.isfinite(gross_ev[ranked]).all() else float("nan"),
        "top_k_long_rows": int((side[ranked] == "long").sum()),
        "top_k_short_rows": int((side[ranked] == "short").sum()),
    }


def metric_slices(
    frame: pd.DataFrame, predictions: Mapping[str, np.ndarray], net_ev: np.ndarray,
    gross_ev: np.ndarray, shared: np.ndarray, *, decision_col: str, side_col: str,
    top_fraction: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    side = frame[side_col].astype(str).str.lower().to_numpy()
    slices: dict[str, np.ndarray] = {"all_oof": shared}
    for side_name in ("long", "short"):
        slices[f"side_{side_name}"] = shared & (side == side_name)
    for month in sorted(decision[shared].dt.strftime("%Y-%m").unique()):
        slices[f"month_{month}"] = shared & decision.dt.strftime("%Y-%m").eq(month).to_numpy()
    iso = decision[shared].dt.isocalendar()
    weeks = (iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)).unique()
    for week in sorted(map(str, weeks)):
        token = decision.dt.isocalendar().year.astype(str) + "-W" + decision.dt.isocalendar().week.astype(str).str.zfill(2)
        slices[f"week_{week}"] = shared & token.eq(week).to_numpy()
    if weeks.size:
        latest = max(map(str, weeks))
        slices["latest_week"] = slices[f"week_{latest}"]
    return {
        name: {
            arm: pooled_global_metrics(values, net_ev, gross_ev, side, mask, top_fraction=top_fraction)
            for arm, values in predictions.items()
        }
        for name, mask in slices.items() if mask.any()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--clean-labels", type=Path, required=True)
    parser.add_argument(
        "--compact-raw-market-context", action="store_true",
        help="Use the separately disclosed May-Jul19 context + h0 raw-market contract.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-input-families", default="")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-train-rows", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--severe-loss-floor", type=float, default=0.01)
    parser.add_argument("--severe-cost-multiple", type=float, default=2.0)
    parser.add_argument("--isotonic-min-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def _join_clean_labels(frame: pd.DataFrame, labels_path: Path) -> np.ndarray:
    """Attach the exact-policy clean event with an exact one-to-one identity join."""
    labels = pd.read_parquet(labels_path)
    needed = [*IDENTITY, "tb_hard_label", "risk_class", "meaningful_mfe_reached"]
    missing = [column for column in needed if column not in labels]
    if missing:
        raise ValueError("clean-label artifact missing columns: " + ", ".join(missing))
    if frame.duplicated(list(IDENTITY)).any() or labels.duplicated(list(IDENTITY)).any():
        raise ValueError("clean-label join requires unique exact row identities")
    if frame.loc[:, list(IDENTITY)].isna().any().any() or labels.loc[:, list(IDENTITY)].isna().any().any():
        raise ValueError("clean-label join identity cannot contain nulls")
    lookup = labels.loc[:, needed].copy()
    merged = frame.loc[:, list(IDENTITY)].merge(
        lookup, on=list(IDENTITY), how="left", validate="one_to_one", sort=False
    )
    if len(merged) != len(frame) or merged["tb_hard_label"].isna().any():
        raise ValueError("clean-label artifact does not exactly cover the execution-EV handoff")
    clean = pd.to_numeric(merged["tb_hard_label"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(clean).all() or not np.isin(clean, [0.0, 1.0]).all():
        raise ValueError("tb_hard_label must be a finite binary clean-event target")
    risk = pd.to_numeric(merged["risk_class"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(risk).all() or not np.isin(risk, [0.0, 1.0, 2.0]).all():
        raise ValueError("risk_class must be the exact-policy {timeout, adverse, favorable} code")
    if not np.array_equal(clean.astype(np.int8), (risk == 2.0).astype(np.int8)):
        raise ValueError("tb_hard_label must match risk_class=favorable-first")
    return clean.astype(bool)


def _compact_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return a predeclared compact context set and h0-only state candidates."""
    core_prefixes = ("catboost_p_", "base_archetype_label__")
    core_names = {
        "existing_alpha_ev", "pred_peak_MFE_12h_ATR", "catboost_entropy",
        "alpha_prediction_uncertainty", "alpha_leaf_support", "base_oof_score",
        "base_margin_to_cutoff", "base_margin_to_cutoff_z", "oof_clean_favorable_probability",
    }
    core = [
        column for column in frame.columns
        if column in core_names or column.startswith(core_prefixes)
    ]
    state = [
        column for column in frame.columns
        if column.startswith("mkt_state__") and column.endswith("__h0")
    ]
    if not core or not state:
        raise ValueError("compact raw-market context is missing predeclared core or h0 state fields")
    values = frame.loc[:, core].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("compact core context contains non-finite values")
    if "raw_state_source_utc_h0" not in frame:
        raise ValueError("compact raw-market context must declare raw_state_source_utc_h0")
    available = pd.to_datetime(frame["raw_state_source_utc_h0"], utc=True, errors="coerce")
    decision = pd.to_datetime(frame["execution_decision_utc"], utc=True, errors="coerce")
    if available.isna().any() or decision.isna().any() or (available > decision).any():
        raise ValueError("raw h0 market state is not point-in-time available at decision")
    return core, state


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frame = pd.read_parquet(args.input)
    if args.compact_raw_market_context:
        provenance: dict[str, Any] = {}
        provenance_payload: dict[str, Any] = {"schema": "compact_raw_market_context_v1"}
    else:
        if args.provenance is None:
            raise ValueError("--provenance is required unless --compact-raw-market-context is set")
        provenance, provenance_payload = _load_provenance(args.provenance)
    families = tuple(item.strip() for item in str(args.additional_input_families).split(",") if item.strip())
    config = ExecutionEVModelAblationConfig(
        n_splits=int(args.n_splits), min_train_rows=int(args.min_train_rows),
        decision_time_col="execution_decision_utc", label_end_time_col="execution_label_end_utc",
        additional_input_families=families, recent_ev_correction_routes=("catboost_predicted_archetype",),
        top_k_fraction=float(args.top_fraction),
    )
    for column in (config.decision_time_col, config.label_end_time_col):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    compact_state_columns: list[str] = []
    if args.compact_raw_market_context:
        raw_columns, compact_state_columns = _compact_feature_columns(frame)
        x = frame.loc[:, raw_columns + compact_state_columns].apply(pd.to_numeric, errors="coerce")
    else:
        raw_columns, archetype_levels = validate_execution_ev_model_ablation_contract(
            frame, provenance, decision_time_col=config.decision_time_col, side_col=config.side_col,
            catboost_archetype_col=config.catboost_archetype_col,
            additional_input_families=config.additional_input_families,
        )
        x = _materialize_feature_matrix(frame, raw_columns, catboost_archetype_col=config.catboost_archetype_col, archetype_levels=archetype_levels)
    net_ev = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(np.float64)
    gross_ev = (
        pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise").to_numpy(np.float64)
        if "execution_gross_ev_12h" in frame
        else np.full(len(frame), np.nan, dtype=np.float64)
    )
    clean_event = _join_clean_labels(frame, args.clean_labels)
    cost = (
        np.abs(pd.to_numeric(frame["execution_cost_return"], errors="raise").to_numpy(np.float64))
        if "execution_cost_return" in frame
        else np.zeros(len(frame), dtype=np.float64)
    )
    severe_floor = np.maximum(float(args.severe_loss_floor), float(args.severe_cost_multiple) * cost)
    side = frame[config.side_col].astype(str).str.lower().to_numpy()
    folds = chronological_purged_splits(
        frame, n_splits=int(args.n_splits), min_train_size=int(args.min_train_rows),
        decision_time_col=config.decision_time_col, label_end_time_col=config.label_end_time_col,
        horizon_hours=12.0, embargo_hours=12.0,
    )
    head_names = (DIRECT, "p_executable_positive_net_ev", "favorable_magnitude_if_positive_net_ev", "p_clean_favorable_first", "favorable_net_magnitude_if_clean", "p_any_net_loss", "conditional_loss_if_any_net_loss", "p_severe_net_loss", "conditional_loss_if_severe_net_loss", FULL, CLEAN_SEVERE)
    predictions = {name: np.full(len(frame), np.nan, dtype=np.float64) for name in head_names}
    fold_id = np.full(len(frame), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []
    for split in folds:
        active_columns = list(x.columns)
        if args.compact_raw_market_context:
            # Each fold admits a raw h0 feature only when its own authorized
            # training rows have >=95% coverage.  Both sides and every arm in
            # that fold see this identical compact set.
            train_coverage = x.iloc[split.train_indices][compact_state_columns].notna().mean()
            admitted_state = [column for column in compact_state_columns if float(train_coverage[column]) >= 0.95]
            active_columns = raw_columns + admitted_state
            if not admitted_state:
                raise ValueError("compact raw-market fold has no >=95%-covered h0 state inputs")
        for side_offset, side_name in enumerate(("long", "short")):
            train = split.train_indices[side[split.train_indices] == side_name]
            valid = split.validation_indices[side[split.validation_indices] == side_name]
            if not len(train) or not len(valid):
                continue
            result = _fit_side_outer_fold(
                x.loc[:, active_columns], net_ev, clean_event, severe_floor, train, valid,
                seed=int(args.seed) + int(split.fold) * 100 + side_offset * 25,
                iterations=int(args.iterations), threads=int(args.threads),
            )
            for name, values in result.items():
                predictions[name][valid] = values
            fold_id[valid] = split.fold
            fold_audit.append({
                "fold": int(split.fold), "side": side_name, "train_rows": int(len(train)),
                "validation_rows": int(len(valid)), "train_end_utc": pd.to_datetime(frame.iloc[train][config.label_end_time_col], utc=True).max().isoformat(),
                "validation_start_utc": split.validation_start.isoformat(),
                "purge_hours": float(split.purge_hours), "embargo_hours": float(split.embargo_hours),
                "compact_state_coverage_rule": ">=0.95 on this fold's training rows" if args.compact_raw_market_context else None,
                "compact_admitted_h0_state_columns": active_columns[len(raw_columns):] if args.compact_raw_market_context else None,
            })
            print(f"[ev-decomposition] fold={split.fold} side={side_name} train={len(train)} valid={len(valid)}", flush=True)
    shared = np.isfinite(fold_id)
    for name in MULTITASK_FEATURES:
        values = predictions[name]
        shared &= np.isfinite(values)
    if not shared.any():
        raise ValueError("decomposition ablation produced no shared outer-OOF rows")

    multitask_prediction, multitask_audit = temporal_multitask_oof_blend(
        frame,
        predictions,
        net_ev,
        fold_id,
        decision_col=config.decision_time_col,
        resolution_col=config.label_end_time_col,
        min_rows=max(500, int(args.isotonic_min_rows)),
    )
    predictions[MULTITASK_BLEND] = multitask_prediction
    shared_multitask_prediction, shared_multitask_audit = (
        temporal_shared_multitask_oof_meta(
            frame,
            predictions,
            net_ev,
            clean_event,
            severe_floor,
            fold_id,
            decision_col=config.decision_time_col,
            resolution_col=config.label_end_time_col,
            side_col=config.side_col,
            min_rows=max(500, int(args.isotonic_min_rows)),
            random_state=int(args.seed) + 70_000,
        )
    )
    predictions[MULTITASK_SHARED] = shared_multitask_prediction
    score_arms = (
        *BASE_SCORE_ARMS,
        MULTITASK_BLEND,
        MULTITASK_SHARED,
    )

    calibrator_audit: dict[str, list[dict[str, Any]]] = {}
    isotonic_predictions: dict[str, np.ndarray] = {}
    hierarchical_predictions: dict[str, np.ndarray] = {}
    hierarchical_audit: dict[str, list[dict[str, Any]]] = {}
    final_predictions: dict[str, np.ndarray] = {}
    correction_reports: dict[str, Any] = {}
    for arm in score_arms:
        mapped, audit = temporal_side_oof_isotonic(
            frame, predictions[arm], net_ev, fold_id,
            decision_col=config.decision_time_col, resolution_col=config.label_end_time_col,
            side_col=config.side_col, min_rows=int(args.isotonic_min_rows),
        )
        mapped_name = f"{arm}__side_oof_isotonic"
        predictions[mapped_name] = mapped
        isotonic_predictions[arm] = mapped
        hierarchical, hierarchy_report = temporal_hierarchical_oof_calibration(
            frame,
            predictions[arm],
            net_ev,
            fold_id,
            decision_col=config.decision_time_col,
            resolution_col=config.label_end_time_col,
            side_col=config.side_col,
            min_rows=int(args.isotonic_min_rows),
        )
        hierarchical_name = f"{arm}__hierarchical_oof"
        predictions[hierarchical_name] = hierarchical
        hierarchical_predictions[arm] = hierarchical
        hierarchical_audit[arm] = hierarchy_report
        corrected, correction = apply_execution_ev_causal_recent_ev_correction(
            frame, hierarchical, net_ev, provenance, route="catboost_predicted_archetype", config=config,
        )
        final_name = f"{hierarchical_name}__causal_recent_ev"
        predictions[final_name] = corrected
        final_predictions[arm] = corrected
        calibrator_audit[arm] = audit
        correction_reports[arm] = correction

    metrics = metric_slices(
        frame, {**{f"{arm}__raw": predictions[arm] for arm in score_arms},
                **{f"{arm}__side_oof_isotonic_only": values for arm, values in isotonic_predictions.items()},
                **{f"{arm}__hierarchical_oof": values for arm, values in hierarchical_predictions.items()},
                **{f"{arm}__hierarchical_oof_recent": values for arm, values in final_predictions.items()}},
        net_ev, gross_ev, shared, decision_col=config.decision_time_col,
        side_col=config.side_col, top_fraction=float(args.top_fraction),
    )
    dominance_scopes = [
        scope
        for scope in metrics
        if scope == "all_oof" or scope.startswith("month_") or scope == "latest_week"
    ]
    dominance_rows: list[dict[str, Any]] = []
    for candidate in (MULTITASK_BLEND, MULTITASK_SHARED):
        for stage in (
            "raw",
            "side_oof_isotonic_only",
            "hierarchical_oof",
            "hierarchical_oof_recent",
        ):
            direct_name = f"{DIRECT}__{stage}"
            candidate_name = f"{candidate}__{stage}"
            stage_rows = []
            for scope in dominance_scopes:
                direct_value = float(
                    metrics[scope][direct_name]["top_k_mean_net_ev"]
                )
                candidate_value = float(
                    metrics[scope][candidate_name]["top_k_mean_net_ev"]
                )
                stage_rows.append(
                    {
                        "scope": scope,
                        "direct_top_k_mean_net_ev": direct_value,
                        "candidate_top_k_mean_net_ev": candidate_value,
                        "candidate_minus_direct": (
                            candidate_value - direct_value
                        ),
                        "candidate_noninferior": (
                            candidate_value >= direct_value
                        ),
                    }
                )
            aggregate_improves = next(
                row["candidate_minus_direct"] > 0.0
                for row in stage_rows
                if row["scope"] == "all_oof"
            )
            dominance_rows.append(
                {
                    "candidate": candidate,
                    "stage": stage,
                    "all_required_scopes_noninferior": all(
                        row["candidate_noninferior"] for row in stage_rows
                    ),
                    "aggregate_strictly_improves": aggregate_improves,
                    "promotion_gate_passed": all(
                        row["candidate_noninferior"] for row in stage_rows
                    )
                    and aggregate_improves,
                    "scopes": stage_rows,
                }
            )
    promoted_stages = [
        f"{row['candidate']}::{row['stage']}"
        for row in dominance_rows
        if row["promotion_gate_passed"]
    ]
    research_primary_score = (
        promoted_stages[0].split("::", 1)[0]
        if promoted_stages
        else DIRECT
    )
    economically_positive_stages = []
    for row in dominance_rows:
        if not row["promotion_gate_passed"]:
            continue
        required = {
            scope_row["scope"]: scope_row["candidate_top_k_mean_net_ev"]
            for scope_row in row["scopes"]
        }
        if required.get("all_oof", -np.inf) > 0.0 and required.get(
            "latest_week", -np.inf
        ) > 0.0:
            economically_positive_stages.append(
                f"{row['candidate']}::{row['stage']}"
            )
    production_primary_score = (
        economically_positive_stages[0].split("::", 1)[0]
        if economically_positive_stages
        else DIRECT
    )
    probability_metrics = {
        "executable_positive_net_ev": _probability_metrics(predictions["p_executable_positive_net_ev"], net_ev > 0.0, shared),
        "clean_favorable_first": _probability_metrics(predictions["p_clean_favorable_first"], clean_event, shared),
        "any_net_loss": _probability_metrics(predictions["p_any_net_loss"], net_ev <= 0.0, shared),
        "severe_net_loss": _probability_metrics(predictions["p_severe_net_loss"], net_ev <= -severe_floor, shared),
    }
    output_columns = [column for column in IDENTITY if column in frame.columns]
    output = frame.loc[:, output_columns].copy()
    output[config.decision_time_col] = frame[config.decision_time_col]
    output[config.label_end_time_col] = frame[config.label_end_time_col]
    output["execution_net_ev_12h"] = net_ev
    output["execution_gross_ev_12h"] = gross_ev
    output["clean_favorable_first_exact_policy"] = clean_event.astype(np.int8)
    output["severe_loss_floor"] = severe_floor
    output["oof_fold"] = fold_id
    for name, values in predictions.items():
        output[name] = values
    output.to_parquet(args.output_dir / "oof_predictions.parquet", index=False, compression="zstd")
    summary = {
        "schema": SCHEMA,
        "status": "strict_side_local_outer_oof_diagnostic_not_promoted",
        "architecture": {
            "direct": "E[execution_net_ev_12h]",
            "decomposed_complete": "P(net_ev>0) * E[net_ev | net_ev>0] - P(net_ev<=0) * E[-net_ev | net_ev<=0]",
            "decomposed_clean_partial_risk": "P(clean favorable-first) * E[max(net_ev,0) | clean] - P(net_ev<=-max(1%,2x_cost)) * E[-net_ev | severe]",
            "clean_target": "exact-policy tb_hard_label / risk_class=favorable-first; it is not net_ev>0",
            "executable_positive_component": "P(net_ev>0) is trained and reported separately; it is used only with E[net_ev|net_ev>0] in the matched complete sign partition",
            "units": "fractional return, net of execution_cost_return exactly once",
            "compact_cost_note": (
                "compact input exposes only the already-cost-adjusted net target; gross/cost audit fields are unavailable and no cost is subtracted again. The severe tail uses its fixed 1% floor."
                if args.compact_raw_market_context else None
            ),
        },
        "shared_outer_oof_rows": int(shared.sum()), "feature_columns": list(x.columns),
        "clean_event_rows": int(clean_event.sum()),
        "additional_input_families": list(families), "fold_audit": fold_audit,
        "probability_head_metrics": probability_metrics,
        "side_oof_isotonic_calibration": calibrator_audit,
        "hierarchical_oof_calibration": hierarchical_audit,
        "causal_recent_ev_correction": correction_reports,
        "multitask_auxiliary_blend": {
            "status": "diagnostic_strict_prior_outer_oof_blend",
            "primary_head": DIRECT,
            "features": list(MULTITASK_FEATURES),
            "fit_audit": multitask_audit,
            "shared_trunk_fit_audit": shared_multitask_audit,
            "shared_trunk_contract": (
                "side-local MLP on strictly prior outer-OOF head outputs; "
                "four direct-EV outputs dominate five auxiliary event/"
                "magnitude outputs; only the averaged direct output is scored"
            ),
            "dominance_gate": dominance_rows,
            "selected_research_primary_score": research_primary_score,
            "production_primary_score": production_primary_score,
            "relative_dominance_proven_stages": promoted_stages,
            "economically_positive_stages": economically_positive_stages,
            "production_promotion_eligible": bool(economically_positive_stages),
        },
        "metrics": metrics,
        "ranking_contract": "one pooled global top-k after side-specific temporal OOF isotonic map and causal recent-EV correction; no timestamp-local quota",
        "compact_contract": (
            "May-Jul19 compact causal context plus raw h0 market-state inputs; fold-local >=95% train-coverage admission; "
            "separately disclosed feature contract, not an absolute-level matched comparison to the 124-feature v5 run"
            if args.compact_raw_market_context else None
        ),
        "sources": {
            "input": {"path": str(args.input), "sha256": _sha256(args.input)},
            "provenance": (
                {"path": str(args.provenance), "sha256": _sha256(args.provenance), "schema": provenance_payload.get("schema")}
                if args.provenance is not None else {"schema": provenance_payload.get("schema"), "status": "compact_predeclared_causal_contract"}
            ),
            "clean_labels": {"path": str(args.clean_labels), "sha256": _sha256(args.clean_labels), "join": "exact_inner_one_to_one_on___ts___symbol_side_name_candidate_id"},
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary["metrics"].get("all_oof", {})), indent=2))


if __name__ == "__main__":
    main()
