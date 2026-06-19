"""Unsupervised HPO for advanced latent-regime encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    stratified_period_sample_positions,
)
from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
    AdvancedRegimeLearningArtifact,
    AdvancedRegimeLearningConfig,
    fit_advanced_regime_learning,
    save_advanced_regime_learning_artifact,
)


DEFAULT_REGIME_HPO_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_regimes": [3, 4, 5, 6],
    "min_regime_duration": [2, 4, 8, 12],
    "null_ratio": [0.5, 1.0],
    "null_block_size": [12, 24, 48],
    "max_depth": [3, 4],
    "min_leaf_fraction": [0.025, 0.05],
    "n_estimators": [40, 80],
    "learning_rate": [0.03, 0.05, 0.08],
    "lgbm_feature_fraction": [0.70, 0.85, 1.0],
    "lgbm_bagging_fraction": [0.70, 0.85, 1.0],
    "lgbm_bagging_freq": [0, 1],
    "lgbm_min_gain_to_split": [0.0, 0.01],
    "lgbm_lambda_l1": [0.0, 0.1],
    "lgbm_lambda_l2": [0.0, 0.1, 1.0],
    "stability_top_m": [40, 80],
    "leaf_embedding_dim": [4, 8, 12],
    "leaf_embedding_max_trees": [32, 64],
    "raw_embedding_dim": [4, 8, 12],
    "bayesian_gmm_covariance_type": ["diag"],
    "bayesian_gmm_weight_concentration_prior": [0.0, 0.01, 0.1, 1.0],
    "bayesian_gmm_reg_covar": [1e-6, 1e-5, 1e-4],
    "bayesian_gmm_max_iter": [100, 200],
    "hdbscan_min_cluster_size": [0],
    "hdbscan_min_cluster_size_fraction": [0.0, 0.02, 0.05],
    "hdbscan_min_samples": [0, 3, 8],
    "hdbscan_cluster_selection_epsilon": [0.0, 0.05],
    "hdbscan_cluster_selection_method": ["eom", "leaf"],
    "hmm_covariance_type": ["diag", "spherical"],
    "hmm_n_iter": [80, 120],
    "hmm_tol": [1e-2, 1e-3],
    "hmm_min_covar": [1e-3, 1e-2],
    "hmm_transmat_self_bias": [0.0, 2.0, 5.0],
    "hmm_startprob_prior": [1.0, 5.0],
    "spectral_n_neighbors": [5, 10, 20],
    "spectral_affinity": ["nearest_neighbors", "rbf"],
    "spectral_assign_labels": ["kmeans", "discretize"],
    "spectral_gamma": [0.5, 1.0, 2.0],
    "kmeans_n_init": [5, 10, 20],
    "kmeans_max_iter": [200, 300],
    "kmeans_tol": [1e-4, 1e-3],
    "kmeans_algorithm": ["lloyd", "elkan"],
    "mfa_regimes": [3, 5],
    "mfa_factors": [2, 3],
    "mfa_l1_lambda": [0.0, 0.001, 0.005],
    "ae_latent_dim": [4, 8],
    "ae_hidden_dim": [16, 32],
    "ae_dropout": [0.03, 0.08],
    "ae_noise": [0.01, 0.03],
    "ae_lambda_sparse": [0.0005, 0.001, 0.003],
    "ae_lambda_contrastive": [0.10, 0.20],
    "ae_lambda_smooth": [0.0, 0.01],
    "ae_temperature": [0.15, 0.25],
}


@dataclass(frozen=True)
class RegimeHPOConfig:
    random_state: int = 42
    max_trials: int = 16
    search_space: Mapping[str, Sequence[Any]] | None = None
    artifact_output_dir: str | Path | None = None
    store_trial_artifacts: bool = False
    store_best_artifact: bool = True
    store_feature_metrics: bool = True
    prefer_non_baseline: bool = True
    objective_mode: str = "learnability"
    structure_objective_weight: float = 0.35
    learnability_objective_weight: float = 0.65
    self_predictability_weight: float = 0.18
    soft_state_quality_weight: float = 0.12
    compression_quality_weight: float = 0.12
    support_quality_weight: float = 0.18
    specialist_feature_helpfulness_weight: float = 0.20
    feature_conditional_learnability_weight: float = 0.20
    self_predictability_target: float = 0.78
    self_predictability_width: float = 0.25
    soft_state_entropy_target: float = 0.35
    soft_state_confidence_target: float = 0.75
    specialist_helpfulness_horizon: int = 1
    feature_conditional_max_features: int = 96
    feature_conditional_incremental_accuracy_target: float = 0.15
    stratified_sample: bool = True
    max_hpo_rows: int = 50000
    hpo_sample_time_bins: int = 32
    max_total_runtime_seconds: float = 0.0
    max_failed_trials: int = 3
    early_stopping_patience: int = 8
    early_stopping_min_trials: int = 6
    early_stopping_min_delta: float = 1e-4
    median_pruner_enabled: bool = True
    median_pruner_warmup_trials: int = 5
    median_pruner_min_delta: float = 0.0
    median_pruner_stop_after_pruned_streak: int = 4
    max_trial_rows: int = 50000
    max_trial_classifier_rows: int = 60000
    max_trial_ae_train_rows: int = 20000
    max_trial_assessment_rows: int = 20000
    max_trial_geometry_rows_per_regime: int = 512
    max_trial_leaf_trees: int = 64
    max_trial_n_estimators: int = 120
    max_trial_ae_epochs: int = 50
    max_trial_mfa_iter: int = 30
    max_trial_bayesian_gmm_iter: int = 200
    max_trial_hmm_iter: int = 120
    max_trial_kmeans_max_iter: int = 300
    min_support_floor: float = 0.03
    turnover_target: float = 0.25
    score_variance_penalty_weight: float = 0.10
    low_support_penalty_weight: float = 0.10
    turnover_penalty_weight: float = 0.10
    trend_vol_replica_penalty_weight: float = 0.10
    compute_cost_penalty_weight: float = 0.02


@dataclass(frozen=True)
class RegimeHPOResult:
    trials: pd.DataFrame
    trial_steps: pd.DataFrame
    trial_model_feature_metrics: pd.DataFrame
    best_config: AdvancedRegimeLearningConfig | None
    best_artifact: AdvancedRegimeLearningArtifact | None
    best_trial_params: dict[str, Any]
    hpo_config: RegimeHPOConfig
    output_paths: dict[str, str]


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _as_choices(value: Sequence[Any] | Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        choices = list(value)
        return choices if choices else [None]
    return [value]


def _normalise_search_space(
    search_space: Mapping[str, Sequence[Any]] | None,
) -> dict[str, list[Any]]:
    raw = dict(search_space) if search_space is not None else dict(DEFAULT_REGIME_HPO_SEARCH_SPACE)
    valid_fields = set(AdvancedRegimeLearningConfig.__dataclass_fields__)
    out: dict[str, list[Any]] = {}
    for key, values in raw.items():
        key_str = str(key)
        if key_str == "random_state" or key_str not in valid_fields:
            continue
        out[key_str] = _as_choices(values)
    return out


def _sample_trial_params(
    search_space: Mapping[str, Sequence[Any]] | None,
    *,
    max_trials: int,
    random_state: int,
) -> list[dict[str, Any]]:
    space = _normalise_search_space(search_space)
    if not space:
        return [{}]
    keys = list(space)
    sizes = [max(1, len(space[key])) for key in keys]
    total = int(np.prod(np.asarray(sizes, dtype=np.int64)))
    cap = max(1, int(max_trials or 1))
    if total <= cap:
        rows: list[dict[str, Any]] = []

        def _walk(i: int, current: dict[str, Any]) -> None:
            if i >= len(keys):
                rows.append(dict(current))
                return
            key = keys[i]
            for value in space[key]:
                current[key] = value
                _walk(i + 1, current)

        _walk(0, {})
        return rows
    rng = np.random.default_rng(int(random_state))
    rows = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = cap * 50
    while len(rows) < cap and attempts < max_attempts:
        attempts += 1
        params = {key: space[key][int(rng.integers(0, len(space[key])))] for key in keys}
        sig = tuple(params[key] for key in keys)
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(params)
    return rows


def _trial_config(
    base_config: AdvancedRegimeLearningConfig,
    params: Mapping[str, Any],
) -> AdvancedRegimeLearningConfig:
    valid_fields = set(AdvancedRegimeLearningConfig.__dataclass_fields__)
    updates = {
        str(key): value
        for key, value in params.items()
        if str(key) in valid_fields and str(key) != "random_state"
    }
    return replace(base_config, **updates)


def _sample_hpo_frame(
    frame: pd.DataFrame,
    *,
    base_config: AdvancedRegimeLearningConfig,
    hpo_config: RegimeHPOConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    max_rows = int(hpo_config.max_hpo_rows or 0)
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame, {
            "hpo_input_rows": int(len(frame)),
            "hpo_sampled_rows": int(len(frame)),
            "hpo_sampling": "none",
        }
    positions = np.arange(len(frame), dtype=np.int64)
    if bool(hpo_config.stratified_sample):
        sampled = stratified_period_sample_positions(
            frame,
            positions,
            max_rows=max_rows,
            timestamp_col=base_config.timestamp_col,
            symbol_col=base_config.symbol_col,
            n_periods=int(hpo_config.hpo_sample_time_bins or base_config.sample_time_bins),
        )
        sampled = np.asarray(sampled, dtype=np.int64)
        sampling = "stratified_period_symbol"
    else:
        rng = np.random.default_rng(int(hpo_config.random_state) + 9100)
        sampled = np.sort(rng.choice(positions, size=max_rows, replace=False)).astype(np.int64)
        sampling = "random"
    if sampled.size == 0:
        return frame, {
            "hpo_input_rows": int(len(frame)),
            "hpo_sampled_rows": int(len(frame)),
            "hpo_sampling": "fallback_none",
        }
    sampled_frame = frame.iloc[np.sort(np.unique(sampled))].copy(deep=False)
    return sampled_frame, {
        "hpo_input_rows": int(len(frame)),
        "hpo_sampled_rows": int(len(sampled_frame)),
        "hpo_sampling": sampling,
    }


def _cap_positive(value: int | float, cap: int | float) -> Any:
    if cap is None:
        return value
    try:
        cap_value = float(cap)
    except Exception:
        return value
    if cap_value <= 0.0 or not np.isfinite(cap_value):
        return value
    if isinstance(value, int):
        return int(min(int(value), int(cap_value)))
    try:
        return min(value, type(value)(cap_value))
    except Exception:
        return min(float(value), cap_value)


def _bounded_trial_config(
    base_config: AdvancedRegimeLearningConfig,
    params: Mapping[str, Any],
    hpo_config: RegimeHPOConfig,
) -> AdvancedRegimeLearningConfig:
    cfg = _trial_config(base_config, params)
    updates = {
        "max_rows": _cap_positive(int(cfg.max_rows), int(hpo_config.max_trial_rows)),
        "sample_time_bins": max(1, min(int(cfg.sample_time_bins), int(hpo_config.hpo_sample_time_bins or cfg.sample_time_bins))),
        "max_classifier_rows": _cap_positive(int(cfg.max_classifier_rows), int(hpo_config.max_trial_classifier_rows)),
        "ae_max_train_rows": _cap_positive(int(cfg.ae_max_train_rows), int(hpo_config.max_trial_ae_train_rows)),
        "regime_assessment_max_auc_rows": _cap_positive(
            int(cfg.regime_assessment_max_auc_rows),
            int(hpo_config.max_trial_assessment_rows),
        ),
        "regime_assessment_max_robustness_rows": _cap_positive(
            int(cfg.regime_assessment_max_robustness_rows),
            int(hpo_config.max_trial_assessment_rows),
        ),
        "regime_assessment_max_geometry_rows_per_regime": _cap_positive(
            int(cfg.regime_assessment_max_geometry_rows_per_regime),
            int(hpo_config.max_trial_geometry_rows_per_regime),
        ),
        "leaf_embedding_max_trees": _cap_positive(
            int(cfg.leaf_embedding_max_trees),
            int(hpo_config.max_trial_leaf_trees),
        ),
        "n_estimators": _cap_positive(int(cfg.n_estimators), int(hpo_config.max_trial_n_estimators)),
        "ae_epochs": _cap_positive(int(cfg.ae_epochs), int(hpo_config.max_trial_ae_epochs)),
        "mfa_max_iter": _cap_positive(int(cfg.mfa_max_iter), int(hpo_config.max_trial_mfa_iter)),
        "bayesian_gmm_max_iter": _cap_positive(
            int(cfg.bayesian_gmm_max_iter),
            int(hpo_config.max_trial_bayesian_gmm_iter),
        ),
        "hmm_n_iter": _cap_positive(int(cfg.hmm_n_iter), int(hpo_config.max_trial_hmm_iter)),
        "kmeans_max_iter": _cap_positive(
            int(cfg.kmeans_max_iter),
            int(hpo_config.max_trial_kmeans_max_iter),
        ),
    }
    return replace(cfg, **updates)


def _float_from_row(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _method_label_column(artifact: AdvancedRegimeLearningArtifact, method: str) -> str | None:
    labels = getattr(artifact, "regime_labels", pd.DataFrame())
    if not isinstance(labels, pd.DataFrame) or labels.empty:
        return None
    preferred = f"{method}_smoothed_regime"
    if preferred in labels.columns:
        return preferred
    fallback = f"{method}_raw_regime"
    if fallback in labels.columns:
        return fallback
    matches = [str(col) for col in labels.columns if str(col).startswith(f"{method}_")]
    return matches[0] if matches else None


def _method_probability_columns(artifact: AdvancedRegimeLearningArtifact, method: str) -> list[str]:
    probs = getattr(artifact, "regime_probabilities", pd.DataFrame())
    if not isinstance(probs, pd.DataFrame) or probs.empty:
        return []
    prefix = f"{method}_regime_prob_"
    return [str(col) for col in probs.columns if str(col).startswith(prefix)]


def _method_transition_columns(artifact: AdvancedRegimeLearningArtifact, method: str) -> list[str]:
    transitions = getattr(artifact, "regime_transition_features", pd.DataFrame())
    if not isinstance(transitions, pd.DataFrame) or transitions.empty:
        return []
    prefix = f"url_{method}_"
    return [str(col) for col in transitions.columns if str(col).startswith(prefix)]


def _row_groups_for_artifact(
    artifact: AdvancedRegimeLearningArtifact,
    frame: pd.DataFrame | None = None,
) -> list[np.ndarray]:
    n = len(getattr(artifact, "regime_labels", pd.DataFrame()))
    if n == 0 and frame is not None:
        n = len(frame)
    keys = getattr(artifact, "row_keys", pd.DataFrame())
    if not isinstance(keys, pd.DataFrame) or keys.empty:
        keys = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(index=np.arange(n))
    if len(keys) != n:
        return [np.arange(n, dtype=np.int64)]
    timestamp_col = next((str(col) for col in keys.columns if "time" in str(col).lower()), None)
    symbol_col = next((str(col) for col in keys.columns if "symbol" in str(col).lower()), None)
    if timestamp_col is not None:
        ts = pd.to_datetime(keys[timestamp_col], utc=True, errors="coerce")
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
        valid_ts = ts.notna().to_numpy(dtype=bool)
    else:
        ts_ns = np.arange(n, dtype=np.int64)
        valid_ts = np.ones(n, dtype=bool)
    if symbol_col is not None:
        symbols = keys[symbol_col].astype(str).to_numpy()
    else:
        symbols = np.repeat("__all__", n)
    groups: list[np.ndarray] = []
    for symbol in pd.unique(symbols):
        pos = np.flatnonzero(symbols == symbol).astype(np.int64)
        if pos.size == 0:
            continue
        valid_pos = pos[valid_ts[pos]]
        invalid_pos = pos[~valid_ts[pos]]
        if valid_pos.size:
            ordered = valid_pos[np.argsort(ts_ns[valid_pos], kind="mergesort")]
            pos = np.concatenate([ordered, invalid_pos])
        groups.append(pos.astype(np.int64, copy=False))
    return groups or [np.arange(n, dtype=np.int64)]


def _lag_positions_from_groups(n: int, groups: Sequence[np.ndarray], lag: int = 1) -> np.ndarray:
    out = np.full(n, -1, dtype=np.int64)
    step = max(1, int(lag))
    for group in groups:
        pos = np.asarray(group, dtype=np.int64)
        if pos.size <= step:
            continue
        out[pos[step:]] = pos[:-step]
    return out


def _temporal_blocks_from_groups(groups: Sequence[np.ndarray], n_splits: int = 3) -> list[np.ndarray]:
    ordered = np.concatenate([np.asarray(group, dtype=np.int64) for group in groups if len(group)])
    if ordered.size == 0:
        return []
    split_count = max(2, min(int(n_splits), ordered.size))
    return [block.astype(np.int64, copy=False) for block in np.array_split(ordered, split_count) if block.size]


def _cv_balanced_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    blocks: Sequence[np.ndarray],
    random_state: int,
    max_rows: int = 20000,
) -> float:
    arr = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != labels.size or arr.shape[1] == 0:
        return 0.0
    finite = np.isfinite(arr).all(axis=1) & (labels >= 0)
    classes, counts = np.unique(labels[finite], return_counts=True)
    if int(np.sum(finite)) < 12 or classes.size < 2 or int(np.min(counts)) < 2:
        return 0.0
    all_pos = np.flatnonzero(finite)
    scores: list[float] = []
    for fold_i, block in enumerate(blocks):
        test_idx = np.asarray(block, dtype=np.int64)
        test_idx = test_idx[finite[test_idx]]
        if test_idx.size < 3 or np.unique(labels[test_idx]).size < 2:
            continue
        train_idx = np.setdiff1d(all_pos, test_idx, assume_unique=False)
        if train_idx.size < 8 or np.unique(labels[train_idx]).size < 2:
            continue
        cap = int(max_rows or 0)
        if cap > 0 and train_idx.size + test_idx.size > cap:
            rng = np.random.default_rng(int(random_state) + 23000 + fold_i)
            train_cap = max(8, int(0.75 * cap))
            test_cap = max(4, cap - train_cap)
            if train_idx.size > train_cap:
                train_idx = np.sort(rng.choice(train_idx, size=train_cap, replace=False)).astype(np.int64)
            if test_idx.size > test_cap:
                test_idx = np.sort(rng.choice(test_idx, size=test_cap, replace=False)).astype(np.int64)
        try:
            model = RandomForestClassifier(
                n_estimators=30,
                max_depth=3,
                min_samples_leaf=max(2, int(0.05 * len(train_idx))),
                random_state=int(random_state) + fold_i,
                n_jobs=1,
            )
            model.fit(arr[train_idx], labels[train_idx])
            pred = model.predict(arr[test_idx])
            scores.append(float(balanced_accuracy_score(labels[test_idx], pred)))
        except Exception:
            continue
    return _clamp01(float(np.nanmean(scores))) if scores else 0.0


def _learnability_from_accuracy(
    accuracy: float,
    *,
    target: float,
    width: float,
) -> float:
    if not np.isfinite(accuracy) or accuracy <= 0.5:
        return 0.0
    baseline_scaled = _clamp01((float(accuracy) - 0.5) / 0.35)
    band = float(np.exp(-((float(accuracy) - float(target)) / max(float(width), 1e-6)) ** 2))
    return _clamp01(0.65 * baseline_scaled + 0.35 * band)


def _regime_self_predictability(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
    *,
    frame: pd.DataFrame | None,
    hpo_config: RegimeHPOConfig,
    random_state: int,
) -> tuple[float, float]:
    label_col = _method_label_column(artifact, method)
    labels_df = getattr(artifact, "regime_labels", pd.DataFrame())
    if label_col is None or not isinstance(labels_df, pd.DataFrame) or label_col not in labels_df:
        return 0.0, 0.0
    labels = pd.to_numeric(labels_df[label_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    n = labels.size
    groups = _row_groups_for_artifact(artifact, frame)
    lag_pos = _lag_positions_from_groups(n, groups, lag=1)
    valid = (lag_pos >= 0) & (labels >= 0) & (labels[lag_pos] >= 0)
    if int(np.sum(valid)) < 12:
        return 0.0, 0.0
    states = sorted(int(v) for v in np.unique(labels[valid]) if int(v) >= 0)
    state_pos = {state: i for i, state in enumerate(states)}
    lag_one_hot = np.zeros((n, max(len(states), 1)), dtype=np.float32)
    for state, idx in state_pos.items():
        lag_one_hot[:, idx] = labels[lag_pos.clip(min=0)] == state
    parts = [lag_one_hot]
    prob_cols = _method_probability_columns(artifact, method)
    probs = getattr(artifact, "regime_probabilities", pd.DataFrame())
    if prob_cols and isinstance(probs, pd.DataFrame):
        prob_arr = probs.reindex(columns=prob_cols).to_numpy(dtype=np.float32, copy=False)
        parts.append(np.nan_to_num(prob_arr[lag_pos.clip(min=0)], nan=0.0, posinf=0.0, neginf=0.0))
    transition_cols = _method_transition_columns(artifact, method)
    transitions = getattr(artifact, "regime_transition_features", pd.DataFrame())
    if transition_cols and isinstance(transitions, pd.DataFrame):
        trans_arr = transitions.reindex(columns=transition_cols).to_numpy(dtype=np.float32, copy=False)
        parts.append(np.nan_to_num(trans_arr[lag_pos.clip(min=0)], nan=0.0, posinf=0.0, neginf=0.0))
    x = np.hstack(parts).astype(np.float32, copy=False)
    y = labels.copy()
    y[~valid] = -1
    accuracy = _cv_balanced_accuracy(
        x,
        y,
        blocks=_temporal_blocks_from_groups(groups, n_splits=3),
        random_state=int(random_state),
        max_rows=int(hpo_config.max_trial_assessment_rows),
    )
    score = _learnability_from_accuracy(
        accuracy,
        target=float(hpo_config.self_predictability_target),
        width=float(hpo_config.self_predictability_width),
    )
    return score, accuracy


def _soft_state_quality(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
    *,
    frame: pd.DataFrame | None,
    hpo_config: RegimeHPOConfig,
) -> dict[str, float]:
    prob_cols = _method_probability_columns(artifact, method)
    probs_df = getattr(artifact, "regime_probabilities", pd.DataFrame())
    if not prob_cols or not isinstance(probs_df, pd.DataFrame):
        return {"soft_state_quality": 0.0}
    probs = probs_df.reindex(columns=prob_cols).to_numpy(dtype=np.float32, copy=False)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, np.inf)
    denom = probs.sum(axis=1, keepdims=True)
    probs = np.divide(
        probs,
        np.maximum(denom, 1e-12),
        out=np.full_like(probs, 1.0 / max(probs.shape[1], 1)),
        where=denom > 1e-12,
    )
    if probs.shape[1] <= 1:
        return {"soft_state_quality": 0.0, "soft_state_entropy_mean": 0.0, "soft_state_prob_max_mean": 1.0}
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=1) / np.log(float(probs.shape[1]))
    prob_max = np.max(probs, axis=1)
    entropy_mean = float(np.nanmean(entropy))
    prob_max_mean = float(np.nanmean(prob_max))
    entropy_score = _clamp01(1.0 - abs(entropy_mean - float(hpo_config.soft_state_entropy_target)) / 0.50)
    confidence_score = _clamp01(1.0 - abs(prob_max_mean - float(hpo_config.soft_state_confidence_target)) / 0.35)
    groups = _row_groups_for_artifact(artifact, frame)
    lag_pos = _lag_positions_from_groups(len(probs), groups, lag=1)
    valid = lag_pos >= 0
    persistence_score = 0.0
    if valid.any():
        prob_change = 0.5 * np.sum(np.abs(probs[valid] - probs[lag_pos[valid]]), axis=1)
        persistence_score = _clamp01(1.0 - float(np.nanmean(prob_change)) / 0.50)
    transition_score = 0.0
    label_col = _method_label_column(artifact, method)
    labels_df = getattr(artifact, "regime_labels", pd.DataFrame())
    if label_col and isinstance(labels_df, pd.DataFrame) and label_col in labels_df and valid.any():
        labels = pd.to_numeric(labels_df[label_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        changed = valid & (labels != labels[lag_pos.clip(min=0)]) & (labels >= 0) & (labels[lag_pos.clip(min=0)] >= 0)
        unchanged = valid & ~changed
        if changed.any() and unchanged.any():
            uncertainty_changed = 1.0 - prob_max[changed]
            uncertainty_unchanged = 1.0 - prob_max[unchanged]
            transition_score = _clamp01(
                0.5 + float(np.nanmean(uncertainty_changed) - np.nanmean(uncertainty_unchanged))
            )
    quality = _clamp01(
        0.35 * entropy_score
        + 0.30 * confidence_score
        + 0.20 * persistence_score
        + 0.15 * transition_score
    )
    return {
        "soft_state_quality": quality,
        "soft_state_entropy_mean": entropy_mean,
        "soft_state_prob_max_mean": prob_max_mean,
        "soft_state_entropy_score": entropy_score,
        "soft_state_confidence_score": confidence_score,
        "soft_state_persistence_score": persistence_score,
        "soft_state_transition_score": transition_score,
    }


def _support_quality(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
    *,
    hpo_config: RegimeHPOConfig,
) -> dict[str, float]:
    label_col = _method_label_column(artifact, method)
    labels_df = getattr(artifact, "regime_labels", pd.DataFrame())
    if label_col is None or not isinstance(labels_df, pd.DataFrame) or label_col not in labels_df:
        return {"support_quality": 0.0}
    labels = pd.to_numeric(labels_df[label_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid = labels >= 0
    if int(np.sum(valid)) == 0:
        return {"support_quality": 0.0}
    _states, counts = np.unique(labels[valid], return_counts=True)
    freq = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    entropy = -float(np.sum(freq * np.log(np.maximum(freq, 1e-12))))
    norm_entropy = entropy / np.log(float(len(freq))) if len(freq) > 1 else 0.0
    min_support = float(np.min(freq)) if freq.size else 0.0
    max_support = float(np.max(freq)) if freq.size else 1.0
    min_support_score = _clamp01(min_support / max(float(hpo_config.min_support_floor), 1e-6))
    dominance_score = _clamp01((1.0 - max_support) / max(1.0 - 1.0 / max(len(freq), 1), 1e-6)) if len(freq) > 1 else 0.0
    quality = _clamp01(0.55 * norm_entropy + 0.35 * min_support_score + 0.10 * dominance_score)
    return {
        "support_quality": quality,
        "label_entropy": float(norm_entropy),
        "label_min_support": min_support,
        "label_max_support": max_support,
        "effective_regime_count": float(np.exp(entropy)),
    }


def _embedding_for_method(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
) -> pd.DataFrame:
    embeddings = getattr(artifact, "method_embeddings", {}) or {}
    if method == "mfa":
        return getattr(artifact, "mfa_responsibilities", pd.DataFrame())
    if method.startswith("raw_pca") or method == "raw_selected_kmeans":
        return embeddings.get("raw_pca", pd.DataFrame())
    if method.startswith("raw_spectral"):
        return embeddings.get("raw_spectral", pd.DataFrame())
    if method.startswith("leaf_umap"):
        return embeddings.get("leaf_umap", pd.DataFrame())
    if method.startswith("leaf_spectral"):
        return embeddings.get("leaf_spectral", pd.DataFrame())
    if method.startswith("leaf_"):
        return embeddings.get("leaf_pca", pd.DataFrame())
    if method.startswith("sparse_ae"):
        return embeddings.get("sparse_ae", pd.DataFrame())
    if method.startswith("contrastive_ae"):
        return embeddings.get("contrastive_ae", pd.DataFrame())
    if method.startswith("contrastive_leaf"):
        return embeddings.get("contrastive_leaf", pd.DataFrame())
    return pd.DataFrame()


def _embedding_compression_score(values: pd.DataFrame) -> float:
    if not isinstance(values, pd.DataFrame) or values.empty:
        return 0.0
    arr = values.to_numpy(dtype=np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] < 3 or arr.shape[1] == 0:
        return 0.0
    std = np.std(arr, axis=0)
    active = std > 1e-8
    if not bool(np.any(active)):
        return 0.0
    arr = arr[:, active]
    if arr.shape[1] == 1:
        return 0.45
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    cov = centered.T @ centered / max(float(arr.shape[0] - 1), 1.0)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(np.asarray(eig, dtype=np.float64), 0.0, np.inf)
    if float(np.sum(eig)) <= 1e-12:
        return 0.0
    share = eig / float(np.sum(eig))
    entropy = -float(np.sum(share * np.log(np.maximum(share, 1e-12))))
    effective_rank = float(np.exp(entropy))
    compactness = _clamp01(1.0 - (effective_rank - 1.0) / max(float(arr.shape[1] - 1), 1.0))
    activity = _clamp01(float(np.mean(active)))
    return _clamp01(0.70 * compactness + 0.30 * activity)


def _compression_quality(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
) -> dict[str, float]:
    embedding_score = _embedding_compression_score(_embedding_for_method(artifact, method))
    diagnostics = getattr(artifact, "diagnostics", {}) if isinstance(getattr(artifact, "diagnostics", {}), Mapping) else {}
    aux_score = 0.0
    diag_key = None
    if method.startswith("sparse_ae"):
        diag_key = "autoencoder"
    elif method.startswith("contrastive_ae"):
        diag_key = "contrastive_autoencoder"
    elif method.startswith("contrastive_leaf"):
        diag_key = "contrastive_leaf_autoencoder"
    if diag_key:
        diag = diagnostics.get(diag_key, {})
        if isinstance(diag, Mapping):
            initial = _float_from_row(diag, "loss_initial", np.nan)
            final = _float_from_row(diag, "loss_final", np.nan)
            if np.isfinite(initial) and np.isfinite(final) and initial > 1e-12:
                aux_score = _clamp01((initial - final) / max(abs(initial), 1e-12))
    elif method == "mfa":
        ll = diagnostics.get("mfa_log_likelihood", [])
        if isinstance(ll, Sequence) and len(ll) >= 2:
            start = float(ll[0])
            end = float(ll[-1])
            aux_score = _clamp01(np.tanh(max(end - start, 0.0) / max(abs(start), 1.0)))
        relevance = getattr(artifact, "mfa_feature_relevance", pd.DataFrame())
        if isinstance(relevance, pd.DataFrame) and "mfa_relevance" in relevance.columns:
            vals = pd.to_numeric(relevance["mfa_relevance"], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                aux_score = max(aux_score, _clamp01(float(np.mean(vals > np.nanmedian(vals)))))
    quality = _clamp01(0.70 * embedding_score + 0.30 * aux_score)
    return {
        "compression_quality": quality,
        "embedding_compression_score": embedding_score,
        "compression_aux_score": aux_score,
    }


def _future_feature_state_target(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    groups: Sequence[np.ndarray],
    horizon: int,
) -> np.ndarray:
    cols = [str(col) for col in feature_columns if str(col) in frame.columns]
    if not cols:
        return np.full(len(frame), -1, dtype=np.int64)
    values = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    med = np.nanmedian(values, axis=0)
    iqr = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scale = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0).astype(np.float32)
    filled = values.copy()
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = np.take(med, np.where(missing)[1])
    z = (filled - med.reshape(1, -1)) / scale.reshape(1, -1)
    future_pos = np.full(len(frame), -1, dtype=np.int64)
    step = max(1, int(horizon))
    for group in groups:
        pos = np.asarray(group, dtype=np.int64)
        if pos.size <= step:
            continue
        future_pos[pos[:-step]] = pos[step:]
    valid = future_pos >= 0
    stress = np.full(len(frame), np.nan, dtype=np.float32)
    if valid.any():
        stress[valid] = np.mean(np.abs(z[future_pos[valid]] - z[valid]), axis=1).astype(np.float32)
    finite = np.isfinite(stress)
    target = np.full(len(frame), -1, dtype=np.int64)
    if int(np.sum(finite)) < 12:
        return target
    threshold = float(np.nanmedian(stress[finite]))
    target[finite] = (stress[finite] > threshold).astype(np.int64)
    if np.unique(target[target >= 0]).size < 2:
        target[finite] = (stress[finite] >= threshold).astype(np.int64)
    return target


def _specialist_feature_helpfulness(
    artifact: AdvancedRegimeLearningArtifact,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    hpo_config: RegimeHPOConfig,
    random_state: int,
) -> tuple[float, float]:
    features = getattr(artifact, "model_regime_features", pd.DataFrame())
    if not isinstance(features, pd.DataFrame) or features.empty or len(features) != len(frame):
        return 0.0, 0.0
    groups = _row_groups_for_artifact(artifact, frame)
    target = _future_feature_state_target(
        frame,
        feature_columns,
        groups=groups,
        horizon=int(hpo_config.specialist_helpfulness_horizon),
    )
    x = features.to_numpy(dtype=np.float32, copy=False)
    accuracy = _cv_balanced_accuracy(
        x,
        target,
        blocks=_temporal_blocks_from_groups(groups, n_splits=3),
        random_state=int(random_state),
        max_rows=int(hpo_config.max_trial_assessment_rows),
    )
    score = _clamp01((accuracy - 0.5) / 0.25)
    return score, accuracy


def _hpo_feature_family(name: str) -> str:
    low = str(name).lower()
    if low.startswith("cov_"):
        return "covariance"
    if low.startswith("corr_"):
        return "correlation"
    if low.startswith("q_"):
        return "quantile"
    if low.startswith("autocorr"):
        return "autocorr"
    if low.startswith("eig_"):
        return "eigen"
    if low.startswith("svd") or "knn" in low:
        return "svd_knn"
    if "fund" in low:
        return "funding"
    if "oi" in low or "open_interest" in low:
        return "open_interest"
    if "volume" in low or "amihud" in low or "liquidity" in low or "rvol" in low:
        return "liquidity"
    if "volatility" in low or "atr" in low or "range" in low or low.startswith("rv_") or "variance" in low:
        return "volatility"
    if "trend" in low or "ema" in low or "momentum" in low or "slope" in low:
        return "trend"
    if "entropy" in low or "efficiency" in low or "coherence" in low:
        return "path_structure"
    return "primitive"


def _feature_conditional_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_features: int,
) -> tuple[np.ndarray, list[str]]:
    features = [str(col) for col in dict.fromkeys(feature_columns) if str(col) in frame.columns]
    if not features:
        return np.zeros((len(frame), 0), dtype=np.float32), []
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros((len(frame), 0), dtype=np.float32), []
    finite_fraction = np.isfinite(values).mean(axis=0)
    keep = finite_fraction >= 0.50
    if not bool(np.any(keep)):
        return np.zeros((len(frame), 0), dtype=np.float32), []
    values = values[:, keep]
    features = [feature for feature, ok in zip(features, keep) if bool(ok)]
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    q25 = np.nanpercentile(values, 25.0, axis=0)
    q75 = np.nanpercentile(values, 75.0, axis=0)
    iqr = (q75 - q25).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(median, np.where(missing)[1])
    scaled = (values - median.reshape(1, -1)) / scale.reshape(1, -1)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=8.0, neginf=-8.0)
    scaled = np.clip(scaled, -8.0, 8.0).astype(np.float32, copy=False)
    cap = int(max_features or 0)
    if cap > 0 and scaled.shape[1] > cap:
        variance = np.nanvar(scaled, axis=0)
        variance = np.where(np.isfinite(variance), variance, 0.0)
        idx = np.argsort(variance, kind="mergesort")[-cap:]
        idx = np.sort(idx).astype(np.int64)
        scaled = scaled[:, idx].astype(np.float32, copy=False)
        features = [features[int(i)] for i in idx]
    return scaled, features


def _feature_conditional_learnability(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    hpo_config: RegimeHPOConfig,
    random_state: int,
) -> dict[str, float]:
    label_col = _method_label_column(artifact, method)
    labels_df = getattr(artifact, "regime_labels", pd.DataFrame())
    if label_col is None or not isinstance(labels_df, pd.DataFrame) or label_col not in labels_df:
        return {
            "feature_conditional_learnability": 0.0,
            "feature_conditional_accuracy_all": 0.0,
            "feature_conditional_accuracy_trend_vol": 0.5,
            "feature_conditional_incremental_accuracy": 0.0,
            "feature_conditional_incremental_score": 0.0,
            "feature_conditional_trend_vol_replica_penalty": 0.0,
        }
    labels = pd.to_numeric(labels_df[label_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    groups = _row_groups_for_artifact(artifact, frame)
    blocks = _temporal_blocks_from_groups(groups, n_splits=3)
    max_features = max(1, int(hpo_config.feature_conditional_max_features or 1))
    x_all, kept_all = _feature_conditional_matrix(
        frame,
        feature_columns,
        max_features=max_features,
    )
    trend_vol_features = [
        str(feature)
        for feature in feature_columns
        if _hpo_feature_family(str(feature)) in {"trend", "volatility"}
    ]
    x_trend_vol, kept_trend_vol = _feature_conditional_matrix(
        frame,
        trend_vol_features,
        max_features=max_features,
    )
    accuracy_all = _cv_balanced_accuracy(
        x_all,
        labels,
        blocks=blocks,
        random_state=int(random_state),
        max_rows=int(hpo_config.max_trial_assessment_rows),
    )
    if x_trend_vol.shape[1] == 0 or not kept_trend_vol:
        accuracy_trend_vol = 0.5
    else:
        accuracy_trend_vol = _cv_balanced_accuracy(
            x_trend_vol,
            labels,
            blocks=blocks,
            random_state=int(random_state) + 17,
            max_rows=int(hpo_config.max_trial_assessment_rows),
        )
        if accuracy_trend_vol <= 0.0 or not np.isfinite(accuracy_trend_vol):
            accuracy_trend_vol = 0.5
    baseline = max(0.5, float(accuracy_trend_vol))
    incremental = max(float(accuracy_all) - baseline, 0.0)
    target = max(float(hpo_config.feature_conditional_incremental_accuracy_target), 1e-6)
    incremental_score = _clamp01(incremental / target)
    all_score = _learnability_from_accuracy(
        accuracy_all,
        target=float(hpo_config.self_predictability_target),
        width=float(hpo_config.self_predictability_width),
    )
    trend_vol_explainability = _clamp01((float(accuracy_trend_vol) - 0.5) / 0.35)
    replica_penalty = _clamp01(trend_vol_explainability * (1.0 - incremental_score))
    if accuracy_all <= 0.5 or not kept_all:
        score = 0.0
    else:
        score = _clamp01(0.60 * incremental_score + 0.30 * all_score + 0.10 * (1.0 - replica_penalty))
    return {
        "feature_conditional_learnability": score,
        "feature_conditional_accuracy_all": float(accuracy_all),
        "feature_conditional_accuracy_trend_vol": float(accuracy_trend_vol),
        "feature_conditional_incremental_accuracy": float(incremental),
        "feature_conditional_incremental_score": float(incremental_score),
        "feature_conditional_trend_vol_replica_penalty": float(replica_penalty),
        "feature_conditional_feature_count_all": float(len(kept_all)),
        "feature_conditional_feature_count_trend_vol": float(len(kept_trend_vol)),
    }


def _learnability_metrics(
    artifact: AdvancedRegimeLearningArtifact,
    method: str,
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    hpo_config: RegimeHPOConfig,
    random_state: int,
) -> dict[str, float]:
    self_score, self_acc = _regime_self_predictability(
        artifact,
        method,
        frame=frame,
        hpo_config=hpo_config,
        random_state=int(random_state) + 101,
    )
    soft = _soft_state_quality(artifact, method, frame=frame, hpo_config=hpo_config)
    support = _support_quality(artifact, method, hpo_config=hpo_config)
    compression = _compression_quality(artifact, method)
    specialist_score, specialist_acc = _specialist_feature_helpfulness(
        artifact,
        frame,
        feature_columns,
        hpo_config=hpo_config,
        random_state=int(random_state) + 202,
    )
    feature_conditional = _feature_conditional_learnability(
        artifact,
        method,
        frame,
        feature_columns,
        hpo_config=hpo_config,
        random_state=int(random_state) + 303,
    )
    components = [
        (float(hpo_config.self_predictability_weight), self_score),
        (float(hpo_config.soft_state_quality_weight), float(soft.get("soft_state_quality", 0.0))),
        (float(hpo_config.compression_quality_weight), float(compression.get("compression_quality", 0.0))),
        (float(hpo_config.support_quality_weight), float(support.get("support_quality", 0.0))),
        (float(hpo_config.specialist_feature_helpfulness_weight), specialist_score),
        (
            float(hpo_config.feature_conditional_learnability_weight),
            float(feature_conditional.get("feature_conditional_learnability", 0.0)),
        ),
    ]
    positive_weight = sum(max(weight, 0.0) for weight, _value in components)
    learnability = _clamp01(
        sum(max(weight, 0.0) * float(value) for weight, value in components)
        / max(positive_weight, 1e-12)
    )
    return {
        "learnability_score": learnability,
        "self_predictability": self_score,
        "self_predictability_balanced_accuracy": self_acc,
        "specialist_feature_helpfulness": specialist_score,
        "specialist_feature_helpfulness_balanced_accuracy": specialist_acc,
        **feature_conditional,
        **soft,
        **support,
        **compression,
    }


def _score_artifact(
    artifact: AdvancedRegimeLearningArtifact,
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    hpo_config: RegimeHPOConfig,
    runtime_seconds: float,
    random_state: int,
) -> dict[str, Any]:
    diagnostics = artifact.regime_diagnostics
    if not isinstance(diagnostics, pd.DataFrame) or diagnostics.empty:
        return {
            "status": "completed_no_methods",
            "hpo_score": float("-inf"),
            "runtime_seconds": float(runtime_seconds),
        }
    diag = diagnostics.copy(deep=False)
    if "TotalScore" not in diag.columns:
        diag = diag.assign(TotalScore=0.0)
    diag["TotalScore"] = pd.to_numeric(diag["TotalScore"], errors="coerce").fillna(0.0)
    candidate = diag
    if bool(hpo_config.prefer_non_baseline) and "is_baseline" in diag.columns:
        non_baseline = diag.loc[~diag["is_baseline"].astype(bool)]
        if not non_baseline.empty:
            candidate = non_baseline
    candidate = candidate.sort_values("TotalScore", ascending=False, kind="mergesort")
    top = candidate.iloc[0].to_dict()
    total_score = _float_from_row(top, "TotalScore")
    score_std = float(np.nanstd(pd.to_numeric(candidate["TotalScore"], errors="coerce").to_numpy(dtype=np.float64)))
    min_support = _float_from_row(top, "min_support")
    turnover = _float_from_row(top, "turnover")
    replica_penalty = _clamp01(_float_from_row(top, "TrendVolReplicaPenalty"))
    support_floor = max(float(hpo_config.min_support_floor), 1e-6)
    support_shortfall = _clamp01(max(support_floor - min_support, 0.0) / support_floor)
    turnover_target = float(np.clip(hpo_config.turnover_target, 0.0, 0.99))
    turnover_excess = _clamp01(max(turnover - turnover_target, 0.0) / max(1.0 - turnover_target, 1e-6))
    score_dispersion_penalty = _clamp01(score_std / 0.5)
    compute_penalty = _clamp01(np.log1p(max(float(runtime_seconds), 0.0)) / np.log1p(600.0))
    penalty_total = (
        float(hpo_config.score_variance_penalty_weight) * score_dispersion_penalty
        + float(hpo_config.low_support_penalty_weight) * support_shortfall
        + float(hpo_config.turnover_penalty_weight) * turnover_excess
        + float(hpo_config.trend_vol_replica_penalty_weight) * replica_penalty
        + float(hpo_config.compute_cost_penalty_weight) * compute_penalty
    )
    structure_score = (
        total_score
        - penalty_total
    )
    method_name = str(top.get("method", ""))
    learnability = _learnability_metrics(
        artifact,
        method_name,
        frame=frame,
        feature_columns=feature_columns,
        hpo_config=hpo_config,
        random_state=int(random_state),
    )
    learnability_score = float(learnability.get("learnability_score", 0.0)) - penalty_total
    mode = str(hpo_config.objective_mode).strip().lower()
    if mode == "structure":
        hpo_score = structure_score
    elif mode == "hybrid":
        sw = max(float(hpo_config.structure_objective_weight), 0.0)
        lw = max(float(hpo_config.learnability_objective_weight), 0.0)
        denom = max(sw + lw, 1e-12)
        hpo_score = (sw * structure_score + lw * learnability_score) / denom
    else:
        hpo_score = learnability_score
    artifact_diag = artifact.diagnostics if isinstance(artifact.diagnostics, Mapping) else {}
    return {
        "status": "completed",
        "hpo_score": float(hpo_score),
        "hpo_objective_mode": mode if mode in {"structure", "hybrid", "learnability"} else "learnability",
        "structure_score": float(structure_score),
        "learnability_hpo_score": float(learnability_score),
        "penalty_total": float(penalty_total),
        "top_method": method_name,
        "top_total_score": total_score,
        "top_nontriviality": _float_from_row(top, "NonTriviality"),
        "top_oos_stability": _float_from_row(top, "OOS_Stability"),
        "top_dwell_quality": _float_from_row(top, "Dwell_Quality"),
        "top_transition_stability": _float_from_row(top, "Transition_Stability"),
        "top_feature_stability": _float_from_row(top, "Feature_Stability"),
        "top_null_robustness": _float_from_row(top, "Null_Robustness"),
        "top_window_robustness": _float_from_row(top, "Window_Robustness"),
        "top_geometry_separation": _float_from_row(top, "Geometry_Separation"),
        "top_min_support": min_support,
        "top_turnover": turnover,
        "top_trend_vol_replica_penalty": replica_penalty,
        "score_std": score_std,
        "score_dispersion_penalty": float(score_dispersion_penalty),
        "support_shortfall": float(support_shortfall),
        "turnover_excess": float(turnover_excess),
        "compute_penalty": float(compute_penalty),
        "runtime_seconds": float(runtime_seconds),
        "assessed_method_count": int(len(diag)),
        "candidate_method_count": int(len(candidate)),
        "kept_method_count": int(len(artifact_diag.get("kept_methods", []))),
        "model_regime_feature_count": int(getattr(artifact, "model_regime_features", pd.DataFrame()).shape[1]),
        "model_regime_candidate_tier": str(artifact_diag.get("model_regime_candidate_tier", "")),
        "model_regime_package_meaningful": bool(artifact_diag.get("model_regime_package_meaningful", False)),
        **learnability,
    }


def _write_hpo_outputs(
    *,
    result: RegimeHPOResult,
    trial_configs: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    trials_path = output_dir / "regime_hpo_trials.csv"
    result.trials.to_csv(trials_path, index=False)
    paths["trials"] = str(trials_path)
    steps_path = output_dir / "regime_hpo_trial_steps.csv"
    result.trial_steps.to_csv(steps_path, index=False)
    paths["trial_steps"] = str(steps_path)
    if bool(result.hpo_config.store_feature_metrics):
        feature_metrics_path = output_dir / "regime_hpo_trial_model_feature_metrics.csv"
        result.trial_model_feature_metrics.to_csv(feature_metrics_path, index=False)
        paths["trial_model_feature_metrics"] = str(feature_metrics_path)
    configs_path = output_dir / "regime_hpo_trial_configs.json"
    configs_path.write_text(
        json.dumps(_json_ready(list(trial_configs)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["trial_configs"] = str(configs_path)
    if result.best_artifact is not None and bool(result.hpo_config.store_best_artifact):
        best_dir = output_dir / "best_advanced_regime_learning"
        saved = save_advanced_regime_learning_artifact(result.best_artifact, best_dir)
        paths.update({f"best_{key}": value for key, value in saved.items()})
    manifest = {
        "schema_version": ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
        "trial_count": int(len(result.trials)),
        "stop_reason": (
            str(result.trials["hpo_stop_reason"].iloc[0])
            if not result.trials.empty and "hpo_stop_reason" in result.trials.columns
            else None
        ),
        "hpo_input_rows": (
            int(result.trials["hpo_input_rows"].iloc[0])
            if not result.trials.empty and "hpo_input_rows" in result.trials.columns
            else None
        ),
        "hpo_sampled_rows": (
            int(result.trials["hpo_sampled_rows"].iloc[0])
            if not result.trials.empty and "hpo_sampled_rows" in result.trials.columns
            else None
        ),
        "hpo_sampling": (
            str(result.trials["hpo_sampling"].iloc[0])
            if not result.trials.empty and "hpo_sampling" in result.trials.columns
            else None
        ),
        "best_trial_id": (
            int(result.trials.sort_values("hpo_score", ascending=False, kind="mergesort")["trial_id"].iloc[0])
            if not result.trials.empty and "hpo_score" in result.trials.columns
            else None
        ),
        "best_trial_params": result.best_trial_params,
        "hpo_config": asdict(result.hpo_config),
        "paths": paths,
    }
    manifest_path = output_dir / "regime_hpo_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["manifest"] = str(manifest_path)
    return paths


def run_advanced_regime_learning_hpo(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    base_config: AdvancedRegimeLearningConfig = AdvancedRegimeLearningConfig(),
    hpo_config: RegimeHPOConfig = RegimeHPOConfig(),
) -> RegimeHPOResult:
    """Run unsupervised HPO over advanced regime-learning parameters."""

    trial_params = _sample_trial_params(
        hpo_config.search_space,
        max_trials=int(hpo_config.max_trials),
        random_state=int(hpo_config.random_state),
    )
    rows: list[dict[str, Any]] = []
    step_parts: list[pd.DataFrame] = []
    feature_metric_parts: list[pd.DataFrame] = []
    trial_configs: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_artifact: AdvancedRegimeLearningArtifact | None = None
    best_config: AdvancedRegimeLearningConfig | None = None
    best_params: dict[str, Any] = {}
    output_dir = Path(hpo_config.artifact_output_dir) if hpo_config.artifact_output_dir else None
    fit_frame, sample_info = _sample_hpo_frame(
        frame,
        base_config=base_config,
        hpo_config=hpo_config,
    )
    hpo_start = time.perf_counter()
    completed_scores: list[float] = []
    failed_trials = 0
    no_improvement_count = 0
    median_pruned_streak = 0
    stop_reason = ""

    for trial_id, params in enumerate(trial_params):
        elapsed_total = time.perf_counter() - hpo_start
        max_runtime = float(hpo_config.max_total_runtime_seconds or 0.0)
        if max_runtime > 0.0 and elapsed_total >= max_runtime:
            stop_reason = "max_total_runtime_seconds"
            break
        params = dict(params)
        trial_cfg = _bounded_trial_config(base_config, params, hpo_config)
        trial_configs.append(
            {
                "trial_id": int(trial_id),
                "params": params,
                "effective_config": asdict(trial_cfg),
            }
        )
        start = time.perf_counter()
        row: dict[str, Any] = {
            "trial_id": int(trial_id),
            "params_json": json.dumps(_json_ready(params), sort_keys=True),
            **sample_info,
            "hpo_elapsed_before_trial_seconds": float(elapsed_total),
            "median_pruner_enabled": bool(hpo_config.median_pruner_enabled),
        }
        try:
            artifact = fit_advanced_regime_learning(fit_frame, feature_columns, config=trial_cfg)
            runtime = time.perf_counter() - start
            row.update(
                _score_artifact(
                    artifact,
                    frame=fit_frame,
                    feature_columns=feature_columns,
                    hpo_config=hpo_config,
                    runtime_seconds=runtime,
                    random_state=int(hpo_config.random_state) + int(trial_id) * 1000,
                )
            )
            row.update({f"param_{key}": value for key, value in params.items()})
            prior_scores = [score for score in completed_scores if np.isfinite(score)]
            median_reference = (
                float(np.median(prior_scores))
                if bool(hpo_config.median_pruner_enabled)
                and len(prior_scores) >= int(hpo_config.median_pruner_warmup_trials)
                else float("nan")
            )
            score = float(row.get("hpo_score", float("-inf")))
            median_pruned = bool(
                np.isfinite(median_reference)
                and np.isfinite(score)
                and score < median_reference - float(hpo_config.median_pruner_min_delta)
            )
            row["median_pruner_reference"] = median_reference
            row["median_pruned"] = median_pruned
            if median_pruned:
                median_pruned_streak += 1
            else:
                median_pruned_streak = 0
            previous_best = best_score
            if output_dir is not None and bool(hpo_config.store_trial_artifacts):
                trial_dir = output_dir / f"trial_{trial_id:03d}"
                saved = save_advanced_regime_learning_artifact(artifact, trial_dir)
                row["artifact_path"] = saved.get("artifact")
                row["manifest_path"] = saved.get("manifest")
            steps = artifact.pipeline_steps.copy()
            if not steps.empty:
                steps.insert(0, "trial_id", int(trial_id))
                steps["hpo_score"] = float(row.get("hpo_score", np.nan))
                steps["top_method"] = str(row.get("top_method", ""))
                step_parts.append(steps)
            if bool(hpo_config.store_feature_metrics):
                metrics = artifact.model_regime_feature_metrics.copy()
                if not metrics.empty:
                    metrics.insert(0, "trial_id", int(trial_id))
                    metrics["hpo_score"] = float(row.get("hpo_score", np.nan))
                    feature_metric_parts.append(metrics)
            if score > best_score:
                best_score = score
                best_artifact = artifact
                best_config = trial_cfg
                best_params = dict(params)
            improvement = bool(np.isfinite(score) and score > previous_best + float(hpo_config.early_stopping_min_delta))
            if improvement:
                no_improvement_count = 0
            elif np.isfinite(score):
                no_improvement_count += 1
            completed_scores.append(score)
            row["best_score_after_trial"] = float(best_score)
            row["no_improvement_count"] = int(no_improvement_count)
            row["median_pruned_streak"] = int(median_pruned_streak)
        except Exception as exc:
            failed_trials += 1
            row.update(
                {
                    "status": "failed",
                    "hpo_score": float("-inf"),
                    "runtime_seconds": float(time.perf_counter() - start),
                    "error": f"{type(exc).__name__}: {exc}",
                    "median_pruner_reference": float("nan"),
                    "median_pruned": False,
                    "best_score_after_trial": float(best_score),
                    "no_improvement_count": int(no_improvement_count),
                    "median_pruned_streak": int(median_pruned_streak),
                }
            )
        rows.append(row)
        completed = len([r for r in rows if str(r.get("status", "")).startswith("completed")])
        if (
            int(hpo_config.max_failed_trials or 0) > 0
            and failed_trials >= int(hpo_config.max_failed_trials)
        ):
            stop_reason = "max_failed_trials"
            break
        if (
            int(hpo_config.early_stopping_patience or 0) > 0
            and completed >= int(hpo_config.early_stopping_min_trials)
            and no_improvement_count >= int(hpo_config.early_stopping_patience)
        ):
            stop_reason = "early_stopping_patience"
            break
        if (
            bool(hpo_config.median_pruner_enabled)
            and int(hpo_config.median_pruner_stop_after_pruned_streak or 0) > 0
            and median_pruned_streak >= int(hpo_config.median_pruner_stop_after_pruned_streak)
        ):
            stop_reason = "median_pruned_streak"
            break

    trials = pd.DataFrame(rows)
    if not trials.empty:
        trials["hpo_stop_reason"] = stop_reason or "completed_max_trials"
        trials["hpo_total_runtime_seconds"] = float(time.perf_counter() - hpo_start)
    if not trials.empty and "hpo_score" in trials.columns:
        trials["hpo_score"] = pd.to_numeric(trials["hpo_score"], errors="coerce")
        trials = trials.sort_values("hpo_score", ascending=False, kind="mergesort").reset_index(drop=True)
    trial_steps = pd.concat(step_parts, ignore_index=True) if step_parts else pd.DataFrame()
    trial_model_feature_metrics = (
        pd.concat(feature_metric_parts, ignore_index=True) if feature_metric_parts else pd.DataFrame()
    )
    result = RegimeHPOResult(
        trials=trials,
        trial_steps=trial_steps,
        trial_model_feature_metrics=trial_model_feature_metrics,
        best_config=best_config,
        best_artifact=best_artifact,
        best_trial_params=best_params,
        hpo_config=hpo_config,
        output_paths={},
    )
    if output_dir is not None:
        paths = _write_hpo_outputs(
            result=result,
            trial_configs=trial_configs,
            output_dir=output_dir,
        )
        result = replace(result, output_paths=paths)
    return result
