#!/usr/bin/env python3
"""Strict-OOF causal-regime × path-archetype supportive-label ablation.

This is the remaining Stage-3/4 section of the long-only supportive-label
funnel.  It keeps the previously materialised H12 path representation strictly
on the target side and tests whether an *observable* market-state mixture adds
economic information beyond the frozen Strict-R3 upstream score.

For every chronological outer fold this script:

1. fits a market/context-only state ontology on historical decision-time rows;
2. fits an H12 realised-path GMM only on resolved historical path labels;
3. trains causal path-membership classifiers under X, X+base, and X+stack;
4. estimates shrinkage-stabilised E[policy net | causal state, realised path]
   from training outcomes only; and
5. scores held rows with hard (J1) and soft (J2) causal×path mappings.

No current or held future path coordinate is an inference feature.  The
resulting output is research-only and has no authority over live admission,
MC1, the canonical score, sizing, or execution.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from run_strict_r3_long_supportive_label_funnel import (
    DEFAULT_LABELS,
    DEFAULT_LEDGER,
    EMBARGO,
    FOLDS,
    IDENTITY,
    MAX_CLUSTER_ROWS,
    MAX_TRAIN_ROWS,
    MIN_CLASS_ROWS,
    P1_FUTURE_FIELDS,
    SEED,
    STACK_FIELDS,
    _align_probabilities,
    _class_policy_priors,
    _score_eligible,
    _train_path_eligible,
    _finite,
    _fit_gmm,
    _fit_p1_transform,
    _joined_population,
    _ledger_fields,
    _matrix,
    _model_classifier,
    _quality_metrics,
    _sample_month_balanced,
    _sha256,
    _transform_p1,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_long_supportive_label_causal_joint_v1"
SIDE = "long"
HISTORY_START = pd.Timestamp("2024-01-01T00:00:00Z")
PATH_K = 8
REGIME_SPECS = (
    ("C1_ward_k4", "ward", 4),
    ("C2_gmm_k4", "gmm", 4),
    ("C2_gmm_k6", "gmm", 6),
)
PATH_FEATURE_MODES = ("causal120", "causal120_plus_base", "causal120_plus_oof_stack")
MAX_WARD_FIT_ROWS = 8_000
MAX_STATE_FIT_ROWS = 60_000
MIN_STATE_TIMESTAMPS = 1_000
JOINT_PRIOR_ROWS = 500


# The state view intentionally excludes candidate-local quantile / peer fields.
# Each selected field describes the contemporaneous broad market, breadth,
# leverage/flow, correlation/dependence, or shared volatility/liquidity state.
MARKET_PREFIXES = (
    "mkt_", "pct_assets_", "market_", "breadth_", "negative_breadth_",
    "median_alt_", "state_spectral_", "cross_asset_", "liquidation_",
    "post_liquidation_", "post_flush_", "prior_volatility", "price_rv_",
    "eig_effective_rank_",
)
BASE_AUX_FIELDS = (
    "prequential_p_adverse", "prequential_p_weak", "prequential_p_clear",
    "prequential_base_score", "prequential_base_rank42",
)


@dataclass(frozen=True)
class RegimeGeometry:
    name: str
    family: str
    k: int
    fields: tuple[str, ...]
    medians: pd.Series
    lower: np.ndarray
    upper: np.ndarray
    scaler: RobustScaler
    pca: PCA
    model: Any
    prototypes: np.ndarray | None
    temperature: float | None


def _market_fields(source_fields: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(
        field for field in source_fields
        if field == "prior_volatility" or field.startswith(MARKET_PREFIXES)
    )
    if len(selected) < 20:
        raise AssertionError(f"market-state view unexpectedly narrow: {len(selected)}")
    return selected


def _month_balanced_timestamps(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    """Deterministically sample timestamp panels, never outcome-ranked rows."""
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__month__"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    chunks: list[pd.DataFrame] = []
    groups = list(work.groupby("__month__", observed=True, sort=True))
    quota = max(1, cap // len(groups))
    for ordinal, (_, group) in enumerate(groups):
        if len(group) <= quota:
            chunks.append(group)
            continue
        key = pd.util.hash_pandas_object(group["__decision_ts__"], index=False).to_numpy(np.uint64)
        take = np.argsort(key ^ np.uint64(seed + ordinal), kind="stable")[:quota]
        chunks.append(group.iloc[take])
    out = pd.concat(chunks, ignore_index=True)
    if len(out) < cap:
        remaining = work.loc[~work["__decision_ts__"].isin(out["__decision_ts__"])]
        key = pd.util.hash_pandas_object(remaining["__decision_ts__"], index=False).to_numpy(np.uint64)
        out = pd.concat([out, remaining.iloc[np.argsort(key, kind="stable")[: cap - len(out)]]], ignore_index=True)
    return out.drop(columns="__month__", errors="ignore")


def _state_panel(raw: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """One observable market-state row per decision timestamp.

    Cross-sectional fields are already point-in-time values.  The median is
    only a deterministic projection from the contemporaneous candidate panel;
    it does not inspect validity or any later path/outcome value.
    """
    values = raw.loc[:, ["__decision_ts__", *fields]].copy()
    for field in fields:
        values[field] = _finite(values[field])
    state = values.groupby("__decision_ts__", sort=True, observed=True)[list(fields)].median().reset_index()
    if state["__decision_ts__"].duplicated().any():
        raise AssertionError("market state panel has duplicate timestamps")
    return state


def _state_matrix(frame: pd.DataFrame, fields: Sequence[str], *, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    return _matrix(frame, fields, medians=medians)


def _fit_state_transform(state_train: pd.DataFrame, fields: Sequence[str]) -> tuple[pd.Series, np.ndarray, np.ndarray, RobustScaler, PCA, np.ndarray]:
    sample = _month_balanced_timestamps(state_train, MAX_STATE_FIT_ROWS, seed=SEED)
    matrix, medians = _state_matrix(sample, fields)
    lower = np.nanquantile(matrix, 0.005, axis=0).astype(np.float32)
    upper = np.nanquantile(matrix, 0.995, axis=0).astype(np.float32)
    clipped = np.clip(matrix, lower, upper)
    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    scaled = scaler.fit_transform(clipped)
    components = min(12, scaled.shape[1], max(2, scaled.shape[0] - 1))
    pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED)
    return medians, lower, upper, scaler, pca, pca.fit_transform(scaled).astype(np.float32)


def _transform_state(frame: pd.DataFrame, geometry: RegimeGeometry) -> np.ndarray:
    matrix, _ = _state_matrix(frame, geometry.fields, medians=geometry.medians)
    matrix = np.clip(matrix, geometry.lower, geometry.upper)
    return geometry.pca.transform(geometry.scaler.transform(matrix)).astype(np.float32)


def _ward_probability(values: np.ndarray, prototypes: np.ndarray, temperature: float) -> np.ndarray:
    distance = ((values[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return softmax(-distance / max(float(temperature), 1e-6), axis=1).astype(np.float32)


def _fit_regime_geometry(name: str, family: str, k: int, state_train: pd.DataFrame, fields: Sequence[str]) -> tuple[RegimeGeometry, np.ndarray, dict[str, Any]]:
    if len(state_train) < MIN_STATE_TIMESTAMPS:
        raise RuntimeError(f"{name}: insufficient train timestamps: {len(state_train)}")
    medians, lower, upper, scaler, pca, sample_latent = _fit_state_transform(state_train, fields)
    raw = _state_matrix(state_train, fields, medians=medians)[0]
    latent = pca.transform(scaler.transform(np.clip(raw, lower, upper))).astype(np.float32)
    if family == "gmm":
        model = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-4,
            random_state=SEED + k, max_iter=200, n_init=2,
        ).fit(sample_latent)
        q_train = model.predict_proba(latent).astype(np.float32)
        prototypes: np.ndarray | None = None
        temperature: float | None = None
    elif family == "ward":
        ward_fit = sample_latent
        if len(ward_fit) > MAX_WARD_FIT_ROWS:
            idx = np.linspace(0, len(ward_fit) - 1, MAX_WARD_FIT_ROWS, dtype=int)
            ward_fit = ward_fit[idx]
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(ward_fit)
        prototypes = np.vstack([np.median(ward_fit[labels == i], axis=0) for i in range(k)]).astype(np.float32)
        nearest = ((ward_fit[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        temperature = float(max(np.median(nearest), 1e-4))
        q_train = _ward_probability(latent, prototypes, temperature)
        model = None
    else:
        raise ValueError(family)
    geometry = RegimeGeometry(
        name=name, family=family, k=k, fields=tuple(fields), medians=medians,
        lower=lower, upper=upper, scaler=scaler, pca=pca, model=model,
        prototypes=prototypes, temperature=temperature,
    )
    hard = q_train.argmax(axis=1)
    # Structural diagnostics are explicitly bounded.  They do not change the
    # fitted state ontology or any held score, and a 2,000-state probe is ample
    # for a relative Ward/GMM quality screen without dominating the research
    # runtime through quadratic pairwise distances.
    sample_eval = latent[np.linspace(0, len(latent) - 1, min(2_000, len(latent)), dtype=int)]
    sample_labels = hard[np.linspace(0, len(hard) - 1, min(2_000, len(hard)), dtype=int)]
    metrics: dict[str, Any] = {
        "regime_arm": name, "family": family, "k": k,
        "train_timestamps": int(len(state_train)),
        "tiny_clusters": int(sum((hard == i).sum() < 100 for i in range(k))),
        "silhouette": float(silhouette_score(sample_eval, sample_labels)) if len(np.unique(sample_labels)) > 1 else np.nan,
        "davies_bouldin": float(davies_bouldin_score(sample_eval, sample_labels)) if len(np.unique(sample_labels)) > 1 else np.nan,
    }
    # A small, train-only bootstrap stability diagnostic.  ARI is invariant to
    # component permutation, so it is appropriate for both Ward and GMM.
    anchor_idx = np.linspace(0, len(sample_latent) - 1, min(2_000, len(sample_latent)), dtype=int)
    anchor = sample_latent[anchor_idx]
    if family == "gmm":
        primary = model.predict(anchor)
    else:
        assert prototypes is not None and temperature is not None
        primary = _ward_probability(anchor, prototypes, temperature).argmax(axis=1)
    aris: list[float] = []
    rng = np.random.default_rng(SEED + k + (0 if family == "ward" else 100))
    for replica in range(2):
        take = rng.integers(0, len(sample_latent), size=min(len(sample_latent), 3_000))
        boot = sample_latent[take]
        if family == "gmm":
            alternative = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, random_state=SEED + replica + 301, max_iter=150, n_init=1).fit(boot)
            alternate_labels = alternative.predict(anchor)
        else:
            alt_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(boot)
            alt_proto = np.vstack([np.median(boot[alt_labels == i], axis=0) for i in range(k)])
            alternate_labels = ((anchor[:, None, :] - alt_proto[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        aris.append(float(adjusted_rand_score(primary, alternate_labels)))
    metrics["bootstrap_ari_mean"] = float(np.mean(aris))
    metrics["bootstrap_ari_min"] = float(np.min(aris))
    return geometry, q_train, metrics


def _regime_probability(state: pd.DataFrame, geometry: RegimeGeometry) -> np.ndarray:
    latent = _transform_state(state, geometry)
    if geometry.family == "gmm":
        return geometry.model.predict_proba(latent).astype(np.float32)
    assert geometry.prototypes is not None and geometry.temperature is not None
    return _ward_probability(latent, geometry.prototypes, geometry.temperature)


def _fit_path_ontology(train: pd.DataFrame) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray]:
    scaler, pca, medians, sample_pca, lower, upper = _fit_p1_transform(train, P1_FUTURE_FIELDS)
    gmm, _, _ = _fit_gmm(sample_pca, k=PATH_K)
    train_latent = _transform_p1(train, P1_FUTURE_FIELDS, scaler, pca, medians, lower, upper)
    raw_probability = gmm.predict_proba(train_latent).astype(np.float32)
    raw_labels = raw_probability.argmax(axis=1)
    raw_priors = _class_policy_priors(train, raw_labels, k=PATH_K)
    order = np.argsort(raw_priors, kind="stable")
    inverse = np.empty(PATH_K, dtype=np.int16)
    inverse[order] = np.arange(PATH_K, dtype=np.int16)
    labels = inverse[raw_labels]
    probability = raw_probability[:, order]
    priors = raw_priors[order]
    return scaler, pca, lower, upper, medians, priors, labels.astype(np.int16), probability


def _path_classifier(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], labels: np.ndarray, *, seed: int) -> np.ndarray:
    x_train, medians = _matrix(train, fields)
    x_held, _ = _matrix(held, fields, medians=medians)
    model = _model_classifier(classes=PATH_K, seed=seed)
    model.fit(x_train, labels)
    return _align_probabilities(model, model.predict_proba(x_held), k=PATH_K)


def _shrunken_joint_map(y: np.ndarray, q: np.ndarray, path_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stable train-only global, regime, path and joint expected EV.

    Joint cells shrink towards the mean of their corresponding regime and path
    parents.  This avoids sparse C×A cells manufacturing admissions.
    """
    resolved = np.isfinite(y)
    y = y[resolved]
    q = q[resolved]
    path_labels = path_labels[resolved]
    if len(y) < 1_000:
        raise AssertionError("causal×path value map lacks resolved train-only policy support")
    global_mean = float(np.mean(y))
    regime_hard = q.argmax(axis=1)
    k_regime = q.shape[1]
    regime = np.full(k_regime, global_mean, dtype=np.float64)
    path = np.full(PATH_K, global_mean, dtype=np.float64)
    for j in range(k_regime):
        local = y[regime_hard == j]
        regime[j] = (local.sum() + JOINT_PRIOR_ROWS * global_mean) / (len(local) + JOINT_PRIOR_ROWS)
    for a in range(PATH_K):
        local = y[path_labels == a]
        path[a] = (local.sum() + JOINT_PRIOR_ROWS * global_mean) / (len(local) + JOINT_PRIOR_ROWS)
    joint = np.empty((k_regime, PATH_K), dtype=np.float64)
    support = np.zeros((k_regime, PATH_K), dtype=np.int32)
    for j in range(k_regime):
        for a in range(PATH_K):
            local = y[(regime_hard == j) & (path_labels == a)]
            support[j, a] = len(local)
            parent = 0.5 * (regime[j] + path[a])
            joint[j, a] = (local.sum() + JOINT_PRIOR_ROWS * parent) / (len(local) + JOINT_PRIOR_ROWS)
    return np.asarray([global_mean]), regime, path, joint, support


def _score_joint(q: np.ndarray, p: np.ndarray, joint: np.ndarray, *, hard: bool) -> np.ndarray:
    if hard:
        return joint[q.argmax(axis=1), p.argmax(axis=1)].astype(np.float32)
    return np.einsum("ij,ik,jk->i", q, p, joint, optimize=True).astype(np.float32)


def _write_metrics(metrics: list[dict[str, Any]], out: Path) -> None:
    pd.DataFrame(metrics).to_parquet(out / "causal_joint_metrics.parquet", index=False, compression="zstd")


def _state_to_candidate(frame: pd.DataFrame, state: pd.DataFrame, probability: np.ndarray, *, prefix: str) -> pd.DataFrame:
    q = pd.DataFrame(probability, columns=[f"{prefix}_{i:02d}" for i in range(probability.shape[1])])
    q.insert(0, "__decision_ts__", state["__decision_ts__"].to_numpy())
    result = frame[["candidate_id", "__decision_ts__"]].merge(q, on="__decision_ts__", how="left", validate="many_to_one")
    if result.filter(like=f"{prefix}_").isna().any(axis=None):
        raise AssertionError("candidate rows missing an observable regime state")
    return result


def _fold(
    *, fold_index: int, ledger: Path, labels_root: Path, source_fields: Sequence[str], market_fields: Sequence[str], max_train_rows: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    fold = FOLDS[fold_index]
    train_raw = _joined_population(ledger, labels_root, start=HISTORY_START, end=fold.start, fields=source_fields, p1_fields=P1_FUTURE_FIELDS)
    held_raw = _joined_population(ledger, labels_root, start=fold.start, end=fold.end, fields=source_fields, p1_fields=P1_FUTURE_FIELDS)
    # Path archetype targets use the complete observed H12 population.  The
    # state/value maps retain only finite policy outcomes inside their
    # train-only mapping kernel; held candidates are scored target-free.
    train = _train_path_eligible(train_raw, cutoff=fold.start)
    held = _score_eligible(held_raw)
    if len(train) < 20_000 or len(held) < 5_000:
        raise RuntimeError(f"{fold.name}: insufficient supervised support train={len(train)}, held={len(held)}")
    train = _sample_month_balanced(train, max_train_rows, seed=SEED + fold_index)
    state_train = _state_panel(train_raw, market_fields)
    state_held = _state_panel(held_raw, market_fields)
    del train_raw, held_raw
    gc.collect()

    path_scaler, path_pca, path_lower, path_upper, path_medians, path_priors, path_labels, _ = _fit_path_ontology(train)
    # The path ontology is target-side and fold-local.  We never concatenate
    # its component IDs across folds; all downstream outputs are invariant bps
    # maps or probabilities used within their originating fold only.
    path_held_latent = _transform_p1(held, P1_FUTURE_FIELDS, path_scaler, path_pca, path_medians, path_lower, path_upper)
    # Recover held gold only for model-quality metrics, never as an input.
    # The ontology GMM itself is represented by a reproducible transform in the
    # stored fold audit instead of a globally meaningful component identity.
    sample_latent = _transform_p1(train, P1_FUTURE_FIELDS, path_scaler, path_pca, path_medians, path_lower, path_upper)
    # Train a GMM anew from the exact transform/sample convention.  This call is
    # deterministic and is separate from the classifier; no held values fit it.
    # The GMM fit subset is the transform helper's train-only monthly sample.
    # To avoid relying on raw held labels for any inference score, it is used
    # only after all classifier outputs have been constructed.
    _, _, _, p1_sample, _, _ = _fit_p1_transform(train, P1_FUTURE_FIELDS)
    path_gmm, _, _ = _fit_gmm(p1_sample, k=PATH_K)
    raw_priors = _class_policy_priors(train, path_gmm.predict(sample_latent), k=PATH_K)
    order = np.argsort(raw_priors, kind="stable")
    inverse = np.empty(PATH_K, dtype=np.int16); inverse[order] = np.arange(PATH_K, dtype=np.int16)
    gold_path_held = inverse[path_gmm.predict(path_held_latent)].astype(np.int16)

    metrics: list[dict[str, Any]] = []
    state_audit: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    wide = held.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "prequential_upstream", "policy_net_bps"]].copy()
    wide["B0_prequential_upstream"] = _finite(held["prequential_upstream"]).to_numpy(np.float32)
    metrics.extend(_quality_metrics(fold=fold, arm="B0_prequential_upstream", feature_mode="frozen_stack", score=wide["B0_prequential_upstream"].to_numpy(float), held=held))

    feature_sets = {
        "causal120": tuple(source_fields),
        "causal120_plus_base": tuple((*source_fields, *BASE_AUX_FIELDS)),
        "causal120_plus_oof_stack": tuple((*source_fields, *STACK_FIELDS)),
    }
    path_probabilities: dict[str, np.ndarray] = {}
    for mode, fields in feature_sets.items():
        probability = _path_classifier(train, held, fields, path_labels, seed=SEED + fold_index * 101 + len(path_probabilities) * 17)
        path_probabilities[mode] = probability
        # A path-only expected-EV control on the same fold-local ontology.
        score = probability @ path_priors
        arm = f"P3_path_gmm_k{PATH_K}_{mode}"
        metrics.extend(_quality_metrics(fold=fold, arm=arm, feature_mode=mode, score=score, held=held, label=gold_path_held, probability=probability))
        wide[arm] = score.astype(np.float32)
        # These are causal classifier outputs, not realised archetype IDs.
        # They permit a later consensus learner to use the predicted path
        # geometry without changing this base-layer experiment's score.
        wide[f"{arm}__entropy"] = (-np.clip(probability, 1e-9, 1.0) * np.log(np.clip(probability, 1e-9, 1.0))).sum(axis=1).astype(np.float32)
        for component in range(PATH_K):
            wide[f"{arm}__p_{component:02d}"] = probability[:, component].astype(np.float32)

    for ordinal, (regime_name, family, k) in enumerate(REGIME_SPECS):
        geometry, q_train_state, structural = _fit_regime_geometry(regime_name, family, k, state_train, market_fields)
        q_held_state = _regime_probability(state_held, geometry)
        structural.update({"fold": fold.name, "cohort": fold.cohort, "state_fields": len(market_fields)})
        state_audit.append(structural)
        train_q_frame = _state_to_candidate(train, state_train, q_train_state, prefix="q")
        held_q_frame = _state_to_candidate(held, state_held, q_held_state, prefix="q")
        q_cols = [column for column in held_q_frame if column.startswith("q_")]
        q_train = train_q_frame.loc[:, q_cols].to_numpy(np.float32)
        q_held = held_q_frame.loc[:, q_cols].to_numpy(np.float32)
        for component in range(k):
            wide[f"{regime_name}__q_{component:02d}"] = q_held[:, component].astype(np.float32)
        y_train = _finite(train["policy_net_bps"]).to_numpy(float)
        _, regime_mu, path_mu, joint_mu, support = _shrunken_joint_map(y_train, q_train, path_labels)
        for j in range(k):
            for a in range(PATH_K):
                map_rows.append({
                    "fold": fold.name, "cohort": fold.cohort, "regime_arm": regime_name,
                    "regime": j, "path_component": a, "train_support": int(support[j, a]),
                    "train_joint_expected_policy_net_bps": float(joint_mu[j, a]),
                    "train_regime_expected_policy_net_bps": float(regime_mu[j]),
                    "train_path_expected_policy_net_bps": float(path_mu[a]),
                })
        regime_score = q_held @ regime_mu
        arm = f"{regime_name}_state_expected_ev"
        metrics.extend(_quality_metrics(fold=fold, arm=arm, feature_mode="market_context_only", score=regime_score, held=held))
        wide[arm] = regime_score.astype(np.float32)
        for mode, probability in path_probabilities.items():
            hard = _score_joint(q_held, probability, joint_mu, hard=True)
            soft = _score_joint(q_held, probability, joint_mu, hard=False)
            for kind, score in (("J1_hard", hard), ("J2_soft", soft), ("J2_soft_base_equal", 0.5 * soft + 0.5 * wide["B0_prequential_upstream"].to_numpy(float))):
                joint_arm = f"{regime_name}_{kind}_{mode}"
                metrics.extend(_quality_metrics(fold=fold, arm=joint_arm, feature_mode=mode, score=np.asarray(score), held=held))
                wide[joint_arm] = np.asarray(score, dtype=np.float32)
    fold_audit = [{
        "fold": fold.name, "cohort": fold.cohort, "status": "ok", "train_rows": int(len(train)), "held_rows": int(len(held)),
        "train_label_cutoff": str(fold.start), "embargo_hours": 12,
        "state_train_timestamps": int(len(state_train)), "state_held_timestamps": int(len(state_held)),
        "market_context_fields": list(market_fields), "path_k": PATH_K,
        "path_ontology": "fold-local target-only GMM; raw component identities never pooled downstream",
    }]
    return metrics, wide, [pd.DataFrame(state_audit), pd.DataFrame(map_rows)], fold_audit, []


def run(*, ledger: Path, labels_root: Path, out: Path, max_train_rows: int = MAX_TRAIN_ROWS) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    source_fields = _ledger_fields(ledger)
    market_fields = _market_fields(source_fields)
    all_metrics: list[dict[str, Any]] = []
    structural_frames: list[pd.DataFrame] = []
    map_frames: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    prediction_root = out / "causal_joint_oof_predictions"
    prediction_root.mkdir(parents=True, exist_ok=False)
    for fold_index, fold in enumerate(FOLDS):
        metrics, wide, extras, audit, _ = _fold(
            fold_index=fold_index, ledger=ledger, labels_root=labels_root,
            source_fields=source_fields, market_fields=market_fields, max_train_rows=max_train_rows,
        )
        all_metrics.extend(metrics)
        structural_frames.append(extras[0]); map_frames.append(extras[1]); fold_audit.extend(audit)
        target = prediction_root / f"fold={fold_index:02d}_{fold.name}.parquet"
        wide.to_parquet(target, index=False, compression="zstd")
        print(json.dumps(audit[0], sort_keys=True, default=str), flush=True)
        del wide, extras
        gc.collect()
    metric_frame = pd.DataFrame(all_metrics)
    _write_metrics(all_metrics, out)
    summary_metrics = metric_frame.loc[metric_frame["metric"].isin(("top_1%_net_ev_bps", "top_2%_net_ev_bps", "top_5%_net_ev_bps", "global_score_policy_residual_spearman"))]
    summary = summary_metrics.groupby(["arm", "feature_mode", "cohort", "metric"], as_index=False).agg(
        mean_value=("value", "mean"), median_value=("value", "median"), worst_value=("value", "min"), folds=("fold", "nunique")
    )
    summary.to_parquet(out / "causal_joint_summary.parquet", index=False, compression="zstd")
    pd.concat(structural_frames, ignore_index=True).to_parquet(out / "causal_regime_structural_audit.parquet", index=False, compression="zstd")
    pd.concat(map_frames, ignore_index=True).to_parquet(out / "joint_training_only_maps.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_audit).to_parquet(out / "causal_joint_fold_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline long-only research only; no live/canonical/admission/execution mutation",
        "ledger": str(ledger.resolve()), "ledger_sha256": _sha256(ledger),
        "labels_root": str(labels_root.resolve()), "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "outer_folds": [{"name": f.name, "start": str(f.start), "end_exclusive": str(f.end), "cohort": f.cohort} for f in FOLDS],
        "label_embargo": "all supervised path/policy labels available strictly before outer fold start minus 12 hours",
        "causal_feature_contract": list(source_fields),
        "market_context_only_state_fields": list(market_fields),
        "regime_specs": [{"name": n, "family": family, "k": k} for n, family, k in REGIME_SPECS],
        "path_ontology": "PCA/RobustScaler/GMM K8 fit only on each outer training fold's target-only H12 paths; components ordered by train-only shrunk policy prior",
        "path_predictor_modes": list(PATH_FEATURE_MODES),
        "joint_maps": "hard and soft C×A policy expected-EV maps fit only on training policy labels; 500-row hierarchical shrinkage toward train-only regime/path parents",
        "prediction_outputs": "candidate identity, existing upstream score, target-free causal-joint scores, and realised policy net for evaluation only; no raw future path coordinate is persisted as an inference input",
        "deferred": [
            {"family": "C3_HDBSCAN", "status": "not_run", "reason": "optional hdbscan dependency unavailable; Stage-4 expansion is gated on C1/C2/J1/J2 economics"},
            {"family": "J3_J4_J5_J6", "status": "not_run", "reason": "sequential funnel: only run richer hierarchy/early-fusion variants if J2 clears the direct-label challenger gate"},
        ],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS)
    args = parser.parse_args()
    result = run(ledger=args.ledger.resolve(), labels_root=args.labels_root.resolve(), out=args.out.resolve(), max_train_rows=args.max_train_rows)
    print(json.dumps({"status": "ok", "out": str(result)}))


if __name__ == "__main__":
    main()
