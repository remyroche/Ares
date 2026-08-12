#!/usr/bin/env python3
"""Build a higher-coverage, recurrent prototype/cluster contract for 2025.

The previous structural contract was intentionally conservative: it promoted
only *exactly recurring* leaf paths.  That kept leaf semantics pure, but left
roughly half of contribution mass unmatched when a monthly model refit moved a
threshold or added a nearby predicate.  This runner keeps the same causal
discipline while making the representation less brittle:

* fit leaf-path prototypes on 2024 models only;
* match 2025 leaves to those fixed prototypes using TF-IDF path neighbourhoods;
* retain match confidence/unmatched mass explicitly;
* discover 3--10 clusters from opportunity-conditioned co-activation and
  train-only residual synergy on 2024 only;
* select by structural support, balance, recurrence and validation sign
  stability -- never by 2025 outcomes.

It is a representation-quality run, not a downstream-model HPO.  Its saved
row features are the common input surface for the later residual/shrinkage
ablations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.cofiring_economic_clusters import (  # noqa: E402
    CoFiringClusterContract,
    materialize_memberships,
    pairwise_cofiring_similarity,
)
from scripts.run_tp6_sl4_frozen_cluster_residual import _load  # noqa: E402


SIDE = "long"
SEED = 20260809
PROTOTYPE_COUNTS = (12, 16, 20, 24, 32, 40)
CLUSTER_COUNTS = tuple(range(3, 11))
TOP_N = 3
MATCH_TEMPERATURE = 0.12
MATCH_THRESHOLD = 0.30
ACTIVE_THRESHOLD = 0.05
MIN_ROW_MATCHED_MASS = 0.80
DEFAULT_BASE = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
DEFAULT_FAMILY = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/meta_family_contribution_matrix.parquet"
DEFAULT_META = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
DEFAULT_RAW = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/strict_base_reasoning"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_quality_20260809_v1"


def _month(value: object) -> str:
    return str(value)[:7]


def _path_tokens(value: object, contribution: float) -> str:
    """Encode a path as coarse, position-aware causal predicate tokens.

    Exact threshold-band tokens are deliberately retained, but each predicate
    also emits a coarser feature/branch token.  This lets a refitted tree map
    a nearby threshold to the same prototype without hiding the resulting
    uncertainty in a hard family ID.
    """

    try:
        path = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        path = []
    tokens: list[str] = ["sign_pos" if float(contribution) >= 0.0 else "sign_neg"]
    for depth, item in enumerate(path or []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature", "unknown"))
        branch = str(item.get("branch", "unknown"))
        raw_band = int(item.get("threshold_band_index", -1))
        raw_count = int(item.get("threshold_band_count", -1))
        if raw_band >= 0 and raw_count > 1:
            coarse_band = min(3, max(0, int(math.floor(4.0 * raw_band / raw_count))))
            band_token = f"band_{coarse_band}"
            exact_band = f"band_{raw_band}_of_{raw_count}"
        else:
            band_token = "band_unknown"
            exact_band = "band_unknown"
        # The first three token types make nearby paths comparable; the last
        # two preserve ordered structural meaning and reduce false matches.
        tokens.extend((
            f"f_{feature}",
            f"fb_{feature}_{branch}",
            f"fbc_{feature}_{branch}_{band_token}",
            f"p{depth}_{feature}_{branch}",
            f"p{depth}_{feature}_{branch}_{exact_band}",
        ))
    return " ".join(tokens)


def _load_raw_month(root: Path, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = root / f"month={month}"
    catalog = pd.read_parquet(folder / "leaf_rule_catalog.parquet").copy()
    leaves = pd.read_parquet(folder / "leaf_assignments.parquet").copy()
    catalog["fold_id"] = catalog["fold_id"].astype(str)
    return catalog, leaves


def _fit_prototypes(catalog: pd.DataFrame, count: int) -> tuple[TfidfVectorizer, KMeans, np.ndarray, np.ndarray]:
    docs = [
        _path_tokens(path, contribution)
        for path, contribution in zip(
            catalog["rule_structural_path_json"].tolist(),
            pd.to_numeric(catalog["ensemble_tree_contribution"], errors="coerce").fillna(0.0).tolist(),
            strict=True,
        )
    ]
    vectorizer = TfidfVectorizer(analyzer=str.split, lowercase=False, norm="l2", sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    model = KMeans(n_clusters=int(count), n_init=20, random_state=SEED + int(count), algorithm="lloyd")
    labels = model.fit_predict(matrix)
    centroids = np.asarray(model.cluster_centers_, dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = np.divide(centroids, np.maximum(norms, 1e-12), out=np.zeros_like(centroids), where=norms > 0)
    return vectorizer, model, labels.astype(np.int16), centroids


def _match_catalog(
    catalog: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    centroids: np.ndarray,
    *,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    docs = [
        _path_tokens(path, contribution)
        for path, contribution in zip(
            catalog["rule_structural_path_json"].tolist(),
            pd.to_numeric(catalog["ensemble_tree_contribution"], errors="coerce").fillna(0.0).tolist(),
            strict=True,
        )
    ]
    matrix = vectorizer.transform(docs)
    similarity = np.asarray(matrix @ centroids.T, dtype=np.float32)
    order = np.argsort(-similarity, axis=1, kind="stable")[:, : min(TOP_N, centroids.shape[0])]
    top_similarity = np.take_along_axis(similarity, order, axis=1)
    best = top_similarity[:, 0]
    unmatched = np.where(best >= float(threshold), 0.0, 1.0 - best / max(float(threshold), 1e-6))
    logits = (top_similarity - top_similarity[:, [0]]) / MATCH_TEMPERATURE
    probabilities = np.exp(np.clip(logits, -60.0, 0.0))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    probabilities *= (1.0 - unmatched)[:, None]
    gap = top_similarity[:, 0] - (top_similarity[:, 1] if top_similarity.shape[1] > 1 else 0.0)
    match = catalog[["fold_id", "tree_index", "leaf_token", "ensemble_tree_contribution"]].copy().reset_index(drop=True)
    # Leaf assignments are serialised as numeric ids in some monthly sidecars
    # and as strings in others.  A canonical string key is required for the
    # assignment-to-catalogue join below.
    match["leaf_token"] = match["leaf_token"].astype(str)
    match["leaf_row"] = np.arange(len(match), dtype=np.int32)
    match["best_similarity"] = best.astype(np.float32)
    match["top2_margin"] = gap.astype(np.float32)
    match["unmatched_probability"] = unmatched.astype(np.float32)
    for pos in range(order.shape[1]):
        match[f"top{pos + 1}_prototype"] = order[:, pos].astype(np.int16)
        match[f"top{pos + 1}_probability"] = probabilities[:, pos].astype(np.float32)
        match[f"top{pos + 1}_similarity"] = top_similarity[:, pos].astype(np.float32)
    return match, order, probabilities.astype(np.float32), similarity


def _materialize_row_features(
    leaves: pd.DataFrame,
    catalog: pd.DataFrame,
    match: pd.DataFrame,
    prototype_count: int,
) -> pd.DataFrame:
    """Vectorise row exposures from fixed per-leaf prototype matches."""

    n_rows = len(leaves)
    total = np.zeros(n_rows, dtype=np.float32)
    matched = np.zeros(n_rows, dtype=np.float32)
    confidence = np.zeros(n_rows, dtype=np.float32)
    margin = np.zeros(n_rows, dtype=np.float32)
    abs_exposure = np.zeros((n_rows, prototype_count), dtype=np.float32)
    signed_exposure = np.zeros((n_rows, prototype_count), dtype=np.float32)
    lookup = match.set_index(["tree_index", "leaf_token"], drop=False)
    leaf_columns = [column for column in leaves.columns if column.startswith("leaf_assignment__")]
    row_index = np.arange(n_rows)
    for column in leaf_columns:
        try:
            tree_index = int(column.rsplit("_", 1)[1])
        except ValueError:
            continue
        local = catalog.loc[pd.to_numeric(catalog["tree_index"], errors="coerce").eq(tree_index)]
        if local.empty:
            continue
        local_token = local["leaf_token"].astype(str).tolist()
        leaf_row_lookup = {
            token: int(lookup.loc[(tree_index, token), "leaf_row"])
            for token in local_token
            if (tree_index, token) in lookup.index
        }
        assigned = leaves[column].astype(str).map(leaf_row_lookup).fillna(-1).to_numpy(dtype=np.int32)
        valid = assigned >= 0
        if not valid.any():
            continue
        rows = row_index[valid]
        leaf_rows = assigned[valid]
        contributions = pd.to_numeric(match.loc[leaf_rows, "ensemble_tree_contribution"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        mass = np.abs(contributions)
        total[rows] += mass
        unmatched = match.loc[leaf_rows, "unmatched_probability"].to_numpy(np.float32)
        best = match.loc[leaf_rows, "best_similarity"].to_numpy(np.float32)
        gap = match.loc[leaf_rows, "top2_margin"].to_numpy(np.float32)
        matched[rows] += mass * (1.0 - unmatched)
        confidence[rows] += mass * best
        margin[rows] += mass * gap
        for pos in range(1, TOP_N + 1):
            prototype = match.loc[leaf_rows, f"top{pos}_prototype"].to_numpy(np.int32)
            probability = match.loc[leaf_rows, f"top{pos}_probability"].to_numpy(np.float32)
            usable = (prototype >= 0) & (probability > 0.0)
            if not usable.any():
                continue
            # Each row has one assignment for a tree, so indexed addition is
            # safe and avoids a row-by-row Python loop.
            abs_exposure[rows[usable], prototype[usable]] += mass[usable] * probability[usable]
            signed_exposure[rows[usable], prototype[usable]] += contributions[usable] * probability[usable]
    denom = np.maximum(total, 1e-12)
    abs_exposure /= denom[:, None]
    signed_exposure /= denom[:, None]
    represented = np.clip(abs_exposure.sum(axis=1), 0.0, 1.0)
    p = np.divide(abs_exposure, np.maximum(represented[:, None], 1e-12), out=np.zeros_like(abs_exposure), where=represented[:, None] > 0)
    order = np.sort(p, axis=1)
    out: dict[str, np.ndarray] = {
        "prototype_matched_mass": np.divide(matched, denom, out=np.zeros_like(matched), where=total > 0),
        "prototype_unmatched_mass": np.clip(1.0 - np.divide(matched, denom, out=np.zeros_like(matched), where=total > 0), 0.0, 1.0),
        "prototype_match_similarity": np.divide(confidence, denom, out=np.zeros_like(confidence), where=total > 0),
        "prototype_top2_margin": np.divide(margin, denom, out=np.zeros_like(margin), where=total > 0),
        "prototype_entropy": (-(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)).astype(np.float32),
        "prototype_assignment_count": (abs_exposure > 1e-8).sum(axis=1).astype(np.int16),
    }
    for idx in range(prototype_count):
        out[f"prototype__{idx:02d}__abs_contribution"] = abs_exposure[:, idx]
        out[f"prototype__{idx:02d}__signed_contribution"] = signed_exposure[:, idx]
    if prototype_count:
        out["prototype_exposure_top2_margin"] = (order[:, -1] - order[:, -2] if prototype_count > 1 else order[:, -1]).astype(np.float32)
        out["prototype_top"] = np.argmax(p, axis=1).astype(np.int16)
    return pd.DataFrame(out, index=leaves.index)


def _opportunity_mask(frame: pd.DataFrame) -> np.ndarray:
    """Causal broad-base opportunity conditioning, fit independently per month."""

    score = pd.to_numeric(frame["base_score"], errors="coerce")
    threshold = score.groupby(frame["month"].astype(str)).transform(lambda values: values.quantile(0.60))
    return (score >= threshold).fillna(False).to_numpy(bool)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.sum(values[usable] * weights[usable]) / max(float(np.sum(weights[usable])), 1e-12)) if usable.any() else 0.0


def _joint_synergy_similarity(exposure: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Train-only pair synergy, conditional on jointly active opportunity rows."""

    n_features = exposure.shape[1]
    effect = np.array([_weighted_mean(residual, exposure[:, idx]) for idx in range(n_features)], dtype=float)
    scale = max(float(np.nanmedian(np.abs(residual - np.nanmedian(residual)))), 50.0)
    similarity = np.eye(n_features, dtype=float)
    rows: list[dict[str, float]] = []
    for i in range(n_features):
        for j in range(i):
            joint_weight = np.minimum(exposure[:, i], exposure[:, j])
            joint_effect = _weighted_mean(residual, joint_weight)
            individual = 0.5 * (effect[i] + effect[j])
            synergy = joint_effect - individual
            same_direction = float(np.sign(joint_effect) == np.sign(individual) and np.sign(individual) != 0)
            # Closer conditional effects and positive joint amplification are
            # a similarity signal; the raw bps value remains in the audit.
            effect_similarity = math.exp(-abs(effect[i] - effect[j]) / scale)
            amplification = 1.0 / (1.0 + math.exp(-synergy / scale))
            value = float(np.clip(0.55 * effect_similarity + 0.30 * amplification + 0.15 * same_direction, 0.0, 1.0))
            similarity[i, j] = similarity[j, i] = value
            rows.append({
                "prototype_i": i, "prototype_j": j,
                "effect_i_bps": effect[i], "effect_j_bps": effect[j],
                "joint_effect_bps": joint_effect, "joint_synergy_bps": synergy,
                "same_direction": same_direction, "synergy_similarity": value,
            })
    return similarity, pd.DataFrame(rows)


def _cluster_deltas(exposure: np.ndarray, labels: np.ndarray, target: np.ndarray) -> np.ndarray:
    represented = np.maximum(exposure.sum(axis=1), 1e-12)
    out: list[float] = []
    for cluster in range(int(labels.max()) + 1):
        membership = exposure[:, labels == cluster].sum(axis=1) / represented
        active = _weighted_mean(target, membership)
        inactive = _weighted_mean(target, 1.0 - membership)
        out.append(active - inactive)
    return np.asarray(out, dtype=float)


def _cluster_candidates(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    prototype_fields: Sequence[str],
    count: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray, pd.DataFrame, pd.DataFrame]:
    disc_abs = np.maximum(discovery.loc[:, prototype_fields].to_numpy(float), 0.0)
    val_abs = np.maximum(validation.loc[:, prototype_fields].to_numpy(float), 0.0)
    disc_signed = discovery.loc[:, [name.replace("__abs_", "__signed_") for name in prototype_fields]].to_numpy(float)
    disc_residual = discovery["residual_bps"].to_numpy(float)
    val_residual = validation["residual_bps"].to_numpy(float)
    opportunity = _opportunity_mask(discovery)
    conditioned_abs = disc_abs * opportunity[:, None]
    conditioned_signed = disc_signed * opportunity[:, None]
    # Co-activation and contribution coherence are conditioned on candidates
    # the broad base would consider; the residual target is training-only.
    base_similarity, pair_audit, _ = pairwise_cofiring_similarity(
        pd.DataFrame(conditioned_abs), pd.DataFrame(conditioned_signed), disc_residual,
        active_threshold=1e-8,
    )
    synergy_similarity, synergy_audit = _joint_synergy_similarity(conditioned_abs, disc_residual)
    similarity = np.clip(0.65 * base_similarity + 0.35 * synergy_similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    labels_by_k: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    months_disc = discovery["month"].astype(str).to_numpy()
    months_val = validation["month"].astype(str).to_numpy()
    for k in CLUSTER_COUNTS:
        if k >= count:
            continue
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(1.0 - similarity)
        labels_by_k[k] = labels.astype(np.int16)
        represented = np.maximum(disc_abs.sum(axis=1), 1e-12)
        masses = np.array([disc_abs[:, labels == c].sum() for c in range(k)], dtype=float)
        shares = masses / max(float(masses.sum()), 1e-12)
        compactness = []
        support = []
        stability = []
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            tri = similarity[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([0.0])
            compactness.append(float(np.mean(tri)))
            train_membership = disc_abs[:, idx].sum(axis=1) / represented
            val_membership = val_abs[:, idx].sum(axis=1) / np.maximum(val_abs.sum(axis=1), 1e-12)
            train_active = []
            val_active = []
            for month in sorted(set(months_disc)):
                train_active.append(float(np.mean(train_membership[months_disc == month] >= ACTIVE_THRESHOLD)))
            for month in sorted(set(months_val)):
                val_active.append(float(np.mean(val_membership[months_val == month] >= ACTIVE_THRESHOLD)))
            support.append(min(float(np.mean(np.asarray(train_active) >= ACTIVE_THRESHOLD)), float(np.mean(np.asarray(val_active) >= ACTIVE_THRESHOLD))))
            stability.append(1.0 - min(1.0, float(np.std(train_active + val_active))))
        disc_delta = _cluster_deltas(disc_abs, labels, disc_residual)
        val_delta = _cluster_deltas(val_abs, labels, val_residual)
        sign_stability = float(np.mean((np.sign(disc_delta) == np.sign(val_delta)) & (np.sign(disc_delta) != 0)))
        balance = -float(np.sum(shares * np.log(np.maximum(shares, 1e-12)))) / max(math.log(k), 1e-12)
        silhouette = float(silhouette_score(1.0 - similarity, labels, metric="precomputed")) if len(set(labels)) > 1 else float("nan")
        represented_mass = float(np.mean(discovery["prototype_matched_mass"].to_numpy(float)))
        score = (
            0.28 * represented_mass
            + 0.20 * balance
            + 0.20 * float(np.mean(support))
            + 0.15 * float(np.mean(compactness))
            + 0.10 * sign_stability
            + 0.07 * float(np.mean(stability))
        )
        valid = bool(
            represented_mass >= MIN_ROW_MATCHED_MASS
            and shares.max() <= 0.65
            and shares.min() >= 0.02
            and min(support, default=0.0) >= 0.50
            and sign_stability >= 0.50
        )
        rows.append({
            "prototype_count": count, "cluster_count": k, "selection_score": score,
            "valid_contract": valid, "represented_mass": represented_mass,
            "balance": balance, "max_mass_share": float(shares.max()), "min_mass_share": float(shares.min()),
            "mean_support": float(np.mean(support)), "min_support": float(np.min(support)),
            "mean_compactness": float(np.mean(compactness)), "mean_activation_stability": float(np.mean(stability)),
            "validation_sign_stability": sign_stability,
            "validation_mean_abs_delta_bps": float(np.mean(np.abs(val_delta))),
            "silhouette": silhouette,
        })
    audit = pd.DataFrame(rows).sort_values(
        ["valid_contract", "selection_score", "validation_sign_stability", "cluster_count"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    return audit, labels_by_k, similarity, pair_audit, synergy_audit


def _prototype_quality(frame: pd.DataFrame, prototype_fields: Sequence[str], count: int) -> dict[str, float]:
    values = np.maximum(frame.loc[:, prototype_fields].to_numpy(float), 0.0)
    months = frame["month"].astype(str).to_numpy()
    support = []
    mass = []
    for idx in range(values.shape[1]):
        active_by_month = [float(np.mean(values[months == month, idx] >= ACTIVE_THRESHOLD)) for month in sorted(set(months))]
        support.append(float(np.mean(np.asarray(active_by_month) >= ACTIVE_THRESHOLD)))
        mass.append(float(np.mean(values[:, idx])))
    shares = np.asarray(mass) / max(float(np.sum(mass)), 1e-12)
    return {
        "prototype_count": count,
        "matched_mass_mean": float(np.mean(frame["prototype_matched_mass"])),
        "matched_mass_p10": float(np.quantile(frame["prototype_matched_mass"], 0.10)),
        "prototype_mean_month_support": float(np.mean(support)),
        "prototype_min_month_support": float(np.min(support)),
        "prototype_balance": -float(np.sum(shares * np.log(np.maximum(shares, 1e-12)))) / max(math.log(count), 1e-12),
        "prototype_min_mass_share": float(np.min(shares)),
    }


def _serialize_vectorizer(vectorizer: TfidfVectorizer, centroids: np.ndarray, path: Path) -> None:
    # NPZ is compact and avoids serialising a large vocabulary into JSON.
    np.savez_compressed(path, idf=np.asarray(vectorizer.idf_, dtype=np.float32), centroids=centroids)
    (path.with_suffix(".vocabulary.json")).write_text(json.dumps(vectorizer.vocabulary_, sort_keys=True) + "\n")


def run(
    *,
    base_path: Path = DEFAULT_BASE,
    family_path: Path = DEFAULT_FAMILY,
    meta_path: Path = DEFAULT_META,
    raw_root: Path = DEFAULT_RAW,
    out: Path = DEFAULT_OUT,
) -> Path:
    if out.exists():
        raise FileExistsError(out)
    frame, _, _ = _load(base_path, family_path, meta_path, development_end=pd.Timestamp("2099-01-01", tz="UTC"))
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    frame["month"] = frame["month"].astype(str).map(_month)
    frame["residual_bps"] = frame["net_bps"].to_numpy(float) - frame["base_expected_bps"].to_numpy(float)
    raw: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    catalogues: list[pd.DataFrame] = []
    for path in sorted(raw_root.glob("month=*/leaf_rule_catalog.parquet")):
        month = path.parent.name.split("=", 1)[1]
        catalog, leaves = _load_raw_month(raw_root, month)
        raw[month] = (catalog, leaves)
        catalogues.append(catalog)
    if not raw:
        raise FileNotFoundError(f"no structural leaf catalogues under {raw_root}")
    all_catalog = pd.concat(catalogues, ignore_index=True)
    dev_catalog = all_catalog.loc[all_catalog["fold_id"].astype(str).str.startswith("2024-")].copy()
    if dev_catalog.empty:
        raise ValueError("no 2024 catalogue rows available to freeze the prototype contract")

    all_result: dict[int, dict[str, Any]] = {}
    prototype_rows: list[dict[str, Any]] = []
    cluster_audits: list[pd.DataFrame] = []
    for prototype_count in PROTOTYPE_COUNTS:
        vectorizer, _, _, centroids = _fit_prototypes(dev_catalog, prototype_count)
        row_parts: list[pd.DataFrame] = []
        match_parts: list[pd.DataFrame] = []
        for month, (catalog, leaves) in raw.items():
            match, _, _, _ = _match_catalog(catalog, vectorizer, centroids)
            features = _materialize_row_features(leaves, catalog, match, prototype_count)
            row = leaves[["candidate_id", "__ts__"]].copy().reset_index(drop=True)
            row["month"] = str(month)
            row_parts.append(pd.concat([row, features.reset_index(drop=True)], axis=1))
            match_parts.append(match.assign(month=str(month)))
        row_features = pd.concat(row_parts, ignore_index=True).drop_duplicates(["candidate_id", "__ts__"], keep="last")
        aligned = frame.merge(row_features, on=["candidate_id", "__ts__", "month"], how="inner", validate="one_to_one")
        dev = aligned.loc[aligned["month"].str.startswith("2024-")].copy()
        held = aligned.loc[aligned["month"].str.startswith("2025-")].copy()
        if len(dev) < 500 or len(held) < 500:
            raise ValueError(f"insufficient 2024/2025 rows after structural alignment for {prototype_count} prototypes")
        dev_months = sorted(dev["month"].unique())
        split = max(2, len(dev_months) - 2)
        discovery = dev.loc[dev["month"].isin(dev_months[:split])].copy()
        validation = dev.loc[dev["month"].isin(dev_months[split:])].copy()
        prototype_fields = [f"prototype__{idx:02d}__abs_contribution" for idx in range(prototype_count)]
        quality = _prototype_quality(dev, prototype_fields, prototype_count)
        cluster_audit, labels_by_k, similarity, pair_audit, synergy_audit = _cluster_candidates(
            discovery, validation, prototype_fields, prototype_count,
        )
        quality["candidate_cluster_valid_count"] = int(cluster_audit["valid_contract"].sum())
        quality["best_cluster_score"] = float(cluster_audit["selection_score"].max())
        prototype_rows.append(quality)
        cluster_audits.append(cluster_audit)
        all_result[prototype_count] = {
            "vectorizer": vectorizer,
            "centroids": centroids,
            "rows": row_features,
            "matches": pd.concat(match_parts, ignore_index=True),
            "aligned": aligned,
            "held": held,
            "prototype_fields": prototype_fields,
            "audit": cluster_audit,
            "labels_by_k": labels_by_k,
            "similarity": similarity,
            "pair_audit": pair_audit,
            "synergy_audit": synergy_audit,
        }

    prototype_audit = pd.DataFrame(prototype_rows).sort_values(
        ["candidate_cluster_valid_count", "best_cluster_score", "prototype_min_month_support", "prototype_count"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    all_cluster_audit = pd.concat(cluster_audits, ignore_index=True).sort_values(
        ["valid_contract", "selection_score", "prototype_count", "cluster_count"],
        ascending=[False, False, True, True], kind="stable",
    ).reset_index(drop=True)
    chosen = all_cluster_audit.iloc[0]
    prototype_count = int(chosen["prototype_count"])
    cluster_count = int(chosen["cluster_count"])
    selected = all_result[prototype_count]
    labels = selected["labels_by_k"][cluster_count]
    proto_fields = selected["prototype_fields"]
    def build_contracts(labels_for_k: np.ndarray, k: int, *, prefix: str) -> list[CoFiringClusterContract]:
        built: list[CoFiringClusterContract] = []
        for cluster in range(k):
            indices = np.flatnonzero(labels_for_k == cluster)
            inner = selected["similarity"][np.ix_(indices, indices)]
            tri = inner[np.triu_indices(len(indices), 1)] if len(indices) > 1 else np.array([0.0])
            built.append(CoFiringClusterContract(
                cluster_id=f"{prefix}_{cluster:02d}",
                family_fields=tuple(proto_fields[idx] for idx in indices),
                family_indices=tuple(int(idx) for idx in indices),
                mean_pair_similarity=float(np.mean(tri)),
                economic_coherence=float("nan"),
            ))
        return built

    contracts = build_contracts(labels, cluster_count, prefix="prototype_cluster")
    # Feed cluster features in stable column order.  The existing membership
    # materialiser is used, so future downstream arms receive the same
    # represented/unassigned/entropy/top-two semantics as existing models.
    all_abs = selected["aligned"].loc[:, proto_fields].copy()
    all_abs.columns = [f"prototype_{idx:02d}" for idx in range(prototype_count)]
    rewritten_contracts = [
        CoFiringClusterContract(
            cluster_id=contract.cluster_id,
            family_fields=tuple(all_abs.columns[idx] for idx in contract.family_indices),
            family_indices=contract.family_indices,
            mean_pair_similarity=contract.mean_pair_similarity,
            economic_coherence=contract.economic_coherence,
        )
        for contract in contracts
    ]
    cluster_features = materialize_memberships(all_abs, rewritten_contracts)
    combined = selected["aligned"][[
        "candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_score", "base_expected_bps", "residual_bps",
        "prototype_matched_mass", "prototype_unmatched_mass", "prototype_match_similarity", "prototype_top2_margin",
        "prototype_entropy", "prototype_exposure_top2_margin", "prototype_assignment_count",
        *proto_fields,
        *[field.replace("__abs_", "__signed_") for field in proto_fields],
    ]].copy()
    combined = pd.concat([combined.reset_index(drop=True), cluster_features.reset_index(drop=True)], axis=1)
    # Preserve every valid K from the selected prototype representation.  The
    # downstream use ablation must compare cluster sizes on identical rows,
    # rather than rediscovering a different cluster definition per arm.
    valid_sizes = sorted(
        int(value)
        for value in all_cluster_audit.loc[
            all_cluster_audit["prototype_count"].eq(prototype_count)
            & all_cluster_audit["valid_contract"].astype(bool),
            "cluster_count",
        ].unique()
    )
    size_sweep = selected["aligned"][["candidate_id", "__ts__", "month"]].copy().reset_index(drop=True)
    contracts_by_size: dict[str, list[dict[str, object]]] = {}
    for candidate_k in valid_sizes:
        candidate_labels = selected["labels_by_k"][candidate_k]
        candidate_contracts = build_contracts(candidate_labels, candidate_k, prefix=f"k{candidate_k:02d}_cluster")
        rewritten = [
            CoFiringClusterContract(
                cluster_id=contract.cluster_id,
                family_fields=tuple(all_abs.columns[idx] for idx in contract.family_indices),
                family_indices=contract.family_indices,
                mean_pair_similarity=contract.mean_pair_similarity,
                economic_coherence=contract.economic_coherence,
            )
            for contract in candidate_contracts
        ]
        candidate_features = materialize_memberships(all_abs, rewritten).add_prefix(f"k{candidate_k:02d}__")
        size_sweep = pd.concat([size_sweep, candidate_features.reset_index(drop=True)], axis=1)
        contracts_by_size[str(candidate_k)] = [contract.to_dict() for contract in rewritten]
    # Explicit 2025 target-only coverage audit.  These outcomes are never
    # consulted for prototype/cluster selection.
    held_features = combined.loc[combined["month"].str.startswith("2025-")].copy()
    target_audit = held_features.groupby("month", sort=True).agg(
        rows=("candidate_id", "size"),
        matched_mass_mean=("prototype_matched_mass", "mean"),
        matched_mass_p10=("prototype_matched_mass", lambda x: float(np.quantile(x, 0.10))),
        rows_matched_ge80=("prototype_matched_mass", lambda x: float(np.mean(x >= MIN_ROW_MATCHED_MASS))),
        match_similarity_mean=("prototype_match_similarity", "mean"),
        unassigned_mass_mean=("cluster_path_unassigned_mass", "mean"),
        cluster_entropy_mean=("cluster_path_entropy", "mean"),
    ).reset_index()
    # Useful pre-model diagnostics: outcome data are reported only after the
    # frozen contract has been selected.
    cluster_diag_rows: list[dict[str, Any]] = []
    for month, block in held_features.groupby("month", sort=True):
        for contract in rewritten_contracts:
            prefix = f"cluster__{contract.cluster_id}__"
            membership = block[f"{prefix}membership"].to_numpy(float)
            active = membership >= ACTIVE_THRESHOLD
            cluster_diag_rows.append({
                "month": month, "cluster_id": contract.cluster_id, "rows": len(block),
                "active_rows": int(active.sum()), "activation_fraction": float(active.mean()),
                "mean_membership": float(membership.mean()),
                "active_net_bps": float(block.loc[active, "net_bps"].mean()) if active.any() else float("nan"),
                "active_residual_bps": float(block.loc[active, "residual_bps"].mean()) if active.any() else float("nan"),
                "membership_residual_ic": float(spearmanr(membership, block["residual_bps"].to_numpy(float)).statistic) if np.unique(membership).size > 1 else float("nan"),
            })
    cluster_target_audit = pd.DataFrame(cluster_diag_rows)
    out.mkdir(parents=True)
    prototype_audit.to_parquet(out / "prototype_candidate_audit.parquet", index=False)
    all_cluster_audit.to_parquet(out / "cluster_candidate_audit.parquet", index=False)
    selected["matches"].to_parquet(out / "prototype_leaf_match_audit.parquet", index=False, compression="zstd")
    combined.to_parquet(out / "prototype_cluster_row_features.parquet", index=False, compression="zstd")
    size_sweep.to_parquet(out / "prototype_cluster_size_sweep_features.parquet", index=False, compression="zstd")
    target_audit.to_parquet(out / "target_2025_coverage_audit.parquet", index=False)
    cluster_target_audit.to_parquet(out / "target_2025_cluster_diagnostics.parquet", index=False)
    selected["pair_audit"].to_parquet(out / "prototype_coactivation_pair_audit.parquet", index=False)
    selected["synergy_audit"].to_parquet(out / "prototype_joint_synergy_audit.parquet", index=False)
    _serialize_vectorizer(selected["vectorizer"], selected["centroids"], out / "prototype_vectorizer.npz")
    (out / "prototype_contract.json").write_text(json.dumps({
        "schema": "tp6_sl4_structural_prototype_contract_v1",
        "side": SIDE,
        "fit_months": sorted(dev_catalog["fold_id"].astype(str).unique()),
        "prototype_count": prototype_count,
        "tokenization": "feature/branch/coarse-band plus ordered predicates and contribution sign",
        "match": {"top_n": TOP_N, "temperature": MATCH_TEMPERATURE, "unmatched_threshold": MATCH_THRESHOLD},
        "selection": "2024 only; 2025 distribution/outcomes excluded",
        "prototype_columns": proto_fields,
    }, indent=2) + "\n")
    (out / "cofiring_cluster_contract.json").write_text(json.dumps({
        "schema": "tp6_sl4_opportunity_conditioned_prototype_cluster_contract_v1",
        "prototype_count": prototype_count,
        "cluster_count": cluster_count,
        "clusters": [contract.to_dict() for contract in rewritten_contracts],
        "discovery": "2024 Apr-Sep; 2024 Oct-Nov validation",
        "similarity": "65% opportunity-conditioned coactivation/contribution + 35% joint residual synergy",
        "selection_uses_2025_outcomes": False,
        "selected_candidate": {key: (value.item() if isinstance(value, np.generic) else value) for key, value in chosen.to_dict().items()},
        "valid_cluster_sizes_for_matched_downstream_ablation": valid_sizes,
        "contracts_by_size": contracts_by_size,
    }, indent=2) + "\n")
    correctness = {
        "schema": "tp6_sl4_structural_prototype_cluster_correctness_v1",
        "prototype_fit_uses_2024_only": True,
        "cluster_selection_uses_2024_only": True,
        "opportunity_condition_is_base_score_only": True,
        "joint_synergy_is_training_only": True,
        "target_2025_outcomes_used_in_selection": False,
        "all_target_rows_have_finite_cluster_features": bool(np.isfinite(held_features.select_dtypes(include=[np.number]).to_numpy(float)).all()),
        "candidate_ids_unique": bool(combined[["candidate_id", "__ts__"]].duplicated().sum() == 0),
        "selection_coverage_gate": MIN_ROW_MATCHED_MASS,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_prototype_cluster_quality_20260809_v1",
        "status": "COMPLETE" if bool(chosen["valid_contract"]) else "COMPLETE_DIAGNOSTIC_CONTRACT_GATE_FAILED",
        "base": str(base_path), "family": str(family_path), "meta": str(meta_path), "raw_root": str(raw_root),
        "selected_prototype_count": prototype_count, "selected_cluster_count": cluster_count,
        "selected_contract_valid": bool(chosen["valid_contract"]),
        "rows": int(len(combined)), "target_2025_rows": int(len(held_features)),
        "artifacts": sorted(path.name for path in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# TP6/SL4 prototype archetype and cluster-quality sweep",
        "",
        "The contract is selected on 2024 only.  2025 is a held distribution and outcome diagnostic, not a selection source.",
        "",
        "## Prototype candidates",
        "",
        prototype_audit.round(4).to_string(index=False),
        "",
        "## Cluster candidates",
        "",
        all_cluster_audit.round(4).to_string(index=False),
        "",
        "## Selected contract",
        "",
        json.dumps(manifest, indent=2),
        "",
        "## Held-2025 coverage",
        "",
        target_audit.round(4).to_string(index=False),
    ]
    (out / "PROTOTYPE_CLUSTER_QUALITY_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({
        "out": str(out), "prototype_count": prototype_count, "cluster_count": cluster_count,
        "valid": bool(chosen["valid_contract"]), "target_rows": int(len(held_features)),
        "target_matched_mass": float(held_features["prototype_matched_mass"].mean()),
    }, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(base_path=args.base, family_path=args.family, meta_path=args.meta, raw_root=args.raw_root, out=args.out)
