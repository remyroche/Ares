"""Causal co-firing/economic structural-cluster contracts.

The older structural cluster implementation clustered family columns with a
generic KMeans geometry and used ``membership * residual`` as its target.  The
utilities here deliberately separate the two concerns:

* family clustering is based on co-firing (Jaccard/NPMI), contribution-profile
  coherence, and train-only economic coherence;
* cluster membership is an exposure/weight, never a target multiplier;
* a contract is selected with balance, compactness, and development-held-out
  conditional differentiation, then frozen before the final OOS population.

The input family matrix is non-negative in the canonical meta-path store, so
``contribution`` means the observed path/contribution mass.  The downstream
varying-coefficient model can still learn a signed economic correction against
the signed net residual target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


EPS = 1e-12


@dataclass(frozen=True)
class CoFiringClusterContract:
    cluster_id: str
    family_fields: tuple[str, ...]
    family_indices: tuple[int, ...]
    mean_pair_similarity: float
    economic_coherence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "family_fields": list(self.family_fields),
            "family_indices": [int(x) for x in self.family_indices],
            "mean_pair_similarity": float(self.mean_pair_similarity),
            "economic_coherence": float(self.economic_coherence),
        }


def _finite_matrix(frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    x = frame.to_numpy(float) if isinstance(frame, pd.DataFrame) else np.asarray(frame, dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 32:
        return 0.0
    a, b = x[mask], y[mask]
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pairwise_cofiring_similarity(
    abs_share: pd.DataFrame,
    contribution: pd.DataFrame,
    residual_bps: np.ndarray,
    *,
    active_threshold: float = 1e-8,
    jaccard_weight: float = 0.30,
    npmi_weight: float = 0.25,
    contribution_weight: float = 0.20,
    economic_weight: float = 0.25,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return a [0,1] family similarity plus pair and family audits.

    NPMI is computed from family co-firing events.  Economic coherence is
    intentionally modest and train-only: it compares the weighted conditional
    residual effect of each family and the sign/correlation of contribution
    profiles.  This is discovery metadata, never an inference feature.
    """

    fields = list(map(str, abs_share.columns))
    if list(map(str, contribution.columns)) != fields:
        raise ValueError("abs_share and contribution must have identical ordered columns")
    a = np.maximum(_finite_matrix(abs_share), 0.0)
    c = _finite_matrix(contribution)
    n_rows, n_fields = a.shape
    active = a > float(active_threshold)
    counts = active.sum(axis=0).astype(float)
    co = (active.T @ active).astype(float)
    union = counts[:, None] + counts[None, :] - co
    jaccard = np.divide(co, np.maximum(union, 1.0), out=np.zeros_like(co), where=union > 0)

    p_i = counts / max(float(n_rows), 1.0)
    p_ij = co / max(float(n_rows), 1.0)
    denom = np.maximum(-np.log(np.maximum(p_ij, EPS)), EPS)
    raw_npmi = np.zeros_like(p_ij)
    nonzero = p_ij > 0
    raw_npmi[nonzero] = np.log(
        np.maximum(p_ij[nonzero] / np.maximum((p_i[:, None] * p_i[None, :])[nonzero], EPS), EPS)
    ) / denom[nonzero]
    npmi = np.clip((raw_npmi + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(npmi, 1.0)

    signed_corr = np.eye(n_fields, dtype=float)
    for i in range(n_fields):
        for j in range(i):
            mask = active[:, i] & active[:, j]
            value = (_safe_corr(c[:, i], c[:, j], mask) + 1.0) / 2.0
            signed_corr[i, j] = signed_corr[j, i] = float(np.clip(value, 0.0, 1.0))

    y = np.asarray(residual_bps, dtype=float)
    scale = max(float(np.nanmedian(np.abs(y - np.nanmedian(y)))), 50.0)
    effects = np.zeros(n_fields, dtype=float)
    for i in range(n_fields):
        w = np.maximum(a[:, i], 0.0)
        ok = np.isfinite(y) & (w > 0)
        effects[i] = float(np.sum(w[ok] * y[ok]) / max(np.sum(w[ok]), EPS)) if ok.any() else 0.0
    effect_gap = np.abs(effects[:, None] - effects[None, :])
    effect_sim = np.exp(-effect_gap / scale)
    econ = 0.5 * effect_sim + 0.5 * signed_corr
    np.fill_diagonal(econ, 1.0)

    total = max(jaccard_weight + npmi_weight + contribution_weight + economic_weight, EPS)
    sim = (
        jaccard_weight * jaccard
        + npmi_weight * npmi
        + contribution_weight * signed_corr
        + economic_weight * econ
    ) / total
    sim = np.clip((sim + sim.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    pair_rows: list[dict[str, object]] = []
    for i in range(n_fields):
        for j in range(i):
            pair_rows.append({
                "family_i": fields[i], "family_j": fields[j],
                "coactivation": float(co[i, j] / max(n_rows, 1)),
                "jaccard": float(jaccard[i, j]), "npmi": float(npmi[i, j]),
                "contribution_coherence": float(signed_corr[i, j]),
                "economic_coherence": float(econ[i, j]),
                "similarity": float(sim[i, j]),
            })
    family_audit = pd.DataFrame({
        "family": fields,
        "active_rate": active.mean(axis=0),
        "economic_effect_bps": effects,
        "mean_contribution": np.mean(a, axis=0),
    })
    return sim, pd.DataFrame(pair_rows), family_audit


def materialize_memberships(
    abs_share: pd.DataFrame,
    contracts: Sequence[CoFiringClusterContract],
) -> pd.DataFrame:
    """Materialize soft membership and contribution exposure fields."""

    a = np.maximum(_finite_matrix(abs_share), 0.0)
    represented = np.minimum(a.sum(axis=1), 1.0)
    out: dict[str, np.ndarray] = {
        "cluster_path_represented_mass": represented.astype("float32"),
        "cluster_path_unassigned_mass": np.clip(1.0 - represented, 0.0, 1.0).astype("float32"),
    }
    memberships: list[np.ndarray] = []
    for contract in contracts:
        idx = np.asarray(contract.family_indices, dtype=int)
        exposure = a[:, idx].sum(axis=1)
        membership = np.divide(exposure, np.maximum(represented, EPS), out=np.zeros(len(a)), where=represented > EPS)
        memberships.append(membership)
        prefix = f"cluster__{contract.cluster_id}__"
        out[prefix + "membership"] = membership.astype("float32")
        out[prefix + "abs_contribution"] = exposure.astype("float32")
        out[prefix + "active"] = (membership > 1e-8).astype("float32")
    if memberships:
        m = np.column_stack(memberships)
        p = m / np.maximum(m.sum(axis=1, keepdims=True), EPS)
        out["cluster_path_entropy"] = (-(p * np.log(np.maximum(p, EPS))).sum(axis=1)).astype("float32")
        order = np.sort(m, axis=1)
        out["cluster_path_top2_margin"] = (order[:, -1] - order[:, -2] if m.shape[1] > 1 else order[:, -1]).astype("float32")
        out["cluster_path_top_cluster"] = np.argmax(m, axis=1).astype("int16")
    return pd.DataFrame(out, index=abs_share.index)


def _cluster_metrics(
    labels: np.ndarray,
    sim: np.ndarray,
    abs_share: np.ndarray,
    residual: np.ndarray,
    *,
    block: str,
) -> dict[str, float]:
    k = int(labels.max()) + 1
    contracts_mass = []
    comp = []
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        contracts_mass.append(float(abs_share[:, members].sum()))
        if len(members) > 1:
            tri = sim[np.ix_(members, members)][np.triu_indices(len(members), 1)]
            comp.append(float(np.mean(tri)))
        else:
            comp.append(0.0)
    mass = np.asarray(contracts_mass, dtype=float)
    mass_share = mass / max(float(mass.sum()), EPS)
    entropy = -float(np.sum(mass_share * np.log(np.maximum(mass_share, EPS)))) / max(np.log(k), EPS)
    balance = float(np.clip(entropy, 0.0, 1.0))
    memberships = []
    represented = np.minimum(abs_share.sum(axis=1), 1.0)
    for cluster in range(k):
        m = np.divide(abs_share[:, labels == cluster].sum(axis=1), np.maximum(represented, EPS), out=np.zeros(len(abs_share)), where=represented > EPS)
        memberships.append(m)
    differentiation = []
    sign_stability = []
    mtx = np.column_stack(memberships)
    for cluster in range(k):
        m = mtx[:, cluster]
        w_active = np.clip(m, 0.0, 1.0)
        w_inactive = 1.0 - w_active
        active_mean = float(np.sum(w_active * residual) / max(np.sum(w_active), EPS))
        inactive_mean = float(np.sum(w_inactive * residual) / max(np.sum(w_inactive), EPS))
        differentiation.append(active_mean - inactive_mean)
        sign_stability.append(float(np.sign(active_mean - inactive_mean)))
    d = np.asarray(differentiation, dtype=float)
    return {
        "block": block,
        "k": float(k),
        "balance_score": balance,
        "max_mass_share": float(mass_share.max()),
        "min_mass_share": float(mass_share.min()),
        "compactness": float(np.mean(comp)) if comp else 0.0,
        "mean_abs_conditional_diff_bps": float(np.mean(np.abs(d))),
        "positive_differentiation_fraction": float(np.mean(np.asarray(sign_stability) > 0)),
        "differentiation_dispersion_bps": float(np.std(d)),
    }


def cluster_conditional_differentiation(
    abs_share: pd.DataFrame,
    labels: np.ndarray,
    residual: np.ndarray,
    *,
    block: str,
) -> pd.DataFrame:
    """Return per-cluster weighted active-vs-inactive differentiation."""

    a = np.maximum(_finite_matrix(abs_share), 0.0)
    represented = np.minimum(a.sum(axis=1), 1.0)
    rows: list[dict[str, object]] = []
    for cluster in range(int(labels.max()) + 1):
        exposure = a[:, labels == cluster].sum(axis=1)
        membership = np.divide(exposure, np.maximum(represented, EPS), out=np.zeros(len(a)), where=represented > EPS)
        wa = np.clip(membership, 0.0, 1.0)
        wi = 1.0 - wa
        active = float(np.sum(wa * residual) / max(np.sum(wa), EPS))
        inactive = float(np.sum(wi * residual) / max(np.sum(wi), EPS))
        rows.append({
            "block": block, "cluster_index": cluster,
            "active_rows": int((membership > 0.05).sum()),
            "mean_membership": float(membership.mean()),
            "active_mean_residual_bps": active,
            "inactive_mean_residual_bps": inactive,
            "active_minus_inactive_residual_bps": active - inactive,
            "abs_differentiation_bps": abs(active - inactive),
            "active_sign": float(np.sign(active - inactive)),
        })
    return pd.DataFrame(rows)


def discover_best_contract(
    train_abs: pd.DataFrame,
    train_contribution: pd.DataFrame,
    train_residual: np.ndarray,
    validation_abs: pd.DataFrame,
    validation_residual: np.ndarray,
    *,
    k_values: Sequence[int] = (4, 5, 6, 7, 8, 9),
    seed: int = 20260813,
    active_threshold: float = 1e-8,
    transport_by_k: Mapping[int, Mapping[str, float]] | None = None,
    transport_weight: float = 0.35,
) -> tuple[list[CoFiringClusterContract], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a contract using train geometry plus held-out development blocks."""

    sim, pair_audit, family_audit = pairwise_cofiring_similarity(
        train_abs,
        train_contribution,
        train_residual,
        active_threshold=active_threshold,
    )
    distance = np.clip(1.0 - sim, 0.0, 1.0)
    candidates: list[dict[str, object]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in sorted(set(int(x) for x in k_values)):
        if k < 2 or k >= len(train_abs.columns):
            continue
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(distance)
        labels_by_k[k] = labels
        sil = float(silhouette_score(distance, labels, metric="precomputed")) if len(np.unique(labels)) > 1 else 0.0
        train_m = _cluster_metrics(labels, sim, _finite_matrix(train_abs), np.asarray(train_residual, float), block="discovery")
        valid_diff = cluster_conditional_differentiation(validation_abs, labels, validation_residual, block="development_validation")
        abs_diff = float(valid_diff.abs_differentiation_bps.mean())
        sign_stable = float(np.mean(valid_diff.active_sign != 0))
        held_score = min(abs_diff / 200.0, 1.0) * sign_stable
        support_fraction = float(np.mean(valid_diff.active_rows >= max(32, int(0.05 * len(validation_abs)))))
        balance_gate = bool(train_m["max_mass_share"] <= 0.65 and train_m["min_mass_share"] >= 0.02)
        support_gate = bool(support_fraction >= 0.75)
        transport = dict((transport_by_k or {}).get(k, {}))
        transport_score = float(np.clip(transport.get("transport_score", 0.0), 0.0, 1.0))
        transport_gate = bool(transport.get("transport_gate", True))
        base_score = 0.25 * train_m["compactness"] + 0.25 * train_m["balance_score"] + 0.25 * held_score + 0.10 * min(sil + 0.5, 1.0)
        score = (1.0 - float(transport_weight)) * base_score + float(transport_weight) * transport_score
        # A cluster that disappears in the next chronological block is not a
        # portable specialist.  Balance and support are gates, not soft
        # preferences; this prevents one dominant cluster with zero OOF
        # support from winning on a large apparent differentiation.
        if not balance_gate:
            score -= 0.50
        if not support_gate:
            score -= 0.50
        if not transport_gate:
            score -= 0.50
        candidates.append({
            "k": k, "silhouette": sil, "selection_score": score,
            "base_selection_score": base_score, "transport_score": transport_score,
            "compactness": train_m["compactness"], "balance_score": train_m["balance_score"],
            "max_mass_share": train_m["max_mass_share"], "min_mass_share": train_m["min_mass_share"],
            "mean_abs_validation_diff_bps": abs_diff,
            "validation_nonzero_differentiation_fraction": sign_stable,
            "heldout_differentiation_score": held_score,
            "validation_support_fraction": support_fraction,
            "balance_gate": balance_gate,
            "support_gate": support_gate,
            "transport_gate": transport_gate,
            "transport_coverage": float(transport.get("coverage", np.nan)),
            "transport_mass_coverage": float(transport.get("mass_coverage", np.nan)),
            "transport_mapping_quality": float(transport.get("mapping_quality", np.nan)),
            "transport_mass_stability": float(transport.get("mass_stability", np.nan)),
            "valid_contract": bool(balance_gate and support_gate and transport_gate),
        })
    audit = pd.DataFrame(candidates).sort_values(["valid_contract", "selection_score", "heldout_differentiation_score", "balance_score", "k"], ascending=[False, False, False, False, True], kind="stable").reset_index(drop=True)
    if audit.empty:
        raise ValueError("no valid co-firing cluster candidates")
    selected_k = int(audit.iloc[0].k)
    selected_labels = labels_by_k[selected_k]
    contracts: list[CoFiringClusterContract] = []
    for cluster in range(selected_k):
        idx = np.flatnonzero(selected_labels == cluster)
        if len(idx) == 0:
            continue
        inner = sim[np.ix_(idx, idx)]
        tri = inner[np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([0.0])
        econ = family_audit.loc[idx, "economic_effect_bps"].to_numpy(float)
        econ_coh = float(np.exp(-np.std(econ) / max(np.nanmedian(np.abs(train_residual)), 50.0)))
        contracts.append(CoFiringClusterContract(
            cluster_id=f"cofire_cluster_{len(contracts):02d}",
            family_fields=tuple(train_abs.columns[idx].astype(str).tolist()),
            family_indices=tuple(int(x) for x in idx),
            mean_pair_similarity=float(np.mean(tri)),
            economic_coherence=econ_coh,
        ))
    validation_diff = cluster_conditional_differentiation(validation_abs, selected_labels, validation_residual, block="development_validation")
    return contracts, audit, pair_audit, validation_diff


def refit_contract(
    abs_share: pd.DataFrame,
    contribution: pd.DataFrame,
    residual: np.ndarray,
    *,
    k: int,
    active_threshold: float = 1e-8,
) -> tuple[list[CoFiringClusterContract], pd.DataFrame, pd.DataFrame]:
    """Refit the selected K on the full development population."""

    sim, pair_audit, family_audit = pairwise_cofiring_similarity(
        abs_share,
        contribution,
        residual,
        active_threshold=active_threshold,
    )
    distance = np.clip(1.0 - sim, 0.0, 1.0)
    labels = AgglomerativeClustering(n_clusters=int(k), metric="precomputed", linkage="average").fit_predict(distance)
    contracts: list[CoFiringClusterContract] = []
    for cluster in range(int(k)):
        idx = np.flatnonzero(labels == cluster)
        inner = sim[np.ix_(idx, idx)]
        tri = inner[np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([0.0])
        econ = family_audit.loc[idx, "economic_effect_bps"].to_numpy(float)
        contracts.append(CoFiringClusterContract(
            cluster_id=f"cofire_cluster_{cluster:02d}",
            family_fields=tuple(abs_share.columns[idx].astype(str).tolist()),
            family_indices=tuple(int(x) for x in idx),
            mean_pair_similarity=float(np.mean(tri)),
            economic_coherence=float(np.exp(-np.std(econ) / max(np.nanmedian(np.abs(residual)), 50.0))),
        ))
    return contracts, pair_audit, family_audit
