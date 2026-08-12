"""Cross-model structural path archetypes and soft row-level transport.

The canonical path store contains one raw leaf catalogue and one leaf
assignment table per monthly model fit.  This module turns those opaque,
fold-local leaves into a transportable contract:

    leaf -> recurrent archetype -> economic/co-firing cluster

Archetypes are first selected from recurrence across independent development
fits.  New leaves are then matched by normalized structural path similarity;
exact family/leaf IDs are never required to survive retraining.  Unmatched
mass is retained explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _path_tokens(value: Any) -> tuple[tuple[str, str, int, int], ...]:
    try:
        path = json.loads(value) if isinstance(value, str) else value
    except Exception:
        path = []
    out = []
    for item in path or []:
        if not isinstance(item, Mapping):
            continue
        out.append((
            str(item.get("feature", "")),
            str(item.get("branch", "")),
            int(item.get("threshold_band_index", -1)),
            int(item.get("threshold_band_count", -1)),
        ))
    return tuple(out)


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b) / max(len(a | b), 1))


@dataclass(frozen=True)
class StructuralArchetype:
    archetype_id: str
    rule_signature: str
    tokens: tuple[tuple[str, str, int, int], ...]
    recurrence_folds: tuple[str, ...]
    recurrence_count: int
    sign: int
    sign_consistency: float
    contribution_median: float
    train_frequency_median: float

    def to_dict(self) -> dict[str, object]:
        return {
            "archetype_id": self.archetype_id,
            "rule_signature": self.rule_signature,
            "tokens": [list(x) for x in self.tokens],
            "recurrence_folds": list(self.recurrence_folds),
            "recurrence_count": int(self.recurrence_count),
            "sign": int(self.sign),
            "sign_consistency": float(self.sign_consistency),
            "contribution_median": float(self.contribution_median),
            "train_frequency_median": float(self.train_frequency_median),
        }


def build_recurrent_archetypes(
    catalog: pd.DataFrame,
    *,
    min_folds: int = 3,
    min_sign_consistency: float = 0.80,
    min_train_frequency: float = 0.005,
    separated_gap: int = 2,
) -> tuple[list[StructuralArchetype], pd.DataFrame]:
    """Build recurrence-first archetypes from development model fits."""

    required = {"rule_signature", "rule_structural_path_json", "fold_id", "ensemble_tree_contribution", "train_leaf_frequency"}
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise KeyError(f"catalog missing {missing}")
    rows: list[dict[str, object]] = []
    ordered_folds = sorted(catalog.fold_id.astype(str).unique())
    fold_pos = {f: i for i, f in enumerate(ordered_folds)}
    for signature, group in catalog.groupby("rule_signature", sort=True):
        folds = sorted(group.fold_id.astype(str).unique(), key=lambda x: fold_pos.get(x, 0))
        signs = np.sign(pd.to_numeric(group.ensemble_tree_contribution, errors="coerce").to_numpy(float))
        signs = signs[np.isfinite(signs) & (signs != 0)]
        if not len(signs):
            continue
        sign = int(np.sign(np.sum(signs)))
        sign_consistency = float(np.mean(signs == sign))
        freq = pd.to_numeric(group.train_leaf_frequency, errors="coerce").to_numpy(float)
        freq_med = float(np.nanmedian(freq)) if np.isfinite(freq).any() else 0.0
        separated = any((fold_pos[b] - fold_pos[a]) >= separated_gap for a in folds for b in folds)
        valid = len(folds) >= min_folds and separated and sign_consistency >= min_sign_consistency and freq_med >= min_train_frequency
        rows.append({
            "rule_signature": str(signature), "recurrence_folds": json.dumps(folds),
            "recurrence_count": len(folds), "sign": sign, "sign_consistency": sign_consistency,
            "contribution_median": float(np.nanmedian(np.abs(pd.to_numeric(group.ensemble_tree_contribution, errors="coerce")))),
            "train_frequency_median": freq_med, "separated": separated, "selected": valid,
        })
    audit = pd.DataFrame(rows)
    selected = audit.loc[audit.selected].sort_values(["recurrence_count", "sign_consistency", "rule_signature"], ascending=[False, False, True], kind="stable")
    archetypes: list[StructuralArchetype] = []
    for idx, row in enumerate(selected.itertuples(index=False)):
        group = catalog.loc[catalog.rule_signature.astype(str).eq(str(row.rule_signature))]
        tokens = _path_tokens(group.rule_structural_path_json.iloc[0])
        archetypes.append(StructuralArchetype(
            archetype_id=f"archetype_{idx:04d}", rule_signature=str(row.rule_signature), tokens=tokens,
            recurrence_folds=tuple(json.loads(row.recurrence_folds)), recurrence_count=int(row.recurrence_count),
            sign=int(row.sign), sign_consistency=float(row.sign_consistency),
            contribution_median=float(row.contribution_median), train_frequency_median=float(row.train_frequency_median),
        ))
    audit["archetype_id"] = audit.rule_signature.map({a.rule_signature: a.archetype_id for a in archetypes})
    return archetypes, audit.reset_index(drop=True)


def _path_similarity(
    tokens: tuple[tuple[str, str, int, int], ...],
    archetype: StructuralArchetype,
    contribution: float,
    train_frequency: float,
) -> tuple[float, float, float, float]:
    a, b = set(tokens), set(archetype.tokens)
    structure = _jaccard(a, b)
    interval = _jaccard({(x[0], x[2], x[3]) for x in tokens}, {(x[0], x[2], x[3]) for x in archetype.tokens})
    c = abs(float(contribution)); ac = abs(float(archetype.contribution_median))
    contribution_score = float(np.exp(-abs(np.log1p(c) - np.log1p(ac))))
    f = max(float(train_frequency), 1e-6); af = max(float(archetype.train_frequency_median), 1e-6)
    activation_score = float(np.exp(-abs(np.log(f) - np.log(af))))
    if np.sign(contribution) != 0 and np.sign(contribution) != archetype.sign:
        contribution_score *= 0.25
    score = 0.50 * structure + 0.20 * interval + 0.15 * contribution_score + 0.15 * activation_score
    return float(score), float(structure), float(interval), float(contribution_score)


def match_catalog_to_archetypes(
    catalog: pd.DataFrame,
    archetypes: Sequence[StructuralArchetype],
    *,
    temperature: float = 0.08,
    unmatched_threshold: float = 0.55,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Soft-match every new leaf to at most ``top_n`` archetypes."""

    if not archetypes:
        raise ValueError("cannot match leaves without archetypes")
    exact = {a.rule_signature: a for a in archetypes}
    match_rows: list[dict[str, object]] = []
    for row in catalog.itertuples(index=False):
        sig = str(row.rule_signature)
        tokens = _path_tokens(row.rule_structural_path_json)
        scored = []
        for arch in archetypes:
            s, structure, interval, contrib = _path_similarity(tokens, arch, row.ensemble_tree_contribution, row.train_leaf_frequency)
            scored.append((s, arch, structure, interval, contrib))
        scored.sort(key=lambda z: (z[0], z[1].archetype_id), reverse=True)
        top = scored[:top_n]
        scores = np.asarray([x[0] for x in top], float)
        probs = np.exp((scores - scores.max()) / max(temperature, 1e-4)); probs /= max(probs.sum(), EPS)
        best = float(scores[0])
        unmatched = float(np.clip((unmatched_threshold - best) / max(unmatched_threshold, EPS), 0.0, 1.0))
        probs *= 1.0 - unmatched
        rec = {
            "fold_id": str(row.fold_id), "tree_index": int(row.tree_index), "leaf_token": str(row.leaf_token),
            "rule_signature": sig, "best_similarity": best, "unmatched_probability": unmatched,
            "top1_structure": float(top[0][2]), "top1_interval": float(top[0][3]), "top1_contribution": float(top[0][4]),
        }
        for j in range(top_n):
            rec[f"top{j + 1}_archetype"] = top[j][1].archetype_id if j < len(top) else None
            rec[f"top{j + 1}_probability"] = float(probs[j]) if j < len(probs) else 0.0
            rec[f"top{j + 1}_similarity"] = float(top[j][0]) if j < len(top) else 0.0
        match_rows.append(rec)
    matches = pd.DataFrame(match_rows)
    return matches, matches.groupby("rule_signature", as_index=False).agg(
        best_similarity=("best_similarity", "mean"), unmatched_probability=("unmatched_probability", "mean"),
        rows=("rule_signature", "size"),
    )


def materialize_row_archetype_exposures(
    leaf_assignments: pd.DataFrame,
    catalog: pd.DataFrame,
    matches: pd.DataFrame,
    archetypes: Sequence[StructuralArchetype],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate leaf contributions into row-level archetype exposures."""

    arch_index = {a.archetype_id: i for i, a in enumerate(archetypes)}
    n = len(leaf_assignments); k = len(archetypes)
    abs_mat = np.zeros((n, k), dtype=np.float32)
    signed_mat = np.zeros((n, k), dtype=np.float32)
    total_abs = np.zeros(n, dtype=np.float32)
    matched_abs = np.zeros(n, dtype=np.float32)
    leaf_cols = [c for c in leaf_assignments.columns if c.startswith("leaf_assignment__")]
    for col in leaf_cols:
        try:
            tree_index = int(col.rsplit("_", 1)[1])
        except ValueError:
            continue
        ctree = catalog.loc[pd.to_numeric(catalog.tree_index, errors="coerce").eq(tree_index)].copy()
        if ctree.empty:
            continue
        by_token = {str(r.leaf_token): r for r in ctree.itertuples(index=False)}
        token_values = leaf_assignments[col].astype(str).to_numpy()
        for token in np.unique(token_values):
            row_idx = np.flatnonzero(token_values == token)
            item = by_token.get(token)
            if item is None:
                continue
            contribution = float(item.ensemble_tree_contribution)
            mass = abs(contribution)
            total_abs[row_idx] += mass
            match = matches.loc[(matches.tree_index == tree_index) & matches.leaf_token.astype(str).eq(token)]
            if match.empty:
                continue
            m = match.iloc[0]
            unmatched = float(m.unmatched_probability)
            matched_abs[row_idx] += mass * (1.0 - unmatched)
            for j in range(1, 4):
                aid = m.get(f"top{j}_archetype")
                prob = float(m.get(f"top{j}_probability", 0.0))
                if aid in arch_index and prob > 0:
                    idx = arch_index[aid]
                    abs_mat[row_idx, idx] += mass * prob
                    signed_mat[row_idx, idx] += contribution * prob
    exposure = np.divide(abs_mat, np.maximum(total_abs[:, None], EPS), out=np.zeros_like(abs_mat), where=total_abs[:, None] > EPS)
    signed = np.divide(signed_mat, np.maximum(total_abs[:, None], EPS), out=np.zeros_like(signed_mat), where=total_abs[:, None] > EPS)
    out: dict[str, np.ndarray] = {
        "archetype_matched_mass": np.divide(matched_abs, np.maximum(total_abs, EPS), out=np.zeros_like(matched_abs), where=total_abs > EPS),
        "archetype_unmatched_mass": np.clip(1.0 - np.divide(matched_abs, np.maximum(total_abs, EPS), out=np.zeros_like(matched_abs), where=total_abs > EPS), 0.0, 1.0),
    }
    for i, arch in enumerate(archetypes):
        out[f"archetype__{arch.archetype_id}__abs_contribution"] = exposure[:, i]
        out[f"archetype__{arch.archetype_id}__signed_contribution"] = signed[:, i]
        out[f"archetype__{arch.archetype_id}__active"] = (exposure[:, i] > 1e-8).astype(np.float32)
    if k:
        p = exposure / np.maximum(exposure.sum(axis=1, keepdims=True), EPS)
        order = np.sort(p, axis=1)
        out["archetype_entropy"] = (-(p * np.log(np.maximum(p, EPS))).sum(axis=1)).astype(np.float32)
        out["archetype_top2_margin"] = (order[:, -1] - order[:, -2] if k > 1 else order[:, -1]).astype(np.float32)
    features = pd.DataFrame(out, index=leaf_assignments.index)
    return features, pd.DataFrame({"total_abs_contribution": total_abs, "matched_abs_contribution": matched_abs})
