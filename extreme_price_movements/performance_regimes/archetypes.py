"""Leaf clustering, archetype intensities, and archetype selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.labels import causal_ewma
from extreme_price_movements.performance_regimes.labels import StrategyPerformanceLabelBundle
from extreme_price_movements.performance_regimes.leaf_deduplication import leaf_jaccard
from extreme_price_movements.performance_regimes.membership import (
    get_active_positions,
    jaccard_sorted_active_positions,
    membership_from_active_positions,
)


@dataclass(frozen=True)
class ArchetypeClusteringConfig:
    w_jaccard: float = 0.50
    w_rule: float = 0.20
    w_effect: float = 0.20
    w_contribution: float = 0.10
    distance_threshold: float = 0.45
    min_cluster_size: int = 1
    method: Literal["hierarchical", "hdbscan", "k_medoids"] = "hierarchical"


@dataclass(frozen=True)
class MarketStateArchetype:
    archetype_id: str
    strategy: str
    direction: Literal["bad", "good"]
    leaf_ids: tuple[str, ...]
    dominant_features: tuple[str, ...]
    dominant_feature_families: tuple[str, ...]
    total_weighted_coverage: float
    mean_edge_mass: float
    mean_contribution_share: float
    mean_stability: float
    activation_timestamps: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ArchetypeBundle:
    archetypes: tuple[MarketStateArchetype, ...]
    diagnostics: pd.DataFrame
    similarity: pd.DataFrame


@dataclass(frozen=True)
class ArchetypeActivityTargets:
    activity: dict[str, pd.Series]
    sample_weights: dict[str, pd.Series]
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class SelectedArchetypeSet:
    selected: tuple[str, ...]
    report: pd.DataFrame


def _leaf_weight(leaf) -> float:
    return float(
        max(0.0, float(getattr(leaf, "contribution_share", 0.0)))
        * max(
            float(getattr(leaf, "label_edge_mass", 0.0)),
            float(getattr(leaf, "perf_edge_mass", 0.0)),
        )
        * max(0.0, float(getattr(leaf, "stability", 0.0)))
    )


def _rule_path_distance(a, b) -> float:
    fa = tuple(getattr(a, "split_path_features", ()))
    fb = tuple(getattr(b, "split_path_features", ()))
    if not fa and not fb:
        return 0.0
    sa, sb = set(fa), set(fb)
    return float(1.0 - len(sa & sb) / max(len(sa | sb), 1))


def _distance(a, b, cfg: ArchetypeClusteringConfig) -> float:
    return _distance_with_jaccard(a, b, cfg, leaf_jaccard(a, b))


def _distance_with_jaccard(a, b, cfg: ArchetypeClusteringConfig, jaccard: float) -> float:
    edge_a = max(float(getattr(a, "label_edge_mass", 0.0)), float(getattr(a, "perf_edge_mass", 0.0)))
    edge_b = max(float(getattr(b, "label_edge_mass", 0.0)), float(getattr(b, "perf_edge_mass", 0.0)))
    contrib_a = float(getattr(a, "contribution_share", 0.0))
    contrib_b = float(getattr(b, "contribution_share", 0.0))
    return float(
        cfg.w_jaccard * (1.0 - float(jaccard))
        + cfg.w_rule * _rule_path_distance(a, b)
        + cfg.w_effect * abs(edge_a - edge_b)
        + cfg.w_contribution * abs(contrib_a - contrib_b)
    )


def _family(feature: str) -> str:
    text = str(feature)
    if "__" in text:
        return text.split("__", 1)[0]
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def _prepared_active_positions(leaf) -> np.ndarray:
    positions = np.asarray(get_active_positions(leaf), dtype=np.int32)
    if positions.size <= 1:
        return positions
    if bool(np.all(positions[1:] > positions[:-1])):
        return positions
    return np.unique(positions).astype(np.int32, copy=False)


def _connected_components(leaves: list, cfg: ArchetypeClusteringConfig) -> tuple[list[list], list[dict[str, object]]]:
    n = len(leaves)
    parent = list(range(n))
    similarity_rows: list[dict[str, object]] = []
    active_positions = [_prepared_active_positions(leaf) for leaf in leaves]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            jaccard = jaccard_sorted_active_positions(active_positions[i], active_positions[j])
            d = _distance_with_jaccard(leaves[i], leaves[j], cfg, jaccard)
            similarity_rows.append(
                {
                    "left_leaf_id": str(getattr(leaves[i], "leaf_uid", "")),
                    "right_leaf_id": str(getattr(leaves[j], "leaf_uid", "")),
                    "strategy": str(getattr(leaves[i], "strategy", "")),
                    "direction": str(getattr(leaves[i], "direction", "")),
                    "distance": d,
                    "jaccard": jaccard,
                }
            )
            if d <= float(cfg.distance_threshold):
                union(i, j)
    groups: dict[int, list] = {}
    for i, leaf in enumerate(leaves):
        groups.setdefault(find(i), []).append(leaf)
    return list(groups.values()), similarity_rows


def cluster_leaves_into_archetypes(
    leaves: Sequence,
    *,
    strategy: str | None = None,
    direction: Literal["bad", "good"] | None = None,
    clustering_config: ArchetypeClusteringConfig = ArchetypeClusteringConfig(),
) -> ArchetypeBundle:
    filtered = [
        leaf
        for leaf in leaves
        if (strategy is None or str(getattr(leaf, "strategy", "")) == str(strategy))
        and (direction is None or str(getattr(leaf, "direction", "")) == str(direction))
    ]
    archetypes: list[MarketStateArchetype] = []
    diag_rows: list[dict[str, object]] = []
    sim_rows: list[dict[str, object]] = []
    groups_by_key: dict[tuple[str, str], list] = {}
    for leaf in filtered:
        groups_by_key.setdefault(
            (str(getattr(leaf, "strategy", "")), str(getattr(leaf, "direction", ""))),
            [],
        ).append(leaf)
    for (strategy_id, direction_id), group_leaves in sorted(groups_by_key.items()):
        clusters, rows = _connected_components(group_leaves, clustering_config)
        sim_rows.extend(rows)
        cluster_i = 0
        for cluster in clusters:
            if len(cluster) < int(clustering_config.min_cluster_size):
                continue
            cluster_i += 1
            feature_counts = Counter(
                feature
                for leaf in cluster
                for feature in getattr(leaf, "split_path_features", ())
            )
            dominant = tuple(name for name, _count in feature_counts.most_common(12))
            family_counts = Counter(_family(feature) for feature in dominant)
            max_len = max(
                (
                    len(getattr(leaf, "timestamp_membership", []))
                    for leaf in cluster
                ),
                default=0,
            )
            activation = np.zeros(max_len, dtype=bool)
            for leaf in cluster:
                positions = get_active_positions(leaf)
                positions = positions[(positions >= 0) & (positions < max_len)]
                activation[positions] = True
            archetype_id = f"strategy_{strategy_id}_{direction_id}_archetype_{cluster_i}"
            weights = [_leaf_weight(leaf) for leaf in cluster]
            edge_mass = [
                max(float(getattr(leaf, "label_edge_mass", 0.0)), float(getattr(leaf, "perf_edge_mass", 0.0)))
                for leaf in cluster
            ]
            archetype = MarketStateArchetype(
                archetype_id=archetype_id,
                strategy=strategy_id,
                direction=direction_id,  # type: ignore[arg-type]
                leaf_ids=tuple(str(getattr(leaf, "leaf_uid", "")) for leaf in cluster),
                dominant_features=dominant,
                dominant_feature_families=tuple(name for name, _count in family_counts.most_common(8)),
                total_weighted_coverage=float(sum(float(getattr(leaf, "weighted_coverage", 0.0)) for leaf in cluster)),
                mean_edge_mass=float(np.nanmean(edge_mass)) if edge_mass else 0.0,
                mean_contribution_share=float(np.nanmean([float(getattr(leaf, "contribution_share", 0.0)) for leaf in cluster])),
                mean_stability=float(np.nanmean([float(getattr(leaf, "stability", 0.0)) for leaf in cluster])),
                activation_timestamps=np.asarray(activation, dtype=bool),
                diagnostics={"leaf_weight_sum": float(np.nansum(weights)), "leaf_count": int(len(cluster))},
            )
            archetypes.append(archetype)
            diag_rows.append(
                {
                    "archetype_id": archetype_id,
                    "strategy": strategy_id,
                    "direction": direction_id,
                    "leaf_count": int(len(cluster)),
                    "total_weighted_coverage": archetype.total_weighted_coverage,
                    "mean_edge_mass": archetype.mean_edge_mass,
                    "mean_contribution_share": archetype.mean_contribution_share,
                    "mean_stability": archetype.mean_stability,
                }
            )
    return ArchetypeBundle(
        archetypes=tuple(archetypes),
        diagnostics=pd.DataFrame(diag_rows),
        similarity=pd.DataFrame(sim_rows),
    )


def build_archetype_activity_intensity(
    archetype: MarketStateArchetype,
    leaves: Sequence,
    timestamps: pd.Index,
    *,
    ewma_halflife: str | int = "3D",
) -> pd.Series:
    leaf_by_id = {str(getattr(leaf, "leaf_uid", "")): leaf for leaf in leaves}
    selected = [leaf_by_id[leaf_id] for leaf_id in archetype.leaf_ids if leaf_id in leaf_by_id]
    if not selected:
        return pd.Series(0.0, index=timestamps, dtype=float)
    weights = np.asarray([_leaf_weight(leaf) for leaf in selected], dtype=float)
    if float(np.nansum(weights)) <= 1e-12:
        weights = np.ones(len(selected), dtype=float)
    denom = float(np.nansum(weights))
    memberships: list[np.ndarray] = []
    for leaf in selected:
        membership = membership_from_active_positions(get_active_positions(leaf), len(timestamps))
        memberships.append(membership.astype(np.float32, copy=False))
    membership_matrix = np.vstack(memberships).astype(np.float32, copy=False)
    raw = weights.astype(np.float32, copy=False) @ membership_matrix
    raw = np.clip(raw / max(denom, 1e-12), 0.0, 1.0)
    return causal_ewma(pd.Series(raw, index=timestamps), halflife=ewma_halflife).clip(0.0, 1.0)


def build_archetype_activity_targets(
    archetypes: Sequence[MarketStateArchetype],
    leaves: Sequence,
    timestamps: pd.Index,
    *,
    ewma_halflife: str | int = "3D",
    w_min: float = 0.25,
    w_max: float = 10.0,
) -> ArchetypeActivityTargets:
    activity: dict[str, pd.Series] = {}
    weights: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    leaf_by_id = {str(getattr(leaf, "leaf_uid", "")): leaf for leaf in leaves}
    for archetype in archetypes:
        y = build_archetype_activity_intensity(
            archetype,
            leaves,
            timestamps,
            ewma_halflife=ewma_halflife,
        )
        selected = [leaf_by_id[leaf_id] for leaf_id in archetype.leaf_ids if leaf_id in leaf_by_id]
        p = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
        balance = y / p + (1.0 - y) / (1.0 - p)
        clarity = (1.0 + 2.0 * (y - p).abs()) ** 2
        confidence_values = [
            float(getattr(leaf, "contribution_share", 0.0))
            * max(float(getattr(leaf, "label_edge_mass", 0.0)), float(getattr(leaf, "perf_edge_mass", 0.0)))
            * float(getattr(leaf, "stability", 0.0))
            for leaf in selected
        ]
        confidence = float(np.nanmean(confidence_values)) if confidence_values else 1.0
        sw = (confidence * balance * clarity).clip(float(w_min), float(w_max))
        activity[archetype.archetype_id] = y
        weights[archetype.archetype_id] = sw
        rows.append(
            {
                "archetype_id": archetype.archetype_id,
                "target_mean": float(y.mean()),
                "target_min": float(y.min()),
                "target_max": float(y.max()),
                "sample_weight_mean": float(sw.mean()),
                "archetype_confidence": confidence,
            }
        )
    return ArchetypeActivityTargets(activity, weights, pd.DataFrame(rows))


def build_archetype_signed_effect_targets(
    archetypes: Sequence[MarketStateArchetype],
    archetype_intensities: Mapping[str, pd.Series],
    labels: StrategyPerformanceLabelBundle,
) -> dict[str, pd.Series]:
    """Build optional signed effect/alignment targets separately from activity."""

    out: dict[str, pd.Series] = {}
    for archetype in archetypes:
        if archetype.strategy not in labels.by_strategy:
            continue
        label_set = labels.by_strategy[archetype.strategy]
        directional = (
            label_set.bad_label
            if archetype.direction == "bad"
            else label_set.good_label
        )
        activity = pd.to_numeric(
            archetype_intensities.get(archetype.archetype_id, pd.Series(0.0, index=directional.index)),
            errors="coerce",
        ).reindex(directional.index).fillna(0.0)
        out[archetype.archetype_id] = (
            activity * (2.0 * pd.to_numeric(directional, errors="coerce").fillna(0.5) - 1.0)
        ).clip(-1.0, 1.0)
    return out


def build_cross_strategy_archetype_features(
    archetype_intensities: pd.DataFrame,
    archetype_metadata: pd.DataFrame,
):
    from extreme_price_movements.performance_regimes.portfolio_calibration import (
        CrossStrategyArchetypeFeatures,
    )

    meta = archetype_metadata.copy()
    values = archetype_intensities.copy()
    bad_cols = meta.loc[meta["direction"].astype(str).eq("bad"), "archetype_id"].astype(str).tolist()
    good_cols = meta.loc[meta["direction"].astype(str).eq("good"), "archetype_id"].astype(str).tolist()
    bad_cols = [col for col in bad_cols if col in values.columns]
    good_cols = [col for col in good_cols if col in values.columns]
    out = pd.DataFrame(index=values.index)
    out["bad_breadth"] = values[bad_cols].mean(axis=1) if bad_cols else 0.0
    out["good_breadth"] = values[good_cols].mean(axis=1) if good_cols else 0.0
    out["bad_concentration"] = values[bad_cols].max(axis=1) if bad_cols else 0.0
    out["good_concentration"] = values[good_cols].max(axis=1) if good_cols else 0.0
    out["bad_dispersion"] = values[bad_cols].std(axis=1).fillna(0.0) if bad_cols else 0.0
    out["good_dispersion"] = values[good_cols].std(axis=1).fillna(0.0) if good_cols else 0.0
    strategies = sorted(meta["strategy"].astype(str).unique()) if not meta.empty else []
    for strategy in strategies:
        s_bad = [
            col for col in bad_cols if str(meta.loc[meta["archetype_id"].eq(col), "strategy"].iloc[0]) == strategy
        ]
        s_good = [
            col for col in good_cols if str(meta.loc[meta["archetype_id"].eq(col), "strategy"].iloc[0]) == strategy
        ]
        bad_intensity = values[s_bad].mean(axis=1) if s_bad else pd.Series(0.0, index=values.index)
        good_intensity = values[s_good].mean(axis=1) if s_good else pd.Series(0.0, index=values.index)
        other_bad = values[[c for c in bad_cols if c not in s_bad]].mean(axis=1) if len(bad_cols) > len(s_bad) else 0.0
        out[f"{strategy}__bad_intensity"] = bad_intensity
        out[f"{strategy}__good_intensity"] = good_intensity
        out[f"{strategy}__specialist_opportunity"] = good_intensity * other_bad
    for left in strategies:
        for right in strategies:
            if left == right:
                continue
            out[f"{left}__bad_x__{right}__good_conflict"] = (
                out.get(f"{left}__bad_intensity", 0.0) * out.get(f"{right}__good_intensity", 0.0)
            )
    return CrossStrategyArchetypeFeatures(out, meta)


def select_useful_archetypes(
    archetype_scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    reference_model_scores: pd.DataFrame,
    max_explained_share: float = 0.95,
    min_marginal_contribution: float = 0.0,
    min_stability: float = 0.50,
) -> SelectedArchetypeSet:
    y = pd.to_numeric(targets.iloc[:, 0], errors="coerce") if not targets.empty else pd.Series(dtype=float)
    ref = pd.to_numeric(reference_model_scores.iloc[:, 0], errors="coerce") if not reference_model_scores.empty else pd.Series(0.0, index=y.index)
    base_loss = float(np.nanmean((y - y.mean()) ** 2)) if len(y) else np.nan
    ref_r2 = 1.0 - float(np.nanmean((y - ref.reindex(y.index).fillna(y.mean())) ** 2)) / max(base_loss, 1e-12)
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    remaining = list(archetype_scores.columns)
    current_pred = pd.Series(float(y.mean()) if len(y) else 0.0, index=y.index)
    current_r2 = 0.0
    while remaining:
        best_col = None
        best_r2 = current_r2
        best_pred = current_pred
        for col in remaining:
            X = archetype_scores[selected + [col]].reindex(y.index).fillna(0.0)
            try:
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)]),
                    y.to_numpy(dtype=float),
                    rcond=None,
                )
                pred = pd.Series(
                    np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)]) @ coef,
                    index=y.index,
                )
            except Exception:
                continue
            r2 = 1.0 - float(np.nanmean((y - pred) ** 2)) / max(base_loss, 1e-12)
            if r2 > best_r2:
                best_col, best_r2, best_pred = col, r2, pred
        if best_col is None:
            break
        marginal = best_r2 - current_r2
        stability = float(1.0 - archetype_scores[best_col].reindex(y.index).fillna(0.0).diff().abs().mean())
        if marginal <= float(min_marginal_contribution) or stability < float(min_stability):
            rows.append({"archetype_id": best_col, "selected": False, "marginal_contribution": marginal, "stability": stability})
            remaining.remove(best_col)
            continue
        selected.append(best_col)
        remaining.remove(best_col)
        current_r2 = best_r2
        current_pred = best_pred
        explained_share = current_r2 / max(ref_r2, 1e-12)
        rows.append({"archetype_id": best_col, "selected": True, "marginal_contribution": marginal, "stability": stability, "explained_share": explained_share})
        if explained_share >= float(max_explained_share):
            break
    return SelectedArchetypeSet(tuple(selected), pd.DataFrame(rows))
