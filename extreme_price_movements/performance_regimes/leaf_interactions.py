"""Leaf-guided pair/triple interaction candidate extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.leaf_deduplication import leaf_jaccard
from extreme_price_movements.performance_regimes.membership import (
    get_active_positions,
    jaccard_sorted_active_positions,
)


@dataclass(frozen=True)
class LeafGuidedInteractionSeeds:
    pairs: pd.DataFrame
    triples: pd.DataFrame
    diagnostics: pd.DataFrame


def _leaf_score(leaf) -> float:
    return float(
        max(0.0, float(getattr(leaf, "oof_contribution", 0.0)))
        * max(
            float(getattr(leaf, "label_edge_mass", 0.0)),
            float(getattr(leaf, "perf_edge_mass", 0.0)),
        )
        * max(0.0, float(getattr(leaf, "stability", 0.0)))
    )


def _normal_pair(pair: tuple[str, str]) -> tuple[str, str]:
    a, b = str(pair[0]), str(pair[1])
    return (a, b) if a <= b else (b, a)


def _normal_triple(triple: tuple[str, str, str]) -> tuple[str, str, str]:
    return tuple(sorted(str(v) for v in triple))  # type: ignore[return-value]


def _prepared_active_positions(leaf) -> np.ndarray:
    positions = np.asarray(get_active_positions(leaf), dtype=np.int32)
    if positions.size <= 1:
        return positions
    if bool(np.all(positions[1:] > positions[:-1])):
        return positions
    return np.unique(positions).astype(np.int32, copy=False)


def _max_possible_jaccard(left_size: int, right_size: int) -> float:
    larger = max(int(left_size), int(right_size))
    if larger <= 0:
        return 0.0
    return float(min(int(left_size), int(right_size)) / larger)


def _top_overlap_leaves(
    leaves: Sequence,
    *,
    max_per_strategy_direction: int,
) -> list:
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for leaf in leaves:
        key = (str(getattr(leaf, "strategy", "")), str(getattr(leaf, "direction", "")))
        grouped[key].append(leaf)
    selected: list = []
    seen: set[int] = set()
    for group in grouped.values():
        ranked = sorted(
            group,
            key=lambda leaf: (-_leaf_score(leaf), str(getattr(leaf, "leaf_uid", ""))),
        )[: max(int(max_per_strategy_direction), 0)]
        for leaf in ranked:
            marker = id(leaf)
            if marker in seen:
                continue
            seen.add(marker)
            selected.append(leaf)
    return selected


def extract_leaf_guided_interactions(
    retained_leaves: Sequence,
    *,
    max_candidates_per_strategy_direction: int = 50,
    max_overlap_leaves_per_strategy_direction: int = 250,
    include_pairs: bool = True,
    include_triples: bool = True,
) -> LeafGuidedInteractionSeeds:
    """Generate bounded split-feature pair/triple seeds from retained leaves."""

    leaves = list(retained_leaves)
    pair_scores: dict[tuple[str, str, str, str], float] = defaultdict(float)
    pair_folds: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    pair_sources: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    triple_scores: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
    triple_folds: dict[tuple[str, str, str, str, str], set[int]] = defaultdict(set)
    triple_sources: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)

    for leaf in leaves:
        features = list(dict.fromkeys(str(v) for v in getattr(leaf, "split_path_features", ()) if str(v)))
        if len(features) < 2:
            continue
        score = _leaf_score(leaf)
        strategy = str(getattr(leaf, "strategy", ""))
        direction = str(getattr(leaf, "direction", ""))
        fold_id = int(getattr(leaf, "fold_id", -1))
        leaf_uid = str(getattr(leaf, "leaf_uid", ""))
        if include_pairs:
            for a, b in combinations(features, 2):
                left, right = _normal_pair((a, b))
                key = (strategy, direction, left, right)
                pair_scores[key] += score
                pair_folds[key].add(fold_id)
                pair_sources[key][leaf_uid] += 1
        if include_triples and len(features) >= 3:
            for a, b, c in combinations(features[:8], 3):
                x, y, z = _normal_triple((a, b, c))
                key = (strategy, direction, x, y, z)
                triple_scores[key] += score
                triple_folds[key].add(fold_id)
                triple_sources[key][leaf_uid] += 1

    overlap_leaf_count = 0
    if include_pairs and max_overlap_leaves_per_strategy_direction > 0:
        overlap_leaves = _top_overlap_leaves(
            leaves,
            max_per_strategy_direction=int(max_overlap_leaves_per_strategy_direction),
        )
        overlap_leaf_count = len(overlap_leaves)
        positions_by_id = {id(leaf): _prepared_active_positions(leaf) for leaf in overlap_leaves}
        for left_leaf, right_leaf in combinations(overlap_leaves, 2):
            left_positions = positions_by_id[id(left_leaf)]
            right_positions = positions_by_id[id(right_leaf)]
            if _max_possible_jaccard(left_positions.size, right_positions.size) < 0.75:
                continue
            overlap = jaccard_sorted_active_positions(left_positions, right_positions)
            if overlap < 0.75:
                continue
            left_features = list(getattr(left_leaf, "split_path_features", ()))
            right_features = list(getattr(right_leaf, "split_path_features", ()))
            for a in left_features[:4]:
                for b in right_features[:4]:
                    if str(a) == str(b):
                        continue
                    x, y = _normal_pair((str(a), str(b)))
                    strategy = str(getattr(left_leaf, "strategy", ""))
                    direction = str(getattr(left_leaf, "direction", ""))
                    if strategy != str(getattr(right_leaf, "strategy", "")):
                        strategy = f"{strategy}__x__{getattr(right_leaf, 'strategy', '')}"
                    if direction != str(getattr(right_leaf, "direction", "")):
                        direction = f"{direction}__x__{getattr(right_leaf, 'direction', '')}"
                    key = (strategy, direction, x, y)
                    pair_scores[key] += overlap * (_leaf_score(left_leaf) + _leaf_score(right_leaf))
                    pair_folds[key].update(
                        [
                            int(getattr(left_leaf, "fold_id", -1)),
                            int(getattr(right_leaf, "fold_id", -1)),
                        ]
                    )
                    pair_sources[key][str(getattr(left_leaf, "leaf_uid", ""))] += 1
                    pair_sources[key][str(getattr(right_leaf, "leaf_uid", ""))] += 1

    pair_rows: list[dict[str, object]] = []
    for key, raw_score in pair_scores.items():
        strategy, direction, left, right = key
        fold_frequency = len(pair_folds[key])
        redundancy_count = sum(pair_sources[key].values())
        score = float(raw_score * fold_frequency / np.sqrt(1.0 + redundancy_count))
        pair_rows.append(
            {
                "strategy": strategy,
                "direction": direction,
                "feature_i": left,
                "feature_j": right,
                "candidate_score": score,
                "fold_frequency": int(fold_frequency),
                "redundancy_count": int(redundancy_count),
                "source_leaf_count": int(len(pair_sources[key])),
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    if not pair_frame.empty:
        pair_frame = (
            pair_frame.sort_values("candidate_score", ascending=False, kind="mergesort")
            .groupby(["strategy", "direction"], group_keys=False)
            .head(int(max_candidates_per_strategy_direction))
            .reset_index(drop=True)
        )

    triple_rows: list[dict[str, object]] = []
    for key, raw_score in triple_scores.items():
        strategy, direction, a, b, c = key
        fold_frequency = len(triple_folds[key])
        redundancy_count = sum(triple_sources[key].values())
        score = float(raw_score * fold_frequency / np.sqrt(1.0 + redundancy_count))
        triple_rows.append(
            {
                "strategy": strategy,
                "direction": direction,
                "feature_i": a,
                "feature_j": b,
                "feature_k": c,
                "candidate_score": score,
                "fold_frequency": int(fold_frequency),
                "redundancy_count": int(redundancy_count),
                "source_leaf_count": int(len(triple_sources[key])),
            }
        )
    triple_frame = pd.DataFrame(triple_rows)
    if not triple_frame.empty:
        triple_frame = (
            triple_frame.sort_values("candidate_score", ascending=False, kind="mergesort")
            .groupby(["strategy", "direction"], group_keys=False)
            .head(int(max_candidates_per_strategy_direction))
            .reset_index(drop=True)
        )

    diagnostics = pd.DataFrame(
        [
            {
                "retained_leaf_count": int(len(leaves)),
                "pair_candidate_count": int(len(pair_frame)),
                "triple_candidate_count": int(len(triple_frame)),
                "include_pairs": bool(include_pairs),
                "include_triples": bool(include_triples),
                "overlap_leaf_count": int(overlap_leaf_count),
                "max_overlap_leaves_per_strategy_direction": int(
                    max_overlap_leaves_per_strategy_direction
                ),
            }
        ]
    )
    return LeafGuidedInteractionSeeds(pair_frame, triple_frame, diagnostics)
