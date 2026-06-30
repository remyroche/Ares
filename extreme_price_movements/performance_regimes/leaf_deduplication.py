"""Jaccard redundancy filtering for retained leaves."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from extreme_price_movements.performance_regimes.membership import (
    get_active_positions,
    jaccard_active_positions,
    jaccard_sorted_active_positions,
)


def leaf_jaccard(a, b) -> float:
    return jaccard_active_positions(get_active_positions(a), get_active_positions(b))


def leaf_quality(leaf) -> float:
    return float(
        max(0.0, float(getattr(leaf, "contribution_share", 0.0)))
        * max(
            float(getattr(leaf, "label_edge_mass", 0.0)),
            float(getattr(leaf, "perf_edge_mass", 0.0)),
        )
        * max(0.0, float(getattr(leaf, "stability", 0.0)))
    )


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


def deduplicate_leaves_by_jaccard(
    leaves: Sequence,
    *,
    overlap_threshold: float = 0.98,
) -> list:
    """Drop redundant same-strategy leaves by timestamp Jaccard quality."""

    ordered = sorted(
        list(leaves),
        key=lambda leaf: (
            str(getattr(leaf, "strategy", "")),
            str(getattr(leaf, "direction", "")),
            -leaf_quality(leaf),
            str(getattr(leaf, "leaf_uid", "")),
        ),
    )
    kept: list[dict[str, object] | None] = []
    kept_by_strategy: dict[str, list[int]] = {}
    for leaf in ordered:
        strategy = str(getattr(leaf, "strategy", ""))
        direction = str(getattr(leaf, "direction", ""))
        quality = leaf_quality(leaf)
        positions = _prepared_active_positions(leaf)
        replace_at: int | None = None
        should_drop = False
        for idx in kept_by_strategy.get(strategy, []):
            other_meta = kept[idx]
            if other_meta is None:
                continue
            other_positions = other_meta["positions"]
            if _max_possible_jaccard(positions.size, other_positions.size) <= float(overlap_threshold):
                continue
            overlap = jaccard_sorted_active_positions(positions, other_positions)
            if overlap <= float(overlap_threshold):
                continue
            q_leaf = quality
            q_other = float(other_meta["quality"])
            same_direction = direction == str(other_meta["direction"])
            if same_direction:
                if q_leaf > q_other:
                    replace_at = idx
                else:
                    should_drop = True
                break
            strong = q_leaf > 0.0 and q_other > 0.0
            similar = min(q_leaf, q_other) / max(max(q_leaf, q_other), 1e-12) >= 0.75
            if strong and similar:
                kept[idx] = None
                should_drop = True
                break
            if q_leaf > q_other:
                replace_at = idx
            else:
                should_drop = True
            break
        if should_drop:
            continue
        meta = {
            "leaf": leaf,
            "strategy": strategy,
            "direction": direction,
            "quality": float(quality),
            "positions": positions,
        }
        if replace_at is not None:
            kept[replace_at] = meta
        else:
            kept.append(meta)
            kept_by_strategy.setdefault(strategy, []).append(len(kept) - 1)
    return [meta["leaf"] for meta in kept if meta is not None]
