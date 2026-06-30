"""Compact timestamp-membership utilities with optional Numba acceleration."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None
    _NUMBA_AVAILABLE = False


def active_positions_from_membership(membership: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(membership, dtype=bool)).astype(np.int32, copy=False)


def membership_from_active_positions(active_positions: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(int(length), dtype=bool)
    pos = np.asarray(active_positions, dtype=np.int64)
    pos = pos[(pos >= 0) & (pos < int(length))]
    out[pos] = True
    return out


def get_active_positions(obj) -> np.ndarray:
    positions = getattr(obj, "active_positions", None)
    if positions is not None:
        return np.asarray(positions, dtype=np.int32)
    return active_positions_from_membership(getattr(obj, "timestamp_membership", np.array([], dtype=bool)))


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _jaccard_sorted_numba(left: np.ndarray, right: np.ndarray) -> float:
        i = 0
        j = 0
        intersection = 0
        union = 0
        last = -1
        while i < left.size or j < right.size:
            if j >= right.size or (i < left.size and left[i] < right[j]):
                value = left[i]
                i += 1
            elif i >= left.size or right[j] < left[i]:
                value = right[j]
                j += 1
            else:
                value = left[i]
                i += 1
                j += 1
                intersection += 1
            if value != last:
                union += 1
                last = value
        if union <= 0:
            return 0.0
        return intersection / union


    @njit(cache=True)
    def _block_active_counts_numba(active_positions: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
        max_block = -1
        for i in range(block_ids.size):
            if block_ids[i] > max_block:
                max_block = block_ids[i]
        if max_block < 0:
            return np.zeros(0, dtype=np.int64)
        counts = np.zeros(max_block + 1, dtype=np.int64)
        for i in range(active_positions.size):
            pos = active_positions[i]
            if pos >= 0 and pos < block_ids.size:
                block = block_ids[pos]
                if block >= 0:
                    counts[block] += 1
        return counts

else:
    _jaccard_sorted_numba = None
    _block_active_counts_numba = None


def jaccard_active_positions(left: np.ndarray, right: np.ndarray) -> float:
    a = np.unique(np.asarray(left, dtype=np.int32))
    b = np.unique(np.asarray(right, dtype=np.int32))
    if a.size == 0 and b.size == 0:
        return 0.0
    if _jaccard_sorted_numba is not None:
        return float(_jaccard_sorted_numba(a, b))
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = np.union1d(a, b).size
    return float(intersection / max(union, 1))


def jaccard_sorted_active_positions(left: np.ndarray, right: np.ndarray) -> float:
    """Jaccard for already sorted unique active-position arrays."""

    a = np.asarray(left, dtype=np.int32)
    b = np.asarray(right, dtype=np.int32)
    if a.size == 0 and b.size == 0:
        return 0.0
    if _jaccard_sorted_numba is not None:
        return float(_jaccard_sorted_numba(a, b))
    i = 0
    j = 0
    intersection = 0
    union = 0
    last = None
    while i < a.size or j < b.size:
        if j >= b.size or (i < a.size and a[i] < b[j]):
            value = int(a[i])
            i += 1
        elif i >= a.size or int(b[j]) < int(a[i]):
            value = int(b[j])
            j += 1
        else:
            value = int(a[i])
            i += 1
            j += 1
            intersection += 1
        if last is None or value != last:
            union += 1
            last = value
    return float(intersection / max(union, 1))


def active_block_count(active_positions: np.ndarray, block_ids: np.ndarray) -> int:
    pos = np.asarray(active_positions, dtype=np.int32)
    blocks = np.asarray(block_ids, dtype=np.int64)
    if _block_active_counts_numba is not None:
        return int(np.sum(_block_active_counts_numba(pos, blocks) > 0))
    valid = pos[(pos >= 0) & (pos < blocks.size)]
    if valid.size == 0:
        return 0
    return int(np.unique(blocks[valid][blocks[valid] >= 0]).size)
