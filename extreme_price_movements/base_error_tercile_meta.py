"""Three-class, side-local base-error target for a residual meta learner.

The target deliberately asks a different question from the opportunity model:
given the frozen same-side base expected-net output, did it overestimate the
trade, land approximately correctly, or underestimate it?  Class boundaries
and class-to-bps reconstruction are fitted on resolved *training* rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


class BaseErrorTercileError(ValueError):
    """Raised when the causal residual-class contract is incomplete."""


@dataclass(frozen=True)
class BaseErrorTercileMap:
    """Training-only side-local tercile boundaries and shrunk residual means.

    Class 0 is an overestimate, class 1 is approximately correct, and class 2
    is an underestimate.  ``side_class_mean_bps`` maps predicted class
    probabilities back to a common-unit residual correction.
    """

    edges_by_side: dict[str, tuple[float, float]]
    side_class_mean_bps: dict[str, tuple[float, float, float]]
    side_class_support: dict[str, tuple[int, int, int]]
    global_class_mean_bps: tuple[float, float, float]
    shrinkage_support: float


def _residual(
    frame: pd.DataFrame,
    *,
    net_column: str,
    base_column: str,
) -> np.ndarray:
    if net_column not in frame or base_column not in frame:
        raise BaseErrorTercileError("net and frozen-base columns are required")
    value = (
        pd.to_numeric(frame[net_column], errors="coerce").to_numpy(float)
        - pd.to_numeric(frame[base_column], errors="coerce").to_numpy(float)
    )
    if not np.isfinite(value).all():
        raise BaseErrorTercileError("base-error labels must be finite")
    return value


def fit_base_error_tercile_map(
    frame: pd.DataFrame,
    *,
    side_column: str = "side_name",
    net_column: str = "net_bps",
    base_column: str = "prequential_base_expected_net_bps",
    shrinkage_support: float = 1_000.0,
) -> BaseErrorTercileMap:
    """Fit per-side train terciles and a conservative class-to-bps map."""
    if side_column not in frame or shrinkage_support <= 0:
        raise BaseErrorTercileError("side column and positive shrinkage are required")
    side = frame[side_column].astype(str).str.lower().to_numpy(object)
    if not set(side).issubset({"long", "short"}):
        raise BaseErrorTercileError("labels require explicit long/short sides")
    residual = _residual(frame, net_column=net_column, base_column=base_column)
    global_edges = np.quantile(residual, (1.0 / 3.0, 2.0 / 3.0))
    global_label = np.digitize(residual, global_edges, right=True)
    global_means = tuple(
        float(residual[global_label == klass].mean())
        if np.any(global_label == klass) else float(residual.mean())
        for klass in range(3)
    )
    edges: dict[str, tuple[float, float]] = {}
    means: dict[str, tuple[float, float, float]] = {}
    supports: dict[str, tuple[int, int, int]] = {}
    for name in ("long", "short"):
        mask = side == name
        if int(mask.sum()) < 6:
            raise BaseErrorTercileError(f"insufficient {name} support for terciles")
        local = residual[mask]
        local_edges = np.quantile(local, (1.0 / 3.0, 2.0 / 3.0))
        if not local_edges[0] < local_edges[1]:
            # Deterministic tiny separation handles an otherwise degenerate
            # residual distribution without inventing an outcome class.
            local_edges = np.array((local_edges[0] - 1e-6, local_edges[1] + 1e-6))
        label = np.digitize(local, local_edges, right=True)
        local_means, local_support = [], []
        for klass in range(3):
            class_mask = label == klass
            count = int(class_mask.sum())
            raw = float(local[class_mask].mean()) if count else global_means[klass]
            shrink = count / (count + float(shrinkage_support))
            local_means.append(shrink * raw + (1.0 - shrink) * global_means[klass])
            local_support.append(count)
        edges[name] = (float(local_edges[0]), float(local_edges[1]))
        means[name] = tuple(local_means)
        supports[name] = tuple(local_support)
    return BaseErrorTercileMap(
        edges_by_side=edges,
        side_class_mean_bps=means,
        side_class_support=supports,
        global_class_mean_bps=global_means,
        shrinkage_support=float(shrinkage_support),
    )


def labels_from_base_error(
    frame: pd.DataFrame,
    mapping: BaseErrorTercileMap,
    *,
    side_column: str = "side_name",
    net_column: str = "net_bps",
    base_column: str = "prequential_base_expected_net_bps",
) -> np.ndarray:
    """Return side-local 0/1/2 labels under a pre-fitted mapping."""
    if side_column not in frame:
        raise BaseErrorTercileError("side column is required")
    residual = _residual(frame, net_column=net_column, base_column=base_column)
    side = frame[side_column].astype(str).str.lower().to_numpy(object)
    labels = np.empty(len(frame), dtype=np.int8)
    for name in ("long", "short"):
        mask = side == name
        if name not in mapping.edges_by_side:
            raise BaseErrorTercileError(f"missing fitted terciles for {name}")
        labels[mask] = np.digitize(residual[mask], mapping.edges_by_side[name], right=True)
    return labels


def expected_base_error_bps(
    probability: Sequence[Sequence[float]],
    sides: Sequence[object],
    mapping: BaseErrorTercileMap,
) -> np.ndarray:
    """Map 3-class probabilities to the training-only residual correction."""
    p = np.asarray(probability, dtype=float)
    side = pd.Series(sides, dtype="string").str.lower().to_numpy(object)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(side):
        raise BaseErrorTercileError("probabilities must be an aligned N×3 matrix")
    if not np.isfinite(p).all() or (p < -1e-8).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise BaseErrorTercileError("probabilities must be finite simplexes")
    if not set(side).issubset({"long", "short"}):
        raise BaseErrorTercileError("sides must be long/short")
    means = np.vstack([mapping.side_class_mean_bps[str(name)] for name in side])
    return np.sum(p * means, axis=1).astype(np.float32)


__all__ = [
    "BaseErrorTercileError", "BaseErrorTercileMap", "expected_base_error_bps",
    "fit_base_error_tercile_map", "labels_from_base_error",
]
