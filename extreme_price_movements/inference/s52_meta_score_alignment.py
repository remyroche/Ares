"""Frozen score-domain bridge for the shared S52 final meta refit.

The final model is fit on every resolved candidate row.  That can change the raw
score scale relative to the OOF champion which trained the residual/MLP
postprocessors and policy rank references.  This module restores that *score
domain* with a side-specific monotonic map.  It never changes within-side
ordering and therefore cannot repair a model whose ordering has drifted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


SCHEMA = "s52_meta_score_alignment_v1"


def _finite_sorted(values: Any) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.sort(out[np.isfinite(out)])


def _quantile_knots(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, probabilities, method="linear")
    except TypeError:  # NumPy < 1.22
        return np.quantile(values, probabilities, interpolation="linear")


def fit_s52_meta_score_alignment(
    final_scores_by_side: dict[str, Any],
    champion_oof_scores_by_side: dict[str, Any],
    *,
    knot_count: int = 4097,
) -> dict[str, Any]:
    """Fit a train-score percentile -> champion-OOF score bridge per side."""
    if int(knot_count) < 33:
        raise ValueError("knot_count must be at least 33")
    probabilities = np.linspace(0.0, 1.0, int(knot_count), dtype=np.float64)
    sides: dict[str, dict[str, Any]] = {}
    for side, raw_values in sorted(final_scores_by_side.items()):
        name = str(side).lower()
        raw = _finite_sorted(raw_values)
        target = _finite_sorted(champion_oof_scores_by_side.get(name, []))
        if raw.size < 64 or target.size < 64:
            continue
        source_knots = _quantile_knots(raw, probabilities)
        target_knots = _quantile_knots(target, probabilities)
        # ``np.interp`` requires a strictly increasing source grid.  Keeping
        # the last target at a duplicated source value preserves monotonicity.
        unique_source, reverse = np.unique(source_knots[::-1], return_index=True)
        keep = source_knots.size - 1 - reverse
        keep.sort()
        source_knots = source_knots[keep]
        target_knots = target_knots[keep]
        if source_knots.size < 2:
            continue
        sides[name] = {
            "final_train_rows": int(raw.size),
            "champion_oof_rows": int(target.size),
            "source_knots": source_knots.astype(np.float32).tolist(),
            "target_knots": target_knots.astype(np.float32).tolist(),
            "final_train_quantiles": {
                "p10": float(_quantile_knots(raw, np.array([0.10]))[0]),
                "p50": float(_quantile_knots(raw, np.array([0.50]))[0]),
                "p90": float(_quantile_knots(raw, np.array([0.90]))[0]),
            },
            "champion_oof_quantiles": {
                "p10": float(_quantile_knots(target, np.array([0.10]))[0]),
                "p50": float(_quantile_knots(target, np.array([0.50]))[0]),
                "p90": float(_quantile_knots(target, np.array([0.90]))[0]),
            },
        }
    if not sides:
        raise ValueError("No side has sufficient scores for S52 score alignment")
    return {
        "schema": SCHEMA,
        "enabled": True,
        "mode": "side_specific_monotonic_quantile_bridge",
        "knot_count_requested": int(knot_count),
        "sides": sides,
    }


def fit_paired_s52_meta_score_alignment(
    source_scores_by_side: dict[str, Any],
    target_scores_by_side: dict[str, Any],
    *,
    minimum_rows: int = 256,
) -> dict[str, Any]:
    """Fit a monotonic same-row bridge into the champion score domain.

    Unlike a quantile bridge, this uses paired scores for the same observations.
    It is intended for final-refit score-domain distillation where the source
    learner's in-sample distribution differs from a checkpoint's OOS
    distribution. Callers remain responsible for fitting this bridge on a
    chronological calibration period and validating it on later rows.
    """
    sides: dict[str, dict[str, Any]] = {}
    for side, raw_source in sorted(source_scores_by_side.items()):
        name = str(side).lower()
        source = np.asarray(raw_source, dtype=np.float64).reshape(-1)
        target = np.asarray(target_scores_by_side.get(name, []), dtype=np.float64).reshape(-1)
        if source.size != target.size:
            raise ValueError(
                f"Paired score size mismatch for {name}: {source.size} != {target.size}"
            )
        finite = np.isfinite(source) & np.isfinite(target)
        source = source[finite]
        target = target[finite]
        if source.size < int(minimum_rows):
            continue
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(source, target)
        source_knots = np.asarray(model.X_thresholds_, dtype=np.float64)
        target_knots = np.asarray(model.y_thresholds_, dtype=np.float64)
        if source_knots.size < 2 or source_knots.size != target_knots.size:
            continue
        sides[name] = {
            "final_train_rows": int(source.size),
            "champion_oof_rows": int(target.size),
            "source_knots": source_knots.astype(np.float32).tolist(),
            "target_knots": target_knots.astype(np.float32).tolist(),
            "final_train_quantiles": {
                "p10": float(np.quantile(source, 0.10)),
                "p50": float(np.quantile(source, 0.50)),
                "p90": float(np.quantile(source, 0.90)),
            },
            "champion_oof_quantiles": {
                "p10": float(np.quantile(target, 0.10)),
                "p50": float(np.quantile(target, 0.50)),
                "p90": float(np.quantile(target, 0.90)),
            },
        }
    if not sides:
        raise ValueError("No side has sufficient paired scores for S52 score alignment")
    return {
        "schema": SCHEMA,
        "enabled": True,
        "mode": "side_specific_same_row_isotonic_bridge",
        "sides": sides,
    }


def apply_s52_meta_score_alignment(
    scores: Any,
    alignment: dict[str, Any] | None,
    *,
    side: str | None,
) -> np.ndarray:
    """Map scores into the frozen champion OOF domain without reordering."""
    values = np.asarray(scores, dtype=np.float64)
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return values.astype(np.float32, copy=False)
    sides = alignment.get("sides")
    if not isinstance(sides, dict):
        return values.astype(np.float32, copy=False)
    side_state = sides.get(str(side or "").lower())
    if not isinstance(side_state, dict):
        return values.astype(np.float32, copy=False)
    source = np.asarray(side_state.get("source_knots", []), dtype=np.float64)
    target = np.asarray(side_state.get("target_knots", []), dtype=np.float64)
    if source.size < 2 or source.size != target.size:
        return values.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    mapped = values.copy()
    mapped[finite] = np.interp(
        values[finite], source, target, left=target[0], right=target[-1]
    )
    return mapped.astype(np.float32, copy=False)
