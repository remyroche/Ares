"""Current-R3 broad predictor for the strict Stage-IV tail stack.

The broad layer is deliberately the existing three-class R3 model, not a
LambdaRank substitute.  Tail and residual layers remain the canonical
LambdaRank models supplied by :mod:`stage_iv_broad_to_tail`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .stage_iv_broad_to_tail import ModelFitter, _default_lgbm_fitter


R3_CLASS_ORDER = ("adverse", "weak", "clear")
R3_OPPORTUNITY_SCORE = "p_clear_minus_p_adverse"
_R3_SEMANTIC_PARAMS = frozenset({"objective", "num_class"})


class StageIVR3FitterError(ValueError):
    """Raised when a purported current-R3 broad fit violates its contract."""


@dataclass(frozen=True)
class _R3OpportunityPredictor:
    """Expose exactly the existing R3 direct ordering coordinate."""

    model: Any

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probability = self._probability(X)
        return (probability[:, 2] - probability[:, 0]).astype(np.float32, copy=False)

    def _probability(self, X: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.model.predict_proba(X), dtype=np.float32)
        if probability.ndim != 2 or probability.shape != (len(X), 3):
            raise StageIVR3FitterError("current R3 broad model must emit [adverse, weak, clear] probabilities")
        if not np.isfinite(probability).all() or (probability < 0.0).any():
            raise StageIVR3FitterError("current R3 broad model emitted invalid probabilities")
        if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
            raise StageIVR3FitterError("current R3 broad simplex does not sum to one")
        return probability

    def predict_context(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the complete frozen R3 simplex and invariant trust fields.

        These are direct same-side OOF outputs.  They are deliberately raw
        model coordinates, not realised-value maps or period aggregates, so a
        downstream tail/meta learner can assess base confidence without a
        target-derived conversion.
        """
        probability = self._probability(X)
        ordered = np.sort(probability, axis=1)
        entropy = -np.sum(
            np.where(probability > 0.0, probability * np.log(np.maximum(probability, 1e-12)), 0.0),
            axis=1,
        )
        return pd.DataFrame({
            "r3_p_adverse": probability[:, 0],
            "r3_p_weak": probability[:, 1],
            "r3_p_clear": probability[:, 2],
            "r3_opportunity_score": probability[:, 2] - probability[:, 0],
            "base_r3_entropy": entropy.astype(np.float32, copy=False),
            "base_r3_top2_margin": (ordered[:, 2] - ordered[:, 1]).astype(np.float32, copy=False),
            "base_r3_max_probability": ordered[:, 2],
        }, index=X.index)


def current_r3_class(
    *,
    robust_clear_event: np.ndarray | pd.Series,
    lower_touch_minute: np.ndarray | pd.Series,
    label_valid: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return frozen R3 classes: adverse / weak-or-unresolved / robust clear.

    A robust clear dominates because it is defined from the pre-adverse path.
    Otherwise any realised lower touch is adverse; valid residual paths are
    weak/unresolved.  Invalid paths retain ``-1`` and are rejected before a
    broad fit rather than becoming a weak economic outcome.
    """
    clear = pd.to_numeric(pd.Series(robust_clear_event), errors="coerce").to_numpy(np.float64)
    lower = pd.to_numeric(pd.Series(lower_touch_minute), errors="coerce").to_numpy(np.float64)
    valid_raw = pd.Series(label_valid)
    valid = valid_raw.astype(bool).to_numpy() if valid_raw.dtype == bool else (
        pd.to_numeric(valid_raw, errors="coerce").to_numpy(np.float64) == 1.0
    )
    if len(clear) != len(lower) or len(clear) != len(valid):
        raise StageIVR3FitterError("R3 target inputs must be row-aligned")
    if (valid & (~np.isfinite(clear) | ~np.isfinite(lower))).any():
        raise StageIVR3FitterError("valid R3 rows require robust-clear and lower-touch inputs")
    output = np.full(len(clear), -1, dtype=np.int8)
    output[valid] = 1
    output[valid & (lower >= 0.0)] = 0
    output[valid & (clear == 1.0)] = 2
    return output


def current_r3_tree_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen current-R3 manifest then remove owned semantics."""
    frozen = dict(params)
    objective = str(frozen.pop("objective", "multiclass")).lower()
    classes = int(frozen.pop("num_class", 3))
    if objective != "multiclass" or classes != 3:
        raise StageIVR3FitterError("frozen broad params are not the current three-class R3 model")
    return frozen


def current_r3_broad_tail_fitter() -> ModelFitter:
    """Return one fitter for R3 broad + canonical tail/residual rankers."""

    def fit(
        X: pd.DataFrame,
        y: np.ndarray,
        weight: np.ndarray,
        layer: str,
        params: Mapping[str, Any],
    ) -> Any:
        if layer != "broad":
            return _default_lgbm_fitter(X, y, weight, layer, params)
        labels = np.asarray(y, dtype=np.int8).reshape(-1)
        if len(labels) != len(X) or set(np.unique(labels)) != {0, 1, 2}:
            raise StageIVR3FitterError("every current-R3 broad fold requires adverse, weak and clear support")
        import lightgbm as lgb

        frozen = dict(params)
        forbidden = {
            "objective", "num_class", "class_weight", "is_unbalance",
            "scale_pos_weight", "__stage_iv_ranker_groups",
        }.intersection(frozen)
        if forbidden:
            raise StageIVR3FitterError(
                "current-R3 broad params must be ordinary tree parameters; "
                f"the fitter owns semantic fields {sorted(forbidden)}"
            )
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, random_state=20260803,
            verbosity=-1, **frozen,
        )
        model.fit(X, labels, sample_weight=np.asarray(weight, dtype=np.float32))
        if tuple(map(int, model.classes_)) != (0, 1, 2):
            raise StageIVR3FitterError("current R3 model class order drifted from adverse/weak/clear")
        return _R3OpportunityPredictor(model)

    return fit


__all__ = [
    "R3_CLASS_ORDER", "R3_OPPORTUNITY_SCORE", "StageIVR3FitterError",
    "current_r3_tree_params",
    "current_r3_class", "current_r3_broad_tail_fitter",
]
