from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.en_uncertainty_ebm import (
    compute_uncertainty_features,
    fit_en_uncertainty_adjuster,
    fit_uncertainty_state,
    uncertainty_weighted_prediction,
)


class DummyEBM:
    term_features_ = [(0,), (1,), (0, 1)]
    term_scores_ = [
        np.asarray([-0.2, 0.0, 0.3, 0.5], dtype=np.float32),
        np.asarray([0.1, -0.1, 0.2, 0.4], dtype=np.float32),
        np.asarray([0.0, 0.1, -0.1, 0.2], dtype=np.float32),
    ]

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift

    def eval_terms(self, X: pd.DataFrame) -> np.ndarray:
        x0 = X["x0"].to_numpy(dtype=np.float32)
        x1 = X["x1"].to_numpy(dtype=np.float32)
        return np.column_stack(
            [
                0.4 * x0 + self.shift,
                -0.2 * x1,
                0.1 * x0 * x1,
            ]
        ).astype(np.float32)


def _raw_predict(model: DummyEBM, X: pd.DataFrame, mode: str) -> np.ndarray:
    logits = model.eval_terms(X).sum(axis=1)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def test_compute_uncertainty_features_are_finite_and_prefixed() -> None:
    X = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, 64, dtype=np.float32),
            "x1": np.linspace(1.0, -1.0, 64, dtype=np.float32),
        }
    )
    state = fit_uncertainty_state(X, ["x0", "x1"], max_bins=8)
    feats = compute_uncertainty_features(
        X,
        [DummyEBM(0.0), DummyEBM(0.05)],
        "classifier",
        _raw_predict,
        state=state,
    )
    assert len(feats) == len(X)
    assert "ebm_unc_conflict" in feats.columns
    assert "ebm_unc_pi_width" in feats.columns
    assert np.isfinite(feats.to_numpy(dtype=np.float32)).all()


def test_en_adjuster_predicts_probability_vector() -> None:
    rng = np.random.default_rng(42)
    n = 128
    X = pd.DataFrame(
        {
            "ebm_unc_conflict": rng.normal(size=n).astype(np.float32),
            "ebm_unc_entropy_mean": rng.uniform(0.1, 0.7, size=n).astype(np.float32),
            "ebm_unc_uncertainty_weight": rng.uniform(0.2, 1.0, size=n).astype(
                np.float32
            ),
        }
    )
    base = np.clip(rng.uniform(0.05, 0.95, size=n), 1e-4, 1 - 1e-4).astype(np.float32)
    y = (base + 0.1 * X["ebm_unc_uncertainty_weight"].to_numpy() > 0.55).astype(
        np.float32
    )
    adjuster = fit_en_uncertainty_adjuster(
        base,
        y,
        X,
        random_state=42,
        n_trials=3,
    )
    assert adjuster is not None
    pred = adjuster.predict(base, X)
    weighted = uncertainty_weighted_prediction(base, X, pred)
    assert pred.shape == base.shape
    assert weighted.shape == base.shape
    assert np.all((pred > 0.0) & (pred < 1.0))
    assert np.all((weighted > 0.0) & (weighted < 1.0))
