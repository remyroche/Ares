from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.cofiring_economic_clusters import (
    discover_best_contract,
    materialize_memberships,
    pairwise_cofiring_similarity,
)
from scripts.run_tp6_sl4_cofiring_cluster_vcgam_oof import _fit_vc_gam


def test_cofiring_similarity_and_membership_are_finite() -> None:
    abs_share = pd.DataFrame(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        columns=["f0", "f1", "f2"],
    )
    sim, pair, _ = pairwise_cofiring_similarity(abs_share, abs_share, np.array([10.0, -5.0, 20.0, -2.0]))
    assert sim.shape == (3, 3)
    assert len(pair) == 3
    assert np.isfinite(sim).all()
    assert np.allclose(np.diag(sim), 1.0)


def test_cluster_discovery_uses_validation_differentiation_and_returns_contract() -> None:
    rng = np.random.default_rng(4)
    x = (rng.random((300, 8)) > 0.72).astype(float)
    x[:, 0] = (rng.random(300) > 0.45).astype(float)
    x[:, 1] = x[:, 0]
    abs_train = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
    validation = pd.DataFrame((rng.random((120, 8)) > 0.72).astype(float), columns=abs_train.columns)
    validation["f0"] = (rng.random(120) > 0.45).astype(float)
    validation["f1"] = validation["f0"]
    residual = rng.normal(0.0, 50.0, len(abs_train))
    contracts, audit, _, diff = discover_best_contract(
        abs_train,
        abs_train,
        residual,
        validation,
        rng.normal(0.0, 50.0, len(validation)),
        k_values=(2, 3, 4),
    )
    assert contracts
    assert audit["valid_contract"].any()
    assert len(diff) == len(contracts)
    features = materialize_memberships(abs_train, contracts)
    assert np.isfinite(features.to_numpy(float)).all()


def test_varying_coefficient_gam_uses_exposure_weight_not_membership_target() -> None:
    rng = np.random.default_rng(8)
    n_train, n_held = 300, 80
    train = pd.DataFrame({"x": rng.normal(size=n_train)})
    held = pd.DataFrame({"x": rng.normal(size=n_held)})
    exposure = np.abs(train["x"].to_numpy()) + 0.25
    held_exposure = np.abs(held["x"].to_numpy()) + 0.25
    membership = np.clip(exposure / 2.0, 0.0, 1.0)
    # Ordinary residual target: it is not multiplied by membership.
    residual = 30.0 * exposure * (1.0 + 0.4 * train["x"].to_numpy()) + rng.normal(0.0, 5.0, n_train)
    pred, _ = _fit_vc_gam(train, held, ["x"], exposure, held_exposure, membership, residual)
    assert len(pred) == n_held
    assert np.isfinite(pred).all()
    assert np.std(pred) > 0.0

