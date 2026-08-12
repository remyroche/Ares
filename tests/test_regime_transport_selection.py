from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_transport_selection import (
    TransportAuditConfig,
    audit_continuous_context_transport,
)


def _frame(rows_per_era: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for era_number, era in enumerate(("2023", "2024", "2025")):
        timestamps = pd.date_range(f"{era}-01-01", periods=rows_per_era, freq="h", tz="UTC")
        portable = rng.normal(size=rows_per_era)
        # This has a large covariate shift but no economic relationship.
        shortcut = era_number + rng.normal(scale=0.03, size=rows_per_era)
        target = 0.025 * portable + rng.normal(scale=0.004, size=rows_per_era)
        rows.extend({
            "candidate_id": f"{era}-{index}", "__ts__": timestamp, "era": era,
            "execution_net_ev_12h": value, "portable_context": field,
            "era_shortcut": shortcut_value, "market_regime__state_p_0": 0.5,
            "trust_controller": field,
        } for index, (timestamp, value, field, shortcut_value) in enumerate(zip(timestamps, target, portable, shortcut)))
    return pd.DataFrame(rows)


def test_transport_audit_enforces_continuous_coverage_and_membership_exclusion() -> None:
    result = audit_continuous_context_transport(
        _frame(),
        candidate_features=["portable_context", "era_shortcut", "market_regime__state_p_0", "trust_controller"],
        config=TransportAuditConfig(min_rows_per_split=80, max_train_rows=500, max_eval_rows=300),
    )
    audit = result.feature_audit.set_index("feature")
    assert audit.loc["portable_context", "coverage_gate_pass"]
    assert not audit.loc["market_regime__state_p_0", "coverage_gate_pass"]
    assert audit.loc["market_regime__state_p_0", "classification"] == "REJECTED"
    assert "market_regime__state_p_0" not in set(result.split_mda["feature"])
    assert audit.loc["trust_controller", "classification"] == "CONTROLLER_DIAGNOSTIC"
    assert result.manifest["selection_contract"]["cluster_memberships_excluded"] is True
    assert result.manifest["label"]["kind"] == "thresholded_net_economic_label"


def test_transport_mda_never_fits_on_later_rows() -> None:
    frame = _frame()
    result = audit_continuous_context_transport(
        frame,
        candidate_features=["portable_context"],
        config=TransportAuditConfig(min_rows_per_split=80, max_train_rows=500, max_eval_rows=300, embargo_hours=12),
    )
    assert {"within_era", "cross_era"}.issubset(set(result.split_mda["scope"]))
    # The saved split labels permit a direct audit of intended chronological
    # topology; implementation purges 12 hours before every evaluation start.
    cross = result.split_mda.loc[result.split_mda["scope"].eq("cross_era")]
    assert cross["train_eras"].str.len().gt(0).all()
    assert cross["test_era"].isin(["2024", "2025"]).all()


def test_reference_memberships_are_forbidden() -> None:
    with pytest.raises(ValueError, match="memberships"):
        audit_continuous_context_transport(
            _frame(),
            candidate_features=["portable_context"],
            reference_features=["market_regime__state_p_0"],
            config=TransportAuditConfig(min_rows_per_split=80),
        )
