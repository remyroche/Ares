from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_r3_cell_day_admission import (
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
    apply_cell_day_trim15_admission_snapshot,
)
from extreme_price_movements.strict_r3_ev_bridge import fit_strict_r3_ev_bridge


def _reserve(activation: pd.Timestamp) -> pd.DataFrame:
    decision = pd.date_range(activation - pd.Timedelta(days=28), periods=28 * 20, freq="72min")
    score = np.linspace(0.0, 1.0, len(decision))
    day_effect = np.repeat(np.linspace(-100.0, 300.0, 28), 20)
    return pd.DataFrame({
        "candidate_id": [f"reserve-{index}" for index in range(len(decision))],
        "__decision_ts__": decision,
        "side_name": "long",
        "final_score": score,
        "policy_net_bps": day_effect + score * 50.0,
        "policy_label_available_ts": decision + pd.Timedelta(hours=12),
        "policy_path_valid": True,
        "ev_score_family_id": "family",
        "geometry_bundle_sha256": "geometry",
        "conversion_bundle_sha256": "conversion",
        "upstream_bundle_sha256": "upstream",
        "stack_is_prequential": True,
    })


def _current(day: str) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["live-low", "live-high"],
        "__decision_ts__": pd.to_datetime([f"{day}T12:00:00Z"] * 2, utc=True),
        "side_name": "long",
        "final_score": [0.05, 0.95],
        "ev_score_family_id": "family",
        "geometry_bundle_sha256": "geometry",
        "conversion_bundle_sha256": "conversion",
        "upstream_bundle_sha256": "upstream",
        "stack_is_prequential": True,
    })


def test_cell_day_bundle_persists_fixed_score_reference_and_equal_day_seed() -> None:
    activation = pd.Timestamp("2026-01-01T00:00:00Z")
    reserve = _reserve(activation)
    bundle = fit_strict_r3_ev_bridge(
        reserve, fit_cutoff=activation,
        producer_lineage={
            "conversion_bundle_sha256": "conversion",
            "upstream_bundle_sha256": "upstream",
        },
    )

    expected = int((reserve["policy_label_available_ts"] < activation).sum())
    assert len(bundle.side_score_references["long"]) == expected
    assert bundle.cell_day_seed["__day__"].nunique() == 28
    assert bundle.cell_day_seed["__cell__"].nunique() == 20


def test_cell_day_snapshot_is_causal_monotone_and_uses_50bps_floor() -> None:
    activation = pd.Timestamp("2026-01-01T00:00:00Z")
    reserve = _reserve(activation)
    bundle = fit_strict_r3_ev_bridge(
        reserve, fit_cutoff=activation,
        producer_lineage={
            "conversion_bundle_sha256": "conversion",
            "upstream_bundle_sha256": "upstream",
        },
    )
    # A post-activation row whose outcome resolves later on the snapshot day
    # is deliberately extreme; strict midnight availability must exclude it.
    resolved = reserve.iloc[:0].copy()
    late = _current("2026-01-02").iloc[[1]].copy()
    late["candidate_id"] = "late-outcome"
    late["policy_net_bps"] = -100000.0
    late["policy_label_available_ts"] = pd.Timestamp("2026-01-03T06:00:00Z")
    late["policy_path_valid"] = True
    resolved = pd.concat([resolved, late], ignore_index=True, sort=False)

    output, audit = apply_cell_day_trim15_admission_snapshot(
        resolved_score_ledger=resolved,
        current_scores=_current("2026-01-03"),
        bundle=bundle,
    )

    assert output["ev_mapping_vintage_mode"].eq(
        CELL_DAY_TRIM_15_CALIBRATION_MODE,
    ).all()
    assert output["causal_21d_side_expected_net_bps"].is_monotonic_increasing
    assert np.array_equal(
        output["causal_21d_side_admitted_ge_50bps"].to_numpy(bool),
        output["causal_21d_side_expected_net_bps"].ge(50.0).to_numpy(bool),
    )
    assert audit["strictly_prior_resolved"].all()


def test_cell_day_snapshot_rejects_another_producer() -> None:
    activation = pd.Timestamp("2026-01-01T00:00:00Z")
    reserve = _reserve(activation)
    bundle = fit_strict_r3_ev_bridge(
        reserve, fit_cutoff=activation,
        producer_lineage={
            "conversion_bundle_sha256": "conversion",
            "upstream_bundle_sha256": "upstream",
        },
    )
    current = _current("2026-01-03")
    current["upstream_bundle_sha256"] = "other"
    with pytest.raises(ValueError, match="mismatched upstream_bundle_sha256"):
        apply_cell_day_trim15_admission_snapshot(
            resolved_score_ledger=reserve.iloc[:0].copy(),
            current_scores=current,
            bundle=bundle,
        )
