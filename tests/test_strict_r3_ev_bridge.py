from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_r3_canonical_current import (
    apply_current_admission_snapshot,
)
from extreme_price_movements.strict_r3_ev_bridge import (
    EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
    EVBridgeSpec,
    apply_strict_r3_ev_bridge,
    fit_strict_r3_ev_bridge,
    load_strict_r3_ev_bridge,
    persist_strict_r3_ev_bridge,
)


FAMILY = "frozen-score-family"
GEOMETRY = "frozen-geometry"


def _resolved(start: str, rows: int, *, producer: str = "old") -> pd.DataFrame:
    decision = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    score = np.linspace(0.01, 0.99, rows)
    return pd.DataFrame({
        "candidate_id": [f"{producer}-{index}" for index in range(rows)],
        "__decision_ts__": decision,
        "side_name": "long",
        "final_score": score,
        "policy_net_bps": -100.0 + 400.0 * score,
        "policy_label_available_ts": decision + pd.Timedelta(hours=12),
        "conversion_bundle_sha256": f"conversion-{producer}",
        "upstream_bundle_sha256": f"upstream-{producer}",
        "geometry_bundle_sha256": GEOMETRY,
        "ev_score_family_id": FAMILY,
        "stack_is_prequential": True,
    })


def _bridge() -> object:
    ledger = _resolved("2025-01-01", 160)
    return fit_strict_r3_ev_bridge(
        ledger,
        fit_cutoff="2025-02-01T00:00:00Z",
        spec=EVBridgeSpec(minimum_residual_rows=20),
    )


def test_bridge_maps_a_fresh_producer_without_any_same_vintage_rows() -> None:
    bundle = _bridge()
    resolved = _resolved("2025-01-01", 160)
    decision = pd.Timestamp("2025-03-01T00:00:00Z")
    current = pd.DataFrame({
        "candidate_id": ["fresh-0", "fresh-1"],
        "__decision_ts__": [decision, decision],
        "side_name": ["long", "long"],
        "final_score": [0.90, 0.15],
        "conversion_bundle_sha256": ["conversion-fresh"] * 2,
        "upstream_bundle_sha256": ["upstream-fresh"] * 2,
        "geometry_bundle_sha256": [GEOMETRY] * 2,
        "ev_score_family_id": [FAMILY] * 2,
        "stack_is_prequential": [True] * 2,
    })

    mapped, _ = apply_current_admission_snapshot(
        resolved_score_ledger=resolved,
        current_scores=current,
        ev_bridge_bundle=bundle,
    )

    assert mapped["policy_net_bps"].isna().all()
    assert mapped["causal_21d_side_expected_net_bps"].notna().all()
    assert mapped.loc[mapped.candidate_id.eq("fresh-0"), "causal_21d_side_admitted_ge_50bps"].item()
    assert not mapped.loc[mapped.candidate_id.eq("fresh-1"), "causal_21d_side_admitted_ge_50bps"].item()
    assert set(mapped["ev_mapping_vintage_mode"]) == {
        "strict_oof_common_bps_bridge_plus_causal_residual_v1",
    }
    # There are no resolved observations from the fresh producer itself.  The
    # correction may still use earlier *common-bps residuals* from the frozen
    # score family, which is precisely the no-drought behavior.
    assert set(mapped["ev_bridge_residual_mapping_status"]).issubset({
        "bridge_prior_only_no_recent_residual_support",
        "bridge_prior_plus_causal_residual",
    })


def test_bridge_residual_correction_is_prequential_and_ignores_future_rows() -> None:
    bundle = _bridge()
    early = _resolved("2025-02-01", 96, producer="early")
    future = _resolved("2025-04-01", 96, producer="future")
    baseline, audit = apply_strict_r3_ev_bridge(early, bundle=bundle)
    appended, appended_audit = apply_strict_r3_ev_bridge(
        pd.concat([early, future], ignore_index=True), bundle=bundle,
    )

    left = baseline.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right = appended.loc[appended.candidate_id.str.startswith("early-")].sort_values(
        "candidate_id", kind="stable",
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        left["causal_21d_side_expected_net_bps"],
        right["causal_21d_side_expected_net_bps"],
    )
    assert audit["strictly_prior_resolved"].all()
    assert appended_audit["strictly_prior_resolved"].all()


def test_bridge_rejects_a_semantically_different_geometry() -> None:
    bundle = _bridge()
    frame = _resolved("2025-02-01", 24)
    frame["geometry_bundle_sha256"] = "different-geometry"
    try:
        apply_strict_r3_ev_bridge(frame, bundle=bundle)
    except ValueError as error:
        assert "geometry/K9" in str(error)
    else:  # pragma: no cover - defensive readability
        raise AssertionError("bridge must reject a changed geometry semantic contract")


def test_immediate_bridge_rejects_a_different_producer_lineage() -> None:
    ledger = _resolved("2025-01-01", 160, producer="reserve")
    bundle = fit_strict_r3_ev_bridge(
        ledger,
        fit_cutoff="2025-02-01T00:00:00Z",
        producer_lineage={
            "conversion_bundle_sha256": "conversion-reserve",
            "upstream_bundle_sha256": "upstream-reserve",
        },
    )
    current = _resolved("2025-03-01", 12, producer="different")
    with pytest.raises(ValueError, match="mismatched conversion_bundle_sha256"):
        apply_strict_r3_ev_bridge(current, bundle=bundle)


def test_exact_producer_reserve_uses_its_own_auditable_calibration_mode() -> None:
    ledger = _resolved("2025-01-01", 160, producer="reserve")
    bundle = fit_strict_r3_ev_bridge(
        ledger,
        fit_cutoff="2025-02-01T00:00:00Z",
        producer_lineage={
            "conversion_bundle_sha256": "conversion-reserve",
            "upstream_bundle_sha256": "upstream-reserve",
        },
    )
    mapped, _ = apply_strict_r3_ev_bridge(
        _resolved("2025-03-01", 12, producer="reserve"), bundle=bundle,
    )
    assert bundle.calibration_mode == EXACT_PRODUCER_RESERVE_CALIBRATION_MODE
    assert set(mapped["ev_mapping_vintage_mode"]) == {
        EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
    }


def test_immediate_snapshot_filters_residual_history_to_its_exact_producer() -> None:
    """A reserve map must not inherit residuals from another model vintage."""
    reserve = _resolved("2025-01-01", 160, producer="reserve")
    bundle = fit_strict_r3_ev_bridge(
        reserve,
        fit_cutoff="2025-02-01T00:00:00Z",
        producer_lineage={
            "conversion_bundle_sha256": "conversion-reserve",
            "upstream_bundle_sha256": "upstream-reserve",
        },
    )
    same_producer = _resolved("2025-02-01", 48, producer="reserve")
    foreign_producer = _resolved("2025-02-01", 48, producer="foreign")
    current = pd.DataFrame({
        "candidate_id": ["live"],
        "__decision_ts__": pd.to_datetime(["2025-03-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
        "final_score": [0.90],
        "conversion_bundle_sha256": ["conversion-reserve"],
        "upstream_bundle_sha256": ["upstream-reserve"],
        "geometry_bundle_sha256": [GEOMETRY],
        "ev_score_family_id": [FAMILY],
        "stack_is_prequential": [True],
    })

    output, _ = apply_current_admission_snapshot(
        resolved_score_ledger=pd.concat([same_producer, foreign_producer], ignore_index=True),
        current_scores=current,
        ev_bridge_bundle=bundle,
    )

    assert output["candidate_id"].tolist() == ["live"]
    assert set(output["ev_mapping_vintage_mode"]) == {
        EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
    }
    # The foreign rows would make an exact-producer bridge reject the frame if
    # they reached the residual correction.  A successful mapping therefore
    # proves the snapshot narrowed to the reserve producer first.
    assert output["causal_21d_side_expected_net_bps"].notna().all()


def test_bridge_round_trip_preserves_frozen_identity(tmp_path) -> None:
    bundle = _bridge()
    manifest = persist_strict_r3_ev_bridge(bundle, tmp_path / "bridge")
    loaded = load_strict_r3_ev_bridge(tmp_path / "bridge")
    assert loaded.fit_cutoff == bundle.fit_cutoff
    assert loaded.ev_score_family_id == FAMILY
    assert loaded.geometry_bundle_sha256 == GEOMETRY
    assert loaded.manifest["bundle_sha256"] == manifest["bundle_sha256"]
