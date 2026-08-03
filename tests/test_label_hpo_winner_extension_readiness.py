from __future__ import annotations

import pandas as pd

from scripts.audit_label_hpo_winner_extension_readiness import build_readiness


def _heads(*, base: bool) -> dict[str, bool]:
    return {
        "base_booster_serialized": base,
        "residual_booster_serialized": True,
        "base_ev_map_serialized": True,
        "admission_calibrator_serialized": True,
    }


def test_readiness_uses_the_earliest_required_source_and_flags_base_recovery() -> None:
    source = {
        "sides": {
            side: {"paired_signal_max": pd.Timestamp("2026-07-20 16:00", tz="UTC")}
            for side in ("long", "short")
        },
        "feature_store_max": pd.Timestamp("2026-07-21 04:00", tz="UTC"),
        "raw_15m_ohlcv_max": pd.Timestamp("2026-07-21 17:15", tz="UTC"),
    }
    artifacts = {
        "sides": {
            side: {
                "heads": _heads(base=False),
                "base_recovery_required": True,
                "extended_score_max": pd.Timestamp("2026-07-17 12:00", tz="UTC"),
            }
            for side in ("long", "short")
        },
        "policy_manifest": {},
    }
    report = build_readiness(
        source, artifacts, requested_end_inclusive="2026-07-23T23:59:59Z"
    )
    assert report["maximum_scoreable_signal_timestamp"] == pd.Timestamp(
        "2026-07-20 16:00", tz="UTC"
    )
    assert report["maximum_exact_policy_signal_timestamp"] == pd.Timestamp(
        "2026-07-20 16:00", tz="UTC"
    )
    assert not report["requested_end_is_currently_available"]
    assert "requested_end_exceeds_locally_available_causal_inputs" in report["blockers"]
    assert report["sides"]["long"]["deterministic_recovery_allowed"]
    assert not report["sides"]["long"]["frozen_scoring_without_refit_available"]
    assert report["missing_stages"][0]["stage"] == "post_recovery_feature_matrix_and_score_export"


def test_readiness_requires_all_serialized_nonbase_heads() -> None:
    source = {
        "sides": {
            side: {"paired_signal_max": pd.Timestamp("2026-07-20", tz="UTC")}
            for side in ("long", "short")
        },
        "feature_store_max": pd.Timestamp("2026-07-21", tz="UTC"),
        "raw_15m_ohlcv_max": pd.Timestamp("2026-07-21", tz="UTC"),
    }
    artifacts = {
        "sides": {
            "long": {
                "heads": _heads(base=True),
                "base_recovery_required": False,
                "extended_score_max": None,
            },
            "short": {
                "heads": {**_heads(base=False), "residual_booster_serialized": False},
                "base_recovery_required": True,
                "extended_score_max": None,
            },
        },
        "policy_manifest": {},
    }
    report = build_readiness(
        source, artifacts, requested_end_inclusive="2026-07-20T00:00:00Z"
    )
    assert report["sides"]["long"]["frozen_scoring_without_refit_available"]
    assert not report["sides"]["short"]["deterministic_recovery_allowed"]
    assert report["sides"]["short"]["missing_nonrecoverable_heads"] == [
        "residual_booster_serialized"
    ]
