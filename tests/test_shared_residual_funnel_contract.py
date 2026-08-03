from pathlib import Path

import pytest

from extreme_price_movements.shared_residual_funnel_contract import (
    COMMON_BPS_RECONSTRUCTION,
    EXACT_RESIDUAL_TARGET,
    SHARED_MODEL_CLASS,
    freeze_shared_residual_handoff,
    load_shared_residual_handoff,
    reconstruct_shared_common_bps,
    write_shared_residual_handoff,
)


def _manifest() -> dict:
    return {
        "selected_arm": "stage_ii_shared_residual_winner",
        "target": EXACT_RESIDUAL_TARGET,
        "reconstruction": COMMON_BPS_RECONSTRUCTION,
        "feature_list": ["base_expected_net", "causal_volatility", "trust_score"],
        "model_class": SHARED_MODEL_CLASS,
        "geometry": {"tp_atr": 6.0, "sl_atr": 4.0, "horizon_hours": 12.0},
        "cost": {"total_cost_bps": 100.0, "application_count": 1},
        "entry": {"convention": "next_bar_after_close", "signal_to_entry_hours": 1.0},
        "label_availability": {
            "signal_to_available_hours": 13.0,
            "strict_comparison": "label_available_ts < fit_cutoff",
        },
        "ranking": {"selection": "pooled_global_after_common_bps_mapping"},
        "calibration": {"kind": "causal_21_day_admission_map"},
    }


def test_frozen_handoff_consumes_and_rechecks_predecessor_winner(tmp_path: Path) -> None:
    predecessor = tmp_path / "stage_ii_winner.json"
    predecessor.write_text('{"winner":"shared"}\n', encoding="utf-8")
    contract = freeze_shared_residual_handoff(
        predecessor_artifacts=[predecessor], **_manifest()
    )
    handoff = write_shared_residual_handoff(contract, tmp_path / "stage_iii_handoff.json")
    loaded = load_shared_residual_handoff(handoff)
    assert loaded.selected_arm == "stage_ii_shared_residual_winner"
    assert loaded.predecessors[0]["path"] == str(predecessor.resolve())

    predecessor.write_text('{"winner":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_shared_residual_handoff(handoff)


def test_handoff_rejects_local_experts_and_hard_routing(tmp_path: Path) -> None:
    predecessor = tmp_path / "winner.json"
    predecessor.write_text("frozen", encoding="utf-8")
    for field, value in (
        ("model_class", "per_regime_residual_experts"),
        ("selected_arm", "local_residual_arm"),
        ("routing", "hard_routing_by_regime"),
    ):
        manifest = _manifest()
        manifest[field] = value
        with pytest.raises(ValueError, match="forbidden|one shared"):
            freeze_shared_residual_handoff(
                predecessor_artifacts=[predecessor], **manifest
            )


def test_common_bps_reconstruction_includes_regime_prior() -> None:
    score = reconstruct_shared_common_bps([10.0, -5.0], [20.0, -10.0], [3.0, 7.0])
    assert score.tolist() == [33.0, -8.0]
