from __future__ import annotations

from scripts.run_regime_geometry_portability_ablation import arm_features


def test_regime_ablation_admits_only_invariant_context_not_fold_local_state_ids() -> None:
    columns = [
        "market_regime__entropy",
        "market_regime__top2_margin",
        "market_regime__state_age_hours",
        "market_regime__state_switch_probability",
        "market_regime__ood_distance_percentile",
        "market_regime__phase_p_stable",
        "market_regime__phase_p_onset",
        "market_regime__phase_p_active",
        "market_regime__phase_p_settling",
        "market_regime__phase_entropy",
        "market_regime__phase_top2_margin",
        "market_regime__state_p_0",
        "regime_state_id",
        "geometry_regime__liquidity__entropy",
        "geometry_regime__liquidity__state_switch_probability",
        "geometry_regime__liquidity__state_p_0",
        "execution_wait_action",
    ]
    arms = arm_features(columns)
    production = {
        field
        for arm, fields in arms.items()
        if "fold_local_membership_diagnostic" not in arm
        for field in fields
    }
    selected = {field for fields in arms.values() for field in fields}
    assert "market_regime__state_p_0" not in selected
    # Fold-local geometry coordinates are intentionally available solely in a
    # named diagnostic ablation.  They are absent from every normal arm and
    # cannot be promoted without the separate coordinate-portability gate.
    assert "geometry_regime__liquidity__state_p_0" not in production
    assert "geometry_regime__liquidity__state_p_0" in arms[
        "B2_liquidity_fold_local_membership_diagnostic"
    ]
    assert "regime_state_id" not in selected
    assert "execution_wait_action" not in selected
    assert "geometry_regime__liquidity__entropy" in arms["A2_liquidity"]
