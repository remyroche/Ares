from __future__ import annotations

import pandas as pd

from extreme_price_movements.stage_i_target_promotion import decide_round3_promotion


def _row(arm: str, *, top10: float, top1: float = 10.0, top5: float = 10.0,
         robust: float = 1.0, worst_era: float = 1.0, worst_side: float = 1.0,
         worst_regime: float = 1.0, latest: float = 1.0, violations: float = 0.0,
         weight_mode: str = "uniform") -> dict:
    return {
        "arm": arm, "weight_mode": weight_mode, "pooled_top10_net_bps": top10,
        "pooled_top1_net_bps": top1, "pooled_top5_net_bps": top5,
        "robust_top10_lift_score": robust, "worst_era_top10_net_bps": worst_era,
        "worst_side_top10_net_bps": worst_side,
        "worst_regime_top10_net_bps": worst_regime,
        "latest_era_top10_net_bps": latest,
        "mapped_ev_monotonicity_violations": violations,
    }


def test_all_three_families_survive_when_challengers_trade_away_robustness() -> None:
    scorecard = pd.DataFrame([
        _row("R3_frozen_control", top10=10.0),
        _row("S__sl2_tp3", top10=11.0, worst_regime=.9),
        _row("O_a0p25__sl2_tp3", top10=12.0, top5=9.0),
    ])
    decision = decide_round3_promotion(scorecard, source_contract={"scorecard_sha256": "a" * 64})
    assert "selected" not in decision
    assert all(not comparison["passed"] for comparison in decision["base_diagnostic_comparisons"])
    assert {row["arm"] for row in decision["finalists"]} == {
        "R3_frozen_control", "S__sl2_tp3", "O_a0p25__sl2_tp3"
    }
    assert all(row["must_advance_to_joint_base_meta_evaluation"] for row in decision["finalists"])


def test_only_best_family_configuration_is_shortlisted_without_base_promotion() -> None:
    scorecard = pd.DataFrame([
        _row("R3_frozen_control", top10=10.0),
        _row("S__sl2_tp3", top10=11.0),
        _row("S__sl3_tp4", top10=9.0, top1=100.0),
        _row("O_a0p25__sl2_tp3", top10=10.5, worst_side=.5),
    ])
    decision = decide_round3_promotion(scorecard, source_contract={"scorecard_sha256": "a" * 64})
    assert "selected" not in decision
    assert [row["arm"] for row in decision["finalists"]] == [
        "R3_frozen_control", "S__sl2_tp3", "O_a0p25__sl2_tp3"
    ]
    s, o = decision["base_diagnostic_comparisons"]
    assert s["passed"] is True and o["passed"] is False
    assert decision["gate_contract"]["scorecard_scope"].startswith("round3_identical")


def test_strict_primary_gate_rejects_a_tie_even_when_all_other_metrics_match() -> None:
    scorecard = pd.DataFrame([
        _row("R3_frozen_control", top10=10.0),
        _row("S__sl2_tp3", top10=10.0),
        _row("O_a0p25__sl2_tp3", top10=9.0),
    ])
    decision = decide_round3_promotion(scorecard, source_contract={"scorecard_sha256": "a" * 64})
    assert "selected" not in decision
    assert decision["base_diagnostic_comparisons"][0]["checks"][1]["passed"] is False


def test_family_shortlist_uses_the_exact_diagnostic_weight_variant() -> None:
    scorecard = pd.DataFrame([
        _row("R3_frozen_control", top10=10.0),
        _row("S__sl2_tp3", top10=11.0, weight_mode="uniform"),
        # This variant ties on the leading metrics, but fails a robustness
        # gate.  Lexical/tie ordering must not let it borrow admission from
        # the passing uniform configuration.
        _row(
            "S__sl2_tp3", top10=11.0, worst_era=0.5,
            weight_mode="hybrid",
        ),
        _row("O_a0p25__sl2_tp3", top10=9.0),
    ])
    decision = decide_round3_promotion(
        scorecard, source_contract={"scorecard_sha256": "a" * 64}
    )
    assert decision["base_diagnostic_comparisons"][0]["passed"] is True
    assert decision["base_diagnostic_comparisons"][0]["weight_mode"] == "uniform"
    assert "selected" not in decision
    scalar = next(row for row in decision["finalists"] if row["family"] == "scalar_S")
    assert scalar["arm"] == "S__sl2_tp3"
    assert scalar["weight_mode"] == "uniform"
