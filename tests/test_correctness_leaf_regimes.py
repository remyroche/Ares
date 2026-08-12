import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.correctness_leaf_regimes import (
    LeafRule,
    aggregate_membership,
    cluster_rules,
    medoid,
    membership_dynamics,
    cluster_state_dynamics,
    rule_similarity,
)
from scripts.run_correctness_leaf_regime_oof import _represent


def _rule(name, conditions, effect=20.):
    return LeafRule(name, tuple(conditions), effect)


def test_similarity_hard_gates_and_medoid() -> None:
    left = _rule("a", (("funding_z", 1, .1), ("spread_z", -1, 1.0)))
    right = _rule("b", (("funding_z", 1, .15), ("spread_z", -1, 1.1)))
    conflict = _rule("c", (("funding_z", -1, .1), ("spread_z", -1, 1.0)))
    m = np.linspace(.1, .9, 100)
    score = rule_similarity(left, right, left_membership=m, right_membership=m * .9)
    assert score.hard_gate_pass and score.total >= .70
    assert not rule_similarity(left, conflict, left_membership=m, right_membership=m).hard_gate_pass
    clusters, table = cluster_rules([left, right, conflict], {"a": m, "b": m * .9, "c": m}, minimum_similarity=.70)
    assert sorted(map(tuple, clusters)) == [("a", "b"), ("c",)]
    assert medoid(["a", "b"], table) == "a"


def test_membership_aggregations_and_dynamics_are_causal_shapes() -> None:
    values = np.array([[.2, .8], [.4, .6]], dtype=float)
    assert np.allclose(aggregate_membership(values, [1, 1], mode="G0_geometric"), [np.sqrt(.08), np.sqrt(.48)])
    assert (aggregate_membership(values, [1, 2], mode="G1_weighted_geometric") >= 0).all()
    assert (aggregate_membership(values, [1, 2], mode="G2_generalized_pminus2") >= 0).all()
    assert (aggregate_membership(values, [1, 2], mode="G3_softmin") >= 0).all()
    frame = pd.DataFrame({"__ts__": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"), "side_name": ["long"] * 4, "m": [.4, .7, .9, .85]})
    out = membership_dynamics(frame, ["m"])
    assert set(["m__velocity_1h", "m__velocity_3h", "m__acceleration", "m__smoothed_membership", "m__hours_active_above_p80", "activation_mass", "activation_entropy"]).issubset(out)
    assert out.loc[2, "m__hours_active_above_p80"] == 1


def test_cluster_state_dynamics_exposes_relative_activation_not_just_posteriors() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "side_name": ["long"] * 4,
            "cluster_a": [.8, .7, .2, .1],
            "cluster_b": [.1, .2, .8, .9],
        }
    )
    out = cluster_state_dynamics(frame, ["cluster_a", "cluster_b"])
    expected = {
        "cluster_state_activation_mass",
        "cluster_state_entropy",
        "cluster_state_top1_probability",
        "cluster_state_top2_margin",
        "cluster_state_dominant_id",
        "cluster_state_switch",
        "cluster_state_switch_probability",
        "cluster_state_age_hours",
    }
    assert expected.issubset(out.columns)
    assert out.loc[2, "cluster_state_switch"] == 1
    assert out.loc[3, "cluster_state_age_hours"] == 2


def test_cluster_representation_returns_all_meta_candidate_families() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": range(8),
            "__ts__": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "side_name": ["long"] * 8,
            "trend_score": np.linspace(-1.0, 1.0, 8),
            "vol_score": np.linspace(.5, 1.2, 8),
        }
    )
    rules = [
        _rule("r0", (("trend_score", 1, -.2), ("vol_score", 1, .55)), effect=.20),
        _rule("r1", (("trend_score", 1, -.1), ("vol_score", 1, .60)), effect=.18),
    ]
    # Rule clusters are formed from this training-safe reference surface; its
    # values are intentionally independent of the held-out ``frame`` labels.
    reference = {"r0": np.linspace(.1, .9, 64), "r1": np.linspace(.09, .81, 64)}
    rep, rule_rows, _, candidate_fields, _ = _represent(frame, rules, reference, "long", "row", 2, minimum_similarity=.70)
    expected = {
        "leafreg__row__f2__slong__c00__G0_geometric",
        "leafreg__row__f2__slong__c00__signed_contribution",
        "leafreg__row__f2__slong__c00__total_contribution_share",
        "leafreg__row__f2__slong__c00__historical_support",
        "leafreg__row__f2__slong__c00__structural_stability",
        "cluster_state_entropy",
        "cluster_state_top2_margin",
        "cluster_state_switch_probability",
        "cluster_state_age_hours",
    }
    assert expected.issubset(candidate_fields)
    assert expected.issubset(rep.columns)
    # Human-readable signature directions must agree with the executable
    # +1 / >= leaf conditions above.
    assert "trend_score:hi" in rule_rows.cluster_signature.iloc[0]
    assert "vol_score:hi" in rule_rows.cluster_signature.iloc[0]
