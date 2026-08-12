from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_portability import PortabilityPolicy
from extreme_price_movements.feature_portability_audit import (
    ChronologicalAuditPolicy,
    run_chronological_feature_portability_audit,
)
from extreme_price_movements.leaf_reasoning_clusters import (
    LeafReasoningClusterError,
    pairwise_leaf_reasoning_similarity,
)
from extreme_price_movements.prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from extreme_price_movements.stage_i_causal_admission import (
    pooled_global_admission_comparison,
)
from extreme_price_movements.stage_i_nested_feature_challenger import (
    NestedFeatureChallengePlan,
    NestedFeatureSet,
)
from extreme_price_movements.stage_i_nested_stack_execution import (
    GuardedMetaArmSpec,
    NestedStackConfig,
    NestedStackInput,
    execute_matched_nested_stack,
)


def test_prior_resolved_value_history_excludes_its_own_and_contemporaneous_outcomes() -> None:
    decision = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-02T00:00:00Z"]
    )
    available = pd.to_datetime(
        ["2024-01-01T13:00:00Z", "2024-01-02T13:00:00Z", "2024-01-02T13:00:00Z"]
    )
    config = PrequentialR3ValueMapConfig(
        side="long", bins=3, min_global_rows=1, bin_shrink_rows=1.0
    )
    baseline, audit, provenance = prequential_same_side_r3_value_map(
        exact_net_bps=[100.0, -1_000.0, 1_000.0],
        decision_timestamps=decision,
        label_available_timestamps=available,
        side="long",
        score=[0.0, 0.0, 0.0],
        config=config,
    )
    changed, _, _ = prequential_same_side_r3_value_map(
        exact_net_bps=[100.0, 99_999.0, -99_999.0],
        decision_timestamps=decision,
        label_available_timestamps=available,
        side="long",
        score=[0.0, 0.0, 0.0],
        config=config,
    )

    np.testing.assert_allclose(baseline, changed)
    np.testing.assert_allclose(baseline[1:], [100.0, 100.0])
    assert audit.loc[1:, "prior_resolved_global_support"].eq(1).all()
    assert provenance["prior_resolution_rule"] == "label_available_ts < decision_ts"


def test_low_support_score_bin_is_shrunk_toward_prior_global_history() -> None:
    decision = pd.to_datetime(["2024-01-01T00:00:00Z"] * 4 + ["2024-01-02T00:00:00Z"])
    available = pd.to_datetime(["2024-01-01T13:00:00Z"] * 4 + ["2024-01-02T13:00:00Z"])
    mapped, audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=[0.0, 0.0, 100.0, 100.0, 0.0],
        decision_timestamps=decision,
        label_available_timestamps=available,
        side="long",
        score=[-0.9, -0.9, 0.9, 0.9, 0.9],
        config=PrequentialR3ValueMapConfig(
            side="long", bins=2, min_global_rows=1, bin_shrink_rows=3.0
        ),
    )

    # The prior bin average is 100 and the prior global average is 50.  With
    # two bin rows and three shrinkage pseudo-rows, the causal value is 70.
    assert mapped[-1] == pytest.approx(70.0)
    assert audit.loc[len(audit) - 1, "prior_resolved_bin_support"] == 2
    assert audit.loc[len(audit) - 1, "value_map_fallback"] == "shrunk_bin_prior_resolved"


def test_leaf_cluster_contract_rejects_raw_leaf_ids_and_hard_separates_side_and_head() -> None:
    summary = pd.DataFrame(
        {
            "rule_instance_id": ["long-clear", "short-clear", "long-adverse"],
            "fold_id": ["f1", "f2", "f3"],
            "side_name": ["long", "short", "long"],
            "head_name": ["clear", "clear", "adverse"],
            "contribution_direction": ["positive"] * 3,
            "rule_signature": ["same"] * 3,
            "activation_rate": [0.2] * 3,
            "contribution_signature": ["same"] * 3,
            "economic_effect": [1.0] * 3,
            "portability_score": [0.95] * 3,
        }
    )
    pairs = pairwise_leaf_reasoning_similarity(summary)
    assert not pairs["compatible"].any()
    assert pairs["compatibility_reason"].str.contains("side_name|head_name", regex=True).all()

    with pytest.raises(LeafReasoningClusterError, match="raw fold-local leaf identifiers"):
        pairwise_leaf_reasoning_similarity(summary.assign(leaf_assignment__model_00=[7, 7, 7]))


def test_low_support_portability_coverage_does_not_override_distribution_gate() -> None:
    reference = np.tile(np.arange(10, dtype=float), 10)
    late = np.concatenate([np.full(95, 9.0), np.full(5, np.nan)])
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
                    "era": "reference",
                    "ret_1h": reference,
                }
            ),
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-02-01", periods=100, freq="h", tz="UTC"),
                    "era": "later",
                    "ret_1h": late,
                }
            ),
        ],
        ignore_index=True,
    )
    policy = ChronologicalAuditPolicy(
        portability=PortabilityPolicy(
            min_coverage=0.95,
            min_finite_support=50,
            min_unique_values=2,
            max_extrapolation_rate=1.0,
            min_bin_support=1,
            min_bins_represented=1,
            min_effect_support=1,
        ),
        min_reference_rows=50,
        distribution_bins=10,
        max_era_shortcut_auc=0.65,
    )
    result = run_chronological_feature_portability_audit(
        frame,
        feature_names=["ret_1h"],
        timestamp_column="ts",
        era_column="era",
        policy=policy,
    )
    later = result.era_audit.loc[result.era_audit.era.eq("later")].iloc[0]
    assert later.coverage == pytest.approx(0.95)
    assert later.era_shortcut_auc > policy.max_era_shortcut_auc
    assert result.dispositions.loc[0, "disposition"] == "ERA_SHORTCUT"


def _nested_plan() -> NestedFeatureChallengePlan:
    feature_set = NestedFeatureSet(
        side="long",
        name="base_only",
        requested_feature_count=None,
        features=("base_x",),
        added_features=(),
        source_ranks={"base_x": 1},
        feature_families={"base_x": "base"},
        family_composition={"base": 1},
        tier_composition={"stable": 1},
        source_hash="c" * 64,
    )
    return NestedFeatureChallengePlan("long", "a" * 64, "b" * 64, "audit", (), (), {}, (feature_set,))


def test_nested_successor_oof_uses_base_only_inputs_and_direct_meta_handoff() -> None:
    count = 72
    signal = pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": np.arange(count),
            "__ts__": signal,
            "__symbol__": "BTC",
            "side_name": "long",
            "decision_ts": signal + pd.Timedelta(hours=1),
            "label_available_ts": signal + pd.Timedelta(hours=13),
            "r3_class": np.arange(count) % 3,
            "exact_net_bps": np.linspace(-120.0, 160.0, count),
            "base_x": np.arange(count, dtype=float),
            # This is an active meta context; a base fit must never receive it.
            "active_meta": np.arange(count, dtype=float),
        }
    )
    base_calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    meta_calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def base_predictor(train: pd.DataFrame, _target: np.ndarray, valid: pd.DataFrame, _set: NestedFeatureSet) -> np.ndarray:
        assert list(train.columns) == ["base_x"]
        assert list(valid.columns) == ["base_x"]
        assert train.base_x.max() < valid.base_x.min()
        base_calls.append((train.copy(), valid.copy()))
        return np.tile([0.2, 0.3, 0.5], (len(valid), 1))

    def selector(
        _train: pd.DataFrame,
        _target: np.ndarray,
        direct: tuple[str, ...],
        _mandatory: tuple[str, ...],
        _spec: GuardedMetaArmSpec,
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        return tuple(direct), {"mode": "guardrail"}

    def meta_predictor(train: pd.DataFrame, _target: np.ndarray, _weight: np.ndarray, valid: pd.DataFrame, _spec: GuardedMetaArmSpec) -> np.ndarray:
        assert "active_meta" in train and "active_meta" in valid
        assert not any("prequential" in name or "expected_net" in name for name in train)
        assert train.active_meta.max() < valid.active_meta.min()
        meta_calls.append((train.copy(), valid.copy()))
        return np.full(len(valid), 0.4)

    result = execute_matched_nested_stack(
        NestedStackInput("long", frame, ("base_x",), ("active_meta",)),
        _nested_plan(),
        base_predictor=base_predictor,
        meta_predictor=meta_predictor,
        meta_arms=(GuardedMetaArmSpec("reliability", "reliability"),),
        meta_feature_selector=selector,
        config=NestedStackConfig(n_validation_folds=4, min_base_train_rows=12, min_meta_train_rows=4),
    )
    assert base_calls and meta_calls
    assert result.meta_outputs[("base_only", "reliability")].fold_provenance.strict_prior_resolved.all()
    assert result.base_outputs["base_only"].columns.isin(["prequential_base_expected_net_bps"]).sum() == 0


def test_final_admission_ranking_is_one_pooled_book_without_side_quotas() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["l1", "l2", "s1", "s2"],
            "side_name": ["long", "long", "short", "short"],
            "__ts__": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "raw_score": [1.0, 0.9, 0.8, 0.7],
            "net_bps": [10.0, 9.0, 8.0, 7.0],
            "causal_21d_side_expected_net_bps": [400.0, 300.0, 200.0, 100.0],
            "causal_21d_side_admitted_ge_50bps": [True, True, True, True],
        }
    )
    comparison = pooled_global_admission_comparison(
        frame.sample(frac=1.0, random_state=19), raw_score_column="raw_score", top_fractions=(0.5,)
    )
    pooled = comparison.loc[
        comparison.comparison.eq("with_admission_mapped_pooled_global")
    ].iloc[0]
    assert pooled.selected_rows == 2
    assert pooled.selected_long_rows == 2
    assert pooled.selected_short_rows == 0
