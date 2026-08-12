from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from extreme_price_movements.leaf_reasoning_grouped_mda import (
    GroupedMDAConfig,
    LeafReasoningGroupedMDAError,
    materialize_leaf_reasoning_grouped_mda,
    write_immutable_leaf_reasoning_grouped_mda,
)
from extreme_price_movements.leaf_reasoning_meta_funnel import (
    FROZEN_META_CONTROL_FEATURES,
    ClusterTaxonomyContract,
    FrozenMetaModelSpec,
    MetaFunnelConfig,
    run_leaf_reasoning_meta_funnel,
    write_immutable_meta_funnel_output,
)


def _spec() -> FrozenMetaModelSpec:
    return FrozenMetaModelSpec(
        family="lightgbm_lgbmregressor",
        params={"objective": "huber", "n_estimators": 9},
        contract_id="grouped_mda_test_huber_v1",
    )


def _ridge(_: FrozenMetaModelSpec) -> Ridge:
    return Ridge(alpha=1.0)


def _ledger() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for transport_index, transport in enumerate(("transport_a", "transport_b")):
        count = 200
        decision = pd.date_range(
            "2024-01-01" if transport_index == 0 else "2024-06-01",
            periods=count,
            freq="h",
            tz="UTC",
        )
        signal = np.linspace(-1.0, 1.0, count)
        base = np.linspace(-5.0, 5.0, count)
        frame = pd.DataFrame({
            "candidate_id": [f"{transport}_{number:03d}" for number in range(count)],
            "side_name": np.where(np.arange(count) % 2, "short", "long"),
            "decision_ts": decision,
            "label_available_ts": decision + pd.Timedelta(hours=2),
            "base_oof_fit_end_ts": decision - pd.Timedelta(hours=1),
            "base_oof_generated_ts": decision,
            "base_same_side_strict_oof": True,
            "base_expected_bps": base,
            "realized_net_bps": base + 60.0 * signal,
            "reasoning_a": signal,
            "transport": transport,
            "meta_partition": np.where(np.arange(count) < 120, "inner_oof", "outer_test"),
            "fold_id": np.where(np.arange(count) < 120, f"{transport}_inner", f"{transport}_outer"),
        })
        frame["realized_cost_bps"] = 2.0
        frame["realized_gross_bps"] = frame["realized_net_bps"] + frame["realized_cost_bps"]
        for index, feature in enumerate(FROZEN_META_CONTROL_FEATURES):
            if feature not in frame:
                frame[feature] = float(index + 1) + signal * float(index + 1) / 100.0
        frame["p_adverse"] = 0.2 - .05 * signal
        frame["p_weak"] = 0.3 - .02 * signal
        frame["p_clear"] = 1.0 - frame["p_adverse"] - frame["p_weak"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _sealed_funnel(tmp_path, ledger: pd.DataFrame):
    result = run_leaf_reasoning_meta_funnel(
        ledger,
        feature_groups={"L0": FROZEN_META_CONTROL_FEATURES, "L1": ("reasoning_a",)},
        model_spec=_spec(),
        model_factory=_ridge,
        config=MetaFunnelConfig(min_train_rows=16, fit_protocol="transport_outer_frozen"),
        stages=("L",),
    )
    return write_immutable_meta_funnel_output(
        result,
        tmp_path / "funnel",
        config=MetaFunnelConfig(min_train_rows=16, fit_protocol="transport_outer_frozen"),
    )


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cluster_root_and_c_funnel(tmp_path, ledger: pd.DataFrame):
    """Build a tiny immutable external C table; the source ledger keeps no C fields."""

    root = tmp_path / "clusters"
    root.mkdir()
    identity = ["candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition"]
    clusters = ledger.loc[:, identity].copy()
    mapping = {f"C{number}::test": f"cluster_{number}" for number in range(1, 5)}
    signal = np.linspace(-1.0, 1.0, len(clusters))
    for number, field in enumerate(mapping.values(), start=1):
        clusters[field] = signal * number
    clusters.to_parquet(root / "candidate_cluster_features.parquet", index=False)
    groups = {
        "C0": [*FROZEN_META_CONTROL_FEATURES, "reasoning_a"],
        **{f"C{number}": [f"cluster_{number}"] for number in range(1, 5)},
    }
    taxonomy = {
        "selection_phase": "threshold_sweep", "linkage": "average",
        "cluster_ids_by_arm": {f"C{number}": [f"C{number}::test"] for number in range(1, 5)},
        "threshold_by_arm": {"C1": .6, "C2": .7, "C3": .8, "C4": .9},
        "exploratory_hard_cap": 20, "production_soft_cap": 12,
    }
    (root / "cluster_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    (root / "cluster_taxonomy_contract.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    (root / "cluster_feature_manifest.json").write_text(json.dumps({
        "schema": "leaf_reasoning_candidate_cluster_materializer_v1", "cluster_id_to_feature": mapping,
    }), encoding="utf-8")
    names = ["candidate_cluster_features.parquet", "cluster_groups.json", "cluster_taxonomy_contract.json", "cluster_feature_manifest.json"]
    (root / "manifest.json").write_text(json.dumps({
        "schema": "leaf_reasoning_candidate_cluster_materializer_v1",
        "status": "STRICT_OOF_CANDIDATE_CLUSTER_FEATURES_MATERIALIZED",
        "outputs": {name: _sha(root / name) for name in names},
    }), encoding="utf-8")
    joined = ledger.merge(clusters, on=identity, validate="one_to_one")
    config = MetaFunnelConfig(min_train_rows=16, fit_protocol="transport_outer_frozen")
    result = run_leaf_reasoning_meta_funnel(
        joined,
        feature_groups={"L0": FROZEN_META_CONTROL_FEATURES, "L2": ("reasoning_a",), "L3": ()},
        cluster_groups=groups,
        cluster_taxonomy=ClusterTaxonomyContract(
            linkage="average", cluster_ids_by_arm=taxonomy["cluster_ids_by_arm"],
            threshold_by_arm=taxonomy["threshold_by_arm"], selection_phase="threshold_sweep",
        ),
        model_spec=_spec(), model_factory=_ridge, config=config, stages=("C",),
    )
    return root, write_immutable_meta_funnel_output(result, tmp_path / "c_funnel", config=config)


def test_grouped_mda_is_transport_local_strict_and_joinable(tmp_path) -> None:
    ledger = _ledger()
    funnel = _sealed_funnel(tmp_path, ledger)
    result = materialize_leaf_reasoning_grouped_mda(
        ledger,
        funnel_root=funnel,
        config=GroupedMDAConfig(repeats=2, phantom_draws=8, random_seed=19),
        model_factory=_ridge,
    )
    summary = result.summary
    assert set(summary.arm) == {"L1"}
    assert set(summary.transport_id) == {"transport_a", "transport_b"}
    assert summary.real_repeat_count.eq(2).all()
    assert summary.phantom_draw_count.eq(8).all()
    assert summary.strict_prior_resolved.all()
    assert summary.ranking_scope.eq("one_pooled_global_post_common_bps_top_k_per_transport").all()
    assert result.real_repeats.strict_prior_resolved.all()
    assert result.phantom_draws.strict_prior_resolved.all()
    l1_gate = result.gates.loc[result.gates.arm.eq("L1")].iloc[0]
    assert l1_gate.grouped_transport_mda_evidence_present
    assert l1_gate.grouped_transport_mda_status in {"PASS", "FAIL"}
    assert {"median_transport_mda_bps", "stable_transport_mda_bps", "phantom_q95_bps"}.issubset(result.advancement.columns)

    output = write_immutable_leaf_reasoning_grouped_mda(result, tmp_path / "mda")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "STRICT_CHRONOLOGICAL_TRANSPORT_GROUPED_MDA_COMPLETE"
    assert (output / "advancement_evidence.parquet").is_file()
    assert pd.read_parquet(output / "grouped_mda_summary.parquet").equals(summary)


def test_grouped_mda_rejects_a_ledger_whose_outer_economics_do_not_match_the_sealed_funnel(tmp_path) -> None:
    ledger = _ledger()
    funnel = _sealed_funnel(tmp_path, ledger)
    altered = ledger.copy()
    outer = altered.meta_partition.eq("outer_test")
    altered.loc[outer, "realized_net_bps"] += 1.0
    altered.loc[outer, "realized_gross_bps"] += 1.0
    with pytest.raises(LeafReasoningGroupedMDAError, match="does not reproduce sealed funnel"):
        materialize_leaf_reasoning_grouped_mda(
            altered,
            funnel_root=funnel,
            config=GroupedMDAConfig(repeats=2, phantom_draws=8),
            model_factory=_ridge,
        )


def test_projected_parquet_source_matches_dataframe_fixture(tmp_path) -> None:
    """The production path must preserve the exact strict MDA calculation."""

    ledger = _ledger()
    funnel = _sealed_funnel(tmp_path, ledger)
    config = GroupedMDAConfig(repeats=2, phantom_draws=8, random_seed=791)
    in_memory = materialize_leaf_reasoning_grouped_mda(
        ledger, funnel_root=funnel, config=config, model_factory=_ridge,
    )
    ledger_path = tmp_path / "wide_ledger.parquet"
    # Extra fields prove the parquet path projects selected contracts rather
    # than loading the whole source table.
    wide = ledger.assign(unused_wide_field=np.arange(len(ledger), dtype=float))
    wide.to_parquet(ledger_path, index=False)
    projected = materialize_leaf_reasoning_grouped_mda(
        ledger_path, funnel_root=funnel, config=config, model_factory=_ridge,
    )
    pd.testing.assert_frame_equal(projected.summary, in_memory.summary)
    pd.testing.assert_frame_equal(projected.real_repeats, in_memory.real_repeats)
    pd.testing.assert_frame_equal(projected.phantom_draws, in_memory.phantom_draws)


def test_projected_parquet_source_keeps_strict_identity_validation(tmp_path) -> None:
    ledger = _ledger()
    funnel = _sealed_funnel(tmp_path, ledger)
    duplicated = pd.concat([ledger, ledger.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicate_identity.parquet"
    duplicated.to_parquet(path, index=False)
    with pytest.raises(LeafReasoningGroupedMDAError, match="one base OOF row"):
        materialize_leaf_reasoning_grouped_mda(
            path,
            funnel_root=funnel,
            config=GroupedMDAConfig(repeats=2, phantom_draws=8),
            model_factory=_ridge,
        )


def test_grouped_mda_hash_bound_projected_cluster_join_for_sealed_c_funnel(tmp_path) -> None:
    ledger = _ledger()
    cluster_root, funnel = _cluster_root_and_c_funnel(tmp_path, ledger)
    with pytest.raises(LeafReasoningGroupedMDAError, match="require --cluster-root"):
        materialize_leaf_reasoning_grouped_mda(
            ledger, funnel_root=funnel, config=GroupedMDAConfig(repeats=2, phantom_draws=8), model_factory=_ridge,
        )
    result = materialize_leaf_reasoning_grouped_mda(
        ledger, funnel_root=funnel, cluster_root=cluster_root,
        config=GroupedMDAConfig(repeats=2, phantom_draws=8, random_seed=121), model_factory=_ridge,
    )
    assert set(result.summary.arm) == {"C1", "C2", "C3", "C4"}
    assert result.cluster_source is not None
    assert result.cluster_source["candidate_cluster_features_sha256"] == _sha(cluster_root / "candidate_cluster_features.parquet")
    output = write_immutable_leaf_reasoning_grouped_mda(result, tmp_path / "c_mda")
    written = json.loads((output / "manifest.json").read_text())
    assert written["source_cluster_feature_artifact"]["manifest_sha256"] == _sha(cluster_root / "manifest.json")


def test_grouped_mda_rejects_cluster_root_with_wrong_feature_contract(tmp_path) -> None:
    ledger = _ledger()
    cluster_root, funnel = _cluster_root_and_c_funnel(tmp_path, ledger)
    payload = json.loads((cluster_root / "cluster_groups.json").read_text())
    payload["C1"] = ["cluster_2"]
    (cluster_root / "cluster_groups.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads((cluster_root / "manifest.json").read_text())
    manifest["outputs"]["cluster_groups.json"] = _sha(cluster_root / "cluster_groups.json")
    (cluster_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LeafReasoningGroupedMDAError, match="taxonomy mapping"):
        materialize_leaf_reasoning_grouped_mda(
            ledger, funnel_root=funnel, cluster_root=cluster_root,
            config=GroupedMDAConfig(repeats=2, phantom_draws=8), model_factory=_ridge,
        )
