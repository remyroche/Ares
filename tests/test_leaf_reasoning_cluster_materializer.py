from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.leaf_reasoning_meta_funnel import (
    ClusterTaxonomyContract,
    build_sequential_arms,
)
from extreme_price_movements.leaf_reasoning_cluster_materializer import (
    IDENTITY,
    LeafReasoningClusterFinalizationConfig,
    LeafReasoningClusterMaterializerError,
    _candidate_cluster_features,
    _c5_ids_from_coverage,
    _family_cache,
    _validate_compact_lineage,
    _cluster_coverage_scores,
    finalize_leaf_reasoning_cluster_taxonomy,
    load_finalized_leaf_reasoning_cluster_artifact,
    load_leaf_reasoning_cluster_candidate_artifact,
    merge_strict_cluster_candidate_features,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(index: int, *, transport: str = "transport_a") -> dict[str, object]:
    return {
        "candidate_id": f"candidate-{index}",
        "decision_ts": pd.Timestamp("2024-02-01", tz="UTC") + pd.Timedelta(hours=index),
        "side_name": "long" if index % 2 == 0 else "short",
        "fold_id": f"{transport}_outer",
        "transport": transport,
        "meta_partition": "outer_test",
    }


def _cluster_root(tmp_path: Path) -> Path:
    root = tmp_path / "clusters"
    root.mkdir()
    candidates = pd.DataFrame([_candidate(0), _candidate(1)])
    ids = {
        "C1": ["c1"], "C2": ["c2"], "C3": ["c3"], "C4": ["c4"],
        "C5": ["c1"], "C6": ["c1"],
    }
    mapping = {"c1": "cluster_1", "c2": "cluster_2", "c3": "cluster_3", "c4": "cluster_4"}
    for number, field in enumerate(mapping.values(), start=1):
        candidates[field] = float(number)
    tables = {
        "candidate_cluster_features.parquet": candidates,
        "cluster_rule_summary.parquet": pd.DataFrame({"rule_signature": ["safe"]}),
        "cluster_assignments.parquet": pd.DataFrame({"rule_signature": ["safe"]}),
        "cluster_summary.parquet": pd.DataFrame({"cluster_id": ["safe"]}),
        "cluster_selection_audit.parquet": pd.DataFrame({"selection_stage": ["C1"]}),
        "cluster_contribution_mass.parquet": pd.DataFrame({"arm": ["C1"]}),
    }
    outputs: dict[str, str] = {}
    for name, table in tables.items():
        path = root / name
        table.to_parquet(path, index=False)
        outputs[name] = _sha(path)
    groups = {
        "C0": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "base_reasoning__g2_safe", "base_reasoning__g3_safe"],
        **{arm: [mapping[value] for value in values] for arm, values in ids.items()},
    }
    taxonomy = {
        "linkage": "average",
        "cluster_ids_by_arm": ids,
        "threshold_by_arm": {"C1": .60, "C2": .70, "C3": .80, "C4": .90},
        "c5_source_arm": "C1",
        "c6_source_arm": "C5",
        "top_decile_coverage_target": .95,
        "top_decile_coverage_by_arm": {"C5": .95},
        "portable_top_decile_coverage_by_arm": {"C5": .95},
        "production_soft_cap": 12,
        "exploratory_hard_cap": 20,
        "c6_best_cross_era_score": .10,
        "c6_best_cross_era_standard_error": .02,
        "c6_compact_cross_era_score": .09,
    }
    payloads = {
        "cluster_groups.json": groups,
        "cluster_taxonomy_contract.json": taxonomy,
        "cluster_feature_manifest.json": {
            "schema": "leaf_reasoning_candidate_cluster_materializer_v1",
            "cluster_id_to_feature": mapping,
        },
    }
    for name, value in payloads.items():
        path = root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        outputs[name] = _sha(path)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "leaf_reasoning_candidate_cluster_materializer_v1",
        "status": "STRICT_OOF_CANDIDATE_CLUSTER_FEATURES_MATERIALIZED",
        "strict_manifest_sha256": "strict-hash",
        "transports": ["transport_a"],
        "outputs": outputs,
    }, sort_keys=True), encoding="utf-8")
    return root


def _threshold_candidate_root(tmp_path: Path) -> Path:
    """A complete C1--C4 root with two transports and one cluster per arm."""
    root = _cluster_root(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "cluster_feature_manifest.json").read_text(encoding="utf-8"))["cluster_id_to_feature"]
    ids = {arm: [f"c{number}"] for number, arm in enumerate(("C1", "C2", "C3", "C4"), start=1)}
    groups = {
        "C0": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "base_reasoning__g2_safe", "base_reasoning__g3_safe"],
        **{arm: [mapping[values[0]]] for arm, values in ids.items()},
    }
    taxonomy = {
        "selection_phase": "threshold_sweep",
        "linkage": "average",
        "cluster_ids_by_arm": ids,
        "threshold_by_arm": {"C1": .60, "C2": .70, "C3": .80, "C4": .90},
        "production_soft_cap": 12,
        "exploratory_hard_cap": 20,
    }
    cluster_summary = pd.DataFrame([
        {"arm": arm, "cluster_id": f"{arm}::{value[0]}", "fold_coverage_fraction": 1.0, "economic_effect_mean": .1}
        for arm, value in ids.items()
    ])
    # The IDs in the fake mapping are short, but the actual candidate clusters
    # remain arm-qualified just like the production materializer.
    taxonomy["cluster_ids_by_arm"] = {
        arm: [f"{arm}::{values[0]}"] for arm, values in ids.items()
    }
    feature_manifest = {f"{arm}::{value[0]}": mapping[value[0]] for arm, value in ids.items()}
    selection = pd.DataFrame([
        {"selection_stage": arm, "selected": True, "cluster_id": f"{arm}::{value[0]}", "top_abs_contribution": 10.0}
        for arm, value in ids.items()
    ])
    mass = pd.DataFrame([
        {
            "arm": arm, "cluster_id": f"{arm}::{value[0]}", "transport": transport,
            "is_top_decile": True, "abs_contribution": 10.0,
        }
        for arm, value in ids.items() for transport in ("transport_a", "transport_b")
    ])
    tables = {
        "cluster_summary.parquet": cluster_summary,
        "cluster_selection_audit.parquet": selection,
        "cluster_contribution_mass.parquet": mass,
    }
    for name, table in tables.items():
        path = root / name
        table.to_parquet(path, index=False)
        manifest["outputs"][name] = _sha(path)
    payloads = {
        "cluster_groups.json": groups,
        "cluster_taxonomy_contract.json": taxonomy,
        "cluster_feature_manifest.json": {
            "schema": "leaf_reasoning_candidate_cluster_materializer_v1",
            "cluster_id_to_feature": feature_manifest,
        },
    }
    for name, value in payloads.items():
        path = root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        manifest["outputs"][name] = _sha(path)
    manifest["transports"] = ["transport_a", "transport_b"]
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root


def test_cluster_artifact_loader_verifies_hashes_mapping_and_exact_feature_schema(tmp_path: Path) -> None:
    root = _cluster_root(tmp_path)
    artifact = load_leaf_reasoning_cluster_candidate_artifact(root)
    assert list(artifact.candidate_features.columns[:len(IDENTITY)]) == list(IDENTITY)
    assert artifact.groups["C5"] == ["cluster_1"]
    assert artifact.taxonomy["cluster_ids_by_arm"]["C6"] == ["c1"]

    path = root / "candidate_cluster_features.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "cluster_1"] = 99.0
    frame.to_parquet(path, index=False)
    with pytest.raises(LeafReasoningClusterMaterializerError, match="hash mismatch"):
        load_leaf_reasoning_cluster_candidate_artifact(root)


def test_threshold_sweep_contract_exposes_only_c0_to_c4_before_finalization() -> None:
    taxonomy = ClusterTaxonomyContract(
        linkage="average",
        selection_phase="threshold_sweep",
        cluster_ids_by_arm={arm: (arm.lower(),) for arm in ("C1", "C2", "C3", "C4")},
    )
    groups = {
        "C0": ("p_adverse", "p_weak", "p_clear", "base_expected_bps", "base_reasoning__g2_safe", "base_reasoning__g3_safe"),
        **{arm: (f"{arm.lower()}_feature",) for arm in ("C1", "C2", "C3", "C4")},
    }
    arms = build_sequential_arms({"L0": groups["C0"][:4]}, groups, cluster_taxonomy=taxonomy)
    assert {arm.arm for arm in arms if arm.stage == "C"} == {"C0", "C1", "C2", "C3", "C4"}


def test_c0_compact_root_must_descend_from_the_same_strict_root(tmp_path: Path) -> None:
    strict = tmp_path / "strict"
    strict.mkdir()
    index = strict / "strict_oof_reasoning_artifact_index.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(index, index=False)
    (strict / "base_prediction_shards").mkdir()
    compact = tmp_path / "compact"
    compact.mkdir()
    manifest = {
        "inputs": {
            "artifact_index": str(index),
            "prediction_shards_root": str(strict / "base_prediction_shards"),
        }
    }
    _validate_compact_lineage(strict, compact, manifest)
    manifest["inputs"]["artifact_index"] = str(tmp_path / "other" / "index.parquet")
    with pytest.raises(LeafReasoningClusterMaterializerError, match="does not match"):
        _validate_compact_lineage(strict, compact, manifest)


def test_cluster_join_is_exact_and_never_overwrites_the_ledger() -> None:
    cluster = pd.DataFrame([_candidate(0), _candidate(1)]).assign(cluster_safe=[.1, .2])
    ledger = pd.DataFrame([_candidate(0), _candidate(1)]).assign(base_expected_bps=[1.0, 2.0])
    joined = merge_strict_cluster_candidate_features(ledger, cluster)
    assert joined.cluster_safe.tolist() == [.1, .2]

    with pytest.raises(LeafReasoningClusterMaterializerError, match="identities are not exact"):
        merge_strict_cluster_candidate_features(ledger.iloc[:1], cluster)
    with pytest.raises(LeafReasoningClusterMaterializerError, match="overwrite"):
        merge_strict_cluster_candidate_features(ledger, cluster.assign(base_expected_bps=1.0))


def test_candidate_cluster_features_keep_transport_scoped_duplicate_candidate_ids(tmp_path: Path) -> None:
    """A stable candidate ID may be re-scored by independent transports."""

    population = pd.DataFrame([
        _candidate(0, transport="transport_a"),
        _candidate(0, transport="transport_b"),
    ]).assign(base_expected_bps=[1.0, 2.0])
    cache = tmp_path / "family.parquet"
    pd.DataFrame([
        {**_candidate(0, transport="transport_a"), "head_name": "p_clear", "contribution_direction": "positive", "rule_signature": "safe", "rule_instance_id": "rule_a", "is_top_decile": True, "family_ensemble_tree_contribution": 1.25},
        {**_candidate(0, transport="transport_b"), "head_name": "p_clear", "contribution_direction": "positive", "rule_signature": "safe", "rule_instance_id": "rule_b", "is_top_decile": True, "family_ensemble_tree_contribution": 2.50},
    ]).to_parquet(cache, index=False)
    assignments = pd.DataFrame([
        {"rule_instance_id": "rule_a", "cluster_id": "C1::safe"},
        {"rule_instance_id": "rule_b", "cluster_id": "C1::safe"},
    ])
    features, names = _candidate_cluster_features(
        population=population,
        cache_paths=[cache],
        assignments=assignments,
        selected={"C1": ["C1::safe"], "C2": [], "C3": [], "C4": []},
    )
    field = names["C1::safe"]
    assert features.loc[features["transport"].eq("transport_a"), field].tolist() == [1.25]
    assert features.loc[features["transport"].eq("transport_b"), field].tolist() == [2.50]
    assert not features.duplicated(list(IDENTITY)).any()


def test_family_cache_joins_top_decile_flags_on_full_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-decile flags must not collide when the same ID appears in two transports."""

    module = importlib.import_module("extreme_price_movements.leaf_reasoning_cluster_materializer")
    population = pd.DataFrame([
        _candidate(0, transport="transport_a"),
        _candidate(0, transport="transport_b"),
    ]).assign(base_expected_bps=[1.0, 2.0])
    index = pd.DataFrame([
        {"transport": "transport_a", "fold_id": "transport_a_outer", "side_model": "long", "head_name": "p_clear", "fold_name": "outer", "artifact_dir": "artifact_a"},
        {"transport": "transport_b", "fold_id": "transport_b_outer", "side_model": "long", "head_name": "p_clear", "fold_name": "outer", "artifact_dir": "artifact_b"},
    ])
    rule_summary = pd.DataFrame([
        {"transport": row.transport, "fold_id": row.fold_id, "side_name": "long", "head_name": "p_clear", "contribution_direction": "positive", "rule_signature": "safe", "rule_instance_id": f"rule_{row.transport}"}
        for row in index.itertuples(index=False)
    ])

    def _fake_extract(path: Path) -> pd.DataFrame:
        transport = "transport_a" if path.name == "artifact_a" else "transport_b"
        candidate = _candidate(0, transport=transport)
        return pd.DataFrame([{
            "candidate_id": candidate["candidate_id"], "__ts__": candidate["decision_ts"],
            "side_name": "long", "fold_id": candidate["fold_id"], "head_name": "p_clear",
            "rule_signature": "safe", "contribution_direction": "positive",
            "family_ensemble_tree_contribution": 1.0,
        }])

    monkeypatch.setattr(module, "extract_leaf_family_contributions", _fake_extract)
    (tmp_path / "cache").mkdir()
    paths, _ = _family_cache(
        root=tmp_path, index=index, population=population,
        rule_summary=rule_summary, cache_dir=tmp_path / "cache",
    )
    cached = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    assert len(cached) == 2
    assert set(cached["transport"]) == {"transport_a", "transport_b"}
    assert not cached.duplicated(list(IDENTITY)).any()


def test_cluster_coverage_scores_count_a_missing_transport_as_zero_coverage() -> None:
    clusters = pd.DataFrame([
        {
            "arm": arm,
            "cluster_id": f"{arm}::one",
            "fold_coverage_fraction": 1.0,
            "economic_effect_mean": .1,
        }
        for arm in ("C1", "C2", "C3", "C4")
    ])
    # Every cluster is present in transport A, while transport B has no active
    # family contribution at all.  It cannot silently be ignored by the 95%
    # all-transport C5 coverage gate.
    mass = pd.DataFrame([
        {
            "arm": arm,
            "cluster_id": f"{arm}::one",
            "transport": "transport_a",
            "is_top_decile": True,
            "abs_contribution": 1.0,
        }
        for arm in ("C1", "C2", "C3", "C4")
    ])
    _coverage, score = _cluster_coverage_scores(
        clusters=clusters,
        mass=mass,
        transports=("transport_a", "transport_b"),
    )
    assert score["min_transport_coverage"].eq(0.).all()


def test_post_c1_c4_selector_uses_development_mda_then_c5_prefix_one_se(tmp_path: Path) -> None:
    candidate_root = _threshold_candidate_root(tmp_path)
    metric_rows: list[dict[str, object]] = []
    mda_rows: list[dict[str, object]] = []
    for number, arm in enumerate(("C1", "C2", "C3", "C4"), start=1):
        for transport in ("transport_a", "transport_b"):
            for tail in (.05, .10):
                metric_rows.append({
                    "arm": arm,
                    "transport_id": transport,
                    "top_fraction": tail,
                    "incremental_global_top_k_net_bps_vs_control": float(number),
                    "worst_month_net_bps_delta_vs_control": -.5,
                })
            # C2 is the stable-MDA winner even though later thresholds have
            # larger raw development tail lifts.
            mda_rows.append({
                "arm": arm,
                "transport_id": transport,
                "cluster_ids_json": json.dumps([f"{arm}::c{number}"]),
                "transport_mda_bps": {"C1": 2., "C2": 5., "C3": 3., "C4": 4.}[arm],
                "phantom_q95_bps": 1.,
                "positive_environment_rate": .8,
            })
    prefix_mda = pd.DataFrame([
        {
            "prefix_size": 1,
            "transport_id": transport,
            "cluster_ids_json": json.dumps(["C2::c2"]),
            "transport_mda_bps": 5.,
            "phantom_q95_bps": 1.,
            "positive_environment_rate": .8,
        }
        for transport in ("transport_a", "transport_b")
    ])
    final_root = finalize_leaf_reasoning_cluster_taxonomy(
        candidate_root,
        pd.DataFrame(metric_rows),
        pd.DataFrame(mda_rows),
        prefix_mda,
        tmp_path / "final",
    )
    final = load_finalized_leaf_reasoning_cluster_artifact(final_root)
    assert final.taxonomy["selection_phase"] == "final"
    assert final.taxonomy["c5_source_arm"] == "C2"
    assert final.taxonomy["cluster_ids_by_arm"]["C5"] == ["C2::c2"]
    assert final.taxonomy["cluster_ids_by_arm"]["C6"] == ["C2::c2"]
    assert final.groups["C5"] == ["cluster_2"]
    assert {
        "coverage_manifest.json", "portable_manifest.json", "diagnostic_manifest.json",
        "c5_contribution_coverage_report.parquet",
    }.issubset({path.name for path in final_root.iterdir()})
    report = pd.read_parquet(final_root / "c5_contribution_coverage_report.parquet")
    assert report.loc[:, [
        "portable_contribution_coverage", "unstable_contribution_coverage",
        "unmatched_contribution_coverage",
    ]].columns.tolist() == [
        "portable_contribution_coverage", "unstable_contribution_coverage",
        "unmatched_contribution_coverage",
    ]
    assert json.loads((final_root / "portable_manifest.json").read_text(encoding="utf-8"))["cluster_ids"] == ["C2::c2"]


def test_c5_reports_portable_unstable_and_unmatched_contribution_separately(tmp_path: Path) -> None:
    root = _threshold_candidate_root(tmp_path)
    artifact = load_leaf_reasoning_cluster_candidate_artifact(root)
    clusters = pd.read_parquet(root / "cluster_summary.parquet")
    mass = pd.read_parquet(root / "cluster_contribution_mass.parquet")
    extras = pd.DataFrame([
        {"arm": "C2", "cluster_id": "C2::unstable", "fold_coverage_fraction": .2, "economic_effect_mean": .1},
        {"arm": "C2", "cluster_id": "C2::unmatched", "fold_coverage_fraction": .9, "economic_effect_mean": .1},
    ])
    clusters = pd.concat([clusters, extras], ignore_index=True)
    c2_rows = mass["arm"].eq("C2")
    mass.loc[c2_rows, "abs_contribution"] = 100.0
    extra_mass = pd.DataFrame([
        {"arm": "C2", "cluster_id": cluster, "transport": transport, "is_top_decile": True, "abs_contribution": value}
        for cluster, value in (("C2::unstable", 2.0), ("C2::unmatched", 1.0))
        for transport in ("transport_a", "transport_b")
    ])
    clusters.to_parquet(root / "cluster_summary.parquet", index=False)
    pd.concat([mass, extra_mass], ignore_index=True).to_parquet(root / "cluster_contribution_mass.parquet", index=False)
    changed_ids = dict(artifact.taxonomy["cluster_ids_by_arm"])
    changed_ids["C2"] = ["C2::c2", "C2::unstable"]
    selection = _c5_ids_from_coverage(
        replace(artifact, taxonomy={**artifact.taxonomy, "cluster_ids_by_arm": changed_ids}),
        "C2",
        config=LeafReasoningClusterFinalizationConfig(),
    )
    report = selection.coverage_report.set_index("transport")
    assert selection.selected_ids == ["C2::c2"]
    assert report["portable_contribution_coverage"].gt(.95).all()
    assert report["unstable_contribution_coverage"].gt(0.0).all()
    assert report["unmatched_contribution_coverage"].gt(0.0).all()
    assert set(selection.diagnostic_manifest["coverage_class"]) == {"unstable", "unmatched"}


def test_post_c1_c4_selector_refuses_to_name_c5_without_grouped_mda_gates(tmp_path: Path) -> None:
    candidate_root = _threshold_candidate_root(tmp_path)
    metrics = pd.DataFrame([
        {
            "arm": arm, "transport_id": transport, "top_fraction": tail,
            "incremental_global_top_k_net_bps_vs_control": 1.,
            "worst_month_net_bps_delta_vs_control": 0.,
        }
        for arm in ("C1", "C2", "C3", "C4")
        for transport in ("transport_a", "transport_b")
        for tail in (.05, .10)
    ])
    mda = pd.DataFrame([
        {
            "arm": arm, "transport_id": transport,
            "cluster_ids_json": json.dumps([f"{arm}::c{number}"]),
            "transport_mda_bps": .5, "phantom_q95_bps": 1.,
            "positive_environment_rate": .8,
        }
        for number, arm in enumerate(("C1", "C2", "C3", "C4"), start=1)
        for transport in ("transport_a", "transport_b")
    ])
    prefix = pd.DataFrame([
        {"prefix_size": 1, "transport_id": transport, "cluster_ids_json": json.dumps(["C1::c1"]), "transport_mda_bps": 2., "phantom_q95_bps": 1., "positive_environment_rate": .8}
        for transport in ("transport_a", "transport_b")
    ])
    with pytest.raises(LeafReasoningClusterMaterializerError, match="no C1--C4 threshold passed"):
        finalize_leaf_reasoning_cluster_taxonomy(candidate_root, metrics, mda, prefix, tmp_path / "final")


def test_meta_cli_cluster_root_joins_only_matching_immutable_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _cluster_root(tmp_path)
    ledger_root = tmp_path / "ledger"
    ledger_root.mkdir()
    ledger = pd.DataFrame([_candidate(0), _candidate(1)]).assign(
        p_adverse=.2, p_weak=.3, p_clear=.5, base_expected_bps=2.0,
    )
    ledger_path = ledger_root / "base_to_meta_reasoning_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    (ledger_root / "meta_ledger_manifest.json").write_text(json.dumps({
        "status": "STRICT_BASE_TO_META_LEDGER_ASSEMBLED",
        "transports": ["transport_a"],
        "source_hashes": {"strict_manifest_00": "strict-hash"},
    }), encoding="utf-8")
    feature_groups = tmp_path / "groups.json"
    feature_groups.write_text(json.dumps({"L0": ["p_adverse", "p_weak", "p_clear", "base_expected_bps"]}), encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"family": "lightgbm_lgbmregressor", "contract_id": "test", "params": {"objective": "huber"}}), encoding="utf-8")

    module = importlib.import_module("scripts.run_leaf_reasoning_meta_funnel")
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "run_leaf_reasoning_meta_funnel", lambda frame, **kwargs: captured.update({"frame": frame, **kwargs}) or object())
    monkeypatch.setattr(module, "write_immutable_meta_funnel_output", lambda *_args, **_kwargs: tmp_path / "out")
    monkeypatch.setattr("sys.argv", [
        "run_leaf_reasoning_meta_funnel.py",
        "--ledger", str(ledger_path),
        "--feature-groups", str(feature_groups),
        "--cluster-root", str(root),
        "--stages", "C",
        "--frozen-model-spec", str(model),
        "--output-root", str(tmp_path / "out-root"),
    ])
    assert module.main() == 0
    assert "cluster_1" in captured["frame"]
    assert captured["cluster_groups"]["C1"] == ("cluster_1",)
    assert captured["cluster_taxonomy"].c5_source_arm == "C1"
