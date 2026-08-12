from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from extreme_price_movements.strict_event_store import (
    StrictEventStoreError,
    build_strict_event_store,
    iter_event_store_parts,
    iter_predecessor_selection_pairs,
    load_strict_event_store,
    source_parts_for_cutoff,
)
from extreme_price_movements.causal_leaf_health_vectorized import (
    materialize_strict_oof_causal_leaf_health_vectorized,
)
from extreme_price_movements.causal_leaf_health_scoped import (
    materialize_strict_oof_causal_leaf_health_scoped,
)
from extreme_price_movements.causal_leaf_health_event_incremental import (
    _materialise_h4_h5_bounded,
    materialize_strict_oof_causal_leaf_health_event_incremental,
)
from extreme_price_movements.causal_leaf_health import CausalLeafHealthConfig, _materialise_h4_h5
from extreme_price_movements.strict_contribution_event_stream import (
    EVENT_COLUMNS,
    build_strict_contribution_event_streams,
    load_strict_contribution_event_streams,
)
from extreme_price_movements.strict_contribution_event_reader import (
    iter_contribution_event_groups,
    iter_contribution_event_timestamp_blocks,
)


CONTRACT = "f" * 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_event_sidecar_index(root: Path, index: pd.DataFrame) -> None:
    """Rewrite an intentionally tampered sidecar index and reseal its hash.

    This lets focused loader tests exercise provenance/type checks after the
    outer index-integrity check, rather than merely failing on the modified
    index bytes.
    """

    index_path = root / "contribution_event_stream_parts.parquet"
    index.to_parquet(index_path, index=False, compression="zstd")
    manifest_path = root / "strict_contribution_event_stream_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["parts"] = _sha256(index_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _spool(tmp_path: Path, *, raw_leaf: bool = False, shared_inner_family: bool = False) -> Path:
    """A minimal sealed spool with one early and one unresolved-at-cutoff row."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    strict_root = tmp_path / "strict"
    strict_root.mkdir()
    strict_manifest = strict_root / "strict_oof_reasoning_manifest.json"
    strict_manifest.write_text(json.dumps({
        "status": "STRICT_OOF_BASE_REASONING_MATERIALIZED",
        "transports": ["transport_a"],
    }))

    spool = tmp_path / "spool"
    candidates_dir = spool / "candidate_parts"
    contributions_dir = spool / "contribution_parts"
    candidates_dir.mkdir(parents=True)
    contributions_dir.mkdir()
    decision = pd.to_datetime([
        "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z", "2024-01-01T00:30:00Z",
    ], utc=True)
    candidates = pd.DataFrame({
        "candidate_id": ["early", "late", "outer"],
        "decision_ts": decision,
        "feature_generation_ts": decision,
        # The second row is not resolved at the 04:00 cutoff despite having a
        # decision time before it.  This catches accidental decision-time
        # rather than resolution-time source filtering.
        "label_available_ts": pd.to_datetime([
            "2024-01-01T01:00:00Z", "2024-01-01T08:00:00Z", "2024-01-01T01:30:00Z",
        ], utc=True),
        "side_name": ["long", "long", "long"],
        "head_name": ["p_clear", "p_clear", "p_clear"],
        "fold_id": ["fold_0", "fold_0", "fold_0"],
        "transport": ["transport_a", "transport_a", "transport_a"],
        "meta_partition": ["inner_oof", "inner_oof", "outer_test"],
        "feature_contract_sha256": [CONTRACT, CONTRACT, CONTRACT],
        "semantic_label": [1.0, 0.0, 1.0],
        "head_prediction": [0.7, 0.3, 0.8],
        "net_bps": [50.0, -50.0, 100.0],
        "base_expected_bps": [20.0, -20.0, 30.0],
        "asset": ["A", "B", "O"],
    })
    contributions = pd.DataFrame({
        "candidate_id": ["early", "late", "outer"],
        "__ts__": decision,
        "side_name": ["long", "long", "long"],
        "head_name": ["p_clear", "p_clear", "p_clear"],
        "fold_id": ["fold_0", "fold_0", "fold_0"],
        "transport": ["transport_a", "transport_a", "transport_a"],
        "meta_partition": ["inner_oof", "inner_oof", "outer_test"],
        "feature_contract_sha256": [CONTRACT, CONTRACT, CONTRACT],
        "rule_signature": [
            "shared_inner" if shared_inner_family else "early_family",
            "shared_inner" if shared_inner_family else "late_family",
            "outer_family",
        ],
        "contribution_direction": ["positive", "positive" if shared_inner_family else "negative", "positive"],
        "family_ensemble_tree_contribution": [0.3, -0.2, 0.4],
        **({"leaf_token": [1, 2, 3]} if raw_leaf else {}),
    })
    candidate_path = candidates_dir / "part_0000.parquet"
    contribution_path = contributions_dir / "part_0000.parquet"
    candidates.to_parquet(candidate_path, index=False)
    contributions.to_parquet(contribution_path, index=False)
    index = pd.DataFrame([{
        "part": 0,
        "candidate_part": candidate_path.name,
        "contribution_part": contribution_path.name,
        "candidate_rows": len(candidates),
        "contribution_rows": len(contributions),
        "candidate_sha256": _sha256(candidate_path),
        "contribution_sha256": _sha256(contribution_path),
    }])
    index.to_parquet(spool / "strict_family_input_parts.parquet", index=False)
    (spool / "strict_family_input_spool_manifest.json").write_text(json.dumps({
        "schema": "strict_oof_family_input_spool_v1",
        "status": "STRICT_OOF_FAMILY_INPUT_SPOOL_COMPLETED",
        "strict_roots": [str(strict_root)],
        "strict_root_manifest_sha256": {str(strict_root): _sha256(strict_manifest)},
        "pair_index": "strict_family_input_parts.parquet",
    }))
    return spool


def _spool_with_multiple_family_rows(tmp_path: Path) -> Path:
    """Make one candidate contribute to two token-free family rows.

    Candidate and contribution physical identity digests are deliberately
    different in this valid shape: the latter hashes a larger family-row
    population.  It catches any invalid equality assertion between the two.
    """

    spool = _spool(tmp_path)
    contribution_path = spool / "contribution_parts" / "part_0000.parquet"
    contribution = pd.read_parquet(contribution_path)
    extra = contribution.iloc[[0]].copy()
    extra["rule_signature"] = "second_early_family"
    extra["contribution_direction"] = "negative"
    extra["family_ensemble_tree_contribution"] = -0.1
    contribution = pd.concat([contribution, extra], ignore_index=True)
    contribution.to_parquet(contribution_path, index=False)
    index_path = spool / "strict_family_input_parts.parquet"
    index = pd.read_parquet(index_path)
    index.loc[index.index[0], "contribution_rows"] = len(contribution)
    index.loc[index.index[0], "contribution_sha256"] = _sha256(contribution_path)
    index.to_parquet(index_path, index=False)
    return spool


def _spool_multi_month_h3(tmp_path: Path) -> Path:
    """Three months that exercise frozen H2 and H3 snapshots.

    January labels are available before February begins, while February labels
    resolve before the first later-in-February score.  A correct monthly H3
    snapshot must therefore use January only throughout February; the
    vectorised reference is the oracle for both that rule and the H2 period
    close semantics.
    """

    spool = _spool(tmp_path)
    candidate_path = spool / "candidate_parts" / "part_0000.parquet"
    contribution_path = spool / "contribution_parts" / "part_0000.parquet"
    decision = pd.to_datetime([
        "2024-01-02T00:00:00Z", "2024-01-05T00:00:00Z",
        "2024-01-08T00:00:00Z", "2024-01-11T00:00:00Z",
        # Its label resolves exactly at the February boundary.  It is a
        # legal H1 predecessor for February candidates, but H3's monthly
        # freeze must keep it out because the reference uses strictly
        # label_available_ts < score_month_start.
        "2024-01-31T23:00:00Z",
        "2024-02-02T00:00:00Z", "2024-02-05T00:00:00Z",
        "2024-02-08T00:00:00Z", "2024-02-11T00:00:00Z",
        "2024-03-02T00:00:00Z", "2024-03-05T00:00:00Z",
        "2024-03-08T00:00:00Z", "2024-03-11T00:00:00Z",
    ], utc=True)
    count = len(decision)
    candidate_ids = [f"month_{index:02d}" for index in range(count)]
    # Varying residuals give H3 a non-degenerate ridge target.  The values
    # themselves are irrelevant—the test checks exact parity with the
    # independently implemented vectorised reference.
    net = np.asarray([20, -10, 30, -20, 200, 45, -35, 55, -45, 70, -60, 80, -70], dtype=float)
    expected = np.asarray([0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0, 5], dtype=float)
    candidates = pd.DataFrame({
        "candidate_id": candidate_ids,
        "decision_ts": decision,
        "feature_generation_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=1),
        "side_name": ["long"] * count,
        "head_name": ["p_clear"] * count,
        "fold_id": ["fold_0"] * count,
        "transport": ["transport_a"] * count,
        "meta_partition": ["inner_oof"] * count,
        "feature_contract_sha256": [CONTRACT] * count,
        "semantic_label": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "head_prediction": [0.70, 0.30, 0.80, 0.20, 0.90, 0.70, 0.30, 0.80, 0.20, 0.70, 0.30, 0.80, 0.20],
        "net_bps": net,
        "base_expected_bps": expected,
        "asset": ["A"] * count,
    })
    contributions = pd.DataFrame({
        "candidate_id": candidate_ids,
        "__ts__": decision,
        "side_name": ["long"] * count,
        "head_name": ["p_clear"] * count,
        "fold_id": ["fold_0"] * count,
        "transport": ["transport_a"] * count,
        "meta_partition": ["inner_oof"] * count,
        "feature_contract_sha256": [CONTRACT] * count,
        "rule_signature": ["monthly_shared_family"] * count,
        "contribution_direction": ["positive"] * count,
        "family_ensemble_tree_contribution": [0.3] * count,
    })
    candidates.to_parquet(candidate_path, index=False)
    contributions.to_parquet(contribution_path, index=False)
    index_path = spool / "strict_family_input_parts.parquet"
    index = pd.read_parquet(index_path)
    index.loc[index.index[0], "candidate_rows"] = len(candidates)
    index.loc[index.index[0], "contribution_rows"] = len(contributions)
    index.loc[index.index[0], "candidate_sha256"] = _sha256(candidate_path)
    index.loc[index.index[0], "contribution_sha256"] = _sha256(contribution_path)
    index.to_parquet(index_path, index=False)
    return spool


def test_event_store_is_token_free_ordered_and_immutable(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    assert store.manifest["row_counts"] == {
        "candidate_rows": 3,
        "contribution_rows": 3,
        "score_rows": 3,
        "resolution_rows": 3,
        "physical_part_rows": 6,
    }
    loaded = load_strict_event_store(store.root)
    assert set(loaded.part_index["dataset"]) == {
        "candidate", "contribution", "score_order", "resolution_order",
    }
    score = pd.concat([frame for _, frame in iter_event_store_parts(loaded, dataset="score_order")])
    resolution = pd.concat([frame for _, frame in iter_event_store_parts(loaded, dataset="resolution_order")])
    # These are reusable globally ordered streams, not merely individually
    # sorted candidate parts.  This avoids a downstream external sort.
    assert list(score["candidate_id"]) == ["early", "outer", "late"]
    assert list(resolution["candidate_id"]) == ["early", "outer", "late"]
    contribution = pd.concat([frame for _, frame in iter_event_store_parts(loaded, dataset="contribution")])
    assert {"semantic_label", "net_bps", "base_expected_bps"}.isdisjoint(contribution.columns)
    assert not any("leaf" in column.lower() for column in contribution.columns)
    with pytest.raises(FileExistsError):
        build_strict_event_store(_spool(tmp_path / "other"), store.root)


def test_contribution_event_streams_pair_candidate_outcomes_once_per_bounded_part(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    loaded = load_strict_contribution_event_streams(streams.root)
    assert loaded.manifest["row_counts"] == {
        "feature_event_rows": 3,
        "resolution_event_rows": 3,
        "physical_parts": 4,
    }
    assert loaded.manifest["storage_guard"]["written_physical_bytes"] > 0
    for item in loaded.part_index.itertuples(index=False):
        frame = pd.read_parquet(loaded.root / item.path)
        assert tuple(frame.columns) == EVENT_COLUMNS
        assert not any("leaf" in column.lower() for column in frame.columns)
        assert frame[item.timestamp_column].is_monotonic_increasing
    physical = loaded.root / loaded.part_index.iloc[0].path
    physical.write_bytes(b"tampered")
    with pytest.raises(StrictEventStoreError, match="physical part"):
        load_strict_contribution_event_streams(loaded.root)


def test_contribution_event_streams_record_and_validate_both_source_pair_hashes(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    index = streams.part_index
    required = {
        "source_candidate_sha256", "source_contribution_sha256",
        "source_candidate_identity_sha256", "source_contribution_identity_sha256",
    }
    assert required.issubset(index.columns)
    candidates = store.part_index.loc[store.part_index["dataset"].eq("candidate")].set_index("path")
    contributions = store.part_index.loc[store.part_index["dataset"].eq("contribution")].set_index("path")
    for item in index.itertuples(index=False):
        candidate = candidates.loc[str(item.source_candidate_path)]
        contribution = contributions.loc[str(item.source_contribution_path)]
        assert item.source_candidate_sha256 == candidate.sha256
        assert item.source_contribution_sha256 == contribution.sha256
        assert item.source_candidate_identity_sha256 == candidate.candidate_identity_sha256
        assert item.source_contribution_identity_sha256 == contribution.candidate_identity_sha256

    tampered = index.copy()
    tampered.loc[tampered.index[0], "source_contribution_sha256"] = "0" * 64
    _rewrite_event_sidecar_index(streams.root, tampered)
    with pytest.raises(StrictEventStoreError, match="source contribution hash"):
        load_strict_contribution_event_streams(streams.root, verify_parts=False)


def test_contribution_event_pair_identity_is_dataset_specific_and_legacy_index_loads(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool_with_multiple_family_rows(tmp_path), tmp_path / "store")
    candidate = store.part_index.loc[store.part_index["dataset"].eq("candidate")].iloc[0]
    contribution = store.part_index.loc[
        store.part_index["path"].astype(str).eq(str(candidate.paired_contribution_path))
    ].iloc[0]
    assert candidate.candidate_identity_sha256 != contribution.candidate_identity_sha256

    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    # A current sidecar records both digests, each against its own canonical
    # dataset.  It must not demand that they are equal.
    loaded = load_strict_contribution_event_streams(streams.root)
    assert len(loaded.part_index) == 4

    # The existing v1 production sidecar index only carries the candidate
    # source hash.  It remains readable and pairs through the canonical
    # candidate->contribution path rather than a false digest equality.
    legacy = streams.part_index.drop(columns=[
        "source_contribution_sha256", "source_candidate_identity_sha256", "source_contribution_identity_sha256",
    ])
    _rewrite_event_sidecar_index(streams.root, legacy)
    legacy_loaded = load_strict_contribution_event_streams(streams.root)
    assert len(legacy_loaded.part_index) == 4


def test_contribution_event_builder_hashes_canonical_pair_before_decoding(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    candidate = store.part_index.loc[store.part_index["dataset"].eq("candidate")].iloc[0]
    contribution = store.root / str(candidate.paired_contribution_path)
    contribution.write_bytes(b"changed-after-event-store-sealing")
    target = tmp_path / "events"
    with pytest.raises(StrictEventStoreError, match="contribution source hash changed before sidecar read"):
        build_strict_contribution_event_streams(store, target)
    assert not target.exists()


def test_contribution_event_loader_enforces_exact_compact_physical_types(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    index = streams.part_index.copy()
    path = streams.root / str(index.iloc[0].path)
    # Deliberately abandon dictionary encoding for exactly one field;
    # recalculate outer hashes so type validation itself (not the hash
    # mismatch) must reject the result.
    table = pq.read_table(path)
    columns = [table[name] for name in table.column_names]
    columns[0] = pc.cast(columns[0], pa.string())
    pq.write_table(pa.Table.from_arrays(columns, names=table.column_names), path, compression="zstd")
    index.loc[index.index[0], "sha256"] = _sha256(path)
    _rewrite_event_sidecar_index(streams.root, index)
    with pytest.raises(StrictEventStoreError, match="physical types"):
        load_strict_contribution_event_streams(streams.root)


def test_legacy_v1_event_sidecar_index_without_new_pair_fields_remains_readable(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    legacy = streams.part_index.drop(columns=[
        "source_contribution_sha256", "source_candidate_identity_sha256", "source_contribution_identity_sha256",
    ])
    _rewrite_event_sidecar_index(streams.root, legacy)
    loaded = load_strict_contribution_event_streams(streams.root)
    assert len(loaded.part_index) == 4


def test_contribution_event_streams_fail_before_writing_when_storage_reserve_is_unavailable(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    target = tmp_path / "events"
    with pytest.raises(StrictEventStoreError, match="insufficient free space"):
        build_strict_contribution_event_streams(
            store, target, max_output_bytes=1, minimum_free_bytes=10**20,
        )
    assert not target.exists()


def test_contribution_event_reader_kway_merges_candidate_groups_without_reordering(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    lookup: dict[tuple[str, str], int] = {}
    feature = list(iter_contribution_event_groups(
        streams, dataset="feature_event_order", contract=CONTRACT, side="long", head="p_clear",
        family_lookup=lookup, batch_rows=1,
    ))
    resolution = list(iter_contribution_event_groups(
        streams, dataset="resolution_event_order", contract=CONTRACT, side="long", head="p_clear",
        family_lookup=lookup, batch_rows=1,
    ))
    assert [item.candidate_id for item in feature] == ["early", "outer", "late"]
    assert [item.candidate_id for item in resolution] == ["early", "outer", "late"]
    assert [item.timestamp_ns for item in feature] == sorted(item.timestamp_ns for item in feature)
    assert [item.timestamp_ns for item in resolution] == sorted(item.timestamp_ns for item in resolution)
    assert all(len(item.family_codes) == 1 for item in feature)
    assert set(lookup) == {("early_family", "positive"), ("outer_family", "positive"), ("late_family", "negative")}
    assert sorted(lookup.values()) == [0, 1, 2]


def test_contribution_event_reader_yields_complete_timestamp_blocks(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    blocks = list(iter_contribution_event_timestamp_blocks(
        streams, dataset="feature_event_order", contract=CONTRACT, side="long", head="p_clear",
        months=["2024-01"], batch_rows=1,
    ))
    assert [block.timestamp_ns for block in blocks] == sorted(block.timestamp_ns for block in blocks)
    assert [group.candidate_id for block in blocks for group in block.groups] == ["early", "outer", "late"]
    assert all(group.timestamp_ns == block.timestamp_ns for block in blocks for group in block.groups)


def test_cutoff_projection_uses_label_resolution_before_any_part_decode(tmp_path: Path, monkeypatch) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")

    def _must_not_decode(*args, **kwargs):  # pragma: no cover - assertion body
        raise AssertionError("cutoff source selection decoded a physical parquet part")

    # Passing a verified descriptor ensures the projection has no legitimate
    # need to re-read the manifest/index either.
    monkeypatch.setattr(pd, "read_parquet", _must_not_decode)
    early = source_parts_for_cutoff(store, "2024-01-01T04:00:00Z")
    assert early.empty, "a mixed physical part must not leak its unresolved late row"
    all_resolved = source_parts_for_cutoff(store, "2024-01-01T09:00:00Z")
    assert set(all_resolved["dataset"]) == {"candidate", "contribution"}
    assert all_resolved["path"].str.contains("outer_test").sum() == 0


def test_predecessor_reader_decodes_only_eligible_inner_oof_contributions(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    pairs = list(iter_predecessor_selection_pairs(store, "2024-01-01T04:00:00Z"))
    assert len(pairs) == 1
    _, candidates, contributions = pairs[0]
    assert candidates["candidate_id"].tolist() == ["early"]
    assert contributions["candidate_id"].tolist() == ["early"]
    assert contributions["rule_signature"].tolist() == ["early_family"]
    assert {"semantic_label", "net_bps", "base_expected_bps"}.isdisjoint(contributions.columns)
    assert not any("leaf" in column.lower() for column in contributions.columns)


def test_sealed_store_rejects_tampered_physical_part_and_raw_leaf_spool(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    candidate_path = store.root / store.part_index.loc[
        store.part_index["dataset"].eq("candidate"), "path"
    ].iloc[0]
    candidate_path.write_bytes(b"tampered")
    with pytest.raises(StrictEventStoreError, match="hash|physical|parquet|invalid"):
        load_strict_event_store(store.root)
    with pytest.raises(StrictEventStoreError, match="forbidden raw leaf"):
        build_strict_event_store(_spool(tmp_path / "raw", raw_leaf=True), tmp_path / "raw_store")


def test_vectorised_health_uses_event_store_and_keeps_a_compact_audit(tmp_path: Path) -> None:
    """The production path must not need a full pandas family-state table."""

    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "health", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB",
    )
    health = pd.read_parquet(root / "base_leaf_health_features_oof.parquet")
    manifest = json.loads((root / "health_materialization_manifest.json").read_text())
    assert len(health) == 3
    assert health.filter(regex=r"^base_health__").notna().all().all()
    assert manifest["contract"]["state_engine"].startswith("vectorized SQL")
    assert manifest["row_counts"]["family_candidate_states"] == 0
    assert (root / "base_leaf_family_candidate_states.parquet").is_file()


def test_vectorised_selected_state_audit_is_source_filtered(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path), tmp_path / "store")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    key = (CONTRACT, "long", "p_clear", "early_family", "positive")
    config = CausalLeafHealthConfig(
        selected_context_families=frozenset({key}),
        selected_covariance_families=frozenset({key}),
        selected_relationship_families=frozenset({key}),
        h3_min_rows=2,
        h3_max_rows_per_family=8,
    )
    root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "health", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB",
    )
    states = pd.read_parquet(root / "base_leaf_family_candidate_states.parquet")
    assert states["rule_signature"].tolist() == ["early_family"]
    assert states["h4_selection_active"].eq(1.0).all()
    assert states["h5_selection_active"].eq(1.0).all()


def test_vectorised_h1_keeps_same_timestamp_labels_out_of_history(tmp_path: Path) -> None:
    """The later inner row sees the early label, never its own/future label."""

    store = build_strict_event_store(_spool(tmp_path, shared_inner_family=True), tmp_path / "store")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "health", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB",
    )
    health = pd.read_parquet(root / "base_leaf_health_features_oof.parquet").set_index("candidate_id")
    metric = "base_health__h1__p_clear__positive__row_support"
    assert health.loc["early", metric] == 0.0
    assert health.loc["outer", metric] == 0.0
    assert health.loc["late", metric] == 1.0


def test_scoped_health_keeps_empty_selection_contract_and_no_global_contribution_join(tmp_path: Path) -> None:
    """The scalable path must remain usable before any H3/H4/H5 family wins."""

    store = build_strict_event_store(_spool(tmp_path, shared_inner_family=True), tmp_path / "store")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    root = materialize_strict_oof_causal_leaf_health_scoped(
        store, tmp_path / "health", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB",
    )
    health = pd.read_parquet(root / "base_leaf_health_features_oof.parquet").set_index("candidate_id")
    manifest = json.loads((root / "health_materialization_manifest.json").read_text())
    assert len(health) == 3
    assert health.filter(regex=r"^base_health__").notna().all().all()
    assert manifest["performance"]["global_contribution_join"] is False
    assert manifest["contract"]["scope_plan"].startswith("candidate-only global H1")
    metric = "base_health__h1__p_clear__positive__row_support"
    assert health.loc["late", metric] == 1.0


def test_scoped_health_matches_reference_feature_contract_without_selected_families(tmp_path: Path) -> None:
    """The scalable plan is a physical refactor, not a new feature contract."""

    store = build_strict_event_store(_spool(tmp_path, shared_inner_family=True), tmp_path / "store")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    reference_root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "reference", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB",
    )
    scoped_root = materialize_strict_oof_causal_leaf_health_scoped(
        store, tmp_path / "scoped", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB",
    )
    reference = pd.read_parquet(reference_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    scoped = pd.read_parquet(scoped_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    assert list(scoped.columns) == list(reference.columns)
    assert scoped.loc[:, ["candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition"]].equals(
        reference.loc[:, ["candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition"]]
    )
    fields = [name for name in reference if name.startswith("base_health__")]
    np.testing.assert_allclose(scoped.loc[:, fields].to_numpy(float), reference.loc[:, fields].to_numpy(float), rtol=1e-6, atol=1e-6)


def test_event_incremental_health_matches_reference_h1_without_selected_families(tmp_path: Path) -> None:
    """The new production path must not reintroduce a scope-wide sort/join."""

    store = build_strict_event_store(_spool(tmp_path, shared_inner_family=True), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    reference_root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "reference", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB",
    )
    incremental_root = materialize_strict_oof_causal_leaf_health_event_incremental(
        store, streams, tmp_path / "incremental", causal_context=timeline,
        context_feature_columns=("regime_entropy",), memory_limit="512MB", temp_disk_limit="1GB",
        batch_rows=1,
    )
    reference = pd.read_parquet(reference_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    incremental = pd.read_parquet(incremental_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    h1 = [name for name in reference if name.startswith("base_health__h1__")]
    assert list(incremental.columns) == [*reference.columns, "base_reasoning__family_contribution_entropy"]
    np.testing.assert_allclose(incremental.loc[:, h1].to_numpy(float), reference.loc[:, h1].to_numpy(float), rtol=1e-6, atol=1e-6)
    manifest = json.loads((incremental_root / "health_materialization_manifest.json").read_text())
    assert manifest["performance"]["global_contribution_join"] is False
    assert manifest["performance"]["scope_contribution_sort"] is False
    assert "base_reasoning__family_contribution_entropy" in incremental
    entropy_audit = incremental_root / "reasoning_entropy_coverage_audit.parquet"
    assert entropy_audit.is_file()
    assert entropy_audit.name in manifest["sha256"]
    assert manifest["contract"]["reasoning_entropy"].startswith("candidate-level Shannon entropy")


def test_event_incremental_health_matches_reference_with_selected_family_states(tmp_path: Path) -> None:
    store = build_strict_event_store(_spool(tmp_path, shared_inner_family=True), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    key = (CONTRACT, "long", "p_clear", "shared_inner", "positive")
    config = CausalLeafHealthConfig(
        selected_context_families=frozenset({key}), selected_covariance_families=frozenset({key}),
        selected_relationship_families=frozenset({key}), h3_min_rows=2, h3_max_rows_per_family=8,
    )
    reference_root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "reference", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB",
    )
    incremental_root = materialize_strict_oof_causal_leaf_health_event_incremental(
        store, streams, tmp_path / "incremental", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB", temp_disk_limit="1GB",
        batch_rows=1,
    )
    reference = pd.read_parquet(reference_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    incremental = pd.read_parquet(incremental_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    assert list(incremental.columns) == [*reference.columns, "base_reasoning__family_contribution_entropy"]
    for section in ("h1", "h2", "h3", "h4", "h5"):
        fields = [name for name in reference if name.startswith(f"base_health__{section}__")]
        np.testing.assert_allclose(
            incremental.loc[:, fields].to_numpy(float), reference.loc[:, fields].to_numpy(float),
            rtol=1e-6, atol=1e-6, err_msg=f"{section} month-frozen parity",
        )


def test_bounded_h4_h5_matches_dataframe_reference_across_batches(tmp_path: Path) -> None:
    """H4/H5 must preserve reference values without decoding selected state.

    Two selected families share every feature timestamp, which forces the
    bounded H4 path to spill/update an equal-timestamp batch.  The later
    transport/fold is a distinct frozen reference block, so this fixture also
    exercises the causal snapshot boundary rather than merely schema parity.
    """

    context = (
        "regime_entropy",
        "continuous_regime__relationship_break__btc_eth__residual_abs_30d",
        "continuous_regime__relationship_break__btc_eth__residual_abs_90d",
    )
    family_a = (CONTRACT, "long", "p_clear", "bounded_family_a", "positive")
    family_b = (CONTRACT, "long", "p_clear", "bounded_family_b", "positive")
    rows: list[dict[str, object]] = []
    for transport in ("transport_a", "transport_b"):
        for day in range(5):
            timestamp = pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(days=day)
            fold = "fold_0" if day < 3 else "fold_1"
            for family, offset in ((family_a, 0.0), (family_b, 0.25)):
                rows.append({
                "candidate_id": f"{transport}_candidate_{day:02d}", "decision_ts": timestamp,
                "side_name": "long", "fold_id": fold, "transport": transport,
                "meta_partition": "meta", "feature_generation_ts": timestamp,
                "label_available_ts": timestamp + pd.Timedelta(hours=12),
                "feature_contract_sha256": CONTRACT, "head_name": "p_clear",
                "rule_signature": family[3], "contribution_direction": family[4],
                "family_ensemble_tree_contribution": np.float32(0.5 + offset),
                "h1_economic_residual_bps": np.float32((-15.0 + 10.0 * day) + offset),
                "h2_instability": np.float32(0.10 * day + offset),
                "h4_selection_active": np.float32(1.0), "h5_selection_active": np.float32(1.0),
                context[0]: np.float32(0.1 * day + offset),
                context[1]: np.float32(0.05 * day - offset),
                context[2]: np.float32(0.10 * day + 2.0 * offset),
                })
    states = pd.DataFrame(rows)
    selected_path = tmp_path / "selected_states.parquet"
    states.to_parquet(selected_path, index=False, compression="zstd")
    config = CausalLeafHealthConfig(
        covariance_max_fields=3, covariance_min_reference_rows=2,
        selected_covariance_families=frozenset({family_a, family_b}),
        selected_relationship_families=frozenset({family_a, family_b}),
    )
    reference_covariance, reference_relationships, reference_audit = _materialise_h4_h5(
        states, context_columns=context, config=config,
    )
    covariance_path = tmp_path / "bounded_covariance.parquet"
    relationship_path = tmp_path / "bounded_relationships.parquet"
    covariance_rows, relationship_rows, audit = _materialise_h4_h5_bounded(
        selected_path, covariance_path=covariance_path, relationship_path=relationship_path,
        context_columns=context, config=config, memory_limit="256MB", temp_disk_limit="256MB",
        batch_rows=2,
    )
    covariance = pd.read_parquet(covariance_path)
    relationships = pd.read_parquet(relationship_path)
    assert covariance_rows == len(reference_covariance) == len(covariance)
    assert relationship_rows == len(reference_relationships) == len(relationships)
    sort_covariance = ["candidate_id", "head_name", "rule_signature", "contribution_direction"]
    sort_relationships = [*sort_covariance, "relationship_pair"]
    covariance = covariance.sort_values(sort_covariance, kind="stable").reset_index(drop=True)
    reference_covariance = reference_covariance.sort_values(sort_covariance, kind="stable").reset_index(drop=True)
    assert list(covariance.columns) == list(reference_covariance.columns)
    for name in covariance.columns:
        if name.startswith("base_health__h4__") or name == "family_ensemble_tree_contribution":
            np.testing.assert_allclose(
                covariance[name].to_numpy(float), reference_covariance[name].to_numpy(float),
                rtol=1e-6, atol=1e-6, equal_nan=True,
            )
        else:
            assert covariance[name].astype(str).tolist() == reference_covariance[name].astype(str).tolist()
    relationships = relationships.sort_values(sort_relationships, kind="stable").reset_index(drop=True)
    reference_relationships = reference_relationships.sort_values(sort_relationships, kind="stable").reset_index(drop=True)
    assert list(relationships.columns) == list(reference_relationships.columns)
    for name in ("family_ensemble_tree_contribution", "relationship_break", "portable_economic_weight"):
        np.testing.assert_allclose(
            relationships[name].to_numpy(float), reference_relationships[name].to_numpy(float),
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )
    assert relationships["material_break"].astype(bool).tolist() == reference_relationships["material_break"].astype(bool).tolist()
    assert audit.equals(reference_audit)
    assert not (tmp_path / "h4_h5_bounded_work").exists()


def test_event_incremental_matches_month_frozen_h2_h3_reference(tmp_path: Path) -> None:
    """H2/H3 must not admit a label resolved in the current score month."""

    store = build_strict_event_store(_spool_multi_month_h3(tmp_path), tmp_path / "store")
    streams = build_strict_contribution_event_streams(store, tmp_path / "events")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime([
            "2023-12-31T00:00:00Z", "2024-01-01T00:00:00Z",
            "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z",
        ], utc=True),
        "regime_entropy": [0.10, 0.20, 0.30, 0.40],
    })
    key = (CONTRACT, "long", "p_clear", "monthly_shared_family", "positive")
    config = CausalLeafHealthConfig(
        selected_context_families=frozenset({key}),
        min_timestamp_support=1, min_day_support=1, min_symbol_support=1,
        min_period_rows=2, min_periods=1, period_close_lag_hours=24,
        h3_min_rows=2, h3_max_rows_per_family=8,
    )
    reference_root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "reference", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB",
    )
    incremental_root = materialize_strict_oof_causal_leaf_health_event_incremental(
        store, streams, tmp_path / "incremental", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB",
        temp_disk_limit="1GB", batch_rows=1,
    )
    reference = pd.read_parquet(reference_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    incremental = pd.read_parquet(incremental_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    for section in ("h1", "h2", "h3", "h4", "h5"):
        fields = [name for name in reference if name.startswith(f"base_health__{section}__")]
        np.testing.assert_allclose(
            incremental.loc[:, fields].to_numpy(float), reference.loc[:, fields].to_numpy(float),
            rtol=1e-6, atol=1e-6, err_msg=f"{section} month-frozen parity",
        )
    h3_availability = "base_health__h3__p_clear__positive__availability"
    h2_periods = "base_health__h2__p_clear__positive__period_count"
    february = incremental.loc[incremental["candidate_id"].isin(["month_05", "month_06", "month_07", "month_08"])]
    assert february[h3_availability].eq(1.0).all()
    march = incremental.loc[incremental["candidate_id"].isin(["month_09", "month_10", "month_11", "month_12"])]
    assert march[h2_periods].eq(1.0).all()


def test_scoped_health_matches_reference_with_selected_family_states(tmp_path: Path) -> None:
    """Selected H3/H4/H5 source filtering must also preserve the reference values."""

    store = build_strict_event_store(_spool(tmp_path, shared_inner_family=True), tmp_path / "store")
    timeline = pd.DataFrame({
        "regime_available_utc": pd.to_datetime(["2023-12-31T00:00:00Z"], utc=True),
        "regime_entropy": [0.25],
    })
    key = (CONTRACT, "long", "p_clear", "shared_inner", "positive")
    config = CausalLeafHealthConfig(
        selected_context_families=frozenset({key}),
        selected_covariance_families=frozenset({key}),
        selected_relationship_families=frozenset({key}),
        h3_min_rows=2,
        h3_max_rows_per_family=8,
    )
    reference_root = materialize_strict_oof_causal_leaf_health_vectorized(
        store, tmp_path / "reference", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB",
    )
    scoped_root = materialize_strict_oof_causal_leaf_health_scoped(
        store, tmp_path / "scoped", causal_context=timeline,
        context_feature_columns=("regime_entropy",), config=config, memory_limit="512MB",
    )
    reference = pd.read_parquet(reference_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    scoped = pd.read_parquet(scoped_root / "base_leaf_health_features_oof.parquet").sort_values("candidate_id").reset_index(drop=True)
    assert list(scoped.columns) == list(reference.columns)
    fields = [name for name in reference if name.startswith("base_health__")]
    np.testing.assert_allclose(scoped.loc[:, fields].to_numpy(float), reference.loc[:, fields].to_numpy(float), rtol=1e-6, atol=1e-6)
