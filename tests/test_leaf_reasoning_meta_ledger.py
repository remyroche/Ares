from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.tp6_portability_data import FROZEN_META_CONTEXT
from extreme_price_movements.leaf_reasoning_meta_ledger import (
    LeafReasoningMetaLedgerError,
    assemble_leaf_reasoning_meta_ledger,
    assemble_leaf_reasoning_meta_ledger_pairs,
    write_immutable_meta_ledger,
)


def _prediction(candidate: str, side: str, timestamp: str, fold: str) -> dict[str, object]:
    decision = pd.Timestamp(timestamp, tz="UTC")
    return {
        "candidate_id": candidate,
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "side_name": side,
        "fold_id": fold,
        "gross_bps": 125.0,
        "net_bps": 25.0,
        "base_expected_bps": 10.0,
        "base_fit_cutoff_ts": decision - pd.Timedelta(hours=1),
        "feature_generation_ts": decision,
        "p_adverse": .2,
        "p_weak": .3,
        "p_clear": .5,
        **{name: .1 for name in FROZEN_META_CONTEXT},
    }


def _source(
    tmp_path: Path, *, raw_leaf: bool = False, transport: str = "transport_a",
    reuse_candidate_id: bool = False,
) -> tuple[Path, Path]:
    strict = tmp_path / "strict"
    compact = tmp_path / "compact"
    strict.mkdir(parents=True)
    (strict / "strict_oof_reasoning_manifest.json").write_text(json.dumps({
        "status": "STRICT_OOF_BASE_REASONING_MATERIALIZED",
        "prediction_shards": "base_prediction_shards/<transport>/<side>/",
        "transports": [transport],
    }))
    compact.mkdir()
    (compact / "base_reasoning_representation_manifest.json").write_text(json.dumps({
        "status": "COMPACT_STRICT_OOF_BASE_REASONING_MATERIALIZED",
        "schema": "base_reasoning_representation_v2",
        "contract": {"leaf_alignment": "opaque local only"},
    }))
    reasoning: list[dict[str, object]] = []
    for side in ("long", "short"):
        shard = strict / "base_prediction_shards" / transport / side
        shard.mkdir(parents=True)
        candidate = "generator-reused-candidate" if reuse_candidate_id else None
        inner = _prediction(candidate or f"{transport}-{side}-inner", side, "2024-01-02", f"{transport}_inner_00")
        outer = _prediction(candidate or f"{transport}-{side}-outer", side, "2024-02-02", f"{transport}_outer")
        pd.DataFrame([inner]).to_parquet(shard / "strict_oof_predictions.parquet", index=False)
        pd.DataFrame([outer]).to_parquet(shard / "outer_predictions.parquet", index=False)
        for partition, row in (("inner_oof", inner), ("outer_test", outer)):
            for head in ("p_adverse", "p_weak", "p_clear"):
                reasoning.append({
                    "candidate_id": row["candidate_id"], "__ts__": row["decision_ts"],
                    "side_name": side, "fold_id": row["fold_id"], "transport": transport,
                    "meta_partition": partition, "head_name": head,
                    "contribution_direction": "positive" if head != "p_weak" else "negative",
                    "base_reasoning__g1_leaf_train_frequency_mean": .4,
                    "base_reasoning__g2_recurrent_family_weight": .3,
                    "base_reasoning__g3_contribution_bundle_weighted_svd_00": .2,
                    **({"leaf_assignment__bad": 5} if raw_leaf else {}),
                })
    pd.DataFrame(reasoning).to_parquet(compact / "base_reasoning_features_oof.parquet", index=False)
    return strict, compact


def test_assembles_inner_and_outer_with_cost_and_head_qualified_features(tmp_path: Path) -> None:
    strict, compact = _source(tmp_path)
    result = assemble_leaf_reasoning_meta_ledger(strict, compact)
    assert len(result.ledger) == 4
    assert set(result.ledger.meta_partition) == {"inner_oof", "outer_test"}
    assert result.ledger.base_same_side_strict_oof.all()
    assert result.ledger.realized_cost_bps.eq(100.0).all()
    assert "base_reasoning__g1_leaf_train_frequency_mean__p_clear__positive" in result.ledger
    assert result.feature_groups["L1"]
    assert result.feature_groups["L2"]
    assert result.feature_groups["L3"]
    assert not any("leaf_assignment" in column for column in result.ledger)
    out = write_immutable_meta_ledger(result, tmp_path / "out")
    assert (out / "base_to_meta_reasoning_ledger.parquet").is_file()
    with pytest.raises(FileExistsError):
        write_immutable_meta_ledger(result, out)


def test_allows_reused_generator_candidate_id_but_rejects_true_full_identity_duplicate(tmp_path: Path) -> None:
    """Candidate IDs alone are not an OOF identity across side/partition rows."""

    strict, compact = _source(tmp_path / "reused", reuse_candidate_id=True)
    result = assemble_leaf_reasoning_meta_ledger(strict, compact)
    assert len(result.ledger) == 4
    assert result.ledger["candidate_id"].nunique() == 1
    assert not result.ledger.duplicated([
        "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
    ]).any()

    strict, compact = _source(tmp_path / "duplicate")
    shard = strict / "base_prediction_shards" / "transport_a" / "long" / "strict_oof_predictions.parquet"
    duplicated = pd.read_parquet(shard)
    pd.concat((duplicated, duplicated.iloc[[0]]), ignore_index=True).to_parquet(shard, index=False)
    with pytest.raises(LeafReasoningMetaLedgerError, match="duplicate inner_oof base candidate identity"):
        assemble_leaf_reasoning_meta_ledger(strict, compact)


def test_rejects_raw_leaf_input_and_missing_outer_partition(tmp_path: Path) -> None:
    strict, compact = _source(tmp_path, raw_leaf=True)
    with pytest.raises(LeafReasoningMetaLedgerError, match="raw fold-local leaf"):
        assemble_leaf_reasoning_meta_ledger(strict, compact)
    strict, compact = _source(tmp_path / "clean")
    table = pd.read_parquet(compact / "base_reasoning_features_oof.parquet")
    table = table.loc[table.meta_partition.ne("outer_test")]
    table.to_parquet(compact / "base_reasoning_features_oof.parquet", index=False)
    with pytest.raises(LeafReasoningMetaLedgerError, match="identities are not identical"):
        assemble_leaf_reasoning_meta_ledger(strict, compact)


def test_combines_independently_proven_transport_pairs_without_overlap(tmp_path: Path) -> None:
    strict_a, compact_a = _source(tmp_path / "a", transport="transport_a")
    strict_b, compact_b = _source(tmp_path / "b", transport="transport_b")
    result = assemble_leaf_reasoning_meta_ledger_pairs(((strict_a, compact_a), (strict_b, compact_b)))
    assert len(result.ledger) == 8
    assert result.manifest["transports"] == ["transport_a", "transport_b"]
    assert len(result.manifest["strict_sources"]) == 2

    duplicate_strict, duplicate_compact = _source(tmp_path / "duplicate", transport="transport_a")
    with pytest.raises(LeafReasoningMetaLedgerError, match="reuse transport"):
        assemble_leaf_reasoning_meta_ledger_pairs(
            ((strict_a, compact_a), (duplicate_strict, duplicate_compact))
        )


def test_joins_only_complete_token_free_causal_health_identity(tmp_path: Path) -> None:
    strict, compact = _source(tmp_path, transport="transport_a")
    health = tmp_path / "health"
    health.mkdir()
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        for partition, suffix, timestamp, fold in (
            ("inner_oof", "inner", "2024-01-02", "transport_a_inner_00"),
            ("outer_test", "outer", "2024-02-02", "transport_a_outer"),
        ):
            rows.append({
                "candidate_id": f"transport_a-{side}-{suffix}",
                "decision_ts": pd.Timestamp(timestamp, tz="UTC"),
                "side_name": side,
                "fold_id": fold,
                "transport": "transport_a",
                "meta_partition": partition,
                "base_health__h1__p_clear__positive__posterior_correctness": .5,
                "base_health__h2__p_clear__positive__instability": .0,
                "base_reasoning__family_contribution_entropy": .42,
            })
    pd.DataFrame(rows).to_parquet(health / "base_leaf_health_features_oof.parquet", index=False)
    (health / "health_materialization_manifest.json").write_text(json.dumps({
        "schema": "causal_leaf_health_v1",
        "status": "CAUSAL_LEAF_HEALTH_MATERIALIZED",
    }))
    result = assemble_leaf_reasoning_meta_ledger(strict, compact, health_root=health)
    assert result.feature_groups["H1"]
    assert result.feature_groups["H2"]
    assert result.feature_groups["S2_reasoning_entropy"] == ["base_reasoning__family_contribution_entropy"]
    assert result.ledger["base_reasoning__family_contribution_entropy"].eq(.42).all()
    assert result.manifest["health_source"] == str(health)

    corrupted = pd.read_parquet(health / "base_leaf_health_features_oof.parquet").iloc[:-1]
    corrupted.to_parquet(health / "base_leaf_health_features_oof.parquet", index=False)
    with pytest.raises(LeafReasoningMetaLedgerError, match="identities are not identical"):
        assemble_leaf_reasoning_meta_ledger(strict, compact, health_root=health)
