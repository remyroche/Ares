from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_predecessor_meta_oof import (
    IDENTITY,
    LINEAGE_COLUMNS,
    PREDECESSOR_FEATURE_COLUMNS,
    PREDECESSOR_SEMANTICS,
    StrictPredecessorMetaOOFConfig,
    StrictPredecessorMetaOOFError,
    join_strict_predecessor_oof_features,
    load_immutable_meta_ledger_for_predecessor,
    load_immutable_strict_predecessor_meta_oof,
    materialize_strict_predecessor_meta_oof,
    write_immutable_strict_predecessor_meta_oof,
)


SOURCE_MAP = {
    "upgrade_portability": "h_upgrade",
    "downgrade_portability": "h_downgrade",
    "unstable_upgrade_share": "h_unstable",
    "covariance_break_share": "h_covariance",
    "support_score": "h_support",
    "reasoning_entropy": "h_entropy",
}


def _config() -> StrictPredecessorMetaOOFConfig:
    return StrictPredecessorMetaOOFConfig(
        source_feature_map=SOURCE_MAP, min_train_rows=3, ridge_alpha=1.0,
        refit_interval_hours=2,
    )


def _ledger() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    for transport in ("A", "B"):
        for side_index, side in enumerate(("long", "short")):
            for index in range(12):
                decision = start + pd.Timedelta(hours=index)
                partition = "inner_oof" if index < 8 else "outer_test"
                x = float(index + side_index * 0.25)
                row: dict[str, object] = {
                    "candidate_id": f"{transport}-{side}-{index}",
                    "decision_ts": decision,
                    "label_available_ts": decision + pd.Timedelta(minutes=30),
                    "side_name": side,
                    "fold_id": f"{transport}_{side}_{partition}",
                    "transport": transport,
                    "meta_partition": partition,
                    "base_expected_bps": x,
                    "realized_net_bps": 2.0 * x + float(index % 3),
                    "base_same_side_strict_oof": True,
                    "h_upgrade": x,
                    "h_downgrade": -x,
                    "h_unstable": float(index % 2),
                    "h_covariance": x / 10.0,
                    "h_support": 1.0 / (index + 1.0),
                    "h_entropy": np.log1p(x),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_materializer_uses_only_prior_resolved_same_side_inner_rows() -> None:
    source = _ledger()
    result = materialize_strict_predecessor_meta_oof(source, config=_config())
    assert list(result.features.columns) == [*IDENTITY, *PREDECESSOR_FEATURE_COLUMNS, *LINEAGE_COLUMNS]
    assert len(result.ledger) == len(source)
    assert result.fit_audit.strict_prior_resolved.all()
    assert result.fit_audit.inner_only_training.all()
    assert result.fit_audit.same_row_or_in_sample_rejected.all()
    assert set(result.manifest["feature_columns"]) == set(PREDECESSOR_FEATURE_COLUMNS)
    values = result.features.loc[:, list(PREDECESSOR_FEATURE_COLUMNS)].to_numpy(float)
    assert np.isfinite(values).all()
    decision = pd.to_datetime(result.features.decision_ts, utc=True)
    assert (pd.to_datetime(result.features.predecessor_oof_fit_end_ts, utc=True) < decision).all()
    assert (pd.to_datetime(result.features.predecessor_oof_generated_ts, utc=True) <= decision).all()
    assert (pd.to_datetime(result.features.predecessor_oof_available_ts, utc=True) <= decision).all()
    assert result.features.predecessor_same_side_strict_oof.all()


def test_materializer_allows_candidate_ids_reused_by_independent_transports() -> None:
    """The six-field identity, not candidate_id alone, defines a ledger row."""

    source = _ledger()
    # This mirrors the transport ledger: candidate IDs are locally generated
    # and therefore repeat across otherwise independent transport/side runs.
    source["candidate_id"] = source["candidate_id"].str.rsplit("-", n=1).str[-1]
    assert source["candidate_id"].duplicated().any()
    assert not source.duplicated(list(IDENTITY)).any()

    result = materialize_strict_predecessor_meta_oof(source, config=_config())

    assert len(result.features) == len(source)
    assert len(result.ledger) == len(source)
    assert not result.features.duplicated(list(IDENTITY)).any()
    assert not result.ledger.duplicated(list(IDENTITY)).any()
    assert result.fit_audit.same_row_or_in_sample_rejected.all()


def test_outer_outcomes_and_same_row_outcomes_cannot_change_features() -> None:
    source = _ledger()
    baseline = materialize_strict_predecessor_meta_oof(source, config=_config()).features
    changed_outer = source.copy()
    changed_outer.loc[changed_outer.meta_partition.eq("outer_test"), "realized_net_bps"] += 1_000_000.0
    outer_changed = materialize_strict_predecessor_meta_oof(changed_outer, config=_config()).features
    pd.testing.assert_frame_equal(
        baseline.loc[:, [*IDENTITY, *PREDECESSOR_FEATURE_COLUMNS]],
        outer_changed.loc[:, [*IDENTITY, *PREDECESSOR_FEATURE_COLUMNS]],
    )

    changed_current = source.copy()
    # This first row has no predecessor observations.  Changing only its own
    # eventual outcome cannot change its same-row OOF component vector.
    target_id = "A-long-0"
    changed_current.loc[changed_current.candidate_id.eq(target_id), "realized_net_bps"] += 1_000_000.0
    current_changed = materialize_strict_predecessor_meta_oof(changed_current, config=_config()).features
    baseline_row = baseline.loc[baseline.candidate_id.eq(target_id), list(PREDECESSOR_FEATURE_COLUMNS)]
    changed_row = current_changed.loc[current_changed.candidate_id.eq(target_id), list(PREDECESSOR_FEATURE_COLUMNS)]
    np.testing.assert_allclose(baseline_row.to_numpy(float), changed_row.to_numpy(float))


def test_exact_identity_join_fails_closed_on_extra_or_missing_candidate() -> None:
    source = _ledger()
    result = materialize_strict_predecessor_meta_oof(source, config=_config())
    with pytest.raises(StrictPredecessorMetaOOFError, match="identities differ"):
        join_strict_predecessor_oof_features(source.iloc[:-1], result.features)
    with pytest.raises(StrictPredecessorMetaOOFError, match="identities differ"):
        join_strict_predecessor_oof_features(source, result.features.iloc[:-1])


def test_requires_exact_six_explicit_source_semantics_and_rejects_leaf_source() -> None:
    source = _ledger()
    missing = dict(SOURCE_MAP)
    missing.pop("reasoning_entropy")
    with pytest.raises(StrictPredecessorMetaOOFError, match="exactly the six"):
        materialize_strict_predecessor_meta_oof(source, config=StrictPredecessorMetaOOFConfig(source_feature_map=missing))
    unsafe = dict(SOURCE_MAP)
    unsafe["reasoning_entropy"] = "raw_leaf_id"
    source["raw_leaf_id"] = 1.0
    with pytest.raises(StrictPredecessorMetaOOFError, match="raw leaf"):
        materialize_strict_predecessor_meta_oof(source, config=StrictPredecessorMetaOOFConfig(source_feature_map=unsafe))


def test_writer_loader_proves_immutable_joined_ledger(tmp_path) -> None:
    result = materialize_strict_predecessor_meta_oof(_ledger(), config=_config(), source_ledger_sha256="a" * 64)
    root = write_immutable_strict_predecessor_meta_oof(result, tmp_path / "predecessor")
    artifact = load_immutable_strict_predecessor_meta_oof(root)
    assert artifact.feature_columns == PREDECESSOR_FEATURE_COLUMNS
    assert len(artifact.ledger) == len(result.ledger)
    assert artifact.ledger_path.name == "base_to_meta_reasoning_ledger_predecessor_oof.parquet"
    with pytest.raises(FileExistsError):
        write_immutable_strict_predecessor_meta_oof(result, root)


def test_source_loader_requires_completed_hashed_immutable_base_ledger(tmp_path) -> None:
    root = tmp_path / "ledger"
    root.mkdir()
    table = root / "base_to_meta_reasoning_ledger.parquet"
    _ledger().to_parquet(table, index=False)
    import hashlib

    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    (root / "meta_ledger_manifest.json").write_text(json.dumps({
        "status": "STRICT_BASE_TO_META_LEDGER_ASSEMBLED",
        "sha256": {table.name: digest},
    }))
    loaded, _, actual = load_immutable_meta_ledger_for_predecessor(root)
    assert len(loaded) == len(_ledger())
    assert actual == digest
    table.write_bytes(b"corrupted")
    with pytest.raises(StrictPredecessorMetaOOFError, match="hash"):
        load_immutable_meta_ledger_for_predecessor(root)


def test_feature_contract_is_stably_ordered() -> None:
    assert tuple(name.removeprefix("predecessor_meta__") for name in PREDECESSOR_FEATURE_COLUMNS) == PREDECESSOR_SEMANTICS
