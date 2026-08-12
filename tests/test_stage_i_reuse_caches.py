from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_meta_offset_cache import (
    assert_fixed_offset_parity,
    causal_mapping_support_frame,
    load_meta_offset_cache,
    materialize_meta_offset_cache,
    meta_mda_fixed_offset_kwargs,
)
from extreme_price_movements.stage_i_target_neutral_cache import (
    atomic_cache_staging,
    groups_from_cached_edges,
    load_target_neutral_cache_for_contract,
    load_target_neutral_cache,
    materialize_relief_geometry_cache,
    materialize_scoped_spearman_groups,
    materialize_target_neutral_cache,
    matrix_frame_from_cache,
    relief_scores_from_geometry,
)


def _identity(n: int = 30) -> pd.DataFrame:
    signal = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(n)],
            "__ts__": signal,
            "__symbol__": np.where(np.arange(n) % 2, "BBB", "AAA"),
            "decision_ts": signal + pd.Timedelta(hours=1),
            "side_name": "long",
        }
    )


def test_target_neutral_cache_roundtrip_and_target_independence(tmp_path) -> None:
    identity = _identity()
    x = np.arange(len(identity), dtype=np.float32)
    features = pd.DataFrame(
        {"f1": x, "f2": 2.0 * x, "f3": np.sin(x), "constant": 1.0}
    )
    cache = materialize_target_neutral_cache(
        tmp_path / "cache", identity=identity, features=features,
        feature_names=list(features), selector_manifest_sha256="a" * 64,
        selector_feature_contract_sha256="b" * 64,
        selector_features_sha256="c" * 64, correlation_rows=30,
    )
    assert cache.manifest["request"]["target_dependent_fields_cached"] == []
    assert not bool(cache.coverage.set_index("feature").loc["constant", "nonconstant"])
    groups = groups_from_cached_edges(cache, ["f1", "f2", "f3"])
    assert any(set(group) == {0, 1} for group in groups)
    # Labels can change without changing any target-neutral request or artifact.
    labels_a = np.arange(len(identity)) % 3
    labels_b = 2 - labels_a
    assert not np.array_equal(labels_a, labels_b)
    again = materialize_target_neutral_cache(
        tmp_path / "cache", identity=identity, features=features,
        feature_names=list(features), selector_manifest_sha256="a" * 64,
        selector_feature_contract_sha256="b" * 64,
        selector_features_sha256="c" * 64, correlation_rows=30,
    )
    assert again.manifest["request_sha256"] == cache.manifest["request_sha256"]
    subset = matrix_frame_from_cache(cache, candidate_ids=["c4", "c1"], feature_names=["f3", "f1"])
    np.testing.assert_allclose(subset.to_numpy(), features.loc[[4, 1], ["f3", "f1"]].to_numpy())
    hot = load_target_neutral_cache_for_contract(
        cache.root, identity=identity, feature_names=list(features),
        selector_manifest_sha256="a" * 64,
        selector_feature_contract_sha256="b" * 64,
        selector_features_sha256="c" * 64,
    )
    assert isinstance(hot.matrix, np.memmap)
    with pytest.raises(ValueError, match="hot-load lineage drift"):
        load_target_neutral_cache_for_contract(
            cache.root, identity=identity, feature_names=list(features),
            selector_manifest_sha256="a" * 64,
            selector_feature_contract_sha256="b" * 64,
            selector_features_sha256="d" * 64,
        )
    scoped, scoped_manifest = materialize_scoped_spearman_groups(
        tmp_path / "scoped", cache=cache,
        train_candidate_ids=identity.candidate_id.iloc[:20],
        active_features=["f1", "f2", "f3"], threshold=0.95,
        random_state=9, max_rows=20,
    )
    assert any(set(group) == {0, 1} for group in scoped)
    assert scoped_manifest["request"]["training_rows_only"] is True
    from extreme_price_movements import lgbm_pipeline

    # A post-cache strict-OOF handoff field is a valid singleton MDA feature;
    # cached raw-field group indices must still map back to the full matrix.
    full = features.loc[:19, ["f1", "f2", "f3"]].copy()
    full["base_raw_score"] = np.linspace(-1.0, 1.0, len(full))
    mapped_groups = lgbm_pipeline._correlation_groups_for_mda(
        full, list(full), threshold=0.95, random_state=9,
        cfg={
            "stage_i_target_neutral_cache_root": str(cache.root),
            "stage_i_target_neutral_cache_request_sha256": cache.manifest["request_sha256"],
            "_stage_i_scoped_train_candidate_ids": identity.candidate_id.iloc[:20].to_numpy(),
        },
    )
    assert any(set(group) == {0, 1} for group in mapped_groups)
    assert all(3 not in group for group in mapped_groups)
    with pytest.raises(ValueError, match="lineage/artifact drift"):
        materialize_scoped_spearman_groups(
            tmp_path / "scoped", cache=cache,
            train_candidate_ids=identity.candidate_id.iloc[1:21],
            active_features=["f1", "f2", "f3"], threshold=0.95,
            random_state=9, max_rows=20,
        )


def test_target_neutral_cache_rejects_matrix_and_artifact_tamper(tmp_path) -> None:
    identity = _identity()
    features = pd.DataFrame({"f1": np.arange(len(identity)), "f2": np.arange(len(identity)) ** 2})
    cache = materialize_target_neutral_cache(
        tmp_path / "cache", identity=identity, features=features,
        feature_names=list(features), selector_manifest_sha256="a" * 64,
        selector_feature_contract_sha256="b" * 64,
        selector_features_sha256="c" * 64,
    )
    changed = features.copy()
    changed.loc[0, "f1"] = 999
    with pytest.raises(ValueError, match="request lineage drift"):
        materialize_target_neutral_cache(
            tmp_path / "cache", identity=identity, features=changed,
            feature_names=list(features), selector_manifest_sha256="a" * 64,
            selector_feature_contract_sha256="b" * 64,
            selector_features_sha256="c" * 64,
        )
    path = cache.root / "coverage_nonconstant_audit.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact drift"):
        load_target_neutral_cache(cache.root)


def test_relief_geometry_scores_match_direct_distance_ordering(tmp_path) -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(80, 5)).astype(np.float32)
    rows = np.arange(len(matrix), dtype=np.int32)
    cache = materialize_relief_geometry_cache(
        tmp_path / "relief", matrix=matrix,
        feature_names=[f"f{i}" for i in range(matrix.shape[1])],
        work_row_ids=rows, random_state=11, anchor_max_rows=32,
        neighbor_candidate_rows=60,
        training_candidate_ids=[f"c{i}" for i in rows],
        fold_lineage_sha256="f" * 64,
    )
    labels = (np.arange(len(rows)) % 3).astype(np.int8)
    cached = relief_scores_from_geometry(cache, labels, neighbors=4)
    # Independent direct scorer over the persisted distance ordering.
    direct = np.zeros(matrix.shape[1], dtype=np.float64)
    used = 0
    for pos, anchor in enumerate(cache.anchor_ids):
        ordered = cache.candidate_ids[cache.candidate_distance_order[pos]]
        ordered = ordered[ordered != anchor]
        hit = ordered[labels[ordered] == labels[anchor]][:4]
        miss = ordered[labels[ordered] != labels[anchor]][:4]
        if not len(hit) or not len(miss):
            continue
        direct += np.mean(np.abs(cache.standardized_matrix[miss] - cache.standardized_matrix[anchor]), axis=0)
        direct -= np.mean(np.abs(cache.standardized_matrix[hit] - cache.standardized_matrix[anchor]), axis=0)
        used += 1
    np.testing.assert_allclose(cached, direct / used, rtol=1e-6, atol=1e-6)
    # Exact parity with the uncached production helper for the same geometry.
    from extreme_price_movements import lgbm_pipeline as lp

    fresh = lp._approx_relief_scores_once(
        lp._standardized_relief_matrix(pd.DataFrame(matrix)),
        labels,
        rng=np.random.default_rng(11),
    )
    # The test temporarily aligns the cache dimensions to production defaults.
    production_cache = materialize_relief_geometry_cache(
        tmp_path / "relief_production", matrix=matrix,
        feature_names=[f"f{i}" for i in range(matrix.shape[1])],
        work_row_ids=rows, random_state=11,
        anchor_max_rows=lp.LGBM_RELIEF_ANCHOR_MAX_ROWS,
        neighbor_candidate_rows=lp.LGBM_RELIEF_NEIGHBOR_CANDIDATES,
        training_candidate_ids=[f"c{i}" for i in rows],
        fold_lineage_sha256="f" * 64,
    )
    production_cached = relief_scores_from_geometry(
        production_cache, labels, neighbors=lp.LGBM_RELIEF_NEIGHBORS
    )
    np.testing.assert_allclose(production_cached, fresh, rtol=1e-6, atol=1e-6)

    with pytest.raises(ValueError, match="exact training matrix"):
        materialize_relief_geometry_cache(
            tmp_path / "relief_bad_scope", matrix=matrix,
            feature_names=[f"f{i}" for i in range(matrix.shape[1])],
            work_row_ids=rows, training_candidate_ids=[f"c{i}" for i in rows[:-1]],
            fold_lineage_sha256="f" * 64, random_state=11,
            anchor_max_rows=32, neighbor_candidate_rows=60,
        )
    with pytest.raises(ValueError, match="contract drift"):
        materialize_relief_geometry_cache(
            tmp_path / "relief", matrix=matrix,
            feature_names=[f"f{i}" for i in range(matrix.shape[1])],
            work_row_ids=rows, training_candidate_ids=[f"c{i}" for i in rows],
            fold_lineage_sha256="e" * 64, random_state=11,
            anchor_max_rows=32, neighbor_candidate_rows=60,
        )


def test_meta_offset_cache_parity_leakage_guard_and_tamper(tmp_path) -> None:
    identity = _identity(18)
    p_clear = np.linspace(0.2, 0.8, len(identity), dtype=np.float32)
    p_adverse = np.linspace(0.6, 0.1, len(identity), dtype=np.float32)
    p_weak = 1.0 - p_clear - p_adverse
    probability = np.column_stack([p_adverse, p_weak, p_clear]).astype(np.float32)
    offset = np.linspace(-20, 40, len(identity), dtype=np.float32)
    folds = np.repeat(np.arange(3), 6).astype(np.int16)
    map_audit = pd.DataFrame({
        "candidate_id": identity.candidate_id,
        "side": "long",
        "prior_resolved_global_support": np.arange(len(identity)),
        "prior_resolved_bin_support": np.arange(len(identity)) + 500,
        "value_map_fallback": "bin",
        "value_map_max_label_available_ts": pd.Series(
            pd.NaT, index=identity.index, dtype="datetime64[ns, UTC]"
        ),
        "prequential_base_expected_net_bps": offset,
        "shrinkage": 0.5,
    })
    map_audit.loc[0, "value_map_fallback"] = "neutral_no_prior_resolved_support"
    map_audit.loc[1:, "value_map_max_label_available_ts"] = (
        identity.loc[1:, "decision_ts"] - pd.Timedelta(minutes=1)
    ).to_numpy()
    support = causal_mapping_support_frame(map_audit, identity)
    assert "prequential_base_expected_net_bps" not in support
    cache = materialize_meta_offset_cache(
        tmp_path / "meta", identity=identity,
        base_oof_probabilities=probability, base_expected_net_bps=offset,
        fold_ids=folds, mapping_support=support,
        target_contract_sha256="1" * 64, economics_sha256="2" * 64,
        base_oof_sha256="3" * 64, folds_sha256="4" * 64,
        feature_contract=["regime", "trust"], side="long",
    )
    baseline = np.linspace(-2, 2, len(identity), dtype=np.float32)
    permuted = baseline[::-1].copy()
    audit = assert_fixed_offset_parity(cache, baseline, permuted)
    assert audit["max_delta_parity_error_bps"] <= 1e-5
    np.testing.assert_allclose(cache.reconstructed_score(baseline), offset + baseline)
    kwargs, provenance = meta_mda_fixed_offset_kwargs(cache)
    assert kwargs["frozen_base_expected_net_units"] == "bps"
    assert not kwargs["frozen_base_expected_net_bps"].flags.writeable
    assert provenance["request_sha256"] == cache.manifest["request_sha256"]
    # The cache binds target/economics/base/folds/features, so a changed target
    # cannot be relabelled as a cache hit.
    with pytest.raises(ValueError, match="request lineage drift"):
        materialize_meta_offset_cache(
            tmp_path / "meta", identity=identity,
            base_oof_probabilities=probability, base_expected_net_bps=offset,
            fold_ids=folds, mapping_support=support,
            target_contract_sha256="9" * 64, economics_sha256="2" * 64,
            base_oof_sha256="3" * 64, folds_sha256="4" * 64,
            feature_contract=["regime", "trust"], side="long",
        )
    path = cache.root / "base_expected_net_bps.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="artifact drift"):
        load_meta_offset_cache(cache.root)


def test_meta_offset_cache_rejects_side_fold_and_future_support(tmp_path) -> None:
    identity = _identity(9)
    probability = np.tile(np.array([[0.2, 0.5, 0.3]], dtype=np.float32), (9, 1))
    offset = np.arange(9, dtype=np.float32)
    audit = pd.DataFrame({
        "candidate_id": identity.candidate_id,
        "side": "long",
        "prior_resolved_global_support": np.ones(9),
        "prior_resolved_bin_support": np.ones(9),
        "value_map_fallback": "bin",
        "value_map_max_label_available_ts": identity.decision_ts - pd.Timedelta(minutes=1),
    })
    support = causal_mapping_support_frame(audit, identity)
    with pytest.raises(ValueError, match="burn-in/unresolved"):
        materialize_meta_offset_cache(
            tmp_path / "bad_fold", identity=identity,
            base_oof_probabilities=probability, base_expected_net_bps=offset,
            fold_ids=np.r_[np.arange(8), -1], mapping_support=support,
            target_contract_sha256="1" * 64, economics_sha256="2" * 64,
            base_oof_sha256="3" * 64, folds_sha256="4" * 64,
            feature_contract=["x"], side="long",
        )
    wrong_side = identity.copy()
    wrong_side.loc[0, "side_name"] = "short"
    with pytest.raises(ValueError, match="same-side"):
        materialize_meta_offset_cache(
            tmp_path / "bad_side", identity=wrong_side,
            base_oof_probabilities=probability, base_expected_net_bps=offset,
            fold_ids=np.arange(9), mapping_support=support,
            target_contract_sha256="1" * 64, economics_sha256="2" * 64,
            base_oof_sha256="3" * 64, folds_sha256="4" * 64,
            feature_contract=["x"], side="long",
        )
    future = audit.copy()
    future["value_map_max_label_available_ts"] = identity.decision_ts
    with pytest.raises(ValueError, match="future/equal"):
        causal_mapping_support_frame(future, identity)


def test_atomic_cache_staging_never_exposes_partial_root(tmp_path) -> None:
    root = tmp_path / "published"
    with pytest.raises(RuntimeError, match="crash"):
        with atomic_cache_staging(root) as staging:
            assert staging is not None
            (staging / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("crash")
    assert not root.exists()
    assert not list(tmp_path.glob(".published.staging.*"))
    assert not (tmp_path / ".published.publish.lock").exists()


def test_production_base_and_meta_helpers_use_hot_matrix_cache(tmp_path, monkeypatch) -> None:
    from scripts import run_stage_i_base_feature_selection as base_cli
    from scripts import run_stage_i_meta_feature_selection as meta_cli

    selector = tmp_path / "selector"
    selector.mkdir()
    identity = _identity(12)
    ledger = identity.copy()
    ledger["label_available_ts"] = ledger.decision_ts + pd.Timedelta(hours=12)
    feature_frame = identity.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    feature_frame["f1"] = np.arange(len(identity), dtype=np.float32)
    feature_frame["f2"] = np.sin(np.arange(len(identity), dtype=np.float32))
    source = selector / "selector_features.parquet"
    feature_frame.to_parquet(source, index=False)
    manifest_path = selector / "manifest.json"
    manifest_path.write_text('{"status":"complete"}\n')
    contract_path = selector / "selector_feature_contract.json"
    # Legacy key remains supported, but the cache stores one canonical order.
    contract_path.write_text('{"features":["f1","f2"]}\n')
    source_sha = sha256(source.read_bytes()).hexdigest()
    integrity = {"selector_features_sha256": source_sha}
    cache_root = tmp_path / "neutral"
    cold, cold_provenance = base_cli._load_target_neutral_matrix(
        selector_dir=selector, cache_dir=cache_root, ledger=ledger,
        selector_manifest_path=manifest_path, selector_contract_path=contract_path,
        selector_artifact_integrity=integrity,
        identity_columns=["candidate_id", "__ts__", "__symbol__"],
    )
    assert cold_provenance["cache_mode"] == "cold"

    original_read_parquet = pd.read_parquet
    def no_source_parquet_decode(path, *args, **kwargs):
        if Path(path) == source:
            raise AssertionError("hot production cache unexpectedly decoded selector parquet")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(base_cli.pd, "read_parquet", no_source_parquet_decode)
    hot, hot_provenance = base_cli._load_target_neutral_matrix(
        selector_dir=selector, cache_dir=cache_root, ledger=ledger,
        selector_manifest_path=manifest_path, selector_contract_path=contract_path,
        selector_artifact_integrity=integrity,
        identity_columns=["candidate_id", "__ts__", "__symbol__"],
    )
    assert hot_provenance["cache_mode"] == "hot"
    np.testing.assert_allclose(hot, cold)
    meta_hot, meta_provenance = meta_cli._load_target_neutral_matrix(
        selector_dir=selector, cache_dir=cache_root, ledger=ledger,
        selector_manifest={
            "artifact_integrity": {"selector_features_sha256": source_sha}
        },
        selector_manifest_path=manifest_path, selector_contract_path=contract_path,
        identity_columns=["candidate_id", "__ts__", "__symbol__"],
    )
    assert meta_provenance["cache_mode"] == "hot"
    np.testing.assert_allclose(meta_hot, cold)


def test_cache_build_benchmark_is_reportable(tmp_path) -> None:
    """Small deterministic benchmark guard; avoids fragile wall-clock gates."""
    identity = _identity(200)
    rng = np.random.default_rng(3)
    features = pd.DataFrame(rng.normal(size=(200, 24)), columns=[f"f{i}" for i in range(24)])
    cache = materialize_target_neutral_cache(
        tmp_path / "cache", identity=identity, features=features,
        feature_names=list(features), selector_manifest_sha256="a" * 64,
        selector_feature_contract_sha256="b" * 64,
        selector_features_sha256="c" * 64, correlation_rows=200,
    )
    cold_bytes = sum(path.stat().st_size for path in cache.root.iterdir() if path.is_file())
    hot = load_target_neutral_cache(cache.root)
    assert cold_bytes > 0
    assert hot.matrix.shape == (200, 24)
    assert isinstance(hot.matrix, np.memmap)
