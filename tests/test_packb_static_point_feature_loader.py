from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import packb_static_point_feature_loader as loader


class _RecordingGuard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def preflight(self, stage: str) -> None:
        self.calls.append(("preflight", stage))

    def checkpoint(self, stage: str) -> None:
        self.calls.append(("checkpoint", stage))


def _write_store(
    tmp_path: Path,
    *,
    symbol: str = "BTC/USD:USD",
    stored_symbol: str | None = None,
    timestamps: list[str] | None = None,
    with_delta: bool = True,
) -> Path:
    root = tmp_path / "features"
    root.mkdir(parents=True)
    (root / "_feature_cache_scan_manifest.json").write_text(
        json.dumps(
            {
                "version": 5,
                "input_signature": {"entries_hash": "synthetic", "path_count": 1},
            }
        ),
        encoding="utf-8",
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(
            timestamps
            or [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-03-01T00:00:00Z",
            ],
            utc=True,
        ),
        name="ts",
    )
    frame = pd.DataFrame(
        {
            "__symbol__": stored_symbol or symbol,
            "ret1h": np.arange(1, len(index) + 1, dtype=np.float32),
            "ret24h": [10.0, np.nan, 30.0][: len(index)],
            # These historical store fields must not become raw Pack-B inputs.
            "base_lgbm_fake_score": np.float32(0.3),
            "resid_event_fake_state": np.float32(0.2),
            "unknown_stored_field": np.float32(99.0),
        },
        index=index,
    )
    path = root / f"symbol={symbol.replace('/', '_')}.parquet"
    frame.to_parquet(path)
    if with_delta:
        delta = pd.DataFrame(
            {
                "__symbol__": [stored_symbol or symbol],
                "ret24h": [20.0],
            },
            index=pd.DatetimeIndex([index[1]], name="ts"),
        )
        delta_dir = Path(str(path) + ".deltas")
        delta_dir.mkdir()
        delta.to_parquet(delta_dir / "part-000001.parquet")
    return root


def _ledger(symbol: str = "BTC/USD:USD") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["late", "early", "middle"],
            "__ts__": pd.to_datetime(
                [
                    "2025-03-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": [symbol, symbol, symbol],
        }
    )


def _contract(
    root: Path, ledger: pd.DataFrame, *, segments: bool = False
) -> tuple[
    loader.CandidateFeatureUniverse,
    loader.FeatureCoverageProfile,
    loader.FrozenFeatureContract,
]:
    coverage_segments = None
    required: tuple[str, ...] = ()
    if segments:
        coverage_segments = {
            "begin": ledger.iloc[[1]].copy(),
            "middle": ledger.iloc[[2]].copy(),
            "end": ledger.iloc[[0]].copy(),
        }
        required = ("begin", "middle", "end")
    return loader.build_fresh_causal_feature_contract(
        ledger,
        feature_store_dir=root,
        coverage_sample_rows=32,
        coverage_segments=coverage_segments,
        min_segment_exact_key_coverage=1.0 if segments else None,
        min_segment_non_null_feature_coverage=0.99 if segments else None,
        min_segment_joint_complete_coverage=0.98 if segments else None,
        min_segment_variance=None,
        required_segment_names=required,
        max_rows_per_batch=2,
        max_columns_per_read=1,
    )


def test_provenance_universe_uses_registry_not_stale_store_fields_and_exact_base_delta_join(
    tmp_path: Path,
) -> None:
    root = _write_store(tmp_path)
    ledger = _ledger()
    guard = _RecordingGuard()
    universe, profile, contract = _contract(root, ledger, segments=True)

    assert contract.feature_columns == ("ret1h", "ret24h")
    assert "base_lgbm_fake_score" not in universe.feature_columns
    assert "resid_event_fake_state" not in universe.feature_columns
    assert (
        dict(universe.rejected_columns)["base_lgbm_fake_score"]
        == "prior_model_output_prefix"
    )
    assert dict(universe.rejected_columns)["unknown_stored_field"] == (
        "not_current_generator_registry_allowlist"
    )
    assert universe.store_scan_manifest_sha256
    assert universe.raw_allowlist_sha256
    assert {segment.name for segment in profile.segments} == {"begin", "middle", "end"}
    assert all(segment.joint_complete_fraction == 1.0 for segment in profile.segments)

    result = loader.load_point_in_time_features(
        ledger,
        feature_store_dir=root,
        feature_contract=contract,
        max_rows_per_batch=1,
        max_columns_per_read=1,
        include_identity=True,
        resource_guard=guard,
    )
    assert result["candidate_id"].tolist() == ["late", "early", "middle"]
    assert result["ret1h"].tolist() == [3.0, 1.0, 2.0]
    # Delta repair supplies the missing base value through read_symbol_features.
    assert result["ret24h"].tolist() == [30.0, 10.0, 20.0]
    assert any(event == "preflight" for event, _stage in guard.calls)
    assert any("point_load_batch" in stage for _event, stage in guard.calls)


def test_monthly_and_column_bounded_reads_and_exact_symbol_payload_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_store(tmp_path)
    ledger = _ledger()
    _universe, _profile, contract = _contract(root, ledger)
    calls: list[tuple[pd.Timestamp | None, pd.Timestamp | None, tuple[str, ...]]] = []
    original = loader.read_symbol_features

    def recording_reader(path: str, **kwargs):
        calls.append(
            (
                kwargs.get("start_ts"),
                kwargs.get("end_ts"),
                tuple(kwargs.get("columns") or ()),
            )
        )
        return original(path, **kwargs)

    monkeypatch.setattr(loader, "read_symbol_features", recording_reader)
    loader.load_point_in_time_features(
        ledger,
        feature_store_dir=root,
        feature_contract=contract,
        max_rows_per_batch=8,
        max_columns_per_read=1,
    )
    assert len(calls) == 4  # Jan + Mar, each projected in two feature blocks.
    assert all(
        columns[0] == "__symbol__" and len(columns) <= 2 for _, _, columns in calls
    )
    assert {str(start)[:7] for start, _end, _columns in calls} == {"2025-01", "2025-03"}

    bad_root = _write_store(tmp_path / "bad", stored_symbol="WRONG/SYMBOL")
    with pytest.raises(
        loader.PackBStaticPointFeatureLoaderError, match="mismatched stored"
    ):
        _contract(bad_root, ledger)

    alias_root = _write_store(
        tmp_path / "alias",
        symbol="BTC_USDT",
        stored_symbol="BTCUSDT",
    )
    alias_ledger = _ledger("BTCUSDT")
    _universe, _profile, alias_contract = _contract(alias_root, alias_ledger)
    alias_result = loader.load_point_in_time_features(
        alias_ledger,
        feature_store_dir=alias_root,
        feature_contract=alias_contract,
    )
    assert alias_result.shape == (3, 2)


def test_no_future_or_asof_fallback_and_duplicate_ledger_keys_fail_closed(
    tmp_path: Path,
) -> None:
    root = _write_store(
        tmp_path,
        timestamps=["2025-01-01T01:00:00Z"],
        with_delta=False,
    )
    future_only = pd.DataFrame(
        {
            "candidate_id": ["need-exact"],
            "__ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "__symbol__": ["BTC/USD:USD"],
        }
    )
    # Build the contract from a store-supported row, then prove a prior row
    # cannot use the future bar by reindex/as-of accident.
    supported = future_only.copy()
    supported["__ts__"] = pd.Timestamp("2025-01-01T01:00:00Z")
    _universe, _profile, contract = _contract(root, supported)
    with pytest.raises(
        loader.PackBStaticPointFeatureLoaderError, match="no as-of/future"
    ):
        loader.load_point_in_time_features(
            future_only,
            feature_store_dir=root,
            feature_contract=contract,
        )
    discovery = loader.profile_point_feature_coverage(
        future_only,
        feature_store_dir=root,
        feature_contract=contract,
        coverage_sample_rows=1,
    )
    assert discovery.missing_exact_rows == 1

    duplicate = pd.concat([supported, supported.assign(candidate_id="different")])
    with pytest.raises(
        loader.PackBStaticPointFeatureLoaderError, match="duplicate exact"
    ):
        loader.load_point_in_time_features(
            duplicate,
            feature_store_dir=root,
            feature_contract=contract,
        )


def test_segment_filters_prune_not_abort_and_evidence_binds_callback_and_matrix(
    tmp_path: Path,
) -> None:
    root = _write_store(tmp_path)
    ledger = _ledger()
    universe, profile, contract = _contract(root, ledger, segments=True)
    bundle = loader.write_loader_evidence_bundle(
        output_dir=tmp_path / "evidence",
        universe=universe,
        coverage_profile=profile,
        feature_contract=contract,
        source_revision="a" * 40,
        max_rows_per_batch=2,
        max_columns_per_read=1,
    )
    callback = loader.make_packb_static_feature_loader(
        feature_store_dir=root,
        feature_contract=contract,
        max_rows_per_batch=2,
        max_columns_per_read=1,
        evidence_bundle=bundle,
    )
    matrix = callback(ledger, list(contract.feature_columns))
    assert list(matrix.columns) == list(contract.feature_columns)
    subset = callback(ledger, [contract.feature_columns[-1]])
    assert list(subset.columns) == [contract.feature_columns[-1]]
    assert callback.packb_static_feature_loader_evidence["loader_contract_sha256"] == (
        bundle.loader_contract_sha256
    )
    first_hash = callback.packb_static_feature_matrix_sha256(ledger, matrix)
    changed = matrix.copy()
    changed.iloc[0, 0] += np.float32(1.0)
    assert callback.packb_static_feature_matrix_sha256(ledger, changed) != first_hash
    assert (tmp_path / "evidence" / "raw_feature_universe.json").is_file()
    assert (tmp_path / "evidence" / "loader_evidence.json").is_file()


def test_weak_segment_feature_is_pruned_before_joint_complete_gate(
    tmp_path: Path,
) -> None:
    root = _write_store(tmp_path, with_delta=False)
    path = next(root.glob("symbol=*.parquet"))
    base = pd.read_parquet(path)
    base.loc[base.index[0], "ret24h"] = np.nan
    base.to_parquet(path)
    ledger = _ledger()
    segments = {
        "begin": ledger.iloc[[1]].copy(),
        "middle": ledger.iloc[[2]].copy(),
        "end": ledger.iloc[[0]].copy(),
    }
    universe, profile, contract = loader.build_fresh_causal_feature_contract(
        ledger,
        feature_store_dir=root,
        coverage_sample_rows=16,
        coverage_segments=segments,
        min_non_null_feature_coverage=0.0,
        min_segment_exact_key_coverage=1.0,
        min_segment_non_null_feature_coverage=0.99,
        min_segment_joint_complete_coverage=0.98,
        required_segment_names=("begin", "middle", "end"),
        max_rows_per_batch=2,
        max_columns_per_read=1,
    )
    assert universe.feature_columns == ("ret1h", "ret24h")
    assert contract.feature_columns == ("ret1h",)
    assert dict(contract.coverage_admission_rejections)["ret24h"].startswith(
        "begin:non_null_coverage"
    )
    assert all(segment.joint_complete_fraction == 1.0 for segment in profile.segments)


def test_outcome_free_feature_cap_is_deterministic_and_bound_in_contract(
    tmp_path: Path,
) -> None:
    root = _write_store(tmp_path)
    ledger = _ledger()
    _universe, _profile, contract = loader.build_fresh_causal_feature_contract(
        ledger,
        feature_store_dir=root,
        coverage_sample_rows=16,
        min_non_null_feature_coverage=0.0,
        max_feature_columns=1,
        max_rows_per_batch=2,
        max_columns_per_read=1,
    )
    assert contract.max_feature_columns == 1
    assert contract.feature_columns == ("ret1h",)
    assert dict(contract.coverage_admission_rejections)["ret24h"].startswith(
        "coverage_family_round_robin_rank_"
    )
