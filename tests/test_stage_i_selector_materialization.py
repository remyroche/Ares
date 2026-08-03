from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _feature_contract_digest,
)
from scripts.materialize_stage_i_selector_sample import (
    _block_path,
    _checkpoint_field_groups,
    _load_or_create_feature_block,
    _measure_exact_selector_block,
    _contract_with_exact_rejections,
    _subset_contract,
)


def _contract(fields: list[str]) -> FrozenFeatureContract:
    digest = "0" * 64
    kwargs = {
        "feature_columns": fields,
        "candidate_universe_sha256": digest,
        "source_schema_sha256": digest,
        "raw_allowlist_sha256": digest,
        "generator_registry_sha256": digest,
        "store_scan_manifest_sha256": digest,
        "coverage_profile_sha256": digest,
        "min_exact_key_coverage": 1.0,
        "min_non_null_feature_coverage": 0.9,
        "max_feature_columns": 20,
        "coverage_admission_rejections": (),
    }
    return FrozenFeatureContract(
        **kwargs,
        feature_contract_sha256=_feature_contract_digest(**kwargs),
    )


def test_coverage_rejection_only_invalidates_its_original_checkpoint(tmp_path):
    fields = ["a", "b", "c", "d", "e", "f"]
    first = _checkpoint_field_groups(fields, ["b"], 2)
    second = _checkpoint_field_groups(fields, ["b", "d"], 2)

    assert first == [(0, ["a"]), (1, ["c", "d"]), (2, ["e", "f"])]
    assert second == [(0, ["a"]), (1, ["c"]), (2, ["e", "f"])]
    assert _block_path(tmp_path, *first[2]) == _block_path(tmp_path, *second[2])


def test_subset_contract_is_self_verifying():
    subset = _subset_contract(_contract(["a", "b", "c"]), ["a", "c"])
    round_trip = FrozenFeatureContract.from_mapping(subset.to_dict())

    assert round_trip.feature_columns == ("a", "c")
    assert round_trip.feature_contract_sha256 == subset.feature_contract_sha256


def test_completed_feature_block_is_reused_without_calling_loader(tmp_path):
    ledger = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "__ts__": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
        }
    )
    block_path = tmp_path / "block.parquet"
    calls = []

    def loader(identity, fields):
        calls.append(tuple(fields))
        output = identity[["candidate_id", "__ts__", "__symbol__"]].copy()
        output["feature_a"] = [1.0, 2.0]
        return output

    first, reused_first = _load_or_create_feature_block(
        ledger,
        block_fields=["feature_a"],
        block_path=block_path,
        loader=loader,
    )
    second, reused_second = _load_or_create_feature_block(
        ledger,
        block_fields=["feature_a"],
        block_path=block_path,
        loader=lambda *_: (_ for _ in ()).throw(AssertionError("loader called")),
    )

    assert calls == [("feature_a",)]
    assert not reused_first and reused_second
    pd.testing.assert_frame_equal(first, second)


def test_causal_warmup_prefix_uses_post_readiness_and_evaluation_gates() -> None:
    n = 100
    ts = pd.date_range("2023-01-01", periods=n, freq="10D", tz="UTC")
    ledger = pd.DataFrame({
        "candidate_id": [f"id-{i}" for i in range(n)], "__ts__": ts,
        "__symbol__": "BTC/USD:USD",
        "side_name": np.where(np.arange(n) % 2, "short", "long"),
    })
    loaded = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    # Overall coverage is only 85%, but the all-null prefix ends before the
    # required 2024 evaluation window and all later observations are ready.
    loaded["oi_dominance"] = np.r_[np.full(15, np.nan), np.arange(85)]
    measured, audit, detail = _measure_exact_selector_block(
        ledger, loaded, ["oi_dominance"], block_index=3
    )
    row = audit.iloc[0]
    assert row.finite_coverage == 0.85
    assert row.status == "accepted"
    assert row.reason.startswith("causal_warmup_prefix")
    assert row.prefix_rows == 15
    assert row.post_readiness_finite_coverage == 1.0
    assert row.required_evaluation_finite_coverage == 1.0
    assert "oi_dominance" in measured
    assert measured.oi_dominance.iloc[:15].isna().all()
    assert not detail.empty and not detail.hard_gate_applied.any()


def test_sporadic_post_readiness_missingness_below_gate_is_rejected() -> None:
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    ledger = pd.DataFrame({
        "candidate_id": [f"id-{i}" for i in range(n)], "__ts__": ts,
        "__symbol__": "BTC/USD:USD", "side_name": "long",
    })
    loaded = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    values = np.arange(n, dtype=float)
    values[10::4] = np.nan
    loaded["sporadic"] = values
    measured, audit, _ = _measure_exact_selector_block(
        ledger, loaded, ["sporadic"], block_index=1
    )
    assert audit.iloc[0].status == "rejected"
    assert audit.iloc[0].reason == "post_readiness_finite_coverage_below_0.90"
    assert "sporadic" not in measured


def test_revised_contract_binds_exact_rejection_reason() -> None:
    original = _contract(["a", "b", "c"])
    revised = _contract_with_exact_rejections(
        original,
        retained_fields=["a", "c"],
        rejection_reasons={"b": "post_readiness_finite_coverage_below_0.90"},
    )
    round_trip = FrozenFeatureContract.from_mapping(revised.to_dict())
    assert round_trip.feature_columns == ("a", "c")
    assert dict(round_trip.coverage_admission_rejections)["b"].startswith("post_readiness")
    assert round_trip.feature_contract_sha256 != original.feature_contract_sha256
