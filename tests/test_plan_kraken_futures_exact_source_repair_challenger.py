from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.base_candidate_population import candidate_identity_sha256
from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.plan_kraken_futures_exact_source_repair_challenger import (
    KrakenRepairChallengerPlanError,
    _validate_patch,
    build_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_raw(root: Path, *, include_patch_timestamp: bool = False) -> None:
    store = PartitionedOHLCVStore(str(root), "1h")
    timestamps = ["2026-06-01T00:00:00Z", "2026-06-01T02:00:00Z"]
    if include_patch_timestamp:
        timestamps.append("2026-06-01T01:00:00Z")
    index = pd.DatetimeIndex(pd.to_datetime(sorted(timestamps), utc=True), name="ts")
    store.save_partitioned(
        "AAA/USD:USD",
        pd.DataFrame(
            {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 2.0,
            },
            index=index,
        ),
        defer_compact=True,
    )


def _write_patch(root: Path) -> Path:
    root.mkdir()
    ledger = pd.DataFrame(
        {
            "symbol": ["AAA/USD:USD"],
            "product_id": ["PF_AAAUSD"],
            "ts": pd.to_datetime(["2026-06-01T01:00:00Z"], utc=True),
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [2.0],
        }
    )
    ledger_path = root / "accepted_candle_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    _write_json(
        root / "manifest.json",
        {
            "schema": "kraken_futures_exact_source_repair_revalidated_patch_v1",
            "status": "REVALIDATED_EXACT_SOURCE_PATCH_NOT_APPLIED",
            "baseline_raw_store_mutated": False,
            "network_calls": 0,
            "synthetic_fill": False,
            "accepted_candle_ledger": {"rows": 1, "sha256": _sha256(ledger_path)},
        },
    )
    return ledger_path


def _write_context(root: Path) -> Path:
    root.mkdir()
    context = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01T00:00:00Z",  # before the patch, excluded
                    "2026-06-01T01:00:00Z",  # short missing, included
                    "2026-06-01T02:00:00Z",  # long available, included
                    "2026-06-01T03:00:00Z",  # short missing, included
                ],
                utc=True,
            ),
            "__symbol__": ["AAA/USD:USD"] * 4,
            "side_name": ["short", "short", "long", "short"],
            "candidate_id": ["before", "missing", "available", "later"],
            "gmm_representation_available": [1.0, 0.0, 1.0, 0.0],
        }
    )
    path = root / "context.parquet"
    context.to_parquet(path, index=False)
    _write_json(
        root / "manifest.json",
        {
            "status": "MATERIALIZED_CANONICAL_CONTEXT_WITH_FROZEN_SIDE_AE_GMM",
            "output": {
                "sha256": _sha256(path),
                "rows": len(context),
                "candidate_identity_sha256": candidate_identity_sha256(
                    context,
                    columns=("__ts__", "__symbol__", "side_name", "candidate_id"),
                ),
            },
        },
    )
    return path


def _write_ae(root: Path) -> None:
    for side in ("long", "short"):
        path = root / side / "loader_evidence"
        path.mkdir(parents=True)
        _write_json(
            path / "frozen_feature_contract.json",
            {
                "feature_columns": ["feature_a", "feature_b"],
                "feature_contract_sha256": "a" * 64,
            },
        )


def _write_feature_store(root: Path) -> Path:
    root.mkdir()
    path = root / "symbol=AAA_USD:USD.parquet"
    pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-06-01T00:00:00Z"], utc=True),
            "__symbol__": ["AAA/USD:USD"],
            "feature_a": [1.0],
        }
    ).to_parquet(path, index=False)
    return path


def _inputs(tmp_path: Path, *, raw_has_patch: bool = False) -> dict[str, Path]:
    patch = tmp_path / "patch"
    _write_patch(patch)
    context = tmp_path / "context"
    _write_context(context)
    ae = tmp_path / "ae"
    _write_ae(ae)
    raw = tmp_path / "raw"
    _write_raw(raw, include_patch_timestamp=raw_has_patch)
    features = tmp_path / "features"
    _write_feature_store(features)
    return {
        "patch": patch,
        "context": context,
        "ae": ae,
        "raw": raw,
        "features": features,
    }


def test_build_plan_scopes_available_and_missing_rows_without_mutating_baselines(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    tracked = [
        paths["patch"] / "accepted_candle_ledger.parquet",
        paths["context"] / "context.parquet",
        paths["features"] / "symbol=AAA_USD:USD.parquet",
        *list(paths["raw"].rglob("*.parquet")),
    ]
    before = {path: path.read_bytes() for path in tracked}

    result = build_plan(
        patch_root=paths["patch"],
        context_root=paths["context"],
        ae_root=paths["ae"],
        raw_root=paths["raw"],
        feature_store=paths["features"],
        destination=tmp_path / "plan",
        expected_rows=1,
        expected_ledger_sha256=None,
    )

    scope = pd.read_parquet(tmp_path / "plan/candidate_recompute_scope.parquet")
    assert result["baseline_artifacts_mutated"] is False
    assert result["candidate_recompute_scope"]["rows"] == 3
    assert scope["candidate_id"].tolist() == ["missing", "available", "later"]
    assert set(scope["side_name"]) == {"long", "short"}
    assert (
        result["candidate_recompute_scope"]["by_side"]["long"][
            "baseline_available_rows"
        ]
        == 1
    )
    assert (
        result["candidate_recompute_scope"]["by_side"]["short"][
            "baseline_unavailable_rows"
        ]
        == 2
    )
    assert result["raw_challenger"]["hard_links"] is False
    assert result["feature_challenger"]["hard_links_for_mutable_feature_files"] is False
    assert before == {path: path.read_bytes() for path in tracked}


def test_plan_fails_closed_when_accepted_candle_already_exists_in_baseline_raw(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path, raw_has_patch=True)

    with pytest.raises(
        KrakenRepairChallengerPlanError, match="already has an accepted"
    ):
        build_plan(
            patch_root=paths["patch"],
            context_root=paths["context"],
            ae_root=paths["ae"],
            raw_root=paths["raw"],
            feature_store=paths["features"],
            destination=tmp_path / "plan",
            expected_rows=1,
            expected_ledger_sha256=None,
        )
    assert not (tmp_path / "plan").exists()


def test_patch_validation_rejects_a_noncanonical_digest_even_when_manifest_matches(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)

    with pytest.raises(
        KrakenRepairChallengerPlanError, match="approved carry-filtered"
    ):
        _validate_patch(
            paths["patch"], expected_rows=1, expected_ledger_sha256="0" * 64
        )
