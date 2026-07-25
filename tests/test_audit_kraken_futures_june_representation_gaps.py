from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_kraken_futures_june_representation_gaps import (
    JuneRepresentationGapAuditError,
    build_audit,
    write_audit_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_inputs(
    tmp_path: Path, *, accepted_ledger_in_june: bool = False
) -> dict[str, Path]:
    context = pd.DataFrame(
        {
            "candidate_id": ["missing-1", "available", "missing-2"],
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01T03:00:00Z",
                    "2026-06-01T04:00:00Z",
                    "2026-06-01T05:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["AAA/USD:USD"] * 3,
            "side_name": ["short"] * 3,
            "gmm_representation_available": [0.0, 1.0, 0.0],
        }
    )
    context_path = tmp_path / "context.parquet"
    context.to_parquet(context_path, index=False)

    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "ohlcv" / "symbol=AAA_USD:USD" / "year=2026"
    raw_dir.mkdir(parents=True)
    timestamps = pd.date_range("2026-05-30T00:00:00Z", "2026-06-01T05:00:00Z", freq="h")
    raw = pd.DataFrame(
        {
            "ts": timestamps,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    # The candidate timestamps remain complete; only their lookback has a gap.
    raw = raw.loc[raw["ts"].ne(pd.Timestamp("2026-06-01T01:00:00Z"))]
    start = int(raw["ts"].min().timestamp())
    end = int(raw["ts"].max().timestamp())
    raw_path = raw_dir / f"part-{start}-{end}.parquet"
    raw.to_parquet(raw_path, index=False)

    prior_scope = tmp_path / "prior_scope.json"
    _write_json(
        prior_scope,
        {
            "schema": "kraken_futures_exact_source_repair_scope_v1",
            "selection": {"top_n": 30},
            "counts": {
                "scoped_missing_source_hours": 6917,
                "scoped_unavailable_candidates": 4227,
            },
        },
    )
    prior_manifest = tmp_path / "prior_manifest.json"
    _write_json(
        prior_manifest,
        {
            "accepted_candle_ledger": {"rows": 6917},
            "endpoint_responses": {"records": 30},
        },
    )
    accepted_timestamp = (
        "2026-06-01T00:00:00Z" if accepted_ledger_in_june else "2026-05-25T07:00:00Z"
    )
    accepted = pd.DataFrame({"ts": pd.to_datetime([accepted_timestamp] * 94, utc=True)})
    revalidated_ledger = tmp_path / "accepted.parquet"
    accepted.to_parquet(revalidated_ledger, index=False)
    revalidated_manifest = tmp_path / "revalidated_manifest.json"
    _write_json(
        revalidated_manifest,
        {
            "status": "REVALIDATED_EXACT_SOURCE_PATCH_NOT_APPLIED",
            "network_calls": 0,
            "accepted_candle_ledger": {
                "rows": 94,
                "sha256": _sha256(revalidated_ledger),
            },
            "rejected_requested_zero_volume_carry_candles": 6823,
        },
    )
    return {
        "context": context_path,
        "raw_root": raw_root,
        "raw_path": raw_path,
        "prior_scope": prior_scope,
        "prior_manifest": prior_manifest,
        "revalidated_manifest": revalidated_manifest,
        "revalidated_ledger": revalidated_ledger,
    }


def _build(paths: dict[str, Path]):
    return build_audit(
        context_path=paths["context"],
        raw_root=paths["raw_root"],
        prior_scope_path=paths["prior_scope"],
        prior_manifest_path=paths["prior_manifest"],
        revalidated_manifest_path=paths["revalidated_manifest"],
        revalidated_ledger_path=paths["revalidated_ledger"],
        expected_counts=None,
    )


def test_audit_is_read_only_and_treats_lookback_gaps_as_association_not_recoverability(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    before = paths["raw_path"].read_bytes()

    report, gaps, exceptions = _build(paths)

    observed = report["observation"]
    assert observed["short_june_candidates"] == 3
    assert observed["short_june_available_candidates"] == 1
    assert observed["short_june_unavailable_candidates"] == 2
    assert observed["target_timestamp_complete_ohlcv"] == 3
    assert observed["unavailable_with_prior_24h_raw_gap"] == 2
    assert observed["unavailable_with_prior_48h_raw_gap"] == 2
    assert observed["associated_distinct_prior_24h_raw_gaps"] == 1
    assert observed["associated_prior_24h_raw_gap_symbols"] == 1
    assert gaps["affected_unavailable_candidate_count"].tolist() == [2]
    assert exceptions.empty
    assert report["network_calls"] == 0
    assert report["baseline_raw_store_mutated"] is False
    assert (
        report["interpretation"]["preceding_raw_gap_overlap_is_recoverability_evidence"]
        is False
    )
    assert paths["raw_path"].read_bytes() == before


def test_artifact_is_atomic_and_refuses_to_overwrite(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    report, gaps, exceptions = _build(paths)
    destination = tmp_path / "audit"

    write_audit_artifact(
        destination=destination,
        report=report,
        gap_table=gaps,
        exceptions=exceptions,
    )

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "READ_ONLY_NO_BROAD_BACKFILL_RECOMMENDED"
    assert manifest["outputs"]["associated_prior_24h_raw_gaps"]["rows"] == 1
    with pytest.raises(JuneRepresentationGapAuditError, match="refusing to overwrite"):
        write_audit_artifact(
            destination=destination,
            report=report,
            gap_table=gaps,
            exceptions=exceptions,
        )


def test_prior_evidence_fails_closed_if_revalidated_ledger_contains_june_rows(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, accepted_ledger_in_june=True)

    with pytest.raises(
        JuneRepresentationGapAuditError, match="contains June accepted candles"
    ):
        _build(paths)
