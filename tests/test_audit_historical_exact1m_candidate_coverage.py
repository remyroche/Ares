from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_historical_exact1m_candidate_coverage as audit


def test_aggregate_four_partition_verification_is_supported():
    source = Path(
        "scripts/audit_historical_exact1m_candidate_coverage.py"
    ).read_text()
    assert "--aggregate-download-verification-manifest" in source
    assert "failure_2024_exact1m_download_verification_v1" in source
    assert 'payload.get("partition_count", -1)) != 4' in source
    assert "aggregate_download_verification_manifest" in source


def _sha(path: Path) -> str:
    return audit._sha256(path)


def _inputs(tmp_path: Path, *, missing_last: bool = False):
    stage = tmp_path / "stage"
    stage.mkdir()
    starts = [
        pd.Timestamp("2024-01-01T01:00:00Z"),
        pd.Timestamp("2024-01-01T02:00:00Z"),
    ]
    path_map = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "timestamp": starts,
            "symbol": ["BTC/USD:USD", "BTC/USD:USD"],
            "path_end_exclusive": [
                starts[0] + pd.Timedelta(minutes=3),
                starts[1] + pd.Timedelta(minutes=3),
            ],
        }
    )
    path_map_path = stage / "candidate_path_map.parquet"
    path_map.to_parquet(path_map_path, index=False)
    staged = stage / "staged_candidates.parquet"
    pd.DataFrame({"candidate_id": ["a", "b"]}).to_parquet(staged, index=False)
    stage_manifest = {
        "schema": "historical_backcast_exact1m_request_stage_v2",
        "path_horizon_minutes": 3,
        "outputs": {
            "candidate_path_map": {"sha256": _sha(path_map_path)},
            "staged_candidates": {"sha256": _sha(staged)},
        },
    }
    stage_manifest_path = stage / "manifest.json"
    stage_manifest_path.write_text(json.dumps(stage_manifest))

    product = tmp_path / "product"
    product.mkdir()
    requests = product / "requests.parquet"
    pd.DataFrame(
        {
            "timestamp": starts,
            "symbol": ["BTC/USD:USD", "BTC/USD:USD"],
            "product_id": ["PF_XBTUSD", "PF_XBTUSD"],
        }
    ).to_parquet(requests, index=False)
    product_manifest_path = product / "manifest.json"
    product_manifest_path.write_text(
        json.dumps(
            {
                "schema": "kraken_historical_product_map_v1",
                "stage_candidates": {"sha256": _sha(staged)},
                "outputs": {
                    "download_candidates_with_product": {
                        "path": str(requests),
                        "sha256": _sha(requests),
                    }
                },
            }
        )
    )
    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "verify_only": True,
                "candidate_sha256": _sha(requests),
                "stage_manifest": {"sha256": _sha(stage_manifest_path)},
                "product_mapping_contract": "frozen_product_id_from_candidate_input",
                "horizon_minutes": 3,
                "symbols": 1,
                "summary": {
                    "ok_symbols": 1,
                    "incomplete_symbols": 0,
                    "failed_symbols": 0,
                },
            }
        )
    )

    root = tmp_path / "data"
    part_dir = (
        root
        / "exchanges/krakenfutures/execution_1m/ohlcv"
        / "symbol=BTC_USD:USD/year=2024"
    )
    part_dir.mkdir(parents=True)
    index = pd.date_range(starts[0], starts[1] + pd.Timedelta(minutes=2), freq="1min")
    if missing_last:
        index = index[:-1]
    pd.DataFrame(
        {
            "ts": index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    ).to_parquet(part_dir / "part-test-1704070800-1704074520.parquet", index=False)
    return stage, product_manifest_path, strict, root


def _run(monkeypatch, tmp_path, *, missing_last=False):
    stage, product, strict, root = _inputs(tmp_path, missing_last=missing_last)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_historical_exact1m_candidate_coverage.py",
            "--stage-dir",
            str(stage),
            "--product-map-manifest",
            str(product),
            "--strict-download-manifest",
            str(strict),
            "--data-root",
            str(root),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    return audit.main()


def test_candidate_level_audit_proves_every_minute(tmp_path, monkeypatch) -> None:
    assert _run(monkeypatch, tmp_path) == 0
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["candidate_coverage_fraction"] == 1.0
    assert manifest["complete_candidates"] == 2


def test_candidate_level_audit_fails_closed_on_missing_minute(
    tmp_path, monkeypatch
) -> None:
    assert _run(monkeypatch, tmp_path, missing_last=True) == 2
    coverage = pd.read_parquet(tmp_path / "output/candidate_coverage.parquet")
    assert coverage["complete_1m_path"].tolist() == [True, False]
    assert coverage.loc[1, "covered_minutes"] == 2
    assert coverage.loc[1, "first_missing_timestamp"] == pd.Timestamp(
        "2024-01-01T02:02:00Z"
    )


def test_strict_manifest_rejects_legacy_product_resolution(
    tmp_path, monkeypatch
) -> None:
    stage, product, strict, root = _inputs(tmp_path)
    payload = json.loads(strict.read_text())
    payload["product_mapping_contract"] = (
        "legacy_current_catalog_or_pf_fallback_not_historical_lineage_safe"
    )
    strict.write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_historical_exact1m_candidate_coverage.py",
            "--stage-dir",
            str(stage),
            "--product-map-manifest",
            str(product),
            "--strict-download-manifest",
            str(strict),
            "--data-root",
            str(root),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(ValueError, match="frozen product IDs"):
        audit.main()
