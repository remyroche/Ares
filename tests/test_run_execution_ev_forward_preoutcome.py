from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_execution_ev_forward_preoutcome import (
    COVERAGE_SCHEMA,
    UPDATE_SCHEMA,
    validate_coverage_manifest,
    validate_update_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidates() -> pd.DataFrame:
    decision = pd.to_datetime(
        [
            "2026-07-28T01:00:00Z",
            "2026-07-28T01:00:00Z",
            "2026-07-28T14:00:00Z",
            "2026-07-28T14:00:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "__ts__": decision - pd.Timedelta(hours=1),
            "__symbol__": ["BTC", "ETH", "BTC", "ETH"],
            "side_name": ["long", "short", "long", "short"],
            "execution_decision_utc": decision,
        }
    )


def test_coverage_manifest_is_bound_to_sources_rows_and_both_sides(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    manifest = {
        "schema": COVERAGE_SCHEMA,
        "candidate_features": {
            "path": str(candidate_path),
            "sha256": _sha(candidate_path),
        },
        "source_manifests": [{"path": str(raw), "sha256": _sha(raw)}],
        "days": [
            {
                "utc_date": "2026-07-28T00:00:00Z",
                "candidate_rows": 4,
                "both_sides_complete": True,
                "all_required_point_in_time_features_complete": True,
            }
        ],
    }
    manifest_path = tmp_path / "coverage.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    coverage = validate_coverage_manifest(
        manifest_path,
        candidate_path=candidate_path,
        candidates=candidates,
        spec={
            "first_decision_exclusive_utc": "2026-07-27T23:59:59Z",
            "requested_last_decision_utc": "2026-08-10T23:59:59Z",
        },
    )
    assert coverage["complete"].tolist() == [True]
    manifest["days"][0]["candidate_rows"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row mismatch"):
        validate_coverage_manifest(
            manifest_path,
            candidate_path=candidate_path,
            candidates=candidates,
            spec={
                "first_decision_exclusive_utc": "2026-07-27T23:59:59Z",
                "requested_last_decision_utc": "2026-08-10T23:59:59Z",
            },
        )


def test_update_manifest_requires_exact_resolved_prefix(tmp_path: Path) -> None:
    candidates = _candidates()
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)
    # Only the first two decisions resolve strictly before the final 14:00
    # decision, so both and only those identities are required.
    updates = candidates.iloc[:2].copy()
    updates["execution_label_end_utc"] = (
        updates["execution_decision_utc"] + pd.Timedelta(hours=12)
    )
    update_path = tmp_path / "updates.parquet"
    updates.to_parquet(update_path, index=False)
    policy = tmp_path / "policy.json"
    source = tmp_path / "source.json"
    policy.write_text("{}", encoding="utf-8")
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "schema": UPDATE_SCHEMA,
        "score_binding": "generated_or_verified_by_locked_scorer",
        "candidate_features_sha256": _sha(candidate_path),
        "updates": {"path": str(update_path), "sha256": _sha(update_path)},
        "exact_policy_label_manifest": {
            "path": str(policy),
            "sha256": _sha(policy),
        },
        "source_manifest": {"path": str(source), "sha256": _sha(source)},
    }
    manifest_path = tmp_path / "updates.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_update_manifest(
        manifest_path,
        candidate_path=candidate_path,
        candidates=candidates,
    ) == update_path
    bad = updates.iloc[:1]
    bad.to_parquet(update_path, index=False)
    manifest["updates"]["sha256"] = _sha(update_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        validate_update_manifest(
            manifest_path,
            candidate_path=candidate_path,
            candidates=candidates,
        )
