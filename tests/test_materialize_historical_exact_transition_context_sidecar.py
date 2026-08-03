from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_historical_exact_transition_context_sidecar import (
    HistoricalTransitionContextError,
    IDENTITY,
    SOURCE_FAMILY_AVAILABLE,
    SOURCE_FAMILY_UNAVAILABLE,
    build_sidecar,
    candidate_decision_time,
    coverage_by_year_month_side,
    run,
)


FEATURES = ("market_pressure", "transition_new__breadth__delta_3h", "state_context__current_state")


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "__ts__": pd.to_datetime([
            "2022-12-30 23:00Z", "2022-12-31 00:00Z", "2023-01-01 00:00Z", "2023-01-01 01:00Z",
        ]),
        "__symbol__": ["A", "B", "C", "D"],
        "side_name": ["long", "short", "long", "short"],
        "candidate_id": ["a", "b", "c", "d"],
    })


def _transition() -> pd.DataFrame:
    source = pd.to_datetime(["2022-12-30 23:00Z", "2022-12-31 00:00Z"])
    return pd.DataFrame({
        "source_utc": source,
        "execution_decision_utc": source + pd.Timedelta(hours=1),
        "market_pressure": [1.0, 2.0],
        "transition_new__breadth__delta_3h": [3.0, 4.0],
        "state_context__current_state": [5.0, 6.0],
        "target__onset_within_1h": [0, 1],
    })


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_preserves_identity_order_exact_join_and_nan_unavailability() -> None:
    candidates = _candidates().iloc[[1, 0, 3, 2]].reset_index(drop=True)
    result = build_sidecar(candidates, _transition(), feature_columns=FEATURES, expected_rows=4)

    assert result[list(IDENTITY)].equals(candidates[list(IDENTITY)])
    assert result["transition_context_available"].tolist() == [True, True, False, False]
    assert result["source_family"].tolist() == [SOURCE_FAMILY_AVAILABLE, SOURCE_FAMILY_AVAILABLE, SOURCE_FAMILY_UNAVAILABLE, SOURCE_FAMILY_UNAVAILABLE]
    assert result.loc[~result["transition_context_available"], list(FEATURES)].isna().all(axis=None)
    assert result.loc[result["transition_context_available"], "market_pressure"].tolist() == [2.0, 1.0]
    assert "target__onset_within_1h" not in result


def test_candidate_decision_time_rejects_disagreement() -> None:
    candidates = _candidates()
    candidates["__decision_ts__"] = candidates["__ts__"] + pd.Timedelta(hours=2)
    with pytest.raises(HistoricalTransitionContextError, match="must equal"):
        candidate_decision_time(candidates)


def test_build_rejects_non_one_hour_transition_time_contract() -> None:
    transition = _transition()
    transition.loc[1, "execution_decision_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(HistoricalTransitionContextError, match=r"source_utc \+ 1h"):
        build_sidecar(_candidates(), transition, feature_columns=FEATURES, expected_rows=4)


def test_build_rejects_outcome_feature_even_if_requested() -> None:
    with pytest.raises(HistoricalTransitionContextError, match="prohibited"):
        build_sidecar(_candidates(), _transition(), feature_columns=("target__onset_within_1h",), expected_rows=4)


def test_coverage_reports_year_month_side() -> None:
    sidecar = build_sidecar(_candidates(), _transition(), feature_columns=FEATURES, expected_rows=4)
    cells = coverage_by_year_month_side(sidecar)
    assert cells == [
        {"year": 2022, "month": 12, "side_name": "long", "candidate_rows": 1, "covered_rows": 1, "unavailable_rows": 0, "coverage": 1.0},
        {"year": 2022, "month": 12, "side_name": "short", "candidate_rows": 1, "covered_rows": 1, "unavailable_rows": 0, "coverage": 1.0},
        {"year": 2023, "month": 1, "side_name": "long", "candidate_rows": 1, "covered_rows": 0, "unavailable_rows": 1, "coverage": 0.0},
        {"year": 2023, "month": 1, "side_name": "short", "candidate_rows": 1, "covered_rows": 0, "unavailable_rows": 1, "coverage": 0.0},
    ]


def test_run_hash_binds_sources_and_writes_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidates_path = tmp_path / "candidates.parquet"; _candidates().to_parquet(candidates_path, index=False)
    transition_path = tmp_path / "transition.parquet"; _transition().to_parquet(transition_path, index=False)
    candidate_manifest = tmp_path / "candidate_manifest.json"
    candidate_manifest.write_text(json.dumps({"outputs": {"candidates": {"sha256": _sha(candidates_path)}}}))
    transition_manifest = tmp_path / "transition_manifest.json"
    transition_manifest.write_text(json.dumps({
        "outputs_sha256": {"hourly_transition_dataset.parquet": _sha(transition_path)},
        "research_only": True, "promotion_evidence": False, "full_schema_matches_frozen_v3": True,
    }))
    # A synthetic schema is intentionally smaller than the production catalog.
    monkeypatch.setattr(
        "scripts.materialize_historical_exact_transition_context_sidecar.decision_feature_catalog",
        lambda columns, strict_frozen_contract: FEATURES,
    )
    destination = tmp_path / "sidecar"
    report = run(
        candidates_path=candidates_path, candidate_manifest_path=candidate_manifest,
        transition_path=transition_path, transition_manifest_path=transition_manifest,
        destination=destination, expected_rows=4, expected_covered_rows=2,
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert report["coverage"]["unavailable_rows"] == 2
    assert manifest["output"]["sha256"] == _sha(destination / "context.parquet")
    assert (destination / "manifest.sha256").read_text() == f"{_sha(destination / 'manifest.json')}  manifest.json\n"
    assert not list(tmp_path.glob(".sidecar.staging-*"))
