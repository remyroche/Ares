from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_primary100_context_sidecar import (
    CANDIDATE_FIELDS,
    DAE_FIELDS,
    GMM_GEOMETRY_FIELDS,
    GMM_POSTERIOR_FIELDS,
    GMM_RISK_FIELDS,
    IDENTITY,
    REPRESENTATION_AVAILABILITY,
    REPRESENTATION_FIELDS,
    TRANSITION_OUTPUT_FIELDS,
    Primary100ContextSidecarError,
    build_sidecar,
    run,
)


def _identity(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "__ts__": pd.date_range("2026-05-01", periods=rows, freq="h", tz="UTC"),
        "__symbol__": ["A", "B", "C", "D"][:rows],
        "side_name": ["long", "short", "long", "short"][:rows],
        "candidate_id": [f"id-{index}" for index in range(rows)],
    })


def _feature(rows: int = 4) -> pd.DataFrame:
    frame = _identity(rows)
    frame["capture_candidate__regime_transition_entropy_12h"] = np.linspace(-.3, .3, rows)
    frame["capture_candidate__regime_transition_entropy_48h"] = np.linspace(.1, .7, rows)
    return frame


def _candidate(rows: int = 4) -> pd.DataFrame:
    frame = _identity(rows)
    for index, field in enumerate(CANDIDATE_FIELDS):
        frame[field] = np.linspace(index + .1, index + .4, rows)
    frame["selected_top40"] = True
    frame["prediction_source"] = "outer_oof_fold_model"
    return frame


def _representation(rows: int = 4) -> pd.DataFrame:
    frame = _identity(rows)
    frame[REPRESENTATION_AVAILABILITY] = [1, 0, 1, 1][:rows]
    for index, field in enumerate(REPRESENTATION_FIELDS):
        frame[field] = np.linspace(index + .2, index + .5, rows)
        frame.loc[frame[REPRESENTATION_AVAILABILITY].eq(0), field] = np.nan
    for index, field in enumerate(CANDIDATE_FIELDS):
        frame[field] = np.linspace(index + .1, index + .4, rows)
    return frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    feature = tmp_path / "features.parquet"; _feature().to_parquet(feature, index=False)
    candidate = tmp_path / "candidate.parquet"; _candidate().to_parquet(candidate, index=False)
    representation = tmp_path / "representation.parquet"; _representation().to_parquet(representation, index=False)
    feature_manifest = tmp_path / "features_manifest.json"
    feature_manifest.write_text(json.dumps({"outputs": {"universe": {"sha256": _sha256(feature)}}}))
    candidate_manifest = tmp_path / "candidate_manifest.json"
    candidate_manifest.write_text(json.dumps({"output": {"sha256": _sha256(candidate)}}))
    representation_manifest = tmp_path / "representation_manifest.json"
    representation_manifest.write_text(json.dumps({"output": {"sha256": _sha256(representation)}, "context": {"sha256": _sha256(candidate)}}))
    return feature, feature_manifest, candidate, candidate_manifest, representation, representation_manifest


def test_build_sidecar_preserves_order_whitelist_and_representation_missingness() -> None:
    feature = _feature().iloc[[2, 0, 3, 1]].reset_index(drop=True)
    result = build_sidecar(feature, _candidate(), _representation(), expected_rows=len(feature))

    assert result[list(IDENTITY)].equals(feature[list(IDENTITY)])
    assert len(result) == len(feature)
    assert set(CANDIDATE_FIELDS).issubset(result)
    assert set(TRANSITION_OUTPUT_FIELDS).issubset(result)
    assert set(DAE_FIELDS).issubset(result)
    assert set(GMM_POSTERIOR_FIELDS).issubset(result)
    assert set(GMM_GEOMETRY_FIELDS).issubset(result)
    assert set(GMM_RISK_FIELDS).issubset(result)
    assert np.isfinite(result[list(CANDIDATE_FIELDS) + list(TRANSITION_OUTPUT_FIELDS)].to_numpy(float)).all()
    unavailable = result[REPRESENTATION_AVAILABILITY].eq(0)
    assert unavailable.sum() == 1
    assert result.loc[unavailable, list(REPRESENTATION_FIELDS)].isna().all(axis=None)
    assert result.loc[~unavailable, list(REPRESENTATION_FIELDS)].notna().all(axis=None)
    prohibited = ("label", "outcome", "mae", "mfe", "timing", "wait", "target_price", "action_")
    assert not any(any(token in column.lower() for token in prohibited) for column in result.columns)


def test_build_sidecar_rejects_available_representation_missingness() -> None:
    representation = _representation()
    representation.loc[0, DAE_FIELDS[0]] = np.nan
    with pytest.raises(Primary100ContextSidecarError, match="only when availability=0"):
        build_sidecar(_feature(), _candidate(), representation, expected_rows=4)


def test_build_sidecar_rejects_candidate_lineage_disagreement() -> None:
    representation = _representation()
    representation.loc[0, "base_oof_score"] += .2
    with pytest.raises(Primary100ContextSidecarError, match="disagree"):
        build_sidecar(_feature(), _candidate(), representation, expected_rows=4)


def test_build_sidecar_rejects_incomplete_exact_four_key_coverage() -> None:
    with pytest.raises(Primary100ContextSidecarError, match="lacks complete"):
        build_sidecar(_feature(), _candidate().iloc[:-1], _representation(), expected_rows=4)


def test_run_writes_hash_bound_atomic_sidecar(tmp_path: Path) -> None:
    feature, feature_manifest, candidate, candidate_manifest, representation, representation_manifest = _write_source_bundle(tmp_path)
    destination = tmp_path / "sidecar"
    report = run(
        features_path=feature, features_manifest_path=feature_manifest,
        candidate_context_path=candidate, candidate_manifest_path=candidate_manifest,
        representation_context_path=representation, representation_manifest_path=representation_manifest,
        destination=destination, expected_rows=4,
    )
    output = destination / "context.parquet"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert output.is_file()
    assert report["output"]["rows"] == 4
    assert manifest["output"]["sha256"] == _sha256(output)
    assert manifest["sources"]["candidate_context"]["sha256"] == _sha256(candidate)
    assert "NaNs are preserved" in report["representation_context"]["missingness"]
