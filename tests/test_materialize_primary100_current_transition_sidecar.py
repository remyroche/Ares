from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.materialize_primary100_current_transition_sidecar import (
    AVAILABILITY,
    CURRENT_SOURCE,
    HORIZON_HOURS,
    PANEL_SCHEMA,
    PANEL_STATUS,
    SCHEMA,
    TransitionSidecarError,
    UNIVERSE_SCHEMA,
    UNIVERSE_STATUS,
    _sha256,
    build_sidecar,
    run,
)


FEATURES = ["context__past_state", "context__mapping_current"]


def _universe() -> pd.DataFrame:
    # Deliberately non-chronological: output must retain this frozen order.
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-05-02T00:00:00Z", "2026-05-01T00:00:00Z", "2026-05-03T00:00:00Z"]),
            "__symbol__": ["B", "A", "C"],
            "side_name": ["short", "long", "long"],
            "candidate_id": ["b", "a", "c"],
            "execution_decision_utc": pd.to_datetime(["2026-05-02T01:00:00Z", "2026-05-01T01:00:00Z", "2026-05-03T01:00:00Z"]),
        }
    )


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_family": [CURRENT_SOURCE, CURRENT_SOURCE, CURRENT_SOURCE, "canonical_spread_febapr2025"],
            "horizon_hours": [HORIZON_HOURS, HORIZON_HOURS, 3, HORIZON_HOURS],
            "cohort_anchor_utc": pd.to_datetime(["2026-05-01T01:00:00Z", "2026-05-02T01:00:00Z", "2026-05-03T01:00:00Z", "2026-05-03T01:00:00Z"]),
            "context_available": [True, True, True, True],
            "context__past_state": [1.0, 2.0, 30.0, 300.0],
            "context__mapping_current": [10.0, 20.0, 300.0, 3_000.0],
            "target__adverse_transition_any": [1.0, 0.0, 1.0, 0.0],
            "future__context": [99.0, 99.0, 99.0, 99.0],
        }
    )


def test_build_sidecar_broadcasts_exact_h12_anchor_and_preserves_unmatched_nans() -> None:
    universe = _universe()
    result = build_sidecar(
        universe, _panel(), FEATURES, expected_rows=3, expected_unmatched_rows=1
    )
    assert result["candidate_id"].tolist() == ["b", "a", "c"]
    assert result[AVAILABILITY].tolist() == [True, True, False]
    assert result["context__past_state"].tolist()[:2] == [2.0, 1.0]
    assert result.loc[2, FEATURES].isna().all()
    assert list(result.columns) == [
        "__ts__", "__symbol__", "side_name", "candidate_id", "execution_decision_utc", AVAILABILITY, *FEATURES
    ]


def test_build_sidecar_refuses_duplicate_current_h12_anchor() -> None:
    panel = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(TransitionSidecarError, match="exactly one row per cohort anchor"):
        build_sidecar(_universe(), panel, FEATURES, expected_rows=3, expected_unmatched_rows=None)


def _write_contracts(tmp_path, *, features=FEATURES) -> tuple:
    universe_path = tmp_path / "universe.parquet"
    panel_path = tmp_path / "transition_research_panel.parquet"
    _universe().iloc[:2].to_parquet(universe_path, index=False)
    panel = _panel()
    for index, feature in enumerate(features):
        if feature not in panel:
            panel[feature] = float(index + 1)
    panel.to_parquet(panel_path, index=False)
    universe_manifest = tmp_path / "universe_manifest.json"
    universe_manifest.write_text(json.dumps({
        "schema": UNIVERSE_SCHEMA,
        "status": UNIVERSE_STATUS,
        "outputs": {"universe": {"sha256": _sha256(universe_path)}},
    }))
    panel_manifest = tmp_path / "panel_manifest.json"
    panel_manifest.write_text(json.dumps({
        "schema": PANEL_SCHEMA,
        "status": PANEL_STATUS,
        "feature_columns": features,
        "feature_count": len(features),
        "target_columns": ["target__adverse_transition_any"],
        "outputs": {"panel": {"sha256": _sha256(panel_path)}},
    }))
    sidecar = tmp_path / "panel_manifest.sha256"
    sidecar.write_text(f"{_sha256(panel_manifest)}  manifest.json\n")
    return universe_path, universe_manifest, panel_path, panel_manifest, sidecar


def test_run_validates_contracts_and_atomically_writes_hash_bound_coverage(tmp_path) -> None:
    universe, universe_manifest, panel, panel_manifest, sidecar = _write_contracts(tmp_path)
    destination = tmp_path / "output"
    report = run(
        universe_path=universe,
        universe_manifest_path=universe_manifest,
        panel_path=panel,
        panel_manifest_path=panel_manifest,
        panel_manifest_sidecar_path=sidecar,
        destination=destination,
        expected_rows=2,
        expected_unmatched_rows=0,
    )
    assert report["schema"] == SCHEMA
    assert (destination / "manifest.sha256").read_text().split()[0] == _sha256(destination / "manifest.json")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["output"]["sha256"] == _sha256(destination / "transition_context.parquet")
    assert set(manifest["outputs"]) == {"report", "coverage_by_hour", "coverage_by_month", "coverage_by_side", "feature_coverage"}
    assert pd.read_csv(destination / "coverage_by_side.csv")["rows"].sum() == 2


def test_run_refuses_target_leakage_in_manifest_whitelist(tmp_path) -> None:
    paths = _write_contracts(tmp_path, features=["context__past_state", "target__adverse_transition_any"])
    with pytest.raises(TransitionSidecarError, match="overlaps target_columns"):
        run(
            universe_path=paths[0], universe_manifest_path=paths[1], panel_path=paths[2],
            panel_manifest_path=paths[3], panel_manifest_sidecar_path=paths[4],
            destination=tmp_path / "output", expected_rows=2, expected_unmatched_rows=0,
        )


def test_run_allows_causal_path_and_return_named_context_features(tmp_path) -> None:
    features = [
        "context__state_mean__median__dir_path_edge_2h",
        "context__state_mean__median__spread_proxy_abs_return_bps_robust_z",
    ]
    paths = _write_contracts(tmp_path, features=features)
    report = run(
        universe_path=paths[0], universe_manifest_path=paths[1], panel_path=paths[2],
        panel_manifest_path=paths[3], panel_manifest_sidecar_path=paths[4],
        destination=tmp_path / "output", expected_rows=2, expected_unmatched_rows=0,
    )
    assert report["feature_columns"] == features
