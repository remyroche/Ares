import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.materialize_2022_2026_regime_episode_ledger import (
    materialize_regime_episode_ledger,
    sha256,
)


def _write_manifest(directory: Path, payload: dict) -> None:
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "manifest.sha256").write_text(
        f"{sha256(manifest)}  manifest.json\n", encoding="utf-8"
    )


def _hourly(start: str, periods: int, *, state_offset: int = 0) -> pd.DataFrame:
    source = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    state = (np.arange(periods) + state_offset) % 2
    return pd.DataFrame(
        {
            "source_utc": source,
            "execution_decision_utc": source + pd.Timedelta(hours=1),
            "segment_id": 1,
            "observable_strength": np.linspace(-2.0, 2.0, periods),
            "observable_dispersion": np.where(state == 0, 0.25, 1.5),
            "target__pooled_state": state,
            "state_context__current_state": state,
            "state_context__nearest_distance": 0.1,
            "target__phase": np.where(state == 0, "stable", "active"),
            "target__event_id": [None] * periods,
            "target__transition_active": state,
            "target__future_outcome_forbidden": np.arange(periods) + 100.0,
        }
    )


def _events(hourly: pd.DataFrame, prefix: str) -> pd.DataFrame:
    anchor = hourly.loc[1, "source_utc"]
    return pd.DataFrame(
        {
            "event_id": [f"{prefix}_event"],
            "segment_id": [1],
            "anchor_source_utc": [anchor],
            "anchor_decision_utc": [anchor + pd.Timedelta(hours=1)],
            "transition_start_utc": [anchor],
            "transition_end_utc": [anchor + pd.Timedelta(hours=3)],
            "target_available_utc": [anchor + pd.Timedelta(hours=13)],
            "source_state": [0],
            "destination_state": [1],
            "transition_archetype": ["state_0_to_state_1"],
            "label_contract": ["synthetic"],
        }
    )


def _source_pair(root: Path) -> tuple[Path, Path]:
    extension, v3 = root / "extension", root / "v3"
    extension.mkdir()
    v3.mkdir()
    early = _hourly("2022-08-30T00:00:00Z", 4)
    later = _hourly("2022-08-30T04:00:00Z", 2)
    # Deliberately leave 06:00 unavailable in v3, then resume at 07:00.
    later = pd.concat([later, _hourly("2022-08-30T07:00:00Z", 2, state_offset=1)], ignore_index=True)
    early_events, later_events = _events(early, "early"), _events(later, "later")
    early.loc[1, "target__event_id"] = "early_event"
    later.loc[1, "target__event_id"] = "later_event"
    early.to_parquet(extension / "hourly_transition_dataset.parquet", index=False)
    later.to_parquet(v3 / "hourly_transition_dataset.parquet", index=False)
    early_events.to_parquet(extension / "transition_events.parquet", index=False)
    later_events.to_parquet(v3 / "transition_events.parquet", index=False)
    geometry = v3 / "pooled_state_geometry.joblib"
    geometry.write_bytes(b"frozen synthetic geometry")
    v3_manifest = {"schema": "pooled_symmetric_regime_transition_research_v1"}
    _write_manifest(v3, v3_manifest)
    extension_manifest = {
        "schema": "frozen_regime_transition_market_extension_v1",
        "frozen_geometry_reused": True,
        "full_schema_matches_frozen_v3": True,
        "source_interval": {"end_utc_exclusive": "2022-08-30T04:00:00Z"},
        "source_hashes": {
            "frozen_template": {"sha256": sha256(v3 / "hourly_transition_dataset.parquet")},
            "frozen_geometry": {"sha256": sha256(geometry)},
            "frozen_manifest": {"sha256": sha256(v3 / "manifest.json")},
        },
    }
    _write_manifest(extension, extension_manifest)
    return extension, v3


def test_materializes_provenance_bound_calendar_profiles_and_explicit_gaps(tmp_path: Path) -> None:
    extension, v3 = _source_pair(tmp_path)
    output = tmp_path / "out"
    report = materialize_regime_episode_ledger(
        extension_dir=extension,
        v3_dir=v3,
        output_dir=output,
        expected_event_count=2,
    )

    hourly = pd.read_parquet(output / "hourly_state_calendar.parquet")
    episodes = pd.read_parquet(output / "transition_episode_ledger.parquet")
    profiles = pd.read_csv(output / "state_profiles.csv")
    coverage = pd.read_csv(output / "coverage_calendar.csv")
    assert len(hourly) == 8
    assert len(episodes) == 2
    assert report["counts"]["transition_events"] == 2
    assert hourly["source_utc"].dt.tz is not None
    assert hourly["execution_decision_utc"].dt.tz is not None
    assert {"source_state", "destination_state", "transition_archetype", "phase"}.issubset(episodes.columns)
    assert episodes["source_artifact_id"].nunique() == 2
    assert not profiles["feature"].str.startswith("target__").any()
    assert not profiles["feature"].str.startswith("state_context__").any()
    assert "target__future_outcome_forbidden" not in report["state_profiles"]["feature_columns"]
    assert "unavailable_before_frozen_candidate_population" in set(coverage["availability_reason"])
    assert "unavailable_internal_source_gap" in set(coverage["availability_reason"])
    assert "unavailable_after_materialized_v3_coverage" in set(coverage["availability_reason"])
    gap = coverage.loc[coverage["availability_reason"].eq("unavailable_internal_source_gap")].iloc[0]
    assert gap["start_utc"].startswith("2022-08-30 06:00:00")
    assert gap["end_utc_exclusive"].startswith("2022-08-30 07:00:00")
    assert (output / "weekly_regime_summary.csv").exists()
    assert (output / "monthly_regime_summary.csv").exists()
    assert (output / "manifest.sha256").read_text().split()[0] == sha256(output / "manifest.json")


def test_rejects_an_extension_without_the_v3_geometry_hash_binding(tmp_path: Path) -> None:
    extension, v3 = _source_pair(tmp_path)
    manifest = json.loads((extension / "manifest.json").read_text())
    manifest["source_hashes"]["frozen_geometry"]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    _write_manifest(extension, manifest)
    try:
        materialize_regime_episode_ledger(
            extension_dir=extension,
            v3_dir=v3,
            output_dir=tmp_path / "out",
            expected_event_count=2,
        )
    except ValueError as error:
        assert "hash-bound" in str(error)
    else:
        raise AssertionError("expected geometry binding validation to fail")
