from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit_stage_c_oi_funding_lineage_blocker import (
    DEFAULT_SOURCES,
    DISPOSITION,
    SourceFamily,
    audit_archived_sidecars,
    run,
    validate_blocked_inventory,
)


def test_archived_f4_f5_sidecars_have_only_nominal_timestamp_columns() -> None:
    inventory = audit_archived_sidecars()
    assert set(inventory.feature_group) == {"F4_oi_dynamics", "F5_funding_crowding"}
    assert inventory.file_count.gt(0).all()
    assert inventory.row_count.gt(0).all()
    assert not inventory.native_observation_clock_proven.any()
    assert not inventory.native_availability_clock_proven.any()
    assert not inventory.bounded_staleness_proven.any()
    assert not inventory.point_in_time_safe.any()
    assert inventory.disposition.eq(DISPOSITION).all()
    # The persisted timestamp/index is intentionally reported but never
    # promoted to source observation or publication time.
    assert inventory.nominal_timestamp_columns.map(json.loads).map(bool).all()
    missing = inventory.missing_required_fields.map(json.loads)
    assert missing.map(lambda names: "source_observed_ts" in names and "available_ts" in names).all()


def test_inventory_fails_closed_if_a_native_clock_appears() -> None:
    inventory = audit_archived_sidecars()
    inventory.loc[inventory.index[0], "native_observation_clock_proven"] = True
    with pytest.raises(ValueError, match="implement a reviewed bounded-staleness as-of adapter"):
        validate_blocked_inventory(inventory)


def test_blocked_manifest_records_exact_missing_fields_without_mutating_sources(tmp_path) -> None:
    oi_dir = tmp_path / "oi"
    funding_dir = tmp_path / "funding"
    oi_dir.mkdir()
    funding_dir.mkdir()
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    pd.DataFrame({"ts": [timestamp], "open_interest": [1.0]}).to_parquet(oi_dir / "A.parquet", index=False)
    pd.DataFrame({"ts": [timestamp], "funding_rate": [0.001]}).to_parquet(funding_dir / "A.parquet", index=False)
    sources = (
        SourceFamily("oi", "F4_oi_dynamics", "open_interest", oi_dir, "open_interest", ("oi_unit", "unit_conversion_price_ts")),
        SourceFamily("funding", "F5_funding_crowding", "funding", funding_dir, "funding_rate", ("funding_value_kind", "settlement_ts")),
    )
    output = tmp_path / "audit"
    manifest = run(output=output, sources=sources)

    assert manifest["status"] == DISPOSITION
    assert (output / "archived_sidecar_schema_inventory.parquet").exists()
    blocked = json.loads((output / "blocked_source_manifest.json").read_text())
    assert blocked["adapter_created"] is False
    assert blocked["feature_admission_changed"] is False
    for source in blocked["sources"]:
        assert "source_observed_ts" in json.loads(source["missing_required_fields"])
        assert "available_ts" in json.loads(source["missing_required_fields"])


def test_real_source_set_stays_fixed_to_the_two_stage_c_groups() -> None:
    assert tuple(source.feature_group for source in DEFAULT_SOURCES) == (
        "F4_oi_dynamics",
        "F5_funding_crowding",
    )
