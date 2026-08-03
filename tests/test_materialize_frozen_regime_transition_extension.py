from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_transition_research import (
    add_causal_transition_features,
    attach_pooled_states,
    fit_pooled_state_geometry,
    materialize_transition_labels,
)
from scripts.materialize_frozen_regime_transition_extension import (
    _validate_target_availability,
    materialize_extension,
)


def _frozen_source(root: Path) -> tuple[Path, Path]:
    market = root / "market.parquet"
    stamp = pd.date_range("2022-08-28", periods=160, freq="h", tz="UTC")
    raw = pd.DataFrame(
        {
            "negative_breadth_pct": (np.arange(len(stamp)) % 13).astype(float),
            "mkt_regime_change__negative_breadth__delta_1h": 1.0,
            "unavailable_oi": np.arange(len(stamp), dtype=float),
            "__symbol__": "BTC_USD:USD",
        },
        index=stamp,
    )
    raw.index.name = "ts"
    raw.to_parquet(market)
    panel = raw.drop(columns="__symbol__").copy()
    panel.insert(0, "source_utc", panel.index)
    panel.insert(1, "execution_decision_utc", panel.index + pd.Timedelta(hours=1))
    panel.insert(2, "segment_id", 0)
    panel = panel.reset_index(drop=True)
    panel, _ = add_causal_transition_features(panel, stems=["negative_breadth_pct"])
    geometry = fit_pooled_state_geometry(
        panel,
        feature_columns=["negative_breadth_pct", "unavailable_oi"],
    )
    panel = attach_pooled_states(panel, geometry)
    labels = materialize_transition_labels(panel, pd.DataFrame())
    frozen = root / "frozen"
    frozen.mkdir()
    labels.to_parquet(frozen / "hourly_transition_dataset.parquet", index=False)
    joblib.dump(geometry, frozen / "pooled_state_geometry.joblib")
    pd.DataFrame(
        columns=[
            "event_id", "segment_id", "anchor_source_utc", "anchor_decision_utc",
            "transition_start_utc", "transition_end_utc", "target_available_utc",
            "source_state", "destination_state", "transition_archetype",
            "origin_dominance", "destination_dominance", "robust_pre_post_shift",
            "label_contract", "economic_failure_event_within_6h",
            "economic_failure_distance_hours",
        ]
    ).to_parquet(frozen / "transition_events.parquet", index=False)
    (frozen / "manifest.json").write_text("{}\n", encoding="utf-8")
    # Simulate an older source interval where the frozen transform's field is
    # present in the schema but unavailable in the historical feed.
    raw["unavailable_oi"] = float("nan")
    raw.to_parquet(market)
    return market, frozen


def test_extension_reuses_schema_and_records_missing_frozen_field(tmp_path: Path) -> None:
    market, frozen = _frozen_source(tmp_path)
    output = tmp_path / "extension"
    report = materialize_extension(
        frozen_source_dir=frozen,
        market_source=market,
        output_dir=output,
        start="2022-08-30T00:00:00Z",
        end="2022-09-01T00:00:00Z",
    )
    expected = pd.read_parquet(frozen / "hourly_transition_dataset.parquet")
    actual = pd.read_parquet(output / "hourly_transition_dataset.parquet")
    assert actual.columns.tolist() == expected.columns.tolist()
    assert len(actual) == 48
    assert actual["source_utc"].min() == pd.Timestamp("2022-08-30T00:00:00Z")
    assert actual["source_utc"].max() == pd.Timestamp("2022-08-31T23:00:00Z")
    assert report["frozen_geometry_reused"] is True
    assert report["full_schema_matches_frozen_v3"] is True
    assert report["fully_missing_frozen_geometry_fields"] == ["unavailable_oi"]
    assert (output / "manifest.sha256").exists()


def test_event_target_availability_is_fail_closed() -> None:
    anchor = pd.Timestamp("2022-10-01T10:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["event_1"],
            "anchor_source_utc": [anchor],
            "target_available_utc": [anchor + pd.Timedelta(hours=13)],
        }
    )
    labels = pd.DataFrame(
        {
            "target__event_id": ["event_1"],
            "target__available_utc": [anchor + pd.Timedelta(hours=13)],
        }
    )
    _validate_target_availability(labels, events)
    labels.loc[0, "target__available_utc"] = anchor + pd.Timedelta(hours=12)
    with pytest.raises(ValueError, match="hourly target availability"):
        _validate_target_availability(labels, events)
