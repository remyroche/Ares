#!/usr/bin/env python3
"""Extend a frozen transition-research market spine without refitting it.

This runner is deliberately narrower than ``materialize_regime_transition_research``.
It adds a historical interval to the market/transition labels while retaining the
already-fitted v3 state geometry and the v3 column contract.  It never fabricates
the native execution-economics overlay: that overlay needs exact admitted-policy
outcomes and is represented only by an explicit availability audit.

The source panel is read with a 24-hour causal lookback and a 12-hour forward
label buffer.  Only rows in the requested source-time interval are emitted.
Consequently, the first eligible rows retain their exact lag history and the
last eligible rows retain their complete before/after target horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_research import (  # noqa: E402
    add_causal_transition_features,
    attach_pooled_states,
    discover_stabilized_transition_events,
    materialize_event_snapshots,
    materialize_transition_labels,
)


DEFAULT_FROZEN_SOURCE = ROOT / (
    "data_perp/artifacts/regime_transition_research_20260726_v3"
)
DEFAULT_MARKET_SOURCE = ROOT / (
    "data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"
)

_IDENTITY_COLUMNS = {"source_utc", "execution_decision_utc", "segment_id"}
_DERIVED_PREFIXES = ("transition_new__", "target__", "state_context__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _template_contract(template: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Recover the v3 raw and generated field order from its frozen schema."""

    required = _IDENTITY_COLUMNS | {"target__pooled_state"}
    missing = required.difference(template.columns)
    if missing:
        raise ValueError(f"frozen template misses required columns: {sorted(missing)}")
    raw = [
        name
        for name in template.columns
        if name not in _IDENTITY_COLUMNS
        and not name.startswith(_DERIVED_PREFIXES)
    ]
    generated = [
        name for name in template.columns if name.startswith("transition_new__")
    ]
    stems: list[str] = []
    for name in generated:
        pieces = name.split("__")
        if len(pieces) < 3:
            raise ValueError(f"invalid generated transition field in template: {name}")
        stem = pieces[1]
        if stem not in stems:
            stems.append(stem)
    if not raw or not stems:
        raise ValueError("frozen template does not expose raw/transition field contract")
    return raw, stems


def _load_source_window(
    path: Path,
    *,
    raw_columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load exact raw fields plus causal/label buffers from the compact source."""

    raw = pd.read_parquet(path)
    if raw.index.name != "ts":
        if "ts" not in raw.columns:
            raise ValueError("compact transition source requires ts index/column")
        raw = raw.set_index("ts")
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True, errors="coerce"))
    raw = raw.loc[raw.index.notna()].sort_index(kind="stable")
    if raw.index.duplicated().any():
        raise ValueError("compact transition source has duplicate hourly timestamps")
    missing = sorted(set(raw_columns).difference(raw.columns))
    if missing:
        raise ValueError(f"compact transition source misses frozen v3 fields: {missing}")
    # The label has a +12h destination and the largest input lag is 24h.
    buffered_start = start - pd.Timedelta(hours=24)
    buffered_end = end + pd.Timedelta(hours=12)
    raw = raw.loc[(raw.index >= buffered_start) & (raw.index < buffered_end), raw_columns]
    if raw.empty:
        raise ValueError("no compact market rows in requested buffered interval")
    numeric = raw.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    result = numeric.copy()
    result.insert(0, "source_utc", result.index)
    result.insert(1, "execution_decision_utc", result.index + pd.Timedelta(hours=1))
    gap = result["source_utc"].diff().ne(pd.Timedelta(hours=1))
    result.insert(2, "segment_id", gap.cumsum().astype(np.int32).to_numpy())
    return result.reset_index(drop=True)


def _source_coverage(
    panel: pd.DataFrame,
    raw_columns: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    requested = panel.loc[
        panel["source_utc"].ge(start) & panel["source_utc"].lt(end), raw_columns
    ]
    coverage = requested.notna().mean().astype(float)
    return pd.DataFrame(
        {
            "field": raw_columns,
            "requested_row_count": int(len(requested)),
            "non_null_count": [int(requested[name].notna().sum()) for name in raw_columns],
            "coverage": [float(coverage[name]) for name in raw_columns],
            "fully_missing": [bool(coverage[name] == 0.0) for name in raw_columns],
            "frozen_geometry_feature": [False] * len(raw_columns),
        }
    )


def _validate_target_availability(labels: pd.DataFrame, events: pd.DataFrame) -> None:
    """Prove that every emitted event target has its declared +13h availability."""

    if events.empty:
        return
    expected = pd.to_datetime(events["anchor_source_utc"], utc=True) + pd.Timedelta(hours=13)
    actual = pd.to_datetime(events["target_available_utc"], utc=True)
    if not actual.equals(expected):
        raise ValueError("event target availability is not anchor_source_utc + 13h")
    attached = labels.loc[labels["target__event_id"].notna(), ["target__event_id", "target__available_utc"]]
    lookup = events.set_index("event_id")["target_available_utc"]
    mapped = attached["target__event_id"].map(lookup)
    if mapped.isna().any() or not pd.to_datetime(attached["target__available_utc"], utc=True).equals(
        pd.to_datetime(mapped, utc=True)
    ):
        raise ValueError("hourly target availability does not match the owning event")


def materialize_extension(
    *,
    frozen_source_dir: Path,
    market_source: Path,
    output_dir: Path,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    """Materialize a schema-identical, frozen-geometry historical market extension."""

    frozen_source_dir = Path(frozen_source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    template_path = frozen_source_dir / "hourly_transition_dataset.parquet"
    geometry_path = frozen_source_dir / "pooled_state_geometry.joblib"
    manifest_path = frozen_source_dir / "manifest.json"
    for path in (template_path, geometry_path, manifest_path, market_source):
        if not path.exists():
            raise FileNotFoundError(path)
    start_utc, end_utc = _utc(start), _utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")

    template = pd.read_parquet(template_path)
    raw_columns, stems = _template_contract(template)
    geometry = joblib.load(geometry_path)
    geometry_columns = list(geometry.feature_columns)
    if set(geometry_columns).difference(raw_columns):
        raise ValueError("frozen state geometry requires fields absent from v3 template")
    panel = _load_source_window(
        market_source,
        raw_columns=raw_columns,
        start=start_utc,
        end=end_utc,
    )
    coverage = _source_coverage(
        panel, raw_columns, start=start_utc, end=end_utc
    )
    coverage.loc[
        coverage["field"].isin(geometry_columns), "frozen_geometry_feature"
    ] = True

    panel, generated = add_causal_transition_features(panel, stems=stems)
    expected_generated = [name for name in template.columns if name.startswith("transition_new__")]
    actual_generated = [name for name in panel.columns if name.startswith("transition_new__")]
    if actual_generated != expected_generated:
        raise ValueError("reconstructed transition features differ from frozen v3 contract")
    panel = attach_pooled_states(panel, geometry)
    event_template = pd.read_parquet(frozen_source_dir / "transition_events.parquet")
    events_all = discover_stabilized_transition_events(panel)
    if events_all.empty:
        # ``discover_stabilized_transition_events`` intentionally returns a
        # column-less frame when no event survives the symmetric gates.
        events_all = pd.DataFrame(
            columns=[
                name
                for name in event_template.columns
                if not name.startswith("economic_failure_")
            ]
        )
    labels_all = materialize_transition_labels(panel, events_all)
    selected = labels_all.loc[
        labels_all["source_utc"].ge(start_utc) & labels_all["source_utc"].lt(end_utc)
    ].copy()
    # Retain every event whose labelled [-12h,+12h) window intersects emitted rows.
    if events_all.empty:
        events = events_all.copy()
    else:
        anchor = pd.to_datetime(events_all["anchor_source_utc"], utc=True)
        events = events_all.loc[
            anchor.lt(end_utc) & (anchor + pd.Timedelta(hours=12)).gt(start_utc)
        ].copy()
    _validate_target_availability(selected, events)
    snapshots = materialize_event_snapshots(labels_all, events)

    expected_columns = template.columns.tolist()
    if set(selected.columns) != set(expected_columns):
        missing = sorted(set(expected_columns).difference(selected.columns))
        extra = sorted(set(selected.columns).difference(expected_columns))
        raise ValueError(f"extension schema differs from v3; missing={missing}, extra={extra}")
    selected = selected.reindex(columns=expected_columns)
    # v3's event table has economic link fields.  The exact execution overlay is
    # intentionally unavailable for 2022, so retain compatible columns as nulls.
    for name in event_template.columns:
        if name not in events:
            events[name] = np.nan
    events = events.reindex(columns=event_template.columns)

    output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_parquet(output_dir / "hourly_transition_dataset.parquet", index=False, compression="zstd")
    events.to_parquet(output_dir / "transition_events.parquet", index=False, compression="zstd")
    snapshots.to_parquet(output_dir / "transition_event_snapshots.parquet", index=False, compression="zstd")
    coverage.to_parquet(output_dir / "source_feature_coverage.parquet", index=False, compression="zstd")
    source_gaps = panel[["source_utc", "segment_id"]].copy()
    source_gaps["gap_from_prior_hours"] = source_gaps["source_utc"].diff().dt.total_seconds().div(3600.0)
    source_gaps = source_gaps.loc[source_gaps["gap_from_prior_hours"].gt(1.0)]
    source_gaps.to_parquet(output_dir / "source_gap_audit.parquet", index=False, compression="zstd")

    files = [
        "hourly_transition_dataset.parquet",
        "transition_events.parquet",
        "transition_event_snapshots.parquet",
        "source_feature_coverage.parquet",
        "source_gap_audit.parquet",
    ]
    report: dict[str, Any] = {
        "schema": "frozen_regime_transition_market_extension_v1",
        "research_only": True,
        "promotion_evidence": False,
        "walk_forward_required": False,
        "frozen_geometry_reused": True,
        "economics_overlay_available": False,
        "economics_overlay_reason": (
            "no exact admitted-policy execution outcome membership exists for this "
            "period; no economic targets were manufactured"
        ),
        "source_interval": {"start_utc": str(start_utc), "end_utc_exclusive": str(end_utc)},
        "causal_input_buffer_hours": 24,
        "forward_target_buffer_hours": 12,
        "source_hashes": {
            "compact_market_source": {"path": str(market_source), "sha256": sha256(market_source)},
            "frozen_template": {"path": str(template_path), "sha256": sha256(template_path)},
            "frozen_geometry": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "frozen_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        },
        "source_rows": int(len(selected)),
        "source_segments_in_buffer": int(panel["segment_id"].nunique()),
        "source_gaps_inside_requested_interval": int(
            source_gaps.loc[
                source_gaps["source_utc"].ge(start_utc) & source_gaps["source_utc"].lt(end_utc)
            ].shape[0]
        ),
        "full_schema_matches_frozen_v3": True,
        "transition_events_intersecting_interval": int(len(events)),
        "target_availability_contract": "event target_available_utc = anchor_source_utc + 13h",
        "fully_missing_source_fields": coverage.loc[coverage["fully_missing"], "field"].tolist(),
        "fully_missing_frozen_geometry_fields": coverage.loc[
            coverage["fully_missing"] & coverage["frozen_geometry_feature"], "field"
        ].tolist(),
        "missingness_handling": (
            "frozen v3 SimpleImputer is reused; no new fit, fill, or feature selection"
        ),
        "outputs_sha256": {name: sha256(output_dir / name) for name in files},
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.sha256").write_text(
        f"{sha256(manifest)}  manifest.json\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-source-dir", type=Path, default=DEFAULT_FROZEN_SOURCE)
    result.add_argument("--market-source", type=Path, default=DEFAULT_MARKET_SOURCE)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--start", required=True)
    result.add_argument("--end", required=True, help="exclusive UTC source-time end")
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(materialize_extension(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
