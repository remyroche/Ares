#!/usr/bin/env python3
"""Materialize a provenance-bound 2022--2026 market-regime episode calendar.

The two input artifacts share one frozen v3 state geometry.  This runner is a
*calendar*, not a model or an economic routing rule: it concatenates their
already-materialized observable market-state labels only after checking the
schema, detached manifests, and frozen-geometry binding.  It deliberately
records unavailable intervals as unavailable -- never as a stable/normal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXTENSION_DIR = ROOT / "data_perp/artifacts/regime_transition_research_2022augdec_frozen_v1"
DEFAULT_V3_DIR = ROOT / "data_perp/artifacts/regime_transition_research_20260726_v3"

HOURLY_NAME = "hourly_transition_dataset.parquet"
EVENT_NAME = "transition_events.parquet"
MANIFEST_NAME = "manifest.json"
MANIFEST_SIDECAR_NAME = "manifest.sha256"
STATE_COLUMN = "target__pooled_state"
PHASE_COLUMN = "target__phase"
REQUIRED_HOURLY_COLUMNS = {
    "source_utc",
    "execution_decision_utc",
    "segment_id",
    STATE_COLUMN,
    "state_context__current_state",
    PHASE_COLUMN,
    "target__event_id",
    "target__transition_active",
}
REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "anchor_source_utc",
    "anchor_decision_utc",
    "source_state",
    "destination_state",
    "transition_archetype",
    "label_contract",
}
PROFILE_EXCLUDED_PREFIXES = ("target__", "state_context__")
PROFILE_EXCLUDED_COLUMNS = {
    "source_utc",
    "execution_decision_utc",
    "segment_id",
    "source_segment_id",
    "calendar_segment_id",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA256 for an artifact file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_manifest(directory: Path) -> tuple[Path, dict[str, Any]]:
    path = directory / MANIFEST_NAME
    sidecar = directory / MANIFEST_SIDECAR_NAME
    if not path.exists():
        raise FileNotFoundError(f"source requires {MANIFEST_NAME}: {directory}")
    # v3 predates the detached-sidecar convention.  Its manifest is still
    # SHA256-bound into this output and into the extension's frozen-geometry
    # provenance; verify a detached checksum whenever a source publishes one.
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip().split()
        if not expected or expected[0] != sha256(path):
            raise ValueError(f"detached manifest checksum does not verify: {directory}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _assert_manifest_output_hash(manifest: dict[str, Any], name: str, path: Path) -> None:
    """Verify a source-output hash when its source manifest publishes one."""

    expected = manifest.get("outputs_sha256", {}).get(name)
    if expected is not None and expected != sha256(path):
        raise ValueError(f"manifest hash mismatch for {path}")


def _load_hourly(directory: Path) -> pd.DataFrame:
    path = directory / HOURLY_NAME
    if not path.exists():
        raise FileNotFoundError(path)
    result = pd.read_parquet(path)
    missing = REQUIRED_HOURLY_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"hourly source misses required columns: {sorted(missing)}")
    for column in ("source_utc", "execution_decision_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"hourly source has invalid UTC {column}")
    result = result.sort_values("source_utc", kind="stable").reset_index(drop=True)
    if result["source_utc"].duplicated().any():
        raise ValueError(f"hourly source has duplicate source UTC rows: {directory}")
    if not (result["execution_decision_utc"] == result["source_utc"] + pd.Timedelta(hours=1)).all():
        raise ValueError("execution decision timestamps must be source UTC + one hour")
    state = pd.to_numeric(result[STATE_COLUMN], errors="coerce")
    context_state = pd.to_numeric(result["state_context__current_state"], errors="coerce")
    unequal = state.notna() & context_state.notna() & state.ne(context_state)
    if unequal.any():
        raise ValueError("pooled-state label and state-context label disagree")
    return result


def _load_events(directory: Path) -> pd.DataFrame:
    path = directory / EVENT_NAME
    if not path.exists():
        raise FileNotFoundError(path)
    result = pd.read_parquet(path)
    missing = REQUIRED_EVENT_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"event source misses required columns: {sorted(missing)}")
    for column in (
        "anchor_source_utc",
        "anchor_decision_utc",
        "transition_start_utc",
        "transition_end_utc",
        "target_available_utc",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
            if result[column].isna().any():
                raise ValueError(f"event source has invalid UTC {column}")
    if result["event_id"].duplicated().any():
        raise ValueError(f"event source has duplicate event ids: {directory}")
    if not (result["anchor_decision_utc"] == result["anchor_source_utc"] + pd.Timedelta(hours=1)).all():
        raise ValueError("event anchor decision timestamps must be source UTC + one hour")
    return result.sort_values("anchor_source_utc", kind="stable").reset_index(drop=True)


def _verify_extension_geometry_binding(
    extension_manifest: dict[str, Any],
    *,
    v3_dir: Path,
    v3_hourly: Path,
    v3_manifest: Path,
) -> None:
    """Prove that the extension explicitly reused the supplied frozen v3 geometry."""

    if extension_manifest.get("frozen_geometry_reused") is not True:
        raise ValueError("extension manifest does not declare frozen geometry reuse")
    if extension_manifest.get("full_schema_matches_frozen_v3") is not True:
        raise ValueError("extension manifest does not declare v3 schema compatibility")
    hashes = extension_manifest.get("source_hashes", {})
    required = {
        "frozen_template": v3_hourly,
        "frozen_geometry": v3_dir / "pooled_state_geometry.joblib",
        "frozen_manifest": v3_manifest,
    }
    for key, path in required.items():
        record = hashes.get(key, {})
        if not path.exists() or record.get("sha256") != sha256(path):
            raise ValueError(f"extension is not hash-bound to supplied v3 {key}")


def _validate_pair(
    *,
    extension_dir: Path,
    v3_dir: Path,
    extension_hourly: pd.DataFrame,
    v3_hourly: pd.DataFrame,
    extension_events: pd.DataFrame,
    v3_events: pd.DataFrame,
    extension_manifest: dict[str, Any],
    v3_manifest: dict[str, Any],
    expected_event_count: int | None,
) -> None:
    if extension_hourly.columns.tolist() != v3_hourly.columns.tolist():
        raise ValueError("hourly schemas differ; refusing to concatenate state geometries")
    if extension_hourly["source_utc"].max() >= v3_hourly["source_utc"].min():
        raise ValueError("extension and v3 hourly windows overlap or are unordered")
    expected_extension_end = extension_manifest.get("source_interval", {}).get("end_utc_exclusive")
    if expected_extension_end is not None and _utc(expected_extension_end) != v3_hourly["source_utc"].min():
        raise ValueError("extension manifest end does not join the supplied v3 beginning")
    extension_states = set(pd.to_numeric(extension_hourly[STATE_COLUMN], errors="coerce").dropna().astype(int))
    v3_states = set(pd.to_numeric(v3_hourly[STATE_COLUMN], errors="coerce").dropna().astype(int))
    if not extension_states.issubset(v3_states):
        raise ValueError("extension exposes state ids not present in frozen v3 geometry")
    if set(extension_events["event_id"]).intersection(v3_events["event_id"]):
        raise ValueError("extension and v3 event ids overlap")
    event_count = len(extension_events) + len(v3_events)
    if expected_event_count is not None and event_count != expected_event_count:
        raise ValueError(f"expected {expected_event_count} transition events, found {event_count}")
    # The extension manifest binds its own outputs; v3 may predate output hashes.
    _assert_manifest_output_hash(extension_manifest, HOURLY_NAME, extension_dir / HOURLY_NAME)
    _assert_manifest_output_hash(extension_manifest, EVENT_NAME, extension_dir / EVENT_NAME)
    _assert_manifest_output_hash(v3_manifest, HOURLY_NAME, v3_dir / HOURLY_NAME)
    _assert_manifest_output_hash(v3_manifest, EVENT_NAME, v3_dir / EVENT_NAME)


def _with_provenance(
    frame: pd.DataFrame,
    *,
    artifact_id: str,
    artifact_path: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "source_artifact_id", artifact_id)
    result.insert(1, "source_artifact_path", str(artifact_path.parent))
    result.insert(2, "source_artifact_sha256", sha256(artifact_path))
    result.insert(3, "source_manifest_sha256", sha256(manifest_path))
    if "segment_id" in result:
        result = result.rename(columns={"segment_id": "source_segment_id"})
    return result


def _calendar_segments(hourly: pd.DataFrame) -> pd.DataFrame:
    result = hourly.sort_values("source_utc", kind="stable").reset_index(drop=True).copy()
    breaks = result["source_utc"].diff().ne(pd.Timedelta(hours=1))
    result.insert(5, "calendar_segment_id", breaks.cumsum().astype(np.int32))
    return result


def _event_phase_by_id(hourly: pd.DataFrame) -> pd.Series:
    labeled = hourly.loc[hourly["target__event_id"].notna(), ["target__event_id", PHASE_COLUMN]]
    if labeled.empty:
        return pd.Series(dtype="object")
    # An event contains several phases.  The phase at its anchor is attached below;
    # this fallback only protects malformed synthetic inputs.
    return labeled.drop_duplicates("target__event_id", keep="first").set_index("target__event_id")[PHASE_COLUMN]


def _build_episode_ledger(hourly: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    anchor_phase = hourly.set_index("source_utc")[PHASE_COLUMN]
    result["phase"] = result["anchor_source_utc"].map(anchor_phase)
    fallback = _event_phase_by_id(hourly)
    result["phase"] = result["phase"].fillna(result["event_id"].map(fallback)).fillna("unavailable")
    result["episode_start_utc"] = result.get("transition_start_utc", result["anchor_source_utc"])
    result["episode_end_utc"] = result.get("transition_end_utc", result["anchor_source_utc"])
    ordered = [
        "event_id", "source_artifact_id", "source_artifact_path", "source_artifact_sha256",
        "source_manifest_sha256", "source_segment_id", "anchor_source_utc",
        "anchor_decision_utc", "episode_start_utc", "episode_end_utc", "target_available_utc",
        "source_state", "destination_state", "transition_archetype", "phase", "label_contract",
    ]
    return result.reindex(columns=ordered + [name for name in result.columns if name not in ordered]).sort_values(
        "anchor_source_utc", kind="stable"
    ).reset_index(drop=True)


def _profile_input_columns(hourly: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for name in hourly.columns:
        if name in PROFILE_EXCLUDED_COLUMNS or name.startswith(PROFILE_EXCLUDED_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(hourly[name]):
            columns.append(name)
    if not columns:
        raise ValueError("no observable numeric market fields available for state profiles")
    return columns


def _state_profiles(hourly: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return state medians as robust, outcome-free standardized contrasts."""

    features = list(feature_columns)
    values = hourly.loc[:, features].apply(pd.to_numeric, errors="coerce")
    global_median = values.median(axis=0)
    robust_scale = (values.sub(global_median).abs().median(axis=0) * 1.4826).replace(0.0, np.nan)
    records: list[dict[str, Any]] = []
    states = sorted(pd.to_numeric(hourly[STATE_COLUMN], errors="coerce").dropna().astype(int).unique())
    for state in states:
        local = values.loc[pd.to_numeric(hourly[STATE_COLUMN], errors="coerce").eq(state)]
        medians = local.median(axis=0)
        contrast = (medians - global_median) / robust_scale
        ranked = contrast.abs().sort_values(ascending=False, kind="stable")
        ranks = {name: index + 1 for index, name in enumerate(ranked.index)}
        for feature in features:
            value = contrast[feature]
            records.append(
                {
                    "state": int(state),
                    "feature": feature,
                    "state_row_count": int(len(local)),
                    "state_median": medians[feature],
                    "global_median": global_median[feature],
                    "robust_scale_mad_1p4826": robust_scale[feature],
                    "robust_standardized_contrast": value,
                    "contrast_direction": "higher" if value > 0 else "lower" if value < 0 else "neutral",
                    "absolute_contrast_rank": int(ranks[feature]),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["state", "absolute_contrast_rank", "feature"], kind="stable"
    ).reset_index(drop=True)


def _period_summary(hourly: pd.DataFrame, frequency: str) -> pd.DataFrame:
    rows = hourly.copy()
    # Periods are calendar labels, not timestamps.  Drop the UTC annotation only
    # after converting explicitly to UTC so pandas does not silently use host time.
    rows["period_utc"] = (
        rows["source_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period(frequency).astype(str)
    )
    grouped = rows.groupby("period_utc", observed=True, sort=True)
    summary = grouped.agg(
        observed_hours=("source_utc", "size"),
        first_source_utc=("source_utc", "min"),
        last_source_utc=("source_utc", "max"),
        transition_active_hours=("target__transition_active", "sum"),
        labeled_transition_hours=("target__event_id", lambda value: int(value.notna().sum())),
        distinct_states=(STATE_COLUMN, lambda value: int(pd.Series(value).dropna().nunique())),
    ).reset_index()
    state_share = (
        rows.assign(**{STATE_COLUMN: pd.to_numeric(rows[STATE_COLUMN], errors="coerce")})
        .groupby(["period_utc", STATE_COLUMN], observed=True)
        .size()
        .rename("hours")
        .reset_index()
    )
    for state in sorted(pd.to_numeric(rows[STATE_COLUMN], errors="coerce").dropna().astype(int).unique()):
        local = state_share.loc[state_share[STATE_COLUMN].eq(state), ["period_utc", "hours"]].rename(
            columns={"hours": f"state_{state}_hours"}
        )
        summary = summary.merge(local, on="period_utc", how="left")
        summary[f"state_{state}_hours"] = summary[f"state_{state}_hours"].fillna(0).astype(int)
        summary[f"state_{state}_share"] = summary[f"state_{state}_hours"] / summary["observed_hours"]
    return summary


def _coverage_calendar(hourly: pd.DataFrame, coverage_start: pd.Timestamp) -> pd.DataFrame:
    """Summarize available and unavailable coverage intervals without normalizing gaps."""

    rows: list[dict[str, Any]] = []
    source = hourly.sort_values("source_utc", kind="stable")
    available_start = source["source_utc"].min()
    if coverage_start < available_start:
        rows.append(
            {
                "start_utc": coverage_start,
                "end_utc_exclusive": available_start,
                "regime_available": False,
                "availability_reason": "unavailable_before_frozen_candidate_population",
                "source_artifact_id": None,
            }
        )
    for _, local in source.groupby("calendar_segment_id", observed=True, sort=True):
        rows.append(
            {
                "start_utc": local["source_utc"].min(),
                "end_utc_exclusive": local["source_utc"].max() + pd.Timedelta(hours=1),
                "regime_available": True,
                "availability_reason": "materialized_frozen_state_geometry",
                "source_artifact_id": "|".join(local["source_artifact_id"].drop_duplicates()),
            }
        )
    previous_end = source["source_utc"].iloc[0]
    for timestamp in source["source_utc"].iloc[1:]:
        if timestamp > previous_end + pd.Timedelta(hours=1):
            rows.append(
                {
                    "start_utc": previous_end + pd.Timedelta(hours=1),
                    "end_utc_exclusive": timestamp,
                    "regime_available": False,
                    "availability_reason": "unavailable_internal_source_gap",
                    "source_artifact_id": None,
                }
            )
        previous_end = timestamp
    rows.append(
        {
            "start_utc": source["source_utc"].max() + pd.Timedelta(hours=1),
            "end_utc_exclusive": pd.NaT,
            "regime_available": False,
            "availability_reason": "unavailable_after_materialized_v3_coverage",
            "source_artifact_id": None,
        }
    )
    return pd.DataFrame.from_records(rows).sort_values("start_utc", kind="stable").reset_index(drop=True)


def materialize_regime_episode_ledger(
    *,
    extension_dir: Path = DEFAULT_EXTENSION_DIR,
    v3_dir: Path = DEFAULT_V3_DIR,
    output_dir: Path,
    coverage_start: str | pd.Timestamp = "2022-01-01T00:00:00Z",
    expected_event_count: int | None = 157,
) -> dict[str, Any]:
    """Build the only allowed concatenation of the frozen extension and v3 spine."""

    extension_dir, v3_dir, output_dir = Path(extension_dir), Path(v3_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if not extension_dir.exists() or not v3_dir.exists():
        raise FileNotFoundError("both frozen source artifact directories are required")
    extension_manifest_path, extension_manifest = _read_manifest(extension_dir)
    v3_manifest_path, v3_manifest = _read_manifest(v3_dir)
    extension_hourly, v3_hourly = _load_hourly(extension_dir), _load_hourly(v3_dir)
    extension_events, v3_events = _load_events(extension_dir), _load_events(v3_dir)
    _verify_extension_geometry_binding(
        extension_manifest,
        v3_dir=v3_dir,
        v3_hourly=v3_dir / HOURLY_NAME,
        v3_manifest=v3_manifest_path,
    )
    _validate_pair(
        extension_dir=extension_dir,
        v3_dir=v3_dir,
        extension_hourly=extension_hourly,
        v3_hourly=v3_hourly,
        extension_events=extension_events,
        v3_events=v3_events,
        extension_manifest=extension_manifest,
        v3_manifest=v3_manifest,
        expected_event_count=expected_event_count,
    )
    hourly = pd.concat(
        [
            _with_provenance(extension_hourly, artifact_id="frozen_extension_2022augdec_v1", artifact_path=extension_dir / HOURLY_NAME, manifest_path=extension_manifest_path),
            _with_provenance(v3_hourly, artifact_id="pooled_transition_research_v3", artifact_path=v3_dir / HOURLY_NAME, manifest_path=v3_manifest_path),
        ],
        ignore_index=True,
    )
    hourly = _calendar_segments(hourly)
    events = pd.concat(
        [
            _with_provenance(extension_events, artifact_id="frozen_extension_2022augdec_v1", artifact_path=extension_dir / EVENT_NAME, manifest_path=extension_manifest_path),
            _with_provenance(v3_events, artifact_id="pooled_transition_research_v3", artifact_path=v3_dir / EVENT_NAME, manifest_path=v3_manifest_path),
        ],
        ignore_index=True,
    )
    ledger = _build_episode_ledger(hourly, events)
    profile_inputs = _profile_input_columns(hourly)
    profiles = _state_profiles(hourly, profile_inputs)
    weekly = _period_summary(hourly, "W-SUN")
    monthly = _period_summary(hourly, "M")
    coverage = _coverage_calendar(hourly, _utc(coverage_start))

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, pd.DataFrame] = {
        "hourly_state_calendar.parquet": hourly,
        "transition_episode_ledger.parquet": ledger,
        "state_profiles.csv": profiles,
        "weekly_regime_summary.csv": weekly,
        "monthly_regime_summary.csv": monthly,
        "coverage_calendar.csv": coverage,
    }
    for name, frame in outputs.items():
        path = output_dir / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            frame.to_csv(path, index=False)
    report: dict[str, Any] = {
        "schema": "regime_episode_calendar_v1",
        "research_only": True,
        "promotion_evidence": False,
        "purpose": "observable frozen market-state and transition episode calendar; no economic outcome routing",
        "state_geometry": {
            "state_column": STATE_COLUMN,
            "geometry_binding": "extension manifest hashes the supplied v3 template, geometry, and manifest",
            "source_state_ids": sorted(pd.to_numeric(hourly[STATE_COLUMN], errors="coerce").dropna().astype(int).unique()),
            "schema_compatible": True,
        },
        "sources": {
            "frozen_extension_2022augdec_v1": {
                "path": str(extension_dir),
                "manifest_sha256": sha256(extension_manifest_path),
                "manifest_detached_checksum_verified": (extension_dir / MANIFEST_SIDECAR_NAME).exists(),
                "hourly_sha256": sha256(extension_dir / HOURLY_NAME),
                "events_sha256": sha256(extension_dir / EVENT_NAME),
                "hourly_rows": int(len(extension_hourly)),
                "events": int(len(extension_events)),
            },
            "pooled_transition_research_v3": {
                "path": str(v3_dir),
                "manifest_sha256": sha256(v3_manifest_path),
                "manifest_detached_checksum_verified": (v3_dir / MANIFEST_SIDECAR_NAME).exists(),
                "hourly_sha256": sha256(v3_dir / HOURLY_NAME),
                "events_sha256": sha256(v3_dir / EVENT_NAME),
                "hourly_rows": int(len(v3_hourly)),
                "events": int(len(v3_events)),
            },
        },
        "counts": {
            "hourly_rows": int(len(hourly)),
            "transition_events": int(len(ledger)),
            "expected_transition_events": expected_event_count,
            "calendar_segments": int(hourly["calendar_segment_id"].nunique()),
        },
        "coverage_contract": {
            "coverage_start_utc": str(_utc(coverage_start)),
            "available_start_utc": str(hourly["source_utc"].min()),
            "available_end_utc_exclusive": str(hourly["source_utc"].max() + pd.Timedelta(hours=1)),
            "unavailable_before_reason": "unavailable_before_frozen_candidate_population",
            "unavailable_after_reason": "unavailable_after_materialized_v3_coverage",
            "normal_state_is_never_imputed_for_unavailable_intervals": True,
        },
        "state_profiles": {
            "method": "per-state observable-feature medians contrasted with all-row medians, standardized by all-row MAD * 1.4826",
            "feature_columns": profile_inputs,
            "forbidden_prefixes": list(PROFILE_EXCLUDED_PREFIXES),
            "outcomes_or_targets_used": False,
        },
        "outputs_sha256": {name: sha256(output_dir / name) for name in outputs},
        "checksum_convention": "all material outputs are listed above; manifest.json is verified by detached manifest.sha256",
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / MANIFEST_SIDECAR_NAME).write_text(
        f"{sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--extension-dir", type=Path, default=DEFAULT_EXTENSION_DIR)
    result.add_argument("--v3-dir", type=Path, default=DEFAULT_V3_DIR)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--coverage-start", default="2022-01-01T00:00:00Z")
    result.add_argument("--expected-event-count", type=int, default=157)
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(_json_safe(materialize_regime_episode_ledger(**vars(args))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
