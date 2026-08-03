#!/usr/bin/env python3
"""Map canonical regime-transition events to reproducible research evidence.

This is an inventory and reconstruction-planning artifact.  It does *not*
impute a score, an execution path, a health value, or an economic failure.  In
particular, it keeps three different notions of availability separate:

* an archived, candidate-identity-exact research lineage;
* the current execution-EV score / current-model-health lineage; and
* a legacy score context that lacks candidate IDs and is therefore not replay
  joinable.

The distinction prevents a broad historical score range from being
misrepresented as a current-model, exact-policy overlay cohort.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_coverage_readiness_20260727_v2"
)
DEFAULT_EVENTS = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "transition_events.parquet"
)
DEFAULT_ACTIVE = Path(
    "data_perp/artifacts/regime_transition_active_head_20260726_v1/"
    "grouped_oof.parquet"
)
DEFAULT_HISTORICAL_SCORE_SOURCES = (
    Path(
        "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/"
        "two_layer_direct_ev_strict_oof.parquet"
    ),
    Path(
        "data_perp/artifacts/feb2025_jul2026_execution_ev_common30_transfer_oof_20260727_v4/"
        "two_layer_direct_ev_strict_oof.parquet"
    ),
)
DEFAULT_CURRENT_SCORES = Path(
    "data_perp/artifacts/execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_RAW_1M_SOURCES = (
    Path(
        "data_perp/artifacts/late2024_execution_ev_hourly_comparator_20260727_v2/"
        "hourly_execution_ev_12h_labels.parquet"
    ),
    Path(
        "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/"
        "exact_1m_execution_ev_12h_labels.parquet"
    ),
    Path(
        "data_perp/artifacts/feb2025_jul2026_execution_ev_common30_transfer_oof_20260727_v4/"
        "exact_1m_execution_ev_12h_labels.parquet"
    ),
    Path(
        "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
        "execution_ev_policy_labels.parquet"
    ),
)
DEFAULT_PRICE_REPLAY_SOURCES = (
    Path(
        "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/"
        "labels.parquet"
    ),
    Path(
        "data_perp/artifacts/mayjul2025_execution_ev_common30_labels_20260727_v2/"
        "labels.parquet"
    ),
    Path(
        "data_perp/artifacts/augoct2025_execution_ev_common30_labels_20260727_v1/"
        "labels.parquet"
    ),
    Path(
        "data_perp/artifacts/nov2025_execution_ev_common30_labels_20260727_v1/"
        "labels.parquet"
    ),
    Path(
        "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
        "execution_ev_policy_labels.parquet"
    ),
)
DEFAULT_POLICY_INPUT_BATCHES = (
    (
        "febapr2025_deployed_policy_inputs",
        Path("data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1/candidates.parquet"),
        Path("data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1/path_targets.parquet"),
        Path("data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1/context.parquet"),
    ),
    (
        "mayjul2025_common30_policy_inputs",
        Path("data_perp/artifacts/mayjul2025_execution_ev_common30_policy_inputs_20260727_v2/candidates.parquet"),
        Path("data_perp/artifacts/mayjul2025_execution_ev_common30_policy_inputs_20260727_v2/path_targets.parquet"),
        Path("data_perp/artifacts/mayjul2025_execution_ev_common30_policy_inputs_20260727_v2/context.parquet"),
    ),
    (
        "augoct2025_common30_policy_inputs",
        Path("data_perp/artifacts/augoct2025_execution_ev_common30_policy_inputs_20260727_v1/candidates.parquet"),
        Path("data_perp/artifacts/augoct2025_execution_ev_common30_policy_inputs_20260727_v1/path_targets.parquet"),
        Path("data_perp/artifacts/augoct2025_execution_ev_common30_policy_inputs_20260727_v1/context.parquet"),
    ),
    (
        "nov2025_jan2026_common30_policy_inputs",
        Path("data_perp/artifacts/nov2025_jan2026_execution_ev_common30_policy_inputs_20260727_v1/candidates.parquet"),
        Path("data_perp/artifacts/nov2025_jan2026_execution_ev_common30_policy_inputs_20260727_v1/path_targets.parquet"),
        Path("data_perp/artifacts/nov2025_jan2026_execution_ev_common30_policy_inputs_20260727_v1/context.parquet"),
    ),
)
DEFAULT_HISTORICAL_HEALTH = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_historical_20260726_v12/"
    "hourly_observable_state.parquet"
)
DEFAULT_CURRENT_HEALTH = Path(
    "data_perp/artifacts/regime_transition_current_model_health_20260727_v1/"
    "hourly_model_health.parquet"
)
DEFAULT_FAILURE_EPISODES = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_historical_20260726_v12/"
    "failure_episodes.parquet"
)
DEFAULT_LEGACY_SCORE_CONTEXT = Path(
    "data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/"
    "meta_postprocessor_compact_ledger.parquet"
)


@dataclass(frozen=True)
class HourCoverage:
    """Unique UTC hourly availability for one contract.

    ``decision_shift_hours`` maps canonical source-time event intervals to the
    time axis held by the source.  Candidate and path artifacts use source
    time; health artifacts conventionally use decision time (+1 hour).
    """

    name: str
    counts: pd.Series
    decision_shift_hours: int
    identity_complete: bool
    lineage: str


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_timestamp(values: pd.Series, *, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce").dt.floor("h")
    if result.isna().any():
        raise ValueError(f"{name} contains non-UTC/invalid timestamps")
    return result


def _read_times(path: Path, column: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=[column])
    if column not in frame:
        raise ValueError(f"{path} lacks {column}")
    return _normalise_timestamp(frame[column], name=f"{path}:{column}")


def _validate_candidate_identity(path: Path, *, timestamp_column: str = "__ts__") -> None:
    """Fail closed when a purported exact candidate source is not joinable."""

    identity = [timestamp_column, "__symbol__", "side_name", "candidate_id"]
    frame = pd.read_parquet(path, columns=identity)
    missing = [column for column in identity if column not in frame]
    if missing:
        raise ValueError(f"{path} lacks candidate identity columns: {missing}")
    frame[timestamp_column] = _normalise_timestamp(
        frame[timestamp_column], name=f"{path}:{timestamp_column}"
    )
    if frame[["__symbol__", "side_name", "candidate_id"]].isna().any().any():
        raise ValueError(f"{path} has null candidate identities")
    if frame["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path} has blank candidate IDs")
    if frame.duplicated(identity).any():
        raise ValueError(f"{path} has duplicate candidate identities")


def _counts_from_times(times: Iterable[pd.Timestamp]) -> pd.Series:
    series = pd.Series(list(times), dtype="datetime64[ns, UTC]")
    if series.empty:
        return pd.Series(dtype="int64")
    return series.value_counts(sort=False).sort_index().astype("int64")


def _union_counts(collection: Iterable[pd.Series]) -> pd.Series:
    values = [item for item in collection if not item.empty]
    if not values:
        return pd.Series(dtype="int64")
    combined = pd.concat(values, axis=1).fillna(0)
    # Several reconstruction artifacts can overlap.  Availability is not a
    # doubled candidate population, so retain the largest witness count.
    return combined.max(axis=1).astype("int64").sort_index()


def _coverage_from_paths(
    name: str,
    paths: Sequence[Path],
    *,
    column: str,
    decision_shift_hours: int,
    identity_complete: bool,
    lineage: str,
) -> HourCoverage:
    return HourCoverage(
        name=name,
        counts=_union_counts(_counts_from_times(_read_times(path, column)) for path in paths),
        decision_shift_hours=decision_shift_hours,
        identity_complete=identity_complete,
        lineage=lineage,
    )


def _policy_input_coverage() -> tuple[HourCoverage, dict[str, list[Path]]]:
    batch_counts: list[pd.Series] = []
    paths_by_batch: dict[str, list[Path]] = {}
    for name, candidates, path_targets, context in DEFAULT_POLICY_INPUT_BATCHES:
        source_counts = [
            _counts_from_times(_read_times(path, "__ts__"))
            for path in (candidates, path_targets, context)
        ]
        # A policy-input hour is usable only when candidate identity, geometry
        # input and policy archetype context all exist.
        shared = pd.concat(source_counts, axis=1).fillna(0).min(axis=1)
        batch_counts.append(shared.loc[shared.gt(0)].astype("int64"))
        paths_by_batch[name] = [candidates, path_targets, context]
    return (
        HourCoverage(
            name="deployed_policy_geometry",
            counts=_union_counts(batch_counts),
            decision_shift_hours=0,
            identity_complete=True,
            lineage="archived_deployed_policy_input_contract",
        ),
        paths_by_batch,
    )


def event_source_hours(event: Mapping[str, Any], *, decision_shift_hours: int = 0) -> pd.DatetimeIndex:
    """Return the inclusive active interval on a source's native hour axis."""

    start = pd.Timestamp(event["transition_start_utc"])
    end = pd.Timestamp(event["transition_end_utc"])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("canonical event timestamps must be timezone-aware UTC")
    start = start.tz_convert("UTC").floor("h") + pd.Timedelta(hours=decision_shift_hours)
    end = end.tz_convert("UTC").floor("h") + pd.Timedelta(hours=decision_shift_hours)
    if end < start:
        raise ValueError("transition end precedes start")
    return pd.date_range(start, end, freq="h", tz="UTC")


def coverage_metrics(event: Mapping[str, Any], source: HourCoverage) -> dict[str, Any]:
    requested = event_source_hours(event, decision_shift_hours=source.decision_shift_hours)
    observed = source.counts.reindex(requested, fill_value=0).astype("int64")
    covered = observed.gt(0)
    return {
        "hours": int(len(requested)),
        "covered_hours": int(covered.sum()),
        "coverage_fraction": float(covered.mean()) if len(covered) else 0.0,
        "min_candidate_rows": int(observed.min()) if len(observed) else 0,
        "full_coverage": bool(covered.all()),
    }


def _nearest_failure(
    anchor: pd.Timestamp,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    if episodes.empty:
        return {
            "nearest_historical_failure_episode_id": None,
            "nearest_historical_failure_distance_hours": np.nan,
            "historical_failure_episode_within_6h": False,
            "historical_failure_episode_within_12h": False,
            "historical_failure_episode_within_24h": False,
        }
    distances = (
        episodes["episode_onset_decision_utc"] - pd.Timestamp(anchor)
    ).abs().dt.total_seconds().div(3600.0)
    position = int(distances.to_numpy().argmin())
    distance = float(distances.iloc[position])
    return {
        "nearest_historical_failure_episode_id": str(episodes.iloc[position]["episode_id"]),
        "nearest_historical_failure_distance_hours": distance,
        "historical_failure_episode_within_6h": bool(distance <= 6.0),
        "historical_failure_episode_within_12h": bool(distance <= 12.0),
        "historical_failure_episode_within_24h": bool(distance <= 24.0),
    }


def assign_reconstruction_reasons(row: Mapping[str, Any]) -> list[str]:
    """Return explicit, non-overlapping evidence gaps in priority order."""

    reasons: list[str] = []
    if not bool(row["active_probability_grouped_oof_full"]):
        reasons.append("MISSING_ACTIVE_PROBABILITY_OOF")
    if not bool(row["candidate_score_identity_exact_full"]):
        reasons.append("MISSING_CANDIDATE_SCORE_LINEAGE")
    if not bool(row["raw_1m_execution_path_full"]):
        reasons.append("MISSING_RAW_1M_EXECUTION_PATH")
    if not bool(row["replay_price_path_full"]):
        reasons.append("MISSING_REPLAY_PRICE_PATH")
    if not bool(row["deployed_policy_geometry_full"]):
        reasons.append("MISSING_DEPLOYED_POLICY_GEOMETRY")
    if not bool(row["health_any_lineage_full"]):
        reasons.append("MISSING_HEALTH_INPUTS")
    if not bool(row["current_score_lineage_full"]):
        reasons.append("CURRENT_SCORE_LINEAGE_NOT_MATERIALIZED")
    if not bool(row["current_model_health_full"]):
        reasons.append("CURRENT_HEALTH_INPUTS_NOT_MATERIALIZED")
    if not bool(row["historical_failure_episode_within_6h"]):
        reasons.append("NO_NATIVE_ECONOMIC_FAILURE_LINK_WITHIN_6H")
    if not reasons:
        reasons.append("ARCHIVAL_COMMON_COHORT_READY")
    return reasons


def _priority(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    """Rank smallest-gap candidates first, then economically relevant ones."""

    reasons = list(row["reconstruction_reason_codes"])
    missing_hard = sum(
        reason
        in {
            "MISSING_ACTIVE_PROBABILITY_OOF",
            "MISSING_CANDIDATE_SCORE_LINEAGE",
            "MISSING_RAW_1M_EXECUTION_PATH",
            "MISSING_REPLAY_PRICE_PATH",
            "MISSING_DEPLOYED_POLICY_GEOMETRY",
            "MISSING_HEALTH_INPUTS",
        }
        for reason in reasons
    )
    current_gap = sum(
        reason
        in {
            "CURRENT_SCORE_LINEAGE_NOT_MATERIALIZED",
            "CURRENT_HEALTH_INPUTS_NOT_MATERIALIZED",
        }
        for reason in reasons
    )
    # Existing economic failures are highest-value validation anchors.  For
    # expansion candidates, large state changes are the next best use of work.
    economic = int(bool(row["canonical_economic_failure_within_6h"]) or bool(row["historical_failure_episode_within_6h"]))
    severity = abs(float(row["robust_pre_post_shift"]))
    return (missing_hard, current_gap - economic, -severity, str(row["event_id"]))


def _event_fingerprint(row: Mapping[str, Any], source_hashes: Mapping[str, str]) -> str:
    payload = {
        "event_id": str(row["event_id"]),
        "start": str(row["transition_start_utc"]),
        "end": str(row["transition_end_utc"]),
        "sources": dict(sorted(source_hashes.items())),
        "reason_codes": list(row["reconstruction_reason_codes"]),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_coverage(
    events: pd.DataFrame,
    sources: Mapping[str, HourCoverage],
    failures: pd.DataFrame,
    source_hashes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    required_event_columns = {
        "event_id",
        "anchor_source_utc",
        "anchor_decision_utc",
        "transition_start_utc",
        "transition_end_utc",
        "robust_pre_post_shift",
        "economic_failure_event_within_6h",
    }
    missing = sorted(required_event_columns.difference(events.columns))
    if missing:
        raise ValueError(f"canonical events missing columns: {missing}")
    work = events.copy()
    for column in (
        "anchor_source_utc",
        "anchor_decision_utc",
        "transition_start_utc",
        "transition_end_utc",
    ):
        work[column] = _normalise_timestamp(work[column], name=f"events:{column}")
    if work["event_id"].duplicated().any():
        raise ValueError("canonical event IDs must be unique")
    failure_work = failures.copy()
    if not failure_work.empty:
        required_failures = {"episode_id", "episode_onset_decision_utc"}
        missing_failures = sorted(required_failures.difference(failure_work.columns))
        if missing_failures:
            raise ValueError(f"failure episodes missing columns: {missing_failures}")
        failure_work["episode_onset_decision_utc"] = _normalise_timestamp(
            failure_work["episode_onset_decision_utc"], name="failure episodes"
        )
    records: list[dict[str, Any]] = []
    for event in work.to_dict("records"):
        record = dict(event)
        record["event_month"] = pd.Timestamp(event["anchor_source_utc"]).strftime("%Y-%m")
        for name, source in sources.items():
            metrics = coverage_metrics(event, source)
            record[f"{name}_hours"] = metrics["hours"]
            record[f"{name}_covered_hours"] = metrics["covered_hours"]
            record[f"{name}_coverage_fraction"] = metrics["coverage_fraction"]
            record[f"{name}_min_candidate_rows"] = metrics["min_candidate_rows"]
            record[f"{name}_full"] = metrics["full_coverage"]
        record["candidate_score_identity_exact_full"] = bool(
            record["historical_score_oof_full"] or record["current_score_oof_full"]
        )
        record["current_score_lineage_full"] = bool(record["current_score_oof_full"])
        record["health_any_lineage_full"] = bool(
            record["historical_health_context_full"] or record["current_model_health_full"]
        )
        record["archival_common_valid_full"] = bool(
            record["historical_score_oof_full"]
            and record["raw_1m_execution_path_full"]
            and record["replay_price_path_full"]
            and record["deployed_policy_geometry_full"]
            and record["active_probability_grouped_oof_full"]
            and record["historical_health_context_full"]
        )
        record["current_lineage_common_valid_full"] = bool(
            record["current_score_oof_full"]
            and record["raw_1m_execution_path_full"]
            and record["replay_price_path_full"]
            and record["deployed_policy_geometry_full"]
            and record["active_probability_grouped_oof_full"]
            and record["current_model_health_full"]
        )
        record["canonical_economic_failure_within_6h"] = bool(
            pd.notna(event["economic_failure_event_within_6h"])
        )
        record.update(_nearest_failure(event["anchor_decision_utc"], failure_work))
        record["reconstruction_reason_codes"] = assign_reconstruction_reasons(record)
        record["reconstruction_reason_code"] = "|".join(record["reconstruction_reason_codes"])
        records.append(record)
    result = pd.DataFrame.from_records(records)
    hashes = source_hashes or {}
    result["coverage_fingerprint_sha256"] = [
        _event_fingerprint(row, hashes) for row in result.to_dict("records")
    ]
    ranked = sorted(
        result.to_dict("records"),
        key=_priority,
    )
    priority_by_event = {str(row["event_id"]): rank + 1 for rank, row in enumerate(ranked)}
    result["reconstruction_priority"] = result["event_id"].astype(str).map(priority_by_event).astype(int)
    return result.sort_values(["anchor_source_utc", "event_id"], kind="stable").reset_index(drop=True)


def summarize_monthly(coverage: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "active_probability_grouped_oof_full",
        "historical_score_oof_full",
        "current_score_oof_full",
        "raw_1m_execution_path_full",
        "replay_price_path_full",
        "deployed_policy_geometry_full",
        "historical_health_context_full",
        "current_model_health_full",
        "archival_common_valid_full",
        "current_lineage_common_valid_full",
        "canonical_economic_failure_within_6h",
        "historical_failure_episode_within_6h",
    ]
    aggregate = coverage.groupby("event_month", sort=True, observed=True).agg(
        events=("event_id", "size"),
        max_abs_state_shift=("robust_pre_post_shift", lambda values: float(np.abs(values).max())),
        **{column: (column, "sum") for column in flags},
    )
    return aggregate.reset_index()


def _source_hashes(paths: Mapping[str, Sequence[Path]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for group, group_paths in paths.items():
        for path in group_paths:
            key = f"{group}:{path.name}"
            hashes[key] = _sha256(path)
    return hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    events = pd.read_parquet(args.events)
    for path in (*DEFAULT_HISTORICAL_SCORE_SOURCES, *DEFAULT_RAW_1M_SOURCES, *DEFAULT_PRICE_REPLAY_SOURCES):
        _validate_candidate_identity(path)
    for _, candidates, path_targets, context in DEFAULT_POLICY_INPUT_BATCHES:
        for path in (candidates, path_targets, context):
            _validate_candidate_identity(path)
    active = _coverage_from_paths(
        "active_probability_grouped_oof",
        [args.active],
        column="source_utc",
        decision_shift_hours=0,
        identity_complete=False,
        lineage="grouped_oof_non_chronological_research_only",
    )
    historical_score = _coverage_from_paths(
        "historical_score_oof",
        DEFAULT_HISTORICAL_SCORE_SOURCES,
        column="__ts__",
        decision_shift_hours=0,
        identity_complete=True,
        lineage="strict_two_layer_oof_candidate_id_exact",
    )
    current_scores = pd.read_parquet(
        DEFAULT_CURRENT_SCORES,
        columns=[
            "__ts__",
            "__symbol__",
            "side_name",
            "candidate_id",
            "causal_recent_isotonic_ev",
            "causal_recent_isotonic_ev__is_oof",
        ],
    )
    if current_scores.duplicated(["__ts__", "__symbol__", "side_name", "candidate_id"]).any():
        raise ValueError("current causal score ledger has duplicate candidate identities")
    score_mask = (
        current_scores["causal_recent_isotonic_ev__is_oof"].fillna(False).astype(bool)
        & pd.to_numeric(current_scores["causal_recent_isotonic_ev"], errors="coerce").notna()
    )
    current_score = HourCoverage(
        name="current_score_oof",
        counts=_counts_from_times(_normalise_timestamp(current_scores.loc[score_mask, "__ts__"], name="current scores")),
        decision_shift_hours=0,
        identity_complete=True,
        lineage="current_causal_recent_ev_candidate_id_exact_oof",
    )
    raw_paths = _coverage_from_paths(
        "raw_1m_execution_path",
        DEFAULT_RAW_1M_SOURCES,
        column="__ts__",
        decision_shift_hours=0,
        identity_complete=True,
        lineage="exact_1m_execution_label",
    )
    replay_prices = _coverage_from_paths(
        "replay_price_path",
        DEFAULT_PRICE_REPLAY_SOURCES,
        column="__ts__",
        decision_shift_hours=0,
        identity_complete=True,
        lineage="exact_1m_entry_exit_price_and_geometry",
    )
    policy_geometry, policy_batches = _policy_input_coverage()
    historical_health = _coverage_from_paths(
        "historical_health_context",
        [DEFAULT_HISTORICAL_HEALTH],
        column="execution_decision_utc",
        decision_shift_hours=1,
        identity_complete=False,
        lineage="archived_failure_first_observable_health",
    )
    current_health = _coverage_from_paths(
        "current_model_health",
        [DEFAULT_CURRENT_HEALTH],
        column="source_utc",
        decision_shift_hours=0,
        identity_complete=False,
        lineage="current_execution_ev_hourly_health",
    )
    legacy_score = _coverage_from_paths(
        "legacy_score_context",
        [DEFAULT_LEGACY_SCORE_CONTEXT],
        column="__ts__",
        decision_shift_hours=0,
        identity_complete=False,
        lineage="old55_context_without_candidate_id_not_replay_joinable",
    )
    failures = pd.read_parquet(DEFAULT_FAILURE_EPISODES)
    source_paths: dict[str, Sequence[Path]] = {
        "events": [args.events],
        "active": [args.active],
        "historical_score": list(DEFAULT_HISTORICAL_SCORE_SOURCES),
        "current_scores": [DEFAULT_CURRENT_SCORES],
        "raw_1m": list(DEFAULT_RAW_1M_SOURCES),
        "price_replay": list(DEFAULT_PRICE_REPLAY_SOURCES),
        "historical_health": [DEFAULT_HISTORICAL_HEALTH],
        "current_health": [DEFAULT_CURRENT_HEALTH],
        "failure_episodes": [DEFAULT_FAILURE_EPISODES],
        "legacy_score": [DEFAULT_LEGACY_SCORE_CONTEXT],
        **{f"policy_inputs:{name}": paths for name, paths in policy_batches.items()},
    }
    source_hashes = _source_hashes(source_paths)
    coverage = build_coverage(
        events,
        {
            item.name: item
            for item in (
                active,
                historical_score,
                current_score,
                raw_paths,
                replay_prices,
                policy_geometry,
                historical_health,
                current_health,
                legacy_score,
            )
        },
        failures,
        source_hashes,
    )
    monthly = summarize_monthly(coverage)
    queue = coverage.sort_values(
        ["reconstruction_priority", "anchor_source_utc", "event_id"], kind="stable"
    ).copy()
    # The queue deliberately retains ready rows: this makes the smallest
    # reproducible cohort auditable rather than silently filtering evidence.
    queue["queue_action"] = np.select(
        [
            queue["current_lineage_common_valid_full"],
            queue["archival_common_valid_full"],
        ],
        [
            "VALIDATE_FULL_CURRENT_LINEAGE_COHORT",
            "BACKFILL_CURRENT_SCORE_AND_HEALTH_ON_ARCHIVAL_EXACT_COHORT",
        ],
        default="RECONSTRUCT_MISSING_CONTRACTS",
    )
    args.output_dir.mkdir(parents=True)
    coverage.to_parquet(args.output_dir / "event_coverage.parquet", index=False)
    monthly.to_parquet(args.output_dir / "monthly_coverage.parquet", index=False)
    queue.to_parquet(args.output_dir / "prioritized_reconstruction_queue.parquet", index=False)
    ready_archival = coverage.loc[coverage["archival_common_valid_full"]]
    minimal_five = ready_archival.head(5)
    summary = {
        "schema": "regime_transition_coverage_readiness_v1",
        "contracts": {
            "event_axis": "canonical active interval [transition_start_utc, transition_end_utc], inclusive, UTC hourly",
            "strict_archival_common": "candidate-ID exact strict-OOF score + exact 1m path + replay price path + archived deployed geometry + grouped-OOF active probability + archived health",
            "strict_current_common": "current causal OOF score + exact 1m path + replay price path + deployed geometry + grouped-OOF active probability + current-model health",
            "active_probability": "grouped OOF only; non-chronological research evidence, never upgraded to policy OOS",
            "legacy_score_context": "reported separately because it lacks candidate IDs and cannot join a replay cohort",
            "economic_failures": "native archived failure episodes are separate from the canonical transition ±6h linkage",
        },
        "counts": {
            "canonical_transition_events": int(len(coverage)),
            "archival_common_valid_events": int(coverage["archival_common_valid_full"].sum()),
            "current_lineage_common_valid_events": int(coverage["current_lineage_common_valid_full"].sum()),
            "canonical_economic_failures_within_6h": int(coverage["canonical_economic_failure_within_6h"].sum()),
            "historical_failure_episodes": int(len(failures)),
            "historical_failure_episode_links_within_6h": int(coverage["historical_failure_episode_within_6h"].sum()),
            "historical_failure_episode_links_within_12h": int(coverage["historical_failure_episode_within_12h"].sum()),
            "historical_failure_episode_links_within_24h": int(coverage["historical_failure_episode_within_24h"].sum()),
        },
        "minimum_archival_common_five_event_cohort": minimal_five.loc[
            :, ["event_id", "anchor_source_utc", "event_month", "reconstruction_reason_code"]
        ].to_dict("records"),
        "economic_failure_target": {
            "requested_range": [60, 100],
            "existing_native_failure_episodes": int(len(failures)),
            "shortfall_to_60": max(0, 60 - int(len(failures))),
            "shortfall_to_100": max(0, 100 - int(len(failures))),
            "claim": "No future failure count is inferred. The queue identifies periods with score/path/geometry coverage where native failure labels still require materialization.",
        },
        "sources": {
            key: [{"path": str(path), "sha256": source_hashes[f"{key}:{path.name}"]} for path in paths]
            for key, paths in source_paths.items()
        },
        "outputs": {},
    }
    for filename in (
        "event_coverage.parquet",
        "monthly_coverage.parquet",
        "prioritized_reconstruction_queue.parquet",
    ):
        output_path = args.output_dir / filename
        summary["outputs"][filename] = {"path": str(output_path), "sha256": _sha256(output_path)}
    _write_json(args.output_dir / "manifest.json", summary)
    return summary


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary["counts"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
