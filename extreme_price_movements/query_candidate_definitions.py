"""Inference-valid training query grammar for residual LambdaRank."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class QueryDefinition:
    name: str
    kind: str
    cycle_hours: int | None = None
    bucket_minutes: int | None = None
    side_local: bool = True

    def manifest(self) -> dict[str, object]:
        return asdict(self)


def recommended_query_definitions() -> tuple[QueryDefinition, ...]:
    """Small predeclared grammar; do not expand post-OOS."""
    return (
        QueryDefinition("q0_exact_timestamp_side", "exact"),
        QueryDefinition("q1_cycle_1h_side", "cycle", cycle_hours=1),
        QueryDefinition("q1_cycle_2h_side", "cycle", cycle_hours=2),
        QueryDefinition("q1_cycle_3h_side", "cycle", cycle_hours=3),
        QueryDefinition("q1_cycle_4h_side", "cycle", cycle_hours=4),
        QueryDefinition("q1_cycle_6h_side", "cycle", cycle_hours=6),
        QueryDefinition("q1_cycle_8h_side", "cycle", cycle_hours=8),
        QueryDefinition("q1_cycle_12h_side", "cycle", cycle_hours=12),
        QueryDefinition("q1_cycle_24h_side", "cycle", cycle_hours=24),
        QueryDefinition("q2_bucket_15m_side", "bucket", bucket_minutes=15),
    )


def query_definitions_by_name(names: Iterable[str]) -> tuple[QueryDefinition, ...]:
    """Resolve a frozen shortlist without inventing a new query grammar."""
    catalogue = {definition.name: definition for definition in recommended_query_definitions()}
    requested = tuple(dict.fromkeys(str(name) for name in names))
    missing = sorted(set(requested).difference(catalogue))
    if missing:
        raise KeyError(f"unknown predeclared query definitions: {missing}")
    return tuple(catalogue[name] for name in requested)


def base_head_query_definitions() -> tuple[QueryDefinition, ...]:
    """Return the bounded, predeclared grammar for complementary base heads.

    The residual/specialist grammar intentionally remains small.  Base heads
    are allowed a modestly wider cycle sweep because a base target may express
    opportunity at a different competition horizon.  These definitions remain
    entirely decision-time derived and side-local; callers must still screen
    query support on their training population before fitting.
    """
    return (
        QueryDefinition("q1_cycle_1h_side", "cycle", cycle_hours=1),
        QueryDefinition("q1_cycle_2h_side", "cycle", cycle_hours=2),
        QueryDefinition("q1_cycle_4h_side", "cycle", cycle_hours=4),
        QueryDefinition("q1_cycle_6h_side", "cycle", cycle_hours=6),
        QueryDefinition("q1_cycle_8h_side", "cycle", cycle_hours=8),
        QueryDefinition("q1_cycle_12h_side", "cycle", cycle_hours=12),
    )


def assign_query_ids(frame: pd.DataFrame, definition: QueryDefinition,
                     *, timestamp_column: str = "__ts__", side_column: str = "side_name") -> pd.Series:
    """Assign side-local query ids from decision-time information only."""
    if timestamp_column not in frame or side_column not in frame:
        raise KeyError("query construction needs timestamp and side columns")
    ts = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
    if definition.kind == "exact":
        # An exact-time query must never silently coarsen to an hourly query.
        # Coarser competition is represented explicitly by q1_cycle_1h_side.
        bucket = ts
    elif definition.kind == "cycle":
        if not definition.cycle_hours:
            raise ValueError("cycle definition requires cycle_hours")
        bucket = ts.dt.floor(f"{definition.cycle_hours}h")
    elif definition.kind == "bucket":
        if not definition.bucket_minutes:
            raise ValueError("bucket definition requires bucket_minutes")
        bucket = ts.dt.floor(f"{definition.bucket_minutes}min")
    else:
        raise ValueError(f"unsupported query kind: {definition.kind}")
    side = frame[side_column].astype(str) if definition.side_local else pd.Series("all", index=frame.index)
    return (bucket.astype("int64").astype(str) + "|" + side).astype("string")


def materialize_query_membership(frame: pd.DataFrame,
                                 definitions: Iterable[QueryDefinition] | None = None) -> pd.DataFrame:
    """Return stable one-membership-per-row records for every grammar arm."""
    if "candidate_id" not in frame:
        raise KeyError("candidate_id is required for stable query membership")
    if frame.candidate_id.duplicated().any():
        raise ValueError("query membership source contains duplicate candidate IDs")
    records: list[pd.DataFrame] = []
    for definition in definitions or recommended_query_definitions():
        out = frame[["candidate_id"]].copy()
        out["query_candidate"] = definition.name
        out["query_id"] = assign_query_ids(frame, definition)
        records.append(out)
    return pd.concat(records, ignore_index=True)
