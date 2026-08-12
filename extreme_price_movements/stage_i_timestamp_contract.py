"""Canonical timestamp lineage for Stage-I selector, MDA, HPO, and mapping.

``__ts__`` is the immutable signal-bar close identity.  The executable
decision is one hour later and the H12 label resolves twelve hours after that
decision (thirteen hours after signal close).  Research splits must use the
decision timestamp while joins continue to use the signal-close identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import pandas as pd


SIGNAL_CLOSE_COLUMN = "__ts__"
DECISION_COLUMN = "decision_ts"
LABEL_AVAILABLE_COLUMN = "label_available_ts"
SIGNAL_TO_DECISION = pd.Timedelta(hours=1)
DECISION_TO_LABEL_AVAILABLE = pd.Timedelta(hours=12)


@dataclass(frozen=True)
class StageITimestampContract:
    signal_close: pd.Series
    decision: pd.Series
    label_available: pd.Series
    audit: Mapping[str, Any]


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Stage-I timestamp contract lacks {column}")
    value = pd.to_datetime(frame[column], utc=True, errors="coerce").reset_index(drop=True)
    if value.isna().any():
        raise ValueError(f"Stage-I timestamp contract has invalid {column}")
    return value


def _timestamp_digest(*series: pd.Series) -> str:
    payload = b"".join(
        value.astype("int64").to_numpy().tobytes()
        for value in series
    )
    return sha256(payload).hexdigest()


def resolve_stage_i_timestamp_contract(frame: pd.DataFrame) -> StageITimestampContract:
    """Resolve and validate the exact Stage-I production timing contract.

    Existing explicit ``decision_ts``/``__decision_ts__`` fields are treated
    as assertions, never trusted as alternate conventions.  The input frame
    is not mutated, so ``__ts__`` remains the candidate identity.
    """
    signal = _utc(frame, SIGNAL_CLOSE_COLUMN)
    available = _utc(frame, LABEL_AVAILABLE_COLUMN)
    expected_decision = signal + SIGNAL_TO_DECISION
    explicit: list[pd.Series] = []
    for column in (DECISION_COLUMN, "__decision_ts__"):
        if column in frame:
            explicit.append(_utc(frame, column))
    if any(not value.equals(expected_decision) for value in explicit):
        raise ValueError("Stage-I decision_ts must equal immutable __ts__ + 1h")
    decision = expected_decision
    expected_available = decision + DECISION_TO_LABEL_AVAILABLE
    if not available.equals(expected_available):
        delta = (available - decision).dt.total_seconds() / 3600.0
        examples = sorted(set(delta.round(6).astype(float).tolist()))[:5]
        raise ValueError(
            "Stage-I label_available_ts must equal decision_ts + 12h; "
            f"observed_hours={examples}"
        )
    audit: Mapping[str, Any] = {
        "schema": "stage_i_signal_decision_label_timing_v1",
        "signal_identity_column": SIGNAL_CLOSE_COLUMN,
        "decision_column": DECISION_COLUMN,
        "label_available_column": LABEL_AVAILABLE_COLUMN,
        "signal_to_decision_hours": 1,
        "decision_to_label_available_hours": 12,
        "signal_to_label_available_hours": 13,
        "rows": int(len(frame)),
        "min_signal_close_ts": signal.min().isoformat() if len(signal) else None,
        "max_signal_close_ts": signal.max().isoformat() if len(signal) else None,
        "min_decision_ts": decision.min().isoformat() if len(decision) else None,
        "max_decision_ts": decision.max().isoformat() if len(decision) else None,
        "timestamp_lineage_sha256": _timestamp_digest(signal, decision, available),
        "signal_timestamp_preserved_as_identity": True,
        "selector_mda_hpo_map_timestamp_semantics": "decision_ts",
    }
    return StageITimestampContract(signal, decision, available, audit)


def attach_stage_i_decision_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with explicit ``decision_ts`` while preserving ``__ts__``."""
    timing = resolve_stage_i_timestamp_contract(frame)
    output = frame.copy()
    output[DECISION_COLUMN] = timing.decision.to_numpy()
    return output


__all__ = [
    "DECISION_COLUMN",
    "DECISION_TO_LABEL_AVAILABLE",
    "LABEL_AVAILABLE_COLUMN",
    "SIGNAL_CLOSE_COLUMN",
    "SIGNAL_TO_DECISION",
    "StageITimestampContract",
    "attach_stage_i_decision_timestamp",
    "resolve_stage_i_timestamp_contract",
]
