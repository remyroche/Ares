"""Canonical timestamp handling for the Ares production pipeline.

UTC is the only storage, join, feature, label, replay, inference, and artifact
timezone. Naive legacy values are interpreted as UTC, never as host-local time.
Europe/Paris is permitted only after this normalization in display surfaces.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

UTC = "UTC"
ErrorsMode = Literal["raise", "coerce", "ignore"]


def to_utc_timestamp(value: Any, *, errors: ErrorsMode = "raise") -> pd.Timestamp:
    """Return one timezone-aware UTC timestamp, treating naive input as UTC."""
    converted = pd.to_datetime(value, utc=True, errors=errors)
    if isinstance(converted, (pd.Series, pd.DatetimeIndex)):
        raise TypeError("to_utc_timestamp expects one scalar timestamp value")
    return pd.Timestamp(converted)


def to_utc_index(values: Any, *, errors: ErrorsMode = "raise") -> pd.DatetimeIndex:
    """Return a timezone-aware UTC index for joins and persisted time axes."""
    converted = pd.to_datetime(values, utc=True, errors=errors)
    if isinstance(converted, pd.Series):
        converted = converted.array
    return pd.DatetimeIndex(converted)


def to_utc_series(values: Any, *, errors: ErrorsMode = "raise") -> pd.Series:
    """Return a timezone-aware UTC Series while retaining the input index."""
    converted = pd.to_datetime(values, utc=True, errors=errors)
    if isinstance(converted, pd.Series):
        return converted
    return pd.Series(converted)


def utc_now() -> pd.Timestamp:
    """Return the current timezone-aware UTC instant."""
    return pd.Timestamp.now(tz=UTC)


def utc_isoformat(value: Any = None) -> str:
    """Serialize an instant as an offset-explicit UTC ISO-8601 timestamp."""
    return to_utc_timestamp(utc_now() if value is None else value).isoformat()


def format_paris_display(value: Any, *, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format a canonical instant for Europe/Paris display only."""
    return to_utc_timestamp(value).tz_convert("Europe/Paris").strftime(fmt)


def timeframe_delta(timeframe: Any) -> pd.Timedelta:
    """Return the fixed bar duration used by the causal signal contract."""
    value = str(timeframe or "").strip().lower()
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    try:
        delta = pd.Timedelta(aliases.get(value, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported fixed timeframe {timeframe!r}") from exc
    if delta <= pd.Timedelta(0):
        raise ValueError(f"Timeframe must be positive, got {timeframe!r}")
    return delta


def causal_decision_timestamps(signal_ts: Any, *, timeframe: Any) -> pd.DatetimeIndex:
    """Return the first instant at which a completed signal bar is observable."""
    return to_utc_index(signal_ts, errors="coerce") + timeframe_delta(timeframe)


def causal_signal_times(
    frame: pd.DataFrame,
    *,
    timeframe: Any,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Resolve signal-open and mandatory signal-close timestamps for replay.

    ``signal_bar_ts`` is authoritative when present. Otherwise the legacy
    ``timestamp`` column is interpreted as the signal bar's opening timestamp.
    An existing ``decision_ts`` is audited but never allowed to move the
    mandatory decision before the completed signal-bar close.
    """
    if "signal_bar_ts" in frame.columns:
        signal = to_utc_index(frame["signal_bar_ts"], errors="coerce")
    elif "signal_timestamp" in frame.columns:
        signal = to_utc_index(frame["signal_timestamp"], errors="coerce")
    elif timestamp_col in frame.columns:
        signal = to_utc_index(frame[timestamp_col], errors="coerce")
    else:
        raise KeyError(
            "Causal replay requires signal_bar_ts, signal_timestamp, or "
            f"{timestamp_col!r}"
        )
    decision = causal_decision_timestamps(signal, timeframe=timeframe)
    if "decision_ts" in frame.columns:
        recorded = to_utc_index(frame["decision_ts"], errors="coerce")
        invalid = recorded.notna() & (recorded < decision)
        if bool(invalid.any()):
            raise ValueError(
                "Recorded decision_ts precedes signal_ts + timeframe for "
                f"{int(invalid.sum())} rows"
            )
    return signal, decision


def causal_execution_times(
    frame: pd.DataFrame,
    *,
    timeframe: Any,
    delay_minutes: int = 0,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Resolve signal, mandatory decision, and executable path timestamps."""
    if int(delay_minutes) < 0:
        raise ValueError("delay_minutes must be non-negative")
    signal, decision = causal_signal_times(
        frame,
        timeframe=timeframe,
        timestamp_col=timestamp_col,
    )
    requested = decision + pd.Timedelta(minutes=int(delay_minutes))
    if "delayed_entry_effective_ts" in frame.columns:
        actual = to_utc_index(frame["delayed_entry_effective_ts"], errors="coerce")
        entry = to_utc_index(
            np.where(actual.notna(), actual.to_numpy(), requested.to_numpy()),
            errors="coerce",
        )
    else:
        entry = requested
    invalid = entry.isna() | (entry < decision)
    if bool(invalid.any()):
        raise ValueError(
            "Executable entry timestamp precedes the mandatory decision timestamp "
            f"for {int(invalid.sum())} rows"
        )
    return signal, decision, entry


def assert_first_path_timestamp(
    *,
    first_path_ts: Any,
    signal_ts: Any,
    timeframe: Any,
) -> None:
    """Enforce that an outcome/execution path cannot overlap its signal bar."""
    first = to_utc_index(first_path_ts, errors="coerce")
    decision = causal_decision_timestamps(signal_ts, timeframe=timeframe)
    invalid = first.isna() | decision.isna() | (first < decision)
    if bool(invalid.any()):
        raise AssertionError(
            "first_path_timestamp must be >= signal_timestamp + timeframe; "
            f"invalid_rows={int(invalid.sum())}"
        )
