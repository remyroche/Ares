from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional


def parse_datetime_to_ms(dt_str: str) -> Optional[int]:
    """Parse a datetime string to milliseconds since epoch (UTC).

    Accepts common ISO-8601 formats like '2024-07-01', '2024-07-01T12:34:56Z',
    '2024-07-01 12:34:56', with or without timezone; assumes UTC if none.
    Returns None if parsing fails.
    """
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    try:
        # Last-resort: fromisoformat without 'Z'
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def resolve_time_window_ms(
    config: dict | None = None,
) -> tuple[Optional[int], Optional[int]]:
    """Resolve t0_ms/t1_ms from config or environment.

    Order of precedence:
      1) config['t0_ms']/config['t1_ms']
      2) config['start_timestamp_ms']/config['end_timestamp_ms']
      3) ENV: ARES_T0_MS/ARES_T1_MS
      4) config['start_datetime']/config['end_datetime'] (ISO strings)
      5) ENV: ARES_START_DATETIME/ARES_END_DATETIME (ISO strings)
    """
    cfg = config or {}

    def as_int(v: object) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    t0 = (
        as_int(cfg.get("t0_ms"))
        or as_int(cfg.get("start_timestamp_ms"))
        or as_int(os.environ.get("ARES_T0_MS"))
    )
    t1 = (
        as_int(cfg.get("t1_ms"))
        or as_int(cfg.get("end_timestamp_ms"))
        or as_int(os.environ.get("ARES_T1_MS"))
    )

    if t0 is None:
        t0 = parse_datetime_to_ms(
            cfg.get("start_datetime") or os.environ.get("ARES_START_DATETIME", "")
        )
    if t1 is None:
        t1 = parse_datetime_to_ms(
            cfg.get("end_datetime") or os.environ.get("ARES_END_DATETIME", "")
        )

    return t0, t1
