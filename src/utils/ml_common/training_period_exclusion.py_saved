from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _to_utc_naive(ts: Any) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    try:
        out = pd.Timestamp(ts)
    except Exception:
        try:
            out = pd.to_datetime(ts, errors="coerce")
        except Exception:
            return None
    if out is None or pd.isna(out):
        return None
    try:
        if out.tzinfo is not None:
            out = out.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            out = out.tz_localize(None)
        except Exception:
            pass
    return out


def _normalize_range(start: Any, end: Any) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = _to_utc_naive(start)
    e = _to_utc_naive(end)
    if s is None or e is None:
        return None
    if e < s:
        s, e = e, s
    return s, e


def _extract_ranges(value: Any) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if value is None:
        return out

    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is None and "from" in value:
            start = value.get("from")
        if end is None and "to" in value:
            end = value.get("to")
        rng = _normalize_range(start, end)
        return [rng] if rng is not None else []

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rng = _normalize_range(item[0], item[1])
                if rng is not None:
                    out.append(rng)
                continue
            if isinstance(item, dict):
                out.extend(_extract_ranges(item))
                continue
        return out

    return out


def _candidate_cache_metadata_paths(cache_path: str) -> List[str]:
    paths: List[str] = []
    base = str(cache_path)

    if base.endswith(".parquet"):
        paths.append(base.replace(".parquet", "_metadata.json"))
        paths.append(base + ".metadata.json")
        paths.append(base.replace(".parquet", ".json"))
    else:
        paths.append(base + "_metadata.json")
        paths.append(base + ".metadata.json")
        paths.append(base + ".json")

    return paths


def _get_nn_cache_path(config: Dict[str, Any]) -> Optional[str]:
    if not isinstance(config, dict):
        return None

    direct = config.get("nn_embeddings_cache_path")
    if isinstance(direct, str) and direct:
        return direct

    meta_cfg = config.get("meta_feature_engineering")
    if isinstance(meta_cfg, dict):
        p = meta_cfg.get("nn_embeddings_cache_path")
        if isinstance(p, str) and p:
            return p
        seq_cfg = meta_cfg.get("nn_sequence_encoder")
        if isinstance(seq_cfg, dict):
            p2 = seq_cfg.get("cache_path")
            if isinstance(p2, str) and p2:
                return p2

    return None


def _blocked_periods_from_nn_cache_metadata(config: Dict[str, Any]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    cache_path = _get_nn_cache_path(config)
    if not cache_path:
        return []

    for meta_path in _candidate_cache_metadata_paths(cache_path):
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        for start_k, end_k in (
            ("training_start", "training_end"),
            ("train_start", "train_end"),
            ("fit_start", "fit_end"),
        ):
            if start_k in payload and end_k in payload:
                rng = _normalize_range(payload.get(start_k), payload.get(end_k))
                if rng is not None:
                    return [rng]

    return []


def get_blocked_training_periods(config: Dict[str, Any]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(config, dict):
        return []

    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    keys = (
        "blocked_training_periods",
        "blocked_periods",
        "training_blackout_periods",
        "nn_training_periods",
        "nn_exclusion_periods",
        "nn_training_exclusion_periods",
    )

    for k in keys:
        ranges.extend(_extract_ranges(config.get(k)))

    integrity_cfg = config.get("data_integrity")
    if isinstance(integrity_cfg, dict):
        for k in keys:
            ranges.extend(_extract_ranges(integrity_cfg.get(k)))

    allow_cache = config.get("allow_nn_cache_metadata_exclusion", True)
    if allow_cache:
        ranges.extend(_blocked_periods_from_nn_cache_metadata(config))

    buffer_days = None
    for k in (
        "nn_training_exclusion_buffer_days",
        "training_exclusion_buffer_days",
    ):
        if k in config:
            buffer_days = config.get(k)
            break
    if buffer_days is None and isinstance(integrity_cfg, dict):
        buffer_days = integrity_cfg.get("nn_training_exclusion_buffer_days")

    try:
        buffer_days_f = float(buffer_days) if buffer_days is not None else 0.0
    except Exception:
        buffer_days_f = 0.0

    if buffer_days_f > 0.0:
        delta = pd.Timedelta(days=buffer_days_f)
        buffered: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in ranges:
            buffered.append((s - delta, e + delta))
        ranges = buffered

    deduped: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    seen = set()
    for s, e in ranges:
        key = (s.value, e.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((s, e))

    deduped.sort(key=lambda t: t[0])
    return deduped


def filter_index_by_blocked_periods(
    index: pd.Index,
    blocked_periods: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.Index:
    if not blocked_periods:
        return index
    if not isinstance(index, pd.DatetimeIndex):
        return index

    idx = pd.to_datetime(index)
    try:
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

    keep = pd.Series(True, index=idx)
    for start, end in blocked_periods:
        s = _to_utc_naive(start)
        e = _to_utc_naive(end)
        if s is None or e is None:
            continue
        keep &= ~((idx >= s) & (idx <= e))

    return index[keep.to_numpy()]
