#!/usr/bin/env python3
"""Audit whether the next no-backfill shadow window can be scored.

This is intentionally read-only unless ``--update-config`` is passed.  It does
not materialize candidates, replay policy, or promote a controller.  Its job is
to make the current frontier explicit: latest feature-store timestamp, replay
maturity cutoff, latest monitored score window, and the next window that can be
scored under the active T1 global-rank contract.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_CONFIG = Path("config/reliability_blend_production_stack.json")
DEFAULT_DATA_ROOT = Path("data_perp")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_next_no_backfill_shadow_window_readiness")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _latest_feature_store_dir(data_root: Path) -> Path | None:
    root = data_root / "features"
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda p: p.name)[-1] if dirs else None


def _feature_delta_duckdb_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.name + ".deltas.duckdb")


def _empty_utc_ts_series() -> pd.Series:
    return pd.Series(dtype="datetime64[ns, UTC]")


def _read_delta_duckdb_timestamps(
    parquet_path: Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    delta_path = _feature_delta_duckdb_path(parquet_path)
    if not delta_path.exists():
        return _empty_utc_ts_series()
    try:
        import duckdb
    except Exception:
        return _empty_utc_ts_series()
    con = None
    try:
        con = duckdb.connect(str(delta_path), read_only=True)
        table_exists = bool(
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'feature_deltas'"
            ).fetchone()[0]
        )
        if not table_exists:
            return pd.Series(dtype="datetime64[ns, UTC]")
        cols = {str(row[1]) for row in con.execute("PRAGMA table_info('feature_deltas')").fetchall()}
        if "ts" not in cols:
            return _empty_utc_ts_series()
        filters: list[str] = []
        params: list[Any] = []
        if start is not None:
            filters.append("ts >= ?")
            params.append(start.to_pydatetime())
        if end is not None:
            filters.append("ts <= ?")
            params.append(end.to_pydatetime())
        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        values = con.execute(f"SELECT ts FROM feature_deltas{where}", params).fetchdf()["ts"]
        return pd.to_datetime(values, utc=True, errors="coerce").dropna()
    except Exception:
        return _empty_utc_ts_series()
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _filter_window_timestamps(
    values: Any,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(ts)
    else:
        ts = pd.Series(ts)
    ts = ts.dropna()
    if ts.empty:
        return _empty_utc_ts_series()
    return ts[(ts >= start) & (ts <= end)].reset_index(drop=True)


def _read_parquet_timestamp_window(
    path: Path,
    *,
    ts_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int = 65536,
) -> pd.Series:
    """Read only timestamps needed for a coverage window.

    The previous implementation read the full timestamp column for every
    feature file. Live feature stores can be large, and the readiness audit only
    needs a narrow next-window slice, so prefer parquet filters and fall back to
    bounded batches instead of a full-column materialization.
    """

    try:
        table = pq.read_table(
            path,
            columns=[ts_col],
            filters=[
                (ts_col, ">=", start.to_pydatetime()),
                (ts_col, "<=", end.to_pydatetime()),
            ],
        )
        return _filter_window_timestamps(table[ts_col].to_pandas(), start=start, end=end)
    except Exception:
        pass

    chunks: list[pd.Series] = []
    try:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=[ts_col], batch_size=int(batch_size)):
            if batch.num_rows <= 0:
                continue
            chunk = _filter_window_timestamps(
                batch.column(0).to_pandas(),
                start=start,
                end=end,
            )
            if not chunk.empty:
                chunks.append(chunk)
    except Exception:
        return _empty_utc_ts_series()
    if not chunks:
        return _empty_utc_ts_series()
    return pd.concat(chunks, ignore_index=True)


def feature_store_bounds(feature_store_dir: Path) -> dict[str, Any]:
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    row_count = 0
    file_count = 0
    missing_ts_files: list[str] = []
    unreadable_files: list[str] = []
    for path in sorted(feature_store_dir.glob("*.parquet")):
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            unreadable_files.append(path.name)
            continue
        names = list(pf.schema_arrow.names)
        ts_col = "ts" if "ts" in names else "timestamp" if "timestamp" in names else ""
        if not ts_col:
            missing_ts_files.append(path.name)
            continue
        ts_idx = names.index(ts_col)
        file_min: pd.Timestamp | None = None
        file_max: pd.Timestamp | None = None
        for rg_idx in range(pf.metadata.num_row_groups):
            group = pf.metadata.row_group(rg_idx)
            row_count += int(group.num_rows)
            stats = group.column(ts_idx).statistics
            if stats is None or not stats.has_min_max:
                continue
            rg_min = _as_utc(stats.min)
            rg_max = _as_utc(stats.max)
            if rg_min is not None:
                file_min = rg_min if file_min is None else min(file_min, rg_min)
            if rg_max is not None:
                file_max = rg_max if file_max is None else max(file_max, rg_max)
        if file_min is not None and file_max is not None:
            mins.append(file_min)
            maxs.append(file_max)
            file_count += 1
        delta_ts = _read_delta_duckdb_timestamps(path)
        if not delta_ts.empty:
            row_count += int(len(delta_ts))
            mins.append(pd.Timestamp(delta_ts.min()))
            maxs.append(pd.Timestamp(delta_ts.max()))
    return {
        "feature_store_dir": str(feature_store_dir),
        "feature_file_count": int(file_count),
        "feature_row_count": int(row_count),
        "feature_timestamp_min": min(mins).isoformat() if mins else None,
        "feature_timestamp_max": max(maxs).isoformat() if maxs else None,
        "missing_ts_file_count": int(len(missing_ts_files)),
        "unreadable_file_count": int(len(unreadable_files)),
        "missing_ts_files_sample": missing_ts_files[:10],
        "unreadable_files_sample": unreadable_files[:10],
    }


def feature_store_hourly_coverage(
    feature_store_dir: Path,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    missing_file_sample_per_low_hour: int = 10,
) -> dict[str, Any]:
    """Measure feature-file timestamp coverage over a specific hourly window."""

    if start is None or end is None or end < start:
        return {
            "coverage_start": start.isoformat() if start is not None else None,
            "coverage_end": end.isoformat() if end is not None else None,
            "coverage_hour_count": 0,
            "coverage_feature_file_count": 0,
            "min_feature_file_coverage": None,
            "mean_feature_file_coverage": None,
            "low_coverage_timestamp_count": 0,
            "low_coverage_timestamps_sample": [],
            "low_coverage_missing_file_count_by_timestamp": {},
            "low_coverage_present_file_count_by_timestamp": {},
            "low_coverage_missing_files_sample_by_timestamp": {},
            "low_coverage_present_files_sample_by_timestamp": {},
            "low_coverage_feature_file_coverage_by_timestamp": {},
            "low_coverage_gap_type_by_timestamp": {},
            "low_coverage_gap_type_counts": {},
            "low_coverage_blocks_threshold_by_timestamp": {},
            "blocking_low_coverage_gap_type_counts": {},
            "blocking_low_coverage_timestamps_sample": [],
        }

    counts: dict[pd.Timestamp, int] = {}
    present_files_by_hour: dict[pd.Timestamp, set[str]] = {}
    feature_files: list[str] = []
    feature_file_count = 0
    hours = pd.date_range(start=start.floor("h"), end=end.floor("h"), freq="1h")
    wanted = set(hours)
    for path in sorted(feature_store_dir.glob("*.parquet")):
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue
        names = list(pf.schema_arrow.names)
        ts_col = "ts" if "ts" in names else "timestamp" if "timestamp" in names else ""
        if not ts_col:
            continue
        ts = _read_parquet_timestamp_window(
            path,
            ts_col=ts_col,
            start=start,
            end=end,
        )
        delta_ts = _read_delta_duckdb_timestamps(path, start=start, end=end)
        if not delta_ts.empty:
            ts = pd.concat([pd.Series(ts), pd.Series(delta_ts)], ignore_index=True)
        present = set(pd.Series(ts).dropna().dt.floor("h").unique())
        relevant = wanted.intersection(present)
        feature_files.append(path.name)
        for hour in relevant:
            normalized_hour = pd.Timestamp(hour)
            counts[normalized_hour] = counts.get(normalized_hour, 0) + 1
            present_files_by_hour.setdefault(normalized_hour, set()).add(path.name)
        feature_file_count += 1

    if feature_file_count <= 0 or len(hours) == 0:
        ratios = np.zeros(len(hours), dtype=np.float64)
    else:
        ratios = np.asarray(
            [counts.get(pd.Timestamp(hour), 0) / float(feature_file_count) for hour in hours],
            dtype=np.float64,
        )
    low = [
        hour.isoformat()
        for hour, ratio in zip(hours, ratios)
        if not np.isfinite(ratio) or float(ratio) < 1.0
    ]
    all_feature_files = set(feature_files)
    missing_counts: dict[str, int] = {}
    present_counts: dict[str, int] = {}
    missing_samples: dict[str, list[str]] = {}
    present_samples: dict[str, list[str]] = {}
    low_coverage_ratios: dict[str, float] = {}
    sample_n = max(0, int(missing_file_sample_per_low_hour))
    for hour, ratio in zip(hours, ratios):
        if np.isfinite(ratio) and float(ratio) >= 1.0:
            continue
        hour_key = pd.Timestamp(hour).isoformat()
        present_for_hour = set(present_files_by_hour.get(pd.Timestamp(hour), set()))
        missing = sorted(all_feature_files.difference(present_for_hour))
        present_sorted = sorted(present_for_hour)
        present_counts[hour_key] = int(len(present_for_hour))
        missing_counts[hour_key] = int(len(missing))
        missing_samples[hour_key] = missing[:sample_n]
        present_samples[hour_key] = present_sorted[:sample_n]
        low_coverage_ratios[hour_key] = float(ratio) if np.isfinite(ratio) else 0.0
    return {
        "coverage_start": start.isoformat(),
        "coverage_end": end.isoformat(),
        "coverage_hour_count": int(len(hours)),
        "coverage_feature_file_count": int(feature_file_count),
        "min_feature_file_coverage": float(np.nanmin(ratios)) if ratios.size else None,
        "mean_feature_file_coverage": float(np.nanmean(ratios)) if ratios.size else None,
        "low_coverage_timestamp_count": int(len(low)),
        "low_coverage_timestamps_sample": low[:10],
        "low_coverage_missing_file_count_by_timestamp": missing_counts,
        "low_coverage_present_file_count_by_timestamp": present_counts,
        "low_coverage_missing_files_sample_by_timestamp": missing_samples,
        "low_coverage_present_files_sample_by_timestamp": present_samples,
        "low_coverage_feature_file_coverage_by_timestamp": low_coverage_ratios,
    }


def _coverage_gap_type(
    *,
    timestamp: str,
    present_count: Any,
    missing_count: Any,
    feature_timestamp_max: pd.Timestamp | None,
) -> str:
    ts = _as_utc(timestamp)
    try:
        present = int(present_count or 0)
    except Exception:
        present = 0
    try:
        missing = int(missing_count or 0)
    except Exception:
        missing = 0
    if missing <= 0:
        return "complete"
    if ts is None or feature_timestamp_max is None:
        return "unknown_low_coverage"
    if ts > feature_timestamp_max.floor("h"):
        return "tail_not_generated_yet"
    if present <= 0:
        return "internal_total_gap"
    if missing == 1:
        return "internal_single_feature_gap"
    return "internal_partial_gap"


def annotate_coverage_gaps(
    coverage: dict[str, Any],
    *,
    feature_timestamp_max: pd.Timestamp | None,
    min_feature_timestamp_coverage: float,
) -> dict[str, Any]:
    """Add actionable gap classes to a coverage dictionary."""

    out = dict(coverage)
    missing_counts = dict(out.get("low_coverage_missing_file_count_by_timestamp") or {})
    present_counts = dict(out.get("low_coverage_present_file_count_by_timestamp") or {})
    ratios = dict(out.get("low_coverage_feature_file_coverage_by_timestamp") or {})
    gap_types: dict[str, str] = {}
    blocking: dict[str, bool] = {}
    blocking_samples: list[str] = []
    min_cov = float(min_feature_timestamp_coverage)
    for timestamp in sorted(missing_counts):
        gap_types[timestamp] = _coverage_gap_type(
            timestamp=timestamp,
            present_count=present_counts.get(timestamp),
            missing_count=missing_counts.get(timestamp),
            feature_timestamp_max=feature_timestamp_max,
        )
        ratio = pd.to_numeric(pd.Series([ratios.get(timestamp)]), errors="coerce").iloc[0]
        is_blocking = bool(not np.isfinite(ratio) or float(ratio) < min_cov)
        blocking[timestamp] = is_blocking
        if is_blocking:
            blocking_samples.append(timestamp)
    blocking_gap_types = [
        gap_types[timestamp] for timestamp, is_blocking in blocking.items() if bool(is_blocking)
    ]
    out["low_coverage_gap_type_by_timestamp"] = gap_types
    out["low_coverage_gap_type_counts"] = dict(Counter(gap_types.values()))
    out["low_coverage_blocks_threshold_by_timestamp"] = blocking
    out["blocking_low_coverage_gap_type_counts"] = dict(Counter(blocking_gap_types))
    out["blocking_low_coverage_timestamps_sample"] = blocking_samples[:10]
    return out


def _coverage_repair_action(
    *,
    min_window_coverage: dict[str, Any],
    full_window_coverage: dict[str, Any],
) -> str:
    min_blockers = set(
        dict(min_window_coverage.get("blocking_low_coverage_gap_type_counts") or {}).keys()
    )
    full_blockers = set(
        dict(full_window_coverage.get("blocking_low_coverage_gap_type_counts") or {}).keys()
    )
    internal = {"internal_total_gap", "internal_partial_gap"}
    single = {"internal_single_feature_gap"}
    if min_blockers.intersection(internal):
        return "backfill_internal_feature_gaps_before_shadow_score"
    if min_blockers.intersection(single):
        return "repair_single_feature_gaps_or_explicit_symbol_exclusion_before_shadow_score"
    if "tail_not_generated_yet" in min_blockers:
        return "wait_for_or_generate_tail_feature_history_before_shadow_score"
    if full_blockers.intersection(internal):
        return "backfill_internal_feature_gaps_before_full_shadow_window"
    if full_blockers.intersection(single):
        return "repair_single_feature_gaps_or_explicit_symbol_exclusion_before_full_shadow_window"
    if "tail_not_generated_yet" in full_blockers:
        return "wait_for_or_generate_tail_feature_history_before_full_shadow_window"
    return "no_feature_coverage_repair_required"


def _controller_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("market_state_controller_validation") or {})


def _candidate_period_from_parquet(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not path.exists():
        return None, None
    try:
        timestamps = pd.to_datetime(
            pd.read_parquet(path, columns=["timestamp"])["timestamp"],
            utc=True,
            errors="coerce",
        ).dropna()
    except Exception:
        return None, None
    if timestamps.empty:
        return None, None
    return pd.Timestamp(timestamps.min()), pd.Timestamp(timestamps.max())


def _score_manifest_period(score_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    manifest_path = score_dir / "manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return None, None
    start = _as_utc(manifest.get("period_start") or manifest.get("window_start"))
    end = _as_utc(manifest.get("period_end") or manifest.get("window_end"))
    if start is not None and end is not None:
        return start, end
    eval_candidates_raw = manifest.get("eval_candidates")
    if eval_candidates_raw:
        cand_start, cand_end = _candidate_period_from_parquet(Path(str(eval_candidates_raw)))
        start = start or cand_start
        end = end or cand_end
    return start, end


def _latest_monitor_summary_window_end(monitor: dict[str, Any]) -> pd.Timestamp | None:
    candidates: list[pd.Timestamp] = []
    summary_path_raw = monitor.get("summary_json")
    if summary_path_raw:
        summary_path = Path(str(summary_path_raw))
        if summary_path.exists():
            try:
                summary = _load_json(summary_path)
            except Exception:
                summary = {}
            for window in summary.get("windows") or []:
                if isinstance(window, dict):
                    ts = _as_utc(window.get("period_end"))
                    if ts is not None:
                        candidates.append(ts)
            metrics_csv_raw = summary.get("window_metrics_csv")
            if metrics_csv_raw:
                metrics_csv = Path(str(metrics_csv_raw))
                if metrics_csv.exists():
                    try:
                        frame = pd.read_csv(metrics_csv, usecols=["period_end"])
                    except Exception:
                        frame = pd.DataFrame()
                    if "period_end" in frame.columns:
                        ts = pd.to_datetime(frame["period_end"], utc=True, errors="coerce").dropna()
                        if not ts.empty:
                            candidates.append(pd.Timestamp(ts.max()))
    return max(candidates) if candidates else None


def _latest_config_window_end(config: dict[str, Any]) -> pd.Timestamp | None:
    controller = _controller_config(config)
    candidates: list[pd.Timestamp] = []
    monitor = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_monitor") or {}
    )
    for window in monitor.get("windows") or []:
        if isinstance(window, dict):
            ts = _as_utc(window.get("period_end"))
            if ts is not None:
                candidates.append(ts)
    monitor_summary_end = _latest_monitor_summary_window_end(monitor)
    if monitor_summary_end is not None:
        candidates.append(monitor_summary_end)
    latest_score = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_score_latest") or {}
    )
    latest_score_end = _as_utc(latest_score.get("period_end"))
    if latest_score_end is not None:
        candidates.append(latest_score_end)
    score_dir_raw = latest_score.get("score_dir")
    if score_dir_raw:
        _score_start, score_end = _score_manifest_period(Path(str(score_dir_raw)))
        if score_end is not None:
            candidates.append(score_end)
    runner = dict(
        controller.get("global_rank_threshold_controller_no_backfill_next_window_runner") or {}
    )
    runner_score_dir_raw = runner.get("score_output_dir")
    if runner_score_dir_raw:
        _runner_score_start, runner_score_end = _score_manifest_period(
            Path(str(runner_score_dir_raw))
        )
        if runner_score_end is not None:
            candidates.append(runner_score_end)
    for key, field in (
        ("global_rank_threshold_controller_no_backfill_shadow_window_discovery", "latest_discovered_window_end"),
    ):
        ts = _as_utc(dict(controller.get(key) or {}).get(field))
        if ts is not None:
            candidates.append(ts)
    return max(candidates) if candidates else None


def _latest_anchor_candidate_manifest(data_root: Path) -> dict[str, Any]:
    manifests = sorted(
        (data_root / "artifacts").glob("*anchor_scored_candidates/t1_anchor_scored_candidate_manifest.json")
    )
    best: dict[str, Any] = {}
    best_ts: pd.Timestamp | None = None
    for path in manifests:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        ts = _as_utc(payload.get("timestamp_max"))
        if ts is not None and (best_ts is None or ts > best_ts):
            best_ts = ts
            best = dict(payload)
            best["manifest_path"] = str(path)
    return best


def build_readiness(
    *,
    config: dict[str, Any],
    config_path: Path,
    data_root: Path,
    feature_store_dir: Path,
    output_dir: Path,
    maturity_buffer_hours: int,
    target_window_hours: int,
    min_timestamp_count: int,
    min_feature_timestamp_coverage: float,
) -> dict[str, Any]:
    active = dict(config.get("active_stack") or {})
    controller = _controller_config(config)
    discovery = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_window_discovery")
        or {}
    )
    bounds = feature_store_bounds(feature_store_dir)
    feature_max = _as_utc(bounds.get("feature_timestamp_max"))
    latest_scored_end = _latest_config_window_end(config)
    next_start = (
        latest_scored_end + pd.Timedelta(hours=1)
        if latest_scored_end is not None
        else None
    )
    maturity_cutoff = (
        feature_max - pd.Timedelta(hours=int(maturity_buffer_hours))
        if feature_max is not None
        else None
    )
    target_end = (
        next_start + pd.Timedelta(hours=max(0, int(target_window_hours) - 1))
        if next_start is not None
        else None
    )
    proposed_end = None
    mature_timestamp_count = 0
    if next_start is not None and maturity_cutoff is not None and maturity_cutoff >= next_start:
        proposed_end = min(target_end, maturity_cutoff) if target_end is not None else maturity_cutoff
        mature_timestamp_count = int((proposed_end - next_start) / pd.Timedelta(hours=1)) + 1

    min_needed_feature_max = (
        next_start
        + pd.Timedelta(hours=max(0, int(min_timestamp_count) - 1))
        + pd.Timedelta(hours=int(maturity_buffer_hours))
        if next_start is not None
        else None
    )
    full_needed_feature_max = (
        next_start
        + pd.Timedelta(hours=max(0, int(target_window_hours) - 1))
        + pd.Timedelta(hours=int(maturity_buffer_hours))
        if next_start is not None
        else None
    )
    minimum_window_end = (
        next_start + pd.Timedelta(hours=max(0, int(min_timestamp_count) - 1))
        if next_start is not None
        else None
    )
    min_window_coverage = feature_store_hourly_coverage(
        feature_store_dir,
        start=next_start,
        end=minimum_window_end,
    )
    full_window_coverage = feature_store_hourly_coverage(
        feature_store_dir,
        start=next_start,
        end=target_end,
    )
    min_cov = float(min_feature_timestamp_coverage)
    min_window_coverage = annotate_coverage_gaps(
        min_window_coverage,
        feature_timestamp_max=feature_max,
        min_feature_timestamp_coverage=min_cov,
    )
    full_window_coverage = annotate_coverage_gaps(
        full_window_coverage,
        feature_timestamp_max=feature_max,
        min_feature_timestamp_coverage=min_cov,
    )
    min_window_feature_coverage_ready = bool(
        min_window_coverage.get("min_feature_file_coverage") is not None
        and float(min_window_coverage["min_feature_file_coverage"]) >= min_cov
    )
    full_window_feature_coverage_ready = bool(
        full_window_coverage.get("min_feature_file_coverage") is not None
        and float(full_window_coverage["min_feature_file_coverage"]) >= min_cov
    )

    failures: list[str] = []
    if active.get("rank_contract") != "anchor_global_policy_rank_reference":
        failures.append("active_rank_contract_not_global_policy_rank_reference")
    if active.get("rank_scope") != "global_over_time":
        failures.append("active_rank_scope_not_global_over_time")
    if feature_max is None:
        failures.append("feature_store_has_no_timestamp_bounds")
    if latest_scored_end is None:
        failures.append("no_existing_no_backfill_window_end_found")
    if mature_timestamp_count < int(min_timestamp_count):
        failures.append("insufficient_matured_timestamps_for_minimum_shadow_window")
    if mature_timestamp_count < int(target_window_hours):
        failures.append("insufficient_matured_timestamps_for_full_shadow_window")
    if not min_window_feature_coverage_ready:
        failures.append("insufficient_feature_timestamp_coverage_for_minimum_shadow_window")
    if not full_window_feature_coverage_ready:
        failures.append("insufficient_feature_timestamp_coverage_for_full_shadow_window")
    if int(discovery.get("appendable_candidate_count") or 0) > 0:
        failures.append("appendable_window_already_exists_discovery_should_be_appended_first")

    scoreable_min_window_now = (
        feature_max is not None
        and latest_scored_end is not None
        and mature_timestamp_count >= int(min_timestamp_count)
        and active.get("rank_contract") == "anchor_global_policy_rank_reference"
        and active.get("rank_scope") == "global_over_time"
        and int(discovery.get("appendable_candidate_count") or 0) == 0
        and min_window_feature_coverage_ready
    )
    scoreable_full_window_now = bool(
        scoreable_min_window_now
        and mature_timestamp_count >= int(target_window_hours)
        and full_window_feature_coverage_ready
    )
    latest_manifest = _latest_anchor_candidate_manifest(data_root)

    def _missing_hours(required: pd.Timestamp | None) -> int | None:
        if required is None or feature_max is None:
            return None
        delta = required - feature_max
        if delta <= pd.Timedelta(0):
            return 0
        return int(np.ceil(delta / pd.Timedelta(hours=1)))

    coverage_repair_action = _coverage_repair_action(
        min_window_coverage=min_window_coverage,
        full_window_coverage=full_window_coverage,
    )
    if scoreable_full_window_now:
        next_action = "materialize_and_score_full_next_no_backfill_shadow_window"
    elif scoreable_min_window_now:
        next_action = "materialize_and_score_partial_next_no_backfill_shadow_window"
    else:
        next_action = (
            coverage_repair_action
            if coverage_repair_action != "no_feature_coverage_repair_required"
            else "wait_for_more_mature_shadow_timestamps_before_next_shadow_score"
        )

    return {
        "generated_by": "audit_next_no_backfill_shadow_window_readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "active_rank_contract": active.get("rank_contract"),
        "active_rank_scope": active.get("rank_scope"),
        "active_heads": active.get("enabled_heads") or [],
        "disabled_heads": active.get("disabled_heads") or [],
        "qfail_active": bool(active.get("qfail_active")),
        "threshold_controller_active": bool(active.get("market_state_threshold_controller_active")),
        **bounds,
        "maturity_buffer_hours": int(maturity_buffer_hours),
        "maturity_cutoff": maturity_cutoff.isoformat() if maturity_cutoff is not None else None,
        "latest_scored_or_discovered_window_end": (
            latest_scored_end.isoformat() if latest_scored_end is not None else None
        ),
        "next_window_start": next_start.isoformat() if next_start is not None else None,
        "target_window_hours": int(target_window_hours),
        "target_window_end": target_end.isoformat() if target_end is not None else None,
        "minimum_window_end": (
            minimum_window_end.isoformat() if minimum_window_end is not None else None
        ),
        "proposed_scoreable_window_end": (
            proposed_end.isoformat() if proposed_end is not None else None
        ),
        "mature_timestamp_count_available": int(mature_timestamp_count),
        "min_timestamp_count": int(min_timestamp_count),
        "min_feature_timestamp_coverage": min_cov,
        "minimum_window_feature_coverage": min_window_coverage,
        "full_window_feature_coverage": full_window_coverage,
        "minimum_window_feature_coverage_ready": min_window_feature_coverage_ready,
        "full_window_feature_coverage_ready": full_window_feature_coverage_ready,
        "scoreable_min_window_now": bool(scoreable_min_window_now),
        "scoreable_full_window_now": bool(scoreable_full_window_now),
        "needed_feature_timestamp_max_for_min_window": (
            min_needed_feature_max.isoformat() if min_needed_feature_max is not None else None
        ),
        "needed_feature_timestamp_max_for_full_window": (
            full_needed_feature_max.isoformat() if full_needed_feature_max is not None else None
        ),
        "missing_feature_hours_for_min_window": _missing_hours(min_needed_feature_max),
        "missing_feature_hours_for_full_window": _missing_hours(full_needed_feature_max),
        "coverage_repair_action": coverage_repair_action,
        "latest_anchor_candidate_manifest": latest_manifest.get("manifest_path"),
        "latest_anchor_candidate_timestamp_max": latest_manifest.get("timestamp_max"),
        "latest_anchor_candidate_rows": latest_manifest.get("rows"),
        "latest_anchor_candidate_deployable_rows": latest_manifest.get("deployable_rows"),
        "latest_anchor_candidate_heads": latest_manifest.get("heads"),
        "discovery_appendable_count": discovery.get("appendable_candidate_count"),
        "discovery_latest_discovered_window_end": discovery.get("latest_discovered_window_end"),
        "failures": failures,
        "status": "scoreable_now" if scoreable_min_window_now else "not_scoreable_yet",
        "next_action": next_action,
        "interpretation": (
            "A later no-backfill shadow window can be scored now."
            if scoreable_min_window_now
            else "No new no-backfill shadow window is scoreable yet under the active "
            "global-rank T1 contract; more mature feature/path history is required."
        ),
    }


def write_readiness(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "next_no_backfill_shadow_window_readiness.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "next_no_backfill_shadow_window_readiness.csv",
        index=False,
    )
    low_coverage_rows = []
    full_coverage = summary.get("full_window_feature_coverage")
    if isinstance(full_coverage, dict):
        missing_counts = dict(full_coverage.get("low_coverage_missing_file_count_by_timestamp") or {})
        present_counts = dict(full_coverage.get("low_coverage_present_file_count_by_timestamp") or {})
        missing_samples = dict(full_coverage.get("low_coverage_missing_files_sample_by_timestamp") or {})
        present_samples = dict(full_coverage.get("low_coverage_present_files_sample_by_timestamp") or {})
        ratios = dict(full_coverage.get("low_coverage_feature_file_coverage_by_timestamp") or {})
        gap_types = dict(full_coverage.get("low_coverage_gap_type_by_timestamp") or {})
        blocks_threshold = dict(
            full_coverage.get("low_coverage_blocks_threshold_by_timestamp") or {}
        )
        for timestamp in sorted(missing_counts):
            low_coverage_rows.append(
                {
                    "timestamp": timestamp,
                    "coverage_gap_type": gap_types.get(timestamp),
                    "feature_file_coverage": ratios.get(timestamp),
                    "blocks_min_feature_coverage_threshold": blocks_threshold.get(timestamp),
                    "present_feature_file_count": present_counts.get(timestamp),
                    "missing_feature_file_count": missing_counts.get(timestamp),
                    "present_feature_files_sample": "|".join(
                        map(str, present_samples.get(timestamp) or [])
                    ),
                    "missing_feature_files_sample": "|".join(
                        map(str, missing_samples.get(timestamp) or [])
                    ),
                }
            )
    low_coverage_frame = pd.DataFrame(low_coverage_rows)
    low_coverage_frame.to_csv(
        output_dir / "next_no_backfill_shadow_window_low_coverage_hours.csv",
        index=False,
    )
    if low_coverage_frame.empty:
        low_coverage_table = "_No low-coverage hours._"
    else:
        low_coverage_table = low_coverage_frame.head(20).to_markdown(index=False)
    lines = [
        "# Next No-Backfill Shadow Window Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Active rank contract: `{summary['active_rank_contract']}`",
        f"- Active rank scope: `{summary['active_rank_scope']}`",
        f"- Feature store: `{summary['feature_store_dir']}`",
        f"- Feature timestamp max: `{summary['feature_timestamp_max']}`",
        f"- Maturity buffer hours: `{summary['maturity_buffer_hours']}`",
        f"- Maturity cutoff: `{summary['maturity_cutoff']}`",
        f"- Latest scored/discovered window end: `{summary['latest_scored_or_discovered_window_end']}`",
        f"- Next window start: `{summary['next_window_start']}`",
        f"- Target window end: `{summary['target_window_end']}`",
        f"- Minimum window end: `{summary['minimum_window_end']}`",
        f"- Proposed scoreable window end: `{summary['proposed_scoreable_window_end']}`",
        f"- Mature timestamps available: `{summary['mature_timestamp_count_available']}`",
        f"- Minimum timestamps required: `{summary['min_timestamp_count']}`",
        f"- Full-window timestamps required: `{summary['target_window_hours']}`",
        f"- Minimum feature timestamp coverage: `{summary['min_feature_timestamp_coverage']}`",
        f"- Minimum-window feature coverage ready: `{summary['minimum_window_feature_coverage_ready']}`",
        f"- Minimum-window min feature-file coverage: `{summary['minimum_window_feature_coverage']['min_feature_file_coverage']}`",
        f"- Full-window feature coverage ready: `{summary['full_window_feature_coverage_ready']}`",
        f"- Full-window min feature-file coverage: `{summary['full_window_feature_coverage']['min_feature_file_coverage']}`",
        f"- Scoreable minimum window now: `{summary['scoreable_min_window_now']}`",
        f"- Scoreable full window now: `{summary['scoreable_full_window_now']}`",
        f"- Needed feature max for minimum window: `{summary['needed_feature_timestamp_max_for_min_window']}`",
        f"- Needed feature max for full window: `{summary['needed_feature_timestamp_max_for_full_window']}`",
        f"- Missing feature hours for minimum window: `{summary['missing_feature_hours_for_min_window']}`",
        f"- Missing feature hours for full window: `{summary['missing_feature_hours_for_full_window']}`",
        f"- Coverage repair action: `{summary['coverage_repair_action']}`",
        f"- Latest anchor candidate max timestamp: `{summary['latest_anchor_candidate_timestamp_max']}`",
        f"- Latest anchor candidate rows: `{summary['latest_anchor_candidate_rows']}`",
        f"- Discovery appendable windows: `{summary['discovery_appendable_count']}`",
        f"- Failures: `{', '.join(summary['failures'])}`",
        f"- Next action: `{summary['next_action']}`",
        "",
        "## Coverage Gap Classification",
        "",
        f"- Minimum-window low-coverage gap counts: `{summary['minimum_window_feature_coverage'].get('low_coverage_gap_type_counts')}`",
        f"- Minimum-window blocking gap counts: `{summary['minimum_window_feature_coverage'].get('blocking_low_coverage_gap_type_counts')}`",
        f"- Full-window low-coverage gap counts: `{summary['full_window_feature_coverage'].get('low_coverage_gap_type_counts')}`",
        f"- Full-window blocking gap counts: `{summary['full_window_feature_coverage'].get('blocking_low_coverage_gap_type_counts')}`",
        "",
        "## Low-Coverage Hours",
        "",
        low_coverage_table,
        "",
        str(summary["interpretation"]),
        "",
    ]
    (output_dir / "next_no_backfill_shadow_window_readiness_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def update_config(config: dict[str, Any], config_path: Path, summary: dict[str, Any]) -> None:
    controller = config.setdefault("market_state_controller_validation", {})
    controller["global_rank_threshold_controller_no_backfill_next_window_readiness"] = {
        "readiness_dir": summary["output_dir"],
        "summary_json": str(
            Path(summary["output_dir"]) / "next_no_backfill_shadow_window_readiness.json"
        ),
        "report_md": str(
            Path(summary["output_dir"]) / "next_no_backfill_shadow_window_readiness_report.md"
        ),
        "readiness_csv": str(
            Path(summary["output_dir"]) / "next_no_backfill_shadow_window_readiness.csv"
        ),
        "generated_by": summary["generated_by"],
        "generated_at_utc": summary["generated_at_utc"],
        "status": summary["status"],
        "feature_store_dir": summary["feature_store_dir"],
        "feature_timestamp_max": summary["feature_timestamp_max"],
        "maturity_buffer_hours": summary["maturity_buffer_hours"],
        "maturity_cutoff": summary["maturity_cutoff"],
        "latest_scored_or_discovered_window_end": summary[
            "latest_scored_or_discovered_window_end"
        ],
        "next_window_start": summary["next_window_start"],
        "target_window_end": summary["target_window_end"],
        "minimum_window_end": summary["minimum_window_end"],
        "proposed_scoreable_window_end": summary["proposed_scoreable_window_end"],
        "mature_timestamp_count_available": summary["mature_timestamp_count_available"],
        "min_timestamp_count": summary["min_timestamp_count"],
        "target_window_hours": summary["target_window_hours"],
        "min_feature_timestamp_coverage": summary["min_feature_timestamp_coverage"],
        "minimum_window_feature_coverage": summary["minimum_window_feature_coverage"],
        "full_window_feature_coverage": summary["full_window_feature_coverage"],
        "minimum_window_feature_coverage_ready": summary[
            "minimum_window_feature_coverage_ready"
        ],
        "full_window_feature_coverage_ready": summary[
            "full_window_feature_coverage_ready"
        ],
        "scoreable_min_window_now": summary["scoreable_min_window_now"],
        "scoreable_full_window_now": summary["scoreable_full_window_now"],
        "needed_feature_timestamp_max_for_min_window": summary[
            "needed_feature_timestamp_max_for_min_window"
        ],
        "needed_feature_timestamp_max_for_full_window": summary[
            "needed_feature_timestamp_max_for_full_window"
        ],
        "missing_feature_hours_for_min_window": summary[
            "missing_feature_hours_for_min_window"
        ],
        "missing_feature_hours_for_full_window": summary[
            "missing_feature_hours_for_full_window"
        ],
        "coverage_repair_action": summary["coverage_repair_action"],
        "latest_anchor_candidate_timestamp_max": summary[
            "latest_anchor_candidate_timestamp_max"
        ],
        "failures": summary["failures"],
        "next_action": summary["next_action"],
        "interpretation": summary["interpretation"],
    }
    config_path.write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-store-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maturity-buffer-hours", type=int, default=16)
    parser.add_argument("--target-window-hours", type=int, default=24)
    parser.add_argument("--min-timestamp-count", type=int, default=3)
    parser.add_argument("--min-feature-timestamp-coverage", type=float, default=0.95)
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    config = _load_json(args.config)
    feature_store_dir = args.feature_store_dir or _latest_feature_store_dir(args.data_root)
    if feature_store_dir is None or not feature_store_dir.exists():
        raise SystemExit("No feature-store directory found; pass --feature-store-dir.")
    summary = build_readiness(
        config=config,
        config_path=args.config,
        data_root=args.data_root,
        feature_store_dir=feature_store_dir,
        output_dir=args.output_dir,
        maturity_buffer_hours=int(args.maturity_buffer_hours),
        target_window_hours=int(args.target_window_hours),
        min_timestamp_count=int(args.min_timestamp_count),
        min_feature_timestamp_coverage=float(args.min_feature_timestamp_coverage),
    )
    write_readiness(summary, args.output_dir)
    if bool(args.update_config):
        update_config(config, args.config, summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
