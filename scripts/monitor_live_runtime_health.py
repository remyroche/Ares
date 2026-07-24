#!/usr/bin/env python3
"""Lightweight live inference runtime monitor.

This intentionally avoids full replay/scoring work. It samples process memory,
recent inference logs, and the tail of the trade ledger so live monitoring does
not become a second heavy inference job.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


TIMING_RE = re.compile(
    r"\[Timing\]\s+(?P<name>[^:]+):\s+stage=(?P<stage>[0-9.]+)s"
    r".*?rss=(?P<rss>[0-9.]+)MB"
)
LOG_TS_RE = re.compile(
    r"^\[(?P<bracket>[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:.]+ UTC)\]"
    r"|^(?P<plain>[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:,]+)"
)

PREDICTION_LEDGER_COLUMNS = [
    "timestamp",
    "decision_ts",
    "signal_bar_ts",
    "symbol",
    "side",
    "strategy_id",
    "portfolio_decision",
    "portfolio_reject_reason",
    "passed_rank_gate",
    "was_traded",
    "threshold_rank_score",
    "threshold_rank_score_source",
    "threshold_basis_mapped_expected_ev_side_archetype",
    "threshold_basis_side_archetype_recent_ev_correction",
    "threshold_basis_corrected_expected_ev",
    "threshold_basis_corrected_expected_ev_rank",
    "threshold_basis_parent_rank",
    "threshold_basis_blended_rank",
    "threshold_basis_ev_rank_blend_weight",
    "threshold_basis_expected_ev_correction_scope",
    "threshold_basis_recalibration_frequency",
    "threshold_basis_reference_asof",
    "threshold_basis_robust_daily_residual_trim_fraction",
    "threshold_basis_robust_daily_residual_normalization",
    "threshold_basis_global_days_retained",
    "policy_rank_pct",
    "normalized_rank_score",
    "final_gate_rank_score",
    "final_gate_threshold",
    "final_gate_rank_score_source",
    "portfolio_gate_rank_score",
    "portfolio_gate_final_threshold",
    "portfolio_gate_rank_score_source",
    "portfolio_ordering_rank_score",
    "portfolio_allocation_rank_score",
    "dynamic_hr_surprise_threshold",
    "dynamic_hr_surprise_applied",
    "dynamic_hr_surprise_reason",
    "dynamic_hr_surprise_head",
    "dynamic_hr_surprise_z_eff",
    "dynamic_hr_surprise_guarded_y",
    "dynamic_hr_surprise_w_lower",
    "dynamic_hr_surprise_w_raise",
    "dynamic_hr_surprise_state_age_days",
    "inference_drift_score",
    "uncertainty_score",
    "estimated_hit_rate",
    "estimated_hit_rate_source",
    "estimated_ev_net_return",
    "estimated_ev_cost_bps",
    "estimated_ev_hit_rate",
    "ev_adjusted_net_return_after_friction",
    "ev_adjusted_rank_score",
    "expected_entry_price",
    "policy_entry_price",
    "expected_fill_slippage_bps",
    "entry_gap_bps",
    "expected_total_entry_friction_bps",
    "ev_haircut_observed_spread_bps",
    "ev_haircut_spread_baseline_bps",
    "ev_haircut_spread_excess_bps",
    "ev_haircut_orderbook_slippage_bps",
    "ev_haircut_adverse_signal_gap_bps",
    "position_size_after_liquidity",
    "policy_stop_price",
    "stop_price",
    "barrier_frac",
    "policy_sl_mult",
    "spread_bps",
    "ticker_spread_bps",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _read_text_tail(path: Path, *, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _tail_csv_rows(path: Path, *, max_rows: int = 250) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline()
        if not header:
            return []
        tail = deque(handle, maxlen=max_rows)
    reader = csv.DictReader([header, *tail])
    return [dict(row) for row in reader]


def _tail_parquet_frame(
    path: Path,
    *,
    columns: list[str],
    max_rows: int,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    read_columns: list[str] | None = None
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        read_columns = [column for column in columns if column in available]
    except Exception:
        read_columns = None
    try:
        frame = pd.read_parquet(path, columns=read_columns)
    except Exception:
        frame = pd.read_parquet(path)
        keep = [column for column in columns if column in frame.columns]
        if keep:
            frame = frame[keep]
    if max_rows > 0 and len(frame) > max_rows:
        return frame.tail(max_rows).copy()
    return frame.copy()


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _series_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _value_stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"n": 0}
    series = pd.Series(clean, dtype="float64")
    return _series_stats(series)


def _bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    lowered = series.astype(str).str.strip().str.lower()
    out = lowered.isin({"1", "true", "t", "yes", "y"})
    if default:
        out = out | series.isna()
    return out.fillna(default)


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _parse_timestamp(value: Any, *, local_timezone: str) -> pd.Timestamp | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo(local_timezone))
    return ts.tz_convert("UTC")


def _parse_log_timestamp(line: str, *, local_timezone: str) -> pd.Timestamp | None:
    match = LOG_TS_RE.search(str(line or ""))
    if not match:
        return None
    raw = match.group("bracket") or match.group("plain") or ""
    raw = raw.replace(",", ".")
    if raw.endswith(" UTC"):
        raw = raw[:-4] + "+00:00"
    return _parse_timestamp(raw, local_timezone=local_timezone)


def _ps_snapshot(pid: int) -> dict[str, Any]:
    cmd = [
        "ps",
        "-p",
        str(pid),
        "-o",
        "pid=,stat=,etime=,rss=,pcpu=,pmem=,command=",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except OSError as exc:
        alive = False
        try:
            os.kill(pid, 0)
            alive = True
        except PermissionError:
            alive = True
        except OSError:
            alive = False
        return {
            "pid": pid,
            "alive": alive,
            "error": f"ps_unavailable:{type(exc).__name__}:{exc}",
        }
    line = result.stdout.strip()
    if not line:
        return {"pid": pid, "alive": False, "error": result.stderr.strip()}
    parts = line.split(None, 6)
    if len(parts) < 7:
        return {"pid": pid, "alive": True, "raw": line}
    return {
        "pid": int(parts[0]),
        "alive": True,
        "stat": parts[1],
        "etime": parts[2],
        "rss_mb": round(_to_float(parts[3]) / 1024.0, 2),
        "pcpu": _to_float(parts[4]),
        "pmem": _to_float(parts[5]),
        "command": parts[6],
    }


def _latest_json_payload(text: str, marker: str) -> dict[str, Any]:
    idx = text.rfind(marker)
    if idx < 0:
        return {}
    line = text[idx:].splitlines()[0]
    brace = line.find("{")
    if brace < 0:
        return {}
    try:
        payload = json.loads(line[brace:])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _timing_bucket(name: str) -> str:
    raw = str(name or "")
    low = raw.lower()
    if low.startswith("load_or_compute_features") or low.startswith(
        "compute_features_hourly"
    ):
        return "feature_generation"
    if "candidate_feature_matrix" in low or "feature_matrix" in low:
        return "feature_generation"
    if low.startswith("model.") or "_pred" in low or "diagnostics" in low:
        return "model_prediction"
    return "other"


def _log_snapshot(
    path: Path,
    *,
    since: pd.Timestamp,
    local_timezone: str,
) -> dict[str, Any]:
    text = _read_text_tail(path)
    all_lines = [line for line in text.splitlines() if line.strip()]
    lines: list[str] = []
    in_window_context = False
    for line in all_lines:
        ts = _parse_log_timestamp(line, local_timezone=local_timezone)
        if ts is None:
            if in_window_context:
                lines.append(line)
            continue
        in_window_context = ts >= since
        if in_window_context:
            lines.append(line)
    timings: list[dict[str, Any]] = []
    feature_lines: list[str] = []
    parity_lines: list[str] = []
    trade_lines: list[str] = []
    errors: list[str] = []
    reject_counter: Counter[str] = Counter()
    for line in lines:
        match = TIMING_RE.search(line)
        if match:
            timings.append(
                {
                    "name": match.group("name"),
                    "stage_seconds": _to_float(match.group("stage")),
                    "rss_mb": _to_float(match.group("rss")),
                    "line": line[-500:],
                }
            )
        line_l = line.lower()
        if (
            "feature parity" in line_l
            or "strict_feature" in line_l
            or "sidecar" in line_l
            or "feature matrix" in line_l
        ):
            feature_lines.append(line[-500:])
        if "parity" in line_l:
            parity_lines.append(line[-500:])
        if "logged trade event" in line_l or "tradecloseemail" in line_l:
            trade_lines.append(line[-500:])
        if any(token in line for token in ("MemoryError", "Killed", "Traceback", "ERROR")):
            errors.append(line[-500:])
        for reason in re.findall(r"([A-Za-z0-9_:-]*rank_below_dynamic_threshold[A-Za-z0-9_:-]*)", line):
            reject_counter[reason] += 1
        for reason in re.findall(r"([A-Za-z0-9_:-]*entry_cap_reached[A-Za-z0-9_:-]*)", line):
            reject_counter[reason] += 1

    timings_sorted = sorted(
        timings,
        key=lambda item: float(item.get("stage_seconds") or 0.0),
        reverse=True,
    )
    feature_timings = [
        item for item in timings_sorted if _timing_bucket(str(item.get("name"))) == "feature_generation"
    ]
    model_timings = [
        item for item in timings_sorted if _timing_bucket(str(item.get("name"))) == "model_prediction"
    ]
    timing_rss_values = [
        _to_float(item.get("rss_mb")) for item in timings if math.isfinite(_to_float(item.get("rss_mb")))
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count_tail": len(all_lines),
        "line_count_since_start": len(lines),
        "latest_inference_monitor": _latest_json_payload(
            text, "INFERENCE_MONITOR_HEARTBEAT"
        ),
        "latest_executable_stop_sentinel": _latest_json_payload(
            text, "EXECUTABLE_STOP_SENTINEL_HEARTBEAT"
        ),
        "top_timing_hotspots": timings_sorted[:5],
        "feature_generation_hotspots": feature_timings[:8],
        "model_prediction_hotspots": model_timings[:8],
        "max_timing_rss_mb": max(
            [_to_float(item.get("rss_mb")) for item in timings] or [float("nan")]
        ),
        "timing_rss_stats": _value_stats(timing_rss_values),
        "recent_feature_or_sidecar_lines": feature_lines[-8:],
        "recent_parity_lines": parity_lines[-8:],
        "recent_trade_lines": trade_lines[-8:],
        "recent_error_lines": errors[-8:],
        "reject_reason_counts_tail": dict(reject_counter),
    }


def _is_entry_row(row: dict[str, Any]) -> bool:
    lifecycle = str(row.get("lifecycle_event") or "").lower()
    action = str(row.get("action") or "").lower()
    return lifecycle.startswith("entry") or action in {"buy", "sell", "entry"}


def _is_exit_row(row: dict[str, Any]) -> bool:
    lifecycle = str(row.get("lifecycle_event") or "").lower()
    status = str(row.get("status") or "").lower()
    return lifecycle.startswith("exit") or status == "closed" or bool(row.get("exit_reason"))


def _entry_audit(row: dict[str, Any]) -> dict[str, Any]:
    expected_spread_bps = _to_float(row.get("expected_spread_bps"))
    entry_spread_bps = _to_float(row.get("entry_spread_bps"))
    expected_entry_price = _to_float(row.get("expected_entry_price"))
    actual_entry_price = _to_float(row.get("actual_entry_price"))
    entry_spread_delta_bps = _to_float(row.get("entry_vs_expected_spread_bps"))
    if not math.isfinite(entry_spread_delta_bps):
        entry_spread_delta_bps = (
            entry_spread_bps - expected_spread_bps
            if math.isfinite(entry_spread_bps) and math.isfinite(expected_spread_bps)
            else float("nan")
        )
    entry_price_gap_bps = _to_float(row.get("entry_gap_bps"))
    if not math.isfinite(entry_price_gap_bps):
        entry_price_gap_bps = (
            10_000.0
            * (actual_entry_price - expected_entry_price)
            / max(abs(expected_entry_price), 1e-12)
            if math.isfinite(actual_entry_price)
            and math.isfinite(expected_entry_price)
            and expected_entry_price != 0.0
            else float("nan")
        )
    return {
        "timestamp": row.get("timestamp"),
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "status": row.get("status"),
        "strategy_id": row.get("strategy_id"),
        "expected_entry_price": row.get("expected_entry_price"),
        "actual_entry_price": row.get("actual_entry_price"),
        "entry_price_gap_bps": entry_price_gap_bps,
        "signal_gap_bps": row.get("signal_gap_bps"),
        "decision_to_entry_seconds": row.get("decision_to_entry_seconds"),
        "expected_spread_bps": expected_spread_bps,
        "entry_spread_bps": entry_spread_bps,
        "entry_spread_delta_bps": entry_spread_delta_bps,
        "expected_fill_slippage_bps": row.get("expected_fill_slippage_bps"),
        "orderbook_slippage_bps": row.get("orderbook_slippage_bps"),
        "expected_total_entry_friction_bps": row.get(
            "expected_total_entry_friction_bps"
        ),
        "estimated_hit_rate": row.get("estimated_hit_rate"),
        "estimated_ev_net_return": row.get("estimated_ev_net_return"),
        "ev_adjusted_net_return_after_friction": row.get(
            "ev_adjusted_net_return_after_friction"
        ),
        "ev_haircut_spread_excess_bps": row.get("ev_haircut_spread_excess_bps"),
        "ev_haircut_adverse_signal_gap_bps": row.get(
            "ev_haircut_adverse_signal_gap_bps"
        ),
        "calibrated_score": row.get("calibrated_score"),
        "rank_percentile": row.get("rank_percentile"),
        "effective_threshold": row.get("effective_threshold"),
        "position_size_after_liquidity": row.get("position_size_after_liquidity"),
        "stop_price": row.get("stop_price"),
        "requested_policy_stop": row.get("requested_policy_stop"),
        "final_placed_stop": row.get("final_placed_stop"),
    }


def _close_audit(row: dict[str, Any]) -> dict[str, Any]:
    expected_spread_bps = _to_float(row.get("expected_spread_bps"))
    actual_exit_spread_bps = _to_float(row.get("actual_exit_spread_bps"))
    exit_spread_delta_bps = _to_float(row.get("exit_vs_expected_spread_bps"))
    if not math.isfinite(exit_spread_delta_bps):
        exit_spread_delta_bps = (
            actual_exit_spread_bps - expected_spread_bps
            if math.isfinite(actual_exit_spread_bps)
            and math.isfinite(expected_spread_bps)
            else float("nan")
        )
    return {
        "timestamp": row.get("timestamp"),
        "exit_time": row.get("exit_time"),
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "status": row.get("status"),
        "strategy_id": row.get("strategy_id"),
        "exit_reason": row.get("exit_reason"),
        "exit_reason_detail": row.get("exit_reason_detail"),
        "close_execution_method": row.get("close_execution_method"),
        "close_execution_detail": row.get("close_execution_detail"),
        "close_price_source": row.get("close_price_source"),
        "close_trigger_type": row.get("close_trigger_type"),
        "close_trigger_reference": row.get("close_trigger_reference"),
        "close_touch_side": row.get("close_touch_side"),
        "expected_spread_bps": expected_spread_bps,
        "actual_exit_spread_bps": actual_exit_spread_bps,
        "exit_spread_delta_bps": exit_spread_delta_bps,
        "actual_exit_bid": row.get("actual_exit_bid"),
        "actual_exit_ask": row.get("actual_exit_ask"),
        "actual_exit_last": row.get("actual_exit_last"),
        "sentinel_executable_price": row.get("sentinel_executable_price"),
        "sentinel_executable_price_source": row.get(
            "sentinel_executable_price_source"
        ),
        "sentinel_stop_distance_bps": row.get("sentinel_stop_distance_bps"),
        "sentinel_stop_breach_overshoot_bps": row.get(
            "sentinel_stop_breach_overshoot_bps"
        ),
        "sentinel_pretrigger_enabled": row.get("sentinel_pretrigger_enabled"),
        "sentinel_pretriggered": row.get("sentinel_pretriggered"),
        "requested_policy_stop": row.get("requested_policy_stop"),
        "final_placed_stop": row.get("final_placed_stop"),
        "stop_price": row.get("stop_price"),
        "shadow_theoretical_exit_price": row.get("shadow_theoretical_exit_price"),
        "shadow_stop_trigger_price": row.get("shadow_stop_trigger_price"),
        "actual_exit_price": row.get("actual_exit_price"),
        "exit_vs_policy_stop_bps": row.get("exit_vs_policy_stop_bps"),
        "exit_vs_peak_giveback_pct": row.get("exit_vs_peak_giveback_pct"),
        "gross_pnl_pct": row.get("gross_pnl_pct"),
        "net_pnl_pct": row.get("net_pnl_pct"),
        "gross_pnl_amount": row.get("gross_pnl_amount"),
        "net_pnl_amount": row.get("net_pnl_amount"),
        "fees_verified": row.get("fees_verified"),
        "fees_estimated": row.get("fees_estimated"),
        "net_pnl_verification_status": row.get("net_pnl_verification_status"),
    }


def _trade_snapshot(
    path: Path,
    *,
    since: pd.Timestamp,
    local_timezone: str,
    spread_delta_warn_bps: float,
    stop_gap_warn_bps: float,
) -> dict[str, Any]:
    rows = _tail_csv_rows(path)
    if not rows:
        return {"path": str(path), "exists": path.exists(), "tail_rows": 0}
    recent: list[dict[str, Any]] = []
    lifecycle_counter: Counter[str] = Counter()
    for row in rows:
        ts = _parse_timestamp(
            row.get("timestamp") or row.get("entry_time"),
            local_timezone=local_timezone,
        )
        if ts is None:
            continue
        if ts < since:
            continue
        lifecycle = str(row.get("lifecycle_event") or row.get("status") or "")
        lifecycle_counter[lifecycle] += 1
        recent.append(row)

    interesting = []
    entry_audits = []
    close_audits = []
    latest_entry_audits = []
    latest_close_audits = []
    entry_alert_rows = []
    close_quality_alert_rows = []
    spread_alert_rows = []
    stop_gap_alert_rows = []
    latest_entries = [row for row in rows if _is_entry_row(row)][-10:]
    latest_closes = [row for row in rows if _is_exit_row(row)][-10:]
    latest_entry_audits = [_entry_audit(row) for row in latest_entries]
    latest_close_audits = [_close_audit(row) for row in latest_closes]
    for row in recent[-10:]:
        expected_spread_bps = _to_float(row.get("expected_spread_bps"))
        entry_spread_bps = _to_float(row.get("entry_spread_bps"))
        actual_exit_spread_bps = _to_float(row.get("actual_exit_spread_bps"))
        exit_vs_policy_stop_bps = _to_float(row.get("exit_vs_policy_stop_bps"))
        entry_spread_delta_bps = (
            entry_spread_bps - expected_spread_bps
            if math.isfinite(entry_spread_bps) and math.isfinite(expected_spread_bps)
            else float("nan")
        )
        exit_spread_delta_bps = (
            actual_exit_spread_bps - expected_spread_bps
            if math.isfinite(actual_exit_spread_bps)
            and math.isfinite(expected_spread_bps)
            else float("nan")
        )
        if (
            math.isfinite(entry_spread_delta_bps)
            and abs(entry_spread_delta_bps) > spread_delta_warn_bps
        ):
            alert = _entry_audit(row)
            spread_alert_rows.append(alert)
            entry_alert_rows.append(alert)
        if (
            math.isfinite(exit_spread_delta_bps)
            and abs(exit_spread_delta_bps) > spread_delta_warn_bps
        ):
            spread_alert_rows.append(_close_audit(row))
        if (
            math.isfinite(exit_vs_policy_stop_bps)
            and abs(exit_vs_policy_stop_bps) > stop_gap_warn_bps
        ):
            alert = _close_audit(row)
            stop_gap_alert_rows.append(alert)
            close_quality_alert_rows.append(alert)
        close_method = str(row.get("close_execution_method") or "").lower()
        if _is_exit_row(row) and close_method == "exchange_last_stop_loss":
            close_quality_alert_rows.append(_close_audit(row))
        if _is_entry_row(row):
            entry_audits.append(_entry_audit(row))
        if _is_exit_row(row):
            close_audits.append(_close_audit(row))
        interesting.append(
            {
                "timestamp": row.get("timestamp"),
                "lifecycle_event": row.get("lifecycle_event"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "status": row.get("status"),
                "exit_reason": row.get("exit_reason"),
                "actual_entry_price": row.get("actual_entry_price"),
                "actual_exit_price": row.get("actual_exit_price"),
                "expected_spread_bps": row.get("expected_spread_bps"),
                "entry_spread_bps": row.get("entry_spread_bps"),
                "actual_exit_spread_bps": row.get("actual_exit_spread_bps"),
                "entry_spread_delta_bps": entry_spread_delta_bps,
                "exit_spread_delta_bps": exit_spread_delta_bps,
                "exit_vs_policy_stop_bps": row.get("exit_vs_policy_stop_bps"),
                "gross_pnl_amount": row.get("gross_pnl_amount"),
                "net_pnl_amount": row.get("net_pnl_amount"),
                "net_pnl_estimated": row.get("net_pnl_estimated"),
                "fees_verified": row.get("fees_verified"),
                "fees_estimated": row.get("fees_estimated"),
                "net_pnl_verification_status": row.get("net_pnl_verification_status"),
                "close_execution_method": row.get("close_execution_method"),
                "close_price_source": row.get("close_price_source"),
            }
        )
    return {
        "path": str(path),
        "exists": path.exists(),
        "tail_rows": len(rows),
        "recent_rows_since_start": len(recent),
        "lifecycle_counts_since_start": dict(lifecycle_counter),
        "recent_trade_audit_rows": interesting,
        "entry_audit_rows_since_start": entry_audits[-10:],
        "close_audit_rows_since_start": close_audits[-10:],
        "latest_entry_audit_rows": latest_entry_audits,
        "latest_close_audit_rows": latest_close_audits,
        "entry_alert_rows": entry_alert_rows[-10:],
        "close_quality_alert_rows": close_quality_alert_rows[-10:],
        "spread_alert_rows": spread_alert_rows[-10:],
        "stop_gap_alert_rows": stop_gap_alert_rows[-10:],
    }


def _prediction_snapshot(
    path: Path | None,
    *,
    since: pd.Timestamp,
    max_rows: int,
) -> dict[str, Any]:
    if path is None:
        return {"enabled": False}
    if not path.exists():
        return {"enabled": True, "path": str(path), "exists": False}
    try:
        frame = _tail_parquet_frame(
            path,
            columns=PREDICTION_LEDGER_COLUMNS,
            max_rows=max_rows,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "path": str(path),
            "exists": True,
            "error": repr(exc),
        }
    if frame.empty:
        return {
            "enabled": True,
            "path": str(path),
            "exists": True,
            "tail_rows": 0,
        }

    time_col = _first_existing(frame.columns, ["decision_ts", "timestamp", "signal_bar_ts"])
    if time_col:
        parsed_ts = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
        recent = frame.loc[parsed_ts >= since].copy()
    else:
        parsed_ts = pd.Series(pd.NaT, index=frame.index)
        recent = frame.iloc[0:0].copy()
    used_since_start = not recent.empty
    analysis = recent if used_since_start else frame.copy()

    policy_score_col = _first_existing(
        analysis.columns,
        ["threshold_rank_score", "policy_rank_pct", "normalized_rank_score"],
    )
    policy_threshold_col = _first_existing(
        analysis.columns,
        [
            "dynamic_hr_surprise_threshold",
            "final_gate_threshold",
            "portfolio_gate_final_threshold",
        ],
    )
    final_score_col = _first_existing(
        analysis.columns,
        ["final_gate_rank_score", "portfolio_gate_rank_score"],
    )
    final_threshold_col = _first_existing(
        analysis.columns,
        ["final_gate_threshold", "portfolio_gate_final_threshold"],
    )
    decision_col = "portfolio_decision" if "portfolio_decision" in analysis.columns else None
    reject_col = (
        "portfolio_reject_reason" if "portfolio_reject_reason" in analysis.columns else None
    )
    traded = (
        _bool_series(analysis["was_traded"])
        if "was_traded" in analysis.columns
        else pd.Series(False, index=analysis.index)
    )
    if decision_col:
        traded = traded | analysis[decision_col].astype(str).str.lower().eq("traded")

    policy_pass = pd.Series(False, index=analysis.index)
    policy_gate_valid = pd.Series(False, index=analysis.index)
    if policy_score_col and policy_threshold_col:
        policy_score = pd.to_numeric(analysis[policy_score_col], errors="coerce")
        policy_threshold = pd.to_numeric(analysis[policy_threshold_col], errors="coerce")
        policy_gate_valid = policy_score.notna() & policy_threshold.notna()
        policy_pass = policy_score >= policy_threshold
    else:
        policy_score = pd.Series(float("nan"), index=analysis.index)
        policy_threshold = pd.Series(float("nan"), index=analysis.index)

    final_pass = pd.Series(False, index=analysis.index)
    final_gate_valid = pd.Series(False, index=analysis.index)
    if final_score_col and final_threshold_col:
        final_score = pd.to_numeric(analysis[final_score_col], errors="coerce")
        final_threshold = pd.to_numeric(analysis[final_threshold_col], errors="coerce")
        final_gate_valid = final_score.notna() & final_threshold.notna()
        final_pass = final_score >= final_threshold
    else:
        final_score = pd.Series(float("nan"), index=analysis.index)
        final_threshold = pd.Series(float("nan"), index=analysis.index)

    recorded_pass = (
        _bool_series(analysis["passed_rank_gate"])
        if "passed_rank_gate" in analysis.columns
        else pd.Series(False, index=analysis.index)
    )
    reject_text = (
        analysis[reject_col].fillna("").astype(str)
        if reject_col
        else pd.Series("", index=analysis.index)
    )
    rank_reject = reject_text.str.contains("rank_below_dynamic_threshold", na=False)
    post_rank_reject = reject_text.str.contains(
        "entry_cap_reached|remaining_total_notional|max_new_entries_per_bar_reached|symbol_entry_block|cooldown",
        na=False,
    )
    capacity_reject = reject_text.str.contains(
        "entry_cap_reached|remaining_total_notional|max_new_entries_per_bar_reached",
        na=False,
    )

    policy_gate_mismatch = policy_gate_valid & ~post_rank_reject & (
        recorded_pass != policy_pass
    )
    final_vs_policy_mismatch = (
        policy_gate_valid & final_gate_valid & ~post_rank_reject & (policy_pass != final_pass)
    )
    traded_below_policy_threshold = traded & policy_gate_valid & ~policy_pass
    rank_rejected_but_policy_pass = rank_reject & policy_gate_valid & policy_pass
    capacity_rejected_policy_pass = capacity_reject & policy_gate_valid & policy_pass

    def _examples(mask: pd.Series, limit: int = 8) -> list[dict[str, Any]]:
        cols = [
            column
            for column in [
                time_col,
                "symbol",
                "side",
                "strategy_id",
                policy_score_col,
                policy_threshold_col,
                final_score_col,
                final_threshold_col,
                "passed_rank_gate",
                decision_col,
                reject_col,
                "dynamic_hr_surprise_z_eff",
                "estimated_ev_net_return",
                "ev_adjusted_net_return_after_friction",
            ]
            if column and column in analysis.columns
        ]
        out = analysis.loc[mask, cols].tail(limit).copy()
        return out.astype(object).where(pd.notna(out), None).to_dict("records")

    by_head: dict[str, Any] = {}
    head_col = (
        "dynamic_hr_surprise_head"
        if "dynamic_hr_surprise_head" in analysis.columns
        else ("strategy_id" if "strategy_id" in analysis.columns else None)
    )
    if head_col:
        group_frame = pd.DataFrame(
            {
                "head": analysis[head_col].astype(str),
                "policy_gate_valid": policy_gate_valid,
                "policy_pass": policy_pass,
                "traded": traded,
                "rank_reject": rank_reject,
                "capacity_reject": capacity_reject,
            },
            index=analysis.index,
        )
        for head, group in group_frame.groupby("head", dropna=False):
            valid = group["policy_gate_valid"]
            by_head[str(head)] = {
                "rows": int(len(group)),
                "policy_gate_valid_rows": int(valid.sum()),
                "policy_pass": int((group["policy_pass"] & valid).sum()),
                "traded": int(group["traded"].sum()),
                "rank_reject": int(group["rank_reject"].sum()),
                "capacity_reject": int(group["capacity_reject"].sum()),
            }

    rank_margin = policy_score - policy_threshold
    return {
        "enabled": True,
        "path": str(path),
        "exists": True,
        "tail_rows": int(len(frame)),
        "latest_timestamp": (
            parsed_ts.dropna().max().isoformat() if parsed_ts.notna().any() else None
        ),
        "rows_since_start": int(len(recent)),
        "analysis_rows": int(len(analysis)),
        "analysis_uses_since_start": used_since_start,
        "policy_score_col": policy_score_col,
        "policy_threshold_col": policy_threshold_col,
        "final_score_col": final_score_col,
        "final_threshold_col": final_threshold_col,
        "decision_counts": (
            analysis[decision_col].fillna("None").astype(str).value_counts().head(20).to_dict()
            if decision_col
            else {}
        ),
        "reject_reason_counts": (
            analysis[reject_col].fillna("None").astype(str).value_counts().head(20).to_dict()
            if reject_col
            else {}
        ),
        "policy_gate_valid_rows": int(policy_gate_valid.sum()),
        "policy_pass_count": int((policy_pass & policy_gate_valid).sum()),
        "recorded_pass_count": int(recorded_pass.sum()),
        "final_gate_valid_rows": int(final_gate_valid.sum()),
        "final_pass_count": int((final_pass & final_gate_valid).sum()),
        "traded_count": int(traded.sum()),
        "rank_reject_count": int(rank_reject.sum()),
        "capacity_reject_count": int(capacity_reject.sum()),
        "policy_gate_mismatch_count": int(policy_gate_mismatch.sum()),
        "final_vs_policy_mismatch_count": int(final_vs_policy_mismatch.sum()),
        "traded_below_policy_threshold_count": int(traded_below_policy_threshold.sum()),
        "rank_rejected_but_policy_pass_count": int(rank_rejected_but_policy_pass.sum()),
        "capacity_rejected_policy_pass_count": int(capacity_rejected_policy_pass.sum()),
        "rank_margin_stats": _series_stats(rank_margin[policy_gate_valid]),
        "dynamic_hr_z_eff_stats": (
            _series_stats(analysis["dynamic_hr_surprise_z_eff"])
            if "dynamic_hr_surprise_z_eff" in analysis.columns
            else {"n": 0}
        ),
        "dynamic_hr_threshold_stats": (
            _series_stats(analysis["dynamic_hr_surprise_threshold"])
            if "dynamic_hr_surprise_threshold" in analysis.columns
            else {"n": 0}
        ),
        "spread_bps_stats": (
            _series_stats(analysis["spread_bps"])
            if "spread_bps" in analysis.columns
            else (
                _series_stats(analysis["ticker_spread_bps"])
                if "ticker_spread_bps" in analysis.columns
                else {"n": 0}
            )
        ),
        "by_head": by_head,
        "examples": {
            "policy_gate_mismatch": _examples(policy_gate_mismatch),
            "final_vs_policy_mismatch": _examples(final_vs_policy_mismatch),
            "traded_below_policy_threshold": _examples(traded_below_policy_threshold),
            "rank_rejected_but_policy_pass": _examples(rank_rejected_but_policy_pass),
            "capacity_rejected_policy_pass": _examples(capacity_rejected_policy_pass),
        },
    }


def _warnings(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    proc = payload.get("process") or {}
    log = payload.get("log") or {}
    trades = payload.get("trades") or {}
    predictions = payload.get("predictions") or {}
    rss_mb = _to_float(proc.get("rss_mb"))
    feature_hotspot = (log.get("feature_generation_hotspots") or [{}])[0]
    model_hotspot = (log.get("model_prediction_hotspots") or [{}])[0]
    timing_rss_max = _to_float((log.get("timing_rss_stats") or {}).get("max"))
    if not proc.get("alive"):
        warnings.append("live_process_not_alive")
    if math.isfinite(rss_mb) and rss_mb >= float(args.rss_warn_mb):
        warnings.append(f"process_rss_above_{args.rss_warn_mb:.0f}mb")
    if math.isfinite(timing_rss_max) and timing_rss_max >= float(args.rss_warn_mb):
        warnings.append(f"timing_rss_above_{args.rss_warn_mb:.0f}mb")
    if _to_float(feature_hotspot.get("stage_seconds")) >= float(
        args.feature_stage_warn_seconds
    ):
        warnings.append("feature_generation_hotspot_slow")
    if _to_float(model_hotspot.get("stage_seconds")) >= float(
        args.model_stage_warn_seconds
    ):
        warnings.append("model_prediction_hotspot_slow")
    if log.get("recent_error_lines"):
        warnings.append("recent_error_lines_present")
    if trades.get("entry_alert_rows"):
        warnings.append("entry_execution_quality_alert")
    if trades.get("close_quality_alert_rows"):
        warnings.append("close_execution_quality_alert")
    if trades.get("spread_alert_rows"):
        warnings.append("trade_spread_delta_alert")
    if trades.get("stop_gap_alert_rows"):
        warnings.append("close_stop_gap_alert")
    if predictions.get("enabled"):
        if predictions.get("policy_gate_mismatch_count", 0):
            warnings.append("prediction_policy_gate_mismatch")
        if predictions.get("rank_rejected_but_policy_pass_count", 0):
            warnings.append("rank_rejected_but_policy_threshold_passed")
        if predictions.get("traded_below_policy_threshold_count", 0):
            warnings.append("traded_below_policy_threshold")
    return warnings


def _sample(args: argparse.Namespace, *, started_at: pd.Timestamp) -> dict[str, Any]:
    payload = {
        "sample_ts": _utc_now().isoformat(),
        "pid": args.pid,
        "process": _ps_snapshot(args.pid),
        "log": _log_snapshot(
            Path(args.log),
            since=started_at,
            local_timezone=str(args.local_timezone),
        ),
        "trades": _trade_snapshot(
            Path(args.trade_log),
            since=started_at,
            local_timezone=str(args.local_timezone),
            spread_delta_warn_bps=float(args.spread_delta_warn_bps),
            stop_gap_warn_bps=float(args.stop_gap_warn_bps),
        ),
        "predictions": _prediction_snapshot(
            Path(args.prediction_ledger) if args.prediction_ledger else None,
            since=started_at,
            max_rows=int(args.max_prediction_rows),
        ),
    }
    payload["warnings"] = _warnings(payload, args)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trade-log", type=Path, default=Path("inference_trades.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-ledger", type=Path)
    parser.add_argument("--max-prediction-rows", type=int, default=5000)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--local-timezone", default="Europe/Paris")
    parser.add_argument(
        "--since",
        help="Use this UTC session boundary instead of the monitor process start.",
    )
    parser.add_argument("--rss-warn-mb", type=float, default=8000.0)
    parser.add_argument("--feature-stage-warn-seconds", type=float, default=30.0)
    parser.add_argument("--model-stage-warn-seconds", type=float, default=10.0)
    parser.add_argument("--spread-delta-warn-bps", type=float, default=25.0)
    parser.add_argument("--stop-gap-warn-bps", type=float, default=25.0)
    args = parser.parse_args()

    started_at = pd.Timestamp(_utc_now())
    if args.since:
        parsed_since = pd.to_datetime(args.since, utc=True, errors="coerce")
        if pd.isna(parsed_since):
            parser.error(f"invalid --since timestamp: {args.since!r}")
        started_at = parsed_since
    deadline = time.monotonic() + max(1, int(args.duration_seconds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    iteration = 0
    while True:
        iteration += 1
        payload = _sample(args, started_at=started_at)
        payload["iteration"] = iteration
        samples.append(payload)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
        proc = payload.get("process", {})
        log = payload.get("log", {})
        trades = payload.get("trades", {})
        predictions = payload.get("predictions", {})
        print(
            json.dumps(
                _json_safe(
                    {
                        "iteration": iteration,
                        "sample_ts": payload["sample_ts"],
                        "alive": proc.get("alive"),
                        "rss_mb": proc.get("rss_mb"),
                        "pcpu": proc.get("pcpu"),
                        "pmem": proc.get("pmem"),
                        "active_positions": (
                            log.get("latest_inference_monitor", {}).get(
                                "active_positions"
                            )
                        ),
                        "sentinel_active_positions": (
                            log.get("latest_executable_stop_sentinel", {}).get(
                                "active_positions"
                            )
                        ),
                        "recent_trade_rows": trades.get("recent_rows_since_start"),
                        "trade_spread_alerts": len(
                            trades.get("spread_alert_rows") or []
                        ),
                        "trade_stop_gap_alerts": len(
                            trades.get("stop_gap_alert_rows") or []
                        ),
                        "entry_quality_alerts": len(
                            trades.get("entry_alert_rows") or []
                        ),
                        "close_quality_alerts": len(
                            trades.get("close_quality_alert_rows") or []
                        ),
                        "recent_errors": len(log.get("recent_error_lines") or []),
                        "top_hotspot": (log.get("top_timing_hotspots") or [{}])[0],
                        "feature_hotspot": (
                            log.get("feature_generation_hotspots") or [{}]
                        )[0],
                        "model_hotspot": (
                            log.get("model_prediction_hotspots") or [{}]
                        )[0],
                        "timing_rss_stats": log.get("timing_rss_stats"),
                        "latest_entry_audit": (
                            trades.get("latest_entry_audit_rows") or [{}]
                        )[-1],
                        "latest_close_audit": (
                            trades.get("latest_close_audit_rows") or [{}]
                        )[-1],
                        "reject_counts_tail": log.get("reject_reason_counts_tail"),
                        "prediction_rows": predictions.get("analysis_rows"),
                        "prediction_policy_score_col": predictions.get("policy_score_col"),
                        "prediction_policy_threshold_col": predictions.get(
                            "policy_threshold_col"
                        ),
                        "prediction_policy_pass": predictions.get("policy_pass_count"),
                        "prediction_rank_reject": predictions.get("rank_reject_count"),
                        "prediction_capacity_reject": predictions.get(
                            "capacity_reject_count"
                        ),
                        "prediction_policy_gate_mismatch": predictions.get(
                            "policy_gate_mismatch_count"
                        ),
                        "prediction_rank_rejected_but_policy_pass": predictions.get(
                            "rank_rejected_but_policy_pass_count"
                        ),
                        "warnings": payload.get("warnings"),
                    }
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() >= deadline or not proc.get("alive", False):
            break
        time.sleep(min(max(1, int(args.interval_seconds)), max(0.0, deadline - time.monotonic())))

    max_rss = max(
        [
            _to_float(sample.get("process", {}).get("rss_mb"))
            for sample in samples
            if sample.get("process", {}).get("alive")
        ]
        or [float("nan")]
    )
    summary = {
        "summary_ts": _utc_now().isoformat(),
        "samples": len(samples),
        "max_process_rss_mb": max_rss,
        "last": samples[-1] if samples else {},
    }
    with args.out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe({"summary": summary}), sort_keys=True) + "\n")
    print(json.dumps(_json_safe({"summary": summary}), sort_keys=True), flush=True)
    return 0 if samples and samples[-1].get("process", {}).get("alive") else 2


if __name__ == "__main__":
    raise SystemExit(main())
