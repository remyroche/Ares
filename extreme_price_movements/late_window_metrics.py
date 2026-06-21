"""Rolling late-window hit-rate surprise diagnostics.

The helper intentionally stays independent from the exploratory performance
regime scripts so train_base, train_meta, and simple_policy_optimiser can share
the same compact metric contract.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_WINDOW_DAYS: tuple[int, int] = (3, 5)
DEFAULT_LATE_DAYS = 56
DEFAULT_MIN_ROWS_PER_DAY = 10
DEFAULT_BAD_SURPRISE_Z_THRESHOLD = -1.5
DEFAULT_BAD_HIT_RATE_DELTA_THRESHOLD = -0.0175


def _json_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _to_day_series(timestamps: Any) -> pd.Series:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    return ts.dt.floor("D")


def _covered_days(rows: pd.DataFrame) -> set[pd.Timestamp]:
    if rows.empty:
        return set()
    covered: set[pd.Timestamp] = set()
    for start, end in zip(rows["start_day"], rows["end_day"]):
        if pd.isna(start) or pd.isna(end):
            continue
        start_ts = pd.Timestamp(start).floor("D")
        end_ts = pd.Timestamp(end).floor("D")
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        for day in pd.date_range(start_ts, end_ts, freq="D", tz="UTC"):
            covered.add(pd.Timestamp(day).floor("D"))
    return covered


def _window_severity(
    row: pd.Series,
    *,
    min_rows: float,
    surprise_z_threshold: float,
    hit_rate_delta_threshold: float,
) -> dict[str, float]:
    z_threshold = max(abs(float(surprise_z_threshold)), 1e-6)
    delta_threshold = max(abs(float(hit_rate_delta_threshold)), 1e-6)
    window_days = max(_json_float(row.get("window_days"), 1.0), 1.0)
    rows = max(_json_float(row.get("n"), 0.0), 1.0)
    z_depth = max(0.0, -_json_float(row.get("hit_rate_surprise_z"), 0.0) / z_threshold)
    delta_depth = max(0.0, -_json_float(row.get("hit_rate_delta"), 0.0) / delta_threshold)
    row_component = float(np.clip(math.sqrt(rows / max(min_rows, 1.0)), 1.0, 4.0))
    duration_component = float(np.clip(math.sqrt(window_days), 1.0, 3.0))
    depth_component = float(np.clip(0.65 * z_depth + 0.35 * delta_depth, 0.1, 8.0))
    return {
        "z_depth": float(z_depth),
        "hit_rate_delta_depth": float(delta_depth),
        "row_support_ratio": float(rows / max(min_rows, 1.0)),
        "severity": float(duration_component * row_component * depth_component),
    }


def _daily_rolling_frame(
    *,
    timestamps: Any,
    actual_hit: Any,
    expected_probability: Any,
    pnl: Any | None,
    window_days: int,
) -> pd.DataFrame:
    days = _to_day_series(timestamps)
    actual = pd.to_numeric(pd.Series(actual_hit), errors="coerce")
    expected = pd.to_numeric(pd.Series(expected_probability), errors="coerce").clip(
        1e-5,
        1.0 - 1e-5,
    )
    pnl_s = (
        pd.to_numeric(pd.Series(pnl), errors="coerce")
        if pnl is not None
        else pd.Series(np.nan, index=actual.index)
    )
    valid = days.notna() & actual.notna() & expected.notna()
    if not bool(valid.any()):
        return pd.DataFrame()

    work = pd.DataFrame(
        {
            "day": days.loc[valid],
            "actual": actual.loc[valid].astype(float).clip(0.0, 1.0),
            "expected": expected.loc[valid].astype(float),
            "pnl": pnl_s.loc[valid].astype(float),
        }
    )
    if work.empty:
        return pd.DataFrame()
    p = work["expected"].to_numpy(dtype=np.float64, copy=False)
    work["variance"] = np.clip(p * (1.0 - p), 1e-6, np.inf)
    daily = work.groupby("day", sort=True).agg(
        n=("actual", "size"),
        hits=("actual", "sum"),
        expected_hits=("expected", "sum"),
        variance=("variance", "sum"),
        pnl_sum=("pnl", "sum"),
    )
    if daily.empty:
        return pd.DataFrame()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(full_index)
    for col in ("n", "hits", "expected_hits", "variance", "pnl_sum"):
        daily[col] = daily[col].fillna(0.0)

    win = max(1, int(window_days))
    rolling = pd.DataFrame(index=daily.index)
    for col in ("n", "hits", "expected_hits", "variance", "pnl_sum"):
        rolling[col] = daily[col].rolling(win, min_periods=win).sum()
    rolling["start_day"] = rolling.index - pd.Timedelta(days=win - 1)
    rolling["end_day"] = rolling.index
    rolling["actual_hit_rate"] = rolling["hits"] / rolling["n"].replace(0.0, np.nan)
    rolling["expected_hit_rate"] = rolling["expected_hits"] / rolling["n"].replace(
        0.0,
        np.nan,
    )
    rolling["hit_rate_delta"] = (
        rolling["actual_hit_rate"] - rolling["expected_hit_rate"]
    )
    rolling["hit_rate_surprise"] = rolling["hits"] - rolling["expected_hits"]
    rolling["hit_rate_surprise_z"] = rolling["hit_rate_surprise"] / np.sqrt(
        rolling["variance"].clip(lower=1e-6)
    )
    rolling["mean_pnl"] = rolling["pnl_sum"] / rolling["n"].replace(0.0, np.nan)
    rolling = rolling.reset_index(names="window_end_day")
    rolling["window_days"] = win
    return rolling


def _clean_worst_row(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    out: dict[str, Any] = {}
    for key in (
        "start_day",
        "end_day",
        "window_days",
        "n",
        "actual_hit_rate",
        "expected_hit_rate",
        "hit_rate_delta",
        "hit_rate_surprise_z",
        "mean_pnl",
        "window_severity",
    ):
        if key not in row:
            continue
        val = row[key]
        if key.endswith("_day"):
            out[key] = pd.Timestamp(val).date().isoformat() if pd.notna(val) else None
        elif key in {"window_days", "n"}:
            out[key] = int(val) if np.isfinite(float(val)) else 0
        else:
            out[key] = _json_float(val)
    return out


def compute_late_window_hit_rate_summary(
    *,
    timestamps: Any,
    actual_hit: Any,
    expected_probability: Any,
    pnl: Any | None = None,
    window_days: Iterable[int] = DEFAULT_WINDOW_DAYS,
    late_days: int = DEFAULT_LATE_DAYS,
    min_rows_per_day: float = DEFAULT_MIN_ROWS_PER_DAY,
    min_rows: float = 0.0,
    bad_surprise_z_threshold: float = DEFAULT_BAD_SURPRISE_Z_THRESHOLD,
    bad_hit_rate_delta_threshold: float = DEFAULT_BAD_HIT_RATE_DELTA_THRESHOLD,
) -> dict[str, Any]:
    """Return compact rolling-window hit-rate surprise diagnostics.

    ``actual_hit`` should be a binary TP/SL outcome and ``expected_probability``
    should be the model/policy probability used for ranking.  For policy rows,
    callers generally pass ``net_gain > 0`` after the TP/SL simulator.
    """

    windows: dict[str, Any] = {}
    day_series = _to_day_series(timestamps)
    valid_days = day_series.loc[day_series.notna()]
    if valid_days.empty:
        return {
            "status": "no_valid_timestamps",
            "definition": "actual_binary_tp_sl_hit_minus_expected_probability",
            "late_days": int(late_days),
            "windows": windows,
        }

    max_day = pd.Timestamp(valid_days.max()).floor("D")
    late_start = max_day - pd.Timedelta(days=max(1, int(late_days)) - 1)
    eligible_late_days = {
        pd.Timestamp(day).floor("D")
        for day in pd.date_range(late_start, max_day, freq="D", tz="UTC")
    }
    for win in tuple(dict.fromkeys(int(w) for w in window_days if int(w) > 0)):
        rolling = _daily_rolling_frame(
            timestamps=timestamps,
            actual_hit=actual_hit,
            expected_probability=expected_probability,
            pnl=pnl,
            window_days=win,
        )
        if rolling.empty:
            windows[f"{win}d"] = {"status": "no_windows", "window_days": int(win)}
            continue
        effective_min_rows = max(float(min_rows), float(min_rows_per_day) * win, 1.0)
        eligible = (
            pd.to_numeric(rolling["n"], errors="coerce").ge(effective_min_rows)
            & pd.to_numeric(rolling["hit_rate_delta"], errors="coerce").notna()
            & pd.to_numeric(rolling["hit_rate_surprise_z"], errors="coerce").notna()
        )
        end_day = pd.to_datetime(rolling["end_day"], utc=True, errors="coerce")
        late_mask = eligible & end_day.ge(late_start)
        bad = (
            late_mask
            & pd.to_numeric(rolling["hit_rate_surprise_z"], errors="coerce").le(
                float(bad_surprise_z_threshold)
            )
            & pd.to_numeric(rolling["hit_rate_delta"], errors="coerce").le(
                float(bad_hit_rate_delta_threshold)
            )
        )
        scored = rolling.copy()
        severities = [
            _window_severity(
                row,
                min_rows=effective_min_rows,
                surprise_z_threshold=float(bad_surprise_z_threshold),
                hit_rate_delta_threshold=float(bad_hit_rate_delta_threshold),
            )
            for _, row in scored.iterrows()
        ]
        sev_df = pd.DataFrame(severities)
        scored["window_z_depth"] = sev_df["z_depth"].to_numpy(dtype=np.float64)
        scored["window_hit_rate_delta_depth"] = sev_df[
            "hit_rate_delta_depth"
        ].to_numpy(dtype=np.float64)
        scored["window_row_support_ratio"] = sev_df["row_support_ratio"].to_numpy(
            dtype=np.float64
        )
        scored["window_severity"] = sev_df["severity"].to_numpy(dtype=np.float64)

        late_rows = scored.loc[late_mask].copy()
        bad_rows = scored.loc[bad].copy()
        worst_row = None
        if not late_rows.empty:
            worst_order = late_rows.sort_values(
                ["hit_rate_surprise_z", "hit_rate_delta"],
                ascending=[True, True],
                kind="mergesort",
            )
            worst_row = worst_order.iloc[0]

        covered_bad_days = _covered_days(bad_rows) & eligible_late_days
        result = {
            "status": "ok" if not late_rows.empty else "no_late_eligible_windows",
            "window_days": int(win),
            "late_start_day": late_start.date().isoformat(),
            "late_end_day": max_day.date().isoformat(),
            "min_rows": float(effective_min_rows),
            "bad_surprise_z_threshold": float(bad_surprise_z_threshold),
            "bad_hit_rate_delta_threshold": float(bad_hit_rate_delta_threshold),
            "eligible_window_count": int(len(late_rows)),
            "bad_window_count": int(len(bad_rows)),
            "bad_window_share": float(len(bad_rows) / max(len(late_rows), 1)),
            "bad_day_share": float(len(covered_bad_days) / max(len(eligible_late_days), 1)),
            "worst": _clean_worst_row(worst_row),
        }
        for col, key_prefix in (
            ("hit_rate_delta", "bad_hit_rate_delta"),
            ("hit_rate_surprise_z", "bad_hit_rate_surprise_z"),
            ("window_severity", "bad_window_severity"),
        ):
            vals = pd.to_numeric(bad_rows.get(col, pd.Series(dtype=float)), errors="coerce")
            vals = vals.loc[vals.notna()]
            result[f"{key_prefix}_mean"] = float(vals.mean()) if len(vals) else 0.0
            result[f"{key_prefix}_p10"] = float(vals.quantile(0.10)) if len(vals) else 0.0
            result[f"{key_prefix}_p50"] = float(vals.quantile(0.50)) if len(vals) else 0.0
            result[f"{key_prefix}_max"] = float(vals.max()) if len(vals) else 0.0
        windows[f"{win}d"] = result

    return {
        "status": "ok",
        "definition": "actual_binary_tp_sl_hit_minus_expected_probability",
        "late_days": int(late_days),
        "windows": windows,
    }


def compute_late_window_hit_rate_summary_from_frame(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    actual_col: str | None = None,
    expected_col: str | None = None,
    pnl_col: str | None = None,
    actual_values: Any | None = None,
    expected_values: Any | None = None,
    pnl_values: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if frame.empty or timestamp_col not in frame.columns:
        return {
            "status": "missing_timestamp_or_empty_frame",
            "definition": "actual_binary_tp_sl_hit_minus_expected_probability",
            "late_days": int(kwargs.get("late_days", DEFAULT_LATE_DAYS)),
            "windows": {},
        }
    if actual_values is None:
        if not actual_col or actual_col not in frame.columns:
            return {
                "status": "missing_actual",
                "definition": "actual_binary_tp_sl_hit_minus_expected_probability",
                "late_days": int(kwargs.get("late_days", DEFAULT_LATE_DAYS)),
                "windows": {},
            }
        actual_values = frame[actual_col]
    if expected_values is None:
        if not expected_col or expected_col not in frame.columns:
            return {
                "status": "missing_expected",
                "definition": "actual_binary_tp_sl_hit_minus_expected_probability",
                "late_days": int(kwargs.get("late_days", DEFAULT_LATE_DAYS)),
                "windows": {},
            }
        expected_values = frame[expected_col]
    if pnl_values is None and pnl_col and pnl_col in frame.columns:
        pnl_values = frame[pnl_col]
    return compute_late_window_hit_rate_summary(
        timestamps=frame[timestamp_col],
        actual_hit=actual_values,
        expected_probability=expected_values,
        pnl=pnl_values,
        **kwargs,
    )


def flatten_late_window_summary(
    summary: Mapping[str, Any],
    *,
    prefix: str = "late_window",
) -> dict[str, Any]:
    """Flatten the compact summary for metric tables that prefer scalar fields."""

    out: dict[str, Any] = {
        f"{prefix}_status": summary.get("status"),
        f"{prefix}_late_days": summary.get("late_days"),
    }
    windows = summary.get("windows", {})
    if not isinstance(windows, Mapping):
        return out
    for label, metrics in windows.items():
        if not isinstance(metrics, Mapping):
            continue
        key_prefix = f"{prefix}_{label}"
        for key in (
            "status",
            "eligible_window_count",
            "bad_window_count",
            "bad_window_share",
            "bad_day_share",
            "bad_hit_rate_delta_mean",
            "bad_hit_rate_delta_p10",
            "bad_hit_rate_surprise_z_mean",
            "bad_hit_rate_surprise_z_p10",
            "bad_window_severity_mean",
            "bad_window_severity_max",
        ):
            if key in metrics:
                out[f"{key_prefix}_{key}"] = metrics[key]
        worst = metrics.get("worst", {})
        if isinstance(worst, Mapping):
            for key in (
                "start_day",
                "end_day",
                "n",
                "actual_hit_rate",
                "expected_hit_rate",
                "hit_rate_delta",
                "hit_rate_surprise_z",
                "mean_pnl",
                "window_severity",
            ):
                if key in worst:
                    out[f"{key_prefix}_worst_{key}"] = worst[key]
    return out
