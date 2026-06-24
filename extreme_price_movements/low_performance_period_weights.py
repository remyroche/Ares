from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_value(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping) and key in cfg:
        return cfg.get(key)
    return default


def _period_weight_path(cfg: Mapping[str, Any] | None = None) -> str:
    raw = (
        os.getenv("EPM_LOW_PERFORMANCE_PERIOD_WEIGHTS_PATH", "")
        or str(_cfg_value(cfg, "low_performance_period_weights_path", "") or "")
    )
    return str(raw).strip()


def low_performance_period_weights_enabled(cfg: Mapping[str, Any] | None = None) -> bool:
    raw_enabled = os.getenv("EPM_LOW_PERFORMANCE_PERIOD_WEIGHTS_ENABLED", "")
    if raw_enabled:
        return _truthy(raw_enabled)
    cfg_enabled = _cfg_value(cfg, "low_performance_period_weights_enabled", None)
    if cfg_enabled is not None:
        return bool(cfg_enabled)
    return bool(_period_weight_path(cfg))


@lru_cache(maxsize=16)
def _load_period_weight_frame_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = payload.get("periods") or payload.get("weights") or payload.get("rows") or []
        else:
            rows = payload
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported low-performance period weight file: {path}")
    if df.empty:
        return df
    df = df.copy()
    for col in ("start_ts", "end_ts", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "weight_multiplier" not in df.columns:
        if "sample_weight_multiplier" in df.columns:
            df["weight_multiplier"] = df["sample_weight_multiplier"]
        elif "weight" in df.columns:
            df["weight_multiplier"] = df["weight"]
        else:
            df["weight_multiplier"] = 1.0
    df["weight_multiplier"] = pd.to_numeric(df["weight_multiplier"], errors="coerce").fillna(1.0)
    if "badness" in df.columns:
        df["badness"] = pd.to_numeric(df["badness"], errors="coerce")
    return df.reset_index(drop=True)


def load_period_weight_frame(cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    path_str = _period_weight_path(cfg)
    if not path_str:
        return pd.DataFrame()
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return _load_period_weight_frame_cached(str(path), int(path.stat().st_mtime_ns)).copy()


def _filter_scope(df: pd.DataFrame) -> pd.DataFrame:
    head = os.getenv("EPM_LOW_PERFORMANCE_PERIOD_HEAD", "").strip()
    strategy_id = os.getenv("EPM_LOW_PERFORMANCE_PERIOD_STRATEGY_ID", "").strip()
    out = df
    if head and "head" in out.columns:
        out = out[out["head"].astype(str).eq(head)]
    if strategy_id and "strategy_id" in out.columns:
        scoped = out[out["strategy_id"].astype(str).eq(strategy_id)]
        if not scoped.empty:
            out = scoped
    return out


def low_performance_period_multiplier(
    timestamps: Any,
    cfg: Mapping[str, Any] | None = None,
    *,
    objective_mode: str | None = None,
    label: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    n = len(ts)
    diag: dict[str, Any] = {
        "low_performance_period_weight_enabled": False,
        "low_performance_period_weight_rows": 0,
        "low_performance_period_weight_matched_rows": 0,
        "low_performance_period_weight_path": _period_weight_path(cfg),
        "low_performance_period_weight_objective_mode": str(objective_mode or ""),
        "low_performance_period_weight_label": str(label or ""),
    }
    if n == 0 or not low_performance_period_weights_enabled(cfg):
        return np.ones(n, dtype=np.float32), diag

    df = load_period_weight_frame(cfg)
    if df.empty:
        diag["low_performance_period_weight_reason"] = "empty_or_missing_manifest"
        return np.ones(n, dtype=np.float32), diag
    df = _filter_scope(df)
    if df.empty:
        diag["low_performance_period_weight_reason"] = "empty_after_scope_filter"
        return np.ones(n, dtype=np.float32), diag

    lo = float(os.getenv("EPM_LOW_PERFORMANCE_PERIOD_WEIGHT_MIN", _cfg_value(cfg, "low_performance_period_weight_min", 0.2)) or 0.2)
    hi = float(os.getenv("EPM_LOW_PERFORMANCE_PERIOD_WEIGHT_MAX", _cfg_value(cfg, "low_performance_period_weight_max", 8.0)) or 8.0)
    hi = max(1.0, hi)
    lo = float(np.clip(lo, 0.0, hi))

    mult = np.ones(n, dtype=np.float64)
    ts_ns = ts.view("int64")
    valid_ts = ~pd.isna(ts)
    for row in df.itertuples(index=False):
        weight = float(getattr(row, "weight_multiplier", 1.0) or 1.0)
        if not np.isfinite(weight):
            continue
        weight = float(np.clip(weight, lo, hi))
        start = getattr(row, "start_ts", pd.NaT)
        end = getattr(row, "end_ts", pd.NaT)
        point = getattr(row, "timestamp", pd.NaT)
        if pd.isna(start) and not pd.isna(point):
            start = point
        if pd.isna(end) and not pd.isna(start):
            end = pd.Timestamp(start) + pd.Timedelta(hours=1)
        if pd.isna(start) or pd.isna(end):
            continue
        start_ns = pd.Timestamp(start).value
        end_ns = pd.Timestamp(end).value
        mask = valid_ts & (ts_ns >= start_ns) & (ts_ns < end_ns)
        if bool(np.any(mask)):
            mult[mask] = np.maximum(mult[mask], weight)

    matched = int(np.sum(mult > 1.000001))
    diag.update(
        {
            "low_performance_period_weight_enabled": True,
            "low_performance_period_weight_rows": int(len(df)),
            "low_performance_period_weight_matched_rows": matched,
            "low_performance_period_weight_matched_fraction": float(matched / max(n, 1)),
            "low_performance_period_weight_min": float(np.min(mult)) if n else float("nan"),
            "low_performance_period_weight_mean_raw": float(np.mean(mult)) if n else float("nan"),
            "low_performance_period_weight_p90_raw": float(np.percentile(mult, 90.0)) if n else float("nan"),
            "low_performance_period_weight_max_raw": float(np.max(mult)) if n else float("nan"),
        }
    )
    return mult.astype(np.float32), diag


def apply_low_performance_period_weights(
    base_weight: Any,
    timestamps: Any,
    cfg: Mapping[str, Any] | None = None,
    *,
    objective_mode: str | None = None,
    label: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    w = np.asarray(base_weight, dtype=np.float32)
    mult, diag = low_performance_period_multiplier(
        timestamps,
        cfg,
        objective_mode=objective_mode,
        label=label,
    )
    if len(mult) != len(w):
        diag["low_performance_period_weight_enabled"] = False
        diag["low_performance_period_weight_reason"] = "length_mismatch"
        return w, diag
    if not bool(diag.get("low_performance_period_weight_enabled", False)):
        return w, diag
    out = np.asarray(w, dtype=np.float64) * np.asarray(mult, dtype=np.float64)
    out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
    mean = float(np.mean(out)) if len(out) else 1.0
    if not np.isfinite(mean) or mean <= 1e-12:
        out = np.ones(len(w), dtype=np.float64)
        mean = 1.0
    out = out / mean
    lo = float(os.getenv("EPM_LOW_PERFORMANCE_PERIOD_FINAL_WEIGHT_MIN", _cfg_value(cfg, "low_performance_period_final_weight_min", 0.05)) or 0.05)
    hi = float(os.getenv("EPM_LOW_PERFORMANCE_PERIOD_FINAL_WEIGHT_MAX", _cfg_value(cfg, "low_performance_period_final_weight_max", 20.0)) or 20.0)
    out = np.clip(out, lo, hi)
    out = out / max(float(np.mean(out)), 1e-12)
    ess = float((out.sum() ** 2) / max(float(np.sum(out**2)), 1e-12)) if len(out) else 0.0
    diag.update(
        {
            "low_performance_period_weight_applied": True,
            "low_performance_period_final_weight_min": float(np.min(out)) if len(out) else float("nan"),
            "low_performance_period_final_weight_mean": float(np.mean(out)) if len(out) else float("nan"),
            "low_performance_period_final_weight_p90": float(np.percentile(out, 90.0)) if len(out) else float("nan"),
            "low_performance_period_final_weight_max": float(np.max(out)) if len(out) else float("nan"),
            "low_performance_period_final_weight_ess": ess,
            "low_performance_period_final_weight_ess_fraction": float(ess / max(len(out), 1)),
        }
    )
    return out.astype(np.float32), diag
