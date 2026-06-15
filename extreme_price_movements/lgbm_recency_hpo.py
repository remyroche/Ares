from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAYS_PER_MONTH = 365.0 / 12.0
DEFAULT_TRAIN_YEARS = 3
DEFAULT_HOLDOUT_MONTHS = 2
BASE_HALF_LIFE_MONTHS = (6.0, 9.0, 12.0)
META_HALF_LIFE_MONTHS = (3.0, 4.5, 6.0)
COMPOSITE_WEIGHTS = (0.3, 0.4, 0.5)


class RecencyHPOComplete(RuntimeError):
    def __init__(
        self,
        *,
        scope: str,
        scope_key: str,
        payload: dict[str, Any],
    ) -> None:
        self.scope = str(scope)
        self.scope_key = str(scope_key)
        self.payload = dict(payload)
        super().__init__(
            f"recency_hpo complete for scope={self.scope} scope_key={self.scope_key}"
        )


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "y", "on"}


def _cfg_or_env(cfg: dict[str, Any] | None, key: str, env: str, default: Any = None) -> Any:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    raw = os.environ.get(env)
    if raw is not None and str(raw).strip() != "":
        return raw
    return cfg_local.get(key, default)


def objective_scope(objective_mode: str | None) -> str:
    mode = str(objective_mode or "train_base").strip().lower()
    return "meta" if mode in {"meta", "train_meta"} else "base"


def half_life_months_to_days(months: float) -> float:
    return float(months) * DAYS_PER_MONTH


def _as_float_list(value: Any, default: tuple[float, ...]) -> list[float]:
    if value is None or value == "":
        return [float(v) for v in default]
    if isinstance(value, str):
        raw_items = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raw_items = list(value)
    else:
        raw_items = [value]
    out: list[float] = []
    for item in raw_items:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out or [float(v) for v in default]


def _explicit_grid_pairs(
    value: Any,
    *,
    scope: str,
) -> list[dict[str, float | str]]:
    """Parse explicit half-life/composite pairs.

    Accepts strings such as ``"9:0.4,12:0.3,9:0.3"`` or a list of
    ``(half_life_months, composite_weight)`` pairs. This avoids accidentally
    running the cross product when we want to re-test only selected schemes.
    """

    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, str):
        raw_items = [
            part.strip()
            for part in value.replace(";", ",").split(",")
            if part.strip()
        ]
        pairs: list[tuple[str, str]] = []
        for item in raw_items:
            if ":" in item:
                left, right = item.split(":", 1)
            elif "=" in item:
                left, right = item.split("=", 1)
            else:
                bits = item.split()
                if len(bits) != 2:
                    continue
                left, right = bits
            pairs.append((left.strip(), right.strip()))
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        pairs = []
        for item in list(value):
            if isinstance(item, dict):
                left = item.get("half_life_months", item.get("half_life", ""))
                right = item.get("composite_weight", item.get("weight", ""))
                pairs.append((str(left), str(right)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
            else:
                continue
    else:
        return []
    out: list[dict[str, float | str]] = []
    for half_life_raw, composite_raw in pairs:
        try:
            half_life_months = float(half_life_raw)
            composite_weight = float(composite_raw)
        except Exception:
            continue
        out.append(
            {
                "scope": scope,
                "half_life_months": float(half_life_months),
                "half_life_days": half_life_months_to_days(half_life_months),
                "composite_weight": float(composite_weight),
            }
        )
    return out


def recency_hpo_grid(
    scope: str,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, float | str]]:
    norm_scope = "meta" if str(scope).strip().lower() == "meta" else "base"
    explicit_pairs = _explicit_grid_pairs(
        _cfg_or_env(
            cfg,
            f"recency_hpo_{norm_scope}_grid_pairs",
            f"EPM_RECENCY_HPO_{norm_scope.upper()}_GRID_PAIRS",
            _cfg_or_env(
                cfg,
                "recency_hpo_grid_pairs",
                "EPM_RECENCY_HPO_GRID_PAIRS",
            ),
        ),
        scope=norm_scope,
    )
    if explicit_pairs:
        return explicit_pairs
    if norm_scope == "meta":
        half_lives = _as_float_list(
            _cfg_or_env(
                cfg,
                "recency_hpo_meta_half_life_months",
                "EPM_RECENCY_HPO_META_HALF_LIFE_MONTHS",
            ),
            META_HALF_LIFE_MONTHS,
        )
    else:
        half_lives = _as_float_list(
            _cfg_or_env(
                cfg,
                "recency_hpo_base_half_life_months",
                "EPM_RECENCY_HPO_BASE_HALF_LIFE_MONTHS",
            ),
            BASE_HALF_LIFE_MONTHS,
        )
    composite_weights = _as_float_list(
        _cfg_or_env(
            cfg,
            "recency_hpo_composite_weights",
            "EPM_RECENCY_HPO_COMPOSITE_WEIGHTS",
        ),
        COMPOSITE_WEIGHTS,
    )
    grid: list[dict[str, float | str]] = []
    for half_life_months in half_lives:
        for composite_weight in composite_weights:
            grid.append(
                {
                    "scope": norm_scope,
                    "half_life_months": float(half_life_months),
                    "half_life_days": half_life_months_to_days(half_life_months),
                    "composite_weight": float(composite_weight),
                }
            )
    return grid


def _timestamp_series(timestamps: Any, n: int) -> pd.Series | None:
    if timestamps is None:
        return None
    try:
        arr = np.asarray(timestamps)
    except Exception:
        return None
    if len(arr) != int(n):
        return None
    ts = pd.to_datetime(pd.Series(arr), utc=True, errors="coerce")
    if not bool(ts.notna().any()):
        return None
    return ts


def composite_decay_from_timestamps(
    timestamps: Any,
    n: int,
    *,
    half_life_days: float,
    composite_weight: float,
) -> np.ndarray | None:
    ts = _timestamp_series(timestamps, n)
    if ts is None:
        return None
    valid = ts.notna().to_numpy(dtype=bool)
    latest = ts.loc[valid].max()
    age_days = (latest - ts).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    if bool(np.any(valid)):
        max_valid_age = float(np.nanmax(age_days[valid]))
    else:
        max_valid_age = 0.0
    age_days[~valid] = max_valid_age
    exp_decay = np.power(
        0.5,
        np.maximum(age_days, 0.0) / max(float(half_life_days), 1e-6),
    )
    blend = float(np.clip(composite_weight, 0.0, 1.0))
    decay = (1.0 - blend) * exp_decay + blend
    return np.clip(decay, 1e-6, 1.0).astype(np.float32)


def recency_hpo_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    raw = (
        os.environ.get("EPM_RECENCY_HPO_ROOT")
        or cfg_local.get("recency_hpo_root")
        or ""
    )
    if str(raw).strip():
        return Path(str(raw)).expanduser()
    return Path(str(cfg_local.get("data_root", "data"))) / "artifacts" / "recency_hpo"


def recency_hpo_winner_path(
    cfg: dict[str, Any] | None,
    scope: str,
) -> Path:
    norm_scope = "meta" if str(scope).strip().lower() == "meta" else "base"
    scoped_env = f"EPM_RECENCY_HPO_{norm_scope.upper()}_WINNER_PATH"
    raw = (
        os.environ.get(scoped_env)
        or os.environ.get("EPM_RECENCY_HPO_WINNER_PATH")
        or (cfg or {}).get(f"recency_hpo_{norm_scope}_winner_path")
        or (cfg or {}).get("recency_hpo_winner_path")
        or ""
    )
    if str(raw).strip():
        return Path(str(raw)).expanduser()
    return recency_hpo_root(cfg) / f"{norm_scope}_winner.json"


def _override_float(scope: str, name: str, cfg: dict[str, Any] | None) -> float | None:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    scope_key = f"recency_hpo_{scope}_{name}"
    generic_key = f"recency_hpo_{name}"
    scope_env = f"EPM_RECENCY_HPO_{scope.upper()}_{name.upper()}"
    generic_env = f"EPM_RECENCY_HPO_{name.upper()}"
    for raw in (
        os.environ.get(scope_env),
        os.environ.get(generic_env),
        cfg_local.get(scope_key),
        cfg_local.get(generic_key),
    ):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return None


def active_recency_hpo_config(
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> dict[str, Any] | None:
    scope = objective_scope(objective_mode)
    override_half_life_days = _override_float(scope, "half_life_days", cfg)
    override_half_life_months = _override_float(scope, "half_life_months", cfg)
    override_composite_weight = _override_float(scope, "composite_weight", cfg)
    if override_half_life_days is None and override_half_life_months is not None:
        override_half_life_days = half_life_months_to_days(override_half_life_months)
    if override_half_life_days is not None and override_composite_weight is not None:
        return {
            "scope": scope,
            "source": "override",
            "half_life_days": float(override_half_life_days),
            "half_life_months": (
                float(override_half_life_months)
                if override_half_life_months is not None
                else float(override_half_life_days) / DAYS_PER_MONTH
            ),
            "composite_weight": float(override_composite_weight),
        }

    use_winner = _truthy(
        _cfg_or_env(cfg, "recency_hpo_use_winner", "EPM_RECENCY_HPO_USE_WINNER", True),
        default=True,
    )
    if not use_winner:
        return None
    path = recency_hpo_winner_path(cfg, scope)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    params = payload.get("winner", payload) if isinstance(payload, dict) else {}
    if not isinstance(params, dict):
        return None
    try:
        half_life_days = float(params.get("half_life_days"))
        composite_weight = float(params.get("composite_weight"))
    except Exception:
        return None
    if not np.isfinite(half_life_days) or not np.isfinite(composite_weight):
        return None
    if half_life_days <= 0.0:
        return None
    out = dict(params)
    out.update(
        {
            "scope": scope,
            "source": str(params.get("source") or path),
            "winner_path": str(path),
            "half_life_days": float(half_life_days),
            "half_life_months": float(
                params.get("half_life_months", half_life_days / DAYS_PER_MONTH)
            ),
            "composite_weight": float(composite_weight),
        }
    )
    return out


def recency_hpo_decay_from_config(
    timestamps: Any,
    n: int,
    *,
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    active = active_recency_hpo_config(cfg, objective_mode)
    if not active:
        return None, None
    decay = composite_decay_from_timestamps(
        timestamps,
        n,
        half_life_days=float(active["half_life_days"]),
        composite_weight=float(active["composite_weight"]),
    )
    if decay is None:
        return None, active
    return decay, active


def recency_hpo_enabled_for_scope(
    cfg: dict[str, Any] | None,
    scope: str,
    *,
    scope_key: str | None = None,
) -> bool:
    enabled = _truthy(
        _cfg_or_env(cfg, "recency_hpo_enabled", "EPM_RECENCY_HPO_ENABLED", False),
        default=False,
    )
    if not enabled:
        return False
    norm_scope = "meta" if str(scope).strip().lower() == "meta" else "base"
    scope_filter = str(
        _cfg_or_env(cfg, "recency_hpo_scope", "EPM_RECENCY_HPO_SCOPE", "both")
        or "both"
    ).strip().lower()
    allowed = {p.strip() for p in scope_filter.split(",") if p.strip()}
    if allowed and "both" not in allowed and norm_scope not in allowed:
        return False
    key_filter_raw = str(
        _cfg_or_env(
            cfg,
            "recency_hpo_scope_key",
            "EPM_RECENCY_HPO_SCOPE_KEY",
            "",
        )
        or os.environ.get("EPM_RECENCY_HPO_STRATEGY_ID", "")
        or ""
    ).strip()
    if key_filter_raw:
        key = str(scope_key or "")
        filters = [p.strip() for p in key_filter_raw.split(",") if p.strip()]
        return any(f == key or f in key for f in filters)
    return True


def recency_hpo_train_oos_masks(
    timestamps: Any,
    *,
    train_years: int = DEFAULT_TRAIN_YEARS,
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ts = _timestamp_series(timestamps, len(np.asarray(timestamps))) if timestamps is not None else None
    if ts is None:
        raise ValueError("recency_hpo requires timestamp-aligned rows")
    valid = ts.notna().to_numpy(dtype=bool)
    latest = ts.loc[valid].max()
    holdout_start = latest - pd.DateOffset(months=int(holdout_months))
    train_start = holdout_start - pd.DateOffset(years=int(train_years))
    train_mask = (
        valid
        & (ts >= train_start).to_numpy(dtype=bool)
        & (ts < holdout_start).to_numpy(dtype=bool)
    )
    oos_mask = (
        valid
        & (ts >= holdout_start).to_numpy(dtype=bool)
        & (ts <= latest).to_numpy(dtype=bool)
    )
    meta = {
        "train_start": train_start.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": latest.isoformat(),
        "train_years": int(train_years),
        "holdout_months": int(holdout_months),
        "train_rows": int(np.sum(train_mask)),
        "oos_rows": int(np.sum(oos_mask)),
    }
    return train_mask.astype(bool), oos_mask.astype(bool), meta


def precision_at_fraction(
    y_true: Any,
    score: Any,
    fraction: float,
) -> float:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.nan_to_num(np.asarray(score, dtype=np.float32).reshape(-1), nan=-np.inf)
    if len(y) == 0 or len(pred) != len(y):
        return float("nan")
    k = int(np.ceil(float(np.clip(fraction, 1e-6, 1.0)) * len(y)))
    k = max(1, min(k, len(y)))
    order = np.argsort(pred, kind="mergesort")
    top = order[-k:]
    return float(np.mean((y[top] >= 0.5).astype(np.float32)))


def precision_score_top_fracs(y_true: Any, score: Any) -> dict[str, float]:
    p10 = precision_at_fraction(y_true, score, 0.10)
    p20 = precision_at_fraction(y_true, score, 0.20)
    p30 = precision_at_fraction(y_true, score, 0.30)
    precision_score = 0.25 * p10 + 0.50 * p20 + 1.00 * p30
    return {
        "p_at_10": float(p10),
        "p_at_20": float(p20),
        "p_at_30": float(p30),
        "precision_score": float(precision_score),
        "rows": int(len(np.asarray(y_true).reshape(-1))),
    }


def final_selection_score(
    y_true: Any,
    score: Any,
    timestamps: Any,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.asarray(score, dtype=np.float32).reshape(-1)
    ts = _timestamp_series(timestamps, len(y))
    if ts is None or len(pred) != len(y):
        raise ValueError("recency_hpo scoring requires aligned y, score, and timestamps")
    latest = ts.loc[ts.notna()].max()
    mask_4w = (ts >= latest - pd.Timedelta(days=28)).to_numpy(dtype=bool)
    mask_8w = (ts >= latest - pd.Timedelta(days=56)).to_numpy(dtype=bool)
    score_4w = precision_score_top_fracs(y[mask_4w], pred[mask_4w])
    score_8w = precision_score_top_fracs(y[mask_8w], pred[mask_8w])
    final_score = (
        0.50 * float(score_4w["precision_score"])
        + 1.00 * float(score_8w["precision_score"])
    )
    return {
        "precision_last_4w": score_4w,
        "precision_last_8w": score_8w,
        "final_selection_score": float(final_score),
        "oos_latest_ts": latest.isoformat(),
    }


def save_recency_hpo_winner(
    cfg: dict[str, Any] | None,
    scope: str,
    payload: dict[str, Any],
) -> Path:
    path = recency_hpo_winner_path(cfg, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)
    return path


__all__ = [
    "BASE_HALF_LIFE_MONTHS",
    "COMPOSITE_WEIGHTS",
    "DEFAULT_HOLDOUT_MONTHS",
    "DEFAULT_TRAIN_YEARS",
    "META_HALF_LIFE_MONTHS",
    "RecencyHPOComplete",
    "active_recency_hpo_config",
    "composite_decay_from_timestamps",
    "final_selection_score",
    "objective_scope",
    "precision_at_fraction",
    "precision_score_top_fracs",
    "recency_hpo_decay_from_config",
    "recency_hpo_enabled_for_scope",
    "recency_hpo_grid",
    "recency_hpo_train_oos_masks",
    "recency_hpo_winner_path",
    "save_recency_hpo_winner",
]
