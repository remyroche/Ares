#!/usr/bin/env python3
"""Hierarchical Gate 3 HPO for side-specific executable soft labels.

The search is intentionally staged:
1. choose a coarse long/short label family,
2. refine long parameters with short fixed,
3. refine short parameters with long fixed,
4. run a small bounded joint polish.

This avoids a large flat parameter space while keeping the search path auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
    _append_fold_ae_gmm_state_features,
    _apply_spread_symbol_universe,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/gate3_side_soft_label_hpo_v1")
TOP_FRACS = (0.10, 0.20, 0.30)
DEFAULT_ROUND_TRIP_COST = 0.0100
EXECUTABLE_MARGIN_COST_FLOOR = 0.0100
FIXED_LONG_MIN_NET_EDGE = 0.0010
FIXED_SHORT_MIN_NET_EDGE = 0.0010
FIRST_TOUCH_UTILITY_MODES = {
    "first_touch_net",
    "first_touch_net_after_cost",
    "first_touch_ev",
    "first_touch_executable_net",
}


def _fold_cache_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _fold_cache_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fold_cache_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_fold_cache_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _fold_cache_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(_fold_cache_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _fold_time_bounds(frame: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(frame.get("__ts__", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    return {
        "rows": int(len(frame)),
        "timestamp_min": ts.min(),
        "timestamp_max": ts.max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)) if "__symbol__" in frame.columns else 0,
    }


def _ae_gmm_state_feature_random_state(*, fold_i: int, seed: int) -> int:
    return int(seed) + int(fold_i) * 101


def _ae_gmm_fold_cache_payload(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    month: str,
    fold_i: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    seed: int,
    random_state: int,
) -> dict[str, Any]:
    return {
        "cache_version": 1,
        "scope": "gate3_side_soft_label_hpo_ae_gmm_fold_features",
        "labels_path": str(Path(labels_path)),
        "feature_dir": str(Path(feature_dir)),
        "feature_list_csv": str(Path(feature_list_csv)),
        "month": str(month),
        "fold_i": int(fold_i),
        "train": _fold_time_bounds(train),
        "valid": _fold_time_bounds(valid),
        "features": {
            "count": int(len(features)),
            "sha256": hashlib.sha256("\n".join(map(str, features)).encode("utf-8")).hexdigest(),
        },
        "include_ae_gmm_state_features": bool(include_ae_gmm_state_features),
        "ae_gmm_state_feature_max_train_rows": int(ae_gmm_state_feature_max_train_rows),
        "ae_gmm_state_feature_max_iter": int(ae_gmm_state_feature_max_iter),
        "seed": int(seed),
        "random_state": int(random_state),
    }


def _load_ae_gmm_fold_cache(
    *,
    cache_dir: Path,
    digest: str,
    expected_payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]] | None:
    meta_path = cache_dir / f"{digest}.json"
    train_path = cache_dir / f"{digest}.train.parquet"
    valid_path = cache_dir / f"{digest}.valid.parquet"
    if not (meta_path.exists() and train_path.exists() and valid_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("payload") != _fold_cache_json_safe(expected_payload):
        return None
    x_train = pd.read_parquet(train_path).astype(np.float32, copy=False)
    x_valid = pd.read_parquet(valid_path).astype(np.float32, copy=False)
    generated = [str(v) for v in meta.get("generated_features", [])]
    diag = dict(meta.get("ae_diag", {}) or {})
    diag["ae_gmm_state_feature_cache_status"] = "hit"
    diag["ae_gmm_state_feature_cache_key"] = str(digest)
    return x_train, x_valid, generated, diag


def _write_ae_gmm_fold_cache(
    *,
    cache_dir: Path,
    digest: str,
    payload: dict[str, Any],
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    generated: list[str],
    ae_diag: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"{digest}.train.parquet"
    valid_path = cache_dir / f"{digest}.valid.parquet"
    meta_path = cache_dir / f"{digest}.json"
    x_train.astype(np.float32, copy=False).to_parquet(train_path, index=False)
    x_valid.astype(np.float32, copy=False).to_parquet(valid_path, index=False)
    meta = {
        "payload": _fold_cache_json_safe(payload),
        "generated_features": [str(v) for v in generated],
        "ae_diag": _fold_cache_json_safe(ae_diag),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class SideParams:
    min_net_edge: float
    temperature: float
    mae_cap_r: float
    hard_mae_cap_r: float
    mae_penalty: float
    mfe_min_r: float
    mfe_bonus: float
    mfe_mae_ratio_min: float
    time_to_mfe_max_bars: float
    exit_bars_min: float
    exit_bars_max: float
    timeout_penalty: float
    late_penalty: float
    dirty_positive_cap: float
    timeout_cap: float
    bad_mae_cap: float
    post_win_mfe_min_r: float
    post_win_mfe_bonus: float
    first_pass_target_r: float
    first_pass_bad_r: float
    first_pass_reward: float
    first_pass_penalty: float
    adverse_pre_mfe_cap_r: float
    adverse_pre_mfe_penalty: float
    underwater_bars_cap: float
    underwater_penalty: float
    ordered_clean_floor: float
    ordered_dirty_cap: float


@dataclass(frozen=True)
class LabelConfig:
    name: str
    family: str
    long: SideParams
    short: SideParams


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sigmoid(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))


def _num_col(frame: pd.DataFrame, name: str, default: float | np.ndarray | pd.Series) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").reset_index(drop=True)
    if isinstance(default, pd.Series):
        return pd.to_numeric(default.reset_index(drop=True), errors="coerce")
    arr = np.asarray(default)
    if arr.ndim == 0:
        return pd.Series(float(arr), index=pd.RangeIndex(len(frame)), dtype=np.float64)
    return pd.Series(arr, index=pd.RangeIndex(len(frame)), dtype=np.float64)


def _threshold_bar_column(prefix: str, threshold_r: float) -> str:
    if str(prefix).endswith("mae"):
        levels = [(0.50, "05r"), (0.75, "075r"), (1.00, "1r"), (1.50, "15r")]
    else:
        levels = [(0.50, "05r"), (0.75, "075r"), (1.00, "1r"), (1.25, "125r"), (1.50, "15r")]
    level = min(levels, key=lambda item: abs(float(item[0]) - float(threshold_r)))
    return f"{prefix}_{level[1]}"


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _base_long() -> SideParams:
    return SideParams(
        min_net_edge=FIXED_LONG_MIN_NET_EDGE,
        temperature=0.0060,
        mae_cap_r=0.60,
        hard_mae_cap_r=0.60,
        mae_penalty=1.20,
        mfe_min_r=1.35,
        mfe_bonus=0.18,
        mfe_mae_ratio_min=1.75,
        time_to_mfe_max_bars=6.0,
        exit_bars_min=0.0,
        exit_bars_max=7.0,
        timeout_penalty=1.40,
        late_penalty=0.75,
        dirty_positive_cap=0.12,
        timeout_cap=0.02,
        bad_mae_cap=0.03,
        post_win_mfe_min_r=0.20,
        post_win_mfe_bonus=0.10,
        first_pass_target_r=1.25,
        first_pass_bad_r=0.75,
        first_pass_reward=1.15,
        first_pass_penalty=1.65,
        adverse_pre_mfe_cap_r=0.65,
        adverse_pre_mfe_penalty=1.30,
        underwater_bars_cap=6.0,
        underwater_penalty=0.35,
        ordered_clean_floor=0.58,
        ordered_dirty_cap=0.05,
    )


def _base_short() -> SideParams:
    return SideParams(
        min_net_edge=FIXED_SHORT_MIN_NET_EDGE,
        temperature=0.0070,
        mae_cap_r=0.55,
        hard_mae_cap_r=0.55,
        mae_penalty=1.45,
        mfe_min_r=1.25,
        mfe_bonus=0.16,
        mfe_mae_ratio_min=1.75,
        time_to_mfe_max_bars=10.0,
        exit_bars_min=4.0,
        exit_bars_max=12.0,
        timeout_penalty=1.20,
        late_penalty=0.65,
        dirty_positive_cap=0.12,
        timeout_cap=0.02,
        bad_mae_cap=0.03,
        post_win_mfe_min_r=0.20,
        post_win_mfe_bonus=0.10,
        first_pass_target_r=1.35,
        first_pass_bad_r=0.75,
        first_pass_reward=1.20,
        first_pass_penalty=1.85,
        adverse_pre_mfe_cap_r=0.60,
        adverse_pre_mfe_penalty=1.45,
        underwater_bars_cap=9.0,
        underwater_penalty=0.30,
        ordered_clean_floor=0.58,
        ordered_dirty_cap=0.04,
    )


def _family_config(name: str) -> LabelConfig:
    long = _base_long()
    short = _base_short()
    if name == "long_fast_short_controlled":
        pass
    elif name == "long_strict_short_decisive":
        long = replace(long, mae_cap_r=0.50, hard_mae_cap_r=0.50, time_to_mfe_max_bars=5.0, exit_bars_max=6.0)
        short = replace(short, hard_mae_cap_r=0.45, mfe_min_r=1.45, mfe_mae_ratio_min=2.05)
    elif name == "short_mae_strict":
        short = replace(short, mae_cap_r=0.42, hard_mae_cap_r=0.42, mae_penalty=1.95, dirty_positive_cap=0.07, bad_mae_cap=0.015)
    elif name == "short_slower_clean":
        short = replace(short, time_to_mfe_max_bars=14.0, exit_bars_max=16.0, mfe_min_r=1.35, hard_mae_cap_r=0.50)
    elif name == "dirty_low_cap":
        long = replace(long, dirty_positive_cap=0.06, bad_mae_cap=0.015, timeout_cap=0.01)
        short = replace(short, dirty_positive_cap=0.06, bad_mae_cap=0.015, timeout_cap=0.01)
    elif name == "post_win_mfe_heavy":
        long = replace(long, post_win_mfe_min_r=0.35, post_win_mfe_bonus=0.24, mfe_bonus=0.25)
        short = replace(short, post_win_mfe_min_r=0.45, post_win_mfe_bonus=0.28, mfe_bonus=0.24)
    elif name == "balanced_loose":
        long = replace(long, hard_mae_cap_r=0.70, time_to_mfe_max_bars=8.0, exit_bars_max=10.0, dirty_positive_cap=0.16)
        short = replace(short, hard_mae_cap_r=0.70, time_to_mfe_max_bars=14.0, exit_bars_max=16.0, dirty_positive_cap=0.16)
    else:
        raise ValueError(f"Unknown family: {name}")
    return LabelConfig(name=f"family_{name}", family=name, long=long, short=short)


FAMILIES: tuple[str, ...] = (
    "long_fast_short_controlled",
    "long_strict_short_decisive",
    "short_mae_strict",
    "short_slower_clean",
    "dirty_low_cap",
    "post_win_mfe_heavy",
    "balanced_loose",
)


def _clip_params(p: SideParams, *, side: str) -> SideParams:
    if side == "long":
        return SideParams(
            min_net_edge=float(np.clip(p.min_net_edge, 0.00025, 0.0018)),
            temperature=float(np.clip(p.temperature, 0.0025, 0.014)),
            mae_cap_r=float(np.clip(p.mae_cap_r, 0.30, 0.90)),
            hard_mae_cap_r=float(np.clip(p.hard_mae_cap_r, 0.30, 0.90)),
            mae_penalty=float(np.clip(p.mae_penalty, 0.20, 2.80)),
            mfe_min_r=float(np.clip(p.mfe_min_r, 0.80, 2.10)),
            mfe_bonus=float(np.clip(p.mfe_bonus, 0.0, 0.60)),
            mfe_mae_ratio_min=float(np.clip(p.mfe_mae_ratio_min, 1.00, 3.00)),
            time_to_mfe_max_bars=float(np.clip(p.time_to_mfe_max_bars, 3.0, 12.0)),
            exit_bars_min=float(np.clip(p.exit_bars_min, 0.0, 8.0)),
            exit_bars_max=float(np.clip(max(p.exit_bars_max, p.exit_bars_min + 1.0), 4.0, 14.0)),
            timeout_penalty=float(np.clip(p.timeout_penalty, 0.25, 2.50)),
            late_penalty=float(np.clip(p.late_penalty, 0.0, 2.00)),
            dirty_positive_cap=float(np.clip(p.dirty_positive_cap, 0.005, 0.25)),
            timeout_cap=float(np.clip(p.timeout_cap, 0.0, 0.10)),
            bad_mae_cap=float(np.clip(p.bad_mae_cap, 0.0, 0.12)),
            post_win_mfe_min_r=float(np.clip(p.post_win_mfe_min_r, 0.0, 1.00)),
            post_win_mfe_bonus=float(np.clip(p.post_win_mfe_bonus, 0.0, 0.60)),
            first_pass_target_r=float(np.clip(p.first_pass_target_r, 0.75, 2.20)),
            first_pass_bad_r=float(np.clip(p.first_pass_bad_r, 0.35, 1.25)),
            first_pass_reward=float(np.clip(p.first_pass_reward, 0.0, 3.00)),
            first_pass_penalty=float(np.clip(p.first_pass_penalty, 0.0, 3.50)),
            adverse_pre_mfe_cap_r=float(np.clip(p.adverse_pre_mfe_cap_r, 0.25, 1.25)),
            adverse_pre_mfe_penalty=float(np.clip(p.adverse_pre_mfe_penalty, 0.0, 3.50)),
            underwater_bars_cap=float(np.clip(p.underwater_bars_cap, 1.0, 14.0)),
            underwater_penalty=float(np.clip(p.underwater_penalty, 0.0, 1.50)),
            ordered_clean_floor=float(np.clip(p.ordered_clean_floor, 0.30, 0.90)),
            ordered_dirty_cap=float(np.clip(p.ordered_dirty_cap, 0.0, 0.20)),
        )
    return SideParams(
        min_net_edge=float(np.clip(p.min_net_edge, 0.00025, 0.0022)),
        temperature=float(np.clip(p.temperature, 0.0025, 0.016)),
        mae_cap_r=float(np.clip(p.mae_cap_r, 0.30, 0.95)),
        hard_mae_cap_r=float(np.clip(p.hard_mae_cap_r, 0.30, 0.95)),
        mae_penalty=float(np.clip(p.mae_penalty, 0.20, 3.20)),
        mfe_min_r=float(np.clip(p.mfe_min_r, 0.80, 2.30)),
        mfe_bonus=float(np.clip(p.mfe_bonus, 0.0, 0.70)),
        mfe_mae_ratio_min=float(np.clip(p.mfe_mae_ratio_min, 1.00, 3.20)),
        time_to_mfe_max_bars=float(np.clip(p.time_to_mfe_max_bars, 4.0, 18.0)),
        exit_bars_min=float(np.clip(p.exit_bars_min, 0.0, 10.0)),
        exit_bars_max=float(np.clip(max(p.exit_bars_max, p.exit_bars_min + 1.0), 6.0, 20.0)),
        timeout_penalty=float(np.clip(p.timeout_penalty, 0.25, 2.50)),
        late_penalty=float(np.clip(p.late_penalty, 0.0, 2.00)),
        dirty_positive_cap=float(np.clip(p.dirty_positive_cap, 0.005, 0.25)),
        timeout_cap=float(np.clip(p.timeout_cap, 0.0, 0.10)),
        bad_mae_cap=float(np.clip(p.bad_mae_cap, 0.0, 0.12)),
        post_win_mfe_min_r=float(np.clip(p.post_win_mfe_min_r, 0.0, 1.20)),
        post_win_mfe_bonus=float(np.clip(p.post_win_mfe_bonus, 0.0, 0.70)),
        first_pass_target_r=float(np.clip(p.first_pass_target_r, 0.75, 2.40)),
        first_pass_bad_r=float(np.clip(p.first_pass_bad_r, 0.35, 1.35)),
        first_pass_reward=float(np.clip(p.first_pass_reward, 0.0, 3.20)),
        first_pass_penalty=float(np.clip(p.first_pass_penalty, 0.0, 3.80)),
        adverse_pre_mfe_cap_r=float(np.clip(p.adverse_pre_mfe_cap_r, 0.25, 1.35)),
        adverse_pre_mfe_penalty=float(np.clip(p.adverse_pre_mfe_penalty, 0.0, 3.80)),
        underwater_bars_cap=float(np.clip(p.underwater_bars_cap, 1.0, 20.0)),
        underwater_penalty=float(np.clip(p.underwater_penalty, 0.0, 1.50)),
        ordered_clean_floor=float(np.clip(p.ordered_clean_floor, 0.30, 0.90)),
        ordered_dirty_cap=float(np.clip(p.ordered_dirty_cap, 0.0, 0.20)),
    )


def _jitter_params(rng: np.random.Generator, p: SideParams, *, side: str, scale: float) -> SideParams:
    values = asdict(p)
    for key in values:
        v = float(values[key])
        if key in {"time_to_mfe_max_bars", "exit_bars_min", "exit_bars_max"}:
            values[key] = v + float(rng.normal(0.0, 2.0 * scale))
        elif key == "min_net_edge":
            values[key] = FIXED_LONG_MIN_NET_EDGE if side == "long" else FIXED_SHORT_MIN_NET_EDGE
        elif key in {
            "mae_penalty",
            "timeout_penalty",
            "late_penalty",
            "mfe_mae_ratio_min",
            "first_pass_reward",
            "first_pass_penalty",
            "adverse_pre_mfe_penalty",
        }:
            values[key] = v + float(rng.normal(0.0, 0.40 * scale))
        elif key in {
            "mfe_min_r",
            "mae_cap_r",
            "hard_mae_cap_r",
            "post_win_mfe_min_r",
            "first_pass_target_r",
            "first_pass_bad_r",
            "adverse_pre_mfe_cap_r",
        }:
            values[key] = v + float(rng.normal(0.0, 0.15 * scale))
        elif key in {
            "mfe_bonus",
            "post_win_mfe_bonus",
            "dirty_positive_cap",
            "timeout_cap",
            "bad_mae_cap",
            "ordered_clean_floor",
            "ordered_dirty_cap",
        }:
            values[key] = v + float(rng.normal(0.0, 0.05 * scale))
        elif key in {"underwater_bars_cap"}:
            values[key] = v + float(rng.normal(0.0, 2.0 * scale))
        elif key in {"underwater_penalty"}:
            values[key] = v + float(rng.normal(0.0, 0.20 * scale))
        else:
            values[key] = v + float(rng.normal(0.0, 0.00035 * scale))
    return _clip_params(SideParams(**values), side=side)


def _suggest_side_params(trial: Any, base: SideParams, *, side: str, prefix: str, radius: float = 1.0) -> SideParams:
    def sf(name: str, low: float, high: float) -> float:
        center = getattr(base, name)
        lo = center - (center - low) * radius
        hi = center + (high - center) * radius
        return float(trial.suggest_float(f"{prefix}_{name}", lo, hi))

    if side == "long":
        p = SideParams(
            min_net_edge=FIXED_LONG_MIN_NET_EDGE,
            temperature=sf("temperature", 0.0025, 0.014),
            mae_cap_r=sf("mae_cap_r", 0.30, 0.90),
            hard_mae_cap_r=sf("hard_mae_cap_r", 0.30, 0.90),
            mae_penalty=sf("mae_penalty", 0.20, 2.80),
            mfe_min_r=sf("mfe_min_r", 0.80, 2.10),
            mfe_bonus=sf("mfe_bonus", 0.0, 0.60),
            mfe_mae_ratio_min=sf("mfe_mae_ratio_min", 1.00, 3.00),
            time_to_mfe_max_bars=sf("time_to_mfe_max_bars", 3.0, 12.0),
            exit_bars_min=sf("exit_bars_min", 0.0, 8.0),
            exit_bars_max=sf("exit_bars_max", 4.0, 14.0),
            timeout_penalty=sf("timeout_penalty", 0.25, 2.50),
            late_penalty=sf("late_penalty", 0.0, 2.00),
            dirty_positive_cap=sf("dirty_positive_cap", 0.005, 0.25),
            timeout_cap=sf("timeout_cap", 0.0, 0.10),
            bad_mae_cap=sf("bad_mae_cap", 0.0, 0.12),
            post_win_mfe_min_r=sf("post_win_mfe_min_r", 0.0, 1.00),
            post_win_mfe_bonus=sf("post_win_mfe_bonus", 0.0, 0.60),
            first_pass_target_r=sf("first_pass_target_r", 0.75, 2.20),
            first_pass_bad_r=sf("first_pass_bad_r", 0.35, 1.25),
            first_pass_reward=sf("first_pass_reward", 0.0, 3.00),
            first_pass_penalty=sf("first_pass_penalty", 0.0, 3.50),
            adverse_pre_mfe_cap_r=sf("adverse_pre_mfe_cap_r", 0.25, 1.25),
            adverse_pre_mfe_penalty=sf("adverse_pre_mfe_penalty", 0.0, 3.50),
            underwater_bars_cap=sf("underwater_bars_cap", 1.0, 14.0),
            underwater_penalty=sf("underwater_penalty", 0.0, 1.50),
            ordered_clean_floor=sf("ordered_clean_floor", 0.30, 0.90),
            ordered_dirty_cap=sf("ordered_dirty_cap", 0.0, 0.20),
        )
    else:
        p = SideParams(
            min_net_edge=FIXED_SHORT_MIN_NET_EDGE,
            temperature=sf("temperature", 0.0025, 0.016),
            mae_cap_r=sf("mae_cap_r", 0.30, 0.95),
            hard_mae_cap_r=sf("hard_mae_cap_r", 0.30, 0.95),
            mae_penalty=sf("mae_penalty", 0.20, 3.20),
            mfe_min_r=sf("mfe_min_r", 0.80, 2.30),
            mfe_bonus=sf("mfe_bonus", 0.0, 0.70),
            mfe_mae_ratio_min=sf("mfe_mae_ratio_min", 1.00, 3.20),
            time_to_mfe_max_bars=sf("time_to_mfe_max_bars", 4.0, 18.0),
            exit_bars_min=sf("exit_bars_min", 0.0, 10.0),
            exit_bars_max=sf("exit_bars_max", 6.0, 20.0),
            timeout_penalty=sf("timeout_penalty", 0.25, 2.50),
            late_penalty=sf("late_penalty", 0.0, 2.00),
            dirty_positive_cap=sf("dirty_positive_cap", 0.005, 0.25),
            timeout_cap=sf("timeout_cap", 0.0, 0.10),
            bad_mae_cap=sf("bad_mae_cap", 0.0, 0.12),
            post_win_mfe_min_r=sf("post_win_mfe_min_r", 0.0, 1.20),
            post_win_mfe_bonus=sf("post_win_mfe_bonus", 0.0, 0.70),
            first_pass_target_r=sf("first_pass_target_r", 0.75, 2.40),
            first_pass_bad_r=sf("first_pass_bad_r", 0.35, 1.35),
            first_pass_reward=sf("first_pass_reward", 0.0, 3.20),
            first_pass_penalty=sf("first_pass_penalty", 0.0, 3.80),
            adverse_pre_mfe_cap_r=sf("adverse_pre_mfe_cap_r", 0.25, 1.35),
            adverse_pre_mfe_penalty=sf("adverse_pre_mfe_penalty", 0.0, 3.80),
            underwater_bars_cap=sf("underwater_bars_cap", 1.0, 20.0),
            underwater_penalty=sf("underwater_penalty", 0.0, 1.50),
            ordered_clean_floor=sf("ordered_clean_floor", 0.30, 0.90),
            ordered_dirty_cap=sf("ordered_dirty_cap", 0.0, 0.20),
        )
    return _clip_params(p, side=side)


def _side_label(
    metrics: pd.DataFrame,
    p: SideParams,
    *,
    round_trip_cost: float,
    path_order_mode: str,
    target_utility_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    raw_u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(-1.0).to_numpy(dtype=np.float64)
    mode = str(target_utility_mode or "net_after_cost").strip().lower()
    first_touch_net = (
        _num_col(metrics, "first_touch_net", raw_u)
        .fillna(pd.Series(raw_u))
        .to_numpy(dtype=np.float64)
    )
    if mode == "geometry_only":
        u = raw_u
        utility_gate = np.ones(len(metrics), dtype=bool)
    elif mode == "raw_positive":
        u = raw_u
        utility_gate = u > float(p.min_net_edge)
    elif mode in FIRST_TOUCH_UTILITY_MODES:
        u = first_touch_net
        utility_gate = u > float(p.min_net_edge)
    else:
        u = raw_u - float(round_trip_cost)
        utility_gate = u > float(p.min_net_edge)
    barrier = pd.to_numeric(metrics["barrier"], errors="coerce").fillna(0.01).clip(lower=1e-8).to_numpy(dtype=np.float64)
    mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(99.0).to_numpy(dtype=np.float64)
    mfe = pd.to_numeric(metrics["mfe_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    bars_mfe = pd.to_numeric(metrics["bars_to_mfe"], errors="coerce").fillna(999.0).to_numpy(dtype=np.float64)
    bars_exit = pd.to_numeric(metrics["bars_policy"], errors="coerce").fillna(999.0).to_numpy(dtype=np.float64)
    timeout = pd.to_numeric(metrics["is_timeout"], errors="coerce").fillna(1.0).to_numpy(dtype=bool)
    ft_available = _num_col(metrics, "first_touch_available", 0.0).fillna(0.0).to_numpy(dtype=bool)
    ft_hit = _num_col(metrics, "first_touch_hit", 0.0).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    ft_clean = _num_col(metrics, "first_touch_clean_exec", 0.0).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    ft_stop = _num_col(metrics, "first_touch_stop", 0.0).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    ft_timeout = _num_col(metrics, "first_touch_timeout", timeout.astype(float)).fillna(pd.Series(timeout.astype(float))).to_numpy(dtype=np.float64) > 0.5
    ft_bar = _num_col(metrics, "first_touch_bar", bars_mfe).fillna(pd.Series(bars_mfe)).to_numpy(dtype=np.float64)
    ft_mae_to_sl = _num_col(metrics, "first_touch_mae_to_sl", mae).fillna(pd.Series(mae)).to_numpy(dtype=np.float64)
    full_path_mae = _num_col(metrics, "first_touch_full_path_mae_norm", mae).fillna(pd.Series(mae)).to_numpy(dtype=np.float64)
    underwater_proxy = _num_col(
        metrics,
        "underwater_bars_before_mfe_proxy",
        np.where(mae > 0.25, bars_mfe, 0.0),
    ).fillna(pd.Series(np.where(mae > 0.25, bars_mfe, 0.0)))
    underwater_bars = (
        _num_col(metrics, "underwater_bars_before_mfe_1r", np.nan)
        .fillna(underwater_proxy)
        .to_numpy(dtype=np.float64)
    )
    adverse_path_proxy = np.where(ft_available, ft_mae_to_sl, np.minimum(full_path_mae, mae))
    max_adverse_before_mfe = (
        _num_col(metrics, "max_adverse_before_mfe_1r", np.nan)
        .fillna(pd.Series(adverse_path_proxy))
        .to_numpy(dtype=np.float64)
    )
    target_bar_col = _threshold_bar_column("bars_to_mfe", float(p.first_pass_target_r))
    bad_bar_col = _threshold_bar_column("bars_to_mae", float(p.first_pass_bad_r))
    threshold_bars_to_mfe = _num_col(metrics, target_bar_col, -1.0).fillna(-1.0).to_numpy(dtype=np.float64)
    threshold_bars_to_mae = _num_col(metrics, bad_bar_col, -1.0).fillna(-1.0).to_numpy(dtype=np.float64)
    has_threshold_order = (threshold_bars_to_mfe > 0.0) | (threshold_bars_to_mae > 0.0)
    ratio = mfe / np.maximum(mae, 0.25)
    late_mfe = np.maximum(bars_mfe - float(p.time_to_mfe_max_bars), 0.0)
    late_exit = np.maximum(bars_exit - float(p.exit_bars_max), 0.0)
    early_exit = np.maximum(float(p.exit_bars_min) - bars_exit, 0.0)
    post_win_mfe = np.maximum(mfe - float(p.post_win_mfe_min_r), 0.0)
    threshold_first_good = (
        (threshold_bars_to_mfe > 0.0)
        & ((threshold_bars_to_mae < 0.0) | (threshold_bars_to_mfe < threshold_bars_to_mae))
        & (threshold_bars_to_mfe <= float(p.time_to_mfe_max_bars))
    )
    threshold_first_bad = (
        (threshold_bars_to_mae > 0.0)
        & ((threshold_bars_to_mfe < 0.0) | (threshold_bars_to_mae < threshold_bars_to_mfe))
    )
    aggregate_first_good = (
        (mfe >= float(p.first_pass_target_r))
        & (bars_mfe <= float(p.time_to_mfe_max_bars))
        & (mae <= float(p.adverse_pre_mfe_cap_r))
        & (~timeout)
    )
    aggregate_first_bad = (
        (mae >= float(p.first_pass_bad_r))
        & ((mfe < float(p.first_pass_target_r)) | (bars_mfe > float(p.time_to_mfe_max_bars)))
    )
    fallback_first_good = np.where(has_threshold_order, threshold_first_good, aggregate_first_good)
    fallback_first_bad = np.where(has_threshold_order, threshold_first_bad, aggregate_first_bad)
    first_pass_good = np.where(ft_available, ft_clean | ft_hit, fallback_first_good)
    first_pass_bad = np.where(
        ft_available,
        ft_stop | (max_adverse_before_mfe >= float(p.first_pass_bad_r)),
        fallback_first_bad,
    )
    adverse_before_profit = np.maximum(
        max_adverse_before_mfe - float(p.adverse_pre_mfe_cap_r),
        0.0,
    )
    underwater_excess = np.maximum(underwater_bars - float(p.underwater_bars_cap), 0.0)
    use_path_order = str(path_order_mode).strip().lower() not in {"", "legacy", "none", "off"}
    edge = (
        u
        - float(p.min_net_edge)
        - float(p.mae_penalty) * barrier * np.maximum(mae - float(p.mae_cap_r), 0.0)
        - float(p.timeout_penalty) * barrier * timeout.astype(np.float64)
        - float(p.late_penalty) * barrier * (late_mfe + 0.50 * late_exit + 0.25 * early_exit)
        + float(p.mfe_bonus) * barrier * np.maximum(mfe - float(p.mfe_min_r), 0.0)
        + float(p.post_win_mfe_bonus) * barrier * post_win_mfe
    )
    if use_path_order:
        edge = (
            edge
            + float(p.first_pass_reward) * barrier * first_pass_good.astype(np.float64)
            - float(p.first_pass_penalty) * barrier * first_pass_bad.astype(np.float64)
            - float(p.adverse_pre_mfe_penalty) * barrier * adverse_before_profit
            - float(p.underwater_penalty) * barrier * underwater_excess
            - 0.50 * float(p.timeout_penalty) * barrier * ft_timeout.astype(np.float64)
        )
    soft = _sigmoid(edge / max(float(p.temperature), 1e-8))
    legacy_hard = (
        utility_gate
        & (~timeout)
        & (mae <= float(p.hard_mae_cap_r))
        & (mfe >= float(p.mfe_min_r))
        & (ratio >= float(p.mfe_mae_ratio_min))
        & (bars_mfe <= float(p.time_to_mfe_max_bars))
        & (bars_exit >= float(p.exit_bars_min))
        & (bars_exit <= float(p.exit_bars_max))
        & (post_win_mfe >= 0.0)
    )
    if use_path_order:
        hard = (
            legacy_hard
            & first_pass_good
            & (~first_pass_bad)
            & (adverse_before_profit <= 0.0)
            & (underwater_bars <= float(p.underwater_bars_cap))
        )
    else:
        hard = legacy_hard
    positive = utility_gate if mode == "geometry_only" else (u > float(p.min_net_edge))
    dirty_positive = positive & ~hard
    cap = np.ones(len(metrics), dtype=np.float64)
    cap = np.where(dirty_positive, np.minimum(cap, float(p.dirty_positive_cap)), cap)
    cap = np.where(timeout, np.minimum(cap, float(p.timeout_cap)), cap)
    cap = np.where(mae > float(p.hard_mae_cap_r), np.minimum(cap, float(p.bad_mae_cap)), cap)
    if use_path_order:
        cap = np.where(first_pass_bad, np.minimum(cap, float(p.ordered_dirty_cap)), cap)
        cap = np.where(adverse_before_profit > 0.0, np.minimum(cap, float(p.ordered_dirty_cap)), cap)
        cap = np.where(underwater_excess > 0.0, np.minimum(cap, float(p.ordered_dirty_cap)), cap)
        if mode in FIRST_TOUCH_UTILITY_MODES:
            cap = np.where(utility_gate, cap, np.minimum(cap, float(p.ordered_dirty_cap)))
        soft = np.where(hard, np.maximum(soft, float(p.ordered_clean_floor)), np.minimum(soft, cap))
    else:
        soft = np.where(hard, np.maximum(soft, 0.65), np.minimum(soft, cap))
    diag = {
        "first_touch_available_rate": _safe_mean(ft_available.astype(float)),
        "first_pass_good_rate": _safe_mean(first_pass_good.astype(float)),
        "first_pass_bad_rate": _safe_mean(first_pass_bad.astype(float)),
        "first_touch_net_mean": _safe_mean(first_touch_net),
        "adverse_before_profit_mean": _safe_mean(adverse_before_profit),
        "underwater_bars_mean": _safe_mean(underwater_bars),
    }
    return (
        np.clip(soft, 0.0, 1.0).astype(np.float32),
        hard.astype(bool),
        dirty_positive.astype(bool),
        first_pass_good.astype(bool),
        first_pass_bad.astype(bool),
        diag,
    )


def _make_side_soft_label(
    metrics: pd.DataFrame,
    config: LabelConfig,
    *,
    round_trip_cost: float,
    path_order_mode: str,
    target_utility_mode: str,
) -> pd.DataFrame:
    side = pd.to_numeric(metrics["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    long_mask = side >= 0.0
    short_mask = ~long_mask
    soft = np.zeros(len(metrics), dtype=np.float32)
    clean = np.zeros(len(metrics), dtype=bool)
    dirty = np.zeros(len(metrics), dtype=bool)
    first_touch_available = _num_col(metrics, "first_touch_available", 0.0).fillna(0.0).to_numpy(dtype=bool)
    first_pass_good_all = np.zeros(len(metrics), dtype=bool)
    first_pass_bad_all = np.zeros(len(metrics), dtype=bool)
    if bool(long_mask.any()):
        s, c, d, g, b, _diag = _side_label(
            metrics.loc[long_mask].reset_index(drop=True),
            config.long,
            round_trip_cost=round_trip_cost,
            path_order_mode=path_order_mode,
            target_utility_mode=target_utility_mode,
        )
        soft[long_mask] = s
        clean[long_mask] = c
        dirty[long_mask] = d
        first_pass_good_all[long_mask] = g
        first_pass_bad_all[long_mask] = b
    if bool(short_mask.any()):
        s, c, d, g, b, _diag = _side_label(
            metrics.loc[short_mask].reset_index(drop=True),
            config.short,
            round_trip_cost=round_trip_cost,
            path_order_mode=path_order_mode,
            target_utility_mode=target_utility_mode,
        )
        soft[short_mask] = s
        clean[short_mask] = c
        dirty[short_mask] = d
        first_pass_good_all[short_mask] = g
        first_pass_bad_all[short_mask] = b
    raw_u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev_u = raw_u - float(round_trip_cost)
    mode = str(target_utility_mode or "").strip().lower()
    if mode == "geometry_only":
        positive = np.ones(len(metrics), dtype=bool)
    elif mode == "raw_positive":
        positive = raw_u > 0.0
    elif mode in FIRST_TOUCH_UTILITY_MODES:
        first_touch_net = (
            _num_col(metrics, "first_touch_net", raw_u)
            .fillna(pd.Series(raw_u))
            .to_numpy(dtype=np.float64)
        )
        positive = first_touch_net > 0.0
    else:
        positive = ev_u > 0.0
    return pd.DataFrame(
        {
            "target_soft": soft.astype(np.float32),
            "target_hard": clean.astype(np.int8),
            "dirty_positive": dirty.astype(np.int8),
            "positive_u": positive.astype(np.int8),
            "first_touch_available": first_touch_available.astype(np.int8),
            "first_pass_good": first_pass_good_all.astype(np.int8),
            "first_pass_bad": first_pass_bad_all.astype(np.int8),
        }
    )


def _sample_weight(
    metrics: pd.DataFrame,
    label: pd.DataFrame,
    *,
    round_trip_cost: float,
    target_utility_mode: str,
) -> pd.Series:
    mode = str(target_utility_mode or "").strip().lower()
    if mode in FIRST_TOUCH_UTILITY_MODES:
        raw_after_cost = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0) - float(round_trip_cost)
        u = _num_col(metrics, "first_touch_net", raw_after_cost).fillna(raw_after_cost)
    else:
        u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0) - float(round_trip_cost)
    mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(metrics["is_timeout"], errors="coerce").fillna(0.0).astype(float)
    clean = pd.to_numeric(label["target_hard"], errors="coerce").fillna(0.0)
    dirty = pd.to_numeric(label["dirty_positive"], errors="coerce").fillna(0.0)
    rank = u.clip(lower=0.0).rank(method="average", pct=True).fillna(0.0)
    w = 1.0 + 1.25 * clean + 1.00 * dirty + 0.90 * (u.gt(0.0) & mae.ge(1.0)).astype(float)
    w = w + 0.45 * (u.gt(0.0) & timeout.gt(0.5)).astype(float) + 0.40 * rank
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.10, 6.0)
    return (w / max(float(w.mean()), 1e-12)).astype(np.float32)


def _cap_train_rows(
    x: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    *,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if int(max_rows) <= 0 or len(x) <= int(max_rows):
        return x.reset_index(drop=True), y.reset_index(drop=True), w.reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(x) - 1, int(max_rows), dtype=np.int64))
    return x.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True), w.iloc[idx].reset_index(drop=True)


def _fit_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    *,
    seed: int,
) -> pd.Series:
    model = ExtraTreesRegressor(
        n_estimators=128,
        max_depth=9,
        min_samples_leaf=35,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(
        x_train.reset_index(drop=True),
        pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=pd.to_numeric(w_train, errors="coerce").fillna(1.0).to_numpy(dtype=np.float32),
    )
    pred = model.predict(x_valid.reset_index(drop=True)).astype(np.float32)
    return pd.Series(pred, index=pd.RangeIndex(len(x_valid)), dtype=np.float32)


def _top_metrics(
    *,
    score: pd.Series,
    label: pd.DataFrame,
    metrics: pd.DataFrame,
    round_trip_cost: float,
    prefix: str = "",
) -> dict[str, float]:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    clean = pd.to_numeric(label["target_hard"].reset_index(drop=True), errors="coerce").fillna(0.0)
    dirty = pd.to_numeric(label["dirty_positive"].reset_index(drop=True), errors="coerce").fillna(0.0)
    first_good = pd.to_numeric(label.get("first_pass_good", pd.Series(0.0, index=label.index)).reset_index(drop=True), errors="coerce").fillna(0.0)
    first_bad = pd.to_numeric(label.get("first_pass_bad", pd.Series(0.0, index=label.index)).reset_index(drop=True), errors="coerce").fillna(0.0)
    ft_available = pd.to_numeric(label.get("first_touch_available", pd.Series(0.0, index=label.index)).reset_index(drop=True), errors="coerce").fillna(0.0)
    raw_u = pd.to_numeric(metrics["u_policy_net"].reset_index(drop=True), errors="coerce").fillna(0.0)
    ev_u = raw_u - float(round_trip_cost)
    mae = pd.to_numeric(metrics["mae_norm"].reset_index(drop=True), errors="coerce").fillna(99.0)
    first_touch_mae = pd.to_numeric(metrics.get("first_touch_mae_to_sl", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True), errors="coerce").fillna(mae)
    first_touch_mae_norm = pd.to_numeric(
        metrics.get("first_touch_mae_norm", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(first_touch_mae)
    first_touch_full_path_mae_norm = pd.to_numeric(
        metrics.get("first_touch_full_path_mae_norm", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(mae)
    first_touch_net = pd.to_numeric(
        metrics.get("first_touch_net", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(raw_u)
    row_cost = pd.to_numeric(
        label.get(
            "round_trip_cost",
            metrics.get("round_trip_cost", pd.Series(float(round_trip_cost), index=metrics.index)),
        ).reset_index(drop=True),
        errors="coerce",
    ).fillna(float(round_trip_cost))
    first_touch_gross = first_touch_net + row_cost
    executable_cost = row_cost.clip(lower=float(EXECUTABLE_MARGIN_COST_FLOOR))
    first_touch_gross_minus_1pct = first_touch_gross - float(EXECUTABLE_MARGIN_COST_FLOOR)
    first_touch_executable_margin = first_touch_gross - executable_cost
    gross_abs_weight = first_touch_gross.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    underwater_proxy = pd.to_numeric(
        metrics.get("underwater_bars_before_mfe_proxy", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    )
    underwater_bars = pd.to_numeric(
        metrics.get("underwater_bars_before_mfe_1r", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(underwater_proxy).fillna(0.0)
    underwater_fraction = pd.to_numeric(
        metrics.get("underwater_fraction_before_mfe_1r", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0)
    mfe_before_mae_1r = pd.to_numeric(
        metrics.get("mfe_1r_before_mae_1r", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0)
    mae_before_mfe_1r = pd.to_numeric(
        metrics.get("mae_1r_before_mfe_1r", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0)
    max_adverse_before_mfe_1r = pd.to_numeric(
        metrics.get("max_adverse_before_mfe_1r", pd.Series(np.nan, index=metrics.index)).reset_index(drop=True),
        errors="coerce",
    ).fillna(mae)
    timeout = pd.to_numeric(metrics["is_timeout"].reset_index(drop=True), errors="coerce").fillna(1.0)
    out: dict[str, float] = {
        f"{prefix}rows": float(len(score_s)),
        f"{prefix}clean_rate": _safe_mean(clean),
        f"{prefix}dirty_positive_rate": _safe_mean(dirty),
        f"{prefix}first_touch_available_rate": _safe_mean(ft_available),
        f"{prefix}first_pass_good_rate": _safe_mean(first_good),
        f"{prefix}first_pass_bad_rate": _safe_mean(first_bad),
        f"{prefix}mean_u_all": _safe_mean(raw_u),
        f"{prefix}mean_ev_all": _safe_mean(ev_u),
        f"{prefix}mean_first_touch_net_all": _safe_mean(first_touch_net),
        f"{prefix}mean_first_touch_gross_all": _safe_mean(first_touch_gross),
        f"{prefix}mean_first_touch_gross_minus_1pct_all": _safe_mean(first_touch_gross_minus_1pct),
        f"{prefix}mean_first_touch_executable_margin_all": _safe_mean(first_touch_executable_margin),
    }
    valid = score_s.notna().to_numpy(dtype=bool)
    valid_idx = np.flatnonzero(valid)
    if not len(valid_idx):
        return out
    order = valid_idx[np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")]
    for frac in TOP_FRACS:
        tag = f"top{int(round(frac * 100)):02d}"
        k = max(1, int(math.ceil(float(frac) * len(order))))
        idx = order[:k]
        selected_clean = clean.iloc[idx]
        out[f"{prefix}{tag}_rows"] = float(k)
        out[f"{prefix}{tag}_clean_precision"] = _safe_mean(selected_clean)
        selected_weight = gross_abs_weight.iloc[idx]
        selected_gross = first_touch_gross.iloc[idx]
        selected_first_good = first_good.iloc[idx]
        weighted_denom = float(selected_weight.sum())
        weighted_clean_num = float(
            (
                selected_first_good.reset_index(drop=True)
                * selected_gross.clip(lower=0.0).reset_index(drop=True)
            ).sum()
        )
        ev_weighted_first_touch_precision = (
            weighted_clean_num / weighted_denom if weighted_denom > 1e-12 else float("nan")
        )
        out[f"{prefix}{tag}_ev_weighted_first_touch_precision"] = ev_weighted_first_touch_precision
        out[f"{prefix}{tag}_ev_weighted_clean_precision"] = ev_weighted_first_touch_precision
        out[f"{prefix}{tag}_mean_first_touch_net"] = _safe_mean(first_touch_net.iloc[idx])
        out[f"{prefix}{tag}_mean_first_touch_gross"] = _safe_mean(selected_gross)
        out[f"{prefix}{tag}_mean_first_touch_gross_minus_1pct"] = _safe_mean(
            first_touch_gross_minus_1pct.iloc[idx]
        )
        out[f"{prefix}{tag}_mean_first_touch_executable_margin"] = _safe_mean(
            first_touch_executable_margin.iloc[idx]
        )
        out[f"{prefix}{tag}_hit_first_touch_gross_minus_1pct"] = _safe_mean(
            first_touch_gross_minus_1pct.iloc[idx].gt(0.0)
        )
        out[f"{prefix}{tag}_hit_first_touch_executable_margin"] = _safe_mean(
            first_touch_executable_margin.iloc[idx].gt(0.0)
        )
        out[f"{prefix}{tag}_first_pass_good_rate"] = _safe_mean(first_good.iloc[idx])
        out[f"{prefix}{tag}_first_pass_bad_rate"] = _safe_mean(first_bad.iloc[idx])
        out[f"{prefix}{tag}_clean_lift"] = (
            out[f"{prefix}{tag}_clean_precision"] / max(out[f"{prefix}clean_rate"], 1e-12)
            if math.isfinite(out.get(f"{prefix}{tag}_clean_precision", float("nan")))
            else float("nan")
        )
        out[f"{prefix}{tag}_dirty_positive_rate"] = _safe_mean(dirty.iloc[idx])
        out[f"{prefix}{tag}_bad_mae_rate"] = _safe_mean(mae.iloc[idx].ge(1.0))
        out[f"{prefix}{tag}_mean_mae_norm"] = _safe_mean(mae.iloc[idx])
        out[f"{prefix}{tag}_p90_mae_norm"] = float(mae.iloc[idx].quantile(0.90)) if len(idx) else float("nan")
        out[f"{prefix}{tag}_first_touch_bad_mae_1r_rate"] = _safe_mean(first_touch_mae_norm.iloc[idx].ge(1.0))
        out[f"{prefix}{tag}_mean_first_touch_mae_norm"] = _safe_mean(first_touch_mae_norm.iloc[idx])
        out[f"{prefix}{tag}_p90_first_touch_mae_norm"] = (
            float(first_touch_mae_norm.iloc[idx].quantile(0.90)) if len(idx) else float("nan")
        )
        out[f"{prefix}{tag}_first_touch_full_path_bad_mae_1r_rate"] = _safe_mean(
            first_touch_full_path_mae_norm.iloc[idx].ge(1.0)
        )
        out[f"{prefix}{tag}_mean_first_touch_full_path_mae_norm"] = _safe_mean(
            first_touch_full_path_mae_norm.iloc[idx]
        )
        out[f"{prefix}{tag}_p90_first_touch_full_path_mae_norm"] = (
            float(first_touch_full_path_mae_norm.iloc[idx].quantile(0.90)) if len(idx) else float("nan")
        )
        out[f"{prefix}{tag}_mean_first_touch_mae_to_sl"] = _safe_mean(first_touch_mae.iloc[idx])
        out[f"{prefix}{tag}_p90_first_touch_mae_to_sl"] = (
            float(first_touch_mae.iloc[idx].quantile(0.90)) if len(idx) else float("nan")
        )
        out[f"{prefix}{tag}_mean_underwater_bars_before_mfe"] = _safe_mean(underwater_bars.iloc[idx])
        out[f"{prefix}{tag}_mean_underwater_fraction_before_mfe"] = _safe_mean(underwater_fraction.iloc[idx])
        out[f"{prefix}{tag}_mfe_1r_before_mae_1r_rate"] = _safe_mean(mfe_before_mae_1r.iloc[idx])
        out[f"{prefix}{tag}_mae_1r_before_mfe_1r_rate"] = _safe_mean(mae_before_mfe_1r.iloc[idx])
        out[f"{prefix}{tag}_mean_max_adverse_before_mfe_1r"] = _safe_mean(max_adverse_before_mfe_1r.iloc[idx])
        out[f"{prefix}{tag}_timeout_rate"] = _safe_mean(timeout.iloc[idx].gt(0.5))
        out[f"{prefix}{tag}_mean_u"] = _safe_mean(raw_u.iloc[idx])
        out[f"{prefix}{tag}_mean_ev"] = _safe_mean(ev_u.iloc[idx])
        out[f"{prefix}{tag}_hit_u"] = _safe_mean(raw_u.iloc[idx].gt(0.0))
        out[f"{prefix}{tag}_hit_ev"] = _safe_mean(ev_u.iloc[idx].gt(0.0))
        out[f"{prefix}{tag}_hit_first_touch_net"] = _safe_mean(first_touch_net.iloc[idx].gt(0.0))
    return out


def _score_fold(
    score: pd.Series,
    valid_label: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    month: str,
    *,
    round_trip_cost: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {"month": month}
    out.update(_top_metrics(score=score, label=valid_label, metrics=valid_metrics, round_trip_cost=round_trip_cost))
    side = pd.to_numeric(valid_metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    for side_name, mask in (("long", side.ge(0.0)), ("short", side.lt(0.0))):
        if int(mask.sum()) < 100:
            continue
        idx = np.flatnonzero(mask.to_numpy(dtype=bool))
        out.update(
            _top_metrics(
                score=score.iloc[idx].reset_index(drop=True),
                label=valid_label.iloc[idx].reset_index(drop=True),
                metrics=valid_metrics.iloc[idx].reset_index(drop=True),
                round_trip_cost=round_trip_cost,
                prefix=f"{side_name}_",
            )
        )
    return out


def _objective(folds: list[dict[str, Any]], *, objective_mode: str) -> float:
    df = pd.DataFrame(folds)
    if df.empty:
        return float("-inf")
    mode = str(objective_mode or "balanced").strip().lower()
    if mode == "pnl_only":
        ev10 = pd.to_numeric(df["top10_mean_ev"], errors="coerce")
        ev20 = pd.to_numeric(df["top20_mean_ev"], errors="coerce")
        ev30 = pd.to_numeric(df["top30_mean_ev"], errors="coerce")
        worst10 = float(ev10.min()) if len(ev10.dropna()) else float("nan")
        objective = (
            1.00 * _safe_mean(ev10)
            + 0.60 * _safe_mean(ev20)
            + 0.35 * _safe_mean(ev30)
        )
        if math.isfinite(worst10):
            objective += 0.75 * worst10
        return float(objective) if math.isfinite(objective) else float("-inf")
    if mode == "path_ordered":
        ev10 = pd.to_numeric(df["top10_mean_ev"], errors="coerce")
        ev20 = pd.to_numeric(df["top20_mean_ev"], errors="coerce")
        good10 = pd.to_numeric(df["top10_first_pass_good_rate"], errors="coerce")
        good20 = pd.to_numeric(df["top20_first_pass_good_rate"], errors="coerce")
        bad_first10 = pd.to_numeric(df["top10_first_pass_bad_rate"], errors="coerce")
        bad_mae10 = pd.to_numeric(df["top10_bad_mae_rate"], errors="coerce")
        bad_mae20 = pd.to_numeric(df["top20_bad_mae_rate"], errors="coerce")
        p90_mae10 = pd.to_numeric(df["top10_p90_mae_norm"], errors="coerce")
        ft_bad_mae10 = pd.to_numeric(
            df.get("top10_first_touch_bad_mae_1r_rate", bad_mae10),
            errors="coerce",
        )
        ft_p90_mae10 = pd.to_numeric(
            df.get("top10_p90_first_touch_mae_norm", p90_mae10),
            errors="coerce",
        )
        mean_mae10 = pd.to_numeric(df["top10_mean_mae_norm"], errors="coerce")
        mfe_before10 = pd.to_numeric(
            df.get("top10_mfe_1r_before_mae_1r_rate", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        mae_before10 = pd.to_numeric(
            df.get("top10_mae_1r_before_mfe_1r_rate", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        max_adv_before10 = pd.to_numeric(
            df.get("top10_mean_max_adverse_before_mfe_1r", mean_mae10),
            errors="coerce",
        )
        underwater_frac10 = pd.to_numeric(
            df.get("top10_mean_underwater_fraction_before_mfe", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        timeout10 = pd.to_numeric(df["top10_timeout_rate"], errors="coerce")
        underwater10 = pd.to_numeric(df["top10_mean_underwater_bars_before_mfe"], errors="coerce")
        long_ev10 = pd.to_numeric(df.get("long_top10_mean_ev", pd.Series(np.nan, index=df.index)), errors="coerce")
        short_ev10 = pd.to_numeric(df.get("short_top10_mean_ev", pd.Series(np.nan, index=df.index)), errors="coerce")
        side_worst_ev = pd.concat([long_ev10, short_ev10], axis=1).min(axis=1)
        worst_ev10 = float(ev10.min()) if len(ev10.dropna()) else float("nan")
        obj = (
            20.0 * _safe_mean(ev10)
            + 10.0 * _safe_mean(ev20)
            + 0.55 * _safe_mean(good10)
            + 0.25 * _safe_mean(good20)
            - 0.55 * _safe_mean(bad_first10)
            - 0.45 * _safe_mean(bad_mae10)
            - 0.20 * _safe_mean(bad_mae20)
            - 0.08 * _safe_mean(np.maximum(p90_mae10 - 3.0, 0.0))
            - 0.25 * _safe_mean(mae_before10)
            - 0.20 * _safe_mean(np.maximum(max_adv_before10 - 1.5, 0.0))
            - 0.15 * _safe_mean(np.maximum(underwater_frac10 - 0.45, 0.0))
            - 0.25 * _safe_mean(timeout10)
            - 0.08 * _safe_mean(np.maximum(underwater10 - 6.0, 0.0))
            - 0.12 * _safe_mean(np.maximum(underwater10 - 10.0, 0.0))
            + 8.0 * _safe_mean(side_worst_ev)
        )
        if math.isfinite(worst_ev10) and worst_ev10 < 0.0:
            obj += 25.0 * worst_ev10
        reject_penalty = 0.0
        if _safe_mean(p90_mae10) > 3.0:
            reject_penalty += 0.25 * (_safe_mean(p90_mae10) - 3.0)
        if _safe_mean(bad_first10) > 0.40:
            reject_penalty += 0.35 * (_safe_mean(bad_first10) - 0.40)
        if _safe_mean(bad_mae10) > 0.55:
            reject_penalty += 0.30 * (_safe_mean(bad_mae10) - 0.55)
        if _safe_mean(max_adv_before10) > 1.50:
            reject_penalty += 0.30 * (_safe_mean(max_adv_before10) - 1.50)
        if _safe_mean(underwater_frac10) > 0.45:
            reject_penalty += 0.25 * (_safe_mean(underwater_frac10) - 0.45)
        if _safe_mean(underwater10) > 10.0:
            reject_penalty += 0.20 * (_safe_mean(underwater10) - 10.0)
        obj -= reject_penalty
        return float(obj) if math.isfinite(obj) else float("-inf")
    if mode == "precision_topk":
        clean10 = pd.to_numeric(df["top10_clean_precision"], errors="coerce")
        clean20 = pd.to_numeric(df["top20_clean_precision"], errors="coerce")
        clean30 = pd.to_numeric(df["top30_clean_precision"], errors="coerce")
        evw_clean10 = pd.to_numeric(
            df.get("top10_ev_weighted_first_touch_precision", df.get("top10_ev_weighted_clean_precision", clean10)),
            errors="coerce",
        )
        evw_clean20 = pd.to_numeric(
            df.get("top20_ev_weighted_first_touch_precision", df.get("top20_ev_weighted_clean_precision", clean20)),
            errors="coerce",
        )
        evw_clean30 = pd.to_numeric(
            df.get("top30_ev_weighted_first_touch_precision", df.get("top30_ev_weighted_clean_precision", clean30)),
            errors="coerce",
        )
        ft_net10 = pd.to_numeric(
            df.get("top10_mean_first_touch_net", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        ft_net20 = pd.to_numeric(
            df.get("top20_mean_first_touch_net", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        ft_net30 = pd.to_numeric(
            df.get("top30_mean_first_touch_net", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        good10 = pd.to_numeric(df["top10_first_pass_good_rate"], errors="coerce")
        good20 = pd.to_numeric(df["top20_first_pass_good_rate"], errors="coerce")
        good30 = pd.to_numeric(df["top30_first_pass_good_rate"], errors="coerce")
        bad_first10 = pd.to_numeric(df["top10_first_pass_bad_rate"], errors="coerce")
        bad_first20 = pd.to_numeric(df["top20_first_pass_bad_rate"], errors="coerce")
        bad_mae10 = pd.to_numeric(df["top10_bad_mae_rate"], errors="coerce")
        bad_mae20 = pd.to_numeric(df["top20_bad_mae_rate"], errors="coerce")
        p90_mae10 = pd.to_numeric(df["top10_p90_mae_norm"], errors="coerce")
        ft_bad_mae10 = pd.to_numeric(
            df.get("top10_first_touch_bad_mae_1r_rate", bad_mae10),
            errors="coerce",
        )
        ft_p90_mae10 = pd.to_numeric(
            df.get("top10_p90_first_touch_mae_norm", p90_mae10),
            errors="coerce",
        )
        mean_mae10 = pd.to_numeric(df["top10_mean_mae_norm"], errors="coerce")
        mfe_before10 = pd.to_numeric(
            df.get("top10_mfe_1r_before_mae_1r_rate", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        mae_before10 = pd.to_numeric(
            df.get("top10_mae_1r_before_mfe_1r_rate", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        max_adv_before10 = pd.to_numeric(
            df.get("top10_mean_max_adverse_before_mfe_1r", mean_mae10),
            errors="coerce",
        )
        underwater_frac10 = pd.to_numeric(
            df.get("top10_mean_underwater_fraction_before_mfe", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        underwater10 = pd.to_numeric(
            df.get("top10_mean_underwater_bars_before_mfe", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        timeout10 = pd.to_numeric(df["top10_timeout_rate"], errors="coerce")
        long_clean10 = pd.to_numeric(
            df.get("long_top10_clean_precision", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        short_clean10 = pd.to_numeric(
            df.get("short_top10_clean_precision", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        long_evw_clean10 = pd.to_numeric(
            df.get(
                "long_top10_ev_weighted_first_touch_precision",
                df.get("long_top10_ev_weighted_clean_precision", long_clean10),
            ),
            errors="coerce",
        )
        short_evw_clean10 = pd.to_numeric(
            df.get(
                "short_top10_ev_weighted_first_touch_precision",
                df.get("short_top10_ev_weighted_clean_precision", short_clean10),
            ),
            errors="coerce",
        )
        long_good10 = pd.to_numeric(
            df.get("long_top10_first_pass_good_rate", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        short_good10 = pd.to_numeric(
            df.get("short_top10_first_pass_good_rate", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        side_min_clean = pd.concat([long_clean10, short_clean10], axis=1).min(axis=1)
        side_min_evw_clean = pd.concat([long_evw_clean10, short_evw_clean10], axis=1).min(axis=1)
        side_min_good = pd.concat([long_good10, short_good10], axis=1).min(axis=1)
        side_gap_good = (long_good10 - short_good10).abs()
        worst_ft_net10 = float(ft_net10.min()) if len(ft_net10.dropna()) else float("nan")
        obj = (
            1.30 * _safe_mean(evw_clean10)
            + 0.95 * _safe_mean(evw_clean20)
            + 0.65 * _safe_mean(evw_clean30)
            + 0.35 * _safe_mean(clean10)
            + 0.25 * _safe_mean(clean20)
            + 0.15 * _safe_mean(clean30)
            + 0.55 * _safe_mean(good10)
            + 0.35 * _safe_mean(good20)
            + 0.20 * _safe_mean(good30)
            + 0.40 * _safe_mean(mfe_before10)
            + 3.00 * _safe_mean(ft_net10)
            + 1.50 * _safe_mean(ft_net20)
            + 0.75 * _safe_mean(ft_net30)
            + 0.50 * _safe_mean(side_min_evw_clean)
            + 0.20 * _safe_mean(side_min_clean)
            + 0.25 * _safe_mean(side_min_good)
            - 0.35 * _safe_mean(bad_first10)
            - 0.20 * _safe_mean(bad_first20)
            - 0.45 * _safe_mean(mae_before10)
            - 0.20 * _safe_mean(ft_bad_mae10)
            - 0.10 * _safe_mean(bad_mae10)
            - 0.12 * _safe_mean(bad_mae20)
            - 0.08 * _safe_mean(timeout10)
            - 0.14 * _safe_mean(np.maximum(mean_mae10 - 1.50, 0.0))
            - 0.18 * _safe_mean(np.maximum(ft_p90_mae10 - 1.50, 0.0))
            - 0.12 * _safe_mean(np.maximum(p90_mae10 - 3.0, 0.0))
            - 0.16 * _safe_mean(np.maximum(max_adv_before10 - 1.0, 0.0))
            - 0.18 * _safe_mean(np.maximum(underwater_frac10 - 0.45, 0.0))
            - 0.08 * _safe_mean(np.maximum(underwater10 - 6.0, 0.0))
            - 0.12 * _safe_mean(np.maximum(underwater10 - 10.0, 0.0))
            - 0.12 * _safe_mean(side_gap_good)
        )
        if _safe_mean(ft_net10) < 0.0:
            obj += 4.00 * _safe_mean(ft_net10)
        if math.isfinite(worst_ft_net10) and worst_ft_net10 < 0.0:
            obj += 2.00 * worst_ft_net10
        if _safe_mean(mae_before10) > 0.40:
            obj -= 0.40 * (_safe_mean(mae_before10) - 0.40)
        if _safe_mean(bad_mae10) > 0.55:
            obj -= 0.35 * (_safe_mean(bad_mae10) - 0.55)
        if _safe_mean(ft_bad_mae10) > 0.50:
            obj -= 0.35 * (_safe_mean(ft_bad_mae10) - 0.50)
        if _safe_mean(ft_p90_mae10) > 1.50:
            obj -= 0.20 * (_safe_mean(ft_p90_mae10) - 1.50)
        if _safe_mean(p90_mae10) > 3.0:
            obj -= 0.20 * (_safe_mean(p90_mae10) - 3.0)
        if _safe_mean(max_adv_before10) > 1.50:
            obj -= 0.30 * (_safe_mean(max_adv_before10) - 1.50)
        if _safe_mean(underwater_frac10) > 0.45:
            obj -= 0.25 * (_safe_mean(underwater_frac10) - 0.45)
        if _safe_mean(underwater10) > 10.0:
            obj -= 0.20 * (_safe_mean(underwater10) - 10.0)
        return float(obj) if math.isfinite(obj) else float("-inf")
    top10 = pd.to_numeric(df["top10_clean_precision"], errors="coerce")
    top20 = pd.to_numeric(df["top20_clean_precision"], errors="coerce")
    top30 = pd.to_numeric(df["top30_clean_precision"], errors="coerce")
    bad10 = pd.to_numeric(df["top10_bad_mae_rate"], errors="coerce")
    bad20 = pd.to_numeric(df["top20_bad_mae_rate"], errors="coerce")
    timeout10 = pd.to_numeric(df["top10_timeout_rate"], errors="coerce")
    mean_ev10 = pd.to_numeric(df["top10_mean_ev"], errors="coerce")
    long10 = pd.to_numeric(df.get("long_top10_clean_precision", pd.Series(np.nan, index=df.index)), errors="coerce")
    short10 = pd.to_numeric(df.get("short_top10_clean_precision", pd.Series(np.nan, index=df.index)), errors="coerce")
    side_min = pd.concat([long10, short10], axis=1).min(axis=1)
    side_gap = (long10 - short10).abs()
    worst_ev = float(mean_ev10.min()) if len(mean_ev10.dropna()) else float("nan")
    obj = (
        1.00 * _safe_mean(top10)
        + 0.70 * _safe_mean(top20)
        + 0.40 * _safe_mean(top30)
        + 0.65 * _safe_mean(side_min)
        + 0.35 * float(top10.min())
        - 0.65 * _safe_mean(bad10)
        - 0.30 * _safe_mean(bad20)
        - 0.25 * _safe_mean(timeout10)
        - 0.20 * _safe_mean(side_gap)
        + 20.0 * _safe_mean(mean_ev10)
    )
    if math.isfinite(worst_ev) and worst_ev < 0.0:
        obj += 30.0 * worst_ev
    return float(obj) if math.isfinite(obj) else float("-inf")


def _summarize_trial(
    stage: str,
    trial_number: int,
    config: LabelConfig,
    fold_rows: list[dict[str, Any]],
    *,
    objective_mode: str,
) -> dict[str, Any]:
    df = pd.DataFrame(fold_rows)
    summary: dict[str, Any] = {
        "stage": stage,
        "trial_number": int(trial_number),
        "label_name": config.name,
        "family": config.family,
        "objective_mode": str(objective_mode),
        "objective": _objective(fold_rows, objective_mode=objective_mode),
        "folds": int(len(fold_rows)),
    }
    for col in (
        "top10_clean_precision",
        "top10_ev_weighted_first_touch_precision",
        "top10_ev_weighted_clean_precision",
        "top20_ev_weighted_first_touch_precision",
        "top20_ev_weighted_clean_precision",
        "top30_ev_weighted_first_touch_precision",
        "top30_ev_weighted_clean_precision",
        "top20_clean_precision",
        "top30_clean_precision",
        "top10_clean_lift",
        "top10_first_pass_good_rate",
        "top20_first_pass_good_rate",
        "top10_first_pass_bad_rate",
        "top10_bad_mae_rate",
        "top10_first_touch_bad_mae_1r_rate",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top20_bad_mae_rate",
        "top10_mean_mae_norm",
        "top10_p90_mae_norm",
        "top10_mean_first_touch_mae_norm",
        "top10_p90_first_touch_mae_norm",
        "top10_mean_first_touch_full_path_mae_norm",
        "top10_p90_first_touch_full_path_mae_norm",
        "top10_mean_first_touch_mae_to_sl",
        "top10_p90_first_touch_mae_to_sl",
        "top10_mean_underwater_bars_before_mfe",
        "top10_mean_underwater_fraction_before_mfe",
        "top10_mfe_1r_before_mae_1r_rate",
        "top10_mae_1r_before_mfe_1r_rate",
        "top10_mean_max_adverse_before_mfe_1r",
        "top10_timeout_rate",
        "top10_mean_ev",
        "top10_mean_first_touch_net",
        "top10_mean_first_touch_gross",
        "top10_mean_first_touch_gross_minus_1pct",
        "top10_mean_first_touch_executable_margin",
        "top10_hit_first_touch_net",
        "top10_hit_first_touch_gross_minus_1pct",
        "top10_hit_first_touch_executable_margin",
        "top10_mean_u",
        "top10_hit_u",
        "top10_hit_ev",
        "top20_mean_first_touch_net",
        "top20_mean_first_touch_gross_minus_1pct",
        "top20_mean_first_touch_executable_margin",
        "top20_mean_ev",
        "top20_hit_first_touch_gross_minus_1pct",
        "top20_hit_first_touch_executable_margin",
        "top20_mean_u",
        "top20_hit_u",
        "top20_hit_ev",
        "top30_mean_first_touch_net",
        "top30_mean_first_touch_gross_minus_1pct",
        "top30_mean_first_touch_executable_margin",
        "top30_mean_ev",
        "top30_hit_first_touch_gross_minus_1pct",
        "top30_hit_first_touch_executable_margin",
        "top30_mean_u",
        "top30_hit_u",
        "top30_hit_ev",
        "long_top10_clean_precision",
        "short_top10_clean_precision",
        "long_top10_ev_weighted_first_touch_precision",
        "short_top10_ev_weighted_first_touch_precision",
        "long_top10_ev_weighted_clean_precision",
        "short_top10_ev_weighted_clean_precision",
        "long_top10_first_pass_good_rate",
        "short_top10_first_pass_good_rate",
        "long_top10_first_pass_bad_rate",
        "short_top10_first_pass_bad_rate",
        "long_top10_bad_mae_rate",
        "short_top10_bad_mae_rate",
        "long_top10_first_touch_full_path_bad_mae_1r_rate",
        "short_top10_first_touch_full_path_bad_mae_1r_rate",
        "long_top10_mfe_1r_before_mae_1r_rate",
        "short_top10_mfe_1r_before_mae_1r_rate",
        "long_top10_mae_1r_before_mfe_1r_rate",
        "short_top10_mae_1r_before_mfe_1r_rate",
        "long_top10_mean_first_touch_net",
        "short_top10_mean_first_touch_net",
        "long_top10_mean_first_touch_executable_margin",
        "short_top10_mean_first_touch_executable_margin",
        "long_top10_hit_first_touch_executable_margin",
        "short_top10_hit_first_touch_executable_margin",
        "long_top10_mean_ev",
        "short_top10_mean_ev",
        "long_top10_hit_first_touch_net",
        "short_top10_hit_first_touch_net",
        "long_top10_hit_u",
        "short_top10_hit_u",
        "long_top10_hit_ev",
        "short_top10_hit_ev",
    ):
        if col in df.columns:
            summary[f"mean_{col}"] = _safe_mean(df[col])
            summary[f"min_{col}"] = float(pd.to_numeric(df[col], errors="coerce").min())
    for prefix, params in (("long", config.long), ("short", config.short)):
        for key, value in asdict(params).items():
            summary[f"{prefix}_{key}"] = value
    return summary


def _prepare_folds(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    months: list[str],
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_feature_store_features: int | None,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    seed: int,
    ae_gmm_state_feature_seed: int = 42,
    ae_gmm_fold_cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = _load_labels(labels_path)
    frame, symbol_filter, _symbols = _apply_spread_symbol_universe(
        frame,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        target_symbol_count=target_symbol_count,
        max_spread_bps=None,
    )
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        feature_matrix = feature_matrix.astype(np.float32, copy=False).reset_index(drop=True)
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix], axis=1, copy=False)
    frame = frame.reset_index(drop=True)
    metrics = _path_metrics(frame).reset_index(drop=True)
    features = _feature_columns(frame)
    month_s = frame["__ts__"].dt.to_period("M").astype(str)
    if not months:
        months = sorted(month_s.dropna().unique().tolist())[1:]
    folds: list[dict[str, Any]] = []
    for i, month in enumerate(months):
        train_mask = month_s < month
        valid_mask = month_s == month
        if int(train_mask.sum()) < 1000 or int(valid_mask.sum()) < 200:
            continue
        train = frame.loc[train_mask].reset_index(drop=True)
        valid = frame.loc[valid_mask].reset_index(drop=True)
        train_metrics = metrics.loc[train_mask].reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].reset_index(drop=True)
        x_train = train[features].astype(np.float32, copy=False).reset_index(drop=True)
        x_valid = valid[features].astype(np.float32, copy=False).reset_index(drop=True)
        random_state = _ae_gmm_state_feature_random_state(
            fold_i=int(i),
            seed=int(ae_gmm_state_feature_seed),
        )
        cache_payload = _ae_gmm_fold_cache_payload(
            labels_path=labels_path,
            feature_dir=feature_dir,
            feature_list_csv=feature_list_csv,
            month=str(month),
            fold_i=int(i),
            train=train,
            valid=valid,
            features=features,
            include_ae_gmm_state_features=bool(include_ae_gmm_state_features),
            ae_gmm_state_feature_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            ae_gmm_state_feature_max_iter=int(ae_gmm_state_feature_max_iter),
            seed=int(ae_gmm_state_feature_seed),
            random_state=int(random_state),
        )
        cache_key = _fold_cache_digest(cache_payload)
        cache_hit = None
        if bool(include_ae_gmm_state_features) and ae_gmm_fold_cache_dir is not None:
            cache_hit = _load_ae_gmm_fold_cache(
                cache_dir=Path(ae_gmm_fold_cache_dir),
                digest=cache_key,
                expected_payload=cache_payload,
            )
        if cache_hit is not None:
            x_train, x_valid, generated, ae_diag = cache_hit
        else:
            x_train, x_valid, generated, ae_diag = _append_fold_ae_gmm_state_features(
                x_train=x_train,
                x_valid=x_valid,
                train_frame=train,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                enabled=bool(include_ae_gmm_state_features),
                max_train_rows=int(ae_gmm_state_feature_max_train_rows),
                gmm_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
                ae_max_iter=int(ae_gmm_state_feature_max_iter),
                random_state=int(random_state),
            )
            ae_diag["ae_gmm_state_feature_cache_status"] = "miss" if bool(include_ae_gmm_state_features) else "disabled"
            ae_diag["ae_gmm_state_feature_cache_key"] = str(cache_key)
            if bool(include_ae_gmm_state_features) and ae_gmm_fold_cache_dir is not None:
                _write_ae_gmm_fold_cache(
                    cache_dir=Path(ae_gmm_fold_cache_dir),
                    digest=cache_key,
                    payload=cache_payload,
                    x_train=x_train,
                    x_valid=x_valid,
                    generated=generated,
                    ae_diag=ae_diag,
                )
        folds.append(
            {
                "month": month,
                "x_train": x_train,
                "x_valid": x_valid,
                "train_frame": train,
                "valid_frame": valid,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "ae_gmm_generated_features": int(len(generated)),
                "ae_gmm_status": ae_diag.get("ae_gmm_state_feature_status"),
                "ae_gmm_cache_status": ae_diag.get("ae_gmm_state_feature_cache_status"),
                "ae_gmm_cache_key": ae_diag.get("ae_gmm_state_feature_cache_key"),
            }
        )
    manifest = {
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "features": int(len(features)),
        "feature_store": feature_report,
        "symbol_universe_filter": symbol_filter,
        "fold_months": [fold["month"] for fold in folds],
        "fold_count": int(len(folds)),
        "model_seed": int(seed),
        "ae_gmm_state_feature_seed": int(ae_gmm_state_feature_seed),
        "ae_gmm_fold_cache_dir": str(ae_gmm_fold_cache_dir) if ae_gmm_fold_cache_dir is not None else None,
        "ae_gmm_fold_cache_statuses": [fold.get("ae_gmm_cache_status") for fold in folds],
    }
    return folds, manifest


def _evaluate_config(
    *,
    folds: list[dict[str, Any]],
    config: LabelConfig,
    max_train_rows: int,
    seed: int,
    stage: str,
    trial_number: int,
    round_trip_cost: float,
    objective_mode: str,
    path_order_mode: str,
    target_utility_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for fold_i, fold in enumerate(folds):
        train_label = _make_side_soft_label(
            fold["train_metrics"],
            config,
            round_trip_cost=round_trip_cost,
            path_order_mode=path_order_mode,
            target_utility_mode=target_utility_mode,
        )
        valid_label = _make_side_soft_label(
            fold["valid_metrics"],
            config,
            round_trip_cost=round_trip_cost,
            path_order_mode=path_order_mode,
            target_utility_mode=target_utility_mode,
        )
        weights = _sample_weight(
            fold["train_metrics"],
            train_label,
            round_trip_cost=round_trip_cost,
            target_utility_mode=target_utility_mode,
        )
        x_train, y_train, w_train = _cap_train_rows(
            fold["x_train"],
            train_label["target_soft"],
            weights,
            max_rows=int(max_train_rows),
        )
        score = _fit_predict(
            x_train,
            y_train,
            w_train,
            fold["x_valid"],
            seed=int(seed) + 10_000 * int(trial_number) + fold_i,
        )
        row = _score_fold(score, valid_label, fold["valid_metrics"], fold["month"], round_trip_cost=round_trip_cost)
        row.update(
            {
                "stage": stage,
                "trial_number": int(trial_number),
                "label_name": config.name,
                "family": config.family,
                "train_rows": int(len(x_train)),
                "train_rows_uncapped": int(fold["train_rows"]),
                "valid_rows": int(fold["valid_rows"]),
                "ae_gmm_generated_features": int(fold["ae_gmm_generated_features"]),
                "ae_gmm_status": fold["ae_gmm_status"],
                "round_trip_cost": float(round_trip_cost),
                "path_order_mode": str(path_order_mode),
                "target_utility_mode": str(target_utility_mode),
            }
        )
        fold_rows.append(row)
    return _summarize_trial(stage, trial_number, config, fold_rows, objective_mode=objective_mode), fold_rows


def _trial_optuna(
    *,
    stage: str,
    base: LabelConfig,
    side: str | None,
    trial: Any,
    trial_number: int,
) -> LabelConfig:
    if stage == "family":
        family = trial.suggest_categorical("family", list(FAMILIES))
        config = _family_config(str(family))
        scale = float(trial.suggest_float("family_jitter_scale", 0.0, 0.45))
        rng = np.random.default_rng(19_000 + int(trial_number))
        return LabelConfig(
            name=f"s51_family_{trial_number:03d}_{family}",
            family=str(family),
            long=_jitter_params(rng, config.long, side="long", scale=scale),
            short=_jitter_params(rng, config.short, side="short", scale=scale),
        )
    if stage == "long_refine":
        return LabelConfig(
            name=f"s51_long_refine_{trial_number:03d}",
            family=base.family,
            long=_suggest_side_params(trial, base.long, side="long", prefix="long", radius=0.55),
            short=base.short,
        )
    if stage == "short_refine":
        return LabelConfig(
            name=f"s51_short_refine_{trial_number:03d}",
            family=base.family,
            long=base.long,
            short=_suggest_side_params(trial, base.short, side="short", prefix="short", radius=0.55),
        )
    if stage == "joint_polish":
        return LabelConfig(
            name=f"s51_joint_polish_{trial_number:03d}",
            family=base.family,
            long=_suggest_side_params(trial, base.long, side="long", prefix="long", radius=0.20),
            short=_suggest_side_params(trial, base.short, side="short", prefix="short", radius=0.20),
        )
    raise ValueError(stage)


def _run_stage(
    *,
    folds: list[dict[str, Any]],
    stage: str,
    base: LabelConfig,
    side: str | None,
    n_trials: int,
    max_train_rows: int,
    seed: int,
    trial_start: int,
    round_trip_cost: float,
    objective_mode: str,
    path_order_mode: str,
    target_utility_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelConfig, int]:
    summaries: list[dict[str, Any]] = []
    fold_rows_all: list[dict[str, Any]] = []
    configs: dict[int, LabelConfig] = {}
    trial_number = int(trial_start)
    try:
        import optuna
    except Exception:
        optuna = None
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            nonlocal trial_number
            config = _trial_optuna(stage=stage, base=base, side=side, trial=trial, trial_number=trial_number)
            summary, folds_rows = _evaluate_config(
                folds=folds,
                config=config,
                max_train_rows=max_train_rows,
                seed=seed,
                stage=stage,
                trial_number=trial_number,
                round_trip_cost=round_trip_cost,
                objective_mode=objective_mode,
                path_order_mode=path_order_mode,
                target_utility_mode=target_utility_mode,
            )
            configs[int(trial_number)] = config
            summaries.append(summary)
            fold_rows_all.extend(folds_rows)
            trial_number += 1
            return float(summary["objective"])

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed) + trial_start))
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    else:
        class _RandomTrial:
            def __init__(self, rng: np.random.Generator) -> None:
                self.rng = rng

            def suggest_float(self, name: str, low: float, high: float) -> float:
                return float(self.rng.uniform(float(low), float(high)))

            def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
                return choices[int(self.rng.integers(0, len(choices)))]

        rng = np.random.default_rng(int(seed) + trial_start)
        for _ in range(int(n_trials)):
            trial = _RandomTrial(rng)
            config = _trial_optuna(stage=stage, base=base, side=side, trial=trial, trial_number=trial_number)
            summary, folds_rows = _evaluate_config(
                folds=folds,
                config=config,
                max_train_rows=max_train_rows,
                seed=seed,
                stage=stage,
                trial_number=trial_number,
                round_trip_cost=round_trip_cost,
                objective_mode=objective_mode,
                path_order_mode=path_order_mode,
                target_utility_mode=target_utility_mode,
            )
            configs[int(trial_number)] = config
            summaries.append(summary)
            fold_rows_all.extend(folds_rows)
            trial_number += 1
    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows_all)
    best_trial = int(summary_df.iloc[0]["trial_number"]) if not summary_df.empty else -1
    best_config = configs.get(best_trial, base)
    return summary_df, fold_df, best_config, trial_number


def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any], best: dict[str, Any]) -> None:
    def table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "rank",
        "stage",
        "trial_number",
        "label_name",
        "family",
        "objective",
        "mean_top10_clean_precision",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_ev_weighted_clean_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_clean_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_clean_precision",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_pass_bad_rate",
        "mean_top20_clean_precision",
        "mean_top30_clean_precision",
        "mean_top10_bad_mae_rate",
        "mean_top10_first_touch_bad_mae_1r_rate",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_p90_mae_norm",
        "mean_top10_mean_first_touch_mae_norm",
        "mean_top10_p90_first_touch_mae_norm",
        "mean_top10_mean_underwater_bars_before_mfe",
        "mean_top10_mean_underwater_fraction_before_mfe",
        "mean_top10_mean_max_adverse_before_mfe_1r",
        "mean_top10_timeout_rate",
        "mean_top10_mean_ev",
        "mean_top10_mean_u",
        "mean_top10_hit_u",
        "mean_top10_hit_ev",
        "mean_long_top10_clean_precision",
        "mean_short_top10_clean_precision",
        "mean_long_top10_ev_weighted_first_touch_precision",
        "mean_short_top10_ev_weighted_first_touch_precision",
    ]
    fold_cols = [
        "stage",
        "label_name",
        "month",
        "top10_clean_precision",
        "top10_ev_weighted_first_touch_precision",
        "top10_ev_weighted_clean_precision",
        "top10_first_pass_good_rate",
        "top10_first_pass_bad_rate",
        "top20_clean_precision",
        "top30_clean_precision",
        "top10_bad_mae_rate",
        "top10_first_touch_bad_mae_1r_rate",
        "top10_mfe_1r_before_mae_1r_rate",
        "top10_mae_1r_before_mfe_1r_rate",
        "top10_p90_mae_norm",
        "top10_mean_first_touch_mae_norm",
        "top10_p90_first_touch_mae_norm",
        "top10_mean_underwater_bars_before_mfe",
        "top10_mean_underwater_fraction_before_mfe",
        "top10_mean_max_adverse_before_mfe_1r",
        "top10_timeout_rate",
        "top10_mean_ev",
        "top10_mean_u",
        "top10_hit_u",
        "top10_hit_ev",
        "long_top10_clean_precision",
        "short_top10_clean_precision",
        "long_top10_ev_weighted_first_touch_precision",
        "short_top10_ev_weighted_first_touch_precision",
        "long_top10_first_pass_good_rate",
        "short_top10_first_pass_good_rate",
        "long_top10_first_pass_bad_rate",
        "short_top10_first_pass_bad_rate",
        "long_top10_mfe_1r_before_mae_1r_rate",
        "short_top10_mfe_1r_before_mae_1r_rate",
        "long_top10_mae_1r_before_mfe_1r_rate",
        "short_top10_mae_1r_before_mfe_1r_rate",
    ]
    best_name = str(summary.iloc[0]["label_name"]) if not summary.empty else ""
    lines = [
        "# S51 Hierarchical Side-Specific Soft-Label HPO",
        "",
        "Scope: Gate 3 soft-label construction HPO. Search is hierarchical: family -> long refine -> short refine -> joint polish.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Months: `{', '.join(manifest['fold_months'])}`",
        f"Features: `{manifest['features']}`",
        "",
        "Primary metrics are top-k clean precision, bad-MAE, timeout, and top-k utility. AUC is intentionally not used.",
        "",
        "## Winner",
        "",
        table(summary.head(1), cols),
        "",
        "## Trial Ranking",
        "",
        table(summary, cols, limit=50),
        "",
        "## Winner Fold Detail",
        "",
        table(folds[folds["label_name"].eq(best_name)], fold_cols),
        "",
        "## Best Parameters",
        "",
        "```json",
        json.dumps(_json_safe(best), indent=2),
        "```",
    ]
    (output_dir / "s51_side_soft_label_hpo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hpo(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_feature_store_features: int | None,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    max_train_rows: int,
    family_trials: int,
    long_trials: int,
    short_trials: int,
    joint_trials: int,
    seed: int,
    round_trip_cost: float,
    objective_mode: str,
    path_order_mode: str,
    target_utility_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        months=months,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        target_symbol_count=target_symbol_count,
        max_feature_store_features=max_feature_store_features,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
        seed=seed,
    )
    if not folds:
        raise RuntimeError("No valid folds prepared")
    trial_start = 0
    base = _family_config("long_fast_short_controlled")
    stage_results: list[pd.DataFrame] = []
    fold_results: list[pd.DataFrame] = []
    stage_plan = [
        ("family", None, int(family_trials)),
        ("long_refine", "long", int(long_trials)),
        ("short_refine", "short", int(short_trials)),
        ("joint_polish", None, int(joint_trials)),
    ]
    for stage, side, n_trials in stage_plan:
        if n_trials <= 0:
            continue
        summary_df, fold_df, base, trial_start = _run_stage(
            folds=folds,
            stage=stage,
            base=base,
            side=side,
            n_trials=n_trials,
            max_train_rows=max_train_rows,
            seed=seed,
            trial_start=trial_start,
            round_trip_cost=round_trip_cost,
            objective_mode=objective_mode,
            path_order_mode=path_order_mode,
            target_utility_mode=target_utility_mode,
        )
        summary_df.to_csv(output_dir / f"s51_{stage}_trials.csv", index=False)
        fold_df.to_csv(output_dir / f"s51_{stage}_folds.csv", index=False)
        stage_results.append(summary_df)
        fold_results.append(fold_df)
    summary_all = pd.concat(stage_results, ignore_index=True).sort_values("objective", ascending=False).reset_index(drop=True)
    summary_all.insert(0, "rank", np.arange(1, len(summary_all) + 1, dtype=np.int32))
    folds_all = pd.concat(fold_results, ignore_index=True)
    best_summary = summary_all.iloc[0].to_dict()
    best_trial = int(best_summary["trial_number"])
    best_config = base
    # Recover the best config by rerunning the last best if necessary from the serialized summary.
    # The complete parameter values are stored in the summary row, so construct directly.
    long_kwargs = {field: float(best_summary[f"long_{field}"]) for field in SideParams.__dataclass_fields__}
    short_kwargs = {field: float(best_summary[f"short_{field}"]) for field in SideParams.__dataclass_fields__}
    best_config = LabelConfig(
        name=str(best_summary["label_name"]),
        family=str(best_summary["family"]),
        long=SideParams(**long_kwargs),
        short=SideParams(**short_kwargs),
    )
    best_payload = {
        "summary": best_summary,
        "config": asdict(best_config),
    }
    paths = {
        "summary": output_dir / "s51_side_soft_label_hpo_trials.csv",
        "folds": output_dir / "s51_side_soft_label_hpo_folds.csv",
        "best": output_dir / "s51_side_soft_label_hpo_best.json",
        "manifest": output_dir / "manifest.json",
    }
    summary_all.to_csv(paths["summary"], index=False)
    folds_all.to_csv(paths["folds"], index=False)
    paths["best"].write_text(json.dumps(_json_safe(best_payload), indent=2), encoding="utf-8")
    manifest.update(
        {
            "scope": "s51_hierarchical_side_soft_label_hpo",
            "labels_path": labels_path,
            "feature_dir": feature_dir,
            "feature_list_csv": feature_list_csv,
            "output_dir": output_dir,
            "stage_trials": {
                "family": int(family_trials),
                "long_refine": int(long_trials),
                "short_refine": int(short_trials),
                "joint_polish": int(joint_trials),
            },
            "total_trials": int(len(summary_all)),
            "max_train_rows": int(max_train_rows),
            "round_trip_cost": float(round_trip_cost),
            "objective_mode": str(objective_mode),
            "path_order_mode": str(path_order_mode),
            "target_utility_mode": str(target_utility_mode),
            "min_net_edge_search": "disabled_fixed",
            "fixed_min_net_edge": {
                "long": float(FIXED_LONG_MIN_NET_EDGE),
                "short": float(FIXED_SHORT_MIN_NET_EDGE),
            },
            "model": {
                "type": "ExtraTreesRegressor",
                "n_estimators": 128,
                "max_depth": 9,
                "min_samples_leaf": 35,
                "max_features": "sqrt",
            },
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(output_dir, summary_all, folds_all, manifest, best_payload)
    return {"best": best_payload, "manifest": manifest, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", type=str, default="")
    parser.add_argument("--spread-baseline-path", type=Path, default=None)
    parser.add_argument("--spread-rank-column", type=str, default="p75_spread_bps")
    parser.add_argument("--target-symbol-count", type=int, default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER)
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument("--family-trials", type=int, default=12)
    parser.add_argument("--long-trials", type=int, default=16)
    parser.add_argument("--short-trials", type=int, default=16)
    parser.add_argument("--joint-trials", type=int, default=12)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument(
        "--objective-mode",
        choices=("balanced", "pnl_only", "path_ordered", "precision_topk"),
        default="balanced",
        help="HPO objective. pnl_only optimizes only top-k EV/PnL after round-trip cost; risk metrics are diagnostics only.",
    )
    parser.add_argument(
        "--path-order-mode",
        choices=("legacy", "s52_first_touch"),
        default="legacy",
        help="legacy keeps S51 behavior; s52_first_touch rewards favorable first passage and caps dirty adverse-first paths.",
    )
    parser.add_argument(
        "--target-utility-mode",
        choices=(
            "net_after_cost",
            "raw_positive",
            "geometry_only",
            "first_touch_net",
            "first_touch_net_after_cost",
            "first_touch_ev",
            "first_touch_executable_net",
        ),
        default="net_after_cost",
        help=(
            "Controls whether S52 target construction requires cost-adjusted policy utility, raw positive utility, "
            "first-touch executable net, or only clean TP/SL path geometry."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_hpo(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, ()),
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=args.spread_rank_column,
        target_symbol_count=args.target_symbol_count,
        max_feature_store_features=args.max_feature_store_features,
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        max_train_rows=int(args.max_train_rows),
        family_trials=int(args.family_trials),
        long_trials=int(args.long_trials),
        short_trials=int(args.short_trials),
        joint_trials=int(args.joint_trials),
        seed=int(args.seed),
        round_trip_cost=float(args.round_trip_cost),
        objective_mode=str(args.objective_mode),
        path_order_mode=str(args.path_order_mode),
        target_utility_mode=str(args.target_utility_mode),
    )
    print(json.dumps(_json_safe(result["best"]["summary"]), indent=2))


if __name__ == "__main__":
    main()
