#!/usr/bin/env python3
"""S52 timestamp x side ranking smoke.

This diagnostic tests the next step after path-ordered label construction:
whether a true cross-sectional ranker can order clean first-passage candidates
better than a pointwise regressor under the same S52 soft labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from lightgbm import LGBMRanker, LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when dependency missing.
    LGBMRanker = None
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False

from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_ROUND_TRIP_COST,
    LabelConfig,
    SideParams,
    _json_safe,
    _make_side_soft_label,
    _objective,
    _prepare_folds,
    _sample_weight,
    _score_fold,
    _summarize_trial,
)


STATE_FEATURE_TOKENS = (
    "gmm",
    "cluster",
    "mahal",
    "entropy",
    "reconstruction",
    "latent_speed",
    "latent_acceleration",
    "cluster_speed",
    "cluster_acceleration",
    "state_spectral",
    "ae_gmm_oof_available",
)
LEDGER_METRIC_COLUMNS = (
    "u_policy_net",
    "ret_net",
    "side",
    "is_timeout",
    "mae_norm",
    "mfe_norm",
    "first_touch_gross",
    "first_touch_net",
    "first_touch_mae_norm",
    "first_touch_mfe_norm",
    "first_touch_full_path_mae_norm",
    "mfe_1r_before_mae_1r",
    "mae_1r_before_mfe_1r",
    "max_adverse_before_mfe_1r",
    "underwater_bars_before_mfe_1r",
    "underwater_fraction_before_mfe_1r",
)
LEDGER_TOP_FRACS = (0.10, 0.20, 0.30)


DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260704_s52_bidirectional_first_touch_tp075_sl075_fast16_bar50_cost100bps_ordercols_v2_labels/"
    "labels"
)
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_BEST_CONFIG = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "s52_precision_topk_pathorder_ftmae_tp075_sl075_fast16_bar50_cost100bps_hpo_v1/"
    "s51_side_soft_label_hpo_best.json"
)
DEFAULT_RANKER_OUTPUT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "s52_timestamp_side_ranker_smoke_v1"
)
DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    "n_estimators": 140,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 35,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
}


def _load_ranker_params(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ranker params must be a JSON object: {path}")
    allowed = set(DEFAULT_RANKER_PARAMS) | {"reg_alpha", "min_split_gain", "max_depth", "subsample_freq"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown ranker params in {path}: {unknown}")
    return dict(payload)


def _ranker_model_params(*, seed: int, ranker_params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(DEFAULT_RANKER_PARAMS)
    if ranker_params:
        params.update(ranker_params)
    params.update(
        {
            "objective": "lambdarank",
            "random_state": int(seed),
            "n_jobs": 2,
            "verbosity": -1,
        }
    )
    return params


def _pointwise_model_params(*, seed: int, model_params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = {
        "n_estimators": 180,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 45,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
    }
    if model_params:
        params.update({k: v for k, v in model_params.items() if k in DEFAULT_RANKER_PARAMS})
    params.update(
        {
            "objective": "regression",
            "random_state": int(seed),
            "n_jobs": 2,
            "verbosity": -1,
        }
    )
    return params


def _ranker_sample_weight(
    metrics: pd.DataFrame,
    label: pd.DataFrame,
    *,
    round_trip_cost: float,
    target_utility_mode: str = "raw_positive",
    mode: str = "base",
) -> pd.Series:
    base = _sample_weight(
        metrics,
        label,
        round_trip_cost=round_trip_cost,
        target_utility_mode=target_utility_mode,
    ).astype(np.float32)
    mode_norm = str(mode or "base").strip().lower()
    if mode_norm in {"", "base", "default"}:
        return base.reset_index(drop=True)

    index = metrics.index
    side = pd.to_numeric(metrics.get("side", pd.Series(1.0, index=index)), errors="coerce").fillna(1.0)
    is_long = side.ge(0.0).astype(float)
    clean = pd.to_numeric(label.get("target_hard", pd.Series(0.0, index=label.index)), errors="coerce").fillna(0.0)
    dirty = pd.to_numeric(label.get("dirty_positive", pd.Series(0.0, index=label.index)), errors="coerce").fillna(0.0)
    positive = pd.to_numeric(label.get("positive_u", pd.Series(0.0, index=label.index)), errors="coerce").fillna(0.0)
    first_good = pd.to_numeric(label.get("first_pass_good", pd.Series(0.0, index=label.index)), errors="coerce").fillna(0.0)
    first_bad = pd.to_numeric(label.get("first_pass_bad", pd.Series(0.0, index=label.index)), errors="coerce").fillna(0.0)
    mae_before = pd.to_numeric(
        metrics.get("mae_1r_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    ft_bad_mae = (
        pd.to_numeric(metrics.get("first_touch_mae_norm", pd.Series(0.0, index=index)), errors="coerce")
        .fillna(0.0)
        .ge(1.0)
        .astype(float)
    )
    adv = pd.to_numeric(
        metrics.get("max_adverse_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    underwater = pd.to_numeric(
        metrics.get("underwater_bars_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    timeout = pd.to_numeric(
        metrics.get("first_touch_timeout", metrics.get("is_timeout", pd.Series(0.0, index=index))),
        errors="coerce",
    ).fillna(0.0).gt(0.5).astype(float)

    w = pd.Series(1.0, index=metrics.index, dtype=np.float64)
    if mode_norm == "execres_clean_dirty":
        w = (
            w
            + 2.00 * clean
            + 1.50 * dirty
            + 1.50 * positive * ft_bad_mae
            + 0.75 * positive * timeout
            + 1.00 * first_good
            + 1.00 * first_bad
        )
    elif mode_norm == "long_clean_dirty":
        long_dirty_path = np.maximum(adv - 1.0, 0.0).clip(upper=3.0) + np.maximum(underwater - 8.0, 0.0).clip(upper=24.0) / 12.0
        w = (
            w
            + is_long * (3.00 * clean + 2.50 * dirty + 2.00 * ft_bad_mae + 1.50 * mae_before + 0.75 * long_dirty_path)
            + (1.0 - is_long) * (1.25 * clean + 1.00 * dirty + 0.75 * first_bad)
            + 0.75 * first_good
        )
    else:
        raise ValueError(f"unknown sample weight mode: {mode}")

    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.10, 6.0)
    w = w / max(float(w.mean()), 1e-12)
    return w.astype(np.float32).reset_index(drop=True)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_config(path: Path) -> LabelConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("config", payload)
    return LabelConfig(
        name=str(cfg["name"]),
        family=str(cfg["family"]),
        long=SideParams(**cfg["long"]),
        short=SideParams(**cfg["short"]),
    )


def _cap_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_rows, size=int(max_rows), replace=False)
    return np.sort(idx.astype(np.int64))


def _group_order(frame: pd.DataFrame, *, mode: str) -> tuple[np.ndarray, np.ndarray]:
    local = frame.reset_index(drop=True)
    ts = pd.to_datetime(local["__ts__"], errors="coerce").astype("int64")
    if str(mode) == "timestamp_side":
        side = pd.to_numeric(local.get("__side__", local.get("side", 1.0)), errors="coerce").fillna(1.0)
        side_key = np.where(side.to_numpy(dtype=np.float64) < 0.0, -1, 1)
        keys = pd.DataFrame({"ts": ts, "side": side_key})
        order = np.lexsort((keys["side"].to_numpy(), keys["ts"].to_numpy()))
        sorted_keys = keys.iloc[order].astype({"ts": "int64", "side": "int64"})
        counts = sorted_keys.groupby(["ts", "side"], sort=False, observed=True).size().to_numpy(dtype=np.int32)
    else:
        order = np.argsort(ts.to_numpy(dtype=np.int64), kind="mergesort")
        sorted_ts = ts.iloc[order].to_numpy(dtype=np.int64)
        _, counts = np.unique(sorted_ts, return_counts=True)
        counts = counts.astype(np.int32, copy=False)
    return order.astype(np.int64, copy=False), counts


def _rank_relevance(
    label: pd.DataFrame,
    metrics: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    group_mode: str,
    relevance_mode: str = "first_touch",
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
) -> np.ndarray:
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    side = pd.to_numeric(metrics["side"], errors="coerce").fillna(1.0)
    if str(group_mode) == "timestamp_side":
        group_key = pd.Series(
            ts.astype(str).to_numpy() + "|" + np.where(side.to_numpy(dtype=np.float64) < 0.0, "S", "L"),
            index=label.index,
        )
    else:
        group_key = pd.Series(ts.astype(str).to_numpy(), index=label.index)

    soft = pd.to_numeric(label["target_soft"], errors="coerce").fillna(0.0)
    first_good = pd.to_numeric(label.get("first_pass_good", 0.0), errors="coerce").fillna(0.0)
    first_bad = pd.to_numeric(label.get("first_pass_bad", 0.0), errors="coerce").fillna(0.0)
    mae_before = pd.to_numeric(metrics.get("mae_1r_before_mfe_1r", 0.0), errors="coerce").fillna(0.0)
    zero_metric = pd.Series(0.0, index=metrics.index)
    underwater_bars = pd.to_numeric(
        metrics.get("underwater_bars_before_mfe_1r", zero_metric),
        errors="coerce",
    ).fillna(0.0)
    underwater_fraction = pd.to_numeric(
        metrics.get("underwater_fraction_before_mfe_1r", zero_metric),
        errors="coerce",
    ).fillna(0.0)
    max_adverse_before = pd.to_numeric(
        metrics.get("max_adverse_before_mfe_1r", zero_metric),
        errors="coerce",
    ).fillna(0.0)
    ft_bad_mae = (
        pd.to_numeric(metrics.get("first_touch_mae_norm", zero_metric), errors="coerce")
        .fillna(0.0)
        .ge(1.0)
        .astype(float)
    )
    full_path_mae = pd.to_numeric(
        metrics.get("first_touch_full_path_mae_norm", metrics.get("mae_norm", zero_metric)),
        errors="coerce",
    ).fillna(0.0)
    full_path_bad_mae = full_path_mae.ge(1.0).astype(float)
    full_path_excess = (full_path_mae - 1.0).clip(lower=0.0, upper=4.0)
    timeout = pd.to_numeric(
        metrics.get("first_touch_timeout", metrics.get("is_timeout", zero_metric)),
        errors="coerce",
    ).fillna(0.0)
    first_touch_net = pd.to_numeric(
        metrics.get("first_touch_net", metrics.get("u_policy_net", zero_metric)),
        errors="coerce",
    ).fillna(0.0)
    first_touch_gross = first_touch_net + float(round_trip_cost)
    fullpath_aware = str(relevance_mode).strip().lower() in {
        "fullpath",
        "full_path",
        "full_path_clean",
        "first_touch_full_path",
        "fullpath_evpath",
        "evpath_fullpath",
    }
    evpath_aware = str(relevance_mode).strip().lower() in {
        "evpath",
        "ev_path",
        "gross_evpath",
        "fullpath_evpath",
        "evpath_fullpath",
    }
    cleangross_aware = str(relevance_mode).strip().lower() in {
        "cleangross",
        "clean_gross",
        "clean_first_touch_gross",
    }
    ordered_clean_ev_aware = str(relevance_mode).strip().lower() in {
        "ordered_clean_ev",
        "path_ordered_clean_ev",
        "strict_ordered_clean_ev",
    }
    soft_ordered_ev_aware = str(relevance_mode).strip().lower() in {
        "soft_ordered_ev",
        "ordered_ev",
        "path_ordered_ev",
    }
    soft_exec_ordered_ev_aware = str(relevance_mode).strip().lower() in {
        "soft_exec_ordered_ev",
        "soft_executable_ordered_ev",
        "soft_net_ordered_ev",
        "path_ordered_soft_net_ev",
    }
    soft_breadth_ordered_ev_aware = str(relevance_mode).strip().lower() in {
        "soft_breadth_ordered_ev",
        "breadth_ordered_ev",
        "path_ordered_breadth_ev",
        "s52_soft_breadth_ordered_ev",
    }
    firstpass_exec_ev_aware = str(relevance_mode).strip().lower() in {
        "firstpass_exec_ev",
        "first_pass_exec_ev",
        "s52_firstpass_exec_ev",
        "path_ordered_firstpass_exec_ev",
    }
    ev_preserving_ordered_aware = str(relevance_mode).strip().lower() in {
        "ev_preserving_ordered",
        "ev_preserving_ordered_ev",
        "s52_ev_preserving_ordered",
        "path_ordered_ev_preserving",
    }
    exec_ordered_ev_aware = str(relevance_mode).strip().lower() in {
        "exec_ordered_ev",
        "executable_ordered_ev",
        "net_ordered_ev",
        "path_ordered_net_ev",
    }
    pathnet_guarded_aware = str(relevance_mode).strip().lower() in {
        "pathnet_guarded",
        "path_net_guarded",
        "fullpath_net_guarded",
        "s52_pathnet_guarded",
        "path_ordered_fullpath_net_guarded",
    }
    score = (
        soft
        + 0.35 * first_good
        - 0.45 * first_bad
        - 0.45 * mae_before
        - 0.45 * ft_bad_mae
        - 0.45 * np.maximum(max_adverse_before - 1.0, 0.0).clip(upper=3.0)
        - 0.25 * np.maximum(underwater_fraction - 0.35, 0.0).clip(upper=1.0)
        - 0.08 * np.maximum(underwater_bars - 6.0, 0.0).clip(upper=24.0)
        - 0.15 * timeout
    )
    slow_path = underwater_bars.gt(10.0) | underwater_fraction.gt(0.45) | max_adverse_before.gt(1.5)
    dirty_path = slow_path | ft_bad_mae.gt(0.5) | mae_before.gt(0.5)
    score = score.mask(dirty_path, np.minimum(score, 0.05 + 0.12 * first_good))
    clean_ordered = first_good.gt(0.5) & ~dirty_path & first_touch_gross.gt(0.0)
    if ordered_clean_ev_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        clean_strict = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.0)
            & underwater_bars.le(8.0)
            & underwater_fraction.le(0.35)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        clean_loose = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & ~dirty_path
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        hard_bad = (
            first_bad.gt(0.5)
            | dirty_path
            | dirty_positive
            | timeout.gt(0.5)
            | first_touch_gross.le(0.0)
        )
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score.mask(clean_loose, 1.0 + gross_rank)
        score = score.mask(
            clean_strict,
            2.0
            + 1.25 * gross_rank
            + 0.50 * first_touch_net.gt(0.0).astype(float)
            + 0.25 * soft.clip(lower=0.0, upper=1.0),
        )
        score = score.mask(hard_bad & ~clean_strict, 0.0)
        relevance = np.zeros(len(score), dtype=np.int32)
        values = score.to_numpy(dtype=np.float64)
        relevance[values > 0.0] = 1
        relevance[values >= 1.5] = 2
        relevance[values >= 2.5] = 3
        relevance[values >= 3.5] = 4
        return np.clip(relevance, 0, 4)
    if soft_breadth_ordered_ev_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        net_positive = first_touch_net.clip(lower=0.0)
        net_rank = net_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        gross_strength = (first_touch_gross / cost_scale).clip(lower=-1.0, upper=3.0)
        net_strength = (first_touch_net / cost_scale).clip(lower=-1.0, upper=3.0)
        ordered_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.00)
            & underwater_bars.le(8.0)
            & underwater_fraction.le(0.35)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        ordered_near_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.25)
            & underwater_bars.le(10.0)
            & underwater_fraction.le(0.45)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        hard_dirty = (
            first_bad.gt(0.5)
            | mae_before.gt(0.5)
            | ft_bad_mae.gt(0.5)
            | max_adverse_before.gt(1.50)
            | underwater_bars.gt(12.0)
            | underwater_fraction.gt(0.50)
            | timeout.gt(0.5)
        )
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        path_pain = (
            0.55 * np.maximum(max_adverse_before - 0.75, 0.0).clip(upper=3.0)
            + 0.10 * np.maximum(underwater_bars - 6.0, 0.0).clip(upper=24.0)
            + 0.35 * np.maximum(underwater_fraction - 0.30, 0.0).clip(upper=1.0)
        )
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.20 * soft.clip(lower=0.0, upper=1.0)
        score = score + 0.30 * first_good.astype(float)
        score = score + 0.85 * gross_rank * ordered_near_clean.astype(float)
        score = score + 0.55 * net_rank * ordered_near_clean.astype(float)
        score = score + 0.20 * gross_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score + 0.20 * net_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score + 1.05 * ordered_near_clean.astype(float)
        score = score + 0.95 * ordered_clean.astype(float)
        score = score - path_pain
        score = score - 1.40 * hard_dirty.astype(float)
        score = score - 1.00 * dirty_positive.astype(float)
        score = score.mask(hard_dirty | dirty_positive, np.minimum(score, 0.05 + 0.05 * first_good))
        nonpositive_cleanish = ordered_near_clean & first_touch_net.le(0.0)
        score = score.mask(nonpositive_cleanish, np.minimum(score, 0.95 + 0.15 * first_good))
        score = score.mask(ordered_near_clean, np.maximum(score, 1.35 + 0.75 * gross_rank + 0.55 * net_rank))
        score = score.mask(
            ordered_clean,
            np.maximum(
                score,
                2.35
                + 0.85 * gross_rank
                + 0.75 * net_rank
                + 0.20 * gross_strength.clip(lower=0.0)
                + 0.20 * net_strength.clip(lower=0.0),
            ),
        )
        pct = score.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        relevance = np.floor(pct.to_numpy(dtype=np.float32) * 5.0).astype(np.int32)
        return np.clip(relevance, 0, 4)
    if firstpass_exec_ev_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        net_positive = first_touch_net.clip(lower=0.0)
        net_rank = net_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        gross_strength = (first_touch_gross / cost_scale).clip(lower=-1.0, upper=3.0)
        net_strength = (first_touch_net / cost_scale).clip(lower=-1.0, upper=3.0)
        ordered_loose = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.25)
            & underwater_bars.le(10.0)
            & underwater_fraction.le(0.45)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        ordered_near = (
            ordered_loose
            & max_adverse_before.le(1.00)
            & underwater_bars.le(8.0)
            & underwater_fraction.le(0.35)
        )
        ordered_clean = (
            ordered_near
            & max_adverse_before.le(0.75)
            & underwater_bars.le(6.0)
            & underwater_fraction.le(0.30)
        )
        hard_dirty = (
            first_bad.gt(0.5)
            | mae_before.gt(0.5)
            | ft_bad_mae.gt(0.5)
            | max_adverse_before.gt(1.25)
            | underwater_bars.gt(10.0)
            | underwater_fraction.gt(0.45)
            | timeout.gt(0.5)
            | first_touch_gross.le(0.0)
        )
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        path_pain = (
            0.85 * np.maximum(max_adverse_before - 0.60, 0.0).clip(upper=3.0)
            + 0.14 * np.maximum(underwater_bars - 4.0, 0.0).clip(upper=24.0)
            + 0.55 * np.maximum(underwater_fraction - 0.25, 0.0).clip(upper=1.0)
        )
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.15 * soft.clip(lower=0.0, upper=1.0)
        score = score + 1.00 * ordered_loose.astype(float)
        score = score + 0.90 * ordered_near.astype(float)
        score = score + 0.80 * ordered_clean.astype(float)
        score = score + 0.80 * gross_rank * ordered_loose.astype(float)
        score = score + 1.05 * net_rank * ordered_near.astype(float)
        score = score + 0.20 * gross_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score + 0.35 * net_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score - path_pain
        score = score - 2.00 * hard_dirty.astype(float)
        score = score - 1.25 * dirty_positive.astype(float)
        score = score.mask(hard_dirty | dirty_positive, np.minimum(score, 0.05))
        score = score.mask(ordered_loose & first_touch_net.le(0.0), np.minimum(score, 1.20))
        score = score.mask(ordered_near & first_touch_net.gt(0.0), np.maximum(score, 2.10 + 1.00 * net_rank))
        score = score.mask(
            ordered_clean & first_touch_net.gt(0.0),
            np.maximum(score, 3.15 + 0.75 * gross_rank + 0.95 * net_rank),
        )
        values = score.to_numpy(dtype=np.float64)
        relevance = np.zeros(len(values), dtype=np.int32)
        relevance[values >= 0.90] = 1
        relevance[values >= 1.80] = 2
        relevance[values >= 2.80] = 3
        relevance[values >= 3.80] = 4
        return np.clip(relevance, 0, 4)
    if ev_preserving_ordered_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        net_positive = first_touch_net.clip(lower=0.0)
        net_rank = net_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        net_strength = (first_touch_net / cost_scale).clip(lower=-2.0, upper=4.0)
        gross_strength = (first_touch_gross / cost_scale).clip(lower=-1.0, upper=4.0)
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        ordered_near_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.25)
            & underwater_bars.le(10.0)
            & underwater_fraction.le(0.45)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        ordered_clean = (
            ordered_near_clean
            & max_adverse_before.le(1.00)
            & underwater_bars.le(8.0)
            & underwater_fraction.le(0.35)
        )
        ordered_ev_clean = ordered_clean & first_touch_net.gt(0.0)
        ordered_ev_strong = (
            ordered_clean
            & first_touch_net.gt(0.25 * cost_scale)
            & max_adverse_before.le(0.85)
            & underwater_bars.le(6.0)
            & underwater_fraction.le(0.30)
        )
        hard_dirty = (
            first_bad.gt(0.5)
            | mae_before.gt(0.5)
            | ft_bad_mae.gt(0.5)
            | max_adverse_before.gt(1.50)
            | underwater_bars.gt(12.0)
            | underwater_fraction.gt(0.50)
            | timeout.gt(0.5)
            | dirty_positive
        )
        path_pain = (
            0.60 * np.maximum(max_adverse_before - 0.75, 0.0).clip(upper=3.0)
            + 0.08 * np.maximum(underwater_bars - 6.0, 0.0).clip(upper=24.0)
            + 0.35 * np.maximum(underwater_fraction - 0.30, 0.0).clip(upper=1.0)
        )
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.15 * soft.clip(lower=0.0, upper=1.0)
        score = score + 0.45 * first_good.astype(float)
        score = score + 0.80 * ordered_near_clean.astype(float)
        score = score + 0.70 * ordered_clean.astype(float)
        score = score + 0.95 * net_rank * ordered_near_clean.astype(float)
        score = score + 0.45 * gross_rank * ordered_near_clean.astype(float)
        score = score + 0.55 * net_strength.clip(lower=0.0) * ordered_ev_clean.astype(float)
        score = score + 0.20 * gross_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score - 0.55 * first_touch_net.le(0.0).astype(float) * ordered_near_clean.astype(float)
        score = score - path_pain
        score = score - 1.75 * hard_dirty.astype(float)
        score = score.mask(hard_dirty, np.minimum(score, 0.05))
        score = score.mask(ordered_near_clean & first_touch_net.le(0.0), np.minimum(score, 0.85))
        score = score.mask(ordered_ev_clean, np.maximum(score, 1.55 + 1.00 * net_rank + 0.40 * gross_rank))
        score = score.mask(
            ordered_ev_strong,
            np.maximum(
                score,
                2.55
                + 0.95 * net_rank
                + 0.50 * gross_rank
                + 0.35 * net_strength.clip(lower=0.0),
            ),
        )
        values = score.to_numpy(dtype=np.float64)
        relevance = np.zeros(len(values), dtype=np.int32)
        relevance[values >= 0.80] = 1
        relevance[values >= 1.50] = 2
        relevance[values >= 2.50] = 3
        relevance[values >= 3.50] = 4
        return np.clip(relevance, 0, 4)
    if soft_ordered_ev_aware or soft_exec_ordered_ev_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        net_positive = first_touch_net.clip(lower=0.0)
        net_rank = net_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        gross_strength = (first_touch_gross / cost_scale).clip(lower=-1.0, upper=3.0)
        net_strength = (first_touch_net / cost_scale).clip(lower=-1.0, upper=3.0)
        ordered_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.25)
            & underwater_bars.le(10.0)
            & underwater_fraction.le(0.45)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        ordered_near_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.60)
            & underwater_bars.le(14.0)
            & underwater_fraction.le(0.55)
            & timeout.lt(0.5)
            & first_touch_gross.gt(0.0)
        )
        hard_dirty = (
            first_bad.gt(0.5)
            | mae_before.gt(0.5)
            | ft_bad_mae.gt(0.5)
            | max_adverse_before.gt(1.75)
            | underwater_bars.gt(18.0)
            | underwater_fraction.gt(0.65)
            | timeout.gt(0.5)
        )
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.25 * soft.clip(lower=0.0, upper=1.0)
        score = score + 0.35 * first_good.astype(float)
        score = score + 0.90 * gross_rank * ordered_near_clean.astype(float)
        score = score + 0.25 * gross_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score + 1.00 * ordered_near_clean.astype(float)
        score = score + 0.75 * ordered_clean.astype(float)
        if soft_exec_ordered_ev_aware:
            score = score + 0.65 * net_rank * ordered_near_clean.astype(float)
            score = score + 0.25 * net_strength.clip(lower=0.0) * ordered_clean.astype(float)
            score = score - 0.35 * first_touch_net.le(0.0).astype(float) * ordered_near_clean.astype(float)
        score = score - 1.15 * hard_dirty.astype(float)
        score = score - 0.80 * dirty_positive.astype(float)
        score = score - 0.45 * np.maximum(max_adverse_before - 1.0, 0.0).clip(upper=3.0)
        score = score - 0.20 * np.maximum(underwater_fraction - 0.35, 0.0).clip(upper=1.0)
        score = score - 0.05 * np.maximum(underwater_bars - 8.0, 0.0).clip(upper=24.0)
        score = score.mask(hard_dirty | dirty_positive, np.minimum(score, 0.15 + 0.08 * first_good))
        if soft_exec_ordered_ev_aware:
            nonpositive_cleanish = ordered_near_clean & first_touch_net.le(0.0)
            score = score.mask(nonpositive_cleanish, np.minimum(score, 1.15 + 0.20 * first_good))
        pct = score.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        relevance = np.floor(pct.to_numpy(dtype=np.float32) * 5.0).astype(np.int32)
        return np.clip(relevance, 0, 4)
    if exec_ordered_ev_aware:
        net_positive = first_touch_net.clip(lower=0.0)
        net_rank = net_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        net_strength = (first_touch_net / cost_scale).clip(lower=-1.0, upper=3.0)
        ordered_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.25)
            & underwater_bars.le(10.0)
            & underwater_fraction.le(0.45)
            & timeout.lt(0.5)
            & first_touch_net.gt(0.0)
        )
        ordered_near_clean = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & max_adverse_before.le(1.60)
            & underwater_bars.le(14.0)
            & underwater_fraction.le(0.55)
            & timeout.lt(0.5)
            & first_touch_net.gt(0.0)
        )
        hard_dirty = (
            first_bad.gt(0.5)
            | mae_before.gt(0.5)
            | ft_bad_mae.gt(0.5)
            | max_adverse_before.gt(1.75)
            | underwater_bars.gt(18.0)
            | underwater_fraction.gt(0.65)
            | timeout.gt(0.5)
            | first_touch_net.le(0.0)
        )
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.20 * soft.clip(lower=0.0, upper=1.0)
        score = score + 1.25 * ordered_near_clean.astype(float)
        score = score + 0.90 * ordered_clean.astype(float)
        score = score + 1.10 * net_rank * ordered_near_clean.astype(float)
        score = score + 0.45 * net_strength.clip(lower=0.0) * ordered_clean.astype(float)
        score = score - 1.35 * hard_dirty.astype(float)
        score = score - 0.90 * dirty_positive.astype(float)
        score = score - 0.55 * np.maximum(max_adverse_before - 1.0, 0.0).clip(upper=3.0)
        score = score - 0.25 * np.maximum(underwater_fraction - 0.35, 0.0).clip(upper=1.0)
        score = score - 0.07 * np.maximum(underwater_bars - 8.0, 0.0).clip(upper=24.0)
        score = score.mask(hard_dirty | dirty_positive, np.minimum(score, 0.05))
        score = score.mask(ordered_near_clean, np.maximum(score, 1.20 + 1.30 * net_rank))
        score = score.mask(
            ordered_clean,
            np.maximum(score, 2.20 + 1.20 * net_rank + 0.35 * net_strength.clip(lower=0.0)),
        )
        relevance = np.zeros(len(score), dtype=np.int32)
        values = score.to_numpy(dtype=np.float64)
        relevance[values > 0.50] = 1
        relevance[values >= 1.50] = 2
        relevance[values >= 2.50] = 3
        relevance[values >= 3.50] = 4
        return np.clip(relevance, 0, 4)
    if pathnet_guarded_aware:
        net_positive = first_touch_net.clip(lower=0.0)
        net_rank = net_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        net_strength = (first_touch_net / cost_scale).clip(lower=-1.0, upper=3.0)
        is_short = side.lt(0.0)
        near_fullpath_cap = pd.Series(np.where(is_short, 1.50, 1.25), index=label.index)
        hard_fullpath_cap = pd.Series(np.where(is_short, 2.50, 2.00), index=label.index)
        near_adverse_cap = pd.Series(np.where(is_short, 1.10, 0.95), index=label.index)
        strict_adverse_cap = pd.Series(np.where(is_short, 0.85, 0.75), index=label.index)
        near_underwater_bars = pd.Series(np.where(is_short, 9.0, 7.0), index=label.index)
        strict_underwater_bars = pd.Series(np.where(is_short, 6.0, 5.0), index=label.index)
        near_underwater_fraction = pd.Series(np.where(is_short, 0.38, 0.32), index=label.index)
        strict_underwater_fraction = pd.Series(np.where(is_short, 0.30, 0.25), index=label.index)
        dirty_positive = pd.to_numeric(
            label.get("dirty_positive", pd.Series(0.0, index=label.index)),
            errors="coerce",
        ).fillna(0.0).gt(0.5)
        ordered_net = (
            first_good.gt(0.5)
            & first_bad.lt(0.5)
            & mae_before.lt(0.5)
            & ft_bad_mae.lt(0.5)
            & timeout.lt(0.5)
            & first_touch_net.gt(0.0)
        )
        path_near = (
            full_path_mae.le(near_fullpath_cap)
            & max_adverse_before.le(near_adverse_cap)
            & underwater_bars.le(near_underwater_bars)
            & underwater_fraction.le(near_underwater_fraction)
        )
        path_strict = (
            full_path_mae.le(1.0)
            & max_adverse_before.le(strict_adverse_cap)
            & underwater_bars.le(strict_underwater_bars)
            & underwater_fraction.le(strict_underwater_fraction)
        )
        exec_near = ordered_net & path_near
        exec_strict = ordered_net & path_strict
        hard_dirty = (
            first_bad.gt(0.5)
            | mae_before.gt(0.5)
            | ft_bad_mae.gt(0.5)
            | timeout.gt(0.5)
            | first_touch_net.le(0.0)
            | full_path_mae.gt(hard_fullpath_cap)
            | max_adverse_before.gt(1.50)
            | underwater_bars.gt(12.0)
            | underwater_fraction.gt(0.50)
            | dirty_positive
        )
        fullpath_quality = (1.0 - (full_path_mae / hard_fullpath_cap).clip(lower=0.0, upper=1.0)).fillna(0.0)
        path_pain = (
            0.95 * np.maximum(max_adverse_before - strict_adverse_cap, 0.0).clip(upper=3.0)
            + 0.12 * np.maximum(underwater_bars - strict_underwater_bars, 0.0).clip(upper=24.0)
            + 0.75 * np.maximum(underwater_fraction - strict_underwater_fraction, 0.0).clip(upper=1.0)
            + 0.70 * np.maximum(full_path_mae - 1.0, 0.0).clip(upper=4.0)
        )
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.10 * soft.clip(lower=0.0, upper=1.0)
        score = score + 1.20 * exec_near.astype(float)
        score = score + 1.40 * exec_strict.astype(float)
        score = score + 1.35 * net_rank * exec_near.astype(float)
        score = score + 0.45 * gross_rank * exec_near.astype(float)
        score = score + 0.45 * net_strength.clip(lower=0.0) * exec_strict.astype(float)
        score = score + 0.55 * fullpath_quality * exec_near.astype(float)
        score = score - path_pain
        score = score - 2.25 * hard_dirty.astype(float)
        score = score.mask(hard_dirty, np.minimum(score, 0.0))
        score = score.mask(ordered_net & ~path_near, np.minimum(score, 0.35))
        score = score.mask(exec_near, np.maximum(score, 1.20 + 1.10 * net_rank + 0.40 * fullpath_quality))
        score = score.mask(
            exec_strict,
            np.maximum(
                score,
                2.60
                + 1.10 * net_rank
                + 0.45 * gross_rank
                + 0.45 * net_strength.clip(lower=0.0)
                + 0.55 * fullpath_quality,
            ),
        )
        values = score.to_numpy(dtype=np.float64)
        relevance = np.zeros(len(values), dtype=np.int32)
        relevance[values >= 0.75] = 1
        relevance[values >= 1.60] = 2
        relevance[values >= 2.70] = 3
        relevance[values >= 3.80] = 4
        return np.clip(relevance, 0, 4)
    if cleangross_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        score = pd.Series(0.0, index=label.index, dtype=np.float64)
        score = score + 0.50 * first_good.astype(float)
        score = score - 0.75 * first_bad.astype(float)
        score = score - 0.50 * dirty_path.astype(float)
        score = score + 2.00 * clean_ordered.astype(float)
        score = score + 1.25 * gross_rank * clean_ordered.astype(float)
        score = score + 0.50 * first_touch_net.gt(0.0).astype(float) * clean_ordered.astype(float)
    if fullpath_aware:
        score = score - 0.45 * full_path_bad_mae - 0.15 * full_path_excess
        dirty_after_touch = first_good.gt(0.5) & full_path_bad_mae.gt(0.5)
        score = score.mask(dirty_after_touch, np.minimum(score, 0.05))
    if evpath_aware:
        gross_positive = first_touch_gross.clip(lower=0.0)
        gross_rank = gross_positive.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
        cost_scale = max(float(round_trip_cost), 1e-6)
        gross_strength = (first_touch_gross / cost_scale).clip(lower=-1.0, upper=3.0)
        cleanish = first_good.gt(0.5) & ~dirty_path
        score = score + 0.45 * gross_rank * cleanish.astype(float)
        score = score + 0.15 * gross_strength.clip(lower=0.0) * cleanish.astype(float)
        score = score - 0.20 * first_touch_gross.le(0.0).astype(float)
    pct = score.groupby(group_key, sort=False).rank(method="average", pct=True).fillna(0.0)
    relevance = np.floor(pct.to_numpy(dtype=np.float32) * 5.0).astype(np.int32)
    return np.clip(relevance, 0, 4)


def _materialized_soft_label(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Use the label artifact's own S52 first-touch target.

    This is needed when the geometry materializer has already encoded the
    path-ordered target. Recomputing labels from a stale HPO config can test a
    different target than the one being validated.
    """
    index = frame.index
    soft = pd.to_numeric(
        frame.get("__first_touch_target_soft__", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    hit = pd.to_numeric(
        frame.get("__first_touch_hit__", metrics.get("first_touch_hit", pd.Series(0.0, index=index))),
        errors="coerce",
    ).fillna(0.0).gt(0.5)
    stop = pd.to_numeric(
        frame.get("__first_touch_stop__", metrics.get("first_touch_stop", pd.Series(0.0, index=index))),
        errors="coerce",
    ).fillna(0.0).gt(0.5)
    timeout = pd.to_numeric(
        frame.get("__first_touch_timeout__", metrics.get("first_touch_timeout", pd.Series(0.0, index=index))),
        errors="coerce",
    ).fillna(0.0).gt(0.5)
    ft_mae = pd.to_numeric(metrics.get("first_touch_mae_norm", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    max_adverse = pd.to_numeric(
        metrics.get("max_adverse_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    underwater_bars = pd.to_numeric(
        metrics.get("underwater_bars_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    underwater_fraction = pd.to_numeric(
        metrics.get("underwater_fraction_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    mfe_before = pd.to_numeric(
        metrics.get("mfe_1r_before_mae_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0).gt(0.5)
    mae_before = pd.to_numeric(
        metrics.get("mae_1r_before_mfe_1r", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0).gt(0.5)
    clean = (
        hit
        & mfe_before
        & ~mae_before
        & ~timeout
        & ft_mae.lt(1.0)
        & max_adverse.le(1.50)
        & underwater_bars.le(10.0)
        & underwater_fraction.le(0.45)
    )
    first_bad = stop | mae_before | ft_mae.ge(1.0) | max_adverse.gt(1.50) | underwater_bars.gt(10.0)
    available = pd.to_numeric(
        metrics.get("first_touch_available", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0).gt(0.5)
    positive = pd.to_numeric(metrics.get("u_policy_net", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0).gt(0.0)
    return pd.DataFrame(
        {
            "target_soft": soft.astype(np.float32),
            "target_hard": clean.astype(np.int8),
            "dirty_positive": (positive & first_bad).astype(np.int8),
            "positive_u": positive.astype(np.int8),
            "first_touch_available": available.astype(np.int8),
            "first_pass_good": (hit & mfe_before & ~mae_before).astype(np.int8),
            "first_pass_bad": first_bad.astype(np.int8),
        }
    )


def _state_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    out: list[str] = []
    for col in columns:
        name = str(col)
        lower = name.lower()
        if any(token in lower for token in STATE_FEATURE_TOKENS):
            out.append(name)
    return list(dict.fromkeys(out))


def _top_mask(score: pd.Series, frac: float) -> np.ndarray:
    values = pd.to_numeric(score, errors="coerce").reset_index(drop=True)
    valid = np.flatnonzero(np.isfinite(values.to_numpy(dtype=np.float64)))
    mask = np.zeros(len(values), dtype=bool)
    if len(valid) == 0:
        return mask
    k = max(1, int(math.ceil(float(frac) * len(valid))))
    order = valid[np.argsort(-values.iloc[valid].to_numpy(dtype=np.float64), kind="mergesort")[:k]]
    mask[order] = True
    return mask


def _scored_ledger(
    *,
    variant: str,
    fold: dict[str, Any],
    score: np.ndarray,
    valid_label: pd.DataFrame,
) -> pd.DataFrame:
    valid_frame = fold["valid_frame"].reset_index(drop=True)
    metrics = fold["valid_metrics"].reset_index(drop=True)
    score_s = pd.Series(score, index=pd.RangeIndex(len(valid_frame)), dtype=np.float32)
    ledger = pd.DataFrame(
        {
            "variant": str(variant),
            "month": str(fold["month"]),
            "__ts__": valid_frame.get("__ts__"),
            "__symbol__": valid_frame.get("__symbol__"),
            "score": score_s,
            "target_soft": pd.to_numeric(valid_label.get("target_soft"), errors="coerce"),
            "target_hard": pd.to_numeric(valid_label.get("target_hard"), errors="coerce"),
            "first_pass_good": pd.to_numeric(valid_label.get("first_pass_good"), errors="coerce"),
            "first_pass_bad": pd.to_numeric(valid_label.get("first_pass_bad"), errors="coerce"),
        }
    )
    side = pd.to_numeric(metrics.get("side", pd.Series(1.0, index=metrics.index)), errors="coerce").fillna(1.0)
    ledger["side_name"] = np.where(side.to_numpy(dtype=np.float64) < 0.0, "short", "long")
    for col in LEDGER_METRIC_COLUMNS:
        if col in metrics.columns:
            ledger[col] = pd.to_numeric(metrics[col], errors="coerce").to_numpy()
    for frac in LEDGER_TOP_FRACS:
        ledger[f"selected_top{int(round(frac * 100)):02d}"] = _top_mask(score_s, frac)
    state_cols = _state_feature_columns(fold["x_valid"].columns)
    if state_cols:
        state_frame = fold["x_valid"].reset_index(drop=True).reindex(columns=state_cols)
        ledger = pd.concat([ledger, state_frame], axis=1, copy=False)
    return ledger


def _safe_mean(series: pd.Series | np.ndarray | list[Any]) -> float:
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    return float(values.mean()) if len(values) else float("nan")


def _safe_quantile(series: pd.Series | np.ndarray | list[Any], q: float) -> float:
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    return float(values.quantile(float(q))) if len(values) else float("nan")


def _bucket_state_feature(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    if len(finite) == 0:
        return pd.Series("missing", index=series.index, dtype=object)
    unique = finite.drop_duplicates()
    name = str(series.name).lower()
    if len(unique) <= 12 or name.endswith("cluster_id") or "cluster_t" in name:
        return values.round(0).astype("Int64").astype(str).replace("<NA>", "missing")
    try:
        bucket = pd.qcut(values, q=5, duplicates="drop")
        return bucket.astype(str).replace("nan", "missing")
    except ValueError:
        return pd.Series("flat", index=series.index, dtype=object)


def _archetype_path_diagnostics(ledger: pd.DataFrame, *, top_col: str = "selected_top10") -> pd.DataFrame:
    state_cols = [col for col in _state_feature_columns(ledger.columns) if col in ledger.columns]
    if ledger.empty or not state_cols:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, variant_frame in ledger.groupby("variant", observed=True, dropna=False):
        for feature in state_cols:
            bucketed = _bucket_state_feature(variant_frame[feature])
            tmp = variant_frame.assign(__bucket__=bucketed)
            for (bucket, side), group in tmp.groupby(["__bucket__", "side_name"], observed=True, dropna=False):
                selected = group[group[top_col].astype(bool)] if top_col in group.columns else group.iloc[:0]
                if len(group) < 100 and len(selected) < 10:
                    continue
                rows.append(
                    {
                        "variant": str(variant),
                        "state_feature": str(feature),
                        "bucket": str(bucket),
                        "side": str(side),
                        "rows": int(len(group)),
                        "selected_rows": int(len(selected)),
                        "selected_share": float(len(selected) / max(len(group), 1)),
                        "all_mfe_before_mae_1r_rate": _safe_mean(group.get("mfe_1r_before_mae_1r", [])),
                        "all_mae_before_mfe_1r_rate": _safe_mean(group.get("mae_1r_before_mfe_1r", [])),
                        "selected_mfe_before_mae_1r_rate": _safe_mean(
                            selected.get("mfe_1r_before_mae_1r", [])
                        ),
                        "selected_mae_before_mfe_1r_rate": _safe_mean(
                            selected.get("mae_1r_before_mfe_1r", [])
                        ),
                        "selected_first_pass_good_rate": _safe_mean(selected.get("first_pass_good", [])),
                        "selected_first_pass_bad_rate": _safe_mean(selected.get("first_pass_bad", [])),
                        "selected_first_touch_bad_mae_1r_rate": _safe_mean(
                            pd.to_numeric(selected.get("first_touch_mae_norm", pd.Series(dtype=float)), errors="coerce").ge(1.0)
                        )
                        if len(selected)
                        else float("nan"),
                        "selected_first_touch_full_path_bad_mae_1r_rate": _safe_mean(
                            pd.to_numeric(
                                selected.get("first_touch_full_path_mae_norm", pd.Series(dtype=float)),
                                errors="coerce",
                            ).ge(1.0)
                        )
                        if len(selected)
                        else float("nan"),
                        "selected_mean_first_touch_mae_norm": _safe_mean(selected.get("first_touch_mae_norm", [])),
                        "selected_p90_first_touch_mae_norm": _safe_quantile(
                            selected.get("first_touch_mae_norm", []),
                            0.90,
                        ),
                        "selected_mean_first_touch_full_path_mae_norm": _safe_mean(
                            selected.get("first_touch_full_path_mae_norm", [])
                        ),
                        "selected_p90_first_touch_full_path_mae_norm": _safe_quantile(
                            selected.get("first_touch_full_path_mae_norm", []),
                            0.90,
                        ),
                        "selected_timeout_rate": _safe_mean(
                            pd.to_numeric(selected.get("is_timeout", pd.Series(dtype=float)), errors="coerce").gt(0.5)
                        )
                        if len(selected)
                        else float("nan"),
                        "selected_mean_u": _safe_mean(selected.get("u_policy_net", [])),
                        "selected_mean_first_touch_gross": _safe_mean(selected.get("first_touch_gross", [])),
                        "selected_score_mean": _safe_mean(selected.get("score", [])),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["variant", "state_feature", "selected_rows", "selected_mfe_before_mae_1r_rate"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def _fit_pointwise_lgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    x_valid: pd.DataFrame,
    *,
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> np.ndarray:
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        raise RuntimeError("lightgbm is not available")
    model = LGBMRegressor(**_pointwise_model_params(seed=int(seed), model_params=model_params))
    model.fit(x_train, y_train, sample_weight=weights)
    return model.predict(x_valid).astype(np.float32)


def _fit_ranker(
    x_train: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    train_label: pd.DataFrame,
    weights: pd.Series,
    x_valid: pd.DataFrame,
    *,
    group_mode: str,
    relevance_mode: str,
    round_trip_cost: float,
    seed: int,
    ranker_params: dict[str, Any] | None = None,
) -> np.ndarray:
    if not _LIGHTGBM_AVAILABLE or LGBMRanker is None:
        raise RuntimeError("lightgbm is not available")
    order, group = _group_order(train_frame, mode=group_mode)
    if len(group) == 0 or len(order) != len(x_train):
        raise RuntimeError(f"invalid rank groups for {group_mode}")
    relevance = _rank_relevance(
        train_label,
        train_metrics,
        train_frame,
        group_mode=group_mode,
        relevance_mode=relevance_mode,
        round_trip_cost=float(round_trip_cost),
    )
    model = LGBMRanker(**_ranker_model_params(seed=int(seed), ranker_params=ranker_params))
    model.fit(
        x_train.reset_index(drop=True).iloc[order],
        relevance[order],
        group=group,
        sample_weight=weights.reset_index(drop=True).iloc[order],
    )
    return model.predict(x_valid.reset_index(drop=True)).astype(np.float32)


def _fit_side_specific_ranker(
    x_train: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    train_label: pd.DataFrame,
    weights: pd.Series,
    x_valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    *,
    relevance_mode: str,
    round_trip_cost: float,
    seed: int,
    ranker_params: dict[str, Any] | None = None,
) -> np.ndarray:
    score = np.full(len(x_valid), np.nan, dtype=np.float32)
    train_side = pd.to_numeric(train_metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    valid_side = pd.to_numeric(valid_metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    for offset, (side_name, train_mask, valid_mask) in enumerate(
        (
            ("long", train_side.ge(0.0), valid_side.ge(0.0)),
            ("short", train_side.lt(0.0), valid_side.lt(0.0)),
        )
    ):
        train_idx = np.flatnonzero(train_mask.to_numpy(dtype=bool))
        valid_idx = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
        if len(train_idx) < 500 or len(valid_idx) == 0:
            continue
        side_score = _fit_ranker(
            x_train.iloc[train_idx].reset_index(drop=True),
            train_frame.iloc[train_idx].reset_index(drop=True),
            train_metrics.iloc[train_idx].reset_index(drop=True),
            train_label.iloc[train_idx].reset_index(drop=True),
            weights.iloc[train_idx].reset_index(drop=True),
            x_valid.iloc[valid_idx].reset_index(drop=True),
            group_mode="timestamp",
            relevance_mode=relevance_mode,
            round_trip_cost=float(round_trip_cost),
            seed=int(seed) + 101 * (offset + 1),
            ranker_params=ranker_params,
        )
        score[valid_idx] = side_score
    if np.isnan(score).any():
        fill = np.nanmedian(score) if np.isfinite(score).any() else 0.0
        score = np.where(np.isfinite(score), score, fill).astype(np.float32)
    return score


def _run_variant(
    *,
    variant: str,
    folds: list[dict[str, Any]],
    config: LabelConfig,
    max_train_rows: int,
    round_trip_cost: float,
    path_order_mode: str,
    target_utility_mode: str,
    target_source: str,
    seed: int,
    ranker_params: dict[str, Any] | None = None,
    sample_weight_mode: str = "base",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[pd.DataFrame]]:
    fold_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for fold_i, fold in enumerate(folds):
        if str(target_source).strip().lower() == "materialized":
            train_label_full = _materialized_soft_label(fold["train_frame"], fold["train_metrics"])
            valid_label = _materialized_soft_label(fold["valid_frame"], fold["valid_metrics"])
        else:
            train_label_full = _make_side_soft_label(
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
        idx = _cap_indices(int(fold["train_rows"]), int(max_train_rows), seed=int(seed) + fold_i * 17)
        x_train = fold["x_train"].iloc[idx].reset_index(drop=True)
        train_frame = fold["train_frame"].iloc[idx].reset_index(drop=True)
        train_metrics = fold["train_metrics"].iloc[idx].reset_index(drop=True)
        train_label = train_label_full.iloc[idx].reset_index(drop=True)
        weights = _ranker_sample_weight(
            train_metrics,
            train_label,
            round_trip_cost=round_trip_cost,
            target_utility_mode="first_touch_net"
            if str(target_source).strip().lower() == "materialized"
            else target_utility_mode,
            mode=str(sample_weight_mode),
        )
        y_train = train_label["target_soft"].reset_index(drop=True)
        variant_name = str(variant)
        cleangross_aware = variant_name.endswith("_cleangross")
        if cleangross_aware:
            variant_name = variant_name.removesuffix("_cleangross")
        ordered_clean_ev_aware = variant_name.endswith("_ordered_clean_ev")
        if ordered_clean_ev_aware:
            variant_name = variant_name.removesuffix("_ordered_clean_ev")
        soft_ordered_ev_aware = variant_name.endswith("_soft_ordered_ev")
        if soft_ordered_ev_aware:
            variant_name = variant_name.removesuffix("_soft_ordered_ev")
        soft_exec_ordered_ev_aware = variant_name.endswith("_soft_exec_ordered_ev")
        if soft_exec_ordered_ev_aware:
            variant_name = variant_name.removesuffix("_soft_exec_ordered_ev")
        soft_breadth_ordered_ev_aware = variant_name.endswith("_soft_breadth_ordered_ev")
        if soft_breadth_ordered_ev_aware:
            variant_name = variant_name.removesuffix("_soft_breadth_ordered_ev")
        firstpass_exec_ev_aware = variant_name.endswith("_firstpass_exec_ev")
        if firstpass_exec_ev_aware:
            variant_name = variant_name.removesuffix("_firstpass_exec_ev")
        ev_preserving_ordered_aware = variant_name.endswith("_ev_preserving_ordered")
        if ev_preserving_ordered_aware:
            variant_name = variant_name.removesuffix("_ev_preserving_ordered")
        exec_ordered_ev_aware = variant_name.endswith("_exec_ordered_ev")
        if exec_ordered_ev_aware:
            variant_name = variant_name.removesuffix("_exec_ordered_ev")
        pathnet_guarded_aware = variant_name.endswith("_pathnet_guarded")
        if pathnet_guarded_aware:
            variant_name = variant_name.removesuffix("_pathnet_guarded")
        evpath_aware = variant_name.endswith("_evpath")
        if evpath_aware:
            variant_name = variant_name.removesuffix("_evpath")
        fullpath_aware = variant_name.endswith("_fullpath")
        if fullpath_aware:
            variant_name = variant_name.removesuffix("_fullpath")
        if ordered_clean_ev_aware:
            relevance_mode = "ordered_clean_ev"
        elif soft_ordered_ev_aware:
            relevance_mode = "soft_ordered_ev"
        elif soft_exec_ordered_ev_aware:
            relevance_mode = "soft_exec_ordered_ev"
        elif soft_breadth_ordered_ev_aware:
            relevance_mode = "soft_breadth_ordered_ev"
        elif firstpass_exec_ev_aware:
            relevance_mode = "firstpass_exec_ev"
        elif ev_preserving_ordered_aware:
            relevance_mode = "ev_preserving_ordered"
        elif exec_ordered_ev_aware:
            relevance_mode = "exec_ordered_ev"
        elif pathnet_guarded_aware:
            relevance_mode = "pathnet_guarded"
        elif cleangross_aware:
            relevance_mode = "cleangross"
        elif fullpath_aware and evpath_aware:
            relevance_mode = "fullpath_evpath"
        elif fullpath_aware:
            relevance_mode = "fullpath"
        elif evpath_aware:
            relevance_mode = "evpath"
        else:
            relevance_mode = "first_touch"
        base_variant = variant_name
        if base_variant == "pointwise_lgbm":
            score = _fit_pointwise_lgbm(
                x_train,
                y_train,
                weights,
                fold["x_valid"],
                seed=int(seed) + fold_i,
                model_params=ranker_params,
            )
            status = "ok"
        elif base_variant == "ranker_timestamp":
            score = _fit_ranker(
                x_train,
                train_frame,
                train_metrics,
                train_label,
                weights,
                fold["x_valid"],
                group_mode="timestamp",
                relevance_mode=relevance_mode,
                round_trip_cost=float(round_trip_cost),
                seed=int(seed) + fold_i,
                ranker_params=ranker_params,
            )
            status = "ok"
        elif base_variant == "ranker_timestamp_side":
            score = _fit_ranker(
                x_train,
                train_frame,
                train_metrics,
                train_label,
                weights,
                fold["x_valid"],
                group_mode="timestamp_side",
                relevance_mode=relevance_mode,
                round_trip_cost=float(round_trip_cost),
                seed=int(seed) + fold_i,
                ranker_params=ranker_params,
            )
            status = "ok"
        elif base_variant == "ranker_side_specific_timestamp":
            score = _fit_side_specific_ranker(
                x_train,
                train_frame,
                train_metrics,
                train_label,
                weights,
                fold["x_valid"],
                fold["valid_metrics"],
                relevance_mode=relevance_mode,
                round_trip_cost=float(round_trip_cost),
                seed=int(seed) + fold_i,
                ranker_params=ranker_params,
            )
            status = "ok"
        else:
            raise ValueError(f"unknown variant: {variant}")
        row = _score_fold(
            pd.Series(score),
            valid_label,
            fold["valid_metrics"],
            fold["month"],
            round_trip_cost=round_trip_cost,
        )
        row.update(
            {
                "variant": variant,
                "stage": variant,
                "trial_number": 0,
                "label_name": f"{config.name}_{variant}",
                "family": config.family,
                "train_rows": int(len(x_train)),
                "train_rows_uncapped": int(fold["train_rows"]),
                "valid_rows": int(fold["valid_rows"]),
                "ranker_status": status,
                "ranker_params": json.dumps(_json_safe(ranker_params or DEFAULT_RANKER_PARAMS), sort_keys=True),
                "sample_weight_mode": str(sample_weight_mode),
                "round_trip_cost": float(round_trip_cost),
                "path_order_mode": str(path_order_mode),
                "target_utility_mode": str(target_utility_mode),
                "target_source": str(target_source),
                "ranker_relevance_mode": str(relevance_mode),
            }
        )
        fold_rows.append(row)
        ledgers.append(
            _scored_ledger(
                variant=variant,
                fold=fold,
                score=score,
                valid_label=valid_label.reset_index(drop=True),
            )
        )
    summary = _summarize_trial(
        variant,
        0,
        LabelConfig(
            name=f"{config.name}_{variant}",
            family=config.family,
            long=config.long,
            short=config.short,
        ),
        fold_rows,
        objective_mode="precision_topk",
    )
    summary["variant"] = variant
    summary["ranker_params"] = json.dumps(_json_safe(ranker_params or DEFAULT_RANKER_PARAMS), sort_keys=True)
    summary["sample_weight_mode"] = str(sample_weight_mode)
    return summary, fold_rows, ledgers


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    archetype_diag: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    top_cols = [
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_pass_bad_rate",
        "mean_top10_first_touch_bad_mae_1r_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_mean_first_touch_mae_norm",
        "mean_top10_p90_first_touch_mae_norm",
        "mean_top10_p90_first_touch_full_path_mae_norm",
        "mean_top10_timeout_rate",
        "mean_top10_mean_ev",
        "mean_top10_mean_first_touch_gross_minus_1pct",
        "mean_top10_mean_first_touch_executable_margin",
        "mean_top10_hit_first_touch_executable_margin",
        "mean_long_top10_mfe_1r_before_mae_1r_rate",
        "mean_short_top10_mfe_1r_before_mae_1r_rate",
        "mean_long_top10_mean_first_touch_executable_margin",
        "mean_short_top10_mean_first_touch_executable_margin",
    ]
    fold_cols = [
        "variant",
        "month",
        "top10_ev_weighted_first_touch_precision",
        "top10_first_pass_good_rate",
        "top10_first_pass_bad_rate",
        "top10_first_touch_bad_mae_1r_rate",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_mfe_1r_before_mae_1r_rate",
        "top10_mae_1r_before_mfe_1r_rate",
        "top10_mean_first_touch_mae_norm",
        "top10_p90_first_touch_mae_norm",
        "top10_p90_first_touch_full_path_mae_norm",
        "top10_timeout_rate",
        "top10_mean_ev",
        "top10_mean_first_touch_gross_minus_1pct",
        "top10_mean_first_touch_executable_margin",
        "top10_hit_first_touch_executable_margin",
    ]
    archetype_cols = [
        "state_feature",
        "bucket",
        "side",
        "selected_rows",
        "selected_mfe_before_mae_1r_rate",
        "selected_mae_before_mfe_1r_rate",
        "selected_first_pass_good_rate",
        "selected_first_pass_bad_rate",
        "selected_mean_first_touch_mae_norm",
        "selected_p90_first_touch_mae_norm",
        "selected_first_touch_full_path_bad_mae_1r_rate",
        "selected_p90_first_touch_full_path_mae_norm",
        "selected_mean_u",
    ]
    long_diag = (
        archetype_diag[
            archetype_diag["side"].astype(str).eq("long")
            & pd.to_numeric(archetype_diag["selected_rows"], errors="coerce").ge(20)
        ].copy()
        if not archetype_diag.empty and "side" in archetype_diag.columns
        else pd.DataFrame()
    )
    short_diag = (
        archetype_diag[
            archetype_diag["side"].astype(str).eq("short")
            & pd.to_numeric(archetype_diag["selected_rows"], errors="coerce").ge(20)
        ].copy()
        if not archetype_diag.empty and "side" in archetype_diag.columns
        else pd.DataFrame()
    )
    best_long = (
        long_diag.sort_values(
            ["selected_mfe_before_mae_1r_rate", "selected_mae_before_mfe_1r_rate"],
            ascending=[False, True],
        )
        if not long_diag.empty
        else long_diag
    )
    worst_long = (
        long_diag.sort_values(
            ["selected_mae_before_mfe_1r_rate", "selected_rows"],
            ascending=[False, False],
        )
        if not long_diag.empty
        else long_diag
    )
    best_short = (
        short_diag.sort_values(
            ["selected_mfe_before_mae_1r_rate", "selected_mae_before_mfe_1r_rate"],
            ascending=[False, True],
        )
        if not short_diag.empty
        else short_diag
    )
    lines = [
        "# S52 Timestamp x Side Ranker Smoke",
        "",
        "Scope: compare pointwise LGBM with LGBMRanker grouped by timestamp and timestamp x side under the same S52 path-ordered labels.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Months: `{', '.join(manifest['fold_months'])}`",
        f"Features: `{manifest['features']}`",
        f"Variants: `{', '.join(manifest['variants'])}`",
        "",
        "## Variant Summary",
        "",
        _fmt(summary, top_cols),
        "",
        "## Fold Metrics",
        "",
        _fmt(folds, fold_cols),
        "",
        "## Archetype / State Diagnostics",
        "",
        "These are live-predictable AE/GMM/state feature buckets evaluated on the selected top10 rows. They are diagnostics/overlays, not hard gates.",
        "",
        "### Best Long State Buckets",
        "",
        _fmt(best_long.head(20), archetype_cols),
        "",
        "### Worst Long State Buckets",
        "",
        _fmt(worst_long.head(20), archetype_cols),
        "",
        "### Best Short State Buckets",
        "",
        _fmt(best_short.head(20), archetype_cols),
        "",
        f"- Scored ledger: `{manifest['outputs'].get('scored_ledger')}`",
        f"- State bucket diagnostics: `{manifest['outputs'].get('archetype_path_diagnostics')}`",
        "",
    ]
    (output_dir / "s52_ranker_smoke.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    best_config_path: Path,
    output_dir: Path,
    months: list[str],
    max_train_rows: int,
    round_trip_cost: float,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    ae_gmm_state_feature_seed: int,
    target_source: str,
    variants: list[str] | None = None,
    ranker_params: dict[str, Any] | None = None,
    sample_weight_mode: str = "base",
    ae_gmm_fold_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if not _LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is required for this smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(best_config_path)
    selected_variants = list(
        variants
        or [
            "pointwise_lgbm",
            "ranker_timestamp",
            "ranker_timestamp_side",
            "ranker_side_specific_timestamp",
        ]
    )
    valid_variants = {
        "pointwise_lgbm",
        "ranker_timestamp",
        "ranker_timestamp_side",
        "ranker_side_specific_timestamp",
        "ranker_timestamp_fullpath",
        "ranker_timestamp_side_fullpath",
        "ranker_side_specific_timestamp_fullpath",
        "ranker_timestamp_evpath",
        "ranker_timestamp_side_evpath",
        "ranker_side_specific_timestamp_evpath",
        "ranker_timestamp_fullpath_evpath",
        "ranker_timestamp_side_fullpath_evpath",
        "ranker_side_specific_timestamp_fullpath_evpath",
        "ranker_timestamp_cleangross",
        "ranker_timestamp_side_cleangross",
        "ranker_side_specific_timestamp_cleangross",
        "ranker_timestamp_ordered_clean_ev",
        "ranker_timestamp_side_ordered_clean_ev",
        "ranker_side_specific_timestamp_ordered_clean_ev",
        "ranker_timestamp_soft_ordered_ev",
        "ranker_timestamp_side_soft_ordered_ev",
        "ranker_side_specific_timestamp_soft_ordered_ev",
        "ranker_timestamp_soft_exec_ordered_ev",
        "ranker_timestamp_side_soft_exec_ordered_ev",
        "ranker_side_specific_timestamp_soft_exec_ordered_ev",
        "ranker_timestamp_soft_breadth_ordered_ev",
        "ranker_timestamp_side_soft_breadth_ordered_ev",
        "ranker_side_specific_timestamp_soft_breadth_ordered_ev",
        "ranker_timestamp_firstpass_exec_ev",
        "ranker_timestamp_side_firstpass_exec_ev",
        "ranker_side_specific_timestamp_firstpass_exec_ev",
        "ranker_timestamp_ev_preserving_ordered",
        "ranker_timestamp_side_ev_preserving_ordered",
        "ranker_side_specific_timestamp_ev_preserving_ordered",
        "ranker_timestamp_exec_ordered_ev",
        "ranker_timestamp_side_exec_ordered_ev",
        "ranker_side_specific_timestamp_exec_ordered_ev",
        "ranker_timestamp_pathnet_guarded",
        "ranker_timestamp_side_pathnet_guarded",
        "ranker_side_specific_timestamp_pathnet_guarded",
    }
    unknown = sorted(set(selected_variants) - valid_variants)
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        months=months,
        spread_baseline_path=None,
        spread_rank_column="p75_spread_bps",
        target_symbol_count=None,
        max_feature_store_features=None,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(ae_gmm_state_feature_max_iter),
        seed=int(seed),
        ae_gmm_state_feature_seed=int(ae_gmm_state_feature_seed),
        ae_gmm_fold_cache_dir=ae_gmm_fold_cache_dir,
    )
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    ledger_parts: list[pd.DataFrame] = []
    for variant in selected_variants:
        summary, rows, ledgers = _run_variant(
            variant=variant,
            folds=folds,
            config=config,
            max_train_rows=int(max_train_rows),
            round_trip_cost=float(round_trip_cost),
            path_order_mode="s52_first_touch",
            target_utility_mode="geometry_only",
            target_source=str(target_source),
            seed=int(seed),
            ranker_params=ranker_params,
            sample_weight_mode=str(sample_weight_mode),
        )
        summaries.append(summary)
        fold_rows.extend(rows)
        ledger_parts.extend(ledgers)
    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    ledger_df = pd.concat(ledger_parts, ignore_index=True) if ledger_parts else pd.DataFrame()
    archetype_diag = _archetype_path_diagnostics(ledger_df)
    paths = {
        "summary": output_dir / "s52_ranker_smoke_summary.csv",
        "folds": output_dir / "s52_ranker_smoke_folds.csv",
        "scored_ledger": output_dir / "s52_ranker_smoke_scored_ledger.parquet",
        "archetype_path_diagnostics": output_dir / "s52_ranker_smoke_archetype_path_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_ranker_smoke.md",
    }
    summary_df.to_csv(paths["summary"], index=False)
    fold_df.to_csv(paths["folds"], index=False)
    ledger_df.to_parquet(paths["scored_ledger"], index=False)
    archetype_diag.to_csv(paths["archetype_path_diagnostics"], index=False)
    manifest.update(
        {
            "scope": "s52_timestamp_side_ranker_smoke",
            "labels_path": str(labels_path),
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "best_config_path": str(best_config_path),
            "output_dir": str(output_dir),
            "max_train_rows": int(max_train_rows),
            "round_trip_cost": float(round_trip_cost),
            "seed": int(seed),
            "model_seed": int(seed),
            "ae_gmm_state_feature_seed": int(ae_gmm_state_feature_seed),
            "target_source": str(target_source),
            "variants": [str(v) for v in selected_variants],
            "ranker_params": _json_safe(ranker_params or DEFAULT_RANKER_PARAMS),
            "sample_weight_mode": str(sample_weight_mode),
            "ae_gmm_fold_cache_dir": str(ae_gmm_fold_cache_dir) if ae_gmm_fold_cache_dir is not None else None,
            "config": asdict(config),
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(output_dir, summary_df, fold_df, archetype_diag, manifest)
    return {
        "output_dir": str(output_dir),
        "summary": str(paths["summary"]),
        "folds": str(paths["folds"]),
        "scored_ledger": str(paths["scored_ledger"]),
        "archetype_path_diagnostics": str(paths["archetype_path_diagnostics"]),
        "report": str(paths["report"]),
        "top": _json_safe(summary_df.head(3).to_dict(orient="records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--best-config-path", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RANKER_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        default="pointwise_lgbm,ranker_timestamp,ranker_timestamp_side,ranker_side_specific_timestamp",
        help="Comma-separated variants to run.",
    )
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=60_000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=32)
    parser.add_argument("--ae-gmm-state-feature-seed", type=int, default=42)
    parser.add_argument(
        "--ae-gmm-fold-cache-dir",
        type=Path,
        default=None,
        help="Optional cache for fold-level AE/GMM-augmented train/valid matrices.",
    )
    parser.add_argument(
        "--target-source",
        choices=("hpo_config", "materialized"),
        default="hpo_config",
        help="Use HPO-config recomputed labels or the materialized __first_touch_target_soft__ label artifact.",
    )
    parser.add_argument(
        "--ranker-params-json",
        type=Path,
        default=None,
        help="Optional JSON object with LightGBM ranker parameter overrides.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=("base", "execres_clean_dirty", "long_clean_dirty"),
        default="base",
    )
    args = parser.parse_args()
    result = run_smoke(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        best_config_path=args.best_config_path,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        max_train_rows=int(args.max_train_rows),
        round_trip_cost=float(args.round_trip_cost),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        ae_gmm_state_feature_seed=int(args.ae_gmm_state_feature_seed),
        target_source=str(args.target_source),
        variants=_parse_csv(args.variants, ()),
        ranker_params=_load_ranker_params(args.ranker_params_json),
        sample_weight_mode=str(args.sample_weight_mode),
        ae_gmm_fold_cache_dir=args.ae_gmm_fold_cache_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
