#!/usr/bin/env python3
"""Market-state-aware top-10 policy calibration over frozen meta predictions.

The AE/MLP -> GMM state representation is observable and frozen.  Realized
outcomes are used only to fit side x policy-archetype residual recognizers on
chronologically prior rows.  Application parameters are selected on a separate
validation period and the final report is produced on untouched OOS rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_meta_residual_archetype_final import (  # noqa: E402
    _autocorr_components,
    _calendar_components_preselected,
)
from scripts.run_meta_residual_extreme_local_champion_overlay import (  # noqa: E402
    FEATURES,
    KEYS,
    PARENT,
    _breakdown,
    _load_joined,
    _metric_row,
    _feature_catalog,
    _rank_for_params,
)
from scripts.run_regime_calibration_model_ablation import (  # noqa: E402
    _NativeShallowLGBMRegressor,
)


@dataclass
class LocalRecognizer:
    side: str
    archetype: str
    features: list[str]
    hit_model: Any
    ev_model: Any
    adverse_model: Any
    favorable_model: Any
    ev_scale: float
    rows: int


@dataclass
class SharedRecognizer:
    features: list[str]
    hit_model: Any
    ev_model: Any
    adverse_model: Any
    favorable_model: Any
    ev_scale: float
    rows: int


EXPANDED_CONTEXT_TOKENS: tuple[str, ...] = (
    "ood_",
    "_ood",
    "leaf_",
    "_leaf",
    "support_drift",
    "leaf_drift",
    "feature_drift",
    "prediction_drift",
    "support_count",
    "support_log",
    "entropy",
    "posterior",
    "mahal",
    "reconstruction",
    "margin_to_cutoff",
    "base_score_z",
    "base_rank_pct",
    "signal_zscore",
)

POLICY_REFERENCE_ID = "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
POLICY_REFERENCE_DIR = Path(
    "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_"
    "hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260708_"
    "hr_threshold_modulation_top15_top5_protected_regime_rank_retained50/"
    "threshold_basis_top_fraction_inflection_8d_hroff_livecompat_20260710"
)


def _feature_block(frame: pd.DataFrame, arm: str) -> list[str]:
    local = [c for c in frame if c.startswith("resid_event_aegmm_")]
    shared = [c for c in frame if c.startswith("resid_event_market_aegmm_")]
    context = [
        c
        for c in frame
        if not c.startswith("resid_event_")
        and any(token in c.lower() for token in EXPANDED_CONTEXT_TOKENS)
    ]
    if arm == "distilled_local":
        return [c for c in FEATURES if c in frame]
    if arm == "full_local":
        return local
    if arm == "shared_market":
        return shared
    if arm == "joint_local_market":
        return list(dict.fromkeys([*local, *shared]))
    if arm == "joint_expanded_context":
        return list(dict.fromkeys([*local, *shared, *context]))
    if arm == "hierarchical_shared_local":
        return list(dict.fromkeys([*local, *context]))
    raise ValueError(f"unknown feature arm: {arm}")


def _shared_feature_block(frame: pd.DataFrame) -> list[str]:
    shared = [c for c in frame if c.startswith("resid_event_market_aegmm_")]
    context = [
        c
        for c in frame
        if not c.startswith("resid_event_")
        and any(token in c.lower() for token in EXPANDED_CONTEXT_TOKENS)
    ]
    return list(dict.fromkeys([*shared, *context]))


def _select_recognizer_features(
    group: pd.DataFrame,
    candidates: list[str],
    *,
    max_features: int = 24,
    sample_rows: int = 45_000,
    seed: int = 42,
) -> list[str]:
    """Nonlinear, score-conditional, time-stable local residual screening."""

    usable = [
        c
        for c in candidates
        if c in group
        and group[c].notna().mean() >= 0.35
        and group[c].nunique(dropna=True) >= 3
    ]
    if not usable:
        return []
    if len(group) > sample_rows:
        thirds = np.array_split(np.arange(len(group), dtype=np.int64), 3)
        per = max(1, sample_rows // 3)
        idx = np.unique(
            np.concatenate(
                [np.linspace(x[0], x[-1], min(per, len(x)), dtype=np.int64) for x in thirds if len(x)]
            )
        )[:sample_rows]
        sample = group.iloc[idx]
    else:
        sample = group
    x = sample[usable].apply(pd.to_numeric, errors="coerce")
    med = x.median(axis=0).fillna(0.0)
    arr = x.fillna(med).to_numpy(dtype=np.float32, copy=False)
    hit_resid = _num(sample, "clean_exec") - _num(sample, "hit_probability", 0.5)
    ev = _num(sample, "ev_after_1pct")
    ev_scale = max(float(np.nanstd(ev)), 1e-4)
    target = hit_resid + 0.25 * np.clip(ev / ev_scale, -3.0, 3.0)
    mi = mutual_info_regression(arr, target, random_state=seed, n_neighbors=5)
    thirds = np.array_split(np.arange(len(sample), dtype=np.int64), 3)
    stable = np.zeros(len(usable), dtype=np.float32)
    for j in range(len(usable)):
        values = arr[:, j]
        signs = []
        strengths = []
        for idx in thirds:
            if len(idx) < 50 or np.std(values[idx]) <= 1e-8:
                continue
            corr = np.corrcoef(values[idx], target[idx])[0, 1]
            if np.isfinite(corr):
                signs.append(np.sign(corr))
                strengths.append(abs(corr))
        if strengths:
            stable[j] = np.float32(
                np.mean(strengths) * (1.0 if abs(np.sum(signs)) == len(signs) else 0.35)
            )
    score = np.asarray(mi, dtype=np.float32) + 0.25 * stable
    order = np.argsort(-score, kind="stable")
    selected: list[str] = []
    for pos in order:
        feature = usable[int(pos)]
        if selected:
            corr = x[selected].corrwith(x[feature]).abs()
            if bool(corr.gt(0.94).any()):
                continue
        selected.append(feature)
        if len(selected) >= int(max_features):
            break
    return selected


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return (
        pd.to_numeric(frame[col], errors="coerce")
        .fillna(default)
        .to_numpy(dtype=np.float32, copy=False)
    )


def _model_params(seed: int) -> dict[str, Any]:
    return {
        "max_depth": 3,
        "num_leaves": 7,
        "n_estimators": 180,
        "learning_rate": 0.035,
        "min_child_samples": 180,
        "min_split_gain": 5e-4,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "seed": seed,
    }


def _fit_recognizers(
    train: pd.DataFrame,
    min_rows: int,
    seed: int,
    feature_candidates: list[str],
    frozen_features: dict[tuple[str, str], list[str]] | None = None,
) -> list[LocalRecognizer]:
    recognizers: list[LocalRecognizer] = []
    base_features = [*feature_candidates, "policy_parent_rank", "hit_probability"]
    for (side, archetype), group in train.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        if len(group) < min_rows:
            continue
        frozen = (frozen_features or {}).get((str(side), str(archetype)))
        features = (
            [c for c in frozen if c in group]
            if frozen
            else _select_recognizer_features(
                group,
                base_features,
                max_features=24,
                sample_rows=15_000,
                seed=seed + len(recognizers) * 31,
            )
        )
        if len(features) < 3:
            continue
        x = group[features].apply(pd.to_numeric, errors="coerce")
        hit_residual = _num(group, "clean_exec") - _num(group, "hit_probability", 0.5)
        ev = _num(group, "ev_after_1pct")
        # The score-conditioned local mean is the expected-EV baseline.  The
        # recognizer learns only the remaining market-state correction.
        score = _num(group, "hit_probability", 0.5)
        order = np.argsort(score, kind="stable")
        bins = np.array_split(order, min(40, max(8, len(group) // 500)))
        expected_ev = np.zeros(len(group), dtype=np.float32)
        for idx in bins:
            expected_ev[idx] = np.float32(np.mean(ev[idx]))
        ev_residual = ev - expected_ev
        weight = (0.5 + np.clip(_num(group, "historical_rank", 0.5), 0, 1)).astype(np.float32)
        hit_model = _NativeShallowLGBMRegressor(_model_params(seed)).fit(
            x, pd.Series(hit_residual, index=group.index), sample_weight=weight
        )
        ev_model = _NativeShallowLGBMRegressor(_model_params(seed + 17)).fit(
            x, pd.Series(ev_residual, index=group.index), sample_weight=weight
        )
        clean = _num(group, "clean_exec")
        rank = np.clip(_num(group, "historical_rank", 0.5), 0.0, 1.0)
        adverse_weight = (0.25 + 2.0 * (rank >= 0.80) + 2.0 * (rank >= 0.90)).astype(
            np.float32
        )
        favorable_weight = (
            0.25 + 2.0 * ((rank >= 0.70) & (rank < 0.90)) + 0.5 * (rank >= 0.90)
        ).astype(np.float32)
        adverse_model = _NativeShallowLGBMRegressor(_model_params(seed + 31)).fit(
            x,
            pd.Series(1.0 - clean, index=group.index),
            sample_weight=adverse_weight,
        )
        favorable_model = _NativeShallowLGBMRegressor(_model_params(seed + 47)).fit(
            x,
            pd.Series(clean, index=group.index),
            sample_weight=favorable_weight,
        )
        ev_scale = float(max(np.nanquantile(np.abs(ev_residual), 0.75), 1e-3))
        recognizers.append(
            LocalRecognizer(
                str(side), str(archetype), features, hit_model, ev_model,
                adverse_model, favorable_model, ev_scale, len(group)
            )
        )
    return recognizers


def _fit_shared_recognizer(
    train: pd.DataFrame,
    feature_candidates: list[str],
    seed: int,
    frozen_features: list[str] | None = None,
) -> SharedRecognizer | None:
    if len(train) < 5_000:
        return None
    keys = ["side_name", "archetype_policy_key"]
    work = train.copy(deep=False)
    hit_residual = pd.Series(
        _num(work, "clean_exec") - _num(work, "hit_probability", 0.5),
        index=work.index,
    )
    ev = pd.Series(_num(work, "ev_after_1pct"), index=work.index)
    hit_group_mean = hit_residual.groupby(
        [work[keys[0]], work[keys[1]]], observed=True
    ).transform("mean")
    ev_group_mean = ev.groupby(
        [work[keys[0]], work[keys[1]]], observed=True
    ).transform("mean")
    hit_target = (hit_residual - hit_group_mean).astype(np.float32)
    ev_target = (ev - ev_group_mean).astype(np.float32)
    screen = work.copy(deep=False)
    screen["clean_exec"] = (
        _num(screen, "hit_probability", 0.5) + hit_target.to_numpy()
    )
    screen["ev_after_1pct"] = ev_target.to_numpy()
    features = (
        [c for c in (frozen_features or []) if c in work]
        if frozen_features
        else _select_recognizer_features(
            screen,
            [*feature_candidates, "policy_parent_rank", "hit_probability"],
            max_features=24,
            sample_rows=45_000,
            seed=seed + 701,
        )
    )
    if len(features) < 3:
        return None
    x = work[features].apply(pd.to_numeric, errors="coerce")
    weight = (
        0.5 + np.clip(_num(work, "historical_rank", 0.5), 0.0, 1.0)
    ).astype(np.float32)
    hit_model = _NativeShallowLGBMRegressor(_model_params(seed + 719)).fit(
        x, hit_target, sample_weight=weight
    )
    ev_model = _NativeShallowLGBMRegressor(_model_params(seed + 733)).fit(
        x, ev_target, sample_weight=weight
    )
    clean = _num(work, "clean_exec")
    rank = np.clip(_num(work, "historical_rank", 0.5), 0.0, 1.0)
    adverse_model = _NativeShallowLGBMRegressor(_model_params(seed + 751)).fit(
        x,
        pd.Series(1.0 - clean, index=work.index),
        sample_weight=(0.25 + 2.0 * (rank >= 0.80) + 2.0 * (rank >= 0.90)),
    )
    favorable_model = _NativeShallowLGBMRegressor(_model_params(seed + 769)).fit(
        x,
        pd.Series(clean, index=work.index),
        sample_weight=(
            0.25 + 2.0 * ((rank >= 0.70) & (rank < 0.90)) + 0.5 * (rank >= 0.90)
        ),
    )
    ev_scale = float(max(np.nanquantile(np.abs(ev_target), 0.75), 1e-3))
    return SharedRecognizer(
        features, hit_model, ev_model, adverse_model, favorable_model, ev_scale, len(work)
    )


def _predict_recognizers(
    frame: pd.DataFrame,
    models: list[LocalRecognizer],
    shared_model: SharedRecognizer | None = None,
) -> pd.DataFrame:
    out = frame.copy(deep=False)
    hit_shift = np.zeros(len(out), dtype=np.float32)
    ev_shift = np.zeros(len(out), dtype=np.float32)
    uncertainty = np.ones(len(out), dtype=np.float32)
    adverse_risk = np.zeros(len(out), dtype=np.float32)
    favorable_opportunity = np.zeros(len(out), dtype=np.float32)
    effect = np.zeros(len(out), dtype=np.int8)
    side = out["side_name"].astype(str).to_numpy()
    arch = out["archetype_policy_key"].astype(str).to_numpy()
    for model in models:
        pos = np.flatnonzero((side == model.side) & (arch == model.archetype))
        if not len(pos):
            continue
        x = out.iloc[pos][model.features].apply(pd.to_numeric, errors="coerce")
        h = np.asarray(model.hit_model.predict(x), dtype=np.float32)
        e = np.asarray(model.ev_model.predict(x), dtype=np.float32)
        hit_shift[pos] = np.clip(h, -0.35, 0.35)
        ev_shift[pos] = np.clip(e / model.ev_scale, -3.0, 3.0)
        adverse_risk[pos] = np.clip(
            np.asarray(model.adverse_model.predict(x), dtype=np.float32), 0.0, 1.0
        )
        favorable_opportunity[pos] = np.clip(
            np.asarray(model.favorable_model.predict(x), dtype=np.float32), 0.0, 1.0
        )
        entropy = _num(out.iloc[pos], "resid_event_aegmm_gmm_entropy", 0.5)
        recon = np.abs(_num(out.iloc[pos], "resid_event_aegmm_dae_reconstruction_error_zscore"))
        uncertainty[pos] = np.clip(0.5 * entropy + 0.25 * recon, 0.0, 3.0)
        effect[pos] = 1
    out = out.copy()
    out["state_expected_hit_shift"] = hit_shift
    out["state_expected_ev_shift"] = ev_shift
    out["state_uncertainty"] = uncertainty
    out["state_adverse_false_positive_risk"] = adverse_risk
    out["state_favorable_near_threshold_probability"] = favorable_opportunity
    out["state_recognizer_applied"] = effect
    shared_hit = np.zeros(len(out), dtype=np.float32)
    shared_ev = np.zeros(len(out), dtype=np.float32)
    shared_applied = np.zeros(len(out), dtype=np.int8)
    shared_adverse = np.zeros(len(out), dtype=np.float32)
    shared_favorable = np.zeros(len(out), dtype=np.float32)
    if shared_model is not None:
        x = out[shared_model.features].apply(pd.to_numeric, errors="coerce")
        shared_hit = np.clip(
            np.asarray(shared_model.hit_model.predict(x), dtype=np.float32),
            -0.25,
            0.25,
        )
        shared_ev = np.clip(
            np.asarray(shared_model.ev_model.predict(x), dtype=np.float32)
            / shared_model.ev_scale,
            -3.0,
            3.0,
        )
        shared_adverse = np.clip(
            np.asarray(shared_model.adverse_model.predict(x), dtype=np.float32),
            0.0,
            1.0,
        )
        shared_favorable = np.clip(
            np.asarray(shared_model.favorable_model.predict(x), dtype=np.float32),
            0.0,
            1.0,
        )
        shared_applied.fill(1)
    out["state_shared_expected_hit_shift"] = shared_hit
    out["state_shared_expected_ev_shift"] = shared_ev
    out["state_shared_recognizer_applied"] = shared_applied
    out["state_shared_adverse_false_positive_risk"] = shared_adverse
    out["state_shared_favorable_near_threshold_probability"] = shared_favorable
    return out


def _corrected_score(
    frame: pd.DataFrame, params: dict[str, Any], *, base_col: str
) -> np.ndarray:
    p = np.clip(_num(frame, base_col, 0.5), 1e-5, 1 - 1e-5)
    h = _num(frame, "state_expected_hit_shift")
    e = _num(frame, "state_expected_ev_shift")
    shared_weight = float(params.get("shared_weight", 0.0))
    h = h + shared_weight * _num(frame, "state_shared_expected_hit_shift")
    e = e + shared_weight * _num(frame, "state_shared_expected_ev_shift")
    u = _num(frame, "state_uncertainty")
    hit_weight = float(params["hit_weight"])
    ev_weight = float(params["ev_weight"])
    uncertainty_weight = float(params["uncertainty_weight"])
    if base_col == "hit_probability":
        hit_weight = float(params.get("probability_hit_weight", hit_weight))
        ev_weight = float(params.get("probability_ev_weight", 0.0))
        uncertainty_weight = float(
            params.get("probability_uncertainty_weight", 0.0)
        )
    raw = hit_weight * h + ev_weight * e
    raw -= uncertainty_weight * u
    if base_col != "hit_probability":
        adverse = _num(frame, "state_adverse_false_positive_risk").copy()
        favorable = _num(
            frame, "state_favorable_near_threshold_probability"
        ).copy()
        adverse += shared_weight * _num(
            frame, "state_shared_adverse_false_positive_risk"
        )
        favorable += shared_weight * _num(
            frame, "state_shared_favorable_near_threshold_probability"
        )
        raw -= float(params.get("adverse_weight", 0.0)) * adverse
        raw += float(params.get("favorable_weight", 0.0)) * favorable
    cap = float(params["cap"])
    delta = np.clip(raw, -cap, cap)
    mode = str(params["mode"])
    if mode == "additive":
        score = p + delta
    elif mode == "multiplicative":
        score = p * np.exp(delta)
    elif mode == "bounded":
        score = expit(logit(p) + delta)
    else:
        raise ValueError(mode)
    return np.clip(score, 1e-5, 1 - 1e-5).astype(np.float32)


def _fit_probability_map(
    score: np.ndarray,
    y: np.ndarray,
    method: str,
    sample_weight: np.ndarray | None = None,
) -> Any:
    if method == "none":
        return None
    if method in {"platt", "weighted_platt"}:
        model = LogisticRegression(C=0.01, max_iter=500)
        model.fit(
            np.clip(logit(np.clip(score, 1e-5, 1 - 1e-5)), -5.0, 5.0).reshape(-1, 1),
            y,
            sample_weight=sample_weight,
        )
        return model
    model = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
    model.fit(score, y, sample_weight=sample_weight)
    return model


def _apply_probability_map(score: np.ndarray, model: Any, method: str) -> np.ndarray:
    if model is None:
        return score
    if method in {"platt", "weighted_platt"}:
        return model.predict_proba(
            np.clip(logit(np.clip(score, 1e-5, 1 - 1e-5)), -5.0, 5.0).reshape(-1, 1)
        )[:, 1]
    return model.predict(score)


def _probability_map_quality(
    mapped: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float | bool]:
    mapped = np.asarray(mapped, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    finite = np.isfinite(mapped) & np.isfinite(y) & np.isfinite(baseline)
    if int(finite.sum()) < 100:
        return {"valid": False, "reason_code": 1.0}
    mapped, y, baseline = mapped[finite], y[finite], baseline[finite]
    residual = y - mapped
    baseline_residual = y - baseline
    unique_levels = int(np.unique(np.round(mapped, 6)).size)
    mapped_std = float(np.std(mapped))
    baseline_std = float(np.std(baseline))
    brier = float(np.mean((y - mapped) ** 2))
    baseline_brier = float(np.mean((y - baseline) ** 2))
    negative = float(np.minimum(residual, 0.0).mean())
    baseline_negative = float(np.minimum(baseline_residual, 0.0).mean())
    valid = bool(
        unique_levels >= 12
        and mapped_std >= max(0.01, 0.20 * baseline_std)
        and brier <= baseline_brier + 0.01
        and negative >= baseline_negative - 0.015
    )
    return {
        "valid": valid,
        "unique_probability_levels": float(unique_levels),
        "mapped_probability_std": mapped_std,
        "baseline_probability_std": baseline_std,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "mean_negative_surprise_quality": negative,
        "baseline_mean_negative_surprise_quality": baseline_negative,
    }


def _rolling_probability_map(
    frame: pd.DataFrame,
    raw_hit: np.ndarray,
    raw_rank: np.ndarray,
    method: str,
    *,
    first_validation_month: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    timestamps = (
        frame["__ts__"]
        if isinstance(frame["__ts__"].dtype, pd.DatetimeTZDtype)
        else pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    )
    month = (
        frame["__calibration_month__"].astype(str)
        if "__calibration_month__" in frame
        else timestamps.dt.strftime("%Y-%m")
    )
    validation_months = sorted(
        value for value in month.dropna().unique() if value >= first_validation_month
    )
    positions: list[np.ndarray] = []
    mapped_parts: list[np.ndarray] = []
    for value in validation_months:
        valid = month.eq(value).to_numpy()
        start = pd.Timestamp(f"{value}-01", tz="UTC")
        train = timestamps.lt(start).to_numpy()
        if int(train.sum()) < 500 or int(valid.sum()) < 100:
            continue
        train_positions = np.flatnonzero(train)
        admitted_local = _top10_mask(raw_rank[train_positions])
        admitted_positions = train_positions[admitted_local]
        train_y = _num(frame.iloc[admitted_positions], "clean_exec")
        sample_weight = (
            np.where(train_y < 0.5, 2.0, 1.0).astype(np.float32)
            if method.startswith("weighted_")
            else None
        )
        calibrator = _fit_probability_map(
            raw_hit[admitted_positions],
            train_y,
            method,
            sample_weight=sample_weight,
        )
        positions.append(np.flatnonzero(valid))
        mapped_parts.append(
            _apply_probability_map(raw_hit[valid], calibrator, method).astype(np.float32)
        )
    if not positions:
        return frame.iloc[0:0].copy(), np.zeros(0, dtype=np.float32)
    order = np.concatenate(positions)
    return frame.iloc[order].copy(), np.concatenate(mapped_parts)


def _top10_mask(score: np.ndarray, selected_rows: int | None = None) -> np.ndarray:
    finite = np.isfinite(score)
    mask = np.zeros(len(score), dtype=bool)
    n = (
        max(1, min(int(selected_rows), int(finite.sum())))
        if selected_rows is not None
        else max(1, int(np.ceil(0.10 * int(finite.sum()))))
    )
    positions = np.flatnonzero(finite)
    keep = positions[np.argpartition(score[finite], -n)[-n:]]
    mask[keep] = True
    return mask


def _causal_8d_residual_overlay(
    frame: pd.DataFrame,
    rank_score: np.ndarray,
    hit_probability: np.ndarray,
    strength: float,
    selected_rows: int,
) -> np.ndarray:
    work = frame[["__ts__", "side_name", "archetype_policy_key", "clean_exec"]].copy()
    work["__row_order__"] = np.arange(len(work), dtype=np.int64)
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    work["residual"] = _num(work, "clean_exec") - hit_probability
    # Live history contains outcomes only for admitted trades.  Build the
    # causal smoother from the prior selected top-10 book, never from rejected
    # candidate outcomes that would be unobservable in production.
    admitted = work.loc[_top10_mask(rank_score, selected_rows)]
    daily = admitted.groupby(["day", "side_name", "archetype_policy_key"], observed=True)["residual"].agg(["mean", "size"]).reset_index()
    daily["prior_residual"] = daily.groupby(["side_name", "archetype_policy_key"], observed=True)["mean"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean()
    )
    mapped = work.merge(daily[["day", "side_name", "archetype_policy_key", "prior_residual"]], on=["day", "side_name", "archetype_policy_key"], how="left", sort=False).sort_values("__row_order__", kind="stable")
    return np.clip(rank_score + float(strength) * mapped["prior_residual"].fillna(0).to_numpy(dtype=np.float32), 1e-5, 1 - 1e-5)


def _score_metrics(
    frame: pd.DataFrame,
    rank_score: np.ndarray,
    hit_probability: np.ndarray,
    name: str,
    selected_rows: int,
) -> dict[str, Any]:
    mask = _top10_mask(rank_score, selected_rows)
    row = _metric_row(frame, mask, name)
    calendar = _calendar_components_preselected(frame.loc[mask].assign(hit_probability=hit_probability[mask]), prob_col="hit_probability", arm=name)
    ac = _autocorr_components(calendar)
    def wabs(col: str) -> float:
        v = pd.to_numeric(ac[col], errors="coerce")
        return float(v.abs().mean()) if v.notna().any() else np.nan
    row.update({
        "mean_abs_signed_surprise_autocorr": wabs("signed_surprise_autocorr_lag1"),
        "mean_abs_negative_surprise_autocorr": wabs("negative_surprise_autocorr_lag1"),
        "mean_abs_positive_surprise_autocorr": wabs("positive_surprise_autocorr_lag1"),
        "mean_negative_surprise": float(np.minimum(_num(frame.loc[mask], "clean_exec") - hit_probability[mask], 0).mean()),
        "mean_positive_surprise": float(np.maximum(_num(frame.loc[mask], "clean_exec") - hit_probability[mask], 0).mean()),
    })
    return row


def _fast_economic_metrics(
    frame: pd.DataFrame,
    rank_score: np.ndarray,
    hit_probability: np.ndarray,
    name: str,
    selected_rows: int,
) -> dict[str, Any]:
    """Cheap first-stage screen; surprise calendars are computed for finalists."""
    mask = _top10_mask(rank_score, selected_rows)
    ev = _num(frame, "ev_after_1pct")
    if "__calibration_week_code__" in frame:
        week_code = frame["__calibration_week_code__"].to_numpy(dtype=np.int32)
        month_code = frame["__calibration_month_code__"].to_numpy(dtype=np.int16)
    else:
        ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        day = ts.dt.floor("D")
        week_code = pd.factorize(
            day - pd.to_timedelta(day.dt.weekday.to_numpy(), unit="D"), sort=True
        )[0].astype(np.int32)
        month_code = pd.factorize(ts.dt.strftime("%Y-%m"), sort=True)[0].astype(
            np.int16
        )

    def grouped_means(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
        selected_codes = codes[mask]
        valid = selected_codes >= 0
        if not valid.any():
            return np.array([np.nan], dtype=np.float32)
        selected_codes = selected_codes[valid]
        selected_values = values[mask][valid]
        size = int(selected_codes.max()) + 1
        sums = np.bincount(selected_codes, weights=selected_values, minlength=size)
        counts = np.bincount(selected_codes, minlength=size)
        return (sums[counts > 0] / counts[counts > 0]).astype(np.float32)

    weekly = grouped_means(ev, week_code)
    monthly = grouped_means(ev, month_code)
    residual = _num(frame.loc[mask], "clean_exec") - hit_probability[mask]
    return {
        "selector": name,
        "selected_rows": int(mask.sum()),
        "mean_ev_after_1pct": float(np.mean(ev[mask])),
        "worst_week_ev": float(np.nanmin(weekly)),
        "worst_month_ev": float(np.nanmin(monthly)),
        "mean_abs_row_surprise": float(np.mean(np.abs(residual))),
        "mean_negative_surprise": float(np.minimum(residual, 0).mean()),
        "mean_positive_surprise": float(np.maximum(residual, 0).mean()),
    }


def _objective(metric: dict[str, Any], baseline: dict[str, Any]) -> float:
    # Cumulative pass objective: economics and surprise magnitude first.
    # Autocorrelation is useful evidence of missing state, but low AC does not
    # rescue a calibrator that remains systematically over/under-confident.
    worst_week_floor = float(baseline["worst_week_ev"]) - 0.0005
    worst_month_floor = float(baseline["worst_month_ev"]) - 0.0005
    negative_floor = float(baseline["mean_negative_surprise"]) - 0.001
    if (
        metric["worst_week_ev"] < worst_week_floor
        or metric["worst_month_ev"] < worst_month_floor
        or metric["mean_negative_surprise"] < negative_floor
    ):
        return -1e6
    return float(
        100.0 * (metric["mean_ev_after_1pct"] - baseline["mean_ev_after_1pct"])
        + 24.0 * (metric["mean_negative_surprise"] - baseline["mean_negative_surprise"])
        + 10.0 * (baseline["mean_positive_surprise"] - metric["mean_positive_surprise"])
        + 6.0 * (baseline["mean_abs_negative_surprise_autocorr"] - metric["mean_abs_negative_surprise_autocorr"])
        + 2.0 * (baseline["mean_abs_positive_surprise_autocorr"] - metric["mean_abs_positive_surprise_autocorr"])
        + 15.0 * (metric["worst_week_ev"] - baseline["worst_week_ev"])
        + 10.0 * (metric["worst_month_ev"] - baseline["worst_month_ev"])
    )


def _fast_objective(metric: dict[str, Any], baseline: dict[str, Any]) -> float:
    if (
        metric["worst_week_ev"] < float(baseline["worst_week_ev"]) - 0.0005
        or metric["worst_month_ev"] < float(baseline["worst_month_ev"]) - 0.0005
        or metric["mean_negative_surprise"]
        < float(baseline["mean_negative_surprise"]) - 0.001
    ):
        return -1e6
    return float(
        100.0 * (metric["mean_ev_after_1pct"] - baseline["mean_ev_after_1pct"])
        + 24.0
        * (metric["mean_negative_surprise"] - baseline["mean_negative_surprise"])
        + 10.0
        * (baseline["mean_positive_surprise"] - metric["mean_positive_surprise"])
        + 15.0 * (metric["worst_week_ev"] - baseline["worst_week_ev"])
        + 10.0 * (metric["worst_month_ev"] - baseline["worst_month_ev"])
    )


def _grid(*, shared_enabled: bool = False) -> list[dict[str, Any]]:
    # Hierarchical application search.  The residual recognizer already
    # estimates continuous effects; a large Cartesian policy grid adds little
    # information and repeatedly re-evaluates nearly identical top-10 books.
    rows: list[dict[str, Any]] = []
    for mode in ("additive", "multiplicative", "bounded"):
        for hit_weight, ev_weight, uncertainty_weight, cap in (
            (0.25, 0.00, 0.000, 0.025),
            (0.50, 0.00, 0.000, 0.050),
            (1.00, 0.00, 0.000, 0.100),
            (0.25, 0.01, 0.000, 0.050),
            (0.25, 0.05, 0.000, 0.100),
            (0.25, 0.10, 0.000, 0.100),
            (0.25, 0.10, 0.005, 0.100),
            (0.50, 0.01, 0.005, 0.050),
            (0.50, 0.025, 0.005, 0.100),
            (1.00, 0.025, 0.005, 0.100),
            (1.00, 0.01, 0.000, 0.050),
        ):
            for shared_weight in ((0.0, 0.25, 0.50) if shared_enabled else (0.0,)):
                rows.append(
                    dict(
                        mode=mode,
                        hit_weight=hit_weight,
                        ev_weight=ev_weight,
                        uncertainty_weight=uncertainty_weight,
                        cap=cap,
                        shared_weight=shared_weight,
                        adverse_weight=0.0,
                        favorable_weight=0.0,
                    )
                )
    if shared_enabled:
        for mode in ("additive", "bounded"):
            for shared_weight in (0.25, 0.50):
                for adverse_weight, favorable_weight in (
                    (0.025, 0.0),
                    (0.05, 0.0),
                    (0.10, 0.0),
                    (0.05, 0.01),
                    (0.10, 0.01),
                    (0.10, 0.025),
                ):
                    rows.append(
                        {
                            "mode": mode,
                            "hit_weight": 0.25,
                            "ev_weight": 0.05,
                            "uncertainty_weight": 0.005,
                            "cap": 0.10,
                            "shared_weight": shared_weight,
                            "adverse_weight": adverse_weight,
                            "favorable_weight": favorable_weight,
                        }
                    )
    return rows


def _merge_observable_context(
    frame: pd.DataFrame,
    *,
    state_artifact: Path,
    expanded_source: Path | None,
    override_existing: bool = False,
) -> pd.DataFrame:
    sources = [state_artifact]
    if expanded_source is not None:
        sources.append(expanded_source)
    out = frame
    for source in sources:
        if not source.exists():
            continue
        names = pq.read_schema(source).names
        wanted = [
            c
            for c in names
            if c.startswith(("resid_event_aegmm_", "resid_event_market_aegmm_"))
            or any(token in c.lower() for token in EXPANDED_CONTEXT_TOKENS)
        ]
        existing = [c for c in wanted if c in out.columns]
        if override_existing and existing:
            out = out.drop(columns=existing)
        wanted = [c for c in wanted if c not in out.columns]
        keys = [c for c in KEYS if c in names]
        if len(keys) != len(KEYS) or not wanted:
            continue
        extra = pd.read_parquet(source, columns=[*keys, *wanted])
        extra["__ts__"] = pd.to_datetime(extra["__ts__"], utc=True, errors="coerce")
        extra = extra.drop_duplicates(KEYS, keep="last")
        out = out.merge(extra, on=KEYS, how="left", validate="one_to_one")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/meta_market_state_threshold_calibration_20260712_v1"))
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--fit-end", default="2026-01-01")
    parser.add_argument("--tune-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--min-local-rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--feature-arm",
        choices=(
            "distilled_local",
            "full_local",
            "shared_market",
            "joint_local_market",
            "joint_expanded_context",
            "hierarchical_shared_local",
        ),
        default="distilled_local",
    )
    parser.add_argument(
        "--expanded-source",
        type=Path,
        default=Path("data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706/s52_trailing_regime_scored_ledger.parquet"),
    )
    parser.add_argument("--champion-ledger", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/champion_frozen_single_source_202501_20260710/frozen_champion_single_source_ledger.parquet"))
    parser.add_argument("--train-oof-predictions-dir", type=Path, default=Path("data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/best_full_oos_fixedfs_streamed_v1/prediction_shards"))
    parser.add_argument("--train-oof-rank-cache", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"))
    parser.add_argument("--state-artifact", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/oos_residual_event_states.parquet"))
    parser.add_argument(
        "--context-state-artifact",
        type=Path,
        default=None,
        help=(
            "Optional state feature source left-joined onto the fixed state-artifact "
            "row universe. Matching AE/GMM context columns override the fixed-state "
            "columns without changing evaluation eligibility."
        ),
    )
    parser.add_argument("--parent-eval-predictions", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"))
    parser.add_argument(
        "--policy-reference-dir", type=Path, default=POLICY_REFERENCE_DIR
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start, fit_end, tune_end, eval_end = [pd.Timestamp(v, tz="UTC") for v in (args.train_start, args.fit_end, args.tune_end, args.eval_end)]
    history, test, coverage = _load_joined(champion_path=args.champion_ledger, parent_eval_path=args.parent_eval_predictions, state_path=args.state_artifact, train_oof_predictions_dir=args.train_oof_predictions_dir, train_oof_rank_cache=args.train_oof_rank_cache, train_start=start, train_end=tune_end, eval_end=eval_end)
    # Reconstruct the exact promoted 95% down-only parent overlay.  Its state
    # references and feature directions are fixed before April-June OOS and
    # before any challenger context is merged.  Challenger state must never
    # redefine the parent ordering or its activity budget.
    parent_95_params = {
        "top_feature_count": 1,
        "threshold": 0.95,
        "alpha_down": 0.02,
        "alpha_up": 0.0,
    }
    parent_catalog = _feature_catalog(history)
    history_parent_rank, _, _ = _rank_for_params(
        history, history, parent_catalog, parent_95_params
    )
    test_parent_rank, _, _ = _rank_for_params(
        history, test, parent_catalog, parent_95_params
    )
    history = history.copy(deep=False)
    test = test.copy(deep=False)
    history["policy_parent_rank"] = history_parent_rank
    test["policy_parent_rank"] = test_parent_rank
    history_keys = history.loc[:, KEYS].copy()
    test_keys = test.loc[:, KEYS].copy()
    history = _merge_observable_context(
        history,
        state_artifact=args.context_state_artifact or args.state_artifact,
        expanded_source=args.expanded_source,
        override_existing=args.context_state_artifact is not None,
    )
    test = _merge_observable_context(
        test,
        state_artifact=args.context_state_artifact or args.state_artifact,
        expanded_source=args.expanded_source,
        override_existing=args.context_state_artifact is not None,
    )
    if not history.loc[:, KEYS].reset_index(drop=True).equals(
        history_keys.reset_index(drop=True)
    ) or not test.loc[:, KEYS].reset_index(drop=True).equals(
        test_keys.reset_index(drop=True)
    ):
        raise AssertionError("Observable context merge changed the fixed parent row universe")
    fit = history.loc[history["__ts__"].lt(fit_end)]
    tune = history.loc[history["__ts__"].ge(fit_end) & history["__ts__"].lt(tune_end)].copy()
    feature_candidates = _feature_block(history, str(args.feature_arm))
    models = _fit_recognizers(
        fit, args.min_local_rows, args.seed, feature_candidates
    )
    shared_candidates = _shared_feature_block(history)
    shared_model = (
        _fit_shared_recognizer(fit, shared_candidates, args.seed)
        if args.feature_arm == "hierarchical_shared_local"
        else None
    )
    frozen_local_features = {
        (model.side, model.archetype): list(model.features) for model in models
    }
    tune = _predict_recognizers(tune, models, shared_model)
    tune["__calibration_month__"] = pd.to_datetime(
        tune["__ts__"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m")
    tune_day = pd.to_datetime(tune["__ts__"], utc=True, errors="coerce").dt.floor("D")
    tune["__calibration_week_code__"] = pd.factorize(
        tune_day - pd.to_timedelta(tune_day.dt.weekday.to_numpy(), unit="D"),
        sort=True,
    )[0].astype(np.int32)
    tune["__calibration_month_code__"] = pd.factorize(
        tune["__calibration_month__"], sort=True
    )[0].astype(np.int16)
    first_validation_month = (fit_end + pd.DateOffset(months=1)).strftime("%Y-%m")
    screen_end = fit_end + pd.DateOffset(months=1)
    tune_left = tune.loc[tune["__ts__"].lt(screen_end)].copy()
    tune_right = tune.loc[tune["__ts__"].ge(screen_end)].copy()
    left_budget = int(np.sum(_num(tune_left, "policy_parent_rank") >= 0.90))
    right_budget = int(np.sum(_num(tune_right, "policy_parent_rank") >= 0.90))
    baseline_tune = _score_metrics(
        tune_right,
        _num(tune_right, "policy_parent_rank"),
        _num(tune_right, "hit_probability", 0.5),
        "parent_95",
        right_budget,
    )
    raw_search: list[dict[str, Any]] = []
    raw_scores: list[np.ndarray] = []
    for params in _grid(shared_enabled=shared_model is not None):
        rank_score = _corrected_score(tune_left, params, base_col="policy_parent_rank")
        hit_score = _corrected_score(tune_left, params, base_col="hit_probability")
        metric = _fast_economic_metrics(
            tune_left,
            rank_score,
            hit_score,
            f"{params['mode']}_none",
            left_budget,
        )
        screen = (
            100.0 * (metric["mean_ev_after_1pct"] - baseline_tune["mean_ev_after_1pct"])
            + 15.0 * (metric["worst_week_ev"] - baseline_tune["worst_week_ev"])
            + 10.0 * (metric["worst_month_ev"] - baseline_tune["worst_month_ev"])
            - 2.0 * metric["mean_abs_row_surprise"]
        )
        raw_search.append({**params, "screen_objective": screen, **metric})
        raw_scores.append(rank_score)
    raw_df = pd.DataFrame(raw_search).sort_values(
        "screen_objective", ascending=False, kind="stable"
    )
    # Rank configurations are screened once. Probability strength and mapping
    # are then evaluated cheaply; only finalists build autocorrelation calendars.
    tune_index_to_position = pd.Series(
        np.arange(len(tune), dtype=np.int64), index=tune.index
    )
    mapped_candidates: list[dict[str, Any]] = []
    mapped_payloads: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
    for pos in list(raw_df.head(6).index):
        rank_params = dict(raw_search[pos])
        raw_rank_all = _corrected_score(
            tune, rank_params, base_col="policy_parent_rank"
        )
        probability_weights = tuple(
            dict.fromkeys(
                (
                    0.0,
                    0.25 * float(rank_params["hit_weight"]),
                    float(rank_params["hit_weight"]),
                )
            )
        )
        for probability_hit_weight in probability_weights:
            params = {**rank_params, "probability_hit_weight": probability_hit_weight}
            raw_hit_all = _corrected_score(tune, params, base_col="hit_probability")
            for map_method in (
                "none",
                "platt",
                "weighted_platt",
                "isotonic",
                "weighted_isotonic",
            ):
                validation, mapped_hit = _rolling_probability_map(
                    tune,
                    raw_hit_all,
                    raw_rank_all,
                    map_method,
                    first_validation_month=first_validation_month,
                )
                score_positions = tune_index_to_position.loc[
                    validation.index
                ].to_numpy(dtype=np.int64)
                raw_rank = raw_rank_all[score_positions]
                validation_selected = _top10_mask(raw_rank, right_budget)
                quality = _probability_map_quality(
                    mapped_hit[validation_selected],
                    _num(validation, "clean_exec")[validation_selected],
                    _num(validation, "hit_probability", 0.5)[validation_selected],
                )
                selector = f"{params['mode']}_{map_method}_p{probability_hit_weight:g}"
                metric = _fast_economic_metrics(
                    validation, raw_rank, mapped_hit, selector, right_budget
                )
                fast_objective = (
                    _fast_objective(metric, baseline_tune)
                    if bool(quality.get("valid", False))
                    else -1e6
                )
                mapped_candidates.append(
                    {
                        **params,
                        "map_method": map_method,
                        **quality,
                        **metric,
                        "fast_objective": fast_objective,
                    }
                )
                mapped_payloads.append((validation, raw_rank, mapped_hit))
    fast_df = pd.DataFrame(mapped_candidates).sort_values(
        "fast_objective", ascending=False, kind="stable"
    )
    detailed: list[dict[str, Any]] = []
    for pos in list(fast_df.head(8).index):
        candidate = mapped_candidates[pos]
        validation, raw_rank, mapped_hit = mapped_payloads[pos]
        metric = _score_metrics(
            validation,
            raw_rank,
            mapped_hit,
            str(candidate["selector"]),
            right_budget,
        )
        detailed.append(
            {
                **candidate,
                **metric,
                "objective": _objective(metric, baseline_tune),
            }
        )
    search_df = pd.DataFrame(detailed).sort_values("objective", ascending=False, kind="stable")
    best = search_df.iloc[0].to_dict()
    if not np.isfinite(float(best.get("objective", np.nan))) or float(
        best.get("objective", -np.inf)
    ) <= 0.0:
        best = {
            "mode": "additive",
            "hit_weight": 0.0,
            "probability_hit_weight": 0.0,
            "probability_ev_weight": 0.0,
            "probability_uncertainty_weight": 0.0,
            "ev_weight": 0.0,
            "uncertainty_weight": 0.0,
            "cap": 0.0,
            "shared_weight": 0.0,
            "adverse_weight": 0.0,
            "favorable_weight": 0.0,
            "map_method": "none",
            "selector": "no_op_parent_fallback",
            "objective": 0.0,
        }
    final_models = _fit_recognizers(
        history,
        args.min_local_rows,
        args.seed,
        feature_candidates,
        frozen_features=frozen_local_features,
    )
    final_shared_model = (
        _fit_shared_recognizer(
            history,
            shared_candidates,
            args.seed,
            frozen_features=list(shared_model.features) if shared_model else None,
        )
        if shared_model is not None
        else None
    )
    scored = _predict_recognizers(test, final_models, final_shared_model)
    raw_rank = _corrected_score(scored, best, base_col="policy_parent_rank")
    raw_hit = _corrected_score(scored, best, base_col="hit_probability")
    # Application maps are calibrated on recognizer-OOS tuning predictions.
    # Refitting them on the final recognizer's in-sample history causes severe
    # probability saturation and invalid surprise diagnostics.
    tune_raw_best = _corrected_score(tune, best, base_col="hit_probability")
    tune_rank_best = _corrected_score(tune, best, base_col="policy_parent_rank")
    tune_admitted = _top10_mask(tune_rank_best)
    tune_admitted_y = _num(tune.iloc[np.flatnonzero(tune_admitted)], "clean_exec")
    final_sample_weight = (
        np.where(tune_admitted_y < 0.5, 2.0, 1.0).astype(np.float32)
        if str(best["map_method"]).startswith("weighted_")
        else None
    )
    final_map = _fit_probability_map(
        tune_raw_best[tune_admitted],
        tune_admitted_y,
        str(best["map_method"]),
        sample_weight=final_sample_weight,
    )
    corrected_hit = _apply_probability_map(
        raw_hit, final_map, str(best["map_method"])
    ).astype(np.float32)
    baseline_rank = _num(scored, "policy_parent_rank")
    baseline_hit = _num(scored, "hit_probability", 0.5)
    eval_budget = int(np.sum(baseline_rank >= 0.90))
    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "parent_95": (baseline_rank, baseline_hit),
        "market_state": (raw_rank, corrected_hit),
    }
    # One diagnostic history-aware comparator only. Multiple strength variants
    # are deliberately disabled so the primary search remains market-state
    # aware rather than tuning recent-performance feedback.
    arms["market_state_plus_causal8d"] = (
        _causal_8d_residual_overlay(
            scored, raw_rank, corrected_hit, 0.25, eval_budget
        ),
        corrected_hit,
    )
    metrics = pd.DataFrame(
        [
            _score_metrics(scored, rank, hit, name, eval_budget)
            for name, (rank, hit) in arms.items()
        ]
    )
    metrics["objective_vs_parent"] = metrics.apply(lambda row: _objective(row.to_dict(), metrics.iloc[0].to_dict()), axis=1)
    winner = metrics.sort_values("objective_vs_parent", ascending=False).iloc[0]["selector"]
    for name, (rank, hit) in arms.items():
        scored[f"rank_{name}"] = rank
        scored[f"hit_probability_{name}"] = hit
        scored[f"selected_{name}"] = _top10_mask(rank, eval_budget)
    scored.to_parquet(args.output_dir / "oos_predictions.parquet", index=False, compression="zstd")
    search_df.to_csv(args.output_dir / "tuning_search.csv", index=False)
    metrics.to_csv(args.output_dir / "summary.csv", index=False)
    comparison_rows: list[dict[str, Any]] = []
    comparison_scope = pd.to_datetime(scored["__ts__"], utc=True).ge(
        pd.Timestamp("2026-05-01", tz="UTC")
    )
    for name, (rank, _hit) in arms.items():
        selected = _top10_mask(rank, eval_budget) & comparison_scope.to_numpy()
        comparison_rows.append(
            {
                "arm": name,
                "layer": "market_state_meta_calibrator",
                "policy_or_model_id": f"{PARENT}_forced_local_tail_0.950",
                "underlying_model_basis": PARENT,
                "scope": "2026-05 through 2026-06; full Apr-Jun top10 selection subset",
                "selected_rows_or_trades": int(selected.sum()),
                "mean_net_ev_or_return_per_trade": float(
                    np.mean(_num(scored, "ev_after_1pct")[selected])
                ),
                "metric_semantics": "label EV after 1pct cost; pre-execution/pre-portfolio",
            }
        )
    policy_summary_path = args.policy_reference_dir / "summary_top_fraction_metrics.csv"
    if policy_summary_path.exists():
        policy_summary = pd.read_csv(policy_summary_path)
        policy_top10 = policy_summary.loc[
            pd.to_numeric(policy_summary.get("basis_top_pct"), errors="coerce").eq(10)
        ]
        if not policy_top10.empty:
            row = policy_top10.iloc[0]
            comparison_rows.append(
                {
                    "arm": "production_8d_policy_reference",
                    "layer": "execution_policy_after_portfolio",
                    "policy_or_model_id": POLICY_REFERENCE_ID,
                    "underlying_model_basis": "saved S52 calibrated_score_regime_ev candidate ledger",
                    "scope": "2026-05 through 2026-06 policy replay",
                    "selected_rows_or_trades": int(row.get("trade_count", 0)),
                    "mean_net_ev_or_return_per_trade": float(
                        row.get("mean_net_return_per_trade", np.nan)
                    ),
                    "metric_semantics": "executable notional net return after 1pct round-trip cost and policy/portfolio",
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "comparison_vs_8d_policy_reference.csv", index=False
    )
    breakdowns = []
    calendars = []
    for name, (rank, hit) in arms.items():
        mask = _top10_mask(rank, eval_budget)
        breakdowns.append(_breakdown(scored, mask, name))
        calendars.append(_calendar_components_preselected(scored.loc[mask].assign(hit_probability=hit[mask]), prob_col="hit_probability", arm=name))
    pd.concat(breakdowns, ignore_index=True).to_csv(args.output_dir / "breakdowns.csv", index=False)
    calendar = pd.concat(calendars, ignore_index=True)
    calendar.to_csv(args.output_dir / "hit_surprise_calendar.csv", index=False)
    _autocorr_components(calendar).to_csv(args.output_dir / "hit_surprise_autocorrelation.csv", index=False)
    joblib.dump({"recognizers": final_models, "shared_recognizer": final_shared_model, "probability_map": final_map, "params": best}, args.output_dir / "calibrator.joblib")
    manifest = {"schema": "market_state_threshold_calibration_v1", "parent": f"{PARENT}_forced_local_tail_0.950", "underlying_model_basis": PARENT, "policy_reference_id": POLICY_REFERENCE_ID, "policy_reference_dir": str(args.policy_reference_dir), "parent_95_params": parent_95_params, "evaluation_activity_budget": eval_budget, "feature_arm": args.feature_arm, "fixed_state_artifact": str(args.state_artifact), "context_state_artifact": str(args.context_state_artifact) if args.context_state_artifact else None, "expanded_source": str(args.expanded_source), "coverage": coverage, "fit_period": [str(start), str(fit_end)], "tuning_period": [str(fit_end), str(tune_end)], "evaluation_period": [str(tune_end), str(eval_end)], "best_tuning_params": best, "winner_oos": winner, "local_recognizers": len(final_models), "shared_recognizer": {"enabled": final_shared_model is not None, "rows": final_shared_model.rows if final_shared_model else 0, "features": final_shared_model.features if final_shared_model else []}, "candidate_feature_count": len(feature_candidates), "selected_features_by_local_recognizer": [{"side": m.side, "archetype": m.archetype, "rows": m.rows, "features": m.features} for m in final_models], "leakage_contract": "The exact promoted 95% down-only parent overlay is frozen before this layer. The fixed state artifact defines row eligibility; an optional context state artifact may only left-join observable context and cannot add rows. AE/MLP-GMM and model-trust state inputs are observable. Local residual recognizers and feature screening use prior OOF outcomes only. The optional shared recognizer is trained on side-archetype-demeaned residuals and cannot replace local encoders. Application parameters use Jan-Mar 2026 only. April-June 2026 is untouched OOS. The 8d production policy is an external comparator only and is not used to fit or select the market-state calibrator."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
