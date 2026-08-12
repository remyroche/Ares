#!/usr/bin/env python3
"""Discover portable causal pair conditions and replay condition specialists.

This runner implements the shared ``Orthogonal Model Evaluation`` brief on top
of the existing Ares ledger, feature registry, LambdaRank contracts and 15m
exit replay.  It is intentionally a sequential funnel:

* one causal context spine per side;
* support/recurrence/non-additivity screen for soft pair states;
* full feature/model response only for the screened shortlist;
* deterministic complementary condition selection;
* frozen condition-specific feature sets and residual LambdaRank specialists;
* compact OOF meta ablation, common-bps ranking and fixed-policy replay.

No target, cost, fold or query convention is redefined here.  The current
side-local R3 expected-net map and the canonical ordinal residual target are
reused from the existing stack.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource as _resource
import shutil
import sys
import time
from itertools import combinations, product
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import config as project_config  # noqa: E402
from extreme_price_movements.conditional_specialists import (  # noqa: E402
    ConditionalSpecialistConfig,
    cosine_distance,
    effective_rows,
    ordinal_residual_grade,
    portability_score,
    soft_regions,
    weighted_corr,
    weighted_jaccard,
    weighted_mean,
)
from extreme_price_movements.prequential_r3_value_map import (  # noqa: E402
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from extreme_price_movements.trailing_exit_grid import net_bps, simulate_h12_stop_trailing_grid  # noqa: E402
from scripts.run_broad_multiview_specialist_lambdarank import (  # noqa: E402
    LONG_HISTORY_FOLDS,
    MAX_TRAIN_ROWS,
    _base,
    _schema,
    _store_rows,
    _utc,
)


OUT = ROOT / "data_perp/artifacts/pair_condition_specialists_20260806_v1"
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
STORE = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet"
BASELINE_PRED = ROOT / "data_perp/artifacts/frozen_residual_query_hpo_20260810_v1/predictions.parquet"
PATH_ROOT = ROOT / "15m_ohlcv_perp"
PATH_ARTIFACT = ROOT / "data_perp/artifacts/h12_query_path_grid_20260805_v2"
SEED = 20260806
DISCOVERY_END = pd.Timestamp("2024-06-01", tz="UTC")
TRANSPORT_FOLDS = LONG_HISTORY_FOLDS[3:]
TAILS = (0.01, 0.05, 0.10)
DISCOVERY_SAMPLE_ROWS = 30_000

# Geometry-only control inputs.  These are decision-time context fields from
# the causal spine; the GMM is fit anew inside each transport fold and is never
# used to define the retained pair conditions.  Keeping this list small makes
# the control cheap and prevents it from becoming a disguised full-context arm.
GEOMETRY_CONTROL_FIELDS = (
    "mkt_drawdown_from_7d_high_atr",
    "mkt_recovery_from_24h_low_atr",
    "breadth_dispersion",
    "downside_breadth_intensity",
    "rv_24h_peer_resid",
    "ob_depth_mkt_resid",
    "mkt_funding_dispersion_z_30d",
    "mkt_oi_flush_z_30d",
    "oi_expansion_compression_balance_24h",
    "mkt_ret_per_oi_change_4h",
)

# The incumbent specialist and residual HPO values are reused verbatim.  The
# condition funnel changes only memberships, feature subsets and routing.
SPECIALIST_PARAMS: dict[str, Any] = {
    "n_estimators": 180,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 16,
    "min_child_samples": 776,
    "min_sum_hessian_in_leaf": 28.08104242513115,
    "min_gain_to_split": 0.003334820113493497,
    "colsample_bytree": 0.8397283415952219,
    "subsample": 0.7300957284014843,
    "subsample_freq": 1,
    "reg_alpha": 0.0001226082411532739,
    "reg_lambda": 1.745657954814456,
    "max_bin": 127,
    "label_gain": [0.0, 0.1, 1.0, 3.0, 7.0, 12.0],
    "verbosity": -1,
    "random_state": SEED,
    "n_jobs": 1,
}
META_PARAMS: dict[str, Any] = {
    "n_estimators": 220,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 52,
    "min_child_samples": 893,
    "min_sum_hessian_in_leaf": 1.13,
    "min_gain_to_split": 0.00893,
    "colsample_bytree": 0.788,
    "subsample": 0.867,
    "subsample_freq": 1,
    "reg_alpha": 0.031,
    "reg_lambda": 0.170,
    "max_bin": 63,
    "label_gain": [0.0, 0.25, 1.0, 3.0, 7.0, 12.0],
    "verbosity": -1,
    "random_state": SEED,
    "n_jobs": 1,
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _peak_rss_mb() -> float:
    """Return this process' peak resident set size in MiB."""
    raw = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def _feature_family_pool(available: set[str]) -> tuple[list[str], list[str], dict[str, str]]:
    """Return causal spine, predictive pool and ownership labels."""

    spine_groups = [
        "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
        "CAUSAL_CONTINUOUS_REGIME_META_FEATURE_KEYS",
        "MODEL_REGIME_XS_META_FEATURE_KEYS",
        "MODEL_REGIME_TAIL_META_FEATURE_KEYS",
        "MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS",
        "RESIDUAL_META_FEATURE_KEYS",
        "BASE_CONTEXT_RELIABILITY_FEATURE_KEYS",
        "OI_FUNDING_META_CANDIDATE_FEATURE_KEYS",
    ]
    predictive_groups = [
        "MODEL_DIRECT_BASE_FEATURE_KEYS",
        "BASE_COMPACT_PRIMITIVE_FEATURE_KEYS",
        "RESIDUAL_BASE_FEATURE_KEYS",
        "OI_FUNDING_BASE_CANDIDATE_FEATURE_KEYS",
        "OI_FUNDING_META_CANDIDATE_FEATURE_KEYS",
        "RESIDUAL_META_FEATURE_KEYS",
        "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
        "MODEL_REGIME_XS_META_FEATURE_KEYS",
        "MODEL_REGIME_TAIL_META_FEATURE_KEYS",
        "MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS",
    ]

    def _keys(groups: list[str]) -> list[str]:
        result: list[str] = []
        for group in groups:
            for value in getattr(project_config, group, []):
                value = str(value)
                if value in available and value not in result:
                    result.append(value)
        return result

    spine = _keys(spine_groups)
    predictive = _keys(predictive_groups)
    owners: dict[str, str] = {}
    for name in getattr(project_config, "MODEL_DIRECT_BASE_FEATURE_KEYS", []):
        owners.setdefault(str(name), "base")
    for group in predictive_groups[1:]:
        for name in getattr(project_config, group, []):
            owners.setdefault(str(name), "meta")
    return spine, predictive, owners


def _stable_select(frame: pd.DataFrame, fields: list[str], cap: int, *, require_families: bool = False) -> list[str]:
    """Coverage/variance/correlation reduction without target-aware selection."""

    if not fields:
        return []
    stats: list[tuple[str, float, float]] = []
    for field in fields:
        if field not in frame:
            continue
        value = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        coverage = float(np.isfinite(value).mean())
        finite = value[np.isfinite(value)]
        scale = float(np.median(np.abs(finite - np.median(finite))) * 1.4826) if len(finite) else 0.0
        if coverage >= 0.90 and np.isfinite(scale) and scale > 1e-8:
            stats.append((field, coverage, scale))
    stats.sort(key=lambda item: (-item[1], -item[2], item[0]))
    selected: list[str] = []
    # Compute one block correlation matrix.  The previous implementation
    # recomputed a full Spearman matrix for every candidate, which made the
    # label-free spine stage needlessly quadratic in wall time.
    stat_fields = [item[0] for item in stats]
    corr_matrix = frame.loc[:, stat_fields].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    for field, coverage, scale in stats:
        if len(selected) >= cap:
            break
        if selected:
            corr = corr_matrix.loc[field, selected].abs().max()
            if np.isfinite(corr) and float(corr) >= 0.995:
                continue
        selected.append(field)
    if len(selected) < min(40, cap):
        for field, _, _ in stats:
            if field not in selected:
                selected.append(field)
            if len(selected) >= min(40, cap):
                break
    return selected[:cap]


def _sample(frame: pd.DataFrame, n: int, *, seed: int = SEED) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    return frame.sample(n, random_state=seed, replace=False).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _rank_series(value: pd.Series) -> np.ndarray:
    return value.rank(method="average", pct=True).to_numpy(np.float32)


def _weighted_top_mean(score: np.ndarray, outcome: np.ndarray, weights: np.ndarray, fraction: float = 0.10) -> float:
    ok = np.isfinite(score) & np.isfinite(outcome) & np.isfinite(weights) & (weights > 0)
    if ok.sum() < 3:
        return float("nan")
    order = np.argsort(score[ok], kind="stable")[::-1]
    n = max(1, int(math.ceil(order.size * fraction)))
    idx = np.flatnonzero(ok)[order[:n]]
    return weighted_mean(outcome[idx], weights[idx])


def _weighted_rate(mask: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(mask, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    return weighted_mean(x[ok], w[ok]) if ok.any() else float("nan")


def _condition_weight(frame: pd.DataFrame, condition: dict[str, Any], activations: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    left = activations[condition["context_feature_a"]][condition["activation_a"]]
    # Unary conditions are used only by the predeclared univariate control.
    # They share the same frozen activation manifest and have no second state.
    # Keeping the identity factor explicit makes the control comparable to a
    # pair state without inventing a synthetic context feature.
    if condition.get("unary", False) or not condition.get("context_feature_b"):
        return left.astype(np.float32, copy=False)
    right = activations[condition["context_feature_b"]][condition["activation_b"]]
    return (left * right).astype(np.float32, copy=False)


def _condition_month_balanced_weights(
    frame: pd.DataFrame,
    membership: np.ndarray,
    *,
    exponent: float = 1.5,
    equal_months: bool = True,
) -> np.ndarray:
    """Build deterministic soft-membership weights with optional month balance."""
    w = np.maximum(np.asarray(membership, dtype=np.float64), 0.01) ** float(exponent)
    if equal_months and len(frame) and "__ts__" in frame:
        month = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m").to_numpy()
        mass = pd.Series(w).groupby(month, sort=False).transform("sum").to_numpy(float)
        positive = mass > 0.0
        if positive.any():
            target = float(np.mean(mass[positive]))
            w = w * np.where(positive, target / mass, 0.0)
    finite = np.isfinite(w) & (w > 0.0)
    if finite.any():
        w = w * (float(len(w)) / float(w[finite].sum()))
    return np.asarray(w, dtype=np.float32)


def _apply_manifest_memberships(frame: pd.DataFrame, manifest: dict[str, Any], fields: list[str]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for field in fields:
        spec = manifest[field]
        value = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        fill = float(spec["fill_median"])
        z = np.where(np.isfinite(value), value, fill)
        scale = max(float(spec["scale"]), 1e-8)
        low = 1.0 / (1.0 + np.exp(np.clip((z - float(spec["q25"])) / scale, -40.0, 40.0)))
        high = 1.0 / (1.0 + np.exp(np.clip((float(spec["q75"]) - z) / scale, -40.0, 40.0)))
        low[~np.isfinite(value)] = 0.0
        high[~np.isfinite(value)] = 0.0
        result[field] = {"low": low.astype(np.float32), "high": high.astype(np.float32)}
    return result


def _base_frame() -> pd.DataFrame:
    frame = _base().copy()
    frame["base_ev_bps"] = frame["prequential_base_expected_net_bps"].astype(np.float32)
    frame["residual_bps"] = (frame.net_bps - frame.base_ev_bps).astype(np.float32)
    frame["residual_grade"] = ordinal_residual_grade(frame.residual_bps.to_numpy(float))
    frame["month"] = pd.to_datetime(frame.__ts__, utc=True).dt.strftime("%Y-%m")
    frame["query_4h"] = pd.to_datetime(frame.__ts__, utc=True).dt.floor("4h")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _fit_ranker(frame: pd.DataFrame, fields: list[str], target: np.ndarray, query: pd.Series, params: dict[str, Any], weights: np.ndarray | None = None) -> tuple[lgb.LGBMRanker, list[str], np.ndarray, np.ndarray]:
    use = frame[["candidate_id", *fields]].copy()
    use["__query__"] = query.to_numpy()
    use["__row__"] = np.arange(len(use), dtype=np.int64)
    use = use.sort_values(["__query__", "candidate_id"], kind="stable")
    sizes = use.groupby("__query__", sort=False).size()
    valid_queries = set(sizes.index[sizes >= 2])
    use = use[use["__query__"].isin(valid_queries)].copy()
    if use.empty:
        raise ValueError("no rankable queries")
    order = use["__row__"].to_numpy(np.int64)
    groups = use.groupby("__query__", sort=False).size().to_numpy(np.int32)
    med = frame[fields].apply(pd.to_numeric, errors="coerce").median()
    X = use[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype(np.float32)
    y = np.asarray(target, dtype=np.int32)[order]
    w = None if weights is None else np.asarray(weights, dtype=np.float32)[order]
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10, **params)
    model.fit(X, y, group=groups, sample_weight=w)
    return model, fields, med.to_numpy(np.float32), order


def _predict(model: lgb.LGBMRanker, frame: pd.DataFrame, fields: list[str], med: np.ndarray) -> np.ndarray:
    X = frame[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(pd.Series(med, index=fields)).fillna(0.0).astype(np.float32)
    return model.predict(X).astype(np.float32)


def _add_fold_gmm_geometry(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    side: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Fit a small causal geometry-only GMM inside one transport fold.

    The returned columns are deliberately restricted to soft memberships and
    uncertainty summaries.  The fit uses only ``train`` context rows, while
    calibration/test are transformed without refitting.  This is a model
    control, not a condition-selection input, so component labels may vary by
    fold without changing the meaning of the retained pair conditions.
    """

    fields = [f for f in GEOMETRY_CONTROL_FIELDS if f in train.columns]
    if len(fields) < 4 or len(train) < 64:
        return train, calibration, test, [], {"status": "unavailable", "side": side, "fields": fields}
    med = train[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        return frame[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).to_numpy(np.float32)

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train = scaler.fit_transform(matrix(train)).astype(np.float32)
    # A fixed, small K is intentional: this is a predeclared control, not a
    # second HPO search.  The diagonal covariance is stable on sparse tails.
    n_components = 4
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=1,
        max_iter=100,
        random_state=SEED + (0 if side == "long" else 1),
    )
    gmm.fit(x_train)

    output_fields = [f"geometry_gmm_p_{i}" for i in range(n_components)] + [
        "geometry_gmm_entropy", "geometry_gmm_top2_margin", "geometry_gmm_ood"
    ]

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        z = frame.copy()
        probs = np.clip(gmm.predict_proba(scaler.transform(matrix(frame))), 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        for i in range(n_components):
            z[f"geometry_gmm_p_{i}"] = probs[:, i].astype(np.float32)
        ordered = np.sort(probs, axis=1)
        z["geometry_gmm_entropy"] = (-np.sum(probs * np.log(probs), axis=1) / np.log(float(n_components))).astype(np.float32)
        z["geometry_gmm_top2_margin"] = (ordered[:, -1] - ordered[:, -2]).astype(np.float32)
        # Mahalanobis-like density proxy, fit-only parameters; high values are
        # OOD but are not used as a target or selection weight.
        log_density = gmm.score_samples(scaler.transform(matrix(frame)))
        train_q = np.nanpercentile(gmm.score_samples(x_train), 5.0)
        z["geometry_gmm_ood"] = (log_density <= train_q).astype(np.float32)
        return z

    metadata = {
        "status": "fit",
        "side": side,
        "fields": fields,
        "n_components": n_components,
        "fit_rows": int(len(train)),
        "control": "geometry_only_gmm",
    }
    return transform(train), transform(calibration), transform(test), output_fields, metadata


def _fit_ridge_residual_arm(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    fields: list[str],
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the causal regularized linear residual control on training rows."""

    use = [f for f in fields if f in train.columns]
    if not use:
        return np.zeros(len(calibration), dtype=np.float32), np.zeros(len(test), dtype=np.float32)
    med = train[use].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        return frame[use].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).to_numpy(np.float32)

    x = matrix(train)
    y = train.residual_bps.to_numpy(np.float32)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if valid.sum() < max(64, len(use) + 8):
        return np.zeros(len(calibration), dtype=np.float32), np.zeros(len(test), dtype=np.float32)
    # The alpha is predeclared and is deliberately strong enough to make this
    # a blend/reliability control rather than an unconstrained second GBM.
    model = Ridge(alpha=25.0, fit_intercept=True)
    model.fit(x[valid], y[valid])
    return model.predict(matrix(calibration)).astype(np.float32), model.predict(matrix(test)).astype(np.float32)


def _within_query_rank(values: np.ndarray, query: pd.Series) -> np.ndarray:
    """Return deterministic percentile ranks within each decision query.

    LambdaRank scores are only identified up to a query-local monotone
    transformation.  Persisting the within-query representation explicitly
    prevents a raw score scale from being mistaken for a cross-query score and
    gives the residual/meta ablations a causal, query-aware specialist input.
    ``method='first'`` is deterministic because the caller preserves the
    canonical candidate ordering inside each query.
    """
    x = pd.Series(np.asarray(values, dtype=np.float64), copy=False)
    q = pd.Series(query.to_numpy(), index=x.index)
    result = x.groupby(q, sort=False).rank(method="first", pct=True)
    return result.fillna(0.5).to_numpy(np.float32)


def _score_metrics(frame: pd.DataFrame, score_col: str, *, scope: str, period: str, system: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tail in TAILS:
        n = max(1, int(math.ceil(len(frame) * tail)))
        top = frame.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
        rows.append({
            "system": system, "scope": scope, "period": period, "tail": tail,
            "rows": int(len(frame)), "trades": int(n),
            "gross_bps": float(top.gross_bps.mean()), "net_bps": float(top.net_bps.mean()),
            "rank_ic_net": float(frame[score_col].corr(frame.net_bps, method="spearman")),
        })
    return rows


def _fit_condition_spine(base: pd.DataFrame, available: set[str], cfg: ConditionalSpecialistConfig, out: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[str]]:
    spine_candidates, predictive_candidates, owners = _feature_family_pool(available)
    dev = base[base.__ts__.lt(DISCOVERY_END)].copy()
    dev_probe = _sample(dev, min(DISCOVERY_SAMPLE_ROWS, len(dev)), seed=cfg.global_seed)
    all_fields = list(dict.fromkeys(spine_candidates + predictive_candidates))
    joined = _store_rows(dev_probe, all_fields)
    dev_probe = dev_probe.merge(joined, on="candidate_id", validate="one_to_one")
    selected_spine: dict[str, list[str]] = {}
    selected_predictive: dict[str, list[str]] = {}
    spine_manifests: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        side_probe = dev_probe[dev_probe.side_name.eq(side)].copy()
        sp = _stable_select(side_probe, spine_candidates, cfg.max_raw_spine_features)
        pred = _stable_select(side_probe, predictive_candidates, 96)
        if len(sp) < cfg.min_raw_spine_features:
            raise RuntimeError(f"{side}: only {len(sp)} stable spine fields; contract requires at least {cfg.min_raw_spine_features}")
        selected_spine[side] = sp
        selected_predictive[side] = pred
        thresholds: dict[str, Any] = {}
        activation_values: dict[str, dict[str, np.ndarray]] = {}
        spine_values = side_probe[["candidate_id", "__ts__", *sp]].copy()
        for field in sp:
            low, high, spec = soft_regions(side_probe[field].to_numpy(float), width_quantile=cfg.soft_transition_width_quantile)
            thresholds[field] = spec
            activation_values[field] = {"low": low, "high": high}
            spine_values[f"__activation__{field}__low"] = low
            spine_values[f"__activation__{field}__high"] = high
        spine_values.to_parquet(out / f"condition_spine_values_{side}.parquet", index=False)
        _write_json(out / f"condition_spine_manifest_{side}.json", {
            "schema": "causal_pair_condition_spine_v1",
            "side": side,
            "discovery_end_utc": DISCOVERY_END.isoformat(),
            "fields": sp,
            "field_owners": {f: owners.get(f, "context") for f in sp},
            "predictive_pool": pred,
            "feature_contract_source": "project registry allowlist intersected with feature-store schema",
            "fit_rows": int(len(side_probe)),
            "config": cfg.to_dict(),
        })
        _write_json(out / f"condition_activation_manifest_{side}.json", {
            "schema": "causal_soft_two_region_activation_v1",
            "side": side,
            "discovery_end_utc": DISCOVERY_END.isoformat(),
            "regions": ["low", "high"],
            "features": thresholds,
        })
        spine_manifests[side] = thresholds
    _write_json(out / "feature_pool_manifest.json", {
        "schema": "pair_condition_feature_pool_v1",
        "available_store_features": len(available),
        "spine_fields_by_side": selected_spine,
        "predictive_fields_by_side": selected_predictive,
        "ownership": owners,
    })
    return spine_manifests, selected_spine, selected_predictive


def _generate_candidates(side_frame: pd.DataFrame, side: str, spine: list[str], activation_manifest: dict[str, Any], cfg: ConditionalSpecialistConfig) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    activations = _apply_manifest_memberships(side_frame, activation_manifest, spine)
    y = (side_frame.net_bps.to_numpy(float) > 50.0).astype(np.float32)
    base_rank = _rank_series(side_frame.base_score)
    net_rank = _rank_series(side_frame.net_bps)
    global_event = float(y.mean())
    global_rank_ic = float(np.corrcoef(base_rank, net_rank)[0, 1])
    base_q80 = float(np.nanquantile(side_frame.base_score, 0.80))
    global_top10 = float(side_frame.loc[side_frame.base_score >= base_q80, "net_bps"].mean())
    marginal: dict[tuple[str, str], tuple[float, float]] = {}
    for field in spine:
        for region in ("low", "high"):
            w = activations[field][region]
            marginal[(field, region)] = (weighted_mean(y, w) - global_event, weighted_mean(net_rank, w))
    rows: list[dict[str, Any]] = []
    qvals = pd.factorize(side_frame.query_4h, sort=True)[0]
    month_codes, month_names = pd.factorize(side_frame.month, sort=True)
    months = side_frame.month.to_numpy()
    for left, right in combinations(spine, 2):
        for left_region, right_region in product(("low", "high"), repeat=2):
            w = (activations[left][left_region] * activations[right][right_region]).astype(np.float32)
            support = float(w.sum())
            eff = effective_rows(w)
            hard = w >= 0.5
            eff_queries = int(np.unique(qvals[hard]).size) if hard.any() else 0
            month_mass = np.bincount(month_codes, weights=w, minlength=len(month_names))
            month_effective_queries = []
            for month_code in range(len(month_names)):
                month_mask = (month_codes == month_code) & (w >= 0.5)
                month_effective_queries.append(int(np.unique(qvals[month_mask]).size) if month_mask.any() else 0)
            supported_months = int(np.sum(month_mass >= 50.0))
            supported_nonadjacent_months = int(sum(
                1 for i, count in enumerate(month_effective_queries)
                if count >= cfg.minimum_month_effective_queries
                and any(month_effective_queries[j] >= cfg.minimum_month_effective_queries for j in range(len(month_effective_queries)) if abs(j - i) >= 1)
            ))
            if (
                eff < cfg.minimum_effective_rows
                or eff_queries < cfg.minimum_effective_queries
                or supported_months < cfg.minimum_supported_months
                or supported_nonadjacent_months < cfg.minimum_nonadjacent_months
            ):
                continue
            event_lift = weighted_mean(y, w) - global_event
            rank_ic = weighted_corr(base_rank, net_rank, w)
            top10 = _weighted_top_mean(side_frame.base_score.to_numpy(float), side_frame.net_bps.to_numpy(float), w)
            interaction = event_lift - marginal[(left, left_region)][0] - marginal[(right, right_region)][0]
            condition_id = f"{side}__{left}__{left_region}__{right}__{right_region}"
            rows.append({
                "condition_id": condition_id, "side": side,
                "context_feature_a": left, "activation_a": left_region,
                "context_feature_b": right, "activation_b": right_region,
                "global_effective_rows": support, "effective_rows": eff,
                "global_effective_queries": eff_queries, "supported_month_count": supported_months,
                "supported_nonadjacent_month_count": supported_nonadjacent_months,
                "membership_mean": float(w.mean()), "membership_p50": float(np.median(w)), "membership_p90": float(np.quantile(w, .90)),
                "event_lift": float(event_lift), "rank_ic": float(rank_ic),
                "top10_net_bps": float(top10), "global_top10_net_bps": float(global_top10),
                "pair_interaction": float(interaction), "joint_activation_hard_share": float(hard.mean()),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"{side}: no supported pair conditions")
    result["candidate_screen_score"] = result.pair_interaction.fillna(0.0) + result.event_lift.fillna(0.0) + result.rank_ic.fillna(0.0) * 25.0
    if len(result) > cfg.maximum_pairs_before_screen:
        result = result.sort_values(["candidate_screen_score", "effective_rows"], ascending=[False, False], kind="stable").head(cfg.maximum_pairs_before_screen).copy()
    return result.reset_index(drop=True), activations


def _monthly_model_response(side_frame: pd.DataFrame, candidates: pd.DataFrame, activations: dict[str, dict[str, np.ndarray]], top_n: int, side: str) -> pd.DataFrame:
    ranked = candidates.sort_values(["candidate_screen_score", "effective_rows"], ascending=[False, False], kind="stable").head(top_n)
    base_rank = _rank_series(side_frame.base_score)
    net_rank = _rank_series(side_frame.net_bps)
    global_y = (side_frame.net_bps.to_numpy(float) > 50.0).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for cond in ranked.to_dict("records"):
        w_all = _condition_weight(side_frame, cond, activations)
        for month, idx in side_frame.groupby("month", sort=True).groups.items():
            idx = np.asarray(idx, dtype=np.int64)
            w = w_all[idx]
            if effective_rows(w) < 50.0:
                continue
            event_lift = weighted_mean(global_y[idx], w) - float(global_y[idx].mean())
            rank_ic = weighted_corr(base_rank[idx], net_rank[idx], w)
            scores = side_frame.base_score.to_numpy(float)[idx]
            net = side_frame.net_bps.to_numpy(float)[idx]
            gross = side_frame.gross_bps.to_numpy(float)[idx]
            top1 = _weighted_top_mean(scores, net, w, 0.01)
            top5 = _weighted_top_mean(scores, net, w, 0.05)
            top10 = _weighted_top_mean(scores, net, w, 0.10)
            top1_base = float(side_frame.loc[idx, "net_bps"].nlargest(max(1, int(math.ceil(len(idx) * .01)))).mean())
            top5_base = float(side_frame.loc[idx, "net_bps"].nlargest(max(1, int(math.ceil(len(idx) * .05)))).mean())
            top10_base = float(side_frame.loc[idx, "net_bps"].nlargest(max(1, int(math.ceil(len(idx) * .10)))).mean())
            rows.append({
                "condition_id": cond["condition_id"], "side": side, "month": str(month),
                "effective_rows": effective_rows(w), "effective_queries": int(np.unique(side_frame.query_4h.to_numpy()[idx][w >= .5]).size) if (w >= .5).any() else 0,
                "event_lift": float(event_lift), "rank_ic": float(rank_ic),
                "pairwise_concordance": float((rank_ic + 1.0) / 2.0) if np.isfinite(rank_ic) else np.nan,
                "delta_rank_ic": float(rank_ic - np.corrcoef(base_rank[idx], net_rank[idx])[0, 1]),
                "top1_net_bps": float(top1), "top5_net_bps": float(top5),
                "top10_net_bps": float(top10),
                "delta_top1_net_bps": float(top1 - top1_base),
                "delta_top5_net_bps": float(top5 - top5_base),
                "delta_top10_net_bps": float(top10 - top10_base),
                "false_positive_rate": float(_weighted_rate(net <= 50.0, w)),
                "adverse_path_rate": float(_weighted_rate(net <= -50.0, w)),
                "execution_conversion_failure_rate": float(_weighted_rate((gross > 0.0) & (net <= 0.0), w)),
                "condition_membership_mean": float(w.mean()),
            })
    return pd.DataFrame(rows)


def _weighted_corr_matrix(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ww = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(yy) & np.isfinite(ww) & (ww > 0.0)
    if ok.sum() < 4:
        return np.full(x.shape[1], np.nan)
    x, yy, ww = x[ok], yy[ok], ww[ok]
    sw = ww.sum()
    mx = (ww[:, None] * x).sum(axis=0) / sw
    my = float(np.dot(ww, yy) / sw)
    dx, dy = x - mx, yy - my
    num = (ww[:, None] * dx * dy[:, None]).sum(axis=0) / sw
    den = np.sqrt((ww[:, None] * dx * dx).sum(axis=0) / sw * (ww * dy * dy).sum() / sw)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)


def _full_feature_response(side_frame: pd.DataFrame, candidates: pd.DataFrame, activations: dict[str, dict[str, np.ndarray]], predictive: list[str], top_n: int, side: str, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = candidates.sort_values(["candidate_screen_score", "effective_rows"], ascending=[False, False], kind="stable").head(top_n)
    fields = [f for f in predictive if f in side_frame]
    global_month_corr: dict[str, np.ndarray] = {}
    rank_features: dict[str, np.ndarray] = {}
    rank_net: dict[str, np.ndarray] = {}
    monthly_index: dict[str, np.ndarray] = {}
    for month, idx in side_frame.groupby("month", sort=True).groups.items():
        idx = np.asarray(idx, dtype=np.int64); monthly_index[str(month)] = idx
        sub = side_frame.iloc[idx]
        x = sub[fields].apply(pd.to_numeric, errors="coerce").rank(pct=True).fillna(.5).to_numpy(np.float32)
        y = sub.net_bps.rank(pct=True).to_numpy(np.float32)
        rank_features[str(month)] = x; rank_net[str(month)] = y
        global_month_corr[str(month)] = _weighted_corr_matrix(x, y, np.ones(len(idx), dtype=np.float32))
    rows: list[dict[str, Any]] = []
    for cond in ranked.to_dict("records"):
        w_all = _condition_weight(side_frame, cond, activations)
        for month, idx in monthly_index.items():
            w = w_all[idx]
            if effective_rows(w) < 50.0:
                continue
            sub = side_frame.iloc[idx]
            diff = _weighted_corr_matrix(rank_features[month], rank_net[month], w) - global_month_corr[month]
            for field, value in zip(fields, diff):
                j = fields.index(field)
                feature_rank = rank_features[month][:, j]
                net_values = sub.net_bps.to_numpy(float)
                gross_values = sub.gross_bps.to_numpy(float)
                top_mask = feature_rank >= 0.80
                bottom_mask = feature_rank <= 0.20
                rows.append({
                    "condition_id": cond["condition_id"], "side": side, "month": month,
                    "feature": field, "effective_rows": effective_rows(w), "differential_rank_ic": float(value) if np.isfinite(value) else np.nan,
                    "condition_rank_ic": float(value + global_month_corr[month][j]) if np.isfinite(value) else np.nan,
                    "top_feature_tail_net_bps": float(weighted_mean(net_values[top_mask], w[top_mask])) if top_mask.any() else np.nan,
                    "bottom_feature_tail_net_bps": float(weighted_mean(net_values[bottom_mask], w[bottom_mask])) if bottom_mask.any() else np.nan,
                    "opportunity_firing_rate": float(_weighted_rate(net_values > 50.0, w)),
                    "non_opportunity_firing_rate": float(_weighted_rate(net_values <= 50.0, w)),
                    "false_positive_rate": float(_weighted_rate((net_values <= 50.0) & top_mask, w)),
                    "adverse_path_rate": float(_weighted_rate((net_values <= -50.0) & top_mask, w)),
                    "execution_conversion_failure_rate": float(_weighted_rate((gross_values > 0.0) & (net_values <= 0.0) & top_mask, w)),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_parquet(out / f"condition_feature_behavior_monthly_{side}.parquet", index=False)
    portability_rows: list[dict[str, Any]] = []
    if not monthly.empty:
        for (condition_id, feature), g in monthly.groupby(["condition_id", "feature"], sort=True):
            vals = g.differential_rank_ic.to_numpy(float)
            portability_rows.append({
                "condition_id": condition_id, "side": side, "feature": feature,
                "supported_months": int(np.isfinite(vals).sum()),
                "portable_differential_rank_ic": portability_score(vals),
                "median_differential_rank_ic": float(np.nanmedian(vals)),
                "mad_differential_rank_ic": float(np.nanmedian(np.abs(vals - np.nanmedian(vals)))),
                "positive_month_fraction": float(np.mean(vals[np.isfinite(vals)] > 0.0)) if np.isfinite(vals).any() else np.nan,
                "worst_month_differential_rank_ic": float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
            })
    portability = pd.DataFrame(portability_rows)
    portability.to_parquet(out / f"condition_feature_portability_{side}.parquet", index=False)
    return monthly, portability


def _weighted_rank_ic_arrays(score: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Spearman proxy used by the discovery-only feature MDA.

    The expensive specialist model is fit once per condition on an earlier
    discovery slice.  MDA permutations are evaluated on a later slice and
    therefore never enter the transport folds or the OOF meta fit.
    """

    x = pd.Series(np.asarray(score, dtype=float)).rank(pct=True).to_numpy(float)
    y = pd.Series(np.asarray(outcome, dtype=float)).rank(pct=True).to_numpy(float)
    return weighted_corr(x, y, np.asarray(weights, dtype=float))


def _permute_within_query(
    matrix: np.ndarray,
    columns: list[int],
    queries: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute one feature group within query boundaries, preserving support."""

    result = np.asarray(matrix, dtype=np.float32).copy()
    query_values = queries.to_numpy()
    for _, positions in pd.Series(np.arange(len(query_values))).groupby(query_values, sort=False).groups.items():
        idx = np.asarray(positions, dtype=np.int64)
        if idx.size < 2:
            continue
        shuffled = idx[rng.permutation(idx.size)]
        result[np.ix_(idx, columns)] = result[np.ix_(shuffled, columns)]
    return result


def _condition_feature_mda_caps(
    dev_frame: pd.DataFrame,
    selected: list[dict[str, Any]],
    predictive: list[str],
    feature_portability: pd.DataFrame,
    activations: dict[str, dict[str, np.ndarray]],
    side: str,
    out: Path,
    cfg: ConditionalSpecialistConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run causal condition-weighted group MDA and the feature-cap funnel.

    This is deliberately a bounded discovery-stage proxy rather than another
    OOS HPO search.  Each condition is split chronologically into 60% fit,
    20% MDA validation and 20% cap validation.  A single incumbent LambdaRank
    specification is fit on the fit slice; group permutations are performed
    within 4-hour queries on the MDA slice, and each predeclared cap is
    refit/evaluated only on the final cap-validation slice.  The selected cap
    and feature order are consequently frozen before the transport pass.
    """

    mda_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    selected_caps: dict[str, int] = {}
    available = [f for f in predictive if f in dev_frame.columns]
    if not available or not selected:
        pd.DataFrame().to_parquet(out / f"condition_feature_mda_{side}.parquet", index=False)
        pd.DataFrame().to_parquet(out / f"condition_feature_cap_ablation_{side}.parquet", index=False)
        return pd.DataFrame(), pd.DataFrame(), selected_caps

    ordered_portable = {}
    if not feature_portability.empty:
        for cid, group in feature_portability.groupby("condition_id", sort=False):
            ordered_portable[str(cid)] = group.sort_values(
                ["portable_differential_rank_ic", "positive_month_fraction", "supported_months", "feature"],
                ascending=[False, False, False, True], kind="stable",
            ).feature.astype(str).tolist()

    for condition_index, condition in enumerate(selected):
        cid = str(condition["condition_id"])
        membership = _condition_weight(dev_frame, condition, activations).astype(np.float32)
        candidate_order = list(dict.fromkeys(ordered_portable.get(cid, []) + available))
        candidate_order = [f for f in candidate_order if f in available]
        # Avoid an unbounded candidate matrix while retaining the full
        # predeclared 40/60/80/100/120 cap funnel when the store supports it.
        candidate_order = candidate_order[: min(cfg.specialist_max_features, len(candidate_order))]
        if len(candidate_order) < max(2, cfg.specialist_min_features):
            selected_caps[cid] = min(len(candidate_order), min(cfg.specialist_feature_caps)) if candidate_order else 0
            continue

        time_order = np.argsort(pd.to_datetime(dev_frame.__ts__, utc=True).to_numpy(), kind="stable")
        n = len(time_order)
        fit_end = max(1, int(math.floor(0.60 * n)))
        mda_end = max(fit_end + 1, int(math.floor(0.80 * n)))
        fit_idx = time_order[:fit_end]
        mda_idx = time_order[fit_end:mda_end]
        cap_idx = time_order[mda_end:]
        if len(mda_idx) < 100 or len(cap_idx) < 100:
            selected_caps[cid] = min(len(candidate_order), min(cfg.specialist_feature_caps))
            continue

        fit_frame = dev_frame.iloc[fit_idx].copy()
        mda_frame = dev_frame.iloc[mda_idx].copy()
        cap_frame = dev_frame.iloc[cap_idx].copy()
        fit_membership = membership[fit_idx]
        mda_membership = membership[mda_idx]
        cap_membership = membership[cap_idx]
        fit_ok = np.isfinite(fit_membership) & (fit_membership > 0.01)
        mda_ok = np.isfinite(mda_membership) & (mda_membership > 0.01)
        cap_ok = np.isfinite(cap_membership) & (cap_membership > 0.01)
        if fit_ok.sum() < 100 or mda_ok.sum() < 100 or cap_ok.sum() < 100:
            selected_caps[cid] = min(len(candidate_order), min(cfg.specialist_feature_caps))
            continue

        # Group fields using only the fit slice; this is the same local
        # redundancy policy as final feature selection and prevents MDA from
        # spending permutations on aliases.
        fit_values = fit_frame.loc[:, candidate_order].apply(pd.to_numeric, errors="coerce")
        corr = fit_values.corr(method="spearman").abs()
        groups: list[list[str]] = []
        representatives: list[str] = []
        for field in candidate_order:
            placed = False
            for group, representative in zip(groups, representatives):
                value = corr.loc[field, representative] if field in corr.index and representative in corr.columns else np.nan
                if np.isfinite(value) and float(value) >= cfg.local_redundancy_spearman:
                    group.append(field)
                    placed = True
                    break
            if not placed:
                groups.append([field])
                representatives.append(field)
        # Keep a bounded model input for the permutation proxy.  The cap
        # ablation below still evaluates larger prefixes from the full ordered
        # candidate list.
        model_fields = representatives[: min(80, len(representatives))]
        if len(model_fields) < 2:
            selected_caps[cid] = min(len(candidate_order), min(cfg.specialist_feature_caps))
            continue
        fit_use = fit_frame.loc[fit_ok].copy()
        fit_w = _condition_month_balanced_weights(
            fit_use,
            fit_membership[fit_ok],
            exponent=cfg.condition_weight_exponent,
            equal_months=cfg.equal_condition_month_weighting,
        )
        try:
            mda_model, mda_fields, mda_medians, _ = _fit_ranker(
                fit_use, model_fields, fit_use.residual_grade.to_numpy(np.int32), fit_use.query_4h,
                SPECIALIST_PARAMS, fit_w,
            )
        except Exception:
            selected_caps[cid] = min(len(candidate_order), min(cfg.specialist_feature_caps))
            continue

        # Prepare the MDA matrix with exactly the medians used by the model.
        mda_matrix = mda_frame.loc[:, mda_fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mda_matrix = mda_matrix.fillna(pd.Series(mda_medians, index=mda_fields)).fillna(0.0).to_numpy(np.float32)
        mda_target = mda_frame.net_bps.to_numpy(float)
        mda_weight = mda_membership.astype(float)
        baseline_pred = mda_model.predict(mda_matrix).astype(np.float32)

        month_values = sorted(mda_frame.month.astype(str).unique().tolist())
        baseline_by_month: dict[str, tuple[float, float]] = {}
        for month in month_values:
            mask = mda_frame.month.astype(str).to_numpy() == month
            mask &= mda_ok
            if mask.sum() < 25:
                continue
            baseline_by_month[month] = (
                _weighted_rank_ic_arrays(baseline_pred[mask], mda_target[mask], mda_weight[mask]),
                _weighted_top_mean(baseline_pred[mask], mda_target[mask], mda_weight[mask], .10),
            )

        group_scores: dict[str, float] = {}
        field_scores: dict[str, float] = {}
        for group_index, group in enumerate(groups):
            model_group = [f for f in group if f in mda_fields]
            if not model_group:
                continue
            columns = [mda_fields.index(f) for f in model_group]
            deltas_rank: list[float] = []
            deltas_top: list[float] = []
            for repeat in range(int(cfg.group_mda_repeats)):
                rng = np.random.default_rng(SEED + condition_index * 10000 + group_index * 100 + repeat)
                permuted = _permute_within_query(mda_matrix, columns, mda_frame.query_4h, rng)
                perm_pred = mda_model.predict(permuted).astype(np.float32)
                for month, (base_rank_ic, base_top10) in baseline_by_month.items():
                    mask = mda_frame.month.astype(str).to_numpy() == month
                    mask &= mda_ok
                    if mask.sum() < 25:
                        continue
                    perm_rank_ic = _weighted_rank_ic_arrays(perm_pred[mask], mda_target[mask], mda_weight[mask])
                    perm_top10 = _weighted_top_mean(perm_pred[mask], mda_target[mask], mda_weight[mask], .10)
                    deltas_rank.append(float(base_rank_ic - perm_rank_ic))
                    deltas_top.append(float(base_top10 - perm_top10))
                    mda_rows.append({
                        "condition_id": cid, "side": side, "group_id": int(group_index),
                        "group_features": json.dumps(group), "representative": model_group[0],
                        "repeat": str(repeat), "month": str(month),
                        "mda_rank_ic": float(base_rank_ic - perm_rank_ic),
                        "mda_top10_net_bps": float(base_top10 - perm_top10),
                    })
            portable_rank = portability_score(deltas_rank)
            portable_top = portability_score(deltas_top)
            score = float(portable_rank + portable_top / 1000.0) if np.isfinite(portable_rank) and np.isfinite(portable_top) else float("nan")
            group_scores[str(group_index)] = score
            for field in group:
                field_scores[field] = score

        # Prefer the portable MDA order; keep any unscored fields at the end
        # in their label-free portable-response order.
        ordered = sorted(candidate_order, key=lambda field: (-float(field_scores.get(field, -np.inf)), field))
        ordered = [f for f in ordered if f in candidate_order]
        cap_month_scores: dict[int, list[float]] = {}
        for cap in cfg.specialist_feature_caps:
            cap_fields = ordered[: min(int(cap), len(ordered))]
            if len(cap_fields) < 2:
                continue
            cap_fit = fit_frame.loc[fit_ok].copy()
            cap_w = fit_membership[fit_ok]
            try:
                cap_model, used, med, _ = _fit_ranker(
                    cap_fit, cap_fields, cap_fit.residual_grade.to_numpy(np.int32), cap_fit.query_4h,
                    SPECIALIST_PARAMS,
                    _condition_month_balanced_weights(cap_fit, cap_w, exponent=cfg.condition_weight_exponent, equal_months=cfg.equal_condition_month_weighting),
                )
            except Exception:
                continue
            cap_matrix = cap_frame.loc[:, used].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            cap_matrix = cap_matrix.fillna(pd.Series(med, index=used)).fillna(0.0).to_numpy(np.float32)
            cap_pred = cap_model.predict(cap_matrix).astype(np.float32)
            cap_values = []
            for month in sorted(cap_frame.month.astype(str).unique().tolist()):
                mask = cap_frame.month.astype(str).to_numpy() == month
                mask &= cap_ok
                if mask.sum() < 25:
                    continue
                rank_ic = _weighted_rank_ic_arrays(cap_pred[mask], cap_frame.net_bps.to_numpy(float)[mask], cap_membership[mask])
                top10 = _weighted_top_mean(cap_pred[mask], cap_frame.net_bps.to_numpy(float)[mask], cap_membership[mask], .10)
                cap_values.append(float(top10))
                cap_month_scores.setdefault(int(cap), []).append(float(top10))
                cap_rows.append({
                    "condition_id": cid, "side": side, "cap": int(cap), "feature_count": int(len(used)),
                    "fields": json.dumps(used), "month": str(month),
                    "validation_rank_ic": float(rank_ic), "validation_top10_net_bps": float(top10),
                })
            if cap_values:
                cap_rows.append({
                    "condition_id": cid, "side": side, "cap": int(cap), "feature_count": int(len(used)),
                    "fields": json.dumps(used), "month": "__portable__",
                    "validation_rank_ic": np.nan,
                    "validation_top10_net_bps": float(portability_score(cap_values)),
                })

        if cap_month_scores:
            # The cap is selected on the held-out cap slice only.  Prefer the
            # smallest cap within one standard error of the best portable
            # top-tail value, which keeps specialists compact.
            cap_scores = {cap: portability_score(values) for cap, values in cap_month_scores.items()}
            best = max(cap_scores.values())
            best_caps = [cap for cap, value in cap_scores.items() if value >= best - 0.5 * max(1.0, abs(best) / math.sqrt(max(1, len(cap_month_scores))))]
            selected_caps[cid] = int(min(best_caps))
            for row in cap_rows:
                if row.get("condition_id") == cid and row.get("month") == "__portable__":
                    row["selected"] = bool(int(row["cap"]) == selected_caps[cid])

        for group_index, group in enumerate(groups):
            score = group_scores.get(str(group_index), np.nan)
            for field in group:
                mda_rows.append({
                    "condition_id": cid, "side": side, "group_id": int(group_index),
                    "group_features": json.dumps(group), "representative": group[0],
                    "repeat": "portable", "month": "__portable__",
                    "mda_rank_ic": np.nan, "mda_top10_net_bps": np.nan,
                    "portable_mda_score": float(score) if np.isfinite(score) else np.nan,
                    "feature": field,
                })

    mda = pd.DataFrame(mda_rows)
    caps = pd.DataFrame(cap_rows)
    mda.to_parquet(out / f"condition_feature_mda_{side}.parquet", index=False)
    caps.to_parquet(out / f"condition_feature_cap_ablation_{side}.parquet", index=False)
    _write_json(out / f"condition_feature_mda_manifest_{side}.json", {
        "schema": "causal_condition_weighted_group_mda_v1",
        "side": side,
        "fit_fraction": 0.60,
        "mda_validation_fraction": 0.20,
        "cap_validation_fraction": 0.20,
        "repeats": int(cfg.group_mda_repeats),
        "caps": list(cfg.specialist_feature_caps),
        "selection": "smallest_cap_within_one_se_of_best_portable_top10",
        "transport_included": False,
    })
    return mda, caps, selected_caps


def _build_response_signatures(candidates: pd.DataFrame, model_monthly: pd.DataFrame, feature_portability: pd.DataFrame, side: str, out: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if candidates.empty:
        return pd.DataFrame(), {}
    cond_ids = candidates.condition_id.tolist()
    model_rows: list[dict[str, Any]] = []
    model_vectors: dict[str, np.ndarray] = {}
    for cid in cond_ids:
        g = model_monthly[model_monthly.condition_id.eq(cid)] if not model_monthly.empty else pd.DataFrame()
        top1 = g.delta_top1_net_bps.to_numpy(float) if not g.empty else np.array([], dtype=float)
        top5 = g.delta_top5_net_bps.to_numpy(float) if not g.empty else np.array([], dtype=float)
        top = g.delta_top10_net_bps.to_numpy(float) if not g.empty else np.array([], dtype=float)
        ric = g.delta_rank_ic.to_numpy(float) if not g.empty else np.array([], dtype=float)
        event = g.event_lift.to_numpy(float) if not g.empty else np.array([], dtype=float)
        vector = np.asarray([portability_score(top1), portability_score(top5), portability_score(top), portability_score(ric), portability_score(event)], dtype=np.float64)
        model_vectors[cid] = vector
        model_rows.append({
            "condition_id": cid, "side": side,
            "portable_delta_top1_net_bps": float(vector[0]), "portable_delta_top5_net_bps": float(vector[1]), "portable_delta_top10_net_bps": float(vector[2]), "portable_delta_rank_ic": float(vector[3]), "portable_event_lift": float(vector[4]),
            "supported_months": int(g.month.nunique()) if not g.empty else 0,
        })
    model_port = pd.DataFrame(model_rows)
    model_port.to_parquet(out / f"condition_model_utility_portability_{side}.parquet", index=False)
    pivot = feature_portability.pivot(index="condition_id", columns="feature", values="portable_differential_rank_ic") if not feature_portability.empty else pd.DataFrame(index=cond_ids)
    pivot = pivot.reindex(cond_ids).fillna(0.0)
    signatures = pivot.to_numpy(np.float32)
    if signatures.shape[0] >= 2 and signatures.shape[1] >= 2:
        ncomp = max(1, min(16, signatures.shape[0] - 1, signatures.shape[1]))
        svd = TruncatedSVD(n_components=ncomp, random_state=SEED).fit(signatures)
        compressed = svd.transform(signatures)
    else:
        compressed = signatures
    rows: list[dict[str, Any]] = []
    for i, cid in enumerate(cond_ids):
        rec = {"condition_id": cid, "side": side, "feature_response_dim": int(compressed.shape[1])}
        for j, value in enumerate(compressed[i]):
            rec[f"feature_response_svd_{j:02d}"] = float(value)
        vec = model_vectors[cid]
        rec.update({"model_delta_top1": float(vec[0]), "model_delta_top5": float(vec[1]), "model_delta_top10": float(vec[2]), "model_delta_rank_ic": float(vec[3]), "model_event_lift": float(vec[4])})
        rows.append(rec)
    result = pd.DataFrame(rows)
    result.to_parquet(out / f"condition_response_signatures_{side}.parquet", index=False)
    return result, model_vectors


def _select_conditions(candidates: pd.DataFrame, signatures: pd.DataFrame, model_vectors: dict[str, np.ndarray], side_frame: pd.DataFrame, activations: dict[str, dict[str, np.ndarray]], cfg: ConditionalSpecialistConfig, side: str, out: Path) -> list[dict[str, Any]]:
    if candidates.empty:
        return []
    fp = candidates.copy()
    feature_port = pd.read_parquet(out / f"condition_feature_portability_{side}.parquet") if (out / f"condition_feature_portability_{side}.parquet").exists() else pd.DataFrame()
    if feature_port.empty:
        feature_diff = pd.Series(0.0, index=fp.condition_id)
    else:
        feature_diff = feature_port.groupby("condition_id").portable_differential_rank_ic.apply(lambda x: float(np.nanmean(np.abs(x))))
    model_port = pd.read_parquet(out / f"condition_model_utility_portability_{side}.parquet")
    model_diff = model_port.set_index("condition_id").portable_delta_top10_net_bps.abs() + 100.0 * model_port.set_index("condition_id").portable_delta_rank_ic.abs()
    fp["feature_differentiation"] = fp.condition_id.map(feature_diff).fillna(0.0)
    fp["model_differentiation"] = fp.condition_id.map(model_diff).fillna(0.0)
    fp["recurrence"] = fp.supported_month_count.clip(0, 12) / 12.0
    fp["relevance"] = (0.25 * fp.pair_interaction.fillna(0.0) + 0.25 * fp.feature_differentiation + 0.25 * fp.model_differentiation + 0.15 * fp.recurrence + 0.10 * np.clip(fp.joint_activation_hard_share, 0.0, 1.0))
    fp = fp.sort_values(["relevance", "effective_rows"], ascending=[False, False], kind="stable")
    selected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    signature_fields = [c for c in signatures.columns if c.startswith("feature_response_svd_")]
    sig_by = signatures.set_index("condition_id")[signature_fields] if (not signatures.empty and signature_fields) else pd.DataFrame()
    for _, row in fp.iterrows():
        cid = str(row.condition_id)
        if any(cid == str(x["condition_id"]) for x in selected):
            continue
        overlaps: list[float] = []
        comp: list[float] = []
        candidate_signature = sig_by.loc[cid].to_numpy(float) if cid in sig_by.index else np.zeros(1)
        for prior in selected:
            pid = str(prior["condition_id"])
            prior_signature = sig_by.loc[pid].to_numpy(float) if pid in sig_by.index else np.zeros(1)
            cond_a = row.to_dict(); cond_b = prior
            overlaps.append(weighted_jaccard(_condition_weight(side_frame, cond_a, activations), _condition_weight(side_frame, cond_b, activations)))
            comp.append(0.45 * cosine_distance(np.asarray(model_vectors.get(cid, np.zeros(3))), np.asarray(model_vectors.get(pid, np.zeros(3)))) + 0.35 * cosine_distance(candidate_signature, prior_signature) + 0.20 * (1.0 - overlaps[-1]))
        max_overlap = max(overlaps) if overlaps else 0.0
        complementarity = float(np.mean(comp)) if comp else 0.0
        marginal = float(row.relevance) + 0.35 * complementarity - 0.25 * max_overlap
        accepted = bool(len(selected) < 3 or (marginal > 0.0 and max_overlap <= 0.75))
        trace.append({"condition_id": cid, "side": side, "relevance": float(row.relevance), "marginal_gain": marginal, "max_membership_overlap": max_overlap, "complementarity": complementarity, "accepted": accepted, "selected_count_before": len(selected)})
        if accepted:
            selected.append(row.to_dict())
        if len(selected) >= 3:
            break
    result = pd.DataFrame(selected)
    result.to_parquet(out / f"condition_selection_trace_{side}.parquet", index=False) if result.empty else None
    pd.DataFrame(trace).to_parquet(out / f"condition_selection_trace_{side}.parquet", index=False)
    _write_json(out / f"selected_conditions_{side}.json", {"schema": "frozen_selected_pair_conditions_v1", "side": side, "conditions": selected, "selection": "relevance_plus_complementarity_greedy", "discovery_end_utc": DISCOVERY_END.isoformat()})
    return selected


def _select_condition_features(
    dev_frame: pd.DataFrame,
    selected: list[dict[str, Any]],
    feature_portability: pd.DataFrame,
    predictive: list[str],
    side: str,
    out: Path,
    cfg: ConditionalSpecialistConfig,
    *,
    mda: pd.DataFrame | None = None,
    selected_caps: dict[str, int] | None = None,
    method: str = "rank_portability",
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    audits: list[dict[str, Any]] = []
    for cond in selected:
        cid = str(cond["condition_id"])
        g = feature_portability[feature_portability.condition_id.eq(cid)].copy() if not feature_portability.empty else pd.DataFrame()
        if method == "condition_group_mda" and mda is not None and not mda.empty:
            gm = mda[(mda.condition_id.astype(str) == cid) & (mda.month.astype(str) == "__portable__")].copy()
            if not gm.empty and "portable_mda_score" in gm:
                mda_order = gm.sort_values(["portable_mda_score", "feature"], ascending=[False, True], kind="stable").feature.astype(str).tolist()
                ordered = list(dict.fromkeys(mda_order + (g.sort_values(["portable_differential_rank_ic", "positive_month_fraction", "supported_months"], ascending=[False, False, False], kind="stable").feature.astype(str).tolist() if not g.empty else [])))
            else:
                ordered = g.sort_values(["portable_differential_rank_ic", "positive_month_fraction", "supported_months"], ascending=[False, False, False], kind="stable").feature.tolist() if not g.empty else []
        else:
            ordered = g.sort_values(["portable_differential_rank_ic", "positive_month_fraction", "supported_months"], ascending=[False, False, False], kind="stable").feature.tolist() if not g.empty else []
        ordered += [f for f in predictive if f not in ordered]
        keep: list[str] = []
        cap = int((selected_caps or {}).get(cid, min(cfg.specialist_feature_caps)))
        cap = max(cfg.specialist_min_features, min(cfg.specialist_max_features, cap))
        for field in ordered:
            if field not in dev_frame or len(keep) >= cap:
                continue
            if keep:
                corr = dev_frame.loc[:, [*keep, field]].apply(pd.to_numeric, errors="coerce").corr(method="spearman").iloc[-1, :-1].abs().max()
                if np.isfinite(corr) and float(corr) >= 0.98:
                    audits.append({"condition_id": cid, "feature": field, "selected": False, "reason": "within_condition_redundant"})
                    continue
            keep.append(field)
            audits.append({"condition_id": cid, "feature": field, "selected": True, "rank": len(keep), "reason": "portable_differential"})
            if len(keep) >= cap:
                break
        if len(keep) < cfg.specialist_min_features:
            keep = [f for f in predictive if f in dev_frame][:cfg.specialist_min_features]
        result[cid] = keep[:cfg.specialist_max_features]
    _write_json(out / f"condition_feature_sets_{side}.json", {
        "schema": "frozen_condition_specific_feature_sets_v1",
        "side": side,
        "sets": result,
        "selected_caps": selected_caps or {},
        "selection_method": method,
        "feature_cap_ablation": list(cfg.specialist_feature_caps),
    })
    pd.DataFrame(audits).to_parquet(out / f"condition_feature_selection_{side}.parquet", index=False)
    return result


def _prepare_condition_outputs(frame: pd.DataFrame, side: str, selected: list[dict[str, Any]], activation_manifest: dict[str, Any], spine: list[str], feature_sets: dict[str, list[str]], *, train_models: dict[str, Any] | None = None, specialist_config: ConditionalSpecialistConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train or apply one specialist per selected condition for one side."""

    fields = list(dict.fromkeys(spine + [f for values in feature_sets.values() for f in values]))
    joined = _store_rows(frame, fields)
    work = frame.merge(joined, on="candidate_id", validate="one_to_one")
    memberships = _apply_manifest_memberships(work, activation_manifest, spine)
    outputs = work[["candidate_id", "__ts__", "side_name", "base_score", "base_ev_bps", "residual_bps", "residual_grade", "query_4h"]].copy()
    models = {} if train_models is None else train_models
    cfg = specialist_config or ConditionalSpecialistConfig(global_seed=SEED)
    prior_raw_fields: list[str] = []
    for cond in selected:
        cid = str(cond["condition_id"]); name = cid.replace("/", "_")
        w = _condition_weight(work, cond, memberships)
        fields_cond = feature_sets[cid]
        if train_models is None:
            fit_mask = np.isfinite(w) & (w > 0.01)
            fit = work.loc[fit_mask].copy()
            if len(fit) > MAX_TRAIN_ROWS:
                # Python's built-in hash is process-randomised; use a stable
                # digest so the frozen specialist sample is reproducible.
                stable_offset = int.from_bytes(hashlib.sha256(cid.encode("utf-8")).digest()[:4], "little") % 1000
                fit = _sample(fit, MAX_TRAIN_ROWS, seed=SEED + stable_offset)
            fit_w = _condition_weight(fit, cond, _apply_manifest_memberships(fit, activation_manifest, spine))
            rank_weights = _condition_month_balanced_weights(
                fit, fit_w, exponent=cfg.condition_weight_exponent,
                equal_months=cfg.equal_condition_month_weighting,
            )
            model, used, med, _ = _fit_ranker(fit, fields_cond, fit.residual_grade.to_numpy(np.int32), fit.query_4h, SPECIALIST_PARAMS, rank_weights)
            raw_fit = _predict(model, fit, used, med)
            residualizer_fields = ["base_score", *prior_raw_fields]
            fit_design = np.column_stack([
                np.ones(len(fit), dtype=np.float32),
                fit.loc[:, residualizer_fields].to_numpy(np.float32) if not prior_raw_fields else np.column_stack([
                    fit.base_score.to_numpy(np.float32),
                    outputs.loc[fit.index, prior_raw_fields].to_numpy(np.float32),
                ]),
            ])
            ok_fit = np.isfinite(raw_fit) & np.isfinite(fit_design).all(axis=1)
            beta = np.linalg.lstsq(fit_design[ok_fit], raw_fit[ok_fit], rcond=None)[0] if ok_fit.sum() >= max(20, fit_design.shape[1] + 2) else np.zeros(fit_design.shape[1], dtype=np.float32)
            models[cid] = {"model": model, "fields": used, "median": med, "condition": cond, "residualizer_fields": residualizer_fields, "residualizer_beta": beta.astype(np.float32)}
        model_info = models[cid]
        raw = _predict(model_info["model"], work, model_info["fields"], model_info["median"])
        outputs[f"condition__{name}__raw"] = raw
        outputs[f"condition__{name}__rank"] = _within_query_rank(raw, outputs["query_4h"])
        outputs[f"condition__{name}__membership"] = w.astype(np.float32)
        gated = (raw * np.power(np.clip(w, 0.0, 1.0), cfg.condition_weight_exponent)).astype(np.float32)
        outputs[f"condition__{name}__gated"] = gated
        outputs[f"condition__{name}__gated_rank"] = _within_query_rank(gated, outputs["query_4h"])
        hard_gated = np.where(w >= 0.5, raw, 0.0).astype(np.float32)
        outputs[f"condition__{name}__hard_gated"] = hard_gated
        outputs[f"condition__{name}__hard_rank"] = _within_query_rank(hard_gated, outputs["query_4h"])
        residualizer_fields = list(model_info.get("residualizer_fields", ["base_score"]))
        beta = np.asarray(model_info.get("residualizer_beta", [0.0, 1.0]), dtype=float)
        design_columns = [np.ones(len(work), dtype=np.float32), work.base_score.to_numpy(np.float32)]
        for previous_field in residualizer_fields[1:]:
            design_columns.append(outputs[previous_field].to_numpy(np.float32))
        design = np.column_stack(design_columns)
        if beta.shape[0] != design.shape[1]:
            beta = np.zeros(design.shape[1], dtype=np.float32)
        innovation = raw - np.asarray(design @ beta, dtype=np.float32)
        outputs[f"condition__{name}__innovation"] = innovation.astype(np.float32)
        outputs[f"condition__{name}__innovation_rank"] = _within_query_rank(innovation, outputs["query_4h"])
        gated_innovation = (innovation * np.power(np.clip(w, 0.0, 1.0), cfg.condition_weight_exponent)).astype(np.float32)
        outputs[f"condition__{name}__gated_innovation"] = gated_innovation
        outputs[f"condition__{name}__gated_innovation_rank"] = _within_query_rank(gated_innovation, outputs["query_4h"])
        hard_gated_innovation = np.where(w >= 0.5, innovation, 0.0).astype(np.float32)
        outputs[f"condition__{name}__hard_gated_innovation"] = hard_gated_innovation
        outputs[f"condition__{name}__hard_innovation_rank"] = _within_query_rank(hard_gated_innovation, outputs["query_4h"])
        # OOD is a transparent membership/support indicator, not an inference
        # target: low membership means the pair state is weakly identified.
        outputs[f"condition__{name}__uncertainty"] = (1.0 - np.clip(w, 0.0, 1.0)).astype(np.float32)
        outputs[f"condition__{name}__ood"] = (w < 0.10).astype(np.float32)
        prior_raw_fields.append(f"condition__{name}__raw")
    return outputs, models


def _meta_arm_fields(outputs: pd.DataFrame, spine: list[str], selected: list[dict[str, Any]], arm: str) -> list[str]:
    base = ["base_score", "base_ev_bps"]
    names = [str(c["condition_id"]).replace("/", "_") for c in selected]
    if arm == "anchor_only":
        return base
    if arm == "ridge_blend":
        # A regularized linear blend of the frozen specialist ranks.  It is
        # intentionally not given the full context spine.
        return list(dict.fromkeys(base + [
            f"condition__{name}__rank" for name in names
            if f"condition__{name}__rank" in outputs
        ]))
    if arm in {"hard_gating", "hard_gated_ranks"}:
        return list(dict.fromkeys(base + [
            f"condition__{name}__hard_rank" for name in names
            if f"condition__{name}__hard_rank" in outputs
        ]))
    if arm in {"probability_only", "condition_probability_only"}:
        return list(dict.fromkeys(base + [
            c for name in names for c in (
                f"condition__{name}__membership",
                f"condition__{name}__uncertainty",
                f"condition__{name}__ood",
            ) if c in outputs
        ]))
    if arm == "gmm_geometry":
        return list(dict.fromkeys(base + [c for c in outputs.columns if c.startswith("geometry_gmm_")]))
    if arm == "full_context_gmm":
        fields = base + spine + [c for c in outputs.columns if c.startswith("geometry_gmm_")]
        for name in names:
            fields += [c for c in outputs.columns if c.startswith(f"condition__{name}__") and any(c.endswith(s) for s in ("__rank", "__gated_rank", "__hard_rank", "__innovation_rank", "__hard_innovation_rank", "__gated_innovation_rank", "__gated_innovation", "__hard_gated_innovation", "__uncertainty", "__ood", "__membership"))]
        return list(dict.fromkeys([f for f in fields if f in outputs]))
    if arm == "memberships":
        suffix = "__membership"
    elif arm == "raw_ranks":
        suffix = "__rank"
    elif arm == "gated_ranks":
        suffix = "__gated_rank"
    elif arm == "innovations":
        suffix = "__innovation"
    elif arm == "gated_innovations":
        suffix = "__gated_innovation"
    else:
        fields = base + spine
        for name in names:
            fields += [c for c in outputs.columns if c.startswith(f"condition__{name}__") and any(c.endswith(s) for s in ("__rank", "__gated_rank", "__hard_rank", "__innovation_rank", "__hard_innovation_rank", "__gated_innovation_rank", "__gated_innovation", "__hard_gated_innovation", "__uncertainty", "__ood", "__membership"))]
        return list(dict.fromkeys([f for f in fields if f in outputs]))
    return list(dict.fromkeys(base + [f"condition__{name}{suffix}" for name in names if f"condition__{name}{suffix}" in outputs]))


def _map_meta_residual(eval_frame: pd.DataFrame, raw: np.ndarray, side: str) -> np.ndarray:
    ts = pd.to_datetime(eval_frame["__ts__"], utc=True).to_numpy(dtype="datetime64[ns]")
    order = np.argsort(ts, kind="stable")
    q = eval_frame.iloc[order].copy()
    score = np.tanh(np.asarray(raw, dtype=float)[order])
    values, _, _ = prequential_same_side_r3_value_map(
        exact_net_bps=q.residual_bps.to_numpy(float),
        decision_timestamps=q.__ts__, label_available_timestamps=q.label_available_ts,
        side=side, score=score,
        config=PrequentialR3ValueMapConfig(side=side, bins=20, min_global_rows=32, bin_shrink_rows=64, mapping_mode="monotone_pava", monotone_min_bin_rows=1),
    )
    out = np.empty(len(q), dtype=np.float32); out[order] = np.clip(values, -50.0, 50.0)
    return out


def _fit_meta_arm(train: pd.DataFrame, cal_map: pd.DataFrame, test: pd.DataFrame, fields: list[str], side: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    if arm == "anchor_only":
        return np.zeros(len(cal_map), dtype=np.float32), np.zeros(len(test), dtype=np.float32)
    fit = train.copy()
    target = fit.residual_grade.to_numpy(np.int32)
    model, used, med, _ = _fit_ranker(fit, fields, target, fit.query_4h, META_PARAMS)
    raw_cal = _predict(model, cal_map, used, med)
    raw_test = _predict(model, test, used, med)
    combined = pd.concat([cal_map, test], ignore_index=True)
    raw = np.concatenate([raw_cal, raw_test])
    mapped = _map_meta_residual(combined, raw, side)
    return mapped[: len(cal_map)], mapped[len(cal_map) :]


def _run_outer(base: pd.DataFrame, spine_manifests: dict[str, dict[str, Any]], spines: dict[str, list[str]], feature_sets: dict[str, dict[str, list[str]]], selected: dict[str, list[dict[str, Any]]], out: Path, *, specialist_config: ConditionalSpecialistConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    specialist_rows: list[pd.DataFrame] = []
    geometry_control_rows: list[dict[str, Any]] = []
    arms = ("anchor_only", "memberships", "raw_ranks", "gated_ranks", "innovations", "gated_innovations", "hard_gating", "probability_only", "full_context", "ridge_blend", "gmm_geometry", "full_context_gmm")
    for fold in TRANSPORT_FOLDS:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        train = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)].copy()
        cal = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)].copy()
        test = base[base.__ts__.between(c, e, inclusive="left")].copy()
        fold_side: list[pd.DataFrame] = []
        for side in ("long", "short"):
            if not selected[side]:
                continue
            tr = train[train.side_name.eq(side)].copy(); ca = cal[cal.side_name.eq(side)].copy(); te = test[test.side_name.eq(side)].copy()
            # Specialist models are fit on train; all output rows are then
            # aligned by candidate_id before the meta split.
            tr_out, models = _prepare_condition_outputs(tr, side, selected[side], spine_manifests[side], spines[side], feature_sets[side], specialist_config=specialist_config)
            ca_out, _ = _prepare_condition_outputs(ca, side, selected[side], spine_manifests[side], spines[side], feature_sets[side], train_models=models, specialist_config=specialist_config)
            te_out, _ = _prepare_condition_outputs(te, side, selected[side], spine_manifests[side], spines[side], feature_sets[side], train_models=models, specialist_config=specialist_config)
            # Join the causal context only after specialists are materialised;
            # this avoids retaining the full store on every model object.
            context_fields = spines[side]
            tr_ctx = tr.merge(_store_rows(tr, context_fields), on="candidate_id", validate="one_to_one")
            ca_ctx = ca.merge(_store_rows(ca, context_fields), on="candidate_id", validate="one_to_one")
            te_ctx = te.merge(_store_rows(te, context_fields), on="candidate_id", validate="one_to_one")
            # Geometry-only GMM is a strict fold-local control.  It is fit on
            # the training context and transformed forward before any meta
            # ranker sees it.
            tr_ctx, ca_ctx, te_ctx, geometry_fields, geometry_meta = _add_fold_gmm_geometry(
                tr_ctx, ca_ctx, te_ctx, side=side
            )
            geometry_control_rows.append({"fold": fold.name, **geometry_meta})
            tr_all = tr_ctx.merge(tr_out.drop(columns=["side_name", "query_4h", "base_score", "base_ev_bps", "residual_bps", "residual_grade"], errors="ignore"), on=["candidate_id", "__ts__"], validate="one_to_one")
            ca_all = ca_ctx.merge(ca_out.drop(columns=["side_name", "query_4h", "base_score", "base_ev_bps", "residual_bps", "residual_grade"], errors="ignore"), on=["candidate_id", "__ts__"], validate="one_to_one")
            te_all = te_ctx.merge(te_out.drop(columns=["side_name", "query_4h", "base_score", "base_ev_bps", "residual_bps", "residual_grade"], errors="ignore"), on=["candidate_id", "__ts__"], validate="one_to_one")
            tr_all["fold"] = fold.name; ca_all["fold"] = fold.name; te_all["fold"] = fold.name
            # Use the first half of calibration for model fitting and the later
            # half exclusively for residual value-map resolution.
            ca_order = ca_all.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
            split = max(2, len(ca_order) // 2)
            ca_fit = ca_order.iloc[:split].copy(); ca_map = ca_order.iloc[split:].copy()
            for arm in arms:
                fields = _meta_arm_fields(tr_all, context_fields, selected[side], arm)
                if arm == "anchor_only":
                    ca_corr = np.zeros(len(ca_map), dtype=np.float32); te_corr = np.zeros(len(te_all), dtype=np.float32)
                elif arm == "ridge_blend":
                    ca_raw, te_raw = _fit_ridge_residual_arm(ca_fit, ca_map, te_all, fields, side)
                    combined = pd.concat([ca_map, te_all], ignore_index=True)
                    mapped = _map_meta_residual(combined, np.concatenate([ca_raw, te_raw]), side)
                    ca_corr, te_corr = mapped[: len(ca_map)], mapped[len(ca_map) :]
                else:
                    ca_corr, te_corr = _fit_meta_arm(ca_fit, ca_map, te_all, fields, side, arm)
                z = te_all[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "base_ev_bps", "residual_bps", "fold"]].copy()
                z[f"score__{arm}"] = z.base_ev_bps.to_numpy(np.float32) + te_corr
                z["arm"] = arm
                fold_side.append(z)
            spec_cols = [c for c in te_out.columns if c.startswith("condition__")]
            spec = te_out[["candidate_id", *spec_cols]].copy(); spec["fold"] = fold.name; spec["side_name"] = side
            specialist_rows.append(spec)
            del models, tr_out, ca_out, te_out, tr_ctx, ca_ctx, te_ctx, tr_all, ca_all, te_all
            gc.collect()
        if fold_side:
            combined = pd.concat(fold_side, ignore_index=True)
            # Each arm is represented once per row; this is the authoritative
            # global ranking frame for the fold.
            for arm in arms:
                score_col = f"score__{arm}"
                arm_frame = combined[combined.arm.eq(arm)].copy()
                arm_frame["month"] = pd.to_datetime(arm_frame.__ts__, utc=True).dt.strftime("%Y-%m")
                metrics += _score_metrics(arm_frame, score_col, scope="global", period=fold.name, system=arm)
                for month, month_frame in arm_frame.groupby("month", sort=True):
                    metrics += _score_metrics(month_frame, score_col, scope="global", period=month, system=arm)
                for side, side_frame in arm_frame.groupby("side_name", sort=True):
                    metrics += _score_metrics(side_frame, score_col, scope=f"side:{side}", period=fold.name, system=arm)
            predictions.append(combined)
            pd.DataFrame(metrics).to_parquet(out / "metrics.checkpoint.parquet", index=False)
            pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.checkpoint.parquet", index=False)
            _write_json(out / "progress.json", {"status": "running", "completed_fold": fold.name})
    pred = pd.concat(predictions, ignore_index=True)
    # Collapse arm duplication into one wide prediction table.
    id_cols = ["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "base_ev_bps", "residual_bps", "fold"]
    wide = pred[id_cols].drop_duplicates(["candidate_id", "fold"], keep="first").copy()
    for arm in arms:
        scores = pred.loc[pred.arm.eq(arm), id_cols[:2] + ["fold", f"score__{arm}"]].drop_duplicates(["candidate_id", "fold"])
        wide = wide.merge(scores, on=["candidate_id", "__ts__", "fold"], how="left", validate="one_to_one")
    wide.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    specs = pd.concat(specialist_rows, ignore_index=True) if specialist_rows else pd.DataFrame()
    specs.to_parquet(out / "condition_specialist_oof.parquet", index=False)
    _write_json(out / "geometry_control_folds.json", geometry_control_rows)
    return wide, pd.DataFrame(metrics), specs


def _load_selection_control_conditions(
    out: Path,
    side: str,
    control: str,
    *,
    maximum_conditions: int = 1,
) -> list[dict[str, Any]]:
    """Load frozen discovery-only definitions for a control replay."""
    audit_path = out / "control_selection_audit.parquet"
    candidate_path = out / f"condition_candidates_{side}.parquet"
    if not audit_path.exists() or not candidate_path.exists():
        return []
    audit = pd.read_parquet(audit_path)
    candidates = pd.read_parquet(candidate_path)
    rows = audit[(audit.control.astype(str) == control) & (audit.side.astype(str) == side)]
    rows = rows.sort_values(["selection_rank", "effective_rows", "condition_id"], kind="stable")
    candidate_by_id = {str(x["condition_id"]): x for x in candidates.to_dict("records")}
    result: list[dict[str, Any]] = []
    for rec in rows.to_dict("records"):
        cid = str(rec.get("condition_id", ""))
        if not cid:
            continue
        if control == "univariate":
            encoded = cid.removeprefix("univariate__")
            try:
                field, region = encoded.rsplit("__", 1)
            except ValueError:
                continue
            condition = {
                "condition_id": f"{side}__control__univariate__{field}__{region}",
                "side": side,
                "context_feature_a": field,
                "activation_a": region,
                "context_feature_b": None,
                "activation_b": None,
                "unary": True,
                "effective_rows": float(rec.get("effective_rows", np.nan)),
                "supported_month_count": int(rec.get("supported_month_count", 0)),
            }
        else:
            condition = candidate_by_id.get(cid)
            if condition is None:
                continue
            condition = dict(condition)
            condition["condition_id"] = f"{side}__control__{control}__{cid}"
            condition["control_source_condition_id"] = cid
            condition["control"] = control
        result.append(condition)
        if len(result) >= int(maximum_conditions):
            break
    return result


def _run_selection_control_replay(
    base: pd.DataFrame,
    spine_manifests: dict[str, dict[str, Any]],
    spines: dict[str, list[str]],
    predictive_by_side: dict[str, list[str]],
    out: Path,
    *,
    specialist_config: ConditionalSpecialistConfig | None = None,
    control_names: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run bounded train-only OOS replays for discovery-selection controls."""
    cfg = specialist_config or ConditionalSpecialistConfig(global_seed=SEED)
    controls = control_names or ("random_supported", "geometry_only", "no_model_utility", "no_feature_behavior", "univariate")
    definitions = {
        control: {
            side: _load_selection_control_conditions(out, side, control, maximum_conditions=1)
            for side in ("long", "short")
        }
        for control in controls
    }
    for control in controls:
        for side in ("long", "short"):
            definitions[control][side] = [
                c for c in definitions[control][side]
                if c.get("context_feature_a") in spines[side]
                and (c.get("unary") or c.get("context_feature_b") in spines[side])
            ]
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in TRANSPORT_FOLDS:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        train = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)].copy()
        calibration = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)].copy()
        test = base[base.__ts__.between(c, e, inclusive="left")].copy()
        for control in controls:
            for side in ("long", "short"):
                print(f"[selection-control] start {control} {fold.name} {side}", flush=True)
                selected_control = definitions[control][side]
                if not selected_control:
                    continue
                tr = train[train.side_name.eq(side)].copy()
                ca = calibration[calibration.side_name.eq(side)].copy()
                te = test[test.side_name.eq(side)].copy()
                if tr.empty or ca.empty or te.empty:
                    continue
                # Control specialists are a bounded compute diagnostic.  The
                # production bank keeps its full predeclared training contract;
                # these controls use a deterministic 120k-row training cap
                # while retaining the complete calibration/test populations.
                if len(tr) > 120_000:
                    stable_seed = int.from_bytes(hashlib.sha256(f"{fold.name}|{side}".encode("utf-8")).digest()[:4], "little")
                    tr = _sample(tr, 120_000, seed=SEED + stable_seed % 1000)
                # The feature-pool manifest is already an availability-checked
                # discovery contract.  Re-querying the full store schema here
                # is both unnecessary and memory-expensive.
                control_features = list(predictive_by_side[side][:40])
                if len(control_features) < 30:
                    continue
                control_feature_sets = {str(cond["condition_id"]): control_features for cond in selected_control}
                tr_out, models = _prepare_condition_outputs(
                    tr, side, selected_control, spine_manifests[side], spines[side], control_feature_sets,
                    specialist_config=cfg,
                )
                print(f"[selection-control] fit {control} {fold.name} {side} rows={len(tr)}", flush=True)
                ca_out, _ = _prepare_condition_outputs(
                    ca, side, selected_control, spine_manifests[side], spines[side], control_feature_sets,
                    train_models=models, specialist_config=cfg,
                )
                te_out, _ = _prepare_condition_outputs(
                    te, side, selected_control, spine_manifests[side], spines[side], control_feature_sets,
                    train_models=models, specialist_config=cfg,
                )

                def _attach(frame: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
                    id_cols = ["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "residual_bps", "base_ev_bps", "label_available_ts"]
                    return frame[id_cols].merge(
                        outputs,
                        on=["candidate_id", "__ts__", "side_name", "residual_bps", "base_ev_bps"],
                        how="inner",
                        validate="one_to_one",
                    )

                ca_frame = _attach(ca, ca_out)
                te_frame = _attach(te, te_out)
                rank_cols = [
                    f"condition__{str(cond['condition_id']).replace('/', '_')}__gated_rank"
                    for cond in selected_control
                ]
                rank_cols = [col for col in rank_cols if col in ca_frame.columns and col in te_frame.columns]
                if not rank_cols:
                    continue
                ca_raw = ca_frame[rank_cols].mean(axis=1).to_numpy(np.float32)
                te_raw = te_frame[rank_cols].mean(axis=1).to_numpy(np.float32)
                combined = pd.concat([ca_frame, te_frame], ignore_index=True)
                mapped = _map_meta_residual(combined, np.concatenate([ca_raw, te_raw]), side)
                te_score = te_frame.base_ev_bps.to_numpy(np.float32) + mapped[len(ca_frame):]
                system = f"condition_control__{control}"
                scored = te_frame[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps"]].copy()
                scored["score"] = te_score
                scored["control"] = system
                scored["fold"] = fold.name
                prediction_rows.append(scored)
                scored["month"] = pd.to_datetime(scored["__ts__"], utc=True).dt.strftime("%Y-%m")
                for period, sub in [(fold.name, scored), *[(str(m), g) for m, g in scored.groupby("month", sort=True)]]:
                    metric_rows += _score_metrics(sub, "score", scope="global", period=period, system=system)
                for side_name, sub in scored.groupby("side_name", sort=True):
                    metric_rows += _score_metrics(sub, "score", scope=f"side:{side_name}", period=fold.name, system=system)
                del models, tr_out, ca_out, te_out, ca_frame, te_frame
                gc.collect()
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    predictions.to_parquet(out / "condition_selection_control_predictions.parquet", index=False)
    metrics.to_parquet(out / "condition_selection_control_metrics.parquet", index=False)
    _write_json(out / "condition_selection_control_manifest.json", {
        "schema": "pair_condition_selection_control_replay_v1",
        "controls": list(controls),
        "conditions_per_control_side": 1,
        "feature_count": 40,
        "training_row_cap": 120000,
        "target": "canonical ordinalized H12 net residual bps",
        "fit_boundary": "label_available_ts < decision_timestamp",
        "conversion": "prequential_same_side_monotone_pava_20_bins",
        "global_ranking": "mapped_common_bps_global_top_k",
        "transport_folds": [fold.name for fold in TRANSPORT_FOLDS],
    })
    return predictions, metrics


def _source(symbol: str) -> pd.DataFrame:
    path = PATH_ROOT / (symbol.lower().replace("_", "") + "_15m.parquet")
    raw = pd.read_parquet(path)
    col = next((c for c in ("ts", "timestamp", "__index_level_0__") if c in raw.columns), None)
    if col is not None:
        raw = raw.set_index(col)
    raw.index = pd.to_datetime(raw.index, utc=True)
    return raw.loc[:, ["open", "high", "low", "close"]][~raw.index.duplicated(keep="last")].sort_index()


def _fixed_exit_metrics(pred: pd.DataFrame, out: Path, score_cols: list[str]) -> pd.DataFrame:
    """Replay the incumbent 3 ATR / 0.5 ATR / 0.25 ATR policy."""

    union: set[str] = set()
    for col in score_cols:
        n = max(1, int(math.ceil(len(pred) * .10)))
        union.update(pred.nlargest(n, col).candidate_id.astype(str))
    chosen = pred[pred.candidate_id.astype(str).isin(union)].copy()
    rows: list[pd.DataFrame] = []
    for symbol, g in chosen.groupby(chosen.candidate_id.str.split("|").str[0], sort=False):
        try:
            bars = _source(symbol)
        except FileNotFoundError:
            continue
        path_file = PATH_ARTIFACT / f"symbol={symbol}.parquet"
        if not path_file.exists():
            continue
        meta = pd.read_parquet(path_file, columns=["candidate_id", "entry_price", "atr_bps"])
        g = g.merge(meta, on="candidate_id", how="inner", validate="one_to_one")
        if g.empty:
            continue
        starts = bars.index.get_indexer(pd.to_datetime(g.__ts__, utc=True)); valid = starts >= 0
        if not valid.any():
            continue
        g = g.loc[valid].copy(); starts = starts[valid]
        e = g.entry_price.to_numpy(float); atr_bps = g.atr_bps.to_numpy(float); atr = e * atr_bps / 10_000.0
        side = np.where(g.side_name.eq("long").to_numpy(), 1.0, -1.0)
        grid = simulate_h12_stop_trailing_grid(bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float), starts.astype(np.int64), e.astype(np.float32), atr.astype(np.float32), side.astype(np.float32), np.asarray([3.0], np.float32), np.asarray([.5], np.float32), np.asarray([.25], np.float32), horizon_bars=48)
        exit_net = net_bps(grid, atr_bps, cost_bps=100.0).reshape(-1)
        z = g[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", *score_cols]].copy()
        z["exit_net_bps"] = exit_net
        rows.append(z)
    if not rows:
        raise RuntimeError("no 15-minute paths matched pair-condition predictions")
    data = pd.concat(rows, ignore_index=True)
    result: list[dict[str, Any]] = []
    for col in score_cols:
        ordered = data.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
        rec: dict[str, Any] = {"system": col, "matched_rows": len(data)}
        for tail in TAILS:
            n = max(1, int(math.ceil(len(ordered) * tail)))
            top = ordered.head(n)
            rec[f"top{int(tail * 100)}_net_bps"] = float(top.exit_net_bps.mean())
            rec[f"top{int(tail * 100)}_gross_bps"] = float(top.exit_net_bps.mean() + 100.0)
        result.append(rec)
    result_df = pd.DataFrame(result).sort_values(["top5_net_bps", "top1_net_bps"], ascending=False)
    result_df.to_parquet(out / "fixed_exit_metrics.parquet", index=False)
    return result_df


def _fixed_exit_incumbent_comparison(
    pred: pd.DataFrame,
    out: Path,
    score_cols: list[str],
    *,
    incumbent_col: str = "incumbent_score",
) -> pd.DataFrame:
    """Compare the challenger and incumbent on one identical path union.

    ``_fixed_exit_metrics`` forms its path union from the score columns being
    replayed.  Adding the incumbent to that list would change the matched
    population and make the comparison optimistic or pessimistic depending on
    the score.  This helper freezes the union from the challenger arms first,
    then ranks every score (including the incumbent) on that same path set.
    """

    if incumbent_col not in pred or not score_cols:
        return pd.DataFrame()
    union: set[str] = set()
    n = max(1, int(math.ceil(len(pred) * .10)))
    for col in score_cols:
        union.update(pred.nlargest(n, col).candidate_id.astype(str))
    chosen = pred[pred.candidate_id.astype(str).isin(union)].copy()
    all_cols = [incumbent_col, *score_cols]
    rows: list[pd.DataFrame] = []
    for symbol, group in chosen.groupby(chosen.candidate_id.str.split("|").str[0], sort=False):
        try:
            bars = _source(symbol)
        except FileNotFoundError:
            continue
        path_file = PATH_ARTIFACT / f"symbol={symbol}.parquet"
        if not path_file.exists():
            continue
        meta = pd.read_parquet(path_file, columns=["candidate_id", "entry_price", "atr_bps"])
        group = group.merge(meta, on="candidate_id", how="inner", validate="one_to_one")
        if group.empty:
            continue
        starts = bars.index.get_indexer(pd.to_datetime(group.__ts__, utc=True))
        valid = starts >= 0
        if not valid.any():
            continue
        group = group.loc[valid].copy(); starts = starts[valid]
        entry = group.entry_price.to_numpy(float)
        atr_bps = group.atr_bps.to_numpy(float)
        atr = entry * atr_bps / 10_000.0
        side = np.where(group.side_name.eq("long").to_numpy(), 1.0, -1.0)
        grid = simulate_h12_stop_trailing_grid(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float),
            starts.astype(np.int64), entry.astype(np.float32), atr.astype(np.float32),
            side.astype(np.float32), np.asarray([3.0], np.float32),
            np.asarray([.5], np.float32), np.asarray([.25], np.float32), horizon_bars=48,
        )
        z = group[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", *all_cols]].copy()
        z["exit_net_bps"] = net_bps(grid, atr_bps, cost_bps=100.0).reshape(-1)
        z["month"] = pd.to_datetime(z["__ts__"], utc=True).dt.strftime("%Y-%m")
        z["week"] = pd.to_datetime(z["__ts__"], utc=True).dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
        rows.append(z)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    data.to_parquet(out / "fixed_exit_same_union_replay.parquet", index=False)
    result: list[dict[str, Any]] = []
    for col in all_cols:
        ordered = data.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
        rec: dict[str, Any] = {"system": col, "matched_rows": int(len(data)), "ranking_union": "challenger_top10_union"}
        for tail in TAILS:
            count = max(1, int(math.ceil(len(ordered) * tail)))
            top = ordered.head(count)
            key = int(tail * 100)
            rec[f"top{key}_net_bps"] = float(top.exit_net_bps.mean())
            rec[f"top{key}_gross_bps"] = float(top.exit_net_bps.mean() + 100.0)
        result.append(rec)
    result_df = pd.DataFrame(result).sort_values(["top5_net_bps", "top1_net_bps"], ascending=False)
    result_df.to_parquet(out / "incumbent_exit_comparison_same_union.parquet", index=False)

    # Persist both global-tail decomposition and monthly/weekly reranking so
    # the worst-period gate cannot be inferred from pooled rows alone.
    detail: list[dict[str, Any]] = []
    for col in all_cols:
        for month, month_frame in data.groupby("month", sort=True):
            ordered = month_frame.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
            for tail in TAILS:
                count = max(1, int(math.ceil(len(ordered) * tail)))
                top = ordered.head(count)
                detail.append({
                    "system": col, "period_type": "month", "period": str(month), "tail": float(tail),
                    "trades": int(len(top)), "gross_bps": float(top.exit_net_bps.mean() + 100.0),
                    "net_bps": float(top.exit_net_bps.mean()), "matched_rows": int(len(data)),
                })
        for week, week_frame in data.groupby("week", sort=True):
            ordered = week_frame.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
            for tail in TAILS:
                count = max(1, int(math.ceil(len(ordered) * tail)))
                top = ordered.head(count)
                detail.append({
                    "system": col, "period_type": "week", "period": str(week), "tail": float(tail),
                    "trades": int(len(top)), "gross_bps": float(top.exit_net_bps.mean() + 100.0),
                    "net_bps": float(top.exit_net_bps.mean()), "matched_rows": int(len(data)),
                })
    pd.DataFrame(detail).to_parquet(out / "fixed_exit_same_union_period_metrics.parquet", index=False)
    return result_df


def _fixed_exit_side_month_metrics(pred: pd.DataFrame, out: Path, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Break the fixed-policy global top-k replay down by side and month.

    Ranking remains global on the mapped common-bps score.  Side/month rows are
    a diagnostic decomposition of those globally selected trades, not separate
    side-local selections.
    """
    union: set[str] = set()
    for col in score_cols:
        n = max(1, int(math.ceil(len(pred) * .10)))
        union.update(pred.nlargest(n, col).candidate_id.astype(str))
    chosen = pred[pred.candidate_id.astype(str).isin(union)].copy()
    rows: list[pd.DataFrame] = []
    for symbol, g in chosen.groupby(chosen.candidate_id.str.split("|").str[0], sort=False):
        try:
            bars = _source(symbol)
        except FileNotFoundError:
            continue
        path_file = PATH_ARTIFACT / f"symbol={symbol}.parquet"
        if not path_file.exists():
            continue
        meta = pd.read_parquet(path_file, columns=["candidate_id", "entry_price", "atr_bps"])
        g = g.merge(meta, on="candidate_id", how="inner", validate="one_to_one")
        if g.empty:
            continue
        starts = bars.index.get_indexer(pd.to_datetime(g.__ts__, utc=True)); valid = starts >= 0
        if not valid.any():
            continue
        g = g.loc[valid].copy(); starts = starts[valid]
        e = g.entry_price.to_numpy(float); atr_bps = g.atr_bps.to_numpy(float); atr = e * atr_bps / 10_000.0
        side_sign = np.where(g.side_name.eq("long").to_numpy(), 1.0, -1.0)
        grid = simulate_h12_stop_trailing_grid(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float),
            starts.astype(np.int64), e.astype(np.float32), atr.astype(np.float32), side_sign.astype(np.float32),
            np.asarray([3.0], np.float32), np.asarray([.5], np.float32), np.asarray([.25], np.float32), horizon_bars=48,
        )
        z = g[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", *score_cols]].copy()
        z["exit_net_bps"] = net_bps(grid, atr_bps, cost_bps=100.0).reshape(-1)
        z["month"] = pd.to_datetime(z["__ts__"], utc=True).dt.strftime("%Y-%m")
        rows.append(z)
    if not rows:
        raise RuntimeError("no 15-minute paths matched pair-condition predictions")
    data = pd.concat(rows, ignore_index=True)
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for col in score_cols:
        system = col.removeprefix("score__")
        ordered = data.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            n = max(1, int(math.ceil(len(ordered) * tail)))
            top = ordered.head(n).copy()
            for (side, month), g in top.groupby(["side_name", "month"], sort=True):
                detail.append({
                    "ranking_scope": "global_matched_union", "system": system, "side": side, "month": month,
                    "tail": tail, "trades": int(len(g)), "matched_rows": int(len(data)),
                    "gross_bps": float(g.exit_net_bps.mean() + 100.0), "net_bps": float(g.exit_net_bps.mean()),
                })
            month_means = top.groupby("month", sort=True).exit_net_bps.mean()
            summary.append({
                "ranking_scope": "global_matched_union", "system": system, "side": "all", "tail": tail,
                "months": int(month_means.size), "mean_month_net_bps": float(month_means.mean()),
                "median_month_net_bps": float(month_means.median()), "min_month_net_bps": float(month_means.min()),
                "max_month_net_bps": float(month_means.max()), "std_month_net_bps": float(month_means.std(ddof=0)),
                "trades": int(len(top)), "matched_rows": int(len(data)),
            })
            for side, side_top in top.groupby("side_name", sort=True):
                month_means = side_top.groupby("month", sort=True).exit_net_bps.mean()
                summary.append({
                    "ranking_scope": "global_matched_union", "system": system, "side": side, "tail": tail,
                    "months": int(month_means.size), "mean_month_net_bps": float(month_means.mean()),
                    "median_month_net_bps": float(month_means.median()), "min_month_net_bps": float(month_means.min()),
                    "max_month_net_bps": float(month_means.max()), "std_month_net_bps": float(month_means.std(ddof=0)),
                    "trades": int(len(side_top)), "matched_rows": int(len(data)),
                })
    detail_df = pd.DataFrame(detail)
    summary_df = pd.DataFrame(summary)
    # Diagnostic only: rank within each side on the same matched replay
    # population.  Production selection remains the global ranking above.
    side_detail: list[dict[str, Any]] = []
    side_summary: list[dict[str, Any]] = []
    for col in score_cols:
        system = col.removeprefix("score__")
        for side, side_data in data.groupby("side_name", sort=True):
            side_ordered = side_data.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
            for tail in TAILS:
                n = max(1, int(math.ceil(len(side_ordered) * tail)))
                top = side_ordered.head(n).copy()
                for month, g in top.groupby("month", sort=True):
                    side_detail.append({
                        "ranking_scope": "side_local_diagnostic", "system": system, "side": side, "month": month,
                        "tail": tail, "trades": int(len(g)), "matched_rows": int(len(data)),
                        "gross_bps": float(g.exit_net_bps.mean() + 100.0), "net_bps": float(g.exit_net_bps.mean()),
                    })
                month_means = top.groupby("month", sort=True).exit_net_bps.mean()
                side_summary.append({
                    "ranking_scope": "side_local_diagnostic", "system": system, "side": side, "tail": tail,
                    "months": int(month_means.size), "mean_month_net_bps": float(month_means.mean()),
                    "median_month_net_bps": float(month_means.median()), "min_month_net_bps": float(month_means.min()),
                    "max_month_net_bps": float(month_means.max()), "std_month_net_bps": float(month_means.std(ddof=0)),
                    "trades": int(len(top)), "matched_rows": int(len(data)),
                })
    pd.DataFrame(side_detail).to_parquet(out / "fixed_exit_side_local_month_metrics.parquet", index=False)
    pd.DataFrame(side_summary).to_parquet(out / "fixed_exit_side_local_month_summary.parquet", index=False)
    detail_df.to_parquet(out / "fixed_exit_side_month_metrics.parquet", index=False)
    summary_df.to_parquet(out / "fixed_exit_side_month_summary.parquet", index=False)
    return detail_df, summary_df


def _fixed_exit_month_metrics(pred: pd.DataFrame, out: Path, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate global top-k separately inside each month and decompose sides."""
    union: set[str] = set()
    for col in score_cols:
        n = max(1, int(math.ceil(len(pred) * .10)))
        union.update(pred.nlargest(n, col).candidate_id.astype(str))
    chosen = pred[pred.candidate_id.astype(str).isin(union)].copy()
    rows: list[pd.DataFrame] = []
    for symbol, g in chosen.groupby(chosen.candidate_id.str.split("|").str[0], sort=False):
        try:
            bars = _source(symbol)
        except FileNotFoundError:
            continue
        path_file = PATH_ARTIFACT / f"symbol={symbol}.parquet"
        if not path_file.exists():
            continue
        meta = pd.read_parquet(path_file, columns=["candidate_id", "entry_price", "atr_bps"])
        g = g.merge(meta, on="candidate_id", how="inner", validate="one_to_one")
        if g.empty:
            continue
        starts = bars.index.get_indexer(pd.to_datetime(g.__ts__, utc=True)); valid = starts >= 0
        if not valid.any():
            continue
        g = g.loc[valid].copy(); starts = starts[valid]
        e = g.entry_price.to_numpy(float); atr_bps = g.atr_bps.to_numpy(float); atr = e * atr_bps / 10_000.0
        side_sign = np.where(g.side_name.eq("long").to_numpy(), 1.0, -1.0)
        grid = simulate_h12_stop_trailing_grid(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float),
            starts.astype(np.int64), e.astype(np.float32), atr.astype(np.float32), side_sign.astype(np.float32),
            np.asarray([3.0], np.float32), np.asarray([.5], np.float32), np.asarray([.25], np.float32), horizon_bars=48,
        )
        z = g[["candidate_id", "__ts__", "side_name", *score_cols]].copy()
        z["exit_net_bps"] = net_bps(grid, atr_bps, cost_bps=100.0).reshape(-1)
        z["month"] = pd.to_datetime(z["__ts__"], utc=True).dt.strftime("%Y-%m")
        rows.append(z)
    if not rows:
        raise RuntimeError("no 15-minute paths matched pair-condition predictions")
    data = pd.concat(rows, ignore_index=True)
    detail: list[dict[str, Any]] = []
    for col in score_cols:
        system = col.removeprefix("score__")
        for month, month_data in data.groupby("month", sort=True):
            ordered = month_data.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable")
            for tail in TAILS:
                n = max(1, int(math.ceil(len(ordered) * tail)))
                top = ordered.head(n)
                detail.append({
                    "ranking_scope": "monthly_global_rank", "system": system, "side": "all", "month": month,
                    "tail": tail, "trades": int(len(top)), "matched_rows": int(len(data)),
                    "gross_bps": float(top.exit_net_bps.mean() + 100.0), "net_bps": float(top.exit_net_bps.mean()),
                })
                for side, side_top in top.groupby("side_name", sort=True):
                    detail.append({
                        "ranking_scope": "monthly_global_rank", "system": system, "side": side, "month": month,
                        "tail": tail, "trades": int(len(side_top)), "matched_rows": int(len(data)),
                        "gross_bps": float(side_top.exit_net_bps.mean() + 100.0), "net_bps": float(side_top.exit_net_bps.mean()),
                    })
    detail_df = pd.DataFrame(detail)
    summary: list[dict[str, Any]] = []
    for (system, side, tail), g in detail_df.groupby(["system", "side", "tail"], sort=True):
        values = g.net_bps.to_numpy(float)
        summary.append({
            "ranking_scope": "monthly_global_rank", "system": system, "side": side, "tail": tail,
            "months": int(len(values)), "mean_month_net_bps": float(values.mean()),
            "median_month_net_bps": float(np.median(values)), "min_month_net_bps": float(values.min()),
            "max_month_net_bps": float(values.max()), "std_month_net_bps": float(values.std(ddof=0)),
            "trades": int(g.trades.sum()), "matched_rows": int(data.shape[0]),
        })
    summary_df = pd.DataFrame(summary)
    detail_df.to_parquet(out / "fixed_exit_month_metrics.parquet", index=False)
    summary_df.to_parquet(out / "fixed_exit_month_summary.parquet", index=False)
    return detail_df, summary_df


def _authoritative_metrics(out: Path) -> pd.DataFrame:
    """Compute pooled global metrics without the runner's per-fold duplication."""

    pred = pd.read_parquet(out / "predictions.parquet").copy()
    pred["month"] = pd.to_datetime(pred["__ts__"], utc=True).dt.strftime("%Y-%m")
    if (out / "predictions_with_incumbent.parquet").exists():
        baseline = pd.read_parquet(out / "predictions_with_incumbent.parquet", columns=["candidate_id", "incumbent_score"])
    else:
        baseline = pd.read_parquet(BASELINE_PRED, columns=["candidate_id", "score"]).rename(columns={"score": "incumbent_score"})
    pred = pred.merge(baseline, on="candidate_id", how="left", validate="one_to_one")
    systems = {"incumbent": "incumbent_score", "anchor_only": "score__anchor_only"}
    systems.update({c.removeprefix("score__"): c for c in pred.columns if c.startswith("score__") and c != "score__anchor_only"})
    rows: list[dict[str, Any]] = []
    for system, score_col in systems.items():
        scopes: list[tuple[str, str, pd.DataFrame]] = [("global", "all", pred)]
        scopes += [(f"side:{side}", "all", sub) for side, sub in pred.groupby("side_name", sort=True)]
        scopes += [("global", str(fold), sub) for fold, sub in pred.groupby("fold", sort=True)]
        scopes += [("global", str(month), sub) for month, sub in pred.groupby("month", sort=True)]
        for scope, period, sub in scopes:
            if sub.empty or score_col not in sub:
                continue
            rows += _score_metrics(sub, score_col, scope=scope, period=period, system=system)
    result = pd.DataFrame(rows)
    result.to_parquet(out / "global_metrics.parquet", index=False)
    return result


def _materialize_side_artifacts(
    out: Path,
    predictions: pd.DataFrame | None = None,
    authoritative: pd.DataFrame | None = None,
    lomo: pd.DataFrame | None = None,
) -> None:
    """Materialise the side-partitioned specialist artifacts.

    The research brief names side-specific OOF, innovation, incremental-value,
    LOMO and resource files.  The modelling pass keeps a combined table for
    efficient joins; this helper creates the explicit side contracts after
    the pass without retraining or copying the large store-backed inputs.
    """
    if predictions is None:
        predictions = pd.read_parquet(out / "predictions.parquet")
    if authoritative is None and (out / "global_metrics.parquet").exists():
        authoritative = pd.read_parquet(out / "global_metrics.parquet")
    if lomo is None and (out / "condition_lomo_results.parquet").exists():
        lomo = pd.read_parquet(out / "condition_lomo_results.parquet")

    oof_path = out / "condition_specialist_oof.parquet"
    oof = pd.read_parquet(oof_path) if oof_path.exists() else pd.DataFrame()
    side_counts = {}
    for side in ("long", "short"):
        side_oof = oof[oof.get("side_name", pd.Series(dtype=object)).eq(side)].copy() if not oof.empty else oof.copy()
        # ``concat`` of long/short fold outputs creates the union of columns;
        # retain only the condition namespace that belongs to this side.  This
        # keeps side-local specialist contracts from silently carrying the
        # other side's all-NaN columns into downstream feature selection.
        if not side_oof.empty:
            keep = [c for c in side_oof.columns if not c.startswith("condition__") or c.startswith(f"condition__{side}__")]
            side_oof = side_oof.loc[:, keep]
        side_oof.to_parquet(out / f"condition_specialist_oof_{side}.parquet", index=False)
        side_counts[side] = int(len(side_oof))

        id_cols = [c for c in ("candidate_id", "__ts__", "fold", "side_name") if c in side_oof.columns]
        innovation_cols = [
            c for c in side_oof.columns
            if c.startswith(f"condition__{side}__")
            and ("innovation" in c or "gated" in c)
        ]
        side_oof[id_cols + innovation_cols].to_parquet(
            out / f"condition_specialist_innovations_{side}.parquet", index=False
        )

        if authoritative is not None and not authoritative.empty:
            side_metrics = authoritative[
                authoritative.scope.eq(f"side:{side}")
            ].copy()
            if not side_metrics.empty:
                anchor = side_metrics[side_metrics.system.eq("anchor_only")][
                    ["period", "tail", "net_bps"]
                ].rename(columns={"net_bps": "anchor_net_bps"})
                side_metrics = side_metrics.merge(
                    anchor, on=["period", "tail"], how="left", validate="many_to_one"
                )
                side_metrics["delta_vs_anchor_net_bps"] = (
                    side_metrics.net_bps - side_metrics.anchor_net_bps
                )
        else:
            side_metrics = pd.DataFrame()
        # Keep the compact, side-local metric table under the name used by the
        # brief.  It is deliberately a metrics view, not a second fit.
        side_metrics.to_parquet(out / f"condition_incremental_value_{side}.parquet", index=False)

        if lomo is not None and not lomo.empty:
            side_lomo = lomo[lomo.side.eq(side)].copy()
        else:
            side_lomo = pd.DataFrame()
        side_lomo.to_parquet(out / f"condition_lomo_results_{side}.parquet", index=False)

        resource_path = out / f"condition_resource_usage_{side}.json"
        resource = {}
        if (out / "condition_resource_usage.json").exists():
            try:
                resource = json.loads((out / "condition_resource_usage.json").read_text())
            except Exception:
                resource = {}
        resource["side"] = side
        resource["specialist_oof_rows"] = side_counts[side]
        _write_json(resource_path, resource)
        pd.DataFrame([resource]).to_parquet(
            out / f"condition_resource_usage_{side}.parquet", index=False
        )


def _materialize_score_calibration(out: Path, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Persist a read-only audit of side-local conversion and global comparability.

    The conversion map is fit prequentially during the OOS pass.  This audit
    does not refit it: it bins the already mapped common-bps scores by side,
    transport fold, month and query to show whether the same score scale is
    comparable across 4-hour query groups before the final pooled ranking.
    """
    pred = predictions if predictions is not None else pd.read_parquet(out / "predictions.parquet")
    work = pred.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    work["query_4h"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("4h")
    rows: list[dict[str, Any]] = []
    score_cols = [c for c in work.columns if c.startswith("score__")]
    for score_col in score_cols:
        system = score_col.removeprefix("score__")
        for side, side_frame in work.groupby("side_name", sort=True):
            x = pd.to_numeric(side_frame[score_col], errors="coerce")
            if x.notna().sum() < 4:
                continue
            # Fixed rank bins make the audit robust to ties and preserve the
            # side-local map's ordering without learning any test labels.
            rank = x.rank(method="first", pct=True)
            bins = np.minimum(19, np.floor(rank.to_numpy(float) * 20.0).astype(int))
            scoped = side_frame.assign(_score=x.to_numpy(float), _score_bin=bins)
            for (fold, month, score_bin), g in scoped.groupby(["fold", "month", "_score_bin"], sort=True):
                rows.append({
                    "system": system, "side": side, "fold": str(fold), "month": str(month),
                    "score_bin": int(score_bin), "rows": int(len(g)),
                    "query_count": int(g.query_4h.nunique()),
                    "mean_mapped_score_bps": float(g._score.mean()),
                    "mean_realised_net_bps": float(g.net_bps.mean()),
                    "mean_conversion_error_bps": float((g.net_bps - g._score).mean()),
                })
    result = pd.DataFrame(rows)
    result.to_parquet(out / "score_calibration.parquet", index=False)
    return result


def _materialize_complementarity(
    out: Path,
    side: str,
    candidates: pd.DataFrame,
    signatures: pd.DataFrame,
    activations: dict[str, dict[str, np.ndarray]],
    model_vectors: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Persist the model/feature/membership complementarity matrix."""
    required = {
        "condition_a", "condition_b", "model_response_distance",
        "feature_response_distance", "membership_overlap",
        "combined_complementarity", "side",
    }
    if candidates.empty or signatures.empty:
        empty = pd.DataFrame(columns=sorted(required))
        empty.to_parquet(out / f"condition_complementarity_{side}.parquet", index=False)
        return empty
    ids = [str(x) for x in signatures.condition_id.dropna().unique()]
    candidate_by_id = candidates.set_index("condition_id", drop=False)
    sig_fields = [c for c in signatures.columns if c.startswith("feature_response_svd_")]
    sig_by = signatures.set_index("condition_id")
    if model_vectors is None:
        model_fields = [
            c for c in signatures.columns
            if c.startswith("model_delta_")
        ]
        model_vectors = {
            cid: sig_by.loc[cid, model_fields].to_numpy(float)
            for cid in ids
            if cid in sig_by.index
        }
    rows: list[dict[str, Any]] = []
    for left_id, right_id in combinations(ids, 2):
        if left_id not in candidate_by_id.index or right_id not in candidate_by_id.index:
            continue
        left = candidate_by_id.loc[left_id].to_dict()
        right = candidate_by_id.loc[right_id].to_dict()
        if left["context_feature_a"] not in activations or right["context_feature_a"] not in activations:
            continue
        wa = _condition_weight(pd.DataFrame(), left, activations)
        wb = _condition_weight(pd.DataFrame(), right, activations)
        va = np.asarray(model_vectors.get(left_id, np.zeros(3)), dtype=float)
        vb = np.asarray(model_vectors.get(right_id, np.zeros(3)), dtype=float)
        model_distance = cosine_distance(va - np.nanmean(va), vb - np.nanmean(vb))
        if sig_fields:
            fa = sig_by.loc[left_id, sig_fields].to_numpy(float) if left_id in sig_by.index else np.zeros(len(sig_fields))
            fb = sig_by.loc[right_id, sig_fields].to_numpy(float) if right_id in sig_by.index else np.zeros(len(sig_fields))
            feature_distance = cosine_distance(fa, fb)
        else:
            feature_distance = 0.0
        overlap = weighted_jaccard(wa, wb)
        combined = 0.45 * model_distance + 0.35 * feature_distance + 0.20 * (1.0 - overlap)
        rows.append({
            "condition_a": left_id, "condition_b": right_id, "side": side,
            "model_response_distance": float(model_distance),
            "feature_response_distance": float(feature_distance),
            "membership_overlap": float(overlap),
            "combined_complementarity": float(combined),
        })
    result = pd.DataFrame(rows, columns=sorted(required))
    result.to_parquet(out / f"condition_complementarity_{side}.parquet", index=False)
    return result


def _materialize_complementarity_from_artifacts(out: Path, side: str) -> pd.DataFrame:
    """Rebuild complementarity from frozen discovery artifacts during finalize."""
    candidate_path = out / f"condition_candidates_{side}.parquet"
    signature_path = out / f"condition_response_signatures_{side}.parquet"
    spine_path = out / f"condition_spine_values_{side}.parquet"
    if not (candidate_path.exists() and signature_path.exists() and spine_path.exists()):
        return _materialize_complementarity(out, side, pd.DataFrame(), pd.DataFrame(), {})
    candidates = pd.read_parquet(candidate_path)
    signatures = pd.read_parquet(signature_path)
    spine_values = pd.read_parquet(spine_path)
    fields = [c for c in spine_values.columns if c.startswith("__activation__")]
    activations: dict[str, dict[str, np.ndarray]] = {}
    for column in fields:
        field, region = column.removeprefix("__activation__").rsplit("__", 1)
        activations.setdefault(field, {})[region] = spine_values[column].to_numpy(np.float32)
    return _materialize_complementarity(out, side, candidates[candidates.condition_id.isin(signatures.condition_id)], signatures, activations)


def _materialize_model_bank_manifest(out: Path, predictions: pd.DataFrame) -> dict[str, Any]:
    """Record the frozen OOF model bank used by condition screening/meta arms."""
    entries = [
        {
            "model_id": "structural_anchor",
            "score_column": "base_ev_bps",
            "source": "prequential_same_side_base_r3_value_map",
            "strict_oof": True,
        },
        {
            "model_id": "incumbent_frozen_stack",
            "score_column": "incumbent_score",
            "source": str(BASELINE_PRED),
            "strict_oof": True,
        },
    ]
    entries.extend({
        "model_id": c.removeprefix("score__"),
        "score_column": c,
        "source": "pair_condition_meta_ablation",
        "strict_oof": True,
    } for c in predictions.columns if c.startswith("score__"))
    weighting_exponent = 1.5
    existing_manifest = out / "run_manifest.json"
    if existing_manifest.exists():
        try:
            weighting_exponent = float(json.loads(existing_manifest.read_text()).get("condition_weight_exponent", weighting_exponent))
        except Exception:
            pass
    manifest = {
        "schema": "pair_condition_model_bank_manifest_v1",
        "target_contract": "H12 net bps with the canonical single 100-bps cost",
        "query_contract": "4h x side",
        "ranking_contract": "common mapped bps then global top-k",
        "transport_folds": [f.name for f in TRANSPORT_FOLDS],
        "models": entries,
        "row_count": int(len(predictions)),
        "candidate_id_unique": bool(predictions.candidate_id.is_unique),
        "condition_weight_exponent": weighting_exponent,
        "equal_condition_month_weighting": True,
        "lomo_artifact": "condition_lomo_results.parquet",
    }
    _write_json(out / "model_bank_manifest.json", manifest)
    return manifest


def _materialize_specialist_standalone_metrics(
    out: Path,
    selected: dict[str, list[dict[str, Any]]],
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate each retained specialist output independently of the meta arm."""
    oof_path = out / "condition_specialist_oof.parquet"
    if not oof_path.exists():
        return pd.DataFrame()
    oof = pd.read_parquet(oof_path)
    pred = predictions if predictions is not None else pd.read_parquet(out / "predictions.parquet")
    # OOF specialist rows carry the frozen candidate/fold identity; timestamp
    # is taken from the prediction table because the compact OOF schema does
    # not duplicate it.
    join_cols = [c for c in ("candidate_id", "fold", "side_name") if c in oof.columns and c in pred.columns]
    work = oof.merge(pred[join_cols + ["__ts__", "net_bps", "gross_bps"]], on=join_cols, how="inner", validate="one_to_one")
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    suffixes = ("rank", "gated_rank", "hard_rank", "innovation_rank", "hard_innovation_rank", "raw", "gated", "hard_gated", "innovation", "gated_innovation", "hard_gated_innovation")
    for side, conditions in selected.items():
        side_work = work[work.side_name.eq(side)]
        for condition in conditions:
            name = str(condition["condition_id"]).replace("/", "_")
            for suffix in suffixes:
                score_column = f"condition__{name}__{suffix}"
                if score_column not in side_work:
                    continue
                base = side_work[["candidate_id", "net_bps", "gross_bps", "month", score_column]].rename(columns={score_column: "_score"})
                for period, frame in [("all", base), *[(str(month), group) for month, group in base.groupby("month", sort=True)]]:
                    frame = frame[np.isfinite(pd.to_numeric(frame._score, errors="coerce"))]
                    if frame.empty:
                        continue
                    for tail in TAILS:
                        n = max(1, int(math.ceil(len(frame) * tail)))
                        top = frame.sort_values(["_score", "candidate_id"], ascending=[False, True], kind="stable").head(n)
                        rows.append({
                            "condition_id": condition["condition_id"], "side": side, "output": suffix,
                            "period": period, "tail": tail, "rows": int(len(frame)), "trades": int(n),
                            "gross_bps": float(top.gross_bps.mean()), "net_bps": float(top.net_bps.mean()),
                            "rank_ic_net": float(frame._score.corr(frame.net_bps, method="spearman")),
                        })
    result = pd.DataFrame(rows)
    result.to_parquet(out / "condition_specialist_standalone_metrics.parquet", index=False)
    return result


def _materialize_control_metrics(out: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    """Fixed-weight model/gating controls on the same OOF population."""
    frame = predictions.copy()
    if "incumbent_score" not in frame.columns and BASELINE_PRED.exists():
        baseline = pd.read_parquet(BASELINE_PRED, columns=["candidate_id", "score"]).rename(columns={"score": "incumbent_score"})
        frame = frame.merge(baseline, on="candidate_id", how="left", validate="one_to_one")
    controls: dict[str, pd.Series] = {
        "anchor_only": frame["score__anchor_only"],
        "equal_pair_average": frame[["score__memberships", "score__raw_ranks", "score__gated_ranks"]].mean(axis=1),
        "anchor_full_context_half_blend": 0.5 * frame["score__anchor_only"] + 0.5 * frame["score__full_context"],
    }
    if "incumbent_score" in frame.columns:
        controls["frozen_multiview_stack"] = frame["incumbent_score"]
    if "score__ridge_blend" in frame.columns:
        controls["regularized_linear_blend"] = frame["score__ridge_blend"]
    if "score__gmm_geometry" in frame.columns:
        controls["geometry_only_gmm"] = frame["score__gmm_geometry"]
    if "score__full_context_gmm" in frame.columns:
        controls["full_context_gmm"] = frame["score__full_context_gmm"]
    if "score__hard_gating" in frame.columns:
        controls["hard_gating"] = frame["score__hard_gating"]
    if "score__probability_only" in frame.columns:
        controls["probability_only"] = frame["score__probability_only"]
    rows: list[dict[str, Any]] = []
    frame["month"] = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    for system, score in controls.items():
        x = frame.assign(_control_score=score)
        for scope, period, sub in [("global", "all", x), *[("global", str(month), g) for month, g in x.groupby("month", sort=True)]]:
            for rec in _score_metrics(sub, "_control_score", scope=scope, period=period, system=system):
                rows.append(rec)
    result = pd.DataFrame(rows)
    result.to_parquet(out / "control_metrics.parquet", index=False)
    return result


def _materialize_discovery_control_audit(
    out: Path,
    selected: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Record the predeclared condition-selection controls.

    The full selection surface is the only arm refit through OOS in this
    runner.  The other controls are intentionally discovery-only screens: the
    artifact makes that fact explicit instead of presenting a proxy as a
    causal OOS result.  This is useful for deciding which controls merit an
    equal-budget refit in the next funnel round.
    """

    rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        candidate_path = out / f"condition_candidates_{side}.parquet"
        if not candidate_path.exists():
            continue
        _materialize_candidate_support_gate_audit(out, side)
        candidates = pd.read_parquet(candidate_path)
        if candidates.empty:
            continue
        selected_ids = {str(c["condition_id"]) for c in selected.get(side, [])}

        def add(control: str, frame: pd.DataFrame, *, model_refit: bool = False, note: str = "") -> None:
            for rank, rec in enumerate(frame.head(3).to_dict("records"), 1):
                rows.append({
                    "control": control,
                    "side": side,
                    "condition_id": str(rec.get("condition_id", "")),
                    "selection_rank": rank,
                    "candidate_screen_score": float(rec.get("candidate_screen_score", np.nan)),
                    "effective_rows": float(rec.get("effective_rows", np.nan)),
                    "supported_month_count": int(rec.get("supported_month_count", 0)),
                    "pair_interaction": float(rec.get("pair_interaction", np.nan)),
                    "event_lift": float(rec.get("event_lift", np.nan)),
                    "rank_ic": float(rec.get("rank_ic", np.nan)),
                    "model_refit_through_oos": bool(model_refit),
                    "status": "authoritative_oos" if model_refit else "discovery_only",
                    "note": note,
                })

        selected_frame = candidates[candidates.condition_id.astype(str).isin(selected_ids)]
        add("full_selection", selected_frame.sort_values(["candidate_screen_score", "effective_rows"], ascending=[False, False]), model_refit=True, note="frozen relevance + complementarity greedy selection")

        pool = candidates[~candidates.condition_id.astype(str).isin(selected_ids)].copy()
        rng = np.random.default_rng(SEED + (0 if side == "long" else 1))
        random_frame = pool.iloc[rng.permutation(len(pool))] if not pool.empty else pool
        add("random_supported", random_frame, note="seeded supported-condition sample; no OOS refit")
        add("geometry_only", candidates.sort_values(["effective_rows", "supported_month_count"], ascending=[False, False]), note="support/recurrence only")
        add("no_model_utility", candidates.sort_values(["pair_interaction", "event_lift", "supported_month_count"], ascending=[False, False, False]), note="activation non-additivity/recurrence only")

        model_path = out / f"condition_model_utility_portability_{side}.parquet"
        if model_path.exists():
            model_port = pd.read_parquet(model_path)
            no_feature = candidates.merge(model_port[["condition_id", "portable_delta_top10_net_bps", "portable_delta_rank_ic"]], on="condition_id", how="left")
            no_feature["model_portability_score"] = no_feature.portable_delta_top10_net_bps.fillna(0.0) + 100.0 * no_feature.portable_delta_rank_ic.fillna(0.0)
            add("no_feature_behavior", no_feature.sort_values(["model_portability_score", "effective_rows"], ascending=[False, False]), note="model-response portability only")

        # A unary proxy is retained for control traceability.  It is not
        # promoted as a single-feature specialist because the current OOS
        # pass intentionally freezes pair conditions.
        unary = (
            candidates.assign(unary_key=candidates.context_feature_a.astype(str) + "__" + candidates.activation_a.astype(str))
            .groupby("unary_key", as_index=False)
            .agg(
                condition_id=("unary_key", "first"),
                candidate_screen_score=("candidate_screen_score", "max"),
                effective_rows=("effective_rows", "max"),
                supported_month_count=("supported_month_count", "max"),
                pair_interaction=("pair_interaction", "max"),
                event_lift=("event_lift", "max"),
                rank_ic=("rank_ic", "max"),
            )
        )
        unary["condition_id"] = "univariate__" + unary.unary_key.astype(str)
        add("univariate", unary.sort_values(["candidate_screen_score", "effective_rows"], ascending=[False, False]), note="univariate discovery proxy; not refit through OOS")

    result = pd.DataFrame(rows)
    # A bounded strict-OOS replay can upgrade only the first retained control
    # condition per side.  The remaining candidates stay discovery-only.
    control_oos_path = out / "condition_selection_control_metrics.parquet"
    if not result.empty and control_oos_path.exists():
        oos_control_metrics = pd.read_parquet(control_oos_path)
        oos_controls = {
            str(x).removeprefix("condition_control__")
            for x in oos_control_metrics["system"].dropna().unique()
        }
        mask = result.control.astype(str).isin(oos_controls) & result.selection_rank.eq(1)
        result.loc[mask, "model_refit_through_oos"] = True
        result.loc[mask, "status"] = "bounded_authoritative_oos"
        result.loc[mask, "note"] = result.loc[mask, "note"].astype(str) + "; one-condition-per-side strict-OOS replay"
    result.to_parquet(out / "control_selection_audit.parquet", index=False)
    return result


def _materialize_candidate_support_gate_audit(out: Path, side: str) -> pd.DataFrame:
    """Materialize the month/query support gate for cached candidates.

    Older discovery checkpoints predate the non-adjacent-month gate.  This
    audit recomputes the exact gate from the frozen activation spine without
    changing condition selection or retraining any model, then adds the
    resulting columns to the cached candidate parquet.  The computation is
    batched over conditions so the 15k-row discovery spine never becomes a
    candidate-by-row dense allocation.
    """
    candidate_path = out / f"condition_candidates_{side}.parquet"
    spine_path = out / f"condition_spine_values_{side}.parquet"
    if not candidate_path.exists() or not spine_path.exists():
        return pd.DataFrame()
    candidates = pd.read_parquet(candidate_path)
    spine = pd.read_parquet(spine_path)
    if candidates.empty or spine.empty:
        return pd.DataFrame()
    spine = spine.copy()
    spine["__query_4h__"] = pd.to_datetime(spine["__ts__"], utc=True).dt.floor("4h")
    spine["__month__"] = pd.to_datetime(spine["__ts__"], utc=True).dt.strftime("%Y-%m")
    query_codes = pd.factorize(spine["__query_4h__"], sort=True)[0]
    month_codes, month_names = pd.factorize(spine["__month__"], sort=True)
    n_rows = len(spine)
    n_queries = int(query_codes.max()) + 1 if len(query_codes) else 0
    n_months = len(month_names)

    # Group boundaries for exact query-presence counts.  Sorting once lets
    # np.maximum.reduceat compute all query memberships for a batch.
    query_order = np.argsort(query_codes, kind="stable")
    query_sorted = query_codes[query_order]
    query_starts = np.r_[0, np.flatnonzero(np.diff(query_sorted)) + 1]
    month_query_code = month_codes.astype(np.int64) * max(n_queries, 1) + query_codes.astype(np.int64)
    month_query_order = np.argsort(month_query_code, kind="stable")
    month_query_sorted = month_query_code[month_query_order]
    month_query_starts = np.r_[0, np.flatnonzero(np.diff(month_query_sorted)) + 1]
    month_query_keys = month_query_sorted[month_query_starts]
    month_query_month = month_query_keys // max(n_queries, 1)

    keys = [
        f"{row.context_feature_a}|{row.activation_a}|{row.context_feature_b}|{row.activation_b}"
        for row in candidates[["context_feature_a", "activation_a", "context_feature_b", "activation_b"]].itertuples(index=False)
    ]
    left_keys = sorted({str(row.context_feature_a) + "|" + str(row.activation_a) for row in candidates.itertuples(index=False)})
    right_keys = sorted({str(row.context_feature_b) + "|" + str(row.activation_b) for row in candidates.itertuples(index=False)})
    activation_cache: dict[str, np.ndarray] = {}
    for key in set(left_keys + right_keys):
        field, region = key.rsplit("|", 1)
        column = f"__activation__{field}__{region}"
        if column in spine:
            activation_cache[key] = pd.to_numeric(spine[column], errors="coerce").fillna(0.0).to_numpy(np.float32)

    records: list[dict[str, Any]] = []
    batch_size = 256
    for start in range(0, len(candidates), batch_size):
        block = candidates.iloc[start : start + batch_size]
        left = [activation_cache.get(str(r.context_feature_a) + "|" + str(r.activation_a)) for r in block.itertuples(index=False)]
        right = [activation_cache.get(str(r.context_feature_b) + "|" + str(r.activation_b)) for r in block.itertuples(index=False)]
        valid = [a is not None and b is not None and len(a) == n_rows and len(b) == n_rows for a, b in zip(left, right)]
        weights = np.zeros((len(block), n_rows), dtype=np.float32)
        for i, (a, b, ok) in enumerate(zip(left, right, valid)):
            if ok:
                weights[i] = a * b
        hard = weights >= 0.5
        month_indicator = np.eye(n_months, dtype=np.float32)[month_codes] if n_months else np.zeros((n_rows, 0), dtype=np.float32)
        mass = weights @ month_indicator if n_months else np.zeros((len(block), 0), dtype=np.float32)
        effective = np.divide(np.square(weights.sum(axis=1)), np.square(weights).sum(axis=1), out=np.zeros(len(block), dtype=np.float32), where=np.square(weights).sum(axis=1) > 0)
        hard_sorted = hard[:, query_order]
        query_present = np.maximum.reduceat(hard_sorted, query_starts, axis=1) if n_queries else np.zeros((len(block), 0), dtype=bool)
        effective_queries = query_present.sum(axis=1)
        month_hard_sorted = hard[:, month_query_order]
        month_query_present = np.maximum.reduceat(month_hard_sorted, month_query_starts, axis=1) if len(month_query_keys) else np.zeros((len(block), 0), dtype=bool)
        month_effective_queries = np.zeros((len(block), n_months), dtype=np.int16)
        for month_code in range(n_months):
            columns = np.flatnonzero(month_query_month == month_code)
            if len(columns):
                month_effective_queries[:, month_code] = month_query_present[:, columns].sum(axis=1)
        supported_month_count = (mass >= 50.0).sum(axis=1)
        valid_months = month_effective_queries >= 50
        valid_month_count = valid_months.sum(axis=1)
        supported_nonadjacent = np.where(valid_month_count >= 2, valid_month_count, 0)
        for i, row in enumerate(block.itertuples(index=False)):
            records.append({
                "condition_id": str(row.condition_id),
                "supported_month_count_recomputed": int(supported_month_count[i]),
                "supported_nonadjacent_month_count": int(supported_nonadjacent[i]),
                "global_effective_queries_recomputed": int(effective_queries[i]),
                "effective_rows_recomputed": float(effective[i]),
                "support_gate_min_effective_rows": float(effective[i]) >= 1000.0,
                "support_gate_min_effective_queries": int(effective_queries[i]) >= 250,
                "support_gate_min_supported_months": int(supported_month_count[i]) >= 3,
                "support_gate_min_nonadjacent_months": int(supported_nonadjacent[i]) >= 3,
            })
    audit = pd.DataFrame(records)
    audit.to_parquet(out / f"candidate_support_gate_audit_{side}.parquet", index=False)
    merged = candidates.drop(columns=[c for c in audit.columns if c != "condition_id" and c in candidates.columns], errors="ignore").merge(audit, on="condition_id", how="left", validate="one_to_one")
    merged.to_parquet(candidate_path, index=False)
    return audit


def _lomo_diagnostics(out: Path, selected: dict[str, list[dict[str, Any]]], monthly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side, conditions in selected.items():
        frame = monthly.get(side, pd.DataFrame())
        for cond in conditions:
            cid = cond["condition_id"]
            g = frame[frame.condition_id.eq(cid)] if not frame.empty else pd.DataFrame()
            for month in sorted(g.month.unique()) if not g.empty else []:
                held = g[g.month.eq(month)].iloc[0]
                train_vals = g.loc[~g.month.eq(month), "delta_top10_net_bps"].to_numpy(float)
                rows.append({"side": side, "condition_id": cid, "held_out_month": month, "support_effective_rows": float(held.effective_rows), "held_out_delta_top10_net_bps": float(held.delta_top10_net_bps), "train_portability_without_month": portability_score(train_vals), "condition_meaning_stable": bool(np.isfinite(train_vals).sum() >= 2 and np.sign(np.nanmedian(train_vals)) == np.sign(held.delta_top10_net_bps) if np.isfinite(train_vals).any() else False)})
    result = pd.DataFrame(rows)
    result.to_parquet(out / "condition_lomo_results.parquet", index=False)
    return result


def _fit_train_only_activation_manifest(
    frame: pd.DataFrame,
    spine: list[str],
    cfg: ConditionalSpecialistConfig,
) -> dict[str, Any]:
    """Fit soft-region thresholds on a LOMO training slice only.

    The discovery manifest is intentionally not reused here.  Re-fitting the
    quantiles for every held-out month makes the portability audit sensitive
    to the actual train/test boundary rather than only replaying a discovery
    diagnostic.  The pair identities and frozen feature sets remain fixed,
    so this is a portability test rather than a second selection pass.
    """

    result: dict[str, Any] = {}
    for field in spine:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        _, _, spec = soft_regions(
            values,
            width_quantile=cfg.soft_transition_width_quantile,
        )
        result[field] = spec
    return result


def _true_lomo_diagnostics(
    base: pd.DataFrame,
    selected: dict[str, list[dict[str, Any]]],
    spines: dict[str, list[str]],
    feature_sets: dict[str, dict[str, list[str]]],
    cfg: ConditionalSpecialistConfig,
    out: Path,
) -> pd.DataFrame:
    """Train each frozen specialist with one discovery month held out.

    ``_lomo_diagnostics`` above is a fast response-sign proxy.  This function
    is the authoritative portability audit: for every retained condition and
    supported discovery month it refits the soft-state thresholds, the
    condition-weighted LambdaRank specialist, and the train-only innovation
    residualizer, then scores the held-out month.  No held-out labels enter
    thresholds, model fitting, weighting, residualization, or score mapping.

    It deliberately uses the small frozen discovery sample materialized by
    the spine stage.  The OOS replay remains untouched; this audit is a
    diagnostic on pre-transport data and therefore cannot alter selection.
    """

    rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        conditions = selected.get(side, [])
        if not conditions:
            continue
        spine = list(spines[side])
        # Reuse the exact frozen discovery candidate population when it is
        # available.  Falling back to the deterministic discovery sample is
        # only for older artifacts that predate spine-values materialization.
        spine_values_path = out / f"condition_spine_values_{side}.parquet"
        if spine_values_path.exists():
            discovery_ids = pd.read_parquet(spine_values_path, columns=["candidate_id"])["candidate_id"].astype(str)
            dev = base[
                base.side_name.eq(side)
                & base.__ts__.lt(DISCOVERY_END)
                & base.candidate_id.astype(str).isin(set(discovery_ids))
            ].copy()
        else:
            dev = base[base.side_name.eq(side) & base.__ts__.lt(DISCOVERY_END)].copy()
            dev = _sample(dev, min(DISCOVERY_SAMPLE_ROWS, len(dev)), seed=SEED + (0 if side == "long" else 1))
        if dev.empty:
            continue

        fields = list(dict.fromkeys(spine + [f for cid in feature_sets[side].values() for f in cid]))
        joined = _store_rows(dev, fields)
        work = dev.merge(joined, on="candidate_id", validate="one_to_one")
        work["month"] = pd.to_datetime(work.__ts__, utc=True).dt.strftime("%Y-%m")

        for held_month in sorted(work.month.dropna().unique()):
            train = work[work.month.ne(held_month)].copy()
            held = work[work.month.eq(held_month)].copy()
            if len(train) < 100 or len(held) < 20:
                continue
            train_manifest = _fit_train_only_activation_manifest(train, spine, cfg)
            train_memberships = _apply_manifest_memberships(train, train_manifest, spine)
            held_memberships = _apply_manifest_memberships(held, train_manifest, spine)
            train_outputs = train[["base_score", "query_4h"]].copy()
            held_outputs = held[["base_score", "query_4h"]].copy()
            prior_raw_fields: list[str] = []

            for condition in conditions:
                cid = str(condition["condition_id"])
                name = cid.replace("/", "_")
                train_w = _condition_weight(train, condition, train_memberships)
                held_w = _condition_weight(held, condition, held_memberships)
                fit_mask = np.isfinite(train_w) & (train_w > 0.01)
                fit = train.loc[fit_mask].copy()
                if len(fit) < max(100, SPECIALIST_PARAMS["min_child_samples"] // 2):
                    continue
                if len(fit) > MAX_TRAIN_ROWS:
                    offset = int.from_bytes(hashlib.sha256(cid.encode("utf-8")).digest()[:4], "little") % 1000
                    fit = _sample(fit, MAX_TRAIN_ROWS, seed=SEED + offset + len(str(held_month)))
                fit_w = _condition_weight(
                    fit,
                    condition,
                    _apply_manifest_memberships(fit, train_manifest, spine),
                )
                rank_weights = _condition_month_balanced_weights(
                    fit,
                    fit_w,
                    exponent=cfg.condition_weight_exponent,
                    equal_months=cfg.equal_condition_month_weighting,
                )
                model, used, med, _ = _fit_ranker(
                    fit,
                    feature_sets[side][cid],
                    fit.residual_grade.to_numpy(np.int32),
                    fit.query_4h,
                    SPECIALIST_PARAMS,
                    rank_weights,
                )
                raw_fit = _predict(model, fit, used, med)
                raw_held = _predict(model, held, used, med)

                residualizer_fields = ["base_score", *prior_raw_fields]
                if prior_raw_fields:
                    fit_design = np.column_stack([
                        np.ones(len(fit), dtype=np.float32),
                        fit.base_score.to_numpy(np.float32),
                        train_outputs.loc[fit.index, prior_raw_fields].to_numpy(np.float32),
                    ])
                else:
                    fit_design = np.column_stack([
                        np.ones(len(fit), dtype=np.float32),
                        fit.base_score.to_numpy(np.float32),
                    ])
                ok = np.isfinite(raw_fit) & np.isfinite(fit_design).all(axis=1)
                beta = (
                    np.linalg.lstsq(fit_design[ok], raw_fit[ok], rcond=None)[0]
                    if ok.sum() >= max(20, fit_design.shape[1] + 2)
                    else np.zeros(fit_design.shape[1], dtype=np.float32)
                )
                held_design_columns = [
                    np.ones(len(held), dtype=np.float32),
                    held.base_score.to_numpy(np.float32),
                ]
                for previous_field in prior_raw_fields:
                    held_design_columns.append(held_outputs[previous_field].to_numpy(np.float32))
                held_design = np.column_stack(held_design_columns)
                innovation = raw_held - np.asarray(held_design @ beta, dtype=np.float32)
                gated = raw_held * np.power(np.clip(held_w, 0.0, 1.0), cfg.condition_weight_exponent)
                gated_rank = _within_query_rank(gated, held.query_4h)
                raw_field = f"condition__{name}__raw"
                # Keep the train output aligned to the full train frame.  The
                # model was fit only on the condition-supported subset, so a
                # positional assignment would silently misalign prior
                # residualizers on the next specialist.
                train_outputs[raw_field] = np.nan
                train_outputs.loc[fit.index, raw_field] = raw_fit.astype(np.float32)
                held_outputs[raw_field] = raw_held.astype(np.float32)
                prior_raw_fields.append(raw_field)

                score = gated_rank
                anchor = held.base_ev_bps.to_numpy(float)
                net = held.net_bps.to_numpy(float)
                finite = np.isfinite(score) & np.isfinite(net)
                if finite.sum() < 20:
                    continue
                score = score[finite]
                net = net[finite]
                anchor = anchor[finite]
                order = np.argsort(score, kind="stable")[::-1]
                rec: dict[str, Any] = {
                    "side": side,
                    "condition_id": cid,
                    "held_out_month": str(held_month),
                    "method": "train_only_thresholds_model_residualizer",
                    "train_rows": int(len(train)),
                    "fit_rows": int(len(fit)),
                    "held_out_rows": int(len(held)),
                    "held_effective_rows": float(effective_rows(held_w)),
                    "held_membership_mean": float(np.nanmean(held_w)),
                    "held_ood_rate": float(np.mean(held_w < 0.10)),
                    "held_rank_ic_net": float(pd.Series(score).corr(pd.Series(net), method="spearman")),
                    "train_weight_exponent": float(cfg.condition_weight_exponent),
                }
                for tail in TAILS:
                    n = max(1, int(math.ceil(len(order) * tail)))
                    take = order[:n]
                    rec[f"held_top{int(tail * 100)}_net_bps"] = float(net[take].mean())
                    rec[f"held_top{int(tail * 100)}_anchor_net_bps"] = float(net[np.argsort(anchor, kind="stable")[::-1][:n]].mean())
                    rec[f"held_top{int(tail * 100)}_delta_net_bps"] = float(rec[f"held_top{int(tail * 100)}_net_bps"] - rec[f"held_top{int(tail * 100)}_anchor_net_bps"])
                rows.append(rec)

    result = pd.DataFrame(rows)
    # Preserve the earlier proxy under an explicit name.  The canonical file
    # now means an actual train-only retrained portability result.
    proxy_path = out / "condition_lomo_results.parquet"
    if proxy_path.exists() and not (out / "condition_lomo_proxy_results.parquet").exists():
        shutil.copyfile(proxy_path, out / "condition_lomo_proxy_results.parquet")
    result.to_parquet(out / "condition_lomo_results.parquet", index=False)
    result.to_parquet(out / "condition_lomo_retrain_results.parquet", index=False)
    return result


def _write_report(
    out: Path,
    cfg: ConditionalSpecialistConfig,
    selected: dict[str, list[dict[str, Any]]],
    metrics: pd.DataFrame,
    exit_metrics: pd.DataFrame | None,
    lomo: pd.DataFrame,
    resource: dict[str, Any],
    exit_side_summary: pd.DataFrame | None = None,
    exit_month_summary: pd.DataFrame | None = None,
    feature_selection_method: str = "rank_portability",
) -> None:
    def _md_table(frame: pd.DataFrame, *, float_digits: int = 2) -> str:
        if frame.empty:
            return "(empty)"
        x = frame.copy().reset_index()
        headers = [str(c) for c in x.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in x.itertuples(index=False, name=None):
            cells = []
            for value in row:
                if isinstance(value, (float, np.floating)) and np.isfinite(value):
                    cells.append(f"{float(value):.{float_digits}f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = ["# Portable pair-condition specialist pipeline (2026-08-06)", "", f"Artifact: `{out.name}/`", "", "## Contract", "", "The pipeline uses the existing TP6/SL4/H12 ledger, 100-bps single cost, native LambdaRank and 4-hour×side queries. Conditions are causal soft pair states fit only before 2024-06-01; the transport folds are July–August, September–October and November 2024. The final ranking is global across both sides.", "", "### Conversion and score comparability", "", "Each specialist is a side-local conversion model trained on the canonical per-row residual (exact H12 net bps minus the base side-local EV), ordinalized at [-150, -50, +50, +150] bps and fit with the native query ranker. Its output is converted with the prior-resolved same-side monotone 20-bin EV map over all 4-hour queries; the resulting bps score is the only value passed to the pooled global top-k ranking. No pooled map or cross-side raw-score comparison is used. OOF bundles persist raw score, within-query percentile rank, membership, uncertainty, OOD, gated score, innovation and gated innovation. `score_calibration.parquet` audits this side-local map by side, fold, month, score bin and query count.", "", f"Feature selection method: `{feature_selection_method}`. The `condition_feature_mda_<side>.parquet` and `condition_feature_cap_ablation_<side>.parquet` files are discovery-only, chronological, condition-weighted group-MDA/cap evidence; transport rows never enter this selection.", "", f"Configuration: `{json.dumps(cfg.to_dict(), sort_keys=True)}`", "", "## Selected conditions", ""]
    for side in ("long", "short"):
        lines.append(f"### {side}")
        if not selected[side]:
            lines.append("No condition passed the support screen.")
        for i, c in enumerate(selected[side], 1):
            lines.append(f"{i}. `{c['condition_id']}`: `{c['context_feature_a']} {c['activation_a']} × {c['context_feature_b']} {c['activation_b']}`; effective rows {c['effective_rows']:.0f}, supported months {int(c['supported_month_count'])}, relevance {c.get('relevance', float('nan')):.4f}.")
    lines += ["", "## Condition definitions, feature behaviour and decisions", "", "Each scalar activation is causal and fit before the discovery cutoff. For a field x, the low membership is `sigmoid((q25 - x) / scale)` and the high membership is `sigmoid((x - q75) / scale)`; pair membership is the product, raised to the configured exponent 1.5 for specialist weighting. Missing values receive zero membership. The semantic labels (low/high) are descriptive only; they are not hand-coded trading rules.", ""]
    definition_rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        selected_ids = {str(c["condition_id"]): c for c in selected[side]}
        feature_path = out / f"condition_feature_selection_{side}.parquet"
        feature_frame = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
        behavior_path = out / f"condition_feature_behavior_monthly_{side}.parquet"
        behavior = pd.read_parquet(behavior_path) if behavior_path.exists() else pd.DataFrame()
        utility_path = out / f"condition_model_utility_monthly_{side}.parquet"
        utility = pd.read_parquet(utility_path) if utility_path.exists() else pd.DataFrame()
        standalone_path = out / "condition_specialist_standalone_metrics.parquet"
        standalone = pd.read_parquet(standalone_path) if standalone_path.exists() else pd.DataFrame()
        for cid, condition in selected_ids.items():
            top_features = []
            if not feature_frame.empty:
                top_features = feature_frame[(feature_frame.condition_id.astype(str).eq(cid)) & feature_frame.selected.astype(bool)].sort_values("rank").feature.astype(str).head(5).tolist()
            b = behavior[behavior.condition_id.astype(str).eq(cid)] if not behavior.empty else pd.DataFrame()
            u = utility[utility.condition_id.astype(str).eq(cid)] if not utility.empty else pd.DataFrame()
            s = standalone[(standalone.condition_id.astype(str).eq(cid)) & standalone.output.eq("rank") & standalone.period.eq("all") & standalone["tail"].eq(.10)] if not standalone.empty else pd.DataFrame()
            deltas = pd.to_numeric(u.get("delta_top10_net_bps", pd.Series(dtype=float)), errors="coerce").dropna()
            delta_ic = pd.to_numeric(u.get("delta_rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
            standalone_top10 = float(s.net_bps.iloc[0]) if not s.empty else float("nan")
            decision = "REJECT — LOW SUPPORT" if int(condition.get("supported_month_count", 0)) < 3 else ("REJECT — NO INCREMENTAL VALUE" if (not np.isfinite(standalone_top10) or standalone_top10 <= 0.0) and (deltas.empty or float(deltas.median()) <= 0.0) else "DIAGNOSTIC ONLY")
            definition_rows.append({
                "side": side,
                "condition": cid,
                "activation": f"{condition['context_feature_a']} {condition['activation_a']} × {condition['context_feature_b']} {condition['activation_b']}",
                "effective_rows": float(condition.get("effective_rows", np.nan)),
                "supported_months": int(condition.get("supported_month_count", 0)),
                "hard_joint_share": float(condition.get("joint_activation_hard_share", np.nan)),
                "membership_p50": float(condition.get("membership_p50", np.nan)),
                "membership_p90": float(condition.get("membership_p90", np.nan)),
                "median_delta_top10": float(deltas.median()) if not deltas.empty else float("nan"),
                "worst_delta_top10": float(deltas.min()) if not deltas.empty else float("nan"),
                "median_delta_rank_ic": float(delta_ic.median()) if not delta_ic.empty else float("nan"),
                "standalone_top10": standalone_top10,
                "top_features": ", ".join(top_features),
                "decision": decision,
            })
    if definition_rows:
        lines.append(_md_table(pd.DataFrame(definition_rows), float_digits=2))
        lines += ["", "The table combines discovery support, monthly feature-response diagnostics, model-response portability and standalone specialist value. A selected condition is not promoted merely because it was useful for discovery; the final decision requires positive portable economics."]
    lines += ["", "## OOS raw H12 meta ablation", "", "The authoritative comparison is the globally ranked OOS top-1/5/10 net table below. `anchor_only` is the no-specialist control; other arms use the same residual target and side-local expected-net mapping.", ""]
    if metrics.empty:
        lines.append("No OOS metrics were materialised.")
    else:
        pooled = metrics[(metrics.scope == "global") & metrics.period.eq("all") & metrics["tail"].isin([.01, .05, .10])]
        lines.append(_md_table(pooled.pivot_table(index="system", columns="tail", values="net_bps", aggfunc="first").rename(columns={.01: "top1", .05: "top5", .10: "top10"})))
    lines += ["", "## Monthly stability", ""]
    if not metrics.empty:
        monthly = metrics[(metrics.scope == "global") & metrics.period.str.match(r"^2024-") & metrics["tail"].eq(.05)]
        lines.append(_md_table(monthly.pivot(index="period", columns="system", values="net_bps")))
    lines += ["", "## Exit-policy replay", "", "The incumbent fixed policy is SL 3 ATR, trailing activation 0.5 ATR, giveback 0.25 ATR on the existing 15-minute/12-hour path source. These results are a coarse execution proxy and use the same global score ranking. The matched union is the pair-condition runner's common replay population; it is not numerically identical to the separately published incumbent exit-grid artifact, so the latter remains the production reference.", ""]
    if exit_metrics is not None and not exit_metrics.empty:
        lines.append(_md_table(exit_metrics))
    if exit_side_summary is not None and not exit_side_summary.empty:
        lines += ["", "### Exit-policy side/month variability", "", "These are decompositions of the globally ranked top-k trades, not side-local rerankings."]
        compact = exit_side_summary[exit_side_summary.side.isin(["long", "short"])].copy()
        compact = compact[["system", "side", "tail", "median_month_net_bps", "min_month_net_bps", "max_month_net_bps", "std_month_net_bps", "trades"]]
        lines.append(_md_table(compact, float_digits=2))
    if exit_month_summary is not None and not exit_month_summary.empty:
        lines += ["", "### Exit-policy month-to-month variability", "", "Each month is globally reranked before taking its top-k tail."]
        compact = exit_month_summary[exit_month_summary.side.isin(["all", "long", "short"])].copy()
        compact = compact[["system", "side", "tail", "median_month_net_bps", "min_month_net_bps", "max_month_net_bps", "std_month_net_bps", "months"]]
        lines.append(_md_table(compact, float_digits=2))
    same_union_path = out / "incumbent_exit_comparison_same_union.parquet"
    same_union_period_path = out / "fixed_exit_same_union_period_metrics.parquet"
    if same_union_path.exists():
        same_union = pd.read_parquet(same_union_path)
        lines += ["", "### Same-union incumbent comparison", "", "This is the authoritative fixed-exit comparison: the challenger top-10 union is frozen first, then the incumbent and every challenger are ranked on exactly the same matched path population. It avoids changing the population by adding the incumbent to the union construction."]
        lines.append(_md_table(same_union, float_digits=2))
    if same_union_period_path.exists():
        same_period = pd.read_parquet(same_union_period_path)
        month5 = same_period[(same_period["period_type"] == "month") & same_period["tail"].eq(.05)].copy()
        week5 = same_period[(same_period["period_type"] == "week") & same_period["tail"].eq(.05)].copy()
        if not month5.empty:
            month_summary = month5.groupby("system", as_index=True).net_bps.agg(["median", "min", "max", "std", "count"]).rename(columns={"median": "median_month_top5_net_bps", "min": "worst_month_top5_net_bps", "max": "best_month_top5_net_bps", "std": "std_month_top5_net_bps", "count": "months"})
            lines += ["", "#### Same-union monthly top-5 stability", ""]
            lines.append(_md_table(month_summary, float_digits=2))
        if not week5.empty:
            week_summary = week5.groupby("system", as_index=True).net_bps.agg(["median", "min", "max", "std", "count"]).rename(columns={"median": "median_week_top5_net_bps", "min": "worst_week_top5_net_bps", "max": "best_week_top5_net_bps", "std": "std_week_top5_net_bps", "count": "weeks"})
            lines += ["", "#### Same-union weekly top-5 stability", ""]
            lines.append(_md_table(week_summary, float_digits=2))
    mda_tables = []
    for side in ("long", "short"):
        path = out / f"condition_feature_cap_ablation_{side}.parquet"
        if not path.exists():
            continue
        cap = pd.read_parquet(path)
        if cap.empty:
            continue
        # The cap file also stores monthly validation rows for diagnostics.  The
        # ``__portable__`` row is the predeclared aggregate used to choose the
        # cap and is the only row shown in the compact report table.
        if "month" in cap.columns:
            cap = cap[cap["month"].eq("__portable__")].copy()
        keep = [c for c in ["condition_id", "cap", "feature_count", "portable_top10_net_bps", "selected"] if c in cap.columns]
        if "portable_top10_net_bps" not in cap.columns and "validation_top10_net_bps" in cap.columns:
            cap = cap.rename(columns={"validation_top10_net_bps": "portable_top10_net_bps"})
            keep = [c for c in ["condition_id", "cap", "feature_count", "portable_top10_net_bps", "selected"] if c in cap.columns]
        if keep:
            view = cap[keep].copy()
            view.insert(0, "side", side)
            mda_tables.append(view)
    if mda_tables:
        lines += ["", "## Condition-weighted group-MDA and feature-cap ablation", "", "Feature order and cap are selected on discovery rows only. The smallest cap within the portable top-10 validation tolerance is retained per condition; transport months are not used in this choice."]
        lines.append(_md_table(pd.concat(mda_tables, ignore_index=True), float_digits=2))
    lines += ["", "## Leave-one-month diagnostic", ""]
    if lomo.empty:
        lines.append("No supported condition-month rows were available.")
    elif "method" in lomo.columns:
        # True LOMO rows are generated by train-only threshold/model fits.  Do
        # not reuse the old proxy's ``condition_meaning_stable`` field here.
        delta_col = "held_top10_delta_net_bps" if "held_top10_delta_net_bps" in lomo.columns else None
        if delta_col is not None:
            positive = lomo.groupby("side")[delta_col].apply(lambda x: float(np.mean(np.asarray(x, dtype=float) > 0.0)))
            lines.append(f"{len(lomo)} train-only retrained condition-month holdouts were evaluated; positive top-10 delta fraction: {float(np.mean(np.asarray(lomo[delta_col], dtype=float) > 0.0)):.3f}.")
            lines.append(_md_table(positive.to_frame("positive_top10_delta_fraction"), float_digits=3))
            summary = lomo.groupby("side")[delta_col].agg(["median", "min", "max", "count"]).rename(columns={"median": "median_top10_delta_net_bps", "min": "worst_top10_delta_net_bps", "max": "best_top10_delta_net_bps", "count": "holdouts"})
            lines.append(_md_table(summary, float_digits=2))
        else:
            lines.append(f"{len(lomo)} train-only retrained condition-month holdouts were evaluated.")
        lines.append("Thresholds, membership weights, LambdaRank specialists and innovation residualizers were fit without the held-out month.")
    else:
        lines.append(f"{len(lomo)} condition-month holdouts were evaluated. Stable-sign fraction: {float(lomo.condition_meaning_stable.mean()):.3f}.")
        lines.append(_md_table(lomo.groupby("side").condition_meaning_stable.mean().to_frame("stable_sign_fraction"), float_digits=3))
    standalone_path = out / "condition_specialist_standalone_metrics.parquet"
    if standalone_path.exists():
        standalone = pd.read_parquet(standalone_path)
        selected_ids = {str(c["condition_id"]) for values in selected.values() for c in values}
        standalone = standalone[standalone.condition_id.astype(str).isin(selected_ids)]
        standalone = standalone[(standalone.period == "all") & standalone.output.isin(["rank", "gated_rank"]) & standalone["tail"].isin([.05, .10])]
        if not standalone.empty:
            lines += ["", "## Specialist standalone performance", "", "Standalone outputs are evaluated on the same OOF rows before entering the residual meta learner."]
            lines.append(_md_table(standalone.pivot_table(index=["condition_id", "output"], columns="tail", values="net_bps", aggfunc="first").rename(columns={.05: "top5_net_bps", .10: "top10_net_bps"})))
    complementarity_tables = []
    for side in ("long", "short"):
        path = out / f"condition_complementarity_{side}.parquet"
        if path.exists():
            comp = pd.read_parquet(path)
            if not comp.empty:
                comp["side"] = side
                complementarity_tables.append(comp.nlargest(5, "combined_complementarity"))
    if complementarity_tables:
        lines += ["", "## Condition complementarity", "", "The selection surface combines centred model-response distance, feature-response distance and one-minus weighted membership overlap."]
        lines.append(_md_table(pd.concat(complementarity_tables, ignore_index=True)[["side", "condition_a", "condition_b", "model_response_distance", "feature_response_distance", "membership_overlap", "combined_complementarity"]]))
    control_path = out / "control_metrics.parquet"
    if control_path.exists():
        controls = pd.read_parquet(control_path)
        controls = controls[(controls.scope == "global") & (controls.period == "all") & controls["tail"].isin([.05, .10])]
        if not controls.empty:
            lines += ["", "## Model and gating controls", "", "The anchor/equal-average/fixed-blend rows are fixed same-population controls. `regularized_linear_blend`, `geometry_only_gmm` and `full_context_gmm` are fold-local OOS arms: the ridge/GMM fits use only training rows and are transformed forward. `full_context_gmm` combines the causal spine, specialist diagnostics and fold-local soft GMM geometry. `memberships` is the soft condition-probability control, `raw_ranks` is no gating, `gated_ranks`/`gated_innovations` are soft membership gating, `hard_gating` applies the predeclared 0.5 membership threshold, and `probability_only` passes condition probabilities without specialist scores. `frozen_multiview_stack` is the incumbent OOF control."]
            lines.append(_md_table(controls.pivot_table(index="system", columns="tail", values="net_bps", aggfunc="first").rename(columns={.05: "top5_net_bps", .10: "top10_net_bps"})))
    control_selection_path = out / "control_selection_audit.parquet"
    if control_selection_path.exists():
        discovery_controls = pd.read_parquet(control_selection_path)
        if not discovery_controls.empty:
            summary = discovery_controls.groupby(["control", "status"], as_index=False).agg(arms=("condition_id", "count"), sides=("side", "nunique"), model_refit_through_oos=("model_refit_through_oos", "max"))
            if (discovery_controls["status"] == "bounded_authoritative_oos").any():
                coverage_note = (
                    "The full selection surface is authoritative through OOS. "
                    "The first retained condition per bounded control (univariate, "
                    "random-supported, geometry-only, no-model-utility and "
                    "no-feature-behavior) also has a strict train-fold-only OOS replay; "
                    "the remaining control rows are discovery-only screens."
                )
            else:
                coverage_note = (
                    "The full selection surface is the only condition-selection arm refit "
                    "through OOS. Univariate, random-supported, geometry-only, "
                    "no-model-utility and no-feature-behavior rows are discovery-only "
                    "control screens and are explicitly not treated as causal performance "
                    "results."
                )
            lines += ["", "## Discovery control coverage", "", coverage_note]
            lines.append(_md_table(summary))
    support_rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        support_path = out / f"candidate_support_gate_audit_{side}.parquet"
        candidate_path = out / f"condition_candidates_{side}.parquet"
        selected_path = out / f"selected_conditions_{side}.json"
        if not support_path.exists() or not candidate_path.exists() or not selected_path.exists():
            continue
        support = pd.read_parquet(support_path)
        selected_ids = {str(x["condition_id"]) for x in json.loads(selected_path.read_text()).get("conditions", [])}
        selected = support[support.condition_id.astype(str).isin(selected_ids)]
        gate_cols = [
            "support_gate_min_effective_rows",
            "support_gate_min_effective_queries",
            "support_gate_min_supported_months",
            "support_gate_min_nonadjacent_months",
        ]
        support_rows.append({
            "side": side,
            "candidate_rows": int(len(support)),
            "candidate_gate_pass_rate": float(support[gate_cols].all(axis=1).mean()) if len(support) else float("nan"),
            "selected_conditions": int(len(selected)),
            "selected_gate_pass": bool(selected[gate_cols].all(axis=1).all()) if len(selected) else False,
            "min_selected_nonadjacent_months": int(selected.supported_nonadjacent_month_count.min()) if len(selected) else 0,
        })
    if support_rows:
        lines += [
            "", "## Candidate recurrence/support gate", "",
            "Candidate support is recomputed from the frozen discovery activation spine using the exact effective-row, query, supported-month and non-adjacent-month thresholds. The retained conditions are selected only from rows passing all four gates; the broader cached candidate pool may include pre-gate discovery rows for audit traceability.",
            "", _md_table(pd.DataFrame(support_rows)),
        ]
    selection_control_path = out / "condition_selection_control_metrics.parquet"
    if selection_control_path.exists():
        selection_controls = pd.read_parquet(selection_control_path)
        selection_controls = selection_controls[(selection_controls.scope == "global") & (selection_controls.period == "all") & selection_controls["tail"].isin([.05, .10])]
        if not selection_controls.empty:
            lines += ["", "## OOS condition-generation controls", "", "These controls are strict train-fold-only replays of one frozen discovery condition per side using the same residual target, side-local map and global common-bps ranking. They are intentionally bounded diagnostics, not promoted specialists."]
            lines.append(_md_table(selection_controls.pivot_table(index="system", columns="tail", values="net_bps", aggfunc="first").rename(columns={.05: "top5_net_bps", .10: "top10_net_bps"})))
    lines += ["", "## Resource usage", "", f"`{json.dumps(resource, sort_keys=True)}`", "", "## Decision", "", "A condition-specialist arm advances only if it improves the incumbent raw/exit top-5 and top-10 net while not worsening the worst transport month. The report intentionally keeps all arms and all condition diagnostics; no pooled uplift is promoted without this gate.", ""]
    (out / "PAIR_CONDITION_SPECIALIST_REPORT.md").write_text("\n".join(lines) + "\n")


def run(
    out: Path = OUT,
    *,
    skip_exit: bool = False,
    condition_weight_exponent: float = 1.5,
    feature_selection_method: str = "rank_portability",
    run_selection_controls: bool = True,
) -> Path:
    if feature_selection_method not in {"rank_portability", "condition_group_mda"}:
        raise ValueError("feature_selection_method must be rank_portability or condition_group_mda")
    started = time.perf_counter()
    cfg = ConditionalSpecialistConfig(
        global_seed=SEED,
        condition_weight_exponent=float(condition_weight_exponent),
    )
    out.mkdir(parents=True, exist_ok=True)
    base = _base_frame(); available = set(_schema())
    cached = all((out / f"condition_activation_manifest_{side}.json").exists() and (out / f"condition_spine_manifest_{side}.json").exists() and (out / f"selected_conditions_{side}.json").exists() and (out / f"condition_feature_sets_{side}.json").exists() for side in ("long", "short")) and (out / "feature_pool_manifest.json").exists()
    if feature_selection_method == "condition_group_mda":
        cached = cached and all((out / f"condition_feature_mda_{side}.parquet").exists() and (out / f"condition_feature_cap_ablation_{side}.parquet").exists() for side in ("long", "short"))
    if cached:
        spine_manifests = {side: json.loads((out / f"condition_activation_manifest_{side}.json").read_text())["features"] for side in ("long", "short")}
        spine_payload = {side: json.loads((out / f"condition_spine_manifest_{side}.json").read_text()) for side in ("long", "short")}
        spines = {side: list(spine_payload[side]["fields"]) for side in ("long", "short")}
        pool = json.loads((out / "feature_pool_manifest.json").read_text())
        predictive_by_side = {side: list(pool["predictive_fields_by_side"][side]) for side in ("long", "short")}
    else:
        spine_manifests, spines, predictive_by_side = _fit_condition_spine(base, available, cfg, out)
    selected: dict[str, list[dict[str, Any]]] = {}; feature_sets: dict[str, dict[str, list[str]]] = {}; monthly: dict[str, pd.DataFrame] = {}
    condition_artifacts: dict[str, Any] = {}
    for side in ("long", "short"):
        if cached and (out / f"condition_model_utility_monthly_{side}.parquet").exists():
            selected_payload = json.loads((out / f"selected_conditions_{side}.json").read_text())
            feature_payload = json.loads((out / f"condition_feature_sets_{side}.json").read_text())
            selected[side] = list(selected_payload.get("conditions", []))
            feature_sets[side] = {str(k): list(v) for k, v in feature_payload.get("sets", {}).items()}
            monthly[side] = pd.read_parquet(out / f"condition_model_utility_monthly_{side}.parquet")
            candidate_file = out / f"condition_candidates_{side}.parquet"
            condition_artifacts[side] = {"candidate_count": int(len(pd.read_parquet(candidate_file))) if candidate_file.exists() else None, "selected_count": len(selected[side]), "reused_discovery": True}
            continue
        dev = _sample(base[(base.side_name.eq(side)) & base.__ts__.lt(DISCOVERY_END)], DISCOVERY_SAMPLE_ROWS, seed=SEED + (0 if side == "long" else 1))
        fields = list(dict.fromkeys(spines[side] + predictive_by_side[side]))
        joined = _store_rows(dev, fields); dev = dev.merge(joined, on="candidate_id", validate="one_to_one")
        activation_manifest = spine_manifests[side]
        candidates, activations = _generate_candidates(dev, side, spines[side], activation_manifest, cfg)
        candidates.to_parquet(out / f"condition_candidates_{side}.parquet", index=False)
        cheap_monthly = _monthly_model_response(dev, candidates, activations, cfg.top_candidates_per_side, side)
        monthly[side] = cheap_monthly
        cheap = candidates.copy()
        grouped = cheap_monthly.groupby("condition_id") if not cheap_monthly.empty else []
        portable_rows = []
        for cid, g in grouped:
            portable_rows.append((cid, portability_score(g.delta_top10_net_bps), portability_score(g.delta_rank_ic), int(g.month.nunique())))
        port_map = {cid: (a, b, c) for cid, a, b, c in portable_rows}
        cheap["portable_delta_top10_net_bps"] = cheap.condition_id.map(lambda x: port_map.get(x, (np.nan, np.nan, 0))[0])
        cheap["portable_delta_rank_ic"] = cheap.condition_id.map(lambda x: port_map.get(x, (np.nan, np.nan, 0))[1])
        cheap["screen_supported_months"] = cheap.condition_id.map(lambda x: port_map.get(x, (np.nan, np.nan, 0))[2])
        cheap["cheap_screen_score"] = cheap.candidate_screen_score.fillna(0.0) + cheap.portable_delta_top10_net_bps.fillna(0.0) + cheap.portable_delta_rank_ic.fillna(0.0) * 100.0
        cheap.to_parquet(out / f"condition_cheap_screen_{side}.parquet", index=False)
        top_full = cheap.sort_values(["cheap_screen_score", "effective_rows"], ascending=[False, False], kind="stable").head(cfg.top_conditions_for_full_feature_scan)
        # Rebuild the model-response matrix on the exact full-response
        # shortlist.  The cheap screen and full feature scan can select
        # different rows; using the former here would make selected condition
        # IDs absent from the leave-one-month audit.
        model_monthly = _monthly_model_response(dev, top_full, activations, len(top_full), side)
        model_monthly.to_parquet(out / f"condition_model_utility_monthly_{side}.parquet", index=False)
        monthly[side] = model_monthly
        _, feature_port = _full_feature_response(dev, top_full, activations, predictive_by_side[side], cfg.top_conditions_for_full_feature_scan, side, out)
        signatures, model_vectors = _build_response_signatures(top_full, model_monthly, feature_port, side, out)
        selected[side] = _select_conditions(top_full, signatures, model_vectors, dev, activations, cfg, side, out)
        # Attach relevance after the selection trace has been generated.
        rel = cheap.set_index("condition_id").cheap_screen_score.to_dict()
        for c in selected[side]:
            c["relevance"] = float(rel.get(c["condition_id"], np.nan))
        mda = None
        selected_caps: dict[str, int] = {}
        if feature_selection_method == "condition_group_mda":
            mda, _, selected_caps = _condition_feature_mda_caps(
                dev, selected[side], predictive_by_side[side], feature_port,
                activations, side, out, cfg,
            )
        feature_sets[side] = _select_condition_features(
            dev,
            selected[side],
            feature_port,
            predictive_by_side[side],
            side,
            out,
            cfg,
            mda=mda,
            selected_caps=selected_caps,
            method=feature_selection_method,
        )
        condition_artifacts[side] = {"candidate_count": len(candidates), "cheap_count": len(cheap), "full_count": len(top_full), "selected_count": len(selected[side])}
        gc.collect()
    _write_json(out / "selected_conditions.json", {"schema": "pair_condition_specialists_selected_v1", "selected": selected})
    _write_json(out / "condition_feature_sets.json", {"schema": "pair_condition_specialists_feature_sets_v1", "sets": feature_sets})
    predictions, metrics, specialist_oof = _run_outer(base, spine_manifests, spines, feature_sets, selected, out, specialist_config=cfg)
    if run_selection_controls:
        _, selection_control_metrics = _run_selection_control_replay(
            base, spine_manifests, spines, predictive_by_side, out, specialist_config=cfg,
        )
    elif (out / "condition_selection_control_metrics.parquet").exists():
        selection_control_metrics = pd.read_parquet(out / "condition_selection_control_metrics.parquet")
    else:
        selection_control_metrics = pd.DataFrame()
    # Keep the cheap response-sign diagnostic for comparison, then replace the
    # canonical LOMO artifact with a true train-only retrained audit.  The
    # latter is slower but is required before portability claims are made.
    _lomo_diagnostics(out, selected, monthly)
    lomo = _true_lomo_diagnostics(base, selected, spines, feature_sets, cfg, out)
    if lomo.empty:
        lomo = pd.read_parquet(out / "condition_lomo_results.parquet")
    exit_metrics = None if skip_exit else _fixed_exit_metrics(predictions, out, [c for c in predictions.columns if c.startswith("score__")])
    exit_side_summary = None if skip_exit else _fixed_exit_side_month_metrics(predictions, out, [c for c in predictions.columns if c.startswith("score__")])[1]
    exit_month_summary = None if skip_exit else _fixed_exit_month_metrics(predictions, out, [c for c in predictions.columns if c.startswith("score__")])[1]
    baseline = pd.read_parquet(BASELINE_PRED, columns=["candidate_id", "score"]).rename(columns={"score": "incumbent_score"})
    predictions_with_incumbent = predictions.merge(baseline, on="candidate_id", how="left", validate="one_to_one")
    predictions_with_incumbent.to_parquet(out / "predictions_with_incumbent.parquet", index=False)
    _fixed_exit_incumbent_comparison(
        predictions_with_incumbent,
        out,
        [c for c in predictions.columns if c.startswith("score__")],
    )
    authoritative = _authoritative_metrics(out)
    for side in ("long", "short"):
        _materialize_complementarity_from_artifacts(out, side)
    _materialize_model_bank_manifest(out, predictions)
    _materialize_specialist_standalone_metrics(out, selected, predictions)
    _materialize_control_metrics(out, predictions)
    _materialize_discovery_control_audit(out, selected)
    _materialize_side_artifacts(out, predictions, authoritative, lomo)
    _materialize_score_calibration(out, predictions)
    resource = {"elapsed_seconds": time.perf_counter() - started, "peak_rss_mb": _peak_rss_mb(), "base_rows": len(base), "transport_rows": len(predictions), "selected_conditions": condition_artifacts, "condition_weight_exponent": cfg.condition_weight_exponent, "equal_condition_month_weighting": cfg.equal_condition_month_weighting, "feature_selection_method": feature_selection_method, "selection_control_rows": int(len(selection_control_metrics))}
    _write_json(out / "condition_resource_usage.json", resource)
    _materialize_side_artifacts(out, predictions, authoritative, lomo)
    _write_report(out, cfg, selected, authoritative, exit_metrics, lomo, resource, exit_side_summary, exit_month_summary, feature_selection_method)
    _write_json(out / "run_manifest.json", {"schema": "portable_pair_condition_specialists_v1", "status": "complete", "discovery_end_utc": DISCOVERY_END.isoformat(), "folds": [f.name for f in TRANSPORT_FOLDS], "target": "canonical ordinalized H12 net residual bps", "cost_bps": 100.0, "query": "4h x side", "specialist_params": SPECIALIST_PARAMS, "meta_params": META_PARAMS, "selected_conditions": condition_artifacts, "skip_exit": skip_exit, "conversion_contract": "side_local_residual_ranker", "ev_mapping_contract": "prequential_same_side_monotone_pava_20_bins_over_all_queries", "global_ranking_contract": "mapped_common_bps_global_top_k", "score_calibration_artifact": "score_calibration.parquet", "condition_weight_exponent": cfg.condition_weight_exponent, "equal_condition_month_weighting": cfg.equal_condition_month_weighting, "feature_selection_method": feature_selection_method, "weight_exponent_ablation": [1.0, 1.5, 2.0], "lomo_artifact": "condition_lomo_results.parquet", "lomo_method": "train_only_thresholds_model_residualizer", "control_selection_artifact": "control_selection_audit.parquet", "condition_selection_control_artifact": "condition_selection_control_metrics.parquet", "candidate_support_gate_artifacts": ["candidate_support_gate_audit_long.parquet", "candidate_support_gate_audit_short.parquet"], "model_control_arms": ["frozen_multiview_stack", "equal_pair_average", "regularized_linear_blend", "geometry_only_gmm", "full_context_gmm"], "gating_control_arms": ["no_gating", "memberships", "raw_ranks", "gated_ranks", "hard_gating", "probability_only", "innovations", "gated_innovations"], "geometry_control_artifact": "geometry_control_folds.json", "same_union_exit_artifact": "incumbent_exit_comparison_same_union.parquet"})
    _write_json(out / "progress.json", {"status": "complete", "folds": [f.name for f in TRANSPORT_FOLDS]})
    return out


def finalize_existing(out: Path = OUT) -> Path:
    """Finalize a run whose OOS artifacts completed before report writing."""

    exponent = 1.5
    feature_selection_method = "rank_portability"
    existing_manifest = out / "run_manifest.json"
    if existing_manifest.exists():
        try:
            manifest_payload = json.loads(existing_manifest.read_text())
            exponent = float(manifest_payload.get("condition_weight_exponent", exponent))
            feature_selection_method = str(manifest_payload.get("feature_selection_method", feature_selection_method))
        except Exception:
            pass
    cfg = ConditionalSpecialistConfig(global_seed=SEED, condition_weight_exponent=exponent)
    selected = {side: json.loads((out / f"selected_conditions_{side}.json").read_text()).get("conditions", []) for side in ("long", "short")}
    metrics = _authoritative_metrics(out)
    exit_metrics = pd.read_parquet(out / "fixed_exit_metrics.parquet") if (out / "fixed_exit_metrics.parquet").exists() else None
    monthly = {side: pd.read_parquet(out / f"condition_model_utility_monthly_{side}.parquet") for side in ("long", "short") if (out / f"condition_model_utility_monthly_{side}.parquet").exists()}
    if (out / "condition_lomo_retrain_results.parquet").exists():
        lomo = pd.read_parquet(out / "condition_lomo_retrain_results.parquet")
    else:
        lomo = _lomo_diagnostics(out, selected, monthly)
    predictions = pd.read_parquet(out / "predictions.parquet")
    if (out / "predictions_with_incumbent.parquet").exists():
        predictions_with_incumbent = pd.read_parquet(out / "predictions_with_incumbent.parquet")
    else:
        baseline = pd.read_parquet(BASELINE_PRED, columns=["candidate_id", "score"]).rename(columns={"score": "incumbent_score"})
        predictions_with_incumbent = predictions.merge(baseline, on="candidate_id", how="left", validate="one_to_one")
        predictions_with_incumbent.to_parquet(out / "predictions_with_incumbent.parquet", index=False)
    _fixed_exit_incumbent_comparison(
        predictions_with_incumbent,
        out,
        [c for c in predictions.columns if c.startswith("score__")],
    )
    exit_side_summary = pd.read_parquet(out / "fixed_exit_side_month_summary.parquet") if (out / "fixed_exit_side_month_summary.parquet").exists() else None
    if exit_metrics is not None and exit_side_summary is None:
        exit_side_summary = _fixed_exit_side_month_metrics(predictions, out, [c for c in predictions.columns if c.startswith("score__")])[1]
    exit_month_summary = pd.read_parquet(out / "fixed_exit_month_summary.parquet") if (out / "fixed_exit_month_summary.parquet").exists() else None
    if exit_metrics is not None and exit_month_summary is None:
        exit_month_summary = _fixed_exit_month_metrics(predictions, out, [c for c in predictions.columns if c.startswith("score__")])[1]
    for side in ("long", "short"):
        _materialize_complementarity_from_artifacts(out, side)
    _materialize_model_bank_manifest(out, predictions)
    _materialize_specialist_standalone_metrics(out, selected, predictions)
    _materialize_control_metrics(out, predictions)
    _materialize_discovery_control_audit(out, selected)
    _materialize_side_artifacts(out, predictions, metrics, lomo)
    _materialize_score_calibration(out, predictions)
    # Preserve measured timing/row information from the completed modelling
    # pass when only the report/finalizer is rerun.  Finalisation must not
    # overwrite valid resource evidence with null placeholders.
    resource = {}
    if (out / "condition_resource_usage.json").exists():
        try:
            resource = json.loads((out / "condition_resource_usage.json").read_text())
        except Exception:
            resource = {}
    resource.update({"transport_rows": len(predictions), "selected_conditions": {side: {"selected_count": len(selected[side]), "reused_discovery": True} for side in selected}, "condition_weight_exponent": cfg.condition_weight_exponent, "equal_condition_month_weighting": cfg.equal_condition_month_weighting})
    if (out / "condition_selection_control_metrics.parquet").exists():
        resource["selection_control_rows"] = int(len(pd.read_parquet(out / "condition_selection_control_metrics.parquet")))
    resource.setdefault("peak_rss_mb", _peak_rss_mb())
    _write_json(out / "condition_resource_usage.json", resource)
    _materialize_side_artifacts(out, predictions, metrics, lomo)
    _write_report(out, cfg, selected, metrics, exit_metrics, lomo, resource, exit_side_summary, exit_month_summary, feature_selection_method)
    _write_json(out / "run_manifest.json", {"schema": "portable_pair_condition_specialists_v1", "status": "complete", "discovery_end_utc": DISCOVERY_END.isoformat(), "folds": [f.name for f in TRANSPORT_FOLDS], "target": "canonical ordinalized H12 net residual bps", "cost_bps": 100.0, "query": "4h x side", "selected_conditions": resource["selected_conditions"], "finalized_existing": True, "conversion_contract": "side_local_residual_ranker", "ev_mapping_contract": "prequential_same_side_monotone_pava_20_bins_over_all_queries", "global_ranking_contract": "mapped_common_bps_global_top_k", "score_calibration_artifact": "score_calibration.parquet", "condition_weight_exponent": cfg.condition_weight_exponent, "equal_condition_month_weighting": cfg.equal_condition_month_weighting, "feature_selection_method": feature_selection_method, "weight_exponent_ablation": [1.0, 1.5, 2.0], "lomo_artifact": "condition_lomo_results.parquet", "lomo_method": "train_only_thresholds_model_residualizer", "control_selection_artifact": "control_selection_audit.parquet", "condition_selection_control_artifact": "condition_selection_control_metrics.parquet", "candidate_support_gate_artifacts": ["candidate_support_gate_audit_long.parquet", "candidate_support_gate_audit_short.parquet"], "model_control_arms": ["frozen_multiview_stack", "equal_pair_average", "regularized_linear_blend", "geometry_only_gmm", "full_context_gmm"], "gating_control_arms": ["no_gating", "memberships", "raw_ranks", "gated_ranks", "hard_gating", "probability_only", "innovations", "gated_innovations"], "geometry_control_artifact": "geometry_control_folds.json", "same_union_exit_artifact": "incumbent_exit_comparison_same_union.parquet"})
    _write_json(out / "progress.json", {"status": "complete", "folds": [f.name for f in TRANSPORT_FOLDS]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-exit", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--condition-weight-exponent", type=float, choices=(1.0, 1.5, 2.0), default=1.5)
    parser.add_argument(
        "--feature-selection-method",
        choices=("rank_portability", "condition_group_mda"),
        default="rank_portability",
        help="Discovery-only specialist feature selection contract; MDA runs chronological group permutations and cap evidence.",
    )
    parser.add_argument(
        "--skip-selection-controls",
        action="store_true",
        help="Skip the bounded OOS discovery-control replay when only finalizing a cached run.",
    )
    args = parser.parse_args()
    print(finalize_existing(args.out) if args.finalize_existing else run(args.out, skip_exit=args.skip_exit, condition_weight_exponent=args.condition_weight_exponent, feature_selection_method=args.feature_selection_method, run_selection_controls=not args.skip_selection_controls))
