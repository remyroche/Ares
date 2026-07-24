#!/usr/bin/env python3
"""Nested OOS contextual ablations around the locked 1m rational policy family."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_ablation import evaluate_results  # noqa: E402
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_RATIONAL,
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
    constrained_params_to_vector,
    simulate_constrained_1m_paths,
)
from extreme_price_movements.simple_policy_1m_contextual import (  # noqa: E402
    apply_robust_state,
    beta_binomial_lower_score,
    fit_robust_state,
    geometry_scaled_params,
    normalized_atr_power,
    posterior_mixture_scale,
    quantize_scales,
    stable_fold_objective,
    support_shrink,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
    _write_json,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    INNER_FOLDS,
    ExperimentData,
    _evaluate,
    _indices_between,
    _objective,
    _optimise,
)


OUTPUT_KEYS = (
    "exit_bars", "exit_price", "gross_return", "net_return", "reason", "mfe", "mae",
    "capital_first_bar", "trailing_first_bar", "capital_binding_bars", "initial_capital_active", "order_valid",
)
OUTPUT_DTYPES = (np.int32, float, float, float, np.int8, float, float, np.int32, np.int32, np.int32, bool, bool)
OUTPUT_FILLS = (-1, np.nan, np.nan, np.nan, 0, np.nan, np.nan, -1, -1, 0, False, True)
SCALE_GRID = (0.80, 0.90, 1.00, 1.10, 1.20)
BETA_GRID = (-0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25)
ATR_POWER_GRID = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40)


def _load_context(rows: pd.DataFrame, rich_path: Path, parent_state_path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    rich = pd.read_parquet(rich_path)
    rich["timestamp"] = pd.to_datetime(rich["timestamp"], utc=True)
    rich["archetype_policy_key"] = rich["archetype_policy_key"].astype(str)
    key = ["timestamp", "symbol", "archetype_policy_key"]
    if rich.duplicated(key).any():
        raise RuntimeError("rich context key is not unique")
    base = rows.copy()
    base["archetype_policy_key"] = base["archetype_policy_key"].astype(str)
    safe_cols = [
        "gmm_ood_score", "cluster_entropy_norm", "mahalanobis_distance", "expected_mahalanobis",
        "dae_reconstruction_error_zscore", "meta_hit_probability_uncertainty_p1mp",
        "meta_parent_rank_uncertainty_p1mp", "expected_net_ev_after_1pct_mlp_direct",
        "state_ev_correction_mlp_direct", "rank_mlp_direct",
        "__regime_source_shock_impulse_score__", "__regime_source_volume_confirmation_score__",
        "__regime_source_dirty_shock_avoid_score__", "__regime_source_not_dirty_shock_score__",
    ]
    merged = base[key].merge(rich[key + safe_cols], on=key, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("rich context coverage is incomplete")
    merged = merged.drop(columns="_merge")

    posterior_cols = [f"resid_event_aegmm_gmm_cluster_posterior_{i}" for i in range(6)]
    state_cols = ["__ts__", "__symbol__", "side_name", "archetype_policy_key", "resid_event_aegmm_gmm_cluster_id"] + posterior_cols
    state = pd.read_parquet(parent_state_path, columns=state_cols).rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})
    state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True)
    state["archetype_policy_key"] = state["archetype_policy_key"].astype(str)
    state_key = ["timestamp", "symbol", "side_name", "archetype_policy_key"]
    if state.duplicated(state_key).any():
        raise RuntimeError("posterior state key is not unique")
    join = base[state_key].merge(state, on=state_key, how="left", validate="one_to_one", indicator=True)
    if not join["_merge"].eq("both").all():
        raise RuntimeError("posterior state coverage is incomplete")
    p = join[posterior_cols].to_numpy(dtype=np.float64)
    hard = pd.to_numeric(join["resid_event_aegmm_gmm_cluster_id"], errors="coerce").to_numpy(dtype=np.float64)
    bad = ~np.isfinite(p).all(axis=1) | (p.sum(axis=1) <= 0.0)
    for i in np.flatnonzero(bad):
        p[i] = 0.0
        if np.isfinite(hard[i]) and 0 <= int(hard[i]) < p.shape[1]:
            p[i, int(hard[i])] = 1.0
        else:
            p[i] = 1.0 / p.shape[1]
    p /= p.sum(axis=1, keepdims=True)
    merged["hard_gmm_cluster"] = np.argmax(p, axis=1).astype(np.int16)
    audit = {
        "rich_rows": int(len(rich)), "candidate_rows": int(len(rows)), "rich_coverage": 1.0,
        "posterior_coverage": 1.0, "posterior_components": p.shape[1], "posterior_fallback_rows": int(bad.sum()),
        "candidate_gmm_cluster_id_null_rate": float(rows["gmm_cluster_id"].isna().mean()),
        "join_key_rich": key, "join_key_posterior": state_key,
    }
    return merged, p, audit


def _load_atr(rows: pd.DataFrame, audit_path: Path) -> np.ndarray:
    audit = pd.read_parquet(audit_path).sort_values("row")
    if len(audit) != len(rows) or not audit["status"].eq("ok").all() or not np.array_equal(audit["row"].to_numpy(), np.arange(len(rows))):
        raise RuntimeError("causal ATR audit does not align exactly to candidate rows")
    if not np.array_equal(audit["symbol"].astype(str).to_numpy(), rows["symbol"].astype(str).to_numpy()):
        raise RuntimeError("causal ATR symbol order mismatch")
    audit_ts = pd.to_datetime(audit["timestamp"], utc=True).to_numpy()
    if not np.array_equal(audit_ts, pd.to_datetime(rows["timestamp"], utc=True).to_numpy()):
        raise RuntimeError("causal ATR timestamp order mismatch")
    return audit["effective_atr_fraction"].to_numpy(dtype=np.float64)


def _empty_outputs(n: int) -> dict[str, np.ndarray]:
    return {k: np.full(n, fill, dtype=dtype) for k, dtype, fill in zip(OUTPUT_KEYS, OUTPUT_DTYPES, OUTPUT_FILLS)}


def _simulate_contextual(
    data: ExperimentData,
    indices: np.ndarray,
    params_by_side: Mapping[str, Mapping[str, Any]],
    scales_all: np.ndarray,
    *,
    family: int = FAMILY_RATIONAL,
    atr_frac_all: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ordered = np.asarray(indices, dtype=np.int64)
    out = _empty_outputs(len(ordered))
    atr = data.atr_frac if atr_frac_all is None else np.asarray(atr_frac_all, dtype=np.float64)
    qscale = quantize_scales(np.asarray(scales_all, dtype=np.float64), step=0.05)
    for scale in np.unique(qscale[ordered]):
        scale_local = np.flatnonzero(qscale[ordered] == scale)
        for side_name, sign in (("long", 1.0), ("short", -1.0)):
            local = scale_local[data.side[ordered[scale_local]] * sign > 0.0]
            if not len(local):
                continue
            result = simulate_constrained_1m_paths(
                ordered[local], data.open0, data.high, data.low, data.close, data.side, atr,
                data.entry_spread, data.exit_spread,
                constrained_params_to_vector(geometry_scaled_params(params_by_side[side_name], float(scale))),
                int(family), data.spec.fee_per_side, data.spec.stop_base_gap_bps,
                data.spec.stop_through_fraction, data.spec.stop_max_gap_bps,
                data.spec.capital_trail_epsilon_atr,
            )
            for key_name, values in zip(OUTPUT_KEYS, result):
                out[key_name][local] = values
    return out


def _score(data: ExperimentData, idx: np.ndarray, params: Mapping[str, Mapping[str, Any]], scales: np.ndarray, atr: np.ndarray | None = None) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    outputs = _simulate_contextual(data, idx, params, scales, atr_frac_all=atr)
    score, diag = _objective(data, idx, outputs)
    if diag["ordering_valid_rate"] < 0.999999 or diag["initial_capital_active_rate"] < 0.999999:
        return -1e6, diag, outputs
    return score, diag, outputs


def _fit_group_scales(
    data: ExperimentData,
    fit_idx: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    labels: np.ndarray,
    *,
    prior_strength: float = 250.0,
    min_support: int = 50,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    labels = np.asarray(labels).astype(str)
    support = {g: int(np.sum(labels[fit_idx] == g)) for g in np.unique(labels[fit_idx])}
    raw = {g: 1.0 for g in support}
    all_scales = np.ones(len(labels), dtype=np.float64)
    for _ in range(2):
        for group in sorted(support):
            if support[group] < min_support:
                continue
            best = (-1e100, raw[group])
            for candidate in SCALE_GRID:
                trial_raw = dict(raw)
                trial_raw[group] = float(candidate)
                trial = np.ones(len(labels), dtype=np.float64)
                for name, value in trial_raw.items():
                    trial[labels == name] = support_shrink(value, support[name], prior_strength)
                score, _, _ = _score(data, fit_idx, params, trial)
                penalized = score - 0.002 * sum(math.log(max(v, 1e-9)) ** 2 for v in trial_raw.values())
                if penalized > best[0]:
                    best = (penalized, float(candidate))
            raw[group] = best[1]
    effective = {g: support_shrink(raw[g], support[g], prior_strength) if support[g] >= min_support else 1.0 for g in support}
    for group, value in effective.items():
        all_scales[labels == group] = value
    return all_scales, effective, support


def _fit_soft_scales(
    data: ExperimentData,
    fit_idx: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    posteriors: np.ndarray,
    *,
    prior_strength: float = 250.0,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    side_name = np.where(data.side > 0.0, "long", "short")
    k = posteriors.shape[1]
    raw = {(side, comp): 1.0 for side in ("long", "short") for comp in range(k)}
    support: dict[tuple[str, int], float] = {}
    for side in ("long", "short"):
        mask = (side_name[fit_idx] == side)
        for comp in range(k):
            weights = posteriors[fit_idx[mask], comp]
            support[(side, comp)] = float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12))

    def build(mapping: Mapping[tuple[str, int], float]) -> np.ndarray:
        result = np.ones(len(data.rows), dtype=np.float64)
        for side in ("long", "short"):
            values = np.asarray([
                support_shrink(mapping[(side, comp)], int(support[(side, comp)]), prior_strength) for comp in range(k)
            ])
            mask = side_name == side
            result[mask] = posterior_mixture_scale(posteriors[mask], values)
        return result

    for _ in range(2):
        for key in sorted(raw):
            best = (-1e100, raw[key])
            for candidate in SCALE_GRID:
                trial = dict(raw); trial[key] = float(candidate)
                score, _, _ = _score(data, fit_idx, params, build(trial))
                penalized = score - 0.002 * sum(math.log(max(v, 1e-9)) ** 2 for v in trial.values())
                if penalized > best[0]: best = (penalized, float(candidate))
            raw[key] = best[1]
    effective = {f"{s}__gmm_{k0}": support_shrink(v, int(support[(s, k0)]), prior_strength) for (s, k0), v in raw.items()}
    support_out = {f"{s}__gmm_{k0}": float(support[(s, k0)]) for s, k0 in raw}
    return build(raw), effective, support_out


def _composite(context: pd.DataFrame, columns: list[str], fit_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pieces, state_out = [], {}
    for column in columns:
        values = pd.to_numeric(context[column], errors="coerce").to_numpy(dtype=np.float64)
        if "mahal" in column or "ood" in column or "reconstruction" in column:
            values = np.log1p(np.maximum(values, 0.0))
        state = fit_robust_state(values[fit_idx])
        pieces.append(apply_robust_state(values, state))
        state_out[column] = {"median": state.median, "scale": state.scale}
    return np.mean(np.column_stack(pieces), axis=1), state_out


def _select_beta(data: ExperimentData, search_idx: np.ndarray, inner_idx: np.ndarray, params: Mapping[str, Mapping[str, Any]], context: pd.DataFrame, columns: list[str]) -> tuple[float, dict[str, Any]]:
    z, state = _composite(context, columns, search_idx)
    scored = []
    for beta in BETA_GRID:
        scales = np.clip(np.exp(beta * z), 0.75, 1.25)
        score, diag, _ = _score(data, inner_idx, params, scales)
        scored.append((score, beta, diag))
    best = max(scored, key=lambda x: x[0])
    return float(best[1]), {"search_scaler": state, "inner_scores": [{"beta": b, "objective": s} for s, b, _ in scored]}


def _atr_reference(data: ExperimentData, fit_idx: np.ndarray) -> np.ndarray:
    ref = np.ones(len(data.rows), dtype=np.float64)
    for sign in (1.0, -1.0):
        mask_fit = fit_idx[data.side[fit_idx] * sign > 0.0]
        median = float(np.median(data.atr_frac[mask_fit]))
        ref[data.side * sign > 0.0] = median
    return ref


def _select_atr_power(data: ExperimentData, search_idx: np.ndarray, inner_idx: np.ndarray, params: Mapping[str, Mapping[str, Any]]) -> tuple[float, list[dict[str, float]]]:
    ref = _atr_reference(data, search_idx)
    rows = []
    for power in ATR_POWER_GRID:
        atr = normalized_atr_power(data.atr_frac, ref, power)
        score, _, _ = _score(data, inner_idx, params, np.ones(len(data.rows)), atr)
        rows.append({"power": float(power), "objective": float(score)})
    return float(max(rows, key=lambda x: x["objective"])["power"]), rows


def _weighted_evaluate(data: ExperimentData, idx: np.ndarray, outputs: Mapping[str, np.ndarray], size_mult: np.ndarray) -> dict[str, Any]:
    rows = data.rows.iloc[idx].reset_index(drop=True)
    metrics, selected = evaluate_results(
        rows, outputs["exit_bars"], outputs["gross_return"], outputs["net_return"], outputs["reason"],
        outputs["mfe"], outputs["mae"], bar_minutes=1, apply_capacity=True,
    )
    chosen = np.flatnonzero(selected)
    if not len(chosen): return metrics
    rank = pd.to_numeric(rows.iloc[chosen]["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)
    base_size = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    mult = size_mult[idx][chosen]
    exposure_ratio = float(np.average(mult, weights=base_size))
    pnl = outputs["net_return"][chosen] * base_size * mult
    gross_pnl = outputs["gross_return"][chosen] * base_size * mult
    fee_pnl = (outputs["gross_return"][chosen] - outputs["net_return"][chosen]) * base_size * mult
    pnl_neutral = outputs["net_return"][chosen] * base_size * (mult / max(exposure_ratio, 1e-9))
    ts = pd.to_datetime(rows.iloc[chosen]["timestamp"], utc=True)
    week = ts.dt.tz_localize(None).dt.to_period("W").astype(str).reset_index(drop=True)
    weekly = pd.Series(pnl).groupby(week).sum().to_numpy(dtype=float)
    weekly_neutral = pd.Series(pnl_neutral).groupby(week).sum().to_numpy(dtype=float)
    equity = np.cumsum(pnl); dd = equity - np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]
    equity_neutral = np.cumsum(pnl_neutral); dd_neutral = equity_neutral - np.maximum.accumulate(np.r_[0.0, equity_neutral])[-len(equity_neutral):]
    mean, std, worst, max_dd = float(weekly.mean()), float(weekly.std()), float(weekly.min()), float(dd.min())
    neutral_objective = float(weekly_neutral.mean() - 0.5 * weekly_neutral.std() + 0.25 * weekly_neutral.min() - 0.10 * abs(dd_neutral.min()))
    metrics.update({
        "gross_pnl_bankroll": float(gross_pnl.sum()), "fee_pnl_bankroll": float(fee_pnl.sum()),
        "net_pnl_bankroll": float(pnl.sum()), "worst_week": worst, "max_drawdown": max_dd,
        "objective": mean - 0.5 * std + 0.25 * worst - 0.10 * abs(max_dd),
        "mean_size_multiplier": exposure_ratio, "oos_exposure_ratio": exposure_ratio,
        "size_multiplier_p10": float(np.quantile(mult, 0.1)), "size_multiplier_p90": float(np.quantile(mult, 0.9)),
        "gross_notional_exposure": float(np.sum(base_size * mult)),
        "exposure_normalized_pnl": float(pnl_neutral.sum()),
        "exposure_normalized_objective": neutral_objective,
    })
    return metrics


def _bar_neutral_sizes(data: ExperimentData, idx: np.ndarray, outputs: Mapping[str, np.ndarray], size_mult: np.ndarray) -> np.ndarray:
    """Normalize admitted entries at each decision timestamp, after size-independent capacity selection."""
    rows = data.rows.iloc[idx].reset_index(drop=True)
    _, selected = evaluate_results(
        rows, outputs["exit_bars"], outputs["gross_return"], outputs["net_return"], outputs["reason"],
        outputs["mfe"], outputs["mae"], bar_minutes=1, apply_capacity=True,
    )
    chosen = np.flatnonzero(selected)
    result = np.asarray(size_mult, dtype=np.float64).copy()
    if not len(chosen): return result
    rank = pd.to_numeric(rows.iloc[chosen]["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)
    base_size = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    ts = pd.to_datetime(rows.iloc[chosen]["timestamp"], utc=True).astype("int64").to_numpy()
    local_mult = result[idx][chosen].copy()
    for timestamp in np.unique(ts):
        local = np.flatnonzero(ts == timestamp)
        normalizer = float(np.average(local_mult[local], weights=base_size[local]))
        local_mult[local] /= max(normalizer, 1e-9)
    target = idx[chosen]
    result[target] = local_mult
    return result


def _bayesian_sizes(
    data: ExperimentData,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    outputs_fit: Mapping[str, np.ndarray],
    context: pd.DataFrame,
    *,
    strength: float,
    ood_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    quality_cols = ["expected_net_ev_after_1pct_mlp_direct", "meta_hit_probability_uncertainty_p1mp"]
    z_ev, _ = _composite(context, quality_cols[:1], fit_idx)
    z_unc, _ = _composite(context, quality_cols[1:], fit_idx)
    z_ood, _ = _composite(context, ["gmm_ood_score", "cluster_entropy_norm"], fit_idx)
    quality = z_ev - 0.5 * z_unc - float(ood_weight) * z_ood
    cuts = np.unique(np.quantile(quality[fit_idx], (0.2, 0.4, 0.6, 0.8)))
    bins = np.digitize(quality, cuts)
    archetype = data.rows["policy_archetype"].astype(str).to_numpy()
    side = np.where(data.side > 0.0, "long", "short")
    finite_fit = np.isfinite(outputs_fit["net_return"])
    hit = outputs_fit["net_return"] > 0.0
    lookup: dict[tuple[str, int], float] = {}
    side_prior: dict[str, float] = {}
    for side_name in ("long", "short"):
        local = np.flatnonzero((side[fit_idx] == side_name) & finite_fit)
        side_prior[side_name] = float((hit[local].sum() + 8.0) / (len(local) + 16.0)) if len(local) else 0.5
    for name in np.unique(archetype[fit_idx]):
        for b in range(len(cuts) + 1):
            local = np.flatnonzero((archetype[fit_idx] == name) & (bins[fit_idx] == b) & finite_fit)
            if len(local) < 20: continue
            lookup[(name, b)] = float(beta_binomial_lower_score(np.array([hit[local].sum()]), np.array([len(local)]), prior_success=16.0, prior_failure=16.0, uncertainty_aversion=1.0)[0])
    signal = np.asarray([lookup.get((archetype[i], int(bins[i])), side_prior[side[i]]) for i in range(len(data.rows))])
    centered = signal - np.asarray([side_prior[s] for s in side])
    raw = np.clip(np.exp(float(strength) * centered), 0.65, 1.20)
    train_rank = data.rank[fit_idx]; train_base = 0.075 + 0.075 * np.power(np.clip(train_rank, 0.0, 1.0), 1.1)
    norm = float(np.average(raw[fit_idx], weights=train_base))
    sizes = np.clip(raw / max(norm, 1e-9), 0.65, 1.20)
    return sizes, {"quality_cuts": cuts.tolist(), "cells": len(lookup), "side_prior": side_prior, "train_normalizer": norm}


def _record(data: ExperimentData, idx: np.ndarray, outputs: Mapping[str, np.ndarray], *, fold: str, ablation: str, parent: str, extras: Mapping[str, Any] | None = None, size_mult: np.ndarray | None = None) -> dict[str, Any]:
    metrics, _ = _evaluate(data, idx, outputs, family=FAMILY_RATIONAL)
    if size_mult is not None:
        metrics.update(_weighted_evaluate(data, idx, outputs, size_mult))
    else:
        metrics.update({"oos_exposure_ratio": 1.0, "exposure_normalized_pnl": metrics["net_pnl_bankroll"], "exposure_normalized_objective": metrics["objective"]})
    row = {"fold": fold, "ablation": ablation, "parent": parent, **metrics}
    if extras: row.update(extras)
    return row


def _local_context_robustness(
    data: ExperimentData,
    idx: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    scales: np.ndarray,
    atr: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, float]:
    """Exact train-only replay under small coherent geometry perturbations."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(int(n)):
        global_scale = float(np.exp(rng.normal(0.0, 0.04)))
        perturbed_params = {side: geometry_scaled_params(values, global_scale) for side, values in params.items()}
        strength = float(np.exp(rng.normal(0.0, 0.06)))
        perturbed_scales = np.clip(1.0 + (scales - 1.0) * strength, 0.70, 1.35)
        atr_scale = float(np.exp(rng.normal(0.0, 0.015)))
        score, _, _ = _score(data, idx, perturbed_params, perturbed_scales, atr * atr_scale)
        scores.append(score)
    values = np.asarray(scores, dtype=np.float64)
    return {
        "local_perturbations": int(n), "local_median_objective_train": float(np.median(values)),
        "local_worst_objective_train": float(values.min()), "local_positive_fraction_train": float(np.mean(values > 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rich-ledger", required=True)
    parser.add_argument("--posterior-state", required=True)
    parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--path-cache-dir", required=True)
    parser.add_argument("--atr-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-trials", type=int, default=64)
    parser.add_argument("--parent-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.monotonic(); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, posteriors, provenance = _load_context(rows, Path(args.rich_ledger), Path(args.posterior_state))
    atr_frac = _load_atr(rows, Path(args.atr_audit))
    deployed, _ = _load_deployed_side_params(Path(args.deployed_parent_summary))
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows, store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"), cache_dir=Path(args.path_cache_dir), spec=spec, rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr_frac, spec, deployed)
    if data.valid.mean() < 0.999: raise RuntimeError("exact replay coverage below 99.9%")
    results: list[dict[str, Any]] = []; params_out: dict[str, Any] = {}; selections: dict[str, Any] = {}
    arch_labels = rows["policy_archetype"].astype(str).to_numpy()
    hard_labels = np.char.add(np.where(data.side > 0.0, "long__gmm_", "short__gmm_"), context["hard_gmm_cluster"].astype(str).to_numpy())
    ae_cols = ["gmm_ood_score", "cluster_entropy_norm", "mahalanobis_distance", "dae_reconstruction_error_zscore"]
    model_cols = ["expected_net_ev_after_1pct_mlp_direct", "state_ev_correction_mlp_direct", "rank_mlp_direct", "meta_parent_rank_uncertainty_p1mp"]
    regime_cols = ["__regime_source_shock_impulse_score__", "__regime_source_volume_confirmation_score__", "__regime_source_dirty_shock_avoid_score__", "__regime_source_not_dirty_shock_score__"]

    for fold_no, fold in enumerate(FOLDS, 1):
        inner = INNER_FOLDS[fold["fold"]]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        print(f"[{fold['fold']}] refit rational parent train={len(train_idx)} outer={len(outer_idx)}", flush=True)
        seeds = [args.seed + fold_no * 10_000 + i * 1_000 for i in range(args.parent_seeds)]
        parent_search, parent_search_diag = _optimise(
            data, search_idx, family=FAMILY_RATIONAL, joint=True,
            trials_per_seed=max(args.parent_trials // 2, 24), seeds=[s + 77 for s in seeds], sampler_kind="tpe",
        )
        parent, parent_diag = _optimise(data, train_idx, family=FAMILY_RATIONAL, joint=True, trials_per_seed=args.parent_trials, seeds=seeds, sampler_kind="tpe")
        params_out[f"{fold['fold']}__parent"] = {
            "params_by_side": parent, "optimizer": parent_diag,
            "inner_selection_parent": parent_search, "inner_selection_optimizer": parent_search_diag,
        }
        ones = np.ones(len(rows), dtype=np.float64)
        _, _, parent_outer = _score(data, outer_idx, parent, ones)
        results.append(_record(data, outer_idx, parent_outer, fold=fold["fold"], ablation="rational_parent", parent="rational_parent"))

        trailing, _ = _optimise(data, train_idx, family=FAMILY_TRAILING_ONLY, joint=True, trials_per_seed=max(args.parent_trials // 2, 24), seeds=[s + 333 for s in seeds], sampler_kind="tpe")
        trailing_out = data.simulate(outer_idx, trailing, FAMILY_TRAILING_ONLY)
        trailing_metrics, _ = _evaluate(data, outer_idx, trailing_out, family=FAMILY_TRAILING_ONLY)
        results.append({"fold": fold["fold"], "ablation": "trailing_only_parent", "parent": "trailing_only_parent", **trailing_metrics, "oos_exposure_ratio": 1.0, "exposure_normalized_pnl": trailing_metrics["net_pnl_bankroll"], "exposure_normalized_objective": trailing_metrics["objective"]})

        print("  hard policy archetype", flush=True)
        arch_scale, arch_map, arch_support = _fit_group_scales(data, train_idx, parent, arch_labels)
        _, _, out = _score(data, outer_idx, parent, arch_scale)
        results.append(_record(data, outer_idx, out, fold=fold["fold"], ablation="hard_policy_archetype", parent="rational_parent", extras={"intervention_rate": float(np.mean(arch_scale[outer_idx] != 1.0))}))

        print("  hard GMM", flush=True)
        hard_scale, hard_map, hard_support = _fit_group_scales(data, train_idx, parent, hard_labels)
        _, _, out = _score(data, outer_idx, parent, hard_scale)
        results.append(_record(data, outer_idx, out, fold=fold["fold"], ablation="hard_gmm", parent="rational_parent", extras={"intervention_rate": float(np.mean(hard_scale[outer_idx] != 1.0))}))

        print("  posterior-weighted GMM", flush=True)
        soft_scale, soft_map, soft_support = _fit_soft_scales(data, train_idx, parent, posteriors)
        _, _, out = _score(data, outer_idx, parent, soft_scale)
        results.append(_record(data, outer_idx, out, fold=fold["fold"], ablation="soft_gmm_posterior", parent="rational_parent", extras={"intervention_rate": float(np.mean(np.abs(soft_scale[outer_idx] - 1.0) > 0.025))}))

        context_scales: dict[str, np.ndarray] = {}
        context_search_scales: dict[str, np.ndarray] = {}
        context_meta: dict[str, Any] = {}
        for name, columns in (("ae_gmm_exit", ae_cols), ("mlp_meta_exit", model_cols), ("regime_exit", regime_cols)):
            beta, meta = _select_beta(data, search_idx, inner_idx, parent_search, context, columns)
            z_search, _ = _composite(context, columns, search_idx)
            context_search_scales[name] = np.clip(np.exp(beta * z_search), 0.75, 1.25)
            z, scaler = _composite(context, columns, train_idx)
            scale = np.clip(np.exp(beta * z), 0.75, 1.25)
            context_scales[name] = scale; context_meta[name] = {"beta": beta, "scaler": scaler, **meta}
            _, _, out = _score(data, outer_idx, parent, scale)
            results.append(_record(data, outer_idx, out, fold=fold["fold"], ablation=name, parent="rational_parent", extras={"selected_beta": beta, "intervention_rate": float(beta != 0.0)}))

        power, power_search = _select_atr_power(data, search_idx, inner_idx, parent_search)
        ref = _atr_reference(data, train_idx); atr_powered = normalized_atr_power(data.atr_frac, ref, power)
        _, _, out = _score(data, outer_idx, parent, ones, atr_powered)
        results.append(_record(data, outer_idx, out, fold=fold["fold"], ablation="atr_power", parent="rational_parent", extras={"selected_atr_power": power}))

        # Sizing uses the frozen parent exits and selects only conservative OOD weight/strength on inner validation.
        _, _, fit_parent_out = _score(data, train_idx, parent, ones)
        size_candidates = []
        search_parent_out = _score(data, search_idx, parent_search, ones)[2]
        inner_parent_out = _score(data, inner_idx, parent_search, ones)[2]
        for strength in (1.5, 3.0, 4.5):
            for ood_weight in (0.0, 0.5, 1.0):
                sizes, _ = _bayesian_sizes(data, search_idx, inner_idx, search_parent_out, context, strength=strength, ood_weight=ood_weight)
                metric = _weighted_evaluate(data, inner_idx, inner_parent_out, sizes)
                size_candidates.append((metric["objective"], strength, ood_weight))
        _, strength, ood_weight = max(size_candidates)
        sizes, size_meta = _bayesian_sizes(data, train_idx, outer_idx, fit_parent_out, context, strength=strength, ood_weight=ood_weight)
        results.append(_record(data, outer_idx, parent_outer, fold=fold["fold"], ablation="bayesian_uncertainty_ood_size", parent="rational_parent", size_mult=sizes, extras={"size_strength": strength, "ood_weight": ood_weight}))
        neutral_sizes = _bar_neutral_sizes(data, outer_idx, parent_outer, sizes)
        results.append(_record(data, outer_idx, parent_outer, fold=fold["fold"], ablation="bayesian_size_bar_neutral", parent="rational_parent", size_mult=neutral_sizes, extras={"size_strength": strength, "ood_weight": ood_weight, "normalization": "base-size-weighted across capacity-admitted entries at each UTC timestamp"}))

        # Nested cumulative: choose one hierarchy and one output module using inner only, then combine with selected ATR and sizing.
        hierarchy_search = {}
        for hname, hlabels in (("hard_policy_archetype", arch_labels), ("hard_gmm", hard_labels)):
            hscale, _, _ = _fit_group_scales(data, search_idx, parent_search, hlabels)
            hierarchy_search[hname] = (_score(data, inner_idx, parent_search, hscale)[0], hname)
        sscale, _, _ = _fit_soft_scales(data, search_idx, parent_search, posteriors)
        hierarchy_search["soft_gmm_posterior"] = (_score(data, inner_idx, parent_search, sscale)[0], "soft_gmm_posterior")
        hierarchy_search["none"] = (_score(data, inner_idx, parent_search, ones)[0], "none")
        chosen_h = max(hierarchy_search.values())[1]
        hscale_final = {"hard_policy_archetype": arch_scale, "hard_gmm": hard_scale, "soft_gmm_posterior": soft_scale, "none": ones}[chosen_h]
        output_candidates = {"none": _score(data, inner_idx, parent_search, ones)[0]}
        for name, scale in context_search_scales.items():
            output_candidates[name] = _score(data, inner_idx, parent_search, scale)[0]
        chosen_output = max(output_candidates, key=output_candidates.get)
        output_scale = ones if chosen_output == "none" else context_scales[chosen_output]
        cumulative_scale = np.clip(hscale_final * output_scale, 0.70, 1.35)
        _, _, cumulative_train = _score(data, train_idx, parent, cumulative_scale, atr_powered)
        _, _, cumulative_outer = _score(data, outer_idx, parent, cumulative_scale, atr_powered)
        cumulative_sizes, cumulative_size_meta = _bayesian_sizes(data, train_idx, outer_idx, cumulative_train, context, strength=strength, ood_weight=ood_weight)
        robustness = _local_context_robustness(
            data, train_idx, parent, cumulative_scale, atr_powered, n=16, seed=args.seed + fold_no * 991,
        )
        results.append(_record(data, outer_idx, cumulative_outer, fold=fold["fold"], ablation="cumulative_nested", parent="rational_parent", size_mult=cumulative_sizes, extras={"selected_hierarchy": chosen_h, "selected_output": chosen_output, "selected_atr_power": power, "size_strength": strength, "ood_weight": ood_weight, **robustness}))
        selections[fold["fold"]] = {
            "parent": parent, "hard_policy_archetype": {"scales": arch_map, "support": arch_support},
            "hard_gmm": {"scales": hard_map, "support": hard_support}, "soft_gmm": {"scales": soft_map, "effective_support": soft_support},
            "context": context_meta, "atr_power": power, "atr_power_inner_search": power_search,
            "size": {**size_meta, "strength": strength, "ood_weight": ood_weight},
            "cumulative": {"hierarchy": chosen_h, "output": chosen_output, "size": cumulative_size_meta},
        }
        pd.DataFrame(results).to_csv(output_dir / "fold_metrics.partial.csv", index=False)
        _write_json(output_dir / "selections.partial.json", selections)
        _write_json(output_dir / "parent_params.partial.json", params_out)

    metrics = pd.DataFrame(results)
    parent_metric = metrics[metrics.ablation == "rational_parent"].set_index("fold")
    summary_rows = []
    for name, group in metrics.groupby("ablation", sort=False):
        vals = group["objective"].to_numpy(dtype=float)
        deltas = [float(row.objective - parent_metric.loc[row.fold, "objective"]) for row in group.itertuples()] if name != "trailing_only_parent" else [np.nan] * len(group)
        summary_rows.append({
            "ablation": name, "folds": len(group), "stable_fold_objective": stable_fold_objective(vals),
            "mean_objective": float(vals.mean()), "mean_delta_vs_rational_parent": float(np.nanmean(deltas)) if np.isfinite(deltas).any() else np.nan,
            "positive_delta_folds": int(np.sum(np.asarray(deltas) > 0)) if np.isfinite(deltas).any() else np.nan,
            "mean_pnl": float(group.net_pnl_bankroll.mean()), "worst_fold_pnl": float(group.net_pnl_bankroll.min()),
            "worst_week": float(group.worst_week.min()), "worst_drawdown": float(group.max_drawdown.min()),
            "total_trades": int(group.n_trades.sum()), "mean_net_return": float(group.mean_net_return.mean()),
            "hit_rate": float(group.hit_rate.mean()), "ordering_violation_rate": float(group.ordering_violation_rate.mean()) if "ordering_violation_rate" in group else np.nan,
            "capital_before_trailing_rate": float(group.capital_before_trailing_rate.mean()) if "capital_before_trailing_rate" in group else np.nan,
            "mean_oos_exposure_ratio": float(group.oos_exposure_ratio.mean()),
            "mean_exposure_normalized_pnl": float(group.exposure_normalized_pnl.mean()),
            "stable_exposure_normalized_objective": stable_fold_objective(group.exposure_normalized_objective.to_numpy(dtype=float)),
        })
    summary = pd.DataFrame(summary_rows).sort_values("stable_fold_objective", ascending=False)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False); summary.to_csv(output_dir / "summary.csv", index=False)
    _write_json(output_dir / "selections.json", selections)
    _write_json(output_dir / "parent_params.json", params_out)
    manifest = {
        "generated_by": "run_simple_policy_1m_contextual_ablation", "elapsed_seconds": time.monotonic() - started,
        "evidence_status": "nested policy-validation OOS; July is not untouched and is excluded from fold selection",
        "parent_contract": "rational and trailing-only refit independently inside every outer training fold",
        "replay_contract": {"timeframe": "1m", "horizon_minutes": 1440, "round_trip_fee": 0.01, "capacity": "8 open / 2 new per bar", "path": path_manifest},
        "atr_contract": "entry-frozen causal ATR audit reused exactly; normalized power preserves x=1 identity",
        "provenance": provenance, "search": {"scale_grid": SCALE_GRID, "beta_grid": BETA_GRID, "atr_power_grid": ATR_POWER_GRID, "scale_quantization": 0.05},
    }
    _write_json(output_dir / "manifest.json", manifest)
    report_cols = ["ablation", "stable_fold_objective", "stable_exposure_normalized_objective", "mean_delta_vs_rational_parent", "positive_delta_folds", "mean_pnl", "mean_exposure_normalized_pnl", "mean_oos_exposure_ratio", "worst_fold_pnl", "worst_week", "worst_drawdown", "total_trades", "mean_net_return", "hit_rate"]
    display = summary[report_cols].copy()
    lines = [
        "# One-minute contextual policy ablation", "",
        "Evidence status: nested policy-validation OOS. July was previously inspected and is not included in selection or headline fold metrics.", "",
        "All exit variants retain the rational-relative family, exact 1m path replay, causal entry-frozen ATR, 1% round-trip fee, embedded spread, and 8-open/2-new capacity. The trailing-only row is an independently refit control.", "",
        "| " + " | ".join(report_cols) + " |", "| " + " | ".join(["---"] * len(report_cols)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in row) + " |")
    lines += ["", "See `fold_metrics.csv`, `selections.json`, and `manifest.json` for fold parameters, supports, train-only scalers, posterior provenance, and intervention details."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
