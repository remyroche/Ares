"""Canonical winner contract for one-minute simple-policy optimisation.

The winner is a total-MFE joint trailing exit with no capital-preservation
layer, followed by train-fitted raw Bayesian sizing.  This module contains the
serialisable sizing state so research refits and later runtime integration can
share one explicit contract.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_contextual import (
    RobustState,
    apply_robust_state,
    beta_binomial_lower_score,
    fit_robust_state,
)


WINNER_POLICY_PATHWAY_ID = "joint_trailing_total_mfe_raw_bayesian_v1"
WINNER_REPLAY_TIMEFRAME = "1m"
WINNER_FORWARD_BARS = 1440
WINNER_BASE_SIZE_POWER = 1.1
WINNER_SIZE_STRENGTH_GRID = (1.5, 3.0, 4.5)
WINNER_OOD_WEIGHT_GRID = (0.0, 0.5, 1.0)
WINNER_SIZE_LOWER = 0.65
WINNER_SIZE_UPPER = 1.20

_EV_COLUMNS = (
    "expected_net_ev_after_1pct_mlp_direct",
    "expected_net_ev_after_1pct",
    "expected_net_ev",
)
_UNCERTAINTY_COLUMNS = (
    "meta_hit_probability_uncertainty_p1mp",
    "meta_hit_probability_uncertainty",
)
_OOD_COLUMNS = ("gmm_ood_score", "cluster_entropy_norm")
_ARCHETYPE_COLUMNS = ("policy_archetype", "archetype_policy_key", "archetype")


def _first_column(rows: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in rows.columns), None)


def _numeric(rows: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None or column not in rows.columns:
        return np.zeros(len(rows), dtype=np.float64)
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)


def _side(rows: pd.DataFrame) -> np.ndarray:
    values = rows.get("side", pd.Series("long", index=rows.index))
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return np.where(numeric.fillna(1.0).to_numpy() > 0.0, "long", "short")
    text = values.astype(str).str.lower().to_numpy()
    return np.where(np.char.startswith(text.astype(str), "short"), "short", "long")


def _archetype(rows: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None:
        return np.repeat("global", len(rows))
    return rows[column].fillna("unknown").astype(str).to_numpy()


def _robust_payload(values: np.ndarray) -> dict[str, float]:
    state = fit_robust_state(values)
    return {"median": float(state.median), "scale": float(state.scale)}


def _apply_robust_payload(values: np.ndarray, payload: Mapping[str, Any]) -> np.ndarray:
    return apply_robust_state(
        values,
        RobustState(float(payload.get("median", 0.0)), float(payload.get("scale", 1.0))),
    )


def fit_raw_bayesian_sizing_state(
    rows: pd.DataFrame,
    net_returns: np.ndarray,
    *,
    strength: float,
    ood_weight: float,
    base_size_power: float = WINNER_BASE_SIZE_POWER,
    lower: float = WINNER_SIZE_LOWER,
    upper: float = WINNER_SIZE_UPPER,
) -> dict[str, Any]:
    """Fit the winner's posterior sizing state using training outcomes only."""
    if len(rows) != len(net_returns):
        raise ValueError("rows/net_returns length mismatch")
    ev_col = _first_column(rows, _EV_COLUMNS)
    uncertainty_col = _first_column(rows, _UNCERTAINTY_COLUMNS)
    ood_cols = [name for name in _OOD_COLUMNS if name in rows.columns]
    archetype_col = _first_column(rows, _ARCHETYPE_COLUMNS)

    ev = _numeric(rows, ev_col)
    uncertainty = _numeric(rows, uncertainty_col)
    ood_components = [_numeric(rows, name) for name in ood_cols]
    ev_state = _robust_payload(ev)
    uncertainty_state = _robust_payload(uncertainty)
    ood_states = {name: _robust_payload(values) for name, values in zip(ood_cols, ood_components)}
    z_ev = _apply_robust_payload(ev, ev_state)
    z_uncertainty = _apply_robust_payload(uncertainty, uncertainty_state)
    z_ood = (
        np.mean(
            [_apply_robust_payload(values, ood_states[name]) for name, values in zip(ood_cols, ood_components)],
            axis=0,
        )
        if ood_cols
        else np.zeros(len(rows), dtype=np.float64)
    )
    quality = z_ev - 0.5 * z_uncertainty - float(ood_weight) * z_ood
    finite_quality = quality[np.isfinite(quality)]
    cuts = (
        np.unique(np.quantile(finite_quality, (0.2, 0.4, 0.6, 0.8)))
        if len(finite_quality)
        else np.array([], dtype=np.float64)
    )
    bins = np.digitize(quality, cuts)
    sides = _side(rows)
    archetypes = _archetype(rows, archetype_col)
    returns = np.asarray(net_returns, dtype=np.float64)
    finite = np.isfinite(returns)
    hit = returns > 0.0

    side_prior: dict[str, float] = {}
    for side_name in ("long", "short"):
        local = (sides == side_name) & finite
        side_prior[side_name] = float((hit[local].sum() + 8.0) / (local.sum() + 16.0))

    cells: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int], float] = {}
    for name in np.unique(archetypes):
        for bin_no in range(len(cuts) + 1):
            local = (archetypes == name) & (bins == bin_no) & finite
            support = int(local.sum())
            if support < 20:
                continue
            score = float(
                beta_binomial_lower_score(
                    np.array([hit[local].sum()]),
                    np.array([support]),
                    prior_success=16.0,
                    prior_failure=16.0,
                    uncertainty_aversion=1.0,
                )[0]
            )
            lookup[(str(name), int(bin_no))] = score
            cells.append({"archetype": str(name), "bin": int(bin_no), "score": score, "support": support})

    signal = np.asarray(
        [lookup.get((str(archetypes[i]), int(bins[i])), side_prior[str(sides[i])]) for i in range(len(rows))],
        dtype=np.float64,
    )
    centered = signal - np.asarray([side_prior[str(value)] for value in sides])
    raw = np.clip(np.exp(float(strength) * centered), float(lower), float(upper))
    rank = pd.to_numeric(rows.get("rank_pct", pd.Series(0.9, index=rows.index)), errors="coerce").fillna(0.9)
    base_size = 0.075 + 0.075 * np.power(np.clip(rank.to_numpy(), 0.0, 1.0), float(base_size_power))
    normalizer = float(np.average(raw, weights=base_size)) if len(raw) else 1.0
    return {
        "policy_id": "raw_bayesian_v1",
        "pathway_id": WINNER_POLICY_PATHWAY_ID,
        "strength": float(strength),
        "ood_weight": float(ood_weight),
        "lower": float(lower),
        "upper": float(upper),
        "base_size_power": float(base_size_power),
        "ev_column": ev_col,
        "uncertainty_column": uncertainty_col,
        "ood_columns": ood_cols,
        "archetype_column": archetype_col,
        "ev_robust_state": ev_state,
        "uncertainty_robust_state": uncertainty_state,
        "ood_robust_states": ood_states,
        "quality_cuts": cuts.tolist(),
        "side_prior": side_prior,
        "cells": cells,
        "train_normalizer": max(normalizer, 1e-9),
        "fit_rows": int(len(rows)),
        "fit_outcome": "positive_net_after_cost_return",
    }


def apply_raw_bayesian_sizing_state(rows: pd.DataFrame, state: Mapping[str, Any]) -> np.ndarray:
    """Apply a frozen state without consulting outcomes or future rows."""
    ev = _numeric(rows, state.get("ev_column"))
    uncertainty = _numeric(rows, state.get("uncertainty_column"))
    z_ev = _apply_robust_payload(ev, state.get("ev_robust_state", {}))
    z_uncertainty = _apply_robust_payload(uncertainty, state.get("uncertainty_robust_state", {}))
    ood_columns = list(state.get("ood_columns", []))
    ood_states = state.get("ood_robust_states", {})
    ood_weight = float(state.get("ood_weight", 0.0))
    z_ood = (
        np.mean(
            [_apply_robust_payload(_numeric(rows, name), ood_states.get(name, {})) for name in ood_columns],
            axis=0,
        )
        if ood_columns and abs(ood_weight) > 1e-12
        else np.zeros(len(rows), dtype=np.float64)
    )
    quality = z_ev - 0.5 * z_uncertainty - ood_weight * z_ood
    bins = np.digitize(quality, np.asarray(state.get("quality_cuts", []), dtype=np.float64))
    sides = _side(rows)
    archetypes = _archetype(rows, state.get("archetype_column"))
    side_prior = {str(k): float(v) for k, v in state.get("side_prior", {}).items()}
    lookup = {
        (str(cell["archetype"]), int(cell["bin"])): float(cell["score"])
        for cell in state.get("cells", [])
    }
    priors = np.asarray([side_prior.get(str(value), 0.5) for value in sides])
    signal = np.asarray(
        [lookup.get((str(archetypes[i]), int(bins[i])), priors[i]) for i in range(len(rows))],
        dtype=np.float64,
    )
    raw = np.exp(float(state.get("strength", 3.0)) * (signal - priors))
    lower = float(state.get("lower", WINNER_SIZE_LOWER))
    upper = float(state.get("upper", WINNER_SIZE_UPPER))
    raw = np.clip(raw, lower, upper)
    return np.clip(raw / max(float(state.get("train_normalizer", 1.0)), 1e-9), lower, upper)
