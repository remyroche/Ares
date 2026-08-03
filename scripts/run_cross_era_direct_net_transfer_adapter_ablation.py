#!/usr/bin/env python3
"""Bounded raw-score transfer ablation for exact 12h direct net EV.

This runner deliberately has no current-label argument.  It selects its
weight profile and the optional side-local correction architecture only from
historical causal OOF rows, serialises a frozen bundle, and leaves scoring of a
future pack to :mod:`score_cross_era_direct_net_transfer_adapter_ablation`.

The score is always a raw common-unit score.  There is intentionally no
isotonic/percentile/z-score mapping in this experiment: a negative raw
within-side rank IC is a rejection signal, never an invitation to remap it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256
from scripts.run_cross_era_tail_payoff_challenger import (
    CURRENT_START,
    Fold,
    _binding,
    _normalise_matrix,
    _safe,
    _top_economics,
    _write_json,
    add_regime_composites,
    chronological_folds,
    feature_arms,
    screen_features,
)


SCHEMA = "cross_era_direct_net_transfer_adapter_ablation_v1"
SIDES = ("long", "short")
TARGETS = ("q25", "q50", "p100", "p200")
WEIGHT_PROFILES = ("uniform", "era_balanced", "era_month_balanced")
SCORE_ARMS = ("parent", "adapter", "reliability", "adapter_reliability")
PARENT_CONFIG: dict[str, Any] = {
    "name": "raw_context_shallow_24_fixed",
    "feature_count": 24,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 300,
    "reg_lambda": 20.0,
    "n_estimators": 120,
    "learning_rate": 0.045,
}
ADAPTER_MIN_ROWS = 5_000
ADAPTER_CLIP_BPS = 150.0
RELIABILITY_REFERENCE_COVERAGE = 0.75
RELIABILITY_PENALTY_SCALE_BPS = 100.0
RELIABILITY_PENALTY_CAP_BPS = 75.0
RELIABILITY_MAX_ITER = 1_000
CURRENT_ERA = "2026_may_jul19"

# These are causal raw state fields or explicit causal composites.  The
# correction layers intentionally cannot consume unstable candidate geometry,
# posterior/DAE/GMM summaries, labels or event-path values.
CORRECTION_STATE_COLUMNS = (
    "regime_transition_entropy_48h",
    "regime_transition_entropy_12h",
    "regime_stability_24h",
    "market_breadth_24h",
    "negative_breadth_pct",
    "eth_btc_ret_24h",
    "xs_dispersion__amihud_illiq",
    "volatility_of_volatility_48",
    "transition_pressure_z",
    "entropy_acceleration_z",
    "entropy_vov_interaction_z",
)
CORRECTION_CONTEXT_COLUMNS = (
    "base_oof_score",
    "base_rank_pct_timestamp_side",
    "base_score_z_timestamp_side",
)
PROHIBITED_CORRECTION_TOKENS = (
    "candidate_group_size",
    "base_rank_timestamp_side",
    "base_margin_to_candidate_cutoff",
    "gmm",
    "dae",
    "posterior",
    "compact_risk",
)


@dataclass(frozen=True)
class ParentFit:
    features: dict[str, list[str]]
    medians: dict[str, pd.Series]
    models: dict[str, Any]


def _hash(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _month(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["__ts__"], utc=True, errors="raise").dt.strftime("%Y-%m")


def _net_bps(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4


def _assert_identity(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing identity columns: {missing}")
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} contains duplicate identities")


def training_weights(
    frame: pd.DataFrame,
    positions: np.ndarray,
    profile: str,
) -> np.ndarray:
    """Return per-side, train-only domain/month balancing weights.

    The caller supplies only train positions.  Therefore neither validation
    rows nor later calendar cells can change a loss weight.  We normalise each
    side to a mean weight of one, preserving the relative loss scale between
    the two independently fitted side models.
    """

    if profile not in WEIGHT_PROFILES:
        raise ValueError(f"unknown weight profile: {profile}")
    local = frame.iloc[np.asarray(positions, dtype=int)].copy()
    if local.empty:
        return np.empty(0, dtype=float)
    local["__month__"] = _month(local)
    values = np.ones(len(local), dtype=float)
    for side in SIDES:
        mask = local["side_name"].astype(str).eq(side).to_numpy()
        if not mask.any():
            continue
        if profile == "uniform":
            values[mask] = 1.0
            continue
        keys = ["era"] if profile == "era_balanced" else ["era", "__month__"]
        counts = local.loc[mask].groupby(keys, observed=True)["candidate_id"].transform("size")
        raw = 1.0 / counts.to_numpy(float)
        values[mask] = raw / raw.mean()
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise AssertionError("training weights must be finite and strictly positive")
    return values


def weight_cell_diagnostics(
    frame: pd.DataFrame, positions: np.ndarray, profile: str
) -> pd.DataFrame:
    local = frame.iloc[np.asarray(positions, dtype=int)].copy()
    local["weight"] = training_weights(frame, positions, profile)
    local["month"] = _month(local)
    return (
        local.groupby(["side_name", "era", "month"], observed=True)
        .agg(rows=("candidate_id", "size"), weight_mass=("weight", "sum"), mean_weight=("weight", "mean"))
        .reset_index()
        .assign(weight_profile=profile)
    )


def correction_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [*CORRECTION_CONTEXT_COLUMNS, *CORRECTION_STATE_COLUMNS, "q25_net_bps", "q50_net_bps", "p_loss_le_100", "p_loss_le_200"]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"correction inputs unavailable: {missing}")
    forbidden = [name for name in columns if any(token in name.lower() for token in PROHIBITED_CORRECTION_TOKENS)]
    if forbidden:
        raise AssertionError(f"prohibited correction inputs: {forbidden}")
    return columns


def add_corrected_transition_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only transformed-space transition interactions for correction heads.

    The cross-era fields are already transformed/winsorised values rather than
    physical probabilities.  In particular ``regime_stability_24h`` is not in
    [0, 1], so the legacy literal ``entropy * (1 - clip(stability))`` is not a
    meaningful instability measure here.  The legacy composites remain in the
    parent-lineage helper solely for exact historical comparability; correction
    heads receive these explicitly transformed-space terms instead.
    """

    result = frame.copy()
    required = {
        "regime_transition_entropy_12h", "regime_transition_entropy_48h",
        "regime_stability_24h", "volatility_of_volatility_48",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"corrected transition inputs unavailable: {missing}")
    entropy12 = pd.to_numeric(result["regime_transition_entropy_12h"], errors="coerce")
    entropy48 = pd.to_numeric(result["regime_transition_entropy_48h"], errors="coerce")
    stability24 = pd.to_numeric(result["regime_stability_24h"], errors="coerce")
    vov = pd.to_numeric(result["volatility_of_volatility_48"], errors="coerce")
    result["transition_pressure_z"] = entropy48 - stability24
    result["entropy_acceleration_z"] = entropy12 - entropy48
    result["entropy_vov_interaction_z"] = entropy48 * vov
    return result


def _fit_quantile(
    matrix: pd.DataFrame, target: np.ndarray, weights: np.ndarray, alpha: float, seed: int
) -> Any:
    if len(target) < 120:
        return float(np.quantile(target, alpha))
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=float(alpha), n_estimators=PARENT_CONFIG["n_estimators"],
        learning_rate=PARENT_CONFIG["learning_rate"], num_leaves=PARENT_CONFIG["num_leaves"],
        max_depth=PARENT_CONFIG["max_depth"], min_child_samples=PARENT_CONFIG["min_child_samples"],
        reg_lambda=PARENT_CONFIG["reg_lambda"], colsample_bytree=.8, subsample=.85, subsample_freq=1,
        random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(matrix, target, sample_weight=weights)
    return model


def _fit_binary(matrix: pd.DataFrame, target: np.ndarray, weights: np.ndarray, seed: int) -> Any:
    if np.unique(target).size < 2:
        return float(np.mean(target))
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=PARENT_CONFIG["n_estimators"], learning_rate=PARENT_CONFIG["learning_rate"],
        num_leaves=PARENT_CONFIG["num_leaves"], max_depth=PARENT_CONFIG["max_depth"],
        min_child_samples=PARENT_CONFIG["min_child_samples"], reg_lambda=PARENT_CONFIG["reg_lambda"],
        colsample_bytree=.8, subsample=.85, subsample_freq=1, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(matrix, target.astype(int), sample_weight=weights)
    return model


def _predict(model: Any, matrix: pd.DataFrame, binary: bool) -> np.ndarray:
    if isinstance(model, (float, int, np.floating)):
        return np.full(len(matrix), float(model), dtype=float)
    if binary:
        return np.clip(np.asarray(model.predict_proba(matrix)[:, 1], dtype=float), 1e-6, 1.0 - 1e-6)
    return np.asarray(model.predict(matrix), dtype=float)


def fit_parent(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    positions: np.ndarray,
    *,
    profile: str,
    seed: int,
) -> dict[str, ParentFit]:
    """Fit fixed-capacity parent heads separately per side on eligible rows."""

    target = _net_bps(frame)
    weights = training_weights(frame, positions, profile)
    output: dict[str, ParentFit] = {}
    for side_index, side in enumerate(SIDES):
        local_mask = frame.iloc[positions]["side_name"].astype(str).eq(side).to_numpy()
        local_pos = np.asarray(positions, dtype=int)[local_mask]
        local_w = weights[local_mask]
        if len(local_pos) < ADAPTER_MIN_ROWS:
            raise ValueError(f"insufficient eligible parent support for {side}: {len(local_pos)}")
        features: dict[str, list[str]] = {}
        medians: dict[str, pd.Series] = {}
        models: dict[str, Any] = {}
        targets: dict[str, tuple[np.ndarray, bool, float | None]] = {
            "q25": (target, False, .25),
            "q50": (target, False, .50),
            "p100": ((target <= -100.0).astype(np.int8), True, None),
            "p200": ((target <= -200.0).astype(np.int8), True, None),
        }
        for target_index, (name, (values, binary, alpha)) in enumerate(targets.items()):
            chosen = screen_features(matrix, np.asarray(values, dtype=float), local_pos, int(PARENT_CONFIG["feature_count"]), multiclass=False)
            median = matrix.iloc[local_pos][chosen].median().fillna(0.0)
            fit_x = matrix.iloc[local_pos][chosen].fillna(median)
            model = (
                _fit_binary(fit_x, np.asarray(values)[local_pos], local_w, seed + side_index * 100 + target_index)
                if binary else _fit_quantile(fit_x, np.asarray(values)[local_pos], local_w, float(alpha), seed + side_index * 100 + target_index)
            )
            features[name], medians[name], models[name] = chosen, median, model
        output[side] = ParentFit(features=features, medians=medians, models=models)
    return output


def score_parent(frame: pd.DataFrame, matrix: pd.DataFrame, bundle: Mapping[str, ParentFit]) -> pd.DataFrame:
    output = frame.loc[:, list(IDENTITY)].copy()
    for side in SIDES:
        pos = np.flatnonzero(frame["side_name"].astype(str).eq(side).to_numpy())
        fit = bundle[side]
        features = fit.features if isinstance(fit, ParentFit) else fit["features"]
        medians = fit.medians if isinstance(fit, ParentFit) else fit["medians"]
        models = fit.models if isinstance(fit, ParentFit) else fit["models"]
        for name in TARGETS:
            binary = name.startswith("p")
            value = _predict(
                models[name],
                matrix.iloc[pos][features[name]].fillna(medians[name]),
                binary,
            )
            column = {"q25": "q25_net_bps", "q50": "q50_net_bps", "p100": "p_loss_le_100", "p200": "p_loss_le_200"}[name]
            output.loc[output.index[pos], column] = value
    return output


def inner_parent_splits(frame: pd.DataFrame, positions: np.ndarray, *, blocks: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strictly causal inner splits for OOF parent predictions used by adapters."""

    if blocks < 2:
        raise ValueError("inner parent cross-fit requires at least two blocks")
    local = frame.iloc[np.asarray(positions, dtype=int)].copy()
    local["__position__"] = np.asarray(positions, dtype=int)
    local = local.sort_values(["__ts__", "candidate_id"], kind="stable")
    chunks = [chunk for chunk in np.array_split(local, blocks) if len(chunk)]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in chunks[1:]:
        start = pd.to_datetime(chunk["__ts__"], utc=True, errors="raise").min()
        train = local.loc[
            pd.to_datetime(local["__ts__"], utc=True, errors="raise").lt(start)
            & pd.to_datetime(local["label_resolution_utc"], utc=True, errors="raise").lt(start),
            "__position__",
        ].to_numpy(int)
        valid = chunk["__position__"].to_numpy(int)
        if len(train):
            result.append((train, valid))
    return result


def crossfit_parent_predictions(
    frame: pd.DataFrame, matrix: pd.DataFrame, positions: np.ndarray, *, profile: str, seed: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return only parent predictions that were not fitted on their own rows."""

    parts: list[pd.DataFrame] = []
    ledger: list[dict[str, Any]] = []
    for split_index, (train, valid) in enumerate(inner_parent_splits(frame, positions)):
        # Both-side support is required because the final correction architecture
        # is side-local but compared on the same outer-validation population.
        if any((frame.iloc[train]["side_name"].astype(str) == side).sum() < ADAPTER_MIN_ROWS for side in SIDES):
            continue
        parent = fit_parent(frame, matrix, train, profile=profile, seed=seed + split_index * 1_000)
        predicted = score_parent(frame.iloc[valid].reset_index(drop=True), matrix.iloc[valid].reset_index(drop=True), parent)
        predicted["__source_position__"] = valid
        parts.append(predicted)
        start = pd.to_datetime(frame.iloc[valid]["__ts__"], utc=True).min()
        if not bool((pd.to_datetime(frame.iloc[train]["label_resolution_utc"], utc=True) < start).all()):
            raise AssertionError("cross-fitted parent chronology violated")
        ledger.append({"inner_split": split_index, "train_rows": int(len(train)), "validation_rows": int(len(valid)), "validation_start": str(start), "max_train_resolution": str(pd.to_datetime(frame.iloc[train]["label_resolution_utc"], utc=True).max())})
    if not parts:
        return pd.DataFrame(columns=[*IDENTITY, "q25_net_bps", "q50_net_bps", "p_loss_le_100", "p_loss_le_200", "__source_position__"]), ledger
    return pd.concat(parts, ignore_index=True), ledger


def _matrix_from_columns(frame: pd.DataFrame, columns: Sequence[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fitted = values.median().fillna(0.0) if medians is None else medians.reindex(columns).fillna(0.0)
    return values.fillna(fitted), fitted


def fit_reliability(
    matrix: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, int]:
    """Fit a scaled, regularised coverage model and fail closed on convergence."""

    reliability = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=.10,
            max_iter=RELIABILITY_MAX_ITER,
            tol=1e-6,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    reliability.fit(
        matrix,
        target,
        logisticregression__sample_weight=weights,
    )
    iterations = int(reliability.named_steps["logisticregression"].n_iter_.max())
    if iterations >= RELIABILITY_MAX_ITER:
        raise RuntimeError(
            f"reliability head failed to converge: iterations={iterations}"
        )
    return reliability, iterations


def fit_corrections(
    parent_oof: pd.DataFrame, source: pd.DataFrame, *, profile: str, seed: int
) -> dict[str, Any]:
    """Fit fixed, side-local residual and q25-coverage heads from OOF parents."""

    if parent_oof.empty:
        return {"sides": {side: {"status": "zero_fallback", "reason": "no_crossfitted_parent_rows"} for side in SIDES}}
    work = parent_oof.merge(source.loc[:, [*IDENTITY, "execution_net_ev_12h", "era", "label_resolution_utc", *CORRECTION_CONTEXT_COLUMNS, *CORRECTION_STATE_COLUMNS]], on=list(IDENTITY), how="inner", validate="one_to_one")
    columns = correction_feature_columns(work)
    # OOF rows are a strict subset of source; recompute their training-only
    # balancing weights rather than borrowing a full-period normalization.
    lookup = pd.Series(source.index.to_numpy(int), index=pd.MultiIndex.from_frame(source.loc[:, list(IDENTITY)]))
    positions = lookup.reindex(pd.MultiIndex.from_frame(work.loc[:, list(IDENTITY)])).to_numpy()
    if pd.isna(positions).any():
        raise AssertionError("cross-fitted parent identities are absent from correction source")
    positions = positions.astype(int)
    full_weights = training_weights(source, positions, profile)
    work["__weight__"] = full_weights
    state: dict[str, Any] = {"columns": columns, "sides": {}}
    for side_index, side in enumerate(SIDES):
        local = work.loc[work["side_name"].astype(str).eq(side)].copy()
        if len(local) < ADAPTER_MIN_ROWS:
            state["sides"][side] = {"status": "zero_fallback", "reason": "insufficient_crossfitted_parent_rows", "rows": int(len(local))}
            continue
        x, medians = _matrix_from_columns(local, columns)
        net = _net_bps(local)
        residual = np.clip(net - local["q25_net_bps"].to_numpy(float), -600.0, 600.0)
        from catboost import CatBoostRegressor
        adapter = CatBoostRegressor(loss_function="RMSE", iterations=120, learning_rate=.035, depth=3, l2_leaf_reg=50.0, random_strength=.5, bagging_temperature=1.0, bootstrap_type="Bayesian", random_seed=seed + side_index, thread_count=4, verbose=False, allow_writing_files=False)
        adapter.fit(x, residual, sample_weight=local["__weight__"].to_numpy(float))
        coverage = (net >= local["q25_net_bps"].to_numpy(float)).astype(int)
        reliability, iterations = fit_reliability(
            x,
            coverage,
            local["__weight__"].to_numpy(float),
            seed=seed + 100 + side_index,
        )
        state["sides"][side] = {
            "status": "fit_crossfitted_parent_rows",
            "rows": int(len(local)),
            "medians": medians,
            "adapter": adapter,
            "reliability": reliability,
            "reliability_iterations": iterations,
            "reliability_converged": True,
            "coverage_prevalence": float(coverage.mean()),
        }
    return state


def apply_corrections(parent: pd.DataFrame, features: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    result = parent.copy()
    joined = parent.merge(features.loc[:, [*IDENTITY, *CORRECTION_CONTEXT_COLUMNS, *CORRECTION_STATE_COLUMNS]], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(joined) != len(parent):
        raise ValueError("correction features do not cover parent scores one-to-one")
    columns = list(state["columns"])
    delta = np.zeros(len(joined), dtype=float)
    penalty = np.zeros(len(joined), dtype=float)
    for side in SIDES:
        mask = joined["side_name"].astype(str).eq(side).to_numpy()
        record = state["sides"].get(side, {})
        if not mask.any() or record.get("status") != "fit_crossfitted_parent_rows":
            continue
        x, _ = _matrix_from_columns(joined.loc[mask], columns, record["medians"])
        delta[mask] = np.clip(np.asarray(record["adapter"].predict(x), dtype=float), -ADAPTER_CLIP_BPS, ADAPTER_CLIP_BPS)
        probability = np.asarray(record["reliability"].predict_proba(x)[:, 1], dtype=float)
        penalty[mask] = np.clip(RELIABILITY_PENALTY_SCALE_BPS * (RELIABILITY_REFERENCE_COVERAGE - probability), 0.0, RELIABILITY_PENALTY_CAP_BPS)
    joined["adapter_delta_bps"] = delta
    joined["reliability_penalty_bps"] = penalty
    joined["score_parent_bps"] = joined["q25_net_bps"]
    joined["score_adapter_bps"] = joined["q25_net_bps"] + delta
    joined["score_reliability_bps"] = joined["q25_net_bps"] - penalty
    joined["score_adapter_reliability_bps"] = joined["q25_net_bps"] + delta - penalty
    return joined


def raw_tail_metrics(frame: pd.DataFrame, score_column: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Raw globally-ranked economics plus the mandatory side/month IC ledger."""

    scored = frame.copy()
    scored["execution_net_ev_12h"] = pd.to_numeric(scored["execution_net_ev_12h"], errors="raise")
    metrics, monthly = _top_economics(scored, score_column)
    rows: list[dict[str, Any]] = []
    scored["month"] = _month(scored)
    for side in SIDES:
        local_side = scored.loc[scored["side_name"].astype(str).eq(side)]
        for period, local in [("all", local_side), *((month, group) for month, group in local_side.groupby("month", sort=True))]:
            rows.append({"side_name": side, "period": str(period), "rows": int(len(local)), "raw_rank_ic": float(local[score_column].corr(local["execution_net_ev_12h"], method="spearman")) if len(local) >= 3 else np.nan, "mapping_eligible": bool(len(local) >= 3 and local[score_column].corr(local["execution_net_ev_12h"], method="spearman") >= 0.0)})
    return metrics, pd.DataFrame(rows)


def severe_calibration_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate p100/p200 calibration without using it as an EV surrogate."""

    net = _net_bps(frame)
    briers: list[float] = []
    eces: list[float] = []
    for threshold, column in ((100.0, "p_loss_le_100"), (200.0, "p_loss_le_200")):
        target = (net <= -threshold).astype(float)
        probability = np.clip(
            pd.to_numeric(frame[column], errors="raise").to_numpy(float),
            1e-6,
            1.0 - 1e-6,
        )
        briers.append(float(np.mean(np.square(probability - target))))
        bins = np.minimum((probability * 10.0).astype(int), 9)
        ece = 0.0
        for index in range(10):
            mask = bins == index
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(probability[mask].mean()) - float(target[mask].mean())
                )
        eces.append(float(ece))
    return {
        "mean_severe_brier": float(np.mean(briers)),
        "mean_severe_ece10": float(np.mean(eces)),
    }


def select_weight_profile(records: pd.DataFrame) -> str:
    """Choose by economics among IC/month-coverage eligible profiles.

    IC is a necessary raw-score diagnostic, not the optimisation objective:
    the paired IC/EV workstream established that higher IC alone can still
    produce a worse global traded tail.  If no profile passes the diagnostic
    gates we still retain one research-only economic winner, explicitly marked
    in the table, and mapping/promotion remain forbidden.
    """

    required = {
        "weight_profile",
        "min_side_ic",
        "min_latest_domain_ic",
        "month_coverage_complete",
        "calibration_no_worse_than_uniform",
        "global_top10_net_ev_bps",
        "worst_month_top10_net_ev_bps",
        "global_top10_cvar05_bps",
    }
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"weight selection records missing: {sorted(missing)}")
    eligible = records.loc[
        records["month_coverage_complete"].astype(bool)
        & records["calibration_no_worse_than_uniform"].astype(bool)
        & records["min_side_ic"].ge(0.0)
        & records["min_latest_domain_ic"].ge(0.0)
    ]
    pool = eligible if len(eligible) else records
    ordered = pool.sort_values(["global_top10_net_ev_bps", "worst_month_top10_net_ev_bps", "global_top10_cvar05_bps", "min_side_ic", "weight_profile"], ascending=[False, False, False, False, True], kind="stable")
    return str(ordered.iloc[0]["weight_profile"])


def _oof_parent(frame: pd.DataFrame, matrix: pd.DataFrame, profile: str, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    ledger: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(chronological_folds(frame)):
        parent = fit_parent(frame, matrix, fold.train, profile=profile, seed=seed + fold_index * 10_000)
        valid_frame = frame.iloc[fold.valid].reset_index(drop=True)
        prediction = score_parent(valid_frame, matrix.iloc[fold.valid].reset_index(drop=True), parent)
        prediction = prediction.merge(valid_frame.loc[:, [*IDENTITY, "execution_net_ev_12h", "era", "label_resolution_utc", *CORRECTION_CONTEXT_COLUMNS, *CORRECTION_STATE_COLUMNS]], on=list(IDENTITY), how="inner", validate="one_to_one")
        prediction["fold"] = fold.name
        parts.append(prediction)
        ledger.append({"fold": fold.name, "train_rows": int(len(fold.train)), "validation_rows": int(len(fold.valid)), "validation_start": str(fold.start), "max_train_resolution": str(pd.to_datetime(frame.iloc[fold.train]["label_resolution_utc"], utc=True).max()), "profile": profile})
    return pd.concat(parts, ignore_index=True), ledger


def _weight_stage(frame: pd.DataFrame, matrix: pd.DataFrame, seed: int) -> tuple[str, pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    oof_by_profile: dict[str, pd.DataFrame] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for profile in WEIGHT_PROFILES:
        oof, ledger = _oof_parent(frame, matrix, profile, seed)
        oof["score_parent_bps"] = oof["q25_net_bps"]
        economics, ic = raw_tail_metrics(oof, "score_parent_bps")
        latest_domain = ic.loc[ic["period"].eq("2026-07")]
        monthly = _top_economics(oof, "score_parent_bps")[1]
        expected_months = {"2025-03", "2025-04", "2026-05", "2026-06", "2026-07"}
        observed_months = set(monthly["month"].astype(str))
        records.append({
            "weight_profile": profile,
            "min_side_ic": float(ic.loc[ic["period"].eq("all"), "raw_rank_ic"].min()),
            "min_latest_domain_ic": float(latest_domain["raw_rank_ic"].min()),
            "month_coverage_complete": bool(expected_months.issubset(observed_months)),
            "raw_ic_gate_passed": bool(
                ic.loc[ic["period"].eq("all"), "mapping_eligible"].all()
                and latest_domain["mapping_eligible"].all()
            ),
            **severe_calibration_summary(oof),
            **economics,
        })
        oof_by_profile[profile], ledgers[profile] = oof, ledger
    table = pd.DataFrame(records)
    uniform = table.loc[table["weight_profile"].eq("uniform")].iloc[0]
    table["calibration_no_worse_than_uniform"] = (
        table["mean_severe_brier"].le(float(uniform["mean_severe_brier"]) + 1e-12)
        & table["mean_severe_ece10"].le(float(uniform["mean_severe_ece10"]) + 1e-12)
    )
    return select_weight_profile(table), table, oof_by_profile, ledgers


def _oof_correction_scores(frame: pd.DataFrame, matrix: pd.DataFrame, profile: str, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    ledger: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(chronological_folds(frame)):
        parent = fit_parent(frame, matrix, fold.train, profile=profile, seed=seed + fold_index * 30_000)
        valid_frame = frame.iloc[fold.valid].reset_index(drop=True)
        parent_valid = score_parent(valid_frame, matrix.iloc[fold.valid].reset_index(drop=True), parent)
        parent_valid = parent_valid.merge(valid_frame.loc[:, [*IDENTITY, "execution_net_ev_12h", "era", "label_resolution_utc", *CORRECTION_CONTEXT_COLUMNS, *CORRECTION_STATE_COLUMNS]], on=list(IDENTITY), how="inner", validate="one_to_one")
        inner_oof, inner_ledger = crossfit_parent_predictions(frame, matrix, fold.train, profile=profile, seed=seed + fold_index * 30_000 + 10_000)
        corrections = fit_corrections(inner_oof, frame, profile=profile, seed=seed + fold_index * 30_000 + 20_000)
        scored = apply_corrections(parent_valid.loc[:, [*IDENTITY, "q25_net_bps", "q50_net_bps", "p_loss_le_100", "p_loss_le_200"]], parent_valid, corrections)
        scored["execution_net_ev_12h"] = parent_valid["execution_net_ev_12h"].to_numpy(float)
        scored["era"] = parent_valid["era"].to_numpy()
        scored["label_resolution_utc"] = parent_valid["label_resolution_utc"].to_numpy()
        scored["fold"] = fold.name
        parts.append(scored)
        ledger.append({"fold": fold.name, "inner_parent_crossfit": inner_ledger, "correction_state": {side: {key: value for key, value in record.items() if key not in {"adapter", "reliability", "medians"}} for side, record in corrections["sides"].items()}})
    return pd.concat(parts, ignore_index=True), ledger


def _final_bundle(frame: pd.DataFrame, matrix: pd.DataFrame, profile: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = np.arange(len(frame), dtype=int)
    parent = fit_parent(frame, matrix, positions, profile=profile, seed=seed)
    inner_oof, ledger = crossfit_parent_predictions(frame, matrix, positions, profile=profile, seed=seed + 100_000)
    corrections = fit_corrections(inner_oof, frame, profile=profile, seed=seed + 200_000)
    serializable_parent = {
        side: {
            "features": fit.features,
            "medians": fit.medians,
            "models": fit.models,
        }
        for side, fit in parent.items()
    }
    return {"schema": SCHEMA, "parent_columns": list(matrix.columns), "parent": serializable_parent, "corrections": corrections, "weight_profile": profile}, {"crossfit_ledger": ledger, "correction_status": {side: {key: value for key, value in record.items() if key not in {"adapter", "reliability", "medians"}} for side, record in corrections["sides"].items()}}


def _domain_transfer_card(frame: pd.DataFrame, matrix: pd.DataFrame, profile: str, seed: int) -> pd.DataFrame:
    old = np.flatnonzero(frame["era"].astype(str).eq("2025_feb_apr").to_numpy())
    recent = np.flatnonzero(frame["era"].astype(str).eq(CURRENT_ERA).to_numpy())
    rows: list[dict[str, Any]] = []
    for direction_index, (train, evaluate, name, causal) in enumerate((
        (old, recent, "old_to_recent_forward", True),
        (recent, old, "recent_to_old_reverse_diagnostic", False),
    )):
        local_seed = seed + direction_index * 500_000
        parent = fit_parent(frame, matrix, train, profile=profile, seed=local_seed)
        evaluate_frame = frame.iloc[evaluate].reset_index(drop=True)
        parent_scored = score_parent(
            evaluate_frame,
            matrix.iloc[evaluate].reset_index(drop=True),
            parent,
        )
        parent_oof, _ = crossfit_parent_predictions(
            frame,
            matrix,
            train,
            profile=profile,
            seed=local_seed + 100_000,
        )
        corrections = fit_corrections(
            parent_oof,
            frame,
            profile=profile,
            seed=local_seed + 200_000,
        )
        scored = apply_corrections(parent_scored, evaluate_frame, corrections)
        scored["execution_net_ev_12h"] = evaluate_frame["execution_net_ev_12h"].to_numpy(float)
        for arm in SCORE_ARMS:
            score_column = f"score_{arm}_bps"
            metrics, ic = raw_tail_metrics(scored, score_column)
            rows.append({
                **metrics,
                "transfer": name,
                "causal_policy_evidence": causal,
                "arm": arm,
                "min_raw_side_month_ic": float(
                    ic.loc[ic["period"].ne("all"), "raw_rank_ic"].min()
                ),
                "raw_ic_gate_passed": bool(ic["mapping_eligible"].all()),
            })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    dataset_path = args.dataset_dir / "cross_era_tail_payoff_dataset.parquet"
    if sha256(dataset_path) != manifest["outputs"]["dataset"]["sha256"]:
        raise ValueError("dataset hash mismatch")
    contract_path = args.dataset_dir / "feature_contract.json"
    contract = json.loads(contract_path.read_text())
    history = pd.read_parquet(dataset_path)
    _assert_identity(history, "historical dataset")
    history["__ts__"] = pd.to_datetime(history["__ts__"], utc=True, errors="raise")
    history["label_resolution_utc"] = pd.to_datetime(history["label_resolution_utc"], utc=True, errors="raise")
    history = history.loc[history["label_resolution_utc"].lt(CURRENT_START)].reset_index(drop=True)
    history, composites = add_regime_composites(history)
    history = add_corrected_transition_inputs(history)
    arms = feature_arms(contract, composites)
    columns = arms["raw_context"]
    matrix = _normalise_matrix(history, columns)
    selected_profile, weight_table, _, weight_ledgers = _weight_stage(history, matrix, args.seed)
    oof, correction_ledger = _oof_correction_scores(history, matrix, selected_profile, args.seed + 500_000)
    selection_rows: list[dict[str, Any]] = []
    ic_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    for arm in SCORE_ARMS:
        column = f"score_{arm}_bps"
        economics, ic = raw_tail_metrics(oof, column)
        ic["arm"] = arm
        ic_parts.append(ic)
        monthly = _top_economics(oof, column)[1].assign(arm=arm)
        monthly_parts.append(monthly)
        latest_domain = ic.loc[ic["period"].eq("2026-07")]
        monthly = _top_economics(oof, column)[1]
        expected_months = {"2025-03", "2025-04", "2026-05", "2026-06", "2026-07"}
        observed_months = set(monthly["month"].astype(str))
        selection_rows.append({"arm": arm, "weight_profile": selected_profile, "score_column": column, "min_side_ic": float(ic.loc[ic["period"].eq("all"), "raw_rank_ic"].min()), "min_latest_domain_ic": float(latest_domain["raw_rank_ic"].min()), "month_coverage_complete": bool(expected_months.issubset(observed_months)), "raw_ic_gate_passed": bool(ic.loc[ic["period"].eq("all"), "mapping_eligible"].all() and latest_domain["mapping_eligible"].all()), **economics})
    selection = pd.DataFrame(selection_rows)
    eligible_architecture = selection.loc[selection["month_coverage_complete"].astype(bool) & selection["raw_ic_gate_passed"].astype(bool)]
    selection_pool = eligible_architecture if len(eligible_architecture) else selection
    ordered_selection = selection_pool.sort_values(["global_top10_net_ev_bps", "worst_month_top10_net_ev_bps", "global_top10_cvar05_bps", "min_side_ic", "arm"], ascending=[False, False, False, False, True], kind="stable")
    selection["selection_eligible"] = selection.index.isin(eligible_architecture.index)
    selection["selected_research_only"] = selection.index == ordered_selection.index[0]
    winner = selection.loc[selection["selected_research_only"]].iloc[0].to_dict()
    final_bundle, final_state = _final_bundle(history, matrix, selected_profile, args.seed + 900_000)
    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "frozen_models.joblib"
    joblib.dump(final_bundle, model_path)
    outputs: dict[str, Any] = {}
    for name, table, suffix in (
        ("weight_selection", weight_table, ".csv"),
        ("historical_oof_all_arms", oof, ".parquet"),
        ("raw_ic_by_side_month", pd.concat(ic_parts, ignore_index=True), ".csv"),
        ("historical_monthly_economics", pd.concat(monthly_parts, ignore_index=True), ".csv"),
        ("architecture_selection", selection.sort_values(["selected_research_only", "global_top10_net_ev_bps"], ascending=[False, False], kind="stable"), ".csv"),
        ("old_to_recent_transfer", _domain_transfer_card(history, matrix, selected_profile, args.seed + 950_000), ".csv"),
    ):
        path = args.output_dir / f"{name}{suffix}"
        table.to_parquet(path, index=False) if suffix == ".parquet" else table.to_csv(path, index=False)
        outputs[name] = {**_hash(path), "rows": int(len(table))}
    ledger_path = args.output_dir / "causal_split_ledger.json"
    _write_json(ledger_path, {"weight_stage": weight_ledgers, "correction_stage": correction_ledger, "strict_rule": "all parent, cross-fitted parent, correction and reliability fits require label_resolution_utc < their validation start"})
    outputs["causal_split_ledger"] = _hash(ledger_path)
    selected_weight_record = weight_table.loc[weight_table["weight_profile"].eq(selected_profile)].iloc[0].to_dict()
    frozen = {"schema": SCHEMA, "status": "frozen_before_current_evaluation", "selection_status": "historical_causal_oof_raw_score_only", "current_outcomes_used_for_selection": False, "dataset": _binding(dataset_path), "dataset_manifest": _binding(args.dataset_dir / "manifest.json"), "feature_contract": _binding(contract_path), "parent": PARENT_CONFIG, "weight_profiles_tested": list(WEIGHT_PROFILES), "selected_weight_profile": selected_profile, "selected_weight_raw_ic_gate_passed": bool(selected_weight_record["raw_ic_gate_passed"]), "selected_weight_calibration_no_worse_than_uniform": bool(selected_weight_record["calibration_no_worse_than_uniform"]), "score_arms": list(SCORE_ARMS), "winner": winner, "winner_raw_ic_gate_passed": bool(winner["raw_ic_gate_passed"]), "mapping": {"enabled": False, "rule": "no_mapping_rescue_when_raw_within_side_ic_negative"}, "global_selection": {"fraction": .10, "scope": "single pooled global book", "secondary_order": "candidate_id ascending"}, "correction_contract": {"adapter": "side-local CatBoost on cross-fitted q25 residual, zero fallback under support", "reliability": "side-local q25-coverage logistic penalty", "transition_inputs": "transformed-space pressure, entropy acceleration and entropy-vov interaction; no legacy probability-style instability composite", "probability_heads": "p100/p200 are parent outputs and remain invariant across correction arms", "prohibited_inputs": list(PROHIBITED_CORRECTION_TOKENS)}, "model": _hash(model_path), "final_state": final_state, "outputs": outputs}
    frozen_path = args.output_dir / "frozen_before_current_evaluation.json"
    _write_json(frozen_path, frozen)
    outputs["frozen"] = _hash(frozen_path)
    report = {"schema": SCHEMA, "status": "completed_historical_selection_current_not_scored", "promotion_eligible": False, "search_breadth": {"weight_profiles": len(WEIGHT_PROFILES), "architecture_arms": len(SCORE_ARMS), "continuous_hpo_trials": 0, "parent_capacity": PARENT_CONFIG["name"]}, "outputs": outputs, "frozen": _hash(frozen_path)}
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    _write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "report": _hash(report_path), "outputs": outputs})
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"))
    value.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_direct_net_transfer_adapter_ablation_20260730_v2"))
    value.add_argument("--seed", type=int, default=42)
    return value


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
