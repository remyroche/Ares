#!/usr/bin/env python3
"""Evaluate strict-OOF Router scores under the frozen Router utility contract.

This evaluator is intentionally *post-score only*.  It reads immutable
target-free Router predictions, performs the production timestamp-local route
with deterministic candidate-id tie breaking, and only then joins resolved
rich-policy outcomes.  It therefore provides a common, causally safe metric
surface for the Router control and every later challenger.

The primary contract is fixed by the Router research specification:

    utility is supplied on the command line.  The historical p=.75/cap=225
    contract remains available, while the optimized Router50 feature-search
    anchor uses p=.50/cap=300 and the weekly stability selector.
    R50_utility = opportunity-weighted timestamp-macro utility recall
    S_router = .70 * R50_utility + .15 * R50_count + .15 * R100_count
    The receipt reports both fold stability and the optimized feature-search
    weekly stability selector (robust weekly mean + 0.5 × lower-tail mean).

All other utility geometries are a robustness envelope only.  No score is
refit, calibrated, or selected by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_router_utility_contract_v1"
# Default to the active Router50 selection contract.  The p=.50/cap=300
# diagnostic remains reproducible by passing explicit CLI values.
PRIMARY_POWER = 0.75
PRIMARY_CAP_BPS = 225.0
PRIMARY_GAMMA = 0.5
PRIMARY_TIMESTAMP_CAP = 2.0
HURDLE_BPS = 50.0
ROUTE_FRACTIONS = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
SENSITIVITY_POWERS = (0.50, 0.75, 1.00)
SENSITIVITY_CAPS = (150.0, 225.0, 300.0)
SENSITIVITY_GAMMAS = (0.0, 0.5, 1.0)
SENSITIVITY_TIMESTAMP_CAPS = (1.5, 2.0, 3.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_contract_months(source: Path) -> tuple[str, ...]:
    contract_path = source / "run_contract.json"
    payload = json.loads(contract_path.read_text())
    months = tuple(str(month) for month in payload.get("months", ()))
    if not months:
        raise AssertionError(f"{contract_path}: no immutable month sequence")
    if tuple(sorted(months)) != months or len(set(months)) != len(months):
        raise AssertionError("source month sequence is not unique and chronological")
    return months


def _validate_target_free(score: pd.DataFrame, score_column: str, source: Path) -> None:
    required = {"candidate_id", "__decision_ts__", score_column}
    missing = required - set(score.columns)
    if missing:
        raise AssertionError(f"{source}: missing target-free score columns {sorted(missing)}")
    forbidden = [
        column for column in score.columns
        if any(token in column.lower() for token in ("policy_", "label", "outcome", "net_bps", "gross_bps", "path_valid"))
    ]
    if forbidden:
        raise AssertionError(f"{source}: target-free score receipt contains outcome-like fields {forbidden}")
    if score["candidate_id"].duplicated().any():
        raise AssertionError(f"{source}: duplicate candidate identity")
    if score[["candidate_id", "__decision_ts__"]].isna().any().any():
        raise AssertionError(f"{source}: null identity")


def _utc_month(frame: pd.DataFrame) -> pd.Series:
    stamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return stamp.dt.strftime("%Y-%m")


def _local_route(frame: pd.DataFrame, score_column: str, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return selected mask and zero-based local rank percentile.

    This preserves the incumbent convention exactly: descending finite score,
    ascending candidate identity for ties, and ``ceil(fraction * n_t)``.
    Non-finite scores are never routed even when the nominal budget is larger.
    """
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score_column]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[score_column], errors="coerce").fillna(-np.inf)
    work = work.sort_values(
        ["__decision_ts__", "__score__", "candidate_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(np.int64)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(np.int64)
    selected = (ordinal + 1 <= np.ceil(float(fraction) * size)) & np.isfinite(work["__score__"].to_numpy(float))
    rank_pct = np.divide(ordinal, size, out=np.zeros(len(work), dtype=float), where=size > 0)
    ordered_index = work["__row__"].to_numpy(np.int64)
    return (
        pd.Series(selected, index=ordered_index).reindex(np.arange(len(frame))).to_numpy(bool),
        pd.Series(rank_pct, index=ordered_index).reindex(np.arange(len(frame))).to_numpy(float),
    )


def _utility(net: np.ndarray, valid: np.ndarray, power: float, cap_bps: float) -> np.ndarray:
    excess = np.where(valid, np.maximum(net - HURDLE_BPS, 0.0), 0.0)
    return np.power(np.minimum(excess, cap_bps) / cap_bps, power, dtype=float)


def _timestamp_primary(
    joined: pd.DataFrame,
    score_column: str,
    fraction: float,
    power: float,
    cap_bps: float,
    gamma: float,
    timestamp_cap: float,
) -> pd.DataFrame:
    selected, rank_pct = _local_route(joined, score_column, fraction)
    work = joined.loc[:, ["candidate_id", "__decision_ts__", "__valid__", "__net__"]].copy()
    work["__selected__"] = selected
    work["__rank_pct__"] = rank_pct
    net = work["__net__"].to_numpy(float)
    valid = work["__valid__"].to_numpy(bool)
    utility = _utility(net, valid, power, cap_bps)
    work["__utility__"] = utility
    work["__selected_utility__"] = np.where(selected, utility, 0.0)
    work["__winner50__"] = valid & (net > 50.0)
    work["__winner100__"] = valid & (net > 100.0)
    work["__winner200__"] = valid & (net > 200.0)
    work["__selected_winner50__"] = selected & work["__winner50__"].to_numpy(bool)
    work["__selected_winner100__"] = selected & work["__winner100__"].to_numpy(bool)
    work["__selected_winner200__"] = selected & work["__winner200__"].to_numpy(bool)
    work["__selected_valid__"] = selected & valid
    work["__selected_net__"] = np.where(work["__selected_valid__"], net, 0.0)
    grouped = work.groupby("__decision_ts__", sort=False).agg(
        candidate_rows=("candidate_id", "size"),
        selected_rows=("__selected__", "sum"),
        valid_rows=("__valid__", "sum"),
        selected_valid_rows=("__selected_valid__", "sum"),
        available_utility=("__utility__", "sum"),
        captured_utility=("__selected_utility__", "sum"),
        winners50=("__winner50__", "sum"),
        selected_winners50=("__selected_winner50__", "sum"),
        winners100=("__winner100__", "sum"),
        selected_winners100=("__selected_winner100__", "sum"),
        winners200=("__winner200__", "sum"),
        selected_winners200=("__selected_winner200__", "sum"),
        selected_net_sum_bps=("__selected_net__", "sum"),
    ).reset_index()
    available = grouped["available_utility"].to_numpy(float)
    grouped["utility_recall"] = np.divide(
        grouped["captured_utility"].to_numpy(float), available,
        out=np.full(len(grouped), np.nan), where=available > 0,
    )
    for threshold in (50, 100, 200):
        denominator = grouped[f"winners{threshold}"].to_numpy(float)
        grouped[f"recall{threshold}"] = np.divide(
            grouped[f"selected_winners{threshold}"].to_numpy(float), denominator,
            out=np.full(len(grouped), np.nan), where=denominator > 0,
        )
    denominator = grouped["selected_valid_rows"].to_numpy(float)
    grouped["selected_net_ev_bps"] = np.divide(
        grouped["selected_net_sum_bps"].to_numpy(float), denominator,
        out=np.full(len(grouped), np.nan), where=denominator > 0,
    )
    positive = grouped["available_utility"] > 0
    median_available = float(grouped.loc[positive, "available_utility"].median()) if positive.any() else np.nan
    weight = np.zeros(len(grouped), dtype=float)
    if np.isfinite(median_available) and median_available > 0:
        weight[positive.to_numpy(bool)] = np.minimum(
            np.power(grouped.loc[positive, "available_utility"].to_numpy(float) / median_available, gamma),
            timestamp_cap,
        )
    grouped["utility_weight"] = weight
    grouped["month"] = _utc_month(grouped)
    grouped["route_fraction"] = fraction
    grouped["power"] = power
    grouped["cap_bps"] = cap_bps
    grouped["gamma"] = gamma
    grouped["timestamp_cap"] = timestamp_cap
    return grouped


def _primary_summary(timestamp: pd.DataFrame) -> dict[str, object]:
    relevant = timestamp["utility_recall"].notna() & (timestamp["utility_weight"] > 0)
    weights = timestamp.loc[relevant, "utility_weight"].to_numpy(float)
    r50_utility = (
        float(np.average(timestamp.loc[relevant, "utility_recall"], weights=weights))
        if weights.sum() else np.nan
    )
    r50_count = float(timestamp.loc[timestamp["recall50"].notna(), "recall50"].mean())
    r100_count = float(timestamp.loc[timestamp["recall100"].notna(), "recall100"].mean())
    r200_count = float(timestamp.loc[timestamp["recall200"].notna(), "recall200"].mean())
    s_router = 0.70 * r50_utility + 0.15 * r50_count + 0.15 * r100_count
    return {
        "timestamps": int(len(timestamp)),
        "relevant_timestamps": int(relevant.sum()),
        "r50_utility": r50_utility,
        "r50_count": r50_count,
        "r100_count": r100_count,
        "r200_count": r200_count,
        "s_router": s_router,
        "mean_selected_net_ev_bps": float(timestamp["selected_net_ev_bps"].mean()),
        "selected_rows": int(timestamp["selected_rows"].sum()),
        "selected_valid_rows": int(timestamp["selected_valid_rows"].sum()),
        "selected_net_bps": float(timestamp["selected_net_sum_bps"].sum()),
        "available_utility": float(timestamp["available_utility"].sum()),
        "captured_utility": float(timestamp["captured_utility"].sum()),
    }


def _period_rows(timestamp: pd.DataFrame, period: str) -> pd.DataFrame:
    work = timestamp.copy()
    if period == "fold":
        work["period"] = work["month"]
    elif period == "year":
        work["period"] = work["month"].str.slice(0, 4)
    elif period == "month":
        work["period"] = work["month"]
    elif period == "week":
        stamp = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
        work["period"] = stamp.dt.to_period("W-SUN").dt.start_time.astype(str)
    else:
        raise ValueError(period)
    return pd.DataFrame([
        {"period_type": period, "period": key, **_primary_summary(group)}
        for key, group in work.groupby("period", sort=True)
    ])


def _stability_summary(folds: pd.DataFrame) -> dict[str, object]:
    values = folds["s_router"].to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise AssertionError("fold Router score is incomplete")
    return {
        "mean_fold_s_router": float(values.mean()),
        "median_fold_s_router": float(np.median(values)),
        "q25_fold_s_router": float(np.quantile(values, .25)),
        "worst_fold_s_router": float(values.min()),
        "fold_std_s_router": float(values.std(ddof=0)),
        "fold_iqr_s_router": float(np.quantile(values, .75) - np.quantile(values, .25)),
        "s_stable": float(.65 * values.mean() + .25 * np.quantile(values, .25) + .10 * values.min()),
    }


def _weekly_stability_summary(weeks: pd.DataFrame) -> dict[str, object]:
    """Feature-selection selector: robust weekly mean plus lower tail."""
    values = weeks["s_router"].to_numpy(float)
    if len(values) < 5 or not np.isfinite(values).all():
        raise AssertionError("weekly Router score is incomplete")
    q05, q10, q15, q20, q80 = np.quantile(values, [.05, .10, .15, .20, .80])
    robust = float(values[(values >= q20) & (values <= q80)].mean())
    lower = float(np.mean([q15, q10, q05]))
    return {
        "weekly_s_router_robust": robust,
        "weekly_s_router_lower": lower,
        "weekly_s_stable": float(robust + .5 * lower),
        "weekly_s_router_q20": float(q20),
        "weekly_s_router_q15": float(q15),
        "weekly_s_router_q10": float(q10),
        "weekly_s_router_q05": float(q05),
        "weekly_s_router_q25": float(np.quantile(values, .25)),
        "weekly_s_router_min": float(values.min()),
    }


def _density_and_boundary(joined: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, rank_pct = _local_route(joined, score_column, .50)
    work = joined.loc[:, ["candidate_id", "__decision_ts__", "__valid__", "__net__"]].copy()
    work["selected"] = selected
    work["local_rank_pct"] = rank_pct
    work["utility"] = _utility(
        work["__net__"].to_numpy(float), work["__valid__"].to_numpy(bool), PRIMARY_POWER, PRIMARY_CAP_BPS,
    )
    work["month"] = _utc_month(work)
    work["winner50"] = work["__valid__"] & work["__net__"].gt(50.0)
    work["winner100"] = work["__valid__"] & work["__net__"].gt(100.0)
    work["winner200"] = work["__valid__"] & work["__net__"].gt(200.0)
    work["negative"] = work["__valid__"] & work["__net__"].le(0.0)
    work["severe_loss"] = work["__valid__"] & work["__net__"].le(-200.0)
    density_rows: list[dict[str, object]] = []
    for name, mask in (("selected", work["selected"]), ("rejected", ~work["selected"])):
        frame = work.loc[mask]
        density_rows.append({
            "cohort": name, "rows": int(len(frame)), "valid_rows": int(frame["__valid__"].sum()),
            "total_utility": float(frame["utility"].sum()), "utility_per_row": float(frame["utility"].mean()),
            "mean_policy_net_bps": float(frame.loc[frame["__valid__"], "__net__"].mean()),
            "positive_utility_retained_fraction": float(frame["utility"].sum() / max(work["utility"].sum(), 1e-12)),
            "raw_excess_over50_bps": float(np.maximum(frame.loc[frame["__valid__"], "__net__"] - 50.0, 0.0).sum()),
        })
    density = pd.DataFrame(density_rows)
    boundaries: list[dict[str, object]] = []
    for start, end, name in ((0.0, .2, "0-20"), (.2, .4, "20-40"), (.4, .5, "40-50_selected_boundary"), (.5, .6, "50-60_rejected_boundary"), (.6, .8, "60-80"), (.8, 1.000001, "80-100")):
        frame = work.loc[(work["local_rank_pct"] >= start) & (work["local_rank_pct"] < end)]
        boundaries.append({
            "band": name, "rows": int(len(frame)), "valid_rows": int(frame["__valid__"].sum()),
            "mean_policy_net_bps": float(frame.loc[frame["__valid__"], "__net__"].mean()),
            "median_policy_net_bps": float(frame.loc[frame["__valid__"], "__net__"].median()),
            "rate_gt50": float(frame["winner50"].mean()), "rate_gt100": float(frame["winner100"].mean()),
            "rate_gt200": float(frame["winner200"].mean()), "negative_rate": float(frame["negative"].mean()),
            "severe_loss_rate": float(frame["severe_loss"].mean()), "total_utility": float(frame["utility"].sum()),
            "utility_per_row": float(frame["utility"].mean()),
        })
    return work, density, pd.DataFrame(boundaries)


def _oracle_summary(joined: pd.DataFrame, score_column: str) -> dict[str, object]:
    selected, _ = _local_route(joined, score_column, .50)
    work = joined.loc[:, ["candidate_id", "__decision_ts__", "__valid__", "__net__"]].copy()
    work["utility"] = _utility(work["__net__"].to_numpy(float), work["__valid__"].to_numpy(bool), PRIMARY_POWER, PRIMARY_CAP_BPS)
    work["model_selected"] = selected
    ordered = work.sort_values(["__decision_ts__", "utility", "candidate_id"], ascending=[True, False, True], kind="stable").copy()
    ordered["ordinal"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    ordered["oracle_selected"] = ordered["ordinal"].to_numpy() <= np.ceil(.50 * size)
    ordered["model_utility"] = np.where(ordered["model_selected"], ordered["utility"], 0.0)
    ordered["oracle_utility"] = np.where(ordered["oracle_selected"], ordered["utility"], 0.0)
    per_timestamp = ordered.groupby("__decision_ts__", sort=False).agg(
        available=("utility", "sum"),
        model_captured=("model_utility", "sum"),
        oracle_captured=("oracle_utility", "sum"),
    ).reset_index()
    valid = per_timestamp["available"] > 0
    median_available = float(per_timestamp.loc[valid, "available"].median()) if valid.any() else np.nan
    weights = np.minimum(np.sqrt(per_timestamp.loc[valid, "available"].to_numpy(float) / median_available), 2.0) if valid.any() else np.array([])
    model_recall = np.divide(per_timestamp.loc[valid, "model_captured"], per_timestamp.loc[valid, "available"])
    oracle_recall = np.divide(per_timestamp.loc[valid, "oracle_captured"], per_timestamp.loc[valid, "available"])
    model_r50 = float(np.average(model_recall, weights=weights)) if len(weights) else np.nan
    oracle_r50 = float(np.average(oracle_recall, weights=weights)) if len(weights) else np.nan
    return {
        "model_r50_utility": model_r50,
        "oracle_r50_utility": oracle_r50,
        "headroom_captured": float((model_r50 - .50) / (oracle_r50 - .50)) if oracle_r50 > .50 else np.nan,
        "remaining_utility_gap": float(oracle_r50 - model_r50),
    }


def _sensitivity(joined: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for power in SENSITIVITY_POWERS:
        for cap_bps in SENSITIVITY_CAPS:
            for gamma in SENSITIVITY_GAMMAS:
                for timestamp_cap in SENSITIVITY_TIMESTAMP_CAPS:
                    timestamp = _timestamp_primary(joined, score_column, .50, power, cap_bps, gamma, timestamp_cap)
                    rows.append({
                        "power": power, "cap_bps": cap_bps, "gamma": gamma, "timestamp_cap": timestamp_cap,
                        **_primary_summary(timestamp),
                    })
                    # The source panel is large.  Releasing each temporary
                    # timestamp frame makes the 81-cell robustness envelope
                    # bounded-memory rather than retaining a pandas object
                    # graph until the complete grid finishes.
                    del timestamp
                    gc.collect()
    return pd.DataFrame(rows)


def _sensitivity_timestamp_base(joined: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Build the p/c-sensitive timestamp numerator once per geometry.

    Routing depends only on the immutable score, not on utility geometry.
    Therefore 81 repeated full-candidate routes would be both redundant and
    memory-expensive.  This routine routes once, then varies only the nine
    p/c utility numerators.  Gamma/cap weighting is applied later to the
    compact timestamp table.
    """
    selected, _ = _local_route(joined, score_column, .50)
    base = joined.loc[:, ["__decision_ts__", "__valid__", "__net__"]].copy()
    base["__selected__"] = selected
    output: list[pd.DataFrame] = []
    net = base["__net__"].to_numpy(float)
    valid = base["__valid__"].to_numpy(bool)
    for power in SENSITIVITY_POWERS:
        for cap_bps in SENSITIVITY_CAPS:
            utility = _utility(net, valid, power, cap_bps)
            work = pd.DataFrame({
                "__decision_ts__": base["__decision_ts__"],
                "available_utility": utility,
                "captured_utility": np.where(selected, utility, 0.0),
            })
            grouped = work.groupby("__decision_ts__", sort=False).sum().reset_index()
            grouped["power"] = power
            grouped["cap_bps"] = cap_bps
            output.append(grouped)
    return pd.concat(output, ignore_index=True)


def _sensitivity_from_timestamp_base(timestamp_base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (power, cap_bps), work in timestamp_base.groupby(["power", "cap_bps"], sort=True):
        available = work["available_utility"].to_numpy(float)
        captured = work["captured_utility"].to_numpy(float)
        valid = available > 0
        recall = np.divide(captured[valid], available[valid])
        median_available = float(np.median(available[valid])) if valid.any() else np.nan
        for gamma in SENSITIVITY_GAMMAS:
            for timestamp_cap in SENSITIVITY_TIMESTAMP_CAPS:
                weights = np.minimum(np.power(available[valid] / median_available, gamma), timestamp_cap) if valid.any() else np.array([])
                rows.append({
                    "power": float(power), "cap_bps": float(cap_bps), "gamma": gamma, "timestamp_cap": timestamp_cap,
                    "timestamps": int(len(work)), "relevant_timestamps": int(valid.sum()),
                    "r50_utility": float(np.average(recall, weights=weights)) if len(weights) else np.nan,
                })
    return pd.DataFrame(rows)


def _oracle_timestamp(joined: pd.DataFrame, score_column: str) -> pd.DataFrame:
    selected, _ = _local_route(joined, score_column, .50)
    work = joined.loc[:, ["candidate_id", "__decision_ts__", "__valid__", "__net__"]].copy()
    work["utility"] = _utility(work["__net__"].to_numpy(float), work["__valid__"].to_numpy(bool), PRIMARY_POWER, PRIMARY_CAP_BPS)
    work["model_selected"] = selected
    ordered = work.sort_values(["__decision_ts__", "utility", "candidate_id"], ascending=[True, False, True], kind="stable").copy()
    ordered["ordinal"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    ordered["oracle_selected"] = ordered["ordinal"].to_numpy() <= np.ceil(.50 * size)
    ordered["model_utility"] = np.where(ordered["model_selected"], ordered["utility"], 0.0)
    ordered["oracle_utility"] = np.where(ordered["oracle_selected"], ordered["utility"], 0.0)
    return ordered.groupby("__decision_ts__", sort=False).agg(
        available=("utility", "sum"), model_captured=("model_utility", "sum"), oracle_captured=("oracle_utility", "sum"),
    ).reset_index()


def _oracle_from_timestamp(per_timestamp: pd.DataFrame) -> dict[str, object]:
    valid = per_timestamp["available"] > 0
    median_available = float(per_timestamp.loc[valid, "available"].median()) if valid.any() else np.nan
    weights = np.minimum(np.sqrt(per_timestamp.loc[valid, "available"].to_numpy(float) / median_available), 2.0) if valid.any() else np.array([])
    model_recall = np.divide(per_timestamp.loc[valid, "model_captured"], per_timestamp.loc[valid, "available"])
    oracle_recall = np.divide(per_timestamp.loc[valid, "oracle_captured"], per_timestamp.loc[valid, "available"])
    model_r50 = float(np.average(model_recall, weights=weights)) if len(weights) else np.nan
    oracle_r50 = float(np.average(oracle_recall, weights=weights)) if len(weights) else np.nan
    return {
        "model_r50_utility": model_r50, "oracle_r50_utility": oracle_r50,
        "headroom_captured": float((model_r50 - .50) / (oracle_r50 - .50)) if oracle_r50 > .50 else np.nan,
        "remaining_utility_gap": float(oracle_r50 - model_r50),
    }


def _frontier(joined: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    pieces: list[pd.DataFrame] = []
    for fraction in ROUTE_FRACTIONS:
        timestamp = _timestamp_primary(
            joined, score_column, fraction, PRIMARY_POWER, PRIMARY_CAP_BPS, PRIMARY_GAMMA, PRIMARY_TIMESTAMP_CAP,
        )
        rows.append({"route_fraction": fraction, **_primary_summary(timestamp)})
        pieces.append(timestamp)
    frontier = pd.DataFrame(rows)
    local = frontier.loc[(frontier["route_fraction"] >= .40) & (frontier["route_fraction"] <= .60), "r50_utility"].to_numpy(float)
    auc_40_60 = float(np.trapz(local, x=np.array([.40, .45, .50, .55, .60]))) if len(local) == 5 else np.nan
    frontier["auc_40_60"] = auc_40_60
    return frontier, pd.concat(pieces, ignore_index=True)


def _load_joined(source: Path, policy_path: Path, score_column: str) -> tuple[pd.DataFrame, dict[str, str]]:
    months = _read_contract_months(source)
    policy = pd.read_parquet(policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identities")
    frames: list[pd.DataFrame] = []
    score_hashes: dict[str, str] = {}
    for month in months:
        path = source / "target_free_scores" / f"month={month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        score = pd.read_parquet(path)
        _validate_target_free(score, score_column, path)
        score_hashes[month] = _sha256(path)
        frames.append(score.loc[:, ["candidate_id", "__decision_ts__", score_column]])
    score = pd.concat(frames, ignore_index=True)
    joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    joined["__net__"] = pd.to_numeric(joined["policy_net_bps"], errors="coerce")
    joined["__valid__"] = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(joined["__net__"])
    return joined, score_hashes


def run(source: Path, policy: Path, score_column: str, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluator artifact: {out}")
    # Create the immutable directory first, then materialise large receipts
    # one month at a time.  If an interruption occurs, the absence of the
    # final summary receipt marks the artifact incomplete; no partial file is
    # ever mistaken for a completed evaluation.
    out.mkdir(parents=True)
    months = _read_contract_months(source)
    policy_frame = pd.read_parquet(policy, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    if policy_frame["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identities")
    primary_parts: list[pd.DataFrame] = []
    oracle_parts: list[pd.DataFrame] = []
    frontier_parts: list[pd.DataFrame] = []
    sensitivity_parts: list[pd.DataFrame] = []
    score_hashes: dict[str, str] = {}
    candidate_root = out / "router_candidate_utility_parts"
    candidate_root.mkdir()
    for month_name in months:
        score_path = source / "target_free_scores" / f"month={month_name}.parquet"
        score = pd.read_parquet(score_path)
        _validate_target_free(score, score_column, score_path)
        score_hashes[month_name] = _sha256(score_path)
        joined = score.loc[:, ["candidate_id", "__decision_ts__", score_column]].merge(
            policy_frame, on="candidate_id", how="left", validate="one_to_one",
        )
        joined["__net__"] = pd.to_numeric(joined["policy_net_bps"], errors="coerce")
        joined["__valid__"] = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(joined["__net__"])
        primary_parts.append(_timestamp_primary(
            joined, score_column, .50, PRIMARY_POWER, PRIMARY_CAP_BPS, PRIMARY_GAMMA, PRIMARY_TIMESTAMP_CAP,
        ))
        routed_rows, _, _ = _density_and_boundary(joined, score_column)
        routed_rows.to_parquet(candidate_root / f"month={month_name}.parquet", index=False, compression="zstd")
        del routed_rows
        oracle_parts.append(_oracle_timestamp(joined, score_column))
        for fraction in ROUTE_FRACTIONS:
            frontier_parts.append(_timestamp_primary(
                joined, score_column, fraction, PRIMARY_POWER, PRIMARY_CAP_BPS, PRIMARY_GAMMA, PRIMARY_TIMESTAMP_CAP,
            ))
        sensitivity_parts.append(_sensitivity_timestamp_base(joined, score_column))
        del score, joined
        gc.collect()
    primary_timestamp = pd.concat(primary_parts, ignore_index=True)
    overall = _primary_summary(primary_timestamp)
    fold = _period_rows(primary_timestamp, "fold")
    month = _period_rows(primary_timestamp, "month")
    year = _period_rows(primary_timestamp, "year")
    week = _period_rows(primary_timestamp, "week")
    stability = _stability_summary(fold)
    weekly_stability = _weekly_stability_summary(week)
    candidate_parts = sorted(candidate_root.glob("month=*.parquet"))
    candidate = pd.concat([pd.read_parquet(path) for path in candidate_parts], ignore_index=True)
    selected = candidate["selected"].astype(bool)
    total_utility = float(candidate["utility"].sum())
    density_rows: list[dict[str, object]] = []
    for name, mask in (("selected", selected), ("rejected", ~selected)):
        frame = candidate.loc[mask]
        valid = frame["__valid__"].astype(bool)
        density_rows.append({
            "cohort": name, "rows": int(len(frame)), "valid_rows": int(valid.sum()),
            "total_utility": float(frame["utility"].sum()), "utility_per_row": float(frame["utility"].mean()),
            "mean_policy_net_bps": float(frame.loc[valid, "__net__"].mean()),
            "positive_utility_retained_fraction": float(frame["utility"].sum() / max(total_utility, 1e-12)),
            "raw_excess_over50_bps": float(np.maximum(frame.loc[valid, "__net__"] - 50.0, 0.0).sum()),
        })
    density = pd.DataFrame(density_rows)
    boundary_rows: list[dict[str, object]] = []
    for start, end, name in ((0.0, .2, "0-20"), (.2, .4, "20-40"), (.4, .5, "40-50_selected_boundary"), (.5, .6, "50-60_rejected_boundary"), (.6, .8, "60-80"), (.8, 1.000001, "80-100")):
        frame = candidate.loc[(candidate["local_rank_pct"] >= start) & (candidate["local_rank_pct"] < end)]
        valid = frame["__valid__"].astype(bool)
        boundary_rows.append({
            "band": name, "rows": int(len(frame)), "valid_rows": int(valid.sum()),
            "mean_policy_net_bps": float(frame.loc[valid, "__net__"].mean()),
            "median_policy_net_bps": float(frame.loc[valid, "__net__"].median()),
            "rate_gt50": float(frame["winner50"].mean()), "rate_gt100": float(frame["winner100"].mean()),
            "rate_gt200": float(frame["winner200"].mean()), "negative_rate": float(frame["negative"].mean()),
            "severe_loss_rate": float(frame["severe_loss"].mean()), "total_utility": float(frame["utility"].sum()),
            "utility_per_row": float(frame["utility"].mean()),
        })
    boundary = pd.DataFrame(boundary_rows)
    del candidate
    gc.collect()
    oracle_timestamp = pd.concat(oracle_parts, ignore_index=True)
    oracle = _oracle_from_timestamp(oracle_timestamp)
    frontier_timestamp = pd.concat(frontier_parts, ignore_index=True)
    frontier = pd.DataFrame([
        {"route_fraction": float(fraction), **_primary_summary(group)}
        for fraction, group in frontier_timestamp.groupby("route_fraction", sort=True)
    ])
    local = frontier.loc[(frontier["route_fraction"] >= .40) & (frontier["route_fraction"] <= .60), "r50_utility"].to_numpy(float)
    frontier["auc_40_60"] = float(np.trapz(local, x=np.array([.40, .45, .50, .55, .60]))) if len(local) == 5 else np.nan
    sensitivity_base = pd.concat(sensitivity_parts, ignore_index=True)
    sensitivity = _sensitivity_from_timestamp_base(sensitivity_base)
    del oracle_parts, frontier_parts, sensitivity_parts, sensitivity_base, policy_frame
    gc.collect()
    primary_timestamp.to_parquet(out / "router_primary_timestamp_metrics.parquet", index=False, compression="zstd")
    fold.to_parquet(out / "router_fold_metrics.parquet", index=False, compression="zstd")
    month.to_parquet(out / "router_month_metrics.parquet", index=False, compression="zstd")
    year.to_parquet(out / "router_year_metrics.parquet", index=False, compression="zstd")
    week.to_parquet(out / "router_week_metrics.parquet", index=False, compression="zstd")
    density.to_parquet(out / "router_opportunity_density.parquet", index=False, compression="zstd")
    boundary.to_parquet(out / "router_boundary_economics.parquet", index=False, compression="zstd")
    frontier.to_parquet(out / "router_route_frontier.parquet", index=False, compression="zstd")
    frontier_timestamp.to_parquet(out / "router_frontier_timestamp_metrics.parquet", index=False, compression="zstd")
    oracle_timestamp.to_parquet(out / "router_oracle_timestamp_metrics.parquet", index=False, compression="zstd")
    sensitivity.to_parquet(out / "router_utility_sensitivity.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "router_control_summary.json", {
        "schema": SCHEMA, "source": str(source), "policy": str(policy), "score_column": score_column,
        "source_contract_sha256": _sha256(source / "run_contract.json"),
        "source_score_sha256_by_month": score_hashes,
        "rows": int(sum(int(part["candidate_rows"].sum()) for part in primary_parts)),
        "valid_label_rows": int(sum(int(part["valid_rows"].sum()) for part in primary_parts)),
        "route_rounding": "ceil(fraction * timestamp_candidate_count)",
        "tie_break": "descending finite score; ascending candidate_id",
        "primary_utility": {"hurdle_bps": HURDLE_BPS, "power": PRIMARY_POWER, "cap_bps": PRIMARY_CAP_BPS,
                            "timestamp_gamma": PRIMARY_GAMMA, "timestamp_cap": PRIMARY_TIMESTAMP_CAP},
        "overall": overall, "fold_stability": stability, "weekly_stability": weekly_stability, "oracle": oracle,
        "causality": "immutable target-free score receipts are routed before resolved policy outcomes are joined",
        "status": "complete",
    })


def main() -> None:
    global PRIMARY_POWER, PRIMARY_CAP_BPS, PRIMARY_GAMMA, PRIMARY_TIMESTAMP_CAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="immutable strict-OOF Router artifact")
    parser.add_argument("--policy", type=Path, required=True, help="canonical rich-policy outcome ledger")
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-power", type=float, default=PRIMARY_POWER)
    parser.add_argument("--primary-cap-bps", type=float, default=PRIMARY_CAP_BPS)
    parser.add_argument("--timestamp-gamma", type=float, default=PRIMARY_GAMMA)
    parser.add_argument("--timestamp-cap", type=float, default=PRIMARY_TIMESTAMP_CAP)
    args = parser.parse_args()
    if args.primary_power <= 0 or args.primary_cap_bps <= 0 or args.timestamp_cap <= 0:
        raise ValueError("utility power, cap, and timestamp cap must be positive")
    PRIMARY_POWER = float(args.primary_power)
    PRIMARY_CAP_BPS = float(args.primary_cap_bps)
    PRIMARY_GAMMA = float(args.timestamp_gamma)
    PRIMARY_TIMESTAMP_CAP = float(args.timestamp_cap)
    run(args.source, args.policy, args.score_column, args.out)


if __name__ == "__main__":
    main()
