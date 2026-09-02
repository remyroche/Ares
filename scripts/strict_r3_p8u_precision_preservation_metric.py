#!/usr/bin/env python3
"""Timestamp-local precision-plus-preservation score for the P8u Base search.

This module is deliberately model- and target-agnostic.  It evaluates a
target-free score only after it has been joined to an outcome panel, while
retaining the full scored candidate population in every timestamp.  Invalid
or unresolved policy rows therefore reduce outcome coverage but never change
which rows were eligible for a top-k rank.

The selection statistic implements the Base contract requested for the P8u
router:

    BaseScore_t = .30 DTP2_t + .30 DTP5_t + .20 DTP10_t
                  + .20 ResidualUR10→30_t

where the DTP terms are normalised by a matched control's equal-timestamp
means and ``ResidualUR10→30`` is the share of residual Router50 positive
utility captured when extending the Base tail from Top-10% to Top-30%.
``UR20`` is also emitted as a diagnostic: it is the total positive utility
captured in the Base Top-20% relative to all positive utility in Router50.
It is intentionally not a second optimisation term—the declared 20% utility
weight is the residual-preservation term above, so adding UR20 as well would
silently change the user's requested BaseScore.
Weekly BaseScores are then aggregated as the mean between their 20th and 80th
percentiles plus half the mean of the 5th, 10th, and 15th percentile weekly
scores.  The latter makes lower-tail weeks an explicit selection criterion
without allowing one week to dominate the entire objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
COMPONENTS = ("dtp2_bps", "dtp5_bps", "dtp10_bps", "residual_utility_recall10_to30")
WEIGHTS = {
    "dtp2_bps": 0.30,
    "dtp5_bps": 0.30,
    "dtp10_bps": 0.20,
    "residual_utility_recall10_to30": 0.20,
}


@dataclass(frozen=True)
class StableSummary:
    score_stable: float
    week_robust_average: float
    week_score_q15: float
    week_score_q10: float
    week_score_q05: float
    base_score_mean: float
    timestamp_rows: int
    week_rows: int


def _require(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"missing P8u Base metric columns: {missing}")


def _rank(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Rank every scored row inside its timestamp with deterministic ties."""
    _require(frame, ("candidate_id", "__decision_ts__", score_column))
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score_column]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    values = pd.to_numeric(work[score_column], errors="coerce")
    if not np.isfinite(values).all():
        raise AssertionError("target-free Base score has non-finite values")
    work[score_column] = values.to_numpy(float)
    ordered = work.sort_values(
        ["__decision_ts__", score_column, "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    ordered["rank"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
    ordered["count"] = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    return ordered.sort_values("__row__", kind="stable").drop(columns="__row__")


def timestamp_components(
    frame: pd.DataFrame,
    *,
    score_column: str,
    outcome_column: str = "policy_net_bps",
    valid_column: str = "policy_ordinal_valid",
    utility_floor_bps: float = 50.0,
) -> pd.DataFrame:
    """Return equal-timestamp precision and preservation components.

    The rank is computed before outcome validity is examined.  For a
    timestamp, DTP values use the mean over resolved selected rows.
    ``ResidualUR10→30`` is:

      (U(Top30) - U(Top10)) / (U(Router50) - U(Top10) + epsilon)

    so positive utility already captured by the high-conviction tip cannot
    inflate the preservation term.  Its numerator and denominator use only
    resolved rows, after ranking the complete target-free candidate set.
    """
    _require(frame, ("candidate_id", "__decision_ts__", score_column, outcome_column, valid_column))
    work = frame.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    ranked = _rank(work, score_column)
    work["__rank__"] = ranked["rank"].to_numpy(np.int64)
    work["__count__"] = ranked["count"].to_numpy(np.int64)
    work["__outcome__"] = pd.to_numeric(work[outcome_column], errors="coerce")
    work["__valid__"] = work[valid_column].fillna(False).astype(bool) & np.isfinite(work["__outcome__"])
    work["__utility__"] = np.maximum(work["__outcome__"].to_numpy(float) - float(utility_floor_bps), 0.0)

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for stamp, group in work.groupby("__decision_ts__", sort=True):
        count = int(group["__count__"].iat[0])
        if count < 1:
            raise AssertionError("empty timestamp in rank metric")
        valid = group.loc[group["__valid__"]].copy()
        item: dict[str, float | int | pd.Timestamp] = {
            "__decision_ts__": stamp,
            "candidate_rows": count,
            "valid_rows": int(len(valid)),
        }
        for fraction, name in ((.02, "dtp2_bps"), (.05, "dtp5_bps"), (.10, "dtp10_bps")):
            selected = group.loc[group["__rank__"].le(int(np.ceil(count * fraction)))]
            resolved = selected.loc[selected["__valid__"]]
            item[name] = float(resolved["__outcome__"].mean()) if len(resolved) else np.nan
            item[f"{name}_coverage"] = float(len(resolved) / len(selected)) if len(selected) else np.nan
        top10 = group.loc[group["__rank__"].le(int(np.ceil(count * .10)))]
        top20 = group.loc[group["__rank__"].le(int(np.ceil(count * .20)))]
        top30 = group.loc[group["__rank__"].le(int(np.ceil(count * .30)))]
        utility_all = float(valid["__utility__"].sum())
        utility_top10 = float(top10.loc[top10["__valid__"], "__utility__"].sum())
        utility_top20 = float(top20.loc[top20["__valid__"], "__utility__"].sum())
        utility_top30 = float(top30.loc[top30["__valid__"], "__utility__"].sum())
        item["utility_recall20"] = utility_top20 / (utility_all + 1e-8)
        item["utility_recall20_eligible"] = int(utility_all > 0.0)
        residual_total = max(utility_all - utility_top10, 0.0)
        residual_selected = max(utility_top30 - utility_top10, 0.0)
        item["residual_utility_recall10_to30"] = residual_selected / (residual_total + 1e-8)
        item["residual_utility_recall10_to30_eligible"] = int(residual_total > 0.0)
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        raise AssertionError("no timestamps available for BaseStable metric")
    return result


def _denominators(control: pd.DataFrame) -> pd.Series:
    _require(control, COMPONENTS)
    values = control.loc[:, list(COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    denominator = values.mean(axis=0, skipna=True).abs()
    # All precision denominators are bps.  The utility-recall denominator is
    # dimensionless.  The floor prevents a near-zero control component from
    # manufacturing a huge selection statistic in an uninformative period.
    floor = pd.Series({
        "dtp2_bps": 1.0, "dtp5_bps": 1.0, "dtp10_bps": 1.0,
        "residual_utility_recall10_to30": 0.01,
    })
    denominator = denominator.where(denominator.ge(floor), floor)
    if not np.isfinite(denominator.to_numpy(float)).all():
        raise AssertionError("invalid frozen-control normalisation denominator")
    return denominator


def _trimmed_mean(values: pd.Series, lower: float = .20, upper: float = .80) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    lo, hi = clean.quantile([lower, upper])
    kept = clean.loc[clean.ge(lo) & clean.le(hi)]
    return float(kept.mean()) if len(kept) else float(clean.median())


def stable_score(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
) -> tuple[StableSummary, pd.DataFrame]:
    """Evaluate candidate components against a matched frozen control.

    `candidate` and `control` must have exactly the same timestamp support.
    The caller is responsible for producing both from target-free ranked
    panels on the identical P8u routed identities.
    """
    _require(candidate, ("__decision_ts__", *COMPONENTS))
    _require(control, ("__decision_ts__", *COMPONENTS))
    left = candidate.loc[:, ["__decision_ts__", *COMPONENTS]].copy()
    right = control.loc[:, ["__decision_ts__", *COMPONENTS]].copy()
    for part in (left, right):
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        if part["__decision_ts__"].duplicated().any():
            raise AssertionError("BaseStable component panel has duplicate timestamps")
    merged = left.merge(right, on="__decision_ts__", suffixes=("", "_control"), how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise AssertionError("candidate/control BaseStable timestamps differ")
    denominator = _denominators(right)
    for component in COMPONENTS:
        merged[f"norm_{component}"] = pd.to_numeric(merged[component], errors="coerce") / float(denominator[component])
    merged["base_score"] = sum(float(WEIGHTS[key]) * merged[f"norm_{key}"] for key in COMPONENTS)
    weekly = merged.groupby(
        merged["__decision_ts__"].dt.isocalendar().year.astype(str) + "-" + merged["__decision_ts__"].dt.isocalendar().week.astype(str),
        sort=True,
    )["base_score"].mean().rename("week_score").reset_index()
    clean_week = weekly["week_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_week) < 4:
        raise AssertionError("BaseStable requires at least four resolved weekly scores")
    robust = _trimmed_mean(clean_week)
    q15, q10, q05 = (float(clean_week.quantile(level)) for level in (.15, .10, .05))
    stable = float(robust + .5 * np.mean([q15, q10, q05]))
    return StableSummary(
        score_stable=stable,
        week_robust_average=robust,
        week_score_q15=q15,
        week_score_q10=q10,
        week_score_q05=q05,
        base_score_mean=float(merged["base_score"].mean()),
        timestamp_rows=len(merged),
        week_rows=len(clean_week),
    ), merged
