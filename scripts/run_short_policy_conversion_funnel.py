#!/usr/bin/env python3
"""Strict-OOF short base policy-conversion LambdaRank funnel.

This is deliberately a *base-layer* experiment.  It ranks short candidates
within an exact decision timestamp.  Absolute trade/no-trade calibration stays
outside this script for the downstream meta/admission layer.

All policy labels are loaded from a target-free identity-preserving ledger.
Missing 1m paths remain null and never become zero-return labels.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_base_target_objective_funnel import (  # noqa: E402
    _coverage_fields,
    _feature_fields,
    _load_candidates,
    _load_features,
    _selected_short_feature_fields,
    _sha256,
)
from scripts.run_strict_r3_ordinal_base_target_ablation import FROZEN_BASE_PARAMS  # noqa: E402


SEED = 17
SIDE = "short"
TAILS = (0.001, 0.0025, 0.005, 0.01, 0.02)
# Keep the narrow tip but expose the broader screen depths required to judge
# whether a ranker is merely lucky at rank one or remains useful to a
# downstream portfolio auction.
TOP_KS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    description: str
    target_kind: str
    alpha_policy: float | None = None
    truncation: int = 8
    gain_family: str = "linear"
    query_hours: int = 1
    absolute_weight: float | None = None
    objective: str = "lambdarank"
    lambdarank_norm: bool = True
    quantile_cutoffs: tuple[float, ...] = (0.40, 0.60, 0.75, 0.85, 0.92, 0.97, 0.99)
    weight_kind: str = "uniform"
    # Optional absolute-alpha target geometry.  These fields are label-only;
    # none becomes an inference input.
    absolute_edges_bps: tuple[float, ...] | None = None
    quantization_bps: float | None = None
    hybrid_relative_weight: float | None = None
    row_weight_kind: str = "none"


ROUND_A: tuple[PolicySpec, ...] = (
    PolicySpec("P0_h12_c1", "Current C1 H12-net relevance control.", "h12_bins"),
    PolicySpec("P1_policy_bps", "Exact canonical-policy bps relevance.", "policy_bps"),
    PolicySpec("P2_policy_rank", "Exact policy within-timestamp percentile relevance.", "policy_rank"),
    PolicySpec("P3_blend_a067", "67% policy rank plus 33% H12 rank.", "blend", 0.67),
    PolicySpec("P3_blend_a075", "75% policy rank plus 25% H12 rank.", "blend", 0.75),
    PolicySpec("P3_blend_a090", "90% policy rank plus 10% H12 rank.", "blend", 0.90),
    PolicySpec("P3_blend_a100", "100% policy rank control.", "blend", 1.00),
    PolicySpec("P4_min_rank", "Minimum of policy and H12 within-timestamp ranks.", "min_rank"),
    PolicySpec("P5_median_policy", "Median local-policy-family within-timestamp rank.", "median_rank"),
    PolicySpec("P6_p25_policy", "P25 local-policy-family within-timestamp rank.", "p25_rank"),
)

GAIN_FAMILIES: dict[str, list[float]] = {
    "linear": [0, 1, 2, 3, 4, 5, 6, 7],
    "mild": [0, 1, 2, 3, 5, 7, 10, 14],
    "moderate": [0, 1, 2, 4, 6, 9, 13, 18],
    "strong": [0, 1, 2, 4, 8, 14, 22, 32],
    "saturating": [0, 1, 2, 3, 4, 5, 6, 6],
    # Explicit economics-first relevance geometry used by the absolute-alpha
    # funnel.  The fixed 100-bps policy cost is already present in the label.
    "cost_deadzone": [0, 1, 2, 3, 5, 7],
    "cost_margin": [0, 1, 2, 4, 7, 11, 16],
}
POLICY_VARIANT_KEYS = (
    "p0_canonical", "p1_sl25", "p2_sl35", "p3_activation40",
    "p4_activation60", "p5_giveback20", "p6_giveback30",
)
PATH_TARGET_KINDS = {
    "early_mfe1_rank", "early_mfe2_rank", "early_mfe3_rank",
    "squeeze_l025_rank", "squeeze_l050_rank", "squeeze_l100_rank",
    "activation_grade", "conversion_quality_a", "conversion_quality_b",
}
PATH_TARGET_COLUMNS = (
    "__path_auxiliary_target_valid__", "__early_mfe_1h_atr__", "__early_mfe_2h_atr__", "__early_mfe_3h_atr__",
    "__max_adverse_before_activation_atr__", "__time_to_2atr_minutes__",
    "__fraction_bars_above_80pct_peak__", "__activation_before_adverse_grade__",
)


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _paths(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    return [root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
            for month in pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left")]


def _load_policy_ledger(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in _paths(root, start, end):
        if not path.exists():
            raise FileNotFoundError(path)
        pieces.append(pd.read_parquet(path))
    frame = pd.concat(pieces, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__", "policy_label_available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("policy ledger has invalid short candidate identities")
    return frame


def _load_supportive_ledger(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in _paths(root, start, end):
        if not path.exists():
            raise FileNotFoundError(path)
        pieces.append(pd.read_parquet(path, columns=["candidate_id", *PATH_TARGET_COLUMNS]))
    return pd.concat(pieces, ignore_index=True)


def _valid_h12(frame: pd.DataFrame) -> np.ndarray:
    return (frame.label_valid.astype(bool) & ~frame.target_invalid.astype(bool)
            & pd.to_numeric(frame.t4_tp6_sl4_net_bps, errors="coerce").notna()).to_numpy(bool)


def _valid_policy(frame: pd.DataFrame) -> np.ndarray:
    return (frame.policy_path_valid.astype(bool)
            & pd.to_numeric(frame.p0_canonical_net_bps, errors="coerce").notna()).to_numpy(bool)


def _percentile(values: pd.Series) -> pd.Series:
    # ``pct=True`` maps the largest outcome to one and has deterministic
    # average-tie handling.  It is calculated inside training timestamps only.
    return values.rank(method="average", pct=True)


def _query_key(frame: pd.DataFrame) -> pd.Series:
    """Use an explicitly materialised causal query key when supplied."""
    return frame["__query__"] if "__query__" in frame.columns else frame["__ts__"]


def _grade_percentile(values: pd.Series) -> pd.Series:
    # 0–50, 50–70, 70–80, 80–90, 90–95, 95–98, 98–99, 99–100.
    finite = values.notna() & np.isfinite(values.to_numpy(float))
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    result.loc[finite] = np.digitize(
        values.loc[finite].to_numpy(float),
        [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99], right=True,
    )
    return result


def _train_quantile_edges(policy: pd.Series, valid: np.ndarray, spec: PolicySpec) -> np.ndarray:
    """Fit absolute policy-label cutoffs only on the training population."""
    values = policy.loc[valid]
    if values.empty:
        raise ValueError("cannot fit policy quantile grades without valid labels")
    return values.quantile(list(spec.quantile_cutoffs)).to_numpy(float)


def _targets(frame: pd.DataFrame, spec: PolicySpec, *, train_quantile_edges: np.ndarray | None = None) -> np.ndarray:
    policy = pd.to_numeric(frame.p0_canonical_net_bps, errors="coerce")
    policy_valid = _valid_policy(frame)
    # Pure policy-net relevance does not require an H12 target column.  Keep
    # that dependency conditional so a policy-only training ledger remains a
    # valid substrate for absolute-alpha target ablations.
    if "t4_tp6_sl4_net_bps" in frame.columns:
        h12 = pd.to_numeric(frame.t4_tp6_sl4_net_bps, errors="coerce")
        h12_valid = _valid_h12(frame)
    else:
        h12 = pd.Series(np.nan, index=frame.index, dtype="float64")
        h12_valid = np.zeros(len(frame), dtype=bool)
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    if spec.target_kind == "policy_edges":
        edges = spec.absolute_edges_bps
        if edges is None or not edges or any(not np.isfinite(value) for value in edges) or tuple(sorted(edges)) != tuple(edges):
            raise ValueError("policy_edges requires strictly increasing finite absolute_edges_bps")
        if len(set(edges)) != len(edges):
            raise ValueError("policy_edges contains duplicate absolute thresholds")
        result.loc[policy_valid] = np.digitize(policy.loc[policy_valid], edges, right=True)
        return result.to_numpy(float)
    if spec.target_kind == "policy_quantized":
        step = spec.quantization_bps
        if step is None or not np.isfinite(step) or step <= 0.0:
            raise ValueError("policy_quantized requires a positive quantization_bps")
        # This makes outcome differences below the declared bps step invisible
        # while keeping grade IDs non-negative for LambdaRank.  Both bounds are
        # fixed before fitting and avoid one remote path creating unbounded
        # relevance values.
        clipped = np.clip(policy.loc[policy_valid].to_numpy(float), -400.0, 600.0)
        result.loc[policy_valid] = np.floor((clipped + 400.0) / float(step)).astype(np.int16)
        return result.to_numpy(float)
    if spec.target_kind == "h12_bins":
        result.loc[h12_valid] = np.digitize(h12.loc[h12_valid], [-200, -100, 0, 100, 200, 400], right=True)
        return result.to_numpy(float)
    if spec.target_kind == "policy_bps":
        result.loc[policy_valid] = np.digitize(policy.loc[policy_valid], [-400, -200, 0, 100, 200, 400], right=True)
        return result.to_numpy(float)
    if spec.target_kind == "policy_coarse":
        result.loc[policy_valid] = np.digitize(policy.loc[policy_valid], [-300, -200, -100, 0, 100, 200, 400], right=True)
        return result.to_numpy(float)
    if spec.target_kind == "policy_deadzone":
        result.loc[policy_valid] = np.digitize(policy.loc[policy_valid], [-300, -100, 100, 250, 500], right=True)
        return result.to_numpy(float)
    if spec.target_kind == "policy_fine_tail":
        result.loc[policy_valid] = np.digitize(policy.loc[policy_valid], [-200, 0, 100, 200, 300, 450, 650], right=True)
        return result.to_numpy(float)
    if spec.target_kind == "train_quantile":
        edges = train_quantile_edges
        if edges is None:
            # This fallback is only for deterministic unit tests.  Production
            # fitting always supplies edges derived from the training fold.
            edges = _train_quantile_edges(policy, policy_valid, spec)
        result.loc[policy_valid] = np.digitize(policy.loc[policy_valid], edges, right=True)
        return result.to_numpy(float)
    if spec.target_kind in PATH_TARGET_KINDS:
        missing = sorted(set(PATH_TARGET_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"path target requires supportive sidecar columns: {missing}")
        valid = policy_valid & frame["__path_auxiliary_target_valid__"].astype(bool).to_numpy()
        def ranked(values: pd.Series) -> pd.Series:
            output = pd.Series(np.nan, index=frame.index, dtype="float64")
            for _, group in frame.loc[valid].assign(_value=values.loc[valid]).groupby(_query_key(frame.loc[valid]), sort=False):
                output.loc[group.index] = _percentile(group._value)
            return output
        early1 = pd.to_numeric(frame["__early_mfe_1h_atr__"], errors="coerce")
        early2 = pd.to_numeric(frame["__early_mfe_2h_atr__"], errors="coerce")
        early = pd.to_numeric(frame["__early_mfe_3h_atr__"], errors="coerce")
        adverse = pd.to_numeric(frame["__max_adverse_before_activation_atr__"], errors="coerce")
        squeeze025 = early - .25 * adverse
        squeeze050 = early - .50 * adverse
        squeeze100 = early - 1.00 * adverse
        activation = pd.to_numeric(frame["__activation_before_adverse_grade__"], errors="coerce")
        if spec.target_kind == "early_mfe1_rank":
            return _grade_percentile(ranked(early1)).to_numpy(float)
        if spec.target_kind == "early_mfe2_rank":
            return _grade_percentile(ranked(early2)).to_numpy(float)
        if spec.target_kind == "early_mfe3_rank":
            return _grade_percentile(ranked(early)).to_numpy(float)
        if spec.target_kind == "squeeze_l025_rank":
            return _grade_percentile(ranked(squeeze025)).to_numpy(float)
        if spec.target_kind == "squeeze_l050_rank":
            return _grade_percentile(ranked(squeeze050)).to_numpy(float)
        if spec.target_kind == "squeeze_l100_rank":
            return _grade_percentile(ranked(squeeze100)).to_numpy(float)
        if spec.target_kind == "activation_grade":
            result.loc[valid] = activation.loc[valid]
            return result.to_numpy(float)
        time = pd.to_numeric(frame["__time_to_2atr_minutes__"], errors="coerce").fillna(721.0)
        persistence = pd.to_numeric(frame["__fraction_bars_above_80pct_peak__"], errors="coerce")
        components = (ranked(early), ranked(-pd.to_numeric(frame["__max_adverse_before_activation_atr__"], errors="coerce")), ranked(-time), ranked(persistence))
        weights = (.35, .25, .20, .20) if spec.target_kind == "conversion_quality_a" else (.40, .30, .15, .15)
        composite = sum(weight * component for weight, component in zip(weights, components, strict=True))
        return _grade_percentile(composite).to_numpy(float)
    policy_rank = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in frame.loc[policy_valid].groupby(_query_key(frame.loc[policy_valid]), sort=False):
        policy_rank.loc[group.index] = _percentile(policy.loc[group.index])
    if spec.target_kind == "policy_rank":
        return _grade_percentile(policy_rank).to_numpy(float)
    h12_rank = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in frame.loc[h12_valid].groupby(_query_key(frame.loc[h12_valid]), sort=False):
        h12_rank.loc[group.index] = _percentile(h12.loc[group.index])
    both = policy_rank.notna() & h12_rank.notna()
    if spec.target_kind == "blend":
        assert spec.alpha_policy is not None
        return _grade_percentile(spec.alpha_policy * policy_rank.where(both) + (1.0 - spec.alpha_policy) * h12_rank.where(both)).to_numpy(float)
    if spec.target_kind == "min_rank":
        return _grade_percentile(pd.concat([policy_rank.where(both), h12_rank.where(both)], axis=1).min(axis=1, skipna=False)).to_numpy(float)
    if spec.target_kind == "relative_quantile":
        return _grade_percentile(policy_rank).to_numpy(float)
    if spec.target_kind == "hybrid_rank":
        if spec.absolute_weight is None:
            raise ValueError("hybrid rank requires absolute_weight")
        absolute = pd.Series(np.nan, index=frame.index, dtype="float64")
        absolute.loc[policy_valid] = np.digitize(policy.loc[policy_valid], [-400, -200, 0, 100, 200, 400], right=True) / 6.0
        return _grade_percentile((1.0 - spec.absolute_weight) * policy_rank + spec.absolute_weight * absolute).to_numpy(float)
    if spec.target_kind == "hybrid_absolute_relative":
        if spec.hybrid_relative_weight is None or not 0.0 <= float(spec.hybrid_relative_weight) <= 1.0:
            raise ValueError("hybrid_absolute_relative requires a relative weight in [0, 1]")
        edges = spec.absolute_edges_bps
        if edges is None or not edges or tuple(sorted(edges)) != tuple(edges):
            raise ValueError("hybrid_absolute_relative requires increasing absolute_edges_bps")
        absolute = pd.Series(np.nan, index=frame.index, dtype="float64")
        absolute.loc[policy_valid] = np.digitize(policy.loc[policy_valid], edges, right=True) / float(len(edges) + 1)
        combined = float(spec.hybrid_relative_weight) * policy_rank + (1.0 - float(spec.hybrid_relative_weight)) * absolute
        valid = combined.notna()
        result.loc[valid] = np.minimum(6, np.floor(np.clip(combined.loc[valid], 0.0, 1.0 - 1e-12) * 7.0)).astype(np.int8)
        return result.to_numpy(float)
    values = frame.loc[policy_valid, [f"{key}_net_bps" for key in POLICY_VARIANT_KEYS]].apply(pd.to_numeric, errors="coerce")
    if spec.target_kind == "median_rank":
        aggregate = values.median(axis=1)
    elif spec.target_kind == "p25_rank":
        aggregate = values.quantile(0.25, axis=1)
    elif spec.target_kind == "trimmed_mean_rank":
        # Seven predeclared policy variants: remove the best and worst valid
        # execution path for each candidate before averaging.  This is a
        # label-only robustness target; variants never become inference
        # inputs.  The calculation remains valid if an individual variant is
        # absent, provided at least three policy paths are available.
        array = values.to_numpy(float)
        counts = np.isfinite(array).sum(axis=1)
        aggregate = pd.Series(np.nan, index=values.index, dtype="float64")
        usable = counts >= 3
        aggregate.loc[usable] = (
            np.nansum(array[usable], axis=1)
            - np.nanmin(array[usable], axis=1)
            - np.nanmax(array[usable], axis=1)
        ) / (counts[usable] - 2)
    elif spec.target_kind in {"mean_family_rank", "p25_family_rank"}:
        # First rank each policy family within its decision timestamp, then
        # aggregate the seven candidate-relative ranks.  This separates
        # cross-policy robustness from the magnitude scale of one exit.
        ranks = values.groupby(_query_key(frame.loc[policy_valid]), sort=False).rank(method="average", pct=True)
        aggregate = ranks.mean(axis=1) if spec.target_kind == "mean_family_rank" else ranks.quantile(0.25, axis=1)
    else:
        raise ValueError(f"unknown target_kind: {spec.target_kind}")
    aggregate_rank = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in aggregate.to_frame("value").join(frame.__ts__).groupby("__ts__", sort=False):
        aggregate_rank.loc[group.index] = _percentile(group.value)
    return _grade_percentile(aggregate_rank).to_numpy(float)


def _label_gains(spec: PolicySpec, relevance: np.ndarray) -> list[float]:
    """Return sufficient fixed gain geometry for every label emitted by a spec."""
    finite = relevance[np.isfinite(relevance)]
    if not len(finite):
        raise ValueError("cannot derive label gains from an empty relevance array")
    largest = int(np.max(finite))
    if spec.target_kind == "policy_quantized":
        return [float(value) for value in range(largest + 1)]
    gains = GAIN_FAMILIES[spec.gain_family]
    if largest >= len(gains):
        raise ValueError(f"{spec.name} emits grade {largest} but {spec.gain_family} has only {len(gains)} gains")
    return gains


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series) -> pd.DataFrame:
    result = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians)
    if result.isna().any().any():
        raise AssertionError("training-only median imputation left null model inputs")
    return result.astype(np.float32)


def _query_order(frame: pd.DataFrame, target: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    working = frame.copy()
    working["_target"] = target
    working["__query__"] = _query_key(working)
    working = working.loc[np.isfinite(working._target)].sort_values(["__query__", "candidate_id"], kind="stable")
    count = working.groupby("__query__", sort=False).size()
    working = working.loc[working.__query__.isin(count.index[count.ge(2)])].copy()
    groups = working.groupby("__query__", sort=False).size().to_numpy(np.int32)
    if not len(working) or int(groups.sum()) != len(working):
        raise ValueError("no valid timestamp LambdaRank queries")
    return working, groups, working._target.to_numpy(np.int32)


def _sample_weights(
    ordered: pd.DataFrame, *, kind: str, train_end: pd.Timestamp, row_kind: str = "none",
) -> np.ndarray:
    """Training-only query/era weights, mean-normalised and tightly bounded."""
    if kind not in {"uniform", "equal_month", "equal_query", "month_query", "recency6m", "recency9m", "opportunity_spread", "opportunity_tercile"}:
        raise ValueError(f"unknown weight_kind: {kind}")
    if row_kind not in {"none", "economic_tail"}:
        raise ValueError(f"unknown row_weight_kind: {row_kind}")
    if ordered.empty:
        raise ValueError("cannot weight an empty training population")

    # Every component below is expressed as a *relative* per-row authority
    # before the final global normalisation.  The former implementation used
    # N / group_size for several components, multiplied them and clipped at 4
    # before normalising.  With hourly queries that made every value exceed
    # the cap, silently turning all W0--W6 variants into uniform weights.
    values = np.ones(len(ordered), dtype=np.float64)
    month = ordered["__ts__"].dt.strftime("%Y-%m")
    query = ordered["__query__"]
    query_size = query.map(query.value_counts()).to_numpy(float)
    query_count = int(query.nunique())
    query_unit = float(len(ordered)) / (float(query_count) * query_size)

    if kind == "equal_month":
        month_size = month.map(month.value_counts()).to_numpy(float)
        values = float(len(ordered)) / (float(month.nunique()) * month_size)
    elif kind == "equal_query":
        values = query_unit
    elif kind == "month_query":
        # Equal total authority per month, then equal authority per timestamp
        # query inside each month.  Rows within a query remain equally weighted.
        month_query_count = query.groupby(month, sort=False).transform("nunique").to_numpy(float)
        values = float(len(ordered)) / (float(month.nunique()) * month_query_count * query_size)
    elif kind in {"recency6m", "recency9m"}:
        half_life = 6.0 if kind == "recency6m" else 9.0
        age = (train_end - ordered["__ts__"]).dt.total_seconds().to_numpy(float) / (30.4375 * 86400.0)
        # Query-level recency authority, so a timestamp with more candidates
        # does not get more total training influence.
        values = query_unit * np.power(.5, age / half_life)
    elif kind in {"opportunity_spread", "opportunity_tercile"}:
        policy = pd.to_numeric(ordered.p0_canonical_net_bps, errors="coerce")
        spread = policy.groupby(query, sort=False).transform(lambda x: x.quantile(.90) - x.quantile(.50)).to_numpy(float)
        values = query_unit
        if kind == "opportunity_spread":
            positive = spread[spread > 0.0]
            scale = np.ones_like(spread) if not len(positive) else np.sqrt(np.maximum(spread, 0.0)) / float(np.median(np.sqrt(positive)))
            values *= np.clip(scale, .5, 2.0)
        else:
            ranks = pd.Series(spread, index=ordered.index).groupby(month, sort=False).rank(pct=True).to_numpy(float)
            values *= np.where(ranks <= .25, .5, np.where(ranks >= .75, 1.5, 1.0))

    if row_kind == "economic_tail":
        # Bounded economic relevance only: avoid spending excessive capacity
        # differentiating routine losses while retaining a modest severe-loss
        # reminder.  All values are resolved training labels, never features.
        policy = pd.to_numeric(ordered.p0_canonical_net_bps, errors="coerce").to_numpy(float)
        row_weight = np.select(
            [policy <= -400.0, policy <= 0.0, policy <= 100.0, policy <= 250.0, policy <= 500.0],
            [1.25, 1.00, 1.10, 1.30, 1.60],
            default=2.00,
        )
        values *= row_weight

    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise AssertionError(f"non-finite or non-positive {kind} sample weights")
    values /= values.mean()
    # Cap *relative* authority only after normalisation.  This preserves the
    # declared recipe while preventing a very small query or rare month from
    # dominating the LambdaRank objective.
    values = np.clip(values, .25, 2.5 if row_kind != "none" else 4.0)
    return values / values.mean()


def _fit(train: pd.DataFrame, test: pd.DataFrame, fields: list[str], spec: PolicySpec, *, train_end: pd.Timestamp, model_overrides: dict[str, Any] | None = None) -> tuple[np.ndarray, Any, dict[str, Any]]:
    train = train.copy()
    train["__query__"] = train["__ts__"].dt.floor(f"{int(spec.query_hours)}h")
    policy = pd.to_numeric(train.p0_canonical_net_bps, errors="coerce")
    valid_policy = _valid_policy(train)
    quantile_edges = _train_quantile_edges(policy, valid_policy, spec) if spec.target_kind == "train_quantile" else None
    target = _targets(train, spec, train_quantile_edges=quantile_edges)
    ordered, groups, relevance = _query_order(train, target)
    medians = ordered.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise AssertionError("policy target has a feature with no training median")
    params = dict(FROZEN_BASE_PARAMS)
    params.pop("num_class", None)
    label_gain = _label_gains(spec, relevance)
    params.update({
        "objective": spec.objective,
        "lambdarank_norm": bool(spec.lambdarank_norm),
        "lambdarank_truncation_level": int(spec.truncation),
        "label_gain": label_gain,
        "random_state": SEED, "seed": SEED,
    })
    if model_overrides:
        params.update(model_overrides)
    # P0/P1 use grades 0..6.  The 8-length gain vectors remain valid: LightGBM
    # accepts unused tail gains, which keeps Round-B geometry comparable.
    model = lgb.LGBMRanker(**params)
    weights = _sample_weights(ordered, kind=spec.weight_kind, train_end=train_end, row_kind=spec.row_weight_kind)
    model.fit(_matrix(ordered, fields, medians), relevance, group=groups, sample_weight=weights)
    score = model.predict(_matrix(test, fields, medians)).astype(np.float32)
    audit = {
        "train_rows": int(len(ordered)), "query_count": int(len(groups)),
        "query_size_p50": float(np.median(groups)), "query_size_p95": float(np.quantile(groups, .95)),
        "target_counts": {str(k): int(v) for k, v in pd.Series(relevance).value_counts().sort_index().items()},
        "objective": spec.objective,
        "lambdarank_norm": bool(spec.lambdarank_norm),
        "lambdarank_truncation_level": int(spec.truncation),
        "gain_family": spec.gain_family,
        "label_gain": [float(value) for value in label_gain],
        "query_hours": int(spec.query_hours),
        "weight_kind": spec.weight_kind,
        "row_weight_kind": spec.row_weight_kind,
        "weight_p01": float(np.quantile(weights, .01)), "weight_p50": float(np.median(weights)), "weight_p99": float(np.quantile(weights, .99)),
        "target_quantile_edges_bps": [] if quantile_edges is None else [float(value) for value in quantile_edges],
        "training_label_available_before_oos": bool((ordered.policy_label_available_at.lt(train_end) | ordered.__label_available_at__.lt(train_end)).all()),
    }
    return score, model, audit


def _policy_query_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[_valid_policy(frame)].copy()


def _within_query_scorecard(frame: pd.DataFrame, *, score_column: str, spec: str, scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _policy_query_rows(frame)
    values: list[dict[str, Any]] = []
    deciles: list[dict[str, Any]] = []
    ic: list[float] = []
    positive: list[bool] = []
    weighted: list[float] = []
    top: dict[int, list[tuple[float, float, float, int]]] = {k: [] for k in TOP_KS}
    for timestamp, group in rows.groupby("__ts__", sort=False):
        if len(group) < 2:
            continue
        score = pd.to_numeric(group[score_column], errors="coerce")
        outcome = pd.to_numeric(group.p0_canonical_net_bps, errors="coerce")
        rho = score.corr(outcome, method="spearman")
        if np.isfinite(rho):
            ic.append(float(rho)); positive.append(bool(rho > 0.0)); weighted.append(float(rho) * len(group))
        mean, median = float(outcome.mean()), float(outcome.median())
        ordered = group.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True], kind="stable")
        for k in TOP_KS:
            chosen = ordered.iloc[:min(k, len(ordered))]
            outcome_k = float(pd.to_numeric(chosen.p0_canonical_net_bps, errors="coerce").mean())
            top[k].append((outcome_k, outcome_k - mean, outcome_k - median, int(len(chosen))))
        rank = score.rank(method="average", pct=True)
        grade = np.minimum(9, np.floor(np.maximum(rank.to_numpy(float) - 1e-12, 0) * 10).astype(int))
        for decile in range(10):
            part = outcome.loc[grade == decile]
            if len(part):
                deciles.append({"spec": spec, "scope": scope, "score_form": score_column, "timestamp": timestamp, "score_decile": decile + 1, "rows": int(len(part)), "policy_net_bps": float(part.mean())})
    values.append({
        "spec": spec, "scope": scope, "score_form": score_column, "query_count": int(len(ic)),
        "policy_ic_mean": float(np.mean(ic)) if ic else float("nan"),
        "policy_ic_median": float(np.median(ic)) if ic else float("nan"),
        "policy_ic_weighted": float(sum(weighted) / sum(len(g) for _, g in rows.groupby("__ts__", sort=False) if len(g) >= 2)) if weighted else float("nan"),
        "policy_ic_positive_fraction": float(np.mean(positive)) if positive else float("nan"),
    })
    for k, result in top.items():
        values.append({
            "spec": spec, "scope": scope, "score_form": score_column, "metric": f"top_{k}_per_timestamp",
            "query_count": int(len(result)), "policy_net_bps": float(np.mean([x[0] for x in result])) if result else float("nan"),
            "uplift_vs_query_mean_bps": float(np.mean([x[1] for x in result])) if result else float("nan"),
            "uplift_vs_query_median_bps": float(np.mean([x[2] for x in result])) if result else float("nan"),
            "mean_selected_rows": float(np.mean([x[3] for x in result])) if result else float("nan"),
        })
    return values, deciles


def _global_tails(frame: pd.DataFrame, *, spec: str, scope: str) -> list[dict[str, Any]]:
    rows = _policy_query_rows(frame).sort_values("score", ascending=False, kind="stable")
    records: list[dict[str, Any]] = []
    for tail in TAILS:
        picked = rows.iloc[:max(1, int(math.ceil(len(rows) * tail)))]
        net = pd.to_numeric(picked.p0_canonical_net_bps, errors="coerce")
        p10 = float(net.quantile(.10))
        records.append({"spec": spec, "scope": scope, "tail_fraction": tail, "rows": int(len(picked)), "policy_net_bps": float(net.mean()), "policy_net_median_bps": float(net.median()), "policy_cvar10_bps": float(net.loc[net.le(p10)].mean())})
    return records


def run(*, out: Path, policies: Path, features_path: Path, candidates_path: Path, fields: list[str], train_start: pd.Timestamp, oos_start: pd.Timestamp, oos_end: pd.Timestamp, specs: tuple[PolicySpec, ...], model_overrides: dict[str, Any] | None = None, supportive_path: Path | None = None) -> Path:
    if out.exists():
        raise FileExistsError(out)
    if not (train_start < oos_start < oos_end):
        raise ValueError("invalid chronological windows")
    out.mkdir(parents=True)
    candidates = _load_candidates(candidates_path, SIDE)
    candidates = candidates.loc[candidates.__ts__.ge(train_start) & candidates.__ts__.lt(oos_end)].copy()
    features = _load_features(features_path, fields, candidates, SIDE)
    policy = _load_policy_ledger(policies, train_start, oos_end)
    ledger = features.merge(policy, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    if any(spec.target_kind in PATH_TARGET_KINDS for spec in specs):
        if supportive_path is None:
            raise ValueError("path target specs require supportive_path")
        support = _load_supportive_ledger(supportive_path, train_start, oos_end)
        ledger = ledger.merge(support, on="candidate_id", how="left", validate="one_to_one")
    if len(ledger) != len(features):
        raise AssertionError("policy-label join altered target-free candidate identities")
    train_population = ledger.loc[ledger.__ts__.lt(oos_start)]
    kept, coverage = _coverage_fields(train_population, fields)
    if set(fields).difference(kept):
        raise ValueError("policy funnel feature contract fails >=90% target-free coverage")
    train = ledger.loc[ledger.__ts__.lt(oos_start) & ledger.entry_executable.astype(bool)].copy()
    test = ledger.loc[ledger.__ts__.ge(oos_start) & ledger.entry_executable.astype(bool)].copy()
    if train.empty or test.empty:
        raise ValueError("empty policy funnel population")
    # Targets access only training labels whose H12 policy path was resolvable
    # before the held decision window.  P0 also has its H12 availability gate.
    train = train.loc[train.policy_label_available_at.lt(oos_start) | train.__label_available_at__.lt(oos_start)].copy()
    query_scorecards: list[dict[str, Any]] = []
    deciles: list[dict[str, Any]] = []
    tail_metrics: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for spec in specs:
        print(f"fitting {spec.name}", flush=True)
        score, model, audit = _fit(train, test, fields, spec, train_end=oos_start, model_overrides=model_overrides)
        prediction = test.loc[:, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "policy_path_valid", "p0_canonical_net_bps", "p0_canonical_gross_bps", "label_valid", "target_invalid", "t4_tp6_sl4_net_bps"]].copy()
        prediction["score"] = score
        prediction["timestamp_percentile"] = prediction.groupby("__ts__", sort=False).score.rank(method="average", pct=True).astype(np.float32)
        prediction.to_parquet(out / f"oos_predictions_{spec.name}.parquet", index=False, compression="zstd")
        model.booster_.save_model(str(out / f"model_{spec.name}.txt"))
        for scope, subset in [("oos", prediction), *[(month, value) for month, value in prediction.groupby(prediction.__ts__.dt.strftime("%Y-%m"), sort=True)]]:
            for column in ("score", "timestamp_percentile"):
                metrics, bins = _within_query_scorecard(subset, score_column=column, spec=spec.name, scope=scope)
                query_scorecards.extend(metrics); deciles.extend(bins)
            tail_metrics.extend(_global_tails(subset, spec=spec.name, scope=scope))
        audits[spec.name] = audit
        del model, score, prediction
        gc.collect()
    pd.DataFrame(query_scorecards).to_parquet(out / "within_timestamp_scorecard.parquet", index=False, compression="zstd")
    pd.DataFrame(deciles).to_parquet(out / "within_timestamp_deciles.parquet", index=False, compression="zstd")
    pd.DataFrame(tail_metrics).to_parquet(out / "global_policy_tail_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_policy_conversion_funnel_v1", "status": "complete", "side": SIDE,
        "train_decision_window": f"[{train_start.isoformat()}, {oos_start.isoformat()})", "oos_decision_window": f"[{oos_start.isoformat()}, {oos_end.isoformat()})",
        "policy_label_contract": "exact completed 1m; entry decision-time; H12 timeout; 100bps cost exactly once",
        "target_semantics": "within-timestamp base ranking only; absolute admission delegated downstream",
        "feature_count": len(fields), "features": fields, "feature_coverage": {name: float(coverage[name]) for name in fields},
        "policy_ledger": str(policies), "policy_manifest_sha256": _sha256(policies / "run_manifest.json"),
        "supportive_path": None if supportive_path is None else str(supportive_path),
        "features_sha256": _sha256(features_path), "candidates_sha256": _sha256(candidates_path),
        "training_rows_pre_target_filter": int(len(train)), "oos_scored_rows": int(len(test)), "oos_policy_valid_rows": int(_valid_policy(test).sum()),
        "specs": [asdict(spec) for spec in specs], "model_overrides": model_overrides or {}, "audits": audits,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected-feature-contract", type=Path, required=True)
    parser.add_argument("--train-start", default="2023-10-01T00:00:00Z")
    parser.add_argument("--oos-start", default="2024-10-01T00:00:00Z")
    parser.add_argument("--oos-end", default="2025-01-01T00:00:00Z")
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--truncation", type=int, choices=(4, 8, 16, 32))
    parser.add_argument("--gain-family", choices=tuple(GAIN_FAMILIES))
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"))
    parser.add_argument("--lambdarank-norm", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    specs = ROUND_A
    if args.spec:
        wanted = set(args.spec)
        specs = tuple(spec for spec in specs if spec.name in wanted)
        if wanted != {spec.name for spec in specs}:
            raise ValueError(f"unknown predeclared spec(s): {sorted(wanted - {spec.name for spec in specs})}")
    if any(value is not None for value in (args.truncation, args.gain_family, args.objective, args.lambdarank_norm)):
        specs = tuple(replace(
            spec,
            truncation=args.truncation or spec.truncation,
            gain_family=args.gain_family or spec.gain_family,
            objective=args.objective or spec.objective,
            lambdarank_norm=spec.lambdarank_norm if args.lambdarank_norm is None else args.lambdarank_norm,
        ) for spec in specs)
    fields, _ = _selected_short_feature_fields(args.selected_feature_contract)
    print(run(out=args.out.resolve(), policies=args.policies.resolve(), features_path=args.features.resolve(), candidates_path=args.candidates.resolve(), fields=fields, train_start=_utc(args.train_start), oos_start=_utc(args.oos_start), oos_end=_utc(args.oos_end), specs=specs))


if __name__ == "__main__":
    main()
