"""Training-safe correctness-leaf regime representations.

The module is deliberately target-agnostic after rule discovery: a caller
fits shallow correctness trees on a chronological training partition, exports
``LeafRule`` objects, aligns only those rules on a *training-safe reference*
population, and applies the frozen medoids later.  It contains no fitting on
held-out labels and no period-level target construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp


EPS = 1e-8


@dataclass(frozen=True)
class LeafRule:
    """A shallow-tree leaf rule in a fold-local robust coordinate system."""

    rule_id: str
    conditions: tuple[tuple[str, int, float], ...]  # (feature, +1/> or -1/<, threshold)
    economic_effect: float
    weight: float = 1.0


@dataclass(frozen=True)
class RuleSimilarity:
    left: str
    right: str
    feature_set: float
    threshold_profile: float
    membership_correlation: float
    economic_signature: float
    total: float
    hard_gate_pass: bool
    reason: str


def _bounds(rule: LeafRule) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for feature, direction, threshold in rule.conditions:
        low, high = result.get(feature, (-np.inf, np.inf))
        if int(direction) > 0:
            low = max(low, float(threshold))
        else:
            high = min(high, float(threshold))
        result[str(feature)] = (low, high)
    return result


def _mechanisms(rule: LeafRule) -> set[str]:
    """Small named mechanism vocabulary; fallback keeps unrelated fields apart."""
    vocabulary = {
        "trend": ("trend", "ret", "momentum", "slope", "efficiency"),
        "volatility": ("vol", "atr", "range", "rv"),
        "breadth": ("breadth", "dispersion", "correlation", "cross"),
        "leverage": ("fund", "oi", "basis", "liquidat", "delever"),
        "liquidity": ("spread", "depth", "volume", "turnover", "impact"),
        "trust": ("trust", "drift", "ood", "uncert", "entropy", "margin"),
    }
    result: set[str] = set()
    for feature, _direction, _threshold in rule.conditions:
        lower = feature.lower()
        hit = {name for name, tokens in vocabulary.items() if any(token in lower for token in tokens)}
        result.update(hit or {f"feature:{feature}"})
    return result


def soft_rule_membership(frame: pd.DataFrame, rule: LeafRule, *, temperature: float = 0.35) -> np.ndarray:
    """Independent directional sigmoid terms combined geometrically."""
    terms = []
    for feature, direction, threshold in rule.conditions:
        value = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        signed = int(direction) * (value - float(threshold)) / max(float(temperature), EPS)
        terms.append(np.clip(expit(np.clip(signed, -30.0, 30.0)), EPS, 1.0))
    if not terms:
        return np.zeros(len(frame), dtype=np.float32)
    return np.exp(np.mean(np.log(np.vstack(terms)), axis=0)).astype(np.float32)


def rule_similarity(
    left: LeafRule,
    right: LeafRule,
    *,
    left_membership: Sequence[float],
    right_membership: Sequence[float],
    minimum_membership_correlation: float = 0.45,
    minimum_shared_mechanisms: int = 1,
) -> RuleSimilarity:
    """Score rules and enforce the requested semantic hard gates."""
    lb, rb = _bounds(left), _bounds(right)
    shared = set(lb) & set(rb)
    feature_set = len(shared) / max(len(set(lb) | set(rb)), 1)
    # A feature cannot point in opposing directions without a common feasible
    # interval.  Nor can two defining intervals be disjoint.
    for feature in shared:
        l_low, l_high = lb[feature]; r_low, r_high = rb[feature]
        if max(l_low, r_low) > min(l_high, r_high):
            return RuleSimilarity(left.rule_id, right.rule_id, feature_set, 0., 0., 0., 0., False, "disjoint_defining_interval")
    left_sign = {(f, d) for f, d, _t in left.conditions}
    right_sign = {(f, d) for f, d, _t in right.conditions}
    if any((feature, -direction) in right_sign for feature, direction in left_sign):
        return RuleSimilarity(left.rule_id, right.rule_id, feature_set, 0., 0., 0., 0., False, "conflicting_direction")
    mechanisms = len(_mechanisms(left) & _mechanisms(right))
    if mechanisms < int(minimum_shared_mechanisms):
        return RuleSimilarity(left.rule_id, right.rule_id, feature_set, 0., 0., 0., 0., False, "no_shared_mechanism")
    if shared:
        profile = []
        for feature in shared:
            l_low, l_high = lb[feature]; r_low, r_high = rb[feature]
            distances = [abs(a - b) for a, b in ((l_low, r_low), (l_high, r_high)) if np.isfinite(a) and np.isfinite(b)]
            profile.append(float(np.exp(-np.mean(distances))) if distances else 1.0)
        threshold_profile = float(np.mean(profile))
    else:
        threshold_profile = 0.0
    lm = np.asarray(left_membership, dtype=float); rm = np.asarray(right_membership, dtype=float)
    valid = np.isfinite(lm) & np.isfinite(rm)
    if valid.sum() < 20 or np.std(lm[valid]) < EPS or np.std(rm[valid]) < EPS:
        corr = 0.0
    else:
        corr = float(np.corrcoef(lm[valid], rm[valid])[0, 1])
    same_direction = np.sign(left.economic_effect) == np.sign(right.economic_effect) and np.sign(left.economic_effect) != 0
    economic = float(np.exp(-abs(abs(left.economic_effect) - abs(right.economic_effect)) / max(np.mean(np.abs([left.economic_effect, right.economic_effect])), EPS))) if same_direction else 0.0
    passed = bool(corr >= minimum_membership_correlation and same_direction)
    total = 0.30 * feature_set + 0.25 * threshold_profile + 0.25 * max(corr, 0.0) + 0.20 * economic if passed else 0.0
    return RuleSimilarity(left.rule_id, right.rule_id, feature_set, threshold_profile, corr, economic, float(total), passed, "ok" if passed else "membership_or_economic_gate")


def cluster_rules(
    rules: Sequence[LeafRule],
    memberships: Mapping[str, Sequence[float]],
    *,
    minimum_similarity: float = 0.70,
) -> tuple[list[list[str]], pd.DataFrame]:
    """Average-linkage clustering at the declared threshold.

    Logical/economic hard gates still apply to *every* inter-cluster pair, but
    0.70 is now the average-linkage criterion rather than an unnecessary
    all-pairs numerical similarity requirement.  This permits a coherent
    cluster with one weaker (but non-conflicting) satellite rule.
    """
    rules = list(rules); by_id = {rule.rule_id: rule for rule in rules}; pairs: list[RuleSimilarity] = []
    for position, left in enumerate(rules):
        for right in rules[position + 1:]:
            pairs.append(rule_similarity(left, right, left_membership=memberships[left.rule_id], right_membership=memberships[right.rule_id]))
    table = pd.DataFrame([item.__dict__ for item in pairs])
    lookup = {(row.left, row.right): row for row in pairs} | {(row.right, row.left): row for row in pairs}
    clusters = [[rule.rule_id] for rule in rules]
    while True:
        candidate: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                values = [lookup[(a, b)] for a in clusters[i] for b in clusters[j]]
                if not values or any(not item.hard_gate_pass for item in values):
                    continue
                mean = float(np.mean([item.total for item in values]))
                if mean < minimum_similarity:
                    continue
                tie = (mean, -i, -j)
                if candidate is None or tie > (candidate[0], -candidate[1], -candidate[2]):
                    candidate = (mean, i, j)
        if candidate is None:
            break
        _mean, i, j = candidate
        clusters[i] = sorted([*clusters[i], *clusters[j]])
        clusters.pop(j)
    return clusters, table


def medoid(cluster: Sequence[str], similarity: pd.DataFrame) -> str:
    """Rule with maximal mean pairwise similarity within a retained cluster."""
    cluster = list(cluster)
    if len(cluster) == 1:
        return cluster[0]
    values: dict[str, list[float]] = {name: [] for name in cluster}
    for row in similarity.itertuples(index=False):
        if row.left in values and row.right in values:
            values[row.left].append(float(row.total)); values[row.right].append(float(row.total))
    return sorted(cluster, key=lambda name: (-float(np.mean(values[name])) if values[name] else np.inf, name))[0]


def aggregate_membership(values: np.ndarray, weights: Sequence[float], *, mode: str, generalized_p: float = -2.0, softmin_temperature: float = 0.10) -> np.ndarray:
    """G0--G3 posterior alternatives for a cluster's frozen rules."""
    x = np.clip(np.asarray(values, dtype=float), EPS, 1.0)
    if x.ndim != 2:
        raise ValueError("membership matrix must be rules × rows")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != x.shape[0] or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("membership weights must be aligned non-negative values")
    w = w / w.sum()
    if mode == "G0_geometric":
        result = np.exp(np.mean(np.log(x), axis=0))
    elif mode == "G1_weighted_geometric":
        result = np.exp(np.sum(w[:, None] * np.log(x), axis=0))
    elif mode == "G2_generalized_pminus2":
        result = np.power(np.sum(w[:, None] * np.power(x, generalized_p), axis=0), 1.0 / generalized_p)
    elif mode == "G3_softmin":
        tau = max(float(softmin_temperature), EPS)
        result = -tau * (logsumexp(-x / tau, axis=0, b=w[:, None]) - np.log(w.sum()))
    else:
        raise ValueError(f"unknown membership aggregation mode: {mode}")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def membership_dynamics(frame: pd.DataFrame, membership_columns: Sequence[str], *, timestamp_column: str = "__ts__", group_columns: Sequence[str] = ("side_name",), half_life_hours: float = 3.0) -> pd.DataFrame:
    """Causal membership velocities, activation duration and concentration."""
    out = frame.copy(); out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    keys = list(group_columns); ordered = out.sort_values([*keys, timestamp_column], kind="stable")
    for name in membership_columns:
        group = ordered.groupby(keys, observed=True)[name]
        ordered[f"{name}__velocity_1h"] = group.diff().fillna(0.).astype(np.float32)
        ordered[f"{name}__velocity_3h"] = group.diff(3).div(3.).fillna(0.).astype(np.float32)
        ordered[f"{name}__acceleration"] = (ordered[f"{name}__velocity_1h"] - ordered[f"{name}__velocity_3h"]).astype(np.float32)
        ordered[f"{name}__smoothed_membership"] = group.transform(lambda x: x.ewm(halflife=float(half_life_hours), adjust=False).mean()).astype(np.float32)
        for threshold, suffix in ((.80, "p80"), (.60, "p60")):
            active = ordered[name].ge(threshold)
            run = active.groupby([ordered[key] for key in keys], observed=True).transform(lambda x: x.astype(int).groupby((~x).cumsum()).cumsum())
            ordered[f"{name}__hours_active_above_{suffix}"] = run.astype(np.float32)
    mass = ordered.loc[:, list(membership_columns)].to_numpy(float).sum(axis=1)
    q = np.divide(ordered.loc[:, list(membership_columns)].to_numpy(float), mass[:, None], out=np.zeros((len(ordered), len(membership_columns))), where=mass[:, None] > EPS)
    ordered["activation_mass"] = mass.astype(np.float32)
    ordered["activation_entropy"] = (-np.sum(np.where(q > EPS, q * np.log(np.maximum(q, EPS)), 0.), axis=1)).astype(np.float32)
    return ordered.reindex(frame.index)


def cluster_state_dynamics(
    frame: pd.DataFrame,
    membership_columns: Sequence[str],
    *,
    timestamp_column: str = "__ts__",
    group_columns: Sequence[str] = ("side_name",),
    switch_half_life_hours: float = 24.0,
) -> pd.DataFrame:
    """Causal summary of the *relative* leaf-cluster activation surface.

    A leaf cluster is not a fitted market-regime model.  These fields are
    therefore deliberately named ``cluster_state_*`` rather than ``regime_*``.
    They expose the information needed by the meta layer beyond a single
    cluster posterior: concentration, dominance, state persistence and the
    recent probability of a dominant-cluster switch.  Every value uses the
    current and earlier decision-time activations only.
    """
    if not membership_columns:
        return frame.copy()
    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    keys = list(group_columns)
    # Candidate-level memberships are reduced to a side/timestamp surface
    # before dynamics are calculated.  This prevents candidate count from
    # changing the definition of a market-time state.
    state = (
        out.groupby([*keys, timestamp_column], observed=True)[list(membership_columns)]
        .mean()
        .reset_index()
        .sort_values([*keys, timestamp_column], kind="stable")
    )
    values = state.loc[:, list(membership_columns)].to_numpy(dtype=float)
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    mass = values.sum(axis=1)
    posterior = np.divide(
        values,
        mass[:, None],
        out=np.zeros_like(values),
        where=mass[:, None] > EPS,
    )
    ranked = np.sort(posterior, axis=1)
    top1 = ranked[:, -1]
    top2 = ranked[:, -2] if posterior.shape[1] > 1 else np.zeros(len(state), dtype=float)
    dominant = np.argmax(posterior, axis=1).astype(np.int16)
    # A completely inactive surface has no meaningful dominant state.
    dominant[mass <= EPS] = -1
    state["cluster_state_activation_mass"] = mass.astype(np.float32)
    state["cluster_state_entropy"] = (
        -np.sum(np.where(posterior > EPS, posterior * np.log(np.maximum(posterior, EPS)), 0.0), axis=1)
    ).astype(np.float32)
    state["cluster_state_top1_probability"] = top1.astype(np.float32)
    state["cluster_state_top2_margin"] = (top1 - top2).astype(np.float32)
    state["cluster_state_dominant_id"] = dominant

    ordered_groups = state.groupby(keys, observed=True, sort=False)
    previous = ordered_groups["cluster_state_dominant_id"].shift(1)
    switched = (
        previous.notna()
        & state["cluster_state_dominant_id"].ne(previous)
        & state["cluster_state_dominant_id"].ge(0)
        & previous.ge(0)
    )
    state["cluster_state_switch"] = switched.astype(np.float32)
    # Shift one step before smoothing: this is an explicitly prequential
    # switch-rate estimate, not a label and not a future-looking rate.
    state["cluster_state_switch_probability"] = (
        ordered_groups["cluster_state_switch"]
        .transform(lambda x: x.shift(1).ewm(halflife=float(switch_half_life_hours), adjust=False).mean())
        .fillna(0.0)
        .astype(np.float32)
    )
    run_id = state["cluster_state_dominant_id"].ne(previous).groupby(
        [state[key] for key in keys], observed=True
    ).cumsum()
    state["cluster_state_age_hours"] = (
        state.groupby([*keys, run_id], observed=True).cumcount() + 1
    ).where(state["cluster_state_dominant_id"].ge(0), 0).astype(np.float32)
    return out.merge(state, on=[*keys, timestamp_column], how="left", validate="many_to_one")


__all__ = ["LeafRule", "RuleSimilarity", "aggregate_membership", "cluster_rules", "cluster_state_dynamics", "medoid", "membership_dynamics", "rule_similarity", "soft_rule_membership"]
