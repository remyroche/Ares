"""Strict-OOF diverse specialist and opportunity-synergy utilities.

The module deliberately keeps two concepts separate:

* *feature diversity* is selected from observable, pre-entry fields only;
* *specialist routing* is selected from opportunity-conditioned co-activation
  and joint predictive synergy, never unconditional score Spearman.

All scores passed to the residual layer must be out-of-fold (or produced by a
model frozen before the scored row).  This file has no implicit global fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


FORBIDDEN_TOKENS = (
    "target", "label", "outcome", "future", "post_entry", "postentry",
    "realized", "realised", "pnl", "mfe", "mae", "barrier", "timeout",
    "exit", "event", "net_bps", "gross_bps",
)


@dataclass(frozen=True)
class SpecialistView:
    name: str
    tokens: tuple[str, ...]
    max_features: int = 10


DEFAULT_SPECIALIST_VIEWS: tuple[SpecialistView, ...] = (
    SpecialistView("trend_price", ("atr", "ema", "momentum", "slope", "efficiency", "chop", "compression", "breakout", "range")),
    SpecialistView("breadth_dependence", ("breadth", "dispersion", "cross_asset", "correlation", "corr", "pc1", "eigen", "decoupling", "coherence")),
    SpecialistView("leverage_flow", ("fund", "funding", "oi", "open_interest", "liquidation", "deleverag", "basis", "crowd", "carry")),
    SpecialistView("liquidity_impact", ("liquid", "spread", "depth", "amihud", "amivest", "impact", "volume", "turnover", "orderbook")),
    SpecialistView("structural_price", ("support", "resistance", "wick", "close_location", "session_pos", "initial_balance", "range_pos", "value_area")),
)


@dataclass(frozen=True)
class SynergyConfig:
    opportunity_quantile: float = 0.80
    specialist_quantile: float = 0.80
    min_joint_rows: int = 300
    min_lift: float = 0.0
    min_synergy: float = 0.0


def is_permitted_feature(name: str) -> bool:
    lower = str(name).lower()
    return not any(token in lower for token in FORBIDDEN_TOKENS)


def permitted_causal_features(
    columns: Iterable[str], *, causal_allowlist: Iterable[str]
) -> list[str]:
    """Return an ordered, registry-backed set of admissible raw features.

    A parquet schema is not a causal contract: it can retain historical labels,
    model predictions, and abandoned repair columns.  Callers must provide the
    current generator-registry allow-list, then this function adds a defensive
    name-level rejection for obvious outcome fields.
    """
    allowed = {str(name) for name in causal_allowlist}
    return [
        str(name) for name in columns
        if str(name) in allowed and is_permitted_feature(str(name))
    ]


def candidate_view_columns(columns: Iterable[str], views: Sequence[SpecialistView] = DEFAULT_SPECIALIST_VIEWS, *, max_candidates_per_view: int = 36) -> dict[str, list[str]]:
    """Return deterministic, non-outcome candidate pools for each view."""
    allowed = [str(column) for column in columns if is_permitted_feature(str(column))]
    result: dict[str, list[str]] = {}
    already_assigned: set[str] = set()
    for view in views:
        matches = [
            column for column in allowed
            if column not in already_assigned and any(token in column.lower() for token in view.tokens)
        ]
        # A raw observable is more interpretable and portable than an opaque
        # high-order composite when both cover the same geometry.
        matches.sort(key=lambda value: (value.count("__") + value.count("_"), value))
        chosen = matches[: int(max_candidates_per_view)]
        result[view.name] = chosen
        already_assigned.update(chosen)
    return result


def select_diverse_features(train: pd.DataFrame, candidate_columns: Mapping[str, Sequence[str]], views: Sequence[SpecialistView] = DEFAULT_SPECIALIST_VIEWS, *, min_coverage: float = .90, correlation_limit: float = .88) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Select stable, mutually non-redundant fields within each view.

    This is deliberately label-free.  Specialist relevance is evaluated later
    by the opportunity-conditioned synergy protocol rather than a univariate
    outcome statistic such as Spearman.
    """
    selected: dict[str, list[str]] = {}
    records: list[dict[str, object]] = []
    used: set[str] = set()
    cap_by_view = {view.name: int(view.max_features) for view in views}
    for view in views:
        pool = [column for column in candidate_columns.get(view.name, ()) if column in train and column not in used]
        stats: list[tuple[str, float, float]] = []
        for column in pool:
            value = pd.to_numeric(train[column], errors="coerce").to_numpy(float)
            coverage = float(np.isfinite(value).mean())
            scale = float(np.nanmedian(np.abs(value - np.nanmedian(value))) * 1.4826) if coverage else 0.0
            if coverage >= min_coverage and np.isfinite(scale) and scale > 1e-8:
                stats.append((column, coverage, scale))
            else:
                records.append({"view": view.name, "feature": column, "coverage": coverage, "robust_scale": scale, "selected": False, "reason": "coverage_or_variance"})
        stats.sort(key=lambda item: (-item[1], -item[2], item[0]))
        kept: list[str] = []
        for column, coverage, scale in stats:
            if len(kept) >= cap_by_view[view.name]:
                records.append({"view": view.name, "feature": column, "coverage": coverage, "robust_scale": scale, "selected": False, "reason": "view_cap"})
                continue
            if kept:
                matrix = train.loc[:, [*kept, column]].apply(pd.to_numeric, errors="coerce")
                corr = matrix.corr(method="spearman").iloc[-1, :-1].abs().max()
                if np.isfinite(corr) and float(corr) >= correlation_limit:
                    records.append({"view": view.name, "feature": column, "coverage": coverage, "robust_scale": scale, "selected": False, "reason": "within_view_redundant"})
                    continue
            kept.append(column); used.add(column)
            records.append({"view": view.name, "feature": column, "coverage": coverage, "robust_scale": scale, "selected": True, "reason": "diverse_observable"})
        selected[view.name] = kept
    return selected, pd.DataFrame(records)


def opportunity_conditioned_synergy(frame: pd.DataFrame, score_columns: Mapping[str, str], *, base_score_column: str, label_column: str, config: SynergyConfig = SynergyConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-only routing diagnostics from opportunity co-activation.

    A pair advances only when, *inside rows the base considers opportunities*,
    both specialists activate together with adequate support and their mean
    score adds predictive separation beyond the better individual specialist.
    Unconditional correlation is intentionally absent from this criterion.
    """
    required = [base_score_column, label_column, *score_columns.values()]
    if frame.loc[:, required].isna().any().any():
        raise ValueError("synergy fit requires finite, aligned train-only scores and labels")
    base = frame[base_score_column].to_numpy(float)
    label = frame[label_column].to_numpy(float)
    opportunity_threshold = float(np.quantile(base, config.opportunity_quantile))
    opportunity = base >= opportunity_threshold
    score_arrays = {name: frame[column].to_numpy(float) for name, column in score_columns.items()}
    thresholds = {name: float(np.quantile(score, config.specialist_quantile)) for name, score in score_arrays.items()}
    rows: list[dict[str, object]] = []
    pair_features = pd.DataFrame(index=frame.index)
    baseline = float(label[opportunity].mean()) if opportunity.any() else np.nan
    for left, right in combinations(score_arrays, 2):
        left_score, right_score = score_arrays[left], score_arrays[right]
        coactive = opportunity & (left_score >= thresholds[left]) & (right_score >= thresholds[right])
        count = int(coactive.sum())
        coactivation_rate = float(count / max(int(opportunity.sum()), 1))
        lift = float(label[coactive].mean() - baseline) if count else np.nan
        joint = .5 * (left_score + right_score)
        # Rank separation is calculated only on the same opportunity support.
        support = opportunity
        if support.sum() >= 2 and np.unique(label[support]).size > 1:
            joint_corr = float(pd.Series(joint[support]).corr(pd.Series(label[support]), method="spearman"))
            left_corr = float(pd.Series(left_score[support]).corr(pd.Series(label[support]), method="spearman"))
            right_corr = float(pd.Series(right_score[support]).corr(pd.Series(label[support]), method="spearman"))
            synergy = joint_corr - max(left_corr, right_corr)
        else:
            synergy = np.nan
        selected = bool(count >= config.min_joint_rows and np.isfinite(lift) and np.isfinite(synergy) and lift >= config.min_lift and synergy >= config.min_synergy)
        key = f"mv_pair__{left}__{right}"
        pair_features[f"{key}__coactive"] = coactive.astype(np.float32)
        pair_features[f"{key}__joint_score"] = joint.astype(np.float32)
        rows.append({"left": left, "right": right, "opportunity_threshold": opportunity_threshold, "left_threshold": thresholds[left], "right_threshold": thresholds[right], "opportunity_rows": int(opportunity.sum()), "coactive_rows": count, "coactivation_rate": coactivation_rate, "conditional_event_lift": lift, "joint_synergy": synergy, "selected_for_router": selected})
    return pd.DataFrame(rows), pair_features


def apply_synergy_features(frame: pd.DataFrame, score_columns: Mapping[str, str], diagnostics: pd.DataFrame, *, base_score_column: str) -> pd.DataFrame:
    """Apply only previously fitted co-activation thresholds to later rows."""
    result = pd.DataFrame(index=frame.index)
    for row in diagnostics.itertuples(index=False):
        if not bool(row.selected_for_router):
            continue
        left, right = str(row.left), str(row.right)
        key = f"mv_pair__{left}__{right}"
        coactive = (frame[base_score_column].to_numpy(float) >= float(row.opportunity_threshold)) & (frame[score_columns[left]].to_numpy(float) >= float(row.left_threshold)) & (frame[score_columns[right]].to_numpy(float) >= float(row.right_threshold))
        result[f"{key}__coactive"] = coactive.astype(np.float32)
        result[f"{key}__joint_score"] = (.5 * (frame[score_columns[left]].to_numpy(float) + frame[score_columns[right]].to_numpy(float))).astype(np.float32)
    return result


def discover_opportunity_views(train: pd.DataFrame, feature_columns: Sequence[str], *, base_score_column: str, label_column: str, max_proxy_features: int = 120, max_views: int = 5, max_features_per_view: int = 10, opportunity_quantile: float = .80, activation_quantile: float = .80, min_joint_rows: int = 150) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Discover specialist views from train-only opportunity co-activation.

    No semantic family or predeclared economic label participates in the view
    assignment.  First each raw field gets an orientation (upper or lower
    activation) from its *conditional event lift* inside the base-opportunity
    set.  We then form views from pairs whose joint activation has additional
    event lift beyond either feature alone.  Ordinary feature correlation is
    used only as a redundancy veto after a view has been discovered.
    """
    base = train[base_score_column].to_numpy(float)
    label = train[label_column].to_numpy(float)
    opportunity = base >= float(np.quantile(base, opportunity_quantile))
    if int(opportunity.sum()) < min_joint_rows or np.unique(label[opportunity]).size < 2:
        raise ValueError("insufficient opportunity-conditioned support for view discovery")
    baseline = float(label[opportunity].mean())
    candidates: list[dict[str, object]] = []
    activations: dict[str, np.ndarray] = {}
    for feature in sorted(dict.fromkeys(map(str, feature_columns))):
        if feature not in train or not is_permitted_feature(feature):
            continue
        value = pd.to_numeric(train[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(value)
        if finite.mean() < .90:
            continue
        median = float(np.nanmedian(value)); scale = float(np.nanmedian(np.abs(value - median)) * 1.4826)
        if not np.isfinite(scale) or scale <= 1e-8:
            continue
        high_threshold, low_threshold = np.nanquantile(value, (activation_quantile, 1. - activation_quantile))
        high = finite & (value >= high_threshold); low = finite & (value <= low_threshold)
        high_lift = float(label[opportunity & high].mean() - baseline) if int((opportunity & high).sum()) >= min_joint_rows else -np.inf
        low_lift = float(label[opportunity & low].mean() - baseline) if int((opportunity & low).sum()) >= min_joint_rows else -np.inf
        upper = high_lift >= low_lift
        activation = high if upper else low
        lift = max(high_lift, low_lift)
        if not np.isfinite(lift):
            continue
        candidates.append({"feature": feature, "coverage": float(finite.mean()), "robust_scale": scale, "orientation": "upper" if upper else "lower", "activation_lift": lift, "activation_rows": int((opportunity & activation).sum())})
        activations[feature] = activation
    candidate_frame = pd.DataFrame(candidates).sort_values(["activation_lift", "feature"], ascending=[False, True], kind="stable").head(int(max_proxy_features)).reset_index(drop=True)
    if len(candidate_frame) < 2:
        raise ValueError("fewer than two usable opportunity-activated fields")
    edges: list[dict[str, object]] = []
    for left, right in combinations(candidate_frame.feature.tolist(), 2):
        both = opportunity & activations[left] & activations[right]
        count = int(both.sum())
        if count < min_joint_rows:
            continue
        joint_lift = float(label[both].mean() - baseline)
        left_lift = float(candidate_frame.loc[candidate_frame.feature.eq(left), "activation_lift"].iloc[0])
        right_lift = float(candidate_frame.loc[candidate_frame.feature.eq(right), "activation_lift"].iloc[0])
        synergy = joint_lift - max(left_lift, right_lift)
        edges.append({"left": left, "right": right, "coactive_rows": count, "coactivation_rate": float(count / opportunity.sum()), "joint_lift": joint_lift, "joint_synergy": synergy, "selected_for_view": False})
    edge_frame = pd.DataFrame(edges)
    views: dict[str, list[str]] = {}
    claimed: set[str] = set()
    if not edge_frame.empty:
        ranked = edge_frame.sort_values(["joint_synergy", "joint_lift", "coactive_rows", "left", "right"], ascending=[False, False, False, True, True], kind="stable")
        for row in ranked.itertuples():
            if len(views) >= max_views:
                break
            if row.joint_synergy <= 0. or row.joint_lift <= 0. or row.left in claimed or row.right in claimed:
                continue
            members = [str(row.left), str(row.right)]
            # Add non-redundant neighbours with positive pair synergy.
            neighbours = ranked.loc[((ranked.left.eq(row.left)) | (ranked.right.eq(row.left)) | (ranked.left.eq(row.right)) | (ranked.right.eq(row.right))) & ranked.joint_synergy.gt(0.)]
            for neighbour in neighbours.itertuples():
                candidate = str(neighbour.right if neighbour.left in members else neighbour.left)
                if candidate in claimed or candidate in members or len(members) >= max_features_per_view:
                    continue
                correlation = train.loc[:, [*members, candidate]].corr(method="spearman").iloc[-1, :-1].abs().max()
                if not np.isfinite(correlation) or correlation < .88:
                    members.append(candidate)
            view_name = f"data_view_{len(views):02d}"
            views[view_name] = members
            claimed.update(members)
            edge_frame.loc[(edge_frame.left.eq(row.left)) & (edge_frame.right.eq(row.right)), "selected_for_view"] = True
    # Isolated but conditionally useful activations are retained only when the
    # synergy graph has insufficient connected components.  This fallback is
    # explicit in the audit and never disguises an individual field as a
    # co-activation-discovered view.
    for row in candidate_frame.itertuples():
        if len(views) >= max_views:
            break
        if row.feature in claimed or row.activation_lift <= 0.:
            continue
        views[f"data_view_{len(views):02d}"] = [str(row.feature)]
        claimed.add(str(row.feature))
    candidate_frame["assigned_view"] = pd.NA
    for view, fields in views.items():
        candidate_frame.loc[candidate_frame.feature.isin(fields), "assigned_view"] = view
    return views, candidate_frame, edge_frame


def discover_broad_opportunity_views(train: pd.DataFrame, feature_columns: Sequence[str], *, base_score_column: str, label_column: str, specialist_count: int = 8, min_features_per_view: int = 40, max_features_per_view: int = 80, max_proxy_features: int = 480, opportunity_quantile: float = .80, activation_quantile: float = .80, min_joint_rows: int = 100, random_state: int = 20260810) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Discover 7--10 broad, diverse specialist views without semantic labels.

    The representation of each field is its *opportunity-conditioned
    activation pattern*, not its name or unconditional correlation.  A small
    random projection makes co-activation clustering tractable on a bounded
    train-only proxy.  Oversized clusters are represented by one feature per
    activation sub-cluster, preserving breadth without allowing a 200-field
    specialist to dominate the ensemble.
    """
    base = train[base_score_column].to_numpy(float)
    label = train[label_column].to_numpy(float)
    opportunity = base >= float(np.quantile(base, opportunity_quantile))
    opp_rows = np.flatnonzero(opportunity)
    if len(opp_rows) < min_joint_rows or np.unique(label[opp_rows]).size < 2:
        raise ValueError("insufficient opportunity-conditioned support for broad view discovery")
    if len(opp_rows) > 12_000:
        opp_rows = np.random.default_rng(random_state).choice(opp_rows, 12_000, replace=False)
    baseline = float(label[opportunity].mean())
    activations: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for feature in sorted(dict.fromkeys(map(str, feature_columns))):
        if feature not in train or not is_permitted_feature(feature):
            continue
        value = pd.to_numeric(train[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(value)
        if finite.mean() < .90:
            continue
        median = float(np.nanmedian(value)); scale = float(np.nanmedian(np.abs(value - median)) * 1.4826)
        if not np.isfinite(scale) or scale <= 1e-8:
            continue
        high_threshold, low_threshold = np.nanquantile(value, (activation_quantile, 1. - activation_quantile))
        high, low = finite & (value >= high_threshold), finite & (value <= low_threshold)
        high_mask, low_mask = opportunity & high, opportunity & low
        high_lift = float(label[high_mask].mean() - baseline) if high_mask.sum() >= min_joint_rows else -np.inf
        low_lift = float(label[low_mask].mean() - baseline) if low_mask.sum() >= min_joint_rows else -np.inf
        upper = high_lift >= low_lift
        activation = high if upper else low
        lift = max(high_lift, low_lift)
        # Broad views are learned from activation co-patterns.  Requiring every
        # member to have positive marginal lift would collapse the causal
        # universe and rule out genuinely synergistic fields.  Retain any
        # coverage-valid field with a measurable orientation; lift remains a
        # ranking/diagnostic, never an admission shortcut.
        if not np.isfinite(lift):
            continue
        activations[feature] = activation
        rows.append({"feature": feature, "coverage": float(finite.mean()), "robust_scale": scale, "orientation": "upper" if upper else "lower", "activation_lift": lift, "activation_rows": int((opportunity & activation).sum())})
    fields = pd.DataFrame(rows).sort_values(["activation_lift", "feature"], ascending=[False, True], kind="stable").head(max_proxy_features).reset_index(drop=True)
    if len(fields) < specialist_count * min_features_per_view:
        raise ValueError("insufficient causal fields for the requested broad specialists")
    names = fields.feature.tolist()
    activation_matrix = np.column_stack([activations[name][opp_rows] for name in names]).astype(np.float32)
    # Sketch patterns rather than computing an O(n_features^2 x n_rows) table.
    projection = np.random.default_rng(random_state + 1).normal(0., 1., size=(len(opp_rows), 16)).astype(np.float32)
    embedding = (activation_matrix.T @ projection) / np.sqrt(float(len(opp_rows)))
    clustering = MiniBatchKMeans(n_clusters=specialist_count, random_state=random_state, n_init=5, batch_size=256).fit(embedding)
    groups = clustering.labels_.astype(int)
    # Mini-batch clustering can leave a small component below the contractual
    # 40-field minimum.  Reassign the closest donor members, retaining a
    # data-driven activation-space assignment rather than falling back to a
    # semantic feature bucket.
    for recipient in range(specialist_count):
        while int((groups == recipient).sum()) < min_features_per_view:
            counts = np.bincount(groups, minlength=specialist_count)
            donor = int(np.argmax(counts))
            if donor == recipient or counts[donor] <= min_features_per_view:
                raise ValueError("cannot rebalance broad activation clusters to the requested minimum size")
            donor_rows = np.flatnonzero(groups == donor)
            distance = np.sum((embedding[donor_rows] - clustering.cluster_centers_[recipient]) ** 2, axis=1)
            groups[int(donor_rows[int(np.argmin(distance))])] = recipient
    fields["activation_cluster"] = groups.astype(int)
    views: dict[str, list[str]] = {}
    for cluster in range(specialist_count):
        local = fields.loc[fields.activation_cluster.eq(cluster)].sort_values(["activation_lift", "feature"], ascending=[False, True], kind="stable")
        # Rebalance sparse clusters from the nearest unassigned high-lift
        # fields only if necessary; this preserves the 40-feature contract.
        if len(local) < min_features_per_view:
            raise ValueError(f"data-discovered activation cluster {cluster} has only {len(local)} fields")
        if len(local) > max_features_per_view:
            k = int(np.ceil(len(local) / max_features_per_view))
            local_embedding = embedding[local.index.to_numpy()]
            sub = MiniBatchKMeans(n_clusters=k, random_state=random_state + cluster + 2, n_init=3, batch_size=128).fit_predict(local_embedding)
            candidates: list[int] = []
            for subcluster in range(k):
                members = local.loc[sub == subcluster]
                candidates.extend(members.index[: int(np.ceil(max_features_per_view / k))].tolist())
            local = local.loc[sorted(set(candidates))].sort_values(["activation_lift", "feature"], ascending=[False, True], kind="stable").head(max_features_per_view)
        view = f"data_view_{cluster:02d}"
        views[view] = local.feature.tolist()
        fields.loc[fields.feature.isin(views[view]), "assigned_view"] = view
    edge_rows: list[dict[str, object]] = []
    for left, right in combinations(views, 2):
        left_active = np.any(np.column_stack([activations[field] for field in views[left]]), axis=1)
        right_active = np.any(np.column_stack([activations[field] for field in views[right]]), axis=1)
        both = opportunity & left_active & right_active
        n = int(both.sum())
        joint_lift = float(label[both].mean() - baseline) if n else np.nan
        left_lift = float(label[opportunity & left_active].mean() - baseline)
        right_lift = float(label[opportunity & right_active].mean() - baseline)
        synergy = joint_lift - max(left_lift, right_lift) if n >= min_joint_rows else np.nan
        edge_rows.append({"left": left, "right": right, "coactive_rows": n, "coactivation_rate": float(n / max(opportunity.sum(), 1)), "joint_lift": joint_lift, "joint_synergy": synergy, "selected_for_view": bool(n >= min_joint_rows and np.isfinite(synergy) and synergy > 0.)})
    return views, fields, pd.DataFrame(edge_rows)


__all__ = [
    "DEFAULT_SPECIALIST_VIEWS", "FORBIDDEN_TOKENS", "SpecialistView", "SynergyConfig",
    "apply_synergy_features", "candidate_view_columns", "is_permitted_feature",
    "discover_broad_opportunity_views", "discover_opportunity_views", "opportunity_conditioned_synergy", "select_diverse_features",
]
