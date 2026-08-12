"""Signature-only G2/G3 leaf-reasoning clustering.

Raw LightGBM leaf IDs/tokens are deliberately rejected here.  They are local
to one model/tree/fold and have no cross-fold meaning.  The input contract is
instead one pre-aggregated row per *rule instance*, using a G2 structural rule
signature plus G1 leaf-health and G3 contribution/economic summaries.

Clusters are strictly side, semantic-head, and contribution-direction local.
The five predeclared similarity components are combined with immutable
weights: structural .35, activation .20, contribution .20, economic .15, and
portability .10.  This module has no model fitting, no label construction, and
no raw leaf assignment input.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
LINKAGES = ("average", "complete")
_RAW_LEAF_ID_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")
_PAIRWISE_COLUMNS = (
    "left_rule_instance_id",
    "right_rule_instance_id",
    "side_name",
    "head_name",
    "contribution_direction",
    "structural_similarity",
    "activation_similarity",
    "contribution_similarity",
    "economic_similarity",
    "portability_similarity",
    "activation_overlap",
    "shared_threshold_feature_count",
    "threshold_interval_overlap",
    "contradictory_defining_split",
    "total_similarity",
    "compatible",
    "compatibility_reason",
)


class LeafReasoningClusterError(ValueError):
    """Raised when a G2/G3 signature clustering contract is invalid."""


@dataclass(frozen=True)
class LeafReasoningClusterConfig:
    """Fixed signature-clustering contract for one threshold/linkage arm."""

    threshold: float = 0.70
    linkage: Literal["average", "complete"] = "average"
    structural_weight: float = 0.35
    activation_weight: float = 0.20
    contribution_weight: float = 0.20
    economic_weight: float = 0.15
    portability_weight: float = 0.10
    minimum_activation_overlap: float = 0.20

    def __post_init__(self) -> None:
        if float(self.threshold) not in THRESHOLDS:
            raise LeafReasoningClusterError(
                f"threshold must be one of {THRESHOLDS}, got {self.threshold!r}"
            )
        if self.linkage not in LINKAGES:
            raise LeafReasoningClusterError(f"linkage must be one of {LINKAGES}")
        values = np.asarray(
            [
                self.structural_weight,
                self.activation_weight,
                self.contribution_weight,
                self.economic_weight,
                self.portability_weight,
            ],
            dtype=float,
        )
        expected = np.asarray([0.35, 0.20, 0.20, 0.15, 0.10], dtype=float)
        if not np.allclose(values, expected, rtol=0.0, atol=1e-12):
            raise LeafReasoningClusterError("G2/G3 similarity weights are predeclared and immutable")
        if not np.isclose(float(self.minimum_activation_overlap), 0.20, rtol=0.0, atol=1e-12):
            raise LeafReasoningClusterError(
                "minimum activation-overlap gate is predeclared and immutable at 0.20"
            )


@dataclass(frozen=True)
class LeafReasoningClusterResult:
    """Pairwise audit, assignments, and medoid/coverage summaries."""

    pairwise_similarity: pd.DataFrame
    assignments: pd.DataFrame
    cluster_summary: pd.DataFrame
    config: LeafReasoningClusterConfig


_REQUIRED = (
    "rule_instance_id",
    "fold_id",
    "side_name",
    "head_name",
    "contribution_direction",
    "rule_signature",
    "activation_rate",
    "contribution_signature",
    "economic_effect",
    "portability_score",
)


def _require_summary_contract(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        raise LeafReasoningClusterError("rule summary is empty")
    lowered = {str(column).lower() for column in summary.columns}
    unsafe_columns = {
        column for column in lowered
        if not column.startswith("base_reasoning__g1_leaf_assignment_count")
    }
    raw = sorted(token for token in _RAW_LEAF_ID_TOKENS if any(token in column for column in unsafe_columns))
    if raw:
        raise LeafReasoningClusterError(
            "raw fold-local leaf identifiers are forbidden; summarize them into "
            f"G1/G2/G3 fields first: {raw}"
        )
    missing = [name for name in _REQUIRED if name not in summary]
    if missing:
        raise LeafReasoningClusterError(f"rule summary is missing required columns: {missing}")
    work = summary.copy()
    if work["rule_instance_id"].isna().any() or work["rule_instance_id"].astype(str).str.strip().eq("").any():
        raise LeafReasoningClusterError("rule_instance_id must be non-empty")
    work["rule_instance_id"] = work["rule_instance_id"].astype(str)
    if work["rule_instance_id"].duplicated().any():
        raise LeafReasoningClusterError("rule_instance_id must be unique")
    for name in ("fold_id", "side_name", "head_name", "contribution_direction", "rule_signature", "contribution_signature"):
        if work[name].isna().any() or work[name].astype(str).str.strip().eq("").any():
            raise LeafReasoningClusterError(f"{name} must be non-empty")
        work[name] = work[name].astype(str)
    for name in ("activation_rate", "economic_effect", "portability_score"):
        work[name] = pd.to_numeric(work[name], errors="coerce")
        if not np.isfinite(work[name].to_numpy(float)).all():
            raise LeafReasoningClusterError(f"{name} must be finite")
    if ((work["activation_rate"] < 0.0) | (work["activation_rate"] > 1.0)).any():
        raise LeafReasoningClusterError("activation_rate must be in [0, 1]")
    if ((work["portability_score"] < 0.0) | (work["portability_score"] > 1.0)).any():
        raise LeafReasoningClusterError("portability_score must be in [0, 1]")
    for name in ("activation_stability", "activation_support", "portability_coverage", "portability_drift"):
        if name in work:
            work[name] = pd.to_numeric(work[name], errors="coerce")
            if not np.isfinite(work[name].to_numpy(float)).all():
                raise LeafReasoningClusterError(f"{name} must be finite when supplied")
    return work


def _json_feature_tokens(value: object) -> frozenset[str]:
    """Extract structural G2 tokens; unknown representations are harmless."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return frozenset()
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        return frozenset()
    tokens: set[str] = set()
    for step in parsed:
        if isinstance(step, Mapping):
            feature = str(step.get("feature", ""))
            decision = str(step.get("decision_type", ""))
            band = str(step.get("threshold_band_index", step.get("threshold_band_state", "")))
            if feature:
                tokens.add("|".join((feature, decision, band)))
        elif isinstance(step, str) and step:
            tokens.add(step)
    return frozenset(tokens)


def _structural_path_steps(value: object) -> tuple[Mapping[str, Any], ...]:
    """Parse the compact G2 path without relying on raw split values.

    The strict catalog deliberately stores threshold *bands*, rather than raw
    fold-local cut-points.  They are comparable across folds and are therefore
    the only threshold representation used by the compatibility gates.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        return ()
    return tuple(step for step in parsed if isinstance(step, Mapping))


def _path_branch(step: Mapping[str, Any]) -> str | None:
    """Return the path-side of a split, with a legacy operator fallback."""

    branch = str(step.get("branch", "")).strip().lower()
    if branch in {"left", "right"}:
        return branch
    decision = str(step.get("decision_type", step.get("operator", ""))).strip().lower()
    if decision in {"<=", "<", "le", "lt"}:
        return "left"
    if decision in {">=", ">", "ge", "gt"}:
        return "right"
    return None


def _path_threshold(step: Mapping[str, Any]) -> tuple[str, float, str] | None:
    """Return a comparable threshold kind/value/key for one structural step."""

    for name in ("threshold", "threshold_value", "split_threshold"):
        raw = step.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return "numeric", value, f"numeric:{value:.12g}"
    try:
        index = float(step.get("threshold_band_index"))
        count = float(step.get("threshold_band_count"))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(index) and np.isfinite(count) and count > 0.0):
        return None
    # The catalog band index is a percentile-like immutable abstraction of the
    # fold-local threshold.  Normalising it preserves only its relative
    # geometry, which is exactly what can safely transfer across folds.
    value = float(np.clip(index / count, 0.0, 1.0))
    return "band", value, f"band:{int(index)}:{int(count)}"


def _path_constraints(value: object) -> tuple[
    dict[tuple[str, str], set[str]],
    dict[tuple[str, str], tuple[float, float]],
]:
    """Build branch identities and feasible threshold intervals by feature.

    An interval is represented in normalised threshold space.  It is a
    compatibility proof only: absent or unparseable structural detail does not
    manufacture a conflict, while explicit opposing/non-overlapping paths do.
    """

    branches: dict[tuple[str, str], set[str]] = {}
    bounds: dict[tuple[str, str], tuple[float, float]] = {}
    for step in _structural_path_steps(value):
        feature = str(step.get("feature", step.get("feature_name", ""))).strip()
        branch = _path_branch(step)
        threshold = _path_threshold(step)
        if not feature or branch is None or threshold is None:
            continue
        kind, threshold_value, threshold_key = threshold
        split_key = (feature, f"{kind}:{threshold_key}")
        branches.setdefault(split_key, set()).add(branch)
        key = (feature, kind)
        lower, upper = bounds.get(key, (float("-inf"), float("inf")))
        if branch == "left":
            upper = min(upper, threshold_value)
        else:
            lower = max(lower, threshold_value)
        bounds[key] = (lower, upper)
    return branches, bounds


def _activation_overlap(left: pd.Series, right: pd.Series) -> float:
    """Conservative overlap proxy from train-time activation mass only."""

    left_rate = float(left["activation_rate"])
    right_rate = float(right["activation_rate"])
    high = max(left_rate, right_rate)
    if high <= EPS:
        return 1.0
    return float(min(left_rate, right_rate) / high)


def _structural_compatibility(left: pd.Series, right: pd.Series) -> tuple[bool, str, int, float, bool]:
    """Apply hard G2 split and threshold-interval compatibility gates."""

    left_branches, left_bounds = _path_constraints(left.get("rule_structural_path_json"))
    right_branches, right_bounds = _path_constraints(right.get("rule_structural_path_json"))
    contradictory_features = sorted({
        feature
        for (feature, threshold), directions in left_branches.items()
        if (other := right_branches.get((feature, threshold))) is not None
        and directions.isdisjoint(other)
    })
    if contradictory_features:
        return False, "contradictory_defining_split:" + ",".join(contradictory_features), 0, 0.0, True

    shared = sorted(set(left_bounds).intersection(right_bounds))
    if not shared:
        # There is no common threshold geometry to contradict.  This is not a
        # positive similarity signal; it merely leaves the similarity weights
        # to determine whether the rules can merge.
        return True, "threshold_interval_not_applicable", 0, 1.0, False
    overlaps: list[float] = []
    incompatible: list[str] = []
    for feature, kind in shared:
        left_lower, left_upper = left_bounds[(feature, kind)]
        right_lower, right_upper = right_bounds[(feature, kind)]
        intersection_lower = max(left_lower, right_lower)
        intersection_upper = min(left_upper, right_upper)
        if intersection_lower > intersection_upper + EPS:
            incompatible.append(feature)
            overlaps.append(0.0)
            continue
        union_lower = min(left_lower, right_lower)
        union_upper = max(left_upper, right_upper)
        finite_span = union_upper - union_lower
        if not np.isfinite(finite_span) or finite_span <= EPS:
            overlaps.append(1.0)
        else:
            overlaps.append(float(np.clip((intersection_upper - intersection_lower) / finite_span, 0.0, 1.0)))
    if incompatible:
        return False, "incompatible_threshold_interval:" + ",".join(sorted(set(incompatible))), len(shared), 0.0, False
    return True, "compatible_threshold_interval", len(shared), float(min(overlaps, default=1.0)), False


def _contribution_vector(value: object) -> dict[str, float]:
    """Parse an optional compact G3 contribution signature into abs weights."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if isinstance(parsed, Mapping):
        output: dict[str, float] = {}
        for key, raw in parsed.items():
            try:
                number = abs(float(raw))
            except (TypeError, ValueError):
                continue
            if np.isfinite(number) and number > 0.0:
                output[str(key)] = number
        return output
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        return {str(name): 1.0 for name in parsed if str(name)}
    return {}


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return float(len(left & right) / max(len(left | right), 1))


def _relative_similarity(left: float, right: float, *, floor: float = 1e-6) -> float:
    scale = max((abs(float(left)) + abs(float(right))) / 2.0, floor)
    return float(np.exp(-abs(float(left) - float(right)) / scale))


def _weighted_cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    keys = sorted(set(left) | set(right))
    x = np.asarray([left.get(key, 0.0) for key in keys], dtype=float)
    y = np.asarray([right.get(key, 0.0) for key in keys], dtype=float)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > EPS else 0.0


def _structural_similarity(left: pd.Series, right: pd.Series) -> float:
    if left["rule_signature"] == right["rule_signature"]:
        return 1.0
    left_tokens = _json_feature_tokens(left.get("rule_structural_path_json"))
    right_tokens = _json_feature_tokens(right.get("rule_structural_path_json"))
    # A missing optional expansion must never make two otherwise distinct
    # rules look identical.  The compact rule signature remains the primary
    # contract; the path only supplies partial credit between distinct ones.
    if not left_tokens or not right_tokens:
        return 0.0
    return _jaccard(left_tokens, right_tokens)


def _activation_similarity(left: pd.Series, right: pd.Series) -> float:
    values = [_relative_similarity(left["activation_rate"], right["activation_rate"])]
    for name in ("activation_stability", "activation_support"):
        if name in left.index and name in right.index:
            values.append(_relative_similarity(left[name], right[name]))
    return float(np.mean(values))


def _contribution_similarity(left: pd.Series, right: pd.Series) -> float:
    if left["contribution_signature"] == right["contribution_signature"]:
        return 1.0
    left_vector = _contribution_vector(left.get("contribution_top_features_json"))
    right_vector = _contribution_vector(right.get("contribution_top_features_json"))
    # As with structural paths, absent optional detail is not evidence of
    # cross-fold contribution equivalence.
    if not left_vector or not right_vector:
        return 0.0
    return _weighted_cosine(left_vector, right_vector)


def _economic_similarity(left: pd.Series, right: pd.Series) -> float:
    left_value, right_value = float(left["economic_effect"]), float(right["economic_effect"])
    if left_value == 0.0 and right_value == 0.0:
        return 1.0
    if left_value * right_value < 0.0:
        return 0.0
    return _relative_similarity(left_value, right_value)


def _portability_similarity(left: pd.Series, right: pd.Series) -> float:
    values = [_relative_similarity(left["portability_score"], right["portability_score"])]
    if "portability_coverage" in left.index and "portability_coverage" in right.index:
        values.append(_relative_similarity(left["portability_coverage"], right["portability_coverage"]))
    if "portability_drift" in left.index and "portability_drift" in right.index:
        values.append(_relative_similarity(left["portability_drift"], right["portability_drift"]))
    return float(np.mean(values))


def pairwise_leaf_reasoning_similarity(
    summary: pd.DataFrame,
    *,
    config: LeafReasoningClusterConfig = LeafReasoningClusterConfig(),
) -> pd.DataFrame:
    """Calculate the five predeclared similarities for every rule-instance pair."""
    work = _require_summary_contract(summary)
    rows: list[dict[str, Any]] = []
    values = list(work.itertuples(index=False))
    # Convert namedtuple rows once; all pair work consumes summaries rather
    # than raw leaf tokens, assignments, or prediction rows.
    records = [pd.Series(item._asdict()) for item in values]
    for position, left in enumerate(records):
        for right in records[position + 1:]:
            identity_compatible = (
                left["side_name"] == right["side_name"]
                and left["head_name"] == right["head_name"]
                and left["contribution_direction"] == right["contribution_direction"]
            )
            activation_overlap = _activation_overlap(left, right)
            threshold_features = 0
            threshold_overlap = 0.0
            contradictory = False
            reasons: list[str] = []
            if not identity_compatible:
                mismatches = [
                    name for name in ("side_name", "head_name", "contribution_direction")
                    if left[name] != right[name]
                ]
                reasons.append("mismatch:" + ",".join(mismatches))
            else:
                structural_ok, structural_reason, threshold_features, threshold_overlap, contradictory = _structural_compatibility(left, right)
                if not structural_ok:
                    reasons.append(structural_reason)
                if activation_overlap + EPS < float(config.minimum_activation_overlap):
                    reasons.append(
                        "minimum_activation_overlap:"
                        f"{activation_overlap:.6f}<{float(config.minimum_activation_overlap):.6f}"
                    )
            compatible = bool(identity_compatible and not reasons)
            reason = "compatible" if compatible else ";".join(reasons)
            structural = _structural_similarity(left, right) if compatible else 0.0
            activation = _activation_similarity(left, right) if compatible else 0.0
            contribution = _contribution_similarity(left, right) if compatible else 0.0
            economic = _economic_similarity(left, right) if compatible else 0.0
            portability = _portability_similarity(left, right) if compatible else 0.0
            total = (
                config.structural_weight * structural
                + config.activation_weight * activation
                + config.contribution_weight * contribution
                + config.economic_weight * economic
                + config.portability_weight * portability
            ) if compatible else 0.0
            rows.append(
                {
                    "left_rule_instance_id": left["rule_instance_id"],
                    "right_rule_instance_id": right["rule_instance_id"],
                    "side_name": left["side_name"],
                    "head_name": left["head_name"],
                    "contribution_direction": left["contribution_direction"],
                    "structural_similarity": structural,
                    "activation_similarity": activation,
                    "contribution_similarity": contribution,
                    "economic_similarity": economic,
                    "portability_similarity": portability,
                    "activation_overlap": float(activation_overlap),
                    "shared_threshold_feature_count": int(threshold_features),
                    "threshold_interval_overlap": float(threshold_overlap),
                    "contradictory_defining_split": bool(contradictory),
                    "total_similarity": float(total),
                    "compatible": bool(compatible),
                    "compatibility_reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=_PAIRWISE_COLUMNS)


def _pair_value(
    lookup: Mapping[tuple[str, str], float], left: str, right: str,
) -> float:
    # Hard-incompatible pairs are intentionally absent from the lookup.  They
    # must behave as zero similarity inside an otherwise shared side/head/
    # direction cell, rather than causing a clustering-time KeyError.
    return 1.0 if left == right else float(lookup.get((left, right), 0.0))


def _linkage_score(
    left: Sequence[str], right: Sequence[str], lookup: Mapping[tuple[str, str], float], linkage: str,
) -> float:
    values = np.asarray([_pair_value(lookup, a, b) for a in left for b in right], dtype=float)
    return float(values.mean()) if linkage == "average" else float(values.min())


def _cluster_one_cell(
    rule_ids: Sequence[str], lookup: Mapping[tuple[str, str], float], *, config: LeafReasoningClusterConfig,
) -> list[list[str]]:
    clusters = [[item] for item in sorted(rule_ids)]
    while True:
        selected: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = _linkage_score(clusters[i], clusters[j], lookup, config.linkage)
                if score < config.threshold:
                    continue
                candidate = (score, -i, -j)
                if selected is None or candidate > (selected[0], -selected[1], -selected[2]):
                    selected = (score, i, j)
        if selected is None:
            break
        _score, i, j = selected
        clusters[i] = sorted([*clusters[i], *clusters[j]])
        clusters.pop(j)
    return clusters


def _medoid(rule_ids: Sequence[str], lookup: Mapping[tuple[str, str], float]) -> tuple[str, float]:
    if len(rule_ids) == 1:
        return str(rule_ids[0]), 1.0
    scores = {
        rule: float(np.mean([_pair_value(lookup, rule, other) for other in rule_ids if other != rule]))
        for rule in rule_ids
    }
    selected = sorted(scores, key=lambda rule: (-scores[rule], rule))[0]
    return selected, scores[selected]


def cluster_leaf_reasoning_signatures(
    summary: pd.DataFrame,
    *,
    config: LeafReasoningClusterConfig = LeafReasoningClusterConfig(),
) -> LeafReasoningClusterResult:
    """Cluster G2/G3 rule summaries and return portable medoid coverage tables."""
    work = _require_summary_contract(summary)
    pairwise = pairwise_leaf_reasoning_similarity(work, config=config)
    lookup = {
        (str(row.left_rule_instance_id), str(row.right_rule_instance_id)): float(row.total_similarity)
        for row in pairwise.loc[pairwise["compatible"]].itertuples(index=False)
    }
    lookup |= {(right, left): value for (left, right), value in list(lookup.items())}
    assignments: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    group_columns = ["side_name", "head_name", "contribution_direction"]
    for group, cell in work.groupby(group_columns, observed=True, sort=True):
        side, head, direction = map(str, group)
        rule_ids = cell["rule_instance_id"].astype(str).tolist()
        available_folds = set(cell["fold_id"].astype(str))
        clusters = _cluster_one_cell(rule_ids, lookup, config=config)
        for number, members in enumerate(clusters, start=1):
            cluster_id = f"{side}::{head}::{direction}::c{number:03d}"
            medoid, medoid_mean = _medoid(members, lookup)
            member_rows = cell.set_index("rule_instance_id").loc[members]
            pair_values = [
                _pair_value(lookup, left, right)
                for pos, left in enumerate(members) for right in members[pos + 1:]
            ]
            folds = set(member_rows["fold_id"].astype(str))
            summaries.append(
                {
                    "cluster_id": cluster_id,
                    "side_name": side,
                    "head_name": head,
                    "contribution_direction": direction,
                    "threshold": float(config.threshold),
                    "linkage": config.linkage,
                    "member_count": int(len(members)),
                    "medoid_rule_instance_id": medoid,
                    "medoid_mean_similarity": float(medoid_mean),
                    "cluster_mean_pairwise_similarity": float(np.mean(pair_values)) if pair_values else 1.0,
                    "cluster_min_pairwise_similarity": float(np.min(pair_values)) if pair_values else 1.0,
                    "fold_coverage_count": int(len(folds)),
                    "available_fold_count": int(len(available_folds)),
                    "fold_coverage_fraction": float(len(folds) / max(len(available_folds), 1)),
                    "activation_rate_mean": float(member_rows["activation_rate"].mean()),
                    "activation_rate_sum": float(member_rows["activation_rate"].sum()),
                    "economic_effect_mean": float(member_rows["economic_effect"].mean()),
                    "portability_score_mean": float(member_rows["portability_score"].mean()),
                }
            )
            for member in members:
                assignments.append(
                    {
                        "rule_instance_id": member,
                        "cluster_id": cluster_id,
                        "side_name": side,
                        "head_name": head,
                        "contribution_direction": direction,
                        "is_medoid": bool(member == medoid),
                        "similarity_to_medoid": _pair_value(lookup, member, medoid),
                    }
                )
    return LeafReasoningClusterResult(
        pairwise_similarity=pairwise.sort_values(
            ["left_rule_instance_id", "right_rule_instance_id"], kind="stable"
        ).reset_index(drop=True),
        assignments=pd.DataFrame(assignments).sort_values("rule_instance_id", kind="stable").reset_index(drop=True),
        cluster_summary=pd.DataFrame(summaries).sort_values("cluster_id", kind="stable").reset_index(drop=True),
        config=config,
    )


def sweep_leaf_reasoning_clusters(
    summary: pd.DataFrame,
    *,
    thresholds: Iterable[float] = THRESHOLDS,
    linkages: Iterable[Literal["average", "complete"]] = LINKAGES,
) -> dict[tuple[float, str], LeafReasoningClusterResult]:
    """Run only the predeclared .6/.7/.8/.9 × average/complete grid."""
    results: dict[tuple[float, str], LeafReasoningClusterResult] = {}
    for threshold in thresholds:
        for linkage in linkages:
            config = LeafReasoningClusterConfig(threshold=float(threshold), linkage=linkage)
            results[(float(threshold), str(linkage))] = cluster_leaf_reasoning_signatures(summary, config=config)
    return results


__all__ = [
    "LINKAGES",
    "THRESHOLDS",
    "LeafReasoningClusterConfig",
    "LeafReasoningClusterError",
    "LeafReasoningClusterResult",
    "cluster_leaf_reasoning_signatures",
    "pairwise_leaf_reasoning_similarity",
    "sweep_leaf_reasoning_clusters",
]
