"""Outcome-free recurrent families for frozen base-tree rules.

This module is intentionally narrower than :mod:`leaf_reasoning_clusters`.
It aligns rules *only* from their frozen structural paths and retains a family
only when it recurs across independently fitted base models and folds.  It is
therefore safe to run before historical correctness, portability, residual,
or economic annotations exist.  Those annotations belong to a later,
strictly-historical hand-off keyed by the emitted ``cluster_id``.

The optional frozen contribution input is a compute/coverage geometry from the
already-fitted base ensemble.  It can bound the catalogue and prioritise a
bounded number of recurrent families, but it is never used in pairwise
alignment and cannot contain a realised outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
LINKAGES = ("average", "complete")
_RAW_LEAF_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")
_OUTCOME_TOKENS = (
    "target", "label", "outcome", "economic", "net", "gross", "pnl", "return",
    "residual", "portability", "correctness", "realised", "realized", "health",
)
_REQUIRED = (
    "rule_instance_id", "fold_id", "model_id", "side_name", "head_name",
    "rule_signature", "rule_structural_path_json",
)
_OPTIONAL = ("base_model_version", "model_layer", "train_leaf_frequency")


class StructuralRuleFamilyError(ValueError):
    """Raised when an outcome-free structural family contract is violated."""


@dataclass(frozen=True)
class StructuralRuleFamilyConfig:
    """Bounded clustering controls for one frozen base-model generation."""

    threshold: float = 0.70
    linkage: Literal["average", "complete"] = "average"
    min_distinct_folds: int = 2
    min_distinct_models: int = 2
    max_rule_instances_per_cell: int = 512
    max_selected_families_per_cell: int = 20
    frozen_contribution_column: str | None = None

    def validate(self) -> None:
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise StructuralRuleFamilyError("threshold must be in [0, 1]")
        if self.linkage not in LINKAGES:
            raise StructuralRuleFamilyError(f"linkage must be one of {LINKAGES}")
        if int(self.min_distinct_folds) < 1 or int(self.min_distinct_models) < 1:
            raise StructuralRuleFamilyError("recurrence requirements must be positive")
        if int(self.max_rule_instances_per_cell) < 1:
            raise StructuralRuleFamilyError("max_rule_instances_per_cell must be positive")
        if int(self.max_selected_families_per_cell) < 1:
            raise StructuralRuleFamilyError("max_selected_families_per_cell must be positive")
        if self.frozen_contribution_column is not None:
            name = str(self.frozen_contribution_column)
            if not name or any(token in name.lower() for token in _OUTCOME_TOKENS):
                raise StructuralRuleFamilyError("frozen contribution column must not be an outcome field")


@dataclass(frozen=True)
class StructuralRuleFamilyResult:
    """Structural audit plus bounded recurrent family assignments."""

    pairwise_similarity: pd.DataFrame
    assignments: pd.DataFrame
    family_summary: pd.DataFrame
    selected_cluster_ids: tuple[str, ...]
    config: StructuralRuleFamilyConfig


@dataclass(frozen=True)
class StructuralRuleFamilyPosteriorResult:
    """Candidate-level, raw-token-free posterior features for a meta model."""

    features: pd.DataFrame
    cluster_id_to_feature: Mapping[str, str]


def _forbid_outcomes(columns: Iterable[object], *, source: str) -> None:
    lowered = [str(column).lower() for column in columns]
    raw = sorted(token for token in _RAW_LEAF_TOKENS if any(token in column for column in lowered))
    if raw:
        raise StructuralRuleFamilyError(
            f"{source} contains raw local leaf identifiers; use only rule instances after local lookup: {raw}"
        )
    forbidden = sorted(token for token in _OUTCOME_TOKENS if any(token in column for column in lowered))
    if forbidden:
        raise StructuralRuleFamilyError(
            f"{source} contains outcome/economic/portability fields forbidden during structural alignment: {forbidden}"
        )


def _non_empty_strings(work: pd.DataFrame, names: Sequence[str]) -> None:
    for name in names:
        if work[name].isna().any() or work[name].astype(str).str.strip().eq("").any():
            raise StructuralRuleFamilyError(f"{name} must be non-empty")
        work[name] = work[name].astype(str)


def _validate_catalogue(
    catalogue: pd.DataFrame, config: StructuralRuleFamilyConfig,
) -> pd.DataFrame:
    config.validate()
    if catalogue.empty:
        raise StructuralRuleFamilyError("structural rule catalogue is empty")
    _forbid_outcomes(catalogue.columns, source="structural rule catalogue")
    missing = sorted(set(_REQUIRED).difference(catalogue.columns))
    if missing:
        raise StructuralRuleFamilyError(f"structural rule catalogue is missing required columns: {missing}")
    if config.frozen_contribution_column is not None and config.frozen_contribution_column not in catalogue:
        raise StructuralRuleFamilyError("declared frozen contribution column is absent")
    work = catalogue.copy()
    _non_empty_strings(work, _REQUIRED)
    if work["rule_instance_id"].duplicated().any():
        raise StructuralRuleFamilyError("rule_instance_id must be unique")
    for name in ("base_model_version", "model_layer"):
        if name not in work:
            work[name] = "__default__"
        _non_empty_strings(work, (name,))
    if "train_leaf_frequency" not in work:
        work["train_leaf_frequency"] = 1.0
    work["train_leaf_frequency"] = pd.to_numeric(work["train_leaf_frequency"], errors="coerce")
    if not np.isfinite(work["train_leaf_frequency"].to_numpy(float)).all() or (work["train_leaf_frequency"] < 0).any():
        raise StructuralRuleFamilyError("train_leaf_frequency must be finite and non-negative")
    if config.frozen_contribution_column is None:
        work["__frozen_abs_contribution__"] = 1.0
    else:
        values = pd.to_numeric(work[config.frozen_contribution_column], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise StructuralRuleFamilyError("frozen contribution geometry must be finite")
        work["__frozen_abs_contribution__"] = values.abs()
    return work


def _path_tokens(value: object) -> frozenset[str]:
    """Canonical structural predicates; no target or realised-path information."""
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
            feature = str(step.get("feature", "")).strip()
            decision = str(step.get("decision_type", "")).strip()
            band = str(step.get("threshold_band_index", step.get("threshold_band_state", ""))).strip()
            if feature:
                tokens.add("|".join((feature, decision, band)))
        elif isinstance(step, str) and step.strip():
            tokens.add(step.strip())
    return frozenset(tokens)


def _structural_similarity(left: pd.Series, right: pd.Series) -> float:
    if str(left["rule_signature"]) == str(right["rule_signature"]):
        return 1.0
    lhs, rhs = _path_tokens(left["rule_structural_path_json"]), _path_tokens(right["rule_structural_path_json"])
    if not lhs or not rhs:
        return 0.0
    return float(len(lhs & rhs) / len(lhs | rhs))


def _bounded_catalogue(work: pd.DataFrame, config: StructuralRuleFamilyConfig) -> pd.DataFrame:
    """Limit pairwise work using only frozen in-model support/geometry."""
    pieces: list[pd.DataFrame] = []
    keys = ["side_name", "head_name", "base_model_version", "model_layer"]
    for _cell_key, cell in work.groupby(keys, observed=True, sort=True):
        # A stable, outcome-free priority.  Frequency prevents a very rare
        # leaf from displacing a recurring rule solely due to a large leaf
        # value; geometry is optional and can only break/weight ties.
        priority = cell["train_leaf_frequency"].to_numpy(float) * np.maximum(
            cell["__frozen_abs_contribution__"].to_numpy(float), 1.0,
        )
        bounded = cell.assign(__structural_priority__=priority).sort_values(
            ["__structural_priority__", "rule_instance_id"], ascending=[False, True], kind="stable",
        ).head(int(config.max_rule_instances_per_cell))
        pieces.append(bounded)
    return pd.concat(pieces, ignore_index=True)


def _linkage_score(
    left: Sequence[str], right: Sequence[str], lookup: Mapping[tuple[str, str], float], linkage: str,
) -> float:
    values = [1.0 if a == b else float(lookup[(a, b)]) for a in left for b in right]
    return float(np.mean(values)) if linkage == "average" else float(np.min(values))


def _cluster_cell(
    ids: Sequence[str], lookup: Mapping[tuple[str, str], float], config: StructuralRuleFamilyConfig,
) -> list[list[str]]:
    clusters = [[str(identifier)] for identifier in sorted(ids)]
    while True:
        best: tuple[float, int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                score = _linkage_score(clusters[left], clusters[right], lookup, config.linkage)
                if score < config.threshold:
                    continue
                candidate = (score, -left, -right)
                if best is None or candidate > (best[0], -best[1], -best[2]):
                    best = (score, left, right)
        if best is None:
            return clusters
        _score, left, right = best
        clusters[left] = sorted([*clusters[left], *clusters[right]])
        clusters.pop(right)


def _medoid(ids: Sequence[str], lookup: Mapping[tuple[str, str], float]) -> tuple[str, float]:
    if len(ids) == 1:
        return str(ids[0]), 1.0
    score = {
        str(identifier): float(np.mean([1.0 if identifier == other else lookup[(str(identifier), str(other))] for other in ids]))
        for identifier in ids
    }
    selected = min(score, key=lambda item: (-score[item], item))
    return selected, score[selected]


def cluster_structural_rule_families(
    catalogue: pd.DataFrame,
    *,
    config: StructuralRuleFamilyConfig = StructuralRuleFamilyConfig(),
) -> StructuralRuleFamilyResult:
    """Align frozen base-tree rules without outcome, economics, or portability.

    The returned family IDs are structural taxonomy keys.  They deliberately
    carry no correctness or payoff assertion; a later prequential stage may
    attach those values only from already-resolved rows.
    """
    work = _bounded_catalogue(_validate_catalogue(catalogue, config), config)
    keys = ["side_name", "head_name", "base_model_version", "model_layer"]
    pair_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for group, cell in work.groupby(keys, observed=True, sort=True):
        side, head, version, layer = map(str, group)
        records = [
            pd.Series(record._asdict())
            for record in cell.sort_values("rule_instance_id", kind="stable").itertuples(index=False)
        ]
        lookup: dict[tuple[str, str], float] = {}
        for position, left in enumerate(records):
            for right in records[position + 1:]:
                similarity = _structural_similarity(left, right)
                left_id, right_id = str(left["rule_instance_id"]), str(right["rule_instance_id"])
                lookup[(left_id, right_id)] = similarity
                lookup[(right_id, left_id)] = similarity
                pair_rows.append({
                    "left_rule_instance_id": left_id,
                    "right_rule_instance_id": right_id,
                    "side_name": side,
                    "head_name": head,
                    "base_model_version": version,
                    "model_layer": layer,
                    "structural_similarity": similarity,
                })
        clusters = _cluster_cell(cell["rule_instance_id"].astype(str).tolist(), lookup, config)
        summaries_for_cell: list[dict[str, Any]] = []
        for ordinal, members in enumerate(clusters, start=1):
            family_id = f"{side}::{head}::{version}::{layer}::s{ordinal:04d}"
            rows = cell.set_index("rule_instance_id").loc[members]
            folds = set(rows["fold_id"].astype(str))
            models = set(rows["model_id"].astype(str))
            medoid, medoid_mean = _medoid(members, lookup)
            pair_scores = [
                1.0 if a == b else lookup[(a, b)]
                for index, a in enumerate(members) for b in members[index + 1:]
            ]
            recurrent = len(folds) >= config.min_distinct_folds and len(models) >= config.min_distinct_models
            # Strictly structural support; optional frozen geometry merely
            # makes an otherwise recurrent bounded taxonomy more useful.
            priority = (
                len(folds) * len(models) * max(float(rows["train_leaf_frequency"].sum()), EPS)
                * max(float(rows["__frozen_abs_contribution__"].sum()), EPS)
            )
            row = {
                "cluster_id": family_id,
                "side_name": side,
                "head_name": head,
                "base_model_version": version,
                "model_layer": layer,
                "threshold": float(config.threshold),
                "linkage": config.linkage,
                "member_count": int(len(members)),
                "distinct_fold_count": int(len(folds)),
                "distinct_model_count": int(len(models)),
                "is_recurrent": bool(recurrent),
                "medoid_rule_instance_id": medoid,
                "medoid_mean_structural_similarity": float(medoid_mean),
                "mean_pairwise_structural_similarity": float(np.mean(pair_scores)) if pair_scores else 1.0,
                "min_pairwise_structural_similarity": float(np.min(pair_scores)) if pair_scores else 1.0,
                "structural_selection_priority": float(priority),
            }
            summaries_for_cell.append(row)
            for member in members:
                assignments.append({
                    "rule_instance_id": member,
                    "cluster_id": family_id,
                    "side_name": side,
                    "head_name": head,
                    "base_model_version": version,
                    "model_layer": layer,
                    "is_medoid": bool(member == medoid),
                    "similarity_to_medoid": 1.0 if member == medoid else float(lookup[(member, medoid)]),
                    "is_recurrent": bool(recurrent),
                })
        recurrent_ids = [row for row in summaries_for_cell if row["is_recurrent"]]
        selected = sorted(
            recurrent_ids,
            key=lambda row: (-row["structural_selection_priority"], row["cluster_id"]),
        )[: int(config.max_selected_families_per_cell)]
        selected_ids = {row["cluster_id"] for row in selected}
        for row in summaries_for_cell:
            row["is_selected"] = row["cluster_id"] in selected_ids
        summaries.extend(summaries_for_cell)
    summary = pd.DataFrame(summaries).sort_values("cluster_id", kind="stable").reset_index(drop=True)
    selected = tuple(summary.loc[summary["is_selected"], "cluster_id"].astype(str).tolist())
    assignment = pd.DataFrame(assignments).sort_values("rule_instance_id", kind="stable").reset_index(drop=True)
    selected_lookup = summary.set_index("cluster_id")["is_selected"]
    assignment["is_selected"] = assignment["cluster_id"].map(selected_lookup).astype(bool)
    pairwise = pd.DataFrame(pair_rows, columns=(
        "left_rule_instance_id", "right_rule_instance_id", "side_name", "head_name",
        "base_model_version", "model_layer", "structural_similarity",
    )).sort_values(["left_rule_instance_id", "right_rule_instance_id"], kind="stable").reset_index(drop=True)
    return StructuralRuleFamilyResult(pairwise, assignment, summary, selected, config)


def _feature_name(cluster_id: str) -> str:
    return "base_structural_family__" + hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:20]


def materialize_structural_family_posteriors(
    activations: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    candidate_id_column: str = "candidate_id",
    contribution_column: str | None = None,
) -> StructuralRuleFamilyPosteriorResult:
    """Convert local rule activations into bounded family-membership posteriors.

    ``activations`` may contain raw local leaf identifiers only upstream of the
    caller's local lookup; this public boundary accepts *rule_instance_id*
    only.  Each selected family gets its absolute contribution share, while
    ``unassigned_mass`` preserves the evidence carried by non-selected
    structural rules.  Thus no raw leaf token is exposed to the meta layer.
    """
    _forbid_outcomes(activations.columns, source="candidate rule activations")
    required = {candidate_id_column, "rule_instance_id"}
    missing = sorted(required.difference(activations.columns))
    if missing:
        raise StructuralRuleFamilyError(f"candidate rule activations are missing {missing}")
    required_assignments = {"rule_instance_id", "cluster_id", "is_selected"}
    missing_assignments = sorted(required_assignments.difference(assignments.columns))
    if missing_assignments:
        raise StructuralRuleFamilyError(f"structural assignments are missing {missing_assignments}")
    if contribution_column is not None:
        if contribution_column not in activations:
            raise StructuralRuleFamilyError("candidate contribution column is absent")
        if any(token in contribution_column.lower() for token in _OUTCOME_TOKENS):
            raise StructuralRuleFamilyError("candidate contribution must not be an outcome field")
    work = activations.loc[:, [candidate_id_column, "rule_instance_id", *([contribution_column] if contribution_column else [])]].copy()
    _non_empty_strings(work, (candidate_id_column, "rule_instance_id"))
    if contribution_column is None:
        work["__weight__"] = 1.0
    else:
        values = pd.to_numeric(work[contribution_column], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise StructuralRuleFamilyError("candidate contribution must be finite")
        work["__weight__"] = values.abs()
    key = assignments.loc[:, ["rule_instance_id", "cluster_id", "is_selected"]].copy()
    if key["rule_instance_id"].duplicated().any():
        raise StructuralRuleFamilyError("structural assignments must map each rule instance exactly once")
    merged = work.merge(key, on="rule_instance_id", how="left", validate="many_to_one")
    if merged["cluster_id"].isna().any():
        raise StructuralRuleFamilyError("candidate activation has no structural family assignment")
    totals = merged.groupby(candidate_id_column, observed=True)["__weight__"].sum().rename("__total__")
    selected = merged.loc[merged["is_selected"].astype(bool)].groupby(
        [candidate_id_column, "cluster_id"], observed=True,
    )["__weight__"].sum().rename("__weight__").reset_index()
    cluster_ids = sorted(key.loc[key["is_selected"].astype(bool), "cluster_id"].astype(str).unique())
    feature_map = {cluster_id: _feature_name(cluster_id) for cluster_id in cluster_ids}
    candidates = pd.DataFrame({candidate_id_column: pd.Index(work[candidate_id_column].unique(), dtype="object")})
    output = candidates.copy()
    for cluster_id in cluster_ids:
        field = feature_map[cluster_id]
        amount = selected.loc[selected["cluster_id"].eq(cluster_id), [candidate_id_column, "__weight__"]]
        value = candidates.merge(amount, on=candidate_id_column, how="left")["__weight__"].fillna(0.0).to_numpy(float)
        denom = candidates[candidate_id_column].map(totals).to_numpy(float)
        output[field] = np.divide(value, denom, out=np.zeros(len(output), dtype=np.float32), where=denom > 0.0).astype(np.float32)
    fields = list(feature_map.values())
    assigned = output.loc[:, fields].sum(axis=1).to_numpy(float) if fields else np.zeros(len(output), dtype=float)
    output["base_structural_family__unassigned_mass"] = np.clip(1.0 - assigned, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(output.drop(columns=[candidate_id_column]).to_numpy(float)).all():
        raise StructuralRuleFamilyError("structural family posteriors must be finite")
    return StructuralRuleFamilyPosteriorResult(output, feature_map)


__all__ = [
    "LINKAGES",
    "StructuralRuleFamilyConfig",
    "StructuralRuleFamilyError",
    "StructuralRuleFamilyPosteriorResult",
    "StructuralRuleFamilyResult",
    "cluster_structural_rule_families",
    "materialize_structural_family_posteriors",
]
