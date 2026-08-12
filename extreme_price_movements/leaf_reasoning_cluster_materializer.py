"""Strict-OOF candidate cluster features for the leaf-reasoning C funnel.

The rule-level clustering implementation deliberately accepts only structural
G2/G3 summaries.  This module is the separate, provenance-checked hand-off
which turns those clusters into compact *candidate* features.  It never emits
an opaque LightGBM leaf token and never reads a final-OOS population.

The C taxonomy is fitted on the supplied completed development strict root.
It uses only catalog/train-time rule attributes and current candidate base
scores for the contribution-coverage selection; realised outer outcomes never
enter the taxonomy or the candidate features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Required by the production parquet contract and avoids loading 3M rows for a schema.
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - production/test environments install pyarrow
    pq = None

from .leaf_family_contributions import extract_leaf_family_contributions
from .leaf_reasoning_clusters import (
    LINKAGES,
    THRESHOLDS,
    LeafReasoningClusterConfig,
    cluster_leaf_reasoning_signatures,
)
from .tp6_portability_data import FROZEN_META_CONTEXT


SCHEMA = "leaf_reasoning_candidate_cluster_materializer_v1"
STATUS = "STRICT_OOF_CANDIDATE_CLUSTER_FEATURES_MATERIALIZED"
IDENTITY = (
    "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
)
HEADS = frozenset(("p_adverse", "p_weak", "p_clear"))
PARTITIONS = frozenset(("inner_oof", "outer_test"))
RAW_LEAF_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")
C_ARMS = ("C1", "C2", "C3", "C4")
ARM_THRESHOLDS = dict(zip(C_ARMS, THRESHOLDS, strict=True))
THRESHOLD_SWEEP_PHASE = "threshold_sweep"
FINAL_PHASE = "final"


class LeafReasoningClusterMaterializerError(ValueError):
    """Raised when a C-stage strict-OOF hand-off cannot be proven."""


@dataclass(frozen=True)
class LeafReasoningClusterMaterializerConfig:
    """Fixed compute and compactness controls for the development C funnel."""

    linkage: str = "average"
    max_rule_instances_per_cell: int = 48
    max_clusters_per_arm: int = 20
    c5_top_decile_coverage_target: float = 0.95
    c5_min_portability: float = 0.50
    c6_soft_cap: int = 12

    def validate(self) -> None:
        if self.linkage not in LINKAGES:
            raise LeafReasoningClusterMaterializerError(
                f"linkage must be one of {LINKAGES}, got {self.linkage!r}"
            )
        if int(self.max_rule_instances_per_cell) < 1:
            raise LeafReasoningClusterMaterializerError("max_rule_instances_per_cell must be positive")
        if not 1 <= int(self.max_clusters_per_arm) <= 20:
            raise LeafReasoningClusterMaterializerError("max_clusters_per_arm must be in [1, 20]")
        if not 1 <= int(self.c6_soft_cap) <= int(self.max_clusters_per_arm):
            raise LeafReasoningClusterMaterializerError("c6_soft_cap must be in [1, max_clusters_per_arm]")
        if not 0.0 < float(self.c5_top_decile_coverage_target) <= 1.0:
            raise LeafReasoningClusterMaterializerError("C5 coverage target must be in (0, 1]")
        if not 0.0 <= float(self.c5_min_portability) <= 1.0:
            raise LeafReasoningClusterMaterializerError("C5 portability threshold must be in [0, 1]")


@dataclass(frozen=True)
class LeafReasoningClusterMaterializationResult:
    output_dir: Path
    candidate_count: int
    feature_count: int
    cluster_counts: Mapping[str, int]
    taxonomy: Mapping[str, Any]


@dataclass(frozen=True)
class LeafReasoningClusterFinalizationConfig:
    """Fixed promotion gates for the post-C1--C4 C5/C6 selector."""

    c5_top_decile_coverage_target: float = 0.95
    c5_min_portability: float = 0.50
    c6_soft_cap: int = 12
    minimum_positive_environment_rate: float = 0.70
    max_worst_month_net_drop_bps: float | None = None

    def validate(self) -> None:
        if not 0.0 < float(self.c5_top_decile_coverage_target) <= 1.0:
            raise LeafReasoningClusterMaterializerError("C5 coverage target must be in (0, 1]")
        if not 0.0 <= float(self.c5_min_portability) <= 1.0:
            raise LeafReasoningClusterMaterializerError("C5 portability threshold must be in [0, 1]")
        if not 1 <= int(self.c6_soft_cap) <= 20:
            raise LeafReasoningClusterMaterializerError("C6 soft cap must be in [1, 20]")
        if not 0.0 <= float(self.minimum_positive_environment_rate) <= 1.0:
            raise LeafReasoningClusterMaterializerError("minimum positive environment rate must be in [0, 1]")
        if self.max_worst_month_net_drop_bps is not None and float(self.max_worst_month_net_drop_bps) > 0.0:
            raise LeafReasoningClusterMaterializerError("max worst-month drop must be <= 0 when declared")


@dataclass(frozen=True)
class LeafReasoningClusterCandidateArtifact:
    """Validated immutable C-stage hand-off consumed by the meta CLI."""

    root: Path
    candidate_features: pd.DataFrame
    groups: Mapping[str, list[str]]
    taxonomy: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _C5CoverageSelection:
    """C5's promoted portable prefix plus explicit contribution accounting."""

    selected_ids: list[str]
    selected_coverage_by_transport: dict[str, float]
    prefix_audit: pd.DataFrame
    coverage_manifest: pd.DataFrame
    portable_manifest: pd.DataFrame
    diagnostic_manifest: pd.DataFrame
    coverage_report: pd.DataFrame


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeafReasoningClusterMaterializerError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LeafReasoningClusterMaterializerError(f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _utc(values: pd.Series, *, source: str) -> pd.Series:
    output = pd.to_datetime(values, utc=True, errors="coerce")
    if output.isna().any():
        raise LeafReasoningClusterMaterializerError(f"{source} has non-UTC or null timestamps")
    return output


def _forbid_raw_leaf(columns: Iterable[object], *, source: str) -> None:
    bad = sorted(
        str(column)
        for column in columns
        if not str(column).lower().startswith("base_reasoning__g1_leaf_assignment_count")
        and any(token in str(column).lower() for token in RAW_LEAF_TOKENS)
    )
    if bad:
        raise LeafReasoningClusterMaterializerError(
            f"{source} leaks raw fold-local leaf identifiers: {bad}"
        )


def _strict_manifest(root: Path) -> dict[str, Any]:
    manifest = _json(root / "strict_oof_reasoning_manifest.json")
    if manifest.get("status") != "STRICT_OOF_BASE_REASONING_MATERIALIZED":
        raise LeafReasoningClusterMaterializerError("strict root is not complete")
    if manifest.get("prediction_shards") != "base_prediction_shards/<transport>/<side>/":
        raise LeafReasoningClusterMaterializerError("strict root has an unexpected prediction-shard contract")
    transports = manifest.get("transports")
    if not isinstance(transports, list) or not transports:
        raise LeafReasoningClusterMaterializerError("strict root declares no completed transports")
    return manifest


def _compact_manifest(root: Path) -> dict[str, Any]:
    manifest = _json(root / "base_reasoning_representation_manifest.json")
    if manifest.get("status") != "COMPACT_STRICT_OOF_BASE_REASONING_MATERIALIZED":
        raise LeafReasoningClusterMaterializerError("compact reasoning root is not complete")
    return manifest


def _resolved_artifact_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a manifest path exactly as the producing process recorded it."""

    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _validate_compact_lineage(
    strict_root: Path,
    compact_root: Path,
    compact_manifest: Mapping[str, Any],
) -> None:
    """Prove that C0 came from the same strict artifacts as the taxonomy.

    C candidate values come from ``strict_root``, but C0 is supplied from the
    compact G2/G3 hand-off.  Accepting an arbitrary completed compact root
    would make the C comparison a mixture of two base model generations.
    """

    inputs = compact_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LeafReasoningClusterMaterializerError("compact reasoning manifest lacks source inputs")
    index_value = inputs.get("artifact_index")
    shards_value = inputs.get("prediction_shards_root")
    if not isinstance(index_value, str) or not isinstance(shards_value, str):
        raise LeafReasoningClusterMaterializerError(
            "compact reasoning manifest lacks strict artifact-index/prediction-shard lineage"
        )
    expected_index = (strict_root / "strict_oof_reasoning_artifact_index.parquet").resolve()
    expected_shards = (strict_root / "base_prediction_shards").resolve()
    if _resolved_artifact_path(index_value) != expected_index:
        raise LeafReasoningClusterMaterializerError(
            "compact reasoning artifact index does not match the supplied strict root"
        )
    if _resolved_artifact_path(shards_value) != expected_shards:
        raise LeafReasoningClusterMaterializerError(
            "compact reasoning prediction shards do not match the supplied strict root"
        )
    if not expected_index.is_file() or not expected_shards.is_dir():
        raise LeafReasoningClusterMaterializerError("strict root lacks the compact hand-off inputs")


def _parquet_columns(path: Path) -> list[str]:
    if pq is None:
        raise LeafReasoningClusterMaterializerError(
            "pyarrow is required to read the compact reasoning schema without materialising it"
        )
    try:
        return list(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as exc:
        raise LeafReasoningClusterMaterializerError(f"invalid parquet schema: {path}") from exc


def _prediction_population(root: Path, transports: Sequence[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    required = {
        "candidate_id", "decision_ts", "side_name", "fold_id", "base_expected_bps",
    }
    if len(set(map(str, transports))) != len(transports):
        raise LeafReasoningClusterMaterializerError("strict root declares duplicate transports")
    for transport in map(str, transports):
        for side in ("long", "short"):
            directory = root / "base_prediction_shards" / transport / side
            for filename, partition in (
                ("strict_oof_predictions.parquet", "inner_oof"),
                ("outer_predictions.parquet", "outer_test"),
            ):
                path = directory / filename
                if not path.is_file():
                    raise LeafReasoningClusterMaterializerError(f"strict root is missing {path}")
                frame = pd.read_parquet(path)
                missing = sorted(required.difference(frame.columns))
                if missing:
                    raise LeafReasoningClusterMaterializerError(f"prediction shard lacks {missing}")
                frame = frame.loc[:, sorted(required)].copy()
                frame["decision_ts"] = _utc(frame["decision_ts"], source=str(path))
                frame["candidate_id"] = frame["candidate_id"].astype("string")
                if frame["candidate_id"].isna().any() or frame["candidate_id"].str.strip().eq("").any():
                    raise LeafReasoningClusterMaterializerError("prediction shard has a null or blank candidate ID")
                frame["side_name"] = frame["side_name"].astype(str).str.lower()
                if not frame["side_name"].eq(side).all():
                    raise LeafReasoningClusterMaterializerError("prediction shard crosses its side directory")
                if frame["candidate_id"].duplicated().any():
                    raise LeafReasoningClusterMaterializerError("prediction shard duplicates candidate IDs")
                frame["transport"] = transport
                frame["meta_partition"] = partition
                parts.append(frame)
    population = pd.concat(parts, ignore_index=True)
    # A candidate ID is deliberately stable across chronological transports:
    # the same market opportunity can be re-scored by another transport's
    # independently fitted base stack.  It is therefore not a population key.
    # The materializer must retain each strict prediction under its complete
    # provenance identity rather than silently rejecting (or, worse, blending)
    # those transport-local observations.
    if population.duplicated(list(IDENTITY)).any():
        raise LeafReasoningClusterMaterializerError("strict prediction population duplicates full candidate identity")
    population["base_expected_bps"] = pd.to_numeric(population["base_expected_bps"], errors="coerce")
    if not np.isfinite(population["base_expected_bps"].to_numpy(float)).all():
        raise LeafReasoningClusterMaterializerError("base expected bps must be finite")
    return population.loc[:, list(IDENTITY) + ["base_expected_bps"]].copy()


def _artifact_index(root: Path, transports: set[str]) -> pd.DataFrame:
    path = root / "strict_oof_reasoning_artifact_index.parquet"
    if not path.is_file():
        raise LeafReasoningClusterMaterializerError("strict root lacks the artifact index")
    index = pd.read_parquet(path)
    required = {
        "transport", "side_model", "head_name", "fold_name", "fold_id", "artifact_dir",
        "feature_contract_sha256", "strict_status",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise LeafReasoningClusterMaterializerError(f"strict artifact index lacks {missing}")
    index = index.copy()
    index["transport"] = index["transport"].astype(str)
    index["side_model"] = index["side_model"].astype(str).str.lower()
    if not set(index["transport"]).issubset(transports):
        raise LeafReasoningClusterMaterializerError("strict artifact index has an undeclared transport")
    if not set(index["side_model"]).issubset({"long", "short"}):
        raise LeafReasoningClusterMaterializerError("strict artifact index has an unknown side")
    if not set(index["head_name"].astype(str)).issubset(HEADS):
        raise LeafReasoningClusterMaterializerError("strict artifact index has an unknown head")
    if not index["strict_status"].astype(str).eq("MATERIALIZED_STRICT_OOF").all():
        raise LeafReasoningClusterMaterializerError("strict artifact index contains an incomplete head artifact")
    expected = {(transport, side, head) for transport in transports for side in ("long", "short") for head in HEADS}
    outer = index.loc[index["fold_name"].astype(str).eq("outer")].copy()
    if outer.duplicated(["transport", "side_model", "head_name"]).any():
        raise LeafReasoningClusterMaterializerError("strict artifact index duplicates an outer side/head artifact")
    actual = set(outer.loc[:, ["transport", "side_model", "head_name"]].itertuples(index=False, name=None))
    if actual != expected:
        raise LeafReasoningClusterMaterializerError("strict artifact index lacks a complete outer side/head contract")
    return index.sort_values(["transport", "side_model", "fold_name", "head_name"], kind="stable").reset_index(drop=True)


def _rule_instance_id(
    *, transport: str, fold_id: str, side: str, head: str, direction: str, signature: str,
) -> str:
    text = "|".join((transport, fold_id, side, head, direction, signature))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _catalog_summary(root: Path, index: pd.DataFrame, config: LeafReasoningClusterMaterializerConfig) -> pd.DataFrame:
    """Build a bounded, token-free rule table from frozen base catalogs only."""

    rows: list[pd.DataFrame] = []
    for item in index.itertuples(index=False):
        artifact = root / str(item.artifact_dir)
        manifest = _json(artifact / "base_reasoning_manifest.json")
        if (
            manifest.get("status") != "MATERIALIZED_STRICT_OOF"
            or str(manifest.get("head_name")) != str(item.head_name)
            or str(manifest.get("side_name", "")).lower() != str(item.side_model)
            or str(manifest.get("fold_id")) != str(item.fold_id)
        ):
            raise LeafReasoningClusterMaterializerError(f"catalog manifest lineage mismatch: {artifact}")
        catalog_path = artifact / "leaf_rule_catalog.parquet"
        if not catalog_path.is_file():
            raise LeafReasoningClusterMaterializerError(f"strict artifact lacks catalog: {artifact}")
        catalog = pd.read_parquet(catalog_path)
        required = {
            "head_name", "side_name", "fold_id", "rule_signature", "rule_structural_path_json",
            "rule_feature_signature", "train_leaf_frequency", "train_target_mean",
            "ensemble_tree_contribution",
        }
        missing = sorted(required.difference(catalog.columns))
        if missing:
            raise LeafReasoningClusterMaterializerError(f"catalog lacks cluster summary fields: {missing}")
        if not catalog["head_name"].astype(str).eq(str(item.head_name)).all() or not catalog["side_name"].astype(str).str.lower().eq(str(item.side_model)).all():
            raise LeafReasoningClusterMaterializerError("catalog crosses its strict scope")
        work = catalog.loc[:, list(required)].copy()
        work["ensemble_tree_contribution"] = pd.to_numeric(work["ensemble_tree_contribution"], errors="coerce")
        work["train_target_mean"] = pd.to_numeric(work["train_target_mean"], errors="coerce")
        work["train_leaf_frequency"] = pd.to_numeric(work["train_leaf_frequency"], errors="coerce")
        if not np.isfinite(work[["ensemble_tree_contribution", "train_target_mean", "train_leaf_frequency"]].to_numpy(float)).all():
            raise LeafReasoningClusterMaterializerError("catalog training statistics must be finite")
        work["contribution_direction"] = np.where(work["ensemble_tree_contribution"].to_numpy(float) >= 0.0, "positive", "negative")
        work["__abs_contribution__"] = work["ensemble_tree_contribution"].abs()
        work["__economic_numerator__"] = work["ensemble_tree_contribution"] * work["train_target_mean"]
        grouped = work.groupby(["rule_signature", "contribution_direction"], observed=True, sort=False)
        summary = grouped.agg(
            activation_rate=("train_leaf_frequency", "sum"),
            contribution_mass=("__abs_contribution__", "sum"),
            economic_numerator=("__economic_numerator__", "sum"),
            rule_structural_path_json=("rule_structural_path_json", "first"),
            contribution_signature=("rule_feature_signature", "first"),
        ).reset_index()
        summary["activation_rate"] = summary["activation_rate"].clip(0.0, 1.0)
        summary["economic_effect"] = np.divide(
            summary["economic_numerator"].to_numpy(float),
            summary["contribution_mass"].to_numpy(float),
            out=np.zeros(len(summary), dtype=float), where=summary["contribution_mass"].to_numpy(float) > 0.0,
        )
        summary["transport"] = str(item.transport)
        summary["fold_id"] = str(item.fold_id)
        summary["side_name"] = str(item.side_model)
        summary["head_name"] = str(item.head_name)
        summary["rule_instance_id"] = [
            _rule_instance_id(
                transport=str(item.transport), fold_id=str(item.fold_id), side=str(item.side_model),
                head=str(item.head_name), direction=str(direction), signature=str(signature),
            )
            for signature, direction in zip(summary["rule_signature"], summary["contribution_direction"], strict=True)
        ]
        # This prefilter is a compute control only.  It is based entirely on
        # base-model training catalog statistics, not realised candidate paths.
        summary["__priority__"] = summary["contribution_mass"] * np.maximum(summary["activation_rate"], 1e-8)
        summary = summary.sort_values(["__priority__", "rule_instance_id"], ascending=[False, True], kind="stable").head(int(config.max_rule_instances_per_cell))
        rows.append(summary)
    if not rows:
        raise LeafReasoningClusterMaterializerError("strict root has no catalog summaries")
    result = pd.concat(rows, ignore_index=True)
    folds = result.groupby(["transport", "side_name", "head_name", "contribution_direction"], observed=True)["fold_id"].nunique().rename("__available_folds__")
    recurrence = result.groupby(["transport", "side_name", "head_name", "contribution_direction", "rule_signature"], observed=True)["fold_id"].nunique().rename("__folds__")
    result = result.merge(folds.reset_index(), on=["transport", "side_name", "head_name", "contribution_direction"], how="left", validate="many_to_one")
    result = result.merge(recurrence.reset_index(), on=["transport", "side_name", "head_name", "contribution_direction", "rule_signature"], how="left", validate="many_to_one")
    result["portability_score"] = np.divide(
        result["__folds__"].to_numpy(float), result["__available_folds__"].to_numpy(float),
        out=np.zeros(len(result), dtype=float), where=result["__available_folds__"].to_numpy(float) > 0.0,
    )
    result["contribution_top_features_json"] = result["rule_structural_path_json"].astype(str)
    result = result.drop(columns=["economic_numerator", "__priority__", "__available_folds__", "__folds__"])
    _forbid_raw_leaf(result.columns, source="cluster rule summary")
    if result["rule_instance_id"].duplicated().any():
        raise LeafReasoningClusterMaterializerError("cluster rule summary duplicates rule instances")
    return result.reset_index(drop=True)


def _cluster_assignments(summary: pd.DataFrame, config: LeafReasoningClusterMaterializerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the immutable four-threshold sweep for exactly one safe linkage."""

    assignment_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for arm, threshold in ARM_THRESHOLDS.items():
        result = cluster_leaf_reasoning_signatures(
            summary.loc[:, [
                "rule_instance_id", "fold_id", "side_name", "head_name", "contribution_direction",
                "rule_signature", "rule_structural_path_json", "activation_rate", "contribution_signature",
                "contribution_top_features_json", "economic_effect", "portability_score",
            ]],
            config=LeafReasoningClusterConfig(threshold=float(threshold), linkage=config.linkage),
        )
        assigned = result.assignments.loc[:, ["rule_instance_id", "cluster_id"]].copy()
        assigned["arm"] = arm
        assigned["cluster_id"] = arm + "::" + assigned["cluster_id"].astype(str)
        assignment_parts.append(assigned)
        cluster = result.cluster_summary.copy()
        cluster["arm"] = arm
        cluster["cluster_id"] = arm + "::" + cluster["cluster_id"].astype(str)
        summary_parts.append(cluster)
    assignments = pd.concat(assignment_parts, ignore_index=True)
    clusters = pd.concat(summary_parts, ignore_index=True)
    if assignments.duplicated(["arm", "rule_instance_id"]).any() or clusters.duplicated(["arm", "cluster_id"]).any():
        raise LeafReasoningClusterMaterializerError("threshold sweep produced ambiguous cluster assignments")
    return assignments, clusters


def _top_decile_flags(population: pd.DataFrame) -> pd.DataFrame:
    out = population.loc[:, list(IDENTITY)].copy()
    out["is_top_decile"] = False
    for transport, cell in population.groupby("transport", observed=True, sort=True):
        count = max(1, int(np.ceil(len(cell) * 0.10)))
        order = cell.sort_values(["base_expected_bps", "candidate_id"], ascending=[False, True], kind="stable").head(count)
        out.loc[order.index, "is_top_decile"] = True
    return out


def _family_cache(
    *, root: Path, index: pd.DataFrame, population: pd.DataFrame, rule_summary: pd.DataFrame,
    cache_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Write token-free per-artifact long cache and collect C-selection mass."""

    rule_key = rule_summary.loc[:, ["transport", "fold_id", "side_name", "head_name", "contribution_direction", "rule_signature", "rule_instance_id"]]
    identity_lookup = population.loc[:, list(IDENTITY)].copy()
    flags = _top_decile_flags(population).loc[:, [*IDENTITY, "is_top_decile"]]
    cache_paths: list[Path] = []
    stats: list[pd.DataFrame] = []
    for number, item in enumerate(index.itertuples(index=False)):
        artifact = root / str(item.artifact_dir)
        family = extract_leaf_family_contributions(artifact)
        _forbid_raw_leaf(family.columns, source="family contribution extraction")
        # ``candidate_id`` is not globally unique across the development
        # transports.  The family extractor is intentionally scoped to one
        # fitted artifact, so add that artifact's immutable transport and
        # partition before crossing into the multi-transport population.
        family["transport"] = str(item.transport)
        family["meta_partition"] = "outer_test" if str(item.fold_name) == "outer" else "inner_oof"
        family = family.merge(
            identity_lookup,
            left_on=["candidate_id", "__ts__", "side_name", "fold_id", "transport", "meta_partition"],
            right_on=list(IDENTITY),
            how="left", validate="many_to_one", indicator=True,
        )
        if not family["_merge"].eq("both").all():
            raise LeafReasoningClusterMaterializerError(
                "family contribution extraction contains a candidate outside the strict prediction population"
            )
        # Canonicalise the extractor's ``__ts__`` field to the population's
        # public ``decision_ts`` identity field.  Keep the full identity in
        # the cache; candidate ID alone is only unique inside one transport.
        family = family.drop(columns=["__ts__", "_merge"])
        if family.empty:
            raise LeafReasoningClusterMaterializerError(f"strict artifact produced no family contributions: {artifact}")
        family = family.merge(
            rule_key,
            on=["transport", "fold_id", "side_name", "head_name", "contribution_direction", "rule_signature"],
            how="inner", validate="many_to_one",
        )
        if len(family) == 0:
            raise LeafReasoningClusterMaterializerError("family contribution has no clusterable catalog rule")
        family = family.merge(flags, on=list(IDENTITY), how="inner", validate="many_to_one")
        family["family_ensemble_tree_contribution"] = pd.to_numeric(
            family["family_ensemble_tree_contribution"], errors="coerce"
        )
        if not np.isfinite(family["family_ensemble_tree_contribution"].to_numpy(float)).all():
            raise LeafReasoningClusterMaterializerError("family contribution values must be finite")
        keep = family.loc[:, [*IDENTITY, "head_name", "contribution_direction", "rule_signature", "rule_instance_id", "is_top_decile", "family_ensemble_tree_contribution"]].copy()
        _forbid_raw_leaf(keep.columns, source="cluster family cache")
        path = cache_dir / f"family_{number:04d}.parquet"
        keep.to_parquet(path, index=False, compression="zstd")
        cache_paths.append(path)
    return cache_paths, pd.DataFrame(stats)


def _cluster_mass(
    cache_paths: Sequence[Path], assignments: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in cache_paths:
        family = pd.read_parquet(path)
        long = family.merge(assignments, on="rule_instance_id", how="inner", validate="many_to_many")
        long["abs_contribution"] = long["family_ensemble_tree_contribution"].abs()
        grouped = long.groupby(["arm", "cluster_id", "transport", "is_top_decile"], observed=True, sort=False)["abs_contribution"].sum().rename("abs_contribution").reset_index()
        parts.append(grouped)
    if not parts:
        raise LeafReasoningClusterMaterializerError("cluster cache has no parts")
    return pd.concat(parts, ignore_index=True).groupby(
        ["arm", "cluster_id", "transport", "is_top_decile"], observed=True, sort=False,
    )["abs_contribution"].sum().reset_index()


def _cluster_coverage_scores(
    *, clusters: pd.DataFrame, mass: pd.DataFrame, transports: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-transport contribution coverage without outcome information."""

    expected_transports = tuple(sorted(dict.fromkeys(map(str, transports))))
    if not expected_transports:
        raise LeafReasoningClusterMaterializerError("C-stage selection requires at least one declared transport")
    top = mass.loc[mass["is_top_decile"].astype(bool)].copy()
    if set(top["transport"].astype(str)).difference(expected_transports):
        raise LeafReasoningClusterMaterializerError("cluster contribution mass has an undeclared transport")
    total_index = pd.MultiIndex.from_product([C_ARMS, expected_transports], names=["arm", "transport"])
    total = (
        top.groupby(["arm", "transport"], observed=True)["abs_contribution"].sum()
        .reindex(total_index, fill_value=0.0)
        .rename("denominator")
        .reset_index()
    )
    cluster_keys = clusters.loc[:, ["arm", "cluster_id"]].drop_duplicates()
    full_index = pd.MultiIndex.from_frame(
        cluster_keys.merge(pd.DataFrame({"transport": expected_transports}), how="cross")
    )
    by_cluster = (
        top.groupby(["arm", "cluster_id", "transport"], observed=True)["abs_contribution"].sum()
        .reindex(full_index, fill_value=0.0)
        .rename("numerator")
        .reset_index()
    )
    by_cluster = by_cluster.merge(total, on=["arm", "transport"], how="left", validate="many_to_one")
    by_cluster["coverage"] = np.divide(
        by_cluster["numerator"].to_numpy(float), by_cluster["denominator"].to_numpy(float),
        out=np.zeros(len(by_cluster), dtype=float), where=by_cluster["denominator"].to_numpy(float) > 0.0,
    )
    portable = clusters.loc[:, ["arm", "cluster_id", "fold_coverage_fraction", "economic_effect_mean"]].copy()
    score = by_cluster.groupby(["arm", "cluster_id"], observed=True).agg(
        top_abs_contribution=("numerator", "sum"),
        min_transport_coverage=("coverage", "min"),
        mean_transport_coverage=("coverage", "mean"),
    ).reset_index().merge(portable, on=["arm", "cluster_id"], how="left", validate="one_to_one")
    if score[["top_abs_contribution", "min_transport_coverage", "mean_transport_coverage", "fold_coverage_fraction"]].isna().any().any():
        raise LeafReasoningClusterMaterializerError("cluster contribution coverage score is incomplete")
    return by_cluster, score


def _select_cluster_ids(
    *,
    clusters: pd.DataFrame,
    mass: pd.DataFrame,
    transports: Sequence[str],
    config: LeafReasoningClusterMaterializerConfig,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Freeze only C1--C4 threshold candidates before their meta evaluation.

    Naming an arm C5 or C6 at this point would be circular: C5's threshold is
    selected only after the C1--C4 development economics/MDA comparison and
    C6 needs its own C5-prefix one-SE evidence.  This function deliberately
    emits the bounded threshold sweep plus label-free coverage evidence only.
    """

    _by_cluster, score = _cluster_coverage_scores(
        clusters=clusters, mass=mass, transports=transports,
    )
    selected: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    for arm in C_ARMS:
        local = score.loc[score["arm"].eq(arm)].sort_values(
            ["top_abs_contribution", "cluster_id"], ascending=[False, True], kind="stable",
        ).head(int(config.max_clusters_per_arm))
        if local.empty:
            raise LeafReasoningClusterMaterializerError(f"{arm} has no top-decile contribution support")
        selected[arm] = local["cluster_id"].astype(str).tolist()
        for row in local.itertuples(index=False):
            audit_rows.append({"selection_stage": arm, "selected": True, **row._asdict()})
    return selected, pd.DataFrame(audit_rows)


def _feature_name(cluster_id: str) -> str:
    return "base_reasoning_cluster__" + hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:20]


def _candidate_cluster_features(
    *, population: pd.DataFrame, cache_paths: Sequence[Path], assignments: pd.DataFrame,
    selected: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    selected_ids = {str(item) for arm in C_ARMS for item in selected[arm]}
    feature_by_cluster = {cluster_id: _feature_name(cluster_id) for cluster_id in sorted(selected_ids)}
    if len(set(feature_by_cluster.values())) != len(feature_by_cluster):
        raise LeafReasoningClusterMaterializerError("cluster feature-name hash collision")
    selected_assignment = assignments.loc[assignments["cluster_id"].isin(selected_ids), ["rule_instance_id", "cluster_id"]].copy()
    selected_assignment["feature"] = selected_assignment["cluster_id"].map(feature_by_cluster)
    positions = population.loc[:, list(IDENTITY)].copy()
    positions["__population_position__"] = np.arange(len(population), dtype=np.int64)
    fields = sorted(set(feature_by_cluster.values()))
    field_position = {name: index for index, name in enumerate(fields)}
    values = np.zeros((len(population), len(fields)), dtype=np.float32)
    for path in cache_paths:
        family = pd.read_parquet(
            path,
            columns=[*IDENTITY, "rule_instance_id", "family_ensemble_tree_contribution"],
        )
        mapped = family.merge(selected_assignment, on="rule_instance_id", how="inner", validate="many_to_many")
        if mapped.empty:
            continue
        grouped = mapped.groupby([*IDENTITY, "feature"], observed=True, sort=False)["family_ensemble_tree_contribution"].sum().reset_index()
        grouped = grouped.merge(positions, on=list(IDENTITY), how="left", validate="many_to_one")
        if grouped["__population_position__"].isna().any():
            raise LeafReasoningClusterMaterializerError("cluster family cache contains an unknown candidate")
        column = grouped["feature"].map(field_position)
        np.add.at(
            values,
            (grouped["__population_position__"].to_numpy(dtype=np.int64), column.to_numpy(dtype=np.int64)),
            grouped["family_ensemble_tree_contribution"].to_numpy(np.float32),
        )
    output = population.loc[:, list(IDENTITY)].copy()
    for position, field in enumerate(fields):
        output[field] = values[:, position]
    _forbid_raw_leaf(output.columns, source="candidate cluster feature output")
    if not np.isfinite(output.loc[:, fields].to_numpy(float)).all():
        raise LeafReasoningClusterMaterializerError("candidate cluster features must be finite")
    return output, feature_by_cluster


def _upstream_c0_features(
    compact_root: Path, *, feature_groups: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    if feature_groups is not None:
        required = ["L0", "L2", "L3"]
        missing = [name for name in required if name not in feature_groups]
        if missing:
            raise LeafReasoningClusterMaterializerError(f"upstream feature groups lack {missing}")
        result = list(dict.fromkeys([*map(str, feature_groups["L0"]), *map(str, feature_groups["L2"]), *map(str, feature_groups["L3"])]))
        _forbid_raw_leaf(result, source="C0 upstream feature groups")
        if not result:
            raise LeafReasoningClusterMaterializerError("C0 upstream compact representation is empty")
        return result
    path = compact_root / "base_reasoning_features_oof.parquet"
    if not path.is_file():
        raise LeafReasoningClusterMaterializerError("compact root lacks base reasoning feature table")
    columns = _parquet_columns(path)
    compact = [name for name in columns if "__g2_" in str(name) or "__g3_" in str(name)]
    result = ["p_adverse", "p_weak", "p_clear", "base_expected_bps", *FROZEN_META_CONTEXT, *sorted(map(str, compact))]
    _forbid_raw_leaf(result, source="C0 inferred compact representation")
    return list(dict.fromkeys(result))


def materialize_leaf_reasoning_cluster_candidates(
    strict_root: str | os.PathLike[str],
    compact_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    config: LeafReasoningClusterMaterializerConfig = LeafReasoningClusterMaterializerConfig(),
    upstream_feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> LeafReasoningClusterMaterializationResult:
    """Materialise token-free C1--C4 threshold-candidate features atomically."""

    config.validate()
    strict_root, compact_root, target = Path(strict_root), Path(compact_root), Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite C-stage artifact: {target}")
    strict_manifest = _strict_manifest(strict_root)
    compact_manifest = _compact_manifest(compact_root)
    _validate_compact_lineage(strict_root, compact_root, compact_manifest)
    transports = [str(value) for value in strict_manifest["transports"]]
    population = _prediction_population(strict_root, transports)
    index = _artifact_index(strict_root, set(transports))
    summary = _catalog_summary(strict_root, index, config)
    assignments, clusters = _cluster_assignments(summary, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        cache = temporary / "family_cache"
        cache.mkdir()
        cache_paths, _unused = _family_cache(
            root=strict_root, index=index, population=population, rule_summary=summary, cache_dir=cache,
        )
        mass = _cluster_mass(cache_paths, assignments)
        selected, selection_audit = _select_cluster_ids(
            clusters=clusters, mass=mass, transports=transports, config=config,
        )
        candidate_features, feature_by_cluster = _candidate_cluster_features(
            population=population, cache_paths=cache_paths, assignments=assignments, selected=selected,
        )
        c0 = _upstream_c0_features(compact_root, feature_groups=upstream_feature_groups)
        groups = {
            "C0": c0,
            **{
                arm: [feature_by_cluster[str(cluster_id)] for cluster_id in selected[arm]]
                for arm in C_ARMS
            },
        }
        missing_feature = sorted(
            set(name for arm in C_ARMS for name in groups[arm]).difference(candidate_features.columns)
        )
        if missing_feature:
            raise LeafReasoningClusterMaterializerError(f"selected cluster groups lack candidate fields: {missing_feature}")
        output_files: dict[str, str] = {}
        tables = {
            "candidate_cluster_features.parquet": candidate_features,
            "cluster_rule_summary.parquet": summary.drop(columns=[name for name in summary if name.startswith("__")]),
            "cluster_assignments.parquet": assignments,
            "cluster_summary.parquet": clusters,
            "cluster_selection_audit.parquet": selection_audit,
            "cluster_contribution_mass.parquet": mass,
        }
        for name, table in tables.items():
            _forbid_raw_leaf(table.columns, source=name)
            path = temporary / name
            table.to_parquet(path, index=False, compression="zstd")
            output_files[name] = _sha256(path)
        taxonomy = {
            "selection_phase": THRESHOLD_SWEEP_PHASE,
            "linkage": config.linkage,
            "cluster_ids_by_arm": selected,
            "threshold_by_arm": {arm: float(value) for arm, value in ARM_THRESHOLDS.items()},
            "selection_metric": "bounded label-free candidate coverage only; C5/C6 require post-C1--C4 development selection",
            "exploratory_hard_cap": int(config.max_clusters_per_arm),
            "production_soft_cap": int(config.c6_soft_cap),
        }
        (temporary / "cluster_groups.json").write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "cluster_taxonomy_contract.json").write_text(json.dumps(_safe(taxonomy), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "cluster_feature_manifest.json").write_text(
            json.dumps({"schema": SCHEMA, "cluster_id_to_feature": feature_by_cluster}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("cluster_groups.json", "cluster_taxonomy_contract.json", "cluster_feature_manifest.json"):
            output_files[name] = _sha256(temporary / name)
        manifest = {
            "schema": SCHEMA,
            "status": STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "strict_root": str(strict_root),
            "compact_root": str(compact_root),
            "strict_manifest_sha256": _sha256(strict_root / "strict_oof_reasoning_manifest.json"),
            "compact_manifest_sha256": _sha256(compact_root / "base_reasoning_representation_manifest.json"),
            "transports": transports,
            "config": asdict(config),
            "contract": {
                "source": "completed strict OOF artifacts and their fitted base catalogs only",
                "taxonomy": "development-only fixed .60/.70/.80/.90 threshold sweep with one average-or-complete linkage",
                "outer_outcomes": "never read for taxonomy or cluster features",
                "raw_leaf_ids": "rejected before every persisted C-stage artifact",
                "c5_c6": "not issued here; the post-C1--C4 selector requires immutable development economics and grouped MDA",
            },
            "row_counts": {"candidates": int(len(candidate_features)), "rule_instances": int(len(summary))},
            "cluster_counts": {arm: int(len(selected[arm])) for arm in C_ARMS},
            "outputs": output_files,
        }
        (temporary / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return LeafReasoningClusterMaterializationResult(
            target, int(len(candidate_features)), len(feature_by_cluster),
            {arm: int(len(selected[arm])) for arm in C_ARMS}, taxonomy,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_leaf_reasoning_cluster_candidate_artifact(
    root: str | os.PathLike[str],
) -> LeafReasoningClusterCandidateArtifact:
    """Load a complete C-stage root and prove its compact feature contract.

    The meta runner must never accept an arbitrary parquet table plus a JSON
    group list.  This loader verifies every declared output hash, the exact
    candidate identity, and the ID-to-feature relationship before the table is
    allowed to cross the strict base-to-meta boundary.
    """

    source = Path(root)
    manifest = _json(source / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise LeafReasoningClusterMaterializerError("cluster candidate root is not complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise LeafReasoningClusterMaterializerError("cluster candidate manifest lacks output hashes")
    required_files = {
        "candidate_cluster_features.parquet",
        "cluster_rule_summary.parquet",
        "cluster_assignments.parquet",
        "cluster_summary.parquet",
        "cluster_selection_audit.parquet",
        "cluster_contribution_mass.parquet",
        "cluster_groups.json",
        "cluster_taxonomy_contract.json",
        "cluster_feature_manifest.json",
    }
    missing_files = sorted(required_files.difference(outputs))
    if missing_files:
        raise LeafReasoningClusterMaterializerError(
            f"cluster candidate manifest lacks output hashes for {missing_files}"
        )
    for name in sorted(required_files):
        expected = outputs.get(name)
        path = source / name
        if not isinstance(expected, str) or not path.is_file() or _sha256(path) != expected:
            raise LeafReasoningClusterMaterializerError(
                f"cluster candidate output hash mismatch: {path}"
            )

    taxonomy = _json(source / "cluster_taxonomy_contract.json")
    phase = str(taxonomy.get("selection_phase", FINAL_PHASE)).lower()
    if phase not in {THRESHOLD_SWEEP_PHASE, FINAL_PHASE}:
        raise LeafReasoningClusterMaterializerError("cluster taxonomy has an unknown selection phase")
    expected_cluster_arms = set(C_ARMS)
    if phase == FINAL_PHASE:
        expected_cluster_arms |= {"C5", "C6"}

    groups_raw = _json(source / "cluster_groups.json")
    expected_groups = {"C0", *expected_cluster_arms}
    if set(groups_raw) != expected_groups or not all(
        isinstance(values, list) and all(isinstance(value, str) and value for value in values)
        for values in groups_raw.values()
    ):
        raise LeafReasoningClusterMaterializerError(
            "cluster groups must contain exactly non-empty C0--C6 string lists"
        )
    groups = {str(name): list(map(str, values)) for name, values in groups_raw.items()}
    _forbid_raw_leaf([field for fields in groups.values() for field in fields], source="cluster group contract")
    if not groups["C0"]:
        raise LeafReasoningClusterMaterializerError("cluster group C0 must retain an upstream compact representation")

    raw_ids = taxonomy.get("cluster_ids_by_arm")
    if not isinstance(raw_ids, Mapping) or set(map(str, raw_ids)) != expected_cluster_arms:
        raise LeafReasoningClusterMaterializerError("cluster taxonomy lacks the exact phase-specific ID contract")
    ids = {str(arm): list(map(str, values)) for arm, values in raw_ids.items()}
    if not all(isinstance(values, list) and all(isinstance(value, str) and value for value in values) for values in raw_ids.values()):
        raise LeafReasoningClusterMaterializerError("cluster taxonomy IDs must be non-empty string lists")

    feature_manifest = _json(source / "cluster_feature_manifest.json")
    if feature_manifest.get("schema") != SCHEMA:
        raise LeafReasoningClusterMaterializerError("cluster feature manifest schema mismatch")
    raw_mapping = feature_manifest.get("cluster_id_to_feature")
    if not isinstance(raw_mapping, Mapping):
        raise LeafReasoningClusterMaterializerError("cluster feature manifest lacks ID-to-feature mapping")
    feature_by_cluster = {str(key): str(value) for key, value in raw_mapping.items()}
    if any(not key or not value for key, value in feature_by_cluster.items()):
        raise LeafReasoningClusterMaterializerError("cluster feature manifest contains blank mapping values")
    if len(set(feature_by_cluster.values())) != len(feature_by_cluster):
        raise LeafReasoningClusterMaterializerError("cluster feature manifest maps multiple IDs to one feature")
    selected_ids = {cluster_id for arm in expected_cluster_arms for cluster_id in ids[arm]}
    if set(feature_by_cluster) != selected_ids:
        raise LeafReasoningClusterMaterializerError(
            "cluster feature manifest IDs disagree with the taxonomy contract"
        )
    for arm in expected_cluster_arms:
        expected = [feature_by_cluster[cluster_id] for cluster_id in ids[arm]]
        if groups[arm] != expected:
            raise LeafReasoningClusterMaterializerError(
                f"cluster group {arm} does not match the immutable taxonomy mapping"
            )

    path = source / "candidate_cluster_features.parquet"
    features = pd.read_parquet(path)
    _forbid_raw_leaf(features.columns, source="candidate cluster feature table")
    missing_identity = sorted(set(IDENTITY).difference(features.columns))
    missing_fields = sorted(set(feature_by_cluster.values()).difference(features.columns))
    unexpected_fields = sorted(set(features.columns).difference({*IDENTITY, *feature_by_cluster.values()}))
    if missing_identity or missing_fields or unexpected_fields:
        raise LeafReasoningClusterMaterializerError(
            "candidate cluster table does not exactly match its feature manifest "
            f"(missing_identity={missing_identity}, missing_fields={missing_fields}, "
            f"unexpected_fields={unexpected_fields})"
        )
    features = features.loc[:, [*IDENTITY, *sorted(feature_by_cluster.values())]].copy()
    features["candidate_id"] = features["candidate_id"].astype("string")
    features["decision_ts"] = _utc(features["decision_ts"], source="candidate cluster decision_ts")
    features["side_name"] = features["side_name"].astype(str).str.lower()
    if features["candidate_id"].isna().any() or features["candidate_id"].str.strip().eq("").any():
        raise LeafReasoningClusterMaterializerError("candidate cluster feature table has a null or blank candidate ID")
    if not set(features["side_name"]).issubset({"long", "short"}):
        raise LeafReasoningClusterMaterializerError("candidate cluster feature table has an unknown side")
    if not set(features["meta_partition"].astype(str)).issubset(PARTITIONS):
        raise LeafReasoningClusterMaterializerError("candidate cluster feature table has an unknown meta partition")
    if features.duplicated(list(IDENTITY)).any():
        raise LeafReasoningClusterMaterializerError("candidate cluster feature table duplicates identity")
    values = features.loc[:, sorted(feature_by_cluster.values())].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise LeafReasoningClusterMaterializerError("candidate cluster feature values must be finite")
    return LeafReasoningClusterCandidateArtifact(source, features, groups, taxonomy, manifest)


def _strict_numeric_table(
    table: pd.DataFrame,
    *,
    required: set[str],
    source: str,
    categorical: set[str] | None = None,
) -> pd.DataFrame:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise LeafReasoningClusterMaterializerError(f"{source} lacks required columns: {missing}")
    result = table.copy()
    categorical = {"arm", "transport_id"} | (set() if categorical is None else set(categorical))
    for name in required.difference(categorical):
        result[name] = pd.to_numeric(result[name], errors="coerce")
        if not np.isfinite(result[name].to_numpy(float)).all():
            raise LeafReasoningClusterMaterializerError(f"{source}.{name} must be finite")
    if "arm" in required:
        result["arm"] = result["arm"].astype(str)
    if "transport_id" in required:
        result["transport_id"] = result["transport_id"].astype(str)
    return result


def _cluster_ids_json(value: object, *, source: str) -> list[str]:
    if not isinstance(value, str):
        raise LeafReasoningClusterMaterializerError(f"{source} must be a JSON string list")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LeafReasoningClusterMaterializerError(f"{source} is not valid JSON") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item for item in parsed):
        raise LeafReasoningClusterMaterializerError(f"{source} must be a non-empty JSON string list")
    return list(parsed)


def _threshold_winner_from_development(
    artifact: LeafReasoningClusterCandidateArtifact,
    metrics: pd.DataFrame,
    mda: pd.DataFrame,
    *,
    config: LeafReasoningClusterFinalizationConfig,
) -> tuple[str, pd.DataFrame]:
    """Select C5's threshold from immutable C1--C4 economics and MDA only."""

    expected_transports = tuple(sorted(map(str, artifact.manifest.get("transports", ()))))
    if len(expected_transports) < 2:
        raise LeafReasoningClusterMaterializerError("C5 threshold selection requires both development transports")
    metric_required = {
        "arm", "transport_id", "top_fraction", "incremental_global_top_k_net_bps_vs_control",
        "worst_month_net_bps_delta_vs_control",
    }
    metric = _strict_numeric_table(metrics, required=metric_required, source="C1--C4 development metrics")
    metric = metric.loc[metric["arm"].isin(C_ARMS) & metric["top_fraction"].isin((.05, .10))].copy()
    if metric.duplicated(["arm", "transport_id", "top_fraction"]).any():
        raise LeafReasoningClusterMaterializerError("C1--C4 development metrics duplicate an arm/transport/tail row")
    mda_required = {
        "arm", "transport_id", "cluster_ids_json", "transport_mda_bps", "phantom_q95_bps", "positive_environment_rate",
    }
    grouped_mda = _strict_numeric_table(
        mda, required=mda_required, source="C1--C4 grouped MDA", categorical={"cluster_ids_json"},
    )
    grouped_mda = grouped_mda.loc[grouped_mda["arm"].isin(C_ARMS)].copy()
    if grouped_mda.duplicated(["arm", "transport_id"]).any():
        raise LeafReasoningClusterMaterializerError("C1--C4 grouped MDA duplicates an arm/transport row")
    audit: list[dict[str, Any]] = []
    eligible: list[tuple[float, float, float, int, str]] = []
    for arm in C_ARMS:
        arm_metric = metric.loc[metric["arm"].eq(arm)]
        arm_mda = grouped_mda.loc[grouped_mda["arm"].eq(arm)]
        complete_metric = {
            (str(row.transport_id), float(row.top_fraction))
            for row in arm_metric.itertuples(index=False)
        } == {(transport, fraction) for transport in expected_transports for fraction in (.05, .10)}
        complete_mda = set(arm_mda["transport_id"]) == set(expected_transports) and len(arm_mda) == len(expected_transports)
        top05 = arm_metric.loc[arm_metric["top_fraction"].eq(.05), "incremental_global_top_k_net_bps_vs_control"]
        top10 = arm_metric.loc[arm_metric["top_fraction"].eq(.10), "incremental_global_top_k_net_bps_vs_control"]
        expected_ids = list(map(str, artifact.taxonomy["cluster_ids_by_arm"][arm]))
        contract_ok = bool(complete_mda and all(
            _cluster_ids_json(value, source="C1--C4 grouped MDA.cluster_ids_json") == expected_ids
            for value in arm_mda["cluster_ids_json"]
        ))
        mda_values = arm_mda["transport_mda_bps"]
        mda_above_phantom = bool(complete_mda and (mda_values > arm_mda["phantom_q95_bps"]).all())
        environment_ok = bool(complete_mda and (arm_mda["positive_environment_rate"] >= float(config.minimum_positive_environment_rate)).all())
        worst_ok = True
        if config.max_worst_month_net_drop_bps is not None:
            worst_ok = bool(
                complete_metric
                and (arm_metric["worst_month_net_bps_delta_vs_control"] >= float(config.max_worst_month_net_drop_bps)).all()
            )
        economics_ok = bool(complete_metric and (top05 > 0.0).all() and (top10 > 0.0).all())
        median_mda = float(mda_values.median()) if complete_mda else float("nan")
        mad_mda = float((mda_values - median_mda).abs().median()) if complete_mda else float("nan")
        stable_mda = median_mda - .5 * mad_mda if complete_mda else float("nan")
        accepted = bool(complete_metric and complete_mda and contract_ok and economics_ok and mda_above_phantom and environment_ok and worst_ok and stable_mda > 0.0)
        audit.append({
            "arm": arm,
            "complete_metrics": complete_metric,
            "complete_grouped_mda": complete_mda,
            "cluster_contract_matches_threshold_arm": contract_ok,
            "economics_positive_both_transports": economics_ok,
            "mda_above_phantom_both_transports": mda_above_phantom,
            "positive_environment_rate_pass": environment_ok,
            "worst_month_pass": worst_ok,
            "stable_transport_mda_bps": stable_mda,
            "median_top05_net_lift_bps": float(top05.median()) if complete_metric else float("nan"),
            "median_top10_net_lift_bps": float(top10.median()) if complete_metric else float("nan"),
            "feature_count": len(artifact.groups[arm]),
            "accepted": accepted,
        })
        if accepted:
            eligible.append((stable_mda, float(top10.median()), float(top05.median()), len(artifact.groups[arm]), arm))
    if not eligible:
        raise LeafReasoningClusterMaterializerError(
            "no C1--C4 threshold passed immutable economics, grouped-MDA, environment, and worst-month gates"
        )
    # Stable grouped MDA is primary; economics, compactness, and arm name only
    # break exact development ties.  All quantities are C1--C4 development data.
    winner = sorted(eligible, key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))[0][-1]
    return winner, pd.DataFrame(audit)


def _c5_ids_from_coverage(
    artifact: LeafReasoningClusterCandidateArtifact,
    source_arm: str,
    *,
    config: LeafReasoningClusterFinalizationConfig,
) -> _C5CoverageSelection:
    """Select a portable C5 prefix and preserve the coverage accounting.

    The C1--C4 bounded source set and C5 portability gate serve different
    purposes.  This function records both explicitly: an unconstrained
    contribution prefix, the portable prefix allowed to advance, and the
    important source families which are unstable or were not matched by the
    bounded source set.  In particular, unstable contribution can never be
    silently used to make the portable 95% gate pass.
    """

    root = artifact.root
    clusters = pd.read_parquet(root / "cluster_summary.parquet")
    mass = pd.read_parquet(root / "cluster_contribution_mass.parquet")
    by_cluster, score = _cluster_coverage_scores(
        clusters=clusters,
        mass=mass,
        transports=tuple(map(str, artifact.manifest["transports"])),
    )
    expected_transports = tuple(sorted(map(str, artifact.manifest["transports"])))
    available = set(map(str, artifact.taxonomy["cluster_ids_by_arm"][source_arm]))
    source_score = score.loc[score["arm"].eq(source_arm)].copy()
    if source_score.empty:
        raise LeafReasoningClusterMaterializerError("winning C threshold has no contribution score")
    source_score["cluster_id"] = source_score["cluster_id"].astype(str)
    source_score["in_coverage_candidate_set"] = source_score["cluster_id"].isin(available)
    source_score["portable"] = source_score["fold_coverage_fraction"].ge(float(config.c5_min_portability))
    source_score["coverage_class"] = np.select(
        [
            source_score["in_coverage_candidate_set"] & source_score["portable"],
            source_score["in_coverage_candidate_set"],
        ],
        ["portable", "unstable"],
        default="unmatched",
    )

    def coverage_for(cluster_ids: Sequence[str]) -> dict[str, float]:
        ids = set(map(str, cluster_ids))
        local = by_cluster.loc[
            by_cluster["arm"].eq(source_arm) & by_cluster["cluster_id"].astype(str).isin(ids)
        ]
        values = local.groupby("transport", observed=True)["coverage"].sum()
        return {transport: float(values.get(transport, 0.0)) for transport in expected_transports}

    def greedy_prefix(
        ranked: pd.DataFrame,
        *,
        selection_kind: str,
    ) -> tuple[list[str], dict[str, float], pd.DataFrame]:
        selected: list[str] = []
        coverage_by_transport = {transport: 0.0 for transport in expected_transports}
        audit_rows: list[dict[str, Any]] = []
        for row in ranked.itertuples(index=False):
            cluster_id = str(row.cluster_id)
            selected.append(cluster_id)
            increment = coverage_for([cluster_id])
            for transport in expected_transports:
                coverage_by_transport[transport] += increment[transport]
            audit_rows.append({
                "selection_kind": selection_kind,
                "cluster_id": cluster_id,
                "selected_prefix_size": len(selected),
                "min_transport_coverage": min(coverage_by_transport.values()),
                **{f"coverage__{key}": value for key, value in coverage_by_transport.items()},
            })
            if min(coverage_by_transport.values()) + 1e-12 >= float(config.c5_top_decile_coverage_target):
                break
        return selected, coverage_by_transport, pd.DataFrame(audit_rows)

    rank_columns = ["top_abs_contribution", "cluster_id"]
    coverage_ranked = source_score.loc[source_score["in_coverage_candidate_set"]].sort_values(
        rank_columns, ascending=[False, True], kind="stable",
    )
    if coverage_ranked.empty:
        raise LeafReasoningClusterMaterializerError("winning C threshold has no materialised coverage cluster")
    coverage_ids, coverage_by_transport, coverage_audit = greedy_prefix(
        coverage_ranked, selection_kind="coverage",
    )
    portable_ranked = source_score.loc[
        source_score["in_coverage_candidate_set"] & source_score["portable"]
    ].sort_values(rank_columns, ascending=[False, True], kind="stable")
    if portable_ranked.empty:
        raise LeafReasoningClusterMaterializerError("winning C threshold has no portable materialised cluster")
    selected, selected_coverage_by_transport, portable_audit = greedy_prefix(
        portable_ranked, selection_kind="portable",
    )
    if min(selected_coverage_by_transport.values()) + 1e-12 < float(config.c5_top_decile_coverage_target):
        raise LeafReasoningClusterMaterializerError(
            "C5 95%-coverage portability gate failed for the post-C1--C4 winning threshold; "
            + json.dumps(selected_coverage_by_transport, sort_keys=True)
        )

    manifest_columns = [
        "arm", "cluster_id", "top_abs_contribution", "min_transport_coverage",
        "mean_transport_coverage", "fold_coverage_fraction", "economic_effect_mean",
        "in_coverage_candidate_set", "portable", "coverage_class",
    ]
    coverage_manifest = source_score.loc[
        source_score["cluster_id"].isin(coverage_ids), manifest_columns
    ].copy()
    coverage_manifest["manifest_role"] = "minimal_coverage_prefix"
    coverage_manifest["selected_for_c5"] = coverage_manifest["cluster_id"].isin(selected)
    portable_manifest = source_score.loc[
        source_score["cluster_id"].isin(selected), manifest_columns
    ].copy()
    portable_manifest["manifest_role"] = "portable_c5_prefix"
    portable_manifest["selected_for_c5"] = True
    diagnostic_manifest = source_score.loc[
        source_score["coverage_class"].isin(("unstable", "unmatched"))
        & source_score["top_abs_contribution"].gt(0.0), manifest_columns
    ].copy()
    diagnostic_manifest["manifest_role"] = "important_unstable_or_unmatched"
    diagnostic_manifest["selected_for_c5"] = False

    report_rows: list[dict[str, Any]] = []
    classes = {
        "coverage_manifest": coverage_ids,
        "portable": source_score.loc[source_score["coverage_class"].eq("portable"), "cluster_id"].astype(str).tolist(),
        "selected_portable": selected,
        "unstable": source_score.loc[source_score["coverage_class"].eq("unstable"), "cluster_id"].astype(str).tolist(),
        "unmatched": source_score.loc[source_score["coverage_class"].eq("unmatched"), "cluster_id"].astype(str).tolist(),
    }
    class_coverage = {name: coverage_for(ids) for name, ids in classes.items()}
    for transport in expected_transports:
        report_rows.append({
            "source_arm": source_arm,
            "transport": transport,
            "coverage_target": float(config.c5_top_decile_coverage_target),
            "coverage_manifest_contribution_coverage": class_coverage["coverage_manifest"][transport],
            "portable_contribution_coverage": class_coverage["portable"][transport],
            "selected_portable_contribution_coverage": class_coverage["selected_portable"][transport],
            "unstable_contribution_coverage": class_coverage["unstable"][transport],
            "unmatched_contribution_coverage": class_coverage["unmatched"][transport],
            "coverage_accounted_fraction": (
                class_coverage["portable"][transport]
                + class_coverage["unstable"][transport]
                + class_coverage["unmatched"][transport]
            ),
        })
    combined_audit = pd.concat([coverage_audit, portable_audit], ignore_index=True)
    return _C5CoverageSelection(
        selected_ids=selected,
        selected_coverage_by_transport=selected_coverage_by_transport,
        prefix_audit=combined_audit,
        coverage_manifest=coverage_manifest.sort_values(rank_columns, ascending=[False, True], kind="stable").reset_index(drop=True),
        portable_manifest=portable_manifest.sort_values(rank_columns, ascending=[False, True], kind="stable").reset_index(drop=True),
        diagnostic_manifest=diagnostic_manifest.sort_values(rank_columns, ascending=[False, True], kind="stable").reset_index(drop=True),
        coverage_report=pd.DataFrame(report_rows),
    )


def _c6_ids_from_prefix_mda(
    c5_ids: Sequence[str],
    prefix_mda: pd.DataFrame,
    *,
    transports: Sequence[str],
    config: LeafReasoningClusterFinalizationConfig,
) -> tuple[list[str], float, float, float, pd.DataFrame]:
    """Apply the one-SE rule to predeclared C5 prefix grouped-MDA evidence."""

    required = {"prefix_size", "transport_id", "cluster_ids_json", "transport_mda_bps", "phantom_q95_bps", "positive_environment_rate"}
    table = _strict_numeric_table(
        prefix_mda, required=required, source="C5-prefix grouped MDA", categorical={"cluster_ids_json"},
    )
    expected_transports = tuple(sorted(map(str, transports)))
    max_prefix = min(len(c5_ids), int(config.c6_soft_cap))
    if max_prefix < 1:
        raise LeafReasoningClusterMaterializerError("C5 has no clusters eligible for C6")
    raw_prefix = table["prefix_size"].to_numpy(float)
    if not np.equal(raw_prefix, np.floor(raw_prefix)).all():
        raise LeafReasoningClusterMaterializerError("C5-prefix MDA prefix_size must be an integer")
    table["prefix_size"] = raw_prefix.astype(int)
    if not table["prefix_size"].between(1, max_prefix).all():
        raise LeafReasoningClusterMaterializerError("C5-prefix MDA declares a prefix outside the predeclared compact cap")
    if table.duplicated(["prefix_size", "transport_id"]).any():
        raise LeafReasoningClusterMaterializerError("C5-prefix MDA duplicates a prefix/transport row")
    expected_sizes = set(range(1, max_prefix + 1))
    if set(table["prefix_size"]) != expected_sizes:
        raise LeafReasoningClusterMaterializerError(
            "C5-prefix MDA must cover every predeclared prefix through the compact cap"
        )
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[int, float, float]] = []
    for prefix_size in range(1, max_prefix + 1):
        local = table.loc[table["prefix_size"].eq(prefix_size)]
        complete = set(local["transport_id"]) == set(expected_transports) and len(local) == len(expected_transports)
        expected_ids = list(map(str, c5_ids[:prefix_size]))
        contract_ok = bool(complete and all(
            _cluster_ids_json(value, source="C5-prefix grouped MDA.cluster_ids_json") == expected_ids
            for value in local["cluster_ids_json"]
        ))
        values = local["transport_mda_bps"]
        mean = float(values.mean()) if complete else float("nan")
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if complete and len(values) > 1 else 0.0
        above_phantom = bool(complete and (values > local["phantom_q95_bps"]).all())
        environment = bool(complete and (local["positive_environment_rate"] >= float(config.minimum_positive_environment_rate)).all())
        accepted = bool(complete and contract_ok and above_phantom and environment and mean > 0.0)
        rows.append({
            "prefix_size": prefix_size, "complete_grouped_mda": complete,
            "cluster_contract_matches_prefix": contract_ok,
            "mean_transport_mda_bps": mean, "standard_error_bps": sem,
            "mda_above_phantom_both_transports": above_phantom,
            "positive_environment_rate_pass": environment, "accepted": accepted,
        })
        if accepted:
            eligible.append((prefix_size, mean, sem))
    if not eligible:
        raise LeafReasoningClusterMaterializerError("no C5 prefix passed grouped-MDA and environment gates for C6")
    best_size, best_score, best_sem = sorted(eligible, key=lambda item: (-item[1], item[0]))[0]
    _ = best_size
    selected_size, compact_score, _compact_sem = next(
        item for item in sorted(eligible, key=lambda item: item[0])
        if item[1] + 1e-12 >= best_score - best_sem
    )
    return list(map(str, c5_ids[:selected_size])), best_score, best_sem, compact_score, pd.DataFrame(rows)


def finalize_leaf_reasoning_cluster_taxonomy(
    candidate_root: str | os.PathLike[str],
    development_metrics: pd.DataFrame,
    grouped_mda: pd.DataFrame,
    c5_prefix_mda: pd.DataFrame,
    output_dir: str | os.PathLike[str],
    *,
    config: LeafReasoningClusterFinalizationConfig = LeafReasoningClusterFinalizationConfig(),
) -> Path:
    """Issue an immutable final C5/C6 overlay after the sequential evidence.

    ``development_metrics`` and ``grouped_mda`` select C5's source threshold
    from C1--C4.  ``c5_prefix_mda`` is the separately materialised nested
    prefix sweep required to choose C6 by a genuine one-standard-error rule.
    No outer outcome is ever read while candidate clusters are created; this
    later selector uses only declared development evidence and records hashes.
    """

    config.validate()
    candidate = load_leaf_reasoning_cluster_candidate_artifact(candidate_root)
    if str(candidate.taxonomy.get("selection_phase", "")).lower() != THRESHOLD_SWEEP_PHASE:
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization requires a C1--C4 threshold-sweep candidate root")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite C-stage finalization: {target}")
    source_arm, threshold_audit = _threshold_winner_from_development(
        candidate, development_metrics, grouped_mda, config=config,
    )
    c5_selection = _c5_ids_from_coverage(
        candidate, source_arm, config=config,
    )
    c5_ids = c5_selection.selected_ids
    coverage_by_transport = c5_selection.selected_coverage_by_transport
    c6_ids, best_score, best_sem, compact_score, c6_audit = _c6_ids_from_prefix_mda(
        c5_ids, c5_prefix_mda, transports=candidate.manifest["transports"], config=config,
    )
    feature_mapping = _json(candidate.root / "cluster_feature_manifest.json")["cluster_id_to_feature"]
    coverage_report = c5_selection.coverage_report.set_index("transport")
    coverage_classes = {
        "coverage_manifest": {
            transport: float(coverage_report.loc[transport, "coverage_manifest_contribution_coverage"])
            for transport in coverage_report.index
        },
        "portable": {
            transport: float(coverage_report.loc[transport, "portable_contribution_coverage"])
            for transport in coverage_report.index
        },
        "selected_portable": {
            transport: float(coverage_report.loc[transport, "selected_portable_contribution_coverage"])
            for transport in coverage_report.index
        },
        "unstable": {
            transport: float(coverage_report.loc[transport, "unstable_contribution_coverage"])
            for transport in coverage_report.index
        },
        "unmatched": {
            transport: float(coverage_report.loc[transport, "unmatched_contribution_coverage"])
            for transport in coverage_report.index
        },
    }
    taxonomy = {
        "selection_phase": FINAL_PHASE,
        "linkage": candidate.taxonomy["linkage"],
        "cluster_ids_by_arm": {
            **{arm: list(candidate.taxonomy["cluster_ids_by_arm"][arm]) for arm in C_ARMS},
            "C5": c5_ids,
            "C6": c6_ids,
        },
        "threshold_by_arm": dict(candidate.taxonomy["threshold_by_arm"]),
        "c5_source_arm": source_arm,
        "c6_source_arm": "C5",
        "top_decile_coverage_target": float(config.c5_top_decile_coverage_target),
        "top_decile_coverage_by_arm": {"C5": float(min(coverage_by_transport.values()))},
        "portable_top_decile_coverage_by_arm": {"C5": float(min(coverage_by_transport.values()))},
        "c5_contribution_coverage_by_class": coverage_classes,
        "c5_manifest_contract": {
            "coverage_manifest": "minimal materialised C1--C4-source prefix covering top-decile contribution",
            "portable_manifest": "portable C5 prefix permitted to advance after the portability gate",
            "diagnostic_manifest": "important unstable or unmatched contribution families; never promoted to C5",
        },
        "production_soft_cap": int(config.c6_soft_cap),
        "exploratory_hard_cap": 20,
        "c6_best_cross_era_score": float(best_score),
        "c6_best_cross_era_standard_error": float(best_sem),
        "c6_compact_cross_era_score": float(compact_score),
        "selection_metric": "C5: C1--C4 development economics plus grouped MDA; C6: C5-prefix grouped-MDA one-SE",
    }
    groups = {
        "C0": list(candidate.groups["C0"]),
        **{arm: list(candidate.groups[arm]) for arm in C_ARMS},
        "C5": [str(feature_mapping[cluster_id]) for cluster_id in c5_ids],
        "C6": [str(feature_mapping[cluster_id]) for cluster_id in c6_ids],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        tables = {
            "threshold_selection_audit.parquet": threshold_audit,
            "c5_coverage_prefix_audit.parquet": c5_selection.prefix_audit,
            "c5_contribution_coverage_report.parquet": c5_selection.coverage_report,
            "c6_prefix_one_se_audit.parquet": c6_audit,
        }
        outputs: dict[str, str] = {}
        for name, table in tables.items():
            path = temporary / name
            table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = _sha256(path)
        manifest_metadata = {
            "schema": SCHEMA,
            "source_arm": source_arm,
            "coverage_target": float(config.c5_top_decile_coverage_target),
            "minimum_portability": float(config.c5_min_portability),
            "contribution_coverage_by_class": coverage_classes,
        }
        for name, value in {
            "cluster_groups.json": groups,
            "cluster_taxonomy_contract.json": taxonomy,
            "coverage_manifest.json": {
                **manifest_metadata,
                "role": "minimal_coverage_prefix",
                "cluster_ids": c5_selection.coverage_manifest["cluster_id"].astype(str).tolist(),
                "clusters": c5_selection.coverage_manifest.to_dict(orient="records"),
            },
            "portable_manifest.json": {
                **manifest_metadata,
                "role": "portable_c5_prefix",
                "cluster_ids": c5_ids,
                "clusters": c5_selection.portable_manifest.to_dict(orient="records"),
            },
            "diagnostic_manifest.json": {
                **manifest_metadata,
                "role": "important_unstable_or_unmatched",
                "cluster_ids": c5_selection.diagnostic_manifest["cluster_id"].astype(str).tolist(),
                "clusters": c5_selection.diagnostic_manifest.to_dict(orient="records"),
            },
        }.items():
            path = temporary / name
            path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            outputs[name] = _sha256(path)
        evidence = {
            "development_metrics_sha256": hashlib.sha256(development_metrics.to_csv(index=False).encode("utf-8")).hexdigest(),
            "grouped_mda_sha256": hashlib.sha256(grouped_mda.to_csv(index=False).encode("utf-8")).hexdigest(),
            "c5_prefix_mda_sha256": hashlib.sha256(c5_prefix_mda.to_csv(index=False).encode("utf-8")).hexdigest(),
        }
        manifest = {
            "schema": SCHEMA,
            "status": "STRICT_OOF_C5_C6_TAXONOMY_FINALIZED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_root": str(candidate.root),
            "candidate_manifest_sha256": _sha256(candidate.root / "manifest.json"),
            "transports": list(candidate.manifest["transports"]),
            "config": asdict(config),
            "contract": {
                "C5": "winner selected only after C1--C4 immutable development economics and grouped MDA",
                "C5_coverage": "portable, unstable, and unmatched contribution coverage are reported separately; unstable families cannot satisfy the portable gate",
                "C6": "smallest predeclared C5 prefix within one standard error of the best grouped-MDA prefix",
                "final_oos": "not read or used",
            },
            "evidence": evidence,
            "outputs": outputs,
        }
        (temporary / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_finalized_leaf_reasoning_cluster_artifact(
    finalization_root: str | os.PathLike[str],
) -> LeafReasoningClusterCandidateArtifact:
    """Load a final C5/C6 overlay while retaining the hashed candidate table."""

    root = Path(finalization_root)
    manifest = _json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "STRICT_OOF_C5_C6_TAXONOMY_FINALIZED":
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization root is not complete")
    candidate_value = manifest.get("candidate_root")
    if not isinstance(candidate_value, str):
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization root lacks its candidate root")
    candidate = load_leaf_reasoning_cluster_candidate_artifact(_resolved_artifact_path(candidate_value))
    if _sha256(candidate.root / "manifest.json") != manifest.get("candidate_manifest_sha256"):
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization candidate manifest hash mismatch")
    candidate_transports = candidate.manifest.get("transports", ())
    if sorted(map(str, manifest.get("transports", ()))) != sorted(map(str, candidate_transports)):
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization transport contract differs from its candidate root")
    outputs = manifest.get("outputs")
    required = {
        "threshold_selection_audit.parquet", "c5_coverage_prefix_audit.parquet",
        "c5_contribution_coverage_report.parquet", "c6_prefix_one_se_audit.parquet",
        "cluster_groups.json", "cluster_taxonomy_contract.json", "coverage_manifest.json",
        "portable_manifest.json", "diagnostic_manifest.json",
    }
    if not isinstance(outputs, Mapping) or not required.issubset(outputs):
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization manifest lacks immutable outputs")
    for name in required:
        path = root / name
        if not path.is_file() or _sha256(path) != outputs[name]:
            raise LeafReasoningClusterMaterializerError(f"C5/C6 finalization output hash mismatch: {path}")
    groups = _json(root / "cluster_groups.json")
    taxonomy = _json(root / "cluster_taxonomy_contract.json")
    coverage_manifest = _json(root / "coverage_manifest.json")
    portable_manifest = _json(root / "portable_manifest.json")
    diagnostic_manifest = _json(root / "diagnostic_manifest.json")
    if str(taxonomy.get("selection_phase", "")).lower() != FINAL_PHASE:
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization taxonomy is not final")
    expected_groups = {"C0", *C_ARMS, "C5", "C6"}
    if set(groups) != expected_groups:
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization groups are incomplete")
    for arm in ("C0", *C_ARMS):
        if list(map(str, groups[arm])) != list(candidate.groups[arm]):
            raise LeafReasoningClusterMaterializerError(f"C5/C6 finalization illegally changes upstream {arm}")
    mapping = _json(candidate.root / "cluster_feature_manifest.json").get("cluster_id_to_feature", {})
    ids = taxonomy.get("cluster_ids_by_arm", {})
    if not isinstance(mapping, Mapping) or not isinstance(ids, Mapping):
        raise LeafReasoningClusterMaterializerError("C5/C6 finalization lacks cluster mapping lineage")
    for arm in ("C5", "C6"):
        values = ids.get(arm)
        if not isinstance(values, list) or list(map(str, groups[arm])) != [str(mapping.get(str(value), "")) for value in values]:
            raise LeafReasoningClusterMaterializerError(f"C5/C6 finalization {arm} mapping mismatch")
    expected_c5_ids = list(map(str, ids.get("C5", ())))
    if portable_manifest.get("role") != "portable_c5_prefix" or list(map(str, portable_manifest.get("cluster_ids", ()))) != expected_c5_ids:
        raise LeafReasoningClusterMaterializerError("C5/C6 portable manifest does not match the C5 taxonomy")
    if coverage_manifest.get("role") != "minimal_coverage_prefix":
        raise LeafReasoningClusterMaterializerError("C5/C6 coverage manifest has an invalid role")
    if diagnostic_manifest.get("role") != "important_unstable_or_unmatched":
        raise LeafReasoningClusterMaterializerError("C5/C6 diagnostic manifest has an invalid role")
    report = pd.read_parquet(root / "c5_contribution_coverage_report.parquet")
    expected_columns = {
        "source_arm", "transport", "portable_contribution_coverage",
        "unstable_contribution_coverage", "unmatched_contribution_coverage",
    }
    if not expected_columns.issubset(report.columns) or report.empty:
        raise LeafReasoningClusterMaterializerError("C5/C6 contribution-coverage report is incomplete")
    return LeafReasoningClusterCandidateArtifact(candidate.root, candidate.candidate_features, groups, taxonomy, candidate.manifest)


def merge_strict_cluster_candidate_features(
    ledger: pd.DataFrame,
    cluster_features: pd.DataFrame,
) -> pd.DataFrame:
    """Exact, token-free candidate join used by the meta CLI before C arms."""

    _forbid_raw_leaf(ledger.columns, source="meta ledger cluster join input")
    _forbid_raw_leaf(cluster_features.columns, source="candidate cluster feature input")
    missing_ledger = sorted(set(IDENTITY).difference(ledger.columns))
    if missing_ledger:
        raise LeafReasoningClusterMaterializerError(f"meta ledger cluster join lacks {missing_ledger}")
    missing = sorted(set(IDENTITY).difference(cluster_features.columns))
    if missing:
        raise LeafReasoningClusterMaterializerError(f"candidate cluster feature input lacks {missing}")
    work = cluster_features.copy()
    work["decision_ts"] = _utc(work["decision_ts"], source="candidate cluster decision_ts")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if work.duplicated(list(IDENTITY)).any():
        raise LeafReasoningClusterMaterializerError("candidate cluster feature input duplicates identity")
    fields = [name for name in work if name not in IDENTITY]
    if not fields:
        raise LeafReasoningClusterMaterializerError("candidate cluster feature input has no feature field")
    values = work.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise LeafReasoningClusterMaterializerError("candidate cluster feature input must be finite")
    overlap = sorted(set(fields).intersection(ledger.columns))
    if overlap:
        raise LeafReasoningClusterMaterializerError(f"candidate cluster features would overwrite ledger fields: {overlap}")
    base = ledger.copy()
    base["decision_ts"] = _utc(base["decision_ts"], source="ledger decision_ts")
    base["side_name"] = base["side_name"].astype(str).str.lower()
    merged = base.merge(work, on=list(IDENTITY), how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise LeafReasoningClusterMaterializerError(
            "candidate cluster and ledger identities are not exact "
            f"(missing_cluster={int(merged['_merge'].eq('left_only').sum())}, "
            f"extra_cluster={int(merged['_merge'].eq('right_only').sum())})"
        )
    return merged.drop(columns="_merge")


__all__ = [
    "LeafReasoningClusterCandidateArtifact",
    "LeafReasoningClusterFinalizationConfig",
    "LeafReasoningClusterMaterializerConfig",
    "LeafReasoningClusterMaterializerError",
    "LeafReasoningClusterMaterializationResult",
    "STATUS",
    "finalize_leaf_reasoning_cluster_taxonomy",
    "load_leaf_reasoning_cluster_candidate_artifact",
    "load_finalized_leaf_reasoning_cluster_artifact",
    "materialize_leaf_reasoning_cluster_candidates",
    "merge_strict_cluster_candidate_features",
]
