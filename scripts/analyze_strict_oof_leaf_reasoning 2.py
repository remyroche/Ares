#!/usr/bin/env python3
"""Bounded diagnosis of strict-OOF base-reasoning leaf artifacts.

The runner deliberately keeps three representations separate:

* G1 health is calculated from the opaque, fold-local leaf assignments;
* G2 rule paths are joined from the catalog only on the same head/side/fold;
* G3 uses only comparable scalar contribution summaries (balance, entropy and
  top shares).  Fold-local SVD axes are never read for cross-fold analysis.

Raw leaf tokens are retained solely in the fold-local health tables.  They are
dropped before signature recurrence and G2/G3 clustering.  Pairwise work is
bounded by a fixed top-80 pre-filter in every side/head/direction cell, then
uses only the immutable threshold/linkage sweep in ``leaf_reasoning_clusters``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_clusters import (  # noqa: E402
    LINKAGES,
    THRESHOLDS,
    LeafReasoningClusterResult,
    sweep_leaf_reasoning_clusters,
)
from extreme_price_movements.performance_regimes.leaf_reasoning_health import (  # noqa: E402
    LeafReasoningHealthColumns,
    LeafReasoningHealthConfig,
    analyze_leaf_reasoning_health,
)


SCHEMA = "strict_oof_leaf_reasoning_analysis_v1"
IDENTITY = ("candidate_id", "__ts__", "side_name")
TREE_PREFIX = "leaf_assignment__"
G3_SCALARS = (
    "base_reasoning__g3_balance",
    "base_reasoning__g3_contrib_entropy",
    "base_reasoning__g3_top1_abs_share",
    "base_reasoning__g3_top3_abs_share",
)
TOP_RULES_PER_CELL = 80


class StrictOOFLeafReasoningAnalysisError(ValueError):
    """Raised when a strict-OOF reasoning artifact does not meet this audit contract."""


@dataclass(frozen=True)
class StrictOOFLeafReasoningAnalysisResult:
    """Fold-local health and raw-token-free cross-fold analysis outputs."""

    leaf_health: pd.DataFrame
    period_health: pd.DataFrame
    month_health: pd.DataFrame
    rule_instance_summary: pd.DataFrame
    rule_prefilter_audit: pd.DataFrame
    signature_recurrence: pd.DataFrame
    cluster_overview: pd.DataFrame
    cluster_assignments: pd.DataFrame
    cluster_pairwise: pd.DataFrame
    artifact_count: int


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _as_utc(frame: pd.DataFrame, column: str) -> pd.Series:
    value = pd.to_datetime(frame[column], utc=True, errors="raise")
    if value.isna().any():
        raise StrictOOFLeafReasoningAnalysisError(f"{column} contains null timestamps")
    return value


def _read_index(path: Path) -> list[Path]:
    """Read a deliberately small, explicit root/path index format."""

    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("artifact_roots", "artifact_dirs", "roots", "paths", "artifacts"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise StrictOOFLeafReasoningAnalysisError(
                "JSON index must be a string list or contain artifact_roots/artifact_dirs/roots/paths"
            )
        return [Path(item) for item in value]
    if path.suffix.lower() in {".csv", ".parquet"}:
        table = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        column = next((name for name in ("artifact_root", "artifact_dir", "root", "path") if name in table), None)
        if not column:
            raise StrictOOFLeafReasoningAnalysisError("tabular index needs artifact_root/artifact_dir/root/path")
        return [Path(str(item)) for item in table[column].dropna().tolist()]
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_reasoning_artifacts(inputs: Sequence[Path]) -> list[Path]:
    """Resolve artifact roots, directories and simple indexes to manifest directories."""

    discovered: set[Path] = set()
    queue = [Path(item) for item in inputs]
    while queue:
        path = queue.pop(0).expanduser()
        if path.is_file() and path.name == "base_reasoning_manifest.json":
            discovered.add(path.parent.resolve())
        elif path.is_dir():
            discovered.update(item.parent.resolve() for item in path.rglob("base_reasoning_manifest.json"))
        elif path.is_file():
            queue.extend(_read_index(path))
        else:
            raise FileNotFoundError(path)
    if not discovered:
        raise StrictOOFLeafReasoningAnalysisError("no base_reasoning_manifest.json found")
    return sorted(discovered)


def _source_scope(path: Path) -> str:
    """Use the strict-OOF collection as a namespace for fold identities."""

    ancestor = next((item for item in (path, *path.parents) if item.name == "strict_oof_base_reasoning"), path.parent)
    return _sha_text(str(ancestor.resolve()))


def _read_narrow_artifact(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest_path = path / "base_reasoning_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "MATERIALIZED_STRICT_OOF":
        raise StrictOOFLeafReasoningAnalysisError(f"artifact is not strict OOF: {path}")
    required = {
        "leaf_assignments.parquet",
        "leaf_rule_catalog.parquet",
        "base_reasoning_predictions.parquet",
        "base_reasoning_labels.parquet",
        "base_reasoning_features.parquet",
    }
    missing = sorted(name for name in required if not (path / name).exists())
    if missing:
        raise StrictOOFLeafReasoningAnalysisError(f"artifact is missing {missing}: {path}")

    prediction = pd.read_parquet(
        path / "base_reasoning_predictions.parquet",
        columns=[*IDENTITY, "base_model_prediction", "head_name", "class_index", "fold_id"],
    )
    labels = pd.read_parquet(
        path / "base_reasoning_labels.parquet",
        columns=[*IDENTITY, "label__r3_class", "label__net_bps", "head_name", "fold_id"],
    )
    feature_schema = set(pq.ParquetFile(path / "base_reasoning_features.parquet").schema.names)
    if not set(G3_SCALARS).issubset(feature_schema):
        raise StrictOOFLeafReasoningAnalysisError(
            f"artifact lacks comparable G3 scalar fields: {sorted(set(G3_SCALARS).difference(feature_schema))}"
        )
    g3 = pd.read_parquet(path / "base_reasoning_features.parquet", columns=[*IDENTITY, *G3_SCALARS])
    catalog = pd.read_parquet(path / "leaf_rule_catalog.parquet")
    assignment_schema = pq.ParquetFile(path / "leaf_assignments.parquet").schema.names
    tree_columns = [name for name in assignment_schema if name.startswith(TREE_PREFIX)]
    if not tree_columns:
        raise StrictOOFLeafReasoningAnalysisError(f"artifact has no {TREE_PREFIX} columns: {path}")

    for frame, name in ((prediction, "predictions"), (labels, "labels"), (g3, "G3 scalars")):
        frame["__ts__"] = _as_utc(frame, "__ts__")
        if frame.duplicated(list(IDENTITY)).any():
            raise StrictOOFLeafReasoningAnalysisError(f"{name} duplicate candidate identities: {path}")
    if not prediction[["head_name", "fold_id"]].nunique(dropna=False).eq(1).all():
        raise StrictOOFLeafReasoningAnalysisError(f"predictions mix head/fold identities: {path}")
    if not labels[["head_name", "fold_id"]].nunique(dropna=False).eq(1).all():
        raise StrictOOFLeafReasoningAnalysisError(f"labels mix head/fold identities: {path}")
    if not prediction["head_name"].astype(str).equals(labels["head_name"].astype(str)) or not prediction["fold_id"].astype(str).equals(labels["fold_id"].astype(str)):
        raise StrictOOFLeafReasoningAnalysisError(f"prediction/label head or fold mismatch: {path}")
    if not prediction.loc[:, list(IDENTITY)].equals(labels.loc[:, list(IDENTITY)]):
        raise StrictOOFLeafReasoningAnalysisError(f"prediction/label candidate identities differ: {path}")
    if not prediction.loc[:, list(IDENTITY)].equals(g3.loc[:, list(IDENTITY)]):
        raise StrictOOFLeafReasoningAnalysisError(f"prediction/G3 candidate identities differ: {path}")
    return manifest, prediction, labels, g3, catalog, tree_columns


def _weekly_period(timestamp: pd.Series) -> pd.Series:
    value = _as_utc(pd.DataFrame({"__ts__": timestamp}), "__ts__")
    return value.dt.normalize() - pd.to_timedelta(value.dt.dayofweek, unit="D")


def _g3_vector(row: pd.Series) -> str:
    values = {
        "balance": abs(float(row["g3_balance_mean"])),
        "entropy": max(float(row["g3_entropy_mean"]), 0.0),
        "top1": max(float(row["g3_top1_abs_share_mean"]), 0.0),
        "top3": max(float(row["g3_top3_abs_share_mean"]), 0.0),
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _health_score(frame: pd.DataFrame) -> pd.Series:
    fields = [
        "health_row_support_pass",
        "health_activation_pass",
        "health_period_support_pass",
        "health_month_support_pass",
        "health_score_support_pass",
        "health_economic_support_pass",
        "health_discrimination_pass",
        "health_calibration_pass",
        "health_economic_pass",
    ]
    return frame.loc[:, fields].astype(float).mean(axis=1).clip(0.0, 1.0)


def _summarize_artifact(path: Path, *, health_config: LeafReasoningHealthConfig) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    manifest, prediction, labels, g3, catalog, tree_columns = _read_narrow_artifact(path)
    head = str(prediction["head_name"].iloc[0])
    fold = str(prediction["fold_id"].iloc[0])
    side = str(prediction["side_name"].iloc[0]).lower()
    class_index = int(prediction["class_index"].iloc[0])
    if side not in {"long", "short"}:
        raise StrictOOFLeafReasoningAnalysisError(f"invalid side {side!r}: {path}")
    if not catalog["head_name"].astype(str).eq(head).all() or not catalog["side_name"].astype(str).str.lower().eq(side).all() or not catalog["fold_id"].astype(str).eq(fold).all():
        raise StrictOOFLeafReasoningAnalysisError(f"catalog head/side/fold does not match assignment artifact: {path}")
    scope = _source_scope(path)
    try:
        feature_contract_sha256 = str(manifest["provenance"]["feature_contract_sha256"])
    except KeyError as exc:
        raise StrictOOFLeafReasoningAnalysisError(f"artifact lacks a feature contract hash: {path}") from exc
    # These narrow tables are loaded once.  The wide assignment parquet is
    # subsequently read one tree column at a time, which bounds peak memory.
    common = prediction.loc[:, [*IDENTITY, "base_model_prediction"]].copy()
    common["label"] = labels["label__r3_class"].eq(class_index).astype(np.float32)
    common["net_bps"] = pd.to_numeric(labels["label__net_bps"], errors="coerce")
    common["week"] = _weekly_period(common["__ts__"])
    common = common.merge(g3.loc[:, [*IDENTITY, *G3_SCALARS]], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(common) != len(prediction):
        raise StrictOOFLeafReasoningAnalysisError(f"narrow artifact join lost rows: {path}")

    health_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    g3_parts: list[pd.DataFrame] = []
    assignment_path = path / "leaf_assignments.parquet"
    for tree_column in tree_columns:
        assignment = pd.read_parquet(assignment_path, columns=[*IDENTITY, tree_column])
        assignment["__ts__"] = _as_utc(assignment, "__ts__")
        if assignment.duplicated(list(IDENTITY)).any() or not assignment.loc[:, list(IDENTITY)].equals(common.loc[:, list(IDENTITY)]):
            raise StrictOOFLeafReasoningAnalysisError(f"tree assignment identity mismatch for {tree_column}: {path}")
        work = common.copy()
        work["leaf_token"] = assignment[tree_column].to_numpy(copy=False)
        if work["leaf_token"].isna().any():
            raise StrictOOFLeafReasoningAnalysisError(f"tree assignment contains null leaf token: {tree_column}")
        work["head"] = head
        work["fold"] = f"{scope}:{fold}"
        work["activation"] = np.float32(1.0)
        work["prediction"] = pd.to_numeric(work["base_model_prediction"], errors="coerce")
        work["is_strict_oof"] = True
        result = analyze_leaf_reasoning_health(
            work.loc[:, [
                "candidate_id", "__ts__", "side_name", "head", "fold", "leaf_token", "activation",
                "prediction", "label", "net_bps", "week", "is_strict_oof",
            ]],
            columns=LeafReasoningHealthColumns(period="week", strict_oof="is_strict_oof"),
            config=health_config,
        )
        for output in (result.leaf_health, result.period_health, result.month_health):
            output["source_scope"] = scope
            output["artifact_dir"] = str(path)
            output["feature_contract_sha256"] = feature_contract_sha256
            output["tree_assignment_column"] = tree_column
        health_parts.append(result.leaf_health)
        period_parts.append(result.period_health)
        month_parts.append(result.month_health)
        g3_by_leaf = (
            work.groupby("leaf_token", observed=True)[list(G3_SCALARS)]
            .mean()
            .rename(columns={
                G3_SCALARS[0]: "g3_balance_mean",
                G3_SCALARS[1]: "g3_entropy_mean",
                G3_SCALARS[2]: "g3_top1_abs_share_mean",
                G3_SCALARS[3]: "g3_top3_abs_share_mean",
            })
            .reset_index()
        )
        g3_by_leaf["tree_assignment_column"] = tree_column
        g3_by_leaf["tree_candidate_rows"] = int(len(work))
        g3_parts.append(g3_by_leaf)

    leaf_health = pd.concat(health_parts, ignore_index=True)
    g3_summary = pd.concat(g3_parts, ignore_index=True)
    # This is the only raw-token catalog join.  It is keyed by the artifact's
    # exact side/head/fold and thus cannot align local tokens across folds.
    catalog_local = catalog.loc[:, [
        "head_name", "side_name", "fold_id", "leaf_token", "model_slot", "tree_index",
        "rule_signature", "rule_structural_path_json",
    ]].copy()
    catalog_local["fold_id"] = f"{scope}:{fold}"
    leaf_for_catalog = leaf_health.rename(columns={"head": "head_name", "fold": "fold_id"})
    merged = leaf_for_catalog.merge(
        catalog_local,
        on=["head_name", "side_name", "fold_id", "leaf_token"],
        how="left",
        validate="one_to_one",
    ).merge(g3_summary, on=["leaf_token", "tree_assignment_column"], how="left", validate="one_to_one")
    if merged["rule_signature"].isna().any() or merged["g3_balance_mean"].isna().any():
        raise StrictOOFLeafReasoningAnalysisError(f"fold-local catalog/G3 join failed: {path}")
    return health_parts, period_parts, month_parts, merged


def _signature_recurrence(rule_summary: pd.DataFrame) -> pd.DataFrame:
    group = ["feature_contract_sha256", "side_name", "head_name", "contribution_direction"]
    available = rule_summary.groupby(group, observed=True)["fold_id"].nunique().rename("available_fold_count").reset_index()
    fold_signature = (
        rule_summary.groupby([*group, "rule_signature", "fold_id"], observed=True)
        .agg(
            fold_economic_effect=("economic_effect", "mean"),
            fold_health_score=("local_health_score", "mean"),
            fold_activation_rate=("activation_rate", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for keys, cell in fold_signature.groupby([*group, "rule_signature"], observed=True, sort=False):
        *prefix, signature = keys
        effect = cell["fold_economic_effect"].to_numpy(dtype=float)
        signs = np.sign(effect)
        positive = int((signs >= 0.0).sum())
        negative = int((signs < 0.0).sum())
        rows.append({
            **dict(zip([*group, "rule_signature"], [*prefix, signature], strict=True)),
            "recurring_fold_count": int(cell["fold_id"].nunique()),
            "economic_sign_consistency": float(max(positive, negative) / max(len(cell), 1)),
            "local_health_score_mean": float(cell["fold_health_score"].mean()),
            "activation_rate_mean": float(cell["fold_activation_rate"].mean()),
            "economic_effect_mean": float(effect.mean()),
            "economic_effect_standard_deviation": float(effect.std(ddof=0)),
        })
    result = pd.DataFrame(rows).merge(available, on=group, how="left", validate="many_to_one")
    result["fold_recurrence_fraction"] = result["recurring_fold_count"] / result["available_fold_count"].clip(lower=1)
    result["cross_fold_recurrence_portability"] = (
        result["fold_recurrence_fraction"]
        * result["economic_sign_consistency"]
        * result["local_health_score_mean"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return result


def _build_rule_summary(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = merged.copy()
    work["g3_balance_mean"] = pd.to_numeric(work["g3_balance_mean"], errors="coerce")
    work["contribution_direction"] = np.where(work["g3_balance_mean"].ge(0.0), "positive", "negative")
    work["economic_effect"] = pd.to_numeric(work["active_economic_mean"], errors="coerce")
    work["activation_rate"] = (
        pd.to_numeric(work["row_support"], errors="coerce")
        / pd.to_numeric(work["tree_candidate_rows"], errors="coerce")
    ).clip(0.0, 1.0)
    work["activation_support"] = pd.to_numeric(work["row_support"], errors="coerce")
    monthly = work[["head_name", "side_name", "fold_id", "leaf_token", "month_support", "active_month_support"]].copy()
    work["activation_stability"] = (
        monthly["active_month_support"] / monthly["month_support"].clip(lower=1)
    ).clip(0.0, 1.0)
    work["local_health_score"] = _health_score(work)
    work["contribution_signature"] = work.apply(
        lambda row: "g3_scalar:"
        f"b{round(float(row['g3_balance_mean']), 2):+.2f}:"
        f"e{round(float(row['g3_entropy_mean']), 2):.2f}:"
        f"t1{round(float(row['g3_top1_abs_share_mean']), 2):.2f}:"
        f"t3{round(float(row['g3_top3_abs_share_mean']), 2):.2f}",
        axis=1,
    )
    work["contribution_top_features_json"] = work.apply(_g3_vector, axis=1)
    work["rule_instance_id"] = work.apply(
        lambda row: _sha_text(
            "|".join(map(str, [
                row["source_scope"], row["head_name"], row["side_name"], row["fold_id"],
                row["model_slot"], row["tree_index"], row["rule_signature"],
            ]))
        ),
        axis=1,
    )
    if work["rule_instance_id"].duplicated().any():
        raise StrictOOFLeafReasoningAnalysisError("raw-token-free rule instance IDs are unexpectedly non-unique")
    recurrence = _signature_recurrence(work)
    keys = ["feature_contract_sha256", "side_name", "head_name", "contribution_direction", "rule_signature"]
    work = work.merge(recurrence, on=keys, how="left", validate="many_to_one")
    work["portability_score"] = work["cross_fold_recurrence_portability"].clip(0.0, 1.0)
    work["portability_coverage"] = work["fold_recurrence_fraction"].clip(0.0, 1.0)
    work["portability_drift"] = (
        work["economic_effect_standard_deviation"].abs()
        / (work["economic_effect_mean"].abs() + 1e-6)
    )
    # Selection is deliberately fixed and outcome/health based only on the
    # already-issued OOF rows.  It is not a model or policy selection rule.
    work["pre_filter_score"] = (
        work["economic_effect"].abs()
        * np.log1p(work["activation_support"].clip(lower=0.0))
        * (0.25 + 0.75 * work["local_health_score"])
    )
    group = ["feature_contract_sha256", "side_name", "head_name", "contribution_direction"]
    work = work.sort_values([*group, "pre_filter_score", "fold_id", "rule_instance_id"], ascending=[True, True, True, True, False, True, True], kind="stable")
    work["pre_filter_rank"] = work.groupby(group, observed=True).cumcount() + 1
    work["selected_for_clustering"] = work["pre_filter_rank"].le(TOP_RULES_PER_CELL)
    # Leaf token stays in the fold-local health output only.  The cross-fold
    # summary and every following operation are structurally token-free.
    cross_fold = work.drop(columns=[
        "leaf_token", "head", "fold", "artifact_dir", "tree_assignment_column", "source_scope",
    ], errors="ignore")
    forbidden = [name for name in cross_fold if "leaf_token" in name or "leaf_assignment" in name or "raw_leaf" in name]
    if forbidden:
        raise StrictOOFLeafReasoningAnalysisError(f"raw leaf token leaked into cross-fold summary: {forbidden}")
    return cross_fold, recurrence


def _cluster_selected_rules(rule_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = rule_summary.loc[rule_summary["selected_for_clustering"]].copy()
    overview: list[pd.DataFrame] = []
    assignments: list[pd.DataFrame] = []
    pairwise: list[pd.DataFrame] = []
    group = ["feature_contract_sha256", "side_name", "head_name", "contribution_direction"]
    for keys, cell in selected.groupby(group, observed=True, sort=True):
        contract, side, head, direction = map(str, keys)
        # The clustering module needs only the documented G2/G3 summary
        # contract.  Removing the feature-contract namespace here is safe
        # because every call contains exactly one contract-compatible cell.
        cluster_input = cell.drop(columns=["feature_contract_sha256"], errors="ignore")
        for (threshold, linkage), result in sweep_leaf_reasoning_clusters(cluster_input).items():
            details = {"feature_contract_sha256": contract, "threshold": threshold, "linkage": linkage}
            summary = result.cluster_summary.assign(**details)
            assignment = result.assignments.assign(**details)
            pairs = result.pairwise_similarity.assign(**details)
            overview.append(summary)
            assignments.append(assignment)
            pairwise.append(pairs)
    return (
        pd.concat(overview, ignore_index=True) if overview else pd.DataFrame(),
        pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame(),
        pd.concat(pairwise, ignore_index=True) if pairwise else pd.DataFrame(),
    )


def analyze_strict_oof_leaf_reasoning(
    inputs: Sequence[Path],
    *,
    health_config: LeafReasoningHealthConfig = LeafReasoningHealthConfig(),
) -> StrictOOFLeafReasoningAnalysisResult:
    """Analyze strict-OOF reasoning roots without fitting or replaying anything."""

    artifacts = discover_reasoning_artifacts(inputs)
    leaf_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    catalog_joined: list[pd.DataFrame] = []
    for artifact in artifacts:
        leaf, period, month, merged = _summarize_artifact(artifact, health_config=health_config)
        leaf_parts.extend(leaf)
        period_parts.extend(period)
        month_parts.extend(month)
        catalog_joined.append(merged)
    leaf_health = pd.concat(leaf_parts, ignore_index=True)
    period_health = pd.concat(period_parts, ignore_index=True)
    month_health = pd.concat(month_parts, ignore_index=True)
    rule_summary, recurrence = _build_rule_summary(pd.concat(catalog_joined, ignore_index=True))
    overview, assignments, pairwise = _cluster_selected_rules(rule_summary)
    return StrictOOFLeafReasoningAnalysisResult(
        leaf_health=leaf_health,
        period_health=period_health,
        month_health=month_health,
        rule_instance_summary=rule_summary.loc[rule_summary["selected_for_clustering"]].reset_index(drop=True),
        rule_prefilter_audit=rule_summary.reset_index(drop=True),
        signature_recurrence=recurrence.reset_index(drop=True),
        cluster_overview=overview,
        cluster_assignments=assignments,
        cluster_pairwise=pairwise,
        artifact_count=len(artifacts),
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def write_analysis(result: StrictOOFLeafReasoningAnalysisResult, output_dir: Path) -> Path:
    """Write a new, immutable analysis directory."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tables = {
            "leaf_health.parquet": result.leaf_health,
            "leaf_period_health.parquet": result.period_health,
            "leaf_month_health.parquet": result.month_health,
            "rule_instance_summary.parquet": result.rule_instance_summary,
            "rule_prefilter_audit.parquet": result.rule_prefilter_audit,
            "signature_recurrence_portability.parquet": result.signature_recurrence,
            "cluster_sweep_overview.parquet": result.cluster_overview,
            "cluster_sweep_assignments.parquet": result.cluster_assignments,
            "cluster_sweep_pairwise.parquet": result.cluster_pairwise,
        }
        for name, table in tables.items():
            table.to_parquet(temporary / name, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "artifact_count": result.artifact_count,
            "contracts": {
                "assignment_read": "one leaf-assignment tree column at a time",
                "label": "binary stored r3_class == artifact semantic class_index",
                "period": "UTC Monday-start weekly period",
                "cross_fold": "G2 catalog signature joined only within fold; raw leaf tokens dropped before recurrence/clustering",
                "g3": "balance, contribution entropy, top-1/top-3 absolute shares only; no fold-local SVD axes",
                "prefilter": f"top {TOP_RULES_PER_CELL} rule instances per feature-contract/side/head/contribution-direction",
                "cluster_sweep": {"thresholds": list(THRESHOLDS), "linkages": list(LINKAGES)},
            },
            "rows": {name: int(len(table)) for name, table in tables.items()},
        }
        (temporary / "manifest.json").write_text(json.dumps(_safe_json(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(inputs: Sequence[Path], output_dir: Path) -> Path:
    return write_analysis(analyze_strict_oof_leaf_reasoning(inputs), output_dir)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="reasoning artifact root, manifest directory, or simple root index")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _args()
    print(run(args.input, args.output_dir))
