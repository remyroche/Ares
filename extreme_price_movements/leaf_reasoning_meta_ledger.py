"""Assemble a provenance-checked base-to-meta reasoning ledger.

The leaf-reasoning materialisers intentionally emit one narrow table per
semantic head.  The meta learner, in contrast, needs one row per candidate
with the same-side base OOF prediction and only compact, head-qualified G1--G3
summaries.  This module is the deliberately strict boundary between the two.

It never creates a score, a label, a leaf family, or a cost map.  In
particular, it is not allowed to promote an ordinary base prediction merely by
setting a Boolean flag: the source strict-OOF manifest must first prove the
base provenance.  Inner rows are the only possible meta-training population;
outer rows are the only possible outer-meta evaluation population.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .tp6_portability_data import FROZEN_META_CONTEXT


SCHEMA = "leaf_reasoning_meta_ledger_v2"
# Candidate IDs identify a generator record, not a unique row in a
# chronological OOF ledger.  The same opaque ID may legitimately recur across
# sides, folds, transports, or inner/outer partitions.  Every source check and
# join below therefore uses this complete immutable identity.
IDENTITY = (
    "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
)
REASONING_IDENTITY = (
    "candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition",
)
RAW_LEAF_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")
HEADS = frozenset(("p_adverse", "p_weak", "p_clear"))
PARTITIONS = frozenset(("inner_oof", "outer_test"))
# S2's predecessor representation has one deliberately narrow candidate-level
# reasoning concentration input.  It is produced from current token-free
# family contribution mass by the causal health materialiser; this name is
# explicit so a broad ``base_reasoning__`` wildcard can never admit raw leaf
# fields into the ledger.
REASONING_CANDIDATE_FIELDS = ("base_reasoning__family_contribution_entropy",)


class LeafReasoningMetaLedgerError(ValueError):
    """Raised when sources cannot prove the strict base-to-meta hand-off."""


@dataclass(frozen=True)
class MetaLedgerResult:
    ledger: pd.DataFrame
    feature_groups: dict[str, list[str]]
    manifest: dict[str, Any]


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise LeafReasoningMetaLedgerError(f"{column} must contain finite UTC timestamps")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeafReasoningMetaLedgerError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LeafReasoningMetaLedgerError(f"JSON artifact must be an object: {path}")
    return value


def _forbid_raw_leaf(columns: Iterable[object], *, source: str) -> None:
    bad = sorted(
        str(name) for name in columns
        if not str(name).lower().startswith("base_reasoning__g1_leaf_assignment_count")
        and any(token in str(name).lower() for token in RAW_LEAF_TOKENS)
    )
    if bad:
        raise LeafReasoningMetaLedgerError(
            f"{source} leaks raw fold-local leaf identifiers: {bad}"
        )


def _strict_source(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "strict_oof_reasoning_manifest.json")
    if manifest.get("status") != "STRICT_OOF_BASE_REASONING_MATERIALIZED":
        raise LeafReasoningMetaLedgerError("strict reasoning source is not complete")
    if manifest.get("prediction_shards") != "base_prediction_shards/<transport>/<side>/":
        raise LeafReasoningMetaLedgerError("unexpected strict prediction-shard contract")
    transports = manifest.get("transports")
    if not isinstance(transports, list) or not transports:
        raise LeafReasoningMetaLedgerError("strict reasoning manifest has no completed transports")
    return manifest


def _compact_source(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "base_reasoning_representation_manifest.json")
    if manifest.get("status") != "COMPACT_STRICT_OOF_BASE_REASONING_MATERIALIZED":
        raise LeafReasoningMetaLedgerError("compact reasoning source is not complete")
    contract = manifest.get("contract", {})
    if not isinstance(contract, dict) or contract.get("leaf_alignment") is None:
        raise LeafReasoningMetaLedgerError("compact reasoning manifest lacks leaf-alignment provenance")
    return manifest


def _health_source(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only a completed, token-free H1--H5 candidate health hand-off."""

    manifest = _read_json(root / "health_materialization_manifest.json")
    if manifest.get("status") != "CAUSAL_LEAF_HEALTH_MATERIALIZED":
        raise LeafReasoningMetaLedgerError("causal leaf-health source is not complete")
    path = root / "base_leaf_health_features_oof.parquet"
    if not path.is_file():
        raise LeafReasoningMetaLedgerError("causal leaf-health source lacks candidate feature table")
    health = pd.read_parquet(path)
    _forbid_raw_leaf(health.columns, source="causal leaf-health feature table")
    required = {
        "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
    }
    missing = sorted(required.difference(health.columns))
    if missing:
        raise LeafReasoningMetaLedgerError(
            f"causal leaf-health table lacks required candidate lineage: {missing}"
        )
    health = health.copy()
    health["decision_ts"] = _utc(health, "decision_ts")
    health["side_name"] = health["side_name"].astype(str).str.lower()
    if health.duplicated(list(IDENTITY)).any():
        raise LeafReasoningMetaLedgerError("causal leaf-health table duplicates candidate identity")
    fields = [
        name for name in health
        if str(name).startswith("base_health__") or str(name) in REASONING_CANDIDATE_FIELDS
    ]
    if not fields:
        raise LeafReasoningMetaLedgerError("causal leaf-health table has no approved H1--H5/reasoning fields")
    values = health.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise LeafReasoningMetaLedgerError("causal leaf-health inference fields must be finite")
    return health, manifest


def _prediction_rows(root: Path, transport: str, side: str, filename: str, partition: str) -> pd.DataFrame:
    path = root / "base_prediction_shards" / transport / side / filename
    if not path.is_file():
        raise LeafReasoningMetaLedgerError(f"missing {partition} prediction shard: {path}")
    result = pd.read_parquet(path)
    required = {
        "candidate_id", "decision_ts", "label_available_ts", "side_name", "fold_id",
        "gross_bps", "net_bps", "base_expected_bps", "base_fit_cutoff_ts",
        "feature_generation_ts", "p_adverse", "p_weak", "p_clear",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise LeafReasoningMetaLedgerError(f"{partition} prediction shard lacks {missing}")
    result = result.copy()
    result["side_name"] = result["side_name"].astype(str).str.lower()
    if not result["side_name"].eq(side).all():
        raise LeafReasoningMetaLedgerError(f"{partition} prediction shard crosses sides")
    for column in ("decision_ts", "label_available_ts", "base_fit_cutoff_ts", "feature_generation_ts"):
        result[column] = _utc(result, column)
    if not result["base_fit_cutoff_ts"].lt(result["decision_ts"]).all():
        raise LeafReasoningMetaLedgerError("base fit cutoff must strictly precede every candidate")
    if not result["feature_generation_ts"].le(result["decision_ts"]).all():
        raise LeafReasoningMetaLedgerError("base prediction was generated after decision time")
    if not result["label_available_ts"].ge(result["decision_ts"]).all():
        raise LeafReasoningMetaLedgerError("base labels cannot resolve before decision time")
    gross = pd.to_numeric(result["gross_bps"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(result["net_bps"], errors="coerce").to_numpy(float)
    cost = gross - net
    if not np.isfinite(cost).all() or (cost < 0.0).any():
        raise LeafReasoningMetaLedgerError("gross/net outcomes cannot prove a non-negative single cost")
    result["realized_gross_bps"] = gross
    result["realized_cost_bps"] = cost
    result["realized_net_bps"] = net
    result["base_oof_fit_end_ts"] = result["base_fit_cutoff_ts"]
    result["base_oof_generated_ts"] = result["feature_generation_ts"]
    # The literal is safe only because `_strict_source` proved the source.
    result["base_same_side_strict_oof"] = True
    result["transport"] = str(transport)
    result["meta_partition"] = partition
    if result.duplicated(list(IDENTITY)).any():
        raise LeafReasoningMetaLedgerError(f"duplicate {partition} base candidate identity")
    return result


def _load_predictions(strict_root: Path, transports: Iterable[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for transport in sorted(map(str, transports)):
        for side in ("long", "short"):
            inner = _prediction_rows(strict_root, transport, side, "strict_oof_predictions.parquet", "inner_oof")
            outer = _prediction_rows(strict_root, transport, side, "outer_predictions.parquet", "outer_test")
            parts.extend((inner, outer))
    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(list(IDENTITY)).any():
        raise LeafReasoningMetaLedgerError("base prediction sources duplicate a full candidate identity")
    return output


def _head_qualified_reasoning(compact_root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    path = compact_root / "base_reasoning_features_oof.parquet"
    if not path.is_file():
        raise LeafReasoningMetaLedgerError(f"missing compact reasoning features: {path}")
    frame = pd.read_parquet(path)
    _forbid_raw_leaf(frame.columns, source="compact reasoning feature table")
    required = {*REASONING_IDENTITY, "contribution_direction", "meta_partition", "transport"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise LeafReasoningMetaLedgerError(
            "compact reasoning source must contain outer and inner partition lineage: "
            f"{missing}"
        )
    frame = frame.copy()
    frame["__ts__"] = _utc(frame, "__ts__")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if not set(frame["head_name"].astype(str)).issubset(HEADS):
        raise LeafReasoningMetaLedgerError("compact reasoning table has an unknown semantic head")
    if not set(frame["meta_partition"].astype(str)).issubset(PARTITIONS):
        raise LeafReasoningMetaLedgerError("compact reasoning table has an unknown meta partition")
    numerical = [
        name for name in frame.columns
        if name.startswith("base_reasoning__")
    ]
    if not numerical:
        raise LeafReasoningMetaLedgerError("compact reasoning table has no G1/G2/G3 fields")
    if not np.isfinite(frame[numerical].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        raise LeafReasoningMetaLedgerError("compact reasoning fields must be finite")
    keys = ["candidate_id", "__ts__", "side_name", "fold_id", "transport", "meta_partition"]
    if frame.duplicated([*keys, "head_name"]).any():
        raise LeafReasoningMetaLedgerError("compact reasoning duplicates candidate/head rows")
    counts = frame.groupby(keys, observed=True)["head_name"].agg(lambda values: set(map(str, values)))
    if not counts.eq(HEADS).all():
        raise LeafReasoningMetaLedgerError("every compact candidate requires all three semantic heads")
    # Direction is intentionally part of every field name; an adverse
    # contribution must never be pooled with clear/weak or the other sign.
    wide = frame.set_index([*keys, "head_name", "contribution_direction"])[numerical].unstack(["head_name", "contribution_direction"])
    wide.columns = [
        f"{field}__{head}__{direction}"
        for field, head, direction in wide.columns.to_flat_index()
    ]
    wide = wide.reset_index().rename(columns={"__ts__": "decision_ts"})
    source_keys = predictions.loc[:, list(IDENTITY)]
    check = wide.merge(source_keys, on=list(IDENTITY), how="outer", validate="one_to_one", indicator=True)
    if not check["_merge"].eq("both").all():
        raise LeafReasoningMetaLedgerError("compact reasoning and prediction identities are not identical")
    return wide


def _feature_groups(ledger: pd.DataFrame) -> dict[str, list[str]]:
    missing_control = sorted(set(FROZEN_META_CONTEXT).difference(ledger.columns))
    if missing_control:
        raise LeafReasoningMetaLedgerError(
            "base prediction shards do not contain the full frozen current-meta "
            f"contract: {missing_control}"
        )
    groups: dict[str, list[str]] = {
        "L0": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", *FROZEN_META_CONTEXT],
        "L1": sorted(name for name in ledger if "__g1_" in name),
        "L2": sorted(name for name in ledger if "__g2_" in name),
        "L3": sorted(name for name in ledger if "__g3_" in name),
    }
    groups["L4"] = []
    # H0 is L4.  Every H1--H5 group is populated only from the immutable
    # causal health hand-off; absence stays fail-closed rather than creating a
    # proxy from current outcomes or raw leaf identifiers.
    groups["H0"] = []
    for number in range(1, 6):
        groups[f"H{number}"] = sorted(
            name for name in ledger
            if str(name).startswith(f"base_health__h{number}__")
        )
    groups["H6"] = []
    # This is not a free-form base-reasoning bucket.  It is only the explicit
    # causal S2 source field and will be consumed through the nested
    # predecessor OOF contract rather than directly by an L/H arm.
    groups["S2_reasoning_entropy"] = [
        name for name in REASONING_CANDIDATE_FIELDS if name in ledger
    ]
    return groups


def _assemble_source_pair(strict: Path, compact: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Materialise one independently proven strict/compact source pair."""
    strict_manifest = _strict_source(strict)
    compact_manifest = _compact_source(compact)
    transports = strict_manifest["transports"]
    predictions = _load_predictions(strict, transports)
    reasoning = _head_qualified_reasoning(compact, predictions)
    ledger = predictions.merge(
        reasoning,
        on=list(IDENTITY),
        how="inner", validate="one_to_one",
    )
    if len(ledger) != len(predictions):
        raise LeafReasoningMetaLedgerError("base/meta ledger dropped candidate rows")
    return ledger, strict_manifest, compact_manifest


def assemble_leaf_reasoning_meta_ledger_pairs(
    source_pairs: Iterable[tuple[str | os.PathLike[str], str | os.PathLike[str]]],
    *,
    health_root: str | os.PathLike[str] | None = None,
) -> MetaLedgerResult:
    """Join one or more independent strict base-to-meta hand-offs.

    Each pair is validated before rows are combined.  This keeps the two
    development transports isolated at source while allowing the meta funnel
    to see both canonical transport partitions in one immutable ledger.  A
    A transport may occur in exactly one pair.  Candidate IDs are explicitly
    *not* globally unique; only a repeated complete candidate identity would
    double-count an outcome or blend incompatible base/reasoning provenance.
    """
    pairs = [(Path(strict), Path(compact)) for strict, compact in source_pairs]
    if not pairs:
        raise LeafReasoningMetaLedgerError("at least one strict/compact source pair is required")
    source_ledgers: list[pd.DataFrame] = []
    source_manifests: list[tuple[Path, Path, dict[str, Any], dict[str, Any]]] = []
    seen_transports: set[str] = set()
    for strict, compact in pairs:
        ledger, strict_manifest, compact_manifest = _assemble_source_pair(strict, compact)
        transports = set(map(str, strict_manifest["transports"]))
        overlap = sorted(seen_transports.intersection(transports))
        if overlap:
            raise LeafReasoningMetaLedgerError(
                f"strict source pairs reuse transport names: {overlap}"
            )
        seen_transports.update(transports)
        source_ledgers.append(ledger)
        source_manifests.append((strict, compact, strict_manifest, compact_manifest))
    ledger = pd.concat(source_ledgers, ignore_index=True)
    if ledger.duplicated(list(IDENTITY)).any():
        raise LeafReasoningMetaLedgerError("strict source pairs duplicate a full candidate identity")
    health_manifest: dict[str, Any] | None = None
    if health_root is not None:
        health, health_manifest = _health_source(Path(health_root))
        ledger = ledger.merge(health, on=list(IDENTITY), how="outer", validate="one_to_one", indicator=True)
        if not ledger["_merge"].eq("both").all():
            missing_health = int(ledger["_merge"].eq("left_only").sum())
            extra_health = int(ledger["_merge"].eq("right_only").sum())
            raise LeafReasoningMetaLedgerError(
                "causal leaf-health and meta ledger identities are not identical "
                f"(missing_health={missing_health}, extra_health={extra_health})"
            )
        ledger = ledger.drop(columns="_merge")
    _forbid_raw_leaf(ledger.columns, source="meta ledger")
    ledger = ledger.sort_values(["transport", "meta_partition", "decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    groups = _feature_groups(ledger)
    transports = sorted(seen_transports)
    manifest = {
        "schema": SCHEMA,
        "status": "STRICT_BASE_TO_META_LEDGER_ASSEMBLED",
        "strict_sources": [str(strict) for strict, _, _, _ in source_manifests],
        "compact_sources": [str(compact) for _, compact, _, _ in source_manifests],
        "transports": transports,
        "row_counts": ledger.groupby(["transport", "meta_partition"], observed=True).size().rename("rows").reset_index().to_dict("records"),
        "contract": {
            "candidate_identity": list(IDENTITY),
            "base_provenance": "same-side strict OOF verified from completed source manifest",
            "meta_fit": "inner_oof only; labels must resolve before each outer evaluation start",
            "meta_evaluation": "outer_test only",
            "cost": "realized_cost_bps = gross_bps - net_bps; no score or cost is reconstructed here",
            "raw_leaf_ids": "rejected before the meta ledger",
            "head_and_direction": "all G1/G2/G3 fields retain semantic-head and contribution-direction suffixes",
            "candidate_reasoning_entropy": "only the explicit causal abs-contribution entropy field may cross from health into the ledger; raw leaf identifiers remain forbidden",
        },
        "source_hashes": {
            f"strict_manifest_{index:02d}": hashlib.sha256(
                (strict / "strict_oof_reasoning_manifest.json").read_bytes()
            ).hexdigest()
            for index, (strict, _, _, _) in enumerate(source_manifests)
        } | {
            f"compact_manifest_{index:02d}": hashlib.sha256(
                (compact / "base_reasoning_representation_manifest.json").read_bytes()
            ).hexdigest()
            for index, (_, compact, _, _) in enumerate(source_manifests)
        },
        "feature_groups": groups,
        "compact_schemas": [compact_manifest.get("schema") for _, _, _, compact_manifest in source_manifests],
        "health_source": str(health_root) if health_root is not None else None,
        "health_manifest_schema": health_manifest.get("schema") if health_manifest else None,
    }
    return MetaLedgerResult(ledger, groups, manifest)


def assemble_leaf_reasoning_meta_ledger(
    strict_root: str | os.PathLike[str],
    compact_root: str | os.PathLike[str],
    *,
    health_root: str | os.PathLike[str] | None = None,
) -> MetaLedgerResult:
    """Backward-compatible single-source wrapper."""
    return assemble_leaf_reasoning_meta_ledger_pairs(
        ((strict_root, compact_root),), health_root=health_root,
    )


def write_immutable_meta_ledger(result: MetaLedgerResult, output_dir: str | os.PathLike[str]) -> Path:
    """Write an atomic immutable hand-off directory for the staged meta runs."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite meta ledger: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        table = temporary / "base_to_meta_reasoning_ledger.parquet"
        result.ledger.to_parquet(table, index=False, compression="zstd")
        (temporary / "meta_feature_groups.json").write_text(
            json.dumps(result.feature_groups, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload = dict(result.manifest)
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        payload["sha256"] = {"base_to_meta_reasoning_ledger.parquet": hashlib.sha256(table.read_bytes()).hexdigest()}
        (temporary / "meta_ledger_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "LeafReasoningMetaLedgerError", "MetaLedgerResult", "assemble_leaf_reasoning_meta_ledger",
    "assemble_leaf_reasoning_meta_ledger_pairs",
    "write_immutable_meta_ledger",
]
