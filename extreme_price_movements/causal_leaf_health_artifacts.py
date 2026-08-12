"""Read completed strict reasoning roots into the causal H1--H5 materialiser.

This adapter is intentionally narrow.  It proves that every per-head label
and contribution belongs to one completed strict base prediction shard before
passing token-free family contributions to :mod:`causal_leaf_health`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .causal_leaf_health import (
    CausalLeafHealthConfig,
    CausalLeafHealthError,
    CausalLeafHealthResult,
    build_causal_leaf_health_states,
    write_immutable_causal_leaf_health,
)
from .leaf_family_contributions import (
    extract_leaf_family_contributions,
    materialize_leaf_family_contributions,
)


STRICT_ROOT_STATUS = "STRICT_OOF_BASE_REASONING_MATERIALIZED"
HEAD_ARTIFACT_STATUS = "MATERIALIZED_STRICT_OOF"


@dataclass(frozen=True)
class StrictOOFFamilyInputs:
    """Verified token-free inputs collected from completed strict roots.

    The object is intentionally narrow: callers receive completed per-head
    candidate labels and same-artifact family contributions, never a raw leaf
    assignment or local leaf token.  It is shared by H1--H5 state generation
    and predecessor-only family selection so those two stages cannot drift in
    their definition of a strict input root.
    """

    candidates: pd.DataFrame
    contributions: pd.DataFrame
    strict_roots: tuple[str, ...]
    strict_root_manifest_sha256: dict[str, str]


@dataclass(frozen=True)
class StrictOOFFamilyInputSpool:
    """Disk-backed, verified strict inputs for the production H1--H5 pass.

    ``candidate_parts`` and ``contribution_parts`` have a one-to-one artifact
    correspondence.  The latter are produced by the bounded contribution
    materialiser, never by reading a complete leaf-assignment table.  This
    object deliberately exposes paths rather than dataframes: a production
    consumer must retain the same bounded-memory property downstream.
    """

    root: Path
    candidate_parts: tuple[Path, ...]
    contribution_parts: tuple[Path, ...]
    strict_roots: tuple[str, ...]
    strict_root_manifest_sha256: dict[str, str]
    manifest_path: Path


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalLeafHealthError(f"invalid strict health source JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CausalLeafHealthError(f"strict health source JSON must be an object: {path}")
    return value


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise CausalLeafHealthError(f"strict source {column} is not valid UTC")
    return value


def _strict_candidate_shards(root: Path) -> pd.DataFrame:
    manifest = _json(root / "strict_oof_reasoning_manifest.json")
    if manifest.get("status") != STRICT_ROOT_STATUS:
        raise CausalLeafHealthError(f"strict health root is not complete: {root}")
    transports = {str(value) for value in manifest.get("transports", [])}
    if not transports:
        raise CausalLeafHealthError("strict health root has no transports")
    rows: list[pd.DataFrame] = []
    for transport in sorted(transports):
        for side in ("long", "short"):
            directory = root / "base_prediction_shards" / transport / side
            for filename, partition in (
                ("strict_oof_predictions.parquet", "inner_oof"),
                ("outer_predictions.parquet", "outer_test"),
            ):
                path = directory / filename
                if not path.is_file():
                    raise CausalLeafHealthError(f"strict health source is missing prediction shard: {path}")
                frame = pd.read_parquet(path)
                required = {
                    "candidate_id", "decision_ts", "label_available_ts", "side_name", "fold_id",
                    "feature_generation_ts", "feature_contract_sha256", "base_expected_bps", "asset",
                    "r3_class",
                }
                missing = sorted(required.difference(frame.columns))
                if missing:
                    raise CausalLeafHealthError(f"strict prediction shard lacks H1 lineage: {missing}")
                frame = frame.copy()
                frame["decision_ts"] = _utc(frame, "decision_ts")
                frame["label_available_ts"] = _utc(frame, "label_available_ts")
                frame["feature_generation_ts"] = _utc(frame, "feature_generation_ts")
                if not frame["side_name"].astype(str).str.lower().eq(side).all():
                    raise CausalLeafHealthError("strict prediction shards cross their side directory")
                if not frame["feature_generation_ts"].le(frame["decision_ts"]).all():
                    raise CausalLeafHealthError("strict health shard feature time is after decision time")
                if not frame["label_available_ts"].ge(frame["decision_ts"]).all():
                    raise CausalLeafHealthError("strict health shard label resolves before decision time")
                frame["transport"] = transport
                frame["meta_partition"] = partition
                rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    keys = ["candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition"]
    if result.duplicated(keys).any():
        raise CausalLeafHealthError("strict prediction roots duplicate candidate identities")
    return result


def _iter_per_head_candidates(root: Path, shard_rows: pd.DataFrame):
    """Yield one fully validated head candidate frame at a time.

    Keeping this boundary artifact-local is important.  An individual head
    can be materialised and released before the next one is read; the old
    collector appended all sixty-six long contribution frames in one list.
    """
    for manifest_path in sorted(root.rglob("base_reasoning_manifest.json")):
        artifact = manifest_path.parent
        manifest = _json(manifest_path)
        if manifest.get("status") != HEAD_ARTIFACT_STATUS:
            raise CausalLeafHealthError(f"per-head strict artifact is incomplete: {artifact}")
        provenance = manifest.get("provenance", {})
        head = str(manifest.get("head_name", ""))
        side = str(manifest.get("side_name", "")).lower()
        fold = str(manifest.get("fold_id", ""))
        contract = str(provenance.get("feature_contract_sha256", ""))
        class_index = provenance.get("class_index")
        if not head or side not in {"long", "short"} or not fold or not contract or class_index is None:
            raise CausalLeafHealthError(f"per-head strict manifest lacks scope lineage: {artifact}")
        prediction_path = artifact / "base_reasoning_predictions.parquet"
        label_path = artifact / "base_reasoning_labels.parquet"
        if not prediction_path.is_file() or not label_path.is_file():
            raise CausalLeafHealthError(f"per-head strict artifact lacks predictions or labels: {artifact}")
        prediction = pd.read_parquet(prediction_path)
        labels = pd.read_parquet(label_path)
        required_prediction = {"candidate_id", "__ts__", "side_name", "head_name", "fold_id", "base_prediction"}
        required_label = {"candidate_id", "__ts__", "side_name", "head_name", "fold_id", "label__r3_class", "label__net_bps", "label__label_available_ts"}
        if missing := sorted(required_prediction.difference(prediction.columns)):
            raise CausalLeafHealthError(f"per-head prediction table lacks {missing}")
        if missing := sorted(required_label.difference(labels.columns)):
            raise CausalLeafHealthError(f"per-head label table lacks {missing}")
        keys = ["candidate_id", "__ts__", "side_name", "head_name", "fold_id"]
        merged = prediction.merge(labels, on=keys, how="inner", validate="one_to_one")
        if len(merged) != len(prediction) or len(merged) != len(labels):
            raise CausalLeafHealthError("per-head strict predictions and labels do not have identical identities")
        merged["__ts__"] = _utc(merged, "__ts__")
        merged["label__label_available_ts"] = _utc(merged, "label__label_available_ts")
        if not merged["head_name"].astype(str).eq(head).all() or not merged["side_name"].astype(str).str.lower().eq(side).all() or not merged["fold_id"].astype(str).eq(fold).all():
            raise CausalLeafHealthError("per-head strict rows cross manifest scope")
        candidate = merged.merge(
            shard_rows,
            left_on=["candidate_id", "__ts__", "side_name", "fold_id"],
            right_on=["candidate_id", "decision_ts", "side_name", "fold_id"],
            how="left", validate="one_to_one", indicator=True,
            suffixes=("_head", ""),
        )
        if not candidate["_merge"].eq("both").all():
            raise CausalLeafHealthError("per-head artifact cannot prove a matching strict prediction shard")
        if not candidate["feature_contract_sha256"].astype(str).eq(contract).all():
            raise CausalLeafHealthError("per-head feature contract differs from strict prediction shard")
        if not candidate["label__label_available_ts"].astype("int64").eq(candidate["label_available_ts"].astype("int64")).all():
            raise CausalLeafHealthError("per-head label availability differs from strict prediction shard")
        candidate = candidate.assign(
            semantic_label=candidate["label__r3_class"].astype(int).eq(int(class_index)).astype(float),
            head_prediction=pd.to_numeric(candidate["base_prediction"], errors="coerce"),
            net_bps=pd.to_numeric(candidate["label__net_bps"], errors="coerce"),
        )
        candidate = candidate.loc[:, [
            "candidate_id", "decision_ts", "feature_generation_ts", "label_available_ts",
            "side_name", "head_name", "fold_id", "transport", "meta_partition",
            "feature_contract_sha256", "semantic_label", "head_prediction", "net_bps",
            "base_expected_bps", "asset",
        ]].copy()
        yield artifact, candidate


def _per_head_inputs(root: Path, shard_rows: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    candidates: list[pd.DataFrame] = []
    contributions: list[pd.DataFrame] = []
    for artifact, candidate in _iter_per_head_candidates(root, shard_rows):
        family = extract_leaf_family_contributions(artifact)
        if family.empty:
            raise CausalLeafHealthError("strict artifact emitted no non-zero family contributions")
        candidates.append(candidate)
        contributions.append(family)
    if not candidates:
        raise CausalLeafHealthError(f"strict health root has no per-head artifacts: {root}")
    return candidates, contributions


def collect_completed_strict_oof_family_inputs(
    strict_roots: Sequence[str | Path],
) -> StrictOOFFamilyInputs:
    """Collect disjoint completed strict roots for a downstream safe consumer.

    This is deliberately not a generic artifact reader.  It keeps all of the
    strict-root and same-artifact lineage checks in this module, then exposes
    only the candidate/head outcomes and token-free family contribution table.
    Consumers must still impose their own chronological cutoff before using an
    outcome for selection or fitting.
    """

    roots = [Path(item) for item in strict_roots]
    if not roots:
        raise CausalLeafHealthError("at least one completed strict root is required")
    if len({str(item.resolve()) for item in roots}) != len(roots):
        raise CausalLeafHealthError("strict family input roots must be distinct")
    candidate_parts: list[pd.DataFrame] = []
    contribution_parts: list[pd.DataFrame] = []
    manifest_hashes: dict[str, str] = {}
    for root in roots:
        shards = _strict_candidate_shards(root)
        candidates, contributions = _per_head_inputs(root, shards)
        candidate_parts.extend(candidates)
        contribution_parts.extend(contributions)
        manifest_path = root / "strict_oof_reasoning_manifest.json"
        manifest_hashes[str(root)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    candidates = pd.concat(candidate_parts, ignore_index=True)
    contributions = pd.concat(contribution_parts, ignore_index=True)
    keys = [
        "candidate_id", "decision_ts", "side_name", "head_name", "fold_id",
        "transport", "meta_partition",
    ]
    if candidates.duplicated(keys).any():
        raise CausalLeafHealthError("strict family roots overlap candidate/head identities")
    # ``extract_leaf_family_contributions`` proves same-artifact token-to-rule
    # lineage but correctly knows nothing about a root's transport partition.
    # Attach that strict candidate provenance here so predecessor selection can
    # never accidentally mix an inner OOF family row with an outer evaluation
    # partition.  This is the same candidate/head identity later enforced by
    # ``_normalise_contributions`` in the H1--H5 builder.
    contribution_keys = ["candidate_id", "__ts__", "side_name", "head_name", "fold_id"]
    lookup = candidates.loc[:, [
        "candidate_id", "decision_ts", "side_name", "head_name", "fold_id",
        "transport", "meta_partition", "feature_contract_sha256",
    ]].rename(columns={"decision_ts": "__ts__"})
    if lookup.duplicated(contribution_keys).any():
        raise CausalLeafHealthError("strict candidate/head provenance is ambiguous for family contributions")
    contributions = contributions.merge(
        lookup, on=contribution_keys, how="left", validate="many_to_one", indicator=True,
    )
    if not contributions["_merge"].eq("both").all():
        raise CausalLeafHealthError("strict family contribution cannot prove candidate/head provenance")
    contributions = contributions.drop(columns="_merge")
    return StrictOOFFamilyInputs(
        candidates=candidates,
        contributions=contributions,
        strict_roots=tuple(str(root) for root in roots),
        strict_root_manifest_sha256=manifest_hashes,
    )


def spool_completed_strict_oof_family_inputs(
    strict_roots: Sequence[str | Path], output_dir: str | Path,
) -> StrictOOFFamilyInputSpool:
    """Stage verified strict inputs without retaining their long tables in RAM.

    This is the production counterpart of
    :func:`collect_completed_strict_oof_family_inputs`.  It performs the same
    root, head, label and strict-shard reconciliation, but writes a compact
    candidate part and a bounded token-free contribution part per artifact.
    The output is immutable and has an explicit artifact pairing manifest.

    The function intentionally does *not* concatenate parts.  Concatenating
    them was the first multi-gigabyte allocation in the old materialiser.
    """

    roots = [Path(item) for item in strict_roots]
    if not roots:
        raise CausalLeafHealthError("at least one completed strict root is required")
    if len({str(item.resolve()) for item in roots}) != len(roots):
        raise CausalLeafHealthError("strict family input roots must be distinct")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite strict family input spool: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    candidate_dir = temporary / "candidate_parts"
    contribution_dir = temporary / "contribution_parts"
    candidate_dir.mkdir()
    contribution_dir.mkdir()
    manifest_hashes: dict[str, str] = {}
    pair_rows: list[dict[str, Any]] = []
    part = 0
    try:
        for root in roots:
            # A root's base prediction shards are materially smaller than its
            # long family table.  They are retained only for this root while
            # each individual head is validated and spooled.
            shards = _strict_candidate_shards(root)
            for artifact, candidate in _iter_per_head_candidates(root, shards):
                candidate_path = candidate_dir / f"part_{part:04d}.parquet"
                contribution_path = contribution_dir / f"part_{part:04d}.parquet"
                candidate.to_parquet(candidate_path, index=False, compression="zstd")
                result = materialize_leaf_family_contributions(artifact, contribution_path)
                if int(result.contribution_row_count) <= 0:
                    raise CausalLeafHealthError("strict artifact emitted no non-zero family contributions")
                pair_rows.append({
                    "part": int(part),
                    "artifact": str(artifact),
                    "candidate_part": str(candidate_path.name),
                    "contribution_part": str(contribution_path.name),
                    "candidate_rows": int(len(candidate)),
                    "contribution_rows": int(result.contribution_row_count),
                    "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                    "contribution_sha256": hashlib.sha256(contribution_path.read_bytes()).hexdigest(),
                })
                part += 1
            del shards
            manifest_path = root / "strict_oof_reasoning_manifest.json"
            manifest_hashes[str(root)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if not pair_rows:
            raise CausalLeafHealthError("strict health root has no per-head artifacts")

        # This disk-backed identity check is the same strict overlap invariant
        # as the in-memory collector.  DuckDB externalises the grouping rather
        # than making a dataframe copy of every candidate part.
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - production dependency
            raise CausalLeafHealthError("duckdb is required for bounded strict input spooling") from exc
        candidate_glob = str(candidate_dir / "*.parquet").replace("'", "''")
        with duckdb.connect(database=":memory:") as connection:
            duplicated = connection.execute(
                "SELECT count(*) FROM ("
                "SELECT candidate_id, decision_ts, side_name, head_name, fold_id, transport, meta_partition "
                f"FROM read_parquet('{candidate_glob}', union_by_name=true) "
                "GROUP BY ALL HAVING count(*) > 1)"
            ).fetchone()[0]
        if int(duplicated):
            raise CausalLeafHealthError("strict family roots overlap candidate/head identities")

        pair_frame = pd.DataFrame(pair_rows)
        pair_path = temporary / "strict_family_input_parts.parquet"
        pair_frame.to_parquet(pair_path, index=False, compression="zstd")
        payload = {
            "schema": "strict_oof_family_input_spool_v1",
            "status": "STRICT_OOF_FAMILY_INPUT_SPOOL_COMPLETED",
            "strict_roots": [str(root) for root in roots],
            "strict_root_manifest_sha256": manifest_hashes,
            "part_count": int(len(pair_rows)),
            "candidate_rows": int(pair_frame["candidate_rows"].sum()),
            "contribution_rows": int(pair_frame["contribution_rows"].sum()),
            "pair_index": pair_path.name,
            "contract": {
                "candidate_lineage": "each candidate part was reconciled one-to-one to a completed strict prediction shard",
                "contribution_lineage": "each contribution part was materialised from its paired artifact with same-artifact token-to-rule reconciliation",
                "raw_leaf_ids": "not present in candidate or contribution outputs",
                "identity": "candidate_id, decision_ts, side, head, fold, transport, meta_partition",
            },
        }
        manifest_path = temporary / "strict_family_input_spool_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return StrictOOFFamilyInputSpool(
        root=target,
        candidate_parts=tuple(target / "candidate_parts" / f"part_{item:04d}.parquet" for item in range(part)),
        contribution_parts=tuple(target / "contribution_parts" / f"part_{item:04d}.parquet" for item in range(part)),
        strict_roots=tuple(str(root) for root in roots),
        strict_root_manifest_sha256=manifest_hashes,
        manifest_path=target / "strict_family_input_spool_manifest.json",
    )


def build_strict_oof_causal_leaf_health(
    strict_roots: Sequence[str | Path],
    *,
    causal_context: pd.DataFrame,
    context_feature_columns: Sequence[str],
    config: CausalLeafHealthConfig = CausalLeafHealthConfig(),
) -> CausalLeafHealthResult:
    """Collect one or more disjoint strict roots into token-free H1--H5 states.

    This is intentionally a development-only materialiser.  It rejects any
    candidate overlap between roots rather than silently double-counting a
    resolved outcome.  The caller must pass an already generated causal regime
    context timeline whose ``regime_available_utc`` establishes the as-of
    boundary.
    """

    collected = collect_completed_strict_oof_family_inputs(strict_roots)
    candidates = collected.candidates
    contributions = collected.contributions
    result = build_causal_leaf_health_states(
        candidates, contributions, causal_context=causal_context,
        context_feature_columns=context_feature_columns, config=config,
    )
    result.manifest["strict_roots"] = list(collected.strict_roots)
    result.manifest["strict_root_manifest_sha256"] = collected.strict_root_manifest_sha256
    return result


def materialize_strict_oof_causal_leaf_health(
    strict_roots: Sequence[str | Path],
    output_dir: str | Path,
    *,
    causal_context: pd.DataFrame,
    context_feature_columns: Sequence[str],
    config: CausalLeafHealthConfig = CausalLeafHealthConfig(),
) -> Path:
    """Build then atomically write the complete H1--H5 strict health root."""

    result = build_strict_oof_causal_leaf_health(
        strict_roots, causal_context=causal_context,
        context_feature_columns=context_feature_columns, config=config,
    )
    return write_immutable_causal_leaf_health(result, output_dir)


__all__ = [
    "StrictOOFFamilyInputs",
    "collect_completed_strict_oof_family_inputs",
    "build_strict_oof_causal_leaf_health",
    "materialize_strict_oof_causal_leaf_health",
]
