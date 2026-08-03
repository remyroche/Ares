#!/usr/bin/env python3
"""Fail closed when older rows cannot join the pooled transition classifier.

The 2022--2023 exact label ledger is valuable, but it must not be called a
pooled transition-classifier training source until it has both (a) the same
decision-time context through the requested period and (b) the same causal
global-book before/after target construction as the current panel.  This
script materializes that proof rather than silently imputing either contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLDER_CONTEXT = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_sidecar_20260730_v1"
DEFAULT_OLDER_LABELS = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1"
DEFAULT_CURRENT_PANEL = ROOT / "data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/pooled_transition_classification_readiness_20260730_v2"
SCHEMA = "pooled_transition_classification_readiness_v2"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verified_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError(f"manifest checksum fails: {root}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def audit_readiness(older_context: pd.DataFrame, older_labels: pd.DataFrame, current_feature_columns: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return coverage plus missing derivations without fabricating a row."""
    missing_identity = sorted(set(IDENTITY).difference(older_context.columns) | set(IDENTITY).difference(older_labels.columns))
    if missing_identity:
        raise ValueError(f"older artifacts lack identities: {missing_identity}")
    if older_context.duplicated(list(IDENTITY)).any() or older_labels.duplicated(list(IDENTITY)).any():
        raise ValueError("older artifact identity is duplicated")
    context = older_context.copy()
    labels = older_labels.copy()
    context["__decision_ts__"] = pd.to_datetime(context["__decision_ts__"], utc=True, errors="raise")
    labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True, errors="raise")
    # The immutable label writer is permitted to choose a different physical
    # parquet order.  Identity equality (not positional equality) is the
    # contract; order is restored only for this audit comparison.
    context_keys = context.loc[:, [*IDENTITY, "__decision_ts__"]].sort_values(list(IDENTITY)).reset_index(drop=True)
    label_keys = labels.loc[:, [*IDENTITY, "__decision_ts__"]].sort_values(list(IDENTITY)).reset_index(drop=True)
    if not context_keys.loc[:, list(IDENTITY)].equals(label_keys.loc[:, list(IDENTITY)]):
        raise ValueError("older context/label identity sets do not align")
    if not context_keys["__decision_ts__"].equals(label_keys["__decision_ts__"]):
        raise ValueError("older context/label decision times do not align")
    context["year"] = context["__decision_ts__"].dt.year
    context["month"] = context["__decision_ts__"].dt.month
    coverage = context.groupby(["year", "month", "side_name"], sort=True).agg(
        candidate_rows=("transition_context_available", "size"),
        transition_context_rows=("transition_context_available", "sum"),
    ).reset_index()
    coverage["transition_context_coverage"] = coverage["transition_context_rows"] / coverage["candidate_rows"]
    old_features = [column for column in older_context.columns if column not in {*IDENTITY, "__decision_ts__", "source_family", "transition_context_available"}]
    common_features = sorted(set(old_features).intersection(current_feature_columns))
    available_until = context.loc[context["transition_context_available"].astype(bool), "__decision_ts__"].max()
    in_2023 = context["__decision_ts__"].dt.year.eq(2023)
    rows_2023 = int(in_2023.sum())
    covered_2023 = int(context.loc[in_2023, "transition_context_available"].astype(bool).sum())
    coverage_2023 = (covered_2023 / rows_2023) if rows_2023 else 0.0
    required_global_book = {
        "causal_global_mapped_ev_percentile", "causal_global_mapped_ev_margin_to_p90",
        "map_reference_rows", "mapped_direct_net",
    }
    missing_map = sorted(required_global_book.difference(older_labels.columns))
    required_horizon = {"before_window_start_utc", "before_window_end_utc", "after_window_start_utc", "after_window_end_utc", "before_target_available_utc", "after_target_available_utc"}
    missing_horizon = sorted(required_horizon.difference(older_labels.columns))
    requirements = [
        {
            "id": "historical_transition_context_through_2023",
            "ready": bool(rows_2023 > 0 and covered_2023 == rows_2023),
            "required": "exact decision-time frozen/current-contract transition fields at every 2023 candidate decision hour",
            "observed": f"latest exact transition context is {available_until}; 2023 exact rows covered={covered_2023}/{rows_2023} ({coverage_2023:.4%})",
            "derivation": "extend/reconstruct the frozen transition source from raw pre-anchor market inputs through 2023, retaining one-hour source-to-decision alignment and explicit raw-coverage exclusions; do not forward-fill 2022 values",
        },
        {
            "id": "common_decision_time_feature_contract",
            "ready": bool(common_features),
            "required": "a versioned common feature mapping with per-field semantic/time parity",
            "observed": f"older sidecar has {len(old_features)} fields and the current v4 panel has {len(current_feature_columns)}; exact name intersection={len(common_features)}",
            "derivation": "materialize a declared common observable geometry from raw inputs for both eras, or define a semantic mapping and prove each mapped field is decision-time available; no column-name guessing",
        },
        {
            "id": "causal_global_book_selection",
            "ready": not missing_map,
            "required": "frozen/raw score, causal 21-day map, reference support and one pooled global top-10 membership before each anchor",
            "observed": f"missing label fields: {missing_map}",
            "derivation": "recover or recompute the historical score ledger and causal map using only labels resolved before every snapshot, then emit candidate membership and mapping provenance",
        },
        {
            "id": "exact_before_after_transition_targets",
            "ready": not missing_horizon,
            "required": "global-book before [s-H,s) and after [s,s+H) aggregates with declared target availability for H=3h/12h",
            "observed": f"missing label fields: {missing_horizon}",
            "derivation": "aggregate only the causally selected historical book into immutable before/after windows, carry every component availability time, then derive active/onset/recovery/reversal targets from those labels",
        },
    ]
    readiness = [
        {
            "requirement": item["id"],
            "ready": bool(item["ready"]),
            "reason": item["observed"],
        }
        for item in requirements
    ]
    missing = [item for item in requirements if not item["ready"]]
    return coverage, readiness, missing


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_root, labels_root, current_root = Path(args.older_context), Path(args.older_labels), Path(args.current_panel)
    context_manifest, context_manifest_path = _verified_manifest(context_root)
    labels_manifest, labels_manifest_path = _verified_manifest(labels_root)
    current_manifest, current_manifest_path = _verified_manifest(current_root)
    context_path = context_root / "context.parquet"
    labels_path = labels_root / "joined_multitask_labels.parquet"
    if context_manifest.get("output", {}).get("sha256") != sha256(context_path):
        raise ValueError("older context parquet checksum fails")
    if labels_manifest.get("outputs", {}).get("joined_multitask_labels", {}).get("sha256") != sha256(labels_path):
        raise ValueError("older label parquet checksum fails")
    if current_manifest.get("outputs", {}).get("panel", {}).get("sha256") != sha256(current_root / "transition_research_panel.parquet"):
        raise ValueError("current transition panel checksum fails")
    coverage, readiness, missing = audit_readiness(
        pd.read_parquet(context_path), pd.read_parquet(labels_path), list(current_manifest["feature_columns"])
    )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    coverage_path, readiness_path, missing_path = temporary / "coverage_by_month_side.csv", temporary / "readiness.csv", temporary / "missing_derivations.json"
    coverage.to_csv(coverage_path, index=False)
    pd.DataFrame(readiness).to_csv(readiness_path, index=False)
    _write_json(missing_path, {"missing_derivations": missing})
    all_requirements_ready = all(bool(item["ready"]) for item in readiness)
    missing_ids = [item["requirement"] for item in readiness if not item["ready"]]
    manifest = {
        "schema": SCHEMA,
        "status": (
            "READY_FOR_POOLED_TRANSITION_CLASSIFICATION"
            if all_requirements_ready
            else "INCOMPLETE_POOLED_TRANSITION_CLASSIFICATION_REQUIREMENTS"
        ),
        "contracts": {"no_imputation": "this is an audit only: no feature, causal-map, candidate membership, or target value is manufactured", "future_safety": "a valid future pooled classifier must retain exact pre-anchor inputs and exact maximum target availability across the declared before/after horizon dependencies", "promotion": "not a model or promotion input"},
        "coverage_rows": int(len(coverage)),
        "all_requirements_ready": all_requirements_ready,
        "missing_requirement_ids": missing_ids,
        "sources": {"older_context": {"path": str(context_path), "sha256": sha256(context_path), "manifest_sha256": sha256(context_manifest_path)}, "older_labels": {"path": str(labels_path), "sha256": sha256(labels_path), "manifest_sha256": sha256(labels_manifest_path)}, "current_panel_manifest_sha256": sha256(current_manifest_path)},
        "outputs_sha256": {"coverage_by_month_side.csv": sha256(coverage_path), "readiness.csv": sha256(readiness_path), "missing_derivations.json": sha256(missing_path)},
    }
    _write_json(temporary / "manifest.json", manifest)
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n")
    os.replace(temporary, output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-context", type=Path, default=DEFAULT_OLDER_CONTEXT)
    parser.add_argument("--older-labels", type=Path, default=DEFAULT_OLDER_LABELS)
    parser.add_argument("--current-panel", type=Path, default=DEFAULT_CURRENT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2, sort_keys=True))
