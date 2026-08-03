#!/usr/bin/env python3
"""Fail-closed historical CatBoost taxonomy/balance/economic merge gate.

This audit reads the sealed exact-1m label shards and compact PIT context
identity index.  Execution EV is used *only* to judge whether the predeclared
fast-class merge is economically coherent; it is never written into a model
matrix or exposed as a candidate feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_support import (  # noqa: E402
    FAST_REALIZATION_WINNER,
    LEGACY_FAST_CLASSES,
    PathArchetypeSupportConfig,
    merge_fast_realization_winner,
    validate_path_archetype_support,
)

SCHEMA = "febapr2025_historical_catboost_merge_gate_v1"
IDENTITY = ["candidate_id", "side_name", "__symbol__", "__ts__"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_labels(labels_root: Path) -> pd.DataFrame:
    index = json.loads((labels_root / "index.json").read_text(encoding="utf-8"))
    if index.get("schema") != "febapr2025_exact1m_path_head_shard_index_v1":
        raise ValueError("label index has an unsupported schema")
    if not index.get("coverage", {}).get("complete"):
        raise ValueError("exact 1m label coverage is incomplete")
    columns = [*IDENTITY, "path_shape_archetype", "path_arch_complete_12h"]
    frames: list[pd.DataFrame] = []
    for shard in index["shards"]:
        path = Path(shard["labels"])
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"label shard hash mismatch: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != int(shard["rows"]):
            raise ValueError(f"label shard row count mismatch: {path}")
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    if len(labels) != int(index["coverage"]["expected_rows"]):
        raise ValueError("label shard rows do not equal sealed index coverage")
    if labels.duplicated(IDENTITY, keep=False).any():
        raise ValueError("label shards contain duplicate exact identities")
    if not labels["path_arch_complete_12h"].astype(bool).all():
        raise ValueError("incomplete 12h archetype labels are in the candidate cohort")
    return labels


def _identity_hash(frame: pd.DataFrame, *, feature_store_symbol: bool = False) -> str:
    key = frame.loc[:, IDENTITY].copy().sort_values(["__ts__", "candidate_id"], kind="stable")
    key["__ts__"] = pd.to_datetime(key["__ts__"], utc=True).astype(str)
    if feature_store_symbol:
        # Historical feature shards retain the immutable store spelling
        # (BTC_USD:USD), while exact execution labels retain the candidate-ID
        # spelling (BTC/USD:USD).  The candidate ID, UTC signal time, and side
        # remain the identity authority; this is a reversible display mapping.
        key["__symbol__"] = key["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    return hashlib.sha256(pd.util.hash_pandas_object(key, index=False).to_numpy().tobytes()).hexdigest()


def _economic_merge_gate(frame: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    source = frame.loc[frame["path_shape_archetype"].isin(LEGACY_FAST_CLASSES)].copy()
    report: dict[str, Any] = {
        "contract": "only_fast_clean_and_fast_early_drawdown_may_merge_v1",
        "source_classes": list(LEGACY_FAST_CLASSES),
        "target_class": FAST_REALIZATION_WINNER,
        "outcome_use": "audit only; execution_net_ev_12h is forbidden from model inputs",
        "sides": {},
    }
    passed = True
    for side, group in source.groupby("side_name", sort=True):
        details: dict[str, Any] = {}
        for label in LEGACY_FAST_CLASSES:
            values = group.loc[group["path_shape_archetype"].eq(label), "execution_net_ev_12h"]
            details[label] = {"rows": int(len(values)), "mean_execution_net_ev_12h": float(values.mean()), "median_execution_net_ev_12h": float(values.median())}
            # The merge is allowed only when both source classes have material
            # support and the same positive policy-EV sign in that side.
            if len(values) < 100 or not float(values.mean()) > 0.0:
                passed = False
        merged = group["execution_net_ev_12h"]
        details[FAST_REALIZATION_WINNER] = {"rows": int(len(merged)), "mean_execution_net_ev_12h": float(merged.mean()), "median_execution_net_ev_12h": float(merged.median())}
        report["sides"][str(side)] = details
    return report, passed and set(report["sides"]) == {"long", "short"}


def run(*, labels_root: Path, context_index: Path, population: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_labels(labels_root)
    context = pd.read_parquet(context_index, columns=IDENTITY)
    if _identity_hash(labels, feature_store_symbol=True) != _identity_hash(context):
        raise ValueError("sealed label identities do not exactly equal PIT context identities")
    population_frame = pd.read_parquet(population, columns=["candidate_id", "execution_net_ev_12h"])
    frame = labels.merge(population_frame, on="candidate_id", how="left", validate="one_to_one")
    if frame["execution_net_ev_12h"].isna().any():
        raise ValueError("economic audit cannot resolve policy EV for every label")
    frame["path_geometry_label"] = merge_fast_realization_winner(frame["path_shape_archetype"])
    side_reports: dict[str, Any] = {}
    support_passed = True
    for side in ("long", "short"):
        cohort = frame.loc[frame["side_name"].eq(side)].copy()
        result = validate_path_archetype_support(
            cohort,
            PathArchetypeSupportConfig(
                label_column="path_geometry_label", timestamp_column="__ts__", side_column="side_name",
                min_global_class_share=0.01, min_month_side_class_share=0.005,
            ),
        )
        for name, table in (("global", result.global_support), ("month", result.month_support), ("side", result.side_support), ("month_side", result.month_side_support), ("violations", result.violations), ("exemptions", result.exemptions)):
            table.to_csv(output_dir / f"{side}_{name}_support.csv", index=False)
        side_reports[side] = {
            "accepted": result.accepted, "recommended_action": result.recommended_action,
            "rows": result.rows, "violations": result.violations.to_dict(orient="records"),
        }
        support_passed &= result.accepted
    economic, economic_passed = _economic_merge_gate(frame)
    _write_json(output_dir / "economic_fast_merge_evidence.json", economic)
    passed = bool(support_passed and economic_passed)
    gate = {
        "schema": SCHEMA,
        "status": "PASS_READY_FOR_SIDE_LOCAL_FS_HPO_GEOMETRY" if passed else "BLOCKED_CLASS_SUPPORT_OR_ECONOMIC_MERGE",
        "labels": {"root": str(labels_root), "index_sha256": _sha256(labels_root / "index.json"), "rows": int(len(labels))},
        "pit_context": {"index": str(context_index), "sha256": _sha256(context_index), "identity_exact_match": True},
        "population_economic_audit": {"path": str(population), "sha256": _sha256(population), "outcome_columns_forbidden_from_inputs": ["execution_net_ev_12h"]},
        "taxonomy": {"raw_label": "path_shape_archetype", "merged_label": "path_geometry_label", "approved_merge": {"source": list(LEGACY_FAST_CLASSES), "target": FAST_REALIZATION_WINNER}},
        "side_local_support": side_reports,
        "economic_merge": economic,
        "passed": passed,
        "next_action": "run per-side FS/HPO/geometry then strict Mar-Apr OOF" if passed else "do not train; approve a new economically coherent taxonomy/geometry before FS/HPO",
        "12h_vs_frozen_v6_24h": "This is a 12h execution-compatible derivative with frozen v6 rule precedence; it is not bitwise comparable to 24h v6 labels.",
        "models_trained": False,
    }
    _write_json(output_dir / "gate.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--context-index", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(labels_root=args.labels_root, context_index=args.context_index, population=args.population, output_dir=args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
