#!/usr/bin/env python3
"""Evaluate hash-bound semantic supportive OOF predictions, fail closed otherwise."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.strict_oof_semantic_support_audit import (
    PREDICTION_LINEAGE_COLUMNS,
    SCHEMA,
    audit_semantic_support,
    semantic_head_specs,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OOF manifest must be a JSON object")
    return payload


def run(
    *,
    semantic_labels: Path,
    semantic_contract: Path,
    oof_predictions: Path,
    output: Path,
    oof_manifest: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    labels = pd.read_parquet(semantic_labels)
    specs = semantic_head_specs(labels.columns)
    available = set(pq.read_schema(oof_predictions).names)
    prediction_columns = set(PREDICTION_LINEAGE_COLUMNS)
    for spec in specs:
        prediction_columns.update(name for name in spec.prediction_aliases if name in available)
    # Reading a large legacy ledger is otherwise needlessly expensive: only
    # identity/lineage and fields which are actual semantic-head candidates are
    # needed for this audit.
    predictions = pd.read_parquet(oof_predictions, columns=sorted(prediction_columns & available))
    audit = audit_semantic_support(
        labels,
        predictions,
        semantic_contract_sha256=_sha256(semantic_contract),
        oof_manifest=_load_manifest(oof_manifest),
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        audit.readiness.to_parquet(stage / "semantic_support_readiness.parquet", index=False, compression="zstd")
        outputs = {"semantic_support_readiness.parquet": _sha256(stage / "semantic_support_readiness.parquet")}
        if not audit.metrics.empty:
            audit.metrics.to_parquet(stage / "semantic_support_metrics.parquet", index=False, compression="zstd")
            outputs["semantic_support_metrics.parquet"] = _sha256(stage / "semantic_support_metrics.parquet")
        if audit.joined is not None:
            used = ["candidate_id", *[column for column in audit.metrics.target_column.dropna().tolist() if column in audit.joined], *[column for column in audit.metrics.valid_column.dropna().tolist() if column in audit.joined], *[column for column in audit.metrics.prediction_column.dropna().tolist() if column in audit.joined]]
            used = list(dict.fromkeys(used))
            audit.joined.loc[:, used].to_parquet(stage / "semantic_support_joined_oof.parquet", index=False, compression="zstd")
            outputs["semantic_support_joined_oof.parquet"] = _sha256(stage / "semantic_support_joined_oof.parquet")
        inputs = {
            "semantic_labels": {"path": str(semantic_labels), "sha256": _sha256(semantic_labels)},
            "semantic_contract": {"path": str(semantic_contract), "sha256": _sha256(semantic_contract)},
            "oof_predictions": {"path": str(oof_predictions), "sha256": _sha256(oof_predictions)},
            "oof_manifest": ({"path": str(oof_manifest), "sha256": _sha256(oof_manifest)} if oof_manifest else None),
        }
        manifest = {
            "schema": SCHEMA,
            "status": audit.status,
            "promotion_eligible": False,
            "inputs": inputs,
            "required_oof_lineage": list(PREDICTION_LINEAGE_COLUMNS),
            "semantic_contract_binding": "oof manifest semantic_target_contract_sha256 must equal semantic contract SHA-256",
            "head_count": len(specs),
            "outputs_sha256": outputs,
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        semantic_labels=args.semantic_labels,
        semantic_contract=args.semantic_contract,
        oof_predictions=args.oof_predictions,
        oof_manifest=args.oof_manifest,
        output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
