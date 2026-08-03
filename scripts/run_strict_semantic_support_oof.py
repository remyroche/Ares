#!/usr/bin/env python3
"""Materialise strict OOF predictions for every semantic supportive head."""

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

from extreme_price_movements.strict_semantic_support_oof import (
    SCHEMA,
    StrictSemanticOOFError,
    generate_strict_semantic_oof,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        value = payload.get("raw_feature_columns") or payload.get("feature_columns")
    else:
        value = payload
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise StrictSemanticOOFError("features JSON must be a list or contain raw_feature_columns/feature_columns")
    return value


def _join_labels(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "decision_ts", "label_available_ts", "symbol", "side"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise StrictSemanticOOFError(f"semantic labels are missing identity columns: {missing}")
    if labels.candidate_id.isna().any() or labels.candidate_id.astype(str).duplicated().any():
        raise StrictSemanticOOFError("semantic labels must be one-to-one by candidate_id")
    if frame.candidate_id.isna().any() or frame.candidate_id.astype(str).duplicated().any():
        raise StrictSemanticOOFError("feature frame must be one-to-one by candidate_id")
    label_identity = labels.loc[:, ["candidate_id", "decision_ts", "label_available_ts"]].copy()
    label_identity["decision_ts"] = pd.to_datetime(label_identity["decision_ts"], utc=True, errors="raise")
    label_identity["label_available_ts"] = pd.to_datetime(label_identity["label_available_ts"], utc=True, errors="raise")
    frame_decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame_available = pd.to_datetime(frame["__label_available_at__"], utc=True, errors="raise")
    identity = frame.loc[:, ["candidate_id"]].copy()
    identity["frame_decision_ts"] = frame_decision.to_numpy()
    identity["frame_label_available_ts"] = frame_available.to_numpy()
    joined_identity = identity.merge(label_identity, on="candidate_id", how="left", validate="one_to_one", indicator=True)
    if not joined_identity["_merge"].eq("both").all():
        raise StrictSemanticOOFError("not every feature candidate has a semantic-label row")
    if not joined_identity.frame_decision_ts.eq(joined_identity.decision_ts).all():
        raise StrictSemanticOOFError("semantic label decision timestamps do not match feature frame")
    if not joined_identity.frame_label_available_ts.eq(joined_identity.label_available_ts).all():
        raise StrictSemanticOOFError("semantic label availability timestamps do not match feature frame")
    label_columns = [column for column in labels.columns if column not in {"decision_ts", "label_end_ts", "label_available_ts", "symbol", "side"}]
    return frame.merge(labels.loc[:, label_columns], on="candidate_id", how="left", validate="one_to_one")


def run(
    *,
    ledger: Path,
    semantic_labels: Path,
    semantic_contract: Path,
    features_json: Path,
    fold_column: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    frame = pd.read_parquet(ledger)
    labels = pd.read_parquet(semantic_labels)
    payload = json.loads(features_json.read_text(encoding="utf-8"))
    merged = _join_labels(frame, labels)
    result = generate_strict_semantic_oof(
        merged,
        feature_columns=_features(payload),
        fold_column=fold_column,
        semantic_contract_sha256=_sha256(semantic_contract),
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        result.predictions.to_parquet(stage / "semantic_head_oof_predictions.parquet", index=False, compression="zstd")
        fold_manifest = (
            merged.groupby([fold_column, "fold_order"], observed=True)
            .agg(
                rows=("candidate_id", "size"),
                min_feature_ts=("__ts__", "min"),
                max_feature_ts=("__ts__", "max"),
                min_label_available=("__label_available_at__", "min"),
                max_label_available=("__label_available_at__", "max"),
            )
            .reset_index()
        )
        fold_manifest.to_parquet(stage / "fold_manifest.parquet", index=False, compression="zstd")
        manifest = {
            **result.manifest,
            "schema": SCHEMA,
            "ledger": {"path": str(ledger), "sha256": _sha256(ledger)},
            "semantic_labels": {"path": str(semantic_labels), "sha256": _sha256(semantic_labels)},
            "semantic_contract": {"path": str(semantic_contract), "sha256": _sha256(semantic_contract)},
            "features_json": {"path": str(features_json), "sha256": _sha256(features_json)},
            "fold_column": fold_column,
            "outputs": {
                "predictions": "semantic_head_oof_predictions.parquet",
                "fold_manifest": "fold_manifest.parquet",
            },
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        }
        manifest["outputs_sha256"] = {
            name: _sha256(stage / name)
            for name in ("semantic_head_oof_predictions.parquet", "fold_manifest.parquet")
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--fold-column", default="oof_fold")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        ledger=args.ledger,
        semantic_labels=args.semantic_labels,
        semantic_contract=args.semantic_contract,
        features_json=args.features_json,
        fold_column=args.fold_column,
        output=args.output,
    ), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
