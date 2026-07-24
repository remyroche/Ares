#!/usr/bin/env python3
"""Materialize immutable, side-local Pack-B outer-OOF fold ledgers.

The pre-March population is the authority for AE/GMM, feature selection, and
HPO only.  Outer-OOF fitting needs a distinct population whose training cutoff
moves with each April--July validation fold.  This script scans the exact same
audited canonical label inventory and emits identity/timing ledgers only:

* long and short are always written separately;
* a training row is admitted only when its 24-hour label resolves strictly
  before the validation start and its signal is at least 25 hours earlier;
* a validation row belongs to exactly one fixed half-open DEC-09 fold; and
* no feature, target, weight, or model value is loaded or persisted here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.training_resource_guard import TrainingResourceGuard
from scripts import materialize_packb_pre_march_population as inner_population

SCHEMA = "packb_outer_oof_population_materialization_v1"
CANONICAL_SIDES = inner_population.CANONICAL_SIDES
OUTER_FOLDS = tuple(
    (
        f"outer_{index}_{pd.Timestamp(start).strftime('%Y%m%d')}",
        pd.Timestamp(start),
        pd.Timestamp(end),
    )
    for index, (start, end) in enumerate(inner_population.OUTER_FOLDS, start=1)
)
OUTPUT_COLUMNS = inner_population.OUTPUT_COLUMNS
DEFAULT_DECISIONS = inner_population.DEFAULT_DECISIONS
DEFAULT_LABELS = inner_population.DEFAULT_LABELS
DEFAULT_AUDIT = inner_population.DEFAULT_AUDIT
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/packb_outer_oof_population_20260724_v1"


class OuterPopulationMaterializationError(ValueError):
    """Raised when fixed outer-OOF ledgers cannot be safely published."""


def strict_outer_train_mask(
    signal: pd.Series, decision: pd.Series, validation_start: pd.Timestamp
) -> pd.Series:
    resolution = decision + inner_population.BASE_LABEL_HORIZON
    return (
        signal.lt(validation_start - inner_population.TRAIN_PURGE)
        & decision.eq(signal + inner_population.DECISION_LAG)
        & resolution.lt(validation_start)
    )


def strict_outer_validation_mask(
    signal: pd.Series,
    decision: pd.Series,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.Series:
    return (
        signal.ge(validation_start)
        & signal.lt(validation_end)
        & decision.eq(signal + inner_population.DECISION_LAG)
    )


def locked_outer_calendar() -> dict[str, Any]:
    return {
        "folds": [
            {
                "name": name,
                "validation_start_utc": start.isoformat(),
                "validation_end_utc": end.isoformat(),
            }
            for name, start, end in OUTER_FOLDS
        ],
        "decision_contract": "__decision_ts__ = __ts__ + 1h",
        "label_resolution_contract": (
            "__label_resolution_ts__ = __decision_ts__ + 24h"
        ),
        "train_rule": (
            "__ts__ < validation_start - 25h AND "
            "__label_resolution_ts__ < validation_start"
        ),
        "validation_rule": ("validation_start <= __ts__ < validation_end"),
        "fold_overlap": "FORBIDDEN",
        "side_pooling": "FORBIDDEN",
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _new_stage(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"


def _writers(stage: Path) -> dict[str, inner_population._LedgerWriter]:
    result: dict[str, inner_population._LedgerWriter] = {}
    for side in CANONICAL_SIDES:
        for name, _, _ in OUTER_FOLDS:
            base = stage / "folds" / name / side
            result[f"{name}/{side}/train"] = inner_population._LedgerWriter(
                base / "train.parquet"
            )
            result[f"{name}/{side}/validation"] = inner_population._LedgerWriter(
                base / "validation.parquet"
            )
    return result


def _write_batch(
    writers: Mapping[str, inner_population._LedgerWriter], frame: pd.DataFrame
) -> None:
    for side in CANONICAL_SIDES:
        local = frame.loc[frame["side_name"].eq(side)]
        signal = local["__ts__"]
        decision = local["__decision_ts__"]
        for name, start, end in OUTER_FOLDS:
            writers[f"{name}/{side}/train"].write(
                local.loc[strict_outer_train_mask(signal, decision, start)]
            )
            writers[f"{name}/{side}/validation"].write(
                local.loc[strict_outer_validation_mask(signal, decision, start, end)]
            )


def _validate_disjoint_validation_ledgers(
    records: Mapping[str, Any], stage: Path
) -> None:
    import pyarrow.parquet as pq

    for side in CANONICAL_SIDES:
        seen: set[str] = set()
        for name, _, _ in OUTER_FOLDS:
            record = records[f"{name}/{side}/validation"]
            path = stage / str(record["path"])
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=65_536, columns=["candidate_id"]
            ):
                values = batch.column(0).to_pylist()
                duplicates = seen.intersection(map(str, values))
                if duplicates:
                    raise OuterPopulationMaterializationError(
                        f"{side} validation candidate appears in multiple folds"
                    )
                seen.update(map(str, values))


def materialize(
    *,
    decisions_path: Path = DEFAULT_DECISIONS,
    labels_dir: Path = DEFAULT_LABELS,
    causal_audit_path: Path = DEFAULT_AUDIT,
    output_dir: Path = DEFAULT_OUTPUT,
    batch_rows: int = 65_536,
    contract_only: bool = False,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Stream and publish the fixed, side-local April--July outer folds."""
    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise OuterPopulationMaterializationError(
            "batch_rows must be a positive integer"
        )
    output_dir = Path(output_dir)
    stage = _new_stage(output_dir)
    guard = resource_guard or TrainingResourceGuard(
        disk_path=stage.parent,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    try:
        guard.preflight("packb_outer_population_preflight")
        dec09 = inner_population.parse_locked_dec09(Path(decisions_path))
        shards = inner_population._audit_shards_and_row_counts(
            Path(labels_dir), Path(causal_audit_path)
        )
        contract = {
            "schema": SCHEMA,
            "status": "CONTRACT_ONLY" if contract_only else "MATERIALIZING",
            "dec09": dec09,
            "calendar": locked_outer_calendar(),
            "input": {
                "labels_dir": str(labels_dir),
                "causal_audit_path": str(causal_audit_path),
                "causal_audit_sha256": inner_population._sha256(
                    Path(causal_audit_path)
                ),
                "canonical_shards": [path.name for path in shards],
            },
            "streaming": {
                "batch_rows": batch_rows,
                "columns": list(OUTPUT_COLUMNS),
                "ordering": (
                    "causal_audit_per_file_order_then_parquet_physical_row_order"
                ),
                "full_frame_load": False,
            },
        }
        if contract_only:
            _atomic_json(stage / "materialization_contract.json", contract)
            os.replace(stage, output_dir)
            return contract

        import pyarrow.parquet as pq

        writers = _writers(stage)
        try:
            for shard in shards:
                parquet = pq.ParquetFile(shard)
                requested = [
                    column
                    for column in (
                        "candidate_id",
                        "side_name",
                        "__ts__",
                        "__decision_ts__",
                        "__signal_ts__",
                        "__symbol__",
                    )
                    if column in parquet.schema.names
                ]
                for batch in parquet.iter_batches(
                    batch_size=batch_rows, columns=requested
                ):
                    _write_batch(
                        writers,
                        inner_population._normalise_batch(
                            batch.to_pandas(), source_name=shard.name
                        ),
                    )
                guard.checkpoint(f"packb_outer_population_shard:{shard.name}")
        finally:
            for writer in writers.values():
                writer.close()

        records = inner_population._ledger_records(writers, stage)
        _validate_disjoint_validation_ledgers(records, stage)
        for side in CANONICAL_SIDES:
            prior_train_rows = -1
            for name, _, _ in OUTER_FOLDS:
                train_rows = int(records[f"{name}/{side}/train"]["rows"])
                if train_rows <= prior_train_rows:
                    raise OuterPopulationMaterializationError(
                        f"{side} outer training population is not expanding"
                    )
                prior_train_rows = train_rows
        guard.checkpoint("packb_outer_population_complete")
        manifest = {
            **contract,
            "status": "MATERIALIZED_IMMUTABLE",
            "ledgers": records,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--causal-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = materialize(
            decisions_path=args.decisions,
            labels_dir=args.labels_dir,
            causal_audit_path=args.causal_audit,
            output_dir=args.output_dir,
            batch_rows=args.batch_rows,
            contract_only=args.contract_only,
        )
    except (OuterPopulationMaterializationError, ValueError, FileExistsError) as exc:
        print(
            json.dumps({"status": "BLOCKED_PRECONDITION_FAILED", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
