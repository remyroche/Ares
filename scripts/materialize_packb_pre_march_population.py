#!/usr/bin/env python3
"""Stream immutable DEC-09 pre-March Pack-B identity/cohort ledgers.

The materializer deliberately projects identity/timing columns only.  It is not
a feature loader, trainer, or OOF generator.  Learned stages must consume its
per-side, fixed-calendar ledgers rather than re-deriving broad pre-March
populations from a mutable label directory.
"""

from __future__ import annotations

import argparse
import hashlib
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

from extreme_price_movements import packb_pre_march_source_authorization as source_auth
from extreme_price_movements.training_resource_guard import TrainingResourceGuard

SCHEMA = "packb_pre_march_population_materialization_v1"
CANONICAL_SIDES = ("long", "short")
OUTPUT_COLUMNS = (
    "candidate_id",
    "side_name",
    "__ts__",
    "__decision_ts__",
    "__label_resolution_ts__",
    "__symbol__",
)
RESOLUTION_CUTOFF = pd.Timestamp("2026-03-01T00:00:00Z")
DECISION_LAG = pd.Timedelta(hours=1)
BASE_LABEL_HORIZON = pd.Timedelta(hours=24)
TRAIN_PURGE = pd.Timedelta(hours=25)
AE_SIGNAL_START = pd.Timestamp("2025-01-01T00:00:00Z")
FS_WINDOW = (pd.Timestamp("2025-11-01T00:00:00Z"), pd.Timestamp("2025-12-01T00:00:00Z"))
HPO_WINDOWS = (
    (pd.Timestamp("2025-12-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-02-01T00:00:00Z")),
    (pd.Timestamp("2026-02-01T00:00:00Z"), pd.Timestamp("2026-03-01T00:00:00Z")),
)
OUTER_FOLDS = (
    ("2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),
    ("2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    ("2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("2026-07-01T00:00:00Z", "2026-07-11T00:00:00Z"),
)
DEFAULT_DECISIONS = ROOT / "config/full_pipeline_decisions_20260724.json"
DEFAULT_LABELS = (
    ROOT
    / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_AUDIT = (
    ROOT / "docs/pipeline_roadmap/20260724/r3/current_label_inventory_audit.json"
)


class PopulationMaterializationError(ValueError):
    """Raised when a canonical population ledger cannot be safely emitted."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PopulationMaterializationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PopulationMaterializationError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise PopulationMaterializationError(f"{name} must have an explicit UTC offset")
    timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise PopulationMaterializationError(f"{name} is not a valid timestamp")
    return timestamp


def locked_calendar() -> dict[str, Any]:
    return {
        "ae_gmm_reference_signal_interval": [
            AE_SIGNAL_START.isoformat(),
            FS_WINDOW[0].isoformat(),
        ],
        "feature_selection_validation_interval": [
            item.isoformat() for item in FS_WINDOW
        ],
        "hpo_validation_intervals": [
            [item.isoformat() for item in window] for window in HPO_WINDOWS
        ],
        "resolution_cutoff_utc": RESOLUTION_CUTOFF.isoformat(),
        "train_predicate": "signal < validation_start - 25h AND decision = signal + 1h AND decision + 24h < validation_start",
        "validation_predicate": "validation_start <= signal < validation_end AND decision = signal + 1h AND decision + 24h < 2026-03-01T00:00:00Z",
        "outer_folds": [list(window) for window in OUTER_FOLDS],
    }


def parse_locked_dec09(path: Path = DEFAULT_DECISIONS) -> dict[str, Any]:
    payload = _json(path)
    decision = payload.get("decisions", {}).get("DEC-09")
    if (
        payload.get("schema_version") != "full_pipeline_decisions_v1"
        or payload.get("status") != "LOCKED_BEFORE_NEW_TRAINING"
    ):
        raise PopulationMaterializationError("locked full_pipeline DEC-09 is required")
    if not isinstance(decision, dict):
        raise PopulationMaterializationError("decisions.DEC-09 is required")
    if (
        _utc(
            decision.get("feature_selection_hpo_resolution_cutoff_utc"),
            name="DEC-09 cutoff",
        )
        != RESOLUTION_CUTOFF
    ):
        raise PopulationMaterializationError("DEC-09 resolution cutoff changed")
    if decision.get("decision_timestamp") != "signal_timestamp + 1 hour":
        raise PopulationMaterializationError(
            "DEC-09 decision-timestamp contract changed"
        )
    inner = decision.get("packb_pre_march_inner_calendar")
    expected_inner = {
        "ae_gmm_reference_signal_interval": [
            "2025-01-01T00:00:00Z",
            "2025-11-01T00:00:00Z",
        ],
        "feature_selection_validation_interval": [
            "2025-11-01T00:00:00Z",
            "2025-12-01T00:00:00Z",
        ],
        "hpo_validation_intervals": [
            ["2025-12-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
            ["2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"],
        ],
    }
    if not isinstance(inner, dict) or any(
        inner.get(key) != value for key, value in expected_inner.items()
    ):
        raise PopulationMaterializationError("DEC-09 pre-March inner calendar changed")
    actual_outer = tuple(
        tuple(item)
        for item in decision.get("outer_folds", ())
        if isinstance(item, list)
    )
    if actual_outer != OUTER_FOLDS:
        raise PopulationMaterializationError("DEC-09 outer-fold calendar changed")
    return {"path": str(path), "sha256": _sha256(path), "calendar": locked_calendar()}


def strict_train_mask(
    signal: pd.Series, decision: pd.Series, validation_start: pd.Timestamp
) -> pd.Series:
    resolved = decision + BASE_LABEL_HORIZON
    return (
        signal.lt(validation_start - TRAIN_PURGE)
        & decision.eq(signal + DECISION_LAG)
        & resolved.lt(validation_start)
    )


def strict_validation_mask(
    signal: pd.Series,
    decision: pd.Series,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.Series:
    resolved = decision + BASE_LABEL_HORIZON
    return (
        signal.ge(validation_start)
        & signal.lt(validation_end)
        & decision.eq(signal + DECISION_LAG)
        & resolved.lt(RESOLUTION_CUTOFF)
    )


def _audit_shards_and_row_counts(
    labels_dir: Path, causal_audit_path: Path
) -> list[Path]:
    """Read one exact audit inventory and reject any tail-mutated shard."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - project dependency.
        raise PopulationMaterializationError(
            "pyarrow is required for materialization"
        ) from exc
    audit = _json(causal_audit_path)
    if (
        audit.get("schema") != "packb_current_canonical_label_inventory_audit_v1"
        or audit.get("status") != "PASS"
    ):
        raise PopulationMaterializationError(
            "current canonical label inventory audit must be PASS"
        )
    entries, inventory, totals = (
        audit.get("per_file"),
        audit.get("inventory"),
        audit.get("totals"),
    )
    if not isinstance(entries, list) or not entries:
        raise PopulationMaterializationError(
            "causal audit has no per_file shard inventory"
        )
    if not isinstance(inventory, dict) or not isinstance(totals, dict):
        raise PopulationMaterializationError("current audit misses inventory or totals")
    exclusions = inventory.get("excluded_unlisted_monolithic_files")
    if exclusions != ["train_global_short_7.parquet"]:
        raise PopulationMaterializationError(
            "current audit must explicitly exclude only train_global_short_7.parquet"
        )
    declared_canonical_count = int(inventory.get("canonical_monthly_files", -1))
    if declared_canonical_count != len(entries):
        raise PopulationMaterializationError(
            "current audit canonical_monthly_files does not match its per_file inventory"
        )
    actual_names = {
        item.name for item in labels_dir.glob("*.parquet") if item.is_file()
    }
    paths: list[Path] = []
    total = 0
    expected_names: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise PopulationMaterializationError(
                "causal audit has an invalid per_file shard"
            )
        name = item["file"]
        if (
            Path(name).name != name
            or not name.endswith(".parquet")
            or name in expected_names
        ):
            raise PopulationMaterializationError(
                "causal audit has an unsafe or duplicate shard name"
            )
        expected_names.add(name)
        try:
            expected_rows = int(item["expected_current_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PopulationMaterializationError(
                "causal audit per_file row count is invalid"
            ) from exc
        path = labels_dir / name
        if not path.is_file():
            raise PopulationMaterializationError(f"missing canonical shard: {name}")
        actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
        if actual_rows != expected_rows:
            raise PopulationMaterializationError(
                f"stale causal audit row count for {name}: audit={expected_rows} actual={actual_rows}"
            )
        paths.append(path)
        total += actual_rows
    allowed_names = expected_names.union(exclusions)
    if actual_names != allowed_names:
        extras, missing = (
            sorted(actual_names - allowed_names),
            sorted(expected_names - actual_names),
        )
        raise PopulationMaterializationError(
            f"causal audit inventory is not exact: extras={extras[:6]} missing={missing[:6]}"
        )
    if (
        int(totals.get("rows", -1)) != total
        or int(inventory.get("expected_current_rows", -1)) != total
    ):
        raise PopulationMaterializationError(
            "causal audit aggregate counts do not match per_file inventory"
        )
    return paths


class _LedgerWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = 0
        self._writer: Any = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame.loc[:, OUTPUT_COLUMNS], preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self._writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _normalise_batch(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    required = {"candidate_id", "__ts__", "__decision_ts__", "side_name"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PopulationMaterializationError(
            f"{source_name} misses required identity columns: {missing}"
        )
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="coerce")
    if (
        signal.isna().any()
        or decision.isna().any()
        or not decision.eq(signal + DECISION_LAG).all()
    ):
        raise PopulationMaterializationError(
            f"{source_name} violates signal/decision +1h contract"
        )
    if "__signal_ts__" in frame:
        declared_signal = pd.to_datetime(
            frame["__signal_ts__"], utc=True, errors="coerce"
        )
        if declared_signal.isna().any() or not declared_signal.eq(signal).all():
            raise PopulationMaterializationError(
                f"{source_name} has inconsistent __signal_ts__"
            )
    side = frame["side_name"].astype("string").str.strip().str.lower()
    if side.isna().any() or not side.isin(CANONICAL_SIDES).all():
        raise PopulationMaterializationError(
            f"{source_name} has non-canonical side_name"
        )
    ids = frame["candidate_id"].astype("string")
    if (
        ids.isna().any()
        or ids.str.strip().eq("").any()
        or ids.ne(ids.str.strip()).any()
    ):
        raise PopulationMaterializationError(f"{source_name} has invalid candidate_id")
    symbols = frame.get(
        "__symbol__", pd.Series(pd.NA, index=frame.index, dtype="string")
    ).astype("string")
    return pd.DataFrame(
        {
            "candidate_id": ids.astype(str),
            "side_name": side.astype(str),
            "__ts__": signal,
            "__decision_ts__": decision,
            "__label_resolution_ts__": decision + BASE_LABEL_HORIZON,
            "__symbol__": symbols,
        }
    )


def _cohort_paths(stage: Path) -> dict[str, _LedgerWriter]:
    writers = {
        "authorized_population": _LedgerWriter(
            stage / "authorized_pre_march_population.parquet"
        )
    }
    for side in CANONICAL_SIDES:
        base = stage / "cohorts" / side
        writers[f"{side}/ae_reference"] = _LedgerWriter(base / "ae_reference.parquet")
        writers[f"{side}/fs_train"] = _LedgerWriter(
            base / "feature_selection_train.parquet"
        )
        writers[f"{side}/fs_valid"] = _LedgerWriter(
            base / "feature_selection_valid.parquet"
        )
        for index in range(1, 4):
            writers[f"{side}/hpo_{index}_train"] = _LedgerWriter(
                base / f"hpo_{index}_train.parquet"
            )
            writers[f"{side}/hpo_{index}_valid"] = _LedgerWriter(
                base / f"hpo_{index}_valid.parquet"
            )
    return writers


def _write_batch_cohorts(
    writers: Mapping[str, _LedgerWriter], frame: pd.DataFrame
) -> None:
    authorized = frame.loc[frame["__label_resolution_ts__"].lt(RESOLUTION_CUTOFF)]
    writers["authorized_population"].write(authorized)
    for side in CANONICAL_SIDES:
        local = authorized.loc[authorized["side_name"].eq(side)]
        signal, decision = local["__ts__"], local["__decision_ts__"]
        ae = local.loc[
            signal.ge(AE_SIGNAL_START)
            & signal.lt(FS_WINDOW[0])
            & local["__label_resolution_ts__"].lt(FS_WINDOW[0])
        ]
        writers[f"{side}/ae_reference"].write(ae)
        writers[f"{side}/fs_train"].write(
            local.loc[strict_train_mask(signal, decision, FS_WINDOW[0])]
        )
        writers[f"{side}/fs_valid"].write(
            local.loc[strict_validation_mask(signal, decision, *FS_WINDOW)]
        )
        for index, window in enumerate(HPO_WINDOWS, start=1):
            writers[f"{side}/hpo_{index}_train"].write(
                local.loc[strict_train_mask(signal, decision, window[0])]
            )
            writers[f"{side}/hpo_{index}_valid"].write(
                local.loc[strict_validation_mask(signal, decision, *window)]
            )


def _ledger_records(
    writers: Mapping[str, _LedgerWriter], stage: Path
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for key, writer in writers.items():
        if writer.rows <= 0:
            raise PopulationMaterializationError(f"required ledger is empty: {key}")
        records[key] = {
            "path": str(writer.path.relative_to(stage)),
            "rows": writer.rows,
            "sha256": _sha256(writer.path),
            "identity_stream_sha256": _identity_stream_sha256(writer.path),
        }
    return records


def _identity_stream_sha256(path: Path) -> str:
    """Hash logical ledger identities independently of Parquet row-group sizing."""
    import pyarrow.parquet as pq

    digest = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=65_536, columns=list(OUTPUT_COLUMNS)
    ):
        values = batch.to_pydict()
        for row in zip(*(values[column] for column in OUTPUT_COLUMNS), strict=True):
            for column, value in zip(OUTPUT_COLUMNS, row, strict=True):
                if column.endswith("_ts__"):
                    encoded = pd.Timestamp(value).isoformat()
                else:
                    encoded = "" if value is None else str(value)
                digest.update(encoded.encode("utf-8"))
                digest.update(b"\x1f")
            digest.update(b"\n")
    return digest.hexdigest()


def _new_stage(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"


def materialize(
    *,
    decisions_path: Path = DEFAULT_DECISIONS,
    labels_dir: Path = DEFAULT_LABELS,
    causal_audit_path: Path = DEFAULT_AUDIT,
    output_dir: Path,
    batch_rows: int = source_auth.DEFAULT_BATCH_ROWS,
    contract_only: bool = False,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Materialize exact pre-March identity ledgers without loading a full shard."""
    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise PopulationMaterializationError("batch_rows must be a positive integer")
    stage = _new_stage(Path(output_dir))
    guard = resource_guard or TrainingResourceGuard(
        disk_path=stage.parent,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    try:
        guard.preflight("packb_pre_march_materialization_preflight")
        dec09 = parse_locked_dec09(Path(decisions_path))
        shards = _audit_shards_and_row_counts(Path(labels_dir), Path(causal_audit_path))
        contract = {
            "schema": SCHEMA,
            "status": "CONTRACT_ONLY" if contract_only else "MATERIALIZING",
            "dec09": dec09,
            "input": {
                "labels_dir": str(labels_dir),
                "causal_audit_path": str(causal_audit_path),
                "causal_audit_sha256": _sha256(Path(causal_audit_path)),
                "canonical_shards": [path.name for path in shards],
            },
            "streaming": {
                "batch_rows": batch_rows,
                "columns": list(OUTPUT_COLUMNS),
                "ordering": "causal_audit_per_file_order_then_parquet_physical_row_order",
                "full_frame_load": False,
            },
        }
        if contract_only:
            _atomic_json(stage / "materialization_contract.json", contract)
            os.replace(stage, output_dir)
            return contract
        population = source_auth.preflight_pre_march_packb_population(
            labels_dir=Path(labels_dir),
            causal_audit_path=Path(causal_audit_path),
            batch_rows=batch_rows,
            checkpoint=lambda checkpoint: guard.checkpoint(
                f"packb_pre_march_source_auth:{checkpoint}"
            ),
            scratch_dir=stage,
        )
        guard.checkpoint("packb_pre_march_population_preflight_complete")
        import pyarrow.parquet as pq

        writers = _cohort_paths(stage)
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
                    _write_batch_cohorts(
                        writers,
                        _normalise_batch(batch.to_pandas(), source_name=shard.name),
                    )
                guard.checkpoint(f"packb_pre_march_shard:{shard.name}")
        finally:
            for writer in writers.values():
                writer.close()
        ledgers = _ledger_records(writers, stage)
        expected_rows = sum(
            int(population["authorized_population_by_side"][side]["authorized_rows"])
            for side in CANONICAL_SIDES
        )
        if ledgers["authorized_population"]["rows"] != expected_rows:
            raise PopulationMaterializationError(
                "authorized ledger rows do not match population preflight"
            )
        guard.checkpoint("packb_pre_march_materialization_complete")
        manifest = {
            **contract,
            "status": "MATERIALIZED_IMMUTABLE",
            "population_preflight": population,
            "ledgers": ledgers,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-rows", type=int, default=source_auth.DEFAULT_BATCH_ROWS
    )
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
    except (
        PopulationMaterializationError,
        source_auth.PackBSourceAuthorizationError,
        FileExistsError,
        ValueError,
    ) as exc:
        print(
            json.dumps({"status": "BLOCKED_PRECONDITION_FAILED", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
