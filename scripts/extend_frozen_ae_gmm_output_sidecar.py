#!/usr/bin/env python3
"""Extend a frozen AE/GMM output sidecar for newly materialized label rows.

Existing outputs are reused only when their serialized AE/GMM state and output
contract match exactly.  Missing row identities are transformed from the
canonical static feature store, then DuckDB rewrites one current-key sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from extreme_price_movements.features_gmm_ae import (
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _canonical_label_files,
    _feature_contract_hash,
    _sha256_file,
    _source_files_signature,
)

KEYS = ("__ts__", "__symbol__", "side")


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_paths(paths: Sequence[Path]) -> str:
    return "[" + ",".join(_sql_literal(path) for path in paths) + "]"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _key_join_condition(left_alias: str, right_alias: str) -> str:
    """Join timestamps by UTC epoch so naive/aware parquet schemas agree."""

    return " AND ".join(
        (
            f"epoch_ns({left_alias}.{_quote_identifier(key)}) = "
            f"epoch_ns({right_alias}.{_quote_identifier(key)})"
            if key == "__ts__"
            else f"{left_alias}.{_quote_identifier(key)} = "
            f"{right_alias}.{_quote_identifier(key)}"
        )
        for key in KEYS
    )


def _restore_case_sensitive_output_names(
    path: Path,
    *,
    output_features: Sequence[str],
) -> None:
    """Undo DuckDB's suffixing of columns that differ only by case.

    The frozen transform intentionally emits both ``AE_reconstruction_error``
    and ``ae_reconstruction_error`` for compatibility with existing model
    contracts. DuckDB resolves identifiers case-insensitively and writes the
    latter as ``ae_reconstruction_error_1``. Parquet itself supports both exact
    names, so restore the serialized feature contract in a streaming rewrite.
    """

    parquet_file = pq.ParquetFile(path)
    actual = list(map(str, parquet_file.schema_arrow.names))
    expected = [*KEYS, *map(str, output_features)]
    if actual == expected:
        return
    missing = [name for name in expected if name not in actual]
    extras = [name for name in actual if name not in expected]
    replacements: dict[str, str] = {}
    for name in missing:
        candidates = [
            candidate
            for candidate in extras
            if candidate.lower().startswith(name.lower() + "_")
            and candidate.rsplit("_", 1)[-1].isdigit()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Cannot restore exact frozen AE/GMM output schema: "
                f"missing={missing[:8]} extras={extras[:8]}"
            )
        replacements[candidates[0]] = name
    renamed = [replacements.get(name, name) for name in actual]
    if renamed != expected or len(set(renamed)) != len(renamed):
        raise RuntimeError(
            "Frozen AE/GMM output schema restoration produced an invalid contract"
        )

    repaired = path.with_suffix(path.suffix + ".schema_repair")
    repaired.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet_file.iter_batches(batch_size=250_000):
            table = pa.Table.from_batches([batch]).rename_columns(renamed)
            if writer is None:
                writer = pq.ParquetWriter(
                    repaired,
                    table.schema,
                    compression="zstd",
                    compression_level=5,
                )
            writer.write_table(table, row_group_size=250_000)
    finally:
        if writer is not None:
            writer.close()
    repaired.replace(path)


def _read_manifest(sidecar_path: Path) -> dict[str, Any]:
    manifest_path = sidecar_path.with_suffix(".manifest.json")
    if not sidecar_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Reuse sidecar requires parquet and manifest: {sidecar_path}"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _materialize_missing_outputs(
    *,
    missing_keys_path: Path,
    feature_dir: Path,
    state: dict[str, Any],
    output_features: Sequence[str],
    output_path: Path,
    batch_rows: int,
) -> int:
    inputs = [str(value) for value in state.get("feature_columns", []) or []]
    if not inputs:
        raise RuntimeError("Frozen AE/GMM state has no ordered input contract")
    fill_map = {
        str(key): np.float32(value)
        for key, value in dict(state.get("cycle_input_fill_values", {}) or {}).items()
    }
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        parquet_file = pq.ParquetFile(missing_keys_path)
        for batch in parquet_file.iter_batches(batch_size=max(1_000, int(batch_rows))):
            frame = pa.Table.from_batches([batch]).to_pandas()
            fetched, report = _load_feature_store_columns(
                frame,
                feature_dir=feature_dir,
                selected_features=[name for name in inputs if name != "side"],
                min_feature_finite_frac=1e-12,
            )
            if "side" in inputs:
                fetched["side"] = pd.to_numeric(
                    frame["side"], errors="raise"
                ).to_numpy(dtype=np.float32, copy=False)
            missing_inputs = [name for name in inputs if name not in fetched.columns]
            if missing_inputs:
                raise RuntimeError(
                    "Incremental AE/GMM transform cannot source frozen inputs: "
                    f"{missing_inputs[:20]}; reader={report.get('reader')}"
                )
            x = fetched.reindex(columns=inputs).apply(pd.to_numeric, errors="coerce")
            x = x.replace([np.inf, -np.inf], np.nan)
            for name in inputs:
                x[name] = x[name].fillna(fill_map.get(name, np.float32(0.0)))
            x = x.astype(np.float32, copy=False)
            values = x.to_numpy(dtype=np.float32, copy=False)
            if not bool(np.isfinite(values).all()):
                raise RuntimeError("Incremental AE/GMM inputs remain non-finite")
            generated = transform_ae_gmm_features(x, state).loc[
                :, list(output_features)
            ]
            out = pd.DataFrame(
                {
                    "__ts__": pd.to_datetime(frame["__ts__"], utc=True),
                    "__symbol__": frame["__symbol__"].astype(str),
                    "side": pd.to_numeric(frame["side"], errors="raise").astype(
                        np.int8
                    ),
                }
            )
            out = pd.concat(
                [out.reset_index(drop=True), generated.reset_index(drop=True)],
                axis=1,
                copy=False,
            )
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                    compression_level=5,
                )
            writer.write_table(table, row_group_size=max(1_000, int(batch_rows)))
            rows += len(out)
            print(f"[ae_gmm_incremental] transformed_missing_rows={rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return int(rows)


def extend_sidecar(
    *,
    labels_path: Path,
    feature_dir: Path,
    state_path: Path,
    reuse_sidecar_path: Path,
    output_path: Path,
    batch_rows: int = 100_000,
) -> dict[str, Any]:
    label_files = _canonical_label_files(labels_path)
    if not label_files:
        raise FileNotFoundError(f"No canonical label partitions in {labels_path}")
    reuse_manifest = _read_manifest(reuse_sidecar_path)
    state_sha256 = _sha256_file(state_path)
    output_features = [
        str(value) for value in reuse_manifest.get("output_features", []) or []
    ]
    if not output_features:
        raise RuntimeError("Reuse sidecar manifest has no output feature contract")
    expected_output_hash = _feature_contract_hash(output_features)
    if str(reuse_manifest.get("state_sha256")) != state_sha256:
        raise RuntimeError("Refusing incremental reuse across different AE/GMM states")
    if str(reuse_manifest.get("output_feature_hash")) != expected_output_hash:
        raise RuntimeError("Refusing incremental reuse across output contracts")
    if str(reuse_manifest.get("input_source_policy")) != (
        "shared_static_store_authoritative_v1"
    ):
        raise RuntimeError("Reuse sidecar has a non-canonical input source policy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f".{output_path.stem}_incremental_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    missing_keys_path = work_dir / "missing_keys.parquet"
    missing_outputs_path = work_dir / "missing_outputs.parquet"
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    for path in (missing_keys_path, missing_outputs_path, temp_output_path):
        path.unlink(missing_ok=True)

    labels_sql = _sql_paths(label_files)
    old_sql = _sql_literal(reuse_sidecar_path)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")
    key_join = _key_join_condition("l", "o")
    con.execute(
        f"""
        COPY (
            SELECT l.__ts__, l.__symbol__, CAST(l.side AS TINYINT) AS side
            FROM read_parquet({labels_sql}, union_by_name=true) AS l
            ANTI JOIN read_parquet({old_sql}) AS o ON {key_join}
        ) TO {_sql_literal(missing_keys_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    missing_rows = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet({_sql_literal(missing_keys_path)})"
        ).fetchone()[0]
    )
    state = load_ae_gmm_state_artifact(state_path)
    transformed_rows = 0
    if missing_rows:
        transformed_rows = _materialize_missing_outputs(
            missing_keys_path=missing_keys_path,
            feature_dir=feature_dir,
            state=state,
            output_features=output_features,
            output_path=missing_outputs_path,
            batch_rows=batch_rows,
        )
        if transformed_rows != missing_rows:
            raise RuntimeError(
                f"Missing-output row mismatch: keys={missing_rows} outputs={transformed_rows}"
            )

    new_relation = (
        f"read_parquet({_sql_literal(missing_outputs_path)})"
        if missing_rows
        else "(SELECT * FROM read_parquet(" + old_sql + ") WHERE false)"
    )
    feature_projection = ",\n".join(
        f"COALESCE(n.{_quote_identifier(name)}, o.{_quote_identifier(name)}) "
        f"AS {_quote_identifier(name)}"
        for name in output_features
    )
    new_join = _key_join_condition("l", "n")
    con.execute(
        f"""
        COPY (
            SELECT
                l.__ts__, l.__symbol__, CAST(l.side AS TINYINT) AS side,
                {feature_projection}
            FROM read_parquet({labels_sql}, union_by_name=true) AS l
            LEFT JOIN read_parquet({old_sql}) AS o ON {key_join}
            LEFT JOIN {new_relation} AS n ON {new_join}
        ) TO {_sql_literal(temp_output_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    _restore_case_sensitive_output_names(
        temp_output_path,
        output_features=output_features,
    )
    source_rows = int(
        sum(pq.ParquetFile(path).metadata.num_rows for path in label_files)
    )
    output_rows = int(pq.ParquetFile(temp_output_path).metadata.num_rows)
    if output_rows != source_rows:
        raise RuntimeError(
            f"Extended sidecar coverage mismatch: output={output_rows} labels={source_rows}"
        )
    null_predicate = " OR ".join(
        f"{_quote_identifier(name)} IS NULL" for name in output_features
    )
    null_rows = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet({_sql_literal(temp_output_path)}) "
            f"WHERE {null_predicate}"
        ).fetchone()[0]
    )
    if null_rows:
        raise RuntimeError(f"Extended sidecar has {null_rows} rows with null outputs")
    output_path.unlink(missing_ok=True)
    temp_output_path.replace(output_path)

    old_rows = int(reuse_manifest.get("rows", 0) or 0)
    overlap_rows = source_rows - missing_rows
    obsolete_rows = max(0, old_rows - overlap_rows)
    contract = {
        **reuse_manifest,
        "schema": "frozen_ae_gmm_selected_output_sidecar_v1",
        "status": "incrementally_extended",
        "path": str(output_path),
        "rows": source_rows,
        "source_rows": source_rows,
        "source_signature": _source_files_signature(label_files),
        "state_path": str(state_path),
        "state_sha256": state_sha256,
        "output_feature_hash": expected_output_hash,
        "input_source_policy": "shared_static_store_authoritative_v1",
        "materialization": "exact_key_reuse_plus_missing_static_store_transform_v1",
        "incremental_reuse": {
            "source_sidecar": str(reuse_sidecar_path),
            "source_manifest": str(reuse_sidecar_path.with_suffix('.manifest.json')),
            "source_rows": old_rows,
            "overlap_rows": overlap_rows,
            "missing_rows_transformed": missing_rows,
            "obsolete_rows_dropped": obsolete_rows,
            "key_columns": list(KEYS),
        },
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (missing_keys_path, missing_outputs_path):
        path.unlink(missing_ok=True)
    try:
        work_dir.rmdir()
    except OSError:
        pass
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--reuse-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = extend_sidecar(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        state_path=args.state_path,
        reuse_sidecar_path=args.reuse_sidecar,
        output_path=args.output,
        batch_rows=args.batch_rows,
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
