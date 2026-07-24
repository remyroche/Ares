#!/usr/bin/env python3
"""Verify frozen base AE/GMM outputs against the shared static feature store.

Labels are used only for UTC timestamp, symbol, and side identities.  The
frozen state's ordered raw inputs are read exclusively through the shared
static-store loader before applying the serialized transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_FEATURE_COLUMNS,
    ae_gmm_cycle_sample_identity_hash,
    ae_gmm_input_feature_order_hash,
    ae_gmm_learned_transform_hash,
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _load_feature_store_columns,
)

KEY_COLUMNS = ("__ts__", "__symbol__", "side")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No parquet label files found under {path}")
    return files


def _parquet_schema_names(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(map(str, pq.read_schema(path).names))
    except Exception:
        return set(map(str, pd.read_parquet(path).columns))


def _duckdb_paths(paths: Sequence[Path]) -> str:
    return ", ".join(
        "'" + str(path.resolve()).replace("'", "''") + "'" for path in paths
    )


def _bme_positions(row_count: int, sample_rows: int) -> np.ndarray:
    """Select deterministic positions across beginning, middle, and end bands."""

    count = max(0, int(row_count))
    cap = max(1, int(sample_rows))
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    band_count = min(3, count)
    base_length, length_remainder = divmod(count, band_count)
    base_take, take_remainder = divmod(cap, band_count)
    selected: list[np.ndarray] = []
    band_start = 0
    for band_index in range(band_count):
        band_length = base_length + (1 if band_index < length_remainder else 0)
        take = min(band_length, base_take + (1 if band_index < take_remainder else 0))
        if take:
            selected.append(
                band_start
                + np.linspace(0, band_length - 1, take, dtype=np.int64)
            )
        band_start += band_length
    return np.sort(np.unique(np.concatenate(selected))).astype(np.int64, copy=False)


def _normalize_side(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    mapped = np.where(
        text.eq("short").to_numpy() | numeric.lt(0.0).fillna(False).to_numpy(),
        -1,
        np.where(text.eq("long").to_numpy() | numeric.gt(0.0).fillna(False).to_numpy(), 1, 0),
    )
    return pd.Series(mapped, index=values.index, dtype=np.int8)


def _read_bme_label_keys(
    labels_path: Path, *, sample_rows: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only deterministic identity rows; DuckDB keeps label scans bounded."""

    files = _label_files(labels_path)
    missing_by_file = {
        str(path): sorted(set(KEY_COLUMNS).difference(_parquet_schema_names(path)))
        for path in files
    }
    missing_by_file = {
        path: missing for path, missing in missing_by_file.items() if missing
    }
    if missing_by_file:
        raise ValueError(f"Label parquet files are missing identity keys: {missing_by_file}")

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - static store already requires DuckDB.
        raise RuntimeError("DuckDB is required for bounded label-key sampling") from exc

    source = f"read_parquet([{_duckdb_paths(files)}])"
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        row_count = int(connection.execute(f"SELECT count(*) FROM {source}").fetchone()[0])
        if row_count <= 0:
            raise ValueError("Labels contain no rows")
        duplicate = connection.execute(
            f"""
            SELECT 1
            FROM {source}
            GROUP BY "__ts__", "__symbol__", "side"
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise ValueError("Labels are not unique by UTC timestamp, symbol and side")
        positions = pd.DataFrame({"_position": _bme_positions(row_count, sample_rows)})
        connection.register("requested_positions", positions)
        keys = connection.execute(
            f"""
            WITH ordered AS (
                SELECT
                    CAST("__ts__" AS TIMESTAMPTZ) AS "__ts__",
                    CAST("__symbol__" AS VARCHAR) AS "__symbol__",
                    "side" AS "side",
                    row_number() OVER (
                        ORDER BY "__ts__", "__symbol__", "side"
                    ) - 1 AS _position
                FROM {source}
            )
            SELECT ordered."__ts__", ordered."__symbol__", ordered."side"
            FROM ordered
            INNER JOIN requested_positions USING (_position)
            ORDER BY _position
            """
        ).fetchdf()
    finally:
        connection.close()

    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="coerce")
    keys["__symbol__"] = keys["__symbol__"].astype(str)
    keys["side"] = _normalize_side(keys["side"])
    if keys["__ts__"].isna().any() or keys["side"].eq(0).any():
        raise ValueError("Sampled labels contain invalid UTC timestamp or side values")
    if keys.duplicated(list(KEY_COLUMNS), keep=False).any():
        raise ValueError("Sampled labels are not unique by UTC timestamp, symbol and side")
    identity_hash = ae_gmm_cycle_sample_identity_hash(
        keys["__ts__"], symbols=keys["__symbol__"], sides=keys["side"]
    )
    return keys.reset_index(drop=True), {
        "label_files": int(len(files)),
        "label_rows": row_count,
        "sample_rows": int(len(keys)),
        "sample_identity_hash": identity_hash,
        "sampling": "canonical_utc_symbol_side_beginning_middle_end_v1",
    }


def _read_sidecar_sample(
    sidecar_path: Path,
    keys: pd.DataFrame,
    output_features: Sequence[str],
) -> pd.DataFrame:
    """Use a keyed sidecar join so a large frozen output file is never loaded whole."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - static store already requires DuckDB.
        raise RuntimeError("DuckDB is required for bounded sidecar sampling") from exc

    transport_names = [f"__aegmm_output_{index:03d}" for index in range(len(output_features))]
    escaped = str(sidecar_path.resolve()).replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        # DuckDB resolves identifiers case-insensitively and renames later
        # collisions at parquet-scan time.  Map the exact parquet schema to the
        # positional DuckDB scan schema so both reconstruction-error aliases
        # remain independently addressable.
        import pyarrow.parquet as pq

        parquet_names = list(map(str, pq.read_schema(sidecar_path).names))
        duck_names = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{escaped}')"
        ).fetchdf()["column_name"].astype(str).tolist()
        if len(parquet_names) != len(duck_names):
            raise RuntimeError("DuckDB/parquet sidecar schemas have different widths")
        source_lookup = dict(zip(parquet_names, duck_names, strict=True))
        quoted_outputs = ", ".join(
            f's."{source_lookup[name].replace(chr(34), chr(34) * 2)}" AS "{transport}"'
            for name, transport in zip(output_features, transport_names, strict=True)
        )
        selected = ", ".join(
            [
                's."__ts__" AS "__ts__"',
                'CAST(s."__symbol__" AS VARCHAR) AS "__symbol__"',
                'CAST(s."side" AS SMALLINT) AS "side"',
                quoted_outputs,
            ]
        )
        connection.register("sample_keys", keys.loc[:, KEY_COLUMNS])
        out = connection.execute(
            f"""
            SELECT {selected}
            FROM read_parquet('{escaped}') AS s
            INNER JOIN sample_keys AS k
                ON CAST(s."__ts__" AS TIMESTAMPTZ) = k."__ts__"
                AND CAST(s."__symbol__" AS VARCHAR) = k."__symbol__"
                AND CAST(s."side" AS SMALLINT) = k."side"
            """
        ).fetchdf()
    finally:
        connection.close()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side"] = _normalize_side(out["side"])
    out = out.rename(
        columns=dict(zip(transport_names, output_features, strict=True))
    )
    return out


def _sidecar_output_features(sidecar_path: Path) -> tuple[list[str], dict[str, Any]]:
    schema = _parquet_schema_names(sidecar_path)
    missing_keys = sorted(set(KEY_COLUMNS).difference(schema))
    if missing_keys:
        raise ValueError(f"Frozen output sidecar is missing join keys: {missing_keys}")
    manifest_path = sidecar_path.with_suffix(".manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = [str(name) for name in manifest.get("output_features", []) or []]
    outputs = declared or sorted(schema.difference(KEY_COLUMNS))
    missing_outputs = [name for name in outputs if name not in schema]
    unsupported = [name for name in outputs if name not in AE_GMM_FEATURE_COLUMNS]
    if missing_outputs or unsupported:
        raise ValueError(
            "Frozen output sidecar output contract is invalid: "
            f"missing_columns={missing_outputs[:12]} unsupported_outputs={unsupported[:12]}"
        )
    if not outputs:
        raise ValueError("Frozen output sidecar declares no generated AE/GMM outputs")
    return outputs, {
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "manifest": manifest,
        "output_feature_count": int(len(outputs)),
    }


def _availability_report(matrix: pd.DataFrame, inputs: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Report raw-input coverage and return inputs unusable by the frozen transform.

    The fitted AE/GMM contract owns its missing-value handling.  Requiring every
    raw input to be finite on every sampled row creates an unintended
    complete-case contract that neither training nor inference uses.  A column
    is blocking only when it is absent or has no finite observation at all;
    partial missingness remains explicit in the report and is passed unchanged
    to the frozen transform.
    """
    report: dict[str, Any] = {}
    missing: list[str] = []
    for name in inputs:
        if name not in matrix.columns:
            report[name] = {"loaded": False, "finite_rows": 0, "missing_rows": int(len(matrix))}
            missing.append(name)
            continue
        values = pd.to_numeric(matrix[name], errors="coerce").to_numpy(dtype=np.float32)
        finite_rows = int(np.isfinite(values).sum())
        report[name] = {
            "loaded": True,
            "finite_rows": finite_rows,
            "missing_rows": int(len(values) - finite_rows),
        }
        if finite_rows == 0:
            missing.append(name)
    return report, missing


def verify_base_frozen_aegmm_static_store_parity(
    *,
    labels_path: Path,
    feature_store_path: Path,
    state_path: Path,
    sidecar_path: Path,
    sample_rows: int = 2048,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Run the read-only frozen representation parity check and return its report."""

    labels_path = Path(labels_path)
    feature_store_path = Path(feature_store_path)
    state_path = Path(state_path)
    sidecar_path = Path(sidecar_path)
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative value")

    state = load_ae_gmm_state_artifact(state_path)
    inputs = [str(name) for name in state.get("feature_columns", []) or []]
    if not inputs:
        raise ValueError("Frozen AE/GMM state has no ordered raw inputs")
    expected_input_hash = str(state.get("input_feature_order_hash") or "")
    actual_input_hash = ae_gmm_input_feature_order_hash(inputs)
    expected_state_hash = str(state.get("cycle_state_hash") or "")
    actual_state_hash = ae_gmm_learned_transform_hash(state)

    keys, sampling = _read_bme_label_keys(labels_path, sample_rows=sample_rows)
    outputs, sidecar_contract = _sidecar_output_features(sidecar_path)
    loaded, loader_report = _load_feature_store_columns(
        keys.loc[:, KEY_COLUMNS].copy(),
        feature_dir=feature_store_path,
        selected_features=[name for name in inputs if name != "side"],
        min_feature_finite_frac=1e-12,
    )
    if "side" in inputs:
        loaded["side"] = pd.to_numeric(
            keys["side"], errors="raise"
        ).to_numpy(dtype=np.float32, copy=False)
    raw = loaded.reindex(index=keys.index, columns=inputs).copy()
    for name in raw.columns:
        raw[name] = pd.to_numeric(raw[name], errors="coerce").astype(np.float32)
    availability, missing_inputs = _availability_report(raw, inputs)

    errors: list[str] = []
    if expected_input_hash and expected_input_hash != actual_input_hash:
        errors.append("state_input_feature_order_hash_mismatch")
    if expected_state_hash and expected_state_hash != actual_state_hash:
        errors.append("state_learned_transform_hash_mismatch")
    if missing_inputs:
        errors.append("missing_or_all_nonfinite_static_raw_inputs")

    output_report: dict[str, Any] = {}
    sidecar_rows = 0
    raw_values = raw.to_numpy(dtype=np.float32, copy=False)
    prefill_complete_rows = int(np.isfinite(raw_values).all(axis=1).sum())
    prefill_nonfinite_values = int((~np.isfinite(raw_values)).sum())
    transformed_inputs = raw.replace([np.inf, -np.inf], np.nan).copy()
    fill_map = {
        str(name): np.float32(value)
        for name, value in dict(state.get("cycle_input_fill_values", {}) or {}).items()
    }
    for name in inputs:
        transformed_inputs[name] = transformed_inputs[name].fillna(
            fill_map.get(name, np.float32(0.0))
        )
    transformed_inputs = transformed_inputs.astype(np.float32, copy=False)
    postfill_finite = bool(
        np.isfinite(transformed_inputs.to_numpy(dtype=np.float32, copy=False)).all()
    )
    if not postfill_finite:
        errors.append("nonfinite_static_raw_inputs_after_frozen_fill")
    if not missing_inputs and postfill_finite:
        generated = transform_ae_gmm_features(transformed_inputs, state).reindex(columns=outputs)
        sidecar = _read_sidecar_sample(sidecar_path, keys, outputs)
        sidecar_rows = int(len(sidecar))
        if sidecar.duplicated(list(KEY_COLUMNS), keep=False).any():
            errors.append("duplicate_sidecar_keys_in_sample")
        merged = keys.merge(
            sidecar,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
            suffixes=("", "_sidecar"),
        )
        for name in outputs:
            actual = pd.to_numeric(generated[name], errors="coerce").to_numpy(dtype=np.float64)
            expected = pd.to_numeric(merged[name], errors="coerce").to_numpy(dtype=np.float64)
            paired = np.isfinite(actual) & np.isfinite(expected)
            abs_diff = np.abs(actual[paired] - expected[paired])
            missing_rows = int(len(actual) - int(paired.sum()))
            mismatch_rows = int(np.sum(abs_diff > tolerance)) + missing_rows
            output_report[name] = {
                "paired_rows": int(paired.sum()),
                "missing_or_nonfinite_rows": missing_rows,
                "max_abs_diff": float(abs_diff.max()) if abs_diff.size else None,
                "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else None,
                "mismatch_rows": mismatch_rows,
                "within_tolerance": bool(mismatch_rows == 0),
            }
            if mismatch_rows:
                errors.append(f"output_mismatch:{name}")

    manifest = sidecar_contract["manifest"]
    state_sha256 = _sha256_file(state_path)
    manifest_state_sha256 = manifest.get("state_sha256")
    if manifest_state_sha256 is not None and str(manifest_state_sha256) != state_sha256:
        errors.append("sidecar_manifest_state_sha256_mismatch")
    report = {
        "schema": "base_frozen_aegmm_static_store_parity_v1",
        "pass": not errors,
        "errors": errors,
        "tolerance": float(tolerance),
        "labels_path": str(labels_path),
        "feature_store_path": str(feature_store_path),
        "state_path": str(state_path),
        "sidecar_path": str(sidecar_path),
        "sampling": sampling,
        "state_hashes": {
            "state_artifact_sha256": state_sha256,
            "cycle_state_hash_expected": expected_state_hash or None,
            "cycle_state_hash_actual": actual_state_hash,
            "input_feature_order_hash_expected": expected_input_hash or None,
            "input_feature_order_hash_actual": actual_input_hash,
            "input_feature_order_hash_matches": expected_input_hash in {"", actual_input_hash},
            "cycle_state_hash_matches": expected_state_hash in {"", actual_state_hash},
            "sidecar_manifest_state_sha256": manifest_state_sha256,
            "sidecar_manifest_state_sha256_matches": (
                None if manifest_state_sha256 is None else str(manifest_state_sha256) == state_sha256
            ),
        },
        "static_input_loader": loader_report,
        "raw_input_feature_order": inputs,
        "raw_input_availability": availability,
        "missing_or_nonfinite_raw_inputs": missing_inputs,
        "row_eligibility": {
            "contract": "frozen_cycle_input_fill_values_then_all_finite_v1",
            "sample_rows": int(len(raw)),
            "prefill_complete_rows": prefill_complete_rows,
            "prefill_nonfinite_values": prefill_nonfinite_values,
            "postfill_all_finite": postfill_finite,
            "rows_scored": int(len(raw)) if postfill_finite else 0,
        },
        "sidecar": {
            "sample_rows_returned": sidecar_rows,
            "output_features": outputs,
            "manifest_path": sidecar_contract["manifest_path"],
        },
        "generated_output_differences": output_report,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = verify_base_frozen_aegmm_static_store_parity(
            labels_path=args.labels,
            feature_store_path=args.feature_store,
            state_path=args.state,
            sidecar_path=args.sidecar,
            sample_rows=args.sample_rows,
            tolerance=args.tolerance,
        )
    except Exception as exc:
        report = {
            "schema": "base_frozen_aegmm_static_store_parity_v1",
            "pass": False,
            "errors": [f"verification_error:{type(exc).__name__}:{exc}"],
        }
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
