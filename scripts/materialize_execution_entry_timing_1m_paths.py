#!/usr/bin/env python3
"""Stage and materialize signed one-minute paths for entry-timing labels.

The ``stage`` subcommand emits only downloader-compatible candidate windows.
The ``materialize`` subcommand reads the immutable canonical Kraken Futures
``execution_1m`` store and emits the train-only path artifact consumed by
``materialize_execution_entry_timing_handoff.py``.  It never downloads, fills,
or rewrites market data.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    canonical_kraken_execution_1m_root,
)


SCHEMA = "execution_entry_timing_1m_paths_v1"
PREDICTION_ROLE = "execution_entry_timing_1m_paths"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
PATH_COLUMNS = ("open", "high", "low", "close")
HORIZON_MINUTES = 12 * 60


@dataclass(frozen=True)
class ColumnMapping:
    timestamp: str
    symbol: str
    side: str
    candidate_id: str
    decision: str
    atr: str | None
    atr_fraction: str | None
    fee: str
    entry_spread: str
    exit_spread: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{source}: {column!r} contains null or invalid UTC timestamps")
    return result


def _strings(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = values.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{source}: {column!r} contains null or blank values")
    return result.astype(str)


def _sides(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = _strings(values, source=source, column=column).str.lower()
    if not result.isin(("long", "short")).all():
        raise ValueError(f"{source}: {column!r} must contain canonical long/short values")
    return result


def _nonnegative(values: pd.Series, *, source: str, column: str, upper: float | None = None) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    array = result.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{source}: {column!r} must be finite and non-negative")
    if upper is not None and (array > upper).any():
        raise ValueError(f"{source}: {column!r} exceeds the allowed maximum {upper}")
    return result.astype(np.float32)


def _required_columns(mapping: ColumnMapping) -> list[str]:
    return list(
        dict.fromkeys(value for value in mapping.__dict__.values() if value is not None)
    )


def _iter_input_batches(path: Path, columns: Sequence[str], batch_rows: int) -> Iterator[pd.DataFrame]:
    if not path.is_file() or path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("--input must be an existing parquet artifact")
    schema = pq.read_schema(path)
    missing = sorted(set(columns).difference(schema.names))
    if missing:
        raise ValueError("input is missing required mapped columns: " + ", ".join(missing))
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=int(batch_rows), columns=list(columns)):
        yield batch.to_pandas()


def _canonical_batch(
    raw: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    source: str,
    decision_delay_minutes: int,
) -> pd.DataFrame:
    if (mapping.atr is None) == (mapping.atr_fraction is None):
        raise ValueError("configure exactly one of --atr-col or --atr-fraction-col")
    values: dict[str, pd.Series] = {
        "__ts__": _utc(raw[mapping.timestamp], source=source, column=mapping.timestamp),
        "__symbol__": _strings(raw[mapping.symbol], source=source, column=mapping.symbol),
        "side_name": _sides(raw[mapping.side], source=source, column=mapping.side),
        "candidate_id": _strings(raw[mapping.candidate_id], source=source, column=mapping.candidate_id),
        "__decision_ts__": _utc(raw[mapping.decision], source=source, column=mapping.decision),
        "fee": _nonnegative(raw[mapping.fee], source=source, column=mapping.fee, upper=1.0),
        "entry_spread": _nonnegative(raw[mapping.entry_spread], source=source, column=mapping.entry_spread),
        "exit_spread": _nonnegative(raw[mapping.exit_spread], source=source, column=mapping.exit_spread),
    }
    if mapping.atr is not None:
        values["atr_1h"] = _nonnegative(
            raw[mapping.atr], source=source, column=mapping.atr
        )
    else:
        assert mapping.atr_fraction is not None
        values["atr_fraction"] = _nonnegative(
            raw[mapping.atr_fraction],
            source=source,
            column=mapping.atr_fraction,
            upper=1.0,
        )
    output = pd.DataFrame(values)
    atr_column = "atr_1h" if mapping.atr is not None else "atr_fraction"
    if (output[atr_column] <= 0.0).any():
        raise ValueError(f"input {atr_column} must be strictly positive for every candidate")
    expected = output["__ts__"] + pd.Timedelta(minutes=int(decision_delay_minutes))
    if not output["__decision_ts__"].eq(expected).all():
        raise ValueError(
            "input decision timestamp must equal signal timestamp + "
            f"{int(decision_delay_minutes)} minutes"
        )
    return output


def _identity_key(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        pd.Timestamp(row["__ts__"]).isoformat(),
        str(row["__symbol__"]),
        str(row["side_name"]),
        str(row["candidate_id"]),
    )


def _candidate_batches(
    path: Path,
    mapping: ColumnMapping,
    *,
    batch_rows: int,
    decision_delay_minutes: int,
) -> Iterator[pd.DataFrame]:
    seen: set[tuple[str, str, str, str]] = set()
    for raw in _iter_input_batches(path, _required_columns(mapping), batch_rows):
        frame = _canonical_batch(
            raw, mapping, source="input", decision_delay_minutes=decision_delay_minutes
        )
        keys = [_identity_key(row) for _, row in frame.iterrows()]
        duplicate = len(keys) != len(set(keys)) or any(key in seen for key in keys)
        if duplicate:
            raise ValueError(f"input has duplicate exact identity on {list(IDENTITY)!r}")
        seen.update(keys)
        yield frame


def _candidate_frame(
    path: Path,
    mapping: ColumnMapping,
    *,
    batch_rows: int,
    decision_delay_minutes: int,
) -> pd.DataFrame:
    parts = list(
        _candidate_batches(
            path,
            mapping,
            batch_rows=batch_rows,
            decision_delay_minutes=decision_delay_minutes,
        )
    )
    if not parts:
        raise ValueError("input has no rows")
    return pd.concat(parts, ignore_index=True)


def _symbol_dir(root: Path, symbol: str) -> Path:
    return root / "ohlcv" / f"symbol={symbol.replace('/', '_')}"


def _store_parts(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    years = range(int(start.year), int((end - pd.Timedelta(nanoseconds=1)).year) + 1)
    directory = _symbol_dir(root, symbol)
    return [
        path
        for year in years
        for path in sorted((directory / f"year={year}").glob("*.parquet"))
    ]


def _load_symbol_bars(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[Path]]:
    parts = _store_parts(root, symbol, start, end)
    if not parts:
        return pd.DataFrame(columns=list(PATH_COLUMNS)), parts
    frames: list[pd.DataFrame] = []
    for path in parts:
        frame = pd.read_parquet(path, columns=["ts", *PATH_COLUMNS])
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        frame = frame.loc[(frame["ts"] >= start) & (frame["ts"] < end)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(PATH_COLUMNS)), parts
    output = pd.concat(frames, ignore_index=True)
    output = output.dropna(subset=["ts"]).drop_duplicates("ts", keep="last")
    output = output.sort_values("ts", kind="stable").set_index("ts")
    for column in PATH_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(np.float32)
    return output, parts


def _path_json(bars: pd.DataFrame) -> str:
    """Vector form is smaller than 720 dicts and is decoded by the timing model."""
    payload = {
        "timestamp": [int(value) for value in bars.index.astype("int64")],
        **{
            column: [float(value) for value in bars[column].to_numpy(dtype=np.float32)]
            for column in PATH_COLUMNS
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def _window_path(
    bars: pd.DataFrame, decision: pd.Timestamp
) -> tuple[str | None, str | None, float | None]:
    expected = pd.date_range(
        decision, periods=HORIZON_MINUTES, freq="min", tz="UTC"
    )
    window = bars.reindex(expected)
    if window.isna().any().any():
        missing = int(window.isna().any(axis=1).sum())
        return None, f"missing_or_nonfinite_minutes={missing}", None
    values = window.loc[:, list(PATH_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        return None, "invalid_nonpositive_or_nonfinite_ohlc", None
    if (window["high"] < window["low"]).any():
        return None, "high_below_low", None
    return _path_json(window), None, float(window["open"].iloc[0])


def _manifest_target(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("--execution-ev-target-manifest must be readable JSON") from exc
    schema = payload.get("schema")
    if schema not in {
        "execution_ev_12h_hourly_policy_labels_v2",
        "execution_ev_deployed_policy_1m_labels_v1",
    }:
        raise ValueError("execution-EV target manifest uses an unsupported signed schema")
    if payload.get("prediction_role") != "execution_ev_12h_labels":
        raise ValueError("execution-EV target manifest has the wrong prediction role")
    signed = payload.get("prediction_role_manifest_sha256")
    if not isinstance(signed, str) or not hmac.compare_digest(signed, _manifest_hash(payload)):
        raise ValueError("execution-EV target manifest signature does not verify")
    if schema == "execution_ev_deployed_policy_1m_labels_v1":
        timing = payload.get("timing") or {}
        exit_contract = payload.get("exit_policy_contract") or {}
        lineage = payload.get("historical_lineage") or {}
        store = payload.get("store") or {}
        if (
            int(timing.get("signal_to_decision_minutes", -1)) != 60
            or int(timing.get("horizon_minutes", -1)) != HORIZON_MINUTES
            or int(exit_contract.get("horizon_minutes", -1)) != HORIZON_MINUTES
            or timing.get("label_available_at") != "decision + full replay horizon"
        ):
            raise ValueError("deployed-policy target has incompatible 1m timing")
        if store.get("contract") != "canonical_kraken_execution_1m_immutable_read_only_v1":
            raise ValueError("deployed-policy target lacks immutable-store coverage")
        if (
            lineage.get("oof_status") != "not_oof"
            or bool(lineage.get("execution_parity_claim"))
            or bool(lineage.get("promotion_eligible"))
            or lineage.get("economics")
            not in {
                "current_frozen_spread_counterfactual",
                "inverse_quote_notional_current_spread_counterfactual",
            }
        ):
            raise ValueError(
                "historical deployed-policy target lacks an allowed "
                "counterfactual lineage"
            )
    return _sha256(path), signed


def _coverage_add(
    by_month: dict[str, dict[str, int]],
    by_symbol: dict[str, dict[str, int]],
    row: Mapping[str, Any],
) -> None:
    month = pd.Timestamp(row["__ts__"]).strftime("%Y-%m")
    symbol = str(row["__symbol__"])
    for bucket, key in ((by_month, month), (by_symbol, symbol)):
        stats = bucket.setdefault(key, {"requested": 0, "complete": 0})
        stats["requested"] += 1
        stats["complete"] += int(bool(row["complete"]))


def _coverage_finalize(
    by_month: dict[str, dict[str, int]], by_symbol: dict[str, dict[str, int]]
) -> dict[str, Any]:
    for bucket in (by_month, by_symbol):
        for stats in bucket.values():
            stats["incomplete"] = stats["requested"] - stats["complete"]
            stats["coverage"] = stats["complete"] / max(stats["requested"], 1)
    return {"by_month": by_month, "by_symbol": by_symbol}


def _write_records(
    writer: pq.ParquetWriter | None,
    temporary: Path,
    records: list[dict[str, Any]],
) -> pq.ParquetWriter | None:
    if not records:
        return writer
    table = pa.Table.from_pandas(pd.DataFrame(records), preserve_index=False)
    if writer is None:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def stage(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output
    if output.exists():
        raise ValueError("refusing to overwrite existing staging output")
    mapping = _mapping(args)
    frame = _candidate_frame(
        args.input,
        mapping,
        batch_rows=args.candidate_batch_rows,
        decision_delay_minutes=args.decision_delay_minutes,
    )
    staged = pd.DataFrame(
        {
            "timestamp": frame["__decision_ts__"],
            "symbol": frame["__symbol__"],
            "candidate_id": frame["candidate_id"],
            "__ts__": frame["__ts__"],
            "side_name": frame["side_name"],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staged.to_parquet(output, index=False, compression="zstd")
    manifest = output.with_suffix(".manifest.json")
    _write_json(
        manifest,
        {
            "schema": "execution_entry_timing_download_stage_v1",
            "source": {"path": str(args.input), "sha256": _sha256(args.input)},
            "output": {"path": str(output), "sha256": _sha256(output)},
            "rows": int(len(staged)),
            "window": {"timestamp_column": "timestamp", "horizon_minutes": HORIZON_MINUTES},
            "identity": list(IDENTITY),
        },
    )
    return {"staging": output, "manifest": manifest}


def materialize(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output
    manifest_path = args.manifest or output.with_suffix(".manifest.json")
    missing_path = args.missing_report or output.with_suffix(".missing.json")
    if output.exists() or manifest_path.exists() or missing_path.exists():
        raise ValueError("refusing to overwrite timing paths, manifest, or missing-window report")
    target_hash, target_signed_hash = _manifest_target(args.execution_ev_target_manifest)
    mapping = _mapping(args)
    root = canonical_kraken_execution_1m_root(args.data_root)
    if not root.is_dir() or not (root / "ohlcv").is_dir():
        raise ValueError(f"canonical execution_1m store does not exist: {root}")
    completed_through = pd.Timestamp(args.completed_through_utc or pd.Timestamp.now(tz="UTC"))
    completed_through = completed_through.tz_localize("UTC") if completed_through.tzinfo is None else completed_through.tz_convert("UTC")
    completed_through = completed_through.floor("min")
    source_parts: dict[str, str] = {}
    by_month: dict[str, dict[str, int]] = {}
    by_symbol: dict[str, dict[str, int]] = {}
    incomplete: list[dict[str, Any]] = []
    requested_rows = 0
    output_rows = 0
    temporary = output.with_name(output.name + ".partial")
    writer: pq.ParquetWriter | None = None
    try:
        candidates = _candidate_frame(
            args.input,
            mapping,
            batch_rows=args.candidate_batch_rows,
            decision_delay_minutes=args.decision_delay_minutes,
        )
        # Load each symbol's immutable partitions once.  The previous
        # candidate-batch outer loop repeatedly reopened the same yearly parts
        # for every 500 rows and made the full timing run unnecessarily slow.
        for symbol, group in candidates.groupby("__symbol__", sort=True):
            emitted: list[dict[str, Any]] = []
            start = group["__decision_ts__"].min()
            end = group["__decision_ts__"].max() + pd.Timedelta(
                minutes=HORIZON_MINUTES
            )
            bars, parts = _load_symbol_bars(root, str(symbol), start, end)
            for part in parts:
                source_parts[str(part.relative_to(root))] = _sha256(part)
            for _, row in group.iterrows():
                end_time = row["__decision_ts__"] + pd.Timedelta(
                    minutes=HORIZON_MINUTES
                )
                reason: str | None = None
                path: str | None = None
                decision_price: float | None = None
                if end_time > completed_through:
                    reason = "window_not_known_completed"
                else:
                    path, reason, decision_price = _window_path(
                        bars, row["__decision_ts__"]
                    )
                audit = {
                    "__ts__": row["__ts__"],
                    "__symbol__": row["__symbol__"],
                    "side_name": row["side_name"],
                    "candidate_id": row["candidate_id"],
                    "__decision_ts__": row["__decision_ts__"],
                    "complete": path is not None,
                    "reason": reason,
                }
                requested_rows += 1
                _coverage_add(by_month, by_symbol, audit)
                if path is None:
                    incomplete.append(audit)
                else:
                    atr = (
                        float(row["atr_1h"])
                        if "atr_1h" in row.index
                        else float(row["atr_fraction"]) * float(decision_price)
                    )
                    emitted.append(
                        {
                            **{key: row[key] for key in IDENTITY},
                            "execution_future_path": path,
                            "atr_1h": np.float32(atr),
                            "decision_price": np.float32(decision_price),
                            "fee": np.float32(row["fee"]),
                            "entry_spread": np.float32(row["entry_spread"]),
                            "exit_spread": np.float32(row["exit_spread"]),
                        }
                    )
                if len(emitted) >= int(args.candidate_batch_rows):
                    writer = _write_records(writer, temporary, emitted)
                    output_rows += len(emitted)
                    emitted = []
            writer = _write_records(writer, temporary, emitted)
            output_rows += len(emitted)
    finally:
        if writer is not None:
            writer.close()
    coverage = _coverage_finalize(by_month, by_symbol)
    missing_payload = {
        "schema": "execution_entry_timing_1m_missing_windows_v1",
        "rows": int(requested_rows),
        "complete_rows": int(output_rows),
        "incomplete_rows": int(len(incomplete)),
        "coverage": coverage,
        "missing_windows": incomplete,
    }
    _write_json(missing_path, missing_payload)
    if incomplete and not args.allow_subset:
        if temporary.exists():
            temporary.unlink()
        raise ValueError(
            f"{len(incomplete)} candidates lack an exact completed {HORIZON_MINUTES}-minute path; "
            f"see {missing_path} or pass --allow-subset explicitly"
        )
    if not output_rows:
        if temporary.exists():
            temporary.unlink()
        raise ValueError("no complete candidate paths available for materialization")
    temporary.replace(output)
    store_digest = hashlib.sha256(
        json.dumps(source_parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "prediction_role": PREDICTION_ROLE,
        "source_artifact_sha256": _sha256(output),
        "source": {"input": str(args.input), "input_sha256": _sha256(args.input)},
        "execution_ev_target_manifest_sha256": target_hash,
        "execution_ev_target_signed_manifest_sha256": target_signed_hash,
        "store": {
            "root": str(root),
            "contract": "canonical_kraken_execution_1m_read_only_v1",
            "parts_sha256": store_digest,
            "parts": source_parts,
        },
        "identity": list(IDENTITY),
        "timing": {
            "decision_column": "__decision_ts__",
            "first_path_timestamp": "__decision_ts__",
            "cadence_minutes": 1,
            "path_minutes": HORIZON_MINUTES,
            "completed_through_utc": completed_through,
        },
        "path": {
            "column": "execution_future_path",
            "encoding": "json_vector_timestamp_ns_float32_ohlc",
            "fixed_length": HORIZON_MINUTES,
        },
        "atr": {
            "output_column": "atr_1h",
            "input_mode": (
                "absolute"
                if mapping.atr is not None
                else "decision_price_times_atr_fraction"
            ),
            "input_column": mapping.atr or mapping.atr_fraction,
            "decision_price_column": "decision_price",
        },
        "cost_accounting": "fee_once_entry_spread_once_exit_spread_once",
        "cost_columns": {
            "fee": "fee",
            "entry_spread_bps": "entry_spread",
            "exit_spread_bps": "exit_spread",
            "policy": "values are carried without deduction; entry-timing labels apply each component once",
        },
        "coverage": coverage,
        "rows": {"requested": int(requested_rows), "output": int(output_rows), "subset": bool(incomplete)},
        "missing_window_report": {"path": str(missing_path), "sha256": _sha256(missing_path)},
    }
    manifest["prediction_role_manifest_sha256"] = _manifest_hash(manifest)
    _write_json(manifest_path, manifest)
    return {"paths": output, "manifest": manifest_path, "missing_report": missing_path}


def _mapping(args: argparse.Namespace) -> ColumnMapping:
    return ColumnMapping(
        timestamp=args.timestamp_col,
        symbol=args.symbol_col,
        side=args.side_col,
        candidate_id=args.candidate_id_col,
        decision=args.decision_ts_col,
        atr=getattr(args, "atr_col", None),
        atr_fraction=getattr(args, "atr_fraction_col", None),
        fee=args.fee_col,
        entry_spread=args.entry_spread_col,
        exit_spread=args.exit_spread_col,
    )


def _add_mapping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="__ts__")
    parser.add_argument("--symbol-col", default="__symbol__")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument("--candidate-id-col", default="candidate_id")
    parser.add_argument("--decision-ts-col", default="__decision_ts__")
    atr_group = parser.add_mutually_exclusive_group(required=False)
    atr_group.add_argument("--atr-col", default=None)
    atr_group.add_argument("--atr-fraction-col", default=None)
    parser.add_argument("--fee-col", default="fee")
    parser.add_argument("--entry-spread-col", default="entry_spread")
    parser.add_argument("--exit-spread-col", default="exit_spread")
    parser.add_argument("--decision-delay-minutes", type=int, default=60)
    parser.add_argument("--candidate-batch-rows", type=int, default=500)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage", help="Emit downloader candidate staging parquet.")
    _add_mapping_arguments(stage_parser)
    stage_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = commands.add_parser("materialize", help="Read canonical 1m paths and emit signed labels.")
    _add_mapping_arguments(materialize_parser)
    materialize_parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    materialize_parser.add_argument("--execution-ev-target-manifest", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--manifest", type=Path, default=None)
    materialize_parser.add_argument("--missing-report", type=Path, default=None)
    materialize_parser.add_argument("--completed-through-utc", default=None)
    materialize_parser.add_argument("--allow-subset", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.candidate_batch_rows < 1 or args.decision_delay_minutes < 0:
        raise SystemExit("candidate-batch-rows must be positive and decision-delay-minutes non-negative")
    if args.atr_col is None and args.atr_fraction_col is None:
        args.atr_col = "atr_1h"
    try:
        paths = stage(args) if args.command == "stage" else materialize(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"entry-timing 1m path materialization failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
