#!/usr/bin/env python3
"""Materialize the accepted Feb--Apr 2025 path-head targets from exact 1m paths.

This is deliberately a target-only materializer.  It consumes the immutable,
complete 720-minute path artifact and the signed deployed-policy ledger; it
does not load hourly OHLCV, construct proxy paths, select features, or fit a
model.  The CatBoost label is a 12-hour execution-compatible derivative of the
frozen v6 deterministic rules.  The source v6 corpus itself used a 24-hour
path, so this artifact records that non-equivalence explicitly rather than
claiming a false bit-for-bit v6 replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_labels import (  # noqa: E402
    PATH_ARCHETYPE_RULE_VERSION,
    _deterministic_path_archetypes_batch,
    _deterministic_realization_strength_batch,
    _summarize_side_relative_path_batch,
)
from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    TARGET_COLUMNS,
    TARGET_SCHEMA,
    build_path_auxiliary_targets,
)

SCHEMA = "febapr2025_exact1m_execution_path_head_labels_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
HORIZON_MINUTES = 720
HORIZONS_HOURS = (1, 2, 4, 8, 12)


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
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_policy_context(candidates_path: Path, labels_path: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(
        candidates_path,
        columns=[
            *IDENTITY,
            "__decision_ts__",
            "__barrier_pct__",
            "atr_fraction",
            "execution_cost_return",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
        ],
    )
    labels = pd.read_parquet(
        labels_path,
        columns=[
            *IDENTITY,
            "execution_decision_utc",
            "execution_entry_price",
            "execution_cost_return",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
            "execution_geometry_key",
            "policy_archetype",
            "execution_label_end_utc",
            "execution_label_available_at",
        ],
    )
    labels = labels.rename(
        columns={
            "execution_cost_return": "policy_execution_cost_return",
            "execution_entry_half_spread_bps": "policy_entry_half_spread_bps",
            "execution_exit_half_spread_bps": "policy_exit_half_spread_bps",
        }
    )
    merged = candidates.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(merged) != len(candidates) or merged["execution_entry_price"].isna().any():
        raise ValueError("every accepted top40 identity must join exactly once to deployed-policy labels")
    if not pd.to_datetime(merged["__decision_ts__"], utc=True).eq(
        pd.to_datetime(merged["execution_decision_utc"], utc=True)
    ).all():
        raise ValueError("candidate and deployed-policy decision timestamps disagree")
    for candidate_col, policy_col in (
        ("execution_cost_return", "policy_execution_cost_return"),
        ("execution_entry_half_spread_bps", "policy_entry_half_spread_bps"),
        ("execution_exit_half_spread_bps", "policy_exit_half_spread_bps"),
    ):
        lhs = pd.to_numeric(merged[candidate_col], errors="coerce").to_numpy(float)
        rhs = pd.to_numeric(merged[policy_col], errors="coerce").to_numpy(float)
        if not np.allclose(lhs, rhs, rtol=0.0, atol=1e-12, equal_nan=False):
            raise ValueError(f"policy accounting mismatch: {candidate_col} != {policy_col}")
    return merged.set_index("candidate_id", verify_integrity=True)


def _decode_paths(values: Iterable[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parsed = [json.loads(value) for value in values]
    expected = {"timestamp", "open", "high", "low", "close"}
    if any(set(item) != expected for item in parsed):
        raise ValueError("exact path encoding is not the signed fixed OHLC vector")
    arrays = tuple(
        np.asarray([item[column] for item in parsed], dtype=np.float64)
        for column in ("open", "high", "low", "close")
    )
    if any(array.shape[1] != HORIZON_MINUTES for array in arrays):
        raise ValueError("path does not contain exactly 720 one-minute observations")
    timestamps = np.asarray([item["timestamp"] for item in parsed], dtype=np.int64)
    minute_ns = int(pd.Timedelta(minutes=1).value)
    if not np.all(np.diff(timestamps, axis=1) == minute_ns):
        raise ValueError("path cadence is not exactly one minute")
    if not all(np.isfinite(array).all() and (array > 0.0).all() for array in arrays):
        raise ValueError("path contains non-finite or non-positive OHLC values")
    return arrays


def _side_sign(side: pd.Series) -> np.ndarray:
    value = side.astype(str).str.lower().to_numpy()
    if not np.isin(value, ("long", "short")).all():
        raise ValueError("only canonical long/short paths are accepted")
    return np.where(value == "short", -1.0, 1.0)


def _execution_adjusted_path(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    side_sign: np.ndarray,
    entry_spread_bps: np.ndarray,
    exit_spread_bps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply each side's executable entry/exit spread exactly once."""
    sign = side_sign[:, None]
    entry_factor = 1.0 + sign * entry_spread_bps[:, None] / 10_000.0
    exit_factor = 1.0 - sign * exit_spread_bps[:, None] / 10_000.0
    return (
        open_[:, 0] * entry_factor[:, 0],
        high * exit_factor,
        low * exit_factor,
        close * exit_factor,
    )


def _soft_triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    *,
    entry: np.ndarray,
    atr_fraction: np.ndarray,
    side_sign: np.ndarray,
) -> dict[str, np.ndarray]:
    upper = np.maximum(1.5 * atr_fraction, 0.015)
    lower = atr_fraction
    long_up = high / entry[:, None] - 1.0 >= upper[:, None]
    short_up = 1.0 - low / entry[:, None] >= upper[:, None]
    long_down = 1.0 - low / entry[:, None] >= lower[:, None]
    short_down = high / entry[:, None] - 1.0 >= lower[:, None]
    up = np.where(side_sign[:, None] > 0.0, long_up, short_up)
    down = np.where(side_sign[:, None] > 0.0, long_down, short_down)
    has_up, has_down = up.any(axis=1), down.any(axis=1)
    up_i, down_i = np.argmax(up, axis=1), np.argmax(down, axis=1)
    ambiguous = has_up & has_down & (up_i == down_i)
    outcome = np.full(len(entry), "timeout", dtype=object)
    outcome[has_up & (~has_down | (up_i < down_i))] = "favorable_first"
    outcome[has_down & (~has_up | (down_i <= up_i))] = "adverse_first_or_conflict"
    return {
        "__soft_tb_upper_hit_12h__": has_up.astype(np.int8),
        "__soft_tb_lower_hit_12h__": has_down.astype(np.int8),
        "__soft_tb_first_event__": outcome,
        "__soft_tb_order_ambiguous__": ambiguous.astype(np.int8),
    }


def _output_batch(paths: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    ids = paths["candidate_id"].astype(str)
    if ids.duplicated().any() or not ids.isin(context.index).all():
        raise ValueError("path identities do not exactly match the accepted policy context")
    frame = context.loc[ids].reset_index()
    for column in IDENTITY:
        if column == "candidate_id":
            continue
        lhs = paths[column].astype(str) if column in {"__symbol__", "side_name"} else pd.to_datetime(paths[column], utc=True)
        rhs = frame[column].astype(str) if column in {"__symbol__", "side_name"} else pd.to_datetime(frame[column], utc=True)
        if not lhs.reset_index(drop=True).eq(rhs).all():
            raise ValueError(f"path identity mismatch on {column}")
    open_, high, low, close = _decode_paths(paths["execution_future_path"])
    sign = _side_sign(frame["side_name"])
    entry_spread = pd.to_numeric(frame["policy_entry_half_spread_bps"], errors="coerce").to_numpy(float)
    exit_spread = pd.to_numeric(frame["policy_exit_half_spread_bps"], errors="coerce").to_numpy(float)
    fee = pd.to_numeric(frame["policy_execution_cost_return"], errors="coerce").to_numpy(float)
    atr_fraction = pd.to_numeric(frame["atr_fraction"], errors="coerce").to_numpy(float)
    barrier = pd.to_numeric(frame["__barrier_pct__"], errors="coerce").to_numpy(float)
    entry, exec_high, exec_low, exec_close = _execution_adjusted_path(
        open_, high, low, close,
        side_sign=sign, entry_spread_bps=entry_spread, exit_spread_bps=exit_spread,
    )
    risk = entry * barrier
    summary = _summarize_side_relative_path_batch(
        exec_high, exec_low, exec_close,
        entry_price=entry,
        risk_distance=risk,
        atr_fraction=atr_fraction,
        side_sign=sign,
        bar_hours=1.0 / 60.0,
        horizons_hours=HORIZONS_HOURS,
        take_profit_r=np.full(len(frame), np.nan),
        trailing_trigger_r=np.full(len(frame), np.nan),
        stop_r=np.ones(len(frame)),
        cost_return=fee,
        archetype_cost_return=fee,
        activation_distance_return=np.full(len(frame), np.nan),
        prefix="path_arch_",
    )
    shape = _deterministic_path_archetypes_batch(summary, prefix="path_arch_")
    strength = _deterministic_realization_strength_batch(summary, prefix="path_arch_")
    combined = np.full(len(frame), None, dtype=object)
    valid_archetype = pd.notna(shape) & pd.notna(strength)
    combined[valid_archetype] = np.char.add(
        np.char.add(shape[valid_archetype].astype(str), "__"), strength[valid_archetype].astype(str)
    )
    # Auxiliary targets intentionally use the unadjusted exact future path and
    # raw decision open, exactly as their frozen v6 target definition specifies.
    aux = build_path_auxiliary_targets(
        entry_price=open_[:, 0], future_high=high, future_low=low,
        atr_fraction=atr_fraction, side_sign=sign, bar_minutes=1, horizon_hours=12,
        include_supportive_columns=False,
    ).as_columns()
    triple = _soft_triple_barrier(
        high, low, entry=open_[:, 0], atr_fraction=atr_fraction, side_sign=sign,
    )
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True)
    encoded_start = np.asarray(
        [json.loads(value)["timestamp"][0] for value in paths["execution_future_path"]],
        dtype=np.int64,
    )
    if not np.array_equal(encoded_start, decision.astype("int64").to_numpy()):
        raise ValueError("first exact-path minute does not match signed policy decision timestamp")
    label_end = decision + pd.Timedelta(hours=12)
    output: dict[str, Any] = {column: frame[column].to_numpy() for column in IDENTITY}
    output.update(
        {
            "__decision_ts__": decision.to_numpy(),
            "__label_end_ts__": label_end.to_numpy(),
            "__label_available_at__": label_end.to_numpy(),
            "execution_geometry_key": frame["execution_geometry_key"].to_numpy(),
            "policy_archetype": frame["policy_archetype"].to_numpy(),
            "execution_entry_price": entry.astype(np.float32),
            "execution_cost_return": fee.astype(np.float32),
            "execution_entry_half_spread_bps": entry_spread.astype(np.float32),
            "execution_exit_half_spread_bps": exit_spread.astype(np.float32),
            "__barrier_pct__": barrier.astype(np.float32),
            "__path_auxiliary_atr_fraction__": atr_fraction.astype(np.float32),
            "path_archetype_rule_version": PATH_ARCHETYPE_RULE_VERSION,
            "path_arch_complete_12h": np.isfinite(summary["path_arch_mfe_12h_r"]).astype(np.int8),
            "path_shape_archetype": pd.array(shape, dtype="string"),
            "path_realization_strength": pd.array(strength, dtype="string"),
            "path_archetype": pd.array(combined, dtype="string"),
        }
    )
    output.update({column: values.astype(np.float32) for column, values in summary.items()})
    output.update(aux)
    output.update(triple)
    return pd.DataFrame(output)


def _support_report(labels_path: Path, output_dir: Path) -> dict[str, Any]:
    columns = [
        "__ts__", "side_name", "path_arch_complete_12h", "path_archetype",
        "__path_auxiliary_target_valid__", "__meaningful_mfe_reached_12h__",
        "__soft_tb_first_event__", "__soft_tb_order_ambiguous__",
        *TARGET_COLUMNS.values(),
    ]
    frame = pd.read_parquet(labels_path, columns=columns)
    frame["month"] = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for (month, side), group in frame.groupby(["month", "side_name"], sort=True):
        valid = group["__path_auxiliary_target_valid__"].astype(bool)
        completed = group["path_arch_complete_12h"].astype(bool)
        row: dict[str, Any] = {
            "month": month, "side_name": side, "rows": int(len(group)),
            "complete_1m_12h_rows": int(completed.sum()),
            "complete_1m_12h_fraction": float(completed.mean()),
            "auxiliary_valid_rows": int(valid.sum()),
            "meaningful_mfe_positive_rows": int(group.loc[valid, "__meaningful_mfe_reached_12h__"].sum()),
            "meaningful_mfe_positive_fraction": float(group.loc[valid, "__meaningful_mfe_reached_12h__"].mean()),
            "soft_tb_favorable_first_rows": int((group["__soft_tb_first_event__"] == "favorable_first").sum()),
            "soft_tb_adverse_first_or_conflict_rows": int((group["__soft_tb_first_event__"] == "adverse_first_or_conflict").sum()),
            "soft_tb_timeout_rows": int((group["__soft_tb_first_event__"] == "timeout").sum()),
            "soft_tb_order_ambiguous_rows": int(group["__soft_tb_order_ambiguous__"].sum()),
        }
        for name, column in TARGET_COLUMNS.items():
            values = pd.to_numeric(group.loc[valid, column], errors="coerce")
            row[f"{name}_finite_rows"] = int(values.notna().sum())
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
        rows.append(row)
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "coverage_support_by_month_side.csv", index=False)
    archetypes = (
        frame.loc[frame["path_arch_complete_12h"].astype(bool)]
        .groupby(["month", "side_name", "path_archetype"], dropna=False, sort=True)
        .size().rename("rows").reset_index()
    )
    archetypes.to_csv(output_dir / "archetype_class_support_by_month_side.csv", index=False)
    return {
        "coverage_support": report.to_dict(orient="records"),
        "archetype_class_support_rows": int(len(archetypes)),
    }


def materialize(
    *, paths_path: Path, candidates_path: Path, policy_labels_path: Path, output_dir: Path,
    batch_rows: int = 512, source_offset: int = 0, source_limit: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _read_policy_context(candidates_path, policy_labels_path)
    source = pq.ParquetFile(paths_path)
    required = {"execution_future_path", *IDENTITY}
    if not required.issubset(source.schema.names):
        raise ValueError("signed exact-path artifact is missing required identity/path fields")
    writer: pq.ParquetWriter | None = None
    rows = 0
    seen: set[str] = set()
    source_offset = int(source_offset)
    source_limit = None if source_limit is None else int(source_limit)
    if source_offset < 0 or (source_limit is not None and source_limit <= 0):
        raise ValueError("source offset must be non-negative and source limit positive")
    source_row = 0
    labels_path = output_dir / "labels.parquet"
    temporary_labels_path = output_dir / "labels.parquet.partial"
    try:
        for batch in source.iter_batches(batch_size=int(batch_rows), columns=list(required)):
            raw = batch.to_pandas()
            batch_start, batch_end = source_row, source_row + len(raw)
            source_row = batch_end
            if batch_end <= source_offset:
                continue
            if source_limit is not None and batch_start >= source_offset + source_limit:
                break
            lo = max(source_offset - batch_start, 0)
            hi = len(raw) if source_limit is None else min(len(raw), source_offset + source_limit - batch_start)
            raw = raw.iloc[lo:hi].reset_index(drop=True)
            if raw.empty:
                continue
            output = _output_batch(raw, context)
            ids = output["candidate_id"].astype(str)
            if ids.duplicated().any() or any(identifier in seen for identifier in ids):
                raise ValueError("duplicate candidate identity in exact path artifact")
            seen.update(ids)
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_labels_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(output)
    finally:
        if writer is not None:
            writer.close()
    if not temporary_labels_path.exists():
        raise ValueError("no labels were written for requested source slice")
    expected = len(context) - source_offset if source_limit is None else min(source_limit, len(context) - source_offset)
    if rows != expected or len(seen) != expected:
        raise ValueError("materialization did not preserve every requested exact identity")
    # A shard becomes discoverable only after the writer is closed, the footer
    # is readable, and its full requested identity slice has passed checks.
    parquet_rows = pq.ParquetFile(temporary_labels_path).metadata.num_rows
    if parquet_rows != rows:
        raise ValueError("atomic shard parquet row count does not match materialized rows")
    temporary_labels_path.replace(labels_path)
    support = _support_report(labels_path, output_dir)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    manifest = {
        "schema": SCHEMA,
        "prediction_role": "historical_path_head_target_labels",
        "labels": {"path": str(labels_path), "sha256": _sha256(labels_path), "rows": int(rows)},
        "source_slice": {"offset": source_offset, "limit": source_limit, "expected_rows": int(expected)},
        "identity": list(IDENTITY),
        "sources": {
            "exact_1m_paths": {"path": str(paths_path), "sha256": _sha256(paths_path)},
            "accepted_top40_candidates": {"path": str(candidates_path), "sha256": _sha256(candidates_path)},
            "signed_deployed_policy_labels": {"path": str(policy_labels_path), "sha256": _sha256(policy_labels_path)},
        },
        "path_contract": {"cadence_minutes": 1, "path_minutes": HORIZON_MINUTES, "all_rows_complete": True},
        "label_timing": {
            "decision_timestamp": "__decision_ts__", "label_end": "decision + 12h",
            "label_available_at": "decision + 12h",
        },
        "catboost_archetype": {
            "rule_version": PATH_ARCHETYPE_RULE_VERSION,
            "output_column": "path_archetype",
            "side_local_identity": "side_name retained in exact identity",
            "geometry": "deployed execution barrier, side-parent policy geometry key retained",
            "cost_accounting": "entry spread once; exit spread once; per-row deployed fee return once",
            "frozen_v6_compatibility": {
                "rule_constants_and_precedence": "identical",
                "not_bitwise_comparable": "frozen v6 labels used 24h paths; this target is policy-horizon 12h",
            },
        },
        "auxiliary": {
            "schema": TARGET_SCHEMA, "primary_targets": TARGET_COLUMNS,
            "path_semantics": "unadjusted exact 1m decision-open future path; no proxy bars",
        },
        "soft_triple_barrier": {
            "upper": "max(1.5 * ATR_fraction, 1.5% return)", "lower": "1.0 * ATR_fraction",
            "same_minute_conflict": "adverse_first_or_conflict", "ambiguity_column": "__soft_tb_order_ambiguous__",
        },
        "parity": {
            "exact_1m_coverage": "passed: 205194/205194 complete paths",
            "v6_auxiliary_definition": "same build_path_auxiliary_targets v6 kernel at bar_minutes=1",
            "v6_catboost_numeric_parity": "not applicable: frozen source requires 24h paths absent from signed artifact",
        },
        "support_reports": support,
        "code_revision": revision,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--source-offset", type=int, default=0)
    parser.add_argument("--source-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(_json_safe(materialize(
        paths_path=args.paths, candidates_path=args.candidates,
        policy_labels_path=args.policy_labels, output_dir=args.output_dir,
        batch_rows=args.batch_rows, source_offset=args.source_offset, source_limit=args.source_limit,
    )), sort_keys=True))


if __name__ == "__main__":
    main()
