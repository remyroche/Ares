#!/usr/bin/env python3
"""Materialise exact-1m post-cost barrier-first H12 labels.

The target is deliberately independent of realised exit cost.  It asks whether
the side-relative price path clears a fixed, versioned gross-cost floor before
the frozen adverse barrier.  Same-minute dual hits are ambiguous with OHLC
data and are conservatively classified as adverse/conflict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_historical_exact_h12_alignment_sidecar import (
    COST_MODEL_ID,
    EXECUTION_POLICY_ID,
)


ALIGNMENT = ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
PATHS = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_paths_20260730_v1/paths.parquet",
    ROOT / "data_perp/artifacts/failure_2024_exact1m_paths_20260730_v2/paths.parquet",
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_events_20260731_v1"
FIXED_COST_BPS = 100.0
HURDLES_BPS = (0.0, 25.0)
HORIZON_MINUTES = 12 * 60
TARGET_ID = "exact_1m_h12_postcost_barrier_first_fixed100bps_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _first_index(mask: np.ndarray) -> int | None:
    found = np.flatnonzero(mask)
    return int(found[0]) if len(found) else None


def classify_postcost_path(
    *,
    high: np.ndarray,
    low: np.ndarray,
    entry_price: float,
    side: str,
    adverse_barrier_pct: float,
    cost_bps: float,
    hurdle_bps: float,
) -> tuple[str, int | None, int | None, int | None]:
    """Return event plus first favorable/adverse minute indices.

    The price path is observed minute-by-minute, but intraminute high/low order
    is unavailable.  An equal first index is therefore an explicit conflict.
    """
    if side not in {"long", "short"}:
        raise ValueError(f"unsupported side: {side}")
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        raise ValueError("entry price must be finite and positive")
    if len(high) != HORIZON_MINUTES or len(low) != HORIZON_MINUTES:
        raise ValueError("exact path must have exactly 720 one-minute bars")
    target_return = (float(cost_bps) + float(hurdle_bps)) / 10_000.0
    if side == "long":
        favorable = high >= entry_price * (1.0 + target_return)
        adverse = low <= entry_price * (1.0 - float(adverse_barrier_pct))
    else:
        favorable = low <= entry_price * (1.0 - target_return)
        adverse = high >= entry_price * (1.0 + float(adverse_barrier_pct))
    favorable_index, adverse_index = _first_index(favorable), _first_index(adverse)
    if favorable_index is None and adverse_index is None:
        return "timeout", None, None, None
    if favorable_index is not None and adverse_index is not None:
        if favorable_index == adverse_index:
            return "adverse_first_or_conflict", favorable_index, adverse_index, favorable_index
        if favorable_index < adverse_index:
            return "clear_cost_first", favorable_index, adverse_index, favorable_index
        return "adverse_first_or_conflict", favorable_index, adverse_index, adverse_index
    if favorable_index is not None:
        return "clear_cost_first", favorable_index, None, favorable_index
    return "adverse_first_or_conflict", None, adverse_index, adverse_index


def _alignment(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[
        "candidate_id", "side", "decision_ts", "entry_ts", "label_end_ts",
        "label_available_ts", "execution_entry_price", "barrier_pct",
        "target_id", "execution_policy_id", "cost_model_id",
    ])
    if frame.candidate_id.duplicated().any():
        raise ValueError("alignment candidate IDs are not unique")
    if frame.target_id.nunique() != 1 or frame.execution_policy_id.nunique() != 1 or frame.cost_model_id.nunique() != 1:
        raise ValueError("alignment contract IDs are not unique")
    if frame.execution_policy_id.iloc[0] != EXECUTION_POLICY_ID or frame.cost_model_id.iloc[0] != COST_MODEL_ID:
        raise ValueError("alignment policy/cost contract differs from frozen policy")
    for column in ("decision_ts", "entry_ts", "label_end_ts", "label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame.entry_ts.eq(frame.decision_ts).all() or not frame.label_end_ts.eq(frame.decision_ts + pd.Timedelta(hours=12)).all():
        raise ValueError("alignment does not describe decision-time exact-H12 candidates")
    return frame.set_index("candidate_id", verify_integrity=True)


def _parse_path(raw: str, decision_ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(raw)
    timestamps = np.asarray(payload["timestamp"], dtype=np.int64)
    high = np.asarray(payload["high"], dtype=np.float64)
    low = np.asarray(payload["low"], dtype=np.float64)
    if len(timestamps) != HORIZON_MINUTES or len(high) != HORIZON_MINUTES or len(low) != HORIZON_MINUTES:
        raise ValueError("path payload does not provide 720 exact 1m bars")
    expected = decision_ts.value + np.arange(HORIZON_MINUTES, dtype=np.int64) * pd.Timedelta(minutes=1).value
    if not np.array_equal(timestamps, expected):
        raise ValueError("path timestamps are not an exact one-minute decision-aligned H12 path")
    if not np.isfinite(high).all() or not np.isfinite(low).all() or (high < low).any():
        raise ValueError("path OHLC extrema are invalid")
    return high, low


def _records_from_batch(batch: pa.RecordBatch, alignment: pd.DataFrame) -> list[dict[str, Any]]:
    source = batch.to_pydict()
    records: list[dict[str, Any]] = []
    for candidate_id, raw in zip(source["candidate_id"], source["execution_future_path"], strict=True):
        if candidate_id not in alignment.index:
            continue
        candidate = alignment.loc[candidate_id]
        high, low = _parse_path(raw, candidate.decision_ts)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "side": candidate.side,
            "decision_ts": candidate.decision_ts,
            "label_end_ts": candidate.label_end_ts,
            "label_available_ts": candidate.label_available_ts,
            "postcost_target_id": TARGET_ID,
            "fixed_cost_bps": FIXED_COST_BPS,
            "adverse_barrier_pct": float(candidate.barrier_pct),
        }
        for hurdle in HURDLES_BPS:
            token = f"h{int(hurdle)}"
            event, favorable_minute, adverse_minute, resolved_minute = classify_postcost_path(
                high=high, low=low, entry_price=float(candidate.execution_entry_price),
                side=str(candidate.side), adverse_barrier_pct=float(candidate.barrier_pct),
                cost_bps=FIXED_COST_BPS, hurdle_bps=hurdle,
            )
            row[f"postcost_{token}_event"] = event
            row[f"postcost_{token}_favorable_minute"] = favorable_minute
            row[f"postcost_{token}_adverse_minute"] = adverse_minute
            row[f"postcost_{token}_resolved_minute"] = resolved_minute
        records.append(row)
    return records


def run(*, alignment_path: Path, path_files: tuple[Path, ...], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    alignment = _alignment(alignment_path)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    writer: pq.ParquetWriter | None = None
    seen: set[str] = set()
    try:
        label_path = stage / "postcost_events.parquet"
        for path in path_files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=["candidate_id", "execution_future_path"]):
                rows = _records_from_batch(batch, alignment)
                if not rows:
                    continue
                frame = pd.DataFrame.from_records(rows)
                if frame.candidate_id.duplicated().any() or any(value in seen for value in frame.candidate_id):
                    raise ValueError("duplicate candidate path across source files")
                seen.update(frame.candidate_id)
                for token in ("h0", "h25"):
                    for suffix in ("favorable_minute", "adverse_minute", "resolved_minute"):
                        frame[f"postcost_{token}_{suffix}"] = pd.to_numeric(
                            frame[f"postcost_{token}_{suffix}"], errors="coerce"
                        ).astype("float64")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(label_path, table.schema, compression="zstd")
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
        if writer is None:
            raise ValueError("no aligned paths found")
        writer.close()
        writer = None
        if seen != set(alignment.index):
            missing = len(set(alignment.index).difference(seen))
            raise ValueError(f"exact path coverage incomplete: missing {missing} aligned candidates")
        labels = pd.read_parquet(label_path)
        if len(labels) != len(alignment) or labels.candidate_id.duplicated().any():
            raise ValueError("materialised label identity is not one-to-one")
        for token in ("h0", "h25"):
            allowed = {"clear_cost_first", "adverse_first_or_conflict", "timeout"}
            if not labels[f"postcost_{token}_event"].isin(allowed).all():
                raise ValueError("unexpected post-cost event")
        labels = labels.merge(alignment.reset_index().loc[:, ["candidate_id", "execution_policy_id", "cost_model_id"]], on="candidate_id", how="inner", validate="one_to_one")
        labels.to_parquet(label_path, index=False, compression="zstd")
        support = []
        for token in ("h0", "h25"):
            support.append(labels.groupby(["side", f"postcost_{token}_event"], as_index=False).size().assign(hurdle=token).rename(columns={f"postcost_{token}_event": "event", "size": "rows"}))
        pd.concat(support, ignore_index=True).to_csv(stage / "support_by_side.csv", index=False)
        _write_json(stage / "contract.json", {
            "target_id": TARGET_ID,
            "policy_id": EXECUTION_POLICY_ID,
            "cost_model_id": COST_MODEL_ID,
            "event_definition": "exact 1m first touch of gross fixed-cost+hurdle before frozen adverse barrier; same-minute high/low dual touch is adverse_first_or_conflict",
            "fixed_cost_bps": FIXED_COST_BPS,
            "hurdles_bps": list(HURDLES_BPS),
            "horizon_minutes": HORIZON_MINUTES,
            "label_availability": "decision_ts + 12 hours",
            "not_model_input": True,
        })
        manifest = {
            "schema": "historical_exact_h12_postcost_events_v1",
            "status": "MATERIALIZED_RESEARCH_ONLY_NO_PROMOTION",
            "contract": json.loads((stage / "contract.json").read_text(encoding="utf-8")),
            "rows": len(labels),
            "inputs": {str(path): _sha256(path) for path in (alignment_path, *path_files)},
            "outputs": {name: _sha256(stage / name) for name in ("postcost_events.parquet", "support_by_side.csv", "contract.json")},
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        if writer is not None:
            writer.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--paths", type=Path, nargs="+", default=list(PATHS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(alignment_path=args.alignment, path_files=tuple(args.paths), output=args.output), indent=2))


if __name__ == "__main__":
    main()
