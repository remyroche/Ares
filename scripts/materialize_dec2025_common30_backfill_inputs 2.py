#!/usr/bin/env python3
"""Seal December-only common30 exact-1m backfill inputs.

This emits no label and reads no target value.  It merely derives the exact
December candidate/context/path input triplet and the compatible unfinished
1m path request ledger from their immutable November--January sources.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/nov2025_jan2026_execution_ev_common30_policy_inputs_20260727_v1"
REQUESTS = ROOT / "data_perp/artifacts/nov2025_jan2026_common30_exact1m_stage_20260730_v1/download_candidates.parquet"
OUT = ROOT / "data_perp/artifacts/dec2025_common30_exact1m_backfill_inputs_20260730_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
MONTH = "2025-12"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _month(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    return frame.loc[frame["__ts__"].dt.strftime("%Y-%m").eq(MONTH)].copy()


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    frames = {
        name: _month(pd.read_parquet(SOURCE / f"{name}.parquet"))
        for name in ("candidates", "context", "path_targets")
    }
    hashes = {name: _sha256(SOURCE / f"{name}.parquet") for name in frames}
    candidates = frames["candidates"]
    if len(candidates) != 44_640 or candidates.duplicated(list(IDENTITY)).any():
        raise ValueError("December common30 candidate identity is incomplete or duplicated")
    expected = candidates.loc[:, list(IDENTITY)].sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    for name, frame in frames.items():
        if frame.duplicated(list(IDENTITY)).any():
            raise ValueError(f"{name} has duplicate December identities")
        actual = frame.loc[:, list(IDENTITY)].sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
        if not expected.equals(actual):
            raise ValueError(f"{name} does not exactly match December candidate identity")
    requests = _month(pd.read_parquet(REQUESTS))
    if requests.empty or requests.duplicated("candidate_id").any() or not requests.candidate_id.isin(candidates.candidate_id).all():
        raise ValueError("December exact-1m repair requests are invalid")
    request_ts = pd.to_datetime(requests["timestamp"], utc=True, errors="raise")
    if not request_ts.eq(requests["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("repair request timestamp is not the candidate decision timestamp")
    output.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    requests.to_parquet(output / "download_candidates.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "dec2025_common30_exact1m_backfill_inputs_v1",
        "status": "SEALED_DECEMBER_COMMON30_EXACT1M_REPAIR_INPUTS",
        "scope": "December 2025 only; 1h candidate identity and nested exact 1m path-request ledger",
        "model_sample_cadence": "1h",
        "exact_replay_bar_cadence": "1m_labels_only",
        "candidate_rows": int(len(candidates)),
        "by_side": candidates["side_name"].value_counts().sort_index().to_dict(),
        "symbols": int(candidates["__symbol__"].nunique()),
        "unfinished_candidate_rows": int(len(requests)),
        "unfinished_unique_symbol_decision_windows": int(requests[["timestamp", "symbol"]].drop_duplicates().shape[0]),
        "request_timestamp_contract": "timestamp = hourly signal __ts__ + 1h decision; fetch [decision, decision + 720m)",
        "no_execution_outcome_evaluated": True,
        "inputs_sha256": {**hashes, "unfinished_request_source": _sha256(REQUESTS)},
        "outputs_sha256": {name: _sha256(output / name) for name in ("candidates.parquet", "context.parquet", "path_targets.parquet", "download_candidates.parquet")},
    }
    _write(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(f"{_sha256(output / 'manifest.json')}  manifest.json\n")
    return output


if __name__ == "__main__":
    print(run())
