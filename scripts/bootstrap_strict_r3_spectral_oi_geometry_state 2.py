#!/usr/bin/env python3
"""Bootstrap the frozen strict-R3 spectral/OI geometry state from causal parents.

Input NPZ contract (all inputs are decision-time, outcome-free values):

* ``timestamps_ns``: UTC int64 hourly timestamps;
* ``symbols``: ordered Unicode symbol vector;
* ``spectral_source_columns``: ordered frozen source fields;
* ``spectral_source``: [rows, source fields] float32;
* ``oi_parent_columns``: ordered frozen OI parent fields;
* ``oi_parents``: [rows, OI parent fields, symbols] float32;
* ``spectral_definition_id`` and ``oi_geometry_definition_id``: scalar strings.

The bootstrap fails on a non-contiguous source.  It never selects/refits the
geometry contract, and emits a state plus exact current-row outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.spectral_oi_geometry_state import (  # noqa: E402
    SpectralOiGeometryState,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(payload: np.lib.npyio.NpzFile, name: str) -> list[str]:
    if name not in payload:
        raise KeyError(f"input is missing {name}")
    return [str(value) for value in np.asarray(payload[name]).reshape(-1)]


def _scalar(payload: np.lib.npyio.NpzFile, name: str) -> str:
    if name not in payload:
        raise KeyError(f"input is missing {name}")
    return str(np.asarray(payload[name]).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--min-periods", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--shrinkage", type=float, default=0.10)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable bootstrap output already exists: {args.out_dir}")
    started = time.perf_counter()
    with np.load(args.input_npz, allow_pickle=False) as payload:
        timestamps = pd.to_datetime(np.asarray(payload["timestamps_ns"], dtype=np.int64), utc=True)
        symbols = _strings(payload, "symbols")
        spectral_columns = _strings(payload, "spectral_source_columns")
        oi_columns = _strings(payload, "oi_parent_columns")
        source = np.asarray(payload["spectral_source"], dtype=np.float32)
        oi = np.asarray(payload["oi_parents"], dtype=np.float32)
        spectral_definition_id = _scalar(payload, "spectral_definition_id")
        oi_definition_id = _scalar(payload, "oi_geometry_definition_id")
    state = SpectralOiGeometryState(
        symbols=symbols,
        spectral_source_columns=spectral_columns,
        oi_parent_columns=oi_columns,
        spectral_definition_id=spectral_definition_id,
        oi_geometry_definition_id=oi_definition_id,
        lookback=args.lookback,
        min_periods=args.min_periods,
        top_k=args.top_k,
        shrinkage=args.shrinkage,
    )
    output = state.bootstrap(timestamps=timestamps, spectral_source=source, oi_parents=oi)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    state_path = args.out_dir / "spectral_oi_geometry_state.npz"
    state.save(state_path)
    rows = []
    for name, values in output.items():
        rows.extend(
            {
                "timestamp": timestamps[-1],
                "symbol": symbol,
                "feature": name,
                "value": float(values[-1, pos]),
            }
            for pos, symbol in enumerate(symbols)
        )
    pd.DataFrame(rows).to_parquet(args.out_dir / "latest_pretransform_state_outputs.parquet", index=False)
    manifest = {
        "schema": "strict_r3_spectral_oi_geometry_bootstrap_v1",
        "input_npz": str(args.input_npz),
        "input_sha256": _sha256(args.input_npz),
        "history_start": timestamps[0].isoformat(),
        "through": timestamps[-1].isoformat(),
        "rows": int(len(timestamps)),
        "symbols": int(len(symbols)),
        "output_features": list(SpectralOiGeometryState.OUTPUTS),
        "state_contract_hash": state.contract_hash,
        "state_sha256": _sha256(state_path),
        "spectral_definition_id": spectral_definition_id,
        "oi_geometry_definition_id": oi_definition_id,
        "outcome_columns_consumed": [],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
