#!/usr/bin/env python3
"""Assemble matched A0/A4 folds and causally calibrate A5 over a longer ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_r5_posterior_contract import _load, _sha, prequential_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, action="append", required=True)
    parser.add_argument("--a4", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    a0 = _load(args.a0, "A0_current")
    a4 = _load(args.a4, "A4_independent_local")
    identity = ["candidate_id", "__decision_ts__"]
    if not a0[identity].equals(a4[identity]):
        raise ValueError("A0/A4 candidate identity mismatch")

    output = a0.copy()
    output["a0_current__expected"] = pd.to_numeric(
        a0["posterior_expected_bps"], errors="coerce",
    )
    output["a0_current__admitted"] = output["a0_current__expected"].ge(50.0)
    calibrated, probability, audit = prequential_calibration(a4)
    output["a5_calibrated__expected"] = calibrated
    output["a5_calibrated__p_positive"] = probability
    output["a5_calibrated__admitted"] = output["a5_calibrated__expected"].ge(50.0)

    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "selection_ledger.parquet", index=False)
    audit.to_parquet(args.out_dir / "prequential_calibration.parquet", index=False)
    manifest = {
        "schema": "a5_longer_prequential_calibration_v1",
        "a0": [{"path": str(path), "sha256": _sha(path)} for path in args.a0],
        "a4": [{"path": str(path), "sha256": _sha(path)} for path in args.a4],
        "rows": int(len(output)),
        "start": str(output["__decision_ts__"].min()),
        "end_exclusive": str(output["__decision_ts__"].max() + pd.Timedelta(hours=1)),
        "calibration": "month t uses only earlier OOS rows resolved before month t",
        "warmup_note": (
            "April-September 2025 use expanding post-geometry history; "
            "October 2025 onward has the full nine-month history contract"
        ),
        "winner_promoted": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
