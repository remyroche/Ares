#!/usr/bin/env python3
"""Create an immutable point-in-time cut of a Strict-R3 source-panel state.

The output contains no data at or after ``--end-exclusive``.  It is intended
for cold-bootstrap parity tests: a feature value at a historic decision must
match whether later market bars are present in the parent panel or not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-state", type=Path, required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    end = _utc(args.end_exclusive)
    source = joblib.load(args.input_state)
    if source.get("schema") != "strict_r3_causal_feature_panel_state_v1":
        raise ValueError("unsupported source-panel state schema")
    panel = source.get("panel")
    if not isinstance(panel, dict):
        raise ValueError("source-panel state lacks a panel dictionary")

    cut_panel: dict[str, object] = {}
    audit: list[dict[str, object]] = []
    for name, values in panel.items():
        if isinstance(values, pd.DataFrame) and isinstance(values.index, pd.DatetimeIndex):
            index = pd.to_datetime(values.index, utc=True)
            kept = values.loc[index < end].copy()
            if kept.empty:
                raise ValueError(f"panel field {name} has no rows before cut")
            max_ts = pd.to_datetime(kept.index, utc=True).max()
            if max_ts >= end:
                raise AssertionError(f"panel field {name} retained future data")
            cut_panel[name] = kept
            audit.append({"field": name, "rows": int(len(kept)), "max_ts": max_ts.isoformat()})
        else:
            cut_panel[name] = values
    output = dict(source)
    output["panel"] = cut_panel
    output["end_exclusive"] = end.isoformat()
    args.out_dir.mkdir(parents=True)
    state_path = args.out_dir / "feature_panel_state.joblib"
    joblib.dump(output, state_path, compress=3)
    pd.DataFrame(audit).to_parquet(args.out_dir / "panel_cut_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_feature_panel_point_in_time_cut_v1",
        "input_state": str(args.input_state),
        "input_state_sha256": _sha(args.input_state),
        "end_exclusive": end.isoformat(),
        "panel_fields": int(len(audit)),
        "state_path": str(state_path),
        "state_sha256": _sha(state_path),
        "future_rows_retained": 0,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
