#!/usr/bin/env python3
"""Apply canonical exact-producer Cell-day admission over a held period."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_cell_day_admission import (  # noqa: E402
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
    apply_cell_day_trim15_admission_snapshot,
)
from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    load_strict_r3_ev_bridge,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--immediate-calibration-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    frame = pd.read_parquet(args.scored_label_ledger)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True,
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError("scored label ledger has duplicate candidate IDs")
    index = pd.read_parquet(args.immediate_calibration_index)
    fitted = index.loc[index["status"].eq("fitted_immediate_exact_producer_calibration")]
    if len(fitted) != 1:
        raise ValueError("period admission requires one fitted exact-producer calibrator")
    bundle_path = Path(str(fitted.iloc[0]["ev_bridge_bundle"]))
    if not bundle_path.is_absolute():
        bundle_path = ROOT / bundle_path
    bundle = load_strict_r3_ev_bridge(bundle_path)
    mapped_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for day, current in frame.groupby(frame["__decision_ts__"].dt.normalize(), sort=True):
        mapped, audit = apply_cell_day_trim15_admission_snapshot(
            resolved_score_ledger=frame,
            current_scores=current.copy(),
            bundle=bundle,
        )
        mapped_parts.append(mapped)
        audit_parts.append(audit)
    mapped = pd.concat(mapped_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    audit = pd.concat(audit_parts, ignore_index=True)
    if len(mapped) != len(frame) or set(mapped["candidate_id"]) != set(frame["candidate_id"]):
        raise AssertionError("Cell-day admission changed held candidate identities")
    args.out_dir.mkdir(parents=True)
    mapped.to_parquet(
        args.out_dir / "score_and_cell_day_admission_provenance.parquet",
        index=False, compression="zstd",
    )
    audit.to_parquet(args.out_dir / "cell_day_admission_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_cell_day_trim15_period_admission_v1",
        "mapping": CELL_DAY_TRIM_15_CALIBRATION_MODE,
        "scored_label_ledger": str(args.scored_label_ledger),
        "scored_label_ledger_sha256": _sha(args.scored_label_ledger),
        "immediate_calibration_index": str(args.immediate_calibration_index),
        "immediate_calibration_index_sha256": _sha(args.immediate_calibration_index),
        "ev_bridge_bundle": str(bundle_path),
        "rows": int(len(mapped)),
        "days": int(mapped["__decision_ts__"].dt.normalize().nunique()),
        "admitted_rows": int(mapped["causal_21d_side_admitted_ge_50bps"].sum()),
        "strictly_prior_resolved": bool(audit["strictly_prior_resolved"].all()),
        "held_outcomes_used_for_same_day_admission": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
