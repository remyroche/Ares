#!/usr/bin/env python3
"""Build canonical bounded-A5 OOS predictions from matched A0/A4 folds.

For held month t, the A5 calibration is fitted only on earlier A4 OOS
predictions whose policy labels resolved before t.  Admission is the fixed
A0>=50 plus timestamp-local pre-trust top-15 population; A5 only reranks it.
"""

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

from extreme_price_movements.strict_r3_a5_trust import (  # noqa: E402
    apply_a5_bounded_10pct,
    fit_a5_calibration,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(paths: list[Path], label: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{label} folds contain duplicate candidate IDs")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, action="append", required=True)
    parser.add_argument("--a4", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable A5 walk-forward exists: {args.out_dir}")
    a0 = _load(args.a0, "A0")
    a4 = _load(args.a4, "A4")
    identity = ["candidate_id", "__decision_ts__"]
    if not a0[identity].equals(a4[identity]):
        raise ValueError("A0/A4 OOS candidate identity mismatch")
    output_parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    a4 = a4.copy()
    a4["month"] = a4["__decision_ts__"].dt.strftime("%Y-%m")
    a0 = a0.copy()
    a0["month"] = a0["__decision_ts__"].dt.strftime("%Y-%m")
    for month, held_a4 in a4.groupby("month", sort=True):
        held_a0 = a0.loc[a0["month"].eq(month)].copy()
        if not held_a0[identity].reset_index(drop=True).equals(
            held_a4[identity].reset_index(drop=True)
        ):
            raise ValueError(f"A0/A4 identity mismatch in {month}")
        cutoff = held_a4["__decision_ts__"].min().floor("D")
        held = held_a0.copy()
        held["trust_posterior_expected_bps"] = pd.to_numeric(
            held_a0["posterior_expected_bps"], errors="coerce",
        )
        held["a4_raw_expected_bps"] = pd.to_numeric(
            held_a4["posterior_expected_bps"], errors="coerce",
        ).to_numpy()
        held["a4_raw_predictive_sd_bps"] = pd.to_numeric(
            held_a4["posterior_predictive_sd"], errors="coerce",
        ).to_numpy()
        try:
            calibration = fit_a5_calibration(
                a4, cutoff=cutoff,
                source_hashes={
                    **{f"a4:{path}": _sha(path) for path in args.a4},
                },
            )
            integrated = apply_a5_bounded_10pct(held, calibration=calibration)
            held = held.merge(integrated, on="candidate_id", how="left", validate="one_to_one")
            status = "fit"
        except ValueError as exc:
            if "prior resolved OOS" not in str(exc):
                raise
            calibration = None
            status = "fail_closed_insufficient_prior_oos_calibration"
            held["a5_calibrated_expected_bps"] = float("nan")
            held["a5_calibrated_p_positive"] = float("nan")
            held["a5_bounded10_expected_bps"] = float("nan")
            held["a5_timestamp_top15"] = False
            held["a5_bounded10_available"] = False
            held["a5_bounded10_admitted"] = False
        output_parts.append(held)
        audit.append({
            "month": month, "cutoff": cutoff,
            "rows": int(len(held)),
            "a0_admitted": int(pd.to_numeric(
                held["trust_posterior_expected_bps"], errors="coerce",
            ).ge(50.0).sum()),
            "a0_top15_admitted": int(held["a5_bounded10_admitted"].sum()),
            "status": status,
            "calibration_prior_oos_rows": (
                0 if calibration is None else calibration.prior_oos_rows
            ),
            "calibration_slope": None if calibration is None else calibration.slope,
            "calibration_intercept": None if calibration is None else calibration.intercept,
            "calibration_predictive_sd_scale": (
                None if calibration is None else calibration.predictive_sd_scale
            ),
        })
    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    if output["candidate_id"].duplicated().any():
        raise AssertionError("A5 held folds overlap")
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "a5_bounded10_oof_predictions.parquet", index=False)
    pd.DataFrame(audit).to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_a5_bounded10_walkforward_v1",
        "rows": int(len(output)), "folds": int(len(audit)),
        "a0": [{"path": str(path), "sha256": _sha(path)} for path in args.a0],
        "a4": [{"path": str(path), "sha256": _sha(path)} for path in args.a4],
        "formula": "A0 + 0.10 * (causally_calibrated_A4 - A0)",
        "admission": "A0>=50 AND timestamp-local top15 by pre-trust final_score",
        "A5_changes_fixed_admission": False,
        "calibration": "month t uses earlier OOS A4 predictions with resolved labels only",
        "outcomes_used_for_held_score_or_admission": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
