#!/usr/bin/env python3
"""Diagnostic for abstaining when the one-field GAM transport is invalid.

This deliberately keeps the canonical TP6/SL4 candidate population and the
canonical 75/25 Base+Consensus baseline unchanged.  It reports a separate
abstention diagnostic rather than promoting a reduced-exposure ranking to the
primary comparison.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    finite = frame.loc[np.isfinite(frame[score].to_numpy(float))].copy()
    n = max(1, int(math.ceil(len(finite) * tail)))
    top = finite.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {
        "tail": float(tail),
        "trades": int(len(top)),
        "scored_rows": int(len(finite)),
        "scored_fraction": float(len(finite) / max(len(frame), 1)),
        "gross_bps_per_trade": float(top.exact_gross_bps.mean()) if len(top) else np.nan,
        "net_bps_per_trade": float(top.exact_net_bps.mean()) if len(top) else np.nan,
        "invalid_trades_in_tail": int((~top.transport_valid.astype(bool)).sum()) if len(top) else 0,
    }


def run(*, predictions_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frame = pd.read_parquet(predictions_path).copy()
    required = {"candidate_id", "exact_net_bps", "exact_gross_bps", "base_plus_consensus25", "gam_input", "transport_valid"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["transport_valid"] = frame["transport_valid"].fillna(0).astype(bool)
    frame["control_abstain_invalid"] = frame["base_plus_consensus25"].where(frame["transport_valid"], -np.inf)
    frame["gam_abstain_invalid"] = frame["gam_input"].where(frame["transport_valid"], -np.inf)
    rows: list[dict[str, object]] = []
    for arm in ("canonical_control", "gated_gam_input", "control_abstain_invalid", "gam_abstain_invalid"):
        score = {"canonical_control": "base_plus_consensus25", "gated_gam_input": "gam_input", "control_abstain_invalid": "control_abstain_invalid", "gam_abstain_invalid": "gam_abstain_invalid"}[arm]
        for tail in TAILS:
            row = _metric(frame, score, tail)
            row["arm"] = arm
            rows.append(row)
    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True)
    metrics.to_parquet(output_dir / "abstention_metrics.parquet", index=False)
    audit = {
        "schema": "tp6_sl4_gam_canonical_abstention_v1",
        "status": "COMPLETE",
        "predictions_path": str(predictions_path),
        "rows": int(len(frame)),
        "transport_valid_rows": int(frame.transport_valid.sum()),
        "transport_invalid_rows": int((~frame.transport_valid).sum()),
        "transport_valid_fraction": float(frame.transport_valid.mean()),
        "primary_comparison": "canonical_control versus gated_gam_input; exact same rows/exits",
        "abstention_note": "abstention arms are diagnostic only: invalid rows are removed from exposure and the remaining valid rows are re-ranked to the requested quota, so their bps/trade is not directly comparable to full-population Top-k",
        "exits": "TP +6 ATR / SL -4 ATR / H12 / 100 bps once",
        "artifacts": ["abstention_metrics.parquet", "run_manifest.json", "TP6_SL4_GAM_CANONICAL_ABSTENTION_REPORT.md"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    report = "\n".join([
        "# TP6/SL4 canonical GAM abstention diagnostic",
        "",
        "This is a diagnostic only. The canonical control and gated GAM arms use the identical candidate population, canonical Base+Consensus score, and TP6/SL4 exit. The abstention arms set invalid-transport scores to `-inf`, remove those rows from exposure, and then refill the requested quota from valid rows; therefore their bps/trade cannot be compared as a like-for-like full-population Top-k result.",
        "",
        f"Transport-valid rows: {int(frame.transport_valid.sum())}/{len(frame)} ({frame.transport_valid.mean():.1%}).",
        "",
        metrics.round(3).to_string(index=False),
        "",
    ])
    (output_dir / "TP6_SL4_GAM_CANONICAL_ABSTENTION_REPORT.md").write_text(report + "\n")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(predictions_path=args.predictions, output_dir=args.output_dir))
