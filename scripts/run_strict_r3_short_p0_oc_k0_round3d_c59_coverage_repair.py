#!/usr/bin/env python3
"""Matched C59 coverage repair for the short P0 -> O -> C -> K0 stack.

The frozen C60 contract contains one development-useful feature whose
2025--2026 availability is only 77.8%.  This ablation removes only that field
and keeps O, the C3 target, seeds, weights, and the analytic K0 implementation
otherwise identical.  It is not a new feature-selection sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair_v1"
ROUND3B = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_refinement_20260822_v2"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair_20260822_v1"
TARGET = next(item for item in r3.TARGETS if item.name == "C3_normalized_regret")
DROP = "ob_trade_size_to_l1_depth_z_24h"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _c59() -> tuple[str, ...]:
    source = json.loads((ROUND3B / "run_manifest.json").read_text())
    fields = tuple(source["conversion"]["feature_contracts"]["C60_mda"])
    if fields.count(DROP) != 1:
        raise AssertionError("coverage-repair drop is not uniquely present in C60")
    output = tuple(field for field in fields if field != DROP)
    if len(output) != 59:
        raise AssertionError("C59 contract length is invalid")
    return output


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, o_fields, _, sources = r3._load_frame()
    fields = _c59()
    prediction, audit = r3._run_target(
        frame, o_fields, fields, TARGET, r3b.C_SEED, "uniform", o_seed=r3b.O_SEED,
    )
    reference = pd.read_parquet(ROUND3B / "C60_mda__uniform_outer_oof_predictions.parquet")
    r3b._assert_fixed_o(prediction, reference)
    monthly, era, summary = r3b._metrics(prediction, "C59_drop_low_coverage")
    values = pd.to_numeric(frame.loc[r3b.r1._valid_label(frame) & frame["__decision_ts__"].ge("2025-01-01"), DROP], errors="coerce")
    coverage = float(values.notna().mean())
    if coverage >= .90:
        raise AssertionError("coverage-repair feature no longer requires an ablation")
    out.mkdir(parents=True)
    prediction.to_parquet(out / "C59_outer_oof_predictions.parquet", index=False, compression="zstd")
    audit.to_parquet(out / "C59_fold_audit.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "C59_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "C59_era_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "matched C coverage repair; research only, no canonical/live mutation",
        "architecture": "frozen P0 -> O250/H6 -> C3 normalized regret C59/uniform -> original analytic K0",
        "change": {"dropped_feature": DROP, "2025_2026_valid_coverage": coverage, "reason": "below 90% later-era availability"},
        "invariants": {"frozen_o_seed": r3b.O_SEED, "c_seed": r3b.C_SEED, "target": TARGET.name, "weights": "uniform", "o_parity_vs_c60": "exact"},
        "metrics": summary,
        "sources": {"round3b_manifest_sha256": _sha256(ROUND3B / "run_manifest.json"), **sources},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Short C59 coverage-repair ablation", "",
        "Only `ob_trade_size_to_l1_depth_z_24h` is removed from the frozen C60 contract because later-era coverage is below 90%.", "",
        pd.DataFrame([summary]).to_markdown(index=False), "", "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_C59_COVERAGE_REPAIR_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
