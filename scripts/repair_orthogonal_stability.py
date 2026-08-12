#!/usr/bin/env python3
"""Recompute the stability table after the first orthogonal run.

The initial run used ``DataFrame.tail`` instead of the ``tail`` column while
building the gate.  This tiny repair is intentionally separate from the large
ablation: it only reads the immutable OOS metrics and cannot change scores.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_long_family_orthogonal_ablation import _stability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/long_family_orthogonal_ablation_20260808_v1")
    out = parser.parse_args().out
    metrics = pd.read_parquet(out / "orthogonal_ablation_metrics.parquet")
    stability = _stability(metrics)
    stability.to_parquet(out / "stability_metrics.parquet", index=False, compression="zstd")
    gate = stability[(stability["tail"] == 0.05) & stability["stability_gate"]].sort_values(
        ["pooled_uplift_bps", "worst_month_uplift_bps"], ascending=False
    )
    winner = str(gate.iloc[0]["arm"]) if not gate.empty else "NO_ARM_PASSES_STABILITY_GATE"
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["winner"] = winner
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path = out / "ORTHOGONAL_FAMILY_ABLATION_REPORT.md"
    report = report_path.read_text()
    report = report.replace(
        "Top-5 stability winner: **NO_ARM_PASSES_STABILITY_GATE**.",
        f"Top-5 stability winner: **{winner}**.",
    )
    report_path.write_text(report)
    print(stability[stability["tail"] == 0.05].sort_values("pooled_uplift_bps", ascending=False).to_string(index=False))
    print("winner", winner)


if __name__ == "__main__":
    main()
