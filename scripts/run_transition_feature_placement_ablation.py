#!/usr/bin/env python3
"""Four-arm OOF placement ablation for the market-transition sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_leaf_target_family_ablation import INPUT, AVAILABILITY, run

SIDECAR = ROOT / "data_perp/artifacts/market_transition_sidecar_2022_2026_20260803_v1/market_transition_features.parquet"
OUT = ROOT / "data_perp/artifacts/transition_feature_placement_ablation_20260803_v1"


def main() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {
        "current": (False, False),
        "correctness_only": (True, False),
        "meta_only": (False, True),
        "correctness_and_meta": (True, True),
    }
    rows = []
    for name, (to_correctness, to_meta) in arms.items():
        artifact = run(
            OUT / name, INPUT, AVAILABILITY, families=("correctness",),
            transition_sidecar=SIDECAR, transition_to_correctness=to_correctness,
            transition_to_meta=to_meta, history_days=None, end_ts="2026-04-01",
        )
        metrics = pd.read_parquet(artifact / "target_family_oof_comparison.parquet")
        metrics["placement_arm"] = name
        rows.append(metrics)
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(OUT / "placement_oof_metrics.parquet", index=False)
    current = result[result.placement_arm.eq("current") & result.arm.eq("target_correctness")][["fold", "top_fraction", "net_bps"]].rename(columns={"net_bps": "current_net_bps"})
    comparison = result[result.arm.eq("target_correctness")].merge(current, on=["fold", "top_fraction"])
    comparison["delta_vs_current_bps"] = comparison.net_bps - comparison.current_net_bps
    comparison.to_parquet(OUT / "placement_oof_comparison.parquet", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"status": "COMPLETED", "arms": list(arms), "target": "existing correctness", "horizons": ["row", "period12h", "period24h", "period72h"], "sidecar": str(SIDECAR), "ranking": "pooled global top-k after common-bps meta map"}, indent=2) + "\n")
    return OUT


if __name__ == "__main__":
    print(main())
