#!/usr/bin/env python3
"""Combine side-local causal target screens under the live pooled-global rank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ARMS = {
    "R0_B3": (
        ROOT / "data_perp/artifacts/tp6_screen_long_r0_20260802_v2/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r0_20260802_v1/target_repair_oof_predictions.parquet",
    ),
    "R2_soft_b25": (
        ROOT / "data_perp/artifacts/tp6_screen_long_robust25_20260802_v1/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_robust25_20260802_v1/target_repair_oof_predictions.parquet",
    ),
    "R2_event_b25": (
        ROOT / "data_perp/artifacts/tp6_screen_long_r2_event25_20260802_v1/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r2_event25_20260802_v1/target_repair_oof_predictions.parquet",
    ),
    "R3_economic_simplex_b25": (
        ROOT / "data_perp/artifacts/tp6_screen_long_r3_20260802_v1/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r3_20260802_v1/target_repair_oof_predictions.parquet",
    ),
    "R3_certainty": (
        ROOT / "data_perp/artifacts/tp6_screen_long_r3_certainty_20260802_v1/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r3_certainty_20260802_v1/target_repair_oof_predictions.parquet",
    ),
    "R3_regime": (
        ROOT / "data_perp/artifacts/tp6_screen_long_r3_regime_20260802_v1/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r3_regime_20260802_v1/target_repair_oof_predictions.parquet",
    ),
    "R3_composite": (
        ROOT / "data_perp/artifacts/tp6_screen_long_r3_composite_20260802_v1/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r3_composite_20260802_v1/target_repair_oof_predictions.parquet",
    ),
}


def _metrics(frame: pd.DataFrame, scope: str, frac: float) -> dict[str, object]:
    n = int(np.ceil(len(frame) * frac))
    chosen = frame.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(n)
    return {
        "scope": scope, "top_fraction": frac, "n": len(chosen),
        "gross_bps": chosen.gross_bps.mean(), "net_bps": chosen.net_bps.mean(),
        "long_n": int(chosen.side_name.eq("long").sum()),
        "short_n": int(chosen.side_name.eq("short").sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    results: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    for arm, (long_path, short_path) in DEFAULT_ARMS.items():
        pieces = [pd.read_parquet(path) for path in (long_path, short_path)]
        frame = pd.concat(pieces, ignore_index=True)
        # The individual screen directories are side shards, so assert that
        # their contract cannot silently become a side quota.
        if set(frame.side_name.unique()) != {"long", "short"}:
            raise ValueError(f"{arm} lacks one side")
        frame["arm"] = arm
        all_predictions.append(frame)
        for frac in (.01, .05, .10, .20):
            row = _metrics(frame, "global", frac); row["arm"] = arm; results.append(row)
            for side, side_frame in frame.groupby("side_name", observed=True):
                row = _metrics(side_frame, str(side), frac); row["arm"] = arm; results.append(row)
            selected = frame.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(int(np.ceil(len(frame)*frac))).copy()
            selected["month"] = pd.to_datetime(selected.__ts__, utc=True).dt.to_period("M").astype(str)
            for (month, side), group in selected.groupby(["month", "side_name"], observed=True):
                monthly.append({"arm": arm, "top_fraction": frac, "month": month, "side": side,
                                "n": len(group), "gross_bps": group.gross_bps.mean(), "net_bps": group.net_bps.mean()})
    args.out.mkdir(parents=True)
    pd.concat(all_predictions, ignore_index=True).to_parquet(args.out / "target_repair_oof_predictions.parquet", index=False)
    pd.DataFrame(results).to_parquet(args.out / "target_repair_results.parquet", index=False)
    pd.DataFrame(monthly).to_parquet(args.out / "target_repair_monthly_selected.parquet", index=False)
    manifest = {"schema": "tp6_sl4_target_repair_pooled_screen_v1", "status": "COMPLETED",
                "contract": {"geometry": "selected TP=+6ATR / SL=-4ATR / H12", "cost_bps": 100,
                             "ranking": "pooled global score_bps after side-local calibration-only bps mapping", "side_quotas": False},
                "arms": list(DEFAULT_ARMS), "metrics": results}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(pd.DataFrame(results).query("scope == 'global'").to_string(index=False))


if __name__ == "__main__":
    main()
