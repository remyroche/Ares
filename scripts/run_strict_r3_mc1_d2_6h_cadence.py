#!/usr/bin/env python3
"""Replay frozen MC1_d2 with a strictly prior-resolved six-hour residual shift.

The frozen static MC1_d2 HGB prediction is reused verbatim.  Only its dynamic
global residual adjustment is made six-hourly: each 6-hour decision block sees
only labels resolved *before* that block.  The structural score-band curve is
anchored at that UTC day's open so it too is strictly causal; this makes the
test a bounded cadence ablation, rather than a target, feature, or upstream
model change.  Offline research only; it never reads or writes live state.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_mc1_d2_historical_parity import (  # noqa: E402
    CORE,
    day_balanced,
    history,
    robust,
    score_bands,
    structural,
    utc,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_exit_bar_15m", *CORE,
    ]
    full_data = pd.read_parquet(args.ledger, columns=columns)
    full_data["__decision_ts__"] = pd.to_datetime(full_data["__decision_ts__"], utc=True)
    full_data["policy_label_available_ts"] = pd.to_datetime(full_data["policy_label_available_ts"], utc=True)
    if not full_data.side_name.astype(str).str.lower().eq("long").all():
        raise ValueError("MC1 six-hour cadence replay is long-only")
    full_data["score_band"] = score_bands(full_data)
    full_data["day"] = full_data.__decision_ts__.dt.normalize()
    full_data["block"] = full_data.__decision_ts__.dt.floor("6h")
    # The calibration source must retain compatible pre-evaluation outcomes.
    # Restricting it to frozen prediction IDs would silently remove the
    # pre-2025 history used by the canonical MC1 control.
    source = day_balanced(full_data)

    frozen = pd.read_parquet(
        args.frozen_predictions,
        columns=["candidate_id", "static_expected_bps"],
    )
    if frozen.candidate_id.duplicated().any():
        raise ValueError("frozen MC1 candidate identity is not unique")
    data = full_data.merge(frozen, on="candidate_id", how="inner", validate="one_to_one")
    if len(data) != len(frozen):
        raise ValueError("frozen MC1 prediction and ledger identities do not match")
    start, end = utc(args.start), utc(args.end)
    blocks = pd.date_range(start, end, freq="6h", inclusive="left", tz="UTC")

    # Daily curve state is recomputed from only prior-resolved source outcomes.
    # It is intentionally kept fixed within the day, so any effect is from the
    # requested finer residual-map cadence rather than a changed score geometry.
    curves: dict[pd.Timestamp, np.ndarray] = {}
    for day in pd.date_range(start, end, freq="D", inclusive="left", tz="UTC"):
        train = history(source, day, None, inclusive=False)
        if len(train) >= 5_000:
            curves[day], _ = structural(train)
        del train
        gc.collect()

    pieces: list[pd.DataFrame] = []
    for index, block in enumerate(blocks, start=1):
        rows = data.loc[data.block.eq(block)].copy()
        if rows.empty or block.normalize() not in curves:
            continue
        recent = history(source, block, 21, inclusive=False)
        curve = curves[block.normalize()]
        shift = robust(recent.net - curve[recent.score_band.to_numpy(int)]) if len(recent) else float("nan")
        rows["recent_shift_bps"] = shift
        rows["mc1_expected_bps"] = rows.static_expected_bps + shift
        rows["mapper_kind"] = "mc1_d2_frozen_static_6h_shift"
        rows["fold_start"] = pd.Timestamp(block.year, block.month, 1, tz="UTC")
        pieces.append(rows.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "policy_label_available_ts",
            "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", "final_score",
            "static_expected_bps", "recent_shift_bps", "mc1_expected_bps", "mapper_kind", "fold_start",
        ]])
        if index % 120 == 0:
            print(json.dumps({"event": "block_complete", "block": block.isoformat(), "shift": shift}), flush=True)
        del rows, recent
        gc.collect()

    output = pd.concat(pieces, ignore_index=True)
    output.to_parquet(args.out_dir / "predictions_mc1_d2_6h_dailycurve.parquet", index=False, compression="zstd")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_6h_cadence_v1", "status": "complete",
        "purpose": "bounded MC1_d2 cadence ablation; frozen static model and six inputs",
        "static_model": "verbatim frozen strict MC1_d2 output",
        "dynamic_shift": "21d 10%-trimmed global residual shift recalculated every 6h",
        "structural_curve": "daily score-band curve fit only on labels resolved before UTC day open",
        "label_availability_boundary": "policy_label_available_ts < 6h decision block",
        "admission": "+50 bps in reporting replay", "auction": "frozen final_score only",
        "exclusions": ["R5", "live state", "exchange I/O", "upstream model changes"],
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "rows": len(output),
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(output)}))


if __name__ == "__main__":
    main()
