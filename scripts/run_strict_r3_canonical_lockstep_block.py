#!/usr/bin/env python3
"""Materialise one recoverable strict-R3 lock-step OOF block.

This worker is deliberately one-cutoff-at-a-time.  It makes the long
walk-forward replay restartable on constrained machines while retaining the
same full 28-day reserve, producer identity, frozen geometry, and target-free
candidate population as the multi-block producer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_canonical_lockstep_walkforward as runner  # noqa: E402
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    CALIBRATION_RESERVE_DAYS,
    FOUR_WEEK_DAYS,
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    persist_four_week_conversion_bundle,
    persist_monthly_upstream_bundle,
    score_monthly_upstream_bundle,
    train_four_week_conversion_bundle,
    train_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import load_geometry_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--held-end", required=True)
    parser.add_argument("--score-chunk-hours", type=int, default=72)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cutoff, held_end = runner._utc(args.cutoff), runner._utc(args.held_end)
    expected_held_end = cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS)
    if held_end != expected_held_end:
        raise ValueError(
            "lock-step blocks must use exactly one complete 28-day held window; "
            f"expected {expected_held_end.isoformat()}",
        )
    if args.score_chunk_hours < 1:
        raise ValueError("score chunk must be positive")
    reserve_start = cutoff - pd.Timedelta(days=CALIBRATION_RESERVE_DAYS)
    fields = runner._fields(args.feature_contract)
    geometry = load_geometry_bundle(args.geometry_bundle)
    if (
        geometry.fit_audit.get("definition_start") != "2024-10-01T00:00:00+00:00"
        or geometry.fit_audit.get("definition_end_exclusive") != "2025-01-01T00:00:00+00:00"
    ):
        raise ValueError("lock-step worker requires frozen Oct-Dec 2024 geometry/K9")
    ledger = pd.read_parquet(args.prequential_ledger, filters=[("__decision_ts__", "<", cutoff)])
    ledger = ledger.loc[ledger["side_name"].astype(str).str.lower().eq("long")].copy()
    for column in (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts", "h12_label_available_ts",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True)
    working = runner._initialise_working_ledger(ledger)
    # Prior OOF blocks are the only new rows not already present in the
    # prequential handoff.  Update known, label-bearing identities only.
    prior_paths = sorted(args.out_dir.glob("bundles/cutoff=*/scores/held_target_free_scores.parquet"))
    for path in prior_paths:
        prior_score = pd.read_parquet(path)
        prior_score["__decision_ts__"] = pd.to_datetime(prior_score["__decision_ts__"], utc=True)
        prior_score = prior_score.loc[prior_score["__decision_ts__"].lt(cutoff)].copy()
        if len(prior_score):
            runner._apply_upstream_scores(working, prior_score)

    source_hashes = {
        "source_panel": runner._sha(args.source_panel),
        "prequential_ledger": runner._sha(args.prequential_ledger),
        "feature_contract": runner._sha(args.feature_contract),
        "geometry_manifest": runner._sha(args.geometry_bundle / "run_manifest.json"),
        "calibration_reserve_days": str(CALIBRATION_RESERVE_DAYS),
        "refit_cadence": f"lockstep_{FOUR_WEEK_DAYS}d",
    }
    block_dir = args.out_dir / "bundles" / f"cutoff={cutoff:%Y%m%d}"
    upstream_dir, conversion_dir, scores_dir = (
        block_dir / "upstream", block_dir / "conversion", block_dir / "scores",
    )
    held_path, reference_path = (
        scores_dir / "held_target_free_scores.parquet",
        scores_dir / "reserve_target_free_scores.parquet",
    )
    if args.resume and held_path.exists() and reference_path.exists():
        print(json.dumps({"event": "already_complete", "cutoff": cutoff.isoformat(), "block_dir": str(block_dir)}))
        return
    raw_reference = runner._read_target_free(args.source_panel, start=reserve_start, end=cutoff, fields=fields)
    raw_held = runner._read_target_free(args.source_panel, start=cutoff, end=held_end, fields=fields)
    prior = working.loc[working["__decision_ts__"].lt(cutoff)].copy().reset_index(drop=True)
    if args.resume and (upstream_dir / "run_manifest.json").exists():
        upstream = load_monthly_upstream_bundle(upstream_dir)
        upstream_status = "resumed"
    else:
        upstream = train_monthly_upstream_bundle(
            cutoff=cutoff, training_ledger=prior, prior42_features=raw_reference,
            base_fields=fields, source_hashes=source_hashes,
            calibration_reserve_days=CALIBRATION_RESERVE_DAYS,
            held_end_exclusive=held_end,
        )
        persist_monthly_upstream_bundle(upstream, upstream_dir)
        upstream_status = "fit"
    reference_upstream = score_monthly_upstream_bundle(
        upstream, raw_reference, allow_prior_reference=True, prior_reference_start=reserve_start,
    )
    held_upstream = score_monthly_upstream_bundle(upstream, raw_held)
    if args.resume and (conversion_dir / "run_manifest.json").exists():
        conversion = load_four_week_conversion_bundle(conversion_dir)
        conversion_status = "resumed"
    else:
        conversion = train_four_week_conversion_bundle(
            cutoff=cutoff, upstream_ledger=prior, frozen_geometry=geometry,
            base_fields=fields, source_hashes=source_hashes,
            calibration_reserve_days=CALIBRATION_RESERVE_DAYS,
        )
        persist_four_week_conversion_bundle(conversion, conversion_dir)
        conversion_status = "fit"
    reference_input = runner._attach_scores(raw_reference, reference_upstream)
    held_input = runner._attach_scores(raw_held, held_upstream)
    del raw_reference, raw_held, reference_upstream, held_upstream, prior, working, ledger
    scored, score_audit = runner._score_conversion_memory_bound(
        conversion, reference=reference_input, held=held_input, chunk_hours=args.score_chunk_hours,
    )
    scored["calibration_reserve_start"] = reserve_start
    scored["calibration_activation_ts"] = cutoff
    score_ts = pd.to_datetime(scored["__decision_ts__"], utc=True, errors="raise")
    scored["calibration_reference_oos_to_all_active_fits"] = (
        scored["__score_role__"].eq("reference") & score_ts.ge(reserve_start) & score_ts.lt(cutoff)
    )
    scored["calibration_reference_contract"] = (
        "full 28-day shared reserve excluded from lock-step upstream and conversion supervised fits"
    )
    scores_dir.mkdir(parents=True, exist_ok=True)
    held = scored.loc[scored["__score_role__"].eq("held")].drop(columns="__score_role__").copy()
    held["stack_is_prequential"] = True
    held.to_parquet(held_path, index=False, compression="zstd")
    scored.loc[scored["__score_role__"].eq("reference")].to_parquet(
        reference_path, index=False, compression="zstd",
    )
    score_audit.to_parquet(scores_dir / "conversion_score_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_lockstep_block_v1",
        "cutoff": cutoff.isoformat(),
        "held_end_exclusive": held_end.isoformat(),
        "reserve_start": reserve_start.isoformat(),
        "reserve_days": CALIBRATION_RESERVE_DAYS,
        "shared_upstream_conversion_cutoff": True,
        "geometry_bundle_sha256": conversion.geometry.bundle_sha256,
        "upstream_bundle_sha256": upstream.manifest["bundle_sha256"],
        "conversion_bundle_sha256": conversion.manifest["bundle_sha256"],
        "upstream_status": upstream_status,
        "conversion_status": conversion_status,
        "reserve_rows": int(scored["__score_role__"].eq("reference").sum()),
        "held_rows": int(len(held)),
        "outcomes_consumed_during_scoring": [],
        "source_hashes": source_hashes,
    }
    (scores_dir / "block_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "block_dir": str(block_dir), **manifest}))


if __name__ == "__main__":
    main()
