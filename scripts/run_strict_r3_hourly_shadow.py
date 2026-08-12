#!/usr/bin/env python3
"""Materialize and run one fail-closed strict-R3 hourly shadow cycle.

This command has no network client and no exchange/order authority.  Source
refresh remains an explicit prior operation.  It builds the complete frozen
universe first, computes every feature on that complete point-in-time panel,
scores only candidates passing the contemporaneous spread/entry gate, and
then delegates to the sealed shadow-only score/admission/portfolio runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import (  # noqa: E402
    StrictR3InferenceBundle,
    validate_live_feature_frame,
)


SCHEMA = "strict_r3_hourly_shadow_orchestration_v1"
CANONICAL_INFERENCE_BUNDLE = (
    ROOT / "config" / "strict_r3_inference_bundle_long_20260801_v5_a5_b10.json"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_hour(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )
    if timestamp != timestamp.floor("h"):
        raise ValueError("strict-R3 decision timestamp must be an exact UTC hour")
    return timestamp


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"hourly shadow stage failed; see {log_path}")


def _commands(
    *,
    bundle_path: Path,
    bundle: StrictR3InferenceBundle,
    state_path: Path,
    decision: pd.Timestamp,
    out_dir: Path,
) -> list[tuple[str, list[str]]]:
    signal = decision - pd.Timedelta(hours=1)
    runtime = dict(bundle.payload.get("runtime") or {})
    history_start = runtime.get("feature_history_start")
    if not history_start:
        raise ValueError("sealed inference bundle lacks feature_history_start")
    grid_dir = out_dir / "candidate_grid"
    feature_dir = out_dir / "features"
    cycle_dir = out_dir / "cycle"
    return [
        (
            "candidate_grid",
            [
                sys.executable,
                str(ROOT / runtime["candidate_materializer"]),
                "--universe-manifest", str(bundle.path("frozen_universe_manifest")),
                "--start", signal.isoformat(),
                "--end-exclusive", decision.isoformat(),
                "--sides", "long",
                "--spread-limit-bps", "100",
                "--out-dir", str(grid_dir),
            ],
        ),
        (
            "features",
            [
                sys.executable,
                str(ROOT / runtime["feature_materializer"]),
                # The complete population, including spread rejects, defines
                # the causal cross-section.  Scoring is filtered separately.
                "--candidates", str(grid_dir / "target_free_candidate_population.parquet"),
                "--out-dir", str(feature_dir),
                "--candidate-start", decision.isoformat(),
                "--history-start", str(history_start),
                "--end-exclusive", (decision + pd.Timedelta(hours=1)).isoformat(),
                "--side", "long",
            ],
        ),
        (
            "shadow_cycle",
            [
                sys.executable,
                str(ROOT / runtime["shadow_cycle"]),
                "--inference-bundle", str(bundle_path),
                "--held-candidates", str(grid_dir / "eligible_candidates.parquet"),
                "--held-features", str(feature_dir / "canonical120_features.parquet"),
                "--portfolio-state-json", str(state_path),
                "--decision-ts", decision.isoformat(),
                "--out-dir", str(cycle_dir),
                "--mode", "shadow-only",
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-bundle", type=Path, default=CANONICAL_INFERENCE_BUNDLE,
        help="Immutable schema-v5 bounded-A5 10% bundle (canonical default).",
    )
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow-only",), default="shadow-only")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable hourly shadow output exists: {args.out_dir}")
    decision = _utc_hour(args.decision_ts)
    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    bundle_audit = bundle.validate(decision_ts=decision)
    args.out_dir.mkdir(parents=True)
    stages = _commands(
        bundle_path=args.inference_bundle,
        bundle=bundle,
        state_path=args.portfolio_state_json,
        decision=decision,
        out_dir=args.out_dir,
    )
    # Candidate and feature parity are fail-before-score gates.  Never allow
    # an invalid live matrix to reach a model, admission map, or auction.
    for name, command in stages[:2]:
        _run(command, log_path=args.out_dir / f"{name}.log")

    grid_manifest = json.loads(
        (args.out_dir / "candidate_grid" / "run_manifest.json").read_text(),
    )
    feature_manifest = json.loads(
        (args.out_dir / "features" / "feature_manifest.json").read_text(),
    )
    population = pd.read_parquet(
        args.out_dir / "candidate_grid" / "target_free_candidate_population.parquet",
    )
    eligible = pd.read_parquet(
        args.out_dir / "candidate_grid" / "eligible_candidates.parquet",
    )
    features = pd.read_parquet(
        args.out_dir / "features" / "canonical120_features.parquet",
    )
    feature_contract = json.loads(bundle.path("feature_contract").read_text())
    scoring_features = features.loc[
        features["candidate_id"].isin(set(eligible["candidate_id"])),
    ].copy()
    feature_parity_audit = validate_live_feature_frame(
        scoring_features,
        fields=list(feature_contract["base_fields_by_side"]["long"]),
        requirements=dict(bundle.payload["feature_parity"]),
    )
    name, command = stages[2]
    _run(command, log_path=args.out_dir / f"{name}.log")
    cycle_manifest = json.loads(
        (args.out_dir / "cycle" / "run_manifest.json").read_text(),
    )
    if len(population) != int(grid_manifest["universe_rows"]):
        raise AssertionError("hourly feature population does not cover the frozen universe")
    if len(features) != len(population):
        raise AssertionError("features were not generated on the complete universe")
    if not set(eligible["candidate_id"]).issubset(set(features["candidate_id"])):
        raise AssertionError("one or more actionable identities lack complete-universe features")
    if grid_manifest.get("spread_gate") != (
        "official_kraken_signal_hour_bid_ask_bps_before_signal_plus_1h_entry"
    ):
        raise AssertionError("hourly grid did not use the contemporaneous spread gate")
    if not all(cycle_manifest.get("checks", {}).values()):
        raise AssertionError("nested strict-R3 shadow cycle failed its invariant set")

    manifest = {
        "schema": SCHEMA,
        "mode": "shadow-only",
        "decision_ts": decision.isoformat(),
        "signal_ts": (decision - pd.Timedelta(hours=1)).isoformat(),
        "inference_bundle_audit": bundle_audit,
        "population_rows": int(len(population)),
        "eligible_rows": int(len(eligible)),
        "rejected_rows": int(len(population) - len(eligible)),
        "feature_rows": int(len(features)),
        "feature_parity_rows": int(len(scoring_features)),
        "feature_parity_audit": feature_parity_audit,
        "mapped_rows": int(cycle_manifest["mapped_rows"]),
        "admitted_rows": int(cycle_manifest["admitted_rows"]),
        "portfolio_accepted_rows": int(cycle_manifest["portfolio_accepted_rows"]),
        "complete_universe_features_before_actionability_filter": True,
        "current_spread_gate": True,
        "future_paths_consumed": [],
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "hashes": {
            "inference_bundle": _sha(args.inference_bundle),
            "portfolio_state": _sha(args.portfolio_state_json),
            "candidate_population": _sha(args.out_dir / "candidate_grid" / "target_free_candidate_population.parquet"),
            "eligible_candidates": _sha(args.out_dir / "candidate_grid" / "eligible_candidates.parquet"),
            "features": _sha(args.out_dir / "features" / "canonical120_features.parquet"),
            "shadow_decisions": _sha(args.out_dir / "cycle" / "shadow_decisions.parquet"),
        },
        "feature_source_contract": feature_manifest.get("bar_source_contract"),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
