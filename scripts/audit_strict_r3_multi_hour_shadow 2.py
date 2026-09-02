#!/usr/bin/env python3
"""Audit consecutive strict-R3 shadow hours with one causal feature build.

The command has no exchange or order authority.  It scores a target-free
multi-hour population once, applies the sealed exact-reserve Cell-day map at
each exact UTC decision hour, and carries hypothetical entries conservatively
for the full H12 policy horizon.  No future outcome is used to infer an early
exit, wallet change, or portfolio capacity.
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
from extreme_price_movements.strict_r3_shadow_portfolio import (  # noqa: E402
    ShadowOpenPosition,
    ShadowPortfolioPolicy,
    ShadowPortfolioState,
    auction_admitted_snapshot,
)
from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    load_cell_day_residual_trust_bundle,
)
from extreme_price_movements.strict_r3_a5_trust import (  # noqa: E402
    apply_a5_bounded_10pct,
    load_a5_bundle,
)


SCHEMA = "strict_r3_multi_hour_shadow_audit_v2_full_a5"
CANONICAL_ADMISSION = "strict_oof_exact_producer_cell_day_trim15_28d_v1"


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
        raise ValueError("multi-hour shadow boundaries must be exact UTC hours")
    return timestamp


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"multi-hour shadow stage failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--start-decision-ts", required=True)
    parser.add_argument("--end-exclusive-decision-ts", required=True)
    parser.add_argument("--wallet", type=float, default=1000.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow-only",), default="shadow-only")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable multi-hour output exists: {args.out_dir}")
    start = _utc_hour(args.start_decision_ts)
    end = _utc_hour(args.end_exclusive_decision_ts)
    decisions = pd.date_range(start, end, freq="h", inclusive="left")
    if decisions.empty:
        raise ValueError("multi-hour shadow window is empty")
    if not pd.notna(args.wallet) or args.wallet <= 0.0:
        raise ValueError("shadow wallet must be positive")

    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    first_audit = bundle.validate(decision_ts=decisions[0])
    last_audit = bundle.validate(decision_ts=decisions[-1])
    runtime = dict(bundle.payload.get("runtime") or {})
    activation = _utc_hour(bundle.payload["activation_ts"])
    first_decision = activation + pd.Timedelta(hours=1)
    args.out_dir.mkdir(parents=True)
    grid_dir = args.out_dir / "candidate_grid"
    feature_dir = args.out_dir / "features"
    score_dir = args.out_dir / "score"
    _run([
        sys.executable, str(ROOT / runtime["candidate_materializer"]),
        "--universe-manifest", str(bundle.path("frozen_universe_manifest")),
        "--start", activation.isoformat(),
        "--end-exclusive", (end - pd.Timedelta(hours=1)).isoformat(),
        "--sides", "long", "--spread-limit-bps", "100",
        "--out-dir", str(grid_dir),
    ], log_path=args.out_dir / "candidate_grid.log")
    _run([
        sys.executable, str(ROOT / runtime["feature_materializer"]),
        "--candidates", str(grid_dir / "target_free_candidate_population.parquet"),
        "--out-dir", str(feature_dir),
        "--candidate-start", first_decision.isoformat(),
        "--history-start", str(runtime["feature_history_start"]),
        "--end-exclusive", end.isoformat(),
        "--side", "long",
    ], log_path=args.out_dir / "features.log")
    grid_manifest = json.loads((grid_dir / "run_manifest.json").read_text())
    features = pd.read_parquet(feature_dir / "canonical120_features.parquet")
    eligible_ids = set(
        pd.read_parquet(
            grid_dir / "eligible_candidates.parquet", columns=["candidate_id"],
        )["candidate_id"]
    )
    scoring_features = features.loc[
        features["candidate_id"].isin(eligible_ids),
    ].copy()
    feature_contract = json.loads(bundle.path("feature_contract").read_text())
    feature_parity_audit = validate_live_feature_frame(
        scoring_features,
        fields=list(feature_contract["base_fields_by_side"]["long"]),
        requirements=dict(bundle.payload["feature_parity"]),
    )
    _run([
        sys.executable, str(ROOT / "scripts" / "score_strict_r3_forward.py"),
        "--schema", "current-v5",
        "--bundle-dir", str(bundle.path("conversion_bundle_dir")),
        "--upstream-bundle-dir", str(bundle.path("upstream_bundle_dir")),
        "--reference-candidates", str(bundle.path("same_model_reference_candidates")),
        "--reference-features", str(bundle.path("same_model_reference_features")),
        "--held-candidates", str(grid_dir / "eligible_candidates.parquet"),
        "--held-features", str(feature_dir / "canonical120_features.parquet"),
        "--out-dir", str(score_dir),
        # The score-piece width is frozen with the canonical producer.  A
        # different width changes chronological K9 support/OOD state inside a
        # piece and would break batch/runtime feature parity.
        "--lockstep-score-chunk-hours", "72",
    ], log_path=args.out_dir / "score.log")
    score_manifest = json.loads((score_dir / "run_manifest.json").read_text())
    score_checks = {
        "target_free_scoring": score_manifest.get("outcome_columns_consumed") == [],
        "no_held_percentiles": score_manifest.get("held_percentile_operations") == 0,
        "exact_lockstep_producer": (
            score_manifest.get("producer_topology") == "exact_lockstep_shared_cutoff"
        ),
        "same_bundle_reference_and_held": bool(
            score_manifest.get("same_bundle_for_reference_and_held")
        ),
        "same_upstream_reference_and_held": bool(
            score_manifest.get("same_upstream_bundle_for_reference_and_held_per_producer")
        ),
        "frozen_geometry": score_manifest.get("geometry_contract") == (
            "one_frozen_oct_dec_2024_geometry_K9_view_temperature_0.25"
        ),
    }
    if not all(score_checks.values()):
        raise AssertionError(f"multi-hour score contract failed: {score_checks}")

    policy = ShadowPortfolioPolicy.from_payload(
        json.loads(bundle.path("portfolio_policy").read_text()),
    )
    trust_bundle = load_cell_day_residual_trust_bundle(
        bundle.path("cell_day_trust_bundle_dir"),
    )
    a4_bundle, a5_calibration = load_a5_bundle(bundle.path("a5_bundle_dir"))
    # These are hypothetical state records, not outcome-labelled positions.
    # Full-H12 expiry is conservative: the audit never uses a later path to
    # free an earlier slot before the timeout is causally known.
    open_records: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    decision_frames: list[pd.DataFrame] = []
    for decision in decisions:
        open_records = [
            record for record in open_records
            if pd.Timestamp(record["expires_at"]) > decision
        ]
        state = ShadowPortfolioState(
            as_of_ts=decision,
            wallet=float(args.wallet),
            open_positions=tuple(
                ShadowOpenPosition(
                    symbol=str(record["symbol"]), side="long",
                    gross_notional=float(record["gross_notional"]),
                    effective_leverage=float(record["effective_leverage"]),
                )
                for record in open_records
            ),
        )
        hour_dir = args.out_dir / "hours" / decision.strftime("%Y%m%dT%H%M%SZ")
        admission_dir = hour_dir / "admission"
        hour_dir.mkdir(parents=True)
        _run([
            sys.executable, str(ROOT / "scripts" / "admit_strict_r3_forward.py"),
            "--resolved-score-label-ledger", str(bundle.path("resolved_score_label_ledger")),
            "--current-predictions", str(score_dir / "predictions.parquet"),
            "--policy-json", str(bundle.path("exit_policy")),
            "--immediate-calibration-index", str(bundle.path("immediate_calibration_index")),
            "--decision-ts", decision.isoformat(),
            "--out-dir", str(admission_dir),
        ], log_path=hour_dir / "admission.log")
        admission_manifest = json.loads((admission_dir / "run_manifest.json").read_text())
        if admission_manifest.get("ev_mapping_vintage_mode") != CANONICAL_ADMISSION:
            raise AssertionError("hour did not use canonical Cell-day admission")
        if int(admission_manifest["mapped_rows"]) != int(admission_manifest["rows"]):
            raise AssertionError("hour contains unmapped actionable candidates")
        admitted = pd.read_parquet(admission_dir / "admitted_predictions.parquet")
        trust_input = admitted.loc[:, ["candidate_id", *trust_bundle.fields]].copy()
        trust_input["raw_expected_bps"] = pd.to_numeric(
            admitted["causal_21d_side_expected_net_bps"], errors="coerce",
        ).to_numpy(float)
        trust = trust_bundle.score(trust_input)
        admitted = admitted.merge(
            trust, on="candidate_id", how="inner", validate="one_to_one",
        )
        posterior = pd.to_numeric(
            admitted["trust_posterior_expected_bps"], errors="coerce",
        )
        admitted["trust_posterior_admitted_ge_50bps"] = (
            posterior.notna() & posterior.ge(50.0)
        )
        a4_input = admitted.loc[:, ["candidate_id", *a4_bundle.fields]].copy()
        a4_input["raw_expected_bps"] = pd.to_numeric(
            admitted["causal_21d_side_expected_net_bps"], errors="coerce",
        ).to_numpy(float)
        a4_score = a4_bundle.score(a4_input)
        admitted = admitted.merge(
            a4_score, on="candidate_id", how="inner", validate="one_to_one",
        )
        a5_score = apply_a5_bounded_10pct(
            admitted, calibration=a5_calibration,
        )
        admitted = admitted.merge(
            a5_score, on="candidate_id", how="inner", validate="one_to_one",
        )
        auction = auction_admitted_snapshot(admitted, state=state, policy=policy)
        accepted = auction["portfolio_accepted"].fillna(False).astype(bool)
        for _, row in auction.loc[accepted].iterrows():
            open_records.append({
                "symbol": str(row["__symbol__"]),
                "gross_notional": float(row["portfolio_gross_notional"]),
                "effective_leverage": float(policy.leverage),
                "expires_at": (decision + pd.Timedelta(hours=12)).isoformat(),
            })
        auction["shadow_action"] = "reject"
        auction.loc[accepted, "shadow_action"] = "hypothetical_entry"
        auction["order_submission_enabled"] = False
        auction["exchange_calls"] = 0
        auction.to_parquet(
            hour_dir / "shadow_decisions.parquet", index=False, compression="zstd",
        )
        decision_frames.append(auction)
        hourly_rows.append({
            "decision_ts": decision,
            "rows": int(len(auction)),
            "mapped": int(auction["causal_21d_side_expected_net_bps"].notna().sum()),
            "raw_cell_day_admitted": int(
                auction["causal_21d_side_admitted_ge_50bps"].fillna(False).sum()
            ),
            "a0_r5_admitted": int(
                auction["trust_posterior_admitted_ge_50bps"].fillna(False).sum()
            ),
            "admitted": int(auction["a5_bounded10_admitted"].fillna(False).sum()),
            "accepted": int(accepted.sum()),
            "open_before": int(len(state.open_positions)),
            "open_after": int(len(open_records)),
            "feature_complete": int(auction["frozen_base_contract_complete"].fillna(False).sum()),
        })

    hourly = pd.DataFrame(hourly_rows)
    all_decisions = pd.concat(decision_frames, ignore_index=True)
    hourly.to_parquet(args.out_dir / "hourly_summary.parquet", index=False, compression="zstd")
    all_decisions.to_parquet(args.out_dir / "shadow_decisions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "mode": "shadow-only",
        "start_decision_ts": start.isoformat(),
        "end_exclusive_decision_ts": end.isoformat(),
        "hours": int(len(hourly)),
        "population_rows": int(grid_manifest["universe_rows"]),
        "eligible_rows": int(grid_manifest["eligible_rows"]),
        "rejected_rows": int(grid_manifest["rejected_rows"]),
        "mapped_rows": int(hourly["mapped"].sum()),
        "admitted_rows": int(hourly["admitted"].sum()),
        "portfolio_accepted_rows": int(hourly["accepted"].sum()),
        "hours_without_mapped_candidates": int(hourly["mapped"].eq(0).sum()),
        "hours_without_admissions": int(hourly["admitted"].eq(0).sum()),
        "hours_without_entries": int(hourly["accepted"].eq(0).sum()),
        "portfolio_state_carry": "causal_conservative_full_H12_no_future_exit_paths",
        "trust_overlay": "frozen R5 nine-month A0 plus bounded-A5 10% inside timestamp top15",
        "conversion_state_prefix_start": first_decision.isoformat(),
        "conversion_state_replayed_from_activation": True,
        "score_checks": score_checks,
        "feature_parity_audit": feature_parity_audit,
        "first_bundle_audit": first_audit,
        "last_bundle_audit": last_audit,
        "current_spread_gate": grid_manifest.get("spread_gate"),
        "future_paths_consumed": [],
        "current_outcomes_consumed": [],
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "hashes": {
            "inference_bundle": _sha(args.inference_bundle),
            "candidate_population": _sha(grid_dir / "target_free_candidate_population.parquet"),
            "eligible_candidates": _sha(grid_dir / "eligible_candidates.parquet"),
            "features": _sha(feature_dir / "canonical120_features.parquet"),
            "scores": _sha(score_dir / "predictions.parquet"),
            "hourly_summary": _sha(args.out_dir / "hourly_summary.parquet"),
            "shadow_decisions": _sha(args.out_dir / "shadow_decisions.parquet"),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
