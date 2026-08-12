#!/usr/bin/env python3
"""Run one immutable strict-R3 shadow scoring/admission cycle.

This is deliberately not an exchange process.  It composes the canonical
target-free scorer and Cell-day admission CLI, verifies their manifests, and
emits hypothetical long candidates only.  The module has no exchange client,
credentials, order, cancel, or position mutation path.  A separate, future
promotion must consume these decisions after hourly replay parity passes.
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

from extreme_price_movements.strict_r3_shadow_portfolio import (  # noqa: E402
    ShadowPortfolioPolicy,
    ShadowPortfolioState,
    auction_admitted_snapshot,
)
from extreme_price_movements.strict_r3_inference_bundle import (  # noqa: E402
    SCHEMA_V5,
    StrictR3InferenceBundle,
)
from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    load_cell_day_residual_trust_bundle,
)
from extreme_price_movements.strict_r3_a5_trust import (  # noqa: E402
    apply_a5_bounded_10pct,
    load_a5_bundle,
)

CANONICAL_ADMISSION_MODE = "strict_oof_exact_producer_cell_day_trim15_28d_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"shadow stage failed rc={completed.returncode}; see {log_path}",
        )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--held-candidates", type=Path, required=True)
    parser.add_argument("--held-features", type=Path, required=True)
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("shadow-only",), default="shadow-only",
        help="No order-capable mode exists in this runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable shadow cycle exists: {args.out_dir}")
    decision_ts = pd.Timestamp(args.decision_ts)
    decision_ts = (
        decision_ts.tz_localize("UTC")
        if decision_ts.tzinfo is None else decision_ts.tz_convert("UTC")
    )
    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    bundle_audit = bundle.validate(decision_ts=decision_ts)
    if str(bundle.payload["schema"]) != SCHEMA_V5:
        raise ValueError("canonical shadow inference requires the 28-day bounded-A5 schema-v5 bundle")
    conversion_bundle_dir = bundle.path("conversion_bundle_dir")
    upstream_bundle_dir = bundle.path("upstream_bundle_dir")
    reference_candidates = bundle.path("same_model_reference_candidates")
    reference_features = bundle.path("same_model_reference_features")
    resolved_score_label_ledger = bundle.path("resolved_score_label_ledger")
    immediate_calibration_index = bundle.path("immediate_calibration_index")
    policy_json = bundle.path("exit_policy")
    portfolio_policy_json = bundle.path("portfolio_policy")
    trust_bundle_dir = bundle.path("cell_day_trust_bundle_dir")
    a5_bundle_dir = bundle.path("a5_bundle_dir")
    args.out_dir.mkdir(parents=True)
    score_dir = args.out_dir / "score"
    admission_dir = args.out_dir / "admission"
    _run([
        sys.executable, str(ROOT / "scripts" / "score_strict_r3_forward.py"),
        "--schema", "current-v5",
        "--bundle-dir", str(conversion_bundle_dir),
        "--upstream-bundle-dir", str(upstream_bundle_dir),
        "--reference-candidates", str(reference_candidates),
        "--reference-features", str(reference_features),
        "--held-candidates", str(args.held_candidates),
        "--held-features", str(args.held_features),
        "--out-dir", str(score_dir),
        "--lockstep-score-chunk-hours", "24",
    ], log_path=args.out_dir / "score.log")
    _run([
        sys.executable, str(ROOT / "scripts" / "admit_strict_r3_forward.py"),
        "--resolved-score-label-ledger", str(resolved_score_label_ledger),
        "--current-predictions", str(score_dir / "predictions.parquet"),
        "--policy-json", str(policy_json),
        "--immediate-calibration-index", str(immediate_calibration_index),
        "--decision-ts", decision_ts.isoformat(),
        "--out-dir", str(admission_dir),
    ], log_path=args.out_dir / "admission.log")

    score_manifest = json.loads((score_dir / "run_manifest.json").read_text())
    admission_manifest = json.loads((admission_dir / "run_manifest.json").read_text())
    checks = {
        "shadow_only": args.mode == "shadow-only",
        "order_submission_disabled": True,
        "exchange_calls_zero": True,
        "target_free_scoring": score_manifest.get("outcome_columns_consumed") == [],
        "no_held_percentiles": score_manifest.get("held_percentile_operations") == 0,
        "exact_lockstep_producer": score_manifest.get("producer_topology") == "exact_lockstep_shared_cutoff",
        "same_bundle_reference_and_held": bool(score_manifest.get("same_bundle_for_reference_and_held")),
        "same_upstream_reference_and_held": bool(score_manifest.get("same_upstream_bundle_for_reference_and_held_per_producer")),
        "frozen_geometry": score_manifest.get("geometry_contract") == "one_frozen_oct_dec_2024_geometry_K9_view_temperature_0.25",
        "canonical_cell_day_admission": admission_manifest.get("ev_mapping_vintage_mode") == CANONICAL_ADMISSION_MODE,
        "current_outcomes_absent": admission_manifest.get("current_outcomes_consumed") == [],
        "policy_cost_once": float(admission_manifest.get("policy_lineage", {}).get("cost_bps_once", float("nan"))) == 100.0,
        "all_rows_mapped": int(admission_manifest.get("mapped_rows", -1)) == int(admission_manifest.get("rows", -2)),
        "feature_complete_fraction_meets_bundle_gate": (
            float(score_manifest.get("held_complete_base_contract_fraction", 0.0))
            >= float(bundle.payload["feature_parity"]["minimum_cycle_complete_fraction"])
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"strict-R3 shadow contract failed: {checks}")

    admitted = pd.read_parquet(admission_dir / "admitted_predictions.parquet")
    if "policy_net_bps" in admitted and admitted["policy_net_bps"].notna().any():
        raise AssertionError("shadow decisions unexpectedly contain current outcomes")
    trust_bundle = load_cell_day_residual_trust_bundle(trust_bundle_dir)
    trust_input = admitted.loc[:, ["candidate_id", *trust_bundle.fields]].copy()
    trust_input["raw_expected_bps"] = pd.to_numeric(
        admitted["causal_21d_side_expected_net_bps"], errors="coerce",
    ).to_numpy(float)
    trust = trust_bundle.score(trust_input)
    admitted = admitted.merge(trust, on="candidate_id", how="inner", validate="one_to_one")
    posterior = pd.to_numeric(admitted["trust_posterior_expected_bps"], errors="coerce")
    admitted["trust_posterior_admitted_ge_50bps"] = (
        posterior.notna() & posterior.ge(50.0)
    )
    a4_bundle, a5_calibration = load_a5_bundle(a5_bundle_dir)
    a4_input = admitted.loc[:, ["candidate_id", *a4_bundle.fields]].copy()
    a4_input["raw_expected_bps"] = pd.to_numeric(
        admitted["causal_21d_side_expected_net_bps"], errors="coerce",
    ).to_numpy(float)
    a4 = a4_bundle.score(a4_input)
    admitted = admitted.merge(a4, on="candidate_id", how="inner", validate="one_to_one")
    a5 = apply_a5_bounded_10pct(admitted, calibration=a5_calibration)
    admitted = admitted.merge(a5, on="candidate_id", how="inner", validate="one_to_one")
    portfolio_policy = ShadowPortfolioPolicy.from_payload(
        json.loads(portfolio_policy_json.read_text()),
    )
    portfolio_state = ShadowPortfolioState.from_payload(
        json.loads(args.portfolio_state_json.read_text()),
        expected_as_of_ts=decision_ts,
    )
    auction = auction_admitted_snapshot(
        admitted, state=portfolio_state, policy=portfolio_policy,
    )
    selected = admitted["a5_bounded10_admitted"].fillna(False).astype(bool)
    decisions = auction.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "final_score", "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "frozen_base_contract_complete",
        "trust_posterior_expected_bps", "trust_residual_q25_bps",
        "trust_p_map_overestimate_100bps", "trust_effective_support",
        "trust_risk_corroborated", "trust_authority",
        "trust_corrected_expected_net_bps", "auction_rank_adjustment_bps",
        "trust_posterior_admitted_ge_50bps",
        "a4_raw_expected_bps", "a4_raw_predictive_sd_bps",
        "a4_effective_support", "a4_p_ev_positive_raw",
        "a5_calibrated_expected_bps", "a5_calibrated_p_positive",
        "a5_bounded10_expected_bps", "a5_timestamp_top15",
        "a5_bounded10_available", "a5_bounded10_admitted",
        "ev_mapping_vintage_mode", "geometry_bundle_sha256",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "portfolio_accepted", "portfolio_rejection_reason",
        "portfolio_priority_rank", "portfolio_initial_margin",
        "portfolio_gross_notional", "portfolio_wallet",
        "portfolio_open_positions_before", "portfolio_committed_margin_before",
        "portfolio_margin_cap", "portfolio_policy_schema", "portfolio_state_schema",
    ]].copy()
    decisions["shadow_action"] = "reject"
    decisions.loc[
        decisions["portfolio_accepted"].fillna(False).astype(bool), "shadow_action",
    ] = "hypothetical_entry"
    decisions["order_submission_enabled"] = False
    decisions["exchange_calls"] = 0
    decisions.to_parquet(
        args.out_dir / "shadow_decisions.parquet", index=False, compression="zstd",
    )
    manifest = {
        "schema": "strict_r3_cell_day_trim15_shadow_cycle_v2",
        "mode": "shadow-only",
        "decision_ts": decision_ts.isoformat(),
        "checks": checks,
        "inference_bundle_audit": bundle_audit,
        "rows": int(len(decisions)),
        "feature_complete_rows": int(decisions["frozen_base_contract_complete"].fillna(False).sum()),
        "mapped_rows": int(decisions["causal_21d_side_expected_net_bps"].notna().sum()),
        "admitted_rows": int(selected.sum()),
        "portfolio_accepted_rows": int(decisions["portfolio_accepted"].sum()),
        "portfolio_open_positions_before": int(len(portfolio_state.open_positions)),
        "portfolio_wallet": float(portfolio_state.wallet),
        "order_submission_enabled": False,
        "exchange_calls": 0,
        "score_manifest_sha256": _sha(score_dir / "run_manifest.json"),
        "admission_manifest_sha256": _sha(admission_dir / "run_manifest.json"),
        "inference_bundle_sha256": _sha(args.inference_bundle),
        "policy_json_sha256": _sha(policy_json),
        "portfolio_policy_json_sha256": _sha(portfolio_policy_json),
        "portfolio_state_json_sha256": _sha(args.portfolio_state_json),
        "calibration_index_sha256": _sha(immediate_calibration_index),
        "cell_day_trust_bundle_manifest_sha256": _sha(
            trust_bundle_dir / "run_manifest.json"
        ),
        "a5_bundle_manifest_sha256": _sha(a5_bundle_dir / "run_manifest.json"),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
