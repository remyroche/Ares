#!/usr/bin/env python3
"""Audit a continuous chain of immutable strict-R3 hourly shadow receipts.

This is an evidence auditor only.  It never scores candidates, reads an
exchange account, changes portfolio state, or submits orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _assert_a5_availability_matches_base_contract(
    current_decisions: pd.DataFrame,
    *,
    decision: pd.Timestamp,
) -> None:
    """Require A5 availability exactly on complete frozen-base rows.

    Incomplete rows retain diagnostic upstream scores, but schema-v5 makes
    both A5 and the portfolio auction fail closed.  Requiring A5 on every
    mapped row would reject the very defense-in-depth contract this auditor
    is intended to prove.
    """
    base_complete = (
        current_decisions["frozen_base_contract_complete"]
        .fillna(False)
        .astype(bool)
    )
    a5_available = (
        current_decisions["a5_bounded10_available"]
        .fillna(False)
        .astype(bool)
    )
    if not a5_available.loc[base_complete].all():
        raise AssertionError(
            f"A5 unavailable on a complete base row at {decision.isoformat()}"
        )
    if a5_available.loc[~base_complete].any():
        raise AssertionError(
            f"A5 available on an incomplete base row at {decision.isoformat()}"
        )


def _successful_runs(pattern: str, start: pd.Timestamp) -> list[tuple[Path, dict]]:
    runs: list[tuple[Path, dict]] = []
    for directory in sorted(ROOT.glob(pattern)):
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "strict_r3_hourly_shadow_orchestration_v1":
            continue
        decision = _utc(manifest["decision_ts"])
        if decision >= start:
            runs.append((directory, manifest))
    runs.sort(key=lambda item: _utc(item[1]["decision_ts"]))
    duplicate = pd.Series([item[1]["decision_ts"] for item in runs]).duplicated(keep=False)
    if duplicate.any():
        values = pd.Series([item[1]["decision_ts"] for item in runs])[duplicate].tolist()
        raise AssertionError(f"multiple successful receipts for decision hours: {values}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-config", type=Path,
        default=(
            ROOT / "config" /
            "strict_r3_untouched_forward_validation_homogeneous28_20260813_v2.json"
        ),
    )
    parser.add_argument(
        "--run-glob",
        default=(
            "data_perp/artifacts/"
            "strict_r3_untouched_homogeneous28_hourly_20260813T*"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.validation_config.read_text())
    start = _utc(config["first_decision_ts"])
    expected_bundle = str(config["inference_bundle"]["sha256"])
    expected_universe_rows = int(config.get("frozen_universe_rows", 170))
    frozen_predecessor = Path(str(config["frozen_reconciliation_predecessor"]))
    runs = _successful_runs(args.run_glob, start)
    if not runs:
        raise AssertionError("no successful untouched hourly receipts")

    rows: list[dict[str, object]] = []
    prior_dir: Path | None = None
    prior_decision: pd.Timestamp | None = None
    geometry_hash: str | None = None
    for directory, manifest in runs:
        decision = _utc(manifest["decision_ts"])
        audit = dict(manifest["inference_bundle_audit"])
        current = dict(manifest["current_feature_parity_audit"])
        overlap = dict(manifest["append_only_overlap_audit"])
        source_manifest = json.loads(
            (directory / "candidate_grid" / "run_manifest.json").read_text()
        )
        decisions = pd.read_parquet(directory / "cycle" / "shadow_decisions.parquet")
        current_decisions = decisions.loc[
            pd.to_datetime(decisions["__decision_ts__"], utc=True).eq(decision)
        ]
        admission = pd.read_parquet(
            directory / "cycle" / "admission" / "causal_21d_admission_audit.parquet"
        )
        admitted_predictions = pd.read_parquet(
            directory / "cycle" / "admission" / "admitted_predictions.parquet"
        )
        predictions = pd.read_parquet(directory / "cycle" / "score" / "predictions.parquet")
        current_predictions = predictions.loc[
            pd.to_datetime(predictions["__decision_ts__"], utc=True).eq(decision)
        ]

        if prior_decision is not None and decision != prior_decision + pd.Timedelta(hours=1):
            raise AssertionError(f"hourly gap before {decision.isoformat()}")
        prefix_source = Path(str(source_manifest.get("immutable_prefix_source", "")))
        expected_source = prior_dir or frozen_predecessor
        if _rooted(prefix_source) != _rooted(expected_source):
            raise AssertionError(
                f"{decision.isoformat()} prefix source {prefix_source} != {expected_source}"
            )
        if manifest["hashes"]["inference_bundle"] != expected_bundle:
            raise AssertionError(f"bundle changed at {decision.isoformat()}")
        if int(audit.get("hashes_verified", 0)) != 23:
            raise AssertionError(f"artifact hash verification failed at {decision.isoformat()}")
        if int(audit.get("runtime_code_hashes_verified", 0)) != 28:
            raise AssertionError(f"runtime code verification failed at {decision.isoformat()}")
        local_geometry = str(audit["geometry_bundle_sha256"])
        geometry_hash = geometry_hash or local_geometry
        if local_geometry != geometry_hash:
            raise AssertionError(f"Geometry/K9 changed at {decision.isoformat()}")
        if manifest.get("future_paths_consumed") != []:
            raise AssertionError(f"future paths consumed at {decision.isoformat()}")
        if int(manifest.get("exchange_calls", -1)) != 0 or bool(
            manifest.get("order_submission_enabled", True)
        ):
            raise AssertionError(f"exchange activity present at {decision.isoformat()}")
        expected_state_source = (
            prior_dir or frozen_predecessor
        ) / "cycle" / "next_portfolio_state.json"
        state_input = Path(str(manifest.get("portfolio_state_input", "")))
        if _rooted(state_input) != _rooted(expected_state_source):
            raise AssertionError(
                f"{decision.isoformat()} portfolio state {state_input} "
                f"!= {expected_state_source}"
            )
        if not bool(manifest.get("portfolio_state_chained_from_previous")):
            raise AssertionError(f"portfolio state was not chained at {decision.isoformat()}")
        state_input_path = _rooted(state_input)
        next_state = Path(str(manifest.get("next_portfolio_state", "")))
        expected_next_state = directory / "cycle" / "next_portfolio_state.json"
        if _rooted(next_state) != _rooted(expected_next_state):
            raise AssertionError(
                f"{decision.isoformat()} next state {next_state} != {expected_next_state}"
            )
        next_state_path = _rooted(next_state)
        if not state_input_path.exists() or not next_state_path.exists():
            raise AssertionError(f"portfolio state artifact missing at {decision.isoformat()}")
        if _sha(state_input_path) != str(manifest["hashes"]["portfolio_state"]):
            raise AssertionError(f"input portfolio state hash failed at {decision.isoformat()}")
        if _sha(next_state_path) != str(manifest["hashes"]["next_portfolio_state"]):
            raise AssertionError(f"next portfolio state hash failed at {decision.isoformat()}")
        input_state = json.loads(state_input_path.read_text())
        output_state = json.loads(next_state_path.read_text())
        if input_state.get("schema") != "strict_r3_shadow_portfolio_state_v2":
            raise AssertionError(f"input portfolio state is not schema-v2 at {decision.isoformat()}")
        if output_state.get("schema") != "strict_r3_shadow_portfolio_state_v2":
            raise AssertionError(f"next portfolio state is not schema-v2 at {decision.isoformat()}")
        if _utc(input_state["as_of_ts"]) != decision:
            raise AssertionError(f"input portfolio state timestamp failed at {decision.isoformat()}")
        if _utc(output_state["as_of_ts"]) != decision + pd.Timedelta(hours=1):
            raise AssertionError(f"next portfolio state timestamp failed at {decision.isoformat()}")
        input_symbols = [str(value["symbol"]) for value in input_state["open_positions"]]
        output_symbols = [str(value["symbol"]) for value in output_state["open_positions"]]
        if len(input_symbols) != len(set(input_symbols)) or len(output_symbols) != len(set(output_symbols)):
            raise AssertionError(f"duplicate open symbol in portfolio state at {decision.isoformat()}")
        cycle_manifest = json.loads((directory / "cycle" / "run_manifest.json").read_text())
        exits = pd.read_parquet(directory / "cycle" / "shadow_exits.parquet")
        realized_exit_rows = int(manifest.get("realized_exit_rows", -1))
        if realized_exit_rows != len(exits) or realized_exit_rows != int(
            cycle_manifest.get("realized_exit_rows", -2)
        ):
            raise AssertionError(f"realized exit count failed at {decision.isoformat()}")
        if len(exits):
            gross = pd.to_numeric(exits["gross_bps"], errors="raise").to_numpy(float)
            cost = pd.to_numeric(exits["cost_bps"], errors="raise").to_numpy(float)
            net = pd.to_numeric(exits["net_bps"], errors="raise").to_numpy(float)
            if not ((cost == 100.0).all() and (abs(net - (gross - cost)) <= 1e-9).all()):
                raise AssertionError(f"exit cost contract failed at {decision.isoformat()}")
        open_before = int(manifest.get("portfolio_open_positions_before", -1))
        open_after = int(manifest.get("portfolio_open_positions_after", -1))
        accepted = int(manifest.get("portfolio_accepted_rows", -1))
        if len(input_symbols) != open_before + realized_exit_rows:
            raise AssertionError(f"portfolio exit progression failed at {decision.isoformat()}")
        if open_after != open_before + accepted or open_after != len(output_symbols):
            raise AssertionError(f"portfolio entry progression failed at {decision.isoformat()}")
        if not abs(float(output_state["wallet"]) - float(cycle_manifest["portfolio_wallet"])) <= 1e-9:
            raise AssertionError(f"portfolio wallet progression failed at {decision.isoformat()}")
        if int(manifest["current_population_rows"]) != expected_universe_rows or int(
            manifest["current_population_unique_symbols"]
        ) != expected_universe_rows:
            raise AssertionError(f"current universe identity failed at {decision.isoformat()}")
        if not bool(manifest.get("complete_universe_features_before_actionability_filter")):
            raise AssertionError(f"complete-universe feature ordering failed at {decision.isoformat()}")
        if not bool(manifest.get("conversion_state_replayed_from_activation")):
            raise AssertionError(f"conversion prefix replay failed at {decision.isoformat()}")
        if not bool(manifest.get("current_spread_gate")):
            raise AssertionError(f"current spread gate failed at {decision.isoformat()}")
        if source_manifest.get("entry") != "first 15-minute open at signal close + one hour":
            raise AssertionError(f"entry contract changed at {decision.isoformat()}")
        if not all(bool(value) for value in current["checks"].values()):
            raise AssertionError(f"feature parity failed at {decision.isoformat()}")
        if int(manifest["mapped_rows"]) != int(manifest["current_feature_parity_rows"]):
            raise AssertionError(f"mapping coverage failed at {decision.isoformat()}")
        if len(current_decisions) != int(manifest["current_feature_parity_rows"]):
            raise AssertionError(f"decision identity failed at {decision.isoformat()}")
        if len(current_predictions) != len(current_decisions):
            raise AssertionError(f"prediction identity failed at {decision.isoformat()}")
        if not current_predictions["stack_is_prequential"].fillna(False).astype(bool).all():
            raise AssertionError(f"non-prequential prediction at {decision.isoformat()}")
        base_complete = current_decisions["frozen_base_contract_complete"].fillna(False).astype(bool)
        incomplete_decisions = current_decisions.loc[~base_complete]
        if len(incomplete_decisions):
            # The scorer deliberately preserves an imputed diagnostic score for
            # rows with a partially unavailable frozen input contract.  Such a
            # row is executable only if the admission layer fails it closed and
            # the portfolio never accepts it.  Requiring 120/120 on every
            # mapped diagnostic row would contradict that declared runtime
            # contract and hide the rejection evidence.
            if incomplete_decisions["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool).any():
                raise AssertionError(
                    f"incomplete base row admitted at {decision.isoformat()}"
                )
            if incomplete_decisions["portfolio_accepted"].fillna(False).astype(bool).any():
                raise AssertionError(
                    f"incomplete base row reached portfolio at {decision.isoformat()}"
                )
            incomplete_ids = set(incomplete_decisions["candidate_id"].astype(str))
            reasons = admitted_predictions.loc[
                admitted_predictions["candidate_id"].astype(str).isin(incomplete_ids),
                "admission_rejection_reason",
            ].fillna("").astype(str)
            if len(reasons) != len(incomplete_ids):
                raise AssertionError(
                    f"incomplete base row missing from admission receipt at {decision.isoformat()}"
                )
            if not reasons.eq("frozen_base_contract_incomplete").all():
                raise AssertionError(
                    f"incomplete base row lacks explicit rejection at {decision.isoformat()}"
                )
        _assert_a5_availability_matches_base_contract(
            current_decisions,
            decision=decision,
        )
        if not admission["strictly_prior_resolved"].fillna(False).astype(bool).all():
            raise AssertionError(f"admission used non-prior labels at {decision.isoformat()}")
        if pd.to_datetime(admission["reference_max_label_available_ts"], utc=True).max() >= decision:
            raise AssertionError(f"admission reference reaches held decision at {decision.isoformat()}")
        for name in ("candidate_population", "eligible_candidates", "features", "predictions"):
            local = overlap[name]
            if local.get("changed_fields") != [] or float(local["max_numeric_delta"]) != 0.0:
                raise AssertionError(f"append-only {name} failed at {decision.isoformat()}")

        rows.append({
            "decision_ts": decision,
            "run_dir": str(directory),
            "population_rows": int(manifest["current_population_rows"]),
            "actionable_rows": int(manifest["current_feature_parity_rows"]),
            "complete_base_contract_rows": int(base_complete.sum()),
            "incomplete_base_contract_rows": int((~base_complete).sum()),
            "all120_complete_fraction": float(current["all_fields_complete_fraction"]),
            "mapped_rows": int(manifest["mapped_rows"]),
            "admitted_rows": int(manifest["admitted_rows"]),
            "portfolio_accepted_rows": int(manifest["portfolio_accepted_rows"]),
            "portfolio_open_positions_before": open_before,
            "portfolio_open_positions_after": open_after,
            "realized_exit_rows": realized_exit_rows,
            "portfolio_wallet": float(output_state["wallet"]),
            "portfolio_state_chained": True,
            "mapped_curve_min_bps": float(admission["mapped_curve_min_bps"].min()),
            "mapped_curve_max_bps": float(admission["mapped_curve_max_bps"].max()),
            "reference_max_label_available_ts": pd.to_datetime(
                admission["reference_max_label_available_ts"], utc=True
            ).max(),
            "append_only_max_numeric_delta": max(
                float(value["max_numeric_delta"]) for value in overlap.values()
            ),
            "exchange_calls": int(manifest["exchange_calls"]),
        })
        prior_dir, prior_decision = directory, decision

    evidence = pd.DataFrame(rows)
    promotion_rules = dict(config.get("promotion_evidence_rules", {}))
    minimum_hours = int(promotion_rules.get("minimum_continuous_hours_before_review", 168))
    minimum_accepted = int(
        promotion_rules.get("minimum_portfolio_accepted_rows_before_review", 30)
    )
    minimum_exits = int(promotion_rules.get("minimum_realized_exit_rows_before_review", 20))
    minimum_complete = float(
        promotion_rules.get("minimum_all120_complete_fraction", 0.9)
    )
    maximum_delta = float(
        promotion_rules.get("maximum_append_only_numeric_delta", 0.0)
    )
    review_checks = {
        "continuous_hours": bool(len(evidence) >= minimum_hours),
        "portfolio_accepted_rows": bool(
            evidence["portfolio_accepted_rows"].sum() >= minimum_accepted
        ),
        "realized_exit_rows": bool(evidence["realized_exit_rows"].sum() >= minimum_exits),
        "feature_completeness": bool(
            evidence["all120_complete_fraction"].min() >= minimum_complete
        ),
        "append_only_parity": bool(
            evidence["append_only_max_numeric_delta"].max() <= maximum_delta
        ),
        "exchange_calls_zero": bool(evidence["exchange_calls"].sum() == 0),
    }
    eligible_for_review = bool(all(review_checks.values()))
    summary = {
        "schema": "strict_r3_untouched_shadow_chain_audit_v1",
        "status": "pass",
        "first_decision_ts": evidence["decision_ts"].min().isoformat(),
        "last_decision_ts": evidence["decision_ts"].max().isoformat(),
        "continuous_hours": int(len(evidence)),
        "bundle_sha256": expected_bundle,
        "geometry_bundle_sha256": geometry_hash,
        "population_rows": int(evidence["population_rows"].sum()),
        "actionable_rows": int(evidence["actionable_rows"].sum()),
        "complete_base_contract_rows": int(evidence["complete_base_contract_rows"].sum()),
        "incomplete_base_contract_rows": int(evidence["incomplete_base_contract_rows"].sum()),
        "mapped_rows": int(evidence["mapped_rows"].sum()),
        "admitted_rows": int(evidence["admitted_rows"].sum()),
        "portfolio_accepted_rows": int(evidence["portfolio_accepted_rows"].sum()),
        "realized_exit_rows": int(evidence["realized_exit_rows"].sum()),
        "last_portfolio_open_positions": int(evidence["portfolio_open_positions_after"].iloc[-1]),
        "last_portfolio_wallet": float(evidence["portfolio_wallet"].iloc[-1]),
        "all_portfolio_states_chained": bool(evidence["portfolio_state_chained"].all()),
        "minimum_all120_complete_fraction": float(evidence["all120_complete_fraction"].min()),
        "maximum_append_only_numeric_delta": float(
            evidence["append_only_max_numeric_delta"].max()
        ),
        "exchange_calls": int(evidence["exchange_calls"].sum()),
        "promotion_review_thresholds": {
            "minimum_continuous_hours": minimum_hours,
            "minimum_portfolio_accepted_rows": minimum_accepted,
            "minimum_realized_exit_rows": minimum_exits,
            "minimum_all120_complete_fraction": minimum_complete,
            "maximum_append_only_numeric_delta": maximum_delta,
        },
        "promotion_review_checks": review_checks,
        "eligible_for_promotion_review": eligible_for_review,
        "production_authorized": False,
        "promotion_note": (
            "eligible for explicit production review; exchange remains disabled"
            if eligible_for_review else
            "chain correctness only; duration and/or realised outcomes remain insufficient"
        ),
        "validation_config": str(args.validation_config),
        "validation_config_sha256": _sha(args.validation_config),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    evidence.to_parquet(args.out / "hourly_chain_audit.parquet", index=False)
    (args.out / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
