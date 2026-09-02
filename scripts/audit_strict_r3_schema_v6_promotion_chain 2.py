#!/usr/bin/env python3
"""Audit continuous schema-v6 shadow evidence for explicit live review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _runs(pattern: str, start: pd.Timestamp) -> list[tuple[Path, dict]]:
    values: list[tuple[Path, dict]] = []
    for directory in ROOT.glob(pattern):
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "strict_r3_hourly_shadow_orchestration_v1":
            continue
        if _utc(manifest["decision_ts"]) >= start:
            values.append((directory, manifest))
    values.sort(key=lambda value: _utc(value[1]["decision_ts"]))
    decisions = pd.Series([value[1]["decision_ts"] for value in values])
    if decisions.duplicated().any():
        raise AssertionError("multiple successful schema-v6 checkpoints for one hour")
    return values


def _exit_audit(decision: pd.Timestamp, run: Path) -> dict | None:
    stamp = decision.strftime("%Y%m%dT%H%M%SZ")
    matches = sorted(ROOT.glob(
        f"data_perp/artifacts/strict_r3_schema_v*_exit_replay_audit_{stamp}_v*/exit_replay_audit.json"
    ))
    valid = []
    for path in matches:
        receipt = json.loads(path.read_text())
        if _rooted(receipt.get("run")) == run.resolve():
            valid.append(receipt)
    if len(valid) > 1:
        raise AssertionError(f"multiple exit replay receipts at {decision.isoformat()}")
    return valid[0] if valid else None


def _current_replay_audit(decision: pd.Timestamp, run: Path) -> dict | None:
    stamp = decision.strftime("%Y%m%dT%H%M%SZ")
    matches = sorted(ROOT.glob(
        f"data_perp/artifacts/strict_r3_schema_v*_current_replay_audit_{stamp}_v*/run_manifest.json"
    ))
    valid = []
    for path in matches:
        receipt = json.loads(path.read_text())
        if _rooted(receipt.get("run")) == run.resolve():
            receipt["__receipt_dir"] = str(path.parent.resolve())
            valid.append(receipt)
    if len(valid) > 1:
        raise AssertionError(
            f"multiple independent current replay receipts at {decision.isoformat()}"
        )
    return valid[0] if valid else None


def _derive_current_replay_input_lineage(
    receipt: dict, run: Path,
) -> bool:
    """Verify immutable scorer inputs for old and new replay receipts alike."""
    receipt_dir = Path(str(receipt.get("__receipt_dir") or ""))
    current_inputs = run / "current_hour_inputs"
    stored_features = (
        current_inputs / "canonical120_features.parquet"
        if (current_inputs / "run_manifest.json").is_file()
        else run / "features/canonical120_features.parquet"
    )
    stored_candidates = (
        current_inputs / "eligible_candidates.parquet"
        if (current_inputs / "run_manifest.json").is_file()
        else run / "candidate_grid/eligible_candidates.parquet"
    )
    stored_manifest_path = run / "cycle/score/run_manifest.json"
    replay_manifest_path = receipt_dir / "score_replay/run_manifest.json"
    if not all(path.is_file() for path in (
        stored_features, stored_candidates, stored_manifest_path,
        replay_manifest_path,
    )):
        return False
    stored = json.loads(stored_manifest_path.read_text()).get("source_hashes") or {}
    replay = json.loads(replay_manifest_path.read_text()).get("source_hashes") or {}
    derived = bool(
        _sha(stored_features)
        == stored.get("held_features")
        == replay.get("held_features")
        and _sha(stored_candidates)
        == stored.get("held_candidates")
        == replay.get("held_candidates")
    )
    declared = receipt.get("input_lineage")
    return bool(
        derived
        and (
            not isinstance(declared, dict)
            or bool(declared.get("all_exact"))
        )
    )


def _live_hour_audit(decision: pd.Timestamp, run: Path) -> dict | None:
    stamp = decision.strftime("%Y%m%dT%H%M%SZ")
    matches = sorted(ROOT.glob(
        f"data_perp/artifacts/strict_r3_schema_v*_live_hour_audit_{stamp}_v*/run_manifest.json"
    ))
    valid = []
    for path in matches:
        receipt = json.loads(path.read_text())
        if _rooted(receipt.get("run")) == run.resolve():
            valid.append(receipt)
    if len(valid) > 1:
        raise AssertionError(f"multiple live-hour receipts at {decision.isoformat()}")
    return valid[0] if valid else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.validation_config.read_text())
    start = _utc(config["first_decision_ts"])
    expected_bundle = str(config["inference_bundle"]["sha256"])
    expected_geometry = str(config["geometry_bundle_sha256"])
    expected_artifacts = int(config["expected_artifact_hashes"])
    expected_runtime = int(config["expected_runtime_code_hashes"])
    maximum_audit_age_seconds = float(
        config["promotion_evidence_rules"].get(
            "maximum_audit_completion_age_seconds", 900.0,
        )
    )
    predecessor = _rooted(config["frozen_reconciliation_predecessor"])
    runs = _runs(args.run_glob, start)
    if not runs:
        raise AssertionError("no successful schema-v6 checkpoints")

    rows: list[dict[str, object]] = []
    prior_run = predecessor
    prior_decision = start - pd.Timedelta(hours=1)
    prior_calibration_hash: str | None = None
    for run, manifest in runs:
        decision = _utc(manifest["decision_ts"])
        if decision != prior_decision + pd.Timedelta(hours=1):
            raise AssertionError(f"hourly gap before {decision.isoformat()}")
        if str(manifest["hashes"]["inference_bundle"]) != expected_bundle:
            raise AssertionError(f"bundle changed at {decision.isoformat()}")
        bundle_audit = manifest["inference_bundle_audit"]
        if int(bundle_audit["hashes_verified"]) != expected_artifacts:
            raise AssertionError(f"artifact verification changed at {decision.isoformat()}")
        if int(bundle_audit["runtime_code_hashes_verified"]) != expected_runtime:
            raise AssertionError(f"runtime verification changed at {decision.isoformat()}")
        if str(bundle_audit["geometry_bundle_sha256"]) != expected_geometry:
            raise AssertionError(f"Geometry/K9 changed at {decision.isoformat()}")
        if manifest.get("future_paths_consumed") != []:
            raise AssertionError(f"future path consumed at {decision.isoformat()}")
        if int(manifest.get("exchange_calls", -1)) != 0 or bool(manifest.get("order_submission_enabled")):
            raise AssertionError(f"exchange activity before promotion at {decision.isoformat()}")
        if bool(config["promotion_evidence_rules"].get(
            "require_completion_inside_live_decision_window", True,
        )) and not (
            bool(manifest.get("live_wall_clock_enforced"))
            and bool(manifest.get("completed_within_live_decision_window"))
            and 0.0 <= float(manifest.get("decision_age_at_completion_seconds", float("inf")))
            <= float(manifest.get("live_decision_freshness_seconds", -1))
        ):
            raise AssertionError(
                f"checkpoint completed outside live decision window at {decision.isoformat()}"
            )
        if int(manifest["current_population_rows"]) != 170 or int(manifest["current_population_unique_symbols"]) != 170:
            raise AssertionError(f"universe identity changed at {decision.isoformat()}")
        if not all(bool(v) for v in manifest["current_feature_parity_audit"]["checks"].values()):
            raise AssertionError(f"feature parity failed at {decision.isoformat()}")
        if not all(
            item.get("changed_fields") == [] and float(item.get("max_numeric_delta", 1.0)) == 0.0
            for item in manifest["append_only_overlap_audit"].values()
        ):
            raise AssertionError(f"append-only parity failed at {decision.isoformat()}")
        state_input = _rooted(manifest["portfolio_state_input"])
        expected_state = prior_run / "cycle" / "next_portfolio_state.json"
        if state_input != expected_state.resolve():
            if not bool(manifest.get("portfolio_state_reconciliation")):
                raise AssertionError(f"portfolio state chain failed at {decision.isoformat()}")
            reconciled = json.loads(state_input.read_text())
            provenance = dict(reconciled.get("bridge_provenance") or {})
            if (
                _rooted(provenance.get("shadow_reference")) != expected_state.resolve()
                or str(provenance.get("shadow_reference_sha256")) != _sha(expected_state)
                or int(provenance.get("matched_positions", -1))
                != int(provenance.get("live_execution_state_overlays", -2))
                or int(provenance.get("matched_positions", -1))
                != len(reconciled.get("open_positions") or [])
                or not str(provenance.get("live_state_sha256") or "")
            ):
                raise AssertionError(
                    f"actual-fill portfolio reconciliation lineage failed at "
                    f"{decision.isoformat()}"
                )
        if _sha(state_input) != str(manifest["hashes"]["portfolio_state"]):
            raise AssertionError(f"portfolio input hash failed at {decision.isoformat()}")
        next_state = _rooted(manifest["next_portfolio_state"])
        if _sha(next_state) != str(manifest["hashes"]["next_portfolio_state"]):
            raise AssertionError(f"portfolio output hash failed at {decision.isoformat()}")
        state = json.loads(next_state.read_text())
        if state.get("schema") != "strict_r3_shadow_portfolio_state_v3_adaptive_exit":
            raise AssertionError(f"portfolio state schema changed at {decision.isoformat()}")

        cycle = json.loads((run / "cycle" / "run_manifest.json").read_text())
        if not all(bool(value) for value in cycle["checks"].values()):
            raise AssertionError(f"cycle check failed at {decision.isoformat()}")
        calibration_hash = str(cycle["runtime_resolved_state_sha256"])
        if prior_calibration_hash is not None:
            changed = calibration_hash != prior_calibration_hash
            crossed_day = decision.normalize() != prior_decision.normalize()
            if changed != crossed_day:
                raise AssertionError(f"daily calibration cadence failed at {decision.isoformat()}")
        exits = int(manifest["realized_exit_rows"])
        decisions = pd.read_parquet(run / "cycle" / "shadow_decisions.parquet")
        current_decisions = decisions.loc[
            pd.to_datetime(decisions["__decision_ts__"], utc=True).eq(decision)
        ]
        route_fraction = float(
            pd.to_numeric(current_decisions["base_route_fraction"], errors="raise")
            .dropna().iloc[0]
        )
        if not np.isclose(route_fraction, 0.30, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"base-route fraction changed at {decision.isoformat()}: "
                f"{route_fraction}"
            )
        routed_rows = int(
            current_decisions["base_route_timestamp"].fillna(False).astype(bool).sum()
        )
        receipt = _exit_audit(decision, run.resolve()) if exits else None
        if exits and receipt is None:
            raise AssertionError(f"missing independent exit replay at {decision.isoformat()}")
        if receipt is not None:
            if int(receipt["comparison"]["actual_exit_rows"]) != exits:
                raise AssertionError(f"exit replay count failed at {decision.isoformat()}")
            if not all(bool(value) for value in receipt["checks"].values()):
                raise AssertionError(f"exit replay check failed at {decision.isoformat()}")
        current_replay = _current_replay_audit(decision, run.resolve())
        if current_replay is None:
            raise AssertionError(
                f"missing independent current replay at {decision.isoformat()}"
            )
        if current_replay.get("status") != "pass":
            raise AssertionError(
                f"independent current replay failed at {decision.isoformat()}"
            )
        if not bool(current_replay.get("admitted_identities_exact")):
            raise AssertionError(
                f"independent admission identity mismatch at {decision.isoformat()}"
            )
        if any(
            not bool(current_replay.get(role, {}).get("all_fields_match"))
            for role in ("features", "model_outputs", "admission_and_trust")
        ):
            raise AssertionError(
                f"independent current replay parity failed at {decision.isoformat()}"
            )
        current_replay_input_lineage_exact = _derive_current_replay_input_lineage(
            current_replay, run.resolve(),
        )
        if not current_replay_input_lineage_exact:
            raise AssertionError(
                f"current replay input lineage failed at {decision.isoformat()}"
            )
        if not (
            bool(current_replay.get("live_wall_clock_enforced"))
            and bool(current_replay.get(
                "audit_completed_within_audit_window",
                current_replay.get("audit_completed_within_live_decision_window"),
            ))
            and float(current_replay.get(
                "audit_completion_decision_age_seconds", float("inf"),
            )) <= maximum_audit_age_seconds
        ):
            raise AssertionError(
                f"independent current replay completed late at {decision.isoformat()}"
            )
        live_hour = _live_hour_audit(decision, run.resolve())
        if live_hour is None:
            raise AssertionError(f"missing live-hour audit at {decision.isoformat()}")
        if not (
            live_hour.get("status") == "pass"
            and bool(live_hour.get("live_wall_clock_enforced"))
            and bool(live_hour.get(
                "audit_completed_within_audit_window",
                live_hour.get("audit_completed_within_live_decision_window"),
            ))
            and float(live_hour.get(
                "audit_completion_decision_age_seconds", float("inf"),
            )) <= maximum_audit_age_seconds
            and all(bool(value) for value in live_hour.get("checks", {}).values())
        ):
            raise AssertionError(f"live-hour audit failed at {decision.isoformat()}")

        rows.append({
            "decision_ts": decision,
            "run": str(run),
            "current_rows": int(manifest["current_feature_parity_rows"]),
            "complete_fraction": float(manifest["current_feature_parity_audit"]["all_fields_complete_fraction"]),
            "missing_rows_skipped": int(manifest["row_local_feature_skip_audit"]["skipped_rows"]),
            "routed_rows": routed_rows,
            "admitted_rows": int(manifest["admitted_rows"]),
            "portfolio_accepted_rows": int(manifest["portfolio_accepted_rows"]),
            "realized_exit_rows": exits,
            "independent_exit_replay": receipt is not None,
            "independent_current_replay": True,
            "live_hour_audit": True,
            "current_replay_max_relative_delta": max(
                float(current_replay[role]["maximum_relative_delta"])
                for role in ("features", "model_outputs", "admission_and_trust")
            ),
            "current_replay_input_lineage_exact": current_replay_input_lineage_exact,
            "runtime_resolved_state_sha256": calibration_hash,
            "append_only_max_numeric_delta": max(float(v["max_numeric_delta"]) for v in manifest["append_only_overlap_audit"].values()),
            "exchange_calls": int(manifest["exchange_calls"]),
        })
        prior_run, prior_decision = run.resolve(), decision
        prior_calibration_hash = calibration_hash

    evidence = pd.DataFrame(rows)
    rules = config["promotion_evidence_rules"]
    checks = {
        "continuous_hours": len(evidence) >= int(rules["minimum_continuous_hours_before_review"]),
        "portfolio_accepted_rows": int(evidence["portfolio_accepted_rows"].sum()) >= int(rules["minimum_portfolio_accepted_rows_before_review"]),
        "realized_exit_rows": int(evidence["realized_exit_rows"].sum()) >= int(rules["minimum_realized_exit_rows_before_review"]),
        "feature_completeness": float(evidence["complete_fraction"].min()) >= float(rules["minimum_all120_complete_fraction"]),
        "append_only_parity": float(evidence["append_only_max_numeric_delta"].max()) <= float(rules["maximum_append_only_numeric_delta"]),
        "exchange_calls_zero": int(evidence["exchange_calls"].sum()) == 0,
        "all_exits_independently_replayed": bool(evidence.loc[evidence.realized_exit_rows.gt(0), "independent_exit_replay"].all()),
        "all_hours_independently_replayed": bool(
            evidence["independent_current_replay"].all()
        ),
        "current_replay_within_0_01pct": float(
            evidence["current_replay_max_relative_delta"].max()
        ) <= 0.0001,
        "all_current_replay_input_lineage_exact": bool(
            evidence["current_replay_input_lineage_exact"].all()
        ),
    }
    summary = {
        "schema": "strict_r3_schema_v6_promotion_chain_audit_v1",
        "status": "pass",
        "first_decision_ts": evidence.decision_ts.min().isoformat(),
        "last_decision_ts": evidence.decision_ts.max().isoformat(),
        "continuous_hours": int(len(evidence)),
        "portfolio_accepted_rows": int(evidence.portfolio_accepted_rows.sum()),
        "realized_exit_rows": int(evidence.realized_exit_rows.sum()),
        "minimum_complete_fraction": float(evidence.complete_fraction.min()),
        "maximum_append_only_numeric_delta": float(evidence.append_only_max_numeric_delta.max()),
        "bundle_sha256": expected_bundle,
        "geometry_bundle_sha256": expected_geometry,
        "inference_completion_maximum_age_seconds": 900.0,
        "independent_audit_completion_maximum_age_seconds": maximum_audit_age_seconds,
        "promotion_review_checks": checks,
        "eligible_for_promotion_review": bool(all(checks.values())),
        "production_authorized": False,
        "validation_config": str(args.validation_config),
        "validation_config_sha256": _sha(args.validation_config),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    evidence.to_parquet(args.out / "hourly_chain_audit.parquet", index=False)
    (args.out / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
