from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_execution_ev_forward_confirmation_readiness import (
    REQUIRED_MODEL_ROLES,
    REQUIRED_SOURCE_CODE,
    SPEC_SCHEMA,
    _identity_hash,
    _stable_payload_hash,
    build_readiness,
    freeze_source_lock,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _write_file(path: Path, value: str = "frozen") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _attach_seal(
    spec: dict[str, object],
    lock_report: dict[str, object],
    tmp_path: Path,
) -> None:
    scored_record = spec["scored_population"]
    population_path = Path(scored_record["path"])
    coverage_path = Path(scored_record["daily_coverage"]["path"])
    frame = pd.read_parquet(population_path)
    seal_path = tmp_path / "preoutcome_seal.json"
    seal = {
        "schema": "execution_ev_forward_preoutcome_seal_v1",
        "status": "sealed_preoutcome_population_not_performance_evidence",
        "source_lock": {"fingerprint": lock_report["lock_fingerprint"]},
        "candidate_identity_sha256": _identity_hash(
            frame,
            scored_record["identity_columns"],
        ),
        "outputs": {
            "scored_population": {"sha256": _sha256(population_path)},
            "daily_coverage": {"sha256": _sha256(coverage_path)},
        },
    }
    seal["seal_fingerprint"] = _stable_payload_hash(seal)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    scored_record["preoutcome_seal"] = _file(seal_path)


def _passing_spec(tmp_path: Path) -> dict[str, object]:
    frozen = _write_file(tmp_path / "frozen.json")
    policy = _write_file(tmp_path / "policy.json")
    spread = _write_file(tmp_path / "spread.csv")
    calibrator = _write_file(tmp_path / "calibrator.joblib")
    feature_contract = _write_file(tmp_path / "features.json")
    decisions = pd.date_range("2026-07-20", periods=14, freq="D", tz="UTC")
    population_rows = 5_000
    timestamps = pd.date_range(
        "2026-07-20", periods=population_rows, freq="4min", tz="UTC"
    )
    population = pd.DataFrame(
        {
            "candidate_id": [f"id_{i}" for i in range(population_rows)],
            "__ts__": timestamps - pd.Timedelta(hours=1),
            "__symbol__": ["BTC", "ETH"] * (population_rows // 2),
            "side_name": ["long", "short"] * (population_rows // 2),
            "execution_decision_utc": timestamps,
            "globally_admitted": [False] * 4_500 + [True] * 500,
            "global_top10_capacity_member": [False] * 4_500 + [True] * 500,
            "mapped_execution_ev": [
                float(i) for i in range(population_rows)
            ],
            "feature_available_at": timestamps,
        }
    )
    population_path = tmp_path / "population.parquet"
    population.to_parquet(population_path, index=False)
    coverage = pd.DataFrame(
        {"utc_date": decisions, "complete": True, "both_sides": True}
    )
    coverage_path = tmp_path / "coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    models = []
    for role in REQUIRED_MODEL_ROLES:
        model = _write_file(tmp_path / "models" / f"{role}.bin", role)
        model_sha256 = _sha256(model)
        lineage = tmp_path / "lineage" / f"{role}.json"
        lineage.parent.mkdir(parents=True, exist_ok=True)
        lineage.write_text(
            json.dumps(
                {
                    "schema": "execution_ev_forward_model_lineage_v1",
                    "models": {
                        role: {
                            "model_sha256": model_sha256,
                            "training_label_end_max_utc": "2026-07-19T15:59:59Z",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        models.append(
            {
                "role": role,
                **_file(model),
                "serialized_final_refit": True,
                "provenance": "frozen_final_refit",
                "training_label_end_max_utc": "2026-07-19T15:59:59Z",
                "feature_contract": _file(feature_contract),
                "training_lineage": _file(lineage),
            }
        )
    return {
        "schema": SPEC_SCHEMA,
        "first_decision_exclusive_utc": "2026-07-19T16:00:00Z",
        "requested_last_decision_utc": "2026-08-02T23:59:59Z",
        "label_horizon_hours": 12,
        "coverage_gate": {
            "minimum_global_topk_rows": 500,
            "minimum_complete_utc_days": 14,
        },
        "frozen_confirmation_manifest": _file(frozen),
        "models": models,
        "scored_population": {
            **_file(population_path),
            "identity_columns": ["candidate_id", "__ts__", "__symbol__", "side_name"],
            "decision_column": "execution_decision_utc",
            "admitted_column": "globally_admitted",
            "coverage_member_column": "global_top10_capacity_member",
            "mapped_score_column": "mapped_execution_ev",
            "availability_columns": ["feature_available_at"],
            "daily_coverage": {
                **_file(coverage_path),
                "date_column": "utc_date",
                "complete_column": "complete",
                "both_sides_column": "both_sides",
            },
        },
        "calibrator": {
            **_file(calibrator),
            "mapping": "causal_recent_side_isotonic_ev_21d",
            "lookback_days": 21,
            "resolved_label_max_utc": "2026-07-19T15:59:59Z",
            "sequential_updates_only_after_resolution": True,
        },
        "policy": _file(policy),
        "spread_baseline": _file(spread),
        "source_code": [
            {"name": name, **_file(frozen)} for name in REQUIRED_SOURCE_CODE
        ],
        "ranking_contract": {
            "mapping": "causal_recent_side_isotonic_ev_21d",
            "scope": "one_pooled_global_top_k_across_timestamps_and_sides",
            "top_k_fraction": 0.1,
            "per_timestamp_quota": False,
            "side_quota": False,
            "asset_quota": False,
            "allow_zero_trades": True,
        },
    }


def test_source_lock_audit_passes_and_can_freeze(tmp_path: Path) -> None:
    spec = _passing_spec(tmp_path)
    report = build_readiness(spec, root=tmp_path, stage="source_lock")
    assert report["ready"]
    assert report["status"] == "ready_to_freeze_source_lock"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    frozen = freeze_source_lock(
        spec,
        report,
        spec_path=spec_path,
        output=lock_path,
    )
    assert frozen["status"] == "frozen_before_forward_outcomes"
    assert lock_path.is_file()


def test_oof_model_and_noncausal_cutoffs_fail_closed(tmp_path: Path) -> None:
    spec = _passing_spec(tmp_path)
    model = spec["models"][0]
    model["serialized_final_refit"] = False
    model["provenance"] = "outer_oof_prediction"
    model["training_label_end_max_utc"] = "2026-07-19T16:00:00Z"
    spec["calibrator"]["resolved_label_max_utc"] = "2026-07-20T00:00:00Z"
    report = build_readiness(spec, root=tmp_path, stage="source_lock")
    assert not report["ready"]
    assert "not_serialized_final_refit:base_long" in report["blockers"]
    assert "oof_or_checkpoint_model_provenance:base_long" in report["blockers"]
    assert "training_label_cutoff_not_before_forward_block:base_long" in report["blockers"]
    assert "calibrator_uses_unresolved_or_forward_labels" in report["blockers"]
    with pytest.raises(ValueError, match="passing source_lock"):
        freeze_source_lock(
            spec,
            report,
            spec_path=tmp_path / "spec.json",
            output=tmp_path / "lock.json",
        )


def test_global_coverage_and_availability_are_required(tmp_path: Path) -> None:
    spec = _passing_spec(tmp_path)
    lock_report = build_readiness(spec, root=tmp_path, stage="source_lock")
    path = Path(spec["scored_population"]["path"])
    frame = pd.read_parquet(path).iloc[:990].copy()
    frame["globally_admitted"] = False
    frame["global_top10_capacity_member"] = False
    frame.loc[frame.index[-99:], "globally_admitted"] = True
    frame.loc[frame.index[-99:], "global_top10_capacity_member"] = True
    frame.loc[0, "feature_available_at"] = frame.loc[0, "execution_decision_utc"] + pd.Timedelta(minutes=1)
    frame.to_parquet(path, index=False)
    spec["scored_population"]["sha256"] = _sha256(path)
    _attach_seal(spec, lock_report, tmp_path)
    # The changed population is not part of the immutable source fingerprint.
    report = build_readiness(
        spec,
        root=tmp_path,
        stage="preoutcome",
        source_lock={
            "schema": "execution_ev_forward_confirmation_source_lock_v1",
            "status": "frozen_before_forward_outcomes",
            "lock_fingerprint": lock_report["lock_fingerprint"],
        },
    )
    assert "minimum_global_topk_coverage_rows_not_met" in report["blockers"]
    assert "feature_available_after_decision:feature_available_at" in report["blockers"]


def test_zero_trades_do_not_block_coverage_when_global_book_is_large(
    tmp_path: Path,
) -> None:
    spec = _passing_spec(tmp_path)
    lock_report = build_readiness(spec, root=tmp_path, stage="source_lock")
    path = Path(spec["scored_population"]["path"])
    frame = pd.read_parquet(path)
    frame["globally_admitted"] = False
    frame["mapped_execution_ev"] = [
        -float(i + 1) for i in range(len(frame))
    ]
    frame["global_top10_capacity_member"] = False
    frame.loc[:499, "global_top10_capacity_member"] = True
    frame.to_parquet(path, index=False)
    spec["scored_population"]["sha256"] = _sha256(path)
    _attach_seal(spec, lock_report, tmp_path)
    report = build_readiness(
        spec,
        root=tmp_path,
        stage="preoutcome",
        source_lock={
            "schema": "execution_ev_forward_confirmation_source_lock_v1",
            "status": "frozen_before_forward_outcomes",
            "lock_fingerprint": lock_report["lock_fingerprint"],
        },
    )
    assert report["ready"]
    assert report["scored_population"]["admitted_rows"] == 0
    assert report["scored_population"]["global_topk_coverage_rows"] == 500


def test_preoutcome_population_rejects_outcome_columns(tmp_path: Path) -> None:
    spec = _passing_spec(tmp_path)
    lock_report = build_readiness(spec, root=tmp_path, stage="source_lock")
    path = Path(spec["scored_population"]["path"])
    frame = pd.read_parquet(path)
    frame["execution_net_ev_12h"] = 0.0
    frame.to_parquet(path, index=False)
    spec["scored_population"]["sha256"] = _sha256(path)
    _attach_seal(spec, lock_report, tmp_path)
    report = build_readiness(
        spec,
        root=tmp_path,
        stage="preoutcome",
        source_lock={
            "schema": "execution_ev_forward_confirmation_source_lock_v1",
            "status": "frozen_before_forward_outcomes",
            "lock_fingerprint": lock_report["lock_fingerprint"],
        },
    )
    assert (
        "scored_preoutcome_contains_outcome_or_label_columns"
        in report["blockers"]
    )


def test_confirmation_requires_exact_identity_paths_and_reconciliation(tmp_path: Path) -> None:
    spec = _passing_spec(tmp_path)
    lock_report = build_readiness(spec, root=tmp_path, stage="source_lock")
    source_lock = {
        "schema": "execution_ev_forward_confirmation_source_lock_v1",
        "status": "frozen_before_forward_outcomes",
        "lock_fingerprint": lock_report["lock_fingerprint"],
    }
    _attach_seal(spec, lock_report, tmp_path)
    population = pd.read_parquet(spec["scored_population"]["path"])
    outcomes = population[
        ["candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc"]
    ].copy()
    outcomes["execution_label_end_utc"] = (
        outcomes["execution_decision_utc"] + pd.Timedelta(hours=12)
    )
    outcomes["exact_1m_path_complete"] = True
    outcomes["execution_gross_ev_12h"] = 0.02
    outcomes["execution_cost_return"] = 0.01
    outcomes["execution_net_ev_12h"] = 0.01
    outcome_path = tmp_path / "outcomes.parquet"
    outcomes.to_parquet(outcome_path, index=False)
    spec["confirmation_outcomes"] = _file(outcome_path)
    spec["canonical_schema_parity"] = {
        "reference": _file(outcome_path),
        "candidate": _file(outcome_path),
    }
    report = build_readiness(
        spec,
        root=tmp_path,
        stage="confirmation",
        source_lock=source_lock,
    )
    assert report["ready"]
    assert report["status"] == "ready_for_one_shot_evaluation"
    outcomes.loc[0, "execution_net_ev_12h"] = -0.5
    outcomes.to_parquet(outcome_path, index=False)
    spec["confirmation_outcomes"]["sha256"] = _sha256(outcome_path)
    spec["canonical_schema_parity"]["reference"]["sha256"] = _sha256(outcome_path)
    spec["canonical_schema_parity"]["candidate"]["sha256"] = _sha256(outcome_path)
    report = build_readiness(
        spec,
        root=tmp_path,
        stage="confirmation",
        source_lock=source_lock,
    )
    assert "confirmation_gross_cost_net_reconciliation_failed" in report["blockers"]


def test_naive_persisted_decision_timestamps_are_rejected(tmp_path: Path) -> None:
    spec = _passing_spec(tmp_path)
    lock_report = build_readiness(spec, root=tmp_path, stage="source_lock")
    path = Path(spec["scored_population"]["path"])
    frame = pd.read_parquet(path)
    frame["execution_decision_utc"] = frame["execution_decision_utc"].dt.tz_localize(None)
    frame.to_parquet(path, index=False)
    spec["scored_population"]["sha256"] = _sha256(path)
    _attach_seal(spec, lock_report, tmp_path)
    report = build_readiness(
        spec,
        root=tmp_path,
        stage="preoutcome",
        source_lock={
            "schema": "execution_ev_forward_confirmation_source_lock_v1",
            "status": "frozen_before_forward_outcomes",
            "lock_fingerprint": lock_report["lock_fingerprint"],
        },
    )
    assert "scored_population_decision_not_utc" in report["blockers"]
