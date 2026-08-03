"""Tests for the read-only C0--C8 Stage-C readiness audit."""

from __future__ import annotations

import json

import pandas as pd

from extreme_price_movements.stage_c_feature_group_readiness import ARM_TO_GROUP, AuditInputs, build_readiness


def _inputs(tmp_path, *, bad_oof: bool = False) -> AuditInputs:
    panel = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "decision_ts": pd.to_datetime(["2024-04-02T00:00:00Z", "2024-04-03T00:00:00Z"]),
        "feature_available_ts": pd.to_datetime(["2024-04-02T00:00:00Z", "2024-04-02T23:00:00Z"]),
        "f1": [1.0, 2.0], "f2": [1.0, 2.0], "f3": [1.0, 2.0], "f6": [1.0, 2.0], "f8": [1.0, 2.0],
    })
    panel_path = tmp_path / "panel.parquet"; panel.to_parquet(panel_path, index=False)
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps({
        "F1_price_continuation_exhaustion": ["f1"], "F2_volume_liquidity_proxies": ["f2"],
        "F3_volatility_transition": ["f3"], "F4_oi_dynamics": [], "F5_funding_crowding": [],
        "F6_cross_sectional_confirmation": ["f6"], "F7_causal_regime_transition": [],
        "F8_predeclared_composites": ["f8"],
        "blocked": {"F4": "no native OI timestamps", "F5": "no native funding timestamps", "F7": "no OOF sidecar"},
    }))
    lineage = pd.DataFrame([
        {"feature_name": "e15", "feature_group": "F0_existing_E15_control"},
        *({"feature_name": name, "feature_group": group} for name, group in (("f1", "F1_price_continuation_exhaustion"), ("f2", "F2_volume_liquidity_proxies"), ("f3", "F3_volatility_transition"), ("f6", "F6_cross_sectional_confirmation"), ("f8", "F8_predeclared_composites"))),
    ])
    lineage_path = tmp_path / "lineage.parquet"; lineage.to_parquet(lineage_path, index=False)
    coverage = pd.DataFrame([
        {"month": "2024-04", "side": "long", "source_symbol": "A", "rows": 2, "non_missing": 2, "feature_name": name, "missing_rate": 0.0}
        for name in ("f1", "f2", "f3", "f6", "f8")
    ])
    coverage_path = tmp_path / "coverage.parquet"; coverage.to_parquet(coverage_path, index=False)
    usable = ["C0", "C1", "C2", "C3", "C6", "C8"]
    identity = pd.DataFrame({"arm": usable, "identical_to_c0": [True] * len(usable)})
    identity_path = tmp_path / "identity.parquet"; identity.to_parquet(identity_path, index=False)
    records = []
    for month in ("2024-04", "2024-05", "2024-06", "2024-07"):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        records.append({"split": "development_oof", "fold": month, "fold_start_utc": start, "train_decision_ts_max": start - pd.Timedelta(hours=13 if not bad_oof else 11), "train_label_available_ts_max": start - pd.Timedelta(hours=1), "purge_embargo_hours": 12})
    stability_path = tmp_path / "stability.parquet"; pd.DataFrame(records).to_parquet(stability_path, index=False)
    results = pd.DataFrame([
        {"arm": arm, "split": "final_oos", "scope": "month", "month": month}
        for arm in usable for month in ("2024-08", "2024-09", "2024-10", "2024-11")
    ])
    results_path = tmp_path / "results.parquet"; results.to_parquet(results_path, index=False)
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps({"schema": "stage_c_conditional_retention_ablation_v4"}))
    return AuditInputs(panel_path, groups_path, lineage_path, coverage_path, identity_path, stability_path, manifest_path, results_path)


def test_readiness_reuses_only_sealed_safe_groups_and_blocks_unproven_sources(tmp_path):
    readiness, coverage, source, report = build_readiness(inputs=_inputs(tmp_path))
    assert set(readiness.arm) == set(ARM_TO_GROUP)
    assert readiness.loc[readiness.arm.isin(["C0", "C1", "C2", "C3", "C6", "C8"]), "availability_status"].eq("REUSABLE_SEALED_V4").all()
    assert readiness.loc[readiness.arm.isin(["C4", "C5", "C7"]), "availability_status"].eq("SOURCE_BLOCKED").all()
    assert not readiness.new_model_fit_started.any()
    assert readiness.loc[readiness.arm.eq("C1"), "can_run_now"].item()
    assert not readiness.loc[readiness.arm.eq("C4"), "can_run_now"].item()
    assert not coverage.empty and not source.empty
    assert report["passed"]


def test_readiness_fails_closed_when_h12_purge_is_not_strict(tmp_path):
    _, _, _, report = build_readiness(inputs=_inputs(tmp_path, bad_oof=True))
    assert not report["passed"]
    assert not report["checks"]["stage1_strict_development_oof_h12_purge"]
