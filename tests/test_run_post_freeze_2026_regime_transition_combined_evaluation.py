from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from scripts.run_post_freeze_2026_regime_transition_combined_evaluation import (
    PostFreezeEvaluationError,
    build_evaluation_panel,
    run,
)


def _sha(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _seal(path, manifest, *, schema, status):
    manifest.update({"schema": schema, "status": status,
                     "training_start_utc": "2022-08-30T00:00:00Z",
                     "training_end_exclusive_utc": "2026-01-01T00:00:00Z",
                     "outputs": {"data": {"sha256": _sha(path)}},
                     "frozen_contextual_coefficients": {"status": "FROZEN_2022_2025_CANDIDATE_CONTEXT",
                                                        "training_start_utc": "2022-08-30T00:00:00Z",
                                                        "training_end_exclusive_utc": "2026-01-01T00:00:00Z",
                                                        "arms": ["baseline_context_free", "regime_only", "transition_only", "combined"]}})
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    detached = path.with_suffix(".manifest.sha256")
    detached.write_text(f"{_sha(manifest_path)}  manifest.json\n")
    return manifest_path, detached


def _sources(tmp_path):
    rows = []
    for month in (1, 2, 3):
        for hour in range(8):
            stamp = pd.Timestamp(f"2026-{month:02d}-02 {hour:02d}:00", tz="UTC")
            rows.append({"__ts__": stamp, "__symbol__": "X", "side_name": "long" if hour % 2 else "short",
                         "candidate_id": f"{month}-{hour}", "execution_net_ev_12h": (hour - 3) / 10,
                         "execution_gross_ev_12h": (hour - 2) / 10, "execution_cost_return": .01,
                         "execution_label_available_at": stamp + pd.Timedelta(hours=12), "__first_touch_target_soft__": hour / 8,
                         "baseline": hour, "regime": 7 - hour, "transition": hour % 3, "combined": hour * .5 + (hour % 3)})
    scores = pd.DataFrame(rows)
    score_path = tmp_path / "scores.parquet"; scores.to_parquet(score_path, index=False)
    score_manifest, score_detached = _seal(score_path, {}, schema="precomputed_post_freeze_scores_v1", status="SEALED_POST_FREEZE_2026_SCORES")
    timestamps = sorted(scores.__ts__.unique())
    regime = pd.DataFrame({"__ts__": timestamps, "regime_available_utc": timestamps, "regime_state_p__0": .5, "regime_entropy": .2})
    transition = pd.DataFrame({"__ts__": timestamps, "transition_available_utc": timestamps, "transition_active_probability": .3, "transition_state_entropy": .1})
    regime_path = tmp_path / "regime.parquet"; regime.to_parquet(regime_path, index=False)
    transition_path = tmp_path / "transition.parquet"; transition.to_parquet(transition_path, index=False)
    regime_manifest, regime_detached = _seal(regime_path, {}, schema="authoritative_regime_sidecar_v2", status="SEALED_POST_FREEZE_2026_AUTHORITATIVE")
    transition_manifest, transition_detached = _seal(transition_path, {}, schema="authoritative_transition_sidecar_v2", status="SEALED_POST_FREEZE_2026_AUTHORITATIVE")
    config = {
        "candidate_scores_path": str(score_path), "candidate_scores_manifest_path": str(score_manifest), "candidate_scores_manifest_sidecar_path": str(score_detached),
        "regime_sidecar_path": str(regime_path), "regime_sidecar_manifest_path": str(regime_manifest), "regime_sidecar_manifest_sidecar_path": str(regime_detached),
        "transition_sidecar_path": str(transition_path), "transition_sidecar_manifest_path": str(transition_manifest), "transition_sidecar_manifest_sidecar_path": str(transition_detached),
        "output_dir": str(tmp_path / "output"), "min_mapping_train_rows": 4, "top_fraction": .10,
        "columns": {"net_ev": "execution_net_ev_12h", "gross_ev": "execution_gross_ev_12h", "cost": "execution_cost_return", "label_available_at": "execution_label_available_at", "alpha_target": "__first_touch_target_soft__", "baseline_context_free": "baseline", "regime_only": "regime", "transition_only": "transition", "combined": "combined"},
        "regime": {"timestamp_column": "__ts__", "available_at_column": "regime_available_utc", "input_columns": ["regime_state_p__0", "regime_entropy"]},
        "transition": {"timestamp_column": "__ts__", "available_at_column": "transition_available_utc", "input_columns": ["transition_active_probability", "transition_state_entropy"]},
    }
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    return scores, regime, transition, config, config_path


def test_exact_timestamp_join_preserves_candidate_rows_and_rejects_lifecycle_inputs(tmp_path) -> None:
    scores, regime, transition, config, _ = _sources(tmp_path)
    panel = build_evaluation_panel(scores, regime, transition, config)
    assert len(panel) == len(scores)
    assert panel.candidate_id.tolist() == scores.candidate_id.tolist()
    config["regime"]["input_columns"] = ["regime_lifecycle_phase"]
    with pytest.raises(PostFreezeEvaluationError, match="lifecycle/ex-post phase"):
        build_evaluation_panel(scores, regime, transition, config)


def test_run_evaluates_all_arms_on_same_rows_and_uses_monthly_global_top10(tmp_path) -> None:
    _, _, _, _, config_path = _sources(tmp_path)
    manifest = run(config_path)
    output = config_path.parent / "output"
    summary = pd.read_csv(output / "metrics_summary.csv")
    periods = pd.read_parquet(output / "period_metrics.parquet")
    mapped = pd.read_parquet(output / "causal_mapped_arm_scores.parquet")
    gates = pd.read_csv(output / "baseline_latest_stability_gates.csv")
    assert manifest["status"] == "SEALED_POST_FREEZE_2026_COMBINED_EVALUATION"
    assert summary.arm.tolist() == ["baseline_context_free", "regime_only", "transition_only", "combined"]
    assert mapped.groupby("arm").size().nunique() == 1
    assert set(periods.period_type) == {"month", "week"}
    assert gates.arm.tolist() == ["baseline_context_free", "regime_only", "transition_only", "combined"]
    assert {"gate_aggregate_economics_positive", "gate_latest_month_coverage", "promotion_gate_pass"}.issubset(gates.columns)
    monthly = periods.loc[periods.period_type.eq("month")]
    assert monthly.global_monthly_selected_rows.eq(1).all()  # ceil(8 * 10%)


def test_run_refuses_unsealed_or_non_v2_regime_sidecar(tmp_path) -> None:
    _, _, _, config, config_path = _sources(tmp_path)
    manifest_path = config_path.parent / "regime.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "authoritative_regime_sidecar_v1"
    manifest_path.write_text(json.dumps(manifest))
    detached = config_path.parent / "regime.manifest.sha256"
    detached.write_text(f"{_sha(manifest_path)}  manifest.json\n")
    with pytest.raises(PostFreezeEvaluationError, match=r"authoritative v2\+"):
        run(config_path, materialize=False)


def test_run_refuses_a_contextual_score_contract_that_learns_in_2026(tmp_path) -> None:
    _, _, _, _, config_path = _sources(tmp_path)
    manifest_path = config_path.parent / "scores.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frozen_contextual_coefficients"]["training_end_exclusive_utc"] = "2026-02-01T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest))
    detached = config_path.parent / "scores.manifest.sha256"
    detached.write_text(f"{_sha(manifest_path)}  manifest.json\n")
    with pytest.raises(PostFreezeEvaluationError, match="not strictly frozen within 2022--2025"):
        run(config_path, materialize=False)
