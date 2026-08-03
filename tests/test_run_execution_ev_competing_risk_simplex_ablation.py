from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_execution_ev_competing_risk_simplex_ablation import (
    CLASS_NAMES,
    LABEL_SCHEMA,
    PAYOFF_GRIDS,
    SIMPLEX_GRIDS,
    apply_offset_temperature_calibrator,
    conditional_class_positions,
    compose_expected_net,
    economic_metrics,
    fit_offset_temperature_calibrator,
    fit_temperature,
    geometry_contract,
    joint_hpo_combinations,
    multiclass_metrics,
    score_bridge_metrics,
    soft_simplex_targets,
    temperature_scale_probabilities,
    true_class_oracle_score,
    _inner_calibration_split,
    _validate_label_artifact,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-07-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["A", "B", "A", "B"],
            "side_name": ["long", "short", "long", "short"],
            "candidate_id": ["a", "b", "c", "d"],
            "timeout": [0, 0, 1, 1],
            "adverse_first": [1, 0, 0, 0],
            "clean_economic_favorable_first": [0, 1, 0, 0],
            "timeout_soft_timeout_viability": [np.nan, np.nan, 0.4, 0.5],
            "timeout_soft_adverse_viability": [np.nan, np.nan, 0.4, 0.2],
            "timeout_soft_clean_economic_favorable_viability": [np.nan, np.nan, 0.2, 0.3],
            "execution_gross_ev_12h": [-0.02, 0.04, -0.01, 0.01],
            "execution_cost_return": [0.01, 0.01, 0.01, 0.01],
            "execution_net_ev_12h": [-0.03, 0.03, -0.02, 0.0],
            "execution_mfe_return_12h": [0.01, 0.08, 0.00, 0.03],
            "first_favorable_minute": [np.nan, 4.0, np.nan, 9.0],
            "first_adverse_minute": [2.0, np.nan, np.nan, np.nan],
        }
    )


def test_expected_net_composition_subtracts_row_cost_once() -> None:
    probability = np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
    payoff = np.array([[-0.02, -0.10, 0.08], [-0.01, -0.10, 0.05]])
    result = compose_expected_net(probability, payoff, np.array([0.01, 0.02]))
    assert np.allclose(result, [0.2 * -0.02 + 0.3 * -0.10 + 0.5 * 0.08 - 0.01, 0.8 * -0.01 + 0.1 * -0.10 + 0.1 * .05 - .02])
    with pytest.raises(ValueError, match="closed simplex"):
        compose_expected_net(np.array([[.2, .2, .2]]), payoff[:1], np.array([.01]))


def test_true_class_oracle_uses_only_train_class_means_and_one_cost() -> None:
    # The revealed class is intentional: this is a named, nonpredictive upper
    # bound, not a probability model or an HPO candidate.
    score = true_class_oracle_score(np.array([0, 2, 1]), np.array([-.01, -.12, .08]), np.array([.01, .02, .03]))
    assert np.allclose(score, [-.02, .06, -.15])
    with pytest.raises(ValueError, match="three train-only"):
        true_class_oracle_score(np.array([0]), np.array([.0, .1]), np.array([.01]))


def test_temperature_preserves_simplex_and_is_trainable() -> None:
    probability = np.array([[.80, .10, .10], [.10, .80, .10], [.10, .10, .80]] * 50)
    target = np.array([0, 1, 2] * 50)
    calibrated = temperature_scale_probabilities(probability, 1.5)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.isfinite(calibrated).all()
    assert .5 <= fit_temperature(probability, target) <= 3.0
    with pytest.raises(ValueError, match="positive temperature"):
        temperature_scale_probabilities(probability, 0.0)


def test_offset_temperature_corrects_class_prior_shift_beyond_scalar_temperature() -> None:
    # All rows have the same raw simplex.  A scalar can only change its common
    # sharpness, whereas two anchored offsets can recover the different
    # calibration-set class prevalence without touching evaluation outcomes.
    probability = np.tile(np.array([.70, .20, .10]), (200, 1))
    target = np.array([0] * 90 + [1] * 90 + [2] * 20)
    scalar_temperature = fit_temperature(probability, target)
    scalar = temperature_scale_probabilities(probability, scalar_temperature)
    calibrator = fit_offset_temperature_calibrator(probability, target)
    calibrated = apply_offset_temperature_calibrator(probability, calibrator)
    assert calibrator["available"] is True
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.allclose(calibrated.mean(axis=0), [.45, .45, .10], atol=.01)
    hard_nll = -np.mean(np.log(scalar[np.arange(len(target)), target]))
    assert calibrator["objective"] < hard_nll - .01


def test_offset_temperature_accepts_fractional_targets_and_falls_back_explicitly() -> None:
    probability = np.tile(np.array([.45, .35, .20]), (150, 1))
    fractional_target = np.tile(np.array([.55, .30, .15]), (150, 1))
    calibrator = fit_offset_temperature_calibrator(probability, fractional_target)
    calibrated = apply_offset_temperature_calibrator(probability, calibrator)
    assert calibrator["available"] is True
    assert np.allclose(calibrated.mean(axis=0), [.55, .30, .15], atol=.01)
    unsupported = fit_offset_temperature_calibrator(probability[:50], fractional_target[:50])
    assert unsupported["available"] is False
    assert unsupported["reason"].startswith("insufficient_rows")
    assert np.allclose(
        apply_offset_temperature_calibrator(probability[:50], unsupported),
        probability[:50],
        atol=1e-12,
        rtol=0.0,
    )


def test_soft_simplex_keeps_hits_hard_and_only_softens_timeouts() -> None:
    result = soft_simplex_targets(_frame())
    assert np.allclose(result[:2], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(result[2], [.4, .4, .2])
    assert np.allclose(result.sum(axis=1), 1.0)


def test_conditional_payoff_rows_are_observed_class_rows_only() -> None:
    frame = _frame()
    frame["competing_risk_class"] = [1, 2, 0, 0]
    assert conditional_class_positions(frame, np.arange(len(frame)), 0).tolist() == [2, 3]
    assert conditional_class_positions(frame, np.arange(len(frame)), 1).tolist() == [0]
    with pytest.raises(ValueError, match="hard simplex"):
        conditional_class_positions(frame, np.arange(len(frame)), 3)


def test_inner_temperature_split_has_12h_purge() -> None:
    timestamps = pd.date_range("2026-05-01", periods=1000, freq="h", tz="UTC")
    panel = pd.DataFrame({"__ts__": timestamps, "label_resolution_utc": timestamps + pd.Timedelta(hours=12)})
    fit, calibration = _inner_calibration_split(panel, np.arange(len(panel)))
    assert len(fit) > 500 and len(calibration) >= 100
    assert panel.iloc[fit]["label_resolution_utc"].max() < panel.iloc[calibration]["__ts__"].min()


def test_global_topk_is_pooled_deterministic_and_keeps_exact_accounting() -> None:
    frame = _frame()
    frame["score"] = 1.0
    first = economic_metrics(frame.sample(frac=1.0, random_state=5), "score", fraction=.5, evaluation="test")
    second = economic_metrics(frame.sample(frac=1.0, random_state=6), "score", fraction=.5, evaluation="test")
    assert first["selected_rows"] == second["selected_rows"] == 2
    assert first["net_ev_bps"] == second["net_ev_bps"]
    assert first["long_rows"] == second["long_rows"] == 1
    assert first["short_rows"] == second["short_rows"] == 1
    assert 0.0 <= first["favorable_touch_rate"] <= 1.0
    assert 0.0 <= first["clean_first_rate"] <= 1.0


def test_score_bridge_binds_one_score_to_path_gross_and_net() -> None:
    frame = pd.concat([_frame()] * 3, ignore_index=True)
    frame["score"] = np.arange(len(frame), dtype=float)
    result = score_bridge_metrics(frame, "score", evaluation="test", scope="pooled_global")
    assert result["rows"] == len(frame)
    assert result["score_unique"] == len(frame)
    assert all(
        name in result
        for name in (
            "mfe_rank_ic",
            "gross_rank_ic",
            "net_rank_ic",
            "favorable_touch_rank_ic",
            "clean_first_rank_ic",
            "adverse_first_rank_ic",
            "timeout_rank_ic",
        )
    )


def test_redundant_primary_geometries_are_explicitly_excluded() -> None:
    assert geometry_contract("primary_floor", 0)["included"] is True
    assert geometry_contract("primary_floor", 100)["included"] is True
    assert geometry_contract("primary_floor", 25)["included"] is False
    assert geometry_contract("primary_floor", 50)["included"] is False
    assert geometry_contract("nofloor", 50)["included"] is True


def test_joint_hpo_has_exactly_16_task_specific_compositions_per_family_side() -> None:
    combinations = joint_hpo_combinations(
        SIMPLEX_GRIDS["logistic"],
        {name: PAYOFF_GRIDS["logistic"] for name in CLASS_NAMES},
    )
    assert len(combinations) == 16
    assert all(set(row["payoff_params"]) == set(CLASS_NAMES) for row in combinations)
    assert {row["classifier_params"]["C"] for row in combinations} == {.10, 1.0}
    assert {row["payoff_params"]["timeout"]["huber_epsilon"] for row in combinations} == {1.20, 1.75}


def test_multiclass_metrics_report_all_requested_classwise_quantities() -> None:
    target = np.array([0, 1, 2, 0, 1, 2])
    probability = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
    result = multiclass_metrics(target, probability)
    assert result["nll"] > 0 and result["rps"] >= 0 and result["simplex_error_max"] < 1e-12
    for name in CLASS_NAMES:
        assert all(f"{name}_{suffix}" in result for suffix in ("brier", "ece10", "auc", "ap"))


def test_label_manifest_validation_fails_closed_on_hash_or_coverage(tmp_path: Path) -> None:
    labels = tmp_path / "execution_ev_cost_aware_competing_risk_labels.parquet"
    labels.write_bytes(b"exact-labels")
    runner = tmp_path / "materialize.py"
    runner.write_text("# bound materializer\n")
    from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import sha256
    manifest = {
        "schema": LABEL_SCHEMA,
        "status": "completed_exact_1m_target_only_not_model_evidence",
        "coverage": {"complete_rows": 1, "rows": 1, "rate": 1.0},
        "event_contract": {"upper_return_floor_included": True, "label_resolution": "execution_decision_utc + 720m"},
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "outputs": {"labels": {"path": str(labels), "sha256": sha256(labels)}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    path, _, _ = _validate_label_artifact(tmp_path, expect_floor=True)
    assert path == labels
    manifest["coverage"]["rate"] = .99
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="100%"):
        _validate_label_artifact(tmp_path, expect_floor=True)
