"""Focused executable contracts for the strict Stage-1 retention runner."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from scripts.run_stage_c_conditional_retention_ablation import (
    EVAL_START,
    DEV_START,
    FROZEN_E15,
    HORIZON_HOURS,
    _development_months,
    _correctness_report,
    _assert_identical_ids,
    _fit_transform,
    _freeze_incremental,
    _group_features,
    _paired_seed,
    _train_mask,
)


def test_feature_selection_uses_training_data_only() -> None:
    records = [
        {"arm": "C1", "side": "long", "split": "development_oof", "incremental_selected": ["train_signal"], "selector": {"gain": {"train_signal": 5.0}}},
        # A deliberately huge final-only gain may not enter the frozen list.
        {"arm": "C1", "side": "long", "split": "final_oos", "incremental_selected": ["final_only_signal"], "selector": {"gain": {"final_only_signal": 1_000_000.0}}},
    ]
    selected, evidence = _freeze_incremental(records, "C1", "long")
    assert selected == ["train_signal"]
    assert evidence["final_oos_labels_used"] is False


def test_scalers_and_clippers_fit_on_training_data_only() -> None:
    train = pd.DataFrame({"feature": np.arange(100, dtype=float)})
    test = pd.DataFrame({"feature": [10_000.0]})
    _, clipped, state = _fit_transform(train, test, ["feature"])
    assert clipped.feature.iloc[0] == state["clip_bounds"]["feature"][1]
    assert clipped.feature.iloc[0] < test.feature.iloc[0]


def test_no_final_oos_feature_selection() -> None:
    records = [
        {"arm": "C2", "side": "short", "split": "development_oof", "incremental_selected": ["dev_a", "dev_b"], "selector": {"gain": {"dev_a": 2.0, "dev_b": 1.0}}},
        {"arm": "C2", "side": "short", "split": "final_oos", "incremental_selected": ["future_winner"], "selector": {"gain": {"future_winner": 99.0}}},
    ]
    frozen, evidence = _freeze_incremental(records, "C2", "short")
    assert frozen == ["dev_a", "dev_b"]
    assert "future_winner" not in frozen
    assert evidence["source"] == "development_oof_train_fold_selectors_only"


def test_h12_purge_embargo_and_label_availability_cutoff() -> None:
    fold_start = EVAL_START
    frame = pd.DataFrame({
        "decision_ts": pd.to_datetime([
            fold_start - pd.Timedelta(hours=HORIZON_HOURS + 1),
            fold_start - pd.Timedelta(hours=HORIZON_HOURS),
            fold_start - pd.Timedelta(hours=HORIZON_HOURS + 1),
        ]),
        "label_available_ts": pd.to_datetime([
            fold_start - pd.Timedelta(seconds=1),
            fold_start - pd.Timedelta(seconds=1),
            fold_start,
        ]),
    })
    assert _train_mask(frame, fold_start).tolist() == [True, False, False]


def test_april_development_oof_has_strict_pre_april_resolved_training_support() -> None:
    assert _development_months() == ["2024-04", "2024-05", "2024-06", "2024-07"]
    april = DEV_START
    frame = pd.DataFrame({
        "decision_ts": pd.to_datetime(["2024-03-31 11:59:00Z", "2024-03-31 12:00:00Z", "2024-04-01 01:00:00Z"]),
        "label_available_ts": pd.to_datetime(["2024-03-31 23:59:00Z", "2024-03-31 23:59:00Z", "2024-04-01 13:00:00Z"]),
    })
    train = frame.loc[_train_mask(frame, april)]
    april_test = frame.loc[frame.decision_ts.ge(april) & frame.decision_ts.lt(april + pd.offsets.MonthBegin(1))]
    assert len(train) == 1
    assert len(april_test) == 1
    assert train.decision_ts.max() < april - pd.Timedelta(hours=HORIZON_HOURS)
    assert train.label_available_ts.max() < april


def test_f0_hash_is_exact_persisted_e15_control() -> None:
    assert hashlib.sha256(FROZEN_E15.read_bytes()).hexdigest() == "a91c1b40ad87f4fab3311aef2865c6bdcc713d2de75bbb7e9623384ac6085ed1"


def test_c_group_isolation_and_source_blocks() -> None:
    groups = _group_features()
    assert groups["C4"] == groups["C5"] == groups["C7"] == []
    assert "side_cont_adverse_rv_12h" not in groups["C1"]
    assert "side_cont_adverse_rv_12h" in groups["C3"]
    assert set(groups["C1"]).isdisjoint(groups["C2"])
    assert set(groups["C2"]).isdisjoint(groups["C3"])


def test_comparison_arms_use_identical_candidate_ids() -> None:
    candidate = pd.DataFrame({
        "candidate_id": ["a", "b"], "split": ["final_oos", "final_oos"],
        "fold": ["2024-08_to_2024-11", "2024-08_to_2024-11"],
    })
    scored = pd.concat([candidate.assign(arm="C0"), candidate.assign(arm="C1")], ignore_index=True)
    audit = _assert_identical_ids(scored, ["C0", "C1"])
    assert audit.identical_to_c0.all()
    assert audit.candidate_id_sha256.nunique() == 1


def test_paired_model_seeds_are_arm_invariant_and_fold_side_specific() -> None:
    c0_seed = _paired_seed(20260731, side_index=1, fold_index=2, phase="development_model")
    c8_seed = _paired_seed(20260731, side_index=1, fold_index=2, phase="development_model")
    assert c0_seed == c8_seed
    assert c0_seed != _paired_seed(20260731, side_index=0, fold_index=2, phase="development_model")
    assert c0_seed != _paired_seed(20260731, side_index=1, fold_index=3, phase="development_model")


def test_correctness_report_fails_closed_when_any_check_fails() -> None:
    report = _correctness_report(checks={"identity": True, "purge": False}, blocked={})
    assert report["passed"] is False
    assert report["checks"] == {"identity": True, "purge": False}
