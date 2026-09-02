from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_causal_sr_e2_mc1_input_ablation as subject


def test_all_candidate_e2_contract_is_independent_of_pairwise_e2() -> None:
    assert len(subject.E2_FEATURES) == 70
    assert all("margin__" not in name for name in subject.E2_FEATURES)
    assert all(not name.startswith("policy_") for name in subject.E2_FEATURES)
    assert subject.E2_OUTPUT not in subject.E2_FEATURES
    assert subject.E2_AVAILABLE not in subject.E2_FEATURES


def test_e2_output_is_a_raw_prediction_not_a_label_calibration() -> None:
    assert subject.E2_OUTPUT == "e2_15m_prequential_raw_policy_bps"
    assert subject.E2_AVAILABLE == "e2_15m_prequential_available"
    assert subject.E2_OUTPUT != "policy_net_bps"


def test_target_free_selection_rejects_outcome_fields() -> None:
    clean = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00Z", "2026-08-01T00:00Z"], utc=True),
    })
    subject._assert_target_free_selection(clean, "clean")
    contaminated = clean.assign(policy_net_bps=1.0)
    try:
        subject._assert_target_free_selection(contaminated, "contaminated")
    except AssertionError as exc:
        assert "outcome fields" in str(exc)
    else:
        raise AssertionError("selection with a policy outcome field must fail")
