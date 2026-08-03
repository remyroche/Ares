from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_short_conditional_payoff_readiness.py"
SPEC = importlib.util.spec_from_file_location("short_conditional_payoff_readiness", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def test_smallest_role_sets_are_action_and_outcome_free() -> None:
    sets = MODULE.recommended_feature_sets()
    assert sets["short_p_net_positive"] == list(MODULE.SCORE_TRIPLET)
    assert "peak_expected_mfe_atr_oof" in sets["short_conditional_favorable_magnitude"]
    assert sets["short_conditional_adverse_severity"][-1] == "mae_expected_before_meaningful_mfe_atr_oof"
    for fields in sets.values(): MODULE.assert_admissible(fields)


def test_forbidden_future_and_action_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        MODULE.assert_admissible(["score_base_alpha", "execution_net_ev_12h"])
    with pytest.raises(ValueError, match="forbidden"):
        MODULE.assert_admissible(["pred_time_to_first_meaningful_mfe__hit_by_2h"])


def test_expected_mae_support_is_prediction_only_composition() -> None:
    frame = pd.DataFrame({
        "pred_mae_before_meaningful_mfe_atr__p_hit": [0.25, 0.8],
        "pred_mae_before_meaningful_mfe_atr__if_hit": [2.0, 1.0],
        "pred_mae_before_meaningful_mfe_atr__if_no_hit": [4.0, 5.0],
    })
    assert MODULE.expected_mae_support(frame).tolist() == pytest.approx([3.5, 1.8])


def test_context_correction_aggregates_coverage_conservatively() -> None:
    correction = Path(__file__).resolve().parents[1] / "scripts" / "write_short_conditional_payoff_readiness_v1_context_correction.py"
    text = correction.read_text(encoding="utf-8")
    assert 'min_finite_fraction=("finite_fraction", "min")' in text
    assert "PARTIAL_EXCLUDE_UNLESS_MISSINGNESS_RULE_PREDECLARED" in text
