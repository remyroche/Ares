from __future__ import annotations

import pandas as pd

from scripts.run_long_support_aware_structural_residual_ablation import (
    PAIRWISE_CONTROL_COLUMN,
    SUPPORT_COLUMN,
    SUPPORT_PROBABILITY,
    _native_specs,
    _custom_control_specs,
    _prepare_support_label,
)


def test_support_label_is_training_only_and_s4_is_explicitly_gated():
    frame = pd.DataFrame({"mfe_mae_label_valid": [True, False, None]})
    labelled = _prepare_support_label(frame)
    assert labelled[SUPPORT_COLUMN].tolist() == [True, False, False]
    stage_one = _native_specs(include_s4=False, s4_weight=2.0, include_r1=False)
    assert {item[1] for item in stage_one} == {
        "S1_uniform", "S2_weight_1_5", "S2_weight_2_0", "S2_weight_3_0",
        "S3_prequential_support_probability",
    }
    assert all(not item[3] or SUPPORT_PROBABILITY not in (SUPPORT_COLUMN,) for item in stage_one)
    stage_two = _native_specs(include_s4=True, s4_weight=2.0, include_r1=True)
    assert any(item[1].startswith("S4_weight_2") and item[2] == 2.0 and item[3] for item in stage_two)
    assert {item[0] for item in stage_two} == {"R3_portability_health", "R1_reasoning_memberships"}


def test_custom_pairwise_control_has_no_support_arm_or_probability():
    specs = _custom_control_specs(include_r1=False)
    assert specs == [("R3_portability_health", "S1_uniform", None, False)]
    assert SUPPORT_COLUMN != PAIRWISE_CONTROL_COLUMN
    assert SUPPORT_PROBABILITY not in {field for spec in specs for field in spec if isinstance(field, str)}
