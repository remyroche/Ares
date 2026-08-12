from __future__ import annotations

import pytest

from scripts.prepare_stage_i_production_inputs import _optional_positive_cap


def test_stage_i_production_input_caps_are_unlimited_at_zero() -> None:
    assert _optional_positive_cap(0, option="--max-feature-columns") is None
    assert _optional_positive_cap(512, option="--max-feature-columns") == 512


def test_stage_i_production_input_caps_reject_negative_values() -> None:
    with pytest.raises(ValueError, match=r"zero \(unlimited\) or positive"):
        _optional_positive_cap(-1, option="--max-feature-columns")
