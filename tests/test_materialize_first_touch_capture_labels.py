from __future__ import annotations

import pytest

from scripts.materialize_first_touch_capture_labels import _parse_side_arm_specs


def test_parse_side_arm_specs_supports_side_specific_geometries() -> None:
    arms = _parse_side_arm_specs(
        "long:ft_long:0.75:0.5:16:0.05;short:ft_short:1.0:0.5:16:0.03:0.4"
    )

    assert set(arms) == {"long", "short"}
    assert arms["long"].name == "ft_long"
    assert arms["long"].tp_r == pytest.approx(0.75)
    assert arms["long"].sl_r == pytest.approx(0.5)
    assert arms["long"].max_bars_to_mfe == pytest.approx(16.0)
    assert arms["long"].max_barrier == pytest.approx(0.05)
    assert arms["long"].trail_r == pytest.approx(0.5)
    assert arms["short"].name == "ft_short"
    assert arms["short"].tp_r == pytest.approx(1.0)
    assert arms["short"].trail_r == pytest.approx(0.4)


def test_parse_side_arm_specs_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="Invalid side"):
        _parse_side_arm_specs("flat:ft_flat:0.75:0.5:16:0.05")
