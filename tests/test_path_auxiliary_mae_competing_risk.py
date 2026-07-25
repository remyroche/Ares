from __future__ import annotations

import numpy as np

from extreme_price_movements.path_auxiliary_mae_competing_risk import (
    build_mae_competing_risk_targets,
)


def test_competing_risk_targets_use_side_local_half_and_full_r() -> None:
    result = build_mae_competing_risk_targets(
        [1.0, 2.1, 3.6, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        ["long", "long", "short", "short"],
        [True, True, True, True],
        stop_atr_by_side={"long": 4.0, "short": 3.525840973},
    )

    assert result["risk_class"].tolist() == [0, 1, 1, 2]
    assert result["adverse_0_5r"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result["stop_1r"].tolist() == [0.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(
        result["stop_if_adverse"], [np.nan, 0.0, 1.0, np.nan], equal_nan=True
    )


def test_invalid_rows_are_excluded_from_fitting_targets() -> None:
    result = build_mae_competing_risk_targets(
        [0.2, np.nan],
        [1.0, np.nan],
        ["long", "short"],
        [True, False],
        stop_atr_by_side={"long": 4.0, "short": 3.525840973},
    )

    assert result["risk_class"].tolist() == [0, -1]
    assert np.isnan(result["stop_if_adverse"]).all()
