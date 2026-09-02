from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.causal_oi_positioning import CausalOIPositioningEngine
from scripts.materialize_causal_oi_positioning import _strict_prior_oi


def test_strict_prior_oi_never_uses_same_timestamp(tmp_path):
    index = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame(index=index)
    pd.DataFrame({"open_interest": [100.0, 110.0]}, index=[index[0], index[4]]).to_parquet(tmp_path / "TEST_USD_USD.parquet")
    values, coverage = _strict_prior_oi(bars, tmp_path / "TEST_USD_USD.parquet", 4)
    assert np.isnan(values.iloc[0])
    assert values.iloc[1] == 100.0
    assert coverage == 4


def test_positioning_zone_does_not_role_reverse():
    assert CausalOIPositioningEngine._orientation_ok(type("Z", (), {"kind": "long_build", "center": 10.0})(), 10.1)
    assert not CausalOIPositioningEngine._orientation_ok(type("Z", (), {"kind": "long_build", "center": 10.0})(), 9.9)
    assert CausalOIPositioningEngine._orientation_ok(type("Z", (), {"kind": "short_build", "center": 10.0})(), 9.9)
