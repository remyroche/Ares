from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.meta_cross_sectional_geometry import (
    CrossSectionalGeometryState,
    geometry_feature_names,
    materialize_cross_sectional_geometry,
)


def _frame(hours: int = 8, symbols: int = 20) -> pd.DataFrame:
    rows = hours * symbols
    ts = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC").repeat(symbols)
    idx = np.arange(rows)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": [f"S{i % symbols}" for i in idx],
            "side_name": np.where(idx % 2 == 0, "long", "short"),
            "archetype_policy_key": np.where(idx % 3 == 0, "breakout", "mixed"),
            "score": (0.2 + 0.7 * ((idx * 17) % symbols) / symbols).astype(np.float32),
            "asset_minus_mkt_oi_chg_1h_rz": np.sin(idx / 7.0).astype(np.float32),
        }
    )


def test_geometry_is_finite_and_row_aligned() -> None:
    frame = _frame()
    out = materialize_cross_sectional_geometry(
        frame,
        score_col="score",
        relative_features=("asset_minus_mkt_oi_chg_1h_rz",),
    )
    assert out.index.equals(frame.index)
    assert set(geometry_feature_names(("asset_minus_mkt_oi_chg_1h_rz",))) == set(
        out.columns
    )
    assert np.isfinite(out.to_numpy(dtype=np.float32)).all()
    assert out["meta_xsgeom_top10_turnover_1h"].between(0.0, 1.0).all()


def test_future_rows_do_not_change_past_geometry() -> None:
    frame = _frame(hours=10)
    cutoff = pd.Timestamp("2026-01-01 06:00:00", tz="UTC")
    past = frame[frame["__ts__"].lt(cutoff)].copy()
    past_only = materialize_cross_sectional_geometry(past, score_col="score")
    full = materialize_cross_sectional_geometry(frame, score_col="score").loc[
        past.index
    ]
    np.testing.assert_allclose(past_only.to_numpy(), full.to_numpy(), atol=1e-7)


def test_live_turnover_state_uses_only_lagged_membership() -> None:
    state = CrossSectionalGeometryState()
    t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    state.update(t0, ["A", "B", "C"])
    assert state.turnover(t0 + pd.Timedelta(hours=1), ["A", "B", "D"], 1) == 0.5
    assert state.turnover(t0 + pd.Timedelta(hours=4), ["A", "B", "C"], 4) == 0.0
