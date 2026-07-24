from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_packb_auxiliary_targets import (
    ATR_COLUMN,
    INVALID_REASON_COLUMN,
    align_signal_atr,
    build_target_frame,
    derive_invalid_reasons,
    wilder_atr_fraction,
)
from scripts.materialize_path_auxiliary_targets import SymbolBars


def _population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["long-a", "short-a"],
            "side_name": ["long", "short"],
            "__ts__": pd.to_datetime(
                ["2026-04-01T00:00:00Z", "2026-04-01T00:00:00Z"], utc=True
            ),
            "__symbol__": ["A", "A"],
            "oos_fold": ["outer_1_20260401", "outer_1_20260401"],
            "selected_top40": [True, True],
            "prediction_source": ["outer_oof_fold_model", "outer_oof_fold_model"],
        }
    )


def _bars() -> SymbolBars:
    index = pd.date_range("2026-04-01T01:00:00Z", periods=12, freq="h")
    return SymbolBars(
        index_ns=index.astype("int64").to_numpy(),
        open=np.full(12, 100.0),
        high=np.linspace(101.0, 120.0, 12),
        low=np.linspace(99.0, 80.0, 12),
    )


def test_signal_atr_is_positive_and_exact_without_asof_fill() -> None:
    bars = _bars()
    close = np.full(12, 100.0)
    atr = wilder_atr_fraction(bars.high, bars.low, close)

    aligned = align_signal_atr(_population(), {"A": bars}, {"A": atr})

    # The synthetic signal is one hour before the first loaded bar.
    assert np.isnan(aligned).all()
    assert np.isfinite(atr).all()
    assert (atr > 0.0).all()


def test_exact_population_is_retained_with_side_normalized_targets() -> None:
    population = _population()

    result = build_target_frame(
        population,
        np.asarray([0.01, 0.01], dtype=np.float32),
        {"A": _bars()},
    )

    assert len(result) == len(population)
    assert result["candidate_id"].is_unique
    assert result["side"].equals(result["side_name"])
    assert result["__path_auxiliary_target_valid__"].eq(1).all()
    assert result[INVALID_REASON_COLUMN].eq("valid").all()
    assert result["__label_end_ts__"].eq(pd.Timestamp("2026-04-01T13:00:00")).all()
    assert result["__bars_to_adverse_extreme_before_mfe_12h__"].equals(
        result["__bars_before_price_stops_decreasing_12h__"]
    )
    long_peak = result.loc[result["side_name"].eq("long"), "__peak_mfe_atr_12h__"]
    short_peak = result.loc[result["side_name"].eq("short"), "__peak_mfe_atr_12h__"]
    assert long_peak.iloc[0] > 0.0
    assert short_peak.iloc[0] > 0.0


def test_invalid_reason_distinguishes_atr_and_missing_bars() -> None:
    frame = _population()
    frame[ATR_COLUMN] = [np.nan, 0.01]
    frame.loc[1, "__symbol__"] = "MISSING"

    reasons = derive_invalid_reasons(frame, {"A": _bars()})

    assert reasons.tolist() == [
        "missing_or_nonpositive_signal_atr",
        "missing_symbol_ohlcv",
    ]
