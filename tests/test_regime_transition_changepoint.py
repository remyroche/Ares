import numpy as np
import pandas as pd

from extreme_price_movements.regime_transition_changepoint import (
    CausalChangePointConfig,
    materialize_causal_changepoint_context,
)
from scripts.run_regime_transition_changepoint_ablation import freeze_score_only_threshold


INPUTS = (
    "negative_breadth_pct",
    "breadth_dispersion",
    "correlation_breakdown_dispersion",
    "funding_deleveraging_divergence",
    "short_covering_score_market",
    "flush_recovery_state",
)


def _panel(values: np.ndarray) -> pd.DataFrame:
    stamp = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "source_utc": stamp,
            "segment_id": 1,
            **{name: values + offset for offset, name in enumerate(INPUTS)},
        }
    )


def test_changepoint_context_is_causal_after_warmup() -> None:
    prefix = np.r_[np.zeros(80), np.full(40, 2.0)].astype(float)
    left = _panel(np.r_[prefix, np.zeros(40)])
    right = _panel(np.r_[prefix, np.full(40, 12.0)])
    config = CausalChangePointConfig(warmup_hours=64, expected_run_hours=24, max_run_hours=48)
    left_context, columns = materialize_causal_changepoint_context(left, config=config)
    right_context, _ = materialize_causal_changepoint_context(right, config=config)
    assert left_context.iloc[:64].isna().all().all()
    np.testing.assert_allclose(
        left_context.loc[: len(prefix) - 1, columns].to_numpy(),
        right_context.loc[: len(prefix) - 1, columns].to_numpy(),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )


def test_changepoint_context_resets_at_a_segment_boundary() -> None:
    values = np.r_[np.zeros(80), np.full(80, 3.0)]
    frame = _panel(values)
    frame.loc[80:, "segment_id"] = 2
    config = CausalChangePointConfig(warmup_hours=64, expected_run_hours=24, max_run_hours=48)
    context, _ = materialize_causal_changepoint_context(frame, config=config)
    assert context.iloc[80:144].isna().all().all()


def test_score_only_threshold_never_accepts_labels() -> None:
    stamp = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    score = np.linspace(0.0, 1.0, len(stamp))
    threshold, rate = freeze_score_only_threshold(
        pd.Series(stamp), pd.Series(np.ones(len(stamp), dtype=int)), score, budget_per_30d=4.0
    )
    assert np.isfinite(threshold)
    assert 0.0 <= rate <= 4.0
