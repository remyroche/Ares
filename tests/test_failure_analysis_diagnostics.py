from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.diagnostics.failure_analysis import (
    FailureAnalysisConfig,
    analyze_failure_diagnostics,
)


def _ledger() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    residuals = [0.0, 0.1, -0.1, 0.0, -3.0, -2.0, 0.1, 0.0]
    rows = []
    for index, (timestamp, residual) in enumerate(zip(timestamps, residuals)):
        rows.append(
            {
                "timestamp": timestamp + pd.Timedelta(hours=8),
                "expected_pnl": 1.0,
                "realized_pnl": 1.0 + residual,
                "bankroll_pnl": 1.0 + residual,
                "base_model": "base_a" if index < 6 else "base_b",
                "side": "long" if index % 2 == 0 else "short",
                "setup": "trend",
                "horizon": "24h",
                "symbol": "AAA" if index < 5 else "BBB",
                "regime": "calm" if index < 4 else "volatile",
                "state_vol": 1.0 if index < 4 else 10.0,
                "state_trend": 2.0 if index < 4 else 20.0,
            }
        )
    return pd.DataFrame(rows)


def _config() -> FailureAnalysisConfig:
    return FailureAnalysisConfig(
        residual_z_window_days=4,
        residual_z_min_periods=3,
        residual_percentile_min_days=3,
        bankroll_drawdown_threshold=2.0,
        market_state_cols=("state_vol", "state_trend"),
    )


def test_failure_days_merge_adjacent_days_and_summarize_episodes() -> None:
    result = analyze_failure_diagnostics(_ledger(), _config())

    failures = result.daily.loc[result.daily["failure_day"]]
    assert pd.Timestamp("2026-01-05", tz="UTC") in set(failures["day"])
    assert pd.Timestamp("2026-01-06", tz="UTC") in set(failures["day"])
    episode = result.episodes.loc[
        result.episodes["start_day"].eq(pd.Timestamp("2026-01-05", tz="UTC"))
    ].iloc[0]
    assert episode["end_day"] == pd.Timestamp("2026-01-06", tz="UTC")
    assert episode["failure_days"] == 2
    assert "causal_residual_p05" in episode["failure_reasons"] or "residual_z" in episode["failure_reasons"]
    assert set(result.episode_comparisons["lookback_hours"]) == {6, 12, 24}


def test_monthly_counterfactual_and_sequence_outputs_are_descriptive() -> None:
    result = analyze_failure_diagnostics(_ledger(), _config())

    assert {"month", "base_model", "side", "setup", "horizon", "realized_pnl"}.issubset(result.monthly_performance)
    side_long = result.counterfactual_removals.loc[
        (result.counterfactual_removals["condition"] == "side")
        & (result.counterfactual_removals["condition_value"] == "long")
    ].iloc[0]
    assert side_long["delta_pnl"] == -side_long["removed_pnl"]
    assert side_long["diagnostic_only"]
    sequence = result.sequence_metrics.sort_values("_timestamp", kind="stable")
    assert sequence.iloc[0]["previous_trade_count"] == 0
    assert np.isclose(sequence.iloc[1]["previous_trade_pnl_mean"], 1.0)
    assert result.manifest["descriptive_only"] and result.manifest["noncausal"]
    assert result.manifest["inference_eligible"] is False


def test_similarity_only_uses_earlier_failure_episodes() -> None:
    ledger = _ledger()
    # Build two separated adverse days with identical observable state vectors.
    ledger.loc[ledger.index.isin([5, 6]), "realized_pnl"] = 1.0
    ledger.loc[ledger.index.isin([4, 7]), "realized_pnl"] = -3.0
    config = FailureAnalysisConfig(
        residual_z_window_days=4,
        residual_z_min_periods=3,
        residual_percentile_min_days=3,
        bankroll_drawdown_threshold=None,
        market_state_cols=("state_vol", "state_trend"),
    )
    result = analyze_failure_diagnostics(ledger, config)

    similarity = result.episode_similarity.sort_values("episode_id")
    assert len(similarity) == 2
    assert not similarity.iloc[0]["historical_similarity_available"]
    assert similarity.iloc[-1]["historical_similarity_available"]
    assert similarity.iloc[-1]["historical_nearest_episode_id"] < similarity.iloc[-1]["episode_id"]
