import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.transition_pattern_catalogue import (
    TransitionPatternConfig,
    causal_predictor_columns,
    materialize_adaptive_transition_phases,
    sample_stable_vs_transition,
    summarize_event_preonset_sequences,
    validate_causal_predictor_columns,
)


def _panel(*, periods: int = 330, segment_id: int = 7) -> pd.DataFrame:
    source = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    state = np.zeros(periods, dtype=np.int16)
    state[220:] = 1
    return pd.DataFrame(
        {
            "source_utc": source,
            "execution_decision_utc": source + pd.Timedelta(hours=1),
            "segment_id": segment_id,
            "target__pooled_state": state,
            "observable_breadth": np.linspace(-1.0, 1.0, periods),
            "observable_volatility": np.linspace(1.0, 4.0, periods),
            "target__future_phase": 1.0,
            "expost__net_ev": -0.02,
            "future_mfe": 2.0,
        }
    )


def _event(panel: pd.DataFrame, *, anchor_index: int = 220, end_hours: int = 3) -> pd.DataFrame:
    anchor = panel.loc[anchor_index, "source_utc"]
    return pd.DataFrame(
        {
            "event_id": ["event_a"],
            "segment_id": [int(panel.loc[anchor_index, "segment_id"])],
            "anchor_source_utc": [anchor],
            "transition_end_utc": [anchor + pd.Timedelta(hours=end_hours)],
            "target_available_utc": [anchor + pd.Timedelta(hours=13)],
            "source_state": [0],
            "destination_state": [1],
        }
    )


def test_adaptive_phase_labels_are_exclusive_available_and_keep_phase_out_of_inputs() -> None:
    panel = _panel()
    event = _event(panel)
    config = TransitionPatternConfig(precondition_hours=48, sequence_horizons_hours=(3, 6, 12, 48))
    labeled = materialize_adaptive_transition_phases(panel, event, config=config)
    anchor = event.loc[0, "anchor_source_utc"]

    def phase_at(hours: int) -> str:
        return str(labeled.loc[labeled["source_utc"].eq(anchor + pd.Timedelta(hours=hours)), "target__pattern_phase"].iloc[0])

    assert phase_at(-48) == "precondition"
    assert phase_at(-24) == "approach"
    assert phase_at(-6) == "acceleration"
    assert phase_at(-3) == "trigger"
    assert phase_at(0) == "active_dislocation"
    assert phase_at(3) == "confirmation"
    assert phase_at(9) == "settled"
    assert phase_at(30) == "stable_destination"
    event_rows = labeled["target__pattern_event_id"].eq("event_a")
    assert labeled.loc[event_rows, "target__pattern_phase_available_utc"].ge(anchor + pd.Timedelta(hours=13)).all()
    assert labeled.loc[event_rows, "target__pattern_transition_context_available_utc"].notna().all()
    assert "target__pattern_phase" not in causal_predictor_columns(labeled)
    assert "target__future_phase" not in causal_predictor_columns(labeled)
    assert "expost__net_ev" not in causal_predictor_columns(labeled)
    assert "future_mfe" not in causal_predictor_columns(labeled)
    with pytest.raises(ValueError, match="non-causal"):
        validate_causal_predictor_columns(labeled, ["target__pattern_phase"])


def test_failed_transition_and_reversal_are_explicit_labels() -> None:
    failed_panel = _panel()
    failed_anchor = 220
    # The active state reaches the event end but immediately falls back before
    # the confirmation window has completed.
    failed_panel.loc[failed_anchor + 3 :, "target__pooled_state"] = 0
    failed = materialize_adaptive_transition_phases(
        failed_panel,
        _event(failed_panel, anchor_index=failed_anchor),
        config=TransitionPatternConfig(precondition_hours=48),
    )
    timestamp = failed_panel.loc[failed_anchor, "source_utc"] + pd.Timedelta(hours=3)
    assert failed.loc[failed["source_utc"].eq(timestamp), "target__pattern_phase"].iloc[0] == "failed_transition"

    reversal_panel = _panel()
    # Confirmation succeeds, then the market returns to its origin state during
    # the reversal search window.
    reversal_panel.loc[232:, "target__pooled_state"] = 0
    reversed_labels = materialize_adaptive_transition_phases(
        reversal_panel,
        _event(reversal_panel, anchor_index=220),
        config=TransitionPatternConfig(precondition_hours=48, reversal_search_hours=40),
    )
    assert "reversal" in set(reversed_labels["target__pattern_phase"].dropna())
    reversal_row = reversed_labels.loc[reversed_labels["target__pattern_phase"].eq("reversal")].iloc[0]
    assert reversal_row["target__pattern_phase_available_utc"] > reversal_row["source_utc"]


def test_sequence_summaries_fail_closed_across_internal_gap_and_stable_sampling_is_event_grouped() -> None:
    panel = _panel()
    # Preserve the same segment id but remove a source hour inside the 12h
    # pre-onset sequence.  A summary may not bridge it by row shifting.
    panel = panel.drop(index=214).reset_index(drop=True)
    event = _event(_panel())
    config = TransitionPatternConfig(precondition_hours=48, sequence_horizons_hours=(3, 12))
    summary = summarize_event_preonset_sequences(
        panel,
        event,
        feature_columns=["observable_breadth"],
        config=config,
    )
    assert summary.loc[0, "sequence__complete_3h"] == 1
    assert summary.loc[0, "sequence__complete_12h"] == 0
    assert np.isnan(summary.loc[0, "sequence__observable_breadth__mean_12h"])
    assert summary.loc[0, "sequence_available_utc"] == event.loc[0, "anchor_source_utc"]

    labeled = materialize_adaptive_transition_phases(_panel(), event, config=config)
    sample = sample_stable_vs_transition(labeled, stable_to_transition_ratio=0.5, random_state=1)
    assert set(sample["target__stable_vs_transition"].unique()) == {0, 1}
    positive_groups = sample.loc[sample["target__stable_vs_transition"].eq(1), "transition_cv_group_id"]
    assert positive_groups.eq("event::event_a").all()
    assert sample["target__pattern_transition_context_available_utc"].notna().all()
