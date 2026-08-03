import pandas as pd

from scripts.run_july2025_common30_all_context_map_refresh import ARM_MAP
from scripts.run_july2025_common30_regime_context_raw_score_extension import _score_metrics


def test_july_context_extensions_cover_exactly_the_compatible_residual_and_gam_arms():
    assert ARM_MAP == {
        "baseline": "baseline_raw_residual",
        "residual_regime_only": "residual_trust_regime_raw",
        "residual_transition_only": "residual_trust_transition_raw",
        "residual_combined": "residual_trust_combined_raw",
        "gam_regime_only": "additive_bounded_gam_regime_raw",
        "gam_transition_only": "additive_bounded_gam_transition_raw",
        "gam_combined": "additive_bounded_gam_combined_raw",
    }


def test_raw_context_diagnostic_selects_one_global_top10_not_a_per_timestamp_book():
    frame = pd.DataFrame({
        "candidate_id": [f"id{i}" for i in range(20)],
        "__ts__": pd.date_range("2025-07-01", periods=20, freq="h", tz="UTC"),
        "__symbol__": "BTC/USD:USD",
        "side_name": ["long", "short"] * 10,
        "raw_score": list(range(20)),
        "execution_net_ev_12h": [.01] * 20,
        "__first_touch_target_soft__": [.5] * 20,
    })
    summary, periods, sides, scored = _score_metrics(frame, "test")
    assert summary["top10_rows"] == 2
    assert scored.selected_global_top10_raw.sum() == 2
    assert periods.loc[periods.period_type.eq("week"), "global_selected_rows"].sum() == 2
    assert sides.global_selected_rows.sum() == 2


def test_extensions_bind_hourly_prejuly_and_no_2026_contracts():
    raw_source = open("scripts/run_july2025_common30_regime_context_raw_score_extension.py").read()
    map_source = open("scripts/run_july2025_common30_all_context_map_refresh.py").read()
    assert 'execution_label_end_utc.lt(START)' in raw_source
    assert 'model_sample_cadence": "1h"' in raw_source
    assert 'exact_replay_bar_cadence": "1m_labels_only"' in raw_source
    assert 'no_2026_fit_tuning_or_selection":True' in map_source
    assert 'july_refreshed_common30' in map_source
