from pathlib import Path


SOURCE = Path("scripts/materialize_augnov2025_frozen_july_oos_bridge.py").read_text()


def test_aug_nov_bridge_is_hourly_and_keeps_one_minute_nested_only():
    assert "'decision_cadence':'1h'" in SOURCE
    assert "'exact_replay_bar_cadence':'1m_labels_only'" in SOURCE
    assert "not (x.__ts__.astype('int64')%pd.Timedelta(hours=1).value==0).all()" in SOURCE


def test_aug_nov_bridge_blocks_future_native_labels_and_preserves_score_pair():
    assert "native.base_label_resolution_utc.lt(CUT)" in SOURCE
    assert "frame.native_label_resolution_utc.lt(CUT)" in SOURCE
    assert "no_aug_nov_native_labels_read':True" in SOURCE
    assert "score_base_alpha" in SOURCE
    assert "score_residual_expected_ev" in SOURCE


def test_aug_nov_bridge_is_explicitly_staged_and_common30_scoped():
    assert "('fit_base','score_base','fit_residual','score_residual','seal')" in SOURCE
    assert "not identical to wider final v3 population" in SOURCE
