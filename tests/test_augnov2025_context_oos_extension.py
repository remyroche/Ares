from pathlib import Path


SOURCE = Path("scripts/run_augnov2025_common30_context_oos_extension.py").read_text()


def test_context_extension_uses_preaug_labels_and_hourly_common30_oos_rows():
    assert "train.execution_label_end_utc.lt(CUT)" in SOURCE
    assert "'decision_cadence':'1h'" in SOURCE
    assert "'exact_replay_bar_cadence':'1m_labels_only'" in SOURCE
    assert "'no_2026_outcomes':True" in SOURCE


def test_context_extension_has_only_the_six_fixed_arms_and_global_selection():
    assert "for family,place in [('lgbm','residual_trust'),('gam','additive_bounded_gam')]" in SOURCE
    assert "for ctx in ['regime','transition','combined']" in SOURCE
    assert "one pooled global raw-score top10 per arm" in SOURCE


def test_context_extension_returns_aggregate_period_metrics_not_selected_rows():
    assert "period_frame=pd.DataFrame(rows)" in SOURCE
    assert "return summary,period_frame,pd.DataFrame(sides),w" in SOURCE
    assert "selected_side=z[z.selected_global_top10]" in SOURCE
