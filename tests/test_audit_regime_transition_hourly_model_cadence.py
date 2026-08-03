from pathlib import Path


def test_cadence_audit_separates_hourly_models_from_minute_replay():
    source = Path(
        "scripts/audit_regime_transition_hourly_model_cadence.py"
    ).read_text()
    assert '"model_sample_cadence": "1h"' in source
    assert '"assessment_sample_cadence": "1h"' in source
    assert '"exact_replay_bar_cadence": "1m"' in source
    assert "never independent model rows" in source
    assert "pd.Timedelta(hours=1)" in source
    assert "pd.Timedelta(hours=12)" in source
