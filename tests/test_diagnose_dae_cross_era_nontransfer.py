from scripts.diagnose_dae_cross_era_nontransfer import LAT, REP

def test_dae_cross_era_diagnostic_excludes_posterior_and_action_features():
    assert len(LAT)==16 and 'dae_reconstruction_error_zscore' in REP
    assert all('posterior' not in x and 'timing' not in x and 'mae' not in x and 'wait' not in x for x in REP)
