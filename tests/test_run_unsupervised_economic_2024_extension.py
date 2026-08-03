from scripts.run_unsupervised_economic_2024_extension import ARMS, RAW_FEATURES, RETAINED_DIAGNOSTICS

def test_2024_extension_has_geometry_dae_and_separate_probability_arms():
    assert {'baseline','gmm_geometry','dae_only','gmm_plus_dae','failure_destination','transition_only','failure_plus_transition'} == set(ARMS)
    assert 'gmm_posterior_max' not in sum(ARMS.values(), [])
    assert all(not any(t in c.lower() for t in ('timing','mae','wait','action','target_price')) for cols in ARMS.values() for c in cols)
    assert len(RAW_FEATURES) >= 30
    assert 'dae_reconstruction_error_zscore' in RETAINED_DIAGNOSTICS
    assert len([x for x in RETAINED_DIAGNOSTICS if x.startswith('dae_fold_')]) == 8
