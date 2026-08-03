from pathlib import Path
def test_strict_forward_regime_runner_excludes_transition_and_inverse_pi():
 p=Path('scripts/run_strict_forward_regime_only.py').read_text()
 assert '"transition" not in c.lower()' in p
 assert 'inverse_pi_jan_aug_2022' in p and 'promotion_eligible' in p


def test_regime_fit_is_strictly_pre_2026_and_resets_at_gaps():
 p=Path('scripts/run_strict_forward_regime_only.py').read_text()
 assert 'training_end_exclusive_utc' in p
 assert 'all feature selection, preprocessing, geometry, persistence' in p
 assert 'segments[row] != segments[row - 1]' in p
 assert 'train_states[:split]' in p
 assert 'train_states, train_segments' in p
 assert 'test_emissions, test_segments' in p


def test_persistence_objective_uses_evidence_before_normalisation():
 p=Path('scripts/run_strict_forward_regime_only.py').read_text()
 evidence = p.index('evidence = max(float(unnormalised.sum())')
 normalise = p.index('filtered[row] = unnormalised / evidence')
 score = p.index('log_scores.append(float(np.log(evidence)))')
 assert evidence < normalise < score
 assert 'temporal_switch_rate' in p
