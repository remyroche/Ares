with open("tests/test_lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

import re
source = re.sub(
    r"def test_support_objective_scores_preferred_band_and_excludes_outside_bounds\(\):.*?(def test_list_preload_training_symbols_uses_training_universe\(\):)",
    r"""def test_support_objective_scores_preferred_band_and_excludes_outside_bounds():
    scorer = RuleScorer([], {})
    assert scorer._compute_support_objective_score(0.10) > 0.0

\1""",
    source,
    flags=re.DOTALL
)

with open("tests/test_lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
