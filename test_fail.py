with open("tests/test_lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

import re
matches = re.finditer(r"def test_support_objective_scores_preferred_band_and_excludes_outside_bounds", source)
for m in matches:
    print(source[m.start():m.start()+500])
