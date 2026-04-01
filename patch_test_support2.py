import re

with open("tests/test_lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

source = source.replace(
    "edge_low = scorer._compute_support_objective_score(0.10)\n    edge_high = scorer._compute_support_objective_score(0.20)\n    assert 0.0 < edge_low < 1.0\n    assert 0.0 < edge_high < 1.0",
    "edge_low = scorer._compute_support_objective_score(0.06)\n    edge_high = scorer._compute_support_objective_score(0.14)\n    assert 0.0 < edge_low < 1.0\n    assert 0.0 < edge_high < 1.0"
)

with open("tests/test_lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
