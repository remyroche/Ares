import re

with open('tests/test_lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Since RuleConsolidator has been stubbed out and intentionally removed in the codebase,
# its related tests should be removed.

content = re.sub(
    r"def test_ridge_pair_diagnostic_prefers_complementary_pair.*?(?=\ndef test)",
    "",
    content,
    flags=re.DOTALL
)
content = re.sub(
    r"def test_evaluate_ridge_pair_composite_accepted.*?(?=\ndef test)",
    "",
    content,
    flags=re.DOTALL
)
content = re.sub(
    r"def test_evaluate_ridge_pair_parent_stronger.*?(?=\ndef test)",
    "",
    content,
    flags=re.DOTALL
)
content = re.sub(
    r"def test_evaluate_ridge_pair_composite_rejected_on_std.*?(?=\ndef test)",
    "",
    content,
    flags=re.DOTALL
)
content = re.sub(
    r"def test_economic_rule_consolidator_composite_accepted.*?(?=\ndef test)",
    "",
    content,
    flags=re.DOTALL
)
content = re.sub(
    r"def test_economic_rule_consolidator_duplicate_pruning.*?(?=\ndef test)",
    "",
    content,
    flags=re.DOTALL
)


with open('tests/test_lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
