with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''                self.lineage.record_merge(
                    child_key,
                    [key_a, key_b],
                    "accepted_composite",
                    1,
                    {"decision_reason": diag["accept_merge_reason"]}
                )''',
'''                self.lineage.record_merge(
                    child_key,
                    [key_a, key_b],
                    "accepted_composite",
                    1,
                    {"decision_reason": diag["decision_reason"]}
                )'''
)

code = code.replace(
'''                    self.lineage.record_merge(
                        "none",
                        [key_a, key_b],
                        "rejected_pair",
                        1,
                        {"decision_reason": diag["accept_merge_reason"]}
                    )''',
'''                    self.lineage.record_merge(
                        "none",
                        [key_a, key_b],
                        "rejected_pair",
                        1,
                        {"decision_reason": diag["decision_reason"]}
                    )'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
