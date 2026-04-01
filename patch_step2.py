import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I will modify _compute_subset_auc to return oof_preds in addition to the learnability score.
# Or, instead of breaking its signature, I will create a new method `_compute_ridge_oof_preds`
# Wait, actually `_compute_subset_auc` already returns `mask_auc, subset_oof_coverage`.
# Let's see where it is called.

import ast

class MethodFinder(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == '_compute_subset_auc':
            self.calls.append(node)
        self.generic_visit(node)

tree = ast.parse(source)
finder = MethodFinder()
finder.visit(tree)

print(f"Found {len(finder.calls)} calls to _compute_subset_auc")
