import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I already added final base score and worst penalty to `assessment_df`.
# Now I need to update the selection process, which is currently done via `select_top_diverse_rules`.
# Let's inspect where `select_top_diverse_rules` is called.

import ast

class MethodFinder(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'select_top_diverse_rules':
            self.calls.append(node)
        self.generic_visit(node)

tree = ast.parse(source)
finder = MethodFinder()
finder.visit(tree)

print(f"Found {len(finder.calls)} calls to select_top_diverse_rules")
