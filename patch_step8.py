import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I need to change the call from `select_top_diverse_rules` to `select_final_regimes` in `run_mining_stage`.
# Let's inspect `run_mining_stage`.

import ast

class CallerFinder(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'select_top_diverse_rules':
            print("Found call to select_top_diverse_rules")
            self.found = True
        self.generic_visit(node)

tree = ast.parse(source)
finder = CallerFinder()
finder.visit(tree)
