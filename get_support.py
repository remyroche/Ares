import ast
with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "RuleScorer":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_compute_support_objective_score":
                print(ast.get_source_segment(source, item))
