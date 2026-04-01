import ast

def get_subset_auc():
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MaskAssessor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "assess_rules":
                    print(ast.get_source_segment(source, item))
                    return

get_subset_auc()
