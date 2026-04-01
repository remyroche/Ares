import ast

def find_ridge_funcs():
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MaskAssessor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and ("ridge" in item.name.lower() or "assess" in item.name.lower()):
                    print(f"MaskAssessor method: {item.name}")
                    if item.name == 'assess_rules':
                         print(ast.get_source_segment(source, item)[:200]) # Print first 200 chars
            break

find_ridge_funcs()
