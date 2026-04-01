import ast

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "MaskAssessor":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "assess_rules":
                s = ast.get_source_segment(source, item)
                with open("assess_rules_body.txt", "w") as f_out:
                    f_out.write(s)
                print("Extracted assess_rules")
                break
