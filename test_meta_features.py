import re
with open("extreme_price_movements/config.py", "r") as f:
    text = f.read()

def count_keys(name):
    match = re.search(f'"{name}": \\[(.*?)\\]', text, re.DOTALL)
    if match:
        items = match.group(1).split(",")
        items = [i.strip() for i in items if i.strip() and not i.strip().startswith("#")]
        return len(items)
    return 0

print("Base meta:", count_keys("meta_feature_keys"))
print("MR meta:", count_keys("mr_meta_feature_keys"))
print("TF meta:", count_keys("tf_meta_feature_keys"))
