import os
for root, dirs, files in os.walk("extreme_price_movements"):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r") as f:
                content = f.read()
                if "mfe" in content.lower():
                    print(f"Found in {os.path.join(root, file)}")
