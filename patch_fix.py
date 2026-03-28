with open("extreme_price_movements/training.py", "r") as f:
    lines = f.readlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == 'if best_m is not None:':
        # the next block looks messed up
        if 'best_m["models_by_h"] = {' in lines[i+1]:
            # we need to fix this indentation
            lines[i+1] = '                best_m["models_by_h"] = {\n'
            out.append(line)
            i += 1
            continue
    out.append(line)
    i += 1

with open("extreme_price_movements/training.py", "w") as f:
    f.writelines(out)
