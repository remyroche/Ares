import re

with open("extreme_price_movements/training.py", "r") as f:
    lines = f.readlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == 'if best_m is not None:':
        out.append(line)
        i += 1
        # The next block until the next dedent needs to be indented
        while i < len(lines):
            l = lines[i]
            if l.startswith('            ') and not l.startswith('                '):
                out.append('    ' + l)
            elif l.strip() == '' or l.startswith('                '):
                out.append('    ' + l) if l.strip() != '' else out.append(l)
            else:
                out.append(l)
                break
            i += 1
        continue

    if line.strip() == 'for _h, _v in per_h_models.items():':
        out.append(line)
        i += 1
        while i < len(lines):
            l = lines[i]
            if l.startswith('            ') and not l.startswith('                '):
                out.append('    ' + l)
            elif l.strip() == '' or l.startswith('                '):
                out.append('    ' + l) if l.strip() != '' else out.append(l)
            else:
                out.append(l)
                break
            i += 1
        continue

    out.append(line)
    i += 1

with open("extreme_price_movements/training.py", "w") as f:
    f.writelines(out)
