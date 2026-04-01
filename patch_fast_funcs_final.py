import sys

def patch_file():
    import os
    os.system('cat extreme_price_movements/fast_funcs_add.py >> extreme_price_movements/fast_funcs.py')
    os.system('rm extreme_price_movements/fast_funcs_add.py')

    filepath = 'extreme_price_movements/features.py'
    with open(filepath, 'r') as f:
        content = f.read()

    # remove the bottom import block we added
    lines = content.splitlines()
    new_lines = []
    skip = False
    for line in lines:
        if line.startswith("# Append our fast functions locally so they can be loaded"):
            skip = True
        if not skip:
            new_lines.append(line)

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

if __name__ == "__main__":
    patch_file()
