import sys

def patch_file():
    filepath = 'extreme_price_movements/fast_funcs_add.py'
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = "from numba import njit, prange\nimport numpy as np\n" + content

    with open(filepath, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    patch_file()
