import sys

def patch_file():
    filepath = 'extreme_price_movements/features.py'
    with open(filepath, 'r') as f:
        content = f.read()

    new_imports = """        bars_since_flip_nb,
        bars_since_flip_nb_parallel,
        binary_entropy_nb,
        binary_entropy_nb_parallel,
        consecutive_bars_nb,
        consecutive_bars_nb_parallel,
        up_down_semivol_ratio_nb,
        up_down_semivol_ratio_nb_parallel,
        up_down_return_mass_ratio_nb,
        up_down_return_mass_ratio_nb_parallel,"""

    content = content.replace("""        bars_since_flip_nb,
        binary_entropy_nb,
        binary_entropy_nb_parallel,""", new_imports)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_file()
