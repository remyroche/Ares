import sys

def patch_file():
    filepath = 'extreme_price_movements/fast_funcs.py'
    with open(filepath, 'r') as f:
        content = f.read()

    new_imports = """    from extreme_price_movements.features import (
        bars_since_flip_nb,
        binary_entropy_nb,
        binary_entropy_nb_parallel,
        ema_nb,"""

    content = content.replace("""    from extreme_price_movements.features import (
        bars_since_flip_nb,
        bars_since_flip_nb_parallel,
        binary_entropy_nb,
        binary_entropy_nb_parallel,
        consecutive_bars_nb,
        consecutive_bars_nb_parallel,
        up_down_semivol_ratio_nb,
        up_down_semivol_ratio_nb_parallel,
        up_down_return_mass_ratio_nb,
        up_down_return_mass_ratio_nb_parallel,
        ema_nb,""", new_imports)

    content = content.replace("""    elif func == bars_since_flip_nb:
        return apply_to_matrix_parallel(df, bars_since_flip_nb_parallel, *args)
    elif func == consecutive_bars_nb:
        return apply_to_matrix_parallel(df, consecutive_bars_nb_parallel, *args)
    elif func == up_down_semivol_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_semivol_ratio_nb_parallel, *args)
    elif func == up_down_return_mass_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_return_mass_ratio_nb_parallel, *args)""", """    elif func == bars_since_flip_nb:
        return apply_to_matrix_parallel(df, bars_since_flip_nb_parallel, *args)
    elif func == consecutive_bars_nb:
        return apply_to_matrix_parallel(df, consecutive_bars_nb_parallel, *args)
    elif func == up_down_semivol_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_semivol_ratio_nb_parallel, *args)
    elif func == up_down_return_mass_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_return_mass_ratio_nb_parallel, *args)""")

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_file()
