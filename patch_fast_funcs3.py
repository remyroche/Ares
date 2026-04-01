import sys

def patch_file():
    filepath = 'extreme_price_movements/fast_funcs.py'
    with open(filepath, 'r') as f:
        content = f.read()

    replacement = """    if func == bars_since_flip_nb:
        return apply_to_matrix_parallel(df, bars_since_flip_nb_parallel, *args)
    elif func == consecutive_bars_nb:
        return apply_to_matrix_parallel(df, consecutive_bars_nb_parallel, *args)
    elif func == up_down_semivol_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_semivol_ratio_nb_parallel, *args)
    elif func == up_down_return_mass_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_return_mass_ratio_nb_parallel, *args)
    elif func == realized_vol_nb:"""

    content = content.replace("""    if func == realized_vol_nb:""", replacement)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_file()
