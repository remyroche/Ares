import sys

def patch_file():
    filepath = 'extreme_price_movements/fast_funcs.py'
    with open(filepath, 'r') as f:
        content = f.read()

    new_imports = """    # Note: New vectorized functions mapping logic
    from extreme_price_movements.features import (
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
        ema_nb,
        ema_nb_parallel,
        realized_vol_nb,
        rolling_std_nb,
        rolling_std_nb_parallel,
        slope_nb,
        slope_nb_parallel,
    )

    if func == bars_since_flip_nb:
        return apply_to_matrix_parallel(df, bars_since_flip_nb_parallel, *args)
    elif func == consecutive_bars_nb:
        return apply_to_matrix_parallel(df, consecutive_bars_nb_parallel, *args)
    elif func == up_down_semivol_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_semivol_ratio_nb_parallel, *args)
    elif func == up_down_return_mass_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_return_mass_ratio_nb_parallel, *args)
    elif func == realized_vol_nb:
        return apply_to_matrix_parallel(df, rolling_std_nb_parallel, *args)
    elif func == rolling_std_nb:
        return apply_to_matrix_parallel(df, rolling_std_nb_parallel, *args)
    elif func == _numba_rolling_zscore_nan_safe_1d:
        return apply_to_matrix_parallel(df, _numba_rolling_zscore_parallel, *args)
    elif func == slope_nb:
        return apply_to_matrix_parallel(df, slope_nb_parallel, *args)
    elif func == binary_entropy_nb:
        return apply_to_matrix_parallel(df, binary_entropy_nb_parallel, *args)
    elif func == ema_nb:
        return apply_to_matrix_parallel(df, ema_nb_parallel, *args)"""

    old_imports = """    # Note: New vectorized functions mapping logic
    from extreme_price_movements.features import (
        bars_since_flip_nb,
        binary_entropy_nb,
        binary_entropy_nb_parallel,
        ema_nb,
        ema_nb_parallel,
        realized_vol_nb,
        rolling_std_nb,
        rolling_std_nb_parallel,
        slope_nb,
        slope_nb_parallel,
    )

    elif func == bars_since_flip_nb:
        return apply_to_matrix_parallel(df, bars_since_flip_nb_parallel, *args)
    elif func == consecutive_bars_nb:
        return apply_to_matrix_parallel(df, consecutive_bars_nb_parallel, *args)
    elif func == up_down_semivol_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_semivol_ratio_nb_parallel, *args)
    elif func == up_down_return_mass_ratio_nb:
        return apply_to_matrix_parallel(df, up_down_return_mass_ratio_nb_parallel, *args)
    elif func == realized_vol_nb:
        return apply_to_matrix_parallel(df, rolling_std_nb_parallel, *args)
    elif func == rolling_std_nb:
        return apply_to_matrix_parallel(df, rolling_std_nb_parallel, *args)
    elif func == _numba_rolling_zscore_nan_safe_1d:
        return apply_to_matrix_parallel(df, _numba_rolling_zscore_parallel, *args)
    elif func == slope_nb:
        return apply_to_matrix_parallel(df, slope_nb_parallel, *args)
    elif func == binary_entropy_nb:
        return apply_to_matrix_parallel(df, binary_entropy_nb_parallel, *args)
    elif func == ema_nb:
        return apply_to_matrix_parallel(df, ema_nb_parallel, *args)"""

    content = content.replace(old_imports, new_imports)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_file()
