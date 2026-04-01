import numpy as np
import pandas as pd
from extreme_price_movements.fast_funcs import (
    apply_to_frame, consecutive_bars_nb, up_down_semivol_ratio_nb, up_down_return_mass_ratio_nb
)

def test_consecutive_bars():
    data = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.float32)
    df = pd.DataFrame({'a': data})
    out = apply_to_frame(df, consecutive_bars_nb)
    expected = np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(out['a'].values, expected)

def test_up_down_semivol():
    data = np.array([1, -1, 2, -2, 3, -3], dtype=np.float32)
    df = pd.DataFrame({'a': data})
    out = apply_to_frame(df, up_down_semivol_ratio_nb, 4)
    assert not np.isnan(out['a'].iloc[-1])
