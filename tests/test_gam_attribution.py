import numpy as np
import pandas as pd

from scripts.run_tp6_sl4_rolling_gam_attribution_ff1 import _permute_within_month


def test_placebo_preserves_monthly_marginal_values_and_missingness():
    train = pd.DataFrame({"month": ["2025-01"] * 5 + ["2025-02"] * 5, "x": [1.0, 2.0, np.nan, 4.0, 5.0, 10.0, 11.0, 12.0, np.nan, 14.0]})
    held = pd.DataFrame({"month": ["2025-03"] * 4, "x": [20.0, np.nan, 22.0, 23.0]})
    tr, te = _permute_within_month(train, held, ["x"], seed=17)
    for original, shuffled in ((train, tr), (held, te)):
        for month, block in original.groupby("month"):
            got = shuffled.loc[shuffled.month.eq(month), "x"]
            expected = block.x
            assert got.isna().sum() == expected.isna().sum()
            assert sorted(got.dropna().tolist()) == sorted(expected.dropna().tolist())
