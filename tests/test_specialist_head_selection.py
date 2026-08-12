import numpy as np
import pandas as pd

from extreme_price_movements.specialist_head_selection import select_complementary_heads


def test_complementary_selector_prefers_incremental_head():
    rng = np.random.default_rng(7)
    base = rng.normal(size=500)
    additive = rng.normal(size=500)
    target = ((base + additive) > 0).astype(int)
    x = pd.DataFrame({"base": base, "duplicate": base + rng.normal(scale=.01, size=500), "incremental": additive, "target": target})
    selected, audit = select_complementary_heads(x, ["duplicate", "incremental"], target_column="target", base_score_column="base")
    assert "incremental" in selected
    assert audit.loc[audit["head"].eq("incremental"), "selected"].item()
