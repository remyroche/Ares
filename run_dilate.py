import numpy as np
import pandas as pd
from extreme_price_movements.lgbm_based_mask_generation import RuleConsolidator

def test_dilate():
    data = pd.DataFrame(
        {"symbol": ["A", "B", "A", "A", "B", "C"]}
    )
    mask = np.array([True, False, False, False, False, False])
    consolidator = RuleConsolidator([], {})
    dilated = consolidator._dilate_mask_by_symbol(mask, data, bars=1)
    print("Dilated with bars=1:", dilated.tolist())
    dilated_2 = consolidator._dilate_mask_by_symbol(mask, data, bars=2)
    print("Dilated with bars=2:", dilated_2.tolist())

test_dilate()
