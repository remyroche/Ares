import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# I see test_dilate_mask_by_symbol_is_symbol_safe failed because dilate mask returned wrong value.
# Wait, let's look at `_dilate_mask_by_symbol`
#     def _dilate_mask_by_symbol(
#         self, mask: np.ndarray, data: pd.DataFrame, bars: int = 1
#     ) -> np.ndarray:
#         if bars <= 0 or "symbol" not in data.columns:
#             return mask.copy()
#         out = mask.copy()
#         symbols = data["symbol"].to_numpy()
#         for i in np.where(mask)[0]:
#             sym = symbols[i]
#             for j in range(i + 1, min(i + bars + 1, len(mask))):
#                 if symbols[j] == sym:
#                     out[j] = True
#         return out
# Why did it return [True, False, False, False, False, False] instead of [True, False, True, False, False, False]?
# Maybe because of the test input.
