# TP Base Percentage Explanation

`tp_base_pct` is the baseline Take Profit percentage (as a decimal) applied when the volatility (ATR) is at its median level.

- **Definition**: It represents the base target return for a trade under normal volatility conditions (ATR ratio = 1.0). The actual TP target is calculated as `k_tp * (ATR / MedianATR) * tp_base_pct`.
- **Range**: In the Optuna optimization (`compare_tbm_parameters.py`), `tp_base_pct` is searched in the range `[0.005, 0.04]`, which corresponds to **0.5% to 4.0%**.
- **Interpretation**: A value like `0.028` represents **2.8%**. While it may look like a small decimal, 2.8% is a significant move for a single trade.
