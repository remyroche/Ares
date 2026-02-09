# Default TP:SL Values in Train Mode

In 'train' mode, the default TP:SL values are defined by the `tp_mult` and `sl_mult` parameters in `extreme_price_movements/config.py`.

The default values are:
- **TP Multiplier (`tp_mult`):** `0.50`
- **SL Multiplier (`sl_mult`):** `0.18`

## Typical Percentage Ranges

These values are multipliers applied to a dynamic **Triple Barrier** width (based on ATR and volatility). In our configuration (`extreme_price_movements/training.py` and `optimise_tpsl_ratio.py`), this barrier is typically clamped between **3% (`lo`)** and **6% (`hi`)**.

Therefore, in a high-volatility environment (barrier ≈ 4.5% - 6%):

-   **Take Profit (TP):** `0.50 * [3% - 6%]` ≈ **1.5% - 3.0%**
-   **Stop Loss (SL):** `0.18 * [3% - 6%]` ≈ **0.54% - 1.08%**

This translates to a Reward:Risk ratio of approximately **2.7:1**.
