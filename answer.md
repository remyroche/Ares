# Default TP:SL Values in Train Mode

In 'train' mode, the default TP:SL values are defined by the `tp_mult` and `sl_mult` parameters in `extreme_price_movements/config.py`.

The default values are:
- **TP Multiplier (`tp_mult`):** `0.50`
- **SL Multiplier (`sl_mult`):** `0.18`

These values are used as multipliers for the dynamic Triple Barrier method (scaled by ATR).

If dynamic barriers (ATR) are unavailable, the fallback fixed values used in `extreme_price_movements/training.py` are:
- **Fixed TP:** `0.05` (5%)
- **Fixed SL:** `0.025` (2.5%)
