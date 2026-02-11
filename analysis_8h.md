# Missing Features for 8h Horizon Analysis

Based on review of `extreme_price_movements/features.py` and `config.py`, the following features are missing to fully support the 8h trading horizon:

1.  **Base Returns & Volatility**
    *   `ret8h`: 8-hour return.
    *   `rv_8h`: 8-hour rolling volatility.

2.  **Trend & Mean Reversion Structure**
    *   `donch_dist_8`: Donchian distance for 8h.
    *   `pullback_8`: Pullback for 8h.
    *   `ft_8`: Failed Thrust for 8h.
    *   `failure_8`: Failure for 8h.
    *   `thrust_decay_8`: Thrust decay over 8h.
    *   `decel_8`: Deceleration for 8h.
    *   `ft_drop_8`: Failed thrust drop for 8h.

3.  **Exhaustion & Risk**
    *   `clv_mean_8`: Close Location Value mean over 8h.
    *   `evr_12`: Effort vs Reward over 12h (1.5x of 8h).
    *   `mfe_8h`: Max Favorable Excursion over 8h.
    *   `mae_8h`: Max Adverse Excursion over 8h.
