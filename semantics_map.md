# Simulator Semantics Map

## `simulate_trade_exit` (Numba - `ridge_position_sizer.py`)
This function simulates PnL based on arrays of H,L,O,C prices.
Returns integer codes:
- `0` = Take Profit (TP) triggered first.
- `1` = Stop Loss (SL) triggered first.
- `2` = Trailing Stop triggered first.
- `3` = Timeout (TO) or data exhaustion (NaN hit).

**Tie-Breaking within same bar:**
Uses open-price proximity to the respective barriers. If distance is equal, strictly orders outcomes by severity: `SL (1) > Trailing (2) > TP (0)`.

## `simulate_trade_hourly` (Python - `engine.py`)
Full simulator logic utilizing stages and more advanced risk management.
Returns string codes:
- `"stop_loss"`: Price hit the static SL price, or the break-even/profit-locked stop.
- `"trailing_stop"`: Trailing stop hit after entering Stage 2/3.
- `"time_exit"`: Survived up to max hold time without hitting other barriers.
- `"giveback_exit"`: Returned a significant portion of unrealized PnL (from peak MFE) back to the market.
- `"early_invalidation"`: Met adverse criteria over time before a profit lock/MFE.
- `"no_entry"` / `"no_path"` / `"limit_not_filled"`: Failures to enter the trade.

**Tie-Breaking:**
Evaluates sequentially based on logic flow. In `engine.py`, the engine processes each bar and applies the logic in the following order:
1. Break-even update.
2. Profit lock update.
3. Trail update.
4. Giveback exit check.
5. Early invalidation check.
6. Stop-loss / Trailing-stop hit check.
Within the SL check, `hit_sl = ll <= sl_price` (for long). If the bar gaps below SL, it simply takes the SL. If both TP and SL could logically trigger within the bar, `engine.py` lacks a formal `TP` hit check because TP is exclusively a "trailing activation" mechanism. It never exits *at* TP, it just activates trailing.

## `_tbm_proxy_target_class` (Meta Labeling - `training.py`)
Determines the outcome class strictly for TBM modeling.
Returns integer labels:
- `0` = SL (Hit SL before TP, or hit both and SL was earlier/tied).
- `1` = TO (Neither hit within horizon).
- `2` = TP (Hit TP before SL).

**Tie-Breaking:**
Relies on `time_to_mfe` and `time_to_mae`. If both hit, and `time_to_mfe < time_to_mae`, TP wins. If `time_to_mfe >= time_to_mae`, SL wins.
