# Confirmed Issues

## 1. Bug: `check_limit_order_fill` returns worse fill price than limit price
- **Severity**: High (Correctness Bug)
- **Evidence**: `extreme_price_movements/limit_order_pricer.py`, lines 283-289.
  ```python
  if is_long:
      fill_price = min(low_price, open_price)  # Use worse of open/low
      did_fill = low_price <= limit_price
  ```
  If a limit order is placed at `limit_price`, and price goes down to hit it, the fill price should be exactly `limit_price` (or better, if the open gaps below the limit price). Using `min(low_price, open_price)` means if `open_price > limit_price` and `low_price < limit_price`, it returns `low_price` (which is lower/better than `limit_price`, simulating positive slippage unrealistic for a limit order) or if it gaps below, it fills at the gap price. However, a limit order *to exit a long position* (sell order) fills at the limit price or higher. Wait, the code in `check_limit_order_fill` is used for *both* entry and exit.
  Let's re-read: For a long entry, limit price is below current price. We want price to drop to `limit_price` to fill. Once it drops to `limit_price`, we fill *at* `limit_price`. If it gaps down (open < limit), we fill at `open`. The current code returns `min(low, open)`. If `open > limit` and `low < limit`, it returns `low`, which means we fill at `low` (a better price than our limit). This is a bug. A limit order guarantees the limit price or better. But it shouldn't guarantee the absolute low of the bar. It should fill at the `limit_price` unless it gaps below.
  For a long exit (sell limit), the logic in `_apply_exit_limit_order` (in `engine.py`) explicitly sets `exit_price = limit_price`. But wait, in `engine.py`, `simulate_trade_hourly` lines 548-558 use `check_limit_order_fill` for exit limit offsets too. This is highly inconsistent.

## 2. Inconsistency: Semantic mismatch between Numba Simulator and Python Simulator
- **Severity**: Medium (Maintainability / Alignment)
- **Evidence**: `engine.py` supports "giveback_exit", "early_invalidation", and staged trailing stops (BE -> Profit Lock -> Trail). `ridge_position_sizer.py` (`simulate_trade_exit`) only supports a single `trailing_pct` triggered once `peak > entry_price`.
- **Consequence**: Policy optimizations evaluated by `simulate_trade_exit` (used in Ridge optimization) operate on simpler dynamics than the actual execution engine `simulate_trade_hourly`. Giveback exits and early invalidations are completely missing from the Numba simulator, meaning the optimizers are blind to them.

## 3. Ambiguity / Bug: Tie-breaking logic in Numba Simulator
- **Severity**: Low (Correctness)
- **Evidence**: `extreme_price_movements/ridge_position_sizer.py:408-419`.
  When resolving collisions using `open` proximity:
  ```python
  d = abs(o - sl_price)
  if d < best_dist or (d == best_dist and 0 < best_rank):
  ```
  If `tp_hit`, `sl_hit`, and `trailing_hit` all occur in the same bar (possible on wide-range bars), the code uses the distance from the open to determine what hit first. This assumes the path went straight from Open to the nearest barrier. While an acceptable proxy, it's deterministic but not causally rigorous. Furthermore, `best_rank` forces `SL > Trailing > TP` on ties.

## 4. Semantic Drift: Training label logic vs Execution logic
- **Severity**: Medium
- **Evidence**: `_tbm_proxy_target_class` in `training.py` determines class based on exact `time_to_mfe` and `time_to_mae`. If SL is hit before TP, it's class 0. In execution (`simulate_trade_hourly`), TP is not an exit, it's a trailing activator.
- **Consequence**: The meta-model classifies paths into TP/SL/TO, but the actual engine doesn't have a rigid "Take Profit" exit, only a trailing stop. Thus, a "TP" prediction from the meta-model maps to a trailing-stop activation in the simulator.

## 5. Inconsistency: Stop Loss execution price in Python Simulator
- **Severity**: Medium (Correctness)
- **Evidence**: `engine.py:586`
  ```python
  if hit_sl:
      exit_price = sl_price
  ```
  If the bar gaps past the SL (e.g., `open < sl_price` for long), a real stop-market order would fill at the open, not at the SL price. The simulator grants execution exactly at the SL price, ignoring gap risk on stops. This rewards strategies that hold through gaps because they don't pay the gap penalty.
