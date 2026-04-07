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
# Improvement Opportunities

1. **Centralize Limit Order Logic**
   - The logic in `check_limit_order_fill` should be fixed so that `fill_price` returns `limit_price` when `open` has not gapped past it, instead of returning the absolute `low_price` (for longs).
   - E.g., for a long limit order (buy):
     `fill_price = min(open_price, limit_price)`
     If `open_price < limit_price`, it gapped down and fills at `open_price`. If it didn't gap, but `low_price <= limit_price`, it fills at exactly `limit_price`.

2. **Align Numba and Python Simulators**
   - Incorporate `giveback_pct` and `early_invalidation` logics into `simulate_trade_exit_batch`. Numba supports these mathematically, they just need to be implemented. This ensures the optimizer is selecting policies that are actually optimal for the real execution engine.
   - Standardize output reasons. If Numba returns ints, map them to an enum shared with the Python engine.

3. **Handle Gap Risk on Stop Losses**
   - In `simulate_trade_hourly` (and Numba), if a SL is triggered, check if `open` gapped past the SL. If so, execute at `open` instead of `sl_price`. This prevents the simulator from overestimating PnL on highly volatile gap-downs/gap-ups.

4. **Clarify Target Labeling vs Execution**
   - Rename `TP` to `Trail Activation` in logging and semantic contexts to clarify that the engine does not rigidly exit at TP.
   - Ensure the meta-model's "Class 2 (TP)" is clearly understood as "Will reach Trail Activation without hitting SL", rather than "Will exit at TP".

# Recommended Tests

1. **Test Limit Order Fill Logic (`test_limit_order_fill_prices`)**
   - **Scenario**: Limit Buy at 100. Bar Open=105, High=110, Low=90, Close=95.
   - **Expected**: `did_fill=True`, `fill_price=100` (Not 90).
   - **Scenario 2 (Gap)**: Limit Buy at 100. Bar Open=95, High=98, Low=90, Close=92.
   - **Expected**: `did_fill=True`, `fill_price=95`.

2. **Test Stop Loss Gap Risk (`test_stop_loss_gap_execution`)**
   - **Scenario**: Long Entry at 100, SL at 95. Bar Open=90, High=92, Low=85, Close=90.
   - **Expected**: Exit at 90 (the open), not 95, due to gap.

3. **Test Tie-Breaker Ordering (`test_tie_breaker_logic`)**
   - **Scenario**: Long Entry at 100. Bar Open=100. High=110 (TP hit), Low=90 (SL hit). Both hit in same bar.
   - **Expected**: The logic should accurately measure distance from Open to SL (10) vs Open to TP (10), and resolve deterministically (e.g., SL wins).
   - **Scenario 2**: Open=105. High=110 (TP dist=5), Low=90 (SL dist=15).
   - **Expected**: TP wins because Open was closer to TP.
