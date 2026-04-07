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
