# Simulator Divergence Investigation Note

## Semantic Gap Summary
The fast Numba simulator (`simulate_trade_exit`) is highly optimized for grid-search throughput but simplifies the rich risk-management rules of the Python execution engine (`simulate_trade_hourly`).

**Missing in Numba:**
1. **Giveback Exits:** The engine exits if unrealized MFE drops by `giveback_pct` after passing the profit lock threshold. Numba lacks this.
2. **Early Invalidation:** The engine exits if the trade shows adverse drift over time without achieving positive MFE. Numba lacks this.
3. **Staged Trailing (BE -> Lock -> Trail):** The engine ratchets stops in stages. Numba activates full trailing immediately when `peak > entry_price`.

## Same-Bar Tie-Break Ambiguity
Both simulators face ambiguity when multiple barriers (e.g., TP and SL) are hit in the same hour.
- **Numba Logic:** Evaluates `abs(open - barrier_price)`. The barrier closest to the open is assumed to be hit first. If exactly equidistant, it enforces `SL > Trailing > TP` (worst outcome first).
- **Materiality:** Same-bar multi-hits typically occur on wide-range bars. The open-proximity proxy is a reasonable, deterministic heuristic without requiring 1-minute data reconstruction. It conservatively penalizes ambiguity.

## Conclusion and Recommendations
**Alignment Action:** We applied precise gap-execution semantics (`resolve_stop_fill`) to the Numba simulator so it correctly models gap slippage on stop-loss triggers. This removes the most optimistic execution bias from policy search.

**Recommendation on Further Convergence:**
Do not port `giveback_exit` or `early_invalidation` into the Numba simulator at this time.
- **Why?** The Numba simulator acts as a fast baseline policy optimizer (`Layer B`). The full Python engine (`Layer C`/Live) uses giveback and early invalidation as safety overlays. Overfitting grid search to those overlays might mask poorly performing base parameters.
- **Future Work:** If policy search produces strategies that heavily rely on givebacks to be profitable, consider an explicitly unified simulation core written entirely in Cython/Numba.

## TBM TP vs Execution Semantics
A note on TBM semantics: TBM labels `Class 2` as "TP" (hit favorable MFE barrier before adverse MAE barrier).
In the Python engine, reaching this favorable distance usually triggers "Trail Activation" rather than an immediate "Take Profit" exit. Thus, the TBM model is predicting whether the trade will reach the trailing activation threshold, *not* whether the trade will exit exactly at that profit margin.
